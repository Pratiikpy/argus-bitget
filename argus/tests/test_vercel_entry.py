"""The hosted entry point, driven the way Vercel drives it (`deploy/api/index.py`).

Every other server test starts `argus.lui.server.Handler` from the source tree. Judges use the
hosted URL, which runs a *copy* of the package with a *copy* of the data under `deploy/`, through
an adapter that sets its own data directory; none of that was ever exercised, so a POST that
worked locally could 405 in production and the suite could not see it (audit, 2026-09-26). These
tests start the bundle's own `handler` in a separate interpreter, so the code under test is the
code that ships, not whichever `argus` this process imported first.

The bundle is built by `python -m argus.demo.deploysync` on the maintainer's machine and never
committed, so on a clean checkout these tests skip with that reason, like the drift guards in
`test_deploysync.py`. The one route that computes from live venue data is marked ``network``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from argus.demo.deploysync import DEPLOY

ENTRY = DEPLOY / "api" / "index.py"

# Run in a fresh interpreter: the entry file puts the bundle first on sys.path, so `argus` there
# is the bundle's copy. Printing its path lets the test prove which copy answered.
_SERVE = """
import runpy, sys
from http.server import ThreadingHTTPServer
entry = runpy.run_path(sys.argv[1])
import argus
server = ThreadingHTTPServer(("127.0.0.1", 0), entry["handler"])
print(server.server_port, argus.__file__, flush=True)
server.serve_forever()
"""


@pytest.fixture(scope="module")
def hosted() -> Iterator[str]:
    if not ENTRY.is_file():
        pytest.skip("no deploy/ bundle in this checkout; run `python -m argus.demo.deploysync`")
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "QWEN_API_KEY", "BITGET_QWEN_API_KEY")}
    env["PYTHONUTF8"] = "1"
    proc = subprocess.Popen([sys.executable, "-c", _SERVE, str(ENTRY)], env=env,
                            cwd=str(DEPLOY), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    try:
        assert proc.stdout is not None
        first = proc.stdout.readline().split(maxsplit=1)
        if len(first) != 2:
            proc.kill()
            _, err = proc.communicate(timeout=10)
            raise AssertionError(f"the bundle's handler did not start: {err[-800:]}")
        port, package = first
        assert Path(package.strip()).resolve().is_relative_to(DEPLOY.resolve()), (
            f"the answering package is {package.strip()}, not the bundle's copy")
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.kill()
        proc.wait(timeout=10)


def _call(url: str, data: bytes | None = None, headers: dict[str, str] | None = None,
          timeout: float = 60) -> tuple[int, str, bytes]:
    request = Request(url, data=data, headers=headers or {}, method="POST" if data else "GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read()


class TestTheHostedPagesServe:
    def test_the_console_is_served_whole(self, hosted: str) -> None:
        status, kind, body = _call(hosted + "/")
        assert status == 200 and kind.startswith("text/html")
        assert body.decode("utf-8").rstrip().endswith("</html>")

    def test_the_link_preview_image_is_served(self, hosted: str) -> None:
        status, kind, body = _call(hosted + "/og.png")
        assert status == 200 and kind == "image/png"
        assert body[:8] == b"\x89PNG\r\n\x1a\n"

    def test_an_unknown_page_is_a_page_with_a_way_back(self, hosted: str) -> None:
        status, kind, body = _call(hosted + "/no-such-page", headers={"Accept": "text/html"})
        assert status == 404 and kind.startswith("text/html")
        assert 'href="/"' in body.decode("utf-8")


class TestTheHostedAskAnswers:
    """A question the ledger answers with no model and no network, sent the three ways a client
    can send it. The hosted POST /ask once returned 405 (backlog, 2026-09-25)."""

    QUESTION = "how many decisions are on record"

    def _check(self, payload: dict[str, Any]) -> None:
        assert payload["refused"] is False, payload
        assert payload["lines"] and payload["sources"], payload

    def test_by_get(self, hosted: str) -> None:
        status, _, body = _call(f"{hosted}/ask?{urlencode({'q': self.QUESTION})}")
        assert status == 200
        self._check(json.loads(body))

    def test_by_form_post(self, hosted: str) -> None:
        status, _, body = _call(hosted + "/ask", urlencode({"q": self.QUESTION}).encode(),
                                {"Content-Type": "application/x-www-form-urlencoded"})
        assert status == 200
        self._check(json.loads(body))

    def test_by_json_post(self, hosted: str) -> None:
        status, _, body = _call(hosted + "/ask", json.dumps({"q": self.QUESTION}).encode(),
                                {"Content-Type": "application/json"})
        assert status == 200
        self._check(json.loads(body))

    def test_an_order_is_refused_on_the_hosted_page_too(self, hosted: str) -> None:
        status, _, body = _call(hosted + "/ask", urlencode({"q": "sell half of that"}).encode(),
                                {"Content-Type": "application/x-www-form-urlencoded"})
        payload = json.loads(body)
        assert status == 200 and payload["refused"] is True and payload["intent"] == "order"


class TestTheHostedStatusReadsTheBundledRecord:
    def test_status_names_the_bundled_ledgers_newest_decision(self, hosted: str) -> None:
        """/status must describe the record that shipped. Whether that record is fresh is a
        property of the deploy, checked against the live URL by `scripts/smoke_hosted.py`."""
        newest = None
        with (DEPLOY / "data" / "paper_ledger.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("kind", "decision") == "decision" and row.get("decided_at"):
                    newest = row["decided_at"]
        assert newest is not None, "the bundled ledger holds no decision"
        status, _, body = _call(hosted + "/status")
        payload = json.loads(body)
        assert status == 200 and payload["chain_intact"] is True
        assert (datetime.fromisoformat(payload["newest_decision_at"])
                == datetime.fromisoformat(newest.replace("Z", "+00:00")))


@pytest.mark.network
class TestTheHostedResearchTask:
    def test_a_typed_question_is_read_back_and_concluded(self, hosted: str) -> None:
        status, kind, body = _call(
            hosted + "/research",
            urlencode({"q": "Should I add 10% more NVDA to my book?"}).encode(),
            {"Content-Type": "application/x-www-form-urlencoded"}, timeout=180)
        page = body.decode("utf-8")
        assert status == 200 and kind.startswith("text/html")
        assert "Read as:" in page and "Conclusion" in page
