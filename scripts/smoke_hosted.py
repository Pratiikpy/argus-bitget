"""Smoke test of the deployed console: a deploy is done only when this passes.

`argus/tests/test_vercel_entry.py` proves the bundle works when started here. This proves the
URL judges open works now: the build Vercel made, the environment it runs in, and a record that
is fresh by the desk's own schedule. Each check is a real request as a visitor would send it.

    python scripts/smoke_hosted.py [--url https://...] [--skip-research]

Exit code 0 when every check passes, 1 otherwise; one line per check either way.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_URL = "https://deploy-topaz-seven-64.vercel.app"


def _call(url: str, data: bytes | None = None, headers: dict[str, str] | None = None,
          timeout: float = 60) -> tuple[int, str, bytes]:
    request = Request(url, data=data, method="POST" if data else "GET",
                      headers={"User-Agent": "argus-smoke/1", **(headers or {})})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read()


def checks(base: str, research: bool) -> list[tuple[str, Callable[[], str]]]:
    form = {"Content-Type": "application/x-www-form-urlencoded"}

    def console() -> str:
        status, kind, body = _call(base + "/")
        assert status == 200 and kind.startswith("text/html"), (status, kind)
        assert body.decode("utf-8").rstrip().endswith("</html>"), "page is cut short"
        return f"{len(body):,} bytes"

    def ask_form() -> str:
        status, _, body = _call(base + "/ask", urlencode(
            {"q": "how many decisions are on record"}).encode(), form)
        payload = json.loads(body)
        assert status == 200 and payload["refused"] is False and payload["sources"], payload
        return payload["lines"][0][:70]

    def ask_json() -> str:
        status, _, body = _call(base + "/ask", json.dumps({"q": "sell half of that"}).encode(),
                                {"Content-Type": "application/json"})
        payload = json.loads(body)
        assert status == 200 and payload.get("intent") == "order" and payload["refused"], payload
        return "an order is refused"

    def status_fresh() -> str:
        status, _, body = _call(base + "/status")
        payload = json.loads(body)
        assert status == 200 and payload["chain_intact"] is True, payload
        assert payload["stale"] is False, (
            f"record is stale: newest decision {payload['newest_decision_at']}, "
            f"last scheduled cycle {payload.get('last_expected_cycle_at')}")
        return f"{payload['entries']} entries, newest {payload['age_hours']} h old"

    def preview_image() -> str:
        status, kind, body = _call(base + "/og.png")
        assert status == 200 and kind == "image/png" and body[:4] == b"\x89PNG", (status, kind)
        return f"{len(body):,} bytes"

    def not_found() -> str:
        status, kind, _ = _call(base + "/no-such-page", headers={"Accept": "text/html"})
        assert status == 404 and kind.startswith("text/html"), (status, kind)
        return "404 page"

    def research_task() -> str:
        status, _, body = _call(base + "/research", urlencode(
            {"q": "Should I add 10% more NVDA to my book?"}).encode(), form, timeout=180)
        page = body.decode("utf-8")
        assert status == 200 and "Read as:" in page and "Conclusion" in page, status
        return "question read back and concluded"

    out: list[tuple[str, Callable[[], str]]] = [
        ("GET /", console), ("POST /ask (form)", ask_form), ("POST /ask (JSON)", ask_json),
        ("GET /status", status_fresh), ("GET /og.png", preview_image),
        ("GET /no-such-page", not_found)]
    if research:
        out.append(("POST /research", research_task))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--skip-research", action="store_true",
                        help="leave out the research task, which reads live venue data")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    failed = 0
    for name, check in checks(base, not args.skip_research):
        started = time.monotonic()
        try:
            note = check()
            verdict = "PASS"
        except (AssertionError, KeyError, ValueError, URLError, TimeoutError) as exc:
            note, verdict = f"{type(exc).__name__}: {exc}"[:300], "FAIL"
            failed += 1
        print(f"{verdict}  {name:<20} {time.monotonic() - started:5.1f}s  {note}")
    print(f"{'all checks passed' if not failed else f'{failed} check(s) failed'} at {base}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
