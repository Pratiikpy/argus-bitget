"""The shared reader (`truth/http.py`): typed failures, bodies kept, and one door to the network."""

from __future__ import annotations

import io
import re
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from argus.truth import http
from argus.truth.failures import ErrorKind

SRC = Path(__file__).resolve().parents[1] / "src" / "argus"


class _Response(io.BytesIO):
    pass


def _serve(monkeypatch: pytest.MonkeyPatch, body: bytes = b"", *,
           error: BaseException | None = None, seen: list[Any] | None = None) -> None:
    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        if seen is not None:
            seen.append((request, timeout))
        if error is not None:
            raise error
        return _Response(body)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)


def test_json_is_read_with_our_user_agent_and_the_params_encoded(
        monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []
    _serve(monkeypatch, b'{"ok": 1}', seen=seen)
    assert http.fetch_json("https://x.test/a", params={"q": "n v"}, timeout=3) == {"ok": 1}
    request, timeout = seen[0]
    assert request.full_url == "https://x.test/a?q=n+v" and timeout == 3
    assert request.get_header("User-agent") == http.USER_AGENT
    assert request.get_header("Accept") == "application/json"


def test_a_caller_may_name_itself_and_post(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []
    _serve(monkeypatch, b"proof", seen=seen)
    got = http.fetch("https://cal.test/digest", data=b"\x01", method="POST", timeout=5,
                     headers={"User-Agent": "SEC contact a@b.test"})
    assert got == b"proof"
    request, _ = seen[0]
    assert request.get_method() == "POST"
    assert request.get_header("User-agent") == "SEC contact a@b.test"


def test_an_http_error_is_typed_and_keeps_its_body(monkeypatch: pytest.MonkeyPatch) -> None:
    headers = Message()
    body = b'{"code": "40034", "msg": "Parameter symbol does not exist"}'
    _serve(monkeypatch, error=urllib.error.HTTPError("https://x.test", 400, "Bad", headers,
                                                     io.BytesIO(body)))
    with pytest.raises(http.HttpError) as caught:
        http.fetch_json("https://x.test", timeout=1)
    assert caught.value.http_status == 400 and caught.value.body == body
    assert http.status_of(caught.value) == 400


def test_a_rate_limit_and_a_404_are_named(monkeypatch: pytest.MonkeyPatch) -> None:
    headers = Message()
    headers["Retry-After"] = "7"
    _serve(monkeypatch, error=urllib.error.HTTPError("u", 429, "slow", headers, io.BytesIO(b"")))
    with pytest.raises(http.RpcError) as caught:
        http.fetch("https://x.test", timeout=1)
    assert caught.value.kind is ErrorKind.RATE_LIMIT and caught.value.retry_after == 7.0
    assert caught.value.retryable
    _serve(monkeypatch, error=urllib.error.HTTPError("u", 404, "gone", Message(), io.BytesIO(b"")))
    with pytest.raises(http.RpcError) as missing:
        http.fetch("https://x.test", timeout=1)
    assert missing.value.kind is ErrorKind.NOT_FOUND and not missing.value.retryable


def test_a_timeout_a_refusal_and_a_non_json_body_are_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, error=urllib.error.URLError(TimeoutError("timed out")))
    with pytest.raises(http.RpcError) as slow:
        http.fetch("https://x.test", timeout=1)
    assert slow.value.kind is ErrorKind.TIMEOUT and http.reason_of(slow.value) == "timeout"
    _serve(monkeypatch, error=ConnectionRefusedError("refused"))
    with pytest.raises(http.RpcError) as refused:
        http.fetch("https://x.test", timeout=1)
    assert refused.value.kind is ErrorKind.TRANSPORT
    _serve(monkeypatch, b"<html>blocked</html>")
    with pytest.raises(http.RpcError) as html:
        http.fetch_json("https://x.test", timeout=1)
    assert html.value.kind is ErrorKind.UNPARSEABLE


def test_max_bytes_caps_the_read(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, b"abcdef")
    assert http.fetch("https://x.test", timeout=1, max_bytes=3) == b"abc"


def test_status_and_reason_read_urllib_errors_too() -> None:
    raw = urllib.error.HTTPError("u", 503, "down", Message(), io.BytesIO(b""))
    assert http.status_of(raw) == 503 and http.reason_of(raw) == "HTTPError"
    assert http.status_of(ValueError()) is None


def test_product_code_reads_the_network_only_through_this_module() -> None:
    """Forty call sites each had their own headers and their own idea of a failure (audit 161).
    A new ``urlopen`` in product code outside the declared exceptions fails here."""
    direct = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith(("eval/", "vendor/")) or rel in http.OWN_TRANSPORT:
            continue
        if rel in ("truth/http.py", "truth/coverage.py", "lui/trace.py"):
            continue  # the reader itself, and the two wrappers that count and trace it
        if re.search(r"\burlopen\(", path.read_text(encoding="utf-8")):
            direct.append(rel)
    assert direct == []


def test_the_bitget_host_is_named_once() -> None:
    """Audit 165: the host was written out in twelve modules; `truth/endpoints.py` names it."""
    literal = [path.relative_to(SRC).as_posix() for path in sorted(SRC.rglob("*.py"))
               if not path.relative_to(SRC).as_posix().startswith(("eval/", "vendor/"))
               and path.name != "endpoints.py"
               and '"https://api.bitget.com' in path.read_text(encoding="utf-8")]
    assert literal == []
