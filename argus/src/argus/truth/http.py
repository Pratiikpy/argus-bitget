"""One way to read a URL, for every module that reads one (audit finding 161).

Until 2026-09-27 each of forty call sites built its own ``Request``, chose its own headers and its
own ``User-Agent`` (forty-eight literals), caught its own subset of what ``urlopen`` can raise, and
turned it into its own message. The same outage read as "HTTP 503" in one answer, "transport
failure" in another and a bare ``URLError`` in a third, and none of them carried the typed kind
`truth/failures.py` already defines for the MCP client.

Every read here goes through ``urllib.request.urlopen``, looked up at call time, so
`truth/coverage.py`'s wrapper still counts it and a test that replaces ``urlopen`` still reaches
it. A failure of any kind — an HTTP status, a timeout, a refused connection, a body that is not the
JSON asked for — is raised as :class:`~argus.truth.failures.RpcError` with its
:class:`~argus.truth.failures.ErrorKind`, the status and a body excerpt when there was one, so a
caller wraps it in its own error type once and the reason survives.

**A read is tried twice when a second try can help (2026-09-28).** This module used to leave
retrying to the caller, and exactly one caller of about thirty (`agents/desk.py`, through
`agents/circuit.call_with_retry`) did, so a rate limit or a dropped connection on any market,
filing or macro read ended that answer (research/harvest/44-httpx.md, against ccxt's ``fetch2``
and Ritik200238/nightwatch's ``HttpClient.get``, which retry inside the one method every read
goes through). Now a GET — no body — that fails as a rate limit, a transport error or a 5xx is
tried once more: after the server's own ``Retry-After`` when it gave one of at most
:data:`MAX_RETRY_WAIT` seconds (a longer one is not waited on), else after
:data:`RETRY_DELAY`. A timeout is not retried: it has already spent the caller's whole budget, and a
second one would double it on the console's request path. A write, or any call with a body, is
never retried here: whether repeating it is safe is the caller's decision. ``retries=0`` opts out.

**Five readers keep their own transport, each for a reason this module cannot serve**
(:data:`OWN_TRANSPORT`, pinned by `tests/test_http.py`): the MCP client reads a server-sent
event stream incrementally against one wall-clock deadline (`market/rpc.py`); the Qwen client
streams tokens and retries by status (`llm/qwen.py`); the signed trading client routes on the
venue's error code inside a 4xx body (`execution/bitget_client.py`); the latency probe times the
raw round trip it exists to measure (`execution/latency_probe.py`); and Yahoo's consensus reader
keeps a cookie session through a crumb handshake (`market/estimates.py`, its own opener). The
evaluation harnesses under `eval/` are not product code and are not held to this.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any

from argus.truth.failures import ErrorKind, RpcError, classify_exception, classify_http

READ_RETRIES = 1
"""Extra tries a failed GET gets by default."""
RETRY_DELAY = 0.5
"""Seconds before a retry when the server named no wait."""
MAX_RETRY_WAIT = 5.0
"""The longest ``Retry-After`` honoured; a server asking for more is answered with the failure."""
RETRY_KINDS = frozenset({ErrorKind.RATE_LIMIT, ErrorKind.TRANSPORT, ErrorKind.UPSTREAM_5XX})
"""`truth/failures.RETRYABLE` less ``TIMEOUT``: see the module note."""

_sleep = time.sleep

OWN_TRANSPORT = frozenset({"market/rpc.py", "llm/qwen.py", "execution/bitget_client.py",
                           "execution/latency_probe.py", "market/estimates.py"})
"""Product modules allowed to call ``urlopen`` (or an opener) directly; see the module docstring."""

USER_AGENT = "ARGUS research console (+https://github.com/Pratiikpy/argus-bitget)"
"""What every request says it is, unless the caller must say something else (SEC EDGAR asks for a
contact address, `market/evidence.py`)."""


class HttpError(RpcError):
    """An HTTP error status, typed like any :class:`RpcError`, with the body the server sent: a
    venue's own error code lives there (Bitget answers a bad symbol with a 400 whose JSON says
    why), and throwing it away leaves only the status."""

    def __init__(self, typed: RpcError, body: bytes) -> None:
        super().__init__(typed.kind, str(typed), http_status=typed.http_status, code=typed.code,
                         data=typed.data, retry_after=typed.retry_after)
        self.body = body


def status_of(exc: BaseException) -> int | None:
    """The HTTP status behind a failed read, from this module's error or from ``urllib``'s own
    (a test double, or a source injected by a caller, may still raise that)."""
    if isinstance(exc, RpcError):
        return exc.http_status
    return int(exc.code) if isinstance(exc, urllib.error.HTTPError) else None


def reason_of(exc: BaseException) -> str:
    """A short reason for a status line: the typed kind, else the exception's class name."""
    return exc.kind.value if isinstance(exc, RpcError) else type(exc).__name__


def url_with(url: str, params: Mapping[str, Any] | None) -> str:
    """``url`` with ``params`` appended as a query string (none added when there are none)."""
    if not params:
        return url
    return f"{url}{'&' if '?' in url else '?'}{urllib.parse.urlencode(params)}"


def fetch(url: str, *, params: Mapping[str, Any] | None = None,
          headers: Mapping[str, str] | None = None, timeout: float,
          data: bytes | None = None, method: str | None = None,
          max_bytes: int | None = None, retries: int | None = None) -> bytes:
    """The body of ``url`` (at most ``max_bytes`` of it, when given). Raises :class:`RpcError`
    for every way the read can fail, after the retry the module note describes; ``retries`` sets
    how many extra tries a GET gets (a call with a body always gets none)."""
    is_read = data is None and (method or "GET").upper() == "GET"
    extra = (READ_RETRIES if retries is None else max(0, retries)) if is_read else 0
    for attempt in range(extra + 1):
        try:
            return _fetch_once(url, params=params, headers=headers, timeout=timeout, data=data,
                               method=method, max_bytes=max_bytes)
        except RpcError as exc:
            wait = RETRY_DELAY
            if exc.retry_after is not None:
                try:
                    wait = float(exc.retry_after)
                except (TypeError, ValueError):
                    wait = RETRY_DELAY
            if attempt == extra or exc.kind not in RETRY_KINDS or wait > MAX_RETRY_WAIT:
                exc.attempts = attempt + 1
                raise
            _sleep(max(0.0, wait))
    raise AssertionError("unreachable")  # pragma: no cover - the loop returns or raises


def _fetch_once(url: str, *, params: Mapping[str, Any] | None,
                headers: Mapping[str, str] | None, timeout: float, data: bytes | None,
                method: str | None, max_bytes: int | None) -> bytes:
    merged = {"User-Agent": USER_AGENT, **(headers or {})}
    request = urllib.request.Request(url_with(url, params), headers=merged, data=data,
                                     method=method)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
        try:
            body: bytes = response.read() if max_bytes is None else response.read(max_bytes)
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read() or b""
        except (OSError, ValueError):
            raw = b""
        after = exc.headers.get("Retry-After") if exc.headers is not None else None
        raise HttpError(classify_http(int(exc.code), raw, retry_after=after), raw) from exc
    except Exception as exc:  # every failure is typed, not only the ones a caller thought of
        raise classify_exception(exc) from exc
    return body


def fetch_text(url: str, *, encoding: str = "utf-8", errors: str = "replace",
               **kwargs: Any) -> str:
    """The body of ``url`` as text."""
    return fetch(url, **kwargs).decode(encoding, errors)


def fetch_json(url: str, **kwargs: Any) -> Any:
    """The body of ``url`` parsed as JSON; a body that is not JSON is an ``UNPARSEABLE``
    :class:`RpcError`, not a ``JSONDecodeError`` the caller forgot to catch."""
    headers = {"Accept": "application/json", **(kwargs.pop("headers", None) or {})}
    body = fetch(url, headers=headers, **kwargs)
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise classify_exception(exc) from exc


__all__ = ["USER_AGENT", "HttpError", "RpcError", "fetch", "fetch_json", "fetch_text", "reason_of",
           "status_of", "url_with"]
