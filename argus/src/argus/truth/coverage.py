"""Which sources an answer reached, and which did not answer — counted at the transport.

Three Season 2 desks show this and ARGUS did not: optic-bitget prints a coverage ratio (rows ok of
rows tried) and caps its confidence by it; baserate labels every figure observed, computed or
unavailable; MirrorLine separates what was checked from what could not be. ARGUS listed the sources
an answer used, which says nothing about the ones it tried and did not get — and every engine here
degrades quietly when a leg fails (a missing headline feed, a dark Skill, a timed-out filing), so a
reader could not tell a complete answer from one built on half its inputs.

Counting has to happen where the calls happen, not in each engine, or the next engine written
forgets it. Every network read in ARGUS goes through ``urllib.request.urlopen`` (31 call sites, no
other HTTP client — checked 2026-09-25), so :func:`install` wraps that one function. While a
:func:`recording` is open on the current context, each call is noted with a readable source name
(:func:`source_name`) and whether it answered. Worker threads inherit the recording only if they
are started with the caller's context, which is what :class:`ContextPool` does; a plain
``ThreadPoolExecutor`` would silently drop their calls and the count would look better than it is.

What is not counted: data read from files (the frozen history, a dated snapshot), and calls made by
subprocesses (the X and Reddit readers run on a schedule, not inside an answer). The line says
"sources reached", not "inputs used", for that reason.
"""

from __future__ import annotations

import contextvars
import io
import json
import threading
import urllib.request
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse


@dataclass
class Record:
    """Calls noted during one answer, by source name: True once any call to it answered."""

    answered: dict[str, bool] = field(default_factory=dict)
    why: dict[str, str] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def note(self, source: str, ok: bool, why: str = "") -> None:
        with self.lock:
            self.answered[source] = self.answered.get(source, False) or ok
            if not ok and why and source not in self.why:
                self.why[source] = why

    @property
    def missing(self) -> list[str]:
        return sorted(s for s, ok in self.answered.items() if not ok)

    def as_dict(self) -> dict[str, Any]:
        return {"reached": sorted(s for s, ok in self.answered.items() if ok),
                "did_not_answer": {s: self.why.get(s, "") for s in self.missing}}

    def line(self) -> str | None:
        """The closing line of an answer, or None when no network source was tried."""
        total = len(self.answered)
        if not total:
            return None
        missing = self.missing
        if not missing:
            return f"Sources reached: {total} of {total} answered."
        named = "; ".join(f"{s} ({self.why[s]})" if self.why.get(s) else s for s in missing)
        return (f"Sources reached: {total - len(missing)} of {total} answered. Did not answer: "
                f"{named} — everything above is built without "
                f"{'it' if len(missing) == 1 else 'them'}.")


_current: contextvars.ContextVar[Record | None] = contextvars.ContextVar("coverage", default=None)


@contextmanager
def recording() -> Iterator[Record]:
    record = Record()
    token = _current.set(record)
    try:
        yield record
    finally:
        _current.reset(token)


class ContextPool(ThreadPoolExecutor):
    """A thread pool whose tasks run in the submitter's context, so their calls are counted."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        return super().submit(contextvars.copy_context().run, fn, *args, **kwargs)


_BITGET_PATHS = (
    ("history-fund-rate", "Bitget funding history"), ("current-fund-rate", "Bitget funding"),
    ("candles", "Bitget candles"), ("tickers", "Bitget tickers"), ("ticker", "Bitget tickers"),
    ("orderbook", "Bitget order book"), ("merge-depth", "Bitget order book"),
    ("contracts", "Bitget contract list"), ("open-interest", "Bitget open interest"),
    ("account-long-short", "Bitget positioning"), ("long-short", "Bitget positioning"),
    ("instruments", "Bitget contract list"),
)
_HOSTS = (
    ("gamma-api.polymarket.com", "Polymarket"), ("data.sec.gov", "SEC XBRL"),
    ("sec.gov", "SEC EDGAR"), ("stlouisfed.org", "FRED"), ("alternative.me", "fear & greed"),
    ("finance.yahoo.com", "Yahoo Finance headlines"), ("r.jina.ai", "Jina reader"),
    ("sosovalue", "SoSoValue"), ("hackathon.bitgetops.com", "Qwen"),
    ("dowjones.io", "news feed MarketWatch"),
)


def source_name(url: str, body: bytes | None = None) -> str:
    """A reader's name for the source behind a URL. Bitget's MCP servers are named by the tool
    called, which is in the JSON-RPC body; feeds by their host."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    # bitget-mcp-server lives at agent.bitget.com/mcp; bitget-signal's documented server is
    # datahub.noxiaohao.com/mcp (`market/evidence.py`).
    mcp = ("bitget-mcp-server" if host == "agent.bitget.com" else
           "bitget-signal" if "noxiaohao" in host else None)
    if mcp is not None:
        tool: str | None = None
        try:
            call = json.loads((body or b"{}").decode("utf-8"))
            params = (call.get("params") or {}) if call.get("method") == "tools/call" else {}
            tool = params.get("name")
            if tool == "do_query":  # bitget-mcp-server routes every entry through one tool
                tool = (params.get("arguments") or {}).get("entry_id") or tool
        except (ValueError, UnicodeDecodeError, AttributeError):
            tool = None
        # The handshake (initialize, notifications) is plumbing, not a source.
        return f"{mcp} {tool}" if tool else ""
    if host.endswith("bitget.com"):
        for fragment, name in _BITGET_PATHS:
            if fragment in parsed.path:
                return name
        return "Bitget market data"
    for fragment, name in _HOSTS:
        if fragment in host:
            return name
    bare = host.removeprefix("www.").removeprefix("feeds.")
    feed = parsed.path.endswith((".xml", "/rss", "/feed", ".html")) or "rss" in parsed.path
    return f"news feed {bare}" if feed else (bare or "unknown source")


_original: Callable[..., Any] | None = None


def _observed(url: Any, *args: Any, **kwargs: Any) -> Any:
    assert _original is not None
    record = _current.get()
    if record is None:
        return _original(url, *args, **kwargs)
    if isinstance(url, urllib.request.Request):
        target, body = url.full_url, url.data if isinstance(url.data, bytes) else None
    else:
        target, body = str(url), args[0] if args and isinstance(args[0], bytes) else \
            kwargs.get("data") if isinstance(kwargs.get("data"), bytes) else None
    name = source_name(target, body)
    if name in ("", "Qwen"):
        return _original(url, *args, **kwargs)  # plumbing or the planner, not a data source
    try:
        response = _original(url, *args, **kwargs)
    except Exception as exc:
        code = getattr(exc, "code", None)
        record.note(name, False, f"HTTP {code}" if code else type(exc).__name__)
        raise
    if name.startswith(("bitget-mcp-server ", "bitget-signal ")):
        # An MCP tool that fails still answers HTTP 200 and says so inside the reply
        # (`"isError": true`; `agent-mcp/src/server.ts:119-124`). Read it, note it, and hand the
        # caller an identical copy of the body.
        response = _Replayed(response)
        # bitget-mcp-server's own upstream failure arrives as a tool result that reads
        # {"success": false, "status_code": 503}, JSON escaped inside the text content; it was
        # counted as answered while the service was down (answer audit, round 3).
        flat = response.body.replace(b" ", b"").replace(b"\\", b"")
        failed = b'"isError":true' in flat or b'"success":false' in flat
        record.note(name, not failed, "tool reported an error" if failed else "")
        return response
    record.note(name, True)
    return response


class _Replayed(io.BytesIO):
    """A consumed HTTP response, readable again, with the attributes callers here use."""

    def __init__(self, response: Any) -> None:
        with response:
            self.body = response.read()
        super().__init__(self.body)
        self.headers = response.headers
        self.status = getattr(response, "status", 200)
        self.url = getattr(response, "url", "")

    def getcode(self) -> int:
        return int(self.status)

    def info(self) -> Any:
        return self.headers

    def __enter__(self) -> _Replayed:
        return self


def install() -> None:
    """Wrap ``urllib.request.urlopen`` once. Outside a recording the wrapper only passes through."""
    global _original
    if _original is None:
        _original = urllib.request.urlopen
        urllib.request.urlopen = _observed
