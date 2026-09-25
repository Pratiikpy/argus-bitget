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

**Why a source did not answer is typed, not guessed (2026-09-25).** A failure is named with the
shared taxonomy in :mod:`argus.market.rpc` — an MCP reply is read as JSON-RPC rather than scanned
for ``"success":false`` in its bytes, so a tool's own refusal (SEP-1303 ``isError``), an upstream
503 carried inside a 200, and a protocol error are told apart, and a network failure says "timed
out" or "unreachable" instead of an exception class name. :attr:`Record.kinds` keeps the machine
kind beside the readable reason; ``as_dict`` publishes both.

What is not counted: data read from files (the frozen history, a dated snapshot), and calls made by
subprocesses (the X and Reddit readers run on a schedule, not inside an answer). The line says
"sources reached", not "inputs used", for that reason.
"""

from __future__ import annotations

import contextvars
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from argus.market.rpc import LABELS, classify_exception, classify_http, reply_failure


@dataclass
class Record:
    """Calls noted during one answer, by source name: True once any call to it answered."""

    answered: dict[str, bool] = field(default_factory=dict)
    why: dict[str, str] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)
    """The :class:`~argus.market.rpc.ErrorKind` of each source's first failure."""
    lock: threading.Lock = field(default_factory=threading.Lock)

    def note(self, source: str, ok: bool, why: str = "", kind: str = "") -> None:
        with self.lock:
            self.answered[source] = self.answered.get(source, False) or ok
            if not ok and why and source not in self.why:
                self.why[source] = why
            if not ok and kind and source not in self.kinds:
                self.kinds[source] = kind

    @property
    def missing(self) -> list[str]:
        return sorted(s for s, ok in self.answered.items() if not ok)

    def as_dict(self) -> dict[str, Any]:
        return {"reached": sorted(s for s, ok in self.answered.items() if ok),
                "did_not_answer": {s: self.why.get(s, "") for s in self.missing},
                "failure_kinds": {s: self.kinds.get(s, "") for s in self.missing}}

    def line(self) -> str | None:
        """The closing line of an answer, or None when no network source was tried."""
        total = len(self.answered)
        if not total:
            return None
        missing = self.missing
        if not missing:
            return f"Sources reached: {total} of {total} answered."
        # Nine tools of one server failing for one reason is one fact, not nine: "bitget-mcp-server
        # equity_calendar (upstream 5xx); bitget-mcp-server equity_estimates_consensus (upstream
        # 5xx); ..." filled a line on every fundamentals answer during the 2026-09-25 outage.
        groups: dict[str, list[str]] = {}
        for s in missing:
            server, _, tool = s.partition(" ")
            key = server if tool and server in ("bitget-mcp-server", "bitget-signal") else s
            groups.setdefault(key, []).append(s)
        parts = []
        for key, members in groups.items():
            reasons = sorted({self.why.get(m, "") for m in members} - {""})
            if len(members) > 1:
                parts.append(f"{key} ({len(members)} tools" + (f": {'; '.join(reasons)}"
                                                               if reasons else "") + ")")
            else:
                parts.append(f"{key} ({self.why[key]})" if self.why.get(key) else key)
        named = "; ".join(parts)
        return (f"Sources reached: {total - len(missing)} of {total} answered. Did not answer: "
                f"{named} — everything above is built without "
                f"{'it' if len(parts) == 1 else 'them'}.")


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
        if isinstance(exc, urllib.error.HTTPError):
            # Typed from the status alone: reading the body here would consume it, and the caller
            # this exception is re-raised to may need it (an MCP 400 carries its JSON-RPC error).
            # "HTTP 503" stays the reader's wording for a status.
            typed = classify_http(int(exc.code), b"")
            why = f"HTTP {exc.code}"
        else:
            typed = classify_exception(exc)
            why = typed.label
        record.note(name, False, why, typed.kind.value)
        raise
    if name.startswith(("bitget-mcp-server ", "bitget-signal ")):
        # An MCP tool that fails still answers HTTP 200 and says so inside the reply
        # (`"isError": true`; `agent-mcp/src/server.ts:119-124`). Read it, note it, and hand the
        # caller an identical copy of the body.
        # bitget-mcp-server's own upstream failure arrives as a tool result that reads
        # {"success": false, "status_code": 503}, JSON escaped inside the text content; it was
        # counted as answered while the service was down (answer audit, round 3). The reply is
        # parsed as JSON-RPC (`market/rpc.py:reply_failure`) rather than scanned as bytes, so that
        # 503 is named an upstream 5xx and a refused argument is named the tool's own error.
        #
        # The body is teed, not read here in full: bitget-signal holds a slow tool's stream open
        # with a ping every 15 s, so reading to EOF inside this wrapper would hang the answer
        # past any deadline the caller set (`market/rpc.py:_read_until`, 2026-09-25).
        return _Teed(response, record, name)
    record.note(name, True)
    return response


class _Teed:
    """An MCP response passed through to the caller, with every byte read kept, and the reply
    judged when the caller closes it — so the caller's own deadline governs the read."""

    def __init__(self, response: Any, record: Record, name: str) -> None:
        self._response = response
        self._record = record
        self._name = name
        self._seen = bytearray()
        self._noted = False
        self.headers = response.headers
        self.status = getattr(response, "status", 200)
        self.url = getattr(response, "url", "")

    def _keep(self, chunk: bytes) -> bytes:
        self._seen.extend(chunk)
        return chunk

    def read(self, amt: int | None = None) -> bytes:
        return self._keep(bytes(self._response.read() if amt is None
                                else self._response.read(amt)))

    def read1(self, amt: int = -1) -> bytes:
        reader = getattr(self._response, "read1", None)
        return self._keep(bytes(reader(amt) if reader is not None else self._response.read(amt)))

    def getcode(self) -> int:
        return int(self.status)

    def info(self) -> Any:
        return self.headers

    def _judge(self, error: BaseException | None) -> None:
        if self._noted:
            return
        self._noted = True
        if error is not None:
            typed = classify_exception(error)
            self._record.note(self._name, False, typed.label, typed.kind.value)
            return
        failed = reply_failure(bytes(self._seen))
        if failed is None:
            self._record.note(self._name, True)
        else:
            self._record.note(self._name, False, LABELS[failed[0]], failed[0].value)

    def close(self) -> None:
        self._judge(None)
        self._response.close()

    def __enter__(self) -> _Teed:
        return self

    def __exit__(self, kind: Any, error: BaseException | None, trace: Any) -> None:
        self._judge(error)
        self._response.close()


def install() -> None:
    """Wrap ``urllib.request.urlopen`` once. Outside a recording the wrapper only passes through."""
    global _original
    if _original is None:
        _original = urllib.request.urlopen
        urllib.request.urlopen = _observed
