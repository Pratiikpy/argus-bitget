"""How a call to a source failed, and what a failed reply looks like.

Moved from `market/rpc.py` on 2026-09-27: the taxonomy (why a call produced no answer, who
owns the failure, whether a retry can help) and the reading of a reply that carries an
error. `truth/coverage.py` records every source an answer reached in these terms and
imported them upward from the market layer; the design notes and sources for both halves
are in `market/rpc.py`'s docstring, beside the client that raises them.
"""

from __future__ import annotations

import contextlib
import json
import urllib.error
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# --- the taxonomy ------------------------------------------------------------------------------

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
"""The JSON-RPC 2.0 codes, as ``schema/draft/schema.ts:312-316`` restates them."""

HEADER_MISMATCH = -32020
MISSING_REQUIRED_CLIENT_CAPABILITY = -32021
UNSUPPORTED_PROTOCOL_VERSION = -32022
"""MCP's reserved range (``schema.ts:434-450``; renumbered from -32001/-32003/-32004 in 2026-07-28,
so both numberings are accepted by :func:`classify_rpc_error`)."""

_LEGACY_UNSUPPORTED_VERSION = -32004


class ErrorKind(StrEnum):
    """Why a call did not produce an answer. Each kind is a different fact with a different owner.

    The first three are the server telling us something; the rest are the path to it failing.
    """

    PROTOCOL = "protocol"
    """A JSON-RPC ``error`` object: malformed request, unknown method, unknown tool, missing
    session. SEP-1303: the call never reached a tool."""

    UNSUPPORTED_VERSION = "unsupported_version"
    """Version negotiation failed: the server named no version this client implements."""

    TOOL_EXECUTION = "tool_execution"
    """``result.isError`` was true: the tool ran and failed, and said so in its own words."""

    DOMAIN = "domain"
    """The tool answered normally, but the payload says the upstream refused — a
    ``success: false`` with a 4xx, or an ``error`` field carrying a message (ToolBench status
    11)."""

    UPSTREAM_5XX = "upstream_5xx"
    """A 5xx — from the MCP host itself, or reported inside a payload from the service behind it."""

    RATE_LIMIT = "rate_limit"
    """HTTP 429, or a server saying so in prose (ToolBench statuses 9 and 10)."""

    AUTH = "auth"
    """HTTP 401/403 (ToolBench statuses 7 and 8). On a keyless service this usually means the
    request was shaped wrong — ``bitget_mcp.py`` documents a 403 caused by a User-Agent."""

    NOT_FOUND = "not_found"
    """HTTP 404 (ToolBench status 6). On a session transport it also means the session expired
    (``basic/transports.mdx``: a client receiving 404 for its session MUST start a new one)."""

    TIMEOUT = "timeout"
    """No reply inside the deadline (ToolBench status 5)."""

    TRANSPORT = "transport"
    """The request never completed: DNS, refused connection, reset (ToolBench status 12)."""

    UNPARSEABLE = "unparseable"
    """A reply arrived that is not JSON-RPC."""

    CORRELATION = "correlation"
    """A reply arrived, but no frame carries the id that was sent. Returning another request's
    answer is worse than returning none (``evidence.py`` found this live, 2026-09-13)."""


RETRYABLE: frozenset[ErrorKind] = frozenset({
    ErrorKind.TIMEOUT, ErrorKind.TRANSPORT, ErrorKind.RATE_LIMIT, ErrorKind.UPSTREAM_5XX,
})
"""Kinds a second attempt can change. page-agent ``errors.ts:41-52`` for the shape; see the module
docstring for why tool-execution and argument errors are not in it here."""


class Outcome(StrEnum):
    """The six columns of the Skill-effectiveness matrix. Every call lands in exactly one."""

    ANSWERED = "answered"
    EMPTY = "empty"
    DOMAIN_ERROR = "domain_error"
    PROTOCOL_ERROR = "protocol_error"
    TRANSPORT_ERROR = "transport_error"
    UPSTREAM_5XX = "upstream_5xx"


_OUTCOME: dict[ErrorKind, Outcome] = {
    ErrorKind.TOOL_EXECUTION: Outcome.DOMAIN_ERROR,
    ErrorKind.DOMAIN: Outcome.DOMAIN_ERROR,
    ErrorKind.PROTOCOL: Outcome.PROTOCOL_ERROR,
    ErrorKind.UNSUPPORTED_VERSION: Outcome.PROTOCOL_ERROR,
    ErrorKind.CORRELATION: Outcome.PROTOCOL_ERROR,
    ErrorKind.UPSTREAM_5XX: Outcome.UPSTREAM_5XX,
    ErrorKind.RATE_LIMIT: Outcome.TRANSPORT_ERROR,
    ErrorKind.AUTH: Outcome.TRANSPORT_ERROR,
    ErrorKind.NOT_FOUND: Outcome.TRANSPORT_ERROR,
    ErrorKind.TIMEOUT: Outcome.TRANSPORT_ERROR,
    ErrorKind.TRANSPORT: Outcome.TRANSPORT_ERROR,
    ErrorKind.UNPARSEABLE: Outcome.TRANSPORT_ERROR,
}


def outcome_of(kind: ErrorKind) -> Outcome:
    """The matrix column a failure kind is counted under. Total over :class:`ErrorKind`."""
    return _OUTCOME[kind]


LABELS: dict[ErrorKind, str] = {
    ErrorKind.PROTOCOL: "protocol error",
    ErrorKind.UNSUPPORTED_VERSION: "protocol version refused",
    ErrorKind.TOOL_EXECUTION: "tool reported an error",
    ErrorKind.DOMAIN: "upstream refused the request",
    ErrorKind.UPSTREAM_5XX: "upstream 5xx",
    ErrorKind.RATE_LIMIT: "rate limited",
    ErrorKind.AUTH: "refused as unauthorised",
    ErrorKind.NOT_FOUND: "not found",
    ErrorKind.TIMEOUT: "timed out",
    ErrorKind.TRANSPORT: "unreachable",
    ErrorKind.UNPARSEABLE: "unparseable reply",
    ErrorKind.CORRELATION: "reply did not match the request",
}
"""What a reader sees. ``TOOL_EXECUTION`` keeps the exact phrase ``market/skills.py`` and the
coverage line already used, so no downstream string match changes meaning."""


class RpcError(RuntimeError):
    """A call that did not produce an answer, with the kind of failure attached.

    Subclassed by each client's historic error type (``BitgetMcpError``), so an ``except`` written
    against the old name still catches, and a caller that wants the kind reads ``.kind``.
    """

    def __init__(
        self, kind: ErrorKind, message: str, *, http_status: int | None = None,
        code: int | None = None, data: Any = None, attempts: int = 1,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.http_status = http_status
        self.code = code
        self.data = data
        self.attempts = attempts
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE

    @property
    def outcome(self) -> Outcome:
        return outcome_of(self.kind)

    @property
    def label(self) -> str:
        return LABELS[self.kind]


_RATE_WORDS = ("rate limit", "too many requests", "ratelimit", "throttl")
_AUTH_WORDS = ("unauthorized", "unauthorised", "forbidden", "not subscribed", "unsubscribed",
               "invalid api key", "authentication")


def _prose_kind(text: str) -> ErrorKind | None:
    """ToolBench reads the server's words (``rapidapi.py:364-378``) when the status says nothing."""
    low = text.lower()
    if any(w in low for w in _RATE_WORDS):
        return ErrorKind.RATE_LIMIT
    if any(w in low for w in _AUTH_WORDS):
        return ErrorKind.AUTH
    return None


def classify_rpc_error(error: Mapping[str, Any], *, http_status: int | None = None) -> RpcError:
    """A JSON-RPC ``error`` object, typed."""
    code = error.get("code")
    message = str(error.get("message", ""))
    data = error.get("data")
    icode = code if isinstance(code, int) else None
    if icode in (UNSUPPORTED_PROTOCOL_VERSION, _LEGACY_UNSUPPORTED_VERSION):
        kind = ErrorKind.UNSUPPORTED_VERSION
    else:
        kind = _prose_kind(message) or ErrorKind.PROTOCOL
    return RpcError(kind, f"JSON-RPC error {code}: {message}", http_status=http_status,
                    code=icode, data=data)


def classify_http(status: int, body: bytes, *, retry_after: str | None = None) -> RpcError:
    """An HTTP error status, typed — reading the body first, because an MCP server returns its
    JSON-RPC errors inside a 400 (both Bitget servers do, for a missing session)."""
    after: float | None = None
    if retry_after:
        with contextlib.suppress(ValueError):
            after = float(retry_after)
    text = body.decode("utf-8", "replace")
    if status == 429:
        return RpcError(ErrorKind.RATE_LIMIT, f"HTTP 429: {text[:200]}", http_status=status,
                        retry_after=after)
    if status in (401, 403):
        return RpcError(ErrorKind.AUTH, f"HTTP {status}: {text[:200]}", http_status=status)
    if status == 404:
        return RpcError(ErrorKind.NOT_FOUND, f"HTTP 404: {text[:200]}", http_status=status)
    if status >= 500:
        return RpcError(ErrorKind.UPSTREAM_5XX, f"HTTP {status}: {text[:200]}",
                        http_status=status, retry_after=after)
    for frame in parse_frames(text):
        error = frame.get("error")
        if isinstance(error, Mapping):
            return classify_rpc_error(error, http_status=status)
    kind = _prose_kind(text) or ErrorKind.PROTOCOL
    return RpcError(kind, f"HTTP {status}: {text[:200]}", http_status=status)


def classify_exception(exc: BaseException) -> RpcError:
    """Anything ``urlopen`` raises, typed. ``HTTPError`` is read for its status and body."""
    if isinstance(exc, RpcError):
        return exc
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read() or b""
        except (OSError, ValueError):
            body = b""
        after = exc.headers.get("Retry-After") if exc.headers is not None else None
        return classify_http(int(exc.code), body, retry_after=after)
    reason: Any = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
        return RpcError(ErrorKind.TIMEOUT, f"timed out ({type(reason).__name__})")
    if isinstance(exc, json.JSONDecodeError | UnicodeDecodeError):
        return RpcError(ErrorKind.UNPARSEABLE, f"{type(exc).__name__}: {exc}")
    return RpcError(ErrorKind.TRANSPORT, f"{type(exc).__name__}: {reason}")


# --- reading a reply ---------------------------------------------------------------------------

def parse_frames(raw: str) -> list[dict[str, Any]]:
    """Every JSON object in a reply, whether it came as a bare body or as SSE ``data:`` lines.

    Streamable HTTP may interleave notifications with the response on one stream, so the caller
    picks its frame by id rather than by position."""
    lines = [ln[5:].strip() for ln in raw.splitlines() if ln.startswith("data:")]
    frames: list[dict[str, Any]] = []
    for line in lines or [raw]:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            frames.append(parsed)
        elif isinstance(parsed, list):  # a JSON-RPC batch reply
            frames.extend(p for p in parsed if isinstance(p, dict))
    return frames


def _status_code(payload: Mapping[str, Any]) -> int | None:
    """An HTTP-like status carried in a payload. The generic names ``code`` and ``status`` are read
    only beside an envelope marker (``success``/``error``/``message``): on a bare data row they are
    as likely to be an instrument code or a state word as a status."""
    keys: tuple[str, ...] = ("status_code", "statusCode", "http_status")
    if any(k in payload for k in ("success", "error", "message")):
        keys = (*keys, "code", "status")
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
        if isinstance(value, str) and value.isdigit() and 100 <= int(value) <= 599:
            return int(value)
    return None


def payload_failure(payload: Any) -> tuple[ErrorKind, str] | None:
    """A failure reported *inside* a normal tool result, or None when the payload is not one.

    Two shapes are live on Bitget's servers. bitget-mcp-server wraps its upstream as
    ``{"success": false, "status_code": 503, ...}`` (seen while that service was down; answer audit
    round 3). bitget-signal answers ``{"error": "Unknown action: latest"}``. An ``error`` key
    holding an *empty* string is not a failure report — it is the upstream's hollow envelope, which
    the matrix counts as empty, not as a refusal (``market/skills.py:_classify``)."""
    if not isinstance(payload, Mapping):
        return None
    status = _status_code(payload)
    if payload.get("success") is False or (status is not None and status >= 400):
        detail = str(payload.get("message") or payload.get("msg") or payload.get("error") or "")
        if status is not None and status >= 500:
            return ErrorKind.UPSTREAM_5XX, f"upstream answered {status} {detail}".strip()
        if status == 429:
            return ErrorKind.RATE_LIMIT, f"upstream answered 429 {detail}".strip()
        kind = _prose_kind(detail) or ErrorKind.DOMAIN
        return kind, (f"upstream answered {status} {detail}" if status else
                      f"success=false {detail}").strip()
    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        return _prose_kind(error) or ErrorKind.DOMAIN, error.strip()[:200]
    if isinstance(error, Mapping) and (error.get("message") or error.get("code")):
        return _prose_kind(str(error.get("message", ""))) or ErrorKind.DOMAIN, \
            str(error.get("message") or error.get("code"))[:200]
    return None


@dataclass(frozen=True, slots=True)
class ToolResult:
    """One ``tools/call`` result, read the way the 2025-11-25 schema defines it."""

    tool: str
    content: tuple[dict[str, Any], ...]
    structured: Any
    is_error: bool

    @classmethod
    def from_result(cls, tool: str, result: Mapping[str, Any]) -> ToolResult:
        content = result.get("content") or []
        blocks = tuple(dict(c) for c in content if isinstance(c, Mapping)) \
            if isinstance(content, list) else ()
        return cls(tool=tool, content=blocks, structured=result.get("structuredContent"),
                   is_error=bool(result.get("isError")))

    @property
    def text(self) -> str:
        """The text channel, joined: where both Bitget servers put their payload and errors."""
        return "\n".join(str(c.get("text", "")) for c in self.content if c.get("type", "text")
                         == "text")

    @property
    def payload(self) -> Any:
        """``structuredContent`` when present, else the text parsed as JSON, else the text."""
        if self.structured is not None:
            return self.structured
        text = self.text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def failure(self) -> tuple[ErrorKind, str] | None:
        """SEP-1303's tool-execution error first, then a failure reported inside the payload."""
        if self.is_error:
            return ErrorKind.TOOL_EXECUTION, self.text[:200]
        return payload_failure(self.payload)


def reply_failure(body: bytes) -> tuple[ErrorKind, str] | None:
    """What went wrong in one raw MCP reply body, or None if it answered. For ``truth/coverage``,
    which sees bytes at the transport and never a parsed result."""
    frames = parse_frames(body.decode("utf-8", "replace"))
    if not frames:
        return (ErrorKind.UNPARSEABLE, "no JSON-RPC frame") if body.strip() else None
    for frame in frames:
        error = frame.get("error")
        if isinstance(error, Mapping):
            failed = classify_rpc_error(error)
            return failed.kind, str(failed)
        result = frame.get("result")
        if isinstance(result, Mapping):
            return ToolResult.from_result("", result).failure()
    return None


__all__ = [
    "HEADER_MISMATCH",
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "LABELS",
    "METHOD_NOT_FOUND",
    "MISSING_REQUIRED_CLIENT_CAPABILITY",
    "PARSE_ERROR",
    "RETRYABLE",
    "UNSUPPORTED_PROTOCOL_VERSION",
    "ErrorKind",
    "Outcome",
    "RpcError",
    "ToolResult",
    "classify_exception",
    "classify_http",
    "classify_rpc_error",
    "outcome_of",
    "parse_frames",
    "payload_failure",
    "reply_failure",
]
