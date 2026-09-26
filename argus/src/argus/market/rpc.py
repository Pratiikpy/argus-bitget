"""One JSON-RPC client for every MCP server ARGUS reads, and a taxonomy that says what broke.

**Why this exists (2026-09-25).** ARGUS talked to two keyless Bitget MCP servers through two
hand-rolled clients. ``market/bitget_mcp.py`` folded every failure — a refused argument, a 503 from
the service's own upstream, a dropped connection — into one ``BitgetMcpError`` whose only content
was a string. ``market/evidence.py`` carried a second client with its own framing, its own id
counter and a status *string* that ``market/skills.py`` then re-parsed with substring checks. Both
pinned an old protocol version and neither ever asked the server what it spoke.
``truth/coverage.py`` found MCP failures by scanning raw bytes for ``"success":false``. Three
places answering one question — *did that call work, and if not, whose fault was it* — and none
of them could tell a rate limit from a bad argument.

Track 3 scores "Skill integration count **and effectiveness**", and effectiveness is not measurable
without that distinction: a tool that refuses a malformed argument and a tool whose upstream is down
are different facts about the integration, and only one of them is ours to fix.

**What was taken, and from where.**

* **Protocol error versus tool-execution error** — MCP SEP-1303
  (``modelcontextprotocol/seps/1303-input-validation-errors-as-tool-execution-errors.md``, Final,
  shipped in 2025-11-25) and ``schema/draft/schema.ts:1823-1837``: a failure *inside* a tool is a
  result with ``isError: true``; failing to *find* the tool, an unsupported method or a malformed
  request is a JSON-RPC ``error`` object. :class:`ErrorKind` keeps them as separate kinds.
  The numeric codes are the spec's (``schema.ts:312-316`` for the JSON-RPC five,
  ``schema.ts:434-450`` for ``HEADER_MISMATCH`` -32020, ``MISSING_REQUIRED_CLIENT_CAPABILITY``
  -32021 and ``UNSUPPORTED_PROTOCOL_VERSION`` -32022, whose ``data.supported`` list is how a client
  recovers, ``schema.ts:463-513``). Licence: the specification repository is mixed MIT/Apache-2.0
  (``licenses/mcp_specification-APACHE-2.0.txt``); only constants and interface shapes are used.
* **A named status for every way a real API call fails** — ToolBench
  ``toolbench/inference/Downstream_tasks/rapidapi.py:286-388`` (Apache-2.0,
  ``licenses/toolbench-APACHE-2.0.txt``). Its thirteen statuses separate timeout (5), 404 (6), not
  subscribed (7), unauthorized (8), too many requests (9), rate limit (10), an ``error`` field in
  the message (11) and a failure sending the request (12). Taken: the idea that each of these is a
  distinct, countable state, and the message-text fallbacks for rate limit and authorisation when a
  server says so in prose rather than in a status code. Rejected: its statuses 3 and 4 (the model
  finishing or giving up) — those are agent-loop states, not transport states, and do not belong
  in a client — and its practice of sleeping inside the classifier on a rate limit.
* **A retryable flag derived from the type, never decided at the call site** — page-agent
  ``packages/llms/src/errors.ts:41-52`` (MIT): ``RETRYABLE_TYPES`` lists the retryable kinds and the
  error carries ``retryable`` computed from its type. Taken as :data:`RETRYABLE`. **Departed from
  deliberately:** page-agent also retries tool-execution errors and invalid arguments, because its
  caller is an LLM that will regenerate the call. Ours is deterministic — the same arguments sent
  again get the same refusal — so here only the network (timeout, transport), the server's capacity
  (rate limit) and its upstream (5xx) are retried.

**What was found by probing, 2026-09-25, before a line of this was written.** Both servers
(``agent.bitget.com/mcp``, bitget-mcp-server 4.0.5; ``datahub.noxiaohao.com/mcp``, the bitget-signal
data service) negotiate correctly per the lifecycle spec
(``docs/specification/2025-11-25/basic/lifecycle.mdx:165-180``): asked for 2024-11-05, 2025-03-26 or
2025-06-18 they echo it; asked for 2026-07-28 or an invented ``1999-01-01`` they answer
**2025-11-25**, their newest. So both clients had been asking for a version two revisions old for no
reason. Neither implements ``server/discover`` — the 2026-07-28 stateless revision's up-front probe
(``docs/specification/2026-07-28/changelog.mdx``, major change 3) — and both answer it with HTTP 400
carrying a JSON-RPC ``-32600 Bad Request: Missing session ID``. That 400 body is itself a protocol
error, not a transport one, which is why :func:`classify_http` reads the body before deciding.

**Version policy.** :data:`SUPPORTED_VERSIONS` stops at 2025-11-25. 2026-07-28 removed the
``initialize`` handshake and sessions altogether; claiming it without implementing per-request
``_meta`` negotiation would be a lie told to the server. :meth:`JsonRpcClient.initialize` probes
``server/discover`` once per endpoint per process and records what came back, so the day a server
starts answering it the negotiation record says so.

**Coverage keeps working.** Every request goes through ``urllib.request.urlopen`` looked up *at call
time*, which is the one function ``truth/coverage.py`` wraps; binding it at import would silently
remove every MCP call from the "Sources reached" line.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

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


# --- the client --------------------------------------------------------------------------------

def _read_until(response: Any, wanted: Any, deadline: float, timeout: float) -> str:
    """Read a reply body until EOF, until the frame with id ``wanted`` is complete, or until the
    wall-clock ``deadline`` — whichever comes first. A notification (``wanted`` None) reads to EOF
    under the same deadline."""
    reader = getattr(response, "read1", None) or response.read
    chunks: list[bytes] = []
    while True:
        if time.monotonic() > deadline:
            pings = b"".join(chunks).count(b": ping")
            held = f" (stream held open by {pings} keep-alive ping(s))" if pings else ""
            raise RpcError(ErrorKind.TIMEOUT, f"no complete reply within {timeout:.0f}s{held}")
        chunk = bytes(reader(65536))
        if not chunk:
            break
        chunks.append(chunk)
        if wanted is not None and b"\n" in chunk:
            text = b"".join(chunks).decode("utf-8", "replace")
            if any(frame.get("id") == wanted for frame in parse_frames(text)
                   if "result" in frame or "error" in frame):
                return text
    return b"".join(chunks).decode("utf-8", "replace")


SUPPORTED_VERSIONS: tuple[str, ...] = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
"""Newest first. Every revision whose ``initialize`` handshake this client implements; see the
module docstring for why 2026-07-28 is not claimed."""

LATEST_SUPPORTED = SUPPORTED_VERSIONS[0]

_DISCOVERED: dict[str, str] = {}
_DISCOVER_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class Negotiation:
    """What the handshake settled, kept so a report can print it rather than assume it."""

    requested: str
    negotiated: str
    server: str
    discover: str
    """What ``server/discover`` said, once per endpoint per process."""

    def as_dict(self) -> dict[str, str]:
        return {"requested": self.requested, "negotiated": self.negotiated,
                "server": self.server, "server_discover": self.discover}


Sleep = Callable[[float], None]


@dataclass
class JsonRpcClient:
    """A session against one Streamable-HTTP MCP endpoint.

    Monotonic ids, the reply frame matched by id, the session id and negotiated version sent on
    every request after the handshake (``MCP-Protocol-Version`` is required on HTTP from
    2025-06-18), and a retry loop that only ever retries a :data:`RETRYABLE` kind. An expired
    session (404 with a session id set) is re-established once, as the transport spec requires.
    """

    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout: float = 45.0
    retries: int = 1
    """Extra attempts for a retryable failure. One: enough to ride out a blip, not enough to turn a
    45-second timeout into three minutes inside a live answer."""
    backoff: float = 0.5
    max_retry_after: float = 10.0
    retry_timeouts: bool = True
    """Whether a TIMEOUT is retried. A client whose slow tools are measured to be slow rather than
    flaky turns this off, so a dead tool costs one deadline and not two (``evidence.py``)."""
    versions: tuple[str, ...] = SUPPORTED_VERSIONS
    client_info: Mapping[str, str] = field(
        default_factory=lambda: {"name": "argus", "version": "1.0"})
    probe_discover: bool = True
    sleep: Sleep = time.sleep
    error_type: type[RpcError] = RpcError

    session: str | None = field(default=None, init=False)
    negotiation: Negotiation | None = field(default=None, init=False)
    calls: int = field(default=0, init=False)
    """HTTP requests actually sent, retries included."""
    _next_id: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    ACCEPT: ClassVar[str] = "application/json, text/event-stream"

    def next_id(self) -> int:
        with self._lock:
            self._next_id += 1
            return self._next_id

    # --- one HTTP exchange ---

    def _headers(self, *, handshake: bool) -> dict[str, str]:
        out = {"Content-Type": "application/json", "Accept": self.ACCEPT, **self.headers}
        if self.session:
            out["Mcp-Session-Id"] = self.session
        if not handshake and self.negotiation is not None:
            out["MCP-Protocol-Version"] = self.negotiation.negotiated
        return out

    def _exchange(self, body: Mapping[str, Any], *, timeout: float, handshake: bool) -> str:
        """POST once. Returns the raw reply text; raises a typed error.

        **The deadline is wall-clock, not per read (found live, 2026-09-25).** The bitget-signal
        server holds a slow tool's SSE stream open with a ``: ping`` comment every 15 seconds; a
        ``cn_market`` call was watched for 150 s receiving nothing but pings. ``urlopen``'s timeout
        bounds each socket read, and a ping arrives inside every one, so the old ``read()`` would
        have waited forever — the first live matrix sweep sat on that call for over twenty minutes.
        The body is therefore read incrementally against one deadline, and reading stops as soon
        as the frame carrying this request's id has arrived."""
        request = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), headers=self._headers(handshake=handshake),
            method="POST",
        )
        self.calls += 1
        wanted = body.get("id")
        deadline = time.monotonic() + timeout
        try:
            # Looked up at call time: `truth/coverage.py` wraps this attribute.
            with urllib.request.urlopen(request, timeout=timeout) as response:
                sid = response.headers.get("Mcp-Session-Id") or \
                    response.headers.get("mcp-session-id")
                raw = _read_until(response, wanted, deadline, timeout)
        except RpcError as exc:
            raise self._typed(exc) from exc
        except Exception as exc:  # every urlopen failure is classified, none is swallowed
            raise self._typed(classify_exception(exc)) from exc
        if sid and not self.session:
            self.session = sid
        return raw

    def _typed(self, error: RpcError) -> RpcError:
        if isinstance(error, self.error_type):
            return error
        return self.error_type(error.kind, str(error), http_status=error.http_status,
                               code=error.code, data=error.data, attempts=error.attempts,
                               retry_after=error.retry_after)

    def _once(self, body: Mapping[str, Any], *, timeout: float, handshake: bool) -> dict[str, Any]:
        raw = self._exchange(body, timeout=timeout, handshake=handshake)
        frames = parse_frames(raw)
        if not frames:
            raise self.error_type(ErrorKind.UNPARSEABLE,
                                  f"unparseable reply from {self.url}: {raw[:200]}")
        wanted = body.get("id")
        for frame in frames:
            if frame.get("id") != wanted:
                continue
            error = frame.get("error")
            if isinstance(error, Mapping):
                raise self._typed(classify_rpc_error(error))
            result = frame.get("result")
            return dict(result) if isinstance(result, Mapping) else {}
        raise self.error_type(
            ErrorKind.CORRELATION,
            f"no JSON-RPC frame with id {wanted!r} in the reply "
            f"(ids seen: {[f.get('id') for f in frames]}); refusing to return another "
            f"request's response",
        )

    def _with_retry(self, body: Mapping[str, Any], *, timeout: float,
                    handshake: bool) -> dict[str, Any]:
        attempt = 0
        renewed = False
        while True:
            attempt += 1
            try:
                return self._once(body, timeout=timeout, handshake=handshake)
            except RpcError as exc:
                exc.attempts = attempt
                if (exc.kind is ErrorKind.NOT_FOUND and self.session and not handshake
                        and not renewed):
                    # Session expired: start a new one and resend, once.
                    renewed = True
                    self.session, self.negotiation = None, None
                    self.initialize()
                    continue
                if not exc.retryable or attempt > self.retries or (
                        exc.kind is ErrorKind.TIMEOUT and not self.retry_timeouts):
                    raise
                wait = self.backoff * (2 ** (attempt - 1))
                if exc.retry_after is not None:
                    wait = min(max(wait, exc.retry_after), self.max_retry_after)
                self.sleep(wait)

    def close(self, *, timeout: float = 5.0) -> None:
        """End the session: an HTTP DELETE carrying ``Mcp-Session-Id``, as the Streamable-HTTP
        transport specifies for a client that no longer needs it (a server may answer 405 when
        it does not allow clients to end sessions; that is not an error here).

        Added 2026-09-26. No client in this project ever ended a session, and ``bitget-mcp-server``
        began refusing new ones with JSON-RPC -32603 "Too many open sessions" (argus/data/
        skill_runs/sweep_2026-09-26_*.log). Best effort: a failure to close is never raised,
        because the work the session served is already done."""
        with self._lock:
            session, self.session, self.negotiation = self.session, None, None
        if not session:
            return
        headers = {**self.headers, "Mcp-Session-Id": session}
        request = urllib.request.Request(self.url, headers=headers, method="DELETE")
        try:
            with urllib.request.urlopen(request, timeout=timeout):
                pass
        except Exception:
            pass

    def __enter__(self) -> JsonRpcClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # --- the protocol ---

    def _discover(self, timeout: float) -> str:
        """Probe ``server/discover`` once per endpoint per process. Never fatal."""
        with _DISCOVER_LOCK:
            known = _DISCOVERED.get(self.url)
        if known is not None:
            return known
        body = {"jsonrpc": "2.0", "id": self.next_id(), "method": "server/discover", "params": {}}
        try:
            result = self._once(body, timeout=timeout, handshake=True)
            versions = result.get("supportedVersions") or result.get("protocolVersions") or []
            note = f"answered: supports {', '.join(map(str, versions)) or 'unstated versions'}"
        except RpcError as exc:
            note = f"not implemented ({exc.label}: {str(exc)[:120]})"
        with _DISCOVER_LOCK:
            _DISCOVERED.setdefault(self.url, note)
            return _DISCOVERED[self.url]

    def initialize(self, *, timeout: float | None = None) -> Negotiation:
        """The lifecycle handshake with version negotiation (``lifecycle.mdx:165-180``).

        Ask for the newest version we implement. Accept the server's answer if we implement it too;
        on ``UNSUPPORTED_PROTOCOL_VERSION`` pick the newest mutual version from ``data.supported``
        and ask once more (``schema.ts:463-513``); otherwise refuse, typed."""
        if self.negotiation is not None:
            return self.negotiation
        wait = self.timeout if timeout is None else timeout
        # A short deadline for the probe: on a dead host it must not double the wait.
        discover = self._discover(min(wait, 10.0)) if self.probe_discover else "not probed"
        requested = self.versions[0]
        for _ in range(2):
            body = {"jsonrpc": "2.0", "id": self.next_id(), "method": "initialize", "params": {
                "protocolVersion": requested, "capabilities": {},
                "clientInfo": dict(self.client_info)}}
            try:
                result = self._with_retry(body, timeout=wait, handshake=True)
            except RpcError as exc:
                offered = exc.data.get("supported") if isinstance(exc.data, Mapping) else None
                mutual = [v for v in self.versions if isinstance(offered, list) and v in offered]
                if exc.kind is ErrorKind.UNSUPPORTED_VERSION and mutual and \
                        mutual[0] != requested:
                    requested = mutual[0]
                    continue
                raise
            answered = str(result.get("protocolVersion", ""))
            if answered not in self.versions:
                raise self.error_type(
                    ErrorKind.UNSUPPORTED_VERSION,
                    f"{self.url} negotiated {answered!r}; this client implements "
                    f"{', '.join(self.versions)}",
                )
            info = result.get("serverInfo") or {}
            server = f"{info.get('name', '?')} {info.get('version', '?')}"
            self.negotiation = Negotiation(requested, answered, server, discover)
            break
        # Required by the lifecycle before normal operation. Some servers answer it with 202 and
        # an empty body; a failure here does not stop tool calls on either Bitget server.
        with contextlib.suppress(RpcError):
            self.notify("notifications/initialized")
        assert self.negotiation is not None
        return self.negotiation

    def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        """A JSON-RPC notification: no id, no reply expected."""
        body: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            body["params"] = dict(params)
        self._exchange(body, timeout=self.timeout, handshake=False)

    def request(self, method: str, params: Mapping[str, Any] | None = None, *,
                timeout: float | None = None) -> dict[str, Any]:
        """One request after the handshake. Returns ``result``; raises :class:`RpcError`."""
        self.initialize(timeout=timeout)
        body = {"jsonrpc": "2.0", "id": self.next_id(), "method": method,
                "params": dict(params or {})}
        return self._with_retry(body, timeout=self.timeout if timeout is None else timeout,
                                handshake=False)

    def call_tool(self, name: str, arguments: Mapping[str, Any], *,
                  timeout: float | None = None) -> ToolResult:
        """``tools/call``. A protocol failure raises; a tool failure comes back in the result,
        as SEP-1303 puts it there — read it with :meth:`ToolResult.failure`."""
        result = self.request("tools/call", {"name": name, "arguments": dict(arguments)},
                              timeout=timeout)
        return ToolResult.from_result(name, result)

    def list_tools(self, *, timeout: float | None = None) -> list[dict[str, Any]]:
        """Every tool the server lists, following ``nextCursor`` pagination."""
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(50):
            page = self.request("tools/list", {"cursor": cursor} if cursor else {},
                                timeout=timeout)
            tools.extend(dict(t) for t in page.get("tools", []) if isinstance(t, Mapping))
            cursor = page.get("nextCursor")
            if not cursor:
                break
        return tools


def _reset_discovery_cache() -> None:
    """For tests: forget every endpoint's ``server/discover`` answer."""
    with _DISCOVER_LOCK:
        _DISCOVERED.clear()


__all__ = [
    "LABELS", "LATEST_SUPPORTED", "RETRYABLE", "SUPPORTED_VERSIONS", "ErrorKind", "JsonRpcClient",
    "Negotiation", "Outcome", "RpcError", "ToolResult", "classify_exception", "classify_http",
    "classify_rpc_error", "outcome_of", "parse_frames", "payload_failure", "reply_failure",
]
