"""ARGUS as a Model Context Protocol server: the research desk, callable by any AI agent.

**Why.** A research desk that only a person in a browser can use is one surface; an agent that can
call it is another, and the one Bitget's own ecosystem is built around (Agent Hub ships
`agent-mcp`; `bitget-signal` and `bitget-mcp-server` are MCP endpoints). The owner's earlier POD
project exposed its scores the same way (`apps/pod-web/src/app/api/mcp/route.ts`), which is where
the idea was taken from; POD's endpoint is a minimal JSON-RPC shim, so this one follows the MCP
specification's Streamable HTTP transport instead: ``initialize`` negotiates a protocol version and
declares the ``tools`` capability, ``notifications/initialized`` is accepted with no body,
``tools/list`` returns each tool's JSON Schema, and ``tools/call`` returns ``content`` blocks with
``isError`` set on failure rather than a transport error — as the spec asks, so an agent can read
the reason and retry.

**Every tool runs the console's own engines.** ``argus_ask`` is the whole console — the same
routing, the same refusals, the same receipts. The typed tools build the research request directly
(no reading of language at all), so an agent that already knows what it wants gets exactly that
engine. No tool writes, trades or changes anything: the desk places no orders, and neither does
this.

**Hardened, and deliberately not replaced by the official SDK (2026-09-25).** The synthesis
(`research/mypr-teardowns/_SYNTHESIS.md`, S15) proposed swapping this hand-rolled layer for the
official MCP Python SDK (modelcontextprotocol/python-sdk, MIT). That was run, not argued
(`eval/mcp_sdk_comparison.py` → `data/mcp_sdk_comparison.json`), on one fuzz corpus of 145 hostile
bodies (`eval/mcp_fuzz.py`) with one oracle:

* **Before hardening, this server failed it**: 3 bodies raised out of :func:`handle_body` — a
  string or number ``params`` (``AttributeError``) and JSON nested past the recursion limit — which
  `lui/server.py` turns into an HTTP 500 with no JSON-RPC body; 3 megabyte inputs were echoed back
  a megabyte long; ``id`` values of the wrong type were echoed; ``1e999`` came back as the
  non-JSON token ``Infinity``; an empty batch got silence instead of an error; and 46 of 145
  cases were clean, with 98 engine calls, most of them carrying schema-invalid arguments.
* **The SDK (2.2.0, low-level ``Server`` over its Streamable HTTP app)** fixed the envelope class —
  zero unstructured exceptions, types enforced by pydantic — but still passed schema-invalid
  arguments to the engine in 86 cases, because its low-level server does not validate
  ``arguments`` against ``inputSchema``; its high-level ``MCPServer`` does validate, but only by
  generating schemas from Python signatures (``mcpserver/tools/base.py:63-117``), which would
  change every schema agents see today. It also echoed a megabyte ``id`` and method name (57 of
  145 clean in all), took a median 1.2-1.7 ms per ``tools/list`` against 0.03-0.06 ms here over
  two runs, needs 17 distributions the project does not have (the SDK's own two and 15
  dependencies: Starlette, uvicorn, OpenTelemetry, PyJWT and pywin32 among them), and is an ASGI
  app — so `lui/server.py`'s synchronous ``ThreadingHTTPServer`` handler could only reach it
  through an event-loop bridge or a second server.

So this layer stays, and takes from the SDK what made it better, with citations:

* **One exception-to-wire boundary** (``shared/jsonrpc_dispatcher.py:701-770``): nothing raised
  below :func:`handle_body` escapes it; anything unexpected becomes JSON-RPC ``-32603``.
* **Tool-name validation per SEP-986** (``shared/tool_name_validation.py:21``): a name outside
  ``^[A-Za-z0-9._-]{1,128}$`` is refused before lookup, and no name is echoed past 64 characters.
* **Envelope typing the SDK gets from pydantic**, written out: ``params`` must be an object, an
  ``id`` must be a string, an integer or null, ``arguments`` must be an object.

and adds what neither had: **arguments are validated against each tool's own published
``inputSchema``** before any engine runs, and a violation is returned as a tool execution error
(``isError: true``) naming the field — the 2025-11-25 specification's rule (SEP-1303), so an agent
can correct itself. The schemas themselves are unchanged. After hardening the same corpus is 145
of 145 clean (`data/mcp_fuzz.json`).

    POST /mcp   {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from typing import Any

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
"""Versions of the MCP specification this server speaks, newest first. A client asking for one of
these gets it back; anything else gets the newest, as the spec's version negotiation says."""
SERVER_INFO = {"name": "argus-research-desk", "version": "1.0.0"}

TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "argus_ask",
        "description": ("Ask the ARGUS research desk anything a trader asks about Bitget-listed "
                        "stocks, ETFs, crypto, gold or oil — portfolio impact, stress, execution, "
                        "price, technicals, earnings, events, hedging, macro, sentiment, news, or "
                        "the desk's own track record. Every figure is computed and sourced; "
                        "questions it cannot source are refused with the reason."),
        "inputSchema": {"type": "object", "properties": {
            "question": {"type": "string", "description": "The question, in any language."},
            "book": {"type": "string",
                     "description": "Optional holdings, e.g. '40% NVDA, 30% MSFT, 30% AAPL'."},
            "memory": {"type": "string",
                       "description": ("Optional: the memory string the previous argus_ask "
                                       "returned. The desk keeps nothing between calls; pass it "
                                       "back and what the trader said earlier (a loss limit, a "
                                       "holding period, a thesis) shapes this answer.")}},
            "required": ["question"]},
    },
    {
        "name": "argus_quote",
        "description": "Live Bitget price, spread, funding and round-trip cost for one or more "
                       "contracts.",
        "inputSchema": {"type": "object", "properties": {
            "symbols": {"type": "array", "items": {"type": "string"},
                        "description": "Tickers or contracts, e.g. ['NVDA', 'BTCUSDT']."}},
            "required": ["symbols"]},
    },
    {
        "name": "argus_portfolio_impact",
        "description": "What adding a position of a given size does to a book's risk: share of "
                       "risk, beta, concentration, a hedge, stress lines and a size ceiling.",
        "inputSchema": {"type": "object", "properties": {
            "add": {"type": "string", "description": "The name to add, e.g. 'TSLA'."},
            "size_percent": {"type": "number", "description": "Target weight, e.g. 15."},
            "book": {"type": "object", "additionalProperties": {"type": "number"},
                     "description": "Current weights in percent, e.g. {'NVDA': 50, 'AAPL': 50}."}},
            "required": ["add"]},
    },
    {
        "name": "argus_stress",
        "description": "What a shock to the Nasdaq (default) or to any named instrument does to a "
                       "book, through each holding's beta, plus the realised worst window.",
        "inputSchema": {"type": "object", "properties": {
            "book": {"type": "object", "additionalProperties": {"type": "number"},
                     "description": "Weights in percent."},
            "shock_percent": {"type": "number", "description": "e.g. -10."},
            "shocked": {"type": "string",
                        "description": "Optional instrument shocked instead of QQQ, e.g. 'oil'."}},
            "required": ["book"]},
    },
    {
        "name": "argus_execution_plan",
        "description": "How to split an order of a stated dollar size on Bitget's live order book, "
                       "with the cost of each slice.",
        "inputSchema": {"type": "object", "properties": {
            "symbol": {"type": "string"}, "usd": {"type": "number"}},
            "required": ["symbol", "usd"]},
    },
    {
        "name": "argus_scoreboard",
        "description": "The standing register: every ARGUS capability, the named rival it was run "
                       "against, and whether it is OWNED, TIED, IMPLEMENTED or LOST.",
        "inputSchema": {"type": "object", "properties": {}},
    },
)


class ToolError(ValueError):
    """A tool call that cannot run as asked. Returned to the agent as ``isError``, never raised."""


TOOL_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
"""SEP-986, as the SDK enforces it (``shared/tool_name_validation.py:21``, MIT)."""

MAX_BATCH = 64
"""Messages per batch. JSON-RPC sets no limit; a server has to, or one POST of ten thousand
``tools/call`` requests is ten thousand engine runs."""

MAX_ID_LENGTH = 256
"""A string ``id`` must be echoed back verbatim, so an unbounded one is an amplifier."""

_ECHO = 64
"""Characters of a client-supplied name ever repeated in an error message."""


def _echo(value: Any) -> str:
    text = repr(value) if isinstance(value, str) else type(value).__name__
    return text if len(text) <= _ECHO else text[:_ECHO] + "..."


def schema_errors(schema: Mapping[str, Any], value: Any, path: str = "") -> list[str]:
    """Violations of one tool's published ``inputSchema``, in words an agent can act on.

    Covers exactly the JSON Schema this server publishes — ``type`` (object, string, number,
    array), ``properties``, ``required``, ``items`` and schema-valued ``additionalProperties`` —
    plus one rule the JSON grammar implies: a number must be finite. Booleans are not numbers
    (JSON Schema's own rule; Python's ``bool`` is an ``int``). Properties not declared are
    allowed, as the schemas do not forbid them.
    """
    where = path or "arguments"
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return [f"{where} must be an object, not {type(value).__name__}"]
        props: Mapping[str, Any] = schema.get("properties", {})
        errors = [f"{path + '.' if path else ''}{name} is required"
                  for name in schema.get("required", []) if name not in value]
        extra = schema.get("additionalProperties")
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if key in props:
                errors += schema_errors(props[key], item, child)
            elif isinstance(extra, Mapping):
                errors += schema_errors(extra, item, f"{where}[{_echo(key)}]")
        return errors
    if kind == "string" and not isinstance(value, str):
        return [f"{where} must be a string, not {type(value).__name__}"]
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f"{where} must be a number, not {type(value).__name__}"]
        if not math.isfinite(value):
            return [f"{where} must be a finite number"]
    if kind == "array":
        if not isinstance(value, list):
            return [f"{where} must be an array, not {type(value).__name__}"]
        items = schema.get("items")
        if isinstance(items, Mapping):
            return [error for i, item in enumerate(value)
                    for error in schema_errors(items, item, f"{where}[{i}]")]
    return []


def _symbol(name: str) -> str:
    from argus.lui.research import _resolve

    hit = _resolve(str(name).strip(), trust_case=False) or _resolve(str(name).strip().upper())
    if hit is None:
        raise ToolError(f"{name!r} is not a contract Bitget lists")
    return hit[0]


def _book(raw: Any) -> dict[str, float]:
    if not raw:
        return {}
    if not isinstance(raw, Mapping):
        raise ToolError("book must be an object of name -> percent")
    weights: dict[str, float] = {}
    for name, value in raw.items():
        try:
            weight = float(value) / 100.0
        except (TypeError, ValueError) as exc:
            raise ToolError(f"weight for {name!r} is not a number") from exc
        if weight <= 0:
            raise ToolError(f"weight for {name!r} must be positive")
        symbol = _symbol(str(name))
        weights[symbol] = weights.get(symbol, 0.0) + weight
    total = sum(weights.values())
    return {s: w / total for s, w in weights.items()}


def _answer_text(payload: Mapping[str, Any]) -> str:
    from argus.lui.provenance import labels as provenance_labels

    lines = [str(line) for line in payload.get("lines") or []]
    sources = [s.get("ref") for s in payload.get("sources") or [] if isinstance(s, dict)]
    text = "\n".join(f"[{tag}] {line}" if tag else line
                     for line, tag in zip(lines, provenance_labels(lines), strict=True))
    if sources:
        text += "\n\nSources: " + "; ".join(str(s) for s in sources if s)
    return text


def _run(request: Any, question: str) -> dict[str, Any]:
    from argus.lui.research import run

    answer = run(question, request)
    return {"refused": answer.refused, "lines": answer.lines,
            "sources": [s.as_dict() if hasattr(s, "as_dict") else s for s in answer.sources],
            "reason": getattr(answer, "reason", "")}


def call_tool(name: str, args: Mapping[str, Any]) -> tuple[str, bool]:
    """Run one tool. Returns (text, is_error)."""
    from decimal import Decimal

    from argus.lui.research import ResearchKind, ResearchRequest, _shock_subject

    if name == "argus_ask":
        from argus.lui import server

        question = str(args.get("question") or "").strip()
        if not question:
            raise ToolError("question is required")
        payload = server.handle_ask(question[:500], [], visitor="mcp",
                                    book=str(args.get("book") or "")[:300],
                                    memory=str(args.get("memory") or "")[:12000])
        text = _answer_text(payload)
        if payload.get("memory") and payload.get("memory") != "[]":
            text += f"\n\nMemory (pass back as `memory` next time): {payload['memory']}"
        return text, bool(payload.get("refused"))
    if name == "argus_quote":
        symbols = [_symbol(s) for s in (args.get("symbols") or [])][:4]
        if not symbols:
            raise ToolError("symbols is required")
        result = _run(ResearchRequest(kind=ResearchKind.QUOTE, symbols=tuple(symbols)),
                      f"quote {' '.join(symbols)}")
        return _answer_text(result), result["refused"]
    if name == "argus_portfolio_impact":
        add = _symbol(str(args.get("add") or ""))
        book = _book(args.get("book"))
        size = args.get("size_percent")
        if size is not None and not 0 < float(size) <= 100:
            # `if size` used to read 0 as "not stated" and silently size the position at 20%.
            raise ToolError("size_percent must be above 0 and at most 100")
        request = ResearchRequest(
            kind=ResearchKind.IMPACT, symbols=(add, *[s for s in book if s != add]), book=book,
            size=float(size) / 100.0 if size is not None else 0.2,
            size_stated=size is not None)
        result = _run(request, f"add {size or 20}% {add}")
        return _answer_text(result), result["refused"]
    if name == "argus_stress":
        book = _book(args.get("book"))
        if not book:
            raise ToolError("book is required")
        shocked = args.get("shocked")
        subject = None
        if shocked:
            subject = _shock_subject(f"if {shocked} drops", set(book))
            if subject is None and str(shocked).lower() not in ("nasdaq", "qqq", "market"):
                subject = _symbol(str(shocked))
        shock = args.get("shock_percent")
        request = ResearchRequest(kind=ResearchKind.STRESS, symbols=tuple(book), book=book,
                                  shock_pct=float(shock) if shock is not None else None,
                                  shock_on=subject)
        result = _run(request, "stress my book")
        return _answer_text(result), result["refused"]
    if name == "argus_execution_plan":
        symbol = _symbol(str(args.get("symbol") or ""))
        try:
            usd = Decimal(str(args.get("usd")))
        except ArithmeticError as exc:
            raise ToolError("usd must be a number") from exc
        if usd <= 0:
            raise ToolError("usd must be positive")
        request = ResearchRequest(kind=ResearchKind.EXECUTION, symbols=(symbol,), notional=usd)
        result = _run(request, f"how should I split a ${usd} order in {symbol}")
        return _answer_text(result), result["refused"]
    if name == "argus_scoreboard":
        from argus.lui.answer import _notes_path

        path = _notes_path().parent / "standing.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = [f"{c['state'].upper():12} {c['subtheme']:22} {c['name']} — vs {c['baseline'][:90]}"
                for c in data.get("capabilities", [])]
        return "\n".join([f"States: {data.get('by_state')}", *rows]), False
    raise ToolError(f"unknown tool {name!r}")


def _result(message_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def handle(message: Any, *, tool: Callable[[str, Mapping[str, Any]], tuple[str, bool]] = call_tool
           ) -> dict[str, Any] | None:
    """One JSON-RPC message in, one response out (None for a notification)."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _error(None, -32600, "not a JSON-RPC 2.0 request")
    method = message.get("method")
    message_id = message.get("id")
    if "id" not in message:
        return None  # a notification (e.g. notifications/initialized) takes no response
    if (isinstance(message_id, bool) or not isinstance(message_id, (str, int, type(None)))
            or (isinstance(message_id, str) and len(message_id) > MAX_ID_LENGTH)):
        # JSON-RPC 2.0 §4: an id is a string, a number or null (and SHOULD NOT be fractional).
        # One that cannot be echoed safely is answered as an invalid request with a null id.
        return _error(None, -32600, f"id must be a string of at most {MAX_ID_LENGTH} "
                                    f"characters or an integer")
    if not isinstance(method, str):
        return _error(message_id, -32600, f"method must be a string, not {_echo(method)}")
    raw_params = message.get("params")
    if raw_params is not None and not isinstance(raw_params, dict):
        return _error(message_id, -32602, "params must be an object")
    params: dict[str, Any] = raw_params or {}
    if method == "initialize":
        asked = str(params.get("protocolVersion") or "")
        version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        return _result(message_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": ("ARGUS is a research desk for Bitget's tokenized US stocks and "
                             "listed crypto/commodity contracts. Every figure it returns is "
                             "computed from live data and names its source; it places no "
                             "orders."),
        })
    if method == "ping":
        return _result(message_id, {})
    if method == "tools/list":
        return _result(message_id, {"tools": list(TOOLS)})
    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments")
        if not isinstance(name, str) or not TOOL_NAME.match(name):
            return _error(message_id, -32602, f"tool name {_echo(name)} is not a valid tool "
                                              f"name (SEP-986: [A-Za-z0-9._-], 1 to 128)")
        spec = next((t for t in TOOLS if t["name"] == name), None)
        if spec is None:
            return _error(message_id, -32602, f"unknown tool {_echo(name)}")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return _error(message_id, -32602,
                          f"arguments must be an object, not {type(arguments).__name__}")
        invalid = schema_errors(spec["inputSchema"], arguments)
        if invalid:
            # A tool execution error, not a protocol error: the 2025-11-25 specification
            # (SEP-1303) asks for input validation to come back where the model will read it.
            text = f"invalid arguments for {name}: " + "; ".join(invalid[:5])
            return _result(message_id, {"content": [{"type": "text", "text": text}],
                                        "isError": True})
        try:
            text, is_error = tool(name, arguments)
        except ToolError as exc:
            text, is_error = str(exc), True
        except Exception as exc:  # an engine failure is the tool's error, not the transport's
            text, is_error = f"the desk could not answer: {type(exc).__name__}", True
        return _result(message_id, {"content": [{"type": "text", "text": text}],
                                    "isError": is_error})
    return _error(message_id, -32601, f"method {_echo(method)} not found")


def _finite(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"number {text[:_ECHO]} overflows")
    return value


def _reject_constant(token: str) -> Any:
    raise ValueError(f"{token} is not JSON")


def _encode(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")


def handle_body(body: bytes, *,
                tool: Callable[[str, Mapping[str, Any]], tuple[str, bool]] = call_tool
                ) -> tuple[int, bytes]:
    """HTTP status and body for a POST to /mcp. Batches are answered as a batch.

    The single exception-to-wire boundary, after the SDK's ``JSONRPCDispatcher._handle_request``
    (``shared/jsonrpc_dispatcher.py:701-770``): whatever goes wrong below, the caller receives a
    JSON-RPC error body, never a raised exception.
    """
    try:
        return _handle_body(body, tool)
    except Exception as exc:  # the boundary itself: nothing below may reach the transport raw
        return 500, _encode(_error(None, -32603, f"internal error: {type(exc).__name__}"))


def _handle_body(body: bytes, tool: Callable[[str, Mapping[str, Any]], tuple[str, bool]]
                 ) -> tuple[int, bytes]:
    try:
        # Strict JSON: Python's parser accepts NaN, Infinity and 1e999 (as inf); JSON does not,
        # and a non-finite number that got in would come back out as a non-JSON reply.
        message = json.loads(body.decode("utf-8") or "null", parse_constant=_reject_constant,
                             parse_float=_finite)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return 400, _encode(_error(None, -32700, "parse error"))
    if isinstance(message, list):
        if not message:
            # JSON-RPC 2.0 §6: an empty batch is one Invalid Request, not silence.
            return 400, _encode(_error(None, -32600, "empty batch"))
        if len(message) > MAX_BATCH:
            return 400, _encode(_error(None, -32600, f"batch of {len(message)} exceeds the "
                                                     f"limit of {MAX_BATCH} messages"))
        replies = [r for r in (handle(m, tool=tool) for m in message) if r is not None]
        return (200, _encode(replies)) if replies else (202, b"")
    reply = handle(message, tool=tool)
    return (202, b"") if reply is None else (200, _encode(reply))


__all__ = [
    "MAX_BATCH",
    "PROTOCOL_VERSIONS",
    "TOOLS",
    "TOOL_NAME",
    "ToolError",
    "call_tool",
    "handle",
    "handle_body",
    "schema_errors",
]
