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

    POST /mcp   {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
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
        request = ResearchRequest(
            kind=ResearchKind.IMPACT, symbols=(add, *[s for s in book if s != add]), book=book,
            size=float(size) / 100.0 if size else 0.2, size_stated=size is not None)
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
    params = message.get("params") or {}
    if message_id is None:
        return None  # a notification (e.g. notifications/initialized) takes no response
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
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        if name not in {t["name"] for t in TOOLS}:
            return _error(message_id, -32602, f"unknown tool {name!r}")
        try:
            text, is_error = tool(name, arguments if isinstance(arguments, Mapping) else {})
        except ToolError as exc:
            text, is_error = str(exc), True
        except Exception as exc:  # an engine failure is the tool's error, not the transport's
            text, is_error = f"the desk could not answer: {type(exc).__name__}", True
        return _result(message_id, {"content": [{"type": "text", "text": text}],
                                    "isError": is_error})
    return _error(message_id, -32601, f"method {method!r} not found")


def handle_body(body: bytes) -> tuple[int, bytes]:
    """HTTP status and body for a POST to /mcp. Batches are answered as a batch."""
    try:
        message = json.loads(body.decode("utf-8") or "null")
    except (UnicodeDecodeError, ValueError):
        return 400, json.dumps(_error(None, -32700, "parse error")).encode()
    if isinstance(message, list):
        replies = [r for r in (handle(m) for m in message) if r is not None]
        return (200, json.dumps(replies).encode()) if replies else (202, b"")
    reply = handle(message)
    return (202, b"") if reply is None else (200, json.dumps(reply, ensure_ascii=False).encode())


__all__ = ["PROTOCOL_VERSIONS", "TOOLS", "ToolError", "call_tool", "handle", "handle_body"]
