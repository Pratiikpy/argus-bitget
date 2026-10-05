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
can correct itself. The schemas themselves are unchanged. After hardening the same corpus was 145
of 145 clean; it is built from the tools' own schemas, so it grew with the four tools added on
2026-09-27 to 195 cases, all clean (`data/mcp_fuzz.json`).

    POST /mcp   {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from argus.lui.plural import resolve_plurals

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
"""Versions of the MCP specification this server speaks, newest first. A client asking for one of
these gets it back; anything else gets the newest, as the spec's version negotiation says."""
SERVER_INFO = {"name": "argus-research-desk", "version": "1.0.0"}

MAX_QUESTION = 500
"""The longest question ``argus_ask`` reads; a longer one is cut and the answer says so."""

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
    # The four below reach the console's newest engines directly, not through question parsing
    # (audit, 2026-09-26: an MCP client could not reach any of them).
    {
        "name": "argus_research_task",
        "description": "One complete research task for adding a name to a book: eight engines "
                       "(price and cost, technicals, news and filings, earnings, past analogues, "
                       "the book's risk, sector and factor exposure, execution) and one verdict "
                       "composed from their figures — add, add smaller, or do not add.",
        "inputSchema": {"type": "object", "properties": {
            "question": {"type": "string", "description": (
                "The question as a trader asks it, e.g. 'I hold 40% NVDA, 30% MSFT, 30% AAPL — "
                "should I add 15% TSLA?'")},
            "book": {"type": "string", "description": (
                "Optional holdings, used when the question names none.")}},
            "required": ["question"]},
    },
    {
        "name": "argus_week_ahead",
        "description": "What to watch in the coming days for a book: scheduled US macro releases, "
                       "each holding's earnings date, this week's 8-K filings, and the days the "
                       "stock market is shut while the perpetuals keep trading.",
        "inputSchema": {"type": "object", "properties": {
            "book": {"type": "string",
                     "description": "Holdings, e.g. '40% NVDA, 60% BTC'. Optional."},
            "days": {"type": "integer", "description": "The window, 1 to 14 days; default 7."}}},
    },
    {
        "name": "argus_exposures",
        "description": "A book's sector weights and its loadings on eight factors (market, size, "
                       "momentum, value, quality, low volatility, rates, crypto), before and "
                       "after an optional added position.",
        "inputSchema": {"type": "object", "properties": {
            "book": {"type": "string", "description": "Holdings, e.g. '50% NVDA, 50% AAPL'."},
            "add": {"type": "string",
                    "description": "Optional position to add, e.g. 'TSLA 15%'."}},
            "required": ["book"]},
    },
    {
        "name": "argus_review_trades",
        "description": "Review the trader's own trades: each one checked against the price path "
                       "and the earnings calendar, the patterns across them, and a checklist to "
                       "run before the next entry.",
        "inputSchema": {"type": "object", "properties": {
            "fills": {"type": "string", "description": (
                "The trades, pasted as a table of fills (date, symbol, side, quantity, price) or "
                "written out, e.g. 'bought 10 NVDA at 180 on 2 Sep, sold at 172 on 9 Sep'.")}},
            "required": ["fills"]},
    },
)


class ToolError(ValueError):
    """A tool call that cannot run as asked. Returned to the agent as ``isError``, never raised."""


ERROR_META = "argus/error"
"""The ``_meta`` key a failed ``tools/call`` result carries its classification under."""


def error_meta(exc: BaseException | None) -> dict[str, Any]:
    """What kind of failure this was and whether asking again can help, for the calling agent.

    Bitget's own agent SDK returns every tool failure as a typed payload — a category from a closed
    set and a ``retryable`` flag (``agent-sdk/src/utils/errors.ts:16-27``,
    ``utils/error-catalog.ts:14-33``, MIT) — so an agent can back off on a rate limit and stop on a
    bad argument without parsing prose (research/harvest/50-bitget-agent-sdk.md). This takes its
    categories, and fills them from the taxonomy the network layer already raises
    (`truth/failures.ErrorKind`), so nothing is classified twice. ``None`` is a refused argument:
    the caller's to fix, never worth repeating unchanged. It goes in ``_meta``, the MCP result's
    extension field, beside the text the model reads, so a client that knows nothing of it loses
    nothing."""
    from argus.truth.failures import ErrorKind, RpcError

    if exc is None or isinstance(exc, ToolError):
        return {"category": "param", "retryable": False}
    if isinstance(exc, RpcError):
        category = {ErrorKind.RATE_LIMIT: "rate", ErrorKind.AUTH: "auth",
                    ErrorKind.NOT_FOUND: "param", ErrorKind.TIMEOUT: "network",
                    ErrorKind.TRANSPORT: "network",
                    ErrorKind.UPSTREAM_5XX: "network"}.get(exc.kind, "unknown")
        meta: dict[str, Any] = {"category": category, "retryable": exc.retryable,
                                "kind": exc.kind.value}
        if exc.retry_after is not None:
            meta["retry_after_seconds"] = exc.retry_after
        return meta
    return {"category": "unknown", "retryable": False}


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


EXAMPLE_ARGUMENTS: dict[str, dict[str, Any]] = {
    "argus_ask": {"question": "where is NVDA trading right now"},
    "argus_quote": {"symbols": ["NVDA", "TSLA"]},
    "argus_portfolio_impact": {"add": "TSLA", "size_percent": 15,
                               "book": {"NVDA": 50, "AAPL": 50}},
    "argus_stress": {"book": {"NVDA": 40, "MSFT": 30, "AAPL": 30}, "shock_percent": -10},
    "argus_execution_plan": {"symbol": "NVDA", "usd": 50000},
    "argus_research_task": {"question": "I hold 40% NVDA, 30% MSFT, 30% AAPL - should I add "
                                        "15% TSLA?"},
    "argus_week_ahead": {"book": "50% NVDA, 50% BTC", "days": 7},
    "argus_exposures": {"book": "40% NVDA, 30% MSFT, 30% AAPL", "add": "TSLA"},
    "argus_review_trades": {"fills": "bought 10 NVDA at 180, sold 10 NVDA at 190"},
}
"""A working call for each tool that takes arguments, shown after an invalid one: the errors said
what was wrong but never how to call it right (fresh-eyes audit, 2026-09-29). Each is exercised by
`tests/test_mcp_server.py`, so an example that stops working fails the build."""


def schema_errors(schema: Mapping[str, Any], value: Any, path: str = "") -> list[str]:
    """Violations of one tool's published ``inputSchema``, in words an agent can act on.

    Covers exactly the JSON Schema this server publishes — ``type`` (object, string, number,
    array), ``properties``, ``required``, ``items`` and schema-valued ``additionalProperties`` —
    plus one rule the JSON grammar implies: a number must be finite. Booleans are not numbers
    (JSON Schema's own rule; Python's ``bool`` is an ``int``). Properties not declared are
    refused at the top level by name, since a misspelt one would otherwise be dropped silently.
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
            elif not path and props:
                # `argus_stress` called with "shock_symbol"/"shock_pct" dropped both silently and
                # ran its QQQ -10% default as if asked (a judge, round 31): a tool's own argument
                # names are the only ones it reads, so any other is refused by name
                errors.append(f"{_echo(key)} is not an argument of this tool (it takes "
                              f"{', '.join(props)})")
        return errors
    if kind == "string" and not isinstance(value, str):
        return [f"{where} must be a string, not {type(value).__name__}"]
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f"{where} must be a number, not {type(value).__name__}"]
        if not math.isfinite(value):
            return [f"{where} must be a finite number"]
    if kind == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
        # JSON Schema's whole number; 5.0 and true are refused (added with `argus_week_ahead`).
        return [f"{where} must be a whole number, not {type(value).__name__}"]
    if kind == "array":
        if not isinstance(value, list):
            return [f"{where} must be an array, not {type(value).__name__}"]
        items = schema.get("items")
        if isinstance(items, Mapping):
            return [error for i, item in enumerate(value)
                    for error in schema_errors(items, item, f"{where}[{i}]")]
    return []


def _symbol(name: str) -> str:
    from argus.lui.research.parse import resolve_name

    hit = (resolve_name(str(name).strip(), trust_case=False)
           or resolve_name(str(name).strip().upper()))
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
        if weight == 0:
            raise ToolError(f"weight for {name!r} must not be zero")
        # A negative weight is a short, as the console reads "-30% TSLA": the tool rejected any
        # short while the console accepted them (a hostile review, round 21)
        symbol = _symbol(str(name))
        weights[symbol] = weights.get(symbol, 0.0) + weight
    gross = sum(abs(w) for w in weights.values())
    if gross == 0:
        raise ToolError("book has no weight")
    if any(w < 0 for w in weights.values()):
        # a book with shorts is weights of equity, leverage included: {NVDA: -50, AAPL: 150} is
        # 2x gross and 100% net, and scaling it to 100% gross made it another book (round 40
        # hostile, C1)
        return weights
    return {s: w / gross for s, w in weights.items()}


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

    from argus.lui.research import ResearchKind, ResearchRequest
    from argus.lui.research.parse import shock_subject

    if name == "argus_ask":
        from argus.lui import server

        question = str(args.get("question") or "").strip()
        if not question:
            from argus.lui.answer import EMPTY_QUESTION_HINT

            raise ToolError(f"question is required. {EMPTY_QUESTION_HINT}")
        payload = server.handle_ask(question[:MAX_QUESTION], [], visitor="mcp",
                                    book=str(args.get("book") or "")[:300],
                                    memory=str(args.get("memory") or "")[:12000])
        text = _answer_text(payload)
        if len(question) > MAX_QUESTION:
            # Cut without a word used to be silent (QA surfaces pass, 2026-09-28): an agent that
            # sent a long question could not tell which half was answered.
            text = (f"Note: only the first {MAX_QUESTION} of the question's {len(question)} "
                    f"characters were read.\n\n{text}")
        if payload.get("memory") and payload.get("memory") != "[]":
            text += f"\n\nMemory (pass back as `memory` next time): {payload['memory']}"
        # A refusal is the tool not delivering its answer, so it is reported as ``isError``: the
        # specification sends tool errors back to the model with their content, which here says
        # what can be asked instead. Reporting it as success would let an agent take a refusal
        # for an answer (weighed and kept after the judge audit of 2026-09-29).
        return text, bool(payload.get("refused"))
    if name == "argus_quote":
        wanted = list(args.get("symbols") or [])
        if not wanted:
            raise ToolError("symbols is required")
        # ["ZZZZ", "NVDA", "; DROP TABLE x", "BTCUSDT"] failed the whole batch on ZZZZ, with no
        # quote for NVDA or BTCUSDT (round 42 hostile, minor 12): good names are quoted and the
        # rest are named as not listed
        quoted: list[str] = []
        unlisted: list[str] = []
        for raw in wanted:
            try:
                hit = _symbol(raw)
            except ToolError:
                unlisted.append(str(raw)[:40])
                continue
            if hit not in quoted:
                quoted.append(hit)
        if not quoted:
            raise ToolError("none of " + ", ".join(repr(u) for u in unlisted)
                            + " is a contract Bitget lists")
        result = _run(ResearchRequest(kind=ResearchKind.QUOTE, symbols=tuple(quoted[:4])),
                      f"quote {' '.join(quoted[:4])}")
        text = _answer_text(result)
        if unlisted:
            text += ("\n\nNot quoted: " + ", ".join(repr(u) for u in unlisted)
                     + (" is" if len(unlisted) == 1 else " are") + " not a contract Bitget lists.")
        if quoted[4:]:
            text += ("\n\nNot quoted: only the first four names are quoted in one call; ask again "
                     "for " + ", ".join(quoted[4:]) + ".")
        return text, result["refused"]
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
        text = _answer_text(result)
        given = [float(v) for v in (args.get("book") or {}).values()]
        stated = sum(given)
        if book and all(v > 0 for v in given) and abs(stated - 100.0) > 0.5:
            # {NVDA: 50, AAPL: 70} was rescaled to 100% with no word, where the console says so
            # (a hostile review, round 29)
            text = (f"Note: the book's weights add up to {stated:g}%, so each was scaled in "
                    f"proportion to make 100% — give weights that sum to 100% (cash included) to "
                    f"have them read as stated.\n{text}")
        return text, result["refused"]
    if name == "argus_stress":
        book = _book(args.get("book"))
        if not book:
            raise ToolError("book is required")
        # weights that add past 100% are leverage and under 100% leave cash: {NVDA: 200, AAPL:
        # 100} was stressed as 67/33 with no word (a hostile review, round 25)
        stated_gross = sum(abs(float(v)) for v in (args.get("book") or {}).values()) / 100.0
        shocked = args.get("shocked")
        subject = None
        if shocked:
            subject = shock_subject(f"if {shocked} drops", set(book))
            if subject is None and str(shocked).lower() not in ("nasdaq", "qqq", "market"):
                subject = _symbol(str(shocked))
        shock = args.get("shock_percent")
        try:
            shock_value = float(shock) if shock is not None else None
        except (TypeError, ValueError) as exc:
            raise ToolError("shock_percent must be a number, e.g. -10") from exc
        if shock_value is not None and shock_value <= -100:
            # -500 was stressed as a -556% loss on a long book (a hostile review, round 22)
            raise ToolError("shock_percent must be above -100: a price cannot fall by more than "
                            "all of it")
        from argus.lui.research.dispatch import MAX_SHOCK_UP

        if shock_value == 0:
            # 0 gave a +0.00% headline and then a tree grown from an "assumed" -10% (round 39
            # hostile, defect 7)
            raise ToolError("shock_percent of 0 moves nothing; give the move to test, e.g. -10, "
                            "or leave it out for the standard -5% and -10%")

        if shock_value is not None and (shock_value > MAX_SHOCK_UP or shock_value != shock_value):
            # +1,000,000 was stressed into a +798,527% book move (round 38 hostile, defect 4)
            raise ToolError(f"shock_percent must be at most {MAX_SHOCK_UP:g}: a larger rise is "
                            f"beyond every history the betas are read from")
        note: tuple[str, ...] = ()
        leverage, cash = None, 0.0
        if stated_gross > 1.0 + 1e-9:
            leverage = round(stated_gross, 4)
            note = (f"the weights add to {stated_gross:.0%}, read as {stated_gross:.2f}x "
                    f"leverage on the account",)
        elif stated_gross < 1.0 - 1e-9:
            cash = round(1.0 - stated_gross, 6)
            note = (f"the weights add to {stated_gross:.0%}, so the other {cash:.0%} is read as "
                    f"cash",)
        request = ResearchRequest(kind=ResearchKind.STRESS, symbols=tuple(book), book=book,
                                  shock_pct=shock_value, shock_on=subject, leverage=leverage,
                                  cash=cash, notes=note)
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
        if usd > Decimal("1e11"):
            # a $1 quadrillion order got a 24-hour schedule and a 2-billion-bps cost (a hostile
            # review, round 23); past $100bn no Bitget book is anywhere near the order
            raise ToolError("usd must be at most 100000000000: an order that size is many times "
                            "any Bitget contract's daily volume, so no schedule for it means "
                            "anything")
        request = ResearchRequest(kind=ResearchKind.EXECUTION, symbols=(symbol,), notional=usd)
        result = _run(request, f"how should I split a ${usd} order in {symbol}")
        return _answer_text(result), result["refused"]
    if name == "argus_scoreboard":
        from argus.lui.answer import desk_notes_path

        path = desk_notes_path().parent / "standing.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = [f"{c['state'].upper():12} {c['subtheme']:22} {c['name']} — vs {c['baseline'][:90]}"
                for c in data.get("capabilities", [])]
        return "\n".join([f"States: {data.get('by_state')}", *rows]), False
    if name == "argus_research_task":
        return _research_task(str(args.get("question") or ""), str(args.get("book") or ""))
    if name == "argus_week_ahead":
        from argus.lui.watchlist import watchlist

        days = args.get("days")
        if days is not None and not (isinstance(days, int) and 1 <= days <= 14):
            raise ToolError("days must be a whole number from 1 to 14")
        book_given = str(args.get("book") or "")[:300]
        lines, sources, _ = watchlist("what should I watch this week", book_given, days=days)
        # a stateless call remembers nothing: the book is the one passed in this call, and a set
        # of weights that does not add to 100% is said to have been scaled (round 39 judge, m-1:
        # "100% gold, 100% oil" came back as "Remembered: your saved book — 50% XAU, 50% CL")
        stated = sum(float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", book_given))
        scaled = (f" (the weights given add to {stated:g}%, so they were scaled to 100%)"
                  if stated and abs(stated - 100) > 0.5 else "")
        lines = [re.sub(r"^Remembered: your saved book — (.*?)\.?$",
                        lambda m: f"The book passed in this call — {m.group(1)}{scaled}.",
                        str(line)) for line in lines]
        return _engine_text(lines, sources), False
    if name == "argus_exposures":
        from argus.lui.exposures import answer as exposures

        holdings = str(args.get("book") or "").strip()
        if not holdings:
            raise ToolError("book is required, e.g. '50% NVDA, 50% AAPL'")
        add = str(args.get("add") or "").strip()
        sized = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", f"{holdings} {add}")]
        if any(x > 1000 for x in sized):
            # "NVDA 1000000%" ran through the regression into a market beta of 10411 (round 38
            # hostile, defect 5)
            raise ToolError("a weight above 1000% of the book is not a position; give weights as "
                            "percent of the book, e.g. book '50% NVDA, 50% AAPL', add 'TSLA 10%'")
        signed = re.match(r"^\s*(?:short\s+)?(?P<name>[A-Za-z][\w.]{0,11})\s*"
                          r"(?P<pct>-\s?\d+(?:\.\d+)?)\s*%\s*$|^\s*"
                          r"(?P<pct2>-\s?\d+(?:\.\d+)?)\s*%\s*(?P<name2>[A-Za-z][\w.]{0,11})"
                          r"\s*$", add)
        if signed is not None:
            # "TSLA -15%" is a 15% short (a hostile review, round 23: it was added long)
            size = (signed.group("pct") or signed.group("pct2") or "").replace(" ", "").lstrip("-")
            ticker = signed.group("name") or signed.group("name2")
            add_words = f"short {size}% {ticker}"
            held = re.search(rf"(?<![\w.])(\d+(?:\.\d+)?)\s*%\s*{re.escape(ticker)}\b|"
                             rf"\b{re.escape(ticker)}\s*(\d+(?:\.\d+)?)\s*%", holdings, re.I)
            if held is not None:
                # an add is added to what is held: "NVDA -100%" on a 100% NVDA book is flat, and
                # it was reported as a -100% book (round 37 hostile audit, defect 15)
                was = float(held.group(1) or held.group(2))
                net = was - float(size)
                if abs(net) < 1e-9:
                    return (f"Adding -{size}% {ticker} to a book that holds {was:g}% {ticker} "
                            f"closes the position: the book is flat, so every sector weight and "
                            f"factor loading is zero. Nothing else was held to measure."
                            if abs(was - 100) < 1e-9 else
                            f"Adding -{size}% {ticker} to the {was:g}% held closes {ticker}; ask "
                            f"for the book without it to see the rest's exposures."), False
                add_words = (f"add {net:g}% {ticker}" if net > 0 else f"short {-net:g}% {ticker}")
        else:
            add_words = f"add {add}" if add else ""
        asked = "what are my sector and factor exposures" + (
            f" if I {add_words}" if add_words else "")
        found = exposures(asked, holdings[:300])
        if found is None:  # pragma: no cover - the question above always asks for exposures
            raise ToolError("the exposures engine did not recognise the request")
        text = _engine_text(found.lines, found.sources)
        weights = [float(x) for x in re.findall(r"(-?\d+(?:\.\d+)?)\s*%", holdings)]
        if weights and all(w > 0 for w in weights) and abs(sum(weights) - 100) > 0.5 \
                and not re.search(r"\bcash\b", holdings, re.I):
            # "60% NVDA, 70% AAPL" was read as 46/54 under "used your saved book", the 130% never
            # said (round 40 hostile, M2) — the same note the impact tool gives
            text = (f"Note: the book's weights add up to {sum(weights):g}%, so each was scaled in "
                    f"proportion to make 100% — give weights that sum to 100% (cash included) to "
                    f"have them read as stated.\n{text}")
        return text, found.refused
    if name == "argus_review_trades":
        from argus.lui.journal import review_trades

        fills = str(args.get("fills") or "").strip()
        if not fills:
            raise ToolError("fills is required")
        reviewed = review_trades(fills[:20000], explicit=True)
        if reviewed is None:
            raise ToolError("no trades could be read from fills. A pasted table needs a header "
                            "row naming its columns (at least side and price, e.g. "
                            "'time,symbol,side,price,qty'); or write the trades out, e.g. "
                            "'bought 10 NVDA at 180 on 2 Sep, sold at 172 on 9 Sep'")
        lines, sources, _ = reviewed
        return _engine_text(lines, sources), False
    raise ToolError(f"unknown tool {name!r}")


def _engine_text(lines: Sequence[str], sources: Sequence[Any]) -> str:
    """An engine's lines and sources, formatted as `argus_ask` formats a console answer."""
    return _answer_text({"lines": list(lines),
                         "sources": [s.as_dict() if hasattr(s, "as_dict") else s
                                     for s in sources]})


_POINTS_BACK: Final = re.compile(r"\b(?:my|the)\s+(?:position|book|holding|portfolio|trade)\b|"
                                 r"\b(?:it|that|this|the\s+rest)\b", re.I)
_CLAUSE_HEAD: Final = re.compile(r"[\u2014\u2013]|,\s*(?:and\s+)?"
                                 r"(?=(?:is|are|should|can|could|what|how|which|does|do)\s)", re.I)


def _research_task(question: str, book: str) -> tuple[str, bool]:
    """The research task as text: what was read, the verdict, each engine's action, each step."""
    from argus.lui.task import read_question, research_task

    question = question.strip()
    if not question:
        raise ToolError("question is required")
    from argus.lui.research import option_shock, signed_book

    # a book of leveraged longs and shorts, or an option position under a volatility move, is one
    # question about the whole position: the add-a-name template read "2x long BTC and -1x
    # inverse ETH" as 100% BTC, and a covered call's vol spike as a sentiment read (round 39
    # judge, C-8 and the leveraged book)
    from argus.lui.research import plan as _plan

    # "Write me a 3-step plan: 20k, AI capex peaking, defined-risk US stock trade" is a plan, not
    # an add to a book (round 41 judge, M12); the console's plan is the same here
    planned = _plan.lines(question[:500])
    if planned:
        return "\n".join(["Read as: a trade plan from a stated view, capital and constraint", "",
                          *planned]), False
    whole = signed_book.lines(question[:500]) or option_shock.lines(question[:500])
    if whole:
        return "\n".join(["Read as: one question about the whole position, answered by the "
                          "engine for it", "", *whole]), False
    reading = read_question(question[:500], book[:300])
    if isinstance(reading, str):
        return reading, True
    task = research_task(reading=reading, asked=question[:500])
    # the further questions are answered here, one by one, by the console itself: "and should I
    # hedge the rest with gold?" was declined with "ask it in the console on its own" (round 39
    # judge, M-4), on the surface that cannot open the console
    out = [f"Read as: {reading.summary}",
           *(f"  ({n})" for n in reading.notes if not n.startswith("not run on this page"))]
    if task.verdict is not None:
        out += ["", f"Verdict: {task.verdict.call}", *task.verdict.lines]
    if task.conclusion:
        out += ["", "What to do, engine by engine:",
                *(f"- {title}: {action}" for title, action in task.conclusion)]
    for step in task.steps:
        state = "not applicable" if not step.applicable else (
            "did not answer" if step.refused else f"{step.seconds:.1f}s")
        out += ["", f"{step.title} ({step.engine}; {state})", *step.lines[:6]]
    if reading.rest:
        from argus.lui import server

        held = ", ".join(f"{w:.0%} {s.removesuffix('USDT')}" for s, w in reading.book.items())
        for clause in reading.rest:
            # "what happens to my position" means nothing alone: a clause that points back at
            # the message is asked with the sentence it points to
            asked = re.sub(r"\bit\b", reading.name.removesuffix("USDT"), clause) \
                if reading.name else clause
            if _POINTS_BACK.search(asked):
                asked = f"{_CLAUSE_HEAD.split(question[:500], maxsplit=1)[0].strip()} — {asked}"
            try:
                said = server.handle_ask(asked, [question[:500]], visitor="mcp", book=held)
            except Exception as exc:  # the task above stands without this part
                out += ["", f"Also asked — “{clause}”: could not run just now "
                            f"({type(exc).__name__})."]
                continue
            lines = [str(x) for x in said.get("lines") or []]
            out += ["", f"Also asked — “{clause}” ({said.get('classified_by', 'console')}):",
                    *lines[:6]]
    return "\n".join(out), False


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
            text = (f"invalid arguments for {name}: " + "; ".join(invalid[:5])
                    + f". A call that works: {json.dumps(EXAMPLE_ARGUMENTS[name])}"
                    if name in EXAMPLE_ARGUMENTS else
                    f"invalid arguments for {name}: " + "; ".join(invalid[:5]))
            return _result(message_id, {"content": [{"type": "text", "text": text}],
                                        "isError": True, "_meta": {ERROR_META: error_meta(None)}})
        failure: BaseException | None = None
        try:
            text, is_error = tool(name, arguments)
            text = resolve_plurals(text)
        except ToolError as exc:
            text, is_error, failure = str(exc), True, exc
        except Exception as exc:  # an engine failure is the tool's error, not the transport's
            text, is_error, failure = f"the desk could not answer: {type(exc).__name__}", True, exc
        body: dict[str, Any] = {"content": [{"type": "text", "text": text}], "isError": is_error}
        if is_error:
            body["_meta"] = {ERROR_META: error_meta(failure)}
        return _result(message_id, body)
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
