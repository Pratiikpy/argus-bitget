"""The Skill-effectiveness matrix: every keyless Bitget tool, called, and every failure named.

**Track 3 is judged on "data sources / Skill integration count *and effectiveness*"**
(`BITGET_AI_BASE_CAMP_S2_HANDBOOK_EN.md:281`). Count is easy to inflate and effectiveness is easy
to fake, and ARGUS had three partial answers to the question: `market/skills.py` swept 19
``bitget-signal`` calls into ok / empty / tool_error / timeout / unavailable, `eval/datacoverage.py`
swept the 67 ``bitget-mcp-server`` catalog entries into ok / empty / error / needs_params, and
`eval/skillreliability.py` repeated the first sweep to separate dead from flaky. None of them could
say *whose* failure a failure was, because the clients underneath them could not: a refused
argument, the service's own upstream answering 503, a missing session and a dropped connection all
arrived as one exception class or one status phrase.

**This matrix is built on the shared client and its taxonomy** (`market/rpc.py`, 2026-09-25): every
call on both servers lands in exactly one of six columns —

* ``answered`` — data came back;
* ``empty`` — the tool answered and carried nothing (``market/skills.py:hollow``: an all-zero yield
  curve built from failed legs is empty, not a reading);
* ``domain_error`` — the tool ran and refused: SEP-1303's ``isError``, or ``{"error": "Unknown
  action"}`` / ``success: false`` with a 4xx inside a normal result;
* ``protocol_error`` — a JSON-RPC ``error`` object: the call never reached a tool;
* ``transport_error`` — timeout, unreachable, rate limit, authorisation, unparseable;
* ``upstream_5xx`` — a 5xx, from the MCP host or reported inside the payload by the service
  behind it.

— and the fine-grained :class:`~argus.market.rpc.ErrorKind` is kept beside the column, so the
matrix can be read at either resolution.

**How each call is chosen — read from the servers, not invented.** ``bitget-signal``: one call per
tool, the action a real user of that Skill would make, with argument values taken from the tool's
own ``inputSchema`` descriptions (``BTC/USDT``, ``BTC,ETH,SOL``, ``sh000001``…) or, where
`market/skills.py` already measured a form that answers, that form. The Skill a tool belongs to is
the ``bitget-signal/skills/*/SKILL.md`` that names it most; three tools (``crypto_price``,
``crypto_derivatives``, ``backtest``) are named by no SKILL.md and are listed as server-only rather
than credited to a Skill. ``bitget-mcp-server``: every catalog entry, with the parameters its own
``params_summary`` marks required filled from the probe symbols; when a form comes back empty or
refused, the next symbol form is tried (``BTCUSDT`` then ``BTC``) and every attempt is published.
The entry is counted under the best outcome any form achieved — stated here because it is a choice:
the question is whether the integration *can* answer, and a symbol spelling is ours to get right.

**What a single sweep is and is not.** ``--rounds`` repeats the whole sweep and the matrix reports
rates; one round is a snapshot and is labelled as one. `eval/skillreliability.py` documents a day on
which the same 19 calls read 3 tool errors and then 13 timeouts: an honest effectiveness figure is a
distribution, and a one-round matrix is the first sample of one, not the figure.

**The legacy readings are recomputed beside the typed ones** and every row where they disagree is
counted, so the value of the taxonomy is a measured number rather than a claim.

    python -m argus.eval.skill_matrix [--rounds N] [--spacing SECONDS]
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from argus.eval.artefact import write
from argus.market.rpc import (
    LABELS,
    ErrorKind,
    Outcome,
    RpcError,
    ToolResult,
    outcome_of,
)
from argus.market.skills import hollow

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "skill_matrix.json"
HISTORY_PATH = REPORT_PATH.with_name("skill_matrix_history.jsonl")
"""One line per sweep: when, and how many tools answered on each server. A single sweep is a
snapshot of one moment, and on 2026-09-25 that moment was an outage — every bitget-mcp-server entry
answered 503, and bitget-signal's upstream feeds came back empty to two independent MCP clients
(this one and Claude Code's own). The rate worth trusting is the one measured across many sweeps,
so each sweep is kept rather than overwritten."""

SIGNAL_SERVER = "bitget-signal"
MCP_SERVER = "bitget-mcp-server"
SERVER_ONLY = "server-only (no SKILL.md names it)"

EQUITY_SYMBOL = "NVDA"
"""The anchor of ``NVDAUSDT``, one of the twelve rTokens; liquid enough that an empty answer is a
statement about the endpoint and not about the company."""

CRYPTO_FORMS: tuple[str, ...] = ("BTCUSDT", "BTC")
"""Symbol spellings tried, in order, on crypto catalog entries. The catalog does not say which one
each entry takes (`params_summary` gives the name and type only)."""

ETF_SYMBOL = "QQQ"
NEWS_LABELS: tuple[str, ...] = ("BTC", "Bitcoin", "NVDA")
"""``news_label_search`` requires a ``label``; the catalog does not enumerate labels. Used only
when the catalog declares the label a string."""

NEWS_LABEL_IDS: tuple[int, ...] = (1,)
"""The catalog declares ``label`` an **integer** (``params_summary`` type, read 2026-09-25) and the
server refuses a string with ``Param 'label' must be of type 'integer'`` — the first live sweep sent
strings and published three domain errors that were the probe's fault. The id-to-topic table is not
published, so the lowest id is sent: enough to show whether the entry answers at all."""

TIMEOUT = 45
"""Seconds per call: `market/skills.py:DEFAULT_TIMEOUT`, which measured slow tools at 15-31 s."""

PAUSE = 0.15
"""Between calls. These are Bitget's services and not ours to hammer."""

_BACKTEST_CONFIG = {
    "name": "rsi_reversion", "symbols": ["BTC/USDT"], "timeframe": "1d",
    "indicators": [{"name": "RSI", "params": {"period": 14}, "key": "rsi"}],
    "entry_conditions": [{"indicator": "rsi", "field": "rsi", "operator": "<", "value": 30}],
    "exit_conditions": [{"indicator": "rsi", "field": "rsi", "operator": ">", "value": 70}],
    "direction": "long",
}
"""Built from the keys ``backtest``'s own ``strategy_config`` description lists. On 2026-09-25
four shapes (this one as a string, as an object, double-encoded, and a minimal config) all came back
``{"error": "the JSON object must be str, bytes or bytearray, not dict"}`` — the server parses the
field twice. Recorded as the domain error it is."""


@dataclass(frozen=True, slots=True)
class SignalCall:
    """One ``bitget-signal`` tool, the Skill it belongs to, and the call a user of it would make."""

    skill: str
    tool: str
    args: Mapping[str, Any]
    source: str
    """Where the argument values came from."""


SIGNAL_CALLS: tuple[SignalCall, ...] = (
    SignalCall("technical-analysis", "technical_analysis", {"action": "rsi", "symbol": "NVDAUSDT"},
               "market/skills.py PROBES (measured answering on rToken symbols)"),
    SignalCall("sentiment-analyst", "sentiment_index", {"action": "current"},
               "market/skills.py PROBES"),
    SignalCall("sentiment-analyst", "derivatives_sentiment",
               {"action": "long_short", "symbol": "BTCUSDT", "period": "4h"},
               "market/skills.py PROBES"),
    SignalCall("macro-analyst", "macro_indicators", {"action": "latest_release"},
               "market/skills.py PROBES"),
    SignalCall("macro-analyst", "rates_yields", {"action": "yield_curve"},
               "market/skills.py PROBES"),
    SignalCall("macro-analyst", "cross_asset", {"action": "correlation"},
               "market/skills.py PROBES"),
    SignalCall("macro-analyst", "global_assets", {"action": "price", "symbol": EQUITY_SYMBOL},
               "inputSchema: 'Yahoo Finance symbol: ... AAPL'"),
    SignalCall("macro-analyst", "cn_market", {"action": "index", "symbol": "sh000001"},
               "inputSchema: 'index (sh000001)'"),
    SignalCall("macro-analyst", "global_data",
               {"action": "forex", "base": "USD", "symbols": "EUR,JPY,CNY"},
               "inputSchema: 'Base currency for forex (default: USD)'"),
    SignalCall("news-briefing", "news_feed", {"action": "latest"}, "market/skills.py PROBES"),
    SignalCall("news-briefing", "tradfi_news", {"action": "earnings", "symbol": EQUITY_SYMBOL},
               "market/skills.py PROBES, with the anchor symbol the schema allows"),
    SignalCall("news-briefing", "social_trending", {"action": "trending", "platform": "xueqiu"},
               "inputSchema platform list (xueqiu is the finance board)"),
    SignalCall("market-intel", "crypto_market", {"action": "trending"},
               "market/skills.py PROBES"),
    SignalCall("market-intel", "defi_analytics", {"action": "tvl_rank", "limit": 10},
               "market/skills.py PROBES"),
    SignalCall("market-intel", "dex_market", {"action": "trending"},
               "inputSchema action enum"),
    SignalCall("market-intel", "network_status", {"action": "eth_gas"},
               "market/skills.py PROBES"),
    SignalCall(SERVER_ONLY, "crypto_price", {"action": "price", "symbol": "BTC,ETH,SOL"},
               "inputSchema: \"e.g. 'BTC,ETH,SOL'\""),
    SignalCall(SERVER_ONLY, "crypto_derivatives",
               {"action": "ticker_24h", "symbol": "BTC/USDT", "exchange": "bitget"},
               "tool description: 'price, klines, ticker_24h, multi_price'; 'e.g. BTC/USDT'"),
    SignalCall(SERVER_ONLY, "backtest",
               {"action": "run", "strategy_config": json.dumps(_BACKTEST_CONFIG),
                "period": "6m", "exchange": "bitget"},
               "strategy_config keys from the tool's own description"),
)


# --- classifying one call ----------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Attempt:
    """One call as it happened."""

    args: Mapping[str, Any]
    outcome: Outcome
    kind: str
    """The :class:`ErrorKind` value, or ``""`` for answered and empty."""
    detail: str
    seconds: float
    legacy: str
    """What the pre-taxonomy classifier would have called the same reply."""

    def as_dict(self) -> dict[str, Any]:
        return {"args": dict(self.args), "outcome": self.outcome.value, "kind": self.kind,
                "detail": self.detail[:240], "seconds": round(self.seconds, 2),
                "legacy": self.legacy}


_RANK = {Outcome.ANSWERED: 0, Outcome.EMPTY: 1, Outcome.DOMAIN_ERROR: 2,
         Outcome.UPSTREAM_5XX: 3, Outcome.PROTOCOL_ERROR: 4, Outcome.TRANSPORT_ERROR: 5}


def _mcp_rows(payload: Any) -> Any:
    """bitget-mcp-server wraps rows as ``{"success": .., "data": {"results": [...]}}``."""
    if isinstance(payload, Mapping) and "data" in payload:
        data = payload.get("data")
        if isinstance(data, Mapping) and "results" in data:
            return data.get("results")
        return data
    return payload


def classify_result(result: ToolResult, *, server: str) -> tuple[Outcome, str, str]:
    """``(outcome, kind, detail)`` for a tool result that arrived."""
    failed = result.failure()
    if failed is not None:
        kind, detail = failed
        return outcome_of(kind), kind.value, detail
    payload = result.payload
    body = _mcp_rows(payload) if server == MCP_SERVER else payload
    if hollow(body):
        return Outcome.EMPTY, "", "answered with no data"
    size = len(body) if isinstance(body, list | dict) else 1
    return Outcome.ANSWERED, "", f"{size} item(s)"


def legacy_reading(result: ToolResult | None, error: RpcError | None, *, server: str) -> str:
    """The classifier each server's probe used before 2026-09-25, applied to the same reply.

    ``bitget-signal``: the status `evidence.BitgetSkillSource.call` produced then, fed through
    `market/skills.py:_classify`. Before the migration a JSON-RPC ``error`` frame fell through as an
    empty payload and read "reachable, returned no data" — so a protocol error was counted EMPTY.
    ``bitget-mcp-server``: `eval/datacoverage.py`'s ok / empty / error, where every refusal of any
    kind was one word."""
    if server == MCP_SERVER:
        if error is not None:
            return "error"
        assert result is not None
        if result.is_error:
            return "needs_params" if "validation error" in result.text else "error"
        payload = result.payload
        if isinstance(payload, Mapping) and payload.get("success") is False:
            return "error"
        rows = _mcp_rows(payload)
        return "ok" if isinstance(rows, list) and rows else "empty"
    if error is not None:
        if error.kind is ErrorKind.PROTOCOL:
            return "empty"  # the pre-migration client read the error frame as an empty payload
        if error.kind is ErrorKind.CORRELATION:
            return "unavailable"
        return "timeout" if error.kind is ErrorKind.TIMEOUT else "unavailable"
    assert result is not None
    if result.is_error:
        return "tool_error"
    payload = result.payload
    if isinstance(payload, Mapping) and payload and all(
            str(k).endswith("error") or k == "url" for k in payload):
        return "empty"
    return "empty" if hollow(payload) else "ok"


Clock = Callable[[], float]


def attempt(call: Callable[[], ToolResult], args: Mapping[str, Any], *, server: str,
            clock: Clock = time.monotonic) -> Attempt:
    """Make one call and classify it. Never raises for a failure the taxonomy names."""
    start = clock()
    try:
        result = call()
    except RpcError as exc:
        return Attempt(args, exc.outcome, exc.kind.value, f"{exc.label}: {exc}",
                       clock() - start, legacy_reading(None, exc, server=server))
    outcome, kind, detail = classify_result(result, server=server)
    return Attempt(args, outcome, kind, detail, clock() - start,
                   legacy_reading(result, None, server=server))


# --- the rows ----------------------------------------------------------------------------------

@dataclass
class Row:
    """One tool or catalog entry, across every round."""

    server: str
    skill: str
    tool: str
    source: str
    attempts: list[list[Attempt]] = field(default_factory=list)
    """One list per round; a catalog entry may try several symbol forms inside a round."""

    def best(self, round_attempts: Sequence[Attempt]) -> Attempt:
        return min(round_attempts, key=lambda a: _RANK[a.outcome])

    @property
    def outcomes(self) -> dict[str, int]:
        counts = {o.value: 0 for o in Outcome}
        for tries in self.attempts:
            counts[self.best(tries).outcome.value] += 1
        return counts

    @property
    def rounds(self) -> int:
        return len(self.attempts)

    def as_dict(self) -> dict[str, Any]:
        latest = self.best(self.attempts[-1]) if self.attempts else None
        return {
            "server": self.server, "skill": self.skill, "tool": self.tool,
            "args_source": self.source, "rounds": self.rounds, "outcomes": self.outcomes,
            "answer_rate": round(self.outcomes[Outcome.ANSWERED.value] / self.rounds, 4)
            if self.rounds else 0.0,
            "latest": latest.as_dict() if latest else None,
            "attempts": [[a.as_dict() for a in tries] for tries in self.attempts],
        }


class SignalClient(Protocol):
    def call_tool(self, tool: str, args: dict[str, Any], *, timeout: int = ...) -> ToolResult: ...


class McpClient(Protocol):
    def call_tool(self, name: str, arguments: Mapping[str, Any], *,
                  timeout: float | None = ...) -> ToolResult: ...


Progress = Callable[[str, Attempt], None]


def _quiet(_: str, __: Attempt) -> None:
    return None


def sweep_signal(client: SignalClient, rows: dict[str, Row], *,
                 calls: Sequence[SignalCall] = SIGNAL_CALLS, timeout: int = TIMEOUT,
                 pause: float = PAUSE, sleep: Callable[[float], None] = time.sleep,
                 progress: Progress = _quiet) -> None:
    """One round over every ``bitget-signal`` tool."""
    for spec in calls:
        # Keyed by server as well as name: `crypto_market` exists on both servers, and a first
        # live sweep merged the two into one row before this was keyed so.
        row = rows.setdefault(f"{SIGNAL_SERVER}:{spec.tool}",
                              Row(SIGNAL_SERVER, spec.skill, spec.tool, spec.source))
        args = dict(spec.args)
        row.attempts.append([attempt(
            partial(client.call_tool, spec.tool, args, timeout=timeout),
            args, server=SIGNAL_SERVER)])
        progress(spec.tool, row.attempts[-1][-1])
        sleep(pause)


def entry_forms(entry: Mapping[str, Any], category: str) -> list[dict[str, Any]]:
    """The argument forms to try for one catalog entry, from its own ``params_summary``.

    Required parameters are always filled; an optional ``symbol`` is filled too, because a user asks
    about a named instrument. A form that comes back empty or refused moves to the next spelling."""
    summary = [p for p in entry.get("params_summary") or [] if isinstance(p, Mapping)]
    params = {str(p.get("name")): bool(p.get("required")) for p in summary}
    types = {str(p.get("name")): str(p.get("type", "")) for p in summary}
    if category == "crypto":
        symbols: tuple[str, ...] = CRYPTO_FORMS
    elif category == "etf":
        symbols = (ETF_SYMBOL,)
    else:
        symbols = (EQUITY_SYMBOL,)
    forms: list[dict[str, Any]] = []
    if "label" in params:
        integer = types.get("label") == "integer"
        labels: tuple[Any, ...] = NEWS_LABEL_IDS if integer else NEWS_LABELS
        forms = [{"label": label} for label in labels]
    elif "symbol" in params or "symbols" in params:
        key = "symbol" if "symbol" in params else "symbols"
        forms = [{key: s} for s in symbols]
        if not params.get(key):
            forms.append({})  # optional symbol: the unfiltered call is a real call too
    else:
        forms = [{}]
    return forms


def sweep_mcp(client: McpClient, catalog: Sequence[tuple[str, Mapping[str, Any]]],
              rows: dict[str, Row], *, timeout: int = TIMEOUT, pause: float = PAUSE,
              sleep: Callable[[float], None] = time.sleep, progress: Progress = _quiet) -> None:
    """One round over every ``bitget-mcp-server`` catalog entry."""
    for category, entry in catalog:
        entry_id = str(entry.get("id", ""))
        row = rows.setdefault(f"{MCP_SERVER}:{entry_id}", Row(
            MCP_SERVER, f"{MCP_SERVER}/{category}", entry_id, "catalog params_summary"))
        tries: list[Attempt] = []
        for form in entry_forms(entry, category):
            got = attempt(
                partial(client.call_tool, "do_query", {"entry_id": entry_id, "params": form},
                        timeout=timeout),
                form, server=MCP_SERVER)
            tries.append(got)
            progress(entry_id, got)
            sleep(pause)
            if got.outcome is Outcome.ANSWERED:
                break
            if got.outcome in (Outcome.TRANSPORT_ERROR, Outcome.UPSTREAM_5XX,
                               Outcome.PROTOCOL_ERROR):
                break  # a different spelling cannot fix the path or the service behind it
        row.attempts.append(tries)


# --- the report --------------------------------------------------------------------------------

def _tally(rows: Sequence[Row]) -> dict[str, Any]:
    counts = {o.value: 0 for o in Outcome}
    for row in rows:
        for key, value in row.outcomes.items():
            counts[key] += value
    total = sum(counts.values())
    return {**counts, "calls": total, "tools": len(rows),
            "tools_answering": sum(1 for r in rows if r.outcomes[Outcome.ANSWERED.value]),
            "answer_rate": round(counts[Outcome.ANSWERED.value] / total, 4) if total else 0.0}


def report(rows: Sequence[Row], *, rounds: int, servers: Mapping[str, Mapping[str, Any]],
           generated_at: str) -> dict[str, Any]:
    """The matrix as a JSON-ready dict."""
    by_skill: dict[str, list[Row]] = {}
    for row in rows:
        by_skill.setdefault(row.skill, []).append(row)
    kinds: dict[str, int] = {}
    disagreements: list[dict[str, str]] = []
    for row in rows:
        for tries in row.attempts:
            best = row.best(tries)
            if best.kind:
                kinds[best.kind] = kinds.get(best.kind, 0) + 1
            legacy_ok = best.legacy in ("ok",)
            typed_ok = best.outcome is Outcome.ANSWERED
            legacy_empty = best.legacy == "empty"
            typed_empty = best.outcome is Outcome.EMPTY
            if legacy_ok != typed_ok or legacy_empty != typed_empty:
                disagreements.append({"tool": row.tool, "legacy": best.legacy,
                                      "typed": best.outcome.value, "kind": best.kind})
    failures = sum(n for n in kinds.values())
    legacy_opaque = sum(1 for row in rows for tries in row.attempts
                        if row.best(tries).legacy in ("error", "unavailable")
                        and row.best(tries).kind)
    signal = [r for r in rows if r.server == SIGNAL_SERVER]
    mcp = [r for r in rows if r.server == MCP_SERVER]
    return {
        "generated_at": generated_at,
        "rounds": rounds,
        "servers": {name: dict(info) for name, info in servers.items()},
        "columns": [o.value for o in Outcome],
        "totals": _tally(rows),
        "by_server": {SIGNAL_SERVER: _tally(signal), MCP_SERVER: _tally(mcp)},
        "by_skill": {skill: _tally(group) for skill, group in sorted(by_skill.items())},
        "failure_kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
        "taxonomy_vs_legacy": {
            "failures_typed": failures,
            "failures_legacy_left_as_one_word": legacy_opaque,
            "disagreements": disagreements,
            "note": ("Each row's legacy reading is the pre-2026-09-25 classifier applied to the "
                     "same reply. 'failures_legacy_left_as_one_word' counts failures the old "
                     "probes could only call 'error' or 'unavailable'; a disagreement is a row "
                     "the old classifier put in a different answered/empty/failed bucket."),
        },
        "rows": [r.as_dict() for r in sorted(rows, key=lambda r: (r.server, r.skill, r.tool))],
        "scope_statement": (
            f"Every bitget-signal tool ({len(signal)}) and every bitget-mcp-server catalog entry "
            f"({len(mcp)}) called {rounds} time(s), keyless, through the same shared client the "
            f"desk and the research console use (market/rpc.py). NOT CLAIMED: that "
            f"{rounds} round(s) characterise a service — `eval/skillreliability.py` shows the "
            f"same calls reading differently a day apart. NOT CLAIMED: that a domain error is "
            f"Bitget's fault; for bitget-mcp-server it can be a symbol spelling the catalog does "
            f"not document, and every attempted form is published so a reader can judge. An "
            f"'empty' is a tool answering with nothing, which is a fact about coverage, never "
            f"read as a quiet market."
        ),
    }


def render(matrix: Mapping[str, Any]) -> list[str]:
    """The matrix as text lines, for the console and the CLI."""
    totals = matrix["totals"]
    columns = [c for c in matrix["columns"]]
    lines = [
        f"SKILL EFFECTIVENESS MATRIX — {matrix['rounds']} round(s), {totals['calls']} calls, "
        f"{matrix['generated_at'][:16]}Z, no API key",
    ]
    for name, info in matrix["servers"].items():
        lines.append(f"  {name}: {info.get('server', '?')}, protocol "
                     f"{info.get('negotiated', '?')} negotiated "
                     f"(asked {info.get('requested', '?')}); server/discover "
                     f"{info.get('server_discover', 'not probed')}")
    head = "  " + f"{'skill':<34}" + "".join(f"{c.replace('_', ' '):>16}" for c in columns)
    lines.append(head)
    for skill, tally in matrix["by_skill"].items():
        lines.append("  " + f"{skill:<34}" + "".join(f"{tally[c]:>16}" for c in columns))
    lines.append("  " + f"{'TOTAL':<34}" + "".join(f"{totals[c]:>16}" for c in columns))
    lines.append(
        f"  answering: {totals['tools_answering']}/{totals['tools']} tools "
        f"({totals['answer_rate']:.0%} of calls answered with data)")
    kinds = matrix.get("failure_kinds") or {}
    if kinds:
        lines.append("  failures by kind: " + ", ".join(
            f"{LABELS.get(ErrorKind(k), k)} {n}" for k, n in kinds.items()))
    versus = matrix.get("taxonomy_vs_legacy") or {}
    if versus:
        lines.append(
            f"  vs the old classifiers: {versus.get('failures_legacy_left_as_one_word', 0)} "
            f"failure(s) they could only call 'error'/'unavailable' are now typed; "
            f"{len(versus.get('disagreements', []))} row(s) they put in a different bucket.")
    return lines


def load(path: Path = REPORT_PATH) -> dict[str, Any] | None:
    """The last published matrix, or None when none has been run."""
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return blob if isinstance(blob, dict) else None


def console_lines(path: Path = REPORT_PATH) -> list[str]:
    """What the research console shows when asked how well the Skills work: the published matrix,
    rendered, with its date — or a plain statement that it has not been run. Integration point for
    `lui/research.py`; reads a file, never the network."""
    matrix = load(path)
    if matrix is None:
        return ["The Skill-effectiveness matrix has not been run yet "
                "(python -m argus.eval.skill_matrix)."]
    sweeps = history(path.with_name(HISTORY_PATH.name))
    if not sweeps:
        return render(matrix)
    calls = sum(int(s.get("calls") or 0) for s in sweeps)
    answered = sum(int(s.get("answered") or 0) for s in sweeps)
    per: dict[str, list[int]] = {}
    for sweep in sweeps:
        for server, counts in (sweep.get("by_server") or {}).items():
            got = per.setdefault(str(server), [0, 0, 0])
            got[0] += int(counts.get("answered") or 0)
            got[1] += int(counts.get("calls") or 0)
            got[2] += int(counts.get("upstream_5xx") or 0)
    first, last = str(sweeps[0].get("at", ""))[:16], str(sweeps[-1].get("at", ""))[:16]
    rate = answered / max(calls, 1)
    lines = [
        f"Actionable: across {len(sweeps)} sweep(s) from {first}Z to {last}Z, Bitget's two Skill "
        f"servers answered {answered} of {calls} calls with data ({rate:.0%}) — "
        + "; ".join(f"{name} {a} of {n}" for name, (a, n, _x) in per.items()) + ".",
    ]
    down = [name for name, (a, n, x) in per.items() if n and x == n]
    if down:
        lines.append(
            f"{', '.join(down)}: every call came back with Bitget's own upstream answering HTTP "
            f"503 inside the tool reply — an outage behind their server, not a malformed request. "
            f"It is swept every three hours, and this line changes when it recovers.")
    lines.append(
        "bitget-signal's news, sentiment, macro and long/short tools came back empty to a second, "
        "independent MCP client as well (checked 2026-09-25), so their empties are the server's, "
        "not this client's.")
    working = [str(r["tool"]) for r in matrix.get("rows") or []
               if (r.get("outcomes") or {}).get("answered")]
    if working:
        lines.append(f"Answering with data in the latest sweep ({matrix['generated_at'][:16]}Z): "
                     + ", ".join(sorted(working)) + ".")
    kinds = matrix.get("failure_kinds") or {}
    if kinds:
        lines.append("Why the rest did not, by kind: " + ", ".join(
            f"{str(k).replace('_', ' ')} {v}" for k, v in kinds.items()) + ".")
    lines.append(f"Data: {len(sweeps)} keyless sweep(s) of all {matrix['totals']['calls']} tools "
                 f"(`eval/skill_matrix.py`); every row with its reply is in skill_matrix.json.")
    return lines


def run(*, rounds: int = 1, spacing: float = 0.0, timeout: int = TIMEOUT,
        progress: Progress = _quiet) -> dict[str, Any]:
    """Sweep both servers ``rounds`` times, live. Imports the clients here so a test importing
    this module never opens a socket."""
    from argus.market.bitget_mcp import BitgetDataService
    from argus.market.evidence import BitgetSkillSource

    signal = BitgetSkillSource()
    data = BitgetDataService()
    signal_negotiation = signal.client.initialize()
    catalog: list[tuple[str, Mapping[str, Any]]] = []
    for category in data.categories():
        key = str(category.get("key", ""))
        listing = data._call("guide", {"category": key}).get("structuredContent") or {}
        for entry in listing.get("entries") or listing.get("items") or []:
            if isinstance(entry, Mapping) and entry.get("id"):
                catalog.append((key, entry))
    listed = [str(t.get("name")) for t in signal.client.list_tools(timeout=timeout)]
    missing = sorted(set(listed) - {c.tool for c in SIGNAL_CALLS})
    if missing:
        raise RuntimeError(f"bitget-signal lists tools this matrix has no call for: {missing}")

    rows: dict[str, Row] = {}
    for index in range(rounds):
        sweep_signal(signal, rows, timeout=timeout, progress=progress)
        sweep_mcp(data.client, catalog, rows, timeout=timeout, progress=progress)
        if index + 1 < rounds and spacing:
            time.sleep(spacing)
    servers = {
        SIGNAL_SERVER: {"endpoint": signal.URL, "tools_listed": len(listed),
                        "http_requests": signal.client.calls, **signal_negotiation.as_dict()},
        MCP_SERVER: {"endpoint": data.client.url, "catalog_entries": len(catalog),
                     "http_requests": data.client.calls, **data.negotiation.as_dict()},
    }
    return report(list(rows.values()), rounds=rounds, servers=servers,
                  generated_at=datetime.now(UTC).isoformat())


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI, live network
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="the Track 3 Skill-effectiveness matrix")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--spacing", type=float, default=0.0,
                        help="seconds between rounds")
    parser.add_argument("--timeout", type=int, default=TIMEOUT)
    args = parser.parse_args(argv)
    def show(tool: str, got: Attempt) -> None:
        print(f"  {tool:<48} {got.outcome.value:<16} {got.seconds:6.1f}s  {got.detail[:60]}",
              file=sys.stderr, flush=True)

    matrix = run(rounds=args.rounds, spacing=args.spacing, timeout=args.timeout, progress=show)
    for line in render(matrix):
        print(line)
    write(REPORT_PATH, matrix)
    record_history(matrix)
    print(f"\nwritten to {REPORT_PATH}; sweep appended to {HISTORY_PATH.name}")
    return 0


def history_row(matrix: Mapping[str, Any]) -> dict[str, Any]:
    """The compact line a sweep leaves in the history: calls and answers, per server."""
    by_server = {
        str(server): {"calls": int(counts.get("calls") or 0),
                      "answered": int(counts.get("answered") or 0),
                      "upstream_5xx": int(counts.get("upstream_5xx") or 0)}
        for server, counts in (matrix.get("by_server") or {}).items()}
    totals = matrix.get("totals") or {}
    return {"at": matrix.get("generated_at"), "calls": int(totals.get("calls") or 0),
            "answered": int(totals.get("answered") or 0), "by_server": by_server}


def record_history(matrix: Mapping[str, Any], path: Path | None = None) -> None:
    target = path or HISTORY_PATH
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(history_row(matrix), separators=(",", ":")) + "\n")


def history(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or HISTORY_PATH
    if not target.exists():
        return []
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "HISTORY_PATH", "REPORT_PATH", "SIGNAL_CALLS", "Attempt", "Row", "SignalCall", "attempt",
    "classify_result", "console_lines", "entry_forms", "history", "history_row", "legacy_reading",
    "load", "main", "record_history", "render", "report", "run", "sweep_mcp", "sweep_signal",
]
