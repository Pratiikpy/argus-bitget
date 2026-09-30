"""The desk's record against simply holding the market, over the same days.

"What's ARGUS's edge over a simple buy-and-hold strategy?" was answered "No open positions" (a
judge's audit, 2026-09-30). It is the first question anyone should ask of a trading agent, and the
record answers it exactly: the desk's net result on its stated capital, from its first decision to
now, against holding QQQ (Bitget's QQQUSDT) over the same span, with no cost charged to the
holder. Nothing here is flattering by construction — the benchmark pays no fees, and a desk with a
handful of settled trades is said to be unable to show an edge either way, whichever side of the
benchmark it sits.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from argus.lui.trace import trace_module
from argus.paper.corrections import is_voided
from argus.paper.ledger import PaperLedger
from argus.paper.performance import evaluate_ledger
from argus.truth.source import Source

EDGE_Q = re.compile(
    r"\b(?:edge|alpha|outperform\w*|beat(?:s|ing)?|better\s+than|worse\s+than|compare[sd]?|"
    r"versus|vs\.?)\b[^?]{0,60}\b(?:buy[\s-]+and[\s-]+hold\w*|b&h|holding\s+(?:the\s+)?(?:market|"
    r"index|qqq|spy|s&p)|the\s+market|the\s+index|(?:the\s+)?benchmark|qqq|spy|s&p(?:\s*500)?)\b|"
    r"\bbuy[\s-]+and[\s-]+hold\b[^?]{0,40}\b(?:edge|better|beat|vs\.?|versus|compare)", re.I)
"""The desk asked to show it does better than holding the market."""

BENCHMARK = "QQQUSDT"
"""The holding the desk is measured against: its risk layer and stress tests read QQQ as the
market, so the comparison uses the same one."""

SHOWS_AN_EDGE_AT = 30
"""Settled trades below which no comparison is called an edge, either way: a rule of thumb, since
under thirty a single trade's luck can move the result more than any skill could be told from."""


def _benchmark_return(start: datetime, closes: Callable[[str], Sequence[tuple[datetime, float]]]
                      ) -> tuple[float, datetime, float, float] | None:
    """Holding the benchmark from the first daily close at or after ``start`` to the last one."""
    series = [(t, c) for t, c in closes(BENCHMARK) if c > 0]
    after = [(t, c) for t, c in series if t >= start]
    if len(after) < 2:
        return None
    first, last = after[0], after[-1]
    return last[1] / first[1] - 1.0, first[0], first[1], last[1]


def _bitget_daily(symbol: str) -> list[tuple[datetime, float]]:
    from argus.market.bitget import fetch_candles

    rows = fetch_candles(symbol, granularity="1D", limit=200)
    return [(row["ts"], float(str(row["close"]))) for row in rows
            if isinstance(row["ts"], datetime)]


def answer(ledger: PaperLedger, *, closes: Callable[[str], Sequence[tuple[datetime, float]]]
           | None = None) -> tuple[list[str], list[Source], dict[str, Any]]:
    """The desk's return against holding QQQ over the same days, and whether it can say anything."""
    perf = evaluate_ledger(ledger)
    if not ledger.entries:
        return (["Bottom line: the record has no decisions yet, so there is nothing to compare "
                 "with holding the market."], [], {})
    start = datetime.fromisoformat(ledger.entries[0].decided_at.replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    desk = float(Decimal(str(perf.net_pnl)) / perf.capital)
    try:
        held = _benchmark_return(start, closes or _bitget_daily)
    except Exception:
        held = None
    trades = perf.trades
    decisions = len(ledger.entries)
    few = trades < SHOWS_AN_EDGE_AT
    if held is None:
        lead = (f"Bottom line: the desk returned {desk:+.2%} on its stated capital since "
                f"{start:%d %b %Y}; QQQ's closes did not answer just now, so the comparison is "
                f"not made rather than made against a stale price.")
        bench: dict[str, Any] = {}
    else:
        ret, since, first, last = held
        side = "ahead of" if desk > ret else "behind"
        lead = (f"Bottom line: since its first decision on {start:%d %b %Y} the desk returned "
                f"{desk:+.2%} on its stated capital, {side} holding QQQ at {ret:+.2%} — "
                + (f"and with {trades} settled trade{'' if trades == 1 else 's'} that shows no "
                   f"edge either way: under {SHOWS_AN_EDGE_AT}, one trade's luck outweighs any "
                   f"skill." if few else
                   "over enough settled trades for the gap to mean something."))
        bench = {"benchmark": BENCHMARK, "from": since.isoformat(), "first_close": first,
                 "last_close": last, "return": ret}
    voided = sum(1 for e in ledger.entries if is_voided(e.seq))
    lines = [
        lead,
        # Abstentions counted as the ledger counts them (`Entry.is_abstention`), voided rows
        # included — the desk refused both — so this answer and the track record agree (878, not
        # 876: a hostile review, 2026-09-30).
        f"The record: {decisions} decisions, {perf.abstentions} of them abstentions, "
        f"{trades} settled trade{'' if trades == 1 else 's'}, net {float(perf.net_pnl):+,.2f} "
        f"USDT on a stated {float(perf.capital):,.0f} USDT. Most of what the desk did was decide "
        f"not to trade, and those decisions are graded against the moves that followed — ask "
        f"\"what is your track record\" for that grading."
        + (f" {voided} of those abstentions are rows first written as trades by a runner defect "
           f"fixed on 20 Sep 2026 — the risk layer had refused them — kept in the chain, voided, "
           f"and never counted as trades."
           if voided else ""),
        "How this is measured: holding QQQ (Bitget's QQQUSDT) from the first daily close on or "
        "after the desk's first decision to the latest close, with no fee charged to the holder, "
        "so the comparison is tilted toward the benchmark, not toward the desk.",
    ]
    sources = [Source("computation", "argus.paper.performance:evaluate_ledger",
                      f"{decisions} decisions, {trades} settled"),
               Source("venue", "bitget /api/v2/mix/market/candles", f"{BENCHMARK} daily closes")]
    return lines, sources, {"desk_return": desk, "trades": trades, **bench}


EDGE_ON_NAME_Q = re.compile(
    r"\b(?:your|the\s+desk'?s?|argus'?s?|its)\s+(?:edge|record|track\s+record|results?|performance|"
    r"hit\s+rate)\s+(?:on|in|with|for|trading)\b", re.I)
"""The desk's record on one name: "What's your edge on NVDA specifically?" opened on a raw ledger
row (a judge's audit, 2026-09-30)."""


def name_answer(ledger: PaperLedger, symbol: str) -> tuple[list[str], list[Source], dict[str, Any]]:
    """What the desk's record shows on one name: decisions, trades, the settled result, and how it
    leaned when it declined — and whether any of it is enough to call an edge."""
    perf = evaluate_ledger(ledger)
    name = symbol.removesuffix("USDT")
    rows = [e for e in ledger.entries if e.symbol == symbol]
    if not rows:
        return ([f"Bottom line: the desk has made no decision on {name}, so it has no record there "
                 f"to measure an edge by — it decides twelve stock perpetuals, and research "
                 f"questions about {name} are answered all the same."], [], {"decisions": 0})
    mine = next((c for c in perf.by_symbol if c.symbol == symbol), None)
    trades = mine.trades if mine is not None else 0
    net = float(mine.net_pnl) if mine is not None else 0.0
    leans = {k: sum(1 for e in rows if e.lean == k) for k in ("up", "down", "none")}
    first = rows[0].decided_at[:10]
    result = (f"{trades} settled trade{'' if trades == 1 else 's'}, net {net:+,.2f} USDT"
              if trades else "no settled trade")
    verdict = (f"the desk shows no measured edge on {name}" if trades < SHOWS_AN_EDGE_AT
               else f"the desk's record on {name}")
    lead = (f"Bottom line: {verdict} — {len(rows)} decisions since {first}, {result}"
            + (f"; under {SHOWS_AN_EDGE_AT} trades one outcome cannot be told from luck."
               if trades < SHOWS_AN_EDGE_AT else "."))
    lines = [
        lead,
        f"When it declined, it leaned up {leans['up']} times, down {leans['down']} and had no lean "
        f"{leans['none']} times; those leans are graded against the moves that followed — ask "
        f"\"what is your track record\".",
        f"Ask \"why did you skip {name}\" for the reasons on the latest decision.",
    ]
    sources = [Source("computation", "argus.paper.performance:evaluate_ledger",
                      f"{name}: {len(rows)} decisions, {trades} settled")]
    return lines, sources, {"decisions": len(rows), "trades": trades, "net": net}


trace_module(globals())
