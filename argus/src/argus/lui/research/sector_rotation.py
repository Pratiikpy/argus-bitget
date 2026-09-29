"""Which sectors money is rotating into: each Select Sector SPDR fund's return against the S&P 500.

"which sectors are money rotating into this month" was answered with the desk's own Sharpe and
P&L, then with Treasury yields (a judge's audit, 2026-09-29): the console read the question and
had no engine for it. This is that engine.

**What it measures, said on every answer.** Rotation here is relative price strength: each of the
eleven sector funds (the same map `lui/research/sector.py` uses for valuation) over the last 21 and
63 trading days, minus SPY over the same days. It is not fund flows, which no keyless source
publishes by sector, and the answer says so rather than calling price "money".

**Why both windows.** A sector ahead over one month and behind over three is a bounce, not a
rotation; one ahead on both is the trend a trader means. The lead names the leaders on the month
and says which of them also lead on the quarter.

Prices are Yahoo Finance's split-adjusted daily closes (`market/equity_history.daily`), the same
series the weekend and gap engines read.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from argus.lui.answer import Source
from argus.lui.research.sector import SECTOR_FUND
from argus.lui.trace import trace_module

MONTH, QUARTER = 21, 63
"""Trading days in a month and a quarter."""

BENCHMARK = "SPY"

SECTOR_ROTATION_Q = re.compile(
    r"\bsectors?\b[^?.]{0,60}\b(?:rotat\w*|money|flow\w*|leading|lagging|lead|lag|outperform\w*|"
    r"underperform\w*|strongest|weakest|hot|cold|in\s+favou?r|out\s+of\s+favou?r|working)\b|"
    r"\b(?:rotat\w*|money|flow\w*|leading|lagging|strongest|weakest|best|worst)\b[^?.]{0,40}"
    r"\bsectors?\b|\bsector\s+(?:rotation|performance|strength|leaders?|laggards?)\b|"
    r"板块轮动|行业轮动|资金流向[^?\uff1f]{0,10}(?:板块|行业)|哪个(?:板块|行业)",
    re.I,
)
"""A question about which sectors lead or lag. Needs "sector" and a word about leading, lagging or
money moving, so "what sector is NVDA in" does not reach it."""


_SECTOR_WORDS: tuple[tuple[str, str], ...] = (
    (r"tech\w*|semis?|semiconductors?", "Technology"),
    (r"energy|oil\s+stocks", "Energy"),
    (r"financials?|banks?", "Financial Services"),
    (r"health\s*care|pharma\w*|biotech", "Healthcare"),
    (r"utilities", "Utilities"),
    (r"industrials?", "Industrials"),
    (r"(?:basic\s+)?materials", "Basic Materials"),
    (r"real\s+estate|reits?", "Real Estate"),
    (r"(?:consumer\s+)?staples|consumer\s+defensive", "Consumer Defensive"),
    (r"(?:consumer\s+)?discretionary|consumer\s+cyclical|retail", "Consumer Cyclical"),
    (r"communications?(?:\s+services)?|media", "Communication Services"),
)
"""Everyday names for the eleven sectors, in the map's own names."""

_NAMED_ROTATION = re.compile(
    r"\b(?:rotat\w*|money|capital|flow\w*|moving|shift\w*)\b[^?.]{0,50}"
    r"\b(?:out\s+of|into|from|to)\b", re.I)
"""Money said to move between sectors named in words: "is capital rotating out of tech and into
energy" (a judge's audit, 2026-09-29) never said "sector"."""


def named_sectors(text: str) -> list[str]:
    """The sectors a question names, in the order it names them."""
    found: list[tuple[int, str]] = []
    for pattern, sector in _SECTOR_WORDS:
        m = re.search(rf"\b(?:{pattern})\b", text, re.I)
        if m is not None and sector not in (name for _, name in found):
            found.append((m.start(), sector))
    return [name for _, name in sorted(found)]


def asks_for_sector_rotation(text: str) -> bool:
    if re.search(r"\bwhat\s+sector\s+(?:is|does)\b", text, re.I):
        return False
    return bool(SECTOR_ROTATION_Q.search(text)) or (
        bool(_NAMED_ROTATION.search(text)) and len(named_sectors(text)) >= 2)


def _returns(ticker: str, daily: Callable[[str], Any]) -> tuple[float, float, str] | None:
    days = daily(ticker)
    if len(days) <= QUARTER:
        return None
    last = days[-1]
    month = last.close / days[-1 - MONTH].close - 1.0
    quarter = last.close / days[-1 - QUARTER].close - 1.0
    return month, quarter, last.day.isoformat()


def answer(text: str, daily: Callable[[str], Any] | None = None) -> tuple[list[str], list[Source],
                                                                          dict[str, Any]]:
    """The eleven sector funds ranked against SPY over a month, with the quarter beside it."""
    if daily is None:
        from argus.market.equity_history import daily as daily_closes

        daily = daily_closes
    bench = _returns(BENCHMARK, daily)
    if bench is None:
        return (["Bottom line: SPY's daily history did not arrive, so no sector can be measured "
                 "against it just now. Try again shortly."], [], {})
    rows: list[dict[str, Any]] = []
    for sector, fund in SECTOR_FUND.items():
        try:
            got = _returns(fund, daily)
        except Exception:
            got = None
        if got is not None:
            rows.append({"sector": sector, "fund": fund, "month": got[0], "quarter": got[1],
                         "vs_spy_month": got[0] - bench[0], "vs_spy_quarter": got[1] - bench[1],
                         "as_of": got[2]})
    if len(rows) < 3:
        return (["Bottom line: too few sector funds' histories arrived to rank them just now. "
                 "Try again shortly."], [], {"sectors": rows})
    rows.sort(key=lambda r: float(r["vs_spy_month"]), reverse=True)
    # Only a sector ahead of SPY is a leader and only one behind it a laggard: a top-three that
    # included a fund 0.5 points behind the market was called "favoured" in the first draft.
    leaders = [r for r in rows if float(r["vs_spy_month"]) > 0][:3]
    laggards = [r for r in rows if float(r["vs_spy_month"]) < 0][::-1][:3]

    def points(value: float) -> str:
        return f"{value * 100:+.1f}"

    def named(group: list[dict[str, Any]]) -> str:
        return ", ".join(f"{r['sector']} ({points(float(r['month']))}%, "
                         f"{points(float(r['vs_spy_month']))} points against SPY)" for r in group)

    both = [str(r["sector"]) for r in leaders if float(r["vs_spy_quarter"]) > 0]
    if leaders:
        lead = (f"Bottom line: over the last month relative strength has favoured "
                f"{named(leaders[:2])}")
        lead += f", and left {named(laggards[:2])}; " if laggards else "; "
        lead += (f"{' and '.join(both)} {'also leads' if len(both) == 1 else 'also lead'} over "
                 f"three months, so that is a trend rather than a bounce." if both else
                 "none of the month's leaders also leads over three months, so read it as a "
                 "bounce more than a rotation.")
    else:
        lead = (f"Bottom line: no sector beat SPY over the last month; the least weak were "
                f"{named(rows[:2])}.")
    asked = [r for name in named_sectors(text) for r in rows if r["sector"] == name]
    general = lead
    if len(asked) >= 2:
        # The sectors the question named, compared first; the full ranking follows.
        a, b = asked[0], asked[1]
        month = float(b["vs_spy_month"]) - float(a["vs_spy_month"])
        quarter = float(b["vs_spy_quarter"]) - float(a["vs_spy_quarter"])
        verdict = ("yes, on both the month and the quarter" if month > 0 and quarter > 0 else
                   "not over the month, but yes over three months" if quarter > 0 else
                   "yes over the month, not over three months" if month > 0 else
                   "no, on both the month and the quarter")
        lead = (f"Bottom line: from {a['sector']} into {b['sector']} — {verdict}: "
                f"{a['sector']} {points(float(a['month']))}% and {b['sector']} "
                f"{points(float(b['month']))}% over the month (SPY {points(bench[0])}%), "
                f"{points(float(a['quarter']))}% and {points(float(b['quarter']))}% over three "
                f"months.")
    rest = general.removeprefix("Bottom line: ")
    lines_head = [lead] if lead == general else [lead, rest[:1].upper() + rest[1:]]
    lines = [*lines_head,
             f"SPY: {points(bench[0])}% over the month and {points(bench[1])}% over three months, "
             f"to {bench[2]}.",
             "Every sector, month then quarter, each against SPY: " + "; ".join(
                 f"{r['sector']} {points(float(r['vs_spy_month']))} / "
                 f"{points(float(r['vs_spy_quarter']))}"
                 for r in rows) + " (points).",
             "Measured: each Select Sector SPDR fund's split-adjusted price over the last 21 and "
             "63 trading days, minus SPY's. This is relative strength, not fund flows, which no "
             "keyless source publishes by sector.",
             "Data: Yahoo Finance daily closes. This is analysis, not advice — you make the call."]
    sources = [Source(kind="venue", ref="https://query1.finance.yahoo.com/v8/finance/chart",
                      detail="daily closes for SPY and the eleven Select Sector SPDR funds"),
               Source(kind="computation", ref="argus.lui.research.sector_rotation",
                      detail="21- and 63-day return minus SPY's, ranked")]
    return lines, sources, {"sectors": rows, "benchmark": {"month": bench[0], "quarter": bench[1]}}


trace_module(globals())
