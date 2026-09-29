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


def asks_for_sector_rotation(text: str) -> bool:
    return bool(SECTOR_ROTATION_Q.search(text)) and not re.search(
        r"\bwhat\s+sector\s+(?:is|does)\b", text, re.I)


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
    lines = [lead,
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
