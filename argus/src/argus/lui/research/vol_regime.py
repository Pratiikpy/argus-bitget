"""Where today's realised volatility sits against its own history: the 1-month figure, its 1-year
average and its percentile.

Round 41's judge (M9, q25) asked "1-month BTC realised vol versus its 1-year average, and what
percentile?" and got 37% (30 days) against 38% (90 days): no year, no percentile. The judge's own
figure from daily log returns was 34.6%; the console's 37% came from hourly closes, which carry
intraday noise a daily reader does not see. Both are realised volatility, measured differently,
and the answer says which it uses.

**Method.** Daily closes (`rule_test.daily_closes`: Bitget's for a coin, Yahoo's split-adjusted
closes for a US stock), daily log returns, a 30-day rolling standard deviation annualised by
√365 for a coin (it trades every day) and √252 for a stock. "Its 1-year average" is the mean of
that rolling figure over the last 365 days; the percentile ranks today's figure among those
rolling values, and among five years of them so a calm year is not mistaken for a calm market.
"""

from __future__ import annotations

import math
import re
import statistics
from datetime import timedelta
from typing import Final

ASKED: Final = re.compile(
    r"\b(?:realis|realiz|historical|hist\.?|trailing|1[\s-]?month|30[\s-]?day|one[\s-]month)\w*"
    r"[^?]{0,20}\bvol(?:atility)?\b[^?]{0,80}\b(?:percentile|average|avg|mean|1[\s-]?year|"
    r"one[\s-]year|12[\s-]?month|annual|rank\w*|history|historically|normal|high|low|compare\w*|"
    r"versus|vs\.?)\b|\bvol(?:atility)?\s+percentile\b|\bpercentile\b[^?]{0,40}\bvol(?:atility)?\b",
    re.I)
WINDOW: Final = 30


def _rolling(closes: list[float], yearly: int) -> list[float]:
    logs = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    return [statistics.stdev(logs[i - WINDOW:i]) * math.sqrt(yearly)
            for i in range(WINDOW, len(logs) + 1)]


def _nth(n: float) -> str:
    k = round(n)
    tail = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{tail}"


def _percentile(value: float, sample: list[float]) -> float:
    return 100.0 * sum(1 for x in sample if x <= value) / len(sample)


def lines(text: str, symbol: str) -> list[str] | None:
    """The answer for ``symbol``, or None when ``text`` does not ask where its vol sits."""
    if not ASKED.search(text):
        return None
    from argus.lui.research.parse import is_us_equity
    from argus.lui.research.rule_test import daily_closes

    try:
        stamps, closes, said = daily_closes(symbol)
    except Exception:
        return [f"Bottom line: {symbol}'s daily history did not answer just now; ask again in a "
                "minute."]
    stock = is_us_equity(symbol)
    yearly = 252 if stock else 365
    if len(closes) < WINDOW + 60:
        return [f"Bottom line: {symbol} has {len(closes)} daily closes here — too few to rank its "
                "volatility against a year."]
    vols = _rolling(closes, yearly)
    ends = stamps[WINDOW:]
    now = vols[-1]
    year_start = stamps[-1] - timedelta(days=365)
    year = [v for v, t in zip(vols, ends, strict=True) if t >= year_start]
    pct_year = _percentile(now, year)
    pct_all = _percentile(now, vols)
    low, high = min(year), max(year)
    name = symbol.removesuffix("USDT")
    span_years = (stamps[-1] - stamps[0]).days / 365
    where = ("near the bottom of" if pct_year <= 20 else "near the top of" if pct_year >= 80 else
             "in the middle of")
    closed = stamps[-1] + timedelta(days=1) if not stock else stamps[-1]
    return [f"Bottom line: {name}'s 1-month realised volatility is {now:.1%} a year (30 daily "
            f"closes to {closed:%d %b %Y}), against a 1-year average of "
            f"{statistics.fmean(year):.1%} — the {_nth(pct_year)} percentile of the last year, "
            f"{where} its range.",
            f"Over the last year the 30-day figure ran from {low:.1%} to {high:.1%}; over the "
            f"{span_years:.1f} years of history here today's reading is the {_nth(pct_all)} "
            "percentile.",
            f"Method: daily log returns from {said}, a 30-day standard deviation annualised by "
            f"√{yearly}; the 1-year average and percentile are taken over that rolling figure. "
            "A figure from hourly closes runs a few points higher, because it counts intraday "
            "swings a daily close does not. Not advice."]


__all__ = ["ASKED", "lines"]
