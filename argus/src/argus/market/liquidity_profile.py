"""When a contract is liquid: its traded value by hour of day, on weekdays against weekends.

"What's the best time of day to trade NVDA perps?" and "does liquidity dry up on weekends for stock
perps like NVDA?" were refused or answered with a one-order cost (answer audit, round 3). Both have
a measured answer in Bitget's own hourly candles: how much value trades in each UTC hour over the
last 30 days, and how a weekend hour compares with a weekday one. For a stock perpetual the US
session matters most, so the regular-hours share is stated beside the hourly profile.

Traded value is the hour's base volume times its close — a proxy for how much size the book
absorbed, not a depth reading; the answer says so. The median is used throughout because one news
hour would otherwise decide the "best" hour.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Profile:
    symbol: str
    hours_read: int
    by_hour: dict[int, float]
    """Median traded value (USDT) in each UTC hour, weekdays only."""
    weekday_median: float
    weekend_median: float
    us_session_share: float
    """Share of weekday traded value inside 13:30-20:00 UTC (the NYSE session, summer time)."""


def profile(symbol: str, bars: Sequence[tuple[datetime, float, float]]) -> Profile | None:
    """``bars`` are (open time UTC, base volume, close) per hour, oldest first."""
    if len(bars) < 24 * 7:
        return None
    weekday: dict[int, list[float]] = defaultdict(list)
    weekday_all: list[float] = []
    weekend_all: list[float] = []
    session = total = 0.0
    for ts, volume, close in bars:
        value = volume * close
        if ts.weekday() >= 5:
            weekend_all.append(value)
            continue
        weekday[ts.hour].append(value)
        weekday_all.append(value)
        total += value
        minute = ts.hour * 60 + ts.minute
        if 13 * 60 + 30 <= minute < 20 * 60:
            session += value
    if not weekday_all or not weekend_all:
        return None
    return Profile(
        symbol=symbol, hours_read=len(bars),
        by_hour={h: statistics.median(v) for h, v in sorted(weekday.items())},
        weekday_median=statistics.median(weekday_all),
        weekend_median=statistics.median(weekend_all),
        us_session_share=session / total if total else 0.0,
    )


def lines(p: Profile, name: str, equity: bool) -> list[str]:
    ranked = sorted(p.by_hour.items(), key=lambda kv: -kv[1])
    best = ranked[:3]
    worst = ranked[-3:]
    ratio = p.weekend_median / p.weekday_median if p.weekday_median else 0.0
    lead = (f"Actionable: {name} trades most in {', '.join(f'{h:02d}:00' for h, _ in best)} UTC "
            f"(a median ${best[0][1]:,.0f} an hour at the busiest) and least in "
            f"{', '.join(f'{h:02d}:00' for h, _ in worst)} UTC (${worst[-1][1]:,.0f}) — "
            f"a large order costs least when split across the busy hours.")
    out = [lead]
    out.append(f"Weekends: a median weekend hour carries {ratio:.0%} of a weekday hour's traded "
               f"value on {name} — "
               + ("liquidity does dry up; expect wider spreads and more impact for the same size."
                  if ratio < 0.6 else
                  "thinner, though not dramatically." if ratio < 0.9 else
                  "about the same as a weekday."))
    if equity:
        out.append(f"{p.us_session_share:.0%} of {name}'s weekday traded value falls inside the US "
                   f"session (13:30-20:00 UTC in summer) — outside it the perpetual trades without "
                   f"its stock's price to anchor it.")
    out.append(f"Measured on Bitget's hourly candles over {p.hours_read} hours, as volume times "
               f"close — how much traded, not how deep the book was; a live depth read is the "
               f"execution plan's.")
    return out
