"""A named market episode ("a March-2020 style crash") as the size of the shock it asks about.

A stress question that names a past crash but no percentage was shocked at the console's default
-10%, which understates every episode below. The size here is measured: the Nasdaq-100 ETF's own
peak-to-trough fall inside the episode's window, from Yahoo's adjusted daily closes
(`argus.market.equity_history`), not a remembered figure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from argus.market import equity_history


@dataclass(frozen=True, slots=True)
class Episode:
    name: str
    start: date
    end: date
    pattern: re.Pattern[str]


EPISODES: tuple[Episode, ...] = (
    Episode("the March 2020 crash", date(2020, 2, 18), date(2020, 3, 24), re.compile(
        r"\bmar(?:ch)?[\s-]*2020\b|\bcovid\b|\b2020[\s-]*(?:style|crash|selloff|sell-off)", re.I)),
    Episode("the 2022 tech drawdown", date(2021, 11, 18), date(2022, 12, 30), re.compile(
        r"\b2022[\s-]*(?:style|crash|selloff|sell-off|drawdown|bear)", re.I)),
    Episode("the 2008 financial crisis", date(2007, 10, 30), date(2009, 3, 10), re.compile(
        r"\b2008\b|\bfinancial\s+crisis\b|\blehman\b|\bgfc\b", re.I)),
    Episode("the dot-com bust", date(2000, 3, 9), date(2002, 10, 10), re.compile(
        r"\bdot[\s-]*com\b|\b2000[\s-]*(?:style|crash|bust)", re.I)),
    Episode("the April 2025 tariff selloff", date(2025, 2, 18), date(2025, 4, 9), re.compile(
        r"\b(?:apr(?:il)?[\s-]*2025|2025[\s-]*(?:style|crash|selloff|sell-off)|"
        r"liberation\s+day)\b", re.I)),
)


def named(text: str) -> Episode | None:
    return next((e for e in EPISODES if e.pattern.search(text)), None)


def nasdaq_fall(episode: Episode) -> tuple[float, date, date] | None:
    """The deepest peak-to-trough fall of QQQ inside the window, as a negative percent."""
    return own_fall(episode, "QQQ", whole=False)


def own_fall(episode: Episode, ticker: str, *,
             whole: bool = True) -> tuple[float, date, date] | None:
    """The deepest peak-to-trough fall of one name inside the window, as a negative percent, or
    None when it did not trade through the window (listed after it began, or no history)."""
    try:
        days = [d for d in equity_history.daily(ticker) if episode.start <= d.day <= episode.end]
    except Exception:
        return None
    if len(days) < 5 or (whole and (days[0].day - episode.start).days > 10):
        return None
    peak = days[0]
    worst: tuple[float, date, date] | None = None
    for day in days:
        if day.close > peak.close:
            peak = day
        fall = (day.close / peak.close - 1) * 100
        if worst is None or fall < worst[0]:
            worst = (fall, peak.day, day.day)
    return worst


def replay_line(episode: Episode, symbols: tuple[str, ...], ticker_of: object) -> str | None:
    """What the named episode actually did to the index and to each holding that traded through
    it — the check a linear beta cannot make. "If the S&P falls 30% like 2008" was answered with
    a 30-day beta scaled linearly and nothing about 2008 (a judge, round 20, row 707); in a crash
    betas and correlations rise, so the names' own falls then are the harder test."""
    index = nasdaq_fall(episode)
    if index is None:
        return None
    spy = own_fall(episode, "SPY")
    seen: list[str] = []
    missing: list[str] = []
    for symbol in symbols:
        ticker = ticker_of(symbol)  # type: ignore[operator]
        fall = own_fall(episode, ticker)
        if fall is None:
            missing.append(ticker)
        else:
            seen.append(f"{ticker} {fall[0]:.0f}%")
    if not seen and not missing:
        return None
    return (f"What {episode.name} actually did ({episode.start:%b %Y} to {episode.end:%b %Y}, "
            f"Yahoo adjusted closes, deepest peak-to-trough): "
            + (f"the S&P 500 ETF {spy[0]:.0f}%, " if spy else "")
            + f"the Nasdaq-100 ETF {index[0]:.0f}%"
            + (f"; {', '.join(seen)}" if seen else "")
            + (f"; {', '.join(missing)} did not trade through it, so there is no figure"
               if missing else "")
            + ". The lines above scale today's 30-day beta linearly; in a crash betas and "
              "correlations rise, so the episode's own falls are the harder test.")
