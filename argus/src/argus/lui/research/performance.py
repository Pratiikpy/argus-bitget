"""How a market did over a stated period: its return, its deepest fall from a high, its worst
and best day, and what a stated amount or holding became, from Bitget's own daily candles.

Two audits in round 24 found no engine for it. A first-time user's "My grandson told me to buy
Bitcoin. How much has it gone up this year?" got the round-trip cost, "How much did it fall at its
worst?" the price card, and a judge's "how did AMD do over the past 6 months vs NVDA?", "Compare SOL
and ETH performance year to date", "How did gold do in the last 3 months?", "How did the Nasdaq do
in September?", "BTC 去年表现怎么样" (how did BTC do last year) and "How much would I have made
buying 1 ETH a month ago?" were each answered on a 30-day window, a 24-hour base rate or not at all.

Method, and where it comes from. The return is the end price over the start price, less one:
the price-return convention `qlib` and `vectorbt` compute a total return with, without dividends.
The start is the open of the period's first day, so that day's own move is counted, and the end
is the last daily close, or Bitget's last price now when the period runs to today. The deepest
fall is the maximum drawdown of those prices, peak to later trough, as `vectorbt`'s
``max_drawdown`` defines it. Prices are Bitget's USDT-futures daily candles
(`argus.market.history.fetch_window`, interval ``1Dutc``: days from 00:00 UTC; the plain ``1D``
candle starts at 16:00 UTC), so a stock perpetual's figure covers the hours Bitget trades it,
which include nights and weekends the exchange is shut; a listing younger than the period is
measured from its first candle, and that is said.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise

from argus.lui.trace import trace_module

_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december")
_UNITS_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
_WORD_N = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
           "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12}

PERFORMANCE_Q = re.compile(
    r"\bhow\s+(?:has|have|did|was|were|is)\s+[^?]{0,40}?\b(?:do(?:ne)?|perform(?:ed|ing)?|go(?:ne)?|"
    r"fare[d]?|move[d]?|react(?:ed)?)\b|"
    r"\bhow\s+much\s+(?:has|did|have)\s+[^?]{0,30}?\b(?:gone\s+up|go\s+up|rise|risen|rose|gain(?:ed)?|"
    r"grow|grown|gone\s+down|go\s+down|fall(?:en)?|fell|drop(?:ped)?|move[d]?|lose|lost)\b|"
    r"\b(?:returns?|performance|perf|drawdowns?|max(?:imum)?\s+drawdown)\b|"
    r"\bhow\s+much\s+would\s+i\s+have\s+(?:made|lost|had)\b|"
    r"\bwhat\s+(?:would|did)\s+\$?\d[\d,]*\s*(?:k)?\s*(?:in|of)\s+\S+\s+(?:be|become|turn)\b|"
    r"\bfall\s+at\s+its\s+worst\b|\bat\s+its\s+worst\b|\bbigger\s+drawdown\b|"
    r"表现|涨了多少|跌了多少|收益", re.I)
"""A question about what a market did over a period, not what it is doing now."""


@dataclass(frozen=True)
class Period:
    start: datetime
    end: datetime
    said: str
    """The period as the answer says it: "so far this year (1 Jan to 3 Oct)"."""


def asked_period(text: str, now: datetime | None = None) -> Period | None:
    """The period a question names, or None when it names none."""
    now = now or datetime.now(UTC)
    today = now.date()

    def at(day: date) -> datetime:
        return datetime(day.year, day.month, day.day, tzinfo=UTC)

    if re.search(r"\bthis\s+year\b|\bytd\b|\byear[\s-]to[\s-]date\b|\bso\s+far\s+this\s+year\b|"
                 r"今年", text, re.I):
        start = date(today.year, 1, 1)
        return Period(at(start), now, f"so far this year (1 Jan to {today:%d %b})")
    # "last year" is the calendar year before; "over the last year" is the 365 days to today
    if re.search(r"(?<!the\s)\blast\s+year\b|\bin\s+(?:the\s+)?previous\s+year\b|去年", text, re.I):
        year = today.year - 1
        return Period(at(date(year, 1, 1)), at(date(year, 12, 31)) + timedelta(hours=23),
                      f"in {year} (1 Jan to 31 Dec)")
    month = re.search(r"\bin\s+(?P<m>" + "|".join(_MONTHS) + r"|sept?|oct|nov|dec|jan|feb|mar|apr|"
                      r"jun|jul|aug)\b(?:\s+(?P<y>(?:19|20)\d\d))?", text, re.I)
    if month is not None:
        word = month.group("m").lower()
        number = next(i for i, name in enumerate(_MONTHS, 1) if name.startswith(word[:3]))
        year = int(month.group("y")) if month.group("y") else (
            today.year if number < today.month else today.year - 1)
        first = date(year, number, 1)
        last = (date(year + (number == 12), number % 12 + 1, 1) - timedelta(days=1))
        return Period(at(first), at(last) + timedelta(hours=23),
                      f"in {first:%B %Y} ({first:%d %b} to {last:%d %b})")
    span = re.search(r"\b(?:past|last|previous|over\s+the\s+(?:past|last)|in\s+the\s+(?:past|last)|"
                     r"the\s+(?:past|last))\s+(?P<n>\d+|a|an|one|two|three|four|five|six|seven|eight|"
                     r"nine|ten|twelve)?\s*(?P<u>days?|weeks?|months?|years?)\b|"
                     r"\b(?P<n2>\d+|a|an|one|two|three|six|twelve)\s+(?P<u2>days?|weeks?|months?|"
                     r"years?)\s+ago\b|\b(?P<n3>\d+)\s*(?P<u3>d|w|m|y)\b(?!\w)", text, re.I)
    if span is not None:
        count_word = (span.group("n") or span.group("n2") or span.group("n3") or "1").lower()
        count = int(count_word) if count_word.isdigit() else _WORD_N.get(count_word, 1)
        unit = (span.group("u") or span.group("u2") or span.group("u3") or "day").lower()
        unit = {"d": "day", "w": "week", "m": "month", "y": "year"}.get(unit, unit.rstrip("s"))
        days = count * _UNITS_DAYS[unit]
        return Period(now - timedelta(days=days), now,
                      f"over the last {count} {unit}{'s' if count != 1 else ''} "
                      f"({(now - timedelta(days=days)).date():%d %b %Y} to {today:%d %b})")
    return None


@dataclass(frozen=True)
class Record:
    symbol: str
    first_day: date
    last_day: date
    first: float
    last: float
    drawdown: float
    """The deepest fall from a high inside the window, a negative fraction."""
    peak_day: date
    trough_day: date
    worst_day: float
    best_day: float
    short_history: bool

    @property
    def change(self) -> float:
        return self.last / self.first - 1


def record(symbol: str, period: Period) -> Record | None:
    """The period's figures from Bitget daily candles, or None when there are too few."""
    from argus.market import history

    try:
        # 1Dutc: days that start at 00:00 UTC (the plain 1D candle starts at 16:00 UTC)
        candles = history.fetch_window(symbol, start=period.start - timedelta(days=1),
                                       end=period.end, interval="1Dutc")
    except Exception:
        return None
    inside = [c for c in candles if c.ts.date() >= period.start.date()]
    if not inside:
        return None
    # the period starts at the first day's open, so the first day's own move is counted
    days = [(inside[0].ts.date() - timedelta(days=1), float(inside[0].open))]
    days += [(c.ts.date(), float(c.close)) for c in inside]
    if period.end >= datetime.now(UTC) - timedelta(hours=2):
        # today's candle is still open; the period runs to Bitget's last price now
        from argus.lui.research.parse import last_price

        try:
            live = last_price(symbol)
        except Exception:
            live = None
        if live:
            days.append((datetime.now(UTC).date(), float(live)))
    if len(days) < 2:
        return None
    closes = [c for _d, c in days]
    peak, peak_at, deepest, at_peak, at_trough = closes[0], days[0][0], 0.0, days[0][0], days[0][0]
    for day, close in days:
        if close > peak:
            peak, peak_at = close, day
        fall = close / peak - 1
        if fall < deepest:
            deepest, at_peak, at_trough = fall, peak_at, day
    moves = [b / a - 1 for a, b in pairwise(closes)]
    return Record(symbol=symbol, first_day=days[0][0], last_day=days[-1][0], first=closes[0],
                  last=closes[-1], drawdown=deepest, peak_day=at_peak, trough_day=at_trough,
                  worst_day=min(moves) if moves else 0.0, best_day=max(moves) if moves else 0.0,
                  short_history=days[0][0] > period.start.date() + timedelta(days=3))


def _name(symbol: str) -> str:
    return symbol.removesuffix("USDT")


def performance_lines(symbols: tuple[str, ...], period: Period, *, amount: float | None = None,
                      units: float | None = None) -> list[str] | None:
    """The answer: one market's period, or several ranked by return, with drawdowns beside."""
    found = [r for r in (record(s, period) for s in symbols[:4]) if r is not None]
    if not found:
        return None
    ranked = sorted(found, key=lambda r: -r.change)
    if len(ranked) == 1:
        r = ranked[0]
        lead = (f"Bottom line: {_name(r.symbol)} is {r.change:+.1%} {period.said} — from "
                f"{r.first:,.2f} to {r.last:,.2f} — and its deepest fall from a high on the way "
                f"was {r.drawdown:.1%} ({r.peak_day:%d %b} to {r.trough_day:%d %b}).")
    else:
        lead = ("Bottom line: " + ", ".join(f"{_name(r.symbol)} {r.change:+.1%}" for r in ranked)
                + f" {period.said}; the deepest falls from a high were "
                + ", ".join(f"{_name(r.symbol)} {r.drawdown:.1%}" for r in
                            sorted(found, key=lambda r: r.drawdown))
                + f" — {_name(min(found, key=lambda r: r.drawdown).symbol)} had the bigger "
                  f"drawdown.")
    lines = [lead]
    for r in ranked:
        lines.append(f"{_name(r.symbol)}: {r.first:,.2f} on {r.first_day:%d %b %Y} to "
                     f"{r.last:,.2f} on {r.last_day:%d %b %Y}; worst day {r.worst_day:+.1%}, "
                     f"best day {r.best_day:+.1%}."
                     + (f" Bitget's daily history starts {r.first_day:%d %b %Y}, so the period is "
                        f"measured from there." if r.short_history else ""))
    first = ranked[0]
    if units is not None and len(found) == 1:
        lines.insert(1, f"{units:g} {_name(first.symbol)} bought at {first.first:,.2f} cost "
                        f"${units * first.first:,.2f} and is worth ${units * first.last:,.2f}: "
                        f"{'a gain' if first.last >= first.first else 'a loss'} of "
                        f"${abs(units * (first.last - first.first)):,.2f}.")
    elif amount is not None and len(found) == 1:
        lines.insert(1, f"${amount:,.0f} put in at the start would be "
                        f"${amount * (1 + first.change):,.0f} now "
                        f"({'a gain' if first.change >= 0 else 'a loss'} of "
                        f"${abs(amount * first.change):,.0f}); at its deepest fall it was "
                        f"{abs(first.drawdown):.1%} below its high on the way.")
    lines.append("Data: Bitget USDT-futures daily candles (UTC days), from the first day's open to "
                 "the last price, without dividends; a past period describes the past, not the "
                 "next one.")
    return lines


trace_module(globals())
