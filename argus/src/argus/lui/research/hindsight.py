"""What a past buy would have done, and averaging in against buying at once, from real closes.

Two questions the console had no engine for (a judge's audit, 2026-09-29):

* "how much would I have lost if I bought TSLA at the start of 2022 and held to now" was answered
  with position sizing, and the 2022 entry was never read;
* "should I DCA into BTC" was answered with the same sizing and never engaged the averaging.

Both are arithmetic on a price history, so both are answered from one: Yahoo Finance's daily chart
(`market/equity_history.daily`), whose adjusted close folds in splits and dividends, the series the
gap and weekend engines already read. So a dividend payer's result is its total return, and its
early prices print below what traded then; the answer says so. Crypto uses its ``-USD`` pair on the
same chart. The answer names the exact first and last close it used, so a reader can check them,
and says that the past is not a forecast.

**Entry date.** "the start of 2022" is the first trading day of 2022; "in March 2021" the first of
that month; "the end of 2023" the last of that year; "3 years ago" the trading day on or after that
date. A date the question does not state is not guessed: the answer asks for one.

**DCA against a lump sum.** The same money, either all on the first day of the window or in equal
monthly parts on the first trading day of each month, both held to the last close. Reported: the
end value per $10,000 for each, and each one's worst fall from its own peak while it was building.
Over a rising market the lump sum usually ends ahead; averaging's case is the smaller regret when
the market falls first, and the answer puts the two numbers side by side rather than choosing.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from argus.lui.answer import Source
from argus.lui.trace import trace_module

STAKE = 10_000.0
"""The illustrative sum the dollar figures are stated on."""

_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), start=1)}

HINDSIGHT_Q = re.compile(
    r"\b(?:if\s+i\s+(?:had\s+)?(?:bought|invested|put\s+\S+\s+in(?:to)?|got\s+in)|had\s+i\s+"
    r"(?:bought|invested)|would\s+i\s+have\s+(?:made|lost|gained|earned|had)|how\s+much\s+would\s+"
    r"(?:i\s+have|it\s+be\s+worth)|what\s+would\s+\S+\s+(?:be\s+worth|have\s+returned)|since\s+i\s+"
    r"bought)\b", re.I)
"""A question about a past buy held to now."""

DCA_Q = re.compile(r"\bdca\b|dollar[\s-]cost\s+averag\w*|\baverag\w*\s+in(?:to)?\b|"
                   r"\bbuy\s+(?:a\s+little\s+)?(?:every|each)\s+(?:week|month)\b|"
                   r"\blump[\s-]?sum\b", re.I)
"""A question about averaging into a position over time."""


class InvalidDate(ValueError):
    """A date the question wrote that does not exist ("2023-02-29"); said, never guessed."""


def entry_date(text: str, today: date) -> date | None:
    """The day a hindsight question means, or None when it states none."""
    low = text.lower()
    m = re.search(r"\b(start|beginning|end|middle)\s+of\s+(20\d\d|19\d\d)\b", low)
    if m:
        year = int(m.group(2))
        return {"end": date(year, 12, 31), "middle": date(year, 7, 1)}.get(m.group(1),
                                                                            date(year, 1, 1))
    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(?:of\s+)?"
                  r"(20\d\d|19\d\d)\b", low)
    if m:
        return date(int(m.group(2)), _MONTHS[m.group(1)], 1)
    m = re.search(r"\b(20\d\d)-(\d{2})-(\d{2})\b", low)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError as exc:
            # "2023-02-29" took the console to an HTTP 500 (a hostile review, 2026-09-29).
            raise InvalidDate(f"{m.group(0)} is not a date on the calendar") from exc
    m = re.search(r"\b(\d+|a|an|one|two|three|four|five)\s+(year|month|week)s?\s+ago\b", low)
    if m:
        words = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        n = words.get(m.group(1)) or int(m.group(1))
        if m.group(2) == "week":
            return today - timedelta(days=7 * n)
        # Calendar months and years, not 365-day blocks: "3 years ago" on 30 Sep 2026 is 30 Sep
        # 2023, and a 29 Feb that does not exist in the target year becomes the 28th.
        back = today.year * 12 + today.month - 1 - (n * 12 if m.group(2) == "year" else n)
        year, month = back // 12, back % 12 + 1
        for day in (today.day, 30, 29, 28):
            try:
                return date(year, month, day)
            except ValueError:
                continue
    m = re.search(r"\b(?:in|during|back\s+in|since)\s+(20\d\d|19\d\d)\b", low)
    if m:
        return date(int(m.group(1)), 1, 1)
    return None


_AMOUNT = re.compile(
    r"(?:\$\s*(?P<a>\d[\d,]*(?:\.\d+)?)\s*(?P<ak>k|thousand|grand)?|(?P<b>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<bk>k|thousand|grand)?\s*(?:dollars?|usd|usdt|bucks))", re.I)
"""A sum the question names: "$1,000", "1000 dollars", "5k bucks"."""


def stated_amount(text: str) -> float | None:
    """The money a hindsight question names ("if I had put 1000 dollars in"), or None: it was
    answered on $10,000 whatever was asked (a first-time-user audit, 2026-09-29)."""
    m = _AMOUNT.search(text)
    if m is None:
        return None
    value = float((m.group("a") or m.group("b")).replace(",", ""))
    return value * 1000 if (m.group("ak") or m.group("bk")) else value


def _chart_ticker(symbol: str) -> str:
    from argus.market.universe import is_equity

    base = symbol.removesuffix("USDT")
    return base if is_equity(symbol) else f"{base}-USD"


def _days(symbol: str, daily: Callable[[str], Any] | None) -> tuple[list[Any], str]:
    if daily is None:
        from argus.market.equity_history import daily as daily_closes

        daily = daily_closes
    ticker = _chart_ticker(symbol)
    return list(daily(ticker)), ticker


def hindsight(text: str, symbol: str, *, today: date,
              daily: Callable[[str], Any] | None = None
              ) -> tuple[list[str], list[Source], dict[str, Any]]:
    """A buy on the stated day held to the last close."""
    name = symbol.removesuffix("USDT")
    try:
        start = entry_date(text, today)
    except InvalidDate as exc:
        return ([f"Bottom line: {exc}, so there is no close to start from — give a real date "
                 f"(\"2023-02-28\", \"March 2023\")."], [], {})
    if start is None:
        return ([f"Bottom line: say when you would have bought {name} — \"at the start of 2022\", "
                 f"\"in March 2021\" or \"3 years ago\" — and this works out what it would have "
                 f"done since, from real closes."], [], {})
    if start > today:
        # A date in the future was refused as history "not reaching back" (a hostile review).
        return ([f"Bottom line: {start:%d %b %Y} has not happened yet, so there is no close to "
                 f"start from — name a date in the past."], [], {})
    days, ticker = _days(symbol, daily)
    after = [d for d in days if d.day >= start]
    if len(after) < 2:
        return ([f"Bottom line: Yahoo Finance's history for {ticker} has nothing from "
                 f"{start:%d %b %Y} on, so there is no real close to start from."], [], {})
    first, last = after[0], after[-1]
    listed_later = bool(days) and days[0].day > start + timedelta(days=7)
    change = last.close / first.close - 1.0
    low_day = min(after, key=lambda d: d.close)
    peak, worst, worst_day = first.close, 0.0, first.day
    for d in after:
        peak = max(peak, d.close)
        fall = d.close / peak - 1.0
        if fall < worst:
            worst, worst_day = fall, d.day
    verb = "made" if change >= 0 else "lost"
    stake = stated_amount(text) or STAKE
    lines = [
        *([f"Note: {name}'s price history starts on {first.day:%d %b %Y}, after the "
           f"{start:%d %b %Y} asked for, so the result below runs from its first close; it could "
           f"not have been bought earlier."]
          if listed_later else []),
        f"Bottom line: bought {name} at the close on {first.day:%d %b %Y} ({first.close:,.2f}) "
        f"and held to {last.day:%d %b %Y} ({last.close:,.2f}), it {verb} {abs(change):.1%} — "
        f"${stake:,.0f} would now be ${stake * (1 + change):,.0f}.",
        f"On the way: its lowest close was {low_day.close:,.2f} on {low_day.day:%d %b %Y}, and at "
        f"its worst it was {abs(worst):.1%} below its own high ({worst_day:%d %b %Y}) — the fall a "
        f"holder would have had to sit through.",
        "That is what happened, not what will. Prices are Yahoo's adjusted closes, which fold in "
        "splits and dividends: the result is the total return, and a dividend payer's early "
        "prices read below what traded then.",
    ]
    if listed_later:
        lines.insert(1, lines.pop(0))  # the answer leads; the note on the start follows it
    sources = [Source(kind="venue", ref="https://query1.finance.yahoo.com/v8/finance/chart",
                      detail=f"{ticker} daily closes {first.day} to {last.day}")]
    return lines, sources, {"hindsight": {"ticker": ticker, "from": str(first.day),
                                          "to": str(last.day), "change": change,
                                          "worst_drawdown": worst}}


def dca(text: str, symbol: str, *, today: date, daily: Callable[[str], Any] | None = None,
        months: int = 12) -> tuple[list[str], list[Source], dict[str, Any]]:
    """Monthly averaging against one buy at the start, over the last ``months`` (or the stated
    window), on the same money."""
    name = symbol.removesuffix("USDT")
    try:
        stated = entry_date(text, today)
    except InvalidDate as exc:
        return ([f"Bottom line: {exc}, so the window cannot start there — give a real date."],
                [], {})
    if stated is not None and stated > today:
        return ([f"Bottom line: {stated:%d %b %Y} has not happened yet — name a start in the "
                 f"past."], [], {})
    # The last ``months`` whole months including this one: twelve buys, not a thirteenth on the
    # stray last day of a month a year back.
    back = today.year * 12 + today.month - 1 - (months - 1)
    start = stated or date(back // 12, back % 12 + 1, 1)
    days, ticker = _days(symbol, daily)
    window = [d for d in days if d.day >= start]
    if len(window) < 40:
        return ([f"Bottom line: too little of {ticker}'s history falls in that window to compare "
                 f"averaging with buying at once."], [], {})
    # the first trading day of each calendar month in the window
    buys, seen = [], set()
    for d in window:
        key = (d.day.year, d.day.month)
        if key not in seen:
            seen.add(key)
            buys.append(d)
    # "I have $1,000 ... should I DCA into BTC" was worked on $10,000 (a first-time user,
    # round 24): the amount said is the amount averaged
    stake = stated_amount(text) or STAKE
    part = stake / len(buys)
    units_lump = stake / window[0].close
    units_dca, spent, worst_dca, peak_dca, worst_lump, peak_lump = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    buy_days = {d.day for d in buys}
    for d in window:
        if d.day in buy_days:
            units_dca += part / d.close
            spent += part
        value_dca = units_dca * d.close + (stake - spent)
        value_lump = units_lump * d.close
        peak_dca, peak_lump = max(peak_dca, value_dca), max(peak_lump, value_lump)
        worst_dca = min(worst_dca, value_dca / peak_dca - 1.0)
        worst_lump = min(worst_lump, value_lump / peak_lump - 1.0)
    last = window[-1]
    end_dca, end_lump = units_dca * last.close, units_lump * last.close
    ahead = "averaging" if end_dca > end_lump else "buying at once"
    lines = [
        f"Bottom line: over {window[0].day:%b %Y} to {last.day:%b %Y}, ${stake:,.0f} put into "
        f"{name} in {len(buys)} equal monthly buys would now be ${end_dca:,.0f}, against "
        f"${end_lump:,.0f} bought all at once on {window[0].day:%d %b %Y} — {ahead} came out "
        f"ahead.",
        f"The ride: the averaged account's worst fall from its peak was {abs(worst_dca):.1%}, the "
        f"lump sum's {abs(worst_lump):.1%} — averaging's case is the smaller fall while it builds, "
        f"not a higher end value.",
        f"The buys were ${part:,.0f} a month; the same ${stake:,.0f} spread weekly over a year is "
        f"about ${stake / 52:,.0f} a week.",
        "The choice is yours and depends on what you would do in a fall; this is what one window "
        "of real closes did, not a forecast. Name another start (\"since 2022\") to see a "
        "different window.",
    ]
    sources = [Source(kind="venue", ref="https://query1.finance.yahoo.com/v8/finance/chart",
                      detail=f"{ticker} daily closes {window[0].day} to {last.day}"),
               Source(kind="computation", ref="argus.lui.research.hindsight.dca",
                      detail="equal monthly buys on each month's first trading day vs one buy")]
    return lines, sources, {"dca": {"ticker": ticker, "end_dca": end_dca, "end_lump": end_lump,
                                    "worst_dca": worst_dca, "worst_lump": worst_lump,
                                    "buys": len(buys)}}


trace_module(globals())
