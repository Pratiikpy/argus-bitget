"""A market's close on a named day.

"What was the closing price of BTC today, Sunday 2026-10-04, on the NYSE?" got Bitget's live
quote with "NYSE is not a ticker" under Left out, and "what did AAPL close at on the NYSE on 2
October?" a refusal (round 44 hostile, M16). A close on a day is a fixed number, so it is read
from the record of that day:

- a coin closes at the end of the UTC day on Bitget's daily candle; it is listed on no stock
  exchange, and saying it is beats pretending it has an NYSE close;
- a US stock closes at the 16:00 New York close, read from Yahoo Finance's daily bars; on a
  weekend or an exchange holiday there was no session, so the answer is the last close before it;
- a day still under way has no close yet, so the answer says what the last price is instead.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from typing import Final

from argus.lui.trace import trace_module

ASKED: Final = re.compile(
    r"\bclos(?:e|ed|ing)(?:\s+price)?\b|\bsettle(?:d|ment)?(?:\s+price)?\b|"
    r"\bend[\s-]of[\s-]day\s+price\b", re.I)
_PRICE_ON: Final = re.compile(r"\bprice\b[^?]{0,40}\bon\s+(?:\w+\s+){0,2}\d|\bprice\s+(?:was|"
                              r"on)\b", re.I)
_EXCHANGE: Final = re.compile(r"\b(NYSE|Nasdaq|NASDAQ|LSE|London\s+Stock\s+Exchange|"
                              r"New\s+York\s+Stock\s+Exchange|stock\s+exchange)\b", re.I)
_WEEKDAYS: Final = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_WEEKDAY: Final = re.compile(r"\b(?:on\s+)?(?P<last>last\s+)?(?P<d>monday|tuesday|wednesday|"
                             r"thursday|friday|saturday|sunday)\b", re.I)
_NOT_A_CLOSE: Final = re.compile(
    # "close a position", "close above", "close to", "how close" are not a close price
    r"\bclos(?:e|ing)\s+(?:my|the|a|half|out|it|this|that|all|position|trade|above|below|to|"
    r"under|over|beyond)\b|\bhow\s+close\b|\bclose\s+(?:by|at)\s+\d|\bshould\s+i\s+close\b|"
    r"\b(?:52|fifty[\s-]two)[\s-]*week\b|\bclosed\s+(?:on\s+)?(?:weekends?|holidays?)\b|"
    r"\b(?:when|what\s+time|what\s+hour)\s+(?:does|do|is|will)\b[^?]{0,30}\bclose\b|\b(?:is|are)\s+(?:the\s+)?(?:market|nyse|"
    r"nasdaq)\s+(?:open|closed)\b", re.I)


def _asked_day(text: str, today: date) -> tuple[date, list[str]] | None:
    """The day asked about and anything to say about how it was read, or None."""
    from argus.lui.journal import find_date

    notes: list[str] = []
    day = find_date(text, today)
    said = _WEEKDAY.search(text)
    if day is not None:
        if said is not None and _WEEKDAYS[day.weekday()] != said.group("d").lower():
            notes.append(f"{day:%d %b %Y} is a {day:%A}, not a {said.group('d').title()} as "
                         f"written; the date was used")
        if re.search(r"\btoday\b", text, re.I) and day != today:
            notes.append(f"{day:%d %b %Y} is not today in UTC (today is {today:%d %b %Y}); the "
                         f"date was used")
        return day, notes
    if re.search(r"\byesterday\b", text, re.I):
        return today - timedelta(days=1), notes
    if said is not None:
        back = (today.weekday() - _WEEKDAYS.index(said.group("d").lower())) % 7 or 7
        return today - timedelta(days=back), notes
    if re.search(r"\btoday\b|\btonight\b", text, re.I):
        return today, notes
    return None


_RANGE_ASKED: Final = re.compile(r"\b(?:high|low|highs|lows|range|open(?:ed|ing)?)\b", re.I)
"""A day's high, low or open, asked of a named day: "What were BTC's high and low on 2026-10-03
UTC?" got the live ticker with the date replaced by now (round 45 hostile, M8)."""


def asked(text: str) -> bool:
    from argus.lui.journal import find_date

    ranged = (_RANGE_ASKED.search(text) is not None
              and find_date(text, datetime.now(UTC).date()) is not None
              and not re.search(r"\b(?:52|fifty)[-\s]*week|\ball[-\s]*time|\b\d+[-\s]*day\b",
                                text, re.I))
    # "What was the S&P 500 yesterday" names a market and a past day and nothing else
    # (round 45 hostile, M17)
    what_was = re.search(r"\bwhat\s+was\s+(?:the\s+)?[A-Za-z&\d .-]{2,24}?\s+(?:at\s+)?"
                         r"(?:yesterday|on\s+(?:mon|tue|wed|thu|fri|sat|sun)\w*|on\s+\d)", text,
                         re.I)
    return ((ASKED.search(text) is not None or _PRICE_ON.search(text) is not None or ranged
             or what_was is not None)
            and _NOT_A_CLOSE.search(text) is None)


def close_lines(text: str, prior: list[str], *, now: datetime | None = None
                ) -> list[str] | None:
    """The close of the first named market on the day the question names, or None."""
    if not asked(text):
        return None
    now = now or datetime.now(UTC)
    today = now.date()
    found = _asked_day(text, today)
    if found is None:
        return None
    day, notes = found
    if day > today:
        return [f"Bottom line: {day:%d %b %Y} has not happened yet, so there is no close to "
                f"read; ask for the price now or for a past day."]
    from argus.lui.research.parse import is_us_equity, research_symbols

    named = list(research_symbols(text)[0])
    if not named:
        return None
    symbol = named[0]
    exchange = _EXCHANGE.search(text)
    from argus.lui.research.rule_test import YAHOO_SERIES

    index = YAHOO_SERIES.get(symbol)
    if is_us_equity(symbol) or (index is not None and index[0].startswith("^")):
        # an index closes when its exchange closes: "What was the S&P 500 yesterday?" is read
        # from Yahoo's ^GSPC, like a stock (round 45 hostile, M17)
        lines = _equity_close(symbol, day, today, now,
                              ticker=index[0] if index is not None else None,
                              record=_RECORD.search(text) is not None)
    else:
        lines = _coin_close(symbol, day, today,
                            wants_range=_RANGE_ASKED.search(text) is not None
                            and ASKED.search(text) is None)
        if lines is not None and exchange is not None:
            name = symbol.removesuffix("USDT")
            lines.insert(1, f"{name} is not listed on the {exchange.group(1)} or any stock "
                            f"exchange: it trades around the clock, so its daily close is the "
                            f"end of the UTC day, read here from Bitget.")
    if lines is None:
        return None
    if notes:
        lines.insert(1, f"Read as: {'; '.join(notes)}.")
    if re.search(r"\bfed\b|\bfederal\s+reserve\b", text, re.I):
        from argus.lui.research.macro import fed_premise_line

        try:
            fed_said = fed_premise_line(text)
        except Exception:
            fed_said = None
        if fed_said is not None:
            # "...was it a record given that the Fed cut rates to negative 2% last month?": the
            # Fed premise is checked beside the close (round 45 hostile, M17)
            lines.insert(1, fed_said[0])
    if re.search(r"\btomorrow'?s?\b|\bnext\s+(?:week|session|day)'?s?\b", text, re.I):
        # "What will BTC have closed at yesterday, tomorrow's close?" mixes a past close with a
        # future one (round 44 hostile, minor 9): the past one is a fact, the future one is not
        lines.insert(1, "Tomorrow's close does not exist yet, and no figure here forecasts it; "
                        "the close above is the past day's, which is on the record.")
    return lines


def _coin_close(symbol: str, day: date, today: date, *, wants_range: bool = False
                ) -> list[str] | None:
    from argus.lui.research.parse import last_price
    from argus.lui.research.performance import price_text
    from argus.market import history

    name = symbol.removesuffix("USDT")
    if day == today:
        try:
            last = float(last_price(symbol) or 0)
        except Exception:
            return None
        if last <= 0:
            return None
        return [f"Bottom line: {name}'s UTC day {day:%d %b %Y} has not closed yet — it closes at "
                f"24:00 UTC; the last price on Bitget is {price_text(last)}.",
                "Data: Bitget's live ticker; ask again after midnight UTC for the day's close."]
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    try:
        candles = history.fetch_window(symbol, start=start, end=start + timedelta(days=1),
                                       interval="1Dutc")
    except Exception:
        return None
    candle = next((c for c in candles if c.ts.date() == day), None)
    if candle is None or float(candle.close) <= 0:
        return None
    close, open_ = float(candle.close), float(candle.open)
    high, low = float(candle.high), float(candle.low)
    if wants_range:
        return [f"Bottom line: on the UTC day {day:%a %d %b %Y}, {name} traded between "
                f"{price_text(low)} and {price_text(high)} on Bitget's perpetual "
                f"({high / low - 1:.2%} from low to high); it opened at {price_text(open_)} and "
                f"closed at {price_text(close)}.",
                "Data: Bitget USDT-futures daily candle for that UTC day (00:00 to 24:00 UTC); "
                "Bitget's own 1D candle starts at 16:00 UTC (midnight UTC+8), so its high and "
                "low can differ."]
    return [f"Bottom line: {name} closed the UTC day {day:%a %d %b %Y} at {price_text(close)} "
            f"on Bitget, {close / open_ - 1:+.2%} from its open of {price_text(open_)}.",
            f"That day's range: low {price_text(float(candle.low))}, high "
            f"{price_text(float(candle.high))}.",
            "Data: Bitget USDT-futures daily candle for that UTC day (00:00 to 24:00 UTC)."]


_RECORD: Final = re.compile(r"\brecord\b|\ball[-\s]*time\s+high\b|\bath\b|\bnew\s+high\b", re.I)


def _equity_close(symbol: str, day: date, today: date, now: datetime, *,
                  ticker: str | None = None, record: bool = False) -> list[str] | None:
    from argus.lui.research.performance import price_text
    from argus.market.equity_history import daily
    from argus.market.rtoken_spot import close_utc, holidays, is_trading_day

    shown = {"^GSPC": "the S&P 500", "^NDX": "the Nasdaq-100"}.get(ticker or "", "")
    ticker = ticker or symbol.removesuffix("USDT").removesuffix("STOCK")
    try:
        bars = daily(ticker)
    except Exception:
        return None
    lines: list[str] = []
    session = day
    if not is_trading_day(day):
        why = ("a weekend" if day.weekday() >= 5 else
               "an exchange holiday" if day in holidays(day.year) else "not a session")
        before = next((b for b in reversed(bars) if b.day < day), None)
        if before is None:
            return None
        return [f"Bottom line: US stock exchanges did not trade on {day:%a %d %b %Y} ({why}), "
                f"so {ticker} has no close that day; its last close before it was "
                f"{price_text(before.close)} on {before.day:%a %d %b}.",
                "Data: Yahoo Finance daily bars, the official 16:00 New York close."]
    if day == today and now < close_utc(day):
        from argus.lui.research.parse import last_price

        try:
            last = float(last_price(symbol) or 0)
        except Exception:
            last = 0.0
        before = next((b for b in reversed(bars) if b.day < day), None)
        lines = [f"Bottom line: {ticker} has not closed yet today — the New York session closes "
                 f"at 16:00 ET ({close_utc(day):%H:%M} UTC)"
                 + (f"; Bitget's last price is {price_text(last)}" if last > 0 else "")
                 + (f", and the previous close was {price_text(before.close)} on "
                    f"{before.day:%a %d %b}." if before else ".")]
        return lines
    bar = next((b for b in bars if b.day == session), None)
    if bar is None:
        return None
    before = next((b for b in reversed(bars) if b.day < session), None)
    change = (f", {bar.close / before.close - 1:+.2%} on the previous close of "
              f"{price_text(before.close)}") if before else ""
    label = shown or ticker
    out = [f"Bottom line: {label} closed at {price_text(bar.close)} on {day:%a %d %b %Y}"
           f"{change}.",
           "Data: Yahoo Finance daily bars, the official 16:00 New York close (split-adjusted)."]
    if record:
        before_day = [b.close for b in bars if b.day < session]
        top = max(before_day) if before_day else None
        if top is not None:
            out.insert(1, (f"A record: it is above every earlier close in the data (the previous "
                           f"high close was {price_text(top)})." if bar.close > top else
                           f"Not a record: the highest close before it was {price_text(top)}, "
                           f"{bar.close / top - 1:+.2%} away.")
                       + f" The data starts {bars[0].day:%d %b %Y}.")
    return out


__all__ = ["ASKED", "asked", "close_lines"]

trace_module(globals())
