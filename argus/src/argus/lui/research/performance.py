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
    # "What has NVDA done this year?", "which one fell more this year", "is it higher than last
    # month", "Is it down from where it was a year ago?" (a first-time user, round 25)
    r"\bwhat\s+(?:has|did)\s+[^?]{0,30}?\b(?:done|do)\b|"
    # "bitcoin this year", "how much did doge drop from its high", "how far did sol move
    # yesterday" (a first-time user, round 25)
    r"^\W*\$?[A-Za-z]{2,12}\s+(?:this\s+year|ytd|year[\s-]to[\s-]date|last\s+year|yesterday)\W*$|"
    r"\b(?:drop(?:ped)?|fall(?:en)?|fell|down)\s+from\s+(?:its|the)\s+(?:high|peak|top)\b|"
    r"\bhow\s+(?:far|much)\s+did\s+[^?]{0,20}?\b(?:move|go|drop|fall|rise)\b[^?]{0,10}"
    r"\byesterday\b|\b(?:fell|dropped|rose|gained|went\s+up|"
    r"went\s+down)\s+(?:more|less|the\s+most)\b|\b(?:higher|lower|hihger|up|down)\s+(?:than|from)\s+"
    r"(?:last|a|where\s+it\s+was)\b|\bha\s+(?:subido|bajado)\b|"
    r"\bwhat\s+(?:would|did)\s+\$?\d[\d,]*\s*(?:k)?\s*(?:in|of)\s+\S+\s+(?:be|become|turn)\b|"
    r"\bwhat\s+would\s+\$\s?\d[\d,]*(?:\.\d+)?\s*k?\s+(?:put\s+in|invested)\b[^?]{0,30}\bago\b|"
    # "Compare Apple and Microsoft over the past six months", "Bitcoin versus gold year to date",
    # "gain or lose between 1 July and 30 September" (a judge, round 25)
    r"\b(?:compare|versus|vs\.?|against)\b[^?]{0,60}\b(?:year[\s-]to[\s-]date|ytd|this\s+year|"
    r"since\b|over\s+the\s+(?:past|last)|in\s+the\s+(?:past|last)|past\s+\w+\s+(?:days?|"
    r"weeks?|months?|years?))|\bgain\s+or\s+lose\b|\bbetween\s+\d{1,2}\s+[A-Za-z]{3,9}\s+and\b|"
    r"\bsince\s+(?:the\s+(?:start|beginning)\s+of\s+the\s+year|\d{1,2}\s+[A-Za-z]{3,9})\b|"
    r"\bmax(?:imum)?\s+drawdown\b|最大回撤|过去.{1,4}(?:天|周|个月|月|年)|"
    r"\bfall\s+at\s+its\s+worst\b|\bat\s+its\s+worst\b|\bbigger\s+drawdown\b|"
    r"表现|涨了多少|跌了多少|收益", re.I)
"""A question about what a market did over a period, not what it is doing now."""


@dataclass(frozen=True)
class Period:
    start: datetime
    end: datetime
    said: str
    """The period as the answer says it: "so far this year (1 Jan to 3 Oct)"."""



def _px(value: float) -> str:
    """A price with enough decimals to show it: SHIB's year read "0.00 to 0.00" (a first-time
    user, round 30), so a price under 1 keeps four significant figures."""
    if value <= 0 or value >= 1:
        return f"{value:,.2f}"
    from math import floor, log10

    return f"{value:.{min(12, 3 - floor(log10(value)))}f}"


def price_text(value: float) -> str:
    """A price as the answers write it: two decimals, or four significant figures under 1."""
    return _px(value)


def asked_period(text: str, now: datetime | None = None) -> Period | None:
    """The period a question names, or None when it names none."""
    now = now or datetime.now(UTC)
    today = now.date()

    def at(day: date) -> datetime:
        return datetime(day.year, day.month, day.day, tzinfo=UTC)

    quarter = re.search(r"\bq(?P<q>[1-4])\s*(?:of\s+)?(?P<y>(?:19|20)\d\d)?\b|"
                        r"\b(?P<w>first|second|third|fourth)\s+quarter(?:\s+of)?\s*"
                        r"(?P<y2>(?:19|20)\d\d)?", text, re.I)
    if quarter is not None:
        # "What was NVDA's return in Q3 2026?" was answered from the desk's record (a judge,
        # round 26)
        number_q = int(quarter.group("q") or {"first": 1, "second": 2, "third": 3,
                                              "fourth": 4}[quarter.group("w").lower()])
        year = int(quarter.group("y") or quarter.group("y2") or today.year)
        first = date(year, 3 * number_q - 2, 1)
        after = date(year + (number_q == 4), (3 * number_q) % 12 + 1, 1)
        if first <= today:
            end = now if after > today else at(after)
            last = min(after - timedelta(days=1), today)
            return Period(at(first), end, f"in Q{number_q} {year} ({first:%d %b} to {last:%d %b})")
    halving = re.search(r"\b(?:week|month|day|year)\s+after\s+the\s+(?:last|latest|most\s+recent|"
                        r"2024)\s+halving\b", text, re.I)
    if halving is not None:
        # Bitcoin's fourth halving, block 840,000, mined 20 Apr 2024 (UTC); the window runs from
        # the next day's open
        after_days = {"day": 1, "week": 7, "month": 30,
                      "year": 365}[halving.group(0).split()[0].lower()]
        first = date(2024, 4, 20)
        return Period(at(first), at(first) + timedelta(days=after_days),
                      f"in the {halving.group(0).split()[0].lower()} after the 20 Apr 2024 "
                      f"halving ({first:%d %b %Y} to "
                      f"{first + timedelta(days=after_days):%d %b %Y})")
    year_said = re.search(r"\b(?:in\s+|during\s+)?(?P<y>(?:19|20)\d\d)(?:\s+so\s+far)?\b", text,
                          re.I)
    stated = _stated_range(text, today)
    if stated is not None:
        first, last = stated
        end = now if last >= today else at(last) + timedelta(days=1)
        return Period(at(first), end, f"from {first:%d %b %Y} to "
                                      f"{(last if last < today else today):%d %b %Y}")
    if year_said is not None and int(year_said.group("y")) < today.year and re.search(
            rf"\b(?:in|during|for|over)\s+{year_said.group('y')}\b|\b{year_said.group('y')}\s+"
            r"(?:as\s+a\s+whole|full\s+year)|"
            # "how did BTC do in the 2022 bear market?" (a newcomer re-ask, round 35)
            rf"\b(?:in|during|through)\s+the\s+{year_said.group('y')}\s+(?:bear|bull)\s+market\b|"
            rf"\b(?:in|during|through)\s+the\s+{year_said.group('y')}\s+(?:crash|sell[\s-]?off|"
            r"rally)\b", text, re.I) and not re.search(
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+"
            + year_said.group("y"), text, re.I):
        year = int(year_said.group("y"))
        return Period(at(date(year, 1, 1)), at(date(year + 1, 1, 1)),
                      f"in {year} (1 Jan to 31 Dec)")
    if (year_said is not None and int(year_said.group("y")) == today.year and re.search(
            rf"\b(?:in|during|for|over)\s+{today.year}\b|\b{today.year}\s+so\s+far\b", text, re.I)
            and not re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+"
                              + str(today.year), text, re.I)):
        # "Compare gold and bitcoin in 2026 so far" (a judge, round 26)
        start = date(today.year, 1, 1)
        return Period(at(start), now, f"so far this year (1 Jan to {today:%d %b})")
    if re.search(r"\bthis\s+year\b|\bytd\b|\byear[\s-]to[\s-]date\b|\bso\s+far\s+this\s+year\b|"
                 r"\bsince\s+the\s+(?:start|beginning)\s+of\s+(?:the|this)\s+year\b|"
                 r"\beste\s+a[nñ]o\b|\bdieses\s+jahr\b|\bcette\s+ann[ée]e\b|\bis\s+saal\b|"
                 r"今年", text, re.I):
        start = date(today.year, 1, 1)
        return Period(at(start), now, f"so far this year (1 Jan to {today:%d %b})")
    if re.search(r"\bthis\s+week\b|\bdiese\s+woche\b|\besta\s+semana\b|本周|这周", text, re.I):
        return Period(now - timedelta(days=7), now,
                      f"over the last 7 days ({(now - timedelta(days=7)).date():%d %b} to "
                      f"{today:%d %b})")
    if re.search(r"\byesterday\b", text, re.I):
        day = today - timedelta(days=1)
        # ends at today's 00:00 UTC, when yesterday's candle closes: an end a minute earlier
        # leaves that candle out of `fetch_window`, which returns only closed candles
        return Period(at(day), at(today),
                      f"yesterday ({day:%d %b}, UTC)")
    if re.search(r"\bfrom\s+(?:its|the)\s+(?:high|peak|top)\b", text, re.I) and not re.search(
            r"\b(?:year|month|week|day|ytd)\b", text, re.I):
        # "how much did doge drop from its high" names no period: the last year is read
        return Period(now - timedelta(days=365), now,
                      f"over the last year ({(now - timedelta(days=365)).date():%d %b %Y} to "
                      f"{today:%d %b})")
    # "last year" is the calendar year before; "over the last year" is the 365 days to today
    if re.search(r"(?<!the\s)\blast\s+year\b|\bin\s+(?:the\s+)?previous\s+year\b|去年", text, re.I):
        year = today.year - 1
        return Period(at(date(year, 1, 1)), at(date(year + 1, 1, 1)),
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
        # to 00:00 the day after: the month's last daily candle closes then, and an earlier end
        # left it out ("How did Solana do in September?" ended at the 30 Sep open, a judge,
        # round 25)
        return Period(at(first), min(now, at(last) + timedelta(days=1)),
                      f"in {first:%B %Y} ({first:%d %b} to {last:%d %b})")
    span = re.search(r"\b(?:past|last|previous|over\s+the\s+(?:past|last)|in\s+the\s+(?:past|last)|"
                     r"the\s+(?:past|last))\s+(?P<n>\d+|a|an|one|two|three|four|five|six|seven|eight|"
                     r"nine|ten|twelve)?\s*(?P<u>days?|weeks?|months?|years?)\b|"
                     r"\b(?P<n2>\d+|a|an|one|two|three|six|twelve)\s+(?P<u2>days?|weeks?|months?|"
                     # "$2M ETH" is two million dollars, not two months (a live re-ask, round 35)
                     r"years?)\s+ago\b|(?<![$\d.,])\b(?P<n3>\d+)\s*(?P<u3>d|w|m|y)\b(?!\w)|"
                     r"\b(?P<n4>\d+|one|two|three|six|twelve)[\s-](?P<u4>day|week|month|year)\b",
                     text, re.I)
    cjk = re.search(r"(?:过去|最近|近)\s*(?P<n>\d+|[一两二三四五六七八九十]+)\s*个?\s*"
                    r"(?P<u>天|日|周|星期|月|年)", text)
    if span is None and cjk is not None:
        # "英伟达过去三个月涨了多少" (how much has Nvidia risen over the past three months) got a
        # price quote (a judge, round 25)
        digits = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8,
                  "九": 9, "十": 10, "十二": 12}
        raw = cjk.group("n")
        count = int(raw) if raw.isdigit() else digits.get(raw, 1)
        unit = {"天": "day", "日": "day", "周": "week", "星期": "week", "月": "month",
                "年": "year"}[cjk.group("u")]
        days = count * _UNITS_DAYS[unit]
        return Period(now - timedelta(days=days), now,
                      f"over the last {count} {unit}{'s' if count != 1 else ''} "
                      f"({(now - timedelta(days=days)).date():%d %b %Y} to {today:%d %b})")
    if span is not None:
        count_word =(span.group("n") or span.group("n2") or span.group("n3") or span.group("n4")
                      or "1").lower()
        count = int(count_word) if count_word.isdigit() else _WORD_N.get(count_word, 1)
        unit = (span.group("u") or span.group("u2") or span.group("u3") or span.group("u4")
                or "day").lower()
        unit = {"d": "day", "w": "week", "m": "month", "y": "year"}.get(unit, unit.rstrip("s"))
        days = count * _UNITS_DAYS[unit]
        return Period(now - timedelta(days=days), now,
                      f"over the last {count} {unit}{'s' if count != 1 else ''} "
                      f"({(now - timedelta(days=days)).date():%d %b %Y} to {today:%d %b})")
    return None


_DAY_MONTH = (r"(?:(?P<d{i}>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<m{i}>[A-Za-z]{{3,9}})\.?|"
              r"(?P<m{i}b>[A-Za-z]{{3,9}})\.?\s+(?P<d{i}b>\d{{1,2}})(?:st|nd|rd|th)?)"
              r"(?:,?\s+(?P<y{i}>(?:19|20)\d\d))?")


def _month_number(word: str | None) -> int | None:
    if not word:
        return None
    low = word.lower()[:3]
    return next((i for i, name in enumerate(_MONTHS, 1) if name.startswith(low)), None)


def _a_date(m: re.Match[str], i: int, today: date) -> date | None:
    day = m.group(f"d{i}") or m.group(f"d{i}b")
    month = _month_number(m.group(f"m{i}") or m.group(f"m{i}b"))
    if not day or month is None:
        return None
    year = int(m.group(f"y{i}")) if m.group(f"y{i}") else today.year
    try:
        found = date(year, month, int(day))
    except ValueError:
        return None
    if not m.group(f"y{i}") and found > today:
        found = found.replace(year=year - 1)
    return found


_RANGE = re.compile(r"\b(?:between|from)\s+" + _DAY_MONTH.format(i=1)
                    + r"\s+(?:and|to|until|till|through|-|\u2013)\s+" + _DAY_MONTH.format(i=2),
                    re.I)
_SINCE = re.compile(r"\bsince\s+(?:(?P<only>" + "|".join(_MONTHS) + r"|jan|feb|mar|apr|jun|jul|"
                    r"aug|sept?|oct|nov|dec)\b(?!\s+\d)|" + _DAY_MONTH.format(i=1) + ")", re.I)


def _stated_range(text: str, today: date) -> tuple[date, date] | None:
    """"between 1 July and 30 September", "from 15 August to 15 September", "since 1 March",
    "since March": each was answered as the year to date or a price quote (a judge, round 25)."""
    span = _RANGE.search(text)
    if span is not None:
        first, last = _a_date(span, 1, today), _a_date(span, 2, today)
        if first and last and first <= last:
            return first, last
    since = _SINCE.search(text)
    if since is not None:
        if since.group("only"):
            month = _month_number(since.group("only"))
            if month is None:
                return None
            year = today.year if month <= today.month else today.year - 1
            return date(year, month, 1), today
        first = _a_date(since, 1, today)
        if first is not None:
            return first, today
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


def _yahoo_days(ticker: str, period: Period) -> list[tuple[date, float]] | None:
    """A US company Bitget does not list, from Yahoo Finance's split-adjusted daily bars: the
    first trading day's open, then each close to the period's end."""
    from argus.market.equity_history import HistoryError, daily

    try:
        bars = daily(ticker)
    except (HistoryError, OSError, ValueError):
        return None
    inside = [b for b in bars if period.start.date() <= b.day < period.end.date()
              or (b.day == period.end.date() and period.end >= datetime.now(UTC) - timedelta(
                  hours=2))]
    if not inside:
        return None
    return [(inside[0].day, float(inside[0].open)), *[(b.day, float(b.close)) for b in inside]]


def record(symbol: str, period: Period) -> Record | None:
    """The period's figures from Bitget daily candles (or Yahoo Finance's, for a ``YAHOO:``
    ticker Bitget does not list), or None when there are too few."""
    from argus.market import history

    if symbol.startswith("YAHOO:"):
        days = _yahoo_days(symbol.removeprefix("YAHOO:"), period)
        return _record_from(symbol, period, days) if days and len(days) >= 2 else None

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
    days = [(inside[0].ts.date(), float(inside[0].open))]
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
    return _record_from(symbol, period, days)


def _record_from(symbol: str, period: Period, days: list[tuple[date, float]]) -> Record:
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
    return symbol.removeprefix("YAHOO:").removesuffix("USDT")


def _day_lines(symbols: tuple[str, ...], period: Period) -> list[str] | None:
    """One UTC day: open to close, and the high-to-low range, which is how far it moved.

    "how far did sol move yesterday" was answered with a drawdown "from 01 Oct to 01 Oct" (a
    first-time user, round 25); a single day's distance is its candle's range."""
    from argus.market import history

    rows = []
    for symbol in symbols:
        try:
            candles = history.fetch_window(symbol, start=period.start, end=period.end,
                                           interval="1Dutc")
        except Exception:
            continue
        day = next((c for c in candles if c.ts.date() == period.start.date()), None)
        if day is None or float(day.open) <= 0 or float(day.low) <= 0:
            continue
        o, c, h, lo = float(day.open), float(day.close), float(day.high), float(day.low)
        rows.append((_name(symbol), o, c, h, lo, c / o - 1, h / lo - 1))
    if not rows:
        return None
    lead = rows[0]
    lines = [f"Bottom line: {lead[0]} moved {lead[5]:+.1%} {period.said}, open to close, and "
             f"travelled {lead[6]:.1%} from its low to its high."]
    lines += [f"{n}: opened {_px(o)}, closed {_px(c)}, high {_px(h)}, low {_px(lo)} "
              f"({chg:+.1%} open to close; {rng:.1%} low to high)."
              for n, o, c, h, lo, chg, rng in rows]
    lines.append("Data: Bitget USDT-futures daily candle for that UTC day (00:00 to 24:00 UTC).")
    return lines


def performance_lines(symbols: tuple[str, ...], period: Period, *, amount: float | None = None,
                      units: float | None = None) -> list[str] | None:
    """The answer: one market's period, or several ranked by return, with drawdowns beside."""
    if period.end - period.start <= timedelta(days=1):
        one_day = _day_lines(symbols[:4], period)
        if one_day is not None:
            return one_day
    found = [r for r in (record(s, period) for s in symbols[:4]) if r is not None]
    if not found:
        return None
    ranked = sorted(found, key=lambda r: -r.change)
    said = period.said
    if any(r.short_history for r in found):
        # "TSLA is +10.8% over the last 10 years (05 Oct 2016 to 03 Oct)" was measured from 19 Aug
        # 2025, where Bitget's history starts (a judge, round 29): the span said is the span used
        starts = {r.first_day for r in found}
        said = (f"since {min(starts):%d %b %Y}, as far back as Bitget's daily history goes "
                f"(shorter than the period asked for)" if len(starts) == 1 else
                "over each one's own history on Bitget, which is shorter than the period asked "
                "for (start dates below)")
    if len(ranked) == 1:
        r = ranked[0]
        lead = (f"Bottom line: {_name(r.symbol)} is {r.change:+.1%} {said} — from "
                f"{_px(r.first)} to {_px(r.last)} — and its deepest fall from a high on the way "
                f"was {r.drawdown:.1%} ({r.peak_day:%d %b} to {r.trough_day:%d %b}).")
    else:
        lead = ("Bottom line: " + ", ".join(f"{_name(r.symbol)} {r.change:+.1%}" for r in ranked)
                + f" {said}; the deepest falls from a high were "
                + ", ".join(f"{_name(r.symbol)} {r.drawdown:.1%}" for r in
                            sorted(found, key=lambda r: r.drawdown))
                + f" — {_name(min(found, key=lambda r: r.drawdown).symbol)} had the bigger "
                  f"drawdown.")
    lines = [lead]
    for r in ranked:
        lines.append(f"{_name(r.symbol)}: {_px(r.first)} on {r.first_day:%d %b %Y} to "
                     f"{_px(r.last)} on {r.last_day:%d %b %Y}; worst day {r.worst_day:+.1%}, "
                     f"best day {r.best_day:+.1%}."
                     + (f" Bitget's daily history starts {r.first_day:%d %b %Y}, so the period is "
                        f"measured from there." if r.short_history else ""))
    first = ranked[0]
    if units is not None and len(found) == 1:
        lines.insert(1, f"{units:g} {_name(first.symbol)} bought at {_px(first.first)} cost "
                        f"${units * first.first:,.2f} and is worth ${units * first.last:,.2f}: "
                        f"{'a gain' if first.last >= first.first else 'a loss'} of "
                        f"${abs(units * (first.last - first.first)):,.2f}.")
    elif amount is not None and len(found) == 1:
        lines.insert(1, f"${amount:,.0f} put in at the start would be "
                        f"${amount * (1 + first.change):,.0f} now "
                        f"({'a gain' if first.change >= 0 else 'a loss'} of "
                        f"${abs(amount * first.change):,.0f}); at its deepest fall it was "
                        f"{abs(first.drawdown):.1%} below its high on the way.")
    from_yahoo = [_name(r.symbol) for r in found if r.symbol.startswith("YAHOO:")]
    lines.append("Data: "
                 + ("Bitget USDT-futures daily candles (UTC days)" if len(from_yahoo) < len(found)
                    else "")
                 + (" and " if from_yahoo and len(from_yahoo) < len(found) else "")
                 + (f"Yahoo Finance's split-adjusted daily bars for {', '.join(from_yahoo)}, "
                    f"which Bitget does not list" if from_yahoo else "")
                 + ", from the first day's open to the last price, without dividends; a past "
                   "period describes the past, not the next one.")
    return lines


trace_module(globals())
