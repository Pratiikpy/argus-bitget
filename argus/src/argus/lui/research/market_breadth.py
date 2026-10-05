"""Market-breadth questions answered from long daily histories: sector rotation, the Magnificent
Seven against the equal-weight index, the 200-day regime, the volatility regime, valuation against
history, and which of several candidates drives an asset.

Round 43's judge audit of the live console put six questions to the router and got the wrong
engine each time (MAJOR 3, 7, 10, 14, 20 and CRITICAL 10):

- "Which S&P 500 sectors have led over the past month, and is money rotating from tech into
  defensives?" returned the S&P index return (+0.8%) and no sector.
- "Is volatility in a high or low regime right now for BTC and for the S&P 500?" covered BTC only,
  with no percentile and no label.
- "How have the Magnificent Seven performed against equal-weight S&P this quarter?" used 30 days,
  not the quarter, and the cap-weighted contract, not equal weight.
- "Is gold or the dollar the stronger driver of silver this quarter?" compared gold's and silver's
  30-day momentum and never touched the dollar.
- "Is the S&P 500 in a bull or bear regime based on its 200 day moving average...?" used Bitget's
  index history, which starts on 22 May 2026, far too short for a 200-day average.
- "Is the Nasdaq 100 more expensive than its 10 year average on forward earnings?" returned a
  4.5-month price return and no multiple.

Every one needs a history that Bitget's young perpetuals do not have, so this module reads Yahoo
Finance's daily chart (``/v8/finance/chart/{ticker}?period1=0&period2=9999999999&interval=1d``;
``range=max`` silently returns monthly bars, so the explicit period is required). Entry point:
:func:`lines`. Each sub-reader is strict about its trigger and returns ``None`` otherwise.

Checked live on 2026-10-06 (bars, first date, last close): ^GSPC 14,311 from 1970-01-02, 7,773.95;
^NDX since 1985; SPY 406 months of history; RSP and the eleven SPDR sector funds answer; GC=F
6,632 from 2000-08-30, 4,166.80; SI=F 61.38; DX-Y.NYB 17,278 from 1971-01-04, 102.14; ^TNX 5.311
(a yield in percent, so a daily *change* in percentage points is the series that is used, not a
return); ^VIX 15.52; BTC-USD 4,402 from 2014-09-17, 85,830. Prices are the raw split-adjusted
``close`` (not ``adjclose``), so every return here is a price return with dividends excluded, and
the answers say so.

Valuation sources, checked on the same day:

- Yahoo quoteSummary through ``EstimatesSource.summary`` (the call ``fundamentals.yahoo_summary``
  makes, with ``topHoldings`` added): QQQ ``summaryDetail.trailingPE`` 30.82, SPY 25.03;
  ``forwardPE`` is an empty object for both funds and for ^NDX and ^GSPC, so Yahoo publishes no
  forward multiple for them. ``topHoldings.equityHoldings.priceToEarnings`` is an earnings yield
  (QQQ 0.03423, SPY 0.04035: 1/x is 29.2x and 24.8x); the source does not say whether its
  earnings are trailing or forward, and the answer says that rather than naming one.
- multpl.com ``/s-p-500-pe-ratio/table/by-month``: its robots.txt lists only a sitemap and
  disallows nothing. The page carries 1,871 monthly rows from 1871; the latest rows (Jul-Oct 2026)
  are marked "Estimate": Oct 2026 26.34, Sep 26.04, Aug 26.11, Oct 2025 28.42. This is the S&P 500's
  price over trailing as-reported earnings, not the Nasdaq-100's and not forward. No keyless
  Nasdaq-100 multiple history was found, so the answer says the 10-year average it measures is the
  S&P 500's.

Method notes. Rotation compares the tech fund (XLK) with the equal average of the three defensive
funds (XLP, XLU, XLV) over the last month and over the three months before it; the verdict is a
description of price strength, never a forecast and never fund flows. The 200-day base rate takes
every session of the asset's own history on which a 200-day average exists, splits them into
closes above and below it, and measures the following 63 sessions; the windows overlap, which the
answer states. Realised volatility is the sample standard deviation of 30 daily log returns,
annualised by 252 sessions (365 for crypto); its percentile is the position among every
30-day reading of the last five years, and the label is the tercile that percentile falls in. The
driver regression is ordinary least squares on common-date daily returns (a yield enters as its
change); unique contribution is the R-squared lost when that driver is removed.
"""

from __future__ import annotations

import math
import re
import threading
import time
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from typing import Any, Final
from urllib.parse import quote

from argus.truth import http
from argus.truth.bounded import BoundedDict

CHART: Final = ("https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
                "?period1=0&period2=9999999999&interval=1d")
MULTPL: Final = "https://www.multpl.com/s-p-500-pe-ratio/table/by-month"
CACHE_SECONDS: Final = 6 * 3600.0
TIMEOUT: Final = 20.0
_UA: Final = {"User-Agent": "Mozilla/5.0 argus-research"}

SECTORS: Final[dict[str, str]] = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "Health Care", "XLE": "Energy",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLI": "Industrials",
    "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real Estate", "XLC": "Communication Services",
}
CYCLICALS: Final = ("XLK", "XLY", "XLF", "XLI", "XLC")
DEFENSIVES: Final = ("XLP", "XLU", "XLV")
MAG7: Final = ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA")
NOISE_PP: Final = 0.5
"""Percentage points below which a sector spread is called level rather than a lead."""

FORWARD_SESSIONS: Final = 63
VOL_WINDOW: Final = 30
SMA_LONG, SMA_SHORT = 200, 50


class BreadthError(RuntimeError):
    """A history could not be read; the answer says so rather than guess."""


# --------------------------------------------------------------------------- data


@dataclass(frozen=True, slots=True)
class Series:
    ticker: str
    dates: list[date]
    closes: list[float]

    def index_on(self, day: date) -> int:
        """Index of the last bar on or before ``day`` (-1 when the series starts later)."""
        return bisect_right(self.dates, day) - 1


_cache: BoundedDict[str, tuple[float, Series]] = BoundedDict(128)
_lock = threading.Lock()


def _today() -> date:
    return datetime.now(UTC).date()


def _parse_chart(ticker: str, payload: dict[str, Any]) -> Series:
    try:
        result = payload["chart"]["result"][0]
        stamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
        offset = int(result.get("meta", {}).get("gmtoffset") or 0)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise BreadthError(f"Yahoo's chart for {ticker} had an unexpected shape") from exc
    dates: list[date] = []
    values: list[float] = []
    for stamp, close in zip(stamps, closes, strict=False):
        if close is None or close <= 0:
            continue
        day = datetime.fromtimestamp(stamp + offset, tz=UTC).date()
        if dates and day <= dates[-1]:
            continue
        dates.append(day)
        values.append(float(close))
    return Series(ticker, dates, values)


def _series(ticker: str) -> Series:
    """Yahoo's daily closes for ``ticker`` since listing, cached for six hours."""
    now = time.monotonic()
    with _lock:
        hit = _cache.get(ticker)
        if hit and now - hit[0] < CACHE_SECONDS:
            return hit[1]
    try:
        payload = http.fetch_json(CHART.format(ticker=quote(ticker, safe="")), timeout=TIMEOUT,
                                  headers=_UA)
    except http.RpcError as exc:
        raise BreadthError(f"Yahoo's daily history for {ticker} did not arrive "
                           f"({type(exc).__name__})") from exc
    series = _parse_chart(ticker, payload)
    if len(series.dates) < 30:
        raise BreadthError(f"Yahoo returned only {len(series.dates)} daily bars for {ticker}")
    with _lock:
        _cache[ticker] = (now, series)
    return series


# --------------------------------------------------------------------------- windows


@dataclass(frozen=True, slots=True)
class Window:
    label: str
    base: date
    """The last close on or before this day is the starting price."""
    end: date | None = None
    """The last close on or before this day is the ending price (None: the latest close)."""


def _months_back(day: date, months: int) -> date:
    index = day.year * 12 + day.month - 1 - months
    year, month = divmod(index, 12)
    last = (date(year + (month + 1) // 12, (month + 1) % 12 + 1, 1) - timedelta(days=1)).day
    return date(year, month + 1, min(day.day, last))


def _quarter_start(day: date) -> date:
    return date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)


_UNITS: Final = {"day": 1, "week": 7}
_TRAILING: Final = re.compile(
    r"\b(?:past|last|trailing|previous|over\s+the\s+(?:past|last)|in\s+the\s+(?:past|last))\s+"
    r"(?:(\d{1,3})\s+)?(day|week|month|year|quarter)s?\b", re.I)


def parse_window(text: str, today: date, default: str = "quarter") -> Window:
    """The period the question names, as a :class:`Window`; ``default`` when it names none
    ("quarter", "ytd" or "3 months"), which the answer then states."""
    q = text.lower()
    if re.search(r"\b(?:last|previous|prior)\s+quarter\b", q):
        this = _quarter_start(today)
        prev = _quarter_start(this - timedelta(days=1))
        end = this - timedelta(days=1)
        return Window(f"last quarter ({prev.day} {prev:%b} to {end.day} {end:%b %Y})",
                      prev - timedelta(days=1), end)
    if re.search(r"\b(?:this\s+quarter|quarter[- ]to[- ]date|qtd|current\s+quarter|"
                 r"so\s+far\s+this\s+quarter)\b", q):
        return Window("the quarter to date", _quarter_start(today) - timedelta(days=1))
    if re.search(r"\b(?:ytd|year[- ]to[- ]date|this\s+year|so\s+far\s+this\s+year|"
                 r"since\s+(?:the\s+start\s+of\s+)?(?:the\s+year|january))\b", q):
        return Window("the year to date", date(today.year - 1, 12, 31))
    if re.search(r"\b(?:this\s+month|month[- ]to[- ]date|mtd)\b", q):
        return Window("the month to date", today.replace(day=1) - timedelta(days=1))
    m = _TRAILING.search(q)
    if m:
        n = int(m.group(1) or 1)
        unit = m.group(2)
        if unit in _UNITS:
            return Window(f"the past {n} {unit}{'s' if n != 1 else ''}",
                          today - timedelta(days=n * _UNITS[unit]))
        months = n * {"month": 1, "year": 12, "quarter": 3}[unit]
        label = f"the past {n} {unit}{'s' if n != 1 else ''}"
        return Window(label, _months_back(today, months))
    if default == "ytd":
        return Window("the year to date (no period was named)", date(today.year - 1, 12, 31))
    if default == "3 months":
        return Window("the past 3 months (no period was named)", _months_back(today, 3))
    return Window("the quarter to date (no period was named)", _quarter_start(today)
                  - timedelta(days=1))


def _return(series: Series, base: date, end: date | None = None) -> float | None:
    """Close-to-close return from the last bar on or before ``base`` to the one on or before
    ``end``; None when the series does not reach back to ``base`` or is stale there."""
    j = series.index_on(end) if end else len(series.dates) - 1
    i = series.index_on(base)
    if i < 0 or j <= i or (base - series.dates[i]).days > 7:
        return None
    return series.closes[j] / series.closes[i] - 1.0


# --------------------------------------------------------------------------- formatting


def _pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:+.{digits}f}%"


def _pp(x: float) -> str:
    return f"{x * 100:+.1f} points"


def _pts(x: float) -> str:
    """A gap in percentage points as prose, without a sign ("by 1.0 points")."""
    return f"{abs(x) * 100:.1f} points"


def _num(x: float, digits: int = 2) -> str:
    return f"{x:,.{digits}f}"


def _day(d: date) -> str:
    return f"{d.day} {d:%b %Y}"


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _failed(what: str, exc: Exception) -> list[str]:
    return [f"Bottom line: {what} could not be read just now ({exc}), so no figure is given.",
            "Data: Yahoo Finance daily chart. Not advice."]


# --------------------------------------------------------------------------- 1. sector rotation

_CRYPTO_WORDS: Final = re.compile(
    r"\b(?:crypto\w*|coins?|tokens?|defi|bitcoin|btc|eth\w*|solana|layer[- ]?[12]|nft|meme)\b",
    re.I)
_SECTOR_Q: Final = re.compile(
    r"\bsectors\b[^?.]{0,80}\b(?:led|lead|leading|leaders?|lag\w*|rotat\w*|outperform\w*|"
    r"underperform\w*|strongest|weakest|best|worst|performing|performed|working|hot|cold)\b|"
    r"\b(?:led|lead|leading|rotat\w*|outperform\w*|strongest|weakest|best|worst)\b[^?.]{0,60}"
    r"\bsectors\b|\bsector\s+rotation\b|"
    r"\brotat\w*\s+(?:out\s+of|from|into)\b[^?.]{0,40}\b(?:defensives?|cyclicals?|tech\w*|"
    r"growth|value)\b", re.I)


def _sector_lines(text: str) -> list[str] | None:
    if not _SECTOR_Q.search(text) or _CRYPTO_WORDS.search(text):
        return None
    try:
        funds = {t: _series(t) for t in SECTORS}
    except BreadthError as exc:
        return _failed("The sector funds' price history", exc)
    last = min(s.dates[-1] for s in funds.values())
    month_ago = _months_back(last, 1)
    quarter_ago = _months_back(last, 3)
    prior_start = _months_back(last, 4)
    m1: dict[str, float] = {}
    m3: dict[str, float] = {}
    prior: dict[str, float] = {}
    for t, fund in funds.items():
        r1 = _return(fund, month_ago, last)
        r3 = _return(fund, quarter_ago, last)
        rp = _return(fund, prior_start, month_ago)
        if r1 is not None:
            m1[t] = r1
        if r3 is not None:
            m3[t] = r3
        if rp is not None:
            prior[t] = rp
    if len(m1) < 8:
        return ["Bottom line: fewer than eight of the eleven sector funds returned a month of "
                "history, so no ranking is given.", "Data: Yahoo Finance daily chart. Not advice."]
    rows = sorted(m1.items(), key=lambda kv: kv[1], reverse=True)

    def spread(group: Sequence[str], book: dict[str, float]) -> float | None:
        if any(t not in book for t in (*group, *DEFENSIVES)):
            return None
        return (sum(book[t] for t in group) / len(group)
                - sum(book[t] for t in DEFENSIVES) / len(DEFENSIVES))

    tech_1m, tech_prior = spread(("XLK",), m1), spread(("XLK",), prior)
    cyc_1m, cyc_3m = spread(CYCLICALS, m1), spread(CYCLICALS, m3)
    top = rows[:3]
    rose = sum(1 for _, v in rows if v > 0)
    head = (f"Bottom line: over the month to {_day(last)} the S&P 500 sectors that led were "
            + ", ".join(f"{SECTORS[t]} ({t}) {_pct(v)}" for t, v in top)
            + f" ({rose} of {len(rows)} sectors rose); the weakest was {SECTORS[rows[-1][0]]} "
            f"({rows[-1][0]}) {_pct(rows[-1][1])}. ")
    if tech_1m is None:
        verdict = "The tech-versus-defensives rotation could not be measured (a fund is missing)."
    elif tech_1m < -NOISE_PP / 100 and tech_prior is not None and tech_prior > NOISE_PP / 100:
        verdict = (f"Yes, by price strength: tech led the defensives by {_pts(tech_prior)} in the "
                   f"three months before, then lagged them by {_pts(tech_1m)} over the last month.")
    elif tech_1m < -NOISE_PP / 100:
        verdict = (f"Defensives beat tech over the last month by {_pts(tech_1m)}, but tech had "
                   "not been leading in the three months before, so this is weakness continuing, "
                   "not a hand-over of leadership.")
    elif tech_1m > NOISE_PP / 100:
        verdict = (f"No: tech beat the defensives over the last month by {_pts(tech_1m)}; money, "
                   "as price shows it, is not moving from tech into defensives.")
    else:
        verdict = ("No clear rotation: tech and the defensives are within "
                   f"{NOISE_PP} points of each other over the last month.")
    out = [head + verdict]
    out.append("Ranked, 1-month then 3-month return: "
               + "; ".join(f"{t} {_pct(v)} / {_pct(m3[t]) if t in m3 else 'n/a'}"
                           for t, v in rows) + ".")
    if tech_1m is not None:
        prior_txt = _pp(tech_prior) if tech_prior is not None else "n/a"
        out.append(f"Technology (XLK) minus the average of the three defensives (XLP, XLU, XLV): "
                   f"{_pp(tech_1m)} over the last month, {prior_txt} over the three months "
                   f"before it ({_day(prior_start)} to {_day(month_ago)}).")
    if cyc_1m is not None and cyc_3m is not None:
        out.append("Cyclicals (XLK, XLY, XLF, XLI, XLC, equal-weighted) minus defensives: "
                   f"{_pp(cyc_1m)} over 1 month, {_pp(cyc_3m)} over 3 months.")
    out.append("This is relative price strength of the eleven Select Sector SPDR funds, not fund "
               "flows (no keyless source publishes those by sector), and it is not a forecast.")
    out.append(f"Data: Yahoo Finance daily closes for the eleven SPDR sector funds to {_day(last)} "
               "(price returns, dividends excluded). Not advice.")
    return out


# --------------------------------------------------------------------------- 2. Magnificent Seven

_MAG7_Q: Final = re.compile(
    r"\b(?:magnificent\s+(?:seven|7)|mag\s*7|mag\s+seven|mag7|magnificent7)\b", re.I)


def _mag7_block(names: dict[str, Series], window: Window) -> list[str]:
    """The basket-versus-index comparison over one window: the answer line, each name, and the
    dates it was measured between."""
    returns = {t: _return(s, window.base, window.end) for t, s in names.items()}
    missing = [t for t, r in returns.items() if r is None]
    if missing:
        return [f"{', '.join(missing)} has no price history reaching back to the start of "
                f"{window.label}, so the comparison is not given."]
    stocks = {t: float(returns[t] or 0.0) for t in MAG7}
    basket = sum(stocks.values()) / len(stocks)
    rsp, spy = float(returns["RSP"] or 0.0), float(returns["SPY"] or 0.0)
    spy_series = names["SPY"]
    base_i = spy_series.index_on(window.base)
    end_i = spy_series.index_on(window.end) if window.end else len(spy_series.dates) - 1
    ranked = sorted(stocks.items(), key=lambda kv: kv[1], reverse=True)
    ahead = sum(1 for v in stocks.values() if v > rsp)
    return [
        f"over {window.label} the Magnificent Seven (equal-weighted basket) returned "
        f"{_pct(basket)} against {_pct(rsp)} for the equal-weight S&P 500 (RSP) and {_pct(spy)} "
        f"for the cap-weighted S&P 500 (SPY): the basket {'beat' if basket > rsp else 'lagged'} "
        f"equal-weight by {_pts(basket - rsp)} and {'beat' if basket > spy else 'lagged'} "
        f"cap-weight by {_pts(basket - spy)}.",
        "Each name: " + ", ".join(f"{t} {_pct(v)}" for t, v in ranked)
        + f". {ahead} of 7 beat RSP.",
        f"Measured from the close of {_day(spy_series.dates[base_i])} to the close of "
        f"{_day(spy_series.dates[end_i])} ({end_i - base_i} sessions).",
    ]


def _mag7_lines(text: str) -> list[str] | None:
    if not _MAG7_Q.search(text):
        return None
    today = _today()
    window = parse_window(text, today)
    try:
        names = {t: _series(t) for t in (*MAG7, "RSP", "SPY")}
    except BreadthError as exc:
        return _failed("The Magnificent Seven or index funds' price history", exc)
    block = _mag7_block(names, window)
    out = ["Bottom line: " + block[0], *block[1:]]
    spy = names["SPY"]
    sessions = len(spy.dates) - 1 - spy.index_on(window.base)
    if sessions < 20 and "quarter" in window.label:
        previous = _mag7_block(names, parse_window("last quarter", today))
        out.append(f"Only {sessions} sessions of this quarter exist so far, so a full quarter for "
                   f"comparison: {previous[0]}")
    out.append("The basket gives each name one seventh at the start and does not rebalance; RSP is "
               "the Invesco S&P 500 Equal Weight ETF, SPY the SPDR S&P 500 ETF.")
    out.append("Data: Yahoo Finance daily closes for AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, "
               "RSP and SPY (price returns, dividends excluded). Not advice.")
    return out


# --------------------------------------------------------------------------- asset aliases


@dataclass(frozen=True, slots=True)
class Asset:
    pattern: re.Pattern[str]
    ticker: str
    label: str
    crypto: bool = False
    change: bool = False
    """A yield: its daily move is a change in percentage points, not a return."""


def _a(pattern: str, ticker: str, label: str, **kw: bool) -> Asset:
    return Asset(re.compile(pattern, re.I), ticker, label, **kw)


ASSETS: Final[tuple[Asset, ...]] = (
    _a(r"\bs\s*&\s*p(?:\s*-?\s*500)?\b|\bspx\b|\bsp500\b|\bspy\b", "^GSPC", "the S&P 500"),
    _a(r"\bnasdaq(?:\s*-?\s*100)?\b|\bndx\b|\bqqq\b", "^NDX", "the Nasdaq 100"),
    _a(r"\brussell(?:\s*2000)?\b|\brut\b", "^RUT", "the Russell 2000"),
    _a(r"\bbitcoin\b|\bbtc\b", "BTC-USD", "BTC", crypto=True),
    _a(r"\bethereum\b|\bether\b|\beth\b", "ETH-USD", "ETH", crypto=True),
    _a(r"\bsolana\b|\bsol\b", "SOL-USD", "SOL", crypto=True),
    _a(r"\bgold\b|\bxau\b", "GC=F", "gold"),
    _a(r"\bsilver\b|\bxag\b", "SI=F", "silver"),
    _a(r"\bcopper\b", "HG=F", "copper"),
    _a(r"\bbrent\b", "BZ=F", "Brent crude"),
    _a(r"\bcrude\b|\bwti\b|\boil\b", "CL=F", "WTI crude"),
    _a(r"\bdollar\b|\bdxy\b|\busd\s+index\b", "DX-Y.NYB", "the US dollar index (DXY)"),
    _a(r"\b10\s*-?\s*(?:year|yr|y)\b|\bten[- ]year\b|\b(?:treasury|bond)\s+yields?\b|\byields?\b",
       "^TNX", "the 10-year Treasury yield", change=True),
    _a(r"\bvix\b|\bvolatility\s+index\b", "^VIX", "the VIX"),
)


def _mentions(text: str) -> list[tuple[int, int, Asset]]:
    """Every named asset with the span of its first mention, in the order mentioned."""
    found: list[tuple[int, int, Asset]] = []
    for asset in ASSETS:
        m = asset.pattern.search(text)
        if m:
            found.append((m.start(), m.end(), asset))
    found.sort(key=lambda f: f[0])
    return found


# --------------------------------------------------------------------------- 3. 200-day regime

_SMA_Q: Final = re.compile(
    r"\b200[\s-]*(?:day|d)\b[\s-]*(?:simple\s+|exponential\s+)?(?:moving\s+average|average|ma|sma)\b|"
    r"\b200[\s-]*dma\b", re.I)
_REGIME_Q: Final = re.compile(
    r"\b(?:regime|bull\w*|bear\w*|uptrend|downtrend|how\s+far|distance|above|below|under|over|"
    r"trend|cross\w*)\b", re.I)
_TICKER: Final = re.compile(r"\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b")
_STOP: Final = frozenset({
    "IS", "THE", "AND", "OR", "OF", "IN", "ON", "AT", "TO", "FOR", "BY", "IT", "AS", "AN", "DAY",
    "MA", "SMA", "EMA", "DMA", "US", "USA", "ETF", "RSI", "HOW", "FAR", "BULL", "BEAR", "ARE",
    "WAS", "NOW", "ITS", "ANY", "CAN", "ALL", "NOT", "BUT", "DOES", "DID", "FROM", "THAN",
    "THAT", "THIS", "WITH", "WHAT", "WHEN", "WILL", "YOU", "YEAR", "PE", "EPS", "ATH", "NYSE",
})


_COMPANIES: Final[dict[str, str]] = {
    "tesla": "TSLA", "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "amazon": "AMZN",
    "alphabet": "GOOGL", "google": "GOOGL", "meta": "META", "netflix": "NFLX",
    "coinbase": "COIN", "palantir": "PLTR",
}
"""Company names a question may use instead of the ticker, for the 200-day reader."""


def _subject(text: str, mentions: list[tuple[int, int, Asset]]) -> tuple[str, str] | None:
    """(ticker, label) the question is about: a named index or crypto asset, else a US ticker."""
    for _, _, asset in mentions:
        if asset.ticker in {"^GSPC", "^NDX", "^RUT", "BTC-USD", "ETH-USD", "SOL-USD"}:
            return asset.ticker, asset.label
    for name, ticker in _COMPANIES.items():
        if re.search(rf"\b{name}\b", text, re.I):
            return ticker, name.capitalize()
    for m in _TICKER.finditer(text):
        sym = m.group(1) or m.group(2)
        if sym and sym not in _STOP and not any(a.pattern.fullmatch(sym) for a in ASSETS):
            return sym, sym
    return None


def _sma(values: list[float], n: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    run = 0.0
    for i, v in enumerate(values):
        run += v
        if i >= n:
            run -= values[i - n]
        if i >= n - 1:
            out[i] = run / n
    return out


def _regime_lines(text: str) -> list[str] | None:
    if not _SMA_Q.search(text) or not _REGIME_Q.search(text):
        return None
    subject = _subject(text, _mentions(text))
    if subject is None:
        return None
    ticker, label = subject
    try:
        s = _series(ticker)
    except BreadthError as exc:
        return _failed(f"{label}'s daily history", exc)
    if len(s.closes) < SMA_LONG + 1:
        return [f"Bottom line: {label} has {len(s.closes)} daily closes on Yahoo, fewer than the "
                f"{SMA_LONG} a 200-day average needs.", "Data: Yahoo Finance daily chart. "
                "Not advice."]
    sma200, sma50 = _sma(s.closes, SMA_LONG), _sma(s.closes, SMA_SHORT)
    last = len(s.closes) - 1
    avg = sma200[last] or 0.0
    price = s.closes[last]
    gap = price / avg - 1.0
    above = price > avg
    run = 0
    for i in range(last, SMA_LONG - 2, -1):
        a = sma200[i]
        if a is None or (s.closes[i] > a) != above:
            break
        run += 1
    cross_day: date | None = None
    cross_kind = ""
    for i in range(last, SMA_LONG - 1, -1):
        a, b = sma50[i], sma200[i]
        pa, pb = sma50[i - 1], sma200[i - 1]
        if a is None or b is None or pa is None or pb is None:
            continue
        if (a > b) != (pa > pb):
            cross_day, cross_kind = s.dates[i], "golden cross" if a > b else "death cross"
            break
    fifty = sma50[last] or 0.0
    line_then = sma200[last - 21]
    slope = ("rising" if line_then and avg > line_then else "falling") if line_then else "flat"
    peak = max(s.closes)
    off_peak = price / peak - 1.0
    name = f"{label} ({ticker})" if label != ticker else ticker
    out = [f"Bottom line: {name} closed at {_num(price)} on {_day(s.dates[last])}, "
           f"{_pct(abs(gap))[1:]} {'above' if above else 'below'} its 200-day moving average of "
           f"{_num(avg)}: {'a bull regime' if above else 'a bear regime'} by the 200-day rule, "
           f"{run} sessions on that side."]
    cross = (f"; the last cross was a {cross_kind} on {_day(cross_day)}" if cross_day else "")
    out.append(f"The 50-day average ({_num(fifty)}) is {'above' if fifty > avg else 'below'} the "
               f"200-day{cross}. The 200-day line itself is {slope} against 21 sessions ago "
               f"({_num(line_then) if line_then else 'n/a'}). The close is {abs(off_peak):.1%} "
               f"{'below' if off_peak < 0 else 'at'} its highest close on record "
               f"({_day(s.dates[s.closes.index(peak)])}).")
    ups: list[float] = []
    downs: list[float] = []
    for i in range(SMA_LONG - 1, last - FORWARD_SESSIONS + 1):
        a = sma200[i]
        if a is None:
            continue
        fwd = s.closes[i + FORWARD_SESSIONS] / s.closes[i] - 1.0
        (ups if s.closes[i] > a else downs).append(fwd)
    years = (s.dates[last] - s.dates[SMA_LONG - 1]).days / 365.25
    if len(ups) >= 250 and len(downs) >= 250 and years >= 3:
        def summary(v: list[float]) -> str:
            ordered = sorted(v)
            med = ordered[len(ordered) // 2]
            pos = sum(1 for x in v if x > 0) / len(v)
            return f"mean {_pct(sum(v) / len(v))}, median {_pct(med)}, positive {pos:.0%}"
        out.append(f"Base rate on {label}'s own history ({_day(s.dates[SMA_LONG - 1])} to "
                   f"{_day(s.dates[last])}, {len(ups) + len(downs):,} sessions): after a close "
                   f"above the 200-day average, the next {FORWARD_SESSIONS} sessions (about "
                   "3 months) "
                   f"returned {summary(ups)} ({len(ups):,} sessions); after a close below, "
                   f"{summary(downs)} ({len(downs):,}). The windows overlap, so the sessions are "
                   "not independent; this is what history did, not a forecast.")
    else:
        out.append(f"No base rate is given: {label}'s history ({years:.1f} years with a 200-day "
                   "average) is too short for a stable above-versus-below comparison.")
    out.append(f"Data: Yahoo Finance daily closes for {ticker} since {_day(s.dates[0])} (price "
               "only, simple moving averages of closes; not Bitget's perpetual, whose history "
               "starts in May 2026). Not advice.")
    return out


# --------------------------------------------------------------------------- 4. volatility regime

_VOL_Q: Final = re.compile(r"\b(?:volatil\w*|vol|vix)\b", re.I)
_VOL_REGIME: Final = re.compile(
    r"\b(?:regime|high\s+or\s+low|low\s+or\s+high|elevated|subdued|compressed|calm|quiet|"
    r"percentile|historically\s+(?:high|low)|high|low)\b", re.I)
_OPTIONS_WORDS: Final = re.compile(
    r"\b(?:implied|iv\s+rank|skew|options?|straddle|earnings)\b", re.I)


def _percentile(sorted_values: list[float], x: float) -> float:
    below = bisect_right(sorted_values, x - 1e-12)
    upto = bisect_right(sorted_values, x + 1e-12)
    return 100.0 * (below + upto) / 2.0 / len(sorted_values)


def _quantile(sorted_values: list[float], q: float) -> float:
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _tercile(pct: float) -> str:
    return "low" if pct < 100 / 3 else "high" if pct > 200 / 3 else "normal"


def _rolling_vol(closes: list[float], window: int, per_year: int) -> list[float]:
    """30-day realised volatility (annualised, as a fraction) ending at each bar that has one."""
    rets = [math.log(b / a) for a, b in pairwise(closes)]
    out: list[float] = []
    for i in range(window, len(rets) + 1):
        w = rets[i - window:i]
        mean = sum(w) / window
        var = sum((r - mean) ** 2 for r in w) / (window - 1)
        out.append(math.sqrt(var * per_year))
    return out


def _vol_lines(text: str) -> list[str] | None:
    if not _VOL_Q.search(text) or not _VOL_REGIME.search(text) or _OPTIONS_WORDS.search(text):
        return None
    mentions = [m for m in _mentions(text) if not m[2].change]
    if not mentions:
        return None
    parts: list[str] = []
    notes: list[str] = []
    seen: set[str] = set()
    snapshot_date: date | None = None
    read: list[str] = []
    realised = False
    for _, _, asset in mentions:
        if asset.ticker in seen:
            continue
        seen.add(asset.ticker)
        try:
            s = _series(asset.ticker)
        except BreadthError as exc:
            parts.append(f"{asset.label}: not readable ({exc})")
            continue
        snapshot_date = max(snapshot_date or s.dates[-1], s.dates[-1])
        read.append(asset.ticker)
        if asset.ticker == "^VIX":
            level = s.closes[-1]
            year = sorted(s.closes[-252:])
            five = sorted(s.closes[-1260:])
            p1, p5 = _percentile(year, level), _percentile(five, level)
            parts.append(f"the VIX is {_num(level, 1)}, {_ordinal(round(p1))} percentile of the "
                         f"last year, {_tercile(p1)}")
            notes.append(f"VIX: 1-year terciles split at {_num(_quantile(year, 1 / 3), 1)} and "
                         f"{_num(_quantile(year, 2 / 3), 1)}; {_ordinal(round(p5))} percentile "
                         "over 5 years.")
            continue
        per_year = 365 if asset.crypto else 252
        vols = _rolling_vol(s.closes, VOL_WINDOW, per_year)
        recent = vols[-5 * per_year:]
        if len(recent) < per_year:
            parts.append(f"{asset.label}: under a year of history, no regime given")
            continue
        current = vols[-1]
        realised = True
        ordered = sorted(recent)
        pct = _percentile(ordered, current)
        low_cut, high_cut = _quantile(ordered, 1 / 3), _quantile(ordered, 2 / 3)
        parts.append(f"{asset.label} is {_tercile(pct)} ({current:.0%}, {_ordinal(round(pct))} "
                     "percentile)")
        name = asset.label[0].upper() + asset.label[1:]
        notes.append(f"{name}: 30-day realised volatility {current:.1%} annualised over "
                     f"{VOL_WINDOW} daily log returns to {_day(s.dates[-1])}; over "
                     f"{len(recent) / per_year:.1f} years the range was {min(recent):.0%} to "
                     f"{max(recent):.0%}; low below {low_cut:.0%}, high above {high_cut:.0%}.")
        if asset.ticker == "^GSPC" and "^VIX" not in {a.ticker for _, _, a in mentions}:
            try:
                v = _series("^VIX")
                read.append("^VIX")
                level = v.closes[-1]
                year = sorted(v.closes[-252:])
                p1 = _percentile(year, level)
                parts.append(f"the VIX (implied, S&P 500) is {_num(level, 1)}, "
                             f"{_ordinal(round(p1))} percentile of the last year, {_tercile(p1)}")
                notes.append(f"VIX: 1-year terciles split at {_num(_quantile(year, 1 / 3), 1)} and "
                             f"{_num(_quantile(year, 2 / 3), 1)}.")
            except BreadthError as exc:
                notes.append(f"VIX was not readable ({exc}).")
    if not parts:
        return None
    out = ["Bottom line: volatility right now: " + "; ".join(parts) + "."]
    out.extend(notes)
    if realised:
        out.append("Labels are terciles of each asset's own five-year history of 30-day realised "
                   "volatility (below the 33rd percentile low, above the 67th high), so 'high' "
                   "means high for that asset, not high against the others.")
    when = f" to {_day(snapshot_date)}" if snapshot_date else ""
    method = ("; realised volatility from daily log returns, annualised by 252 sessions "
              "(365 for crypto)" if realised else "")
    out.append(f"Data: Yahoo Finance daily closes for {', '.join(read)}{when}{method}. "
               "Not advice.")
    return out


# --------------------------------------------------------------------------- 5. valuation

_VAL_Q: Final = re.compile(
    r"\b(?:expensive|cheap\w*|overvalued|undervalued|valuation|valued|p\s*/\s*e|pe\s+ratio|"
    r"price[- ]to[- ]earnings|earnings\s+multiple|multiples?)\b", re.I)
_VAL_HISTORY: Final = re.compile(
    r"\b(?:(?:10|ten)[- ]?(?:year|yr)s?|decade|(?:historical|historic|long[- ]run|long[- ]term)"
    r"\s+average|own\s+history|history)\b", re.I)
_VAL_INDEX: Final = re.compile(
    r"\bs\s*&\s*p(?:\s*-?\s*500)?\b|\bspx\b|\bsp500\b|\bspy\b|\bnasdaq(?:\s*-?\s*100)?\b|"
    r"\bndx\b|\bqqq\b", re.I)
_NASDAQ: Final = re.compile(r"\bnasdaq(?:\s*-?\s*100)?\b|\bndx\b|\bqqq\b", re.I)
_FORWARD: Final = re.compile(
    r"\bforward\b|\bnext\s+(?:12|twelve)\b|\b(?:est\w*|expected)\s+earnings", re.I)
_ROW: Final = re.compile(
    r"<td>\s*([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})\s*</td>\s*<td>(.*?)</td>", re.S)
_VALUE: Final = re.compile(r"(\d+(?:\.\d+)?)\s*$")

_pe_cache: list[tuple[float, list[tuple[date, float]]]] = []


def parse_multpl(html: str) -> list[tuple[date, float]]:
    """(date, P/E) rows of multpl's by-month table, newest first."""
    rows: list[tuple[date, float]] = []
    for stamp, cell in _ROW.findall(html):
        m = _VALUE.search(re.sub(r"<[^>]+>|&#?\w+;", " ", cell).strip())
        if m is None:
            continue
        rows.append((datetime.strptime(re.sub(r"\s+", " ", stamp), "%b %d, %Y").date(),
                     float(m.group(1))))
    return rows


def _sp_pe_history() -> list[tuple[date, float]]:
    now = time.monotonic()
    with _lock:
        if _pe_cache and now - _pe_cache[0][0] < CACHE_SECONDS:
            return _pe_cache[0][1]
    try:
        html = http.fetch_text(MULTPL, timeout=TIMEOUT, headers=_UA)
    except http.RpcError as exc:
        raise BreadthError(f"multpl.com did not answer ({type(exc).__name__})") from exc
    rows = parse_multpl(html)
    if len(rows) < 240:
        raise BreadthError(f"multpl.com's P/E table had only {len(rows)} readable rows")
    with _lock:
        _pe_cache[:] = [(now, rows)]
    return rows


def _fund_multiple(ticker: str) -> tuple[float | None, float | None]:
    """(trailing P/E, holdings-weighted P/E from the earnings yield) of a fund, from Yahoo."""
    from argus.market.estimates import EstimatesSource

    summary = EstimatesSource().summary(ticker, "summaryDetail,topHoldings")
    trailing = ((summary.get("summaryDetail") or {}).get("trailingPE") or {}).get("raw")
    yield_ = (((summary.get("topHoldings") or {}).get("equityHoldings") or {})
              .get("priceToEarnings") or {}).get("raw")
    return (float(trailing) if trailing else None,
            1.0 / float(yield_) if yield_ else None)


def _valuation_lines(text: str) -> list[str] | None:
    if not (_VAL_Q.search(text) and _VAL_HISTORY.search(text) and _VAL_INDEX.search(text)):
        return None
    asks_nasdaq = bool(_NASDAQ.search(text))
    forward = bool(_FORWARD.search(text))
    sources: list[str] = []
    out: list[str] = []
    hist_line = ""
    hist_cmp = ""
    try:
        rows = _sp_pe_history()
        latest_day, latest = rows[0]
        window = [v for d, v in rows if d > _months_back(latest_day, 120)]
        avg10 = sum(window) / len(window)
        full = sum(v for _, v in rows) / len(rows)
        pct = _percentile(sorted(window), latest)
        gap = latest / avg10 - 1.0
        hist_cmp = (f"the S&P 500's own multiple is {latest:.1f}x ({_day(latest_day)}), "
                    f"{abs(gap):.0%} {'above' if gap > 0 else 'below'} its 10-year average of "
                    f"{avg10:.1f}x ({_ordinal(round(pct))} percentile of the {len(window)} "
                    "monthly readings)")
        hist_line = (f"S&P 500 trailing P/E history (multpl.com, price over trailing as-reported "
                     f"earnings): now {latest:.1f}x; 10-year average {avg10:.1f}x, median "
                     f"{sorted(window)[len(window) // 2]:.1f}x, range {min(window):.1f}x to "
                     f"{max(window):.1f}x; average since {rows[-1][0].year} {full:.1f}x. The most "
                     "recent months are multpl's estimates.")
        sources.append("multpl.com S&P 500 P/E by month")
    except BreadthError as exc:
        hist_line = f"The S&P 500's P/E history could not be read just now ({exc})."
    qqq: tuple[float | None, float | None] = (None, None)
    spy: tuple[float | None, float | None] = (None, None)
    try:
        qqq = _fund_multiple("QQQ")
        spy = _fund_multiple("SPY")
        sources.append("Yahoo Finance quoteSummary for QQQ and SPY")
    except Exception as exc:  # any Yahoo failure becomes a stated gap
        out.append(f"Yahoo's fund multiples could not be read just now ({type(exc).__name__}).")
    if asks_nasdaq:
        now_txt = (f"QQQ, which tracks the Nasdaq 100, trades on {qqq[0]:.1f}x trailing earnings "
                   "(Yahoo)" if qqq[0] else "QQQ's current multiple could not be read")
        head = ("Bottom line: that cannot be answered as asked, because no free source I can "
                "read publishes the Nasdaq 100's forward P/E or its 10-year history. "
                f"What can be measured: {now_txt}"
                + (f", and {qqq[1]:.1f}x on its holdings' earnings yield" if qqq[1] else "")
                + (f"; {hist_cmp}" if hist_cmp else "") + ".")
    else:
        head = ("Bottom line: " + (hist_cmp[0].upper() + hist_cmp[1:] if hist_cmp else
                                   "the S&P 500's multiple history could not be read just now")
                + ("; that is a trailing multiple, not a forward one" if forward else "") + ".")
    out.insert(0, head)
    if hist_line:
        out.append(hist_line)
    if qqq[0] and spy[0]:
        out.append(f"Same Yahoo basis, trailing P/E: QQQ {qqq[0]:.1f}x against SPY {spy[0]:.1f}x, "
                   f"a Nasdaq 100 premium of {qqq[0] / spy[0] - 1:.0%} to the S&P 500 today; the "
                   "premium's own history is not available.")
    if forward:
        out.append("Forward earnings: Yahoo returns no forward P/E for QQQ, SPY, ^NDX or ^GSPC "
                   "(the field is empty), and multpl's history is trailing; no forward figure "
                   "or forward history is given.")
    if asks_nasdaq:
        out.append("The 10-year average above is for the S&P 500; the Nasdaq 100's own 10-year "
                   "average is not available here, so the Nasdaq 100 is not called more or less "
                   "expensive than its own history.")
    out.append("Data: " + ("; ".join(sources) if sources else "no source answered")
               + ". Not advice.")
    return out


# --------------------------------------------------------------------------- 6. drivers

_DRIVER_LEAD: Final = re.compile(
    r"(?:\b(?:drivers?|influences?|factors?|forces?|determinants?|reasons?)\s+(?:of|for|behind|in)"
    r"|\b(?:drives?|driving|moves?|moving|explains?|affects?|influences?)"
    r")\s+(?:the\s+)?(?:price\s+of\s+)?(?:the\s+)?$", re.I)
_COMPARE: Final = re.compile(
    r"\b(?:or|vs\.?|versus|than|stronger|bigger|more|most|main|biggest|which|better)\b", re.I)
_DRIVER_WORD: Final = re.compile(
    r"\b(?:drivers?|driving|drives?|driven|moves?|moving|explains?|affects?|influences?|"
    r"determinants?)\b", re.I)


@dataclass(frozen=True, slots=True)
class Fit:
    beta: float
    corr: float
    r2: float


def _solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    n = len(b)
    m = [[*row, b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-14:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def _r2(y: list[float], xs: list[list[float]]) -> float | None:
    """R-squared of ``y`` on the columns ``xs`` (with an intercept)."""
    n = len(y)
    ym = sum(y) / n
    cols = [[v - sum(c) / n for v in c] for c in xs]
    yc = [v - ym for v in y]
    syy = sum(v * v for v in yc)
    if syy <= 0:
        return None
    gram = [[sum(p * q for p, q in zip(a, b, strict=True)) for b in cols] for a in cols]
    rhs = [sum(p * q for p, q in zip(a, yc, strict=True)) for a in cols]
    beta = _solve(gram, rhs)
    if beta is None:
        return None
    return max(0.0, min(1.0, sum(b * r for b, r in zip(beta, rhs, strict=True)) / syy))


def fit_one(y: list[float], x: list[float]) -> Fit | None:
    n = len(y)
    ym, xm = sum(y) / n, sum(x) / n
    sxy = sum((a - xm) * (b - ym) for a, b in zip(x, y, strict=True))
    sxx = sum((a - xm) ** 2 for a in x)
    syy = sum((b - ym) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None
    corr = sxy / math.sqrt(sxx * syy)
    return Fit(sxy / sxx, corr, corr * corr)


def _drivers_parse(text: str) -> tuple[Asset, list[Asset]] | None:
    mentions = _mentions(text)
    if len(mentions) < 2 or not _DRIVER_WORD.search(text):
        return None
    target: Asset | None = None
    for start, _, asset in mentions:
        if _DRIVER_LEAD.search(text[:start]):
            target = asset
            break
    if target is None:
        return None
    candidates = [a for _, _, a in mentions if a.ticker != target.ticker and a.ticker != "^VIX"]
    if not candidates or not (len(candidates) >= 2 or _COMPARE.search(text)):
        return None
    return target, candidates


def _drivers_lines(text: str) -> list[str] | None:
    parsed = _drivers_parse(text)
    if parsed is None:
        return None
    target, candidates = parsed
    today = _today()
    window = parse_window(text, today, default="3 months")
    try:
        data = {a.ticker: _series(a.ticker) for a in (target, *candidates)}
    except BreadthError as exc:
        return _failed(f"The price history for {target.label} and its candidate drivers", exc)
    shared = sorted(set.intersection(*(set(s.dates) for s in data.values())))
    common = [d for d in shared if d >= window.base]
    short = ""
    if len(common) < 21:
        short = (f"{window.label[0].upper()}{window.label[1:]} has only "
                 f"{max(len(common) - 1, 0)} daily returns shared by the series, too few for a "
                 "regression, so the past 3 months are used instead: ")
        window = Window("the past 3 months", _months_back(today, 3))
        common = [d for d in shared if d >= window.base]
        if len(common) < 21:
            return [f"Bottom line: only {max(len(common) - 1, 0)} daily returns are shared by "
                    f"{target.label} and its candidate drivers, too few for a regression.",
                    "Data: Yahoo Finance daily chart. Not advice."]
    lookup = {t: dict(zip(s.dates, s.closes, strict=True)) for t, s in data.items()}

    def moves(asset: Asset) -> list[float]:
        v = [lookup[asset.ticker][d] for d in common]
        if asset.change:
            return [b - a for a, b in pairwise(v)]
        return [100.0 * (b / a - 1.0) for a, b in pairwise(v)]

    y = moves(target)
    xs = {a.ticker: moves(a) for a in candidates}
    n = len(y)
    fits = {a.ticker: fit_one(y, xs[a.ticker]) for a in candidates}
    usable = [a for a in candidates if fits[a.ticker] is not None]
    if not usable:
        return [f"Bottom line: none of the candidate drivers moved enough over {window.label} "
                "to measure.", "Data: Yahoo Finance daily chart. Not advice."]
    r2s = {t: f.r2 for t, f in fits.items() if f is not None}
    best = max(usable, key=lambda a: r2s[a.ticker])
    joint = _r2(y, [xs[a.ticker] for a in usable]) if len(usable) > 1 else None
    unique: dict[str, float] = {}
    if joint is not None:
        for a in usable:
            rest = [xs[b.ticker] for b in usable if b.ticker != a.ticker]
            r = _r2(y, rest)
            if r is not None:
                unique[a.ticker] = joint - r
    noise = 2.0 / math.sqrt(n)

    def detail(a: Asset) -> str:
        f = fits[a.ticker]
        assert f is not None
        scale = 0.1 if a.change else 1.0
        unit = "+10bp" if a.change else "+1%"
        return (f"R-squared {f.r2:.2f}, correlation {f.corr:+.2f}, beta {f.beta * scale:+.2f} "
                f"({target.label} {f.beta * scale:+.2f}% per {unit} in {a.label})")

    def describe(a: Asset) -> str:
        return f"{a.label}: {detail(a)}"

    others = [a for a in usable if a is not best]
    runner = max(others, key=lambda a: r2s[a.ticker]) if others else None
    lead = detail(best)
    bf = fits[best.ticker]
    assert bf is not None
    if runner is not None:
        rf = fits[runner.ticker]
        assert rf is not None
        if bf.r2 - rf.r2 < 0.05:
            tail = (f"; the gap to {runner.label} (R-squared {rf.r2:.2f}) is under 0.05, too "
                    "close to call a clear winner")
        else:
            tail = f", ahead of {runner.label} at {rf.r2:.2f}"
    else:
        tail = ""
    out = [f"Bottom line: {short}over {window.label} ({n} daily returns to {_day(common[-1])}), "
           f"{best.label} was the stronger driver of {target.label} on this measure: {lead}"
           f"{tail}."]
    out.append("Each alone: " + "; ".join(describe(a) for a in usable) + ".")
    if joint is not None:
        out.append(f"Together they explain {joint:.0%} of {target.label}'s daily moves; unique "
                   "contribution (R-squared lost if removed): "
                   + ", ".join(f"{next(a.label for a in usable if a.ticker == t)} {u:.2f}"
                               for t, u in unique.items()) + ".")
    out.append(f"With {n} days a correlation within about +/-{noise:.2f} of zero cannot be told "
               "from noise. The series are daily closes of different markets that stop trading "
               "at different times, which understates every link; this is association, not proof "
               "of cause.")
    tickers = ", ".join(a.ticker for a in (target, *candidates))
    yield_note = ("; the 10-year yield enters as its daily change in percentage points"
                  if any(a.change for a in candidates) else "")
    out.append(f"Data: Yahoo Finance daily closes for {tickers}; returns are close-to-close"
               f"{yield_note}. Not advice.")
    return out


# --------------------------------------------------------------------------- entry point

_READERS: Final = (_drivers_lines, _mag7_lines, _valuation_lines, _regime_lines, _vol_lines,
                   _sector_lines)


def lines(text: str, prior: Sequence[str] = ()) -> list[str] | None:
    """The answer when ``text`` is one of the six market-breadth questions, else None.

    ``prior`` is accepted so the dispatcher can call every reader the same way; each question
    here names its own assets and period, so it is not read."""
    del prior
    for reader in _READERS:
        got = reader(text)
        if got is not None:
            return got
    return None
