"""Research questions a professional asks that the console answered as something else.

Round 45 (hostile review, 2026-10-06) found eight questions answered by the wrong engine, or by
none. Each part below says which finding it fixes. All of them are computed from daily closes the
console can read (Yahoo Finance for stocks, coins, gold and the dollar index; Bitget daily candles
for the buy-schedule test; Bitget's public tickers for funding), on the days every series traded,
and each says what it measured and what it could not.

- Lead-lag thesis (M2): "semiconductor stocks lead Bitcoin by a few days" is tested by the
  cross-correlation of daily returns at lags -5 to +5 and a permutation test that shifts one
  series around a circle (keeping its autocorrelation) and asks how often the best lag out of the
  tested set is as strong by chance. The follow-up "what lag maximizes the correlation, and is it
  significant?" is answered from the same analysis, found through the earlier question.
- Buy schedule (M3): "bought ETH every Monday for the last year with $200" against putting it
  all in at the start, with the drawdown of each. "What if Bitcoin instead?" reuses the schedule;
  "which had the worse drawdown?" compares the schedules of the coins in the thread.
- Rebalance to a cap (M4): proposed weights under the cap (the excess goes to the names still
  under it, the rest to cash) and the worst drawdown of the original and capped book over the same
  window, in the same turn and in the follow-up that asks for it.
- Concentration (M5): each holding's share of the book's risk (weight x covariance share), the
  biggest, and the pairwise correlations of the past year, with the threshold used to call one.
- Volatility-adjusted comparison (M11): twelve-month return, volatility, return per unit of
  volatility and Sharpe against the Treasury bill, then a verdict.
- Breadth (M12): the share of S&P 500 members above their 50-day average, from the member list
  and each member's closes; when those cannot be read, the equal-weight against cap-weight proxy,
  labelled as one.
- Rolling correlation (M13): rolling correlation of daily returns, BTC against the ICE dollar
  index (Yahoo DX-Y.NYB), latest and range over the period.
- Funding extremes (M14): the most extreme funding on either side among liquid perpetuals, ranked
  on the cost per day (contracts settle every one, four or eight hours), with the contract's own
  symbol, and a squeeze reading that states what the data shows and predicts nothing.

Everything returns None for a question it does not own, so the dispatcher can try it first.
"""

from __future__ import annotations

import math
import re
import statistics
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Final

import numpy as np
from numpy.typing import NDArray

from argus.lui.numbers import price, sig
from argus.lui.trace import trace_module

LAGS: Final = 5
"""The lead-lag test covers lags -LAGS to +LAGS."""
PERMUTATIONS: Final = 3000
TRADING_DAYS: Final = 252
LIQUID_USD: Final = 5_000_000.0
"""A perpetual counts as liquid for the funding ranking at this 24-hour volume (the filter the
console's other funding answer uses)."""
HIGH_CORR, LOW_CORR = 0.60, 0.30
"""Pairs at or above HIGH_CORR are called highly correlated, below LOW_CORR weakly."""


class Unavailable(RuntimeError):
    """The data a question needs could not be read; the answer says so in one line."""


# --------------------------------------------------------------------------- data


def _today() -> date:
    return datetime.now(UTC).date()


_ALIAS: Final = {"XAU": "GC=F", "GOLD": "GC=F", "PAXG": "GC=F", "XAUT": "GC=F", "XAG": "SI=F",
                 "SP500": "^GSPC", "SPX": "^GSPC", "NAS100": "^NDX", "NDX": "^NDX",
                 "DXY": "DX-Y.NYB"}
"""Bitget perpetual bases whose Yahoo series is a commodity or an index, so a name read from the
text and the same asset in the table of named assets are one series, not two."""


def yahoo_ticker(symbol: str) -> str:
    """The Yahoo ticker for a Bitget perpetual name or for a ticker already in Yahoo's form."""
    if not symbol.endswith("USDT"):
        return symbol
    base = symbol.removesuffix("USDT")
    if base in _ALIAS:
        return _ALIAS[base]
    from argus.lui.research.parse import is_us_equity

    return base if is_us_equity(symbol) else f"{base}-USD"


def _closes(ticker: str) -> dict[date, float]:
    from argus.market import equity_history

    try:
        return {d.day: float(d.close) for d in equity_history.daily(ticker) if d.close > 0}
    except Exception as exc:
        raise Unavailable(f"Yahoo's daily closes for {ticker} did not arrive "
                          f"({type(exc).__name__})") from exc


@dataclass(frozen=True, slots=True)
class Panel:
    """Closes of several series on the days all of them have one."""

    days: list[date]
    close: dict[str, list[float]]

    def returns(self, ticker: str) -> list[float]:
        c = self.close[ticker]
        return [c[i] / c[i - 1] - 1.0 for i in range(1, len(c))]


def panel(tickers: Sequence[str], start: date, end: date | None = None) -> Panel:
    """Aligned closes from the first shared day on or after ``start``. Returns between aligned
    days span weekends and holidays for a coin, so a Friday-to-Monday coin move counts once."""
    series = {t: _closes(t) for t in tickers}
    shared = set.intersection(*(set(s) for s in series.values()))
    days = sorted(d for d in shared if d >= start and (end is None or d <= end))
    if len(days) < 30:
        raise Unavailable(f"only {len(days)} days on which {', '.join(tickers)} all have a close "
                          f"since {start:%d %b %Y}")
    return Panel(days, {t: [series[t][d] for d in days] for t in tickers})


def _corr(a: Sequence[float], b: Sequence[float]) -> float:
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((x - mb) ** 2 for x in b))
    if sa == 0 or sb == 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / (sa * sb)


def _max_drawdown(values: Sequence[float]) -> tuple[float, int, int]:
    """(depth as a negative fraction, index of the peak, index of the trough)."""
    peak_i, worst, w_peak, w_trough = 0, 0.0, 0, 0
    for i, v in enumerate(values):
        if v > values[peak_i]:
            peak_i = i
        dd = v / values[peak_i] - 1.0
        if dd < worst:
            worst, w_peak, w_trough = dd, peak_i, i
    return worst, w_peak, w_trough


def _p(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.1 else f"p={p:.2f}"


def _span(days: Sequence[date]) -> str:
    return f"{days[0]:%d %b %Y} to {days[-1]:%d %b %Y}"


def _fail(why: str, example: str = "") -> list[str]:
    return [f"Bottom line: not computed — {why}." + (f" {example}" if example else "")]


# --------------------------------------------------------------------------- named assets

_ASSETS: Final[tuple[tuple[re.Pattern[str], str, str], ...]] = tuple(
    (re.compile(p, re.I), t, lab) for p, t, lab in (
        (r"\bsoxx\b", "SOXX", "semiconductor stocks (SOXX)"),
        (r"\bsemiconductors?\b|\bsemis\b|\bchip\s+(?:stocks|makers|companies|sector|names)\b|"
         r"\bsmh\b|\bsox\b", "SMH", "semiconductor stocks (SMH)"),
        (r"\bnasdaq(?:\s*-?\s*100)?\b|\bndx\b|\bqqq\b", "^NDX", "the Nasdaq 100"),
        (r"\bs\s*&\s*p(?:\s*-?\s*500)?\b|\bspx\b|\bsp500\b|\bspy\b", "^GSPC", "the S&P 500"),
        (r"\bgold\b|\bxau\b", "GC=F", "gold"),
        (r"\bsilver\b|\bxag\b", "SI=F", "silver"),
        (r"\bcopper\b", "HG=F", "copper"),
        (r"\bdxy\b|\bdollar\s+index\b|\busd\s+index\b|\bdollar\b", "DX-Y.NYB",
         "the US dollar index (ICE DXY)"),
        (r"\bbitcoin\b|\bbtc\b", "BTC-USD", "Bitcoin"),
        (r"\bethereum\b|\bether\b|\beth\b", "ETH-USD", "Ethereum"),
        (r"\bsolana\b|\bsol\b", "SOL-USD", "Solana"),
    ))


def _named(text: str) -> list[tuple[str, str]]:
    """(Yahoo ticker, label) for each asset the text names, in the order named."""
    from argus.lui.research import research_symbols

    found: list[tuple[int, str, str]] = []
    for pattern, ticker, label in _ASSETS:
        m = pattern.search(text)
        if m and all(f[1] != ticker for f in found):
            found.append((m.start(), ticker, label))
    for symbol in research_symbols(text)[0]:
        ticker = yahoo_ticker(symbol)
        if any(pattern.fullmatch(symbol.removesuffix("USDT")) for pattern, _, _ in _ASSETS):
            continue  # an asset the table already names ("SPY" is the S&P 500 here)
        if all(f[1] != ticker for f in found):
            found.append((len(text), ticker, symbol.removesuffix("USDT")))
    found.sort(key=lambda f: f[0])
    return [(t, lab) for _, t, lab in found]


def _short(ticker: str) -> str:
    return ticker.removesuffix("-USD")


# --------------------------------------------------------------------------- 1. lead-lag (M2)

_LEAD_VERB: Final = re.compile(r"\b(?:lead|leads|led|leading|front[- ]?runs?|front[- ]?running|"
                               r"precede[sd]?|predict(?:s|ed|ing)?)\b", re.I)
_LAG_Q: Final = re.compile(r"\blag\b[^?]{0,60}\b(?:correlation|significan\w*|maximi[sz]\w*)|"
                           r"\b(?:correlation|significan\w*)\b[^?]{0,60}\blag\b|"
                           r"\bwhat\s+lag\b|\bhow\s+many\s+days\b[^?]{0,40}\blead\w*", re.I)


@dataclass(frozen=True, slots=True)
class Thesis:
    leader: str
    leader_label: str
    follower: str
    follower_label: str
    years: float


def _thesis(text: str) -> Thesis | None:
    """"semiconductor stocks lead Bitcoin by a few days": the asset before the verb leads, the
    one after follows."""
    verb = _LEAD_VERB.search(text)
    if verb is None:
        return None
    before, after = _named(text[:verb.start()]), _named(text[verb.end():])
    if not before or not after:
        return None
    (lead, lead_label), (follow, follow_label) = before[-1], after[0]
    if lead == follow:
        return None
    years = 1.0 if re.search(r"\b(?:last|past|one|1)\s+year\b|\bover\s+a\s+year\b", text, re.I) \
        else 2.0
    return Thesis(lead, lead_label, follow, follow_label, years)


def _xcorr(x: NDArray[np.float64], y: NDArray[np.float64], k: int) -> float:
    """Correlation of ``x`` on day t with ``y`` on day t+k (positive k: ``x`` first)."""
    n = len(x)
    a, b = (x[:n - k], y[k:]) if k >= 0 else (x[-k:], y[:n + k])
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / den if den > 0 else 0.0


@dataclass(frozen=True, slots=True)
class LeadLag:
    days: list[date]
    r: dict[int, float]
    best_all: int
    p_all: float
    best_lead: int
    p_lead: float
    noise: float
    """Half-width of the 95% band for a single lag if nothing were related: 1.96 / sqrt(n)."""


def lead_lag(x: Sequence[float], y: Sequence[float], days: list[date]) -> LeadLag:
    """Cross-correlation of two return series at lags -LAGS..+LAGS and permutation p-values.

    The null shifts ``y`` around a circle by a random amount (never within 10 days of its true
    position), which keeps each series' own autocorrelation and fat tails and breaks only the
    link between them. The statistic is the largest absolute correlation over the lags tested
    (all eleven, or the five where ``x`` leads), so searching the lags is paid for in the p-value
    rather than ignored. Seeded: the same data gives the same answer."""
    xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    lags = range(-LAGS, LAGS + 1)
    r = {k: _xcorr(xa, ya, k) for k in lags}
    leads = range(1, LAGS + 1)
    obs_all = max(abs(v) for v in r.values())
    obs_lead = max(abs(r[k]) for k in leads)
    rng = np.random.default_rng(45)
    n = len(xa)
    hit_all = hit_lead = 0
    for s in rng.integers(10, n - 10, size=PERMUTATIONS):
        shifted = np.roll(ya, int(s))
        rs = {k: abs(_xcorr(xa, shifted, k)) for k in lags}
        hit_all += max(rs.values()) >= obs_all
        hit_lead += max(rs[k] for k in leads) >= obs_lead
    return LeadLag(
        days=days, r=r,
        best_all=max(lags, key=lambda k: abs(r[k])),
        p_all=(1 + hit_all) / (1 + PERMUTATIONS),
        best_lead=max(leads, key=lambda k: abs(r[k])),
        p_lead=(1 + hit_lead) / (1 + PERMUTATIONS),
        noise=1.96 / math.sqrt(n))


def _lead_text(th: Thesis, a: LeadLag, *, follow_up: bool) -> list[str]:
    n_days = len(a.days) - 1
    k, r = a.best_lead, a.r[a.best_lead]
    supported = a.p_lead < 0.05
    sign = "moves with" if r > 0 else "moves against"
    zero = a.r[0]
    window = f"{_span(a.days)}, {n_days} daily returns"
    if follow_up:
        kb, rb = a.best_all, a.r[a.best_all]
        where = ("the same day (lag 0)" if kb == 0 else
                 f"{abs(kb)} trading days with {th.leader_label if kb > 0 else th.follower_label}"
                 f" first")
        lead = (f"Bottom line: the correlation is strongest at {where}: r={rb:+.2f}, "
                f"{'significant' if a.p_all < 0.05 else 'not significant'} "
                f"({_p(a.p_all)} for the best of the eleven lags from -5 to +5).")
        out = [lead,
               f"For the thesis itself, {th.leader_label} first and {th.follower_label} later: "
               f"the best of lags 1 to 5 is {k} day{'s' if k != 1 else ''}, r={r:+.2f}, "
               f"{_p(a.p_lead)} after testing those five — "
               + ("significant." if supported else "not significant, so no lead shows.")]
    elif supported:
        lead = (f"Bottom line: yes, in this data — {th.leader_label} led {th.follower_label} by "
                f"{k} trading day{'s' if k != 1 else ''}: r={r:+.2f} ({th.follower_label} "
                f"{sign} it {k} day{'s' if k != 1 else ''} later), {_p(a.p_lead)} after testing "
                f"the five lags 1 to 5.")
        out = [lead]
    else:
        lead = (f"Bottom line: no — the data does not show {th.leader_label} leading "
                f"{th.follower_label}. The strongest lead is {k} day{'s' if k != 1 else ''} at "
                f"r={r:+.2f}, and a lead that strong turns up by chance in {a.p_lead:.0%} of "
                f"shuffled tests ({_p(a.p_lead)}, five lags tried).")
        out = [lead]
    table = ", ".join(f"{k2:+d}: {v:+.2f}" for k2, v in a.r.items())
    out += [f"Correlation of daily returns by lag, {th.leader_label} on day t with "
            f"{th.follower_label} on day t+lag (positive lag = {th.leader_label} first): {table}.",
            f"Same-day (lag 0) correlation is r={zero:+.2f}"
            + ("; whatever link there is, is contemporaneous, not a lead. "
               if abs(zero) > a.noise else ". ")
            + f"A single lag needs |r| above {a.noise:.2f} to clear noise on its own, before "
            f"allowing for trying eleven. Window: {window}.",
            f"Data: Yahoo Finance daily closes ({th.leader} and {th.follower}) "
            f"on the days both have a close, so a weekend coin "
            f"move counts once; significance by {PERMUTATIONS} circular-shift permutations "
            f"(autocorrelation kept), seeded. Correlation is not causation; two years is one "
            f"market regime."]
    return out


def _lead_lag_lines(text: str, prior: Sequence[str]) -> list[str] | None:
    thesis = _thesis(text)
    follow_up = thesis is None
    if thesis is None:
        earlier = [t for t in (_thesis(q) for q in prior[-3:]) if t is not None]
        if not _LAG_Q.search(text) or not earlier:
            return None
        thesis = earlier[-1]
    start = _today() - timedelta(days=int(365 * thesis.years))
    try:
        p = panel([thesis.leader, thesis.follower], start)
    except Unavailable as exc:
        return _fail(str(exc))
    result = lead_lag(p.returns(thesis.leader), p.returns(thesis.follower), p.days)
    return _lead_text(thesis, result, follow_up=follow_up)


# --------------------------------------------------------------------------- 2. buy schedule (M3)

_WEEKDAYS: Final = ("mon", "tues", "wednes", "thurs", "fri", "satur", "sun")
_BUY_VERB: Final = re.compile(r"\b(?:buy|bought|buying|invest\w*|put|dca|dollar[\s-]cost|"
                              r"purchas\w*|accumulat\w*|stack\w*)\b", re.I)
_CADENCE: Final = re.compile(
    r"\b(?:every|each|on)\s+(?P<wd>mon|tues|wednes|thurs|fri|satur|sun)days?\b|"
    r"\b(?P<fn>every\s+(?:two|2)\s+weeks|fortnight\w*|bi[\s-]?weekly)\b|"
    r"\b(?P<wk>every\s+week|weekly|each\s+week|a\s+week|per\s+week)\b|"
    r"\b(?P<dy>every\s+day|daily|each\s+day|a\s+day|per\s+day)\b|"
    r"\b(?P<mo>every\s+month|monthly|each\s+month|a\s+month|per\s+month)\b", re.I)
_AMOUNT: Final = re.compile(r"\$\s?(?P<a>\d[\d,]*(?:\.\d+)?)\s*(?P<k>k)?\b|"
                            r"\b(?P<b>\d[\d,]*(?:\.\d+)?)\s*(?:dollars|usd|usdt|bucks)\b", re.I)
_NUMBER_WORDS: Final = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12}
_PERIOD: Final = re.compile(
    r"\b(?:last|past|previous|preceding)\s+(?P<n>\d+|a|an|one|two|three|four|five|six|seven|"
    r"eight|nine|ten|twelve)?\s*(?P<u>year|month|week)s?\b|"
    r"\bfor\s+(?:the\s+)?(?:a\s+|one\s+|whole\s+)?(?P<u2>year)\b|"
    r"\b(?:for|over|during|across)\s+(?:the\s+)?(?P<n3>\d+|a|an|one|two|three|four|five|six|"
    r"seven|eight|nine|ten|twelve)\s+(?P<u3>year|month|week)s?\b", re.I)
_MONTHS: Final = ("january", "february", "march", "april", "may", "june", "july", "august",
                  "september", "october", "november", "december")
_SINCE: Final = re.compile(r"\bsince\s+(?:(?P<m>" + "|".join(_MONTHS) + r")\s+)?(?P<y>20\d\d)\b",
                           re.I)


@dataclass(frozen=True, slots=True)
class Schedule:
    symbol: str
    amount: float
    cadence: str
    weekday: int
    days: int
    stated_period: bool
    since: str = ""
    """"1 Jan 2024" when the period was given as a start date rather than a length."""

    @property
    def phrase(self) -> str:
        if self.cadence == "week":
            return (f"every {_WEEKDAYS[self.weekday].title()}day" if self.weekday >= 0
                    else "every week (Mondays)")
        return {"day": "every day", "fortnight": "every two weeks",
                "month": "on the first of every month"}[self.cadence]

    @property
    def span(self) -> str:
        if self.since:
            return f"the period since {self.since}"
        if self.days % 365 == 0:
            n = self.days // 365
            return "the last year" if n == 1 else f"the last {n} years"
        months = round(self.days / 30.4375)
        if months >= 1 and self.days == _unit_days(months, "month"):
            return f"the last {months} months"
        return f"the last {self.days} days"


def _unit_days(count: int, unit: str) -> int:
    """Calendar days in ``count`` years (365), months (365 / 12 each, so twelve make a year) or
    weeks."""
    return {"year": 365 * count, "month": round(30.4375 * count), "week": 7 * count}[unit]


def _period_days(text: str) -> tuple[int, str] | None:
    """(days, start label) of the period the text states: a length ("the last 6 months") or a
    start ("since January 2024", the label then being "1 Jan 2024")."""
    m = _PERIOD.search(text)
    if m:
        unit = (m.group("u") or m.group("u2") or m.group("u3")).lower()
        raw = (m.group("n") or m.group("n3") or "1").lower()
        count = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw, 1)
        return _unit_days(count, unit), ""
    s = _SINCE.search(text)
    if s:
        month = _MONTHS.index(s.group("m").lower()) + 1 if s.group("m") else 1
        start = date(int(s.group("y")), month, 1)
        days = (_today() - start).days
        return (days, f"{start.day} {start:%b %Y}") if days > 0 else None
    return None


def _schedule(text: str, base: Schedule | None = None) -> Schedule | None:
    """A buy schedule stated in ``text``; pieces it leaves out come from ``base`` (the thread)."""
    if base is None and not _BUY_VERB.search(text):
        return None
    cadence = _CADENCE.search(text)
    amount = _AMOUNT.search(text)
    named = _named(text)
    coin = next((t for t, _ in named if t.endswith("-USD")), None)
    if coin is None and base is None:
        return None
    if cadence is None and amount is None and base is None:
        return None
    kind, weekday = (base.cadence, base.weekday) if base else ("week", 0)
    if cadence is not None:
        if cadence.group("wd"):
            kind, weekday = "week", _WEEKDAYS.index(cadence.group("wd").lower())
        elif cadence.group("fn"):
            kind = "fortnight"
        elif cadence.group("wk"):
            kind, weekday = "week", -1
        elif cadence.group("dy"):
            kind = "day"
        else:
            kind = "month"
    elif base is None:
        return None
    value = base.amount if base else 0.0
    if amount is not None:
        raw = float((amount.group("a") or amount.group("b")).replace(",", ""))
        value = raw * (1000 if (amount.group("k") or "").lower() == "k" else 1)
    found = _period_days(text)
    days, since = found if found else (0, "")
    return Schedule(symbol=(coin.removesuffix("-USD") + "USDT") if coin else
                    (base.symbol if base else ""),
                    amount=value, cadence=kind, weekday=weekday,
                    days=days if days else (base.days if base else 365),
                    stated_period=found is not None or (base.stated_period if base else False),
                    since=since if found else (base.since if base else ""))


@dataclass(frozen=True, slots=True)
class Plan:
    """What one schedule would have done."""

    schedule: Schedule
    first: date
    last: date
    buys: int
    invested: float
    coins: float
    value_now: float
    lump_value: float
    dca_drawdown: float
    dca_dd_date: date
    """The day of the low (the peak it is measured from moves with every purchase)."""
    lump_drawdown: float
    lump_dd_dates: tuple[date, date]
    worst_below_cost: float
    price_now: float
    avg_cost: float
    first_open: float


def _daily_bars(symbol: str, days: int) -> list[tuple[date, float, float]]:
    """(day, open, close) from Bitget's daily candles, oldest first."""
    from argus.market import history

    try:
        rows = history.fetch_range(symbol, days=days + 10, interval="1Dutc")
    except Exception as exc:
        raise Unavailable(f"Bitget's daily candles for {symbol} did not arrive "
                          f"({type(exc).__name__})") from exc
    bars = [(c.ts.date(), float(c.open), float(c.close)) for c in rows if c.close > 0]
    if len(bars) < 14:
        raise Unavailable(f"Bitget returned only {len(bars)} daily candles for {symbol}")
    return bars


def simulate(schedule: Schedule, bars: Sequence[tuple[date, float, float]],
             now_price: float | None = None) -> Plan:
    """Buy ``amount`` on each scheduled day at that day's open; value every day at its close.

    The lump sum puts the same total in at the first purchase date's open. Drawdown is
    flow-adjusted: money added by a purchase raises the peak it is measured from, so a fresh
    purchase cannot hide an earlier fall, and the cash not yet invested is not counted."""
    start = _today() - timedelta(days=schedule.days)
    use = [b for b in bars if b[0] >= start]
    if len(use) < 7:
        raise Unavailable("fewer than a week of daily candles in that period")
    buy_days: set[date] = set()
    month_start: dict[tuple[int, int], date] = {}
    for day, _o, _c in bars:
        month_start.setdefault((day.year, day.month), day)
    firsts = set(month_start.values())
    next_fortnight: date | None = None
    for day, _o, _c in use:
        if (schedule.cadence == "day"
                or (schedule.cadence == "week" and day.weekday() == max(schedule.weekday, 0))
                or (schedule.cadence == "month" and day in firsts)):
            buy_days.add(day)
        elif schedule.cadence == "fortnight":
            if next_fortnight is None and day.weekday() == max(schedule.weekday, 0):
                next_fortnight = day
            if next_fortnight is not None and day >= next_fortnight:
                buy_days.add(day)
                next_fortnight = day + timedelta(days=14)
    if not buy_days:
        raise Unavailable("no purchase day fell inside that period")
    coins = invested = 0.0
    peak = 0.0
    worst, w_date = 0.0, use[0][0]
    worst_cost = 0.0
    buys = 0
    first: date | None = None
    for day, opened, closed in use:
        if day in buy_days:
            coins += schedule.amount / opened
            invested += schedule.amount
            peak += schedule.amount
            buys += 1
            first = first or day
        if first is None:
            continue
        value = coins * closed
        if value > peak:
            peak = value
        dd = value / peak - 1.0
        if dd < worst:
            worst, w_date = dd, day
        worst_cost = min(worst_cost, value / invested - 1.0)
    assert first is not None
    after = [b for b in use if b[0] >= first]
    lump_coins = invested / after[0][1]
    lump_series = [lump_coins * b[2] for b in after]
    lump_dd, p_i, t_i = _max_drawdown([invested, *lump_series])
    last_close = use[-1][2]
    px = now_price if now_price and now_price > 0 else last_close
    return Plan(schedule=schedule, first=first, last=use[-1][0], buys=buys, invested=invested,
                coins=coins, value_now=coins * px, lump_value=lump_coins * px,
                dca_drawdown=worst, dca_dd_date=w_date, lump_drawdown=lump_dd,
                lump_dd_dates=(after[max(p_i - 1, 0)][0], after[max(t_i - 1, 0)][0]),
                worst_below_cost=worst_cost, price_now=px, avg_cost=invested / coins,
                first_open=after[0][1])


def _plan_for(schedule: Schedule) -> Plan:
    from argus.lui.research.parse import last_price

    bars = _daily_bars(schedule.symbol, schedule.days)
    return simulate(schedule, bars, last_price(schedule.symbol))


def _usd(x: float) -> str:
    return f"${x:,.0f}" if abs(x) >= 100 or x == round(x) else f"${x:,.2f}"


def _pct(x: float) -> str:
    return f"{x:+.1%}"


def _plan_lines(plan: Plan, *, assumed: bool) -> list[str]:
    s = plan.schedule
    coin = s.symbol.removesuffix("USDT")
    dca_ret = plan.value_now / plan.invested - 1.0
    lump_ret = plan.lump_value / plan.invested - 1.0
    gap = plan.value_now - plan.lump_value
    better = "ahead of" if gap > 0 else "behind"
    out = [
        f"Bottom line: buying {_usd(s.amount)} of {coin} {s.phrase} over {s.span} "
        f"({plan.buys} buys, {_usd(plan.invested)} in) would now be worth {_usd(plan.value_now)}, "
        f"{_pct(dca_ret)}; the same {_usd(plan.invested)} put in all at the start would be worth "
        f"{_usd(plan.lump_value)}, {_pct(lump_ret)} — the schedule finishes {_usd(abs(gap))} "
        f"{better} the lump sum.",
        f"The schedule bought {sig(plan.coins)} {coin} at an average {price(plan.avg_cost)}; the "
        f"lump sum bought at {price(plan.first_open)} on {plan.first:%d %b %Y}; the price now is "
        f"{price(plan.price_now)}.",
        f"Worst drawdown of the position, measured from its own peak and adjusting for new money: "
        f"schedule {plan.dca_drawdown:.1%} (low on {plan.dca_dd_date:%d %b %Y}), lump sum "
        f"{plan.lump_drawdown:.1%} ({plan.lump_dd_dates[0]:%d %b %Y} peak to "
        f"{plan.lump_dd_dates[1]:%d %b %Y}). At its worst point "
        f"the schedule stood {plan.worst_below_cost:.1%} against the money it had put in.",
        ("Period not stated, so the last 12 months. " if assumed else "")
        + f"Data: Bitget daily {coin}USDT perpetual candles (UTC days), each purchase at that "
        f"day's open, valued at the latest price; no fees, funding or slippage counted. "
        f"Days are UTC days; the first purchase was {plan.first:%A %d %b %Y}. A past result, not a "
        f"forecast."]
    return [x for x in out if x]


_SWAP: Final = re.compile(r"\binstead\b|\bwhat\s+if\s+i\s+(?:did|do|had|bought|invested|used|"
                          r"went)\b|\bsame\s+(?:for|on|with|thing)\b|\band\s+(?:on|for|with)\b",
                          re.I)
_DD_ASK: Final = re.compile(r"\bdrawdown\b|\bworst\s+(?:fall|loss|decline)\b|\bdeeper\b", re.I)


def _thread_schedules(prior: Sequence[str]) -> list[Schedule]:
    """Each schedule stated or implied in the thread, oldest first, a swap carrying the last one's
    terms."""
    out: list[Schedule] = []
    for q in prior:
        base = out[-1] if out else None
        s = _schedule(q, None)
        if s is not None and (_AMOUNT.search(q) and _CADENCE.search(q)):
            out.append(s)
        elif base is not None and _SWAP.search(q) and any(
                t.endswith("-USD") for t, _ in _named(q)):
            swapped = _schedule(q, base)
            if swapped is not None:
                out.append(swapped)
    return out


def _dca_lines(text: str, prior: Sequence[str]) -> list[str] | None:
    has_cadence = _CADENCE.search(text) is not None
    stated = _schedule(text) if (has_cadence and _BUY_VERB.search(text)) else None
    thread = _thread_schedules(prior[-6:])
    if stated is not None:
        if _AMOUNT.search(text) is None:
            if not re.search(r"\bwhat\s+would\b|\bwould\s+have\b|\bcompar\w*|\bif\s+i\b", text,
                             re.I):
                return None
            return _fail("say how much each purchase is", "Example: \"bought ETH every Monday "
                         "with $200 for the last year\".")
        try:
            return _plan_lines(_plan_for(stated), assumed=not stated.stated_period)
        except Unavailable as exc:
            return _fail(str(exc))
    if not thread:
        return None
    if _DD_ASK.search(text) and re.search(
            r"\bwors\w*|\bbigger\b|\blarger\b|\bdeeper\b|\bwhich\b|\bmore\b", text, re.I) \
            and not _named(text):
        base = thread[-1]
        coins: list[str] = []
        for s in [*thread]:
            if s.symbol not in coins:
                coins.append(s.symbol)
        try:
            plans = [_plan_for(Schedule(c, base.amount, base.cadence, base.weekday, base.days,
                                        base.stated_period, base.since)) for c in coins]
        except Unavailable as exc:
            return _fail(str(exc))
        return _drawdown_lines(plans)
    if _SWAP.search(text) and any(t.endswith("-USD") for t, _ in _named(text)) \
            and re.search(r"\?|\bdid\b|\bwhat\s+if\b", text, re.I) and not re.search(
            r"\d\s*%", text):
        swapped = _schedule(text, thread[-1])
        if swapped is None:
            return None
        try:
            return _plan_lines(_plan_for(swapped), assumed=False)
        except Unavailable as exc:
            return _fail(str(exc))
    return None


def _drawdown_lines(plans: list[Plan]) -> list[str]:
    s = plans[0].schedule
    names = [p.schedule.symbol.removesuffix("USDT") for p in plans]
    worst = min(plans, key=lambda p: p.dca_drawdown)
    worst_name = worst.schedule.symbol.removesuffix("USDT")
    if len(plans) == 1:
        p = plans[0]
        head = (f"Bottom line: {worst_name}'s schedule had a worst drawdown of "
                f"{p.dca_drawdown:.1%} against {p.lump_drawdown:.1%} for the lump sum, over "
                f"{s.span} ({_usd(s.amount)} {s.phrase}).")
    else:
        rows = ", ".join(f"{n} {p.dca_drawdown:.1%}" for n, p in zip(names, plans, strict=True))
        lumps = ", ".join(f"{n} {p.lump_drawdown:.1%}" for n, p in zip(names, plans, strict=True))
        lump_worst = min(plans, key=lambda p: p.lump_drawdown).schedule.symbol.removesuffix("USDT")
        head = (f"Bottom line: {worst_name} had the worse drawdown on the same schedule "
                f"({_usd(s.amount)} {s.phrase} over {s.span}): {rows}. As a lump sum it was "
                f"{lumps}" + (f" — {lump_worst} worse there too." if lump_worst == worst_name
                              else f" — {lump_worst} was worse as a lump sum.")
                )
    detail = [f"{p.schedule.symbol.removesuffix('USDT')}: schedule {p.dca_drawdown:.1%} "
              f"(low on {p.dca_dd_date:%d %b %Y}), lump sum "
              f"{p.lump_drawdown:.1%}, worst point against money in {p.worst_below_cost:.1%}, "
              f"{_usd(p.value_now)} now on {_usd(p.invested)}." for p in plans]
    return [head, *detail,
            "Drawdown is measured from the position's own peak and adjusts for money added by each "
            "purchase, so it is not the coin's price fall; a schedule's early buys are small, so "
            "it usually draws down less than the lump sum. Data: Bitget daily perpetual candles, "
            "each purchase at that day's open; no fees counted."]


# --------------------------------------------------------------------------- 3. book weights

_CAP: Final = re.compile(
    r"\b(?:caps?|capped|capping|limit(?:s|ed|ing)?|max(?:imum)?|ceiling)\b[^?.%]{0,50}?"
    r"(?P<a>\d+(?:\.\d+)?)\s*%|"
    r"\b(?:no|any|each|every)\s+(?:single\s+)?(?:position|name|holding|asset)\b[^?.%]{0,40}?"
    r"\b(?:above|over|more\s+than|exceed\w*|greater\s+than)\s+(?P<b>\d+(?:\.\d+)?)\s*%|"
    r"\b(?:at\s+most|no\s+more\s+than|not\s+(?:more|above|over))\s+(?P<c>\d+(?:\.\d+)?)\s*%",
    re.I)
_CAP_CONTEXT: Final = re.compile(r"\b(?:position|name|holding|asset|weight|single|any\s+one)\b",
                                 re.I)


def _book(text: str, prior: Sequence[str]) -> dict[str, float]:
    """The weights stated in ``text``, else in the latest earlier turn that states two or more."""
    from argus.lui.research.parse import holding_pairs

    for source in (text, *reversed(prior)):
        pairs = holding_pairs(source)
        if len({s for _, s, _w in pairs}) >= 2:
            out: dict[str, float] = {}
            for _, symbol, weight in pairs:
                out.setdefault(symbol, weight)
            return out
    return {}


def _cap_in(text: str) -> float | None:
    m = _CAP.search(text)
    if m is None or not _CAP_CONTEXT.search(text):
        return None
    return float(m.group("a") or m.group("b") or m.group("c")) / 100


def cap_weights(weights: dict[str, float], cap: float) -> tuple[dict[str, float], float]:
    """``weights`` with no name above ``cap``. The excess goes to the names still under the cap in
    proportion to their weight, never lifting one past it; whatever they cannot take is cash.
    Returns the new weights and the cash share (cash the book already held included)."""
    out = dict(weights)
    cash = max(0.0, 1.0 - sum(weights.values()))
    over = [s for s, w in out.items() if w > cap + 1e-12]
    remaining = sum(out[s] - cap for s in over)
    for s in over:
        out[s] = cap
    room = {s: cap - w for s, w in out.items() if w < cap - 1e-12}
    while remaining > 1e-12 and room:
        base = sum(out[s] for s in room)
        used = 0.0
        for s in list(room):
            share = remaining * (out[s] / base if base > 0 else 1.0 / len(room))
            take = min(share, room[s])
            out[s] += take
            room[s] -= take
            used += take
            if room[s] <= 1e-12:
                del room[s]
        remaining -= used
        if used <= 1e-12:
            break
    return out, cash + max(remaining, 0.0)


def _name(symbol: str) -> str:
    return symbol.removesuffix("USDT")


def _weights_text(weights: dict[str, float], cash: float) -> str:
    parts = [f"{w:.0%} {_name(s)}" for s, w in weights.items()]
    if cash > 0.0005:
        parts.append(f"{cash:.0%} cash")
    return ", ".join(parts)


def _book_curve(weights: dict[str, float], p: Panel) -> list[float]:
    """Value of 1 held at ``weights`` with the book rebalanced to them every day, cash flat."""
    rets = {s: p.returns(yahoo_ticker(s)) for s in weights}
    n = len(next(iter(rets.values())))
    value, curve = 1.0, [1.0]
    for i in range(n):
        value *= 1.0 + sum(w * rets[s][i] for s, w in weights.items())
        curve.append(value)
    return curve


Drawdown = tuple[float, int, int]


def _drawdowns(weights: dict[str, float], capped: dict[str, float],
               years: float = 3.0) -> tuple[Panel, Drawdown, Drawdown]:
    """The original and the capped book's worst drawdown over the same days."""
    p = panel([yahoo_ticker(s) for s in weights], _today() - timedelta(days=int(365 * years)))
    return p, _max_drawdown(_book_curve(weights, p)), _max_drawdown(_book_curve(capped, p))


def _dd_sentence(p: Panel, before: Drawdown, after: Drawdown) -> str:
    return (f"Worst drawdown over {_span(p.days)}: original {before[0]:.1%} "
            f"({p.days[before[1]]:%d %b %Y} peak to {p.days[before[2]]:%d %b %Y}), capped "
            f"{after[0]:.1%} ({p.days[after[1]]:%d %b %Y} peak to {p.days[after[2]]:%d %b %Y}).")


def _noted(out: list[str] | None, text: str, prior: Sequence[str]) -> list[str] | None:
    """``out`` with a line after the first one when the stated weights add to more than 100%:
    the numbers are rescaled, and a rewritten book must not pass as the user's own (round 42
    hostile, finding 3, which the weight readers here would otherwise repeat)."""
    if out is None:
        return None
    total = sum(_book(text, prior).values())
    if total <= 1.0 + 1e-9:
        return out
    note = (f"Note: your weights add to {total:.0%}, which means leverage or an error; they are "
            f"shown as shares of that total, rescaled to 100%.")
    return [out[0], note, *out[1:]]


_REBAL_WORDS: Final = re.compile(r"\brebalanc\w*|\bcaps?\b|\bcapped\b|\bcapping\b", re.I)


def _rebalance_core(text: str, prior: Sequence[str]) -> list[str] | None:
    if not _REBAL_WORDS.search(text):
        return None
    asks_drawdown = _DD_ASK.search(text) is not None
    cap = _cap_in(text)
    if cap is None and asks_drawdown:
        cap = next((c for q in reversed(prior[-6:]) if (c := _cap_in(q)) is not None), None)
    if cap is None:
        return None
    weights = _book(text, prior)
    if not weights:
        return None
    total = sum(weights.values())
    if total > 1.0 + 1e-9:
        weights = {s: w / total for s, w in weights.items()}
    capped, cash = cap_weights(weights, cap)
    cash_before = max(0.0, 1.0 - sum(weights.values()))
    over = [s for s, w in weights.items() if w > cap + 1e-12]
    if not over:
        return [f"Bottom line: no change — nothing in {_weights_text(weights, cash_before)} is "
                f"above {cap:.0%}."]
    names = ", ".join(f"{_name(s)} {weights[s]:.0%} to {cap:.0%}" for s in over)
    excess = sum(weights[s] - cap for s in over)
    stuck = [s for s in weights if s not in over and weights[s] >= cap - 1e-12]
    absorbed = sum(capped[s] - weights[s] for s in weights if s not in over)
    if absorbed <= 0.0005:
        where = (f"{excess:.0%} of the book comes off ({names}) and all of it goes to cash: "
                 + ("the names left are already at or above the cap, so they cannot take any"
                    if stuck else "no name is under the cap to take it"))
    elif cash - cash_before > 0.0005:
        where = (f"{excess:.0%} of the book comes off ({names}); {absorbed:.0%} goes to the names "
                 f"under the cap, up to the cap, and the other {cash - cash_before:.0%} to cash")
    else:
        where = (f"{excess:.0%} of the book comes off ({names}) and goes to the names under the "
                 f"cap, in proportion to their weight")
    proposal = (f"Bottom line: with no position above {cap:.0%}, "
                f"{_weights_text(weights, cash_before)} becomes {_weights_text(capped, cash)}.")
    data = ("Data: Yahoo Finance daily closes on the days every holding traded (coins as their "
            "USD pairs), three years, each book held at its weights and rebalanced daily, cash "
            "earning nothing; a past result, not a forecast.")
    try:
        p, before, after = _drawdowns(weights, capped)
    except Unavailable as exc:
        return [proposal if not asks_drawdown else
                f"Bottom line: the drawdown of the capped book could not be compared — {exc}.",
                f"{where[0].upper()}{where[1:]}."]
    drawdown = _dd_sentence(p, before, after)
    if not asks_drawdown:
        return [proposal, f"{where[0].upper()}{where[1:]}.", drawdown, data]
    saved = after[0] - before[0]
    head = (f"Bottom line: the {cap:.0%} cap ({_weights_text(capped, cash)}) would have "
            + (f"cut the worst drawdown from {before[0]:.1%} to {after[0]:.1%}, "
               f"{saved * 100:.1f} points shallower," if saved > 0.0005 else
               f"left the worst drawdown at {after[0]:.1%} against {before[0]:.1%} for the "
               f"original, no shallower,")
            + f" over {_span(p.days)}.")
    return [head, drawdown, f"{where[0].upper()}{where[1:]}.", data]


def _rebalance_lines(text: str, prior: Sequence[str]) -> list[str] | None:
    return _noted(_rebalance_core(text, prior), text, prior)


# --------------------------------------------------------------------------- 4. concentration (M5)

_CONC_Q: Final = re.compile(
    r"\bconcentrat\w*|\brisk\s+contribut\w*|\bbiggest\s+risk\b|\bwhere\s+is\s+(?:most\s+of\s+)?"
    r"my\s+risk\b|\bmost\s+of\s+my\s+risk\b|\btoo\s+much\s+(?:in|exposure)\b|\boverweight\w*",
    re.I)


def risk_shares(weights: dict[str, float], cov: list[list[float]]) -> list[float]:
    """Each holding's share of the book's variance: w_i * (cov w)_i / (w' cov w)."""
    w = [weights[s] for s in weights]
    n = len(w)
    cw = [sum(cov[i][j] * w[j] for j in range(n)) for i in range(n)]
    var = sum(w[i] * cw[i] for i in range(n))
    return [w[i] * cw[i] / var for i in range(n)] if var > 0 else [0.0] * n


def _cov(a: Sequence[float], b: Sequence[float]) -> float:
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / (len(a) - 1)


def _word(r: float) -> str:
    return "high" if abs(r) >= HIGH_CORR else "moderate" if abs(r) >= LOW_CORR else "low"


def _concentration_core(text: str, prior: Sequence[str]) -> list[str] | None:
    if not _CONC_Q.search(text):
        return None
    weights = _book(text, prior[-3:])
    if not weights:
        return None
    total = sum(weights.values())
    if total > 1.0 + 1e-9:
        weights = {s: w / total for s, w in weights.items()}
    try:
        p = panel([yahoo_ticker(s) for s in weights], _today() - timedelta(days=365))
    except Unavailable as exc:
        return _fail(str(exc))
    names = list(weights)
    rets = [p.returns(yahoo_ticker(s)) for s in names]
    cov = [[_cov(a, b) for b in rets] for a in rets]
    shares = risk_shares(weights, cov)
    vols = [math.sqrt(cov[i][i] * TRADING_DAYS) for i in range(len(names))]
    book_vol = math.sqrt(sum(weights[names[i]] * weights[names[j]] * cov[i][j]
                             for i in range(len(names)) for j in range(len(names)))
                         * TRADING_DAYS)
    top = max(range(len(names)), key=lambda i: shares[i])
    heavy = max(range(len(names)), key=lambda i: weights[names[i]])
    pairs = sorted(((_corr(rets[i], rets[j]), names[i], names[j])
                    for i in range(len(names)) for j in range(i + 1, len(names))),
                   key=lambda t: -abs(t[0]))
    strongest = pairs[0]
    rows = "; ".join(f"{_name(n)} {weights[n]:.0%} of the money, {shares[i]:.0%} of the risk "
                     f"(own volatility {vols[i]:.0%} a year)" for i, n in enumerate(names))
    same = top == heavy
    lead = (f"Bottom line: your biggest concentration risk is {_name(names[top])}: "
            f"{weights[names[top]]:.0%} of the money and {shares[top]:.0%} of the book's risk"
            + (f" (the largest weight is {_name(names[heavy])}, but {_name(names[top])} moves "
               f"more and so carries more of the risk)" if not same else "")
            + f". The most correlated pair is {_name(strongest[1])} and {_name(strongest[2])} "
            f"at {strongest[0]:+.2f} ({_word(strongest[0])}).")
    corr_rows = "; ".join(f"{_name(a)}-{_name(b)} {r:+.2f} ({_word(r)})" for r, a, b in pairs)
    return [lead,
            f"Risk share = weight x covariance with the book, over the book's variance: {rows}. "
            f"The book as held moves about {book_vol:.0%} a year.",
            f"Correlation of daily returns over the past year ({len(rets[0])} returns, "
            f"{_span(p.days)}): {corr_rows}. High means {HIGH_CORR:.2f} or more, moderate "
            f"{LOW_CORR:.2f} to {HIGH_CORR:.2f}, low below {LOW_CORR:.2f}.",
            "Data: Yahoo Finance daily closes on the days every holding traded (coins as their "
            "USD pairs), so the stocks' closes line up with the coin's on the same dates. "
            "Correlations move: they rise in selloffs, which is when concentration matters. "
            "Shared exposure (several holdings in one sector or theme) is not measured here."]


def _concentration_lines(text: str, prior: Sequence[str]) -> list[str] | None:
    return _noted(_concentration_core(text, prior[-3:]), text, prior[-3:])


# --------------------------------------------------------------------------- 5. vol-adjusted (M11)

_VOL_ADJ: Final = re.compile(
    r"\badjusted\s+for\s+(?:volatility|risk)\b|\brisk[\s-]*adjusted\b|\bvolatility[\s-]*adjusted\b|"
    r"\bper\s+unit\s+of\s+(?:risk|volatility)\b|\breturn\s+per\s+(?:unit\s+of\s+)?"
    r"(?:risk|volatility)\b|\bsharpe\b|\breturn[\s/]+(?:to\s+)?(?:vol|volatility|risk)\b", re.I)
_VERSUS: Final = re.compile(r"\boutperform\w*|\bbeat\b|\bbetter\b|\bversus\b|\bvs\.?\b|\bcompar\w*|"
                            r"\bor\b|\bagainst\b", re.I)


def tbill_rate(start: date, end: date) -> float | None:
    """Average 3-month Treasury bill yield over the window, a fraction a year (FRED DGS3MO, else
    Yahoo's ^IRX); None when neither answers."""
    from argus.market.skill_mirror import fred_series

    try:
        rows = fred_series("DGS3MO", days=(_today() - start).days + 10)
        inside = [v for d, v in rows if start <= d <= end]
    except Exception:
        inside = []
    if not inside:
        try:
            from argus.market import equity_history

            inside = [d.close for d in equity_history.daily("^IRX") if start <= d.day <= end]
        except Exception:
            inside = []
    return sum(inside) / len(inside) / 100 if inside else None


def _window_days(text: str, default: int = 365) -> tuple[int, str]:
    m = re.search(r"\b(?:last|past|previous|over\s+the\s+last)\s+(?P<n>\d+|a|one|two|three|five|"
                  r"six|twelve)?\s*(?P<u>year|month|week)s?\b|\b(?P<n2>\d+)[\s-]*(?P<u2>year|month)\b",
                  text, re.I)
    if m is None:
        return default, "12 months"
    unit = (m.group("u") or m.group("u2")).lower()
    raw = (m.group("n") or m.group("n2") or "1").lower()
    count = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw, 1)
    days = _unit_days(count, unit)
    return days, f"{count} {unit}{'s' if count != 1 else ''}"


@dataclass(frozen=True, slots=True)
class RiskReturn:
    total: float
    annual: float
    vol: float
    ratio: float
    sharpe: float | None
    worst_dd: float


def risk_return(closes: Sequence[float], days: Sequence[date], rf: float | None) -> RiskReturn:
    rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
    total = closes[-1] / closes[0] - 1.0
    span = max((days[-1] - days[0]).days, 1)
    annual = (1.0 + total) ** (365.0 / span) - 1.0
    vol = statistics.stdev(rets) * math.sqrt(TRADING_DAYS)
    return RiskReturn(total, annual, vol, annual / vol if vol > 0 else 0.0,
                      (annual - rf) / vol if rf is not None and vol > 0 else None,
                      _max_drawdown(list(closes))[0])


def _vol_adjusted_lines(text: str, prior: Sequence[str]) -> list[str] | None:
    del prior
    from argus.lui.research.parse import holding_pairs

    if not _VOL_ADJ.search(text) or not _VERSUS.search(text) or holding_pairs(text):
        return None
    named = _named(text)
    if len(named) != 2:
        return None
    days, label = _window_days(text)
    start = _today() - timedelta(days=days)
    try:
        p = panel([t for t, _ in named], start)
    except Unavailable as exc:
        return _fail(str(exc))
    rf = tbill_rate(p.days[0], p.days[-1])
    res = [risk_return(p.close[t], p.days, rf) for t, _ in named]
    (a, la), (b, lb) = named
    ra, rb = res
    key_a = ra.sharpe if ra.sharpe is not None and rb.sharpe is not None else ra.ratio
    key_b = rb.sharpe if ra.sharpe is not None and rb.sharpe is not None else rb.ratio
    metric = "Sharpe ratio" if ra.sharpe is not None and rb.sharpe is not None else \
        "return per unit of volatility"
    winner = la if key_a > key_b else lb
    raw_winner, raw_loser = (la, lb) if ra.total > rb.total else (lb, la)
    raw_hi, raw_lo = max(ra.total, rb.total), min(ra.total, rb.total)
    verb = "outperformed" if key_a > key_b else "did not outperform"
    answer = f"{la} {verb} {lb} once adjusted for volatility"
    head = (f"Bottom line: {'yes' if key_a > key_b else 'no'} — over the last {label} "
            f"({_span(p.days)}) {answer}: {metric} {key_a:.2f} against {key_b:.2f}. "
            f"On raw return {raw_winner} was ahead ({raw_hi:+.1%} against {raw_lo:+.1%} for "
            f"{raw_loser})"
            + ("" if raw_winner == winner else ", so the ranking flips once risk is counted")
            + ".")

    def row(lab: str, r: RiskReturn) -> str:
        sh = f", Sharpe {r.sharpe:.2f}" if r.sharpe is not None else ""
        return (f"{lab}: return {r.total:+.1%}, volatility {r.vol:.1%} a year, return per unit "
                f"of volatility {r.ratio:.2f}{sh}, worst drawdown {r.worst_dd:.1%}")

    out = [head, row(la, ra) + ".", row(lb, rb) + ".",
           (f"Sharpe uses the 3-month Treasury bill, averaging {rf:.2%} over the window; "
            if rf is not None else "The Treasury bill rate could not be read, so Sharpe is not "
                                    "shown; ")
           + "return per unit of volatility is the annualised return over annualised volatility.",
           f"Data: Yahoo Finance daily closes ({a}, {b}) on the days both traded, so a coin's "
           f"weekend moves are inside the Monday figure; volatility is the standard deviation of "
           f"those returns times the square root of {TRADING_DAYS}. One year is one regime; a "
           f"ratio difference this size may not be statistically distinguishable."]
    return out


# --------------------------------------------------------------------------- 6. breadth (M12)

_BREADTH_Q: Final = re.compile(
    r"(?:\bpercent(?:age)?\b|%\s+of\b|\bshare\b|\bproportion\b|\bfraction\b|\bhow\s+many\b)"
    r"[^?]{0,40}?\b(?:s\s*&\s*p(?:\s*-?\s*500)?|sp500|spx)\b[^?]{0,70}?"
    r"\b(?P<dir>above|over|under|below)\b[^?]{0,30}?\b(?P<n>20|50|100|200)[\s-]*(?:day|d)\b|"
    r"\bbreadth\b[^?]{0,60}\b(?:s\s*&\s*p(?:\s*-?\s*500)?|sp500)\b[^?]{0,60}?"
    r"\b(?P<n2>20|50|100|200)[\s-]*(?:day|d)\b", re.I)
SPARK: Final = ("https://query1.finance.yahoo.com/v8/finance/spark?symbols={symbols}"
                "&range=1y&interval=1d")
MEMBERS_CSV: Final = ("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/"
                      "data/constituents.csv")
MEMBERS_WIKI: Final = ("https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies"
                       "&action=raw")
_UA: Final = {"User-Agent": "Mozilla/5.0 argus-research"}


def sp500_members() -> list[str]:
    """Yahoo tickers of the S&P 500 members: the datasets/s-and-p-500-companies list, else the
    Wikipedia table (class shares written with a dash, as Yahoo has them)."""
    from argus.truth import http

    symbols: list[str] = []
    try:
        import csv
        import io

        body = http.fetch_text(MEMBERS_CSV, timeout=20.0, headers=_UA)
        symbols = [r["Symbol"].strip() for r in csv.DictReader(io.StringIO(body))
                   if r.get("Symbol")]
    except Exception:
        symbols = []
    if len(symbols) < 450:
        try:
            body = http.fetch_text(MEMBERS_WIKI, timeout=20.0, headers=_UA)
            symbols = re.findall(r"\{\{(?:Nyse|Nasdaq)Symbol\|([A-Z.\-]+)\}\}", body)
        except Exception:
            symbols = []
    if len(symbols) < 450:
        raise Unavailable("the list of S&P 500 members could not be read")
    return [s.replace(".", "-") for s in dict.fromkeys(symbols)]


def _spark(batch: list[str]) -> dict[str, tuple[date, list[float]]]:
    from argus.truth import http

    try:
        payload = http.fetch_json(SPARK.format(symbols=",".join(batch)), timeout=30.0,
                                  headers=_UA)
    except Exception:
        return {}
    out: dict[str, tuple[date, list[float]]] = {}
    for sym, row in (payload or {}).items() if isinstance(payload, dict) else []:
        try:
            pairs = [(t, c) for t, c in zip(row["timestamp"], row["close"], strict=False)
                     if c is not None and c > 0]
            if pairs:
                out[sym] = (datetime.fromtimestamp(pairs[-1][0], tz=UTC).date(),
                            [float(c) for _, c in pairs])
        except (KeyError, TypeError, ValueError):
            continue
    return out


BREADTH_CACHE_SECONDS: Final = 900.0
_breadth_cache: dict[str, tuple[float, list[str], dict[str, tuple[date, list[float]]]]] = {}
_breadth_lock = threading.Lock()


def _member_closes() -> tuple[list[str], dict[str, tuple[date, list[float]]]]:
    """The member list and each member's past year of closes: 26 requests of 20 symbols (Yahoo's
    limit per call) in parallel, kept for fifteen minutes so a second ask is instant."""
    now = time.monotonic()
    with _breadth_lock:
        hit = _breadth_cache.get("sp500")
        if hit and now - hit[0] < BREADTH_CACHE_SECONDS:
            return hit[1], hit[2]
    members = sp500_members()
    batches = [members[i:i + 20] for i in range(0, len(members), 20)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        parts = list(pool.map(_spark, batches))
    data: dict[str, tuple[date, list[float]]] = {}
    for part in parts:
        data.update(part)
    if len(data) < 0.9 * len(members):
        raise Unavailable(f"Yahoo returned closes for only {len(data)} of {len(members)} members")
    with _breadth_lock:
        _breadth_cache["sp500"] = (now, members, data)
    return members, data


def breadth_counts(closes: dict[str, list[float]], window: int) -> tuple[int, int]:
    """(members above their ``window``-day average, members with enough history)."""
    above = counted = 0
    for series in closes.values():
        if len(series) < window:
            continue
        counted += 1
        above += series[-1] > sum(series[-window:]) / window
    return above, counted


def _breadth_lines(text: str, prior: Sequence[str]) -> list[str] | None:
    del prior
    m = _BREADTH_Q.search(text)
    if m is None:
        return None
    window = int(m.group("n") or m.group("n2") or 50)
    below = (m.group("dir") or "").lower() in ("under", "below")
    try:
        members, data = _member_closes()
    except Unavailable as exc:
        return _breadth_proxy(window, str(exc))
    closes = {s: v[1] for s, v in data.items()}
    above, counted = breadth_counts(closes, window)
    if counted < 0.9 * len(members):
        return _breadth_proxy(window, f"only {counted} of {len(members)} members have {window} "
                                      f"daily closes to average")
    other = 200 if window == 50 else 50
    above_o, counted_o = breadth_counts(closes, other)
    latest = max(d for d, _ in data.values())
    share = above / counted
    mood = ("a majority of the index is participating" if share > 0.5 else
            "fewer than half of the members are above the line")
    asked = (f"{(counted - above) / counted:.1%} of S&P 500 stocks ({counted - above} of "
             f"{counted}) are below their {window}-day average, {share:.1%} above" if below else
             f"{share:.1%} of S&P 500 stocks ({above} of {counted}) are above their "
             f"{window}-day average")
    side, n_o = ("below", counted_o - above_o) if below else ("above", above_o)
    out = [f"Bottom line: {asked} — {mood} (latest closes, {latest:%d %b %Y}).",
           (f"For comparison, {n_o / counted_o:.1%} ({n_o} of {counted_o}) are {side} their "
            f"{other}-day average." if counted_o else ""),
           f"Data: the current S&P 500 member list ({len(members)} tickers, "
           f"{len(data)} answered by Yahoo Finance, {counted} with at least {window} daily "
           f"closes in the past year); each member's latest close against the simple average of "
           f"its last {window} closes. Today's members only, so it is the breadth of the index "
           f"as listed now."]
    return [x for x in out if x]


def _breadth_proxy(window: int, why: str) -> list[str]:
    try:
        out = []
        for ticker in ("SPY", "RSP"):
            rows = _closes(ticker)
            days = sorted(rows)
            vals = [rows[d] for d in days]
            out.append((ticker, vals[-1], sum(vals[-window:]) / window, days[-1]))
    except Unavailable as exc:
        return _fail(f"{why}, and the proxy could not be read either ({exc})")
    (_, spy_l, spy_a, day), (_, rsp_l, rsp_a, _) = out
    return [f"Bottom line: the true share of S&P 500 stocks above their {window}-day average "
            f"could not be computed here ({why}). As a proxy only: the cap-weighted S&P 500 "
            f"(SPY) is {spy_l / spy_a - 1:+.1%} from its {window}-day average and the "
            f"equal-weighted one (RSP) {rsp_l / rsp_a - 1:+.1%} ({day:%d %b %Y}).",
            "Equal-weight above its average while cap-weight lags points to broad participation; "
            "the reverse points to a narrow, megacap-led market. It is not the percentage "
            "asked for. Data: Yahoo Finance daily closes of SPY and RSP."]


# --------------------------------------------------------------------------- 7. rolling (M13)

_ROLL_Q: Final = re.compile(r"\b(?:correlat\w*|relat\w*|relationship|move\s+with|moves\s+with|"
                            r"track\w*|inverse\w*)\b", re.I)
_DXY: Final = re.compile(r"\bdxy\b|\bdollar\s+index\b|\busd\s+index\b|\bdollar\b(?![\s-]*cost)",
                         re.I)


def rolling_corr(a: Sequence[float], b: Sequence[float], window: int) -> list[float]:
    """Correlation of the last ``window`` returns, one value per day from the ``window``th."""
    return [_corr(a[i - window:i], b[i - window:i]) for i in range(window, len(a) + 1)]


def _dollar_panel(coin: str, start: date) -> tuple[Panel, str, str]:
    try:
        return (panel([coin, "DX-Y.NYB"], start), "DX-Y.NYB",
                "the ICE US Dollar Index (DXY, Yahoo DX-Y.NYB)")
    except Unavailable:
        from argus.market.skill_mirror import fred_series

        try:
            rows = dict(fred_series("DTWEXBGS", days=(_today() - start).days + 120))
        except Exception as exc:
            raise Unavailable("neither Yahoo's DXY nor FRED's dollar index could be read") \
                from exc
        coin_close = _closes(coin)
        days = sorted(d for d in rows if d in coin_close and d >= start)
        if len(days) < 80:
            raise Unavailable("too few days with both the coin and the dollar index") from None
        return (Panel(days, {coin: [coin_close[d] for d in days],
                             "DTWEXBGS": [rows[d] for d in days]}), "DTWEXBGS",
                "the Federal Reserve's broad trade-weighted dollar index (FRED DTWEXBGS, not "
                "ICE DXY)")


def _rolling_lines(text: str, prior: Sequence[str]) -> list[str] | None:
    del prior
    if not (_ROLL_Q.search(text) and _DXY.search(text)):
        return None
    if not re.search(r"\brolling\b|\bover\s+time\b|\bthis\s+year\b|\bytd\b|\byear\s+to\s+date\b|"
                     r"\brelat\w*|\brelationship\b", text, re.I) and re.search(
            r"\bgold\b|\bxau\b|\bnasdaq\b|\bqqq\b|\bs\s*&\s*p\b|\boil\b|\b(?:last|past)\s+\d+\s+"
            r"days?\b", text, re.I):
        # "Correlation of BTC with gold and the dollar index, last 90 days" got a rolling
        # 60-day BTC-dollar series over a year, gold and the 90 days dropped (round 45 re-ask):
        # a plain correlation over a stated window, or one naming other series, is the
        # correlation matrix's
        return None
    named = [(t, lab) for t, lab in _named(text) if t in ("BTC-USD", "ETH-USD", "SOL-USD")]
    if not named:
        return None
    coin, coin_label = named[0]
    m = re.search(r"\b(?P<n>\d+)[\s-]*(?:day|d)\b", text, re.I)
    window = int(m.group("n")) if m else 60
    if window < 10 or window > 250:
        return None
    today = _today()
    if re.search(r"\bthis\s+year\b|\bytd\b|\byear\s+to\s+date\b", text, re.I):
        begin, label = date(today.year, 1, 1), f"so far this year (from 1 Jan {today.year})"
    else:
        begin, label = today - timedelta(days=365), "over the past 12 months"
    try:
        p, dxy_key, dxy_label = _dollar_panel(coin, begin - timedelta(days=int(window * 1.6) + 10))
    except Unavailable as exc:
        return _fail(str(exc))
    a, b = p.returns(coin), p.returns(dxy_key)
    series = rolling_corr(a, b, window)
    ends = p.days[window:]
    inside = [(d, r) for d, r in zip(ends, series, strict=True) if d >= begin]
    if len(inside) < 10:
        return _fail(f"only {len(inside)} rolling {window}-day readings {label}")
    values = [r for _, r in inside]
    lo = min(inside, key=lambda t: t[1])
    hi = max(inside, key=lambda t: t[1])
    latest = inside[-1]
    mean = sum(values) / len(values)
    neg = sum(v < 0 for v in values) / len(values)
    flavour = ("the dollar and " + coin_label + " have mostly moved in opposite directions"
               if mean < -LOW_CORR / 2 else
               "the dollar and " + coin_label + " have mostly moved together"
               if mean > LOW_CORR / 2 else "no steady link between the dollar and " + coin_label)
    dxy_note = "" if dxy_key == "DX-Y.NYB" else (" The ICE DXY series was not readable, so the "
                                                  "Fed's broad index stands in; it is a "
                                                  "different basket.")
    return [f"Bottom line: the rolling {window}-day correlation of daily returns between "
            f"{coin_label} and {dxy_label} is {latest[1]:+.2f} now ({latest[0]:%d %b %Y}), and "
            f"ranged from {lo[1]:+.2f} ({lo[0]:%d %b}) to {hi[1]:+.2f} ({hi[0]:%d %b}) "
            f"{label} — {flavour}.",
            f"Average {mean:+.2f}; negative on {neg:.0%} of the {len(inside)} days. Below "
            f"|{LOW_CORR:.2f}| is weak, {LOW_CORR:.2f} to {HIGH_CORR:.2f} moderate, above "
            f"{HIGH_CORR:.2f} strong.{dxy_note}",
            f"Data: Yahoo Finance daily closes ({coin} and {dxy_key}) on the days both traded, "
            f"returns between those days (a weekend coin move counts once); correlation of the "
            f"latest {window} returns at each date. Correlation is not causation and a "
            f"{window}-day window is noisy."]


# --------------------------------------------------------------------------- 8. funding (M14)

_FUND_Q: Final = re.compile(r"\bfunding\b", re.I)
_EXTREME: Final = re.compile(
    r"\bextreme\b|\blong\s+or\s+short\b|\b(?:either|both)\s+sides?\b|"
    r"\bboth\s+(?:long\s+and\s+short|directions)\b", re.I)
_FUND_ASK: Final = re.compile(
    r"\bwhich\b|\bwhat\s+(?:perp\w*|contract|coin|token|pair)\b|\bnow\b|\bcurrently\b|\btoday\b|"
    r"\btop\b|\brank\w*|\blist\b|\bshow\b", re.I)
_SQUEEZE: Final = re.compile(r"\bsqueeze\b", re.I)


@dataclass(frozen=True, slots=True)
class FundingRow:
    symbol: str
    rate: float
    interval: float
    last: float
    change: float
    volume: float
    open_interest: float

    @property
    def per_day(self) -> float:
        return self.rate * 24.0 / self.interval


def funding_rows() -> list[FundingRow]:
    """Every liquid, trading USDT perpetual with its funding, from Bitget's public endpoints."""
    from argus.market.bitget import public_get

    tickers = public_get("/api/v2/mix/market/tickers", {"productType": "usdt-futures"}) or []
    contracts = public_get("/api/v2/mix/market/contracts",
                           {"productType": "USDT-FUTURES"}) or []
    meta = {c.get("symbol"): c for c in contracts}
    rows: list[FundingRow] = []
    for t in tickers:
        try:
            symbol = str(t["symbol"])
            volume = float(t.get("usdtVolume") or t.get("quoteVolume") or 0)
            if volume < LIQUID_USD or t.get("fundingRate") in (None, ""):
                continue
            info = meta.get(symbol, {})
            if info.get("symbolStatus", "normal") != "normal":
                continue
            last = float(t["lastPr"])
            rows.append(FundingRow(
                symbol=symbol, rate=float(t["fundingRate"]),
                interval=float(info.get("fundInterval") or 8) or 8.0, last=last,
                change=float(t.get("change24h") or 0.0), volume=volume,
                open_interest=float(t.get("holdingAmount") or 0.0) * last))
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def _funding_cap(symbol: str) -> float | None:
    from argus.market.bitget import public_get

    try:
        rows = public_get("/api/v2/mix/market/current-fund-rate",
                          {"symbol": symbol, "productType": "USDT-FUTURES"}) or []
        row = rows[0]
        rate = float(row["fundingRate"])
        limit = float(row["minFundingRate"] if rate < 0 else row["maxFundingRate"])
        return abs(rate / limit) if limit else None
    except Exception:
        return None


def _fund_row(r: FundingRow) -> str:
    return (f"{r.symbol} {r.rate:+.4%} every {r.interval:g}h ({r.per_day:+.2%} a day)")


def _squeeze_reading(r: FundingRow, cap_used: float | None) -> str:
    short_side = r.rate < 0
    crowded = "shorts" if short_side else "longs"
    payer = "shorts pay longs" if short_side else "longs pay shorts"
    against = r.change > 0 if short_side else r.change < 0
    move = (f"price has moved against them, {r.change:+.1%} in 24 hours" if against else
            f"price has not moved against them, {r.change:+.1%} in 24 hours")
    cap = f"; funding is at {cap_used:.0%} of the contract's cap" if cap_used else ""
    mech = ("a rise forcing them to buy back" if short_side else "a fall forcing them to sell")
    return (f"Squeeze reading: {r.symbol} is a crowded-{crowded[:-1]} setup on funding — "
            f"{payer} — and {move}{cap}. Open interest is {_usd(r.open_interest)} against "
            f"{_usd(r.volume)} traded in 24 hours. A {'short' if short_side else 'long'} squeeze "
            f"needs {mech}; funding shows the crowding, not when or whether it breaks, and "
            f"Bitget's public feed gives open interest now but not its history, so whether "
            f"positions are building or unwinding cannot be read. This is not a prediction.")


def _funding_lines(text: str, prior: Sequence[str]) -> list[str] | None:
    follow_up = bool(_SQUEEZE.search(text)) and any(
        _FUND_Q.search(q) and _EXTREME.search(q) for q in prior[-2:])
    if not follow_up and not (_FUND_Q.search(text) and _EXTREME.search(text)):
        return None
    if not follow_up and not _FUND_ASK.search(text):
        return None
    try:
        rows = funding_rows()
    except Exception as exc:
        return _fail(f"Bitget's tickers did not arrive ({type(exc).__name__})")
    if len(rows) < 10:
        return _fail("too few liquid Bitget perpetuals could be read")
    ranked = sorted(rows, key=lambda r: -abs(r.per_day))
    longs = sorted((r for r in rows if r.rate > 0), key=lambda r: -r.per_day)
    shorts = sorted((r for r in rows if r.rate < 0), key=lambda r: r.per_day)
    top = ranked[0]
    side_name = "short side (negative funding, shorts pay)" if top.rate < 0 else \
        "long side (positive funding, longs pay)"
    other = longs[0] if top.rate < 0 else shorts[0] if shorts else None
    cap = _funding_cap(top.symbol)
    if follow_up and not _EXTREME.search(text):
        return [f"Bottom line: on the {side_name}, {top.symbol} — "
                + ("yes, it reads as a crowded-short setup" if top.rate < 0 else
                   "it reads as a crowded-long setup, so the squeeze risk is on the long side")
                + f"; {_fund_row(top)}.", _squeeze_reading(top, cap)]
    lead = (f"Bottom line: {top.symbol} has the most extreme funding among liquid Bitget "
            f"perpetuals: {_fund_row(top)} — the {side_name}.")
    if other is not None:
        lead += f" On the other side the most extreme is {_fund_row(other)}."
    return [lead,
            "Most negative (shorts pay): " + "; ".join(
                f"{r.symbol} {r.rate:+.4%} ({r.per_day:+.2%}/day)" for r in shorts[:4]) + ".",
            "Most positive (longs pay): " + "; ".join(
                f"{r.symbol} {r.rate:+.4%} ({r.per_day:+.2%}/day)" for r in longs[:4]) + ".",
            _squeeze_reading(top, cap),
            f"Data: Bitget public tickers and contract list; liquid means at least "
            f"{_usd(LIQUID_USD)} traded in 24 hours ({len(rows)} contracts qualify), ranked by "
            f"funding per day because contracts settle every 1, 4 or 8 hours. Rates are the "
            f"current rate for the next settlement."]


# --------------------------------------------------------------------------- entry point

_READERS: Final = (_funding_lines, _breadth_lines, _rolling_lines, _dca_lines, _lead_lag_lines,
                   _vol_adjusted_lines, _rebalance_lines, _concentration_lines)


def lines(text: str, prior: Sequence[str] = ()) -> list[str] | None:
    """The answer when ``text`` is one of the questions above, else None. ``prior`` is the earlier
    user questions, oldest first; a follow-up is read through it."""
    for reader in _READERS:
        got = reader(text, prior)
        if got is not None:
            return got
    return None


trace_module(globals())
