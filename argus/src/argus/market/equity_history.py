"""A US stock's own daily history since listing, and the gaps across the hours it does not trade.

Bitget's stock perpetuals are young — NVDAUSDT has about 400 daily bars — so the weekend record
they carry is some seventy weekends. The company behind each one has traded for decades, and every
Friday-close-to-Monday-open since is an observation of what a closure did to its price. baserate
(Jayanng, a Season 2 desk, proprietary: its idea is used here, none of its code) builds its weekend
risk on 1,227 such episodes per stock from 1999; ARGUS read none of them until 2026-09-25, when the
same leveraged-weekend question was put to both and ARGUS answered a different question.

Prices are Yahoo Finance's daily chart (``/v8/finance/chart``, ``period1=0``, ``interval=1d``),
which is keyless and carries ``adjclose``. A raw gap across a split weekend reads as a 75% crash,
so every open is scaled by that day's ``adjclose / close`` before a gap is taken; dividends move
the factor by a few basis points and are left in.

A closure here is the time from one regular-session close to the next regular-session open with at
least one calendar day in between that had no session: weekends, holidays and the long weekends
they make. ``weekend_only`` keeps the ones that span a Saturday.
"""

from __future__ import annotations

import bisect
import itertools
import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from argus.truth import http
from argus.truth.bounded import BoundedDict

CHART = ("https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
         "?period1=0&period2=9999999999&interval=1d")
CACHE_SECONDS = 6 * 3600.0
MAX_CLOSURE_DAYS = 5
"""The longest real closure is a Friday holiday after a Thursday close, reopening on a Monday
holiday's Tuesday — five calendar days at most. A longer stretch between two bars is a hole in the
data (a halt, a missing row), and counting it as a weekend would invent a gap."""


class HistoryError(RuntimeError):
    """The history could not be read; callers say so rather than answer from nothing."""


@dataclass(frozen=True, slots=True)
class Day:
    day: date
    open: float
    close: float
    """Both split-adjusted (``adjclose`` basis)."""


@dataclass(frozen=True, slots=True)
class Gap:
    closed: date
    reopened: date
    move: float
    """Reopen over the last close, minus one."""


_cache: BoundedDict[str, tuple[float, list[Day]]] = BoundedDict(256)
_lock = threading.Lock()


def _fetch(ticker: str, *, timeout: float = 20.0) -> dict[str, Any]:
    try:
        payload: dict[str, Any] = http.fetch_json(
            CHART.format(ticker=ticker), timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 argus-research"})
    except http.RpcError as exc:
        raise HistoryError(f"Yahoo daily history for {ticker} did not arrive: "
                           f"{type(exc).__name__}") from exc
    return payload


def parse(payload: dict[str, Any]) -> list[Day]:
    """Adjusted daily bars from a chart payload, oldest first; rows with a missing field dropped."""
    try:
        result = payload["chart"]["result"][0]
        stamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        adj = result["indicators"]["adjclose"][0]["adjclose"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HistoryError(f"unexpected chart shape: {exc}") from exc
    days: list[Day] = []
    for stamp, opened, closed, adjusted in zip(stamps, quote["open"], quote["close"], adj,
                                               strict=False):
        if not opened or not closed or not adjusted or closed <= 0:
            continue
        factor = adjusted / closed
        days.append(Day(day=datetime.fromtimestamp(stamp, tz=UTC).date(),
                        open=opened * factor, close=adjusted))
    return days


def daily(ticker: str, *, fetch: Any = _fetch) -> list[Day]:
    now = time.monotonic()
    with _lock:
        hit = _cache.get(ticker)
        if hit and now - hit[0] < CACHE_SECONDS:
            return hit[1]
    days = parse(fetch(ticker))
    if len(days) < 250:
        raise HistoryError(f"only {len(days)} daily bars for {ticker}")
    with _lock:
        _cache[ticker] = (now, days)
    return days


def closure_gaps(days: list[Day], *, weekend_only: bool = True) -> list[Gap]:
    """Close-to-open moves across every closure longer than a normal overnight."""
    gaps: list[Gap] = []
    for before, after in itertools.pairwise(days):
        span = (after.day - before.day).days
        if span < 2 or span > MAX_CLOSURE_DAYS:
            continue
        spans_saturday = any(
            date.fromordinal(before.day.toordinal() + k).weekday() == 5 for k in range(1, span))
        if weekend_only and not spans_saturday:
            continue
        gaps.append(Gap(closed=before.day, reopened=after.day, move=after.open / before.close - 1))
    return gaps


@dataclass(frozen=True, slots=True)
class GapRecord:
    n: int
    since: date
    up: float
    flat: float
    down_1_5: float
    down_5: float
    worst: Gap
    beyond: int
    """Gaps at or past the stated adverse distance."""

    def as_dict(self) -> dict[str, Any]:
        return {"n": self.n, "since": self.since.isoformat(), "up": self.up, "flat": self.flat,
                "down_1_5": self.down_1_5, "down_5": self.down_5,
                "worst": {"move": self.worst.move, "closed": self.worst.closed.isoformat(),
                          "reopened": self.worst.reopened.isoformat()},
                "beyond": self.beyond}


def record(gaps: list[Gap], *, side: str, adverse: float) -> GapRecord:
    """The closure record from ``side``'s point of view: ``adverse`` is the move against it, as a
    positive fraction, that counts as reached (the liquidation distance)."""
    if not gaps:
        raise HistoryError("no closures to describe")
    sign = -1.0 if side == "short" else 1.0
    moves = [sign * g.move for g in gaps]  # positive = in the position's favour
    n = len(moves)
    worst_index = min(range(n), key=lambda i: moves[i])
    return GapRecord(
        n=n, since=gaps[0].closed,
        up=sum(m > 0.01 for m in moves) / n, flat=sum(abs(m) <= 0.01 for m in moves) / n,
        down_1_5=sum(-0.05 <= m < -0.01 for m in moves) / n,
        down_5=sum(m < -0.05 for m in moves) / n,
        worst=gaps[worst_index], beyond=sum(m <= -adverse for m in moves))


TREND_BAND = 0.02
"""A 20-day average more than 2% above (below) the 50-day is an up (down) trend. The band is
baserate's (`TREND_UP_SMA_MULTIPLIER = 1.02`); its fixed +/-8% five-day thresholds are not
used — momentum here is the stock's own tercile, so a quiet utility and NVDA are read on their own
scale."""


@dataclass(frozen=True, slots=True)
class State:
    trend: str
    """up, down or flat: the 20-day average against the 50-day."""
    momentum: str
    """low, mid or high: the five-day return's tercile in this stock's own history."""
    five_day: float

    def words(self) -> str:
        trend = {"up": "in an uptrend", "down": "in a downtrend", "flat": "without a trend"}
        pace = {"low": "a weak", "mid": "an ordinary", "high": "a strong"}
        return f"{trend[self.trend]} after {pace[self.momentum]} week ({self.five_day:+.1%})"


def _states(days: list[Day]) -> dict[date, State]:
    closes = [d.close for d in days]
    fives = [closes[i] / closes[i - 5] - 1 for i in range(5, len(closes))]
    if len(fives) < 30:
        return {}
    ordered = sorted(fives)
    low_cut, high_cut = ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3]
    states: dict[date, State] = {}
    for i in range(50, len(days)):
        sma20 = sum(closes[i - 19:i + 1]) / 20
        sma50 = sum(closes[i - 49:i + 1]) / 50
        trend = ("up" if sma20 > sma50 * (1 + TREND_BAND) else
                 "down" if sma20 < sma50 * (1 - TREND_BAND) else "flat")
        five = closes[i] / closes[i - 5] - 1
        momentum = "low" if five < low_cut else "high" if five > high_cut else "mid"
        states[days[i].day] = State(trend=trend, momentum=momentum, five_day=five)
    return states


@dataclass(frozen=True, slots=True)
class Matched:
    state: State
    n: int
    against_share: float
    against_low: float
    against_high: float
    all_against_share: float
    worst: Gap | None

    @property
    def differs(self) -> bool:
        return not self.against_low <= self.all_against_share <= self.against_high

    def as_dict(self) -> dict[str, Any]:
        return {"trend": self.state.trend, "momentum": self.state.momentum,
                "five_day": self.state.five_day, "n": self.n,
                "against_share": self.against_share,
                "interval": [self.against_low, self.against_high],
                "all_against_share": self.all_against_share, "differs": self.differs,
                "worst": None if self.worst is None else {
                    "move": self.worst.move, "closed": self.worst.closed.isoformat()}}


def matched_record(days: list[Day], gaps: list[Gap], *, side: str) -> Matched | None:
    """Weekends that began in the same state as the latest day, and whether they behaved
    differently from all weekends: the share that opened more than 1% against ``side``, with a
    Wilson interval, against the unconditional share. None when there is no state to read."""
    from argus.risk.calibration import wilson

    states = _states(days)
    if not days or days[-1].day not in states:
        return None
    now = states[days[-1].day]
    sign = -1.0 if side == "short" else 1.0
    against_all = [sign * g.move < -0.01 for g in gaps]
    matched = [g for g in gaps if (s := states.get(g.closed)) is not None
               and (s.trend, s.momentum) == (now.trend, now.momentum)]
    if len(matched) < 10 or not against_all:
        return None
    hits = sum(sign * g.move < -0.01 for g in matched)
    low, high = wilson(hits, len(matched))
    worst = min(matched, key=lambda g: sign * g.move)
    return Matched(state=now, n=len(matched), against_share=hits / len(matched),
                   against_low=low, against_high=high,
                   all_against_share=sum(against_all) / len(against_all), worst=worst)


# --- the weekend band: scaled to today's volatility, held to its stated coverage ------------------

EWMA_LAMBDA = 0.94
"""RiskMetrics' daily decay (J.P. Morgan/Reuters, *RiskMetrics Technical Document*, 1996)."""
VOL_DAYS = 20
BAND_QUANTILES = (0.1, 0.5, 0.9)
BAND_WARMUP = 260
"""Five years of weekends before the band's coverage is tracked and corrected."""
CONFORMAL_MIN = 50
ACI_GAMMA = 0.005
"""Adaptive conformal inference's step (Gibbs and Candès, NeurIPS 2021): after a miss the band's
level rises by ``gamma * 0.8``, after a hit it falls by ``gamma * 0.2``."""


@dataclass(frozen=True, slots=True)
class Friday:
    """What was known at one close, point in time."""

    vol: float
    trend: str
    five_day: float
    ewma_vol: float = 0.0
    """RiskMetrics' exponentially weighted volatility at the same close."""


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy's default, type 7)."""
    if not sorted_values:
        raise ValueError("no values")
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def fridays(days: Sequence[Day]) -> dict[date, Friday]:
    """The state on every day with enough history, from closes up to and including that day."""
    closes = [d.close for d in days]
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    ewma: list[float] = []
    variance = rets[0] ** 2 if rets else 0.0
    for r in rets:
        variance = EWMA_LAMBDA * variance + (1 - EWMA_LAMBDA) * r * r
        ewma.append(math.sqrt(variance))
    out: dict[date, Friday] = {}
    for i in range(50, len(days)):
        window = rets[i - VOL_DAYS:i]
        mean = sum(window) / len(window)
        vol = math.sqrt(sum((r - mean) ** 2 for r in window) / (len(window) - 1))
        sma20 = sum(closes[i - 19:i + 1]) / 20
        sma50 = sum(closes[i - 49:i + 1]) / 50
        trend = ("up" if sma20 > sma50 * (1 + TREND_BAND) else
                 "down" if sma20 < sma50 * (1 - TREND_BAND) else "flat")
        out[days[i].day] = Friday(vol=vol, trend=trend, five_day=closes[i] / closes[i - 5] - 1,
                                  ewma_vol=ewma[i - 1])
    return out


Forecast = tuple[float, float, float]


def scaled_forecast(standardised: Sequence[float], moves: Sequence[float],
                    vol: float) -> Forecast:
    """Filtered historical simulation: the quantiles of earlier moves divided by their own
    volatility, times today's; with nothing to scale by, the plain record's quantiles."""
    if not standardised or vol <= 0:
        return (quantile(moves, 0.1), quantile(moves, 0.5), quantile(moves, 0.9))
    return (quantile(standardised, 0.1) * vol, quantile(standardised, 0.5) * vol,
            quantile(standardised, 0.9) * vol)


def widened(band: Forecast, vol: float, scores: Sequence[float], level: float) -> Forecast:
    """The band widened at both ends by the ``level`` quantile of earlier conformity scores
    (conformalized quantile regression, Romano, Patterson and Candès, NeurIPS 2019)."""
    if len(scores) < CONFORMAL_MIN or vol <= 0:
        return band
    widen = quantile(scores, min(max(level, 0.0), 1.0)) * vol
    return (band[0] - widen, band[1], band[2] + widen)


@dataclass(frozen=True, slots=True)
class Band:
    """Where the next reopening should land, and how well such bands have done for this stock."""

    p10: float
    p50: float
    p90: float
    """Moves from Friday's close to Monday's open, as fractions."""
    ewma_vol: float
    """The stock's daily volatility at the last close (RiskMetrics EWMA)."""
    weekends: int
    since: date
    tracked: int
    """Earlier weekends on which the band was forecast out of sample and scored."""
    covered: float
    """Share of those that opened inside their own 10-90 band."""

    def as_dict(self) -> dict[str, Any]:
        return {"p10": self.p10, "p50": self.p50, "p90": self.p90, "ewma_vol": self.ewma_vol,
                "weekends": self.weekends, "since": self.since.isoformat(),
                "tracked": self.tracked, "covered": self.covered}


def weekend_band(days: Sequence[Day]) -> Band | None:
    """The next weekend's 10/50/90 band: every earlier weekend scaled to its own Friday's EWMA
    volatility and rescaled to today's, widened by adaptive conformal inference so that its bands
    keep covering 80% of what happened when the market drifts.

    Chosen by measurement, not assumed: `eval/weekend_quantiles.py` scored this and five other
    forecasts walk-forward over 17,044 weekends of 13 stocks (``data/weekend_quantiles.json``,
    2026-09-27). It had the lowest pooled pinball loss (26.37bp against 27.99bp for the plain
    record and 27.84bp for baserate's trend-and-momentum match), beat the plain record in both
    halves of the sample for all 13 stocks, and its 10-90 bands covered 79.4% of weekends against
    77.2% for the plain record and 75.1% for baserate's match. This function repeats that walk
    for one stock and returns the forecast for the coming weekend; a test holds it to the
    evaluation's own arithmetic. None with too little history."""
    state = fridays(days)
    if not days or days[-1].day not in state:
        return None
    pairs = [(g, state[g.closed]) for g in closure_gaps(list(days)) if g.closed in state]
    if len(pairs) <= BAND_WARMUP:
        return None
    moves: list[float] = []
    standardised: list[float] = []
    scores: list[float] = []
    level = 0.8
    tracked = covered = 0
    for index, (gap, friday) in enumerate(pairs):
        if index >= BAND_WARMUP:
            band = scaled_forecast(standardised, moves, friday.ewma_vol)
            adaptive = widened(band, friday.ewma_vol, scores, level)
            if friday.ewma_vol > 0:
                bisect.insort(scores, max(band[0] - gap.move, gap.move - band[2])
                              / friday.ewma_vol)
            missed = not adaptive[0] <= gap.move <= adaptive[2]
            if len(scores) > CONFORMAL_MIN:
                level += ACI_GAMMA * ((1.0 if missed else 0.0) - 0.2)
            tracked += 1
            covered += not missed
        bisect.insort(moves, gap.move)
        if friday.ewma_vol > 0:
            bisect.insort(standardised, gap.move / friday.ewma_vol)
    now = state[days[-1].day]
    p10, p50, p90 = widened(scaled_forecast(standardised, moves, now.ewma_vol), now.ewma_vol,
                            scores, level)
    return Band(p10=p10, p50=p50, p90=p90, ewma_vol=now.ewma_vol, weekends=len(pairs),
                since=pairs[0][0].closed, tracked=tracked, covered=covered / tracked)
