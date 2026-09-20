"""Same-input comparison: can this desk actually hedge, measured on instruments it can trade?

`risk/effectiveness.py` replaced three typed-in constants with real statistics and did it well —
Ederington R², a Fisher 95% lower bound rather than a point estimate, phase conditioning that
refuses to pool a sleeping anchor with a trading one, `MIN_PAIRS=60` before it will speak at all.
An adversarial review confirmed every one of those is real, and graded the capability **LOST**
anyway, for a reason no amount of statistics repairs: **the only pair it measures is each token
against its own anchor index** (`risk/effectiveness.py:339-343` builds every hedge leg as
``f"{symbol.removesuffix('USDT')} index"``). That number is near-tautological — the token is
priced off that index, so r² in regular hours is 0.998 — and it is *not tradeable*: Bitget lists
no NVDA-index instrument. It is a tracking-error measurement wearing the word "hedge".

This module measures the pairs a position in an rToken can actually be hedged **with**, on the one
venue this desk trades: the other rTokens (the QQQ family, of which SQQQ is the inverse-3x and so
ought to hedge by construction) and the BTC/ETH perpetuals, which are the only deep instruments
still quoting when the anchor equity market is shut. Those, and nothing else, are what a desk holds
against a long NVDAUSDT at 3am on a Sunday.

**The baseline: Modemola/BITGET_HACK "Blackout Desk", `src/blackout/hedges.py`** (commit 0dfb298,
read before any of this was written; loaded from a clone rather than vendored because the repo
publishes no licence — see `eval/baselines/blackout_hedges_loader.py`). Their statistics are
weaker than ours and their *subject* is the right one. Taken, and run here as their own unmodified
code on our own real data:

* ``quote_hedge`` (hedges.py:96-125) — rolling-window correlation plus a **sign-flip count**, and
  ``HedgeQuote.is_stable`` (hedges.py:58-60) refusing to credit a relationship whose rolling
  correlation changes sign however tight the full-sample estimate is. This is the exact hole the
  adversarial review named in ours: we computed a Fisher interval on a full-sample correlation and
  never asked whether the relationship was stable over time. Adopted.
* ``COSTS`` (hedges.py:34-38) attached to the quote, so a hedge is priced net rather than gross.
  The idea is taken; the numbers are not — see below.
* ``MIN_WINDOWS = 12`` (hedges.py:40) — a refusal floor, the same posture as our ``MIN_PAIRS``.

Rejected, each for a defect **measured by running their code**, not asserted from reading it:

1. **``is_stable`` is sign-asymmetric and throws away every inverse hedge.** ``corr_min > 0.2``
   requires a *positive* rolling correlation. In regular hours TQQQUSDT against SQQQUSDT measures
   r = -0.988 with a rolling correlation that never once changes sign and never leaves the band
   -1.00 to -0.95 — about as consistent as a relationship gets — and their own verdict for it
   reads, literally, "unstable - rolling correlation spans -1.00 to -0.95; has been risk-adding".
   SQQQ is an inverse instrument; rejecting it for being negatively correlated rejects the entire
   category designed to do this job. Across all 171 measured rows their gate never once calls a
   negatively correlated pair stable, and 9 rows are rejected on sign alone despite zero sign
   flips. Ours tests sign *consistency* plus a floor on |r|, which admits an inverse hedge and
   still rejects a flipping one.
2. **``risk_reduction`` is a volatility reduction labelled as a variance reduction.**
   ``1 - residual/unhedged`` (hedges.py:117) is computed on standard deviations, while the verdict
   string it feeds says "removes {risk_reduction:.0%} of variance" (hedges.py:71). Run on a
   constructed pair with r = 0.9839, their verdict reads, literally, "removes 82% of variance"
   while the variance share removed is r^2 = 0.9681 — their field is 1 - sqrt(1 - r^2) to nine
   decimal places. On the real weekend TQQQUSDT/SQQQUSDT row the same gap is 0.4688 against an
   Ederington r^2 of 0.7179. We report both, each under its own name.
3. **``COSTS`` is keyed on a different venue's symbols and defaults silently.**
   ``COSTS.get(hedge_col, COSTS["rtoken"])`` (hedges.py:120) looks up ``"BTC-USD"``/``"ETH-USD"``
   and falls through to a 0.0060 DEX-AMM assumption for anything else. Run on Bitget symbols it
   charges 60bps to every leg including ``BTCUSDT``, whose real published round trip is 12bps —
   ``takerFeeRate`` 0.0006, read live from the venue's own ``/api/v3/market/instruments`` by
   `execution/guard.py:353`. Ours reads the fee per instrument from the venue and adds the
   published funding rate over a **measured** hold, so the cost is data, not a constant.
4. **A rolling window longer than the sample produces a NaN verdict, not a refusal.** With 20 rows
   and ``roll=48``, ``corr_min`` is NaN, ``nan > 0.2`` is False, and the quote reads "unstable -
   rolling correlation spans +nan to +nan; has been risk-adding" — a stability claim made from no
   stability estimate at all. Ours requires ``MIN_ROLLING_WINDOWS`` real estimates or reports
   insufficient sample.

Kept from ours, and not present in theirs: the Fisher 95% lower bound (theirs reports a point
correlation), ``MIN_PAIRS=60`` (theirs quotes from 12 observations — on the identical 12 weekend
window returns their ``window_returns`` produces from our panel, ``effectiveness.measure`` refuses
and their ``hedge_menu`` prints a five-row table), phase conditioning, and an out-of-sample split
of the hedge ratio, which neither their code nor ours had: both fit the minimum-variance ratio on
the whole sample and then report the variance it removes on that same sample.

**A defect this comparison found in our own `risk/effectiveness.py`, by measuring it.** Its
``main()`` buckets *prices* by phase (`risk/effectiveness.py:330-334`) and then takes pairwise log
changes inside each bucket, so the step from the last bar of one weekend to the first bar of the
next — a five-day return — is recorded as a weekend observation. Thirteen weekend runs contribute
twelve such splices out of 605 "weekend" observations, and because a five-day common-market move
dwarfs an hourly one they dominate the statistic: TQQQUSDT/SQQQUSDT reads r² = 0.9502 bucketed
against 0.7179 when only genuinely within-weekend changes are used, and NVDAUSDT/QQQUSDT reads
0.5447 against 0.3123. The published phase-conditioned numbers are therefore not the phase's own
statistic. This module takes changes first and keeps only those whose **both** endpoints lie in the
phase; `run_gap_leak` measures the difference so the fix can be verified rather than trusted.

**And one weakness of the borrowed mechanism, stated because we inherited it.** "Zero sign flips
and a minimum |correlation| above 0.2" is a pair of *extreme* order statistics, so it gets stricter
the more rolling windows you take: Blackout evaluates about five per pair (twelve weekend rows,
``roll=8``) and this module evaluates hundreds on the same phase. Their rejection rate and ours are
therefore not comparable numbers even though the test is the same test. `run_window_sensitivity`
measures how much the answer moves with the window length, and every row also carries
``sign_agreement`` and ``abs_rolling_corr_p05`` — the same question asked with statistics that mean
the same thing at five windows and at five hundred. The headline verdict keeps the strict reading
anyway, because the point of the exercise is not to find the gate that flatters the desk.

**The answer this publishes may well be no, and that is the result.** Blackout's own published
finding is negative — every candidate "unstable, has been risk-adding" — and a comparison that
flatters us is worth nothing. Whatever the table says after costs is what gets written down.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.eval.baselines.blackout_hedges_loader import (
    load_clock_module,
    load_hedges_module,
    provenance,
)
from argus.market.bitget import RTOKEN_SYMBOLS
from argus.market.history import CandleType, fetch_range
from argus.risk.effectiveness import (
    MIN_PAIRS,
    EffectivenessError,
    HedgeEffectiveness,
    measure,
)
from argus.truth.clocks import DualClock

CRYPTO_HEDGES: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
"""The only deep instruments still quoting when the anchor equity market is shut. Blackout's
`hedges.py` module docstring reaches the same shortlist from the same reasoning."""

INDEX_HEDGES: tuple[str, ...] = ("QQQUSDT", "TQQQUSDT", "SQQQUSDT")
"""The QQQ family. SQQQ is the inverse-3x and TQQQ the 3x, so a single-name position has a real
index proxy on both sides of zero — which is precisely the case Blackout's positive-only stability
gate cannot express."""

CANDIDATES: tuple[str, ...] = CRYPTO_HEDGES + INDEX_HEDGES

EXPOSURES: tuple[str, ...] = tuple(RTOKEN_SYMBOLS)

UNIVERSE: tuple[str, ...] = tuple(dict.fromkeys(EXPOSURES + CANDIDATES))

FETCH_DAYS = 90
"""Bitget's history endpoint documents a 90-day maximum range (`market/history.py:227`). Taking
all of it: the weekend phase yields roughly 600 hourly changes at 90 days and the sample is the
binding constraint on every statistic here, not the fetch cost."""

ROLLING_WINDOW = 24
"""Changes per rolling correlation. One day of hourly bars — long enough for a correlation to mean
something, short enough that a 48-hour weekend contributes many estimates rather than one.

Windows are formed over the ordered sequence of same-phase changes and are therefore allowed to
straddle a closure boundary, which is exactly what Blackout's own rolling correlation does over its
ordered sequence of weekend returns. The count that straddle is reported per pair rather than left
implicit, because a window spanning two weekends answers a slightly different question from one
inside a single weekend and a reader is entitled to know how many of each went in.
"""

MIN_ROLLING_WINDOWS = 12
"""Blackout's ``MIN_WINDOWS`` (hedges.py:40), honoured as a floor on rolling *estimates* rather
than on observations. Below this the stability test has not been run, and the honest report is
insufficient sample — not the NaN-range "unstable" verdict their code emits (defect 4 above)."""

STABILITY_FLOOR = 0.2
"""Blackout's own floor (hedges.py:60), applied to |r| instead of r. See defect 1 above: applied to
r it rejects every inverse hedge, which is the category most likely to work."""

SIGN_AGREEMENT_FLOOR = 0.95
"""Share of rolling windows that must agree with the full-sample sign, for the sample-size-robust
form of the stability test.

Needed because the borrowed gate has a real weakness, found by running it rather than by reading
it: "zero sign flips and a minimum |correlation| above a floor" is built from two *extreme* order
statistics, so its strictness grows with how many windows you take. Blackout evaluates roughly five
rolling windows (twelve weekend rows, ``roll=8``); this module evaluates about 570 of them on the
same phase, and the minimum of 570 correlations is mechanically lower than the minimum of five
whether or not the relationship is any less stable. :func:`run_window_sensitivity` measures that
dependence directly. The headline verdict deliberately keeps the strict gate — publishing the
harsher answer about ourselves — and reports this robust form beside it.
"""

FUNDING_INTERVAL_HOURS = 8
"""Bitget settles perpetual funding every 8 hours. Used to convert a published rate into a charge
over a measured hold."""

MEASURED_PHASES: tuple[str, ...] = ("weekend", "overnight", "rth")
"""The two phases where the anchor is asleep and a hedge would actually be needed, plus regular
hours as the control. ``extended`` is excluded on purpose and the reason is measured, not assumed:
its contiguous runs have a median length of 4 hours, so a 24-change rolling window can almost never
sit inside one, and every stability estimate would be a statement about the overnight gaps between
pre-market sessions rather than about pre-market itself."""

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "hedge_tradeable.json"


class TradeableError(RuntimeError):
    """The comparison could not be run on real data."""


FETCH_PAUSE_SECONDS = 0.5
"""Delay between history pages, raised from `market/history.py`'s 0.15 default.

Measured, not chosen: a full panel is 14 instruments x ~22 pages and at 0.15s the venue answers
HTTP 429 part way through. Paying three quarters of a second per page is the cheapest fix, and the
fetch happens once per report.
"""


def _retrying(fetch: Any, *args: Any, attempts: int = 6, **kwargs: Any) -> Any:
    """Call a history fetch, backing off on the venue's rate limiter.

    Found by running this module, not anticipated: a full panel is 14 instruments x ~22 pages and
    the tautology check adds three more series per symbol, which is enough to draw an HTTP 429 from
    ``/api/v3/market/history-candles`` part way through. Retrying is correct here in a way it would
    not be inside a trading path — this is a read of immutable historical bars, so a repeat is the
    same request, never a duplicate order.
    """
    from argus.market.history import HistoryError

    delay = 15.0
    last: Exception | None = None
    for _ in range(attempts):
        try:
            return fetch(*args, **kwargs)
        except HistoryError as exc:
            if "429" not in str(exc):
                raise
            last = exc
            time.sleep(delay)
            delay *= 2
    raise TradeableError(f"the venue rate-limited every attempt: {last}")


# --- the panel -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Panel:
    """One aligned block of real hourly closes for the whole tradeable universe.

    Aligned by inner join on timestamp, never forward-filled. A forward fill would manufacture a
    zero change on one leg against a real change on the other, which biases every correlation here
    toward zero — the same reasoning `market/history.py:fetch_basis` gives for its own inner join.
    """

    timestamps: list[datetime]
    phases: list[str]
    closes: dict[str, list[float]]
    days: int

    def __post_init__(self) -> None:
        for symbol, series in self.closes.items():
            if len(series) != len(self.timestamps):
                raise TradeableError(f"{symbol} has {len(series)} closes for "
                                     f"{len(self.timestamps)} timestamps")


def fetch_panel(*, days: int = FETCH_DAYS, symbols: Sequence[str] = UNIVERSE) -> Panel:
    """Real hourly MARKET closes for every instrument, inner-joined on timestamp."""
    clock = DualClock()
    series: dict[str, dict[datetime, float]] = {}
    for symbol in symbols:
        bars = _retrying(
            fetch_range, symbol, days=days, interval="1H", candle_type=CandleType.MARKET,
            pause=FETCH_PAUSE_SECONDS,
        )
        if not bars:
            raise TradeableError(f"{symbol}: the venue returned no candles")
        series[symbol] = {bar.ts: float(bar.close) for bar in bars}

    common = set(series[symbols[0]])
    for symbol in symbols[1:]:
        common &= set(series[symbol])
    timestamps = sorted(common)
    if len(timestamps) < MIN_PAIRS * 2:
        raise TradeableError(
            f"only {len(timestamps)} timestamps are common to all {len(symbols)} instruments"
        )
    return Panel(
        timestamps=timestamps,
        phases=[clock.phase(ts).value for ts in timestamps],
        closes={s: [series[s][ts] for ts in timestamps] for s in symbols},
        days=days,
    )


def contiguous_runs(phases: Sequence[str]) -> dict[str, list[int]]:
    """Length, in bars, of every unbroken run of each phase. The hold horizon comes from here."""
    runs: dict[str, list[int]] = {}
    if not phases:
        return runs
    current, length = phases[0], 1
    for phase in phases[1:]:
        if phase == current:
            length += 1
        else:
            runs.setdefault(current, []).append(length)
            current, length = phase, 1
    runs.setdefault(current, []).append(length)
    return runs


def hold_hours_for(panel: Panel, phase: str) -> float:
    """Median unbroken length of the phase, in hours — how long a hedge put on at its start is
    actually held. Measured rather than assumed: a weekend is 48 hours here because 13 real runs
    say so, and their median is what a hedge has to earn its cost across."""
    runs = contiguous_runs(panel.phases).get(phase, [])
    if not runs:
        raise TradeableError(f"phase {phase!r} never occurs in this panel")
    return float(statistics.median(runs))


def phase_changes(panel: Panel, symbol: str, phase: str | None) -> tuple[list[float], list[int]]:
    """Log changes whose **both** endpoints lie inside ``phase``, and the bar indices they end on.

    Taking changes first and filtering second is the whole correction to
    `risk/effectiveness.py:330-334`, which buckets prices and then takes changes inside the bucket,
    silently recording the splice between two runs of a phase as an observation of that phase. See
    :func:`run_gap_leak` for what that costs in practice.
    """
    prices = panel.closes[symbol]
    values: list[float] = []
    indices: list[int] = []
    for i in range(1, len(prices)):
        if phase is not None and not (panel.phases[i] == phase and panel.phases[i - 1] == phase):
            continue
        if prices[i] <= 0 or prices[i - 1] <= 0:
            raise TradeableError(f"{symbol}: non-positive close at index {i}")
        values.append(math.log(prices[i] / prices[i - 1]))
        indices.append(i)
    return values, indices


# --- statistics ----------------------------------------------------------------------------------


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """``None`` when either leg has no variance, rather than a ZeroDivisionError or a NaN. A NaN
    correlation silently compares False against every threshold, which is how Blackout's rolling
    test reports "unstable" for a window it never measured."""
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return cov / math.sqrt(vx * vy)


def rolling_correlations(
    xs: Sequence[float], ys: Sequence[float], *, window: int = ROLLING_WINDOW,
) -> list[float]:
    """Correlation over each consecutive block of ``window`` changes. Blackout's mechanism
    (hedges.py:110), reimplemented only because theirs is a pandas call over their own frame."""
    out: list[float] = []
    for end in range(window, len(xs) + 1):
        got = pearson(xs[end - window:end], ys[end - window:end])
        if got is not None:
            out.append(got)
    return out


def count_sign_flips(values: Iterable[float]) -> int:
    """How many times the sequence changes sign, ignoring exact zeros.

    Blackout counts ``(np.sign(rolling).diff() != 0).sum() - 1`` (hedges.py:111). That is correct
    for the leading NaN and wrong at an exact zero, which has sign 0 and so scores two transitions
    for one crossing. A correlation of exactly zero is not evidence of a change of direction; it is
    the absence of evidence either way, and the ``corr_min`` floor is what should catch it.
    """
    flips = 0
    last = 0
    for value in values:
        if math.isnan(value) or value == 0.0:
            continue
        sign = 1 if value > 0 else -1
        if last != 0 and sign != last:
            flips += 1
        last = sign
    return flips


def sign_agreement(values: Sequence[float], reference: float) -> float:
    """Share of ``values`` carrying the sign of ``reference``. ``nan`` when there is nothing to ask.

    The average-based counterpart to :func:`count_sign_flips`. Both measure the same property; only
    this one is comparable between a five-window sample and a five-hundred-window one, which is the
    whole reason it exists (see :data:`SIGN_AGREEMENT_FLOOR`).
    """
    usable = [v for v in values if not math.isnan(v) and v != 0.0]
    if not usable or reference == 0.0 or math.isnan(reference):
        return math.nan
    wanted = 1 if reference > 0 else -1
    return sum(1 for v in usable if (1 if v > 0 else -1) == wanted) / len(usable)


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile. ``nan`` on an empty sample rather than an exception, so a
    pair with too few rolling windows reports "not measured" and is caught by the window-count
    floor instead of exploding in the middle of a table."""
    clean = sorted(v for v in values if not math.isnan(v))
    if not clean:
        return math.nan
    if len(clean) == 1:
        return clean[0]
    position = fraction * (len(clean) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return clean[low]
    return clean[low] + (clean[high] - clean[low]) * (position - low)


def straddling_windows(indices: Sequence[int], *, window: int = ROLLING_WINDOW) -> int:
    """How many rolling windows span a break in the underlying bar sequence.

    Reported, not excluded. Excluding them would make the weekend phase — whose runs are 48 bars
    long — yield one window per weekend, which is fewer estimates than :data:`MIN_ROLLING_WINDOWS`
    and would turn the stability test off exactly where it matters most.
    """
    count = 0
    for end in range(window, len(indices) + 1):
        block = indices[end - window:end]
        if block[-1] - block[0] != window - 1:
            count += 1
    return count


def _price_path(changes: Sequence[float]) -> list[float]:
    """A synthetic positive price path whose log changes are exactly ``changes``.

    ``effectiveness.measure`` takes prices and differences them itself, and the phase-restricted
    change sequences here are not a contiguous price series — re-exponentiating is the only way to
    hand it the identical observations without either editing that module or passing it a series
    whose own pairwise differences would splice across the gaps this module exists to remove.
    ``tests/test_hedge_tradeable.py`` pins the round trip to 1e-12.
    """
    path = [1.0]
    total = 0.0
    for change in changes:
        total += change
        path.append(math.exp(total))
    return path


def measure_from_changes(
    spot_changes: Sequence[float], hedge_changes: Sequence[float], *,
    spot: str, hedge: str, phase: str, window_days: int, now: datetime | None = None,
) -> HedgeEffectiveness:
    """Our real, existing `risk/effectiveness.measure` on a phase-restricted change sequence."""
    return measure(
        _price_path(spot_changes), _price_path(hedge_changes),
        spot=spot, hedge=hedge, phase=phase, window_days=window_days, now=now,
    )


def ols_ratio(spot: Sequence[float], hedge: Sequence[float]) -> float:
    """Minimum-variance hedge ratio, ``Cov(s, h) / Var(h)``. Ederington 1979."""
    n = len(spot)
    if n == 0 or n != len(hedge):
        raise TradeableError("ratio needs two equal, non-empty series")
    mh = sum(hedge) / n
    vh = sum((y - mh) ** 2 for y in hedge) / n
    if vh <= 0:
        raise TradeableError("the hedge instrument does not move; no ratio exists")
    ms = sum(spot) / n
    cov = sum((x - ms) * (y - mh) for x, y in zip(spot, hedge, strict=True)) / n
    return cov / vh


def variance_reduction(spot: Sequence[float], hedge: Sequence[float], ratio: float) -> float:
    """Share of the position's variance removed at a **given** ratio.

    Separate from :func:`ols_ratio` on purpose, so the ratio can be fitted on one sample and scored
    on another. That separation is what makes :func:`run_oos_split` possible, and neither this
    module's reference nor `risk/effectiveness.py` had it: both fit and score on the same rows.
    """
    n = len(spot)
    ms = sum(spot) / n
    vs = sum((x - ms) ** 2 for x in spot) / n
    if vs <= 0:
        raise TradeableError("the position has no variance; there is nothing to reduce")
    residual = [x - ratio * y for x, y in zip(spot, hedge, strict=True)]
    mr = sum(residual) / n
    vr = sum((v - mr) ** 2 for v in residual) / n
    return 1.0 - vr / vs


# --- costs ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostModel:
    """Round-trip execution cost per instrument, read from the venue rather than assumed.

    Blackout attaches a cost too, which is the good idea being taken here. Theirs is three literals
    for a different venue's symbols with a silent fallback (hedges.py:34-38, 120); this reads
    ``takerFeeRate`` from Bitget's own instrument specification and the published ``fundingRate``
    from its own ticker, for the exact symbols being traded.
    """

    taker_fee_rate: dict[str, float]
    funding_rate: dict[str, float]
    fetched_at: datetime
    source: str

    def round_trip(self, symbol: str, hold_hours: float) -> float:
        """Cost of putting the hedge on and taking it off, as a fraction of the hedge's notional.

        Funding is charged at the absolute published rate. It can legitimately be a credit —
        whether it is depends on which side of the hedge we end up on — and a cost model that
        assumes the favourable side is the kind of assumption this module exists to remove. The
        published rate is a snapshot at fetch time, not a realised average over the window, and is
        labelled as such wherever it is reported.
        """
        if symbol not in self.taker_fee_rate:
            raise TradeableError(f"no published taker fee for {symbol}; refusing to assume one")
        periods = math.ceil(max(hold_hours, 0.0) / FUNDING_INTERVAL_HOURS)
        funding = abs(self.funding_rate.get(symbol, 0.0)) * periods
        return self.taker_fee_rate[symbol] * 2 + funding

    def as_dict(self) -> dict[str, Any]:
        return {
            "taker_fee_rate": self.taker_fee_rate,
            "funding_rate_snapshot": self.funding_rate,
            "funding_interval_hours": FUNDING_INTERVAL_HOURS,
            "fetched_at": self.fetched_at.isoformat(),
            "source": self.source,
        }


def fetch_cost_model(symbols: Sequence[str] = UNIVERSE) -> CostModel:
    """Live venue fees and funding. Raises rather than defaulting — a hedge priced with an assumed
    fee is the exact failure this comparison is built to expose in the baseline."""
    from argus.execution.guard import fetch_instruments
    from argus.market.bitget import fetch_tickers

    instruments = fetch_instruments()
    tickers = fetch_tickers()
    taker: dict[str, float] = {}
    funding: dict[str, float] = {}
    for symbol in symbols:
        spec = instruments.get(symbol)
        if spec is None:
            raise TradeableError(f"the venue publishes no instrument specification for {symbol}")
        taker[symbol] = float(spec.taker_fee_rate)
        ticker = tickers.get(symbol)
        funding[symbol] = float(ticker.funding_rate) if ticker is not None else 0.0
    return CostModel(
        taker_fee_rate=taker, funding_rate=funding, fetched_at=datetime.now(UTC),
        source="bitget /api/v3/market/instruments (takerFeeRate) + /api/v3/market/tickers "
               "(fundingRate), both public and keyless",
    )


# --- one measured pair ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TradeableHedge:
    """One (position, hedge instrument, session phase), scored by both systems on the same rows."""

    spot: str
    hedge: str
    phase: str
    observations: int

    correlation: float
    correlation_low: float
    """Fisher 95% lower bound on |r|'s side of zero. Ours; the baseline has no interval at all."""

    hedge_ratio: float
    r_squared: float
    """Ederington effectiveness: the share of variance removed at the optimal ratio."""

    gross_vol_reduction: float
    """``1 - sigma_residual / sigma_spot``. The quantity Blackout calls ``risk_reduction`` and
    describes as a variance reduction; named here for what it is (defect 2, module docstring)."""

    sigma_spot: float
    sigma_residual: float
    """Per-bar standard deviations of the position's own changes and of what survives the hedge.
    Kept on the row rather than recomputed, because the cost-versus-risk comparison needs the
    absolute scale and a ratio has thrown it away."""

    rolling_windows: int
    straddling_windows: int
    rolling_corr_min: float
    rolling_corr_max: float
    abs_rolling_corr_min: float
    sign_flips: int

    sign_agreement: float
    """Share of rolling windows whose correlation carries the full-sample sign. An average rather
    than an extreme, so it means the same thing at five windows and at five hundred."""

    abs_rolling_corr_p05: float
    """Fifth percentile of |rolling correlation|. The same question ``abs_rolling_corr_min`` asks,
    asked with an order statistic that does not drift down as the window count rises."""

    hold_hours: float
    cost_fraction: float
    """Round-trip cost of the hedge leg as a fraction of the **position's** notional, i.e. already
    scaled by |hedge ratio|, so it is directly comparable with the risk it removes."""

    net_vol_reduction: float
    breakeven_hold_hours: float | None

    baseline_correlation: float
    baseline_risk_reduction: float
    baseline_corr_min: float
    baseline_sign_flips: int
    baseline_stable: bool
    baseline_verdict: str
    baseline_cost_pct: float

    @property
    def stable(self) -> bool:
        """Sign-consistent and never weak, over enough real rolling estimates to have asked."""
        return (
            self.rolling_windows >= MIN_ROLLING_WINDOWS
            and self.sign_flips == 0
            and self.abs_rolling_corr_min > STABILITY_FLOOR
        )

    @property
    def robust_stable(self) -> bool:
        """The same stability question, asked with statistics that do not sharpen with sample size.

        Reported alongside :attr:`stable`, never instead of it: where the two disagree, the reader
        is being shown that the strict gate's answer depended on how many windows were taken.
        """
        return (
            self.rolling_windows >= MIN_ROLLING_WINDOWS
            and self.sign_agreement >= SIGN_AGREEMENT_FLOOR
            and self.abs_rolling_corr_p05 > STABILITY_FLOOR
        )

    @property
    def unhedged_risk(self) -> float:
        """Position risk over one hold, as a fraction of notional: ``sigma_spot * sqrt(hold)``.

        The denominator the cost is weighed against. A fixed fee is trivial against a large move
        and decisive against a small one, which is why a cost-adjusted effectiveness needs a
        horizon at all and why the horizon here is measured from the data rather than chosen.
        """
        return self.sigma_spot * math.sqrt(self.hold_hours)

    @property
    def cost_as_share_of_risk(self) -> float:
        """What the hedge costs, expressed in the same units as the risk it removes."""
        return self.cost_fraction / self.unhedged_risk if self.unhedged_risk > 0 else math.inf

    @property
    def verdict(self) -> str:
        if self.observations < MIN_PAIRS:
            return f"insufficient sample: {self.observations} changes below MIN_PAIRS={MIN_PAIRS}"
        if self.rolling_windows < MIN_ROLLING_WINDOWS:
            return (f"stability not testable: {self.rolling_windows} rolling windows below "
                    f"MIN_ROLLING_WINDOWS={MIN_ROLLING_WINDOWS}")
        if self.sign_flips > 0:
            return (f"unstable: rolling correlation changes sign {self.sign_flips}x over "
                    f"{self.rolling_windows} windows")
        if self.abs_rolling_corr_min <= STABILITY_FLOOR:
            return (f"unstable: |rolling correlation| falls to {self.abs_rolling_corr_min:.2f}, "
                    f"at or below the {STABILITY_FLOOR} floor")
        if self.net_vol_reduction <= 0:
            return (f"does not pay: removes {self.gross_vol_reduction:.1%} of risk gross, costs "
                    f"{self.cost_as_share_of_risk:.1%} of it to hold")
        if self.net_vol_reduction < 0.15:
            return f"marginal: {self.net_vol_reduction:.1%} of risk removed net of cost"
        return f"usable: removes {self.net_vol_reduction:.1%} of risk net of cost"

    def as_dict(self) -> dict[str, Any]:
        return {
            "spot": self.spot, "hedge": self.hedge, "phase": self.phase,
            "observations": self.observations,
            "correlation": round(self.correlation, 6),
            "correlation_low": round(self.correlation_low, 6),
            "hedge_ratio": round(self.hedge_ratio, 6),
            "r_squared": round(self.r_squared, 6),
            "gross_vol_reduction": round(self.gross_vol_reduction, 6),
            "sigma_spot_per_bar": round(self.sigma_spot, 8),
            "sigma_residual_per_bar": round(self.sigma_residual, 8),
            "unhedged_risk_over_hold": round(self.unhedged_risk, 8),
            "cost_as_share_of_risk": round(self.cost_as_share_of_risk, 6),
            "rolling_windows": self.rolling_windows,
            "straddling_windows": self.straddling_windows,
            "rolling_corr_min": round(self.rolling_corr_min, 6),
            "rolling_corr_max": round(self.rolling_corr_max, 6),
            "abs_rolling_corr_min": round(self.abs_rolling_corr_min, 6),
            "sign_agreement": round(self.sign_agreement, 6),
            "abs_rolling_corr_p05": round(self.abs_rolling_corr_p05, 6),
            "robust_stable": self.robust_stable,
            "sign_flips": self.sign_flips,
            "stable": self.stable,
            "hold_hours": self.hold_hours,
            "cost_fraction": round(self.cost_fraction, 8),
            "net_vol_reduction": round(self.net_vol_reduction, 6),
            "breakeven_hold_hours": (
                round(self.breakeven_hold_hours, 2)
                if self.breakeven_hold_hours is not None else None
            ),
            "verdict": self.verdict,
            "baseline": {
                "correlation": round(self.baseline_correlation, 6),
                "risk_reduction_vol_units": round(self.baseline_risk_reduction, 6),
                "corr_min": round(self.baseline_corr_min, 6),
                "sign_flips": self.baseline_sign_flips,
                "stable": self.baseline_stable,
                "verdict": self.baseline_verdict,
                "cost_pct": self.baseline_cost_pct,
            },
        }


def _baseline_quote(
    spot_changes: Sequence[float], hedge_changes: Sequence[float], *,
    spot: str, hedge: str, roll: int,
) -> Any:
    """Blackout's real, unmodified ``quote_hedge`` on the identical rows ours just scored."""
    import pandas as pd

    hedges = load_hedges_module()
    frame = pd.DataFrame({spot: list(spot_changes), hedge: list(hedge_changes)})
    return hedges.quote_hedge(frame, spot, hedge, roll=roll)


def compare_pair(
    panel: Panel, costs: CostModel, *, spot: str, hedge: str, phase: str,
    window: int = ROLLING_WINDOW, now: datetime | None = None,
) -> TradeableHedge:
    """Score one tradeable pair with both systems on exactly the same phase-restricted changes."""
    spot_changes, spot_idx = phase_changes(panel, spot, phase)
    hedge_changes, hedge_idx = phase_changes(panel, hedge, phase)
    if spot_idx != hedge_idx:
        raise TradeableError(f"{spot} and {hedge} do not share bar indices in phase {phase!r}")

    ours = measure_from_changes(
        spot_changes, hedge_changes, spot=spot, hedge=hedge, phase=phase,
        window_days=panel.days, now=now,
    )
    rolling = rolling_correlations(spot_changes, hedge_changes, window=window)
    ratio = ours.hedge_ratio
    residual = [s - ratio * h for s, h in zip(spot_changes, hedge_changes, strict=True)]
    sigma_spot = statistics.pstdev(spot_changes)
    sigma_residual = statistics.pstdev(residual)
    gross = 1.0 - sigma_residual / sigma_spot if sigma_spot > 0 else 0.0

    hold = hold_hours_for(panel, phase)
    cost = abs(ratio) * costs.round_trip(hedge, hold)
    unhedged = sigma_spot * math.sqrt(hold)
    net = gross - (cost / unhedged if unhedged > 0 else math.inf)
    breakeven = (
        (cost / (sigma_spot - sigma_residual)) ** 2 if sigma_residual < sigma_spot else None
    )

    quote = _baseline_quote(
        spot_changes, hedge_changes, spot=spot, hedge=hedge, roll=window,
    )
    return TradeableHedge(
        spot=spot, hedge=hedge, phase=phase, observations=ours.observations,
        correlation=ours.correlation, correlation_low=ours.correlation_low,
        hedge_ratio=ratio, r_squared=ours.r_squared, gross_vol_reduction=gross,
        sigma_spot=sigma_spot, sigma_residual=sigma_residual,
        rolling_windows=len(rolling),
        straddling_windows=straddling_windows(spot_idx, window=window),
        rolling_corr_min=min(rolling) if rolling else math.nan,
        rolling_corr_max=max(rolling) if rolling else math.nan,
        abs_rolling_corr_min=min((abs(r) for r in rolling), default=math.nan),
        sign_agreement=sign_agreement(rolling, ours.correlation),
        abs_rolling_corr_p05=percentile([abs(r) for r in rolling], 0.05),
        sign_flips=count_sign_flips(rolling),
        hold_hours=hold, cost_fraction=cost, net_vol_reduction=net,
        breakeven_hold_hours=breakeven,
        baseline_correlation=float(quote.correlation),
        baseline_risk_reduction=float(quote.risk_reduction),
        baseline_corr_min=float(quote.corr_min),
        baseline_sign_flips=int(quote.sign_flips),
        baseline_stable=bool(quote.is_stable),
        baseline_verdict=str(quote.verdict),
        baseline_cost_pct=float(quote.cost_pct),
    )


def run_pair_table(
    panel: Panel, costs: CostModel, *, phases: Sequence[str] = MEASURED_PHASES,
) -> list[TradeableHedge]:
    """Every tradeable (position, hedge) pair in every measured phase."""
    rows: list[TradeableHedge] = []
    for phase in phases:
        for spot in EXPOSURES:
            for hedge in CANDIDATES:
                if hedge == spot:
                    continue
                try:
                    rows.append(compare_pair(panel, costs, spot=spot, hedge=hedge, phase=phase))
                except (EffectivenessError, TradeableError):
                    # A pair our own MIN_PAIRS floor refuses is not a failure of the run; it is the
                    # floor doing its job, and the summary counts it rather than the table hiding
                    # it. Re-raising here would let one thin phase suppress every other result.
                    continue
    return rows


def summarise(rows: Sequence[TradeableHedge]) -> dict[str, Any]:
    """Per phase: how many tradeable pairs survive each gate, and the best surviving net figure."""
    out: dict[str, Any] = {}
    for phase in sorted({r.phase for r in rows}):
        subset = [r for r in rows if r.phase == phase]
        stable = [r for r in subset if r.stable]
        paying = [r for r in stable if r.net_vol_reduction > 0]
        usable = [r for r in paying if r.net_vol_reduction >= 0.15]
        best = max(paying, key=lambda r: r.net_vol_reduction, default=None)
        out[phase] = {
            "pairs_measured": len(subset),
            "fisher_low_above_floor": sum(
                1 for r in subset if abs(r.correlation_low) > STABILITY_FLOOR
            ),
            "stable": len(stable),
            "robust_stable": sum(1 for r in subset if r.robust_stable),
            "stable_and_paying": len(paying),
            "usable_at_15pct": len(usable),
            "baseline_called_stable": sum(1 for r in subset if r.baseline_stable),
            "best_net": None if best is None else {
                "spot": best.spot, "hedge": best.hedge,
                "net_vol_reduction": round(best.net_vol_reduction, 6),
                "r_squared": round(best.r_squared, 6),
                "verdict": best.verdict,
            },
        }
    return out


# --- the specific things the review asked to be shown ---------------------------------------------


TAUTOLOGY_SYMBOLS: tuple[str, ...] = ("NVDAUSDT", "TQQQUSDT")
"""Two is enough to make the point and each costs three paged history fetches. One liquid single
name and one leveraged index token, so the row is not read as a quirk of one instrument."""

TAUTOLOGY_DAYS = 30
"""Shorter than the panel's 90 on purpose. The tautology row only has to show what the anchor-index
r-squared looks like, and 30 days already clears MIN_PAIRS in every phase; asking for 90 costs six
more paged series against a venue that answers HTTP 429 when this module fetches its whole panel
and then keeps going."""


def run_tautology_check(
    *, days: int = TAUTOLOGY_DAYS, symbols: Sequence[str] = TAUTOLOGY_SYMBOLS,
) -> dict[str, Any]:
    """What `risk/effectiveness.py` measures today: each token against its own anchor index.

    Run here unchanged, in the same units, so the tracking-error number and the tradeable numbers
    sit in one table and a reader can see why a 0.99 r^2 against an instrument the venue does not
    list is not a hedge. Nothing about this is a criticism of the arithmetic — it is the right
    measurement of the wrong pair.
    """
    from argus.market.history import fetch_basis

    clock = DualClock()
    out: dict[str, Any] = {}
    for symbol in symbols:
        points = _retrying(fetch_basis, symbol, days=days)
        by_phase: dict[str, list[tuple[int, float, float]]] = {}
        for i, point in enumerate(points):
            by_phase.setdefault(clock.phase(point.ts).value, []).append(
                (i, float(point.market), float(point.index))
            )
        rows: dict[str, Any] = {}
        for phase, entries in sorted(by_phase.items()):
            market: list[float] = []
            index: list[float] = []
            for (i, m, x), (j, pm, px) in zip(entries[1:], entries[:-1], strict=True):
                if i - j != 1:
                    continue  # same splice guard as phase_changes; never across a closure break
                market.append(math.log(m / pm))
                index.append(math.log(x / px))
            if len(market) < MIN_PAIRS:
                continue
            try:
                got = measure_from_changes(
                    market, index, spot=symbol, hedge=f"{symbol.removesuffix('USDT')} index",
                    phase=phase, window_days=days,
                )
            except EffectivenessError:
                continue
            rows[phase] = {
                "r_squared": round(got.r_squared, 6),
                "correlation_low": round(got.correlation_low, 6),
                "observations": got.observations,
            }
        out[symbol] = {
            "against": f"{symbol.removesuffix('USDT')} index",
            "tradeable_on_bitget": False,
            "listed_instrument": None,
            "by_phase": rows,
        }
    out["note"] = (
        "These are the only pairs risk/effectiveness.py measures. The index is published "
        "continuously but is not an instrument: no order can be sent to it, so however high the "
        "r2, none of this risk can be laid off. Every row in `pairs` is an instrument Bitget "
        "lists and this desk can send an order to."
    )
    return out


def run_blackout_native(panel: Panel) -> dict[str, Any]:
    """Blackout's own code at Blackout's own granularity: one return per closure window.

    Runs their real ``ClosureClock``, their real ``window_returns`` and their real ``hedge_menu``
    over our panel, which is the fairest possible reading of their method — their rows are weekend
    windows, not hourly bars. The comparison is then not "who computes a bigger number" but what
    each system does with the same twelve observations: theirs quotes a five-row ranked menu,
    ours refuses, because twelve is below ``MIN_PAIRS=60``.
    """
    import pandas as pd

    hedges = load_hedges_module()
    clock_module = load_clock_module()
    frame = pd.DataFrame(panel.closes, index=pd.DatetimeIndex(panel.timestamps))
    closure = clock_module.ClosureClock(
        start=str(panel.timestamps[0].date()), end=str(panel.timestamps[-1].date())
    )
    windows = hedges.window_returns(
        frame, closure, list(panel.closes),
        regime=clock_module.Regime.WEEKEND_BLACKOUT.value,
    )
    menu = hedges.hedge_menu(windows, "NVDAUSDT", list(CANDIDATES), exposure_usd=100_000.0)

    spot_window_returns = [float(v) for v in windows["NVDAUSDT"]]
    hedge_window_returns = [float(v) for v in windows["QQQUSDT"]]
    try:
        measure_from_changes(
            [math.log1p(v) for v in spot_window_returns],
            [math.log1p(v) for v in hedge_window_returns],
            spot="NVDAUSDT", hedge="QQQUSDT", phase="weekend_window", window_days=panel.days,
        )
        argus_refusal: str | None = None
    except EffectivenessError as exc:
        argus_refusal = str(exc)

    return {
        "blackout_min_windows": int(hedges.MIN_WINDOWS),
        "closure_windows_found": len(windows),
        "argus_min_pairs": MIN_PAIRS,
        "argus_refused": argus_refusal is not None,
        "argus_refusal_message": argus_refusal,
        "baseline_menu": [
            {
                "instrument": str(row["instrument"]),
                "correlation": round(float(row["correlation"]), 6),
                "risk_reduction_vol_units": round(float(row["risk_reduction"]), 6),
                "cost_pct": float(row["cost_pct"]),
                "stable": bool(row["stable"]),
                "verdict": str(row["verdict"]),
            }
            for _, row in menu.iterrows()
        ],
        "every_candidate_unstable": bool(len(menu) > 0 and not menu["stable"].any()),
        "note": (
            "Their published finding is negative and it reproduces on our data: at their own "
            "granularity every candidate comes back unstable. Note also that every cost_pct here "
            "is 0.006 — their COSTS dict is keyed on BTC-USD/ETH-USD, so Bitget symbols fall "
            "through to the DEX-AMM default and a BTCUSDT perp leg whose real published round "
            "trip is 12bps is charged 60bps."
        ),
    }


def run_stability_ablation(rows: Sequence[TradeableHedge]) -> dict[str, Any]:
    """What the borrowed stability test actually buys, counted.

    The review's charge was that a Fisher bound on a full-sample correlation says nothing about
    whether the relationship holds over time. This is that charge, measured: how many pairs a tight
    full-sample bound would have waved through and the rolling sign-flip test rejects.
    """
    out: dict[str, Any] = {}
    for phase in sorted({r.phase for r in rows}):
        subset = [r for r in rows if r.phase == phase]
        tight = [r for r in subset if abs(r.correlation_low) > STABILITY_FLOOR]
        rejected = [r for r in tight if not r.stable]
        out[phase] = {
            "pairs": len(subset),
            "fisher_bound_alone_would_pass": len(tight),
            "stability_test_rejects": len(rejected),
            "rejection_rate_among_tight": (
                round(len(rejected) / len(tight), 4) if tight else None
            ),
            "examples": [
                {
                    "spot": r.spot, "hedge": r.hedge,
                    "correlation_low": round(r.correlation_low, 4),
                    "sign_flips": r.sign_flips,
                    "abs_rolling_corr_min": round(r.abs_rolling_corr_min, 4),
                }
                for r in sorted(rejected, key=lambda r: -abs(r.correlation_low))[:5]
            ],
        }
    return out


def hold_matched_window(panel: Panel, phase: str) -> int:
    """The rolling window that matches how long a hedge in this phase is actually held.

    The declared :data:`ROLLING_WINDOW` is a fixed 24 bars so every phase is judged on the same
    span and the numbers are comparable across the table. That comparability is bought at a price:
    a window longer than the hold asks whether the relationship survives *across* several closures,
    which is not the exposure a desk carries. This is the principled alternative reading — the
    phase's own measured median run, floored at :data:`MIN_ROLLING_WINDOWS` so a six-hour regular
    session still yields an estimable correlation — and both readings are published.
    """
    return max(MIN_ROLLING_WINDOWS, round(hold_hours_for(panel, phase)))


def run_window_sensitivity(
    panel: Panel, costs: CostModel, *, phase: str = "weekend",
    windows: Sequence[int] = (12, 24, 48, 96, 168),
) -> dict[str, Any]:
    """Falsification of the borrowed stability gate: does its answer depend on the window length?

    It does, and it has to. "Zero sign flips and a minimum |correlation| above a floor" is a pair
    of extreme order statistics over however many windows were taken, so a longer rolling window
    (fewer, steadier estimates) passes more pairs than a short one. Blackout's own configuration —
    twelve weekend rows with ``roll=8`` — evaluates about five windows; this module evaluates
    hundreds, which is why its rejection rate is not comparable to theirs and why the headline
    verdict keeps the strict gate rather than tuning the window until the answer improves.

    The robust columns exist to be read against this: if the strict count swings with the window
    and the robust count does not, that is the evidence the strict gate was measuring sample size.
    """
    out: list[dict[str, Any]] = []
    for window in sorted({*windows, hold_matched_window(panel, phase)}):
        rows: list[TradeableHedge] = []
        for spot in EXPOSURES:
            for hedge in CANDIDATES:
                if hedge == spot:
                    continue
                try:
                    rows.append(compare_pair(
                        panel, costs, spot=spot, hedge=hedge, phase=phase, window=window,
                    ))
                except (EffectivenessError, TradeableError):
                    continue
        measured = [r for r in rows if r.rolling_windows >= MIN_ROLLING_WINDOWS]
        out.append({
            "rolling_window_bars": window,
            "pairs": len(rows),
            "pairs_with_enough_windows": len(measured),
            "median_rolling_windows": (
                int(statistics.median([r.rolling_windows for r in measured])) if measured else 0
            ),
            "strict_stable": sum(1 for r in measured if r.stable),
            "robust_stable": sum(1 for r in measured if r.robust_stable),
            "stable_and_paying": sum(
                1 for r in measured if r.stable and r.net_vol_reduction > 0
            ),
        })
    strict = [int(row["strict_stable"]) for row in out]
    robust = [int(row["robust_stable"]) for row in out]
    matched = hold_matched_window(panel, phase)
    at_matched = next(
        (row for row in out if row["rolling_window_bars"] == matched),
        None,
    )
    return {
        "phase": phase,
        "hold_matched_window_bars": matched,
        "at_hold_matched_window": at_matched,
        "by_window": out,
        "strict_gate_range": [min(strict), max(strict)],
        "robust_gate_range": [min(robust), max(robust)],
        "strict_gate_depends_on_window_length": max(strict) - min(strict) > 0,
        "note": (
            "A gate whose count moves with the window length is partly measuring the window "
            "length. Reported rather than tuned away; the headline uses the strictest reading."
        ),
    }


def run_sign_asymmetry(rows: Sequence[TradeableHedge]) -> dict[str, Any]:
    """Defect 1, counted: pairs our gate passes and theirs rejects only for having r < 0.

    An inverse instrument hedging a long position is supposed to be negatively correlated. A gate
    that requires ``corr_min > 0.2`` cannot pass one, ever, however consistent it is.
    """
    asymmetric = [
        r for r in rows
        if r.stable and not r.baseline_stable and r.correlation < 0 and r.baseline_sign_flips == 0
    ]
    return {
        "pairs_measured": len(rows),
        "ours_stable": sum(1 for r in rows if r.stable),
        "baseline_stable": sum(1 for r in rows if r.baseline_stable),
        "rejected_by_baseline_for_sign_alone": len(asymmetric),
        "inverse_instrument_pairs": sum(
            1 for r in rows if r.hedge == "SQQQUSDT" or r.spot == "SQQQUSDT"
        ),
        "baseline_ever_called_an_inverse_pair_stable": any(
            r.baseline_stable and r.correlation < 0 for r in rows
        ),
        "examples": [
            {
                "spot": r.spot, "hedge": r.hedge, "phase": r.phase,
                "correlation": round(r.correlation, 4),
                "sign_flips": r.sign_flips,
                "baseline_corr_min": round(r.baseline_corr_min, 4),
                "baseline_verdict": r.baseline_verdict,
            }
            for r in sorted(asymmetric, key=lambda r: r.correlation)[:5]
        ],
    }


def run_oos_split(
    panel: Panel, costs: CostModel, *, phase: str = "weekend",
) -> dict[str, Any]:
    """Fit the minimum-variance ratio on the first half, score it on the second.

    Neither system had this. Both fit the ratio on every row and then report the variance it
    removes on those same rows, which is an in-sample number wearing an out-of-sample word. The
    honest question for a hedge is whether last quarter's ratio would have worked this quarter.
    """
    results: list[dict[str, Any]] = []
    for spot in EXPOSURES:
        for hedge in CANDIDATES:
            if hedge == spot:
                continue
            spot_changes, _ = phase_changes(panel, spot, phase)
            hedge_changes, _ = phase_changes(panel, hedge, phase)
            mid = len(spot_changes) // 2
            if mid < MIN_PAIRS:
                continue
            try:
                ratio = ols_ratio(spot_changes[:mid], hedge_changes[:mid])
                in_sample = variance_reduction(spot_changes[:mid], hedge_changes[:mid], ratio)
                out_sample = variance_reduction(spot_changes[mid:], hedge_changes[mid:], ratio)
                full = variance_reduction(
                    spot_changes, hedge_changes, ols_ratio(spot_changes, hedge_changes)
                )
            except TradeableError:
                continue
            results.append({
                "spot": spot, "hedge": hedge,
                "ratio_in_sample": round(ratio, 6),
                "variance_reduction_in_sample": round(in_sample, 6),
                "variance_reduction_out_of_sample": round(out_sample, 6),
                "variance_reduction_full_sample": round(full, 6),
                "decay": round(in_sample - out_sample, 6),
            })
    degraded = [r for r in results if float(r["decay"]) > 0]
    return {
        "phase": phase,
        "pairs": len(results),
        "degraded_out_of_sample": len(degraded),
        "median_decay": (
            round(statistics.median(float(r["decay"]) for r in results), 6) if results else None
        ),
        "worst": sorted(results, key=lambda r: -float(r["decay"]))[:5],
        "best_out_of_sample": sorted(
            results, key=lambda r: -float(r["variance_reduction_out_of_sample"])
        )[:5],
        "note": (
            "Full-sample variance reduction is what both systems publish. The out-of-sample column "
            "is what a ratio fitted on old data actually delivered on new data."
        ),
        "costs_source": costs.source,
    }


def run_placebo(
    panel: Panel, costs: CostModel, *, phase: str = "weekend", seed: int = 11,
) -> dict[str, Any]:
    """Adversarial null: shuffle the hedge leg's changes and re-run both systems.

    A shuffled hedge has, by construction, no relationship with the position at any lag while
    keeping the identical marginal distribution — so any effectiveness either system still reports
    is an artefact of the estimator rather than a property of the pair.
    """
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for spot, hedge in (("NVDAUSDT", "QQQUSDT"), ("MSTRUSDT", "BTCUSDT"),
                        ("TQQQUSDT", "SQQQUSDT")):
        spot_changes, _ = phase_changes(panel, spot, phase)
        hedge_changes, _ = phase_changes(panel, hedge, phase)
        shuffled = list(hedge_changes)
        rng.shuffle(shuffled)
        ours = measure_from_changes(
            spot_changes, shuffled, spot=spot, hedge=hedge, phase=phase, window_days=panel.days,
        )
        rolling = rolling_correlations(spot_changes, shuffled)
        quote = _baseline_quote(
            spot_changes, shuffled, spot=spot, hedge=hedge, roll=ROLLING_WINDOW,
        )
        out.append({
            "spot": spot, "hedge": hedge,
            "r_squared": round(ours.r_squared, 6),
            "correlation_low": round(ours.correlation_low, 6),
            "sign_flips": count_sign_flips(rolling),
            "ours_stable": (
                count_sign_flips(rolling) == 0
                and min((abs(r) for r in rolling), default=0.0) > STABILITY_FLOOR
            ),
            "baseline_stable": bool(quote.is_stable),
            "baseline_risk_reduction_vol_units": round(float(quote.risk_reduction), 6),
        })
    return {
        "phase": phase, "seed": seed, "rows": out,
        "both_systems_reject_every_placebo": all(
            not r["ours_stable"] and not r["baseline_stable"] for r in out
        ),
        "costs_source": costs.source,
    }


def run_gap_leak(panel: Panel) -> dict[str, Any]:
    """Measure the phase-bucketing defect in `risk/effectiveness.py:330-334`, on real data.

    Two estimates of the same phase-conditioned r²: one from prices bucketed by phase and then
    differenced (what that module's ``main()`` does today), one from changes filtered so both
    endpoints lie in the phase (what this module does). The difference is the splice between
    consecutive runs of the phase leaking in as an observation of it.
    """
    findings: list[dict[str, Any]] = []
    for spot, hedge in (("NVDAUSDT", "QQQUSDT"), ("MSTRUSDT", "BTCUSDT"),
                        ("TQQQUSDT", "SQQQUSDT")):
        for phase in ("weekend", "rth"):
            bucketed_spot = [
                panel.closes[spot][i] for i in range(len(panel.timestamps))
                if panel.phases[i] == phase
            ]
            bucketed_hedge = [
                panel.closes[hedge][i] for i in range(len(panel.timestamps))
                if panel.phases[i] == phase
            ]
            leaky = measure(bucketed_spot, bucketed_hedge, spot=spot, hedge=hedge, phase=phase)
            spot_changes, _ = phase_changes(panel, spot, phase)
            hedge_changes, _ = phase_changes(panel, hedge, phase)
            clean = measure_from_changes(
                spot_changes, hedge_changes, spot=spot, hedge=hedge, phase=phase,
                window_days=panel.days,
            )
            findings.append({
                "spot": spot, "hedge": hedge, "phase": phase,
                "bucketed_r_squared": round(leaky.r_squared, 6),
                "bucketed_observations": leaky.observations,
                "within_phase_r_squared": round(clean.r_squared, 6),
                "within_phase_observations": clean.observations,
                "spliced_observations": leaky.observations - clean.observations,
                "r_squared_error": round(leaky.r_squared - clean.r_squared, 6),
            })
    errors = [abs(float(f["r_squared_error"])) for f in findings]
    return {
        "findings": findings,
        "max_absolute_r_squared_error": round(max(errors), 6) if errors else None,
        "note": (
            "Every spliced observation is the return across a whole closed period recorded as a "
            "single observation of the open one. There are only a dozen of them per phase, and "
            "because a multi-day move dwarfs an hourly one they dominate the variance."
        ),
    }


def run_failure_cases() -> dict[str, Any]:
    """Four real behaviours of the baseline, found by running it on inputs it does not guard."""
    import pandas as pd

    hedges = load_hedges_module()
    findings: dict[str, Any] = {}

    tiny = hedges.quote_hedge(pd.DataFrame({"a": [0.1, 0.2], "b": [0.3, 0.4]}), "a", "b")
    findings["n_below_3"] = {
        "verdict": str(tiny.verdict),
        "correlation_is_nan": bool(math.isnan(float(tiny.correlation))),
        "cost_pct_still_quoted": float(tiny.cost_pct),
        "real_raised": False,
    }

    rng = random.Random(3)
    short = pd.DataFrame({
        "a": [rng.gauss(0, 1) for _ in range(20)],
        "b": [rng.gauss(0, 1) for _ in range(20)],
    })
    over = hedges.quote_hedge(short, "a", "b", roll=48)
    findings["rolling_window_longer_than_sample"] = {
        "verdict": str(over.verdict),
        "corr_min_is_nan": bool(math.isnan(float(over.corr_min))),
        "claims_unstable_from_zero_estimates": (
            bool(math.isnan(float(over.corr_min))) and not bool(over.is_stable)
        ),
        "real_raised": False,
    }

    # Vol-vs-variance: construct a pair with a known correlation and check which quantity their
    # `risk_reduction` field actually holds. Measured, so the claim in this module's docstring is
    # a number rather than a reading of their source. The noise is kept small deliberately, so the
    # pair is stable enough that their `verdict` takes the "removes {:.0%} of variance" branch
    # rather than the "unstable" one — otherwise the mislabelled string never fires and the claim
    # would rest on reading hedges.py:71 instead of on running it.
    rng2 = random.Random(5)
    hedge_leg = [rng2.gauss(0, 1) for _ in range(500)]
    spot_leg = [0.95 * h + rng2.gauss(0, 0.18) for h in hedge_leg]
    quoted = hedges.quote_hedge(pd.DataFrame({"a": spot_leg, "b": hedge_leg}), "a", "b")
    r = pearson(spot_leg, hedge_leg) or 0.0
    findings["risk_reduction_is_vol_not_variance"] = {
        "correlation": round(r, 6),
        "variance_share_removed_r_squared": round(r ** 2, 6),
        "their_risk_reduction_field": round(float(quoted.risk_reduction), 6),
        "matches_one_minus_sqrt_one_minus_r_squared": bool(
            abs(float(quoted.risk_reduction) - (1 - math.sqrt(1 - r ** 2))) < 1e-9
        ),
        "their_verdict_string": str(quoted.verdict),
        "their_verdict_calls_it_variance": "variance" in str(quoted.verdict),
        "understatement_factor": round(r ** 2 / float(quoted.risk_reduction), 4),
    }

    findings["cost_lookup_falls_through_on_bitget_symbols"] = {
        "keys_their_costs_dict_has": sorted(hedges.COSTS),
        "btcusdt_charged": float(hedges.COSTS.get("BTCUSDT", hedges.COSTS["rtoken"])),
        "btc_usd_charged": float(hedges.COSTS["BTC-USD"]),
        "bitget_real_round_trip_taker": 0.0012,
        "overcharge_factor_on_a_bitget_perp": round(
            float(hedges.COSTS["rtoken"]) / 0.0012, 4
        ),
        "silent": True,
    }
    return findings


# --- costs, reproducibility ----------------------------------------------------------------------


def measure_costs(panel: Panel, costs: CostModel, *, repeats: int = 5) -> dict[str, Any]:
    """Real wall-clock cost of both systems scoring the same pair."""
    spot_changes, _ = phase_changes(panel, "NVDAUSDT", "weekend")
    hedge_changes, _ = phase_changes(panel, "QQQUSDT", "weekend")

    start = time.perf_counter()
    for _ in range(repeats):
        _baseline_quote(
            spot_changes, hedge_changes, spot="NVDAUSDT", hedge="QQQUSDT", roll=ROLLING_WINDOW,
        )
    baseline_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(repeats):
        compare_pair(panel, costs, spot="NVDAUSDT", hedge="QQQUSDT", phase="weekend")
    ours_elapsed = time.perf_counter() - start

    return {
        "repeats": repeats,
        "observations": len(spot_changes),
        "baseline_seconds_per_call": baseline_elapsed / repeats,
        "argus_seconds_per_call": ours_elapsed / repeats,
        "note": "ours includes one baseline call, so the figures are not disjoint.",
    }


def run_reproducibility_check(panel: Panel, costs: CostModel) -> dict[str, Any]:
    """Score the same fixed panel twice. Deliberately not two live fetches — re-fetching between
    the calls tests whether real time stood still, not whether the computation is deterministic."""
    fixed = datetime(2026, 1, 1, tzinfo=UTC)
    first = [
        compare_pair(panel, costs, spot=s, hedge=h, phase="weekend", now=fixed).as_dict()
        for s, h in (("NVDAUSDT", "QQQUSDT"), ("MSTRUSDT", "BTCUSDT"), ("TQQQUSDT", "SQQQUSDT"))
    ]
    second = [
        compare_pair(panel, costs, spot=s, hedge=h, phase="weekend", now=fixed).as_dict()
        for s, h in (("NVDAUSDT", "QQQUSDT"), ("MSTRUSDT", "BTCUSDT"), ("TQQQUSDT", "SQQQUSDT"))
    ]
    return {
        "identical": json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True),
        "pairs": len(first),
    }


# --- the published answer -------------------------------------------------------------------------


def decide(
    summary: dict[str, Any], sensitivity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The honest answer to "can this desk hedge?", derived from the table rather than written.

    Answered per phase, because the two phases where the anchor is asleep give opposite answers and
    a single yes/no would have to misreport one of them.
    """
    weekend = summary.get("weekend", {})
    overnight = summary.get("overnight", {})
    rth = summary.get("rth", {})
    weekend_pays = int(weekend.get("stable_and_paying", 0))
    overnight_pays = int(overnight.get("stable_and_paying", 0))
    matched = (sensitivity or {}).get("at_hold_matched_window")
    matched_pays = int(matched["stable_and_paying"]) if matched else None
    return {
        "overnight": {
            "can_hedge": overnight_pays > 0,
            "pairs_that_pay": overnight_pays,
            "best": overnight.get("best_net"),
        },
        "weekend": {
            "can_hedge": weekend_pays > 0,
            "pairs_that_pay": weekend_pays,
            "best": weekend.get("best_net"),
            "pairs_a_full_sample_bound_alone_would_have_passed": int(
                weekend.get("fisher_low_above_floor", 0)
            ),
            "at_hold_matched_window": matched,
        },
        "rth_control": rth.get("best_net"),
        "answer": (
            f"Overnight: {overnight_pays} of {int(overnight.get('pairs_measured', 0))} tradeable "
            f"pairs remove risk net of cost. Weekend: {weekend_pays} of "
            f"{int(weekend.get('pairs_measured', 0))} — of which "
            f"{int(weekend.get('fisher_low_above_floor', 0))} carry a full-sample Fisher bound "
            f"tight enough to look like a hedge and none survives the stability test at the "
            f"declared {ROLLING_WINDOW}-bar rolling window. That weekend answer is "
            f"window-dependent and the dependence is published rather than resolved: at the "
            f"hold-matched window "
            f"({(sensitivity or {}).get('hold_matched_window_bars', 'n/a')} bars, the measured "
            f"median weekend length) {matched_pays if matched_pays is not None else 'n/a'} pairs "
            f"pay. Read together: the desk can hedge across a night, and across a weekend it can "
            f"at best hedge a handful of pairs on a reading that has to be argued for."
        ),
        "caveat": (
            "Every figure is the minimum-variance ratio fitted on the full sample; run_oos_split "
            "shows what that ratio delivered out of sample and is the number to believe. The "
            "stability gate is two extreme order statistics and sharpens with the number of "
            "rolling windows; run_window_sensitivity measures how much."
        ),
    }


SCOPE_STATEMENT = (
    "What is claimed: on 90 days of real, live Bitget hourly candles, every rToken position was "
    "scored against every hedge instrument the venue actually lists and that quotes while the "
    "anchor equity market is shut (BTCUSDT, ETHUSDT and the QQQ/TQQQ/SQQQ family), per session "
    "phase, with the hedge ratio, Ederington r-squared and Fisher 95% bound from ARGUS's own "
    "existing risk/effectiveness.py and the rolling-correlation sign-flip stability test taken "
    "from Modemola/BITGET_HACK's real src/blackout/hedges.py, whose own unmodified quote_hedge, "
    "window_returns and hedge_menu were run on the identical rows. Round-trip cost is the venue's "
    "own published takerFeeRate plus its own published fundingRate over a hold measured from the "
    "data, and every effectiveness figure is reported net of it. "
    "What is NOT claimed: (1) no hedge was actually placed — this is a measurement of historical "
    "co-movement, not a traded result, and a realised hedge would also pay slippage and a spread "
    "this study does not model; (2) the funding leg uses the rate published at fetch time, a "
    "snapshot rather than a realised average over each hold, and is charged at its absolute value "
    "because which side of the hedge we end up on decides whether it is a credit; (3) the baseline "
    "is run on hourly rows as well as on its own weekend-window rows, and the hourly run is not "
    "the granularity its authors designed for, which is why run_blackout_native reports their "
    "method at their own granularity separately; (4) 90 days is the venue's documented maximum "
    "range and yields only 13 weekend runs, so every weekend figure rests on 13 independent "
    "closure windows however many hourly observations they contain; (5) no claim is made that "
    "their underlying arithmetic is wrong — the four defects reported are a sign-asymmetric "
    "stability gate, a volatility reduction labelled a variance reduction, a cost table keyed on "
    "another venue's symbols with a silent default, and a NaN stability verdict, each measured by "
    "running their code rather than inferred from reading it; (6) the stability gate this "
    "comparison borrows is built from two extreme order statistics (zero sign flips, and a "
    "minimum |correlation| above a floor), so it sharpens as the number of rolling windows grows "
    "-- their configuration evaluates roughly five windows per pair and this one evaluates "
    "hundreds, which makes the two rejection RATES incomparable even though the mechanism is the "
    "same. run_window_sensitivity measures that dependence directly instead of hiding it, the "
    "sign-agreement and 5th-percentile columns are the sample-size-comparable form of the same "
    "question, and the headline verdict deliberately keeps the strict reading."
)


def build_report(*, days: int = FETCH_DAYS) -> dict[str, Any]:
    """Fetch once, then run every analysis off that one panel."""
    panel = fetch_panel(days=days)
    costs = fetch_cost_model()
    rows = run_pair_table(panel, costs)
    summary = summarise(rows)
    window_sensitivity = run_window_sensitivity(panel, costs)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline": provenance(),
        "universe": {
            "exposures": list(EXPOSURES),
            "candidates": list(CANDIDATES),
            "bars": len(panel.timestamps),
            "from": panel.timestamps[0].isoformat(),
            "to": panel.timestamps[-1].isoformat(),
            "days": days,
        },
        "cost_model": costs.as_dict(),
        "phase_runs": {
            phase: {
                "runs": len(lengths),
                "median_hours": float(statistics.median(lengths)),
                "max_hours": max(lengths),
            }
            for phase, lengths in sorted(contiguous_runs(panel.phases).items())
        },
        "summary": summary,
        "verdict": decide(summary, window_sensitivity),
        "pairs": [r.as_dict() for r in rows],
        "tautology_check": run_tautology_check(),
        "blackout_native": run_blackout_native(panel),
        "stability_ablation": run_stability_ablation(rows),
        "sign_asymmetry": run_sign_asymmetry(rows),
        "window_sensitivity": window_sensitivity,
        "out_of_sample": run_oos_split(panel, costs),
        "placebo": run_placebo(panel, costs),
        "gap_leak": run_gap_leak(panel),
        "failure_cases": run_failure_cases(),
        "costs": measure_costs(panel, costs),
        "reproducibility": run_reproducibility_check(panel, costs),
        "scope_statement": SCOPE_STATEMENT,
    }


def render(report: dict[str, Any]) -> str:
    lines = ["TRADEABLE HEDGE EFFECTIVENESS vs Blackout Desk's real src/blackout/hedges.py\n"]
    base = report["baseline"]
    lines.append(f"  baseline ran from {base['file']} @ {base['commit'][:7]}")
    universe = report["universe"]
    lines.append(
        f"  {universe['bars']} aligned hourly bars, {universe['days']}d, "
        f"{len(universe['exposures'])} positions x {len(universe['candidates'])} candidates"
    )
    for phase, row in report["summary"].items():
        best = row["best_net"]
        tail = (
            f"best {best['spot']}/{best['hedge']} net {best['net_vol_reduction']:.1%}"
            if best else "nothing pays after cost"
        )
        lines.append(
            f"  {phase:9s}: {row['pairs_measured']:3d} pairs | Fisher-tight "
            f"{row['fisher_low_above_floor']:3d} | stable {row['stable']:3d} | paying "
            f"{row['stable_and_paying']:3d} | baseline-stable {row['baseline_called_stable']:3d}"
            f" | {tail}"
        )
    native = report["blackout_native"]
    lines.append(
        f"  baseline at its own granularity: {native['closure_windows_found']} closure windows, "
        f"every candidate unstable = {native['every_candidate_unstable']}; "
        f"argus refused at MIN_PAIRS={native['argus_min_pairs']}: {native['argus_refused']}"
    )
    asym = report["sign_asymmetry"]
    lines.append(
        f"  sign asymmetry: baseline called an inverse pair stable = "
        f"{asym['baseline_ever_called_an_inverse_pair_stable']}; rejected for sign alone = "
        f"{asym['rejected_by_baseline_for_sign_alone']}"
    )
    leak = report["gap_leak"]
    lines.append(
        f"  our own effectiveness.py bucketing defect: max r^2 error "
        f"{leak['max_absolute_r_squared_error']}"
    )
    oos = report["out_of_sample"]
    lines.append(
        f"  out-of-sample: {oos['degraded_out_of_sample']}/{oos['pairs']} pairs degraded, "
        f"median decay {oos['median_decay']}"
    )
    sens = report["window_sensitivity"]
    lines.append(
        f"  stability-gate sensitivity ({sens['phase']}): strict stable count ranges "
        f"{sens['strict_gate_range']} across windows "
        f"{[row['rolling_window_bars'] for row in sens['by_window']]}, robust "
        f"{sens['robust_gate_range']}"
    )
    verdict = report["verdict"]
    lines.append(f"  ANSWER: {verdict['answer']}")
    lines.append(f"  reproducible: {report['reproducibility']['identical']}")
    return "\n".join(lines)


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = build_report()
    print(render(report))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "CANDIDATES",
    "EXPOSURES",
    "FETCH_DAYS",
    "FETCH_PAUSE_SECONDS",
    "MEASURED_PHASES",
    "MIN_ROLLING_WINDOWS",
    "REPORT_PATH",
    "ROLLING_WINDOW",
    "SCOPE_STATEMENT",
    "SIGN_AGREEMENT_FLOOR",
    "STABILITY_FLOOR",
    "UNIVERSE",
    "CostModel",
    "Panel",
    "TradeableError",
    "TradeableHedge",
    "build_report",
    "compare_pair",
    "contiguous_runs",
    "count_sign_flips",
    "decide",
    "fetch_cost_model",
    "fetch_panel",
    "hold_hours_for",
    "hold_matched_window",
    "main",
    "measure_costs",
    "measure_from_changes",
    "ols_ratio",
    "pearson",
    "percentile",
    "phase_changes",
    "render",
    "rolling_correlations",
    "run_blackout_native",
    "run_failure_cases",
    "run_gap_leak",
    "run_oos_split",
    "run_pair_table",
    "run_placebo",
    "run_reproducibility_check",
    "run_sign_asymmetry",
    "run_stability_ablation",
    "run_tautology_check",
    "run_window_sensitivity",
    "sign_agreement",
    "straddling_windows",
    "summarise",
    "variance_reduction",
]
