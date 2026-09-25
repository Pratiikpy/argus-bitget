"""Event studies — estimating the effect, not just asserting the chain.

`agents/causality.py` builds a transmission chain from an event to a price move and
`paper/chains.py` grades each link against what actually happened. That is more than the temporal
correlation most systems stop at, and it is still not an *estimate of the effect*. The chain says
"this link held". It does not say how large the abnormal move was, whether it was distinguishable
from the stock's ordinary behaviour, or whether the same thing happens the next time the same kind
of event occurs.

This module is the missing half. It is the standard MacKinlay (1997) pipeline — estimation window,
market model, abnormal returns, cumulative abnormal returns — with the significance tests that
separate a real event study from a t-test on a handful of overlapping observations.

**The four tests, and why all four.** A single test would be a choice about which assumption to be
wrong about.

* **Patell (1976)** standardises each abnormal return by its own prediction-error-corrected
  standard error, so a volatile name does not dominate the average purely by being volatile. The
  correction has three parts and the third — the forecast-extrapolation penalty — is the one most
  implementations drop.
* **Boehmer, Musumeci and Poulsen (1991)** fixes what Patell cannot: events *cause* volatility.
  If the residual variance rises on the event day, a test that divides by the estimation-window
  variance rejects far too often. BMP re-scales by the cross-sectional dispersion of the
  standardised returns, so an event that widens the distribution is not mistaken for one that
  shifts it.
* **Corrado (1989)** assumes nothing about the distribution. Abnormal returns are fat-tailed and
  skewed, and with a small number of events a parametric test is driven by whichever one moved
  most. Ranks are not.
* **The generalised sign test** asks only whether more names moved up than usually do, which is
  the one statement that survives an outlier entirely.

**Event clustering is the failure this module is most careful about.** When several events share a
timestamp, their abnormal returns are correlated and every parametric test over-rejects — by a
factor of roughly ``sqrt(1 + (N-1) * r)`` in the correlation ``r``. On a universe of twelve
tokenised US equities driven by one underlying market, that correlation is not small. The
Kolari-Pynnonen (2010) deflation is therefore applied to every parametric statistic here and
reported beside the unadjusted one, so the size of the correction is visible rather than absorbed.

**What changes in a 7x24 market, stated rather than ignored.** The classical method assumes daily
bars and a market that closes, which gives a clean event date and non-overlapping returns. rTokens
trade continuously against an equity that does not, so:

1. The clock is hourly and the windows are counted in bars, not days. Nothing here divides by 252.
2. The market proxy is an **equal-weighted portfolio of the other rTokens**, excluding the event
   name so a symbol cannot benchmark against itself. With twelve names driven by one underlying
   index, this is a stronger control than a single proxy and a weaker one than a true factor model.
3. The estimation window is **purged** from the event window by an explicit gap, because hourly
   returns are serially correlated and an estimation window ending at the event is estimating
   partly on the event.

Read against `research/architecture/eventedge.md`, our teardown of the one event-study
implementation in the local corpus. It implements the market model, abnormal and cumulative
abnormal returns and a bootstrap interval correctly, and it has **only a plain t-test** — no
Patell, no BMP, no rank test, no clustering adjustment and no multiple-testing correction across
its thirty-six windows. Those five gaps are exactly what this module adds; the parts it gets right
were followed rather than reinvented.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from math import sqrt
from statistics import NormalDist
from typing import Any

MIN_ESTIMATION_BARS = 120
"""Fewest bars the market model may be fitted on.

Below this the residual standard error is itself so uncertain that standardising by it does more
harm than the standardisation removes. MacKinlay's daily convention is 250 bars; 120 hourly bars is
five days, which is the floor at which a beta on this venue is worth having at all.
"""

DEFAULT_ESTIMATION_BARS = 480
DEFAULT_GAP_BARS = 24
"""Bars purged between the end of the estimation window and the start of the event window.

Not cosmetic. The desk's own hold horizon is 24 hours, so an estimation window running right up to
the event would be fitted partly on returns the event window then scores.
"""

MIN_EVENTS = 5
"""Fewest events before any test statistic is reported.

Every test here is asymptotic in the number of events. With three events a z-score is a number with
no distribution behind it, and reporting one would be worse than reporting nothing because it
would look like evidence.
"""


class EventStudyError(ValueError):
    """Raised rather than returning a statistic that cannot be computed."""


@dataclass(frozen=True, slots=True)
class MarketModel:
    """``R_i = alpha + beta * R_m + e``, fitted on the estimation window alone."""

    alpha: float
    beta: float
    residual_sd: float
    observations: int
    market_mean: float
    market_variance_sum: float
    """``sum (R_m - mean)^2`` over the estimation window. Kept because Patell's third correction
    term needs it and recomputing it from a stored variance loses the exact denominator."""

    def expected(self, market_return: float) -> float:
        return self.alpha + self.beta * market_return

    def abnormal(self, actual: float, market_return: float) -> float:
        return actual - self.expected(market_return)

    def forecast_sd(self, market_return: float) -> float:
        """Patell's prediction-error-corrected standard error for one out-of-sample bar.

        ``S^2 * (1 + 1/M + (R_m - mean_m)^2 / sum(R_m - mean_m)^2)``. The three terms are the
        residual variance, the error in estimating the mean, and the penalty for forecasting at a
        market return far from the estimation window's centre. The third is the one that is usually
        dropped, and it matters precisely on the days an event study is about — the days the market
        itself moved.
        """
        if self.market_variance_sum <= 0:
            raise EventStudyError("the market series has no variance in the estimation window")
        inflation = (
            1.0
            + 1.0 / self.observations
            + (market_return - self.market_mean) ** 2 / self.market_variance_sum
        )
        return self.residual_sd * sqrt(inflation)


def fit_market_model(asset: Sequence[float], market: Sequence[float]) -> MarketModel:
    """Ordinary least squares of asset on market. Raises rather than returning a degenerate fit."""
    n = len(asset)
    if n != len(market):
        raise EventStudyError("the asset and market series must cover the same bars")
    if n < MIN_ESTIMATION_BARS:
        raise EventStudyError(
            f"{n} estimation bar(s) is below the {MIN_ESTIMATION_BARS} a market model needs; "
            f"a beta from fewer is a number without a standard error worth quoting"
        )
    mean_a = sum(asset) / n
    mean_m = sum(market) / n
    variance_sum = sum((m - mean_m) ** 2 for m in market)
    if variance_sum <= 0:
        raise EventStudyError("the market series is constant; beta is undefined, not zero")
    covariance = sum((a - mean_a) * (m - mean_m) for a, m in zip(asset, market, strict=True))
    beta = covariance / variance_sum
    alpha = mean_a - beta * mean_m
    residuals = [a - alpha - beta * m for a, m in zip(asset, market, strict=True)]
    # Two parameters estimated, so n - 2 degrees of freedom. Dividing by n understates the
    # residual standard error and every standardised statistic built on it comes out too large.
    residual_variance = sum(r * r for r in residuals) / (n - 2)
    if residual_variance <= 0:
        raise EventStudyError("the market model fits exactly; there is no residual to standardise")
    return MarketModel(
        alpha=alpha,
        beta=beta,
        residual_sd=sqrt(residual_variance),
        observations=n,
        market_mean=mean_m,
        market_variance_sum=variance_sum,
    )


@dataclass(frozen=True, slots=True)
class EventWindow:
    """One event: its abnormal returns, and the model they were measured against."""

    symbol: str
    at: datetime
    abnormal: tuple[float, ...]
    standardised: tuple[float, ...]
    """Each abnormal return over its own Patell forecast standard error."""

    model: MarketModel
    estimation_abnormal: tuple[float, ...]
    """The estimation-window residuals, kept for the rank test and the sign test's base rate."""

    @property
    def car(self) -> float:
        """Cumulative abnormal return over the event window."""
        return sum(self.abnormal)

    @property
    def scar(self) -> float:
        """Standardised CAR. Summing L standardised returns scales the error by sqrt(L)."""
        if not self.abnormal:
            return 0.0
        return sum(self.standardised) / sqrt(len(self.standardised))

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "at": self.at.isoformat(),
            "car": round(self.car, 6),
            "scar": round(self.scar, 4),
            "beta": round(self.model.beta, 4),
            "residual_sd": round(self.model.residual_sd, 6),
        }


def _normal_p(z: float) -> float:
    """Two-sided p-value for a standard normal statistic."""
    return 2.0 * (1.0 - NormalDist().cdf(abs(z)))


def patell_z(windows: Sequence[EventWindow]) -> float:
    """Patell (1976): the sum of standardised CARs over the square root of their summed variance.

    ``z = sum(SCAR_i) / sqrt(sum((M_i - 2) / (M_i - 4)))``. The variance term is not ``N``: a
    standardised residual computed with an *estimated* standard error is t-distributed, not normal,
    and its variance exceeds one by exactly that ratio. Using ``sqrt(N)`` instead — which is the
    common shortcut — overstates the statistic, most on the shortest estimation windows.
    """
    if len(windows) < MIN_EVENTS:
        raise EventStudyError(
            f"{len(windows)} event(s) is below the {MIN_EVENTS} an asymptotic test needs"
        )
    total = sum(w.scar for w in windows)
    variance = 0.0
    for w in windows:
        m = w.model.observations
        if m <= 4:
            raise EventStudyError("an estimation window of four bars or fewer has no variance term")
        variance += (m - 2) / (m - 4)
    return total / sqrt(variance)


def bmp_t(windows: Sequence[EventWindow]) -> float:
    """Boehmer, Musumeci and Poulsen (1991): a cross-sectional t on the standardised returns.

    ``t = sqrt(N) * mean(SCAR) / sd(SCAR)``. The difference from Patell is the denominator, and it
    is the whole point: Patell divides by the *estimation-window* variance, which assumes the event
    did not change the variance. Events change the variance — that is most of what an event is — so
    on any real sample Patell over-rejects and this does not.
    """
    n = len(windows)
    if n < MIN_EVENTS:
        raise EventStudyError(
            f"{n} event(s) is below the {MIN_EVENTS} an asymptotic test needs"
        )
    scars = [w.scar for w in windows]
    mean = sum(scars) / n
    variance = sum((s - mean) ** 2 for s in scars) / (n - 1)
    if variance <= 0:
        raise EventStudyError(
            "every event produced an identical standardised return; the cross-sectional "
            "dispersion this test divides by is zero"
        )
    return sqrt(n) * mean / sqrt(variance)


def corrado_rank_z(windows: Sequence[EventWindow]) -> float:
    """Corrado (1989): rank each firm's event-window returns within its own pooled series.

    Each firm's estimation-window and event-window abnormal returns are pooled and ranked
    **within that firm**, then scaled to (0, 1) by dividing by one plus the pooled length. Under
    the null the expected scaled rank is 0.5, and the standard error comes from the time series of
    cross-sectional mean ranks across the whole pooled window, not from the event bars alone.

    The distinction matters and is easy to get wrong: ranking across *events at a point in time*
    rather than within each firm's own history produces a statistic that measures which name moved
    most, not whether any of them moved abnormally.
    """
    n = len(windows)
    if n < MIN_EVENTS:
        raise EventStudyError(
            f"{n} event(s) is below the {MIN_EVENTS} an asymptotic test needs"
        )
    lengths = {len(w.abnormal) for w in windows}
    if len(lengths) != 1:
        raise EventStudyError("the rank test needs every event window to be the same length")
    event_len = lengths.pop()
    if event_len < 1:
        raise EventStudyError("an empty event window has nothing to rank")

    pooled_ranks: list[list[float]] = []
    for w in windows:
        series = [*w.estimation_abnormal, *w.abnormal]
        order = sorted(range(len(series)), key=lambda i: series[i])
        ranks = [0.0] * len(series)
        for position, index in enumerate(order, start=1):
            ranks[index] = position / (1.0 + len(series))
        pooled_ranks.append(ranks)

    pooled_len = min(len(r) for r in pooled_ranks)
    # Cross-sectional mean rank at each pooled bar, aligned from the end so the event bars line up
    # across firms whose estimation windows differ in length.
    means: list[float] = []
    for offset in range(pooled_len):
        column = [r[len(r) - pooled_len + offset] for r in pooled_ranks]
        means.append(sum(column) / len(column))
    variance = sum((m - 0.5) ** 2 for m in means) / len(means)
    if variance <= 0:
        raise EventStudyError("the pooled rank series has no variance; the test is undefined")
    event_mean = sum(means[-event_len:]) / event_len
    return (event_mean - 0.5) / sqrt(variance / event_len)


def generalised_sign_z(windows: Sequence[EventWindow]) -> float:
    """Cowan (1992): more positives than this sample usually produces?

    The null proportion is each firm's own estimation-window rate of positive abnormal returns,
    averaged — not 0.5. Using 0.5 assumes the market model leaves a symmetric residual, and on a
    trending sample it does not, which turns a drift into a finding.
    """
    n = len(windows)
    if n < MIN_EVENTS:
        raise EventStudyError(
            f"{n} event(s) is below the {MIN_EVENTS} an asymptotic test needs"
        )
    rates = []
    for w in windows:
        if not w.estimation_abnormal:
            raise EventStudyError("the sign test needs an estimation window to form its null rate")
        rates.append(sum(1 for r in w.estimation_abnormal if r > 0) / len(w.estimation_abnormal))
    p = sum(rates) / n
    if p <= 0.0 or p >= 1.0:
        raise EventStudyError(
            "the estimation-window residuals are all one sign; there is no null proportion"
        )
    positives = sum(1 for w in windows if w.car > 0)
    return (positives - n * p) / sqrt(n * p * (1.0 - p))


def average_cross_correlation(windows: Sequence[EventWindow]) -> float:
    """Mean pairwise correlation of estimation-window abnormal returns across events.

    This is the number the clustering adjustment needs, and computing it from the *estimation*
    residuals rather than the event window is deliberate: the event window is where the effect is
    supposed to be, so correlation measured there confounds the thing being corrected for with the
    thing being tested.

    **Computed in O(N·L), not O(N²·L).** Each series is centred and scaled to unit length, so a
    pairwise correlation is a dot product ``z_i · z_j``, and the sum over every pair is
    ``(‖Σ z_i‖² − m) / 2`` for the ``m`` series that have any variance. That is the same number
    the pairwise loop produced (a constant series has no correlation with anything, and was
    skipped there exactly as it is excluded from ``m`` here); what changed is the cost. The loop
    took about 1.7s for 108 events on 480-bar windows and grew with the square of the event count,
    which made a study of a few thousand real events — one event class over a quarter of 5-minute
    perception, `eval/eventdriven_rivals.py` — impractical. The equivalence is pinned against the
    original loop in `tests/test_eventdriven_rivals.py`.
    """
    if len(windows) < 2:
        return 0.0
    length = min(len(w.estimation_abnormal) for w in windows)
    if length < 2:
        return 0.0
    total = [0.0] * length
    usable = 0
    for w in windows:
        tail = w.estimation_abnormal[-length:]
        mean = sum(tail) / length
        centred = [x - mean for x in tail]
        norm = sqrt(sum(x * x for x in centred))
        if norm <= 0:
            continue
        usable += 1
        for k, x in enumerate(centred):
            total[k] += x / norm
    if usable < 2:
        return 0.0
    pair_sum = (sum(x * x for x in total) - usable) / 2.0
    return pair_sum / (usable * (usable - 1) / 2.0)


def kolari_pynnonen(statistic: float, *, events: int, correlation: float) -> float:
    """Deflate a parametric statistic for cross-sectional correlation between events.

    ``adj = sqrt((1 - r) / (1 + (N - 1) * r))``. At ``r = 0`` it is 1 and nothing changes; at
    twelve tokenised equities sharing one underlying market it is not close to 1, and a test that
    skips it rejects far more often than its stated level.

    A negative average correlation is clipped to zero rather than used to *inflate* the statistic.
    The adjustment exists to remove over-rejection from shared movement; letting a negative sample
    correlation make a result more significant would be using a correction as an enhancement.
    """
    if events < 2:
        return statistic
    r = max(0.0, correlation)
    denominator = 1.0 + (events - 1) * r
    if denominator <= 0:
        return statistic
    return statistic * sqrt((1.0 - r) / denominator)


@dataclass(frozen=True, slots=True)
class EventStudyResult:
    """Every test, adjusted and unadjusted, plus what the answer is allowed to mean."""

    label: str
    windows: tuple[EventWindow, ...]
    event_bars: int
    correlation: float

    @property
    def events(self) -> int:
        return len(self.windows)

    @property
    def average_car(self) -> float:
        return sum(w.car for w in self.windows) / self.events if self.windows else 0.0

    @property
    def average_car_bps(self) -> float:
        return self.average_car * 10_000

    @property
    def statistics(self) -> dict[str, dict[str, float]]:
        """Each test, its raw value, its clustering-adjusted value, and both p-values."""
        out: dict[str, dict[str, float]] = {}
        for name, value, parametric in (
            ("patell", patell_z(self.windows), True),
            ("bmp", bmp_t(self.windows), True),
            ("corrado_rank", corrado_rank_z(self.windows), False),
            ("generalised_sign", generalised_sign_z(self.windows), False),
        ):
            adjusted = (
                kolari_pynnonen(value, events=self.events, correlation=self.correlation)
                if parametric else value
            )
            out[name] = {
                "statistic": round(value, 4),
                "adjusted": round(adjusted, 4),
                "p_value": round(_normal_p(value), 6),
                "adjusted_p_value": round(_normal_p(adjusted), 6),
                "clustering_adjusted": parametric,
            }
        return out

    @property
    def verdict(self) -> str:
        """What survives, stated by the test that survived it — and by the ones that did not."""
        stats = self.statistics
        survivors = [
            name for name, row in stats.items()
            if row["adjusted_p_value"] <= 0.05
        ]
        direction = "positive" if self.average_car > 0 else "negative"
        if not survivors:
            return (
                f"NO EFFECT ESTABLISHED across {self.events} event(s). The average cumulative "
                f"abnormal return is {self.average_car_bps:+.1f}bps and no test rejects at 5% "
                f"after the clustering adjustment. An average that looks large and a statistic "
                f"that does not reject is the ordinary case with few, correlated events"
            )
        if len(survivors) == len(stats):
            return (
                f"EFFECT ESTABLISHED: a {direction} average abnormal return of "
                f"{self.average_car_bps:+.1f}bps over {self.events} event(s), significant at 5% "
                f"under all four tests including the non-parametric ones"
            )
        failed = [name for name in stats if name not in survivors]
        return (
            f"PARTIAL: a {direction} average abnormal return of {self.average_car_bps:+.1f}bps "
            f"rejects under {', '.join(survivors)} but not under {', '.join(failed)}. Where a "
            f"parametric test rejects and the rank test does not, the result is usually being "
            f"carried by one or two events"
        )

    @property
    def clustering_note(self) -> str:
        factor = kolari_pynnonen(1.0, events=self.events, correlation=self.correlation)
        return (
            f"Average pairwise correlation of estimation-window abnormal returns across events is "
            f"{self.correlation:.3f}, which deflates every parametric statistic by a factor of "
            f"{factor:.3f}. A study of tokenised equities that skips this is testing as though "
            f"twelve names driven by one market were twelve independent experiments."
        )

    def render(self) -> str:
        stats = self.statistics
        lines = [
            f"EVENT STUDY — {self.label}: {self.events} event(s), {self.event_bars}-bar window",
            "",
            f"  average CAR {self.average_car_bps:+.1f}bps",
            "",
            f"{'test':>20}{'statistic':>12}{'adjusted':>11}{'p':>10}{'adj p':>10}",
        ]
        for name, row in stats.items():
            lines.append(
                f"{name:>20}{row['statistic']:>12.3f}{row['adjusted']:>11.3f}"
                f"{row['p_value']:>10.4f}{row['adjusted_p_value']:>10.4f}"
            )
        lines += ["", f"  {self.clustering_note}", "", f"  {self.verdict}"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "events": self.events,
            "event_bars": self.event_bars,
            "average_car": round(self.average_car, 8),
            "average_car_bps": round(self.average_car_bps, 3),
            "cross_correlation": round(self.correlation, 5),
            "statistics": self.statistics,
            "clustering_note": self.clustering_note,
            "verdict": self.verdict,
            "windows": [w.as_dict() for w in self.windows],
        }


def returns_from(closes: Sequence[float]) -> list[float]:
    """Simple returns. One shorter than the price series, which every index below accounts for."""
    if len(closes) < 2:
        return []
    out = []
    for previous, current in pairwise(closes):
        out.append(0.0 if previous == 0 else current / previous - 1.0)
    return out


def market_proxy(
    panel: dict[str, Sequence[float]], *, exclude: str
) -> list[float]:
    """An equal-weighted return series over every symbol except the event name.

    Excluding the event symbol is not a nicety. With twelve names, leaving it in gives it about an
    eighth of its own benchmark, which pulls the fitted beta toward one and shrinks the very
    abnormal return the study is trying to measure.
    """
    others = [list(v) for k, v in panel.items() if k != exclude]
    if not others:
        raise EventStudyError(
            f"no symbols left to benchmark {exclude} against once it is excluded from its own "
            f"market proxy"
        )
    length = min(len(v) for v in others)
    if length == 0:
        raise EventStudyError("the market proxy has no overlapping bars")
    return [sum(v[len(v) - length + t] for v in others) / len(others) for t in range(length)]


def build_window(
    symbol: str,
    at: datetime,
    timestamps: Sequence[datetime],
    asset_returns: Sequence[float],
    market_returns: Sequence[float],
    *,
    event_bars: int,
    estimation_bars: int = DEFAULT_ESTIMATION_BARS,
    gap_bars: int = DEFAULT_GAP_BARS,
) -> EventWindow | None:
    """Fit the model before the event and measure the window after it.

    ``timestamps`` are the timestamps of the *return* series, so ``timestamps[t]`` is the bar at
    which ``asset_returns[t]`` was realised. Returns None when the history around the event is too
    short, rather than fitting on whatever is available — a market model fitted on ninety bars for
    one event and five hundred for another makes the two standardised returns incomparable, which
    is precisely what standardising was for.
    """
    if not (len(timestamps) == len(asset_returns) == len(market_returns)):
        raise EventStudyError("timestamps, asset returns and market returns must align")
    if event_bars < 1:
        raise EventStudyError("an event window of zero bars measures nothing")
    if gap_bars < 0:
        raise EventStudyError("a negative gap would overlap the estimation and event windows")

    index = bisect_right(list(timestamps), at)
    if index >= len(timestamps):
        return None
    estimation_end = index - gap_bars
    estimation_start = estimation_end - estimation_bars
    if estimation_start < 0 or estimation_end - estimation_start < MIN_ESTIMATION_BARS:
        return None
    if index + event_bars > len(timestamps):
        return None

    model = fit_market_model(
        asset_returns[estimation_start:estimation_end],
        market_returns[estimation_start:estimation_end],
    )
    abnormal = tuple(
        model.abnormal(asset_returns[t], market_returns[t])
        for t in range(index, index + event_bars)
    )
    standardised = tuple(
        a / model.forecast_sd(market_returns[index + offset])
        for offset, a in enumerate(abnormal)
    )
    estimation_abnormal = tuple(
        model.abnormal(asset_returns[t], market_returns[t])
        for t in range(estimation_start, estimation_end)
    )
    return EventWindow(
        symbol=symbol,
        at=at,
        abnormal=abnormal,
        standardised=standardised,
        model=model,
        estimation_abnormal=estimation_abnormal,
    )


def study(
    label: str, windows: Sequence[EventWindow], *, event_bars: int
) -> EventStudyResult:
    """Assemble the result.

    Raises when there are too few events for any statistic to mean anything.
    """
    if len(windows) < MIN_EVENTS:
        raise EventStudyError(
            f"{len(windows)} event(s) is below the {MIN_EVENTS} this study needs. Reporting a "
            f"z-score from fewer would put a number where a refusal belongs"
        )
    return EventStudyResult(
        label=label,
        windows=tuple(windows),
        event_bars=event_bars,
        correlation=average_cross_correlation(windows),
    )


__all__ = [
    "DEFAULT_ESTIMATION_BARS",
    "DEFAULT_GAP_BARS",
    "MIN_ESTIMATION_BARS",
    "MIN_EVENTS",
    "EventStudyError",
    "EventStudyResult",
    "EventWindow",
    "MarketModel",
    "average_cross_correlation",
    "bmp_t",
    "build_window",
    "corrado_rank_z",
    "fit_market_model",
    "generalised_sign_z",
    "kolari_pynnonen",
    "market_proxy",
    "patell_z",
    "returns_from",
    "study",
]
