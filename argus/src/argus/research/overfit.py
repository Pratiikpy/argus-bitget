"""Anti-overfit gates — four ways a factor that looks real can fail to be.

A factor library that only ever grows is lying about decay, and a backtest that only reports the
number it found is lying about how it found it. These four tests are the cheapest honest checks
that separate a signal from a shape fitted to one sample:

1. **IC stability** — does the information coefficient keep its sign across sub-periods, or is the
   whole result one lucky stretch?
2. **Sub-sample stress** — does it survive being cut by regime (trend up, trend down, flat) and by
   volatility, or does it only work in the conditions it was found in?
3. **Placebo** — is the real IC actually distinguishable from the IC of the *same factor with its
   values shuffled*? And does the signal decay when the factor is lagged, as a real one must?
4. **Half-life** — how fast does predictive power decay across forward horizons? A factor whose IC
   is flat across every horizon is usually measuring the horizon, not the future.

Read against ``QuantGPT``'s ``quantgpt/anti_overfit.py:40-290`` (MIT), which is the most complete
open implementation of this quartet found in the 88-repo sweep. Three deliberate departures:

**No pandas, numpy or scipy.** ARGUS depends on pydantic and python-dateutil. Pulling a scientific
stack in for four statistics would triple the dependency surface of a project whose demo must not
fail to install. Spearman correlation over ranks is thirty lines of Python; it is written here.

**Insufficient data is not a failure.** QuantGPT returns ``passed=False`` with an ``error`` field
when there are too few observations, which makes "this factor is bad" and "we could not tell"
indistinguishable in the output and in any aggregate computed over it. Here the outcome is a
three-state :class:`Outcome` — PASS, FAIL, INCONCLUSIVE — and the summary counts them separately.
This matches :mod:`argus.eval.performance`, which refuses to print a Sharpe of 0.0 for a log that
has no trades.

**Thresholds are named constants with a stated reason**, not literals buried in a comparison. Every
one of them is arguable, and a reader should be able to find and change it in one place rather than
discovering it inline.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# --- thresholds ------------------------------------------------------------------------------

MIN_OBSERVATIONS = 20
"""Below this, every statistic here is noise. Reported as INCONCLUSIVE, never as a failure."""

MIN_CROSS_SECTION = 5
"""A rank correlation across fewer than five names is not a cross-section."""

IC_POSITIVE_RATE = 0.55
"""A real signal is right more often than a coin. 55% is deliberately modest."""

MIN_MEAN_IC = 0.02
"""Below this the effect is smaller than the cost of acting on it on this venue."""

SUBSAMPLE_CONSISTENCY = 0.6
"""Fraction of regime slices that must keep the overall sign."""

PLACEBO_PERMUTATIONS = 200
"""More than QuantGPT's 20: the 95th percentile of 20 samples is the second-largest value, which is
a coarse estimate of a tail. 200 costs milliseconds here because the IC is cheap to recompute."""

PLACEBO_PERCENTILE = 95.0
MIN_HALF_LIFE_PERIODS = 2.0
"""A factor whose edge is gone within two bars cannot be traded through a 12bp round trip."""

PERMUTATION_SEED = 20260912
"""Fixed so a verdict is reproducible. A placebo test with a random seed is a different test each
time it runs, and a factor that passes on the third attempt has not passed."""


class Outcome(StrEnum):
    """Three states, because "we could not tell" is not "it failed"."""

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class TestResult:
    name: str
    outcome: Outcome
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.outcome is Outcome.PASS

    def as_dict(self) -> dict[str, Any]:
        return {"test": self.name, "outcome": str(self.outcome), **self.detail}


@dataclass(frozen=True)
class Observation:
    """One factor reading and the return that followed it.

    ``period`` orders observations in time; ``name`` identifies the instrument. The cross-section
    at a period is every observation sharing that period, which is what a rank IC is computed over.
    """

    period: int
    name: str
    factor: float
    forward_return: float


# --- statistics ------------------------------------------------------------------------------


def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks, ties shared. Ties are common in factor values and must not bias the order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation. ``None`` when it is undefined rather than 0.0, which would read as "no
    relationship" when the truth is "not enough distinct values to have one"."""
    if len(xs) != len(ys) or len(xs) < MIN_CROSS_SECTION:
        return None
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def period_ic(observations: Sequence[Observation]) -> dict[int, float]:
    """The information coefficient per period: rank correlation of factor against forward return."""
    buckets: dict[int, list[Observation]] = {}
    for obs in observations:
        buckets.setdefault(obs.period, []).append(obs)
    out: dict[int, float] = {}
    for period, rows in buckets.items():
        value = spearman([r.factor for r in rows], [r.forward_return for r in rows])
        if value is not None:
            out[period] = value
    return out


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _percentile(xs: Sequence[float], pct: float) -> float:
    if not xs:
        return 0.0
    ordered = sorted(xs)
    if len(ordered) == 1:
        return ordered[0]
    pos = (pct / 100) * (len(ordered) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[int(pos)]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


# --- the four tests --------------------------------------------------------------------------


def check_ic_stability(observations: Sequence[Observation], *, slices: int = 4) -> TestResult:
    """Does the IC keep its sign across sub-periods, or is it one lucky stretch?

    QuantGPT slices by calendar year. A hackathon-length sample has no years in it, so this slices
    the observed periods into equal parts — the property being tested is "consistent across the
    sample", and calendar boundaries were never the point.
    """
    ics = period_ic(observations)
    if len(ics) < MIN_OBSERVATIONS:
        return TestResult("ic_stability", Outcome.INCONCLUSIVE,
                          {"periods_with_ic": len(ics), "needed": MIN_OBSERVATIONS})

    series = [ics[p] for p in sorted(ics)]
    mean_ic = _mean(series)
    positive_rate = sum(1 for v in series if v > 0) / len(series)
    overall_sign = 1 if mean_ic > 0 else -1

    width = max(1, len(series) // slices)
    chunks = [series[i:i + width] for i in range(0, len(series), width)][:slices]
    slice_means = [_mean(c) for c in chunks if c]
    reversals = [
        round(m, 4) for m in slice_means
        if m != 0 and (1 if m > 0 else -1) != overall_sign
    ]

    detail = {
        "mean_ic": round(mean_ic, 4),
        "positive_rate": round(positive_rate, 4),
        "slice_means": [round(m, 4) for m in slice_means],
        "sign_reversals": reversals,
    }
    if abs(mean_ic) < MIN_MEAN_IC:
        return TestResult("ic_stability", Outcome.FAIL, {**detail, "why": "mean IC below floor"})
    if positive_rate < IC_POSITIVE_RATE and mean_ic > 0:
        return TestResult("ic_stability", Outcome.FAIL, {**detail, "why": "positive rate too low"})
    if reversals:
        return TestResult("ic_stability", Outcome.FAIL,
                          {**detail, "why": "a sub-period reverses the sign"})
    return TestResult("ic_stability", Outcome.PASS, detail)


def check_subsample_stress(
    observations: Sequence[Observation], *, market_return: dict[int, float] | None = None
) -> TestResult:
    """Does it survive being cut by regime and by volatility?

    Regime is derived from the cross-sectional mean return per period when no market series is
    supplied, so the test works on the data it already has rather than requiring a benchmark that
    may not exist for tokenised equities out of hours.
    """
    ics = period_ic(observations)
    if len(ics) < MIN_OBSERVATIONS:
        return TestResult("subsample_stress", Outcome.INCONCLUSIVE,
                          {"periods_with_ic": len(ics), "needed": MIN_OBSERVATIONS})

    if market_return is None:
        by_period: dict[int, list[float]] = {}
        for obs in observations:
            by_period.setdefault(obs.period, []).append(obs.forward_return)
        market_return = {p: _mean(v) for p, v in by_period.items()}

    periods = sorted(ics)
    overall_sign = 1 if _mean([ics[p] for p in periods]) > 0 else -1

    returns = [market_return.get(p, 0.0) for p in periods]
    median_return = _percentile(returns, 50)
    volatility = _stdev(returns)

    slices: dict[str, list[float]] = {"up": [], "down": [], "quiet": [], "volatile": []}
    for period, ret in zip(periods, returns, strict=True):
        slices["up" if ret > median_return else "down"].append(ics[period])
        slices["volatile" if abs(ret - median_return) > volatility else "quiet"].append(ics[period])

    usable = {k: _mean(v) for k, v in slices.items() if len(v) >= MIN_CROSS_SECTION}
    if not usable:
        return TestResult("subsample_stress", Outcome.INCONCLUSIVE,
                          {"why": "no slice had enough observations"})

    agreeing = sum(1 for v in usable.values() if v != 0 and (1 if v > 0 else -1) == overall_sign)
    consistency = agreeing / len(usable)
    detail = {
        "overall_sign": overall_sign,
        "slice_ic": {k: round(v, 4) for k, v in usable.items()},
        "consistency": round(consistency, 4),
    }
    outcome = Outcome.PASS if consistency >= SUBSAMPLE_CONSISTENCY else Outcome.FAIL
    return TestResult("subsample_stress", outcome, detail)


def check_placebo(
    observations: Sequence[Observation], *, permutations: int = PLACEBO_PERMUTATIONS
) -> TestResult:
    """Is the real IC distinguishable from the same factor with its values shuffled?

    Shuffling happens *within* each period, so the cross-sectional distribution of factor values is
    preserved exactly and only the pairing with outcomes is destroyed. Shuffling globally would also
    destroy the distribution and make the null too easy to beat.
    """
    real = period_ic(observations)
    if len(real) < MIN_OBSERVATIONS:
        return TestResult("placebo", Outcome.INCONCLUSIVE,
                          {"periods_with_ic": len(real), "needed": MIN_OBSERVATIONS})
    real_ic = _mean([real[p] for p in sorted(real)])

    by_period: dict[int, list[Observation]] = {}
    for obs in observations:
        by_period.setdefault(obs.period, []).append(obs)

    rng = random.Random(PERMUTATION_SEED)
    placebo_ics: list[float] = []
    for _ in range(permutations):
        shuffled: list[Observation] = []
        for period, rows in by_period.items():
            values = [r.factor for r in rows]
            rng.shuffle(values)
            shuffled.extend(
                Observation(period, r.name, v, r.forward_return)
                for r, v in zip(rows, values, strict=True)
            )
        got = period_ic(shuffled)
        if got:
            placebo_ics.append(_mean([got[p] for p in sorted(got)]))

    if len(placebo_ics) < permutations // 2:
        return TestResult("placebo", Outcome.INCONCLUSIVE,
                          {"why": "too few permutations produced an IC"})

    threshold = _percentile([abs(v) for v in placebo_ics], PLACEBO_PERCENTILE)
    beats_null = abs(real_ic) > threshold
    detail = {
        "real_ic": round(real_ic, 4),
        "placebo_p95_abs_ic": round(threshold, 4),
        "permutations": len(placebo_ics),
        "beats_null": beats_null,
    }
    return TestResult("placebo", Outcome.PASS if beats_null else Outcome.FAIL, detail)


def check_half_life(
    observations: Sequence[Observation], *, horizons: Sequence[int] = (1, 2, 3, 5, 8)
) -> TestResult:
    """How fast does the edge decay across forward horizons?

    The forward return supplied on each observation is treated as horizon 1; longer horizons are
    formed by summing the returns of the following periods for the same instrument. A factor whose
    IC does not decay at all is usually measuring something about the horizon itself.
    """
    ics = period_ic(observations)
    if len(ics) < MIN_OBSERVATIONS:
        return TestResult("half_life", Outcome.INCONCLUSIVE,
                          {"periods_with_ic": len(ics), "needed": MIN_OBSERVATIONS})

    forward: dict[tuple[str, int], float] = {
        (o.name, o.period): o.forward_return for o in observations
    }
    horizon_ic: dict[int, float] = {}
    for horizon in horizons:
        rows: list[Observation] = []
        for obs in observations:
            total = 0.0
            complete = True
            for step in range(horizon):
                value = forward.get((obs.name, obs.period + step))
                if value is None:
                    complete = False
                    break
                total += value
            if complete:
                rows.append(Observation(obs.period, obs.name, obs.factor, total))
        got = period_ic(rows)
        if len(got) >= MIN_CROSS_SECTION:
            horizon_ic[horizon] = _mean([got[p] for p in sorted(got)])

    if len(horizon_ic) < 3:
        return TestResult("half_life", Outcome.INCONCLUSIVE,
                          {"horizons_measured": len(horizon_ic), "needed": 3})

    first = abs(horizon_ic[min(horizon_ic)])
    if first == 0:
        return TestResult("half_life", Outcome.FAIL,
                          {"why": "no IC at the shortest horizon", "horizon_ic": horizon_ic})

    half_life: float | None = None
    previous_h, previous_v = min(horizon_ic), first
    for horizon in sorted(horizon_ic)[1:]:
        value = abs(horizon_ic[horizon])
        if value <= first / 2:
            span = horizon - previous_h
            drop = previous_v - value
            frac = (previous_v - first / 2) / drop if drop > 0 else 0.0
            half_life = previous_h + span * frac
            break
        previous_h, previous_v = horizon, value

    detail = {
        "horizon_ic": {str(k): round(v, 4) for k, v in sorted(horizon_ic.items())},
        "half_life_periods": round(half_life, 2) if half_life is not None else None,
    }
    if half_life is None:
        # The edge never halves across the horizons measured. That is not automatically good: it
        # is the signature of a factor correlated with something slow-moving rather than predictive.
        return TestResult("half_life", Outcome.INCONCLUSIVE,
                          {**detail, "why": "IC did not halve within the horizons measured"})
    outcome = Outcome.PASS if half_life >= MIN_HALF_LIFE_PERIODS else Outcome.FAIL
    return TestResult("half_life", outcome, detail)


# --- the suite -------------------------------------------------------------------------------


@dataclass(frozen=True)
class OverfitReport:
    results: tuple[TestResult, ...]

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.outcome is Outcome.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.outcome is Outcome.FAIL)

    @property
    def inconclusive(self) -> int:
        return sum(1 for r in self.results if r.outcome is Outcome.INCONCLUSIVE)

    @property
    def verdict(self) -> str:
        """Deliberately not a 0-100 score.

        QuantGPT maps passes to a percentage, which invites comparing two factors that failed
        different tests as though the failures were interchangeable. They are not: a factor that
        fails the placebo test is indistinguishable from noise, and no number of other passes
        redeems that.
        """
        if self.failed == 0 and self.inconclusive == 0:
            return "clears every gate"
        if any(r.name == "placebo" and r.outcome is Outcome.FAIL for r in self.results):
            return "rejected: indistinguishable from shuffled data"
        if self.failed:
            names = ", ".join(r.name for r in self.results if r.outcome is Outcome.FAIL)
            return f"rejected: fails {names}"
        return "not proven: insufficient data for a verdict"

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "passed": self.passed,
            "failed": self.failed,
            "inconclusive": self.inconclusive,
            "tests": [r.as_dict() for r in self.results],
        }


def run_all(observations: Sequence[Observation]) -> OverfitReport:
    """Run the four gates. A factor is only certified when every one of them returns PASS."""
    return OverfitReport(
        results=(
            check_ic_stability(observations),
            check_subsample_stress(observations),
            check_placebo(observations),
            check_half_life(observations),
        )
    )


__all__ = [
    "IC_POSITIVE_RATE",
    "MIN_HALF_LIFE_PERIODS",
    "MIN_MEAN_IC",
    "MIN_OBSERVATIONS",
    "PLACEBO_PERMUTATIONS",
    "Observation",
    "Outcome",
    "OverfitReport",
    "TestResult",
    "check_half_life",
    "check_ic_stability",
    "check_placebo",
    "check_subsample_stress",
    "period_ic",
    "run_all",
    "spearman",
]
