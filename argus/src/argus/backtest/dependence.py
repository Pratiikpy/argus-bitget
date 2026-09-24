"""Inference that survives autocorrelation — the correction every Sharpe in this repository needs.

Every interval this project has published treats return observations as independent. They are not.
Hourly rToken returns carry serial correlation, and the consequence is one-directional: with
positive autocorrelation the independent-and-identically-distributed standard error is **too
small**, so the confidence interval is too narrow and the significance is overstated. That is the
error that flatters, which is why it is worth the module.

Three instruments, each answering a different question, each read from a working implementation
before being written here rather than derived from memory:

**Lo (2002) HAC standard error** — the closed form. ``SE = sqrt((1 + SR^2/2) * eta(q) / T)``, where
``eta(q) = 1 + 2 * sum_k (1 - k/(q+1)) * rho_k`` applies the Bartlett kernel of Newey and West
(1987) to the sample autocorrelations. At ``q = 0`` it collapses to the Gaussian formula the rest
of the field uses, so the ratio ``eta`` is itself the report: it says by exactly what factor the
naive interval was too narrow.

**The stationary bootstrap of Politis and Romano (1994)** — the non-parametric answer, for when the
asymptotic normality of a Sharpe ratio is not something to lean on. Blocks of geometrically
distributed length, wrapped circularly. The geometric length is the whole point: a fixed block
length makes the resampled series non-stationary at the seams, and a random one whose distribution
is memoryless does not.

**White's Reality Check (2000)** — the question the deflated Sharpe and PBO both decline to answer:
not "is this Sharpe real?" nor "is my selection stable?", but "is the best of my N strategies
actually better than *this specific benchmark*, given that I tried N?" It is the test that puts
buy-and-hold on the other side of the table, which for a Track 1 entry is the comparison that
matters.

Read before writing, per the standing rule, from
``best-of-the-best/repos/agent-backtest-lab/abl/multipletest/hac.py:64-160`` and ``spa.py:51-157``.
Two deliberate departures from that reference are recorded where they occur:

1. It floors ``eta`` at ``1e-12`` before taking a square root. The Bartlett kernel is chosen
   precisely because it makes the Newey-West estimator positive semi-definite, which in the scalar
   case means ``eta`` cannot be negative — so a negative value is a defect in the computation, not
   a property of the data, and flooring it silently converts a bug into a plausible number. This
   implementation raises.
2. It is written on numpy. This package has no numeric dependency and is not acquiring one for a
   kernel weight and a geometric draw, both of which are three lines.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor, log, log10, sqrt
from statistics import NormalDist
from typing import Any

from argus.backtest.metrics import MetricError, _is_negligible

MIN_OBSERVATIONS = 30
"""Fewest observations before a dependence correction means anything.

Lower than the bootstrap would like and deliberately so: the functions here raise on their own
specific degeneracies, and a single blanket threshold that silently allows a twelve-lag correction
on forty observations would be worse than no threshold at all.
"""

DEFAULT_RESAMPLES = 2_000
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 20260913
"""Seeded by default so a published interval is reproducible from the number alone.

An unseeded bootstrap produces a different interval on every run, which means a figure in a
document can never be checked against the code that made it.
"""


def auto_lag_truncation(n: int) -> int:
    """Newey and West (1994) automatic bandwidth: ``floor(4 * (T/100)^(2/9))``.

    A rule, not a theorem — it is the conventional automatic choice and it is reported alongside
    every result here so a reader can vary it. At 4,318 hourly bars it gives 9; at 252 daily
    bars, 4.
    """
    if n < 10:
        return 0
    # int() truncates toward zero, which is the floor for the positive value this always is.
    return int(4.0 * (n / 100.0) ** (2.0 / 9.0))


def suggested_block_length(n: int) -> float:
    """``sqrt(T)`` — the practitioner fallback, and folklore rather than theory.

    Kept as the fallback for a series too short or too degenerate for the real rule, and named a
    suggestion so nothing downstream reads it as optimal. Prefer :func:`optimal_block_length`,
    which is data-driven and was verified against a working implementation.
    """
    return sqrt(max(2, n))


def optimal_block_length(returns: Sequence[float]) -> tuple[float, float]:
    """Politis and White (2004), with the Patton, Politis and White (2009) correction.

    Returns ``(stationary, circular)`` — the two bootstraps want different block lengths because
    their variance expansions differ by a constant, and using one for the other is a silent
    mis-specification rather than a visible one.

    **This was shipped as NOT IMPLEMENTED earlier the same day** because the rule could not be
    verified from any source then in hand, and approximating it would have been a guess wearing a
    citation. It is implemented now because the `arch` package turned out to be installed on this
    machine and the algorithm could be read: transcribed from
    ``arch/bootstrap/base.py:73-122`` (``_single_optimal_block``), line by line.

    Two details a summary of that source got wrong, and which reading it caught:

    * The truncation lag is **doubled** — ``m = 2 * max(opt_m, 1)``, not ``opt_m`` itself. Using
      the undoubled value gives a systematically short block and therefore a systematically narrow
      interval, which is the direction that flatters.
    * ``acv`` holds an auto**covariance** divided by the full sample length, not an
      autocorrelation. The significance screen uses a separately normalised quantity.

    The procedure: find the first run of ``k_n`` consecutive insignificant autocorrelations
    (Andrews 1991) against the band ``2 * sqrt(log10(n)/n)``; build a flat-top lag window over the
    doubled truncation; then ``b = (2 g^2 / d)^(1/3) n^(1/3)``, capped at
    ``ceil(min(3 sqrt(n), n/3))``.

    Falls back to :func:`suggested_block_length` for both values, rather than raising, when the
    series carries no usable autocovariance structure — a white-noise series genuinely has no
    optimal block length, and refusing to bootstrap it at all would be worse than using blocks of
    one.
    """
    n = len(returns)
    if n < MIN_OBSERVATIONS:
        raise MetricError(
            f"{n} observation(s) is below the {MIN_OBSERVATIONS} this rule needs; the lag screen "
            f"alone inspects about sqrt(n) lags"
        )
    mean = sum(returns) / n
    eps = [r - mean for r in returns]
    # Fourth appearance of the same failure in this package. A constant series of 0.01 leaves
    # deviations around 1e-18 rather than 0, and every quantity below is a *ratio* of such
    # quantities, so the scale cancels and the rule returns a confident block length computed
    # entirely from floating-point dust. Caught by a test that expected the fallback.
    if _is_negligible(sqrt(sum(v * v for v in eps) / n), mean):
        fallback = suggested_block_length(n)
        return fallback, fallback
    b_max = ceil(min(3.0 * sqrt(n), n / 3.0))
    k_n = max(5, int(log10(n)))
    m_max = ceil(sqrt(n)) + k_n
    cv = 2.0 * sqrt(log10(n) / n)

    acv = [0.0] * (m_max + 1)
    abs_acorr = [0.0] * (m_max + 1)
    opt_m: int | None = None
    for i in range(m_max + 1):
        if i + 1 >= n:
            break
        v1 = sum(v * v for v in eps[i + 1:])
        v2 = sum(v * v for v in eps[: -(i + 1)])
        cross = sum(a * b for a, b in zip(eps[i:], eps[: n - i], strict=True))
        acv[i] = cross / n
        abs_acorr[i] = abs(cross) / sqrt(v1 * v2) if v1 > 0 and v2 > 0 else 0.0
        if i >= k_n and opt_m is None and all(abs_acorr[j] < cv for j in range(i - k_n, i)):
            opt_m = i - k_n

    m = min(2 * max(opt_m, 1) if opt_m is not None else m_max, m_max)

    g = 0.0
    lr_acv = acv[0]
    for k in range(1, m + 1):
        lam = 1.0 if k / m <= 0.5 else 2.0 * (1.0 - k / m)
        g += 2.0 * lam * k * acv[k]
        lr_acv += 2.0 * lam * acv[k]

    if g == 0.0 or lr_acv == 0.0:
        fallback = suggested_block_length(n)
        return fallback, fallback
    d_sb = 2.0 * lr_acv**2
    d_cb = 4.0 / 3.0 * lr_acv**2
    scale = n ** (1.0 / 3.0)
    b_sb = min(((2.0 * g**2) / d_sb) ** (1.0 / 3.0) * scale, float(b_max))
    b_cb = min(((2.0 * g**2) / d_cb) ** (1.0 / 3.0) * scale, float(b_max))
    return max(1.0, b_sb), max(1.0, b_cb)


def autocorrelations(returns: Sequence[float], lags: int) -> list[float]:
    """Sample autocorrelations rho_1 .. rho_lags. Raises on a series with no variance."""
    if lags <= 0:
        return []
    n = len(returns)
    if n < 2:
        raise MetricError("autocorrelation needs at least two observations")
    mean = sum(returns) / n
    centred = [r - mean for r in returns]
    denominator = sum(v * v for v in centred)
    if denominator <= 0 or _is_negligible(sqrt(denominator / n), mean):
        raise MetricError("a constant series has no autocorrelation structure to estimate")
    out: list[float] = []
    for k in range(1, lags + 1):
        if k >= n:
            out.append(0.0)
            continue
        out.append(sum(a * b for a, b in zip(centred[:-k], centred[k:], strict=True)) / denominator)
    return out


def newey_west_eta(rhos: Sequence[float]) -> float:
    """``1 + 2 * sum_k (1 - k/(q+1)) * rho_k`` under the Bartlett kernel.

    The factor by which the long-run variance exceeds the one-period variance. Above 1 the naive
    interval is too narrow; below 1 the series mean-reverts and the naive interval is conservative.
    """
    q = len(rhos)
    if q == 0:
        return 1.0
    return 1.0 + 2.0 * sum((1.0 - k / (q + 1.0)) * rho for k, rho in enumerate(rhos, start=1))


@dataclass(frozen=True, slots=True)
class HACSharpe:
    """A Sharpe ratio with an interval that admits the returns are not independent."""

    sharpe: float
    """Per-observation, not annualised. Annualising before the interval is computed would scale the
    point estimate and the error by the same factor and hide nothing, but it invites the reader to
    compare an hourly interval with a daily one."""

    observations: int
    lags: int
    eta: float
    se_iid: float
    se_hac: float
    low: float
    high: float
    rhos: tuple[float, ...]

    @property
    def widening(self) -> float:
        """How many times wider the honest interval is than the independent one."""
        return self.se_hac / self.se_iid if self.se_iid > 0 else 1.0

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    @property
    def verdict(self) -> str:
        naive_significant = abs(self.sharpe) > NormalDist().inv_cdf(0.975) * self.se_iid
        if self.excludes_zero:
            return (
                f"significant under the autocorrelation correction; the interval is "
                f"{self.widening:.2f}x wider than the independent one and still excludes zero"
            )
        if naive_significant:
            return (
                f"SIGNIFICANT ONLY IF THE RETURNS ARE ASSUMED INDEPENDENT — the correction widens "
                f"the interval by {self.widening:.2f}x and it then covers zero. This is the case "
                f"the correction exists to catch"
            )
        return "not significant either way; the correction is not what decides this one"

    def as_dict(self) -> dict[str, Any]:
        return {
            "sharpe_per_observation": round(self.sharpe, 6),
            "observations": self.observations,
            "lags": self.lags,
            "eta": round(self.eta, 5),
            "se_iid": round(self.se_iid, 6),
            "se_hac": round(self.se_hac, 6),
            "widening": round(self.widening, 4),
            "ci_low": round(self.low, 6),
            "ci_high": round(self.high, 6),
            "excludes_zero": self.excludes_zero,
            "autocorrelations": [round(r, 5) for r in self.rhos],
            "verdict": self.verdict,
        }


def hac_sharpe(
    returns: Sequence[float], *, lags: int | None = None, alpha: float = DEFAULT_ALPHA
) -> HACSharpe:
    """Lo (2002): a Sharpe interval corrected for serial correlation.

    ``SE = sqrt((1 + SR^2/2) * eta(q) / T)``. The ``(1 + SR^2/2)`` term is the asymptotic variance
    of the Sharpe estimator itself, and ``eta(q)`` is the serial-correlation correction; at
    ``q = 0`` the second disappears and this is the textbook formula.
    """
    n = len(returns)
    if n < MIN_OBSERVATIONS:
        raise MetricError(
            f"{n} observation(s) is below the {MIN_OBSERVATIONS} a dependence correction needs; "
            f"estimating six autocorrelations from forty points is fitting noise to noise"
        )
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    sd = sqrt(variance)
    if variance <= 0 or _is_negligible(sd, mean):
        raise MetricError("a constant series has no Sharpe ratio and therefore no interval")
    sharpe = mean / sd
    q = auto_lag_truncation(n) if lags is None else lags
    q = max(0, min(q, n - 1))
    rhos = autocorrelations(returns, q)
    eta = newey_west_eta(rhos)
    if eta <= 0:
        raise MetricError(
            f"the Newey-West correction came out at {eta:.6f}. The Bartlett kernel makes this "
            f"estimator positive semi-definite, so a non-positive value is a computation defect "
            f"and not a property of the data; flooring it would turn that defect into a number"
        )
    base = 1.0 + 0.5 * sharpe**2
    se_iid = sqrt(base / n)
    se_hac = sqrt(base * eta / n)
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    return HACSharpe(
        sharpe=sharpe,
        observations=n,
        lags=q,
        eta=eta,
        se_iid=se_iid,
        se_hac=se_hac,
        low=sharpe - z * se_hac,
        high=sharpe + z * se_hac,
        rhos=tuple(rhos),
    )


def _geometric(rng: random.Random, p: float) -> int:
    """A draw from the geometric distribution on {1, 2, ...} with mean ``1/p``.

    Inverse transform: ``floor(log(U) / log(1 - p)) + 1``. ``random`` has no geometric variate and
    a rejection loop would be slower and no clearer.
    """
    if not 0.0 < p <= 1.0:
        raise MetricError(f"the block probability must be in (0, 1], got {p}")
    if p == 1.0:
        return 1
    u = rng.random()
    while u <= 0.0:  # log(0) is undefined; a zero draw is astronomically rare but not impossible
        u = rng.random()
    return floor(log(u) / log(1.0 - p)) + 1


def stationary_bootstrap_indices(
    n: int, *, block_length: float, rng: random.Random
) -> list[int]:
    """One resample of ``[0, n)`` under Politis and Romano (1994).

    Geometric block lengths with mean ``block_length``, uniform starting points, wrapped circularly
    so every observation is equally likely to appear — without the wrap, the observations near the
    end of the series are systematically under-sampled.
    """
    if n < 2:
        raise MetricError("a bootstrap over fewer than two observations resamples nothing")
    if block_length <= 0:
        raise MetricError("the mean block length must be positive")
    p = min(1.0, 1.0 / block_length)
    out: list[int] = []
    while len(out) < n:
        start = rng.randrange(n)
        length = min(_geometric(rng, p), n - len(out))
        out.extend((start + k) % n for k in range(length))
    return out


def moving_block_indices(n: int, *, block_length: int, rng: random.Random) -> list[int]:
    """The fixed-length circular block bootstrap, for comparison against the stationary one.

    Included because the difference between the two is worth being able to show rather than assert:
    a fixed block length leaves the resample non-stationary, and on a series with a strong seasonal
    period equal to the block length, the difference is visible rather than theoretical.
    """
    if n < 2:
        raise MetricError("a bootstrap over fewer than two observations resamples nothing")
    if block_length < 1:
        raise MetricError("the block length must be at least one observation")
    out: list[int] = []
    while len(out) < n:
        start = rng.randrange(n)
        length = min(block_length, n - len(out))
        out.extend((start + k) % n for k in range(length))
    return out


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """A percentile interval from a dependence-preserving resample."""

    statistic: float
    low: float
    high: float
    resamples: int
    block_length: float
    alpha: float
    failures: int
    """Resamples that produced no computable statistic — a flat draw, typically. Reported rather
    than dropped silently, because a large count means the interval rests on fewer draws than the
    headline says."""

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "statistic": round(self.statistic, 6),
            "ci_low": round(self.low, 6),
            "ci_high": round(self.high, 6),
            "resamples": self.resamples,
            "usable_resamples": self.resamples - self.failures,
            "block_length": round(self.block_length, 3),
            "alpha": self.alpha,
            "excludes_zero": self.excludes_zero,
        }


def _sharpe_of(values: Sequence[float]) -> float | None:
    """None rather than a number when the draw is degenerate.

    The negligible-variance guard is not optional here and the test suite found out the hard way:
    a constant series of 0.02 returns has a sample variance around 1e-36 rather than zero, so a
    plain ``variance <= 0`` test passes it through and the bootstrap resamples a Sharpe of 1e17.
    Third appearance of this failure in this package, which is why it now goes through the one
    shared guard in `metrics` rather than being re-tested locally.
    """
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    if variance <= 0:
        return None
    sd = sqrt(variance)
    if _is_negligible(sd, mean):
        return None
    return mean / sd


def bootstrap_sharpe(
    returns: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    block_length: float | None = None,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> BootstrapInterval:
    """A percentile interval for the Sharpe ratio under the stationary bootstrap.

    The non-parametric companion to :func:`hac_sharpe`. Where they disagree, the disagreement is
    informative: the closed form assumes the Sharpe estimator is asymptotically normal, and on a
    skewed or fat-tailed series it is not.
    """
    n = len(returns)
    if n < MIN_OBSERVATIONS:
        raise MetricError(
            f"{n} observation(s) is below the {MIN_OBSERVATIONS} a block bootstrap needs"
        )
    observed = _sharpe_of(returns)
    if observed is None:
        raise MetricError("a constant series has no Sharpe ratio to resample")
    # The data-driven rule, not sqrt(n). On an AR(0.6) series of 2,000 points the folklore gives
    # 44.7 and the rule gives 21.3, so the fallback was roughly twice too long — and a block
    # longer than the dependence needs throws away independent information for nothing.
    length = optimal_block_length(returns)[0] if block_length is None else block_length
    rng = random.Random(seed)
    draws: list[float] = []
    failures = 0
    for _ in range(resamples):
        indices = stationary_bootstrap_indices(n, block_length=length, rng=rng)
        value = _sharpe_of([returns[i] for i in indices])
        if value is None:
            failures += 1
            continue
        draws.append(value)
    if len(draws) < resamples // 2:
        raise MetricError(
            f"only {len(draws)} of {resamples} resamples produced a Sharpe ratio; an interval "
            f"from that many draws would be an interval about the resampling, not the returns"
        )
    draws.sort()
    lo_index = max(0, min(len(draws) - 1, int(alpha / 2.0 * len(draws))))
    hi_index = max(0, min(len(draws) - 1, int((1.0 - alpha / 2.0) * len(draws))))
    return BootstrapInterval(
        statistic=observed,
        low=draws[lo_index],
        high=draws[hi_index],
        resamples=resamples,
        block_length=length,
        alpha=alpha,
        failures=failures,
    )


@dataclass(frozen=True, slots=True)
class RealityCheck:
    """White (2000): did the best of N strategies really beat the benchmark?"""

    p_value: float
    best_index: int
    best_mean_excess: float
    strategies: int
    observations: int
    resamples: int
    block_length: float

    @property
    def verdict(self) -> str:
        if self.p_value <= 0.05:
            return (
                f"the best of {self.strategies} strategies beats the benchmark at p="
                f"{self.p_value:.4f}, after paying for the fact that {self.strategies} were tried"
            )
        return (
            f"NOT ESTABLISHED: p={self.p_value:.4f}. The best of {self.strategies} strategies is "
            f"not distinguishable from what the best of {self.strategies} random ones would have "
            f"looked like against this benchmark"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "p_value": round(self.p_value, 6),
            "best_index": self.best_index,
            "best_mean_excess": round(self.best_mean_excess, 8),
            "strategies": self.strategies,
            "observations": self.observations,
            "resamples": self.resamples,
            "block_length": round(self.block_length, 3),
            "verdict": self.verdict,
        }


def reality_check(
    strategies: Sequence[Sequence[float]],
    benchmark: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    block_length: float | None = None,
    seed: int = DEFAULT_SEED,
) -> RealityCheck:
    """White's Reality Check on the excess-return series, via the stationary bootstrap.

    ``strategies`` is one return series per strategy; ``benchmark`` is the series each is measured
    against. The statistic is ``sqrt(T) * max_n mean(f_n)`` where ``f_n`` is strategy *n* minus the
    benchmark, and the null distribution comes from resampling the ``f`` series **recentred** on
    their own observed means — the recentring is what imposes "no strategy beats the benchmark",
    and omitting it is the mistake that turns this test into a rubber stamp.

    Hansen's (2005) studentised variant is deliberately **not** implemented: the reference this was
    read from does not implement it either and flags it as an extension, so writing one here would
    mean writing it from memory against the standing rule.
    """
    if not strategies:
        raise MetricError("a reality check over no strategies has no best strategy")
    n = len(benchmark)
    if n < MIN_OBSERVATIONS:
        raise MetricError(
            f"{n} observation(s) is below the {MIN_OBSERVATIONS} this bootstrap needs"
        )
    if any(len(s) != n for s in strategies):
        raise MetricError("every strategy must be scored over the same periods as the benchmark")

    excess = [[s[t] - benchmark[t] for t in range(n)] for s in strategies]
    means = [sum(f) / n for f in excess]
    best = max(range(len(means)), key=lambda i: means[i])
    observed = sqrt(n) * means[best]

    # Chosen from the benchmark-relative series, which is what the bootstrap actually resamples.
    length = (
        optimal_block_length(excess[best])[0] if block_length is None else block_length
    )
    rng = random.Random(seed)
    at_least_as_extreme = 0
    for _ in range(resamples):
        indices = stationary_bootstrap_indices(n, block_length=length, rng=rng)
        centred_max = max(
            sqrt(n) * (sum(f[i] for i in indices) / n - mean)
            for f, mean in zip(excess, means, strict=True)
        )
        if centred_max >= observed:
            at_least_as_extreme += 1
    return RealityCheck(
        p_value=at_least_as_extreme / resamples,
        best_index=best,
        best_mean_excess=means[best],
        strategies=len(strategies),
        observations=n,
        resamples=resamples,
        block_length=length,
    )


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "MIN_OBSERVATIONS",
    "BootstrapInterval",
    "HACSharpe",
    "RealityCheck",
    "auto_lag_truncation",
    "autocorrelations",
    "bootstrap_sharpe",
    "hac_sharpe",
    "moving_block_indices",
    "newey_west_eta",
    "optimal_block_length",
    "reality_check",
    "stationary_bootstrap_indices",
    "suggested_block_length",
]
