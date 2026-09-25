"""Hansen's SPA and Romano-Wolf's StepM — a trial-corrected discovery gate that keeps its power.

**Why this module exists: the deflated Sharpe gate could not pass anything.** The "Factor-discovery
safety" capability claimed that ARGUS's deflated Sharpe ratio (`metrics.deflated_sharpe`) "refuses
the exact discovery" a 400-trial `searchoff` pool would otherwise report. It does. It also refuses a
genuinely skilled rule planted in that same pool at a per-bar Sharpe of 0.20 — a t-statistic above
six — every single time (`eval/general_factorsafety_comparison.py`). The deflation benchmarks the
best trial against ``sqrt(V) * E[max of N standard normals]``, where ``V`` is the variance of the
trials' Sharpe ratios. That formula assumes every trial's true Sharpe is zero, so all of ``V`` is
estimation noise. In a searched grammar pool it is not: most candidates are fast-switching rules
that the 12 bps cost destroys, their true Sharpes are strongly negative, and the dispersion they
add inflates ``V`` — and with it the bar — far past anything a real edge clears. A gate with no
power is not a trial correction; it is a refusal that happens to be right when nothing is there.

**What replaces it, and where it came from.** The general-purpose answer to "the best of N
correlated models beat the benchmark — or did it?" comes from forecast evaluation in econometrics,
not from trading: White's (2000) Reality Check, Hansen's (2005) Test for Superior Predictive
Ability, and Romano and Wolf's (2005) StepM. The strongest open implementation is Kevin Sheppard's
``arch`` package (``arch.bootstrap.SPA`` / ``StepM``; NCSA-style licence, attribution retained in
``eval/baselines/arch_spa_stepm.py``). It was read in full (arch 8.0.0,
``arch/bootstrap/multiple_comparison.py:355-808``), vendored for the comparison, run on the same
pool, and it beat the deflated Sharpe gate outright at matched false-discovery control. This module
is ARGUS's adaptation of it. What was taken, what was changed, and why:

* **Taken, unchanged in meaning:** the three recentring rules (``lower`` / ``consistent`` /
  ``upper``, ``multiple_comparison.py:661-680``), the ``2 log log n`` validity threshold for the
  consistent p-value (``:711-725``), the percentile critical value (``:742-765``), and the StepM
  step-down that removes rejected models and recomputes the critical value on the rest
  (``:456-480``).
* **Changed — studentization is applied, not just computed.** ``arch``'s ``SPA`` accepts
  ``studentize=True`` (its default), computes a variance per model in ``_compute_variance``, and
  then never divides by it: ``compute()`` and ``_simulate_values()`` compare raw means
  (``:634-680``). The flag only changes a label in ``_info``. Sheppard's own earlier MATLAB
  implementation, ``MFE-Toolbox/bootstrap/bsds.m``, does divide — its default is ``'STUDENTIZED'``,
  documented as "generally leads to better power" — and Hansen (2005) defines the statistic on the
  studentized scale. This module studentizes, and ``studentize=False`` reproduces ``arch`` exactly
  so the difference is an ablation rather than an assertion.
* **Changed — the statistic is floored at zero**, as ``bsds.m`` does (``min(perf, 0)`` on losses):
  when every model is worse than the benchmark the p-value is 1, not merely large.
* **Changed — the variance estimator.** ``arch``'s default is the stationary-bootstrap kernel sum,
  O(T^2 K); this package is pure Python, so that is minutes per call. The estimator used instead is
  ``arch``'s own ``nested=True`` path (``:693-697``): ``T * var(bootstrap means)`` over the same
  index draws. Both converge to the same long-run variance; parity with ``arch``'s
  ``nested=True`` is exact and tested.
* **Added — a block-sum fast path.** A stationary-bootstrap resample is a list of circular blocks,
  so each resampled mean is a sum of prefix-sum differences, O(blocks) instead of O(T) per model
  per draw. The draws themselves come from :func:`dependence.stationary_bootstrap_indices`'s own
  RNG procedure, block for block, so a seed here names the same resample it names there.

**What this gate still does not do.** It tests whether each candidate beats a benchmark *in the
sample it is given*. It does not make an in-sample winner an out-of-sample one; `searchoff` still
reports the out-of-sample half separately, and a survivor here is a candidate for that check, not a
substitute for it.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from math import log, sqrt
from typing import Any

from argus.backtest.dependence import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    MIN_OBSERVATIONS,
    _geometric,
    optimal_block_length,
)
from argus.backtest.metrics import MetricError, _is_negligible

DEFAULT_SIZE = 0.05
"""Family-wise error rate the StepM controls, and the level at which an SPA p-value rejects."""

Block = tuple[int, int]
"""``(start, length)`` of one circular block of a stationary-bootstrap resample."""


def stationary_bootstrap_blocks(n: int, *, block_length: float, rng: random.Random) -> list[Block]:
    """One Politis-Romano resample of ``[0, n)``, as the blocks it is made of.

    Consumes ``rng`` exactly as :func:`argus.backtest.dependence.stationary_bootstrap_indices` does
    — a ``randrange(n)`` start, then a geometric length truncated to what is left — so expanding
    the blocks reproduces that function's index list for the same generator state, draw for draw
    (``tests/test_snooping.py`` checks it). Returning blocks rather than indices is what lets a
    resampled mean cost O(blocks) instead of O(n).
    """
    if n < 2:
        raise MetricError("a bootstrap over fewer than two observations resamples nothing")
    if block_length <= 0:
        raise MetricError("the mean block length must be positive")
    p = min(1.0, 1.0 / block_length)
    out: list[Block] = []
    filled = 0
    while filled < n:
        start = rng.randrange(n)
        length = min(_geometric(rng, p), n - filled)
        out.append((start, length))
        filled += length
    return out


def expand_blocks(blocks: Sequence[Block], n: int) -> list[int]:
    """The index list a block list stands for. Used by tests and by callers that need indices."""
    return [(start + k) % n for start, length in blocks for k in range(length)]


def _prefix(column: Sequence[float]) -> list[float]:
    out = [0.0]
    running = 0.0
    for value in column:
        running += value
        out.append(running)
    return out


def _block_mean(prefix: list[float], blocks: Sequence[Block], n: int) -> float:
    total = 0.0
    for start, length in blocks:
        end = start + length
        if end <= n:
            total += prefix[end] - prefix[start]
        else:
            total += (prefix[n] - prefix[start]) + prefix[end - n]
    return total / n


@dataclass(frozen=True, slots=True)
class _Draws:
    """Everything the SPA and StepM need, computed once and shared between them."""

    means: tuple[float, ...]
    """Observed mean excess return of each model over the benchmark."""
    omegas: tuple[float, ...]
    """Long-run standard deviation of each model's excess, ``sqrt(T * var(bootstrap means))``."""
    boot: tuple[tuple[float, ...], ...]
    """``boot[b][k]``: model ``k``'s mean excess in resample ``b``."""
    observations: int
    block_length: float


def _excess(
    strategies: Sequence[Sequence[float]], benchmark: Sequence[float]
) -> list[list[float]]:
    if not strategies:
        raise MetricError("a data-snooping test over no strategies has no best strategy")
    n = len(benchmark)
    if n < MIN_OBSERVATIONS:
        raise MetricError(
            f"{n} observation(s) is below the {MIN_OBSERVATIONS} this bootstrap needs"
        )
    if any(len(s) != n for s in strategies):
        raise MetricError("every strategy must be scored over the same periods as the benchmark")
    excess = [[s[t] - benchmark[t] for t in range(n)] for s in strategies]
    for k, column in enumerate(excess):
        if any(v != v for v in column):
            raise MetricError(f"strategy {k} has a NaN excess return; the test cannot rank it")
    return excess


def _draw(
    excess: list[list[float]],
    *,
    resamples: int,
    block_length: float | None,
    seed: int,
    indices: Sequence[Sequence[int]] | None,
) -> _Draws:
    n = len(excess[0])
    means = [sum(column) / n for column in excess]
    if block_length is None:
        # Chosen from the best model's excess, as `dependence.reality_check` does: the block
        # length must be one number for a joint resample, and the model the test is about is the
        # one whose dependence matters most.
        best = max(range(len(means)), key=lambda k: means[k])
        block_length = optimal_block_length(excess[best])[0]

    boot: list[tuple[float, ...]] = []
    if indices is None:
        prefixes = [_prefix(column) for column in excess]
        rng = random.Random(seed)
        for _ in range(resamples):
            blocks = stationary_bootstrap_blocks(n, block_length=block_length, rng=rng)
            boot.append(tuple(_block_mean(prefix, blocks, n) for prefix in prefixes))
    else:
        # Externally supplied resamples — how parity with another implementation's exact draws is
        # tested. Direct summation, O(n) per model per draw; not the path production takes.
        if len(indices) != resamples:
            raise MetricError(f"{len(indices)} index draws supplied for {resamples} resamples")
        for draw in indices:
            if len(draw) != n:
                raise MetricError("every supplied resample must have one index per observation")
            boot.append(tuple(sum(column[i] for i in draw) / n for column in excess))

    omegas: list[float] = []
    for k in range(len(means)):
        # arch's nested estimator (multiple_comparison.py:693-697): T times the population
        # variance of the resampled means. Centring on the resample average rather than on the
        # observed mean is what `numpy.var` does, and what parity with it requires.
        column = [row[k] for row in boot]
        centre = sum(column) / len(column)
        variance = n * sum((v - centre) ** 2 for v in column) / len(column)
        omegas.append(sqrt(variance) if variance > 0 else 0.0)
    return _Draws(
        means=tuple(means), omegas=tuple(omegas), boot=tuple(boot),
        observations=n, block_length=block_length,
    )


def _recentring(draws: _Draws) -> dict[str, tuple[float, ...]]:
    """Hansen's three choices of what to subtract from each resampled mean.

    Read from ``arch/bootstrap/multiple_comparison.py:661-680`` and ``bsds.m``'s ``gc/gl/gu``:
    ``upper`` recentres every model to zero (White's Reality Check — every model is treated as
    relevant, however bad); ``lower`` leaves a model worse than the benchmark centred on its own
    negative mean; ``consistent`` does the same only for models worse by more than
    ``sqrt(omega^2 / T * 2 log log T)``, the rule that makes the test consistent (Hansen 2005).
    """
    n = draws.observations
    upper = draws.means
    lower = tuple(max(m, 0.0) for m in draws.means)
    loglog = 2.0 * log(log(n))
    consistent = tuple(
        m if m >= -sqrt(omega * omega / n * loglog) else 0.0
        for m, omega in zip(draws.means, draws.omegas, strict=True)
    )
    return {"lower": lower, "consistent": consistent, "upper": upper}


def _scales(draws: _Draws, studentize: bool) -> tuple[float, ...]:
    if not studentize:
        return tuple(1.0 for _ in draws.means)
    for k, (omega, mean) in enumerate(zip(draws.omegas, draws.means, strict=True)):
        if omega <= 0 or _is_negligible(omega, mean):
            raise MetricError(
                f"strategy {k} has no resampling variance — its excess over the benchmark is "
                f"constant, so a studentized statistic for it is undefined; drop it or pass "
                f"studentize=False"
            )
    return draws.omegas


def _simulated_max(
    draws: _Draws, subtract: tuple[float, ...], scale: tuple[float, ...], active: Sequence[int],
    *, floor: bool,
) -> list[float]:
    out: list[float] = []
    for row in draws.boot:
        top = max((row[k] - subtract[k]) / scale[k] for k in active)
        out.append(max(top, 0.0) if floor else top)
    return out


def _percentile(values: list[float], q: float) -> float:
    """``numpy.percentile``'s default linear interpolation, which ``arch`` uses (``:764``)."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@dataclass(frozen=True, slots=True)
class SpaTest:
    """Hansen (2005): does the best of N models beat the benchmark, having tried N?"""

    p_lower: float
    p_consistent: float
    p_upper: float
    """White's Reality Check p-value. Conservative when poor models are in the set."""
    best_index: int
    best_mean_excess: float
    statistic: float
    strategies: int
    observations: int
    resamples: int
    block_length: float
    studentized: bool

    @property
    def rejects(self) -> bool:
        """The consistent p-value at :data:`DEFAULT_SIZE` — Hansen's recommended one."""
        return self.p_consistent <= DEFAULT_SIZE

    @property
    def verdict(self) -> str:
        if self.rejects:
            return (
                f"model {self.best_index} beats the benchmark at p={self.p_consistent:.4f} "
                f"(consistent), after paying for all {self.strategies} models tried"
            )
        return (
            f"NOT ESTABLISHED: consistent p={self.p_consistent:.4f}. The best of "
            f"{self.strategies} is not distinguishable from what the best of {self.strategies} "
            f"with no edge would look like on this data"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "p_lower": round(self.p_lower, 6),
            "p_consistent": round(self.p_consistent, 6),
            "p_upper": round(self.p_upper, 6),
            "best_index": self.best_index,
            "best_mean_excess": round(self.best_mean_excess, 8),
            "statistic": round(self.statistic, 6),
            "strategies": self.strategies,
            "observations": self.observations,
            "resamples": self.resamples,
            "block_length": round(self.block_length, 3),
            "studentized": self.studentized,
            "rejects": self.rejects,
            "verdict": self.verdict,
        }


def _spa_from(draws: _Draws, *, studentize: bool, resamples: int) -> SpaTest:
    scale = _scales(draws, studentize)
    everyone = range(len(draws.means))
    statistics = [m / s for m, s in zip(draws.means, scale, strict=True)]
    best = max(everyone, key=lambda k: statistics[k])
    observed = statistics[best]
    pvalues: dict[str, float] = {}
    for name, subtract in _recentring(draws).items():
        simulated = _simulated_max(draws, subtract, scale, everyone, floor=studentize)
        # Strictly greater, as arch (`:655`) and Hansen define it; a tie is not evidence against.
        pvalues[name] = sum(1 for v in simulated if v > observed) / len(simulated)
    return SpaTest(
        p_lower=pvalues["lower"], p_consistent=pvalues["consistent"], p_upper=pvalues["upper"],
        best_index=best, best_mean_excess=draws.means[best], statistic=observed,
        strategies=len(draws.means), observations=draws.observations, resamples=resamples,
        block_length=draws.block_length, studentized=studentize,
    )


@dataclass(frozen=True, slots=True)
class StepMResult:
    """Romano and Wolf (2005): *which* of N models beat the benchmark, with the FWER held."""

    superior: tuple[int, ...]
    """Indices of the models rejected as no better than the benchmark, ascending."""
    steps: tuple[tuple[int, ...], ...]
    """The rejections made at each step-down round, in order."""
    size: float
    strategies: int
    observations: int
    resamples: int
    block_length: float
    studentized: bool
    spa: SpaTest

    @property
    def verdict(self) -> str:
        if not self.superior:
            return (
                f"NONE of {self.strategies} models beats the benchmark with the family-wise error "
                f"held at {self.size:.0%}"
            )
        return (
            f"{len(self.superior)} of {self.strategies} models beat the benchmark with the "
            f"family-wise error held at {self.size:.0%}: {list(self.superior)[:10]}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "superior": list(self.superior),
            "steps": [list(s) for s in self.steps],
            "size": self.size,
            "strategies": self.strategies,
            "observations": self.observations,
            "resamples": self.resamples,
            "block_length": round(self.block_length, 3),
            "studentized": self.studentized,
            "spa": self.spa.as_dict(),
            "verdict": self.verdict,
        }


class SnoopingBootstrap:
    """One set of stationary-bootstrap draws over one family of models, asked several questions.

    The draws are the expensive part — every model's resampled mean in every resample — and the SPA
    p-values, the StepM step-down, and the studentized and raw variants of both all read the same
    draws. Building them once is what makes an ablation of studentization a comparison on
    identical resamples rather than on two different ones.
    """

    def __init__(
        self,
        strategies: Sequence[Sequence[float]],
        benchmark: Sequence[float],
        *,
        resamples: int = DEFAULT_RESAMPLES,
        block_length: float | None = None,
        seed: int = DEFAULT_SEED,
        indices: Sequence[Sequence[int]] | None = None,
    ) -> None:
        if resamples < 1:
            raise MetricError("a bootstrap needs at least one resample")
        self.resamples = resamples
        self._draws = _draw(
            _excess(strategies, benchmark), resamples=resamples, block_length=block_length,
            seed=seed, indices=indices,
        )

    @property
    def block_length(self) -> float:
        return self._draws.block_length

    def spa(self, *, studentize: bool = True) -> SpaTest:
        """Hansen's SPA p-values. ``studentize=False`` is ``arch`` 8.0.0's actual behaviour."""
        return _spa_from(self._draws, studentize=studentize, resamples=self.resamples)

    def stepm(self, *, size: float = DEFAULT_SIZE, studentize: bool = True) -> StepMResult:
        """StepM on the consistent recentring, as ``arch``'s ``StepM.compute`` (``:456-480``).

        Round one rejects every model whose statistic exceeds the ``1 - size`` quantile of the
        simulated maximum over all models. Each later round drops the models already rejected
        and recomputes that quantile over the rest, which can only lower it; it stops when a
        round rejects nothing. It also stops once every model is rejected — the condition
        ``arch`` 8.0.0 gets wrong (it compares the latest round's count with ``k`` rather than
        the running total, then takes the maximum of an empty selection and raises; fixed
        upstream after that release, bashtage/arch#862).
        """
        if not 0.0 < size < 1.0:
            raise MetricError(f"size must be in (0, 1), got {size}")
        draws = self._draws
        scale = _scales(draws, studentize)
        subtract = _recentring(draws)["consistent"]
        statistics = [m / s for m, s in zip(draws.means, scale, strict=True)]
        k = len(draws.means)

        rejected: list[int] = []
        steps: list[tuple[int, ...]] = []
        while len(rejected) < k:
            done = set(rejected)
            active = [j for j in range(k) if j not in done]
            simulated = _simulated_max(draws, subtract, scale, active, floor=studentize)
            critical = _percentile(simulated, 100.0 * (1.0 - size))
            found = tuple(j for j in active if statistics[j] > critical)
            if not found:
                break
            steps.append(found)
            rejected.extend(found)

        return StepMResult(
            superior=tuple(sorted(rejected)), steps=tuple(steps), size=size, strategies=k,
            observations=draws.observations, resamples=self.resamples,
            block_length=draws.block_length, studentized=studentize,
            spa=self.spa(studentize=studentize),
        )


def spa_test(
    strategies: Sequence[Sequence[float]],
    benchmark: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    block_length: float | None = None,
    seed: int = DEFAULT_SEED,
    studentize: bool = True,
    indices: Sequence[Sequence[int]] | None = None,
) -> SpaTest:
    """Hansen's SPA test on ``strategies`` (one return series each) against ``benchmark``.

    ``indices`` supplies the resamples explicitly, for parity tests against another
    implementation's exact draws. See :class:`SnoopingBootstrap` for asking several questions of
    one set of draws.
    """
    return SnoopingBootstrap(
        strategies, benchmark, resamples=resamples, block_length=block_length, seed=seed,
        indices=indices,
    ).spa(studentize=studentize)


def stepm(
    strategies: Sequence[Sequence[float]],
    benchmark: Sequence[float],
    *,
    size: float = DEFAULT_SIZE,
    resamples: int = DEFAULT_RESAMPLES,
    block_length: float | None = None,
    seed: int = DEFAULT_SEED,
    studentize: bool = True,
    indices: Sequence[Sequence[int]] | None = None,
) -> StepMResult:
    """Romano-Wolf StepM against ``benchmark``; see :meth:`SnoopingBootstrap.stepm`."""
    return SnoopingBootstrap(
        strategies, benchmark, resamples=resamples, block_length=block_length, seed=seed,
        indices=indices,
    ).stepm(size=size, studentize=studentize)


__all__ = [
    "DEFAULT_SIZE",
    "Block",
    "SnoopingBootstrap",
    "SpaTest",
    "StepMResult",
    "expand_blocks",
    "spa_test",
    "stationary_bootstrap_blocks",
    "stepm",
]
