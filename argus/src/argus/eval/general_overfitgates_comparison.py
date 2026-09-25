"""ARGUS's overfitting gates against the general-purpose tools built to stop a NaN reaching a gate.

``eval/standing.py`` scores "Overfitting gates that raise instead of returning NaN" as OWNED against
one rival, vectorbt — a trading library. The owner's rule of 2026-09-25 is that the best
implementation of a *function* often lives outside trading, and losing to a general-purpose tool
is still losing. So this module first names the function in general terms and then runs the
strongest general-purpose implementations of it on the same inputs.

**The function, stated generally.** A statistical decision gate — a selection-bias correction
(deflated / probabilistic Sharpe), a multiple-testing correction (Benjamini-Hochberg, Bonferroni),
an overfitting probability (CSCV PBO), a minimum track record — that must *refuse* every input on
which it has no honest answer, instead of handing a NaN (or a finite number built from a NaN) to
a threshold comparison. ``NaN > t`` and ``NaN < t`` are both ``False`` (IEEE 754), so whichever
way round a caller writes the gate, one of the two shapes admits it. In general-software terms
this is **runtime contract enforcement on a numeric function** (argument and return validation),
**IEEE-754 floating-point exception trapping**, and — for finding the inputs that break it —
**property-based adversarial search**.

**Rivals considered, and why these were run.** Stars were not the criterion; adoption, maintenance
and whether the method is the reference design for the function were.

* **pydantic v2** (MIT, v2.13 here; the most-installed Python validation library, maintained,
  Rust core). ``Annotated[float, AllowInfNan(False)]`` — ``FiniteFloat``,
  ``pydantic/types.py:643`` (``AllowInfNan`` at ``types.py:386-407``; enforced by
  ``pydantic_core.core_schema.float_schema(allow_inf_nan=...)``, ``core_schema.py:706-708``) —
  plus ``validate_call(validate_return=True)`` (``pydantic/validate_call_decorator.py:74,99``)
  is the reference design for "validate every argument at the boundary and validate the result
  too". **Run**, wrapped around vectorbt's real vendored DSR math, with the contract a pydantic
  user would write for this signature.
* **numpy / scipy floating-point trapping** (BSD). ``np.errstate(all="raise")``,
  ``scipy.special.errstate(all="raise")`` and warnings-as-errors turn the IEEE invalid /
  divide-by-zero / overflow flags into exceptions — the zero-code general mechanism a numpy user
  reaches for. **Run**, both around the gate alone and around the whole producer pipeline.
* **scipy.stats.false_discovery_control** (BSD) and **statsmodels multipletests** (BSD) — the
  general-purpose multiple-testing implementations. scipy validates its p-values with one
  NaN-proof comparison, ``ps == clip(ps, 0, 1)`` (``scipy/stats/_morestats.py:4791-4794``);
  statsmodels validates nothing (``statsmodels/stats/multitest.py:99`` onward). **Run.**
* **Hypothesis** (MPL-2.0, used as a tool, nothing vendored) — the standard property-based
  testing engine; its float strategy deliberately draws NaN, ±inf, subnormals and boundary values.
  For the *adversarial search* half of the function it is the general-purpose best, and it is set
  against the hand-picked adversarial inputs ``eval/dsr_comparison.py`` used. **Run**,
  derandomized so the search is reproducible.
* Considered, not run: JAX ``jax_debug_nans`` / ``jax_debug_infs`` (Apache-2.0) — a debugging mode
  its own docs mark as slow and not for production, and it would need the formula ported to
  ``jax.numpy``, so it would not be vectorbt's code on the same input; icontract / deal (MIT) —
  design-by-contract with the same pre/post-condition semantics as pydantic's and a fraction of its
  adoption; pandera / Great Expectations — dataframe validation, not function contracts.

**What the measurement found, stated before the code so it cannot be softened later** (the live
numbers are in the artefact; the prose below is checked against them by the test):

1. Before this module, ARGUS's gates *lost* to the general-purpose contract on the named property.
   The pydantic-wrapped vectorbt DSR refused every non-finite argument; ARGUS's ``deflated_sharpe``
   returned NaN for a NaN skew or kurtosis and for an infinite observed Sharpe, and returned 0.5
   for an infinite kurtosis — the NaN-variance guard ``dsr_comparison`` added covered one argument
   of four. Hypothesis found silent failures in every ARGUS gate it was pointed at (see
   ``hypothesis_search``); the hand-picked adversarial set had exercised two inputs.
2. The general tools do not win everywhere. pydantic's return contract cannot tell a *wrong*
   finite value from a right one: on an observed Sharpe large enough to overflow ``SR**2`` vectorbt
   returns 0.5 and the contract passes it. The numpy trap cannot see a NaN that arrives as an input
   (propagating a quiet NaN raises no IEEE flag): around the gate alone it refuses only where the
   arithmetic itself raises a flag — an infinite argument driving an invalid operation, or an
   overflow — and on the real-producer tier, where every poison arrives as a NaN, it refuses none
   of the cases the bare library passes silently. And vectorbt's formula, contract or no contract,
   certifies any strategy with DSR = 1.0 on a single trial with a positive variance
   (``norm.ppf(0) = -inf``), which ARGUS's exact one-trial reduction does not.
3. What was adapted, with attribution: pydantic's two rules — every float argument finite at the
   boundary, and the result validated before it leaves — and scipy's NaN-proof positive-form range
   check, reimplemented with ``math.isfinite`` in ``backtest/metrics.py`` and
   ``backtest/validation.py`` (no new runtime dependency), every arithmetic failure normalised to
   ``MetricError``, and the same p-value contract added to the desk's own Benjamini-Hochberg in
   ``desk/review.py``. The pre-adaptation code is frozen below (``_Pre*``) so the before/after is a
   rerunnable ablation, not a memory.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import json
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from math import exp, isfinite, sqrt
from pathlib import Path
from statistics import NormalDist
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
from pydantic import Field, ValidationError, validate_call

from argus.backtest import metrics as argus_metrics
from argus.backtest import validation as argus_validation
from argus.backtest.metrics import MetricError
from argus.desk import review as desk_review
from argus.eval.dsr_comparison import DsrCase, designed_cases, swept_cases

if TYPE_CHECKING:  # Hypothesis is a dev dependency, imported at call time below.
    from hypothesis.strategies import SearchStrategy

_ROOT = Path(__file__).resolve().parents[3]
CANDLES = _ROOT / "data" / "regime_candles_fixture.json"
ARTEFACT = _ROOT / "data" / "general_overfitgates_comparison.json"

GATE_THRESHOLD = 0.95
"""The DSR gate level ARGUS's own factor lab uses (``research/factor_lab.py``: ``dsr <= 0.95``)."""

ROUND_TRIP_FEE = 0.0012
"""12 bps round-trip taker, charged 6 bps per side on every change of position."""

HOURLY_PER_YEAR = 24 * 365

POISONS: tuple[float, ...] = (math.nan, math.inf, -math.inf)


class GeneralOverfitGatesError(RuntimeError):
    """The comparison could not run: a baseline or the real candle file failed to load."""


# =============================================================================================
# The pre-adaptation ARGUS gates, frozen verbatim (docstrings and comments removed, logic
# untouched) from backtest/metrics.py, backtest/validation.py and desk/review.py as they stood on
# 2026-09-25 before this comparison's findings were adapted into them. Kept so the ablation can be
# rerun forever rather than remembered. Not used by anything but this module.
# =============================================================================================


def _pre_is_negligible(sd: float, reference: float) -> bool:
    scale = max(abs(reference), 1e-12)
    return sd <= scale * 1e-12


def _pre_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _pre_stdev(xs: list[float], *, sample: bool = True) -> float:
    n = len(xs)
    if n < 2:
        raise MetricError("standard deviation needs at least two observations")
    mu = _pre_mean(xs)
    denom = n - 1 if sample else n
    return sqrt(sum((x - mu) ** 2 for x in xs) / denom)


def _pre_sharpe(returns: list[float], *, periods_per_year: int, risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        raise MetricError("Sharpe needs at least two returns")
    excess = [r - risk_free / periods_per_year for r in returns]
    sd = _pre_stdev(excess)
    mu = _pre_mean(excess)
    if _pre_is_negligible(sd, mu):
        raise MetricError("Sharpe is undefined for a zero-variance series")
    return _pre_mean(excess) / sd * sqrt(periods_per_year)


def _pre_sortino(returns: list[float], *, periods_per_year: int, target: float = 0.0) -> float:
    if len(returns) < 2:
        raise MetricError("Sortino needs at least two returns")
    downside = [min(0.0, r - target) for r in returns]
    dd = sqrt(sum(d * d for d in downside) / len(returns))
    if _pre_is_negligible(dd, _pre_mean(returns) - target):
        raise MetricError("Sortino is undefined when there is no downside deviation")
    return (_pre_mean(returns) - target) / dd * sqrt(periods_per_year)


def _pre_probabilistic_sharpe(
    observed: float, *, benchmark: float, n: int, skew: float = 0.0, kurtosis: float = 3.0
) -> float:
    if n < 2:
        raise MetricError("probabilistic Sharpe needs at least two observations")
    if observed != observed or benchmark != benchmark:
        raise MetricError(
            "probabilistic Sharpe cannot be computed from a NaN observed or benchmark"
        )
    denom = sqrt(1 - skew * observed + (kurtosis - 1) / 4 * observed ** 2)
    if denom <= 0:
        raise MetricError("probabilistic Sharpe denominator is non-positive")
    z = (observed - benchmark) * sqrt(n - 1) / denom
    return NormalDist().cdf(z)


def _pre_deflated_sharpe(
    observed: float, *, n: int, trials: int, variance_of_trials: float,
    skew: float = 0.0, kurtosis: float = 3.0,
) -> float:
    if trials < 1:
        raise MetricError("trials must be at least 1 — an unrecorded trial count voids the gate")
    if variance_of_trials < 0 or variance_of_trials != variance_of_trials:
        raise MetricError("variance of trial Sharpes cannot be negative or NaN")
    if n < 2:
        raise MetricError("deflated Sharpe needs at least two observations")
    if trials == 1:
        expected_max = 0.0
    else:
        euler = 0.5772156649015329
        nd = NormalDist()
        a = nd.inv_cdf(1 - 1 / trials)
        b = nd.inv_cdf(1 - 1 / (trials * exp(1)))
        expected_max = sqrt(variance_of_trials) * ((1 - euler) * a + euler * b)
    return _pre_probabilistic_sharpe(
        observed, benchmark=expected_max, n=n, skew=skew, kurtosis=kurtosis
    )


def _pre_out_of_sample_decay(in_sample: float, out_of_sample: float) -> dict[str, float | bool]:
    ratio = out_of_sample / in_sample if in_sample != 0 else 0.0
    return {
        "in_sample_sharpe": round(in_sample, 3),
        "out_of_sample_sharpe": round(out_of_sample, 3),
        "retention_ratio": round(ratio, 3),
        "breaches_half_alert": ratio < 0.5,
    }


_PRE_MIN_OBSERVATIONS = 10


def _pre_moments(returns: Sequence[float]) -> tuple[float, float, float, float]:
    n = len(returns)
    if n < _PRE_MIN_OBSERVATIONS:
        raise MetricError(f"{n} observation(s) is below the {_PRE_MIN_OBSERVATIONS} needed")
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    sd = sqrt(variance)
    if variance <= 0 or _pre_is_negligible(sd, mean):
        raise MetricError("a zero-variance series has no Sharpe and therefore no track record")
    skew = sum(((r - mean) / sd) ** 3 for r in returns) / n
    kurtosis = sum(((r - mean) / sd) ** 4 for r in returns) / n
    return mean, sd, skew, kurtosis


def _pre_min_track_record_length(
    returns: Sequence[float], *, benchmark_sharpe: float = 0.0, confidence: float = 0.95,
) -> float:
    mean, sd, skew, kurtosis = _pre_moments(returns)
    sharpe = mean / sd
    if sharpe <= benchmark_sharpe:
        raise MetricError("observed Sharpe does not exceed the benchmark")
    z = NormalDist().inv_cdf(confidence)
    adjustment = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if adjustment <= 0:
        raise MetricError("the higher-moment adjustment is non-positive")
    return 1.0 + adjustment * (z / (sharpe - benchmark_sharpe)) ** 2


def _pre_probability_of_overfitting(
    matrix: Sequence[Sequence[float]], *, groups: int = 8
) -> float:
    """Frozen CSCV; returns only the PBO (the frozen result dataclass is not needed here)."""
    if groups < 2 or groups % 2:
        raise MetricError(f"groups must be an even number of at least 2, got {groups}")
    rows = [list(r) for r in matrix]
    if not rows:
        raise MetricError("no observations to cross-validate")
    strategies = len(rows[0])
    if strategies < 2:
        raise MetricError("CSCV compares strategies against each other; it needs at least two")
    if any(len(r) != strategies for r in rows):
        raise MetricError("every observation must carry a return for every strategy")
    if len(rows) < groups * _PRE_MIN_OBSERVATIONS:
        raise MetricError("too few observations for the blocks")
    size = len(rows) // groups
    blocks = [rows[i * size:(i + 1) * size] for i in range(groups)]

    def sharpe_of(block_set: Sequence[Sequence[Sequence[float]]], index: int) -> float:
        series = [row[index] for block in block_set for row in block]
        n = len(series)
        mean = sum(series) / n
        variance = sum((v - mean) ** 2 for v in series) / (n - 1) if n > 1 else 0.0
        return mean / sqrt(variance) if variance > 0 else 0.0

    ranks: list[float] = []
    half = groups // 2
    for train_idx in combinations(range(groups), half):
        test_idx = [i for i in range(groups) if i not in train_idx]
        train = [blocks[i] for i in train_idx]
        test = [blocks[i] for i in test_idx]
        in_sample = [sharpe_of(train, s) for s in range(strategies)]
        best = max(range(strategies), key=lambda s: in_sample[s])
        out_sample = [sharpe_of(test, s) for s in range(strategies)]
        chosen = out_sample[best]
        below = sum(1 for v in out_sample if v < chosen)
        tied = sum(1 for v in out_sample if v == chosen) - 1
        ranks.append((below + tied / 2) / (strategies - 1))
    return sum(1 for r in ranks if r < 0.5) / len(ranks)


def _pre_benjamini_hochberg(p_values: Sequence[float], *, fdr: float = 0.05) -> list[bool]:
    if not p_values:
        return []
    if any(not isfinite(p) or p < 0 or p > 1 for p in p_values):
        raise MetricError("every p-value must be a finite number in [0, 1]")
    m = len(p_values)
    ordered = sorted(range(m), key=lambda i: p_values[i])
    cutoff = -1
    for rank, index in enumerate(ordered, start=1):
        if p_values[index] <= rank / m * fdr:
            cutoff = rank
    keep = {ordered[i] for i in range(cutoff)} if cutoff > 0 else set()
    return [i in keep for i in range(m)]


def _pre_bonferroni(p_values: Sequence[float], *, alpha: float = 0.05) -> list[bool]:
    if any(not isfinite(p) or p < 0 or p > 1 for p in p_values):
        raise MetricError("every p-value must be a finite number in [0, 1]")
    threshold = alpha / len(p_values) if p_values else alpha
    return [p <= threshold for p in p_values]


def _pre_desk_benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p_values[i] * m / rank)
        adjusted[i] = min(1.0, running)
    return adjusted


# =============================================================================================
# Outcomes and grading — one vocabulary for every arm.
# =============================================================================================

GRADES = ("correct_value", "correct_refusal", "over_refusal", "untyped_refusal", "silent_wrong")
"""``silent_wrong`` is the cardinal sin: a NaN, an out-of-range value, a value on an input that
has no honest answer, or a value that disagrees with the high-precision reference. An
``untyped_refusal`` is loud but raises something other than the arm's own documented refusal
type, so a caller written against that type (``except MetricError``) crashes instead of
recording the refusal. ``over_refusal`` refuses an input that had a well-defined answer."""


@dataclass(frozen=True)
class Outcome:
    kind: str  # "value" | "refused" | "raised_untyped" | "silent_nonfinite" | "silent_out_of_range"
    value: float | None = None
    detail: str | None = None


def _run_probability(
    call: Callable[[], float], typed: tuple[type[BaseException], ...]
) -> Outcome:
    """Call one arm; classify what came back. numpy's RuntimeWarnings are part of the behaviour
    under test, not noise — they are silenced here only so they do not print, and the trapped arms
    re-enable them as errors inside their own call."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = call()
    except typed as exc:
        return Outcome("refused", None, f"{type(exc).__name__}: {exc}"[:160])
    except Exception as exc:
        return Outcome("raised_untyped", None, f"{type(exc).__name__}: {exc}"[:160])
    value = float(raw)
    if not math.isfinite(value):
        return Outcome("silent_nonfinite", None, repr(value))
    if not 0.0 <= value <= 1.0:
        return Outcome("silent_out_of_range", value, None)
    return Outcome("value", value, None)


def grade(expect: str, reference: float | None, outcome: Outcome) -> str:
    """``expect`` is ``"value"``, ``"refuse"`` or ``"value_or_refuse"`` (an input with a real
    answer that double precision cannot reach without overflow: either the right value or a
    refusal is acceptable, a wrong value is not)."""
    if outcome.kind in ("silent_nonfinite", "silent_out_of_range"):
        return "silent_wrong"
    if outcome.kind == "raised_untyped":
        return "untyped_refusal"
    if outcome.kind == "refused":
        return "over_refusal" if expect == "value" else "correct_refusal"
    if expect == "refuse" or reference is None or outcome.value is None:
        return "silent_wrong"
    ok = math.isclose(outcome.value, reference, rel_tol=1e-7, abs_tol=1e-9)
    return "correct_value" if ok else "silent_wrong"


def admitted(outcome: Outcome, *, negated: bool) -> bool:
    """Would this output pass a DSR gate at ``GATE_THRESHOLD``?

    ``negated`` is the shape ``not (dsr < t)`` — a reject-filter written the natural way round —
    which admits NaN. The positive shape ``dsr > t`` rejects NaN but still admits a wrong finite
    value. A refusal passes neither.
    """
    if outcome.kind in ("refused", "raised_untyped"):
        return False
    value = math.nan if outcome.value is None else outcome.value
    return (not value < GATE_THRESHOLD) if negated else value > GATE_THRESHOLD


# =============================================================================================
# The high-precision reference (mpmath, 50 digits) — independent of every arm.
# =============================================================================================


def reference_dsr(case: DsrCase) -> float | None:
    """Bailey & Lopez de Prado DSR at 50 significant digits, or ``None`` where it is undefined.

    Undefined means: a non-finite argument, ``n < 2``, ``trials < 1``, a negative variance, or a
    non-positive higher-moment radicand. **One stated choice:** for a single trial the expected
    maximum of one draw is its mean, 0, which is exact; the extreme-value approximation both
    implementations share is ``-inf`` there (``Phi^-1(0)``) and is not used. mpmath is present in
    the dev environment through sympy (torch's dependency); it is imported lazily.
    """
    import mpmath  # type: ignore[import-untyped]

    values = (case.observed, case.variance_of_trials, case.skew, case.kurtosis)
    if not all(math.isfinite(v) for v in values):
        return None
    if case.n < 2 or case.trials < 1 or case.variance_of_trials < 0:
        return None
    with mpmath.workdps(50):
        sr = mpmath.mpf(case.observed)
        if case.trials == 1:
            sr0 = mpmath.mpf(0)
        else:
            gamma = mpmath.euler
            trials = mpmath.mpf(case.trials)

            def phi_inv(p: Any) -> Any:
                return mpmath.sqrt(2) * mpmath.erfinv(2 * p - 1)

            a = phi_inv(1 - 1 / trials)
            b = phi_inv(1 - 1 / (trials * mpmath.e))
            sr0 = mpmath.sqrt(mpmath.mpf(case.variance_of_trials)) * ((1 - gamma) * a + gamma * b)
        radicand = 1 - mpmath.mpf(case.skew) * sr + (mpmath.mpf(case.kurtosis) - 1) / 4 * sr**2
        if radicand <= 0:
            return None
        z = (sr - sr0) * mpmath.sqrt(case.n - 1) / mpmath.sqrt(radicand)
        # Beyond |z| = 40 the normal tail is below 1e-349, which is 0.0 or 1.0 in a double;
        # clamped here because mpmath's erfc series check itself overflows on |z| ~ 1e150.
        if abs(z) > 40:
            return 1.0 if z > 0 else 0.0
        return float(mpmath.ncdf(z))


# =============================================================================================
# The DSR arms.
# =============================================================================================

_Finite = Annotated[float, Field(allow_inf_nan=False)]
_NonNegativeFinite = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
_Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
_AtLeastTwo = Annotated[int, Field(ge=2)]
_AtLeastOne = Annotated[int, Field(ge=1)]


def _vectorbt_value(symbols: Any, case: DsrCase) -> float:
    """vectorbt's real, vendored ``deflated_sharpe_ratio`` — no paraphrase."""
    result = symbols.deflated_sharpe_ratio(
        est_sharpe=np.array([case.observed]),
        var_sharpe=case.variance_of_trials,
        nb_trials=case.trials,
        backtest_horizon=case.n,
        skew=np.array([case.skew]),
        kurtosis=np.array([case.kurtosis]),
    )
    return float(result[0])


def pydantic_contract(symbols: Any) -> Callable[..., float]:
    """vectorbt's real DSR behind the contract a pydantic user writes for this signature."""

    @validate_call(validate_return=True)
    def gated(
        observed: _Finite, n: _AtLeastTwo, trials: _AtLeastOne,
        variance_of_trials: _NonNegativeFinite, skew: _Finite, kurtosis: _Finite,
    ) -> _Probability:
        return _vectorbt_value(symbols, DsrCase(
            "pydantic", observed=observed, n=n, trials=trials,
            variance_of_trials=variance_of_trials, skew=skew, kurtosis=kurtosis,
        ))

    return gated


def _trapped(call: Callable[[], float]) -> float:
    """IEEE-754 trapping: numpy, scipy.special and Python warnings all promoted to exceptions.

    Configured the way a careful user would, not the bluntest way: underflow is left alone in both
    numpy and scipy.special. With ``all="raise"`` the first run of this arm refused 61 perfectly
    valid inputs because ``erfc`` of a large argument underflows on its way to a correct DSR near
    0 — a strawman, since numpy's own default ignores underflow for exactly that reason.
    """
    import scipy.special

    # scipy-stubs types the underflow option as Literal["ignored", ...]; scipy itself accepts and
    # reports "ignore" (``scipy.special.geterr()`` returns it), so the stub is what is wrong here.
    special = scipy.special.errstate(all="raise", underflow="ignore")  # type: ignore[arg-type]
    with np.errstate(divide="raise", over="raise", invalid="raise", under="ignore"), special, \
            warnings.catch_warnings():
        warnings.simplefilter("error")
        return call()


def _trap_types() -> tuple[type[BaseException], ...]:
    import scipy.special

    return (FloatingPointError, RuntimeWarning, scipy.special.SpecialFunctionError)


def dsr_arms(symbols: Any) -> dict[str, tuple[Callable[[DsrCase], float],
                                                tuple[type[BaseException], ...]]]:
    """Every arm as ``case -> float`` plus the exception type(s) that are its documented refusal."""
    gated = pydantic_contract(symbols)

    def argus_current(c: DsrCase) -> float:
        return argus_metrics.deflated_sharpe(
            c.observed, n=c.n, trials=c.trials, variance_of_trials=c.variance_of_trials,
            skew=c.skew, kurtosis=c.kurtosis,
        )

    def argus_pre(c: DsrCase) -> float:
        return _pre_deflated_sharpe(
            c.observed, n=c.n, trials=c.trials, variance_of_trials=c.variance_of_trials,
            skew=c.skew, kurtosis=c.kurtosis,
        )

    def pydantic_arm(c: DsrCase) -> float:
        return gated(c.observed, c.n, c.trials, c.variance_of_trials, c.skew, c.kurtosis)

    return {
        "argus_current": (argus_current, (MetricError,)),
        "argus_pre_adaptation": (argus_pre, (MetricError,)),
        "vectorbt_bare": (lambda c: _vectorbt_value(symbols, c), ()),
        "numpy_scipy_trap_around_vectorbt": (
            lambda c: _trapped(lambda: _vectorbt_value(symbols, c)), _trap_types(),
        ),
        "pydantic_contract_around_vectorbt": (pydantic_arm, (ValidationError,)),
    }


# =============================================================================================
# The inputs. Tier 1 is the exact input set the standing row is already measured on.
# =============================================================================================


@dataclass(frozen=True)
class GateCase:
    tier: str
    case: DsrCase
    expect: str
    reference: float | None


def _with(case: DsrCase, name: str, **changes: Any) -> DsrCase:
    return dataclasses.replace(case, name=name, **changes)


def _gate_case(tier: str, case: DsrCase, *, overflow: bool = False) -> GateCase:
    reference = reference_dsr(case)
    expect = "refuse" if reference is None else ("value_or_refuse" if overflow else "value")
    return GateCase(tier, case, expect, reference)


def standing_input_cases() -> list[GateCase]:
    """The 5 designed + 1,215 swept cases `eval/dsr_comparison.py` already measures the row on."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # designed_cases() runs np.var(ddof=1) on one value
        cases = (*designed_cases(), *swept_cases())
    return [_gate_case("standing_input", c) for c in cases]


def _finite_bases() -> list[DsrCase]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cases = designed_cases()
    return [c for c in cases if math.isfinite(c.variance_of_trials)]


def poisoned_cases() -> list[GateCase]:
    """Every float argument of every finite designed case replaced by NaN, +inf and -inf; the two
    integer arguments pushed out of their domain; a negative variance. No honest answer exists for
    any of these, so the only correct outcome is a refusal."""
    out: list[GateCase] = []
    for base in _finite_bases():
        for argument in ("observed", "variance_of_trials", "skew", "kurtosis"):
            for poison in POISONS:
                label = f"{base.name}:{argument}={poison!r}"
                out.append(_gate_case("poisoned", _with(base, label, **{argument: poison})))
        for n in (1, 0, -1):
            out.append(_gate_case("poisoned", _with(base, f"{base.name}:n={n}", n=n)))
        for trials in (0, -1):
            out.append(_gate_case(
                "poisoned", _with(base, f"{base.name}:trials={trials}", trials=trials)))
        out.append(_gate_case(
            "poisoned", _with(base, f"{base.name}:variance=-0.1", variance_of_trials=-0.1)))
    return out


def edge_valid_cases() -> list[GateCase]:
    """Valid inputs outside the swept grid: one trial with a finite variance (the case a first
    backtest produces when a caller passes the variance explicitly), and a zero variance across
    many trials (every variant scored the same Sharpe)."""
    out: list[GateCase] = []
    for observed in (-2.0, 0.0, 0.05, 1.2):
        for variance in (0.0, 0.3):
            out.append(_gate_case("edge_valid", DsrCase(
                f"one_trial:observed={observed}:variance={variance}", observed=observed, n=500,
                trials=1, variance_of_trials=variance)))
    for observed in (0.05, 1.2):
        out.append(_gate_case("edge_valid", DsrCase(
            f"zero_variance:observed={observed}", observed=observed, n=500, trials=10,
            variance_of_trials=0.0)))
    return out


def overflow_cases() -> list[GateCase]:
    """Finite arguments whose arithmetic leaves double precision. A real answer exists (the
    reference computes it); a refusal is acceptable, a wrong value is not."""
    specs = (
        ("observed=1e200", {"observed": 1e200}),
        ("observed=-1e200", {"observed": -1e200}),
        ("kurtosis=1e300", {"kurtosis": 1e300}),
        ("skew=-1e300", {"skew": -1e300}),
        ("trials=1e17", {"trials": 10**17}),
        ("variance=1e308", {"variance_of_trials": 1e308}),
    )
    base = _finite_bases()[0]
    return [_gate_case("overflow", _with(base, label, **change), overflow=True)
            for label, change in specs]


# =============================================================================================
# Real-life producer inputs: real Bitget candles, a real strategy family, the general-purpose stack.
# =============================================================================================

FAMILY: tuple[tuple[str, int, int], ...] = (
    ("momentum_1h", 1, 1), ("momentum_6h", 6, 1), ("momentum_24h", 24, 1),
    ("reversion_1h", 1, -1), ("reversion_6h", 6, -1), ("reversion_24h", 24, -1),
)
SHOCK = ("shock_follow_5pct", 0.05)
"""A filter that follows an hourly move beyond 5%. On most of these symbols it never fires in the
window, so its return stream is all zeros — the ordinary way a strategy sweep produces a variant
with no Sharpe, and the way a NaN enters a gate from real data rather than from a test."""

_WARMUP = 24


def load_closes(path: Path = CANDLES) -> dict[str, np.ndarray]:
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GeneralOverfitGatesError(
            f"could not read the real candle file {path}: {exc}"
        ) from exc
    return {sym: np.array([float(p) for _, p in rows]) for sym, rows in blob["series"].items()}


def family_net_returns(closes: np.ndarray) -> dict[str, np.ndarray]:
    """Net hourly returns for every variant, over one common window so ``n`` is shared."""
    r = np.diff(closes) / closes[:-1]
    bars = range(_WARMUP, len(r))
    out: dict[str, np.ndarray] = {}
    def trend(lookback: int, sign: int) -> Callable[[int], float]:
        return lambda t: sign * float(np.sign(r[t - lookback:t].sum()))

    shock_label, shock_size = SHOCK

    def shock(t: int) -> float:
        return float(np.sign(r[t - 1])) if abs(r[t - 1]) > shock_size else 0.0

    variants: list[tuple[str, Callable[[int], float]]] = [
        (label, trend(lookback, sign)) for label, lookback, sign in FAMILY
    ]
    variants.append((shock_label, shock))
    for label, position_at in variants:
        position = np.array([position_at(t) for t in bars])
        previous = np.concatenate(([0.0], position[:-1]))
        cost = (ROUND_TRIP_FEE / 2) * np.abs(position - previous)
        out[label] = position * r[_WARMUP:] - cost
    return out


@dataclass(frozen=True)
class ProducedInputs:
    """What the numpy/scipy stack hands a DSR gate — vectorbt's accessor path
    (``vectorbt/returns/accessors.py:594-611``) with Pearson kurtosis, which is what the formula
    both implementations share expects."""

    sharpe: float
    skew: float
    kurtosis: float


def produce(net: np.ndarray) -> ProducedInputs:
    import scipy.stats

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sharpe = float(np.mean(net) / np.std(net, ddof=1))
        skew = float(scipy.stats.skew(net))
        kurtosis = float(scipy.stats.kurtosis(net, fisher=False))
    return ProducedInputs(sharpe, skew, kurtosis)


def _nan_var(values: Sequence[float]) -> float:
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.var(np.array(values, dtype=float), ddof=1))


@dataclass(frozen=True)
class RealScenario:
    """One real gate decision: the inputs as produced, plus the raw returns for the arm that traps
    the whole pipeline rather than the gate alone."""

    gate_case: GateCase
    nets: tuple[np.ndarray, ...]
    selected: int


def real_scenarios(closes: dict[str, np.ndarray] | None = None) -> list[RealScenario]:
    """Per symbol: (a) the family gate — the best variant by Sharpe (NaN skipped, as pandas'
    ``idxmax`` does), deflated by the family's trial count and ``np.var(sharpes, ddof=1)``; and
    (b) the single-strategy gate a first-time user runs — the best variant alone, ``trials=1``,
    variance from the same numpy call on one value."""
    series = load_closes() if closes is None else closes
    out: list[RealScenario] = []
    for symbol in sorted(series):
        nets = family_net_returns(series[symbol])
        labels = list(nets)
        produced = [produce(nets[k]) for k in labels]
        sharpes = [p.sharpe for p in produced]
        finite = [i for i, s in enumerate(sharpes) if math.isfinite(s)]
        best = max(finite, key=lambda i: sharpes[i])
        n = len(nets[labels[best]])
        family = DsrCase(
            f"{symbol}:family_of_{len(labels)}:best={labels[best]}", observed=sharpes[best], n=n,
            trials=len(labels), variance_of_trials=_nan_var(sharpes),
            skew=produced[best].skew, kurtosis=produced[best].kurtosis,
        )
        single = DsrCase(
            f"{symbol}:single_trial:{labels[best]}", observed=sharpes[best], n=n, trials=1,
            variance_of_trials=_nan_var([sharpes[best]]),
            skew=produced[best].skew, kurtosis=produced[best].kurtosis,
        )
        all_nets = tuple(nets[k] for k in labels)
        out.append(RealScenario(_gate_case("real_producer", family), all_nets, best))
        out.append(RealScenario(_gate_case("real_producer", single), (nets[labels[best]],), 0))
    return out


def trapped_pipeline(symbols: Any, scenario: RealScenario) -> float:
    """numpy/scipy trapping around the WHOLE pipeline: Sharpes, moments, variance, then the gate."""
    import scipy.stats

    def run() -> float:
        sharpes = [float(np.mean(x) / np.std(x, ddof=1)) for x in scenario.nets]
        chosen = scenario.nets[scenario.selected]
        case = DsrCase(
            "trapped", observed=sharpes[scenario.selected], n=len(chosen),
            trials=len(scenario.nets), variance_of_trials=float(np.var(sharpes, ddof=1)),
            skew=float(scipy.stats.skew(chosen)),
            kurtosis=float(scipy.stats.kurtosis(chosen, fisher=False)),
        )
        return _vectorbt_value(symbols, case)

    return _trapped(run)


def kurtosis_convention_gap(scenarios: Sequence[RealScenario], symbols: Any) -> dict[str, Any]:
    """vectorbt's accessor passes ``scipy.stats.kurtosis`` with its default ``fisher=True`` —
    *excess* kurtosis — into a formula whose ``(kurtosis - 1) / 4`` term expects Pearson kurtosis
    (``accessors.py:610`` into ``metrics.py``'s ``deflated_sharpe_ratio``). Measured on the real
    family cases: how far the DSR moves between the two conventions."""
    gaps: list[float] = []
    for s in scenarios:
        c = s.gate_case.case
        if s.gate_case.tier != "real_producer" or c.trials == 1 or s.gate_case.reference is None:
            continue
        pearson = _vectorbt_value(symbols, c)
        excess = _vectorbt_value(symbols, _with(c, c.name, kurtosis=c.kurtosis - 3.0))
        gaps.append(abs(pearson - excess))
    return {
        "family_cases_measured": len(gaps),
        "max_abs_dsr_gap": max(gaps) if gaps else None,
        "finding": (
            "vectorbt's accessor feeds excess kurtosis (scipy's fisher=True default) into a "
            "Pearson-kurtosis formula. On per-period hourly Sharpes the SR**2 term it scales is "
            "tiny, so the measured DSR gap on these real cases is small; it grows with the square "
            "of the per-period Sharpe. Recorded as a vectorbt defect, not an ARGUS advantage: "
            "ARGUS's gate takes kurtosis from its caller and cannot tell which convention it got."
        ),
    }


# =============================================================================================
# Running the DSR head-to-head.
# =============================================================================================


@dataclass
class ArmScore:
    grades: dict[str, int] = field(default_factory=lambda: dict.fromkeys(GRADES, 0))
    by_tier: dict[str, dict[str, int]] = field(default_factory=dict)
    false_admissions_negated_gate: int = 0
    false_admissions_positive_gate: int = 0
    silent_examples: list[str] = field(default_factory=list)
    untyped_examples: list[str] = field(default_factory=list)

    def add(self, gc: GateCase, outcome: Outcome) -> None:
        g = grade(gc.expect, gc.reference, outcome)
        self.grades[g] += 1
        tier = self.by_tier.setdefault(gc.tier, dict.fromkeys(GRADES, 0))
        tier[g] += 1
        should_pass = gc.reference is not None and gc.reference > GATE_THRESHOLD
        should_pass_negated = gc.reference is not None and not gc.reference < GATE_THRESHOLD
        if admitted(outcome, negated=True) and not should_pass_negated:
            self.false_admissions_negated_gate += 1
        if admitted(outcome, negated=False) and not should_pass:
            self.false_admissions_positive_gate += 1
        shown = outcome.value if outcome.kind == "value" else (outcome.detail or outcome.kind)
        if g == "silent_wrong" and len(self.silent_examples) < 6:
            self.silent_examples.append(
                f"{gc.case.name} -> {shown} (reference {gc.reference})")
        if g == "untyped_refusal" and len(self.untyped_examples) < 4:
            self.untyped_examples.append(f"{gc.case.name} -> {outcome.detail}")

    @property
    def silent(self) -> int:
        return self.grades["silent_wrong"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "grades": dict(self.grades),
            "silent_wrong": self.silent,
            "by_tier": self.by_tier,
            "false_admissions_negated_gate": self.false_admissions_negated_gate,
            "false_admissions_positive_gate": self.false_admissions_positive_gate,
            "silent_examples": self.silent_examples,
            "untyped_examples": self.untyped_examples,
        }


def run_dsr_head_to_head(symbols: Any, scenarios: Sequence[RealScenario]) -> dict[str, Any]:
    arms = dsr_arms(symbols)
    cases = [
        *standing_input_cases(), *poisoned_cases(), *edge_valid_cases(), *overflow_cases(),
        *(s.gate_case for s in scenarios),
    ]
    scores = {name: ArmScore() for name in arms}
    for gc in cases:
        for name, (fn, typed) in arms.items():
            scores[name].add(gc, _run_probability(functools.partial(fn, gc.case), typed))

    pipeline = ArmScore()
    for s in scenarios:
        pipeline.add(s.gate_case, _run_probability(
            functools.partial(trapped_pipeline, symbols, s), _trap_types()))

    tiers: dict[str, int] = {}
    expectations: dict[str, dict[str, int]] = {}
    for gc in cases:
        tiers[gc.tier] = tiers.get(gc.tier, 0) + 1
        bucket = expectations.setdefault(gc.tier, {})
        bucket[gc.expect] = bucket.get(gc.expect, 0) + 1
    return {
        "cases": len(cases),
        "tiers": tiers,
        "expectations": expectations,
        "gate_threshold": GATE_THRESHOLD,
        "arms": {name: score.as_dict() for name, score in scores.items()},
        "numpy_scipy_trap_whole_pipeline_real_producer_only": pipeline.as_dict(),
        "real_producer_cases": [
            {
                "case": s.gate_case.case.name,
                "observed": _json_float(s.gate_case.case.observed),
                "variance_of_trials": _json_float(s.gate_case.case.variance_of_trials),
                "skew": _json_float(s.gate_case.case.skew),
                "kurtosis": _json_float(s.gate_case.case.kurtosis),
                "n": s.gate_case.case.n,
                "trials": s.gate_case.case.trials,
                "expect": s.gate_case.expect,
                "reference": s.gate_case.reference,
            }
            for s in scenarios
        ],
    }


def _json_float(x: float) -> float | str:
    """Keep a non-finite producer value visible in a strict-JSON artefact (``null`` would hide
    which poison it was)."""
    return x if math.isfinite(x) else repr(x)


# =============================================================================================
# Multiple-testing head-to-head.
# =============================================================================================


def bh_reference(p_values: Sequence[float], q: float) -> list[bool]:
    """Exact Benjamini-Hochberg (1995) in rational arithmetic: reject the k smallest, k the
    largest rank with p_(k) <= k/m * q. Independent of every arm."""
    m = len(p_values)
    exact = [Fraction(p) for p in p_values]
    order = sorted(range(m), key=lambda i: exact[i])
    k = 0
    for rank, i in enumerate(order, start=1):
        if exact[i] <= Fraction(rank, m) * Fraction(q):
            k = rank
    keep = set(order[:k])
    return [i in keep for i in range(m)]


def bonferroni_reference(p_values: Sequence[float], q: float) -> list[bool]:
    m = len(p_values)
    return [Fraction(p) <= Fraction(q) / m for p in p_values]


def mt_arms() -> dict[str, tuple[str, Callable[[Sequence[float], float], list[bool]],
                                  tuple[type[BaseException], ...]]]:
    """``name -> (method, decisions(p, q), refusal types)``. The desk's BH returns q-values and
    its caller keeps a rule when ``q < ALPHA`` (``desk/review.py``'s ``_corrected``); scipy returns
    adjusted p-values and the caller compares with ``<=``. Each arm is run the way its own caller
    uses it."""
    import scipy.stats
    from statsmodels.stats.multitest import multipletests

    def scipy_bh(p: Sequence[float], q: float) -> list[bool]:
        return [bool(a <= q) for a in scipy.stats.false_discovery_control(list(p), method="bh")]

    def sm(method: str) -> Callable[[Sequence[float], float], list[bool]]:
        return lambda p, q: [bool(r) for r in multipletests(list(p), alpha=q, method=method)[0]]

    return {
        "argus_bh_current": (
            "bh", lambda p, q: argus_validation.benjamini_hochberg(p, fdr=q), (MetricError,)),
        "argus_bh_pre_adaptation": (
            "bh", lambda p, q: _pre_benjamini_hochberg(p, fdr=q), (MetricError,)),
        "argus_desk_bh_current": (
            "bh", lambda p, q: [v < q for v in desk_review.benjamini_hochberg(p)],
            (MetricError,)),
        "argus_desk_bh_pre_adaptation": (
            "bh", lambda p, q: [v < q for v in _pre_desk_benjamini_hochberg(p)], (MetricError,)),
        "scipy_false_discovery_control": ("bh", scipy_bh, (ValueError,)),
        "statsmodels_fdr_bh": ("bh", sm("fdr_bh"), (ValueError,)),
        "argus_bonferroni_current": (
            "bonferroni", lambda p, q: argus_validation.bonferroni(p, alpha=q), (MetricError,)),
        "argus_bonferroni_pre_adaptation": (
            "bonferroni", lambda p, q: _pre_bonferroni(p, alpha=q), (MetricError,)),
        "statsmodels_bonferroni": ("bonferroni", sm("bonferroni"), (ValueError,)),
    }


@dataclass(frozen=True)
class MtCase:
    tier: str
    name: str
    p_values: tuple[float, ...]
    q: float
    threshold_applies: bool = True

    @property
    def valid(self) -> bool:
        ps_ok = all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in self.p_values)
        return ps_ok and math.isfinite(self.q) and 0.0 < self.q < 1.0


def real_p_value_cases(closes: dict[str, np.ndarray] | None = None) -> list[MtCase]:
    """One-sided t-test p-values (scipy's ``ttest_1samp``, ``alternative='greater'``) for every
    variant of every real symbol — the general-purpose producer; the never-trading variant's
    p-value is NaN exactly as scipy returns it for a constant series."""
    import scipy.stats

    series = load_closes() if closes is None else closes
    out: list[MtCase] = []
    for symbol in sorted(series):
        nets = family_net_returns(series[symbol])
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ps = tuple(
                float(scipy.stats.ttest_1samp(x, 0.0, alternative="greater").pvalue)
                for x in nets.values()
            )
        out.append(MtCase("real_producer", f"{symbol}:ttest_p_values", ps, 0.05))
        finite = tuple(p for p in ps if math.isfinite(p))
        out.append(MtCase("real_producer_nan_dropped", f"{symbol}:finite_only", finite, 0.05))
    return out


def grid_p_value_cases() -> list[MtCase]:
    grid = (0.001, 0.01, 0.025, 0.04, 0.2, 0.9)
    return [
        MtCase("grid", f"grid:{ps}:q={q}", ps, q)
        for ps in itertools.product(grid, repeat=3) for q in (0.05, 0.10)
    ]


def poisoned_p_value_cases() -> list[MtCase]:
    base = (0.001, 0.02, 0.2)
    out = [
        MtCase("poisoned_p", f"p0={bad!r}", (bad, *base[1:]), 0.05)
        for bad in (*POISONS, -0.1, 1.5)
    ]
    out += [MtCase("poisoned_q", f"q={q!r}", base, q) for q in (*POISONS, -0.1, 2.0)]
    return out


@dataclass
class MtScore:
    correct_decisions: int = 0
    wrong_decisions: int = 0
    correct_refusals: int = 0
    over_refusals: int = 0
    untyped_refusals: int = 0
    silent_on_invalid: int = 0
    permissive_on_invalid: int = 0
    wrong_examples: list[str] = field(default_factory=list)
    silent_examples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "correct_decisions": self.correct_decisions,
            "wrong_decisions": self.wrong_decisions,
            "correct_refusals": self.correct_refusals,
            "over_refusals": self.over_refusals,
            "untyped_refusals": self.untyped_refusals,
            "silent_on_invalid_input": self.silent_on_invalid,
            "permissive_on_invalid_input": self.permissive_on_invalid,
            "wrong_examples": self.wrong_examples,
            "silent_examples": self.silent_examples,
        }


def run_mt_head_to_head(real: Sequence[MtCase] | None = None) -> dict[str, Any]:
    cases = [*(real_p_value_cases() if real is None else real), *grid_p_value_cases(),
             *poisoned_p_value_cases()]
    arms = mt_arms()
    scores = {name: MtScore() for name in arms}
    for case in cases:
        for name, (method, fn, typed) in arms.items():
            if case.tier == "poisoned_q" and name.startswith("argus_desk"):
                continue  # the desk BH takes no threshold argument; nothing to poison
            score = scores[name]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    decisions = fn(case.p_values, case.q)
            except typed:
                if case.valid:
                    score.over_refusals += 1
                else:
                    score.correct_refusals += 1
                continue
            except Exception as exc:
                score.untyped_refusals += 1
                if len(score.silent_examples) < 4:
                    score.silent_examples.append(f"{case.name}: {type(exc).__name__}")
                continue
            if not case.valid:
                score.silent_on_invalid += 1
                if any(decisions):
                    score.permissive_on_invalid += 1
                if len(score.silent_examples) < 4:
                    score.silent_examples.append(f"{case.name} -> {decisions}")
                continue
            reference = (bh_reference if method == "bh" else bonferroni_reference)(
                case.p_values, case.q)
            if decisions == reference:
                score.correct_decisions += 1
            else:
                score.wrong_decisions += 1
                if len(score.wrong_examples) < 4:
                    score.wrong_examples.append(
                        f"{case.name}: got {decisions}, exact reference {reference}")
    tiers: dict[str, int] = {}
    for case in cases:
        tiers[case.tier] = tiers.get(case.tier, 0) + 1
    return {
        "cases": len(cases),
        "tiers": tiers,
        "arms": {name: s.as_dict() for name, s in scores.items()},
    }


# =============================================================================================
# Hypothesis: the general-purpose adversarial search, against the pre and the current gates.
# =============================================================================================


def _prob_ok(x: Any) -> str | None:
    if not isinstance(x, float) or not math.isfinite(x):
        return "silent_nonfinite_output"
    return None if 0.0 <= x <= 1.0 else "silent_out_of_range"


def _finite_ok(x: Any) -> str | None:
    return None if isinstance(x, float) and math.isfinite(x) else "silent_nonfinite_output"


@dataclass(frozen=True)
class SearchTarget:
    name: str
    strategy: SearchStrategy[Any]
    call: Callable[[Any], Any]
    valid: Callable[[Any], bool]
    check: Callable[[Any], str | None]


def _targets(version: str) -> list[SearchTarget]:
    from hypothesis import strategies as st

    pre = version == "pre_adaptation"
    floats = st.floats(allow_nan=True, allow_infinity=True, width=64)
    ints = st.integers(min_value=-3, max_value=10_000)
    returns = st.lists(floats, min_size=10, max_size=30)
    fin = math.isfinite

    dsr = _pre_deflated_sharpe if pre else argus_metrics.deflated_sharpe
    psr = _pre_probabilistic_sharpe if pre else argus_metrics.probabilistic_sharpe
    sharpe = _pre_sharpe if pre else argus_metrics.sharpe
    sortino = _pre_sortino if pre else argus_metrics.sortino
    decay = _pre_out_of_sample_decay if pre else argus_metrics.out_of_sample_decay
    moments = _pre_moments if pre else argus_validation.moments
    mintrl = _pre_min_track_record_length if pre else argus_validation.min_track_record_length
    bh = _pre_benjamini_hochberg if pre else argus_validation.benjamini_hochberg
    bonf = _pre_bonferroni if pre else argus_validation.bonferroni
    desk_bh = _pre_desk_benjamini_hochberg if pre else desk_review.benjamini_hochberg

    def pbo(matrix: list[list[float]]) -> float:
        if pre:
            return _pre_probability_of_overfitting(matrix, groups=2)
        return argus_validation.probability_of_overfitting(matrix, groups=2).pbo

    unit = st.floats(min_value=0.0, max_value=1.0)
    return [
        SearchTarget(
            "deflated_sharpe", st.tuples(floats, ints, ints, floats, floats, floats),
            lambda a: dsr(a[0], n=a[1], trials=a[2], variance_of_trials=a[3], skew=a[4],
                          kurtosis=a[5]),
            lambda a: all(fin(a[i]) for i in (0, 3, 4, 5)), _prob_ok),
        SearchTarget(
            "probabilistic_sharpe", st.tuples(floats, floats, ints, floats, floats),
            lambda a: psr(a[0], benchmark=a[1], n=a[2], skew=a[3], kurtosis=a[4]),
            lambda a: all(fin(a[i]) for i in (0, 1, 3, 4)), _prob_ok),
        SearchTarget(
            "sharpe", returns, lambda xs: sharpe(xs, periods_per_year=HOURLY_PER_YEAR),
            lambda xs: all(fin(x) for x in xs), _finite_ok),
        SearchTarget(
            "sortino", returns, lambda xs: sortino(xs, periods_per_year=HOURLY_PER_YEAR),
            lambda xs: all(fin(x) for x in xs), _finite_ok),
        SearchTarget(
            "out_of_sample_decay", st.tuples(floats, floats), lambda a: decay(a[0], a[1]),
            lambda a: fin(a[0]) and fin(a[1]),
            lambda d: _finite_ok(float(d["retention_ratio"]))),
        SearchTarget(
            "moments", returns, lambda xs: moments(xs), lambda xs: all(fin(x) for x in xs),
            lambda out: None if all(fin(v) for v in out) else "silent_nonfinite_output"),
        SearchTarget(
            "min_track_record_length", st.tuples(returns, floats, floats),
            lambda a: mintrl(a[0], benchmark_sharpe=a[1], confidence=a[2]),
            lambda a: all(fin(x) for x in a[0]) and fin(a[1]) and 0.0 < a[2] < 1.0,
            lambda x: None if isinstance(x, float) and fin(x) and x > 0 else
            "silent_nonfinite_output"),
        SearchTarget(
            "probability_of_overfitting",
            st.lists(st.lists(floats, min_size=3, max_size=3), min_size=20, max_size=24),
            pbo, lambda m: all(fin(x) for row in m for x in row), _prob_ok),
        SearchTarget(
            "benjamini_hochberg", st.tuples(st.lists(unit, min_size=1, max_size=8), floats),
            lambda a: bh(a[0], fdr=a[1]), lambda a: fin(a[1]) and 0.0 < a[1] < 1.0,
            lambda out: None),
        SearchTarget(
            "bonferroni", st.tuples(st.lists(unit, min_size=1, max_size=8), floats),
            lambda a: bonf(a[0], alpha=a[1]), lambda a: fin(a[1]) and 0.0 < a[1] < 1.0,
            lambda out: None),
        SearchTarget(
            "desk_benjamini_hochberg", st.lists(floats, min_size=1, max_size=8),
            lambda ps: desk_bh(ps), lambda ps: all(fin(p) and 0.0 <= p <= 1.0 for p in ps),
            lambda out: None if all(fin(v) and 0.0 <= v <= 1.0 for v in out)
            else "silent_nonfinite_output"),
    ]


def _make_probe(t: SearchTarget, sink: dict[str, str]) -> Callable[[Any], None]:
    """One property check, recording the first input seen for each failure class. Built by a
    factory because ``@given`` refuses a test function with default arguments."""

    def probe(args: Any) -> None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = t.call(args)
        except MetricError:
            return
        except Exception as exc:
            sink.setdefault(f"raised_untyped:{type(exc).__name__}", repr(args)[:240])
            return
        bad = t.check(result)
        if bad is None and not t.valid(args):
            bad = "silent_accepted_invalid_input"
        if bad is not None:
            sink.setdefault(bad, repr(args)[:240])

    return probe


def hypothesis_search(max_examples: int = 300,
                      only: Sequence[str] | None = None) -> dict[str, Any]:
    """Run the same derandomized search against the pre-adaptation and the current gates.

    Property, per function: it raises ``MetricError``, or it returns an in-range finite value
    from inputs that are all finite and in-domain. Anything else is a failure class — a
    non-finite result, an out-of-range result, a finite result accepted from an invalid input, or
    an exception other than ``MetricError``. ``derandomize=True`` with no example database makes
    the draw sequence a function of the code, the Hypothesis version and one more input that is
    easy to miss: about one draw in twenty is a literal Hypothesis harvests from the project
    modules already imported (``hypothesis/internal/conjecture/providers.py``,
    ``_get_local_constants`` and ``_maybe_draw_constant(p=0.05)``; test files are skipped). So
    the same entry point draws the same inputs, but a caller that has imported more of ARGUS
    first can draw different ones. Measured: the failure classes per function were identical
    under ``python -m`` and under the test suite, while the first input witnessed for a class
    differed — which is why the test pins the classes and not the witnesses. Finding nothing in
    ``max_examples`` draws is evidence, not proof.
    """
    import hypothesis
    from hypothesis import HealthCheck, given, settings

    out: dict[str, Any] = {"hypothesis_version": hypothesis.__version__,
                           "max_examples_per_function": max_examples, "versions": {}}
    for version in ("pre_adaptation", "current"):
        found: dict[str, dict[str, str]] = {}
        for target in _targets(version):
            if only is not None and target.name not in only:
                continue
            classes: dict[str, str] = {}
            probe = _make_probe(target, classes)
            searched = given(target.strategy)(probe)
            run = settings(
                max_examples=max_examples, derandomize=True, database=None, deadline=None,
                suppress_health_check=list(HealthCheck),
            )(searched)
            run()
            found[target.name] = classes
        out["versions"][version] = {
            "functions_searched": len(found),
            "functions_with_failures": sorted(k for k, v in found.items() if v),
            "failure_classes": sum(len(v) for v in found.values()),
            "classes": found,
        }
    return out


HAND_DESIGNED_ADVERSARIAL = (
    "deflated_sharpe(variance_of_trials=NaN) — the single-trial np.var(ddof=1) case",
    "probabilistic_sharpe(observed=NaN) — the ablation's tripped case",
)
"""What `eval/dsr_comparison.py` hand-picked as its adversarial set: two inputs, two functions."""


# =============================================================================================
# Ablation — each adapted guard group, tripped on the pre code and on the current code.
# =============================================================================================


def _kind(call: Callable[[], Any]) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = call()
    except MetricError:
        return "refused"
    except Exception as exc:
        return f"raised_untyped:{type(exc).__name__}"
    if isinstance(out, float) and not math.isfinite(out):
        return "silent_nonfinite"
    if isinstance(out, (list, tuple)) and any(
            isinstance(v, float) and not math.isfinite(v) for v in out):
        return "silent_nonfinite"
    return f"returned:{out!r}"[:80]


def ablation() -> list[dict[str, Any]]:
    nan, inf = math.nan, math.inf
    base = {"n": 500, "trials": 10, "variance_of_trials": 0.3}
    series = [0.01, -0.02, 0.015, 0.003, -0.004, 0.02, -0.01, 0.007, 0.012, -0.006, 0.009]
    with_nan = [*series[:5], nan, *series[6:]]
    matrix = [[0.01 * ((i * 7) % 5 - 2), 0.005 * (i % 3 - 1), 0.002 * (i % 4 - 1.5)]
              for i in range(20)]
    poisoned_matrix = [[nan if i == 3 and j == 0 else v for j, v in enumerate(row)]
                       for i, row in enumerate(matrix)]
    specs: list[tuple[str, str, Callable[[], Any], Callable[[], Any]]] = [
        ("finite-argument guard (pydantic allow_inf_nan=False)",
         "deflated_sharpe(skew=NaN)",
         lambda: _pre_deflated_sharpe(1.2, skew=nan, **base),  # type: ignore[arg-type]
         lambda: argus_metrics.deflated_sharpe(1.2, skew=nan, **base)),  # type: ignore[arg-type]
        ("finite-argument guard",
         "deflated_sharpe(kurtosis=+inf) — pre returned a finite 0.5",
         lambda: _pre_deflated_sharpe(1.2, kurtosis=inf, **base),  # type: ignore[arg-type]
         lambda: argus_metrics.deflated_sharpe(1.2, kurtosis=inf, **base)),  # type: ignore[arg-type]
        ("finite-argument guard",
         "probabilistic_sharpe(observed=+inf)",
         lambda: _pre_probabilistic_sharpe(inf, benchmark=0.0, n=500),
         lambda: argus_metrics.probabilistic_sharpe(inf, benchmark=0.0, n=500)),
        ("arithmetic failures normalised to MetricError",
         "deflated_sharpe(observed=1e200) — SR**2 overflows",
         lambda: _pre_deflated_sharpe(1e200, **base),  # type: ignore[arg-type]
         lambda: argus_metrics.deflated_sharpe(1e200, **base)),  # type: ignore[arg-type]
        ("arithmetic failures normalised to MetricError",
         "deflated_sharpe(kurtosis=-5, observed=3) — negative radicand",
         lambda: _pre_deflated_sharpe(3.0, kurtosis=-5.0, **base),  # type: ignore[arg-type]
         lambda: argus_metrics.deflated_sharpe(3.0, kurtosis=-5.0, **base)),  # type: ignore[arg-type]
        ("arithmetic failures normalised to MetricError",
         "deflated_sharpe(trials=10**17) — 1 - 1/trials rounds to 1.0",
         lambda: _pre_deflated_sharpe(1.2, n=500, trials=10**17, variance_of_trials=0.3),
         lambda: argus_metrics.deflated_sharpe(1.2, n=500, trials=10**17,
                                               variance_of_trials=0.3)),
        ("finite-series guard", "sharpe(returns with one NaN)",
         lambda: _pre_sharpe(with_nan, periods_per_year=HOURLY_PER_YEAR),
         lambda: argus_metrics.sharpe(with_nan, periods_per_year=HOURLY_PER_YEAR)),
        ("finite-series guard", "moments(returns with one NaN)",
         lambda: _pre_moments(with_nan), lambda: argus_validation.moments(with_nan)),
        ("finite-series guard", "min_track_record_length(returns with one NaN)",
         lambda: _pre_min_track_record_length(with_nan),
         lambda: argus_validation.min_track_record_length(with_nan)),
        ("finite-matrix guard", "probability_of_overfitting(one NaN return)",
         lambda: _pre_probability_of_overfitting(poisoned_matrix, groups=2),
         lambda: argus_validation.probability_of_overfitting(poisoned_matrix, groups=2).pbo),
        ("threshold guard (scipy's positive-form range check)",
         "benjamini_hochberg(fdr=+inf)",
         lambda: _pre_benjamini_hochberg([0.5, 0.9], fdr=inf),
         lambda: argus_validation.benjamini_hochberg([0.5, 0.9], fdr=inf)),
        ("threshold guard", "bonferroni(alpha=2.0)",
         lambda: _pre_bonferroni([0.5, 0.9], alpha=2.0),
         lambda: argus_validation.bonferroni([0.5, 0.9], alpha=2.0)),
        ("p-value contract on the desk's BH (scipy false_discovery_control's own check)",
         "desk benjamini_hochberg([NaN, 0.001])",
         lambda: _pre_desk_benjamini_hochberg([nan, 0.001]),
         lambda: desk_review.benjamini_hochberg([nan, 0.001])),
        ("finite-argument guard", "out_of_sample_decay(in_sample=NaN)",
         lambda: _pre_out_of_sample_decay(nan, 0.5),
         lambda: argus_metrics.out_of_sample_decay(nan, 0.5)),
    ]
    rows: list[dict[str, Any]] = []
    for group, case, before, after in specs:
        pre_kind, cur_kind = _kind(before), _kind(after)
        rows.append({
            "guard": group, "case": case, "pre_adaptation": pre_kind, "current": cur_kind,
            "load_bearing": cur_kind == "refused" and pre_kind != "refused",
        })
    rows.append({
        "guard": "return post-condition (pydantic validate_return=True)",
        "case": "no input found that reaches it: with every argument finite and the radicand "
                "checked, NormalDist().cdf of a finite or infinite z is already in [0, 1]",
        "pre_adaptation": "absent", "current": "present, defence in depth",
        "load_bearing": False,
    })
    return rows


# =============================================================================================
# Verdict and scope, computed from the measurement.
# =============================================================================================


def verdict(dsr: dict[str, Any], mt: dict[str, Any], search: dict[str, Any]) -> dict[str, Any]:
    arms = dsr["arms"]
    silent = {name: a["silent_wrong"] for name, a in arms.items()}
    untyped = {name: a["grades"]["untyped_refusal"] for name, a in arms.items()}
    over = {name: a["grades"]["over_refusal"] for name, a in arms.items()}
    general = ("pydantic_contract_around_vectorbt", "numpy_scipy_trap_around_vectorbt")

    def key(name: str) -> tuple[int, int, int]:
        # Silent-wrong first (the cardinal sin), then untyped refusals, then over-refusals.
        return silent[name], untyped[name], over[name]

    best_general = min(general, key=key)
    pre_lost = key("argus_pre_adaptation") > key(best_general)
    current, rival = key("argus_current"), key(best_general)
    if current < rival:
        outcome = "argus_wins"
    elif current == rival:
        outcome = "tie"
    else:
        outcome = "rival_wins"
    pre_classes = search["versions"]["pre_adaptation"]["failure_classes"]
    cur_classes = search["versions"]["current"]["failure_classes"]
    mt_arms = mt["arms"]
    return {
        "dsr_outcome_now": outcome,
        "best_general_purpose_arm": best_general,
        "silent_wrong_by_arm": silent,
        "untyped_refusals_by_arm": untyped,
        "over_refusals_by_arm": over,
        "real_producer_tier": {
            "argus_current": arms["argus_current"]["by_tier"].get("real_producer"),
            "numpy_scipy_trap_whole_pipeline":
                dsr["numpy_scipy_trap_whole_pipeline_real_producer_only"]["grades"],
        },
        "argus_lost_before_adaptation": pre_lost,
        "hypothesis_failure_classes_pre": pre_classes,
        "hypothesis_failure_classes_current": cur_classes,
        "hand_designed_adversarial_inputs": len(HAND_DESIGNED_ADVERSARIAL),
        "multiple_testing_silent_on_invalid": {
            name: a["silent_on_invalid_input"] for name, a in mt_arms.items()},
        "multiple_testing_wrong_decisions": {
            name: a["wrong_decisions"] for name, a in mt_arms.items()},
        "summary": (
            f"DSR gate, {dsr['cases']:,} identical inputs: silent-wrong outputs — ARGUS now "
            f"{silent['argus_current']}, ARGUS before adaptation {silent['argus_pre_adaptation']}, "
            f"pydantic contract {silent['pydantic_contract_around_vectorbt']}, numpy/scipy trap "
            f"{silent['numpy_scipy_trap_around_vectorbt']}, vectorbt bare "
            f"{silent['vectorbt_bare']}. Before adaptation ARGUS "
            f"{'lost to' if pre_lost else 'did not lose to'} the best general-purpose arm "
            f"({best_general}). Hypothesis found {pre_classes} failure classes in the "
            f"pre-adaptation gates against the {len(HAND_DESIGNED_ADVERSARIAL)} inputs the "
            f"hand-picked adversarial set had tried, and {cur_classes} in the current gates."
        ),
    }


def scope_statement(dsr: dict[str, Any], search: dict[str, Any]) -> str:
    """Assembled from the live run, so no count in it can go stale."""
    pyd = dsr["arms"]["pydantic_contract_around_vectorbt"]["silent_wrong"]
    trap = dsr["arms"]["numpy_scipy_trap_around_vectorbt"]["silent_wrong"]
    pre_classes = search["versions"]["pre_adaptation"]["failure_classes"]
    return (
        f"Claimed: on {dsr['cases']:,} identical inputs — the 1,220 the standing row is already "
        f"measured on, plus poisoned, overflow, edge-valid and real-producer tiers (real Bitget "
        f"hourly candles, a seven-variant strategy family net of 12 bps) — ARGUS's current "
        f"deflated_sharpe returns "
        f"{dsr['arms']['argus_current']['silent_wrong']} silent-wrong outputs, against {pyd} for "
        f"vectorbt behind a pydantic contract and {trap} behind numpy/scipy trapping; and the "
        f"same derandomized Hypothesis search that found {pre_classes} failure classes across "
        f"ARGUS's gates before adaptation finds "
        f"{search['versions']['current']['failure_classes']} now. Where the general-purpose arms "
        f"still fail — pydantic's contract passes vectorbt's 0.5 on an overflowing Sharpe and "
        f"its 1.0 on a single trial; the numpy trap cannot see a NaN that arrives as an argument "
        f"— the counts are in the artefact, run, not argued.\n\n"
        "NOT claimed: that ARGUS was ahead before this comparison. It was behind the "
        "general-purpose contract on the property the capability is named for, and the fix was "
        "taken from that contract. Not claimed: that Hypothesis finding nothing proves there is "
        "nothing; it is a bounded search. Not claimed: that refusing an overflowing input beats "
        "computing it — a refusal is the acceptable answer the grading allows, not the ideal "
        "one. Not claimed: that the numpy trap loses on real data — wrapped around the whole "
        "pipeline rather than the gate alone, it refuses at the source every real case ARGUS "
        "refuses at the gate, and ties there. Not claimed: that a gate can detect a caller "
        "passing excess kurtosis where Pearson is expected; vectorbt's own accessor does exactly "
        "that, and no argument check can tell the two apart.\n"
    )


def main(max_examples: int = 300) -> dict[str, Any]:
    """Run every tier and every arm; return the serialisable report.

    Raises:
        GeneralOverfitGatesError: the vendored vectorbt baseline or the real candles failed to load.
    """
    from argus.eval.baselines.vectorbt_loader import (
        VectorbtBaselineLoadError,
        load_vectorbt_baseline,
    )

    try:
        symbols = load_vectorbt_baseline()
    except VectorbtBaselineLoadError as exc:
        raise GeneralOverfitGatesError(f"vectorbt baseline failed to load: {exc}") from exc

    import hypothesis
    import mpmath
    import pydantic
    import scipy
    import statsmodels

    closes = load_closes()
    scenarios = real_scenarios(closes)
    dsr = run_dsr_head_to_head(symbols, scenarios)
    mt = run_mt_head_to_head(real_p_value_cases(closes))
    search = hypothesis_search(max_examples=max_examples)
    return {
        "capability": "Overfitting gates that raise instead of returning NaN",
        "general_function": (
            "fail-loud statistical decision gates: runtime contract enforcement on numeric "
            "functions (argument and return validation), IEEE-754 exception trapping, "
            "multiple-testing control, and property-based adversarial search for inputs that "
            "break the contract"
        ),
        "rivals_run": {
            "pydantic_contract_around_vectorbt": "pydantic v2 validate_call + FiniteFloat + "
            "validate_return (MIT), around vectorbt's real vendored DSR",
            "numpy_scipy_trap_around_vectorbt": "np.errstate(all='raise') + "
            "scipy.special.errstate(all='raise') + warnings-as-errors (BSD)",
            "scipy_false_discovery_control": "scipy.stats.false_discovery_control (BSD)",
            "statsmodels_multipletests": "statsmodels.stats.multitest.multipletests (BSD)",
            "hypothesis": "property-based search, derandomized (MPL-2.0, used, not vendored)",
        },
        "versions": {
            "numpy": np.__version__, "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__, "pydantic": pydantic.VERSION,
            "hypothesis": hypothesis.__version__, "mpmath": mpmath.__version__,
        },
        "real_input": {
            "file": "data/regime_candles_fixture.json",
            "symbols": len(closes),
            "bars_per_symbol": sorted({len(v) for v in closes.values()}),
            "family": [label for label, _, _ in FAMILY] + [SHOCK[0]],
            "fee_round_trip": ROUND_TRIP_FEE,
        },
        "dsr_gate": dsr,
        "vectorbt_accessor_kurtosis_convention": kurtosis_convention_gap(scenarios, symbols),
        "multiple_testing": mt,
        "hypothesis_search": search,
        "hand_designed_adversarial_baseline": list(HAND_DESIGNED_ADVERSARIAL),
        "ablation": ablation(),
        "verdict": verdict(dsr, mt, search),
        "scope_statement": scope_statement(dsr, search),
    }


def render(report: dict[str, Any]) -> str:
    lines = ["GENERAL-PURPOSE RIVALS — overfitting gates that raise instead of returning NaN", ""]
    dsr = report["dsr_gate"]
    lines.append(f"DSR gate: {dsr['cases']:,} cases, tiers {dsr['tiers']}")
    for name, arm in dsr["arms"].items():
        lines.append(
            f"  {name:40s} silent_wrong={arm['silent_wrong']:4d} "
            f"untyped={arm['grades']['untyped_refusal']:3d} "
            f"over_refusal={arm['grades']['over_refusal']:3d} "
            f"false-admit(negated)={arm['false_admissions_negated_gate']:3d}")
    pipe = dsr["numpy_scipy_trap_whole_pipeline_real_producer_only"]
    lines.append(f"  numpy trap, whole pipeline (real tier only): silent_wrong="
                 f"{pipe['silent_wrong']} grades={pipe['grades']}")
    lines.append("")
    lines.append(f"multiple testing: {report['multiple_testing']['cases']} cases")
    for name, arm in report["multiple_testing"]["arms"].items():
        lines.append(
            f"  {name:34s} wrong={arm['wrong_decisions']:3d} "
            f"silent_on_invalid={arm['silent_on_invalid_input']:2d} "
            f"permissive={arm['permissive_on_invalid_input']:2d} "
            f"over_refusal={arm['over_refusals']:2d}")
    lines.append("")
    s = report["hypothesis_search"]["versions"]
    for version in ("pre_adaptation", "current"):
        lines.append(f"hypothesis {version}: {s[version]['failure_classes']} failure classes in "
                     f"{s[version]['functions_with_failures']}")
    lines.append("")
    for row in report["ablation"]:
        lines.append(f"  ablation {row['case'][:60]:60s} pre={row['pre_adaptation'][:28]:28s} "
                     f"now={row['current'][:12]} load_bearing={row['load_bearing']}")
    lines.append("")
    lines.append(report["verdict"]["summary"])
    return "\n".join(lines)


if __name__ == "__main__":
    from argus.eval.artefact import write as write_artefact

    result = main()
    print(render(result))
    undefined = write_artefact(ARTEFACT, result)
    print(f"\nsaved -> {ARTEFACT} ({len(undefined)} non-finite value(s) written as null)")


__all__ = [
    "ARTEFACT",
    "GATE_THRESHOLD",
    "GRADES",
    "HAND_DESIGNED_ADVERSARIAL",
    "ArmScore",
    "GateCase",
    "GeneralOverfitGatesError",
    "MtCase",
    "Outcome",
    "RealScenario",
    "ablation",
    "admitted",
    "bh_reference",
    "bonferroni_reference",
    "dsr_arms",
    "edge_valid_cases",
    "family_net_returns",
    "grade",
    "grid_p_value_cases",
    "hypothesis_search",
    "load_closes",
    "main",
    "mt_arms",
    "overflow_cases",
    "poisoned_cases",
    "poisoned_p_value_cases",
    "produce",
    "pydantic_contract",
    "real_p_value_cases",
    "real_scenarios",
    "reference_dsr",
    "render",
    "run_dsr_head_to_head",
    "run_mt_head_to_head",
    "scope_statement",
    "standing_input_cases",
    "trapped_pipeline",
    "verdict",
]
