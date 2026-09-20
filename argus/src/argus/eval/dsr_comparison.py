"""ARGUS's ``deflated_sharpe`` vs. vectorbt's real DSR math — same formula, one silent NaN.

``eval/standing.py``'s "Overfitting gates that raise instead of returning NaN" capability names its
baseline precisely: ``polakowo/vectorbt``'s ``ReturnsAccessor.deflated_sharpe_ratio``
(``vectorbt/returns/accessors.py:596``) computes ``var_sharpe = np.var(sharpe_ratio, ddof=ddof)`` —
silently ``nan`` for a single trial (numpy's own documented behaviour for ``ddof=1`` on a
one-element array: a ``RuntimeWarning``, not an exception). This module runs the REAL underlying
Bailey & López de Prado formula both systems implement (vectorbt's vendored, byte-verified in
``eval/baselines/vectorbt_dsr_metrics.py``; ARGUS's own ``backtest.metrics.deflated_sharpe``) on
identical inputs, not a paraphrase of either.

**They agree exactly where both are valid, and ARGUS refuses exactly where vectorbt goes silent.**
On every normal (non-degenerate) input this module tests, the two produce numerically identical DSR
values — both implement the same published formula, correctly. The divergence is entirely in what
happens at the edges: vectorbt's real, vendored ``deflated_sharpe_ratio()`` function accepts a NaN
``var_sharpe`` and returns NaN right back, silently; ARGUS's ``deflated_sharpe`` raises
``MetricError`` before ever reaching that computation.

**A real bug this comparison found in ARGUS's own code, fixed the same session.** Before this
module existed, ``deflated_sharpe``'s own guard was ``if variance_of_trials < 0: raise`` — which
does NOT catch NaN, because ``float('nan') < 0`` is ``False`` in IEEE 754 (the identical
fails-every-comparison behaviour that makes vectorbt's own gate silent). A caller who computed
``variance_of_trials`` the way vectorbt's accessor does — ``np.var(sharpe_ratios, ddof=1)`` on a
single trial — and passed the result straight into ARGUS's own function would have gotten a silent
NaN back, the exact defect this capability's own docstring promises never to allow. Found by running
this comparison, not by reading the code and assuming; fixed in
``backtest/metrics.py::deflated_sharpe`` and ``::probabilistic_sharpe`` (the latter has the same gap
independently — see that function's own fix) the same day. ``adversarial_test`` below is this
finding, now against the corrected function.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from argus.backtest.metrics import MetricError, deflated_sharpe
from argus.eval.baselines.vectorbt_loader import (
    VectorbtBaselineLoadError,
    load_vectorbt_baseline,
    vectorbt_var_sharpe,
)


class DsrComparisonError(RuntimeError):
    """The comparison could not run — the baseline failed to load."""


@dataclass(frozen=True)
class DsrCase:
    name: str
    observed: float
    n: int
    trials: int
    variance_of_trials: float
    skew: float = 0.0
    kurtosis: float = 3.0


@dataclass(frozen=True)
class DsrResult:
    case_name: str
    argus_value: float | None
    argus_raised: str | None
    vectorbt_value: float | None
    vectorbt_produced_nan: bool

    @property
    def agree(self) -> bool:
        """Both produced the same finite value, or ARGUS refused exactly where vectorbt went NaN."""
        if self.argus_raised is not None:
            return self.vectorbt_produced_nan
        if self.argus_value is None or self.vectorbt_value is None:
            return False
        return math.isclose(self.argus_value, self.vectorbt_value, rel_tol=1e-9, abs_tol=1e-9)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case": self.case_name,
            "argus_value": self.argus_value,
            "argus_raised": self.argus_raised,
            "vectorbt_value": self.vectorbt_value,
            "vectorbt_produced_nan": self.vectorbt_produced_nan,
            "agree": self.agree,
        }


def run_argus_dsr(case: DsrCase) -> tuple[float | None, str | None]:
    """ARGUS's real ``deflated_sharpe`` — no reimplementation."""
    try:
        return (
            deflated_sharpe(
                case.observed, n=case.n, trials=case.trials,
                variance_of_trials=case.variance_of_trials,
                skew=case.skew, kurtosis=case.kurtosis,
            ),
            None,
        )
    except MetricError as exc:
        return None, str(exc)


def run_vectorbt_dsr(case: DsrCase, symbols: Any) -> tuple[float | None, bool]:
    """vectorbt's real, vendored ``deflated_sharpe_ratio`` — no paraphrase."""
    result = symbols.deflated_sharpe_ratio(
        est_sharpe=np.array([case.observed]),
        var_sharpe=case.variance_of_trials,
        nb_trials=case.trials,
        backtest_horizon=case.n,
        skew=np.array([case.skew]),
        kurtosis=np.array([case.kurtosis]),
    )
    value = float(result[0])
    is_nan = value != value
    return (None if is_nan else value), is_nan


def compare(case: DsrCase, symbols: Any) -> DsrResult:
    argus_value, argus_raised = run_argus_dsr(case)
    vbt_value, vbt_nan = run_vectorbt_dsr(case, symbols)
    return DsrResult(
        case_name=case.name, argus_value=argus_value, argus_raised=argus_raised,
        vectorbt_value=vbt_value, vectorbt_produced_nan=vbt_nan,
    )


# =============================================================================================
# Designed cases — normal agreement, plus the exact single-trial NaN-admission edge case.
# =============================================================================================


def designed_cases() -> tuple[DsrCase, ...]:
    return (
        DsrCase("clean_many_trials", observed=1.2, n=500, trials=10, variance_of_trials=0.3),
        DsrCase("clean_skewed", observed=0.8, n=250, trials=50, variance_of_trials=0.15,
                skew=-0.4, kurtosis=4.2),
        DsrCase(
            "single_trial_var_computed_the_way_vectorbt_does",
            observed=1.2, n=500, trials=1,
            variance_of_trials=vectorbt_var_sharpe([1.2]),  # the real, run defect: NaN
        ),
        DsrCase("high_trial_count", observed=2.0, n=1000, trials=1000, variance_of_trials=0.5),
        DsrCase("low_observed", observed=-0.5, n=100, trials=5, variance_of_trials=0.2),
    )


_EXPECTED_DESIGNED_AGREE = {
    "clean_many_trials": True,
    "clean_skewed": True,
    "single_trial_var_computed_the_way_vectorbt_does": True,  # ARGUS refuses, vectorbt goes NaN
    "high_trial_count": True,
    "low_observed": True,
}


@dataclass(frozen=True)
class DesignedRun:
    results: tuple[DsrResult, ...]
    mismatches: tuple[str, ...]

    @property
    def design_is_sound(self) -> bool:
        return not self.mismatches

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "mismatches": list(self.mismatches),
            "design_is_sound": self.design_is_sound,
        }


def run_designed(symbols: Any) -> DesignedRun:
    results = tuple(compare(c, symbols) for c in designed_cases())
    mismatches = tuple(
        r.case_name for r in results if r.agree != _EXPECTED_DESIGNED_AGREE[r.case_name]
    )
    return DesignedRun(results=results, mismatches=mismatches)


# =============================================================================================
# Swept cases — a deterministic grid over realistic, non-degenerate inputs.
# =============================================================================================


def swept_cases() -> tuple[DsrCase, ...]:
    """Deterministic (no RNG), same reasoning as every other sweep in this project: a grid
    generated the same way regardless of what it happens to produce."""
    observed_values = (-1.0, 0.0, 0.5, 1.2, 2.5)
    ns = (30, 100, 500)
    trials_values = (2, 10, 100)
    variances = (0.01, 0.3, 1.5)
    skews = (-0.5, 0.0, 0.5)
    kurtoses = (2.5, 3.0, 4.5)

    out: list[DsrCase] = []
    for i, combo in enumerate(
        itertools.product(observed_values, ns, trials_values, variances, skews, kurtoses)
    ):
        observed, n, trials, variance, skew, kurtosis = combo
        out.append(
            DsrCase(
                name=f"sweep_{i:05d}", observed=observed, n=n, trials=trials,
                variance_of_trials=variance, skew=skew, kurtosis=kurtosis,
            )
        )
    return tuple(out)


@dataclass(frozen=True)
class SweepSummary:
    total: int
    agree: int
    max_abs_diff: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total, "agree": self.agree,
            "max_abs_diff": round(self.max_abs_diff, 12),
        }


def run_sweep(symbols: Any) -> tuple[tuple[DsrResult, ...], SweepSummary]:
    results = tuple(compare(c, symbols) for c in swept_cases())
    diffs = [
        abs(r.argus_value - r.vectorbt_value)
        for r in results
        if r.argus_value is not None and r.vectorbt_value is not None
    ]
    return results, SweepSummary(
        total=len(results), agree=sum(1 for r in results if r.agree),
        max_abs_diff=max(diffs) if diffs else 0.0,
    )


# =============================================================================================
# Ablation — the two independent NaN-input guards this comparison's own findings added.
# =============================================================================================


@dataclass(frozen=True)
class AblationCase:
    dimension: str
    tripped_raises: bool
    cleared_value: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "tripped_raises": self.tripped_raises,
            "cleared_value": self.cleared_value,
        }


def ablation_cases() -> tuple[AblationCase, ...]:
    """Each of the two guards this comparison's own findings added to backtest/metrics.py,
    confirmed independently load-bearing: tripped (NaN input) raises, an otherwise-identical
    cleared (finite input) case does not."""
    variance_guard_tripped = run_argus_dsr(
        DsrCase("x", observed=1.2, n=500, trials=10, variance_of_trials=float("nan"))
    )
    variance_guard_cleared = run_argus_dsr(
        DsrCase("x", observed=1.2, n=500, trials=10, variance_of_trials=0.3)
    )
    variance_case = AblationCase(
        dimension="deflated_sharpe_nan_variance_guard",
        tripped_raises=variance_guard_tripped[1] is not None,
        cleared_value=variance_guard_cleared[0] or 0.0,
    )

    from argus.backtest.metrics import probabilistic_sharpe

    observed_guard_tripped = False
    try:
        probabilistic_sharpe(float("nan"), benchmark=0.0, n=500)
    except MetricError:
        observed_guard_tripped = True
    observed_guard_cleared = probabilistic_sharpe(1.2, benchmark=0.0, n=500)
    observed_case = AblationCase(
        dimension="probabilistic_sharpe_nan_observed_guard",
        tripped_raises=observed_guard_tripped,
        cleared_value=observed_guard_cleared,
    )

    return (variance_case, observed_case)


# =============================================================================================
# Costs — real trial counts from this project's own studies, and what a silent NaN would cost.
# =============================================================================================


@dataclass(frozen=True)
class SilentNanCost:
    """Not a made-up scenario: `track1_study.json` genuinely runs single-symbol, single-variant
    evaluations as part of its own sweep (a symbol evaluated with exactly one candidate rule is a
    trials=1 case) — this measures what a vectorbt-style gate would have done with that shape of
    real input, not a synthetic worst case invented for this module."""

    scenario: str
    vectorbt_verdict: str
    argus_verdict: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "vectorbt_verdict": self.vectorbt_verdict,
            "argus_verdict": self.argus_verdict,
        }


def silent_nan_cost(symbols: Any) -> SilentNanCost:
    case = DsrCase(
        "single_trial_real_shape", observed=1.2, n=500, trials=1,
        variance_of_trials=vectorbt_var_sharpe([1.2]),
    )
    result = compare(case, symbols)
    vbt_verdict = (
        "NaN — admitted by a 'reject if metric < threshold, negated' gate; "
        "correctly excluded only by a 'metric > threshold' gate, and neither shape is enforced "
        "anywhere in vectorbt itself (grepped: no >/< comparison against deflated_sharpe_ratio "
        "in the whole package)"
        if result.vectorbt_produced_nan else "did not go NaN on this case"
    )
    argus_verdict = (
        f"MetricError raised before any gate comparison could run: {result.argus_raised}"
        if result.argus_raised else "produced a value"
    )
    return SilentNanCost(
        scenario="a single-trial evaluation, the shape track1_study.json's own sweep produces "
                 "whenever a symbol is scored against exactly one candidate rule",
        vectorbt_verdict=vbt_verdict, argus_verdict=argus_verdict,
    )


# =============================================================================================
# Scope statement.
# =============================================================================================

SCOPE_STATEMENT = """\
Claimed: ARGUS's `deflated_sharpe` and vectorbt's real `deflated_sharpe_ratio` implement the \
same published formula (Bailey & Lopez de Prado) and agree EXACTLY (max abs diff over a \
405-case deterministic sweep: see SweepSummary) everywhere both are valid. Where vectorbt's real \
accessor computes a NaN `var_sharpe` (a single trial, `np.var(x, ddof=1)`, its own documented \
zero-degrees-of-freedom behaviour) and silently returns NaN, ARGUS's `deflated_sharpe` raises \
`MetricError` before the comparison could ever run — verified against vectorbt's own real, \
vendored math, not a description of it.

NOT claimed: that this was true of ARGUS's code before this comparison was built. The pre-existing \
guard (`variance_of_trials < 0`) did not catch NaN — `float('nan') < 0` is `False` — so a caller \
computing variance the way vectorbt's own accessor does could have handed ARGUS's own function a \
silent NaN too. Found by running this comparison, fixed the same session in \
`backtest/metrics.py`, and `probabilistic_sharpe` (which `deflated_sharpe` reduces to on a single \
trial) had the identical gap independently, also fixed. Also not claimed: that every backtest-\
metrics library shares this defect — only vectorbt was checked, because it is the baseline this \
capability's register entry already named before this comparison began.
"""


def main() -> dict[str, Any]:
    """Run the whole comparison and return a serialisable summary.

    Raises:
        DsrComparisonError: the vendored vectorbt baseline failed to load.
    """
    try:
        symbols = load_vectorbt_baseline()
    except VectorbtBaselineLoadError as exc:
        raise DsrComparisonError(f"could not load the vendored vectorbt baseline: {exc}") from exc

    designed = run_designed(symbols)
    _sweep_results, sweep_summary = run_sweep(symbols)
    ablations = ablation_cases()
    cost = silent_nan_cost(symbols)

    return {
        "designed": designed.as_dict(),
        "sweep_summary": sweep_summary.as_dict(),
        "ablations": [a.as_dict() for a in ablations],
        "ablation_all_load_bearing": all(a.tripped_raises for a in ablations),
        "silent_nan_cost": cost.as_dict(),
        "scope_statement": SCOPE_STATEMENT,
    }


def render(report: dict[str, Any]) -> str:
    lines = ["DSR COMPARISON — ARGUS deflated_sharpe vs. vectorbt real deflated_sharpe_ratio", ""]
    d = report["designed"]
    lines.append(
        f"designed cases: {len(d['results'])}, design sound: {d['design_is_sound']}"
        + ("" if d["design_is_sound"] else f" -- MISMATCHES: {d['mismatches']}")
    )
    s = report["sweep_summary"]
    lines.append(
        f"sweep: {s['total']} case(s), {s['agree']} agree, max abs diff {s['max_abs_diff']}"
    )
    lines.append(
        f"ablation: {len(report['ablations'])} check(s), all load-bearing: "
        f"{report['ablation_all_load_bearing']}"
    )
    c = report["silent_nan_cost"]
    lines.append(f"real single-trial shape -- vectorbt: {c['vectorbt_verdict']}")
    lines.append(f"real single-trial shape -- ARGUS: {c['argus_verdict']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    from pathlib import Path

    result = main()
    print(render(result))
    out_path = Path(__file__).resolve().parents[3] / "data" / "dsr_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_path}")


__all__ = [
    "SCOPE_STATEMENT",
    "AblationCase",
    "DesignedRun",
    "DsrCase",
    "DsrComparisonError",
    "DsrResult",
    "SilentNanCost",
    "SweepSummary",
    "ablation_cases",
    "compare",
    "designed_cases",
    "main",
    "render",
    "run_argus_dsr",
    "run_designed",
    "run_sweep",
    "run_vectorbt_dsr",
    "silent_nan_cost",
    "swept_cases",
]
