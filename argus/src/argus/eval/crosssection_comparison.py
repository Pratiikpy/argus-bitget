"""ARGUS's ``CrossRank`` vs. qlib's real ``CSRankNorm`` — same panels, both systems, run.

``eval/standing.py``'s "Cross-sectional factor evaluation" capability names its baseline as
"microsoft/qlib, which groups a (datetime, instrument) frame by datetime and ranks within each
group" — ``qlib/data/dataset/processor.py``'s ``CSRankNorm``, vendored byte-verified in
``eval/baselines/qlib_cs_processor.py`` (see that package's docstring) and actually executed here,
not paraphrased, against ``argus.research.grammar.CrossRank.combine()`` on identical input.

**What agrees and what doesn't, both verified by running the real code, not derived on paper.**
Both group by date and rank within each group with midrank tie handling — on every synthetic panel
this module runs (hand-designed and swept), the two agree EXACTLY on relative ORDER: an ``argsort``
of ARGUS's output and an ``argsort`` of qlib's output on the same input are always identical. They
disagree on SCALE: ARGUS's ``combine()`` returns values in ``[0, 1]`` (rank position ÷ (n-1));
qlib's ``CSRankNorm`` additionally re-centers and rescales toward a roughly-unit-std distribution
(``(pct_rank - 0.5) * 3.46``, its own docstring's derivation) — a further normalization step ARGUS's
``crossrank`` deliberately does not apply, because ARGUS's own portfolio-weight step is a separate
operator (``CrossScale``, sum-of-abs-values-to-one) with a different job.

**A real, verified divergence in a real edge case.** A single-instrument cross-section: ARGUS's
``combine()`` explicitly returns ``0.5`` — neutral, "neither high nor low among one name," per its
own docstring. qlib's real, unmodified ``CSRankNorm`` has no such guard: ``pandas.rank(pct=True)``
on a one-row group returns ``1.0`` (the only element is trivially rank 1 of 1), which the ``(x -
0.5) * 3.46`` transform turns into ``1.73`` — the maximum value the transform can produce, treating
the sole instrument as though it were the standout winner of a large cross-section it is not
actually being compared against. Confirmed by running both real functions on a one-row input, not
inferred from reading. ``adversarial_test`` below is this finding.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import pandas as pd

from argus.eval.baselines.qlib_loader import QlibBaselineLoadError, load_qlib_baseline
from argus.research.grammar import CrossRank


class CrossSectionComparisonError(RuntimeError):
    """The comparison could not run — the baseline failed to load."""


def run_argus(values: list[float]) -> list[float]:
    """ARGUS's real ``CrossRank.combine`` — no reimplementation."""
    return CrossRank.combine(values)


def run_qlib(values: list[float], cs_rank_norm: Any) -> list[float]:
    """qlib's real, vendored ``CSRankNorm`` — no paraphrase.

    Args:
        values: One date-group's raw values.
        cs_rank_norm: The vendored ``CSRankNorm`` class, from :func:`load_qlib_baseline`.
    """
    df = pd.DataFrame({"datetime": ["d"] * len(values), "value": values}).set_index("datetime")
    if df.empty:
        return []
    processor = cs_rank_norm(fields_group=None)
    out = processor(df.copy())
    return [float(v) for v in out["value"].tolist()]


def _argsort_with_tie_groups(values: list[float]) -> tuple[tuple[int, ...], ...]:
    """Return the input's own tie-grouped order (ties collapsed to a frozenset of indices).

    Two rankings "agree" when they produce the same tie-grouped order, not merely the same
    ``sorted()`` index list — a strict ``argsort`` breaks ties by index position, which would
    call two rankings that both correctly tie two symbols together a "disagreement" if they
    happened to list the tied pair in a different order.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    groups: list[list[int]] = []
    for i in order:
        if groups and values[groups[-1][-1]] == values[i]:
            groups[-1].append(i)
        else:
            groups.append([i])
    return tuple(tuple(sorted(g)) for g in groups)


@dataclass(frozen=True)
class RankOrderComparison:
    values: tuple[float, ...]
    argus_output: tuple[float, ...]
    qlib_output: tuple[float, ...]
    same_order: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "values": list(self.values),
            "argus_output": list(self.argus_output),
            "qlib_output": list(self.qlib_output),
            "same_order": self.same_order,
        }


def compare_ordering(values: list[float], cs_rank_norm: Any) -> RankOrderComparison:
    argus_out = run_argus(values)
    qlib_out = run_qlib(values, cs_rank_norm)
    same = _argsort_with_tie_groups(argus_out) == _argsort_with_tie_groups(qlib_out)
    return RankOrderComparison(
        values=tuple(values), argus_output=tuple(argus_out), qlib_output=tuple(qlib_out),
        same_order=same,
    )


# =============================================================================================
# Designed panels — hand-built to probe ties, extremes, and the single-instrument edge case.
# =============================================================================================


def designed_panels() -> tuple[tuple[float, ...], ...]:
    return (
        (1.0, 2.0, 3.0, 4.0, 5.0),
        (5.0, 3.0, 3.0, 1.0),
        (1.0, 1.0, 1.0),
        (10.0, -5.0, 0.0, 0.0, 3.0, 3.0, 3.0),
        (0.0,),
        (),
        (-1.0, -1.0, 2.0, 2.0, 2.0, 5.0),
    )


# =============================================================================================
# Swept panels — a deterministic grid over realistic (size, tie-density, spread) combinations.
# =============================================================================================


def swept_panels() -> tuple[tuple[float, ...], ...]:
    """Deterministic (no RNG), same reasoning as ``mandate_comparison.swept_scenarios``: a grid
    generated the same way regardless of what it happens to produce, not tuned to any case."""
    sizes = (1, 2, 3, 5, 8, 12)  # 12 matches the real RTOKEN_SYMBOLS universe size
    tie_densities = (0, 1, 2)  # how many repeated values to inject
    spreads = ((1.0,), (1.0, -1.0), (0.01,))  # step size / sign pattern for generated values

    out: list[tuple[float, ...]] = []
    for size, ties, spread in itertools.product(sizes, tie_densities, spreads):
        step = spread[0]
        sign = spread[1] if len(spread) > 1 else 1.0
        values = [round(i * step * (sign if i % 2 else 1.0), 6) for i in range(size)]
        for t in range(min(ties, max(size - 1, 0))):
            values[t] = values[-1] if values else 0.0
        out.append(tuple(values))
    return tuple(out)


# =============================================================================================
# Ablation — ARGUS's own two design choices, each shown to matter.
# =============================================================================================


def _naive_rank_no_midrank(values: list[float]) -> list[float]:
    """CrossRank.combine() with midrank tie handling removed — ties broken by first-seen order.

    Not vendored, not a competitor's code: this is ARGUS's own transform with one of its two
    design choices deliberately disabled, to show that choice is load-bearing (the standard
    ablation shape — remove a component, see if the output changes).
    """
    n = len(values)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for position, i in enumerate(order):
        ranks[i] = position / (n - 1)
    return ranks


def _naive_rank_no_single_name_guard(values: list[float]) -> list[float]:
    """CrossRank.combine() with the n<=1 neutral guard removed — divides by (n-1) unguarded."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [float("nan")]  # what an unguarded (below + (ties-1)/2) / (n-1) produces: 0/0
    out: list[float] = []
    for v in values:
        below = sum(1 for other in values if other < v)
        ties = sum(1 for other in values if other == v)
        out.append((below + (ties - 1) / 2) / (n - 1))
    return out


@dataclass(frozen=True)
class AblationCase:
    dimension: str
    real_output: tuple[float, ...]
    ablated_output: tuple[float, ...]
    differs: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "real_output": list(self.real_output),
            "ablated_output": list(self.ablated_output),
            "differs": self.differs,
        }


def ablation_cases() -> tuple[AblationCase, ...]:
    tie_case = [5.0, 3.0, 3.0, 1.0]
    real_tie = run_argus(tie_case)
    ablated_tie = _naive_rank_no_midrank(tie_case)
    midrank_case = AblationCase(
        dimension="midrank_tie_handling",
        real_output=tuple(real_tie), ablated_output=tuple(ablated_tie),
        differs=tuple(real_tie) != tuple(ablated_tie),
    )

    single_case = [7.0]
    real_single = run_argus(single_case)
    ablated_single = _naive_rank_no_single_name_guard(single_case)
    guard_case = AblationCase(
        dimension="single_name_neutral_guard",
        real_output=tuple(real_single), ablated_output=tuple(ablated_single),
        differs=real_single != ablated_single
        or any(v != v for v in ablated_single),  # NaN != NaN is itself the divergence
    )

    return (midrank_case, guard_case)


# =============================================================================================
# Costs — the measured size of qlib's single-name distortion, on qlib's own scale.
# =============================================================================================


@dataclass(frozen=True)
class SingleNameDistortionCost:
    qlib_neutral_value: float
    """What qlib's own transform centers on for a genuinely neutral rank (mid-cross-section)."""
    qlib_single_name_value: float
    """What qlib's real, unmodified code actually returns for a one-instrument group."""
    distortion_magnitude: float
    argus_single_name_value: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "qlib_neutral_value": self.qlib_neutral_value,
            "qlib_single_name_value": self.qlib_single_name_value,
            "distortion_magnitude": round(self.distortion_magnitude, 4),
            "argus_single_name_value": self.argus_single_name_value,
        }


def single_name_distortion_cost(cs_rank_norm: Any) -> SingleNameDistortionCost:
    """qlib's own docstring derives the transform's centre as ``(pct_rank.mean() - 0.5) * 3.46
    == 0`` (pct_rank's mean is 0.5 by construction of a percentile rank). Not assumed from the
    docstring alone: verified here by running the real function on a large, evenly-spaced,
    tie-free group, where the mean output should sit close to 0 — an actual measurement, not a
    single hand-picked "this index looks like the middle" value (which, checked directly first,
    is NOT the transform's centre for any specific finite group — e.g. a 3-element group's own
    middle element outputs 0.577, not 0, because its percentile rank is 2/3, not exactly 0.5)."""
    large_balanced_group = [float(i) for i in range(1, 101)]  # 1..100, evenly spaced, no ties
    qlib_large = run_qlib(large_balanced_group, cs_rank_norm)
    qlib_mean = sum(qlib_large) / len(qlib_large)
    qlib_single = run_qlib([7.0], cs_rank_norm)[0]
    return SingleNameDistortionCost(
        qlib_neutral_value=round(qlib_mean, 6),
        qlib_single_name_value=qlib_single,
        distortion_magnitude=abs(qlib_single - qlib_mean),
        argus_single_name_value=run_argus([7.0])[0],
    )


# =============================================================================================
# Scope statement.
# =============================================================================================

def scope_statement(swept_count: int) -> str:
    """Assembled with the live sweep size, not a typed-in snapshot of it.

    The panel count used to be a literal "324-panel" in this prose while `main()` computed and
    published a separate, correct `swept_count` two lines below it — the same
    stale-number-in-scope_statement defect found and fixed elsewhere this session
    (`quarantine_comparison.py`, `dsr_comparison.py`). Taking the count as a parameter means the
    two can no longer read differently.
    """
    return f"""\
Claimed: within cross-sectional RANKING (grouping a panel by date and ordering instruments \
within each date), ARGUS's `crossrank` and qlib's real `CSRankNorm` agree exactly on relative \
order on every panel tested — hand-designed and a {swept_count}-panel deterministic sweep — \
verified by running both real functions, not derived. Where they diverge is a real, run-verified \
edge case: \
a single-instrument date qlib's real code turns into an extreme value (1.73 on its own scale, \
the maximum the transform can produce) while ARGUS's own explicit guard returns a neutral 0.5, \
matching `CrossRank`'s own documented design rationale.

NOT claimed: that ARGUS's `crossrank` is a complete substitute for qlib's `CSRankNorm`. qlib's \
extra step (re-centering and rescaling toward unit variance, the `(x - 0.5) * 3.46` transform) \
is a genuine additional normalization ARGUS's own `crossrank` does not perform — by design, not \
oversight: `CrossScale` is ARGUS's own separate operator for turning a raw signal into a \
portfolio weight (sum-of-abs-values to one), a different job than reshaping a rank's \
distribution toward unit variance. Also not claimed: that the single-name divergence has \
material real-world impact on ARGUS's own decisions today — RTOKEN_SYMBOLS is a fixed 12-name \
universe, and how often a real trading day's data actually shrinks a cross-sectional group to \
one instrument has not been separately measured against live data here (would need a live pull \
this module does not make) — NOT VERIFIED, stated plainly rather than assumed favourable.
"""


def main() -> dict[str, Any]:
    """Run the whole comparison and return a serialisable summary.

    Raises:
        CrossSectionComparisonError: the vendored qlib baseline failed to load.
    """
    try:
        baseline = load_qlib_baseline()
    except QlibBaselineLoadError as exc:
        raise CrossSectionComparisonError(
            f"could not load the vendored qlib baseline: {exc}"
        ) from exc

    designed = [compare_ordering(list(p), baseline.cs_rank_norm) for p in designed_panels()]
    swept = [compare_ordering(list(p), baseline.cs_rank_norm) for p in swept_panels()]
    ablations = ablation_cases()
    cost = single_name_distortion_cost(baseline.cs_rank_norm)

    return {
        "designed": [c.as_dict() for c in designed],
        "designed_all_agree": all(c.same_order for c in designed),
        "swept_count": len(swept),
        "swept_all_agree": all(c.same_order for c in swept),
        "ablations": [a.as_dict() for a in ablations],
        "ablation_all_load_bearing": all(a.differs for a in ablations),
        "single_name_distortion": cost.as_dict(),
        "scope_statement": scope_statement(len(swept)),
    }


def render(report: dict[str, Any]) -> str:
    lines = ["CROSS-SECTION COMPARISON — ARGUS crossrank vs. qlib real CSRankNorm", ""]
    lines.append(
        f"designed panels: {len(report['designed'])}, all agree on order: "
        f"{report['designed_all_agree']}"
    )
    lines.append(
        f"swept panels: {report['swept_count']}, all agree on order: {report['swept_all_agree']}"
    )
    lines.append(
        f"ablation: {len(report['ablations'])} check(s), all load-bearing: "
        f"{report['ablation_all_load_bearing']}"
    )
    d = report["single_name_distortion"]
    lines.append(
        f"single-name distortion: qlib returns {d['qlib_single_name_value']} for one "
        f"instrument (neutral would be {d['qlib_neutral_value']}, magnitude "
        f"{d['distortion_magnitude']}); ARGUS returns {d['argus_single_name_value']}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    from pathlib import Path

    from argus.eval.artefact import write as write_artefact

    result = main()
    print(render(result))
    out_path = Path(__file__).resolve().parents[3] / "data" / "crosssection_comparison.json"
    undefined = write_artefact(out_path, result)
    if undefined:
        print(f"\nnon-finite (written as null): {', '.join(undefined)}")
    print(f"\nsaved -> {out_path}")


__all__ = [
    "AblationCase",
    "CrossSectionComparisonError",
    "RankOrderComparison",
    "SingleNameDistortionCost",
    "ablation_cases",
    "compare_ordering",
    "designed_panels",
    "main",
    "render",
    "run_argus",
    "run_qlib",
    "scope_statement",
    "single_name_distortion_cost",
    "swept_panels",
]
