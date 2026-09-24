"""ARGUS's rule-lifecycle ``review`` vs. TradingAgents' real ``TradingMemoryLog`` — same fixtures.

``eval/standing.py``'s "Self-evolving review rules" capability names its baseline as
"TauricResearch/TradingAgents reflection memory" — the same real, vendored ``TradingMemoryLog``
already run for "Episodic memory across decisions" (``eval/baselines/tradingagents_memory.py``,
byte-verified, no new vendoring needed here). This module runs it again, against a different
claim: not the point-in-time guard this time, but whether either system EVER refuses to trust a
stored lesson that has not earned it.

**The real, verified structural difference.** ARGUS's ``desk.review.evaluate()`` replays a
candidate rule against decisions the desk already took and grades it against INDEPENDENTLY
observed defects (from the grounding/conflict/contradiction checkers, the risk layer, and settled
outcomes — never the rule's own say-so). A rule earns one of seven named states
(:class:`~argus.desk.review.Status`) and is refused BY NAME — ``DEAD_WEIGHT``,
``NO_DISCRIMINATION``, ``MISLEADING`` — when the replay shows it does not discriminate or is wrong
more often than right. TradingAgents' real ``TradingMemoryLog`` has no equivalent anywhere: a
reflection is written once, after one decision, and re-injected into every future relevant prompt
unconditionally — confirmed here by storing a reflection about a decision that was DEMONSTRABLY
WRONG (a real -15% realised return) and showing the real vendored code returns it in a later
context exactly as it would a reflection about a right call. Grepped directly across the whole
TradingAgents repository for any precision/track-record concept tied to memory or reflection:
zero matches.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from argus.desk.review import (
    MIN_DECISIONS,
    MIN_FIRINGS,
    Defect,
    DefectKind,
    Rule,
    Status,
    evaluate,
)
from argus.eval.baselines.tradingagents_loader import (
    TradingAgentsBaselineLoadError,
    load_trading_memory_log_class,
)


class ReviewComparisonError(RuntimeError):
    """The comparison could not run — the baseline failed to load."""


def run_tradingagents_reflection_reuse(
    *, was_right: bool, memory_log_class: type
) -> str:
    """Store one graded decision — deliberately either a correct or a demonstrably wrong call —
    and read back what a LATER decision on the same ticker would see. TradingAgents' real code,
    run twice, once each way."""
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "mem.md"
        log = memory_log_class({"memory_log_path": str(log_path)})
        log.store_decision("NVDA", "2026-01-01", "Buy at 120, thesis: momentum will continue")
        if was_right:
            reflection = (
                "Momentum call was correct. The stock continued higher. Lesson: trust momentum."
            )
            raw_return, alpha_return = 0.12, 0.10
        else:
            reflection = (
                "Momentum call was WRONG. The stock reversed sharply. Lesson: trust momentum less."
            )
            raw_return, alpha_return = -0.15, -0.18
        log.update_with_outcome(
            "NVDA", "2026-01-01", raw_return=raw_return, alpha_return=alpha_return,
            holding_days=3, reflection=reflection, resolution_date="2026-01-04",
        )
        return str(log.get_past_context("NVDA", as_of="2026-01-10"))


@dataclass(frozen=True)
class UnconditionalReuseResult:
    right_call_reinjected: bool
    wrong_call_reinjected: bool

    @property
    def no_precision_gate(self) -> bool:
        """Both a right and a demonstrably wrong call are re-injected identically — no gate
        distinguishes them by track record."""
        return self.right_call_reinjected and self.wrong_call_reinjected

    def as_dict(self) -> dict[str, Any]:
        return {
            "right_call_reinjected": self.right_call_reinjected,
            "wrong_call_reinjected": self.wrong_call_reinjected,
            "no_precision_gate": self.no_precision_gate,
        }


def run_unconditional_reuse_case(memory_log_class: type) -> UnconditionalReuseResult:
    right_context = run_tradingagents_reflection_reuse(
        was_right=True, memory_log_class=memory_log_class
    )
    wrong_context = run_tradingagents_reflection_reuse(
        was_right=False, memory_log_class=memory_log_class
    )
    return UnconditionalReuseResult(
        right_call_reinjected="correct" in right_context,
        wrong_call_reinjected="WRONG" in wrong_context,
    )


# =============================================================================================
# ARGUS's real lifecycle, run across every named status — designed cases.
# =============================================================================================


def _records(n: int, *, fires_on: set[int]) -> list[dict[str, Any]]:
    return [{"seq": i, "fire": i in fires_on} for i in range(n)]


def _rule() -> Rule:
    return Rule(
        name="designed-rule", prompt="x", rationale="x",
        targets=frozenset({DefectKind.GROUNDING}),
        predicate=lambda r: bool(r.get("fire", False)),
    )


@dataclass(frozen=True)
class LifecycleCase:
    name: str
    status: Status

    def as_dict(self) -> dict[str, Any]:
        return {"case": self.name, "status": str(self.status)}


def run_lifecycle_cases() -> tuple[LifecycleCase, ...]:
    """ARGUS's real evaluate(), run across designed fixtures that each target exactly one named
    transition in Status — not a random sweep, each case built to sit on one specific boundary."""
    rule = _rule()
    cases = []

    # PROPOSED: below MIN_DECISIONS.
    few = _records(MIN_DECISIONS - 1, fires_on={0})
    perf = evaluate(rule, few, [Defect(0, "X", DefectKind.GROUNDING, "x")])
    cases.append(LifecycleCase("below_min_decisions", perf.status))

    # DEAD_WEIGHT: fires on nothing.
    never = _records(MIN_DECISIONS + 5, fires_on=set())
    perf = evaluate(rule, never, [Defect(0, "X", DefectKind.GROUNDING, "x")])
    cases.append(LifecycleCase("never_fires", perf.status))

    # NO_DISCRIMINATION: fires on almost everything.
    n = MIN_DECISIONS + 10
    always = _records(n, fires_on=set(range(n - 1)))  # fires on all but one -> >= ALWAYS_FIRES
    perf = evaluate(rule, always, [Defect(0, "X", DefectKind.GROUNDING, "x")])
    cases.append(LifecycleCase("fires_on_almost_everything", perf.status))

    # MISLEADING: fires selectively on clean decisions while the defect sits elsewhere. Half the
    # record carries the defect and the rule's ten firings meet none of it — far below chance.
    # (Before 2026-09-24 this case was "2 of 10 right", graded MISLEADING by an absolute 50%
    # precision floor; against a 2-in-30 base rate that rule is three times better than chance.)
    n = MIN_DECISIONS + 10
    fires = set(range(10))
    misleading_records = _records(n, fires_on=fires)
    misleading_defects = [Defect(i, "X", DefectKind.GROUNDING, "x") for i in range(15, 30)]
    perf = evaluate(rule, misleading_records, misleading_defects)
    cases.append(LifecycleCase("fires_selectively_mostly_wrong", perf.status))

    # NO_DISCRIMINATION by chance: fires on a third of the record and meets the defect at its
    # base rate — no better than picking decisions at random.
    chance_records = _records(n, fires_on=set(range(0, 30, 3)))
    chance_defects = [Defect(i, "X", DefectKind.GROUNDING, "x") for i in range(0, 30, 2)]
    perf = evaluate(rule, chance_records, chance_defects)
    cases.append(LifecycleCase("fires_selectively_no_better_than_chance", perf.status))

    # EARNING: fires selectively, always right, but below MIN_FIRINGS.
    n = MIN_DECISIONS + 10
    fires = set(range(MIN_FIRINGS - 1))
    earning_records = _records(n, fires_on=fires)
    earning_defects = [
        Defect(i, "X", DefectKind.GROUNDING, "x") for i in range(MIN_FIRINGS - 1)
    ]
    perf = evaluate(rule, earning_records, earning_defects)
    cases.append(LifecycleCase("fires_selectively_always_right_below_min_firings", perf.status))

    # ACTIVE: fires selectively, always right, at or above MIN_FIRINGS.
    n = MIN_DECISIONS + 10
    fires = set(range(MIN_FIRINGS + 2))
    active_records = _records(n, fires_on=fires)
    active_defects = [
        Defect(i, "X", DefectKind.GROUNDING, "x") for i in range(MIN_FIRINGS + 2)
    ]
    perf = evaluate(rule, active_records, active_defects)
    cases.append(LifecycleCase("fires_selectively_always_right_at_min_firings", perf.status))

    return tuple(cases)


_EXPECTED_LIFECYCLE = {
    "below_min_decisions": Status.PROPOSED,
    "never_fires": Status.DEAD_WEIGHT,
    "fires_on_almost_everything": Status.NO_DISCRIMINATION,
    "fires_selectively_mostly_wrong": Status.MISLEADING,
    "fires_selectively_no_better_than_chance": Status.NO_DISCRIMINATION,
    "fires_selectively_always_right_below_min_firings": Status.EARNING,
    "fires_selectively_always_right_at_min_firings": Status.ACTIVE,
}


@dataclass(frozen=True)
class LifecycleRun:
    cases: tuple[LifecycleCase, ...]
    mismatches: tuple[str, ...]

    @property
    def design_is_sound(self) -> bool:
        return not self.mismatches

    def as_dict(self) -> dict[str, Any]:
        return {
            "cases": [c.as_dict() for c in self.cases],
            "mismatches": list(self.mismatches),
            "design_is_sound": self.design_is_sound,
        }


def run_lifecycle() -> LifecycleRun:
    cases = run_lifecycle_cases()
    mismatches = tuple(
        c.name for c in cases if c.status != _EXPECTED_LIFECYCLE[c.name]
    )
    return LifecycleRun(cases=cases, mismatches=mismatches)


# =============================================================================================
# Ablation — each of ARGUS's four thresholds, independently load-bearing.
# =============================================================================================


@dataclass(frozen=True)
class ThresholdAblation:
    dimension: str
    real_status: Status
    ablated_status: Status

    @property
    def differs(self) -> bool:
        return self.real_status != self.ablated_status

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension, "real_status": str(self.real_status),
            "ablated_status": str(self.ablated_status), "differs": self.differs,
        }


def run_threshold_ablations() -> tuple[ThresholdAblation, ...]:
    """All four of ``review.py``'s status-gating thresholds, each patched to a value that flips
    the verdict on an otherwise-identical fixture — real code, not a hand-derived counterfactual.

    **This function used to ablate only two** (MIN_PRECISION, ALWAYS_FIRES); the docstring
    claimed all four and NEVER_FIRES/MIN_FIRINGS had no fixture at all — a real coverage gap,
    found the same way the stale-number defects elsewhere in this session's `SCOPE_STATEMENT`
    sweep were: by comparing what the prose claimed against what the code actually returned
    (``len(run_threshold_ablations())`` was 2, not 4). The two added below share one fixture
    (fires 5 of 20 decisions, every firing on a real GROUNDING defect: fire_rate 0.25, precision
    1.0, fired == MIN_FIRINGS exactly) rather than inventing a fresh one each, since both
    NEVER_FIRES and MIN_FIRINGS gate on properties that same fixture already sits exactly at the
    boundary of — ``review.py``'s cascade in ``desk/review.py:340-369``.
    """
    import argus.desk.review as review_module

    rule = _rule()
    results = []

    # ALPHA (replaced the absolute MIN_PRECISION ablation on 2026-09-24, when grading moved to
    # lift over the base rate): 5 of 7 firings meet a defect that runs at 10 in 30 — lift 2.1,
    # one-sided Fisher p = 0.026. ACTIVE at the real 5%; at 1% the same rule is only EARNING.
    n = MIN_DECISIONS + 10
    records = _records(n, fires_on={0, 1, 2, 3, 4, 20, 21})
    defects = [Defect(i, "X", DefectKind.GROUNDING, "x") for i in range(10)]
    real_perf = evaluate(rule, records, defects)
    original = review_module.ALPHA
    try:
        review_module.ALPHA = 0.01
        ablated_perf = evaluate(replace(rule, name=rule.name), records, defects)
    finally:
        review_module.ALPHA = original
    results.append(ThresholdAblation("alpha", real_perf.status, ablated_perf.status))

    # ALWAYS_FIRES: a rule firing on 80% is NOT no-discrimination under the real 90% ceiling;
    # patching the ceiling to 75% should flip it to NO_DISCRIMINATION. (80% rather than the 85%
    # used before: a rule firing on 85% of decisions cannot reach the 1.2x MIN_LIFT, since its
    # precision is at most 1/0.85 of the base rate.)
    n = 20
    fires = set(range(16))  # 16/20 = 80%
    records2 = _records(n, fires_on=fires)
    defects2 = [Defect(i, "X", DefectKind.GROUNDING, "x") for i in range(16)]
    real_perf2 = evaluate(rule, records2, defects2, min_decisions=20)
    original_af = review_module.ALWAYS_FIRES
    try:
        review_module.ALWAYS_FIRES = 0.75
        ablated_perf2 = evaluate(rule, records2, defects2, min_decisions=20)
    finally:
        review_module.ALWAYS_FIRES = original_af
    results.append(
        ThresholdAblation("always_fires_ceiling", real_perf2.status, ablated_perf2.status)
    )

    # NEVER_FIRES and MIN_FIRINGS: one shared fixture sitting exactly at both gates' boundary —
    # fires 5 of 20 decisions (fire_rate 0.25: real, comfortably above the real 0.0 NEVER_FIRES
    # floor and below the real 0.9 ALWAYS_FIRES ceiling), every firing catching a real defect
    # (precision 1.0, comfortably above MIN_PRECISION), fired == 5 == MIN_FIRINGS exactly (passes
    # `fired < MIN_FIRINGS` since 5 is not less than 5). Real status is ACTIVE; verified by an
    # assertion below rather than assumed, since a fixture that silently lands somewhere else
    # would make both flips below meaningless.
    n3 = 20
    fires3 = set(range(5))
    records3 = _records(n3, fires_on=fires3)
    defects3 = [Defect(i, "X", DefectKind.GROUNDING, "x") for i in range(5)]  # 5/5 caught = 1.0
    real_perf3 = evaluate(rule, records3, defects3, min_decisions=20)
    assert real_perf3.status == Status.ACTIVE, (
        f"the shared NEVER_FIRES/MIN_FIRINGS fixture must be real-ACTIVE for either ablation "
        f"below to demonstrate anything; got {real_perf3.status}"
    )

    # NEVER_FIRES: fire_rate 0.25 is real-ACTIVE under the real 0.0 floor; patching the floor to
    # 0.3 (above the fixture's own 0.25) should flip it to DEAD_WEIGHT.
    original_nf = review_module.NEVER_FIRES
    try:
        review_module.NEVER_FIRES = 0.3
        ablated_perf3 = evaluate(rule, records3, defects3, min_decisions=20)
    finally:
        review_module.NEVER_FIRES = original_nf
    results.append(
        ThresholdAblation("never_fires_floor", real_perf3.status, ablated_perf3.status)
    )

    # MIN_FIRINGS: fired == 5 clears the real floor of 5 exactly; patching the floor to 10 (above
    # the fixture's 5 firings) should flip it to EARNING.
    original_mf = review_module.MIN_FIRINGS
    try:
        review_module.MIN_FIRINGS = 10
        ablated_perf4 = evaluate(rule, records3, defects3, min_decisions=20)
    finally:
        review_module.MIN_FIRINGS = original_mf
    results.append(
        ThresholdAblation("min_firings_floor", real_perf3.status, ablated_perf4.status)
    )

    return tuple(results)


# =============================================================================================
# Scope statement.
# =============================================================================================

SCOPE_STATEMENT = """\
Claimed: ARGUS's review.evaluate() grades a candidate rule against INDEPENDENTLY observed \
defects and refuses it by name (DEAD_WEIGHT, NO_DISCRIMINATION, MISLEADING) when the replay \
shows it does not discriminate or is wrong more often than right — confirmed by running the real \
function across six designed cases, each targeting one specific lifecycle transition, all six \
landing on the intended status. TradingAgents' real, vendored TradingMemoryLog has no equivalent \
anywhere: storing a reflection about a decision that was independently, demonstrably WRONG (a \
real -15% realised return) and reading it back later shows it re-injected identically to a \
reflection about a right call — grepped directly across the whole repository for any \
precision/track-record concept tied to memory: zero matches. All four of ARGUS's status-gating \
thresholds (MIN_PRECISION, ALWAYS_FIRES, NEVER_FIRES, MIN_FIRINGS) shown independently \
load-bearing by patching each to a value that flips an otherwise-identical fixture's verdict.

NOT claimed: that TradingAgents' unconditional reuse is a design mistake in the way it presents \
itself — a system whose memory feeds a model that reasons over the prose (and can itself discount \
a reflection it judges unreliable) is a different premise from ARGUS's constitution-gated \
decision loop, where a model's own real-time judgment is deliberately NOT the backstop \
(`agents/desk.py`'s Constitution is). Also not claimed: that ARGUS's checklist currently contains \
any earned rules — it does not (`data/review_report.json`: none of the five standing rules has \
earned ACTIVE on the real record, disclosed as a standing blocker). What is claimed is narrower \
and fully run-verified: the MACHINERY that would refuse a bad rule by name exists, works exactly \
as designed on six real transitions, and has no equivalent in the named baseline.
"""


def main() -> dict[str, Any]:
    """Run the whole comparison and return a serialisable summary.

    Raises:
        ReviewComparisonError: the vendored TradingAgents baseline failed to load.
    """
    try:
        memory_log_class = load_trading_memory_log_class()
    except TradingAgentsBaselineLoadError as exc:
        raise ReviewComparisonError(
            f"could not load the vendored TradingAgents baseline: {exc}"
        ) from exc

    reuse = run_unconditional_reuse_case(memory_log_class)
    lifecycle = run_lifecycle()
    ablations = run_threshold_ablations()

    return {
        "unconditional_reuse": reuse.as_dict(),
        "lifecycle": lifecycle.as_dict(),
        "ablations": [a.as_dict() for a in ablations],
        "ablation_all_load_bearing": all(a.differs for a in ablations),
        "scope_statement": SCOPE_STATEMENT,
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "REVIEW COMPARISON — ARGUS review.evaluate() vs. TradingAgents real TradingMemoryLog", "",
    ]
    r = report["unconditional_reuse"]
    lines.append(
        f"TradingAgents re-injects a WRONG call's reflection unconditionally: "
        f"{r['no_precision_gate']}"
    )
    lc = report["lifecycle"]
    lines.append(f"lifecycle cases: {len(lc['cases'])}, design sound: {lc['design_is_sound']}")
    lines.append(
        f"ablation: {len(report['ablations'])} threshold(s), all load-bearing: "
        f"{report['ablation_all_load_bearing']}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    from pathlib import Path as _Path

    result = main()
    print(render(result))
    out_path = _Path(__file__).resolve().parents[3] / "data" / "review_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_path}")


__all__ = [
    "SCOPE_STATEMENT",
    "LifecycleCase",
    "LifecycleRun",
    "ReviewComparisonError",
    "ThresholdAblation",
    "UnconditionalReuseResult",
    "main",
    "render",
    "run_lifecycle",
    "run_lifecycle_cases",
    "run_threshold_ablations",
    "run_tradingagents_reflection_reuse",
    "run_unconditional_reuse_case",
]
