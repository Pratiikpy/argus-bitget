"""desk/rule_lifecycle.py: admission, probation, CUSUM retirement and re-admission, on planted data.

Each test builds a stream whose truth is known (which rule really predicts the defect, and when it
stops) and checks the stage the lifecycle reaches, so the claims in the module docstring are pinned
to behaviour rather than to prose.
"""

from __future__ import annotations

import random

import pytest

from argus.desk import rule_lifecycle as rl
from argus.desk.review import DefectKind, Status
from argus.desk.rule_lifecycle import (
    BASE_WINDOW,
    PROBATION_FIRINGS,
    Candidate,
    Lifecycle,
    SetEvidence,
    Stage,
    candidates_from,
    grade,
    local_grade,
    parameters,
)

KIND = DefectKind.CONFLICT
TARGETS = frozenset({KIND})


def _evidence(observed: range | list[int], marked: set[int]) -> SetEvidence:
    seen = frozenset(observed)
    return SetEvidence(eligible_by_kind={KIND: seen}, marked_by_kind={KIND: frozenset(marked)},
                       everything=seen)


def _planted(n: int, *, seed: int, base: float, lift: float, fire: float,
             decays_at: int | None = None, base_after: float | None = None,
             change_at: int | None = None) -> tuple[set[int], set[int]]:
    """(firings, defects) for one rule on ``n`` decisions."""
    rng = random.Random(seed)
    fires: set[int] = set()
    marked: set[int] = set()
    for i in range(n):
        b = base_after if base_after is not None and change_at is not None and i >= change_at \
            else base
        p = b
        if rng.random() < fire:
            fires.add(i)
            if decays_at is None or i < decays_at:
                p = min(0.95, b * lift)
        if rng.random() < p:
            marked.add(i)
    return fires, marked


def _run(lifecycle: Lifecycle, n: int, marked: set[int], step: int = 20) -> list[Stage]:
    stages = []
    for upto in range(step, n + 1, step):
        lifecycle.step(_evidence(range(upto), marked), at=f"d{upto}")
        stages.append(lifecycle.stage_of(next(iter(lifecycle.tracks))))
    return stages


def test_names_must_be_unique() -> None:
    c = Candidate("a", TARGETS, frozenset())
    with pytest.raises(ValueError, match="unique"):
        Lifecycle([c, c])


def test_grade_counts_only_decisions_after_the_floor() -> None:
    cand = Candidate("r", TARGETS, frozenset({1, 2, 3, 30, 31}))
    perf = grade(cand, frozenset(range(40)), frozenset({1, 2, 30}), floor=10)
    assert perf.decisions == 29
    assert perf.fired == 2
    assert perf.caught == 1


def test_a_real_rule_is_admitted_then_confirmed_on_fresh_decisions() -> None:
    fires, marked = _planted(1200, seed=3, base=0.3, lift=2.2, fire=0.2)
    life = Lifecycle([Candidate("real", TARGETS, frozenset(fires))])
    _run(life, 1200, marked)
    kinds = [(t.source, t.target) for t in life.transitions]
    assert (Stage.CANDIDATE, Stage.PROBATION) in kinds
    assert (Stage.PROBATION, Stage.ACTIVE) in kinds
    assert life.stage_of("real") is Stage.ACTIVE
    assert life.warning(include_probation=False) == frozenset({"real"})


def test_probation_never_counts_the_decisions_that_admitted_it() -> None:
    fires, marked = _planted(400, seed=5, base=0.3, lift=2.5, fire=0.25)
    life = Lifecycle([Candidate("real", TARGETS, frozenset(fires))])
    for upto in range(20, 401, 20):
        life.step(_evidence(range(upto), marked), at=str(upto))
        track = life.tracks["real"]
        if track.stage is Stage.PROBATION:
            admitted = track.admitted_on
            life.step(_evidence(range(upto), marked), at="same evidence again")
            # nothing new observed, so the probation cannot have been judged
            assert life.stage_of("real") is Stage.PROBATION
            assert life.tracks["real"].admitted_on == admitted
            return
    pytest.fail("the planted rule was never admitted")


def test_a_rule_that_stops_working_is_retired_by_the_cusum() -> None:
    fires, marked = _planted(3000, seed=11, base=0.3, lift=2.2, fire=0.25, decays_at=1000)
    life = Lifecycle([Candidate("decays", TARGETS, frozenset(fires))])
    _run(life, 3000, marked)
    decayed = [t for t in life.transitions if t.target is Stage.RETIRED
               and t.reason.startswith("decayed")]
    assert decayed, [t.as_dict() for t in life.transitions]
    assert life.stage_of("decays") is not Stage.ACTIVE


def _falsely_decayed(seeds: range, **planted: float | int) -> int:
    """How many seeds end with a "decayed" retirement of a rule that never lost its lift."""
    count = 0
    for seed in seeds:
        fires, marked = _planted(3000, seed=seed, **planted)  # type: ignore[arg-type]
        life = Lifecycle([Candidate("steady", TARGETS, frozenset(fires))])
        _run(life, 3000, marked)
        count += any(t.reason.startswith("decayed") for t in life.transitions)
    return count


def test_a_falling_base_rate_adds_at_most_one_false_retirement_in_twelve() -> None:
    """Measured 2026-09-26. With the base rate read from the 100 decisions *before* each firing,
    a fall from 50% to 20% retired a rule that kept its 1.8x lift in 12 of 12 seeds. Read from the
    centred window it is 6 of 12, against the monitor's own false-alarm rate on a stable lift-2.2
    rule with no shift at all of 5 of 12 (seeds 100-111 and 300-311). Both counts are pinned, the
    stable one included, because it is a weakness the module docstring states rather than hides."""
    shifted = _falsely_decayed(range(100, 112), base=0.5, lift=1.8, fire=0.25, base_after=0.2,
                               change_at=1000)
    stable = _falsely_decayed(range(300, 312), base=0.3, lift=2.2, fire=0.25)
    assert (shifted, stable) == (6, 5)
    assert shifted <= stable + 1


def test_a_firing_is_not_scored_until_half_a_window_after_it_is_observed() -> None:
    """The centred base rate needs decisions after the firing; until they are observed the firing
    waits, and nothing about it can retire the rule."""
    life = Lifecycle([Candidate("r", TARGETS, frozenset({500}))])
    track = life.tracks["r"]
    track.stage = Stage.ACTIVE
    life.step(_evidence(range(540), set(range(0, 540, 3))), at="49 decisions after")
    assert 500 not in track.monitored
    life.step(_evidence(range(560), set(range(0, 560, 3))), at="59 decisions after")
    assert 500 in track.monitored


def test_a_null_rule_is_not_left_warning() -> None:
    fires, marked = _planted(1500, seed=23, base=0.3, lift=1.0, fire=0.2)
    life = Lifecycle([Candidate("null", TARGETS, frozenset(fires))])
    _run(life, 1500, marked)
    assert life.stage_of("null") in {Stage.CANDIDATE, Stage.RETIRED}


def test_a_duplicate_of_an_admitted_rule_is_not_admitted() -> None:
    fires, marked = _planted(600, seed=29, base=0.3, lift=2.5, fire=0.25)
    life = Lifecycle([Candidate("first", TARGETS, frozenset(fires)),
                      Candidate("copy", TARGETS, frozenset(fires))])
    for upto in range(20, 601, 20):
        life.step(_evidence(range(upto), marked), at=str(upto))
    warning = life.warning()
    assert len(warning) == 1, warning


def test_retirement_resets_the_floor_so_readmission_uses_fresh_evidence() -> None:
    fires, marked = _planted(3000, seed=11, base=0.3, lift=2.2, fire=0.25, decays_at=1000)
    life = Lifecycle([Candidate("decays", TARGETS, frozenset(fires))])
    _run(life, 3000, marked)
    track = life.tracks["decays"]
    if track.retirements:
        retire = next(t for t in life.transitions if t.target is Stage.RETIRED)
        assert track.floor >= int(retire.at[1:]) - 20 - 1


def test_candidates_from_tolerates_a_raising_predicate() -> None:
    def boom(record: dict[str, object]) -> bool:
        raise RuntimeError("broken")

    records = [{"seq": 1, "x": 1}, {"seq": 2, "x": 0}]
    out = candidates_from([("boom", TARGETS, boom),
                           ("x", TARGETS, lambda r: bool(r["x"]))], records)
    assert out[0].fires == frozenset()
    assert out[1].fires == frozenset({1})


def test_parameters_report_the_fixed_thresholds() -> None:
    fixed = parameters()
    assert fixed["probation_firings"] == PROBATION_FIRINGS
    assert parameters(Lifecycle([])) == fixed


def test_set_evidence_grades_a_multi_kind_rule_on_everything() -> None:
    ev = SetEvidence(eligible_by_kind={KIND: frozenset({1}), DefectKind.LEAN: frozenset({2})},
                     marked_by_kind={KIND: frozenset({1}), DefectKind.LEAN: frozenset({2})},
                     everything=frozenset({1, 2, 3}))
    both = frozenset({KIND, DefectKind.LEAN})
    assert ev.eligible(both) == frozenset({1, 2, 3})
    assert ev.marked(both) == frozenset({1, 2})
    assert ev.eligible(TARGETS) == frozenset({1})


# --- local_grade: rolling base rate, not the one pooled over the whole window ---------------------


def _diluted_fixture() -> tuple[Candidate, frozenset[int], frozenset[int]]:
    """A rule that only ever fires in a rare-defect regime (base 5%, its own local lift 3x), whose
    graded window also covers a later, unrelated common-defect regime (base 60%) it never touches.
    Pooling the base rate over the whole window buries the rule's real local lift under the second
    regime's; a rolling local rate does not. Numbers pinned 2026-09-28."""
    rng = random.Random(7)
    n = 1000
    fires: set[int] = set()
    marked: set[int] = set()
    for i in range(n):
        if i < 500:
            base = 0.05
            if rng.random() < 0.4:
                fires.add(i)
                base = 0.05 * 3.0
        else:
            base = 0.6
        if rng.random() < min(0.95, base):
            marked.add(i)
    return Candidate("r", TARGETS, frozenset(fires)), frozenset(range(n)), frozenset(marked)


def test_local_grade_admits_a_rule_a_pooled_grade_refuses() -> None:
    """The defect this fix targets, measured 2026-09-28: admission's own note reads a pooled 71%
    base rate and a 1.2x lift on the real S3 fixture (`eval/review_rivals.py::suite_base_shift`);
    this is the same failure in miniature, with a hand-checked cause and a hand-checked cure."""
    cand, eligible, marked = _diluted_fixture()
    pooled = grade(cand, eligible, marked)
    local = local_grade(cand, eligible, marked)
    assert pooled.status in (Status.MISLEADING, Status.NO_DISCRIMINATION), pooled.note
    assert local.status is Status.ACTIVE, local.note
    assert (local.lift or 0.0) > (pooled.lift or 0.0)
    assert local.base_rate is not None and local.base_rate < 0.2  # tracks the 5% regime, not 71%


def test_local_rate_needs_local_min_decisions_on_both_sides() -> None:
    wide = list(range(0, 200, 2))  # 100 decisions: plenty of history and future either side.
    marked = frozenset(wide[::4])
    assert rl._local_rate(wide, marked, 50, half=20, local_min=20) is not None
    narrow = list(range(0, 30, 2))  # 15 decisions: even the middle can't find 20 neighbours.
    assert rl._local_rate(narrow, frozenset(narrow[::4]), 7, half=20, local_min=20) is None


def test_local_rate_refuses_a_degenerate_neighbourhood() -> None:
    window = list(range(100))
    assert rl._local_rate(window, frozenset(), 50, half=20, local_min=5) is None  # base 0
    assert rl._local_rate(window, frozenset(window), 50, half=20, local_min=5) is None  # base 1


def test_sir_upper_tail_is_none_without_variance() -> None:
    assert rl._sir_upper_tail(5.0, 5.0, 0.0) is None


def test_sir_upper_tail_rewards_more_catches_than_expected() -> None:
    small = rl._sir_upper_tail(9.0, 5.0, 2.5)
    big = rl._sir_upper_tail(5.0, 5.0, 2.5)
    assert small is not None and big is not None
    assert small < 0.05 < big


def test_local_grade_respects_its_own_base_window_argument() -> None:
    """``local_grade`` is not wired into :class:`Lifecycle` (see its docstring: pre-registered,
    measured, reverted 2026-09-28) but is kept, tested and documented on its own — pinned here by
    showing two window sizes disagree on the diluted fixture."""
    cand, eligible, marked = _diluted_fixture()
    narrow = local_grade(cand, eligible, marked, base_window=20)
    wide = local_grade(cand, eligible, marked, base_window=BASE_WINDOW)
    assert narrow.status is not Status.ACTIVE
    assert wide.status is Status.ACTIVE


def test_lifecycle_does_not_call_local_grade(monkeypatch: pytest.MonkeyPatch) -> None:
    """The revert, pinned directly: `Lifecycle._grade` calls `grade`, never `local_grade`."""
    calls: list[str] = []
    real_grade = rl.grade

    def spy(*args: object, **kwargs: object) -> object:
        calls.append("grade")
        return real_grade(*args, **kwargs)  # type: ignore[arg-type]

    def boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("Lifecycle must not call local_grade")

    monkeypatch.setattr(rl, "grade", spy)
    monkeypatch.setattr(rl, "local_grade", boom)
    fires, marked = _planted(200, seed=1, base=0.3, lift=2.0, fire=0.2)
    life = Lifecycle([Candidate("r", TARGETS, frozenset(fires))])
    _run(life, 200, marked)
    assert calls  # grade() was actually exercised, not just imported
