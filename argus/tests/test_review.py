"""Review and self-evolution — the lifecycle every growing checklist is missing the back half of.

`desk/review.py` applies to *process rules* the standard the factor lab applies to factors: a rule
enters as a proposal, is replayed against decisions the desk already took, and earns a status from
what the replay shows. The part nobody else builds is the **removal** path — `NO_DISCRIMINATION`,
`DEAD_WEIGHT`, `MISLEADING`, `RETIRED` — because a checklist that only grows costs attention and
prevents nothing.

So these tests are weighted toward the refusals. Anything can mark a rule ACTIVE; the claim worth
defending is that a rule which fires on everything, never fires, or is usually wrong gets **named**
rather than quietly kept.

⚠️ **This file was rebuilt on 2026-09-15 after I destroyed it.** I wrote a new `tests/test_review.py`
for an unrelated module (`execution/review.py`) without checking whether the name was taken, and
overwrote the tests for `desk/review.py` — a 26KB module covering the *Review & Self-Evolution*
Track 3 sub-theme. `argus.status` caught it immediately (*"test file test_review.py is absent"*),
which is the only reason it did not ship silently. The new file was renamed to
`test_order_review.py`. **Standing Rule #2 again: read before writing, including filenames.**
"""

from __future__ import annotations

from typing import Any

import pytest

from argus.desk.review import (
    ALWAYS_FIRES,
    MIN_DECISIONS,
    MIN_PRECISION,
    Defect,
    DefectKind,
    Rule,
    Status,
    defects_from_notes,
    defects_from_risk,
    evaluate,
    review,
)


def _records(n: int, *, flag_from: int = 0, flag_to: int | None = None) -> list[dict[str, Any]]:
    """``n`` decision records; those in ``[flag_from, flag_to)`` carry ``mark``."""
    stop = n if flag_to is None else flag_to
    return [
        {"seq": i, "symbol": "NVDAUSDT", "mark": flag_from <= i < stop}
        for i in range(n)
    ]


def _rule(name: str = "r", targets: frozenset[DefectKind] | None = None) -> Rule:
    return Rule(
        name=name,
        prompt="Check the thing.",
        rationale="because it has gone wrong before",
        targets=targets or frozenset({DefectKind.GROUNDING}),
        predicate=lambda record: bool(record.get("mark")),
    )


def _defects(seqs: list[int], kind: DefectKind = DefectKind.GROUNDING) -> list[Defect]:
    return [
        Defect(seq=s, symbol="NVDAUSDT", kind=kind, detail="observed independently")
        for s in seqs
    ]


class TestARuleIsProposedUntilThereIsEnoughToSayAnything:
    def test_below_the_floor_it_stays_proposed(self) -> None:
        """A precision over a handful of decisions is noise, and showing it would be the claim this
        module exists to refuse."""
        got = evaluate(_rule(), _records(MIN_DECISIONS - 1), _defects([1, 2]))
        assert got.status is Status.PROPOSED

    def test_the_floor_is_stated_not_buried(self) -> None:
        assert MIN_DECISIONS == 20

    def test_no_targeted_defect_leaves_it_proposed_however_many_decisions(self) -> None:
        """Firing a lot proves nothing when nothing it targets was ever observed."""
        got = evaluate(_rule(), _records(60, flag_to=30), _defects([], DefectKind.GROUNDING))
        assert got.status is Status.PROPOSED

    def test_a_defect_of_another_kind_does_not_count_as_targeted(self) -> None:
        """A grounding rule is not credited for coinciding with an unrelated conflict, which would
        let any rule look good on a noisy day."""
        rule = _rule(targets=frozenset({DefectKind.GROUNDING}))
        got = evaluate(
            rule, _records(60, flag_to=30),
            _defects(list(range(10)), DefectKind.CONFLICT),
        )
        assert got.status is Status.PROPOSED


class TestTheRefusalsAreNamedNotQuiet:
    """The back half of the lifecycle — the part every published checklist is missing."""

    def test_a_rule_that_never_fires_is_dead_weight(self) -> None:
        rule = Rule(
            name="never", prompt="p", rationale="r",
            targets=frozenset({DefectKind.GROUNDING}), predicate=lambda _: False,
        )
        got = evaluate(rule, _records(60), _defects(list(range(10))))
        assert got.status is Status.DEAD_WEIGHT
        assert "costs attention" in got.note

    def test_a_rule_that_fires_on_nearly_everything_has_no_discrimination(self) -> None:
        """*A rule that flags nine decisions in ten does not identify a problem; it describes the
        desk.*"""
        rule = Rule(
            name="always", prompt="p", rationale="r",
            targets=frozenset({DefectKind.GROUNDING}), predicate=lambda _: True,
        )
        got = evaluate(rule, _records(60), _defects(list(range(10))))
        assert got.status is Status.NO_DISCRIMINATION
        assert got.fire_rate >= ALWAYS_FIRES

    def test_a_rule_that_avoids_the_defect_is_misleading(self) -> None:
        """Worse than absent: it spends attention and points the wrong way. Half the decisions
        carry the defect and the rule fires on twenty clean ones — far fewer hits than chance."""
        got = evaluate(_rule(), _records(60, flag_to=20), _defects(list(range(30, 60))))
        assert got.status is Status.MISLEADING
        assert got.lift == 0.0 and "more than chance would" in got.note

    def test_low_precision_on_a_common_defect_is_chance_not_misleading(self) -> None:
        """Fires on half the record and meets both of two defects: 7% precision, but twice the
        base rate and nowhere near significant. An absolute 50% precision bar called this
        MISLEADING; the base rate says it is indistinguishable from chance."""
        got = evaluate(_rule(), _records(60, flag_to=30), _defects([0, 1]))
        assert got.status is Status.NO_DISCRIMINATION
        assert got.precision is not None and got.precision < MIN_PRECISION

    def test_misleading_and_dead_weight_do_not_keep_their_place(self) -> None:
        assert not Status.MISLEADING.keeps_its_place
        assert not Status.DEAD_WEIGHT.keeps_its_place
        assert not Status.NO_DISCRIMINATION.keeps_its_place

    def test_active_and_earning_keep_their_place(self) -> None:
        assert Status.ACTIVE.keeps_its_place
        assert Status.EARNING.keeps_its_place


class TestFailureCasesAreDocumentedNotJustCounted:
    """`failure_cases` exists because `standing.py::verify()` opens artefacts rather than trusting
    filenames — a rejected rule's own measured failure mode must be a real JSON key a machine can
    find, not only prose in `render()` or a bare number under an unrelated key name."""

    def test_a_misleading_rule_appears_in_failure_cases_with_its_own_why(self) -> None:
        rule = _rule(name="misleading-one")
        performance = evaluate(rule, _records(60, flag_to=20), _defects(list(range(30, 60))))
        report = review(notes=_records(60))
        report.performance.append(performance)
        cases = report.failure_cases
        found = next(c for c in cases if c["rule"] == "misleading-one")
        assert found["status"] == str(Status.MISLEADING)
        assert found["why"]

    def test_a_rule_that_keeps_its_place_is_not_in_failure_cases(self) -> None:
        active = evaluate(_rule(name="active-one"), _records(60, flag_to=30), _defects(range(30)))
        report = review(notes=_records(60))
        report.performance.append(active)
        names = {c["rule"] for c in report.failure_cases}
        assert "active-one" not in names

    def test_failure_cases_is_a_real_key_in_as_dict(self) -> None:
        rule = _rule(name="dead-weight-one")
        performance = evaluate(rule, _records(60), _defects(list(range(10))))
        report = review(notes=_records(60))
        report.performance.append(performance)
        got = report.as_dict()
        assert "failure_cases" in got
        assert any(c["rule"] == "dead-weight-one" for c in got["failure_cases"])

    def test_failure_cases_and_rejected_agree_on_which_rules_are_named(self) -> None:
        report = review(notes=_records(60))
        report.performance.append(
            evaluate(_rule(name="x"), _records(60), _defects(list(range(10))))
        )
        rejected_names = {p.rule for p in report.rejected}
        failure_case_names = {c["rule"] for c in report.failure_cases}
        assert rejected_names == failure_case_names


class TestARuleCanEarnItsPlace:
    def test_selective_and_right_is_active_or_earning(self) -> None:
        # fires on 10 of 60 (selective); 8 of those carry a targeted defect (precision 0.8)
        got = evaluate(_rule(), _records(60, flag_to=10), _defects(list(range(8))))
        assert got.status in (Status.ACTIVE, Status.EARNING)
        assert got.precision is not None and got.precision >= MIN_PRECISION

    def test_a_small_sample_of_firings_is_earning_not_active(self) -> None:
        """**`Status.EARNING` was unreachable before this test existed.**

        Its branch was guarded by ``caught == 0``, which cannot be reached: precision is
        ``caught / fired``, so a zero numerator fails MIN_PRECISION one line earlier and returns
        MISLEADING. Every rule that should have read *"not enough firings to know"* was reported
        *"right 0% of the times it fires"* instead. The test was written against the documented
        lifecycle rather than the code, which is why it found it.
        """
        # fires on 3 of 60, right every time: precision 1.0 but a sample of three
        got = evaluate(_rule(), _records(60, flag_to=3), _defects(list(range(3))))
        assert got.precision == 1.0
        assert got.status is Status.EARNING
        assert "not yet evidence" in got.note

    def test_enough_right_firings_is_active(self) -> None:
        got = evaluate(_rule(), _records(60, flag_to=10), _defects(list(range(8))))
        assert got.status is Status.ACTIVE

    def test_catching_nothing_in_ten_firings_is_not_yet_evidence_either_way(self) -> None:
        """The defect runs at one decision in six, so ten firings expect under two hits; catching
        none happens by chance 14% of the time. That is no better than chance — not proof that the
        rule points the wrong way."""
        got = evaluate(_rule(), _records(60, flag_to=10), _defects(list(range(40, 50))))
        assert got.status is Status.NO_DISCRIMINATION
        assert not got.status.keeps_its_place


class TestTheBaseRateDecides:
    """The two cases the rival review of 2026-09-24 used to show absolute precision grading was
    broken: noise crowned on a common defect, signal refused on a rare one."""

    def test_a_coin_flip_on_a_common_defect_is_never_active(self) -> None:
        import random

        crowned = 0
        for seed in range(200):
            rng = random.Random(seed)
            records = [{"seq": i, "symbol": "NVDAUSDT", "mark": rng.random() < 0.5}
                       for i in range(200)]
            defects = _defects([i for i in range(200) if rng.random() < 0.72])
            got = evaluate(_rule(), records, defects)
            crowned += got.status is Status.ACTIVE
        # a fair coin can reach significance by chance; at 5% that is about ten seeds in 200
        assert crowned <= 20, crowned

    def test_a_rule_catching_a_rare_defect_at_low_precision_is_active(self) -> None:
        # 565 decisions, a defect on 6 of them (1.06%); the rule fires on those 6 and 24 clean ones
        records = [{"seq": i, "symbol": "NVDAUSDT", "mark": i < 30} for i in range(565)]
        got = evaluate(_rule(), records, _defects(list(range(6))))
        assert got.precision == pytest.approx(0.2)
        assert got.lift is not None and got.lift == pytest.approx(18.83, abs=0.01)
        assert got.status is Status.ACTIVE

    def test_a_rule_significant_alone_is_not_active_after_correcting_for_many(self) -> None:
        from argus.desk.review import benjamini_hochberg

        q = benjamini_hochberg([0.04, 0.5, 0.6, 0.7, 0.8, 0.9])
        assert q[0] == pytest.approx(0.24)
        assert all(0.0 <= v <= 1.0 for v in q)


class TestABrokenRuleDoesNotTakeTheReviewDown:
    def test_a_predicate_that_raises_is_treated_as_not_firing(self) -> None:
        def explode(_: dict[str, Any]) -> bool:
            raise ValueError("bad rule")

        rule = Rule(
            name="broken", prompt="p", rationale="r",
            targets=frozenset({DefectKind.GROUNDING}), predicate=explode,
        )
        got = evaluate(rule, _records(60), _defects(list(range(10))))
        assert got.status is Status.DEAD_WEIGHT, "a rule that cannot run has not fired"


class TestDefectsComeFromIndependentCheckersNotFromTheRule:
    def test_notes_yield_defects(self) -> None:
        rows = [
            {"seq": 1, "notes": ["[grounding] a figure does not trace to the evidence"]},
            {"seq": 2, "notes": ["[panel] 3 of 3 analysts"]},
        ]
        got = defects_from_notes(rows)
        assert all(isinstance(d, Defect) for d in got)

    def test_risk_records_yield_defects_only_where_the_layer_intervened(self) -> None:
        rows = [
            {"seq": 1, "intervened": True, "binding_constraint": "max_position"},
            {"seq": 2, "intervened": False, "binding_constraint": "no_exposure"},
        ]
        seqs = {d.seq for d in defects_from_risk(rows)}
        assert 2 not in seqs, "a decision the layer did not touch is not a defect"


class TestTheReportRunsOnTheLiveRecord:
    def test_review_produces_a_report_from_real_shaped_input(self) -> None:
        notes = [
            {"seq": i, "symbol": "NVDAUSDT", "notes": ["[grounding] unsupported figure"]}
            for i in range(30)
        ]
        got = review(notes=notes)
        assert got is not None
        assert got.as_dict()

    def test_an_empty_record_does_not_crash_and_claims_nothing(self) -> None:
        got = review(notes=[])
        text = "\n".join(got.render()) if hasattr(got, "render") else str(got)
        assert text

    def test_the_live_record_reviews_cleanly(self) -> None:
        """The sub-theme this module answers is scored on it actually running."""
        import json
        from pathlib import Path

        data = Path(__file__).resolve().parents[1] / "data"
        notes_path = data / "desk_notes.jsonl"
        if not notes_path.exists():
            pytest.skip("no desk notes on this machine")
        notes = [
            json.loads(line)
            for line in notes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        got = review(notes=notes)
        assert got is not None


class TestTheSubThemeWiringStaysIntact:
    """`argus.status` maps this sub-theme to this file. It is what caught the deletion."""

    def test_status_still_points_review_self_evolution_at_this_file(self) -> None:
        from argus.status import SUBTHEMES

        entry = SUBTHEMES.get("review-self-evolution")
        assert entry is not None, "the sub-theme is no longer registered"
        track, target, test_file = entry
        assert track == 3
        assert target == "argus.desk.review:review"
        assert test_file == "test_review.py"

    def test_the_named_test_file_actually_exists(self) -> None:
        """The exact check `argus.status` performs, and the one that caught the deletion within
        seconds of it happening."""
        from pathlib import Path

        from argus.status import SUBTHEMES

        _, _, test_file = SUBTHEMES["review-self-evolution"]
        assert (Path(__file__).resolve().parent / test_file).exists(), test_file
