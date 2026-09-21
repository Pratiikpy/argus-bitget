"""The review's held-out test: does a rule's grade survive decisions it was not measured on?

The property that matters is not the survival rate — it is that the split cannot leak. These
tests are shaped around the three ways a held-out test quietly stops being one: a random split,
a held-out half graded against the other half's defects, and a threshold relaxed to manufacture
verdicts on the smaller window.
"""

from __future__ import annotations

from typing import Any

import pytest

from argus.eval.reviewoos import ReviewOosError, RuleSplit, run, split_records


def _records(n: int, *, start: int = 1) -> list[dict[str, Any]]:
    return [{"seq": i, "notes": []} for i in range(start, start + n)]


class TestTheSplitCannotLeak:
    def test_it_is_chronological_not_random(self) -> None:
        """A random split grades a rule on a decision that preceded the ones it was fitted on."""
        shuffled = [{"seq": s, "notes": []} for s in (9, 3, 7, 1, 5, 2, 8, 4, 6, 10)]
        early, late = split_records(shuffled, share=0.5)
        assert [r["seq"] for r in early] == [1, 2, 3, 4, 5]
        assert [r["seq"] for r in late] == [6, 7, 8, 9, 10]

    def test_every_early_decision_precedes_every_late_one(self) -> None:
        early, late = split_records(_records(40), share=0.5)
        assert max(r["seq"] for r in early) < min(r["seq"] for r in late)

    def test_the_two_halves_are_disjoint_and_complete(self) -> None:
        early, late = split_records(_records(37), share=0.5)
        assert not {r["seq"] for r in early} & {r["seq"] for r in late}
        assert len(early) + len(late) == 37


class TestItRefusesADegenerateSplit:
    def test_an_empty_record_set_raises(self) -> None:
        """An empty held-out set proves nothing, so it must not report a pass."""
        with pytest.raises(ReviewOosError, match="no decisions"):
            split_records([])

    @pytest.mark.parametrize("share", [0.0, 1.0])
    def test_a_split_that_empties_one_side_raises(self, share: float) -> None:
        with pytest.raises(ReviewOosError, match="leaves one side empty"):
            split_records(_records(10), share=share)


class TestNotGradeableIsNotAFailure:
    """A rule that fires too rarely to judge has not failed — it has not been tested.

    Merging the two would make a smaller held-out window look like worse generalisation, which is
    an artefact of the split size rather than a fact about the rule.
    """

    def test_a_proposed_verdict_is_not_gradeable(self) -> None:
        split = RuleSplit("r", "misleading", "proposed", 0, 0, 0, 0)
        assert not split.gradeable_out
        assert not split.survived
        assert not split.flipped

    def test_a_flip_requires_both_halves_to_be_gradeable(self) -> None:
        split = RuleSplit("r", "proposed", "misleading", 0, 1, 0, 1)
        assert split.gradeable_out
        assert split.flipped is True
        assert not split.survived

    def test_the_same_verdict_on_both_halves_survives(self) -> None:
        split = RuleSplit("r", "misleading", "misleading", 3, 3, 0, 0)
        assert split.survived
        assert not split.flipped


class TestTheLiveRun:
    @pytest.fixture(scope="class")
    def report(self) -> dict[str, Any]:
        return run()

    def test_both_halves_carry_decisions(self, report: dict[str, Any]) -> None:
        assert report["in_sample_decisions"] > 0
        assert report["out_of_sample_decisions"] > 0

    def test_defects_are_counted_within_each_window(self, report: dict[str, Any]) -> None:
        """Grading the held-out half against the in-sample half's defects is the leak running
        backwards, and it would show up as the two counts being identical to the whole record."""
        assert report["in_sample_defects"] > 0
        assert report["out_of_sample_defects"] > 0

    def test_gradeable_and_not_gradeable_account_for_every_rule(
        self, report: dict[str, Any]
    ) -> None:
        assert (
            report["gradeable_out_of_sample"] + report["not_gradeable_out_of_sample"]
            == report["rules"]
        )

    def test_survived_and_flipped_account_for_every_gradeable_rule(
        self, report: dict[str, Any]
    ) -> None:
        assert report["survived"] + report["flipped"] == report["gradeable_out_of_sample"]

    def test_the_scope_statement_says_what_this_is_not(
        self, report: dict[str, Any]
    ) -> None:
        """It tests whether a rule's grade generalises, not whether the rule would have been
        discovered again — STANDING_RULES are hand-written, not generated in-sample."""
        scope = report["scope_statement"]
        assert "NOT CLAIMED" in scope
        assert "generated" in scope

    def test_the_rate_is_none_rather_than_zero_when_nothing_is_gradeable(self) -> None:
        """A survival rate of 0% and 'nothing could be graded' are different facts."""
        report = run(notes=_records(4), rules=())
        assert report["survival_rate"] is None
