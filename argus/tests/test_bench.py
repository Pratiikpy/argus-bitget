"""ARGUS-BENCH tests — a benchmark that cannot fail its author is not a benchmark.

Every criterion is tested twice: once on a payload where it should pass, once on a payload where it
should fail or be undefined. The specific thing guarded is the conflation of UNDEFINED with a score
of zero, because that is the failure that would let this scorecard flatter us — a desk with no
trades would read as a desk with a perfect takeover rate, and an artefact that is simply missing
would read as a criterion met.
"""

from __future__ import annotations

from argus.eval.bench import (
    BenchReport,
    Criterion,
    Grade,
    decision_consistency,
    human_takeover_rate,
    incremental_value,
    risk_violation_rate,
    stress_behaviour,
)


class TestDecisionConsistency:
    def test_unanimous_verdicts_over_few_replays_are_weak_not_passing(self) -> None:
        """Three replays agreeing is weak evidence of determinism. Grading it PASS would let a
        cheap experiment stand in for an expensive one."""
        payload = {"models": [{"decisions": 3, "verdict_consistency": 1.0}]}
        got = decision_consistency(payload)
        assert got.grade is Grade.WEAK
        assert got.value == 1.0

    def test_unanimity_over_many_replays_passes(self) -> None:
        payload = {"models": [{"decisions": 5, "verdict_consistency": 1.0}]}
        assert decision_consistency(payload).grade is Grade.PASS

    def test_a_model_that_flips_fails(self) -> None:
        payload = {"models": [
            {"decisions": 5, "verdict_consistency": 1.0},
            {"decisions": 5, "verdict_consistency": 0.6},
        ]}
        got = decision_consistency(payload)
        assert got.grade is Grade.FAIL
        assert got.value == 0.5

    def test_a_model_run_once_cannot_disagree_with_itself(self) -> None:
        payload = {"models": [{"decisions": 1, "verdict_consistency": 1.0}]}
        got = decision_consistency(payload)
        assert got.grade is Grade.UNDEFINED
        assert "nothing could disagree with itself" in got.detail

    def test_a_missing_artefact_is_undefined_and_says_how_to_produce_it(self) -> None:
        got = decision_consistency({})
        assert got.grade is Grade.UNDEFINED
        assert "argus.eval.bakeoff" in got.detail


class TestRiskViolationRate:
    def test_a_sound_exhaustive_sweep_with_no_violations_passes(self) -> None:
        payload = {"swept": 6720, "sound": True, "reduced": 2928, "violations": []}
        got = risk_violation_rate(payload)
        assert got.grade is Grade.PASS
        assert got.value == 0.0

    def test_any_violation_fails(self) -> None:
        payload = {"swept": 6720, "sound": False, "reduced": 10, "violations": [{"state": 1}]}
        assert risk_violation_rate(payload).grade is Grade.FAIL

    def test_an_unsound_sweep_fails_even_with_no_violations_listed(self) -> None:
        """Soundness is the prover's own verdict. Zero listed violations from an unsound sweep is
        an absence of evidence, not evidence of absence."""
        payload = {"swept": 6720, "sound": False, "reduced": 10, "violations": []}
        assert risk_violation_rate(payload).grade is Grade.FAIL

    def test_the_denominator_is_the_swept_state_space(self) -> None:
        payload = {"swept": 100, "sound": False, "reduced": 0, "violations": [1, 2, 3]}
        assert risk_violation_rate(payload).value == 0.03

    def test_a_missing_artefact_is_undefined(self) -> None:
        assert risk_violation_rate({}).grade is Grade.UNDEFINED


class TestStressBehaviour:
    def test_surviving_every_scenario_passes(self) -> None:
        payload = {"survives_all": True, "results": [{"survives": True}] * 9}
        got = stress_behaviour(payload)
        assert got.grade is Grade.PASS
        assert got.value == 1.0

    def test_a_failed_scenario_is_weak_not_passing(self) -> None:
        payload = {"survives_all": False, "results": [{"survives": True}, {"survives": False}]}
        got = stress_behaviour(payload)
        assert got.grade is Grade.WEAK
        assert got.value == 0.5

    def test_it_states_that_it_does_not_test_the_desk_deciding_under_stress(self) -> None:
        """The caveat must travel with the number. This tests a held position, not a decision."""
        payload = {"survives_all": True, "results": [{"survives": True}]}
        assert "does not test the desk deciding" in stress_behaviour(payload).detail

    def test_a_missing_artefact_is_undefined(self) -> None:
        assert stress_behaviour({}).grade is Grade.UNDEFINED


class TestHumanTakeoverRate:
    def test_no_actionable_decision_is_undefined_not_a_perfect_score(self) -> None:
        """The failure this whole grade exists to prevent. A desk that never traded has not earned
        a takeover rate of zero; it has no takeover rate at all."""
        payload = {"rate": None, "verdict": "UNDEFINED over 42 decisions"}
        got = human_takeover_rate(payload)
        assert got.grade is Grade.UNDEFINED
        assert got.value is None

    def test_an_undefined_rate_still_reports_whether_the_policy_is_reachable(self) -> None:
        """The rate alone cannot tell a working policy from dead code, and the scorecard says so.
        Reachability is the measurement that can: a condition the live record has produced is one
        that would fire in production, even with no decision yet reaching the venue."""
        payload = {
            "rate": None,
            "verdict": "UNDEFINED over 57 decision(s)",
            "reachability": {
                "available": True, "cycles_examined": 76, "observed": 2, "total_conditions": 5,
                "conditions_observed": {"ungrounded_figure": 17, "unresolved_conflict": 9},
            },
        }
        got = human_takeover_rate(payload)
        assert got.grade is Grade.UNDEFINED
        assert "not dead code" in got.detail
        assert "ungrounded_figure (17)" in got.detail

    def test_no_observed_condition_is_reported_as_indistinguishable_from_dead_code(self) -> None:
        """The honest reading when nothing has fired. Silence is not evidence the policy works."""
        payload = {
            "rate": None, "verdict": "UNDEFINED",
            "reachability": {"available": True, "cycles_examined": 76, "observed": 0,
                             "total_conditions": 5, "conditions_observed": {}},
        }
        assert "not distinguishable from dead code" in human_takeover_rate(payload).detail

    def test_a_moderate_rate_passes(self) -> None:
        assert human_takeover_rate({"rate": 0.1, "verdict": ""}).grade is Grade.PASS

    def test_a_zero_rate_is_weak_because_dead_code_looks_identical(self) -> None:
        assert human_takeover_rate({"rate": 0.0, "verdict": ""}).grade is Grade.WEAK

    def test_escalating_more_than_half_fails(self) -> None:
        """Above half, the desk is not operating unattended in any meaningful sense."""
        assert human_takeover_rate({"rate": 0.8, "verdict": ""}).grade is Grade.FAIL

    def test_a_missing_artefact_is_undefined(self) -> None:
        assert human_takeover_rate(None).grade is Grade.UNDEFINED


class TestIncrementalValue:
    def _payload(self, comparisons: list[dict[str, float]], desk: float, best: float
                 ) -> dict[str, object]:
        return {
            "comparisons": comparisons,
            "desk": {"total_bps": desk},
            "baselines": [{"total_bps": best}],
            "verdict": "",
        }

    def test_being_beaten_by_a_baseline_fails(self) -> None:
        payload = self._payload([{"wins": 5, "losses": 30, "p_value": 0.001}], 0.0, 500.0)
        assert incremental_value(payload).grade is Grade.FAIL

    def test_beating_a_baseline_passes(self) -> None:
        payload = self._payload([{"wins": 30, "losses": 5, "p_value": 0.001}], 500.0, 0.0)
        assert incremental_value(payload).grade is Grade.PASS

    def test_nothing_separating_is_weak_not_a_pass(self) -> None:
        """An inconclusive comparison is a sample too small to settle the question, which is not
        the same as evidence that the desk is as good as the rule."""
        payload = self._payload([{"wins": 20, "losses": 18, "p_value": 0.7}], 0.0, 200.0)
        assert incremental_value(payload).grade is Grade.WEAK

    def test_a_large_win_margin_with_a_weak_p_value_still_does_not_pass(self) -> None:
        payload = self._payload([{"wins": 22, "losses": 18, "p_value": 0.4}], 900.0, 10.0)
        assert incremental_value(payload).grade is Grade.WEAK

    def test_the_value_is_the_edge_over_the_best_baseline(self) -> None:
        payload = self._payload([{"wins": 20, "losses": 20, "p_value": 1.0}], 100.0, 300.0)
        assert incremental_value(payload).value == -200.0

    def test_a_missing_artefact_is_undefined(self) -> None:
        assert incremental_value({}).grade is Grade.UNDEFINED


class TestTheScorecard:
    def _criterion(self, name: str, grade: Grade) -> Criterion:
        return Criterion(name, grade, 1.0, "", "detail", "data/x.json")

    def test_undefined_criteria_are_counted_separately_from_failures(self) -> None:
        report = BenchReport(criteria=(
            self._criterion("a", Grade.PASS),
            self._criterion("b", Grade.UNDEFINED),
            self._criterion("c", Grade.FAIL),
        ))
        assert len(report.passing) == 1
        assert len(report.undefined) == 1
        assert len(report.graded) == 2

    def test_the_verdict_names_undefined_as_different_from_scoring_zero(self) -> None:
        report = BenchReport(criteria=(self._criterion("a", Grade.UNDEFINED),))
        assert "different statement from scoring zero" in report.verdict

    def test_the_verdict_names_what_failed(self) -> None:
        report = BenchReport(criteria=(self._criterion("incremental value", Grade.FAIL),))
        assert "Failing: incremental value" in report.verdict

    def test_the_verdict_names_what_is_weak(self) -> None:
        report = BenchReport(criteria=(self._criterion("consistency", Grade.WEAK),))
        assert "Weak: consistency" in report.verdict

    def test_it_says_it_was_run_against_its_own_author_first(self) -> None:
        """A benchmark its author passes is not a benchmark. That has to be in the output, not
        only in the docstring."""
        report = BenchReport(criteria=(self._criterion("a", Grade.PASS),))
        assert "run against its own author first" in report.verdict

    def test_every_criterion_names_the_artefact_it_was_read_from(self) -> None:
        """So a judge can check the number without reading the code."""
        report = BenchReport(criteria=(self._criterion("a", Grade.PASS),))
        for criterion in report.criteria:
            assert criterion.source.startswith("data/")

    def test_the_rendered_report_shows_grades_and_details(self) -> None:
        text = BenchReport(criteria=(self._criterion("a", Grade.PASS),)).render()
        assert "ARGUS-BENCH" in text
        assert "PASS" in text
        assert "detail" in text

    def test_the_dict_carries_the_counts(self) -> None:
        report = BenchReport(criteria=(
            self._criterion("a", Grade.PASS), self._criterion("b", Grade.UNDEFINED),
        ))
        got = report.as_dict()
        assert got["total"] == 2
        assert got["passing"] == 1
        assert got["undefined"] == 1
