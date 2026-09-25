"""The overfitting gates against general-purpose rivals (`eval/general_overfitgates_comparison`).

Every rival here runs its real code in this environment — pydantic's ``validate_call`` around
vectorbt's vendored DSR, numpy/scipy floating-point trapping, scipy's and statsmodels' multiple-
testing corrections, and a derandomized Hypothesis search — so the module runs live, once, and the
assertions pin what it measured, including where ARGUS lost before adaptation and where the
general-purpose arms still beat the bare library. The prose in the module's docstring is checked
against the same run, so a later edit cannot soften a finding without failing here.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any

import pytest

from argus.backtest.metrics import MetricError
from argus.desk import review as desk_review
from argus.eval import artefact
from argus.eval import general_overfitgates_comparison as gog
from argus.eval.dsr_comparison import DsrCase


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return gog.main()


def _arm(report: dict[str, Any], name: str) -> dict[str, Any]:
    arm: dict[str, Any] = report["dsr_gate"]["arms"][name]
    return arm


class TestGrading:
    def test_a_nan_output_is_silent_wrong_whatever_was_expected(self) -> None:
        nan = gog.Outcome("silent_nonfinite", None, "nan")
        for expect in ("value", "refuse", "value_or_refuse"):
            assert gog.grade(expect, 0.4, nan) == "silent_wrong"

    def test_a_value_on_an_input_with_no_answer_is_silent_wrong(self) -> None:
        assert gog.grade("refuse", None, gog.Outcome("value", 0.5)) == "silent_wrong"

    def test_a_refusal_of_a_valid_input_is_over_refusal(self) -> None:
        refused = gog.Outcome("refused", None, "MetricError")
        assert gog.grade("value", 0.7, refused) == "over_refusal"
        assert gog.grade("value_or_refuse", 0.7, refused) == "correct_refusal"

    def test_an_exception_outside_the_arm_contract_is_untyped(self) -> None:
        assert gog.grade("refuse", None, gog.Outcome("raised_untyped")) == "untyped_refusal"

    def test_a_value_is_scored_against_the_reference(self) -> None:
        assert gog.grade("value", 0.7, gog.Outcome("value", 0.7 + 1e-12)) == "correct_value"
        assert gog.grade("value", 0.7, gog.Outcome("value", 0.71)) == "silent_wrong"

    def test_the_negated_gate_admits_nan_and_the_positive_gate_does_not(self) -> None:
        nan = gog.Outcome("silent_nonfinite", None, "nan")
        assert gog.admitted(nan, negated=True)
        assert not gog.admitted(nan, negated=False)
        refused = gog.Outcome("refused")
        assert not gog.admitted(refused, negated=True)
        assert not gog.admitted(refused, negated=False)


class TestReferences:
    def test_the_reference_is_undefined_on_every_non_finite_argument(self) -> None:
        base = DsrCase("base", observed=1.2, n=500, trials=10, variance_of_trials=0.3)
        assert gog.reference_dsr(base) is not None
        for poison in gog.POISONS:
            for case in (
                dataclasses.replace(base, observed=poison),
                dataclasses.replace(base, variance_of_trials=poison),
                dataclasses.replace(base, skew=poison),
                dataclasses.replace(base, kurtosis=poison),
            ):
                assert gog.reference_dsr(case) is None

    def test_one_trial_reduces_exactly_to_psr_against_zero(self) -> None:
        case = DsrCase("one", observed=0.05, n=500, trials=1, variance_of_trials=0.3)
        z = 0.05 * math.sqrt(499) / math.sqrt(1 + 2 / 4 * 0.05**2)
        reference = gog.reference_dsr(case)
        assert reference is not None
        assert math.isclose(reference, 0.5 * math.erfc(-z / math.sqrt(2)), rel_tol=1e-12)

    def test_bh_reference_matches_a_hand_worked_case(self) -> None:
        # m = 4, q = 0.05: thresholds 0.0125, 0.025, 0.0375, 0.05. p_(3) = 0.03 <= 0.0375, so the
        # three smallest are rejected although p_(2) = 0.028 > 0.025 — the step-up property.
        assert gog.bh_reference([0.03, 0.001, 0.2, 0.028], 0.05) == [True, True, False, True]
        assert gog.bonferroni_reference([0.012, 0.013], 0.025) == [True, False]


class TestDsrHeadToHead:
    def test_the_case_set_is_the_standing_set_plus_four_harder_tiers(
        self, report: dict[str, Any]
    ) -> None:
        dsr = report["dsr_gate"]
        assert dsr["tiers"] == {"standing_input": 1220, "poisoned": 72, "edge_valid": 10,
                                "overflow": 6, "real_producer": 24}
        assert dsr["cases"] == 1332

    def test_argus_now_makes_no_silent_wrong_untyped_or_over_refusal(
        self, report: dict[str, Any]
    ) -> None:
        argus = _arm(report, "argus_current")
        assert argus["silent_wrong"] == 0
        assert argus["grades"]["untyped_refusal"] == 0
        assert argus["grades"]["over_refusal"] == 0
        assert argus["false_admissions_negated_gate"] == 0
        assert argus["false_admissions_positive_gate"] == 0

    def test_argus_lost_to_the_general_purpose_contract_before_adaptation(
        self, report: dict[str, Any]
    ) -> None:
        verdict = report["verdict"]
        assert verdict["best_general_purpose_arm"] == "pydantic_contract_around_vectorbt"
        assert verdict["argus_lost_before_adaptation"] is True
        assert verdict["dsr_outcome_now"] == "argus_wins"
        assert verdict["silent_wrong_by_arm"] == {
            "argus_current": 0, "argus_pre_adaptation": 28, "vectorbt_bare": 95,
            "numpy_scipy_trap_around_vectorbt": 57, "pydantic_contract_around_vectorbt": 5,
        }

    def test_the_contract_passes_a_wrong_finite_value(self, report: dict[str, Any]) -> None:
        """Docstring finding 2: overflow gives 0.5, one trial gives 1.0, and the contract passes."""
        silent = _arm(report, "pydantic_contract_around_vectorbt")["silent_examples"]
        assert any(s.startswith("observed=1e200 -> 0.5") for s in silent)
        assert any(s.startswith("one_trial:observed=-2.0:variance=0.3 -> 1.0") for s in silent)
        assert _arm(report, "pydantic_contract_around_vectorbt")["by_tier"]["poisoned"] == {
            "correct_value": 0, "correct_refusal": 72, "over_refusal": 0, "untyped_refusal": 0,
            "silent_wrong": 0}

    def test_the_trap_around_the_gate_cannot_see_a_nan_argument(
        self, report: dict[str, Any]
    ) -> None:
        """Docstring finding 2: on real data the poison is a NaN and raises no IEEE flag."""
        trap = _arm(report, "numpy_scipy_trap_around_vectorbt")["by_tier"]["real_producer"]
        bare = _arm(report, "vectorbt_bare")["by_tier"]["real_producer"]
        assert trap["silent_wrong"] == bare["silent_wrong"] == 17
        assert trap["correct_refusal"] == 0

    def test_the_trap_around_the_whole_pipeline_ties_argus_on_real_data(
        self, report: dict[str, Any]
    ) -> None:
        tier = report["verdict"]["real_producer_tier"]
        assert tier["argus_current"] == tier["numpy_scipy_trap_whole_pipeline"]
        assert tier["argus_current"]["silent_wrong"] == 0

    def test_every_real_symbol_contributes_a_family_and_a_single_trial_case(
        self, report: dict[str, Any]
    ) -> None:
        cases = report["dsr_gate"]["real_producer_cases"]
        assert report["real_input"]["symbols"] == 12
        assert len(cases) == 24
        assert sum(":single_trial:" in c["case"] for c in cases) == 12

    def test_the_kurtosis_convention_gap_is_measured_not_assumed(
        self, report: dict[str, Any]
    ) -> None:
        gap = report["vectorbt_accessor_kurtosis_convention"]
        assert gap["family_cases_measured"] == 7
        assert 0.0 < gap["max_abs_dsr_gap"] < 1e-3


class TestMultipleTesting:
    def test_every_current_argus_arm_refuses_every_invalid_input(
        self, report: dict[str, Any]
    ) -> None:
        arms = report["multiple_testing"]["arms"]
        for name in ("argus_bh_current", "argus_desk_bh_current", "argus_bonferroni_current"):
            assert arms[name]["silent_on_invalid_input"] == 0
            assert arms[name]["wrong_decisions"] == 0
            assert arms[name]["over_refusals"] == 0
            assert arms[name]["untyped_refusals"] == 0

    def test_the_general_purpose_libraries_accept_invalid_input(
        self, report: dict[str, Any]
    ) -> None:
        arms = report["multiple_testing"]["arms"]
        assert arms["scipy_false_discovery_control"]["silent_on_invalid_input"] == 5
        assert arms["statsmodels_fdr_bh"]["silent_on_invalid_input"] == 15
        assert arms["statsmodels_bonferroni"]["permissive_on_invalid_input"] == 4

    def test_every_arm_agrees_with_the_exact_reference_on_valid_input(
        self, report: dict[str, Any]
    ) -> None:
        for arm in report["multiple_testing"]["arms"].values():
            assert arm["wrong_decisions"] == 0

    def test_the_desk_bh_refuses_a_nan_p_value(self) -> None:
        with pytest.raises(MetricError):
            desk_review.benjamini_hochberg([math.nan, 0.001])


class TestHypothesisSearch:
    def test_the_search_found_silent_failures_in_every_pre_adaptation_gate(
        self, report: dict[str, Any]
    ) -> None:
        pre = report["hypothesis_search"]["versions"]["pre_adaptation"]
        assert pre["functions_searched"] == 11
        assert len(pre["functions_with_failures"]) == 11
        for classes in pre["classes"].values():
            assert any(kind.startswith("silent") for kind in classes)

    def test_the_same_search_finds_nothing_in_the_current_gates(
        self, report: dict[str, Any]
    ) -> None:
        current = report["hypothesis_search"]["versions"]["current"]
        assert current["functions_searched"] == 11
        assert current["failure_classes"] == 0

    def test_the_search_is_reproducible(self) -> None:
        first = gog.hypothesis_search(max_examples=40, only=["deflated_sharpe", "sharpe"])
        second = gog.hypothesis_search(max_examples=40, only=["deflated_sharpe", "sharpe"])
        assert first["versions"] == second["versions"]
        assert first["versions"]["pre_adaptation"]["failure_classes"] > 0


class TestAblation:
    def test_every_adapted_guard_is_load_bearing(self, report: dict[str, Any]) -> None:
        rows = report["ablation"]
        guards = [r for r in rows if r["pre_adaptation"] != "absent"]
        assert len(guards) == 14
        assert all(r["load_bearing"] for r in guards)
        assert all(r["current"] == "refused" for r in guards)

    def test_the_pre_gate_returned_a_finite_half_for_infinite_kurtosis(
        self, report: dict[str, Any]
    ) -> None:
        by_case = {r["case"]: r for r in report["ablation"]}
        assert by_case["deflated_sharpe(kurtosis=+inf) — pre returned a finite 0.5"][
            "pre_adaptation"] == "returned:0.5"
        assert by_case["deflated_sharpe(skew=NaN)"]["pre_adaptation"] == "silent_nonfinite"
        assert by_case["probabilistic_sharpe(observed=+inf)"]["pre_adaptation"] == (
            "silent_nonfinite")

    def test_the_return_post_condition_is_reported_as_not_load_bearing(
        self, report: dict[str, Any]
    ) -> None:
        post = [r for r in report["ablation"] if r["pre_adaptation"] == "absent"]
        assert len(post) == 1 and post[0]["load_bearing"] is False


def test_scope_statement_names_the_loss_before_adaptation(report: dict[str, Any]) -> None:
    text = report["scope_statement"]
    assert "NOT claimed: that ARGUS was ahead before this comparison" in text
    assert f"{report['dsr_gate']['cases']:,} identical inputs" in text


def test_render(report: dict[str, Any]) -> None:
    text = gog.render(report)
    assert "DSR gate" in text and "hypothesis pre_adaptation" in text


class TestCommittedArtefact:
    def test_it_is_strict_json(self) -> None:
        assert artefact.is_strict(gog.ARTEFACT)

    def test_it_matches_this_run(self, report: dict[str, Any]) -> None:
        """Nothing in the report is timed or seeded, so the committed file must equal a rerun —
        except the first input Hypothesis witnessed for each failure class, which depends on the
        project modules already imported (see ``hypothesis_search``). The classes themselves are
        compared exactly; only the witness strings are set aside."""
        committed = json.loads(gog.ARTEFACT.read_text(encoding="utf-8"))
        assert _without_witnesses(committed) == _without_witnesses(
            json.loads(artefact.dumps(report)))

    def test_the_committed_file_carries_no_local_path(self) -> None:
        text = gog.ARTEFACT.read_text(encoding="utf-8").lower()
        for marker in ("c:\\\\", "c:/", "/users/", "appdata", artefact.SCRATCH_DIR,
                       "site-packages"):
            assert marker not in text


def _without_witnesses(report: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(json.dumps(report))
    for version in out["hypothesis_search"]["versions"].values():
        version["classes"] = {fn: sorted(kinds) for fn, kinds in version["classes"].items()}
    return out
