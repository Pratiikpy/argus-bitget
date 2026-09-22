"""Tests for the ARGUS-vs-FinanceBench information-extraction comparison.

Live-network tests: real SEC XBRL data is the whole point of the comparison (real filed values
for real companies). Every expensive function is wrapped in a module-scoped fixture and called
once per test session. See `eval/infoextract_comparison.py`'s module docstring for the findings
these tests pin.
"""

from __future__ import annotations

import pytest

from argus.eval.infoextract_comparison import (
    DESIGNED_CASES,
    FINANCEBENCH_METRICS_GENERATED,
    SCOPE_STATEMENT,
    measure_costs,
    render,
    run_ablation,
    run_baseline_reproduced,
    run_designed_cases,
    run_failure_cases,
    run_reproducibility_check,
)


@pytest.fixture(scope="module")
def baseline() -> dict:
    return run_baseline_reproduced()


@pytest.fixture(scope="module")
def cases() -> dict:
    return run_designed_cases()


@pytest.fixture(scope="module")
def ablation() -> dict:
    return run_ablation()


@pytest.fixture(scope="module")
def failures() -> dict:
    return run_failure_cases()


@pytest.fixture(scope="module")
def costs() -> dict:
    return measure_costs(repeats=2)


@pytest.fixture(scope="module")
def reproducibility() -> dict:
    return run_reproducibility_check()


class TestBaselineReproduced:
    def test_every_mode_from_the_real_repo_is_present(self, baseline: dict) -> None:
        assert set(baseline) == set(FINANCEBENCH_METRICS_GENERATED)

    def test_oracle_is_the_best_realistic_ceiling_and_still_not_perfect(
        self, baseline: dict
    ) -> None:
        assert 0.8 < baseline["oracle"]["accuracy"] < 1.0

    def test_every_non_oracle_condition_scores_below_oracle(self, baseline: dict) -> None:
        for mode in FINANCEBENCH_METRICS_GENERATED:
            if mode == "oracle":
                continue
            assert baseline[mode]["accuracy"] < baseline["oracle"]["accuracy"]

    def test_incontext_has_a_material_confidently_wrong_rate(self, baseline: dict) -> None:
        assert baseline["inContext"]["confidently_wrong_rate"] >= 0.3


class TestDesignedCases:
    def test_every_configured_case_was_attempted(self, cases: dict) -> None:
        assert cases["n_cases"] == len(DESIGNED_CASES)

    def test_most_or_all_cases_resolve(self, cases: dict) -> None:
        assert cases["n_resolved"] >= cases["n_cases"] - 1

    def test_a_case_that_does_not_resolve_names_why(self, cases: dict) -> None:
        for case in cases["cases"]:
            if not case["resolved"]:
                assert case["status"]

    def test_a_resolved_case_carries_a_real_filed_date_and_form(self, cases: dict) -> None:
        for case in cases["cases"]:
            if case["resolved"]:
                assert case["filed"] is not None
                assert case["form"]


class TestAblation:
    def test_the_naive_fetch_is_genuinely_ambiguous(self, ablation: dict) -> None:
        assert ablation["naive_no_filter_n_matches"] == 2
        assert ablation["naive_is_ambiguous"]
        assert ablation["naive_values_diverge"]

    def test_argus_resolves_to_exactly_one_quarterly_value(self, ablation: dict) -> None:
        assert ablation["argus_resolves_to_exactly_one"]
        assert ablation["argus_quarterly_only_is_quarterly"]

    def test_argus_picks_the_true_quarterly_value_not_the_cumulative_one(
        self, ablation: dict
    ) -> None:
        assert ablation["argus_quarterly_only_value"] in ablation["naive_no_filter_values"]
        assert ablation["argus_quarterly_only_value"] == -120929000.0


class TestFailureCases:
    def test_an_unknown_ticker_reports_why_rather_than_raising(self, failures: dict) -> None:
        case = failures["unknown_ticker"]
        assert not case["raised"]
        assert case["reports_why_rather_than_guessing"]
        assert case["n_facts"] == 0

    def test_point_in_time_withholds_facts_filed_after_the_cutoff(self, failures: dict) -> None:
        case = failures["point_in_time_withholds_future_filings"]
        assert case["all_filed_before_cutoff"]
        assert case["n_facts_visible_at_2020"] > 0


class TestCosts:
    def test_the_real_fetch_is_measured_and_positive(self, costs: dict) -> None:
        assert costs["argus_live_xbrl_fetch_seconds_per_call"] > 0


class TestReproducibility:
    def test_repeated_fetches_agree(self, reproducibility: dict) -> None:
        assert reproducibility["identical"]


class TestMain:
    def test_render_produces_readable_text(
        self, baseline: dict, cases: dict, ablation: dict, reproducibility: dict,
    ) -> None:
        report = {
            "baseline_reproduced": baseline,
            "designed_cases": cases,
            "ablation": ablation,
            "reproducibility": reproducibility,
        }
        text = render(report)
        assert "INFORMATION EXTRACTION" in text
        assert "FinanceBench" in text


def test_scope_statement_is_honest_about_what_is_not_claimed() -> None:
    assert "NOT claimed" in SCOPE_STATEMENT
    assert SCOPE_STATEMENT.count("NOT claimed") >= 2
