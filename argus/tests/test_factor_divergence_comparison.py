"""Tests for the ARGUS-vs-Alphalens rToken factor divergence comparison (Track 1's "rToken
Factor Strategies" sub-theme). Live-network tests: real Bitget candle history for the real
rToken universe is the whole point of the comparison. Expensive functions are wrapped in
module-scoped fixtures so each real fetch happens once per test session. See
`eval/factor_divergence_comparison.py`'s own module docstring for what these tests pin.
"""

from __future__ import annotations

import pytest

from argus.eval.factor_divergence_comparison import (
    SCOPE_STATEMENT,
    measure_costs,
    render,
    run_ablation,
    run_base_case,
    run_failure_cases,
    run_oos_check,
    run_reproducibility_check,
)


@pytest.fixture(scope="module")
def base_case() -> dict:
    return run_base_case()


@pytest.fixture(scope="module")
def oos() -> dict:
    return run_oos_check()


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


class TestBaseCase:
    def test_most_real_symbols_are_usable_on_both_real_sides(self, base_case: dict) -> None:
        assert base_case["n_symbols_market"] >= 8
        assert base_case["n_symbols_index"] >= 8

    def test_real_ic_readings_exist_on_both_real_sides(self, base_case: dict) -> None:
        assert base_case["n_market_ic_readings"] > 0
        assert base_case["n_index_ic_readings"] > 0

    def test_the_ic_difference_is_a_real_bootstrap_interval_not_a_bare_point_estimate(
        self, base_case: dict
    ) -> None:
        from argus.backtest.dependence import MIN_OBSERVATIONS

        diff = base_case["ic_difference"]
        assert diff["mean"] is not None
        if diff["n"] >= MIN_OBSERVATIONS:
            assert diff["ci_low"] is not None
            assert diff["ci_high"] is not None
            assert diff["ci_low"] <= diff["mean"] <= diff["ci_high"]

    def test_the_naive_significance_test_ran_on_the_same_real_data(
        self, base_case: dict
    ) -> None:
        naive = base_case["ic_difference_naive_significance"]
        if naive is not None:
            assert "p_value" in naive
            assert "naive_significant_at_5pct" in naive


class TestOosCheck:
    def test_both_real_windows_produced_a_reading(self, oos: dict) -> None:
        first_mean = oos["first_window"]["ic_difference"]["mean"]
        second_mean = oos["second_window"]["ic_difference"]["mean"]
        assert first_mean is not None
        assert second_mean is not None

    def test_sign_agreement_is_reported_not_assumed(self, oos: dict) -> None:
        assert isinstance(oos["sign_agrees"], bool)


class TestAblation:
    def test_the_gated_and_ungated_variants_both_ran_on_real_data(self, ablation: dict) -> None:
        assert ablation["gated_alpha23"]["ic_difference"]["mean"] is not None
        assert ablation["ungated_high_delta"]["ic_difference"]["mean"] is not None

    def test_whether_the_gate_changes_the_sign_is_reported(self, ablation: dict) -> None:
        assert isinstance(ablation["gate_changes_the_sign"], bool)


class TestFailureCases:
    def test_a_constant_high_series_produces_an_all_zero_factor(self, failures: dict) -> None:
        assert failures["constant_high_series_produces_all_zero_factor"]

    def test_a_single_symbol_panel_cannot_produce_a_real_cross_sectional_ic(
        self, failures: dict
    ) -> None:
        assert failures["single_symbol_panel_ic_is_all_nan"]

    def test_an_empty_panel_produces_an_empty_ic_not_a_crash(self, failures: dict) -> None:
        assert failures["empty_panel_produces_empty_ic"]


class TestCosts:
    def test_both_real_sides_measured_a_positive_cost(self, costs: dict) -> None:
        assert costs["argus_grammar_seconds_per_symbol"] > 0
        assert costs["alphalens_ic_seconds_per_call"] > 0


class TestReproducibility:
    def test_repeated_computation_on_the_same_fetched_data_agrees(
        self, reproducibility: dict
    ) -> None:
        assert reproducibility["identical"]


class TestMain:
    def test_render_produces_readable_text(
        self,
        base_case: dict,
        oos: dict,
        ablation: dict,
        failures: dict,
        costs: dict,
        reproducibility: dict,
    ) -> None:
        report = {
            "base_case": base_case,
            "oos_check": oos,
            "ablation": ablation,
            "failure_cases": failures,
            "costs": costs,
            "reproducibility": reproducibility,
            "scope_statement": SCOPE_STATEMENT,
        }
        text = render(report)
        assert "RTOKEN FACTOR DIVERGENCE" in text
        assert "IC difference" in text


def test_scope_statement_is_honest_about_what_is_not_claimed() -> None:
    assert "NOT claimed" in SCOPE_STATEMENT
    assert "Froot" in SCOPE_STATEMENT
    assert "arbitrage" in SCOPE_STATEMENT.lower()
