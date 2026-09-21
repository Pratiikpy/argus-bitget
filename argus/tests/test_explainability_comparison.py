"""Tests for the ARGUS-vs-TradingAgents decision-explainability comparison.

Live-network tests: real Bitget ticker data is the whole point of the comparison (real current
price, real 24h high/low as the "facts" both real systems are checked against). Every expensive
function is wrapped in a module-scoped fixture and called once per test session. See
`eval/explainability_comparison.py`'s module docstring for the findings these tests pin.
"""

from __future__ import annotations

import pytest

from argus.eval.explainability_comparison import (
    HALLUCINATION_FACTORS,
    REAL_SYMBOLS,
    SCOPE_STATEMENT,
    measure_costs,
    render,
    run_ablation,
    run_baseline_reproduced,
    run_failure_cases,
    run_positive_control,
    run_reproducibility_check,
    run_same_input_comparison,
)


@pytest.fixture(scope="module")
def baseline() -> dict:
    return run_baseline_reproduced()


@pytest.fixture(scope="module")
def same_input() -> dict:
    return run_same_input_comparison()


@pytest.fixture(scope="module")
def control() -> dict:
    return run_positive_control()


@pytest.fixture(scope="module")
def ablation() -> dict:
    return run_ablation()


@pytest.fixture(scope="module")
def failures() -> dict:
    return run_failure_cases()


@pytest.fixture(scope="module")
def costs() -> dict:
    return measure_costs(repeats=5)


@pytest.fixture(scope="module")
def reproducibility() -> dict:
    return run_reproducibility_check()


class TestBaselineReproduced:
    def test_the_real_validator_accepts_a_wildly_fabricated_price(self, baseline: dict) -> None:
        assert baseline["real_entry_price_validated"] == 999_999.0
        assert not baseline["real_validator_raised"]

    def test_the_real_rendered_output_carries_the_fabricated_price(self, baseline: dict) -> None:
        assert "999999.0" in baseline["real_rendered"]


class TestSameInputComparison:
    def test_every_configured_symbol_and_factor_was_checked(self, same_input: dict) -> None:
        assert same_input["n_cases"] == len(REAL_SYMBOLS) * len(HALLUCINATION_FACTORS)

    def test_tradingagents_real_validator_never_catches_a_fabricated_price(
        self, same_input: dict
    ) -> None:
        assert same_input["n_tradingagents_catches"] == 0

    def test_argus_grounding_catches_every_fabricated_price(self, same_input: dict) -> None:
        assert same_input["n_argus_catches"] == same_input["n_cases"]

    def test_each_case_names_the_real_symbol_and_the_real_current_price(
        self, same_input: dict
    ) -> None:
        for case in same_input["cases"]:
            assert case["symbol"] in REAL_SYMBOLS
            assert case["real_current_price"] > 0


class TestPositiveControl:
    def test_the_real_current_price_resolves_for_every_symbol(self, control: dict) -> None:
        assert control["all_real_figures_resolve"]
        assert len(control["results"]) == len(REAL_SYMBOLS)


class TestAblation:
    def test_tolerance_is_the_load_bearing_boundary(self, ablation: dict) -> None:
        assert ablation["just_inside_resolves"]
        assert not ablation["just_outside_resolves"]
        assert ablation["tolerance_is_the_load_bearing_boundary"]


class TestFailureCases:
    def test_a_negative_price_is_accepted_without_error(self, failures: dict) -> None:
        case = failures["negative_price"]
        assert case["real_value_accepted"] == -50.0
        assert not case["real_validator_raised"]

    def test_a_percent_string_is_dropped_to_none_not_converted(self, failures: dict) -> None:
        case = failures["percent_string_dropped_to_none"]
        assert case["real_dropped_to_none_not_converted"]

    def test_a_placeholder_string_is_dropped_to_none(self, failures: dict) -> None:
        case = failures["placeholder_string_dropped_to_none"]
        assert case["real_result"] is None


class TestCosts:
    def test_both_real_checks_are_measured_and_positive(self, costs: dict) -> None:
        assert costs["tradingagents_validator_seconds_per_call"] > 0
        assert costs["argus_grounding_seconds_per_call"] > 0


class TestReproducibility:
    def test_repeated_checks_agree(self, reproducibility: dict) -> None:
        assert reproducibility["identical"]


class TestMain:
    def test_render_produces_readable_text(
        self, same_input: dict, control: dict, costs: dict, reproducibility: dict,
    ) -> None:
        report = {
            "same_input_comparison": same_input,
            "positive_control": control,
            "costs": costs,
            "reproducibility": reproducibility,
        }
        text = render(report)
        assert "DECISION EXPLAINABILITY" in text
        assert "TradingAgents catches" in text


def test_scope_statement_is_honest_about_what_is_not_claimed() -> None:
    assert "NOT claimed" in SCOPE_STATEMENT
    assert SCOPE_STATEMENT.count("NOT claimed") >= 2
