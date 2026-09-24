"""Tests for the ARGUS-vs-hftbacktest execution-assistance (decision-latency) comparison.

Live-network test: `run_same_input_comparison` fetches a real, live VIX reading. Reading
hftbacktest's real, locally-cloned source for the point-in-time grep is fast and offline. Every
expensive function is wrapped in a module-scoped fixture and called once per test session. See
`eval/execassist_comparison.py`'s module docstring for the findings these tests pin.
"""

from __future__ import annotations

import pytest

from argus.eval.execassist_comparison import (
    SCOPE_STATEMENT,
    THINKING_BUDGETS_MS,
    measure_costs,
    render,
    run_ablation,
    run_baseline_read,
    run_failure_cases,
    run_reproducibility_check,
    run_same_input_comparison,
)


@pytest.fixture(scope="module")
def baseline() -> dict:
    try:
        return run_baseline_read()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def same_input() -> dict:
    return run_same_input_comparison()


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


class TestBaselineRead:
    def test_every_configured_term_was_searched(self, baseline: dict) -> None:
        assert len(baseline["terms_searched"]) == 5

    def test_zero_decision_latency_representation_found(self, baseline: dict) -> None:
        assert baseline["total_hits"] == 0
        assert baseline["zero_decision_latency_representation"]
        assert baseline["hits"] == {}

    def test_the_real_trait_has_entry_and_response_only(self, baseline: dict) -> None:
        assert baseline["real_latency_model_trait_has_entry_and_response_only"]

    def test_the_real_constructor_has_exactly_two_params(self, baseline: dict) -> None:
        assert baseline["real_constant_latency_constructor_has_exactly_two_params"]


class TestSameInputComparison:
    def test_every_configured_budget_was_priced(self, same_input: dict) -> None:
        assert len(same_input["results"]) == len(THINKING_BUDGETS_MS)

    def test_costs_are_strictly_increasing_with_think_time(self, same_input: dict) -> None:
        assert same_input["costs_strictly_increasing_with_think_ms"]

    def test_a_real_live_vix_level_was_used(self, same_input: dict) -> None:
        assert same_input["vix_level_used"] > 0


class TestAblation:
    def test_cost_scales_with_sqrt_of_think_time(self, ablation: dict) -> None:
        assert ablation["matches_sqrt_scaling"]

    def test_the_quadrupled_case_costs_roughly_double(self, ablation: dict) -> None:
        assert ablation["measured_ratio"] is not None
        assert 1.9 < ablation["measured_ratio"] < 2.1


class TestFailureCases:
    def test_zero_deliberation_is_allowed_not_rejected(self, failures: dict) -> None:
        case = failures["zero_deliberation_is_allowed"]
        assert case["think_ms"] == 0
        assert not case["raised"]

    def test_negative_deliberation_raises(self, failures: dict) -> None:
        case = failures["negative_deliberation"]
        assert case["raised"]

    def test_negative_think_ms_in_cost_fn_raises(self, failures: dict) -> None:
        case = failures["negative_think_ms_in_cost_fn"]
        assert case["raised"]


class TestCosts:
    def test_the_real_cost_function_is_measured_and_fast(self, costs: dict) -> None:
        assert costs["argus_cost_fn_seconds_per_call"] > 0
        assert costs["argus_cost_fn_seconds_per_call"] < 0.01


class TestReproducibility:
    def test_repeated_calls_agree(self, reproducibility: dict) -> None:
        assert reproducibility["identical"]


class TestMain:
    def test_render_produces_readable_text(
        self, baseline: dict, same_input: dict, ablation: dict, reproducibility: dict,
    ) -> None:
        report = {
            "baseline_read": baseline,
            "same_input_comparison": same_input,
            "ablation": ablation,
            "reproducibility": reproducibility,
        }
        text = render(report)
        assert "EXECUTION ASSISTANCE" in text
        assert "hftbacktest" in text


def test_scope_statement_is_honest_about_what_is_not_claimed() -> None:
    assert "NOT claimed" in SCOPE_STATEMENT
    assert SCOPE_STATEMENT.count("NOT claimed") >= 2
