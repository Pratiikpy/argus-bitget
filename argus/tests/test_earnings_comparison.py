"""Tests for the ARGUS-vs-QuantConnect SUE (earnings surprise) comparison.

Live-network tests: real SEC EDGAR quarterly EPS is the whole point of the comparison. Expensive
functions are wrapped in module-scoped fixtures so each real fetch happens once per test session.
See `eval/earnings_comparison.py`'s module docstring for the findings these tests pin.
"""

from __future__ import annotations

import pytest

from argus.eval.earnings_comparison import (
    ANCHORS,
    SCOPE_STATEMENT,
    measure_costs,
    render,
    run_baseline_reproduced_cases,
    run_failure_cases,
    run_ranking_case,
    run_reproducibility_check,
    run_silent_failure_cases,
)


@pytest.fixture(scope="module")
def baseline() -> dict:
    return run_baseline_reproduced_cases()


@pytest.fixture(scope="module")
def silent_failures() -> dict:
    return run_silent_failure_cases()


@pytest.fixture(scope="module")
def ranking() -> dict:
    return run_ranking_case()


@pytest.fixture(scope="module")
def failures() -> dict:
    return run_failure_cases()


@pytest.fixture(scope="module")
def costs() -> dict:
    return measure_costs(repeats=20)


@pytest.fixture(scope="module")
def reproducibility() -> dict:
    return run_reproducibility_check()


class TestBaselineReproduced:
    def test_most_or_all_real_anchors_have_enough_history(self, baseline: dict) -> None:
        assert len(baseline["symbols_checked"]) >= len(ANCHORS) - 1

    def test_every_checked_anchor_agrees_exactly(self, baseline: dict) -> None:
        assert baseline["all_agree"], baseline["results"]
        for result in baseline["results"]:
            if "skipped" not in result:
                assert result["agree"], result


class TestSilentFailureCases:
    def test_linear_growth_produces_a_real_infinite_sue(self, silent_failures: dict) -> None:
        case = silent_failures["linear_growth_zero_variance"]
        assert case["real_is_infinite"]
        assert not case["real_raised"]
        assert case["real_warnings"]

    def test_argus_refuses_the_linear_growth_case(self, silent_failures: dict) -> None:
        assert silent_failures["linear_growth_zero_variance"]["argus_refused"]

    def test_all_flat_eps_produces_a_real_nan_sue(self, silent_failures: dict) -> None:
        case = silent_failures["all_flat_eps"]
        assert case["real_is_nan"]
        assert not case["real_raised"]

    def test_argus_refuses_the_all_flat_case(self, silent_failures: dict) -> None:
        assert silent_failures["all_flat_eps"]["argus_refused"]


class TestRankingCase:
    def test_the_real_ranking_puts_the_constructed_artifact_first(self, ranking: dict) -> None:
        assert ranking["real_top_is_the_constructed_artifact"]

    def test_argus_excludes_the_constructed_artifact_from_its_ranking(
        self, ranking: dict
    ) -> None:
        assert ranking["argus_excludes_the_constructed_artifact"]
        assert "CONSTRUCTED_LINEAR" in ranking["argus_skipped"]

    def test_argus_still_ranks_the_real_anchors(self, ranking: dict) -> None:
        assert len(ranking["argus_ranked"]) >= len(ANCHORS) - 1


class TestFailureCases:
    def test_the_real_reference_raises_a_bare_indexerror_on_insufficient_history(
        self, failures: dict
    ) -> None:
        case = failures["insufficient_history"]
        assert case["real_raised_indexerror"]

    def test_argus_refuses_with_a_typed_sueerror_on_the_same_input(
        self, failures: dict
    ) -> None:
        assert failures["insufficient_history"]["argus_refused_with_sueerror"]


class TestCosts:
    def test_costs_are_measured_on_both_real_sides(self, costs: dict) -> None:
        assert costs["real_vendored_seconds_per_call"] > 0
        assert costs["argus_pure_python_seconds_per_call"] > 0
        assert costs["argus_faster_by_factor"] is not None


class TestReproducibility:
    def test_repeated_real_fetches_agree(self, reproducibility: dict) -> None:
        assert reproducibility["identical"]


class TestMain:
    def test_render_produces_readable_text(
        self, baseline: dict, silent_failures: dict, ranking: dict, failures: dict,
        costs: dict, reproducibility: dict,
    ) -> None:
        report = {
            "baseline_reproduced": baseline,
            "silent_failure_cases": silent_failures,
            "ranking_case": ranking,
            "failure_cases": failures,
            "costs": costs,
            "reproducibility": reproducibility,
            "scope_statement": SCOPE_STATEMENT,
        }
        text = render(report)
        assert "EARNINGS SUE" in text
        assert "baseline reproduced" in text


def test_scope_statement_is_honest_about_what_is_not_claimed() -> None:
    assert "NOT claimed" in SCOPE_STATEMENT
    assert "forecasting value" in SCOPE_STATEMENT
