"""Tests for the ARGUS-vs-pytaa cross-asset rotation comparison.

Unlike most comparison test files in this directory, several cases here genuinely need live
network calls — real Bitget candle history for a real cross-asset universe (rTokens, crypto
majors, commodity tokens) is the whole point of the comparison. The expensive, network-touching
functions are wrapped in module-scoped fixtures so each real call happens once per test session,
not once per assertion. See `eval/rotation_comparison.py`'s module docstring for the findings
these tests pin.
"""

from __future__ import annotations

import pytest

from argus.eval.rotation_comparison import (
    DESIGN_SYMBOLS,
    SCOPE_STATEMENT,
    measure_costs,
    render,
    run_baseline_reproduced_cases,
    run_boundary_check,
    run_failure_cases,
    run_missing_weight_sweep,
    run_oos_check,
    run_reproducibility_check,
    run_silent_data_loss_cases,
)


@pytest.fixture(scope="module")
def baseline() -> dict:
    return run_baseline_reproduced_cases()


@pytest.fixture(scope="module")
def oos() -> dict:
    return run_oos_check()


@pytest.fixture(scope="module")
def failures() -> dict:
    return run_failure_cases()


@pytest.fixture(scope="module")
def boundary() -> dict:
    return run_boundary_check()


@pytest.fixture(scope="module")
def costs() -> dict:
    return measure_costs(repeats=20)


class TestBaselineReproduced:
    def test_every_design_symbol_with_history_agrees(self, baseline: dict) -> None:
        assert baseline["results"], "no symbols were checked — did the real fetch fail entirely?"
        assert baseline["all_agree"], baseline["results"]

    def test_defined_scores_match_to_floating_point(self, baseline: dict) -> None:
        for result in baseline["results"]:
            if not result["real_is_nan"]:
                assert result["argus_score"] is not None
                assert abs(result["argus_score"] - result["real_score"]) < 1e-9

    def test_gold_or_other_short_history_symbols_are_refused_not_nan(self, baseline: dict) -> None:
        """At least one symbol in the design sample is expected to have real NaN behaviour on the
        real side (XAUUSDT's real listing history is genuinely short) — if the fetch succeeded,
        confirm ARGUS refused rather than producing a number."""
        nan_results = [r for r in baseline["results"] if r["real_is_nan"]]
        for r in nan_results:
            assert r["argus_refused"]


class TestOOSWiderUniverse:
    def test_the_wider_real_universe_also_agrees(self, oos: dict) -> None:
        if oos["n_checked"] == 0:
            pytest.skip("no held-out symbols returned real history this run")
        assert oos["all_agree"], oos["results"]

    def test_the_held_out_set_excludes_the_design_symbols(self, oos: dict) -> None:
        assert not set(oos["held_out_symbols"]) & set(DESIGN_SYMBOLS)


class TestSilentDataLoss:
    def test_both_designed_cases_show_real_underallocation(self) -> None:
        report = run_silent_data_loss_cases()
        assert report["every_case_the_real_function_silently_underallocates"]
        assert report["every_case_argus_refuses"]

    def test_the_three_negative_case_loses_three_quarters(self) -> None:
        report = run_silent_data_loss_cases()
        case = {c["label"]: c for c in report["cases"]}["three_risk_assets_negative"]
        assert case["real_fraction_vanished"] == pytest.approx(0.75)
        assert case["is_neg"] == 3


class TestMissingWeightSweep:
    def test_the_closed_form_holds_at_every_swept_point(self) -> None:
        report = run_missing_weight_sweep()
        assert report["closed_form_confirmed_at_every_point"]
        assert len(report["points"]) == 5

    def test_full_flight_to_safety_vanishes_the_whole_book(self) -> None:
        report = run_missing_weight_sweep()
        last = report["points"][-1]
        assert last["allocated"] == pytest.approx(0.0)
        assert last["vanished"] == pytest.approx(1.0)


class TestFailureCases:
    def test_empty_safe_assets_silently_drops_weight_in_the_real_function(
        self, failures: dict
    ) -> None:
        case = failures["empty_safe_assets"]
        assert not case["real_raised"]
        assert case["real_silently_drops_a_quarter_with_zero_safe_assets_defined"]
        assert case["argus_refused"]

    def test_all_nan_scores_return_a_silent_zero_book_in_the_real_function(
        self, failures: dict
    ) -> None:
        case = failures["all_scores_nan"]
        assert not case["real_raised"]
        assert case["real_returns_a_silent_all_zero_book"]
        assert case["argus_refused"]

    def test_one_day_of_history_is_nan_in_the_real_function_not_an_error(
        self, failures: dict
    ) -> None:
        case = failures["one_day_of_real_history"]
        assert case["real_score_is_nan"]
        assert not case["real_raised"]
        assert case["argus_refused"]


class TestBoundaryCheck:
    def test_the_boundary_is_confirmed_exact_on_both_sides(self, boundary: dict) -> None:
        assert boundary["boundary_confirmed_exact"], boundary["results"]

    def test_twelve_points_is_nan_in_the_real_function(self, boundary: dict) -> None:
        below = boundary["results"]["one_below_minimum"]
        assert below["real_is_nan"]
        assert below["argus_refused"]

    def test_thirteen_points_agrees_exactly(self, boundary: dict) -> None:
        exact = boundary["results"]["exactly_minimum"]
        assert not exact["real_is_nan"]
        assert not exact["argus_refused"]
        assert exact["agree"]


class TestCosts:
    def test_costs_are_measured_on_both_real_sides(self, costs: dict) -> None:
        assert costs["real_pandas_signal_seconds_per_call"] > 0
        assert costs["argus_pure_python_seconds_per_call"] > 0
        assert costs["argus_faster_by_factor"] is not None
        assert costs["argus_faster_by_factor"] > 1


class TestReproducibility:
    def test_repeated_runs_are_identical(self) -> None:
        report = run_reproducibility_check()
        assert report["identical"]


class TestMain:
    def test_main_runs_end_to_end_and_render_produces_readable_text(
        self, baseline: dict, oos: dict, failures: dict, boundary: dict, costs: dict,
    ) -> None:
        report = {
            "baseline_reproduced": baseline,
            "silent_data_loss": run_silent_data_loss_cases(),
            "missing_weight_sweep": run_missing_weight_sweep(),
            "failure_cases": failures,
            "boundary_check": boundary,
            "costs": costs,
            "oos_wider_universe": oos,
            "reproducibility": run_reproducibility_check(),
            "scope_statement": SCOPE_STATEMENT,
        }
        text = render(report)
        assert "CROSS-ASSET ROTATION" in text
        assert "baseline reproduced" in text


def test_scope_statement_is_honest_about_what_oos_means_here() -> None:
    assert "backtest Sharpe" in SCOPE_STATEMENT
    assert "in-sample/out-of-sample" in SCOPE_STATEMENT
    assert "~13 months" in SCOPE_STATEMENT
