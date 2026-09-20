"""Tests for the ARGUS-vs-QuantConnect After-Hours / Pre-Holiday comparison.

Live-network tests: real Bitget candle history is the whole point of the comparison. Expensive
functions are wrapped in module-scoped fixtures so each real fetch happens once per test session.
See `eval/afterhours_comparison.py`'s module docstring for the findings these tests pin.
"""

from __future__ import annotations

import pytest

from argus.eval.afterhours_comparison import (
    SCOPE_STATEMENT,
    measure_costs,
    render,
    run_base_case,
    run_failure_cases,
    run_reproducibility_check,
)


@pytest.fixture(scope="module")
def base_case() -> dict:
    return run_base_case()


@pytest.fixture(scope="module")
def failures() -> dict:
    return run_failure_cases()


@pytest.fixture(scope="module")
def costs() -> dict:
    return measure_costs(repeats=10)


@pytest.fixture(scope="module")
def reproducibility() -> dict:
    return run_reproducibility_check()


class TestBaseCase:
    def test_the_holiday_phase_was_dead_before_the_real_calendar_was_wired_in(
        self, base_case: dict
    ) -> None:
        assert not base_case["without_holiday_calendar_has_holiday_phase"]

    def test_the_real_calendar_unlocks_real_holiday_sessions(self, base_case: dict) -> None:
        if base_case["real_holiday_sessions_found"] == 0:
            pytest.skip("no real holiday fell inside the live trailing window this run")
        assert base_case["with_holiday_calendar_has_holiday_phase"]
        assert base_case["with_holiday_calendar_by_phase_holiday"] is not None

    def test_the_blind_long_backtest_ran_on_every_real_holiday_session(
        self, base_case: dict
    ) -> None:
        backtest = base_case["blind_long_backtest"]
        assert backtest["n_sessions"] == base_case["real_holiday_sessions_found"]
        if backtest["n_sessions"] == 0:
            pytest.skip("no real holiday session to backtest this run")
        for session in backtest["sessions"]:
            assert session["real_decision_went_long"]

    def test_the_real_win_rate_is_reported_not_assumed(self, base_case: dict) -> None:
        backtest = base_case["blind_long_backtest"]
        if backtest["n_sessions"] == 0:
            pytest.skip("no real holiday session to backtest this run")
        assert 0.0 <= backtest["win_rate_gross_pct"] <= 100.0
        assert 0.0 <= backtest["win_rate_net_of_fee_pct"] <= 100.0


class TestFailureCases:
    def test_liquidates_when_already_long_and_the_holiday_has_passed(
        self, failures: dict
    ) -> None:
        assert failures["liquidates_when_already_long_and_no_holiday"]

    def test_holds_when_already_long_and_a_holiday_is_still_near(self, failures: dict) -> None:
        assert failures["holds_when_already_long_and_holiday_still_near"]

    def test_stays_flat_when_no_holiday_and_not_invested(self, failures: dict) -> None:
        assert failures["stays_flat_when_no_holiday_and_not_invested"]


class TestCosts:
    def test_costs_are_measured_on_both_real_sides(self, costs: dict) -> None:
        assert costs["quantconnect_decision_seconds_per_call"] > 0
        assert costs["argus_property_seconds_per_call"] > 0


class TestReproducibility:
    def test_repeated_computation_on_the_same_fetched_data_agrees(
        self, reproducibility: dict
    ) -> None:
        assert reproducibility["identical"]


class TestMain:
    def test_render_produces_readable_text(
        self, base_case: dict, failures: dict, costs: dict, reproducibility: dict,
    ) -> None:
        report = {
            "base_case": base_case,
            "failure_cases": failures,
            "costs": costs,
            "reproducibility": reproducibility,
            "scope_statement": SCOPE_STATEMENT,
        }
        text = render(report)
        assert "AFTER-HOURS INFORMATION PRICING" in text
        assert "blind-long win rate" in text


def test_scope_statement_is_honest_about_what_is_not_claimed() -> None:
    assert "NOT claimed" in SCOPE_STATEMENT
    assert "full Lean engine" in SCOPE_STATEMENT
