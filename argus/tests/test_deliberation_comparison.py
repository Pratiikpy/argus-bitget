"""Tests for the ARGUS-vs-LatencySensitiveBench deliberation-cost comparison.

`linear_decay_price` is an independent, clean-room reimplementation (no permissive license to
vendor under — see its own module docstring), so its correctness is proven here by pinning
reference output vectors computed by running LatencySensitiveBench's OWN real, unmodified
`TradeMarket._apply_linear_decay` once locally (the exact command is in
`eval/baselines/latencybench_reimpl.py`'s docstring) — not by trusting the reimplementation alone.
ARGUS's own `latency_slippage_bps`/`deliberation_cost_bps` are called directly, never reimplemented.
"""

from __future__ import annotations

import pytest

from argus.agents.meta_pm import THINKING_MS
from argus.eval.baselines.latencybench_reimpl import LatencyBenchReimplError, linear_decay_price
from argus.eval.deliberation_comparison import (
    DECAY_WINDOW_SECONDS,
    SCOPE_STATEMENT,
    SWEPT_DELAYS_SECONDS,
    argus_cost_bps,
    argus_cost_is_unbounded,
    linear_decay_collapses_argus_real_tiers,
    linear_decay_cost_bps,
    linear_decay_is_bounded_past_its_cap,
    measure_costs,
    run_ablation,
    run_thinking_budget_cases,
    swept_comparison,
)


class TestLinearDecayMatchesTheRealReferenceVectors:
    """Six (input, output) pairs computed by running LatencySensitiveBench's own real,
    unmodified `_apply_linear_decay` method once, locally — see
    `eval/baselines/latencybench_reimpl.py`'s module docstring for the exact command run and the
    commit (`abceb6a374a94fe1b1ea41251ce64d21dbaf6647`) it was run against."""

    @pytest.mark.parametrize(
        ("high", "low", "delay", "expected_high", "expected_low"),
        [
            (110.0, 90.0, 0.0, 110.0, 90.0),
            (110.0, 90.0, 0.5, 106.66666666666667, 93.33333333333334),
            (110.0, 90.0, 0.75, 105.0, 95.0),
            (110.0, 90.0, 1.5, 100.0, 100.0),
            (110.0, 90.0, 2.0, 100.0, 100.0),
            (110.0, 90.0, -1.0, 110.0, 90.0),
        ],
    )
    def test_matches_the_real_reference_vector(
        self, high: float, low: float, delay: float, expected_high: float, expected_low: float,
    ) -> None:
        got_high, got_low = linear_decay_price(
            high, low, delay, average_price=100.0, decay_window=1.5,
        )
        assert got_high == pytest.approx(expected_high)
        assert got_low == pytest.approx(expected_low)

    def test_a_non_positive_decay_window_is_refused(self) -> None:
        with pytest.raises(LatencyBenchReimplError, match="must be positive"):
            linear_decay_price(110.0, 90.0, 1.0, average_price=100.0, decay_window=0.0)


class TestArgusCostIsCalledDirectlyNotReimplemented:
    def test_zero_delay_costs_nothing(self) -> None:
        assert argus_cost_bps(0.0) == 0.0

    def test_cost_grows_with_delay(self) -> None:
        assert argus_cost_bps(10.0) > argus_cost_bps(1.0) > argus_cost_bps(0.1)

    def test_cost_scales_with_volatility(self) -> None:
        from decimal import Decimal

        low_vol = argus_cost_bps(5.0, annualised_vol=Decimal("0.1"))
        high_vol = argus_cost_bps(5.0, annualised_vol=Decimal("1.0"))
        assert high_vol > low_vol


class TestLinearDecayCostBps:
    def test_zero_delay_is_zero_cost(self) -> None:
        assert linear_decay_cost_bps(0.0) == 0.0

    def test_at_the_cap_cost_is_close_to_half_the_spread(self) -> None:
        """Not exactly spread/2: the bps cost is normalised against `high` (the price actually
        quoted), and `high` itself is spread/2 above the average — a small, expected
        second-order gap (200bps spread -> ~99.01bps cost, not exactly 100bps)."""
        from argus.eval.deliberation_comparison import REFERENCE_SPREAD_BPS

        cost = linear_decay_cost_bps(DECAY_WINDOW_SECONDS)
        assert cost == pytest.approx(REFERENCE_SPREAD_BPS / 2, rel=0.02)

    def test_cost_is_identical_at_the_cap_and_far_past_it(self) -> None:
        at_cap = linear_decay_cost_bps(DECAY_WINDOW_SECONDS)
        far_past = linear_decay_cost_bps(1000.0)
        assert at_cap == pytest.approx(far_past)


class TestThinkingBudgetCases:
    def test_covers_every_real_thinking_tier(self) -> None:
        cases = run_thinking_budget_cases()
        assert len(cases) == len(THINKING_MS)
        assert {c.delay_seconds for c in cases} == {ms / 1000.0 for ms in THINKING_MS.values()}

    def test_the_decisive_finding_holds_on_the_real_run(self) -> None:
        """The core adversarial result: run for real, not asserted."""
        cases = run_thinking_budget_cases()
        assert linear_decay_collapses_argus_real_tiers(cases) is True

    def test_as_dict_serialises(self) -> None:
        cases = run_thinking_budget_cases()
        d = cases[0].as_dict()
        assert "linear_decay_bps" in d and "argus_bps" in d


class TestSweptComparison:
    def test_covers_the_declared_delay_grid(self) -> None:
        points = swept_comparison()
        assert len(points) == len(SWEPT_DELAYS_SECONDS)

    def test_linear_decay_is_bounded_past_its_cap_on_the_real_sweep(self) -> None:
        points = swept_comparison()
        assert linear_decay_is_bounded_past_its_cap(points) is True

    def test_argus_cost_is_unbounded_on_the_real_sweep(self) -> None:
        points = swept_comparison()
        assert argus_cost_is_unbounded(points) is True


class TestAblation:
    def test_the_specific_decay_window_value_is_load_bearing(self) -> None:
        result = run_ablation()
        assert result.real_decay_window_collapses_real_tiers is True
        assert result.widened_decay_window_distinguishes_them is True
        assert result.the_specific_window_value_is_load_bearing is True


class TestCosts:
    def test_both_costs_are_measured_and_positive(self) -> None:
        costs = measure_costs()
        assert costs["linear_decay_us_per_call"] > 0
        assert costs["argus_us_per_call"] > 0


class TestScopeStatement:
    def test_names_what_is_not_claimed(self) -> None:
        assert "NOT claimed" in SCOPE_STATEMENT
        assert "decay_window" in SCOPE_STATEMENT
