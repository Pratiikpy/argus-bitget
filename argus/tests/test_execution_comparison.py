"""Tests for the ARGUS-vs-crypto_sor cross-asset execution comparison.

Live-network AND real-subprocess tests: real Bitget order-book/funding data, and a real Node/
ts-node subprocess running crypto_sor's real, vendored `CompositeOrderBook.newOrder()`, are the
whole point of the comparison. Each subprocess spawn costs real wall-clock time (~2s, measured —
see `TestCosts`), so every expensive function is wrapped in a module-scoped fixture and called
once per test session. See `eval/execution_comparison.py`'s module docstring for the findings
these tests pin.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from argus.cost.model import CostModel
from argus.desk.execution import HedgeLegQuote
from argus.eval.execution_comparison import (
    LEG_PAIRS,
    SCOPE_STATEMENT,
    measure_costs,
    render,
    run_base_case,
    run_divergence_sweep,
    run_failure_cases,
    run_leg_pair_sweep,
    run_reproducibility_check,
)


def _recorded_legs() -> tuple[HedgeLegQuote, HedgeLegQuote]:
    """The legs as measured live and recorded in `data/execution_comparison.json` (base_case):
    NVDA 0.984bps slippage and zero funding, BTC 0.0066bps slippage and 0.81bps funding.

    The divergence the comparison is about only exists while the leg that is cheaper to enter is
    the dearer one to hold. That is a market state, not a property of either system: on
    2026-09-23 live funding no longer had that shape, no break-even existed, and four tests that
    asserted today's market would reproduce it failed. The mechanism is pinned on the recorded
    measurement, through the real crypto_sor subprocess; the live run is still exercised below and
    must produce a well-formed result, whatever the market does."""
    return (
        HedgeLegQuote(symbol="NVDAUSDT", slippage_bps=Decimal("0.9842284392337896788902555422"),
                      cost_model=CostModel.bitget_perp(funding_rate=Decimal("0"))),
        HedgeLegQuote(symbol="BTCUSDT", slippage_bps=Decimal("0.006579817462703949667028337300"),
                      cost_model=CostModel.bitget_perp(funding_rate=Decimal("0.000081"))),
    )


@pytest.fixture(scope="module")
def base_case() -> dict:
    return run_base_case(legs=_recorded_legs())


@pytest.fixture(scope="module")
def sweep() -> dict:
    return run_divergence_sweep(legs=_recorded_legs())


@pytest.fixture(scope="module")
def live_base_case() -> dict:
    return run_base_case()


class TestTheLiveRunIsWellFormed:
    def test_live_legs_are_measured_and_both_systems_answer(self, live_base_case: dict) -> None:
        assert live_base_case["real_crypto_sor_pick_at_entry"] in ("NVDAUSDT", "BTCUSDT")
        assert live_base_case["argus_pick_at_entry"] in ("NVDAUSDT", "BTCUSDT")
        if live_base_case["break_even_holding_days"] is None:
            assert live_base_case["argus_pick_past_break_even"] is None


@pytest.fixture(scope="module")
def pairs() -> dict:
    return run_leg_pair_sweep()


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
    def test_the_two_real_systems_agree_at_zero_holding(self, base_case: dict) -> None:
        assert base_case["agree_at_entry"]

    def test_crypto_leg_has_materially_tighter_slippage(self, base_case: dict) -> None:
        assert float(base_case["crypto_slippage_bps"]) < float(base_case["rtoken_slippage_bps"])

    def test_a_real_break_even_horizon_was_found(self, base_case: dict) -> None:
        assert base_case["break_even_holding_days"] is not None

    def test_the_real_systems_diverge_past_the_break_even(self, base_case: dict) -> None:
        assert base_case["diverges_past_break_even"]
        assert base_case["argus_pick_past_break_even"] != base_case["real_crypto_sor_pick_at_entry"]


class TestDivergenceSweep:
    def test_at_least_one_point_agrees_and_one_disagrees(self, sweep: dict) -> None:
        """Not every holding-day value should disagree — zero holding is definitionally an
        agreement, since funding has not accrued yet."""
        assert 0 < sweep["n_disagreements"] < sweep["n_points"]

    def test_the_real_crypto_sor_pick_never_changes_across_the_sweep(self, sweep: dict) -> None:
        picks = {p["real_crypto_sor_pick"] for p in sweep["points"]}
        assert len(picks) == 1

    def test_the_argus_pick_does_change_across_the_sweep(self, sweep: dict) -> None:
        picks = {p["argus_pick"] for p in sweep["points"]}
        assert len(picks) > 1


class TestLegPairSweep:
    def test_all_configured_pairs_were_checked(self, pairs: dict) -> None:
        assert pairs["n_pairs"] == len(LEG_PAIRS)

    def test_every_live_pair_diverges_exactly_when_a_break_even_exists(self, pairs: dict) -> None:
        """How many pairs diverge is a fact about today's funding rates (on 2026-09-23 it was 3 of
        5, where the recorded run had nearly all); what must hold on any day is that a pair
        diverges only past a real break-even, and never without one."""
        assert pairs["n_ok"] > 0
        for row in (r for r in pairs["results"] if "error" not in r):
            if row["break_even_holding_days"] is None:
                assert not row["diverges_past_break_even"]


class TestFailureCases:
    def test_a_single_leg_fills_without_error(self, failures: dict) -> None:
        case = failures["single_leg"]
        assert not case["real_raised"]
        assert case["executions"]

    def test_an_exchange_filter_matching_nothing_returns_empty_not_an_error(
        self, failures: dict
    ) -> None:
        case = failures["exchange_filter_matches_nothing"]
        assert case["real_returns_empty_list_not_an_error"]
        assert not case["real_raised"]

    def test_an_order_larger_than_the_book_silently_underfills(self, failures: dict) -> None:
        case = failures["order_larger_than_book"]
        assert case["real_silently_underfills_with_no_flag"]
        assert not case["real_raised"]


class TestCosts:
    def test_the_subprocess_is_measurably_slower_than_pure_python(self, costs: dict) -> None:
        assert costs["crypto_sor_subprocess_seconds_per_call"] > 0
        assert costs["argus_pure_python_seconds_per_call"] > 0
        assert costs["argus_faster_by_factor"] is not None
        assert costs["argus_faster_by_factor"] > 1


class TestReproducibility:
    def test_repeated_subprocess_runs_agree(self, reproducibility: dict) -> None:
        assert reproducibility["identical"]


class TestMain:
    def test_render_produces_readable_text(
        self, base_case: dict, sweep: dict, pairs: dict, failures: dict, costs: dict,
        reproducibility: dict,
    ) -> None:
        report = {
            "base_case": base_case,
            "divergence_sweep": sweep,
            "leg_pair_sweep": pairs,
            "failure_cases": failures,
            "costs": costs,
            "reproducibility": reproducibility,
            "scope_statement": SCOPE_STATEMENT,
        }
        text = render(report)
        assert "CROSS-ASSET EXECUTION" in text
        assert "break-even" in text


def test_scope_statement_is_honest_about_what_is_not_claimed() -> None:
    assert "NOT claimed" in SCOPE_STATEMENT
    assert "validated hedge" in SCOPE_STATEMENT
