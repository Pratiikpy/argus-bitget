"""Simulated-session and exit-cost tests.

The claim under test is narrow and important: that a stress number here is **measured against a
book**, not a fee divided by a multiplier. The decisive property is that the simulation can return
an answer arithmetic structurally cannot — that a position is *unexitable at any price*.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from argus.desk.workbench import STANDARD_SCENARIOS, Scenario, stress_position
from argus.sim.book import Side
from argus.sim.market import (
    MarketError,
    SessionConfig,
    build_session,
    exit_cost_bps,
    maker_taker_split,
    measure_exit,
)

D = Decimal


class TestTheSessionIsAMarket:
    def test_warmup_produces_a_two_sided_book(self) -> None:
        s = build_session()
        assert s.book.best_bid is not None
        assert s.book.best_ask is not None
        assert s.book.best_bid < s.book.best_ask
        assert s.depth > 0

    def test_agents_learn_they_were_filled(self) -> None:
        """A real bug this caught: without routing, no agent knows its own position.

        Inventory skew can never engage and every maker_fill_rate reads 0. A simulated market in
        which nobody knows their own position is not a market.
        """
        s = build_session()
        split = maker_taker_split(s)
        assert split["maker"] > 0, "a warmup with no passive fills has not built a book"
        assert split["taker"] > 0
        mm = next(a for a in s.agents if a.agent_id == "mm")
        assert mm.position != 0, "the market maker traded and must know it"
        assert mm.maker_fill_rate == D("1"), "a market maker only ever rests"

    def test_a_thinner_session_really_is_thinner(self) -> None:
        deep = build_session(SessionConfig(depth_multiplier=D("1")))
        thin = build_session(SessionConfig(depth_multiplier=D("0.33")))
        assert thin.depth < deep.depth
        # Our measured off-hours regime is ~1/3 of RTH; the simulation should land near it
        # rather than merely in the right direction.
        assert D("0.2") < thin.depth / deep.depth < D("0.6")

    def test_the_session_is_deterministic(self) -> None:
        """A stress number that changes between runs cannot be argued about."""
        a = build_session(SessionConfig(seed=99))
        b = build_session(SessionConfig(seed=99))
        assert a.depth == b.depth
        assert a.book.best_bid == b.book.best_bid
        assert a.book.traded_volume == b.book.traded_volume

    def test_a_different_seed_gives_a_different_market(self) -> None:
        a = build_session(SessionConfig(seed=1))
        b = build_session(SessionConfig(seed=2))
        assert (a.depth, a.book.traded_volume) != (b.depth, b.book.traded_volume)

    def test_non_positive_depth_is_refused(self) -> None:
        with pytest.raises(MarketError, match="depth_multiplier"):
            SessionConfig(depth_multiplier=D("0"))

    def test_non_positive_reference_price_is_refused(self) -> None:
        with pytest.raises(MarketError, match="reference_price"):
            SessionConfig(reference_price=D("0"))


class TestExitIsMeasuredNotAssumed:
    def test_a_small_exit_barely_moves_the_book(self) -> None:
        s = build_session()
        m = measure_exit(s, quantity=D("50"), side=Side.SELL)
        assert m.fully_exited
        assert m.levels_walked == 1
        assert m.slippage_bps > 0, "even a touch-sized exit crosses the spread"
        assert m.slippage_bps < D("2")

    def test_a_larger_exit_walks_more_levels_and_costs_more(self) -> None:
        s = build_session()
        small = measure_exit(s, quantity=D("50"))
        s2 = build_session()
        large = measure_exit(s2, quantity=D("800"))
        assert large.levels_walked > small.levels_walked
        assert large.slippage_bps > small.slippage_bps

    def test_the_same_exit_costs_more_on_a_thin_book(self) -> None:
        """The relationship the arithmetic version asserts; here it is produced."""
        deep = exit_cost_bps(quantity=D("600"), depth_multiplier=D("1"), reference_price=D("100"))
        thin = exit_cost_bps(quantity=D("600"), depth_multiplier=D("0.33"),
                             reference_price=D("100"))
        assert thin.slippage_bps > deep.slippage_bps

    def test_a_position_can_be_unexitable_at_any_price(self) -> None:
        """The answer arithmetic structurally cannot give.

        A fee divided by a liquidity multiplier is always finite, so it always says the position
        can be left. On a thin book it sometimes cannot.
        """
        m = exit_cost_bps(quantity=D("50000"), depth_multiplier=D("0.33"),
                          reference_price=D("100"))
        assert not m.fully_exited
        assert m.unfilled > 0
        assert m.quantity_filled < m.quantity_requested

    def test_slippage_is_signed_as_a_cost_on_both_sides(self) -> None:
        """Selling below the mid and buying above it are both costs."""
        for side in (Side.SELL, Side.BUY):
            s = build_session()
            m = measure_exit(s, quantity=D("100"), side=side)
            assert m.slippage_bps > 0, f"{side} exit should report a positive cost"

    def test_vwap_is_worse_than_the_mid(self) -> None:
        s = build_session()
        m = measure_exit(s, quantity=D("400"), side=Side.SELL)
        assert m.vwap < m.mid_before

    def test_zero_quantity_is_refused(self) -> None:
        with pytest.raises(MarketError, match="quantity"):
            measure_exit(build_session(), quantity=D("0"))

    def test_a_one_sided_book_cannot_price_an_exit(self) -> None:
        """Rather than measuring against a price that does not exist."""
        s = build_session()
        s.book.cancel_all("mm")
        for agent in s.agents:
            s.book.cancel_all(agent.agent_id)
        s.book.cancel_all("seed")
        if s.book.mid is None:
            with pytest.raises(MarketError, match="one-sided"):
                measure_exit(s, quantity=D("10"))

    def test_the_measurement_serialises(self) -> None:
        m = exit_cost_bps(quantity=D("200"), depth_multiplier=D("1"), reference_price=D("100"))
        payload = m.as_dict()
        assert set(payload) >= {"requested", "filled", "fully_exited", "slippage_bps",
                                "levels_walked", "depth_available"}


class TestStressUsesTheSimulation:
    def test_stress_results_are_marked_as_measured(self) -> None:
        results = stress_position(
            quantity=D("500"), entry_price=D("100"), loss_tolerance_pct=D("10")
        )
        assert all(r.exit_measured for r in results)

    def test_the_arithmetic_fallback_is_marked_as_not_measured(self) -> None:
        results = stress_position(
            quantity=D("500"), entry_price=D("100"), loss_tolerance_pct=D("10"), simulate=False
        )
        assert not any(r.exit_measured for r in results)

    def test_an_unexitable_position_does_not_survive_however_good_its_mark(self) -> None:
        """The case the arithmetic version got wrong.

        `liquidity_collapse` loses only 3% — comfortably inside a 10% tolerance — so the old code
        reported `survives=True`. The simulation finds that a large part of the position cannot be
        liquidated at all, and a position you cannot leave has not survived.
        """
        results = stress_position(
            quantity=D("2000"), entry_price=D("100"), loss_tolerance_pct=D("10")
        )
        collapse = next(r for r in results if r.scenario == "liquidity_collapse")
        assert collapse.pnl_pct == D("-3")
        assert -collapse.pnl_pct < D("10"), "the mark alone is comfortably inside tolerance"
        assert not collapse.exitable
        assert collapse.unfilled_quantity > 0
        assert not collapse.survives, "a position that cannot be liquidated has not survived"

    def test_the_arithmetic_version_would_have_passed_it(self) -> None:
        """Stated as a test so the improvement is checkable rather than asserted."""
        arithmetic = stress_position(
            quantity=D("2000"), entry_price=D("100"), loss_tolerance_pct=D("10"), simulate=False
        )
        collapse = next(r for r in arithmetic if r.scenario == "liquidity_collapse")
        assert collapse.survives is True
        assert collapse.exitable is True

    def test_a_small_position_survives_the_same_scenario(self) -> None:
        """The finding is about size against depth, not about the scenario being unsurvivable."""
        results = stress_position(
            quantity=D("50"), entry_price=D("100"), loss_tolerance_pct=D("10")
        )
        collapse = next(r for r in results if r.scenario == "liquidity_collapse")
        assert collapse.exitable
        assert collapse.survives

    def test_a_breach_of_tolerance_still_fails(self) -> None:
        results = stress_position(
            quantity=D("50"), entry_price=D("100"), loss_tolerance_pct=D("10")
        )
        gap = next(r for r in results if r.scenario == "gap_down")
        assert gap.pnl_pct == D("-12")
        assert not gap.survives

    def test_measured_exit_cost_exceeds_the_bare_fee(self) -> None:
        """Fee plus realised slippage, not fee alone."""
        from argus.cost.model import CostModel

        fee = CostModel.bitget_perp().round_trip_bps()
        results = stress_position(
            quantity=D("500"), entry_price=D("100"), loss_tolerance_pct=D("10")
        )
        assert all(r.exit_cost_bps > fee for r in results)

    def test_every_standard_scenario_is_evaluated(self) -> None:
        results = stress_position(
            quantity=D("100"), entry_price=D("100"), loss_tolerance_pct=D("10")
        )
        assert [r.scenario for r in results] == [s.name for s in STANDARD_SCENARIOS]

    def test_a_custom_scenario_works(self) -> None:
        results = stress_position(
            quantity=D("100"), entry_price=D("100"), loss_tolerance_pct=D("5"),
            scenarios=(Scenario("flash_crash", D("-20"), D("0.1")),),
        )
        assert len(results) == 1
        assert results[0].scenario == "flash_crash"
        assert not results[0].survives
