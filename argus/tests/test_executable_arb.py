"""Tests for `research/executable_arb.py`, the exact two-book arbitrage optimum.

Constructed books only; no network. The same function is checked against HiGHS on every real
captured snapshot in `tests/test_general_arb_comparison.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from argus.market.depth import Level, OrderBook
from argus.research.executable_arb import ExecutableArbError, best_arbitrage, leg_optimum

FEE = Decimal("0.0006")


def lv(price: str, qty: str) -> Level:
    return Level(Decimal(price), Decimal(qty))


def book(symbol: str, bids: list[Level], asks: list[Level]) -> OrderBook:
    return OrderBook(symbol, datetime(2026, 9, 25, tzinfo=UTC), tuple(bids), tuple(asks))


class TestLegOptimum:
    def test_nothing_to_do_when_the_touch_does_not_clear_the_fees(self) -> None:
        leg = leg_optimum([lv("100", "5")], [lv("100.1", "5")], buy_fee=FEE, sell_fee=FEE)
        assert leg.quantity == 0
        assert leg.net == 0
        assert leg.top_of_book_edge_bps is not None and leg.top_of_book_edge_bps < 0

    def test_walks_levels_while_the_marginal_unit_still_pays(self) -> None:
        asks = [lv("100", "1"), lv("100.05", "2"), lv("100.50", "10")]
        bids = [lv("100.40", "2"), lv("100.30", "5")]
        leg = leg_optimum(asks, bids, buy_fee=FEE, sell_fee=FEE)
        # unit-by-unit marginal: 1 @ (100.40, 100.00) pays, 1 @ (100.40, 100.05) pays,
        # 1 @ (100.30, 100.05) pays 100.30*.9994 - 100.05*1.0006 = 0.1298 > 0, then the 100.50
        # ask is above every bid: stop at 3.
        assert leg.quantity == Decimal("3")
        expected = (Decimal("100.40") * 2 + Decimal("100.30")) * (1 - FEE) - (
            Decimal("100") + Decimal("100.05") * 2) * (1 + FEE)
        assert leg.net == expected
        assert leg.buy_levels == 2 and leg.sell_levels == 2

    def test_a_notional_cap_binds_partway_through_a_level(self) -> None:
        asks = [lv("100", "10")]
        bids = [lv("101", "10")]
        leg = leg_optimum(asks, bids, buy_fee=FEE, sell_fee=FEE, max_notional=Decimal("500"))
        assert leg.capped
        # Decimal division leaves the last of 28 digits to rounding; the budget is spent to it.
        assert abs(leg.buy_notional * (1 + FEE) - Decimal("500")) < Decimal("1e-20")
        assert leg.net > 0

    def test_a_quantity_cap_binds(self) -> None:
        leg = leg_optimum([lv("100", "10")], [lv("101", "10")], buy_fee=FEE, sell_fee=FEE,
                          max_quantity=Decimal("2.5"))
        assert leg.capped and leg.quantity == Decimal("2.5")

    def test_empty_ladders_trade_nothing_and_have_no_touch(self) -> None:
        leg = leg_optimum([], [lv("100", "1")], buy_fee=FEE, sell_fee=FEE)
        assert leg.quantity == 0 and leg.top_of_book_edge_bps is None

    @pytest.mark.parametrize("fee", [Decimal("-0.0001"), Decimal("1")])
    def test_a_fee_outside_zero_to_one_is_refused(self, fee: Decimal) -> None:
        with pytest.raises(ExecutableArbError):
            leg_optimum([lv("100", "1")], [lv("101", "1")], buy_fee=fee, sell_fee=FEE)

    def test_a_negative_cap_is_refused(self) -> None:
        with pytest.raises(ExecutableArbError):
            leg_optimum([lv("100", "1")], [lv("101", "1")], buy_fee=FEE, sell_fee=FEE,
                        max_notional=Decimal("-1"))


class TestBestArbitrage:
    def test_picks_the_direction_that_pays_and_applies_survival(self) -> None:
        spot = book("RXUSDT", [lv("99.90", "5")], [lv("100.00", "5")])
        perp = book("XUSDT", [lv("100.40", "5")], [lv("100.45", "5")])
        arb = best_arbitrage(spot, perp, fee_a=Decimal("0.001"), fee_b=FEE)
        assert arb.monetizable
        assert arb.best.buy_venue == "RXUSDT" and arb.best.sell_venue == "XUSDT"
        assert arb.expected_net == arb.best.net * Decimal("0.92") * Decimal("0.95")

    def test_a_wide_spot_touch_behind_a_wide_mid_gap_is_refused(self) -> None:
        # Mid-to-mid gap 25bps, which a flat 14.6bps hurdle accepts; the touch cannot deliver it.
        spot = book("RWIDEUSDT", [lv("99.70", "50")], [lv("100.30", "50")])
        perp = book("WIDEUSDT", [lv("100.24", "50")], [lv("100.26", "50")])
        arb = best_arbitrage(spot, perp, fee_a=Decimal("0.001"), fee_b=FEE)
        assert not arb.monetizable
        assert arb.expected_net == 0

    def test_survival_probabilities_outside_zero_to_one_are_refused(self) -> None:
        spot = book("RXUSDT", [lv("99.90", "5")], [lv("100.00", "5")])
        perp = book("XUSDT", [lv("100.40", "5")], [lv("100.45", "5")])
        with pytest.raises(ExecutableArbError):
            best_arbitrage(spot, perp, fee_a=FEE, fee_b=FEE,
                           execution_probability=Decimal("1.2"))
