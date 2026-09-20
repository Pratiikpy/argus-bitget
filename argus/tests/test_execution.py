"""Tests for `argus.desk.execution` — pure unit tests, no live network, no subprocess.

The live-data, real-subprocess comparison against crypto_sor's real router is
`tests/test_execution_comparison.py`.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from argus.cost.model import CostModel
from argus.desk.execution import (
    ExecutionError,
    HedgeLegQuote,
    break_even_holding_days,
    choose_hedge_leg,
)


def _leg(symbol: str, slippage_bps: str, funding_rate: str = "0") -> HedgeLegQuote:
    return HedgeLegQuote(
        symbol=symbol,
        slippage_bps=Decimal(slippage_bps),
        cost_model=CostModel.bitget_perp(funding_rate=Decimal(funding_rate)),
    )


class TestChooseHedgeLeg:
    def test_picks_the_cheaper_leg_at_zero_holding(self) -> None:
        cheap = _leg("A", "0.01")
        expensive = _leg("B", "1.0")
        decision = choose_hedge_leg([cheap, expensive], Decimal("5000"), holding_days=Decimal("0"))
        assert decision.cheapest_total_cost_symbol == "A"
        assert decision.cheapest_entry_cost_symbol == "A"
        assert decision.agrees_with_entry_only_routing

    def test_funding_can_flip_the_choice_over_time(self) -> None:
        """Matches the real, measured NVDAUSDT/BTCUSDT case: tighter execution, real funding."""
        rtoken = _leg("RTOKEN", "0.9", "0")
        crypto = _leg("CRYPTO", "0.007", "0.000081")
        at_zero = choose_hedge_leg([rtoken, crypto], Decimal("5000"), holding_days=Decimal("0"))
        assert at_zero.cheapest_total_cost_symbol == "CRYPTO"

        later = choose_hedge_leg([rtoken, crypto], Decimal("5000"), holding_days=Decimal("7"))
        assert later.cheapest_total_cost_symbol == "RTOKEN"
        assert not later.agrees_with_entry_only_routing

    def test_refuses_fewer_than_two_legs(self) -> None:
        with pytest.raises(ExecutionError, match="at least two"):
            choose_hedge_leg([_leg("A", "0.1")], Decimal("5000"), holding_days=Decimal("0"))

    def test_refuses_non_positive_notional(self) -> None:
        with pytest.raises(ExecutionError, match="notional"):
            choose_hedge_leg(
                [_leg("A", "0.1"), _leg("B", "0.2")], Decimal("0"), holding_days=Decimal("0"),
            )

    def test_refuses_negative_holding_days(self) -> None:
        with pytest.raises(ExecutionError, match="holding_days"):
            choose_hedge_leg(
                [_leg("A", "0.1"), _leg("B", "0.2")], Decimal("5000"),
                holding_days=Decimal("-1"),
            )

    def test_costs_are_json_serialisable(self) -> None:
        import json

        decision = choose_hedge_leg(
            [_leg("A", "0.1"), _leg("B", "0.2")], Decimal("5000"), holding_days=Decimal("1"),
        )
        json.dumps(decision.as_dict())


class TestBreakEvenHoldingDays:
    def test_matches_the_real_measured_case(self) -> None:
        """Pinned against the real, measured NVDAUSDT/BTCUSDT break-even
        (`eval/execution_comparison.py`, 2026-09-15): entry saving ~0.98bps, funding 0.81bps
        per 8h settlement — three settlements (one day) already exceeds the saving."""
        rtoken = _leg("RTOKEN", "0.98", "0")
        crypto = _leg("CRYPTO", "0.007", "0.000081")
        be = break_even_holding_days(crypto, rtoken, notional=Decimal("5000"))
        assert be == Decimal("1")

    def test_none_when_the_cheaper_leg_never_loses(self) -> None:
        cheap_forever = _leg("CHEAP", "0.01", "0")
        expensive = _leg("EXPENSIVE", "0.5", "0")
        be = break_even_holding_days(cheap_forever, expensive, notional=Decimal("5000"))
        assert be is None

    def test_refuses_when_the_supposedly_cheaper_leg_is_not_cheaper(self) -> None:
        with pytest.raises(ExecutionError, match="not actually cheaper"):
            break_even_holding_days(
                _leg("A", "1.0"), _leg("B", "0.1"), notional=Decimal("5000"),
            )
