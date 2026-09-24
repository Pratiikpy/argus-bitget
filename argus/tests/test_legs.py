"""Multi-leg coordination — the second of two execution-plumbing gaps the plan names ("no
idempotency, no multi-leg orders"). Deliberately not `nautilus_trader`'s `ContingencyType`
(Oco/Oto/Ouo models sequential linkage); this measures **legging risk** — one leg filled, the
other not — which that taxonomy does not name. See `execution/legs.py`'s module docstring for why.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from argus.execution.legs import (
    Leg,
    LegExposureError,
    LegGroup,
    LegGroupError,
    assess_legging_risk,
    leg_exposure_notional,
)
from argus.execution.orders import Order, OrderState

T0 = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


def _order(oid: str, *, symbol: str = "NVDAUSDT", side: str = "BUY", qty: str = "10") -> Order:
    return Order(
        client_order_id=oid, symbol=symbol, side=side, quantity=Decimal(qty),
        approved_intent_hash="h",
    )


def _fill(order: Order, quantity: Decimal, *, at: datetime) -> None:
    order.transition(OrderState.SUBMITTED, at=at, reason="submitted to venue")
    order.transition(OrderState.ACCEPTED, at=at, reason="accepted")
    order.apply_fill(quantity, at=at)


class TestLegGroupConstruction:
    def test_a_single_leg_is_rejected(self) -> None:
        with pytest.raises(LegGroupError):
            LegGroup(group_id="g1", legs=(Leg(order=_order("a"), role="primary"),))

    def test_the_same_order_twice_is_rejected(self) -> None:
        same = _order("a")
        with pytest.raises(LegGroupError):
            LegGroup(
                group_id="g1",
                legs=(Leg(order=same, role="primary"), Leg(order=same, role="hedge")),
            )

    def test_two_distinct_legs_construct_cleanly(self) -> None:
        group = LegGroup(
            group_id="g1",
            legs=(Leg(order=_order("a"), role="primary"), Leg(order=_order("b"), role="hedge")),
        )
        assert len(group.legs) == 2


class TestFillStateProperties:
    def test_neither_leg_filled_reads_none_filled(self) -> None:
        group = LegGroup(
            group_id="g1",
            legs=(Leg(order=_order("a"), role="p"), Leg(order=_order("b"), role="h")),
        )
        assert group.none_filled
        assert not group.all_filled
        assert not group.is_legging

    def test_both_legs_fully_filled_reads_all_filled(self) -> None:
        a, b = _order("a"), _order("b")
        _fill(a, Decimal("10"), at=T0)
        _fill(b, Decimal("10"), at=T0)
        group = LegGroup(group_id="g1", legs=(Leg(order=a, role="p"), Leg(order=b, role="h")))
        assert group.all_filled
        assert not group.none_filled
        assert not group.is_legging

    def test_one_leg_filled_and_the_other_not_is_legging(self) -> None:
        a, b = _order("a"), _order("b")
        _fill(a, Decimal("10"), at=T0)
        group = LegGroup(group_id="g1", legs=(Leg(order=a, role="p"), Leg(order=b, role="h")))
        assert group.is_legging
        assert not group.all_filled
        assert not group.none_filled

    def test_a_partial_fill_on_one_leg_is_also_legging(self) -> None:
        a, b = _order("a"), _order("b")
        _fill(a, Decimal("4"), at=T0)  # partial, qty=10
        _fill(b, Decimal("10"), at=T0)  # full
        group = LegGroup(group_id="g1", legs=(Leg(order=a, role="p"), Leg(order=b, role="h")))
        assert group.is_legging


class TestAnyLegDead:
    def test_a_denied_leg_with_no_fill_anywhere_is_not_yet_dangerous(self) -> None:
        a, b = _order("a"), _order("b")
        a.transition(OrderState.DENIED, at=T0, reason="constitution refused")
        group = LegGroup(group_id="g1", legs=(Leg(order=a, role="p"), Leg(order=b, role="h")))
        assert group.any_leg_dead
        assert group.none_filled  # nothing has filled — no naked exposure exists yet

    def test_a_dead_leg_alongside_a_filled_one_is_the_dangerous_case(self) -> None:
        a, b = _order("a"), _order("b")
        _fill(a, Decimal("10"), at=T0)
        b.transition(OrderState.SUBMITTED, at=T0, reason="submitted")
        b.transition(OrderState.REJECTED, at=T0, reason="venue rejected")
        group = LegGroup(group_id="g1", legs=(Leg(order=a, role="p"), Leg(order=b, role="h")))
        assert group.any_leg_dead
        assert not group.none_filled


class TestFirstFillAt:
    def test_none_before_anything_fills(self) -> None:
        group = LegGroup(
            group_id="g1",
            legs=(Leg(order=_order("a"), role="p"), Leg(order=_order("b"), role="h")),
        )
        assert group.first_fill_at() is None

    def test_reads_the_earlier_of_the_two_legs_fills(self) -> None:
        a, b = _order("a"), _order("b")
        t1, t2 = T0, T0 + timedelta(minutes=5)
        _fill(a, Decimal("10"), at=t2)
        _fill(b, Decimal("10"), at=t1)
        group = LegGroup(group_id="g1", legs=(Leg(order=a, role="p"), Leg(order=b, role="h")))
        assert group.first_fill_at() == t1


class TestLegExposureNotional:
    def test_both_legs_filled_opposite_sides_equal_size_nets_to_zero(self) -> None:
        a = _order("a", symbol="NVDAUSDT", side="BUY", qty="10")
        b = _order("b", symbol="TSLAUSDT", side="SELL", qty="10")
        _fill(a, Decimal("10"), at=T0)
        _fill(b, Decimal("10"), at=T0)
        group = LegGroup(group_id="g1", legs=(Leg(order=a, role="p"), Leg(order=b, role="h")))
        exposure = leg_exposure_notional(
            group, marks={"NVDAUSDT": Decimal("100"), "TSLAUSDT": Decimal("100")}
        )
        assert exposure == Decimal("0")

    def test_only_one_leg_filled_reads_the_full_naked_notional(self) -> None:
        a = _order("a", symbol="NVDAUSDT", side="BUY", qty="10")
        b = _order("b", symbol="TSLAUSDT", side="SELL", qty="10")
        _fill(a, Decimal("10"), at=T0)
        group = LegGroup(group_id="g1", legs=(Leg(order=a, role="p"), Leg(order=b, role="h")))
        exposure = leg_exposure_notional(group, marks={"NVDAUSDT": Decimal("100")})
        assert exposure == Decimal("1000")  # 10 * 100, long, unhedged

    def test_a_missing_mark_for_a_filled_leg_raises_rather_than_reading_zero(self) -> None:
        a = _order("a", symbol="NVDAUSDT", side="BUY", qty="10")
        b = _order("b", symbol="TSLAUSDT", side="SELL", qty="10")
        _fill(a, Decimal("10"), at=T0)
        group = LegGroup(group_id="g1", legs=(Leg(order=a, role="p"), Leg(order=b, role="h")))
        with pytest.raises(LegExposureError):
            leg_exposure_notional(group, marks={})

    def test_a_ratio_spread_weight_is_applied(self) -> None:
        a = _order("a", symbol="NVDAUSDT", side="BUY", qty="20")
        b = _order("b", symbol="TSLAUSDT", side="SELL", qty="10")
        _fill(a, Decimal("20"), at=T0)
        _fill(b, Decimal("10"), at=T0)
        group = LegGroup(
            group_id="g1",
            legs=(
                Leg(order=a, role="p", weight=Decimal("0.5")),  # 20 * 0.5 = 10, nets vs b's 10
                Leg(order=b, role="h", weight=Decimal("1")),
            ),
        )
        exposure = leg_exposure_notional(
            group, marks={"NVDAUSDT": Decimal("100"), "TSLAUSDT": Decimal("100")}
        )
        assert exposure == Decimal("0")


class TestAssessLeggingRisk:
    def test_nothing_filled_yet_continues(self) -> None:
        group = LegGroup(
            group_id="g1",
            legs=(Leg(order=_order("a"), role="p"), Leg(order=_order("b"), role="h")),
        )
        verdict = assess_legging_risk(group, now=T0, max_legging_seconds=60)
        assert verdict.demands == "continue"

    def test_everything_filled_continues(self) -> None:
        a, b = _order("a"), _order("b")
        _fill(a, Decimal("10"), at=T0)
        _fill(b, Decimal("10"), at=T0)
        group = LegGroup(group_id="g1", legs=(Leg(order=a, role="p"), Leg(order=b, role="h")))
        verdict = assess_legging_risk(group, now=T0, max_legging_seconds=60)
        assert verdict.demands == "continue"

    def test_legging_inside_the_window_continues(self) -> None:
        a, b = _order("a"), _order("b")
        _fill(a, Decimal("10"), at=T0)
        group = LegGroup(group_id="g1", legs=(Leg(order=a, role="p"), Leg(order=b, role="h")))
        verdict = assess_legging_risk(
            group, now=T0 + timedelta(seconds=30), max_legging_seconds=60,
        )
        assert verdict.demands == "continue"
        assert "30" in verdict.reason

    def test_legging_past_the_window_demands_unwind(self) -> None:
        a, b = _order("a"), _order("b")
        _fill(a, Decimal("10"), at=T0)
        group = LegGroup(group_id="g1", legs=(Leg(order=a, role="p"), Leg(order=b, role="h")))
        verdict = assess_legging_risk(
            group, now=T0 + timedelta(seconds=90), max_legging_seconds=60,
        )
        assert verdict.demands == "unwind"
        assert "90" in verdict.reason

    def test_a_dead_leg_with_a_fill_elsewhere_demands_unwind_regardless_of_the_clock(self) -> None:
        a, b = _order("a"), _order("b")
        _fill(a, Decimal("10"), at=T0)
        b.transition(OrderState.SUBMITTED, at=T0, reason="submitted")
        b.transition(OrderState.REJECTED, at=T0, reason="venue rejected")
        group = LegGroup(group_id="g1", legs=(Leg(order=a, role="p"), Leg(order=b, role="h")))
        # Even at the very same instant, before any time-box could possibly have elapsed.
        verdict = assess_legging_risk(group, now=T0, max_legging_seconds=6000)
        assert verdict.demands == "unwind"
        assert "terminal" in verdict.reason

    def test_a_dead_leg_with_nothing_filled_anywhere_does_not_demand_unwind(self) -> None:
        """Nothing is open, so there is nothing to unwind — a denied leg before anything filled
        is simply a group that never started, not a legging incident."""
        a, b = _order("a"), _order("b")
        a.transition(OrderState.DENIED, at=T0, reason="constitution refused")
        group = LegGroup(group_id="g1", legs=(Leg(order=a, role="p"), Leg(order=b, role="h")))
        verdict = assess_legging_risk(group, now=T0, max_legging_seconds=60)
        assert verdict.demands == "continue"
