"""Order state machine tests.

Checked against NautilusTrader's real enum at ``crates/model/src/enums.rs:1304-1388``, not against
memory of it. Where ARGUS departs (the UNKNOWN state), the test says so.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from argus.execution.orders import (
    DuplicateOrder,
    IllegalTransition,
    Order,
    OrderBook,
    OrderState,
    UnauthorisedOrder,
)

UTC = ZoneInfo("UTC")
T0 = datetime(2026, 3, 9, 14, 30, tzinfo=UTC)


def _order(oid: str = "c-1", qty: str = "100") -> Order:
    return Order(
        client_order_id=oid,
        symbol="rNVDA",
        side="SELL",
        quantity=Decimal(qty),
        approved_intent_hash="abc123",
    )


class TestPredicatesMatchNautilus:
    """Mirrors is_open / is_inflight / is_closed / is_cancellable from enums.rs:1345-1388."""

    def test_submitted_is_inflight_but_not_open(self) -> None:
        """enums.rs:1340-1343 documents this exact trap."""
        assert OrderState.SUBMITTED.is_inflight is True
        assert OrderState.SUBMITTED.is_open is False
        assert OrderState.SUBMITTED.is_live is True  # must still be reconciled

    def test_pending_states_are_open_and_inflight_simultaneously(self) -> None:
        for s in (OrderState.PENDING_UPDATE, OrderState.PENDING_CANCEL):
            assert s.is_open is True
            assert s.is_inflight is True

    def test_closed_set_matches_nautilus(self) -> None:
        expected = {
            OrderState.DENIED, OrderState.REJECTED, OrderState.CANCELLED,
            OrderState.EXPIRED, OrderState.FILLED, OrderState.VOIDED,
        }
        assert {s for s in OrderState if s.is_closed} == expected

    def test_cancellable_set_matches_nautilus(self) -> None:
        expected = {
            OrderState.ACCEPTED, OrderState.TRIGGERED,
            OrderState.PENDING_UPDATE, OrderState.PARTIALLY_FILLED,
        }
        assert {s for s in OrderState if s.is_cancellable} == expected

    def test_no_state_is_both_open_and_closed(self) -> None:
        assert not any(s.is_open and s.is_closed for s in OrderState)


class TestDeniedIsNotRejected:
    """ARGUS needs this distinction more than Nautilus does: 'the Constitution stopped it' and
    'Bitget refused it' are different facts, and Track 2 is judged on the first."""

    def test_denied_never_reaches_the_venue(self) -> None:
        book = OrderBook()
        o = book.deny(_order(), at=T0, reason="concentration cap binds")
        assert o.state is OrderState.DENIED
        assert o.venue_order_id is None
        assert o.state.is_closed

    def test_denied_and_rejected_are_distinct_states(self) -> None:
        assert OrderState.DENIED is not OrderState.REJECTED

    def test_denied_is_terminal(self) -> None:
        book = OrderBook()
        o = book.deny(_order(), at=T0, reason="veto")
        with pytest.raises(IllegalTransition):
            o.transition(OrderState.SUBMITTED, at=T0, reason="sneak past the gate")


class TestTimeoutIsNotRejection:
    """ARGUS addition — Nautilus has no UNKNOWN. The failure being prevented: assuming a timed-out
    request failed, retrying, and creating duplicate exposure."""

    def test_timeout_lands_in_unknown_and_stays_live(self) -> None:
        book = OrderBook()
        book.submit(_order(), at=T0)
        book.mark_unknown("c-1", at=T0 + timedelta(seconds=3))

        o = book.get("c-1")
        assert o.state is OrderState.UNKNOWN
        assert o.state.is_live is True
        assert o.state.is_closed is False
        assert o in book.live_orders()
        assert o in book.needs_reconciliation()

    def test_unknown_can_only_be_resolved_by_reconciliation(self) -> None:
        """Every exit from UNKNOWN goes through the venue. There is no path that assumes."""
        book = OrderBook()
        book.submit(_order(), at=T0)
        book.mark_unknown("c-1", at=T0)

        book.reconcile(
            "c-1", venue_state=OrderState.FILLED, at=T0, venue_filled=Decimal("100")
        )
        o = book.get("c-1")
        assert o.state is OrderState.FILLED
        assert o.filled_quantity == Decimal("100")

    def test_venue_may_report_a_fill_we_never_saw(self) -> None:
        """The timeout hid a real fill. The venue is the source of truth and we adopt it."""
        book = OrderBook()
        book.submit(_order(), at=T0)
        book.mark_unknown("c-1", at=T0)
        book.reconcile(
            "c-1", venue_state=OrderState.PARTIALLY_FILLED, at=T0, venue_filled=Decimal("40")
        )
        assert book.get("c-1").filled_quantity == Decimal("40")
        assert book.get("c-1").residual == Decimal("60")

    def test_impossible_reconciliation_is_refused(self) -> None:
        book = OrderBook()
        book.submit(_order(qty="100"), at=T0)
        book.mark_unknown("c-1", at=T0)
        with pytest.raises(ValueError, match="impossible state"):
            book.reconcile(
                "c-1", venue_state=OrderState.FILLED, at=T0, venue_filled=Decimal("250")
            )


class TestDuplicateOrders:
    """Challenge Mode control: a replayed authorisation must not create a second position."""

    def test_resubmitting_the_same_id_raises(self) -> None:
        book = OrderBook()
        book.submit(_order("c-1"), at=T0)
        with pytest.raises(DuplicateOrder, match="second position"):
            book.submit(_order("c-1"), at=T0)

    def test_book_holds_one_order_after_a_duplicate_attempt(self) -> None:
        book = OrderBook()
        book.submit(_order("c-1"), at=T0)
        with pytest.raises(DuplicateOrder):
            book.submit(_order("c-1"), at=T0)
        assert len(book) == 1


class TestAuthorisation:
    def test_order_without_approved_intent_hash_is_refused_at_construction(self) -> None:
        with pytest.raises(UnauthorisedOrder, match="unauthorised"):
            Order(
                client_order_id="c-9", symbol="rNVDA", side="SELL",
                quantity=Decimal("10"), approved_intent_hash="   ",
            )

    def test_zero_quantity_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            Order(
                client_order_id="c-9", symbol="rNVDA", side="SELL",
                quantity=Decimal("0"), approved_intent_hash="abc",
            )


class TestFills:
    def test_partial_fill_leaves_a_residual_to_replan(self) -> None:
        book = OrderBook()
        o = book.submit(_order(qty="100"), at=T0)
        o.transition(OrderState.ACCEPTED, at=T0, reason="ack")
        o.apply_fill(Decimal("40"), at=T0)

        assert o.state is OrderState.PARTIALLY_FILLED
        assert o.residual == Decimal("60")
        assert o.state.is_live

    def test_fills_completing_the_quantity_move_to_filled(self) -> None:
        book = OrderBook()
        o = book.submit(_order(qty="100"), at=T0)
        o.transition(OrderState.ACCEPTED, at=T0, reason="ack")
        o.apply_fill(Decimal("40"), at=T0)
        o.apply_fill(Decimal("60"), at=T0)

        assert o.state is OrderState.FILLED
        assert o.residual == Decimal("0")
        assert o.state.is_closed

    def test_overfill_raises_rather_than_being_absorbed(self) -> None:
        book = OrderBook()
        o = book.submit(_order(qty="100"), at=T0)
        o.transition(OrderState.ACCEPTED, at=T0, reason="ack")
        with pytest.raises(ValueError, match="never something to absorb silently"):
            o.apply_fill(Decimal("101"), at=T0)

    def test_a_filled_order_can_still_be_voided(self) -> None:
        """Venues retroactively bust fills. Without VOIDED that correction has nowhere to go and
        our position record silently disagrees with the venue's."""
        book = OrderBook()
        o = book.submit(_order(qty="100"), at=T0)
        o.transition(OrderState.ACCEPTED, at=T0, reason="ack")
        o.apply_fill(Decimal("100"), at=T0)
        assert o.state is OrderState.FILLED

        o.transition(OrderState.VOIDED, at=T0 + timedelta(minutes=5), reason="venue busted fill")
        assert o.state is OrderState.VOIDED


class TestIllegalTransitions:
    def test_cannot_jump_from_initialised_to_filled(self) -> None:
        o = _order()
        with pytest.raises(IllegalTransition, match="not a legal venue transition"):
            o.transition(OrderState.FILLED, at=T0, reason="wishful")

    def test_cancelled_is_terminal(self) -> None:
        book = OrderBook()
        o = book.submit(_order(), at=T0)
        o.transition(OrderState.ACCEPTED, at=T0, reason="ack")
        o.transition(OrderState.CANCELLED, at=T0, reason="pulled")
        with pytest.raises(IllegalTransition):
            o.transition(OrderState.ACCEPTED, at=T0, reason="undo")

    def test_cancel_request_may_be_refused_and_the_order_keeps_working(self) -> None:
        """PENDING_CANCEL -> ACCEPTED is legal: the venue refused the cancel."""
        book = OrderBook()
        o = book.submit(_order(), at=T0)
        o.transition(OrderState.ACCEPTED, at=T0, reason="ack")
        o.transition(OrderState.PENDING_CANCEL, at=T0, reason="cancel requested")
        o.transition(OrderState.ACCEPTED, at=T0, reason="venue refused cancel")
        assert o.state.is_open

    def test_cancel_can_race_a_fill(self) -> None:
        book = OrderBook()
        o = book.submit(_order(), at=T0)
        o.transition(OrderState.ACCEPTED, at=T0, reason="ack")
        o.transition(OrderState.PENDING_CANCEL, at=T0, reason="cancel requested")
        o.transition(OrderState.FILLED, at=T0, reason="filled before cancel landed")
        assert o.state is OrderState.FILLED


class TestHistoryIsTheAuditArtefact:
    def test_every_transition_is_recorded_in_order(self) -> None:
        book = OrderBook()
        o = book.submit(_order(), at=T0)
        o.transition(OrderState.ACCEPTED, at=T0 + timedelta(seconds=1), reason="ack")
        o.apply_fill(Decimal("100"), at=T0 + timedelta(seconds=2))

        assert [(t.frm, t.to) for t in o.history] == [
            (OrderState.INITIALISED, OrderState.SUBMITTED),
            (OrderState.SUBMITTED, OrderState.ACCEPTED),
            (OrderState.ACCEPTED, OrderState.FILLED),
        ]
        assert all(t.reason for t in o.history)

    def test_order_carries_the_authorising_intent_hash(self) -> None:
        """Execution Proof: the approved order and the executed order must be linkable."""
        book = OrderBook()
        o = book.submit(_order(), at=T0)
        assert o.approved_intent_hash == "abc123"
