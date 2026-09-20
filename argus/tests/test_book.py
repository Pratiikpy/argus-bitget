"""Foundation 3 — the real portfolio. Pins the arithmetic against the verified Nautilus formula,
the hedge-mode keying fact confirmed live, and the reconciliation bug caught and fixed before this
module shipped."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import ClassVar

import pytest

from argus.desk.book import (
    Book,
    HedgeLink,
    Lot,
    OpenOrderReservation,
    Position,
    PositionError,
    PositionSide,
    ReconciliationStatus,
    VenueMarginSnapshot,
    order_reservation,
    parse_venue_margin_snapshot,
    reconcile_position,
)
from argus.execution.orders import Order, OrderState, deterministic_client_order_id

T0 = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


def _lot(*, side: str, quantity: str, price: str, ts: datetime = T0) -> Lot:
    return Lot(
        order_id="o1", approved_intent_hash="h1", side=side,
        quantity=Decimal(quantity), price=Decimal(price), commission=Decimal("0"), ts_filled=ts,
    )


class TestLotValidation:
    def test_zero_quantity_raises(self) -> None:
        with pytest.raises(PositionError):
            _lot(side="buy", quantity="0", price="100")

    def test_negative_price_raises(self) -> None:
        with pytest.raises(PositionError):
            _lot(side="buy", quantity="1", price="-1")


class TestPositionOpeningAndAdding:
    """The weighted-average formula, verified directly against
    nautilus_trader/crates/model/src/position.rs:999-1010 before this module was written."""

    def test_a_single_opening_fill_sets_entry_price_exactly(self) -> None:
        position = Position(symbol="NVDAUSDT", side=PositionSide.LONG)
        position.apply_fill(_lot(side="buy", quantity="10", price="100"))
        assert position.quantity == Decimal("10")
        assert position.entry_price == Decimal("100")

    def test_a_second_add_produces_the_weighted_average(self) -> None:
        position = Position(symbol="NVDAUSDT", side=PositionSide.LONG)
        position.apply_fill(_lot(side="buy", quantity="10", price="100"))
        position.apply_fill(_lot(side="buy", quantity="10", price="120"))
        # (10*100 + 10*120) / 20 = 110, the exact Nautilus formula
        assert position.entry_price == Decimal("110")
        assert position.quantity == Decimal("20")

    def test_unequal_sizes_weight_correctly(self) -> None:
        position = Position(symbol="NVDAUSDT", side=PositionSide.LONG)
        position.apply_fill(_lot(side="buy", quantity="30", price="100"))
        position.apply_fill(_lot(side="buy", quantity="10", price="140"))
        # (30*100 + 10*140) / 40 = 110
        assert position.entry_price == Decimal("110")

    def test_short_side_opens_on_a_sell(self) -> None:
        position = Position(symbol="NVDAUSDT", side=PositionSide.SHORT)
        position.apply_fill(_lot(side="sell", quantity="5", price="200"))
        assert position.quantity == Decimal("5")
        assert position.entry_price == Decimal("200")

    def test_ts_opened_set_once_on_first_fill(self) -> None:
        t1 = T0
        t2 = datetime(2026, 9, 15, 13, 0, tzinfo=UTC)
        position = Position(symbol="NVDAUSDT", side=PositionSide.LONG)
        position.apply_fill(_lot(side="buy", quantity="1", price="100", ts=t1))
        position.apply_fill(_lot(side="buy", quantity="1", price="100", ts=t2))
        assert position.ts_opened == t1
        assert position.ts_last == t2


class TestPositionReducingAndRealizedPnl:
    def test_a_partial_close_books_realized_pnl_on_only_the_closed_quantity(self) -> None:
        position = Position(symbol="NVDAUSDT", side=PositionSide.LONG)
        position.apply_fill(_lot(side="buy", quantity="10", price="100"))
        position.apply_fill(_lot(side="sell", quantity="4", price="120"))
        assert position.quantity == Decimal("6")
        assert position.realized_pnl == Decimal("80")  # 4 * (120 - 100)
        assert position.entry_price == Decimal("100")  # unchanged by a close

    def test_a_full_close_zeroes_entry_price(self) -> None:
        position = Position(symbol="NVDAUSDT", side=PositionSide.LONG)
        position.apply_fill(_lot(side="buy", quantity="10", price="100"))
        position.apply_fill(_lot(side="sell", quantity="10", price="90"))
        assert position.quantity == Decimal("0")
        assert position.entry_price == Decimal("0")
        assert position.realized_pnl == Decimal("-100")
        assert position.is_flat

    def test_short_side_pnl_sign_is_inverted(self) -> None:
        position = Position(symbol="NVDAUSDT", side=PositionSide.SHORT)
        position.apply_fill(_lot(side="sell", quantity="10", price="100"))
        position.apply_fill(_lot(side="buy", quantity="10", price="90"))  # covers, price fell
        assert position.realized_pnl == Decimal("100")

    def test_a_closing_fill_larger_than_the_position_flips_it(self) -> None:
        """The Nautilus reversal case (position.rs:545-551), reproduced for correctness even
        though hedge-mode (symbol, side) keying does not naturally reach it."""
        position = Position(symbol="NVDAUSDT", side=PositionSide.LONG)
        position.apply_fill(_lot(side="buy", quantity="10", price="100"))
        position.apply_fill(_lot(side="sell", quantity="15", price="110"))
        assert position.side is PositionSide.SHORT
        assert position.quantity == Decimal("5")
        assert position.entry_price == Decimal("110")
        assert position.realized_pnl == Decimal("100")  # 10 * (110 - 100), only the closed part


class TestUnrealizedPnl:
    def test_flat_position_reports_zero_not_none(self) -> None:
        position = Position(symbol="NVDAUSDT", side=PositionSide.LONG)
        assert position.unrealized_pnl(Decimal("999")) == Decimal("0")

    def test_long_gains_when_mark_rises(self) -> None:
        position = Position(symbol="NVDAUSDT", side=PositionSide.LONG)
        position.apply_fill(_lot(side="buy", quantity="10", price="100"))
        assert position.unrealized_pnl(Decimal("110")) == Decimal("100")

    def test_short_gains_when_mark_falls(self) -> None:
        position = Position(symbol="NVDAUSDT", side=PositionSide.SHORT)
        position.apply_fill(_lot(side="sell", quantity="10", price="100"))
        assert position.unrealized_pnl(Decimal("90")) == Decimal("100")


class TestRealizedPnlSince:
    """The windowed-PnL replay built for eval.freqtrade_baseline.freqtrade_low_profit_pairs —
    realized_pnl is a lifetime running total, this is the same arithmetic filtered to a cutoff."""

    def test_full_history_cutoff_matches_lifetime_realized_pnl(self) -> None:
        t1 = T0
        t2 = datetime(2026, 9, 15, 13, 0, tzinfo=UTC)
        position = Position(symbol="NVDAUSDT", side=PositionSide.LONG)
        position.apply_fill(_lot(side="buy", quantity="10", price="100", ts=t1))
        position.apply_fill(_lot(side="sell", quantity="4", price="120", ts=t2))
        very_early = datetime(2000, 1, 1, tzinfo=UTC)
        assert position.realized_pnl_since(very_early) == position.realized_pnl

    def test_future_cutoff_after_all_fills_is_zero(self) -> None:
        position = Position(symbol="NVDAUSDT", side=PositionSide.LONG)
        position.apply_fill(_lot(side="buy", quantity="10", price="100"))
        position.apply_fill(_lot(side="sell", quantity="4", price="120"))
        far_future = datetime(2100, 1, 1, tzinfo=UTC)
        assert position.realized_pnl_since(far_future) == Decimal("0")

    def test_cutoff_between_two_closing_fills_counts_only_the_later_one(self) -> None:
        t1 = T0
        t2 = datetime(2026, 9, 15, 13, 0, tzinfo=UTC)
        t3 = datetime(2026, 9, 15, 14, 0, tzinfo=UTC)
        cutoff = datetime(2026, 9, 15, 13, 30, tzinfo=UTC)
        position = Position(symbol="NVDAUSDT", side=PositionSide.LONG)
        position.apply_fill(_lot(side="buy", quantity="20", price="100", ts=t1))
        position.apply_fill(_lot(side="sell", quantity="5", price="120", ts=t2))  # before cutoff
        position.apply_fill(_lot(side="sell", quantity="5", price="130", ts=t3))  # after cutoff
        assert position.realized_pnl_since(cutoff) == Decimal("150")  # 5 * (130 - 100)
        assert position.realized_pnl == Decimal("250")  # the unwindowed total, for contrast

    def test_cutoff_exactly_on_a_fill_timestamp_is_inclusive(self) -> None:
        t1 = T0
        t2 = datetime(2026, 9, 15, 13, 0, tzinfo=UTC)
        position = Position(symbol="NVDAUSDT", side=PositionSide.LONG)
        position.apply_fill(_lot(side="buy", quantity="10", price="100", ts=t1))
        position.apply_fill(_lot(side="sell", quantity="10", price="110", ts=t2))
        assert position.realized_pnl_since(t2) == Decimal("100")

    def test_arbitrary_replay_seed_self_corrects_for_a_short_position(self) -> None:
        """The replay always seeds a temporary LONG position; proves it still produces the
        correct SHORT-side PnL, via apply_fill's own reversal-from-flat branch."""
        t1 = T0
        t2 = datetime(2026, 9, 15, 13, 0, tzinfo=UTC)
        position = Position(symbol="NVDAUSDT", side=PositionSide.SHORT)
        position.apply_fill(_lot(side="sell", quantity="10", price="100", ts=t1))
        position.apply_fill(_lot(side="buy", quantity="10", price="90", ts=t2))  # covers; +100
        assert position.realized_pnl_since(T0) == Decimal("100")
        assert position.realized_pnl_since(t2) == Decimal("100")

    def test_a_reversal_fill_is_handled_within_the_window(self) -> None:
        """Mirrors test_a_closing_fill_larger_than_the_position_flips_it, windowed."""
        t1 = T0
        t2 = datetime(2026, 9, 15, 13, 0, tzinfo=UTC)
        position = Position(symbol="NVDAUSDT", side=PositionSide.LONG)
        position.apply_fill(_lot(side="buy", quantity="10", price="100", ts=t1))
        position.apply_fill(_lot(side="sell", quantity="15", price="110", ts=t2))
        assert position.realized_pnl_since(t2) == Decimal("100")  # 10 * (110 - 100)

    def test_flat_position_with_no_lots_is_zero(self) -> None:
        position = Position(symbol="NVDAUSDT", side=PositionSide.LONG)
        assert position.realized_pnl_since(T0) == Decimal("0")


class TestOpenOrderReservation:
    def test_reserves_notional_at_the_order_price(self) -> None:
        r = order_reservation(
            client_order_id="c1", symbol="NVDAUSDT", side="buy",
            remaining_quantity=Decimal("10"), price=Decimal("100"),
        )
        assert r.reserved == Decimal("1000")

    def test_zero_remaining_reserves_nothing(self) -> None:
        r = order_reservation(
            client_order_id="c1", symbol="NVDAUSDT", side="buy",
            remaining_quantity=Decimal("0"), price=Decimal("100"),
        )
        assert r.reserved == Decimal("0")


class TestVenueMarginSnapshotParsing:
    """Field names verified live against mcp__bitget-agentic__account_overview, 2026-09-15."""

    LIVE_SHAPED_PAYLOAD: ClassVar[dict[str, str]] = {
        "accountEquity": "0", "usdtEquity": "0", "unrealisedPnl": "0",
        "imr": "0", "mmr": "0", "mgnRatio": "0", "positionMgnRatio": "0",
        "positionValue": "0", "leverage": "0",
    }

    def test_parses_every_field_from_the_live_shaped_payload(self) -> None:
        snap = parse_venue_margin_snapshot(self.LIVE_SHAPED_PAYLOAD, fetched_at=T0)
        assert isinstance(snap, VenueMarginSnapshot)
        assert snap.account_equity == Decimal("0")
        assert snap.mgn_ratio == Decimal("0")
        assert snap.fetched_at == T0

    def test_a_missing_field_raises_rather_than_defaulting(self) -> None:
        incomplete = {k: v for k, v in self.LIVE_SHAPED_PAYLOAD.items() if k != "mmr"}
        with pytest.raises(PositionError, match="mmr"):
            parse_venue_margin_snapshot(incomplete, fetched_at=T0)


class TestReconciliation:
    def test_exact_match(self) -> None:
        diff = reconcile_position(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            local_quantity=Decimal("10"), venue_quantity=Decimal("10"), checked_at=T0,
        )
        assert diff.status is ReconciliationStatus.MATCHED

    def test_small_drift_within_tolerance(self) -> None:
        diff = reconcile_position(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            local_quantity=Decimal("10000"), venue_quantity=Decimal("10000.5"), checked_at=T0,
        )
        assert diff.status is ReconciliationStatus.WITHIN_TOLERANCE

    def test_large_drift_is_manual_review(self) -> None:
        diff = reconcile_position(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            local_quantity=Decimal("10"), venue_quantity=Decimal("15"), checked_at=T0,
        )
        assert diff.status is ReconciliationStatus.MANUAL_REVIEW

    def test_local_zero_but_venue_holds_a_position_is_manual_review_not_tolerated(self) -> None:
        """Pins the bug caught and fixed before this module shipped: the tolerance branch's
        original `local_quantity == 0 or ...` guard made an untracked venue position read as
        'within tolerance' — the exact silent-miss this module exists to prevent."""
        diff = reconcile_position(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            local_quantity=Decimal("0"), venue_quantity=Decimal("50"), checked_at=T0,
        )
        assert diff.status is ReconciliationStatus.MANUAL_REVIEW


class TestBookKeyingIsHedgeModeAware:
    """The fact verified live (account_overview: holdMode=hedge_mode) that neither a generic
    Nautilus-style model nor a naive 'positions by symbol' design would surface."""

    def test_a_long_and_short_on_the_same_symbol_are_independent(self) -> None:
        book = Book()
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            lot=_lot(side="buy", quantity="10", price="100"),
        )
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.SHORT,
            lot=_lot(side="sell", quantity="5", price="100"),
        )
        assert book.positions[("NVDAUSDT", PositionSide.LONG)].quantity == Decimal("10")
        assert book.positions[("NVDAUSDT", PositionSide.SHORT)].quantity == Decimal("5")
        assert book.total_open_positions() == 2


class TestBookReservations:
    def test_reserve_and_release_round_trip(self) -> None:
        book = Book()
        book.reserve(OpenOrderReservation(
            client_order_id="c1", symbol="NVDAUSDT", side="buy", reserved=Decimal("500"),
        ))
        assert book.total_reserved() == Decimal("500")
        book.release("c1")
        assert book.total_reserved() == Decimal("0")

    def test_releasing_an_unknown_id_does_not_raise(self) -> None:
        Book().release("never-existed")


class TestBookUnrealizedPnlAggregation:
    def test_sums_only_positions_with_a_mark(self) -> None:
        book = Book()
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            lot=_lot(side="buy", quantity="10", price="100"),
        )
        book.apply_fill(
            symbol="MSFTUSDT", side=PositionSide.LONG,
            lot=_lot(side="buy", quantity="1", price="400"),
        )
        # Only NVDAUSDT gets a mark; MSFTUSDT is silently excluded from the sum, not zeroed.
        total = book.total_unrealized_pnl({"NVDAUSDT": Decimal("110")})
        assert total == Decimal("100")


class TestTotalGrossNotional:
    def test_sums_across_symbols(self) -> None:
        book = Book()
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            lot=_lot(side="buy", quantity="10", price="100"),
        )
        book.apply_fill(
            symbol="MSFTUSDT", side=PositionSide.LONG,
            lot=_lot(side="buy", quantity="2", price="400"),
        )
        assert book.total_gross_notional() == Decimal("1800")  # 1000 + 800

    def test_a_hedge_mode_long_and_short_do_not_cancel(self) -> None:
        book = Book()
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            lot=_lot(side="buy", quantity="10", price="100"),
        )
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.SHORT,
            lot=_lot(side="sell", quantity="10", price="100"),
        )
        assert book.total_gross_notional() == Decimal("2000")  # both legs count

    def test_empty_book_is_zero(self) -> None:
        assert Book().total_gross_notional() == Decimal("0")


class TestTotalSignedNotional:
    def test_a_single_long_is_positive(self) -> None:
        book = Book()
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            lot=_lot(side="buy", quantity="10", price="100"),
        )
        assert book.total_signed_notional() == Decimal("1000")

    def test_a_single_short_is_negative(self) -> None:
        book = Book()
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.SHORT,
            lot=_lot(side="sell", quantity="10", price="100"),
        )
        assert book.total_signed_notional() == Decimal("-1000")

    def test_a_hedge_mode_long_and_short_in_one_symbol_nets(self) -> None:
        """Unlike total_gross_notional, this DOES net a hedge against itself — it answers
        'how directionally skewed is the desk', not 'how much margin risk is carried'."""
        book = Book()
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            lot=_lot(side="buy", quantity="10", price="100"),
        )
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.SHORT,
            lot=_lot(side="sell", quantity="6", price="100"),
        )
        assert book.total_signed_notional() == Decimal("400")

    def test_long_and_short_across_symbols_offset(self) -> None:
        book = Book()
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            lot=_lot(side="buy", quantity="10", price="100"),
        )
        book.apply_fill(
            symbol="MSFTUSDT", side=PositionSide.SHORT,
            lot=_lot(side="sell", quantity="2", price="500"),
        )
        assert book.total_signed_notional() == Decimal("0")  # 1000 - 1000

    def test_empty_book_is_zero(self) -> None:
        assert Book().total_signed_notional() == Decimal("0")


class TestWeights:
    def test_a_single_long_position_weights_positive(self) -> None:
        book = Book()
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            lot=_lot(side="buy", quantity="10", price="100"),
        )
        weights = book.weights(Decimal("10000"))
        assert weights == {"NVDAUSDT": pytest.approx(0.1)}

    def test_a_short_position_weights_negative(self) -> None:
        book = Book()
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.SHORT,
            lot=_lot(side="sell", quantity="10", price="100"),
        )
        weights = book.weights(Decimal("10000"))
        assert weights == {"NVDAUSDT": pytest.approx(-0.1)}

    def test_equal_long_and_short_in_hedge_mode_nets_to_zero(self) -> None:
        """The deliberate netting `weights()` documents — correct for beta, not for margin."""
        book = Book()
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            lot=_lot(side="buy", quantity="10", price="100"),
        )
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.SHORT,
            lot=_lot(side="sell", quantity="10", price="100"),
        )
        weights = book.weights(Decimal("10000"))
        assert weights == {"NVDAUSDT": pytest.approx(0.0)}

    def test_nonpositive_equity_returns_zeroed_weights_not_a_raise(self) -> None:
        book = Book()
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            lot=_lot(side="buy", quantity="10", price="100"),
        )
        assert book.weights(Decimal("0")) == {"NVDAUSDT": 0.0}

    def test_gross_weights_does_not_net_a_hedge_against_itself(self) -> None:
        book = Book()
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            lot=_lot(side="buy", quantity="10", price="100"),
        )
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.SHORT,
            lot=_lot(side="sell", quantity="10", price="100"),
        )
        gross = book.gross_weights(Decimal("10000"))
        assert gross == {
            "NVDAUSDT:long": pytest.approx(0.1),
            "NVDAUSDT:short": pytest.approx(0.1),
        }

    def test_gross_weights_nonpositive_equity_returns_empty(self) -> None:
        book = Book()
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            lot=_lot(side="buy", quantity="10", price="100"),
        )
        assert book.gross_weights(Decimal("0")) == {}


class TestOrderBookIsWireCompatibleWithBook:
    """`execution.orders.OrderBook` (the order lifecycle) and `desk.book.Book` (positions) were
    built independently and nothing in the codebase connects them yet — no glue code exists because
    none is needed: an `Order`'s own fields map directly onto `Lot`'s. This test proves that end to
    end rather than asserting it, since the two modules have never been run together before."""

    def test_a_filled_order_folds_directly_into_a_position(self) -> None:
        client_order_id = deterministic_client_order_id(
            market_state_hash="m1", approved_intent_hash="a1", side="BUY", quantity=Decimal("10"),
        )
        order = Order(
            client_order_id=client_order_id, symbol="NVDAUSDT", side="BUY",
            quantity=Decimal("10"), approved_intent_hash="a1",
        )
        order.transition(OrderState.SUBMITTED, at=T0, reason="submitted to venue")
        order.apply_fill(Decimal("10"), at=T0)
        assert order.state is OrderState.FILLED

        lot = Lot(
            order_id=order.client_order_id, approved_intent_hash=order.approved_intent_hash,
            side=order.side, quantity=order.filled_quantity, price=Decimal("450"),
            commission=Decimal("0.45"), ts_filled=T0,
        )
        book = Book()
        position = book.apply_fill(symbol=order.symbol, side=PositionSide.LONG, lot=lot)
        assert position.quantity == Decimal("10")
        assert position.entry_price == Decimal("450")
        assert position.lots[0].order_id == client_order_id


class TestHedgeCluster:
    def test_transitive_closure_across_two_links(self) -> None:
        book = Book()
        a = ("NVDAUSDT", PositionSide.LONG)
        b = ("TQQQUSDT", PositionSide.SHORT)
        c = ("QQQUSDT", PositionSide.SHORT)
        book.link_hedge(HedgeLink(source=a, target=b, relationship="HEDGE",
                                   hedge_ratio=None, ts_created=T0))
        book.link_hedge(HedgeLink(source=b, target=c, relationship="HEDGE",
                                   hedge_ratio=None, ts_created=T0))
        assert book.hedge_cluster(*a) == {a, b, c}

    def test_an_unlinked_position_clusters_with_only_itself(self) -> None:
        book = Book()
        solo = ("AAPLUSDT", PositionSide.LONG)
        assert book.hedge_cluster(*solo) == {solo}
