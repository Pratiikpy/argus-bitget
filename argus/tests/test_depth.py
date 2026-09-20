"""Order-book tests — the arithmetic that decides what a trade really costs.

The failure this file guards against is not a crash. It is a sweep that walks the wrong side, or
prices a partial fill as though it were complete, or reports the quoted touch for a size that would
consume ten levels — each of which returns a plausible number and makes an expensive instrument look
cheap. Every one of those is asserted against a hand-built book whose answer can be worked out on
paper.

The live tests at the end run against the real venue and skip when it is unreachable, because a
network failure is not a defect in this module.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from argus.market.depth import (
    MAX_LEVELS,
    DepthError,
    Level,
    OrderBook,
    fetch_orderbook,
    impact_check,
)

NOW = datetime(2026, 9, 14, tzinfo=UTC)


def _book(
    bids: list[tuple[str, str]] | None = None, asks: list[tuple[str, str]] | None = None,
) -> OrderBook:
    """A book whose levels are round numbers, so every expected answer is checkable by hand."""
    return OrderBook(
        symbol="TESTUSDT", fetched_at=NOW,
        bids=tuple(
            Level(price=Decimal(p), quantity=Decimal(q))
            for p, q in (bids or [("99", "10"), ("98", "10"), ("97", "100")])
        ),
        asks=tuple(
            Level(price=Decimal(p), quantity=Decimal(q))
            for p, q in (asks or [("101", "10"), ("102", "10"), ("103", "100")])
        ),
    )


class TestTheQuoteIsNotTheCost:
    def test_a_size_inside_the_touch_pays_the_half_spread(self) -> None:
        """$101 buys one unit at 101 against a mid of 100: exactly 100bps, which is half of the
        200bps quoted spread. The quote is a round trip; one side pays half of it."""
        sweep = _book().sweep(Decimal("101"), direction="BUY")
        assert sweep.levels_consumed == 1
        assert sweep.slippage_bps == pytest.approx(Decimal("100"), abs=Decimal("0.01"))
        assert sweep.complete

    def test_a_size_that_walks_two_levels_pays_more_than_the_quote(self) -> None:
        book = _book()
        small = book.sweep(Decimal("101"), direction="BUY").slippage_bps
        large = book.sweep(Decimal("2000"), direction="BUY").slippage_bps
        # The smallest possible taker trade pays exactly half the quoted spread; anything that
        # walks a second level pays strictly more than that floor.
        assert small == pytest.approx(book.spread_bps / 2, abs=Decimal("0.01"))
        assert large > small
        assert book.sweep(Decimal("2000"), direction="BUY").levels_consumed == 2

    def test_the_quoted_spread_is_the_touch(self) -> None:
        assert _book().spread_bps == pytest.approx(Decimal("200"), abs=Decimal("0.01"))
        assert _book().mid == Decimal("100")

    def test_selling_walks_the_bids_not_the_asks(self) -> None:
        """Getting the side backwards prices the trade you did not do, and returns a number."""
        book = _book(asks=[("101", "1"), ("200", "1000")])
        buy = book.sweep(Decimal("5000"), direction="BUY")
        sell = book.sweep(Decimal("5000"), direction="SELL")
        assert buy.slippage_bps > sell.slippage_bps
        assert book.side("SELL")[0].price == Decimal("99")

    def test_both_directions_report_a_positive_cost(self) -> None:
        book = _book()
        assert book.sweep(Decimal("500"), direction="BUY").slippage_bps > 0
        assert book.sweep(Decimal("500"), direction="SELL").slippage_bps > 0


class TestPartialFillsAreNotHiddenCheapFills:
    def test_a_size_the_book_cannot_absorb_is_marked_incomplete(self) -> None:
        book = _book(asks=[("101", "1"), ("102", "1")])
        sweep = book.sweep(Decimal("1000000"), direction="BUY")
        assert not sweep.complete
        assert sweep.filled_notional < Decimal("1000000")

    def test_the_measured_helper_refuses_an_incomplete_sweep(self) -> None:
        """A partial sweep's slippage describes the part that filled, so returning it would price
        the cheap half of a trade that could not be done."""
        from argus.market import depth as module

        book = _book(asks=[("101", "1")])
        original = module.fetch_orderbook
        module.fetch_orderbook = lambda *a, **k: book  # type: ignore[assignment]
        try:
            assert module.measured_spread_bps("TESTUSDT", Decimal("1000000")) is None
            assert module.measured_spread_bps("TESTUSDT", Decimal("50")) is not None
        finally:
            module.fetch_orderbook = original  # type: ignore[assignment]

    def test_an_unreachable_book_returns_none_rather_than_a_guess(self) -> None:
        from argus.market import depth as module
        from argus.market.bitget import BitgetError

        original = module.fetch_orderbook

        def boom(*_args: object, **_kwargs: object) -> OrderBook:
            raise BitgetError("transport failure")

        module.fetch_orderbook = boom  # type: ignore[assignment]
        try:
            assert module.measured_spread_bps("TESTUSDT", Decimal("100")) is None
        finally:
            module.fetch_orderbook = original  # type: ignore[assignment]


class TestTheInverseQuestion:
    def test_executable_within_stops_at_the_budget(self) -> None:
        """A symmetric book, so the mid is exactly 100 and the arithmetic is checkable: the first
        ask sits 10bps above it and the second a thousand bps above, so a 50bps budget buys all of
        level one and none of level two."""
        book = _book(bids=[("99.9", "10")], asks=[("100.1", "10"), ("110", "1000")])
        assert book.mid == Decimal("100")
        allowed = book.executable_within(Decimal("50"), direction="BUY")
        assert allowed == pytest.approx(Decimal("1001"), abs=Decimal("1"))

    def test_a_bigger_budget_carries_more_size(self) -> None:
        book = _book()
        assert book.executable_within(Decimal("400")) >= book.executable_within(Decimal("120"))

    def test_a_zero_budget_is_refused(self) -> None:
        with pytest.raises(DepthError, match="budget must be positive"):
            _book().executable_within(Decimal("0"))


class TestItRefusesBadBooks:
    def test_a_crossed_book_is_a_feed_error_not_an_arbitrage(self) -> None:
        with pytest.raises(DepthError, match="crossed"):
            OrderBook(
                symbol="X", fetched_at=NOW,
                bids=(Level(price=Decimal("101"), quantity=Decimal("1")),),
                asks=(Level(price=Decimal("100"), quantity=Decimal("1")),),
            )

    def test_a_one_sided_book_cannot_be_priced(self) -> None:
        with pytest.raises(DepthError, match="one-sided"):
            OrderBook(
                symbol="X", fetched_at=NOW,
                bids=(Level(price=Decimal("99"), quantity=Decimal("1")),), asks=(),
            )

    def test_a_zero_notional_sweep_is_refused(self) -> None:
        with pytest.raises(DepthError, match="positive notional"):
            _book().sweep(Decimal("0"))

    def test_an_unknown_side_is_refused(self) -> None:
        with pytest.raises(DepthError, match="BUY or SELL"):
            _book().side("HOLD")


class TestImbalance:
    def test_a_balanced_book_is_a_half(self) -> None:
        book = _book(bids=[("99", "10")], asks=[("101", "10")])
        assert book.imbalance() == pytest.approx(Decimal("0.5"), abs=Decimal("0.01"))

    def test_a_bid_heavy_book_reads_above_a_half(self) -> None:
        book = _book(bids=[("99", "100")], asks=[("101", "1")])
        assert book.imbalance() > Decimal("0.9")


class TestAgainstTheModelledImpact:
    def test_it_reports_the_measured_cost_beside_the_modelled_one(self) -> None:
        rows = impact_check(
            _book(), [Decimal("500"), Decimal("2000")], adv_notional=Decimal("1000000"),
        )
        assert len(rows) == 2
        assert all("measured_slippage_bps" in row for row in rows)
        assert all("modelled_impact_bps" in row for row in rows)

    def test_without_an_adv_only_the_measured_column_appears(self) -> None:
        """A participation rate needs a denominator. Inventing one to fill the column would be the
        exact failure this comparison exists to expose."""
        rows = impact_check(_book(), [Decimal("500")])
        assert "modelled_impact_bps" not in rows[0]


class TestTheLiveVenue:
    def test_a_real_book_prices_a_real_size(self) -> None:
        from argus.market.bitget import BitgetError

        try:
            book = fetch_orderbook("NVDAUSDT", limit=50)
        except (BitgetError, DepthError) as exc:
            pytest.skip(f"venue unreachable: {exc}")
        assert book.bids and book.asks
        assert book.spread_bps > 0
        sweep = book.sweep(Decimal("25000"), direction="BUY")
        # The measurement that motivated this module: a real size costs more than the quote.
        assert sweep.slippage_bps >= book.spread_bps / 2
        assert sweep.levels_consumed >= 1

    def test_the_level_cap_is_the_venues_own(self) -> None:
        from argus.market.bitget import BitgetError

        try:
            book = fetch_orderbook("NVDAUSDT", limit=MAX_LEVELS + 500)
        except (BitgetError, DepthError) as exc:
            pytest.skip(f"venue unreachable: {exc}")
        assert len(book.asks) <= MAX_LEVELS

    def test_the_runner_charges_the_measured_spread(self) -> None:
        """The wiring, not the module: the paper ledger used to be handed the quoted touch."""
        from argus.market.bitget import BitgetError
        from argus.paper.runner import _executable_spread_bps

        try:
            fetch_orderbook("NVDAUSDT", limit=5)
        except (BitgetError, DepthError) as exc:
            pytest.skip(f"venue unreachable: {exc}")
        small = _executable_spread_bps(
            "NVDAUSDT", Decimal("1000"), "BUY", fallback=Decimal("0.6"),
        )
        large = _executable_spread_bps(
            "NVDAUSDT", Decimal("100000"), "BUY", fallback=Decimal("0.6"),
        )
        assert large >= small > 0

    def test_a_zero_size_falls_back_to_the_quote(self) -> None:
        from argus.paper.runner import _executable_spread_bps

        assert _executable_spread_bps(
            "NVDAUSDT", Decimal("0"), "BUY", fallback=Decimal("0.6"),
        ) == Decimal("0.6")
