"""Time in force, price protection, stop orders and latency in the simulated market
(research/harvest/45-nautilus-hftbacktest.md)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from argus.sim.agents import ScriptedAgent
from argus.sim.book import BookError, Order, OrderBook, Side, TimeInForce
from argus.sim.market import SessionConfig, SimulatedSession, StopOrder, build_session, measure_exit

D = Decimal


def _book() -> OrderBook:
    """Asks of 10 at 100.00, 100.01 and 100.02; bids of 10 at 99.99 and 99.98."""
    book = OrderBook("T", tick_size=D("0.01"))
    for i, price in enumerate(("100.00", "100.01", "100.02")):
        book.submit(Order(agent_id=f"maker{i}", side=Side.SELL, price=D(price), quantity=D(10)))
    for i, price in enumerate(("99.99", "99.98")):
        book.submit(Order(agent_id=f"bid{i}", side=Side.BUY, price=D(price), quantity=D(10)))
    return book


def _taker(quantity: str, price: str = "100.05", tif: TimeInForce = TimeInForce.GTC) -> Order:
    return Order(agent_id="taker", side=Side.BUY, price=D(price), quantity=D(quantity),
                 time_in_force=tif)


class TestTimeInForce:
    def test_gtc_rests_what_does_not_fill(self) -> None:
        book = _book()
        order = _taker("40")
        book.submit(order)
        assert order.filled == D(30) and order.cancelled == ""
        assert book.best_bid == D("100.05")

    def test_ioc_cancels_the_remainder(self) -> None:
        book = _book()
        order = _taker("40", tif=TimeInForce.IOC)
        book.submit(order)
        assert order.filled == D(30) and "immediate-or-cancel" in order.cancelled
        assert book.best_bid == D("99.99")

    def test_fok_fills_whole_or_not_at_all(self) -> None:
        book = _book()
        too_big = _taker("31", tif=TimeInForce.FOK)
        assert book.submit(too_big) == [] and too_big.filled == 0
        assert "30 available of 31" in too_big.cancelled
        fits = _taker("30", tif=TimeInForce.FOK)
        book.submit(fits)
        assert fits.filled == D(30)

    def test_post_only_that_would_take_is_refused_whole(self) -> None:
        book = _book()
        crossing = _taker("5", price="100.00", tif=TimeInForce.GTX)
        assert book.submit(crossing) == [] and "post-only" in crossing.cancelled
        resting = _taker("5", price="99.99", tif=TimeInForce.GTX)
        book.submit(resting)
        assert resting.cancelled == "" and book.best_bid == D("99.99")

    def test_fok_does_not_count_the_takers_own_orders_as_liquidity(self) -> None:
        book = _book()
        book.submit(Order(agent_id="taker", side=Side.SELL, price=D("100.03"), quantity=D(50)))
        order = _taker("31", price="100.03", tif=TimeInForce.FOK)
        assert book.submit(order) == []


class TestPriceProtection:
    def test_a_walk_stops_at_the_band_and_the_rest_is_cancelled(self) -> None:
        book = _book()
        order = _taker("30")
        book.submit(order, price_protection_ticks=1)
        assert order.filled == D(20) and "price protection" in order.cancelled
        assert book.best_ask == D("100.02") and book.best_bid == D("99.99")

    def test_an_order_inside_the_band_is_untouched(self) -> None:
        book = _book()
        order = _taker("15")
        book.submit(order, price_protection_ticks=5)
        assert order.filled == D(15) and order.cancelled == ""

    def test_a_negative_band_is_refused(self) -> None:
        with pytest.raises(BookError):
            _book().submit(_taker("1"), price_protection_ticks=-1)


def _session(**cfg: object) -> tuple[SimulatedSession, ScriptedAgent]:
    session = build_session(SessionConfig(**cfg))  # type: ignore[arg-type]
    scripted = ScriptedAgent("scripted")
    session.agents.append(scripted)
    return session, scripted


class TestStops:
    @staticmethod
    def _quiet() -> SimulatedSession:
        """The fixed book above, no agents, so every trade is the test's own."""
        from random import Random

        return SimulatedSession(book=_book(), agents=[], config=SessionConfig(), rng=Random(0))

    def test_a_sell_stop_fires_when_a_trade_reaches_its_trigger(self) -> None:
        session = self._quiet()
        stop = StopOrder(trigger=D("99.99"), order=Order(
            agent_id="stopper", side=Side.SELL, price=D("0.01"), quantity=D(5),
            time_in_force=TimeInForce.IOC))
        session.submit_stop(stop)
        assert session.stops == [stop]  # nothing has traded yet
        session._submit(Order(agent_id="seller", side=Side.SELL, price=D("99.99"),
                              quantity=D(1)))
        assert session.stops == [] and stop.order.filled == D(5)

    def test_a_buy_stop_waits_until_its_trigger_trades(self) -> None:
        session = self._quiet()
        stop = StopOrder(trigger=D("100.02"), order=_taker("1", price="100.05"))
        session.submit_stop(stop)
        session._submit(_taker("5", price="100.00"))  # trades at 100.00: below the trigger
        assert stop in session.stops and stop.order.filled == 0
        session._submit(_taker("30", price="100.02"))  # walks to 100.02: the trigger
        assert session.stops == []

    def test_the_trigger_rule_by_side(self) -> None:
        buy = StopOrder(trigger=D(10), order=_taker("1"))
        sell = StopOrder(trigger=D(10), order=Order(agent_id="s", side=Side.SELL,
                                                    price=D(1), quantity=D(1)))
        assert buy.triggered_by(D(10)) and not buy.triggered_by(D("9.99"))
        assert sell.triggered_by(D(10)) and not sell.triggered_by(D("10.01"))


class TestLatency:
    def test_an_order_reaches_the_book_after_the_entry_latency(self) -> None:
        session, scripted = _session(entry_latency_steps=2)
        bid = session.book.best_bid
        assert bid is not None
        order = Order(agent_id="scripted", side=Side.BUY, price=bid - D("5"), quantity=D(1))
        scripted.enqueue(order)
        session.step(1)
        assert order.order_id == 0 and any(o is order for _, o in session.in_flight)
        session.step(2)
        assert order.order_id != 0 and not any(o is order for _, o in session.in_flight)

    def test_a_fill_is_heard_after_the_response_latency(self) -> None:
        session, scripted = _session(response_latency_steps=3)
        ask = session.book.best_ask
        assert ask is not None
        scripted.enqueue(Order(agent_id="scripted", side=Side.BUY, price=ask + D(1),
                               quantity=D(1), time_in_force=TimeInForce.IOC))
        session.step(1)
        assert scripted.position == 0
        assert any(e.agent_id == "scripted" for _, e in session.unheard)
        session.step(3)
        assert scripted.position == D(1)

    def test_a_negative_latency_is_refused(self) -> None:
        from argus.sim.market import MarketError

        with pytest.raises(MarketError):
            SessionConfig(entry_latency_steps=-1)


def test_an_exit_leaves_nothing_resting_behind() -> None:
    session, _ = _session()
    depth = session.book.total_depth(Side.BUY, levels=50)
    result = measure_exit(session, quantity=depth + D(1000))
    assert not result.fully_exited
    assert session.book.best_ask is None or session.book.best_ask > D("0.01")
