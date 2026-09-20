"""Matching engine and agent-zoo tests.

The engine is checked against ABIDES's own documented example rather than against its own output:
an inbound 100 against 30/50/20 across three levels must produce exactly those three fills, at
those three prices. A test that only asserted self-consistency would pass with price priority
inverted.

The property that matters most here is the one ABIDES does not have: **every execution carries a
maker/taker flag**, because a simulator that cannot say which side paid the fee reintroduces the
cost-blindness the rest of this codebase makes unconstructible.
"""

from __future__ import annotations

import contextlib
from decimal import Decimal
from random import Random

import pytest

from argus.sim.agents import (
    AdaptiveMarketMaker,
    LadderConfig,
    NoiseAgent,
    ScriptedAgent,
    ValueAgent,
)
from argus.sim.book import (
    BookError,
    Liquidity,
    Order,
    OrderBook,
    Side,
)

D = Decimal


def _book(tick: str = "0.01") -> OrderBook:
    return OrderBook("rNVDA", tick_size=D(tick))


def _ask(b: OrderBook, price: str, qty: str, who: str = "mm") -> list:
    return b.submit(Order(agent_id=who, side=Side.SELL, price=D(price), quantity=D(qty)))


def _bid(b: OrderBook, price: str, qty: str, who: str = "mm") -> list:
    return b.submit(Order(agent_id=who, side=Side.BUY, price=D(price), quantity=D(qty)))


class TestMatchingWalksTheBook:
    """``order_book.py:109-134`` — the documented example, reproduced exactly."""

    def test_abides_own_example(self) -> None:
        b = _book()
        _ask(b, "100.00", "30")
        _ask(b, "100.01", "50")
        _ask(b, "100.02", "20")

        ex = b.submit(Order(agent_id="taker", side=Side.BUY, price=D("100.02"), quantity=D("100")))
        ours = [(e.price, e.quantity) for e in ex if e.agent_id == "taker"]
        assert ours == [(D("100.00"), D("30")), (D("100.01"), D("50")), (D("100.02"), D("20"))]
        assert b.traded_volume == D("100")
        assert b.last_trade == D("100.02")
        assert b.levels(Side.SELL) == []

    def test_each_match_produces_two_executions(self) -> None:
        """One per counterparty. ABIDES notifies over a bus; with no bus, losing the resting side
        would mean the maker never learns it was filled."""
        b = _book()
        _ask(b, "100.00", "30")
        ex = b.submit(Order(agent_id="taker", side=Side.BUY, price=D("100.00"), quantity=D("30")))
        assert len(ex) == 2
        assert {e.agent_id for e in ex} == {"taker", "mm"}
        assert {e.sequence for e in ex} == {ex[0].sequence}

    def test_the_aggressor_is_taker_and_the_resting_order_is_maker(self) -> None:
        """The field ABIDES does not have, and the only basis for charging a fee correctly."""
        b = _book()
        _ask(b, "100.00", "30")
        ex = b.submit(Order(agent_id="taker", side=Side.BUY, price=D("100.00"), quantity=D("10")))
        by_agent = {e.agent_id: e for e in ex}
        assert by_agent["taker"].liquidity is Liquidity.TAKER
        assert by_agent["mm"].liquidity is Liquidity.MAKER

    def test_the_resting_order_sets_the_price(self) -> None:
        """A buyer willing to pay 101 against an ask at 100 pays 100, not 101."""
        b = _book()
        _ask(b, "100.00", "30")
        ex = b.submit(Order(agent_id="taker", side=Side.BUY, price=D("101.00"), quantity=D("30")))
        assert all(e.price == D("100.00") for e in ex)

    def test_an_unmatched_remainder_rests(self) -> None:
        b = _book()
        _ask(b, "100.00", "30")
        b.submit(Order(agent_id="taker", side=Side.BUY, price=D("100.00"), quantity=D("50")))
        assert b.best_bid == D("100.00")
        assert b.depth_at(Side.BUY, D("100.00")) == D("20")

    def test_a_non_crossing_order_just_rests(self) -> None:
        b = _book()
        _ask(b, "100.05", "30")
        ex = _bid(b, "100.00", "30", who="buyer")
        assert ex == []
        assert b.spread == D("0.05")
        assert b.mid == D("100.025")

    def test_time_priority_within_a_level(self) -> None:
        b = _book()
        _ask(b, "100.00", "10", who="first")
        _ask(b, "100.00", "10", who="second")
        ex = b.submit(Order(agent_id="taker", side=Side.BUY, price=D("100.00"), quantity=D("10")))
        makers = [e.agent_id for e in ex if e.liquidity is Liquidity.MAKER]
        assert makers == ["first"]

    def test_sell_side_walks_downward(self) -> None:
        b = _book()
        _bid(b, "100.02", "20")
        _bid(b, "100.01", "50")
        _bid(b, "100.00", "30")
        ex = b.submit(Order(agent_id="t", side=Side.SELL, price=D("100.00"), quantity=D("100")))
        ours = [(e.price, e.quantity) for e in ex if e.agent_id == "t"]
        assert ours == [(D("100.02"), D("20")), (D("100.01"), D("50")), (D("100.00"), D("30"))]


class TestBookInvariants:
    def test_a_one_sided_book_has_no_mid(self) -> None:
        """None rather than falling back to the last trade — inventing a midpoint is how a
        simulated market maker quotes around a price that does not exist."""
        b = _book()
        _ask(b, "100.00", "10")
        assert b.mid is None
        assert b.spread is None
        assert b.best_bid is None

    def test_sub_tick_prices_are_refused(self) -> None:
        """ABIDES leaves tick size to the agents; we enforce it."""
        b = _book("0.01")
        with pytest.raises(BookError, match="tick_size"):
            _ask(b, "100.005", "10")

    def test_self_trading_is_refused_not_matched(self) -> None:
        """A wash trade inflates volume — the input the PoV market maker sizes against."""
        b = _book()
        _ask(b, "100.00", "10", who="same")
        with pytest.raises(BookError, match="wash trade"):
            b.submit(Order(agent_id="same", side=Side.BUY, price=D("100.00"), quantity=D("5")))

    def test_zero_and_negative_quantities_are_refused(self) -> None:
        b = _book()
        for q in ("0", "-5"):
            with pytest.raises(BookError, match="quantity"):
                _ask(b, "100.00", q)

    def test_non_positive_prices_are_refused(self) -> None:
        b = _book()
        with pytest.raises(BookError, match="price"):
            b.submit(Order(agent_id="x", side=Side.SELL, price=D("0"), quantity=D("1")))

    def test_zero_tick_size_is_refused(self) -> None:
        with pytest.raises(BookError, match="tick_size"):
            OrderBook("rNVDA", tick_size=D("0"))

    def test_bids_descend_and_asks_ascend(self) -> None:
        b = _book()
        for p in ("99.98", "100.00", "99.99"):
            _bid(b, p, "10")
        for p in ("100.03", "100.01", "100.02"):
            _ask(b, p, "10")
        assert [p for p, _ in b.levels(Side.BUY)] == [D("100.00"), D("99.99"), D("99.98")]
        assert [p for p, _ in b.levels(Side.SELL)] == [D("100.01"), D("100.02"), D("100.03")]

    def test_cancel_removes_and_reports(self) -> None:
        b = _book()
        order = Order(agent_id="mm", side=Side.SELL, price=D("100.00"), quantity=D("10"))
        b.submit(order)
        assert b.cancel(order.order_id) is True
        assert b.cancel(order.order_id) is False
        assert b.best_ask is None

    def test_cancel_all_is_how_a_ladder_requotes(self) -> None:
        b = _book()
        for p in ("100.01", "100.02", "100.03"):
            _ask(b, p, "10", who="mm")
        _ask(b, "100.04", "10", who="other")
        assert b.cancel_all("mm") == 3
        assert [p for p, _ in b.levels(Side.SELL)] == [D("100.04")]

    def test_total_depth_sums_the_top_levels(self) -> None:
        b = _book()
        for p in ("100.01", "100.02", "100.03"):
            _ask(b, p, "10")
        assert b.total_depth(Side.SELL, levels=2) == D("20")
        assert b.total_depth(Side.SELL, levels=5) == D("30")


class TestAgentZoo:
    def test_a_noise_agent_does_nothing_on_a_one_sided_book(self) -> None:
        """No mid, no view, no order — rather than quoting around an invented price."""
        b = _book()
        _ask(b, "100.00", "10")
        assert NoiseAgent("n").wakeup(b, Random(1)) == []

    def test_a_noise_agent_trades_both_ways_over_many_wakeups(self) -> None:
        b = _book()
        _bid(b, "99.99", "500")
        _ask(b, "100.01", "500")
        agent = NoiseAgent("n")
        rng = Random(7)
        sides = {o.side for _ in range(60) for o in agent.wakeup(b, rng)}
        assert sides == {Side.BUY, Side.SELL}, "noise with a directional bias is not noise"

    def test_the_zoo_is_deterministic_under_a_seed(self) -> None:
        """A stress test whose result changes between runs cannot be used to argue anything."""
        def run(seed: int) -> list[tuple[str, str]]:
            b = _book()
            _bid(b, "99.99", "500")
            _ask(b, "100.01", "500")
            agent = NoiseAgent("n")
            rng = Random(seed)
            return [(str(o.side), str(o.price)) for _ in range(20) for o in agent.wakeup(b, rng)]

        assert run(42) == run(42)
        assert run(42) != run(43)

    def test_a_value_agent_buys_below_its_fundamental(self) -> None:
        b = _book()
        _bid(b, "99.99", "100")
        _ask(b, "100.01", "100")
        # r_bar far above the mid, kappa high so the prior dominates the observation.
        agent = ValueAgent("v", r_bar=D("200"), kappa=D("0.9"), sigma_n=D("0"))
        orders = agent.wakeup(b, Random(3))
        assert orders and all(o.side is Side.BUY for o in orders)

    def test_a_value_agent_sells_above_its_fundamental(self) -> None:
        b = _book()
        _bid(b, "99.99", "100")
        _ask(b, "100.01", "100")
        agent = ValueAgent("v", r_bar=D("10"), kappa=D("0.9"), sigma_n=D("0"))
        orders = agent.wakeup(b, Random(3))
        assert orders and all(o.side is Side.SELL for o in orders)

    def test_the_fundamental_estimate_shrinks_toward_the_prior(self) -> None:
        """estimate = (1 - kappa)*observed + kappa*r_bar — value_agent.py."""
        agent = ValueAgent("v", r_bar=D("200"), kappa=D("0.25"), sigma_n=D("0"))
        assert agent.estimate(D("100"), Random(1)) == D("125")

    def test_kappa_outside_zero_to_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="kappa"):
            ValueAgent("v", r_bar=D("100"), kappa=D("1.5"))

    def test_the_market_maker_provides_two_sided_depth(self) -> None:
        """ABIDES calls this agent 'critically important for market depth'."""
        b = _book()
        _bid(b, "99.99", "10", who="seed")
        _ask(b, "100.01", "10", who="seed")
        mm = AdaptiveMarketMaker("mm", config=LadderConfig(num_ticks=5, min_order_size=D("20")))
        for order in mm.wakeup(b, Random(1)):
            b.submit(order)
        assert len(b.levels(Side.BUY, 10)) >= 5
        assert len(b.levels(Side.SELL, 10)) >= 5
        assert b.mid is not None

    def test_the_ladder_requotes_rather_than_stacking(self) -> None:
        b = _book()
        _bid(b, "99.99", "10", who="seed")
        _ask(b, "100.01", "10", who="seed")
        mm = AdaptiveMarketMaker("mm")
        for _ in range(3):
            for order in mm.wakeup(b, Random(1)):
                b.submit(order)
        levels = b.levels(Side.BUY, 20)
        assert len(levels) <= 6, "a ladder that does not cancel stacks duplicate depth every wakeup"

    def test_inventory_skew_is_off_by_default_and_works_when_on(self) -> None:
        """Matching ABIDES, where skew_beta defaults to 0."""
        assert LadderConfig().skew_beta == D("0")

        def centre(beta: Decimal, position: Decimal) -> Decimal:
            b = _book()
            _bid(b, "99.99", "10", who="seed")
            _ask(b, "100.01", "10", who="seed")
            mm = AdaptiveMarketMaker("mm", config=LadderConfig(skew_beta=beta))
            mm.position = position
            orders = mm.wakeup(b, Random(1))
            buys = [o.price for o in orders if o.side is Side.BUY]
            return max(buys)

        flat = centre(D("0"), D("0"))
        long_skewed = centre(D("5"), D("100"))
        assert long_skewed < flat, "a long market maker should shift its ladder down"

    def test_the_market_maker_does_nothing_on_an_empty_book(self) -> None:
        assert AdaptiveMarketMaker("mm").wakeup(_book(), Random(1)) == []

    def test_a_scripted_agent_submits_exactly_what_it_is_told(self) -> None:
        """How an ARGUS decision enters the simulation — no behaviour of its own."""
        b = _book()
        agent = ScriptedAgent("argus")
        order = Order(agent_id="argus", side=Side.BUY, price=D("100.00"), quantity=D("5"))
        agent.enqueue(order)
        assert agent.wakeup(b, Random(1)) == [order]
        assert agent.wakeup(b, Random(1)) == [], "the queue drains; it does not repeat"


class TestPositionAndFeeAccounting:
    def test_a_fill_moves_position_and_cash_in_opposite_directions(self) -> None:
        b = _book()
        _ask(b, "100.00", "10", who="mm")
        buyer = ScriptedAgent("buyer")
        ex = b.submit(Order(agent_id="buyer", side=Side.BUY, price=D("100.00"), quantity=D("10")))
        for e in ex:
            if e.agent_id == "buyer":
                buyer.on_fill(e)
        assert buyer.position == D("10")
        assert buyer.cash == D("1000000") - D("1000")

    def test_fees_are_not_applied_by_the_agent(self) -> None:
        """Deliberate: an agent that nets its own fees can use a different rate from the one the
        run is scored on, which is the divergence that makes a simulated maker look profitable."""
        b = _book()
        _ask(b, "100.00", "10", who="mm")
        buyer = ScriptedAgent("buyer")
        for e in b.submit(Order(agent_id="buyer", side=Side.BUY,
                                price=D("100.00"), quantity=D("10"))):
            if e.agent_id == "buyer":
                buyer.on_fill(e)
        # Exactly notional, no fee netted anywhere.
        assert buyer.cash == D("999000")

    def test_maker_fill_rate_is_reported(self) -> None:
        """The number that says whether a strategy earned the maker rate it assumed."""
        b = _book()
        maker = ScriptedAgent("mm")
        b.submit(Order(agent_id="mm", side=Side.SELL, price=D("100.00"), quantity=D("10")))
        for e in b.submit(Order(agent_id="t", side=Side.BUY,
                                price=D("100.00"), quantity=D("10"))):
            if e.agent_id == "mm":
                maker.on_fill(e)
        assert maker.maker_fill_rate == D("1")

    def test_an_agent_with_no_fills_has_no_fill_rate(self) -> None:
        assert ScriptedAgent("idle").maker_fill_rate == D("0")

    def test_mark_to_market_uses_the_supplied_price(self) -> None:
        agent = ScriptedAgent("a", cash=D("1000"))
        agent.position = D("10")
        assert agent.mark_to_market(D("50")) == D("1500")


class TestASimulatedSession:
    """The end-to-end property: a populated book behaves like a book."""

    def test_a_populated_market_produces_trades_and_two_sided_depth(self) -> None:
        b = _book()
        _bid(b, "99.99", "100", who="seed")
        _ask(b, "100.01", "100", who="seed")
        mm = AdaptiveMarketMaker("mm", config=LadderConfig(num_ticks=4, min_order_size=D("30")))
        noise = [NoiseAgent(f"n{i}") for i in range(4)]
        value = ValueAgent("v", r_bar=D("101"), kappa=D("0.3"), sigma_n=D("0.2"))
        rng = Random(11)

        for _ in range(25):
            for order in mm.wakeup(b, rng):
                b.submit(order)
            for agent in (*noise, value):
                for order in agent.wakeup(b, rng):
                    # A wash trade or an off-tick price is refused, as designed.
                    with contextlib.suppress(BookError):
                        b.submit(order)
        assert b.traded_volume > 0, "a populated market with aggressive flow must trade"
        assert b.best_bid is not None and b.best_ask is not None
        assert b.best_bid < b.best_ask, "the book must never cross itself"

    def test_the_book_never_crosses_itself(self) -> None:
        b = _book()
        rng = Random(5)
        _bid(b, "99.99", "200", who="seed")
        _ask(b, "100.01", "200", who="seed")
        mm = AdaptiveMarketMaker("mm")
        for _ in range(15):
            for order in mm.wakeup(b, rng):
                b.submit(order)
            if b.best_bid is not None and b.best_ask is not None:
                assert b.best_bid < b.best_ask
