"""The console answers an arbitrage question on the live books, by the exact two-book walk
(lui/arbitrage.py; capability 08). Books and fees are stubbed here; the walk is the real one."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from argus.lui import arbitrage
from argus.market.depth import Level, OrderBook

NOW = datetime(2026, 9, 28, 14, tzinfo=UTC)


def book(symbol: str, bid: str, ask: str, size: str = "10") -> OrderBook:
    return OrderBook(symbol=symbol, fetched_at=NOW,
                     bids=(Level(Decimal(bid), Decimal(size)),),
                     asks=(Level(Decimal(ask), Decimal(size)),))


@pytest.fixture
def venue(monkeypatch: pytest.MonkeyPatch) -> dict[str, OrderBook]:
    books: dict[str, OrderBook] = {}
    monkeypatch.setattr("argus.market.depth.fetch_orderbook",
                        lambda symbol, **_: books[symbol])
    monkeypatch.setattr("argus.market.crossasset_feed.fetch_taker_bps", lambda symbol: 6.0)
    monkeypatch.setattr(arbitrage, "_spot_fee",
                        lambda spot: Decimal("0.001") if spot.startswith("RNVDA") else None)
    return books


def test_the_question_is_recognised_only_with_a_name() -> None:
    assert arbitrage.asks_for_arbitrage("is there an arbitrage between NVDA spot and perp")
    assert arbitrage.asks_for_arbitrage("NVDA spot vs perp")
    assert not arbitrage.asks_for_arbitrage("is there an arbitrage anywhere")
    assert not arbitrage.asks_for_arbitrage("what is the NVDA price")


def test_a_gap_that_does_not_cover_both_fees_is_a_no(venue: dict[str, OrderBook]) -> None:
    venue["RNVDAUSDT"] = book("RNVDAUSDT", "231.22", "231.31")
    venue["NVDAUSDT"] = book("NVDAUSDT", "231.45", "231.46")
    lines, sources, data = arbitrage.answer("NVDA spot vs perp arbitrage")
    assert lines[0].startswith("Bottom line: no — at the touch the gap between RNVDAUSDT spot "
                               "and NVDAUSDT perpetual does not cover the two taker fees "
                               "(16bps together)")
    assert data["arbitrage"]["monetizable"] is False
    assert any("mid-to-mid gap" in line for line in lines)
    assert "executable_arb" in sources[-1].ref


def test_a_real_gap_is_sized_and_netted(venue: dict[str, OrderBook]) -> None:
    venue["RNVDAUSDT"] = book("RNVDAUSDT", "199.9", "200.0", size="100")
    venue["NVDAUSDT"] = book("NVDAUSDT", "205.0", "205.1", size="100")
    lines, _, data = arbitrage.answer("NVDA spot vs perp arbitrage")
    assert lines[0].startswith("Bottom line: yes — buying 100.0000 on the RNVDAUSDT spot and "
                               "selling on the NVDAUSDT perpetual nets about $")
    assert Decimal(data["arbitrage"]["best"]["net"]) > 1


def test_a_gain_under_a_dollar_is_not_called_a_yes(venue: dict[str, OrderBook]) -> None:
    venue["RNVDAUSDT"] = book("RNVDAUSDT", "199.9", "200.0", size="0.01")
    venue["NVDAUSDT"] = book("NVDAUSDT", "205.0", "205.1", size="0.01")
    lines, _, _ = arbitrage.answer("NVDA spot vs perp arbitrage")
    assert lines[0].startswith("Bottom line: technically, by $")


def test_no_spot_leg_is_said_plainly(venue: dict[str, OrderBook]) -> None:
    lines, _, data = arbitrage.answer("TSLA spot vs perp arbitrage")
    assert lines[0].startswith("Bottom line: Bitget lists no spot rToken for TSLA (RTSLAUSDT)")
    assert data["arbitrage"] is None
