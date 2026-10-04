"""Round 34's judge and hostile-reviewer audits, run offline: each fix on the phrasing that
exposed it, with every live figure stubbed."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from argus.lui import account_math, claims, memory, server
from argus.lui.research import macro_moves, quick_stats, venue_compare


class TestBookAcrossTurns:
    def test_units_sold_and_bought_update_the_book(self) -> None:
        book = memory.Fact(kind="book", subject="", value="2",
                           text="I currently hold 2 BTC and 10 ETH as my portfolio",
                           at="2026-10-04")
        got = memory.apply_trades([book], "Update that: I sold all 10 ETH and bought 50 SOL "
                                          "instead. Confirm my new holdings.")
        held = memory.get(got, "book")
        assert held is not None and held.text == "I hold 2 BTC, 50 SOL"
        assert held.replaces.startswith("“I currently hold 2 BTC and 10 ETH")

    def test_a_weights_book_is_left_to_the_sale_reader(self) -> None:
        book = memory.Fact(kind="book", subject="", value="2", text="I hold 60% NVDA, 40% AAPL",
                           at="2026-10-04")
        assert memory.apply_trades([book], "I bought 10 TSLA") == [book]

    def test_how_many_do_i_hold_is_not_a_stated_book(self) -> None:
        assert server._book_overridden("exactly how many ETH do I hold?", "I hold 2 BTC") is None


class TestAccountMath:
    def test_a_parametric_var_is_rechecked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import position_math

        monkeypatch.setattr(position_math, "_last",
                            lambda s: {"BTCUSDT": 84_891.0, "ETHUSDT": 2_695.8}[s])
        said = account_math.lines("I calculated that my portfolio of 3 BTC and 20 ETH has a "
                                  "1-day 95% VaR of $9,400 using BTC vol at 55% annualized and "
                                  "ETH vol at 70% annualized, assuming 0.85 correlation. Did I do "
                                  "that right?")
        assert said is not None and said[0].startswith("Bottom line: no")
        assert "$14,9" in said[0]

    def test_tax_lots_are_valued_and_harvesting_explained(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import position_math

        monkeypatch.setattr(position_math, "_last",
                            lambda s: {"BTCUSDT": 85_000.0, "ETHUSDT": 2_700.0}[s])
        said = account_math.lines("bought 2 BTC at $98,000 each, now down. Bought 50 ETH at "
                                  "$2,200 each, now up. Walk me through tax-loss harvesting.")
        assert said is not None and "unrealised loss of $26,000" in said[0]
        assert "unrealised gain of $25,000" in said[0] and "wash-sale" in said[1]

    def test_the_cost_of_n_coins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import position_math

        monkeypatch.setattr(position_math, "_last", lambda s: 2_700.0)
        said = account_math.lines("calculate the total USD cost of buying 3 ETH")
        assert said is not None and "$8,100" in said[0]

    def test_a_price_per_coin_is_not_a_position(self) -> None:
        said = "bought 2 BTC at $98,000 each, now down. Bought 50 ETH at $2,200 each."
        assert server._each_expanded(said) == said


class TestQuickStats:
    def test_momentum_or_reversion_has_a_verdict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import history

        start = datetime(2026, 7, 1, tzinfo=UTC)
        closes = [100 * (1.01 if i % 2 else 0.99) ** i for i in range(95)]
        bars = [type("C", (), {"close": c, "ts": start + timedelta(days=i)})()
                for i, c in enumerate(closes)]
        monkeypatch.setattr(history, "fetch_window", lambda *a, **k: bars)
        said = quick_stats.momentum_lines("does BTC show momentum or mean reversion?", [])
        assert said is not None and "lag-1 autocorrelation" in said[0]

    def test_depth_is_walked_on_every_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import depth

        class Book:
            def __init__(self, bps: float) -> None:
                self.bps = bps

            def sweep(self, notional: Any, direction: str) -> Any:
                return type("W", (), {"slippage_bps": self.bps, "complete": True})()

        books = {"SOLUSDT": Book(4.0), "XRPUSDT": Book(2.0), "DOGEUSDT": Book(3.0)}
        monkeypatch.setattr(depth, "fetch_orderbook", lambda s, limit: books[s])
        said = quick_stats.depth_lines("compare the order book depth for SOL, XRP and DOGE — "
                                       "which absorbs $500k with the least slippage?", [])
        assert said is not None and said[0].startswith("Bottom line: XRP absorbs a $500,000")


class TestClaims:
    def test_a_delisted_perpetual_is_checked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import bitget

        monkeypatch.setattr(bitget, "public_get", lambda *a, **k: [{"symbolStatus": "normal"}])
        said = claims.corporate_event_line("Did Bitget delist the ETH perpetual futures contract "
                                           "last week?", ("ETHUSDT",))
        assert said is not None and "has not been delisted" in said

    def test_spot_has_no_leverage(self) -> None:
        said = claims.spot_leverage_line("buy 1 BTC on the Bitget spot market using 20x leverage")
        assert said is not None and "a spot buy has no leverage" in said
        assert claims.spot_leverage_line("what is the spot price of BTC") is None


class TestRoutes:
    def test_the_bond_market_splits_hot_and_cool_releases(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        now = datetime.now(UTC)
        times = [now - timedelta(days=d) for d in (40, 70, 100, 130)]
        moves = dict(zip(times, (0.02, 0.03, -0.01, -0.02), strict=True))
        monkeypatch.setattr(macro_moves, "_surprises",
                            lambda ts: dict(zip(times, (5.0, 4.0, -6.0, -4.0), strict=True)))
        said = macro_moves._surprise_line(moves, macro_moves._surprises(times), "CPI")
        assert said is not None and "after the 2 hotter ones the book averaged +2.5%" in said
        assert "after the 2 cooler ones -1.5%" in said

    def test_venue_gaps_against_other_exchanges(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import bitget

        monkeypatch.setattr(bitget, "public_get", lambda *a, **k: [{"lastPr": "85000"}])
        monkeypatch.setattr(venue_compare, "_read", lambda v, b: 85_010.0)
        said = venue_compare.lines("is there an arbitrage gap for BTC with other exchanges?",
                                   ("BTCUSDT",))
        assert said is not None and said[0].startswith("Bottom line: no gap worth capturing")

    def test_a_follow_up_that_repeats_the_first_answer_is_not_one(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        same = {"lines": ["Bottom line: a whale is a holder big enough to move the price."],
                "classified_by": "newcomer"}
        monkeypatch.setattr(server, "_answer", lambda *a, **k: dict(same))
        got = server._as_follow_up("should a 16 year old be trading this stuff",
                                   ["what is a whale"], {"refused": True, "classified_by":
                                                         "declined"},
                                   now=None, visitor="local", book="")
        assert got is None


class TestLiveReAskRound34:
    def test_an_age_is_not_a_weight(self) -> None:
        from argus.lui.research.analogue import _ALL_IN_Q

        assert _ALL_IN_Q.search("I'm 58, planning to retire")
        assert not _ALL_IN_Q.search("I'm 40/60 BTC/ETH - how did this book do on CPI days?")

    def test_the_second_tax_lot_is_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import position_math

        monkeypatch.setattr(position_math, "_last",
                            lambda s: {"BTCUSDT": 85_000.0, "ETHUSDT": 2_700.0}[s])
        said = account_math.lines("I bought 1 ETH at $3,500 and 1 BTC at $70,000 - which lot "
                                  "should I sell for tax-loss harvesting?")
        assert said is not None and "ETH:" in said[0] and "BTC:" in said[0]

    def test_deepest_book(self) -> None:
        assert quick_stats.DEPTH_ASKED.search("which has the deepest book for a $1M buy: BNB, "
                                              "ADA or AVAX?")

    def test_gets_back_to_that_level(self) -> None:
        assert quick_stats.BACK_TO_LEVEL.search("what are the odds it gets back to that level?")

    def test_higher_on_another_exchange(self) -> None:
        assert venue_compare.ASKED.search("does BTC trade higher on Coinbase than on Bitget?")

    def test_a_trade_update_starts_from_the_units_said_before(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import parse

        monkeypatch.setattr(parse, "saved_book_lines", lambda text: ["Bottom line: priced."])
        got = server._round27_follow_up("i just sold 2 ETH and picked up 20 SOL, whats my book "
                                        "now?", ["my holdings: 1 BTC and 5 ETH"], now=None,
                                        visitor="local", book="")
        assert got is not None and "1 BTC, 3 ETH, 20 SOL" in got["lines"][0]
