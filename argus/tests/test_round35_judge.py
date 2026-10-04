"""Round 35's judge audit, run offline: each fix on the phrasing that exposed it, with every live
figure stubbed."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from argus.lui import claims, exposures, memory, server
from argus.lui.research import literature, personal_profile, quick_stats, tracking, venue_facts
from argus.lui.research.kinds import ResearchKind, ResearchRequest
from argus.lui.research.parse import holding_pairs, with_book

BOOK_SAID = ("Here is my book: $40,000 BTC, $25,000 ETH, $15,000 NVDA, $20,000 GLD. Remember that "
             "I cannot add any new position larger than $10,000 without checking with you first.")


class TestRememberedLimit:
    def test_the_limit_and_the_dollar_book_are_kept(self) -> None:
        facts = memory.extract(BOOK_SAID)
        cap = memory.get(facts, "cap_usd")
        held = memory.get(facts, "book")
        assert cap is not None and cap.value == "10000"
        assert held is not None and "$40,000 BTC" in held.text

    def test_a_larger_position_is_set_against_it(self) -> None:
        facts = memory.extract(BOOK_SAID)
        request = ResearchRequest(kind=ResearchKind.IMPACT, symbols=("SOLUSDT",),
                                  notional=Decimal("18000"))
        said = memory.size_limit_line(request, facts)
        assert said is not None and said.startswith("Against your own limit")
        assert "$8,000 over your $10,000" in said

    def test_a_smaller_one_is_inside_it(self) -> None:
        facts = memory.extract(BOOK_SAID)
        request = ResearchRequest(kind=ResearchKind.IMPACT, symbols=("SOLUSDT",),
                                  notional=Decimal("5000"))
        said = memory.size_limit_line(request, facts)
        assert said is not None and said.startswith("Inside your own limit")

    def test_a_dollar_add_is_sized_against_the_saved_book(self) -> None:
        request = ResearchRequest(kind=ResearchKind.IMPACT, symbols=("SOLUSDT",),
                                  notional=Decimal("18000"))
        got = with_book(request, "my book: $40,000 BTC, $25,000 ETH, $15,000 NVDA, $20,000 GLD",
                        "I want to add $18,000 of SOL to my book. Is that fine?")
        assert got is not None and got.size_stated and abs(got.size - 18 / 118) < 1e-3


class TestDollarBooks:
    def test_a_dollar_value_is_not_a_coin_count(self) -> None:
        assert holding_pairs("My book is $30000 BTC, $20000 COIN, $25000 NVDA, $25000 MSFT.") == []

    def test_an_unlisted_stock_keeps_its_share(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(exposures, "unlisted_row", lambda t: {
            "kind": "stock", "currency": "USD", "weights": {"Health Care": 1.0},
            "ticker": t} if t == "PFE" else None)
        book, _, notes = exposures.parse("My book is $25000 AAPL, $25000 JPM, $25000 XOM, "
                                         "$25000 PFE. What is the sector exposure of this book?")
        assert book == {"AAPLUSDT": 0.25, "JPMUSDT": 0.25, "XOMUSDT": 0.25, "YAHOO:PFE": 0.25}
        assert any("PFE is not a contract Bitget lists" in n for n in notes)

    def test_the_mcp_book_keeps_it_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(exposures, "unlisted_row", lambda t: {
            "kind": "stock", "currency": "USD", "weights": {"Health Care": 1.0},
            "ticker": t} if t == "PFE" else None)
        book, _, _ = exposures.parse("what are my sector and factor exposures",
                                     "25% AAPL, 25% JPM, 25% XOM, 25% PFE")
        assert abs(book["YAHOO:PFE"] - 0.25) < 1e-9 and abs(sum(book.values()) - 1) < 1e-9


class TestTables:
    def test_prices_with_their_day(self) -> None:
        said = ["Bottom line: 2 levels, each with its own source below.",
                "BTC: 84,923.20, +0.44% over 24 hours — Bitget's USDT perpetual, last price.",
                "ETH: 2,692.32, -0.46% over 24 hours — Bitget's USDT perpetual, last price."]
        table = venue_facts.levels_table(said)
        assert table is not None and table["rows"] == [["BTC", "84,923.20", "+0.44%"],
                                                       ["ETH", "2,692.32", "-0.46%"]]

    def test_funding_day_by_day(self) -> None:
        said = ["Bottom line: ...", "BTC day by day (UTC, newest first; each day's settlements "
                "averaged): 04 Oct +0.0062%; 03 Oct +0.0082%."]
        table = venue_facts.funding_table(said)
        assert table is not None and table["rows"] == [["04 Oct", "+0.0062%"],
                                                       ["03 Oct", "+0.0082%"]]


class TestPersonalThesis:
    FIRST = ("I hold 60% BTC and 40% ETH. I'm a 29-year-old software engineer, 10-year horizon, "
             "high risk tolerance. What is your personalized thesis for me?")
    THEN = ("Actually, scratch that - I now have a mortgage and two young kids, and I need this "
            "exact money for a house down payment in 2 years. Does your thesis change?")

    def test_the_circumstances_are_read(self) -> None:
        before = personal_profile.read([self.FIRST])
        after = personal_profile.read([self.FIRST, self.THEN])
        assert (before.age, before.horizon_years, before.tolerance) == (29, 10.0, "high")
        assert after.need_years == 2.0 and after.need_for == "house down payment"
        assert after.obligations == ("a mortgage", "children") and after.years == 2.0

    def test_the_change_is_measured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stretch = personal_profile.Stretches(
            years=2.0, first=date(2017, 11, 9), last=date(2026, 10, 2), count=2522, worst=-0.72,
            median=1.05, best=15.19, down_share=0.30, deepest=-0.86,
            deepest_from=date(2018, 1, 7), deepest_to=date(2018, 12, 15))
        monkeypatch.setattr(personal_profile, "stretches", lambda book, years: stretch)
        said = personal_profile.lines(personal_profile.read([self.FIRST, self.THEN]),
                                      {"BTCUSDT": 0.6, "ETHUSDT": 0.4}, "60% BTC, 40% ETH",
                                      changed=True, before=personal_profile.read([self.FIRST]))
        assert said[0].startswith("What changed: the horizon, from 10 years to 2")
        assert any("ended down in 30% of them; the worst lost 72%" in x for x in said)
        assert any("cannot be waited out" in x for x in said)

    def test_a_follow_up_is_routed_to_it(self) -> None:
        assert personal_profile.CHANGED.search(self.THEN)


class TestRoutes:
    def test_which_engine_answered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(server, "_answer", lambda *a, **k: {
            "lines": ["Bottom line: 3 levels."], "classified_by": "research",
            "matched": "research"})
        said = server._engine_explained("Explain why your last answer chose that engine, and "
                                        "what other engine could have answered it.",
                                        ["Give me BTC and ETH prices"], now=None,
                                        visitor="local", book="")
        assert said is not None and said[0].startswith("Bottom line: your last question")
        assert any(x.startswith("How routing works") for x in said)

    def test_a_book_sharpe_goes_to_the_book(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import book as book_engine

        monkeypatch.setattr(book_engine, "_book_history_lines",
                            lambda w, c, q, days_back=500: (["Bottom line: held at ..."], None))
        said = server._book_history_answer("What is the max drawdown of this book over the last "
                                           "year, and its Sharpe ratio?", "40% BTC, 60% ETH")
        assert said is not None and said[0] == "Bottom line: held at ..."

    def test_tracking_error_on_a_coin_written_as_an_rtoken(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import rtoken_spot

        def bars(symbol: str, *, spot: bool, days: int = 30) -> dict[int, tuple[float, float]]:
            return {i * 3_600_000: (100.0, 100.0 + (0.05 if spot and i % 2 else 0.0) + i * 0.01)
                    for i in range(200)}

        monkeypatch.setattr(rtoken_spot, "hourly_bars", bars)
        said = tracking.lines("Compare rSOL and rETH against their perpetuals. What is the "
                              "tracking error on each over the last month?")
        assert said is not None and said[0].startswith("Bottom line: rSOL and rETH are not on")
        assert any(x.startswith("SOL spot against the SOL perpetual: tracking error") for x in said)

    def test_momentum_tested_on_bitget_data(self) -> None:
        assert quick_stats.MOMENTUM_ASKED.search("Now test momentum on Bitget BTC data - is BTC's "
                                                 "recent trend likely to continue or reverse?")


class TestPremisesAndPapers:
    def test_an_eps_said_above_consensus_that_is_below(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import estimates

        class Source:
            def summary(self, ticker: str, modules: str) -> dict[str, Any]:
                return {"earningsTrend": {"trend": [{
                    "period": "0q", "endDate": "2026-10-31",
                    "earningsEstimate": {"avg": {"raw": 2.47}, "numberOfAnalysts": {"raw": 43}}}]}}

        monkeypatch.setattr(estimates, "EstimatesSource", Source)
        said = claims.consensus_line("NVDA's whisper number for next quarter is $1.20 EPS, way "
                                     "above consensus.", ("NVDAUSDT",))
        assert said is not None and "1.20 is 51% below it, not above" in said

    def test_the_subject_word_outweighs_generic_ones(self) -> None:
        wanted = literature._words("the momentum factor in asset returns")
        assert (literature._match(wanted, "Momentum crashes")
                > literature._match(wanted, "The Impact of ESG Factors on Asset Returns"))


class TestNewcomerReCheck:
    def test_a_past_bear_market_is_a_period(self) -> None:
        from argus.lui.research.performance import asked_period

        period = asked_period("how did BTC do in the 2022 bear market?")
        assert period is not None and period.said.startswith("in 2022")

    def test_a_bear_market_now_is_measured_not_defined(self) -> None:
        from argus.lui import newcomer

        assert quick_stats.BEAR_ASKED.search("is NVDA in a bear market right now?")
        assert newcomer.reply("is NVDA in a bear market right now?", named=True) is None

    def test_a_stop_level_is_not_a_price_claim(self) -> None:
        assert server._price_premise("my stop loss on ETH is at 2500, does it still protect "
                                     "me over the weekend?") is None


class TestExitCost:
    """Build list 4.1: the leg asked, at the size said, walked on a captured book by hand."""

    @staticmethod
    def _book() -> object:
        from datetime import UTC, datetime

        from argus.market.depth import Level, OrderBook

        return OrderBook(symbol="SOLUSDT", fetched_at=datetime.now(UTC),
                         bids=(Level(Decimal("99.9"), Decimal("10")),
                               Level(Decimal("99.8"), Decimal("100"))),
                         asks=(Level(Decimal("100.1"), Decimal("10")),
                               Level(Decimal("100.2"), Decimal("100"))))

    def test_closing_units_walks_the_bids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import exit_cost, parse
        from argus.market import crossasset_feed, depth

        monkeypatch.setattr(depth, "fetch_orderbook", lambda s, limit: self._book())
        monkeypatch.setattr(crossasset_feed, "fetch_taker_bps", lambda s: 6.0)
        monkeypatch.setattr(parse, "last_price", lambda s: 100.0)
        said = exit_cost.lines("I hold 20 SOL. What would it cost me to close it out right now?")
        # $2,000 sold: $999 at 99.9 (10 SOL), $1,001 at 99.8 (10.03 SOL): average 99.85 against
        # a mid of 100 is 15.0bps, plus 6bps fee = 21.0bps
        assert said is not None and said[0].startswith("Bottom line: closing $2,000 of SOL costs "
                                                       "about 21.0bps ($4)")
        assert "selling into the bids" in said[1] and "over 2 levels" in said[1]

    def test_a_round_trip_walks_both_sides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import exit_cost, parse
        from argus.market import crossasset_feed, depth

        monkeypatch.setattr(depth, "fetch_orderbook", lambda s, limit: self._book())
        monkeypatch.setattr(crossasset_feed, "fetch_taker_bps", lambda s: 6.0)
        monkeypatch.setattr(parse, "last_price", lambda s: 100.0)
        said = exit_cost.lines("what does it cost to enter and exit a $500 SOL position?")
        # $500 inside the first level each side: 10bps each way plus 6bps fee = 32bps
        assert said is not None and "a round trip in $500 of SOL costs about 32.0bps" in said[0]


class TestHoldingsSaid:
    def test_holdings_after_a_colon_are_kept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import parse

        monkeypatch.setattr(parse, "_value_holdings", lambda held, notes: (
            {s: 1 / len(held["holdings_units"]) for s in held["holdings_units"]}, []))
        held = memory.get(memory.extract("my holdings: 3 BTC and 20 ETH"), "book")
        assert held is not None and held.text == "my holdings: 3 BTC and 20 ETH"
