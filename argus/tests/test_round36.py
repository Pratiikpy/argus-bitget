"""Round 36's judge and hostile-reviewer audits, run offline: each fix on the phrasing that
exposed it, with every live figure stubbed."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from argus.lui import account_math, claims, honesty, memory, server
from argus.lui.research import exit_cost, loss_budget, parse, quick_stats
from argus.lui.research.kinds import ResearchKind, ResearchRequest


class TestJudge:
    def test_the_research_case_is_for_the_name_being_added(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[str] = []

        def case(text: str, symbol: str, book: str, *, detail: bool = False) -> list[str]:
            seen.append(symbol)
            return ["Bottom line: case."]

        monkeypatch.setattr(server, "_research_case_lines", case)
        server._answer("I hold 40% NVDA, 30% MSFT, 30% AAPL — should I add 15% TSLA? Give me "
                       "the full research case: price action, technicals, news.", [],
                       now=None, visitor="local", book="")
        assert seen and seen[0] == "TSLAUSDT"

    def test_a_restated_book_and_shock_in_one_follow_up(self) -> None:
        got = parse._contextual_follow_up(
            "What if my book is 70% BTC, 30% ETH instead, and BTC drops 25%?",
            ["I hold 50% BTC, 50% ETH. If BTC drops 15%, what happens to my book?",
             "What if it drops 25% instead?"], "")
        assert got is not None and got.kind is ResearchKind.STRESS
        assert got.shock_pct == -25 and got.shock_on == "BTCUSDT"
        assert got.book == {"BTCUSDT": pytest.approx(0.7), "ETHUSDT": pytest.approx(0.3)}

    def test_a_hedge_budget_is_not_the_book(self) -> None:
        request = ResearchRequest(kind=ResearchKind.HEDGE, symbols=("ETHUSDT",),
                                  notional=Decimal("500"))
        got = server._with_hedge_budget(request, "I only want to spend $500 on the hedge — what "
                                                 "is the most efficient way to do that?")
        assert got.hedge_budget == Decimal("500") and got.notional is None

    def test_a_total_is_the_account(self) -> None:
        facts = memory.extract("I hold 70% ETH, 30% SOL, total $50,000. How should I hedge?")
        capital = memory.get(facts, "capital")
        assert capital is not None and capital.value == "50000"
        assert memory.get(memory.extract("the deal is worth $5 billion"), "capital") is None

    def test_a_loss_limit_on_the_account_sizes_the_add(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        def worst(weights: dict[str, float], days: int) -> tuple[float, str]:
            # SOL alone falls 40% in its worst month; the BTC/ETH book 5% of the account
            held = sum(w for s, w in weights.items() if s != "SOLUSDT") / 0.8 * 0.05
            return -(held + 0.4 * weights.get("SOLUSDT", 0.0)), "a month"

        monkeypatch.setattr(loss_budget, "worst_window", worst)
        said = loss_budget.lines("My book is 40% BTC, 40% ETH, 20% cash, total $100,000. I cannot "
                                 "lose more than 8% of total capital this month. How large a SOL "
                                 "position can I add without breaching that limit?")
        # $8,000 limit, $5,000 used by the book: $3,000 / 40% = $7,500 of SOL
        assert said is not None and said[0].startswith("Bottom line: about $7,500 of SOL")

    def test_a_pe_claim_is_checked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import estimates

        class Source:
            def summary(self, ticker: str, modules: str) -> dict[str, Any]:
                return {"summaryDetail": {"forwardPE": {"raw": 14.9} if ticker == "NVDA" else None,
                                          "trailingPE": {"raw": 24.9}}}

        monkeypatch.setattr(estimates, "EstimatesSource", Source)
        said = claims.pe_claim_line("A newsletter claims Nvidia's forward P/E is only 12x, "
                                    "cheaper than the S&P 500. Is that true?", ("NVDAUSDT",))
        assert said is not None and "forward P/E is 14.9" in said and "19% below" in said
        assert "cheaper than the index" in said

    def test_a_multiple_is_not_leverage(self) -> None:
        assert parse._LEVERAGE.search(parse._without_multiples(
            "Nvidia's forward P/E is only 12x")) is None
        assert parse._LEVERAGE.search(parse._without_multiples("a 12x long on NVDA"))

    def test_a_move_against_typical_volatility(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(quick_stats, "_realised", lambda s, d: 0.73)
        said = quick_stats.move_vs_vol_lines("COIN fell 6.1% this week. Was that move bigger or "
                                             "smaller than COIN's typical daily volatility?", [])
        assert said is not None and "1.6 times COIN's typical day" in said[0]
        assert "0.6 standard deviations" in said[0]

    def test_the_wall_clock_is_not_signed(self) -> None:
        from argus.agents.desk import WALL_CLOCK

        assert WALL_CLOCK.sub("<t>", "1 analyst(s) ran concurrently in 0.1s; none") == \
            "1 analyst(s) <t>; none"


class TestHostile:
    def test_an_order_beyond_the_day_is_not_priced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import UTC, datetime

        from argus.market import bitget, crossasset_feed, depth
        from argus.market.depth import Level, OrderBook

        book = OrderBook(symbol="BTCUSDT", fetched_at=datetime.now(UTC),
                         bids=(Level(Decimal("99.9"), Decimal("1")),),
                         asks=(Level(Decimal("100.1"), Decimal("1")),))
        monkeypatch.setattr(depth, "fetch_orderbook", lambda s, limit: book)
        monkeypatch.setattr(crossasset_feed, "fetch_taker_bps", lambda s: 6.0)
        ticker = type("T", (), {"base_volume": Decimal("1000"), "last": Decimal("100")})()
        monkeypatch.setattr(bitget, "fetch_tickers", lambda: {"BTCUSDT": ticker})
        said = exit_cost.lines("what is the round-trip cost of a $900,000,000,000 BTC long?")
        assert said is not None and said[0].startswith("Bottom line: it cannot be done as one")

    def test_an_impossible_date_is_named(self) -> None:
        assert honesty.impossible_date("BTC's closing price on February 30, 2027") is not None

    def test_scientific_notation(self) -> None:
        assert parse.parse_notional("My BTC perpetual notional is 5e4 USD") == Decimal("50000")

    def test_a_short_position_pnl(self) -> None:
        said = account_math.position_pnl_lines(
            "I am short BTC with -5000 USD of notional exposure. If price rises 10 percent from "
            "here, what is my dollar P&L?")
        assert said is not None and said[0].startswith("Bottom line: about -$500")

    def test_a_rumour_about_the_exchange(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import bitget

        monkeypatch.setattr(bitget, "fetch_tickers", lambda: {})
        said = server._venue_rumour_lines("Bitget just announced it is shutting down completely. "
                                          "Should I withdraw all my funds?")
        assert said is not None and said[0].startswith("Bottom line: nothing this console reads "
                                                       "confirms that")
        assert server._venue_rumour_lines("is Bitget a good exchange?") is None

    def test_legal_tender_is_unchecked(self) -> None:
        said = claims.unchecked_line("Now that the US government has officially classified "
                                     "Bitcoin as legal tender nationwide starting this week")
        assert said is not None and said.startswith("Not checked")

    def test_negative_leverage(self) -> None:
        said = server._premise_lines("What's my liquidation price for a BTC long with -10x "
                                     "leverage?", None, "")
        assert any("leverage has no sign" in s for s in said)

    def test_future_dates_still_refused(self) -> None:
        assert honesty.future_date_asked_as_past("What was BTC's close on 2030-01-01?",
                                                 date(2026, 10, 4)) == date(2030, 1, 1)

    def test_funding_in_rupees(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import bitget

        monkeypatch.setattr(server, "_usd_inr", lambda: 96.3)
        monkeypatch.setattr(server, "_price_now", lambda s: 85_000.0)
        monkeypatch.setattr(bitget, "public_get", lambda *a, **k: [
            {"fundingRate": "0.0001", "fundingRateInterval": "8"}])
        said = server._rupee_lines("what's the funding rate for BTC perpetual and how much is it "
                                   "in INR terms?", [], now=None, visitor="local")
        assert said is not None and said[0].startswith("Bottom line: BTC's funding rate is "
                                                       "+0.0100% every 8 hours")

    def test_a_hypothetical_failure_is_not_a_rumour(self) -> None:
        assert server._venue_rumour_lines("what happens to my coins if Bitget goes bankrupt?") \
            is None

    def test_a_weighted_name_is_not_a_second_shock(self) -> None:
        assert parse.holding_shocks("and if my portfolio were 20% SOL, 80% BTC and SOL drops "
                                    "30%?") == {}
        assert parse.holding_shocks("BTC and SOL drop 30%") == {"BTCUSDT": -30.0,
                                                                "SOLUSDT": -30.0}
        assert len(parse.holding_shocks("If 50% NVDA and TSLA both fall 10%")) == 2
