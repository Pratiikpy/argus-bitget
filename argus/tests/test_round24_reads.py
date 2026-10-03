"""Round 24: the questions an audit round's re-asks found misread, each read correctly."""

from __future__ import annotations

import pytest

from argus.lui.research import parse


class TestRound23LiveReAsks:
    def test_a_stated_period_without_a_preposition_is_read(self) -> None:
        assert parse._period_days("compare the last 3 months of NVDA vs AMD returns") == 90
        assert parse._period_days("NVDA over the last 2 weeks") == 14


class TestHostileReviewRound24:
    def test_every_leg_is_marked_at_the_traders_price(self) -> None:
        from argus.lui.journal import position_and_pnl

        def first(text: str) -> str:
            said = position_and_pnl(text, price=lambda s: 1.0)
            assert said is not None, text
            return said[0][0]

        assert "+$750.00" in first("I'm short 50 AAPL at 210, mark is 195. What's my P&L?")
        assert "+$14,000.00" in first("Short 2 BTC perp from 95k, BTC is at 88k. Unrealized?")
        assert "+$1,000.00" in first(
            "I'm long 100 NVDA at 180. Oops, I meant short, not long. Price is now 170. P&L?")
        assert "+$2,000.00" in first("Long 300 AMD bought at 150, short 100 NVDA sold at 190. "
                                     "AMD now 160, NVDA now 200. Net P&L?")
        assert "+$2,600.00" in first("Sold 100 MSFT short at 500, covered 40 at 480, rest still "
                                     "open; MSFT now 470. Total P&L?")
        assert "+$30,000.00" in first("I'm short 5 NQ from 21,000; NQ is now 20,700. P&L?")
        assert "-$10,000.00" in first("I hold 100 oz of gold at 2,400. Gold now 2,300. P&L?")

    def test_futures_moves_are_contracts_times_multiplier_times_points(self) -> None:
        from argus.lui import server

        def lead(text: str) -> str:
            said = server._futures_move_lines(text)
            assert said is not None, text
            return said[0]

        assert "a gain of $6,000" in lead("I bought 2 NQ futures. Nasdaq 100 rallies 150 points.")
        assert "a loss of $600" in lead("Long 3 micro E-mini S&P contracts. The index moves 40 "
                                        "points against me. Loss?")
        assert "a loss of $40,000" in lead("I'm long 5 E-mini Nasdaq contracts and the index "
                                           "drops 2% from 20,000. Dollar loss?")
        assert "a loss of $18,000" in lead("Short 4 ES contracts. Index climbs 1.5% from 6000.")

    def test_stated_funding_options_and_leverage_from_entry(self) -> None:
        from argus.lui import server

        paid = server._stated_funding("Short $200k notional ETH perp, funding -0.01% per 8h. "
                                      "Over 3 days do I pay or receive?")
        assert paid is not None and "short pays about $60.00 a day" in paid[0]
        assert "pays about $180.00" in paid[0]
        hourly = server._stated_funding("Long $100,000 BTC perp, funding 0.03% per hour. "
                                        "What's the cost over one day?")
        assert hourly is not None and "pays about $720.00 a day" in hourly[0]
        spread = server._option_lines("250/260 NVDA call spread, paid 3.20, 5 contracts. Max "
                                      "profit?")
        assert spread is not None and "maximum profit $3,400" in spread[0]
        puts = server._option_lines("I sold 2 puts on SPY strike 500 for $6 premium. SPY expires "
                                    "at 480. P&L?")
        assert puts is not None and "-$2,800" in puts[1]

    def test_amount_forms_are_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import server

        monkeypatch.setattr(parse, "_last_prices", lambda: {"TSLAUSDT": 370.0})
        parse._PRICED.clear()

        assert server._plain_amounts("25 lakh rupees in NVDA") == "INR 2500000 in NVDA"
        assert server._plain_amounts("₹2 crore in NVDA") == "INR 20000000 in NVDA"
        assert server._plain_amounts("CHF 80'000 in META") == "CHF 80000 in META"
        priced = parse.priced_book("I hold 2m dollars of NVDA, 1.5m of MSFT")
        assert priced is not None and priced.value == 3_500_000
        short = parse.priced_book("TSLA -50 shares")
        assert short is not None and short.weights == {"TSLAUSDT": -1.0}


class TestFirstTimeUserRound24:
    def test_plain_questions_get_plain_answers(self) -> None:
        from argus.lui import newcomer

        for asked in ("what even is a stock", "what is a margin call",
                      "and what's isolated vs cross margin", "whats the safest thing on here",
                      "should i revenge trade to win back my losses",
                      "I am 68 and retired. Is this suitable for someone like me?",
                      "how do people make money trading", "can i turn 300 into 3000",
                      "i keep losing money trading. what am i doing wrong",
                      "what leverage is safe then",
                      "is a stablecoin safe", "can i buy half a share", "Do I get voting rights?"):
            said = newcomer.reply(asked)
            assert said is not None and said.lines, asked

    def test_hinglish_and_follow_ups_are_read(self) -> None:
        from argus.lui import server

        for spoken, english in (
                ("order place kar sakta hu?", "how do I place a trade"),
                ("leverage kya hota hai bhai", "what is leverage"),
                ("mujhe trading sikhna hai, kahan se start karu",
                 "I am new to trading, where do I start")):
            said = server._restated(spoken, [])
            assert said is not None and said[0] == english, spoken
        aur = server._restated("aur 2x pe?",
                               ["solana pe 5x leverage lu toh liquidation kahan hoga"])
        assert aur is None or "2x" in aur[0]

    def test_kelly_and_rules_and_rebalance(self) -> None:
        from argus.lui import server

        kelly = server._kelly_lines("I have $20,000. Kelly says what for a strategy with 55% win "
                                    "rate and 1:1 payoff?", [])
        assert kelly is not None and "$2,000 of your $20,000" in kelly[0]
        half = server._kelly_lines("half Kelly then?", [
            "I have $20,000. Kelly says what for a strategy with 55% win rate and 1:1 payoff?"])
        assert half is not None and "$1,000" in half[0]
        broken = server._cap_check_lines("My rule: never more than 20% in one name. Am I breaking "
                                         "it?", [], "35% NVDA, 25% TSLA, 20% AAPL, 20% cash")
        assert broken is not None and "NVDA (35%), TSLA (25%)" in broken[0]
        trades = server._equal_weight_trades("Rebalance me to equal weight - what trades do I need "
                                             "on a 60k account?", "50% NVDA, 30% AAPL, 20% MSFT")
        assert trades is not None
        assert "sell $10,000 of NVDA; buy $2,000 of AAPL; buy $8,000 of MSFT" in trades[0]

    def test_periods_are_read(self) -> None:
        from datetime import UTC, datetime

        from argus.lui.research.performance import PERFORMANCE_Q, asked_period

        now = datetime(2026, 10, 3, tzinfo=UTC)
        ytd = asked_period("how much has it gone up this year", now)
        assert ytd is not None and ytd.start == datetime(2026, 1, 1, tzinfo=UTC)
        last = asked_period("BTC last year", now)
        assert last is not None and last.start.year == 2025
        rolling = asked_period("over the last year", now)
        assert rolling is not None and rolling.start.year == 2025 and rolling.start.month == 10
        september = asked_period("How did the Nasdaq do in September?", now)
        assert september is not None and september.start.month == 9
        six = asked_period("how did AMD do over the past 6 months", now)
        assert six is not None and (now - six.start).days == 180
        assert PERFORMANCE_Q.search("which one had the bigger drawdown in that window?")

    def test_leverage_from_the_stated_entry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import server
        from argus.market import bitget

        monkeypatch.setattr(bitget, "maintenance_margin_rate", lambda s, n: 0.004)
        monkeypatch.setattr(server, "_price_now", lambda s: 84_500.0)
        said = server._levered_entry_lines("i'm long 0.5 btc at 10x from 120000, where's my "
                                           "liquidation")
        assert said is not None and "liquidated near 108,480.00" in said[0]
        assert any("would already have been liquidated" in x for x in said)


class TestRound24Remainder:
    def test_a_release_is_anchored_where_it_is_first_traded(self) -> None:
        from datetime import UTC, datetime

        from argus.research.event_reactions import release_anchor

        # Apple, 30 Oct 2025: released 16:30 New York, 8-K accepted 2025-10-31T00:30:35Z (EDGAR)
        after_close = release_anchor(datetime(2025, 10, 31, 0, 30, 35, tzinfo=UTC))
        assert after_close == datetime(2025, 10, 30, 20, 0, tzinfo=UTC)
        # a pre-open 8-K is first traded from the previous close
        pre_open = release_anchor(datetime(2026, 1, 27, 11, 0, tzinfo=UTC))
        assert pre_open == datetime(2026, 1, 26, 21, 0, tzinfo=UTC)
        # one accepted inside the session is its own anchor
        inside = datetime(2026, 3, 4, 15, 0, tzinfo=UTC)
        assert release_anchor(inside) == inside

    def test_follow_ups_that_lean_on_the_question_before(self) -> None:
        from argus.lui import server

        assert server._leaning_follow_up(
            "what's the bear case?", ["I am bullish on MSFT because Azure growth is accelerating. "
                                      "Test my thesis.", "and how has the stock done?"]) == \
            "what would prove it wrong?"
        assert server._leaning_follow_up(
            "is the sentiment positive or negative?",
            ["What are the latest news headlines on Apple?"]) == \
            "is the sentiment positive or negative on AAPL?"
        assert server._leaning_follow_up(
            "what did the stock do the day after?", ["When does GOOGL report earnings?"]) == \
            "how much does GOOGL usually move on earnings?"
        assert server._leaning_follow_up("is it dangerous", ["what is a perp"]) == \
            "what leverage is safe then"
        assert server._leaning_follow_up("should I DCA into BTC", ["I have $1,000"]) == \
            "should I DCA into BTC with $1,000?"
        stop = server._STOP_SAID.search("what if the stop is 6% instead?")
        assert stop is not None

    def test_holdings_are_ranked_by_their_own_hourly_fit(self, monkeypatch: pytest.MonkeyPatch
                                                         ) -> None:
        from datetime import UTC, datetime, timedelta
        from types import SimpleNamespace

        from argus.lui import server
        from argus.market import history

        start = datetime(2026, 9, 1, tzinfo=UTC)
        base = [100.0 * (1.01 if i % 2 else 0.99) ** (i % 7) for i in range(200)]
        paths = {"BTCUSDT": base, "MSTRUSDT": [2 * x for x in base],
                 "AAPLUSDT": [100.0 + (i % 3) for i in range(200)]}

        def fake(symbol: str, *, days: int = 30, interval: str = "1H") -> list[object]:
            return [SimpleNamespace(ts=start + timedelta(hours=i), close=c)
                    for i, c in enumerate(paths[symbol])]

        monkeypatch.setattr(history, "fetch_range", fake)
        fits = server._hourly_fit(["MSTRUSDT", "AAPLUSDT"], "BTCUSDT")
        assert fits["MSTRUSDT"][0] > 0.99 and abs(fits["MSTRUSDT"][1] - 1.0) < 1e-9
        assert fits["AAPLUSDT"][0] < fits["MSTRUSDT"][0]
