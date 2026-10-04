"""Round 32's judge and hostile-reviewer audits, run offline: each fix is checked on the phrasing
that exposed it and on the near-miss beside it, with every live figure stubbed."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from argus.lui import account_math, claims, server
from argus.lui.research import inflation_hedge, macro_moves, parse


class TestAccountMath:
    def test_successive_losses_compound_on_the_balance(self) -> None:
        said = account_math.lines("My portfolio dropped 50% in March, then dropped another 50% in "
                                  "April. What is my total percentage loss, and if I started "
                                  "with $2,000,000, what is my balance now?")
        assert said is not None and "-75.00%" in said[0]
        assert "$500,000 now" in said[1]

    def test_margin_and_exposure_in_k_and_m(self) -> None:
        said = account_math.lines("I deposit 1.5M USDT and open a 300k notional position at 20x "
                                  "leverage. What is my required margin in k, and what is my "
                                  "effective exposure in M?")
        assert said is not None and "$15,000 of margin (15.0k)" in said[0] and "0.30M" in said[0]
        assert "0.20x" in said[2]

    def test_equity_from_realised_and_unrealised(self) -> None:
        said = account_math.lines("My realized PnL this week is -$3,200 and my unrealized PnL is "
                                  "-$1,800. My starting equity was $10,000. What is my current "
                                  "equity and my total PnL as a percentage?")
        assert said is not None and said[0].startswith("Bottom line: equity is $5,000")
        assert "-50.0%" in said[0]

    def test_a_stated_move_past_liquidation_loses_the_margin(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import position_math
        from argus.market import bitget

        monkeypatch.setattr(position_math, "_last", lambda symbol: 85_000.0)
        monkeypatch.setattr(bitget, "maintenance_margin_rate", lambda symbol, notional: 0.004)
        said = account_math.lines("I have 0.001 BTC as margin and open a 125x long on BTCUSDT. If "
                                  "BTC drops 2% from my entry price, am I liquidated, and what is "
                                  "my PnL in USD?")
        assert said is not None and said[0].startswith("Bottom line: yes")
        assert "0.40% move" in said[0] and "$85.00" in said[0]

    def test_a_move_short_of_liquidation_is_pnl_on_margin(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import bitget

        monkeypatch.setattr(bitget, "maintenance_margin_rate", lambda symbol, notional: 0.004)
        said = account_math.lines("100 usdt margin, 10x long BTC, BTC rises 3%, what is my pnl")
        assert said is not None and said[0].startswith("Bottom line: no")
        assert "+30% on the $100" in said[0]

    def test_a_plain_question_is_not_taken(self) -> None:
        assert account_math.lines("what is BTC doing today") is None


class TestStatedBooks:
    def test_each_dollar_size_is_written_out(self) -> None:
        said = server._each_expanded("I hold equity perps: TSLA, AAPL, and AMZN, $50,000 notional "
                                     "each. What is my risk?")
        assert said.endswith("(I hold $50,000 TSLA, $50,000 AAPL, $50,000 AMZN.)")

    def test_a_resize_takes_each_sum_after_its_name(self) -> None:
        said = server._each_expanded("Now assume I double my TSLA position to $100,000 and keep "
                                     "AAPL and AMZN at $50,000 each.")
        assert said.endswith("(I hold $100,000 TSLA, $50,000 AAPL, $50,000 AMZN.)")

    def test_equal_notional_is_equal_weights(self) -> None:
        said = server._each_expanded("long BTC, long ETH, long SOL, all 5x leverage, equal "
                                     "notional. How should I size a hedge?")
        assert said.endswith("(I hold 34% BTC, 33% ETH, 33% SOL.)")

    @pytest.mark.parametrize("text", ["I hold $5,000 BTC and $3,000 ETH, risk?",
                                      "I hold 50% NVDA, 50% AAPL, equal weights",
                                      "compare BTC and ETH"])
    def test_a_book_already_written_is_left_alone(self, text: str) -> None:
        assert server._each_expanded(text) == text

    def test_a_position_size_is_not_a_price_claim(self) -> None:
        assert server._price_premise("keep AAPL and AMZN at $50,000 each") is None


class TestJudgeRoutes:
    @pytest.mark.parametrize("said", [
        "Given my book, write me a personalized thesis for the next quarter — what should I "
        "specifically be worried about because of what I hold?",
        "What does a market drop do to my book, and what should I be worried about given what I "
        "hold?",
    ])
    def test_a_personal_thesis_is_asked(self, said: str) -> None:
        assert server._PERSONAL_THESIS.search(said)

    def test_a_personal_thesis_without_a_book_asks_for_it(self) -> None:
        said = server._personal_thesis_lines("write me a personalized thesis for my book", "", [],
                                             now=None, visitor="local")
        assert said is not None and "needs the book" in said[0]

    def test_the_thesis_leads_with_the_crowded_name_and_the_fall(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        answers = {
            "what does a 10% market drop do to my book": [
                "Bottom line: If QQQ moves -10%: your book moves about -14.79%, hardest hit AMD."],
            "what is my risk": [
                "Bottom line: the risk is concentrated in AVAX (35% of the money, 53% of the "
                "risk)."],
            "what should I keep an eye on this week": ["Bottom line: FOMC minutes on Wednesday."],
            "when does AMD report earnings": ["Bottom line: AMD reports in 30 days, on 03 Nov."],
        }
        monkeypatch.setattr(server, "_answer", lambda q, *a, **k: {"lines": answers.get(q, [])})
        said = server._personal_thesis_lines("what should I be worried about given what I hold",
                                             "35% AMD, 35% AVAX, 30% XAUUSDT", [], now=None,
                                             visitor="local")
        assert said is not None
        assert said[0].startswith("Bottom line: the worry in this book")
        assert "AVAX — 53% of the risk on 35% of the money" in said[0]
        assert "takes about 14.79% off the whole book" in said[0]
        assert any(x.startswith("Earnings in the book: AMD") for x in said)

    def test_a_named_valuation_ratio_is_not_the_price_ratio(self) -> None:
        assert not parse._RATIO_Q.search("Compare NVDA and AMD on trailing P/E ratio")
        assert not parse._RATIO_Q.search("what is the sharpe ratio")
        assert parse._RATIO_Q.search("what is the ETH/BTC price ratio")

    def test_hawkish_moves_rates_up(self) -> None:
        from argus.lui.research.macro import RATES_UP

        assert RATES_UP.search("what happens if the Fed surprises hawkish")

    def test_rates_rising_is_a_macro_question(self) -> None:
        assert parse._MACRO.search("what happens to my book if rates rise")


class TestBookOnReleases:
    def test_book_moves_are_weighted_per_release(self, monkeypatch: pytest.MonkeyPatch) -> None:
        now = datetime.now(UTC)
        times = [now - timedelta(days=40), now - timedelta(days=100), now - timedelta(days=160)]
        moves = {("AUSDT", times[0]): 0.10, ("BUSDT", times[0]): -0.10,
                 ("AUSDT", times[1]): -0.04, ("BUSDT", times[1]): -0.08,
                 ("AUSDT", times[2]): 0.02, ("BUSDT", times[2]): 0.04}
        from argus.research import event_reactions

        monkeypatch.setattr(event_reactions, "fomc_decisions", lambda: times)
        monkeypatch.setattr(macro_moves, "_around", lambda symbol, at: moves[(symbol, at)])
        said = macro_moves.book_event_lines([("AUSDT", 0.5), ("BUSDT", 0.5)], "FOMC")
        assert said is not None
        assert f"its worst was -6.0% after {times[1]:%d %b %Y}" in said[0]
        assert "on the last 3 Fed decisions" in said[0]


class TestInflationHedge:
    def test_the_year_asked_leads_and_the_hot_years_follow(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        cpi = {2020: 0.014, 2021: 0.07, 2022: 0.065, 2023: 0.034}
        returns = {"XAUUSDT": {2020: 0.25, 2021: -0.04, 2022: 0.0, 2023: 0.13},
                   "BTCUSDT": {2020: 3.0, 2021: 0.6, 2022: -0.64, 2023: 1.55}}
        monkeypatch.setattr(inflation_hedge, "_cpi_by_year", lambda: cpi)
        monkeypatch.setattr(inflation_hedge, "_year_returns", lambda s: returns[s])
        monkeypatch.setattr(inflation_hedge, "FIRST_YEAR", 2020)
        said = inflation_hedge.lines("in 2022 which of gold or bitcoin held its value",
                                     ("XAUUSDT", "BTCUSDT"), today=date(2024, 1, 5))
        assert said is not None
        assert said[0].startswith("Bottom line: in 2022, with US prices up 6.5%")
        assert "gold held its value better and still lost to inflation" in said[0]
        assert "(2021, 2022)" in said[1] and "neither kept its value" in said[1]

    def test_the_question_is_recognised(self) -> None:
        assert inflation_hedge.ASKED.search("gold vs bitcoin as an inflation hedge")
        assert inflation_hedge.ASKED.search("which one protects purchasing power better")
        assert not inflation_hedge.ASKED.search("what did CPI print last month")


class TestClaims:
    def test_halving(self) -> None:
        said = claims.halving_line("Bitcoin's halving just happened this month")
        assert said is not None and "20 Apr 2024" in said and "2028" in said
        assert claims.halving_line("when is the next bitcoin halving") is None

    def test_founder(self) -> None:
        said = claims.founder_line("Compare Solana founder Sam Bankman-Fried's token SOL",
                                   ("SOLUSDT",))
        assert said is not None and "Anatoly Yakovenko" in said
        assert claims.founder_line("Vitalik Buterin founded Ethereum, how is ETH",
                                   ("ETHUSDT",)) is None

    def test_usdt_delisting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import bitget

        rows: list[dict[str, Any]] = [{"symbol": "BTCUSDT", "symbolStatus": "normal"}] * 3
        monkeypatch.setattr(bitget, "public_get", lambda *a, **k: rows)
        said = claims.usdt_line("Since Bitget delisted USDT last week, which stablecoin?")
        assert said is not None and "3 USDT-margined perpetuals" in said

    def test_spot_pair(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import bitget

        monkeypatch.setattr(bitget, "spot_taker_fee",
                            lambda symbol: 0.001 if symbol == "RCOINUSDT" else None)
        said = claims.spot_line("Now that COIN trades as a spot pair directly on Bitget")
        assert said is not None and "rCOIN (RCOINUSDT)" in said

    def test_an_approval_is_said_to_be_unchecked(self) -> None:
        said = claims.unchecked_line("the SEC approved a 50x leveraged BTC ETF yesterday")
        assert said is not None and said.startswith("Not checked:")
        assert claims.unchecked_line("what is the SEC") is None

    def test_the_full_name_of_the_fed_is_a_fed_claim(self) -> None:
        assert server._FED_CLAIM.search("The Federal Reserve cut its benchmark rate to 0% in an "
                                        "emergency meeting")


def test_year_odds_route_counts_every_window(monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.lui.research import dispatch

    moves = [0.5] * 60 + [-0.2] * 40
    monkeypatch.setattr(dispatch, "span_moves", lambda symbol, days: (moves, 1000,
                                                                      date(2023, 10, 1)))
    got = server._round27_follow_up("what are the odds BTC is up a year from now?", [], now=None,
                                    visitor="local", book="")
    assert got is not None and "in 60 of 100 one-year windows" in got["lines"][0]


class TestNewcomerReRun:
    def test_a_liquidation_is_explained_on_the_users_numbers(self) -> None:
        prior = ["i had a btc futures position and it got liquidated overnight, what does that "
                 "even mean, like where did my money go"]
        got = server._round27_follow_up("it was 10x leverage on 100 usdt margin", prior, now=None,
                                        visitor="local", book="")
        assert got is not None
        assert "100 USDT of margin at 10x held a 1,000 USDT position" in got["lines"][0]

    def test_money_left_on_the_exchange_is_a_custody_answer(self) -> None:
        got = server._round27_follow_up(
            "wait i meant is it safe to just leave my money sitting there long term",
            ["is bitget safe for a total beginner"], now=None, visitor="local", book="")
        assert got is not None and "custody" in got["lines"][0]

    def test_buying_fees_are_the_exchanges(self) -> None:
        from argus.lui import newcomer

        said = newcomer.reply("how do fees work when i buy crypto, is it free to just buy some")
        assert said is not None and said.lines[0].startswith("Bottom line: buying is not free")
        free = newcomer.reply("is this site free")
        assert free is not None and "costs nothing" in free.lines[0]


def test_a_stated_funding_rate_is_used_and_held_against_the_cap(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.market import bitget

    monkeypatch.setattr(bitget, "public_get", lambda *a, **k: [
        {"minFundingRate": "-0.003", "maxFundingRate": "0.003"}])
    said = account_math.lines("If the ETHUSDT funding rate is -500% every 8 hours, how much would "
                              "I earn shorting $10,000 of ETH?")
    assert said is not None and "pay about $50,000.00 each settlement" in said[0]
    assert "cannot happen" in said[1] and "$30.00" in said[1]
