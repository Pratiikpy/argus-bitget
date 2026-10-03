"""Round 29's three audits (judge, first-time user, hostile reviewer), run offline: each fix is
checked on the phrasing that exposed it and on the near-miss beside it, and each new figure is
computed from a hand-made input whose answer can be read off it."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest

from argus.lui import concepts, honesty, newcomer, question, server
from argus.lui.research import research_symbols, session, starter, venue_facts
from argus.lui.research.parse import stated_multiple
from argus.market.history import Candle


def _candle(ts: datetime, close: float) -> Candle:
    px = Decimal(str(close))
    return Candle(ts=ts, open=px, high=px, low=px, close=px, volume=Decimal(1))


class TestWordsAndTickers:
    def test_coin_margined_is_not_coinbase(self) -> None:
        assert question.coin_as_ticker("Bitget's BTCUSD inverse coin-margined perpetual") == (
            "Bitget's BTCUSD inverse coin-margined perpetual")
        assert "COINUSDT" not in research_symbols("funding on the coin-margined perp")[0]

    def test_crypto_twitter_is_the_site(self) -> None:
        registry = {"BTCUSDT": object()}
        assert honesty._unlisted_candidate("everyone on crypto twitter says dyor", registry) is None
        assert honesty._unlisted_candidate("should I buy twitter stock", registry) is not None

    def test_a_thousands_separator_in_leverage(self) -> None:
        found = stated_multiple("What would a 10,000x leveraged long on BTCUSDT liquidate at?")
        assert found is not None and found.group(1) == "10000"

    def test_perpetual_contract_is_its_own_name(self) -> None:
        asked = concepts.concept_asked("wait whats a perpetual contract tho")
        assert asked is not None and asked.name == "perpetual contract"


class TestNoFalseReferent:
    @pytest.mark.parametrize("said", ["is crypto trading really that risky tho",
                                      "my coworker made like 2000 bucks in a week is that normal"])
    def test_an_unplaced_question_is_not_told_that_has_nothing_to_refer_to(self, said: str
                                                                           ) -> None:
        got = question.classify(said, now=datetime(2026, 10, 4, tzinfo=UTC))
        assert "nothing to refer to" not in got.reason

    def test_a_question_to_the_desk_still_asks_which(self) -> None:
        got = question.classify("why did you do that?", now=datetime(2026, 10, 4, tzinfo=UTC))
        assert got.intent is question.Intent.AMBIGUOUS


class TestNewcomerRoutes:
    def test_saved_up_is_first_money(self) -> None:
        assert starter.amount_of(
            "im 19, i saved up 150 bucks from my part time job, what should i actually do with "
            "it") == 150

    @pytest.mark.parametrize(("said", "term"), [("what does dyor even stand for", "DYOR"),
                                                ("wtf is ngmi", "NGMI"),
                                                ("whats a rug pull", "rug pull")])
    def test_slang(self, said: str, term: str) -> None:
        lines = newcomer.slang_lines(said)
        assert lines is not None and lines[0].startswith(f"Bottom line: {term}:")

    def test_slang_needs_a_question(self) -> None:
        assert newcomer.slang_lines("gm everyone") is None

    def test_rent_money_and_risk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import dispatch

        monkeypatch.setattr(dispatch, "span_moves",
                            lambda symbol, days: ([-0.26, 0.31, 0.02] if symbol == "BTCUSDT"
                                                  else [-0.35, 0.4, 0.01], 1000,
                                                  date(2023, 10, 1)))
        lines = server._goal_money_lines(
            "that 200 usdt is kind of my rent money for next month, should i even be touching it",
            [], "")
        assert lines is not None
        assert lines[0].startswith("Bottom line: money you need for rent soon is better kept out")
        assert "BTC -26% and ETH -35% — on $200 that is $70" in lines[0]
        assert server._GOAL_MONEY.search("house downpayment in a couple years")
        assert server._GOAL_ASK.search("is trading even a smart idea for that")
        risky = server._how_risky_lines()
        assert risky is not None and "worst week was -26%" in risky[0]

    def test_start_with_reads_the_minimums(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import bitget

        monkeypatch.setattr(bitget, "public_get", lambda path, params: [
            {"minTradeUSDT": "1"} if "spot" in path else {"minTradeUSDT": "5"}])
        lines = server._start_with_lines()
        assert lines is not None and "about $1.00 on spot and $5.00 on the perpetual" in lines[0]
        assert server._START_WITH.search("how much money should i even start with as a beginner")

    def test_leverage_for_a_first_timer(self) -> None:
        said = ("my friend said just use like 20x leverage on a trade so i make more money "
                "faster, is that a good idea for someone who never traded before")
        found = server._LEVERAGE_NEW.search(said)
        assert found is not None and found.group("x") == "20" and server._NEW_TO_IT.search(said)

    def test_what_i_told_you(self) -> None:
        told = server._TOLD_YOU.search("wait hold on, remind me what i just told u about my sol")
        assert told is not None and told.group("x") == "sol"


class TestHostileFixes:
    def test_premises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(server, "_price_now", lambda symbol: 84_900.0)
        said = server._premise_lines(
            "Now that BTC has broken above $150,000 for the first time, should I take profit?",
            None, "")
        assert said == ["Premise check: BTC is 84,900.00 now on Bitget, below the 150,000 the "
                        "question takes as given."]
        two = server._premise_lines("I have 5 BTC long at 10x, which is also 8 BTC long at 10x",
                                    None, "")
        assert two and two[0].startswith("Your question gives two sizes")
        monkeypatch.setattr(server, "_size_step", lambda symbol: "0.0001")
        contracts = server._premise_lines("I opened 10 contracts on BTCUSDT at 10x", None, "")
        assert contracts and '"10 contracts" is read as 10 BTC' in contracts[0]

    def test_no_deal_on_record(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import watchlist

        monkeypatch.setattr(watchlist, "recent_8k", lambda tickers, since: {
            "NVDA": [watchlist.Filed("NVDA", date(2026, 10, 1), "7.01 regulation fd")]})
        line = server._deal_check("NVDAUSDT")
        assert line is not None and "no 8-K reporting a completed acquisition" in line
        assert "1 8-K filed in all" in line
        monkeypatch.setattr(watchlist, "recent_8k", lambda tickers, since: {
            "NVDA": [watchlist.Filed("NVDA", date(2026, 10, 1), "2.01 completion of acquisition")]})
        assert server._deal_check("NVDAUSDT") is None

    def test_stated_prices_are_each_worked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(server, "_price_now", lambda symbol: 84_000.0)
        lines = server._stated_price_lines(
            "With BTC at $85,000, and also BTC at $95,000, what's a 10% drawdown in dollar terms "
            "for 1 BTC?")
        assert lines == ["Bottom line: at $85,000: 1 x $85,000 x 10% = $8,500; at $95,000: 1 x "
                         "$95,000 x 10% = $9,500.",
                         "The question gives 2 different prices for BTC; each is worked above "
                         "rather than one picked.",
                         "At Bitget's live price, 84,000.00: $8,400."]

    def test_funding_thresholds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import bitget

        monkeypatch.setattr(bitget, "public_get", lambda path, params: [
            {"fundingRate": "0.0001", "fundingRateInterval": "8"}])
        month = venue_facts.funding_threshold_lines(
            "If BTC funding stays at its current rate, is the monthly cost of holding a long "
            "higher than 5% a month?")
        assert month is not None and month[0].startswith("Bottom line: below")
        assert "90 settlements a month come to +0.90%" in month[0]
        stated = venue_facts.funding_threshold_lines(
            "BTC funding is running at 87 bps per 8-hour interval. Annualized, is that above or "
            "below 30% a year?")
        assert stated is not None and stated[0].startswith("Premise check: the question's 87 bps")
        assert any("+952.7% a year — above the 30%" in x for x in stated)

    def test_basis_points_is_a_unit(self) -> None:
        from argus.lui.research import desk_answers

        assert desk_answers.perp_vs_spot_lines("that funding rate in basis points", []) is None
        assert server._IN_UNITS.search("what is that same funding rate in basis points")

    def test_no_coin_margined_perpetual(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import bitget

        def listed(path: str, params: dict[str, str]) -> list[dict[str, str]]:
            if path.endswith("contracts"):
                return [] if params["productType"] == "COIN-FUTURES" else [{"symbol": "BTCPERP"}]
            return [{"fundingRate": "0.0001", "fundingRateInterval": "8"}]

        monkeypatch.setattr(bitget, "public_get", listed)
        lines = venue_facts.margin_kind_lines("funding on Bitget's BTCUSD coin-margined perpetual")
        assert lines is not None and "lists no coin-margined (inverse) perpetual" in lines[0]
        assert "BTCPERP (USDC-margined)" in lines[1]


class TestJudgeFixes:
    def test_holiday_by_name_and_the_desk(self) -> None:
        line = session.holiday_line(
            "Is Thanksgiving (26 Nov 2026) a trading session, and does your desk treat it "
            "specially?", datetime(2026, 10, 4, tzinfo=UTC))
        assert line is not None and line.startswith("Bottom line: Thu 26 Nov is a US market "
                                                    "holiday")
        assert "reads the day as HOLIDAY" in line
        assert session.holiday_line("what did NVDA do on 3 Oct",
                                    datetime(2026, 10, 4, tzinfo=UTC)) is None

    def test_macd_shape(self) -> None:
        from argus.lui.research.technicals import _macd_shape

        fading = _macd_shape(0.036, {"bars_since_cross": 13.0, "narrowing_bars": 6.0,
                                     "widening_bars": 0.0})
        assert fading == ("momentum up but fading",
                          "The last MACD cross was bullish (MACD crossed above its signal), 13 "
                          "bars (52h) ago, and the gap has narrowed for 6 bars, so that momentum "
                          "is fading toward the next cross.")

    def test_size_and_name_carried(self) -> None:
        carried = server._carry_size_and_name(
            "Why one-minute children and not Bitget's own TWAP default?",
            ["how should I split a $100k order in NVDA", "now make it $250k NVDA"],
            {"lines": ["Assumed: no size was stated, so $50,000 is worked — give the size",
                       "Assumed: no instrument was named, so NVDA is the worked example"]})
        assert carried is not None and carried[0].endswith("— for $250k of NVDA")

    def test_verify_steps_and_filing_pit(self) -> None:
        assert server._VERIFY_STEPS.search("Walk me through verifying that yourself, step by step")
        assert server._FILING_PIT.search(
            "would that answer have been different if I'd asked the day before the filing "
            "became public?")
        assert any("content_hash" in x for x in server._VERIFY_LINES)

    def test_rtoken_dollars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(server, "_price_now", lambda symbol: 234.78)
        line = server._rtoken_dollars(
            ["Bottom line: … short 1.01 NVDAUSDT (Bitget's NVDA stock perpetual) for every 1",
             "A typical night moved 128 bp unhedged and 4 bp hedged."], "200", "NVDA")
        assert line == ("For your 200 RNVDA (about $46,956 at 234.78): short about 202.00 NVDAUSDT;"
                        " a typical night's swing is about $601 unhedged and $19 hedged.")

    def test_sentiment_accuracy_reads_the_register(self) -> None:
        lines = server._measured_accuracy_lines()
        assert lines is not None and "LOST" in lines[0] and "RoB-RT" in lines[0]
        assert "LOST — LOST" not in lines[0]

    def test_a_short_history_names_its_span(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import performance

        start = datetime(2025, 8, 19, tzinfo=UTC)
        monkeypatch.setattr(performance, "record", lambda symbol, period: _short_record(start))
        period = performance.Period(datetime(2016, 10, 5, tzinfo=UTC),
                                    datetime(2026, 10, 3, tzinfo=UTC),
                                    "over the last 10 years (05 Oct 2016 to 03 Oct)")
        lines = performance.performance_lines(("TSLAUSDT",), period)
        assert lines is not None
        assert "since 19 Aug 2025, as far back as Bitget's daily history goes" in lines[0]
        assert "10 years" not in lines[0]


def _short_record(start: datetime) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(symbol="TSLAUSDT", change=0.108, first=334.9, last=371.1,
                           drawdown=-0.394, peak_day=date(2025, 12, 22),
                           trough_day=date(2026, 7, 29), first_day=start.date(),
                           last_day=date(2026, 10, 3), worst_day=-0.097, best_day=0.076,
                           short_history=True)


def test_mcp_names_a_rescaled_book(monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.lui import mcp_server

    monkeypatch.setattr(mcp_server, "_run", lambda request, label: {"lines": ["Bottom line: x"],
                                                                     "refused": False})
    monkeypatch.setattr(mcp_server, "_answer_text", lambda payload: "Bottom line: x")
    text, _refused = mcp_server.call_tool("argus_portfolio_impact",
                                          {"add": "TSLA", "size_percent": 15,
                                           "book": {"NVDA": 50, "AAPL": 70}})
    assert text.startswith("Note: the book's weights add up to 120%")



def test_an_unread_trending_list_is_said(monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.lui.research import early_signals

    rows = [early_signals.Scored("SANDUSDT", 3e7, 9.0, 0.51, -0.005, False)]
    monkeypatch.setattr(early_signals, "scan", lambda **_kw: rows)
    monkeypatch.setattr(early_signals, "_trending", lambda: None)
    lines = early_signals.lines()
    assert lines is not None and lines[-1].startswith("CoinGecko's trending list did not answer")


class TestLiveReAsk29:
    """What the live re-ask of round 29 in new phrasings still caught."""

    def test_phrasings(self) -> None:
        assert server._HOW_RISKY.search("is day trading crypto actually that dangerous")
        assert server._START_WITH.search("whats the least amount i need to start trading")
        assert server._RTOKEN_FOLLOW.search("would QQQ be better than that?")

    def test_a_bare_slang_follow_up(self) -> None:
        lines = newcomer.slang_lines("and hodl?")
        assert lines is not None and lines[0].startswith("Bottom line: HODL:")
