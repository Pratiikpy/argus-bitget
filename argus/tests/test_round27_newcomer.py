"""Round 27's first-time-user findings, run offline: each route is checked to fire on the ask a
newcomer actually typed and to stay quiet on the near-miss beside it, and each new answer is
computed from a small hand-made series whose result can be read off it."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, ClassVar

import pytest

from argus.lui import macro_thesis, newcomer, server, thesis_answer, watchlist
from argus.lui import memory as mem
from argus.lui.research import desk_answers, desk_followups, dispatch, early_signals, starter
from argus.market.history import Candle


def _candle(ts: datetime, close: float, volume: float = 1.0) -> Candle:
    px = Decimal(str(close))
    return Candle(ts=ts, open=px, high=px, low=px, close=px, volume=Decimal(str(volume)))


class TestFollowUpsAreReadAsTheirOwnQuestion:
    @pytest.mark.parametrize("said", ["what if 5x", "and at 3x?", "10x instead", "ok and 2x",
                                      "with 3x leverage"])
    def test_a_bare_new_multiple(self, said: str) -> None:
        assert server._LEVERAGE_AGAIN.match(said)

    @pytest.mark.parametrize("said", ["whts the best coin to 10x", "is 5x safe on btc",
                                      "what if btc falls 5%"])
    def test_not_a_bare_multiple(self, said: str) -> None:
        assert not server._LEVERAGE_AGAIN.match(said)

    def test_a_coin_that_will_multiple_is_not_leverage(self) -> None:
        assert server._MOONSHOT.search("whts the best coin to 10x")
        assert server._MOONSHOT.search("which memecoin will 100x")
        assert server._MOONSHOT.search("any 10x gems?")
        assert not server._MOONSHOT.search("what happens at 10x leverage on BTC")

    def test_holding_through_a_report(self) -> None:
        assert server._HOLD_THROUGH.search("so should i hold thru earnings or not")
        assert server._HOLD_THROUGH.search("is it smart holding NVDA through the report")
        assert not server._HOLD_THROUGH.search("how long should i hold NVDA")

    def test_a_name_alone(self) -> None:
        assert server._BARE_NAME.match("pepe?")
        assert server._BARE_NAME.match("  $SOL ")
        assert not server._BARE_NAME.match("pepe price now")

    def test_fees_pointer_and_vague_pointer(self) -> None:
        assert server._FEES_AGAIN.match("and the fees?")
        assert server._FEES_AGAIN.match("ok what about fees on that")
        assert not server._FEES_AGAIN.match("what are the fees on a $500 BTC order")
        assert server._VAGUE_AGAIN.match("and that?")
        assert not server._VAGUE_AGAIN.match("and that coin, is it risky?")

    def test_reaction_last_times_and_routine(self) -> None:
        assert server._REACT_LAST.search("how did it react last 4 times")
        assert not server._REACT_LAST.search("how did the desk do last week")
        assert server._ROUTINE.search("ok give me a weekly routine")
        assert server._SURPRISE.search("tell me something i dont know")
        assert server._CONCENTRATED.search("am i too concentrated?")
        assert server._FASTER.search("what if i want it faster")

    def test_mechanism_how_and_long_term(self) -> None:
        assert server._MEANS_FOR.search("what does CPI mean for crypto")
        assert server._MEANS_FOR.search("how does inflation affect bitcoin")
        assert not server._MEANS_FOR.search("what did CPI print last month")
        assert server._HOW_IS.match("hows the s&p")
        assert server._HOW_IS.match("how is nvda looking today")
        assert not server._HOW_IS.match("how is my book doing against the S&P this year")
        assert server._LONG_TERM.search("btc vs eth which better long term")
        assert server._LONG_TERM.search("which is better for the long run, gold or btc")

    def test_typos_mended(self) -> None:
        mended = server._TYPOS.sub(lambda m: server._TYPO_FIX[m.group(0).lower()],
                                   "btc vs eht which better long term")
        assert mended == "btc vs eth which better long term"


class TestFeeArithmetic:
    def test_ten_round_trips_a_day_on_500(self) -> None:
        lines = desk_answers.fee_burn_lines(
            "if i trade 10 times a day with $500 how much do fees eat")
        assert lines is not None
        lead = lines[0]
        assert "$180 a month on perpetuals, 36% of the $500" in lead
        assert "$0.60 (0.12% taker round trip)" in lead and "$6.00 a day" in lead
        assert "$300 a month (0.20% a round trip)" in lead

    def test_weekly_count_and_no_count(self) -> None:
        weekly = desk_answers.fee_burn_lines("fees if I do 7 trades a week on $1,000?")
        assert weekly is not None and "$36.00 a month" in weekly[0]
        assert desk_answers.fee_burn_lines("what are bitget's fees") is None
        assert desk_answers.fee_burn_lines("I trade 10 times a day, any tips?") is None

    def test_charge_reaches_the_schedule(self) -> None:
        lines = desk_answers.fees_lines("how much does bitget charge per trade")
        assert lines is not None and "0.10% a side on spot" in lines[0]


class TestMarginOnALeveragedTrade:
    @pytest.mark.parametrize(("said", "margin"), [
        ("i wanna use 20x leverage on btc with 100 usdt", "100"),
        ("10x long ETH with $250 of margin", "250"),
    ])
    def test_reads_the_margin(self, said: str, margin: str) -> None:
        found = dispatch.MARGIN_WITH.search(said)
        assert found is not None and found.group("m") == margin

    @pytest.mark.parametrize("said", ["10x long with 2.5% stop", "short it with 20x",
                                      "with 3 times leverage"])
    def test_not_a_margin(self, said: str) -> None:
        assert dispatch.MARGIN_WITH.search(said) is None


class TestFundingBacktestFollowUps:
    @pytest.fixture
    def tested(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import crossasset_feed, history

        start = datetime(2026, 7, 1, tzinfo=UTC)
        hours = [start + timedelta(hours=i) for i in range(24 * 100)]
        # BTC climbs 0.01% an hour, so every 24-hour window is up 0.24% before costs
        closes = {h: 100 * (1.0001 ** i) for i, h in enumerate(hours)}
        settlements = [(int((start + timedelta(hours=8 * i)).timestamp() * 1000),
                        -0.0001 if i % 6 in (0, 1) else 0.0001) for i in range(270)]
        monkeypatch.setattr(crossasset_feed, "fetch_funding", lambda symbol: settlements)
        monkeypatch.setattr(history, "fetch_range", lambda symbol, days, interval: [
            _candle(h, c) for h, c in closes.items()])

    def test_fees_are_netted(self, tested: None) -> None:
        prior = ["backtest buying BTC when funding is negative"]
        lines = desk_followups.funding_rule_lines("does that include fees", prior)
        assert lines is not None and lines[0].startswith("Bottom line: no")
        assert "from +0.24% to +0.12%" in lines[0]

    def test_count_names_the_separate_stretches(self, tested: None) -> None:
        prior = ["backtest buying BTC when funding is negative"]
        lines = desk_followups.funding_rule_lines("how many trades was that", prior)
        assert lines is not None
        # two settlements in a row negative every 48 hours: one stretch each
        assert "90 negative funding settlements" in lines[0]
        assert "about 45 separate stretches" in lines[0]

    def test_luck_is_the_interval_against_the_base(self, tested: None) -> None:
        prior = ["backtest buying BTC when funding is negative"]
        lines = desk_followups.funding_rule_lines("is that statistically real or luck", prior)
        assert lines is not None
        # every window is up after negative and after any settlement alike: 100% inside 96%-100%
        assert "inside that range" in lines[0] and "cannot be told from luck" in lines[0]


class TestRealYieldPremise:
    found: ClassVar[dict[str, Any]] = {
        "real": {"then": 2.19, "now": 2.88, "from": "2026-06-05", "to": "2026-10-01"},
        "rates": {"n": 62, "slope_per_10bp": -0.0016, "corr": -0.05, "t": -0.4,
                  "from": "2026-07-06", "to": "2026-10-01", "ten_year": 5.24}}

    def test_falling_real_yields_contradicted_by_a_rise(self) -> None:
        result, line, _ = macro_thesis.test("real yields are falling", self.found, "XAU")
        assert result == "contradicted"
        assert "2.19%" in line and "2.88%" in line and "+69bp" in line and "false so far" in line

    def test_rising_real_yields_supported(self) -> None:
        result, line, _ = macro_thesis.test("real rates keep rising", self.found, "XAU")
        assert result == "supported" and "as the reason says" in line

    def test_flat_is_not_measurable(self) -> None:
        flat = {**self.found, "real": {**self.found["real"], "now": 2.22}}
        result, _line, _ = macro_thesis.test("real yields are falling", flat, "XAU")
        assert result == "not measurable"

    def test_price_calls_are_broken_by_the_price(self) -> None:
        assert thesis_answer._PRICE_CALL.search("nvidia is gonna keep ripping")
        assert not thesis_answer._PRICE_CALL.search("AI datacenter spend")


class TestNewcomerShortVersion:
    lines: ClassVar[list[str]] = [
        "Bottom line: this console makes no buy or sell call — BTC is +4.7% over 30 days.",
        "Terms: beta = how far it moves for each 1% move in the index.",
        "BTC moves 0.82x the Nasdaq-100 (QQQ) while US markets are open.",
        "BTC's worst 24 hours in the last 30 days (-4.21%) would cost about $421.",
        "Return shape over 719 hourly bars: excess kurtosis 10.0, skew +0.65.",
        "Sources reached: 2 of 2 answered.",
        "Without leverage the most you can lose is what you put in.",
        "Data: live Bitget hourly candles."]

    def test_the_lead_and_two_plain_lines_stay(self) -> None:
        cut = server._newcomer_cut(self.lines)
        assert cut[0] == self.lines[0]
        assert cut[1].startswith("BTC's worst 24 hours")
        assert cut[2].startswith("Without leverage")
        assert "show the full analysis" in cut[3] and cut[4].startswith("Data:")
        assert not any("kurtosis" in x or "Nasdaq-100" in x for x in cut)

    def test_who_said_they_are_new_and_the_way_back(self) -> None:
        assert server._NEWCOMER_SAID.search("im a noob, should i buy btc rn")
        assert server._NEWCOMER_SAID.search("I'm new to crypto")
        assert not server._NEWCOMER_SAID.search("what is the new listing this week")
        assert server._FULL_ASKED.match("show the full analysis")
        assert server._FULL_ASKED.match("more detail")
        assert not server._FULL_ASKED.match("show more about NVDA earnings")


class TestFirstMoneyAndMemory:
    def test_grow_my_sum_is_a_first_money_question(self) -> None:
        assert starter.amount_of("i want to grow my $300 slowly, i'm 22") == 300.0
        assert starter.amount_of("i want to grow my portfolio") is None

    def test_the_sum_is_kept_and_recalled_as_the_budget(self) -> None:
        facts = mem.extract("i want to grow my $300 slowly, i'm 22",
                            now=datetime(2026, 10, 3, tzinfo=UTC))
        capital = [f for f in facts if f.kind == "capital"]
        assert capital and capital[0].value == "300"
        said = mem.recall_one("remind me what i said my budget was", facts)
        assert said is not None and "grow my $300" in said[0]

    def test_a_sum_of_a_name_is_not_capital(self) -> None:
        facts = mem.extract("I have $300 of ETH", now=datetime(2026, 10, 3, tzinfo=UTC))
        assert not [f for f in facts if f.kind == "capital" and f.value == "300"]

    def test_routine_names_its_steps(self) -> None:
        lines = server._routine_lines(300.0)
        assert "for $300" in lines[0]
        assert any(x.startswith("Monday:") for x in lines)

    def test_faster_is_a_bigger_bet(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import date

        monkeypatch.setattr(dispatch, "span_moves",
                            lambda symbol, days: ([-0.14, 0.02, -0.03], 1000, date(2023, 9, 1)))
        lines = server._faster_lines(300.0)
        assert "-14.0%" in lines[1] and "-$42 unleveraged" in lines[1] and "-$126 at 3x" in lines[1]
        assert "$108 a month" in lines[2]

    def test_trust_points_at_the_pages_that_check_it(self) -> None:
        answered = newcomer.reply("should i trust you")
        assert answered is not None
        assert answered.lines[0].startswith("Bottom line: trust it only as far as you can check")
        assert "/wrong" in answered.lines[1] and "/proof" in answered.lines[1]


class TestBookAndMarkets:
    def test_concentration_says_yes_with_its_yardstick(self, monkeypatch: pytest.MonkeyPatch
                                                       ) -> None:
        from argus.market import history

        start = datetime(2026, 9, 1, tzinfo=UTC)

        def series(symbol: str, days: int, interval: str) -> list[Candle]:
            step = 1.002 if symbol.startswith("BTC") else 1.003
            return [_candle(start + timedelta(hours=i), 100 * step ** (i % 7)) for i in range(200)]

        monkeypatch.setattr(history, "fetch_range", series)
        lines = server._concentration_lines("BTC 60%, ETH 30%, NVDA 10%")
        assert lines is not None
        assert lines[0].startswith("Bottom line: yes — BTC is 60% of the book by value")
        assert "as evenly as 2.2 equal-sized ones" in lines[0]
        assert "Yardstick used here" in lines[1]

    def test_long_term_is_three_years_and_no_pick(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import history

        start = datetime(2023, 10, 1, tzinfo=UTC)

        def window(symbol: str, **_kw: Any) -> list[Candle]:
            path = ([100, 200, 100, 150] if symbol.startswith("BTC") else [100, 50, 80, 120])
            return [_candle(start + timedelta(days=i), path[i * len(path) // 400])
                    for i in range(400)]

        monkeypatch.setattr(history, "fetch_window", window)
        lines = server._long_term_lines(["BTCUSDT", "ETHUSDT"])
        assert lines is not None
        assert "no measurement can say which is better" in lines[0]
        assert "BTC +50%, its deepest fall from a peak -50%, now -25% from its high" in lines[0]
        assert "ETH +20%, its deepest fall from a peak -50%, now 0% from its high" in lines[0]

    def test_weekend_trading_from_the_contract_candles(self, monkeypatch: pytest.MonkeyPatch
                                                       ) -> None:
        from argus.market import history

        start = datetime(2026, 9, 25, tzinfo=UTC)  # a Friday
        candles = [_candle(start + timedelta(hours=i), 400,
                           volume=10 if (start + timedelta(hours=i)).weekday() >= 5 else 100)
                   for i in range(24 * 8)]
        monkeypatch.setattr(history, "fetch_range", lambda symbol, days, interval: candles)
        lines = desk_followups.weekend_lines("can i trade it on weekends",
                                             ["whats TSLA doing today"])
        assert lines is not None
        assert lines[0].startswith("Bottom line: yes — the TSLA perpetual on Bitget traded in "
                                   "48 of the 48 weekend hours")
        assert "about 10% of a weekday hour" in lines[0]
        assert "US market is shut" in lines[1]
        assert desk_followups.weekend_lines("what did TSLA do last weekend", []) is None

    def test_anything_big_this_week_is_the_week(self) -> None:
        assert watchlist.asks_for_watchlist("anything big this week")

    def test_other_scripts_are_counted_not_listed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [early_signals.Scored("龙虾USDT", 3e7, 2.4, -0.55, 0.0005, False),
                early_signals.Scored("SANDUSDT", 3e7, 9.0, 0.51, -0.005, True)]
        monkeypatch.setattr(early_signals, "scan", lambda **_kw: rows)
        lines = early_signals.lines(now=datetime(2026, 10, 3, tzinfo=UTC))
        assert lines is not None and "SAND scores highest" in lines[0]
        assert not any("龙虾" in x for x in lines)
        assert any("1 contract named in another script is left off" in x for x in lines)
        assert early_signals.TODAY_ASKED.search("what should i look at today?")


class TestCriticalOpeners:
    """Findings 1-3: openers kept as notes, a view-and-reason never tested, and the desk's own
    universe answering a question about a coin nobody asked the desk about."""

    @pytest.mark.parametrize(("said", "amount"), [
        ("i only got like 200 bucks where do i even start", 200.0),
        ("im 22, student, can lose maybe $300 total, want to grow it slowly. make me a plan",
         300.0),
    ])
    def test_first_money_openers(self, said: str, amount: float) -> None:
        assert starter.amount_of(said) == amount

    def test_a_view_with_its_reason_is_a_thesis(self) -> None:
        said = "i think nvidia is gonna keep ripping bc of AI datacenter spend, am i right"
        assert server._as_thesis(said) == ("Thesis: i think nvidia is gonna keep ripping because "
                                           "of AI datacenter spend. Test it.")
        assert server._as_thesis("i think nvidia is great, right?") == (
            "i think nvidia is great, right?")

    def test_enough_to_buy_reads_bitgets_minimums(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import bitget

        def listed(path: str, params: dict[str, str]) -> list[dict[str, str]]:
            if "spot" in path:
                return [{"minTradeUSDT": "1"}]
            return [{"minTradeUSDT": "5", "minTradeNum": "0.0001"}]

        monkeypatch.setattr(bitget, "public_get", listed)
        lines = desk_followups.enough_to_buy_lines("is that enough to buy bitcoin?",
                                                   ["i only got like 200 bucks"])
        assert lines is not None
        assert lines[0].startswith("Bottom line: yes — $200 is enough to buy BTC on Bitget")
        assert "spot from $1.00" in lines[0] and "perpetual from $5.00" in lines[0]
        small = desk_followups.enough_to_buy_lines("is $0.50 enough to buy bitcoin", [])
        assert small is not None and small[0].startswith("Bottom line: not yet — $0.50 is below")


class TestPagesWithoutJavaScript:
    def test_the_suggestions_are_in_the_served_html(self) -> None:
        for group, asks in server.HOME_CHIPS:
            assert f"__CHIPS_{group}__" not in server.PAGE
            for ask in asks:
                assert ask.replace("&", "&amp;") in server.PAGE
        assert server.PAGE.count('class="chip"') == sum(len(a) for _g, a in server.HOME_CHIPS)

    def test_the_book_is_set_against_what_was_paid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        prices = {"BTCUSDT": 84_640.0, "ETHUSDT": 2_678.4}
        monkeypatch.setattr(server, "_price_now", lambda symbol: prices[symbol])
        line = server._cost_basis_line("0.05 BTC bought at 92000, 1.2 ETH at 3100, 300 USDT")
        assert line == ("Against what you paid: BTC -8.0% (-$368) from 92,000 to 84,640; "
                        "ETH -13.6% (-$506) from 3,100 to 2,678.4 (Bitget last price, before "
                        "fees).")
        assert server._cost_basis_line("40% NVDA, 60% BTC") is None
        assert server._ABOUT_MY_BOOK.search("how risky is my book")


class TestLiveReAskFindings:
    """What the live re-ask of round 27 in new phrasings still caught."""

    def test_the_wall_as_a_refusal_is_replaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        wall = ("ETH trades on Bitget (ETHUSDT) but is not one of the twelve stock perpetuals "
                "the desk decides on, so there is no decision about it on the record")
        asked: list[str] = []

        def answered(text: str, prior: list[str], **_kw: Any) -> dict[str, Any]:
            asked.append(text)
            if "newbie" in text:
                return {"lines": [wall], "sources": [], "data": {}, "refused": True,
                        "reason": wall, "classified_by": "patterns"}
            return {"lines": ["Bottom line: holding ETH has meant this."], "sources": [],
                    "data": {}, "refused": False, "reason": "", "classified_by": "research"}

        monkeypatch.setattr(server, "_answer", answered)
        got = server.handle_ask("total newbie here, should i get some eth now", [])
        assert got["lines"][0] == "Bottom line: holding ETH has meant this."
        assert asked[-1] == "should I buy ETH" and not got["refused"]

    def test_what_did_i_say_reads_memory(self) -> None:
        facts = mem.extract("want to build up my $500 slowly",
                            now=datetime(2026, 10, 3, tzinfo=UTC))
        said = mem.recall_one("what did i say my budget was", facts)
        assert said is not None and "$500" in said[0]

    def test_a_daily_routine_says_daily(self) -> None:
        assert server._routine_lines(None, daily=True)[0].startswith(
            "Bottom line: a daily five-minute check")
        assert server._routine_lines(None)[0].startswith("Bottom line: a weekly routine")
