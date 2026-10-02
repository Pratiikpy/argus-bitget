"""Round 22 audit findings, each pinned to the reader or engine that got it wrong. Offline: every
price is injected."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from argus.lui import memory
from argus.lui.research import parse


class TestScenarioReading:
    def test_a_rate_move_is_not_the_price_shock(self) -> None:
        text = "I have 1.5M USD in SPY. If SPY falls 4% and rates rise 0.5%, what's my loss?"
        shocks = [m.group(1) for m in parse.shock_numbers(text)]
        assert shocks == ["4"]
        assert parse.stated_direction(text, text.index("4%")) == -1
        yield_text = "If QQQ drops 6% and the 10-year yield climbs 40 basis points, what then?"
        assert parse.RATE_MOVE.search(yield_text).group("n") == "40"
        assert parse.stated_direction(yield_text, yield_text.index("6%")) == -1

    def test_a_scenario_loss_is_not_the_remembered_loss_limit(self) -> None:
        assert memory._recall_match("If SPY falls 4%, what's my loss?") is None
        assert memory._recall_match("what's my loss limit?") is not None

    def test_a_short_of_it_is_read(self) -> None:
        assert parse.SHORT_OF_IT.search("Actually I'm short it, not long.")
        assert not parse.SHORT_OF_IT.search("what does a short do to my book?")

    def test_a_return_multiple_is_not_leverage(self) -> None:
        found = parse.stated_multiple("my friend made 3x on 20x lev on sol last month")
        assert found is not None and found.group(1) == "20"
        plain = parse.stated_multiple("I am long SOL at 5x")
        assert plain is not None and plain.group(1) == "5"

    def test_a_price_verb_is_not_a_resize(self) -> None:
        assert not parse._RESIZE_BY.search("If NVDA drops by 12%, how much do I lose?")
        assert parse._RESIZE_BY.search("drop my NVDA position by 12%")

    def test_steps_compound_and_a_self_correction_keeps_the_last_word(self) -> None:
        from argus.lui.server import _restated

        stepped = _restated("If NVDA falls 5%, then rebounds 3%, then falls 2%, where does my "
                            "book end up?", [])
        assert stepped is not None and "moves -4.11%" in stepped[0]
        assert stepped[1] is not None and "0.95 \u00d7 1.03 \u00d7 0.98" in stepped[1]
        fixed = _restated("I'm short NVDA via 5 SQQQ... no wait, I'm long 500 SQQQ. If the Nasdaq "
                          "rallies 4%, what's my P&L?", [])
        assert fixed == ("I'm long 500 SQQQ. If the Nasdaq rallies 4%, what's my P&L?", None)


class TestMoney:
    @pytest.mark.parametrize(("written", "value"), [
        ("40.000,50", 40000.50), ("2.000", 2000.0), ("10 000", 10000.0),
        ("1,234.56", 1234.56), ("12,5", 12.5), ("500", 500.0)])
    def test_money_is_read_as_written(self, written: str, value: float) -> None:
        assert parse.money_number(written)[0] == pytest.approx(value)

    def test_foreign_amounts_are_restated_in_dollars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rates = {"EURUSDUSDT": 1.10, "GBPUSDUSDT": 1.30, "USDJPYUSDT": 150.0}
        monkeypatch.setattr(parse, "_last_price", rates.get)
        text, said = parse.in_us_dollars("I hold ¥5,000,000 of NVDA (yen).")
        assert "$33333.33" in text and "USDJPY" in said[0]
        text, said = parse.in_us_dollars("1,234.56 EUR in ETH and 2.000 USD in BTC")
        assert "$1358.02" in text and "$2000.00" in text and len(said) == 2
        assert parse.asked_currency_amount("what's the loss in pounds?", 1300.0) == "£1,000"

    def test_contracts_and_futures_are_multiplied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(parse, "_last_prices", lambda: {"TSLAUSDT": 400.0})
        monkeypatch.setattr(parse, "_last_price", {"SP500USDT": 7000.0}.get)
        parse._PRICED.clear()
        shorted = parse.priced_book("I'm short 30 TSLA contracts at 100 shares each.")
        assert shorted is not None and shorted.value == pytest.approx(1_200_000)
        assert shorted.weights == {"TSLAUSDT": -1.0}
        futures = parse.priced_book("I am long 2 ES contracts.")
        assert futures is not None and futures.value == pytest.approx(700_000)

    def test_a_short_dollar_leg_is_kept(self) -> None:
        parse._PRICED.clear()
        book = parse.priced_book("short $30,000 QQQ\n$20,000 NVDA")
        assert book is not None and book.weights["QQQUSDT"] == pytest.approx(-0.6)
        stated = parse.priced_book("$40,000 short AAPL")
        assert stated is not None and stated.weights == {"AAPLUSDT": -1.0}


class TestNewcomers:
    def test_how_am_i_doing_reads_the_traders_own_book(self, monkeypatch: pytest.MonkeyPatch
                                                       ) -> None:
        from argus.lui import server

        for asked in ("hows my stuff doing", "am i gonna lose money",
                      "am i up or down on my stuff?"):
            assert server._OWN_LOSS_Q.search(asked), asked
        monkeypatch.setattr(parse, "last_price", {"NVDAUSDT": 240.0}.get)
        monkeypatch.setattr(parse, "_last_prices", lambda: {"NVDAUSDT": 240.0})
        parse._PRICED.clear()
        lines = server._own_pnl("bought 3 nvda at 180 and 0.01 btc")
        assert lines[0].startswith("Bottom line: on what you said you paid, you are up about "
                                   "$180.00 (+33.3%)")
        assert any("No live price could be read for BTC" in x for x in lines)

    def test_follow_ups_the_console_suggested_are_read(self) -> None:
        from argus.lui import concepts, newcomer, server

        found = concepts.concept_asked("wait so what's liquidation exactly")
        assert found is not None and found.name == "liquidation"
        assert server._STOP_MINE.search("ok where do i put mine on tesla")
        reply = newcomer.reply("can u just buy it for me")
        assert reply is not None and reply.lines[0].startswith("Bottom line: no")
        assert server._CREDENTIALS.search("can you place the order for me if i give u my login")
        assert server._IS_THAT_GOOD.search("is that good?")

    def test_hinglish_fear_and_a_first_sum_are_read(self) -> None:
        from argus.lui.server import _restated

        assert _restated("mujhe darr lag raha hai, sab paisa doob jayega kya?", []) == (
            "I'm scared of losing all my money", None)
        assert _restated("bhai 500 dollar hai, kya kharidu? stock ya crypto", []) == (
            "I have $500, should I start with stocks or crypto?", None)

    def test_european_scenario_words_are_read(self) -> None:
        from argus.lui.server import _from_european

        said = _from_european("J'ai 10 000 € en NVDA. Si NVDA baisse de 12 %, combien je "
                              "perds ?")
        assert "if NVDA drops 12%" in said and "how much do I lose" in said

    def test_the_intro_and_a_refused_follow_up_are_honest(self) -> None:
        from argus.lui import intro

        lines, _s, _d = intro.answer()
        assert any("new to trading or not" in x for x in lines)
        said, _s, _d = intro.that_number_answer("ok where do i put mine on tesla",
                                                answered=False)
        assert "was not answered" in said[0]


class TestTraderMemory:
    def test_a_hold_after_and_is_the_horizon(self) -> None:
        found = memory.extract("I'm a swing trader and hold positions about 3 months.")
        horizon = next(f for f in found if f.kind == "horizon")
        assert horizon.value == str(3 * 720) and horizon.text.startswith("I hold")

    def test_a_per_trade_risk_is_not_a_max_loss(self) -> None:
        from argus.lui.memory_model import _valid

        assert _valid("max_loss", "1", "", "I never risk more than 1% of my account per trade") \
            is None

    def test_the_cap_and_a_sale_to_it(self) -> None:
        from argus.lui import server

        cap = memory.Fact(kind="cap", subject="", value="0.35",
                          text="no single position above 35%", at="2026-10-02")
        token = server._MEMORY.set((cap,))
        try:
            lines = server._sell_to_cap("I hold 50% NVDA, 30% TSLA, 20% BTC")
        finally:
            server._MEMORY.reset(token)
        assert lines is not None and "NVDA from 50% to 35% of the book" in lines[0]
        assert "sell 30% of the position" in lines[0]
        assert server._SELL_TO_CAP.search("how much of it should I sell to get under my cap?")
        assert server._RISKIEST_HELD.search("which one of my holdings is riskiest?")

    def test_a_sizing_follow_up_keeps_the_rule_and_takes_the_new_stop(self) -> None:
        from argus.lui.server import _sizing_follow_up

        first = ("I have a $20,000 account and I risk 1% per trade. How big should an NVDA long "
                 "be with a stop at 220?")
        assert _sizing_follow_up("and if I put the stop at 228 instead?", [first]) == (
            first.replace("220", "228"))
        moved = _sizing_follow_up("what about the same trade in TSLA with a 5% stop?", [first])
        assert moved is not None and "a TSLA long" in moved and "with a 5% stop" in moved
        assert "$20,000 account" in moved and "1% per trade" in moved


class TestEngines:
    def test_a_short_is_trimmed_as_a_short(self) -> None:
        from argus.desk.portfolio import resize

        after = resize({"NVDAUSDT": 0.6, "TSLAUSDT": -0.4}, "TSLAUSDT", -0.1)
        assert after["TSLAUSDT"] == pytest.approx(-0.1)
        assert after["NVDAUSDT"] == pytest.approx(0.9)

    def test_the_rebalance_respects_a_cap(self) -> None:
        from argus.lui.research.book import _under_cap

        capped = _under_cap({"NVDA": 0.40, "BTC": 0.32, "TSLA": 0.28}, 0.35)
        assert capped["NVDA"] == pytest.approx(0.35)
        assert sum(capped.values()) == pytest.approx(1.0)
        assert max(capped.values()) <= 0.35 + 1e-9

    def test_this_week_or_next_runs_to_the_end_of_next_week(self) -> None:
        from argus.lui.watchlist import NEW_YORK, window

        _start, end, stated = window("What macro events this week or next could hit my book?",
                                     datetime(2026, 10, 2, 18, tzinfo=UTC))
        assert stated and end.astimezone(NEW_YORK).date() == date(2026, 10, 12)

    def test_margins_are_tested_on_gross_margin(self) -> None:
        from argus.lui import drivers

        def quarter(end: str, value: float) -> drivers.Quarter:
            return drivers.Quarter(end=date.fromisoformat(end), value=value)

        ends = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31",
                "2026-06-30"]
        found = {"ticker": "TSLA", "cik": 1318605, "revenue": [quarter(e, 100.0) for e in ends],
                 "gross_profit": [quarter(e, g) for e, g in zip(
                     ends, (17.0, 17.2, 18.0, 20.1, 19.0, 16.8), strict=True)],
                 "gross_profit_tag": "GrossProfit"}
        result, line, evidence = drivers.test("margins are shrinking", found)
        assert result == "supported" and "16.8%" in line and evidence
        unit, why, _e = drivers.test("deliveries are falling",
                                     {**found, "revenue_tag": "Revenues"})
        assert unit == "not tested" and "not the same measure" in why

    def test_a_review_with_no_habit_keeps_the_loss_size(self) -> None:
        from argus.lui import journal

        def closed(symbol: str, entry: float, exit_: float) -> journal.Trade:
            return journal.Trade(symbol=symbol, side="long", entry=entry, exit=exit_, opened=None,
                                 closed=None, qty=None, lots=1, averaged_down=False,
                                 flags=frozenset())

        trades = [closed("NVDAUSDT", 188.0, 176.0), closed("TSLAUSDT", 300.0, 330.0)]
        kept = journal._loss_size_check(trades)
        assert kept is not None and kept["key"] == "loss_size" and "-6.4%" in kept["check"]

    def test_impossible_shocks_are_refused_on_mcp(self) -> None:
        import json

        from argus.lui import mcp_server

        _status, body = mcp_server.handle_body(json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "argus_stress",
                       "arguments": {"book": {"NVDA": 100}, "shock_percent": -500}}}).encode())
        reply = json.loads(body)["result"]
        assert reply["isError"] and "above -100" in reply["content"][0]["text"]

    def test_every_unlisted_name_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import question

        monkeypatch.setattr(question, "_listed_on_bitget", lambda token: False)
        _found, said = question.extract_symbols(
            "What happens to my XYZQ and FOOBAR positions if they each fall 20%?")
        assert said.startswith("XYZQ and FOOBAR are not listed on Bitget")

    def test_a_polymarket_slug_suffix_is_dropped(self) -> None:
        from argus.market import prediction

        events = [{"markets": [{"question": "Will MicroStrategy announce holding 1M+ BTC by "
                                            "December 31, 2026?-bV81", "active": True,
                                "closed": False, "volumeNum": 50_000,
                                "outcomePrices": "[\"0.3\", \"0.7\"]",
                                "endDate": "2026-12-31T00:00:00Z",
                                "slug": "x"}]}]
        found = prediction.relevant(events, ("MicroStrategy",),
                                    now=datetime(2026, 10, 2, tzinfo=UTC))
        assert found and found[0].question.endswith("2026?")


def test_a_side_correction_keeps_the_amount_said_before() -> None:
    from argus.lui.server import _restated

    said = _restated("Actually I'm short it, not long. If NVDA rises 10%, what's my P&L?",
                     ["I'm long 100 NVDA. What's my risk?"])
    assert said == ("I'm short 100 NVDA. If NVDA rises 10%, what's my P&L?", None)
