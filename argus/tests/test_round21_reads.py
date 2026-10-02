"""Round 21 audit findings, each pinned to the reader that misread it. Offline."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from argus.lui import memory


class TestNewcomers:
    def test_definitions_answer_the_question_asked(self) -> None:
        from argus.lui import concepts

        stop = concepts.concept_asked("what is a stop loss and do i need one")
        assert stop is not None and stop.name == "stop loss"
        assert (concepts.for_you(stop, "what is a stop loss and do i need one") or "").startswith(
            "For you: with leverage, yes")
        lever = concepts.concept_asked("is 10x leverage ok for small acount")
        assert lever is not None and lever.name == "leverage"
        said = concepts.for_you(lever, "is 10x leverage ok for small acount") or ""
        assert "about 10% against the position" in said and "that one move is the account" in said

    def test_trading_for_me_and_a_nameless_buy_are_answered(self) -> None:
        from argus.lui import intro, newcomer

        reply = newcomer.reply("can u just trade for me")
        assert reply is not None and reply.lines[0].startswith("Bottom line: no")
        assert intro.NO_NAME_BUY_Q.search("whats a good thing to buy rn")

    def test_hinglish_and_follow_ups_are_read(self) -> None:
        from argus.lui import intro
        from argus.lui.server import _from_hinglish

        assert _from_hinglish("bhai solana lena chahiye kya abhi?") == "should I buy solana now?"
        assert _from_hinglish("btc bechna chahiye?") == "should I sell btc now?"
        assert intro.THAT_NUMBER_Q.search("what does that mean lol")
        assert intro.THAT_NUMBER_Q.search("explain that simpler")

    def test_a_wrong_premise_about_the_move_is_said_first(self) -> None:
        from argus.lui.server import _move_premise

        lines = ["TSLA is +4.18% over 24 hours on Bitget."]
        assert (_move_premise("tsla down bad today why", lines) or "").startswith(
            "Bottom line: premise check — TSLA is up 4.18%")
        assert _move_premise("whats up with tsla today", lines) is None


class TestBooksAndShorts:
    def test_name_first_share_counts_are_held(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import parse

        monkeypatch.setattr(parse, "_last_prices", lambda: {
            "NVDAUSDT": 236.0, "TSLAUSDT": 372.0, "BTCUSDT": 85_000.0})
        parse._PRICED.clear()
        book = parse.priced_book("NVDA 5 shares, TSLA 3 shares, 0.01 BTC")
        assert book is not None and set(book.weights) == {"NVDAUSDT", "TSLAUSDT", "BTCUSDT"}
        assert round(book.value) == 5 * 236 + 3 * 372 + 850

    def test_minus_signs_and_share_shorts_keep_their_side(self) -> None:
        from argus.lui.research.parse import _held_short, holding_pairs

        pairs = {s: w for _at, s, w in holding_pairs("NVDA 50%, TSLA -30%, AAPL 20%")}
        assert pairs["TSLAUSDT"] == -0.3 and pairs["NVDAUSDT"] == 0.5
        assert _held_short("I'm short 200 shares of NVDA. If NVDA rallies 12%", "NVDAUSDT")

    def test_a_rule_is_not_a_shock_size(self) -> None:
        from argus.lui.research.parse import detect

        request = detect("What is my book's worst case if the Nasdaq drops 20%, and does that "
                         "break my 30% rule?")
        assert request is not None and request.shock_pct == -20.0

    def test_mcp_accepts_a_short_weight(self) -> None:
        from argus.lui.mcp_server import _book

        book = _book({"NVDA": 50, "TSLA": -50})
        assert book == {"NVDAUSDT": 0.5, "TSLAUSDT": -0.5}


class TestMemoryLimits:
    def test_per_trade_risk_and_a_weight_cap_are_their_own_facts(self) -> None:
        said = memory.extract("I won't risk more than 2% of my account per trade, and my account "
                              "is $50,000. No single position may exceed 30% of my book.")
        kinds = {f.kind: f.value for f in said}
        assert kinds["trade_risk"] == "0.02" and kinds["cap"] == "0.3"
        assert "budget" not in kinds
        held = memory.risk_budget_usd(said)
        assert held is not None and held[0] == 1000

    def test_drawdown_tolerance_in_words(self) -> None:
        said = memory.extract("My max drawdown tolerance is half a percent.")
        assert [(f.kind, f.value) for f in said] == [("max_loss", "0.005")]

    def test_hold_for_about_a_month_is_a_horizon(self) -> None:
        said = memory.extract("I only hold for about a month.")
        assert ("horizon", "720") in [(f.kind, f.value) for f in said]

    def test_a_euro_account_is_converted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import parse

        monkeypatch.setattr(parse, "_last_price", lambda symbol: 1.1)
        capital = memory.Fact(kind="capital", subject="", value="1000",
                              text="My account has €1,000", at="2026-10-02")
        dollars, said = memory.in_dollars(capital)
        assert round(dollars) == 1100 and said.startswith("€1,000 (about $1,100")


class TestTheses:
    def test_fading_momentum_is_a_falling_claim(self) -> None:
        from argus.lui import thesis

        reason = thesis.Reason("momentum is fading", thesis.Kind.MOMENTUM)
        tested = thesis._momentum(reason, {"macd_histogram": 0.45, "rsi": 69}, "NVDA")
        assert tested.result is thesis.Result.CONTRADICTED

    def test_rules_are_profile_not_reasons(self) -> None:
        from argus.lui import thesis

        parts = thesis.profile_parts("I am bullish on SOL because it grows, I won't risk more "
                                     "than 2% of my account per trade, and my account is $50,000. "
                                     "Test my thesis.")
        assert "my account is $50,000" in parts
        assert any(p.startswith("I won't risk more than 2%") for p in parts)
        assert [r.text for r in thesis.reasons(
            "My max drawdown tolerance is 15%, horizon 6 months, and my thesis is AI capex keeps "
            "accelerating.")] == ["AI capex keeps accelerating"]

    def test_a_case_with_no_reasons_gets_the_standard_ones(self) -> None:
        from argus.lui import thesis_answer

        built = thesis_answer.standard_case("Test my thesis: bearish case for NVDA with reasons")
        assert built is not None and built[1] == "bear"
        assert built[0].startswith("I am bearish on NVDA because valuations look stretched")

    def test_a_risk_question_after_a_thesis_is_answered_too(self) -> None:
        from argus.lui import thesis_answer

        _claim, also = thesis_answer.split_other_question(
            "My thesis is AI capex keeps accelerating. What is my biggest risk right now?")
        assert also == "What is my biggest risk right now?"


class TestNumbersAndScenarios:
    def test_a_stated_funding_rate_is_used(self) -> None:
        from argus.lui.server import _stated_funding

        lines = _stated_funding("If funding on MSTR is 50bps per 8h, what does it cost me to "
                                "short $10,000 for a day?") or []
        assert lines and "earns about $150 a day" in lines[0]

    def test_options_beside_a_stock_are_priced_at_expiry(self,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import server

        monkeypatch.setattr(server, "_price_now", lambda symbol: 400.0)
        lines = server._stock_with_options(
            "I am long 1000 TSLA and 10 TSLA puts, strike 330, paid $4.10. If TSLA rallies 15%, "
            "what is my net?") or []
        assert lines and "+$55,900" in lines[0] and "-$4,100" in lines[0]

    def test_a_levered_move_is_one_figure(self) -> None:
        from argus.lui.multistep import _levered_move

        said = _levered_move("I'm 3x leveraged long $30k of TSLA perp. What happens if TSLA falls "
                             "25%?") or ""
        assert "-75% on the margin" in said and "$7,500" in said and "$10,000 of margin" in said

    def test_next_month_is_the_calendar_month(self) -> None:
        from argus.lui.watchlist import window

        start, end, stated = window("what if CPI comes in hot next month",
                                    datetime(2026, 10, 2, 12, tzinfo=UTC))
        assert stated and start.date().isoformat() == "2026-11-01"
        assert end.date().isoformat() == "2026-12-01"

    def test_a_planned_position_is_set_beside_the_rule(self) -> None:
        from argus.lui.research import sizing

        lines, _s, _d = sizing.answer("If I put $20,000 into MSFT with a stop at 5% below entry, "
                                      "how much should I buy if I risk 1% of a $200,000 account?",
                                      None)
        assert lines[1].startswith("Your $20,000 would lose about $1,024")

    def test_plural_and_stale_record(self) -> None:
        from argus.lui.agent_page import _staleness
        from argus.lui.answer import plural

        assert plural(1, "day") == "1 day" and plural(3, "day") == "3 days"
        old = {"generated_at": "2026-10-02T11:02:22Z"}
        assert "hours old" in _staleness(old, datetime(2026, 10, 2, 17, tzinfo=UTC))
        assert _staleness(old, datetime(2026, 10, 2, 12, tzinfo=UTC)) == ""


def test_which_fits_my_thesis_names_the_thesis_name() -> None:
    from argus.lui import server

    facts = memory.extract("I'm bullish on NVDA because AI capex keeps rising.")
    token = server._MEMORY.set(tuple(facts))
    try:
        said = server._fits_thesis(
            "Compare NVDA and AMD on valuation — which fits my thesis?",
            ["Bottom line: AMD is the more expensive on 2 of 3 measures"]) or ""
    finally:
        server._MEMORY.reset(token)
    assert said.startswith("Your thesis is on NVDA") and "NVDA is the cheaper" in said
