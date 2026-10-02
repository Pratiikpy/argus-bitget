"""Round 18 of the §27 audits (2026-10-01): simultaneous moves on several holdings.

Each test pins one finding (tracker rows 612 onward). Live sources are never reached.
"""

from __future__ import annotations

from argus.lui.research.dispatch import _own_move_lines
from argus.lui.research.kinds import ResearchKind
from argus.lui.research.parse import holding_shocks


def _request(text: str):  # type: ignore[no-untyped-def]
    from argus.lui.research.parse import detect

    request = detect(text)
    assert request is not None
    return request


class TestHoldingShocks:
    def test_two_named_drops_are_read_with_their_signs(self) -> None:
        moves = holding_shocks("what if NVDA drops 20% and TSLA drops 15%?")
        assert moves == {"NVDAUSDT": -20.0, "TSLAUSDT": -15.0}

    def test_a_rise_is_positive(self) -> None:
        moves = holding_shocks("BTC drops 20% and NVDA rises 10%")
        assert moves == {"BTCUSDT": -20.0, "NVDAUSDT": 10.0}

    def test_one_named_move_is_not_a_multi_shock(self) -> None:
        assert holding_shocks("if NVDA drops 10% what happens to my book?") == {}

    def test_plain_text_has_no_shocks(self) -> None:
        assert holding_shocks("how risky is my book") == {}


class TestOwnMoveLines:
    def test_dollar_book_two_moves(self) -> None:
        text = ("I hold $30k of NVDA and $20k of TSLA. What happens if NVDA drops 20% "
                "and TSLA drops 15%?")
        lines = _own_move_lines(text, _request(text), holding_shocks(text))
        assert "18.0%" in lines[0] and "$9,000" in lines[0]
        assert any("no beta" in line for line in lines)

    def test_cash_is_held_flat(self) -> None:
        text = ("I hold 50% BTC, 30% NVDA and 20% cash, about $20,000 total. What happens "
                "if BTC drops 20% and NVDA drops 10%?")
        lines = _own_move_lines(text, _request(text), holding_shocks(text))
        assert "13.0%" in lines[0]
        assert any("20% in cash" in line for line in lines)

    def test_a_name_outside_the_book_does_not_reach_it(self) -> None:
        text = "I hold 60% NVDA and 40% AAPL. What if TSLA drops 20% and AMD drops 10%?"
        lines = _own_move_lines(text, _request(text), holding_shocks(text))
        assert "none of the names" in lines[0]

    def test_the_request_is_a_stress(self) -> None:
        text = "I hold 60% NVDA and 40% AAPL. What if NVDA drops 20% and AAPL drops 10%?"
        assert _request(text).kind is ResearchKind.STRESS


class TestSizingReads:
    def test_a_stated_fifty_percent_leverage_is_the_size(self) -> None:
        request = _request("My book: 100% NVDA. I can't lose more than 10%. "
                           "Should I add 50% leverage on MSTR?")
        assert request.size == 0.5
        assert "MSTRUSDT" in request.symbols

    def test_a_dollar_amount_for_new_money_is_a_candidate_not_a_book(self) -> None:
        request = _request("should I put $500 in Bitcoin")
        assert request.book == {}
        assert str(request.notional) == "500"

    def test_a_named_book_keeps_its_own_notional_out_of_the_request(self) -> None:
        request = _request("I hold 60% NVDA and 40% AAPL, about $20,000. Add $500 of TSLA?")
        assert request.book
        assert request.notional is None


class TestCappedLegSentence:
    @staticmethod
    def _after(limit: str):  # type: ignore[no-untyped-def]
        from argus.lui.memory import Fact, after

        facts = [Fact(kind="max_loss", subject="", value=limit, text="max loss", at=0.0),
                 Fact(kind="capital", subject="", value="50000", text="capital", at=0.0)]
        lines = ["Bottom line: size SOLUSDT so that its worst observed 24 hours (-3.3%) stays "
                 "inside it"]
        return " ".join(after(lines, _request("how big should my SOL short be"), facts))

    def test_a_leg_larger_than_the_account_does_not_claim_an_exact_cost(self) -> None:
        text = self._after("0.15")
        assert "even if this leg were the whole $50,000" in text
        assert "costs exactly that" not in text

    def test_a_binding_limit_still_says_the_leg_costs_exactly_the_allowance(self) -> None:
        text = self._after("0.01")
        assert "costs exactly that" in text


class TestUnreadNames:
    def test_unlisted_name_beside_a_listed_one_is_flagged(self):
        from argus.lui.research.parse import unread_names
        assert unread_names("Compare NVDA and ZZZQX") == ["ZZZQX"]
        assert unread_names("buy NVDA, ZZZQX and AMD") == ["ZZZQX"]

    def test_listed_names_and_acronyms_are_not_flagged(self):
        from argus.lui.research.parse import unread_names
        assert unread_names("Compare NVDA and TSLA") == []
        assert unread_names("What does the CPI and NVDA do") == []
        assert unread_names("NVDA vs BTC") == []


class TestRound19Routing:
    def test_recall_forms(self) -> None:
        from argus.lui.memory import recall_asked

        assert recall_asked("remind me what horizon I said")
        assert recall_asked("what drawdown did I say")
        assert recall_asked("what's my drawdown limit")
        assert not recall_asked("what is bitcoin")

    def test_own_loss_tolerance_is_not_a_performance_question(self) -> None:
        from datetime import UTC, datetime

        from argus.lui.question import Intent, classify

        question = classify("I can't stomach more than 10% drawdown", now=datetime.now(UTC))
        assert question.intent is not Intent.PERFORMANCE

    def test_desk_performance_still_routes_to_performance(self) -> None:
        from datetime import UTC, datetime

        from argus.lui.question import Intent, classify

        question = classify("how much did the desk lose", now=datetime.now(UTC))
        assert question.intent is Intent.PERFORMANCE

    def test_newcomer_loss_and_beginner_safety(self) -> None:
        from argus.lui.newcomer import reply

        loss = reply("what happens if I lose money")
        safe = reply("is crypto safe for a beginner")
        assert loss is not None and "liquidat" in " ".join(loss.lines).lower()
        assert safe is not None

    def test_rivals_by_name_and_all(self) -> None:
        from argus.lui.rivals import asks_about_a_rival, asks_which_rival

        assert asks_about_a_rival("Did the desk beat Bitget's TWAP")
        assert asks_which_rival("is it better than every rival")


class TestRound19Orders:
    def test_an_order_is_not_a_disclosure(self) -> None:
        from datetime import UTC, datetime

        from argus.lui import memory_model

        class Boom:
            def __getattr__(self, name: str) -> object:
                raise AssertionError("the model was asked to read an order")

        now = datetime.now(UTC)
        assert memory_model.extract("buy $500 of BTC", Boom(), now) == []

    def test_a_sized_order_is_priced_not_declined(self) -> None:
        from argus.lui.server import _PRICEABLE_ORDER

        assert _PRICEABLE_ORDER.match("buy $500 of BTC")
        assert _PRICEABLE_ORDER.match("Sell $2000 of NVDA")
        assert not _PRICEABLE_ORDER.match("Buy 0.5 BTC at market")
        assert not _PRICEABLE_ORDER.match("cancel all my orders")
        assert not _PRICEABLE_ORDER.match("buy low sell high")

    def test_the_preview_says_it_is_a_perpetual(self) -> None:
        from argus.lui import agenthub

        line = agenthub.lines_for(agenthub.Order("BTCUSDT", "buy", "0.0059"))[0]
        assert "USDT-perpetual" in line and "spot is a different market" in line

    def test_what_can_i_ask_gives_examples(self) -> None:
        from argus.lui.intro import ASK_Q, ask_answer

        assert ASK_Q.search("what can I ask you")
        assert ASK_Q.search("give me some example questions")
        assert not ASK_Q.search("what is bitcoin")
        assert ask_answer()

    def test_the_unverifiable_multiple_is_not_printed(self) -> None:
        from argus.lui.research import fundamentals

        assert all(label != "EV/EBITDA" for label, _ in fundamentals.VALUATION_MEASURES)


class TestRound19Scenarios:
    def test_hedge_is_names_the_instrument_to_measure(self) -> None:
        from argus.lui.research.parse import hedge_instruments

        assert "XAUUSDT" in hedge_instruments("is gold a good hedge for my NVDA position")

    def test_rate_cut_is_modelled_not_only_described(self) -> None:
        from argus.lui.research.macro import _rate_scenario

        sens = {"pct_per_10bp": -1.4, "corr_10y": -0.6, "days": 61}
        line = _rate_scenario("What would a Fed rate cut do to my NVDA holding?", "NVDA", sens)
        assert line is not None
        assert "-25bp" in line and "+3.5%" in line and "quarter point" in line
        assert "not a forecast" in line
        assert "rates explain about" not in line

    def test_a_loose_fit_says_how_little_rates_explain(self) -> None:
        from argus.lui.research.macro import _rate_scenario

        sens = {"pct_per_10bp": -1.33, "corr_10y": -0.24, "days": 62}
        line = _rate_scenario("What would a Fed rate cut do to my NVDA holding?", "NVDA", sens)
        assert line is not None and "rates explain about 6% of its daily moves" in line

    def test_hike_size_is_read_and_a_plain_yield_question_is_not_a_scenario(self) -> None:
        from argus.lui.research.macro import _rate_scenario

        sens = {"pct_per_10bp": -1.0, "corr_10y": -0.7, "days": 61}
        hike = _rate_scenario("what if the Fed hikes 50bp, what happens to tech", "QQQ", sens)
        assert hike is not None and "+50bp" in hike and "-5.0%" in hike
        assert _rate_scenario("what is the 10-year yield doing", "QQQ", sens) is None
