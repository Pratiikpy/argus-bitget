"""Every question a Track 3 judge audit (2026-09-24) found misrouted, pinned to where it now goes.

The audit asked 44 questions of the live console. When the model was not consulted, the
deterministic reader sent these to the wrong engine — usually to an unrelated row of the decision
log, which looks like an answer and is worse than a refusal. Each test below is one of those
questions, or its nearest neighbour, and states the engine it must reach. None needs the network.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from argus.lui.research import (
    EXECUTION_DEFAULT_NOTIONAL,
    EXECUTION_LARGE_NOTIONAL,
    ResearchKind,
    detect,
    follow_up,
    with_book,
)
from argus.lui.server import _language_note
from argus.lui.status_page import Check, render, sweep_lines
from argus.lui.task import price_chart, risk_chart


@pytest.mark.parametrize(
    ("question", "kind", "symbol"),
    [
        ("did NVDA beat last quarter", ResearchKind.FUNDAMENTALS, "NVDAUSDT"),
        ("wut abt nvda earnigns when", ResearchKind.FUNDAMENTALS, "NVDAUSDT"),
        ("英伟达下个季度财报什么时候公布", ResearchKind.FUNDAMENTALS, "NVDAUSDT"),
        ("特斯拉超买了吗", ResearchKind.TECHNICALS, "TSLAUSDT"),
        ("what is the order book depth on NVDA", ResearchKind.EXECUTION, "NVDAUSDT"),
        ("show me the order book for NVDA", ResearchKind.EXECUTION, "NVDAUSDT"),
        ("how liquid is MSTR", ResearchKind.EXECUTION, "MSTRUSDT"),
        ("what is the sentiment on COIN right now", ResearchKind.SENTIMENT, "COINUSDT"),
        ("what does a stronger dollar do to gold", ResearchKind.MACRO, "XAUUSDT"),
    ],
)
def test_the_question_reaches_its_engine(question: str, kind: ResearchKind, symbol: str) -> None:
    request = detect(question)
    assert request is not None, question
    assert request.kind is kind
    assert request.symbols[0] == symbol


def test_a_macro_question_about_a_stated_book_is_asked_of_that_book() -> None:
    request = detect("I hold 40% NVDA, 30% MSFT, 30% GOOGL — what does a Fed rate cut do to my "
                     "book?")
    assert request is not None and request.kind is ResearchKind.MACRO
    assert set(request.book) == {"NVDAUSDT", "MSFTUSDT", "GOOGLUSDT"}


def test_my_crypto_is_read_as_bitcoin_and_said() -> None:
    request = detect("what is happening with the dollar and how does that affect my crypto")
    assert request is not None and request.symbols == ("BTCUSDT",)
    assert any("bitcoin" in n for n in request.notes)


class TestExecutionWithoutASize:
    def test_no_symbol_and_no_size_is_worked_on_a_stated_example(self) -> None:
        request = detect("how should I split a large sell order to minimize slippage")
        assert request is not None and request.kind is ResearchKind.EXECUTION
        assert request.symbols == ("NVDAUSDT",)
        assert request.notional == EXECUTION_LARGE_NOTIONAL
        assert any("worked example" in n for n in request.notes)

    def test_a_named_symbol_without_a_size_gets_the_default_size_said(self) -> None:
        request = detect("how to split an order for BTCUSDT")
        assert request is not None and request.notional == EXECUTION_DEFAULT_NOTIONAL
        assert any("no size was stated" in n for n in request.notes)

    @pytest.mark.parametrize("question", [
        "did my order fill",
        "is my NVDA order filled",
        "What was the realized slippage on the last GOOGL execution the desk ran?",
        "Place a limit order to sell 10 ETH at 5000",
    ])
    def test_orders_that_exist_or_are_instructions_are_not_plans(self, question: str) -> None:
        request = detect(question)
        assert request is None or request.kind is not ResearchKind.EXECUTION

    def test_split_evenly_is_not_an_order(self) -> None:
        request = detect("I've got a bunch of Nvidia and Apple, split evenly. Would throwing some "
                         "TSLA in make it riskier?")
        assert request is not None and request.kind is ResearchKind.IMPACT


class TestFollowUps:
    def test_the_previous_question_is_asked_of_the_new_name(self) -> None:
        request = follow_up("and what about COIN?",
                            ["I hold 50% NVDA, 50% AAPL — what does adding 20% TSLA do to my "
                             "risk?"])
        assert request is not None and request.kind is ResearchKind.IMPACT
        assert request.symbols[0] == "COINUSDT"
        assert dict(request.book) == {"NVDAUSDT": 0.5, "AAPLUSDT": 0.5}

    def test_a_technicals_question_carries_over(self) -> None:
        request = follow_up("what about ETH", ["is BTC overbought"])
        assert request is not None and request.kind is ResearchKind.TECHNICALS
        assert request.symbols == ("ETHUSDT",)

    def test_without_history_or_a_name_there_is_no_follow_up(self) -> None:
        assert follow_up("and what about COIN?", []) is None
        assert follow_up("and what about it?", ["is BTC overbought"]) is None


def test_the_saved_book_wins_over_a_book_the_question_never_stated() -> None:
    """One of three identical calls read the add as 100% of the book (judge audit, defect 1)."""
    request = detect("what does adding 20% TSLA do to my risk")
    assert request is not None
    invented = request.__class__(**{**request.__dict__, "book": {"TSLAUSDT": 1.0}})
    fixed = with_book(invented, "50% NVDA, 50% AAPL", "what does adding 20% TSLA do to my risk")
    assert fixed is not None and set(fixed.book) == {"NVDAUSDT", "AAPLUSDT"}


class TestLanguageNote:
    def test_spanish_is_told_the_answer_is_in_english(self) -> None:
        note = _language_note("¿qué pasa con el oro si la Fed baja las tasas?")
        assert note is not None and note.startswith("Respuesta en inglés")

    def test_chinese_is_told_in_chinese(self) -> None:
        note = _language_note("英伟达财报")
        assert note is not None and "英文" in note

    @pytest.mark.parametrize("question", ["la la land", "el nino and oil", "what is NVDA's price"])
    def test_english_gets_no_note(self, question: str) -> None:
        assert _language_note(question) is None


class TestCharts:
    def test_the_price_chart_states_its_range_in_words(self) -> None:
        points = [(datetime(2026, 9, d, tzinfo=UTC), 100.0 + d) for d in range(1, 21)]
        svg = price_chart(points, "NVDA")
        assert "<svg" in svg and "101.00 to 120.00" in svg

    def test_the_risk_chart_is_drawn_from_contributions(self) -> None:
        report = {"risk_after": {"volatility": 0.01, "contributions": [
            {"symbol": "NVDAUSDT", "weight": 0.5, "contribution": 0.007},
            {"symbol": "AAPLUSDT", "weight": 0.5, "contribution": 0.003}]}}
        svg = risk_chart(report)
        assert "NVDA 50% of the money, 70% of the risk" in svg

    def test_no_data_draws_nothing(self) -> None:
        assert price_chart([], "NVDA") == ""
        assert risk_chart({}) == ""


class TestStatusPage:
    def test_a_surface_that_did_not_answer_is_shown_as_not_answering(self, tmp_path) -> None:
        checks = [Check("Bitget market API", "all tickers", True, "805 contracts", 500.0),
                  Check("bitget-signal", "technical-analysis Skill", False,
                        "did not answer (TimeoutError)", 12000.0)]
        page = render({"entries": 627, "chain_intact": True, "age_hours": 2.0, "stale": False,
                       "next_cycle_at": "2026-09-24T13:30:00+00:00"}, checks, 0.0,
                      sweep_lines(tmp_path), "data:,")
        assert "1 of 2" in page and "no answer" in page and "24 Sep 13:30 UTC" in page

    def test_missing_sweeps_are_simply_absent(self, tmp_path) -> None:
        assert sweep_lines(tmp_path) == []


class TestHedgeQuestionsReachTheHedgePlanner:
    def test_a_stated_book_is_hedged_as_a_book(self) -> None:
        request = detect("I hold 50% NVDA 50% AAPL, how do I hedge it")
        assert request is not None and request.kind is ResearchKind.HEDGE
        assert dict(request.book) == {"NVDAUSDT": 0.5, "AAPLUSDT": 0.5}

    def test_one_name_is_a_book_of_one(self) -> None:
        request = detect("what is the cheapest way to hedge my NVDA position")
        assert request is not None and request.kind is ResearchKind.HEDGE
        assert dict(request.book) == {"NVDAUSDT": 1.0}

    def test_crypto_without_a_name_is_bitcoin_and_said(self) -> None:
        request = detect("how should I hedge my crypto after a macro shock")
        assert request is not None and request.kind is ResearchKind.HEDGE
        assert request.symbols == ("BTCUSDT",)

    def test_the_saved_book_is_hedged_when_none_is_named(self) -> None:
        request = with_book(detect("how should I protect my portfolio"), "60% NVDA, 40% MSFT",
                            "how should I protect my portfolio")
        assert request is not None and request.kind is ResearchKind.HEDGE
        assert set(request.book) == {"NVDAUSDT", "MSFTUSDT"}

    def test_an_instruction_to_hedge_stays_an_order(self) -> None:
        request = detect("hedge my book")
        assert request is None or request.kind is not ResearchKind.HEDGE
