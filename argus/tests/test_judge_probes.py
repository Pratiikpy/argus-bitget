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
    pattern_reading_wins,
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
        ("who is selling NVDA", ResearchKind.FUNDAMENTALS, "NVDAUSDT"),
        ("are insiders selling TSLA", ResearchKind.FUNDAMENTALS, "TSLAUSDT"),
        ("who is buying AAPL", ResearchKind.FUNDAMENTALS, "AAPLUSDT"),
    ],
)
def test_the_question_reaches_its_engine(question: str, kind: ResearchKind, symbol: str) -> None:
    request = detect(question)
    assert request is not None, question
    assert request.kind is kind
    assert request.symbols[0] == symbol


@pytest.mark.parametrize(
    ("question", "wins"),
    [
        ("who is selling NVDA", True),
        ("summarize NVDA's last earnings call", True),
        ("what is the order book depth on NVDA", True),
        ("did NVDA beat last quarter", False),
        ("what is the sentiment on COIN right now", False),
    ],
)
def test_the_patterns_keep_the_questions_only_one_engine_answers(question: str,
                                                                  wins: bool) -> None:
    # The live console read "who is selling NVDA" as a news question when the model went first.
    assert pattern_reading_wins(detect(question), question) is wins


def test_selling_pressure_on_a_coin_is_not_a_filings_question() -> None:
    request = detect("is there selling pressure on BTC")
    assert request is None or request.kind is not ResearchKind.FUNDAMENTALS


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

    def test_hedge_my_book_is_a_request_for_a_plan(self) -> None:
        """The console never places the hedge; "hedge my book" asks for the plan."""
        request = with_book(detect("hedge my book"), "50% NVDA, 50% AAPL", "hedge my book")
        assert request is not None and request.kind is ResearchKind.HEDGE
        assert set(request.book) == {"NVDAUSDT", "AAPLUSDT"}

    def test_a_saved_crypto_book_replaces_the_bitcoin_default(self) -> None:
        q = "hedge my crypto after a macro shock"
        request = with_book(detect(q), "60% BTC, 40% ETH", q)
        assert request is not None and dict(request.book) == {"BTCUSDT": 0.6, "ETHUSDT": 0.4}
        assert not any("read as bitcoin" in n for n in request.notes)

    def test_an_order_for_a_named_size_is_still_not_research(self) -> None:
        assert detect("buy 10 NVDA now") is None


class TestTheCriticsSecondRound:
    """Questions the win-plan critics found still misrouted after the first fixes."""

    def test_biggest_risk_in_my_portfolio_is_a_book_question(self) -> None:
        q = "what is the biggest risk in my portfolio this week"
        request = with_book(detect(q), "40% NVDA, 30% MSFT, 30% AAPL", q)
        assert request is not None and request.kind is ResearchKind.BOOK
        assert set(request.book) == {"NVDAUSDT", "MSFTUSDT", "AAPLUSDT"}

    def test_an_earnings_call_question_reaches_the_release_reader(self) -> None:
        request = detect("summarize NVDA's last earnings call and what management guided")
        assert request is not None and request.kind is ResearchKind.FUNDAMENTALS
        assert request.symbols == ("NVDAUSDT",)

    def test_who_is_selling_is_a_fundamentals_question(self) -> None:
        request = detect("who is selling NVDA - insiders or institutions?")
        assert request is not None and request.kind is ResearchKind.FUNDAMENTALS

    def test_a_valuation_comparison_keeps_both_names(self) -> None:
        request = detect("is NVDA expensive versus MSFT on valuation")
        assert request is not None and request.symbols == ("NVDAUSDT", "MSFTUSDT")


class TestEarningsReleaseParsing:
    TEXT = ("NVIDIA Announces Financial Results • Revenue of $96.2 billion, up 106% from a year "
            "ago SANTA CLARA — NVIDIA today reported revenue for the second quarter of "
            "$96.2 billion, up 18% from the previous quarter. Outlook NVIDIA's outlook for the "
            "third quarter is as follows: • Revenue is expected to be $108.0 billion, plus or "
            "minus 2%. • Gross margins are expected to be 74.0%, plus or minus 50 basis "
            "points. Highlights Data Center revenue was $89.0 billion.")

    def test_revenue_and_guidance_are_read_from_sentences(self) -> None:
        from argus.market.earnings_release import parse

        got = parse(self.TEXT)
        assert got["revenue"] == pytest.approx(96.2e9)
        assert got["guided_revenue"] == pytest.approx(108.0e9)
        assert got["guided_band_pct"] == 2.0
        assert all(not s.endswith(":") for s in got["outlook"])
        assert any("74.0%" in s for s in got["outlook"])


class TestEventCalendar:
    def test_cpi_dates_are_read_from_the_bls_table(self) -> None:
        from argus.market.calendar import parse_cpi

        text = ("| September 2026 | Oct. 14, 2026 | 08:30 AM |\n"
                "| October 2026 | Nov. 10, 2026 | 08:30 AM |")
        assert [d.isoformat() for d in parse_cpi(text)] == ["2026-10-14", "2026-11-10"]

    def test_fomc_decision_days_are_the_last_day_of_each_meeting(self) -> None:
        from argus.market.calendar import parse_fomc

        page = ("2026 FOMC Meetings <div class='fomc-meeting__month'><strong>October</strong></div>"
                "<div class='fomc-meeting__date'>27-28</div>")
        assert [d.isoformat() for d in parse_fomc(page)] == ["2026-10-28"]


def test_a_hedge_that_is_paid_to_hold_says_so() -> None:
    from argus.lui.research import _hedge_cost_text

    assert "pays about 13.0bps more" in _hedge_cost_text(-13.0)
    assert "about 9.5bps to put on" in _hedge_cost_text(9.5)


def test_one_insider_filing_reads_as_one(monkeypatch: pytest.MonkeyPatch) -> None:
    # "across the 1 most recent Form 4 filings ... 1 open-market sale lines" was on the live
    # console for TSLA (2026-09-24).
    from decimal import Decimal
    from types import SimpleNamespace

    import argus.market.insider as insider
    from argus.lui.research import _ownership_flow

    def trade(code: str, acquired: bool) -> SimpleNamespace:
        return SimpleNamespace(code=code, acquired=acquired, pre_arranged=False,
                               shares=Decimal(100), notional=Decimal(1000), accession="a",
                               accepted_at=datetime(2026, 9, 9, tzinfo=UTC))

    class OneFiling:
        def __init__(self, *_: object, **__: object) -> None: ...

        def trades(self, *_: object, **__: object) -> tuple[list[SimpleNamespace], None]:
            return [trade("S", False), trade("M", True)], None

    monkeypatch.setattr(insider, "InsiderSource", OneFiling)
    lines, _ = _ownership_flow("TSLA", None)
    text = " ".join(lines)
    assert "in the latest Form 4 filing (filed 09 Sep)" in text
    assert "1 open-market sale," in text and "($1,000)" in text
    assert "filings" not in text.split("Insiders, from")[1].split(":")[0]
    assert "which says nothing" in text
