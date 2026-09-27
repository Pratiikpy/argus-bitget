"""How a trader's book is read: cash, shorts, resizes, saved books and unlisted names.

Gathered on 2026-09-27 from the three audit-round files (2026-09-25), which grouped tests by
the day a defect was found rather than by what they pin (audit finding 170).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from argus.lui import research
from argus.lui.question import Intent, classify, is_order_instruction
from argus.lui.research import ResearchKind


def _read(text: str) -> research.ResearchRequest:
    request = research.parse.read_request(text)
    assert request is not None, text
    return request


NOON = datetime(2026, 9, 25, 12, tzinfo=UTC)


def _read_3(text: str) -> research.ResearchRequest:
    request = research.detect(text)
    assert request is not None, text
    return request


def test_a_saved_book_keeps_its_cash() -> None:
    book, cash = research.split_cash("50% BTC, 50% cash", research.parse_book("50% BTC, 50% cash"))
    assert book == {"BTCUSDT": 0.5} and cash == 0.5
    request = research.with_book(research.ResearchRequest(kind=ResearchKind.STRESS, symbols=()),
                                 "50% BTC, 50% cash", "what if the nasdaq drops 10%")
    assert request is not None and request.cash == 0.5 and dict(request.book) == {"BTCUSDT": 0.5}
    assert "50% cash" in request.notes[-1]


@pytest.mark.parametrize(("text", "shock"), [
    ("stress test my book: 30% TSLA, 70% cash", None),
    ("what is the expected shortfall at 99% confidence for 60% NVDA 40% AAPL", None),
    ("what happens if QQQ falls 10%? I hold 30% TSLA, 70% cash", -10.0),
])
def test_neither_cash_nor_a_confidence_level_is_read_as_a_shock(
        text: str, shock: float | None) -> None:
    request = research.parse._with_stated_cash(research.parse.read_request(text), text)
    assert request is not None and request.shock_pct == shock


def test_an_all_cash_book_is_answered_not_asked_for() -> None:
    request = research.parse.read_request(
        "how much VaR do I have at 95% if I hold nothing but stablecoins")
    assert request is not None and request.cash == 1.0
    answer = research.run("q", request)
    assert "no market move to stress" in answer.lines[0]


def test_a_resize_within_a_cash_book_moves_weight_to_cash() -> None:
    from argus.desk.portfolio import resize

    assert resize({"BTCUSDT": 0.5}, "BTCUSDT", 0.35) == {"BTCUSDT": 0.35}


def test_the_saved_book_is_stated_back() -> None:
    lines = research.saved_book_lines("40% NVDA, 30% cash, 30% AAPL, risk budget 20%")
    assert "40% NVDA, 30% AAPL, with 30% in cash" in lines[0]
    assert "your risk budget is 20%" in lines[0]
    assert "no book is saved" in research.saved_book_lines("")[0]
    assert research.MY_BOOK_QUESTION.search("what's my stated risk budget right now")
    assert not research.MY_BOOK_QUESTION.search("what is my book's beta")


@pytest.mark.parametrize(("text", "named"), [
    ("should I hedge with gold or with TLT?", ("XAUUSDT", "TLTUSDT")),
    ("Should I hedge my Apple position with oil or with gold?", ("CLUSDT", "XAUUSDT")),
    ("hedge 50% NVDA 50% TSLA with QQQ or SMH", ("QQQUSDT", "SMHUSDT")),
])
def test_named_hedges_are_read_as_candidates(text: str, named: tuple[str, ...]) -> None:
    assert research.hedge_instruments(text) == named


def test_named_hedges_are_not_holdings() -> None:
    request = _read("hedge 50% NVDA 50% TSLA with QQQ or SMH")
    assert request.kind is ResearchKind.HEDGE
    assert set(request.book) == {"NVDAUSDT", "TSLAUSDT"}
    apple = _read("Should I hedge my Apple position with oil or with gold?")
    assert apple.kind is ResearchKind.HEDGE and apple.symbols == ("AAPLUSDT",)


def test_a_saved_book_fills_a_hedge_that_named_only_hedges() -> None:
    planned = research.ResearchRequest(kind=ResearchKind.HEDGE, symbols=("XAUUSDT", "TLTUSDT"),
                                       book={"XAUUSDT": 0.5, "TLTUSDT": 0.5})
    request = research.with_book(planned, "40% NVDA, 30% AAPL, 30% MSFT",
                                 "I have a stock-heavy book, should I hedge with gold or with TLT?")
    assert request is not None and set(request.book) == {"NVDAUSDT", "AAPLUSDT", "MSFTUSDT"}


@pytest.mark.parametrize(("text", "shock", "subject"), [
    ("what does a 10% drop in gold do to my book?", -10.0, "XAUUSDT"),
    ("what if gold drops 10%?", -10.0, "XAUUSDT"),
    ("what if the nasdaq drops 20%? I hold 60% NVDA 40% BTC", -20.0, None),
])
def test_named_shock_size_and_subject(text: str, shock: float, subject: str | None) -> None:
    request = _read(text)
    assert request.kind is ResearchKind.STRESS
    assert request.shock_pct == shock and request.shock_on == subject
    assert research.pattern_reading_wins(request, text)


def test_a_shock_on_a_held_name_after_the_verb() -> None:
    assert research.parse.shock_subject("a 10% drop in gold do to 50% XAU 50% NVDA",
                                   {"XAUUSDT", "NVDAUSDT"}) == "XAUUSDT"


@pytest.mark.parametrize("text", ["what is the sharpe ratio of my book", "what's my max drawdown",
                                  "good take profit for a nvda long from here"])
def test_my_book_and_my_trade_are_not_the_desk(text: str) -> None:
    assert classify(text, now=NOON).intent is not Intent.PERFORMANCE


def test_a_short_is_a_negative_weight() -> None:
    assert research.parse_book("im short 20% TSLA and long 80% NVDA") == {
        "TSLAUSDT": -0.2, "NVDAUSDT": 0.8}
    request = _read_3("im short 20% TSLA and long 80% NVDA, what does that do to my risk")
    assert request.kind is ResearchKind.BOOK and request.book["TSLAUSDT"] == -0.2
    assert any("read as short" in note for note in request.notes)


def test_the_rest_and_all_cash() -> None:
    request = _read_3("how risky is my book, im 70% cash the rest in btc")
    assert request.kind is ResearchKind.BOOK
    assert request.cash == pytest.approx(0.7) and request.book["BTCUSDT"] == pytest.approx(0.3)
    cash = _read_3("im 100% cash rn, whats my risk")
    assert cash.cash == 1.0


def test_an_unlisted_name_in_a_saved_book_is_said() -> None:
    planned = research.ResearchRequest(kind=ResearchKind.BOOK, symbols=())
    request = research.with_book(planned, "20% DOGSHIT, 80% BTC", "how risky is my book")
    assert request is not None and any("DOGSHIT" in note for note in request.notes)
    assert research.unread_holdings("i hold DOGSHIT coin 20% and BTC 80%") == [("DOGSHIT", 20.0)]


@pytest.mark.parametrize("text", [
    "correlation matrix for my book", "concentration check on my big book",
    "whats the risk on my mixed bag of crypto and stocks", "how balanced is my book",
    "beta of my book to spx and ndx", "rebalance this to equal risk pls",
])
def test_book_questions_reach_the_book_engine(text: str) -> None:
    assert _read_3(text).kind is ResearchKind.BOOK


def test_rebalancing_to_equal_risk_is_a_plan_not_an_order() -> None:
    assert not is_order_instruction("rebalance this to equal risk pls")
    assert is_order_instruction("rebalance my book")


def test_the_share_of_book_reads_percent_in_words() -> None:
    assert research.parse._SHARE_OF_BOOK.search("how many shares is 30 percent of 80k in nvda")
