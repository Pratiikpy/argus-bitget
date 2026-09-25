"""Defects a ten-agent audit of real answers found on 2026-09-25, each pinned by its mechanism.

Ten auditors asked 20+ questions each across ten families (portfolio, stress, quote, execution,
direction, sentiment, fundamentals, the desk's record, other languages, loose phrasing), read every
line, recomputed what they could, and reported 39 findings. These tests keep the fixes fixed.
"""

# ruff: noqa: RUF001 — real Japanese text, full-width punctuation included
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from argus.lui import research
from argus.lui.question import classify, is_order_instruction
from argus.lui.research import ResearchKind

NOW = datetime(2026, 9, 25, 10, tzinfo=UTC)


def _kind(text: str) -> str | None:
    request = research._detect(text)
    return None if request is None else request.kind.value


# --- cash is a holding, never a shock ---

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
    request = research._with_stated_cash(research._detect(text), text)
    assert request is not None and request.shock_pct == shock


def test_an_all_cash_book_is_answered_not_asked_for() -> None:
    request = research._detect("how much VaR do I have at 95% if I hold nothing but stablecoins")
    assert request is not None and request.cash == 1.0
    answer = research.run("q", request)
    assert "no market move to stress" in answer.lines[0]


def test_a_resize_within_a_cash_book_moves_weight_to_cash() -> None:
    from argus.desk.portfolio import resize

    assert resize({"BTCUSDT": 0.5}, "BTCUSDT", 0.35) == {"BTCUSDT": 0.35}


# --- orders and non-orders ---

@pytest.mark.parametrize(("text", "order"), [
    ("open interest on ETH futures", False), ("long short ratio for DOGE", False),
    ("long/short ratio", False), ("short interest on TSLA", False),
    ("can you place a limit order for me", True), ("could you buy 10 NVDA for me", True),
    ("buy 1 BTC now", True), ("short NVDA", True), ("can you explain the risk layer", False),
])
def test_an_order_is_an_instruction_and_a_market_measure_is_not(
        text: str, order: bool) -> None:
    assert is_order_instruction(text) is order


@pytest.mark.parametrize("text", ["what's your win rate", "what's your biggest loss",
                                  "what is the desk's max drawdown"])
def test_a_scored_metric_is_a_performance_question_the_ngram_cannot_relabel(text: str) -> None:
    from argus.lui.ngram import reclassify
    from argus.lui.question import Intent

    question, _ = reclassify(classify(text, now=NOW))
    assert question.intent is Intent.PERFORMANCE


# --- routing to the engine that answers ---

@pytest.mark.parametrize(("text", "kind"), [
    ("is AAPL expensive right now", "fundamentals"),
    ("what's the dividend yield on AAPL", "fundamentals"),
    ("does COIN own bitcoin on its balance sheet", "fundamentals"),
    ("why is the market so bearish today", "sentiment"),
    ("where should I put my stop loss if I long BTC here", "analogue"),
    ("has this setup happened before for BTC", "analogue"),
    ("how volatile has BTC been today, give me the range", "quote"),
    ("how much has Ethereum gone up in the last 24 hours", "quote"),
    ("Wie hoch ist die Finanzierungsrate für BTC?", "quote"),
    ("BTCの資金調達率はいくらですか？", "quote"),
    ("Was ist der aktuelle Bitcoinpreis?", "quote"),
    ("whats xrp trading at", "quote"),
    ("how has BTC done over the last week", "quote"),
    ("how many BTC is $10,000", "quote"),
    ("is the funding rate annualized or per interval", "quote"),
    ("I want to buy $500 of DOGEUSDT, does it matter how I place it?", "execution"),
    ("stress my book", "stress"),
])
def test_the_audits_questions_reach_their_engine(text: str, kind: str) -> None:
    assert _kind(text) == kind


@pytest.mark.parametrize("text", ["¿Dónde estará el precio de Tesla el próximo mes?",
                                  "比特币下个月价格"])
def test_a_price_asked_for_a_future_time_is_still_refused_in_any_language(text: str) -> None:
    assert research._detect(text) is None


def test_ordinary_words_are_not_read_as_lowercase_crypto() -> None:
    assert research.research_symbols("I dot my i")[0] == ()
    assert research.research_symbols("link the page")[0] == ()


def test_a_stop_question_keeps_its_side() -> None:
    request = research._detect("give me a stop loss for a short on ETH at current price")
    assert request is not None and request.side == "short"


def test_the_horizon_of_a_period_question() -> None:
    assert research._period_days("how has BTC done over the last week") == 7
    assert research._period_days("BTC over the past 3 days") == 3
    assert research._period_days("compare to last month") == 30


# --- sizes and the book ---

def test_an_order_size_in_units_is_priced(monkeypatch: pytest.MonkeyPatch) -> None:
    from decimal import Decimal

    from argus.market import bitget

    class T:
        last = Decimal("80000")

    monkeypatch.setattr(bitget, "fetch_tickers", lambda: {"BTCUSDT": T()})
    priced = research._unit_notional("Split a 200 BTC sell order for me", "BTCUSDT")
    assert priced is not None
    notional, note = priced
    assert notional == Decimal("16000000.0") and "200 BTC read as $16,000,000" in note


def test_the_saved_book_is_stated_back() -> None:
    lines = research.saved_book_lines("40% NVDA, 30% cash, 30% AAPL, risk budget 20%")
    assert "40% NVDA, 30% AAPL, with 30% in cash" in lines[0]
    assert "your risk budget is 20%" in lines[0]
    assert "no book is saved" in research.saved_book_lines("")[0]
    assert research.MY_BOOK_QUESTION.search("what's my stated risk budget right now")
    assert not research.MY_BOOK_QUESTION.search("what is my book's beta")


def test_a_holiday_question_is_answered_directly() -> None:
    assert "is not a US market holiday" in (research.holiday_line("is today a US holiday",
                                                                  NOW) or "")
    assert "is a US market holiday" in (research.holiday_line(
        "is thursday a holiday", datetime(2026, 11, 23, 15, tzinfo=UTC)) or "")


def test_the_hurdle_is_explained_from_the_code_that_sets_it() -> None:
    lines, sources = research.hurdle_lines()
    assert "12bps of Bitget taker fees" in lines[0] and "18.8bps" in lines[0]
    assert any("default_hurdle_bps" in s.ref for s in sources)


# --- the fundamentals answer leads with what was asked ---

def test_the_fundamentals_lead_follows_the_question() -> None:
    lines = ["Actionable: AAPL's next report date is not published yet.",
             "Valuation on 2026-09-24: P/E (trailing 12m) 38.1, dividend yield 0.31%.",
             "Institutions: 6,493 holders own 76.6% of the shares (as of 2026-09-23)."]
    assert research._fundamentals_focus(lines, "what's the dividend yield on AAPL", "AAPL")[0] \
        .startswith("Actionable: Valuation on")
    assert research._fundamentals_focus(lines, "who owns AAPL", "AAPL")[0] \
        .startswith("Actionable: Institutions:")
    assert research._fundamentals_focus(lines, "AAPL earnings", "AAPL") == lines
    missing: Any = research._fundamentals_focus(lines[:1], "AAPL dividend", "AAPL")
    assert "holds no dividend figure" in missing[0]
