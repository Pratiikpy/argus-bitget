"""Which engine a question reaches, and the details read from its words.

Gathered on 2026-09-27 from the three audit-round files (2026-09-25), which grouped tests by
the day a defect was found rather than by what they pin (audit finding 170).
"""

# ruff: noqa: RUF001 — real Japanese and Chinese text, full-width punctuation included
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from argus.lui import research
from argus.lui.question import Intent, classify, coin_as_ticker, is_order_instruction
from argus.lui.research import ResearchKind

MORNING = datetime(2026, 9, 25, 10, tzinfo=UTC)


def _kind(text: str) -> str | None:
    request = research.parse.read_request(text)
    return None if request is None else request.kind.value


def _read(text: str) -> research.ResearchRequest:
    request = research.parse.read_request(text)
    assert request is not None, text
    return request


NOON = datetime(2026, 9, 25, 12, tzinfo=UTC)


def _read_3(text: str) -> research.ResearchRequest:
    request = research.detect(text)
    assert request is not None, text
    return request


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

    question, _ = reclassify(classify(text, now=MORNING))
    assert question.intent is Intent.PERFORMANCE


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
    assert research.parse.read_request(text) is None


def test_ordinary_words_are_not_read_as_lowercase_crypto() -> None:
    assert research.research_symbols("I dot my i")[0] == ()
    assert research.research_symbols("link the page")[0] == ()


def test_a_stop_question_keeps_its_side() -> None:
    request = research.parse.read_request("give me a stop loss for a short on ETH at current price")
    assert request is not None and request.side == "short"


def test_the_horizon_of_a_period_question() -> None:
    assert research.parse._period_days("how has BTC done over the last week") == 7
    assert research.parse._period_days("BTC over the past 3 days") == 3
    assert research.parse._period_days("compare to last month") == 30


@pytest.mark.parametrize("text", [
    "what is the long/short ratio on SOL?", "are more traders long than short on XRP?",
    "SOL long short ratio",
])
def test_long_short_questions_are_recognised(text: str) -> None:
    assert research.LONG_SHORT_QUESTION.search(text)


@pytest.mark.parametrize(("text", "hours"), [
    ("funding cost to hold BTC long a week", 168),
    ("cost of holding ETH short for 3 days", 72),
    ("what does it cost to hold NVDA overnight?", None),
])
def test_holding_cost_is_a_quote_with_its_period(text: str, hours: int | None) -> None:
    assert research.hold_cost_question(text)
    request = _read(text)
    assert request.kind is ResearchKind.QUOTE
    found = research.parse._HOLD_PERIOD.search(text)
    assert found is not None
    if hours is not None:
        amount = float(found.group(1) or 1)
        unit = found.group(2).lower()
        assert amount * (1 if unit[0] == "h" else 24 if unit[0] == "d" else 168) == hours


def test_should_i_hold_is_the_odds_engine() -> None:
    request = _read("should I hold NVDA for a week?")
    assert request.kind is ResearchKind.ANALOGUE and request.horizon_hours == 168


@pytest.mark.parametrize("text", [
    "what is the 200-day moving average of NVDA?", "NVDA daily RSI", "is BTC above its 50 day ema?",
    "is SPY in a death cross?", "has ETH had a golden cross?",
])
def test_daily_chart_questions_reach_technicals(text: str) -> None:
    assert research.daily_technicals_asked(text)
    assert _read(text).kind is ResearchKind.TECHNICALS


def test_a_period_move_is_not_a_moving_average() -> None:
    assert not research.daily_technicals_asked("what was NVDA up over 3 days?")
    assert research.parse._period_days("what was NVDA up over 3 days?") == 3


def test_a_stated_multiple_keeps_the_leverage_engine() -> None:
    request = _read("what price does a 5x ETH short get liquidated at?")
    assert request.kind is ResearchKind.LEVERAGE and request.leverage == 5.0
    assert request.side == "short"
    assert research.pattern_reading_wins(request, "what price does a 5x ETH short get "
                                                  "liquidated at?")


def test_decay_questions_find_the_fund() -> None:
    assert research.leveraged_fund_asked(
        "how much does TQQQ decay if QQQ goes sideways for a month?") == "TQQQ"
    assert research.leveraged_fund_asked("is SOXL bad to hold long-term?") == "SOXL"
    assert research.leveraged_fund_asked("what is TQQQ trading at?") is None


def test_crypto_etf_questions_read_the_funds() -> None:
    assert _read("how are spot ETH ETFs doing?").symbols == ("ETHUSDT",)
    assert _read("are bitcoin ETFs seeing inflows?").kind is ResearchKind.SENTIMENT


def test_a_ticker_list_is_not_shouting() -> None:
    named, _ = research.research_symbols("compare BTC ETH SOL XRP DOGE ADA AVAX LINK DOT")
    assert len(named) == 9
    assert research.research_symbols("IS IT A GOOD TIME TO BUY")[0] == ()


@pytest.mark.parametrize("text", ["did the desk ever trade",
                                  "how many decisions have you made total",
                                  "have you placed any trades yet?"])
def test_track_record_questions(text: str) -> None:
    assert classify(text, now=NOON).intent is Intent.PERFORMANCE


def test_a_named_trade_question_is_not_the_track_record() -> None:
    assert classify("did you trade NVDA", now=NOON).intent is not Intent.PERFORMANCE


def test_take_profit_and_weekend_gap_are_the_odds_engine() -> None:
    tp = _read_3("good take profit for a nvda long from here")
    assert tp.kind is ResearchKind.ANALOGUE and tp.horizon_hours == 24
    gap = _read_3("is my qqq perp long gonna gap over the weekend")
    assert gap.kind is ResearchKind.ANALOGUE and gap.weekend


@pytest.mark.parametrize(("text", "days"), [
    ("sol 30 day price change", 30), ("btc 7 day change", 7), ("nvda 30 day return", 30),
    ("tsla 52 week high and low", 365), ("what is tsla 52w high and low", 365),
])
def test_period_questions(text: str, days: int) -> None:
    assert research.parse._period_days(text) == days
    assert _read_3(text).kind is ResearchKind.QUOTE


@pytest.mark.parametrize("text", ["whats nvda stock at", "nvda at?", "where is nvda at"])
def test_price_at_is_a_quote(text: str) -> None:
    assert _read_3(text).kind is ResearchKind.QUOTE


def test_spread_and_liquidity_questions_are_quotes() -> None:
    assert _read_3("if the spread on btc widens does my execution cost go up").kind \
        is ResearchKind.QUOTE
    assert _read_3("whats the best time of day to trade nvda perps").kind is ResearchKind.QUOTE
    assert _read_3("does liquidity dry up on weekends for nvda").kind is ResearchKind.QUOTE


def test_inflation_questions() -> None:
    assert research.macro._CPI_Q.search("whats the latest CPI number")
    assert research.macro._CPI_Q.search("inflation rate")
    assert research.macro._PCE_Q.search("whats pce inflation looking like")


@pytest.mark.parametrize(("text", "kind", "symbol"), [
    ("متى موعد أرباح شركة آبل القادمة؟", ResearchKind.FUNDAMENTALS, "AAPLUSDT"),
    ("هل يجب أن أشتري نيفيديا غدا؟", ResearchKind.ANALOGUE, "NVDAUSDT"),
    ("MSTR의 기술적 지표는 어때?", ResearchKind.TECHNICALS, "MSTRUSDT"),
])
def test_arabic_and_korean(text: str, kind: ResearchKind, symbol: str) -> None:
    request = _read_3(text)
    assert request.kind is kind and request.symbols[0] == symbol


def test_lowercase_coin() -> None:
    assert coin_as_ticker("macd on coin, bullish or bearish crossover") \
        == "macd on COIN, bullish or bearish crossover"
    assert coin_as_ticker("is this a good coin to buy") == "is this a good coin to buy"
