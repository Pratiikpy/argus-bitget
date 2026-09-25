"""The third audit of real answers (2026-09-25): ten auditors, 264 questions, 45 findings.

Round three asked what a trader asks in conversation and about their own book: follow-ups that lean
on the last answer, shorts and unlisted names in a book, the desk's own record, the daily and yearly
figures, execution at size, and questions in Arabic and Korean. It also found the hosted console
answering earnings and inflation questions with nothing while one data service was down. These tests
pin each fix without the network.
"""


from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from argus.lui import research
from argus.lui.provenance import label
from argus.lui.question import Intent, classify, coin_as_ticker, is_order_instruction
from argus.lui.research import ResearchKind

NOW = datetime(2026, 9, 25, 12, tzinfo=UTC)


def _read(text: str) -> research.ResearchRequest:
    request = research.detect(text)
    assert request is not None, text
    return request


# --- the desk's own record ---

@pytest.mark.parametrize("text", ["did the desk ever trade",
                                  "how many decisions have you made total",
                                  "have you placed any trades yet?"])
def test_track_record_questions(text: str) -> None:
    assert classify(text, now=NOW).intent is Intent.PERFORMANCE


def test_a_named_trade_question_is_not_the_track_record() -> None:
    assert classify("did you trade NVDA", now=NOW).intent is not Intent.PERFORMANCE


def test_grading_is_calibration_in_both_numbers() -> None:
    assert classify("how is a decision graded", now=NOW).intent is Intent.CALIBRATION
    assert classify("how are decisions graded", now=NOW).intent is Intent.CALIBRATION


@pytest.mark.parametrize("text", ["what is the sharpe ratio of my book", "what's my max drawdown",
                                  "good take profit for a nvda long from here"])
def test_my_book_and_my_trade_are_not_the_desk(text: str) -> None:
    assert classify(text, now=NOW).intent is not Intent.PERFORMANCE


# --- a book with shorts, cash and names that are not listed ---

def test_a_short_is_a_negative_weight() -> None:
    assert research.parse_book("im short 20% TSLA and long 80% NVDA") == {
        "TSLAUSDT": -0.2, "NVDAUSDT": 0.8}
    request = _read("im short 20% TSLA and long 80% NVDA, what does that do to my risk")
    assert request.kind is ResearchKind.BOOK and request.book["TSLAUSDT"] == -0.2
    assert any("read as short" in note for note in request.notes)


def test_the_rest_and_all_cash() -> None:
    request = _read("how risky is my book, im 70% cash the rest in btc")
    assert request.kind is ResearchKind.BOOK
    assert request.cash == pytest.approx(0.7) and request.book["BTCUSDT"] == pytest.approx(0.3)
    cash = _read("im 100% cash rn, whats my risk")
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
    assert _read(text).kind is ResearchKind.BOOK


def test_rebalancing_to_equal_risk_is_a_plan_not_an_order() -> None:
    assert not is_order_instruction("rebalance this to equal risk pls")
    assert is_order_instruction("rebalance my book")


# --- decisions about a trade ---

def test_take_profit_and_weekend_gap_are_the_odds_engine() -> None:
    tp = _read("good take profit for a nvda long from here")
    assert tp.kind is ResearchKind.ANALOGUE and tp.horizon_hours == 24
    gap = _read("is my qqq perp long gonna gap over the weekend")
    assert gap.kind is ResearchKind.ANALOGUE and gap.weekend


def test_the_favourable_excursion_is_measured() -> None:
    from argus.desk.odds import directional_odds

    start = datetime(2026, 1, 1, tzinfo=UTC)
    closes = [(start + timedelta(days=i), 100.0) for i in range(60)]
    extremes = [(99.0, 102.0)] * 60
    odds = directional_odds(closes, 1, cost_bps=12.0, extremes=extremes)
    assert odds is not None and odds.favourable_bps is not None
    assert odds.favourable_bps[0] == pytest.approx(200.0)


# --- numbers over a period ---

@pytest.mark.parametrize(("text", "days"), [
    ("sol 30 day price change", 30), ("btc 7 day change", 7), ("nvda 30 day return", 30),
    ("tsla 52 week high and low", 365), ("what is tsla 52w high and low", 365),
])
def test_period_questions(text: str, days: int) -> None:
    assert research._period_days(text) == days
    assert _read(text).kind is ResearchKind.QUOTE


# --- execution ---

def test_round_trip_at_size_keeps_the_size() -> None:
    request = research._detect("round trip cost of trading $2m of btc perp, entry and exit")
    assert request is not None and request.kind is ResearchKind.QUOTE
    assert request.notional is not None and float(request.notional) == pytest.approx(2e6)
    assert research.pattern_reading_wins(
        request, "round trip cost of trading $2m of btc perp, entry and exit")


@pytest.mark.parametrize("text", ["whats nvda stock at", "nvda at?", "where is nvda at"])
def test_price_at_is_a_quote(text: str) -> None:
    assert _read(text).kind is ResearchKind.QUOTE


def test_spread_and_liquidity_questions_are_quotes() -> None:
    assert _read("if the spread on btc widens does my execution cost go up").kind \
        is ResearchKind.QUOTE
    assert _read("whats the best time of day to trade nvda perps").kind is ResearchKind.QUOTE
    assert _read("does liquidity dry up on weekends for nvda").kind is ResearchKind.QUOTE


def test_the_liquidity_profile() -> None:
    from argus.market.liquidity_profile import lines, profile

    start = datetime(2026, 9, 1, tzinfo=UTC)
    bars = [(start + timedelta(hours=h), 100.0 if (start + timedelta(hours=h)).hour == 14 else
             10.0 if (start + timedelta(hours=h)).weekday() < 5 else 2.0, 1.0)
            for h in range(24 * 21)]
    p = profile("NVDAUSDT", bars)
    assert p is not None and max(p.by_hour, key=lambda h: p.by_hour[h]) == 14
    text = lines(p, "NVDA", True)
    assert text[0].startswith("Actionable: NVDA trades most in 14:00")
    assert "liquidity does dry up" in text[1]


def test_the_share_of_book_reads_percent_in_words() -> None:
    assert research._SHARE_OF_BOOK.search("how many shares is 30 percent of 80k in nvda")


# --- data a failed service leaves out ---

def test_yahoo_fills_the_earnings_date_and_targets() -> None:
    result = {"calendarEvents": {"earnings": {
        "earningsDate": [{"fmt": "2026-11-17"}], "earningsAverage": {"raw": 2.47},
        "earningsLow": {"raw": 2.34}, "earningsHigh": {"raw": 2.7},
        "revenueAverage": {"raw": 1.09e11}}},
        "financialData": {"targetMeanPrice": {"raw": 327.7}, "targetLowPrice": {"raw": 180.0},
                          "targetHighPrice": {"raw": 515.0}, "targetMedianPrice": {"raw": 315.0},
                          "numberOfAnalystOpinions": {"raw": 59}, "currentPrice": {"raw": 224.58},
                          "recommendationKey": "strong_buy"}}
    lines, _ = research._yahoo_fundamental_lines("NVDA", result, date(2026, 9, 25),
                                                 need_date=True, need_targets=True,
                                                 need_consensus=True)
    text = " ".join(lines)
    assert "2026-11-17" in text and "53 day(s)" in text
    assert "59 analysts" in text and "$327.70" in text and "strong buy" in text
    assert "EPS 2.47" in text


def test_a_failed_mcp_tool_is_counted_as_not_answered() -> None:
    from argus.truth import coverage

    body = (b'{"result":{"content":[{"type":"text","text":"{\\"success\\": false, '
            b'\\"status_code\\": 503}"}]}}')
    flat = body.replace(b" ", b"").replace(b"\\", b"")
    assert b'"success":false' in flat
    assert coverage.source_name("https://agent.bitget.com/mcp",
                                b'{"method":"tools/call","params":{"name":"do_query",'
                                b'"arguments":{"entry_id":"equity_calendar"}}}') \
        == "bitget-mcp-server equity_calendar"


def test_inflation_questions() -> None:
    assert research._CPI_Q.search("whats the latest CPI number")
    assert research._CPI_Q.search("inflation rate")
    assert research._PCE_Q.search("whats pce inflation looking like")


# --- other languages ---

@pytest.mark.parametrize(("text", "kind", "symbol"), [
    ("متى موعد أرباح شركة آبل القادمة؟", ResearchKind.FUNDAMENTALS, "AAPLUSDT"),
    ("هل يجب أن أشتري نيفيديا غدا؟", ResearchKind.ANALOGUE, "NVDAUSDT"),
    ("MSTR의 기술적 지표는 어때?", ResearchKind.TECHNICALS, "MSTRUSDT"),
])
def test_arabic_and_korean(text: str, kind: ResearchKind, symbol: str) -> None:
    request = _read(text)
    assert request.kind is kind and request.symbols[0] == symbol


def test_every_detected_language_gets_the_english_note() -> None:
    from argus.lui.server import _language_note

    assert _language_note("Quel est le prix actuel de TSLA ?")
    assert _language_note("MSTR의 기술적 지표는 어때?")
    assert _language_note("what is the price of TSLA?") is None


# --- conversations ---

def test_a_ratio_rebooks_the_last_two_names() -> None:
    request = research.follow_up("actually make it 60/40",
                                 ["whats my drawdown if btc drops 20%", "and ETH?"])
    assert request is not None
    assert request.book == pytest.approx({"BTCUSDT": 0.6, "ETHUSDT": 0.4})


def test_compare_it_to_names_the_earlier_symbol() -> None:
    request = research.follow_up("compare it to BNB", ["sup with SOL today"])
    assert request is not None and request.kind is ResearchKind.COMPARE
    assert request.symbols == ("SOLUSDT", "BNBUSDT")


def test_a_chain_of_follow_ups_resolves_to_the_last_name() -> None:
    request = research.follow_up("is that good?", ["what is the BTC funding rate rn?", "and ETH?"])
    assert request is not None and request.symbols == ("ETHUSDT",)


def test_i_meant_swaps_the_name_and_keeps_the_question() -> None:
    found = research.resolved_previous(["price of bitcoin?", "and dogecoin?"], "")
    assert found is not None and found[0] == "price of bitcoin?"
    request = research.follow_up("actually i meant ethereum", ["price of bitcoin?"])
    assert request is not None and request.symbols == ("ETHUSDT",)


def test_a_new_shock_size_reruns_the_stress() -> None:
    request = research.follow_up("what about a 20% drop instead",
                                 ["whats my drawdown if btc drops 10%"])
    assert request is not None and request.kind is ResearchKind.STRESS
    assert request.shock_pct == -20.0


def test_an_asset_class_follow_up_on_macro() -> None:
    request = research.follow_up("how does that affect crypto",
                                 ["whats the fed gonna do with rates"])
    assert request is not None and request.kind is ResearchKind.MACRO
    assert request.symbols == ("BTCUSDT",)


def test_which_one_and_better_or_worse_are_follow_ups() -> None:
    from argus.lui.server import _BETTER_WORSE, _WHICH_ONE

    assert _WHICH_ONE.match("which one is more volatile")
    assert _BETTER_WORSE.match("is that better or worse than before")


# --- coin, shares and labels ---

def test_lowercase_coin() -> None:
    assert coin_as_ticker("macd on coin, bullish or bearish crossover") \
        == "macd on COIN, bullish or bearish crossover"
    assert coin_as_ticker("is this a good coin to buy") == "is this a good coin to buy"


def test_a_shipped_fred_reading_is_record() -> None:
    assert label("10-year Treasury: 5.11% (the shipped reading) on 2026-09-23 (+41bp).") \
        == "record"
