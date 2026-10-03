"""What the engines compute and how the answer states it.

Gathered on 2026-09-27 from the three audit-round files (2026-09-25), which grouped tests by
the day a defect was found rather than by what they pin (audit finding 170).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from argus.lui import research
from argus.lui.provenance import label
from argus.lui.question import Intent, classify
from argus.lui.research import ResearchKind

MORNING = datetime(2026, 9, 25, 10, tzinfo=UTC)


def _read(text: str) -> research.ResearchRequest:
    request = research.parse.read_request(text)
    assert request is not None, text
    return request


NOON = datetime(2026, 9, 25, 12, tzinfo=UTC)


def test_an_order_size_in_units_is_priced(monkeypatch: pytest.MonkeyPatch) -> None:
    from decimal import Decimal

    from argus.market import bitget

    class T:
        last = Decimal("80000")

    monkeypatch.setattr(bitget, "fetch_tickers", lambda: {"BTCUSDT": T()})
    priced = research.parse._unit_notional("Split a 200 BTC sell order for me", "BTCUSDT")
    assert priced is not None
    notional, note = priced
    assert notional == Decimal("16000000.0") and "200 BTC read as $16,000,000" in note


def test_a_holiday_question_is_answered_directly() -> None:
    assert "is not a US market holiday" in (research.holiday_line("is today a US holiday",
                                                                  MORNING) or "")
    assert "is a US market holiday" in (research.holiday_line(
        "is thursday a holiday", datetime(2026, 11, 23, 15, tzinfo=UTC)) or "")


def test_the_hurdle_is_explained_from_the_code_that_sets_it() -> None:
    lines, sources = research.hurdle_lines()
    assert "12bps of Bitget taker fees" in lines[0] and "18.8bps" in lines[0]
    assert any("default_hurdle_bps" in s.ref for s in sources)


def test_the_fundamentals_lead_follows_the_question(monkeypatch: pytest.MonkeyPatch) -> None:
    # the declared dividend is read from SEC XBRL when the lines carry none (round 30); offline
    # here, it is said to be missing unless stubbed
    monkeypatch.setattr(research.fundamentals, "_declared_dividend", lambda ticker: None)
    lines = ["Bottom line: AAPL's next report date is not published yet.",
             "Valuation on 2026-09-24: P/E (trailing 12m) 38.1, dividend yield 0.31%.",
             "Institutions: 6,493 holders own 76.6% of the shares (as of 2026-09-23)."]
    assert research.fundamentals._fundamentals_focus(
        lines, "what's the dividend yield on AAPL", "AAPL")[0].startswith(
        "Bottom line: Valuation on")
    assert research.fundamentals._fundamentals_focus(lines, "who owns AAPL", "AAPL")[0] \
        .startswith("Bottom line: Institutions:")
    assert research.fundamentals._fundamentals_focus(lines, "AAPL earnings", "AAPL") == lines
    missing: Any = research.fundamentals._fundamentals_focus(lines[:1], "AAPL dividend", "AAPL")
    assert "holds no dividend figure" in missing[0]
    monkeypatch.setattr(research.fundamentals, "_declared_dividend",
                        lambda ticker: f"{ticker} declared $0.27 a share for the quarter ending "
                                       f"27 Jun 2026 (10-Q filed 31 Jul 2026, SEC EDGAR XBRL).")
    filed: Any = research.fundamentals._fundamentals_focus(lines[:1], "AAPL dividend", "AAPL")
    assert filed[0].startswith("Bottom line: AAPL declared $0.27 a share")


def test_long_short_line_reads_the_split() -> None:
    from argus.market.long_short import LongShort, lines

    text = lines(LongShort("SOLUSDT", 0.75, 0.761, 0.50, 0), "SOL")[0]
    assert "75% of accounts are long" in text and "ratio of 3.00" in text
    assert "-1.1 points" in text and "50% is long" in text and "contrarian" in text
    assert label(text) == "live"


def test_daily_technicals_compute_average_rsi_and_the_last_cross(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # 150 days falling then 150 rising: the 50-day crosses up through the 200-day inside the rise
    closes = [100 - i * 0.2 for i in range(150)] + [70 + i * 0.5 for i in range(150)]
    for _module in (research.analogue, research.quote):
        monkeypatch.setattr(_module, "_daily_closes", lambda symbol: (closes, "test closes"))
    lines, _ = research.quote._daily_technicals("SPYUSDT", "is SPY in a death cross?")
    assert lines[0].startswith("Bottom line: No — SPY is in golden-cross territory.")
    assert "golden cross" in lines[0] and "trading days ago" in lines[0]
    assert any(line.startswith("RSI(14, 1D)") for line in lines)
    average = next(line for line in lines if "200-day simple moving average" in line)
    assert f"{sum(closes[-200:]) / 200:,.2f}" in average


def test_options_loans_and_life_savings_lead_with_the_scope() -> None:
    lead, _ = research.analogue._scope_lead("should I buy BTC calls?", "BTCUSDT")
    assert "no US-listed options chain" in lead[0] and "not priced here" in lead[0]
    assert "borrowing against" in research.analogue._scope_lead(
        "can I take a loan against my BTC?", "")[0][0]
    assert "not a licensed adviser" in research.analogue._scope_lead(
        "I am 62, should I put all my savings in BTC?", "")[0][0]
    assert research.analogue._scope_lead(
        "what happens on a margin call at 10x BTC?", "BTCUSDT") == ([], [])


_CHAIN = ("Options on NVDA (Cboe, delayed): 30-day implied vol 48.1%; put/call 0.58 by today's "
          "volume, 0.73 by open interest.")


def test_an_options_question_leads_with_the_chain_it_read_not_a_denial() -> None:
    # Stranger QA, 2026-09-29: "options are outside what this desk reads" led an answer whose own
    # later line gave the chain's put/call ratio.
    lead, sources = research.analogue._scope_lead(
        "what's the options put/call ratio for NVDA", "NVDAUSDT", ["Bottom line: x", _CHAIN])
    assert lead == [f"Bottom line: {_CHAIN}"] and sources == []
    buying, _ = research.analogue._scope_lead("should I buy NVDA calls?", "NVDAUSDT", [_CHAIN])
    assert buying[0] == f"Bottom line: {_CHAIN}" and "not priced here" in buying[1]
    assert not any("outside what this desk reads" in line for line in (*lead, *buying))


def test_an_options_question_reads_the_chain_when_the_answer_has_none(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.lui.answer import Source
    from argus.lui.research import positioning

    seen = Source(kind="venue", ref="cboe delayed_quotes/options", detail="NVDA")
    monkeypatch.setattr(positioning, "_options_line", lambda ticker: (_CHAIN, seen, {}))
    lead, sources = research.analogue._scope_lead("should I buy NVDA calls?", "NVDAUSDT", [])
    assert lead[0] == f"Bottom line: {_CHAIN}" and sources == [seen]
    monkeypatch.setattr(positioning, "_options_line", lambda ticker: None)
    failed, _ = research.analogue._scope_lead("should I buy NVDA calls?", "NVDAUSDT", [])
    assert "did not answer just now" in failed[0]


def test_put_call_ratio_is_answered_not_logged() -> None:
    assert _read("NVDA put/call ratio").kind is ResearchKind.SENTIMENT


def test_leveraged_decay_measures_sideways_windows_and_the_formula() -> None:
    from argus.research.leveraged_decay import measure

    start = date(2020, 1, 1)
    days = [start + timedelta(days=i) for i in range(400)]
    # an index that zig-zags 1% a day and goes nowhere: the textbook case for decay
    index = {d: 101.0 if i % 2 else 100.0 for i, d in enumerate(days)}
    fund: dict[date, float] = {}
    level = 100.0
    previous = None
    for d in days:
        if previous is not None:
            level *= 1 + 3 * (index[d] / index[previous] - 1)
        fund[d] = level
        previous = d
    result = measure("TQQQ", fund, index, 21)
    assert result.windows > 300
    assert result.median_shortfall is not None and result.median_shortfall < 0
    assert -0.1 < result.formula < 0


@pytest.mark.parametrize(("line", "expected"), [
    ("Bottom line: NVDA last 226.3 USDT on Bitget (+1.28% over 24h)", "live"),
    ("Bottom line: BTC funding is +0.0046% per 8h settlement.", "live"),
    ("Bottom line: US spot ETH ETFs, 24 Sep: net +$66m (creations less redemptions)", "live"),
    ("Bottom line: the data source holds no revenue figure for NVDA.", "missing"),
    ("Missing: the 200-day average needs 200 daily closes.", "missing"),
    ("Whether an rToken pays dividends is set by Bitget's terms, which this desk has not "
     "verified.", "missing"),
    ("Liquidation price: a 10x long opened at the last price is liquidated near 77,008",
     "computed"),
    ("Bottom line: over 21 trading days in which QQQ went sideways, TQQQ fell short across 1213 "
     "overlapping windows of its own history.", "record"),
    ("Computed by ARGUS from Bitget daily candles, 499 days.", None),
    ("Size NVDA so that its worst observed 24 hours (-4.4%) is a loss you would accept",
     "computed"),
])
def test_new_lines_carry_their_provenance(line: str, expected: str | None) -> None:
    assert label(line) == expected


def test_a_lead_with_one_gap_is_still_computed() -> None:
    # seen on the live page: the named-hedge answer was chipped MISSING for its TLT clause
    assert label("Bottom line: Of the hedges you named, XAU (short $25,700) removes about 11% of "
                 "the book's variance; TLT could not be measured (only 95 hours).") == "computed"


def test_grading_is_calibration_in_both_numbers() -> None:
    assert classify("how is a decision graded", now=NOON).intent is Intent.CALIBRATION
    assert classify("how are decisions graded", now=NOON).intent is Intent.CALIBRATION


def test_the_favourable_excursion_is_measured() -> None:
    from argus.desk.odds import directional_odds

    start = datetime(2026, 1, 1, tzinfo=UTC)
    closes = [(start + timedelta(days=i), 100.0) for i in range(60)]
    extremes = [(99.0, 102.0)] * 60
    odds = directional_odds(closes, 1, cost_bps=12.0, extremes=extremes)
    assert odds is not None and odds.favourable_bps is not None
    assert odds.favourable_bps[0] == pytest.approx(200.0)


def test_round_trip_at_size_keeps_the_size() -> None:
    request = research.parse.read_request(
        "round trip cost of trading $2m of btc perp, entry and exit")
    assert request is not None and request.kind is ResearchKind.QUOTE
    assert request.notional is not None and float(request.notional) == pytest.approx(2e6)
    assert research.pattern_reading_wins(
        request, "round trip cost of trading $2m of btc perp, entry and exit")


def test_the_liquidity_profile() -> None:
    from argus.market.liquidity_profile import lines, profile

    start = datetime(2026, 9, 1, tzinfo=UTC)
    bars = [(start + timedelta(hours=h), 100.0 if (start + timedelta(hours=h)).hour == 14 else
             10.0 if (start + timedelta(hours=h)).weekday() < 5 else 2.0, 1.0)
            for h in range(24 * 21)]
    p = profile("NVDAUSDT", bars)
    assert p is not None and max(p.by_hour, key=lambda h: p.by_hour[h]) == 14
    text = lines(p, "NVDA", True)
    assert text[0].startswith("Bottom line: NVDA trades most in 14:00")
    assert "liquidity does dry up" in text[1]


def test_yahoo_fills_the_earnings_date_and_targets() -> None:
    result = {"calendarEvents": {"earnings": {
        "earningsDate": [{"fmt": "2026-11-17"}], "earningsAverage": {"raw": 2.47},
        "earningsLow": {"raw": 2.34}, "earningsHigh": {"raw": 2.7},
        "revenueAverage": {"raw": 1.09e11}}},
        "financialData": {"targetMeanPrice": {"raw": 327.7}, "targetLowPrice": {"raw": 180.0},
                          "targetHighPrice": {"raw": 515.0}, "targetMedianPrice": {"raw": 315.0},
                          "numberOfAnalystOpinions": {"raw": 59}, "currentPrice": {"raw": 224.58},
                          "recommendationKey": "strong_buy"}}
    lines, _ = research.fundamentals._yahoo_fundamental_lines("NVDA", result, date(2026, 9, 25),
                                                 need_date=True, need_targets=True,
                                                 need_consensus=True)
    text = " ".join(lines)
    assert "2026-11-17" in text and "53 days" in text
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


def test_every_detected_language_gets_the_english_note() -> None:
    from argus.lui.server import _language_note

    assert _language_note("Quel est le prix actuel de TSLA ?")
    assert _language_note("MSTR의 기술적 지표는 어때?")
    assert _language_note("what is the price of TSLA?") is None


def test_a_shipped_fred_reading_is_record() -> None:
    assert label("10-year Treasury: 5.11% (the shipped reading) on 2026-09-23 (+41bp).") \
        == "record"
