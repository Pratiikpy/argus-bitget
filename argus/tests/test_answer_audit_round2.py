"""The second audit of real answers (2026-09-25): 41 findings, each pinned by its mechanism.

Round one's fixes held (no regressions). Round two asked what a trader asks next: the hedge they
have in mind, the daily chart, a leveraged fund, a follow-up to the last answer, a question the desk
cannot answer and should say so. These tests keep each fix fixed without the network.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from argus.lui import research
from argus.lui.provenance import label
from argus.lui.research import ResearchKind


def _read(text: str) -> research.ResearchRequest:
    request = research._detect(text)
    assert request is not None, text
    return request


# --- hedges the trader names are measured, not ignored ---

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


# --- Bitget's own long/short split, for every perpetual ---

def test_long_short_line_reads_the_split() -> None:
    from argus.market.long_short import LongShort, lines

    text = lines(LongShort("SOLUSDT", 0.75, 0.761, 0.50, 0), "SOL")[0]
    assert "75% of accounts are long" in text and "ratio of 3.00" in text
    assert "-1.1 points" in text and "50% is long" in text and "contrarian" in text
    assert label(text) == "live"


@pytest.mark.parametrize("text", [
    "what is the long/short ratio on SOL?", "are more traders long than short on XRP?",
    "SOL long short ratio",
])
def test_long_short_questions_are_recognised(text: str) -> None:
    assert research.LONG_SHORT_QUESTION.search(text)


# --- the cost of holding, and whether to hold ---

@pytest.mark.parametrize(("text", "hours"), [
    ("funding cost to hold BTC long a week", 168),
    ("cost of holding ETH short for 3 days", 72),
    ("what does it cost to hold NVDA overnight?", None),
])
def test_holding_cost_is_a_quote_with_its_period(text: str, hours: int | None) -> None:
    assert research.hold_cost_question(text)
    request = _read(text)
    assert request.kind is ResearchKind.QUOTE
    found = research._HOLD_PERIOD.search(text)
    assert found is not None
    if hours is not None:
        amount = float(found.group(1) or 1)
        unit = found.group(2).lower()
        assert amount * (1 if unit[0] == "h" else 24 if unit[0] == "d" else 168) == hours


def test_should_i_hold_is_the_odds_engine() -> None:
    request = _read("should I hold NVDA for a week?")
    assert request.kind is ResearchKind.ANALOGUE and request.horizon_hours == 168


# --- the daily chart ---

@pytest.mark.parametrize("text", [
    "what is the 200-day moving average of NVDA?", "NVDA daily RSI", "is BTC above its 50 day ema?",
    "is SPY in a death cross?", "has ETH had a golden cross?",
])
def test_daily_chart_questions_reach_technicals(text: str) -> None:
    assert research.daily_technicals_asked(text)
    assert _read(text).kind is ResearchKind.TECHNICALS


def test_a_period_move_is_not_a_moving_average() -> None:
    assert not research.daily_technicals_asked("what was NVDA up over 3 days?")
    assert research._period_days("what was NVDA up over 3 days?") == 3


def test_daily_technicals_compute_average_rsi_and_the_last_cross(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # 150 days falling then 150 rising: the 50-day crosses up through the 200-day inside the rise
    closes = [100 - i * 0.2 for i in range(150)] + [70 + i * 0.5 for i in range(150)]
    monkeypatch.setattr(research, "_daily_closes", lambda symbol: (closes, "test closes"))
    lines, _ = research._daily_technicals("SPYUSDT", "is SPY in a death cross?")
    assert lines[0].startswith("Actionable: No — SPY is in golden-cross territory.")
    assert "golden cross" in lines[0] and "trading day(s) ago" in lines[0]
    assert any(line.startswith("RSI(14, 1D)") for line in lines)
    average = next(line for line in lines if "200-day simple moving average" in line)
    assert f"{sum(closes[-200:]) / 200:,.2f}" in average


# --- leverage: the liquidation price itself ---

def test_a_stated_multiple_keeps_the_leverage_engine() -> None:
    request = _read("what price does a 5x ETH short get liquidated at?")
    assert request.kind is ResearchKind.LEVERAGE and request.leverage == 5.0
    assert request.side == "short"
    assert research.pattern_reading_wins(request, "what price does a 5x ETH short get "
                                                  "liquidated at?")


# --- questions outside the desk say so first ---

def test_options_loans_and_life_savings_lead_with_the_scope() -> None:
    assert "options are outside" in research._scope_lead("should I buy NVDA calls?", "")[0]
    assert "borrowing against" in research._scope_lead("can I take a loan against my BTC?", "")[0]
    assert "not a licensed adviser" in research._scope_lead(
        "I am 62, should I put all my savings in BTC?", "")[0]
    assert research._scope_lead("what happens on a margin call at 10x BTC?", "BTCUSDT") == []


def test_put_call_ratio_is_answered_not_logged() -> None:
    assert _read("NVDA put/call ratio").kind is ResearchKind.SENTIMENT


# --- leveraged funds ---

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


def test_decay_questions_find_the_fund() -> None:
    assert research.leveraged_fund_asked(
        "how much does TQQQ decay if QQQ goes sideways for a month?") == "TQQQ"
    assert research.leveraged_fund_asked("is SOXL bad to hold long-term?") == "SOXL"
    assert research.leveraged_fund_asked("what is TQQQ trading at?") is None


# --- a named shock keeps its size and its subject ---

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
    assert research._shock_subject("a 10% drop in gold do to 50% XAU 50% NVDA",
                                   {"XAUUSDT", "NVDAUSDT"}) == "XAUUSDT"


# --- follow-ups read against the previous question ---

def test_a_new_book_reruns_the_previous_question() -> None:
    request = research.follow_up("actually make it 70% NVDA 30% AAPL",
                                 ["what does a 10% Nasdaq drop do to 50% NVDA 50% AAPL"])
    assert request is not None and request.kind is ResearchKind.STRESS
    assert request.book == {"NVDAUSDT": 0.7, "AAPLUSDT": 0.3}
    assert request.shock_pct == -10.0


def test_is_that_bullish_reads_the_previous_name() -> None:
    request = research.follow_up("is that bullish?", ["ETH open interest"])
    assert request is not None and request.kind is ResearchKind.SENTIMENT
    assert request.symbols == ("ETHUSDT",)


def test_bare_follow_ups_and_why_are_recognised() -> None:
    from argus.lui.server import _BARE_WHY

    assert research.BARE_FOLLOW.match("what about that one?")
    assert research.BARE_FOLLOW.match("tell me more")
    assert not research.BARE_FOLLOW.match("what about COIN?")
    assert _BARE_WHY.match("why?") and _BARE_WHY.match("how come")
    assert not _BARE_WHY.match("why did BTC fall?")


# --- leads: the figure asked for comes first ---

def test_crypto_etf_questions_read_the_funds() -> None:
    assert _read("how are spot ETH ETFs doing?").symbols == ("ETHUSDT",)
    assert _read("are bitcoin ETFs seeing inflows?").kind is ResearchKind.SENTIMENT


def test_a_ticker_list_is_not_shouting() -> None:
    named, _ = research.research_symbols("compare BTC ETH SOL XRP DOGE ADA AVAX LINK DOT")
    assert len(named) == 9
    assert research.research_symbols("IS IT A GOOD TIME TO BUY")[0] == ()


# --- provenance labels on the new lines ---

@pytest.mark.parametrize(("line", "expected"), [
    ("Actionable: NVDA last 226.3 USDT on Bitget (+1.28% over 24h)", "live"),
    ("Actionable: BTC funding is +0.0046% per 8h settlement.", "live"),
    ("Actionable: US spot ETH ETFs, 24 Sep: net +$66m (creations less redemptions)", "live"),
    ("Actionable: the data source holds no revenue figure for NVDA.", "missing"),
    ("Missing: the 200-day average needs 200 daily closes.", "missing"),
    ("Whether an rToken pays dividends is set by Bitget's terms, which this desk has not "
     "verified.", "missing"),
    ("Liquidation price: a 10x long opened at the last price is liquidated near 77,008",
     "computed"),
    ("Actionable: over 21 trading days in which QQQ went sideways, TQQQ fell short across 1213 "
     "overlapping windows of its own history.", "record"),
    ("Computed by ARGUS from Bitget daily candles, 499 days.", None),
    ("Size NVDA so that its worst observed 24 hours (-4.4%) is a loss you would accept",
     "computed"),
])
def test_new_lines_carry_their_provenance(line: str, expected: str | None) -> None:
    assert label(line) == expected
