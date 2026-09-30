"""How questions are read, as the round-7 role audits found them misread (2026-09-30)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from argus.lui.journal import position_and_pnl
from argus.lui.research import detect, research_symbols
from argus.lui.research.kinds import ResearchKind

NOW = datetime(2026, 9, 30, 12, tzinfo=UTC)


def test_p_and_l_is_never_the_platinum_contract() -> None:
    """A hostile review: "what is my current P&L?" was answered with platinum's price."""
    text = "I hold 100 shares of NVDA bought at $200, what is my current P&L?"
    assert research_symbols(text)[0] == ("NVDAUSDT",)
    held = position_and_pnl(text, now=NOW, price=lambda s: 228.14)
    assert held is not None
    assert "+$2,814.00 unrealised" in held[0][0]


def test_one_fill_with_a_profit_question_is_a_position() -> None:
    """A first-time user: "i bought 2 shares of tesla at 250 how am i doing" got an order cost."""
    held = position_and_pnl("i bought 2 shares of tesla at 250 how am i doing", now=NOW,
                            price=lambda s: 354.5)
    assert held is not None
    assert "you hold 2 TSLA at an average cost of $250.00" in held[0][0]
    assert "+$209.00" in held[0][0]


def test_one_fill_without_a_profit_question_is_left_alone() -> None:
    assert position_and_pnl("I bought 100 NVDA at 200, should I add more?", now=NOW,
                            price=lambda s: 228.0) is None


@pytest.mark.parametrize("text", ["what is the S&P 500 doing", "S&P 500 price",
                                  "how is the s&p today"])
def test_the_s_and_p_is_the_index(text: str) -> None:
    assert research_symbols(text)[0] == ("SP500USDT",)


@pytest.mark.parametrize(("text", "book", "cash"), [
    ("build me a portfolio: NVDA 70%, TSLA 20%, GOOGL 10%",
     {"NVDAUSDT": 0.7, "TSLAUSDT": 0.2, "GOOGLUSDT": 0.1}, 0.0),
    ("NVDA 70%, TSLA 20%, GOOGL 10% — what does that look like",
     {"NVDAUSDT": 0.7, "TSLAUSDT": 0.2, "GOOGLUSDT": 0.1}, 0.0),
    ("40% TSLA, 30% NVDA, 30% cash — biggest risk?", {"TSLAUSDT": 0.4, "NVDAUSDT": 0.3}, 0.3),
])
def test_names_given_with_weights_are_that_book(text: str, book: dict[str, float],
                                                cash: float) -> None:
    """A hostile review: stated weights were replaced by an equal-risk book without a word."""
    request = detect(text)
    assert request is not None and request.kind is ResearchKind.BOOK
    assert request.book == pytest.approx(book)
    assert request.cash == pytest.approx(cash)


def test_weights_over_one_hundred_percent_are_rescaled_and_said() -> None:
    request = detect("portfolio NVDA 40%, TSLA 40%, GOOGL 40%, how risky")
    assert request is not None and request.kind is ResearchKind.BOOK
    assert sum(request.book.values()) == pytest.approx(1.0)
    assert any("add up to 120%" in note for note in request.notes)


@pytest.mark.parametrize("text", [
    "I hold 50% NVDA, 50% AAPL — what if I add 10% TSLA",
    "I hold 60% NVDA 40% AAPL, what if QQQ drops 10%",
])
def test_an_add_or_a_shock_on_a_weighted_book_keeps_its_engine(text: str) -> None:
    request = detect(text)
    assert request is not None and request.kind is not ResearchKind.BOOK


def test_a_point_of_failure_is_a_question_about_the_book() -> None:
    from argus.lui.research.parse import _BOOK_QUESTION_STRONG

    assert _BOOK_QUESTION_STRONG.search("What's the biggest single point of failure in this book?")
    assert _BOOK_QUESTION_STRONG.search("what could sink my portfolio")


def test_swing_levels_are_the_nearest_fractal_points_either_side() -> None:
    from argus.lui.research.technicals import swing_levels

    lows = [10, 9, 8, 7, 8, 9, 10, 11, 12, 11, 10, 9, 10, 11, 12]
    highs = [x + 2 for x in lows]
    highs[8] = 20  # a swing high
    candles = [{"low": lo, "high": hi} for lo, hi in zip(lows, highs, strict=True)]
    support, resistance = swing_levels("BTCUSDT", 12.5, candles=candles) or (None, None)
    assert support == 9  # the later swing low (index 11), nearer than 7
    assert resistance == 20


def test_a_level_question_leads_with_the_level() -> None:
    from argus.lui.research.technicals import _answer_the_level_asked

    lines = ["Bottom line: momentum turning down.",
             "Price 100: nearest resistance 105 (5.0% above); nearest support 99 (1.0% below)."]
    assert _answer_the_level_asked("what's BTC's key support level", lines)[0].startswith(
        "Bottom line: Price 100: nearest resistance")


@pytest.mark.parametrize(("text", "bad"), [
    ("What happened on 2025-13-01?", "2025-13-01"),
    ("BTC close on 2025-02-29", "2025-02-29"),
    ("what did NVDA do on February 30, 2025", "February 30, 2025"),
    ("price on 2024-02-29", None),
    ("on March 3 2025", None),
])
def test_a_date_no_calendar_has_is_named(text: str, bad: str | None) -> None:
    from argus.lui.honesty import impossible_date

    assert impossible_date(text) == bad


@pytest.mark.parametrize(("text", "term"), [
    ("whats a perp", "perpetual contract"),
    ("waht is dca", "dollar-cost averaging"),
    ("whats sharpe ratio", "Sharpe ratio"),
    ("wat is leverage", "leverage"),
])
def test_a_newcomer_spelling_still_finds_the_glossary(text: str, term: str) -> None:
    from argus.lui.concepts import concept_asked

    found = concept_asked(text)
    assert found is not None and found.name == term


@pytest.mark.parametrize("text", ["whats my drawdown", "what is the sharpe"])
def test_the_desks_own_figure_is_not_a_definition(text: str) -> None:
    from argus.lui.concepts import concept_asked

    assert concept_asked(text) is None


@pytest.mark.parametrize(("text", "kind"), [
    ("is this financial advice", "lines"),
    ("how do i buy crypto", "lines"),
    ("can i lose more money than i put in", "lines"),
    ("whats a good stock for beginners", "reask"),
    ("should i buy the dip", "reask"),
])
def test_the_questions_a_newcomer_asks_first_are_answered(text: str, kind: str) -> None:
    from argus.lui.newcomer import reply

    found = reply(text)
    assert found is not None
    assert bool(found.lines) is (kind == "lines")


def test_how_to_buy_a_named_stock_is_not_the_newcomer_answer() -> None:
    from argus.lui.newcomer import reply

    assert reply("how do i buy NVDA") is None


@pytest.mark.parametrize(("text", "options"), [
    ("how much should i put in nvda if i have 3k", False),
    ("put $500 into BTC?", False),
    ("should I buy NVDA puts", True),
    ("buy a put on SPY", True),
])
def test_put_the_verb_is_not_a_put_option(text: str, options: bool) -> None:
    from argus.lui.research.parse import _OPTIONS_Q

    assert bool(_OPTIONS_Q.search(text)) is options


def test_an_amount_for_one_name_uses_the_money_stated() -> None:
    from argus.lui.research.sizing import capital_lines, stated_capital

    request = detect("how much should i put in nvda if i have 3k")
    assert request is not None and request.kind is ResearchKind.IMPACT
    assert float(request.notional or 0) == 3000
    assert stated_capital("i got 2 grand") == 2000
    lead = capital_lines("NVDA", 3000, -0.033)[0]
    assert lead.startswith("Bottom line: with $3,000, holding about $909 of NVDA")


@pytest.mark.parametrize("text", ["what does it cost to buy $500 of BTC",
                                  "how much does it cost to buy $500 of ETH"])
def test_the_cost_of_a_buy_is_an_execution_question(text: str) -> None:
    request = detect(text)
    assert request is not None and request.kind is ResearchKind.EXECUTION
    assert float(request.notional or 0) == 500


def test_us_as_the_reader_is_not_the_desk() -> None:
    from argus.lui.research.parse import about_the_desk

    assert not about_the_desk("What are funding rates telling us about crowd positioning in ETH?")
    assert about_the_desk("why did we short TSLA")


def test_a_view_on_one_name_is_a_research_question() -> None:
    from argus.lui.research.fundamentals import pattern_reading_wins

    request = detect("Give me a quick take on ETH.")
    assert request is not None and request.kind is ResearchKind.IMPACT
    assert pattern_reading_wins(request, "Give me a quick take on ETH.")


@pytest.mark.parametrize(("text", "why"), [
    ("why", True), ("Why do you say that?", True), ("what's that based on", True),
    ("why did you skip NVDA", False), ("why is BTC down", False),
])
def test_a_request_for_reasons_is_told_from_a_new_question(text: str, why: bool) -> None:
    from argus.lui.server import _WHY_THAT

    assert bool(_WHY_THAT.match(text)) is why


def test_the_desk_against_holding_the_market_says_what_one_trade_cannot_show() -> None:
    """A judge's audit: "What's ARGUS's edge over buy-and-hold?" got "No open positions"."""
    from argus.lui import edge
    from argus.paper.ledger import PaperLedger

    ledger = PaperLedger(path=Path(__file__).resolve().parents[1] / "data" / "paper_ledger.jsonl")
    if not ledger.entries:
        pytest.skip("no ledger in this checkout")
    start = datetime.fromisoformat(ledger.entries[0].decided_at.replace("Z", "+00:00"))
    closes = [(start.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=d),
               100.0 + d) for d in range(1, 11)]
    lines, _, data = edge.answer(ledger, closes=lambda symbol: closes)
    assert lines[0].startswith("Bottom line: since its first decision on")
    assert "holding QQQ at +8.91%" in lines[0]  # 101 to 110
    if data["trades"] < edge.SHOWS_AN_EDGE_AT:
        assert "shows no edge either way" in lines[0]
    assert edge.EDGE_Q.search("What's ARGUS's edge over a simple buy-and-hold strategy?")


def test_the_risk_mechanism_is_read_from_the_code_that_enforces_it() -> None:
    """A judge's audit: "how does your risk control layer work, step by step" got a count."""
    from argus.lui import riskexplain
    from argus.risk.constitution import ConstitutionPolicy

    lines, _, _ = riskexplain.answer()
    text = " ".join(lines)
    assert f"${ConstitutionPolicy().max_position_notional:,} a position" in text
    assert riskexplain.RISK_HOW_Q.search("How does your risk control layer actually work?")
    assert not riskexplain.RISK_HOW_Q.search("what did the risk layer block")


@pytest.mark.parametrize("text", ["does BTC outperform QQQ", "is ETH beating BTC this month",
                                  "has NVDA done better than AMD"])
def test_which_of_two_names_did_better_is_a_comparison(text: str) -> None:
    from argus.lui.research.fundamentals import pattern_reading_wins

    request = detect(text)
    assert request is not None and request.kind is ResearchKind.COMPARE
    assert pattern_reading_wins(request, text)


def test_the_model_allowance_left_is_counted_per_visitor() -> None:
    from argus.lui import server

    visitor = "allowance-test-visitor"
    server._VISITS.pop(visitor, None)
    assert server.allowance_left(visitor) == server.MODEL_CALLS_PER_VISITOR_PER_HOUR
    server._VISITS[visitor] = [__import__("time").monotonic()] * 35
    assert server.allowance_left(visitor) == server.MODEL_CALLS_PER_VISITOR_PER_HOUR - 35
    assert "a.model_left" in server.PAGE
    server._VISITS.pop(visitor, None)


def test_an_index_around_a_fed_date_is_the_backdrop_and_stands_over_the_model() -> None:
    """Live, 2026-09-30: once "S&P 500" resolved, the hosted model refused the question as a
    prediction; the backdrop, said as not a forecast, is the honest answer and now stands."""
    from argus.lui.research.fundamentals import pattern_reading_wins

    text = "What will the S&P 500 do after the next FOMC meeting?"
    request = detect(text)
    assert request is not None and request.kind is ResearchKind.MACRO
    assert pattern_reading_wins(request, text)


_NVDA_ACTIONS = [
    {"ex_dividend_date": "2024-06-10", "split_numerator": "10.0000", "split_denominator": "1.0000"},
    {"ex_dividend_date": "2021-07-20", "split_numerator": "4.0000", "split_denominator": "1.0000"},
    {"ex_dividend_date": "2026-06-01", "amount": 0.01},
]


def test_a_split_that_never_happened_is_named_and_the_shares_are_valued() -> None:
    """A hostile review, 2026-09-30: a made-up 3-for-1 split went unchallenged and the 30 shares
    were never valued."""
    from datetime import date

    from argus.lui.research import splits

    lines, _, data = splits.check(
        "after NVDA's 3-for-1 split last week, what's my 30 share position worth", "NVDAUSDT",
        price=lambda s: 228.4, rows=lambda t: _NVDA_ACTIONS,
        today=date(2026, 9, 30)) or ([], [], {})
    assert "last split was 10-for-1 on 10 Jun 2024" in lines[0]
    assert "30 shares of NVDA at the last Bitget price, 228.40, are worth $6,852.00." in lines[1]
    assert data["value"] == pytest.approx(6852.0)


@pytest.mark.parametrize(("text", "account", "entry"), [
    ("I have $25k, risk 1%, NVDA stop 3%, how many shares should I buy", 25000, None),
    ("stop loss at 3 percent, want to risk one percent of my 25000 account on NVDA", 25000, None),
    ("I want to risk $300, my stop is 3%, how many shares should I buy of a $50 stock", None, 50),
])
def test_sizing_reads_the_account_and_the_entry_as_traders_write_them(
        text: str, account: float | None, entry: float | None) -> None:
    from argus.lui.research import sizing

    assert sizing.asks_for_size(text)
    _, _, data = sizing.answer(text)
    assert data["sizing"]["account"] == account
    assert data["sizing"]["entry"] == entry


@pytest.mark.parametrize(("text", "iso", "noted"), [
    ("NVDA on 13/02/2024", "NVDA on 2024-02-13", False),
    ("NVDA on 02/13/2024", "NVDA on 2024-02-13", False),
    ("NVDA close on 03/04/2024", "NVDA close on 2024-03-04", True),
])
def test_numeric_dates_are_read_as_the_day_they_are(text: str, iso: str, noted: bool) -> None:
    from argus.lui.honesty import iso_dates

    rewritten, note = iso_dates(text)
    assert rewritten == iso
    assert bool(note) is noted


def test_a_future_day_asked_in_the_past_tense_has_not_happened() -> None:
    from datetime import date

    from argus.lui.honesty import future_date_asked_as_past

    assert future_date_asked_as_past("what was NVDA's price on 2030-01-01",
                                     date(2026, 9, 30)) == date(2030, 1, 1)
    assert future_date_asked_as_past("what was NVDA's price on 2024-01-01",
                                     date(2026, 9, 30)) is None


def test_a_named_day_is_a_window_on_the_record() -> None:
    from argus.lui.question import resolve_window

    window = resolve_window("what did we do on 2026-09-20", now=NOW)
    assert window is not None and window.start == datetime(2026, 9, 20, tzinfo=UTC)
    assert window.end == datetime(2026, 9, 21, tzinfo=UTC)
    assert resolve_window("decisions on Sep 20", now=NOW) == window
