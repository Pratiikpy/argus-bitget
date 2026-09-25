"""Defects found by reading a sample of 27 answers across every kind of question (2026-09-25).

Each is pinned by its mechanism: a stated risk budget read from the next holding, a resize that
dropped the weight it was given, a claimed move's size ignored, a premium measured as the overnight
move, a VaR question answered without VaR, and odds or analogue questions routed elsewhere.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from argus.lui import research
from argus.lui.research import ResearchKind


@pytest.mark.parametrize(("text", "budget"), [
    ("what's my max size in NVDA with a 25% risk budget, I hold 50% AAPL 50% MSFT", 0.25),
    ("risk budget of 20%", 0.20),
    ("15% max risk per name", 0.15),
    ("with a risk budget, I hold 50% AAPL", None),
])
def test_the_budget_is_the_number_beside_the_budget_words(text: str,
                                                          budget: float | None) -> None:
    assert research.parse_budget(text) == budget


def test_a_resize_keeps_the_resized_names_stated_weight() -> None:
    request = research._detect("resize NVDA to 20% in 40% NVDA 60% AAPL")
    assert request is not None and request.kind is ResearchKind.IMPACT
    assert dict(request.book) == {"NVDAUSDT": 0.4, "AAPLUSDT": 0.6}
    assert request.symbols == ("NVDAUSDT", "AAPLUSDT") and request.target == 0.2


def test_the_resize_example_the_console_suggests_is_itself_answerable() -> None:
    request = research._detect("what if I trim NVDA to 10%? I hold 30% NVDA, 40% AAPL, 30% MSFT")
    assert request is not None and request.target == 0.1
    assert request.book["NVDAUSDT"] == 0.3


@pytest.mark.parametrize(("text", "size"), [
    ("NVDA is up 5% today because of earnings", 5.0), ("NVDA rallied 8 percent", 8.0),
    ("why is NVDA up", None),
])
def test_a_claimed_moves_size_is_read(text: str, size: float | None) -> None:
    move = research._MOVE_CLAIM.search(text)
    assert move is not None
    assert research._stated_move_size(text, move.end()) == size


def test_a_price_level_is_not_a_move_claim() -> None:
    assert research._MOVE_CLAIM.search("NVDA is up to 230") is None


def test_a_var_question_on_a_book_is_a_stress_with_that_book() -> None:
    request = research._detect("what is the VaR of 50% BTC 50% ETH")
    assert request is not None and request.kind is ResearchKind.STRESS
    assert dict(request.book) == {"BTCUSDT": 0.5, "ETHUSDT": 0.5}
    assert research.pattern_reading_wins(request, "what is the VaR of 50% BTC 50% ETH")
    scaled = research._detect("expected shortfall for 30% NVDA 20% AAPL")
    assert scaled is not None and scaled.notes and "scaled to 100%" in scaled.notes[0]


@pytest.mark.parametrize(("text", "hours"), [
    ("what are the odds NVDA is higher in 24 hours", 24),
    ("chance BTC ends the week lower", 168),
    ("odds NVDA closes the month higher", 720),
])
def test_odds_questions_reach_the_odds_engine_at_their_horizon(text: str, hours: int) -> None:
    assert research._DIRECTIONAL.search(text)
    assert research._horizon(text)[0] == hours


def test_explicit_analogue_words_keep_the_analogue_engine() -> None:
    request = research._detect("analogues for the current BTC setup")
    assert request is not None and request.kind is ResearchKind.ANALOGUE
    assert research.pattern_reading_wins(request, "analogues for the current BTC setup")
    horizon_only = research._detect("where's the key support level on QQQ into next week")
    assert not (horizon_only is not None and horizon_only.kind is ResearchKind.ANALOGUE
                and research._ANALOGUE.search("where's the key support level on QQQ into next "
                                              "week"))


@pytest.mark.parametrize(("text", "routed"), [
    ("ETH open interest", True), ("how crowded is NVDA", True), ("BTC long/short ratio", True),
    ("is NVDA overbought?", False),
])
def test_open_interest_questions_go_to_the_positioning_answer(text: str, routed: bool) -> None:
    assert bool(research.OPEN_INTEREST_QUESTION.search(text)) is routed


def test_while_shut_the_premium_is_the_gap_at_the_close_not_the_overnight_move(
        monkeypatch: pytest.MonkeyPatch) -> None:
    close = datetime(2026, 9, 24, 20, tzinfo=UTC)
    monkeypatch.setattr(research, "_stock_last", lambda symbol, service: 224.58)
    monkeypatch.setattr(research, "_last_regular_close", lambda now: close)
    monkeypatch.setattr(research, "_perp_at_close", lambda symbol, at: 224.72)
    monkeypatch.setattr(research, "_yahoo_close", lambda symbol, at: 224.58)
    line, _source = research._premium_line("NVDAUSDT", research.Decimal("225.89"), False)
    assert "a 6.2bps premium at that close" in line
    assert "moved +0.52% since, which is the overnight move, not a premium" in line
    opened: Any = research._premium_line("NVDAUSDT", research.Decimal("225.89"), True)
    assert opened is not None and "58.3bps premium (both live)" in opened[0]
