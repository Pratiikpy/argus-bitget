"""Directional questions get the contract's record at that horizon, never a prediction."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from argus.desk.odds import directional_odds, wilson
from argus.lui import research
from argus.lui.research import ResearchKind, _detect, _horizon, pattern_reading_wins
from argus.market import history, universe

START = datetime(2025, 1, 3, 16, tzinfo=UTC)  # a Friday


@pytest.fixture(autouse=True)
def frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("live fetch disabled in tests")

    monkeypatch.setattr(research, "_fetch_live", _fail)
    monkeypatch.setattr(history, "fetch", _fail)
    monkeypatch.setattr(universe, "_fetch_live", _fail)
    monkeypatch.setattr(universe, "_CACHE", None)


def _closes(levels: list[float]) -> list[tuple[datetime, float]]:
    return [(START + timedelta(days=i), v) for i, v in enumerate(levels)]


def test_wilson_matches_the_textbook_interval() -> None:
    low, high = wilson(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-4) and high == pytest.approx(0.5962, abs=1e-4)


def test_overlapping_windows_are_counted_as_independent_draws() -> None:
    levels = [100.0 * (1.01 if i % 2 else 0.99) ** (i % 3) for i in range(200)]
    odds = directional_odds(_closes(levels), 4, cost_bps=12)
    assert odds is not None
    assert odds.windows == 196 and odds.independent == pytest.approx(49.0)


def test_a_steady_rise_is_a_lean_up_and_clears_the_cost() -> None:
    levels = [100.0 * 1.002 ** i for i in range(120)]
    odds = directional_odds(_closes(levels), 2, cost_bps=12)
    assert odds is not None
    assert odds.higher_share == 1.0 and not odds.coin_flip
    assert odds.cleared_share == 1.0  # +40bps every two days against a 12bps hurdle
    short = directional_odds(_closes(levels), 2, cost_bps=12, side="short")
    assert short is not None and short.cleared_share == 0.0


def test_the_weekend_reads_friday_to_monday_closes_only() -> None:
    levels = []
    for i in range(70):
        day = (START + timedelta(days=i)).weekday()
        levels.append(100.0 + (5.0 if day in (5, 6, 0) else 0.0) + i * 0.0)
    odds = directional_odds(_closes(levels), 1, cost_bps=12, weekend=True)
    assert odds is not None
    assert odds.windows == 10 and odds.independent == 10  # ten Fridays in 70 days
    assert odds.last_move_bps is None  # no "after a move like the last one" on weekends


def test_too_little_history_is_none_not_a_number() -> None:
    assert directional_odds(_closes([100.0, 101.0]), 1, cost_bps=12) is None


@pytest.mark.parametrize(("text", "hours", "weekend", "side"), [
    ("Will MSTR be higher in 48 hours?", 48, False, "long"),
    ("will nvda go up tomorrow", 24, False, "long"),
    ("Will BTC drop over the weekend?", 72, True, "short"),
    ("is TSLA going to close green next week", 168, False, "long"),
    ("Will ETH be higher in 4 hours?", 4, False, "long"),
    ("will gold go down in 3 days", 72, False, "short"),
])
def test_a_directional_question_becomes_a_record_at_its_horizon(
        text: str, hours: int, weekend: bool, side: str) -> None:
    request = _detect(text)
    assert request is not None and request.kind is ResearchKind.ANALOGUE
    assert (request.horizon_hours, request.weekend, request.side) == (hours, weekend, side)
    assert pattern_reading_wins(request, text)


@pytest.mark.parametrize("text", [
    "what will BTC be next week", "where will NVDA trade tomorrow", "predict SOL for next week",
])
def test_a_price_level_is_still_not_answered_as_a_direction(text: str) -> None:
    request = _detect(text)
    assert request is None or request.horizon_hours is None


def test_an_unstated_horizon_is_said() -> None:
    assert _horizon("will NVDA go up?") == (24, False, "no horizon was stated, so the next 24 "
                                                      "hours are read")


def test_the_stop_line_reads_the_bars_lows_and_highs_not_the_closes() -> None:
    closes = _closes([100.0] * 40)
    # every day dips to 97 and spikes to 104 intraday while closing flat
    extremes = [(97.0, 104.0)] * 40
    long = directional_odds(closes, 2, cost_bps=12, extremes=extremes)
    short = directional_odds(closes, 2, cost_bps=12, side="short", extremes=extremes)
    assert long is not None and long.adverse_p90_bps == pytest.approx(-300.0)
    assert short is not None and short.adverse_p90_bps == pytest.approx(400.0)
    assert directional_odds(closes, 2, cost_bps=12) is not None
    no_extremes = directional_odds(closes, 2, cost_bps=12)
    assert no_extremes is not None and no_extremes.adverse_p90_bps is None
