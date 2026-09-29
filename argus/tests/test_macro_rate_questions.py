"""A rate question is answered in the direction it asks, and a named year by that year's record
(`lui/research/macro.py`; stranger QA of 2026-09-29: a 2022-style hike was illustrated with a
cut)."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from argus.lui.research import macro
from argus.market import equity_history


@pytest.mark.parametrize("text", [
    "how would my book perform in a 2022-style rate hike", "what if the Fed hikes again",
    "what happens if rates rise", "rising yields and my book", "if the Fed raises rates",
])
def test_a_question_about_rates_going_up_is_read_as_up(text: str) -> None:
    assert macro.RATES_UP.search(text)


@pytest.mark.parametrize("text", [
    "what if the Fed cuts", "how does my book react to rates", "what is the 10-year yield",
])
def test_other_rate_questions_are_not(text: str) -> None:
    assert not macro.RATES_UP.search(text)


def _days(pairs: list[tuple[str, float]]) -> list[equity_history.Day]:
    return [equity_history.Day(day=date.fromisoformat(d), open=c, close=c) for d, c in pairs]


def test_a_named_year_is_answered_by_what_the_book_did_that_year(
        monkeypatch: pytest.MonkeyPatch) -> None:
    history = {
        "NVDA": _days([("2021-12-30", 100.0), ("2021-12-31", 100.0), ("2022-12-30", 50.0)]),
        "AAPL": _days([("2021-12-31", 100.0), ("2022-06-01", 90.0), ("2022-12-30", 75.0)]),
    }
    monkeypatch.setattr(equity_history, "daily", lambda ticker: history[ticker])
    monkeypatch.setattr(macro, "_fred_year", lambda series, year: (
        ("2021-12-31", 1.52), ("2022-12-30", 3.88)))
    found = macro._book_year_line({"NVDAUSDT": 0.5, "AAPLUSDT": 0.5}, 2022)
    assert found is not None
    line, source = found
    assert "In 2022 the 10-year went from 1.52% to 3.88% (+236bp, FRED)" in line
    assert "returned -37.5% (NVDA -50.0%, AAPL -25.0%" in line
    assert "not a forecast" in line and "FRED DGS10" in source.ref


def test_a_holding_with_no_history_that_year_is_named_not_hidden(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def daily(ticker: str) -> Any:
        if ticker == "COIN":
            raise equity_history.HistoryError("no bars")
        return _days([("2021-12-31", 100.0), ("2022-12-30", 80.0)])

    monkeypatch.setattr(equity_history, "daily", daily)
    monkeypatch.setattr(macro, "_fred_year", lambda series, year: None)
    found = macro._book_year_line({"NVDAUSDT": 0.7, "COINUSDT": 0.3}, 2022)
    assert found is not None
    assert "COIN has no 2022 history and is left out" in found[0]
    assert "covers 70% of the book" in found[0]


def test_too_little_history_gives_no_line(monkeypatch: pytest.MonkeyPatch) -> None:
    def daily(ticker: str) -> Any:
        raise equity_history.HistoryError("no bars")

    monkeypatch.setattr(equity_history, "daily", daily)
    assert macro._book_year_line({"NVDAUSDT": 1.0}, 2022) is None
