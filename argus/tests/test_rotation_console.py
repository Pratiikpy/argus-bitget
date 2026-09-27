"""The console's risk-on/off rotation answer (`lui/rotation.py`), on constructed histories."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from argus.lui import rotation


def _series(end: date, monthly_growth: float, days: int = 420) -> list[Any]:
    start = end - timedelta(days=days)
    out, price = [], 100.0
    for i in range(days + 1):
        day = start + timedelta(days=i)
        price *= (1 + monthly_growth) ** (1 / 30)
        out.append(SimpleNamespace(day=day, close=price))
    return out


def _daily(growth: dict[str, float], end: date) -> Any:
    def daily(ticker: str) -> list[Any]:
        return _series(end, growth[ticker])
    return daily


GROWTH = {"NVDA": 0.03, "AAPL": 0.01, "QQQ": 0.02, "BTC-USD": 0.05, "ETH-USD": 0.04,
          "GLD": -0.01}


def _book(text: str) -> dict[str, float]:
    return {"NVDAUSDT": 0.5, "MSFTUSDT": 0.5} if text else {}


@pytest.mark.parametrize("text", ["should I be risk-on or risk-off right now?",
                                  "rotate between stocks, crypto and gold?",
                                  "which asset class should I hold, stocks or crypto or gold?"])
def test_rotation_questions_are_recognised(text: str) -> None:
    assert rotation.asks_for_rotation(text)


@pytest.mark.parametrize("text", ["is NVDA overbought?", "what does a 10% drop do to my book"])
def test_other_questions_are_not(text: str) -> None:
    assert not rotation.asks_for_rotation(text)


def test_the_answer_names_the_regime_the_scores_and_the_history_it_used() -> None:
    today = date.today()
    lines, sources, data = rotation.rotation("risk on or off?", "", daily=_daily(GROWTH, today),
                                             parse_book=_book)
    assert lines[0].startswith("Bottom line: partial risk-off — 1 of 6 assets negative")
    assert "XAU -" in lines[1] and "BTC +" in lines[1]
    assert "No book is saved" in lines[2]
    assert "BTC on BTC-USD" in lines[-1] and "not a forecast" in lines[-1]
    assert data["rotation"]["is_neg"] == 1
    assert {s.ref for s in sources} >= {"Yahoo Finance daily history"}


def test_a_saved_book_gets_its_trades_and_names_what_the_rule_ignores() -> None:
    lines, _, _ = rotation.rotation("risk on or off?", "50% NVDA 50% MSFT",
                                    daily=_daily(GROWTH, date.today()), parse_book=_book)
    trades = next(line for line in lines if line.startswith("From your book:"))
    assert "NVDA 50% → " in trades and "bps of the book at the taker fee" in trades
    assert any(line.startswith("Held outside the rule's universe") and "MSFT" in line
               for line in lines)


def test_a_short_history_is_refused_clause_by_clause_not_answered_in_part() -> None:
    def daily(ticker: str) -> list[Any]:
        return _series(date.today(), GROWTH[ticker], days=200 if ticker == "GLD" else 420)

    lines, _, data = rotation.rotation("risk on or off?", "", daily=daily, parse_book=_book)
    assert data == {"refused": True}
    assert lines[0].startswith("Bottom line: the rotation rule will not give a reading today")
    assert any("XAUUSDT" in line for line in lines[1:])
