"""A beginner's first-money question (`lui/research/starter.py`)."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from argus.lui.research import starter


@pytest.mark.parametrize(("text", "amount"), [
    ("I have 5000 dollars, what should I do", 5000.0),
    ("i have 8k to invest", 8000.0),
    ("I got $2,500 where should I put it", 2500.0),
    ("I have 3 kids, what should I do", None),
])
def test_the_sum_is_read_only_when_it_is_money(text: str, amount: float | None) -> None:
    """A first-time-user audit, 2026-09-29: "I have 5000 dollars, what should I do" was refused."""
    assert starter.amount_of(text) == amount


def test_the_answer_gives_no_advice_and_shows_the_year_on_that_sum() -> None:
    def daily(ticker: str) -> list[Any]:
        start = date(2025, 10, 1)
        closes = [100.0, 80.0, 120.0] + [110.0] * 30
        return [SimpleNamespace(day=start + timedelta(days=i), close=c, open=c)
                for i, c in enumerate(closes)]

    lines, _, data = starter.answer("I have 5000 dollars, what should I do, and what is a stop "
                                    "loss", today=date(2026, 9, 30), daily=daily)
    assert lines[0].startswith("Bottom line: this console will not tell you what to buy")
    assert lines[1] == ("In the Nasdaq-100 (QQQ): $5,000 would be $5,500 a year later (+10%); at "
                        "its lowest it was worth $4,000, and its worst fall from a high on the way "
                        "was 20%.")
    assert any(line.startswith("You also asked about stop loss:") for line in lines)
    assert data["stake"] == 5000.0
