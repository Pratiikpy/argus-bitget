"""'Is the US market open?' is answered from the clock, holidays included."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from argus.lui.research import SESSION_QUESTION, session_status


@pytest.mark.parametrize("text", [
    "is the US market open right now?", "when does the stock market open",
    "美股现在开盘了吗", "The US stock market is open right now", "are US markets closed today?",
])
def test_session_questions_are_recognised(text: str) -> None:
    assert SESSION_QUESTION.search(text)


@pytest.mark.parametrize("text", [
    "is NVDA overbought?", "what's the sentiment on BTC", "is the market too greedy",
])
def test_other_questions_are_not(text: str) -> None:
    assert not SESSION_QUESTION.search(text)


def test_open_closed_weekend_and_holiday() -> None:
    assert "is open at 15:00 UTC" in session_status(datetime(2026, 9, 24, 15, tzinfo=UTC))[0][0]
    shut = session_status(datetime(2026, 9, 25, 4, 35, tzinfo=UTC))[0][0]
    assert "closed at 04:35 UTC — it is outside regular hours" in shut
    assert "opens Fri 25 Sep 13:30 UTC, in 8h 55m" in shut
    assert "it is the weekend" in session_status(datetime(2026, 9, 26, 15, tzinfo=UTC))[0][0]
    thanksgiving = session_status(datetime(2026, 11, 26, 15, tzinfo=UTC))[0][0]
    assert "it is a US market holiday" in thanksgiving and "Fri 27 Nov 14:30 UTC" in thanksgiving
