"""A question about another person's or fund's holdings is refused, not answered from this desk's
own ledger (QA inventory, 2026-09-28: "What is Elon Musk's position in TSLA?" got "No open
positions")."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from argus.lui.question import Intent, classify, someone_elses_holdings

NOW = datetime(2026, 9, 28, 12, tzinfo=UTC)


@pytest.mark.parametrize(("text", "holder"), [
    ("What is Elon Musk's position in TSLA?", "Elon Musk"),
    ("what is Cathie Wood's portfolio", "Cathie Wood"),
    ("Does Berkshire still own AAPL?", "Berkshire"),
    ("Is Michael Burry short NVDA?", "Michael Burry"),
])
def test_a_named_holder_is_found(text: str, holder: str) -> None:
    assert someone_elses_holdings(text) == holder


@pytest.mark.parametrize("text", [
    "What's our position in NVDA?", "What's the desk's exposure?", "Is ARGUS long NVDA?",
    "What's my position", "what positions are open", "Are we long TSLA?",
    "What's NVDA's exposure to the Nasdaq?",
])
def test_the_desk_and_the_trader_are_not_someone_else(text: str) -> None:
    assert someone_elses_holdings(text) is None


def test_the_misread_question_is_now_refused_with_its_reason() -> None:
    q = classify("What is Elon Musk's position in TSLA?", now=NOW)
    assert q.intent is Intent.UNSUPPORTED
    assert "not Elon Musk's positions" in q.reason
    assert classify("What's our position in NVDA?", now=NOW).intent is Intent.POSITION
