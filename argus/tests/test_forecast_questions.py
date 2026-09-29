"""A question for a future price is recognised and declined honestly (`lui/research/parse.py`
`price_forecast_asked`; judge audit of 2026-09-29: "What will TSLA's stock price be next Friday?"
was read as a routing miss, because the possessive broke the pattern)."""

from __future__ import annotations

import pytest

from argus.lui.research.parse import price_forecast_asked


@pytest.mark.parametrize("text", [
    "What will TSLA's stock price be next Friday?",
    "What will TSLA\u2019s stock price be next Friday?",
    "TSLA price by friday",
    "what will NVDA be worth next week",
    "where will BTC be tomorrow",
])
def test_a_future_price_is_a_forecast_question(text: str) -> None:
    assert price_forecast_asked(text)


@pytest.mark.parametrize("text", [
    "what was the price on monday", "where is TSLA trading", "did that pattern predict anything",
    "what is the price next to the book",
])
def test_a_present_or_past_price_is_not(text: str) -> None:
    assert not price_forecast_asked(text)
