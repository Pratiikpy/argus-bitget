"""Plain definitions with a live reading (`lui/concepts.py`)."""

from __future__ import annotations

import pytest

from argus.lui import concepts


@pytest.mark.parametrize(("text", "named", "concept"), [
    ("explain what RSI means like I'm new to trading", (), "RSI"),
    ("what does funding rate mean", (), "funding rate"),
    ("what is open interest", (), "open interest"),
    ("what is a perp", (), "perpetual contract"),
    ("what is NVDA RSI", ("NVDAUSDT",), None),
    ("what is the sharpe", (), None),
    ("what's my max drawdown", (), None),
    ("what is the funding rate on BTC", ("BTCUSDT",), None),
])
def test_only_a_question_about_what_a_word_means_is_taken(
        text: str, named: tuple[str, ...], concept: str | None) -> None:
    """A judge's audit, 2026-09-29: "explain what RSI means" was answered with a decision trace;
    the record's own questions ("what is the sharpe") must still reach the record."""
    found = concepts.concept_asked(text, named)
    assert (found.name if found else None) == concept


def test_the_definition_leads_and_the_live_line_is_the_engines(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import argus.lui.research as research

    monkeypatch.setattr(research, "run", lambda q, r: SimpleNamespace(
        lines=["Bottom line: momentum turning down.", "RSI(14, 4h) 42.5 — neutral."], sources=[]))
    rsi = next(c for c in concepts.CONCEPTS if c.name == "RSI")
    lines, _, data = concepts.answer(rsi, None)
    assert lines[0].startswith("Bottom line: RSI, the relative strength index")
    assert lines[2] == ("Right now on BTC (an example; name a contract for its own reading): "
                        "RSI(14, 4h) 42.5 — neutral.")
    assert data["example_symbol"] == "BTCUSDT"
