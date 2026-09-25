"""Multi-part questions (`lui/multistep.py`), offline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from argus.lui import multistep
from argus.lui.research import ResearchKind


def _kinds(question: str, book: str = "") -> list[str] | None:
    found = multistep.parts(question, book)
    return None if found is None else [p.request.kind.value for p in found]


def test_three_parts_each_to_its_engine() -> None:
    found = multistep.parts("Is NVDA overbought, and what would adding 20% of it do to my book? "
                            "Then how do I split a $50k buy?", "40% MSFT, 30% AAPL, 30% GOOGL")
    assert found is not None
    assert [p.request.kind for p in found] == [ResearchKind.TECHNICALS, ResearchKind.IMPACT,
                                               ResearchKind.EXECUTION]
    assert found[1].request.symbols[0] == "NVDAUSDT" and found[1].inherited == "NVDAUSDT"
    assert float(found[2].request.notional) == pytest.approx(50_000)


@pytest.mark.parametrize("question", [
    "what if the nasdaq drops 10%? I hold 60% NVDA 40% AAPL",
    "I hold 50% NVDA, 50% AAPL — what does adding 20% TSLA do to my risk?",
    "compare NVDA and AMD risk", "is TSLA riskier than NVDA",
    "how many shares outstanding does apple have? is that normal?",
    "take profit for a TSLA long? and where should my stop go?",
])
def test_one_question_is_left_whole(question: str) -> None:
    assert multistep.parts(question) is None


def test_the_book_stated_earlier_carries_to_a_later_part() -> None:
    found = multistep.parts("I hold 40% NVDA 60% AAPL. what if the nasdaq drops 10%? "
                            "and how do I hedge it?")
    assert found is not None and [p.request.kind for p in found] == [ResearchKind.STRESS,
                                                                     ResearchKind.HEDGE]
    assert found[1].request.book == {"NVDAUSDT": 0.4, "AAPLUSDT": 0.6}


def test_two_noun_phrases_with_their_own_names() -> None:
    assert _kinds("BTC funding rate and ETH open interest") == ["quote", "sentiment"]


def test_different_horizons_on_one_name_are_two_parts() -> None:
    assert _kinds("should I hold NVDA for a week? and what's the weekend gap risk on it?") == [
        "analogue", "analogue"]


@dataclass
class _Result:
    lines: list[str]
    sources: list[Any] = field(default_factory=list)
    refused: bool = False


def test_the_answer_leads_with_every_part_and_keeps_one_bold_lead() -> None:
    found = multistep.parts("where is NVDA trading? and when does it report earnings?")
    assert found is not None

    def run(text: str, request: Any) -> _Result:
        return _Result(lines=[f"Actionable: {request.kind.value} for {text}", "detail",
                              "Data: somewhere."])

    lines, _, unread = multistep.answer("q", found, run)
    assert lines[0].startswith("Actionable: your question has 2 parts") and unread == 0
    assert lines[1].startswith("1. Quote") and lines[2].startswith("2. Fundamentals")
    assert sum(line.startswith("Actionable") for line in lines) == 1
    assert lines[-1].startswith("Data:")


def test_a_fifth_part_is_named_not_dropped() -> None:
    found = [multistep.Part(text=f"part {i}", request=type("R", (), {
        "kind": ResearchKind.QUOTE})()) for i in range(6)]

    def run(text: str, request: Any) -> _Result:
        return _Result(lines=["Actionable: x"])

    lines, _, _ = multistep.answer("q", found, run)
    assert any(line.startswith("Assumed: only the first 4 parts") and "part 5" in line
               for line in lines)
