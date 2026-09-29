"""A hedge question that names only the hedges is measured against the S&P 500, and says so."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from argus.lui.research import dispatch
from argus.lui.research.parse import detect


def _request(text: str) -> Any:
    request = detect(text)
    assert request is not None
    return request


def test_hedges_named_without_holdings_are_measured_against_the_market(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Judge audit, 2026-09-29: the README's "Should I hedge with gold or with TLT?" was refused
    for want of holdings when asked verbatim."""
    seen: dict[str, Any] = {}

    def plan(book: dict[str, float], value: Decimal, raw: str) -> Any:
        seen["book"] = book
        return ["Bottom line: measured."], [], {}

    monkeypatch.setattr(dispatch, "_hedge_plan", plan)
    text = "Should I hedge with gold or with TLT?"
    answer = dispatch.run(text, _request(text))
    assert not answer.refused
    assert seen["book"] == {dispatch.DEFAULT_HEDGED_BOOK: 1.0}
    assert any("no holdings were stated" in line and "S&P 500" in line for line in answer.lines)


def test_a_named_holding_is_still_the_book(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def plan(book: dict[str, float], value: Decimal, raw: str) -> Any:
        seen["book"] = book
        return ["Bottom line: measured."], [], {}

    monkeypatch.setattr(dispatch, "_hedge_plan", plan)
    text = "should I hedge my NVDA with gold or with TLT?"
    dispatch.run(text, _request(text))
    assert seen["book"] == {"NVDAUSDT": 1.0}
