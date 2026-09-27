"""The rules deciding between the planner's reading and the patterns' (`lui/arbiter.py`)."""

from __future__ import annotations

from typing import Any

from argus.lui import arbiter
from argus.lui.research import ResearchKind, ResearchRequest
from argus.lui.research import detect as detect_research

AUDIT: dict[str, Any] = {"attempted": True, "applied": True, "detail": "model"}


class _Planner:
    """A language-model stand-in: answers ``complete_json`` with one fixed plan."""

    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan = plan

    def complete_json(self, messages: list[dict[str, str]], **_: Any) -> dict[str, Any]:
        return dict(self.plan)


def test_a_leverage_reading_with_no_leverage_named_falls_to_the_patterns() -> None:
    text = "Long MSTR perp into earnings — funding looks cheap"
    planned = ResearchRequest(kind=ResearchKind.LEVERAGE, symbols=("MSTRUSDT",))
    patterned = detect_research(text)
    got, audit = arbiter.leverage_needs_leverage(text, "", planned, patterned, AUDIT)
    assert got is None or got.kind is not ResearchKind.LEVERAGE
    assert audit["detail"] == "a leverage reading with no leverage named was dropped"
    kept, same = arbiter.leverage_needs_leverage("MSTR at 10x, where is liquidation?", "",
                                                 planned, patterned, AUDIT)
    assert kept is planned and same is AUDIT


def test_an_add_the_model_read_as_a_holding_is_the_patterns_add() -> None:
    text = "nvda 40%, msft 20%, cash rest — want to add 5k of coin"
    book = {"NVDAUSDT": 0.4, "MSFTUSDT": 0.2}
    planned = ResearchRequest(kind=ResearchKind.IMPACT, symbols=("NVDAUSDT",), book=book)
    patterned = ResearchRequest(kind=ResearchKind.IMPACT, symbols=("COINUSDT",), book=book)
    got, audit = arbiter.the_add_is_not_a_holding(text, "", planned, patterned, AUDIT)
    assert got is not None and got.symbols[0] == "COINUSDT"
    assert audit["detail"] == "the model's add was a holding; the patterns' add kept"
    outside = ResearchRequest(kind=ResearchKind.IMPACT, symbols=("TSLAUSDT",), book=book)
    assert arbiter.the_add_is_not_a_holding(text, "", outside, patterned, AUDIT)[0] is outside


def test_the_patterns_specific_reading_stands_over_a_different_kind() -> None:
    text = "order book depth on NVDA"
    patterned = detect_research(text)
    assert patterned is not None
    planned = ResearchRequest(kind=ResearchKind.QUOTE, symbols=("NVDAUSDT",))
    got, audit = arbiter.the_specific_reading_stands(text, "", planned, patterned, AUDIT)
    assert got is not None and got.kind is patterned.kind
    assert audit["detail"] == "the patterns' specific reading kept over the model's"
    # the same reading from both is left as the planner's
    same, untouched = arbiter.the_specific_reading_stands(text, "", patterned, patterned, AUDIT)
    assert same is patterned and untouched is AUDIT


def test_not_research_reads_both_planners_and_only_a_confident_language_model() -> None:
    assert arbiter.not_research({"model": {"why": "kind model: refuse at 0.31"}}) == ("refuse",
                                                                                      0.31)
    assert arbiter.not_research({"model": {"kind": "none", "confidence": 0.95}}) == ("refuse",
                                                                                     0.95)
    assert arbiter.not_research({"model": {"kind": "record", "confidence": 0.9}}) == ("record",
                                                                                      0.9)
    assert arbiter.not_research({"model": {"kind": "none", "confidence": 0.5}}) is None
    assert arbiter.not_research("not a dict") is None


def test_arbitrate_takes_the_planners_reading_and_says_so() -> None:
    planner = _Planner({"kind": "technicals", "symbols": ["NVDA"], "confidence": 0.9,
                        "why": "an indicator question"})
    reading = arbiter.arbitrate("is NVDA overbought?", book="", model=planner,
                                instruction=False, desk_first=False, audit=dict(AUDIT))
    assert reading.via == "research-model"
    assert reading.request is not None and reading.request.kind is ResearchKind.TECHNICALS


def test_arbitrate_refuses_a_future_price_whatever_the_planner_read() -> None:
    planner = _Planner({"kind": "impact", "symbols": ["BTC"], "confidence": 0.9, "why": "x"})
    reading = arbiter.arbitrate("what will BTC's exact price be a year from now?", book="",
                                model=planner, instruction=False, desk_first=False,
                                audit=dict(AUDIT))
    assert reading.via == "forecast-refusal" and reading.request is None


def test_a_confident_not_research_verdict_binds_the_patterns() -> None:
    planner = _Planner({"kind": "none", "confidence": 0.95, "why": "a joke request"})
    reading = arbiter.arbitrate("tell me a joke about NVDA", book="", model=planner,
                                instruction=False, desk_first=False, audit=dict(AUDIT))
    assert reading.kind_said == ("refuse", 0.95)
    assert reading.via == ""
