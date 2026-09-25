"""The research fan-out (`lui/fanout.py`), offline: scripted researchers, a scripted supervisor,
no network and no model."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from argus.lui import fanout
from argus.lui.answer import Answer, Source
from argus.lui.research import ResearchKind, ResearchRequest, _question

K = ResearchKind
CLASSES = {"NVDAUSDT": "equity", "BTCUSDT": "crypto", "XAUUSDT": "commodity",
           "AMDUSDT": "equity"}


def _classify(symbol: str) -> str:
    return CLASSES.get(symbol, "crypto")


def _request(kind: ResearchKind, *symbols: str, book: dict[str, float] | None = None
             ) -> ResearchRequest:
    return ResearchRequest(kind=kind, symbols=symbols, book=book or {})


def _answer(text: str, request: ResearchRequest, lines: list[str],
            sources: list[Source] | None = None, *, refused: bool = False, reason: str = "",
            reached: tuple[str, ...] = ()) -> Answer:
    return Answer(question=_question(text, request), lines=lines, sources=sources or [],
                  refused=refused, reason=reason,
                  data={"coverage": {"reached": list(reached), "did_not_answer": {}}})


def _plan(kind: ResearchKind, *symbols: str, **kw: Any) -> list[str]:
    return [u.id for u in fanout.plan_units(_request(kind, *symbols, **kw), classify=_classify)]


# --- the plan -----------------------------------------------------------------------------------


def test_the_table_plans_only_what_each_engine_can_answer_for_the_name() -> None:
    # an equity has a company behind it and no crypto crowd; a coin the other way round
    assert _plan(K.NEWS, "NVDAUSDT") == ["technicals:NVDAUSDT", "fundamentals:NVDAUSDT",
                                         "analogue:NVDAUSDT"]
    assert _plan(K.NEWS, "BTCUSDT") == ["technicals:BTCUSDT", "sentiment:BTCUSDT",
                                        "analogue:BTCUSDT"]
    # only the twelve traded names have an event study
    assert "event:NVDAUSDT" in _plan(K.FUNDAMENTALS, "NVDAUSDT")
    assert _plan(K.FUNDAMENTALS, "AMDUSDT") == ["news:AMDUSDT", "technicals:AMDUSDT"]


def test_a_fund_gets_no_company_researcher() -> None:
    # QQQ, TQQQ and SQQQ are funds in the Instrument Master: no earnings, consensus or holders
    assert [fanout.asset_class(s) for s in ("QQQUSDT", "TQQQUSDT", "SQQQUSDT", "NVDAUSDT")] == [
        "fund", "fund", "fund", "equity"]
    plan = [u.id for u in fanout.plan_units(_request(K.NEWS, "QQQUSDT"))]
    assert plan == ["technicals:QQQUSDT", "analogue:QQQUSDT"]


def test_book_researchers_read_the_book_and_need_one() -> None:
    book = {"NVDAUSDT": 0.6, "AAPLUSDT": 0.4}
    assert _plan(K.STRESS, "NVDAUSDT", "AAPLUSDT", book=book) == ["hedge:book", "macro:book"]
    units = fanout.plan_units(_request(K.STRESS, *book, book=book), classify=_classify)
    assert units[0].request.book == book and units[0].request.kind is K.HEDGE
    # no book: no book researcher is planned, and the name's own macro read remains
    assert _plan(K.HEDGE, "NVDAUSDT") == ["macro:NVDAUSDT"]


def test_a_comparison_reads_both_names_and_the_cap_holds() -> None:
    plan = _plan(K.COMPARE, "NVDAUSDT", "AMDUSDT")
    assert plan == ["technicals:NVDAUSDT", "technicals:AMDUSDT", "news:NVDAUSDT"]
    assert len(plan) == fanout.MAX_UNITS - 1


def test_kinds_without_a_plan_are_answered_alone() -> None:
    assert _plan(K.VENUE, "NVDAUSDT") == []
    assert fanout.fan_out("what is bitget's rtoken offering", _request(K.VENUE, "NVDAUSDT"),
                          run=lambda t, r: None, classify=_classify) is None


def test_each_researcher_reads_its_own_topic_never_the_question() -> None:
    units = fanout.plan_units(_request(K.NEWS, "NVDAUSDT"), classify=_classify)
    assert all("NVDA" in u.topic and "?" not in u.topic for u in units)
    assert units[0].request.parsed_by == "fanout" and units[0].request.symbols == ("NVDAUSDT",)


# --- compression --------------------------------------------------------------------------------


def test_compression_keeps_the_lead_and_one_figure_line_verbatim() -> None:
    answer = _answer("q", _request(K.TECHNICALS, "NVDAUSDT"), [
        "Actionable: momentum turning down.", "RSI(14, 4h) 45.9 — neutral.",
        "MACD histogram -0.12.", "Assumed: 30 days.", "Sources reached: 4 of 4 answered.",
        "Data: bitget-signal."])
    lines, why = fanout.compress(answer)
    assert lines == ["Momentum turning down.", "RSI(14, 4h) 45.9 — neutral."] and why == ""


def test_a_long_lead_is_kept_whole_and_alone() -> None:
    lead = "Actionable: " + "a very long lead that says 1.5% " * 20
    lines, _ = fanout.compress(_answer("q", _request(K.NEWS, "NVDAUSDT"),
                                       [lead, "second line 2.0%"]))
    assert lines == [lead.removeprefix("Actionable: ")[:1].upper()
                     + lead.removeprefix("Actionable: ")[1:]]


def test_verbatim_violations_catch_a_changed_number() -> None:
    finding = fanout.Finding(unit=fanout.plan_units(_request(K.NEWS, "NVDAUSDT"),
                                                    classify=_classify)[0],
                             answered=True, lines=["RSI 45.9 and MACD -0.12"],
                             full=["Actionable: RSI 45.9.", "MACD -0.12 today."])
    assert fanout.verbatim_violations(finding) == []
    finding.lines = ["RSI 46.9 and MACD -0.12"]
    assert fanout.verbatim_violations(finding) == ["46.9"]


# --- running and merging ------------------------------------------------------------------------


def _runner(*, fail: str = "", refuse: str = "", slow: str = "", seen: list[str] | None = None
            ) -> Any:
    shared = Source(kind="venue", ref="bitget /api/v3/market/candles", detail="shared")

    def run(text: str, request: ResearchRequest) -> Answer:
        kind = request.kind.value
        if seen is not None:
            seen.append(text)
        if kind == fail:
            raise ConnectionError("feed down")
        if kind == slow:
            time.sleep(5.0)
        if kind == refuse:
            return _answer(text, request, ["no data"], refused=True, reason="the Skill was dark")
        own = Source(kind="computation", ref=f"argus.{kind}", detail=kind)
        return _answer(text, request, [f"Actionable: {kind} lead 1.25%.", f"{kind} detail 7.",
                                       "Sources reached: 2 of 2 answered.",
                                       f"Data: {kind} data."],
                       [shared, own], reached=(f"{kind} feed", "Bitget candles"))

    return run


def test_the_merged_answer_keeps_the_question_first_and_cites_each_finding() -> None:
    seen: list[str] = []
    question = "what's the news on NVDA today"
    merged = fanout.fan_out(question, _request(K.NEWS, "NVDAUSDT"), run=_runner(seen=seen),
                            classify=_classify)
    assert merged is not None and not merged.refused
    assert merged.lines[0] == "Actionable: news lead 1.25%."  # the question's own lead leads
    assert merged.lines[-1] == "Data: news data."  # and its Data line still ends the answer
    assert question in seen and len(seen) == 4  # the primary read the question; units, topics
    finding = next(line for line in merged.lines if line.startswith("Technicals (NVDA):"))
    # the shared candle source keeps its first number; the unit's own source is new
    assert finding.endswith("[1][3]")
    keys = [(s.kind, s.ref) for s in merged.sources]
    assert len(keys) == len(set(keys)) == 5
    data = merged.data["fanout"]
    assert data["merged"] and [f["answered"] for f in data["findings"]] == [True, True, True]
    # four researchers' own feeds and the candles they share, each counted once
    assert "Sources reached across all researchers: 5 of 5 answered." in merged.lines
    # the question's own coverage line is superseded, never printed beside the merged one
    assert not any(line.startswith("Sources reached: ") for line in merged.lines)


def test_a_failed_or_refused_researcher_costs_only_itself() -> None:
    merged = fanout.fan_out("news on NVDA", _request(K.NEWS, "NVDAUSDT"),
                            run=_runner(fail="technicals", refuse="fundamentals"),
                            classify=_classify)
    assert merged is not None
    assert "Not answered — Technicals (NVDA): ConnectionError: feed down." in merged.lines
    assert "Not answered — Company (NVDA): the Skill was dark." in merged.lines
    assert any(line.startswith("Own history (NVDA): Analogue lead") for line in merged.lines)
    assert any(line.startswith("Wider read: 3 independent researchers") and
               "1 answered, 2 did not" in line for line in merged.lines)


def test_a_researcher_past_the_deadline_is_reported_not_waited_for() -> None:
    started = time.perf_counter()
    merged = fanout.fan_out("news on NVDA", _request(K.NEWS, "NVDAUSDT"),
                            run=_runner(slow="analogue"), classify=_classify, deadline_s=0.3)
    # the slow researcher sleeps 5s; the answer returns long before it (margin for a loaded box)
    assert time.perf_counter() - started < 4.0
    assert merged is not None
    assert "Not answered — Own history (NVDA): did not finish within 0.3s." in merged.lines


def test_a_refused_question_is_returned_alone() -> None:
    merged = fanout.fan_out("news on NVDA", _request(K.NEWS, "NVDAUSDT"),
                            run=_runner(refuse="news"), classify=_classify)
    assert merged is not None and merged.refused
    assert merged.data["fanout"]["merged"] is False
    assert not any(line.startswith("Wider read") for line in merged.lines)


def test_a_unit_cannot_fan_out_again() -> None:
    inner: list[Any] = []
    lock = threading.Lock()

    def run(text: str, request: ResearchRequest) -> Answer:
        nested = fanout.fan_out(text, request, run=_runner(), classify=_classify)
        with lock:
            inner.append(nested)
        return _runner()(text, request)

    outer = fanout.fan_out("news on NVDA", _request(K.NEWS, "NVDAUSDT"), run=run,
                           classify=_classify)
    assert outer is not None and inner and all(n is None for n in inner)


# --- the model supervisor -----------------------------------------------------------------------


class _Supervisor:
    def __init__(self, reply: Any = None, error: Exception | None = None) -> None:
        self.reply, self.error, self.calls = reply, error, 0
        self.seen: list[dict[str, Any]] = []

    def complete_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.calls += 1
        self.seen = messages
        if self.error is not None:
            raise self.error
        return self.reply


def test_the_model_chooses_from_the_menu_and_nothing_else() -> None:
    client = _Supervisor({"units": [
        {"id": "event:NVDAUSDT", "why": "earnings season"},
        {"id": "made-up:NVDAUSDT", "why": "x"},
        {"id": "event:NVDAUSDT", "why": "again"},
        {"id": "macro:NVDAUSDT", "why": "rates"},
        {"id": "quote:NVDAUSDT", "why": "price"},
        {"id": "news:NVDAUSDT", "why": "one too many"}]})
    units, audit = fanout.plan_units_with_model("should I buy NVDA", _request(
        K.TECHNICALS, "NVDAUSDT"), client, classify=_classify)
    assert [u.id for u in units] == ["event:NVDAUSDT", "macro:NVDAUSDT", "quote:NVDAUSDT"]
    assert audit["rejected"] == ["made-up:NVDAUSDT"] and audit["overflow"] == ["news:NVDAUSDT"]
    assert client.calls == 1
    prompt = client.seen[-1]["content"]
    assert "technicals:NVDAUSDT" not in prompt  # the question's own engine is not on the menu
    assert "sentiment:NVDAUSDT" not in prompt  # nor an engine that cannot answer for an equity


@pytest.mark.parametrize("client", [
    _Supervisor(error=TimeoutError("gateway")),
    _Supervisor({"units": [{"id": "nonsense"}]}),
    _Supervisor(["not", "an", "object"]),
])
def test_a_supervisor_that_fails_falls_back_to_the_table(client: _Supervisor) -> None:
    request = _request(K.NEWS, "NVDAUSDT")
    units, audit = fanout.plan_units_with_model("news?", request, client, classify=_classify)
    assert [u.id for u in units] == _plan(K.NEWS, "NVDAUSDT")
    assert str(audit["planner"]).startswith("table")


def test_fan_out_with_a_model_supervisor_runs_its_choice() -> None:
    client = _Supervisor({"units": [{"id": "macro:NVDAUSDT", "why": "rates"}]})
    merged = fanout.fan_out("news on NVDA", _request(K.NEWS, "NVDAUSDT"), run=_runner(),
                            client=client, classify=_classify)
    assert merged is not None
    assert [f["unit"] for f in merged.data["fanout"]["findings"]] == ["macro:NVDAUSDT"]
    assert merged.data["fanout"]["planner"] == "model"
