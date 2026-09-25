"""The S24 evaluation's own machinery (`eval/research_depth.py`), offline: the independent
recomputation agrees with the desk where the desk is right and catches it where it is not, and the
Qwen recorder writes before it parses and never exceeds its allowance. No model is called."""

from __future__ import annotations

import json
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from argus.desk.portfolio import align, beta, rebalance, tail_contributions, worst_window
from argus.eval import research_depth as rd
from argus.llm.qwen import BudgetExhausted, QwenError

START = datetime(2026, 8, 1, tzinfo=UTC)


def _is_open(at: datetime) -> bool:
    return at.weekday() < 5 and 14 <= at.hour < 21


def _raw(seed: int = 3, hours: int = 400) -> dict[str, dict[datetime, float]]:
    rng = random.Random(seed)
    stamps = [START + timedelta(hours=i) for i in range(hours)]
    qqq = [rng.gauss(0, 0.004) for _ in stamps]
    out = {"QQQUSDT": dict(zip(stamps, qqq, strict=True))}
    for symbol, b in (("NVDAUSDT", 1.5), ("AAPLUSDT", 0.7)):
        out[symbol] = {t: b * q + rng.gauss(0, 0.004) for t, q in zip(stamps, qqq, strict=True)}
    return out


BOOK = {"NVDAUSDT": 0.6, "AAPLUSDT": 0.4}


def test_the_independent_implementations_agree_with_the_desk() -> None:
    raw = _raw()
    _, columns = align(raw)
    frame = rd.Frame.build(raw, _is_open)
    b = rd.np_beta(frame.cols["NVDAUSDT"], frame.cols["QQQUSDT"])
    assert b is not None and math.isclose(b, beta(columns["NVDAUSDT"], columns["QQQUSDT"]) or 0,
                                          rel_tol=1e-9)
    move, start = rd.np_worst(BOOK, frame.cols)
    desk = worst_window(weights=BOOK, columns=columns)
    assert math.isclose(move, desk.move_pct or 0, rel_tol=1e-9) and start == desk.start_index
    cvar, parts = rd.np_cvar(BOOK, frame.cols)
    tail = tail_contributions(BOOK, columns)
    assert tail is not None and math.isclose(cvar, tail.cvar, rel_tol=1e-9)
    assert all(math.isclose(parts[s], v, rel_tol=1e-9) for s, v in tail.contributions)
    hits, total, extreme = rd.np_frequency([0.1, -0.2, 0.05, -0.3], bars=2, shock=-10.0)
    assert (hits, total) == (2, 2) and math.isclose(extreme, (1.05 * 0.7 - 1) * 100)


def test_the_one_pass_answer_is_checked_and_a_mis_weighted_window_is_explained() -> None:
    raw = _raw()
    frame = rd.Frame.build(raw, _is_open)
    move = rd.np_shock(BOOK, frame.session(True), "QQQUSDT", -10.0)
    _, columns = align(raw)
    # the STRESS branch's copilot call: before = the book less its first name, size = its weight
    used = rebalance({"AAPLUSDT": 0.4}, "NVDAUSDT", 0.6)
    as_used = worst_window(weights=used, columns=columns).move_pct or 0.0
    lines = [f"Actionable: If QQQ moves -10%: your book moves about {move:+.2f}% (market).",
             f"If QQQ moves -5%: your book moves about {move / 2 + 1:+.2f}% (market).",
             "What actually happened, not a model — the worst 24-bar window in the observed "
             f"history would have moved this book {as_used:+.2f}% — driven by NVDA"]
    rows = rd.recompute_single(lines, BOOK, "QQQUSDT", raw, _is_open)
    assert [r["ok"] for r in rows] == [True, False, False]
    window = rows[2]
    assert window["explained_by_weights_used"] is True
    assert window["weights_used"] == pytest.approx({"AAPLUSDT": 0.16, "NVDAUSDT": 0.6})


def test_single_pass_angles_are_read_from_its_wording() -> None:
    assert rd.single_angles([
        "If QQQ moves -10%: your book moves about -8.00%",
        "Hedge: short QQQ worth about 80% of the book's value",
        "What actually happened, not a model — the worst 24-bar window",
    ]) == ["beta_shock", "hedge_sizing", "realised_window"]


class _Inner:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.calls += 1
        return SimpleNamespace(content=self.content, finish_reason="stop", usage=SimpleNamespace(
            prompt_tokens=100, completion_tokens=10, total_tokens=110))


def test_the_recorder_writes_every_reply_before_parsing_and_stops_at_its_cap(tmp_path: Path
                                                                             ) -> None:
    log = tmp_path / "calls.jsonl"
    client = rd.RecordingClient(_Inner('{"units": [{"id": "news:NVDAUSDT"}]}'), cap=2, log=log)
    assert client.complete_json([{"role": "user", "content": "q"}]) == {
        "units": [{"id": "news:NVDAUSDT"}]}
    bad = rd.RecordingClient(_Inner("not json at all"), cap=1, log=log)
    with pytest.raises(QwenError):
        bad.complete_json([{"role": "user", "content": "q"}])
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [r["content"] for r in rows] == ['{"units": [{"id": "news:NVDAUSDT"}]}',
                                           "not json at all"]  # the unparseable one is kept too
    with pytest.raises(BudgetExhausted):
        bad.complete_json([{"role": "user", "content": "q"}])
    assert len(log.read_text(encoding="utf-8").splitlines()) == 2  # a refused call spends nothing


def test_a_citation_to_another_researchers_source_is_caught() -> None:
    from argus.lui.answer import Source

    mine, theirs = Source("venue", "a"), Source("venue", "b")
    unit = SimpleNamespace(id="news:NVDAUSDT")
    finding = SimpleNamespace(unit=unit, answered=True, sources=[mine], lines=["x 1"],
                              full=["x 1"])
    result = SimpleNamespace(findings=[finding])
    merged = SimpleNamespace(sources=[mine, theirs], data={"fanout": {"findings": [
        {"unit": "news:NVDAUSDT", "sources": [1, 2, 9]}]}})
    report = rd.fanout_integrity(merged, result)
    assert report["bad_citations"] == ["news:NVDAUSDT:[2]", "news:NVDAUSDT:[9]"]
    assert report["verbatim_violations"] == 0


def test_a_small_allowance_is_spread_across_the_kinds() -> None:
    def pair(n: int, kind: str) -> tuple[rd.Item, Any]:
        return rd.Item("c", str(n), f"q{n}", kind), SimpleNamespace(kind=kind)

    selected = [pair(0, "news"), pair(1, "news"), pair(2, "news"), pair(3, "macro"),
                pair(4, "macro"), pair(5, "quote")]
    order = [item.id for item, _ in rd.round_robin(selected)]
    assert order == ["0", "3", "5", "1", "4", "2"]  # one of each kind before a second of any


def test_the_summaries_report_losses_beside_gains() -> None:
    row = {"kind": "news", "first": "single", "units": ["a", "b"],
           "single": {"cited_sources": 2, "reached_sources": 5, "latency_ms": 100.0,
                      "chars": 300, "refused": False},
           "fanout": {"cited_sources": 6, "reached_sources": 9, "latency_ms": 900.0,
                      "chars": 900, "answered": 1, "not_answered": {"b:X": "dark"},
                      "merged": True, "verbatim_violations": 0, "bad_citations": []}}
    summary = rd.summarise_fanout([row])
    assert summary["distinct_cited_sources"]["fanout_vs_single"] == {"more": 1, "same": 0,
                                                                     "fewer": 0}
    assert summary["latency_ms"]["added_median"] == 800.0
    assert summary["latency_ms"]["fanout_slower_on"] == 1
    assert summary["answer_length_chars"]["fanout_median"] == 900.0
