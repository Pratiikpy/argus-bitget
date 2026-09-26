"""Questions about the desk's own record reach the record, in the kind they ask for (2026-09-26).

The figures quoted from `data/lui_record_routing_2026-09-26.json` are pinned to the sets they were
measured on, and the phrasing families added from the tuning half are checked here on sentences of
their own, not on the sets' questions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from argus.eval import record_routing
from argus.lui.research import about_the_desk

DATA = Path(__file__).resolve().parents[1] / "data"


class TestTheDeskIsTheSubject:
    @pytest.mark.parametrize(
        "text",
        [
            "are we long or short META right now",
            "why did we sell AMZN on friday",
            "what information supported that MSTR trade",
            "which inputs drove the decision on AAPL",
            "我们现在的敞口多大",
            "为什么周三卖出特斯拉",
        ],
    )
    def test_the_record_is_asked_about(self, text: str) -> None:
        assert about_the_desk(text)

    @pytest.mark.parametrize(
        "text",
        [
            "are we heading into a recession",
            "我们是不是要进入衰退了",
            "how did markets react to the fed decision",
            "what would adding 10% TSLA do to our book",
            "机构持仓最近有什么变化",
            "is the ai trade in nvda overheated",
            "are we still in a bull market",
        ],
    )
    def test_the_market_or_a_plan_is_not(self, text: str) -> None:
        assert not about_the_desk(text)


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("are we long or short META right now", "position"),
        ("why did we sell AMZN on friday", "decision_why"),
        ("show me every decision from yesterday", "decision_list"),
        ("why did the desk sit out the CPI print", "abstention_why"),
        ("what data was the desk reading before the TSLA call", "evidence"),
        ("were we overconfident last week", "calibration"),
        ("could the log have been edited after the fact", "integrity"),
        ("is tomorrow a trading day", "session"),
        ("how much did we make this week", "performance"),
        ("本周做了哪些操作", "decision_list"),
        ("为什么选择观望", "abstention_why"),
        ("上周的决策是基于什么信息", "evidence"),
    ],
)
def test_a_record_question_reaches_its_kind(text: str, kind: str) -> None:
    with record_routing._offline():
        got, _by = record_routing.route(text)
    assert got == kind


class TestThePublishedFigures:
    report = json.loads((DATA / "lui_record_routing_2026-09-26.json").read_text("utf-8"))

    def test_the_sets_are_the_ones_measured(self) -> None:
        test_half = DATA / "lui_record_intents_2026-09-26_test.jsonl"
        round2 = DATA / "lui_record_intents_2026-09-26_round2.jsonl"
        assert hashlib.sha256(test_half.read_bytes()).hexdigest() == self.report["test_sha256"]
        assert (
            hashlib.sha256(round2.read_bytes()).hexdigest()
            == self.report["round_2"]["set_sha256"]
        )

    def test_the_halves_partition_the_first_set(self) -> None:
        whole = record_routing.load(DATA / "lui_record_intents_2026-09-26.jsonl")
        halves = [
            record_routing.load(DATA / f"lui_record_intents_2026-09-26_{h}.jsonl")
            for h in ("tune", "test")
        ]
        assert sorted(r["id"] for h in halves for r in h) == sorted(r["id"] for r in whole)

    def test_the_held_out_gain_is_what_the_headline_says(self) -> None:
        r2 = self.report["round_2"]
        assert r2["before_public_commit_5637fa0"]["correct"] == 131
        assert r2["after_held_out"]["correct"] == 147
        assert r2["after_held_out"]["rows"] == 200
        assert "131 to 147" in self.report["headline"]
