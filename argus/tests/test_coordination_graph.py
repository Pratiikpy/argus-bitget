"""The persistent account graph (research/harvest/11-coortweet.md): pair weights, CooRTweet's
symmetry and percentile, and the incremental catch over the per-story screen."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.eval.coordination_graph_eval import pseudonym, run
from argus.truth.coordination_graph import Share, components, edges, percentile_cut, strong


def test_pairs_weigh_the_stories_they_share_and_symmetry_the_balance() -> None:
    shares = [Share("a", "s1", 3), Share("b", "s1", 1), Share("a", "s2"), Share("b", "s2"),
              Share("c", "s2")]
    graph = {(e.left, e.right): e for e in edges(shares)}
    assert graph[("a", "b")].weight == 2
    assert graph[("a", "b")].symmetry == pytest.approx(2 / 4)  # a posted 4, b posted 2
    assert graph[("a", "c")].weight == 1 and graph[("a", "c")].symmetry == 1


def test_a_story_carried_by_a_crowd_adds_no_pairs() -> None:
    shares = [Share(f"u{i}", "big") for i in range(61)]
    assert edges(shares) == []


def test_the_percentile_follows_the_graph_and_one_meeting_is_not_enough() -> None:
    shares = [Share(x, s) for s, xs in {"s1": "ab", "s2": "ab", "s3": "ab", "s4": "cd",
                                        "s5": "ef", "s6": "gh"}.items() for x in xs]
    graph = edges(shares)
    assert percentile_cut(graph, 0.5) == 1
    assert [(e.left, e.right) for e in strong(graph)] == [("a", "b")]
    assert components(strong(graph)) == [{"a", "b"}]


def test_the_eval_finds_a_pair_the_per_story_screen_missed(tmp_path: Path) -> None:
    def snap(clusters: list[dict[str, object]]) -> str:
        return json.dumps({"kind": "snapshot", "payload": {"snapshot": {"crowd": {
            "items": 3, "clusters": clusters}}}})

    rows = [snap([{"cluster_id": "s1", "sources": ["@a", "@b"], "coordinated": False}]),
            snap([{"cluster_id": "s2", "sources": ["@a", "@b"], "coordinated": False}]),
            snap([{"cluster_id": "s3", "sources": ["@c", "@d", "@e"], "coordinated": True}]),
            snap([{"cluster_id": "s4", "sources": ["@c", "@d", "@e"], "coordinated": True}]),
            json.dumps({"kind": "note", "payload": {"text": "a" + chr(0x2028) + "b"}},
                       ensure_ascii=False)]
    (tmp_path / "ledger.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    report = run(tmp_path)
    assert report["input"]["stories"] == 4 and report["input"]["stories_flagged_per_story"] == 2
    found = report["incremental"]["examples"]
    assert [(e["left"], e["right"]) for e in found] == [(pseudonym("@a"), pseudonym("@b"))]
    # No handle reaches the published report: pairs and groups carry pseudonyms only.
    assert "@" not in json.dumps(report["graph"]) + json.dumps(report["incremental"])
    assert pseudonym("u/x").startswith("reddit:") and pseudonym("@a").startswith("x:")
