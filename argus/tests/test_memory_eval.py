"""Pins the counting logic in `argus.eval.memory_eval`, not the extraction regex or the console.

`extraction()` and `effect()` read their fixed statement/task lists from module globals and call
`argus.lui.memory.extract` / `argus.lui.server.handle_ask` to do the real work; both are stubbed
here with hand-built, deterministic replies so every count below can be worked out by hand from
the table the test builds. That keeps this file about memory_eval's own arithmetic (how many
recalled, how many changed by memory, how the leaked/ablation counts are bucketed) rather than
about whether the regex or the console happen to be right today. `run()` is checked only for its
own glue: which keys it produces and that the artefact it writes round-trips.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from argus.eval import memory_eval
from argus.lui import memory as memory_mod
from argus.lui import server as server_mod


def _fact(kind: str, text: str = "") -> memory_mod.Fact:
    return memory_mod.Fact(kind=kind, subject="", value="", text=text, at="2026-01-01")


class TestExtraction:
    def test_recalled_counts_statements_whose_kind_was_extracted(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(memory_eval, "STATEMENTS", (
            ("stmt1", "max_loss"), ("stmt2", "budget"), ("stmt3", "horizon"),
        ))
        monkeypatch.setattr(memory_eval, "NOT_ABOUT_ME", ())
        replies = {"stmt1": [_fact("max_loss")], "stmt2": [_fact("horizon")], "stmt3": []}

        def fake_extract(text: str, now: Any = None, price_of: Any = None) -> list[memory_mod.Fact]:
            return replies[text]

        monkeypatch.setattr(memory_mod, "extract", fake_extract)
        result = memory_eval.extraction()
        assert result["statements"] == 3
        assert result["recalled"] == 1
        assert result["missed"] == [("stmt2", "budget", ["horizon"]), ("stmt3", "horizon", [])]

    def test_false_memories_are_kept_only_when_a_kind_was_extracted(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(memory_eval, "STATEMENTS", ())
        monkeypatch.setattr(memory_eval, "NOT_ABOUT_ME", ("q1", "q2"))
        replies = {"q1": [], "q2": [_fact("style")]}

        def fake_extract(text: str, now: Any = None, price_of: Any = None) -> list[memory_mod.Fact]:
            return replies[text]

        monkeypatch.setattr(memory_mod, "extract", fake_extract)
        result = memory_eval.extraction()
        assert result["not_about_me"] == 2
        assert result["false_memories"] == [("q2", ["style"])]

    def test_empty_statement_and_not_about_me_sets_report_all_zero(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(memory_eval, "STATEMENTS", ())
        monkeypatch.setattr(memory_eval, "NOT_ABOUT_ME", ())
        monkeypatch.setattr(memory_mod, "extract", lambda text, now=None, price_of=None: [])
        assert memory_eval.extraction() == {
            "statements": 0, "recalled": 0, "missed": [], "not_about_me": 0, "false_memories": [],
        }


_EFFECT_TABLE: dict[tuple[str, str, str | None], dict[str, Any]] = {
    ("stmt A", "memory-eval-a", None): {"memory": "[FACT_A]"},
    ("stmt B", "memory-eval-a", None): {"memory": "[FACT_B]"},
    ("stmt C", "memory-eval-a", None): {"memory": "[FACT_C]"},
    ("qA", "memory-eval-a", "[FACT_A]"): {
        "lines": ["Remembered: fact A used"], "classified_by": "pattern"},
    ("qA", "memory-eval-a", ""): {"lines": ["no memory answer"], "classified_by": "pattern"},
    ("qA", "memory-eval-b", ""): {"lines": ["no memory answer other"], "classified_by": "pattern"},
    ("qB", "memory-eval-a", "[FACT_B]"): {
        "lines": ["plain answer, no thesis line"], "classified_by": "pattern"},
    ("qB", "memory-eval-a", ""): {
        "lines": ["Your thesis on B leaked without memory"], "classified_by": "pattern"},
    ("qB", "memory-eval-b", ""): {
        "lines": ["Your thesis on B leaked to other trader"], "classified_by": "pattern"},
    ("qC", "memory-eval-a", "[FACT_C]"): {"lines": ["no marker with"], "classified_by": "pattern"},
    ("qC", "memory-eval-a", ""): {"lines": ["no marker without"], "classified_by": "qwen"},
    ("qC", "memory-eval-b", ""): {"lines": ["no marker other"], "classified_by": "pattern"},
}
"""Three hand-built tasks: A shows an effect, B is carried by the ablation and leaks to another
trader, C is stored but changes nothing and is answered by a different engine each time."""


def _fake_handle_ask(
    text: str, history: list[Any], *, visitor: str, memory: str | None = None,
) -> dict[str, Any]:
    return _EFFECT_TABLE[(text, visitor, memory)]


class TestEffect:
    def test_changed_ablation_and_leak_are_counted_from_hand_built_tasks(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(memory_eval, "TASKS", (
            ("stmt A", "qA", "Remembered:"),
            ("stmt B", "qB", "Your thesis on B"),
            ("stmt C", "qC", "Remembered:"),
        ))
        monkeypatch.setattr(server_mod, "handle_ask", _fake_handle_ask)
        result = memory_eval.effect()
        assert result["tasks"] == 3
        assert result["changed_by_memory"] == 1
        assert result["ablation_carried"] == 1
        assert result["leaked_to_other_trader"] == 1
        row_a, row_b, row_c = result["rows"]
        assert row_a["stored"] is True
        assert row_a["with_memory_carries"] and not row_a["without_memory_carries"]
        assert not row_b["with_memory_carries"]
        assert row_b["without_memory_carries"] and row_b["other_trader_carries"]
        assert row_c["same_engine"] is False

    def test_no_tasks_reports_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(memory_eval, "TASKS", ())

        def unreached(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("handle_ask should not be called when there are no tasks")

        monkeypatch.setattr(server_mod, "handle_ask", unreached)
        assert memory_eval.effect() == {
            "tasks": 0, "changed_by_memory": 0, "leaked_to_other_trader": 0,
            "ablation_carried": 0, "rows": [],
        }


class TestRun:
    def test_offline_omits_effect_and_writes_a_sanitised_artefact(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(memory_eval, "REPORT_PATH", tmp_path / "memory_eval.json")
        monkeypatch.setattr(memory_eval, "STATEMENTS", (("stmt1", "max_loss"),))
        monkeypatch.setattr(memory_eval, "NOT_ABOUT_ME", ())
        monkeypatch.setattr(
            memory_mod, "extract", lambda text, now=None, price_of=None: [_fact("max_loss")])
        report = memory_eval.run(online=False)
        assert "effect" not in report
        assert report["extraction"] == {
            "statements": 1, "recalled": 1, "missed": [], "not_about_me": 0, "false_memories": [],
        }
        on_disk = json.loads((tmp_path / "memory_eval.json").read_text(encoding="utf-8"))
        assert on_disk == report
        datetime.fromisoformat(report["generated_at"])

    def test_online_includes_the_effect_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(memory_eval, "REPORT_PATH", tmp_path / "memory_eval.json")
        monkeypatch.setattr(memory_eval, "STATEMENTS", (("stmt1", "max_loss"),))
        monkeypatch.setattr(memory_eval, "NOT_ABOUT_ME", ())
        monkeypatch.setattr(
            memory_mod, "extract", lambda text, now=None, price_of=None: [_fact("max_loss")])
        monkeypatch.setattr(memory_eval, "TASKS", (("stmt X", "qX", "Remembered:"),))
        table = {
            ("stmt X", "memory-eval-a", None): {"memory": "[FACT_X]"},
            ("qX", "memory-eval-a", "[FACT_X]"): {
                "lines": ["Remembered: x used"], "classified_by": "pattern"},
            ("qX", "memory-eval-a", ""): {"lines": ["no marker"], "classified_by": "pattern"},
            ("qX", "memory-eval-b", ""): {"lines": ["no marker"], "classified_by": "pattern"},
        }
        monkeypatch.setattr(
            server_mod, "handle_ask",
            lambda text, history, *, visitor, memory=None: table[(text, visitor, memory)])
        report = memory_eval.run(online=True)
        assert report["effect"]["tasks"] == 1
        assert report["effect"]["changed_by_memory"] == 1
