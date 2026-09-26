"""Pins the counting logic in `argus.eval.multistep_eval`, not `lui/multistep.parts` itself.

`recall()` scores a hand-built `MULTI` set against `argus.lui.multistep.parts`, and `false_splits()`
scans the fixed `CORPORA` files under `DATA` for a split on a single question. Both are stubbed here
so the arithmetic under test — which questions count as "split correctly", which single questions
count as a false split, whether a missing corpus file is skipped rather than crashing — can be
worked out by hand instead of depending on the real decomposer's behaviour today.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from argus.eval import multistep_eval
from argus.lui import multistep as multistep_mod


def _part(kind: str, text: str = "") -> Any:
    return SimpleNamespace(request=SimpleNamespace(kind=SimpleNamespace(value=kind)), text=text)


class TestRecall:
    def test_two_part_and_single_label_questions_are_scored_by_hand(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(multistep_eval, "MULTI", (
            ("q1", ("quote", "impact")),   # split correctly, in order
            ("q2", ("quote", "impact")),   # split, but the wrong second part
            ("q3", ("analogue",)),         # single label: correct means answered whole (no split)
            ("q4", ("analogue",)),         # single label, but wrongly split into two parts
        ))
        replies = {
            "q1": [_part("quote"), _part("impact")],
            "q2": [_part("quote"), _part("hedge")],
            "q3": [],
            "q4": [_part("analogue"), _part("technicals")],
        }
        monkeypatch.setattr(multistep_mod, "parts", lambda question, book="": replies[question])
        result = multistep_eval.recall()
        assert result["questions"] == 4
        assert result["split_correctly"] == 2
        rows = {r["question"]: r for r in result["rows"]}
        assert rows["q1"]["ok"] is True and rows["q1"]["got"] == ["quote", "impact"]
        assert rows["q2"]["ok"] is False and rows["q2"]["got"] == ["quote", "hedge"]
        assert rows["q3"]["ok"] is True and rows["q3"]["got"] == []
        assert rows["q4"]["ok"] is False and rows["q4"]["got"] == ["analogue", "technicals"]

    def test_no_questions_reports_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(multistep_eval, "MULTI", ())
        monkeypatch.setattr(
            multistep_mod, "parts",
            lambda *a, **k: pytest.fail("parts() should not be called with no questions"))
        assert multistep_eval.recall() == {"questions": 0, "split_correctly": 0, "rows": []}


class TestFalseSplits:
    def test_a_missing_corpus_is_skipped_and_only_split_rows_are_reported(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        (tmp_path / "corpus1.jsonl").write_text(
            json.dumps({"text": "single q1"}) + "\n" + json.dumps({"text": "single q2"}) + "\n",
            encoding="utf-8")
        (tmp_path / "corpus2.jsonl").write_text(
            json.dumps({"text": "single q3"}) + "\n", encoding="utf-8")
        monkeypatch.setattr(multistep_eval, "DATA", tmp_path)
        monkeypatch.setattr(
            multistep_eval, "CORPORA", ("corpus1.jsonl", "missing.jsonl", "corpus2.jsonl"))
        replies = {
            "single q1": None,
            "single q2": [_part("quote", text="p1"), _part("impact", text="p2")],
            "single q3": None,
        }
        monkeypatch.setattr(multistep_mod, "parts", lambda question: replies[question])
        result = multistep_eval.false_splits()
        # "missing.jsonl" does not exist and contributes neither to the total nor a crash.
        assert result["single_questions"] == 3
        assert result["split"] == 1
        assert result["rows"] == [
            {"corpus": "corpus1.jsonl", "text": "single q2", "parts": ["p1", "p2"]},
        ]

    def test_no_corpora_reports_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(multistep_eval, "DATA", tmp_path)
        monkeypatch.setattr(multistep_eval, "CORPORA", ())
        assert multistep_eval.false_splits() == {"single_questions": 0, "split": 0, "rows": []}


class TestRun:
    def test_combines_recall_and_false_splits_and_writes_the_artefact(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        (tmp_path / "corpus1.jsonl").write_text(
            json.dumps({"text": "single q2"}) + "\n", encoding="utf-8")
        monkeypatch.setattr(multistep_eval, "REPORT_PATH", tmp_path / "multistep_eval.json")
        monkeypatch.setattr(multistep_eval, "DATA", tmp_path)
        monkeypatch.setattr(multistep_eval, "CORPORA", ("corpus1.jsonl",))
        monkeypatch.setattr(multistep_eval, "MULTI", ())
        monkeypatch.setattr(
            multistep_mod, "parts",
            lambda question, book="": [_part("quote", text="p1"), _part("impact", text="p2")])
        report = multistep_eval.run()
        assert report["multi"] == {"questions": 0, "split_correctly": 0, "rows": []}
        assert report["single"]["single_questions"] == 1
        assert report["single"]["split"] == 1
        datetime.fromisoformat(report["generated_at"])
        on_disk = json.loads((tmp_path / "multistep_eval.json").read_text(encoding="utf-8"))
        assert on_disk == report
