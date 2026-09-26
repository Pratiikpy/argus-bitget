"""The decision-record answers say what the record holds (readiness backlog L43).

"show me decision 25" printed "net None, direction wrong" for a decision that took no position, and
"what evidence backed that" described the gathering mechanism without any evidence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from argus.lui.answer import EVIDENCE_SHOWN, _evidence_lines
from argus.lui.phrasebook import Language
from argus.lui.question import classify

NOW = datetime(2026, 9, 26, 12, tzinfo=UTC)


@pytest.fixture
def notes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ARGUS_DATA_DIR", str(tmp_path))
    path = tmp_path / "desk_notes.jsonl"
    rows = [
        {"seq": 7, "notes": [], "sources": ["filing", "news"]},
        {"seq": 9, "notes": [], "sources": ["news"],
         "evidence": [f"[e{i}] (news, credibility 0.70, available 2026-09-26) item {i}"
                      for i in range(EVIDENCE_SHOWN + 2)]},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


class TestTheEvidenceAnswer:
    def test_recorded_items_are_quoted_and_the_rest_counted(self, notes: Path) -> None:
        lines = _evidence_lines(9, Language.EN)
        assert f"{EVIDENCE_SHOWN + 2} evidence items" in lines[0]
        assert lines[1].strip().startswith("[e0]")
        assert len(lines) == 1 + EVIDENCE_SHOWN + 1
        assert "2 more" in lines[-1]

    def test_an_older_row_names_its_sources_and_says_the_items_were_not_kept(
        self, notes: Path
    ) -> None:
        (line,) = _evidence_lines(7, Language.EN)
        assert "filing, news" in line
        assert "not written down" in line

    def test_a_decision_with_no_notes_says_so(self, notes: Path) -> None:
        (line,) = _evidence_lines(8, Language.EN)
        assert "cannot be listed" in line


class TestADecisionNamedInChinese:
    @pytest.mark.parametrize("text", ["决策25是什么", "显示决策 25", "第25号决策为什么",
                                      "序号 25 的证据"])
    def test_the_number_is_the_decision(self, text: str) -> None:
        assert classify(text, now=NOW).seq == 25

    def test_a_decision_without_a_number_names_none(self) -> None:
        assert classify("决策是怎么做的", now=NOW).seq is None


def test_a_settled_decision_without_a_position_reports_no_pnl() -> None:
    from argus.lui.server import handle_ask

    lines = handle_ask("show me decision 25", [], now=NOW)["lines"]
    settled = [line for line in lines if line.startswith("Settled")]
    assert settled, lines
    assert "no position, so no P&L" in settled[0]
    assert "None" not in settled[0]
    assert "direction wrong" not in settled[0]
