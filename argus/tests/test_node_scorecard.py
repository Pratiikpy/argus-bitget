"""Each console part measured alone, laid out as JPMorgan's Ask David (build-list 5.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.eval import node_scorecard as ns
from argus.lui.status_page import node_lines


def test_a_missing_artefact_drops_its_row_rather_than_inventing_one(tmp_path: Path) -> None:
    assert ns.card(tmp_path) == []
    (tmp_path / "provenance_unseen.json").write_text(json.dumps({
        "collected": "2026-09-25",
        "first_read": {"content_lines": 156, "unlabelled": 9, "coverage": 0.942}}),
        encoding="utf-8")
    rows = ns.card(tmp_path)
    assert [r["node"] for r in rows] == ["reflection"]
    assert rows[0]["value"] == "147/156 (94.2%)"
    assert node_lines(tmp_path)[0][0] == "Part measured alone: reflection"


def test_the_live_artefacts_fill_all_six_nodes() -> None:
    from argus.truth.paths import DATA_DIR

    rows = ns.card(DATA_DIR)
    if len(rows) < 6:
        pytest.skip("the measured artefacts are not all in this checkout")
    assert [r["node"] for r in rows] == ["supervisor", "structured data",
                                         "unstructured documents", "analytics",
                                         "personalisation", "reflection"]
    assert all(r["value"] and "None" not in r["value"] for r in rows)
