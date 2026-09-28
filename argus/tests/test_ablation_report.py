"""The consolidated ablation table reads each study's own record and says how much evidence each
row carries (research/harvest/19-shap.md)."""

from __future__ import annotations

import json
from pathlib import Path

from argus.eval import ablation_report as ar


def _write(tmp_path: Path) -> dict[str, Path]:
    pipeline = tmp_path / "ablations.json"
    pipeline.write_text(json.dumps({"results": [
        {"component": "analyst selection", "frames": 12, "changed": 12},
        {"component": "as-of gate", "frames": 12, "changed": 0},
    ]}), encoding="utf-8")
    gates = tmp_path / "population_rct.json"
    gates.write_text(json.dumps({"dimensions": {
        "order_size": {"component": "order_size", "frames": 1, "changed": 0,
                       "risk_violations": 0, "status": "INERT"},
        "per_symbol": {"component": "per_symbol", "frames": 0, "changed": 0,
                       "risk_violations": 0, "status": "UNTESTED"},
    }}), encoding="utf-8")
    return {"pipeline": pipeline, "risk gates": gates}


def test_each_row_states_how_much_evidence_it_has(tmp_path: Path) -> None:
    blob = ar.report(_write(tmp_path))
    status = {r["component"]: r["status"] for r in blob["rows"]}
    assert status == {"analyst selection": "CHANGES DECISIONS", "as-of gate": "INERT",
                      "order_size": "TOO FEW FRAMES", "per_symbol": "UNTESTED"}
    assert blob["by_status"]["TOO FEW FRAMES"] == 1
    assert "1 specified and never run" in blob["reading"]


def test_an_unreadable_study_is_named_not_skipped(tmp_path: Path) -> None:
    sources = _write(tmp_path)
    sources["missing"] = tmp_path / "nope.json"
    assert ar.report(sources)["unreadable"] == ["missing: nope.json unreadable"]


def test_the_latex_table_escapes_and_lists_the_unrun(tmp_path: Path) -> None:
    tex = ar.to_latex(ar.report(_write(tmp_path)))
    assert tex.startswith(r"\begin{tabular}") and tex.rstrip().endswith(r"\end{tabular}")
    assert r"order\_size" in tex and "NOT RUN" in tex


def test_the_live_studies_read(tmp_path: Path) -> None:
    """Against the committed artefacts: every row readable, nothing invented."""
    blob = ar.report()
    assert blob["unreadable"] == [] and blob["rows"]
    assert all(r["status"] in {"CHANGES DECISIONS", "INERT", "TOO FEW FRAMES", "UNTESTED"}
               for r in blob["rows"])
