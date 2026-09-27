"""A rerun of a harness reproduces its published artefact (audit finding 147)."""

from __future__ import annotations

import json
from pathlib import Path

from argus.eval import harness_validity as hv


def test_time_stamps_are_ignored_and_findings_are_not(tmp_path: Path) -> None:
    published = tmp_path / "argus" / "data"
    published.mkdir(parents=True)
    (published / "a.json").write_text(json.dumps(
        {"measured_at": "2026-09-01", "wins": 3, "rows": [{"at": "x", "v": 1}]}), encoding="utf-8")
    (published / "b.json").write_text(json.dumps({"wins": 3}), encoding="utf-8")
    fresh = tmp_path / "scratch"
    fresh.mkdir()
    (fresh / "a").write_text(json.dumps(
        {"measured_at": "2026-09-27", "wins": 3, "rows": [{"at": "y", "v": 1}]}), encoding="utf-8")
    (fresh / "b").write_text(json.dumps({"wins": 4}), encoding="utf-8")
    (fresh / "c").write_text("{}", encoding="utf-8")
    got = hv.reproduction({"argus/data/a.json": fresh / "a", "argus/data/b.json": fresh / "b",
                           "argus/data/c.json": fresh / "c", "argus/notes.md": fresh / "c"},
                          tmp_path)
    assert got == {"argus/data/a.json": "identical", "argus/data/b.json": "differs",
                   "argus/data/c.json": "new"}


def test_the_claim_check_harness_reproduces_its_artefact_offline() -> None:
    run = hv._run_child("claimcheck_comparison", "baseline", [], 600)
    assert run["status"] == "ok" and run["network_attempts"] == 0
    assert run["reproduction"]["argus/data/claimcheck_comparison.json"] == "identical"
