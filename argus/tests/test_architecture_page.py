"""The /architecture page shows the checked report as recorded (research/harvest/57)."""

from __future__ import annotations

import json
from pathlib import Path

from argus.lui.architecture_page import load, render

REPORT = {
    "generated_at": "2026-09-28T01:02:03+00:00", "modules": 10, "internal_imports": 20,
    "sound": True, "violations": [], "import_time_cycles": 0,
    "cycles": [{"members": ["argus.a", "argus.b"], "import_time": False}],
    "packages": [{"package": "truth", "modules": 3, "afferent": 9, "efferent": 0,
                  "instability": 0.0},
                 {"package": "vendor", "modules": 1, "afferent": 0, "efferent": 0,
                  "instability": None}],
    "verdict": "10 modules. every declared contract holds.",
}


def test_the_page_shows_the_recorded_figures(tmp_path: Path) -> None:
    (tmp_path / "architecture.json").write_text(json.dumps(REPORT), encoding="utf-8")
    page = render(load(tmp_path))
    assert "Every declared contract holds." in page and "10 modules. Every declared" in page
    assert "safe at load" in page and "a → b" in page
    assert "2026-09-28 01:02 UTC" in page and "<td class='n'>—</td>" in page


def test_a_violation_is_listed(tmp_path: Path) -> None:
    broken = {**REPORT, "violations": [{"contract": "layer order", "where": "argus.risk.x:3",
                                        "detail": "imports argus.eval"}]}
    (tmp_path / "architecture.json").write_text(json.dumps(broken), encoding="utf-8")
    page = render(load(tmp_path))
    assert "argus.risk.x:3" in page and "<p class='ok'>" not in page


def test_an_unreadable_report_shows_nothing_rather_than_something(tmp_path: Path) -> None:
    assert "could not be read" in render(load(tmp_path))
