"""The recorded stops are graded on the 24 hours after the run, never before."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from argus.eval import stopquality_prospective as sp

RECORDED = datetime(2026, 9, 25, 5, tzinfo=UTC)


class Bar:
    def __init__(self, hour: int, low: float, close: float) -> None:
        self.ts, self.low, self.close = RECORDED + timedelta(hours=hour), low, close


@pytest.fixture
def recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "stops.json"
    source.write_text(json.dumps({"prospective": {
        "recorded_at": RECORDED.isoformat(),
        "stops": {"NVDAUSDT": {"rook": 99.0, "argus_distance": 0.03, "last": 100.0}}}}))
    monkeypatch.setattr(sp, "SOURCE", source)
    monkeypatch.setattr(sp, "REPORT", tmp_path / "out.json")
    return source


def test_nothing_is_graded_before_the_horizon(recorded: Path) -> None:
    out = sp.grade(now=RECORDED + timedelta(hours=23))
    assert out["graded"] is False and "nothing is graded before it" in out["reason"]


def test_a_dip_to_98_takes_the_tight_stop_and_not_the_measured_one(recorded: Path) -> None:
    bars = [Bar(h, 100.0 if h != 7 else 98.0, 100.0) for h in range(24)]
    out = sp.grade(now=RECORDED + timedelta(hours=25), fetch=lambda symbol: bars)
    [row] = out["rows"]
    assert row["rook_touched"] is True and row["argus_touched"] is False
    assert row["argus_stop"] == pytest.approx(97.0)
