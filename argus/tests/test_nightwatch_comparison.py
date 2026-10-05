"""The desk's lean against NIGHTWATCH AI's signal (build-list 5.4), offline: NIGHTWATCH's answers
are scripted here; its real run needs Node and its clone and is recorded in the artefact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.eval import nightwatch_comparison as nc


def _row(i: int, argus: str, nightwatch: str | None, move: float) -> dict[str, object]:
    return {"key": str(i), "symbol": "NVDA", "cycle": f"2026-09-{15 + i % 10}T14:00",
            "argus": argus, "nightwatch": nightwatch, "move_8h_bps": move,
            "move_mark_bps": move, "horizon_hours": 2.0}


def test_the_paired_test_counts_only_the_disagreements() -> None:
    rows = [_row(0, "up", "down", 50.0), _row(1, "up", "down", 60.0),
            _row(2, "down", "up", 40.0), _row(3, "up", "up", 30.0),
            _row(4, "up", "down", 5.0)]  # under the 20 bps flat band: not scored
    got = nc._paired(rows, "move_8h_bps")
    assert got["argus_right_nightwatch_wrong"] == 2
    assert got["nightwatch_right_argus_wrong"] == 1
    assert got["sign_test_p_two_sided"] == pytest.approx(1.0)


def test_a_score_sets_each_side_beside_calling_up_every_time() -> None:
    rows = [_row(i, "down", "up", -30.0) for i in range(12)]
    ours, theirs = nc._score(rows, "argus", "move_8h_bps"), nc._score(rows, "nightwatch",
                                                                       "move_8h_bps")
    assert ours["hit_rate_pct"] == 100.0 and theirs["hit_rate_pct"] == 0.0
    assert ours["always_up_pct"] == 0.0
    assert nc._score([], "argus", "move_8h_bps") == {"calls": 0}


def test_a_name_outside_its_universe_is_left_out_not_scored_as_a_miss(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    marks = tmp_path / "marks.jsonl"
    marks.write_text("\n".join(json.dumps({
        "seq": i, "symbol": sym, "decided_at": "2026-09-20T14:00:00+00:00",
        "marked_at": "2026-09-20T16:00:00+00:00", "horizon_hours": 2.0, "move_bps": "40",
        "lean": "up"}) for i, sym in enumerate(("NVDAUSDT", "SQQQUSDT"))), encoding="utf-8")
    monkeypatch.setattr(nc, "MARKS", marks)
    hour = 3_600_000
    start = 1_789_900_000_000
    candles = [{"ts": start + k * hour, "open": 1, "high": 1, "low": 1, "close": 100 + k,
                "volume": 1} for k in range(400)]
    monkeypatch.setattr(nc, "signals", lambda instants, inputs: {
        "0": {"key": "0", "direction": "LONG", "status": "SIGNAL"},
        "1": {"key": "1", "skipped": "not in its universe"}})
    got = nc.score({"candles": {"NVDA": candles, "SQQQ": candles}, "macro_daily": {}})
    assert got["instants"] == 1
    assert got["skipped"] == {"not in its universe": 1}
    assert got["nightwatch_signals"] == 1
