"""The stop-quality scorer: noise hits on held-out bars, ARGUS's distance fitted out of sample."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from argus.eval import stopquality_comparison as sq


class Bar:
    def __init__(self, close: float, low: float, day: int = 0) -> None:
        self.close, self.low, self.high = close, low, close
        self.ts = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=day)


def test_a_tight_stop_is_hit_by_noise_and_the_fitted_one_is_not(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # every day closes at 100 and dips 1% to 3% intraday, repeating
    bars = [Bar(100.0, 100.0 * (1 - [0.01, 0.02, 0.03][i % 3]), i) for i in range(300)]
    run = {"stdout_log": ["[bitget] ticker ok"],
           "result": {"report": {"bias": "long", "invalidation_price": 99.5},
                      "snapshot": {"last": 100.0}}}
    (tmp_path / "NVDAUSDT.json").write_text(json.dumps(run), encoding="utf-8")
    monkeypatch.setattr(sq, "DATA", tmp_path)
    monkeypatch.setattr(sq, "REPORT", tmp_path / "out.json")
    out: dict[str, Any] = sq.score(fetch=lambda symbol: bars)
    [row] = out["rows"]
    assert row["rook_distance"] == pytest.approx(0.005)
    assert row["rook_noise_hit_rate_oos"] == 1.0  # every day dips past 0.5%
    assert row["argus_distance_fit_on_first_60pct"] == pytest.approx(0.03)
    assert row["argus_noise_hit_rate_oos"] == pytest.approx(1 / 3, abs=0.02)


def _rook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = {"result": {"report": {"bias": "long", "invalidation_price": 99.5},
                      "snapshot": {"last": 100.0}}}
    (tmp_path / "NVDAUSDT.json").write_text(json.dumps(run), encoding="utf-8")
    monkeypatch.setattr(sq, "DATA", tmp_path)
    monkeypatch.setattr(sq, "BARS", tmp_path / "bars")
    monkeypatch.setattr(sq, "REPORT", tmp_path / "out.json")


def test_argus_distance_is_the_consoles_directional_odds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A regression in desk/odds.py must reach this harness: it calls the console's function."""
    from argus.desk import odds

    _rook(tmp_path, monkeypatch)

    def broken(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("console broken")

    monkeypatch.setattr(odds, "directional_odds", broken)
    bars = [Bar(100.0, 99.0, i) for i in range(60)]
    with pytest.raises(RuntimeError, match="console broken"):
        sq.score(fetch=lambda symbol: bars)


def test_collected_bars_score_offline_and_the_recorded_prospective_block_is_kept(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _rook(tmp_path, monkeypatch)
    bars = [Bar(100.0, 100.0 * (1 - [0.01, 0.02, 0.03][i % 3]), i) for i in range(90)]
    assert sq.collect(fetch=lambda symbol: bars) == {"NVDAUSDT": 90}
    recorded = {"recorded_at": "2026-09-25T05:16:27+00:00",
                "stops": {"NVDAUSDT": {"rook": 99.5, "argus_distance": 0.03, "last": 100.0}}}
    (tmp_path / "out.json").write_text(json.dumps({"prospective": recorded}), encoding="utf-8")
    out = sq.score()  # from the saved files: no fetch
    [row] = out["rows"]
    assert row["bars"] == 90 and row["argus_distance_fit_on_first_60pct"] == pytest.approx(0.03)
    assert row["argus_distance_former_copy"] == pytest.approx(0.03)
    assert out["prospective"] == recorded
    assert json.loads((tmp_path / "out.json").read_text("utf-8"))["prospective"] == recorded
