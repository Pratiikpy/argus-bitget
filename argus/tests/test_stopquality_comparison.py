"""The stop-quality scorer: noise hits on held-out bars, ARGUS's distance fitted out of sample."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from argus.eval import stopquality_comparison as sq


class Bar:
    def __init__(self, close: float, low: float) -> None:
        self.close, self.low = close, low


def test_a_tight_stop_is_hit_by_noise_and_the_fitted_one_is_not(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # every day closes at 100 and dips 1% to 3% intraday, repeating
    bars = [Bar(100.0, 100.0 * (1 - [0.01, 0.02, 0.03][i % 3])) for i in range(300)]
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
