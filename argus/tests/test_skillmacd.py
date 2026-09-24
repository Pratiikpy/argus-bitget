"""`eval/skillmacd.py` — the Skill's MACD fields, classified against a recomputation."""

from __future__ import annotations

from typing import Any

import pytest

from argus.eval import skillmacd


class _Client:
    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def call(self, tool: str, args: dict[str, Any], *, timeout: int = 30) -> tuple[Any, str]:
        return self.payload, "ok"


OURS = {"dif": 1568.0, "dea": 1715.0, "histogram": -147.0, "cross": ""}


@pytest.fixture(autouse=True)
def recomputed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skillmacd, "indicators", lambda s: OURS)


def test_the_measured_btc_reading_is_swapped_and_its_cross_contradicts_the_lines() -> None:
    row = skillmacd.check("BTCUSDT", _Client({"macd": 1568.4, "signal": -148.3,
                                               "histogram": 1716.3, "cross": "golden_cross"}))
    assert row["verdict"] == "swapped" and row["cross_wrong"] is True


def test_a_fixed_skill_reads_straight() -> None:
    row = skillmacd.check("BTCUSDT", _Client({"macd": 1568.4, "signal": 1716.3,
                                               "histogram": -148.3, "cross": "death_cross"}))
    assert row["verdict"] == "straight" and row["cross_wrong"] is False


def test_different_macd_lines_mean_no_field_conclusion() -> None:
    row = skillmacd.check("BTCUSDT", _Client({"macd": 900.0, "signal": -148.3,
                                               "histogram": 1716.3}))
    assert row["verdict"] == "unchecked"


def test_no_series_is_unchecked_not_straight() -> None:
    row = skillmacd.check("TQQQUSDT", _Client({"error": "No OHLCV data for TQQQUSDT/4h"}))
    assert row["verdict"] == "unchecked"
