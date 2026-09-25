"""Open interest: the live figure against its own volume and the venue, and its record."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from argus.market import open_interest as oi


def _rows() -> list[dict[str, str]]:
    rows = [{"symbol": "ETHUSDT", "holdingAmount": "1000", "lastPr": "2000",
             "usdtVolume": "1000000", "fundingRate": "0.0001"}]
    # Nine liquid peers with 0.5 to 4.5 days of volume held open, one illiquid name ignored.
    for i in range(9):
        rows.append({"symbol": f"P{i}USDT", "holdingAmount": str(0.5 + i * 0.5),
                     "lastPr": "1000000", "usdtVolume": "1000000", "fundingRate": "0"})
    rows.append({"symbol": "TINYUSDT", "holdingAmount": "99", "lastPr": "1",
                 "usdtVolume": "10", "fundingRate": "0"})
    return rows


def test_notional_turnover_and_rank_are_read_from_one_ticker_call(tmp_path: Path) -> None:
    reading = oi.read("ETHUSDT", fetch=_rows, history=tmp_path / "none.jsonl")
    assert reading is not None
    assert reading.notional == 2_000_000 and reading.turnover_days == 2.0
    assert reading.peers == 10 and reading.rank_pct == 30.0
    text = " ".join(oi.lines(reading, "ETH"))
    assert "1,000.00 ETH held open" in text and "2.00 days of today's volume" in text
    assert "higher than 30% of the 10 USDT perpetuals" in text
    assert "not known yet" in text and "longs pay shorts" in text


def test_the_change_is_stated_only_against_a_snapshot_from_about_a_day_ago(
        tmp_path: Path) -> None:
    now = datetime(2026, 9, 26, 12, tzinfo=UTC)
    path = tmp_path / "history.jsonl"
    oi.record(path, fetch=lambda: [{**_rows()[0], "holdingAmount": "800"}],
              now=now - timedelta(hours=30))
    assert oi.read("ETHUSDT", fetch=_rows, history=path, now=now).change_24h is None
    oi.record(path, fetch=lambda: [{**_rows()[0], "holdingAmount": "800"}],
              now=now - timedelta(hours=23))
    reading = oi.read("ETHUSDT", fetch=_rows, history=path, now=now)
    assert reading.change_24h == 0.25
    assert "changed +25.0%" in " ".join(oi.lines(reading, "ETH"))


def test_only_liquid_contracts_are_recorded(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    assert oi.record(path, fetch=_rows) == 10
    assert "TINYUSDT" not in json.loads(path.read_text("utf-8"))["holding"]
