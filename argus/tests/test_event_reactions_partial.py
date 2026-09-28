"""One name whose candles cannot be read is reported and left out of the event study; the other
names are still measured (research/event_reactions.build; the 2026-09-27 19:00 cycle lost the
whole study to COINUSDT and TQQQUSDT)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from argus.market.history import HistoryError
from argus.research import event_reactions


def test_an_unreadable_name_is_named_and_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    start = datetime(2026, 1, 5, tzinfo=UTC)
    series = {start + timedelta(hours=h): 100.0 + (h % 7) * 0.1 for h in range(2000)}

    def hourly(symbol: str) -> dict[datetime, float]:
        if symbol == "COINUSDT":
            raise HistoryError("HTTP 429: Too Many Requests")
        return dict(series)

    monkeypatch.setattr(event_reactions, "_hourly", hourly)
    monkeypatch.setattr(event_reactions, "earnings_releases", lambda ticker: [])
    monkeypatch.setattr(event_reactions, "cpi_releases", lambda: ([], "test"))
    monkeypatch.setattr(event_reactions, "fomc_decisions", lambda: [])
    report = event_reactions.build(now=start + timedelta(hours=2000))
    assert set(report["history"]["unavailable"]) == {"COINUSDT"}
    assert "429" in report["history"]["unavailable"]["COINUSDT"]
    measured = {row["symbol"] for row in report["reactions"]}
    assert "COINUSDT" not in measured and "NVDAUSDT" in measured
