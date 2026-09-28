"""Status history: recorded probe rounds, chained, and the uptime computed from them
(research/harvest/54-upptime.md)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from argus.lui.status_history import read, record, uptime, verify

T0 = datetime(2026, 9, 28, 12, tzinfo=UTC)


@dataclass
class Probe:
    surface: str
    what: str
    ok: bool
    ms: float = 100.0


def _history(tmp_path: Path) -> Path:
    """Ten rounds an hour apart; the order book fails on three of them."""
    path = tmp_path / "status_history.jsonl"
    for hour in range(10):
        record([Probe("Bitget market API", "all tickers", True),
                Probe("Bitget market API", "order book", hour not in (2, 5, 7))],
               at=T0 - timedelta(hours=9 - hour), path=path)
    return path


def test_the_uptime_is_answered_over_all_probes(tmp_path: Path) -> None:
    rows = read(_history(tmp_path))
    table = {u["what"]: u for u in uptime(rows, now=T0)}
    assert table["all tickers"]["24 h"] == {"probes": 10, "uptime": 100.0}
    assert table["order book"]["24 h"] == {"probes": 10, "uptime": 70.0}


def test_a_window_counts_only_its_own_probes(tmp_path: Path) -> None:
    rows = read(_history(tmp_path))
    later = {u["what"]: u for u in uptime(rows, now=T0 + timedelta(hours=18))}
    # 18 h later the rounds at hours 3-9 are inside 24 h (the edge included); 5 and 7 failed.
    assert later["order book"]["24 h"] == {"probes": 7, "uptime": 71.4}
    assert later["order book"]["7 d"]["probes"] == 10


def test_the_chain_holds_and_a_deleted_failure_is_caught(tmp_path: Path) -> None:
    path = _history(tmp_path)
    rows = read(path)
    assert verify(rows) is None
    failures = [i for i, r in enumerate(rows) if not r["ok"]]
    del rows[failures[0]]
    assert verify(rows) is not None


def test_an_edited_result_is_caught(tmp_path: Path) -> None:
    rows = read(_history(tmp_path))
    failed = next(r for r in rows if not r["ok"])
    failed["ok"] = True
    assert "changed" in (verify(rows) or "")


def test_the_page_shows_it_or_says_there_is_none(tmp_path: Path) -> None:
    from argus.lui.status_page import _uptime_rows

    assert "No probe rounds recorded yet" in _uptime_rows([], T0.timestamp())
    shown = _uptime_rows(read(_history(tmp_path)), T0.timestamp())
    assert "70.0% of 10" in shown and "hash chain holds" in shown
