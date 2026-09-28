"""Cross-asset snapshots are kept as versions and read as of any instant
(research/harvest/43-iceberg-deltalake.md: delta-rs's ``load_as_version`` contract)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from argus.market import crossasset_feed as feed

T0 = datetime(2026, 9, 28, 12, tzinfo=UTC)
HOUR = 3_600_000


def _snapshot(at: datetime, close: float) -> str:
    end = int(at.timestamp() // 3600) * HOUR
    blob = {"written_at": at.isoformat(), "end_ms": end,
            "closes": {"spot:BTCUSDT": [[end - 2 * HOUR, close], [end - HOUR, close + 1]]},
            "funding": {}, "fees_bps": {}, "slippage": {}, "missing": {}}
    return json.dumps(blob, separators=(",", ":")) + "\n"


def _write(history: Path, hours: int) -> None:
    for h in range(hours):
        at = T0 + timedelta(hours=h)
        feed.keep_version(_snapshot(at, 100.0 + h), at=at, history=history)


def test_each_instant_reads_the_version_current_then(tmp_path: Path) -> None:
    _write(tmp_path, 4)
    then = feed.load_as_of(T0 + timedelta(hours=1, minutes=30), history=tmp_path)
    assert then.written_at == T0 + timedelta(hours=1)
    assert then.close["spot:BTCUSDT"][0] == 101.0


def test_a_later_write_leaves_an_earlier_read_unchanged(tmp_path: Path) -> None:
    _write(tmp_path, 2)
    before = feed.load_as_of(T0 + timedelta(hours=1), history=tmp_path)
    _write(tmp_path, 5)
    assert feed.load_as_of(T0 + timedelta(hours=1), history=tmp_path) == before


def test_the_future_is_refused(tmp_path: Path) -> None:
    _write(tmp_path, 2)
    with pytest.raises(feed.FeedError, match="at or before"):
        feed.load_as_of(T0 - timedelta(minutes=1), history=tmp_path)


def test_old_versions_are_pruned_but_never_the_newest(tmp_path: Path) -> None:
    feed.keep_version(_snapshot(T0, 1.0), at=T0, history=tmp_path)
    later = T0 + timedelta(days=feed.KEEP_DAYS + 1)
    feed.keep_version(_snapshot(later, 2.0), at=later, history=tmp_path)
    kept = feed.versions(tmp_path)
    assert [r["written_at"] for r in kept] == [later.isoformat()]
    assert len(list(tmp_path.glob("*.json.gz"))) == 1


def test_a_changed_version_is_caught(tmp_path: Path) -> None:
    import gzip

    _write(tmp_path, 1)
    [version] = list(tmp_path.glob("*.json.gz"))
    version.write_bytes(gzip.compress(_snapshot(T0, 999.0).encode()))
    with pytest.raises(feed.FeedError, match="does not match its digest"):
        feed.load_as_of(T0, history=tmp_path)
