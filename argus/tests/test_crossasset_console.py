"""The console's cross-asset hedge answer (`lui/crossasset.py`) over a constructed snapshot."""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from argus.lui import crossasset
from argus.lui.research.parse import holding_pairs
from argus.market import crossasset_feed as feed

END = datetime(2026, 9, 26, 12, tzinfo=UTC)


def _snapshot(seed: int = 3) -> feed.Snapshot:
    rng = random.Random(seed)
    end_ms = int(END.timestamp() // 3600) * feed.HOUR_MS
    hours = [end_ms - (62 * 24 - i) * feed.HOUR_MS for i in range(62 * 24)]
    closes: dict[str, list[list[float]]] = {}
    common = [rng.gauss(0, 0.004) for _ in hours]
    for key, beta in (("spot:RNVDAUSDT", 1.0), ("perp:NVDAUSDT", 1.0), ("perp:QQQUSDT", 0.7),
                      ("perp:BTCUSDT", 0.3), ("perp:ETHUSDT", 0.35)):
        price, rows = 100.0, []
        for t, shock in zip(hours, common, strict=True):
            price *= math.exp(beta * shock + rng.gauss(0, 0.002))
            rows.append([t, price])
        closes[key] = rows
    blob = {"written_at": (END - timedelta(hours=1)).isoformat(), "end_ms": end_ms,
            "closes": closes,
            "funding": {s: [[end_ms - k * 8 * feed.HOUR_MS, 0.0001] for k in range(40)]
                        for s in ("NVDAUSDT", "QQQUSDT", "BTCUSDT", "ETHUSDT")},
            "fees_bps": {s: 6.0 for s in feed.PERPS},
            "slippage": {s: [[1000.0, 1.0], [40000.0, 4.0]] for s in feed.PERPS},
            "missing": {}}
    return feed.from_blob(blob)


@pytest.mark.parametrize(("text", "book", "expected"), [
    ("should I hedge over the weekend?", "40% RNVDAUSDT, 30% RTSLAUSDT, 30% BTC", True),
    ("hedge $30k RTSLA and $20k ETH?", "", True),
    ("should I hedge NVDA?", "50% NVDA, 50% MSFT", False),
    ("what is RNVDA's price?", "40% RNVDAUSDT, 60% BTC", False),
])
def test_only_a_hedge_question_from_a_mixed_book_is_routed_here(
        text: str, book: str, expected: bool) -> None:
    assert crossasset.asks_for_cross_asset(text, book) is expected


def test_weights_and_dollars_are_both_read() -> None:
    spot, perps, nav, stated = crossasset.read_book("40% RNVDAUSDT, 30% RTSLAUSDT, 30% BTC",
                                                    holding_pairs)
    assert spot == {"RNVDAUSDT": 0.4, "RTSLAUSDT": 0.3} and perps == {"BTCUSDT": 0.3}
    assert nav == crossasset.DEFAULT_NAV and not stated
    spot, perps, nav, stated = crossasset.read_book("hedge $30k RTSLA and $20k ETH",
                                                    holding_pairs)
    assert nav == 50_000 and stated
    assert spot == {"RTSLAUSDT": 0.6} and perps == {"ETHUSDT": 0.4}


def test_the_answer_states_the_decision_the_book_and_the_inputs_age() -> None:
    lines, sources, data = crossasset.answer(
        "should I hedge?", "50% RNVDAUSDT, 50% BTC", snapshot=_snapshot(), now=END)
    assert lines[0].startswith("Bottom line: ")
    assert any(line.startswith("Read as: RNVDAUSDT spot 50%, BTCUSDT 50%") for line in lines)
    assert "read 1.0 hours ago" in lines[-1] and "no order is sent" in lines[-1]
    assert data["crossasset"]["horizon_hours"] >= 1
    assert {s.ref for s in sources} >= {"argus.desk.crossasset.route"}


def test_no_rtoken_named_asks_for_one_and_a_missing_snapshot_is_said(tmp_path: Path,
                                                                     monkeypatch) -> None:
    lines, _, data = crossasset.answer("hedge my BTC?", "", snapshot=_snapshot(), now=END)
    assert data == {"refused": True} and "name the rToken spot you hold" in lines[0]
    monkeypatch.setattr(crossasset, "load", lambda: feed.load(tmp_path / "absent.json"))
    lines, _, data = crossasset.answer("hedge 50% RNVDA and 50% BTC?", "", now=END)
    assert data == {"refused": True} and "inputs are not on this server" in lines[0]


def test_the_snapshot_marks_carried_bars_instead_of_inventing_them() -> None:
    snap = feed.from_blob({"written_at": END.isoformat(), "end_ms": 4 * feed.HOUR_MS,
                           "closes": {"perp:BTCUSDT": [[0, 10.0], [2 * feed.HOUR_MS, 12.0]]},
                           "funding": {}, "missing": {}})
    assert snap.close["perp:BTCUSDT"] == [10.0, 10.0, 12.0, 12.0]
    assert snap.real["perp:BTCUSDT"] == [True, False, True, False]
