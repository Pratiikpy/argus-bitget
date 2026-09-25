"""Tests for the XA arena (`eval/xa_arena.py`), its tape (`eval/xa_tape.py`) and the rival
runners' shared line protocol (`eval/baselines/xa_rivals/_proto.py`), on small synthetic tapes.
The real frozen tape and the rival clones are exercised only when present on disk."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from argus.eval import xa_arena as xa
from argus.eval import xa_tape
from argus.eval.baselines import xa_rivals

HOUR = 3_600_000
KEYS_SPOT = ("RNVDAUSDT", "RAAPLUSDT", "RTSLAUSDT", "BTCUSDT")
KEYS_PERP = ("BTCUSDT", "ETHUSDT", "NVDAUSDT")


def _synthetic_blobs(start: datetime, hours: int) -> dict[str, Any]:
    t0 = int(start.timestamp() * 1000)
    blobs: dict[str, Any] = {}
    for n, sym in enumerate(KEYS_SPOT):
        blobs[f"spot_{sym}.json"] = [
            [t0 + j * HOUR, 100 + n + 0.01 * j, 101 + n + 0.01 * j, 99 + n + 0.01 * j,
             100 + n + 0.01 * (j + 1), 1e5] for j in range(hours)]
    for n, sym in enumerate(KEYS_PERP):
        blobs[f"perp_{sym}.json"] = [
            [t0 + j * HOUR, 200 + n + 0.02 * j, 201 + n + 0.02 * j, 199 + n + 0.02 * j,
             200 + n + 0.02 * (j + 1), 1e6] for j in range(hours)]
        blobs[f"funding_{sym}.json"] = [[t0 + j * HOUR, 0.0001] for j in range(0, hours, 8)]
    blobs["fees.json"] = {"perp_taker_bps": {s: 6.0 for s in KEYS_PERP}, "spot_taker_bps": 10.0}
    curve = [{"notional": 1_000, "slippage_bps": 1.0}, {"notional": 40_000, "slippage_bps": 5.0}]
    blobs["slippage.json"] = {**{f"spot:{s}": {"curve": curve} for s in KEYS_SPOT},
                              **{f"perp:{s}": {"curve": curve} for s in KEYS_PERP}}
    blobs["fng.json"] = [[t0 + d * 24 * HOUR, 50.0] for d in range(hours // 24)]
    return blobs


def _write_tape(tmp: Path, blobs: dict[str, Any]) -> None:
    files = {}
    for name, blob in blobs.items():
        text = json.dumps(blob, separators=(",", ":"))
        (tmp / name).write_text(text, encoding="utf-8")
        files[name] = {"sha256": hashlib.sha256(text.encode()).hexdigest(),
                       "bytes": len(text.encode()), "rows": None}
    (tmp / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")


def test_tape_load_verifies_bytes_and_aligns_series(tmp_path: Path) -> None:
    blobs = _synthetic_blobs(datetime(2026, 6, 1, tzinfo=UTC), 72)
    del blobs["spot_RNVDAUSDT.json"][10]  # a missing bar is carried forward and marked stale
    _write_tape(tmp_path, blobs)
    tape = xa_tape.load(tmp_path)
    assert len(tape.hours) == 72
    k = "spot:RNVDAUSDT"
    assert tape.stale[k][10] and not tape.stale[k][9]
    assert tape.close[k][10] == tape.close[k][9]
    assert tape.funding_observed_from == tape.hours[0]
    assert tape.index_of(tape.hours[5]) == 5
    with pytest.raises(xa_tape.TapeError):
        tape.index_of(tape.hours[5] + 1)
    (tmp_path / "fees.json").write_text("{}", encoding="utf-8")
    with pytest.raises(xa_tape.TapeError, match="does not match"):
        xa_tape.load(tmp_path)


def test_tape_digest_is_order_independent() -> None:
    a = {"files": {"x": {"sha256": "1"}, "y": {"sha256": "2"}}}
    b = {"files": {"y": {"sha256": "2"}, "x": {"sha256": "1"}}}
    assert xa_tape.tape_digest(a) == xa_tape.tape_digest(b)


def test_metrics_on_known_series() -> None:
    assert xa.returns([110.0, 99.0], start=100.0) == pytest.approx([0.1, -0.1])
    assert xa.max_drawdown([110.0, 99.0, 120.0], start=100.0) == pytest.approx(0.1)
    xs = [-0.05, -0.01, 0.0, 0.01] * 5
    assert xa.cvar(xs, 0.05) == pytest.approx(0.05)
    r = [0.001, -0.002, 0.003, 0.0]
    assert xa.ce(r, 5.0) == pytest.approx(
        (statistics.fmean(r) - 2.5 * statistics.pvariance(r)) * 8760)
    assert math.isnan(xa.sharpe([0.0, 0.0]))


def test_holm_is_monotone_and_capped() -> None:
    adj = xa.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)
    assert xa.holm({"a": 0.9, "b": 0.8})["a"] == 1.0


def test_stationary_bootstrap_and_paired_test_detect_a_real_difference() -> None:
    paths = xa.stationary_bootstrap(200, mean_block=24.0, reps=300, seed=7)
    assert len(paths) == 300 and all(len(p) == 200 and max(p) < 200 for p in paths)
    assert paths == xa.stationary_bootstrap(200, mean_block=24.0, reps=300, seed=7)
    import random

    rng = random.Random(1)
    b = [rng.gauss(0, 0.01) for _ in range(200)]
    a = [v + 0.002 for v in b]
    t = xa.paired_test(a, b, statistics.fmean, paths)
    assert t["diff"] == pytest.approx(0.002)
    assert t["p_a_better"] < 0.05 < t["p_b_better"]


def test_newey_west_alpha_recovers_intercept_and_beta() -> None:
    import random

    rng = random.Random(3)
    x = [rng.gauss(0, 0.01) for _ in range(2000)]
    y = [0.0001 + 0.5 * v + rng.gauss(0, 0.001) for v in x]
    out = xa.newey_west_alpha(y, x)
    assert out["beta"] == pytest.approx(0.5, abs=0.02)
    assert out["alpha_annual"] == pytest.approx(0.0001 * 8760, rel=0.1)
    assert out["t_alpha"] > 3


def _tiny_arena(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> xa_tape.Tape:
    start = datetime(2026, 6, 20, tzinfo=UTC)
    _write_tape(tmp_path, _synthetic_blobs(start, 24 * 20))
    tape = xa_tape.load(tmp_path)
    monkeypatch.setitem(xa.PERIODS, "T", (datetime(2026, 6, 28, tzinfo=UTC),
                                          datetime(2026, 7, 8, tzinfo=UTC)))
    monkeypatch.setattr(xa, "BOOK_SPOT", {"RNVDAUSDT": 0.5})
    monkeypatch.setattr(xa, "BOOK_PERP", {"BTCUSDT": 0.25})
    return tape


def test_hold_marks_the_book_to_market_and_pays_funding(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    tape = _tiny_arena(tmp_path, monkeypatch)
    res = xa.run(tape, xa.Hold(), "T")
    led = res.ledger
    assert led.fills == 0 and led.fees == 0.0
    s, e = res.start_i, res.end_i - 1
    spot_q = 50_000.0 / tape.open["spot:RNVDAUSDT"][s]
    perp_q = 25_000.0 / tape.open["perp:BTCUSDT"][s]
    pnl_perp = perp_q * (tape.close["perp:BTCUSDT"][e] - tape.open["perp:BTCUSDT"][s])
    want = 50_000.0 + spot_q * tape.close["spot:RNVDAUSDT"][e] + pnl_perp - led.funding
    assert led.nav[-1] == pytest.approx(want, rel=1e-12)
    assert led.funding > 0  # a long pays a positive rate


def test_fills_charge_fee_and_slippage_at_next_open(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    tape = _tiny_arena(tmp_path, monkeypatch)

    class OneShort(xa.Hold):
        name = "one_short"

        def decide(self, msg: dict[str, Any]) -> dict[str, Any]:
            if self.decisions == 0:
                self.decisions += 1
                return {"orders": [{"kind": "perp", "symbol": "BTCUSDT", "usd": -10_000.0}]}
            return {"orders": []}

    res = xa.run(tape, OneShort(), "T")
    assert res.ledger.fills == 1
    assert res.ledger.fees == pytest.approx(10_000.0 * 6.0 / 1e4)
    assert res.ledger.slippage == pytest.approx(
        10_000.0 * (1.0 + 4.0 * 9_000 / 39_000) / 1e4)


def test_gross_limit_refuses_an_oversized_order(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    tape = _tiny_arena(tmp_path, monkeypatch)

    class Huge(xa.Hold):
        name = "huge"

        def decide(self, msg: dict[str, Any]) -> dict[str, Any]:
            return {"orders": [{"kind": "perp", "symbol": "ETHUSDT", "usd": 10_000_000.0}]}

    res = xa.run(tape, Huge(), "T")
    assert res.ledger.fills == 0 and res.ledger.refused_orders > 0


def test_contestants_only_ever_see_the_past(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    tape = _tiny_arena(tmp_path, monkeypatch)
    seen: list[tuple[int, int]] = []

    class Spy(xa.Hold):
        name = "spy"

        def decide(self, msg: dict[str, Any]) -> dict[str, Any]:
            seen.append((int(msg["i"]), len(self.h.hours)))
            return {"orders": []}

    xa.run(tape, Spy(), "T")
    assert seen and all(n == i + 1 for i, n in seen)


def test_argus_contestant_runs_deterministically(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    tape = _tiny_arena(tmp_path, monkeypatch)
    a = xa.run(tape, xa.Argus(xa.RouterConfig(lookback_hours=96, min_blocks=3)), "T")
    b = xa.run(tape, xa.Argus(xa.RouterConfig(lookback_hours=96, min_blocks=3)), "T")
    assert a.ledger.nav == b.ledger.nav
    assert a.finish["decisions"] == len(a.ledger.nav) - 1


def test_proto_tape_rebuilds_daily_rows_from_hourly_bars() -> None:
    sys.path.insert(0, str(Path(xa_rivals.__file__).parent))
    try:
        from _proto import Tape as ProtoTape
    finally:
        sys.path.pop(0)
    t = ProtoTape({"keys": ["spot:X"]})
    base = int(datetime(2026, 6, 1, tzinfo=UTC).timestamp() * 1000)
    for j in range(30):
        t.push({"i": j, "ts": base + (j + 1) * HOUR,
                "bars": {"spot:X": [10.0 + j, 11.0 + j, 9.0 + j, 10.5 + j, 1.0, 0.0]}})
    rows = t.daily_rows("spot:X", limit=5)
    assert [r[0] for r in rows] == [base, base + 24 * HOUR]
    assert rows[0][1:] == [10.0, 34.0, 9.0, 33.5]
    assert t.change("spot:X", 24) == pytest.approx(39.5 / 15.5 - 1)
    assert t.close("spot:X", 100) is None


def test_rival_pins_hash_committed_bytes(tmp_path: Path) -> None:
    lf = tmp_path / "a.py"
    crlf = tmp_path / "b.py"
    lf.write_bytes(b"x = 1\ny = 2\n")
    crlf.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert xa_rivals.committed_sha256(lf) == xa_rivals.committed_sha256(crlf)


def test_missing_clone_is_reported_not_crashed(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARGUS_RESEARCH_DIR", str(tmp_path))
    with pytest.raises(xa_rivals.RivalUnavailable, match="clone missing"):
        xa_rivals.verify("omni")
    xa.UNAVAILABLE.clear()
    assert xa.rival_contestants() == []
    assert "all_rivals" in xa.UNAVAILABLE
    xa.UNAVAILABLE.clear()


@pytest.mark.skipif(not all(xa_rivals.clone_path(n).is_dir() for n in xa_rivals.PINS),
                    reason="rival clones not present")
def test_local_rival_clones_match_their_pins() -> None:
    for name in xa_rivals.PINS:
        xa_rivals.verify(name)


@pytest.mark.skipif(not (xa_tape.TAPE_DIR / "manifest.json").exists(),
                    reason="frozen tape not present")
def test_frozen_tape_verifies_and_covers_both_periods() -> None:
    tape = xa_tape.load()
    for start, end in xa.PERIODS.values():
        tape.index_of(int(start.timestamp() * 1000))
        tape.index_of(int(end.timestamp() * 1000) - HOUR)
    assert tape.funding_observed_from <= int(xa.PERIODS["B"][0].timestamp() * 1000)
    assert tape.funding_observed_from > int(xa.PERIODS["A"][0].timestamp() * 1000)
