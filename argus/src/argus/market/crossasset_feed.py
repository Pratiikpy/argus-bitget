"""The live inputs the cross-asset hedge router needs, kept as a snapshot the console can read.

`desk/crossasset.py` routes hedges for a book that holds rToken spot and crypto perpetuals at once,
and until 2026-09-27 it ran only on the frozen tape of its evaluation arena (`eval/xa_tape.py`):
no question a trader could type reached it (audit finding 164). The router needs sixty days of
hourly closes for every instrument in play, each perpetual's funding settlements and the cost of
trading it, which is about a hundred reads of Bitget's public endpoints — too many to make while a
trader waits. So this module reads them on the data job's schedule and writes one snapshot
(:data:`SNAPSHOT_PATH`); the console's answer (`lui/crossasset.py`) reads the snapshot, says how
old it is, and prices the book at the moment of the question.

**Same endpoints, same shapes as the arena's tape** (`eval/xa_tape.py` records the measurements
behind each choice): hourly candles from `/api/v2/spot/market/history-candles` and
`/api/v2/mix/market/history-candles`, paged back 200 bars at a time by ``endTime``; funding from
`/api/v3/market/history-fund-rate`; the taker fee from `/api/v3/market/instruments`; slippage walked
on a live order book at five sizes (`market/depth.py`). Every read goes through `truth/http.py`.
A series that did not arrive is recorded as missing, never filled in: the router refuses a model it
cannot estimate, and the answer says which series was absent.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.truth import http
from argus.truth.endpoints import BITGET_API
from argus.truth.paths import DATA_DIR

BASE = BITGET_API
SNAPSHOT_PATH = DATA_DIR / "crossasset_snapshot.json"
HISTORY_DIR = DATA_DIR / "crossasset_history"
"""Every snapshot written, one compressed file per version, and ``versions.jsonl`` listing them.

**Why.** ``write_snapshot`` replaced the one file on every run, so nothing could answer "what did
the router see an hour before this decision" (research/harvest/43-iceberg-deltalake.md). The study
picked delta-rs for its ``DeltaTable.load_as_version`` time travel (``table.py:558-596``). What is
kept here is that interface — read the version current at an instant, earlier reads unaffected by
later writes — without the dependency: a compiled table engine for one feed that writes a few
hundred kilobytes an hour is more machinery than the question needs. Each version is the exact
bytes ``write_snapshot`` produced, gzipped, named by its SHA-256 so a changed file is caught on
read; :data:`KEEP_DAYS` bounds the disk it takes."""

KEEP_DAYS = 30
"""Versions older than this are pruned when a new one is written; the newest is never pruned."""
HOUR_MS = 3_600_000
PAGE = 200
LOOKBACK_DAYS = 62
"""The router's sixty-day covariance lookback (`RouterConfig.lookback_hours`), plus two days so the
newest bar at answer time still has a full window behind it."""

SPOT: tuple[str, ...] = ("RNVDAUSDT", "RAAPLUSDT", "RTSLAUSDT", "RQQQUSDT", "RSPYUSDT")
PERPS: tuple[str, ...] = ("NVDAUSDT", "AAPLUSDT", "TSLAUSDT", "QQQUSDT", "SPYUSDT", "SQQQUSDT",
                          "BTCUSDT", "ETHUSDT")
"""The arena's universe (`eval/xa_arena.py` HEDGE_PERPS and the spot rTokens it models)."""

SLIPPAGE_NOTIONALS: tuple[int, ...] = (1_000, 5_000, 10_000, 20_000, 40_000)
DEFAULT_TAKER_BPS = 6.0


class FeedError(RuntimeError):
    """A read the snapshot could not do without; the message names it."""


def _data(path: str, **params: Any) -> Any:
    payload = http.fetch_json(f"{BASE}{path}", params=params, timeout=30)
    if isinstance(payload, dict) and payload.get("code") not in (None, "00000", 0):
        raise FeedError(f"{path}: {payload.get('code')} {payload.get('msg')}")
    return payload.get("data") if isinstance(payload, dict) else payload


def fetch_closes(symbol: str, *, spot: bool, end_ms: int, days: int = LOOKBACK_DAYS,
                 read: Callable[..., Any] = _data) -> list[tuple[int, float]]:
    """Hourly ``(open_ms, close)`` for ``days`` before ``end_ms``, oldest first."""
    path = ("/api/v2/spot/market/history-candles" if spot
            else "/api/v2/mix/market/history-candles")
    params: dict[str, Any] = {"symbol": symbol, "granularity": "1h" if spot else "1H",
                              "limit": PAGE}
    if not spot:
        params["productType"] = "usdt-futures"
    start_ms = end_ms - days * 24 * HOUR_MS
    seen: dict[int, float] = {}
    cursor = end_ms
    while cursor > start_ms:
        rows = read(path, endTime=cursor, **params) or []
        for row in rows:
            ts = int(row[0])
            if start_ms <= ts < end_ms and float(row[4]) > 0:
                seen[ts] = float(row[4])
        cursor -= PAGE * HOUR_MS
    return sorted(seen.items())


def fetch_funding(symbol: str, *, read: Callable[..., Any] = _data) -> list[tuple[int, float]]:
    """Every settlement the venue serves for ``symbol`` (about ninety days), oldest first."""
    out: dict[int, float] = {}
    for cursor in range(1, 8):
        data = read("/api/v3/market/history-fund-rate", category="USDT-FUTURES", symbol=symbol,
                    limit=100, cursor=cursor) or {}
        rows = data.get("resultList") or [] if isinstance(data, dict) else []
        for row in rows:
            out[int(row["fundingRateTimestamp"])] = float(row["fundingRate"])
        if len(rows) < 100:
            break
    return sorted(out.items())


def fetch_taker_bps(symbol: str, *, read: Callable[..., Any] = _data) -> float:
    rows = read("/api/v3/market/instruments", category="USDT-FUTURES", symbol=symbol) or []
    return float(rows[0]["takerFeeRate"]) * 10_000 if rows else DEFAULT_TAKER_BPS


def fetch_slippage(symbol: str) -> list[tuple[float, float]]:
    """``(notional, slippage_bps)`` walked on the live book, both sides averaged."""
    from argus.market.depth import DepthError, fetch_orderbook

    book = fetch_orderbook(symbol, limit=200, category="USDT-FUTURES")
    curve: list[tuple[float, float]] = []
    for n in SLIPPAGE_NOTIONALS:
        sides = []
        for direction in ("BUY", "SELL"):
            try:
                sides.append(float(book.sweep(Decimal(n), direction=direction).slippage_bps))
            except DepthError:
                continue
        if sides:
            curve.append((float(n), sum(sides) / len(sides)))
    return curve


def write_snapshot(path: Path = SNAPSHOT_PATH, *, now: datetime | None = None,
                   history: Path | None = None) -> dict[str, Any]:
    """Read every input the router needs and write the snapshot. Returns what was written."""
    at = now or datetime.now(UTC)
    end_ms = int(at.timestamp() // 3600) * HOUR_MS
    closes: dict[str, list[tuple[int, float]]] = {}
    missing: dict[str, str] = {}
    for kind, symbols in (("spot", SPOT), ("perp", PERPS)):
        for symbol in symbols:
            try:
                closes[f"{kind}:{symbol}"] = fetch_closes(symbol, spot=kind == "spot",
                                                          end_ms=end_ms)
            except (http.RpcError, FeedError, ValueError, KeyError, IndexError) as exc:
                missing[f"{kind}:{symbol}"] = str(exc)[:160]
            time.sleep(0.05)
    funding: dict[str, list[tuple[int, float]]] = {}
    fees: dict[str, float] = {}
    slippage: dict[str, list[tuple[float, float]]] = {}

    def absent(label: str, symbol: str, exc: Exception) -> None:
        # each input is named when it did not arrive, never invented
        missing[f"{label}:{symbol}"] = f"{type(exc).__name__}: {str(exc)[:120]}"

    for symbol in PERPS:
        try:
            funding[symbol] = fetch_funding(symbol)
        except Exception as exc:
            absent("funding", symbol, exc)
        try:
            fees[symbol] = fetch_taker_bps(symbol)
        except Exception as exc:
            absent("fee", symbol, exc)
        try:
            slippage[symbol] = fetch_slippage(symbol)
        except Exception as exc:
            absent("slippage", symbol, exc)
    blob = {"written_at": at.isoformat(), "end_ms": end_ms,
            "closes": {k: [[t, c] for t, c in v] for k, v in closes.items()},
            "funding": {k: [[t, r] for t, r in v] for k, v in funding.items()},
            "fees_bps": fees,
            "slippage": {k: [[n, s] for n, s in v] for k, v in slippage.items()},
            "missing": missing}
    text = json.dumps(blob, separators=(",", ":")) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    if path == SNAPSHOT_PATH or history is not None:
        keep_version(text, at=at, history=history or HISTORY_DIR)
    return blob


def keep_version(text: str, *, at: datetime, history: Path = HISTORY_DIR) -> str:
    """Store one snapshot as a version and prune what has aged out. Returns the version's digest."""
    import gzip
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    history.mkdir(parents=True, exist_ok=True)
    (history / f"{digest[:16]}.json.gz").write_bytes(gzip.compress(text.encode("utf-8"), 9))
    rows = versions(history)
    rows.append({"written_at": at.isoformat(), "sha256": digest})
    cutoff = at - timedelta(days=KEEP_DAYS)
    kept = [r for r in rows if datetime.fromisoformat(r["written_at"]) >= cutoff] or rows[-1:]
    live = {r["sha256"][:16] for r in kept}
    for stale in history.glob("*.json.gz"):
        if stale.name.split(".")[0] not in live:
            stale.unlink()
    (history / "versions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8", newline="\n")
    return digest


def versions(history: Path = HISTORY_DIR) -> list[dict[str, Any]]:
    """Every kept version, oldest first."""
    index = history / "versions.jsonl"
    if not index.exists():
        return []
    return [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_as_of(at: datetime, history: Path = HISTORY_DIR) -> Snapshot:
    """The snapshot the router would have read at ``at``: the newest version written at or before
    it. Refuses rather than substituting a later one, which would be reading the future."""
    import gzip
    import hashlib

    earlier = [r for r in versions(history) if datetime.fromisoformat(r["written_at"]) <= at]
    if not earlier:
        raise FeedError(f"no cross-asset snapshot was written at or before {at.isoformat()}")
    chosen = earlier[-1]
    raw_text = gzip.decompress((history / f"{chosen['sha256'][:16]}.json.gz").read_bytes())
    if hashlib.sha256(raw_text).hexdigest() != chosen["sha256"]:
        raise FeedError(f"the version written {chosen['written_at']} does not match its digest")
    return from_blob(json.loads(raw_text.decode("utf-8")))


@dataclass(frozen=True)
class Snapshot:
    written_at: datetime
    hours: tuple[int, ...]
    close: dict[str, list[float]]
    real: dict[str, list[bool]]
    funding: dict[str, list[tuple[int, float]]]
    fees_bps: dict[str, float]
    slippage: dict[str, list[tuple[float, float]]]
    missing: dict[str, str]

    def age_hours(self, now: datetime) -> float:
        return (now - self.written_at).total_seconds() / 3600


def load(path: Path = SNAPSHOT_PATH) -> Snapshot:
    """The snapshot on one hourly grid; a missing bar carries the last close and is marked."""
    if not path.exists():
        raise FeedError(f"no cross-asset snapshot at {path}; run "
                        f"`python -m argus.market.crossasset_feed`")
    blob = json.loads(path.read_text(encoding="utf-8"))
    return from_blob(blob)


def from_blob(blob: Mapping[str, Any]) -> Snapshot:
    series = {k: {int(t): float(c) for t, c in rows} for k, rows in blob["closes"].items() if rows}
    if not series:
        raise FeedError("the snapshot holds no closes")
    start = min(min(v) for v in series.values())
    hours = tuple(range(start, int(blob["end_ms"]), HOUR_MS))
    close: dict[str, list[float]] = {}
    real: dict[str, list[bool]] = {}
    for k, by_ts in series.items():
        last = float("nan")
        c: list[float] = []
        r: list[bool] = []
        for ts in hours:
            if ts in by_ts:
                last = by_ts[ts]
                r.append(True)
            else:
                r.append(False)
            c.append(last)
        close[k], real[k] = c, r
    return Snapshot(
        written_at=datetime.fromisoformat(str(blob["written_at"])), hours=hours, close=close,
        real=real,
        funding={k: [(int(t), float(v)) for t, v in rows] for k, rows in blob["funding"].items()},
        fees_bps={k: float(v) for k, v in blob.get("fees_bps", {}).items()},
        slippage={k: [(float(n), float(s)) for n, s in rows]
                  for k, rows in blob.get("slippage", {}).items()},
        missing=dict(blob.get("missing", {})))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - network
    del argv
    blob = write_snapshot()
    print(f"cross-asset snapshot: {len(blob['closes'])} series, "
          f"{len(blob['funding'])} funding histories, {len(blob['missing'])} missing -> "
          f"{SNAPSHOT_PATH}")
    for name, why in blob["missing"].items():
        print(f"  missing {name}: {why}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
