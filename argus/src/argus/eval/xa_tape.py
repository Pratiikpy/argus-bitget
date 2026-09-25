"""XA-TAPE: the one frozen input every cross-asset contestant is run on.

The cross-asset arena (`eval/xa_arena.py`) compares ARGUS's hedge router with the five systems that
lead the Cross-Asset Execution sub-theme. A comparison is only same-input if every arm reads the
same bytes, so the market is recorded once, written to ``data/arena/xa_tape/`` with a SHA-256 per
file, and :func:`load` refuses to hand out a tape whose bytes changed.

What is on the tape, and where each series comes from (all public, keyless Bitget endpoints):

* **Hourly OHLCV** — spot rTokens (``R*USDT``, `/api/v2/spot/market/history-candles`) and USDT-M
  perpetuals (`/api/v2/mix/market/history-candles`). The page size (200) and the
  paging-by-``endTime`` pattern are the ones `market/rtoken_spot.py:268-306` already uses and
  measured.
* **Funding settlements** — `/api/v3/market/history-fund-rate`, the endpoint `research/carry.py:73`
  reads. Measured 2026-09-25: the venue serves **270 settlements (about 90 days) and no more** for
  every symbol probed (cursor 4 returns empty, and the v2 endpoint ends at the same place). So the
  tape has two periods: one where funding is observed and charged, and an earlier one where it is
  *unobserved* and recorded as such — never as zero.
* **Fees** — perpetual ``takerFeeRate`` read from `/api/v3/market/instruments` (0.0006 on every
  symbol probed). The instruments endpoint returns no fee for SPOT, so spot is charged 10 bps, the
  rate `research/executable_arb.py:17` records for spot rTokens.
* **Slippage** — one live order-book sweep per instrument at several notionals
  (`market/depth.py:OrderBook.sweep`). A 2026-09-25 book applied to the whole replay: stated as the
  limitation it is, because historic L2 is not served by REST.
* **Fear & Greed** — alternative.me daily index, the series VIGIL's own backtest reads
  (`norbert351/vigil src/backtest.js:24-28`).
* **Venue risk tables** — rToken collateral discount tiers (`/api/v3/market/discount-rate`) and the
  perpetual maintenance-margin ladder (`/api/v3/market/position-tier`), which Omni's margin model
  needs (`Jayanng/Omni omni/venue_risk.py:74-136`).
* **Macro calendar** — the FOMC and CPI dates already recorded, with their sources, in
  ``data/event_calendar.json``.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

BASE = "https://api.bitget.com"
TAPE_DIR = Path(__file__).resolve().parents[3] / "data" / "arena" / "xa_tape"
EVENT_CALENDAR = Path(__file__).resolve().parents[3] / "data" / "event_calendar.json"
HOUR_MS = 3_600_000
PAGE = 200

TAPE_END = datetime(2026, 9, 25, tzinfo=UTC)
"""Fixed so a re-freeze reproduces the same window rather than drifting with the clock."""

TAPE_DAYS = 210
"""Thirty days of warm-up plus two ninety-day scoring periods (see :data:`PERIODS`)."""

SPOT_SYMBOLS: tuple[str, ...] = (
    "RNVDAUSDT", "RAAPLUSDT", "RTSLAUSDT", "RQQQUSDT", "RSPYUSDT", "BTCUSDT", "ETHUSDT",
)
PERP_SYMBOLS: tuple[str, ...] = (
    "NVDAUSDT", "AAPLUSDT", "TSLAUSDT", "QQQUSDT", "SPYUSDT", "SQQQUSDT", "BTCUSDT", "ETHUSDT",
)
SLIPPAGE_NOTIONALS: tuple[int, ...] = (1_000, 5_000, 10_000, 20_000, 40_000)

SPOT_TAKER_BPS = 10.0
"""The instruments endpoint publishes no spot fee; see the module docstring."""


class TapeError(RuntimeError):
    """The tape is missing, incomplete, or its bytes no longer match the manifest."""


# --- fetching (network; run once by :func:`freeze`) -------------------------------------------


def _get(path: str, retries: int = 5) -> Any:  # pragma: no cover - network
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{BASE}{path}" if path.startswith("/") else path,
                                         headers={"Accept": "application/json",
                                                  "User-Agent": "argus-xa-tape/1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(0.6 * (attempt + 1))
            continue
        if isinstance(payload, dict) and payload.get("code") not in (None, "00000", 0):
            if str(payload.get("code")) == "429":
                time.sleep(1.0 * (attempt + 1))
                continue
            raise TapeError(f"{path}: {payload.get('code')} {payload.get('msg')}")
        return payload.get("data") if isinstance(payload, dict) and "data" in payload else payload
    raise TapeError(f"{path}: {last}")


def _fetch_bars(symbol: str, *, spot: bool, start_ms: int,
                end_ms: int) -> list[list[float]]:  # pragma: no cover - network
    """Hourly ``[ts, open, high, low, close, quote_volume]`` rows in ``[start_ms, end_ms)``."""
    path = ("/api/v2/spot/market/history-candles" if spot
            else "/api/v2/mix/market/history-candles")
    gran = "1h" if spot else "1H"
    extra = "" if spot else "&productType=usdt-futures"
    seen: dict[int, list[float]] = {}
    cursor = end_ms
    while cursor > start_ms:
        rows = _get(f"{path}?symbol={symbol}&granularity={gran}&endTime={cursor}"
                    f"&limit={PAGE}{extra}") or []
        for r in rows:
            ts = int(r[0])
            if start_ms <= ts < end_ms:
                # spot rows: [ts, o, h, l, c, baseVol, usdtVol, quoteVol];
                # mix rows:  [ts, o, h, l, c, baseVol, quoteVol]
                qv = float(r[6]) if len(r) > 6 else 0.0
                seen[ts] = [ts, float(r[1]), float(r[2]), float(r[3]), float(r[4]), qv]
        cursor -= PAGE * HOUR_MS
        time.sleep(0.08)
    return [seen[k] for k in sorted(seen)]


def _fetch_funding(symbol: str) -> list[list[float]]:  # pragma: no cover - network
    out: dict[int, float] = {}
    for cursor in range(1, 8):
        data = _get(f"/api/v3/market/history-fund-rate?category=USDT-FUTURES&symbol={symbol}"
                    f"&limit=100&cursor={cursor}") or {}
        rows = data.get("resultList") or [] if isinstance(data, dict) else []
        if not rows:
            break
        for row in rows:
            out[int(row["fundingRateTimestamp"])] = float(row["fundingRate"])
        if len(rows) < 100:
            break
    return [[k, out[k]] for k in sorted(out)]


def _fetch_fng() -> list[list[float]]:  # pragma: no cover - network
    data = _get("https://api.alternative.me/fng/?limit=400&format=json") or []
    rows = [[int(d["timestamp"]) * 1000, float(d["value"])] for d in data]
    return sorted(rows)


def _fetch_fees() -> dict[str, Any]:  # pragma: no cover - network
    perp: dict[str, float] = {}
    for sym in PERP_SYMBOLS:
        rows = _get(f"/api/v3/market/instruments?category=USDT-FUTURES&symbol={sym}") or []
        perp[sym] = float(rows[0]["takerFeeRate"]) * 10_000 if rows else 6.0
    return {"perp_taker_bps": perp, "spot_taker_bps": SPOT_TAKER_BPS,
            "spot_source": "research/executable_arb.py:17 (instruments publishes no spot fee)",
            "perp_source": "/api/v3/market/instruments takerFeeRate"}


def _fetch_slippage() -> dict[str, Any]:  # pragma: no cover - network
    from argus.market.depth import DepthError, fetch_orderbook

    out: dict[str, Any] = {}
    for kind, symbols, category in (("spot", SPOT_SYMBOLS, "SPOT"),
                                    ("perp", PERP_SYMBOLS, "USDT-FUTURES")):
        for sym in symbols:
            try:
                book = fetch_orderbook(sym, limit=200, category=category)
            except DepthError as exc:
                out[f"{kind}:{sym}"] = {"error": str(exc)}
                continue
            curve = []
            for n in SLIPPAGE_NOTIONALS:
                sides = []
                complete = True
                for direction in ("BUY", "SELL"):
                    try:
                        sw = book.sweep(Decimal(n), direction=direction)
                    except DepthError:
                        complete = False
                        continue
                    sides.append(float(sw.slippage_bps))
                    complete = complete and sw.complete
                curve.append({"notional": n, "slippage_bps": sum(sides) / len(sides)
                              if sides else None, "complete": complete})
            out[f"{kind}:{sym}"] = {"measured_at": book.fetched_at.isoformat(), "curve": curve}
    return out


def _fetch_venue_risk() -> dict[str, Any]:  # pragma: no cover - network
    disc = _get("/api/v3/market/discount-rate") or []
    wanted = {"RNVDA", "RAAPL", "RTSLA", "RQQQ", "RSPY"}
    discount = [e for e in disc if str(e.get("coin", "")).upper() in wanted]
    tiers = {sym: _get(f"/api/v3/market/position-tier?category=USDT-FUTURES&symbol={sym}")
             for sym in PERP_SYMBOLS}
    return {"discount_rate": discount, "position_tier": tiers}


MAG7_EXTRA_PERPS: tuple[str, ...] = ("METAUSDT", "AMZNUSDT", "MSFTUSDT", "GOOGLUSDT")
"""The four Mag7 perpetuals Crossfire averages (`crossfire/config.py:193-201`) that no contestant
trades; recorded so Crossfire's Mag7-vs-BTC divergence is computed on all seven names, not three."""


def _fetch_reality() -> dict[str, Any]:  # pragma: no cover - network
    """The session payloads Omni's classifier reads (`omni/session.py:105-160`), from the three
    public Reality endpoints its own `omni/bitget_public.py:104-118` calls."""
    return {
        "states": _get("/api/v3/reality/market/states"),
        "stock_info": {s: _get(f"/api/v3/reality/market/stock-info?symbol={s}")
                       for s in ("RNVDAUSDT", "RAAPLUSDT", "RTSLAUSDT")},
        "calendar": {c: _get(f"/api/v3/reality/market/calendar?code={c}")
                     for c in ("NVDA", "AAPL", "TSLA")},
    }


def freeze_extra(*, days: int = TAPE_DAYS, end: datetime = TAPE_END,
                 out_dir: Path = TAPE_DIR) -> dict[str, Any]:  # pragma: no cover - network
    """Add the Mag7 perpetuals and Omni's session payloads to an existing tape, re-hashing only
    what was added. Every earlier file keeps its bytes and its hash."""
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    end_ms = int(end.timestamp() * 1000)
    start_ms = int((end - timedelta(days=days)).timestamp() * 1000)
    blobs: dict[str, Any] = {f"perp_{s}.json": _fetch_bars(s, spot=False, start_ms=start_ms,
                                                           end_ms=end_ms)
                             for s in MAG7_EXTRA_PERPS}
    blobs["reality.json"] = _fetch_reality()
    for name, blob in blobs.items():
        text = json.dumps(blob, separators=(",", ":"), allow_nan=False)
        (out_dir / name).write_text(text, encoding="utf-8")
        manifest["files"][name] = {"sha256": hashlib.sha256(text.encode()).hexdigest(),
                                   "bytes": len(text.encode()),
                                   "rows": len(blob) if isinstance(blob, list) else None}
    manifest["extra_frozen_at"] = datetime.now(UTC).isoformat()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dict(manifest)


def freeze(*, days: int = TAPE_DAYS, end: datetime = TAPE_END,
           out_dir: Path = TAPE_DIR) -> dict[str, Any]:  # pragma: no cover - network
    """Record the tape and its manifest. Run once; every arena run then reads it with
    :func:`load`."""
    end_ms = int(end.timestamp() * 1000)
    start_ms = int((end - timedelta(days=days)).timestamp() * 1000)
    out_dir.mkdir(parents=True, exist_ok=True)
    blobs: dict[str, Any] = {}
    for sym in SPOT_SYMBOLS:
        blobs[f"spot_{sym}.json"] = _fetch_bars(sym, spot=True, start_ms=start_ms, end_ms=end_ms)
    for sym in PERP_SYMBOLS:
        blobs[f"perp_{sym}.json"] = _fetch_bars(sym, spot=False, start_ms=start_ms, end_ms=end_ms)
        blobs[f"funding_{sym}.json"] = _fetch_funding(sym)
    blobs["fng.json"] = _fetch_fng()
    blobs["fees.json"] = _fetch_fees()
    blobs["slippage.json"] = _fetch_slippage()
    blobs["venue_risk.json"] = _fetch_venue_risk()
    cal = json.loads(EVENT_CALENDAR.read_text(encoding="utf-8"))
    blobs["macro_calendar.json"] = {k: cal[k] for k in ("fomc", "cpi", "fomc_source", "cpi_source")}

    files: dict[str, Any] = {}
    for name, blob in blobs.items():
        text = json.dumps(blob, separators=(",", ":"), allow_nan=False)
        (out_dir / name).write_text(text, encoding="utf-8")
        rows = len(blob) if isinstance(blob, list) else None
        files[name] = {"sha256": hashlib.sha256(text.encode()).hexdigest(),
                       "bytes": len(text.encode()), "rows": rows}
    manifest = {
        "name": "xa_tape", "frozen_at": datetime.now(UTC).isoformat(),
        "window_start": datetime.fromtimestamp(start_ms / 1000, UTC).isoformat(),
        "window_end": end.isoformat(), "files": files,
        "licence_of_data": "Bitget public market data and alternative.me Fear & Greed; "
                           "recorded for evaluation, not redistributed as a product",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# --- loading -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tape:
    """Every series aligned to one hourly grid, forward-filled only where a bar is absent.

    ``stale[key][i]`` is True where the price at hour ``i`` was carried forward rather than
    traded, so a fill on it can be counted instead of hidden.
    """

    hours: tuple[int, ...]
    """Bar open times (ms), one per hour, strictly increasing."""

    open: dict[str, list[float]]
    high: dict[str, list[float]]
    low: dict[str, list[float]]
    close: dict[str, list[float]]
    quote_volume: dict[str, list[float]]
    stale: dict[str, list[bool]]
    first_traded: dict[str, int]
    """Index of the first real bar per series; before it the instrument did not exist."""

    funding: dict[str, list[tuple[int, float]]]
    fng: list[tuple[int, float]]
    fees: dict[str, Any]
    slippage: dict[str, Any]
    venue_risk: dict[str, Any]
    macro: dict[str, Any]
    reality: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def funding_observed_from(self) -> int:
        """The first hour (ms) from which every perpetual on the tape has a published settlement."""
        return max(rows[0][0] for rows in self.funding.values() if rows)

    def index_of(self, ts_ms: int) -> int:
        """Index of the bar whose open time is ``ts_ms`` (bars are exactly one hour apart)."""
        i = (ts_ms - self.hours[0]) // HOUR_MS
        if not 0 <= i < len(self.hours) or self.hours[i] != ts_ms:
            raise TapeError(f"{ts_ms} is not a bar open time on this tape")
        return int(i)


def key(kind: str, symbol: str) -> str:
    return f"{kind}:{symbol}"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(tape_dir: Path = TAPE_DIR) -> dict[str, Any]:
    """The manifest, after checking every file's bytes against it. Raises on any mismatch."""
    mpath = tape_dir / "manifest.json"
    if not mpath.exists():
        raise TapeError(f"no tape at {tape_dir}; run `python -m argus.eval.xa_tape` first")
    manifest: dict[str, Any] = json.loads(mpath.read_text(encoding="utf-8"))
    for name, meta in manifest["files"].items():
        path = tape_dir / name
        if not path.exists():
            raise TapeError(f"tape file {name} is missing")
        if _sha(path) != meta["sha256"]:
            raise TapeError(f"tape file {name} does not match its manifest hash")
    return manifest


def _align(rows: Sequence[Sequence[float]], hours: Sequence[int]) -> tuple[
        list[float], list[float], list[float], list[float], list[float], list[bool], int]:
    by_ts = {int(r[0]): r for r in rows}
    o: list[float] = []
    h: list[float] = []
    lo: list[float] = []
    c: list[float] = []
    v: list[float] = []
    stale: list[bool] = []
    first = -1
    last_close: float | None = None
    for i, ts in enumerate(hours):
        r = by_ts.get(ts)
        if r is not None and float(r[4]) > 0:
            if first < 0:
                first = i
            o.append(float(r[1]))
            h.append(float(r[2]))
            lo.append(float(r[3]))
            c.append(float(r[4]))
            v.append(float(r[5]))
            stale.append(False)
            last_close = float(r[4])
        else:
            px = last_close if last_close is not None else float("nan")
            o.append(px)
            h.append(px)
            lo.append(px)
            c.append(px)
            v.append(0.0)
            stale.append(True)
    return o, h, lo, c, v, stale, first


def build(blobs: dict[str, Any], manifest: dict[str, Any]) -> Tape:
    """Assemble a :class:`Tape` from the raw file contents (used by :func:`load` and by tests)."""
    series: dict[str, Sequence[Sequence[float]]] = {}
    for name, blob in blobs.items():
        if name.startswith("spot_"):
            series[key("spot", name[5:-5])] = blob
        elif name.startswith("perp_"):
            series[key("perp", name[5:-5])] = blob
    starts = [int(rows[0][0]) for rows in series.values() if rows]
    ends = [int(rows[-1][0]) for rows in series.values() if rows]
    if not starts:
        raise TapeError("the tape holds no bars")
    hours = tuple(range(min(starts), max(ends) + HOUR_MS, HOUR_MS))
    o: dict[str, list[float]] = {}
    h: dict[str, list[float]] = {}
    lo: dict[str, list[float]] = {}
    c: dict[str, list[float]] = {}
    v: dict[str, list[float]] = {}
    st: dict[str, list[bool]] = {}
    first: dict[str, int] = {}
    for k, rows in series.items():
        o[k], h[k], lo[k], c[k], v[k], st[k], first[k] = _align(rows, hours)
    funding = {name[8:-5]: [(int(a), float(b)) for a, b in blob]
               for name, blob in blobs.items() if name.startswith("funding_")}
    return Tape(
        hours=hours, open=o, high=h, low=lo, close=c, quote_volume=v, stale=st,
        first_traded=first, funding=funding,
        fng=[(int(a), float(b)) for a, b in blobs.get("fng.json", [])],
        fees=blobs.get("fees.json", {}), slippage=blobs.get("slippage.json", {}),
        venue_risk=blobs.get("venue_risk.json", {}), macro=blobs.get("macro_calendar.json", {}),
        reality=blobs.get("reality.json", {}), manifest=manifest,
    )


def load(tape_dir: Path = TAPE_DIR) -> Tape:
    """The frozen tape, byte-verified. Refuses a tape that changed since it was frozen."""
    manifest = verify(tape_dir)
    blobs = {name: json.loads((tape_dir / name).read_text(encoding="utf-8"))
             for name in manifest["files"]}
    return build(blobs, manifest)


def tape_digest(manifest: dict[str, Any]) -> str:
    """One hash over every file hash — what a result cites to say which tape it ran on."""
    joined = "".join(f"{n}:{m['sha256']}" for n, m in sorted(manifest["files"].items()))
    return hashlib.sha256(joined.encode()).hexdigest()


def daily_closes(tape: Tape, k: str, *, hour_utc: int = 0) -> list[tuple[int, float]]:
    """Closes sampled at ``hour_utc`` each day (the bar ending at that hour), for the daily-cadence
    contestants. Only real (non-stale-before-listing) values are returned."""
    out: list[tuple[int, float]] = []
    first = tape.first_traded.get(k, -1)
    for i, ts in enumerate(tape.hours):
        end = ts + HOUR_MS
        if first >= 0 and i >= first and (end // HOUR_MS) % 24 == hour_utc:
            out.append((end, tape.close[k][i]))
    return out


def iter_keys(tape: Tape, kind: str) -> Iterable[str]:
    return (k for k in tape.close if k.startswith(kind + ":"))


def main() -> int:  # pragma: no cover - CLI
    import sys

    manifest = freeze_extra() if "--extra" in sys.argv else freeze()
    total = sum(m["bytes"] for m in manifest["files"].values())
    print(f"froze {len(manifest['files'])} files, {total/1e6:.2f} MB, digest "
          f"{tape_digest(manifest)[:16]} -> {TAPE_DIR}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "PERP_SYMBOLS", "SPOT_SYMBOLS", "SPOT_TAKER_BPS", "TAPE_DIR", "Tape", "TapeError", "build",
    "daily_closes", "freeze", "iter_keys", "key", "load", "tape_digest", "verify",
]
