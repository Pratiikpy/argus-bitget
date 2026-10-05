"""Years of 8-hour perpetual funding settlements, kept so a funding backtest runs on any host.

The funding-rule backtest (`lui/research/signal_test.py`) needs the settlement record back years:
"buy BTC when funding is negative, sell when it turns positive, over the last two years". Two facts
decide where that record comes from:

- **Bitget keeps about ninety days.** ``/api/v2/mix/market/history-fund-rate`` returns 270
  settlements for BTCUSDT and an empty page after that (probed 2026-10-05). It is the venue the
  console trades, so its recent settlements are used where they exist.
- **Binance keeps every settlement since 2019** (``/fapi/v1/fundingRate``), but answers HTTP 451 to
  US hosts: the hosted console, on Vercel in the US, got "did not answer" on the live re-ask of
  round 41 while the desk's machine read it fine.

So the desk's machine sweeps Binance's record into ``data/funding_history.json`` (each symbol's
``[settlement ms, rate]`` pairs) and the deploy ships it; at question time the reader asks Binance
first, and when it does not answer it serves the snapshot topped up with Bitget's own recent
settlements past the snapshot's end — and the answer names which record it ran on.

    python -m argus.market.funding_history     # writes data/funding_history.json
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from argus.truth import http
from argus.truth.endpoints import BITGET_API
from argus.truth.paths import DATA_DIR

PATH: Final = DATA_DIR / "funding_history.json"
BINANCE: Final = "https://fapi.binance.com/fapi/v1/fundingRate"
BITGET: Final = f"{BITGET_API}/api/v2/mix/market/history-fund-rate"
SYMBOLS: Final = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "ADAUSDT",
                  "AVAXUSDT", "LINKUSDT", "LTCUSDT", "SUIUSDT", "TRXUSDT")
YEARS: Final = 5

Rows = list[tuple[datetime, float]]


def binance(symbol: str, since: datetime) -> Rows:
    """Every settlement since ``since`` from Binance, oldest first. Raises when it does not
    answer (HTTP 451 from US hosts)."""
    out: Rows = []
    start = int(since.timestamp() * 1000)
    for _ in range(40):
        rows = http.fetch_json(BINANCE, params={"symbol": symbol, "startTime": start,
                                                "limit": 1000}, timeout=30.0)
        if not rows:
            break
        out += [(datetime.fromtimestamp(int(r["fundingTime"]) / 1000, UTC),
                 float(r["fundingRate"])) for r in rows]
        if len(rows) < 1000:
            break
        start = int(rows[-1]["fundingTime"]) + 1
    return out


def bitget(symbol: str) -> Rows:
    """Bitget's own recent settlements (about ninety days), oldest first."""
    out: Rows = []
    for page in range(1, 10):
        body = http.fetch_json(BITGET, params={"symbol": symbol, "productType": "usdt-futures",
                                               "pageSize": 100, "pageNo": page}, timeout=20.0)
        rows = (body or {}).get("data") or []
        if not rows:
            break
        out += [(datetime.fromtimestamp(int(r["fundingTime"]) / 1000, UTC),
                 float(r["fundingRate"])) for r in rows]
    return sorted(out)


def load(path: Path = PATH) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def snapshot_rows(symbol: str, snapshot: dict[str, Any]) -> Rows:
    pairs = (snapshot.get("series") or {}).get(symbol) or []
    return [(datetime.fromtimestamp(int(ms) / 1000, UTC), float(rate)) for ms, rate in pairs]


def history(symbol: str, since: datetime, *, snapshot: dict[str, Any] | None = None
            ) -> tuple[Rows, str]:
    """(settlements since ``since``, the record they came from)."""
    try:
        rows = binance(symbol, since)
        if rows:
            return rows, f"Binance's {symbol} settlement record (every 8 hours)"
    except Exception:
        pass
    if snapshot is None:
        # the caller passes the snapshot it serves (the console's data directory); this layer
        # reads only its own default path
        snapshot = load()
    kept = [r for r in snapshot_rows(symbol, snapshot) if r[0] >= since]
    try:
        recent = bitget(symbol)
    except Exception:
        recent = []
    end = kept[-1][0] if kept else since
    joined = kept + [r for r in recent if r[0] > end]
    if not joined:
        return [], ""
    said = ("Binance's settlement record as swept on the desk's machine"
            + (f" to {end:%d %b %Y}, then Bitget's own settlements" if recent and kept else "")
            + " (Binance does not answer this host)")
    if not kept:
        said = "Bitget's own settlements (about 90 days; Binance does not answer this host)"
    return joined, said


def update(path: Path = PATH, *, now: datetime | None = None) -> dict[str, Any]:
    """Sweep Binance for every symbol in :data:`SYMBOLS` and write the snapshot. A symbol that
    does not answer keeps its previous rows."""
    stamp = now or datetime.now(UTC)
    previous = load(path).get("series") or {}
    series: dict[str, list[list[float]]] = {}
    for symbol in SYMBOLS:
        try:
            rows = binance(symbol, stamp - timedelta(days=365 * YEARS + 30))
            series[symbol] = [[int(t.timestamp() * 1000), rate] for t, rate in rows]
        except Exception:
            if symbol in previous:
                series[symbol] = previous[symbol]
    out = {"generated_at": stamp.isoformat(),
           "source": "Binance USD-M /fapi/v1/fundingRate, every 8-hour settlement",
           "series": series}
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8",
                    newline="\n")
    return out


def main() -> int:  # pragma: no cover - CLI, live network
    out = update()
    for symbol, rows in out["series"].items():
        print(symbol, len(rows))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["PATH", "SYMBOLS", "binance", "bitget", "history", "load", "snapshot_rows", "update"]
