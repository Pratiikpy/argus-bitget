"""A frozen, hashed copy of the hourly rToken return history every allocation bake-off reads.

`eval/allocation_comparison.py` re-fetches its 60 days of candles live on every run, which is right
for a report that should move with the market and wrong for a claim that must be reproduced: two
runs an hour apart compare different books, and "reproducibility.identical=true" there only ever
meant *the same process computed the same thing twice from one fetch*. A result a reviewer can
re-derive needs the input pinned, so this module fetches once, aligns once, writes the aligned
return columns to ``data/allocation_returns_snapshot.json`` with a SHA-256 of their canonical
encoding, and every later reader verifies that digest before computing anything.

The fetch deliberately reaches further back than the 60 days the older comparison used. The
venue's hourly endpoint pages back well beyond its documented 90-day range (measured 2026-09-25:
QQQUSDT returned 4,799 hourly bars from 2026-03-09 when asked for 200 days), and statistical power
in a walk-forward test is the number of non-overlapping held-out windows — nine was the old count,
which is too few for a Holm-corrected sign test to separate anything but the largest effects.

Alignment is the same intersection rule `desk.portfolio.align` uses (no forward fill), applied to
the same simple returns `desk.portfolio.returns` computes, so a column here is exactly what the
live desk would have computed from the same candles.

    python -m argus.eval.allocation_snapshot --days 185
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from argus.eval.artefact import write

SNAPSHOT_PATH = Path(__file__).resolve().parents[3] / "data" / "allocation_returns_snapshot.json"


class SnapshotError(RuntimeError):
    """The snapshot is missing, or its content no longer matches the digest it was written with."""


def digest(stamps: Sequence[str], columns: dict[str, list[float]]) -> str:
    """SHA-256 over a canonical encoding: sorted names, the timestamps, every value by ``repr``.

    ``repr`` of a Python float round-trips exactly, so two machines that load the same file and
    hash it get the same digest; hashing the JSON text instead would hash whitespace choices.
    """
    h = hashlib.sha256()
    h.update("|".join(stamps).encode())
    for name in sorted(columns):
        h.update(name.encode())
        h.update(",".join(repr(float(v)) for v in columns[name]).encode())
    return h.hexdigest()


def fetch_snapshot(days: int = 185, *, attempts: int = 6, backoff: float = 20.0,
                   pause: float = 5.0, page_pause: float = 0.6,
                   ) -> dict[str, Any]:  # pragma: no cover - live network
    """Fetch every rToken's hourly market candles back ``days``, align, and return the snapshot.

    Gentle on purpose. The first attempt at 400 days with the history module's default 0.15s page
    spacing drew HTTP 429 on six of twelve symbols (2026-09-25, a venue shared with every other
    process on this machine), and a snapshot with half the universe is a different experiment, so
    pages are spaced ``page_pause`` apart, a 429 backs off from ``backoff`` seconds doubling, and a
    symbol that still fails is named in ``fetch_failures`` -- :func:`main` refuses to write it.
    185 days is enough: TQQQUSDT and SQQQUSDT were listed on 2026-03-31, so the aligned
    intersection of all twelve cannot start earlier, however far back the others go."""
    from argus.desk.portfolio import returns
    from argus.market.bitget import RTOKEN_SYMBOLS
    from argus.market.history import CandleType, fetch_window

    start = datetime.now(UTC) - timedelta(days=days)
    series: dict[str, dict[datetime, float]] = {}
    first_bar: dict[str, str] = {}
    bars_per_symbol: dict[str, int] = {}
    failures: dict[str, str] = {}
    for symbol in RTOKEN_SYMBOLS:
        last_error = ""
        for attempt in range(attempts):
            try:
                bars = fetch_window(symbol, start=start, interval="1H",
                                    candle_type=CandleType.MARKET, pause=page_pause)
            except Exception as exc:  # retried, then recorded -- never silently dropped
                last_error = f"{type(exc).__name__}: {str(exc)[:100]}"
                time.sleep(backoff * (2 ** attempt))
                continue
            if bars:
                first_bar[symbol] = bars[0].ts.isoformat()
                bars_per_symbol[symbol] = len(bars)
            series[symbol] = returns([(c.ts, float(c.close)) for c in bars])
            break
        else:
            failures[symbol] = last_error
        time.sleep(pause)
    common = sorted(set.intersection(*(set(v) for v in series.values()))) if series else []
    columns = {name: [series[name][t] for t in common] for name in sorted(series)}
    stamps = [t.isoformat() for t in common]
    return {
        "fetched_at": datetime.now(UTC).isoformat(),
        "requested_days": days,
        "interval": "1H",
        "candle_type": "MARKET",
        "symbols": sorted(columns),
        "first_bar_per_symbol": first_bar,
        "bars_per_symbol": bars_per_symbol,
        "fetch_failures": failures,
        "n_aligned_returns": len(stamps),
        "timestamps": stamps,
        "columns": columns,
        "digest": digest(stamps, columns),
    }


def load_snapshot(path: Path = SNAPSHOT_PATH) -> tuple[list[str], dict[str, list[float]], str]:
    """``(timestamps, columns, digest)``, after re-hashing the content and checking the digest."""
    if not path.exists():
        raise SnapshotError(
            f"{path} does not exist; run `python -m argus.eval.allocation_snapshot`")
    blob = json.loads(path.read_text(encoding="utf-8"))
    stamps = list(blob["timestamps"])
    columns = {str(k): [float(v) for v in vals] for k, vals in blob["columns"].items()}
    recomputed = digest(stamps, columns)
    if recomputed != blob["digest"]:
        raise SnapshotError(
            f"snapshot digest mismatch: file says {blob['digest'][:12]}, content hashes to "
            f"{recomputed[:12]} -- the input was edited after it was written"
        )
    return stamps, columns, recomputed


def main() -> int:  # pragma: no cover - CLI, live network
    import argparse

    parser = argparse.ArgumentParser(description="freeze the rToken hourly return history")
    parser.add_argument("--days", type=int, default=185)
    args = parser.parse_args()
    snap = fetch_snapshot(args.days)
    if snap["fetch_failures"]:
        print(f"REFUSED -- symbols not fetched, snapshot not written: {snap['fetch_failures']}")
        return 1
    write(SNAPSHOT_PATH, snap, indent=0)
    print(f"{len(snap['symbols'])} symbols, {snap['n_aligned_returns']} aligned hourly returns "
          f"from {snap['timestamps'][0] if snap['timestamps'] else '-'}; first bar per symbol: "
          f"{snap['first_bar_per_symbol']}; digest {snap['digest'][:16]}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = ["SNAPSHOT_PATH", "SnapshotError", "digest", "fetch_snapshot", "load_snapshot"]
