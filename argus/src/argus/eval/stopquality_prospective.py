"""The stops recorded at the Rook head-to-head, graded on what the market then did.

`eval/stopquality_comparison.py` scored both desks' stops on history. At the same run it recorded
each stop — Rook's invalidation price, ARGUS's distance from the same last price — with the time it
was recorded. This module reads Bitget's hourly bars from that moment and says, for each name,
whether each stop was touched inside its 24-hour horizon: the out-of-sample check that no amount of
history can replace. It refuses to grade before the horizon has passed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parents[3] / "data"
SOURCE = DATA / "stopquality_comparison.json"
REPORT = DATA / "stopquality_prospective.json"
HORIZON = timedelta(hours=24)


def grade(*, now: datetime | None = None, fetch: Any = None) -> dict[str, Any]:
    from argus.market.history import CandleType, fetch_window

    record = json.loads(SOURCE.read_text("utf-8"))["prospective"]
    recorded = datetime.fromisoformat(record["recorded_at"])
    clock = now or datetime.now(UTC)
    if clock < recorded + HORIZON:
        return {"graded": False, "reason": f"the horizon ends {recorded + HORIZON:%Y-%m-%d %H:%M} "
                                           f"UTC; nothing is graded before it"}
    loader = fetch or (lambda s: fetch_window(s, start=recorded, end=recorded + HORIZON,
                                              interval="1H", candle_type=CandleType.MARKET,
                                              pause=0.05))
    rows = []
    for symbol, stop in record["stops"].items():
        last = stop.get("last")
        if not last:
            continue
        bars = [b for b in loader(symbol) if recorded <= b.ts < recorded + HORIZON]
        if not bars:
            rows.append({"symbol": symbol, "error": "no bars in the window"})
            continue
        low = min(float(b.low) for b in bars)
        argus_price = float(last) * (1 - float(stop["argus_distance"])) \
            if stop.get("argus_distance") else None
        rows.append({
            "symbol": symbol, "last": last, "window_low": low,
            "rook_stop": stop.get("rook"),
            "rook_touched": stop.get("rook") is not None and low <= float(stop["rook"]),
            "argus_stop": argus_price,
            "argus_touched": argus_price is not None and low <= argus_price,
            "close_after_24h": float(bars[-1].close)})
    graded = [r for r in rows if "error" not in r]
    out = {"graded": True, "recorded_at": record["recorded_at"],
           "graded_at": clock.isoformat(), "rows": rows,
           "rook_touched": sum(r["rook_touched"] for r in graded),
           "argus_touched": sum(r["argus_touched"] for r in graded), "names": len(graded)}
    REPORT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main() -> int:  # pragma: no cover - CLI
    out = grade()
    if not out["graded"]:
        print(out["reason"])
        return 0
    print(f"stops touched in the 24h after the run: Rook {out['rook_touched']} of "
          f"{out['names']}, ARGUS {out['argus_touched']} of {out['names']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
