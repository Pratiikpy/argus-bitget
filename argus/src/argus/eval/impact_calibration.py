"""Calibrate the square-root impact law on Bitget's own order books.

`cost/model.py` carried an impact law that did not match the one it cited. cvxportfolio's model
(`cvxportfolio/costs.py`, `TransactionCost`, equation 2.2 of Boyd et al., *Multi-Period Trading
via Convex Optimization*, 2017) is, in dollars,

    b * sigma * |x| ** (3/2) / V ** (1/2)

— so per dollar traded the cost is ``b * sigma * (|x| / V) ** (1/2)``: the square root of
participation, scaled by volatility, with ``b`` "typically of the order of 1". ARGUS charged
``coefficient * participation ** 1.5`` per dollar with a coefficient of 10 nobody had measured and
no volatility at all, which put the modelled impact of every order it priced at 0.0005bps against a
measured median of 2.4bps on the same books (2026-09-24, from `data/depth.json`). The rival review
of that day named it; this module measures the replacement.

**The data.** `data/depth.json` records, for each of the twelve traded names, the slippage of
sweeping $1k, $5k, $25k and $100k through the live 50-level book. For each sweep the walk beyond
half the spread is regressed on ``sigma * sqrt(size / ADV)`` through the origin, with ``sigma`` the
name's daily return volatility over the last 30 daily bars and ADV its 24-hour notional volume, both
from Bitget. The slope is ``b``.

**What a book sweep is, and is not.** It is the cost of taking the size *now*, with no
replenishment — an upper bound for an order worked patiently over the day, which is what the
square-root law describes. So the calibrated ``b`` is the urgent end of the range, and the free
exponent fitted alongside it (log-log slope of walk on participation) is reported rather than
used: a book is steeper than the square root at small sizes, and the published law is kept.

    python -m argus.eval.impact_calibration
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parents[3] / "data"
DEPTH_PATH = DATA / "depth.json"
REPORT_PATH = DATA / "impact_calibration.json"
SIGMA_DAYS = 30


@dataclass(frozen=True, slots=True)
class SweepPoint:
    symbol: str
    notional: float
    participation: float
    sigma_daily_bps: float
    walk_bps: float
    """Slippage beyond half the quoted spread: the part the size itself cost."""


def points(depth: dict[str, Any], adv: dict[str, float],
           sigma_bps: dict[str, float]) -> list[SweepPoint]:
    """Every complete buy-side sweep with a known ADV and volatility."""
    out: list[SweepPoint] = []
    for book in depth.get("books", []):
        symbol = str(book.get("symbol"))
        if symbol not in adv or symbol not in sigma_bps or adv[symbol] <= 0:
            continue
        half_spread = float(book.get("spread_bps", 0.0)) / 2
        for sweep in book.get("sweeps", []):
            if sweep.get("side") != "BUY" or not sweep.get("complete"):
                continue
            size = float(sweep["requested_notional"])
            out.append(SweepPoint(symbol, size, size / adv[symbol], sigma_bps[symbol],
                                  max(0.0, float(sweep["slippage_bps"]) - half_spread)))
    return out


def fit(rows: list[SweepPoint]) -> dict[str, float]:
    """``b`` by least squares through the origin, its R-squared, and the free exponent."""
    xs = [r.sigma_daily_bps * math.sqrt(r.participation) for r in rows]
    ys = [r.walk_bps for r in rows]
    b = sum(x * y for x, y in zip(xs, ys, strict=True)) / sum(x * x for x in xs)
    mean = statistics.fmean(ys)
    residual = sum((y - b * x) ** 2 for x, y in zip(xs, ys, strict=True))
    total = sum((y - mean) ** 2 for y in ys)
    logs = [(math.log(r.participation), math.log(r.walk_bps / r.sigma_daily_bps))
            for r in rows if r.walk_bps > 0]
    mx = statistics.fmean(x for x, _ in logs)
    my = statistics.fmean(y for _, y in logs)
    slope = (sum((x - mx) * (y - my) for x, y in logs)
             / sum((x - mx) ** 2 for x, _ in logs))
    return {"b": b, "r_squared": 1 - residual / total if total else 0.0,
            "free_exponent": slope,
            "median_sigma_daily_bps": statistics.median(r.sigma_daily_bps for r in rows),
            "median_walk_bps": statistics.median(ys),
            "median_old_model_bps": statistics.median(10 * r.participation ** 1.5
                                                      for r in rows)}


def measure() -> dict[str, Any]:  # pragma: no cover - live
    from argus.market.bitget import fetch_tickers
    from argus.market.history import CandleType, fetch

    depth = json.loads(DEPTH_PATH.read_text(encoding="utf-8"))
    tickers = fetch_tickers()
    adv: dict[str, float] = {}
    sigma: dict[str, float] = {}
    for book in depth.get("books", []):
        symbol = str(book.get("symbol"))
        ticker = tickers.get(symbol)
        if ticker is None:
            continue
        adv[symbol] = float(ticker.base_volume) * float(ticker.last)
        bars = fetch(symbol, interval="1D", candle_type=CandleType.MARKET, recent=True,
                     limit=SIGMA_DAYS + 1)
        closes = [float(b.close) for b in bars]
        moves = [b / a - 1 for a, b in pairwise(closes) if a > 0]
        if len(moves) >= 10:
            sigma[symbol] = statistics.pstdev(moves) * 10_000
    rows = points(depth, adv, sigma)
    return {"generated_at": datetime.now(UTC).isoformat(),
            "depth_measured_at": depth.get("generated_at"),
            "points": len(rows), **fit(rows),
            "rows": [asdict(r) for r in rows]}


def main() -> int:  # pragma: no cover - CLI
    report = measure()
    REPORT_PATH.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{report['points']} sweeps: b = {report['b']:.3f}, R^2 = {report['r_squared']:.2f}, "
          f"free exponent {report['free_exponent']:.2f}, median sigma "
          f"{report['median_sigma_daily_bps']:.0f}bps/day; measured walk median "
          f"{report['median_walk_bps']:.2f}bps against the old model's "
          f"{report['median_old_model_bps']:.4f}bps")
    print(f"written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["SweepPoint", "fit", "points"]
