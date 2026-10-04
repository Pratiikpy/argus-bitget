"""Volatility-targeted sizing (`backtest/vol_target.py`) against its reference, and against fixed
size, on the same Bitget daily series.

Two questions, answered separately (build-list 1.6):

1. **Does the pure-Python forecast match the reference?** ``garchmethod``'s own
   ``walkforward_garch`` (MIT, ``research/repos-t2/garchmethod/scripts/garch_forecast.py``, run
   from its clone with the ``arch`` package, Student-t GARCH(1,1)) and ARGUS's pure-Python
   Student-t fit are run on the same closes; reported are the correlation of the two annualised
   forecast series, their median absolute gap, and how often the size each implies differs by
   more than 0.1.
2. **Does sizing by the forecast help?** The same signal — the reference's own demo, an EMA 9/21
   crossover long/flat — run at fixed size and at forecast-targeted size through
   ``backtest/engine.py``, which charges Bitget's taker fee on every change of weight (the
   reference's ``compare.py`` charges none). Both arms start on the first forecast day; a third
   arm scales the targeted weights to the fixed arm's average exposure, read from the whole
   sample, so the drawdowns compare like with like (an evaluation, not a rule anyone could run).

    python -m argus.eval.vol_target_comparison      # writes data/vol_target_comparison.json
"""

from __future__ import annotations

import json
import math
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.backtest import vol_target as vt
from argus.truth.paths import DATA_DIR

REFERENCE = (Path(__file__).resolve().parents[4] / "research" / "repos-t2" / "garchmethod"
             / "scripts")
OUT = DATA_DIR / "vol_target_comparison.json"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def _closes(symbol: str) -> tuple[list[datetime], list[float]]:
    from argus.market.history import fetch_window

    bars = fetch_window(symbol, start=datetime.now(UTC) - timedelta(days=1825), interval="1Dutc",
                        pause=0.05)
    return [b.ts for b in bars], [float(b.close) for b in bars]


def _reference(stamps: list[datetime], closes: list[float]) -> list[float | None]:
    """The reference's annualised forecast at each close (``None`` before its first)."""
    from importlib import import_module

    import pandas as pd

    sys.path.insert(0, str(REFERENCE))
    try:
        walkforward_garch = import_module("garch_forecast").walkforward_garch
    finally:
        sys.path.remove(str(REFERENCE))
    prices = pd.DataFrame({"date": [s.replace(tzinfo=None) for s in stamps], "close": closes})
    out = walkforward_garch(prices, periods_per_year=365)
    # the reference's row k is close k+1 (it drops the first close); its forecast is for k+2
    series: list[float | None] = [None] * len(closes)
    for k, value in enumerate(out["fcast_vol_ann"].tolist()):
        if value == value:  # not NaN
            series[k + 1] = float(value)
    return series


def _ema(values: list[float], span: int) -> list[float]:
    k, out, current = 2 / (span + 1), [], values[0]
    for v in values:
        current = current + k * (v - current)
        out.append(current)
    return out


def _arms(stamps: list[datetime], closes: list[float]) -> dict[str, Any]:
    from argus.backtest.engine import Bar, run
    from argus.cost.model import CostModel

    multipliers, target = vt.sizes(closes, 365)
    first = next(i for i, m in enumerate(multipliers) if m is not None)
    fast, slow = _ema(closes, 9), _ema(closes, 21)
    signal = [1.0 if f > s else 0.0 for f, s in zip(fast, slow, strict=True)]
    bars = [Bar(ts=t, close=Decimal(str(c))) for t, c in zip(stamps[first:], closes[first:],
                                                                strict=True)]
    fixed_w = signal[first:]
    sized_w = [s * (m or vt.MIN_SIZE) for s, m in zip(signal[first:], multipliers[first:],
                                                      strict=True)]

    def arm(name: str, weights: list[float]) -> dict[str, Any]:
        result = run(name, "x", bars, lambda _b, i: weights[i], cost=CostModel.bitget_perp(),
                     periods_per_year=365, max_weight=vt.MAX_SIZE)
        held = [w for w in result.weights if w]
        return {"net": result.net.as_dict(), "cost_bps": round(result.total_cost_bps, 1),
                "average_weight_when_in": round(sum(held) / len(held), 3) if held else 0.0}

    # the same targeted weights scaled so their average while in the market is the fixed arm's
    # 1.0: the targeted arm above holds more on average (its target is the training window's
    # volatility, which the later years ran below), so its drawdown also measures holding more.
    # The scale is read from the whole sample, so this arm is an evaluation, never a rule to run.
    held = [w for w in sized_w if w]
    scale = len(held) / sum(held) if held else 1.0
    matched_w = [w * scale for w in sized_w]
    return {"from": stamps[first].date().isoformat(), "days": len(bars),
            "target_vol_pct": round(target, 1),
            "fixed": arm("EMA 9/21, fixed size", fixed_w),
            "vol_targeted": arm("EMA 9/21, GARCH-targeted size", sized_w),
            "vol_targeted_same_exposure": arm("EMA 9/21, GARCH-targeted, exposure-matched",
                                              matched_w)}


def run_all() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for symbol in SYMBOLS:
        stamps, closes = _closes(symbol)
        ours = [None if v is None else math.sqrt(v * 365) for v in vt.forecasts(closes)]
        row: dict[str, Any] = {"closes": len(closes)}
        if REFERENCE.exists():
            theirs = _reference(stamps, closes)
            pairs = [(a, b) for a, b in zip(ours, theirs, strict=True)
                     if a is not None and b is not None]
            if pairs:
                xs, ys = [a for a, _ in pairs], [b for _, b in pairs]
                mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
                cov = sum((x - mx) * (y - my) for x, y in pairs)
                corr = cov / math.sqrt(sum((x - mx) ** 2 for x in xs)
                                       * sum((y - my) ** 2 for y in ys))
                gaps = sorted(abs(x - y) for x, y in pairs)
                target = 50.0
                differ = sum(1 for x, y in pairs
                             if abs(vt.size(x, target) - vt.size(y, target)) > 0.1)
                row["reference_match"] = {
                    "days_compared": len(pairs), "correlation": round(corr, 4),
                    "median_abs_gap_vol_points": round(gaps[len(gaps) // 2], 2),
                    "share_of_days_size_differs_by_over_0_1": round(differ / len(pairs), 3),
                    "mean_ours": round(mx, 1), "mean_reference": round(my, 1)}
        row["sizing"] = _arms(stamps, closes)
        rows[symbol] = row
    return {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "symbols": rows}


def main() -> int:  # pragma: no cover - CLI, live network
    result = run_all()
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
