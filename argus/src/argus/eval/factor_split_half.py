"""The split-half reliability gate, measured: planted factors first, then every live primitive.

`research/factor_lab.py` gained a fifth certification requirement on 2026-09-25 — GoEmotions'
split-half reliability test (google-research, Apache-2.0, `goemotions/ppca.py:87-99,136-217`),
uncentred so that a factor's mean payoff is what gets tested. A gate earns its place only by what it
does, so this module measures three things and publishes them whichever way they fall.

1. **It must fail noise.** Factors whose readings are coin flips, scored against the *real* hourly
   returns of each instrument, so the null is tested on the fat tails and weekend gaps it will meet.
   The false-pass rate should sit near :data:`~argus.research.factor_lab.SPLIT_HALF_ALPHA`.
2. **It must pass a real edge.** Factors that read the sign of the next bar correctly with a
   planted probability above one half, on the same real returns. The pass rate at each accuracy is
   the gate's power, and the accuracy at which it first reaches 80% is the smallest edge it can see.
3. **What it does to today's library.** Every vetted primitive on each of the twelve instruments
   `data/track1_study.json` covers, 90 days of hourly bars: which of them pass today's gates — the
   lab's cost and out-of-sample gates, the four anti-overfit gates of `research/overfit_gates.py`,
   and the lab's full certification — and how many of those fail split-half. Then
   :func:`~argus.research.factor_lab.reliable_components` on each instrument's library: how many
   independent payoff dimensions eight primitives actually reproduce.

No model is called anywhere in this module.

    python -m argus.eval.factor_split_half
"""

from __future__ import annotations

import random
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.backtest.engine import Bar
from argus.research.factor_lab import (
    PRIMITIVES,
    SPLIT_HALF_ALPHA,
    SPLIT_HALF_BLOCK_BARS,
    SPLIT_HALF_MIN_BLOCKS,
    Evaluator,
    Factor,
    FactorLab,
    Lifecycle,
    payoff_halves,
    reliable_components,
    split_half,
)

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "factor_split_half.json"

SYMBOLS = ("NVDAUSDT", "TSLAUSDT", "AAPLUSDT", "MSFTUSDT", "METAUSDT", "GOOGLUSDT", "AMZNUSDT",
           "COINUSDT", "MSTRUSDT", "QQQUSDT", "TQQQUSDT", "SQQQUSDT")
"""The twelve instruments `data/track1_study.json` reports on."""

BLOCK = SPLIT_HALF_BLOCK_BARS
"""The lab's own block, so the planted checks measure the gate exactly as it runs."""

NOISE_PER_SYMBOL = 40
PLANTED_ACCURACIES = (0.54, 0.56, 0.58, 0.60, 0.62)
PLANTED_PER_LEVEL = 20
SWEEP_BLOCKS = (24, 48, 72, 120, 168)
"""Block sizes re-measured on real returns. The lab's 120 was chosen on synthetic fat-tailed series
before this ran (`factor_lab.SPLIT_HALF_BLOCK_BARS`); this sweep is the check, not the choice."""
SWEEP_PER_SYMBOL = 15
SEED = 20260925


def fetch_bars(symbol: str, days: int = 90) -> list[Bar]:  # pragma: no cover - network
    """90 days of hourly bars. Retries only on HTTP 429: the first full run lost MSFTUSDT and
    METAUSDT to Bitget's rate limit, and a symbol dropped for pacing is a gap, not a result."""
    import time

    from argus.market.history import CandleType, HistoryError, fetch_range

    for attempt in range(4):
        try:
            candles = fetch_range(symbol, days=days, interval="1H", candle_type=CandleType.MARKET)
            break
        except HistoryError as exc:
            if "429" not in str(exc) or attempt == 3:
                raise
            time.sleep(10.0 * (attempt + 1))
    return [Bar(ts=c.ts, close=c.close,
                extra={"volume": float(c.volume), "high": float(c.high), "low": float(c.low)})
            for c in candles]


def next_returns(bars: Sequence[Bar]) -> list[float]:
    """The return that followed each bar, as `factor_lab.Evaluator.payoff_series` pairs them."""
    out = []
    for i in range(len(bars) - 1):
        a, b = float(bars[i].close), float(bars[i + 1].close)
        out.append((b - a) / a if a > 0 else 0.0)
    return out


def planted(returns: Sequence[float], rng: random.Random, *, accuracy: float | None) -> list[float]:
    """A factor reading per bar. ``None`` is a coin flip; otherwise the reading has the sign of the
    return that follows with probability ``accuracy`` — an edge whose size is known exactly."""
    values = []
    for r in returns:
        if accuracy is None or r == 0:
            values.append(1.0 if rng.random() < 0.5 else -1.0)
            continue
        right = rng.random() < accuracy
        values.append((1.0 if r > 0 else -1.0) * (1.0 if right else -1.0))
    return values


def planted_checks(returns_by_symbol: dict[str, list[float]]) -> dict[str, Any]:
    rng = random.Random(SEED)
    noise_pass = noise_total = 0
    for returns in returns_by_symbol.values():
        for _ in range(NOISE_PER_SYMBOL):
            noise_total += 1
            noise_pass += split_half(planted(returns, rng, accuracy=None), returns,
                                     block=BLOCK).passed
    power: dict[str, Any] = {}
    for accuracy in PLANTED_ACCURACIES:
        passed = total = 0
        for returns in returns_by_symbol.values():
            for _ in range(PLANTED_PER_LEVEL):
                total += 1
                passed += split_half(planted(returns, rng, accuracy=accuracy), returns,
                                     block=BLOCK).passed
        power[f"{accuracy:.2f}"] = {"passed": passed, "of": total, "rate": round(passed / total, 4)}
    detectable = next((a for a, row in power.items() if row["rate"] >= 0.8), None)
    sweep: dict[str, Any] = {}
    for block in SWEEP_BLOCKS:
        noise = real = total = 0
        for returns in returns_by_symbol.values():
            for _ in range(SWEEP_PER_SYMBOL):
                total += 1
                noise += split_half(planted(returns, rng, accuracy=None), returns,
                                    block=block).passed
                real += split_half(planted(returns, rng, accuracy=0.60), returns,
                                   block=block).passed
        sweep[str(block)] = {"noise_false_pass_rate": round(noise / total, 4),
                             "power_at_0.60": round(real / total, 4), "of": total}
    return {
        "noise": {"passed": noise_pass, "of": noise_total,
                  "false_pass_rate": round(noise_pass / noise_total, 4),
                  "nominal_alpha": SPLIT_HALF_ALPHA},
        "planted_real_edge_power": power,
        "smallest_accuracy_detected_80pct": detectable,
        "block": BLOCK,
        "block_sweep_on_real_returns": sweep,
    }


def library_on(symbol: str, bars: list[Bar]) -> dict[str, Any]:
    """Every primitive through today's gates and through split-half, on one instrument."""
    from argus.research.overfit_gates import GatesError, screen

    evaluator = Evaluator(bars)
    lab = FactorLab(evaluator=evaluator)
    for name in PRIMITIVES:
        lab.submit(Factor(name=name, expression=name, rationale="session-structure hypothesis"))
    reached_dsr = {r.factor.name for r in lab.records if r.state is Lifecycle.OOS_TESTED}
    lab.gate()
    certified_before = {
        r.factor.name for r in lab.records
        if r.state is Lifecycle.CERTIFIED or r.rejection_reason.startswith("split-half")
    }
    rows = []
    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    for name in PRIMITIVES:
        factor = Factor(name=name, expression=name, rationale="")
        reliability = evaluator.split_half(factor)
        try:
            anti = screen(bars, name).cleared_every_gate
        except GatesError:
            anti = False
        rows.append({
            "factor": name,
            "passes_cost_and_oos": name in reached_dsr,
            "passes_anti_overfit": anti,
            "certified_before_split_half": name in certified_before,
            "split_half": reliability.as_dict(),
        })
        values, returns = evaluator.payoff_series(factor)
        xs, ys = payoff_halves(values, returns, block=BLOCK)
        x_rows.append(xs)
        y_rows.append(ys)
    blocks = min(len(r) for r in x_rows)
    x = [[x_rows[f][b] for f in range(len(x_rows))] for b in range(blocks)]
    y = [[y_rows[f][b] for f in range(len(y_rows))] for b in range(blocks)]
    return {"symbol": symbol, "bars": len(bars), "factors": rows,
            "library": reliable_components(x, y)}


def summarise(libraries: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for lib in libraries for row in lib["factors"]]

    def failing(flag: str) -> dict[str, Any]:
        passing = [r for r in rows if r[flag]]
        fail = [f"{lib['symbol']}:{r['factor']}" for lib in libraries for r in lib["factors"]
                if r[flag] and r["split_half"]["outcome"] != "pass"]
        return {"currently_passing": len(passing), "fail_split_half": len(fail),
                "which": fail}

    return {
        "factor_instrument_pairs": len(rows),
        "split_half_pass": sum(1 for r in rows if r["split_half"]["outcome"] == "pass"),
        "split_half_inconclusive": sum(1 for r in rows
                                       if r["split_half"]["outcome"] == "inconclusive"),
        "of_those_passing_cost_and_oos": failing("passes_cost_and_oos"),
        "of_those_passing_anti_overfit": failing("passes_anti_overfit"),
        "of_those_certified_before_split_half": failing("certified_before_split_half"),
        "reliable_payoff_dimensions": {lib["symbol"]: lib["library"]["reliable_components"]
                                       for lib in libraries},
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - network
    import time

    from argus.eval.artefact import write

    symbols = tuple(argv) if argv else SYMBOLS
    libraries: list[dict[str, Any]] = []
    returns_by_symbol: dict[str, list[float]] = {}
    failed: dict[str, str] = {}
    for symbol in symbols:
        try:
            bars = fetch_bars(symbol)
        except Exception as exc:  # reported per symbol in the artefact, not swallowed
            failed[symbol] = f"{type(exc).__name__}: {exc}"
            continue
        if len(bars) < BLOCK * SPLIT_HALF_MIN_BLOCKS:
            failed[symbol] = f"only {len(bars)} bars"
            continue
        returns_by_symbol[symbol] = next_returns(bars)
        libraries.append(library_on(symbol, bars))
        print(f"  {symbol}: {len(bars)} bars", file=sys.stderr)
        time.sleep(2.0)  # the venue rate-limits consecutive 90-day pulls
    if not libraries:
        print(f"no instrument could be fetched; nothing written: {failed}", file=sys.stderr)
        return 1
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "block_bars": BLOCK,
        "symbols_failed": failed,
        "planted": planted_checks(returns_by_symbol),
        "summary": summarise(libraries),
        "libraries": libraries,
        "qwen_calls": 0,
    }
    write(REPORT_PATH, report)
    p, s = report["planted"], report["summary"]
    print("noise false-pass rate:", p["noise"]["false_pass_rate"], f"({p['noise']['passed']}/"
          f"{p['noise']['of']})")
    print("planted power:", {k: v["rate"] for k, v in p["planted_real_edge_power"].items()})
    for key in ("of_those_passing_cost_and_oos", "of_those_passing_anti_overfit",
                "of_those_certified_before_split_half"):
        print(f"{key}: {s[key]['currently_passing']} passing, {s[key]['fail_split_half']} fail "
              f"split-half {s[key]['which']}")
    print("reliable payoff dimensions:", s["reliable_payoff_dimensions"])
    print(f"written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))


__all__ = ["BLOCK", "SYMBOLS", "library_on", "next_returns", "planted", "planted_checks",
           "summarise"]
