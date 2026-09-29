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

**The bars are frozen (2026-09-29).** The first runs fetched ninety days live on every call, so the
published result could not be reproduced: by the time anyone re-ran it the window had moved. The
twelve series now live in ``data/factor_split_half_bars.json`` with a sha256 per symbol and one over
the whole set, checked on every load, and the module reads them by default. ``--fetch`` re-pulls the
fixture first (Bitget public candles, keyless). The frozen window is the published run's own, the
ninety days ending at :data:`WINDOW_END` (the 2026-09-25 artefact's ``generated_at``), re-pulled on
2026-09-29. **The venue's candles for that window had moved in the meantime**: re-pulled, the
bounds and the bar count match the 2026-09-25 run exactly (2,159 bars, 06-27 15:00 to 09-25 13:00
UTC) but NVDAUSDT's split-half statistics differ from the published ones by up to a few percent, so
the artefact was regenerated from the fixture rather than claimed to be the old one. Each row also
keeps the open and the quote volume, which the lab does not use and `eval/factor_quality_rivals.py`
needs (FactorMiner's panel carries ``open``, ``amount`` and ``vwap``).

    python -m argus.eval.factor_split_half            # from the fixture
    python -m argus.eval.factor_split_half --fetch    # re-freeze first, then run
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
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
from argus.truth.paths import DATA_DIR

DATA = DATA_DIR
REPORT_PATH = DATA / "factor_split_half.json"
FIXTURE = DATA / "factor_split_half_bars.json"

WINDOW_DAYS = 90
WINDOW_END = datetime(2026, 9, 25, 14, 17, 35, 964747, tzinfo=UTC)
"""The instant the published 2026-09-25 run was generated. The fixture is the ninety days before it,
so the frozen evidence is the window that run measured, not whichever ninety days precede a later
re-freeze."""

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


class FixtureError(RuntimeError):
    """The frozen bars are missing, altered, or not prices."""


def pull_candles(symbol: str, *, start: datetime, end: datetime,
                 pause: float = 0.15) -> list[list[str]]:  # pragma: no cover - network
    """Raw hourly rows ``[ts, open, high, low, close, volume, quote_volume]`` in ``[start, end]``.

    The paging is `market/history.fetch_window`'s (walk ``endTime`` back to the oldest bar already
    held; stop when a page is empty or the window stops moving) with `history.fetch`'s request
    parameters, read through the same `history._get`. It is repeated here only because
    `history.Candle` drops the seventh field, the quote volume, which FactorMiner's panel needs as
    ``amount``. Retries only on HTTP 429, as :func:`fetch_bars` does.
    """
    import time

    from argus.market import history

    seen: dict[int, list[str]] = {}
    cursor: datetime = end
    while True:
        params = {"category": "USDT-FUTURES", "symbol": symbol, "interval": "1H",
                  "type": str(history.CandleType.MARKET), "limit": str(history.MAX_LIMIT),
                  "endTime": str(int(cursor.timestamp() * 1000))}
        page: list[list[str]] = []
        for attempt in range(4):
            try:
                page = history._get(params)
                break
            except history.HistoryError as exc:
                if "429" not in str(exc) or attempt == 3:
                    raise
                time.sleep(10.0 * (attempt + 1))
        page = [row for row in page if len(row) >= 7]
        if not page:
            break
        new = {int(row[0]): [str(v) for v in row[:7]] for row in page if int(row[0]) not in seen}
        if not new:
            break
        seen.update(new)
        oldest = datetime.fromtimestamp(min(int(row[0]) for row in page) / 1000, tz=UTC)
        if oldest <= start or oldest >= cursor:
            break  # covered, or the window stopped moving (the venue is clamping)
        cursor = oldest
        time.sleep(pause)
    lo, hi = start.timestamp() * 1000, end.timestamp() * 1000
    rows = []
    for ms in sorted(seen):
        if lo <= ms <= hi:
            row = seen[ms]
            if any(Decimal(v) <= 0 for v in row[1:5]):
                raise FixtureError(f"{symbol}: non-positive price in candle {row}")
            rows.append([datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat(), *row[1:7]])
    return rows


def _digest(rows: Any) -> str:
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode("utf-8")).hexdigest()


def freeze(symbols: Sequence[str] = SYMBOLS, *, end: datetime = WINDOW_END,
           days: int = WINDOW_DAYS, path: Path = FIXTURE) -> Path:  # pragma: no cover - network
    """Pull every symbol's window once and write the hashed fixture every later run reads."""
    import time

    from argus.truth.artefact import write

    start = end - timedelta(days=days)
    series: dict[str, Any] = {}
    for symbol in symbols:
        rows = pull_candles(symbol, start=start, end=end)
        series[symbol] = {"sha256": _digest(rows), "rows": rows}
        print(f"  froze {symbol}: {len(rows)} bars", file=sys.stderr)
        time.sleep(2.0)  # the venue rate-limits consecutive 90-day pulls
    write(path, {
        "interval": "1H",
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "fetched_at": datetime.now(UTC).isoformat(),
        "source": "Bitget public /api/v3/market/history-candles, category USDT-FUTURES, type "
                  "MARKET, keyless; row = [ts, open, high, low, close, volume, quote_volume]",
        "sha256": _digest({s: series[s]["sha256"] for s in sorted(series)}),
        "series": series,
    }, indent=0)
    return path


def load_fixture(path: Path = FIXTURE) -> tuple[dict[str, list[list[str]]], str]:
    """Every frozen series and the digest over the whole set, each hash checked before use."""
    if not path.exists():
        raise FixtureError(f"{path} is missing; run `python -m argus.eval.factor_split_half "
                           f"--fetch` once to freeze it")
    blob = json.loads(path.read_text(encoding="utf-8"))
    series: dict[str, list[list[str]]] = {}
    for symbol, entry in blob["series"].items():
        if _digest(entry["rows"]) != entry["sha256"]:
            raise FixtureError(f"{path}: {symbol}'s rows do not match their recorded sha256")
        series[symbol] = entry["rows"]
    overall = _digest({s: blob["series"][s]["sha256"] for s in sorted(blob["series"])})
    if overall != blob["sha256"]:
        raise FixtureError(f"{path}: the set of series does not match its recorded sha256")
    return series, overall


def bars_from_rows(rows: Sequence[Sequence[str]]) -> list[Bar]:
    """Fixture rows as the lab's bars: exactly the conversion :func:`fetch_bars` applies."""
    return [Bar(ts=datetime.fromisoformat(r[0]), close=Decimal(r[4]),
                extra={"volume": float(r[5]), "high": float(r[2]), "low": float(r[3])})
            for r in rows]


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


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - writes the artefact
    from argus.truth.artefact import write

    args = list(argv or [])
    if "--fetch" in args:
        args.remove("--fetch")
        freeze()
    symbols = tuple(args) if args else SYMBOLS
    series, digest = load_fixture()
    libraries: list[dict[str, Any]] = []
    returns_by_symbol: dict[str, list[float]] = {}
    failed: dict[str, str] = {}
    for symbol in symbols:
        if symbol not in series:
            failed[symbol] = f"not in {FIXTURE.name}"
            continue
        bars = bars_from_rows(series[symbol])
        if len(bars) < BLOCK * SPLIT_HALF_MIN_BLOCKS:
            failed[symbol] = f"only {len(bars)} bars"
            continue
        returns_by_symbol[symbol] = next_returns(bars)
        libraries.append(library_on(symbol, bars))
        print(f"  {symbol}: {len(bars)} bars", file=sys.stderr)
    if not libraries:
        print(f"no instrument in the fixture; nothing written: {failed}", file=sys.stderr)
        return 1
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "bars": {"fixture": f"data/{FIXTURE.name}", "sha256": digest,
                 "window_end": WINDOW_END.isoformat(), "days": WINDOW_DAYS},
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


__all__ = ["BLOCK", "FIXTURE", "SYMBOLS", "WINDOW_END", "FixtureError", "bars_from_rows",
           "freeze", "library_on", "load_fixture", "next_returns", "planted", "planted_checks",
           "summarise"]
