"""The desk's lean against NIGHTWATCH AI's signal, on the same instants and the same moves (5.4).

NIGHTWATCH AI (``drained69/nightwatchai``, MIT, a Season 2 AI Trading Desk entry; cloned at
``research/repos-s2new/drained69~nightwatchai``) publishes a directional signal per symbol and an
accuracy ledger graded eight hours later (``server/signal-history.mjs``: a hit when the move is the
called way, FLAT when it is under 20 bps). ARGUS's desk states a lean on every decision it abstains
from, graded at its marks (`paper/marks.py`, `eval/refusal.py`). Here both are asked about the same
symbol at the same instants — every mark on a name both cover — and graded on the same realised
moves: the desk's own ~2h mark, and the eight-hour move on the R-pair NIGHTWATCH reads.

**How NIGHTWATCH is run.** Its own functions, from its clone, unmodified
(`baselines/nightwatch_signal.mjs`): ``computeIndicators`` on the Bitget R-pair hourly candles
closed before the instant, ``mergeMarketRow``, the five skill readers and ``synthesizeSignal``
through ``LocalNightwatchEngine.research``, with a fresh user's default preferences and a macro
snapshot rebuilt from Yahoo daily closes stamped at 21:30 UTC, after the US close, so no instant
sees a close that had not printed. **Not reproduced**: its news store and its live overlay (spot
book depth and a market-intel snapshot), which exist only at the moment of a live call — so this
is NIGHTWATCH's engine without those inputs, and its published live record (15 of 38 non-flat
calls right at 8h on 2026-10-05, ``GET /signals/stats``) is reported beside it.

**How it is scored.** Each side's hit rate on the calls it made, against calling ``up`` every time
on the same instants (the base rate a lean must beat in a rising tape), with a 95% interval from a
bootstrap over decision cycles: the names decided in one cycle share one market move.

    python -m argus.eval.nightwatch_comparison --collect   # saves the inputs, then scores
    python -m argus.eval.nightwatch_comparison             # scores from the saved inputs
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from argus.truth import artefact, http
from argus.truth.paths import DATA_DIR

RIVAL: Final = Path(__file__).resolve().parents[4] / "research" / "repos-s2new" / \
    "drained69~nightwatchai"
RUNNER: Final = Path(__file__).with_name("baselines") / "nightwatch_signal.mjs"
INPUTS: Final = DATA_DIR / "h2h_nightwatch" / "inputs.json"
OUT: Final = DATA_DIR / "nightwatch_comparison.json"
MARKS: Final = DATA_DIR / "refusal_marks.jsonl"
SPOT: Final = "https://api.bitget.com/api/v2/spot/market/candles"
MACRO: Final = {"dxy": "DX-Y.NYB", "spx": "^GSPC", "ndx": "^NDX", "vix": "^VIX", "ust10y": "^TNX"}
HORIZON_8H: Final = timedelta(hours=8)
FLAT_BPS: Final = 20.0
"""NIGHTWATCH's own FLAT band (``server/signal-history.mjs:63-68``)."""


def _marks() -> list[dict[str, Any]]:
    return [json.loads(line) for line in MARKS.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _spot_candles(pair: str, start: datetime, end: datetime) -> list[dict[str, float]]:
    """Hourly spot candles for ``pair``, paged back from ``end`` 200 bars at a time."""
    rows: dict[int, dict[str, float]] = {}
    cursor = int(end.timestamp() * 1000)
    while cursor > start.timestamp() * 1000:
        body = http.fetch_json(SPOT, params={"symbol": pair, "granularity": "1h", "limit": 200,
                                             "endTime": cursor}, timeout=20)
        data = body.get("data") or []
        if not data:
            break
        for r in data:
            rows[int(r[0])] = {"ts": int(r[0]), "open": float(r[1]), "high": float(r[2]),
                               "low": float(r[3]), "close": float(r[4]), "volume": float(r[6])}
        oldest = min(int(r[0]) for r in data)
        if oldest >= cursor:
            break
        cursor = oldest - 1
    return sorted(rows.values(), key=lambda c: c["ts"])


def _yahoo_daily(ticker: str) -> list[list[float]]:
    """Daily closes, each stamped 21:30 UTC on its day — after the US close it belongs to."""
    body = http.fetch_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                           params={"interval": "1d", "range": "6mo"},
                           headers={"User-Agent": "Mozilla/5.0 argus-research"}, timeout=30)
    result = body["chart"]["result"][0]
    out = []
    for stamp, close in zip(result["timestamp"], result["indicators"]["quote"][0]["close"],
                            strict=False):
        if close is None:
            continue
        day = datetime.fromtimestamp(stamp, UTC).date()
        at = datetime(day.year, day.month, day.day, 21, 30, tzinfo=UTC)
        out.append([int(at.timestamp() * 1000), float(close)])
    return out


def collect() -> dict[str, Any]:  # pragma: no cover - live network
    marks = _marks()
    first = min(datetime.fromisoformat(m["decided_at"]) for m in marks) - timedelta(days=10)
    last = max(datetime.fromisoformat(m["marked_at"]) for m in marks) + timedelta(hours=12)
    names = sorted({m["symbol"].removesuffix("USDT") for m in marks})
    candles: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for name in [*names, "BTC"]:
        pair = "BTCUSDT" if name == "BTC" else f"R{name}USDT"
        try:
            got = _spot_candles(pair, first, last)
        except http.RpcError as exc:
            failures[pair] = http.reason_of(exc)
            continue
        if got:
            candles[name] = got
        else:
            failures[pair] = "no candles"
    macro = {}
    for key, ticker in MACRO.items():
        try:
            macro[key] = _yahoo_daily(ticker)
        except (http.RpcError, KeyError, IndexError) as exc:
            failures[ticker] = type(exc).__name__
    blob = {"fetched_at": datetime.now(UTC).isoformat(timespec="seconds"), "candles": candles,
            "macro_daily": macro, "failures": failures}
    INPUTS.parent.mkdir(parents=True, exist_ok=True)
    INPUTS.write_text(json.dumps(blob) + "\n", encoding="utf-8", newline="\n")
    return blob


def _close_at_or_after(candles: list[dict[str, Any]], at_ms: int) -> float | None:
    """The close of the first bar that ends at or after ``at_ms``."""
    for c in candles:
        if c["ts"] + 3_600_000 >= at_ms:
            return float(c["close"])
    return None


def _last_closed(candles: list[dict[str, Any]], at_ms: int) -> float | None:
    done = [c for c in candles if c["ts"] + 3_600_000 <= at_ms]
    return float(done[-1]["close"]) if done else None


def signals(instants: list[dict[str, Any]], inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """NIGHTWATCH's signal at each instant, from its own code run by Node."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.json", Path(tmp) / "out.json"
        src.write_text(json.dumps({"instants": instants, "candles": inputs["candles"],
                                   "macro_daily": inputs["macro_daily"]}), encoding="utf-8",
                       newline="\n")
        env = {**os.environ, "NIGHTWATCH_DIR": str(RIVAL)}
        subprocess.run(["node", str(RUNNER), str(src), str(dst)], check=True, env=env,
                       capture_output=True, text=True, timeout=1800)
        return {r["key"]: r for r in json.loads(dst.read_text(encoding="utf-8"))}


def _cycle_interval(calls: list[tuple[str, bool]], draws: int = 2000,
                    seed: int = 7) -> tuple[float, float] | None:
    by_cycle: dict[str, list[bool]] = defaultdict(list)
    for cycle, hit in calls:
        by_cycle[cycle].append(hit)
    groups = list(by_cycle.values())
    if len(groups) < 10:
        return None
    rng = random.Random(seed)
    rates = []
    for _ in range(draws):
        pick = [g for _ in groups for g in [groups[rng.randrange(len(groups))]]]
        n = sum(len(g) for g in pick)
        rates.append(sum(sum(g) for g in pick) / n)
    rates.sort()
    return round(100 * rates[int(0.025 * draws)], 1), round(100 * rates[int(0.975 * draws)], 1)


def _score(rows: list[dict[str, Any]], call: str, move: str) -> dict[str, Any]:
    """Hit rate of ``call`` on ``move`` over the rows where both exist, beside always-up."""
    usable = [r for r in rows if r.get(call) in ("up", "down") and r.get(move) is not None
              and abs(r[move]) >= FLAT_BPS]
    if not usable:
        return {"calls": 0}
    hits = [(r["cycle"], (r[call] == "up") == (r[move] > 0)) for r in usable]
    up = [(r["cycle"], r[move] > 0) for r in usable]
    interval = _cycle_interval(hits)
    return {"calls": len(usable), "cycles": len({r["cycle"] for r in usable}),
            "hit_rate_pct": round(100 * sum(h for _, h in hits) / len(hits), 1),
            "hit_rate_ci95": list(interval) if interval else None,
            "always_up_pct": round(100 * sum(h for _, h in up) / len(up), 1),
            "share_called_up_pct": round(100 * sum(r[call] == "up" for r in usable)
                                         / len(usable), 1)}


def _paired(rows: list[dict[str, Any]], move: str) -> dict[str, Any]:
    """Instants one side called right and the other wrong, and the exact two-sided sign test on
    them: the instants where both agree say nothing about which is better."""
    from math import comb

    usable = [r for r in rows if r.get(move) is not None and abs(r[move]) >= FLAT_BPS]
    ours = sum(1 for r in usable if (r["argus"] == "up") == (r[move] > 0)
               and (r["nightwatch"] == "up") != (r[move] > 0))
    theirs = sum(1 for r in usable if (r["nightwatch"] == "up") == (r[move] > 0)
                 and (r["argus"] == "up") != (r[move] > 0))
    n = ours + theirs
    tail = sum(comb(n, k) for k in range(min(ours, theirs) + 1)) / 2 ** n if n else 1.0
    return {"argus_right_nightwatch_wrong": ours, "nightwatch_right_argus_wrong": theirs,
            "sign_test_p_two_sided": round(min(1.0, 2 * tail), 4)}


def score(inputs: dict[str, Any]) -> dict[str, Any]:
    marks = [m for m in _marks() if m["lean"] in ("up", "down")]
    covered = set(inputs["candles"]) - {"BTC"}
    instants, rows = [], []
    for m in marks:
        name = m["symbol"].removesuffix("USDT")
        if name not in covered:
            continue
        at = datetime.fromisoformat(m["decided_at"])
        key = f"{m['seq']}"
        instants.append({"key": key, "symbol": name, "at_ms": int(at.timestamp() * 1000)})
        candles = inputs["candles"][name]
        entry = _last_closed(candles, int(at.timestamp() * 1000))
        exit_8h = _close_at_or_after(candles, int((at + HORIZON_8H).timestamp() * 1000))
        rows.append({"key": key, "symbol": name, "cycle": m["decided_at"][:16],
                     "argus": m["lean"], "horizon_hours": m["horizon_hours"],
                     "move_mark_bps": float(m["move_bps"]),
                     "move_8h_bps": (10_000 * (exit_8h / entry - 1)
                                     if entry and exit_8h else None)})
    got = signals(instants, inputs)
    skipped = {k: v["skipped"] for k, v in got.items() if "skipped" in v}
    # only the instants NIGHTWATCH could answer: a name outside its universe is not a miss
    rows = [r for r in rows if r["key"] not in skipped]
    for row in rows:
        signal = got.get(row["key"]) or {}
        direction = signal.get("direction")
        row["nightwatch_status"] = signal.get("status")
        row["nightwatch_direction"] = direction
        call = {"LONG": "up", "SHORT": "down"}.get(str(direction))
        row["nightwatch"] = call if signal.get("status") == "SIGNAL" else None
        row["nightwatch_any_direction"] = call
    two_hour = [r for r in rows if 1.5 <= r["horizon_hours"] < 4]
    both = [r for r in rows if r["nightwatch"] is not None]
    return {
        "instants": len(rows), "symbols": sorted({r["symbol"] for r in rows}),
        "skipped": dict(sorted({v: sum(1 for x in skipped.values() if x == v)
                                for v in skipped.values()}.items())),
        "nightwatch_signals": sum(r["nightwatch"] is not None for r in rows),
        "nightwatch_no_trade": sum(r["nightwatch_status"] == "NO_TRADE" for r in rows),
        "at_desk_mark_about_2h": {
            "argus": _score(two_hour, "argus", "move_mark_bps"),
            "nightwatch_signals": _score(two_hour, "nightwatch", "move_mark_bps"),
            "nightwatch_any_direction": _score(two_hour, "nightwatch_any_direction",
                                               "move_mark_bps")},
        "at_8h_nightwatch_horizon": {
            "argus": _score(rows, "argus", "move_8h_bps"),
            "nightwatch_signals": _score(rows, "nightwatch", "move_8h_bps"),
            "nightwatch_any_direction": _score(rows, "nightwatch_any_direction", "move_8h_bps")},
        "same_instants_both_called_8h": {
            "argus": _score(both, "argus", "move_8h_bps"),
            "nightwatch": _score(both, "nightwatch", "move_8h_bps"),
            "paired": _paired(both, "move_8h_bps")},
        "flat_band_bps": FLAT_BPS,
        "rows": rows,
    }


def run(*, out: Path = OUT, fresh: bool = False) -> dict[str, Any]:
    inputs = collect() if fresh else json.loads(INPUTS.read_text(encoding="utf-8"))
    blob = {"rival": "drained69/nightwatchai LocalNightwatchEngine.research, run unmodified",
            "inputs": "data/h2h_nightwatch/inputs.json",
            "inputs_fetched": inputs.get("fetched_at"),
            "not_reproduced": ["its news store", "its live overlay (spot book depth, market "
                               "intel)"],
            **score(inputs)}
    artefact.write(out, blob)
    return blob


def main() -> int:  # pragma: no cover - CLI
    blob = run(fresh="--collect" in sys.argv)
    print(json.dumps({k: v for k, v in blob.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
