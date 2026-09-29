r"""G5 -- stress-window replay of run 2's registered policy (21_T2_QUANT_SEARCH.md row G5).

Answers: does any nine-day stress window break the policy's stated loss bounds (the 4% venue
stop, widened per name under run2-a7, and the weekend freeze that flattens every US-session leg
before Friday 20:00 UTC)?

Method. Run 2 is FROZEN (paper-traded on Bitget Demo): this script never imports it as a running
service, never restarts anything, and places no order. It only reads run 2's own code
(``scripts/strategy_replay.py`` and the ``sentiment_agent`` package it imports) as a library and
replays historical data through it -- the same production ``analysis.armsim.ArmSimulator`` path
G2 already used for the pre-genesis audit (``research/harvest/quant/T2_STRATEGY_AUDIT.md``,
"Real-kernel replay" section). No Qwen/LLM call, no network order, at any point.

Window selection is objective, not hand-picked: take the exact 216-hour (9-day) window set G2
replayed (every day 00:00 UTC with a full window of history behind it), compute four statistics
per window from the same Demo/live candle history the replay already loads
(``research/harvest/quant/replay/{history.json,history_marks.json}``), and pick the window that
is most extreme on each statistic:

* **crash**  -- most negative peak-to-trough drawdown of BTCUSDT within the window.
* **squeeze** -- largest trough-to-peak rally of BTCUSDT within the window.
* **flat**   -- lowest realised volatility (population stdev of hourly log returns) of BTCUSDT.
* **weekend gap** -- largest |Friday 20:00 UTC -> Monday 00:00 UTC| price move on any
  weekend-frozen instrument (``AssetClass.follows_us_session``: every US equity and index in the
  universe) that falls inside the window.

BTCUSDT is used for (a)-(c) because it is the cleanest, most liquid, 24/7-quoted instrument in
the universe (T2_STRATEGY_AUDIT.md D1) and the only one that is not itself subject to the
weekend freeze, so its price path is not an artefact of the freeze rule under test.

For each selected window the registered arm (``fade_7.5bp_registered``, run 2's actual trigger
and sizing as shipped) is replayed through the identical ``Policy``/``RiskKernel``/
``ArmSimulator`` construction ``strategy_replay.run_window`` uses, but calling
``ArmSimulator.run`` directly so the individual ``ClosedTrade`` records (with each trade's
``exit_reason``) survive for inspection -- ``run_window``'s own closure discards them after
extracting summary metrics. Every stop-caused exit's realised loss (net of fees, as a fraction of
its notional) is compared against ``Policy.stop_for(symbol)``, the exact per-symbol stop distance
(widened for the four names with wide Demo-live gaps, run2-a7) the venue was asked to honour.

Usage::

    python argus/scripts/t2_stress_windows.py [--out argus/data/t2_stress_windows.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import statistics
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]  # .../bitget
WORKSPACE_ROOT = REPO_ROOT.parent  # the parent that also holds the sibling checkouts below

# Sibling checkouts, not part of this repo. Overridable by environment variable so this script
# stays runnable from a different machine or layout without a hardcoded personal path baked in.
RUN2_ROOT = Path(os.environ.get("T2_RUN2_ROOT", WORKSPACE_ROOT / "t2-run2"))
REPLAY_DATA = REPO_ROOT / "research" / "harvest" / "quant" / "replay"
DEFAULT_OUT = REPO_ROOT / "argus" / "data" / "t2_stress_windows.json"

HOUR = timedelta(hours=1)
REGISTERED_ARM = "fade_7.5bp_registered"


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise SystemExit(
            f"{what} not found at {path}. This script reads run 2's frozen checkout read-only; "
            "set T2_RUN2_ROOT if it lives somewhere other than a sibling of this repo."
        )
    return path


def _load_strategy_replay() -> ModuleType:
    """Import run 2's ``scripts/strategy_replay.py`` as a library, read-only (module docstring)."""
    src = RUN2_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    path = _require(RUN2_ROOT / "scripts" / "strategy_replay.py", "run 2's strategy_replay.py")
    spec = importlib.util.spec_from_file_location("t2run2_strategy_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module.__name__] = (
        module  # dataclass() needs the module registered to resolve types
    )
    spec.loader.exec_module(module)
    return module


def _band(values: list[float]) -> dict[str, float | int]:
    vals = sorted(values)
    n = len(vals)
    return {
        "n": n,
        "p10": vals[max(0, round(0.1 * (n - 1)))],
        "median": vals[max(0, round(0.5 * (n - 1)))],
        "p90": vals[max(0, round(0.9 * (n - 1)))],
    }


# --------------------------------------------------------------------------------------------
# Per-window statistics: crash / squeeze / flat (BTCUSDT) and weekend gap (US-session names)
# --------------------------------------------------------------------------------------------


def _closes(hist: Any, symbol: str, start, end) -> list[tuple[Any, float]]:
    candles = [c for c in hist.marks.get(symbol, ()) if start <= c.open_time < end]
    candles.sort(key=lambda c: c.open_time)
    return [(c.open_time, float(c.close)) for c in candles]


def _crash_squeeze_flat(closes: list[tuple[Any, float]]) -> dict[str, Any] | None:
    prices = [p for _, p in closes]
    if len(prices) < 2:
        return None
    log_rets = [
        math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices)) if prices[i - 1] > 0
    ]
    peak = prices[0]
    trough = prices[0]
    max_dd = 0.0
    max_rally = 0.0
    for p in prices:
        peak = max(peak, p)
        max_dd = min(max_dd, p / peak - 1)
        trough = min(trough, p)
        max_rally = max(max_rally, p / trough - 1)
    return {
        "max_drawdown": max_dd,
        "max_rally": max_rally,
        "hourly_log_return_stdev": statistics.pstdev(log_rets) if len(log_rets) > 1 else None,
        "n_hours": len(prices),
    }


def _weekend_gaps(
    closes_by_symbol: dict[str, dict[Any, float]], symbols: list[str], start, end
) -> list[dict[str, Any]]:
    """Every |Friday 20:00 UTC -> Monday 00:00 UTC| move inside [start, end) (kernel/guards.py's
    ``weekend_phase``: policy v1 freeze is Friday 20:00 -> Monday 00:00 UTC, 52 hours)."""
    gaps: list[dict[str, Any]] = []
    t = start
    while t < end:
        if t.weekday() == 4 and t.hour == 20:  # Friday 20:00 UTC
            monday = t + timedelta(hours=52)
            for symbol in symbols:
                p_fri = closes_by_symbol.get(symbol, {}).get(t)
                p_mon = closes_by_symbol.get(symbol, {}).get(monday)
                if p_fri and p_mon and p_fri > 0:
                    gaps.append(
                        {
                            "symbol": symbol,
                            "friday_2000z": t.isoformat(),
                            "monday_0000z": monday.isoformat(),
                            "gap_bps": (p_mon / p_fri - 1) * 10_000,
                        }
                    )
        t += HOUR
    return gaps


def score_windows(sr: ModuleType, hist: Any, spans: list[tuple[Any, Any]]) -> list[dict[str, Any]]:
    weekend_symbols = sorted(
        e.symbol for e in sr.POLICY_V2.universe if e.asset_class.follows_us_session
    )
    closes_by_symbol = {
        s: {c.open_time: float(c.close) for c in cs} for s, cs in hist.marks.items()
    }
    rows = []
    for start, end in spans:
        btc = _crash_squeeze_flat(_closes(hist, "BTCUSDT", start, end))
        gaps = _weekend_gaps(closes_by_symbol, weekend_symbols, start, end)
        worst_gap = max(gaps, key=lambda g: abs(g["gap_bps"])) if gaps else None
        rows.append(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "btc": btc,
                "worst_weekend_gap": worst_gap,
                "n_weekend_gap_observations": len(gaps),
            }
        )
    return rows


def select_stress_windows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    with_btc = [r for r in rows if r["btc"] is not None]
    with_vol = [r for r in with_btc if r["btc"]["hourly_log_return_stdev"] is not None]
    with_gap = [r for r in rows if r["worst_weekend_gap"] is not None]
    crash = min(with_btc, key=lambda r: r["btc"]["max_drawdown"])
    squeeze = max(with_btc, key=lambda r: r["btc"]["max_rally"])
    flat = min(with_vol, key=lambda r: r["btc"]["hourly_log_return_stdev"])
    weekend = max(with_gap, key=lambda r: abs(r["worst_weekend_gap"]["gap_bps"]))
    return {"crash": crash, "squeeze": squeeze, "flat": flat, "weekend_gap": weekend}


# --------------------------------------------------------------------------------------------
# Running the registered arm through the production kernel/planner/book, keeping trade detail
# --------------------------------------------------------------------------------------------


def run_registered_arm(sr: ModuleType, hist: Any, start, end) -> Any:
    """The exact construction ``strategy_replay.run_window`` uses, calling ``ArmSimulator.run``
    directly (instead of through ``run_window``'s closure) so ``ArmResult.trades`` survives."""
    policy = sr.Policy.model_validate(
        sr.POLICY_V2.model_copy(
            update={
                "scoring_window": sr.ScoringWindow(
                    start=start, end=end, basis="G5 stress-window replay"
                )
            }
        ).model_dump()
    )
    sim = sr.ArmSimulator(
        kernel=sr.RiskKernel(policy, sr.ManualClock(start)),
        policy=policy,
        demo_marks=hist.marks,
        spreads_bps=hist.spreads_bps,
        starting_equity=sr.STARTING_EQUITY,
        specs=hist.specs,
    )
    rule = sr.ARMS[REGISTERED_ARM]
    rng = random.Random(0)
    schedule = [
        (t, w, sr.kernel_inputs(sim, hist, t))
        for t, w in rule(hist, policy, start, end, rng)
        if t < end
    ]
    result = sim.run(
        sr._spec(REGISTERED_ARM), schedule, start=start, until=end + HOUR, ci_resamples=0
    )
    return policy, result


def trade_detail(policy: Any, trade: Any) -> dict[str, Any]:
    notional = float(trade.max_abs_qty * trade.entry_avg)
    pnl_pct_of_notional = float(trade.net_pnl) / notional if notional > 0 else None
    stop_bound = policy.stop_for(trade.symbol)
    # Round-trip execution cost (fee + half-spread, both sides) that a stop fill pays on top of
    # the pure price distance -- the tolerance a "did the stop bound the loss" check must allow
    # for before calling a stop-filled trade a genuine gap through the stop.
    breached = None
    if trade.exit_reason == "stop_filled":
        breached = (
            abs(pnl_pct_of_notional) > stop_bound * 1.15
            if pnl_pct_of_notional is not None
            else None
        )
    return {
        "symbol": trade.symbol,
        "opened_at": trade.opened_at.isoformat(),
        "closed_at": trade.closed_at.isoformat(),
        "direction": trade.direction,
        "entry_avg": float(trade.entry_avg),
        "exit_avg": float(trade.exit_avg),
        "net_pnl": float(trade.net_pnl),
        "notional": notional,
        "pnl_pct_of_notional": pnl_pct_of_notional,
        "exit_reason": trade.exit_reason,
        "policy_stop_bound_pct": stop_bound,
        "stop_gapped_through": breached,
    }


def summarise_window(
    sr: ModuleType, hist: Any, label: str, start, end, rule_note: str
) -> dict[str, Any]:
    policy, result = run_registered_arm(sr, hist, start, end)
    m = result.metrics
    trades = [trade_detail(policy, t) for t in result.trades]
    exit_reason_counts: dict[str, int] = {}
    for t in trades:
        exit_reason_counts[t["exit_reason"]] = exit_reason_counts.get(t["exit_reason"], 0) + 1
    stop_trades = [t for t in trades if t["exit_reason"] == "stop_filled"]
    weekend_trades = [
        t for t in trades if t["exit_reason"].startswith("protective_exit:weekend_freeze")
    ]
    window_end_trades = [
        t for t in trades if t["exit_reason"].startswith("protective_exit:window_end")
    ]
    any_stop_breach = any(t["stop_gapped_through"] for t in stop_trades)
    return {
        "label": label,
        "selection_rule": rule_note,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "metrics": {
            "closed_trades": m.n_closed_trades,
            "total_return": m.total_return,
            "sharpe_ann": m.sharpe_ann,
            "max_drawdown": m.max_drawdown,
            "win_rate": m.win_rate,
            "fees_paid": m.fees_paid,
            "turnover": m.turnover,
        },
        "exit_reason_counts": exit_reason_counts,
        "stop_filled_trades": stop_trades,
        "weekend_freeze_trades": weekend_trades,
        "window_end_trades": window_end_trades,
        "any_stop_gapped_through": any_stop_breach,
        "worst_single_trade_pnl_pct_of_notional": (
            min(
                (t["pnl_pct_of_notional"] for t in trades if t["pnl_pct_of_notional"] is not None),
                default=None,
            )
        ),
        "trades": trades,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    sr = _load_strategy_replay()
    history_path = _require(REPLAY_DATA / "history.json", "replay history.json")
    marks_path = _require(REPLAY_DATA / "history_marks.json", "replay history_marks.json")
    probe_path = _require(
        RUN2_ROOT / "validation" / "demo_venue" / "universe_probe.json",
        "run 2's universe_probe.json",
    )
    hist = sr.load(history_path, marks_path, probe_path)

    funding_from = max(
        series[sr.FUNDING_LOOKBACK][0]
        for series in hist.funding.values()
        if len(series) > sr.FUNDING_LOOKBACK
    )
    spans = sr.windows(hist, 216, funding_from)  # the exact 51 windows G2 replayed
    scored = score_windows(sr, hist, spans)
    picked = select_stress_windows(scored)

    rules = {
        "crash": "most negative peak-to-trough drawdown of BTCUSDT hourly closes within the window",
        "squeeze": "largest trough-to-peak rally of BTCUSDT hourly closes within the window",
        "flat": "lowest population stdev of BTCUSDT hourly log returns within the window",
        "weekend_gap": (
            "largest |Friday 20:00 UTC -> Monday 00:00 UTC| move on any weekend-frozen "
            "instrument (every US equity/index in the universe) falling inside the window"
        ),
    }

    windows_out = {}
    for key, row in picked.items():
        start = next(s for s, _ in spans if s.isoformat() == row["start"])
        end = next(e for _, e in spans if e.isoformat() == row["end"])
        windows_out[key] = summarise_window(sr, hist, key, start, end, rules[key])
        windows_out[key]["window_stats"] = row

    any_break = any(w["any_stop_gapped_through"] for w in windows_out.values())
    max_dd_overall = min(w["metrics"]["max_drawdown"] for w in windows_out.values())

    report = {
        "method": __doc__,
        "n_candidate_windows": len(spans),
        "window_hours": 216,
        "candidate_windows_scored": scored,
        "selection_rules": rules,
        "windows": windows_out,
        "verdict": {
            "any_stress_window_breaks_stop_bound": any_break,
            "worst_book_level_max_drawdown_across_stress_windows": max_dd_overall,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    for key, w in windows_out.items():
        print(
            f"{key:12s} {w['start']} -> {w['end']}  trades={w['metrics']['closed_trades']:2d} "
            f"return={w['metrics']['total_return']:+.5f} "
            f"max_dd={w['metrics']['max_drawdown']:+.5f} "
            f"win_rate={w['metrics']['win_rate']}  stop_gap_through={w['any_stop_gapped_through']}"
        )
    print(f"any_stress_window_breaks_stop_bound = {any_break}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
