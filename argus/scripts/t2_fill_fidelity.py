r"""G6 -- sim-vs-venue fidelity (21_T2_QUANT_SEARCH.md row G6).

For every real Bitget Demo fill recorded by run 1 and run 2, replay the same order (same symbol,
side, quantity, reference price) through the production ``analysis.armsim.ArmSimulator`` fill
model -- taker only, filled at the reference Demo mark moved by half the measured Demo spread,
plus the Demo taker fee (``armsim.py`` module docstring, "The cost model") -- and compare the
simulated fill price and fee against what the venue actually returned.

Both run 1 (``t2-sentiment-agent``) and run 2 (``t2-run2``) are read-only here: their own
published ``public/orders.json`` is the source of every real fill (``order.reference_price`` is
the exact Demo mark the kernel's planner sized the order against -- ``planning_price()`` in
``kernel/planner.py`` -- which is the same ``reference`` argument ``ArmSimulator._taker_price``
takes, so the two are directly comparable at the same instant). No network call, no order, no
LLM call; run 2 is never imported as a running service, only its ``sentiment_agent`` package is
read as a library for the fill-cost formula it already uses in the replay G2 ran.

Usage::

    python argus/scripts/t2_fill_fidelity.py [--out argus/data/t2_fill_fidelity.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]  # .../bitget
WORKSPACE_ROOT = REPO_ROOT.parent

# Sibling checkouts, not part of this repo; overridable so this script has no hardcoded personal
# path baked in.
RUN2_ROOT = Path(os.environ.get("T2_RUN2_ROOT", WORKSPACE_ROOT / "t2-run2"))
RUN1_ROOT = Path(os.environ.get("T2_RUN1_ROOT", WORKSPACE_ROOT / "t2-sentiment-agent"))
REPLAY_DATA = REPO_ROOT / "research" / "harvest" / "quant" / "replay"
DEFAULT_OUT = REPO_ROOT / "argus" / "data" / "t2_fill_fidelity.json"

# Every place a real Demo fill ledger was searched for, so a run that finds none still says
# exactly where it looked (task requirement: "say exactly which files you searched").
CANDIDATE_LEDGERS = [
    ("run1", "public/orders.json"),
    ("run1", "var/site/t2-sentiment-agent-live/orders.json"),
    ("run1", "var/public-dryrun/orders.json"),
    ("run1", "var/public-simulated/orders.json"),
    ("run2", "public/orders.json"),
    ("run2", "var/site/t2-sentiment-agent-run2/orders.json"),
    ("run2", "var/public-dryrun/orders.json"),
    ("run2", "var/public-simulated/orders.json"),
]
ROOTS = {"run1": RUN1_ROOT, "run2": RUN2_ROOT}


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise SystemExit(
            f"{what} not found at {path}. Set T2_RUN2_ROOT/T2_RUN1_ROOT if the checkouts live "
            "somewhere other than siblings of this repo."
        )
    return path


def _load_strategy_replay() -> ModuleType:
    src = RUN2_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    path = _require(RUN2_ROOT / "scripts" / "strategy_replay.py", "run 2's strategy_replay.py")
    spec = importlib.util.spec_from_file_location("t2run2_strategy_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module.__name__] = module
    spec.loader.exec_module(module)
    return module


def _pctile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def find_real_fills() -> tuple[list[dict[str, Any]], list[str]]:
    """Every real (state == "filled") order + fill pair in the canonical run 1 / run 2 published
    ledgers. Returns (records, paths_searched) -- every path searched is reported even when it
    was missing or held no fills, per the task's "say exactly which files you searched" rule."""
    seen_fills: set[str] = set()  # de-dupe: the same exec_id can appear in more than one mirror
    records: list[dict[str, Any]] = []
    searched: list[str] = []
    for label, rel in CANDIDATE_LEDGERS:
        path = ROOTS[label] / rel
        # recorded relative to the run's own root: no machine path reaches the artefact
        where = f"{label}:{Path(rel).as_posix()}"
        searched.append(where)
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for order in data.get("orders", []):
            if order.get("state") != "filled":
                continue
            for fill in order.get("fills") or []:
                exec_id = fill.get("exec_id")
                if exec_id in seen_fills:
                    continue
                seen_fills.add(exec_id)
                records.append({"run": label, "source_file": where, "order": order, "fill": fill})
    return records, searched


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
    from sentiment_agent.types import Side

    policy = sr.POLICY_V2
    # Only the fill-cost model is used (half_spread/fee_rate, from the same spreads_bps/specs the
    # replay loads); no schedule is run, so any start time on the clock is fine.
    any_symbol = next(iter(hist.marks))
    clock_start = hist.marks[any_symbol][0].open_time
    sim = sr.ArmSimulator(
        kernel=sr.RiskKernel(policy, sr.ManualClock(clock_start)),
        policy=policy,
        demo_marks=hist.marks,
        spreads_bps=hist.spreads_bps,
        starting_equity=sr.STARTING_EQUITY,
        specs=hist.specs,
    )

    real_fills, searched = find_real_fills()
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in real_fills:
        order, fill = item["order"], item["fill"]
        symbol = order["symbol"]
        exec_id = str(fill.get("exec_id") or "")
        if not exec_id.isdigit():
            # var/public-simulated/orders.json (both runs) records the ungoverned-twin / synthetic
            # arm, not a real Demo fill: its exec_id is a literal "sim-fill-00000001", not a Bitget
            # venue order id (real ones are large all-digit numbers, e.g. "1488055070676463616").
            # Checked and excluded on that basis, not by filename, so the check is verifiable.
            skipped.append(
                {
                    "run": item["run"],
                    "source_file": item["source_file"],
                    "symbol": symbol,
                    "exec_id": exec_id,
                    "reason": (
                        "exec_id is not a real Bitget venue order id "
                        "(synthetic/simulated fill, not a real Demo fill)"
                    ),
                }
            )
            continue
        if symbol not in sim.specs or symbol not in sim.spreads_bps:
            skipped.append(
                {
                    "run": item["run"],
                    "source_file": item["source_file"],
                    "symbol": symbol,
                    "exec_id": exec_id,
                    "reason": "no Demo spec/spread for this symbol in the replay data",
                }
            )
            continue
        side = Side.BUY if order["side"] == "buy" else Side.SELL
        side_sign = 1 if side is Side.BUY else -1
        reference_price = Decimal(str(order["reference_price"]))
        qty = Decimal(str(fill["exec_qty"]))
        real_price = Decimal(str(fill["exec_price"]))
        real_fee = Decimal(str(fill["fee_paid"]))

        # ArmSimulator._taker_price (armsim.py's _ArmRun), reproduced from its two public building
        # blocks (half_spread, fee_rate) since that formula itself is a private method of the
        # per-run helper class, not of ArmSimulator: reference * (1 +/- half spread), taker fee on
        # the filled notional -- exactly "the cost model" armsim.py's module docstring states.
        half = sim.half_spread(symbol)
        one = Decimal(1)
        sim_price = (
            reference_price * (one + half) if side is Side.BUY else reference_price * (one - half)
        )
        sim_fee = qty * sim_price * sim.fee_rate(symbol)

        real_cost_bps = float((real_price / reference_price - 1) * side_sign) * 10_000
        sim_cost_bps = float((sim_price / reference_price - 1) * side_sign) * 10_000
        notional_ref = float(reference_price * qty)
        real_fee_bps = float(real_fee) / notional_ref * 10_000 if notional_ref else None
        sim_fee_bps = float(sim_fee) / notional_ref * 10_000 if notional_ref else None

        records.append(
            {
                "run": item["run"],
                "source_file": item["source_file"],
                "symbol": symbol,
                "side": order["side"],
                "purpose": order.get("purpose"),
                "executed_at": fill.get("executed_at"),
                "venue_order_id": order.get("venue_order_id"),
                "exec_id": fill.get("exec_id"),
                "qty": float(qty),
                "reference_price": float(reference_price),
                "real_fill_price": float(real_price),
                "sim_fill_price": float(sim_price),
                "real_fee": float(real_fee),
                "sim_fee": float(sim_fee),
                "real_execution_cost_bps_vs_reference": real_cost_bps,
                "sim_execution_cost_bps_vs_reference": sim_cost_bps,
                "price_diff_bps": real_cost_bps - sim_cost_bps,
                "real_fee_bps_of_notional": real_fee_bps,
                "sim_fee_bps_of_notional": sim_fee_bps,
                "fee_diff_bps_of_notional": (
                    (real_fee_bps - sim_fee_bps)
                    if real_fee_bps is not None and sim_fee_bps is not None
                    else None
                ),
            }
        )

    price_diffs = [r["price_diff_bps"] for r in records]
    fee_diffs = [
        r["fee_diff_bps_of_notional"] for r in records if r["fee_diff_bps_of_notional"] is not None
    ]
    mean_price_diff = statistics.fmean(price_diffs) if price_diffs else None
    mean_fee_diff = statistics.fmean(fee_diffs) if fee_diffs else None

    summary = {
        "n": len(records),
        "n_real_fills_found": len(real_fills),
        "n_skipped": len(skipped),
        "price_diff_bps": {
            "mean": mean_price_diff,
            "p95_abs": _pctile([abs(v) for v in price_diffs], 0.95),
            "min": min(price_diffs) if price_diffs else None,
            "max": max(price_diffs) if price_diffs else None,
        },
        "fee_diff_bps_of_notional": {
            "mean": mean_fee_diff,
            "p95_abs": _pctile([abs(v) for v in fee_diffs], 0.95) if fee_diffs else None,
        },
        # price_diff_bps = (real execution cost vs reference) - (sim's assumed execution cost vs
        # reference), in bps of the reference price, signed positive when the real fill cost MORE
        # than the sim assumed. A positive mean means the sim under-states real trading cost
        # (the sim is optimistic); negative means the sim over-states it (pessimistic).
        "sim_is_optimistic_on_price": (mean_price_diff is not None and mean_price_diff > 0),
        "sim_is_pessimistic_on_price": (mean_price_diff is not None and mean_price_diff < 0),
    }

    report = {
        "method": __doc__,
        "ledgers_searched": searched,
        "skipped": skipped,
        "records": records,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"n fills compared: {len(records)}")
    for r in records:
        print(
            f"  {r['run']:5s} {r['symbol']:10s} {r['side']:4s} ref={r['reference_price']:<12g} "
            f"real={r['real_fill_price']:<12g} sim={r['sim_fill_price']:<12g} "
            f"price_diff_bps={r['price_diff_bps']:+.3f}"
        )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
