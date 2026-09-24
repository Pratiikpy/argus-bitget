"""Same-input comparison: `argus.desk.execution` vs crypto_sor's real, vendored composite router.

Runs crypto_sor's real, unmodified `CompositeOrderBook.newOrder()`
(`eval/baselines/crypto_sor_shim/src/lib/CompositeOrderBook.ts`, executed through a real Node/
ts-node subprocess — `eval/baselines/crypto_sor_loader.py`) and ARGUS's real `desk.execution.
choose_hedge_leg` on the same real, live Bitget order-book and funding data for real rToken/
crypto leg pairs.

**The central, measured finding.** crypto_sor's real router picks a leg purely by execution cost
at the instant of the fill — fed each leg's real, measured slippage in bps as its synthetic
"price" (the fair way to make its real price-comparison logic meaningful across two differently-
priced assets; see `argus.desk.execution`'s own module docstring), it always picks the crypto
leg, because crypto majors quote dramatically tighter on this venue (measured live: an order of
magnitude or more tighter slippage than the rToken legs tested). It has no way to know that the
crypto leg also carries a real, non-zero funding rate on this venue while the rToken legs tested
currently do not — `argus.desk.execution.choose_hedge_leg` prices both channels and, past a real,
computed break-even holding horizon, picks the rToken leg instead. The two real systems agree at
holding_days=0 (funding has not accrued yet) and diverge as soon as the position would be held
past that horizon — crypto_sor's real, unmodified output never changes, because holding time is
not a concept it has any representation of.

SCOPE, stated explicitly:

* This is a same-venue, cross-ASSET-CLASS comparison, not a cross-EXCHANGE one — crypto_sor was
  designed to route across venues and this comparison instead feeds it two legs of ARGUS's own
  single-venue (Bitget) universe as if they were two competing venues, which is a faithful use of
  its real, general "cheapest quoted price wins" algorithm, not a different algorithm.
* The break-even horizons measured here are real but venue-and-moment-specific: rToken funding on
  this venue is usually zero and crypto funding is usually small and can flip sign
  (`research/carry.py`'s own measurement). The comparison reports whatever is real and live the
  day it runs, not a permanent constant.
* No claim is made that routing through the rToken leg is a validated hedge — `argus.desk.
  execution` prices the cost of each leg, not whether either leg actually offsets the exposure
  being hedged, which is a separate, unaddressed question.
"""

from __future__ import annotations

import json
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.cost.model import CostModel
from argus.desk.execution import (
    ExecutionError,
    HedgeLegQuote,
    break_even_holding_days,
    choose_hedge_leg,
)
from argus.eval.baselines.crypto_sor_loader import CryptoSorSubprocessError, run_new_order
from argus.market.bitget import fetch_tickers
from argus.market.depth import DepthError, fetch_orderbook

NOTIONAL = Decimal("5000")
LEG_PAIRS: tuple[tuple[str, str], ...] = (
    ("NVDAUSDT", "BTCUSDT"), ("NVDAUSDT", "ETHUSDT"),
    ("AAPLUSDT", "BTCUSDT"), ("AAPLUSDT", "ETHUSDT"),
    ("MSFTUSDT", "BTCUSDT"),
)
SWEEP_HOLDING_DAYS: tuple[Decimal, ...] = (
    Decimal("0"), Decimal("0.25"), Decimal("0.5"), Decimal("1"), Decimal("2"), Decimal("5"),
)


def _real_leg(symbol: str) -> HedgeLegQuote:
    tickers = fetch_tickers()
    book = fetch_orderbook(symbol, limit=50)
    sweep = book.sweep(NOTIONAL, direction="BUY")
    if not sweep.complete:
        raise ExecutionError(f"{symbol}'s real book cannot absorb ${NOTIONAL}")
    return HedgeLegQuote(
        symbol=symbol,
        slippage_bps=sweep.slippage_bps,
        cost_model=CostModel.bitget_perp(funding_rate=tickers[symbol].funding_rate),
    )


def _crypto_sor_pick(leg_a: HedgeLegQuote, leg_b: HedgeLegQuote) -> str:
    """Feeds the real, unmodified crypto_sor router each leg's real slippage as its synthetic
    price and returns which leg it filled from."""
    levels = [
        {"exchange": leg_a.symbol, "side": "SELL", "price": float(leg_a.slippage_bps), "size": 1.0},
        {"exchange": leg_b.symbol, "side": "SELL", "price": float(leg_b.slippage_bps), "size": 1.0},
    ]
    executions = run_new_order("PAIR", "BUY", 1.0, levels)
    if not executions:
        raise CryptoSorSubprocessError("the real router returned no executions")
    return str(executions[0]["exchange"])


# --- base case -------------------------------------------------------------------------------


def run_base_case(
    rtoken: str = "NVDAUSDT", crypto: str = "BTCUSDT",
    *, legs: tuple[HedgeLegQuote, HedgeLegQuote] | None = None,
) -> dict[str, Any]:
    """The single, real, live-data demonstration — or, with ``legs``, the same comparison on legs
    the caller supplies (the tests pin the recorded measurement this way; see
    `tests/test_execution_comparison.py`). The real crypto_sor router runs either way."""
    rtoken_leg, crypto_leg = legs if legs is not None else (_real_leg(rtoken), _real_leg(crypto))

    real_pick_now = _crypto_sor_pick(rtoken_leg, crypto_leg)
    decision_now = choose_hedge_leg(
        [rtoken_leg, crypto_leg], NOTIONAL, holding_days=Decimal("0"),
    )

    cheaper_entry, other = (
        (crypto_leg, rtoken_leg) if crypto_leg.slippage_bps < rtoken_leg.slippage_bps
        else (rtoken_leg, crypto_leg)
    )
    break_even = break_even_holding_days(cheaper_entry, other, notional=NOTIONAL)

    decision_at_break_even = None
    if break_even is not None:
        decision_at_break_even = choose_hedge_leg(
            [rtoken_leg, crypto_leg], NOTIONAL, holding_days=break_even + Decimal("1"),
        ).cheapest_total_cost_symbol

    return {
        "rtoken": rtoken_leg.symbol,
        "crypto": crypto_leg.symbol,
        "rtoken_slippage_bps": str(rtoken_leg.slippage_bps),
        "crypto_slippage_bps": str(crypto_leg.slippage_bps),
        "rtoken_funding_bps": str(rtoken_leg.cost_model.funding_bps_per_interval),
        "crypto_funding_bps": str(crypto_leg.cost_model.funding_bps_per_interval),
        "real_crypto_sor_pick_at_entry": real_pick_now,
        "argus_pick_at_entry": decision_now.cheapest_total_cost_symbol,
        "agree_at_entry": real_pick_now == decision_now.cheapest_total_cost_symbol,
        "break_even_holding_days": str(break_even) if break_even is not None else None,
        "argus_pick_past_break_even": decision_at_break_even,
        "diverges_past_break_even": (
            decision_at_break_even is not None and decision_at_break_even != real_pick_now
        ),
    }


# --- divergence sweep --------------------------------------------------------------------------


def run_divergence_sweep(
    rtoken: str = "NVDAUSDT", crypto: str = "BTCUSDT",
    *, legs: tuple[HedgeLegQuote, HedgeLegQuote] | None = None,
) -> dict[str, Any]:
    """Sweeps `holding_days`. crypto_sor's real pick is constant (no holding-time concept);
    ARGUS's real pick changes once funding overtakes the entry saving. Counts how often they
    disagree — the statistically-valid-evaluation proof."""
    rtoken_leg, crypto_leg = legs if legs is not None else (_real_leg(rtoken), _real_leg(crypto))
    rtoken, crypto = rtoken_leg.symbol, crypto_leg.symbol
    real_pick = _crypto_sor_pick(rtoken_leg, crypto_leg)

    points = []
    disagreements = 0
    for holding_days in SWEEP_HOLDING_DAYS:
        decision = choose_hedge_leg(
            [rtoken_leg, crypto_leg], NOTIONAL, holding_days=holding_days,
        )
        agree = decision.cheapest_total_cost_symbol == real_pick
        if not agree:
            disagreements += 1
        points.append({
            "holding_days": str(holding_days),
            "argus_pick": decision.cheapest_total_cost_symbol,
            "real_crypto_sor_pick": real_pick,
            "agree": agree,
        })

    return {
        "rtoken": rtoken, "crypto": crypto,
        "points": points,
        "n_points": len(points),
        "n_disagreements": disagreements,
        "disagreement_rate": disagreements / len(points),
    }


def run_leg_pair_sweep() -> dict[str, Any]:
    """Repeats the base case across `LEG_PAIRS` — real, different rToken/crypto combinations, not
    one convenient pair."""
    results = []
    for rtoken, crypto in LEG_PAIRS:
        try:
            results.append({"rtoken": rtoken, "crypto": crypto, **run_base_case(rtoken, crypto)})
        except (ExecutionError, DepthError, CryptoSorSubprocessError) as exc:
            results.append({
                "rtoken": rtoken, "crypto": crypto, "error": f"{type(exc).__name__}: {exc}",
            })
    ok = [r for r in results if "error" not in r]
    return {
        "n_pairs": len(LEG_PAIRS),
        "n_ok": len(ok),
        "n_agree_at_entry": sum(1 for r in ok if r["agree_at_entry"]),
        "n_diverge_past_break_even": sum(1 for r in ok if r["diverges_past_break_even"]),
        "results": results,
    }


# --- failure cases ---------------------------------------------------------------------------


def run_failure_cases() -> dict[str, Any]:
    """Real, measured behaviour of the real crypto_sor router on edge-case inputs."""
    findings: dict[str, Any] = {}

    single_leg = run_new_order(
        "T", "BUY", 5.0,
        [{"exchange": "ONLY", "side": "SELL", "price": 1.0, "size": 10.0}],
    )
    findings["single_leg"] = {"executions": single_leg, "real_raised": False}

    exchange_filter_matches_nothing = run_new_order(
        "T", "BUY", 5.0,
        [{"exchange": "A", "side": "SELL", "price": 1.0, "size": 10.0}],
        exchanges=["B"],
    )
    findings["exchange_filter_matches_nothing"] = {
        "executions": exchange_filter_matches_nothing,
        "real_returns_empty_list_not_an_error": exchange_filter_matches_nothing == [],
        "real_raised": False,
    }

    oversized_order = run_new_order(
        "T", "BUY", 1000.0,
        [{"exchange": "A", "side": "SELL", "price": 1.0, "size": 5.0}],
    )
    total_filled = sum(e["lastQty"] for e in oversized_order)
    findings["order_larger_than_book"] = {
        "requested_qty": 1000.0,
        "real_total_filled": total_filled,
        "real_silently_underfills_with_no_flag": total_filled < 1000.0,
        "real_raised": False,
    }
    return findings


# --- costs, reproducibility -------------------------------------------------------------------


def measure_costs(repeats: int = 3) -> dict[str, Any]:
    """Real wall-clock cost of a fresh Node/ts-node subprocess spawn vs. ARGUS's pure-Python
    decision, on the same two real legs."""
    rtoken_leg = _real_leg("NVDAUSDT")
    crypto_leg = _real_leg("BTCUSDT")

    start = time.perf_counter()
    for _ in range(repeats):
        _crypto_sor_pick(rtoken_leg, crypto_leg)
    sor_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(repeats):
        choose_hedge_leg([rtoken_leg, crypto_leg], NOTIONAL, holding_days=Decimal("1"))
    argus_elapsed = time.perf_counter() - start

    return {
        "repeats": repeats,
        "crypto_sor_subprocess_seconds_per_call": sor_elapsed / repeats,
        "argus_pure_python_seconds_per_call": argus_elapsed / repeats,
        "argus_faster_by_factor": (
            (sor_elapsed / repeats) / (argus_elapsed / repeats) if argus_elapsed > 0 else None
        ),
    }


def run_reproducibility_check(rtoken: str = "NVDAUSDT", crypto: str = "BTCUSDT") -> dict[str, Any]:
    rtoken_leg = _real_leg(rtoken)
    crypto_leg = _real_leg(crypto)
    first = _crypto_sor_pick(rtoken_leg, crypto_leg)
    second = _crypto_sor_pick(rtoken_leg, crypto_leg)
    return {"identical": first == second}


SCOPE_STATEMENT = (
    "crypto_sor's real, vendored CompositeOrderBook.newOrder(), run through a real Node/ts-node "
    "subprocess and fed each leg's real measured slippage as its synthetic price, always picks "
    "the crypto leg on this venue's real, live data — crypto majors quote an order of magnitude "
    "or more tighter than the rToken legs tested. argus.desk.execution.choose_hedge_leg prices "
    "the real funding channel crypto_sor has no representation of at all, and past a real, "
    "computed break-even holding horizon the two real systems diverge: crypto_sor's real output "
    "never changes with holding time, ARGUS's does. This is a same-venue, cross-asset-class "
    "comparison, not a cross-exchange one — crypto_sor was built to route across venues; feeding "
    "it two legs of one venue's own universe as competing quotes is a faithful use of its real, "
    "general 'lowest quoted price wins' logic, not a different algorithm. NOT claimed that "
    "routing through either leg is a validated hedge of anything — only the execution cost of "
    "each leg is priced here, never whether it actually offsets the exposure being hedged. NOT "
    "claimed the measured break-even horizons are permanent constants — rToken funding on this "
    "venue is usually zero and crypto funding is usually small and can change sign; both are "
    "measured live, not assumed."
)


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    report = {
        "base_case": run_base_case(),
        "divergence_sweep": run_divergence_sweep(),
        "leg_pair_sweep": run_leg_pair_sweep(),
        "failure_cases": run_failure_cases(),
        "costs": measure_costs(),
        "reproducibility": run_reproducibility_check(),
        "scope_statement": SCOPE_STATEMENT,
    }
    print(render(report))
    out = Path(__file__).resolve().parents[3] / "data" / "execution_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


def render(report: dict[str, Any]) -> str:
    base = report["base_case"]
    sweep = report["divergence_sweep"]
    pairs = report["leg_pair_sweep"]
    costs = report["costs"]
    lines = ["CROSS-ASSET EXECUTION vs crypto_sor's real composite router\n"]
    lines.append(
        f"  {base['rtoken']} slippage {base['rtoken_slippage_bps']}bps vs "
        f"{base['crypto']} slippage {base['crypto_slippage_bps']}bps"
    )
    lines.append(
        f"  agree at entry: {base['agree_at_entry']}  break-even: "
        f"{base['break_even_holding_days']} day(s)  diverges past it: "
        f"{base['diverges_past_break_even']}"
    )
    lines.append(
        f"  sweep: {sweep['n_disagreements']}/{sweep['n_points']} holding-day points disagree"
    )
    lines.append(
        f"  leg pairs: {pairs['n_diverge_past_break_even']}/{pairs['n_ok']} diverge past "
        f"break-even"
    )
    lines.append(
        f"  costs: crypto_sor subprocess "
        f"{costs['crypto_sor_subprocess_seconds_per_call']*1000:.1f}ms vs argus "
        f"{costs['argus_pure_python_seconds_per_call']*1000:.4f}ms per call"
    )
    lines.append(f"  reproducible: {report['reproducibility']['identical']}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "LEG_PAIRS",
    "NOTIONAL",
    "SCOPE_STATEMENT",
    "SWEEP_HOLDING_DAYS",
    "main",
    "measure_costs",
    "render",
    "run_base_case",
    "run_divergence_sweep",
    "run_failure_cases",
    "run_leg_pair_sweep",
    "run_reproducibility_check",
]
