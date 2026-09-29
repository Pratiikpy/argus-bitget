"""Does `desk/stress_search`'s failure probability hold out of sample, and does it beat the general
tool?

On the frozen twelve-rToken hourly snapshot and the recorded order books, each symbol's 24-hour
windows are split in time: the first half estimates the chance a long loses more than its limit,
the second half is what happened.

**The truth is the price alone.** Until 2026-09-29 the "realised" figure was ARGUS's own search run
on the second half, so whatever the book rule added sat in both the estimate and the truth, and a
book too shallow to price the exit counted as a certain breach in both
(Activity/28_CAPABILITY_CLOSE_PLAN_2.md §46). Now the realised chance is the share of held-out
windows whose move, plus the median exit cost, took the long past its limit. Only non-overlapping
windows (every 24th) are used for it, so neighbouring truths are not the same hours counted twice.

**The arms.**

* ``argus``: `desk/stress_search.search` on the first half, every observed move against every
  recorded book deep enough to price the exit;
* ``historical``: the general tool, historical simulation: the share of the first half's moves
  that, with the median exit cost, breach the limit. It is the frequency skfolio's and
  riskfolio's historical VaR are read from;
* ``normal``: the method of `Open-Finance-Lab/AgenticTrading`'s `run_monte_carlo_stress`
  (`risk_agent_pool/agents/stress_testing.py:335-441`, read via `gh api`, licence NOASSERTION, so
  described and rebuilt here, not copied), in closed form: a normal fit to the first half. Kept as
  context.

The search question is unchanged: a uniform random search at a fixed budget, the shape of the same
repository's `run_reverse_stress_test` (`:445-530`), run 200 times per budget.

    python -m argus.eval.stress_search_comparison
"""

from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime
from decimal import Decimal
from statistics import NormalDist
from typing import Any

from argus.desk.stress import closes_from_returns, horizon_moves
from argus.desk.stress_search import (
    EXIT_FEE_BPS,
    exit_costs,
    random_search,
    search,
    tape_books,
)
from argus.eval.allocation_snapshot import load_snapshot
from argus.truth.artefact import write
from argus.truth.paths import DATA_DIR

OUT = DATA_DIR / "stress_search_comparison.json"
TOLERANCES = (Decimal("3"), Decimal("5"), Decimal("10"))
NOTIONAL = Decimal("50000")
"""The desk's per-position cap (`risk/constitution.max_position_notional`): the size tested."""
BUDGETS = (100, 1000, 10000)
RUNS = 200


def normal_probability(pcts: list[float], *, tolerance: float, cost_pct: float) -> float:
    """P(loss > tolerance) for a long under a normal fit to the moves: the move must fall below
    ``-(tolerance - cost)``."""
    mean, sd = statistics.fmean(pcts), statistics.pstdev(pcts)
    if sd <= 0:
        return 0.0
    return NormalDist(mean, sd).cdf(-(tolerance - cost_pct))


def price_truth(moves: list[float], *, tolerance: float, cost_pct: float) -> float:
    """Share of ``moves`` (percent) whose loss, with the exit cost, reaches ``tolerance``."""
    if not moves:
        return 0.0
    return sum(1 for m in moves if -m + cost_pct >= tolerance) / len(moves)


def main() -> int:  # pragma: no cover - CLI
    stamps, columns, digest = load_snapshot()
    at = [datetime.fromisoformat(s) for s in stamps]
    rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    depth: dict[str, dict[str, int]] = {}
    for symbol in sorted(columns):
        books = tape_books(symbol)
        moves = horizon_moves(closes_from_returns(list(zip(at, columns[symbol], strict=True))),
                              bars=24)
        if not books or len(moves) < 200:
            continue
        half = len(moves) // 2
        # Non-overlapping split: the 24 windows either side of the cut share bars, so they go.
        fit, test = moves[:half - 24], moves[half:]
        test_disjoint = [float(m.pct) for m in test[::24]]
        price = books[-1].mid
        quantity = (NOTIONAL / price).quantize(Decimal("0.001"))
        costs = exit_costs(books, quantity, long=True)
        priced = [c for c in costs if c.complete]
        depth[symbol] = {"books": len(costs), "too_shallow": len(costs) - len(priced)}
        if not priced:
            continue
        median_cost = float(statistics.median(float(c.slippage_bps) for c in priced)
                            + float(EXIT_FEE_BPS)) / 100
        fit_pcts = [float(m.pct) for m in fit]
        vol = statistics.pstdev(fit_pcts)
        for tolerance in TOLERANCES:
            ours = search(symbol, fit, books, quantity=quantity, tolerance_pct=tolerance)
            realised = price_truth(test_disjoint, tolerance=float(tolerance), cost_pct=median_cost)
            historical = price_truth(fit_pcts, tolerance=float(tolerance), cost_pct=median_cost)
            normal = normal_probability(fit_pcts, tolerance=float(tolerance),
                                        cost_pct=median_cost)
            own_truth = search(symbol, test, books, quantity=quantity, tolerance_pct=tolerance)
            rows.append({
                "symbol": symbol, "tolerance_pct": float(tolerance),
                "quantity": str(quantity), "fit_windows": len(fit),
                "test_windows_disjoint": len(test_disjoint), "fit_volatility_pct": round(vol, 4),
                "median_exit_cost_pct": round(median_cost, 5),
                "argus": float(ours.failure_probability), "historical": historical,
                "normal": normal, "realised": realised,
                "argus_error": abs(float(ours.failure_probability) - realised),
                "historical_error": abs(historical - realised),
                "normal_error": abs(normal - realised),
                "argus_rule_truth": float(own_truth.failure_probability),
            })
        exact = search(symbol, moves, books, quantity=quantity, tolerance_pct=Decimal("5"))
        if exact.most_likely is not None:
            target = exact.most_likely.probability
            for budget in BUDGETS:
                found = sum(
                    1 for seed in range(RUNS)
                    if (hit := random_search(symbol, moves, books, quantity=quantity,
                                             tolerance_pct=Decimal("5"), budget=budget,
                                             seed=seed)) is not None
                    and hit.probability >= target)
                search_rows.append({"symbol": symbol, "budget": budget,
                                    "space": exact.evaluated, "runs": RUNS,
                                    "found_most_likely": found / RUNS})

    def mean(key: str, subset: list[dict[str, Any]]) -> float:
        return statistics.fmean(r[key] for r in subset) if subset else 0.0

    def block(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {"rows": len(subset),
                **{f"{arm}_mae": round(mean(f"{arm}_error", subset), 5)
                   for arm in ("argus", "historical", "normal")},
                "argus_closer_than_historical": sum(
                    r["argus_error"] < r["historical_error"] for r in subset),
                "historical_closer_than_argus": sum(
                    r["historical_error"] < r["argus_error"] for r in subset)}

    by_tolerance = {str(t): block([r for r in rows if r["tolerance_pct"] == float(t)])
                    for t in TOLERANCES}
    vols = sorted({(r["fit_volatility_pct"], r["symbol"]) for r in rows}, reverse=True)
    volatile = {name for _, name in vols[:max(1, len(vols) // 4)]}
    blob: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input": {"snapshot_digest": digest, "horizon_bars": 24,
                  "books": "data/book_tape.jsonl", "notional": str(NOTIONAL)},
        "truth": "the share of non-overlapping held-out 24h windows whose move, with the median "
                 "exit cost, took a long past its limit; ARGUS's own book rule is kept beside it "
                 "as argus_rule_truth, as context",
        # The summary first, so the per-row list is not overwritten by its own count.
        "calibration": {**block(rows), "rows": rows, "by_tolerance": by_tolerance},
        "ablation": {"without_the_book_term": "historical: the same moves with the median exit "
                                              "cost in place of every recorded book",
                     "all": block(rows)},
        "adversarial": {"most_volatile_quarter_of_names": sorted(volatile),
                        **block([r for r in rows if r["symbol"] in volatile])},
        "failure_cases": {"books_too_shallow_for_the_exit": depth,
                          "note": "a book whose 50 stored levels end before the exit is filled "
                                  "leaves the calculation (desk/stress_search.UNKNOWN_DEPTH)"},
        "search": {"rows": search_rows, "by_budget": {
            str(b): mean("found_most_likely", [r for r in search_rows if r["budget"] == b])
            for b in BUDGETS}},
        "caveats": [
            "the books were recorded 2026-09-14..28, after most of the price history; the same "
            "books serve both halves",
            "the realised share uses every 24th held-out window only, so each is a few dozen "
            "observations per name; the per-name errors are coarse",
        ],
    }
    write(OUT, blob)
    print(OUT)
    print(json.dumps(by_tolerance, indent=1))
    print(blob["search"]["by_budget"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
