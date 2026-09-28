"""Does `desk/stress_search`'s failure probability hold out of sample, and does enumeration beat
the searches the field uses?

Two questions, on the frozen twelve-rToken hourly snapshot and the recorded order books:

1. **Calibration, out of sample.** Each symbol's 24-hour windows are split in time. The first half
   estimates the chance a position loses more than its limit; the second half is what happened.
   ARGUS's estimate enumerates observed moves against recorded books. The rival is the method of
   `Open-Finance-Lab/AgenticTrading`'s `run_monte_carlo_stress` (`risk_agent_pool/agents/
   stress_testing.py:335-441`, read via `gh api`, licence NOASSERTION, so described and rebuilt
   here, not copied): shocks drawn from a normal distribution with the sample's mean and standard
   deviation. Its probability is taken in closed form, the limit of its Monte Carlo, so sampling
   noise is not counted against it. The same exit cost is added to both, so the comparison is of
   the price model alone.
2. **Search.** The study's kill criterion (research/harvest/16-adaptive-stress-testing.md): a
   uniform random search at a fixed budget — the shape of the same repository's
   `run_reverse_stress_test` (`:445-530`) — is run 200 times per budget, and the share of runs
   that find the exact most likely failure is recorded.

    python -m argus.eval.stress_search_comparison
"""

from __future__ import annotations

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


def main() -> int:  # pragma: no cover - CLI
    stamps, columns, digest = load_snapshot()
    at = [datetime.fromisoformat(s) for s in stamps]
    rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    for symbol in sorted(columns):
        books = tape_books(symbol)
        moves = horizon_moves(closes_from_returns(list(zip(at, columns[symbol], strict=True))),
                              bars=24)
        if not books or len(moves) < 200:
            continue
        half = len(moves) // 2
        # Non-overlapping split: the 24 windows either side of the cut share bars, so they go.
        fit, test = moves[:half - 24], moves[half:]
        price = books[-1].mid
        quantity = (NOTIONAL / price).quantize(Decimal("0.001"))
        costs = exit_costs(books, quantity, long=True)
        median_cost = float(statistics.median(float(c.slippage_bps) for c in costs)
                            + float(EXIT_FEE_BPS)) / 100
        for tolerance in TOLERANCES:
            ours = search(symbol, fit, books, quantity=quantity, tolerance_pct=tolerance)
            realised = search(symbol, test, books, quantity=quantity, tolerance_pct=tolerance)
            normal = normal_probability([float(m.pct) for m in fit], tolerance=float(tolerance),
                                        cost_pct=median_cost)
            rows.append({
                "symbol": symbol, "tolerance_pct": float(tolerance),
                "quantity": str(quantity), "fit_windows": len(fit), "test_windows": len(test),
                "argus": float(ours.failure_probability), "normal": normal,
                "realised": float(realised.failure_probability),
                "argus_error": abs(float(ours.failure_probability)
                                   - float(realised.failure_probability)),
                "normal_error": abs(normal - float(realised.failure_probability)),
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

    by_tolerance = {
        str(t): {"argus_mae": mean("argus_error", [r for r in rows
                                                   if r["tolerance_pct"] == float(t)]),
                 "normal_mae": mean("normal_error", [r for r in rows
                                                     if r["tolerance_pct"] == float(t)]),
                 "argus_closer": sum(r["argus_error"] < r["normal_error"] for r in rows
                                     if r["tolerance_pct"] == float(t)),
                 "normal_closer": sum(r["normal_error"] < r["argus_error"] for r in rows
                                      if r["tolerance_pct"] == float(t))}
        for t in TOLERANCES}
    blob: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input": {"snapshot_digest": digest, "horizon_bars": 24,
                  "books": "data/book_tape.jsonl", "notional": str(NOTIONAL)},
        "calibration": {"rows": rows, "by_tolerance": by_tolerance,
                        "argus_mae": mean("argus_error", rows),
                        "normal_mae": mean("normal_error", rows)},
        "search": {"rows": search_rows, "by_budget": {
            str(b): mean("found_most_likely", [r for r in search_rows if r["budget"] == b])
            for b in BUDGETS}},
        "caveats": [
            "the books were recorded 2026-09-14..28, after most of the price history; the same "
            "books serve both halves",
            "overlapping 24-hour windows: neighbouring observations are dependent, so the errors "
            "are descriptive, not a significance test",
        ],
    }
    write(OUT, blob)
    print(OUT)
    print(by_tolerance)
    print(blob["search"]["by_budget"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
