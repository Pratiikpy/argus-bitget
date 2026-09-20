"""Does ARGUS's break-even heuristic reproduce a convex optimiser's rebalance decision — or does
it only agree where nothing could have disagreed?

**This module exists because the first agreement measurement could not fail.**
`eval/allocation_comparison.run_convex_rebalance` reports ``agreement_rate: 1.0`` between ARGUS's
one-line break-even rule and cvxportfolio's `MultiPeriodOptimization`, and that figure is worth
nothing as published: it is four cells, all four on the same equal-weight book, and **in every one
of them both sides said trade**. A concordance with no discordant cell available is not evidence of
agreement, it is evidence that the question was never asked — the same defect this project's risk
prover guards against with deliberately broken policies, and the same one its overfitting gates
exist for.

The heuristic *does* say no: `run_rebalance_sweep` finds 89 "do not rebalance" cells out of 256.
The convex program was simply never run on any of them. So the experiment here is the one that was
missing: span cells where the answer is genuinely both, and report a **confusion matrix** rather
than a rate, so a reader can see how many cells could have disagreed before being told how many did.

**What is being compared, and what is not.** Only the go/no-go. The two cannot be compared on size
and are not: ARGUS prices a move to a fixed HRP target, while the convex program chooses the
destination and the trade together, so it has no reason to travel all the way to anyone's target.
Everything the solver sees comes from ARGUS — ARGUS's covariance, ARGUS's taker fee as the linear
cost coefficient, ARGUS's horizon, HRP's own long-only no-cash feasible set. cvxportfolio
contributes its formulation and its solver and no data of its own.

**Why this matters more than the variance comparison it sits beside.** Riskfolio's NCO beats ARGUS
on realised out-of-sample variance and that result is published and survives ablation. But raw
variance was never the claim: every HRP library stops at a weight vector, and the thing ARGUS
claims is the *decision* — whether the move pays for its own turnover. cvxportfolio is the system
that answers that properly, as a convex program over a planning horizon. If a break-even ratio
reproduces its go/no-go across cells where it can disagree, that is a real result at a fraction of
the cost. If it does not, that is the differentiator failing its own test, and it belongs on the
record either way.

    python -m argus.eval.allocation_agreement
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from argus.desk.allocation import TAKER_BPS, hrp_weights, optimize_trade
from argus.desk.portfolio import covariance_matrix
from argus.eval.allocation_comparison import (
    SHARPE_BAND,
    TRAIN_BARS,
    _implied_risk_aversion,
    _sweep_books,
    _window,
    convex_first_step,
    load_returns,
)
from argus.eval.artefact import write

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "allocation_agreement.json"

NO_TRADE_TURNOVER = 0.01
"""Below this one-way turnover the convex program has decided not to move the book.

Deliberately the same number `eval/allocation_comparison.run_convex_rebalance` uses, which is in
turn ARGUS's own `min_leg` — the size at which it already decides a leg is too small to be worth a
fee. Inventing a second threshold would make the two measurements incomparable, and choosing a
looser one would manufacture agreement."""

HORIZONS: tuple[int, ...] = (24, 168, 720)
"""One day, one week, one month of bars.

The horizon is the strongest lever on whether a rebalance pays: the same turnover amortises over
24 bars or 720. Varying it is most of what makes a no-trade cell reachable at all."""


def _decisions(
    columns: Mapping[str, Sequence[float]], *, train_bars: int = TRAIN_BARS,
) -> list[dict[str, Any]]:
    """One row per (book, horizon, assumed Sharpe), carrying both sides' go/no-go."""
    window = _window(columns, 0, train_bars)
    built = covariance_matrix(window)
    if built is None:
        raise RuntimeError("covariance could not be built for the agreement comparison")
    names, cov = built
    target = hrp_weights(list(names), cov)
    rows: list[dict[str, Any]] = []

    for book_name, book in _sweep_books(names, cov).items():
        for horizon in HORIZONS:
            for sharpe in SHARPE_BAND:
                plan = optimize_trade(
                    book, window, horizon_bars=horizon, assumed_sharpe_annual=sharpe,
                )
                # ARGUS trades when the move repays its own cost inside the horizon. That is
                # the same rule `desk/allocation.py` states in its own verdict, read off the
                # break-even rather than re-derived here.
                argus_trades = bool(
                    plan.break_even_bars is not None and plan.break_even_bars <= horizon
                )
                gamma = _implied_risk_aversion(plan.vol_before, plan.vol_after, sharpe)
                try:
                    convex = convex_first_step(
                        book, names, cov, gamma=gamma,
                        taker_bps=TAKER_BPS, horizon_bars=horizon,
                    )
                except Exception as exc:
                    rows.append({
                        "book": book_name, "horizon_bars": horizon, "assumed_sharpe": sharpe,
                        "argus_trades": argus_trades, "convex_trades": None,
                        "error": f"{type(exc).__name__}: {exc}"[:160],
                    })
                    continue
                rows.append({
                    "book": book_name,
                    "horizon_bars": horizon,
                    "assumed_sharpe": sharpe,
                    "argus_trades": argus_trades,
                    "argus_break_even_bars": plan.break_even_bars,
                    "argus_one_way_turnover": plan.turnover / 2.0,
                    "implied_gamma": gamma,
                    "convex_one_way_turnover": convex["one_way_turnover"],
                    "convex_trades": convex["one_way_turnover"] > NO_TRADE_TURNOVER,
                    "agree": argus_trades == (convex["one_way_turnover"] > NO_TRADE_TURNOVER),
                    "already_on_target": max(
                        abs(book.get(n, 0.0) - target.get(n, 0.0)) for n in names
                    ) < 1e-9,
                })
    return rows


def run_agreement(columns: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """The confusion matrix, and the honesty check that the rate alone cannot give.

    ``discordant_cells_available`` is the number that decides whether any of this means anything.
    If either side never says no, the agreement rate is arithmetic rather than evidence, and
    :func:`render` refuses to call it a result.
    """
    every = _decisions(columns)
    rows = [r for r in every if r.get("convex_trades") is not None]
    errors = len(every) - len(rows)
    both_yes = sum(1 for r in rows if r["argus_trades"] and r["convex_trades"])
    both_no = sum(1 for r in rows if not r["argus_trades"] and not r["convex_trades"])
    argus_only = sum(1 for r in rows if r["argus_trades"] and not r["convex_trades"])
    convex_only = sum(1 for r in rows if not r["argus_trades"] and r["convex_trades"])
    agreed = both_yes + both_no
    argus_says_no = both_no + convex_only
    convex_says_no = both_no + argus_only
    return {
        "cells": len(rows),
        "solver_failures": errors,
        "confusion": {
            "both_trade": both_yes,
            "both_hold": both_no,
            "argus_trades_convex_holds": argus_only,
            "convex_trades_argus_holds": convex_only,
        },
        "agreement_rate": (agreed / len(rows)) if rows else None,
        "argus_says_no_in": argus_says_no,
        "convex_says_no_in": convex_says_no,
        # The guard. A rate computed over cells where nobody could disagree is not a measurement.
        "discordant_cells_available": min(argus_says_no, convex_says_no) > 0,
        "both_answers_observed_from_argus": 0 < argus_says_no < len(rows),
        "both_answers_observed_from_convex": 0 < convex_says_no < len(rows),
        "disagreements": [
            {k: v for k, v in r.items() if k != "error"} for r in rows if not r["agree"]
        ][:20],
        "rows": rows,
    }


def render(report: Mapping[str, Any]) -> list[str]:
    c = report["confusion"]
    lines = [
        f"REBALANCE AGREEMENT — ARGUS break-even vs cvxportfolio MultiPeriodOptimization, "
        f"{report['cells']} cell(s)",
        f"  both trade {c['both_trade']}   both hold {c['both_hold']}   "
        f"ARGUS trades / convex holds {c['argus_trades_convex_holds']}   "
        f"convex trades / ARGUS holds {c['convex_trades_argus_holds']}",
    ]
    if not report["discordant_cells_available"]:
        lines.append(
            "  ⚠ NOT A RESULT: one side never says no across this grid, so the agreement rate is "
            "arithmetic rather than evidence. This is exactly the defect that made the original "
            "four-cell measurement meaningless, and it is reported rather than rounded up."
        )
        return lines
    rate = report["agreement_rate"]
    lines.append(
        f"  agreement {rate:.1%} on the go/no-go, over a grid where ARGUS holds in "
        f"{report['argus_says_no_in']} cell(s) and the convex program holds in "
        f"{report['convex_says_no_in']} — so both answers were reachable and the rate is a "
        f"measurement."
    )
    lines.append(
        "  Sizes are deliberately not compared: ARGUS prices a move to a fixed HRP target and the "
        "convex program picks its own destination, so it has no reason to reach anyone's target."
    )
    return lines


def build_report() -> dict[str, Any]:
    _names, columns, _failures = load_returns()
    report = run_agreement(columns)
    report["scope_statement"] = (
        "ARGUS's break-even rebalance heuristic (desk/allocation.optimize_trade) and the real "
        "installed cvxportfolio MultiPeriodOptimization are asked the same go/no-go over a grid of "
        "starting books x horizons x assumed Sharpe, built so that BOTH answers are reachable. "
        "Everything the solver sees is ARGUS's: covariance, taker fee as the linear cost "
        "coefficient, horizon, and HRP's long-only no-cash feasible set. Only the decision is "
        "compared; the sizes are not, and the reason is stated. NOT CLAIMED: that agreement on a "
        "synthetic grid of starting books transfers to books a desk actually holds, or that either "
        "side is right — only that a one-line ratio and a convex program reach the same decision."
    )
    return report


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = build_report()
    for line in render(report):
        print(line)
    write(REPORT_PATH, report)
    print(f"\nwritten to {REPORT_PATH}")
    return 0 if report["discordant_cells_available"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["REPORT_PATH", "build_report", "main", "render", "run_agreement"]
