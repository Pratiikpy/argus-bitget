"""Same-input comparison: `desk/constrained.py` against skfolio's `MeanRisk`, on the frozen rToken
return snapshot, under the limits a portfolio manager actually states.

skfolio (BSD-3-Clause, 1.3.1 installed) is the specialist this capability was studied against
(research/harvest/46-cvxpy-allocation.md §8 items 1-2): its ``linear_constraints`` grammar and its
``cardinality``/``group_cardinalities`` are exactly what `desk/constrained.py` adopts. Both are run
here on one covariance, under each scenario, and each result is graded on the same three things:

* **the variance** of the book it returns, under one covariance, so a lower number is better;
* **whether the book obeys the limits it was given**, recomputed here from the weights rather than
  taken from either solver's status;
* **what happens when it cannot**: an error, a refusal, or a book that silently breaks a limit.

skfolio's cardinality needs a mixed-integer solver. It is given its own recommended one, SCIP
(`optimization/convex/_base.py:183-186`), so the comparison is against skfolio as its authors
intend it to run; the default-solver failure is recorded separately. The tracking-HRP objective
ARGUS uses by default has no skfolio equivalent, so it is checked against CVXPY directly.

Two things skfolio does that are not errors of this module but are measured because a trader would
meet them: ``MeanRisk`` hard-codes ``raise_if_group_missing=False`` (`_base.py:1086-1089`), so a
limit naming a misspelt group is dropped with a warning and the book returned breaks it; and it has
no per-name risk-share cap, which is recorded as not run rather than scored.

    python -m argus.eval.constrained_comparison
"""

from __future__ import annotations

import importlib
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from argus.desk.constrained import ConstraintError, Limits, allocate, parse_limit
from argus.eval.allocation_snapshot import load_snapshot
from argus.truth.artefact import write
from argus.truth.paths import DATA_DIR

OUT = DATA_DIR / "constrained_comparison.json"
GROUPS: dict[str, tuple[str, ...]] = {
    "megacap": ("AAPLUSDT", "AMZNUSDT", "GOOGLUSDT", "METAUSDT", "MSFTUSDT"),
    "crypto": ("COINUSDT", "MSTRUSDT"),
    "index": ("QQQUSDT", "TQQQUSDT", "SQQQUSDT"),
    "growth": ("NVDAUSDT", "TSLAUSDT"),
}
SCENARIOS: tuple[tuple[str, str, Limits], ...] = (
    ("no limits", "long-only, fully invested", Limits()),
    ("15% cap", "no name above 15%", Limits(max_weight=0.15)),
    ("group limits", "crypto at most 10%, index at most 30%, megacaps at least 40%",
     Limits(groups=GROUPS, linear=("crypto <= 0.1", "index <= 0.3", "megacap >= 0.4"))),
    ("relative limit", "NVDA at least half of MSFT, TSLA at most 5%",
     Limits(linear=("NVDAUSDT >= 0.5 * MSFTUSDT", "TSLAUSDT <= 0.05"))),
    ("at most 4 names", "cardinality 4, 40% cap", Limits(max_names=4, max_weight=0.4)),
    ("one per group", "at most one name from each group, at most 5 names",
     Limits(groups=GROUPS, max_names=5,
            max_names_per_group={"megacap": 1, "crypto": 1, "index": 1, "growth": 1})),
)
MISSPELT = Limits(groups=GROUPS, linear=("cryptos >= 0.15",))
"""A trader's typo: the group is ``crypto``."""


def _version(package: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def covariance(columns: Mapping[str, Sequence[float]], names: Sequence[str]) -> list[list[float]]:
    size = len(columns[names[0]])
    means = {k: sum(columns[k]) / size for k in names}
    return [[sum((columns[a][t] - means[a]) * (columns[b][t] - means[b]) for t in range(size))
             / (size - 1) for b in names] for a in names]


def variance(weights: Mapping[str, float], names: Sequence[str],
             cov: Sequence[Sequence[float]]) -> float:
    w = [weights.get(k, 0.0) for k in names]
    return sum(w[i] * cov[i][j] * w[j] for i in range(len(w)) for j in range(len(w)))


def breaches(weights: Mapping[str, float], names: Sequence[str], limits: Limits,
             tol: float = 1e-6) -> list[str]:
    """Every limit the weights break, recomputed from the weights alone."""
    w = [weights.get(k, 0.0) for k in names]
    out: list[str] = []
    if abs(sum(w) - limits.budget) > tol:
        out.append(f"sum {sum(w):.6f}")
    if min(w) < limits.min_weight - tol or max(w) > limits.max_weight + tol:
        out.append(f"weight range [{min(w):.4f}, {max(w):.4f}]")
    groups = {k: v for k, v in limits.groups.items()}
    for text in limits.linear:
        try:
            lim = parse_limit(text, names, groups)
        except ConstraintError:
            continue
        excess = sum(c * v for c, v in zip(lim.coeffs, w, strict=True)) - lim.bound
        if (abs(excess) if lim.equality else excess) > tol:
            out.append(f"{text} (by {excess:.4f})")
    held = {k for k, v in zip(names, w, strict=True) if abs(v) > 1e-6}
    if limits.max_names is not None and len(held) > limits.max_names:
        out.append(f"{len(held)} names held")
    for group, cap in limits.max_names_per_group.items():
        if len(held & set(limits.groups[group])) > cap:
            out.append(f"{group}: {len(held & set(limits.groups[group]))} names")
    return out


def run_skfolio(columns: Mapping[str, Sequence[float]], names: Sequence[str], limits: Limits,
                solver: str | None = None) -> dict[str, Any]:  # pragma: no cover - optional
    import numpy as np
    import pandas as pd
    # skfolio ships no type information; its classes are reached as Any, like cvxpy below.
    measures: Any = importlib.import_module("skfolio.measures")
    optimization: Any = importlib.import_module("skfolio.optimization")
    RiskMeasure, MeanRisk = measures.RiskMeasure, optimization.MeanRisk
    ObjectiveFunction = optimization.ObjectiveFunction

    frame = pd.DataFrame({k: list(columns[k]) for k in names})
    kwargs: dict[str, Any] = {
        "risk_measure": RiskMeasure.VARIANCE,
        "objective_function": ObjectiveFunction.MINIMIZE_RISK,
        "min_weights": limits.min_weight, "max_weights": limits.max_weight,
        "budget": limits.budget,
    }
    if limits.groups:
        kwargs["groups"] = np.array(
            [[next((g for g, m in limits.groups.items() if k in m), "ungrouped")
              for k in names]])
    if limits.linear:
        kwargs["linear_constraints"] = list(limits.linear)
    if limits.max_names is not None:
        kwargs["cardinality"] = limits.max_names
    if limits.max_names_per_group:
        kwargs["group_cardinalities"] = dict(limits.max_names_per_group)
    if solver:
        kwargs["solver"] = solver
    started = time.perf_counter()
    caught: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            model = MeanRisk(**kwargs).fit(frame)
            caught = [str(w.message)[:200] for w in seen]
        weights = {k: float(v) for k, v in zip(names, model.weights_, strict=True)}
        return {"ok": True, "weights": weights, "seconds": time.perf_counter() - started,
                "warnings": caught, "solver": solver or "CLARABEL"}
    except Exception as exc:  # the failure is the measurement
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                "seconds": time.perf_counter() - started, "solver": solver or "CLARABEL"}


def run_ours(names: Sequence[str], cov: Sequence[Sequence[float]], limits: Limits,
             objective: str = "min-variance") -> dict[str, Any]:
    started = time.perf_counter()
    try:
        book = allocate(names, cov, limits, objective=objective)  # type: ignore[arg-type]
    except ConstraintError as exc:
        return {"ok": False, "error": f"refused: {exc}",
                "seconds": time.perf_counter() - started}
    return {"ok": book.status != "infeasible", "status": book.status, "weights": book.weights,
            "seconds": time.perf_counter() - started, "nodes": book.supports_searched,
            "binding": list(book.binding), "reason": book.reason}


def grade(result: dict[str, Any], names: Sequence[str], cov: Sequence[Sequence[float]],
          limits: Limits) -> dict[str, Any]:
    if not result.get("ok"):
        return {k: v for k, v in result.items() if k != "weights"}
    weights = result["weights"]
    return {
        **{k: v for k, v in result.items() if k != "weights"},
        "annual_volatility": (variance(weights, names, cov) * 24 * 365) ** 0.5,
        "breaches": breaches(weights, names, limits),
        "holdings": {k: round(v, 6) for k, v in weights.items() if abs(v) > 1e-6},
    }


def run_tracking_check(names: Sequence[str], cov: Sequence[Sequence[float]]
                       ) -> list[dict[str, Any]]:  # pragma: no cover - optional
    """The default objective (least tracking variance from HRP) against CVXPY on the same
    problem, the covariance scaled to a unit mean diagonal for both."""
    cp: Any = importlib.import_module("cvxpy")
    import numpy as np

    out: list[dict[str, Any]] = []
    n = len(names)
    sigma = np.array(cov)
    sigma = sigma / np.trace(sigma) * n
    for label, _, limits in SCENARIOS[:4]:
        ours = allocate(names, cov, limits)
        target = np.array([ours.target[k] for k in names])
        w = cp.Variable(n)
        cons = [cp.sum(w) == limits.budget, w >= limits.min_weight, w <= limits.max_weight]
        for text in limits.linear:
            lim = parse_limit(text, names, limits.groups)
            expr = np.array(lim.coeffs) @ w
            cons.append(expr == lim.bound if lim.equality else expr <= lim.bound)
        problem = cp.Problem(cp.Minimize(cp.quad_form(w - target, cp.psd_wrap(sigma))), cons)
        problem.solve(solver="CLARABEL")
        mine = np.array([ours.weights[k] for k in names])
        value = float((mine - target) @ sigma @ (mine - target))
        out.append({"scenario": label, "argus": value, "cvxpy_clarabel": float(problem.value),
                    "argus_minus_cvxpy": value - float(problem.value),
                    "argus_breaches": breaches(ours.weights, names, limits)})
    return out


def main() -> int:  # pragma: no cover - CLI
    _, columns, digest = load_snapshot()
    names = sorted(columns)
    cov = covariance(columns, names)
    rows: list[dict[str, Any]] = []
    for label, plain, limits in SCENARIOS:
        counted = limits.max_names is not None or bool(limits.max_names_per_group)
        row: dict[str, Any] = {
            "scenario": label, "limits": plain,
            "argus": grade(run_ours(names, cov, limits), names, cov, limits),
            "skfolio": grade(run_skfolio(columns, names, limits,
                                         solver="SCIP" if counted else None), names, cov, limits),
        }
        if counted:
            row["skfolio_default_solver"] = grade(run_skfolio(columns, names, limits),
                                                  names, cov, limits)
        a, s = row["argus"], row["skfolio"]
        if a.get("ok") and s.get("ok") and not a["breaches"] and not s["breaches"]:
            diff = a["annual_volatility"] - s["annual_volatility"]
            row["verdict"] = ("tie" if abs(diff) < 1e-6 else
                              "argus lower" if diff < 0 else "skfolio lower")
            row["volatility_difference"] = diff
        else:
            row["verdict"] = "see breaches and errors"
        rows.append(row)

    typo: dict[str, Any] = {
        "limit": MISSPELT.linear[0], "intended_group": "crypto",
        "argus": grade(run_ours(names, cov, MISSPELT), names, cov, MISSPELT),
        "skfolio": grade(run_skfolio(columns, names, MISSPELT), names, cov, MISSPELT),
    }
    if typo["skfolio"].get("ok"):
        intended = replace(MISSPELT, linear=("crypto >= 0.15",))
        held = run_skfolio(columns, names, MISSPELT)["weights"]
        typo["skfolio"]["breaks_the_intended_limit"] = breaches(held, names, intended)

    counted_rows = [r for r in rows if r["verdict"] != "see breaches and errors"]
    blob = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input": {"snapshot_digest": digest, "names": names,
                  "observations": len(columns[names[0]])},
        "objective": "minimum variance (skfolio MeanRisk, RiskMeasure.VARIANCE)",
        "versions": {name: _version(name) for name in ("skfolio", "cvxpy", "pyscipopt")},
        "scenarios": rows,
        "misspelt_group": typo,
        "tracking_objective_vs_cvxpy": run_tracking_check(names, cov),
        "not_run": ["per-name risk-share cap: skfolio has no equivalent constraint"],
        "summary": {
            "comparable": len(counted_rows),
            "ties": sum(r["verdict"] == "tie" for r in counted_rows),
            "argus_lower": sum(r["verdict"] == "argus lower" for r in counted_rows),
            "skfolio_lower": sum(r["verdict"] == "skfolio lower" for r in counted_rows),
        },
    }
    write(OUT, blob)
    print(OUT)
    for r in rows:
        print(r["scenario"], "|", r["verdict"], "|", r["argus"].get("annual_volatility"),
              r["skfolio"].get("annual_volatility") or r["skfolio"].get("error"))
    print("typo:", typo["argus"].get("error"), "|", typo["skfolio"].get("warnings"),
          typo["skfolio"].get("breaks_the_intended_limit"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
