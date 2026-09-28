"""Allocation under a trader's own limits (research/harvest/46-cvxpy-allocation.md): the quadratic
program against CVXPY on the same inputs, the count cap against brute force over every set of
names, the risk-share cap against its own definition, and the limit language against skfolio's."""

from __future__ import annotations

import itertools
import random

import pytest

from argus.desk.constrained import (
    ConstraintError,
    Limits,
    _convex,
    _Problem,
    _tracking,
    allocate,
    parse_limit,
)


def book(n: int, seed: int) -> tuple[list[str], list[list[float]]]:
    """Returns driven by one common factor plus noise: correlated the way the rTokens are."""
    rng = random.Random(seed)
    periods = 300
    common = [rng.gauss(0, 0.01) for _ in range(periods)]
    cols = [[(0.6 + 0.1 * i) * common[t] + rng.gauss(0, 0.003 + 0.001 * i) for t in range(periods)]
            for i in range(n)]
    means = [sum(c) / periods for c in cols]
    cov = [[sum((cols[i][t] - means[i]) * (cols[j][t] - means[j]) for t in range(periods))
            / (periods - 1) for j in range(n)] for i in range(n)]
    return [f"N{i}" for i in range(n)], cov


GROUPS = {"front": ("N0", "N1", "N2"), "back": ("N5", "N6", "N7")}
CASES = [
    Limits(),
    Limits(max_weight=0.2),
    Limits(linear=("front <= 0.3",), groups=GROUPS),
    Limits(linear=("N0 >= 2 * N1", "back >= 0.35"), groups=GROUPS),
    Limits(min_weight=-0.2, max_weight=0.5, linear=("N3 + N4 == 0.25",)),
]


@pytest.mark.parametrize("limits", CASES)
@pytest.mark.parametrize("objective", ["track-hrp", "min-variance"])
def test_the_convex_solution_matches_cvxpy(limits: Limits, objective: str) -> None:
    cp = pytest.importorskip("cvxpy")
    np = pytest.importorskip("numpy")
    names, cov = book(8, seed=11)
    ours = allocate(names, cov, limits, objective=objective)  # type: ignore[arg-type]
    assert ours.status == "optimal"
    # Both solvers see the covariance scaled to a unit mean diagonal. Unscaled, hourly variances
    # put the optimum near 1e-7, under CLARABEL's absolute tolerance, and it stops early.
    sigma = np.array(cov) / np.trace(np.array(cov)) * 8
    target = np.zeros(8) if objective == "min-variance" else np.array(
        [ours.target[k] for k in names])
    w = cp.Variable(8)
    cons = [cp.sum(w) == limits.budget, w >= limits.min_weight, w <= limits.max_weight]
    for text in limits.linear:
        lim = parse_limit(text, names, limits.groups)
        expr = np.array(lim.coeffs) @ w
        cons.append(expr == lim.bound if lim.equality else expr <= lim.bound)
    problem = cp.Problem(cp.Minimize(cp.quad_form(w - target, cp.psd_wrap(sigma))), cons)
    problem.solve(solver="CLARABEL")
    assert problem.status == "optimal"
    mine = [ours.weights[k] for k in names]
    ours_value = float((np.array(mine) - target) @ sigma @ (np.array(mine) - target))
    # The objective is flat along near-collinear names, so two exact solvers can report weights
    # apart by ~1e-3 at the same value: compare the value, and check our weights meet every limit.
    assert ours_value <= problem.value * (1 + 1e-6) + 1e-12
    assert ours_value >= problem.value - 1e-7  # CLARABEL stops within ~1e-8 of its optimum
    assert abs(sum(mine) - limits.budget) < 1e-9
    assert all(limits.min_weight - 1e-9 <= v <= limits.max_weight + 1e-9 for v in mine)
    for text in limits.linear:
        lim = parse_limit(text, names, limits.groups)
        excess = sum(c * v for c, v in zip(lim.coeffs, mine, strict=True)) - lim.bound
        assert (abs(excess) if lim.equality else excess) < 1e-9


@pytest.mark.parametrize("cap", [2, 3, 5])
def test_the_count_cap_is_the_exact_optimum(cap: int) -> None:
    names, cov = book(9, seed=5)
    limits = Limits(max_names=cap, max_weight=0.6)
    result = allocate(names, cov, limits)
    assert result.status == "optimal" and result.holdings <= cap
    scale = sum(cov[i][i] for i in range(9)) / 9
    p = _Problem(cov=[[v / scale for v in row] for row in cov],
                 target=[result.target[k] for k in names], lower=0.0, upper=0.6, budget=1.0,
                 limits=(), risk_share=None)
    brute = min(_tracking(p, w) for combo in itertools.combinations(range(9), cap)
                if (w := _convex(p, list(combo))) is not None)
    found = _tracking(p, [result.weights[k] for k in names])
    assert found == pytest.approx(brute, rel=1e-9, abs=1e-15)


def test_a_group_count_cap_holds() -> None:
    names, cov = book(8, seed=2)
    result = allocate(names, cov, Limits(groups=GROUPS, max_names_per_group={"front": 1}))
    held = {k for k, v in result.weights.items() if v > 1e-9}
    assert len(held & set(GROUPS["front"])) <= 1


def test_the_risk_share_cap_holds_and_is_labelled_local() -> None:
    names, cov = book(8, seed=3)
    free = allocate(names, cov, Limits())
    assert max(free.risk_shares.values()) > 0.2
    capped = allocate(names, cov, Limits(max_risk_share=0.2))
    assert capped.status == "locally optimal"
    assert max(capped.risk_shares.values()) <= 0.2 + 1e-7
    assert abs(sum(capped.risk_shares.values()) - 1) < 1e-9
    assert "not a proven optimum" in capped.reason


def test_conflicting_limits_name_what_to_relax_and_return_no_book() -> None:
    names, cov = book(6, seed=4)
    result = allocate(names, cov, Limits(linear=("N0 >= 0.6", "N0 + N1 <= 0.5", "N2 >= 0.1")))
    assert result.status == "infeasible" and result.weights == {}
    assert set(result.conflict) == {"N0 >= 0.6", "N0 + N1 <= 0.5"}
    too_tight = allocate(names, cov, Limits(max_weight=0.1))
    assert too_tight.status == "infeasible" and "short of 100%" in too_tight.reason


def test_the_limit_language() -> None:
    names = ["AAPLUSDT", "MSFTUSDT", "NVDAUSDT"]
    groups = {"semis": ("NVDAUSDT",)}
    lim = parse_limit("aaplusdt >= 0.5 * MSFTUSDT", names, groups)
    assert lim.coeffs == (-1.0, 0.5, 0.0) and lim.bound == 0.0 and not lim.equality
    assert parse_limit("semis <= 30%", names, groups).bound == pytest.approx(0.3)
    both = parse_limit("AAPLUSDT + 2*NVDAUSDT - 0.1 == MSFTUSDT", names, groups)
    assert both.coeffs == (1.0, -1.0, 2.0) and both.bound == pytest.approx(0.1) and both.equality
    for bad in ("TSLAUSDT <= 0.2", "AAPLUSDT < 0.2", "AAPLUSDT <= 0.2 <= 0.3",
                "AAPLUSDT * MSFTUSDT <= 0.1", "0.2 <= 0.3", "AAPLUSDT +"):
        with pytest.raises(ConstraintError):
            parse_limit(bad, names, groups)


def test_undefined_groups_refuse_rather_than_drop_the_limit() -> None:
    names, cov = book(5, seed=1)
    with pytest.raises(ConstraintError, match="neither a holding nor a group"):
        allocate(names, cov, Limits(linear=("tech <= 0.4",)))
    with pytest.raises(ConstraintError, match="not defined"):
        allocate(names, cov, Limits(max_names_per_group={"tech": 1}))
