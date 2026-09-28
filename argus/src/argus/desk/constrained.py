"""The book a trader's own limits allow: the HRP book, bent as little as possible to fit.

`desk/allocation.py` proposes the hierarchical-risk-parity book and prices the move to it. It cannot
take an instruction. A portfolio manager does not ask for "the HRP book"; they ask for a book with
*at most six names*, *no more than 30% in any one*, *semis under half*, *no single holding carrying
more than a quarter of the risk*. Before this module the desk had no way to hear any of that: the
only limit anywhere was `risk/sizing.py`'s clamp on a single trade, applied after the fact.

**What is solved.** Minimise the tracking variance ``(w - t)' S (w - t)`` from the HRP target ``t``
(or, with ``objective="min-variance"``, the variance ``w' S w``) subject to

* a weight range per name and a budget ``sum(w) = B``;
* linear limits written the way skfolio writes them — ``"NVDAUSDT + AMDUSDT <= 0.4"``,
  ``"semis <= 0.5"``, ``"AAPLUSDT >= 0.5 * MSFTUSDT"`` — where a group name stands for the sum of
  its members (skfolio `utils/equations.py:48-130`, grammar `:557-771`, BSD-3-Clause);
* a cap on the number of names held, overall and per group (skfolio
  `optimization/convex/_base.py:597-598, 971-1020`, `cardinality`/`group_cardinalities`);
* a cap on each name's share of portfolio risk, ``w_i (S w)_i <= s * w' S w`` (Riskfolio-Lib
  `Portfolio.py:3031-3038`, BSD-3-Clause): the Euler contributions `desk/portfolio.decompose`
  already measures, now enforced rather than only reported.

Tracking the HRP book rather than minimising variance is a deliberate choice, not a default copied
from either library: the module docstring of `desk/allocation.py` explains why this desk does not
trust an optimiser's reading of a near-singular covariance, and the least disturbance to HRP that
satisfies the stated limits keeps that stance while obeying the trader.

**How, and why not a library.** The shipped package depends on pydantic and python-dateutil only,
so there is no CVXPY here. The convex part is a quadratic program, solved exactly by Goldfarb and
Idnani's dual active-set method (:func:`quadprog`); the objective and the feasible set are convex,
so the minimum is the global one — and `eval/constrained_comparison.py` checks exactly that against
skfolio's `MeanRisk` run through CVXPY on the same inputs. A first version used an augmented
Lagrangian with projected-gradient inner solves; on a twelve-name book it had not finished one
unconstrained solve in two minutes, and was replaced.

Two things differ from the libraries on purpose, and both are corrections:

* **An unknown name refuses.** skfolio's `equations_to_matrix` defaults to
  ``raise_if_group_missing=False`` and drops a limit that names a group it cannot find, with a
  warning. A trader's limit that silently disappears is worse than no limit, so here it raises.
* **"At most N names" is solved exactly.** skfolio compiles cardinality to a mixed-integer program,
  which needs a MIQP-capable solver (none ships with CVXPY's defaults). Here a branch and bound
  over which names are held does the same job: each node's quadratic program without the count
  caps bounds every book below it, so the book returned is provably the best, not a relaxation's
  rounding. Past :data:`MAX_NODES` nodes it refuses rather than guess.

The risk-share cap is not convex. With it, the result is a **local** minimum from several starts,
labelled ``"locally optimal"``, never ``"optimal"``. Every limit is re-checked on the final weights
and a result that fails its own check is refused, not returned.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from argus.desk.allocation import AllocationError, hrp_weights, portfolio_variance

MAX_NAMES = 60
"""A pure-Python dense solve; past this the book is bigger than anything the desk holds."""
FEASIBILITY_TOL = 1e-7
"""A limit counts as met when its excess is at most this, in weight units (or risk-share units)."""
BINDING_TOL = 1e-5

Objective = Literal["track-hrp", "min-variance"]
Status = Literal["optimal", "locally optimal", "infeasible"]


class ConstraintError(ValueError):
    """A limit that cannot be read, or names something the book does not contain."""


@dataclass(frozen=True, slots=True)
class Limit:
    """One linear limit, normalised to ``coeffs . w <= bound`` (or ``==`` when ``equality``)."""

    text: str
    coeffs: tuple[float, ...]
    bound: float
    equality: bool = False


@dataclass(frozen=True, slots=True)
class Limits:
    min_weight: float = 0.0
    max_weight: float = 1.0
    budget: float = 1.0
    linear: tuple[str, ...] = ()
    groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    """Group name to its member names."""
    max_names: int | None = None
    max_names_per_group: Mapping[str, int] = field(default_factory=dict)
    max_risk_share: float | None = None


@dataclass(frozen=True, slots=True)
class ConstrainedBook:
    status: Status
    weights: dict[str, float]
    """Empty when infeasible: a book that breaks a stated limit is never returned."""
    target: dict[str, float]
    objective: Objective
    variance: float | None
    target_variance: float
    tracking_volatility: float | None
    """Volatility of the difference from the HRP book: what the limits cost, in risk."""
    risk_shares: dict[str, float]
    binding: tuple[str, ...]
    """The limits the solution presses against: the ones that shaped the book."""
    conflict: tuple[str, ...]
    """When infeasible, the limits that cannot hold together."""
    supports_searched: int
    """Quadratic programs solved: 1 without a count cap, the branch-and-bound nodes with one."""
    reason: str

    @property
    def holdings(self) -> int:
        return sum(1 for v in self.weights.values() if abs(v) > 1e-9)


# --- the limit language -----------------------------------------------------------------------

_TOKEN = re.compile(
    r"\s*(?:(?P<num>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?)|(?P<op><=|>=|==|=|[+\-*])"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_.\-]*))"
)


def _tokens(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    position = 0
    stripped = text.rstrip()
    while position < len(stripped):
        match = _TOKEN.match(stripped, position)
        if match is None or match.end() == position:
            raise ConstraintError(f"cannot read {text!r} at “{stripped[position:].strip()}”")
        kind = match.lastgroup
        assert kind is not None
        out.append((kind, match.group(kind)))
        position = match.end()
    return out


def _number(token: str) -> float:
    return float(token[:-1]) / 100 if token.endswith("%") else float(token)


def parse_limit(text: str, names: Sequence[str],
                groups: Mapping[str, Sequence[str]] | None = None) -> Limit:
    """Read one limit in skfolio's grammar: sums of ``[number *] name`` terms and constants either
    side of one of ``<=``, ``>=``, ``==``. A name is a holding or a group (the sum of its members);
    names match without regard to case. ``30%`` reads as ``0.3``."""
    groups = groups or {}
    lookup: dict[str, list[float]] = {}
    index = {n: i for i, n in enumerate(names)}
    for n, i in index.items():
        row = [0.0] * len(names)
        row[i] = 1.0
        lookup[n.upper()] = row
    for g, members in groups.items():
        if g.upper() in lookup:
            raise ConstraintError(f"group {g!r} has the same name as a holding")
        missing = [m for m in members if m not in index]
        if missing:
            raise ConstraintError(f"group {g!r} names {', '.join(missing)}, not in the book")
        lookup[g.upper()] = [1.0 if n in members else 0.0 for n in names]

    tokens = _tokens(text)
    comparisons = [i for i, (k, v) in enumerate(tokens)
                   if k == "op" and v in ("<=", ">=", "==", "=")]
    if len(comparisons) != 1:
        raise ConstraintError(f"{text!r} needs exactly one of <=, >= or ==")
    split = comparisons[0]
    operator = tokens[split][1]

    def side(part: list[tuple[str, str]]) -> tuple[list[float], float]:
        vector = [0.0] * len(names)
        constant = 0.0
        position = 0
        if not part:
            raise ConstraintError(f"{text!r} has an empty side")
        while position < len(part):
            sign = 1.0
            while position < len(part) and part[position] in (("op", "+"), ("op", "-")):
                sign *= -1.0 if part[position][1] == "-" else 1.0
                position += 1
            if position >= len(part):
                raise ConstraintError(f"{text!r} ends with an operator")
            factor = 1.0
            row: list[float] | None = None
            kind, value = part[position]
            if kind == "num":
                factor = _number(value)
            elif kind == "name":
                if value.upper() not in lookup:
                    raise ConstraintError(
                        f"{text!r} names {value!r}, which is neither a holding nor a group")
                row = lookup[value.upper()]
            else:
                raise ConstraintError(f"{text!r}: “{value}” is out of place")
            position += 1
            if position < len(part) and part[position] == ("op", "*"):
                position += 1
                if position >= len(part):
                    raise ConstraintError(f"{text!r} ends with *")
                kind2, value2 = part[position]
                if kind2 == "name" and row is None:
                    if value2.upper() not in lookup:
                        raise ConstraintError(
                            f"{text!r} names {value2!r}, which is neither a holding nor a group")
                    row = lookup[value2.upper()]
                elif kind2 == "num" and row is not None:
                    factor = _number(value2)
                else:
                    raise ConstraintError(f"{text!r}: only number * name is linear")
                position += 1
            if row is None:
                constant += sign * factor
            else:
                vector = [v + sign * factor * r for v, r in zip(vector, row, strict=True)]
            if position < len(part) and part[position] not in (("op", "+"), ("op", "-")):
                raise ConstraintError(f"{text!r}: “{part[position][1]}” is out of place")
        return vector, constant

    left, left_c = side(tokens[:split])
    right, right_c = side(tokens[split + 1:])
    if operator == ">=":
        coeffs = [b - a for a, b in zip(left, right, strict=True)]
        bound = left_c - right_c
    else:
        coeffs = [a - b for a, b in zip(left, right, strict=True)]
        bound = right_c - left_c
    if not any(abs(c) > 0 for c in coeffs):
        raise ConstraintError(f"{text!r} does not involve any holding")
    return Limit(text=text.strip(), coeffs=tuple(coeffs), bound=bound,
                 equality=operator in ("==", "="))


# --- the solver ------------------------------------------------------------------------------

RIDGE = 1e-9
"""Added to the scaled covariance's diagonal (mean diagonal 1) so the dual method's Hessian is
positive definite. It moves an objective by at most 1e-9 times the squared weight norm."""
MAX_NODES = 20000
"""Branch-and-bound nodes before an exact count-limited search refuses rather than guesses."""


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _matvec(m: Sequence[Sequence[float]], v: Sequence[float]) -> list[float]:
    return [_dot(row, v) for row in m]


def _linsolve(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting; ``None`` when singular."""
    size = len(rhs)
    a = [[*row, rhs[i]] for i, row in enumerate(matrix)]
    scale = max((abs(v) for row in matrix for v in row), default=0.0) or 1.0
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) <= 1e-13 * scale:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        head = a[col]
        inv = 1.0 / head[col]
        for r in range(col + 1, size):
            factor = a[r][col] * inv
            if factor:
                row = a[r]
                for c in range(col, size + 1):
                    row[c] -= factor * head[c]
    out = [0.0] * size
    for r in range(size - 1, -1, -1):
        out[r] = (a[r][size] - sum(a[r][c] * out[c] for c in range(r + 1, size))) / a[r][r]
    return out


@dataclass(frozen=True, slots=True)
class _Row:
    normal: tuple[float, ...]
    bound: float
    equality: bool = False
    """``normal . x >= bound``, or ``==`` when ``equality``."""


def quadprog(g: list[list[float]], a: list[float], rows: Sequence[_Row],
             tol: float = 1e-11) -> list[float] | None:
    """Minimise ``x'Gx/2 + a'x`` subject to ``rows``, or ``None`` when no point satisfies them.

    Goldfarb and Idnani's dual active-set method (Math. Programming 27, 1983, §3): start at the
    unconstrained minimum, add the most violated constraint, and move along the direction that
    raises it while holding the active ones, dropping any active constraint whose multiplier would
    turn negative. Finite, exact up to rounding, and needs no feasible starting point — which the
    count-limited search depends on, since most of its nodes start infeasible. The direction is
    solved from the full KKT system each step rather than by the paper's factorisation updates:
    slower per step, simpler to verify, and the books here have at most :data:`MAX_NAMES` names.
    """
    n = len(a)
    x = _linsolve([row[:] for row in g], [-v for v in a])
    if x is None:
        raise ConstraintError("the covariance is singular even after the ridge")
    active: list[int] = []
    duals: list[float] = []
    limit = 50 * (n + len(rows)) + 100
    for _ in range(limit):
        chosen: int | None = None
        worst = -tol
        for j, row in enumerate(rows):
            if j in active:
                continue
            slack = _dot(row.normal, x) - row.bound
            if row.equality and abs(slack) > tol:
                chosen = j
                break
            if not row.equality and slack < worst:
                chosen, worst = j, slack
        if chosen is None:
            return x
        normal = list(rows[chosen].normal)
        bound = rows[chosen].bound
        if rows[chosen].equality and _dot(normal, x) - bound > 0:
            normal, bound = [-v for v in normal], -bound
        added = 0.0
        for _ in range(limit):
            q = len(active)
            kkt = [g[i][:] + [rows[j].normal[i] for j in active] for i in range(n)]
            kkt += [list(rows[j].normal) + [0.0] * q for j in active]
            solved = _linsolve(kkt, normal + [0.0] * q)
            if solved is None:
                return None
            z, r = solved[:n], solved[n:]
            partial, drop = math.inf, -1
            for idx, j in enumerate(active):
                if not rows[j].equality and r[idx] > 1e-14 and duals[idx] / r[idx] < partial:
                    partial, drop = duals[idx] / r[idx], idx
            curvature = _dot(z, normal)
            if math.sqrt(_dot(z, z)) <= 1e-13 or curvature <= 1e-18:
                if drop < 0:
                    return None  # the new constraint cannot be met with the active ones
                duals = [d - partial * ri for d, ri in zip(duals, r, strict=True)]
                added += partial
                del active[drop], duals[drop]
                continue
            full = -(_dot(normal, x) - bound) / curvature
            step = min(partial, full)
            x = [xi + step * zi for xi, zi in zip(x, z, strict=True)]
            duals = [d - step * ri for d, ri in zip(duals, r, strict=True)]
            added += step
            if full <= partial:
                active.append(chosen)
                duals.append(added)
                break
            del active[drop], duals[drop]
        else:
            return None
    return None


@dataclass(frozen=True, slots=True)
class _Problem:
    cov: list[list[float]]
    """Scaled so its mean diagonal is one: tolerances then mean the same on every book."""
    target: list[float]
    lower: float
    upper: float
    budget: float
    limits: tuple[Limit, ...]
    risk_share: float | None


def _rows(p: _Problem, index: Sequence[int]) -> list[_Row]:
    rows = [_Row(tuple(1.0 for _ in index), p.budget, True)]
    k = len(index)
    for pos in range(k):
        unit = tuple(1.0 if q == pos else 0.0 for q in range(k))
        rows.append(_Row(unit, p.lower))
        rows.append(_Row(tuple(-u for u in unit), -p.upper))
    for limit in p.limits:
        coeffs = tuple(limit.coeffs[i] for i in index)
        if limit.equality:
            rows.append(_Row(coeffs, limit.bound, True))
        else:
            rows.append(_Row(tuple(-c for c in coeffs), -limit.bound))
    return rows


def _expand(n: int, index: Sequence[int], x: Sequence[float]) -> list[float]:
    w = [0.0] * n
    for i, v in zip(index, x, strict=True):
        w[i] = v
    return w


def _tracking(p: _Problem, w: Sequence[float]) -> float:
    d = [a - b for a, b in zip(w, p.target, strict=True)]
    return _dot(d, _matvec(p.cov, d))


def _convex(p: _Problem, index: Sequence[int]) -> list[float] | None:
    """The global minimum over the names in ``index`` (the rest held at zero), or ``None``."""
    if not index:
        return None
    k = len(index)
    if not k * p.lower - 1e-12 <= p.budget <= k * p.upper + 1e-12:
        return None
    g = [[2 * (p.cov[i][j] + (RIDGE if i == j else 0.0)) for j in index] for i in index]
    st = _matvec(p.cov, p.target)
    a = [-2 * st[i] for i in index]
    x = quadprog(g, a, _rows(p, index))
    return None if x is None else _expand(len(p.target), index, x)


def _shares(p: _Problem, w: Sequence[float]) -> list[float]:
    sw = _matvec(p.cov, w)
    total = _dot(w, sw)
    return [w[i] * sw[i] / total for i in range(len(w))] if total > 0 else [0.0] * len(w)


def _risk_excess(p: _Problem, w: Sequence[float]) -> float:
    assert p.risk_share is not None
    return max(0.0, max(_shares(p, w)) - p.risk_share)


def _with_risk_cap(p: _Problem, index: Sequence[int], start: Sequence[float]
                   ) -> list[float] | None:
    """Sequential linear-constraint QP with a trust region for ``w_i (Sw)_i <= s w'Sw``.

    Each step linearises the risk constraints at the current point and solves the QP exactly with
    the other limits intact; a step is kept only if it lowers ``tracking + mu * excess``. Local:
    the constraint is not convex, which is why the caller labels the result so.
    """
    assert p.risk_share is not None
    s = p.risk_share
    n = len(p.target)
    k = len(index)
    g = [[2 * (p.cov[i][j] + (RIDGE if i == j else 0.0)) for j in index] for i in index]
    st = _matvec(p.cov, p.target)
    a = [-2 * st[i] for i in index]
    base_rows = _rows(p, index)
    mu = 1e3

    def merit(w: Sequence[float]) -> float:
        sw = _matvec(p.cov, w)
        total = _dot(w, sw)
        excess = sum(max(0.0, w[i] * sw[i] - s * total) for i in index)
        return _tracking(p, w) + mu * excess

    x = [start[i] for i in index]
    radius = 0.25
    current = merit(_expand(n, index, x))
    for _ in range(300):
        w = _expand(n, index, x)
        sw = _matvec(p.cov, w)
        total = _dot(w, sw)
        rows = list(base_rows)
        for pos, i in enumerate(index):
            value = w[i] * sw[i] - s * total
            grad = [w[i] * p.cov[i][j] - 2 * s * sw[j] for j in index]
            grad[pos] += sw[i]
            # value + grad.(x' - x) <= 0   ->   -grad.x' >= value - grad.x
            rows.append(_Row(tuple(-v for v in grad), value - _dot(grad, x)))
        for pos in range(k):
            unit = tuple(1.0 if q == pos else 0.0 for q in range(k))
            rows.append(_Row(unit, x[pos] - radius))
            rows.append(_Row(tuple(-u for u in unit), -(x[pos] + radius)))
        trial = quadprog(g, a, rows)
        if trial is None:
            radius *= 0.25
            if radius < 1e-9:
                break
            continue
        value = merit(_expand(n, index, trial))
        moved = math.sqrt(sum((t - c) ** 2 for t, c in zip(trial, x, strict=True)))
        if value < current - 1e-15:
            x, current = trial, value
            radius = min(1.0, radius * 2)
            if moved < 1e-10:
                break
        else:
            radius *= 0.25
            if radius < 1e-9:
                break
    w = _expand(n, index, x)
    return w if _risk_excess(p, w) <= FEASIBILITY_TOL else None


def _solve_on(p: _Problem, index: Sequence[int]) -> list[float] | None:
    convex = _convex(p, index)
    if convex is None or p.risk_share is None:
        return convex
    n = len(p.target)
    starts = [convex, _expand(n, index, [p.budget / len(index)] * len(index))]
    best: list[float] | None = None
    for start in starts:
        w = _with_risk_cap(p, index, start)
        if w is not None and not _violated(p, w, FEASIBILITY_TOL) and (
                best is None or _tracking(p, w) < _tracking(p, best)):
            best = w
    return best


def _violated(p: _Problem, w: Sequence[float], tol: float) -> list[str]:
    out: list[str] = []
    if abs(sum(w) - p.budget) > tol:
        out.append(f"weights summing to {p.budget:g}")
    if any(v < p.lower - tol or v > p.upper + tol for v in w):
        out.append(f"weights between {p.lower:g} and {p.upper:g}")
    for limit in p.limits:
        g = _dot(limit.coeffs, w) - limit.bound
        if g > tol or (limit.equality and -g > tol):
            out.append(limit.text)
    if p.risk_share is not None and _risk_excess(p, w) > tol:
        out.append(f"no holding above {p.risk_share:.0%} of risk")
    return out


@dataclass(frozen=True, slots=True)
class _Counts:
    max_names: int
    groups: tuple[tuple[frozenset[int], int], ...]

    def allows(self, chosen: set[int]) -> bool:
        return len(chosen) <= self.max_names and all(
            len(chosen & members) <= cap for members, cap in self.groups)


def _search(p: _Problem, counts: _Counts) -> tuple[list[float] | None, int]:
    """Exact minimum under the count caps by branch and bound: a node fixes some names out and
    some in; its convex minimum without the counts bounds everything below it, and a node whose
    minimum already respects the counts is solved."""
    n = len(p.target)
    best: tuple[float, list[float]] | None = None
    nodes = 0
    stack: list[tuple[frozenset[int], frozenset[int]]] = [(frozenset(), frozenset())]
    while stack:
        out, into = stack.pop()
        nodes += 1
        if nodes > MAX_NODES:
            raise ConstraintError(
                f"the exact search for a count-limited book passed {MAX_NODES} nodes; it stops "
                "there rather than return a book it cannot show is the best")
        if not counts.allows(set(into)):
            continue
        index = [i for i in range(n) if i not in out]
        relaxed = _convex(p, index)
        if relaxed is None:
            continue
        bound = _tracking(p, relaxed)
        if best is not None and bound >= best[0] - 1e-13:
            continue
        held = {i for i in index if abs(relaxed[i]) > 1e-9}
        if counts.allows(held | into):
            w = relaxed if p.risk_share is None else _solve_on(p, sorted(held | into))
            if w is not None:
                value = _tracking(p, w)
                if best is None or value < best[0]:
                    best = (value, w)
            continue
        free = [i for i in held if i not in into]
        if not free:
            continue
        pick = min(free, key=lambda i: abs(relaxed[i]))
        stack.append((out, into | {pick}))
        stack.append((out | {pick}, into))  # popped first: dropping names finds books quickly
    return (None if best is None else best[1]), nodes


def allocate(names: Sequence[str], cov: Sequence[Sequence[float]], limits: Limits, *,
             objective: Objective = "track-hrp",
             target: Mapping[str, float] | None = None) -> ConstrainedBook:
    """The book closest to ``target`` (the HRP book unless given) that meets every limit."""
    n = len(names)
    if n == 0 or n > MAX_NAMES:
        raise ConstraintError(f"a book of {n} names is outside 1..{MAX_NAMES}")
    if len(set(names)) != n or len(cov) != n or any(len(r) != n for r in cov):
        raise ConstraintError("names and covariance do not line up")
    if not limits.min_weight <= limits.max_weight:
        raise ConstraintError("the smallest weight allowed exceeds the largest")
    if limits.max_risk_share is not None and not 0 < limits.max_risk_share <= 1:
        raise ConstraintError("a risk-share cap must be above 0 and at most 100%")
    if limits.max_names is not None and limits.max_names < 1:
        raise ConstraintError("a book must hold at least one name")
    for group in limits.max_names_per_group:
        if group not in limits.groups:
            raise ConstraintError(f"a name count is set for group {group!r}, which is not defined")
    parsed = tuple(parse_limit(t, names, limits.groups) for t in limits.linear)

    if target is None:
        try:
            base = hrp_weights(names, cov) if n >= 3 else dict.fromkeys(names, 1.0 / n)
        except AllocationError as exc:
            raise ConstraintError(f"the HRP book could not be built: {exc}") from exc
        base = {k: v * limits.budget for k, v in base.items()}
    else:
        base = {k: float(target.get(k, 0.0)) for k in names}
    scale = sum(cov[i][i] for i in range(n)) / n
    if not scale > 0:
        raise ConstraintError("the covariance has no variance on its diagonal")
    p = _Problem(cov=[[cov[i][j] / scale for j in range(n)] for i in range(n)],
                 target=[0.0] * n if objective == "min-variance" else [base[k] for k in names],
                 lower=limits.min_weight, upper=limits.max_weight, budget=limits.budget,
                 limits=parsed, risk_share=limits.max_risk_share)

    counted = limits.max_names is not None or bool(limits.max_names_per_group)
    most = min(n, limits.max_names or n)
    reach = most * limits.max_weight
    if limits.min_weight >= 0 and reach < limits.budget - 1e-12:
        said = (f"{most} names at no more than {limits.max_weight:.0%} each add up "
                f"to {reach:.0%}, short of {limits.budget:.0%}")
        return ConstrainedBook(status="infeasible", weights={}, target=base, objective=objective,
                               variance=None,
                               target_variance=portfolio_variance(base, names, cov),
                               tracking_volatility=None, risk_shares={}, binding=(),
                               conflict=(said,), supports_searched=0,
                               reason=f"no set of weights meets every limit: {said}")
    if counted:
        position = {k: i for i, k in enumerate(names)}
        counts = _Counts(
            max_names=min(limits.max_names or n, n),
            groups=tuple((frozenset(position[m] for m in limits.groups[g]), cap)
                         for g, cap in limits.max_names_per_group.items()))
        solution, searched = _search(p, counts)
    else:
        solution, searched = _solve_on(p, list(range(n))), 1

    target_variance = portfolio_variance(base, names, cov)
    if solution is None:
        conflict = _conflict(p, n)
        why = "no set of weights meets every limit"
        if not n * p.lower - 1e-12 <= p.budget <= n * p.upper + 1e-12:
            why += ": " + "; ".join(conflict)
        elif conflict:
            why += "; relaxing any one of these would allow a book: " + "; ".join(conflict)
        if counted:
            why += f" (exact search over {searched} nodes)"
        return ConstrainedBook(status="infeasible", weights={}, target=base, objective=objective,
                               variance=None, target_variance=target_variance,
                               tracking_volatility=None, risk_shares={}, binding=(),
                               conflict=tuple(conflict), supports_searched=searched, reason=why)
    broken = _violated(p, solution, FEASIBILITY_TOL * 10)
    if broken:  # pragma: no cover - the solver's own guarantee; kept as the last line of defence
        raise ConstraintError("the solution failed its own check on: " + "; ".join(broken))

    w = [0.0 if abs(v) < 1e-12 else v for v in solution]
    weights = dict(zip(names, w, strict=True))
    variance = portfolio_variance(weights, names, cov)
    diff = {k: weights[k] - base[k] for k in names}
    tracking = math.sqrt(max(0.0, portfolio_variance(diff, names, cov)))
    shares = dict(zip(names, _shares(p, w), strict=True)) if variance > 0 else {}
    binding = [lim.text for lim in parsed
               if lim.equality or abs(_dot(lim.coeffs, w) - lim.bound) <= BINDING_TOL]
    if limits.max_weight < limits.budget and any(
            abs(v - limits.max_weight) <= BINDING_TOL for v in w):
        binding.append(f"at most {limits.max_weight:.0%} in any one name")
    if limits.max_names is not None and sum(1 for v in w if abs(v) > 1e-9) >= limits.max_names:
        binding.append(f"at most {limits.max_names} names")
    if (limits.max_risk_share is not None and shares
            and max(shares.values()) >= limits.max_risk_share - BINDING_TOL):
        binding.append(f"no holding above {limits.max_risk_share:.0%} of risk")
    what = "the HRP book" if objective == "track-hrp" else "the lowest variance"
    reason = f"closest to {what} that meets every limit"
    if counted:
        reason += f"; exact search over {searched} nodes"
    if limits.max_risk_share is not None:
        reason += ("; the risk-share cap is not convex, so this is the best of several local "
                   "solutions, not a proven optimum")
    status: Status = "optimal" if limits.max_risk_share is None else "locally optimal"
    return ConstrainedBook(status=status, weights=weights, target=base, objective=objective,
                           variance=variance, target_variance=target_variance,
                           tracking_volatility=tracking, risk_shares=shares,
                           binding=tuple(binding), conflict=(), supports_searched=searched,
                           reason=reason)


def _conflict(p: _Problem, n: int) -> list[str]:
    """The limits whose removal alone would let the linear limits hold: the ones to relax."""
    if not n * p.lower - 1e-12 <= p.budget <= n * p.upper + 1e-12:
        return [f"weights between {p.lower:g} and {p.upper:g} cannot sum to {p.budget:g}"]
    out: list[str] = []
    for drop in range(len(p.limits)):
        rest = p.limits[:drop] + p.limits[drop + 1:]
        trial = _Problem(cov=p.cov, target=p.target, lower=p.lower, upper=p.upper,
                         budget=p.budget, limits=rest, risk_share=None)
        if _convex(trial, list(range(n))) is not None:
            out.append(p.limits[drop].text)
    return out


__all__ = ["MAX_NODES", "ConstrainedBook", "ConstraintError", "Limit", "Limits", "allocate",
           "parse_limit", "quadprog"]
