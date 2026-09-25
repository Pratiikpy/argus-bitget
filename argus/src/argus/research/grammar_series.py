"""Evaluate a factor over a whole series at once: each node once, shared subtrees once.

**The lesson taken from Polars** (``pola-rs/polars``, MIT; run head to head in
:mod:`argus.eval.general_grammar_comparison`). :meth:`argus.research.grammar.Expr.evaluate` answers
one bar at a time, and a ``Window`` re-evaluates its whole operand subtree for every bar of its
lookback. Nested windows therefore cost the *product* of their lookbacks per bar — the reason the
search needs ``searchoff.MAX_NODE_COST`` and the reason ``grammar.validate`` now carries a cost
bound. A columnar engine evaluates each expression node once over the full column and hands the
column upward, so nesting costs the *sum* of the lookbacks, and a subtree that appears twice is
computed once (Polars' common-subexpression elimination). Same trees, same numbers, different
evaluation order.

**Bit-identical, by construction rather than by re-implementation.** Rolling sums maintained
incrementally would be faster and would differ from the per-bar path in the last floating-point
place, which a duplicate-detection and trial-accounting system cannot tolerate. So this module does
not re-implement a single operator. For every node it builds the *real* grammar node with its
children replaced by precomputed columns (:class:`_Column`) and calls that node's own
``evaluate``. The arithmetic is therefore the grammar's own, operation for operation;
``tests/test_grammar_series.py`` checks equality with ``==`` on every bar, not a tolerance.

**What is deliberately not taken from Polars.** Vectorised SIMD kernels and incremental rolling
aggregates — the source of most of Polars' raw speed — because of the bit-identity requirement
above. The comparison reports the remaining speed gap rather than hiding it.

**One semantic difference, stated.** Per bar, ``IfElse`` evaluates only the branch it takes. Here a
branch is evaluated over the whole series if it is taken at *any* bar. The values are identical;
the only observable difference is that a branch which raises (a cross-sectional node with no
panel context) raises here if it is taken anywhere, exactly as a full per-bar sweep would.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from argus.research.grammar import (
    BinOp,
    Corr,
    CrossContext,
    Delay,
    Expr,
    IfElse,
    Kind,
    Signal,
    UnOp,
    Window,
)


@dataclass(frozen=True, slots=True, eq=False)
class _Column(Expr):
    """A precomputed child: the real node above it reads ``values[i]`` where it would have
    evaluated the subtree at bar ``i``. Carries the child's kind so the parent's own construction
    checks (a condition must be boolean, and so on) run unchanged."""

    values: tuple[float, ...]
    column_kind: Kind

    @property
    def kind(self) -> Kind:
        return self.column_kind

    def evaluate(self, bars: Sequence[Any], i: int, ctx: CrossContext | None = None) -> float:
        return self.values[i]

    def canonical(self) -> str:
        return "column()"


def evaluate_series(
    expr: Expr, bars: Sequence[Any], ctx: CrossContext | None = None
) -> list[float]:
    """``[expr.evaluate(bars, i, ctx) for i in range(len(bars))]``, computed column by column."""
    memo: dict[Any, tuple[float, ...]] = {}
    return list(_series(expr, bars, ctx, memo))


def series_cost(expr: Expr) -> int:
    """Node evaluations per bar under this evaluator: one per distinct node, times its lookback for
    the windowed ones. Additive where :func:`argus.research.grammar.cost` is multiplicative."""
    seen: set[Any] = set()
    return _series_cost(expr, seen)


def _key(expr: Expr) -> Any:
    # Structural identity (frozen dataclass equality), so a repeated subtree is computed once. Not
    # `canonical()`: that rounds constants to six digits, and two constants it merges are still
    # two different columns.
    try:
        hash(expr)
    except TypeError:
        return ("id", id(expr))
    return expr


def _series_cost(expr: Expr, seen: set[Any]) -> int:
    key = _key(expr)
    if key in seen:
        return 0
    seen.add(key)
    own = expr.lookback if isinstance(expr, Window | Corr) else 1
    return own + sum(_series_cost(c, seen) for c in expr.children)


def _series(
    expr: Expr, bars: Sequence[Any], ctx: CrossContext | None,
    memo: dict[Any, tuple[float, ...]],
) -> tuple[float, ...]:
    key = _key(expr)
    cached = memo.get(key)
    if cached is not None:
        return cached
    n = len(bars)
    node = _rebuild(expr, bars, ctx, memo)
    out = tuple(node.evaluate(bars, i, ctx) for i in range(n))
    memo[key] = out
    return out


def _col(child: Expr, bars: Sequence[Any], ctx: CrossContext | None,
         memo: dict[Any, tuple[float, ...]]) -> _Column:
    return _Column(_series(child, bars, ctx, memo), child.kind)


def _rebuild(expr: Expr, bars: Sequence[Any], ctx: CrossContext | None,
             memo: dict[Any, tuple[float, ...]]) -> Expr:
    """The same node, its children replaced by their columns. Leaves, cross-sectional nodes (a
    lookup, which never recurses) and any node type not listed are returned unchanged, so they are
    evaluated per bar exactly as before."""
    if isinstance(expr, Window):
        return Window(expr.op, expr.lookback, _col(expr.operand, bars, ctx, memo))
    if isinstance(expr, Corr):
        return Corr(expr.lookback, _col(expr.left, bars, ctx, memo),
                    _col(expr.right, bars, ctx, memo))
    if isinstance(expr, Delay):
        return Delay(expr.lookback, _col(expr.operand, bars, ctx, memo))
    if isinstance(expr, BinOp):
        return BinOp(expr.op, _col(expr.left, bars, ctx, memo), _col(expr.right, bars, ctx, memo))
    if isinstance(expr, UnOp):
        return UnOp(expr.op, _col(expr.operand, bars, ctx, memo))
    if isinstance(expr, Signal):
        return Signal(_col(expr.operand, bars, ctx, memo))
    if isinstance(expr, IfElse):
        cond = _col(expr.condition, bars, ctx, memo)
        taken = any(v > 0 for v in cond.values)
        skipped = any(not v > 0 for v in cond.values)
        # A branch never taken is never read; a zero column of the right kind stands in for it so
        # the real IfElse's construction check still sees two branches of one kind.
        placeholder = tuple(0.0 for _ in range(len(bars)))
        then = (_col(expr.then, bars, ctx, memo) if taken
                else _Column(placeholder, expr.then.kind))
        otherwise = (_col(expr.otherwise, bars, ctx, memo) if skipped
                     else _Column(placeholder, expr.otherwise.kind))
        return IfElse(cond, then, otherwise)
    return expr


__all__ = ["evaluate_series", "series_cost"]
