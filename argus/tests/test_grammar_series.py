"""Columnar evaluation (`argus.research.grammar_series`) and the static cost bound
(`argus.research.grammar.cost` / `MAX_COST`).

The series evaluator's one non-negotiable property is bit-identity with the per-bar path: trial
accounting and duplicate detection compare numbers with ``==``, so "close enough" is a different
factor. The cost bound's is that it is an honest upper bound and that it refuses the tree that used
to pass ``validate`` while costing hours per bar.
"""

from __future__ import annotations

import ast
import json
import random
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from argus.backtest.engine import Bar
from argus.research import crosssection
from argus.research.grammar import (
    EXPANDED,
    MAX_COST,
    ORIGINAL_EIGHT,
    BinOp,
    Const,
    Corr,
    CrossContext,
    CrossRank,
    Delay,
    Expr,
    Field,
    GrammarError,
    IfElse,
    Kind,
    Ref,
    Signal,
    Window,
    cost,
    validate,
)
from argus.research.grammar_series import evaluate_series, series_cost
from argus.research.searchoff import node_cost, random_expr

ROOT = Path(__file__).resolve().parents[1]
CANDLES = ROOT / "data" / "regime_candles_fixture.json"


def _bars(n: int = 600) -> list[Bar]:
    series = json.loads(CANDLES.read_text(encoding="utf-8"))["series"]["NVDAUSDT"][:n]
    return [Bar(ts=datetime.fromisoformat(ts), close=Decimal(c),
                extra={"volume": 1000.0 + k % 17, "high": float(c) * 1.001,
                       "low": float(c) * 0.999})
            for k, (ts, c) in enumerate(series)]


def _per_bar(expr: Expr, bars: Sequence[Bar], ctx: CrossContext | None = None) -> list[float]:
    return [expr.evaluate(bars, i, ctx) for i in range(len(bars))]


class _Counting(Expr):
    """A NUMBER leaf that counts how often it is evaluated."""

    def __init__(self, calls: list[int]) -> None:
        self.calls = calls

    @property
    def kind(self) -> Kind:
        return Kind.NUMBER

    def evaluate(self, bars: Sequence[Any], i: int, ctx: CrossContext | None = None) -> float:
        self.calls[0] += 1
        return float(i % 7)

    def canonical(self) -> str:
        return f"counting{id(self)}"


class TestBitIdentity:
    def test_every_shipped_single_symbol_factor(self) -> None:
        bars = _bars()
        trees = [*ORIGINAL_EIGHT.values(), *EXPANDED.values()]
        for tree in trees:
            assert evaluate_series(tree, bars) == _per_bar(tree, bars), tree.canonical()

    def test_random_search_trees(self) -> None:
        bars = _bars(300)
        rng = random.Random(9)
        checked = 0
        while checked < 120:
            tree = random_expr(rng)
            if node_cost(tree) > 3_000:
                continue
            assert evaluate_series(tree, bars) == _per_bar(tree, bars), tree.canonical()
            checked += 1

    def test_cross_sectional_nodes_read_the_same_context(self) -> None:
        bars = _bars(50)
        inner = Window("mean", 6, Ref(Field.CLOSE))
        node = CrossRank(inner)
        ctx = CrossContext("NVDAUSDT", {node.canonical(): [k / 50 for k in range(50)]})
        tree = Signal(BinOp("sub", node, Const(0.5)))
        assert evaluate_series(tree, bars, ctx) == _per_bar(tree, bars, ctx)

    def test_a_cross_sectional_node_without_context_still_refuses(self) -> None:
        with pytest.raises(GrammarError, match="cross"):
            evaluate_series(Signal(CrossRank(Ref(Field.CLOSE))), _bars(10))

    def test_a_branch_never_taken_is_never_evaluated(self) -> None:
        """Per bar, IfElse only evaluates the branch it takes; a never-taken branch holding a
        cross-sectional node must not raise here either."""
        bars = _bars(40)
        never = BinOp("gt", Const(0.0), Const(1.0))
        tree = Signal(IfElse(never, CrossRank(Ref(Field.CLOSE)), Const(0.25)))
        assert evaluate_series(tree, bars) == _per_bar(tree, bars) == [0.25] * 40


class TestCost:
    def test_the_rules_read_off_evaluate(self) -> None:
        close = Ref(Field.CLOSE)
        assert cost(close) == 1
        assert cost(Window("mean", 10, close)) == 11
        assert cost(Window("mean", 10, Window("std", 5, close))) == 1 + 10 * 6
        assert cost(Delay(24, close)) == 2
        assert cost(Corr(8, close, Window("mean", 3, close))) == 1 + 8 * (1 + 4)
        assert cost(IfElse(BinOp("gt", close, Const(1.0)), Window("mean", 9, close), close)) \
            == 1 + 3 + 10

    def test_it_is_an_upper_bound_on_leaf_evaluations(self) -> None:
        bars = _bars(200)
        rng = random.Random(4)
        for _ in range(200):
            calls = [0]
            leaf = _Counting(calls)
            depth = rng.randint(1, 3)
            tree: Expr = leaf
            for _ in range(depth):
                pick = rng.random()
                if pick < 0.4:
                    tree = Window(rng.choice(Window.OPS), rng.choice((2, 5, 12)), tree)
                elif pick < 0.6:
                    tree = Delay(rng.choice((1, 6)), tree)
                elif pick < 0.8:
                    tree = Corr(rng.choice((3, 7)), tree, Ref(Field.CLOSE))
                else:
                    tree = BinOp("add", tree, Const(1.0))
            tree.evaluate(bars, len(bars) - 1)
            assert calls[0] <= cost(tree), tree.canonical()

    def test_the_tree_that_used_to_pass_is_refused_and_the_ablation_accepts_it(self) -> None:
        close: Expr = Ref(Field.CLOSE)
        for _ in range(4):
            close = Window("mean", 512, close)
        tree = Signal(close)
        assert cost(tree) > 6.8e10
        with pytest.raises(GrammarError, match="cost"):
            validate(tree)
        assert validate(tree, max_cost=10**30) is tree   # the pre-fix behaviour

    def test_every_shipped_factor_is_far_inside_the_bound(self) -> None:
        trees: list[Expr] = [*ORIGINAL_EIGHT.values(), *EXPANDED.values()]
        for value in vars(crosssection).values():
            if isinstance(value, dict):
                trees.extend(v for v in value.values() if isinstance(v, Signal))
        assert max(cost(t) for t in trees) < MAX_COST / 100
        for tree in trees:
            if isinstance(tree, Signal):
                validate(tree)

    def test_series_cost_is_additive_and_shares_subtrees(self) -> None:
        close = Ref(Field.CLOSE)
        chain = Window("mean", 48, Window("std", 48, Window("mean", 24, close)))
        assert cost(chain) == 1 + 48 * (1 + 48 * (1 + 24))
        assert series_cost(chain) == 48 + 48 + 24 + 1
        shared = Window("mean", 96, close)
        twice = BinOp("sub", shared, BinOp("mul", Const(2.0), shared))
        assert series_cost(twice) == series_cost(shared) + 3

    def test_the_series_path_evaluates_a_nested_chain_the_per_bar_path_cannot_afford(
        self,
    ) -> None:
        bars = _bars(700)
        close: Expr = Ref(Field.CLOSE)
        for _ in range(4):
            close = Window("mean", 512, close)
        values = evaluate_series(close, bars)
        assert len(values) == 700
        # Spot-check the last bar against a direct nested computation over plain lists.
        level = [float(b.close) for b in bars]
        for _ in range(4):
            level = [sum(level[max(0, i - 511): i + 1]) / len(level[max(0, i - 511): i + 1])
                     for i in range(len(level))]
        assert values[-1] == pytest.approx(level[-1], rel=1e-12)


def test_the_series_module_has_no_execution_surface() -> None:
    src = ROOT / "src" / "argus" / "research" / "grammar_series.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    forbidden = {"eval", "exec", "compile", "__import__", "open", "getattr", "setattr"}
    called = {
        (n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", ""))
        for n in ast.walk(tree) if isinstance(n, ast.Call)
    }
    assert not (called & forbidden)
