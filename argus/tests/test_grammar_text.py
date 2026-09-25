"""The factor grammar's text surface and normal form (`argus.research.grammar_text`).

The parser is the first way a model's *text* can become a factor, so the properties that matter are
the ones CEL guarantees for its own compile step: every well-typed tree round-trips, every
ill-typed or malformed input is refused with the column it went wrong at, input size and nesting
are bounded, and nothing in the module can reach the interpreter.
"""

from __future__ import annotations

import ast
import json
import random
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from argus.backtest.engine import Bar
from argus.research import crosssection
from argus.research.grammar import (
    EXPANDED,
    ORIGINAL_EIGHT,
    BinOp,
    Expr,
    Field,
    GrammarError,
    Kind,
    Ref,
    Signal,
    UnOp,
)
from argus.research.grammar_text import (
    MAX_NESTING,
    MAX_TEXT,
    GrammarTextError,
    normal_form,
    parse,
    parse_factor,
)
from argus.research.searchoff import node_cost, random_expr

SRC = Path(__file__).resolve().parents[1] / "src" / "argus" / "research" / "grammar_text.py"
CANDLES = Path(__file__).resolve().parents[1] / "data" / "regime_candles_fixture.json"


def _shipped() -> list[Expr]:
    trees: list[Expr] = [*ORIGINAL_EIGHT.values(), *EXPANDED.values()]
    for value in vars(crosssection).values():
        if isinstance(value, dict):
            trees.extend(v for v in value.values() if isinstance(v, Signal))
    return trees


def _bars(n: int = 400) -> list[Bar]:
    series = json.loads(CANDLES.read_text(encoding="utf-8"))["series"]["NVDAUSDT"][:n]
    return [Bar(ts=datetime.fromisoformat(ts), close=Decimal(c)) for ts, c in series]


class TestRoundTrip:
    def test_every_shipped_tree_round_trips(self) -> None:
        trees = _shipped()
        assert len(trees) > len(ORIGINAL_EIGHT) + len(EXPANDED)  # cross-sectional ones included
        for tree in trees:
            assert parse(tree.canonical()).canonical() == tree.canonical()

    def test_random_search_trees_round_trip(self) -> None:
        rng = random.Random(11)
        for _ in range(2000):
            tree = random_expr(rng)
            assert parse(tree.canonical()).canonical() == tree.canonical()

    def test_whitespace_between_tokens_is_accepted(self) -> None:
        pretty = "signal(\n  if( gt( zscore(48, close), 1.0 ),\n      1, 0 ) )"
        assert parse_factor(pretty).canonical() == EXPANDED["vol_adjusted_momentum"].canonical()


class TestRefusals:
    @pytest.mark.parametrize(
        ("text", "needle", "column"),
        [
            ("foo(close)", "unknown function 'foo'", 1),
            ("signal(closee)", "unknown field 'closee'", 8),
            ("signal(and(close, gt(close, 1)))", "and needs two booleans", 8),
            ("signal(if(close, 1, 0))", "the condition must be a boolean", 8),
            ("signal(mean(4.5, close))", "whole number of bars", 13),
            ("signal(mean(0, close))", "lookback=0", 8),
            ("signal(1e999)", "constants must be finite", 8),
            ("signal(gt(close, 1)) extra", "expected end of input", 22),
            ("signal(gt(close, 1, 2))", "gt takes 2 argument(s)", 19),
            ("signal(close.x)", "expected ')'", 13),
            ("signal(in_phase(lunch))", "unknown session phase 'lunch'", 17),
            ("signal(gt(close, 1)", "expected ')'", 20),
        ],
    )
    def test_refused_with_the_column(self, text: str, needle: str, column: int) -> None:
        with pytest.raises(GrammarTextError) as info:
            parse(text)
        assert needle in str(info.value)
        assert info.value.position + 1 == column

    def test_a_factor_must_be_rooted_in_signal(self) -> None:
        with pytest.raises(GrammarError, match="rooted in Signal"):
            parse_factor("gt(close, 1)")

    def test_the_cost_bound_applies_to_text(self) -> None:
        with pytest.raises(GrammarError, match="cost"):
            parse_factor("signal(mean(512,mean(512,mean(512,mean(512,close)))))")

    def test_input_length_is_bounded(self) -> None:
        with pytest.raises(GrammarTextError, match="limit"):
            parse("signal(" + " " * MAX_TEXT + "close)")

    def test_nesting_is_bounded_before_python_recursion(self) -> None:
        deep = "neg(" * (MAX_NESTING + 2) + "close" + ")" * (MAX_NESTING + 2)
        with pytest.raises(GrammarTextError, match="nesting"):
            parse(deep)
        pathological = "neg(" * 50_000 + "close" + ")" * 50_000
        with pytest.raises(GrammarTextError):
            parse(pathological[:MAX_TEXT])

    def test_non_ascii_and_symbols_are_unexpected_characters(self) -> None:
        # The full-width "c" is the input under test: a homoglyph must not parse as a field name.
        homoglyph = "signal(" + chr(0xFF43) + "lose)"
        for text in (homoglyph, "signal(close * 1)", "signal([close])", "Signal(close)"):
            with pytest.raises(GrammarTextError, match="unexpected character"):
                parse(text)


class TestNoExecutionSurface:
    """The same AST scan tests/test_grammar.py runs over the grammar itself."""

    def test_the_parser_calls_nothing_dangerous(self) -> None:
        tree = ast.parse(SRC.read_text(encoding="utf-8"))
        forbidden = {"eval", "exec", "compile", "__import__", "open", "getattr", "setattr"}
        called: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name):
                    called.add(fn.id)
                elif isinstance(fn, ast.Attribute):
                    called.add(fn.attr)
        assert not (called & forbidden), f"grammar_text calls {called & forbidden}"
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
        } | {
            node.module.split(".")[0]
            for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not (imported & {"subprocess", "os", "sys", "socket", "shutil", "pathlib",
                                "importlib", "pickle"})


class TestNormalForm:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("neg(neg(close))", "close"),
            ("sub(close,neg(volume))", "add(close,volume)"),
            ("lt(volume,close)", "gt(close,volume)"),
            ("add(add(close,volume),high)", "add(close,add(volume,high))"),
            ("and(and(gt(close,1),lt(volume,5)),gt(high,2))",
             "and(gt(high,2),and(lt(volume,5),gt(close,1)))"),
            ("corr(48,close,volume)", "corr(48,volume,close)"),
            ("mean(24,neg(neg(close)))", "mean(24,close)"),
        ],
    )
    def test_equivalent_spellings_collide(self, a: str, b: str) -> None:
        assert normal_form(parse(a)) == normal_form(parse(b))

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("sub(close,volume)", "sub(volume,close)"),
            ("div(close,volume)", "div(volume,close)"),
            ("gt(close,volume)", "lt(close,volume)"),
            ("mean(24,close)", "mean(48,close)"),
            ("neg(close)", "close"),
            ("sub(close,volume)", "add(close,volume)"),
            ("mul(add(close,volume),high)", "add(mul(close,high),volume)"),
        ],
    )
    def test_distinct_factors_do_not_collide(self, a: str, b: str) -> None:
        assert normal_form(parse(a)) != normal_form(parse(b))

    def test_exact_rewrites_preserve_values(self) -> None:
        """The double-negation and subtract-a-negation rewrites claim IEEE exactness; checked by
        value on random search trees over real closes, with ``==``, not a tolerance."""
        bars = _bars()
        rng = random.Random(5)
        checked = 0
        while checked < 150:
            base = random_expr(rng)
            if node_cost(base) > 2_000:
                continue
            assert base.kind is Kind.NUMBER
            double_neg = UnOp("neg", UnOp("neg", base))
            sub_neg = BinOp("sub", Ref(Field.CLOSE), UnOp("neg", base))
            plain_add = BinOp("add", Ref(Field.CLOSE), base)
            assert normal_form(double_neg) == normal_form(base)
            assert normal_form(sub_neg) == normal_form(plain_add)
            for i in range(0, len(bars), 37):
                assert double_neg.evaluate(bars, i) == base.evaluate(bars, i)
                assert sub_neg.evaluate(bars, i) == plain_add.evaluate(bars, i)
            checked += 1

    def test_relational_flip_is_exact(self) -> None:
        bars = _bars()
        gt = parse("gt(mean(24,close),close)")
        lt = parse("lt(close,mean(24,close))")
        assert normal_form(gt) == normal_form(lt)
        assert [gt.evaluate(bars, i) for i in range(len(bars))] == [
            lt.evaluate(bars, i) for i in range(len(bars))]

    def test_canonical_is_unchanged(self) -> None:
        """normal_form is additive: canonical() still spells a double negation as written, so no
        stored trial identity changes."""
        assert parse("neg(neg(close))").canonical() == "neg(neg(close))"
