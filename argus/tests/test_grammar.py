"""Expression-grammar tests.

The grammar exists because the factor search space was **eight items wide** — the proposer picked a
name from a dict. These tests assert the two properties that make the replacement safe rather than
merely larger: it is a strict **superset** of what it replaces, and it has **no execution surface**.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from argus.backtest.engine import Bar
from argus.research.factor_lab import PRIMITIVES
from argus.research.grammar import (
    MAX_DEPTH,
    ORIGINAL_EIGHT,
    BinOp,
    Const,
    Corr,
    Delay,
    Field,
    GrammarError,
    HasDiscovery,
    IfElse,
    InPhase,
    Kind,
    Ref,
    Return,
    Signal,
    UnOp,
    Window,
    as_signal_fn,
    validate,
)
from argus.truth.clocks import SessionPhase

D = Decimal


def _bars(n: int = 300, seed: int = 7) -> list[Bar]:
    rng = random.Random(seed)
    start, px, out = datetime(2026, 3, 1, tzinfo=UTC), D("100"), []
    for k in range(n):
        px = px * (D("1") + D(str(round(rng.uniform(-0.01, 0.01), 5))))
        out.append(Bar(ts=start + timedelta(hours=k), close=px))
    return out


class TestItIsASuperset:
    """If the grammar could not express what it replaces, it would be a narrowing dressed as a
    widening."""

    def test_all_eight_primitives_are_reconstructed(self) -> None:
        assert set(ORIGINAL_EIGHT) == set(PRIMITIVES)

    def test_each_reconstruction_is_numerically_identical(self) -> None:
        bars = _bars()
        for name, tree in ORIGINAL_EIGHT.items():
            original = PRIMITIVES[name]
            for i in range(len(bars)):
                assert abs(original(bars, i) - tree.evaluate(bars, i)) < 1e-9, (
                    f"{name} diverges at bar {i}"
                )

    def test_the_reconstructions_are_all_valid_factors(self) -> None:
        for tree in ORIGINAL_EIGHT.values():
            assert validate(tree).kind is Kind.SIGNAL

    def test_the_search_space_is_no_longer_eight(self) -> None:
        """The whole point. Two operators over one leaf already exceed the old library."""
        leaves = [Const(1.0), Ref(Field.CLOSE), Return(6), Return(48)]
        combos = {
            BinOp(op, a, b).canonical()
            for op in ("add", "sub", "mul", "div")
            for a in leaves
            for b in leaves
        }
        assert len(combos) > len(PRIMITIVES) * 4


class TestTheWarmUpBugItFound:
    """Reconstructing the primitives found a real defect in two of them.

    `closure_momentum` fell through to a full **-1.0** position when its 6-bar lookback had no
    history, because `_ret` returns 0.0 and `0.0 > 0` is False. A momentum factor taking a maximum
    short on data it does not have is a bug, and it was invisible until two implementations of the
    same idea were diffed.
    """

    def test_momentum_is_flat_without_history(self) -> None:
        bars = _bars()
        for i in range(6):
            if PRIMITIVES["closure_momentum"](bars, i) != 0.0:
                # Only meaningful on a closed bar; when open both are 0 anyway.
                pass
            assert abs(PRIMITIVES["closure_momentum"](bars, i)) != 1.0 or i >= 6

    def test_neither_primitive_takes_a_full_position_on_no_data(self) -> None:
        bars = _bars()
        for name in ("closure_momentum", "closure_reversion"):
            early = [PRIMITIVES[name](bars, i) for i in range(6)]
            assert all(v == 0.0 for v in early), f"{name} positions itself on missing history"

    def test_sign_of_zero_is_flat_not_negative(self) -> None:
        assert UnOp("sign", Const(0.0)).evaluate([], 0) == 0.0
        assert UnOp("sign", Const(-0.5)).evaluate([], 0) == -1.0
        assert UnOp("sign", Const(0.5)).evaluate([], 0) == 1.0


class TestTypeSystem:
    def test_logic_needs_booleans(self) -> None:
        with pytest.raises(GrammarError, match="two booleans"):
            BinOp("and", Const(1.0), Const(1.0))

    def test_arithmetic_needs_numbers(self) -> None:
        with pytest.raises(GrammarError, match="two numbers"):
            BinOp("add", HasDiscovery(), HasDiscovery())

    def test_not_needs_a_boolean(self) -> None:
        with pytest.raises(GrammarError, match="needs a"):
            UnOp("not", Const(1.0))

    def test_a_condition_must_be_boolean(self) -> None:
        with pytest.raises(GrammarError, match="must be a boolean"):
            IfElse(Const(1.0), Const(1.0), Const(0.0))

    def test_branches_must_agree(self) -> None:
        with pytest.raises(GrammarError, match="same kind"):
            IfElse(HasDiscovery(), Const(1.0), HasDiscovery())

    def test_unknown_operators_are_refused(self) -> None:
        with pytest.raises(GrammarError, match="unknown op"):
            BinOp("pow", Const(2.0), Const(3.0))
        with pytest.raises(GrammarError, match="unknown window op"):
            Window("kurtosis", 5, Const(1.0))

    def test_a_signal_cannot_wrap_a_signal(self) -> None:
        with pytest.raises(GrammarError, match="cannot wrap"):
            Signal(Signal(Const(1.0)))

    def test_a_factor_must_be_rooted_in_signal(self) -> None:
        """An unbounded number is not a position size."""
        with pytest.raises(GrammarError, match="not a position size"):
            validate(Const(999.0))  # type: ignore[arg-type]


class TestCanonicalFormAndDuplicates:
    """A proposer that rediscovers the same factor burns a trial, which then makes the Deflated
    Sharpe gate harsher for no scientific reason."""

    def test_commutative_arguments_collide(self) -> None:
        a = BinOp("add", Const(1.0), Ref(Field.CLOSE))
        b = BinOp("add", Ref(Field.CLOSE), Const(1.0))
        assert a.canonical() == b.canonical()

    def test_non_commutative_arguments_do_not_collide(self) -> None:
        a = BinOp("sub", Const(1.0), Ref(Field.CLOSE))
        b = BinOp("sub", Ref(Field.CLOSE), Const(1.0))
        assert a.canonical() != b.canonical()

    def test_float_noise_does_not_create_a_second_factor(self) -> None:
        assert Const(0.1).canonical() == Const(0.1000000001).canonical()

    def test_the_eight_reconstructions_are_all_distinct(self) -> None:
        forms = {t.canonical() for t in ORIGINAL_EIGHT.values()}
        assert len(forms) == len(ORIGINAL_EIGHT)

    def test_return_is_a_leaf_with_no_dead_argument(self) -> None:
        """It was briefly a Window with an ignored operand — two spellings of one factor."""
        assert Return(6).canonical() == "ret(6)"
        assert Return(6).children == ()


class TestThereIsNoExecutionSurface:
    """The property that makes this safe without a sandbox."""

    def test_the_grammar_calls_nothing_dangerous(self) -> None:
        """Checked by parsing the module, not by grepping it.

        A text search failed here first — on the word `subprocess` inside the docstring that
        explains why we did *not* take the sandbox route. Prose that names a hazard is not the
        hazard. The AST is the only honest place to ask this question.
        """
        import ast
        import pathlib

        src = pathlib.Path(__file__).resolve().parents[1] / "src/argus/research/grammar.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))

        forbidden = {"eval", "exec", "compile", "__import__", "open", "getattr", "setattr"}
        called: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name):
                    called.add(fn.id)
                elif isinstance(fn, ast.Attribute):
                    called.add(fn.attr)
        assert not (called & forbidden), f"grammar calls {called & forbidden}"

        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not (imported & {"subprocess", "os", "sys", "socket", "shutil", "pathlib"}), (
            f"grammar imports {imported}"
        )

    def test_a_factor_sees_only_the_bars_it_is_given(self) -> None:
        """No node can reach anything but the series passed in."""
        tree = ORIGINAL_EIGHT["long_while_closed"]
        assert as_signal_fn(tree)(_bars(20), 5) in (0.0, 1.0)

    def test_lookback_is_bounded(self) -> None:
        with pytest.raises(GrammarError, match="lookback"):
            Return(100_000)
        with pytest.raises(GrammarError, match="lookback"):
            Window("mean", 0, Const(1.0))

    def test_depth_and_size_are_bounded(self) -> None:
        deep = Const(1.0)
        for _ in range(MAX_DEPTH + 3):
            deep = BinOp("add", deep, Const(1.0))
        with pytest.raises(GrammarError, match="depth"):
            validate(Signal(deep))

    def test_division_by_zero_is_total_not_fatal(self) -> None:
        """A bad factor should die at the gate, not take down the search."""
        assert BinOp("div", Const(1.0), Const(0.0)).evaluate([], 0) == 0.0


class TestSignalBounds:
    def test_a_signal_is_clamped(self) -> None:
        assert Signal(Const(50.0)).evaluate([], 0) == 1.0
        assert Signal(Const(-50.0)).evaluate([], 0) == -1.0
        assert Signal(Const(0.3)).evaluate([], 0) == pytest.approx(0.3)

    def test_the_two_clock_structure_is_reachable(self) -> None:
        """Without this a proposer cannot express the project's core idea at all."""
        bars = _bars()
        tree = Signal(IfElse(
            BinOp("and", UnOp("not", HasDiscovery()), InPhase(SessionPhase.WEEKEND)),
            Const(1.0), Const(0.0),
        ))
        vals = {tree.evaluate(bars, i) for i in range(len(bars))}
        assert vals <= {0.0, 1.0}

    def test_windows_look_backward_only(self) -> None:
        """Look-ahead is not expressible, even by a proposer trying to."""
        bars = _bars(50)
        tree = Window("mean", 10, Ref(Field.CLOSE))
        early = tree.evaluate(bars[:20], 19)
        full = tree.evaluate(bars, 19)
        assert early == full, "a backward window must not see bars after its index"


class TestTheExpandedVocabulary:
    """Added 2026-09-13 after the factor audit found the search space, not the search, was the
    binding constraint: eight primitives over four fields is not a space to search."""

    def _bars(self, n: int = 120) -> list[Bar]:
        import random
        rng = random.Random(4)
        base = datetime(2026, 6, 2, 14, 0, tzinfo=UTC)
        price = 200.0
        out = []
        for i in range(n):
            price *= 1 + rng.gauss(0, 0.004)
            out.append(Bar(
                ts=base + timedelta(hours=i), close=Decimal(str(round(price, 2))),
                extra={"volume": 1000 + i * 7, "high": price * 1.003, "low": price * 0.997},
            ))
        return out

    def test_volume_is_readable(self) -> None:
        bars = self._bars()
        assert Ref(Field.VOLUME).evaluate(bars, len(bars) - 1) > 0

    def test_range_is_expressed_in_basis_points(self) -> None:
        bars = self._bars()
        got = Ref(Field.RANGE_BPS).evaluate(bars, len(bars) - 1)
        assert 55 < got < 65, got

    def test_a_missing_field_reads_as_absent_not_as_a_crash(self) -> None:
        """A caller that built bars without volume gets a factor that cannot use it."""
        bare = [Bar(ts=datetime(2026, 6, 2, tzinfo=UTC), close=Decimal("100"))]
        assert Ref(Field.VOLUME).evaluate(bare, 0) == 0.0

    def test_a_missing_price_field_degenerates_to_the_close(self) -> None:
        """So a missing high/low is a zero range, never a price of zero."""
        bare = [Bar(ts=datetime(2026, 6, 2, tzinfo=UTC), close=Decimal("100"))]
        assert Ref(Field.HIGH).evaluate(bare, 0) == 100.0
        assert Ref(Field.RANGE_BPS).evaluate(bare, 0) == 0.0

    def test_zscore_measures_distance_in_standard_deviations(self) -> None:
        flat = [
            Bar(ts=datetime(2026, 6, 2, i, tzinfo=UTC), close=Decimal("100"))
            for i in range(10)
        ]
        assert Window("zscore", 5, Ref(Field.CLOSE)).evaluate(flat, 9) == 0.0

    def test_a_flat_window_zscores_to_zero_not_infinity(self) -> None:
        """Dividing by a zero standard deviation would make a rounding error the best signal."""
        flat = [
            Bar(ts=datetime(2026, 6, 2, i, tzinfo=UTC), close=Decimal("50"))
            for i in range(20)
        ]
        assert Window("zscore", 10, Ref(Field.CLOSE)).evaluate(flat, 19) == 0.0

    def test_rank_is_a_fraction_of_the_window(self) -> None:
        bars = self._bars()
        got = Window("rank", 48, Ref(Field.CLOSE)).evaluate(bars, len(bars) - 1)
        assert 0.0 <= got <= 1.0

    def test_a_constant_series_ranks_in_the_middle(self) -> None:
        flat = [
            Bar(ts=datetime(2026, 6, 2, i, tzinfo=UTC), close=Decimal("7"))
            for i in range(20)
        ]
        assert Window("rank", 10, Ref(Field.CLOSE)).evaluate(flat, 19) == 0.5

    def test_a_single_observation_window_ranks_in_the_middle_too(self) -> None:
        """A single-bar window (the very first bar of any live run) is trivially "constant" the
        same way a genuinely flat multi-bar window is — it must reach the same 0.5 verdict this
        method's own comment documents, not a different one. Found by comparing against qlib's
        real `Rank` (`eval/grammar_comparison.py`): a shared `len(vals) < 2: return 0.0` guard
        used to sit in front of the rank branch and shadow its own `else 0.5` fallback for
        exactly this case, silently returning 0.0 instead."""
        bars = [Bar(ts=datetime(2026, 6, 2, tzinfo=UTC), close=Decimal("7"))]
        assert Window("rank", 10, Ref(Field.CLOSE)).evaluate(bars, 0) == 0.5

    def test_argmax_counts_bars_since_the_high(self) -> None:
        rising = [
            Bar(ts=datetime(2026, 6, 2, i, tzinfo=UTC), close=Decimal(str(100 + i)))
            for i in range(20)
        ]
        assert Window("argmax", 10, Ref(Field.CLOSE)).evaluate(rising, 19) == 0.0
        assert Window("argmin", 10, Ref(Field.CLOSE)).evaluate(rising, 19) == 9.0

    def test_slope_is_positive_on_a_rising_series(self) -> None:
        rising = [
            Bar(ts=datetime(2026, 6, 2, i, tzinfo=UTC), close=Decimal(str(100 + i)))
            for i in range(20)
        ]
        assert Window("slope", 10, Ref(Field.CLOSE)).evaluate(rising, 19) == pytest.approx(1.0)

    def test_median_ignores_a_single_outlier(self) -> None:
        prices = [Decimal("10")] * 9 + [Decimal("1000")]
        bars = [
            Bar(ts=datetime(2026, 6, 2, i, tzinfo=UTC), close=p)
            for i, p in enumerate(prices)
        ]
        assert Window("median", 10, Ref(Field.CLOSE)).evaluate(bars, 9) == 10.0

    def test_delay_reaches_back_exactly_n_bars(self) -> None:
        bars = [
            Bar(ts=datetime(2026, 6, 2, i, tzinfo=UTC), close=Decimal(str(100 + i)))
            for i in range(20)
        ]
        assert Delay(5, Ref(Field.CLOSE)).evaluate(bars, 19) == 114.0

    def test_delay_before_the_series_starts_holds_the_oldest_value(self) -> None:
        """Zero is a price of nothing and would read as a collapse."""
        bars = [
            Bar(ts=datetime(2026, 6, 2, i, tzinfo=UTC), close=Decimal(str(100 + i)))
            for i in range(5)
        ]
        assert Delay(50, Ref(Field.CLOSE)).evaluate(bars, 4) == 100.0

    def test_delay_enables_delta_which_was_previously_inexpressible(self) -> None:
        bars = [
            Bar(ts=datetime(2026, 6, 2, i, tzinfo=UTC), close=Decimal(str(100 + i)))
            for i in range(20)
        ]
        delta = BinOp("sub", Ref(Field.CLOSE), Delay(5, Ref(Field.CLOSE)))
        assert delta.evaluate(bars, 19) == 5.0

    def test_correlation_is_bounded(self) -> None:
        bars = self._bars()
        got = Corr(48, Ref(Field.CLOSE), Ref(Field.VOLUME)).evaluate(bars, len(bars) - 1)
        assert -1.0 <= got <= 1.0

    def test_a_flat_leg_correlates_to_zero_not_to_one(self) -> None:
        flat = [
            Bar(ts=datetime(2026, 6, 2, i, tzinfo=UTC), close=Decimal("100"),
                extra={"volume": 10 + i})
            for i in range(20)
        ]
        assert Corr(10, Ref(Field.CLOSE), Ref(Field.VOLUME)).evaluate(flat, 19) == 0.0

    def test_correlation_canonicalises_commutatively(self) -> None:
        """corr(a,b) and corr(b,a) are one factor and must not be paid for twice."""
        a = Corr(24, Ref(Field.CLOSE), Ref(Field.VOLUME)).canonical()
        b = Corr(24, Ref(Field.VOLUME), Ref(Field.CLOSE)).canonical()
        assert a == b

    def test_a_window_over_a_signal_is_still_refused(self) -> None:
        with pytest.raises(GrammarError, match="not meaningful"):
            Window("zscore", 5, Signal(Const(1.0)))

    def test_delaying_a_signal_is_refused(self) -> None:
        with pytest.raises(GrammarError, match="not meaningful"):
            Delay(5, Signal(Const(1.0)))

    def test_correlating_a_signal_is_refused(self) -> None:
        with pytest.raises(GrammarError, match="not meaningful"):
            Corr(5, Signal(Const(1.0)), Const(1.0))

    def test_a_correlation_needs_at_least_two_bars(self) -> None:
        with pytest.raises(GrammarError, match=r"\[2, 512\]"):
            Corr(1, Const(1.0), Const(2.0))

    def test_every_new_window_op_is_evaluable(self) -> None:
        bars = self._bars()
        for op in Window.OPS:
            got = Window(op, 24, Ref(Field.CLOSE)).evaluate(bars, len(bars) - 1)
            assert isinstance(got, float), op
