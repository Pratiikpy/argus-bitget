"""A typed expression grammar for factors — the fix for an eight-item search space.

**The problem this solves.** `argus.research.factor_lab` shipped with eight hand-written
primitives, and a proposer that picks a *name* from them. Its own comment says why: *"model-authored
code in an evaluator is an arbitrary-execution surface, and sandboxing it is a larger problem than
this sub-theme needs."* That judgement was correct. Its consequence was not visible at the time: a
certification machine pointed at a search space eight items wide will correctly certify nothing,
forever, and will never report that the problem is upstream. Our 0/8 result is a measurement of the
search space, not of the gates.

**Why a grammar rather than a sandbox.** The obvious fix is to let the model emit code and run it
somewhere safe. We tore down the leading candidate for that (`open-webui/open-terminal`, MIT) and
found a remote shell API rather than a sandbox — `subprocess.Popen(command, shell=True)`, with all
isolation delegated to Docker, no per-command resource limits and no command validation. Every
control that matters would have been ours to build anyway.

A grammar gets the same benefit with **no execution surface at all**. The proposer emits a *tree*;
the evaluator validates it against a type system and then *interprets* it. There is no path from a
model's output to the Python interpreter, so injection and exfiltration do not arise. The search
space goes from 8 to combinatorially large, which is the thing we actually wanted.

**Resource exhaustion is a separate property, and it was not true here until it was measured.**
This docstring used to say it "does not arise". It did: ``Window`` re-evaluates its operand once
per bar of its lookback, so windows nest *multiplicatively*, and :func:`validate` bounded only
depth and size. ``signal(mean(512,mean(512,mean(512,mean(512,close)))))`` is depth 6 and size 6,
passed ``validate``, and costs 512**4 ~ 6.9e10 node evaluations per bar — measured at 0.39 to
0.61 us per node on the development machine, seven to twelve hours for one bar. Found by running
Google CEL (``eval/general_grammar_comparison.py``), a general-purpose safe-expression language
that bounds the same thing at runtime: ``comprehension_max_iterations = 10000`` per evaluation,
``google/cel-cpp runtime/runtime_options.h:83``. The fix here is static rather than a runtime
counter, because the tree is fully known before evaluation: :func:`cost` is an upper bound on node
evaluations per call, and :func:`validate` refuses anything above :data:`MAX_COST`.

**What the grammar deliberately cannot express.** No unbounded recursion, no data access beyond the
bar series it is handed, no I/O, no imports, no user-defined functions. A factor is a pure function
of `(bars, index)` and nothing else. If a hypothesis genuinely cannot be written here, that is a
signal to extend the grammar deliberately — not to open an execution hole.

**Canonical form and duplicate detection.** A proposer left to itself will rediscover the same
factor under different spellings and burn trials on it, which then inflates the trial counter and
makes the Deflated Sharpe gate harsher for no scientific reason. Every node therefore has a
canonical serialisation with commutative arguments sorted, so `add(a, b)` and `add(b, a)` collide.
AlphaAgent does duplicate control on its own AST representation for the same reason.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any

from argus.truth.clocks import DualClock, SessionPhase

_CLOCK = DualClock()


class GrammarError(ValueError):
    """The expression is not well-formed, or not well-typed."""


class Kind(StrEnum):
    """The type system. Three kinds, checked at construction.

    Small on purpose: a type system a proposer cannot satisfy is a search space of zero, which is
    the mistake this module exists to undo.
    """

    NUMBER = "number"
    """A real value at a point in time."""

    BOOLEAN = "boolean"
    """A truth value at a point in time."""

    SIGNAL = "signal"
    """A target weight in [-1, 1]. Only a complete factor has this kind."""


# --- the data a factor is allowed to see ------------------------------------------------------
# Deliberately narrow. Anything not on this list cannot be reached from inside an expression, which
# is what makes the grammar safe without a sandbox.

class Field(StrEnum):
    """What the grammar can observe about a bar.

    The first four were the whole vocabulary, and the factor audit
    (`research/architecture/factor-discovery-audit.md`) found that the binding constraint on
    discovery was not the search but what there was to search over: with close, one-bar return,
    time-to-discovery and basis, an expression cannot say anything about volume, range, or where in
    its recent distribution the price sits. Qlib's handlers expose 158 to 360 features; eight
    hand-written primitives over four fields is not a search space.

    Every field added here must be **derivable from the bar itself** and present at the decision
    instant. `VOLUME`, `HIGH` and `LOW` come from the venue's own candle and are read out of
    ``Bar.extra`` — see :func:`_extra`, which returns 0.0 when a caller did not carry them rather
    than inventing a value.
    """

    CLOSE = "close"
    RETURN_1 = "return_1"
    HOURS_TO_DISCOVERY = "hours_to_discovery"
    BASIS_BPS = "basis_bps"

    VOLUME = "volume"
    """Traded base volume for the bar. Absent in a caller that did not carry it, and 0.0 then."""

    HIGH = "high"
    LOW = "low"
    RANGE_BPS = "range_bps"
    """(high - low) / close in basis points. Intrabar realised range — the cheapest volatility
    observation available, and one no expression could previously make."""


class Expr(ABC):
    """A node. Immutable, typed, and serialisable to a canonical string."""

    @property
    @abstractmethod
    def kind(self) -> Kind: ...

    @abstractmethod
    def evaluate(self, bars: Sequence[Any], i: int, ctx: CrossContext | None = None) -> float: ...

    @abstractmethod
    def canonical(self) -> str:
        """A spelling-independent identity. Two expressions with the same canonical form are the
        same factor and must not both consume a trial."""

    @property
    def depth(self) -> int:
        return 1 + max((c.depth for c in self.children), default=0)

    @property
    def children(self) -> tuple[Expr, ...]:
        return ()

    @property
    def size(self) -> int:
        return 1 + sum(c.size for c in self.children)

    @property
    def is_cross_sectional(self) -> bool:
        """True for a node that compares this instrument against the others at the same instant.

        Only :class:`CrossRank` and :class:`CrossScale` set this. It is a property of the node, not
        of the tree; :attr:`needs_panel` is the question a caller actually asks.
        """
        return False

    @property
    def needs_panel(self) -> bool:
        """True if anything anywhere in this tree needs the whole universe to evaluate.

        The check a single-symbol caller must make before running an expression. Without it a tree
        containing one cross-sectional node deep inside looks evaluable right up to the moment it
        raises, and the sweep would discover that per candidate rather than per batch.
        """
        return self.is_cross_sectional or any(c.needs_panel for c in self.children)

    def __str__(self) -> str:
        return self.canonical()


# --- leaves -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Const(Expr):
    value: float

    @property
    def kind(self) -> Kind:
        return Kind.NUMBER

    def evaluate(self, bars: Sequence[Any], i: int, ctx: CrossContext | None = None) -> float:
        return self.value

    def canonical(self) -> str:
        # Rounded so 0.1 and 0.10000000001 are one factor, not two trials.
        return f"{self.value:.6g}"


@dataclass(frozen=True, slots=True)
class Ref(Expr):
    """A reference to one observable field at the current bar."""

    field: Field

    @property
    def kind(self) -> Kind:
        return Kind.NUMBER

    def evaluate(self, bars: Sequence[Any], i: int, ctx: CrossContext | None = None) -> float:
        bar = bars[i]
        if self.field is Field.CLOSE:
            return float(bar.close)
        if self.field is Field.RETURN_1:
            return _ret(bars, i, 1)
        if self.field is Field.HOURS_TO_DISCOVERY:
            return _CLOCK.state(bar.ts).hours_to_next_discovery
        if self.field is Field.VOLUME:
            return _extra(bar, "volume")
        if self.field is Field.HIGH:
            return _extra(bar, "high", default=float(bar.close))
        if self.field is Field.LOW:
            return _extra(bar, "low", default=float(bar.close))
        if self.field is Field.RANGE_BPS:
            close = float(bar.close)
            if close <= 0:
                return 0.0
            high = _extra(bar, "high", default=close)
            low = _extra(bar, "low", default=close)
            return (high - low) / close * 10_000.0
        return _basis_bps(bar)

    def canonical(self) -> str:
        return str(self.field)


@dataclass(frozen=True, slots=True)
class InPhase(Expr):
    """Is the anchor market in this session phase?

    The grammar's one piece of domain knowledge, and the reason it can express anything the eight
    primitives could. Without it a proposer cannot reach the two-clock structure at all.
    """

    phase: SessionPhase

    @property
    def kind(self) -> Kind:
        return Kind.BOOLEAN

    def evaluate(self, bars: Sequence[Any], i: int, ctx: CrossContext | None = None) -> float:
        return 1.0 if _CLOCK.phase(bars[i].ts) is self.phase else 0.0

    def canonical(self) -> str:
        return f"in_phase({self.phase})"


@dataclass(frozen=True, slots=True)
class HasDiscovery(Expr):
    """Is genuine price discovery happening right now?

    Distinct from any single phase: EXTENDED has a quote and no discovery, which is the
    distinction the whole project rests on.
    """

    @property
    def kind(self) -> Kind:
        return Kind.BOOLEAN

    def evaluate(self, bars: Sequence[Any], i: int, ctx: CrossContext | None = None) -> float:
        return 1.0 if _CLOCK.phase(bars[i].ts).has_price_discovery else 0.0

    def canonical(self) -> str:
        return "has_discovery()"


# --- windowed operators -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Return(Expr):
    """Return over ``lookback`` bars. A leaf, not a window.

    It was briefly modelled as ``Window("ret", n, operand)`` with the operand ignored — a node
    whose argument does nothing is a node two different trees can spell differently while meaning
    the same thing, which defeats the canonical form that duplicate detection depends on.

    **Insufficient history returns 0.0, which the grammar treats as flat.** The hand-written
    primitive this replaces returned ``-1.0`` in that case: a momentum factor taking a full short
    position on a window it had no data for. See the note in :data:`ORIGINAL_EIGHT`.
    """

    lookback: int

    def __post_init__(self) -> None:
        if not 1 <= self.lookback <= 512:
            raise GrammarError(f"lookback={self.lookback} must be in [1, 512]")

    @property
    def kind(self) -> Kind:
        return Kind.NUMBER

    def evaluate(self, bars: Sequence[Any], i: int, ctx: CrossContext | None = None) -> float:
        return _ret(bars, i, self.lookback)

    def canonical(self) -> str:
        return f"ret({self.lookback})"


@dataclass(frozen=True, slots=True)
class Window(Expr):
    """A backward-looking aggregate. **Backward only** — there is no forward window in the
    grammar, so look-ahead cannot be expressed even by a proposer trying to."""

    op: str
    lookback: int
    operand: Expr

    OPS = (
        "mean", "sum", "max", "min", "std",
        # Added from the 101-Formulaic-Alphas operator vocabulary and Qlib's expression set, each
        # because it lets an expression say something the first five cannot:
        "zscore",    # (x - mean) / std over the window — volatility-adjusted, the audit's #1 gap
        "rank",      # where the current value sits in its own recent distribution, 0..1
        "argmax",    # bars since the window's maximum (ts_argmax); a recency measure
        "argmin",
        "median",    # a middle robust to the one outlier a mean chases
        "slope",     # least-squares trend per bar over the window
    )

    def __post_init__(self) -> None:
        if self.op not in self.OPS:
            raise GrammarError(f"unknown window op {self.op!r}; allowed: {self.OPS}")
        if not 1 <= self.lookback <= 512:
            raise GrammarError(f"lookback={self.lookback} must be in [1, 512]")
        if self.operand.kind is Kind.SIGNAL:
            raise GrammarError("a window over a signal is not meaningful")

    @property
    def kind(self) -> Kind:
        return Kind.NUMBER

    @property
    def children(self) -> tuple[Expr, ...]:
        return (self.operand,)

    def evaluate(self, bars: Sequence[Any], i: int, ctx: CrossContext | None = None) -> float:
        start = max(0, i - self.lookback + 1)
        vals = [self.operand.evaluate(bars, j, ctx) for j in range(start, i + 1)]
        if not vals:
            return 0.0
        if self.op == "mean":
            return sum(vals) / len(vals)
        if self.op == "sum":
            return sum(vals)
        if self.op == "max":
            return max(vals)
        if self.op == "min":
            return min(vals)
        if self.op == "median":
            ordered = sorted(vals)
            mid = len(ordered) // 2
            return (
                ordered[mid] if len(ordered) % 2
                else (ordered[mid - 1] + ordered[mid]) / 2
            )
        if self.op == "argmax":
            # Bars since the maximum, newest-relative. A window whose high was 30 bars ago is a
            # different state from one that just made it, and no aggregate distinguishes them.
            return float(len(vals) - 1 - vals.index(max(vals)))
        if self.op == "argmin":
            return float(len(vals) - 1 - vals.index(min(vals)))
        if self.op == "rank":
            # Own guard, checked before the shared `len(vals) < 2` return below: `rank` needs
            # neither `mu` nor `sd`, and a single-observation window (a real, live case — the
            # very first bar any rank-based factor ever sees) is trivially "constant", so it must
            # reach the same 0.5 verdict as a genuinely flat multi-bar window does, not the 0.0
            # every other under-length op returns. A single shared `len(vals) < 2: return 0.0`
            # used to sit in front of this whole block and silently shadowed this branch's own
            # `else 0.5` for exactly that case — found by comparing against qlib's real
            # `Rank`, whose pandas-rank convention gives a single-observation window 1.0, not
            # 0.0, and confirmed as a real gap against this method's own documented intent
            # ("a constant series ranks 0.5") rather than a difference in convention.
            current = vals[-1]
            below = sum(1 for v in vals if v < current)
            ties = sum(1 for v in vals if v == current)
            # Midrank for ties, scaled to 0..1. A constant series ranks 0.5 — neither high nor low
            # in its own distribution, which is the honest reading.
            return (below + (ties - 1) / 2) / (len(vals) - 1) if len(vals) > 1 else 0.5
        if len(vals) < 2:
            return 0.0
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))
        if self.op == "std":
            return sd
        if self.op == "zscore":
            # Zero, not infinity, when the window did not move: a flat window carries no
            # information about how unusual the current value is, and dividing by it would let a
            # rounding error become a signal.
            return 0.0 if sd <= 1e-12 else (vals[-1] - mu) / sd
        if self.op == "slope":
            n = len(vals)
            mean_x = (n - 1) / 2
            denom = sum((k - mean_x) ** 2 for k in range(n))
            if denom <= 0:
                return 0.0
            return sum((k - mean_x) * (vals[k] - mu) for k in range(n)) / denom
        return sd

    def canonical(self) -> str:
        return f"{self.op}({self.lookback},{self.operand.canonical()})"


@dataclass(frozen=True, slots=True)
class Delay(Expr):
    """The value of an expression ``n`` bars ago.

    The single most-used operator in the 101 Formulaic Alphas, and the one whose absence was most
    limiting here: without it the grammar cannot express *change* of anything except the built-in
    one-bar return. ``sub(x, delay(n, x))`` is delta; ``div(x, delay(n, x))`` is a ratio; both were
    inexpressible.

    Backward only, like :class:`Window`. Before the series starts it returns the oldest value it
    has rather than zero — zero is a price of nothing and would read as a collapse.
    """

    lookback: int
    operand: Expr

    def __post_init__(self) -> None:
        if not 1 <= self.lookback <= 512:
            raise GrammarError(f"lookback={self.lookback} must be in [1, 512]")
        if self.operand.kind is Kind.SIGNAL:
            raise GrammarError("delaying a signal is not meaningful")

    @property
    def kind(self) -> Kind:
        return Kind.NUMBER

    @property
    def children(self) -> tuple[Expr, ...]:
        return (self.operand,)

    def evaluate(self, bars: Sequence[Any], i: int, ctx: CrossContext | None = None) -> float:
        return self.operand.evaluate(bars, max(0, i - self.lookback), ctx)

    def canonical(self) -> str:
        return f"delay({self.lookback},{self.operand.canonical()})"


@dataclass(frozen=True, slots=True)
class Corr(Expr):
    """Rolling Pearson correlation between two expressions over ``lookback`` bars.

    The one operator that needs two operands, and the reason several of the 101 alphas cannot be
    written without it — "price moves with volume" is a correlation, not an arithmetic combination.

    Returns 0.0 when either leg is flat over the window. That is the honest value: an undefined
    correlation is not a strong one, and returning something large would make a degenerate window
    the most attractive signal in the search.
    """

    lookback: int
    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        if not 2 <= self.lookback <= 512:
            raise GrammarError(f"lookback={self.lookback} must be in [2, 512]")
        for side in (self.left, self.right):
            if side.kind is Kind.SIGNAL:
                raise GrammarError("correlating a signal is not meaningful")

    @property
    def kind(self) -> Kind:
        return Kind.NUMBER

    @property
    def children(self) -> tuple[Expr, ...]:
        return (self.left, self.right)

    def evaluate(self, bars: Sequence[Any], i: int, ctx: CrossContext | None = None) -> float:
        start = max(0, i - self.lookback + 1)
        xs = [self.left.evaluate(bars, j, ctx) for j in range(start, i + 1)]
        ys = [self.right.evaluate(bars, j, ctx) for j in range(start, i + 1)]
        n = len(xs)
        if n < 2:
            return 0.0
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        if sxx <= 1e-24 or syy <= 1e-24:
            return 0.0
        sxy = sum((xs[k] - mx) * (ys[k] - my) for k in range(n))
        return max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))

    def canonical(self) -> str:
        # Commutative: corr(a,b) and corr(b,a) are the same factor and must not be paid for twice.
        legs = sorted((self.left.canonical(), self.right.canonical()))
        return f"corr({self.lookback},{legs[0]},{legs[1]})"


# --- cross-sectional ----------------------------------------------------------------------------
#
# Everything above this line looks at one instrument's own history. That was the grammar's largest
# structural gap, and it is measured rather than asserted: the port of the 101 Formulaic Alphas
# (`research/architecture/alpha101-port.md`) found **46 of 101 blocked on cross-sectional rank
# alone** — `rank(x)` in that vocabulary means "where does this name sit among all names right
# now", and a time-series rank within one instrument's own window is a different question with the
# same spelling.
#
# The design decision that matters here is what these nodes do when they are evaluated without a
# universe. They **raise**. Returning 0.5, or the time-series rank, would let forty-six factors
# silently degenerate into something else and still produce numbers — and a sweep cannot tell a
# degenerate factor from a real one by looking at its output. :mod:`argus.research.panel` is the
# only thing that can evaluate them, and it builds the context first.


@dataclass(frozen=True, slots=True)
class CrossContext:
    """Cross-sectional values, precomputed for the whole universe, for one symbol's evaluation.

    ``values`` maps a node's canonical form to that node's already-ranked series **for the symbol
    currently being evaluated**. The panel evaluator builds one of these per symbol, innermost
    nodes first, so a nested cross-sectional expression resolves bottom-up.

    Frozen and passed explicitly rather than held in a module-level variable. The panel evaluates
    symbols in a loop today and could evaluate them in threads tomorrow; a global context would
    make that change silently wrong, and the bug would look like a factor that works until the
    sweep gets faster.
    """

    symbol: str
    values: Mapping[str, Sequence[float]]

    def lookup(self, canonical: str, i: int) -> float:
        series = self.values.get(canonical)
        if series is None:
            raise GrammarError(
                f"{canonical} was not precomputed for {self.symbol}; a cross-sectional node must "
                f"be evaluated through argus.research.panel, which computes the universe first"
            )
        if not 0 <= i < len(series):
            raise GrammarError(
                f"{canonical} has {len(series)} value(s) for {self.symbol}, index {i} requested; "
                f"the panel is not aligned with the bars being evaluated"
            )
        return series[i]


class _CrossSectional(Expr):
    """Shared refusal. Subclasses differ only in how they combine the universe's values."""

    operand: Expr

    @property
    def kind(self) -> Kind:
        return Kind.NUMBER

    @property
    def children(self) -> tuple[Expr, ...]:
        return (self.operand,)

    @property
    def is_cross_sectional(self) -> bool:
        return True

    def evaluate(self, bars: Sequence[Any], i: int, ctx: CrossContext | None = None) -> float:
        if ctx is None:
            raise GrammarError(
                f"{self.canonical()} compares this instrument against the rest of the universe "
                f"and cannot be evaluated from one symbol's bars. Run it through "
                f"argus.research.panel.evaluate_panel, which computes every symbol first. "
                f"Returning a neutral value here would turn a cross-sectional factor into a "
                f"different factor with the same name."
            )
        return ctx.lookup(self.canonical(), i)

    @staticmethod
    def combine(values: Sequence[float]) -> list[float]:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class CrossRank(_CrossSectional):
    """Where this instrument sits among the universe at this instant, scaled to 0..1.

    ``rank()`` as the 101 Formulaic Alphas mean it, and as Qlib computes it by grouping a
    (datetime, instrument) frame by datetime. Ties take the midrank and a single-name universe
    ranks 0.5 — neither high nor low among one name, which is the only honest reading and is
    exactly the value that would have been silently returned had the refusal above not been
    written.
    """

    operand: Expr

    def __post_init__(self) -> None:
        if self.operand.kind is Kind.SIGNAL:
            raise GrammarError("ranking a signal across the universe is not meaningful")

    def canonical(self) -> str:
        return f"crossrank({self.operand.canonical()})"

    @staticmethod
    def combine(values: Sequence[float]) -> list[float]:
        n = len(values)
        if n <= 1:
            return [0.5] * n
        out: list[float] = []
        for v in values:
            below = sum(1 for other in values if other < v)
            ties = sum(1 for other in values if other == v)
            out.append((below + (ties - 1) / 2) / (n - 1))
        return out


@dataclass(frozen=True, slots=True)
class CrossScale(_CrossSectional):
    """Normalise so the universe's absolute values sum to one at this instant.

    ``scale()`` in the 101 vocabulary, and the operator roughly eight of them need in order to turn
    a raw number into a portfolio weight. A universe that is entirely zero scales to zero rather
    than dividing by it: no position is the correct reading of no signal, and an infinity here
    would become the most attractive weight in the book.
    """

    operand: Expr

    def __post_init__(self) -> None:
        if self.operand.kind is Kind.SIGNAL:
            raise GrammarError("scaling a signal across the universe is not meaningful")

    def canonical(self) -> str:
        return f"crossscale({self.operand.canonical()})"

    @staticmethod
    def combine(values: Sequence[float]) -> list[float]:
        total = sum(abs(v) for v in values)
        if total <= 1e-12:
            return [0.0] * len(values)
        return [v / total for v in values]


# --- arithmetic and logic ---------------------------------------------------------------------

_COMMUTATIVE = {"add", "mul", "and", "or"}


@dataclass(frozen=True, slots=True)
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr

    ARITH = ("add", "sub", "mul", "div")
    COMPARE = ("gt", "lt")
    LOGIC = ("and", "or")

    def __post_init__(self) -> None:
        allowed = self.ARITH + self.COMPARE + self.LOGIC
        if self.op not in allowed:
            raise GrammarError(f"unknown op {self.op!r}; allowed: {allowed}")
        if self.op in self.LOGIC:
            if self.left.kind is not Kind.BOOLEAN or self.right.kind is not Kind.BOOLEAN:
                raise GrammarError(f"{self.op} needs two booleans")
        elif self.left.kind is not Kind.NUMBER or self.right.kind is not Kind.NUMBER:
            raise GrammarError(f"{self.op} needs two numbers")

    @property
    def kind(self) -> Kind:
        return Kind.NUMBER if self.op in self.ARITH else Kind.BOOLEAN

    @property
    def children(self) -> tuple[Expr, ...]:
        return (self.left, self.right)

    def evaluate(self, bars: Sequence[Any], i: int, ctx: CrossContext | None = None) -> float:
        a, b = self.left.evaluate(bars, i, ctx), self.right.evaluate(bars, i, ctx)
        if self.op == "add":
            return a + b
        if self.op == "sub":
            return a - b
        if self.op == "mul":
            return a * b
        if self.op == "div":
            # Total, not raising. A division by zero mid-search should kill the *factor*, not the
            # run — and returning 0 is the neutral signal rather than a silently large one.
            return a / b if abs(b) > 1e-12 else 0.0
        if self.op == "gt":
            return 1.0 if a > b else 0.0
        if self.op == "lt":
            return 1.0 if a < b else 0.0
        if self.op == "and":
            return 1.0 if (a > 0 and b > 0) else 0.0
        return 1.0 if (a > 0 or b > 0) else 0.0

    def canonical(self) -> str:
        a, b = self.left.canonical(), self.right.canonical()
        if self.op in _COMMUTATIVE and b < a:
            a, b = b, a          # add(x,y) and add(y,x) are one factor, not two trials
        return f"{self.op}({a},{b})"


@dataclass(frozen=True, slots=True)
class UnOp(Expr):
    op: str
    operand: Expr

    OPS = ("neg", "abs", "sign", "not")

    def __post_init__(self) -> None:
        if self.op not in self.OPS:
            raise GrammarError(f"unknown op {self.op!r}; allowed: {self.OPS}")
        want = Kind.BOOLEAN if self.op == "not" else Kind.NUMBER
        if self.operand.kind is not want:
            raise GrammarError(f"{self.op} needs a {want}")

    @property
    def kind(self) -> Kind:
        return Kind.BOOLEAN if self.op == "not" else Kind.NUMBER

    @property
    def children(self) -> tuple[Expr, ...]:
        return (self.operand,)

    def evaluate(self, bars: Sequence[Any], i: int, ctx: CrossContext | None = None) -> float:
        v = self.operand.evaluate(bars, i, ctx)
        if self.op == "neg":
            return -v
        if self.op == "abs":
            return abs(v)
        if self.op == "sign":
            return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
        return 0.0 if v > 0 else 1.0

    def canonical(self) -> str:
        return f"{self.op}({self.operand.canonical()})"


@dataclass(frozen=True, slots=True)
class IfElse(Expr):
    """The node that makes session-conditional factors expressible."""

    condition: Expr
    then: Expr
    otherwise: Expr

    def __post_init__(self) -> None:
        if self.condition.kind is not Kind.BOOLEAN:
            raise GrammarError("the condition must be a boolean")
        if self.then.kind is not self.otherwise.kind:
            raise GrammarError("both branches must have the same kind")

    @property
    def kind(self) -> Kind:
        return self.then.kind

    @property
    def children(self) -> tuple[Expr, ...]:
        return (self.condition, self.then, self.otherwise)

    def evaluate(self, bars: Sequence[Any], i: int, ctx: CrossContext | None = None) -> float:
        chosen = self.then if self.condition.evaluate(bars, i, ctx) > 0 else self.otherwise
        return chosen.evaluate(bars, i, ctx)

    def canonical(self) -> str:
        return (f"if({self.condition.canonical()},{self.then.canonical()},"
                f"{self.otherwise.canonical()})")


@dataclass(frozen=True, slots=True)
class Signal(Expr):
    """The root of a complete factor. Clamps to [-1, 1].

    A factor must terminate here, which is what stops a proposer returning an unbounded number as
    a position size — the grammar's one economic constraint, and it is structural.
    """

    operand: Expr

    def __post_init__(self) -> None:
        if self.operand.kind is Kind.SIGNAL:
            raise GrammarError("a signal cannot wrap a signal")

    @property
    def kind(self) -> Kind:
        return Kind.SIGNAL

    @property
    def children(self) -> tuple[Expr, ...]:
        return (self.operand,)

    def evaluate(self, bars: Sequence[Any], i: int, ctx: CrossContext | None = None) -> float:
        return max(-1.0, min(1.0, self.operand.evaluate(bars, i, ctx)))

    def canonical(self) -> str:
        return f"signal({self.operand.canonical()})"


# --- validation and helpers --------------------------------------------------------------------

MAX_DEPTH = 12
MAX_SIZE = 96

MAX_COST = 100_000
"""Upper bound on node evaluations one ``evaluate(bars, i)`` call may perform.

At the measured 0.39 to 0.61 us per node evaluation that is 40 to 60 ms per bar, so a factor swept
over the 1,439 hourly bars of a 60-day series is bounded at one to one and a half minutes. Every
tree this repository ships (`ORIGINAL_EIGHT`, `EXPANDED`, the cross-sectional and divergence
factors) costs under 300. The bound exists for the tree nobody meant to write — see the module
docstring for the one that passed `validate` before this existed. CEL's analogue is a runtime
iteration budget (``comprehension_max_iterations = 10000``, ``google/cel-cpp
runtime/runtime_options.h:83``); a static bound is possible here because every lookback is a
literal in the tree.
"""


def cost(expr: Expr) -> int:
    """An upper bound on node evaluations for one ``expr.evaluate(bars, i)`` call.

    Read off the real ``evaluate`` methods above, not assumed:

    * ``Window`` evaluates its operand once per bar of its lookback: ``1 + lookback * cost(x)``.
    * ``Corr`` does that for both legs: ``1 + lookback * (cost(a) + cost(b))``.
    * ``Delay`` evaluates its operand **once**, at a shifted index: ``1 + cost(x)``. (The search's
      own ``searchoff.node_cost`` multiplies by the lookback here, which over-estimates; that is a
      research budget and is left as it is.)
    * ``IfElse`` evaluates the condition and one branch: ``1 + cost(c) + max(cost(t), cost(o))``.
    * A cross-sectional node is a lookup at evaluation time, but the panel evaluates its operand
      for every symbol at every bar to build the context, so the operand's cost is charged here.

    A node type this function does not know is charged ``1 + sum(children)``, the cost of a node
    that evaluates each child once — the shape of every other node in the grammar.
    """
    if isinstance(expr, Window):
        return 1 + expr.lookback * cost(expr.operand)
    if isinstance(expr, Corr):
        return 1 + expr.lookback * (cost(expr.left) + cost(expr.right))
    if isinstance(expr, IfElse):
        return 1 + cost(expr.condition) + max(cost(expr.then), cost(expr.otherwise))
    return 1 + sum(cost(c) for c in expr.children)


def validate(
    expr: Expr,
    *,
    max_depth: int = MAX_DEPTH,
    max_size: int = MAX_SIZE,
    max_cost: int = MAX_COST,
) -> Signal:
    """Check a proposed factor before it is ever evaluated.

    Depth and size limits are a research control rather than a safety one: an arbitrarily deep
    tree is a way to overfit by construction, and capping depth keeps the hypothesis legible enough
    for a human to argue with. The cost limit is a safety control: without it a six-node tree can
    occupy the evaluator for hours per bar (see :data:`MAX_COST`). Injection safety comes from the
    grammar having no execution surface at all.
    """
    if not isinstance(expr, Signal):
        raise GrammarError(
            f"a factor must be rooted in Signal, got {type(expr).__name__}. An unbounded number is "
            f"not a position size."
        )
    if expr.depth > max_depth:
        raise GrammarError(f"depth {expr.depth} exceeds {max_depth}")
    if expr.size > max_size:
        raise GrammarError(f"size {expr.size} exceeds {max_size}")
    estimated = cost(expr)
    if estimated > max_cost:
        raise GrammarError(
            f"cost {estimated} node evaluations per bar exceeds {max_cost}; nested windows "
            f"multiply, so this tree would occupy the evaluator far longer than any factor needs"
        )
    return expr


def as_signal_fn(expr: Signal, *, ctx: CrossContext | None = None) -> Any:
    """Adapt a validated tree to the `(bars, i) -> float` shape the evaluator already runs.

    ``ctx`` is bound here rather than passed per call because the single-symbol evaluator's
    signature is fixed by every caller that already exists. A cross-sectional expression reaches
    this path only through :mod:`argus.research.panel`, which builds the context first and binds it
    per symbol; a caller that adapts one without a context gets the refusal from
    :meth:`CrossRank.evaluate`, not a quietly wrong number.
    """
    def _fn(bars: Sequence[Any], i: int) -> float:
        return expr.evaluate(bars, i, ctx)
    return _fn


def _extra(bar: Any, key: str, *, default: float = 0.0) -> float:
    """Read an optional per-bar field the caller may not have carried.

    Returns ``default`` rather than raising: a caller that built bars without volume should get a
    factor that cannot use volume, not a crash in the middle of a sweep. The default for a price
    field is the close, so a missing high or low degenerates to a zero range rather than to a
    price of zero.
    """
    # Direct attribute access, never `getattr`. `tests/test_grammar.py` parses this module and
    # fails if the grammar can reach any introspection or execution builtin, and `getattr` is on
    # that list for a good reason: a grammar that can name an attribute dynamically is a grammar
    # that can read one it was never meant to. The rule is absolute, so this reads the field it
    # wants by name and treats its absence as absence.
    try:
        extra = bar.extra
    except AttributeError:
        return default
    if not isinstance(extra, dict):
        return default
    value = extra.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ret(bars: Sequence[Any], i: int, n: int) -> float:
    j = i - n
    if j < 0:
        return 0.0
    a, b = float(bars[j].close), float(bars[i].close)
    return (b - a) / a if a > 0 else 0.0


def _basis_bps(bar: Any) -> float:
    """Read the basis off a bar, without dynamic attribute access.

    This used `getattr(bar, "extra", None)`, which an AST audit of this module correctly flagged:
    `getattr` is the one call here that could take a *computed* name, and a grammar whose whole
    claim is "no execution surface" cannot make an exception for convenience. A direct access
    inside `try` has the same tolerance and no such hole.
    """
    try:
        extra = bar.extra or {}
        return float(Decimal(str(extra.get("basis_bps", 0))))
    except (ArithmeticError, AttributeError, TypeError, ValueError):
        return 0.0


# --- the eight originals, now expressible ------------------------------------------------------
# Proof that the grammar is a superset rather than a replacement: every hand-written primitive is
# reconstructed here as a tree. If one of these could not be written, the grammar would be a
# narrowing dressed as a widening.

_CLOSED = UnOp("not", HasDiscovery())

# **Two of these are not faithful reproductions, and deliberately so.** `closure_momentum` and
# `closure_reversion` as originally written fall through to a full +/-1.0 position when the
# lookback window has no history, because `_ret` returns 0.0 and `0.0 > 0` is False. A momentum
# factor that takes a maximum short on six bars of missing data is a bug, not a convention. The
# grammar versions return flat. They are identical from the first bar where history exists.
ORIGINAL_EIGHT: dict[str, Signal] = {
    "long_while_closed": Signal(IfElse(_CLOSED, Const(1.0), Const(0.0))),
    "long_while_open": Signal(IfElse(HasDiscovery(), Const(1.0), Const(0.0))),
    "weekend_only": Signal(IfElse(InPhase(SessionPhase.WEEKEND), Const(1.0), Const(0.0))),
    "closure_momentum": Signal(IfElse(
        _CLOSED, UnOp("sign", Return(6)), Const(0.0))),
    "closure_reversion": Signal(IfElse(
        _CLOSED, UnOp("neg", UnOp("sign", Return(6))), Const(0.0))),
    "near_reopen": Signal(IfElse(
        BinOp("and", _CLOSED, BinOp("lt", Ref(Field.HOURS_TO_DISCOVERY), Const(4.0))),
        Const(1.0), Const(0.0))),
    "slow_trend": Signal(IfElse(
        BinOp("gt", Return(48), Const(0.0)), Const(1.0), Const(0.0))),
    "slow_fade": Signal(IfElse(
        BinOp("gt", Return(48), Const(0.0)), Const(-1.0), Const(1.0))),
}

# Factors the original eight could not express, one per new capability.
#
# Each is here to test a *capability*, not because anyone believes it works — the anti-overfit
# gates in `research/overfit.py` and the Deflated Sharpe in the Track 1 study decide that, and on
# the first eight they decided no. What the audit established is that the eight could not even ask
# most of the interesting questions: nothing about volume, nothing volatility-adjusted, nothing
# about where a price sits in its own recent distribution, and no change of anything except the
# built-in return.
#
# (The docstring that was briefly here sat *inside* the dict literal, where Python concatenated it
# onto the first key — a nine-entry table whose first factor was named after its own documentation.
# It parsed, imported and would have run. Caught by reading the keys back.)
EXPANDED: dict[str, Signal] = {
    # Volatility-adjusted momentum: the audit's named #1 gap. A 2% move in a calm week and a 2%
    # move in a violent one are different events, and only a z-score tells them apart.
    "vol_adjusted_momentum": Signal(IfElse(
        BinOp("gt", Window("zscore", 48, Ref(Field.CLOSE)), Const(1.0)),
        Const(1.0), Const(0.0))),
    "vol_adjusted_reversion": Signal(IfElse(
        BinOp("gt", Window("zscore", 48, Ref(Field.CLOSE)), Const(1.5)),
        Const(-1.0), Const(0.0))),
    # Where the price sits in its own recent range, which no aggregate could previously say.
    "range_breakout": Signal(IfElse(
        BinOp("gt", Window("rank", 96, Ref(Field.CLOSE)), Const(0.95)),
        Const(1.0), Const(0.0))),
    # Recency of the high. A window whose peak was long ago is a different state from one that
    # just made it, and every aggregate treats them identically.
    "stale_high_fade": Signal(IfElse(
        BinOp("gt", Window("argmax", 96, Ref(Field.CLOSE)), Const(72.0)),
        Const(-1.0), Const(0.0))),
    # Trend by least squares rather than by endpoint difference: robust to a single noisy close.
    "fitted_trend": Signal(IfElse(
        BinOp("gt", Window("slope", 48, Ref(Field.CLOSE)), Const(0.0)),
        Const(1.0), Const(-1.0))),
    # Delta, which needed `delay` to exist at all.
    "delta_momentum": Signal(IfElse(
        BinOp("gt", BinOp("sub", Ref(Field.CLOSE), Delay(24, Ref(Field.CLOSE))), Const(0.0)),
        Const(1.0), Const(0.0))),
    # Price-volume agreement. A rise the volume confirms is a different event from one it does
    # not, and this is the first expression in the grammar that can say so.
    "volume_confirmed_trend": Signal(IfElse(
        BinOp("gt", Corr(48, Ref(Field.CLOSE), Ref(Field.VOLUME)), Const(0.3)),
        Const(1.0), Const(0.0))),
    # Intrabar range as a regime filter: trade the trend only when the tape is calm.
    "calm_tape_trend": Signal(IfElse(
        BinOp(
            "and",
            BinOp("lt", Ref(Field.RANGE_BPS), Window("mean", 96, Ref(Field.RANGE_BPS))),
            BinOp("gt", Return(48), Const(0.0)),
        ),
        Const(1.0), Const(0.0))),
    # Volume shock while the anchor sleeps — the one that is most specific to this venue.
    "overnight_volume_shock": Signal(IfElse(
        BinOp(
            "and",
            _CLOSED,
            BinOp("gt", Window("zscore", 96, Ref(Field.VOLUME)), Const(2.0)),
        ),
        UnOp("sign", Return(6)), Const(0.0))),
}


__all__ = [
    "EXPANDED",
    "MAX_COST",
    "MAX_DEPTH",
    "MAX_SIZE",
    "ORIGINAL_EIGHT",
    "BinOp",
    "Const",
    "Corr",
    "CrossContext",
    "CrossRank",
    "CrossScale",
    "Delay",
    "Expr",
    "Field",
    "GrammarError",
    "HasDiscovery",
    "IfElse",
    "InPhase",
    "Kind",
    "Ref",
    "Return",
    "Signal",
    "UnOp",
    "Window",
    "as_signal_fn",
    "cost",
    "validate",
]
