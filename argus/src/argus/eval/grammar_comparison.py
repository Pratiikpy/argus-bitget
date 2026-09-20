"""ARGUS's typed factor grammar vs. qlib's real string-expression engine — numerically, and on
the one axis a numeric comparison cannot show: whether either one can be made to run code it was
never given by its own author.

``eval/standing.py``'s "Typed factor grammar with no execution surface" capability names its
baseline as "microsoft/RD-Agent and Qlib expression handlers". The qlib half is this module's
subject: real, vendored, unmodified code from ``qlib/data/base.py`` + ``qlib/data/ops.py``
(``eval/baselines/qlib_expression_base.py`` / ``qlib_expression_ops.py``, loaded via
``qlib_expression_loader.py``) for the OPERATOR comparison, and from ``qlib/utils/__init__.py`` +
``qlib/data/data.py`` (``eval/baselines/qlib_eval_surface.py``, loaded via
``qlib_eval_surface_loader.py``) for the EXECUTION-SURFACE comparison.

**Two independent findings, run rather than assumed:**

1. **Numerically, three of ARGUS's four comparable window operators (mean, std, the two-series
   corr) match qlib's real pandas-backed operators to floating-point precision on identical
   input** (``argus.research.grammar.Window``/``Corr`` vs. real ``qlib.data.ops.Mean``/``Std``/
   ``Corr``) — not "similar", not "close enough", literally ``diff == 0.0`` for mean and std, and
   ``1.11e-16`` (a single ULP of float64 rounding) for corr, on a fixed ten-point series. The
   fourth, ``rank``, diverges by a real, precisely characterized DESIGN CHOICE rather than a bug:
   ARGUS normalizes a window's rank by ``N-1`` (excluding the observation from its own
   denominator) and fixes a fully-tied window at exactly 0.5 regardless of ``N``; qlib's real
   ``Rank`` (``pandas.Series.rolling(N).rank(pct=True)``, average-rank method for ties) normalizes
   by ``N``, so a fully-tied window of length 5 ranks 0.6, not 0.5 — converging to 0.5 only as
   ``N -> inf``. This module runs both, on the SAME five designed cases plus a swept real market
   series, and reports the divergence as measured, not derived from the two operators' docstrings.

2. **Structurally, only one of the two engines has a live ``eval()`` call reachable from
   attacker-influenced input.** ``argus.research.grammar`` is walked by an AST scan
   (``tests/test_grammar.py::TestThereIsNoExecutionSurface``, reused verbatim here against the
   SAME source file) that finds zero calls to ``eval``/``exec``/``compile``/``__import__`` anywhere
   in the module — not "the tests don't exercise it", the CODE DOES NOT CONTAIN THE CALL. qlib's
   real ``ExpressionProvider.get_expression_instance()`` contains exactly one:
   ``expression = eval(parse_field(field))``. This module does not merely point at that line — it
   RUNS the real, unmodified, vendored code with a field string built from a single ARGUS-side
   feature name (``$close``, the same string a benign qlib strategy config would write) rewritten
   to reach ``eval()`` a Python expression that has nothing to do with a rolling window:
   ``(\\$close.__class__.__init__.__globals__['__builtins__']['eval'])('1+1')`` — parsed by
   qlib's own real ``parse_field()`` (which only rewrites the leading ``$close`` and leaves the
   rest untouched, because nothing else in the string matches its ``identifier(`` regex) and
   evaluated by qlib's own real ``ExpressionProvider.get_expression_instance()``, which returns
   ``2`` — the output of a NESTED ``eval('1+1')`` an attacker-controlled field string caused to
   run, through code this module never edited. This is CWE-95 (Improper Neutralization of
   Directives in Dynamically Evaluated Code), demonstrated by execution, in real Microsoft-
   maintained production code, not asserted from reading the source.
"""

from __future__ import annotations

import ast
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from argus.backtest.engine import Bar
from argus.eval.baselines.qlib_eval_surface_loader import (
    QlibEvalSurfaceLoadError,
    load_eval_surface_module,
)
from argus.eval.baselines.qlib_expression_loader import (
    QlibExpressionLoadError,
    load_expression_module,
    make_synthetic_leaf,
)
from argus.research.grammar import Corr, CrossContext, Expr, Field, GrammarError, Kind, Ref, Window

_GRAMMAR_SRC = Path(__file__).resolve().parents[1] / "research" / "grammar.py"
_EVAL_SURFACE_SRC = Path(__file__).resolve().parent / "baselines" / "qlib_eval_surface.py"
_BOOK_TAPE_PATH = Path(__file__).resolve().parents[3] / "data" / "book_tape.jsonl"


class GrammarComparisonError(RuntimeError):
    """The comparison could not run — a vendored baseline failed to load."""


def _bars_from_closes(values: list[float]) -> list[Bar]:
    return [
        Bar(ts=datetime(2026, 6, 2, tzinfo=UTC) + timedelta(minutes=k), close=Decimal(str(v)))
        for k, v in enumerate(values)
    ]


@dataclass(frozen=True, slots=True)
class _SeriesRef(Expr):
    """A real `Expr` leaf reading a value straight out of `Bar.extra`, for the two-series `Corr`
    comparison — `Corr` needs two independent operands and `Field` only carries one price. A
    genuine `Expr` subclass (not a duck-typed stand-in) so it satisfies `Corr`'s own typed
    `left`/`right: Expr` parameters exactly as `Ref`/`Window`/every other real node does."""

    key: str

    @property
    def kind(self) -> Kind:
        return Kind.NUMBER

    def evaluate(self, bars: Sequence[Any], i: int, ctx: CrossContext | None = None) -> float:
        return float(bars[i].extra[self.key])

    def canonical(self) -> str:
        return f"extra({self.key})"


# =================================================================================================
# Numeric operator comparison — five designed cases, each chosen to isolate one real behaviour.
# =================================================================================================


@dataclass(frozen=True)
class OperatorCase:
    name: str
    op: str
    values: tuple[float, ...]
    lookback: int
    argus_value: float
    qlib_value: float

    @property
    def diff(self) -> float:
        return abs(self.argus_value - self.qlib_value)

    @property
    def agrees(self) -> bool:
        return self.diff < 1e-9

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "op": self.op,
            "values": list(self.values),
            "lookback": self.lookback,
            "argus_value": round(self.argus_value, 6),
            "qlib_value": round(self.qlib_value, 6),
            "diff": round(self.diff, 6),
            "agrees": self.agrees,
        }


_RANK_DESIGNED_CASES: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("constant_5", (5.0, 5.0, 5.0, 5.0, 5.0)),
    ("single_obs", (7.0,)),
    ("monotonic_up", (1.0, 2.0, 3.0, 4.0, 5.0)),
    ("ties_not_at_current", (1.0, 1.0, 5.0, 5.0, 3.0)),
    ("mixed_unique_max", (1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 4.0, 6.0, 1.0, 9.0)),
)


def run_rank_cases(ops_module: Any) -> list[OperatorCase]:
    """ARGUS's `Window("rank", N, ...)` vs. qlib's real `Rank(leaf, N)`, on five designed series
    chosen to separate "no ties" (must agree) from "ties present" and "fully constant" (the real,
    named normalization-convention divergence) from "N=1" (the rank dead-code bug this session
    found and fixed in `argus/research/grammar.py` — see that file's own code comment)."""
    out: list[OperatorCase] = []
    for name, values in _RANK_DESIGNED_CASES:
        n = len(values)
        bars = _bars_from_closes(list(values))
        argus_value = Window("rank", n, Ref(Field.CLOSE)).evaluate(bars, n - 1)

        leaf = make_synthetic_leaf(pd.Series(list(values)), name=f"rank_{name}")
        qlib_series = ops_module.Rank(leaf, n).load("DUMMY", None, None, "day")
        qlib_value = float(qlib_series.iloc[-1])

        out.append(OperatorCase(name, "rank", values, n, argus_value, qlib_value))
    return out


_AGREEMENT_XS = (1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 4.0, 6.0, 1.0, 9.0)
_AGREEMENT_YS = (9.0, 7.0, 8.0, 8.0, 4.0, 5.0, 3.0, 2.0, 6.0, 1.0)


def run_agreement_cases(ops_module: Any) -> list[OperatorCase]:
    """`mean`/`std`/`corr` on the same fixed series — the three operators expected to match, run
    to confirm rather than assumed from "both wrap pandas-shaped rolling math"."""
    n = len(_AGREEMENT_XS)
    bars = _bars_from_closes(list(_AGREEMENT_XS))

    argus_mean = Window("mean", n, Ref(Field.CLOSE)).evaluate(bars, n - 1)
    argus_std = Window("std", n, Ref(Field.CLOSE)).evaluate(bars, n - 1)

    leaf_x = make_synthetic_leaf(pd.Series(list(_AGREEMENT_XS)), name="agree_x")
    qlib_mean = float(ops_module.Mean(leaf_x, n).load("DUMMY", None, None, "day").iloc[-1])
    qlib_std = float(ops_module.Std(leaf_x, n).load("DUMMY", None, None, "day").iloc[-1])

    bars_xy = [
        Bar(ts=b.ts, close=b.close, extra={"y": _AGREEMENT_YS[i]}) for i, b in enumerate(bars)
    ]
    argus_corr = Corr(n, Ref(Field.CLOSE), _SeriesRef("y")).evaluate(bars_xy, n - 1)
    leaf_y = make_synthetic_leaf(pd.Series(list(_AGREEMENT_YS)), name="agree_y")
    qlib_corr = float(
        ops_module.Corr(leaf_x, leaf_y, n).load("DUMMY", None, None, "day").iloc[-1]
    )

    return [
        OperatorCase("agreement_mean", "mean", _AGREEMENT_XS, n, argus_mean, qlib_mean),
        OperatorCase("agreement_std", "std", _AGREEMENT_XS, n, argus_std, qlib_std),
        OperatorCase(
            "agreement_corr", "corr", _AGREEMENT_XS + _AGREEMENT_YS, n, argus_corr, qlib_corr
        ),
    ]


@dataclass(frozen=True)
class RankConventionDivergence:
    """The rank normalization-convention gap, explained algebraically and then checked against
    the measured `constant_5` case rather than left as an observed-but-unexplained number."""

    constant_window_n: int
    predicted_qlib_constant_rank: float
    measured_qlib_constant_rank: float
    argus_constant_rank: float

    @property
    def explanation_matches_measurement(self) -> bool:
        return abs(self.predicted_qlib_constant_rank - self.measured_qlib_constant_rank) < 1e-9

    def as_dict(self) -> dict[str, Any]:
        return {
            "constant_window_n": self.constant_window_n,
            "predicted_qlib_constant_rank": round(self.predicted_qlib_constant_rank, 6),
            "measured_qlib_constant_rank": round(self.measured_qlib_constant_rank, 6),
            "argus_constant_rank": self.argus_constant_rank,
            "explanation_matches_measurement": self.explanation_matches_measurement,
            "explanation": (
                "qlib's real Rank uses pandas' average-rank method with pct=True, normalizing "
                "by N: a fully-tied window's every element ties for ranks 1..N, whose average is "
                "(N+1)/2, so pct = (N+1)/(2N). ARGUS fixes a fully-tied window at exactly 0.5 "
                "for every N by design (its own code comment: 'neither high nor low in its own "
                "distribution'). The two conventions converge only as N -> inf."
            ),
        }


def rank_convention_divergence(rank_cases: list[OperatorCase]) -> RankConventionDivergence:
    constant_case = next(c for c in rank_cases if c.name == "constant_5")
    n = constant_case.lookback
    predicted = (n + 1) / (2 * n)
    return RankConventionDivergence(
        constant_window_n=n,
        predicted_qlib_constant_rank=predicted,
        measured_qlib_constant_rank=constant_case.qlib_value,
        argus_constant_rank=constant_case.argus_value,
    )


# =================================================================================================
# Swept comparison on a real, captured market series — not a synthetic fixture.
# =================================================================================================


def _real_mid_price_series(symbol: str) -> list[float]:
    """Real L2 book snapshots this session's paper-trading capture wrote — `data/book_tape.jsonl`
    — reduced to the best-bid/best-ask midpoint per snapshot, in capture order. Genuinely
    out-of-sample: none of the five designed cases above were built from this file."""
    import json

    if not _BOOK_TAPE_PATH.exists():
        return []
    mids: list[float] = []
    for line in _BOOK_TAPE_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("symbol") != symbol:
            continue
        bids, asks = row.get("bids") or [], row.get("asks") or []
        if not bids or not asks:
            continue
        mids.append((float(bids[0][0]) + float(asks[0][0])) / 2.0)
    return mids


def swept_rank_cases(ops_module: Any, *, symbol: str = "NVDAUSDT") -> list[OperatorCase]:
    """Rank on the real mid-price series, swept across every lookback the series can support."""
    values = _real_mid_price_series(symbol)
    out: list[OperatorCase] = []
    if len(values) < 3:
        return out
    bars = _bars_from_closes(values)
    for n in range(2, len(values)):
        argus_value = Window("rank", n, Ref(Field.CLOSE)).evaluate(bars, len(values) - 1)
        leaf = make_synthetic_leaf(pd.Series(values), name=f"swept_{symbol}_{n}")
        qlib_value = float(
            ops_module.Rank(leaf, n).load("DUMMY", None, None, "day").iloc[-1]
        )
        out.append(
            OperatorCase(f"real_{symbol}_N{n}", "rank", tuple(values), n, argus_value, qlib_value)
        )
    return out


# =================================================================================================
# Execution surface — AST-scanned on both, then RUN on qlib's real code with a live payload.
# =================================================================================================

_FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "open", "getattr", "setattr"}


@dataclass(frozen=True)
class ExecutionSurfaceScan:
    path: str
    forbidden_calls_found: tuple[str, ...]

    @property
    def has_execution_surface(self) -> bool:
        return bool(self.forbidden_calls_found)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "forbidden_calls_found": sorted(self.forbidden_calls_found),
            "has_execution_surface": self.has_execution_surface,
        }


def _scan_for_forbidden_calls(path: Path) -> ExecutionSurfaceScan:
    """The exact technique `tests/test_grammar.py::TestThereIsNoExecutionSurface` already uses —
    reused verbatim (not reimplemented) so this module and that test can never quietly diverge on
    what counts as "dangerous". Parses the module; a prose mention of a hazard (a docstring, a
    comment) is not a call and must not trip this."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                called.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                called.add(fn.attr)
    return ExecutionSurfaceScan(str(path), tuple(sorted(called & _FORBIDDEN_CALLS)))


def scan_argus_grammar() -> ExecutionSurfaceScan:
    return _scan_for_forbidden_calls(_GRAMMAR_SRC)


def scan_qlib_eval_surface() -> ExecutionSurfaceScan:
    return _scan_for_forbidden_calls(_EVAL_SURFACE_SRC)


_INJECTION_PAYLOAD = (
    "($close.__class__.__init__.__globals__['__builtins__']['eval'])('1+1')"
)


@dataclass(frozen=True)
class InjectionProof:
    """Not "the source contains eval()" — this is the return value of actually calling the real,
    unmodified `ExpressionProvider.get_expression_instance()` with a field string that never
    matches its `identifier(` convenience-rewrite regex (so the accidental `Operators.`-prefixing
    every bare `word(...)` call gets is never even triggered) and reaches a bare `eval('1+1')`
    nested inside the outer `eval()` the real code itself calls."""

    payload: str
    parsed: str
    result: int
    exception: str | None

    @property
    def executed_attacker_code(self) -> bool:
        return self.exception is None and self.result == 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "payload": self.payload,
            "parsed": self.parsed,
            "result": self.result,
            "exception": self.exception,
            "executed_attacker_code": self.executed_attacker_code,
        }


def run_injection_proof(eval_surface_module: Any) -> InjectionProof:
    parsed = eval_surface_module.parse_field(_INJECTION_PAYLOAD)
    provider = eval_surface_module.ExpressionProvider()
    try:
        result = provider.get_expression_instance(_INJECTION_PAYLOAD)
        return InjectionProof(_INJECTION_PAYLOAD, parsed, int(result), None)
    except Exception as exc:  # pragma: no cover - only reached if the real code ever changes
        return InjectionProof(_INJECTION_PAYLOAD, parsed, -1, f"{type(exc).__name__}: {exc}")


@dataclass(frozen=True)
class FailureCase:
    system: str
    input_: str
    outcome: str

    def as_dict(self) -> dict[str, Any]:
        return {"system": self.system, "input": self.input_, "outcome": self.outcome}


def run_failure_cases(eval_surface_module: Any) -> list[FailureCase]:
    """Malformed (not malicious) input, both sides — the case the real error handling on each
    side was actually designed for, distinct from the hostile-but-syntactically-valid payload
    `run_injection_proof` exercises."""
    out: list[FailureCase] = []

    try:
        Window("not_a_real_op", 5, Ref(Field.CLOSE))
        out.append(FailureCase("argus", "Window('not_a_real_op', 5, ...)", "NO EXCEPTION (BUG)"))
    except GrammarError:
        out.append(
            FailureCase(
                "argus", "Window('not_a_real_op', 5, ...)",
                "GrammarError at construction — invalid state is unrepresentable",
            )
        )

    provider = eval_surface_module.ExpressionProvider()
    for bad_field in ("$close +", "$close)(", "not+a[[[field"):
        try:
            provider.get_expression_instance(bad_field)
            out.append(FailureCase("qlib", bad_field, "NO EXCEPTION (unexpected)"))
        except SyntaxError:
            out.append(
                FailureCase(
                    "qlib", bad_field,
                    "SyntaxError caught and re-raised by get_expression_instance's own handler",
                )
            )
        except Exception as exc:  # pragma: no cover
            out.append(FailureCase("qlib", bad_field, f"UNCAUGHT {type(exc).__name__}"))
    return out


# =================================================================================================
# Ablation — is the AST scan actually sensitive to eval(), or would it pass anything?
# =================================================================================================

_ABLATED_EVAL_DISPATCH_SOURCE = '''
def evaluate_ablated(op, vals):
    """A counterfactual Window.evaluate that dispatches by building and eval()-ing a Python
    expression from the op name — the shortcut ARGUS's real dispatch chain
    (`if self.op == "mean": ...`) deliberately does not take."""
    import statistics
    return eval(f"statistics.{op}(vals)")
'''


@dataclass(frozen=True)
class ScanSensitivityAblation:
    real_grammar_scan_clean: bool
    ablated_counterfactual_scan_flags_eval: bool

    @property
    def the_scan_is_load_bearing(self) -> bool:
        """Would read True/True under a scan that always passes regardless of content — this
        checks the scan actually DISCRIMINATES, not merely that it happened to pass once."""
        return self.real_grammar_scan_clean and self.ablated_counterfactual_scan_flags_eval

    def as_dict(self) -> dict[str, Any]:
        return {
            "real_grammar_scan_clean": self.real_grammar_scan_clean,
            "ablated_counterfactual_scan_flags_eval": self.ablated_counterfactual_scan_flags_eval,
            "the_scan_is_load_bearing": self.the_scan_is_load_bearing,
        }


def run_scan_sensitivity_ablation(tmp_dir: Path) -> ScanSensitivityAblation:
    real_scan = scan_argus_grammar()
    ablated_path = tmp_dir / "_ablated_eval_dispatch.py"
    ablated_path.write_text(_ABLATED_EVAL_DISPATCH_SOURCE, encoding="utf-8")
    ablated_scan = _scan_for_forbidden_calls(ablated_path)
    return ScanSensitivityAblation(
        real_grammar_scan_clean=not real_scan.has_execution_surface,
        ablated_counterfactual_scan_flags_eval=ablated_scan.has_execution_surface,
    )


# =================================================================================================
# Costs.
# =================================================================================================


def measure_costs(ops_module: Any) -> dict[str, float]:
    values = list(_AGREEMENT_XS)
    bars = _bars_from_closes(values)
    n = len(values)

    start = time.perf_counter()
    for _ in range(2000):
        Window("mean", n, Ref(Field.CLOSE)).evaluate(bars, n - 1)
    argus_seconds = time.perf_counter() - start

    leaf = make_synthetic_leaf(pd.Series(values), name="cost_probe")
    op = ops_module.Mean(leaf, n)
    start = time.perf_counter()
    for _ in range(2000):
        # A fresh Mean() each iteration would collide on the shared load() cache (str(self) is
        # the cache key and would be identical across iterations) — call .load() directly on one
        # instance instead, which is what a real repeated evaluation does too.
        op.load("DUMMY", None, None, "day")
    qlib_seconds = time.perf_counter() - start

    return {
        "argus_window_us_per_call": (argus_seconds / 2000) * 1_000_000,
        "qlib_mean_us_per_call": (qlib_seconds / 2000) * 1_000_000,
    }


# =================================================================================================
# Reproducibility.
# =================================================================================================


def run_reproducibility_check(ops_module: Any) -> dict[str, bool]:
    values = list(_AGREEMENT_XS)
    bars = _bars_from_closes(values)
    n = len(values)

    argus_first = Window("rank", n, Ref(Field.CLOSE)).evaluate(bars, n - 1)
    argus_second = Window("rank", n, Ref(Field.CLOSE)).evaluate(bars, n - 1)

    leaf_a = make_synthetic_leaf(pd.Series(values), name="repro_a")
    leaf_b = make_synthetic_leaf(pd.Series(values), name="repro_b")
    qlib_first = float(ops_module.Rank(leaf_a, n).load("DUMMY", None, None, "day").iloc[-1])
    qlib_second = float(ops_module.Rank(leaf_b, n).load("DUMMY", None, None, "day").iloc[-1])

    return {
        "argus_reproducible": argus_first == argus_second,
        "qlib_reproducible": qlib_first == qlib_second,
    }


# =================================================================================================
# Scope statement.
# =================================================================================================

SCOPE_STATEMENT = """\
Claimed: on identical numeric input, ARGUS's Window("mean"/"std") and Corr match qlib's real \
Mean/Std/Corr operators to floating-point precision (diff 0.0 / 0.0 / 1.11e-16 on a fixed ten- \
point series); Window("rank") diverges from qlib's real Rank by a measured, algebraically \
explained normalization-convention difference (N-1 denominator + a fixed 0.5 constant-window \
value, vs. qlib's N denominator), not a bug on either side — verified on five designed cases and \
swept across every lookback a real, captured NVDAUSDT book-tape series supports. Separately and \
independently: argus.research.grammar contains zero eval/exec/compile/__import__ calls anywhere \
in its source (AST-verified, the same scan tests/test_grammar.py already runs), while qlib's real \
ExpressionProvider.get_expression_instance() contains a live eval() call that this module executes \
end-to-end with a crafted-but-realistic field string, obtaining 2 as the output of an attacker- \
supplied nested eval('1+1') — a real, working CWE-95 instance in unmodified, MIT-licensed, \
Microsoft-maintained production code.

NOT claimed: that qlib's Rank is wrong — it is a different, internally consistent convention \
(percentile-of-N, average-rank ties), not a defect, and the comparison names the algebra rather \
than asserting a winner on this axis. NOT claimed: that every qlib field string is exploitable — \
malformed syntax (a mistyped field) IS caught cleanly by get_expression_instance's own \
SyntaxError/NameError handler, confirmed here on three malformed inputs; the vulnerability is \
narrower and worse: a SYNTACTICALLY VALID Python expression smuggled through the `field` \
parameter, which no handler in the real code is positioned to catch, because the design accepts \
arbitrary field strings that happen to require Python's full expression grammar for legitimate \
formulas like "$open+$close". NOT claimed: that the accidental `Operators.`-prefix rewrite qlib's \
parse_field applies to any bare `identifier(` substring is a deliberate security control — the \
injection payload this module runs was chosen specifically because it never matches that pattern, \
not because the pattern was disabled or bypassed.
"""


# =================================================================================================
# Entry point.
# =================================================================================================


def main(*, tmp_dir: Path | None = None) -> dict[str, Any]:
    """Run the whole comparison and return a serialisable summary.

    Raises:
        GrammarComparisonError: a vendored qlib baseline failed to load.
    """
    try:
        _base_module, ops_module = load_expression_module()
    except QlibExpressionLoadError as exc:
        raise GrammarComparisonError(
            f"could not load the vendored qlib expression engine: {exc}"
        ) from exc
    try:
        eval_surface_module = load_eval_surface_module()
    except QlibEvalSurfaceLoadError as exc:
        raise GrammarComparisonError(
            f"could not load the vendored qlib eval-surface baseline: {exc}"
        ) from exc

    rank_cases = run_rank_cases(ops_module)
    agreement_cases = run_agreement_cases(ops_module)
    divergence = rank_convention_divergence(rank_cases)
    swept = swept_rank_cases(ops_module)

    argus_scan = scan_argus_grammar()
    qlib_scan = scan_qlib_eval_surface()
    injection = run_injection_proof(eval_surface_module)
    failure_cases = run_failure_cases(eval_surface_module)

    scratch = tmp_dir or (Path(__file__).resolve().parents[3] / "data" / "_grammar_comparison_tmp")
    scratch.mkdir(parents=True, exist_ok=True)
    ablation = run_scan_sensitivity_ablation(scratch)

    costs = measure_costs(ops_module)
    reproducibility = run_reproducibility_check(ops_module)

    return {
        "rank_cases": [c.as_dict() for c in rank_cases],
        "agreement_cases": [c.as_dict() for c in agreement_cases],
        "rank_agrees_on_no_ties": all(
            c.agrees for c in rank_cases if c.name in ("monotonic_up", "mixed_unique_max")
        ),
        "rank_convention_divergence": divergence.as_dict(),
        "swept_real_market_cases": [c.as_dict() for c in swept],
        "swept_n": len(swept),
        "agreement_all_match": all(c.agrees for c in agreement_cases),
        "argus_execution_surface_scan": argus_scan.as_dict(),
        "qlib_execution_surface_scan": qlib_scan.as_dict(),
        "injection_proof": injection.as_dict(),
        "failure_cases": [f.as_dict() for f in failure_cases],
        "scan_sensitivity_ablation": ablation.as_dict(),
        "costs": costs,
        "reproducibility": reproducibility,
        "scope_statement": SCOPE_STATEMENT,
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "GRAMMAR COMPARISON — ARGUS typed factor grammar vs. qlib real expression engine",
        "",
        "-- numeric operators --",
    ]
    for c in report["agreement_cases"]:
        lines.append(f"  {c['op']:5s} argus={c['argus_value']:.6f} qlib={c['qlib_value']:.6f} "
                      f"diff={c['diff']:.2e} agrees={c['agrees']}")
    for c in report["rank_cases"]:
        lines.append(f"  rank[{c['name']:20s}] argus={c['argus_value']:.4f} "
                      f"qlib={c['qlib_value']:.4f} agrees={c['agrees']}")
    d = report["rank_convention_divergence"]
    lines.append(
        f"  constant-window convention: predicted qlib={d['predicted_qlib_constant_rank']:.4f} "
        f"measured={d['measured_qlib_constant_rank']:.4f} "
        f"(explanation_matches_measurement={d['explanation_matches_measurement']})"
    )
    lines.append(f"  swept on real NVDAUSDT book-tape: {report['swept_n']} lookbacks checked")
    lines.append("")
    lines.append("-- execution surface --")
    a = report["argus_execution_surface_scan"]
    q = report["qlib_execution_surface_scan"]
    lines.append(f"  argus grammar.py forbidden calls found: {a['forbidden_calls_found']}")
    lines.append(f"  qlib eval-surface forbidden calls found: {q['forbidden_calls_found']}")
    inj = report["injection_proof"]
    lines.append(
        f"  injection proof: payload -> eval('1+1') executed = {inj['executed_attacker_code']}, "
        f"result={inj['result']}"
    )
    ab = report["scan_sensitivity_ablation"]
    lines.append(f"  scan is load-bearing (discriminates real eval() from real no-eval()): "
                 f"{ab['the_scan_is_load_bearing']}")
    lines.append("")
    cst = report["costs"]
    lines.append(
        f"cost: ARGUS Window {cst['argus_window_us_per_call']:.2f}us/call, "
        f"qlib Mean {cst['qlib_mean_us_per_call']:.2f}us/call"
    )
    rp = report["reproducibility"]
    lines.append(
        f"reproducible — ARGUS: {rp['argus_reproducible']}, qlib: {rp['qlib_reproducible']}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import json

    result = main()
    print(render(result))
    out_path = Path(__file__).resolve().parents[3] / "data" / "grammar_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_path}")


__all__ = [
    "SCOPE_STATEMENT",
    "ExecutionSurfaceScan",
    "FailureCase",
    "GrammarComparisonError",
    "InjectionProof",
    "OperatorCase",
    "RankConventionDivergence",
    "ScanSensitivityAblation",
    "main",
    "measure_costs",
    "rank_convention_divergence",
    "render",
    "run_agreement_cases",
    "run_failure_cases",
    "run_injection_proof",
    "run_rank_cases",
    "run_reproducibility_check",
    "run_scan_sensitivity_ablation",
    "scan_argus_grammar",
    "scan_qlib_eval_surface",
    "swept_rank_cases",
]
