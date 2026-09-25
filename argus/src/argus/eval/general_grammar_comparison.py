"""The factor grammar against the best *general-purpose* systems for what it actually does.

``eval/grammar_comparison.py`` measured the grammar against qlib's expression engine, a trading
system. The owner's rule of 2026-09-25 is that the best implementation of a function often lives
outside trading, and losing to a general tool is still losing. Stated in general terms, the grammar
is four functions, and each has a stronger general-purpose home than any quant library:

* **a typed, non-Turing-complete expression language for machine-authored formulas** — Google's
  Common Expression Language. Formal spec and conformance suite (``cel-expr/cel-spec``), and in
  production as the policy language of Kubernetes admission control, Envoy and Google Cloud IAM
  Conditions. Run here through ``cel-expr-python`` 0.1.x, Google's own binding over ``cel-cpp``.
* **a columnar expression engine with rolling operators** — Polars (``pola-rs/polars``, MIT), whose
  expression trees are typed, lazily schema-checked and evaluated column-at-a-time in Rust.
* **a canonical form for duplicate detection** — SymPy (``sympy/sympy``, BSD), whose automatic
  canonicalisation (``core/add.py:198`` ``Add.flatten``, ``core/relational.py:309``
  ``Relational.canonical``) is the reference for "two spellings, one expression".
* **bounded evaluation cost** — CEL again: ``comprehension_max_iterations = 10000`` per
  evaluation (``google/cel-cpp runtime/runtime_options.h:83``).

Every rival runs its real, installed code (isolated venv, never ARGUS's) on the **same** inputs:
neutral expression trees that each system builds natively (``baselines/general_grammar_rivals.py``)
and the 1,439 real hourly NVDAUSDT closes in ``data/regime_candles_fixture.json``. Rival output is
recorded write-through to ``data/general_grammar_rivals_recorded.json`` and re-run live whenever
``ARGUS_GRAMMAR_RIVAL_PYTHON`` names the rival interpreter.

**What the rivals found, and what was done about it** (numbers in the artefact, not here):

1. **Resource exhaustion was reachable.** ``grammar.validate`` bounded depth and size but not cost,
   and a six-node nested-window tree passed it while costing ~6.9e10 node evaluations per bar.
   CEL refuses the analogous nested comprehension at runtime. Fixed by a static cost bound in
   ``grammar.validate`` (``grammar.cost`` / ``MAX_COST``); the ablation below re-runs the old
   behaviour.
2. **There was no text surface.** CEL compiles untrusted text and type-checks it with a positioned
   error before touching data; the grammar could only be built by Python code. Adopted as
   ``research/grammar_text.py`` (parser over the canonical form, no execution surface, same AST
   scan).
3. **Duplicate detection missed algebraic identities** SymPy canonicalises. Adopted as
   ``grammar_text.normal_form``.
4. **Nested windows cost the product of their lookbacks; Polars pays the sum.** Adopted as
   ``research/grammar_series.py`` (columnar, bit-identical to the per-bar path). Polars remains far
   faster in raw terms, and that loss is reported, not hidden.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import statistics
import subprocess
import tempfile
import time
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.backtest.engine import Bar
from argus.eval import artefact
from argus.research.grammar import (
    EXPANDED,
    MAX_COST,
    ORIGINAL_EIGHT,
    Expr,
    Field,
    GrammarError,
    Ref,
    Signal,
    Window,
    cost,
    validate,
)
from argus.research.grammar_series import evaluate_series, series_cost
from argus.research.grammar_text import normal_form, parse

_ROOT = Path(__file__).resolve().parents[3]
DATA = _ROOT / "data"
CANDLES = DATA / "regime_candles_fixture.json"
RECORDED = DATA / "general_grammar_rivals_recorded.json"
OUT = DATA / "general_grammar_comparison.json"
RUNNER = Path(__file__).resolve().parent / "baselines" / "general_grammar_rivals.py"
RIVAL_PYTHON_ENV = "ARGUS_GRAMMAR_RIVAL_PYTHON"

SYMBOL = "NVDAUSDT"
LOOKBACKS = (6, 24, 96)
WINDOW_OPS = ("mean", "sum", "max", "min", "std", "median", "zscore", "rank", "slope",
              "argmax", "argmin")
TIMING_REPEATS = 11
TIMING_WARMUP = 2
BUDGET_WIDTH = 512
TOLERANCE = 1e-9


class GeneralGrammarComparisonError(RuntimeError):
    """The comparison could not run honestly — missing data, or a stale rival recording."""


def _f(name: str) -> list[Any]:
    return ["field", name]


def _c(value: float) -> list[Any]:
    return ["const", value]


# --- the corpus: one definition, handed to every system ---------------------------------------

TYPECHECK_CASES: tuple[dict[str, Any], ...] = (
    {"id": "bool_op_on_number", "ill_typed": True,
     "tree": ["and", _f("close"), ["gt", _f("close"), _c(1.0)]]},
    {"id": "number_as_condition", "ill_typed": True,
     "tree": ["if", _f("close"), _c(1.0), _c(0.0)]},
    {"id": "not_on_number", "ill_typed": True, "tree": ["not", _f("close")]},
    {"id": "compare_boolean_to_number", "ill_typed": True,
     "tree": ["gt", ["gt", _f("close"), _c(1.0)], _c(1.0)]},
    {"id": "arithmetic_on_boolean", "ill_typed": True,
     "tree": ["add", ["gt", _f("close"), _c(1.0)], _c(1.0)]},
    {"id": "branches_disagree", "ill_typed": True,
     "tree": ["if", ["gt", _f("close"), _c(1.0)], _c(1.0), ["gt", _f("close"), _c(2.0)]]},
    {"id": "or_on_numbers", "ill_typed": True, "tree": ["or", _f("volume"), _f("close")]},
    {"id": "negate_boolean", "ill_typed": True, "tree": ["neg", ["gt", _f("close"), _c(1.0)]]},
    {"id": "abs_of_boolean", "ill_typed": True, "tree": ["abs", ["lt", _f("close"), _c(1.0)]]},
    {"id": "undeclared_field", "ill_typed": True, "tree": ["gt", _f("closee"), _c(1.0)]},
    {"id": "control_compare", "ill_typed": False, "tree": ["gt", _f("close"), _c(1.0)]},
    {"id": "control_if", "ill_typed": False,
     "tree": ["if", ["gt", _f("close"), _c(1.0)], _c(1.0), _c(0.0)]},
    {"id": "control_and", "ill_typed": False,
     "tree": ["and", ["gt", _f("close"), _c(1.0)], ["lt", _f("volume"), _c(5.0)]]},
    {"id": "control_arith", "ill_typed": False,
     "tree": ["neg", ["sign", ["sub", _f("close"), _f("volume")]]]},
    {"id": "control_not", "ill_typed": False, "tree": ["not", ["gt", _f("close"), _f("high")]]},
)

_A, _B, _H = _f("close"), _f("volume"), _f("high")

DEDUP_PAIRS: tuple[dict[str, Any], ...] = (
    {"id": "commutative_add", "class": "equivalent", "a": ["add", _A, _B], "b": ["add", _B, _A]},
    {"id": "commutative_mul_const", "class": "equivalent",
     "a": ["mul", _A, _c(2.0)], "b": ["mul", _c(2.0), _A]},
    {"id": "commutative_and", "class": "equivalent",
     "a": ["and", ["gt", _A, _c(1.0)], ["lt", _B, _c(5.0)]],
     "b": ["and", ["lt", _B, _c(5.0)], ["gt", _A, _c(1.0)]]},
    {"id": "symmetric_corr", "class": "equivalent",
     "a": ["corr", 48, _A, _B], "b": ["corr", 48, _B, _A]},
    {"id": "double_negation", "class": "equivalent", "a": ["neg", ["neg", _A]], "b": _A},
    {"id": "subtract_negation", "class": "equivalent",
     "a": ["sub", _A, ["neg", _B]], "b": ["add", _A, _B]},
    {"id": "associativity", "class": "equivalent",
     "a": ["add", ["add", _A, _B], _H], "b": ["add", _A, ["add", _B, _H]]},
    {"id": "relational_flip", "class": "equivalent", "a": ["gt", _A, _B], "b": ["lt", _B, _A]},
    {"id": "noncommutative_sub", "class": "distinct", "a": ["sub", _A, _B], "b": ["sub", _B, _A]},
    {"id": "noncommutative_div", "class": "distinct", "a": ["div", _A, _B], "b": ["div", _B, _A]},
    {"id": "opposite_comparison", "class": "distinct", "a": ["gt", _A, _B], "b": ["lt", _A, _B]},
    {"id": "different_lookback", "class": "distinct",
     "a": ["mean", 24, _A], "b": ["mean", 48, _A]},
    {"id": "different_threshold", "class": "distinct",
     "a": ["gt", _A, _c(0.1)], "b": ["gt", _A, _c(0.2)]},
    # Strictly different constants that ARGUS's own convention (6 significant digits) merges on
    # purpose. Scored separately: under strict semantics these are false merges, and are reported
    # as such rather than counted under whichever reading flatters us.
    {"id": "float_noise_1e-11", "class": "convention",
     "a": ["gt", _A, _c(0.1)], "b": ["gt", _A, _c(0.10000000001)]},
    {"id": "float_near_4e-7", "class": "convention",
     "a": ["gt", _A, _c(0.1)], "b": ["gt", _A, _c(0.1000004)]},
)

# The shipped factors both engines can state from close alone, as neutral trees. Each is checked
# against the real object in `grammar.EXPANDED` / `ORIGINAL_EIGHT` (``same_trees``), so the rival
# is handed the same factor, not a paraphrase of it.
TIMED_FACTORS: tuple[tuple[str, list[Any]], ...] = (
    ("slow_trend", ["signal", ["if", ["gt", ["ret", 48], _c(0.0)], _c(1.0), _c(0.0)]]),
    ("slow_fade", ["signal", ["if", ["gt", ["ret", 48], _c(0.0)], _c(-1.0), _c(1.0)]]),
    ("vol_adjusted_momentum",
     ["signal", ["if", ["gt", ["zscore", 48, _A], _c(1.0)], _c(1.0), _c(0.0)]]),
    ("vol_adjusted_reversion",
     ["signal", ["if", ["gt", ["zscore", 48, _A], _c(1.5)], _c(-1.0), _c(0.0)]]),
    ("range_breakout", ["signal", ["if", ["gt", ["rank", 96, _A], _c(0.95)], _c(1.0), _c(0.0)]]),
    ("stale_high_fade",
     ["signal", ["if", ["gt", ["argmax", 96, _A], _c(72.0)], _c(-1.0), _c(0.0)]]),
    ("fitted_trend", ["signal", ["if", ["gt", ["slope", 48, _A], _c(0.0)], _c(1.0), _c(-1.0)]]),
    ("delta_momentum",
     ["signal", ["if", ["gt", ["sub", _A, ["delay", 24, _A]], _c(0.0)], _c(1.0), _c(0.0)]]),
)


def argus_text(tree: list[Any]) -> str:
    """A neutral tree in the grammar's own canonical spelling."""
    op = tree[0]
    if op == "field":
        return str(tree[1])
    if op == "const":
        return repr(float(tree[1]))
    if op == "ret":
        return f"ret({int(tree[1])})"
    if op in WINDOW_OPS or op == "delay":
        return f"{op}({int(tree[1])},{argus_text(tree[2])})"
    if op == "corr":
        return f"corr({int(tree[1])},{argus_text(tree[2])},{argus_text(tree[3])})"
    return f"{op}({','.join(argus_text(t) for t in tree[1:])})"


def _numeric_ops() -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = []
    for n in LOOKBACKS:
        for op in WINDOW_OPS:
            ops.append({"id": f"{op}_{n}", "op": op, "lookback": n, "tree": [op, n, _A]})
        ops.append({"id": f"corr_{n}", "op": "corr", "lookback": n,
                    "tree": ["corr", n, _A, _f("return_1")]})
        ops.append({"id": f"delay_{n}", "op": "delay", "lookback": n, "tree": ["delay", n, _A]})
    return ops


def load_closes(symbol: str = SYMBOL) -> list[float]:
    if not CANDLES.exists():
        raise GeneralGrammarComparisonError(f"{CANDLES} is missing")
    series = json.loads(CANDLES.read_text(encoding="utf-8"))["series"][symbol]
    return [float(Decimal(close)) for _, close in series]


def _bars(symbol: str = SYMBOL) -> list[Bar]:
    series = json.loads(CANDLES.read_text(encoding="utf-8"))["series"][symbol]
    return [Bar(ts=datetime.fromisoformat(ts), close=Decimal(close)) for ts, close in series]


def build_corpus() -> dict[str, Any]:
    corpus: dict[str, Any] = {
        "typecheck": list(TYPECHECK_CASES),
        "dedup": list(DEDUP_PAIRS),
        "closes": load_closes(),
        "numeric_ops": _numeric_ops(),
        "timed_factors": [{"id": k, "tree": t} for k, t in TIMED_FACTORS],
        "timing_repeats": TIMING_REPEATS,
        "timing_warmup": TIMING_WARMUP,
        "budget_width": BUDGET_WIDTH,
    }
    corpus["digest"] = hashlib.sha256(
        json.dumps(corpus, sort_keys=True).encode("utf-8")).hexdigest()
    return corpus


def same_trees() -> dict[str, bool]:
    """Each timed neutral tree parses to exactly the shipped factor of the same name."""
    shipped = {**ORIGINAL_EIGHT, **EXPANDED}
    return {name: parse(argus_text(tree)).canonical() == shipped[name].canonical()
            for name, tree in TIMED_FACTORS}


# --- ARGUS side -------------------------------------------------------------------------------


def _probe_bars() -> list[Bar]:
    start = datetime.fromisoformat("2026-06-02T00:00:00+00:00")
    return [Bar(ts=start, close=Decimal("2"), extra={"volume": 2.0, "high": 2.0, "low": 2.0})
            for _ in range(3)]


def argus_typecheck() -> list[dict[str, Any]]:
    rows = []
    bars = _probe_bars()
    for case in TYPECHECK_CASES:
        text = argus_text(case["tree"])
        try:
            expr = parse(text)
        except GrammarError as exc:
            rows.append({"id": case["id"], "outcome": "static",
                         "detail": str(exc).splitlines()[0]})
            continue
        expr.evaluate(bars, 2)
        rows.append({"id": case["id"], "outcome": "evaluated", "detail": ""})
    return rows


def argus_dedup() -> list[dict[str, Any]]:
    rows = []
    for pair in DEDUP_PAIRS:
        a, b = parse(argus_text(pair["a"])), parse(argus_text(pair["b"]))
        rows.append({"id": pair["id"],
                     "canonical_merged": a.canonical() == b.canonical(),
                     "normal_form_merged": normal_form(a) == normal_form(b)})
    return rows


def argus_numeric(bars: Sequence[Bar]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for spec in _numeric_ops():
        expr = parse(argus_text(spec["tree"]))
        out[spec["id"]] = [expr.evaluate(bars, i) for i in range(len(bars))]
    return out


def _median_seconds(fn: Any, repeats: int = TIMING_REPEATS) -> tuple[float, Any]:
    """Median wall time over ``repeats`` calls, after the same warm-up the rival gets."""
    for _ in range(TIMING_WARMUP):
        fn()
    samples, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), result


def argus_timing(bars: Sequence[Bar]) -> dict[str, dict[str, Any]]:
    shipped = {**ORIGINAL_EIGHT, **EXPANDED}
    out: dict[str, dict[str, Any]] = {}
    for name, _ in TIMED_FACTORS:
        tree = shipped[name]
        per_bar, values = _median_seconds(
            lambda t=tree: [t.evaluate(bars, i) for i in range(len(bars))])
        series, series_values = _median_seconds(lambda t=tree: evaluate_series(t, bars))
        out[name] = {"per_bar_seconds": per_bar, "series_seconds": series,
                     "series_identical_to_per_bar": series_values == values,
                     "values": values}
    return out


def _node_rate_us(bars: Sequence[Bar]) -> float:
    """Measured microseconds per node evaluation on the per-bar path."""
    tree = Window("mean", 96, Ref(Field.CLOSE))
    seconds, _ = _median_seconds(lambda: [tree.evaluate(bars, i) for i in range(len(bars))], 3)
    return seconds / (cost(tree) * len(bars)) * 1e6


def argus_budget(bars: Sequence[Bar]) -> dict[str, Any]:
    rate = _node_rate_us(bars)
    rows = []
    for depth in range(1, 5):
        inner: Expr = Ref(Field.CLOSE)
        for _ in range(depth):
            inner = Window("mean", BUDGET_WIDTH, inner)
        tree = Signal(inner)
        per_bar_cost = cost(tree)
        try:
            validate(tree)
            verdict = "accepted"
        except GrammarError:
            verdict = "refused"
        # The ablation: the same tree through validate with the cost bound disabled — what
        # grammar.validate did before this comparison found the gap.
        try:
            validate(tree, max_cost=10**30)
            ablated = "accepted"
        except GrammarError:
            ablated = "refused"
        # The unclamped chain, not the Signal: Signal clamps to [-1, 1], and a clamped price mean
        # would compare 1.0 against Polars' ~220 and measure nothing.
        seconds, values = _median_seconds(lambda t=inner: evaluate_series(t, bars), 3)
        rows.append({
            "depth": depth,
            "cost_per_bar": per_bar_cost,
            "validate": verdict,
            "validate_without_cost_bound": ablated,
            "per_bar_path_estimated_seconds_per_bar": per_bar_cost * rate / 1e6,
            "series_cost_per_bar": series_cost(tree),
            "series_seconds_full_series": seconds,
            "series_last": values[-1],
        })
    return {"us_per_node": rate, "max_cost": MAX_COST, "rows": rows}


def execution_surface_scan() -> dict[str, list[str]]:
    """The AST scan ``tests/test_grammar.py`` runs, over all three grammar modules."""
    forbidden = {"eval", "exec", "compile", "__import__", "open", "getattr", "setattr"}
    out: dict[str, list[str]] = {}
    research = Path(__file__).resolve().parents[1] / "research"
    for name in ("grammar.py", "grammar_text.py", "grammar_series.py"):
        tree = ast.parse((research / name).read_text(encoding="utf-8"))
        called: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name):
                    called.add(fn.id)
                elif isinstance(fn, ast.Attribute):
                    called.add(fn.attr)
        out[name] = sorted(called & forbidden)
    return out


# --- rivals -----------------------------------------------------------------------------------


def run_rivals(corpus: dict[str, Any], python: str) -> dict[str, Any]:
    """Run the rival runner under the rival interpreter and record its output write-through."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "corpus.json"
        out_path = Path(tmp) / "out.json"
        corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
        subprocess.run([python, str(RUNNER), str(corpus_path), str(out_path)], check=True,
                       timeout=900)
        result: dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8"))
    artefact.write(RECORDED, result, indent=1)
    return result


def load_rivals(corpus: dict[str, Any]) -> tuple[dict[str, Any], str]:
    python = os.environ.get(RIVAL_PYTHON_ENV)
    if python:
        return run_rivals(corpus, python), "live"
    if not RECORDED.exists():
        raise GeneralGrammarComparisonError(
            f"no rival recording at {RECORDED}; set {RIVAL_PYTHON_ENV} to the rival interpreter")
    recorded: dict[str, Any] = json.loads(RECORDED.read_text(encoding="utf-8"))
    if recorded.get("corpus_digest") != corpus["digest"]:
        raise GeneralGrammarComparisonError(
            "the rival recording was made on a different corpus; re-run with "
            f"{RIVAL_PYTHON_ENV} set")
    return recorded, "recorded"


# --- scoring ----------------------------------------------------------------------------------


def score_typecheck(argus_rows: list[dict[str, Any]], rivals: dict[str, Any]) -> dict[str, Any]:
    ill = {c["id"] for c in TYPECHECK_CASES if c["ill_typed"]}
    controls = {c["id"] for c in TYPECHECK_CASES if not c["ill_typed"]}
    systems = {"argus": argus_rows, "cel": rivals["typecheck"]["cel"],
               "polars": rivals["typecheck"]["polars"]}
    out: dict[str, Any] = {}
    for name, rows in systems.items():
        by_id = {r["id"]: r for r in rows}
        out[name] = {
            "ill_typed_rejected_statically": sum(by_id[i]["outcome"] == "static" for i in ill),
            "ill_typed_rejected_at_runtime": sum(by_id[i]["outcome"] == "runtime" for i in ill),
            "ill_typed_silently_evaluated": sorted(
                i for i in ill if by_id[i]["outcome"] == "evaluated"),
            "controls_falsely_rejected": sorted(
                i for i in controls if by_id[i]["outcome"] != "evaluated"),
            "rows": rows,
        }
    out["ill_typed_total"] = len(ill)
    out["controls_total"] = len(controls)
    return out


def score_dedup(argus_rows: list[dict[str, Any]], rivals: dict[str, Any]) -> dict[str, Any]:
    cls = {p["id"]: p["class"] for p in DEDUP_PAIRS}
    systems: dict[str, dict[str, bool | None]] = {
        "argus_canonical": {r["id"]: r["canonical_merged"] for r in argus_rows},
        "argus_normal_form": {r["id"]: r["normal_form_merged"] for r in argus_rows},
    }
    for rival in ("cel", "polars", "sympy"):
        systems[rival] = {r["id"]: r["merged"] for r in rivals["dedup"][rival]}
    out: dict[str, Any] = {}
    for name, merged in systems.items():
        eq = [i for i, c in cls.items() if c == "equivalent"]
        out[name] = {
            "equivalent_merged": sum(merged[i] is True for i in eq),
            "equivalent_inexpressible": sorted(i for i in eq if merged[i] is None),
            "equivalent_missed": sorted(i for i in eq if merged[i] is False),
            "distinct_falsely_merged": sorted(
                i for i, c in cls.items() if c == "distinct" and merged[i] is True),
            "convention_pairs_merged": sorted(
                i for i, c in cls.items() if c == "convention" and merged[i] is True),
        }
    out["equivalent_total"] = sum(c == "equivalent" for c in cls.values())
    return out


def _defined(value: float | None) -> bool:
    return value is not None and value == value and abs(value) != float("inf")


def score_numeric(argus_series: dict[str, list[float]], rivals: dict[str, Any]
                  ) -> dict[str, Any]:
    """Per operator: agreement on full windows, and every bar where the two conventions differ.

    Polars returns null/NaN where a statistic is undefined (one sample, a flat leg); ARGUS returns
    0.0 there by design (``grammar.Window``/``Corr`` docstrings). Those bars are counted as
    ``undefined_in_polars`` rather than folded into a numeric difference, so a convention
    difference can never masquerade as, or hide, an arithmetic one.
    """
    rows = []
    for spec in _numeric_ops():
        mine = argus_series[spec["id"]]
        theirs = rivals["numeric"].get(spec["id"])
        if theirs is None:
            rows.append({"id": spec["id"], "polars": "inexpressible natively"})
            continue
        full_from = spec["lookback"] - 1
        full = [(a, b) for i, (a, b) in enumerate(zip(mine, theirs, strict=True))
                if i >= full_from]
        warm = [(a, b) for i, (a, b) in enumerate(zip(mine, theirs, strict=True))
                if i < full_from]
        scale = max(1.0, max(abs(a) for a in mine))
        diffs = [abs(a - b) for a, b in full if _defined(b)]
        full_diff = max(diffs, default=0.0)
        undefined_full = [a for a, b in full if not _defined(b)]
        rows.append({
            "id": spec["id"],
            "full_window_bars": len(full),
            "full_window_max_abs_diff": full_diff,
            "full_window_agrees": full_diff <= TOLERANCE * scale,
            "full_window_undefined_in_polars": len(undefined_full),
            "argus_values_where_polars_undefined": sorted(set(undefined_full))[:5],
            "warmup_bars": len(warm),
            "warmup_undefined_in_polars": sum(1 for _, b in warm if not _defined(b)),
            "warmup_numeric_disagreements": sum(
                1 for a, b in warm if _defined(b) and abs(a - b) > TOLERANCE * scale),
        })
    expressible = [r for r in rows if "full_window_agrees" in r]
    return {
        "rows": rows,
        "ops_checked": len(rows),
        "polars_inexpressible": sorted(r["id"] for r in rows if "full_window_agrees" not in r),
        "full_window_agree": sum(r["full_window_agrees"] for r in expressible),
        "full_window_expressible": len(expressible),
        "full_window_bars_undefined_in_polars": sum(
            r["full_window_undefined_in_polars"] for r in expressible),
    }


def score_timing(argus_rows: dict[str, dict[str, Any]], rivals: dict[str, Any]
                 ) -> dict[str, Any]:
    """ARGUS's per-bar and columnar timings against Polars' on the same factor trees.

    Only a live run (``ARGUS_GRAMMAR_RIVAL_PYTHON`` set) times both sides on one machine in one
    sitting. From the recording, the Polars seconds are the recording's and the ARGUS seconds are
    this run's, so the speed-up is indicative; the committed artefact was made live. The loss it
    shows is one to two orders of magnitude, far outside that caveat.
    """
    rows = []
    for name, _ in TIMED_FACTORS:
        mine = argus_rows[name]
        theirs = rivals["timing"][name]
        row: dict[str, Any] = {
            "factor": name,
            "argus_per_bar_seconds": mine["per_bar_seconds"],
            "argus_series_seconds": mine["series_seconds"],
            "series_identical_to_per_bar": mine["series_identical_to_per_bar"],
        }
        if "inexpressible" in theirs:
            row["polars"] = f"inexpressible natively ({theirs['inexpressible']})"
        else:
            vals = theirs["values"]
            row["polars_seconds"] = theirs["median_seconds"]
            row["polars_speedup_over_argus_series"] = (
                mine["series_seconds"] / theirs["median_seconds"])
            row["signal_agreement"] = sum(
                1 for a, b in zip(mine["values"], vals, strict=True)
                if _defined(b) and abs(a - b) < 1e-9) / len(vals)
        rows.append(row)
    return {"rows": rows}


SCOPE_STATEMENT = """\
CLAIMED, measured on the same neutral trees and the same 1,439 real hourly NVDAUSDT closes, every \
rival running its real installed code: (1) type checking — ARGUS's text path and CEL both reject \
every ill-typed case before touching data and accept every well-typed control; Polars resolves \
some of the same cases only at execution and evaluates others silently (counts in 'typecheck'). \
(2) Cost — before this comparison grammar.validate accepted a six-node tree costing ~6.9e10 node \
evaluations per bar; CEL refuses the analogous nested comprehension at its 10,000-iteration \
runtime budget. validate now refuses it statically at the same nesting depth CEL does; the \
ablation re-runs the old validate and shows it accepting. (3) Duplicate identity — on the \
equivalent-pair corpus ARGUS's canonical() missed the algebraic identities SymPy merges; \
normal_form() merges them and also merges the symmetric rolling correlation SymPy cannot see \
inside an uninterpreted function. CEL's serialised AST and Polars' meta.eq merge only identical \
spellings. (4) Numerics — Polars' rolling kernels agree with ARGUS's operators on full windows to \
within 1e-9 where Polars has a native kernel; warm-up conventions differ and are counted, and \
argmax/argmin have no native Polars rolling kernel. \
NOT CLAIMED: that ARGUS is faster than Polars — it is not, by the factor in 'timing', and the \
columnar evaluator adopted here narrows the nested-window gap from multiplicative to additive \
without touching Polars' constant factor, because incremental rolling kernels would break the \
bit-identity between the per-bar and series paths that trial accounting depends on. NOT CLAIMED: \
that CEL is weaker as a language — it is a general policy language with a formal spec and a \
conformance suite, and it cannot express a rolling window natively only because it deliberately \
has no fold. NOT CLAIMED: that normal_form is bit-exact — re-association is exact only up to \
rounding, stated in grammar_text's docstring. NOT RUN: a hostile-input corpus against the rivals; \
the execution-surface axis here is the AST scan of ARGUS's three grammar modules plus CEL's \
observed refusal of undeclared references. NOT WIRED: grammar_series and normal_form are \
available to callers but the search (research/searchoff.py) still uses the per-bar path and \
canonical()."""


def main(*, rivals: dict[str, Any] | None = None) -> dict[str, Any]:
    corpus = build_corpus()
    source = "supplied"
    if rivals is None:
        rivals, source = load_rivals(corpus)
    bars = _bars()
    typecheck = score_typecheck(argus_typecheck(), rivals)
    dedup = score_dedup(argus_dedup(), rivals)
    numeric = score_numeric(argus_numeric(bars), rivals)
    timing = score_timing(argus_timing(bars), rivals)
    budget = {"argus": argus_budget(bars), "cel": rivals["budget"]["cel"],
              "polars": rivals["budget"]["polars"]}
    for mine, theirs in zip(budget["argus"]["rows"], budget["polars"], strict=True):
        # Same nested mean(512) tree, same closes: does the columnar ARGUS path land on the
        # value Polars computes for the deepest nesting either can reach?
        mine["abs_diff_vs_polars_last"] = abs(mine["series_last"] - theirs["last"])
    return {
        "rival_source": source,
        "rival_versions": rivals["versions"],
        "rival_python": rivals["python"],
        "corpus_digest": corpus["digest"],
        "bars": len(bars),
        "symbol": SYMBOL,
        "same_trees": same_trees(),
        "execution_surface_scan": execution_surface_scan(),
        "typecheck": typecheck,
        "dedup": dedup,
        "numeric": numeric,
        "timing": timing,
        "budget": budget,
        "scope_statement": SCOPE_STATEMENT,
    }


def render(report: dict[str, Any]) -> str:
    tc, dd = report["typecheck"], report["dedup"]
    lines = [f"GENERAL-PURPOSE RIVALS vs the factor grammar ({report['rival_source']} rivals: "
             f"{report['rival_versions']})", "", "-- type checking --"]
    for name in ("argus", "cel", "polars"):
        r = tc[name]
        lines.append(f"  {name:7s} static {r['ill_typed_rejected_statically']}/"
                     f"{tc['ill_typed_total']}  runtime {r['ill_typed_rejected_at_runtime']}  "
                     f"silent {r['ill_typed_silently_evaluated']}  "
                     f"false-rejects {r['controls_falsely_rejected']}")
    lines.append("-- duplicate identity --")
    for name in ("argus_canonical", "argus_normal_form", "cel", "polars", "sympy"):
        r = dd[name]
        lines.append(f"  {name:18s} merged {r['equivalent_merged']}/{dd['equivalent_total']}  "
                     f"missed {r['equivalent_missed']}  false {r['distinct_falsely_merged']}")
    nm = report["numeric"]
    lines.append(f"-- numerics: {nm['full_window_agree']}/{nm['full_window_expressible']} "
                 f"agree on full windows; Polars inexpressible: {nm['polars_inexpressible']}")
    lines.append("-- cost budget (nested mean(512)) --")
    for row in report["budget"]["argus"]["rows"]:
        lines.append(f"  depth {row['depth']}: cost/bar {row['cost_per_bar']:.3g} "
                     f"validate={row['validate']} (without bound: "
                     f"{row['validate_without_cost_bound']}); series "
                     f"{row['series_seconds_full_series']:.2f}s")
    for row in report["budget"]["cel"]:
        lines.append(f"  CEL depth {row['depth']}: {row['verdict']} {row['detail'][:60]}")
    lines.append("-- timing (1,439 bars) --")
    for row in report["timing"]["rows"]:
        lines.append(f"  {row['factor']:24s} per-bar {row['argus_per_bar_seconds']:.3f}s "
                     f"series {row['argus_series_seconds']:.3f}s "
                     f"polars {row.get('polars_seconds', float('nan')):.4f}s")
    return "\n".join(lines)


if __name__ == "__main__":
    result = main()
    print(render(result))
    artefact.write(OUT, result)
    print(f"\nsaved -> {OUT}")


__all__ = [
    "DEDUP_PAIRS",
    "SCOPE_STATEMENT",
    "TIMED_FACTORS",
    "TYPECHECK_CASES",
    "GeneralGrammarComparisonError",
    "argus_text",
    "build_corpus",
    "load_rivals",
    "main",
    "render",
    "run_rivals",
    "same_trees",
]
