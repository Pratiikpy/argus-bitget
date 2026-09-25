"""Runs the general-purpose rivals of the factor grammar on the comparison's corpus.

Executed by the *rival* interpreter, not by ARGUS's: CEL (``cel-expr-python``, Google's official
binding over ``google/cel-cpp``), Polars and SymPy are installed in an isolated venv and never in
ARGUS's own. This file therefore imports nothing from ``argus`` and loads every rival through
``importlib`` so the package's type check does not need their stubs.

    <rival-python> general_grammar_rivals.py <corpus.json> <out.json>

The corpus is written by :func:`argus.eval.general_grammar_comparison.build_corpus`; expressions
arrive as neutral trees (``["add", ["field", "close"], ["const", 1.0]]``) and each rival builds its
own native expression from the same tree, so every system is handed the same input rather than a
spelling chosen to suit it. Rival versions are recorded in the output.
"""

from __future__ import annotations

import importlib
import json
import statistics
import sys
import time
import warnings
from collections.abc import Callable
from importlib import metadata
from typing import Any

_CEL_BINARY = {"add": "+", "sub": "-", "mul": "*", "div": "/", "gt": ">", "lt": "<",
               "and": "&&", "or": "||"}
_FIELDS = ("close", "volume", "high", "low", "return_1")


class Inexpressible(Exception):
    """The rival's own language has no native construct for this node."""


# --- CEL --------------------------------------------------------------------------------------


def _cel_text(node: list[Any]) -> str:
    op = node[0]
    if op == "field":
        return str(node[1])
    if op == "const":
        return repr(float(node[1]))
    if op in _CEL_BINARY:
        return f"({_cel_text(node[1])} {_CEL_BINARY[op]} {_cel_text(node[2])})"
    if op == "not":
        return f"!{_cel_text(node[1])}"
    if op == "neg":
        return f"-{_cel_text(node[1])}"
    if op in ("abs", "sign"):
        return f"math.{op}({_cel_text(node[1])})"
    if op == "if":
        return f"({_cel_text(node[1])} ? {_cel_text(node[2])} : {_cel_text(node[3])})"
    raise Inexpressible(op)


def _cel_env(cel: Any, ext_math: Any) -> Any:
    return cel.NewEnv(variables={f: cel.Type.DOUBLE for f in _FIELDS},
                      extensions=[ext_math.ExtMath()])


# --- Polars -----------------------------------------------------------------------------------


def _polars_expr(pl: Any, node: list[Any]) -> Any:
    op = node[0]
    if op == "field":
        return pl.col(node[1])
    if op == "const":
        return pl.lit(float(node[1]))
    if op in ("add", "sub", "mul", "div", "gt", "lt", "and", "or"):
        a, b = _polars_expr(pl, node[1]), _polars_expr(pl, node[2])
        return {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b,
                "div": lambda: a / b, "gt": lambda: a > b, "lt": lambda: a < b,
                "and": lambda: a & b, "or": lambda: a | b}[op]()
    if op == "not":
        return ~_polars_expr(pl, node[1])
    if op == "neg":
        return -_polars_expr(pl, node[1])
    if op == "abs":
        return _polars_expr(pl, node[1]).abs()
    if op == "sign":
        return _polars_expr(pl, node[1]).sign()
    if op == "if":
        return (pl.when(_polars_expr(pl, node[1])).then(_polars_expr(pl, node[2]))
                .otherwise(_polars_expr(pl, node[3])))
    if op in _POLARS_WINDOWS:
        return _POLARS_WINDOWS[op](pl, int(node[1]), _polars_expr(pl, node[2]))
    if op == "ret":
        close = pl.col("close")
        prev = close.shift(int(node[1]))
        return ((close - prev) / prev).fill_null(0.0)
    if op == "signal":
        return _polars_expr(pl, node[1]).clip(-1.0, 1.0)
    if op == "delay":
        x = _polars_expr(pl, node[2])
        return x.shift(int(node[1])).fill_null(x.first())
    if op == "corr":
        return pl.rolling_corr(_polars_expr(pl, node[2]), _polars_expr(pl, node[3]),
                               window_size=int(node[1]), min_samples=1)
    raise Inexpressible(op)


def _p_zscore(pl: Any, n: int, x: Any) -> Any:
    mu, sd = x.rolling_mean(n, min_samples=1), x.rolling_std(n, min_samples=1)
    return pl.when(sd <= 1e-12).then(0.0).otherwise((x - mu) / sd)


def _p_rank(pl: Any, n: int, x: Any) -> Any:
    # Polars' average-method rank r of the newest value among k = min(i+1, n) samples. ARGUS
    # normalises by k-1 and fixes k == 1 at 0.5; the rescaling below is that convention, applied
    # outside the rival so the rival's own kernel is what ranks.
    r = x.rolling_rank(n, method="average", min_samples=1)
    k = pl.int_range(1, pl.len() + 1).clip(upper_bound=n).cast(pl.Float64)
    return pl.when(k <= 1).then(0.5).otherwise((r - 1.0) / (k - 1.0))


def _p_slope(pl: Any, n: int, x: Any) -> Any:
    idx = pl.int_range(0, pl.len()).cast(pl.Float64)
    return pl.rolling_cov(idx, x, window_size=n, min_samples=1) / idx.rolling_var(
        n, min_samples=1)


_POLARS_WINDOWS: dict[str, Callable[[Any, int, Any], Any]] = {
    "mean": lambda pl, n, x: x.rolling_mean(n, min_samples=1),
    "sum": lambda pl, n, x: x.rolling_sum(n, min_samples=1),
    "max": lambda pl, n, x: x.rolling_max(n, min_samples=1),
    "min": lambda pl, n, x: x.rolling_min(n, min_samples=1),
    "std": lambda pl, n, x: x.rolling_std(n, min_samples=1),
    "median": lambda pl, n, x: x.rolling_median(n, min_samples=1),
    "zscore": _p_zscore,
    "rank": _p_rank,
    "slope": _p_slope,
}


# --- SymPy ------------------------------------------------------------------------------------


def _sympy_expr(sp: Any, node: list[Any]) -> Any:
    op = node[0]
    if op == "field":
        return sp.Symbol(node[1], real=True)
    if op == "const":
        return sp.Float(float(node[1]))
    if op in ("add", "sub", "mul", "div"):
        a, b = _sympy_expr(sp, node[1]), _sympy_expr(sp, node[2])
        return {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[op]
    if op == "gt":
        return sp.StrictGreaterThan(_sympy_expr(sp, node[1]), _sympy_expr(sp, node[2]))
    if op == "lt":
        return sp.StrictLessThan(_sympy_expr(sp, node[1]), _sympy_expr(sp, node[2]))
    if op == "and":
        return sp.And(_sympy_expr(sp, node[1]), _sympy_expr(sp, node[2]))
    if op == "or":
        return sp.Or(_sympy_expr(sp, node[1]), _sympy_expr(sp, node[2]))
    if op == "not":
        return sp.Not(_sympy_expr(sp, node[1]))
    if op == "neg":
        return -_sympy_expr(sp, node[1])
    if op == "abs":
        return sp.Abs(_sympy_expr(sp, node[1]))
    if op == "sign":
        return sp.sign(_sympy_expr(sp, node[1]))
    if op in _POLARS_WINDOWS or op in ("argmax", "argmin", "delay"):
        # An uninterpreted function: SymPy knows nothing about rolling windows, and this is the
        # honest encoding of that — identity by structure, no algebra applied inside.
        return sp.Function(op)(sp.Integer(int(node[1])), _sympy_expr(sp, node[2]))
    if op == "corr":
        return sp.Function("corr")(sp.Integer(int(node[1])), _sympy_expr(sp, node[2]),
                                   _sympy_expr(sp, node[3]))
    raise Inexpressible(op)


def _sympy_identity(sp: Any, node: list[Any]) -> str:
    expr = _sympy_expr(sp, node)
    expr = expr.replace(lambda e: getattr(e, "is_Relational", False), lambda e: e.canonical)
    return str(sp.srepr(expr))


# --- axes -------------------------------------------------------------------------------------


def _cel_outcome(env: Any, tree: list[Any]) -> dict[str, str]:
    """Compile (CEL's checker runs here, against the declared variable types), then evaluate."""
    try:
        text = _cel_text(tree)
    except Inexpressible as exc:
        return {"outcome": "inexpressible", "detail": str(exc)}
    try:
        compiled = env.compile(text)
    except Exception as exc:
        return {"outcome": "static", "detail": str(exc).splitlines()[0][:240]}
    try:
        compiled.eval(data={f: 2.0 for f in _FIELDS})
    except Exception as exc:
        return {"outcome": "runtime", "detail": f"{type(exc).__name__}: {exc}"[:240]}
    return {"outcome": "evaluated", "detail": ""}


def _typecheck(cases: list[dict[str, Any]], cel: Any, ext_math: Any, pl: Any) -> dict[str, Any]:
    env = _cel_env(cel, ext_math)
    frame = pl.DataFrame({f: [1.0, 2.0, 3.0] for f in _FIELDS})
    out: dict[str, Any] = {"cel": [], "polars": []}
    for case in cases:
        tree = case["tree"]
        out["cel"].append({"id": case["id"], **_cel_outcome(env, tree)})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                expr = _polars_expr(pl, tree)
            except Inexpressible as exc:
                out["polars"].append({"id": case["id"], "outcome": "inexpressible",
                                      "detail": str(exc)})
                continue
            except Exception as exc:
                out["polars"].append({"id": case["id"], "outcome": "static",
                                      "detail": f"at construction: {type(exc).__name__}: {exc}"
                                      [:240]})
                continue
            lazy = frame.lazy().select(expr.alias("out"))
            try:
                lazy.collect_schema()
            except Exception as exc:
                out["polars"].append({"id": case["id"], "outcome": "static",
                                      "detail": f"schema: {type(exc).__name__}: "
                                      f"{str(exc).splitlines()[0]}"[:240]})
                continue
            try:
                lazy.collect()
                outcome, detail = "evaluated", ""
            except Exception as exc:
                outcome = "runtime"
                detail = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"[:240]
            notes = sorted({str(w.message).splitlines()[0][:160] for w in caught})
            out["polars"].append({"id": case["id"], "outcome": outcome, "detail": detail,
                                  "warnings": notes})
    return out


def _dedup(pairs: list[dict[str, Any]], cel: Any, ext_math: Any, pl: Any, sp: Any
           ) -> dict[str, Any]:
    env = _cel_env(cel, ext_math)
    out: dict[str, Any] = {"cel": [], "polars": [], "sympy": []}
    for pair in pairs:
        a, b = pair["a"], pair["b"]
        try:
            merged: bool | None = (env.compile(_cel_text(a)).serialize()
                                   == env.compile(_cel_text(b)).serialize())
        except Inexpressible:
            merged = None
        out["cel"].append({"id": pair["id"], "merged": merged})
        try:
            merged = bool(_polars_expr(pl, a).meta.eq(_polars_expr(pl, b)))
        except Inexpressible:
            merged = None
        out["polars"].append({"id": pair["id"], "merged": merged})
        try:
            merged = _sympy_identity(sp, a) == _sympy_identity(sp, b)
        except Inexpressible:
            merged = None
        out["sympy"].append({"id": pair["id"], "merged": merged})
    return out


def _numeric(corpus: dict[str, Any], pl: Any) -> dict[str, Any]:
    closes = corpus["closes"]
    frame = _frame(pl, closes)
    series: dict[str, Any] = {}
    for spec in corpus["numeric_ops"]:
        try:
            expr = _polars_expr(pl, spec["tree"])
        except Inexpressible:
            series[spec["id"]] = None
            continue
        values = frame.select(expr.alias("v"))["v"].to_list()
        series[spec["id"]] = [None if v is None else float(v) for v in values]
    return series


def _frame(pl: Any, closes: list[float]) -> Any:
    close = pl.Series("close", closes, dtype=pl.Float64)
    prev = close.shift(1)
    ret = ((close - prev) / prev).fill_null(0.0)
    return pl.DataFrame({"close": close, "return_1": ret})


def _timing(corpus: dict[str, Any], pl: Any) -> dict[str, Any]:
    frame = _frame(pl, corpus["closes"])
    out: dict[str, Any] = {}
    for spec in corpus["timed_factors"]:
        try:
            expr = _polars_expr(pl, spec["tree"])
        except Inexpressible as exc:
            out[spec["id"]] = {"inexpressible": str(exc)}
            continue
        for _ in range(int(corpus["timing_warmup"])):
            # Polars' first calls on a fresh process include thread-pool start-up (measured at
            # tens of milliseconds); timing them would charge Polars for a one-off cost.
            frame.select(expr.alias("v"))
        samples = []
        for _ in range(int(corpus["timing_repeats"])):
            start = time.perf_counter()
            frame.select(expr.alias("v"))
            samples.append(time.perf_counter() - start)
        values = frame.select(expr.alias("v"))["v"].to_list()
        out[spec["id"]] = {"median_seconds": statistics.median(samples),
                           "worst_seconds": max(samples),
                           "values": [None if v is None else float(v) for v in values]}
    return out


def _budget(corpus: dict[str, Any], cel: Any, ext_math: Any, pl: Any) -> dict[str, Any]:
    width = int(corpus["budget_width"])
    env = cel.NewEnv(variables={"w": cel.Type.List(cel.Type.DOUBLE)},
                     extensions=[ext_math.ExtMath()])
    data = {"w": [float(v) for v in corpus["closes"][:width]]}
    cel_rows = []
    for depth in range(1, 5):
        inner = "a1"
        for level in range(depth, 0, -1):
            inner = f"w.map(a{level}, {inner})"
        start = time.perf_counter()
        try:
            env.compile(inner).eval(data=data)
            verdict, detail = "evaluated", ""
        except Exception as exc:
            verdict, detail = "refused", str(exc).splitlines()[0][:200]
        cel_rows.append({"depth": depth, "iterations_requested": width ** depth,
                         "verdict": verdict, "detail": detail,
                         "seconds": time.perf_counter() - start})
    frame = _frame(pl, corpus["closes"])
    polars_rows = []
    for depth in range(1, 5):
        expr = pl.col("close")
        for _ in range(depth):
            expr = expr.rolling_mean(width, min_samples=1)
        start = time.perf_counter()
        values = frame.select(expr.alias("v"))["v"].to_list()
        polars_rows.append({"depth": depth, "seconds": time.perf_counter() - start,
                            "last": float(values[-1])})
    return {"cel": cel_rows, "polars": polars_rows}


def main(corpus_path: str, out_path: str) -> None:
    with open(corpus_path, encoding="utf-8") as fh:
        corpus = json.load(fh)
    cel = importlib.import_module("cel_expr_python.cel")
    ext_math = importlib.import_module("cel_expr_python.ext.ext_math")
    pl = importlib.import_module("polars")
    sp = importlib.import_module("sympy")
    result = {
        "versions": {name: metadata.version(name)
                     for name in ("cel-expr-python", "polars", "sympy")},
        "python": sys.version.split()[0],
        "corpus_digest": corpus["digest"],
        "typecheck": _typecheck(corpus["typecheck"], cel, ext_math, pl),
        "dedup": _dedup(corpus["dedup"], cel, ext_math, pl, sp),
        "numeric": _numeric(corpus, pl),
        "timing": _timing(corpus, pl),
        "budget": _budget(corpus, cel, ext_math, pl),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
