"""S24 research depth, measured: the scenario tree and the research fan-out against the single pass.

Two capabilities were built for the Track 3 console on 2026-09-25 — the second-order stress tree
(`desk/stress_tree.py`, after gpt-researcher's recursive deep research) and the isolated
researcher fan-out (`lui/fanout.py`, after open_deep_research's supervisor and researcher
subgraphs). Neither is worth anything until it is run against what the console already says to the
same question. This module runs both, on real questions written by people who never read this
repository, and scores them against the console's current one-pass answer on four things:

* **Distinct evidence sources** — distinct ``(kind, ref)`` sources an answer cites, and distinct
  network sources it reached (`truth/coverage.py`). The tree adds analysis, not data: it reads the
  same candles, and the score says so rather than counting its computation source as evidence.
* **Scenario coverage** (stress only) — distinct scenario angles an answer computes a figure for.
  The one-pass angles are read from its lines; the tree's from its answered nodes.
* **Factual errors found by recomputation.** Every figure the tree states, and every beta-shock,
  loss-share, hedge-size and realised-window figure the one-pass stress answer states, is
  recomputed here with an independent implementation (numpy, written from the definitions, not
  from `desk/portfolio.py`) over the same aligned history, and a disagreement is an error. For the
  fan-out, whose figures all come from the engines the single pass already runs, what can be wrong
  is what the fan-out itself does: every number in every finding must appear in its researcher's
  own answer (verbatim compression), and every citation must resolve to a source that researcher
  returned. The engines' own figures behind a finding (a Skill's RSI, an analyst target) are not
  recomputed here — **NOT VERIFIED** by this module; they are the same figures the single pass
  shows, checked, where they are, by those engines' own tests.
* **Latency** — wall clock per answer, live. Order effects are real (a second call finds the candle
  cache warm), so the fan-out and the single pass alternate which runs first, and both orders are
  reported.

A loss is recorded as a loss: the fan-out is slower and longer, and those figures are published
beside the gains.

**Results, runs of 2026-09-25 and 2026-09-26.**

* Stress tree, 84 stress questions seen: 61 compared, 12 stated no book, 11 misread by the
  console's reader and set aside. Scenario angles, median: 3 in the one-pass answer, 10 in the
  tree, 11 in the two together; the tree has more on 61 of 61. The one-pass answer has two angles
  the tree does not compute (hedge sizing, loss share), which is why the tree is appended to that
  answer and never replaces it. Recomputation: 0 errors in 1,300 tree figures; **28 errors in 277
  one-pass figures**, every one the realised-window line on a multi-name book (28 of 28 such
  books), every one reproduced exactly by the weights `desk/portfolio.rebalance` makes from the
  STRESS branch's ``copilot`` call — a defect in the one-pass answer, with the fix
  (`desk/stress_tree.book_worst_window`) waiting on its call site. Evidence: the tree adds no data
  source (it reads the same candles). Latency, median: 187ms inline, 603ms with a two-worker pool
  (slower on 45 of 61), against 455ms for the one-pass answer itself.
* Fan-out, 46 questions across twelve kinds: distinct cited sources median 2 → 13 (more on 46 of
  46), network sources reached 2 → 13.5; 131 of 131 researchers answered; 0 verbatim violations,
  0 bad citations. Slower on 45 of 46 (median +4.3s, p90 +10.3s) and about twice as long.
* Model supervisor against the plan table, 28 questions, 28 Qwen calls (11,457 tokens), all
  recorded in ``data/research_depth_qwen_calls.jsonl``: it lost — fewer cited sources on 14, more
  on 7, median 11 against 13, and 9.8s end to end against 6.0s.

**Contamination, stated.** The question corpora are `data/lui_*_2026-09-25.jsonl` (three held-out
routing corpora) and `data/research_bench_corpus_*.json` (read-only). The plan table and the tree's
catalogue were written before either was run on them; nothing in either was changed in response to
these rows. The reading of each question is the console's no-token path (the kind model through
`plan_with_model`, then the patterns where they win — `eval/figurecheck.py`'s simplification of
`lui/server.py`), and questions it misreads are counted and set aside, because a comparison of two
answers to a misread question measures the reader, not the answer.

**The model supervisor spends Qwen** — at most :data:`QWEN_CAP` calls, counted across runs from
the call log, and every paid response is appended to ``data/research_depth_qwen_calls.jsonl`` the
moment it arrives, before it is parsed. No test calls it.

    python -m argus.eval.research_depth stress --save
    python -m argus.eval.research_depth fanout --save
    python -m argus.eval.research_depth supervisor --save --calls 28   # spends Qwen
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from argus.eval import artefact

DATA = Path(__file__).resolve().parents[3] / "data"
STRESS_REPORT = DATA / "research_depth_stress.json"
STRESS_ROWS = DATA / "research_depth_stress_rows.jsonl"
FANOUT_REPORT = DATA / "research_depth_fanout.json"
FANOUT_ROWS = DATA / "research_depth_fanout_rows.jsonl"
SUPERVISOR_ROWS = DATA / "research_depth_supervisor_rows.jsonl"
QWEN_LOG = DATA / "research_depth_qwen_calls.jsonl"
QWEN_CAP = 30
"""The Qwen allowance for this evaluation, across every run of it."""

LUI_CORPORA = ("lui_blind_corpus_2026-09-25.jsonl", "lui_heldout_corpus_2026-09-25.jsonl",
               "lui_final_heldout_2026-09-25.jsonl")
BENCH_CORPORA = ("research_bench_corpus_a.json", "research_bench_corpus_b.json",
                 "research_bench_corpus_c.json", "research_bench_corpus_d.json")

PER_KIND = 4
"""Fan-out questions taken per kind, in corpus order: enough for at least thirty across the kinds
the plan covers, without letting the commonest kind dominate the averages."""

TOLERANCE_2DP = 0.006
"""How far a figure printed to two decimals may sit from its recomputation: half a unit in the last
place, plus float slack."""


@dataclass(frozen=True)
class Item:
    corpus: str
    id: str
    text: str
    label: str


def items() -> list[Item]:
    """Every question in the held-out routing corpora and the research bench, in file order."""
    out: list[Item] = []
    for name in LUI_CORPORA:
        path = DATA / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out.append(Item(name, str(row["id"]), str(row["text"]), str(row["expected"])))
    for name in BENCH_CORPORA:
        path = DATA / name
        if not path.exists():
            continue
        for i, question in enumerate(json.loads(path.read_text(encoding="utf-8"))["questions"]):
            out.append(Item(name, str(i), str(question["text"]), str(question["label"])))
    return out


LABEL_KIND = {
    "impact": "impact", "profile": "impact", "stress": "stress", "compare": "compare",
    "execution": "execution", "quote": "quote", "technicals": "technicals",
    "fundamentals": "fundamentals", "analogue": "analogue", "event": "event", "hedge": "hedge",
    "macro": "macro", "sentiment": "sentiment", "news": "news",
}


def read(text: str) -> Any:
    """The request the console would build with no language model: the kind model's plan, then
    the patterns where they win. `eval/figurecheck.py` reads the corpora the same way."""
    from argus.lui.kindmodel import LocalPlanner, kind_model
    from argus.lui.research import (
        detect,
        pattern_reading_wins,
        plan_with_model,
        price_forecast_asked,
    )

    model = kind_model()
    planned = plan_with_model(text, LocalPlanner(model))[0] if model is not None else None
    if planned is not None and price_forecast_asked(text):
        planned = None
    patterned = detect(text)
    if patterned is not None and (planned is None or pattern_reading_wins(patterned, text)):
        return patterned
    return planned


# =============================================================================================
# Independent recomputation (numpy, from the definitions)
# =============================================================================================


@dataclass(frozen=True)
class Frame:
    """The loaded history, aligned again here without `desk/portfolio.align`."""

    stamps: list[datetime]
    cols: dict[str, np.ndarray]
    open_mask: np.ndarray

    @classmethod
    def build(cls, raw: Mapping[str, Mapping[datetime, float]],
              is_open: Callable[[datetime], bool]) -> Frame:
        common = set.intersection(*(set(v) for v in raw.values()))
        stamps = sorted(common)
        cols = {k: np.array([v[t] for t in stamps], dtype=float) for k, v in raw.items()}
        return cls(stamps, cols, np.array([bool(is_open(t)) for t in stamps]))

    def session(self, open_session: bool) -> dict[str, np.ndarray]:
        mask = self.open_mask if open_session else ~self.open_mask
        return {k: v[mask] for k, v in self.cols.items()}


def np_beta(asset: np.ndarray, bench: np.ndarray) -> float | None:
    if len(asset) < 20 or len(asset) != len(bench):
        return None
    var = float(np.var(bench, ddof=1))
    if var <= 0:
        return None
    return float(np.cov(asset, bench, ddof=1)[0, 1]) / var


def np_book_beta(weights: Mapping[str, float], cols: Mapping[str, np.ndarray],
                 bench: str) -> float:
    total = 0.0
    for symbol, weight in weights.items():
        if symbol in cols:
            b = np_beta(cols[symbol], cols[bench])
            if b is not None:
                total += weight * b
    return total


def np_shock(weights: Mapping[str, float], cols: Mapping[str, np.ndarray], bench: str,
             shock: float) -> float:
    return np_book_beta(weights, cols, bench) * shock


def np_worst(weights: Mapping[str, float], cols: Mapping[str, np.ndarray],
             bars: int = 24) -> tuple[float, int]:
    names = [s for s, w in weights.items() if w != 0.0 and s in cols]
    matrix = np.vstack([1.0 + cols[s] for s in names])
    growth = np.lib.stride_tricks.sliding_window_view(matrix, bars, axis=1).prod(axis=2) - 1.0
    book = np.array([weights[s] for s in names]) @ growth
    start = int(np.argmin(book))
    return float(book[start]) * 100.0, start


def np_cvar(weights: Mapping[str, float], cols: Mapping[str, np.ndarray],
            confidence: float = 0.95) -> tuple[float, dict[str, float]]:
    """skfolio's historical CVaR (the definition `desk/portfolio.tail_contributions` cites), and
    each position's Euler contribution, written again from the formula."""
    names = [s for s, w in weights.items() if w != 0.0 and s in cols]
    n = min(len(cols[s]) for s in names)
    matrix = np.vstack([cols[s][:n] for s in names])
    w = np.array([weights[s] for s in names])
    book = w @ matrix
    k = (1.0 - confidence) * n
    ik = max(0, math.ceil(k) - 1)
    order = np.argsort(book, kind="stable")
    worst, edge = order[:ik], order[ik]
    weight_edge = ik / k - 1.0

    def loss(series: np.ndarray) -> float:
        return float(-series[worst].sum() / k + series[edge] * weight_edge)

    cvar = loss(book)
    return cvar, {s: weights[s] * loss(matrix[i]) for i, s in enumerate(names)}


def np_frequency(returns: Sequence[float], bars: int, shock: float) -> tuple[int, int, float]:
    levels = np.cumprod(1.0 + np.array(returns, dtype=float))
    moves = (levels[bars:] / levels[:-bars] - 1.0) * 100.0
    hits = moves <= shock if shock < 0 else moves >= shock
    extreme = float(moves.min() if shock < 0 else moves.max())
    return int(hits.sum()), len(moves), extreme


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8)


def recompute_tree(tree: Any, raw: Mapping[str, Mapping[datetime, float]],
                   is_open: Callable[[datetime], bool]) -> list[dict[str, Any]]:
    """Every answered node's figures, recomputed; one row per figure checked."""
    from argus.desk.stress_tree import HEDGE_COLUMN, RISK_BUDGET, TRIM_STEPS

    inputs = tree.inputs
    frame = Frame.build(raw, is_open)
    weights = {s: w for s, w in inputs.weights.items() if s in frame.cols}
    shocked, shock = inputs.shocked, inputs.shock_pct
    open_cols, shut_cols = frame.session(True), frame.session(False)
    hedge_key = HEDGE_COLUMN.format(symbol=inputs.shocked_label)
    rows: list[dict[str, Any]] = []

    def check(node: Any, name: str, want: float) -> None:
        got = node.figures.get(name)
        rows.append({"node": node.id, "angle": node.angle, "figure": name, "stated": got,
                     "recomputed": want, "ok": got is not None and _close(float(got), want)})

    for node in tree.answered:
        f, angle = node.figures, node.angle
        if angle == "beta_shock":
            check(node, "book_beta", np_book_beta(weights, open_cols, shocked))
            check(node, "book_move_pct", np_shock(weights, open_cols, shocked, shock))
        elif angle in ("hedge_residual", "hedged_tail"):
            hedge = -np_book_beta(weights, open_cols, shocked)
            hedged_w = {**weights, hedge_key: hedge}
            hedged_c = {**frame.cols, hedge_key: frame.cols[shocked]}
            check(node, "hedge_weight" if angle == "hedge_residual" else "hedged_cvar",
                  hedge if angle == "hedge_residual" else np_cvar(hedged_w, hedged_c)[0])
            if angle == "hedge_residual":
                check(node, "unhedged_pct", np_worst(weights, frame.cols)[0])
                check(node, "hedged_pct", np_worst(hedged_w, hedged_c)[0])
            else:
                check(node, "unhedged_cvar", np_cvar(weights, frame.cols)[0])
        elif angle == "idiosyncratic":
            symbol = node.subject
            own, _ = np_worst({symbol: weights[symbol]}, frame.cols)
            check(node, "own_worst_book_pct", own)
            b = np_beta(open_cols[symbol], open_cols[shocked]) or 0.0
            check(node, "beta_implied_book_pct", weights[symbol] * b * shock)
        elif angle == "realised_window":
            move, start = np_worst(weights, frame.cols)
            check(node, "move_pct", move)
            check(node, "open_hours", float(frame.open_mask[start:start + 24].sum()))
        elif angle == "window_driver":
            halved = {**weights, node.subject: weights[node.subject] / 2.0}
            check(node, "halved_move_pct", np_worst(halved, frame.cols)[0])
        elif angle == "tail":
            cvar, contributions = np_cvar(weights, frame.cols)
            gross = sum(abs(w) for w in weights.values())
            excess = {s: contributions[s] / cvar - abs(weights[s]) / gross for s in contributions}
            top = max(excess, key=lambda s: excess[s])
            check(node, "cvar", cvar)
            check(node, "top_share", contributions[top] / cvar)
        elif angle == "tail_trim":
            symbol = node.subject
            target = f["to_weight"]
            trimmed = {**weights, symbol: target}
            trimmed = {s: w for s, w in trimmed.items() if w != 0.0}
            cvar, contributions = np_cvar(trimmed, frame.cols)
            check(node, "cvar_after", cvar)
            check(node, "share_after", contributions.get(symbol, 0.0) / cvar)
            step = f["from_weight"] / TRIM_STEPS
            if target + step <= f["from_weight"] + 1e-12:
                above = {**weights, symbol: target + step}
                c2, k2 = np_cvar(above, frame.cols)
                rows.append({"node": node.id, "angle": angle, "figure": "largest_grid_weight",
                             "stated": target, "recomputed": target + step,
                             "ok": k2[symbol] / c2 > RISK_BUDGET})
        elif angle == "session_gap":
            check(node, "open_beta", np_book_beta(weights, open_cols, shocked))
            check(node, "shut_beta", np_book_beta(weights, shut_cols, shocked))
        elif angle == "shut_shock":
            check(node, "shut_move_pct", np_shock(weights, shut_cols, shocked, shock))
            check(node, "open_move_pct", np_shock(weights, open_cols, shocked, shock))
        elif angle == "shock_frequency":
            series = [v for _, v in sorted(raw[shocked].items())]
            hits, total, extreme = np_frequency(series, 24, shock)
            check(node, "occurrences", float(hits))
            check(node, "observations", float(total))
            check(node, "extreme_pct", extreme)
        elif angle == "history_shock":
            check(node, "book_move_at_extreme_pct",
                  np_shock(weights, open_cols, shocked, f["extreme_pct"]))
            check(node, "book_move_at_stated_pct", np_shock(weights, open_cols, shocked, shock))
    return rows


_SHOCK_LINE = re.compile(r"If (?P<name>\S+) moves (?P<shock>[-+]?\d+(?:\.\d+)?)%: your book moves "
                         r"about (?P<move>[-+]\d+\.\d+)%")
_WINDOW_LINE = re.compile(r"worst 24-bar window in the observed history would have moved this "
                          r"book (?P<move>[-+]\d+\.\d+)%")
_SHARE_LINE = re.compile(r"(?P<name>[A-Z0-9]+) is (?P<w>\d+)% of the book but (?P<s>\d+)% of its "
                         r"loss")
_HEDGE_LINE = re.compile(r"Hedge: (?:short|long) (?P<name>\S+) worth about (?P<b>\d+)% of the "
                         r"book's value neutralises its market exposure \(book beta "
                         r"(?P<beta>-?\d+\.\d+)")


def recompute_single(lines: Sequence[str], book: Mapping[str, float], shocked: str,
                     raw: Mapping[str, Mapping[datetime, float]],
                     is_open: Callable[[datetime], bool]) -> list[dict[str, Any]]:
    """The one-pass stress answer's checkable figures, recomputed on the book it was asked about.

    A realised-window figure that disagrees is also recomputed on the weights
    `desk/portfolio.rebalance` produces from the arguments the STRESS branch passes it
    (``before`` = the book less its first name, ``size`` = that name's weight), so a mismatch can be
    attributed to its cause rather than only counted."""
    from argus.desk.portfolio import rebalance

    frame = Frame.build(raw, is_open)
    weights = {s: w for s, w in book.items() if s in frame.cols}
    open_cols = frame.session(True)
    rows: list[dict[str, Any]] = []
    text = "\n".join(lines)
    for m in _SHOCK_LINE.finditer(text):
        want = np_shock(weights, open_cols, shocked, float(m.group("shock")))
        rows.append({"figure": f"beta shock {m.group('shock')}%", "stated": float(m.group("move")),
                     "recomputed": want,
                     "ok": abs(float(m.group("move")) - want) <= TOLERANCE_2DP})
    for m in _WINDOW_LINE.finditer(text):
        want, _ = np_worst(weights, frame.cols)
        row: dict[str, Any] = {"figure": "realised worst 24h", "stated": float(m.group("move")),
                               "recomputed": want,
                               "ok": abs(float(m.group("move")) - want) <= TOLERANCE_2DP}
        if not row["ok"]:
            add = next(iter(book))
            before = {s: w for s, w in book.items() if s != add} or {}
            used = rebalance(before, add, book[add] if len(book) > 1 else 1.0)
            as_used, _ = np_worst({s: w for s, w in used.items() if s in frame.cols}, frame.cols)
            row.update(weights_used=used, recomputed_on_weights_used=as_used,
                       explained_by_weights_used=abs(float(m.group("move")) - as_used)
                       <= TOLERANCE_2DP)
        rows.append(row)
    total_beta = np_book_beta(weights, open_cols, shocked)
    for m in _SHARE_LINE.finditer(text):
        symbol = next((s for s in weights if s.removesuffix("USDT") == m.group("name")), None)
        if symbol is None or total_beta == 0:
            continue
        b = np_beta(open_cols[symbol], open_cols[shocked]) or 0.0
        share = weights[symbol] * b / total_beta * 100.0
        rows.append({"figure": f"loss share {m.group('name')}", "stated": float(m.group("s")),
                     "recomputed": share, "ok": abs(float(m.group("s")) - share) <= 0.51})
    for m in _HEDGE_LINE.finditer(text):
        rows.append({"figure": "hedge book beta", "stated": float(m.group("beta")),
                     "recomputed": total_beta,
                     "ok": abs(float(m.group("beta")) - total_beta) <= TOLERANCE_2DP})
    return rows


SINGLE_ANGLES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("beta_shock", re.compile(r"your book moves about|this book moves about|barely moves with")),
    ("realised_window", re.compile(r"What actually happened")),
    ("var", re.compile(r"Value at risk|Volatility at \d")),
    ("hedge_sizing", re.compile(r"^Hedge:", re.M)),
    ("loss_share", re.compile(r"of its loss when|loss is spread roughly")),
    ("single_name", re.compile(r"if one name falls")),
)
"""The scenario angles a one-pass stress answer can contain, read from its own wording."""


def single_angles(lines: Sequence[str]) -> list[str]:
    text = "\n".join(lines)
    return sorted({angle for angle, pattern in SINGLE_ANGLES if pattern.search(text)})


def evidence(answer: Any) -> dict[str, Any]:
    """An answer's size and its sources: cited ``(kind, ref)`` pairs and network sources reached."""
    lines = [str(line) for line in getattr(answer, "lines", []) or []]
    cited = {(str(s.kind), str(s.ref)) for s in getattr(answer, "sources", []) or []}
    data = getattr(answer, "data", {}) or {}
    fan = data.get("fanout") or {}
    reached = fan.get("reached") if fan.get("merged") else (data.get("coverage") or {}).get(
        "reached", [])
    return {"lines": len(lines), "chars": sum(len(line) for line in lines),
            "cited_sources": len(cited), "reached_sources": len(reached or []),
            "refused": bool(getattr(answer, "refused", False))}


# =============================================================================================
# Part 1 — the stress tree against the one-pass stress answer
# =============================================================================================


def _done_keys(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["key"]] = row
    return out


def _append(path: Path, row: Mapping[str, Any]) -> None:
    """Write one row and push it to disk before the next question starts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(artefact.sanitise(dict(row)), allow_nan=False, default=str,
                      ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _median_ms(fn: Callable[[], Any], runs: int = 3) -> tuple[float, Any]:
    times: list[float] = []
    result: Any = None
    for _ in range(runs):
        started = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - started) * 1000)
    return statistics.median(times), result


def stress_row(item: Item) -> dict[str, Any]:
    from argus.desk import stress_tree
    from argus.lui import research

    key = f"{item.corpus}#{item.id}"
    base: dict[str, Any] = {"key": key, "corpus": item.corpus, "text": item.text}
    request = read(item.text)
    if request is None or request.kind is not research.ResearchKind.STRESS:
        return {**base, "status": "misread",
                "read_as": None if request is None else str(request.kind)}
    book = dict(request.book)
    if not request.symbols and request.shock_on and not request.cash:
        book = {request.shock_on: 1.0}  # the STRESS branch's own reading of a book-less shock
    if request.cash >= 0.999 or not book:
        return {**base, "status": "no book stated", "request": request.as_dict()}
    started = time.perf_counter()
    single = research.run(item.text, request)
    single_ms = (time.perf_counter() - started) * 1000
    if single.refused:
        return {**base, "status": "single pass refused", "reason": single.reason}
    shocked = request.shock_on or research.BENCHMARK
    data = research.load((*book, shocked))
    is_open = research._is_open()
    shock = request.shock_pct if request.shock_pct not in (None, 0) else -10.0
    inputs = stress_tree.prepare(data.raw, is_open=is_open, weights=book, shocked=shocked,
                                 shock_pct=float(shock), cash=request.cash)
    ms_seq, tree = _median_ms(lambda: stress_tree.grow(inputs, concurrency=1))
    ms_par, _ = _median_ms(lambda: stress_tree.grow(inputs, concurrency=2))
    tree_checks = recompute_tree(tree, data.raw, is_open)
    single_checks = recompute_single([str(x) for x in single.lines], book, shocked, data.raw,
                                     is_open)
    one = single_angles([str(x) for x in single.lines])
    ev = evidence(single)
    return {
        **base, "status": "compared", "live": data.live, "book": book, "cash": request.cash,
        "shocked": shocked, "shock_pct": shock, "shock_stated": request.shock_pct is not None,
        "single": {**ev, "latency_ms": round(single_ms, 1), "angles": one},
        "tree": {"latency_ms_concurrency_1": round(ms_seq, 2),
                 "latency_ms_concurrency_2": round(ms_par, 2),
                 "answered": len(tree.answered), "levels": tree.levels, "angles": tree.angles,
                 "stops": tree.stops, "lines": tree.render(),
                 "not_answered": [n.as_dict() for n in tree.nodes if n.status != "answered"]},
        "integrated_angles": sorted(set(one) | set(tree.angles)),
        "tree_errors": [c for c in tree_checks if not c["ok"]],
        "tree_figures_checked": len(tree_checks),
        "single_errors": [c for c in single_checks if not c["ok"]],
        "single_figures_checked": len(single_checks),
    }


def summarise_stress(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    compared = [r for r in rows if r.get("status") == "compared"]
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[str(r.get("status"))] = statuses.get(str(r.get("status")), 0) + 1

    def med(values: Sequence[float]) -> float | None:
        return round(statistics.median(values), 2) if values else None

    single_cov = [len(r["single"]["angles"]) for r in compared]
    tree_cov = [len(r["tree"]["angles"]) for r in compared]
    union_cov = [len(r["integrated_angles"]) for r in compared]
    single_err = sum(len(r["single_errors"]) for r in compared)
    single_chk = sum(r["single_figures_checked"] for r in compared)
    tree_err = sum(len(r["tree_errors"]) for r in compared)
    tree_chk = sum(r["tree_figures_checked"] for r in compared)
    seq = [r["tree"]["latency_ms_concurrency_1"] for r in compared]
    par = [r["tree"]["latency_ms_concurrency_2"] for r in compared]
    angle_counts: dict[str, int] = {}
    for r in compared:
        for a in r["tree"]["angles"]:
            angle_counts[a] = angle_counts.get(a, 0) + 1
    window_errors = [e for r in compared for e in r["single_errors"]
                     if e["figure"] == "realised worst 24h"]
    return {
        "questions_seen": len(rows), "by_status": statuses, "compared": len(compared),
        "scenario_coverage": {
            "single_pass_median_angles": med(single_cov), "tree_median_angles": med(tree_cov),
            "single_plus_tree_median_angles": med(union_cov),
            "tree_more_angles_than_single": sum(t > s for t, s in zip(tree_cov, single_cov,
                                                                      strict=True)),
            "tree_fewer_angles_than_single": sum(t < s for t, s in zip(tree_cov, single_cov,
                                                                       strict=True)),
            "single_angles_the_tree_lacks": sorted({a for r in compared
                                                    for a in r["single"]["angles"]
                                                    if a not in r["tree"]["angles"]}),
            "tree_angle_frequency": dict(sorted(angle_counts.items())),
        },
        "factual_errors_by_recomputation": {
            "single_pass": {"figures_checked": single_chk, "errors": single_err,
                            "realised_window_errors": len(window_errors),
                            "realised_window_errors_explained_by_rebalanced_weights": sum(
                                1 for e in window_errors if e.get("explained_by_weights_used"))},
            "tree": {"figures_checked": tree_chk, "errors": tree_err},
        },
        "distinct_evidence_sources": {
            "single_pass_median_cited": med([r["single"]["cited_sources"] for r in compared]),
            "note": "the tree reads the same candles the single pass loaded: it adds one "
                    "computation source (argus.desk.stress_tree) and no data source",
        },
        "latency_ms": {
            "single_pass_median": med([r["single"]["latency_ms"] for r in compared]),
            "tree_median_concurrency_1": med(seq), "tree_median_concurrency_2": med(par),
            "tree_p90_concurrency_1": (round(sorted(seq)[int(0.9 * (len(seq) - 1))], 2)
                                       if seq else None),
            "concurrency_2_slower_on": sum(p > s for p, s in zip(par, seq, strict=True)),
        },
    }


def run_stress(limit: int | None) -> dict[str, Any]:
    done = _done_keys(STRESS_ROWS)
    targets = [i for i in items() if i.label == "stress"]
    ran = 0
    for item in targets:
        key = f"{item.corpus}#{item.id}"
        if key in done:
            continue
        if limit is not None and ran >= limit:
            break
        try:
            row = stress_row(item)
        except Exception as exc:  # a harness failure is recorded against its question, not hidden
            row = {"key": key, "corpus": item.corpus, "text": item.text, "status": "harness error",
                   "error": f"{type(exc).__name__}: {exc}"}
        _append(STRESS_ROWS, row)
        done[key] = row
        ran += 1
        print(f"[stress] {key}: {row['status']}")
    rows = [done[f"{i.corpus}#{i.id}"] for i in targets if f"{i.corpus}#{i.id}" in done]
    return {"generated_at": datetime.now(UTC).isoformat(), "part": "stress tree",
            "complete": len(rows) == len(targets), "summary": summarise_stress(rows),
            "rows": rows}


# =============================================================================================
# Part 2 — the fan-out against the single pass
# =============================================================================================


def fanout_selection() -> list[tuple[Item, Any]]:
    """Up to :data:`PER_KIND` questions per kind the plan covers, read correctly by the console
    and with at least one researcher to add, in corpus order."""
    from argus.lui.fanout import PLAN, plan_units

    taken: dict[str, int] = {}
    out: list[tuple[Item, Any]] = []
    for item in items():
        kind = LABEL_KIND.get(item.label)
        if kind is None or kind == "stress" or taken.get(kind, 0) >= PER_KIND:
            continue
        request = read(item.text)
        if request is None or str(request.kind) != kind or request.kind not in PLAN:
            continue
        if not plan_units(request):
            continue
        taken[kind] = taken.get(kind, 0) + 1
        out.append((item, request))
    return out


def _fan(text: str, request: Any, units: Sequence[Any], planner: str,
         audit: Mapping[str, Any]) -> tuple[Any, Any, float]:
    from argus.lui import fanout, research

    started = time.perf_counter()
    result = fanout.run_units(text, request, units, research.run, planner=planner, audit=audit)
    merged = fanout.merge(result, overflow=audit.get("overflow", []))
    return merged, result, (time.perf_counter() - started) * 1000


def fanout_integrity(merged: Any, result: Any) -> dict[str, Any]:
    """What the fan-out itself could have got wrong: numbers a finding changed, and citations that
    do not resolve to the researcher that made the finding."""
    from argus.lui.fanout import _source_key, verbatim_violations

    violations = {f.unit.id: verbatim_violations(f) for f in result.findings if f.answered}
    bad_citations: list[str] = []
    records = ((merged.data or {}).get("fanout") or {}).get("findings", [])
    for finding, record in zip(result.findings, records, strict=False):
        own = {_source_key(s) for s in finding.sources}
        for n in record.get("sources", []):
            if not 1 <= n <= len(merged.sources) or _source_key(merged.sources[n - 1]) not in own:
                bad_citations.append(f"{finding.unit.id}:[{n}]")
    return {"verbatim_violations": sum(len(v) for v in violations.values()),
            "violations": {k: v for k, v in violations.items() if v},
            "bad_citations": bad_citations}


def fanout_row(item: Item, request: Any, position: int) -> dict[str, Any]:
    from argus.lui import fanout, research

    key = f"{item.corpus}#{item.id}"
    units = fanout.plan_units(request)
    first = "single" if position % 2 == 0 else "fanout"

    def single() -> tuple[Any, float]:
        started = time.perf_counter()
        answer = research.run(item.text, request)
        return answer, (time.perf_counter() - started) * 1000

    if first == "single":
        one, one_ms = single()
        merged, result, fan_ms = _fan(item.text, request, units, "table", {"planner": "table"})
    else:
        merged, result, fan_ms = _fan(item.text, request, units, "table", {"planner": "table"})
        one, one_ms = single()
    ev_one, ev_fan = evidence(one), evidence(merged)
    return {
        "key": key, "corpus": item.corpus, "text": item.text, "kind": str(request.kind),
        "first": first, "units": [u.id for u in units],
        "single": {**ev_one, "latency_ms": round(one_ms, 1)},
        "fanout": {**ev_fan, "latency_ms": round(fan_ms, 1),
                   "answered": sum(f.answered for f in result.findings),
                   "not_answered": {f.unit.id: f.reason for f in result.findings
                                    if not f.answered},
                   "merged": bool(((merged.data or {}).get("fanout") or {}).get("merged")),
                   **fanout_integrity(merged, result)},
        "fanout_lines": [str(x) for x in merged.lines],
    }


def _compare(rows: Sequence[Mapping[str, Any]], a: str, b: str, metric: str) -> dict[str, int]:
    wins = sum(1 for r in rows if r[b][metric] > r[a][metric])
    ties = sum(1 for r in rows if r[b][metric] == r[a][metric])
    return {"more": wins, "same": ties, "fewer": len(rows) - wins - ties}


def summarise_fanout(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def med(values: Sequence[float]) -> float | None:
        return round(statistics.median(values), 1) if values else None

    added = [r["fanout"]["latency_ms"] - r["single"]["latency_ms"] for r in rows]
    by_kind: dict[str, dict[str, Any]] = {}
    for r in rows:
        bucket = by_kind.setdefault(r["kind"], {"questions": 0, "units_answered": 0,
                                                "units_planned": 0})
        bucket["questions"] += 1
        bucket["units_answered"] += r["fanout"]["answered"]
        bucket["units_planned"] += len(r["units"])
    reasons: dict[str, int] = {}
    for r in rows:
        for uid, why in r["fanout"]["not_answered"].items():
            label = f"{uid.split(':')[0]}: {why[:80]}"
            reasons[label] = reasons.get(label, 0) + 1
    return {
        "questions": len(rows),
        "distinct_cited_sources": {
            "single_median": med([r["single"]["cited_sources"] for r in rows]),
            "fanout_median": med([r["fanout"]["cited_sources"] for r in rows]),
            "fanout_vs_single": _compare(rows, "single", "fanout", "cited_sources"),
        },
        "network_sources_reached": {
            "single_median": med([r["single"]["reached_sources"] for r in rows]),
            "fanout_median": med([r["fanout"]["reached_sources"] for r in rows]),
            "fanout_vs_single": _compare(rows, "single", "fanout", "reached_sources"),
        },
        "researchers": {
            "planned": sum(len(r["units"]) for r in rows),
            "answered": sum(r["fanout"]["answered"] for r in rows),
            "not_answered_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
            "by_kind": by_kind,
        },
        "factual_errors_by_recomputation": {
            "verbatim_violations": sum(r["fanout"]["verbatim_violations"] for r in rows),
            "bad_citations": sum(len(r["fanout"]["bad_citations"]) for r in rows),
            "not_verified": "the engines' own figures inside each finding are the single pass's "
                            "figures and are not recomputed here",
        },
        "latency_ms": {
            "single_median": med([r["single"]["latency_ms"] for r in rows]),
            "fanout_median": med([r["fanout"]["latency_ms"] for r in rows]),
            "added_median": med(added),
            "added_p90": round(sorted(added)[int(0.9 * (len(added) - 1))], 1) if added else None,
            "single_first_added_median": med([r["fanout"]["latency_ms"] - r["single"]["latency_ms"]
                                              for r in rows if r["first"] == "single"]),
            "fanout_first_added_median": med([r["fanout"]["latency_ms"] - r["single"]["latency_ms"]
                                              for r in rows if r["first"] == "fanout"]),
            "fanout_slower_on": sum(1 for a in added if a > 0),
        },
        "answer_length_chars": {
            "single_median": med([r["single"]["chars"] for r in rows]),
            "fanout_median": med([r["fanout"]["chars"] for r in rows]),
        },
        "single_refused": sum(r["single"]["refused"] for r in rows),
        "fanout_not_merged": sum(not r["fanout"]["merged"] for r in rows),
    }


def run_fanout(limit: int | None) -> dict[str, Any]:
    done = _done_keys(FANOUT_ROWS)
    selected = fanout_selection()
    ran = 0
    for position, (item, request) in enumerate(selected):
        key = f"{item.corpus}#{item.id}"
        if key in done:
            continue
        if limit is not None and ran >= limit:
            break
        try:
            row = fanout_row(item, request, position)
        except Exception as exc:
            row = {"key": key, "text": item.text, "status": "harness error",
                   "error": f"{type(exc).__name__}: {exc}"}
        _append(FANOUT_ROWS, row)
        done[key] = row
        ran += 1
        print(f"[fanout] {key} {row.get('kind', '')}: "
              f"{row.get('fanout', {}).get('answered', '-')} answered")
    rows = [done[f"{i.corpus}#{i.id}"] for i, _ in selected if f"{i.corpus}#{i.id}" in done]
    good = [r for r in rows if "fanout" in r]
    return {"generated_at": datetime.now(UTC).isoformat(), "part": "fan-out",
            "selected": len(selected), "complete": len(rows) == len(selected),
            "harness_errors": [r for r in rows if "fanout" not in r],
            "summary": summarise_fanout(good), "rows": rows}


# =============================================================================================
# Part 3 — a model supervisor against the plan table (spends Qwen, capped)
# =============================================================================================


def calls_spent() -> int:
    """Paid calls already made by this evaluation, read from its own write-through log."""
    if not QWEN_LOG.exists():
        return 0
    return sum(1 for line in QWEN_LOG.read_text(encoding="utf-8").splitlines() if line.strip())


class RecordingClient:
    """A ``complete_json`` for :func:`argus.lui.fanout.plan_units_with_model` that spends one
    Qwen call per request, never more, and writes every response to disk the moment it arrives."""

    def __init__(self, inner: Any, *, cap: int, log: Path = QWEN_LOG) -> None:
        self._inner = inner
        self._cap = cap
        self._log = log
        self.calls = 0
        self.question = ""

    def complete_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        from argus.llm.qwen import BudgetExhausted, QwenError, Thinking, extract_json_object

        if self.calls >= self._cap:
            raise BudgetExhausted(f"the evaluation's {self._cap}-call allowance is spent")
        self.calls += 1
        record: dict[str, Any] = {
            "at": datetime.now(UTC).isoformat(), "question": self.question,
            "prompt_sha256": hashlib.sha256(json.dumps(messages).encode()).hexdigest()}
        started = time.perf_counter()
        try:
            got = self._inner.complete(messages, json_mode=True,
                                       max_tokens=int(kwargs.get("max_tokens", 400)),
                                       thinking=Thinking.OFF)
        except Exception as exc:
            _append(self._log, {**record, "ok": False, "error": f"{type(exc).__name__}: {exc}",
                                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)})
            raise
        usage = got.usage
        _append(self._log, {**record, "ok": True, "content": got.content,
                            "finish_reason": got.finish_reason,
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens,
                            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)})
        try:
            parsed = json.loads(extract_json_object(got.content))
        except ValueError as exc:
            raise QwenError(f"the supervisor's reply was not JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise QwenError("the supervisor's reply was not a JSON object")
        return parsed


def round_robin(selected: Sequence[tuple[Item, Any]]) -> list[tuple[Item, Any]]:
    """The fan-out selection reordered one kind at a time — the first question of every kind, then
    the second of every kind — so a Qwen allowance smaller than the selection is spread across the
    kinds instead of spent on the first few in corpus order."""
    rank: dict[str, int] = {}
    keyed: list[tuple[int, int, tuple[Item, Any]]] = []
    for position, pair in enumerate(selected):
        kind = str(pair[1].kind)
        keyed.append((rank.get(kind, 0), position, pair))
        rank[kind] = rank.get(kind, 0) + 1
    return [pair for _, _, pair in sorted(keyed, key=lambda row: (row[0], row[1]))]


def run_supervisor(calls: int) -> dict[str, Any]:
    from argus.llm.qwen import QwenClient
    from argus.lui import fanout

    remaining = min(calls, QWEN_CAP - calls_spent())
    done = _done_keys(SUPERVISOR_ROWS)
    selected = round_robin(fanout_selection())
    if remaining > 0:
        client = RecordingClient(QwenClient(max_retries=1, cache=False), cap=remaining)
        for position, (item, request) in enumerate(selected):
            key = f"{item.corpus}#{item.id}"
            if key in done:
                continue
            if client.calls >= remaining:
                break
            client.question = key
            model_units, audit = fanout.plan_units_with_model(item.text, request, client)
            table_units = fanout.plan_units(request)
            order = ("table", "model") if position % 2 == 0 else ("model", "table")
            measured: dict[str, dict[str, Any]] = {}
            for planner in order:
                units = table_units if planner == "table" else model_units
                merged, result, ms = _fan(item.text, request, units, planner,
                                          audit if planner == "model" else {"planner": "table"})
                measured[planner] = {**evidence(merged), "latency_ms": round(ms, 1),
                                     "units": [u.id for u in units],
                                     "answered": sum(f.answered for f in result.findings),
                                     **fanout_integrity(merged, result)}
            row: dict[str, Any] = {"key": key, "text": item.text, "kind": str(request.kind),
                                   "first": order[0], "audit": audit, **measured}
            _append(SUPERVISOR_ROWS, row)
            done[key] = row
            print(f"[supervisor] {key}: model {measured['model']['units']} vs table "
                  f"{measured['table']['units']}")
    rows = [done[f"{i.corpus}#{i.id}"] for i, _ in selected if f"{i.corpus}#{i.id}" in done]

    def med(values: Sequence[float]) -> float | None:
        return round(statistics.median(values), 1) if values else None

    plan_ms: dict[str, float] = {}
    if QWEN_LOG.exists():
        for line in QWEN_LOG.read_text(encoding="utf-8").splitlines():
            if line.strip():
                call = json.loads(line)
                plan_ms[str(call.get("question", ""))] = float(call.get("elapsed_ms", 0.0))
    end_to_end = [r["model"]["latency_ms"] + plan_ms[r["key"]] for r in rows if r["key"] in plan_ms]
    fell_back = [r for r in rows if str(r["audit"].get("planner", "")).startswith("table")]
    same_plan = sum(1 for r in rows if sorted(r["model"]["units"]) == sorted(r["table"]["units"]))
    summary = {
        "questions": len(rows), "qwen_calls_spent_total": calls_spent(),
        "model_fell_back_to_table": len(fell_back),
        "identical_plans": same_plan,
        "rejected_ids": sum(len(r["audit"].get("rejected", [])) for r in rows),
        "overflow_ids": sum(len(r["audit"].get("overflow", [])) for r in rows),
        "answered_units": {"table": sum(r["table"]["answered"] for r in rows),
                           "model": sum(r["model"]["answered"] for r in rows)},
        "distinct_cited_sources": {
            "table_median": med([r["table"]["cited_sources"] for r in rows]),
            "model_median": med([r["model"]["cited_sources"] for r in rows]),
            "model_vs_table": _compare(rows, "table", "model", "cited_sources")},
        "network_sources_reached": {
            "table_median": med([r["table"]["reached_sources"] for r in rows]),
            "model_median": med([r["model"]["reached_sources"] for r in rows]),
            "model_vs_table": _compare(rows, "table", "model", "reached_sources")},
        "latency_ms": {"table_median": med([r["table"]["latency_ms"] for r in rows]),
                       "model_median": med([r["model"]["latency_ms"] for r in rows]),
                       "model_planning_call_median": med(list(plan_ms.values())),
                       "model_end_to_end_median": med(end_to_end),
                       "note": "model_median runs the model's chosen units only; the planning "
                               "call it waits for first is model_planning_call_median, read from "
                               "the call log, and the two together are model_end_to_end_median"},
        "integrity": {"verbatim_violations": sum(r[p]["verbatim_violations"] for r in rows
                                                 for p in ("table", "model")),
                      "bad_citations": sum(len(r[p]["bad_citations"]) for r in rows
                                           for p in ("table", "model"))},
    }
    return {"generated_at": datetime.now(UTC).isoformat(), "part": "model supervisor",
            "summary": summary, "rows": rows}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - live CLI
    parser = argparse.ArgumentParser(description="S24 research depth against the single pass")
    parser.add_argument("part", choices=("stress", "fanout", "supervisor"))
    parser.add_argument("--limit", type=int, default=None, help="questions to run this call")
    parser.add_argument("--calls", type=int, default=0, help="Qwen calls for the supervisor part")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args(argv)
    if args.part == "stress":
        report = run_stress(args.limit)
        path = STRESS_REPORT
    elif args.part == "fanout":
        report = run_fanout(args.limit)
        path = FANOUT_REPORT
    else:
        if not os.environ.get("BITGET_QWEN_API_KEY"):
            print("no Qwen key in the environment; the supervisor part spends none")
            return 1
        report = run_supervisor(args.calls)
        path = FANOUT_REPORT
        if path.exists():
            merged = json.loads(path.read_text(encoding="utf-8"))
            merged["supervisor"] = report
            report = merged
    print(json.dumps(report.get("summary") or report.get("supervisor", {}).get("summary"),
                     indent=2, default=str))
    if args.save:
        artefact.write(path, report)
        print(f"saved -> {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "Frame",
    "RecordingClient",
    "evidence",
    "fanout_integrity",
    "items",
    "np_beta",
    "np_cvar",
    "np_frequency",
    "np_worst",
    "read",
    "recompute_single",
    "recompute_tree",
    "single_angles",
    "summarise_fanout",
    "summarise_stress",
]
