"""Factor quality against FactorMiner's published libraries, through ARGUS's own gates.

Register row 34 ("Factor Discovery Agent: hypothesis to tradable factor") had a measured null for
ARGUS (`data/factor_split_half.json`: of 96 factor-instrument pairs, the 19 that clear the cost and
out-of-sample gates all fail split-half) and no rival measured. Whether that null is a tie (nobody
finds an edge on these instruments at this power) or a loss (a rival's library passes the same gate)
is what this module runs. The plan is `Activity/28_CAPABILITY_CLOSE_PLAN_2.md` section 34.

**The rival, run with its own engine.** FactorMiner (MIT,
`research/repos-t2/minihellboy~factorminer`) ships the paper's Appendix P catalogue,
``factorminer/core/library_io.py:PAPER_FACTORS`` (110 formulas: its own test pins the count at
``tests/test_expression_tree.py:119``; the plan's "112" was a miscount), and an Alpha101
catalogue, ``factorminer/benchmark/catalogs.py:ALPHA101_CLASSIC`` (12) with
``build_alpha101_adapted`` (60 window variants). The clone requires Python 3.12 (PEP 695
``type`` statements, e.g. ``factorminer/domain/evidence.py:13``) and ARGUS's venv is 3.11, so the
formulas are evaluated in a separate interpreter (default ``research/_venvs/factorminer``, its
dependencies pinned to the clone's ``uv.lock``) by :data:`DRIVER`, which calls FactorMiner's own
code end to end: ``data/preprocessor.compute_derived_features`` (``vwap``, ``returns``),
``data/tensor_builder.compute_target`` and ``build_tensor``, and
``application/validation_pipeline.ValidationPipeline.evaluate_candidate`` for the signals and for
FactorMiner's verdict. Nothing of FactorMiner is reimplemented here.

**FactorMiner's own verdict** is its Stage-1 quality gate at its defaults
(``validation_pipeline.py:41-42,224-237``): paper IC ``|mean RankIC| >= 0.04`` and paper ICIR
``|mean|/std >= 0.5``, cross-sectional Spearman IC per period (``evaluation/metrics.py``) against
its own target, the next bar's open-to-close return (``tensor_builder.compute_target``). The
correlation-admission stage after it depends on the order candidates arrive in and on the library
already held, so it is not a property of one factor; it is not scored.

**ARGUS's gates, identically.** Each factor-instrument pair goes through exactly what
``research/factor_lab.Evaluator.score`` and ``Evaluator.split_half`` apply to a primitive: the
backtest engine at ``CostModel.bitget_perp()`` (12 bps round trip) and ``HOURLY_PER_YEAR``, the cost
gate (net Sharpe > 0), the out-of-sample gate (a scoreable chronological OOS slice with Sharpe > 0),
and ``factor_lab.split_half`` at ``BLOCK = 120``. The lab cannot take these factors directly
(``Factor`` only accepts names in ``PRIMITIVES``, by design), so the gates are called with a signal
that reads precomputed weights; ARGUS's own eight run through the same function, and their rows are
checked field by field against `data/factor_split_half.json` before anything is written. A mismatch
raises. That parity is what makes "identical gates" a measured statement.

**From a cross-sectional score to a position.** FactorMiner's factors are cross-sectional scores
(``CsRank``) meant to rank a wide panel. Each bar, the twelve scores are ranked and mapped onto
``[-1, 1]`` (highest +1, lowest -1, ties averaged, missing flat): every instrument's weight is its
leg of the factor's own long-short book. Two further arms are run so the conversion cannot decide
the verdict alone:

* ``oriented`` flips a factor whose FactorMiner RankIC on the first 65% of bars is negative, since
  FactorMiner's gate scores ``|IC|`` and a practitioner would trade the sign it found. The
  orientation is fixed on the engine's in-sample slice, so the OOS gate stays out of sample (the
  split-half gate covers the whole window either way).
* ``placebo`` rotates the as-written weights by half the sample against the returns. It keeps each
  factor's turnover and cost, removes any link to what follows, and is the empirical null for all
  three gates together, beside the split-half noise rate measured on coin flips.

**Verdict rule** (the plan's, stated before the run): a rival family whose pass-all-gates rate,
split-half included, is above the calibrated noise rate (`data/factor_split_half.json`
``planted.noise``, re-derived from the frozen bars) at one-sided binomial p < 0.05 and above
ARGUS's own rate means **LOST**; every arm at the noise rate means **TIED**, and the reason it
cannot be won is written into the artefact. The binomial treats pairs as independent; they are not
(the twelve legs of one factor sum to zero each bar, and many formulas are near-duplicates), so the
p-value overstates significance, which can only push toward LOST, not away from it.

**Fairness caveat, published with the result:** a 12-name cross-section is thin for factors built
for wide A-share panels, and several use features that mean little here (``amount``/``vwap`` on
perpetual tokens, "overnight" on a market that trades around the clock). Results are reported per
FactorMiner category so a category that works is visible even when the family does not.

No model is called. The bars are the frozen fixture `data/factor_split_half_bars.json`.

    python -m argus.eval.factor_quality_rivals
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from argus.backtest.engine import Bar, run
from argus.backtest.metrics import HOURLY_PER_YEAR, MetricError
from argus.cost.model import CostModel
from argus.eval.factor_split_half import (
    BLOCK,
    FIXTURE,
    REPORT_PATH,
    SYMBOLS,
    bars_from_rows,
    load_fixture,
    next_returns,
)
from argus.research.factor_lab import PRIMITIVES, split_half
from argus.truth.paths import DATA_DIR, REPOSITORY_ROOT

OUT = DATA_DIR / "factor_quality_rivals.json"
FACTORMINER_ROOT = Path(os.environ.get(
    "ARGUS_FACTORMINER_ROOT",
    REPOSITORY_ROOT / "research" / "repos-t2" / "minihellboy~factorminer"))
FACTORMINER_PYTHON = Path(os.environ.get(
    "ARGUS_FACTORMINER_PYTHON",
    REPOSITORY_ROOT / "research" / "_venvs" / "factorminer" / "Scripts" / "python.exe"))

OOS_FRACTION = 0.35
"""The engine's default chronological OOS share (`backtest/engine.run`), which the lab uses."""
IN_SAMPLE_FRACTION = 1.0 - OOS_FRACTION
PLACEBO_SHIFT = 0.5
"""The placebo rotates weights by this share of the sample against the returns."""
SIGNIFICANCE = 0.05

FAMILIES = ("argus", "paper", "alpha101", "alpha101_adapted")
ARMS = ("as_written", "oriented", "placebo")


class RivalRunError(RuntimeError):
    """FactorMiner could not be run as specified, or the gates are not the lab's gates."""


# =================================================================================================
# The panel FactorMiner reads
# =================================================================================================

PANEL_COLUMNS = ("datetime", "asset_id", "open", "high", "low", "close", "volume", "amount")
"""FactorMiner's loader schema (`data/loader.py:5`): ``datetime, asset_id, open, high, low, close,
volume, amount``. ``vwap`` and ``returns`` are derived by FactorMiner's own preprocessor."""


def to_panel(series: Mapping[str, Sequence[Sequence[str]]]) -> list[list[str]]:
    """Fixture rows ``[ts, open, high, low, close, volume, quote_volume]`` as panel rows.

    ``amount`` is the venue's quote volume (USDT traded in the bar), which is what FactorMiner means
    by it: turnover in the quote currency, so ``vwap = amount / volume`` is a real average price.
    Every asset must carry the same timestamps; a ragged panel would put NaNs where the venue had
    bars, and the rank-to-weight step would silently change the cross-section's size.
    """
    stamps: list[str] | None = None
    rows: list[list[str]] = []
    for symbol in sorted(series):
        own = [r[0] for r in series[symbol]]
        if stamps is None:
            stamps = own
        elif own != stamps:
            raise RivalRunError(f"{symbol}'s timestamps differ from the rest of the panel")
        for r in series[symbol]:
            if len(r) < 7:
                raise RivalRunError(f"{symbol}: row {r[0]} has no quote volume")
            rows.append([r[0], symbol, r[1], r[2], r[3], r[4], r[5], r[6]])
    return rows


def cross_sectional_weights(signals: np.ndarray) -> np.ndarray:
    """``(assets, periods)`` scores to per-asset weights in ``[-1, 1]``, each bar on its own.

    Ranks among the finite scores at that bar (ties share their average rank), mapped linearly so
    the highest is +1 and the lowest -1. A bar with fewer than two finite scores is flat, and so is
    a missing score. Uses only the bar's own cross-section, so it adds no look-ahead of its own.
    """
    scores = np.asarray(signals, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("signals must be (assets, periods)")
    weights = np.zeros_like(scores)
    for t in range(scores.shape[1]):
        column = scores[:, t]
        finite = np.flatnonzero(np.isfinite(column))
        n = len(finite)
        if n < 2:
            continue
        values = column[finite]
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        weights[finite, t] = 2.0 * ranks / (n - 1) - 1.0
    return weights


# =================================================================================================
# ARGUS's gates
# =================================================================================================


@dataclass(frozen=True, slots=True)
class GateResult:
    """One factor-instrument pair through the lab's cost, OOS and split-half gates."""

    gross_sharpe: float | None
    net_sharpe: float | None
    oos_sharpe: float | None
    net_return: float | None
    trades: int
    passes_cost: bool
    passes_oos: bool
    split_half: dict[str, Any]
    unscoreable: str = ""

    @property
    def passes_cost_and_oos(self) -> bool:
        return self.passes_cost and self.passes_oos

    @property
    def passes_split_half(self) -> bool:
        return bool(self.split_half["outcome"] == "pass")

    @property
    def passes_all(self) -> bool:
        return self.passes_cost_and_oos and self.passes_split_half

    def as_dict(self) -> dict[str, Any]:
        return {
            "gross_sharpe": self.gross_sharpe, "net_sharpe": self.net_sharpe,
            "oos_sharpe": self.oos_sharpe, "net_return": self.net_return, "trades": self.trades,
            "passes_cost": self.passes_cost, "passes_oos": self.passes_oos,
            "passes_cost_and_oos": self.passes_cost_and_oos,
            "split_half": self.split_half, "passes_split_half": self.passes_split_half,
            "passes_all": self.passes_all, "unscoreable": self.unscoreable,
        }


def gate(bars: Sequence[Bar], values: Sequence[float], *, name: str = "factor",
         cost: CostModel | None = None) -> GateResult:
    """``values[i]`` is the weight held from bar ``i`` to ``i + 1``; one per bar but the last.

    The cost and OOS gates are `factor_lab.Evaluator.score`'s, in its order (OOS is only reached
    by a factor that survived cost), and the split-half gate is `Evaluator.split_half`'s: the same
    readings paired with the return that followed each.
    """
    if len(values) != len(bars) - 1:
        raise ValueError(f"{len(values)} weights for {len(bars)} bars; need one per bar but "
                         f"the last")
    weights = [0.0 if not math.isfinite(v) else float(v) for v in values]

    def signal(_bars: Sequence[Bar], i: int) -> float:
        return weights[i]

    reliability = split_half(weights, next_returns(bars), block=BLOCK).as_dict()
    try:
        result = run(name, "lab", bars, signal, cost=cost or CostModel.bitget_perp(),
                     periods_per_year=HOURLY_PER_YEAR)
    except MetricError as exc:
        return GateResult(None, None, None, None, 0, False, False, reliability,
                          unscoreable=f"unscoreable: {exc}")
    net = result.net.sharpe
    passes_cost = net > 0
    oos = result.out_of_sample.sharpe if result.out_of_sample is not None else None
    passes_oos = passes_cost and oos is not None and oos > 0
    return GateResult(
        gross_sharpe=round(result.gross.sharpe, 3), net_sharpe=round(net, 3),
        oos_sharpe=None if oos is None else round(oos, 3),
        net_return=round(result.net.total_return, 6), trades=result.trades,
        passes_cost=passes_cost, passes_oos=passes_oos, split_half=reliability)


def primitive_values(bars: Sequence[Bar], name: str) -> list[float]:
    """An ARGUS primitive's readings, as `Evaluator.payoff_series` takes them."""
    signal: Callable[[Sequence[Bar], int], float] = PRIMITIVES[name]
    return [float(signal(bars, i)) for i in range(len(bars) - 1)]


def check_parity(argus_rows: Sequence[Mapping[str, Any]], published: Mapping[str, Any]) -> int:
    """Every ARGUS pair must reproduce `factor_split_half.json` exactly, or the gates differ."""
    reference = {(lib["symbol"], f["factor"]): f for lib in published["libraries"]
                 for f in lib["factors"]}
    checked = 0
    for row in argus_rows:
        ref = reference.get((row["symbol"], row["factor"]))
        if ref is None:
            raise RivalRunError(f"{row['symbol']}:{row['factor']} is not in {REPORT_PATH.name}")
        mine = row["as_written"]
        if mine["passes_cost_and_oos"] != ref["passes_cost_and_oos"] or \
                mine["split_half"] != ref["split_half"]:
            raise RivalRunError(
                f"{row['symbol']}:{row['factor']}: this module's gates give "
                f"{mine['passes_cost_and_oos']}/{mine['split_half']} and the lab's give "
                f"{ref['passes_cost_and_oos']}/{ref['split_half']}; the gates are not identical")
        checked += 1
    return checked


# =================================================================================================
# FactorMiner, in its own interpreter
# =================================================================================================

DRIVER = r'''
import json, sys
import numpy as np
import pandas as pd

root, panel_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
in_sample_fraction = float(sys.argv[4])
sys.path.insert(0, root)
from factorminer.application.validation_pipeline import ValidationPipeline
from factorminer.benchmark.catalogs import ALPHA101_CLASSIC, build_alpha101_adapted
from factorminer.core.library_io import PAPER_FACTORS
from factorminer.data.preprocessor import compute_derived_features
from factorminer.data.tensor_builder import TensorConfig, build_tensor, compute_target
from factorminer.evaluation.metrics import compute_factor_stats

df = pd.read_csv(panel_path, parse_dates=["datetime"])
df = compute_target(compute_derived_features(df))
ds = build_tensor(df, TensorConfig())
data = np.asarray(ds.data, dtype=np.float64)
target = np.asarray(ds.target, dtype=np.float64)
pipe = ValidationPipeline(data_tensor=data, returns=target)
M, T = target.shape
half = T // 2
split = int((T - 1) * in_sample_fraction)
full = pipe._build_data_dict()
truncated = {k: v[:, :half] for k, v in full.items()}

entries = [("paper", f["name"], f["formula"], f["category"]) for f in PAPER_FACTORS]
entries += [("alpha101", e.name, e.formula, e.category) for e in ALPHA101_CLASSIC]
entries += [("alpha101_adapted", e.name, e.formula, e.category) for e in build_alpha101_adapted()]
stack = np.full((len(entries), M, T), np.nan)
rows = []
for k, (family, name, formula, category) in enumerate(entries):
    res = pipe.evaluate_candidate(name, formula, fast_screen=False)
    row = {"index": k, "family": family, "name": name, "formula": formula, "category": category,
           "parse_ok": bool(res.parse_ok), "stage_passed": int(res.stage_passed),
           "admitted": bool(res.admitted), "rejection_reason": res.rejection_reason,
           "ic_mean": float(res.ic_mean), "ic_paper_mean": float(res.ic_paper_mean),
           "icir": float(res.icir), "ic_paper_icir": float(res.ic_paper_icir),
           "ic_win_rate": float(res.ic_win_rate), "causal": None, "in_sample_ic_mean": None,
           "has_signal": res.signals is not None}
    if res.signals is not None:
        sig = np.asarray(res.signals, dtype=np.float64)
        stack[k] = sig
        try:
            _, early = pipe.kernel.compute_signals(formula=formula, data_dict=truncated,
                                                  returns_shape=(M, half))
            row["causal"] = bool(np.allclose(early, sig[:, :half], equal_nan=True,
                                             rtol=1e-9, atol=1e-12))
        except Exception as exc:
            row["causal"] = None
            row["causal_error"] = f"{type(exc).__name__}: {exc}"
        row["in_sample_ic_mean"] = float(compute_factor_stats(sig[:, :split],
                                                              target[:, :split])["ic_mean"])
    rows.append(row)
np.save(out_dir + "/signals.npy", stack)
meta = {"asset_ids": [str(a) for a in ds.asset_ids], "periods": int(T),
        "first": str(ds.timestamps[0]), "last": str(ds.timestamps[-1]),
        "feature_names": list(ds.feature_names), "paper_factors": len(PAPER_FACTORS),
        "alpha101_classic": len(ALPHA101_CLASSIC), "in_sample_periods": split,
        "numpy": np.__version__, "pandas": pd.__version__, "python": sys.version.split()[0]}
json.dump({"meta": meta, "factors": rows}, open(out_dir + "/verdicts.json", "w"))
'''
"""The script FactorMiner's interpreter runs. It imports only FactorMiner and its dependencies."""


def factorminer_commit(root: Path = FACTORMINER_ROOT) -> str | None:
    head = root / ".git" / "HEAD"
    if not head.exists():
        return None
    ref = head.read_text("utf-8").strip()
    if ref.startswith("ref: "):
        target = root / ".git" / ref[5:]
        if target.exists():
            return target.read_text("utf-8").strip()
        packed = root / ".git" / "packed-refs"
        if packed.exists():
            for line in packed.read_text("utf-8").splitlines():
                if line.endswith(ref[5:]):
                    return line.split()[0]
        return None
    return ref


def run_factorminer(panel: Sequence[Sequence[str]], *, python: Path = FACTORMINER_PYTHON,
                    root: Path = FACTORMINER_ROOT) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate every catalogue formula with FactorMiner's own engine on ``panel``."""
    if not python.exists():
        raise RivalRunError(f"FactorMiner's interpreter is not at {python}; create it with "
                            f"`uv venv -p 3.12` and install the clone's pinned dependencies")
    if not (root / "factorminer").is_dir():
        raise RivalRunError(f"FactorMiner is not cloned at {root}")
    with tempfile.TemporaryDirectory() as tmp:
        panel_path = Path(tmp) / "panel.csv"
        panel_path.write_text(
            ",".join(PANEL_COLUMNS) + "\n" + "\n".join(",".join(r) for r in panel) + "\n",
            encoding="utf-8", newline="\n")
        done = subprocess.run(
            [str(python), "-c", DRIVER, str(root), str(panel_path), tmp, str(IN_SAMPLE_FRACTION)],
            capture_output=True, text=True, check=False, timeout=3600)
        if done.returncode != 0:
            raise RivalRunError(f"FactorMiner's driver exited {done.returncode}: "
                                f"{done.stderr[-2000:]}")
        verdicts = json.loads((Path(tmp) / "verdicts.json").read_text("utf-8"))
        signals = np.load(Path(tmp) / "signals.npy")
    return verdicts, signals


# =================================================================================================
# The comparison
# =================================================================================================


def binomial_tail(k: int, n: int, p: float) -> float:
    """``P(X >= k)`` for ``X ~ Binomial(n, p)``, exact, in log space."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * math.log1p(-p) for i in range(k, n + 1)]
    top = max(logs)
    return min(1.0, math.exp(top) * sum(math.exp(v - top) for v in logs))


def rate_block(rows: Sequence[Mapping[str, Any]], arm: str, noise: float) -> dict[str, Any]:
    n = len(rows)
    cost_oos = sum(1 for r in rows if r[arm]["passes_cost_and_oos"])
    split = sum(1 for r in rows if r[arm]["passes_split_half"])
    passed = sum(1 for r in rows if r[arm]["passes_all"])
    return {
        "pairs": n,
        "pass_cost": sum(1 for r in rows if r[arm]["passes_cost"]),
        "pass_cost_and_oos": cost_oos,
        "pass_split_half": split,
        "pass_all": passed,
        "split_half_rate": round(split / n, 4) if n else None,
        "pass_all_rate": round(passed / n, 4) if n else None,
        "split_half_p_vs_noise": round(binomial_tail(split, n, noise), 6) if n else None,
        "pass_all_p_vs_noise": round(binomial_tail(passed, n, noise), 6) if n else None,
        "which_pass_all": [f"{r['symbol']}:{r['factor']}" for r in rows if r[arm]["passes_all"]],
    }


def summarise(rows: Sequence[Mapping[str, Any]], noise: float,
              factors: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, Any] = {}
    for family in FAMILIES:
        mine = [r for r in rows if r["family"] == family]
        if not mine:
            continue
        by_family[family] = {arm: rate_block(mine, arm, noise) for arm in ARMS if arm in mine[0]}
    by_category: dict[str, Any] = {}
    for family in FAMILIES[1:]:
        cats = sorted({str(r["category"]) for r in rows if r["family"] == family})
        by_category[family] = {
            cat: {arm: rate_block([r for r in rows if r["family"] == family
                                   and r["category"] == cat], arm, noise)
                  for arm in ("as_written", "oriented")}
            for cat in cats}
    fm = {}
    for family in FAMILIES[1:]:
        own = [f for f in factors if f["family"] == family]
        admitted = {f["name"] for f in own if f["stage_passed"] >= 1}
        pairs = [r for r in rows if r["family"] == family and r["factor"] in admitted]
        fm[family] = {
            "factors": len(own),
            "parsed_with_signal": sum(1 for f in own if f["has_signal"]),
            "non_causal": [f["name"] for f in own if f["causal"] is False],
            "causality_unchecked": [f["name"] for f in own if f["has_signal"]
                                    and f["causal"] is None],
            "passes_own_ic_threshold_alone": sum(1 for f in own if f["ic_paper_mean"] >= 0.04),
            "passes_own_ic_icir_gate": len(admitted),
            "best_paper_icir": round(max((f["ic_paper_icir"] for f in own), default=0.0), 4),
            "its_passing_factors_through_argus_gates": {
                arm: rate_block(pairs, arm, noise) for arm in ("as_written", "oriented")
            } if pairs else None,
        }
    return {"by_family": by_family, "by_category": by_category, "factorminer_own_verdict": fm}


def verdict(summary: Mapping[str, Any], noise: float, *,
            power_80pct_at: str | None = None) -> dict[str, Any]:
    """The plan's rule, applied to the best rival arm (the one most favourable to the rival)."""
    fam = summary["by_family"]
    argus_rate = fam["argus"]["as_written"]["pass_all_rate"]
    above: list[str] = []
    for family in FAMILIES[1:]:
        if family not in fam:
            continue
        for arm in ("as_written", "oriented"):
            block = fam[family][arm]
            if block["pass_all_p_vs_noise"] < SIGNIFICANCE and block["pass_all_rate"] > argus_rate:
                above.append(f"{family}/{arm}")
    argus_above = fam["argus"]["as_written"]["pass_all_p_vs_noise"] < SIGNIFICANCE
    if above:
        return {"state": "LOST", "families_above_noise_and_argus": above,
                "reason": "a FactorMiner library passes every ARGUS gate, split-half included, "
                          "more often than noise and more often than ARGUS's own rules"}
    if not argus_above:
        return {"state": "TIED", "families_above_noise_and_argus": [],
                "reason": (f"no arm, ARGUS's or FactorMiner's, passes every gate above the "
                           f"calibrated noise rate ({noise:.4f}). On twelve instruments over 90 "
                           f"days of hourly bars the split-half gate reaches 80% power only at "
                           f"{power_80pct_at or 'an unmeasured'} directional accuracy, so an edge "
                           f"below that is invisible to every searcher; the comparison cannot "
                           f"separate the libraries, and that is a property of the evidence, not "
                           f"of either searcher")}
    return {"state": "NOT_DECIDED_BY_RULE", "families_above_noise_and_argus": [],
            "reason": "ARGUS's rules pass above noise and no rival family does; the plan's rule "
                      "names only LOST and TIED"}


def compare(series: Mapping[str, Sequence[Sequence[str]]], verdicts: Mapping[str, Any],
            signals: np.ndarray, *, symbols: Sequence[str] = SYMBOLS) -> list[dict[str, Any]]:
    """Every factor-instrument pair, ARGUS's eight and FactorMiner's catalogues, through the
    gates."""
    assets = list(verdicts["meta"]["asset_ids"])
    if assets != sorted(series):
        raise RivalRunError(f"FactorMiner's asset order {assets} is not the panel's")
    bars_by = {s: bars_from_rows(series[s]) for s in symbols}
    periods = int(verdicts["meta"]["periods"])
    shift = int(periods * PLACEBO_SHIFT)
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        bars = bars_by[symbol]
        for name in PRIMITIVES:
            rows.append({"family": "argus", "factor": name, "category": "session/time-series",
                         "symbol": symbol,
                         "as_written": gate(bars, primitive_values(bars, name)).as_dict()})
    for f in verdicts["factors"]:
        if not f["has_signal"] or f["causal"] is False:
            continue
        weights = cross_sectional_weights(signals[f["index"]])
        sign = -1.0 if (f["in_sample_ic_mean"] or 0.0) < 0 else 1.0
        placebo = np.roll(weights, shift, axis=1)
        for symbol in symbols:
            m = assets.index(symbol)
            bars = bars_by[symbol]
            if len(bars) != periods:
                raise RivalRunError(f"{symbol}: {len(bars)} bars against {periods} periods")
            w = weights[m, :-1]
            rows.append({
                "family": f["family"], "factor": f["name"], "category": f["category"],
                "symbol": symbol, "orientation": sign,
                "factorminer_passes_ic_icir": f["stage_passed"] >= 1,
                "as_written": gate(bars, w.tolist()).as_dict(),
                "oriented": gate(bars, (sign * w).tolist()).as_dict(),
                "placebo": gate(bars, placebo[m, :-1].tolist()).as_dict(),
            })
    return rows


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - runs FactorMiner
    from argus.truth.artefact import write

    series, digest = load_fixture()
    published = json.loads(REPORT_PATH.read_text("utf-8"))
    if published.get("bars", {}).get("sha256") != digest:
        raise RivalRunError(f"{REPORT_PATH.name} was not computed from {FIXTURE.name}; re-run "
                            f"`python -m argus.eval.factor_split_half` first")
    noise_block = published["planted"]["noise"]
    noise = noise_block["passed"] / noise_block["of"]
    verdicts, signals = run_factorminer(to_panel(series))
    print(f"FactorMiner evaluated {len(verdicts['factors'])} formulas", file=sys.stderr)
    rows = compare(series, verdicts, signals)
    parity = check_parity([r for r in rows if r["family"] == "argus"], published)
    summary = summarise(rows, noise, verdicts["factors"])
    decision = verdict(summary, noise,
                       power_80pct_at=published["planted"]["smallest_accuracy_detected_80pct"])
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "qwen_calls": 0,
        "bars": {"fixture": f"data/{FIXTURE.name}", "sha256": digest},
        "rival": {"name": "FactorMiner", "licence": "MIT",
                  "clone": "research/repos-t2/minihellboy~factorminer",
                  "commit": factorminer_commit(), "engine_meta": verdicts["meta"],
                  "own_gate": "Stage 1 of ValidationPipeline.evaluate_candidate at defaults: "
                              "|mean RankIC| >= 0.04 and |ICIR| >= 0.5 against the next bar's "
                              "open-to-close return"},
        "gates": {"cost": "net Sharpe > 0 at CostModel.bitget_perp() (12 bps round trip)",
                  "oos": f"chronological last {OOS_FRACTION:.0%} Sharpe > 0, after cost",
                  "split_half": f"factor_lab.split_half, block {BLOCK}",
                  "parity_with_factor_split_half": f"{parity} ARGUS pairs identical"},
        "conversion": "each bar, the twelve scores ranked onto [-1, 1] (ties averaged, missing "
                      "flat): every instrument's weight is its leg of the factor's long-short book",
        "arms": {"as_written": "the formula's published direction",
                 "oriented": f"flipped where FactorMiner's RankIC on the first "
                             f"{IN_SAMPLE_FRACTION:.0%} of bars is negative",
                 "placebo": f"as_written rotated {PLACEBO_SHIFT:.0%} of the sample against the "
                            f"returns: the all-gates null"},
        "noise": {"source": f"{REPORT_PATH.name} planted.noise", "passed": noise_block["passed"],
                  "of": noise_block["of"], "rate": round(noise, 4),
                  "power_80pct_at_accuracy": published["planted"][
                      "smallest_accuracy_detected_80pct"]},
        "fairness": "FactorMiner's factors are cross-sectional scores built for wide A-share "
                    "panels; twelve names is a thin cross-section for them, amount/vwap and "
                    "'overnight' mean little on round-the-clock perpetual tokens, and each "
                    "per-instrument leg carries market exposure the full book would net out. "
                    "Per-category results are published for that reason.",
        "summary": summary,
        "verdict": decision,
        "factorminer_factors": verdicts["factors"],
        "rows": rows,
    }
    write(OUT, report)
    for family, arms in summary["by_family"].items():
        for arm, block in arms.items():
            print(f"{family:>17} {arm:>10}: {block['pairs']} pairs, cost+oos "
                  f"{block['pass_cost_and_oos']}, split-half {block['pass_split_half']}, all "
                  f"{block['pass_all']} ({block['pass_all_rate']}) "
                  f"p={block['pass_all_p_vs_noise']}")
    print("verdict:", decision["state"], "-", decision["reason"])
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))


__all__ = ["DRIVER", "OUT", "PANEL_COLUMNS", "GateResult", "RivalRunError", "binomial_tail",
           "check_parity", "compare", "cross_sectional_weights", "gate", "primitive_values",
           "rate_block", "run_factorminer", "summarise", "to_panel", "verdict"]
