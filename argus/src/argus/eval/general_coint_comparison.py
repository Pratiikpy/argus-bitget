"""Pairwise cointegration screening with false-discovery control: ARGUS vs. the general-purpose
econometrics stack, on the same series.

**The general function.** Given a universe of *N* non-stationary series, return the pairs whose
linear combination is stationary, while controlling the share of returned pairs that are not.
Two layers: a residual-based cointegration test per pair, then a multiple-testing procedure across
the *N(N-1)/2* tests — tests that are *dependent*, because every series appears in *N-1* of them.
Nothing in that is specific to trading. The same screen runs on macro panels, commodity curves and
climate series, and the strongest implementations of each layer live in general-purpose libraries,
not in pairs-trading repositories. ``eval/cointegration_comparison.py`` compared ARGUS against
trading code (Lean, FinceptTerminal) and against one statsmodels function (``adfuller``); this
module asks the harder question — whether the general stack, used the way an econometrician would,
does the whole job better.

**Rivals, chosen by evidence rather than stars.**

* ``statsmodels`` 0.15 (BSD-3; 11.6k stars, pushed 2026-09-25) — ``tsa.stattools.coint``
  (Engle-Granger with MacKinnon 2010 surfaces) and ``stats.multitest.multipletests``: ``fdr_bh``,
  ``fdr_by`` (Benjamini-Yekutieli, valid under *arbitrary* dependence), ``fdr_tsbh`` (the
  two-stage adaptive procedure of Benjamini, Krieger & Yekutieli 2006, which estimates the null
  share and so buys power), and ``holm``.
* ``arch`` 8.0 (University of Illinois/NCSA licence — permissive, attribution required; Kevin
  Sheppard; 1.6k stars, pushed 2026-09-24; source read at
  ``research/repos-themed/bashtage~arch``, commit 704bb70e). Its ``engle_granger``
  (``arch/unitroot/_engle_granger.py:25-105``) is an independent implementation: BIC lag
  selection by default and its **own re-simulated** critical-value and p-value surfaces
  (``arch/unitroot/critical_values/engle_granger.py`` — for one regressor with a constant,
  ``SMALL_PARAMETERS[('c', 1)] = [2.8505, 1.4605, 0.0343]`` against MacKinnon's
  ``[2.9200, 1.5012, 0.0398]``). Its ``phillips_ouliaris`` (``_phillips_ouliaris.py:134-310``)
  implements Phillips & Ouliaris (1990): ``Za``/``Zt`` correct the residual AR(1) with a
  Bartlett-kernel long-run variance instead of searching over lag orders, and ``Pz`` is
  invariant to which series is placed on the left — Engle-Granger is not.
* ``scipy`` 1.17 (BSD-3) — ``stats.false_discovery_control`` (BH and BY), the most widely
  installed FDR implementation in Python.

Considered and not run, with the reason: R's ``urca``/``tseries``/``egcm`` (no R on this machine);
Johansen's trace test (``statsmodels.tsa.vector_ar.vecm.coint_johansen`` returns critical values at
three levels and no p-value, so it cannot feed any FDR procedure without a p-value surface this
module would have to invent); ``puolival/multipy`` (112 stars, last pushed 2024-08, and adds
nothing for BH/BY that statsmodels lacks).

**What is compared, on what.**

1. *Monte Carlo universes with known truth*, shaped exactly like the production scan's input: 12
   series, 1,440 hourly bars (60 days), the first 60% for selection (``scan()``'s default
   ``train_fraction``). Three scenarios — ``null`` (12 independent random walks: every discovery
   is false), ``planted`` (three clusters sharing a common stochastic trend with idiosyncratic
   AR(1) spreads at phi 0.96 / 0.975 / 0.985, i.e. half-lives of 17 / 27 / 46 bars, plus four
   unrelated walks: 7 true pairs of 66), and ``broken`` (the same, but one member of two clusters
   decouples at the train/test boundary, so 4 of the 7 in-sample relationships are gone out of
   sample — truth is judged on the held-out window, which is the window a trader lives in).
2. *The real rToken universe*: 60 days of Bitget hourly closes for the 12 production symbols,
   frozen to ``data/general_coint_rtoken_snapshot.json`` so the run is reproducible offline.

Every system receives identical series. ARGUS runs through its real production ``scan()``;
nothing is re-derived.

**Licence posture.** statsmodels, arch and scipy are imported, never vendored. Anything taken from
arch's source into ARGUS must carry its copyright notice (NCSA clause 1).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from argus.backtest.validation import benjamini_hochberg, bonferroni
from argus.research.cointegration import ScanReport, scan

TRAIN_FRACTION = 0.6
"""``scan()``'s own default. The selection p-values come from this slice for every system."""

Q = 0.05
"""Target false-discovery rate, ``scan()``'s own default ``q``."""

N_BARS = 1440
"""Sixty days of hourly bars — the production scan's ``--days 60 --interval 1H``."""

CLUSTERS: tuple[tuple[str, int, float], ...] = (("A", 3, 0.96), ("B", 3, 0.975), ("C", 2, 0.985))
"""(label, members, idiosyncratic AR(1) coefficient). Half-lives ln2/(1-phi): 17, 27, 46 bars.

Chosen so that, on an 864-bar selection slice after a 66-test correction, detection is near-certain,
middling and hard respectively. A first pilot at phi 0.98/0.99/0.995 left every method at almost
zero power after correction — a universe where nobody finds anything cannot rank anybody."""

N_SINGLETONS = 4
"""Unrelated random walks in the planted universes, so 12 series and 66 pairs in total."""

DECOUPLING_CLUSTERS = ("A", "B")
"""In the ``broken`` scenario the last member of these clusters turns into a random walk at the
train/test boundary: its in-sample relationships are real and then stop existing."""

TRAIN_TESTS = (
    "argus_eg", "sm_coint", "arch_eg_bic", "arch_eg_aic", "arch_eg_tstat", "arch_po_zt",
    "arch_po_za", "arch_po_pz",
)
"""Per-pair tests run on the selection slice. ``argus_eg`` is read from ``scan()`` itself.
``arch_eg_tstat`` is arch's general-to-specific lag selection (``unitroot.py:185-189``), the
choice with the best size under a negative moving-average root (Ng & Perron 1995)."""

FULL_TESTS = ("arch_eg_bic@full", "arch_po_zt@full", "arch_po_pz@full")
"""The general-purpose *natural* protocol: test the whole sample, no held-out window.
``statsmodels.coint`` is not repeated here: on the selection slice it is identical to ARGUS's own
test to 1e-11, and at 0.56s a pair it was a third of the rival budget for a number that
``arch_eg_bic@full`` already answers."""

OOS_FILTERED = ("argus_eg", "argus_eg_both", "argus_eg+pre05", "arch_eg_bic_both",
                "arch_eg_tstat", "arch_po_pz")
"""Tests whose corrected discoveries are also re-scored after ARGUS's own out-of-sample gate
(suffix ``+oos`` on the correction), so the gate's contribution can be separated from the test's."""

CORRECTIONS = (
    "argus_bh", "argus_bonf", "sm_bh", "sm_by", "sm_tsbh", "sm_holm", "scipy_bh", "scipy_by",
)

PRODUCTION = "argus_eg|argus_bh"
"""What ``ScanReport.survivors_fdr`` returns — checked to be identical in every universe."""

CONFIRMED = "argus_confirmed"
"""``survivors_fdr`` further filtered by ARGUS's own out-of-sample gate: the frozen-ratio spread
is stationary at 5% on the held-out 40% and has a finite half-life (the statistical half of
``PairResult.tradeable``; the cost half is not a statistical property and is left out)."""

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_PATH = ROOT / "data" / "general_coint_rtoken_snapshot.json"
ARTEFACT_PATH = ROOT / "data" / "general_coint_comparison.json"

Pair = tuple[str, str]


class GeneralCointComparisonError(RuntimeError):
    """The comparison could not run as specified."""


# =================================================================================================
# Universes.
# =================================================================================================


@dataclass(frozen=True)
class Universe:
    """One set of series and, where it is known, which pairs are genuinely cointegrated."""

    scenario: str
    seed: int
    series: dict[str, list[float]]
    truth_train: frozenset[Pair] = frozenset()
    """Pairs cointegrated over the selection slice."""
    truth_forward: frozenset[Pair] = frozenset()
    """Pairs cointegrated over the held-out slice — the truth every scenario is scored on."""
    strength: dict[Pair, str] = field(default_factory=dict)


def _walk(rng: random.Random, n: int, start: float) -> list[float]:
    out = [start]
    for _ in range(n - 1):
        out.append(out[-1] + rng.gauss(0.0, 1.0))
    return out


def null_universe(seed: int, *, n_series: int = 12, n_bars: int = N_BARS) -> Universe:
    """Independent random walks: no pair is cointegrated, so every discovery is false."""
    rng = random.Random(seed)
    series = {f"N{i:02d}": _walk(rng, n_bars, rng.uniform(50.0, 300.0)) for i in range(n_series)}
    return Universe("null", seed, series)


def planted_universe(
    seed: int, *, n_bars: int = N_BARS, broken: bool = False,
    train_fraction: float = TRAIN_FRACTION,
) -> Universe:
    """Clusters sharing one stochastic trend each, plus unrelated walks.

    Member ``j`` of a cluster is ``level + loading * trend + u``, where ``u`` is AR(1) at the
    cluster's ``phi``; any two members therefore have a stationary spread. With ``broken`` the last
    member of each cluster in :data:`DECOUPLING_CLUSTERS` switches ``u`` to a random walk at the
    split, so its pairs are cointegrated in-sample and not after.
    """
    rng = random.Random(seed)
    split = int(n_bars * train_fraction)
    series: dict[str, list[float]] = {}
    truth: set[Pair] = set()
    forward: set[Pair] = set()
    strength: dict[Pair, str] = {}
    for label, members, phi in CLUSTERS:
        trend = _walk(rng, n_bars, 0.0)
        names = [f"{label}{j}" for j in range(members)]
        decoupled = {names[-1]} if broken and label in DECOUPLING_CLUSTERS else set()
        for name in names:
            loading = rng.uniform(0.5, 2.0)
            level = rng.uniform(50.0, 300.0)
            u = [0.0]
            for t in range(1, n_bars):
                rho = 1.0 if (name in decoupled and t >= split) else phi
                u.append(rho * u[-1] + rng.gauss(0.0, 1.0))
            series[name] = [level + loading * trend[t] + u[t] for t in range(n_bars)]
        for i in range(members):
            for k in range(i + 1, members):
                pair = (names[i], names[k])
                truth.add(pair)
                strength[pair] = f"phi={phi}"
                if not decoupled.intersection(pair):
                    forward.add(pair)
    for s in range(N_SINGLETONS):
        series[f"Z{s}"] = _walk(rng, n_bars, rng.uniform(50.0, 300.0))
    return Universe(
        "broken" if broken else "planted", seed, series, frozenset(truth), frozenset(forward),
        strength,
    )


def build_universe(scenario: str, seed: int) -> Universe:
    if scenario == "null":
        return null_universe(seed)
    if scenario == "planted":
        return planted_universe(seed)
    if scenario == "broken":
        return planted_universe(seed, broken=True)
    raise GeneralCointComparisonError(f"unknown scenario {scenario!r}")


# =================================================================================================
# The rivals, called exactly as their own documentation shows.
# =================================================================================================


_PO_KIND: dict[str, Literal["Za", "Zt", "Pu", "Pz"]] = {
    "arch_po_zt": "Zt", "arch_po_za": "Za", "arch_po_pz": "Pz",
}


def rival_pvalues(y: Sequence[float], x: Sequence[float], tests: Sequence[str]) -> dict[str, float]:
    """p-values for one pair (``y`` regressed on ``x``) from each named general-purpose test.

    A test that raises is recorded as ``nan`` and treated downstream as a non-rejection, and the
    failure is counted — a rival is never silently credited or penalised for an exception.
    """
    import numpy as np
    from arch.unitroot.cointegration import engle_granger, phillips_ouliaris
    from statsmodels.tsa.stattools import coint

    ya = np.asarray(y, dtype=float)
    xa = np.asarray(x, dtype=float)
    out: dict[str, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in tests:
            base = name.split("@")[0]
            try:
                if base == "sm_coint":
                    out[name] = float(coint(ya, xa)[1])
                elif base == "arch_eg_bic":
                    out[name] = float(engle_granger(ya, xa, trend="c").pvalue)
                elif base == "arch_eg_aic":
                    out[name] = float(engle_granger(ya, xa, trend="c", method="aic").pvalue)
                elif base == "arch_eg_tstat":
                    out[name] = float(engle_granger(ya, xa, trend="c", method="t-stat").pvalue)
                elif base in _PO_KIND:
                    out[name] = float(
                        phillips_ouliaris(ya, xa, trend="c", test_type=_PO_KIND[base]).pvalue
                    )
                else:
                    raise GeneralCointComparisonError(f"unknown test {name!r}")
            except GeneralCointComparisonError:
                raise
            except Exception:
                out[name] = math.nan
    return out


def reject(pvalues: Sequence[float], correction: str, *, q: float = Q) -> list[bool]:
    """Apply one multiple-testing procedure. ``nan`` p-values become 1.0 (never rejected)."""
    clean = [1.0 if not math.isfinite(p) else min(max(p, 0.0), 1.0) for p in pvalues]
    if not clean:
        return []
    if correction == "argus_bh":
        return benjamini_hochberg(clean, fdr=q)
    if correction == "argus_bonf":
        return bonferroni(clean, alpha=q)
    if correction.startswith("sm_"):
        from statsmodels.stats.multitest import multipletests

        method = {"sm_bh": "fdr_bh", "sm_by": "fdr_by", "sm_tsbh": "fdr_tsbh",
                  "sm_holm": "holm"}[correction]
        return [bool(r) for r in multipletests(clean, alpha=q, method=method)[0]]
    if correction.startswith("scipy_"):
        from scipy.stats import false_discovery_control

        if correction == "scipy_bh":
            adjusted = false_discovery_control(clean, method="bh")
        elif correction == "scipy_by":
            adjusted = false_discovery_control(clean, method="by")
        else:
            raise GeneralCointComparisonError(f"unknown correction {correction!r}")
        return [bool(a <= q) for a in adjusted]
    raise GeneralCointComparisonError(f"unknown correction {correction!r}")


# =================================================================================================
# One universe, every system.
# =================================================================================================


def _key(pair: Pair) -> str:
    return f"{pair[0]}/{pair[1]}"


PRESCREENED_TESTS = ("argus_eg", "arch_eg_bic", "arch_eg_tstat", "arch_po_zt", "arch_po_pz")
"""Tests re-scored after the per-series unit-root pre-test, at 5% and at 10%."""


def _unit_root_pvalue(values: Sequence[float], train_fraction: float) -> float:
    """ADF p-value (constant, AIC lags) of one series' selection slice; 1.0 when untestable."""
    from argus.research.cointegration import CointegrationError, adf

    head = [float(v) for v in values][:int(len(values) * train_fraction)]
    try:
        return adf(head, regression="c").pvalue
    except CointegrationError:
        return 1.0


def evaluate_series(
    series: Mapping[str, Sequence[float]], *, train_fraction: float = TRAIN_FRACTION,
    q: float = Q, train_tests: Sequence[str] = TRAIN_TESTS,
    full_tests: Sequence[str] = FULL_TESTS, corrections: Sequence[str] = CORRECTIONS,
) -> dict[str, Any]:
    """Run ARGUS's real ``scan()`` and every rival pipeline on identical series.

    Returns the discoveries of each pipeline (``"<test>|<correction>"``), the identity checks
    between ARGUS and the libraries it claims to reproduce, per-system wall time, and rival
    failures.
    """
    if len({len(v) for v in series.values()}) > 1:
        raise GeneralCointComparisonError(
            "every series must cover the same bars; align on timestamp first (load_snapshot does)"
        )
    head = int(len(next(iter(series.values()))) * train_fraction)  # pair_test's own split
    started = time.perf_counter()
    report: ScanReport = scan(series, q=q, train_fraction=train_fraction)
    argus_seconds = time.perf_counter() - started

    pairs: list[Pair] = [(p.left, p.right) for p in report.pairs]
    pvalues: dict[str, list[float]] = {"argus_eg": [p.in_sample.pvalue for p in report.pairs]}
    rival_tests = [t for t in train_tests if t != "argus_eg"]
    rival_seconds = dict.fromkeys([*rival_tests, *full_tests], 0.0)
    per_pair: list[dict[str, float]] = []
    for left, right in pairs:
        y, x = list(series[left]), list(series[right])
        row: dict[str, float] = {}
        for name in rival_tests:
            t0 = time.perf_counter()
            row.update(rival_pvalues(y[:head], x[:head], [name]))
            rival_seconds[name] += time.perf_counter() - t0
        for name in full_tests:
            t0 = time.perf_counter()
            row.update(rival_pvalues(y, x, [name]))
            rival_seconds[name] += time.perf_counter() - t0
        per_pair.append(row)
    for name in [*rival_tests, *full_tests]:
        pvalues[name] = [row[name] for row in per_pair]

    # Engle-Granger is not symmetric: `scan()` always puts the alphabetically-first series on the
    # left. A series that merely looks stationary in-sample makes every regression with it on the
    # LEFT reject, and none with it on the right. The symmetric variant requires both directions
    # to reject (the larger p-value) — with ARGUS's own engle_granger, and with arch's.
    from argus.research.cointegration import CointegrationError
    from argus.research.cointegration import engle_granger as argus_engle_granger

    t0 = time.perf_counter()
    argus_reverse: list[float] = []
    for lft, r in pairs:
        try:
            argus_reverse.append(
                argus_engle_granger(list(series[r])[:head], list(series[lft])[:head]).pvalue
            )
        except CointegrationError:
            argus_reverse.append(math.nan)
    argus_both_seconds = time.perf_counter() - t0
    pvalues["argus_eg_both"] = [
        max(f, b) if math.isfinite(b) else f
        for f, b in zip(pvalues["argus_eg"], argus_reverse, strict=True)
    ]
    if "arch_eg_bic" in pvalues:
        t0 = time.perf_counter()
        reverse = [
            rival_pvalues(list(series[r])[:head], list(series[lft])[:head], ["arch_eg_bic"])[
                "arch_eg_bic"
            ]
            for lft, r in pairs
        ]
        rival_seconds["arch_eg_bic_both"] = time.perf_counter() - t0
        pvalues["arch_eg_bic_both"] = [
            max(f, b) if math.isfinite(f) and math.isfinite(b) else math.nan
            for f, b in zip(pvalues["arch_eg_bic"], reverse, strict=True)
        ]

    # Engle and Granger's own step zero, which none of the libraries performs: a cointegration
    # test presumes both series are I(1). A series that already looks stationary on its own makes
    # every regression with it on the left "cointegrated", so one chance near-stationary walk
    # produces up to N-1 correlated false discoveries at once. Pre-tested with ARGUS's real adf()
    # (which reproduces statsmodels' adfuller) on each series' selection slice; an excluded pair
    # keeps its place in the family with p = 1, so the correction still counts it as examined.
    t0 = time.perf_counter()
    unit_root_p = {
        name: _unit_root_pvalue(series[name], train_fraction)
        for name in sorted({n for pair in pairs for n in pair})
    }
    prescreen_seconds = time.perf_counter() - t0
    for level, tag in ((0.05, "pre05"), (0.10, "pre10")):
        for test in PRESCREENED_TESTS:
            if test not in pvalues:
                continue
            pvalues[f"{test}+{tag}"] = [
                1.0 if min(unit_root_p[lft], unit_root_p[r]) < level else p
                for (lft, r), p in zip(pairs, pvalues[test], strict=True)
            ]

    discoveries: dict[str, list[str]] = {}
    for test in pvalues:
        for corr in corrections:
            flags = reject(pvalues[test], corr, q=q)
            discoveries[f"{test}|{corr}"] = [
                _key(pair) for pair, hit in zip(pairs, flags, strict=True) if hit
            ]
    survivors = [_key((p.left, p.right)) for p in report.survivors_fdr]
    oos_ok = {
        _key((p.left, p.right)) for p in report.pairs
        if p.out_of_sample is not None and p.out_of_sample.stationary_at_5pct
        and p.half_life_bars is not None
    }
    discoveries[CONFIRMED] = [k for k in survivors if k in oos_ok]
    for test in OOS_FILTERED:
        for corr in ("argus_bh", "sm_by"):
            if f"{test}|{corr}" in discoveries:
                discoveries[f"{test}|{corr}+oos"] = [
                    k for k in discoveries[f"{test}|{corr}"] if k in oos_ok
                ]

    diffs = [
        abs(a - b) for a, b in zip(pvalues["argus_eg"], pvalues.get("sm_coint", []), strict=False)
        if math.isfinite(b)
    ]
    bh_sets = {
        c: tuple(discoveries.get(f"argus_eg|{c}", ()))
        for c in ("argus_bh", "sm_bh", "scipy_bh") if c in corrections
    }
    return {
        "pairs": [_key(p) for p in pairs],
        "pvalues": {k: [round(v, 8) if math.isfinite(v) else None for v in vs]
                    for k, vs in pvalues.items()},
        "discoveries": discoveries,
        "checks": {
            "scan_survivors_fdr_equals_production": sorted(survivors)
            == sorted(discoveries.get(PRODUCTION, [])),
            "argus_vs_statsmodels_coint_max_abs_p_diff": max(diffs) if diffs else None,
            "bh_implementations_agree": len(set(bh_sets.values())) <= 1,
        },
        "unit_root_pvalues": {k: round(v, 6) for k, v in unit_root_p.items()},
        "rival_failures": {
            name: sum(1 for v in pvalues[name] if not math.isfinite(v))
            for name in [*rival_tests, *full_tests]
        },
        "seconds": {"argus_scan": argus_seconds, "unit_root_prescreen": prescreen_seconds,
                    "argus_eg_reverse": argus_both_seconds, **rival_seconds},
    }


def evaluate_seeded(job: tuple[str, int]) -> dict[str, Any]:
    """Worker entry point: build the universe inside the worker so nothing large is pickled."""
    scenario, seed = job
    universe = build_universe(scenario, seed)
    result = evaluate_series(universe.series)
    result.update({
        "scenario": scenario,
        "seed": seed,
        "truth_train": sorted(_key(p) for p in universe.truth_train),
        "truth_forward": sorted(_key(p) for p in universe.truth_forward),
        "strength": {_key(p): s for p, s in universe.strength.items()},
    })
    result.pop("pvalues")  # the discoveries carry the result; 66x11 p-values per rep do not
    return result


# =================================================================================================
# Scoring against known truth.
# =================================================================================================


@dataclass(frozen=True)
class PipelineScore:
    pipeline: str
    reps: int
    fdr: float
    fdr_se: float
    fwer: float
    power: float | None
    power_se: float | None
    power_by_strength: dict[str, float]
    mean_discoveries: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "reps": self.reps,
            "fdr": round(self.fdr, 4),
            "fdr_se": round(self.fdr_se, 4),
            "fwer": round(self.fwer, 4),
            "power": None if self.power is None else round(self.power, 4),
            "power_se": None if self.power_se is None else round(self.power_se, 4),
            "power_by_strength": {k: round(v, 4) for k, v in self.power_by_strength.items()},
            "mean_discoveries": round(self.mean_discoveries, 3),
        }


def _se(values: Sequence[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def score_pipeline(results: Sequence[Mapping[str, Any]], pipeline: str) -> PipelineScore:
    """False-discovery proportion, family-wise error and power, scored on the held-out truth."""
    fdp: list[float] = []
    any_false: list[float] = []
    power: list[float] = []
    found: dict[str, list[float]] = {}
    sizes: list[float] = []
    for r in results:
        found_pairs = set(r["discoveries"][pipeline])
        truth = set(r["truth_forward"])
        false = len(found_pairs - truth)
        fdp.append(false / len(found_pairs) if found_pairs else 0.0)
        any_false.append(1.0 if false else 0.0)
        sizes.append(float(len(found_pairs)))
        if truth:
            power.append(len(found_pairs & truth) / len(truth))
            for pair in truth:
                found.setdefault(r["strength"].get(pair, "?"), []).append(
                    1.0 if pair in found_pairs else 0.0
                )
    return PipelineScore(
        pipeline=pipeline, reps=len(results),
        fdr=statistics.fmean(fdp) if fdp else 0.0, fdr_se=_se(fdp),
        fwer=statistics.fmean(any_false) if any_false else 0.0,
        power=statistics.fmean(power) if power else None,
        power_se=_se(power) if power else None,
        power_by_strength={k: statistics.fmean(v) for k, v in sorted(found.items())},
        mean_discoveries=statistics.fmean(sizes) if sizes else 0.0,
    )


def paired_power_gain(
    results: Sequence[Mapping[str, Any]], pipeline: str, against: str = PRODUCTION,
) -> tuple[float, float]:
    """Mean and standard error of ``power(pipeline) - power(against)`` on the SAME universes."""
    gains: list[float] = []
    for r in results:
        truth = set(r["truth_forward"])
        if not truth:
            continue
        mine = len(set(r["discoveries"][pipeline]) & truth) / len(truth)
        theirs = len(set(r["discoveries"][against]) & truth) / len(truth)
        gains.append(mine - theirs)
    if not gains:
        return 0.0, 0.0
    return statistics.fmean(gains), _se(gains)


def controls_fdr(score: PipelineScore, *, q: float = Q) -> bool:
    """Empirical FDR within two Monte Carlo standard errors of the target."""
    return score.fdr <= q + 2 * max(score.fdr_se, 1e-9)


def is_general(pipeline: str) -> bool:
    """True when a pipeline uses any general-purpose component — a rival test or a rival
    correction. ARGUS-native pipelines use only ARGUS's own test (with or without its own
    unit-root pre-screen) and ARGUS's own corrections."""
    if pipeline == CONFIRMED:
        return False
    test, _, correction = pipeline.partition("|")
    return not test.startswith("argus_eg") or not correction.startswith("argus_")


def decide(
    by_scenario: Mapping[str, Mapping[str, PipelineScore]],
    planted: Sequence[Mapping[str, Any]], *, q: float = Q,
) -> dict[str, Any]:
    """Who wins the function, computed from the scores rather than written by hand.

    Eligibility: a pipeline must hold the false-discovery rate — empirical FDR within two Monte
    Carlo standard errors of ``q`` — in BOTH the null and the planted universes. A power advantage
    bought with false discoveries is not an advantage.

    * ARGUS's production pipeline fails eligibility and a general pipeline passes: the rival wins.
    * Both eligible: the best general pipeline's paired power gain over production on the planted
      universes decides — above two standard errors the rival wins, below minus two ARGUS wins,
      otherwise a tie.
    * Neither eligible: a tie in failure, reported as such.

    Full-sample ("@full") pipelines are scored and reported but do not enter this verdict: they
    select on 1,440 bars where ARGUS selects on 864 and holds 576 back, so they answer a different
    question. :func:`protocol_finding` reports that comparison separately.
    """
    null, plant = by_scenario["null"], by_scenario["planted"]

    def eligible(p: str) -> bool:
        return controls_fdr(null[p], q=q) and controls_fdr(plant[p], q=q)

    general = [p for p in plant if is_general(p) and "@full" not in p and eligible(p)]
    ranked: list[tuple[float, float, str]] = []
    for p in general:
        gain, se = paired_power_gain(planted, p)
        ranked.append((gain, se, p))
    ranked.sort(key=lambda t: (-t[0], plant[t[2]].fdr, t[2]))
    best = ranked[0] if ranked else None
    production_ok = eligible(PRODUCTION)
    if not production_ok:
        verdict = "rival_wins" if best is not None else "tie"
    elif best is None:
        verdict = "argus_wins"
    elif best[0] > 2 * best[1] and best[0] > 0:
        verdict = "rival_wins"
    elif best[0] < -2 * best[1]:
        verdict = "argus_wins"
    else:
        verdict = "tie"

    native = [p for p in plant if not is_general(p) and p not in (PRODUCTION, CONFIRMED)
              and "@full" not in p and eligible(p)]
    native_ranked = sorted(
        ((*paired_power_gain(planted, p), p) for p in native), key=lambda t: (-t[0], t[2])
    )

    def row(t: tuple[float, float, str]) -> dict[str, Any]:
        return {"pipeline": t[2], "paired_power_gain_vs_production": round(t[0], 4),
                "se": round(t[1], 4), "planted_power": plant[t[2]].power,
                "planted_fdr": round(plant[t[2]].fdr, 4), "null_fwer": round(null[t[2]].fwer, 4)}

    return {
        "verdict": verdict,
        "rule": "eligible = FDR <= q + 2 MC s.e. in null AND planted; then paired power gain vs "
                "production on planted, +/- 2 s.e.; same selection slice only",
        "production_controls_fdr": production_ok,
        "production": {"null_fdr": round(null[PRODUCTION].fdr, 4),
                       "null_fdr_se": round(null[PRODUCTION].fdr_se, 4),
                       "planted_fdr": round(plant[PRODUCTION].fdr, 4),
                       "planted_power": plant[PRODUCTION].power},
        "eligible_general_pipelines": len(general),
        "best_general_pipeline": None if best is None else row(best),
        "general_pipelines_beating_production": [
            row(t) for t in ranked if t[0] > 2 * t[1] and t[0] > 0
        ],
        "top_general": [row(t) for t in ranked[:10]],
        "best_argus_native_candidate": row(native_ranked[0]) if native_ranked else None,
    }


def protocol_finding(
    by_scenario: Mapping[str, Mapping[str, PipelineScore]], *, q: float = Q,
) -> dict[str, Any]:
    """The general stack's natural protocol (test the whole sample) against ARGUS's (select on
    60%, confirm out of sample), scored on held-out truth in the planted and broken universes."""
    out: dict[str, Any] = {}
    for scenario in ("planted", "broken"):
        if scenario not in by_scenario:
            continue
        scores = by_scenario[scenario]
        rows = [p for p in scores if "@full" in p and p.endswith(("|sm_bh", "|sm_by"))]
        rows += [PRODUCTION, CONFIRMED]
        out[scenario] = [
            {"pipeline": p, "fdr": round(scores[p].fdr, 4), "fdr_se": round(scores[p].fdr_se, 4),
             "fwer": round(scores[p].fwer, 4), "power": scores[p].power,
             "controls_fdr": controls_fdr(scores[p], q=q)}
            for p in rows if p in scores
        ]
    return out


# =================================================================================================
# The real rToken universe.
# =================================================================================================


def load_snapshot(path: Path = SNAPSHOT_PATH) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """The frozen real closes, aligned on timestamp (an inner join: no fill, no guessing)."""
    blob = json.loads(path.read_text(encoding="utf-8"))
    stamps: list[str] = blob["timestamps"]
    closes: dict[str, list[float]] = {s: [float(v) for v in vs] for s, vs in blob["closes"].items()}
    for symbol, values in closes.items():
        if len(values) != len(stamps):
            raise GeneralCointComparisonError(f"{symbol}: {len(values)} closes, {len(stamps)} bars")
    meta = {k: v for k, v in blob.items() if k not in ("closes", "timestamps")}
    meta["bars"] = len(stamps)
    meta["first_bar"] = stamps[0] if stamps else None
    meta["last_bar"] = stamps[-1] if stamps else None
    meta["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return closes, meta


def freeze_snapshot(raw: Mapping[str, Sequence[tuple[str, float]]], meta: Mapping[str, Any],
                    path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    """Write fetched ``(timestamp, close)`` rows as one timestamp axis plus aligned closes."""
    from argus.eval.artefact import sanitise

    common = set.intersection(*(set(t for t, _ in rows) for rows in raw.values()))
    stamps = sorted(common)
    closes = {s: [dict(rows)[t] for t in stamps] for s, rows in raw.items()}
    blob = {**meta, "timestamps": stamps, "closes": closes,
            "dropped_unaligned_bars": {s: len(rows) - len(stamps) for s, rows in raw.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitise(blob), allow_nan=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return blob


# =================================================================================================
# Orchestration.
# =================================================================================================


CANDIDATES_CONSIDERED = [
    {"system": "statsmodels 0.15 (coint, multipletests)", "licence": "BSD-3-Clause",
     "evidence": "11.6k stars, pushed 2026-09-25; the default applied-econometrics stack; "
                 "ARGUS's own adf() was built by reading its stattools.py", "run": True},
    {"system": "arch 8.0 (engle_granger, phillips_ouliaris)", "licence": "NCSA (permissive)",
     "evidence": "1.6k stars, pushed 2026-09-24, Kevin Sheppard (Oxford); independent "
                 "re-simulated EG surfaces; the only maintained Python Phillips-Ouliaris",
     "run": True},
    {"system": "scipy 1.17 (false_discovery_control)", "licence": "BSD-3-Clause",
     "evidence": "15k stars; the most-installed FDR implementation in Python", "run": True},
    {"system": "statsmodels coint_johansen", "licence": "BSD-3-Clause",
     "evidence": "returns critical values at 90/95/99% only, no p-value: cannot feed an FDR "
                 "procedure without inventing a p-value surface", "run": False},
    {"system": "R urca / tseries / egcm", "licence": "GPL",
     "evidence": "no R interpreter on this machine (checked PATH and Program Files)", "run": False},
    {"system": "puolival/multipy", "licence": "BSD-3-Clause",
     "evidence": "112 stars, last push 2024-08; BH/BY already covered by statsmodels and scipy",
     "run": False},
]


SCOPE_STATEMENT = """\
Claimed: on identical series, ARGUS's production screen (research/cointegration.py scan(): \
Engle-Granger with AIC lags and MacKinnon 2010 surfaces on the first 60%, Benjamini-Hochberg at \
q=0.05) is scored against every combination of seven general-purpose cointegration tests \
(statsmodels coint; arch engle_granger with BIC and AIC; arch Phillips-Ouliaris Za, Zt, Pz) and \
eight multiple-testing procedures (ARGUS BH/Bonferroni; statsmodels fdr_bh, fdr_by, fdr_tsbh, \
holm; scipy BH/BY), on the selection slice and on the full sample, over Monte Carlo universes \
with known truth shaped exactly like the production input, and on the frozen real rToken \
universe. The verdict is computed by decide(), not written by hand.

NOT claimed: that the planted universes reproduce real rToken dynamics beyond their shape and \
half-lives; that the Monte Carlo covers heavy tails, volatility clustering or structural breaks \
other than the one decoupling scenario; that Johansen or R's urca were compared (see \
CANDIDATES_CONSIDERED for why); that arch's or statsmodels' code runs inside ARGUS's shipped \
import graph — both are eval-only, imported here and nowhere on the desk's path.
"""


def run_monte_carlo(
    reps: Mapping[str, int], *, workers: int = 1,
    evaluate: Callable[[tuple[str, int]], dict[str, Any]] = evaluate_seeded,
) -> dict[str, list[dict[str, Any]]]:
    """Every scenario, on fixed seeds: null 10_000+i, planted 20_000+i, broken 30_000+i."""
    base = {"null": 10_000, "planted": 20_000, "broken": 30_000}
    jobs = [(scenario, base[scenario] + i) for scenario, n in reps.items() for i in range(n)]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            done = list(pool.map(evaluate, jobs, chunksize=1))
    else:
        done = [evaluate(job) for job in jobs]
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in reps}
    for r in done:
        out[r["scenario"]].append(r)
    return out


def summarise(
    monte_carlo: Mapping[str, Sequence[Mapping[str, Any]]], real: Mapping[str, Any] | None,
    real_meta: Mapping[str, Any] | None, *, q: float = Q,
) -> dict[str, Any]:
    """Score every pipeline in every scenario and compute the verdict."""
    pipelines = sorted(next(iter(monte_carlo.values()))[0]["discoveries"])
    scores = {
        scenario: {p: score_pipeline(results, p) for p in pipelines}
        for scenario, results in monte_carlo.items() if results
    }
    decision = decide(scores, monte_carlo["planted"], q=q)
    protocol = protocol_finding(scores, q=q)
    all_results = [r for rs in monte_carlo.values() for r in rs]
    diffs = [r["checks"]["argus_vs_statsmodels_coint_max_abs_p_diff"] for r in all_results
             if r["checks"]["argus_vs_statsmodels_coint_max_abs_p_diff"] is not None]
    seconds: dict[str, float] = {}
    for r in all_results:
        for k, v in r["seconds"].items():
            seconds[k] = seconds.get(k, 0.0) + v
    n_pair_calls = sum(len(r["pairs"]) for r in all_results)
    headline = [PRODUCTION, CONFIRMED, "argus_eg|sm_by", "argus_eg|sm_tsbh", "sm_coint|sm_bh"]
    for key in ("best_general_pipeline", "best_argus_native_candidate"):
        if decision[key]:
            headline.append(decision[key]["pipeline"])
    headline += ["argus_eg+pre05|argus_bh", "arch_po_pz|sm_bh", "arch_eg_bic|sm_bh"]
    headline += ["sm_coint@full|sm_bh", "arch_eg_bic@full|sm_bh", "arch_po_pz@full|sm_bh"]
    headline = list(dict.fromkeys(p for p in headline if p in pipelines))
    return {
        "config": {
            "q": q, "train_fraction": TRAIN_FRACTION, "n_bars": N_BARS,
            "clusters": [{"label": c, "members": m, "phi": phi,
                          "half_life_bars": round(math.log(2) / (1 - phi), 1)}
                         for c, m, phi in CLUSTERS],
            "singletons": N_SINGLETONS, "decoupling_clusters": list(DECOUPLING_CLUSTERS),
            "reps": {s: len(rs) for s, rs in monte_carlo.items()},
        },
        "candidates_considered": CANDIDATES_CONSIDERED,
        "identity_checks": {
            "scan_survivors_fdr_equals_production_in_every_universe": all(
                r["checks"]["scan_survivors_fdr_equals_production"] for r in all_results),
            "argus_vs_statsmodels_coint_max_abs_p_diff": max(diffs) if diffs else None,
            "argus_bh_equals_statsmodels_and_scipy_bh_in_every_universe": all(
                r["checks"]["bh_implementations_agree"] for r in all_results),
        },
        "rival_failures": {
            k: sum(r["rival_failures"].get(k, 0) for r in all_results)
            for k in all_results[0]["rival_failures"]
        },
        "seconds_per_pair": {k: v / n_pair_calls for k, v in sorted(seconds.items())},
        "headline": {
            scenario: [scores[scenario][p].as_dict() for p in headline]
            for scenario in scores
        },
        "all_scores": {
            scenario: [s.as_dict() for s in by_p.values()] for scenario, by_p in scores.items()
        },
        "decision": decision,
        "protocol_finding": protocol,
        "real_rtoken_universe": None if real is None else {
            "meta": dict(real_meta or {}),
            "pairs_tested": len(real["pairs"]),
            "discoveries": {p: real["discoveries"][p] for p in sorted(real["discoveries"])
                            if real["discoveries"][p]},
            "pipelines_with_no_discovery": sum(1 for v in real["discoveries"].values() if not v),
            "lowest_pvalues": {
                k: sorted(
                    ({"pair": pair, "p": p} for pair, p in zip(real["pairs"], vs, strict=True)
                     if p is not None), key=lambda d: d["p"])[:3]
                for k, vs in real["pvalues"].items()
            },
            "checks": real["checks"],
        },
        "scope_statement": SCOPE_STATEMENT,
    }


def render(summary: Mapping[str, Any]) -> str:
    lines = ["GENERAL-PURPOSE RIVALS — pairwise cointegration screening with FDR control", ""]
    for scenario, rows in summary["headline"].items():
        lines.append(f"-- {scenario} ({summary['config']['reps'][scenario]} universes) --")
        for r in rows:
            power = "   n/a" if r["power"] is None else f"{r['power']:6.3f}"
            lines.append(
                f"  {r['pipeline']:30s} FDR={r['fdr']:.3f}±{r['fdr_se']:.3f} "
                f"FWER={r['fwer']:.3f} power={power} found={r['mean_discoveries']:.2f}"
            )
        lines.append("")
    d = summary["decision"]
    lines.append(f"verdict: {d['verdict']}  best general: {d['best_general_pipeline']}")
    if summary.get("real_rtoken_universe"):
        real = summary["real_rtoken_universe"]
        lines.append(f"real rToken universe: {real['pairs_tested']} pairs; "
                     f"discoveries: {real['discoveries'] or 'none by any pipeline'}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    from argus.eval.artefact import write

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--null", type=int, default=60)
    parser.add_argument("--planted", type=int, default=60)
    parser.add_argument("--broken", type=int, default=40)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=ARTEFACT_PATH)
    parser.add_argument("--raw-out", type=Path, default=None,
                        help="optional path for the per-universe results")
    parser.add_argument("--from-raw", type=Path, default=None,
                        help="re-score a saved --raw-out file instead of re-running the universes")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    if args.from_raw is not None:
        mc = json.loads(args.from_raw.read_text(encoding="utf-8"))
    else:
        mc = run_monte_carlo(
            {"null": args.null, "planted": args.planted, "broken": args.broken},
            workers=args.workers,
        )
        if args.raw_out is not None:
            write(args.raw_out, mc)
    real: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None
    if SNAPSHOT_PATH.exists():
        closes, meta = load_snapshot()
        real = evaluate_series(closes)
    summary = summarise(mc, real, meta)
    summary["wall_seconds"] = round(time.perf_counter() - started, 1)
    write(args.out, summary)
    print(render(summary))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "CANDIDATES_CONSIDERED",
    "CONFIRMED",
    "CORRECTIONS",
    "FULL_TESTS",
    "PRODUCTION",
    "SCOPE_STATEMENT",
    "TRAIN_TESTS",
    "GeneralCointComparisonError",
    "PipelineScore",
    "Universe",
    "build_universe",
    "controls_fdr",
    "decide",
    "evaluate_seeded",
    "evaluate_series",
    "freeze_snapshot",
    "is_general",
    "load_snapshot",
    "main",
    "null_universe",
    "paired_power_gain",
    "planted_universe",
    "protocol_finding",
    "reject",
    "render",
    "rival_pvalues",
    "run_monte_carlo",
    "score_pipeline",
    "summarise",
]
