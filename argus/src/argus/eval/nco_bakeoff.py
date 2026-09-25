"""Can ARGUS's allocator beat Nested Clustered Optimization -- every serious implementation of it,
tuned as well as it can be -- on a half of the history nothing was tuned on?

`eval/allocation_comparison.py` settled that ARGUS's own NCO (`desk/allocation.py:nco_weights`)
**ties** Riskfolio-Lib's real NCO: 7.97 vs 7.97bps mean realised out-of-sample volatility, weights
equal to 1.3e-05. Its register row stayed TIED with five conditions open, and its own blockers named
what closing them takes: sweep NCO's own hyperparameters (the tie was shown only at one
configuration), reproduce the method's published result rather than one library's output, ablate
the mechanism, and try a real algorithmic idea that might beat the specialist outright. This module
does all four, and it runs every rival unmodified:

* **Riskfolio-Lib 7.3.0** (BSD-3), `HCPortfolio.optimization(model="NCO")` -- its shipped default
  (``linkage="single"``, ``method_cov="hist"``, `HCPortfolio.py:760-782`) *and* a grid over the
  knobs it exposes: seven covariance estimators (`ParamsEstimation.py:147-275` -- ``hist``,
  ``ewma1`` at its default decay and at a half-life matched to ARGUS's, ``ledoit``, ``oas``,
  the de Prado ``fixed``/``spectral`` denoisers, ``gerber1``), three linkages, the ``stdsil``
  cluster-count rule and a spearman codependence; plus its plain long-only minimum variance
  (`Portfolio.optimization(model="Classic", rm="MV", obj="MinRisk")`), the Markowitz baseline NCO
  was invented to beat.
* **skfolio 1.3.1** (BSD-3), `optimization/cluster/_nco.py` -- a genuinely different NCO: its
  cluster count is a different two-order gap statistic (`utils/stats.py:570-640`, dispersion
  ``sum D_i / 2|C_i|`` with up to ``max(8, sqrt(n))`` clusters), and its outer weights are fitted on
  *cross-validated* out-of-sample cluster returns (`_nco.py:407-478`), not on the in-sample
  synthetic covariance Riskfolio and ARGUS use. Source read at skfolio commit 0223b231
  (2026-09-23); imported as an installed package, or -- so the shared interpreter need not change
  -- from a ``pip install --target`` directory named by ``ARGUS_SKFOLIO_PATH``. Its default and
  five covariance-estimator variants, with and without the CV step.
* **López de Prado's own code** for the method, *Machine Learning for Asset Managers* (2020) ch. 7,
  as transcribed in `emoen/Machine-Learning-for-Asset-Managers` (Apache-2.0, commit 5292b3c4, a
  clone named by ``ARGUS_MLAM_PATH``): `ch7_portfolio_construction.py:optPort_nco` with the ONC
  k-means clustering of `ch4_optimal_clustering.py:clusterKMeansBase`. Its inner and outer solves
  are the closed-form *unconstrained* minimum variance (`allocate_cvo`), so it may short -- a larger
  feasible set than every long-only allocator here, reported as such and never mixed into the
  long-only ranking.

**Reproducing it.** ``pip install riskfolio-lib==7.3.0 skfolio==1.3.1 cvxpy scipy pandas
scikit-learn matplotlib``, ``git clone https://github.com/emoen/Machine-Learning-for-Asset-Managers``
at 5292b3c4 and point ``ARGUS_MLAM_PATH`` at it. A rival that cannot be loaded is recorded per
window as an error in the artefact and listed under ``stage_failures`` -- never silently dropped.

**The protocol, fixed before the holdout was computed.** The frozen snapshot
(`eval/allocation_snapshot.py`, SHA-256 checked on load) is walked forward with the same 480-bar
estimation / 96-bar held-out / 96-bar step the older comparison uses. The origins are split in time:
the first half is the **selection** period, the second the **holdout**. Every family -- ARGUS's,
Riskfolio's, skfolio's -- picks its best configuration by mean realised volatility on the selection
period only, so the specialist gets exactly the tuning privilege ARGUS gets. The decisive numbers
are the holdout comparisons of those picks, and of each library's shipped default, against ARGUS's
pick: per-window log volatility ratios, an exact sign test, a Wilcoxon signed-rank test, a
stationary-bootstrap 95% interval on the mean log ratio, and Holm's step-down correction across the
whole family of comparisons. The verdict rule is stated in :func:`verdict` and applied mechanically.

**Beyond realised volatility.** Realised volatility on nine or forty-odd windows is a noisy score
with no ground truth. :func:`run_monte_carlo` adds the experiment de Prado used to justify NCO
(snippets 7.7-7.9): draw a sample from a *known* covariance, allocate, and score the allocation
against the true optimum -- weight RMSE and, for the long-only allocators, the true variance
``w' Sigma0 w`` relative to the true optimum's. Run on de Prado's block-diagonal process (his own
generator, unmodified) and on a process calibrated to the real selection-period covariance, with
Gaussian and Student-t(4) draws.

    python -m argus.eval.nco_bakeoff            # full run: selection, holdout, everything
    python -m argus.eval.nco_bakeoff --phase selection
    python -m argus.eval.nco_bakeoff --reuse-selection   # reuse a same-snapshot checkpoint
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from statistics import fmean
from typing import Any

from argus.desk.allocation import (
    AllocationError,
    correlation_distance,
    cut_clusters,
    hrp_weights,
    minimum_variance_weights,
    single_linkage,
    two_diff_gap_stat,
    ward_linkage,
)
from argus.desk.nco_ensemble import NCOConfig, ensemble_nco_weights, estimate, sample_covariance
from argus.eval.allocation_snapshot import load_snapshot
from argus.eval.artefact import write

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "nco_bakeoff.json"
CHECKPOINT_PATH = DATA / "nco_bakeoff_selection.json"

TRAIN_BARS = 480
TEST_BARS = 96
STEP_BARS = 96
TAKER_BPS = 6.0
"""Same walk-forward geometry and the same fee as `eval/allocation_comparison.py`."""

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_BLOCK = 4.0
"""Mean block length, in windows, of the stationary bootstrap on per-window log ratios. Adjacent
held-out windows touch, so a volatility regime can span two or three of them; resampling single
windows would treat that persistence as independent evidence."""

SKFOLIO_ENV = "ARGUS_SKFOLIO_PATH"
"""Optional: a ``pip install --target`` directory holding skfolio, used only when ``import skfolio``
fails in the running interpreter."""

MLAM_ENV = "ARGUS_MLAM_PATH"
"""Required for the de Prado rival: a clone of `emoen/Machine-Learning-for-Asset-Managers`."""

Weights = dict[str, float]
Columns = Mapping[str, Sequence[float]]
Transform = Callable[[dict[str, list[float]], int], dict[str, list[float]]]


class RivalUnavailable(RuntimeError):
    """A rival library could not be loaded. Raised, never skipped: a bake-off that quietly drops
    the specialist is the failure this module exists to prevent."""


# --- loading the rivals -----------------------------------------------------------------------


def _ensure_skfolio() -> None:
    try:
        import skfolio  # type: ignore[import-not-found, unused-ignore]
        return
    except ImportError:
        pass
    target = os.environ.get(SKFOLIO_ENV)
    if not target or not Path(target).exists():
        raise RivalUnavailable(
            f"skfolio is not importable; `pip install skfolio==1.3.1`, or set {SKFOLIO_ENV} to a "
            f"`pip install --target` directory holding it (currently {target!r})")
    if target not in sys.path:
        sys.path.insert(0, target)
    import skfolio  # type: ignore[import-not-found, unused-ignore]  # noqa: F401


PLOT_ONLY_IMPORTS = ("seaborn",)
"""Modules de Prado's chapter files import at the top and use only under ``if __name__ ==
'__main__'`` (`ch7_portfolio_construction.py:7` imports seaborn; its only uses are the heatmaps at
:93 and :105, inside the ``__main__`` block from :85). When one is not installed, an empty module
stands in for it so the allocation code runs unmodified; nothing the bake-off calls touches it.
Without this the first full run recorded de Prado's NCO as 20/20 failed windows and lost the whole
Monte Carlo stage to ``ModuleNotFoundError: seaborn`` -- a missing plotting library, not a
result."""


def _stub_plot_only_imports() -> list[str]:
    """Install an empty stand-in for each absent plot-only module; returns the names stubbed."""
    import importlib.util
    import types

    stubbed: list[str] = []
    for name in PLOT_ONLY_IMPORTS:
        if name in sys.modules or importlib.util.find_spec(name) is not None:
            continue
        sys.modules[name] = types.ModuleType(name)
        stubbed.append(name)
    return stubbed


def _mlam(module: str = "ch7_portfolio_construction") -> Any:
    """One of de Prado's chapter modules, imported from the clone unmodified. ``MPLBACKEND=Agg``
    because the modules import `matplotlib.pylab` at the top for their own ``__main__`` plots."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    root = os.environ.get(MLAM_ENV)
    if not root or not Path(root).exists():
        raise RivalUnavailable(
            f"set {MLAM_ENV} to a clone of github.com/emoen/Machine-Learning-for-Asset-Managers "
            f"(commit 5292b3c4); currently {root!r}")
    if root not in sys.path:
        sys.path.insert(0, root)
    _stub_plot_only_imports()
    import importlib

    return importlib.import_module(f"Machine_Learning_for_Asset_Managers.{module}")


# --- helpers ----------------------------------------------------------------------------------


def _names(columns: Columns) -> list[str]:
    return sorted(columns)


def _frame(columns: Columns) -> Any:
    import pandas as pd

    names = _names(columns)
    return pd.DataFrame({n: list(columns[n]) for n in names})


def _rows(columns: Columns) -> list[list[float]]:
    names = _names(columns)
    return [[float(columns[n][t]) for n in names] for t in range(len(columns[names[0]]))]


def realised_vol(weights: Mapping[str, float], columns: Columns) -> float:
    """Sample standard deviation (ddof=1) of the fixed-weight portfolio's realised bar returns --
    the same score `eval/allocation_comparison.py:_realised_vol` uses."""
    names = _names(columns)
    size = len(columns[names[0]])
    port = [sum(weights.get(n, 0.0) * columns[n][i] for n in names) for i in range(size)]
    mean = sum(port) / size
    return math.sqrt(sum((p - mean) ** 2 for p in port) / (size - 1))


def _window(columns: Columns, start: int, size: int) -> dict[str, list[float]]:
    return {n: list(v[start:start + size]) for n, v in columns.items()}


def _effective_positions(weights: Mapping[str, float]) -> float:
    total = sum(w * w for w in weights.values())
    return 1.0 / total if total > 0 else float("nan")


# --- the candidates ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One allocator: who wrote it, how it is configured, and whether it is a library default."""

    label: str
    family: str
    fn: Callable[[Columns], Weights]
    config: dict[str, Any] = field(default_factory=dict)
    default: bool = False
    long_only: bool = True


def _ensemble(cols: Columns, config: NCOConfig) -> Weights:
    return ensemble_nco_weights(cols, config)


def _argus_minvar(cols: Columns, config: NCOConfig) -> Weights:
    """ARGUS's long-only minimum variance (`allocation.py:minimum_variance_weights`, active set) on
    the configured covariance estimate -- NCO with the clustering step removed."""
    return minimum_variance_weights(_names(cols), estimate(_rows(cols), config))


def _config_dict(c: NCOConfig) -> dict[str, Any]:
    return {"estimator": c.estimator, "halflife": c.halflife, "n_boot": c.n_boot,
            "block": c.block, "seed": c.seed}


def argus_candidates() -> list[Candidate]:
    """ARGUS's family, fixed before any holdout window was scored: its NCO on three covariance
    estimators (sample, Ledoit-Wolf constant correlation, EWMA with a 240-bar half-life), each with
    and without 32-resample bagging, and its long-only minimum variance -- NCO with the clustering
    removed -- on the same three estimators. Nine configurations."""
    configs = [
        NCOConfig("sample"),
        NCOConfig("lw_cc"),
        NCOConfig("ewma", halflife=240.0),
        NCOConfig("sample", n_boot=32),
        NCOConfig("lw_cc", n_boot=32),
        NCOConfig("ewma", halflife=240.0, n_boot=32),
    ]
    out = [
        Candidate(c.label, "argus", partial(_ensemble, config=c),
                  {**_config_dict(c), "clustering": "nco"}, default=(c == NCOConfig("sample")))
        for c in configs
    ]
    for c in (NCOConfig("sample"), NCOConfig("lw_cc"), NCOConfig("ewma", halflife=240.0)):
        est = c.estimator if c.halflife is None else f"{c.estimator}{c.halflife:g}"
        out.append(Candidate(f"argus_minvar[{est}]", "argus", partial(_argus_minvar, config=c),
                             {**_config_dict(c), "clustering": "none"}))
    return out


def candidate_from_config(label: str, config: Mapping[str, Any]) -> Candidate:
    """Rebuild an ARGUS candidate from the config recorded in an artefact."""
    c = NCOConfig(str(config.get("estimator", "sample")), config.get("halflife"),
                  int(config.get("n_boot", 0) or 0), float(config.get("block", 24.0)),
                  int(config.get("seed", 20260925)))
    fn = (partial(_argus_minvar, config=c) if config.get("clustering") == "none"
          else partial(_ensemble, config=c))
    return Candidate(label, "argus", fn, dict(config))


RISKFOLIO_COVS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("hist", {}),
    ("ewma1", {}),
    ("ewma1", {"d": 0.5 ** (1 / 240)}),
    ("ledoit", {}),
    ("oas", {}),
    ("fixed", {}),
    ("spectral", {}),
    ("gerber1", {}),
)


def _riskfolio_nco(cols: Columns, **options: Any) -> Weights:
    from argus.eval.allocation_comparison import riskfolio_weights

    return riskfolio_weights(cols, model="NCO", **dict(options))[0]


def _riskfolio_minvar(cols: Columns, method_cov: str = "hist",
                      dict_cov: Mapping[str, Any] | None = None) -> Weights:
    import riskfolio as rp  # type: ignore[import-untyped, unused-ignore]

    frame = _frame(cols)
    port = rp.Portfolio(returns=frame)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        port.assets_stats(method_mu="hist", method_cov=method_cov, dict_cov=dict(dict_cov or {}))
        result = port.optimization(model="Classic", rm="MV", obj="MinRisk", rf=0, l=0, hist=True)
    if result is None:
        raise AllocationError("riskfolio returned no minimum-variance weights")
    return {n: float(result["weights"][n]) for n in _names(cols)}


def riskfolio_candidates() -> list[Candidate]:
    out: list[Candidate] = []
    for linkage in ("single", "ward", "average"):
        for cov, dict_cov in RISKFOLIO_COVS:
            tag = cov if not dict_cov else f"{cov}(d={dict_cov['d']:.5f})"
            options: dict[str, Any] = {"linkage": linkage, "method_cov": cov}
            if dict_cov:
                options["dict_cov"] = dict(dict_cov)
            out.append(Candidate(
                f"riskfolio_nco[{tag},{linkage}]", "riskfolio",
                partial(_riskfolio_nco, **options), dict(options),
                default=(linkage == "single" and cov == "hist" and not dict_cov),
            ))
    extras: tuple[tuple[str, dict[str, Any]], ...] = (
        ("hist,ward,stdsil", {"linkage": "ward", "method_cov": "hist", "opt_k_method": "stdsil"}),
        ("hist,ward,spearman", {"linkage": "ward", "method_cov": "hist",
                                "codependence": "spearman"}),
    )
    for extra, options in extras:
        out.append(Candidate(f"riskfolio_nco[{extra}]", "riskfolio",
                             partial(_riskfolio_nco, **options), dict(options)))
    for cov, dict_cov in (("hist", {}), ("ledoit", {}), ("ewma1", {"d": 0.5 ** (1 / 240)})):
        tag = cov if not dict_cov else f"{cov}(d={dict_cov['d']:.5f})"
        out.append(Candidate(f"riskfolio_minvar[{tag}]", "riskfolio",
                             partial(_riskfolio_minvar, method_cov=cov, dict_cov=dict_cov),
                             {"model": "Classic", "rm": "MV", "obj": "MinRisk", "method_cov": cov,
                              "dict_cov": dict_cov}))
    return out


def _skfolio_nco(cols: Columns, cov: str | None, cv: Any) -> Weights:
    _ensure_skfolio()
    from skfolio.moments import (  # type: ignore[import-not-found, unused-ignore]
        DenoiseCovariance,
        EWCovariance,
        GerberCovariance,
        LedoitWolf,
    )
    from skfolio.optimization import (  # type: ignore[import-not-found, unused-ignore]
        MeanRisk,
        NestedClustersOptimization,
    )
    from skfolio.prior import EmpiricalPrior  # type: ignore[import-not-found, unused-ignore]

    makers: dict[str, Callable[[], Any]] = {
        "ledoit_wolf": LedoitWolf,
        "denoise": DenoiseCovariance,
        "ew240": lambda: EWCovariance(half_life=240),
        "gerber": GerberCovariance,
    }

    def estimator() -> Any:
        if cov is None:
            return None
        return MeanRisk(prior_estimator=EmpiricalPrior(covariance_estimator=makers[cov]()))

    model = NestedClustersOptimization(
        inner_estimator=estimator(), outer_estimator=estimator(), cv=cv,
    )
    model.fit(_frame(cols))
    return {n: float(w) for n, w in zip(_names(cols), model.weights_, strict=True)}


def skfolio_candidates() -> list[Candidate]:
    out = [Candidate("skfolio_nco[default]", "skfolio",
                     (lambda cols: _skfolio_nco(cols, None, None)),
                     {"cv": "KFold(5)", "covariance": "EmpiricalCovariance"}, default=True),
           Candidate("skfolio_nco[cv=ignore]", "skfolio",
                     (lambda cols: _skfolio_nco(cols, None, "ignore")),
                     {"cv": "ignore", "covariance": "EmpiricalCovariance"})]
    for cov in ("ledoit_wolf", "denoise", "ew240", "gerber"):
        out.append(Candidate(f"skfolio_nco[{cov}]", "skfolio",
                             partial(_skfolio_nco, cov=cov, cv=None),
                             {"cv": "KFold(5)", "covariance": cov}))
    return out


def _deprado_nco(cols: Columns, seed: int = 0) -> Weights:
    """de Prado's `optPort_nco`, unmodified, on the ddof=1 sample covariance, ``maxNumClusters =
    N/2`` as his snippet 7.8 passes it. `KMeans` there takes no ``random_state``, so the global
    numpy generator is seeded immediately before the call -- the only way to make it reproducible
    without editing his code -- and his diagnostic ``print`` is captured."""
    import numpy as np

    ch7 = _mlam()
    names = _names(cols)
    cov = np.cov(np.array(_rows(cols)), rowvar=False)
    np.random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        w = ch7.optPort_nco(cov, None, int(len(names) / 2))
    return {n: float(v) for n, v in zip(names, np.asarray(w).flatten(), strict=True)}


def reference_candidates() -> list[Candidate]:
    def hrp(cols: Columns) -> Weights:
        names = _names(cols)
        return hrp_weights(names, sample_covariance(_rows(cols)))

    def equal(cols: Columns) -> Weights:
        names = _names(cols)
        return dict.fromkeys(names, 1.0 / len(names))

    return [
        Candidate("argus_hrp", "reference", hrp),
        Candidate("equal_weight", "reference", equal),
        Candidate("deprado_nco[mlam_ch7]", "deprado", _deprado_nco,
                  {"clustering": "ONC k-means", "solver": "closed-form unconstrained"},
                  default=True, long_only=False),
    ]


# --- the walk-forward -------------------------------------------------------------------------


def origins(total: int, train: int = TRAIN_BARS, test: int = TEST_BARS,
            step: int = STEP_BARS) -> list[int]:
    return list(range(0, max(0, total - train - test) + 1, step))


def split(all_origins: Sequence[int]) -> tuple[list[int], list[int]]:
    """First half selection, second half holdout, in time order. An odd count gives the holdout
    the extra window, because the holdout is where the claim is made."""
    half = len(all_origins) // 2
    return list(all_origins[:half]), list(all_origins[half:])


def evaluate(candidate: Candidate, columns: Columns, window_origins: Sequence[int], *,
             train: int = TRAIN_BARS, test: int = TEST_BARS,
             transform: Transform | None = None,
             ) -> dict[str, Any]:
    """Run one allocator over the given origins. Every number per window is kept, so the
    statistics are computed from the record rather than from a summary of it."""
    vols: list[float | None] = []
    turnover: list[float | None] = []
    eff: list[float | None] = []
    largest: list[float | None] = []
    short: list[float | None] = []
    errors: list[str] = []
    seconds: list[float] = []
    previous: Weights | None = None
    for origin in window_origins:
        fit = _window(columns, origin, train)
        if transform is not None:
            fit = transform(fit, origin)
        held = _window(columns, origin + train, test)
        started = time.perf_counter()
        try:
            weights = candidate.fn(fit)
        except Exception as exc:  # a failure is a result, recorded per window
            errors.append(f"{origin}: {type(exc).__name__}: {str(exc)[:120]}")
            vols.append(None)
            turnover.append(None)
            eff.append(None)
            largest.append(None)
            short.append(None)
            previous = None
            continue
        seconds.append(time.perf_counter() - started)
        vols.append(realised_vol(weights, held) * 10_000)
        turnover.append(
            None if previous is None
            else sum(abs(weights.get(n, 0.0) - previous.get(n, 0.0)) for n in weights)
        )
        eff.append(_effective_positions(weights))
        largest.append(max(weights.values()))
        short.append(-sum(w for w in weights.values() if w < 0))
        previous = weights
    finite = [v for v in vols if v is not None]
    moves = [t for t in turnover if t is not None]
    return {
        "label": candidate.label,
        "family": candidate.family,
        "config": {k: v for k, v in candidate.config.items()},
        "library_default": candidate.default,
        "long_only": candidate.long_only,
        "origins": list(window_origins),
        "oos_vol_bps": vols,
        "mean_oos_vol_bps": fmean(finite) if finite else None,
        "n_windows_scored": len(finite),
        "n_errors": len(errors),
        "errors": errors[:5],
        "turnover_per_rebalance": fmean(moves) if moves else None,
        "cost_bps_per_rebalance": fmean(moves) * TAKER_BPS if moves else None,
        "mean_effective_positions": fmean([e for e in eff if e is not None]) if finite else None,
        "mean_largest_weight": fmean([x for x in largest if x is not None]) if finite else None,
        "mean_gross_short": fmean([s for s in short if s is not None]) if finite else None,
        "seconds_per_allocation": fmean(seconds) if seconds else None,
    }


# --- statistics -------------------------------------------------------------------------------


def _sign_p(wins: int, trials: int) -> float | None:
    if trials == 0:
        return None
    probs = [math.comb(trials, k) * 0.5 ** trials for k in range(trials + 1)]
    return min(1.0, sum(p for p in probs if p <= probs[wins] * (1 + 1e-12)))


def _stationary_bootstrap_ci(values: Sequence[float], *, resamples: int = BOOTSTRAP_RESAMPLES,
                             block: float = BOOTSTRAP_BLOCK, seed: int = 7) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    p = 1.0 / block
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        current = rng.randrange(n)
        for _ in range(n):
            total += values[current]
            current = rng.randrange(n) if rng.random() < p else (current + 1) % n
        means.append(total / n)
    means.sort()
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def compare(argus: Sequence[float | None], rival: Sequence[float | None]) -> dict[str, Any]:
    """Paired per-window comparison. ``log_ratio = ln(rival / argus)``: positive means ARGUS's
    realised volatility was lower in that window."""
    from scipy.stats import wilcoxon  # type: ignore[import-untyped, unused-ignore]

    pairs = [(a, b) for a, b in zip(argus, rival, strict=True) if a is not None and b is not None]
    ratios = [math.log(b / a) for a, b in pairs]
    wins = sum(1 for r in ratios if r > 1e-9)
    losses = sum(1 for r in ratios if r < -1e-9)
    nonzero = [r for r in ratios if abs(r) > 1e-9]
    wilcoxon_p = float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 6 else None
    lo, hi = _stationary_bootstrap_ci(ratios) if ratios else (float("nan"), float("nan"))
    mean_a = fmean(a for a, _ in pairs) if pairs else float("nan")
    mean_b = fmean(b for _, b in pairs) if pairs else float("nan")
    return {
        "n_windows": len(pairs),
        "argus_mean_oos_vol_bps": mean_a,
        "rival_mean_oos_vol_bps": mean_b,
        "rival_over_argus_mean_vol": mean_b / mean_a if pairs else None,
        "mean_log_ratio": fmean(ratios) if ratios else None,
        "bootstrap_ci95_mean_log_ratio": [lo, hi],
        "argus_wins": wins,
        "rival_wins": losses,
        "ties": len(ratios) - wins - losses,
        "sign_test_p": _sign_p(wins, wins + losses),
        "wilcoxon_p": wilcoxon_p,
    }


def holm(pvalues: Mapping[str, float | None], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Holm-Bonferroni step-down. A missing p-value is not tested and never rejects."""
    tested = sorted(((k, v) for k, v in pvalues.items() if v is not None), key=lambda kv: kv[1])
    out: dict[str, dict[str, Any]] = {
        k: {"threshold": None, "rejects": False} for k, v in pvalues.items() if v is None
    }
    still = True
    m = len(tested)
    for rank, (label, p) in enumerate(tested):
        threshold = alpha / (m - rank)
        still = still and p <= threshold
        out[label] = {"threshold": threshold, "rejects": still}
    return out


def verdict(comparisons: Mapping[str, Mapping[str, Any]],
            corrected: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The mechanical rule, written before the holdout was run.

    **WIN** against a rival: the Wilcoxon p survives Holm across the whole family *and* the
    bootstrap interval on the mean log ratio lies entirely above zero. **LOSS**: the same, below
    zero. Anything else is a **TIE** -- including a mean that favours one side inside the noise.
    The capability is OWNED-eligible on this criterion only if it WINS against every rival in the
    decisive set; one LOSS makes it LOST; otherwise TIED.
    """
    per: dict[str, str] = {}
    for label, row in comparisons.items():
        lo, hi = row["bootstrap_ci95_mean_log_ratio"]
        significant = bool(corrected.get(label, {}).get("rejects"))
        if significant and lo > 0:
            per[label] = "WIN"
        elif significant and hi < 0:
            per[label] = "LOSS"
        else:
            per[label] = "TIE"
    if per and all(v == "WIN" for v in per.values()):
        overall = "WIN"
    elif any(v == "LOSS" for v in per.values()):
        overall = "LOSS"
    else:
        overall = "TIE"
    return {"per_rival": per, "overall": overall}


# --- selection and holdout --------------------------------------------------------------------


def _pick(rows: Sequence[Mapping[str, Any]]) -> str:
    scored = [r for r in rows if r["mean_oos_vol_bps"] is not None and r["n_errors"] == 0]
    if not scored:
        raise AllocationError("no candidate in this family scored every selection window")
    return str(min(scored, key=lambda r: r["mean_oos_vol_bps"])["label"])


def run_selection(columns: Columns, window_origins: Sequence[int],
                  candidates: Sequence[Candidate], log: Callable[[str], None] = print,
                  reuse: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Score every candidate on the selection windows and pick each family's best.

    ``reuse`` takes rows from an earlier run of this same function on the same snapshot and the
    same origins (the caller checks both); a candidate found there is not recomputed. Every
    allocator here is deterministic -- `run_reproducibility` proves it for the decisive ones -- so a
    reused row is the row a recomputation would produce."""
    rows: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        prior = (reuse or {}).get(cand.label)
        if prior is not None and list(prior.get("origins", [])) == list(window_origins):
            rows[cand.label] = {**prior, "config": dict(cand.config), "reused": True}
        else:
            rows[cand.label] = evaluate(cand, columns, window_origins)
        log(f"  selection {cand.label:48} {rows[cand.label]['mean_oos_vol_bps']}")
    picks: dict[str, str] = {}
    for family in ("argus", "riskfolio", "skfolio"):
        members = [rows[c.label] for c in candidates if c.family == family]
        if members:
            picks[family] = _pick(members)
    return {"origins": list(window_origins), "results": rows, "picks": picks}


def decisive_rivals(candidates: Sequence[Candidate], picks: Mapping[str, str]) -> list[str]:
    """Each library's shipped default, each library's selection-period pick, and de Prado's own
    code. Deduplicated -- a library whose default is also its pick is compared once."""
    labels: list[str] = []
    for family in ("riskfolio", "skfolio"):
        for cand in candidates:
            if cand.family == family and cand.default and cand.label not in labels:
                labels.append(cand.label)
        if family in picks and picks[family] not in labels:
            labels.append(picks[family])
    return labels


def run_holdout(columns: Columns, window_origins: Sequence[int],
                candidates: Sequence[Candidate], picks: Mapping[str, str],
                log: Callable[[str], None] = print) -> dict[str, Any]:
    by_label = {c.label: c for c in candidates}
    argus_label = picks["argus"]
    rivals = decisive_rivals(candidates, picks)
    context = [c.label for c in candidates
               if c.label != argus_label and c.label not in rivals]
    rows: dict[str, dict[str, Any]] = {}
    for label in [argus_label, *rivals, *context]:
        rows[label] = evaluate(by_label[label], columns, window_origins)
        log(f"  holdout {label:48} {rows[label]['mean_oos_vol_bps']}")
    argus_vols = rows[argus_label]["oos_vol_bps"]
    comparisons = {r: compare(argus_vols, rows[r]["oos_vol_bps"]) for r in rivals}
    corrected = holm({r: comparisons[r]["wilcoxon_p"] for r in rivals})
    for r in rivals:
        comparisons[r]["holm"] = corrected[r]
    context_comparisons = {
        r: compare(argus_vols, rows[r]["oos_vol_bps"]) for r in context
    }
    oracle: dict[str, Any] = {}
    for family in ("riskfolio", "skfolio"):
        members = [rows[c.label] for c in candidates
                   if c.family == family and rows[c.label]["n_errors"] == 0
                   and rows[c.label]["mean_oos_vol_bps"] is not None]
        if members:
            best = min(members, key=lambda r: r["mean_oos_vol_bps"])
            oracle[family] = {
                "label": best["label"],
                "note": "chosen WITH hindsight on the holdout itself -- biased in the rival's "
                        "favour, so a win against it is stronger than a win against the "
                        "selection-period pick, and a loss to it is weaker evidence",
                "comparison": compare(argus_vols, best["oos_vol_bps"]),
            }
    return {
        "origins": list(window_origins),
        "argus_pick": argus_label,
        "oracle_rivals": oracle,
        "decisive_rivals": rivals,
        "results": rows,
        "comparisons": comparisons,
        "context_comparisons": context_comparisons,
        "verdict": verdict(comparisons, corrected),
    }


def run_holdout_dense(columns: Columns, first_test_bar: int, candidates: Sequence[Candidate],
                      labels: Sequence[str], *, train: int = TRAIN_BARS // 2,
                      test: int = TEST_BARS // 2, log: Callable[[str], None] = print,
                      ) -> dict[str, Any]:
    """The decisive comparison again at half the estimation and held-out lengths -- roughly twice
    the windows -- restricted to held-out windows that start inside the holdout period, so no bar
    the selection step scored is scored again. The picks are NOT re-chosen for this geometry: it
    asks whether the verdict survives a split length nobody tuned on, not which configuration
    would win under it."""
    total = len(next(iter(columns.values())))
    start = max(0, first_test_bar - train)
    window_origins = list(range(start, total - train - test + 1, test))
    by_label = {c.label: c for c in candidates}
    rows = {lab: evaluate(by_label[lab], columns, window_origins, train=train, test=test)
            for lab in labels}
    scenario = _scenario(rows, labels[0], dropped=[])
    log(f"  holdout (dense {train}/{test}): {scenario['verdict']['overall']}")
    return {"train_bars": train, "test_bars": test, "origins": window_origins, **scenario}


# --- ablation ---------------------------------------------------------------------------------


def _nco_variant(cov: Sequence[Sequence[float]], names: Sequence[str], *, linkage: str = "ward",
                 k: int | None = None, intra: str = "minvar", outer: str = "minvar",
                 labels_override: Sequence[int] | None = None) -> Weights:
    """NCO with one component swapped, built from `desk/allocation.py`'s own primitives so every
    arm differs from `nco_weights` in exactly the component named and nothing else."""
    n = len(names)
    distance = correlation_distance(cov)
    merges = ward_linkage(distance) if linkage == "ward" else single_linkage(distance)
    chosen = k if k is not None else two_diff_gap_stat(distance, merges)
    labels = list(labels_override) if labels_override is not None else cut_clusters(
        merges, n, max(1, min(chosen, n)))
    clusters = sorted(set(labels))
    intra_w: list[dict[str, float]] = []
    for c in clusters:
        members = [names[i] for i in range(n) if labels[i] == c]
        sub = [[cov[names.index(a)][names.index(b)] for b in members] for a in members]
        if intra == "minvar":
            w = minimum_variance_weights(members, sub)
        else:
            inv = [1.0 / sub[i][i] for i in range(len(members))]
            w = {m: inv[i] / sum(inv) for i, m in enumerate(members)}
        intra_w.append({name: w.get(name, 0.0) for name in names})
    kk = len(clusters)
    ccov = [[sum(intra_w[p][names[a]] * cov[a][b] * intra_w[q][names[b]]
                 for a in range(n) for b in range(n)) for q in range(kk)] for p in range(kk)]
    cnames = [f"c{i}" for i in range(kk)]
    inter = (minimum_variance_weights(cnames, ccov) if outer == "minvar"
             else dict.fromkeys(cnames, 1.0 / kk))
    out = dict.fromkeys(names, 0.0)
    for i in range(kk):
        for name, weight in intra_w[i].items():
            out[name] += weight * inter[f"c{i}"]
    return out


def ablation_candidates(pick: NCOConfig) -> list[Candidate]:
    """One arm per component of ARGUS's picked allocator, each removing exactly one thing."""
    def on_estimate(transform: Callable[[list[list[float]], list[str]], Weights],
                    config: NCOConfig) -> Callable[[Columns], Weights]:
        def fn(cols: Columns) -> Weights:
            names = _names(cols)
            return transform(estimate(_rows(cols), config), names)
        return fn

    base = NCOConfig(pick.estimator, pick.halflife)
    arms: list[Candidate] = []
    if pick.n_boot:
        arms.append(Candidate("ablate:no_bagging", "ablation",
                              (lambda cols: ensemble_nco_weights(cols, base)), {}))
    if pick.estimator != "sample":
        arms.append(Candidate("ablate:sample_covariance", "ablation",
                              (lambda cols: ensemble_nco_weights(
                                  cols, NCOConfig("sample", None, pick.n_boot, pick.block,
                                                  pick.seed))), {}))
    arms += [
        Candidate("ablate:no_clustering(k=1)", "ablation",
                  on_estimate(lambda cov, names: minimum_variance_weights(names, cov), base), {}),
        Candidate("ablate:single_linkage", "ablation",
                  on_estimate(lambda cov, names: _nco_variant(cov, names, linkage="single"),
                              base), {}),
        Candidate("ablate:intra_inverse_variance", "ablation",
                  on_estimate(lambda cov, names: _nco_variant(cov, names, intra="ivp"), base), {}),
        Candidate("ablate:outer_equal_weight", "ablation",
                  on_estimate(lambda cov, names: _nco_variant(cov, names, outer="equal"), base),
                  {}),
    ]

    def random_clusters(cols: Columns) -> Weights:
        names = _names(cols)
        cov = estimate(_rows(cols), base)
        distance = correlation_distance(cov)
        merges = ward_linkage(distance)
        k = two_diff_gap_stat(distance, merges)
        labels = cut_clusters(merges, len(names), max(1, min(k, len(names))))
        shuffled = list(labels)
        random.Random(len(cols[names[0]]) + int(1e6 * abs(cols[names[0]][0]))).shuffle(shuffled)
        return _nco_variant(cov, names, labels_override=shuffled)

    arms.append(Candidate("ablate:random_clusters_same_sizes", "ablation", random_clusters, {}))
    return arms


def _closed_form_minvar(cols: Columns, config: NCOConfig) -> Weights:
    """``S^-1 1 / 1'S^-1 1`` -- the unconstrained minimum variance, shorts allowed."""
    from argus.desk.allocation import invert_matrix

    names = _names(cols)
    inverse = invert_matrix(estimate(_rows(cols), config))
    raw = [sum(row) for row in inverse]
    total = sum(raw)
    return {n: raw[i] / total for i, n in enumerate(names)}


def _bagged_minvar(cols: Columns, config: NCOConfig) -> Weights:
    from argus.desk.nco_ensemble import stationary_bootstrap_indices

    names = _names(cols)
    rows = _rows(cols)
    rng = random.Random(config.seed)
    total = dict.fromkeys(names, 0.0)
    for _ in range(config.n_boot):
        pick = stationary_bootstrap_indices(len(rows), config.block, rng)
        w = minimum_variance_weights(names, estimate([rows[k] for k in pick], config))
        for n in names:
            total[n] += w[n]
    return {n: total[n] / config.n_boot for n in names}


def minvar_ablation_candidates(config: NCOConfig) -> list[Candidate]:
    """Arms around a long-only minimum-variance pick, each changing exactly one thing: put NCO's
    clustering back, drop the long-only constraint, swap the covariance estimator, bag it."""
    arms = [
        Candidate("ablate:add_nco_clustering", "ablation", partial(_ensemble, config=config)),
        Candidate("ablate:drop_long_only(closed_form)", "ablation",
                  partial(_closed_form_minvar, config=config), long_only=False),
        Candidate("ablate:add_bagging32", "ablation",
                  partial(_bagged_minvar, config=NCOConfig(config.estimator, config.halflife,
                                                           32, config.block, config.seed))),
    ]
    for other in (NCOConfig("sample"), NCOConfig("lw_cc"), NCOConfig("ewma", halflife=240.0)):
        if (other.estimator, other.halflife) != (config.estimator, config.halflife):
            arms.append(Candidate(f"ablate:covariance={other.label[10:-1]}", "ablation",
                                  partial(_argus_minvar, config=other)))
    return arms


def run_ablation(columns: Columns, window_origins: Sequence[int], arms: Sequence[Candidate],
                 base_vols: Sequence[float | None], log: Callable[[str], None] = print,
                 ) -> dict[str, Any]:
    """Each arm on the holdout, compared per window with the full allocator it was cut from.
    ``versus_full.mean_log_ratio > 0`` means removing the component made volatility WORSE, i.e.
    the component earns its place."""
    out: dict[str, Any] = {}
    for arm in arms:
        row = evaluate(arm, columns, window_origins)
        row["versus_full"] = compare(base_vols, row["oos_vol_bps"])
        out[arm.label] = row
        log(f"  ablation {arm.label:40} {row['mean_oos_vol_bps']}")
    return out


# --- adversarial ------------------------------------------------------------------------------


def _bad_tick(fit: dict[str, list[float]], origin: int) -> dict[str, list[float]]:
    """One corrupted print per estimation window: a +/-15% hourly return on a random asset at a
    random bar -- a bad tick of the kind a thin 24/7 venue produces. Seeded by the origin."""
    rng = random.Random(9_000 + origin)
    names = sorted(fit)
    victim = names[rng.randrange(len(names))]
    bar = rng.randrange(len(fit[victim]))
    out = {n: list(v) for n, v in fit.items()}
    out[victim][bar] = 0.15 if rng.random() < 0.5 else -0.15
    return out


ADVERSARIAL_UNIVERSES: dict[str, tuple[str, ...]] = {
    "without_leveraged_and_inverse_etfs": ("TQQQUSDT", "SQQQUSDT"),
    "single_stocks_only": ("QQQUSDT", "TQQQUSDT", "SQQQUSDT"),
}
"""The inverse pair lets a minimum-variance allocator build a near-flat synthetic position inside
the book (`allocation_comparison.py:INVERSE_PAIR`). Removing it asks whether an advantage survives
when the only way to lower variance is genuine diversification."""


def run_adversarial(columns: Columns, window_origins: Sequence[int],
                    candidates: Sequence[Candidate], labels: Sequence[str],
                    log: Callable[[str], None] = print) -> dict[str, Any]:
    by_label = {c.label: c for c in candidates}
    argus_label = labels[0]
    scenarios: dict[str, Any] = {}
    for name, dropped in ADVERSARIAL_UNIVERSES.items():
        sub = {n: v for n, v in columns.items() if n not in dropped}
        rows = {lab: evaluate(by_label[lab], sub, window_origins) for lab in labels}
        scenarios[name] = _scenario(rows, argus_label, dropped=list(dropped))
        log(f"  adversarial {name}: {scenarios[name]['verdict']['overall']}")
    rows = {lab: evaluate(by_label[lab], columns, window_origins, transform=_bad_tick)
            for lab in labels}
    scenarios["bad_tick_in_estimation_window"] = _scenario(rows, argus_label, dropped=[])
    tick = scenarios["bad_tick_in_estimation_window"]
    log(f"  adversarial bad tick: {tick['verdict']['overall']}")
    return scenarios


def _scenario(rows: Mapping[str, dict[str, Any]], argus_label: str,
              dropped: list[str]) -> dict[str, Any]:
    argus_vols = rows[argus_label]["oos_vol_bps"]
    comps = {lab: compare(argus_vols, row["oos_vol_bps"])
             for lab, row in rows.items() if lab != argus_label}
    corrected = holm({lab: c["wilcoxon_p"] for lab, c in comps.items()})
    for lab in comps:
        comps[lab]["holm"] = corrected[lab]
    return {
        "dropped_symbols": dropped,
        "mean_oos_vol_bps": {lab: row["mean_oos_vol_bps"] for lab, row in rows.items()},
        "errors": {lab: row["n_errors"] for lab, row in rows.items()},
        "comparisons": comps,
        "verdict": verdict(comps, corrected),
    }


# --- Monte Carlo with a known truth -----------------------------------------------------------


def run_monte_carlo(real_cov: Sequence[Sequence[float]], real_names: Sequence[str],
                    argus_pick: Candidate, *, sims: int = 120, n_obs: int = TRAIN_BARS,
                    seed: int = 0, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Allocate on a sample drawn from a KNOWN covariance and score against the true optimum.

    Two processes. ``deprado_block``: his own `formTrueMatrix` (4 blocks of 3, within-block
    correlation 0.5, his snippet 7.7 shape at this universe's size), drawn exactly as his
    `simCovMu` draws (`np.random.multivariate_normal`). ``calibrated``: the real selection-period
    sample covariance taken as the truth, drawn Gaussian and Student-t(4) (fat tails, rescaled to
    the same covariance). Scored two ways: RMSE of weights against the true optimum -- de Prado's
    own metric, snippet 7.9 -- and, for long-only allocators, the true variance ``w'S0w`` over the
    true long-only optimum's, minus one (the variance a mis-estimated allocation actually costs).
    """
    import numpy as np

    ch7 = _mlam()
    mc = _mlam("ch2_monte_carlo_experiment")
    np.random.seed(seed)
    mu_b, cov_b = mc.formTrueMatrix(4, 3, 0.5)
    block_cov = np.asarray(cov_b, dtype=float)
    processes = {
        "deprado_block_gaussian": (block_cov, "gaussian"),
        "calibrated_gaussian": (np.asarray(real_cov, dtype=float), "gaussian"),
        "calibrated_student_t4": (np.asarray(real_cov, dtype=float), "t4"),
    }

    allocators: dict[str, Callable[[Columns], Weights]] = {
        "argus_nco[sample]": partial(_ensemble, config=NCOConfig("sample")),
        "argus_minvar[sample]": partial(_argus_minvar, config=NCOConfig("sample")),
        argus_pick.label: argus_pick.fn,
        "riskfolio_nco[hist,single]": partial(_riskfolio_nco, linkage="single"),
        "riskfolio_nco[hist,ward]": partial(_riskfolio_nco, linkage="ward"),
        "riskfolio_minvar[hist]": partial(_riskfolio_minvar, method_cov="hist"),
        "skfolio_nco[default]": partial(_skfolio_nco, cov=None, cv=None),
    }
    out: dict[str, Any] = {"sims": sims, "n_obs": n_obs, "seed": seed, "processes": {}}
    for pname, (true_cov, dist) in processes.items():
        n = true_cov.shape[0]
        names = [f"a{i:02d}" for i in range(n)] if pname.startswith("deprado") else list(real_names)
        w_true_lo = _exact_long_only_minvar(true_cov)
        best_var = float(w_true_lo @ true_cov @ w_true_lo)
        w_true_ls = np.asarray(mc.optPort(true_cov, None)).flatten()
        rng = np.random.default_rng(seed + 1)
        rmse: dict[str, list[float]] = {k: [] for k in allocators}
        excess: dict[str, list[float]] = {k: [] for k in allocators}
        rmse_ls: dict[str, list[float]] = {"markowitz_unconstrained": [], "deprado_nco": []}
        failures: dict[str, int] = dict.fromkeys(allocators, 0)
        for s in range(sims):
            if pname.startswith("deprado"):
                np.random.seed(seed + 1000 + s)
                x = np.random.multivariate_normal(np.asarray(mu_b).flatten(), true_cov,
                                                  size=n_obs)
            elif dist == "gaussian":
                x = rng.multivariate_normal(np.zeros(n), true_cov, size=n_obs)
            else:
                z = rng.multivariate_normal(np.zeros(n), true_cov, size=n_obs)
                g = rng.chisquare(4, size=(n_obs, 1)) / 4.0
                x = z / np.sqrt(g) * math.sqrt((4 - 2) / 4)
            cols = {names[i]: x[:, i].tolist() for i in range(n)}
            for key, fn in allocators.items():
                try:
                    w = fn(cols)
                except Exception:
                    failures[key] += 1
                    continue
                wv = np.array([w[x_] for x_ in names])
                rmse[key].append(float(np.sqrt(np.mean((wv - w_true_lo) ** 2))))
                excess[key].append(float(wv @ true_cov @ wv) / best_var - 1.0)
            emp = np.cov(x, rowvar=False)
            rmse_ls["markowitz_unconstrained"].append(
                float(np.sqrt(np.mean((np.asarray(mc.optPort(emp, None)).flatten()
                                       - w_true_ls) ** 2))))
            np.random.seed(seed + 5000 + s)
            with contextlib.redirect_stdout(io.StringIO()):
                nco = np.asarray(ch7.optPort_nco(emp, None, int(n / 2))).flatten()
            rmse_ls["deprado_nco"].append(float(np.sqrt(np.mean((nco - w_true_ls) ** 2))))
        summary = {
            key: {
                "mean_weight_rmse": fmean(rmse[key]) if rmse[key] else None,
                "mean_true_variance_excess": fmean(excess[key]) if excess[key] else None,
                "median_true_variance_excess": (sorted(excess[key])[len(excess[key]) // 2]
                                                if excess[key] else None),
                "failures": failures[key],
            }
            for key in allocators
        }
        paired: dict[str, Any] = {}
        for key in allocators:
            if key == argus_pick.label:
                continue
            a, b = excess[argus_pick.label], excess[key]
            if len(a) == len(b) and a:
                diffs = [bb - aa for aa, bb in zip(a, b, strict=True)]
                paired[key] = _paired_stats(diffs)
        out["processes"][pname] = {
            "true_long_only_optimum_variance": best_var,
            "long_only": summary,
            "argus_pick_vs_each_on_true_variance_excess": paired,
            "unconstrained_reproduction_of_deprado_snippet_7_9": {
                "markowitz_mean_rmse": fmean(rmse_ls["markowitz_unconstrained"]),
                "nco_mean_rmse": fmean(rmse_ls["deprado_nco"]),
                "nco_reduces_rmse": fmean(rmse_ls["deprado_nco"])
                < fmean(rmse_ls["markowitz_unconstrained"]),
                "paired": _paired_stats([
                    m - d for m, d in zip(rmse_ls["markowitz_unconstrained"],
                                          rmse_ls["deprado_nco"], strict=True)]),
            },
        }
        log(f"  monte carlo {pname}: done")
    return out


def _exact_long_only_minvar(cov: Any) -> Any:
    """The true long-only minimum-variance weights of a KNOWN covariance, by an interior-point
    solver (cvxpy + Clarabel) at tight tolerance -- deliberately not ARGUS's own active-set
    `minimum_variance_weights`, because the truth an allocator is scored against must not come from
    one of the allocators being scored."""
    import cvxpy as cp  # type: ignore[import-untyped, unused-ignore]
    import numpy as np

    cvx: Any = cp
    n = cov.shape[0]
    w = cvx.Variable(n)
    problem = cvx.Problem(cvx.Minimize(cvx.quad_form(w, cvx.psd_wrap(cov))),
                          [cvx.sum(w) == 1, w >= 0])
    problem.solve(solver=cvx.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
    out = np.clip(np.asarray(w.value, dtype=float).flatten(), 0.0, None)
    return out / out.sum()


def _paired_stats(diffs: Sequence[float]) -> dict[str, Any]:
    """Differences oriented so positive favours the first-named side (ARGUS's pick, or NCO)."""
    from scipy.stats import wilcoxon  # type: ignore[import-untyped, unused-ignore]

    nonzero = [d for d in diffs if abs(d) > 1e-15]
    wins = sum(1 for d in diffs if d > 1e-15)
    losses = sum(1 for d in diffs if d < -1e-15)
    return {
        "n": len(diffs),
        "mean_difference": fmean(diffs) if diffs else None,
        "favourable": wins,
        "unfavourable": losses,
        "sign_test_p": _sign_p(wins, wins + losses),
        "wilcoxon_p": float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 6 else None,
    }


# --- reproducibility --------------------------------------------------------------------------


def run_reproducibility(columns: Columns, window_origins: Sequence[int],
                        candidates: Sequence[Candidate], labels: Sequence[str],
                        first: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute the decisive rows from the frozen snapshot and require bit-identical per-window
    volatilities -- including the bagged allocator's seeded bootstrap and skfolio's CV folds."""
    by_label = {c.label: c for c in candidates}
    identical: dict[str, bool] = {}
    for label in labels:
        again = evaluate(by_label[label], columns, window_origins)
        identical[label] = again["oos_vol_bps"] == first[label]["oos_vol_bps"]
    return {"recomputed": list(labels), "identical": identical,
            "all_identical": all(identical.values())}


# --- orchestration ----------------------------------------------------------------------------


METHODS_STUDIED: tuple[dict[str, Any], ...] = (
    {"method": "Nested Clustered Optimization (original)",
     "source": "Lopez de Prado, 'A Robust Estimator of the Efficient Frontier' (SSRN 3469961, "
               "2019); Machine Learning for Asset Managers (Cambridge 2020) ch. 7",
     "code_read": "ml-for-asset-managers (Apache-2.0, 5292b3c4) "
                  "ch7_portfolio_construction.py:21-49 optPort_nco, :51-78 allocate_cvo; "
                  "ch4_optimal_clustering.py:29-67 clusterKMeansBase (ONC)",
     "run_here": "walk-forward (deprado_nco[mlam_ch7]) and his own snippet 7.7-7.9 Monte Carlo",
     "taken": "the Monte Carlo design (known truth, weight RMSE) as the ground-truth test",
     "not_taken": "the closed-form unconstrained intra/inter solve: it shorts, and the desk's "
                  "feasible set is long-only"},
    {"method": "NCO as shipped by Riskfolio-Lib 7.3.0",
     "source": "HCPortfolio.py:341-407 _hierarchical_clustering, :684-757 _intra/_inter_weights, "
               ":259-338 _opt_w; ParamsEstimation.py:147-275 covar_matrix",
     "run_here": "shipped default and a 26-configuration grid (8 covariance estimators x 3 "
                 "linkages, stdsil k-rule, spearman codependence) plus Classic MinRisk/MV",
     "taken": "reproduced to 1.3e-05 earlier (desk/allocation.py:nco_weights)"},
    {"method": "NCO as shipped by skfolio 1.3.1",
     "source": "optimization/cluster/_nco.py:304-480 (CV-fitted outer estimator), "
               "cluster/_hierarchical.py:169-207, utils/stats.py:570-640 (k rule)",
     "run_here": "default (KFold(5) outer), cv='ignore', and LedoitWolf / Denoise / EW(240) / "
                 "Gerber covariance priors",
     "not_taken": "the CV-fitted outer step as an ARGUS component: it is measured here as a rival "
                  "configuration instead, so its value is a number rather than an assumption"},
    {"method": "Linear shrinkage toward constant correlation",
     "source": "Ledoit & Wolf (2004) 'Honey, I Shrunk the Sample Covariance Matrix', JPM 30(4); "
               "PyPortfolioOpt risk_models.py:602-655 (MIT, a6638d2e)",
     "run_here": "argus_nco[lw_cc], argus_minvar[lw_cc]; port verified to 1e-12 vs PyPortfolioOpt",
     "note": "neither rival ships this target natively (both wrap scikit-learn's scaled-identity "
             "LedoitWolf/OAS, which are run here as riskfolio 'ledoit'/'oas' and skfolio "
             "LedoitWolf)"},
    {"method": "Exponentially weighted covariance",
     "source": "RiskMetrics (J.P. Morgan 1996); Riskfolio covar_matrix 'ewma1'; skfolio "
               "EWCovariance",
     "run_here": "ARGUS ewma240, Riskfolio ewma1 at its default d=0.94 and at matched d, skfolio "
                 "EWCovariance(half_life=240)"},
    {"method": "Random-matrix denoising",
     "source": "MLAM ch. 2 (constant-residual-eigenvalue); Riskfolio 'fixed'/'spectral'; skfolio "
               "DenoiseCovariance",
     "run_here": "as rival configurations (Riskfolio fixed/spectral, skfolio denoise)"},
    {"method": "Resampled / bagged allocation",
     "source": "Michaud (1998) Efficient Asset Management; Politis & Romano (1994) stationary "
               "bootstrap, JASA 89(428)",
     "run_here": "argus_nco[*,bag32]; argus minimum variance bagged as an ablation arm"},
    {"method": "Nonlinear shrinkage",
     "source": "Ledoit & Wolf (2020) 'Analytical nonlinear shrinkage of large-dimensional "
               "covariance matrices', Annals of Statistics 48(5)",
     "run_here": "NOT RUN. Its advantage over linear shrinkage is a large-dimensional result "
                 "(concentration N/T bounded away from zero); here N/T = 12/480 = 0.025. Stated "
                 "as the reason, not measured -- the one method in this list that is read and "
                 "argued about rather than run."},
)
"""What was read, where, and whether it was run. The register's `best_method_studied` attestation
points here: every entry names a source a reader can open, and every one but the last was run."""


SCOPE_STATEMENT = (
    "Every rival is run unmodified from its real distribution: Riskfolio-Lib 7.3.0 NCO (shipped "
    "default and a 26-configuration grid over the covariance estimators, linkages, cluster-count "
    "rule and codependence it exposes, plus its long-only minimum variance), skfolio 1.3.1 "
    "NestedClustersOptimization (default with cross-validated outer weights, the no-CV variant, "
    "and four covariance estimators), and Lopez de Prado's own chapter-7 NCO code (ONC k-means, "
    "closed-form unconstrained solves, so it may short and is never ranked with the long-only "
    "allocators). Selection and holdout are disjoint in time; every family is tuned on the "
    "selection half only and compared on the holdout. NOT CLAIMED: that realised volatility is "
    "the only thing a desk should minimise -- the minimum-variance allocators here reach their "
    "numbers partly through an instrument held against its own inverse, which is why the "
    "adversarial section re-runs the decisive comparison without the leveraged and inverse ETFs. "
    "NOT CLAIMED: that a TIE is a win, or that a win on this universe and this history generalises "
    "beyond them."
)


def stage_failures(report: Mapping[str, Any]) -> dict[str, Any]:
    """Everything the report says did not run, gathered in one place.

    `METHODS_STUDIED` and `SCOPE_STATEMENT` describe what the protocol runs; this is what the run
    actually managed. A stage that raised and a candidate that errored on any holdout window are
    both listed, so a reader never has to find a failure by scanning forty rows -- the first full
    run lost de Prado's rival and the whole Monte Carlo stage to a missing plotting import, and
    the only trace was an ``error`` key several thousand lines into the file."""
    out: dict[str, Any] = {}
    for name in ("holdout_dense", "ablation_nco_mechanism", "ablation_pick", "adversarial",
                 "monte_carlo", "reproducibility"):
        section = report.get(name)
        if isinstance(section, Mapping) and "error" in section:
            out[name] = section["error"]
    holdout = report.get("holdout")
    if isinstance(holdout, Mapping):
        for label, row in holdout.get("results", {}).items():
            if row.get("n_errors"):
                out[f"holdout:{label}"] = {"windows_failed": row["n_errors"],
                                           "first_error": (row.get("errors") or [None])[0]}
    return out


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI, hours of compute
    import argparse

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="NCO bake-off")
    parser.add_argument("--phase", choices=("selection", "all"), default="all")
    parser.add_argument("--mc-sims", type=int, default=120)
    parser.add_argument("--reuse-selection", action="store_true",
                        help="reuse selection rows from a checkpoint on the same snapshot; off by "
                             "default, because a row reused across a code change is stale")
    args = parser.parse_args(argv)

    stamps, columns, snap_digest = load_snapshot()
    all_origins = origins(len(stamps))
    selection_origins, holdout_origins = split(all_origins)
    candidates = [*argus_candidates(), *riskfolio_candidates(), *skfolio_candidates(),
                  *reference_candidates()]
    print(f"snapshot {snap_digest[:12]}: {len(columns)} symbols x {len(stamps)} bars, "
          f"{len(selection_origins)} selection + {len(holdout_origins)} holdout windows")
    started = time.perf_counter()
    reuse: dict[str, Any] = {}
    if args.reuse_selection and CHECKPOINT_PATH.exists():
        prior = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if prior.get("snapshot", {}).get("digest") == snap_digest:
            reuse = dict(prior["selection"]["results"])
    selection = run_selection(columns, selection_origins,
                              [c for c in candidates if c.family in ("argus", "riskfolio",
                                                                     "skfolio")], reuse=reuse)
    selection["n_rows_reused"] = sum(1 for r in selection["results"].values() if r.get("reused"))
    report: dict[str, Any] = {
        "snapshot": {"digest": snap_digest, "n_symbols": len(columns), "n_bars": len(stamps),
                     "first": stamps[0], "last": stamps[-1],
                     "path": "data/allocation_returns_snapshot.json"},
        "protocol": {"train_bars": TRAIN_BARS, "test_bars": TEST_BARS, "step_bars": STEP_BARS,
                     "n_selection_windows": len(selection_origins),
                     "n_holdout_windows": len(holdout_origins),
                     "selection_period": [stamps[selection_origins[0]],
                                          stamps[selection_origins[-1] + TRAIN_BARS + TEST_BARS
                                                 - 1]],
                     "holdout_period": [stamps[holdout_origins[0] + TRAIN_BARS],
                                        stamps[holdout_origins[-1] + TRAIN_BARS + TEST_BARS - 1]],
                     "taker_bps": TAKER_BPS,
                     "selection_rule": "lowest mean realised OOS vol on the selection windows, "
                                       "per family, among configurations with zero errors",
                     "verdict_rule": (verdict.__doc__ or "").strip()},
        "methods_studied": METHODS_STUDIED,
        "plot_only_imports_stubbed": _stub_plot_only_imports(),
        "selection": selection,
    }
    write(CHECKPOINT_PATH, report)
    if args.phase == "selection":
        print(f"selection picks: {selection['picks']} ({time.perf_counter() - started:.0f}s)")
        return 0

    picks = selection["picks"]
    by_label = {c.label: c for c in candidates}
    pick = by_label[picks["argus"]]
    holdout = run_holdout(columns, holdout_origins, candidates, picks)
    report["holdout"] = holdout
    decisive = [picks["argus"], *holdout["decisive_rivals"]]

    def stage(name: str, fn: Callable[[], Any]) -> None:
        """A later stage that fails is recorded as a failure, not allowed to lose the holdout."""
        try:
            report[name] = fn()
        except Exception as exc:
            report[name] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            print(f"  stage {name} FAILED: {exc}")

    stage("holdout_dense", lambda: run_holdout_dense(
        columns, holdout_origins[0] + TRAIN_BARS, candidates, decisive))
    nco_rows = [selection["results"][c.label] for c in candidates
                if c.family == "argus" and c.config.get("clustering") == "nco"]
    best_nco = by_label[_pick(nco_rows)]
    best_nco_cfg = NCOConfig(str(best_nco.config["estimator"]), best_nco.config["halflife"],
                             int(best_nco.config["n_boot"]), float(best_nco.config["block"]),
                             int(best_nco.config["seed"]))
    stage("ablation_nco_mechanism", lambda: {
        "base": best_nco.label,
        "arms": run_ablation(columns, holdout_origins, ablation_candidates(best_nco_cfg),
                             holdout["results"][best_nco.label]["oos_vol_bps"]),
    })
    if pick.config.get("clustering") == "none":
        pick_cfg = NCOConfig(str(pick.config["estimator"]), pick.config["halflife"])
        stage("ablation_pick", lambda: {
            "base": pick.label,
            "arms": run_ablation(columns, holdout_origins, minvar_ablation_candidates(pick_cfg),
                                 holdout["results"][pick.label]["oos_vol_bps"]),
        })
    stage("adversarial", lambda: run_adversarial(columns, holdout_origins, candidates, decisive))
    names = sorted(columns)
    sel_fit = {n: list(columns[n][: selection_origins[-1] + TRAIN_BARS + TEST_BARS])
               for n in names}
    stage("monte_carlo", lambda: run_monte_carlo(
        sample_covariance(_rows(sel_fit)), names, pick, sims=args.mc_sims))
    stage("reproducibility", lambda: run_reproducibility(
        columns, holdout_origins, candidates, decisive, holdout["results"]))
    report["costs"] = {
        label: {"seconds_per_allocation": row["seconds_per_allocation"],
                "turnover_per_rebalance": row["turnover_per_rebalance"],
                "cost_bps_per_rebalance": row["cost_bps_per_rebalance"]}
        for label, row in holdout["results"].items()
    }
    report["stage_failures"] = stage_failures(report)
    report["scope_statement"] = SCOPE_STATEMENT
    report["elapsed_seconds"] = time.perf_counter() - started
    write(REPORT_PATH, report)
    if report["stage_failures"]:
        print(f"NOTE -- not everything ran: {json.dumps(report['stage_failures'])}")
    print(json.dumps({"picks": picks, "verdict": holdout["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "ADVERSARIAL_UNIVERSES",
    "METHODS_STUDIED",
    "REPORT_PATH",
    "Candidate",
    "argus_candidates",
    "candidate_from_config",
    "compare",
    "decisive_rivals",
    "evaluate",
    "holm",
    "minvar_ablation_candidates",
    "origins",
    "realised_vol",
    "reference_candidates",
    "riskfolio_candidates",
    "run_ablation",
    "run_adversarial",
    "run_holdout",
    "run_holdout_dense",
    "run_monte_carlo",
    "run_reproducibility",
    "run_selection",
    "skfolio_candidates",
    "split",
    "stage_failures",
    "verdict",
]
