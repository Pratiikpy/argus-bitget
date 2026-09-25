"""Nested Clustered Optimization with the estimation step taken seriously.

`desk/allocation.py:nco_weights` reproduces Riskfolio-Lib's real NCO to 1.3e-05 and ties it on
walk-forward realised volatility (`data/allocation_comparison.json`). A tie is where reproduction
ends; beating the specialist needs something the specialist does not do. This module is the attempt,
and `eval/nco_bakeoff.py` is where it is judged, on a held-out half of the history that none of the
choices below were tuned on.

**Where NCO leaves error on the table.** NCO fixes the *optimiser's* instability -- López de Prado's
point is that a minimum-variance solve on a covariance with strong clusters amplifies noise, and
solving inside clusters first contains it. It does nothing about the *input's* noise: every weight
is still a deterministic function of one sample covariance from one window. Two classic answers to
that, and one that NCO's own literature points at without either library shipping it:

1. **Shrink the input** -- Ledoit & Wolf (2004), "Honey, I shrunk the sample covariance matrix",
   J. Portfolio Management 30(4), toward the constant-correlation target. Adapted, with the same
   algebra, from PyPortfolioOpt (MIT, Copyright (c) 2018 Robert Andrew Martin),
   `pypfopt/risk_models.py:602-655` (`_ledoit_wolf_constant_correlation`, clone a6638d2e). Neither
   rival's NCO offers this target: Riskfolio's ``method_cov="ledoit"`` and skfolio's
   ``LedoitWolf`` both wrap scikit-learn's scaled-identity target, which on twelve assets
   correlated at 0.9 shrinks toward a structure the data flatly contradicts.
2. **Weight recent history more** -- an exponentially weighted covariance. Riskfolio does offer
   this (``method_cov="ewma1"``, `ParamsEstimation.py` ``covar_matrix``, decay ``d``), so on its own
   it cannot be an advantage over the specialist, only a configuration both sides can reach; it is
   here so the bake-off can test that rather than assume it.
3. **Average over resamples of the estimation window** -- bootstrap aggregation of the whole NCO
   map, in the spirit of Michaud's resampled efficiency (Michaud 1998, "Efficient Asset
   Management") applied to NCO rather than to Markowitz. A stationary bootstrap (Politis & Romano
   1994, J. American Statistical Association 89(428)) keeps the serial dependence of hourly bars
   inside each resample. Neither Riskfolio's ``HCPortfolio`` nor skfolio's
   ``NestedClustersOptimization`` bags; skfolio's cross-validated outer step (`_nco.py:407-478`)
   is the nearest idea and is a different one -- it de-biases the *outer* weights, while bagging
   averages the *whole* map, clustering included.

**What is not claimed here.** That any of these helps. Each is a mechanism with a literature behind
it and a failure mode: shrinkage toward the wrong target adds bias, a short half-life adds noise,
bagging a clustering step can blur a real cluster boundary. Which, if any, lowers realised
out-of-sample volatility on these instruments is an empirical question with an answer in
`data/nco_bakeoff.json`, and that file -- not this docstring -- is what the capability register
reads.

Pure Python, like the rest of `desk/`: the shipped import graph stays pydantic + python-dateutil.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from argus.desk.allocation import AllocationError, nco_weights

MIN_ROWS = 60
"""Fewest aligned observations before any estimator here is attempted (matches `allocation.py`)."""


# --- covariance estimators --------------------------------------------------------------------


def _as_matrix(columns: Mapping[str, Sequence[float]]) -> tuple[list[str], list[list[float]]]:
    names = sorted(columns)
    if not names:
        raise AllocationError("no columns")
    size = len(columns[names[0]])
    if any(len(columns[n]) != size for n in names):
        raise AllocationError("columns are not aligned")
    if size < MIN_ROWS:
        raise AllocationError(f"{size} observation(s) is below the {MIN_ROWS} needed")
    rows = [[float(columns[n][t]) for n in names] for t in range(size)]
    return names, rows


def sample_covariance(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    """Unbiased (ddof=1) sample covariance -- pandas' ``DataFrame.cov`` and Riskfolio's ``hist``."""
    t = len(rows)
    n = len(rows[0])
    means = [sum(r[j] for r in rows) / t for j in range(n)]
    centred = [[r[j] - means[j] for j in range(n)] for r in rows]
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            value = sum(c[i] * c[j] for c in centred) / (t - 1)
            out[i][j] = out[j][i] = value
    return out


def ewma_covariance(rows: Sequence[Sequence[float]], halflife: float) -> list[list[float]]:
    """Exponentially weighted covariance, newest row heaviest, weights summing to one.

    Weight on row ``t`` is proportional to ``0.5 ** ((T - 1 - t) / halflife)``. The mean is the same
    weighted mean, and the sum is divided by ``1 - sum(w^2)``, the reliability-weights correction
    that makes the estimator unbiased for i.i.d. rows -- it reduces to ``ddof=1`` when every weight
    is equal, so an infinite half-life is exactly :func:`sample_covariance`.
    """
    if halflife <= 0:
        raise AllocationError("halflife must be positive")
    t = len(rows)
    n = len(rows[0])
    raw = [0.5 ** ((t - 1 - k) / halflife) for k in range(t)]
    total = sum(raw)
    w = [x / total for x in raw]
    correction = 1.0 - sum(x * x for x in w)
    means = [sum(w[k] * rows[k][j] for k in range(t)) for j in range(n)]
    centred = [[rows[k][j] - means[j] for j in range(n)] for k in range(t)]
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            value = sum(w[k] * centred[k][i] * centred[k][j] for k in range(t)) / correction
            out[i][j] = out[j][i] = value
    return out


def ledoit_wolf_constant_correlation(
    rows: Sequence[Sequence[float]],
) -> tuple[list[list[float]], float]:
    """Ledoit-Wolf (2004) shrinkage toward the constant-correlation target: ``(matrix, delta)``.

    Same algebra, term for term, as PyPortfolioOpt's `_ledoit_wolf_constant_correlation`
    (`pypfopt/risk_models.py:602-655`, MIT): ``S`` is the ddof=1 sample covariance, the target
    ``F`` keeps each variance and replaces every correlation by the average one ``r_bar``, and the
    intensity is ``delta = clip(kappa / T, 0, 1)`` with ``kappa = (pi - rho) / gamma`` estimated
    from the demeaned rows. PyPortfolioOpt mixes a ddof=1 ``S`` with ``/T`` moment estimates inside
    ``pi`` and ``theta``; that mix is reproduced rather than tidied, because the point of the port
    is to return PyPortfolioOpt's number (``tests/test_nco_ensemble.py`` checks it against the
    cloned library to 1e-12).
    """
    t = len(rows)
    n = len(rows[0])
    s = sample_covariance(rows)
    means = [sum(r[j] for r in rows) / t for j in range(n)]
    xm = [[r[j] - means[j] for j in range(n)] for r in rows]
    std = [math.sqrt(s[i][i]) for i in range(n)]
    if any(v <= 0 for v in std):
        raise AllocationError("a zero-variance column has no correlation to shrink toward")
    r_bar = (sum(s[i][j] / (std[i] * std[j]) for i in range(n) for j in range(n)) - n) / (
        n * (n - 1)
    )
    f = [[s[i][i] if i == j else r_bar * std[i] * std[j] for j in range(n)] for i in range(n)]

    # help_ = Xm'Xm / T (the /T second moment), y = Xm**2
    help_ = [[sum(r[i] * r[j] for r in xm) / t for j in range(n)] for i in range(n)]
    y_cross = [[sum((r[i] ** 2) * (r[j] ** 2) for r in xm) / t for j in range(n)] for i in range(n)]
    pi_mat = [
        [y_cross[i][j] - 2.0 * help_[i][j] * s[i][j] + s[i][j] ** 2 for j in range(n)]
        for i in range(n)
    ]
    pi_hat = sum(sum(row) for row in pi_mat)

    term1 = [[sum((r[i] ** 3) * r[j] for r in xm) / t for j in range(n)] for i in range(n)]
    theta = [
        [
            0.0 if i == j else (
                term1[i][j] - help_[i][i] * s[i][j] - help_[i][j] * s[i][i] + s[i][i] * s[i][j]
            )
            for j in range(n)
        ]
        for i in range(n)
    ]
    rho_hat = sum(pi_mat[i][i] for i in range(n)) + r_bar * sum(
        (std[j] / std[i]) * theta[i][j] for i in range(n) for j in range(n)
    )
    gamma_hat = sum((s[i][j] - f[i][j]) ** 2 for i in range(n) for j in range(n))
    if gamma_hat <= 0:
        return s, 0.0
    kappa = (pi_hat - rho_hat) / gamma_hat
    delta = max(0.0, min(1.0, kappa / t))
    shrunk = [[delta * f[i][j] + (1.0 - delta) * s[i][j] for j in range(n)] for i in range(n)]
    return shrunk, delta


# --- resampling --------------------------------------------------------------------------------


def stationary_bootstrap_indices(size: int, mean_block: float, rng: random.Random) -> list[int]:
    """Politis & Romano (1994): blocks of geometric length (mean ``mean_block``), wrapping around.

    Each position either continues the current block (probability ``1 - 1/mean_block``) or jumps to
    a fresh uniform start. Wrapping makes every row equally likely to be drawn, which a plain
    moving-block bootstrap does not (rows near the ends are under-sampled there).
    """
    if mean_block < 1:
        raise AllocationError("mean block length must be at least 1")
    p = 1.0 / mean_block
    out: list[int] = []
    current = rng.randrange(size)
    for _ in range(size):
        out.append(current)
        current = rng.randrange(size) if rng.random() < p else (current + 1) % size
    return out


# --- the allocator -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NCOConfig:
    """One point in the design space `eval/nco_bakeoff.py` searches, stated explicitly.

    ``estimator``: ``"sample"`` (Riskfolio's ``hist``), ``"ewma"`` (needs ``halflife``) or
    ``"lw_cc"`` (Ledoit-Wolf constant correlation). ``n_boot = 0`` means no bagging; otherwise
    that many stationary-bootstrap resamples of mean block length ``block`` are each run through
    the estimator and NCO, and the weights averaged. ``seed`` makes the resampling reproducible.
    """

    estimator: str = "sample"
    halflife: float | None = None
    n_boot: int = 0
    block: float = 24.0
    seed: int = 20260925

    @property
    def label(self) -> str:
        est = self.estimator if self.halflife is None else f"{self.estimator}{self.halflife:g}"
        return f"argus_nco[{est}{'' if not self.n_boot else f',bag{self.n_boot}'}]"


def estimate(rows: Sequence[Sequence[float]], config: NCOConfig) -> list[list[float]]:
    if config.estimator == "sample":
        return sample_covariance(rows)
    if config.estimator == "ewma":
        if config.halflife is None:
            raise AllocationError("the ewma estimator needs a halflife")
        return ewma_covariance(rows, config.halflife)
    if config.estimator == "lw_cc":
        return ledoit_wolf_constant_correlation(rows)[0]
    raise AllocationError(f"unknown estimator {config.estimator!r}")


def ensemble_nco_weights(
    columns: Mapping[str, Sequence[float]], config: NCOConfig | None = None,
) -> dict[str, float]:
    """NCO on the configured covariance estimate, optionally bagged over bootstrap resamples.

    A resample whose covariance NCO refuses (a singular cluster sub-matrix, which a bootstrap can
    produce by repeating rows) is skipped and counted rather than allowed to sink the whole
    estimate; if more than half are refused the call raises, because an average over the few
    survivors of a degenerate input is not an estimate of anything.
    """
    config = config if config is not None else NCOConfig()
    names, rows = _as_matrix(columns)
    if not config.n_boot:
        return nco_weights(names, estimate(rows, config))
    rng = random.Random(config.seed)
    total = dict.fromkeys(names, 0.0)
    used = 0
    for _ in range(config.n_boot):
        pick = stationary_bootstrap_indices(len(rows), config.block, rng)
        sample = [rows[k] for k in pick]
        try:
            weights = nco_weights(names, estimate(sample, config))
        except AllocationError:
            continue
        used += 1
        for name in names:
            total[name] += weights[name]
    if used * 2 < config.n_boot:
        raise AllocationError(
            f"only {used} of {config.n_boot} bootstrap resamples produced an allocation"
        )
    return {name: total[name] / used for name in names}


__all__ = [
    "MIN_ROWS",
    "NCOConfig",
    "ensemble_nco_weights",
    "estimate",
    "ewma_covariance",
    "ledoit_wolf_constant_correlation",
    "sample_covariance",
    "stationary_bootstrap_indices",
]
