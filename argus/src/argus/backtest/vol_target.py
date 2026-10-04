"""Volatility-targeted sizing: a walk-forward GARCH(1,1) forecast turned into a position size.

Build-list 1.6. The reference is ``milesdeutscher/garchmethod`` (MIT, cloned at
``research/repos-t2/garchmethod``): ``scripts/garch_forecast.py`` fits GARCH(1,1) walk-forward —
the first forecast after 500 days, parameters re-estimated every 21 days on an expanding window,
the recursion rolled forward between refits, a forecast made at the close of day *t* for day *t+1*
— and ``scripts/vol_target.py`` sizes ``target / forecast``, clipped to ``[0.25, 2.0]``. Both are
taken: the windows, the refit cadence, the timing and the caps are the reference's.

**What is different, and why.**

* **Pure Python.** The hosted console ships no numpy, scipy or ``arch``, so the fit is written
  here: the reference's Student-t GARCH(1,1) likelihood (``math.lgamma``), maximised by
  Nelder-Mead over ``(omega, alpha, beta, nu)``, warm-started from the last refit. A Gaussian
  quasi-likelihood was tried first and rejected: it matched ``arch``'s own normal fit, but on
  bitcoin (tails of about 2.9 degrees of freedom) it put persistence at 0.79 against the
  t-likelihood's 0.92, and its forecasts correlated 0.80 with the reference. The Student-t fit
  lands within half a log-likelihood point of ``arch`` and its forecasts correlate 0.99 with the
  reference's (``eval/vol_target_comparison.py``, ``data/vol_target_comparison.json``).
* **The state at a refit.** The reference starts the roll from the fitted model's conditional
  variance of day *t-1* and feeds it day *t*'s residual (`garch_forecast.py:96-104`); here the
  variance of day *t* is carried through the recursion, so a refit does not shift the series a
  day.
* **The comparison pays fees.** The reference's ``compare.py`` multiplies returns by the size and
  charges nothing; a size that changes every day trades every day, so here both arms run through
  ``backtest/engine.py``, which charges Bitget's taker fee on every change of weight.
* **The target.** The reference aims at 15% a year, which on bitcoin is about a third of its
  volatility and so mostly measures holding less. Here the default target is the realised
  volatility of the training window, known before the first sized day, so the targeted arm holds
  about as much on average and the comparison measures *when* it holds it.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

MIN_TRAIN: Final = 500
REFIT_EVERY: Final = 21
MAX_SIZE: Final = 2.0
MIN_SIZE: Final = 0.25


@dataclass(frozen=True)
class Fit:
    mu: float
    omega: float
    alpha: float
    beta: float
    nu: float


def _neg_loglik(eps: Sequence[float], params: Sequence[float]) -> float:
    """Student-t GARCH(1,1) negative log-likelihood (``arch``'s ``StudentsT`` with ``GARCH(1,1)``,
    the reference's choice), parameters ``(omega, alpha, beta, nu)``."""
    omega, alpha, beta, nu = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999 or not 2.05 < nu < 200:
        return math.inf
    s2 = sum(e * e for e in eps) / len(eps)  # the backcast: arch starts from the sample variance
    const = (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(math.pi * (nu - 2)))
    half = (nu + 1) / 2
    total = 0.0
    for e in eps:
        total += const - 0.5 * math.log(s2) - half * math.log1p(e * e / ((nu - 2) * s2))
        s2 = omega + alpha * e * e + beta * s2
    return -total


def _nelder_mead(func: Callable[[Sequence[float]], float], start: Sequence[float],
                 scale: Sequence[float], steps: int = 300, tol: float = 1e-7) -> list[float]:
    """Nelder-Mead over n parameters (reflection 1, expansion 2, contraction 0.5, shrink 0.5)."""
    n = len(start)
    simplex = [list(start)]
    for k in range(n):
        point = list(start)
        point[k] += scale[k]
        simplex.append(point)
    values = [func(p) for p in simplex]
    for _ in range(steps):
        order = sorted(range(n + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        if abs(values[-1] - values[0]) < tol * (abs(values[0]) + tol):
            break
        centre = [sum(p[k] for p in simplex[:-1]) / n for k in range(n)]
        worst = simplex[-1]
        refl = [c + (c - w) for c, w in zip(centre, worst, strict=True)]
        fr = func(refl)
        if fr < values[0]:
            exp = [c + 2 * (c - w) for c, w in zip(centre, worst, strict=True)]
            fe = func(exp)
            simplex[-1], values[-1] = (exp, fe) if fe < fr else (refl, fr)
        elif fr < values[-2]:
            simplex[-1], values[-1] = refl, fr
        else:
            con = [c + 0.5 * (w - c) for c, w in zip(centre, worst, strict=True)]
            fc = func(con)
            if fc < values[-1]:
                simplex[-1], values[-1] = con, fc
            else:
                anchor = simplex[0]
                simplex = [anchor, *([a + 0.5 * (p_k - a) for a, p_k in zip(anchor, p, strict=True)]
                                     for p in simplex[1:])]
                values = [values[0], *(func(p) for p in simplex[1:])]
    best = min(range(n + 1), key=lambda i: values[i])
    return simplex[best]


def fit(returns: Sequence[float], start: Sequence[float] | None = None) -> Fit:
    """Student-t GARCH(1,1) by maximum likelihood on percent returns, as the reference fits it
    with ``arch`` (constant mean, ``dist="t"``). ``start`` warm-starts from the last refit."""
    mu = sum(returns) / len(returns)
    eps = [r - mu for r in returns]
    var = sum(e * e for e in eps) / len(eps)
    guess = list(start) if start is not None else [0.05 * var, 0.08, 0.90, 5.0]
    scale = [max(1e-4, 0.5 * guess[0]), 0.03, 0.03, 1.5]
    omega, alpha, beta, nu = _nelder_mead(lambda p: _neg_loglik(eps, p), guess, scale)
    return Fit(mu=mu, omega=omega, alpha=alpha, beta=beta, nu=nu)


def forecasts(closes: Sequence[float], *, min_train: int = MIN_TRAIN,
              refit_every: int = REFIT_EVERY) -> list[float | None]:
    """The variance of the next day's percent return, forecast at each close from closes up to
    it; ``forecasts[i]`` is made at close ``i`` for the return from ``i`` to ``i+1``. ``None``
    before ``min_train`` returns exist."""
    returns = [100 * (b / a - 1) for a, b in itertools.pairwise(closes)]
    out: list[float | None] = [None] * len(closes)
    params: Fit | None = None
    s2 = 0.0
    start: list[float] | None = None
    for t in range(min_train, len(returns) + 1):
        # returns[t-1] is the return into close t; at close t the data are returns[:t]
        if params is None or (t - min_train) % refit_every == 0:
            params = fit(returns[:t], start)
            start = [params.omega, params.alpha, params.beta, params.nu]
            # the variance of day t, carried through the recursion from the sample variance
            mean = sum(returns[:t]) / t
            s2 = sum((r - mean) ** 2 for r in returns[:t]) / t
            for r in returns[:t]:
                e = r - params.mu
                s2 = params.omega + params.alpha * e * e + params.beta * s2
        else:
            e = returns[t - 1] - params.mu
            s2 = params.omega + params.alpha * e * e + params.beta * s2
        out[t] = s2
    return out


def size(forecast_vol: float | None, target_vol: float, *, max_size: float = MAX_SIZE,
         min_size: float = MIN_SIZE) -> float:
    """``target / forecast``, clipped (``garchmethod scripts/vol_target.py:27-32``); the minimum
    when there is no forecast."""
    if forecast_vol is None or forecast_vol <= 0 or math.isnan(forecast_vol):
        return min_size
    return min(max_size, max(min_size, target_vol / forecast_vol))


def sizes(closes: Sequence[float], periods_per_year: int,
          target_vol: float | None = None) -> tuple[list[float | None], float]:
    """Each close's size multiplier (``None`` before the first forecast) and the annualised
    target used: unless one is given, the realised volatility of the training window (the first
    ``MIN_TRAIN`` returns), which is known before the first sized day. A median of the forecasts
    themselves would hold about as much on average but is read from days not yet seen."""
    variance = forecasts(closes)
    annual = [None if v is None else math.sqrt(v * periods_per_year) for v in variance]
    if all(a is None for a in annual):
        return [None] * len(closes), 0.0
    if target_vol is None:
        train = [100 * (b / a - 1) for a, b in itertools.pairwise(closes[:MIN_TRAIN + 1])]
        mean = sum(train) / len(train)
        target_vol = math.sqrt(sum((r - mean) ** 2 for r in train) / len(train)
                               * periods_per_year)
    return [None if a is None else size(a, target_vol) for a in annual], target_vol
