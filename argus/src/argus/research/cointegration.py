"""Cointegration, mean-reversion speed, and whether a pair is tradeable after costs.

Track 1's *Arbitrage* sub-theme was half-covered: `research/arbitrage_study.py` measures the
**basis** between an rToken and its underlying (median 1.12bps, only 3.6% of hours clearing the
12bps round trip) and `desk/diversification.py` measures correlation. Neither asks the other half of
the question — whether two instruments share a *stochastic trend*, so that the spread between them
is stationary and its deviations are a thing you can trade rather than a random walk you can only
ride.

Everything here is reimplemented in pure Python against a read of the reference, not from memory.
The reference is statsmodels 0.14.6, installed on this machine at
``statsmodels/tsa/stattools.py`` of an installed statsmodels. ARGUS takes no numpy dependency,
so the arithmetic is ours; the *procedure* and the *constants* are theirs, cited line by line.

**What was copied, with citation**

* The ADF regression, exactly as ``statsmodels/tsa/stattools.py:304-357`` builds it: difference the
  series, lag the differences, then *overwrite the first column with the level* so the t-statistic
  of that column is the test statistic.
* The default lag ceiling ``ceil(12 * (nobs/100)**0.25)`` (``stattools.py:289``) and its cap at
  ``nobs//2 - ntrend - 1`` (``stattools.py:299``).
* AIC lag selection over a **fixed** sample, then a refit at the chosen lag over the *longer* sample
  that lag allows (``stattools.py:326-346``). Comparing information criteria across different sample
  sizes would be meaningless, and refitting on more data afterwards is free accuracy; statsmodels
  does both and the order matters.
* MacKinnon's 2010 critical-value and p-value coefficient tables, transcribed from
  ``statsmodels/tsa/adfvalues.py`` (tables at lines 52-59, 97-104, 276-319) and reproduced below
  with their line numbers.
* The Engle-Granger procedure of ``stattools.py:1702`` (``coint``): regress y0 on [y1, 1], run
  the ADF on the residuals with **no constant** (``regression="n"``), and take critical values
  from the ``N=2`` table — not the ordinary ADF table, because the residuals were fitted rather
  than observed, and the statistic's distribution depends on how many series were cointegrated.
* Even statsmodels' ``nobs - 1`` quirk in that critical-value lookup, which its own comment cannot
  explain ("the -1 is to match egranger in Stata, I do not know why", ``stattools.py:1835``). Copied
  deliberately so our numbers agree with the reference; flagged here so nobody mistakes it for our
  own derivation.

**What was deliberately not copied, and why it matters more than what was**

Every pairs implementation read for this module — statsmodels' own example notebooks, the
``FinceptTerminal`` statistical-arbitrage strategy, the mlfinlab-style labs — fits the hedge ratio
on the whole sample and then reports the spread's z-score using the whole sample's mean and standard
deviation. Both are look-ahead. A spread that is stationary *because you fitted it to be* is not
evidence of anything, and a z-score computed against a mean the market had not yet printed is not a
signal that existed at the time.

So:

1. :func:`pair_test` fits the hedge ratio on a **training slice only** and tests stationarity on
   the held-out spread formed with that frozen ratio. Both the in-sample and the out-of-sample
   statistic are reported, because the gap between them is the finding.
2. :func:`zscores` computes each z from data strictly before it, with a stated minimum history.
3. :func:`scan` corrects for the fact that testing every pair of *n* instruments is *n(n-1)/2*
   hypotheses. At 12 rTokens that is 66 tests, and at a 5% level three of them are expected to look
   cointegrated when nothing is. Benjamini-Hochberg and Bonferroni are both reported.
4. Nothing is called tradeable on statistics alone. A pair trade is **four taker legs** — open two,
   close two — so at Bitget's 6bps per side the round trip is 24bps of gross notional before
   funding. :class:`PairResult` reports the entry z-score required to clear that, and how often the
   spread has historically been that extreme. On this venue that is usually the whole answer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

# --- MacKinnon (2010) tables, transcribed from statsmodels/tsa/adfvalues.py ---------------------
#
# Only the rows this module can actually reach are kept. `N` is the number of series involved: 1 for
# an ordinary unit-root test, 2 for the Engle-Granger residual test with a single regressor. Keeping
# unused rows would be decoration, and a table nobody exercises is a table nobody has checked.

TAU_2010: dict[tuple[str, int], tuple[tuple[float, ...], ...]] = {
    # adfvalues.py:282-296 — N=1, regression "c", rows are 1% / 5% / 10%
    ("c", 1): (
        (-3.43035, -6.5393, -16.786, -79.433),
        (-2.86154, -2.8903, -4.234, -40.040),
        (-2.56677, -1.5384, -2.809, 0.0),
    ),
    # adfvalues.py:282-296 — N=2, the Engle-Granger case
    ("c", 2): (
        (-3.89644, -10.9519, -33.527, 0.0),
        (-3.33613, -6.1101, -6.823, 0.0),
        (-3.04445, -4.2412, -2.720, 0.0),
    ),
    # adfvalues.py, tau_nc_2010 — N=1, no constant: the residual test inside Engle-Granger
    ("n", 1): (
        (-2.56574, -2.2358, -3.627, 0.0),
        (-1.94100, -0.2686, -3.365, 31.223),
        (-1.61682, 0.2656, -2.714, 25.364),
    ),
}
"""Polynomial coefficients in ``1/nobs``: ``crit = c0 + c1/n + c2/n**2 + c3/n**3``.

``adfvalues.py:407-450`` evaluates exactly this polynomial; with ``nobs`` infinite only ``c0``
survives, which is the asymptotic critical value.
"""

# p-value coefficients. adfvalues.py applies `small_scaling = [1, 1, 1e-2]` (line 42) and
# `large_scaling = [1, 1e-1, 1e-1, 1e-2]` (line 87) to the published tables; the scaled values are
# transcribed here so the scaling cannot be applied twice by accident.
TAU_SMALL_P: dict[tuple[str, int], tuple[float, ...]] = {
    ("c", 1): (2.1659, 1.4412, 0.038269),
    ("c", 2): (2.9200, 1.5012, 0.039796),
    ("n", 1): (0.6344, 1.2378, 0.032496),
}
TAU_LARGE_P: dict[tuple[str, int], tuple[float, ...]] = {
    ("c", 1): (1.7339, 0.93202, -0.12745, -0.010368),
    ("c", 2): (2.1945, 0.64695, -0.29198, -0.042377),
    ("n", 1): (0.4797, 0.93557, -0.06999, 0.033066),
}
TAU_STAR: dict[tuple[str, int], float] = {("c", 1): -1.61, ("c", 2): -2.62, ("n", 1): -1.04}
TAU_MIN: dict[tuple[str, int], float] = {("c", 1): -18.83, ("c", 2): -18.86, ("n", 1): -19.04}
TAU_MAX: dict[tuple[str, int], float] = {("c", 1): 2.74, ("c", 2): 0.92, ("n", 1): math.inf}

MIN_OBSERVATIONS = 60
"""Below this a cointegration test is arithmetic, not evidence.

MacKinnon's finite-sample corrections are polynomials in ``1/nobs`` fitted on simulations that do
not extend to tiny samples, and the ADF's power against a slow mean-reverter is poor even at a
thousand points. Sixty is a floor, not a comfort.
"""

TAKER_BPS_PER_LEG = 6.0
"""Bitget taker fee per side, measured. A pair round trip is four of these."""

LEGS_PER_ROUND_TRIP = 4
"""Open two, close two. The single most common omission in published pairs backtests."""


class CointegrationError(ValueError):
    """Raised rather than returning a statistic computed from too little or degenerate data."""


# --- ordinary least squares --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Regression:
    """One OLS fit, with everything the tests below need and nothing they do not."""

    params: tuple[float, ...]
    stderr: tuple[float, ...]
    tvalues: tuple[float, ...]
    resid: tuple[float, ...]
    nobs: int
    rank: int
    ssr: float

    @property
    def llf(self) -> float:
        """Gaussian log-likelihood, the form statsmodels' OLS uses for its information criteria."""
        n = self.nobs
        if n == 0 or self.ssr <= 0:
            return math.inf
        return -0.5 * n * (math.log(2 * math.pi) + math.log(self.ssr / n) + 1.0)

    @property
    def aic(self) -> float:
        """``-2 * llf + 2 * k``. Only differences matter here, but the constant is kept so the
        number can be compared against statsmodels directly during validation."""
        return -2 * self.llf + 2 * self.rank

    @property
    def bic(self) -> float:
        return -2 * self.llf + math.log(self.nobs) * self.rank

    endog: tuple[float, ...] = ()
    """The dependent variable, kept only so :attr:`rsquared` can be computed without refitting."""

    @property
    def rsquared(self) -> float:
        """Centred R-squared. Used for one thing: refusing a cointegration test on two series that
        are collinear to machine precision, the case statsmodels warns about at
        ``stattools.py:1817-1825``."""
        if not self.endog:
            return 0.0
        mean = sum(self.endog) / self.nobs
        total = sum((y - mean) ** 2 for y in self.endog)
        return 1.0 - (self.ssr / total) if total > 0 else 1.0


def _solve(matrix: list[list[float]], rhs: list[float]) -> tuple[list[float], list[list[float]]]:
    """Gauss-Jordan with partial pivoting: returns the solution and the inverse.

    statsmodels reaches for ``pinv`` (QR under the hood), which is more stable on ill-conditioned
    designs. Ours is the textbook method with pivoting, which is adequate here because the design
    matrices are small (a level, a handful of lagged differences, a constant) and because every
    number this module produces is validated against statsmodels on real series in
    ``tests/test_cointegration.py``. Where they disagree, the test fails — that is the guard, rather
    than an assurance in a docstring.
    """
    size = len(matrix)
    work = [row[:] + [1.0 if i == j else 0.0 for j in range(size)] for i, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(work[r][col]))
        if abs(work[pivot][col]) < 1e-12:
            raise CointegrationError(
                "the design matrix is singular: a column is constant or duplicated"
            )
        work[col], work[pivot] = work[pivot], work[col]
        scale = work[col][col]
        work[col] = [v / scale for v in work[col]]
        for row in range(size):
            if row == col:
                continue
            factor = work[row][col]
            if factor:
                work[row] = [v - factor * w for v, w in zip(work[row], work[col], strict=True)]
    inverse = [row[size:] for row in work]
    solution = [sum(inverse[i][j] * rhs[j] for j in range(size)) for i in range(size)]
    return solution, inverse


def ols(endog: Sequence[float], exog: Sequence[Sequence[float]]) -> Regression:
    """Least squares of ``endog`` on the columns of ``exog`` (each inner sequence is a column)."""
    if not exog:
        raise CointegrationError("a regression needs at least one column")
    nobs = len(endog)
    if any(len(col) != nobs for col in exog):
        raise CointegrationError("every column must be the same length as the dependent variable")
    k = len(exog)
    if nobs <= k:
        raise CointegrationError(f"{nobs} observation(s) cannot fit {k} parameter(s)")

    gram = [[sum(a * b for a, b in zip(ci, cj, strict=True)) for cj in exog] for ci in exog]
    moment = [sum(c * y for c, y in zip(col, endog, strict=True)) for col in exog]
    params, inverse = _solve(gram, moment)

    fitted = [sum(params[j] * exog[j][i] for j in range(k)) for i in range(nobs)]
    resid = [y - f for y, f in zip(endog, fitted, strict=True)]
    ssr = sum(r * r for r in resid)
    dof = nobs - k
    sigma2 = ssr / dof if dof > 0 else 0.0
    stderr = [math.sqrt(sigma2 * inverse[j][j]) if inverse[j][j] > 0 else 0.0 for j in range(k)]
    tvalues = [
        (params[j] / stderr[j]) if stderr[j] > 0 else math.inf * (1 if params[j] > 0 else -1)
        for j in range(k)
    ]
    return Regression(
        params=tuple(params), stderr=tuple(stderr), tvalues=tuple(tvalues), resid=tuple(resid),
        nobs=nobs, rank=k, ssr=ssr, endog=tuple(endog),
    )


# --- the augmented Dickey-Fuller test ------------------------------------------------------------


def _normal_cdf(x: float) -> float:
    """``scipy.stats.norm.cdf`` without scipy. ``adfvalues.py:223-268`` calls it on the fitted
    polynomial; ``erf`` gives the same value to machine precision."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mackinnon_p(stat: float, *, regression: str = "c", n: int = 1) -> float:
    """MacKinnon's approximate p-value. Transcribed from ``adfvalues.py:223 (mackinnonp)``.

    The two-branch structure is theirs and is load-bearing: a different polynomial is fitted either
    side of ``TAU_STAR``, and outside ``[TAU_MIN, TAU_MAX]`` the answer is clamped rather than
    extrapolated — a cubic fitted over a finite range will happily return a negative probability
    beyond it.
    """
    key = (regression, n)
    if key not in TAU_STAR:
        raise CointegrationError(f"no MacKinnon table for regression={regression!r}, N={n}")
    if stat > TAU_MAX[key]:
        return 1.0
    if stat < TAU_MIN[key]:
        return 0.0
    coefficients = TAU_SMALL_P[key] if stat <= TAU_STAR[key] else TAU_LARGE_P[key]
    value = sum(c * stat ** i for i, c in enumerate(coefficients))
    return _normal_cdf(value)


def mackinnon_crit(*, regression: str = "c", n: int = 1, nobs: int) -> dict[str, float]:
    """1%, 5% and 10% critical values at this sample size (``adfvalues.py:407-450``)."""
    key = (regression, n)
    if key not in TAU_2010:
        raise CointegrationError(f"no MacKinnon table for regression={regression!r}, N={n}")
    if nobs <= 0:
        raise CointegrationError("nobs must be positive")
    out: dict[str, float] = {}
    for label, coefficients in zip(("1%", "5%", "10%"), TAU_2010[key], strict=True):
        out[label] = sum(c * (1.0 / nobs) ** i for i, c in enumerate(coefficients))
    return out


@dataclass(frozen=True, slots=True)
class ADFResult:
    """The test statistic and what it has to beat."""

    statistic: float
    pvalue: float
    usedlag: int
    nobs: int
    critical_values: dict[str, float]
    regression: str

    @property
    def stationary_at_5pct(self) -> bool:
        return self.statistic < self.critical_values["5%"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "statistic": round(self.statistic, 6),
            "pvalue": round(self.pvalue, 6),
            "usedlag": self.usedlag,
            "nobs": self.nobs,
            "critical_values": {k: round(v, 4) for k, v in self.critical_values.items()},
            "stationary_at_5pct": self.stationary_at_5pct,
        }


def _design(
    diffs: Sequence[float], levels: Sequence[float], lags: int, regression: str,
) -> tuple[list[float], list[list[float]]]:
    """The ADF design matrix, built the way ``stattools.py:304-307`` builds it.

    Column 0 is the **level** (not the contemporaneous difference the lag matrix starts with —
    statsmodels overwrites it, and the t-statistic of that overwritten column is the whole test).
    Columns 1..lags are lagged differences. The trend columns are appended last so column 0 keeps
    its index; statsmodels prepends them during lag selection and appends them for the final fit,
    which is why its ``startlag`` bookkeeping exists and ours does not need to.
    """
    nobs = len(diffs) - lags
    if nobs <= 0:
        raise CointegrationError("not enough observations for the requested lag order")
    endog = list(diffs[-nobs:])
    columns: list[list[float]] = [list(levels[-nobs - 1:-1])]
    for lag in range(1, lags + 1):
        start = len(diffs) - nobs - lag
        columns.append(list(diffs[start:start + nobs]))
    if regression in ("c", "ct"):
        columns.append([1.0] * nobs)
    if regression == "ct":
        columns.append([float(i + 1) for i in range(nobs)])
    return endog, columns


def adf(
    series: Sequence[float], *, regression: str = "c", maxlag: int | None = None,
    autolag: str | None = "aic",
) -> ADFResult:
    """Augmented Dickey-Fuller. Null hypothesis: the series has a unit root.

    ``regression="n"`` drops the constant, which is what the Engle-Granger residual test needs: the
    residuals of a regression that already contained an intercept have mean zero by construction, so
    fitting another one both wastes a degree of freedom and changes the statistic's distribution.
    """
    if regression not in ("c", "n", "ct"):
        raise CointegrationError(f"regression must be 'c', 'n' or 'ct', not {regression!r}")
    values = [float(v) for v in series]
    if len(values) < MIN_OBSERVATIONS:
        raise CointegrationError(
            f"{len(values)} observation(s) is below the {MIN_OBSERVATIONS} this module will test"
        )
    diffs = [b - a for a, b in pairwise(values)]
    ntrend = 0 if regression == "n" else len(regression)

    if maxlag is None:
        maxlag = math.ceil(12.0 * (len(values) / 100.0) ** 0.25)  # stattools.py:289
        maxlag = min(len(values) // 2 - ntrend - 1, maxlag)  # stattools.py:299
    if maxlag < 0:
        raise CointegrationError("the sample is too short for any lag order")

    usedlag = maxlag
    if autolag:
        if autolag.lower() not in ("aic", "bic"):
            raise CointegrationError("autolag must be 'aic', 'bic' or None")
        # Every candidate is fitted on the SAME rows — those the largest lag order allows — so the
        # information criteria are comparable. statsmodels does this at stattools.py:304-326 and it
        # is the step a naive implementation gets wrong: comparing an AIC computed on 1,400 points
        # with one computed on 1,390 selects the sample size, not the lag order.
        best: tuple[float, int] | None = None
        for lag in range(0, maxlag + 1):
            endog, columns = _design(diffs, values, maxlag, regression)
            keep = [columns[0], *columns[1:lag + 1], *columns[maxlag + 1:]]
            fit = ols(endog, keep)
            criterion = fit.aic if autolag.lower() == "aic" else fit.bic
            if best is None or criterion < best[0]:
                best = (criterion, lag)
        assert best is not None
        usedlag = best[1]

    # Refit at the chosen lag over the longer sample that lag permits (stattools.py:344-357).
    endog, columns = _design(diffs, values, usedlag, regression)
    fit = ols(endog, columns)
    statistic = fit.tvalues[0]
    nobs = fit.nobs
    n_series = 1
    return ADFResult(
        statistic=statistic,
        pvalue=mackinnon_p(statistic, regression=regression, n=n_series),
        usedlag=usedlag, nobs=nobs,
        critical_values=mackinnon_crit(regression=regression, n=n_series, nobs=nobs),
        regression=regression,
    )


# --- Engle-Granger -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CointegrationResult:
    """One Engle-Granger test: the fitted relationship and whether its residual is stationary."""

    statistic: float
    pvalue: float
    critical_values: dict[str, float]
    hedge_ratio: float
    intercept: float
    residuals: tuple[float, ...]
    nobs: int

    @property
    def cointegrated_at_5pct(self) -> bool:
        return self.statistic < self.critical_values["5%"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "statistic": round(self.statistic, 6),
            "pvalue": round(self.pvalue, 6),
            "critical_values": {k: round(v, 4) for k, v in self.critical_values.items()},
            "hedge_ratio": round(self.hedge_ratio, 6),
            "intercept": round(self.intercept, 6),
            "nobs": self.nobs,
            "cointegrated_at_5pct": self.cointegrated_at_5pct,
        }


def engle_granger(
    y0: Sequence[float], y1: Sequence[float], *, maxlag: int | None = None,
    autolag: str | None = "aic",
) -> CointegrationResult:
    """Two-step Engle-Granger, following ``stattools.py:1702 (coint)`` exactly.

    Step one regresses ``y0`` on ``[y1, 1]``; step two runs the ADF on the residuals with no
    constant. The critical values come from the ``N=2`` table because the residual is a *fitted*
    series — using the ordinary ADF table here is the classic error, and it makes spurious pairs
    look significant, because a fitted residual is stationary more often than an observed one.
    """
    a = [float(v) for v in y0]
    b = [float(v) for v in y1]
    if len(a) != len(b):
        raise CointegrationError(f"series differ in length: {len(a)} vs {len(b)}")
    if len(a) < MIN_OBSERVATIONS:
        raise CointegrationError(
            f"{len(a)} observation(s) is below the {MIN_OBSERVATIONS} this module will test"
        )
    stage_one = ols(a, [b, [1.0] * len(a)])
    if stage_one.rsquared > 1 - 1e-7:
        raise CointegrationError(
            "the two series are collinear to machine precision; a cointegration test on them is "
            "not meaningful (statsmodels warns and returns -inf here)"
        )
    residual_test = adf(stage_one.resid, regression="n", maxlag=maxlag, autolag=autolag)
    # N=2, regression "c", and nobs-1: stattools.py:1835 keeps the -1 to match Stata's egranger and
    # says outright that it does not know why. Copied for agreement, flagged so it is not mistaken
    # for a derivation of ours.
    crit = mackinnon_crit(regression="c", n=2, nobs=len(a) - 1)
    pvalue = mackinnon_p(residual_test.statistic, regression="c", n=2)
    return CointegrationResult(
        statistic=residual_test.statistic, pvalue=pvalue, critical_values=crit,
        hedge_ratio=stage_one.params[0], intercept=stage_one.params[1],
        residuals=stage_one.resid, nobs=len(a),
    )


# --- mean-reversion speed ------------------------------------------------------------------------


def half_life(spread: Sequence[float]) -> float | None:
    """Ornstein-Uhlenbeck half-life in bars, or ``None`` when the spread does not mean-revert.

    The discrete OU model ``ds_t = lambda * (mu - s_{t-1}) dt + noise`` is estimated as the AR(1)
    regression of the change on the lagged level; the half-life is ``ln(2) / lambda``. A
    non-negative slope means the series is diverging rather than reverting, and the honest answer
    there is *no half-life*, not a huge one — the formula would return a number and the number would
    describe a process that does not exist.
    """
    values = [float(v) for v in spread]
    if len(values) < MIN_OBSERVATIONS:
        raise CointegrationError(
            f"{len(values)} observation(s) is below the {MIN_OBSERVATIONS} needed for a half-life"
        )
    lagged = values[:-1]
    changes = [b - a for a, b in pairwise(values)]
    fit = ols(changes, [lagged, [1.0] * len(lagged)])
    slope = fit.params[0]
    if slope >= 0:
        return None
    return math.log(2.0) / -slope


def zscores(spread: Sequence[float], *, lookback: int = 240) -> list[float | None]:
    """Point-in-time z-scores: each uses only the ``lookback`` bars **before** it.

    The whole-sample z-score is the standard presentation in every pairs write-up read for this
    module and it cannot be traded: at the moment of the signal, the mean and standard deviation it
    is measured against had not happened yet. Entries before enough history exist are ``None``
    rather than computed from a short window, because an early z built on twenty points is mostly a
    statement about those twenty points.
    """
    if lookback < 20:
        raise CointegrationError("a z-score needs at least 20 prior observations to mean anything")
    values = [float(v) for v in spread]
    out: list[float | None] = []
    for i, value in enumerate(values):
        if i < lookback:
            out.append(None)
            continue
        window = values[i - lookback:i]
        mean = sum(window) / lookback
        variance = sum((w - mean) ** 2 for w in window) / lookback
        sd = math.sqrt(variance)
        out.append(None if sd <= 0 else (value - mean) / sd)
    return out


# --- one pair, end to end ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PairResult:
    """Everything needed to decide whether a pair is worth trading, including the reasons not to."""

    left: str
    right: str
    in_sample: CointegrationResult
    out_of_sample: ADFResult | None
    half_life_bars: float | None
    z_now: float | None
    spread_sd: float
    gross_notional: float
    required_z: float
    extreme_share: float
    """Share of point-in-time z-scores whose absolute value cleared ``required_z``. The honest
    frequency with which this pair has *ever* been worth the four legs."""

    @property
    def cost_bps(self) -> float:
        return TAKER_BPS_PER_LEG * LEGS_PER_ROUND_TRIP

    @property
    def tradeable(self) -> bool:
        """Cointegrated out of sample, mean-reverting, and extreme often enough to pay for itself.

        Four conditions, all of which a published pairs backtest typically checks at most two of.
        """
        return (
            self.out_of_sample is not None
            and self.out_of_sample.stationary_at_5pct
            and self.half_life_bars is not None
            and self.extreme_share > 0.0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "pair": f"{self.left}/{self.right}",
            "in_sample": self.in_sample.as_dict(),
            "out_of_sample": None if self.out_of_sample is None else self.out_of_sample.as_dict(),
            "half_life_bars": None if self.half_life_bars is None else round(
                self.half_life_bars, 2
            ),
            "z_now": None if self.z_now is None else round(self.z_now, 3),
            "required_z_to_clear_costs": round(self.required_z, 3),
            "extreme_share": round(self.extreme_share, 5),
            "cost_bps": self.cost_bps,
            "tradeable": self.tradeable,
        }


def pair_test(
    left: str, right: str, y0: Sequence[float], y1: Sequence[float], *,
    train_fraction: float = 0.6, lookback: int = 240,
) -> PairResult:
    """Fit the hedge ratio in-sample, then test everything else on data the fit never saw.

    The frozen ratio is the point. Refitting it on the test slice would make the out-of-sample ADF a
    second in-sample test wearing different labels, which is the most common way a pairs study
    reports a stationary spread that was never tradeable.
    """
    if not 0.2 <= train_fraction <= 0.9:
        raise CointegrationError("train_fraction must leave a usable sample on both sides")
    a = [float(v) for v in y0]
    b = [float(v) for v in y1]
    if len(a) != len(b):
        raise CointegrationError(f"series differ in length: {len(a)} vs {len(b)}")
    split = int(len(a) * train_fraction)
    in_sample = engle_granger(a[:split], b[:split])

    beta, alpha = in_sample.hedge_ratio, in_sample.intercept
    spread = [x - beta * y - alpha for x, y in zip(a, b, strict=True)]
    held_out = spread[split:]
    out_of_sample: ADFResult | None = None
    if len(held_out) >= MIN_OBSERVATIONS:
        out_of_sample = adf(held_out, regression="n")

    life = half_life(spread) if len(spread) >= MIN_OBSERVATIONS else None
    zs = zscores(spread, lookback=lookback) if len(spread) > lookback else []
    z_now = zs[-1] if zs else None

    mean = sum(spread) / len(spread)
    sd = math.sqrt(sum((s - mean) ** 2 for s in spread) / len(spread))
    gross = abs(a[-1]) + abs(beta * b[-1])
    cost = TAKER_BPS_PER_LEG * LEGS_PER_ROUND_TRIP
    # To pay four taker legs the spread must travel `cost` bps of the gross notional needed to hold
    # both sides. Expressed in standard deviations, that is the entry the pair has to reach before
    # the trade is worth doing at all.
    required = (cost / 10_000.0) * gross / sd if sd > 0 else math.inf
    measured = [z for z in zs if z is not None]
    extreme = (
        sum(1 for z in measured if abs(z) >= required) / len(measured) if measured else 0.0
    )
    return PairResult(
        left=left, right=right, in_sample=in_sample, out_of_sample=out_of_sample,
        half_life_bars=life, z_now=z_now, spread_sd=sd, gross_notional=gross,
        required_z=required, extreme_share=extreme,
    )


# --- many pairs, honestly ------------------------------------------------------------------------


# `benjamini_hochberg` and `bonferroni` are NOT reimplemented here. `backtest/validation.py:351,372`
# already carries both, validated and with input checking, and a second copy of a correction is how
# two parts of one system come to disagree about what survived. Imported at use rather than at
# module level so this module stays importable with no backtest machinery loaded.

@dataclass(frozen=True, slots=True)
class ScanReport:
    """Every pair tested, with the multiple-testing arithmetic stated rather than implied."""

    pairs: tuple[PairResult, ...]
    q: float

    hypotheses: tuple[float, ...] = ()
    """Every p-value examined, when that is more than the pairs still being reported.

    :func:`narrow` filters the pairs down to those involving one instrument and leaves this field
    holding the original set, so the multiple-testing arithmetic keeps counting the hypotheses that
    were actually tested. Reporting eleven pairs as though only eleven were examined, when sixty-six
    were, understates the burden six-fold — which is precisely how a scan of everything becomes a
    story about one thing.
    """

    @property
    def tests(self) -> int:
        """Hypotheses examined, not rows displayed."""
        return len(self.hypotheses) if self.hypotheses else len(self.pairs)

    @property
    def reported(self) -> int:
        return len(self.pairs)

    @property
    def expected_false_positives(self) -> float:
        """How many pairs would look cointegrated at 5% if none of them were."""
        return 0.05 * self.tests

    @property
    def bonferroni_threshold(self) -> float:
        return 0.05 / self.tests if self.tests else 0.05

    @property
    def survivors_fdr(self) -> tuple[PairResult, ...]:
        """Benjamini-Hochberg rather than Bonferroni as the headline: at 66 tests Bonferroni wants
        p < 0.00076, which on 60 days of hourly data rejects real relationships along with the
        spurious ones. Both are reported, because choosing the more permissive correction after
        seeing the results is itself a form of the problem this corrects."""
        from argus.backtest.validation import benjamini_hochberg

        universe = list(self.hypotheses) or [p.in_sample.pvalue for p in self.pairs]
        flags = benjamini_hochberg(universe, fdr=self.q)
        surviving = [pv for pv, keep in zip(universe, flags, strict=True) if keep]
        if not surviving:
            return ()
        cutoff = max(surviving)
        return tuple(p for p in self.pairs if p.in_sample.pvalue <= cutoff)

    @property
    def survivors_bonferroni(self) -> tuple[PairResult, ...]:
        limit = self.bonferroni_threshold
        return tuple(p for p in self.pairs if p.in_sample.pvalue <= limit)

    @property
    def tradeable(self) -> tuple[PairResult, ...]:
        keep = {(p.left, p.right) for p in self.survivors_fdr}
        return tuple(p for p in self.pairs if (p.left, p.right) in keep and p.tradeable)

    @property
    def median_required_z(self) -> float | None:
        """The middle pair's entry threshold. Says which constraint is actually binding."""
        finite = sorted(p.required_z for p in self.pairs if math.isfinite(p.required_z))
        if not finite:
            return None
        mid = len(finite) // 2
        return finite[mid] if len(finite) % 2 else (finite[mid - 1] + finite[mid]) / 2

    @property
    def binding_constraint(self) -> str:
        """Which of the two hurdles fails, named rather than left for the reader to infer.

        Worth separating because the answer here is the opposite of the one `arbitrage_study.py`
        found for the rToken basis. There the fee was larger than the entire effect. Here the
        spreads are wide relative to 24bps — the median pair needs a fraction of a standard
        deviation to cover costs — and what fails is stationarity. A wide spread is only an
        opportunity if it comes back; otherwise it is divergence risk wearing an opportunity's
        clothes, and reporting the low cost hurdle without that sentence would be an invitation to
        trade the worst pairs hardest.
        """
        median = self.median_required_z
        if median is None:
            return ""
        if median < 1.0:
            return (
                f"Costs are not the binding constraint: the median pair needs only "
                f"|z| >= {median:.2f} to clear {TAKER_BPS_PER_LEG * LEGS_PER_ROUND_TRIP:.0f}bps, "
                f"because these spreads are wide. What fails is stationarity — a wide spread that "
                f"does not revert is divergence risk, not edge"
            )
        return (
            f"Costs are the binding constraint: the median pair needs |z| >= {median:.2f} before "
            f"the four legs pay for themselves"
        )

    @property
    def verdict(self) -> str:
        naive = sum(1 for p in self.pairs if p.in_sample.pvalue <= 0.05)
        if not self.pairs:
            return "No pair had enough overlapping history to test"
        shown = "" if self.reported == self.tests else f" ({self.reported} shown)"
        lines = (
            f"{self.tests} pair(s) tested{shown}. {naive} reached p <= 0.05 naively, against "
            f"{self.expected_false_positives:.1f} expected by chance alone at that level. "
            f"{len(self.survivors_fdr)} survive Benjamini-Hochberg at q={self.q:.2f} and "
            f"{len(self.survivors_bonferroni)} survive Bonferroni (p <= "
            f"{self.bonferroni_threshold:.5f})."
        )
        if not self.tradeable:
            return lines + (
                " None is tradeable: after a four-leg round trip of "
                f"{TAKER_BPS_PER_LEG * LEGS_PER_ROUND_TRIP:.0f}bps, no surviving pair has both a "
                "stationary out-of-sample spread and a history of reaching the entry that pays for "
                "it. Reported as no opportunity rather than as the best of a bad list."
                f" {self.binding_constraint}"
            )
        best = min(self.tradeable, key=lambda p: p.in_sample.pvalue)
        return lines + (
            f" {len(self.tradeable)} tradeable after costs; the strongest is {best.left}/"
            f"{best.right}, half-life {best.half_life_bars:.1f} bars, entry needs "
            f"|z| >= {best.required_z:.1f} which has happened {best.extreme_share:.1%} of the time"
        )

    def render(self) -> str:
        head = (
            f"{'pair':24} {'p(EG)':>9} {'half-life':>10} {'z now':>7} {'need |z|':>9} "
            f"{'seen':>7}"
        )
        rows = [head, "-" * len(head)]
        for p in sorted(self.pairs, key=lambda x: x.in_sample.pvalue):
            life = "none" if p.half_life_bars is None else f"{p.half_life_bars:.1f}"
            now = "n/a" if p.z_now is None else f"{p.z_now:+.2f}"
            rows.append(
                f"{p.left + '/' + p.right:24} {p.in_sample.pvalue:9.4f} {life:>10} {now:>7} "
                f"{p.required_z:9.2f} {p.extreme_share:6.2%}"
            )
        rows += ["", self.verdict]
        return "\n".join(rows)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tests": self.tests,
            "reported": self.reported,
            "q": self.q,
            "expected_false_positives_at_5pct": round(self.expected_false_positives, 2),
            "bonferroni_threshold": round(self.bonferroni_threshold, 6),
            "survivors_fdr": [f"{p.left}/{p.right}" for p in self.survivors_fdr],
            "survivors_bonferroni": [f"{p.left}/{p.right}" for p in self.survivors_bonferroni],
            "tradeable": [f"{p.left}/{p.right}" for p in self.tradeable],
            "pairs": [p.as_dict() for p in self.pairs],
            "median_required_z": (
                None if self.median_required_z is None else round(self.median_required_z, 4)
            ),
            "binding_constraint": self.binding_constraint,
            "verdict": self.verdict,
        }


def narrow(report: ScanReport, symbol: str) -> ScanReport | None:
    """The pairs involving one instrument, with the full hypothesis count carried along.

    ``None`` when that instrument appears in none of them, because an empty table and "we did not
    test this" are different statements and only one of them is true here.
    """
    kept = tuple(p for p in report.pairs if symbol in (p.left, p.right))
    if not kept:
        return None
    return ScanReport(
        pairs=kept, q=report.q,
        hypotheses=report.hypotheses or tuple(p.in_sample.pvalue for p in report.pairs),
    )


def scan(
    series: Mapping[str, Sequence[float]], *, q: float = 0.05, train_fraction: float = 0.6,
    lookback: int = 240,
) -> ScanReport:
    """Every unordered pair, tested the same way, with the test count carried into the verdict."""
    names = sorted(series)
    results: list[PairResult] = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = series[left], series[right]
            size = min(len(a), len(b))
            if size < MIN_OBSERVATIONS * 2:
                continue
            try:
                results.append(pair_test(
                    left, right, list(a[-size:]), list(b[-size:]),
                    train_fraction=train_fraction, lookback=lookback,
                ))
            except CointegrationError:
                # A pair that cannot be tested is dropped from the count of tests rather than
                # counted as a non-rejection: including it would dilute the multiple-testing
                # correction with hypotheses that were never examined.
                continue
    return ScanReport(pairs=tuple(results), q=q)


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import json
    import sys
    from pathlib import Path

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from argus.market.bitget import RTOKEN_SYMBOLS
    from argus.market.history import CandleType, fetch_range

    parser = argparse.ArgumentParser(description="which rToken pairs are actually cointegrated?")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--interval", default="1H")
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--lookback", type=int, default=240)
    parser.add_argument("--symbols", default="")
    args = parser.parse_args()

    wanted = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or list(
        RTOKEN_SYMBOLS
    )
    closes: dict[str, list[float]] = {}
    for symbol in wanted:
        try:
            bars = fetch_range(
                symbol, days=args.days, interval=args.interval, candle_type=CandleType.MARKET,
            )
        except Exception as exc:
            print(f"  {symbol}: no history ({type(exc).__name__})")
            continue
        if len(bars) >= MIN_OBSERVATIONS * 2:
            closes[symbol] = [float(b.close) for b in bars]

    if len(closes) < 2:
        print("fewer than two symbols had usable history; nothing to test")
        return 1

    report = scan(
        closes, train_fraction=args.train_fraction, lookback=args.lookback,
    )
    print(report.render())
    out = Path(__file__).resolve().parents[3] / "data" / "cointegration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "LEGS_PER_ROUND_TRIP",
    "MIN_OBSERVATIONS",
    "TAKER_BPS_PER_LEG",
    "ADFResult",
    "CointegrationError",
    "CointegrationResult",
    "PairResult",
    "Regression",
    "ScanReport",
    "adf",
    "engle_granger",
    "half_life",
    "mackinnon_crit",
    "mackinnon_p",
    "narrow",
    "ols",
    "pair_test",
    "scan",
    "zscores",
]
