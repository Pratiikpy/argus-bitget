"""Performance metrics, with the bugs that are endemic to this field designed out.

Track 1 is scored purely quantitatively on Sharpe, Sortino, max drawdown, turnover, out-of-sample
decay and rolling 30-day stability. Those numbers decide the outcome, so the implementations have
to be right — and in the corpus we tore down, they frequently were not.

Specific defects these functions exist to avoid, each found by reading real code:

* **vectorbt's Deflated Sharpe silently returns NaN.** A gate that returns NaN and is compared with
  ``>`` passes everything. Here :func:`deflated_sharpe` raises rather than returning a value that
  cannot be compared.
* **Qlib measures turnover as gross notional, not net delta**, overstating cost 2-3x on
  mean-reversion. :func:`turnover` takes the net change in position.
* **Annualisation is where Sharpe is quietly inflated.** The factor must match the sampling
  frequency of the returns, and passing daily returns with an hourly factor multiplies the result
  by about five. It is a required argument here — there is no default to get wrong.
* **Riskfolio-Lib reports swapped EVaR/TG values.** Nothing is taken from a library's report
  without recomputation.

The multiple-testing correction matters more than any single number. A search that runs hundreds of
trials and reports the best one has not found an edge; it has found the maximum of a noise
distribution. :func:`deflated_sharpe` takes the trial count and is the gate that consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from statistics import NormalDist

# Sampling frequencies, stated explicitly so a caller never has to guess one.
HOURLY_PER_YEAR = 24 * 365
DAILY_PER_YEAR = 252
WEEKLY_PER_YEAR = 52


class MetricError(ValueError):
    """A metric could not be computed honestly.

    Raised rather than returning NaN or a sentinel. A gate comparing against NaN passes
    everything, which is how a broken statistic becomes a green light.
    """


def _is_negligible(sd: float, reference: float) -> bool:
    """Is this deviation indistinguishable from zero, given floating-point noise?

    An exact ``sd == 0`` test does not work and the failure is dangerous. A constant series of
    0.01 returns produces ``sd = 1.8e-18`` rather than 0.0, so the guard never fires and Sharpe
    comes back as **8.7e16** — a number that passes every ``sharpe > threshold`` gate ever written.
    Caught by a test that expected the guard to fire.
    """
    scale = max(abs(reference), 1e-12)
    return sd <= scale * 1e-12


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _stdev(xs: list[float], *, sample: bool = True) -> float:
    n = len(xs)
    if n < 2:
        raise MetricError("standard deviation needs at least two observations")
    mu = _mean(xs)
    denom = n - 1 if sample else n
    return sqrt(sum((x - mu) ** 2 for x in xs) / denom)


def sharpe(returns: list[float], *, periods_per_year: int, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio.

    ``periods_per_year`` is required. Defaulting it is how an hourly series gets annualised as if
    it were daily and the ratio comes out roughly five times too large.
    """
    if len(returns) < 2:
        raise MetricError("Sharpe needs at least two returns")
    excess = [r - risk_free / periods_per_year for r in returns]
    sd = _stdev(excess)
    mu = _mean(excess)
    if _is_negligible(sd, mu):
        raise MetricError(
            "Sharpe is undefined for a zero-variance series "
            f"(sd={sd:.3e} is floating-point noise around mean {mu:.3e})"
        )
    return _mean(excess) / sd * sqrt(periods_per_year)


def sortino(returns: list[float], *, periods_per_year: int, target: float = 0.0) -> float:
    """Annualised Sortino ratio — downside deviation only.

    Downside deviation divides by the **full** sample count, not the count of losing periods.
    Dividing by the losers alone is a common error that inflates the ratio precisely when losses
    are rare, which is exactly when the inflation is most misleading.
    """
    if len(returns) < 2:
        raise MetricError("Sortino needs at least two returns")
    downside = [min(0.0, r - target) for r in returns]
    dd = sqrt(sum(d * d for d in downside) / len(returns))
    if _is_negligible(dd, _mean(returns) - target):
        raise MetricError("Sortino is undefined when there is no downside deviation")
    return (_mean(returns) - target) / dd * sqrt(periods_per_year)


def max_drawdown(equity: list[float]) -> float:
    """Largest peak-to-trough decline as a positive fraction.

    Denominator is the running peak, not the starting value. Using the start understates drawdown
    for any strategy that made money before losing it.
    """
    if not equity:
        raise MetricError("max drawdown needs an equity curve")
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst


def turnover(weights: list[float]) -> float:
    """Sum of absolute **changes** in position — net delta, not gross notional.

    Qlib's gross-notional measure overstates cost 2-3x on mean-reversion. Holding a position from
    one bar to the next costs nothing and must contribute nothing.
    """
    if len(weights) < 2:
        return 0.0
    return sum(abs(weights[i] - weights[i - 1]) for i in range(1, len(weights)))


def probabilistic_sharpe(
    observed: float, *, benchmark: float, n: int, skew: float = 0.0, kurtosis: float = 3.0
) -> float:
    """P(true Sharpe > benchmark), per Bailey & Lopez de Prado.

    Corrects for the non-normality that makes a naive Sharpe optimistic: negative skew and fat
    tails both inflate it, and trading returns reliably have both.
    """
    if n < 2:
        raise MetricError("probabilistic Sharpe needs at least two observations")
    denom = sqrt(1 - skew * observed + (kurtosis - 1) / 4 * observed ** 2)
    if denom <= 0:
        raise MetricError("probabilistic Sharpe denominator is non-positive")
    z = (observed - benchmark) * sqrt(n - 1) / denom
    return NormalDist().cdf(z)


def deflated_sharpe(
    observed: float, *, n: int, trials: int, variance_of_trials: float,
    skew: float = 0.0, kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio — the gate that consumes the trial count.

    The single most important statistic in this project. Running ``trials`` strategy variants and
    reporting the best is guaranteed to produce an impressive Sharpe from pure noise; the DSR asks
    whether the observed value beats what that search would have produced by chance.

    Raises on bad input rather than returning NaN. vectorbt's implementation returns NaN silently,
    and a gate comparing NaN with ``>`` admits everything it was built to stop.
    """
    if trials < 1:
        raise MetricError("trials must be at least 1 — an unrecorded trial count voids the gate")
    if variance_of_trials < 0:
        raise MetricError("variance of trial Sharpes cannot be negative")
    if n < 2:
        raise MetricError("deflated Sharpe needs at least two observations")

    if trials == 1:
        expected_max = 0.0
    else:
        # Expected maximum of `trials` draws, via the standard extreme-value approximation.
        euler = 0.5772156649015329
        nd = NormalDist()
        a = nd.inv_cdf(1 - 1 / trials)
        b = nd.inv_cdf(1 - 1 / (trials * exp(1)))
        expected_max = sqrt(variance_of_trials) * ((1 - euler) * a + euler * b)

    return probabilistic_sharpe(
        observed, benchmark=expected_max, n=n, skew=skew, kurtosis=kurtosis
    )


@dataclass(frozen=True, slots=True)
class Performance:
    """A full result. Every field Track 1 is scored on, plus the honesty checks."""

    periods: int
    periods_per_year: int
    total_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    turnover: float
    win_rate: float
    trades: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "periods": self.periods,
            "total_return_pct": round(100 * self.total_return, 3),
            "sharpe": round(self.sharpe, 3),
            "sortino": round(self.sortino, 3),
            "max_drawdown_pct": round(100 * self.max_drawdown, 3),
            "turnover": round(self.turnover, 3),
            "win_rate_pct": round(100 * self.win_rate, 2),
            "trades": self.trades,
        }


def evaluate(
    returns: list[float], weights: list[float], *, periods_per_year: int
) -> Performance:
    """Score a return stream. Costs must already be subtracted — this does not net them.

    Deliberately: a metrics module that applies costs invites two code paths, one of which
    eventually forgets. Cost is applied once, in the engine, on the way in.
    """
    if not returns:
        raise MetricError("no returns to evaluate")

    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r))

    wins = sum(1 for r in returns if r > 0)
    active = sum(1 for r in returns if r != 0)
    return Performance(
        periods=len(returns),
        periods_per_year=periods_per_year,
        total_return=equity[-1] - 1.0,
        sharpe=sharpe(returns, periods_per_year=periods_per_year),
        sortino=sortino(returns, periods_per_year=periods_per_year),
        max_drawdown=max_drawdown(equity),
        turnover=turnover(weights),
        win_rate=wins / active if active else 0.0,
        trades=active,
    )


def out_of_sample_decay(in_sample: float, out_of_sample: float) -> dict[str, float | bool]:
    """The handbook's own alert: out-of-sample Sharpe below half of in-sample.

    Reported rather than hidden. A strategy that decays past this threshold is overfitted, and
    saying so is worth more than a number that quietly fails the judge's check instead of ours.
    """
    ratio = out_of_sample / in_sample if in_sample != 0 else 0.0
    return {
        "in_sample_sharpe": round(in_sample, 3),
        "out_of_sample_sharpe": round(out_of_sample, 3),
        "retention_ratio": round(ratio, 3),
        "breaches_half_alert": ratio < 0.5,
    }
