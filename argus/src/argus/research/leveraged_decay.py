"""What a daily-rebalanced leveraged fund loses when its index goes nowhere.

"How much does TQQQ decay if QQQ goes sideways for a month?" was declined (2026-09-25 audit,
round 2). It has a measured answer and a formula, and both are given:

* **Measured.** Every window of the asked length in the fund's own history (split-adjusted daily
  closes) in which the index finished within ``FLAT_BAND`` of where it started — a sideways
  stretch — and what the fund actually did over the same days, against ``L`` times the index's
  move over the window. That shortfall is the decay a holder took: path loss plus the fund's
  expense ratio and financing cost. Windows start on every day, so they overlap and are not
  independent; the count says how much history there is, not how many separate tests.
* **Formula.** A fund that resets to ``L`` times its index every day compounds ``(1 + L*r)``
  daily; to second order its log return over ``T`` days trails ``L`` times the index's by
  ``(L^2 - L)/2 * sigma^2 * T`` (Avellaneda & Zhang, "Path-dependence of leveraged ETF returns",
  SIAM J. Financial Math. 2010; Cheng & Madhavan 2009). With the index's recent daily variance
  that is the decay the current market implies, before fees.

The two usually differ, and the answer says why: the formula is today's volatility, the measured
windows had whatever volatility those sideways stretches had.
"""

from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass
from datetime import date

FUNDS: dict[str, tuple[str, float]] = {
    "TQQQ": ("QQQ", 3.0), "SQQQ": ("QQQ", -3.0), "QLD": ("QQQ", 2.0), "QID": ("QQQ", -2.0),
    "UPRO": ("SPY", 3.0), "SPXL": ("SPY", 3.0), "SPXU": ("SPY", -3.0), "SPXS": ("SPY", -3.0),
    "SSO": ("SPY", 2.0), "SDS": ("SPY", -2.0), "SOXL": ("SOXX", 3.0), "SOXS": ("SOXX", -3.0),
    "TNA": ("IWM", 3.0), "TZA": ("IWM", -3.0), "UDOW": ("DIA", 3.0), "SDOW": ("DIA", -3.0),
    "TECL": ("XLK", 3.0), "FAS": ("XLF", 3.0), "LABU": ("XBI", 3.0), "NVDL": ("NVDA", 2.0),
    "TSLL": ("TSLA", 2.0), "BITX": ("IBIT", 2.0), "ETHU": ("ETHA", 2.0),
}
"""Leveraged funds and the index each resets against daily, with its multiple — each fund's
stated daily target in its issuer's prospectus (ProShares, Direxion, GraniteShares, Volatility
Shares)."""

FLAT_BAND = 0.02
"""An index move within +/-2% over the window counts as sideways."""
VOL_DAYS = 60


@dataclass(frozen=True, slots=True)
class Decay:
    fund: str
    index: str
    multiple: float
    days: int
    windows: int
    median_fund: float | None
    median_index: float | None
    median_shortfall: float | None
    """The fund's return less ``multiple`` times the index's, per window, at the median."""
    worst_shortfall: float | None
    best_shortfall: float | None
    formula: float
    daily_vol: float


def measure(fund: str, fund_closes: dict[date, float], index_closes: dict[date, float],
            days: int) -> Decay:
    """``fund_closes`` and ``index_closes`` map a date to a split-adjusted close."""
    index, multiple = FUNDS[fund]
    dates = sorted(set(fund_closes) & set(index_closes))
    fund_moves: list[float] = []
    index_moves: list[float] = []
    shortfalls: list[float] = []
    for start, end in zip(dates, dates[days:], strict=False):
        if index_closes[start] <= 0 or fund_closes[start] <= 0:
            continue
        index_move = index_closes[end] / index_closes[start] - 1
        if abs(index_move) <= FLAT_BAND:
            fund_move = fund_closes[end] / fund_closes[start] - 1
            fund_moves.append(fund_move)
            index_moves.append(index_move)
            shortfalls.append(fund_move - multiple * index_move)
    recent = [index_closes[d] for d in dates[-(VOL_DAYS + 1):]]
    returns = [b / a - 1 for a, b in itertools.pairwise(recent) if a > 0]
    sigma = statistics.pstdev(returns) if len(returns) > 10 else 0.0
    formula = math.expm1(-(multiple * multiple - multiple) / 2 * sigma * sigma * days)
    return Decay(
        fund=fund, index=index, multiple=multiple, days=days, windows=len(fund_moves),
        median_fund=statistics.median(fund_moves) if fund_moves else None,
        median_index=statistics.median(index_moves) if index_moves else None,
        median_shortfall=statistics.median(shortfalls) if shortfalls else None,
        worst_shortfall=min(shortfalls) if shortfalls else None,
        best_shortfall=max(shortfalls) if shortfalls else None,
        formula=formula, daily_vol=sigma)


def lines(result: Decay) -> list[str]:
    period = f"{result.days} trading days"
    out: list[str] = []
    if (result.median_shortfall is not None and result.best_shortfall is not None
            and result.worst_shortfall is not None):
        out.append(
            f"Actionable: over {period} in which {result.index} went sideways (ended within "
            f"±{FLAT_BAND:.0%}), {result.fund} fell short of {result.multiple:+g}x the index's "
            f"move by a median {-result.median_shortfall:.1%} — from {-result.best_shortfall:.1%}"
            f" to {-result.worst_shortfall:.1%} across {result.windows} overlapping windows of "
            f"its own history. That is the decay a holder actually took, fees and financing "
            f"included.")
    out.append(
        f"{'Actionable: ' if not out else ''}At {result.index}'s last {VOL_DAYS} days of "
        f"volatility ({result.daily_vol:.2%} a day), a fund reset to {result.multiple:+g}x daily "
        f"loses about {-result.formula:.1%} over {period} of a flat index from compounding alone, "
        f"before fees — (L²-L)/2 x variance x days, Avellaneda & Zhang (2010).")
    out.append(f"The formula uses today's volatility; the measured stretches had their own, so "
               f"the two differ — a calmer sideways market decays {result.fund} less, a choppy "
               f"one more. Nothing here is a forecast of the next {period}.")
    return out
