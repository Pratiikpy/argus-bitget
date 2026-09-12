"""Track 1 — execution-aware alpha on the session boundary.

The Open Theme names "execution-aware alpha that incorporates fees, slippage, funding, market
impact, and strategy capacity" as an example direction. This module is that, built on the one
structural fact we have measured rather than assumed: **price discovery attenuates roughly 7x while
the anchor market is shut** (RTH 32.17 bps/hr, weekend 4.53 bps/hr on NVDAUSDT over 90 days).

Every variant below is a session-conditioned rule. None of them is expected to beat a 12bps round
trip — the measured intraday edge on this venue is ~0.00% and the fee is larger than nearly every
effect we have found. The value of running them is that the result is *measured* rather than
assumed, and the strategies that fail are reported as failing.

Each variant is deliberately simple and few in number. A sweep of 500 parameter combinations would
produce a beautiful Sharpe from noise; with the Deflated Sharpe gate consuming the trial count, a
large sweep is self-defeating, which is the correct incentive.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from argus.backtest.engine import Bar
from argus.truth.clocks import DualClock, SessionPhase

_CLOCK = DualClock()


def _phase(bar: Bar) -> SessionPhase:
    return _CLOCK.phase(bar.ts)


def _ret(bars: Sequence[Bar], i: int, lookback: int) -> float:
    """Return over the last `lookback` bars, ending at `i`. Uses no future information."""
    j = i - lookback
    if j < 0:
        return 0.0
    a, b = float(bars[j].close), float(bars[i].close)
    return (b - a) / a if a > 0 else 0.0


# --- the variants -----------------------------------------------------------------------------

def hold_through_closure(bars: Sequence[Bar], i: int) -> float:
    """Long only while the anchor is shut, flat while it trades.

    Tests the crudest form of the overnight-drift claim. If the ~0.10% overnight drift we measured
    elsewhere is real and persistent, this captures it; if it is smaller than the cost of getting
    in and out at every session boundary, this loses — which is the expected outcome and the point.
    """
    return 1.0 if not _phase(bars[i]).has_price_discovery else 0.0


def trade_only_when_open(bars: Sequence[Bar], i: int) -> float:
    """The mirror image. Long only during RTH.

    Included as a control: if `hold_through_closure` wins, this must lose, and if both win the
    result is a drift in the instrument rather than a session effect.
    """
    return 1.0 if _phase(bars[i]).has_price_discovery else 0.0


def weekend_only(bars: Sequence[Bar], i: int) -> float:
    """Long across weekends only — the longest closures, where attenuation is deepest.

    Turnover is low by construction: roughly one round trip per week, which is the only way a
    12bps fee is survivable at all.
    """
    return 1.0 if _phase(bars[i]) in (SessionPhase.WEEKEND, SessionPhase.HOLIDAY) else 0.0


def closure_momentum(bars: Sequence[Bar], i: int) -> float:
    """Carry the direction established during the closure into the reopen.

    This is the rule the gap study already falsified: continuation was 50.6% overnight and the
    weekend's 57.7% failed its train/test split. Kept in the sweep precisely because a strategy
    that has been shown not to work must be *shown* not to work in the backtest too, with the fee
    included. A sweep containing only plausible winners is a rigged sweep.
    """
    if _phase(bars[i]).has_price_discovery:
        return 0.0
    move = _ret(bars, i, 6)
    return 1.0 if move > 0 else (-1.0 if move < 0 else 0.0)


def closure_reversion(bars: Sequence[Bar], i: int) -> float:
    """The opposite: fade the closure move.

    If the token over-reacts to information it cannot properly price against a 7x-damped
    reference, this is the trade. It is the direct competitor to `closure_momentum`, and exactly
    one of them can be right.
    """
    return -closure_momentum(bars, i)


def low_turnover_carry(bars: Sequence[Bar], i: int) -> float:
    """Hold long continuously, but stand aside in the thinnest session.

    Designed around the fee rather than around a signal: near-zero turnover, one decision per
    weekend. If any session rule survives 12bps, it will look like this one.
    """
    return 0.0 if _phase(bars[i]) in (SessionPhase.WEEKEND, SessionPhase.HOLIDAY) else 1.0


VARIANTS = {
    "hold_through_closure": hold_through_closure,
    "trade_only_when_open": trade_only_when_open,
    "weekend_only": weekend_only,
    "closure_momentum": closure_momentum,
    "closure_reversion": closure_reversion,
    "low_turnover_carry": low_turnover_carry,
}


def bars_from_candles(candles: list[dict[str, object]]) -> list[Bar]:
    """Adapt market history into engine bars."""
    out: list[Bar] = []
    for c in candles:
        ts = c["ts"]
        close = c["close"]
        if isinstance(ts, str) or not hasattr(ts, "tzinfo"):
            continue
        out.append(Bar(ts=ts, close=Decimal(str(close))))  # type: ignore[arg-type]
    return out
