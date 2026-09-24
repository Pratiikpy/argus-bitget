"""Track 1 — the five remaining Alpha Factory sub-themes, as runnable strategies.

Each is a signal function for :mod:`argus.backtest.engine`, so every one is scored net of the 12bps
round trip, lagged by construction, and split chronologically into out-of-sample.

Sub-theme coverage, with the method each is built from:

===========================  =========================================================
Sub-theme                    Method
===========================  =========================================================
After-Hours Pricing          Barclay & Hendershott weighted price contribution, adapted
                             to a 7x-attenuated reference rather than a closed one
Cross-Market Correlation     Rolling hedge ratio with a staleness correction; the usable
                             ratio, not the correlation
rToken Factors               Session-boundary factor family — basis, time-to-discovery,
                             attenuation state. Not generic price factors
Cross-Asset Rotation         Regime-conditioned allocation penalised by hedgeability
Open Theme                   Execution-aware alpha: optimise net-of-everything, not
                             predicted return
===========================  =========================================================

**None of these is expected to beat the fee.** The measured intraday edge on this venue is roughly
zero and the round trip is 12bps, so the honest prior is that most will lose. They are built anyway
because a measured negative with the machinery visible is a result, and because the one that does
survive can only be found by running all of them under the same gate.
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
    j = i - lookback
    if j < 0:
        return 0.0
    a, b = float(bars[j].close), float(bars[i].close)
    return (b - a) / a if a > 0 else 0.0


def _vol(bars: Sequence[Bar], i: int, lookback: int) -> float:
    """Realised volatility over the lookback, as a plain standard deviation of bar returns."""
    if i - lookback < 1:
        return 0.0
    rets = []
    for k in range(i - lookback + 1, i + 1):
        a, b = float(bars[k - 1].close), float(bars[k].close)
        if a > 0:
            rets.append((b - a) / a)
    if len(rets) < 2:
        return 0.0
    mu = sum(rets) / len(rets)
    return float((sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)) ** 0.5)


def _basis_bps(bar: Bar) -> float:
    """Token-vs-index basis in bps, if the bar carries an index. Zero when unknown.

    Returning zero for "unknown" rather than guessing keeps a missing index out of the signal
    instead of turning it into a fabricated edge.
    """
    extra = bar.extra or {}
    index = extra.get("index")
    if index is None:
        return 0.0
    idx = float(index)
    if idx <= 0:
        return 0.0
    return (float(bar.close) - idx) / idx * 10_000


# ---------------------------------------------------------------------------------------------
# After-Hours Information Pricing
# ---------------------------------------------------------------------------------------------

def afterhours_attenuation_fade(bars: Sequence[Bar], i: int) -> float:
    """Fade moves that happen against a damped reference.

    The measured fact: weekend index movement runs at ~1/7th of RTH intensity. A large token move
    during a closed session is therefore moving against a reference that cannot confirm it, which
    is the definition of an unconfirmed move. This fades it.
    """
    phase = _phase(bars[i])
    if phase.has_price_discovery:
        return 0.0
    move = _ret(bars, i, 6)
    vol = _vol(bars, i, 48)
    if vol <= 0:
        return 0.0
    # Only act when the closed-session move is large relative to normal.
    z = move / (vol * (6 ** 0.5))
    if z > 1.5:
        return -1.0
    if z < -1.5:
        return 1.0
    return 0.0


def afterhours_discovery_countdown(bars: Sequence[Bar], i: int) -> float:
    """Scale exposure by how far the anchor still is from reopening.

    Gap risk is a function of time-to-discovery, so exposure is reduced as the remaining closure
    lengthens rather than held flat across a 66-hour weekend.
    """
    state = _CLOCK.state(bars[i].ts)
    if state.phase.has_price_discovery:
        return 1.0
    hours = state.hours_to_next_discovery
    return max(0.0, 1.0 - hours / 72.0)


# ---------------------------------------------------------------------------------------------
# Cross-Market Correlation
# ---------------------------------------------------------------------------------------------

def crossmarket_basis_reversion(bars: Sequence[Bar], i: int) -> float:
    """Trade the token back toward its index when the basis stretches.

    The usable form of the cross-market idea. Measured: median basis 4.27bps and only 7.72% of
    28,067 observations clear the fee — so the threshold is deliberately set above the round trip
    rather than at zero.
    """
    basis = _basis_bps(bars[i])
    if basis > 15:
        return -1.0   # token rich versus its index
    if basis < -15:
        return 1.0    # token cheap
    return 0.0


def crossmarket_stale_corrected(bars: Sequence[Bar], i: int) -> float:
    """The same idea, but only when the index is genuinely moving.

    A basis measured against a stale reference is measuring staleness. During a closed session the
    index still updates but at ~1/7th intensity, so this restricts the trade to sessions where the
    reference is alive enough to mean-revert toward.
    """
    if not _phase(bars[i]).has_price_discovery:
        return 0.0
    return crossmarket_basis_reversion(bars, i)


# ---------------------------------------------------------------------------------------------
# rToken Factors
# ---------------------------------------------------------------------------------------------

def rtoken_session_boundary(bars: Sequence[Bar], i: int) -> float:
    """Position only across the session boundary itself.

    The rToken-native factor: the instrument's distinguishing feature is that it trades through a
    boundary its anchor does not. This holds exposure only in the hours immediately around the
    close and the reopen, where that asymmetry is largest.
    """
    state = _CLOCK.state(bars[i].ts)
    if state.phase.has_price_discovery:
        return 0.0
    hours = state.hours_to_next_discovery
    return 1.0 if hours <= 4.0 else 0.0


def rtoken_attenuation_carry(bars: Sequence[Bar], i: int) -> float:
    """Hold when realised volatility is below what the session implies.

    A closed session should be quiet. When the token is quieter still, carry is cheap; when it is
    louder than a damped reference can justify, something is happening that cannot be priced.
    """
    phase = _phase(bars[i])
    if phase.has_price_discovery:
        return 0.0
    recent = _vol(bars, i, 12)
    baseline = _vol(bars, i, 168)
    if baseline <= 0:
        return 0.0
    return 1.0 if recent < baseline * 0.7 else 0.0


# ---------------------------------------------------------------------------------------------
# Cross-Asset Rotation
# ---------------------------------------------------------------------------------------------

def rotation_hedgeability_weighted(bars: Sequence[Bar], i: int) -> float:
    """Allocate by trend, penalised when no hedge is placeable.

    This is where Track 2's hedgeability work becomes Track 1 alpha: a position with a good signal
    and an empty hedge menu carries a higher effective risk charge than its volatility implies, so
    it is held at reduced size rather than full.
    """
    trend = _ret(bars, i, 24)
    if trend <= 0:
        return 0.0
    state = _CLOCK.state(bars[i].ts)
    # Unhedgeable sessions get a third of the weight of hedgeable ones.
    return 1.0 if state.phase.has_price_discovery else 0.33


def rotation_regime_switch(bars: Sequence[Bar], i: int) -> float:
    """Long in low-volatility regimes, flat in high-volatility ones.

    Regime is measured, not predicted: current realised volatility against its own longer history.
    """
    fast = _vol(bars, i, 24)
    slow = _vol(bars, i, 240)
    if slow <= 0:
        return 0.0
    return 1.0 if fast < slow else 0.0


# ---------------------------------------------------------------------------------------------
# Open Theme — execution-aware alpha
# ---------------------------------------------------------------------------------------------

def execution_aware_alpha(bars: Sequence[Bar], i: int) -> float:
    """Trade only when the expected move plausibly exceeds the full cost of capturing it.

    The Open Theme names execution-aware alpha explicitly. The rule here is not a price prediction
    — it is a *cost gate* applied to one: estimate the move from realised volatility over the
    holding horizon, and take the position only if that estimate clears the 12bps round trip with
    margin. Turnover is suppressed by construction, which is the only way a 12bps fee is survivable.
    """
    vol = _vol(bars, i, 48)
    if vol <= 0:
        return 0.0
    # Expected absolute move over a 12-bar hold, in bps.
    expected_move_bps = vol * (12 ** 0.5) * 10_000
    if expected_move_bps < 24:      # need 2x the round trip before acting at all
        return 0.0
    trend = _ret(bars, i, 24)
    return 1.0 if trend > 0 else 0.0


def execution_aware_low_turnover(bars: Sequence[Bar], i: int) -> float:
    """The same gate, held far longer.

    Turnover is the mechanism by which a real edge is destroyed: a 45bps gross edge round-tripped
    four times is negative. This changes position at most once per day.
    """
    if i % 24 != 0:
        # Hold whatever the last decision was by recomputing it at the last decision point.
        anchor = i - (i % 24)
        if anchor < 48:
            return 0.0
        return execution_aware_alpha(bars, anchor)
    return execution_aware_alpha(bars, i)


TRACK1_VARIANTS = {
    # After-hours
    "afterhours_attenuation_fade": afterhours_attenuation_fade,
    "afterhours_discovery_countdown": afterhours_discovery_countdown,
    # Cross-market
    "crossmarket_basis_reversion": crossmarket_basis_reversion,
    "crossmarket_stale_corrected": crossmarket_stale_corrected,
    # rToken factors
    "rtoken_session_boundary": rtoken_session_boundary,
    "rtoken_attenuation_carry": rtoken_attenuation_carry,
    # Rotation
    "rotation_hedgeability_weighted": rotation_hedgeability_weighted,
    "rotation_regime_switch": rotation_regime_switch,
    # Open theme
    "execution_aware_alpha": execution_aware_alpha,
    "execution_aware_low_turnover": execution_aware_low_turnover,
}

SUBTHEME_OF = {
    "afterhours_attenuation_fade": "t1-afterhours",
    "afterhours_discovery_countdown": "t1-afterhours",
    "crossmarket_basis_reversion": "t1-crossmarket",
    "crossmarket_stale_corrected": "t1-crossmarket",
    "rtoken_session_boundary": "t1-rtokenfactor",
    "rtoken_attenuation_carry": "t1-rtokenfactor",
    "rotation_hedgeability_weighted": "t1-crossasset",
    "rotation_regime_switch": "t1-crossasset",
    "execution_aware_alpha": "t1-afopen",
    "execution_aware_low_turnover": "t1-afopen",
}


def bars_with_basis(
    market: list[tuple[object, Decimal]], index: dict[object, Decimal]
) -> list[Bar]:
    """Build bars carrying the index, so basis strategies have something to read."""
    out: list[Bar] = []
    for ts, close in market:
        idx = index.get(ts)
        out.append(Bar(ts=ts, close=close, extra={"index": idx} if idx else None))  # type: ignore[arg-type]
    return out
