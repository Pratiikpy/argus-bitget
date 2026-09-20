"""Position sizing — growth-optimal, and refused unless the confidence feeding it is calibrated.

ARGUS sizes every trade the same. That is not obviously wrong, but it throws away the one thing the
desk produces that ordinary systems do not: a stated probability. Kelly turns that probability into
a size, and half-Kelly does it without betting the book on the estimate being exact.

Taken from ``Bastion`` ``src/risk/kelly.ts:6-21`` (MIT), which is the cleanest implementation in the
sweep: ``f* = (p·b - q) / b``, clamped at zero so a negative edge produces no position rather than a
short, and halved by default because a full-Kelly stake on a mis-estimated edge is how a good
strategy goes to zero.

**The gate that matters more than the formula.** Sizing on confidence amplifies whatever that
confidence is worth. If the desk says 0.9 and is right 60% of the time, Kelly does not express an
edge — it levers a bias. So :func:`size` will not use a confidence that has not passed a calibration
check, and :func:`calibration_gate` is the thing that decides. This is why the master plan lists the
calibration check as a *prerequisite* to this module rather than a companion: the order is the whole
safety property.

ARGUS already measures expected calibration error in :mod:`argus.eval.observatory`, so the gate is
a read rather than new machinery. What is new is the refusal: below the floor, or on too small a
sample, sizing falls back to the fixed fraction and says why.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from argus.eval.observatory import Prediction, expected_calibration_error

KELLY_FRACTION = Decimal("0.5")
"""Half-Kelly. Full Kelly is growth-optimal only if the edge estimate is exact, and it never is."""

MAX_FRACTION = Decimal("0.25")
"""No single position takes more than a quarter of the book, whatever the arithmetic says."""

FIXED_FRACTION = Decimal("0.05")
"""The fallback when confidence cannot be trusted. Deliberately modest."""

MAX_ECE = 0.15
"""Above this, stated confidence is not tracking outcomes well enough to size on."""

MIN_GRADED = 20
"""Calibration on fewer graded outcomes is noise; the scorecard uses 5 to *report* a number, which
is a lower bar than betting size on it."""


@dataclass(frozen=True)
class CalibrationGate:
    """Whether the desk's confidence has earned the right to set position size."""

    passed: bool
    reason: str
    ece: float | None
    graded: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "ece": None if self.ece is None else round(self.ece, 4),
            "graded_outcomes": self.graded,
        }


def calibration_gate(predictions: Sequence[Prediction]) -> CalibrationGate:
    """Decide whether confidence may drive size.

    Two ways to fail, kept distinct because they call for different responses. Too few graded
    outcomes is a wait; a poor calibration error is a fix.
    """
    if len(predictions) < MIN_GRADED:
        return CalibrationGate(
            passed=False,
            reason=(
                f"{len(predictions)} graded outcome(s), {MIN_GRADED} needed before confidence "
                f"may set size"
            ),
            ece=None,
            graded=len(predictions),
        )
    ece = expected_calibration_error(list(predictions))
    if ece > MAX_ECE:
        return CalibrationGate(
            passed=False,
            reason=(
                f"expected calibration error {ece:.3f} exceeds {MAX_ECE}; sizing on this "
                f"confidence would lever a bias rather than express an edge"
            ),
            ece=ece,
            graded=len(predictions),
        )
    return CalibrationGate(
        passed=True,
        reason=f"calibration error {ece:.3f} over {len(predictions)} graded outcomes",
        ece=ece,
        graded=len(predictions),
    )


def kelly_fraction(win_probability: float, payoff: Decimal) -> Decimal:
    """``f* = (p·b - q) / b``, floored at zero.

    A negative result means the bet has no edge. Returning zero rather than the negative number is
    deliberate: a negative Kelly is an instruction to take the other side, and this function sizes a
    decision that has already been made rather than reversing it.
    """
    if payoff <= 0:
        return Decimal("0")
    p = Decimal(str(win_probability))
    if not (Decimal("0") <= p <= Decimal("1")):
        raise ValueError(f"win probability {win_probability} is not a probability")
    q = Decimal("1") - p
    raw = (p * payoff - q) / payoff
    return max(Decimal("0"), raw)


@dataclass(frozen=True)
class Sizing:
    """What fraction of the book to commit, and on what basis."""

    fraction: Decimal
    basis: str
    gate: CalibrationGate
    raw_kelly: Decimal | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "fraction": str(self.fraction),
            "basis": self.basis,
            "raw_kelly": None if self.raw_kelly is None else str(self.raw_kelly),
            "gate": self.gate.as_dict(),
        }


def size(
    *,
    win_probability: float,
    payoff: Decimal,
    predictions: Sequence[Prediction],
    risk_multiplier: Decimal = Decimal("1"),
    session_multiplier: Decimal = Decimal("1"),
) -> Sizing:
    """Size a position, using confidence only if it has earned the right to be used.

    ``risk_multiplier`` is the circuit breaker's de-risking ladder
    (:func:`argus.risk.circuit.risk_multiplier`), applied last so that a halted book sizes to zero
    however good the arithmetic looks.

    ``session_multiplier`` is :func:`argus.risk.session_risk.throttle` — the measured volatility of
    the path the position will actually live through, against a regular-hours baseline. It is a
    separate argument rather than folded into ``risk_multiplier`` because the two answer different
    questions and a single number could not be attributed: the circuit breaker reacts to **our**
    drawdown, the session throttle to **the market's** clock. Both may only reduce; neither may
    raise a position above what the arithmetic above produced.
    """
    if risk_multiplier < 0 or session_multiplier < 0:
        raise ValueError("a risk multiplier may not be negative")
    if session_multiplier > 1:
        raise ValueError(
            "the session throttle may only reduce; a multiplier above one would let a risk layer "
            "add exposure, which is the invariant agents/desk.py enforces on the Constitution"
        )
    applied = risk_multiplier * session_multiplier
    gate = calibration_gate(predictions)
    if not gate.passed:
        return Sizing(
            fraction=min(MAX_FRACTION, FIXED_FRACTION * applied),
            basis=(
                f"fixed fraction; confidence not usable ({gate.reason})"
                + (f"; session throttle x{session_multiplier}" if session_multiplier < 1 else "")
            ),
            gate=gate,
        )

    raw = kelly_fraction(win_probability, payoff)
    staked = raw * KELLY_FRACTION * applied
    return Sizing(
        fraction=min(MAX_FRACTION, staked),
        basis=(
            f"half-Kelly on calibrated confidence ({gate.reason})"
            + (f"; session throttle x{session_multiplier}" if session_multiplier < 1 else "")
        ),
        gate=gate,
        raw_kelly=raw,
    )


__all__ = [
    "FIXED_FRACTION",
    "KELLY_FRACTION",
    "MAX_ECE",
    "MAX_FRACTION",
    "MIN_GRADED",
    "CalibrationGate",
    "Sizing",
    "calibration_gate",
    "kelly_fraction",
    "size",
]
