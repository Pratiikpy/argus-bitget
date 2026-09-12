"""Earnings — seven surprises, scored separately, because they disagree.

Track 2's Earnings sub-theme asks how an agent "autonomously interprets earnings / conference calls
and executes". The field default is *EPS beat -> buy*, and it is wrong often enough to be an
anti-pattern we named and banned.

The reason it is wrong is structural, not a matter of degree. A print carries **seven independent
surprises** and they routinely point in opposite directions:

=========================  =============================================================
Surprise                   What it measures
=========================  =============================================================
reported                   the printed number against the prior period
consensus                  the printed number against what analysts expected
guidance                   forward guidance against the prior guide
narrative                  what management says the business is doing
valuation                  what the price already implies
management_credibility     whether this team's prior guides held
qa                         what the Q&A revealed, especially what was refused
=========================  =============================================================

A beat with cut guidance and evasive Q&A is a *different asset* from a beat with raised guidance,
and a single "surprise" scalar destroys that distinction. This module keeps them apart, weights
them by what actually moves price, and refuses to collapse a contradiction into an average.

The measured constraint from this venue applies here as everywhere: an expected move that does not
exceed the 12bps round trip is not a small opportunity.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

ROUND_TRIP_BPS = Decimal("12")

SURPRISES = (
    "reported",
    "consensus",
    "guidance",
    "narrative",
    "valuation",
    "management_credibility",
    "qa",
)

# Weights reflect what moves price, not what is easiest to measure. Guidance dominates because a
# forward cut reprices every future period while a reported beat prices one that has passed; the
# ordering is from the post-earnings-drift literature, not from convenience.
WEIGHTS: dict[str, Decimal] = {
    "reported": Decimal("0.10"),
    "consensus": Decimal("0.20"),
    "guidance": Decimal("0.30"),
    "narrative": Decimal("0.12"),
    "valuation": Decimal("0.13"),
    "management_credibility": Decimal("0.05"),
    "qa": Decimal("0.10"),
}


class EarningsError(ValueError):
    """The decomposition was given something it cannot honestly score."""


@dataclass(frozen=True, slots=True)
class Surprises:
    """Seven scores in [-1, +1]. Positive is bullish for that dimension."""

    reported: Decimal = Decimal("0")
    consensus: Decimal = Decimal("0")
    guidance: Decimal = Decimal("0")
    narrative: Decimal = Decimal("0")
    valuation: Decimal = Decimal("0")
    management_credibility: Decimal = Decimal("0")
    qa: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        for name in SURPRISES:
            value = getattr(self, name)
            if not Decimal("-1") <= value <= Decimal("1"):
                raise EarningsError(f"{name}={value} is outside [-1, 1]")

    def as_dict(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in SURPRISES}

    @property
    def headline(self) -> Decimal:
        """What a naive reader sees: the printed number against consensus.

        Isolated deliberately, because this is the number "EPS beat -> buy" acts on.
        """
        return (self.reported + self.consensus) / Decimal("2")

    @property
    def forward(self) -> Decimal:
        """What the print says about the future — guidance, narrative, credibility, Q&A."""
        parts = (self.guidance, self.narrative, self.management_credibility, self.qa)
        return sum(parts, Decimal("0")) / Decimal(len(parts))

    @property
    def weighted(self) -> Decimal:
        return sum(
            (getattr(self, name) * WEIGHTS[name] for name in SURPRISES), Decimal("0")
        )

    @property
    def is_contradictory(self) -> bool:
        """Does the headline point one way and the forward view the other?

        This is the case the whole module exists for, and the one a single scalar erases.
        """
        if self.headline == 0 or self.forward == 0:
            return False
        return (self.headline > 0) != (self.forward > 0)

    @property
    def dispersion(self) -> Decimal:
        """Spread across the seven. High dispersion means the print is genuinely ambiguous,
        and an ambiguous print deserves a smaller position, not a louder opinion."""
        values = [getattr(self, name) for name in SURPRISES]
        mean = sum(values, Decimal("0")) / Decimal(len(values))
        var = sum(((v - mean) ** 2 for v in values), Decimal("0")) / Decimal(len(values))
        return var.sqrt()


@dataclass(frozen=True, slots=True)
class EarningsRead:
    """The decomposition's verdict, with the contradiction stated rather than averaged away."""

    symbol: str
    surprises: Surprises
    expected_move_bps: Decimal
    confidence: float
    dominant: str
    explanation: str

    @property
    def clears_round_trip(self) -> bool:
        return abs(self.expected_move_bps) > ROUND_TRIP_BPS

    @property
    def direction(self) -> str:
        if not self.clears_round_trip:
            return "neutral"
        return "bullish" if self.expected_move_bps > 0 else "bearish"

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "surprises": self.surprises.as_dict(),
            "headline": str(round(self.surprises.headline, 3)),
            "forward": str(round(self.surprises.forward, 3)),
            "weighted": str(round(self.surprises.weighted, 3)),
            "contradictory": self.surprises.is_contradictory,
            "dispersion": str(round(self.surprises.dispersion, 3)),
            "dominant": self.dominant,
            "expected_move_bps": str(round(self.expected_move_bps, 1)),
            "clears_round_trip": self.clears_round_trip,
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
            "explanation": self.explanation,
        }


def decompose(
    symbol: str,
    surprises: Surprises,
    *,
    typical_move_bps: Decimal = Decimal("300"),
) -> EarningsRead:
    """Turn seven surprises into an expected move, without averaging away a contradiction.

    ``typical_move_bps`` scales a unit surprise to basis points. 300bps is a conservative default
    for a large-cap earnings reaction; it is a scaling constant, not a prediction, and it is stated
    here rather than buried so it can be argued with.
    """
    weighted = surprises.weighted
    expected = weighted * typical_move_bps

    dominant = max(
        SURPRISES,
        key=lambda name: abs(getattr(surprises, name) * WEIGHTS[name]),
    )

    # A contradictory print is genuinely less certain, and dispersion compounds that. Both
    # discounts are applied to *confidence*, never to the direction — shrinking the signal itself
    # would quietly turn a disagreement into a weak consensus.
    confidence = 0.85
    if surprises.is_contradictory:
        confidence *= 0.6
    confidence *= float(max(Decimal("0.4"), Decimal("1") - surprises.dispersion))

    if surprises.is_contradictory:
        head = "beat" if surprises.headline > 0 else "miss"
        fwd = "improving" if surprises.forward > 0 else "deteriorating"
        explanation = (
            f"headline {head}, forward view {fwd} — the print contradicts itself; "
            f"{dominant} dominates the weighting"
        )
    else:
        explanation = f"{dominant} dominates; headline and forward view agree"

    return EarningsRead(
        symbol=symbol,
        surprises=surprises,
        expected_move_bps=expected,
        confidence=confidence,
        dominant=dominant,
        explanation=explanation,
    )


def from_response(symbol: str, response: dict[str, Any]) -> EarningsRead:
    """Build a read from an Earnings Analyst's JSON.

    A missing surprise is scored **zero**, never imputed from the others. Imputing would let one
    strong dimension manufacture six others and turn a thin read into a confident one.
    """
    raw = response.get("surprises") or {}
    values: dict[str, Decimal] = {}
    for name in SURPRISES:
        try:
            values[name] = max(
                Decimal("-1"), min(Decimal("1"), Decimal(str(raw.get(name, 0) or 0)))
            )
        except Exception:
            values[name] = Decimal("0")
    return decompose(symbol, Surprises(**values))


# The case the field gets wrong, kept as a named example because it is the whole argument.
HEADLINE_BEAT_ROTTEN_CORE = Surprises(
    reported=Decimal("0.8"),                 # EPS +12% vs +8% consensus
    consensus=Decimal("0.6"),
    guidance=Decimal("-0.7"),                # FY guidance cut 7%
    narrative=Decimal("-0.4"),               # margin pressure acknowledged
    valuation=Decimal("-0.2"),
    management_credibility=Decimal("-0.3"),
    qa=Decimal("-0.6"),                      # three datacentre questions refused
)


__all__ = [
    "HEADLINE_BEAT_ROTTEN_CORE", "ROUND_TRIP_BPS", "SURPRISES", "WEIGHTS",
    "EarningsError", "EarningsRead", "Surprises", "decompose", "from_response",
]
