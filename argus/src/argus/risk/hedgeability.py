"""The hedgeability surface — a ranked menu and a priced residual, never a fraction.

The original ARGUS specification treated hedgeability as a single number: what share of risk is
hedgeable right now. That is directionally right and scientifically weak, because it collapses
three different questions — *which* instrument, at *what* cost, with *what* certainty — into one
scalar that cannot be acted on.

The replacement computes, per candidate instrument:

    risk_reduction
  x correlation_confidence
  x liquidity_availability
  x execution_probability
  x basis_stability
  ------------------------
  - execution_cost
  - collateral_cost
  - model_uncertainty
    ------------------------
  = economically hedgeable risk

and emits a **ranked menu** plus an explicitly **priced residual**. The headline metric is
**Risk Neutralisation Efficiency** — marginal risk removed per unit of all-in cost — which lets the
agent's chosen hedge be scored against the best feasible one. That comparison is an ablation
nobody in the corpus runs, and it converts "did the LLM choose well?" into a measurable question.

Verified context from the teardowns: **zero of the four closest tokenized-equity competitors
compute hedgeable-now versus carried in any form**, binary or continuous, and Bastion — the closest
system of all — has no session awareness at all. The empty-menu case below is therefore not an edge
case; for 65.5 hours a week it is the normal state of the world, and it is the state nothing else
models.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from argus.risk.effectiveness import HedgeEffectiveness, Provenance

_ZERO = Decimal("0")
_ONE = Decimal("1")


def _unit(name: str, value: Decimal) -> Decimal:
    if not _ZERO <= value <= _ONE:
        raise ValueError(f"{name} must lie in [0, 1]; got {value}")
    return value


@dataclass(frozen=True, slots=True)
class HedgeCandidate:
    """One instrument that might absorb some of the risk, with every factor stated separately.

    Nothing here is allowed to default optimistically. A missing estimate is an argument the caller
    must make explicitly, because an unstated assumption of perfect correlation confidence is how a
    hedge menu becomes a wish list.
    """

    instrument: str
    risk_reduction: Decimal
    """Fraction of the position's risk this instrument would remove if it executed perfectly."""

    correlation_confidence: Decimal
    """How much we trust the correlation estimate *in the current regime*.

    Distinct from the correlation itself. A 0.71 correlation measured across a shut session is a
    stale print, not a relationship — which is why the cross-market engine reports a usable hedge
    ratio rather than a correlation.
    """

    liquidity_availability: Decimal
    execution_probability: Decimal
    """Probability the hedge actually fills at a usable price. Zero when the venue is shut."""

    basis_stability: Decimal
    execution_cost_bps: Decimal
    collateral_cost_bps: Decimal = _ZERO
    model_uncertainty_bps: Decimal = _ZERO
    """A penalty for the estimate itself being shaky. Charged in bps so it competes with real
    costs on the same scale rather than being argued about qualitatively."""

    provenance: Provenance = Provenance.ASSUMED
    """Whether the three statistical factors above were measured or asserted.

    Defaults to ASSUMED, which is the pessimistic label and therefore the safe one: a caller that
    forgets to say where its numbers came from is recorded as having asserted them. The fields were
    asserted for the whole life of this module before `risk/effectiveness.py` existed — three
    constants typed into the source of a risk layer whose project rule is *never guess* — and this
    field is what makes the difference visible in the record instead of only in the diff.
    """

    evidence: str = ""
    """One line naming the sample behind a MEASURED candidate: phase, observations, window."""

    def __post_init__(self) -> None:
        if self.provenance is Provenance.MEASURED and not self.evidence:
            raise ValueError(
                "a measured candidate must name its sample; an unattributed measurement is an "
                "assertion with a better label"
            )
        _unit("risk_reduction", self.risk_reduction)
        _unit("correlation_confidence", self.correlation_confidence)
        _unit("liquidity_availability", self.liquidity_availability)
        _unit("execution_probability", self.execution_probability)
        _unit("basis_stability", self.basis_stability)
        for name in ("execution_cost_bps", "collateral_cost_bps", "model_uncertainty_bps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} may not be negative")

    @property
    def effective_risk_reduction(self) -> Decimal:
        """Risk actually expected to be removed, after every way the hedge can disappoint.

        Multiplicative on purpose: these are independent ways to fail, and a hedge that is 90%
        likely to fill on an instrument we are 90% confident about in a book with 90% of normal
        depth removes appreciably less risk than any single factor suggests.
        """
        return (
            self.risk_reduction
            * self.correlation_confidence
            * self.liquidity_availability
            * self.execution_probability
            * self.basis_stability
        )

    @property
    def all_in_cost_bps(self) -> Decimal:
        return self.execution_cost_bps + self.collateral_cost_bps + self.model_uncertainty_bps

    @property
    def risk_neutralisation_efficiency(self) -> Decimal:
        """Risk removed per bp of all-in cost. The metric the menu is ranked by.

        A free hedge that removes nothing is not efficient, it is irrelevant — so zero effective
        reduction scores zero regardless of cost.
        """
        if self.effective_risk_reduction == _ZERO:
            return _ZERO
        if self.all_in_cost_bps == _ZERO:
            # A genuinely costless, effective hedge. Rare; ranked first without dividing by zero.
            return Decimal("Infinity")
        return self.effective_risk_reduction / self.all_in_cost_bps

    @property
    def is_placeable(self) -> bool:
        return self.execution_probability > _ZERO and self.liquidity_availability > _ZERO


@dataclass(frozen=True, slots=True)
class HedgeabilitySurface:
    """The ranked menu, the residual, and the reason the residual exists."""

    candidates: tuple[HedgeCandidate, ...]
    session_note: str = ""

    @property
    def menu(self) -> tuple[HedgeCandidate, ...]:
        """Placeable candidates, best Risk Neutralisation Efficiency first."""
        placeable = [c for c in self.candidates if c.is_placeable]
        return tuple(
            sorted(
                placeable,
                key=lambda c: (
                    # Infinity sorts first; Decimal comparison handles it directly.
                    c.risk_neutralisation_efficiency,
                    c.effective_risk_reduction,
                ),
                reverse=True,
            )
        )

    @property
    def best(self) -> HedgeCandidate | None:
        menu = self.menu
        return menu[0] if menu else None

    @property
    def is_empty(self) -> bool:
        """No hedge is placeable. For 65.5 hours a week this is the normal state of the world."""
        return not self.menu

    def residual_after(self, chosen: HedgeCandidate | None) -> Decimal:
        """Fraction of risk carried unhedged. Always priced, never ignored.

        With no hedge chosen the residual is the whole position — which is the honest description
        of a token held naked across a shut anchor session.
        """
        if chosen is None:
            return _ONE
        return _ONE - chosen.effective_risk_reduction

    def efficiency_of_choice(self, chosen: HedgeCandidate | None) -> Decimal:
        """How close the agent's choice came to the best feasible hedge, in [0, 1].

        This is the ablation nobody runs: it scores the *decision*, not the outcome. An agent that
        picks the third-best hedge on a lucky day still picked the third-best hedge.
        """
        best = self.best
        if best is None:
            # Nothing was placeable; declining to hedge is the only correct choice, so it scores 1.
            return _ONE if chosen is None else _ZERO
        if chosen is None:
            return _ZERO
        if best.risk_neutralisation_efficiency in (_ZERO, Decimal("Infinity")):
            return _ONE if chosen.instrument == best.instrument else _ZERO
        ratio = chosen.risk_neutralisation_efficiency / best.risk_neutralisation_efficiency
        return min(_ONE, ratio)


def open_market_candidate(
    instrument: str,
    risk_reduction: Decimal = _ZERO,
    *,
    execution_probability: Decimal = _ZERO,
    execution_cost_bps: Decimal = Decimal("5"),
    measured: HedgeEffectiveness | None = None,
) -> HedgeCandidate:
    """The anchor equity while its market is open: reachable by somebody, not by us.

    This exists because the runner used to hand the desk an *empty* surface whenever the anchor was
    awake, which put a false sentence into the record of every regular-hours decision: "no hedge
    placeable". During regular hours the hedge plainly is placeable — by somebody. What is true is
    narrower and more interesting: **ARGUS has no equity broker**, so the hedge exists, the venue is
    open, the liquidity is there, and we personally have no route to it.

    Those are different facts and a risk record that conflates them is misleading in the direction
    that flatters us: "the market gave me no hedge" excuses an unhedged position, while "I have not
    built the connection" does not.

    So the candidate is constructed with full liquidity and a real cost, and
    ``execution_probability`` defaults to zero for the honest reason rather than the convenient one.
    Supply a non-zero probability the day a broker route exists, and the menu becomes live with no
    other change.

    ``execution_cost_bps`` defaults to 5bps as a plausible all-in retail equity cost. It is
    **NOT VERIFIED** against a broker we have integrated, because we have not integrated one; it is
    here so the menu's arithmetic is exercised with a non-zero number rather than a flattering zero.

    **``measured`` is how the three statistical factors stop being guesses.** Pass a fresh
    :class:`~argus.risk.effectiveness.HedgeEffectiveness` for this instrument's regular-hours phase
    and the risk reduction becomes Ederington's R², the correlation confidence becomes the Fisher
    lower bound, and the basis stability becomes the unit-ratio effectiveness — each with the sample
    that produced it recorded on the candidate. Without it the caller must supply
    ``risk_reduction`` itself and the candidate is stamped ASSUMED.
    """
    if measured is not None:
        return HedgeCandidate(
            instrument=instrument,
            risk_reduction=measured.risk_reduction,
            correlation_confidence=measured.correlation_confidence,
            liquidity_availability=_ONE,
            execution_probability=execution_probability,
            basis_stability=measured.basis_stability,
            execution_cost_bps=execution_cost_bps,
            provenance=Provenance.MEASURED,
            evidence=(
                f"{measured.phase}, n={measured.observations}, {measured.window_days}d, "
                f"measured {measured.measured_at:%Y-%m-%d %H:%MZ}"
            ),
        )
    return HedgeCandidate(
        instrument=instrument,
        risk_reduction=risk_reduction,
        # Asserted, and labelled as asserted. The anchor is the thing the token tracks and while the
        # anchor trades the relationship is observed rather than assumed — but *this constructor*
        # has observed nothing, and 0.98 was a number somebody found plausible.
        correlation_confidence=Decimal("0.98"),
        liquidity_availability=_ONE,
        execution_probability=execution_probability,
        basis_stability=Decimal("0.95"),
        execution_cost_bps=execution_cost_bps,
        provenance=Provenance.ASSUMED,
    )


def shut_market_candidate(
    instrument: str, risk_reduction: Decimal = _ZERO, *,
    measured: HedgeEffectiveness | None = None,
) -> HedgeCandidate:
    """A hedge whose venue is closed: theoretically ideal, practically unavailable.

    Constructing these explicitly matters. The alternative — omitting shut instruments from the
    menu — loses the fact that a perfectly good hedge exists and cannot be reached, which is
    precisely the information the Sleeping-Anchor problem turns on.

    A measurement may be supplied here too, and it changes only what the record says: with
    ``execution_probability`` and ``liquidity_availability`` both zero the effective reduction is
    zero however good the statistics are. That is the point of keeping the factors separate — "this
    hedge would work and cannot be placed" is a different sentence from "this hedge would not work".
    """
    if measured is not None:
        return HedgeCandidate(
            instrument=instrument,
            risk_reduction=measured.risk_reduction,
            correlation_confidence=measured.correlation_confidence,
            liquidity_availability=_ZERO,
            execution_probability=_ZERO,
            basis_stability=measured.basis_stability,
            execution_cost_bps=_ZERO,
            provenance=Provenance.MEASURED,
            evidence=(
                f"{measured.phase}, n={measured.observations}, {measured.window_days}d, "
                f"measured {measured.measured_at:%Y-%m-%d %H:%MZ}"
            ),
        )
    return HedgeCandidate(
        instrument=instrument,
        risk_reduction=risk_reduction,
        correlation_confidence=Decimal("0.95"),
        liquidity_availability=_ZERO,
        execution_probability=_ZERO,
        basis_stability=Decimal("0.9"),
        execution_cost_bps=_ZERO,
        provenance=Provenance.ASSUMED,
    )
