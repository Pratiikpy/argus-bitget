"""Decision and Constitution vocabularies, and the asymmetry that holds both halves of Track 2.

Bitget's Track 2 positioning is two requirements that pull against each other:

    "The LLM is the primary trading decision-maker, not just an assistant. The Agent must sense the
     environment, make independent judgments, and autonomously place orders with risk controls."

Most of the field resolves the tension by quietly abandoning one side. Our teardowns found four
systems that keep the LLM off the order path entirely — ``inalpha`` does it by explicit design
(``trade-plan.ts:74-303``), AI-Trader's LLM only writes market-summary prose, Vibe-Trading's system
prompt forbids recommendations, FinRobot has no broker at all. Each is a defensible engineering
choice and each fails this track's positioning rule.

**ARGUS resolves it with an asymmetry rather than a compromise: the Constitution may only reduce.**

It can shrink a trade, demand a hedge, delay it, flatten a position, or refuse outright. It can
never create a trade, never flip a side, never increase size. So the economic choice remains the
LLM's — it chose the direction and the thesis, and no rule invented one for it — while the risk
layer stays genuinely binding rather than advisory.

That property is enforced here by :func:`apply_constraint`, and the enforcement is tested. It is
the difference between "we have a risk layer" and "the risk layer cannot become the real trader".
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Final, Protocol


class Verdict(StrEnum):
    """What the LLM decided. Seven outcomes, because a direction alone is not a decision."""

    TRADE = "trade"
    REDUCE = "reduce"
    HEDGE = "hedge"
    DELAY = "delay"
    NO_TRADE = "no_trade"
    """Evidence was adequate and the answer is no."""

    HUMAN_REVIEW = "human_review"
    DATA_INSUFFICIENT = "data_insufficient"
    """Evidence was **inadequate**. Distinct from NO_TRADE on purpose.

    Conflating the two hides the system's most important failure mode: an agent that cannot tell
    "I looked and the answer is no" from "I could not see" will confidently do the wrong thing the
    moment its data breaks. Level 1 failures land here and nowhere else.
    """

    @property
    def opens_exposure(self) -> bool:
        """Creates *new* risk. Drives the invalidation requirement and the risk checks."""
        return self in {Verdict.TRADE, Verdict.HEDGE}

    @property
    def carries_quantity(self) -> bool:
        """Acts with a size — which is not the same as opening exposure.

        REDUCE is the case that makes this distinction necessary. It lowers risk rather than
        creating it, but "reduce" without a quantity is not an instruction, it is a sentiment.
        A live run caught exactly this: the model returned REDUCE with a size and an earlier
        version of this code silently zeroed it, turning a real decision into a no-op.
        """
        return self in {Verdict.TRADE, Verdict.HEDGE, Verdict.REDUCE}

    @property
    def is_abstention(self) -> bool:
        """Abstentions are scored economically, not assumed good — see Abstention Value."""
        return self in {Verdict.NO_TRADE, Verdict.DELAY, Verdict.DATA_INSUFFICIENT}


class ConstitutionVerdict(StrEnum):
    """What the risk layer did to the LLM's decision. Every one of these is a *reduction*."""

    ALLOW = "allow"
    RESIZE = "resize"
    REQUIRE_HEDGE = "require_hedge"
    DELAY = "delay"
    REJECT = "reject"
    FLATTEN = "flatten"
    """Close existing exposure. The one verdict that can act without an LLM proposal —
    and it can only ever reduce the book toward zero."""


class ConstitutionViolation(RuntimeError):
    """The Constitution attempted something outside its authority.

    This is an internal invariant breach, not a risk event: it means the risk layer tried to act
    like a trader. It must be impossible, and if it ever fires the system halts rather than trades.
    """


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class Intent:
    """A proposed action. Produced by the LLM; narrowed, never authored, by the Constitution."""

    symbol: str
    side: Side
    quantity: Decimal
    verdict: Verdict
    stated_confidence: float
    thesis: str
    invalidation: tuple[str, ...] = ()
    """What would make this thesis wrong. Mandatory for any intent that opens exposure —
    a position with no falsifier cannot be monitored, only hoped over."""

    required_hedge: tuple[str, ...] = ()

    lean: str = "none"
    """The direction the desk would take if forced, stated even when it declines to trade.

    **This exists because the desk had no gradeable directional view and we proved it rather than
    assuming it.** `side` is a required field of the decision contract, so it looked like a view;
    measured over the 53 settled abstentions carrying a counterfactual it was **BUY in all 53**,
    directionally right 3.8% of the time against a base rate of up-moves of exactly 3.8%. A field
    that agrees with the base rate to the decimal carries no information: on a decision it has
    declined to make, the model fills `side` in because the schema demands it.

    An abstention says the edge does not clear the hurdle. It does not say the desk has no view
    about direction, and the hurdle frontier put a number on why that matters: trading beats
    abstaining at 56% directional accuracy, and nothing on record says whether this desk clears it.
    The lean costs no risk and makes the question answerable. ``none`` is a legitimate answer and
    means the direction genuinely cannot be called.
    """

    lean_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.quantity < 0:
            raise ValueError("quantity may not be negative")
        if not 0.0 <= self.stated_confidence <= 1.0:
            raise ValueError("stated_confidence must lie in [0, 1]")
        if self.verdict.carries_quantity and self.quantity <= 0:
            raise ValueError(f"{self.verdict} requires a positive quantity")
        if self.verdict.opens_exposure and not self.invalidation:
            raise ValueError(
                f"{self.verdict} requires at least one invalidation condition: a position "
                f"whose thesis cannot be falsified cannot be monitored, only hoped over"
            )
        if not self.thesis.strip():
            raise ValueError("an intent must carry a thesis")


@dataclass(frozen=True, slots=True)
class ConstitutionRuling:
    """The risk layer's response, with the binding constraint named.

    ``reason`` is machine-readable on purpose: "which constraint bound, how often" is the evidence
    that the risk layer does real work, and free text cannot be aggregated.
    """

    verdict: ConstitutionVerdict
    binding_constraint: str
    reason: str
    resulting_intent: Intent

    def authorise(self, order: Submittable) -> Authorised:
        """Mint the capability that lets an order reach :meth:`OrderBook.submit`.

        The only way to construct an :class:`Authorised`, anywhere. See its docstring for why that
        matters. Raises rather than returning a refusal, because a caller that ignores a returned
        ``None`` is exactly the failure this replaces.
        """
        if self.verdict is ConstitutionVerdict.REJECT:
            raise ConstitutionViolation(
                f"{order.symbol}: the Constitution returned REJECT ({self.binding_constraint}); "
                f"a rejected ruling cannot authorise an order"
            )
        intent = self.resulting_intent
        if order.symbol != intent.symbol:
            raise ConstitutionViolation(
                f"authorisation is for {intent.symbol}, order is for {order.symbol}"
            )
        if order.side.strip().lower() != str(intent.side).strip().lower():
            raise ConstitutionViolation(
                f"{order.symbol}: the Constitution approved {intent.side}, the order says "
                f"{order.side}"
            )
        if order.quantity != intent.quantity:
            raise ConstitutionViolation(
                f"{order.symbol}: the Constitution approved quantity {intent.quantity}, the order "
                f"carries {order.quantity} — an authorisation is for a size, not a direction"
            )
        return Authorised(order, self, _token=_AUTHORISATION)


class Submittable(Protocol):
    """The parts of an order this module needs to check an authorisation against.

    A structural type rather than an import of :class:`argus.execution.orders.Order`, so the
    decision layer stays free of the execution layer. Execution obeys decision; the dependency runs
    that way and not the other.
    """

    @property
    def symbol(self) -> str: ...
    @property
    def side(self) -> str: ...
    @property
    def quantity(self) -> Decimal: ...


_AUTHORISATION: Final = object()
"""The capability. Module-private, held by nothing outside this file.

An object identity rather than a string or a flag, because a string can be guessed, copied out of a
traceback, or arrived at by accident; a private object cannot be obtained without already having a
reference to it, and the only code holding one is :meth:`ConstitutionRuling.authorise`."""


class Authorised:
    """An order the Constitution actually approved. Unconstructable by hand.

    **The asymmetry used to be true only because every caller remembered to check.** The invariants
    — never increase, never reverse a side, never turn an abstention into a trade — are enforced in
    :func:`apply_constraint` and proved across 2,177,280 swept states, and all of that binds only if
    you go *through* ``apply_constraint``. Nothing forced you to. ``OrderBook.submit`` accepted a
    bare ``Order``, and two call sites in this repository already built one and submitted it having
    never consulted the Constitution at all. A new code path that forgot the check was a naked
    directional order, and the sweep would not have caught it, because the sweep drives
    ``ConstitutionPolicy`` and not ``OrderBook``.

    So the check moved from discipline into the type system. ``submit`` now accepts only this, this
    can only be minted by :meth:`ConstitutionRuling.authorise`, and that method holds the sole
    reference to a module-private sentinel. Forgetting the Constitution is no longer a bug that
    reaches the venue; it is a ``TypeError`` at the call site.

    **And the authorisation is bound to the content, not merely to the act.** ``authorise`` refuses
    unless the order's symbol, side and quantity match the ruling's ``resulting_intent`` exactly.
    That is deliberately stricter than a token alone: on 2026-09-20 this project found two live
    ledger rows recording ``quantity: 1`` against a ruling of ``quantity_after: 0``. A bare
    capability would have waved those through, because a ruling *was* obtained — it simply was not
    the one the order carried.

    Read from Ritapossible/Ballast's ``ballast/enforcer.py``, which is where the pattern was taken
    from: a module-private ``_ADMISSION`` sentinel, an ``Admitted`` whose ``__init__`` raises unless
    it is handed that object, and an executor that accepts nothing else. What is added here is the
    content binding above, and the fact that the same property is asserted inside the whole-domain
    sweep rather than only in unit tests.
    """

    __slots__ = ("order", "ruling")

    def __init__(self, order: Submittable, ruling: ConstitutionRuling, *, _token: object) -> None:
        if _token is not _AUTHORISATION:
            raise ConstitutionViolation(
                "an Authorised order cannot be constructed directly; it is minted only by "
                "ConstitutionRuling.authorise(), which is the whole point of the type"
            )
        self.order = order
        self.ruling = ruling

    def __repr__(self) -> str:
        return (
            f"Authorised({self.order.symbol} {self.order.side} {self.order.quantity} "
            f"under {self.ruling.verdict})"
        )


def apply_constraint(
    original: Intent,
    *,
    verdict: ConstitutionVerdict,
    binding_constraint: str,
    reason: str,
    resized_quantity: Decimal | None = None,
    required_hedge: tuple[str, ...] = (),
) -> ConstitutionRuling:
    """Narrow an intent, and refuse structurally to do anything else.

    Every guard below exists because its absence would let the risk layer author an economic
    decision, which would make the LLM a narrator and fail Track 2's positioning rule. These are
    invariants, not policy — violating one is a bug, so it raises rather than logs.
    """
    if verdict is ConstitutionVerdict.ALLOW:
        if resized_quantity is not None and resized_quantity != original.quantity:
            raise ConstitutionViolation("ALLOW cannot change quantity; use RESIZE")
        return ConstitutionRuling(verdict, binding_constraint, reason, original)

    if verdict is ConstitutionVerdict.RESIZE:
        if resized_quantity is None:
            raise ConstitutionViolation("RESIZE requires a resized_quantity")
        if resized_quantity > original.quantity:
            raise ConstitutionViolation(
                f"RESIZE may only reduce: {resized_quantity} > {original.quantity}. The "
                f"Constitution cannot size a position up — that would make it the trader."
            )
        if resized_quantity <= 0:
            raise ConstitutionViolation("RESIZE to zero or below must be expressed as REJECT")
        narrowed = _replace_quantity(original, resized_quantity)
        return ConstitutionRuling(verdict, binding_constraint, reason, narrowed)

    if verdict is ConstitutionVerdict.REQUIRE_HEDGE:
        if not required_hedge:
            raise ConstitutionViolation("REQUIRE_HEDGE must name at least one hedge instrument")
        qty = original.quantity if resized_quantity is None else resized_quantity
        if qty > original.quantity:
            raise ConstitutionViolation("REQUIRE_HEDGE may not increase quantity")
        narrowed = _replace_quantity(original, qty, required_hedge=required_hedge)
        return ConstitutionRuling(verdict, binding_constraint, reason, narrowed)

    if verdict is ConstitutionVerdict.DELAY:
        delayed = _replace_verdict(original, Verdict.DELAY)
        return ConstitutionRuling(verdict, binding_constraint, reason, delayed)

    if verdict is ConstitutionVerdict.REJECT:
        rejected = _replace_verdict(original, Verdict.NO_TRADE, quantity=Decimal("0"))
        return ConstitutionRuling(verdict, binding_constraint, reason, rejected)

    if verdict is ConstitutionVerdict.FLATTEN:
        # The only verdict that acts without an LLM proposal, and it can only move the book
        # toward zero. It never opens new exposure.
        flat = _replace_verdict(original, Verdict.REDUCE)
        return ConstitutionRuling(verdict, binding_constraint, reason, flat)

    raise ConstitutionViolation(f"unhandled constitution verdict: {verdict}")


def _replace_quantity(
    intent: Intent, quantity: Decimal, *, required_hedge: tuple[str, ...] = ()
) -> Intent:
    return Intent(
        symbol=intent.symbol,
        side=intent.side,                    # never flipped
        quantity=quantity,
        verdict=intent.verdict,
        stated_confidence=intent.stated_confidence,
        thesis=intent.thesis,
        invalidation=intent.invalidation,
        required_hedge=required_hedge or intent.required_hedge,
    )


def _replace_verdict(
    intent: Intent, verdict: Verdict, *, quantity: Decimal | None = None
) -> Intent:
    qty = intent.quantity if quantity is None else quantity
    return Intent(
        symbol=intent.symbol,
        side=intent.side,                    # never flipped
        quantity=qty,
        verdict=verdict,
        stated_confidence=intent.stated_confidence,
        thesis=intent.thesis,
        # Invalidation conditions survive a downgrade: they are why the trade was refused or
        # delayed, and dropping them loses the reason on the way to the ledger.
        invalidation=intent.invalidation,
        required_hedge=intent.required_hedge,
    )
