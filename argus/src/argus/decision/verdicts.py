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

**m-of-K agreement, stated rather than implied (added 2026-09-25, item S23).** The analyst panel
reaches the Meta-PM as a plurality — ``Panel.consensus`` picks the most common signal — and a
plurality hides how much agreement produced it: one bullish analyst beside two neutral ones and
three bullish analysts out of three both read "bullish". GoEmotions turns noisy multi-rater labels
into trusted ones with a single stated knob, ``CheckAgreement(ex, min_agreement, ...)``: keep a
label only if at least *m* raters chose it, and report the result at 1+, 2+ and 3+ side by side
(``google-research/goemotions`` ``analyze_data.py:63-67`` and ``:222-243``, Apache-2.0; licence
text and the "Used in" record at ``argus/licenses/google-research-goemotions-APACHE-2.0.txt``).
:func:`m_of_k` and :func:`agreement_ladder` are that rule, adapted: changed so the labels are the
two mutually exclusive directions rather than independent emotions, which means both directions
reaching *m* is a *split* and yields no call — a multi-label keep-both would be a panel voting for
a long and a short at once. Neutral and insufficient-evidence votes count in *K* and for neither
side, because an analyst who looked and saw nothing is a real vote against acting.

The vote is information, not authority. It never creates or reverses a decision — nothing here
takes an :class:`Intent` and returns a different one — and :func:`backs` only says whether the
panel's agreement reached the intent's side. Whether a threshold should *bind*, as a reduction, is
a question for `eval/decision_primitives.py`'s measurement against realised moves, not for a
constant chosen here.

**Measured, and it loses (2026-09-26).** The panels' individual signals were never persisted, so the
evaluation reconstructs them from each decision's conflict and panel lines — every assignment
consistent with `agents/conflict.py`'s rules — and scores a panel only where all of them give the
same call; graded against the settled move with `eval/shadow.py`'s dead zone. Over 641 recorded
decisions: 1-of-K was right 47.6% of the time (99 of 208 graded calls), 2-of-K 45.1% (88 of 195),
3-of-K 42.1% (8 of 19); the desk's own plurality 49.0% (175 of 357); calling "up" every time 52.9%
(309 of 584). Decisiveness falls as the threshold rises — 53% of determinable panels call a
direction at 1-of-K, 42% at 2-of-K, 6% at 3-of-K, none at 4-of-K — and accuracy does not rise to pay
for it: on this record the panel's direction is not predictive at any threshold, and every
threshold trails the plurality and the always-up base rate. So the vote stays what it is here, a
transparent reading of how much the panel agreed, and nothing binds on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Final, Literal, Protocol


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


Direction = Literal["up", "down"]

UP_SIGNALS: Final = frozenset({"bullish", "long", "buy", "positive"})
DOWN_SIGNALS: Final = frozenset({"bearish", "short", "sell", "negative"})
"""The two sides, spelled the ways `agents/conflict.py`'s ``_opposed`` already accepts them, so a
vote and a directional conflict can never disagree about which signals oppose."""

DEFAULT_AGREEMENT: Final = 2
"""The stated threshold: a direction is called when at least two analysts chose it.

Fixed on 2026-09-25, before `eval/decision_primitives.py` graded any threshold against a realised
move, as the smallest number of analysts that is agreement rather than one opinion. Panels here
hold two to four analysts, so 2-of-K is a majority on the small ones and a coalition on the large
ones. Every threshold is reported beside it (:func:`agreement_ladder`) so a reader who prefers
another can read that one instead of trusting this."""


def direction_of(signal: str) -> Direction | None:
    """The side a signal votes for, or ``None`` for neutral, insufficient evidence or unknown."""
    low = signal.strip().lower()
    if low in UP_SIGNALS:
        return "up"
    if low in DOWN_SIGNALS:
        return "down"
    return None


@dataclass(frozen=True, slots=True)
class AgreementVote:
    """What *m*-of-*K* agreement says about one panel at one threshold."""

    threshold: int
    """*m*: votes a direction needs before it is called."""

    voters: int
    """*K*: every analyst that returned a view, directional or not."""

    up: int
    down: int

    @property
    def abstaining(self) -> int:
        """Neutral and insufficient-evidence votes. They count in *K* and for neither side."""
        return self.voters - self.up - self.down

    @property
    def split(self) -> bool:
        """Both directions reached the threshold. Possible only when *m* is at most half of *K*."""
        return self.up >= self.threshold and self.down >= self.threshold

    @property
    def call(self) -> Direction | None:
        """The direction at least *m* analysts agree on, unless the other side also got there."""
        if self.split:
            return None
        if self.up >= self.threshold:
            return "up"
        if self.down >= self.threshold:
            return "down"
        return None

    @property
    def decisive(self) -> bool:
        return self.call is not None

    def as_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold, "voters": self.voters, "up": self.up, "down": self.down,
            "abstaining": self.abstaining, "split": self.split, "call": self.call,
        }

    def render(self) -> str:
        outcome = (
            "split" if self.split else self.call if self.call is not None else "no call"
        )
        return f"{self.threshold}-of-{self.voters}: {outcome}"


def m_of_k(signals: Sequence[str], threshold: int = DEFAULT_AGREEMENT) -> AgreementVote:
    """GoEmotions' ``CheckAgreement`` over directions (see the module docstring for the change).

    ``signals`` are the analysts' own labels, one per view. A threshold below one would call a
    direction nobody chose, so it is refused rather than clamped.
    """
    if threshold < 1:
        raise ValueError("an agreement threshold below 1 calls a direction nobody chose")
    sides = [direction_of(s) for s in signals]
    return AgreementVote(
        threshold=threshold,
        voters=len(sides),
        up=sum(1 for s in sides if s == "up"),
        down=sum(1 for s in sides if s == "down"),
    )


def agreement_ladder(signals: Sequence[str]) -> tuple[AgreementVote, ...]:
    """The vote at every threshold from 1 to *K*, side by side — GoEmotions' 1+/2+/3+ report
    (``analyze_data.py:222-243``), so decisiveness can be read against consensus strength."""
    return tuple(m_of_k(signals, m) for m in range(1, len(signals) + 1))


def backs(intent: Intent, vote: AgreementVote) -> bool | None:
    """Whether the panel's *m*-of-*K* call is on the side the intent acts on.

    ``None`` when the intent opens no exposure — an abstention has no side to back — and a
    no-call or a split is ``False``: the intent acts where the panel did not agree to.
    """
    if not intent.verdict.opens_exposure:
        return None
    wanted: Direction = "up" if intent.side is Side.BUY else "down"
    return vote.call == wanted


def agreement_note(signals: Sequence[str], *, threshold: int = DEFAULT_AGREEMENT) -> str:
    """One line for the decision record: the tally, and the call at every threshold.

    The stated threshold is marked, so the line reads as a policy with its alternatives shown
    rather than as one number. It contains none of the phrases `paper/runner.py` flags on.
    """
    ladder = agreement_ladder(signals)
    if not ladder:
        return "[agreement] no analyst returned a view; no direction can be called"
    first = ladder[0]
    rungs = "; ".join(
        vote.render() + (" (stated threshold)" if vote.threshold == threshold else "")
        for vote in ladder
    )
    return (
        f"[agreement] {first.voters} analyst(s): {first.up} up, {first.down} down, "
        f"{first.abstaining} neither — {rungs}"
    )


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
