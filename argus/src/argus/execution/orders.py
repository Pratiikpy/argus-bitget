"""Order state machine, modelled on NautilusTrader's and corrected against its actual source.

Reference: ``crates/model/src/enums.rs:1304-1388`` (LGPLv3 — we link the compiled library where we
need its engine and reimplement the state model here; no Nautilus code is vendored).

Three things were taken from reading that file rather than from memory, and each corrects a design
error in ARGUS's first draft:

1. **DENIED is not REJECTED.** Nautilus separates an order refused by its *own* risk engine
   (``Denied``) from one refused by the *venue* (``Rejected``). ARGUS needs this distinction more
   than Nautilus does: "how many trades did the Constitution stop" and "how many did Bitget refuse"
   are different questions, and collapsing them destroys the attribution that Track 2 is judged on.

2. **PENDING_UPDATE and PENDING_CANCEL are open *and* in-flight at once.** The order is working at
   the venue while a modify/cancel request is outstanding. A binary live/not-live flag cannot
   express that, and the risk layer must count the exposure while the request is outstanding.

3. **SUBMITTED must be included when reconciling.** Nautilus documents the trap in a comment at
   ``enums.rs:1340-1343``: a venue that reports a resting order as pending maps it to ``Submitted``,
   and filtering on ``is_open()`` alone *silently drops it from reconciliation*. Our
   :meth:`OrderState.is_live` therefore spans open, in-flight and unknown.

**UNKNOWN is ARGUS's own addition and is labelled as such.** Nautilus has no such state. We add it
because a request that times out has an genuinely unknown outcome — the venue may hold the order,
may have filled it, may never have seen it. Treating a timeout as a rejection and retrying is how
duplicate exposure is created, and it is the failure that stranded fourteen jobs on another
platform of ours: the code assumed a step had happened, nothing errored, and the state sat wrong
forever.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from argus.decision.verdicts import Authorised


class OrderState(StrEnum):
    """Order lifecycle. Values mirror Nautilus's ``OrderStatus`` except where noted."""

    INITIALISED = "initialised"
    """Instantiated inside ARGUS. Not yet seen by anything external."""

    DENIED = "denied"
    """Refused by **our own** Constitution Kernel — never sent to the venue.

    Distinct from REJECTED on purpose. This is the count that proves the risk layer is doing work.
    """

    SUBMITTED = "submitted"
    """Sent to the venue, awaiting acknowledgement. In-flight, not yet working.

    Must be included in reconciliation sweeps: some venues report a resting order as pending, which
    maps here, and filtering it out loses the order.
    """

    ACCEPTED = "accepted"
    """Acknowledged by the venue as received and valid. Working."""

    TRIGGERED = "triggered"
    """A stop order's trigger price was hit at the venue."""

    PENDING_UPDATE = "pending_update"
    """A modify request is outstanding. The order is still working — open *and* in-flight."""

    PENDING_CANCEL = "pending_cancel"
    """A cancel request is outstanding. Still working until the venue confirms."""

    PARTIALLY_FILLED = "partially_filled"
    """Some quantity filled. The residual is work to re-plan, never work to abandon."""

    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    """Reached a GTD expiration."""

    REJECTED = "rejected"
    """Refused by the **venue**."""

    VOIDED = "voided"
    """Terminal after an authoritative venue void or fill correction.

    Rare and important: a venue can retroactively bust a fill. Without this state that correction
    has nowhere to go and the position record silently disagrees with the venue's.
    """

    UNKNOWN = "unknown"
    """**ARGUS addition — not in Nautilus.** We asked and we do not know.

    Reachable only from an in-flight state via timeout. Resolvable only by reconciliation against
    the venue — never by assumption, and never by resubmitting.
    """

    # --- predicates, mirroring Nautilus's is_open / is_closed / is_inflight / is_cancellable ---

    @property
    def is_open(self) -> bool:
        """Working at the venue. Deliberately excludes SUBMITTED, which is in-flight."""
        return self in {
            OrderState.ACCEPTED,
            OrderState.TRIGGERED,
            OrderState.PENDING_UPDATE,
            OrderState.PENDING_CANCEL,
            OrderState.PARTIALLY_FILLED,
        }

    @property
    def is_inflight(self) -> bool:
        """A request to the venue is outstanding."""
        return self in {
            OrderState.SUBMITTED,
            OrderState.PENDING_UPDATE,
            OrderState.PENDING_CANCEL,
        }

    @property
    def is_closed(self) -> bool:
        return self in {
            OrderState.DENIED,
            OrderState.REJECTED,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
            OrderState.FILLED,
            OrderState.VOIDED,
        }

    @property
    def is_cancellable(self) -> bool:
        return self in {
            OrderState.ACCEPTED,
            OrderState.TRIGGERED,
            OrderState.PENDING_UPDATE,
            OrderState.PARTIALLY_FILLED,
        }

    @property
    def is_live(self) -> bool:
        """Does this state imply exposure the risk layer must account for?

        Open **or** in-flight **or** unknown. UNKNOWN counts as live: assuming otherwise leaves a
        position nobody is watching. This is the predicate reconciliation sweeps must use.
        """
        return self.is_open or self.is_inflight or self is OrderState.UNKNOWN


_LEGAL: dict[OrderState, frozenset[OrderState]] = {
    OrderState.INITIALISED: frozenset({OrderState.DENIED, OrderState.SUBMITTED}),
    OrderState.SUBMITTED: frozenset({
        OrderState.ACCEPTED,
        OrderState.REJECTED,
        OrderState.FILLED,            # IOC/market can fill straight through
        OrderState.PARTIALLY_FILLED,
        OrderState.CANCELLED,
        OrderState.UNKNOWN,
    }),
    OrderState.ACCEPTED: frozenset({
        OrderState.TRIGGERED,
        OrderState.PENDING_UPDATE,
        OrderState.PENDING_CANCEL,
        OrderState.PARTIALLY_FILLED,
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.EXPIRED,
        OrderState.VOIDED,
        OrderState.UNKNOWN,
    }),
    OrderState.TRIGGERED: frozenset({
        OrderState.PENDING_UPDATE,
        OrderState.PENDING_CANCEL,
        OrderState.PARTIALLY_FILLED,
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.EXPIRED,
        OrderState.UNKNOWN,
    }),
    OrderState.PENDING_UPDATE: frozenset({
        OrderState.ACCEPTED,
        OrderState.TRIGGERED,
        OrderState.PARTIALLY_FILLED,
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.EXPIRED,
        OrderState.REJECTED,          # the modify itself can be refused
        OrderState.UNKNOWN,
    }),
    OrderState.PENDING_CANCEL: frozenset({
        OrderState.ACCEPTED,          # cancel refused, order still working
        OrderState.CANCELLED,
        OrderState.PARTIALLY_FILLED,  # raced a fill
        OrderState.FILLED,
        OrderState.EXPIRED,
        OrderState.REJECTED,
        OrderState.UNKNOWN,
    }),
    OrderState.PARTIALLY_FILLED: frozenset({
        OrderState.PENDING_UPDATE,
        OrderState.PENDING_CANCEL,
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.EXPIRED,
        OrderState.VOIDED,
        OrderState.UNKNOWN,
    }),
    # Reconciliation is the ONLY exit from UNKNOWN, and it can land wherever the venue says.
    OrderState.UNKNOWN: frozenset({
        OrderState.ACCEPTED,
        OrderState.TRIGGERED,
        OrderState.PARTIALLY_FILLED,
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
        OrderState.VOIDED,
    }),
    # Terminal. VOIDED is reachable from FILLED because a venue can bust a fill after the fact.
    OrderState.FILLED: frozenset({OrderState.VOIDED}),
    OrderState.DENIED: frozenset(),
    OrderState.CANCELLED: frozenset(),
    OrderState.REJECTED: frozenset(),
    OrderState.EXPIRED: frozenset(),
    OrderState.VOIDED: frozenset(),
}


class IllegalTransition(RuntimeError):
    """An order was moved between states the venue protocol does not permit."""


class DuplicateOrder(RuntimeError):
    """A client order id was submitted twice.

    Challenge Mode's 'duplicate order' control targets exactly this: a replayed authorisation must
    never produce two positions.
    """


class UnauthorisedOrder(RuntimeError):
    """No approved-intent hash, or one matching no Constitution verdict."""


def deterministic_client_order_id(
    *, market_state_hash: str, approved_intent_hash: str, side: str, quantity: Decimal
) -> str:
    """The idempotency key that closes the gap this module's own docstring names.

    **The defect, verified in source before this function existed.** `client_order_id` was built
    as ``f"{decision_id}-1"`` where ``decision_id`` came from ``f"paper-{symbol}-{int(now.
    timestamp())}"`` (`paper/runner.py`) — wall-clock at the moment of the call. This module's own
    docstring already names the exact consequence: *"treating a timeout as a rejection and
    retrying is how duplicate exposure is created."* :class:`OrderState.UNKNOWN` exists precisely
    because a request can time out with the venue's outcome unresolved — and the next cycle, or a
    restarted process, would re-evaluate the same market state and propose the same order under a
    **freshly wall-clock-stamped, therefore different**, ``client_order_id``. Neither
    :class:`OrderBook`'s own duplicate guard below nor Bitget's server-side ``clientOid`` dedup
    (`agent-sdk/src/generated/catalog.ts`, confirmed live) can catch a duplicate whose id changes
    on every attempt.

    **The fix is determinism, not detection.** Two calls that are genuinely the same decision must
    produce the identical id; two calls that are genuinely different decisions must not collide.
    The four inputs are chosen for exactly that:

    * ``market_state_hash`` — which market instant this decision was made for
      (``MarketFrame.state_hash()``). Identical on a true retry of the same cycle; different on a
      later cycle with fresh evidence.
    * ``approved_intent_hash`` — the Constitution-approved intent's content
      (``proof.approved_intent_hash``).
    * ``side`` and ``quantity`` **after** mandate narrowing, not before — `agents/desk.py` computes
      ``approved_intent_hash`` prior to applying a trader mandate, and two different mandates can
      narrow the same approved intent to two different final sizes. Hashing only the pre-mandate
      approval would collide two orders that are legitimately different, which `OrderBook.submit`
      would then wrongly refuse as a duplicate.

    Bitget's own ``clientOid`` is documented at 64 characters maximum
    (`agent-sdk/src/generated/catalog.ts`, e.g. the transfer endpoints); the digest below is well
    inside that bound by construction.
    """
    payload = f"{market_state_hash}:{approved_intent_hash}:{side}:{quantity}"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
    return f"argus-{digest}"


@dataclass(frozen=True, slots=True)
class Transition:
    at: datetime
    frm: OrderState
    to: OrderState
    reason: str


@dataclass(slots=True)
class Order:
    """One order and its complete, append-only history.

    The history is the audit artefact. Reconstructing what happened must never require reading
    logs, because "did the approved order become the executed order" has to be answerable
    mechanically — that is Execution Proof, one of the five Track-2 proof systems.
    """

    client_order_id: str
    symbol: str
    side: str
    quantity: Decimal
    approved_intent_hash: str
    """Binds this order to the Constitution verdict that authorised it."""

    state: OrderState = OrderState.INITIALISED
    venue_order_id: str | None = None
    filled_quantity: Decimal = Decimal("0")
    history: list[Transition] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("order quantity must be positive")
        if not self.approved_intent_hash.strip():
            raise UnauthorisedOrder(
                f"{self.client_order_id}: no approved_intent_hash. An order that cannot be traced "
                f"to a Constitution verdict is unauthorised however well-formed it looks."
            )

    def transition(self, to: OrderState, *, at: datetime, reason: str) -> None:
        if to not in _LEGAL[self.state]:
            raise IllegalTransition(
                f"{self.client_order_id}: {self.state} -> {to} is not a legal venue transition"
            )
        self.history.append(Transition(at=at, frm=self.state, to=to, reason=reason))
        self.state = to

    def apply_fill(self, quantity: Decimal, *, at: datetime) -> None:
        if quantity <= 0:
            raise ValueError("fill quantity must be positive")
        if self.filled_quantity + quantity > self.quantity:
            raise ValueError(
                f"{self.client_order_id}: fills ({self.filled_quantity + quantity}) would exceed "
                f"order quantity ({self.quantity}) — an overfill is a venue or accounting bug, "
                f"never something to absorb silently"
            )
        self.filled_quantity += quantity
        target = (
            OrderState.FILLED
            if self.filled_quantity == self.quantity
            else OrderState.PARTIALLY_FILLED
        )
        if self.state is not target:
            self.transition(target, at=at, reason=f"fill {quantity}")

    @property
    def residual(self) -> Decimal:
        """Unfilled quantity. A partial fill leaves work to re-plan, not work to abandon."""
        return self.quantity - self.filled_quantity


class OrderBook:
    """Tracks orders, refuses duplicates, and resolves unknowns only by reconciliation."""

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}

    def deny(self, order: Order, *, at: datetime, reason: str) -> Order:
        """The Constitution refused this order. It never reaches the venue."""
        order.transition(OrderState.DENIED, at=at, reason=reason)
        self._orders[order.client_order_id] = order
        return order

    def submit(self, authorised: Authorised, *, at: datetime) -> Order:
        """Send an order the Constitution approved. There is no other overload.

        **The signature is the control.** This used to take a bare :class:`Order`, and the
        may-only-reduce guarantee therefore held only for callers who remembered to obtain a ruling
        first — which every production path did, and two evaluation paths in this repository did
        not. A capability cannot be forgotten: an order that never met the Constitution cannot be
        expressed here, because :class:`Authorised` cannot be constructed without one.

        See :class:`argus.decision.verdicts.Authorised` for why the capability is also bound to the
        order's symbol, side and quantity rather than to the mere fact that some ruling existed.
        """
        order = authorised.order
        if not isinstance(order, Order):  # pragma: no cover - structural guard
            raise TypeError(f"an Authorised must carry an Order, not {type(order).__name__}")
        if order.client_order_id in self._orders:
            raise DuplicateOrder(
                f"{order.client_order_id} already submitted — a replayed authorisation must not "
                f"create a second position"
            )
        order.transition(OrderState.SUBMITTED, at=at, reason="submitted to venue")
        self._orders[order.client_order_id] = order
        return order

    def mark_unknown(self, client_order_id: str, *, at: datetime) -> None:
        """A request timed out. The outcome is unknown — and unknown is not rejected."""
        self._orders[client_order_id].transition(
            OrderState.UNKNOWN, at=at, reason="request timed out; outcome unknown"
        )

    def reconcile(
        self,
        client_order_id: str,
        *,
        venue_state: OrderState,
        at: datetime,
        venue_filled: Decimal | None = None,
    ) -> None:
        """Resolve an order against what the venue actually holds.

        The only legitimate exit from UNKNOWN. Venue-reported fills are adopted rather than argued
        with — the venue is the source of truth for execution.
        """
        order = self._orders[client_order_id]
        if venue_filled is not None and venue_filled != order.filled_quantity:
            if venue_filled > order.quantity:
                raise ValueError(
                    f"{client_order_id}: venue reports {venue_filled} filled against an order of "
                    f"{order.quantity} — refusing to reconcile an impossible state"
                )
            order.filled_quantity = venue_filled
        order.transition(venue_state, at=at, reason="reconciled against venue")

    def live_orders(self) -> list[Order]:
        """Everything carrying exposure: open, in-flight, or unknown.

        Using ``is_open`` alone here is the documented Nautilus trap — a venue reporting a resting
        order as pending maps it to SUBMITTED, and it vanishes from the sweep.
        """
        return [o for o in self._orders.values() if o.state.is_live]

    def needs_reconciliation(self) -> list[Order]:
        return [o for o in self._orders.values() if o.state is OrderState.UNKNOWN]

    def get(self, client_order_id: str) -> Order:
        return self._orders[client_order_id]

    def __len__(self) -> int:
        return len(self._orders)
