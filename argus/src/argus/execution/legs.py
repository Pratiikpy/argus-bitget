"""Multi-leg order coordination — the second of two small execution-plumbing gaps the product plan
names as blocking Cross-Asset Execution Agent (Track 2) and arbitrage feasibility (Track 1):
*"no idempotency, no multi-leg orders"*. Idempotency shipped earlier the same session
(`execution.orders.deterministic_client_order_id`); this closes the second.

**Read first, per standing rule #3.** `nautilus_trader`'s own model
(`crates/model/src/enums.rs:586-593`) names a ``ContingencyType`` for linked orders — ``Oco``
(one-cancels-other), ``Oto`` (one-triggers-other), ``Ouo`` (one-updates-other, proportional
quantity). **Deliberately not reused here, and the reason is stated rather than hidden**: all three
model *sequential/conditional* linkage between orders that are not meant to be live and filling at
the same time. What a cross-asset execution agent actually needs — a token leg and an equity-linked
leg (or a spread's two legs) submitted **simultaneously**, each capable of filling independently of
the other — is a different problem FIX's own taxonomy does not name: **legging risk**, the state
where one leg has filled and the other has not, leaving real, unhedged, naked directional exposure
for as long as the imbalance persists. This module measures that risk directly and answers exactly
one question a coordinator needs: has this group been unevenly filled for too long to keep waiting?

**Deliberately not a new order-execution engine.** `execution.orders.Order`/`OrderBook` already own
the single-instrument lifecycle (submit, fill, cancel, reconcile) and nothing here duplicates it —
:class:`LegGroup` is a thin, read-only overlay that references already-tracked `Order` objects by
identity, the same relationship `desk.book.HedgeLink` has to `Book`'s positions (informational,
never a second source of truth for state a caller must keep in sync by hand).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from argus.execution.orders import Order, OrderState

_FILLING_STATES = frozenset({OrderState.PARTIALLY_FILLED, OrderState.FILLED})
_DEAD_STATES = frozenset({
    OrderState.DENIED, OrderState.REJECTED, OrderState.CANCELLED, OrderState.EXPIRED,
    OrderState.VOIDED,
})


class LegGroupError(RuntimeError):
    """A `LegGroup` cannot honestly be constructed from what was supplied."""


@dataclass(frozen=True, slots=True)
class Leg:
    """One order inside a coordinated group, plus what it is *for* — not a new order type, a
    reference to an existing one."""

    order: Order
    role: str
    """Free text (``"primary"``, ``"hedge"``, ``"long_leg"``...) rather than an enum: the roles a
    spread needs vary by strategy, and boxing this into a closed set would make the caller invent
    a role that doesn't fit rather than name the real one."""

    weight: Decimal = Decimal("1")
    """This leg's intended size ratio relative to the group's unit — ``1`` for a 1:1 pair,
    something else for a ratio spread (two of A against one of B). Applied only in
    :func:`leg_exposure_notional`; the order's own `quantity` is never rescaled by it."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "client_order_id": self.order.client_order_id, "role": self.role,
            "weight": str(self.weight), "state": str(self.order.state),
            "filled_quantity": str(self.order.filled_quantity),
        }


@dataclass(frozen=True, slots=True)
class LegGroup:
    """Two or more orders meant to be filled together. Read-only over orders an
    `execution.orders.OrderBook` already owns — this never transitions an order itself."""

    group_id: str
    legs: tuple[Leg, ...]
    purpose: str = ""

    def __post_init__(self) -> None:
        if len(self.legs) < 2:
            raise LegGroupError(
                f"{self.group_id}: a leg group needs at least two legs; one order is not a "
                f"coordination problem"
            )
        ids = [leg.order.client_order_id for leg in self.legs]
        if len(set(ids)) != len(ids):
            raise LegGroupError(
                f"{self.group_id}: the same order appears in this group more than once"
            )

    @property
    def all_filled(self) -> bool:
        return all(leg.order.state is OrderState.FILLED for leg in self.legs)

    @property
    def none_filled(self) -> bool:
        return all(leg.order.filled_quantity == 0 for leg in self.legs)

    @property
    def is_legging(self) -> bool:
        """Uneven fill exists right now — the state this module exists to measure and time-box.

        Neither "nothing has happened yet" (:attr:`none_filled`) nor "the group completed as
        designed" (:attr:`all_filled`) is legging; both are the honest edges of the same
        arithmetic, not special-cased away.
        """
        return not self.none_filled and not self.all_filled

    @property
    def any_leg_dead(self) -> bool:
        """A leg reached a terminal non-fill state (denied, rejected, cancelled, expired, voided)
        while carrying zero fill — it will never complete its side of the coordination, ever,
        which is a stronger and more urgent fact than an ordinary legging timeout."""
        return any(
            leg.order.state in _DEAD_STATES and leg.order.filled_quantity == 0
            for leg in self.legs
        )

    def first_fill_at(self) -> datetime | None:
        """The instant the FIRST leg first carried any fill — read from each order's own append-
        only `history`, never tracked as separate caller-supplied state that could drift from it.

        ``None`` while :attr:`none_filled` is true; that is the honest "imbalance has not started"
        case, not a missing measurement.
        """
        stamps = [
            transition.at
            for leg in self.legs
            for transition in leg.order.history
            if transition.to in _FILLING_STATES
        ]
        return min(stamps) if stamps else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id, "purpose": self.purpose,
            "legs": [leg.as_dict() for leg in self.legs],
            "all_filled": self.all_filled, "none_filled": self.none_filled,
            "is_legging": self.is_legging, "any_leg_dead": self.any_leg_dead,
        }


class LegExposureError(RuntimeError):
    """A leg's exposure cannot be honestly priced — surfaced rather than silently read as zero."""


def leg_exposure_notional(group: LegGroup, marks: Mapping[str, Decimal]) -> Decimal:
    """Net signed notional currently exposed from this group's filled legs, weight-adjusted.

    A group designed to offset (a long leg and a short leg of comparable size) reads near zero once
    both sides have filled; a group with only one side filled reads the full one-sided notional —
    the naked exposure legging risk actually is. Signed: positive means net long, negative net
    short, so a caller can tell not just *how much* exposure exists but which *direction* it needs
    to hedge or unwind.

    Raises rather than returning zero when a FILLED-or-partially-filled leg's mark price is
    missing — a silent zero here would understate real risk exactly when a caller most needs the
    true figure, the same discipline `desk.book.Book.total_unrealized_pnl` uses for missing marks
    at the aggregate level (there, missing coverage is reported to the caller instead; here, with
    only a handful of legs per group, refusing outright is the more useful failure).
    """
    total = Decimal("0")
    for leg in group.legs:
        if leg.order.filled_quantity == 0:
            continue
        mark = marks.get(leg.order.symbol)
        if mark is None:
            raise LegExposureError(
                f"{group.group_id}: leg {leg.order.client_order_id} ({leg.order.symbol}) has "
                f"filled quantity but no mark price was supplied for it"
            )
        # `Order.side` is stored exactly as the caller passed it — no normalisation happens in
        # `Order.__post_init__`. Both real call sites store it uppercase (`agents/desk.py:704`
        # upper-cases before constructing the live order; `execution/preflight.py:123` passes the
        # literal "BUY"), unlike `desk.book.Lot.side`, which is lowercase throughout that module.
        # `.upper()` here matches the real `Order` convention and stays safe if a future caller
        # passes lowercase instead.
        direction = Decimal("1") if leg.order.side.upper() == "BUY" else Decimal("-1")
        total += direction * leg.order.filled_quantity * leg.weight * mark
    return total


@dataclass(frozen=True, slots=True)
class LeggingVerdict:
    """What to do about this group's current fill imbalance, and why."""

    group_id: str
    demands: str
    """``"continue"`` (no action — either nothing has filled, everything has, or the imbalance is
    still inside its allowed window) or ``"unwind"`` (cancel the residual of any still-working leg
    and flatten whatever has filled — the imbalance has run too long, or a leg died with the
    group only partly filled)."""

    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"group_id": self.group_id, "demands": self.demands, "reason": self.reason}


def assess_legging_risk(
    group: LegGroup, *, now: datetime, max_legging_seconds: int,
) -> LeggingVerdict:
    """Has this group been unevenly filled for longer than it is allowed to be?

    Evaluated in severity order, same discipline as `risk.circuit.assess` and
    `agents.desk.ConstitutionPolicy.rule`: a dead leg with partial fill elsewhere demands UNWIND
    outright, before the ordinary time-box is even checked — waiting out a clock for a leg that
    can structurally never complete only compounds the naked exposure it already carries.
    """
    if group.any_leg_dead and not group.none_filled:
        return LeggingVerdict(
            group.group_id, "unwind",
            "a leg reached a terminal non-fill state while the group carries a fill elsewhere; "
            "it cannot complete as designed and the naked exposure must not be left open",
        )
    if not group.is_legging:
        why = "nothing has filled yet" if group.none_filled else "every leg is filled"
        return LeggingVerdict(group.group_id, "continue", f"no imbalance: {why}")
    first_fill = group.first_fill_at()
    assert first_fill is not None  # is_legging implies at least one fill exists
    elapsed = (now - first_fill).total_seconds()
    if elapsed > max_legging_seconds:
        return LeggingVerdict(
            group.group_id, "unwind",
            f"legging for {elapsed:.0f}s, past the {max_legging_seconds}s window this group is "
            f"allowed to wait for the remaining leg(s)",
        )
    return LeggingVerdict(
        group.group_id, "continue",
        f"legging for {elapsed:.0f}s, still inside the {max_legging_seconds}s allowed window",
    )


__all__ = [
    "Leg",
    "LegExposureError",
    "LegGroup",
    "LegGroupError",
    "LeggingVerdict",
    "assess_legging_risk",
    "leg_exposure_notional",
]
