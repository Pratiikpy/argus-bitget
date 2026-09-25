"""Fill confirmation: neither the order-status response nor the position is trusted on its own.

**The defect, read in source before this module existed.** Every path in ARGUS that talks to a real
venue decides what happened to an order from a single read of a single signal, taken immediately:

* `execution/bitget_client.py` ``reconcile`` maps one ``/api/v2/mix/order/detail`` response onto
  an :class:`~argus.execution.orders.OrderState` and returns it. Nothing reads the position.
* `execution/preflight.py` ``_order_round_trip`` placed a market order and called ``reconcile`` on
  the very next line, so a venue that had not yet indexed the order returned an empty detail,
  mapped to UNKNOWN, and the probe failed on a timing accident rather than on the signed path it
  exists to prove. It now polls through :func:`confirm_fill`.
* `demo/flow.py:322-323` does the same and prints whatever state the one read returned as the
  execution leg's outcome.

A status response and a position are two different records kept by the venue, and they can
disagree. Three of the ways they disagree are the expensive ones: the status says FILLED while the
position never moved (the book believes in exposure it does not have); the status says REJECTED
while the position did move (exposure the book believes it does not have, which is the dangerous
direction because nothing will ever close it); and the status is empty right after placement while
the order is still being indexed (a timing accident read as an outcome).

What was taken, from which file, under which licence
----------------------------------------------------

SWE-bench (``SWE-bench/SWE-bench``, MIT, Copyright (c) 2023 Carlos E Jimenez, John Yang, Alexander
Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik R Narasimhan; local clone
``mypr/06_agent_evaluation_and_benchmarks/swe_bench``):

- **No signal is not a pass.** ``swebench/harness/grading.py:155-160`` refuses to score a run whose
  status map is empty *and* whose log carries no positive evidence the suite executed
  (``SUITE_RAN``, ``:31-46``, where every count must be non-zero), because under ``FAIL_ONLY``
  grading an empty map would otherwise score as "everything passed". Adapted here as
  :attr:`FillVerdict.NO_EVIDENCE`: an order the venue has no readable record of, beside a position
  that did not move, is never booked as a confirmed non-fill. Changed: the positive evidence is a
  recognisable venue status rather than a regex over a log.
- **Two independent signals must agree.** ``grading.py:162-175`` cross-checks the parsed test
  statuses against the test command's own exit code and invalidates the run when they disagree,
  rather than trusting either. Adapted here as the settle condition: the status's filled quantity,
  signed by the order's side, must equal the observed position delta. Changed: a disagreement is
  not discarded as invalid; it is recorded as its own verdict, :attr:`FillVerdict.DISAGREED`, and
  its own order state, :attr:`~argus.execution.orders.OrderState.DISAGREED`, because on a venue the
  exposure behind a disagreement is real and has to stay visible to the risk layer.

OSWorld (``xlang-ai/OSWorld``, Apache-2.0, Copyright 2024 XLANG NLP Lab; local clone
``mypr/06_agent_evaluation_and_benchmarks/osworld``; licence and "Used in" record at
``argus/licenses/osworld-APACHE-2.0.txt``):

- **Settle by polling against a deadline, and require both signals on the same read.**
  ``desktop_env/controllers/setup.py:100-129`` (``_wait_for_chrome_ws_ready``) polls until a
  deadline and returns only when two readiness signals match on one read (the stderr marker's GUID
  equals the live endpoint's GUID), keeping ``last_state`` so the failure says what was last seen.
  Adapted from ``setup.py:100-129`` in :func:`confirm_fill`, changed: the wall clock is an injected
  monotonic clock; the fixed one-second sleep is an exponential backoff capped at ``max_wait_s``
  and never allowed to overshoot the deadline; running out of time returns a typed verdict
  carrying every observation instead of raising, because a verdict the order book records is worth
  more than an exception that loses the trace; and one matching read is not enough — the agreement
  must hold on a re-read ``stable_for_s`` later, for the reason :func:`confirm_fill` gives.
- **The state right after an action is not the state to judge.** ``desktop_env/desktop_env.py:
  458-466`` runs ``evaluator.postconfig`` between the agent's last action and the evaluation read,
  an explicit admission that the two differ. Here that gap is the order being indexed and the
  position being booked.

Rejected, and why: OSWorld's own ``step`` sleeps a fixed ``pause`` and then reads
(``desktop_env.py:453``); a fixed sleep is either too short on a slow venue or wasted on a fast one,
and it is the pattern this module replaces. SWE-bench's per-format log parsers have no analogue.

**What this module does not do.** It does not attribute a position delta among several live orders
on one symbol: the delta is what moved, and two orders sharing it cannot be told apart from the
position alone. :func:`confirm_in_book` refuses in that case rather than guessing. And it does not
read a position from Bitget: the ARGUS client speaks the v2 mix API, and no local source documents
the response of its position endpoint (the v3 ``getPositionInfo`` entry in
`agent-sdk/src/generated/catalog.ts` names the path and query but no response fields). Until a
verified read exists, a Bitget confirmation can reach :attr:`FillVerdict.UNCORROBORATED` and never
:attr:`FillVerdict.CONFIRMED`, and it says so.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol

from argus.execution.orders import OrderBook, OrderState

_ZERO = Decimal("0")

TERMINAL_STATES: frozenset[OrderState] = frozenset({
    OrderState.FILLED,
    OrderState.CANCELLED,
    OrderState.REJECTED,
    OrderState.EXPIRED,
    OrderState.DENIED,
    OrderState.VOIDED,
})
"""States after which the venue will change nothing more about the order.

DENIED is here because the paper venue (`paper/venue.py`) reports a decision the risk layer refused
as DENIED with nothing filled; a real venue never reports it."""


class ConfirmationError(RuntimeError):
    """A confirmation was asked for in a form that cannot be answered honestly."""


class VenueReadError(RuntimeError):
    """One read of one signal failed. Recorded in the trace and retried until the deadline."""


@dataclass(frozen=True, slots=True)
class StatusReading:
    """What the venue's order record says, reduced to the two facts the settle check needs.

    ``state`` is ``None`` when the venue returned nothing recognisable: an empty detail, an order it
    has no record of, or a state word no mapping knows. That is absence of evidence, and it is kept
    apart from every real state so it can never be read as one.
    """

    state: OrderState | None
    filled: Decimal
    """Unsigned quantity the venue says has filled. Zero when ``state`` is ``None``."""

    raw: str = ""
    """The venue's own word for the state, kept for the trace."""


class StatusSource(Protocol):
    """The venue's order record, read by client order id."""

    def order_status(self, client_order_id: str, *, symbol: str) -> StatusReading: ...


class PositionSource(Protocol):
    """The venue's position record: the signed net quantity held on a symbol."""

    def position(self, symbol: str) -> Decimal: ...


class FillVerdict(StrEnum):
    """What the two signals established by the deadline. Closed set; each maps to one book state."""

    CONFIRMED = "confirmed"
    """Both signals agree the whole order filled."""

    CONFIRMED_PARTIAL = "confirmed_partial"
    """Both agree on a terminal fill smaller than the order: the residual is work to re-plan."""

    CONFIRMED_UNFILLED = "confirmed_unfilled"
    """Both agree nothing filled and the venue has finished with the order."""

    OVERFILLED = "overfilled"
    """Both agree on more than the order's quantity: a venue or accounting defect, not absorbed."""

    DISAGREED = "disagreed"
    """The status and the position contradict each other when time ran out."""

    UNSETTLED = "unsettled"
    """Consistent so far, but the order was still working at the deadline."""

    UNCORROBORATED = "uncorroborated"
    """Only one signal could be read. Never booked as a fill."""

    NO_EVIDENCE = "no_evidence"
    """The venue never produced a readable status. Never booked as a non-fill."""

    @property
    def is_confirmed(self) -> bool:
        return self in {
            FillVerdict.CONFIRMED, FillVerdict.CONFIRMED_PARTIAL, FillVerdict.CONFIRMED_UNFILLED,
        }


@dataclass(frozen=True, slots=True)
class Observation:
    """One poll: what each signal said, and why that was or was not settled."""

    attempt: int
    elapsed_s: float
    state: str
    status_filled: str | None
    position_delta: str | None
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "elapsed_s": round(self.elapsed_s, 4),
            "state": self.state,
            "status_filled": self.status_filled,
            "position_delta": self.position_delta,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class FillConfirmation:
    """The verdict, both signals' last readings, and the whole poll trace behind it."""

    client_order_id: str
    symbol: str
    side: str
    quantity: Decimal
    """The order's unsigned quantity: what a full fill would be."""

    verdict: FillVerdict
    status_state: OrderState | None
    status_filled: Decimal | None
    position_delta: Decimal | None
    observations: tuple[Observation, ...]
    reason: str

    @property
    def is_confirmed(self) -> bool:
        return self.verdict.is_confirmed

    @property
    def order_state(self) -> OrderState:
        """The state the order book should record. A single signal never produces a final state."""
        if self.verdict is FillVerdict.CONFIRMED:
            return OrderState.FILLED
        if self.verdict in (FillVerdict.CONFIRMED_PARTIAL, FillVerdict.CONFIRMED_UNFILLED):
            if self.status_state is None:  # pragma: no cover - settle requires a terminal state
                raise ConfirmationError("a confirmed verdict without a status state")
            return self.status_state
        if self.verdict in (FillVerdict.DISAGREED, FillVerdict.OVERFILLED):
            return OrderState.DISAGREED
        if self.verdict is FillVerdict.UNSETTLED and self.status_state is not None:
            return self.status_state
        return OrderState.UNKNOWN

    def render(self) -> str:
        filled = "unread" if self.status_filled is None else str(self.status_filled)
        delta = "unread" if self.position_delta is None else str(self.position_delta)
        state = "none" if self.status_state is None else self.status_state.value
        return (
            f"[fill] {self.client_order_id} {self.side} {self.quantity} {self.symbol}: "
            f"{self.verdict.value} after {len(self.observations)} read(s) — status {state} "
            f"filled {filled}, position moved {delta}. {self.reason}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": str(self.quantity),
            "verdict": self.verdict.value,
            "order_state": self.order_state.value,
            "status_state": None if self.status_state is None else self.status_state.value,
            "status_filled": None if self.status_filled is None else str(self.status_filled),
            "position_delta": None if self.position_delta is None else str(self.position_delta),
            "reads": len(self.observations),
            "observations": [o.as_dict() for o in self.observations],
            "reason": self.reason,
        }


def direction(side: str) -> Decimal:
    """+1 for a buy, -1 for a sell. Anything else is refused rather than guessed.

    `paper/ledger.py` ``settle`` learned this the expensive way: an unrecognised side that fell
    through to one branch inverted every P&L sign silently.
    """
    word = side.strip().upper()
    if word in ("BUY", "LONG"):
        return Decimal("1")
    if word in ("SELL", "SHORT"):
        return Decimal("-1")
    raise ConfirmationError(f"side {side!r} is neither a buy nor a sell; refusing to sign a fill")


@dataclass(frozen=True, slots=True)
class _Read:
    status: StatusReading | None
    status_error: str
    delta: Decimal | None
    position_error: str


def _read(
    status: StatusSource, position: PositionSource | None, *, client_order_id: str, symbol: str,
    position_before: Decimal | None,
) -> _Read:
    """Read both signals once. A failure of one is recorded and never stops the other."""
    reading: StatusReading | None = None
    status_error = ""
    try:
        reading = status.order_status(client_order_id, symbol=symbol)
        if reading.state is OrderState.UNKNOWN:
            # A state word nothing maps is not a state. Kept as absence, with the word preserved.
            reading = StatusReading(state=None, filled=_ZERO, raw=reading.raw or "unknown")
    except Exception as exc:  # a venue read may fail in any way; the trace keeps what it was
        status_error = f"{type(exc).__name__}: {str(exc)[:160]}"
    delta: Decimal | None = None
    position_error = ""
    if position is None:
        position_error = "no position source on this venue"
    elif position_before is None:
        position_error = "the position before the order was not read"
    else:
        try:
            delta = position.position(symbol) - position_before
        except Exception as exc:
            position_error = f"{type(exc).__name__}: {str(exc)[:160]}"
    return _Read(reading, status_error, delta, position_error)


def _settled(read: _Read, sign: Decimal) -> bool:
    """Terminal status, readable position, and the two agree on the signed filled quantity."""
    return (
        read.status is not None
        and read.status.state in TERMINAL_STATES
        and read.delta is not None
        and read.delta == sign * read.status.filled
    )


def _observe(attempt: int, elapsed: float, read: _Read, sign: Decimal) -> Observation:
    if read.status is None:
        state = "error"
    elif read.status.state is None:
        state = "none"
    else:
        state = read.status.state.value
    filled = None if read.status is None else str(read.status.filled)
    delta = None if read.delta is None else str(read.delta)
    if _settled(read, sign):
        note = "settled: terminal status and the position agree"
    elif read.status is None:
        note = f"status unreadable ({read.status_error})"
    elif read.status.state is None:
        note = f"the venue returned no recognisable order record ({read.status.raw or 'empty'})"
    elif read.status.state not in TERMINAL_STATES:
        note = f"order still working ({read.status.state.value})"
    elif read.delta is None:
        note = f"position unreadable ({read.position_error})"
    else:
        note = f"status fill {sign * read.status.filled} and position delta {read.delta} disagree"
    return Observation(attempt, elapsed, state, filled, delta, note)


def _verdict_settled(read: _Read, quantity: Decimal) -> tuple[FillVerdict, str]:
    if read.status is None:  # pragma: no cover - _settled requires a status
        raise ConfirmationError("a settled read without a status")
    filled = read.status.filled
    # Zero first: an order of zero (nothing authorised) that filled nothing is a confirmed
    # non-fill, not a "full fill" of nothing.
    if filled == 0:
        return (
            FillVerdict.CONFIRMED_UNFILLED,
            "the venue finished with the order and neither record shows a fill",
        )
    if filled == quantity:
        return FillVerdict.CONFIRMED, "the order record and the position agree on a full fill"
    if filled < quantity:
        return (
            FillVerdict.CONFIRMED_PARTIAL,
            f"both records agree {filled} of {quantity} filled; the residual "
            f"{quantity - filled} is work to re-plan",
        )
    return (
        FillVerdict.OVERFILLED,
        f"both records agree {filled} filled against an order of {quantity}; an overfill is a "
        f"venue or accounting defect and is not absorbed",
    )


def _verdict_at_deadline(read: _Read) -> tuple[FillVerdict, str]:
    """Classify the last read when the deadline ran out before the signals settled."""
    status = read.status
    if status is None or status.state is None:
        if read.delta is not None and read.delta != 0:
            return (
                FillVerdict.DISAGREED,
                f"the position moved by {read.delta} while the venue shows no record of the "
                f"order; that exposure is real and unattributed",
            )
        return (
            FillVerdict.NO_EVIDENCE,
            "the venue never returned a readable status; an order with no record and a position "
            "that did not move is absence of evidence, not a confirmed non-fill",
        )
    if read.delta is None:
        return (
            FillVerdict.UNCORROBORATED,
            f"the status reads {status.state.value} with {status.filled} filled and the position "
            f"could not be read ({read.position_error}); one signal is not a confirmation",
        )
    if status.state in TERMINAL_STATES:
        return (
            FillVerdict.DISAGREED,
            f"the venue finished with the order ({status.state.value}, {status.filled} filled) "
            f"but the position moved by {read.delta}; the two records contradict each other",
        )
    return (
        FillVerdict.UNSETTLED,
        f"the order was still {status.state.value} at the deadline with {status.filled} filled "
        f"and the position moved by {read.delta}; nothing about it is final yet",
    )


def confirm_fill(
    *,
    status: StatusSource,
    position: PositionSource | None,
    client_order_id: str,
    symbol: str,
    side: str,
    quantity: Decimal,
    position_before: Decimal | None,
    deadline_s: float = 30.0,
    first_wait_s: float = 0.25,
    max_wait_s: float = 4.0,
    stable_for_s: float = 1.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> FillConfirmation:
    """Poll both signals until they settle or the deadline passes. Never a fixed sleep.

    ``position_before`` is the signed position on ``symbol`` read *before* the order was sent; the
    delta from it is the position signal. ``None`` (the read failed, or the venue has none) leaves
    the confirmation at best :attr:`FillVerdict.UNCORROBORATED`.

    ``quantity`` may be zero, meaning nothing was authorised to fill; a settled read then confirms
    that nothing did, and any fill both signals agree on is :attr:`FillVerdict.OVERFILLED`.

    ``stable_for_s`` is how long an agreement must hold before it is believed: after the first
    agreeing read, one more read is taken that much later, and it must agree on the same numbers.
    **This was not in the first draft, and the fault injection is what showed it had to be.** A
    venue that reports REJECTED at once and books the position half a second later produces a
    first read on which the two records *agree* — nothing filled, nothing moved — and a one-read
    settle confirmed a non-fill while the exposure was arriving. OSWorld's readiness check can
    trust one matching read because a GUID does not arrive late; a position can. A position that
    lags by more than ``stable_for_s`` is still missed, and nothing bounded can promise otherwise.
    Zero means one agreeing read is enough, which is right only where both records are derived
    from one durable write, as on the paper ledger (`paper/venue.py`).

    ``deadline_s=0`` takes exactly one read, which is what a replay over recorded state wants.
    """
    if quantity < 0:
        raise ConfirmationError(f"order quantity {quantity} is negative; the side carries the sign")
    if deadline_s < 0 or first_wait_s <= 0 or max_wait_s < first_wait_s or stable_for_s < 0:
        raise ConfirmationError(
            f"poll parameters must satisfy deadline >= 0, 0 < first wait <= max wait and "
            f"stability >= 0, got {deadline_s}/{first_wait_s}/{max_wait_s}/{stable_for_s}"
        )
    sign = direction(side)
    # Without a position source, or without the position before the order, the second signal can
    # never arrive; once the status is final there is nothing left to wait for.
    can_corroborate = position is not None and position_before is not None
    start = clock()
    wait = first_wait_s
    observations: list[Observation] = []
    attempt = 0
    agreed_at: float | None = None
    agreed_on: tuple[OrderState | None, Decimal, Decimal | None] | None = None
    while True:
        attempt += 1
        read = _read(
            status, position, client_order_id=client_order_id, symbol=symbol,
            position_before=position_before,
        )
        elapsed = clock() - start
        observations.append(_observe(attempt, elapsed, read, sign))
        settled = _settled(read, sign)
        if settled and read.status is not None:
            now_on = (read.status.state, read.status.filled, read.delta)
            if agreed_at is None or now_on != agreed_on:
                agreed_at, agreed_on = elapsed, now_on
        else:
            agreed_at, agreed_on = None, None
        stable = agreed_at is not None and (
            elapsed - agreed_at >= stable_for_s
            or (elapsed >= deadline_s and len(observations) > 1 and elapsed > agreed_at)
        )
        final_alone = (
            not can_corroborate
            and read.status is not None
            and read.status.state in TERMINAL_STATES
        )
        if stable or final_alone or elapsed >= deadline_s:
            if stable:
                verdict, reason = _verdict_settled(read, quantity)
            elif settled:
                seen = _verdict_settled(read, quantity)[1]
                verdict, reason = (
                    FillVerdict.UNSETTLED,
                    f"the two records agreed only on the read taken at the deadline; an agreement "
                    f"seen once, with no time to hold, is not believed ({seen})",
                )
            else:
                verdict, reason = _verdict_at_deadline(read)
            return FillConfirmation(
                client_order_id=client_order_id,
                symbol=symbol,
                side=side.strip().upper(),
                quantity=quantity,
                verdict=verdict,
                status_state=None if read.status is None else read.status.state,
                status_filled=None if read.status is None else read.status.filled,
                position_delta=read.delta,
                observations=tuple(observations),
                reason=reason,
            )
        # Once the records agree, the next read is the one that checks the agreement held. Never
        # sleep past the deadline: the last read happens at it, not after it.
        pause = wait if agreed_at is None else stable_for_s - (elapsed - agreed_at)
        sleep(min(pause, max(deadline_s - elapsed, 0.0)))
        if agreed_at is None:
            wait = min(wait * 2, max_wait_s)


def confirm_in_book(
    book: OrderBook,
    *,
    client_order_id: str,
    status: StatusSource,
    position: PositionSource | None,
    position_before: Decimal | None,
    at: datetime,
    deadline_s: float = 30.0,
    first_wait_s: float = 0.25,
    max_wait_s: float = 4.0,
    stable_for_s: float = 1.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> FillConfirmation:
    """Confirm one submitted order and record the verdict in the book.

    Refuses when another live order shares the symbol: the position delta would then be the sum of
    both, and attributing it to either one would be a guess this module is written not to make.
    """
    order = book.get(client_order_id)
    sharing = [
        o.client_order_id for o in book.live_orders()
        if o.symbol == order.symbol and o.client_order_id != client_order_id
    ]
    if sharing:
        raise ConfirmationError(
            f"{client_order_id}: {len(sharing)} other live order(s) on {order.symbol} "
            f"({', '.join(sharing)}); a position delta cannot be attributed between them"
        )
    confirmation = confirm_fill(
        status=status, position=position, client_order_id=client_order_id,
        symbol=order.symbol, side=order.side, quantity=order.quantity,
        position_before=position_before, deadline_s=deadline_s, first_wait_s=first_wait_s,
        max_wait_s=max_wait_s, stable_for_s=stable_for_s, clock=clock, sleep=sleep,
    )
    book.apply_confirmation(
        client_order_id,
        target=confirmation.order_state,
        filled=confirmation.status_filled if confirmation.is_confirmed else None,
        at=at,
        reason=confirmation.render(),
    )
    return confirmation


__all__ = [
    "TERMINAL_STATES",
    "ConfirmationError",
    "FillConfirmation",
    "FillVerdict",
    "Observation",
    "PositionSource",
    "StatusReading",
    "StatusSource",
    "VenueReadError",
    "confirm_fill",
    "confirm_in_book",
    "direction",
]
