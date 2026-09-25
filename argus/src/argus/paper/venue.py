"""The paper ledger as a venue: an order record and a position record, read back from disk.

ARGUS's paper desk never sends an order to an exchange; the hash-chained ledger *is* its venue. That
makes it tempting to treat ``ledger.record(...)`` returning an entry as proof the position exists —
the paper equivalent of trusting an order-status response. It is not proof, for the same reason:

* **The return value is what the writer believes it wrote.** The 2026-09-12 incident had two cycles
  each holding a stale head, each appending a *different* decision as seq 41, each returning its own
  entry. Both writers were told they had written seq 41; the file held both.
* **The row is not the book.** A position is what the ledger holds across rows, after settlements.
  A row can be right and the aggregate wrong (a duplicate row, a settlement that never landed), and
  the reverse.
* **Neither is the authorisation.** Ledger seq 264 and 265 recorded ``quantity: 1`` against a
  Constitution ruling of ``quantity_after: 0`` (`paper/corrections.py`). The row and the book agreed
  with each other perfectly; both disagreed with what the risk layer had allowed.

So a recorded decision is confirmed three ways, with :func:`argus.execution.confirm.confirm_fill`
doing the first two exactly as it would against a real venue: the **order record** (the row for
this sequence number, re-read from the file on disk, not taken from the writer's return value)
against the **position record** (the symbol's open exposure, re-derived from every row on disk
before and after), and the agreed fill against the **authorised ceiling** (the Constitution's own
ruling, which the protocol and the venue guard may only have reduced).

The paper venue settles instantly — nothing is in flight — so on the live path the poll ends on its
first read unless another writer is mid-append. The code path is nonetheless the one a real venue
uses, which is the point: a check that exists only for the exchange is a check nobody has exercised.

**Positions come from raw row quantities, not from the voided-row overlay.** `paper/corrections.py`
marks seq 264 and 265 as abstentions for every *reader of results*; a venue's position record is
what the ledger holds, and it holds those rows. Reading the overlay here would hide the very defect
the authorisation check exists to surface.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from argus.execution.confirm import (
    FillConfirmation,
    FillVerdict,
    PositionSource,
    StatusReading,
    StatusSource,
    VenueReadError,
    confirm_fill,
    direction,
)
from argus.execution.orders import OrderState

PAPER_ORDER_PREFIX = "paper-seq-"

FILL_FLAG = "fill not confirmed"
"""Written into a desk note when the ledger's two records disagree. `paper/runner.py` flags it."""

CEILING_FLAG = "exceeds its authorisation"
"""Written into a desk note when a recorded position is larger than the Constitution allowed."""

_ZERO = Decimal("0")


def paper_order_id(seq: int) -> str:
    """The paper venue's order id for a ledger row. The sequence number is the order."""
    return f"{PAPER_ORDER_PREFIX}{seq}"


def _seq_of(client_order_id: str) -> int:
    if not client_order_id.startswith(PAPER_ORDER_PREFIX):
        raise VenueReadError(f"{client_order_id!r} is not a paper order id")
    try:
        return int(client_order_id[len(PAPER_ORDER_PREFIX):])
    except ValueError as exc:
        raise VenueReadError(f"{client_order_id!r} carries no sequence number") from exc


def _decimal(raw: object) -> Decimal:
    try:
        return Decimal(str(raw))
    except (InvalidOperation, ValueError) as exc:
        raise VenueReadError(f"{raw!r} is not a quantity") from exc


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Every decision row in the file, as written. A torn line fails the read, never the row.

    A torn final line means another writer is mid-append; the read is refused and the confirmation
    loop reads again, rather than confirming against half a file. A ledger that does not exist yet
    holds nothing, which is a fact about it rather than a failure to read it.
    """
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VenueReadError(f"ledger unreadable: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VenueReadError(f"ledger line {number} is not JSON ({exc.msg})") from exc
        if row.get("kind", "decision") == "decision":
            rows.append(row)
    return rows


def _open_at(row: Mapping[str, Any], at: datetime | None, *, inclusive: bool) -> bool:
    """Does this row carry open exposure at ``at``? ``None`` means as the file stands now."""
    if at is None:
        return row.get("settled_at") is None
    decided = datetime.fromisoformat(str(row["decided_at"]))
    if decided > at or (decided == at and not inclusive):
        return False
    settled = row.get("settled_at")
    return settled is None or datetime.fromisoformat(str(settled)) > at


class LedgerVenue:
    """The ledger as a :class:`~argus.execution.confirm.StatusSource` and ``PositionSource``.

    ``LedgerVenue(path)`` re-reads the file on every call — the live form, where a read must see
    what is durably on disk now. ``LedgerVenue.frozen(rows, as_of=...)`` answers from a fixed
    snapshot as it stood at an instant — the replay form, used to re-check recorded history.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        rows: Sequence[Mapping[str, Any]] | None = None,
        as_of: datetime | None = None,
        inclusive: bool = True,
    ) -> None:
        if (path is None) == (rows is None):
            raise ValueError("a LedgerVenue reads either a path or a snapshot of rows, not both")
        self._path = path
        self._rows = None if rows is None else list(rows)
        self._as_of = as_of
        self._inclusive = inclusive

    @classmethod
    def frozen(
        cls, rows: Sequence[Mapping[str, Any]], *, as_of: datetime | None, inclusive: bool = True,
    ) -> LedgerVenue:
        return cls(rows=rows, as_of=as_of, inclusive=inclusive)

    def _snapshot(self) -> Sequence[Mapping[str, Any]]:
        if self._rows is not None:
            return self._rows
        if self._path is None:  # pragma: no cover - the constructor guarantees one of the two
            raise VenueReadError("a LedgerVenue with neither a path nor rows")
        return read_rows(self._path)

    def order_status(self, client_order_id: str, *, symbol: str) -> StatusReading:
        seq = _seq_of(client_order_id)
        found = [r for r in self._snapshot() if int(r["seq"]) == seq]
        if not found:
            return StatusReading(state=None, filled=_ZERO, raw=f"no row with seq {seq}")
        if len(found) > 1:
            # The 2026-09-12 signature. Which row is "the" order is exactly what cannot be said.
            return StatusReading(
                state=None, filled=_ZERO, raw=f"seq {seq} appears {len(found)} times",
            )
        row = found[0]
        if str(row["symbol"]) != symbol:
            return StatusReading(
                state=None, filled=_ZERO, raw=f"seq {seq} is a {row['symbol']} row, not {symbol}",
            )
        quantity = _decimal(row["quantity"])
        # A paper fill is immediate and complete at the recorded price; a zero row is a decision the
        # risk layer (or the desk itself) refused, which the state machine calls DENIED.
        state = OrderState.FILLED if quantity > 0 else OrderState.DENIED
        return StatusReading(state=state, filled=quantity, raw=str(row.get("verdict", "")))

    def position(self, symbol: str) -> Decimal:
        total = _ZERO
        for row in self._snapshot():
            if str(row["symbol"]) != symbol:
                continue
            quantity = _decimal(row["quantity"])
            if quantity <= 0 or not _open_at(row, self._as_of, inclusive=self._inclusive):
                continue
            total += direction(str(row["side"])) * quantity
        return total


@dataclass(frozen=True, slots=True)
class PaperFillCheck:
    """The two-record confirmation of one ledger row, plus the check against its authorisation."""

    seq: int
    confirmation: FillConfirmation
    authorised_quantity: Decimal | None
    """The Constitution's ruling. ``None`` when there is no ruling to check against."""

    authorised_side: str | None

    @property
    def within_authorisation(self) -> bool | None:
        """Is what the records agree on no larger than what was authorised? ``None``: unknowable.

        Measured on the agreed status fill, and only when the two records agreed — a disagreement
        is already an alarm, and the ceiling of a quantity nobody can pin down is not a finding.
        """
        if self.authorised_quantity is None or not self.confirmation.is_confirmed:
            return None
        filled = self.confirmation.status_filled or _ZERO
        if filled == 0:
            return True
        if self.authorised_side is not None and (
            direction(self.authorised_side) != direction(self.confirmation.side)
        ):
            return False
        return filled <= self.authorised_quantity

    @property
    def alarm(self) -> bool:
        """The records disagree, or agree on something the Constitution did not allow."""
        return not self.confirmation.is_confirmed or self.within_authorisation is False

    def render(self) -> str:
        c = self.confirmation
        if not c.is_confirmed and c.verdict is not FillVerdict.OVERFILLED:
            return f"[fill] seq {self.seq}: {FILL_FLAG} — {c.render()}"
        ceiling = (
            "no Constitution ruling to check it against" if self.authorised_quantity is None
            else f"the Constitution authorised {self.authorised_quantity}"
        )
        if c.verdict is FillVerdict.OVERFILLED or self.within_authorisation is False:
            return (
                f"[fill] seq {self.seq}: the recorded position {CEILING_FLAG} — the ledger row and "
                f"the book agree on {c.status_filled} {c.side}, {ceiling}"
            )
        return (
            f"[fill] seq {self.seq}: {c.verdict.value} — the ledger row and the book agree on "
            f"{c.status_filled} after {len(c.observations)} read(s); {ceiling}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "verdict": self.confirmation.verdict.value,
            "status_filled": (
                None if self.confirmation.status_filled is None
                else str(self.confirmation.status_filled)
            ),
            "position_delta": (
                None if self.confirmation.position_delta is None
                else str(self.confirmation.position_delta)
            ),
            "authorised_quantity": (
                None if self.authorised_quantity is None else str(self.authorised_quantity)
            ),
            "within_authorisation": self.within_authorisation,
            "alarm": self.alarm,
        }


def position_or_none(venue: PositionSource, symbol: str) -> Decimal | None:
    """The position before an order, or ``None`` when it cannot be read — never a guessed zero."""
    try:
        return venue.position(symbol)
    except VenueReadError:
        return None


def confirm_recorded(
    *,
    seq: int,
    symbol: str,
    side: str,
    quantity: Decimal,
    status: StatusSource,
    position: PositionSource | None,
    position_before: Decimal | None,
    authorised_quantity: Decimal | None,
    authorised_side: str | None,
    deadline_s: float = 2.0,
    first_wait_s: float = 0.05,
    max_wait_s: float = 0.5,
    stable_for_s: float = 0.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> PaperFillCheck:
    """Confirm one recorded decision against the ledger's two records and its authorisation.

    ``side`` and ``quantity`` are what the writer *intended* to record. ``status`` and
    ``position`` may be two different venues — the replay reads the order record from a cycle's
    own log and the position from the file — or the same one on the live path.

    ``stable_for_s`` defaults to zero here, and only here: both of the ledger's records are derived
    from one durable write that finished before the first read, so there is no second service
    whose booking can lag the first. A real venue keeps the default of
    :func:`~argus.execution.confirm.confirm_fill`.
    """
    confirmation = confirm_fill(
        status=status, position=position, client_order_id=paper_order_id(seq), symbol=symbol,
        side=side, quantity=quantity, position_before=position_before, deadline_s=deadline_s,
        first_wait_s=first_wait_s, max_wait_s=max_wait_s, stable_for_s=stable_for_s,
        clock=clock, sleep=sleep,
    )
    return PaperFillCheck(
        seq=seq, confirmation=confirmation, authorised_quantity=authorised_quantity,
        authorised_side=authorised_side,
    )


__all__ = [
    "CEILING_FLAG",
    "FILL_FLAG",
    "PAPER_ORDER_PREFIX",
    "LedgerVenue",
    "PaperFillCheck",
    "confirm_recorded",
    "paper_order_id",
    "position_or_none",
    "read_rows",
]
