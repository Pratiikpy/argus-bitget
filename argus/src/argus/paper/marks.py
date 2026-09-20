"""Short-horizon marks — the observation that makes a refusal gradeable.

The ledger settles a decision **once**, at the hold period, and that settlement is write-once
because the entries are hash-chained. Measured across every settled abstention on 2026-09-15, the
realised horizon was a minimum of 24.02h, a median of 24.57h and a maximum of 36.39h.

**That is the one horizon at which a refusal cannot be scored.** Decisions are taken every two
hours, so a 24-hour window overlaps its neighbour by roughly 92%; the 178 settled abstentions on
record span three calendar days and therefore carry on the order of **three independent
observations**, not 178. Any directional accuracy computed over them is a number with three things
behind it.

So this module records a **second, shorter observation** of the same decision, and it does it
beside the ledger rather than inside it:

* the chain is not touched, and no entry is rewritten — `risk_records.jsonl` established the same
  pattern for the same reason;
* a decision can carry several marks at several horizons, which a single ``settled_at`` field
  cannot express;
* every mark carries **the horizon actually measured**, never an assumed one.

**Why ~2h and not 1h.** The committed cycle schedule is 13:30/15:30/17:30/19:30 UTC. The earliest a
decision can be observed again is therefore the next cycle, about two hours later — a one-hour
horizon is not observable at this cadence at all, whatever would be preferable in principle. Two
hours is also exactly the spacing between decisions, so consecutive 2h windows do **not** overlap,
and it is the horizon `data/hurdle_clearance.json` already measures the universe over.

**The last cycle of the day is different, and is not hidden.** A decision taken at 19:30 UTC is not
seen again until 13:30 the next day, an 18-hour gap. Those marks are recorded with their real
horizon and land in a different bucket; they are not quietly averaged in with the 2h ones. That is
the same error `eval/bookcalib.py` was found making, and it is not repeated here.

This module only *observes*. It scores nothing and it decides nothing — `eval/refusal.py` reads
these marks against the leans in the ledger, and is where any claim about refusal quality is made.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

MARKS_PATH = Path(__file__).resolve().parents[3] / "data" / "refusal_marks.jsonl"

MIN_MARK_HOURS = 1.5
"""How old a decision must be before it is marked.

Below the cycle spacing, so the first cycle after a decision takes the mark, and above zero so a
decision is never marked inside the cycle that produced it — that would be reading the price the
decision already saw and calling it an outcome.
"""

MAX_MARK_HOURS = 23.0
"""Above this a decision is left to the ledger's own settlement instead.

A mark and a settlement at the same horizon would be two records of one observation, and the second
would look like corroboration.
"""

SCHEMA = 1


class MarkError(ValueError):
    """Raised rather than writing a mark that cannot be interpreted."""


@dataclass(frozen=True, slots=True)
class Mark:
    """One later look at the price a decision declined to act on.

    ``horizon_hours`` is measured from the decision, never assumed from a schedule. A cycle that
    runs late, or a decision taken in the last cycle before an overnight gap, produces a real
    horizon different from the nominal one, and the difference is the whole reason this field
    exists.
    """

    seq: int
    symbol: str
    decided_at: str
    marked_at: str
    horizon_hours: float
    entry_price: str
    mark_price: str
    move_bps: str
    lean: str
    """The direction the desk said it would take if forced. ``"none"`` is a real answer and means

    the desk could not call the direction — it is not a missing value, and `eval/refusal.py` must
    exclude it from accuracy rather than score it as a miss.
    """

    schema: int = SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Mark:
        return cls(
            seq=int(raw["seq"]),
            symbol=str(raw["symbol"]),
            decided_at=str(raw["decided_at"]),
            marked_at=str(raw["marked_at"]),
            horizon_hours=float(raw["horizon_hours"]),
            entry_price=str(raw["entry_price"]),
            mark_price=str(raw["mark_price"]),
            move_bps=str(raw["move_bps"]),
            lean=str(raw.get("lean", "none")),
            schema=int(raw.get("schema", SCHEMA)),
        )

    @property
    def move(self) -> float:
        return float(self.move_bps)

    @property
    def is_directional(self) -> bool:
        return self.lean in ("up", "down")

    @property
    def lean_was_right(self) -> bool | None:
        """Did the move go the way the desk said it would, or ``None`` when it named no way.

        **``None``, not ``False``.** A desk that declined to call the direction has not made a
        wrong call; scoring "none" as a miss would punish the honest answer the PM prompt
        explicitly invites, and would make a desk that always guessed look better than one that
        admitted uncertainty.
        """
        if not self.is_directional:
            return None
        return (self.lean == "up") == (self.move > 0)


def read_marks(path: Path = MARKS_PATH) -> list[Mark]:
    """Every mark on disk, or none. A missing file is an absent record, not an empty finding."""
    if not path.exists():
        return []
    out: list[Mark] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(Mark.from_dict(json.loads(line)))
    return out


def already_marked(path: Path = MARKS_PATH) -> set[int]:
    """Sequence numbers that already carry a mark.

    One mark per decision. A second look at a later horizon would be a different measurement and is
    deliberately not taken here: re-marking the same decision each cycle would produce a pile of
    overlapping windows, which is the exact defect this module exists to avoid.
    """
    return {m.seq for m in read_marks(path)}


def mark(
    *,
    seq: int,
    symbol: str,
    decided_at: datetime,
    marked_at: datetime,
    entry_price: Decimal,
    mark_price: Decimal,
    lean: str,
    path: Path = MARKS_PATH,
) -> Mark:
    """Record one later observation of a decision's price. Appends; never rewrites."""
    if entry_price <= 0:
        raise MarkError(f"decision {seq} has a non-positive entry price; no move can be measured")
    horizon = (marked_at - decided_at).total_seconds() / 3600.0
    if horizon <= 0:
        raise MarkError(
            f"decision {seq} would be marked at or before the moment it was taken "
            f"({horizon:.3f}h); that reads the price the decision already had"
        )
    move = (mark_price - entry_price) / entry_price * Decimal("10000")
    row = Mark(
        seq=seq,
        symbol=symbol,
        decided_at=decided_at.isoformat(),
        marked_at=marked_at.isoformat(),
        horizon_hours=round(horizon, 4),
        entry_price=str(entry_price),
        mark_price=str(mark_price),
        move_bps=str(round(move, 4)),
        lean=(lean or "none").lower(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row.as_dict()) + "\n")
    return row


def due(
    entries: list[Any], *, now: datetime, governs_from_seq: int = 0, path: Path = MARKS_PATH
) -> list[Any]:
    """Which ledger entries should be marked in this cycle.

    An entry qualifies when it is an abstention, is not marked yet, was taken **under a protocol
    that declared marking**, and its age sits inside the window between :data:`MIN_MARK_HOURS` and
    :data:`MAX_MARK_HOURS`. Entries older than the window are skipped rather than marked late: a
    mark taken at eighteen hours is a real observation and is recorded as such, but one taken at
    twenty-three is indistinguishable from the settlement that follows it.

    **``governs_from_seq`` is not an optimisation — it is the whole point.** When v3 was committed,
    48 decisions taken the previous day were still inside the age window and would have been
    marked. They were taken under v2, which declared one observation per abstention. Marking them
    now would add a second observation to decisions whose governing protocol did not provide for
    one, chosen *after* their outcomes were already visible on the tape — a retroactive measurement
    on a known result, which is the precise thing pre-registration exists to prevent. Those
    decisions keep the single observation their protocol declared, and their refusal accuracy stays
    UNDEFINED rather than being back-filled.
    """
    seen = already_marked(path)
    out = []
    for entry in entries:
        if entry.seq in seen or not entry.is_abstention:
            continue
        if entry.seq < governs_from_seq:
            continue
        age = (now - datetime.fromisoformat(entry.decided_at)).total_seconds() / 3600.0
        if MIN_MARK_HOURS <= age <= MAX_MARK_HOURS:
            out.append(entry)
    return out
