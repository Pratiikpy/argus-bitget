"""Batch writes with a per-item fallback that reports exactly which items failed, and why.

**The defect this closes, in our own settlement path.** `paper/runner.py` ``_settle_due`` settled
each due ledger row in a plain loop, and any one row that could not be settled — a side that is
neither BUY nor SELL, a non-positive entry price, a sequence number written twice (the 2026-09-12
concurrent-writer incident produced exactly that) — raised out of the loop, out of ``run_once``,
and out of the process. The cycle then decided nothing, and because the bad row was still due on the
next cycle, so did every cycle after it: one poisoned row stops the desk. And each settlement
rewrote the whole ledger file on its own, so a cycle settling twelve rows rewrote it twelve times.

What was taken, from which file, under which licence
----------------------------------------------------

mem0 (``mem0ai/mem0``, Apache-2.0, Copyright 2023 Taranjeet Singh; local clone
``mypr/04_memory_and_personalization/mem0``; licence and "Used in" record at
``argus/licenses/mem0-APACHE-2.0.txt``):

- **Try the batch; on failure fall back to one item at a time; log each failure; never let one bad
  record take the rest with it.** ``mem0/memory/main.py:993-1005`` (batch embedding, then per-text
  embedding with a warning per failure) and ``main.py:1047-1077`` (batch vector insert, then
  per-record insert, and "only records confirmed to be stored make it into history ... a record the
  store rejected must never be reported back as a successful ADD"). Adapted in :func:`write_batch`,
  changed in two ways that matter when the store is a ledger or a venue rather than a vector index:

  1. **Probe before resending.** mem0's fallback re-inserts every record after a failed batch call
     (``main.py:1064-1070``). A vector insert under a fresh UUID is harmless to repeat; an order is
     not, and a batch call that raised may have landed part of its work first. So when the batch
     raises, each item is first checked with ``applied`` — does the store already hold it? — and
     only an item the store positively does not hold is written again. An item whose presence
     cannot be established is reported :attr:`ItemStatus.UNKNOWN` and **not** resent, the same
     refusal :class:`argus.execution.orders.OrderState.UNKNOWN` encodes: a timeout is not a
     rejection, and retrying one is how duplicate exposure is made.
  2. **Positive evidence per item.** An item counts as written only if the batch response names it
     as succeeded. mem0 treats a batch call that returned as a batch that fully succeeded; a venue
     batch endpoint can return success while listing failures, or say nothing about an item at
     all. An item the response does not mention is treated exactly like one from a batch that
     raised: probed, and resent only if the store does not hold it. This is SWE-bench's rule
     (``swebench/harness/grading.py:155-160``, MIT) that the absence of a failure signal is not a
     success signal.

Rejected: mem0's pattern of logging the failure and returning the survivors silently. Every item's
outcome here is in the returned :class:`BatchReport`, because the caller — the settlement pass, the
cycle summary — has to print which rows failed, not merely that some did.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

T = TypeVar("T")


class ItemStatus(StrEnum):
    """Every outcome an item in a batch can have. Closed set."""

    WRITTEN_IN_BATCH = "written_in_batch"
    """The batch call succeeded and named this item as written."""

    WRITTEN_SINGLY = "written_singly"
    """Written by the per-item fallback after the store confirmed it did not already hold it."""

    ALREADY_APPLIED = "already_applied"
    """The store already held it — the batch landed it before failing, or someone else did."""

    FAILED = "failed"
    """Refused by the store on its merits, or the per-item write raised. The reason is recorded."""

    REFUSED = "refused"
    """Never sent: it failed validation, or named the same key twice in one batch."""

    UNKNOWN = "unknown"
    """The store could not say whether it holds this item. Not resent; reconcile before retrying."""


_WRITTEN = frozenset({ItemStatus.WRITTEN_IN_BATCH, ItemStatus.WRITTEN_SINGLY,
                      ItemStatus.ALREADY_APPLIED})


@dataclass(frozen=True, slots=True)
class BatchResponse:
    """What the store said about a batch call, item by item."""

    succeeded: frozenset[str]
    failed: Mapping[str, str] = field(default_factory=dict)
    """Key to the store's own reason."""


@dataclass(frozen=True, slots=True)
class ItemOutcome:
    key: str
    status: ItemStatus
    detail: str = ""

    @property
    def written(self) -> bool:
        return self.status in _WRITTEN

    def as_dict(self) -> dict[str, str]:
        return {"key": self.key, "status": self.status.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class BatchReport:
    """Every item's outcome, in input order, and whether the batch path had to be abandoned."""

    outcomes: tuple[ItemOutcome, ...]
    fell_back: bool
    batch_error: str | None
    unexpected: tuple[str, ...] = ()
    """Keys the store's response named that were never sent. Recorded, never trusted."""

    @property
    def failed(self) -> tuple[ItemOutcome, ...]:
        return tuple(o for o in self.outcomes if not o.written)

    @property
    def failed_keys(self) -> tuple[str, ...]:
        return tuple(o.key for o in self.failed)

    @property
    def written_keys(self) -> tuple[str, ...]:
        return tuple(o.key for o in self.outcomes if o.written)

    @property
    def all_written(self) -> bool:
        return not self.failed

    def count(self, status: ItemStatus) -> int:
        return sum(1 for o in self.outcomes if o.status is status)

    def render(self) -> str:
        head = (
            f"[batch] {len(self.written_keys)} of {len(self.outcomes)} written"
            + (f"; the batch call failed ({self.batch_error}) and each item was handled singly"
               if self.fell_back else "")
        )
        lines = [head]
        lines.extend(f"  {o.key}: {o.status.value} — {o.detail}" for o in self.failed)
        if self.unexpected:
            lines.append(f"  the store named keys never sent: {', '.join(self.unexpected)}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": len(self.outcomes),
            "written": len(self.written_keys),
            "failed_keys": list(self.failed_keys),
            "fell_back": self.fell_back,
            "batch_error": self.batch_error,
            "by_status": {s.value: self.count(s) for s in ItemStatus if self.count(s)},
            "unexpected": list(self.unexpected),
            "outcomes": [o.as_dict() for o in self.outcomes],
        }


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:200]}"


def write_batch(
    items: Sequence[T],
    *,
    key: Callable[[T], str],
    batch: Callable[[Sequence[T]], BatchResponse],
    single: Callable[[T], None],
    applied: Callable[[T], bool | None] | None = None,
    validate: Callable[[T], str | None] | None = None,
) -> BatchReport:
    """Write ``items`` in one batch call, falling back to one call per item. Never raises per item.

    ``applied(item)`` answers "does the store already hold this?" with ``True``, ``False``, or
    ``None`` for "cannot tell". Pass ``None`` for the whole probe only when the store is idempotent
    by construction, so writing an item twice is harmless; otherwise every item the batch did not
    positively confirm is probed before it is written again.

    ``validate(item)`` returns a reason to refuse the item before anything is sent, or ``None``.
    """
    slots: list[ItemOutcome | None] = []
    seen: set[str] = set()
    to_send: list[tuple[int, T]] = []
    for item in items:
        k = key(item)
        if k in seen:
            # The earlier occurrence keeps its place; this one is refused. A batch that names one
            # order twice is asking for two positions, and the second is never what was meant.
            slots.append(ItemOutcome(
                k, ItemStatus.REFUSED, "the same key appears more than once in one batch",
            ))
            continue
        seen.add(k)
        reason = validate(item) if validate is not None else None
        if reason is not None:
            slots.append(ItemOutcome(k, ItemStatus.REFUSED, reason))
            continue
        slots.append(None)
        to_send.append((len(slots) - 1, item))

    def settle_one(item: T, why: str) -> ItemOutcome:
        k = key(item)
        if applied is not None:
            try:
                held = applied(item)
            except Exception as exc:
                return ItemOutcome(
                    k, ItemStatus.UNKNOWN, f"{why}; the probe raised {_describe(exc)}",
                )
            if held is None:
                return ItemOutcome(
                    k, ItemStatus.UNKNOWN,
                    f"{why}, and the store cannot say whether it holds this item; not resent",
                )
            if held:
                return ItemOutcome(
                    k, ItemStatus.ALREADY_APPLIED, f"{why}; the store already holds it",
                )
        try:
            single(item)
        except Exception as exc:
            return ItemOutcome(k, ItemStatus.FAILED, f"{why}; written singly and refused: "
                                                      f"{_describe(exc)}")
        return ItemOutcome(k, ItemStatus.WRITTEN_SINGLY, f"{why}; written singly")

    fell_back = False
    batch_error: str | None = None
    unexpected: tuple[str, ...] = ()
    if to_send:
        try:
            response = batch([item for _, item in to_send])
        except Exception as exc:
            fell_back = True
            batch_error = _describe(exc)
            for slot, item in to_send:
                slots[slot] = settle_one(item, "the batch call raised")
        else:
            sent = {key(item) for _, item in to_send}
            unexpected = tuple(sorted((set(response.succeeded) | set(response.failed)) - sent))
            for slot, item in to_send:
                k = key(item)
                if k in response.failed:
                    # A refusal on the merits is final: repeating it repeats the refusal, and for
                    # an order a misreported refusal would become a second position.
                    slots[slot] = ItemOutcome(k, ItemStatus.FAILED, response.failed[k])
                elif k in response.succeeded:
                    slots[slot] = ItemOutcome(
                        k, ItemStatus.WRITTEN_IN_BATCH, "confirmed by the batch response",
                    )
                else:
                    slots[slot] = settle_one(item, "the batch response did not mention it")

    outcomes = tuple(o for o in slots if o is not None)
    if len(outcomes) != len(slots):  # pragma: no cover - every sent slot is filled above
        raise RuntimeError("a batch item was left without an outcome")
    return BatchReport(
        outcomes=outcomes,
        fell_back=fell_back,
        batch_error=batch_error,
        unexpected=unexpected,
    )


__all__ = [
    "BatchReport",
    "BatchResponse",
    "ItemOutcome",
    "ItemStatus",
    "write_batch",
]
