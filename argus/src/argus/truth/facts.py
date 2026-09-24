"""Facts carry five times, not one — and the store, not the caller, enforces the bound.

The defect this module exists to make impossible was found in code, twice, in widely cited
architectures:

  * FinMem  — `memorydb.py:138-218` populates `temp_date_list` at lines 169 and 195 and never
    checks it before ranking. A decision on date *t* can retrieve a memory from *t+100*.
  * FinAgent — `memory/basic_memory.py:62` performs an unbounded `similarity_search`; worse, the
    environment computes `days_future = now + 14`, placing future state directly in the
    observation.

Both failed the same way: retrieval was *asked* to be point-in-time by convention, and the
convention was not kept. So here the `as_of` bound is a required argument on the query interface.
A caller who omits it gets a `TypeError` from Python itself, not a silent leak.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class TimeKind(StrEnum):
    EVENT = "event_time"
    """When the thing happened in the world."""

    PUBLISH = "publish_time"
    """When the source published it."""

    INGEST = "ingest_time"
    """When we received it."""

    AVAILABLE = "available_at"
    """When it became usable by a decision. This is the only one retrieval is bounded by."""


@dataclass(frozen=True, slots=True)
class Fact:
    """One piece of evidence, with its full temporal provenance.

    `available_at` is deliberately not derived automatically from `publish_time`. GDELT's
    `seendate` is crawl time rather than publish time; a filing's acceptance timestamp differs
    from its dissemination. Forcing the ingestor to state `available_at` explicitly means the
    assumption is recorded rather than assumed.
    """

    claim: str
    source_id: str
    event_time: datetime
    publish_time: datetime
    ingest_time: datetime
    available_at: datetime

    revision_id: int = 0
    supersedes: str | None = None
    """content_hash of the fact this revises, if any. Restatements do not overwrite history."""

    locator: str | None = None
    """Where in the source this came from — e.g. "10-Q p84 bbox(72,331,540,388)"."""

    credibility: float = 1.0
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("event_time", "publish_time", "ingest_time", "available_at"):
            value: datetime = getattr(self, name)
            if value.tzinfo is None:
                raise ValueError(
                    f"{name} must be timezone-aware; naive datetimes cannot be ordered safely"
                )
        if not 0.0 <= self.credibility <= 1.0:
            raise ValueError("credibility must lie in [0, 1]")

    @property
    def content_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.claim.encode())
        h.update(self.source_id.encode())
        h.update(self.event_time.isoformat().encode())
        h.update(str(self.revision_id).encode())
        return h.hexdigest()[:16]

    @property
    def publication_lag(self) -> float:
        """Seconds between the event happening and it becoming usable.

        Useful on its own: a strategy whose edge collapses as this grows is trading the reporting
        pipeline, not the market.
        """
        return (self.available_at - self.event_time).total_seconds()


class LookAheadError(RuntimeError):
    """Raised when a fact that was not yet available is asked to influence a decision.

    This is an error rather than a filter-and-continue because a caller that constructs an
    impossible query has a bug, and silently returning fewer rows hides it.
    """


class AsOfStore:
    """Evidence store where the point-in-time bound is enforced at the interface.

    `query()` takes `as_of` as a keyword-only required argument. There is no default, no
    `as_of=None` meaning "latest", and no module-level "current time" that a caller could forget
    to override. That is the whole design.
    """

    def __init__(self, facts: Iterable[Fact] | None = None) -> None:
        self._facts: list[Fact] = list(facts or [])

    def add(self, fact: Fact) -> None:
        self._facts.append(fact)

    def __len__(self) -> int:
        return len(self._facts)

    def query(
        self,
        *,
        as_of: datetime,
        source_id: str | None = None,
        limit: int | None = None,
    ) -> Sequence[Fact]:
        """Every fact available at `as_of`, newest-available first.

        Superseded revisions are resolved *as they stood at `as_of`*: if a restatement arrived
        after `as_of`, the caller sees the original figure, which is what a decision-maker at that
        instant would have had. Returning the restated value is the subtler cousin of look-ahead
        and is why financial backtests silently beat reality.
        """
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")

        visible = [f for f in self._facts if f.available_at <= as_of]
        if source_id is not None:
            visible = [f for f in visible if f.source_id == source_id]

        superseded = {f.supersedes for f in visible if f.supersedes is not None}
        current = [f for f in visible if f.content_hash not in superseded]

        current.sort(key=lambda f: (f.available_at, f.content_hash), reverse=True)
        return current[:limit] if limit is not None else current

    def assert_no_future_facts(self, as_of: datetime, facts: Sequence[Fact]) -> None:
        """Guard for anything that assembles evidence by a path other than `query`.

        Used by the adversarial suite's future-data injection control: if any route into a
        decision ever yields a fact stamped after the decision instant, this is what catches it.
        """
        for f in facts:
            if f.available_at > as_of:
                raise LookAheadError(
                    f"fact {f.content_hash!r} from {f.source_id!r} is available_at "
                    f"{f.available_at.isoformat()}, after as_of {as_of.isoformat()}"
                )
