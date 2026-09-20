"""The Register — falsifiable claims, committed before the fact, resolved mechanically.

Every claim made about a market is made into a void. A call is posted, a backtest reports a Sharpe,
an agent opens a position with a thesis, a Playbook is listed with a description. None of it is
mechanically connected to what happened next, and by the time anyone could check, the claim has
scrolled away. That is why every leaderboard in this field is a survivorship artefact and every
track record is marketing.

This module is the smallest thing that fixes it: a claim you cannot edit, about an outcome you
cannot influence, resolved by a predicate that executes against a point-in-time source, with the
commitment anchored to Bitcoin before the outcome is known.

**The one rule that makes it a register and not a comment section.** A claim whose resolution
predicate cannot execute against a named point-in-time source is **refused at registration**, with
the reason. No prose, no "NVDA looks strong", no unfalsifiable hedging. That constraint kills the
entire genre of ungradable market commentary, and it is enforced by the machinery `truth/facts.py`
already imposes: a query without an as-of date is a type error, and there is no "latest".

**Why this is built the crude way first.** Build effort can be compressed; elapsed time cannot. A
register that opens today and is finished over the following year holds a year of record. One that
is perfect in six months holds six months less, permanently, and never catches up to itself. So the
schema, the refusal, the resolver and the anchor exist before the API, the network, the grading and
everything else — because the only input here that cannot be bought is the clock, and it starts on
the first commitment.

**What it refuses to do to itself.** A claim cannot be edited, deleted, or resolved early. The
as-of gate applies to resolution exactly as it applies to evidence: :func:`resolve` will not look at
a price stamped before ``resolves_at``, so a resolver run too early returns PENDING rather than an
answer. `eval/leakage.py` exists in this codebase because a model that has read the outcome is not
forecasting; a register that resolves early would be committing the same error by hand.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parents[3] / "data"
REGISTER_PATH = DATA / "register.jsonl"
"""Append-only. One JSON object per line, each carrying the hash of the line before it."""

SCHEMA_VERSION = 1

MIN_HORIZON_SECONDS = 3600
"""A claim must resolve at least an hour after it is registered.

Not arbitrary: below this the gap between "committed" and "observable" is small enough that clock
skew between our host and the venue's stamps could let a claim be registered about a bar that has
already printed. An hour is far outside any plausible skew and still short enough for intraday
claims.
"""


class Predicate(StrEnum):
    """The resolution rules that can actually execute. Adding one is a deliberate act.

    Kept deliberately small. Every member here is a rule whose inputs are a point-in-time price
    series and whose output is a boolean that no human touches — which is the whole contract. A
    predicate that needed a judgement call would make this a comment section with extra steps.
    """

    CLOSE_ABOVE = "close_above"
    """The close at ``resolves_at`` is strictly above ``threshold``."""

    CLOSE_BELOW = "close_below"
    RETURN_OVER_ABOVE_BPS = "return_over_above_bps"
    """The return from registration to ``resolves_at`` exceeds ``threshold`` basis points."""

    RETURN_OVER_BELOW_BPS = "return_over_below_bps"
    ABS_MOVE_ABOVE_BPS = "abs_move_above_bps"
    """The absolute move exceeds ``threshold`` bps — a volatility claim, direction-free."""


class Status(StrEnum):
    PENDING = "pending"
    TRUE = "true"
    FALSE = "false"
    UNRESOLVABLE = "unresolvable"
    """The source could not answer at the stated time. Never silently a FALSE: a claim we cannot
    grade is a hole in the record, and scoring it as a miss would flatter the claimant by turning
    an unknown into a known."""


class RegisterError(ValueError):
    """Raised at registration. A refused claim never reaches the file."""


@dataclass(frozen=True, slots=True)
class Claim:
    """One falsifiable statement, with everything needed to grade it and nothing else."""

    claimant: str
    subject: str
    """The instrument, exactly as the data source names it."""

    predicate: Predicate
    threshold: str
    """Decimal as a string, so the committed bytes are exact and no float rounding can move it."""

    registered_at: str
    resolves_at: str
    source: str
    """The point-in-time data binding, e.g. ``bitget:1H:market``. Named so a stranger can re-derive
    the answer from the same series rather than trusting ours."""

    confidence: float
    """The claimant's stated probability. This is what makes the record scorable by calibration
    rather than by hit rate — and calibration is the only thing on a leaderboard that cannot be
    gamed without actually being right."""

    nonce: str
    """Separates two identical claims made at one instant, so neither can be denied later."""

    invalidation: str = ""
    """What would make the claim moot rather than wrong — a halt, a delisting, a data outage."""

    schema: int = SCHEMA_VERSION
    status: Status = Status.PENDING
    resolved_at: str | None = None
    observed: str | None = None
    """The value the predicate actually saw, recorded so the resolution can be re-checked."""

    prev_hash: str = ""
    entry_hash: str = ""

    @property
    def horizon_seconds(self) -> float:
        return (
            datetime.fromisoformat(self.resolves_at) - datetime.fromisoformat(self.registered_at)
        ).total_seconds()

    def content(self) -> dict[str, Any]:
        """Exactly the committed fields — never the resolution.

        The hash covers what was claimed, not how it turned out. Including the outcome would make
        the digest unverifiable until resolution, and the entire point is that the commitment is
        checkable *before* anyone knows the answer.
        """
        blob = asdict(self)
        for mutable in ("status", "resolved_at", "observed", "prev_hash", "entry_hash"):
            blob.pop(mutable, None)
        return blob

    def digest(self) -> str:
        payload = json.dumps(self.content(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"{self.prev_hash}{payload}".encode()).hexdigest()

    def render(self) -> str:
        mark = {
            Status.PENDING: "·", Status.TRUE: "✓", Status.FALSE: "✗",
            Status.UNRESOLVABLE: "?",
        }[self.status]
        return (
            f"{mark} {self.subject} {self.predicate} {self.threshold} by "
            f"{self.resolves_at[:16]} (p={self.confidence:.2f}) [{self.entry_hash[:12]}]"
        )


_SUBJECT = re.compile(r"^[A-Z0-9]{3,20}$")
_SOURCE = re.compile(r"^[a-z0-9]+:[A-Za-z0-9]+:[a-z]+$")


def validate(claim: Claim) -> None:
    """Refuse anything that cannot be resolved mechanically. **Do not soften this later.**

    Every check here exists because its absence would let an ungradable claim onto a permanent
    record, and a register that carries one unfalsifiable entry is a comment section that has not
    admitted it yet.
    """
    if not claim.claimant.strip():
        raise RegisterError("a claim needs a claimant; an anonymous record cannot be scored")
    if not _SUBJECT.match(claim.subject):
        raise RegisterError(
            f"subject {claim.subject!r} is not an instrument symbol the source can be queried with"
        )
    if not _SOURCE.match(claim.source):
        raise RegisterError(
            f"source {claim.source!r} must name a point-in-time binding as venue:interval:kind, "
            f"e.g. 'bitget:1H:market' — a claim whose source is not named cannot be re-derived"
        )
    try:
        Decimal(claim.threshold)
    except Exception as exc:
        raise RegisterError(f"threshold {claim.threshold!r} is not a decimal") from exc
    if not 0.0 <= claim.confidence <= 1.0:
        raise RegisterError(
            f"confidence {claim.confidence} is not a probability; the record is scored by "
            f"calibration and an out-of-range number would corrupt every Brier score after it"
        )
    try:
        registered = datetime.fromisoformat(claim.registered_at)
        resolves = datetime.fromisoformat(claim.resolves_at)
    except ValueError as exc:
        raise RegisterError("timestamps must be ISO-8601 with an offset") from exc
    if registered.tzinfo is None or resolves.tzinfo is None:
        raise RegisterError("timestamps must carry a timezone; a naive stamp is unverifiable")
    if claim.horizon_seconds < MIN_HORIZON_SECONDS:
        raise RegisterError(
            f"a claim must resolve at least {MIN_HORIZON_SECONDS}s after registration; this one "
            f"resolves in {claim.horizon_seconds:.0f}s, inside the window where clock skew could "
            f"let it be registered about a bar that has already printed"
        )


def _read(path: Path) -> list[Claim]:
    if not path.exists():
        return []
    out: list[Claim] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        blob = json.loads(line)
        blob["predicate"] = Predicate(blob["predicate"])
        blob["status"] = Status(blob["status"])
        out.append(Claim(**blob))
    return out


def register(
    claims: Sequence[Claim], *, path: Path = REGISTER_PATH,
) -> list[Claim]:
    """Validate, chain and append. Returns the claims as committed, with their hashes.

    Appending only. There is no update and no delete, and there will not be one: the day a claim can
    be edited, every claim before it becomes a story rather than a record.
    """
    existing = _read(path)
    prev = existing[-1].entry_hash if existing else ""
    written: list[Claim] = []
    for claim in claims:
        validate(claim)
        chained = replace(claim, prev_hash=prev)
        chained = replace(chained, entry_hash=chained.digest())
        written.append(chained)
        prev = chained.entry_hash
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for claim in written:
            blob = asdict(claim)
            blob["predicate"] = str(claim.predicate)
            blob["status"] = str(claim.status)
            handle.write(json.dumps(blob, sort_keys=True) + "\n")
    return written


def verify_chain(path: Path = REGISTER_PATH) -> tuple[bool, str]:
    """Recompute every hash. Returns ``(intact, note)`` and never raises on a broken chain.

    A broken chain is a finding to be reported, not an exception to be caught by whoever happens to
    call this — the whole value of the file is that somebody else can check it.
    """
    rows = _read(path)
    if not rows:
        return True, "empty register"
    prev = ""
    for index, claim in enumerate(rows):
        if claim.prev_hash != prev:
            return False, f"entry {index} claims prev {claim.prev_hash[:12]}, chain has {prev[:12]}"
        if claim.digest() != claim.entry_hash:
            return False, f"entry {index} hash does not match its own content"
        prev = claim.entry_hash
    return True, f"{len(rows)} claim(s), chain intact, head {prev[:16]}"


def batch_digest(claims: Sequence[Claim]) -> str:
    """The digest submitted to the Bitcoin calendars: the head hash of this batch.

    One anchor per batch rather than per claim: a calendar submission costs a round trip, and the
    chain already binds every earlier claim into the head, so anchoring the head anchors them all.
    """
    if not claims:
        raise RegisterError("cannot anchor an empty batch")
    return claims[-1].entry_hash


def make_claim(
    *, claimant: str, subject: str, predicate: Predicate, threshold: str, resolves_at: datetime,
    source: str, confidence: float, invalidation: str = "", nonce: str = "",
    now: datetime | None = None,
) -> Claim:
    """Build an unregistered claim. Validation happens at :func:`register`, never here."""
    stamp = now or datetime.now(UTC)
    return Claim(
        claimant=claimant, subject=subject, predicate=predicate, threshold=str(threshold),
        registered_at=stamp.isoformat(), resolves_at=resolves_at.isoformat(), source=source,
        confidence=confidence, invalidation=invalidation,
        nonce=nonce or hashlib.sha256(
            f"{subject}{predicate}{threshold}{stamp.isoformat()}".encode()
        ).hexdigest()[:12],
    )


__all__ = [
    "MIN_HORIZON_SECONDS",
    "REGISTER_PATH",
    "SCHEMA_VERSION",
    "Claim",
    "Predicate",
    "RegisterError",
    "Status",
    "batch_digest",
    "make_claim",
    "register",
    "validate",
    "verify_chain",
]
