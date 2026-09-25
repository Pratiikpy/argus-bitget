"""A durable, typed human-in-the-loop pause: the desk stops, asks, and continues from that point.

**What was wrong.** `decision/escalation.py` decides *when* the desk must not act alone, and until
this module the answer to that question was the end of the decision. An escalation turned the
intent into ``HUMAN_REVIEW`` with quantity zero and the desk finished the cycle on it
(`agents/desk.py`, stage 2c; the 40-repo synthesis recorded it as "`HUMAN_REVIEW` is a final
verdict. Nothing pauses and resumes"). A human who read the escalation and agreed with the machine
had no path back into the decision: saying "yes" produced nothing, because there was nothing left
to say yes *to*. The in-flight state — the panel, the proof, the frame the model saw — was gone the
moment the cycle returned. So "human-takeover rate" measured how often the desk gave up, not how
often a human took over.

**What this module is.** A typed request/response pair, a continuation persisted to disk with a
state signature, an append-only event log, and a resume that is refused unless every one of those
checks out. `agents/desk.py` pauses on a policy escalation, a human answers (approve, reject, or
modify the size), and :meth:`argus.agents.desk.TradingDesk.resume` continues *from the paused
point* — the Constitution, the model's revision, the mandate and the order — on the state as it
was, not on a re-run of the analysts.

What was taken, from which file, under which licence
----------------------------------------------------

Microsoft Agent Framework (``microsoft/agent-framework``, MIT, Copyright (c) Microsoft Corporation;
local clone ``mypr/09_orchestration_and_tools/microsoft_agent_framework``):

- **The request is typed by its (request, response) pair.**
  ``_workflows/_request_info_mixin.py:36-73`` — ``is_request_supported`` and
  ``_find_response_handler`` route a response only when *both* the pending request and the answer
  are instances of a registered pair; a response of the wrong shape finds no handler rather than
  being coerced. Here there is one request type and one response type, so the pairing is expressed
  as :func:`check_response`: an answer is accepted only if it names this request, uses an action
  this request allows, and carries exactly the fields that action needs. Rejected: MAF's
  decorator-and-registry machinery (``_discover_response_handlers``, :76-110), which exists to
  support many executor classes each with many handlers — one desk with one pause point does not
  need a registry, and adding one would be structure without a second use.
- **A resume is refused against a changed pipeline.** ``_workflows/_workflow.py:372-375`` computes
  a canonical fingerprint of the graph topology and ``_workflows/_runner.py:316-320,378-382``
  refuses to restore a checkpoint whose fingerprint differs — "Workflow graph has changed since the
  checkpoint was created." Here the fingerprint is ``pipeline_signature``, the desk's own hash of
  the stages a resume runs and the fields its continuation carries
  (:func:`argus.agents.desk.resume_signature`). A continuation written before someone added a stage
  is not silently replayed through a pipeline it never saw.
- **JSON outside, pickle inside, and a restricted unpickler.** ``_checkpoint_encoding.py:1-45``
  (the stated security model), ``:158-223`` (``_RestrictedUnpickler``: an allowlist of ``module:
  qualname`` keys, framework types by module prefix, top-level classes only, enum members only
  through a guarded ``getattr``). :class:`_RestrictedUnpickler` below is an adaptation of that
  class with the prefix changed to ``argus.`` and MAF's blocklist replaced by one naming this
  module's own loaders. **One addition MAF does not make**: the pickle's SHA-256 is written beside
  it and checked *before* a single byte is unpickled, so a truncated or edited continuation is
  refused by a hash comparison rather than discovered by executing it.

LangGraph (``langchain-ai/langgraph``, MIT, Copyright (c) 2024 LangChain, Inc.; local clone
``mypr/09_orchestration_and_tools/langgraph``):

- **A re-executed step replays the answer it was already given instead of asking again.**
  ``libs/langgraph/langgraph/types.py:887-1021`` — ``interrupt()`` keeps a per-task
  ``interrupt_counter()`` (:1004) and, when the node is re-run after a resume, returns the recorded
  resume value for that index (:1005-1011) rather than raising ``GraphInterrupt`` a second time.
  Here the counter is ``PauseRequest.sequence`` and the replay is :meth:`PauseStore.prior`: if the
  desk is re-run for a decision a human has already answered, it concludes with that answer rather
  than paging the human twice, and if the answer has already been acted on it refuses to act again.
- **The pause carries the schema of the answer it expects.** ``types.py:887-930``'s
  ``response_schema`` is surfaced to the client "so they can render a typed input form".
  :meth:`PauseRequest.response_schema` is the same idea as a real JSON Schema, and the CLI prints
  it.

**One deliberate departure from LangGraph, and why.** LangGraph resumes by *re-executing* the node
from the start and replaying recorded values, which is correct when the node is deterministic.
The desk's first half is not: it calls a language model, and a re-run can produce a different
proposal. Replaying a human's "approve" onto a proposal the human never saw would be forging their
consent. So the replay is keyed by the *proposal's* hash, not by position alone: the same proposal
replays the answer, a different one supersedes the old request and asks again. And a resume does
not re-execute the model's half at all — it restores the continuation MAF-style and runs only the
deterministic-or-bounded second half.

**What a human may do.** Release the hold at the proposed size, release it smaller, or refuse.
A human may not enlarge the position, flip its side, or originate a trade the model did not
propose: :func:`resolve_intent` raises if the result would. Two reasons, both binding. Track 2's
positioning requires *the LLM* to be the decision-maker, and a human-sized trade would not be the
model's decision. And a released intent is still ruled by the Constitution and the mandate on
resume — a human can release a hold, not the risk layer.

**Every answer has a window.** A proposal is priced at the instant it was made. An approval given
two hours later approves a price that no longer exists, so a request carries ``expires_at`` and both
answering and resuming after it are refused and recorded as ``expired``. Thirty minutes is a policy
choice stated as a parameter, not a measurement.

**The takeover rate is read from the event log**, not from live objects. The log is the only record
that survives the process that made the decisions; :func:`summarise` rebuilds the escalation
outcomes from ``decided`` events and hands them to :func:`argus.decision.escalation.takeover_rate`,
so the rate here and the rate `eval/bench.py` reports are one computation. It adds what only a
pause can measure: how often a human answered, how often they overrode the machine, and how often
nobody answered in time.

Model-originated ``HUMAN_REVIEW`` is **not** paused. `agents/meta_pm.py` returns it for a malformed
answer — a verdict it could not parse, exposure with no falsifier, size with no quantity — and in
every case the quantity is already zero. There is no proposal a human could approve without
inventing one, so it stays a final verdict and is counted separately as ``model_requested_review``.

The CLI::

    python -m argus.decision.pause list [--all]
    python -m argus.decision.pause show REQUEST_ID
    python -m argus.decision.pause answer REQUEST_ID approve|reject|modify [--quantity Q]
                                          --reviewer NAME [--note TEXT] [--at ISO8601]
    python -m argus.decision.pause rate

``answer`` records the answer; the desk process resumes it (:meth:`TradingDesk.resume`), because
resuming can call the model once and the CLI holds no model.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import pickle
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum, EnumMeta, StrEnum
from pathlib import Path
from typing import Any, Protocol

from argus.decision.escalation import Escalation, TakeoverRate, takeover_rate
from argus.decision.verdicts import Intent, Side, Verdict
from argus.proof.autonomy import hash_intent

PAUSE_FORMAT = "argus-pause/1"
"""Written into every persisted file. A reader meeting another format refuses, never guesses."""

DEFAULT_ANSWER_WINDOW = timedelta(minutes=30)
"""How long a proposal stays answerable. A policy choice, stated as one — see the module doc."""

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "pauses"
"""``argus/data/pauses``: one directory per request under ``requests/``, one ``events.jsonl``."""


class HumanAction(StrEnum):
    """The three answers a human may give. Each may only release up to what the model proposed."""

    APPROVE = "approve"
    REJECT = "reject"
    MODIFY_SIZE = "modify_size"


class PauseError(RuntimeError):
    """A pause could not be created, answered or resumed as asked."""


class PauseIntegrityError(PauseError):
    """The persisted continuation is not the one that was written. Refused before it is executed."""


class PauseSignatureError(PauseError):
    """The pipeline that would resume this pause is not the one that paused it."""


class ResponseMismatch(PauseError):
    """An answer that does not fit the request it names — MAF's "no handler for this pair"."""


class PauseExpired(PauseError):
    """The answer window closed; the proposal it priced no longer exists."""


class AlreadyResolved(PauseError):
    """The request was already answered, resumed, expired or superseded."""


class Resumable(Protocol):
    """What a continuation must offer: a canonical, address-free view of the state it carries."""

    def signature_state(self) -> dict[str, Any]: ...


# --- canonical form and hashing ------------------------------------------------------------------


def _default(obj: Any) -> Any:
    """JSON for the non-JSON types a decision state carries. Anything else is refused.

    Refusing is the point. A ``default=str`` fallback would serialise an unknown object as its
    ``repr``, and a repr that carries a memory address makes the same state hash differently in two
    processes — which is exactly the property the kill-and-resume test exists to prove.
    """
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, set | frozenset):
        return sorted(obj, key=repr)
    raise TypeError(f"{type(obj).__name__} has no canonical form in a pause state")


def canonical(state: Any) -> str:
    """Sorted keys, no whitespace, strict JSON. The byte string the state hash is taken over."""
    return json.dumps(
        state, sort_keys=True, separators=(",", ":"), default=_default, allow_nan=False,
        ensure_ascii=False,
    )


def state_hash(state: Mapping[str, Any]) -> str:
    """The state signature. Full SHA-256: a truncated hash is a collision budget nobody chose."""
    return hashlib.sha256(canonical(state).encode("utf-8")).hexdigest()


def intent_to_dict(intent: Intent) -> dict[str, Any]:
    return {
        "symbol": intent.symbol,
        "side": intent.side.value,
        "quantity": str(intent.quantity),
        "verdict": intent.verdict.value,
        "stated_confidence": intent.stated_confidence,
        "thesis": intent.thesis,
        "invalidation": list(intent.invalidation),
        "required_hedge": list(intent.required_hedge),
        "lean": intent.lean,
        "lean_confidence": intent.lean_confidence,
    }


def intent_from_dict(blob: Mapping[str, Any]) -> Intent:
    return Intent(
        symbol=str(blob["symbol"]),
        side=Side(blob["side"]),
        quantity=Decimal(str(blob["quantity"])),
        verdict=Verdict(blob["verdict"]),
        stated_confidence=float(blob["stated_confidence"]),
        thesis=str(blob["thesis"]),
        invalidation=tuple(str(x) for x in blob.get("invalidation", ())),
        required_hedge=tuple(str(x) for x in blob.get("required_hedge", ())),
        lean=str(blob.get("lean", "none")),
        lean_confidence=float(blob.get("lean_confidence", 0.0)),
    )


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise PauseError(f"timestamp {value!r} carries no timezone; a pause is never naive")
    return parsed


# --- the typed pair ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PauseRequest:
    """What the desk asks, and everything a human needs to answer it without opening the code."""

    request_id: str
    decision_id: str
    sequence: int
    """How many times this decision had already paused — LangGraph's ``interrupt_counter()``."""

    symbol: str
    as_of: datetime
    expires_at: datetime
    proposal: Intent
    """The intent the desk would have acted on had it not escalated — after the adversary, before
    the Constitution. The ceiling on anything a human may release."""

    proposal_hash: str
    escalation: Escalation
    state_hash: str
    pipeline_signature: str
    allowed_actions: tuple[HumanAction, ...] = (
        HumanAction.APPROVE, HumanAction.REJECT, HumanAction.MODIFY_SIZE,
    )

    @property
    def max_quantity(self) -> Decimal:
        return self.proposal.quantity

    def response_schema(self) -> dict[str, Any]:
        """JSON Schema for the answer. LangGraph surfaces one "so they can render a typed form"."""
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": f"answer to {self.request_id}",
            "type": "object",
            "required": ["request_id", "action", "reviewer"],
            "properties": {
                "request_id": {"const": self.request_id},
                "action": {"enum": [a.value for a in self.allowed_actions]},
                "quantity": {
                    "type": ["string", "null"],
                    "description": (
                        f"required for {HumanAction.MODIFY_SIZE.value} only: a decimal greater "
                        f"than 0 and at most {self.max_quantity}; absent otherwise"
                    ),
                },
                "reviewer": {"type": "string", "minLength": 1},
                "note": {"type": "string"},
            },
            "additionalProperties": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": PAUSE_FORMAT,
            "request_id": self.request_id,
            "decision_id": self.decision_id,
            "sequence": self.sequence,
            "symbol": self.symbol,
            "as_of": self.as_of.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "proposal": intent_to_dict(self.proposal),
            "proposal_hash": self.proposal_hash,
            "escalation": self.escalation.as_dict(),
            "state_hash": self.state_hash,
            "pipeline_signature": self.pipeline_signature,
            "allowed_actions": [a.value for a in self.allowed_actions],
            "response_schema": self.response_schema(),
        }

    @classmethod
    def from_dict(cls, blob: Mapping[str, Any]) -> PauseRequest:
        if blob.get("format") != PAUSE_FORMAT:
            raise PauseError(f"not a {PAUSE_FORMAT} request: format={blob.get('format')!r}")
        proposal = intent_from_dict(blob["proposal"])
        request = cls(
            request_id=str(blob["request_id"]),
            decision_id=str(blob["decision_id"]),
            sequence=int(blob["sequence"]),
            symbol=str(blob["symbol"]),
            as_of=_parse_time(blob["as_of"]),
            expires_at=_parse_time(blob["expires_at"]),
            proposal=proposal,
            proposal_hash=str(blob["proposal_hash"]),
            escalation=Escalation.from_dict(dict(blob["escalation"])),
            state_hash=str(blob["state_hash"]),
            pipeline_signature=str(blob["pipeline_signature"]),
            allowed_actions=tuple(HumanAction(a) for a in blob["allowed_actions"]),
        )
        # The stored hash must be the hash of the stored proposal. An edited request.json that
        # raised the quantity would otherwise raise the ceiling a human is allowed to release.
        if hash_intent(proposal) != request.proposal_hash:
            raise PauseIntegrityError(
                f"{request.request_id}: the proposal on disk does not match its recorded hash"
            )
        return request

    def render(self) -> str:
        p = self.proposal
        lines = [
            f"{self.request_id}  {self.symbol}  asked {self.as_of.isoformat()}  "
            f"answer by {self.expires_at.isoformat()}",
            f"  proposal: {p.verdict.value} {p.side.value} {p.quantity} "
            f"(confidence {p.stated_confidence:.2f})",
            f"  thesis: {p.thesis}",
        ]
        if p.invalidation:
            lines.append(f"  invalidated by: {'; '.join(p.invalidation)}")
        lines.append("  " + self.escalation.render().replace("\n", "\n  "))
        lines.append(
            f"  answer: {', '.join(a.value for a in self.allowed_actions)} "
            f"(modify_size takes a quantity in (0, {self.max_quantity}])"
        )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class HumanResponse:
    """The answer. Typed by action; carries who gave it, because a takeover has an owner."""

    request_id: str
    action: HumanAction
    reviewer: str
    answered_at: datetime
    quantity: Decimal | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": PAUSE_FORMAT,
            "request_id": self.request_id,
            "action": self.action.value,
            "quantity": None if self.quantity is None else str(self.quantity),
            "reviewer": self.reviewer,
            "note": self.note,
            "answered_at": self.answered_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, blob: Mapping[str, Any]) -> HumanResponse:
        if blob.get("format") != PAUSE_FORMAT:
            raise PauseError(f"not a {PAUSE_FORMAT} response: format={blob.get('format')!r}")
        raw_quantity = blob.get("quantity")
        return cls(
            request_id=str(blob["request_id"]),
            action=HumanAction(blob["action"]),
            reviewer=str(blob["reviewer"]),
            answered_at=_parse_time(blob["answered_at"]),
            quantity=None if raw_quantity is None else Decimal(str(raw_quantity)),
            note=str(blob.get("note", "")),
        )


def check_response(request: PauseRequest, response: HumanResponse) -> None:
    """Accept an answer only if it fits this request — the typed pair, enforced.

    The MAF rule (``_request_info_mixin.py:59-73``): a response is routed only when both it and the
    pending request match a registered pair. Here that means it names this request, uses an allowed
    action, and carries exactly the fields that action needs — no quantity on approve or reject, a
    quantity in ``(0, proposal]`` on modify. A modify at the full proposed size is accepted and is
    the same as an approve; a quantity on an approve is refused rather than ignored, because a
    reviewer who typed one meant something by it.
    """
    if response.request_id != request.request_id:
        raise ResponseMismatch(
            f"the answer names {response.request_id!r}, the request is {request.request_id!r}"
        )
    if response.action not in request.allowed_actions:
        raise ResponseMismatch(
            f"{response.action.value!r} is not an answer this request accepts "
            f"({', '.join(a.value for a in request.allowed_actions)})"
        )
    if not response.reviewer.strip():
        raise ResponseMismatch(
            "an answer must name its reviewer: a takeover with no owner is not a takeover"
        )
    if response.action is HumanAction.MODIFY_SIZE:
        if response.quantity is None:
            raise ResponseMismatch("modify_size needs a quantity")
        if not response.quantity.is_finite() or response.quantity <= 0:
            raise ResponseMismatch(
                f"modify_size quantity must be positive, got {response.quantity}; to refuse the "
                f"trade, answer reject"
            )
        if response.quantity > request.max_quantity:
            raise ResponseMismatch(
                f"modify_size may only reduce: {response.quantity} exceeds the proposed "
                f"{request.max_quantity}. A human may release a hold, not originate a larger trade"
            )
    elif response.quantity is not None:
        raise ResponseMismatch(
            f"{response.action.value} takes no quantity; to change the size, answer modify_size"
        )


def resolve_intent(proposal: Intent, response: HumanResponse) -> Intent:
    """The intent a human's answer releases. May only ever be the proposal or something smaller.

    Checked rather than assumed, like `escalation.apply`: a result that enlarged the position or
    changed its side would be a human originating a trade, and the property is enforced by raising.
    """
    if response.action is HumanAction.APPROVE:
        released = proposal
    elif response.action is HumanAction.MODIFY_SIZE:
        if response.quantity is None:
            raise ResponseMismatch("modify_size needs a quantity")
        released = replace(proposal, quantity=response.quantity)
    else:
        released = replace(proposal, verdict=Verdict.NO_TRADE, quantity=Decimal("0"))
    if released.side is not proposal.side or released.quantity > proposal.quantity:
        raise PauseError(
            "a human answer produced a larger or reversed position than the model proposed"
        )
    if released.symbol != proposal.symbol:
        raise PauseError("a human answer changed the instrument")
    return released


# --- the restricted unpickler --------------------------------------------------------------------

_ALLOWED_PREFIX = "argus."
_BUILTIN_ALLOWED: frozenset[str] = frozenset({
    # Taken from MAF `_checkpoint_encoding.py:108-150`, trimmed to what a desk state holds.
    "builtins:object", "builtins:int", "builtins:float", "builtins:str", "builtins:bytes",
    "builtins:bool", "builtins:set", "builtins:frozenset", "builtins:list", "builtins:dict",
    "builtins:tuple", "builtins:complex", "builtins:slice", "builtins:range",
    "copyreg:_reconstructor",
    "datetime:datetime", "datetime:date", "datetime:time", "datetime:timedelta",
    "datetime:timezone",
    "decimal:Decimal",
    "collections:OrderedDict", "collections:defaultdict", "collections:deque",
    "collections:Counter",
})
_GETATTR_KEYS: frozenset[str] = frozenset({"builtins:getattr", "__builtin__:getattr"})
_BLOCKED_PREFIX = "argus.decision.pause:"
"""This module's own loaders may never be reconstructed from a continuation (MAF blocks its own
encoder and decoder the same way, `_checkpoint_encoding.py:98-103`)."""


def _type_key(obj: type) -> str:
    return f"{obj.__module__}:{obj.__qualname__}"


class _RestrictedUnpickler(pickle.Unpickler):
    """Adapted from MAF `_checkpoint_encoding.py:158-223` (MIT). See the module docstring.

    Not a security boundary on its own — MAF's own docstring says the same of theirs — which is why
    the byte hash is checked first. It is the second wall: a continuation that somehow passed the
    hash still cannot name ``os.system``.
    """

    def _allowed(self, resolved: type) -> bool:
        key = _type_key(resolved)
        if key.startswith(_BLOCKED_PREFIX):
            return False
        return key in _BUILTIN_ALLOWED or resolved.__module__.startswith(_ALLOWED_PREFIX)

    def _restricted_getattr(self, obj: Any, name: str) -> Any:
        if not isinstance(obj, type) or not isinstance(name, str):
            raise pickle.UnpicklingError("blocked: attribute traversal on a non-type")
        resolved = getattr(obj, name)
        if isinstance(resolved, type):
            if not self._allowed(resolved):
                raise pickle.UnpicklingError(f"blocked nested type {_type_key(resolved)}")
            return resolved
        if isinstance(obj, EnumMeta) and self._allowed(obj):
            members: Mapping[str, Any] = obj.__members__
            if name in members and members[name] is resolved:
                return resolved
        raise pickle.UnpicklingError(f"blocked attribute {_type_key(obj)}.{name}")

    def find_class(self, module: str, name: str) -> Any:
        key = f"{module}:{name}"
        if key.startswith(_BLOCKED_PREFIX):
            raise pickle.UnpicklingError(f"blocked: {key}")
        if key in _GETATTR_KEYS:
            return self._restricted_getattr
        if key in _BUILTIN_ALLOWED:
            return super().find_class(module, name)
        if module.startswith(_ALLOWED_PREFIX):
            # Dotted names traverse attributes of an allowed module; only top-level classes pass.
            if "." in name:
                raise pickle.UnpicklingError(f"blocked: {key}")
            resolved = super().find_class(module, name)
            if isinstance(resolved, type):
                return resolved
            raise pickle.UnpicklingError(f"blocked non-type global {key}")
        raise pickle.UnpicklingError(f"blocked: {key} is not a type a desk continuation holds")


def restricted_loads(payload: bytes) -> Any:
    return _RestrictedUnpickler(io.BytesIO(payload)).load()


# --- the store -----------------------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    """Write-then-rename: a kill mid-write leaves the old file or the new one, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _safe_id(decision_id: str) -> str:
    cleaned = _SAFE.sub("_", decision_id).strip("._") or "decision"
    if len(cleaned) > 80:
        digest = hashlib.sha256(decision_id.encode()).hexdigest()[:10]
        cleaned = f"{cleaned[:69]}-{digest}"
    return cleaned


LIFECYCLE = ("paused", "answered", "resumed", "expired", "superseded")
"""The events that change a request's status. ``decided`` is per decision, not per request."""


@dataclass(frozen=True, slots=True)
class Prior:
    """What the log says already happened to this decision — the input to LangGraph's replay."""

    request: PauseRequest
    status: str
    response: HumanResponse | None
    same_proposal: bool


class PauseStore:
    """Requests, continuations, answers and the event log, under one directory.

    The event log is the commit point. A request directory is written first and the ``paused``
    event appended last, so a kill between the two leaves an orphan directory that nothing lists —
    never a listed request whose continuation is missing.
    """

    def __init__(self, root: Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    def _dir(self, request_id: str) -> Path:
        if _SAFE.search(request_id) or request_id in {"", ".", ".."}:
            raise PauseError(f"{request_id!r} is not a request id")
        return self.root / "requests" / request_id

    # -- the log --

    def append_event(self, kind: str, *, at: datetime, **fields: Any) -> dict[str, Any]:
        event = {"event": kind, "at": at.isoformat(), **fields}
        line = canonical(event)
        self.root.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event

    def events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        out: list[dict[str, Any]] = []
        for raw in self.events_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                # A kill mid-append can leave one torn final line. It is skipped and not repaired:
                # the event it would have recorded did not commit, which is what a torn line means.
                continue
        return out

    def status(self, request_id: str) -> str | None:
        status: str | None = None
        for event in self.events():
            if event.get("request_id") == request_id and event.get("event") in LIFECYCLE:
                status = str(event["event"])
        return status

    def request_ids(self) -> list[str]:
        seen: list[str] = []
        for event in self.events():
            if event.get("event") == "paused":
                rid = str(event["request_id"])
                if rid not in seen:
                    seen.append(rid)
        return seen

    # -- writing --

    def record_decision(
        self,
        *,
        decision_id: str,
        symbol: str,
        at: datetime,
        reachable: bool,
        escalation: Escalation,
        model_requested_review: bool,
    ) -> None:
        """One ``decided`` event per desk decision — the denominator the takeover rate needs."""
        self.append_event(
            "decided", at=at, decision_id=decision_id, symbol=symbol, reachable=reachable,
            escalation=escalation.as_dict(), model_requested_review=model_requested_review,
        )

    def pause(
        self,
        continuation: Resumable,
        *,
        decision_id: str,
        symbol: str,
        as_of: datetime,
        proposal: Intent,
        escalation: Escalation,
        pipeline_signature: str,
        answer_window: timedelta = DEFAULT_ANSWER_WINDOW,
    ) -> PauseRequest:
        """Persist the continuation and the request, verify both read back, then commit."""
        if not escalation.required:
            raise PauseError("a pause needs an escalation; nothing here asks for a human")
        if as_of.tzinfo is None:
            raise PauseError("a pause is never naive: as_of carries no timezone")
        state = continuation.signature_state()
        signature = state_hash(state)
        sequence = sum(
            1 for e in self.events()
            if e.get("event") == "paused" and e.get("decision_id") == decision_id
        )
        request = PauseRequest(
            request_id=f"{_safe_id(decision_id)}-p{sequence}",
            decision_id=decision_id,
            sequence=sequence,
            symbol=symbol,
            as_of=as_of,
            expires_at=as_of + answer_window,
            proposal=proposal,
            proposal_hash=hash_intent(proposal),
            escalation=escalation,
            state_hash=signature,
            pipeline_signature=pipeline_signature,
        )
        payload = pickle.dumps(continuation, protocol=pickle.HIGHEST_PROTOCOL)
        blob = {
            "format": PAUSE_FORMAT,
            "request_id": request.request_id,
            "pipeline_signature": pipeline_signature,
            "state_hash": signature,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "payload_b64": base64.b64encode(payload).decode("ascii"),
            "state": json.loads(canonical(state)),
        }
        folder = self._dir(request.request_id)
        _atomic_write(folder / "continuation.json", json.dumps(blob, indent=1))
        _atomic_write(folder / "request.json", json.dumps(request.as_dict(), indent=2))
        # Read it back through the same path a resume will take, *now*. The worst moment to learn a
        # continuation cannot be restored is after a human has answered it.
        self.load_continuation(request.request_id, pipeline_signature=pipeline_signature)
        self.append_event(
            "paused", at=as_of, request_id=request.request_id, decision_id=decision_id,
            sequence=sequence, symbol=symbol, state_hash=signature,
            proposal_hash=request.proposal_hash,
            triggers=[t.value for t in escalation.triggers],
            expires_at=request.expires_at.isoformat(),
        )
        return request

    def answer(self, response: HumanResponse, *, now: datetime) -> PauseRequest:
        request = self.load_request(response.request_id)
        status = self.status(request.request_id)
        if status != "paused":
            raise AlreadyResolved(f"{request.request_id} is {status or 'unknown'}, not pending")
        if now > request.expires_at:
            self.append_event("expired", at=now, request_id=request.request_id,
                              decision_id=request.decision_id, stage="answer")
            raise PauseExpired(
                f"{request.request_id} closed at {request.expires_at.isoformat()}; the proposal "
                f"was priced at {request.as_of.isoformat()} and is no longer the market"
            )
        check_response(request, response)
        _atomic_write(
            self._dir(request.request_id) / "response.json",
            json.dumps(response.as_dict(), indent=2),
        )
        self.append_event(
            "answered", at=response.answered_at, request_id=request.request_id,
            decision_id=request.decision_id, action=response.action.value,
            quantity=None if response.quantity is None else str(response.quantity),
            proposed_quantity=str(request.proposal.quantity), reviewer=response.reviewer,
            asked_at=request.as_of.isoformat(),
        )
        return request

    def expire(self, request: PauseRequest, *, now: datetime, stage: str) -> None:
        self.append_event("expired", at=now, request_id=request.request_id,
                          decision_id=request.decision_id, stage=stage)

    def supersede(self, request: PauseRequest, *, at: datetime, by_proposal: str) -> None:
        self.append_event("superseded", at=at, request_id=request.request_id,
                          decision_id=request.decision_id, by_proposal=by_proposal)

    def mark_resumed(
        self, request: PauseRequest, *, at: datetime, outcome: Mapping[str, Any], replayed: bool
    ) -> None:
        self.append_event(
            "resumed", at=at, request_id=request.request_id, decision_id=request.decision_id,
            replayed=replayed, outcome=dict(outcome),
        )

    # -- reading --

    def load_request(self, request_id: str) -> PauseRequest:
        path = self._dir(request_id) / "request.json"
        if not path.exists():
            raise PauseError(f"no request {request_id!r} under {self.root}")
        return PauseRequest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_response(self, request_id: str) -> HumanResponse | None:
        path = self._dir(request_id) / "response.json"
        if not path.exists():
            return None
        return HumanResponse.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_continuation(self, request_id: str, *, pipeline_signature: str) -> Any:
        """Restore the continuation, or refuse. Four checks, in the order that keeps them safe.

        1. The format is this one. 2. The pipeline that will resume it is the one that paused it
        (MAF's graph-signature rule). 3. The pickle's bytes hash to what was written — checked
        before any unpickling, so a tampered or torn file is never executed. 4. The restored
        object's state hashes to the recorded state hash — the property a kill must not break.
        """
        path = self._dir(request_id) / "continuation.json"
        if not path.exists():
            raise PauseError(f"no continuation for {request_id!r}")
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PauseIntegrityError(f"{request_id}: continuation is not JSON ({exc})") from exc
        if blob.get("format") != PAUSE_FORMAT:
            raise PauseIntegrityError(f"{request_id}: format {blob.get('format')!r}")
        if blob.get("pipeline_signature") != pipeline_signature:
            raise PauseSignatureError(
                f"{request_id} was paused by pipeline {str(blob.get('pipeline_signature'))[:16]}… "
                f"and this desk is {pipeline_signature[:16]}…; the stages or the continuation's "
                f"fields have changed since, so it is not resumed through a pipeline it never saw"
            )
        try:
            payload = base64.b64decode(blob["payload_b64"], validate=True)
        except (KeyError, ValueError) as exc:
            raise PauseIntegrityError(f"{request_id}: payload is not valid base64") from exc
        if hashlib.sha256(payload).hexdigest() != blob.get("payload_sha256"):
            raise PauseIntegrityError(
                f"{request_id}: the continuation's bytes do not match their recorded hash; it was "
                f"edited or torn after it was written, and it is refused before being unpickled"
            )
        try:
            restored = restricted_loads(payload)
        except (pickle.UnpicklingError, AttributeError, EOFError, TypeError) as exc:
            raise PauseIntegrityError(f"{request_id}: continuation refused ({exc})") from exc
        if not callable(getattr(restored, "signature_state", None)):
            raise PauseIntegrityError(f"{request_id}: the restored object is not a continuation")
        recomputed = state_hash(restored.signature_state())
        if recomputed != blob.get("state_hash"):
            raise PauseIntegrityError(
                f"{request_id}: restored state hashes to {recomputed[:16]}…, recorded "
                f"{str(blob.get('state_hash'))[:16]}…"
            )
        return restored

    def prior(self, decision_id: str, proposal_hash: str) -> Prior | None:
        """The latest request this decision already raised, and what became of it.

        LangGraph's replay (`types.py:1004-1011`), keyed by proposal as the module docstring says.
        """
        latest: str | None = None
        for event in self.events():
            if event.get("event") == "paused" and event.get("decision_id") == decision_id:
                latest = str(event["request_id"])
        if latest is None:
            return None
        request = self.load_request(latest)
        return Prior(
            request=request,
            status=self.status(latest) or "paused",
            response=self.load_response(latest),
            same_proposal=request.proposal_hash == proposal_hash,
        )

    def pending(self, *, now: datetime | None = None) -> list[PauseRequest]:
        out = []
        for rid in self.request_ids():
            if self.status(rid) != "paused":
                continue
            request = self.load_request(rid)
            if now is not None and now > request.expires_at:
                continue
            out.append(request)
        return out


# --- the takeover rate, from events --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HumanLoopSummary:
    """The takeover rate plus what only a real pause can measure."""

    takeover: TakeoverRate
    model_requested_review: int
    paused: int
    answered: int
    approved: int
    modified: int
    rejected: int
    expired: int
    superseded: int
    pending: int
    resumed: int
    replayed: int
    median_answer_seconds: float | None

    @property
    def override_rate(self) -> float | None:
        """How often a human changed what the machine proposed, over the answers given.

        None, not zero, when nobody has answered — zero would read as "humans always agreed".
        """
        if self.answered == 0:
            return None
        return (self.modified + self.rejected) / self.answered

    @property
    def unanswered_rate(self) -> float | None:
        """Pauses that closed with no answer. A pause nobody answers is a halt, not oversight."""
        closed = self.answered + self.expired
        if closed == 0:
            return None
        return self.expired / closed

    def as_dict(self) -> dict[str, Any]:
        return {
            "takeover": self.takeover.as_dict(),
            "model_requested_review": self.model_requested_review,
            "paused": self.paused,
            "answered": self.answered,
            "approved": self.approved,
            "modified": self.modified,
            "rejected": self.rejected,
            "expired": self.expired,
            "superseded": self.superseded,
            "pending": self.pending,
            "resumed": self.resumed,
            "replayed": self.replayed,
            "median_answer_seconds": self.median_answer_seconds,
            "override_rate": None if self.override_rate is None else round(self.override_rate, 5),
            "unanswered_rate": (
                None if self.unanswered_rate is None else round(self.unanswered_rate, 5)
            ),
        }


def summarise(events: Iterable[Mapping[str, Any]]) -> HumanLoopSummary:
    """Rebuild the takeover rate and the human-loop counts from the log alone.

    A decision re-run under the same id (the replay case) is counted once, at its latest record:
    counting it twice would inflate both the numerator and the denominator with a decision that was
    only made once.
    """
    decided: dict[str, tuple[bool, Escalation, bool]] = {}
    status: dict[str, str] = {}
    answers: dict[str, Mapping[str, Any]] = {}
    replayed = 0
    waits: list[float] = []
    for event in events:
        kind = event.get("event")
        if kind == "decided":
            decided[str(event["decision_id"])] = (
                bool(event["reachable"]),
                Escalation.from_dict(dict(event["escalation"])),
                bool(event.get("model_requested_review", False)),
            )
        elif kind in LIFECYCLE:
            rid = str(event["request_id"])
            status[rid] = str(kind)
            if kind == "answered":
                answers[rid] = event
                asked = event.get("asked_at")
                if asked:
                    waits.append(
                        (_parse_time(str(event["at"])) - _parse_time(str(asked))).total_seconds()
                    )
            elif kind == "resumed" and event.get("replayed"):
                replayed += 1
    outcomes = [(reachable, esc) for reachable, esc, _ in decided.values()]
    actions = [str(a["action"]) for a in answers.values()]
    ordered = sorted(waits)
    median: float | None = None
    if ordered:
        mid = len(ordered) // 2
        median = (
            ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        )
    return HumanLoopSummary(
        takeover=takeover_rate(outcomes),
        model_requested_review=sum(1 for *_, review in decided.values() if review),
        paused=len(status),
        answered=len(answers),
        approved=actions.count(HumanAction.APPROVE.value),
        modified=actions.count(HumanAction.MODIFY_SIZE.value),
        rejected=actions.count(HumanAction.REJECT.value),
        expired=sum(1 for s in status.values() if s == "expired"),
        superseded=sum(1 for s in status.values() if s == "superseded"),
        pending=sum(1 for s in status.values() if s == "paused"),
        resumed=sum(1 for s in status.values() if s == "resumed"),
        replayed=replayed,
        median_answer_seconds=median,
    )


# --- CLI -----------------------------------------------------------------------------------------


_ACTION_WORDS = {
    "approve": HumanAction.APPROVE,
    "reject": HumanAction.REJECT,
    "modify": HumanAction.MODIFY_SIZE,
    "modify_size": HumanAction.MODIFY_SIZE,
}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m argus.decision.pause",
        description="list, inspect and answer the decisions the desk paused for a human",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="pause store directory")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="pending requests (all requests with --all)")
    listing.add_argument("--all", action="store_true")
    listing.add_argument("--at", help="ISO-8601 instant to judge expiry against (default: now)")
    show = sub.add_parser("show", help="one request, its state and its answer schema")
    show.add_argument("request_id")
    answer = sub.add_parser("answer", help="answer a pending request")
    answer.add_argument("request_id")
    answer.add_argument("action", choices=sorted(_ACTION_WORDS))
    answer.add_argument("--quantity", help="required for modify: the size to release")
    answer.add_argument("--reviewer", required=True, help="who is taking this decision")
    answer.add_argument("--note", default="")
    answer.add_argument("--at", help="ISO-8601 instant of the answer (default: now)")
    sub.add_parser("rate", help="takeover rate and human-loop counts, from the event log")
    args = parser.parse_args(argv)

    store = PauseStore(args.root)
    out = sys.stdout

    def instant(raw: str | None) -> datetime:
        return datetime.now(UTC) if raw is None else _parse_time(raw)

    if args.command == "list":
        now = instant(args.at)
        rids = store.request_ids()
        shown = 0
        for rid in rids:
            status = store.status(rid) or "unknown"
            request = store.load_request(rid)
            if status == "paused" and now > request.expires_at:
                status = "paused (window closed)"
            if not args.all and status != "paused":
                continue
            out.write(f"[{status}] {request.render()}\n\n")
            shown += 1
        if shown == 0:
            out.write("nothing is waiting for a human\n" if not args.all else "no requests\n")
        return 0

    if args.command == "show":
        request = store.load_request(args.request_id)
        response = store.load_response(args.request_id)
        out.write(json.dumps({
            "status": store.status(args.request_id),
            "request": request.as_dict(),
            "response": None if response is None else response.as_dict(),
        }, indent=2) + "\n")
        return 0

    if args.command == "answer":
        try:
            quantity = None if args.quantity is None else Decimal(args.quantity)
        except InvalidOperation:
            sys.stderr.write(f"--quantity {args.quantity!r} is not a number\n")
            return 2
        at = instant(args.at)
        response = HumanResponse(
            request_id=args.request_id, action=_ACTION_WORDS[args.action],
            reviewer=args.reviewer, answered_at=at, quantity=quantity, note=args.note,
        )
        try:
            store.answer(response, now=at)
        except PauseError as exc:
            sys.stderr.write(f"refused: {exc}\n")
            return 1
        out.write(
            f"recorded: {response.action.value} on {args.request_id} by {args.reviewer}. The desk "
            f"resumes it from the paused point on its next pass; the Constitution and the mandate "
            f"still rule on what you released.\n"
        )
        return 0

    out.write(json.dumps(summarise(store.events()).as_dict(), indent=2) + "\n")
    return 0


__all__ = [
    "DEFAULT_ANSWER_WINDOW",
    "DEFAULT_ROOT",
    "LIFECYCLE",
    "PAUSE_FORMAT",
    "AlreadyResolved",
    "HumanAction",
    "HumanLoopSummary",
    "HumanResponse",
    "PauseError",
    "PauseExpired",
    "PauseIntegrityError",
    "PauseRequest",
    "PauseSignatureError",
    "PauseStore",
    "Prior",
    "ResponseMismatch",
    "Resumable",
    "canonical",
    "check_response",
    "intent_from_dict",
    "intent_to_dict",
    "main",
    "resolve_intent",
    "restricted_loads",
    "state_hash",
    "summarise",
]


if __name__ == "__main__":
    raise SystemExit(main())
