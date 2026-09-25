"""The trajectory circuit-breaker, and the retry policy every external call goes through.

`risk/circuit.py` is a *book* breaker: it reads realised drawdown and loss streaks and tightens the
desk's appetite. It cannot see the thing this module watches, which is the **decision process
itself** going wrong — the model returning something that is not a decision, the desk proposing the
same order again and again as if it had not noticed it already holds it, a call that never answers.
Those are agent failures, not market ones, and a book breaker fires on them only after they have
cost money. This module fires before.

**Three things, one outcome.**

1. *A retry policy with a named give-up* around the Meta-PM call and, through
   :func:`call_with_retry`, any venue call. A failure is classified before it is retried: only a
   transient fault (connection, timeout, rate limit, 5xx) is worth another attempt; an invalid
   output or an auth failure is not, because retrying a deterministic failure buys the same failure
   and, on a metered key, pays for it.
2. *A trajectory breaker* over the desk's decision history: a step budget, a run of invalid model
   outputs, and a run of functionally-equivalent order intents.
3. *One outcome for every stop:* the decision degrades to a safe no-op, the reason is recorded in
   the decision's notes and in :attr:`DeskRun.circuit`, and **no exception leaves the desk loop**. A
   breaker that raises is a breaker that takes the cycle — and every symbol still to be decided in
   it — down with the one decision it was protecting.

**Taken, file by file (all read at source in the local research clones).**

* WebArena ``run.py:161-216`` (Apache-2.0) and VisualWebArena ``run.py:198-244`` (MIT) —
  ``early_stop``: three checks in a fixed order, the step budget first, then *k* consecutive parse
  failures, then *k* equivalent actions (``is_equivalent``, not string equality); defaults 3 and 3
  (webarena ``run.py:101-110``). Kept: the order, the defaults, and the rule that the stop carries a
  reason string. **Changed:** WebArena's step budget is per task; a trading desk has no task end, so
  the budget here is decisions per symbol per rolling window. And WebArena's ``is_equivalent``
  compares browser actions; ours is :func:`equivalent` — same instrument, same side, size within a
  tolerance, inside a time window — because two orders for 50 and 51 NVDA an hour apart are the same
  action for every purpose a loop detector has, and a literal-string match would miss them.
* TravelPlanner ``agents/tool_agents.py:216-226`` (MIT) — the literal-repeat detector (the same
  action three times in a row ends the episode) and ``:255-270``, a per-tool retry ceiling with
  ``invalidAction`` counted as its own tool. Kept: the ceiling of three, and invalid outputs counted
  separately from repeats. Rejected: ending the whole run — a desk that stops deciding every symbol
  because one symbol looped has turned a local fault into a global outage.
* browser_use ``agent/views.py:157-245`` (MIT) — ``ActionLoopDetector``: a rolling window of action
  hashes and an **escalating nudge** (5 / 8 / 12 repeats) that is *told to the model* rather than
  enforced. Kept: the nudge, as a ``[circuit]`` line in the frame one repeat before the trip, so the
  model is warned before it is overruled. Changed: browser_use never blocks; with money at stake the
  last rung is a stop, not a louder suggestion.
* LangGraph ``libs/langgraph/langgraph/types.py:427-446`` (MIT) — ``RetryPolicy`` fields and
  defaults (0.5s initial, x2 backoff, 128s cap, 3 attempts, jitter) and
  ``pregel/_retry.py:667-672`` — the interval formula ``min(max_interval, initial *
  factor**(attempt-1))`` plus ``uniform(0, 1)`` jitter. Adapted, with an injectable clock, sleep and
  seeded RNG so every retry path is testable without waiting. ``_internal/_retry.py:1-29``
  (``default_retry_on``: connection errors and 5xx retry, value/type errors do not) is the model for
  :func:`classify`. Its ``TimeoutPolicy`` documents that a timeout on synchronous code is
  cooperative; ours is the same in spirit — :func:`call_with_retry` stops *waiting* at the deadline
  and the abandoned call finishes on a daemon thread that cannot hold the process open.
* page_agent ``packages/llms/src/errors.ts:30-52`` (MIT) — a fixed enum of error types with
  ``retryable`` *derived from the type*, not decided at the call site; and
  ``packages/core/src/PageAgentCore.ts:117-132`` — every retry is a visible entry in the same trail
  as real actions. Kept as :class:`FaultKind` and :meth:`CallOutcome.render`. Its three-way event
  separation (``PageAgentCore.ts:44-59``: persistent history / transient activity / an error log
  kept out of the model's context) is the shape of :class:`CircuitRecord`: ``trips`` are persistent
  (they change the decision and are counted in the takeover rate), ``nudges`` are transient context,
  and ``faults`` are the error log — recorded, never fed back to the model as evidence.

**Measured** in `eval/circuit_replay.py` (``data/circuit_replay.json``): the recorded paper ledger
replayed through :func:`assess`, and every injected fault family caught with the right reason.
"""

from __future__ import annotations

import queue
import threading
import time
import urllib.error
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from random import Random
from typing import Any, Generic, TypeVar

T = TypeVar("T")

# --- the retry policy --------------------------------------------------------------------------


class FaultKind(StrEnum):
    """What went wrong with a call. Retryability belongs to the kind (page_agent errors.ts)."""

    TRANSIENT = "transient"
    """Connection refused or reset, rate limit, a 5xx. The same request may well succeed."""

    TIMEOUT = "timeout"
    """No answer inside the deadline. Retryable once the deadline allows."""

    INVALID_OUTPUT = "invalid_output"
    """An answer arrived and was not a decision. The client already re-asked (qwen.py:369-487);
    asking again at this layer pays for the same failure."""

    PERMANENT = "permanent"
    """Auth, a bad request, a bug. Retrying cannot help."""

    @property
    def retryable(self) -> bool:
        return self in (FaultKind.TRANSIENT, FaultKind.TIMEOUT)


_TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def classify(exc: BaseException) -> FaultKind:
    """Map an exception to a :class:`FaultKind`. LangGraph ``default_retry_on``, typed."""
    if isinstance(exc, TimeoutError):
        return FaultKind.TIMEOUT
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int):
        if status in _TRANSIENT_STATUS:
            return FaultKind.TRANSIENT
        if 400 <= status < 500:
            return FaultKind.PERMANENT
    if isinstance(exc, ConnectionError | urllib.error.URLError):
        return FaultKind.TRANSIENT
    name = type(exc).__name__
    text = str(exc).lower()
    if name == "QwenError":
        if "json" in text or "valid" in text or "missing" in text:
            return FaultKind.INVALID_OUTPUT
        if any(t in text for t in ("timed out", "timeout", "gateway", "503", "502", "504", "429")):
            return FaultKind.TRANSIENT
        return FaultKind.PERMANENT
    if isinstance(exc, ValueError | KeyError | TypeError):
        return FaultKind.INVALID_OUTPUT
    if isinstance(exc, OSError):
        return FaultKind.TRANSIENT
    return FaultKind.PERMANENT


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """LangGraph ``types.py:427-446``'s fields and defaults, plus a per-attempt timeout."""

    initial_interval: float = 0.5
    backoff_factor: float = 2.0
    max_interval: float = 128.0
    max_attempts: int = 3
    jitter: bool = True
    timeout_seconds: float | None = None
    """Per attempt. ``None`` leaves the deadline to the client's own transport timeout."""

    def interval(self, attempt: int, rng: Random) -> float:
        """Seconds to wait after failed attempt ``attempt`` (1-based). `_retry.py:667-672`."""
        base = min(self.max_interval, self.initial_interval * self.backoff_factor ** (attempt - 1))
        return base + rng.uniform(0, 1) if self.jitter else base


PM_POLICY = RetryPolicy(max_attempts=1, timeout_seconds=900.0)
"""The Meta-PM call: **one** attempt at this layer, a 900s deadline.

One, because the Qwen client already retries transport failures with 2**n backoff
(`llm/qwen.py:501-528`) and re-asks on invalid JSON three times (`qwen.py:369-487`); a second layer
of retries would multiply paid calls on the one path where the key is metered. 900s is eight times
the longest streamed full-reasoning call measured (112s, `qwen.py:291-295`), so a healthy call never
reaches it and a hung one cannot hold the cycle."""

VENUE_POLICY = RetryPolicy(max_attempts=3, timeout_seconds=20.0)
"""For a venue call: LangGraph's three attempts, and a deadline far above a normal REST round trip.
Only transient faults are retried, so a rejected order is never re-sent.

**Only for idempotent calls.** A timed-out order submission may still have reached the venue, so an
order may go through this policy only if it carries the deterministic client order id
(`execution/orders.py` ``deterministic_client_order_id``), which the venue rejects as a duplicate on
a second arrival. A bare order without one must use ``max_attempts=1``: a retry that doubles a
position is worse than the timeout it was meant to survive."""


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    number: int
    fault: FaultKind | None
    error: str
    seconds: float
    waited: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.number, "fault": None if self.fault is None else self.fault.value,
            "error": self.error, "seconds": round(self.seconds, 3), "waited": round(self.waited, 3),
        }


@dataclass(frozen=True)
class CallOutcome(Generic[T]):
    """What a guarded call produced: a value, or a named give-up. Never an exception."""

    name: str
    value: T | None
    attempts: tuple[AttemptRecord, ...]
    gave_up: str | None = None

    @property
    def ok(self) -> bool:
        return self.gave_up is None

    @property
    def fault(self) -> FaultKind | None:
        return self.attempts[-1].fault if self.attempts else None

    def render(self) -> list[str]:
        """Every retry is visible, like page_agent's history entries; a clean call says nothing."""
        lines = [
            f"[circuit] {self.name} attempt {a.number} failed ({a.fault}): {a.error}"
            for a in self.attempts if a.fault is not None
        ]
        if self.gave_up is not None:
            lines.append(f"[circuit] {self.gave_up}")
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "ok": self.ok, "gave_up": self.gave_up,
            "attempts": [a.as_dict() for a in self.attempts],
        }


def _run_with_timeout(fn: Callable[[], T], seconds: float | None) -> T:
    """Run ``fn``; raise ``TimeoutError`` if it has not returned in ``seconds``.

    The call runs on a daemon thread, so an abandoned call can never keep the interpreter alive.
    Cooperative, like LangGraph's ``TimeoutPolicy``: the caller stops waiting, the call is not
    killed, and its eventual result is discarded.
    """
    if seconds is None:
        return fn()
    box: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            box.put((True, fn()))
        except BaseException as exc:  # carried back to the caller's thread, re-raised there
            box.put((False, exc))

    threading.Thread(target=target, name="circuit-call", daemon=True).start()
    try:
        ok, payload = box.get(timeout=seconds)
    except queue.Empty:
        raise TimeoutError(f"no answer in {seconds:g}s") from None
    if ok:
        return payload  # type: ignore[no-any-return]
    raise payload


def call_with_retry(
    fn: Callable[[], T],
    *,
    name: str,
    policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    rng: Random | None = None,
    classifier: Callable[[BaseException], FaultKind] = classify,
    propagate: tuple[type[BaseException], ...] = (),
) -> CallOutcome[T]:
    """Call ``fn`` under ``policy``. Returns a :class:`CallOutcome`; never raises for ``fn``'s sake.

    The give-up is *named*: ``"gave up on meta_pm.decide after 1 attempt(s): timeout — no answer in
    900s"`` is what the decision records, so a stop is attributable from the trail alone.

    ``propagate`` names exceptions that are the *caller's* control rather than a fault of the call —
    a spent token budget (`llm/qwen.py` ``BudgetExhausted``, raised before any request is made) is
    the runner deciding the cycle is over, and turning it into a recorded no-op would write a
    decision the budget exists to prevent. Those are re-raised at once, unretried.
    """
    rng = rng or Random(0)
    attempts: list[AttemptRecord] = []
    waited = 0.0
    for number in range(1, max(1, policy.max_attempts) + 1):
        started = clock()
        try:
            value = _run_with_timeout(fn, policy.timeout_seconds)
        except Exception as exc:  # the whole point: nothing escapes as an exception
            if propagate and isinstance(exc, propagate):
                raise
            kind = classifier(exc)
            attempts.append(AttemptRecord(
                number, kind, f"{type(exc).__name__}: {str(exc)[:160]}", clock() - started, waited,
            ))
            if not kind.retryable or number >= policy.max_attempts:
                why = "not retryable" if not kind.retryable else "attempts exhausted"
                return CallOutcome(
                    name, None, tuple(attempts),
                    gave_up=(
                        f"gave up on {name} after {number} attempt(s) ({why}): {kind.value} — "
                        f"{type(exc).__name__}: {str(exc)[:120]}"
                    ),
                )
            waited = policy.interval(number, rng)
            sleep(waited)
            continue
        attempts.append(AttemptRecord(number, None, "", clock() - started, waited))
        return CallOutcome(name, value, tuple(attempts))
    raise AssertionError("unreachable: the loop returns on its last attempt")


# --- the trajectory ----------------------------------------------------------------------------


def aware(at: datetime) -> datetime:
    """A naive timestamp read as UTC.

    The ledger always writes aware ones; a caller may not, and a naive-vs-aware comparison raising
    inside the breaker would be the breaker taking the loop down.
    """
    return at if at.tzinfo is not None else at.replace(tzinfo=UTC)


class Trip(StrEnum):
    """Why the breaker stopped a decision. Each is one WebArena ``early_stop`` branch, or ours."""

    STEP_BUDGET = "step_budget"
    INVALID_OUTPUTS = "invalid_outputs"
    EQUIVALENT_ORDERS = "equivalent_orders"
    CALL_GAVE_UP = "call_gave_up"


INVALID_MARKERS: tuple[str, ...] = ("[unparseable verdict", "[circuit] no model decision")
"""Thesis prefixes that mark an answer as not-a-decision: `meta_pm._to_intent`'s unparseable
verdict, and this module's own degraded no-op. A deliberate ``human_review`` is *not* invalid — it
is the model asking for a human, which is a decision."""


@dataclass(frozen=True, slots=True)
class Step:
    """One decision in the trajectory, reduced to what equivalence and validity need."""

    symbol: str
    at: datetime
    verdict: str
    side: str
    quantity: Decimal
    valid: bool = True

    @property
    def opens_exposure(self) -> bool:
        return self.verdict in ("trade", "hedge") and self.quantity > 0

    @classmethod
    def from_record(cls, row: Any) -> Step | None:
        """A ledger `Entry` or its dict form. ``None`` for a row that is not a decision."""
        def get(key: str, default: Any = None) -> Any:
            if isinstance(row, Mapping):
                return row.get(key, default)
            return getattr(row, key, default)

        kind = get("kind", "decision")
        if kind not in (None, "decision"):
            return None
        raw_at = get("decided_at") or get("at")
        try:
            at = raw_at if isinstance(raw_at, datetime) else datetime.fromisoformat(str(raw_at))
            quantity = Decimal(str(get("quantity", "0")))
        except (ValueError, InvalidOperation):
            return None
        thesis = str(get("thesis", ""))
        return cls(
            symbol=str(get("symbol", "")), at=aware(at), verdict=str(get("verdict", "")).lower(),
            side=str(get("side", "")).lower(), quantity=quantity,
            valid=not thesis.startswith(INVALID_MARKERS),
        )


@dataclass(frozen=True, slots=True)
class Thresholds:
    """WebArena's defaults where they transfer (3 and 3), and a desk-shaped step budget."""

    max_steps: int = 48
    """Decisions on one symbol inside :attr:`window`: one every 30 minutes.

    **Found by replaying the ledger, 2026-09-25.** First set to 24 — one an hour, six times the
    four-cycle paper schedule (13:30, 15:30, 17:30, 19:30 UTC) — from the schedule alone. The replay
    of all 701 recorded decisions then tripped **10 times** (1.43 per 100), every one on NVDAUSDT
    on 2026-09-13, when 30 real decisions were taken inside 24 hours during manual cycles. A budget
    that stops real decisions is a false trip, so it was raised to 48, 1.6x the recorded maximum;
    ``data/circuit_replay.json`` carries the sensitivity table (trips at 24, 32 and 48)."""

    invalid_outputs: int = 3
    """Consecutive not-a-decision answers, across symbols: the model is shared, so three in a row
    is the endpoint failing, not one symbol being hard (webarena ``parsing_failure_th``)."""

    equivalent_orders: int = 3
    """The k-th equivalent exposure-opening intent on one symbol inside the window is stopped
    (webarena ``repeating_action_failure_th``, TravelPlanner's three-in-a-row)."""

    nudge_at: int = 2
    """At this many equivalent priors the model is told, before it decides (browser_use)."""

    size_tolerance: Decimal = Decimal("0.10")
    """Two sizes within 10% of the larger are the same order."""

    window: timedelta = timedelta(hours=24)


DEFAULT_THRESHOLDS = Thresholds()


def equivalent(a: Step, b: Step, *, tolerance: Decimal = DEFAULT_THRESHOLDS.size_tolerance) -> bool:
    """The trading ``is_equivalent``: same instrument, same side, size within ``tolerance``.

    Only exposure-opening intents can be equivalent. Repeated abstention is the desk's conservative
    default, not a loop — nothing reaches the venue, so there is nothing for the breaker to stop.
    """
    if not (a.opens_exposure and b.opens_exposure):
        return False
    if a.symbol != b.symbol or a.side != b.side:
        return False
    largest = max(a.quantity, b.quantity)
    return abs(a.quantity - b.quantity) <= tolerance * largest


@dataclass(frozen=True)
class Reading:
    """The breaker's answer for one decision."""

    trip: Trip | None
    reason: str
    nudge: str | None = None
    equivalent_priors: int = 0
    steps_in_window: int = 0
    invalid_run: int = 0

    @property
    def tripped(self) -> bool:
        return self.trip is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "trip": None if self.trip is None else self.trip.value, "reason": self.reason,
            "nudge": self.nudge, "equivalent_priors": self.equivalent_priors,
            "steps_in_window": self.steps_in_window, "invalid_run": self.invalid_run,
        }


def _in_window(history: Sequence[Step], current: Step, window: timedelta) -> list[Step]:
    now = aware(current.at)
    return [
        s for s in history if s.symbol == current.symbol and now - window <= aware(s.at) <= now
    ]


def nudge_for(
    history: Sequence[Step], *, symbol: str, now: datetime,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> str | None:
    """browser_use's escalating nudge, computed before the model decides.

    Counts the most-repeated exposure-opening order on ``symbol`` in the window. At
    ``nudge_at`` the model is told plainly that one more equivalent order will be stopped.
    """
    now = aware(now)
    recent = [s for s in history if s.symbol == symbol
              and now - thresholds.window <= aware(s.at) <= now and s.opens_exposure]
    best = 0
    anchor: Step | None = None
    for s in recent:
        n = sum(1 for o in recent if equivalent(s, o, tolerance=thresholds.size_tolerance))
        if n > best:
            best, anchor = n, s
    if anchor is None or best < thresholds.nudge_at:
        return None
    return (
        f"[circuit] you have proposed {anchor.verdict.upper()} {anchor.side.upper()} "
        f"~{anchor.quantity} {symbol} {best} time(s) in the last "
        f"{thresholds.window.total_seconds() / 3600:g}h. If the position already carries that "
        f"view, repeating the order adds size, not information; an equivalent order at this point "
        f"will be stopped by the circuit breaker."
    )


def assess(
    history: Sequence[Step], current: Step, *, thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> Reading:
    """WebArena ``early_stop``'s three checks, in its order, over a trading trajectory.

    ``history`` is every prior decision (any symbol, any order); ``current`` is the one being made.
    Pure: a replay of the ledger through this function is the same computation the desk ran live.
    """
    now = aware(current.at)
    ordered = sorted((s for s in history if aware(s.at) <= now), key=lambda s: aware(s.at))
    window = _in_window(ordered, current, thresholds.window)

    steps = len(window) + 1
    if steps > thresholds.max_steps:
        return Reading(
            Trip.STEP_BUDGET,
            f"step budget: {steps} decisions on {current.symbol} in "
            f"{thresholds.window.total_seconds() / 3600:g}h exceeds {thresholds.max_steps}",
            steps_in_window=steps,
        )

    run = 0
    if not current.valid:
        run = 1
        for s in reversed(ordered):
            if s.valid:
                break
            run += 1
    if run >= thresholds.invalid_outputs:
        return Reading(
            Trip.INVALID_OUTPUTS,
            f"{run} consecutive model answers were not decisions (threshold "
            f"{thresholds.invalid_outputs}); the model endpoint is failing, not one symbol",
            steps_in_window=steps, invalid_run=run,
        )

    priors = [s for s in window if equivalent(s, current, tolerance=thresholds.size_tolerance)]
    if current.opens_exposure and len(priors) + 1 >= thresholds.equivalent_orders:
        return Reading(
            Trip.EQUIVALENT_ORDERS,
            f"equivalent order {len(priors) + 1} of {thresholds.equivalent_orders}: "
            f"{current.verdict.upper()} {current.side.upper()} {current.quantity} {current.symbol} "
            f"matches {len(priors)} prior intent(s) in the last "
            f"{thresholds.window.total_seconds() / 3600:g}h (same side, size within "
            f"{thresholds.size_tolerance:.0%})",
            equivalent_priors=len(priors), steps_in_window=steps, invalid_run=run,
        )
    return Reading(None, "", equivalent_priors=len(priors), steps_in_window=steps,
                   invalid_run=run)


# --- the record the desk keeps ------------------------------------------------------------------


@dataclass
class CircuitRecord:
    """page_agent's three-way separation, for one decision.

    ``trips`` are persistent — they changed the decision and are what the takeover rate counts.
    ``nudges`` are transient context the model was shown. ``faults`` are the error log: every failed
    attempt, kept for audit and never fed back to the model as evidence.
    """

    trips: list[dict[str, Any]] = field(default_factory=list)
    nudges: list[str] = field(default_factory=list)
    faults: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tripped(self) -> bool:
        return bool(self.trips)

    def as_dict(self) -> dict[str, Any]:
        return {"trips": list(self.trips), "nudges": list(self.nudges),
                "faults": list(self.faults)}


def trips_per_100(trips: int, decisions: int) -> float | None:
    """The human-takeover-rate style count: stops per hundred decisions."""
    return None if decisions == 0 else round(100.0 * trips / decisions, 3)


__all__ = [
    "DEFAULT_THRESHOLDS",
    "INVALID_MARKERS",
    "PM_POLICY",
    "VENUE_POLICY",
    "AttemptRecord",
    "CallOutcome",
    "CircuitRecord",
    "FaultKind",
    "Reading",
    "RetryPolicy",
    "Step",
    "Thresholds",
    "Trip",
    "assess",
    "aware",
    "call_with_retry",
    "classify",
    "equivalent",
    "nudge_for",
    "trips_per_100",
]
