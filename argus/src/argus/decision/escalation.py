"""When the desk must stop and ask a human — and the rate at which it does.

Track 2's Open Theme names "human-takeover rate" as one of five things an agent benchmark should
measure. A teardown of every *trading* harness in the local corpus found none that measures how
often an agent hands control to a human or defines when it must. (Outside trading it exists:
τ²-bench defines transfer to a human in its domain policies and scores gold transfer tasks, and
HiL-Bench measures when an agent asks — an earlier version of this paragraph said no repository
anywhere did, corrected after the rival review of 2026-09-24.)
`research/architecture/agent-evaluation-harnesses.md` records the trading evidence file by file
— `live-trade-bench/systems/stock_system.py:48-75` has no abort path,
`TraderHarness/agents/protocol.py:12-21` returns `None` so there is nothing that *could* carry an
escalation, `agent-backtest-lab/abl/types.py:13-31` has three directions and no fourth state, and
`DARWIN/cto/llm.ts:66-72` clamps the model's output to a schema with no escalation verb.

**ARGUS had the state and not the policy.** `decision/verdicts.py:42` defines `HUMAN_REVIEW`, and
until this module the only thing that produced it was `agents/meta_pm.py:366-374`, on two
*structural* conditions: a verdict that opens exposure without naming a falsifier, and a verdict
that carries size while naming zero. Both are malformed-output triggers. Neither is a risk
condition. A desk that escalates only when the model returns a badly-shaped answer has not decided
what it should refuse to decide alone.

**The five conditions here are risk conditions, and each is deterministic.** No model is asked
whether to escalate — asking a model whether it should be overruled is not a control.

1. **UNRESOLVED_CONFLICT** — the analyst panel split directionally *and* the debate failed to
   converge. Two irreconcilable readings of the same evidence; picking one is a coin flip wearing a
   thesis.
2. **THESIS_ALREADY_REFUTED** — the adversary found an invalidation condition the thesis itself
   named to be already true at decision time. Acting on a thesis that is refuted on arrival is the
   clearest case there is.
3. **UNGROUNDED_FIGURE** — a numeric figure in the thesis does not resolve to a known fact. A
   decision justified by a number nobody can source must not reach the venue.
4. **UNDERLYING_HALTED** — the anchor equity is halted. The token keeps trading and the thing it
   references has stopped pricing; that is a human's call, not a threshold's.
5. **SIZE_BEYOND_MANDATE** — the position exceeds the share of the book this mandate permits a
   machine to take unattended.

**Escalation may only reduce.** This mirrors the Constitution's asymmetry and is enforced the same
way: an escalation converts an action into a hold-and-ask, sets quantity to zero, and can never
create, enlarge or reverse a position. :func:`apply` raises rather than returning if it is ever
asked to do otherwise, so the property is a build failure and not a convention.

**A desk that escalates everything is useless, which is why the rate is the measurement.** Zero
escalations across a long record means the conditions are dead code; a high rate means the desk
cannot operate unattended. Both are findings, and `eval/bench.py` reports the rate with the
denominator beside it rather than the count alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any

from argus.decision.verdicts import Intent, Verdict


class Trigger(StrEnum):
    """Why a decision was handed to a human. Each is checkable without a model."""

    UNRESOLVED_CONFLICT = "unresolved_conflict"
    THESIS_ALREADY_REFUTED = "thesis_already_refuted"
    UNGROUNDED_FIGURE = "ungrounded_figure"
    UNDERLYING_HALTED = "underlying_halted"
    SIZE_BEYOND_MANDATE = "size_beyond_mandate"


TRIGGER_REASONS: dict[Trigger, str] = {
    Trigger.UNRESOLVED_CONFLICT: (
        "the analyst panel split on direction and the debate did not converge, so the desk holds "
        "two irreconcilable readings of the same evidence and neither has earned the position"
    ),
    Trigger.THESIS_ALREADY_REFUTED: (
        "an invalidation condition the thesis named for itself is already true, so the position "
        "would be opened on a thesis that is refuted on arrival"
    ),
    Trigger.UNGROUNDED_FIGURE: (
        "a numeric figure in the thesis does not resolve to any known fact, so the justification "
        "contains a number nobody can source"
    ),
    Trigger.UNDERLYING_HALTED: (
        "the anchor equity is halted while the token continues to trade, so the reference this "
        "position is priced against has stopped pricing"
    ),
    Trigger.SIZE_BEYOND_MANDATE: (
        "the position exceeds the share of the book this mandate permits to be taken unattended"
    ),
}

DEFAULT_UNATTENDED_FRACTION = Decimal("0.25")
"""Largest share of the book a machine may take without a human.

A quarter, and the number is a policy choice rather than a measurement — which is why it is a
parameter with a stated default rather than a constant buried in a comparison. Real desks set this
by mandate and so does ours: `desk/personalisation.py` carries per-profile limits, and this is the
floor that applies when no profile says otherwise.
"""


class EscalationError(ValueError):
    """Raised rather than letting an escalation do anything but reduce."""


@dataclass(frozen=True, slots=True)
class Signals:
    """Everything the policy reads. Deterministic inputs only, no model in the loop.

    Constructed by the caller from the parts of a desk run that already exist — the conflict
    report, the adversary's challenge, the grounding report, the halt feed and the mandate — so
    that this module depends on none of them and can be tested without any of them.
    """

    directional_split: bool = False
    debate_converged: bool = True
    thesis_refuted_by_own_falsifier: bool = False
    refuted_condition: str = ""
    unresolved_figures: int = 0
    underlying_halted: bool = False
    halt_reason: str = ""
    position_fraction_of_book: Decimal = Decimal("0")
    unattended_fraction: Decimal = DEFAULT_UNATTENDED_FRACTION


@dataclass(frozen=True, slots=True)
class Escalation:
    """The decision to hand over, and everything a human needs to pick it up."""

    triggers: tuple[Trigger, ...]
    detail: tuple[str, ...]

    @property
    def required(self) -> bool:
        return bool(self.triggers)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(TRIGGER_REASONS[t] for t in self.triggers)

    def render(self) -> str:
        if not self.required:
            return "No escalation: the desk may act on this unattended."
        lines = ["HANDED TO A HUMAN — the desk declines to decide this alone:"]
        for trigger, reason, extra in zip(
            self.triggers, self.reasons, self.detail, strict=True
        ):
            lines.append(f"  {trigger.value}: {reason}")
            if extra:
                lines.append(f"    {extra}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "triggers": [t.value for t in self.triggers],
            "reasons": list(self.reasons),
            "detail": list(self.detail),
        }


def assess(intent: Intent, signals: Signals) -> Escalation:
    """Which escalation conditions this decision meets. All of them, not the first.

    Reporting only the first trigger would let a decision be cleared by fixing one condition while
    another still held, and would understate the takeover rate's causes in the benchmark.

    **An abstention is never escalated.** A desk that says no has already declined to act, and
    asking a human to approve doing nothing converts a decision into an interruption. Escalation is
    for decisions that would otherwise *reach the venue*.
    """
    if not intent.verdict.opens_exposure:
        return Escalation(triggers=(), detail=())

    triggers: list[Trigger] = []
    detail: list[str] = []

    if signals.directional_split and not signals.debate_converged:
        triggers.append(Trigger.UNRESOLVED_CONFLICT)
        detail.append("the panel disagreed on direction and the debate ended unresolved")

    if signals.thesis_refuted_by_own_falsifier:
        triggers.append(Trigger.THESIS_ALREADY_REFUTED)
        detail.append(signals.refuted_condition or "the refuted condition was not named")

    if signals.unresolved_figures > 0:
        triggers.append(Trigger.UNGROUNDED_FIGURE)
        detail.append(
            f"{signals.unresolved_figures} figure(s) in the thesis resolve to no known fact"
        )

    if signals.underlying_halted:
        triggers.append(Trigger.UNDERLYING_HALTED)
        detail.append(signals.halt_reason or "the halt reason was not published")

    if signals.position_fraction_of_book > signals.unattended_fraction:
        triggers.append(Trigger.SIZE_BEYOND_MANDATE)
        detail.append(
            f"{signals.position_fraction_of_book:.1%} of the book against an unattended limit of "
            f"{signals.unattended_fraction:.1%}"
        )

    return Escalation(triggers=tuple(triggers), detail=tuple(detail))


def apply(intent: Intent, escalation: Escalation) -> Intent:
    """Convert an intent into a hold-and-ask. May only ever reduce.

    Returns the intent unchanged when no escalation is required, so a caller can apply this
    unconditionally without branching — the branch is where a control gets skipped.
    """
    if not escalation.required:
        return intent
    if not intent.verdict.opens_exposure:
        raise EscalationError(
            "an escalation was raised on a decision that opens no exposure; escalation exists to "
            "stop an action reaching the venue, and there is no action here to stop"
        )
    escalated = Intent(
        symbol=intent.symbol,
        side=intent.side,
        quantity=Decimal("0"),
        verdict=Verdict.HUMAN_REVIEW,
        stated_confidence=intent.stated_confidence,
        thesis=intent.thesis,
        invalidation=intent.invalidation,
    )
    # The asymmetry, checked rather than assumed. An escalation that produced size, or that turned
    # a refusal into an action, would be the risk layer creating exposure — the exact thing the
    # Constitution forbids and the one property this module must never break.
    if escalated.quantity != Decimal("0"):
        raise EscalationError("an escalation produced a non-zero quantity")
    if escalated.verdict.opens_exposure:
        raise EscalationError("an escalation produced a verdict that opens exposure")
    return escalated


@dataclass(frozen=True, slots=True)
class TakeoverRate:
    """How often the desk handed over, over what, and broken down by cause."""

    escalations: int
    actionable: int
    """Decisions that would otherwise have reached the venue. The honest denominator.

    Dividing by *all* decisions would let a desk that abstains constantly report a takeover rate
    near zero while escalating every single decision it actually made.
    """

    total_decisions: int
    by_trigger: dict[str, int]

    @property
    def rate(self) -> float | None:
        """None when nothing was actionable — not zero. Zero would read as "never needed to ask"."""
        if self.actionable <= 0:
            return None
        return self.escalations / self.actionable

    @property
    def verdict(self) -> str:
        if self.rate is None:
            return (
                f"UNDEFINED over {self.total_decisions} decision(s): none would have reached the "
                f"venue, so there was nothing a human could have been asked about. This is the "
                f"state of a desk that has not yet traded, and it is not a takeover rate of zero"
            )
        if self.escalations == 0:
            return (
                f"0% over {self.actionable} actionable decision(s). Either the conditions never "
                f"fired or they are dead code; at this sample size those are not distinguishable"
            )
        if self.rate > 0.5:
            return (
                f"{self.rate:.0%} of {self.actionable} actionable decision(s) were handed to a "
                f"human. Above half, the desk is not operating unattended in any meaningful sense"
            )
        return (
            f"{self.rate:.0%} of {self.actionable} actionable decision(s) were handed to a human, "
            f"most often for {max(self.by_trigger, key=lambda k: self.by_trigger[k])}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "escalations": self.escalations,
            "actionable_decisions": self.actionable,
            "total_decisions": self.total_decisions,
            "rate": None if self.rate is None else round(self.rate, 5),
            "by_trigger": dict(sorted(self.by_trigger.items())),
            "verdict": self.verdict,
        }


def takeover_rate(
    outcomes: Sequence[tuple[bool, Escalation]], *, total_decisions: int | None = None
) -> TakeoverRate:
    """``outcomes`` are ``(would_have_reached_the_venue, escalation)`` per decision."""
    actionable = sum(1 for reachable, _ in outcomes if reachable)
    escalations = sum(1 for reachable, e in outcomes if reachable and e.required)
    by_trigger: dict[str, int] = {}
    for reachable, escalation in outcomes:
        if not reachable:
            continue
        for trigger in escalation.triggers:
            by_trigger[trigger.value] = by_trigger.get(trigger.value, 0) + 1
    return TakeoverRate(
        escalations=escalations,
        actionable=actionable,
        total_decisions=len(outcomes) if total_decisions is None else total_decisions,
        by_trigger=by_trigger,
    )


__all__ = [
    "DEFAULT_UNATTENDED_FRACTION",
    "TRIGGER_REASONS",
    "Escalation",
    "EscalationError",
    "Signals",
    "TakeoverRate",
    "Trigger",
    "apply",
    "assess",
    "takeover_rate",
]
