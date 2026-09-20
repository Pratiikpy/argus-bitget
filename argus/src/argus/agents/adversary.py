"""The standing adversary — one model whose only job is to falsify the decision just made.

TradingAgents runs a Bull Researcher against a Bear Researcher for N rounds and has a Research
Manager judge the transcript (`bull_researcher.py`, `bear_researcher.py`,
`research_manager.py:17-70`). It is the reference implementation for this idea and the audit in
`research/architecture/agent-architecture-audit.md` ranks its absence here as a HIGH-weight gap.

**What is deliberately not copied.** Assigning a model the bull side produces a bull case whether or
not one exists: an agent told to argue for buying will argue for buying a company in liquidation.
That is debate as theatre, and its output is uninformative precisely when the evidence is one-sided
— the case where a desk most needs to know. The transcript looks like rigour and contains none.

**What is built instead.** The adversary is not given a side. It is given *the decision that was
actually made*, with its thesis, its stated invalidation conditions and the same evidence, and asked
one question: **what would have to be true for this to be wrong, and is any of it already true?**

Three checks come out of that, and the third is the one no reference system performs:

1. **The strongest counter-case** — the best argument against the thesis, in the adversary's words.
2. **The weakest link** — which single piece of evidence the thesis leans on hardest, and what
   happens to the conclusion if that piece is wrong.
3. **Whether the thesis is already refuted.** Every intent that opens exposure must state what
   would make it wrong (`Intent.invalidation`, enforced at construction). A thesis whose own
   falsifier is *already satisfied by the evidence in front of it* is self-refuting, and nothing
   downstream would notice: the Constitution reasons about size and session, the grounding checker
   about whether figures resolve, the conflict checker about whether analysts agreed. None of them
   reads the invalidation conditions back against the evidence. This does.

**The asymmetry is the same one the Constitution obeys, and is enforced the same way.** A challenge
may refuse a trade or cut its size. It may never create one, enlarge one, reverse a side, or turn an
abstention into a position — `apply_constraint` raises rather than logs if it tries. An adversary
that could talk the desk *into* a trade would be a second strategy wearing a critic's clothes, and
the decision would no longer be the decision-maker's.

Failure is not fatal. If the adversary is unavailable, malformed or times out, the decision stands
unchanged and the absence is recorded by name — a critic that takes the desk down when its upstream
is slow is worse than no critic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol

from argus.agents.quarantine import render_for_prompt
from argus.decision.verdicts import (
    ConstitutionRuling,
    ConstitutionVerdict,
    Intent,
    apply_constraint,
)
from argus.truth.evidence import Evidence

MAX_EVIDENCE = 24
"""Evidence items shown to the adversary. Bounded so one noisy cycle cannot blow the prompt."""

SEVERE = 0.7
"""Severity at or above which a challenge forces a refusal rather than a reduction.

Stated rather than tuned: there is no held-out set of adversary judgements to tune against yet, so
it is a deliberately high bar. Below it the decision is cut, not killed — a critic that vetoes on
any doubt is a critic that abstains for you."""

REDUCE_FLOOR = 0.4
"""Below this the challenge is recorded and changes nothing. Above it the position is halved."""


class Outcome(StrEnum):
    """What the adversary did to the decision."""

    NOT_RUN = "not_run"
    """No adversary configured, or the decision opened no exposure. Recorded, not hidden."""

    UNAVAILABLE = "unavailable"
    """The adversary was asked and could not answer. The decision stands and says so."""

    UPHELD = "upheld"
    """Challenged and survived. The strongest counter-case is recorded anyway."""

    REDUCED = "reduced"
    REFUSED = "refused"
    SELF_REFUTED = "self_refuted"
    """An invalidation condition the thesis itself named is already true."""

    @property
    def changed_the_decision(self) -> bool:
        return self in {Outcome.REDUCED, Outcome.REFUSED, Outcome.SELF_REFUTED}


class Critic(Protocol):
    """The one call this module needs, so the transport is swappable in tests."""

    def complete_json(
        self, messages: list[dict[str, Any]], *, required_keys: tuple[str, ...] = ...,
        max_tokens: int = ..., thinking: Any = ...,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class Challenge:
    """The adversary's reading of one decision."""

    outcome: Outcome
    counter_case: str
    weakest_link: str
    severity: float
    already_refuted: bool
    refuted_condition: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": str(self.outcome),
            "counter_case": self.counter_case,
            "weakest_link": self.weakest_link,
            "severity": self.severity,
            "already_refuted": self.already_refuted,
            "refuted_condition": self.refuted_condition,
            "detail": self.detail,
        }

    def render(self) -> str:
        if self.outcome is Outcome.NOT_RUN:
            return f"[adversary] not run — {self.detail}"
        if self.outcome is Outcome.UNAVAILABLE:
            return f"[adversary] unavailable — {self.detail}; the decision stands unchallenged"
        if self.outcome is Outcome.SELF_REFUTED:
            return (
                f"[adversary] the thesis is already refuted by its own falsifier: "
                f"{self.refuted_condition!r} is satisfied by the evidence in front of it"
            )
        head = {
            Outcome.UPHELD: "challenged and upheld",
            Outcome.REDUCED: "challenged; position cut",
            Outcome.REFUSED: "challenged; trade refused",
        }[self.outcome]
        return (
            f"[adversary] {head} (severity {self.severity:.2f}). "
            f"Counter-case: {self.counter_case[:160]}. "
            f"Weakest link: {self.weakest_link[:120]}"
        )


SYSTEM_PROMPT = """You are the standing adversary on a tokenized-equity trading desk. A decision \
has already been made by the portfolio manager. Your only job is to try to falsify it.

You are NOT arguing a side. You are not the bear. If the decision is sound, say so — a critic who \
objects to everything is ignored, and a manufactured objection costs the desk a good trade.

You are given the decision, the thesis behind it, the conditions the thesis itself says would make \
it wrong, and the same evidence the desk saw.

Answer three questions:

1. counter_case: the single strongest argument that this decision is wrong. One or two sentences. \
If there is no real counter-case, say so plainly.
2. weakest_link: which single piece of the evidence the thesis leans on hardest, and what happens \
to the conclusion if that piece is wrong or stale.
3. already_refuted: look at the invalidation conditions the thesis stated. Is ANY of them ALREADY \
TRUE given the evidence shown? This is not about what might happen later — only about what the \
evidence in front of you already says. If yes, name the exact condition in refuted_condition.

severity: 0.0 to 1.0, how much your counter-case should reduce conviction in this decision. \
0.0 means the decision is sound. Use the high end only when you have a concrete reason, not a \
general feeling that markets are uncertain.

Reply with only this JSON object:
{"counter_case": "...", "weakest_link": "...", "already_refuted": false, \
"refuted_condition": "", "severity": 0.0}"""


def _prompt(intent: Intent, evidence: list[Evidence]) -> str:
    lines = [
        f"DECISION: {intent.verdict} {intent.side} {intent.quantity} {intent.symbol}",
        f"STATED CONFIDENCE: {intent.stated_confidence}",
        "",
        "THESIS:",
        intent.thesis,
        "",
        "THE THESIS SAYS IT IS WRONG IF:",
    ]
    lines.extend(f"  - {c}" for c in intent.invalidation)
    lines += ["", "EVIDENCE THE DESK SAW:"]
    # Screened and spotlit here too. The adversary reads the same third-party text as the
    # analysts, so a defence applied on one path and not the other is not a defence.
    lines.append(render_for_prompt(evidence[:MAX_EVIDENCE]))
    return "\n".join(lines)


def _coerce_severity(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value != value:
        return 0.0
    return max(0.0, min(1.0, value))


def challenge(
    intent: Intent,
    evidence: list[Evidence],
    *,
    critic: Critic | None,
    severe: float = SEVERE,
    reduce_floor: float = REDUCE_FLOOR,
) -> tuple[ConstitutionRuling | None, Challenge]:
    """Attack one decision and return the narrowing it earns, if any.

    Returns ``(ruling, challenge)``. ``ruling`` is ``None`` when nothing changes — the caller keeps
    the intent it had. When it is not ``None`` it is produced by ``apply_constraint``, so the
    reduce-only invariants are enforced structurally rather than trusted here.
    """
    if not intent.verdict.opens_exposure or intent.quantity <= 0:
        return None, Challenge(
            Outcome.NOT_RUN, "", "", 0.0, False, "",
            detail="the decision opens no exposure, so there is nothing to falsify",
        )
    if critic is None:
        return None, Challenge(
            Outcome.NOT_RUN, "", "", 0.0, False, "",
            detail="no adversary is configured; the decision is unchallenged",
        )

    try:
        raw = critic.complete_json(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _prompt(intent, evidence)},
            ],
            required_keys=("counter_case", "severity"),
            max_tokens=700,
        )
    except Exception as exc:  # a critic must never take the desk down
        return None, Challenge(
            Outcome.UNAVAILABLE, "", "", 0.0, False, "",
            detail=f"{type(exc).__name__}: {str(exc)[:120]}",
        )

    counter = str(raw.get("counter_case", "")).strip()[:600]
    weakest = str(raw.get("weakest_link", "")).strip()[:400]
    severity = _coerce_severity(raw.get("severity"))
    refuted_flag = bool(raw.get("already_refuted"))
    condition = str(raw.get("refuted_condition", "")).strip()[:300]

    # A self-refutation claim is only honoured when the adversary can name the condition, and the
    # condition must be one the thesis actually stated. Otherwise the critic could veto anything by
    # asserting a falsifier nobody wrote down.
    named = condition and any(
        condition.lower()[:40] in c.lower() or c.lower()[:40] in condition.lower()
        for c in intent.invalidation
    )
    if refuted_flag and named:
        ruling = apply_constraint(
            intent, verdict=ConstitutionVerdict.REJECT,
            binding_constraint="adversary_self_refuted",
            reason=(
                f"the thesis states it is wrong if {condition!r}, and the adversary finds that "
                f"already satisfied by the evidence"
            ),
        )
        return ruling, Challenge(
            Outcome.SELF_REFUTED, counter, weakest, max(severity, severe), True, condition,
        )

    if severity >= severe:
        ruling = apply_constraint(
            intent, verdict=ConstitutionVerdict.REJECT,
            binding_constraint="adversary_severe",
            reason=f"adversary severity {severity:.2f}: {counter[:200]}",
        )
        return ruling, Challenge(Outcome.REFUSED, counter, weakest, severity, False, "")

    if severity >= reduce_floor:
        cut = (intent.quantity / Decimal("2")).quantize(Decimal("0.00000001"))
        if cut <= 0:
            ruling = apply_constraint(
                intent, verdict=ConstitutionVerdict.REJECT,
                binding_constraint="adversary_reduced_to_nothing",
                reason=f"adversary severity {severity:.2f} halves a position already at minimum",
            )
            return ruling, Challenge(Outcome.REFUSED, counter, weakest, severity, False, "")
        ruling = apply_constraint(
            intent, verdict=ConstitutionVerdict.RESIZE,
            binding_constraint="adversary_reduced",
            reason=f"adversary severity {severity:.2f}: {counter[:200]}",
            resized_quantity=cut,
        )
        return ruling, Challenge(Outcome.REDUCED, counter, weakest, severity, False, "")

    return None, Challenge(Outcome.UPHELD, counter, weakest, severity, False, "")


def build_critic(budget_tokens: int = 30_000) -> Critic | None:
    """A model for the adversary, or ``None`` when one cannot be made. Never raises."""
    try:
        from argus.llm.qwen import QwenClient, TokenBudget

        return QwenClient(budget=TokenBudget(limit=budget_tokens))
    except Exception:  # missing key, bad URL, anything
        return None


def explain(record: Challenge) -> str:
    return json.dumps(record.as_dict(), separators=(",", ":"))


__all__ = [
    "MAX_EVIDENCE",
    "REDUCE_FLOOR",
    "SEVERE",
    "SYSTEM_PROMPT",
    "Challenge",
    "Critic",
    "Outcome",
    "build_critic",
    "challenge",
    "explain",
]
