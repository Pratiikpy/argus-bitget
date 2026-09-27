"""The gate autopsy: which Constitution gates fired, passed, or never ran, read from the record.

The analysis behind `eval/autopsy.py`, whose docstring keeps the history of why it exists (two
published explanations of the desk's abstention, both wrong, and the distinction this makes between
a gate that passed and a gate that never ran). It lives here, beside the gate vocabulary in
`risk/gatechain.py`, because product code runs it: `desk/postmortem.py` fires it after a breaker
trip or an escalation, and a module below the top layer may not import the evaluation harness
(`tests/test_eval_boundary.py`). Pure arithmetic over ledger entries and risk records; nothing here
fetches, asks a model, or writes a file.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import fmean, median
from typing import Any

from argus.risk.gatechain import (
    CHAIN,
    FIRED,
    NOTHING_BOUND,
    PASSED,
    RECORD_SCHEMA,
    UNREACHED,
    Gate,
)


class AutopsyError(ValueError):
    """Raised rather than reporting a cause inferred from an empty record."""


def _constraint_of(record: dict[str, Any]) -> str:
    """The constraint that bound, or a refusal naming why this record cannot be read.

    Refuses rather than defaults. ``record.get(key, "none")`` was wrong twice over: a **null**
    ruling has the key present with value ``None``, so the default never applied and ``str(None)``
    produced the string ``"None"`` — which matched no gate and fell through the walk without
    decrementing ``still_live``; and a schema-1 ``"none"`` means something different from a
    schema-2 one.
    """
    if int(record.get("chain_schema", 1)) != RECORD_SCHEMA:
        raise AutopsyError(
            f"risk record seq={record.get('seq')} is schema "
            f"{record.get('chain_schema', 1)}, and this chain reads schema {RECORD_SCHEMA}. "
            f"Schema 1 wrote 'none' for both gate 1 and the all-clear, so reading it here would "
            f"report gates 2-7 UNREACHED on decisions that passed them. Run "
            f"`python -m argus.eval.migrate_risk_records` first."
        )
    value = record.get("binding_constraint")
    if value is None:
        raise AutopsyError(
            f"risk record seq={record.get('seq')} carries a null binding_constraint. "
            f"`paper/runner.py` writes null when the Constitution produced no ruling at all, and a "
            f"decision with no ruling has no place in a reachability funnel — it must not be "
            f"silently counted as having reached every gate."
        )
    return str(value)


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """What happened to one gate across the whole record."""

    gate: Gate
    fired: int
    reached: int
    decisions: int

    @property
    def status(self) -> str:
        if self.fired:
            return FIRED
        return PASSED if self.reached else UNREACHED

    @property
    def note(self) -> str:
        if self.fired:
            return f"bound on {self.fired} of {self.reached} decision(s) that reached it"
        if self.reached:
            return f"evaluated {self.reached} decision(s) and allowed every one"
        return (
            f"never executed: an earlier gate returned first on all {self.decisions} decision(s), "
            f"so this rule has not been exercised and its permissiveness is unmeasured"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "order": self.gate.order, "name": self.gate.name,
            "checks": self.gate.what_it_checks, "status": self.status,
            "reached": self.reached, "fired": self.fired, "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class SessionCoverage:
    """Which sessions the desk has actually been asked to decide in."""

    by_phase: dict[str, int]
    with_price_discovery: int
    hours_to_discovery: tuple[float, ...]

    @property
    def total(self) -> int:
        return sum(self.by_phase.values())

    @property
    def is_single_phase(self) -> bool:
        return len(self.by_phase) == 1

    @property
    def note(self) -> str:
        if not self.total:
            return "no decisions on record"
        phases = ", ".join(f"{k}={v}" for k, v in sorted(self.by_phase.items()))
        head = (
            f"{self.total} decision(s) across {len(self.by_phase)} session phase(s): {phases}. "
            f"{self.with_price_discovery} were taken while the anchor market had price discovery"
        )
        if self.with_price_discovery:
            return head
        return (
            f"{head} — **none**. Median {median(self.hours_to_discovery):.1f}h from the next "
            f"discovery, minimum {min(self.hours_to_discovery):.1f}h. A policy that trades on "
            f"price discovery, evaluated only where there is none, has not been evaluated"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "by_phase": dict(sorted(self.by_phase.items())),
            "with_price_discovery": self.with_price_discovery,
            "median_hours_to_discovery": (
                round(median(self.hours_to_discovery), 2) if self.hours_to_discovery else None
            ),
            "min_hours_to_discovery": (
                round(min(self.hours_to_discovery), 2) if self.hours_to_discovery else None
            ),
            "single_phase_only": self.is_single_phase,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class Autopsy:
    """The decomposition: what stopped a position being taken, and what was never asked."""

    decisions: int
    proposed_exposure: int
    coverage: SessionCoverage
    gates: tuple[GateOutcome, ...]
    confidences: tuple[float, ...]
    gate_population: int = 0
    """Decisions the gate funnel is computed over — the risk records, not the ledger.

    Usually smaller than :attr:`decisions`, because the risk log began after the ledger did. Stated
    rather than smoothed over: a funnel quietly computed on a subset while the headline quotes the
    full decision count is a denominator swap, and it flatters whichever number is smaller.
    """

    hours_to_first_discovery: float | None = None

    @property
    def unreached(self) -> tuple[GateOutcome, ...]:
        return tuple(g for g in self.gates if g.status == UNREACHED)

    @property
    def confidence_would_have_bound(self) -> bool | None:
        """Would the 0.55 floor have refused anything, had it ever run?

        A secondary fact and labelled as one. It is evidence that the floor is not the hidden cause,
        but it is **not** the reason the floor is not the cause — the reason is that it never ran.
        Conflating the two is how "the model is confident" becomes "the gate was generous".
        """
        if not self.confidences:
            return None
        return min(self.confidences) < 0.55

    @property
    def falsifier_has_fired(self) -> bool:
        """Has the desk now abstained through a session that had price discovery?

        This is the whole point of the module and it is computed, never asserted: the sampling
        explanation survives only while every decision on record was taken with the anchor shut.
        """
        return bool(self.coverage.with_price_discovery) and not self.proposed_exposure

    @property
    def falsifier(self) -> str:
        if self.falsifier_has_fired:
            return (
                f"**FIRED.** {self.coverage.with_price_discovery} decision(s) were taken with the "
                f"anchor market open and the desk proposed nothing on any of them. The sampling "
                f"explanation this module published is refuted, and the rewritten conclusion is "
                f"above — not a defence of the old one"
            )
        if self.hours_to_first_discovery is None:
            return (
                "one cycle in a session with price discovery in which the desk still proposes "
                "nothing would move the cause from the sampling to the desk"
            )
        return (
            f"the next session with price discovery is {self.hours_to_first_discovery:.1f}h away, "
            f"and the committed cycle schedule (13:30/15:30/17:30/19:30 UTC) sits inside US "
            f"regular hours. If the desk abstains through a full session of those, the cause is "
            f"the desk and this conclusion is refuted"
        )

    @property
    def verdict(self) -> str:
        if not self.decisions:
            return "UNDEFINED: no decisions on record; nothing to explain"
        # `and not self.unreached` is load-bearing. Some decisions proposing exposure does not make
        # the whole chain exercised: the ones that stopped at gate 1 still never reached the gates
        # below, and the listing under this verdict would show them UNREACHED while the sentence
        # claimed the opposite. Only claim "exercised" when nothing is actually unreached.
        if self.proposed_exposure and not self.unreached:
            return (
                f"{self.proposed_exposure} of {self.decisions} decision(s) proposed exposure, so "
                f"the risk chain has live inputs and its outcomes below are exercised rather than "
                f"unreached"
            )
        unreached = ", ".join(g.gate.name for g in self.unreached)
        confidence_note = ""
        if self.confidences:
            confidence_note = (
                f" Stated confidence runs a median {median(self.confidences):.2f} and never below "
                f"{min(self.confidences):.2f}, so the 0.55 floor would not have "
                f"bound either — but that is a second fact, not the reason: the reason is that it "
                f"was never evaluated."
            )
        scope = (
            f" The funnel is computed over the {self.gate_population} decision(s) carrying a risk "
            f"record, which is fewer than the {self.decisions} on the ledger because the risk log "
            f"began later; the session finding below covers all {self.decisions}."
            if self.gate_population and self.gate_population != self.decisions else ""
        )
        refuted = ""
        if self.falsifier_has_fired:
            refuted = (
                f" **The sampling explanation this module published is REFUTED.** "
                f"{self.coverage.with_price_discovery} decision(s) were taken with the anchor "
                f"market open — the session it claimed the desk had never been offered — and the "
                f"desk proposed nothing on any of them. The cause is the desk's own conviction "
                f"about direction, not when it was asked. Whether that refusal is *correct* is "
                f"still UNDEFINED: eval/shadow.py has 0 scored calls against a floor of 20, and "
                f"the first decisions carrying a directional lean have not settled yet."
            )
        return (
            f"0 of {self.decisions} decision(s) proposed exposure. The first gate "
            f"({CHAIN[0].name}) returned on every one, so {len(self.unreached)} downstream gate(s) "
            f"never executed at all: {unreached}. Their permissiveness is unmeasured, not "
            f"demonstrated.{scope}{confidence_note} {self.coverage.note}{refuted}"
        )

    def render(self) -> str:
        lines = [
            f"ABSTENTION AUTOPSY — {self.decisions} decision(s), "
            f"{self.proposed_exposure} proposing exposure",
            "",
            "  SESSION COVERAGE",
            f"    {self.coverage.note}",
            "",
            "  CONSTITUTION CHAIN, in source order",
        ]
        for outcome in self.gates:
            lines.append(f"    {outcome.status:<10} {outcome.gate.order}. {outcome.gate.name}")
            lines.append(f"               {outcome.note}")
        lines += ["", f"  {self.verdict}", "", f"  FALSIFIER — {self.falsifier}"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "decisions": self.decisions,
            "gate_population": self.gate_population,
            "proposed_exposure": self.proposed_exposure,
            "session_coverage": self.coverage.as_dict(),
            "gates": [g.as_dict() for g in self.gates],
            "unreached_gates": [g.gate.name for g in self.unreached],
            "confidence_median": (
                round(median(self.confidences), 4) if self.confidences else None
            ),
            "confidence_mean": round(fmean(self.confidences), 4) if self.confidences else None,
            "confidence_min": round(min(self.confidences), 4) if self.confidences else None,
            "confidence_would_have_bound": self.confidence_would_have_bound,
            "verdict": self.verdict,
            "falsifier": self.falsifier,
        }


def _phase_has_discovery(phase: str) -> bool:
    """Whether a recorded session phase is one in which the anchor market prices.

    Matched against the recorded string rather than recomputed from the timestamp: the ledger stores
    what the clock said **at decision time**, and recomputing it now would silently re-date every
    historical decision under today's calendar.
    """
    return phase.strip().lower() in {"regular", "rth", "open"}


def examine(
    entries: Sequence[Any],
    risk_records: Sequence[dict[str, Any]],
    *,
    hours_to_first_discovery: float | None = None,
    chain: Sequence[Gate] = CHAIN,
) -> Autopsy:
    """Decompose why no position was taken, from the ledger and the risk records.

    ``risk_records`` carry the constraint that bound on each decision. A gate is counted as
    **reached** on a decision only if every gate before it in ``chain`` declined to bind on that
    same decision — which is what the source does, and the only way to tell an unreached gate from
    a permissive one.
    """
    if not entries:
        raise AutopsyError("no decisions on record; there is nothing to explain")

    phases: dict[str, int] = {}
    hours: list[float] = []
    confidences: list[float] = []
    proposed = 0
    discovery = 0
    for entry in entries:
        phase = str(getattr(entry, "session_phase", "unknown"))
        phases[phase] = phases.get(phase, 0) + 1
        if _phase_has_discovery(phase):
            discovery += 1
        hours.append(float(getattr(entry, "hours_to_discovery", 0.0)))
        confidences.append(float(getattr(entry, "stated_confidence", 0.0)))
        if not getattr(entry, "is_abstention", True):
            proposed += 1

    bound = [_constraint_of(r) for r in risk_records]
    total = len(risk_records) or len(entries)

    # **Refuse a vocabulary this walk cannot interpret.** A value that matches no gate and is not
    # NOTHING_BOUND used to fall through every branch below without ever decrementing `still_live`,
    # silently inflating `reached` — the denominator of every PASSED claim this module makes. The
    # live instance was a null ruling: `paper/runner.py:307` writes JSON null when `ruling is None`,
    # and `str(None)` is the string "None", which matches nothing.
    #
    # This module convicts `vibe-trading` of a `checked_limits` field that is a static literal.
    # Silently dropping constraint families would be the same defect wearing our own name.
    legal = {g.name for g in chain} | {NOTHING_BOUND}
    unknown = sorted({b for b in bound if b not in legal})
    if unknown:
        raise AutopsyError(
            f"risk records carry {len(unknown)} constraint value(s) this chain cannot interpret: "
            f"{', '.join(unknown)}. Known: {', '.join(sorted(legal))}. A funnel that ignores a "
            f"value it does not recognise reports a larger `reached` than it measured."
        )

    outcomes: list[GateOutcome] = []
    # A decision reaches a **terminal** gate only if no terminal gate before it bound. The ceiling
    # gates are different: since 2026-09-15 the Constitution evaluates every one of them and applies
    # the minimum, so they all run on any decision that survives the terminals. Subtracting a
    # ceiling's firings from the ones after it would report gates as UNREACHED that demonstrably
    # executed — the same class of error as the D-1 defect this module was built to catch, pointing
    # the other way.
    still_live = total
    ceiling_live: int | None = None
    for gate in chain:
        if not gate.terminal and ceiling_live is None:
            # Freeze the denominator at the point the terminals stop taking decisions away.
            ceiling_live = still_live
        # Every gate — gate 1 included — is counted by its own name. See `_constraint_of`.
        fired = sum(1 for b in bound if b == gate.name)
        reached = still_live if gate.terminal else (ceiling_live or 0)
        outcomes.append(GateOutcome(
            gate=gate, fired=fired, reached=reached, decisions=total
        ))
        still_live -= fired

    return Autopsy(
        decisions=len(entries),
        gate_population=total,
        proposed_exposure=proposed,
        coverage=SessionCoverage(
            by_phase=phases, with_price_discovery=discovery, hours_to_discovery=tuple(hours),
        ),
        gates=tuple(outcomes),
        confidences=tuple(confidences),
        hours_to_first_discovery=hours_to_first_discovery,
    )


__all__ = [
    "CHAIN",
    "FIRED",
    "NOTHING_BOUND",
    "PASSED",
    "RECORD_SCHEMA",
    "UNREACHED",
    "Autopsy",
    "AutopsyError",
    "Gate",
    "GateOutcome",
    "SessionCoverage",
    "examine",
]
