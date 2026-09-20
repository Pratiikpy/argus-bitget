"""Why has this desk never traded? Answered from the record, with the unreached gates named.

Track 2 is 50% quantitative and ARGUS has no Sharpe, no max drawdown and no win rate, because zero
positions have settled. Every report that touches this says so. None of them said **why**, and the
two explanations that got published were both wrong:

* *the fee* — refuted by the hurdle frontier, which measured a median absolute move of 137bps
  against an 18.8bps total hurdle. Moves clear the cost seven times over.
* *the confidence gate* — refuted here. `agents/desk.py:651` returns on ``quantity <= 0`` before
  reaching the 0.55 floor, so the floor **has never executed**. All 88 risk records read
  ``binding_constraint: none``. A constraint that never runs cannot be the binding one, and naming
  it as such pointed an iteration at loosening a gate that is not in the path.

**The distinction this module exists to make is between a gate that passed and a gate that never
ran.** They look identical in every counter that only tallies rejections: both contribute zero. But
"the confidence floor allowed 170 decisions through" and "the confidence floor was never evaluated"
support opposite conclusions, and only the second is true here. A funnel that reports survival rates
without reporting *reachability* will call an unreachable gate a permissive one every time.

**~~The measured cause is in the sampling, not the desk.~~ REFUTED 2026-09-14.** This module
published a third explanation, and it has now failed its own test:

> *"All 170 recorded decisions were taken in a ``weekend`` session, between 18 and 52 hours from
> price discovery… The desk has never been offered a session in which its own stated policy would
> trade. 'It refuses to trade' has therefore never been an observation about the desk — it is an
> observation about when we asked it."*

On 2026-09-14 the 13:30 UTC cycle ran **inside US regular hours with the anchor market open**. The
desk was offered exactly the session this module said it had never been offered, and it proposed
**nothing on all twelve instruments**. The falsifier named below fired on the first attempt.
:attr:`Autopsy.falsifier_has_fired` computes it from the record rather than leaving it to a reader.

**So three published explanations have now been wrong: the fee, the confidence gate, and the
sampling.** What the record actually shows, after the RTH session:

* **It is not the confidence floor.** All twelve RTH decisions state confidence between 0.58 and
  0.88 — every one *above* the 0.55 floor. The floor still never executes, because gate 1 returns
  first on a zero quantity.
* **It is not the fee.** `eval/hurdle.py` is unchanged and still right: a median absolute move of
  137bps against an 18.8bps hurdle, 95% of instants clearing it, and trading beating abstaining at
  **56.0% directional accuracy**.
* **It is the desk's own conviction about direction.** The RTH theses say so in their own words —
  *"the edge is not clearly above the 16.27bps total hurdle"*. Read against the frontier that is not
  a claim about cost, which is cleared seven times over; it is the desk declining to bet that it can
  call direction better than 56%.

**And whether that refusal is correct is still UNDEFINED, which is the honest end of this.**
`eval/shadow.py` has **0 scored calls** against a floor of 20: all 69 settled decisions declined to
lean. The twelve RTH decisions are the first that carry a directional lean (up/down, 0.38-0.62
confidence) and they settle 24 hours later. Until they score, "the desk correctly refuses to bet on
an edge it does not have" and "the desk is too timid" are both consistent with the record, and this
module will not choose between them.

**Prior art, read before this was written** (`research/architecture/gate-attribution.md`). Four
systems in the corpus record why an order was refused, and **none distinguishes a gate that allowed
from a gate that never ran**. `nautilus_trader`'s `OrderDeniedReason`
(`crates/model/src/events/order/denied_reason.rs:69`) is the best taxonomy available — 37 typed
variants carrying their own comparison operands — and records only the reason that fired.
`freqtrade` returns the first protection that locks and nothing about the rest
(`plugins/protections/iprotection.py:17-22`). `factorminer` carries one reason string with good
context (`evaluation/admission.py:21-85`).

The near miss is `vibe-trading`, which puts a ``checked_limits`` list in every decision envelope
(`agent/src/live/sdk_order_gate.py:398`) — and builds it as a **static literal** at `:283`, `:469`
and `:544` that is never mutated. An order denied by its first gate is recorded as having had all
twelve limits checked. The field reads like reachability evidence and is a constant, which is worse
than `freqtrade` recording nothing. Ours is derived by walking :data:`CHAIN` in source order and
subtracting what each gate took — the same short-circuit the policy performs — so reachability is
computed rather than declared.

**A second near miss, found 2026-09-15 by searching all 667 cloned repositories rather than the
four this section was written from** (`eval/corpusclaim.py`, `data/corpus_claims.json`).
`AlgoVaultLabs~crypto-quant-signal-mcp/src/lib/scorer-input-codes.ts:82` declares
``NOT_EVALUATED: 0`` and comments it *"Distinct from NEUTRAL"* — **the same distinction this module
makes, between a rule that ran and allowed and one that never ran at all.** It is applied to a
signal adjustment rather than to an order-gate chain, so the claim above stands as scoped; but the
idea is not ours alone, and saying so is cheaper than being told. The narrower claim — that no
system in the corpus reports it *for a gate chain* — is what the search actually supports.

**What would falsify the rewritten conclusion.** The sampling falsifier has already fired, so the
open question has moved: it is no longer *why* the desk abstains but *whether it should*. Two
observations would settle it, and both arrive on their own:

* **The twelve RTH leans settle.** If they score materially above the 56.0% break-even, the desk had
  a directional edge and declined to use it — the abstention was timidity and the conviction was
  miscalibrated. If they score at or below it, the refusal was correct.
* **A cycle in which the desk proposes exposure.** Gates 2-7 have still never executed, so their
  permissiveness remains unmeasured rather than demonstrated.

The original falsifier worked exactly as intended: it was written down before the evidence existed,
it was checked automatically every cycle, and when it fired the conclusion was rewritten rather than
defended. That is the only part of this module's history worth keeping.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median
from typing import Any

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "abstention_autopsy.json"

FIRED = "FIRED"
"""The gate executed and changed or refused the intent."""

PASSED = "PASSED"
"""The gate executed and let the intent through. A real permission."""

UNREACHED = "UNREACHED"
"""The gate never executed, because an earlier one returned first.

Not the same as PASSED and never to be counted with it. This is the state that makes an unexercised
risk layer look like a permissive one.
"""


class AutopsyError(ValueError):
    """Raised rather than reporting a cause inferred from an empty record."""


@dataclass(frozen=True, slots=True)
class Gate:
    """One rule in the Constitution's ordered chain, as it is written in the source.

    ``order`` is the position in `agents.desk.ConstitutionPolicy.rule`. The chain is a sequence of
    early returns, so a gate is reachable on a given decision only if every gate before it declined
    to return. Recording the order here rather than inferring it keeps the funnel honest when the
    policy is reordered: a rule moved above the exposure check becomes reachable, and this file must
    be updated with it or the test below fails.
    """

    order: int
    name: str
    what_it_checks: str
    terminal: bool = True
    """Does this gate END the evaluation, or only contribute a ceiling?

    **The distinction was added on 2026-09-15 and it changes what UNREACHED means.** The chain used
    to be seven early returns, so any gate could leave every later one unexecuted. It was
    demonstrated that this let an order resized by `unhedgeable_gap` bypass `risk_budget` entirely —
    approved at 20,000 while the circuit breaker sized the book to zero.

    The Constitution now evaluates **every** ceiling gate and applies the minimum, so:

    * a **terminal** gate (`no_exposure`, `min_confidence`, `oracle_stale`) still ends the
      evaluation, and gates after it are genuinely UNREACHED;
    * a **ceiling** gate (`unhedgeable_gap`, `risk_budget`, `session_volatility`, `max_position`)
      always runs once the terminals pass. It can be UNREACHED only because a terminal fired —
      never because another ceiling bound first.

    Reporting a ceiling as UNREACHED merely because a different ceiling was tighter would understate
    how much of the layer is exercised, which is the opposite of this module's purpose.
    """


CHAIN: tuple[Gate, ...] = (
    Gate(1, "no_exposure", "verdict carries no quantity, or quantity <= 0", terminal=True),
    Gate(2, "min_confidence", "stated confidence below the 0.55 floor", terminal=True),
    Gate(3, "oracle_stale", "NAV stale while the anchor is shut", terminal=True),
    Gate(4, "unhedgeable_gap", "no hedge placeable and size above the unhedged cap",
         terminal=False),
    Gate(5, "gross_exposure", "whole-book gross notional above the configured cap",
         terminal=False),
    Gate(6, "signed_exposure", "whole-book net directional notional above the configured cap",
         terminal=False),
    Gate(7, "hedge_integrity", "REDUCE would leave a linked hedge position's partner exposed",
         terminal=False),
    Gate(8, "margin_usage", "venue-reported margin ratio at or above the policy cap",
         terminal=False),
    Gate(9, "factor_exposure", "measured book exposure to a named factor above its configured "
         "cap", terminal=False),
    Gate(10, "scenario_loss", "worst modelled benchmark shock already past the loss floor",
         terminal=False),
    Gate(11, "liquidation_cost", "worst position's forced-exit slippage proxy at or above the "
         "policy cap", terminal=False),
    Gate(12, "per_symbol_underperformance", "one symbol's windowed realized PnL below its "
         "configured floor — narrows that symbol only, not the whole order", terminal=False),
    Gate(13, "risk_budget", "circuit breaker's drawdown ladder sizes the book below the request",
         terminal=False),
    Gate(14, "session_volatility", "measured path volatility above the regular-hours baseline",
         terminal=False),
    Gate(15, "max_position", "size above the position cap", terminal=False),
)
"""The Constitution's chain, in source order (`agents/desk.py:651-700`).

Written out because the risk records carry only the constraint that *bound*, and a reader cannot
tell from ``binding_constraint: none`` which of the other four were even consulted.
"""


NOTHING_BOUND = "none"
"""Every gate ran and allowed. **Not** the same as gate 1 returning — see :data:`RECORD_SCHEMA`."""

RECORD_SCHEMA = 2
"""The risk-record vocabulary this module can read.

**Schema 1 wrote ``"none"`` for two different outcomes** — gate 1 returning (no exposure proposed)
*and* the terminal all-clear (passed all seven) — so a reader could not tell them apart, and this
module guessed wrong in the dangerous direction: it counted every ``"none"`` as gate 1 firing,
which makes gates 2-7 read UNREACHED on a decision that in fact reached and passed all of them.

Schema 2 gives gate 1 its own name (`agents/desk.py`), leaving ``"none"`` to mean only *nothing
bound*. Records are refused rather than reinterpreted, because a schema-1 file read under schema-2
rules would silently report the **opposite** of the truth on this project's single differentiator.
Run ``python -m argus.eval.migrate_risk_records`` to convert.
"""


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


def main() -> int:  # pragma: no cover - CLI
    from argus.paper.ledger import PaperLedger
    from argus.paper.runner import LEDGER_PATH, RISK_PATH
    from argus.truth.clocks import DualClock

    ledger = PaperLedger(path=LEDGER_PATH)
    records: list[dict[str, Any]] = []
    if RISK_PATH.exists():
        records = [
            json.loads(x) for x in RISK_PATH.read_text(encoding="utf-8").splitlines() if x.strip()
        ]
    state = DualClock().state(datetime.now(UTC), nav_age_seconds=60)
    report = examine(
        ledger.entries, records,
        hours_to_first_discovery=(
            0.0 if state.phase.has_price_discovery else state.hours_to_next_discovery
        ),
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    print(report.render())
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "CHAIN",
    "FIRED",
    "PASSED",
    "UNREACHED",
    "Autopsy",
    "AutopsyError",
    "Gate",
    "GateOutcome",
    "SessionCoverage",
    "examine",
]
