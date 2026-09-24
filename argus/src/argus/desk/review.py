"""Review and self-evolution — a checklist that has to earn its place, and can lose it.

Track 3's sub-theme asks: *after trading, how does AI help the trader review and iterate their
research framework?* The examples it gives are "auto-generated review reports", "identify bad
decision patterns" and "reusable checklists". The first two are common; the third is where every
system in the corpus stops short, and where the honest difficulty is.

**A checklist is a claim about the future.** "Always check whether the figures in the thesis trace
back to the evidence" asserts that doing so will prevent a class of error. Almost every published
trading checklist is a list of things that sound prudent, assembled by someone sensible, and never
tested against anything. It grows monotonically — items are added after each painful episode and
never removed — until it is long enough that nobody reads it, at which point it has negative value:
it costs attention and prevents nothing.

This module applies the standard the factor lab already applies to factors, to *process rules*
instead. A rule enters as a proposal. It is replayed against the decisions the desk has already
taken, and it earns a status from what that replay shows:

* how often it **fires** — a rule that fires on everything discriminates nothing, and one that
  never fires is dead weight. Both are named and refused.
* how often firing **coincided with a defect that was independently observed** — the precision.
  The defects come from the desk's own checkers (grounding, conflict, contradiction), the risk
  layer's interventions, and, once trades settle, the outcomes. They are not invented for the rule.
* how much of the observed defect it **would have caught** — the recall.

A rule below :data:`MIN_DECISIONS` stays ``PROPOSED``. It is never presented as validated, because
a precision computed over four decisions is a number with no content. A rule that fires on almost
everything, never fires, or is usually wrong when it does is refused by name — :class:`Status`
carries the back half of the lifecycle that every growing checklist is missing.

**What this can and cannot see today is stated in the report itself.** With no settled trades, no
rule about outcomes can be assessed at all — being right about direction is not yet observable —
and :attr:`ReviewReport.unassessable` says so by name rather than letting a reader assume the
checklist covers ground it does not. The defects that *are* observable now are process defects:
a figure in a thesis that traces to nothing, analysts that agreed only because they ran in
sequence, a thesis that contradicts itself.

    python -m argus.desk.review
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from math import comb
from pathlib import Path
from typing import Any

MIN_DECISIONS = 20
"""Fewest reviewed decisions before a rule may be called anything but PROPOSED.

A precision over a handful of decisions is noise. Twenty is not a statistical guarantee — it is the
point below which the number should not be shown at all, and it is stated rather than buried so a
reader can discount accordingly."""

ALWAYS_FIRES = 0.9
"""Fire rate above which a rule is refused for having no discrimination.

A rule that flags nine decisions in ten does not identify a problem; it describes the desk. Such a
rule feels protective and is pure cost, which is exactly how checklists rot."""

NEVER_FIRES = 0.0
"""A rule that never fired on the reviewed set tells us nothing about itself, good or bad."""

MIN_PRECISION = 0.5
"""Kept for callers that report it; no longer a grading threshold.

It used to decide MISLEADING on its own, and an absolute precision floor ignores the base rate: a
rule firing at random on decisions where the targeted defect runs at 72% scores 72% "precision",
and was graded ACTIVE on 1,000 of 1,000 random seeds, while a rule catching every one of a defect
that occurs on 1% of decisions, at 20% precision (18.8 times the base rate), was graded MISLEADING
(rival review, 2026-09-24). Grading is now by lift over the base rate and its significance."""

ALPHA = 0.05
"""Significance a rule's lift needs, after the Benjamini-Hochberg correction across every rule
graded together, to be ACTIVE. Twelve rules tested at 5% each would crown one by chance in half
of all reviews; the correction is what keeps a checklist from being a list of coincidences."""

SUGGESTIVE = 0.2
"""A lift above :data:`MIN_LIFT` with an uncorrected p-value below this is EARNING — worth watching,
not yet evidence. Above it the rule is indistinguishable from chance and is named that way."""

MIN_LIFT = 1.2
"""The smallest lift over the base rate worth a trader's attention at all."""

MIN_FIRINGS = 5
"""Firings below which a rule that looks right is called EARNING rather than ACTIVE.

**This constant exists because `Status.EARNING` was unreachable.** The lifecycle documented it as
*"fires and is usually right, but has not yet cleared the evidence bar"*, and the branch that
returned it was guarded by ``p.caught == 0`` — which is impossible to reach, because precision is
``caught / fired`` and a zero numerator fails the MIN_PRECISION test one line earlier. Every rule
that would have been EARNING was reported MISLEADING instead: *"right 0% of the times it fires"*,
said about a rule that had simply not fired enough times to know.

Found by a test written against the documented lifecycle rather than against the code. Three of one
in a row is a precision of 100% and a sample of three; calling that ACTIVE would be the same error
the module was built to prevent, one level up.
"""


class Status(StrEnum):
    """Where a rule stands. The lifecycle every growing checklist is missing the back half of."""

    PROPOSED = "proposed"
    """Not enough decisions to say anything. The honest default."""

    EARNING = "earning"
    """Fires and is usually right, but has not yet cleared the evidence bar."""

    ACTIVE = "active"
    """Earned its place: it fires selectively and is right when it does."""

    NO_DISCRIMINATION = "no_discrimination"
    """Fires on nearly everything. Refused by name, not quietly dropped."""

    DEAD_WEIGHT = "dead_weight"
    """Never fires. Costs attention, prevents nothing."""

    MISLEADING = "misleading"
    """Fires often and is usually wrong. Worse than absent."""

    RETIRED = "retired"
    """Was active, has stopped earning. Kept in the record so the history is legible."""

    @property
    def keeps_its_place(self) -> bool:
        return self in {Status.ACTIVE, Status.EARNING}


class DefectKind(StrEnum):
    """What went wrong in a decision, by the checker that found it.

    Every kind here is observed by something independent of the rules being tested. A rule that
    both defines a defect and detects it would be grading its own homework.
    """

    GROUNDING = "grounding"
    """A figure in the thesis does not trace to anything the desk was given."""

    CONFLICT = "conflict"
    """Analysts disagreed, or agreed for a reason that does not count as agreement."""

    CONTRADICTION = "contradiction"
    """A claim in the thesis is contradicted by the evidence, or by itself."""

    RISK_INTERVENTION = "risk_intervention"
    """The Constitution had to reduce the decision."""

    OUTCOME = "outcome"
    """The decision was graded against what happened and was wrong. Needs settlement."""


@dataclass(frozen=True, slots=True)
class Defect:
    """One observed problem, attributed to one decision by one checker."""

    seq: int
    symbol: str
    kind: DefectKind
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq, "symbol": self.symbol,
            "kind": str(self.kind), "detail": self.detail[:200],
        }


# Markers copied from the modules that emit them, exactly as `paper/runner.py` does, so a reworded
# check fails a test rather than silently stopping being seen. Guessing these once already cost a
# missed live grounding failure.
_MARKERS: tuple[tuple[str, DefectKind], ...] = (
    ("do not resolve to anything", DefectKind.GROUNDING),
    ("unattributable number", DefectKind.GROUNDING),
    ("unsupported fact", DefectKind.GROUNDING),
    ("is contradicted by", DefectKind.CONTRADICTION),
    ("defect in the reasoning", DefectKind.CONTRADICTION),
    ("contradicts itself", DefectKind.CONTRADICTION),
    ("[conflict:", DefectKind.CONFLICT),
    ("disagreement(s) unresolved", DefectKind.CONFLICT),
)


def defects_from_notes(rows: Sequence[dict[str, Any]]) -> list[Defect]:
    """Read the desk's own check output into attributed defects.

    Only the ``flags`` are read, never the whole note list: a note that reports a check *ran* is not
    a defect, and counting it as one would make every decision look broken.
    """
    out: list[Defect] = []
    for row in rows:
        seq = int(row.get("seq", 0))
        symbol = str(row.get("symbol", ""))
        for flag in row.get("flags", []) or []:
            low = str(flag).lower()
            for marker, kind in _MARKERS:
                if marker in low:
                    out.append(Defect(seq=seq, symbol=symbol, kind=kind, detail=str(flag)))
                    break
    return out


def defects_from_risk(rows: Sequence[dict[str, Any]]) -> list[Defect]:
    """An intervention is a defect in the *proposal*, not in the risk layer.

    The Constitution reducing a position means the decision that reached it was too large for the
    state it was taken in. That is worth reviewing even though the system handled it correctly.
    """
    out: list[Defect] = []
    for row in rows:
        if not row.get("intervened"):
            continue
        out.append(Defect(
            seq=int(row.get("seq", 0)), symbol=str(row.get("symbol", "")),
            kind=DefectKind.RISK_INTERVENTION,
            detail=str(row.get("reason") or row.get("binding_constraint") or "intervened"),
        ))
    return out


def defects_from_outcomes(entries: Sequence[Any]) -> list[Defect]:
    """Settled decisions that went the wrong way.

    Reads ``direction_correct`` only when it is actually present: an unsettled decision is not a
    correct one, and treating ``None`` as success is how a log of abstentions becomes a winning
    record.
    """
    out: list[Defect] = []
    for entry in entries:
        correct = getattr(entry, "direction_correct", None)
        if correct is None or correct:
            continue
        out.append(Defect(
            seq=int(getattr(entry, "seq", 0)), symbol=str(getattr(entry, "symbol", "")),
            kind=DefectKind.OUTCOME,
            detail=f"direction wrong; net {getattr(entry, 'net_pnl', 'unknown')}",
        ))
    return out


@dataclass(frozen=True, slots=True)
class Rule:
    """One checklist item, and the predicate that decides whether it fires on a decision.

    ``targets`` names the defect kinds the rule claims to prevent. Precision is measured against
    those kinds only: a grounding rule is not credited for coinciding with an unrelated conflict,
    which would let any rule look good on a noisy day.
    """

    name: str
    prompt: str
    """What the trader is actually asked to check, in the imperative."""

    rationale: str
    targets: frozenset[DefectKind]
    predicate: Callable[[dict[str, Any]], bool]
    """Given one decision's review record, did the condition this rule warns about hold?"""


@dataclass(frozen=True, slots=True)
class RulePerformance:
    """What a replay showed about one rule."""

    rule: str
    prompt: str
    decisions: int
    fired: int
    caught: int
    """Fired on a decision that independently showed a defect the rule claims to prevent."""

    false_alarms: int
    missed: int
    """Decisions with a targeted defect where the rule stayed silent."""

    status: Status
    note: str = ""
    base_rate: float | None = None
    """Share of reviewed decisions that carried a targeted defect, whether or not the rule fired."""

    p_value: float | None = None
    """One-sided Fisher exact p-value that the rule fires on the defect more than chance would."""

    q_value: float | None = None
    """``p_value`` after the Benjamini-Hochberg correction across the rules reviewed together."""

    @property
    def lift(self) -> float | None:
        """Precision over the base rate: how much likelier a firing is to meet the defect than a
        decision picked at random. 1.0 is a coin; below 1.0 the rule points away from it."""
        if self.precision is None or not self.base_rate:
            return None
        return self.precision / self.base_rate

    @property
    def fire_rate(self) -> float:
        return self.fired / self.decisions if self.decisions else 0.0

    @property
    def precision(self) -> float | None:
        """``None`` when it never fired — unknown, not zero."""
        return self.caught / self.fired if self.fired else None

    @property
    def recall(self) -> float | None:
        total = self.caught + self.missed
        return self.caught / total if total else None

    def render(self) -> str:
        precision = "n/a" if self.precision is None else f"{self.precision:.0%}"
        recall = "n/a" if self.recall is None else f"{self.recall:.0%}"
        lift = "n/a" if self.lift is None else f"{self.lift:.1f}x"
        return (
            f"{self.status.upper():17} {self.rule}: fired {self.fired}/{self.decisions} "
            f"({self.fire_rate:.0%}), precision {precision}, recall {recall}, lift {lift}"
            + (f" — {self.note}" if self.note else "")
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule, "prompt": self.prompt, "status": str(self.status),
            "decisions": self.decisions, "fired": self.fired, "caught": self.caught,
            "false_alarms": self.false_alarms, "missed": self.missed,
            "fire_rate": self.fire_rate, "precision": self.precision, "recall": self.recall,
            "base_rate": self.base_rate, "lift": self.lift, "p_value": self.p_value,
            "q_value": self.q_value, "note": self.note,
        }


def evaluate(
    rule: Rule, records: Sequence[dict[str, Any]], defects: Sequence[Defect],
    *, min_decisions: int = MIN_DECISIONS,
) -> RulePerformance:
    """Replay one rule over the decisions already taken and grade it.

    The grading is deliberately unkind. A rule is credited only when it fires on a decision that an
    independent checker found a targeted defect in; coinciding with some other kind of problem does
    not count, and neither does firing on a clean decision.
    """
    targeted = {d.seq for d in defects if d.kind in rule.targets}
    fired_on: set[int] = set()
    for record in records:
        try:
            hit = bool(rule.predicate(record))
        except Exception:  # a broken predicate must not take the review down
            hit = False
        if hit:
            fired_on.add(int(record.get("seq", 0)))

    reviewed = {int(record.get("seq", 0)) for record in records}
    targeted &= reviewed  # a defect on a decision outside this review is not part of its base rate
    total = len(records)
    fired = len(fired_on)
    caught = len(fired_on & targeted)
    base = len(targeted) / total if total else None
    p_value = (_upper_tail(total, len(targeted), fired, caught)
               if total and fired and targeted else None)
    performance = RulePerformance(
        rule=rule.name, prompt=rule.prompt, decisions=total, fired=fired, caught=caught,
        false_alarms=fired - caught, missed=len(targeted - fired_on),
        status=Status.PROPOSED, base_rate=base, p_value=p_value,
    )
    status, note = _status_for(
        performance, min_decisions=min_decisions, targeted_total=len(targeted)
    )
    return RulePerformance(
        rule=performance.rule, prompt=performance.prompt, decisions=total, fired=fired,
        caught=caught, false_alarms=performance.false_alarms, missed=performance.missed,
        status=status, note=note, base_rate=base, p_value=p_value,
    )


def _hypergeometric(total: int, marked: int, drawn: int, hit: int) -> float:
    return (comb(marked, hit) * comb(total - marked, drawn - hit)) / comb(total, drawn)


def _upper_tail(total: int, marked: int, drawn: int, hit: int) -> float:
    """P(at least ``hit`` marked items in ``drawn`` draws without replacement) — the one-sided
    Fisher exact test that a rule's firings meet the defect more often than chance."""
    top = min(marked, drawn)
    return min(1.0, sum(_hypergeometric(total, marked, drawn, i) for i in range(hit, top + 1)))


def _lower_tail(total: int, marked: int, drawn: int, hit: int) -> float:
    """P(at most ``hit``): the test that a rule avoids the defect more than chance would."""
    bottom = max(0, drawn - (total - marked))
    return min(1.0, sum(_hypergeometric(total, marked, drawn, i) for i in range(bottom, hit + 1)))


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg (1995) adjusted q-values, in the order given."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p_values[i] * m / rank)
        adjusted[i] = min(1.0, running)
    return adjusted


def _status_for(
    p: RulePerformance, *, min_decisions: int, targeted_total: int
) -> tuple[Status, str]:
    """The lifecycle decision, in one place so the thresholds are legible."""
    if p.decisions < min_decisions:
        return Status.PROPOSED, (
            f"only {p.decisions} reviewed decision(s); below {min_decisions} no rate is meaningful"
        )
    if targeted_total == 0:
        # A rule cannot be wrong about something that has never happened. Calling it misleading
        # because its precision is zero would condemn every rule that guards a rare failure — and
        # the rare failures are the ones a checklist exists for.
        return Status.PROPOSED, (
            "no defect of the kind(s) this rule targets has been observed yet, so firing neither "
            "confirms nor refutes it; it cannot be graded on this record"
        )
    if p.fire_rate <= NEVER_FIRES:
        return Status.DEAD_WEIGHT, "never fired on any reviewed decision; it costs attention only"
    if p.fire_rate >= ALWAYS_FIRES:
        return Status.NO_DISCRIMINATION, (
            f"fires on {p.fire_rate:.0%} of decisions, so it describes the desk rather than "
            f"identifying a problem"
        )
    precision = p.precision or 0.0
    base = p.base_rate or 0.0
    lift = p.lift if p.lift is not None else 0.0
    against = (f"right {precision:.0%} of the times it fires against a {base:.0%} base rate "
               f"(lift {lift:.1f}x)")
    if lift < 1.0:
        below = _lower_tail(p.decisions, targeted_total, p.fired, p.caught)
        if below < ALPHA:
            return Status.MISLEADING, (
                f"{against}: it fires on clean decisions more than chance would (p={below:.3f}), "
                f"so it sends the reader away from the problem"
            )
        return Status.NO_DISCRIMINATION, (
            f"{against}: no better than picking decisions at random (p={below:.2f})"
        )
    p_value = p.p_value if p.p_value is not None else 1.0
    if lift < MIN_LIFT or p_value >= SUGGESTIVE:
        return Status.NO_DISCRIMINATION, (
            f"{against}: indistinguishable from chance on this record (p={p_value:.2f})"
        )
    if p.fired < MIN_FIRINGS:
        return Status.EARNING, (
            f"{against}, but on only {p.fired} firing(s); below {MIN_FIRINGS} the rate is not yet "
            f"evidence"
        )
    if p_value >= ALPHA:
        return Status.EARNING, (
            f"{against}; suggestive (p={p_value:.3f}) but not yet evidence"
        )
    return Status.ACTIVE, f"{against}; significant on its own (p={p_value:.2g})"


@dataclass
class ReviewReport:
    """The auto-generated review: what went wrong, how often, and what the checklist has earned."""

    generated_at: datetime
    decisions: int
    defects: list[Defect] = field(default_factory=list)
    performance: list[RulePerformance] = field(default_factory=list)
    unassessable: list[str] = field(default_factory=list)
    """Rules or patterns that cannot be judged yet, named rather than silently omitted."""

    @property
    def by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.defects:
            out[str(d.kind)] = out.get(str(d.kind), 0) + 1
        return out

    @property
    def recurring(self) -> list[tuple[str, int]]:
        """Defect kinds seen more than once — the "bad decision patterns" the sub-theme asks for."""
        return sorted(
            ((k, n) for k, n in self.by_kind.items() if n > 1), key=lambda kv: -kv[1]
        )

    @property
    def checklist(self) -> list[RulePerformance]:
        """The items that have earned a place, worst-status last."""
        order = {Status.ACTIVE: 0, Status.EARNING: 1}
        return sorted(
            (p for p in self.performance if p.status.keeps_its_place),
            key=lambda p: (order.get(p.status, 9), -(p.precision or 0.0)),
        )

    @property
    def rejected(self) -> list[RulePerformance]:
        return [p for p in self.performance if not p.status.keeps_its_place]

    @property
    def failure_cases(self) -> list[dict[str, Any]]:
        """Every rejected rule's own measured failure mode, named rather than left implicit in
        `rejected`'s raw numbers — this is what makes the claim "none of the five standing rules
        earned a place" a documented finding rather than an assertion a reader has to derive."""
        return [
            {"rule": p.rule, "status": str(p.status), "precision": p.precision, "why": p.note}
            for p in self.rejected
        ]

    @property
    def clean_rate(self) -> float:
        flagged = len({d.seq for d in self.defects})
        return 1.0 - (flagged / self.decisions) if self.decisions else 0.0

    def render(self) -> list[str]:
        lines = [
            f"[review] {self.decisions} decision(s) reviewed; "
            f"{len({d.seq for d in self.defects})} carried at least one observed defect "
            f"({self.clean_rate:.0%} clean)",
        ]
        if self.by_kind:
            lines.append("[review] defects by checker: " + ", ".join(
                f"{k} {n}" for k, n in sorted(self.by_kind.items(), key=lambda kv: -kv[1])
            ))
        if self.recurring:
            lines.append("[review] recurring pattern(s): " + ", ".join(
                f"{k} x{n}" for k, n in self.recurring
            ))
        else:
            lines.append("[review] no defect kind recurred; nothing here is yet a pattern")
        lines.append("[review] checklist, by what the replay earned:")
        if self.checklist:
            lines.extend(f"[review]   {p.render()}" for p in self.checklist)
        else:
            lines.append("[review]   nothing has earned a place yet")
        for p in self.rejected:
            lines.append(f"[review]   {p.render()}")
        for item in self.unassessable:
            lines.append(f"[review] cannot assess: {item}")
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "decisions": self.decisions,
            "clean_rate": self.clean_rate,
            "defects_by_kind": self.by_kind,
            "recurring": [{"kind": k, "count": n} for k, n in self.recurring],
            "checklist": [p.as_dict() for p in self.checklist],
            "rejected": [p.as_dict() for p in self.rejected],
            "failure_cases": self.failure_cases,
            "unassessable": list(self.unassessable),
            "defects": [d.as_dict() for d in self.defects],
        }


# --- the standing rules ------------------------------------------------------------------------
#
# Each one is a hypothesis about what prevents a defect, phrased as something a person can do.
# None of them is assumed to work; the replay decides.

def _notes_text(record: dict[str, Any]) -> str:
    return " ".join(str(n) for n in record.get("notes", []) or []).lower()


_PANEL = re.compile(
    r"panel:\s*(?P<analysts>\d+)\s*analysts?,\s*(?P<sources>\d+)\s*distinct sources,\s*"
    r"independence\s*(?P<independence>[\d.]+)",
    re.I,
)


def panel_stats(record: dict[str, Any]) -> dict[str, float]:
    """Analyst count, distinct sources and the independence score for one decision.

    Parsed from the panel note the desk already writes rather than recomputed, so the review reads
    the same number the decision-maker was shown. A record without the line yields an empty dict,
    and every rule below treats that as "cannot tell" rather than as a passing value — a missing
    measurement must not look like a good one.
    """
    match = _PANEL.search(" ".join(str(n) for n in record.get("notes", []) or []))
    if match is None:
        return {}
    return {
        "analysts": float(match.group("analysts")),
        "sources": float(match.group("sources")),
        "independence": float(match.group("independence")),
    }


NARROW_SOURCES = 6.0
"""Distinct-source count below which the evidence base is called narrow.

Chosen from the observed spread on this record (2 to 21 sources) as roughly its lower third, and
stated rather than tuned: the replay below is what decides whether the rule is worth keeping, so a
threshold that turns out to be wrong shows up as NO_DISCRIMINATION or MISLEADING rather than
quietly flattering itself."""

LOW_INDEPENDENCE = 2.5
"""Independence score below which the panel is treated as one view wearing several hats.

Same basis: observed range 1.0 to 5.25 on this record."""


STANDING_RULES: tuple[Rule, ...] = (
    Rule(
        name="narrow-evidence-base",
        prompt=(
            f"If the panel read fewer than {NARROW_SOURCES:.0f} distinct sources, ask where each "
            f"number in the thesis came from before accepting it."
        ),
        rationale=(
            "A thesis can only ground its figures in what the panel actually read. The fewer the "
            "distinct sources, the more places a number can appear from nowhere. This keys on the "
            "source count, which is known *before* the grounding checker runs, so it is a genuine "
            "predictor rather than a restatement of the checker's verdict."
        ),
        targets=frozenset({DefectKind.GROUNDING}),
        predicate=lambda r: (
            (stats := panel_stats(r)) != {} and stats["sources"] < NARROW_SOURCES
        ),
    ),
    Rule(
        name="low-independence-panel",
        prompt=(
            f"If panel independence is under {LOW_INDEPENDENCE}, treat agreement as one opinion, "
            f"not several."
        ),
        rationale=(
            "The desk computes an independence score and discounts it for shared provenance. A low "
            "score means the analysts were reading the same things, so their agreement carries "
            "less information than their number suggests."
        ),
        targets=frozenset({DefectKind.CONFLICT}),
        predicate=lambda r: (
            (stats := panel_stats(r)) != {} and stats["independence"] < LOW_INDEPENDENCE
        ),
    ),
    Rule(
        name="thin-panel",
        prompt="If fewer than three analysts ran, ask what the silent ones would have read.",
        rationale=(
            "A panel of two produces a confident narrow view. The note states how many ran and "
            "what went unexamined."
        ),
        targets=frozenset({DefectKind.GROUNDING, DefectKind.CONFLICT}),
        predicate=lambda r: (
            (stats := panel_stats(r)) != {} and stats["analysts"] < 3
        ),
    ),
    Rule(
        name="sequential-agreement",
        prompt="When analysts agree, check whether they ran independently or one after another.",
        rationale=(
            "The panel note flags sequential execution explicitly. Agreement reached by reading "
            "each other raises confidence without raising information. Its harm is an outcome "
            "question, so this rule cannot be validated until decisions settle."
        ),
        targets=frozenset({DefectKind.OUTCOME}),
        predicate=lambda r: "contagion" in _notes_text(r),
    ),
    Rule(
        name="unhedged-risk-is-stated-not-assumed",
        prompt="If the hedge menu is empty, confirm the residual risk was priced, not ignored.",
        rationale=(
            "Live notes show 100% of risk carried as priced residual with nothing placeable for "
            "43 hours. That is acceptable only if it was a decision."
        ),
        targets=frozenset({DefectKind.RISK_INTERVENTION, DefectKind.OUTCOME}),
        predicate=lambda r: "hedge menu empty" in _notes_text(r),
    ),
)


def review(
    *,
    notes: Sequence[dict[str, Any]],
    risk: Sequence[dict[str, Any]] = (),
    entries: Sequence[Any] = (),
    rules: Sequence[Rule] = STANDING_RULES,
    min_decisions: int = MIN_DECISIONS,
    now: datetime | None = None,
) -> ReviewReport:
    """Grade the desk's own process against its own record."""
    defects = [
        *defects_from_notes(notes),
        *defects_from_risk(risk),
        *defects_from_outcomes(entries),
    ]
    performance = _corrected([evaluate(r, notes, defects, min_decisions=min_decisions)
                              for r in rules])
    report = ReviewReport(
        generated_at=now or datetime.now(UTC),
        decisions=len(notes),
        defects=defects,
        performance=performance,
    )

    settled = sum(
        1 for e in entries if getattr(e, "direction_correct", None) is not None
    )
    if settled == 0:
        report.unassessable.append(
            "any rule whose target is an OUTCOME defect: no decision has settled, so being right "
            "is not yet observable. These rules are reported on their process behaviour only."
        )
    if not any(d.kind is DefectKind.RISK_INTERVENTION for d in defects):
        report.unassessable.append(
            "rules targeting risk interventions: the Constitution has never bound on this record, "
            "so nothing has tested them"
        )
    return report


def _corrected(performance: list[RulePerformance]) -> list[RulePerformance]:
    """Benjamini-Hochberg across every rule that was graded, and ACTIVE only where it survives."""
    graded = [i for i, perf in enumerate(performance) if perf.p_value is not None]
    q_values = benjamini_hochberg([performance[i].p_value or 1.0 for i in graded])
    out = list(performance)
    for i, q in zip(graded, q_values, strict=True):
        perf = out[i]
        status, note = perf.status, perf.note
        if status is Status.ACTIVE and q >= ALPHA:
            status = Status.EARNING
            note = (f"{note}, but not after correcting for the {len(graded)} rules graded "
                    f"together (q={q:.3f})")
        elif status is Status.ACTIVE:
            note = f"{note}; survives the correction across {len(graded)} rules (q={q:.2g})"
        out[i] = replace(perf, status=status, note=note, q_value=q)
    return out


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    from argus.paper.runner import LEDGER_PATH, NOTES_PATH, RISK_PATH

    parser = argparse.ArgumentParser(description="ARGUS review and self-evolution")
    parser.add_argument("--min-decisions", type=int, default=MIN_DECISIONS)
    parser.add_argument("--save", default="", help="write the report to this path")
    args = parser.parse_args(argv)

    from argus.paper.ledger import PaperLedger

    report = review(
        notes=_load(NOTES_PATH),
        risk=_load(RISK_PATH),
        entries=PaperLedger(path=LEDGER_PATH).entries,
        min_decisions=args.min_decisions,
    )
    for line in report.render():
        print(line)
    if args.save:
        Path(args.save).write_text(
            json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8"
        )
        print(f"saved -> {args.save}")
    return 0


__all__ = [
    "ALWAYS_FIRES",
    "LOW_INDEPENDENCE",
    "MIN_DECISIONS",
    "MIN_PRECISION",
    "NARROW_SOURCES",
    "NEVER_FIRES",
    "STANDING_RULES",
    "Defect",
    "DefectKind",
    "ReviewReport",
    "Rule",
    "RulePerformance",
    "Status",
    "defects_from_notes",
    "defects_from_outcomes",
    "defects_from_risk",
    "evaluate",
    "main",
    "panel_stats",
    "review",
]


if __name__ == "__main__":
    raise SystemExit(main())
