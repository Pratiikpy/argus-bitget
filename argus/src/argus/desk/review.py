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

from argus.backtest.metrics import MetricError

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

    LEAN = "lean"
    """The desk's lean — the direction it would take if forced — was contradicted by the move that
    followed, by more than `eval.shadow`'s dead zone. Observable without a single trade settling,
    which is why it exists: it is the only outcome-shaped defect this record carries. Added
    2026-09-25 for `desk/rule_proposer.py`; :func:`review` does not read it, so the standing
    report's figures are unchanged by its existence."""


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


def defects_from_leans(entries: Sequence[Any]) -> list[Defect]:
    """Settled decisions whose lean the market then contradicted.

    Reuses `eval/shadow.py`'s own grading rather than restating it: the move comes from
    :func:`argus.eval.shadow.move_of` (counterfactual move for an abstention, exit against entry for
    a fill) and a move inside :data:`argus.eval.shadow.DEAD_ZONE_BPS` is not a direction, so it is
    neither a defect nor a clean call. A lean of ``none`` is an answer, not a wrong one, and an
    unsettled decision has no outcome — both are skipped. Settlement seals are not decisions.
    """
    from argus.eval.shadow import DEAD_ZONE_BPS, move_of

    out: list[Defect] = []
    for entry in entries:
        if str(getattr(entry, "kind", "decision")) != "decision":
            continue
        lean = str(getattr(entry, "lean", "none"))
        if lean not in {"up", "down"}:
            continue
        moved = move_of(entry)
        if moved is None or abs(moved[0]) <= DEAD_ZONE_BPS:
            continue
        if (moved[0] > 0) == (lean == "up"):
            continue
        out.append(Defect(
            seq=int(getattr(entry, "seq", 0)), symbol=str(getattr(entry, "symbol", "")),
            kind=DefectKind.LEAN,
            detail=f"leaned {lean}; the market moved {moved[0]:+.1f}bps ({moved[1]})",
        ))
    return out


def lean_settled(entries: Sequence[Any]) -> set[int]:
    """Sequences whose lean was actually graded — the only decisions a LEAN rule can be judged on.

    A decision that declined to lean, or met a flat tape, or has not settled, is neither right nor
    wrong about direction; counting it as clean would dilute every LEAN rule's base rate with
    decisions that could never have carried the defect.
    """
    from argus.eval.shadow import DEAD_ZONE_BPS, move_of

    graded: set[int] = set()
    for entry in entries:
        if str(getattr(entry, "kind", "decision")) != "decision":
            continue
        if str(getattr(entry, "lean", "none")) not in {"up", "down"}:
            continue
        moved = move_of(entry)
        if moved is not None and abs(moved[0]) > DEAD_ZONE_BPS:
            graded.add(int(getattr(entry, "seq", 0)))
    return graded


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
    """Benjamini-Hochberg (1995) adjusted q-values, in the order given.

    Refuses a p-value outside [0, 1], NaN included, the way ``scipy.stats.false_discovery_control``
    does (``scipy/stats/_morestats.py:4791-4794``) and with the same positive-form check. Without
    it a NaN sorted to an arbitrary rank and ``min(running, nan)`` kept ``running``, so a NaN
    p-value came back with the q-value of its neighbour — ``[nan, 0.001]`` returned
    ``[0.001, 0.001]`` — measured in ``eval/general_overfitgates_comparison.py``.
    """
    if any(not 0.0 <= p <= 1.0 for p in p_values):
        raise MetricError("every p-value must be a finite number in [0, 1]")
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
        # **Fixed 2026-09-25.** This read ``"contagion" in notes``, and the concurrent panel's own
        # note says the opposite in the same word — "agreement is consensus rather than
        # contagion" — so the rule fired on 599 of 611 recorded decisions (98%), almost all of
        # them panels that ran concurrently and could not have been contagion. Found while
        # building the feature vocabulary for `desk/rule_proposer.py`, which had to parse the
        # panel note exactly. Only the sequential wording ("may be contagion") now fires it.
        predicate=lambda r: "may be contagion" in _notes_text(r),
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


# --- rules a machine can write --------------------------------------------------------------------
#
# The standing rules above are Python lambdas, which is fine for rules a person writes and
# unacceptable for rules a model writes: a model's output is untrusted text, and executing it is
# not an option. So a proposed rule is *data* — a conjunction of at most three comparisons over a
# fixed vocabulary of decision features — and it becomes a :class:`Rule` only by being compiled
# here, where every feature name, operator and value is checked against the vocabulary first.
#
# **The vocabulary is the leakage boundary.** Every feature below is read from a line the desk
# writes *before* any checker runs — the panel, the quarantine, the skills probe, the memory, the
# hedge menu — or from the ledger's decision-time fields. Nothing reads the grounding, conflict,
# claim, debate, adversary or constitution lines, or the flags, because a rule that keys on the
# checker's verdict "predicts" the defect by restating it; that is the same distinction
# `narrow-evidence-base`'s rationale draws by hand. `tests/test_rule_proposer.py` enforces it on
# every recorded decision: stripping every checker line from a record must leave its features
# unchanged. Settlement fields (exit price, counterfactual move, P&L, direction_correct) are
# likewise never read — they are the future.

MAX_CONDITIONS = 3
"""A rule is a conjunction of at most this many comparisons.

A longer conjunction can describe any single decision exactly, which is how a rule written from
one contrast pair memorises its target instead of naming a cause."""

FeatureValue = float | bool | str
"""What a feature reads: a number, a flag or a label. ``None`` means the record cannot say."""


@dataclass(frozen=True, slots=True)
class Feature:
    """One thing about a decision that was known before any checker looked at it."""

    name: str
    kind: str
    """``number``, ``flag`` or ``label`` — decides which comparisons are legal."""

    description: str
    read: Callable[[dict[str, Any]], FeatureValue | None]

    def value(self, record: dict[str, Any]) -> FeatureValue | None:
        try:
            return self.read(record)
        except (TypeError, ValueError, KeyError, AttributeError):
            return None


def _joined_notes(record: dict[str, Any]) -> str:
    return "\n".join(str(n) for n in record.get("notes", []) or [])


def _first(pattern: re.Pattern[str], record: dict[str, Any]) -> re.Match[str] | None:
    return pattern.search(_joined_notes(record))


def _number(pattern: re.Pattern[str], group: str) -> Callable[[dict[str, Any]], float | None]:
    def read(record: dict[str, Any]) -> float | None:
        match = _first(pattern, record)
        return None if match is None else float(match.group(group))
    return read


_STANCE = re.compile(
    r"panel:.*?->\s*(?P<stance>[a-z_]+)\s+at\s+(?P<confidence>[\d.]+)\s+after provenance", re.I
)
_RUN = re.compile(
    r"^\[panel\]\s*(?P<run>\d+)\s+of\s+(?P<of>\d+)\s+analysts run:\s*(?P<which>[^\n]*)$",
    re.I | re.M,
)
_CAUSAL = re.compile(r"^causal chain:\s*(?P<links>\d+)\s+links", re.I | re.M)
_HEDGE = re.compile(r"^hedge menu empty: nothing placeable for (?P<hours>[\d.]+)h", re.I | re.M)
_SCREENED = re.compile(
    r"^\[quarantine\]\s*(?:(?P<withheld>\d+)\s+of\s+)?(?P<screened>\d+)\s+evidence item\(s\)"
    r"\s+(?:screened|withheld)",
    re.I | re.M,
)
_SKILLS = re.compile(
    r"^\[skills\]\s*(?P<answered>\d+)\s+of\s+(?P<calls>\d+)\s+official-Skill calls answered"
    r"[^\n]*?;\s*(?P<reached>\d+)\s+of\s+\d+\s+Skills reached",
    re.I | re.M,
)
_MEMORY = re.compile(
    r"^\[memory\]\s*(?P<prior>\d+)\s+prior decision\(s\)[^\n]*?,\s*(?P<graded>\d+)"
    r"\s+of them graded",
    re.I | re.M,
)
_UNSPENT = re.compile(r"^\[panel\]\s*(?P<bps>[\d.]+)bps of deliberation not spent", re.I | re.M)


def _analysts_run(record: dict[str, Any]) -> float | None:
    match = _first(_RUN, record)
    return None if match is None else float(match.group("run"))


def _ran(analyst: str) -> Callable[[dict[str, Any]], bool | None]:
    def read(record: dict[str, Any]) -> bool | None:
        match = _first(_RUN, record)
        if match is None:
            return None
        return analyst in {w.strip().lower() for w in match.group("which").split(",")}
    return read


def _channel(name: str) -> Callable[[dict[str, Any]], bool | None]:
    def read(record: dict[str, Any]) -> bool | None:
        if "sources" not in record:
            return None
        return name in {str(s) for s in record.get("sources") or []}
    return read


def _withheld(record: dict[str, Any]) -> float | None:
    match = _first(_SCREENED, record)
    if match is None:
        return None
    return float(match.group("withheld") or 0)


def _decision(field_name: str) -> Callable[[dict[str, Any]], FeatureValue | None]:
    """A decision-time ledger field attached by :func:`attach_decisions`; never a settlement one."""
    def read(record: dict[str, Any]) -> FeatureValue | None:
        raw = (record.get("decision") or {}).get(field_name)
        if raw is None:
            return None
        if isinstance(raw, bool | str):
            return raw
        return float(raw)
    return read


def _hour(record: dict[str, Any]) -> float | None:
    raw = record.get("at")
    return None if not raw else float(datetime.fromisoformat(str(raw)).hour)


def _stance(record: dict[str, Any]) -> str | None:
    match = _first(_STANCE, record)
    return None if match is None else match.group("stance").lower()


def _stance_confidence(record: dict[str, Any]) -> float | None:
    match = _first(_STANCE, record)
    return None if match is None else float(match.group("confidence"))


def _stat(name: str) -> Callable[[dict[str, Any]], float | None]:
    def read(record: dict[str, Any]) -> float | None:
        return panel_stats(record).get(name)
    return read


DECISION_FIELDS = (
    "session_phase", "hours_to_discovery", "stated_confidence", "lean", "lean_confidence",
    "verdict",
)
"""The ledger fields a rule may read: all fixed at decision time and hashed with the intent.

Everything else on a ledger entry — ``exit_price``, ``counterfactual_move_bps``, ``net_pnl``,
``direction_correct``, ``settled_at`` — is written at settlement and is the outcome itself."""


DESK_FEATURES: dict[str, Feature] = {f.name: f for f in (
    Feature("analysts", "number", "analysts on the panel (panel line)", _stat("analysts")),
    Feature("sources", "number", "distinct evidence sources the panel read", _stat("sources")),
    Feature("independence", "number", "panel independence score after provenance discount",
            _stat("independence")),
    Feature("panel_stance", "label",
            "the panel's aggregate view: bullish, bearish, neutral or insufficient_evidence",
            _stance),
    Feature("panel_confidence", "number", "the panel's confidence after provenance discount",
            _stance_confidence),
    Feature("analysts_run", "number", "evidence analysts actually run, of three",
            _analysts_run),
    Feature("ran_event", "flag", "the event analyst ran", _ran("event")),
    Feature("ran_sentiment", "flag", "the sentiment analyst ran", _ran("sentiment")),
    Feature("ran_earnings", "flag", "the earnings analyst ran", _ran("earnings")),
    Feature("concurrent_panel", "flag",
            "analysts ran concurrently, so none could read another's answer",
            lambda r: "ran concurrently" in _joined_notes(r).lower()),
    Feature("deliberation_unspent_bps", "number",
            "deliberation budget left unspent, in bps of the round trip",
            _number(_UNSPENT, "bps")),
    Feature("causal_links", "number", "links in the causal chain the thesis states",
            _number(_CAUSAL, "links")),
    Feature("hedge_gap_hours", "number", "hours with nothing placeable on the hedge menu",
            _number(_HEDGE, "hours")),
    Feature("evidence_screened", "number", "evidence items the quarantine screened",
            _number(_SCREENED, "screened")),
    Feature("evidence_withheld", "number", "evidence items the quarantine withheld", _withheld),
    Feature("skill_calls_answered", "number", "official Bitget Skill calls that answered",
            _number(_SKILLS, "answered")),
    Feature("skills_reached", "number", "official Bitget Skills reached, of five",
            _number(_SKILLS, "reached")),
    Feature("memory_prior", "number", "prior decisions on this symbol shown to the PM",
            _number(_MEMORY, "prior")),
    Feature("memory_graded", "number", "of those prior decisions, how many had been graded",
            _number(_MEMORY, "graded")),
    Feature("sentiment_demoted", "flag", "the sentiment analyst ran at its demoted weight",
            lambda r: "sentiment is demoted" in _joined_notes(r).lower()),
    Feature("has_filing", "flag", "a filing channel carried evidence", _channel("filing")),
    Feature("has_sec_edgar", "flag", "SEC EDGAR carried evidence", _channel("sec-edgar")),
    Feature("has_social", "flag", "a social channel carried evidence", _channel("social")),
    Feature("has_news", "flag", "a news channel carried evidence", _channel("news")),
    Feature("has_macro", "flag", "a macro channel carried evidence", _channel("macro")),
    Feature("symbol", "label", "the instrument", lambda r: str(r.get("symbol") or "") or None),
    Feature("hour_utc", "number", "hour of the decision, UTC", _hour),
    Feature("session_phase", "label", "rth, extended or weekend", _decision("session_phase")),
    Feature("hours_to_discovery", "number", "hours until the anchor market next discovers price",
            _decision("hours_to_discovery")),
    Feature("stated_confidence", "number", "the PM's stated confidence in its verdict",
            _decision("stated_confidence")),
    Feature("lean", "label", "the direction the desk would take if forced: up, down or none",
            _decision("lean")),
    Feature("lean_confidence", "number", "the desk's confidence in its lean",
            _decision("lean_confidence")),
    Feature("verdict", "label", "the final verdict: no_trade, buy, sell", _decision("verdict")),
)}
"""The vocabulary a machine-written rule may use on ARGUS's own decisions."""


def attach_decisions(
    notes: Sequence[dict[str, Any]], entries: Sequence[Any]
) -> list[dict[str, Any]]:
    """Each desk-notes row with its ledger decision's decision-time fields under ``decision``.

    Joined on ``seq``, which the two files share (verified on the record: 611 of 611 notes rows
    match a ledger decision with the same symbol). Only :data:`DECISION_FIELDS` are copied, so a
    settlement field cannot reach a rule even by accident.
    """
    by_seq: dict[int, Any] = {}
    for entry in entries:
        if str(getattr(entry, "kind", "decision")) == "decision":
            by_seq[int(getattr(entry, "seq", 0))] = entry
    out: list[dict[str, Any]] = []
    for row in notes:
        entry = by_seq.get(int(row.get("seq", 0)))
        merged = dict(row)
        if entry is not None:
            merged["decision"] = {
                name: getattr(entry, name, None) for name in DECISION_FIELDS
            }
        out.append(merged)
    return out


def features_of(
    record: dict[str, Any], vocabulary: dict[str, Feature] | None = None
) -> dict[str, FeatureValue]:
    """Every feature the record can answer. A feature it cannot answer is absent, never zero."""
    vocab = DESK_FEATURES if vocabulary is None else vocabulary
    out: dict[str, FeatureValue] = {}
    for name, feature in vocab.items():
        value = feature.value(record)
        if value is not None:
            out[name] = value
    return out


class SpecError(ValueError):
    """A proposed rule that cannot be compiled. The message is fed back to whoever wrote it."""


_NUMBER_OPS = frozenset({"<", "<=", ">", ">=", "==", "!="})
_LABEL_OPS = frozenset({"==", "!=", "in"})
_FLAG_OPS = frozenset({"=="})


@dataclass(frozen=True, slots=True)
class Condition:
    """One comparison. A record that cannot answer the feature does not satisfy it."""

    feature: str
    op: str
    value: FeatureValue | tuple[str, ...]

    def holds(self, features: dict[str, FeatureValue]) -> bool:
        if self.feature not in features:
            return False
        got = features[self.feature]
        want = self.value
        if self.op == "in":
            return isinstance(want, tuple) and str(got) in want
        if self.op == "==":
            return got == want
        if self.op == "!=":
            return got != want
        if isinstance(got, bool) or not isinstance(got, float | int):
            return False
        if isinstance(want, bool) or not isinstance(want, float | int):
            return False
        return {
            "<": got < want, "<=": got <= want, ">": got > want, ">=": got >= want,
        }[self.op]

    def render(self) -> str:
        shown = list(self.value) if isinstance(self.value, tuple) else self.value
        return f"{self.feature} {self.op} {shown!r}"

    def as_dict(self) -> dict[str, Any]:
        value = list(self.value) if isinstance(self.value, tuple) else self.value
        return {"feature": self.feature, "op": self.op, "value": value}


def condition_from(raw: Any, vocabulary: dict[str, Feature] | None = None) -> Condition:
    """Validate one untrusted comparison against the vocabulary. Raises :class:`SpecError`."""
    vocab = DESK_FEATURES if vocabulary is None else vocabulary
    if not isinstance(raw, dict):
        raise SpecError(f"a condition must be an object, got {type(raw).__name__}")
    name, op, value = raw.get("feature"), raw.get("op"), raw.get("value")
    if not isinstance(name, str) or name not in vocab:
        raise SpecError(f"unknown feature {name!r}; use one of: {', '.join(sorted(vocab))}")
    kind = vocab[name].kind
    if kind == "number":
        if op not in _NUMBER_OPS:
            raise SpecError(f"{name} is a number; op must be one of {sorted(_NUMBER_OPS)}")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise SpecError(f"{name} is a number; value {value!r} is not")
        return Condition(name, op, float(value))
    if kind == "flag":
        if op not in _FLAG_OPS or not isinstance(value, bool):
            raise SpecError(f"{name} is a flag; write it as {{op: '==', value: true|false}}")
        return Condition(name, op, value)
    if op not in _LABEL_OPS:
        raise SpecError(f"{name} is a label; op must be one of {sorted(_LABEL_OPS)}")
    if op == "in":
        if (not isinstance(value, list) or not value
                or not all(isinstance(v, str) for v in value)):
            raise SpecError(f"{name} 'in' takes a non-empty list of strings")
        return Condition(name, op, tuple(v.lower() if name != "symbol" else v for v in value))
    if not isinstance(value, str):
        raise SpecError(f"{name} is a label; value {value!r} is not a string")
    return Condition(name, op, value if name == "symbol" else value.lower())


_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{2,60}$")


@dataclass(frozen=True, slots=True)
class RuleSpec:
    """A rule as data: what to ask, why, what it claims to prevent, and exactly when it fires."""

    name: str
    prompt: str
    rationale: str
    targets: frozenset[DefectKind]
    conditions: tuple[Condition, ...]

    def fires(self, record: dict[str, Any], vocabulary: dict[str, Feature] | None = None) -> bool:
        return self.matches(features_of(record, vocabulary))

    def matches(self, features: dict[str, FeatureValue]) -> bool:
        """The same test on features already read — a replay over hundreds of decisions reads each
        record once, not once per candidate rule."""
        return all(c.holds(features) for c in self.conditions)

    def compile(self, vocabulary: dict[str, Feature] | None = None) -> Rule:
        """The :class:`Rule` the replay grades. The predicate closes over validated data only."""
        return Rule(
            name=self.name, prompt=self.prompt, rationale=self.rationale, targets=self.targets,
            predicate=lambda record: self.fires(record, vocabulary),
        )

    def render(self) -> str:
        return " AND ".join(c.render() for c in self.conditions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "prompt": self.prompt, "rationale": self.rationale,
            "targets": sorted(str(t) for t in self.targets),
            "conditions": [c.as_dict() for c in self.conditions],
            "fires_when": self.render(),
        }


def spec_from(
    raw: Any, *, allowed_targets: frozenset[DefectKind] | None = None,
    vocabulary: dict[str, Feature] | None = None,
) -> RuleSpec:
    """Validate an untrusted rule object into a :class:`RuleSpec`. Raises :class:`SpecError`
    with a message specific enough to be fed straight back to the model that wrote it."""
    if not isinstance(raw, dict):
        raise SpecError(f"a rule must be an object, got {type(raw).__name__}")
    name = str(raw.get("name") or "").strip().lower()
    if not _SLUG.match(name):
        raise SpecError(f"name {name!r} must be a lowercase slug of 3-61 chars (a-z, 0-9, -)")
    prompt = str(raw.get("prompt") or "").strip()
    rationale = str(raw.get("rationale") or "").strip()
    if len(prompt) < 12 or len(prompt) > 300:
        raise SpecError("prompt must be one imperative sentence of 12-300 characters")
    if len(rationale) < 12 or len(rationale) > 600:
        raise SpecError("rationale must be 12-600 characters")
    raw_targets = raw.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise SpecError("targets must be a non-empty list of defect kinds")
    try:
        targets = frozenset(DefectKind(str(t).lower()) for t in raw_targets)
    except ValueError:
        raise SpecError(
            f"targets {raw_targets!r} must be from {[str(k) for k in DefectKind]}"
        ) from None
    if allowed_targets is not None and not targets <= allowed_targets:
        raise SpecError(f"targets must be within {sorted(str(t) for t in allowed_targets)}")
    raw_conditions = raw.get("conditions")
    if not isinstance(raw_conditions, list) or not 1 <= len(raw_conditions) <= MAX_CONDITIONS:
        raise SpecError(f"conditions must be a list of 1 to {MAX_CONDITIONS} comparisons")
    conditions = tuple(condition_from(c, vocabulary) for c in raw_conditions)
    if len({c.feature for c in conditions}) != len(conditions):
        raise SpecError("each feature may appear in at most one condition")
    return RuleSpec(
        name=name, prompt=prompt, rationale=rationale, targets=targets, conditions=conditions,
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
    graded_p = [(i, perf.p_value) for i, perf in enumerate(performance)
                if perf.p_value is not None]
    graded = [i for i, _ in graded_p]
    # Not `p_value or 1.0`: that turned an exact p of 0.0 — the most significant result a rule can
    # have — into 1.0, because 0.0 is falsy. The None case it was written for is already excluded.
    q_values = benjamini_hochberg([p for _, p in graded_p])
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
    "DECISION_FIELDS",
    "DESK_FEATURES",
    "LOW_INDEPENDENCE",
    "MAX_CONDITIONS",
    "MIN_DECISIONS",
    "MIN_PRECISION",
    "NARROW_SOURCES",
    "NEVER_FIRES",
    "STANDING_RULES",
    "Condition",
    "Defect",
    "DefectKind",
    "Feature",
    "FeatureValue",
    "ReviewReport",
    "Rule",
    "RulePerformance",
    "RuleSpec",
    "SpecError",
    "Status",
    "attach_decisions",
    "benjamini_hochberg",
    "condition_from",
    "defects_from_leans",
    "defects_from_notes",
    "defects_from_outcomes",
    "defects_from_risk",
    "evaluate",
    "features_of",
    "lean_settled",
    "main",
    "panel_stats",
    "review",
    "spec_from",
]


if __name__ == "__main__":
    raise SystemExit(main())
