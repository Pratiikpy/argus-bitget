"""The back half of a review rule's life: probation, confirmation, and retirement.

`desk/review.py` grades a rule against the whole record every time it runs, and its
:class:`~argus.desk.review.Status` names a ``RETIRED`` state — *"was active, has stopped earning"* —
that no code path ever returns. A rule graded on everything since the desk started keeps its grade
long after it stops working, because months of good history outvote a fortnight of bad; and a rule
that crossed its evidence bar on the one day chance favoured it is never re-examined. Those are the
two failures a *self-evolving* checklist exists to prevent, and a stateless grade cannot see either.

This module gives each candidate rule a lifecycle, run online over the decisions as their outcomes
become observable:

``CANDIDATE`` → ``PROBATION``
    The rule is graded by review's own :func:`~argus.desk.review._status_for` on every decision
    observed since its evidence floor, with the Benjamini-Hochberg correction across every
    candidate graded at the same step (:func:`~argus.desk.review._corrected`). Only ``ACTIVE``
    after correction is admitted, and not if an already-admitted rule for the same defect fires on
    the same decisions (Jaccard ≥ :data:`REDUNDANT_JACCARD`) — a checklist carrying one warning
    twice costs twice the attention for the same catch.
``PROBATION`` → ``ACTIVE`` or ``RETIRED``
    An independent replication. Only decisions observed *after* admission count — never the ones
    that admitted it — and once the rule has fired :data:`PROBATION_FIRINGS` times on them it is
    judged once: lift above 1 with a one-sided Fisher p-value at or below :data:`PROBATION_ALPHA`
    confirms it; anything else retires it. Too few firings extends probation: a rule that has not
    fired has not failed.
``ACTIVE`` → ``RETIRED``
    Monitored firing by firing with a Bernoulli CUSUM (Page 1954; the Bernoulli form of Reynolds &
    Stoumbos 1999) that accumulates the log-likelihood ratio of "this rule is now no better than
    the base rate" against "this rule still carries :data:`REFERENCE_LIFT` times the base rate".
    Crossing :data:`CUSUM_THRESHOLD` retires it. Both hypotheses are stated *relative to the base
    rate around that firing* — the :data:`BASE_WINDOW` observed decisions centred on it, half before
    and half after — so a desk whose defect rate falls is not read as a rule that decayed. A firing
    is scored only once half a window of decisions after it has been observed, which delays
    detection by that many decisions.

    **Corrected 2026-09-26, after measuring the first version.** The monitor first estimated the
    base rate from the :data:`BASE_WINDOW` decisions *before* each firing. After a step fall in the
    base rate (50% to 20%, a rule keeping its 1.8x lift) that estimate lags by up to a hundred
    decisions, every firing in the lag looks worse than base, and the CUSUM retired the rule in 12
    of 12 seeds over 3,000 decisions. With the centred window it is 6 of 12, against 5 of 12 for a
    stable lift-2.2 rule with no shift at all — the shift now adds about one false retirement in
    twelve to the monitor's own false-alarm rate. That rate is itself a measured weakness, not a
    tuned one: at a 20-to-1 threshold a genuinely good rule is falsely retired once in 3,000
    decisions in about half of all seeds, and is re-admitted on fresh evidence within tens of
    decisions (`tests/test_rule_lifecycle.py` pins both numbers). A rule that really stopped working
    is retired in 12 of 12 seeds either way.
``RETIRED`` → ``PROBATION``
    Re-admission is possible, on evidence observed after retirement only.

**What was read, and what was taken.** The two rival lifecycles that exist for trading lessons:

* ``cholhwanjung/trading-agent`` (no licence file — behaviour read, nothing copied):
  `memory/admission.py:34-39,74-151` admits a pattern after n ≥ 5 with a one-sided sign test at
  p ≤ 0.05, then `:154-199` holds it on probation for independent live samples (≥ 5, p ≤ 0.20,
  promotion samples excluded, extended while too few); `memory/retention.py:25-73` retires an
  active lesson whose recent 30-day mean flips sign (≥ 3 samples) or that duplicates another.
  **Taken:** probation as independent replication on data the admission never saw, extension
  rather than failure when the rule has not fired, the looser probation threshold (0.20, "stricter
  than a coin flip but looser than promotion"), de-duplication at admission, and re-admission from
  fresh evidence only. **Rejected:** the sign test's null of 0.5 — on this record the defect base
  rates run from 2% to 61%, so against 0.5 every rule on a common defect "passes" and every rule on
  a rare one "fails" (the coin-flip defect `desk/review.py` was corrected for on 2026-09-24); no
  correction across the patterns tested each day; retirement on a point-estimate mean flip over
  three samples; and calendar windows (7 and 30 days) where firing counts are what carry evidence.
* ``mnemox-ai/tradememory-protocol`` (MIT): `owm/changepoint.py:185-355` runs Bayesian online
  changepoint detection plus a CUSUM on a lesson's win/loss stream, and `owm/drift.py:165-210` a
  plain CUSUM against a target win rate. **Taken:** sequential detection of decay firing by firing,
  rather than re-grading a window. **Rejected:** a target that is a fixed or running win rate
  (`changepoint.py:338-346`) rather than the base rate of the defect; and what happens on
  detection — `mcp_server.py:532` multiplies the lesson's confidence by 0.7 and keeps it, so a
  lesson that stopped working is still recalled. Here it is retired.

Every threshold below was fixed before `eval/review_rivals.py` first ran it, and is reported
alongside its results.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol

from argus.desk.review import (
    MIN_DECISIONS,
    MIN_FIRINGS,
    DefectKind,
    RulePerformance,
    Status,
    _corrected,
    _status_for,
    _upper_tail,
)

PROBATION_FIRINGS = MIN_FIRINGS
"""Firings on post-admission decisions before a probation verdict — review's own evidence floor."""

PROBATION_ALPHA = 0.20
"""One-sided Fisher p-value a replication must reach. Looser than admission, because the family
correction was already paid at admission; stricter than a coin, because a replication that merely
points the right way at p = 0.5 would pass half of all null rules (cholhwanjung's
`memory/admission.py:39` states the same trade-off for the same reason)."""

REFERENCE_LIFT = 1.5
"""The lift the CUSUM's "still working" hypothesis assumes.

Chosen so the monitor's zero-drift point sits near review's :data:`~argus.desk.review.MIN_LIFT`
(1.2): at a 30% base rate the expected increment per firing is zero at a true lift of about 1.24, so
a rule above review's own admission floor drifts toward keeping its place and one below it drifts
toward retirement. Capped at half-way to certainty, ``(1 + b) / 2``, where 1.5 times the base rate
would exceed it."""

CUSUM_THRESHOLD = math.log(20.0)
"""Retire when the accumulated evidence favours "no better than chance" by 20 to 1."""

BASE_WINDOW = 100
"""Observed decisions the monitor's base rate is estimated over, centred on the firing being scored
— local enough to follow a desk whose defect rate is changing, long enough that one bad day does
not move it."""

REDUNDANT_JACCARD = 0.8
"""Firing-set overlap above which a candidate duplicates an admitted rule for the same defect."""


class Stage(StrEnum):
    """Where a rule is in its life. Only ``PROBATION`` and ``ACTIVE`` put a warning in front of a
    trader, and the two are reported apart."""

    CANDIDATE = "candidate"
    PROBATION = "probation"
    ACTIVE = "active"
    RETIRED = "retired"

    @property
    def warns(self) -> bool:
        return self in {Stage.PROBATION, Stage.ACTIVE}


@dataclass(frozen=True, slots=True)
class Candidate:
    """A rule as the lifecycle sees it: what it targets, and which decisions it fires on.

    ``fires`` is computed from decision-time features only (the leakage boundary in
    `desk/review.py`), so knowing it for a decision whose outcome is not yet observed is not a
    leak — it is the warning the rule would have shown when the decision was taken."""

    name: str
    targets: frozenset[DefectKind]
    fires: frozenset[int]


class Evidence(Protocol):
    """What has been observed by the time the lifecycle steps."""

    def eligible(self, targets: frozenset[DefectKind]) -> frozenset[int]:
        """Decisions a rule with these targets can be graded on, whose outcome is observed."""
        ...

    def marked(self, targets: frozenset[DefectKind]) -> frozenset[int]:
        """Of those, the ones that carried one of the targeted defects."""
        ...


@dataclass(frozen=True, slots=True)
class Transition:
    """One change of stage, with the evidence that caused it in words."""

    rule: str
    at: str
    source: Stage
    target: Stage
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "at": self.at, "from": str(self.source),
                "to": str(self.target), "reason": self.reason}


@dataclass
class _Track:
    candidate: Candidate
    stage: Stage = Stage.CANDIDATE
    floor: int = -1
    """Only decisions with a higher sequence count toward admission — reset on retirement."""
    admitted_on: frozenset[int] = frozenset()
    admission_lift: float | None = None
    admission_q: float | None = None
    confirmed_lift: float | None = None
    cusum: float = 0.0
    monitored: set[int] = field(default_factory=set)
    retirements: int = 0


def _lift(caught: int, fired: int, marked: int, total: int) -> float | None:
    if not fired or not marked or not total:
        return None
    return (caught / fired) / (marked / total)


def grade(
    candidate: Candidate, eligible: frozenset[int], marked: frozenset[int], *,
    floor: int = -1, min_decisions: int = MIN_DECISIONS,
) -> RulePerformance:
    """review's own grading of one candidate, on the observed decisions after ``floor``.

    The counts :func:`argus.desk.review.evaluate` would make from the same decisions and defects,
    taken from firing sets rather than by re-running a predicate per decision, then the same
    :func:`~argus.desk.review._status_for`. Uncorrected: the caller applies
    :func:`~argus.desk.review._corrected` across everything graded together, as review does."""
    window = {s for s in eligible if s > floor}
    hits = marked & window
    fired = candidate.fires & window
    caught = len(fired & hits)
    total = len(window)
    p_value = (_upper_tail(total, len(hits), len(fired), caught)
               if total and fired and hits else None)
    perf = RulePerformance(
        rule=candidate.name, prompt="", decisions=total, fired=len(fired),
        caught=caught, false_alarms=len(fired) - caught, missed=len(hits - fired),
        status=Status.PROPOSED, base_rate=len(hits) / total if total else None,
        p_value=p_value,
    )
    status, note = _status_for(perf, min_decisions=min_decisions, targeted_total=len(hits))
    return replace(perf, status=status, note=note)


class Lifecycle:
    """Every candidate's stage, advanced one observation step at a time."""

    def __init__(
        self,
        candidates: Sequence[Candidate],
        *,
        min_decisions: int = MIN_DECISIONS,
        probation_firings: int = PROBATION_FIRINGS,
        probation_alpha: float = PROBATION_ALPHA,
        reference_lift: float = REFERENCE_LIFT,
        cusum_threshold: float = CUSUM_THRESHOLD,
        base_window: int = BASE_WINDOW,
        redundant_jaccard: float = REDUNDANT_JACCARD,
    ) -> None:
        names = [c.name for c in candidates]
        if len(set(names)) != len(names):
            raise ValueError("candidate names must be unique")
        self.tracks = {c.name: _Track(c) for c in candidates}
        self.min_decisions = min_decisions
        self.probation_firings = probation_firings
        self.probation_alpha = probation_alpha
        self.reference_lift = reference_lift
        self.cusum_threshold = cusum_threshold
        self.base_window = base_window
        self.redundant_jaccard = redundant_jaccard
        self.transitions: list[Transition] = []

    # --- queries ----------------------------------------------------------------------------------

    def stage_of(self, name: str) -> Stage:
        return self.tracks[name].stage

    def warning(self, *, include_probation: bool = True) -> frozenset[str]:
        """Rules that put a warning in front of the trader now."""
        keep = {Stage.ACTIVE, Stage.PROBATION} if include_probation else {Stage.ACTIVE}
        return frozenset(n for n, t in self.tracks.items() if t.stage in keep)

    def counts(self) -> dict[str, int]:
        out = {str(s): 0 for s in Stage}
        for t in self.tracks.values():
            out[str(t.stage)] += 1
        return out

    # --- the step ---------------------------------------------------------------------------------

    def step(self, evidence: Evidence, *, at: str) -> list[Transition]:
        """Advance every rule on what ``evidence`` shows. Admitted rules are judged first, so a
        candidate is never blocked as redundant by a rule that is retiring in the same step."""
        cache: dict[frozenset[DefectKind], tuple[frozenset[int], frozenset[int]]] = {}

        def seen(targets: frozenset[DefectKind]) -> tuple[frozenset[int], frozenset[int]]:
            if targets not in cache:
                eligible = evidence.eligible(targets)
                cache[targets] = (eligible, evidence.marked(targets) & eligible)
            return cache[targets]

        before = len(self.transitions)
        for track in self.tracks.values():
            if track.stage is Stage.PROBATION:
                self._judge_probation(track, seen, at)
            elif track.stage is Stage.ACTIVE:
                self._monitor(track, seen, at)
        self._admit(seen, at)
        return self.transitions[before:]

    def _move(self, track: _Track, target: Stage, at: str, reason: str) -> None:
        self.transitions.append(Transition(track.candidate.name, at, track.stage, target, reason))
        track.stage = target

    def _retire(self, track: _Track, eligible: frozenset[int], at: str, reason: str) -> None:
        track.floor = max(eligible) if eligible else track.floor
        track.admitted_on = frozenset()
        track.cusum = 0.0
        track.monitored = set()
        track.retirements += 1
        self._move(track, Stage.RETIRED, at, reason)

    def _grade(
        self, track: _Track, eligible: frozenset[int], marked: frozenset[int]
    ) -> RulePerformance:
        return grade(track.candidate, eligible, marked, floor=track.floor,
                     min_decisions=self.min_decisions)

    def _admit(
        self,
        seen: Callable[[frozenset[DefectKind]], tuple[frozenset[int], frozenset[int]]],
        at: str,
    ) -> None:
        waiting = [t for t in self.tracks.values()
                   if t.stage in {Stage.CANDIDATE, Stage.RETIRED}]
        if not waiting:
            return
        graded = _corrected([self._grade(t, *seen(t.candidate.targets)) for t in waiting])
        ranked = sorted(
            ((t, p) for t, p in zip(waiting, graded, strict=True) if p.status is Status.ACTIVE),
            key=lambda tp: (tp[1].q_value if tp[1].q_value is not None else 1.0,
                            -(tp[1].lift or 0.0), tp[0].candidate.name),
        )
        for track, perf in ranked:
            eligible, _ = seen(track.candidate.targets)
            window = frozenset(s for s in eligible if s > track.floor)
            mine = track.candidate.fires & window
            duplicate = self._duplicate_of(track, mine, window)
            if duplicate is not None:
                continue
            track.admitted_on = window
            track.admission_lift = perf.lift
            track.admission_q = perf.q_value
            again = " (re-admitted on evidence since its retirement)" if track.retirements else ""
            self._move(track, Stage.PROBATION, at,
                       f"admitted{again}: {perf.note}")

    def _duplicate_of(
        self, track: _Track, mine: frozenset[int], window: frozenset[int]
    ) -> str | None:
        for other in self.tracks.values():
            if other is track or not other.stage.warns:
                continue
            if other.candidate.targets != track.candidate.targets:
                continue
            theirs = other.candidate.fires & window
            union = mine | theirs
            if union and len(mine & theirs) / len(union) >= self.redundant_jaccard:
                return other.candidate.name
        return None

    def _judge_probation(
        self,
        track: _Track,
        seen: Callable[[frozenset[DefectKind]], tuple[frozenset[int], frozenset[int]]],
        at: str,
    ) -> None:
        eligible, marked = seen(track.candidate.targets)
        post = {s for s in eligible if s > track.floor} - track.admitted_on
        hits = marked & post
        fired = track.candidate.fires & post
        if len(fired) < self.probation_firings or not hits or len(hits) == len(post):
            return  # not yet judgeable: a rule that has not fired has not failed
        caught = len(fired & hits)
        lift = _lift(caught, len(fired), len(hits), len(post))
        p_value = _upper_tail(len(post), len(hits), len(fired), caught)
        said = (f"{caught} of {len(fired)} post-admission firings met the defect against a "
                f"{len(hits) / len(post):.0%} base rate (lift {lift or 0.0:.2f}, p={p_value:.3f})")
        if lift is not None and lift > 1.0 and p_value <= self.probation_alpha:
            track.confirmed_lift = lift
            track.cusum = 0.0
            track.monitored = set(fired)
            self._move(track, Stage.ACTIVE, at, f"confirmed on decisions it was not admitted "
                                                f"on: {said}")
        else:
            self._retire(track, eligible, at, f"failed replication: {said}")

    def _monitor(
        self,
        track: _Track,
        seen: Callable[[frozenset[DefectKind]], tuple[frozenset[int], frozenset[int]]],
        at: str,
    ) -> None:
        eligible, marked = seen(track.candidate.targets)
        window = sorted(s for s in eligible if s > track.floor)
        fresh = sorted((track.candidate.fires & frozenset(window)) - track.monitored)
        half = self.base_window // 2
        for seq in fresh:
            i = bisect.bisect_left(window, seq)
            if len(window) - (i + 1) < half:
                # Its base rate needs `half` observed decisions after it; so do all later firings.
                # Left unmonitored until a later step observes them.
                break
            track.monitored.add(seq)
            around = window[max(0, i - half):i] + window[i + 1:i + 1 + half]
            if len(around) < self.min_decisions:
                continue
            base = sum(1 for s in around if s in marked) / len(around)
            if not 0.0 < base < 1.0:
                continue
            p_work = min(self.reference_lift * base, (1.0 + base) / 2.0)
            hit = seq in marked
            step = (math.log(base / p_work) if hit
                    else math.log((1.0 - base) / (1.0 - p_work)))
            track.cusum = max(0.0, track.cusum + step)
            if track.cusum >= self.cusum_threshold:
                self._retire(track, frozenset(eligible), at, (
                    f"decayed: the evidence that it is now no better than the base rate reached "
                    f"{track.cusum:.2f} (threshold {self.cusum_threshold:.2f}) at decision {seq}, "
                    f"base rate {base:.0%}, confirmed lift was {track.confirmed_lift or 0.0:.2f}"
                ))
                return

    # --- the record -------------------------------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameters": parameters(self),
            "stages": self.counts(),
            "rules": {
                name: {
                    "stage": str(t.stage), "targets": sorted(str(k) for k in t.candidate.targets),
                    "admission_lift": t.admission_lift, "admission_q": t.admission_q,
                    "confirmed_lift": t.confirmed_lift, "retirements": t.retirements,
                }
                for name, t in sorted(self.tracks.items()) if t.stage is not Stage.CANDIDATE
                or t.retirements
            },
            "transitions": [x.as_dict() for x in self.transitions],
        }


def parameters(lifecycle: Lifecycle | None = None) -> dict[str, float | int]:
    """The fixed thresholds, as run."""
    if lifecycle is None:
        return {
            "min_decisions": MIN_DECISIONS, "probation_firings": PROBATION_FIRINGS,
            "probation_alpha": PROBATION_ALPHA, "reference_lift": REFERENCE_LIFT,
            "cusum_threshold": CUSUM_THRESHOLD, "base_window": BASE_WINDOW,
            "redundant_jaccard": REDUNDANT_JACCARD,
        }
    return {
        "min_decisions": lifecycle.min_decisions,
        "probation_firings": lifecycle.probation_firings,
        "probation_alpha": lifecycle.probation_alpha,
        "reference_lift": lifecycle.reference_lift,
        "cusum_threshold": lifecycle.cusum_threshold,
        "base_window": lifecycle.base_window,
        "redundant_jaccard": lifecycle.redundant_jaccard,
    }


@dataclass(frozen=True)
class SetEvidence:
    """:class:`Evidence` over plain sets: per defect kind, the decisions a rule can be graded on and
    the ones that carried the defect, both already restricted to what has been observed."""

    eligible_by_kind: dict[DefectKind, frozenset[int]]
    marked_by_kind: dict[DefectKind, frozenset[int]]
    everything: frozenset[int]
    """Observed decisions, for a rule targeting more than one kind (review grades those on all)."""

    def eligible(self, targets: frozenset[DefectKind]) -> frozenset[int]:
        if len(targets) == 1:
            (kind,) = tuple(targets)
            return self.eligible_by_kind.get(kind, frozenset())
        return self.everything

    def marked(self, targets: frozenset[DefectKind]) -> frozenset[int]:
        out: set[int] = set()
        for kind in targets:
            out |= self.marked_by_kind.get(kind, frozenset())
        return frozenset(out)


def candidates_from(
    rules: Iterable[tuple[str, frozenset[DefectKind], Callable[[dict[str, Any]], bool]]],
    records: Sequence[dict[str, Any]],
) -> list[Candidate]:
    """Evaluate each rule's predicate once per decision. A predicate that raises does not fire —
    the same tolerance :func:`argus.desk.review.evaluate` applies."""
    out: list[Candidate] = []
    for name, targets, predicate in rules:
        fires: set[int] = set()
        for record in records:
            try:
                if predicate(record):
                    fires.add(int(record.get("seq", 0)))
            except Exception:  # a broken predicate must not take the lifecycle down
                continue
        out.append(Candidate(name=name, targets=targets, fires=frozenset(fires)))
    return out


__all__ = [
    "BASE_WINDOW",
    "CUSUM_THRESHOLD",
    "PROBATION_ALPHA",
    "PROBATION_FIRINGS",
    "REDUNDANT_JACCARD",
    "REFERENCE_LIFT",
    "Candidate",
    "Evidence",
    "Lifecycle",
    "SetEvidence",
    "Stage",
    "Transition",
    "candidates_from",
    "grade",
    "parameters",
]
