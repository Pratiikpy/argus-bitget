"""Review rules written from failure contrasts — the generator the review was missing.

`desk/review.py` grades rules and `eval/reviewoos.py` checks whether a grade survives a held-out
window, and both say the same thing about themselves: *every rule they grade was written by hand*.
`reviewoos`'s own docstring names the gap — "the stronger experiment needs a generator". This
module is that generator. It pairs a decision that carried a defect with the most comparable
decision that did not, and asks for **one checkable rule** that separates them. Whether the rule is
kept is not decided here: :mod:`argus.eval.regression_gate` admits it only if it fixes its own
target case and breaks no previously-correct decision on the replayed record.

**What was taken, from where (all three MIT, licences verified upstream).**

* DSPy SIMBA's ``append_a_rule`` (`dspy/teleprompt/simba_utils.py:108-172`, signature
  ``OfferFeedback`` at `:174-211`): a rule is written from a *contrast* — a better and a worse
  trajectory on comparable input — and the instruction to the writer is to "rely on contrasting the
  behavior of the worse trajectory against the better trajectory", "avoid boilerplate", and be
  "concrete and generalizable". SIMBA's buckets are sorted so the sharpest contrasts come first
  (`simba.py:219-249`); here the sharpest contrast is the good decision *nearest* the bad one, so
  whatever differs between them is the smallest candidate cause.
* τ-bench's fault taxonomy fields (`tau_bench/auto_error_identification.py:31-58`): every
  diagnosed failure carries an **author** (user / agent / environment) and a **fault type** with a
  free-text description. Taken as fields only — τ-bench spends two LLM classifications per failure
  to fill them (`:115-175`); here the fault type is the defect kind an independent checker already
  assigned, and the author is either derived from which features differ (deterministic proposer)
  or stated by the model and validated against the enum.

**What was rejected, and why.**

* SIMBA appends the advice as free text to a predictor's instructions (`simba_utils.py:166-170`)
  and keeps it if the mini-batch score improves. Free text cannot be replayed against 611 recorded
  decisions, so it cannot be gated. The rule here is *data* — a conjunction of at most three
  comparisons over the decision-feature vocabulary in `desk/review.py` — compiled into the same
  :class:`~argus.desk.review.Rule` the hand-written checklist uses. A model's output is never
  executed.
* SIMBA's acceptance test is an in-sample score on the batch the rule was written from. That is the
  exact leak `reviewoos` exists to catch; the gate here adds a regression set and the evaluation
  adds a chronological held-out window.
* SIMBA and τ-bench both need a model for every step. :class:`ContrastProposer` writes a rule from
  the same pair with no model at all, so the model proposer always has a free baseline to beat; a
  model that cannot beat arithmetic on the same input has not earned its call.
  :class:`InductionProposer` is the stronger bar: classical rule induction (RIPPER's grow phase),
  which sees every labelled decision in the fit window rather than one pair. **Found 2026-09-25 by a
  planted-cause test:** a pair-only proposer, model or arithmetic, cannot in general recover a
  two-feature cause, because the nearest clean neighbour differs on only one side of it — so the
  specialist that sees the whole window has to be in the comparison, or the model is only being
  compared with something that cannot win.

**Where it runs.** On ARGUS's own decisions (desk notes joined to the ledger's decision-time
fields), and — the gap the plan itself names (`Activity/08_TRADING_OS_PLAN.md:530`: the review
looks at ARGUS's decisions, not the trader's) — on **a trader's own journal**. ARGUS had no
structure for a trader's logged decisions: `lui/memory.py` keeps a trader's stated facts in their
own browser and never sees an outcome, and `desk/personalisation.py` judges proposals against a
profile rather than logging decisions. :func:`journal_records` is that structure: one JSON line per
trade the trader took, turned into records, defects and a feature vocabulary of its own, so the same
proposer and the same gate run on it unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

from argus.desk.review import (
    DESK_FEATURES,
    MAX_CONDITIONS,
    Defect,
    DefectKind,
    Feature,
    FeatureValue,
    RuleSpec,
    SpecError,
    condition_from,
    features_of,
    spec_from,
)

# --- the taxonomy (τ-bench fields) ----------------------------------------------------------------


class FaultAuthor(StrEnum):
    """Who the fault belongs to. τ-bench's three authors, named for a trading desk."""

    DESK = "desk"
    """The desk's own process: the panel, the PM, how confident it said it was."""

    ENVIRONMENT = "environment"
    """What the desk was given: the evidence, the channels, the session, the instrument."""

    TRADER = "trader"
    """A human's own decision, from a trader journal."""


_ENVIRONMENT_FEATURES = frozenset({
    "sources", "evidence_screened", "evidence_withheld", "skill_calls_answered", "skills_reached",
    "memory_prior", "memory_graded", "has_filing", "has_sec_edgar", "has_social", "has_news",
    "has_macro", "symbol", "hour_utc", "session_phase", "hours_to_discovery", "hedge_gap_hours",
})
"""Desk features that describe what the desk was handed rather than what it did with it."""


def author_of(features: Sequence[str], *, journal: bool = False) -> FaultAuthor:
    """The author implied by which features separate a pair.

    Majority of the separating features; a tie goes to the desk, because a desk that was handed
    thin evidence and still decided is the part that can change its behaviour."""
    if journal:
        return FaultAuthor.TRADER
    env = sum(1 for f in features if f in _ENVIRONMENT_FEATURES)
    return FaultAuthor.ENVIRONMENT if env > len(features) - env else FaultAuthor.DESK


DEFECT_MEANING: dict[DefectKind, str] = {
    DefectKind.GROUNDING: "a figure in the thesis traced to nothing the desk was given",
    DefectKind.CONFLICT: "the analysts disagreed and the decision was taken in spite of it",
    DefectKind.CONTRADICTION: "a claim in the thesis was contradicted by the evidence or itself",
    DefectKind.RISK_INTERVENTION: "the Constitution had to reduce the decision",
    DefectKind.OUTCOME: "the trade lost: the market went against the position",
    DefectKind.LEAN: "the direction the desk leaned was contradicted by the move that followed",
}


# --- pairing ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContrastPair:
    """One bad decision and the most comparable decision that did not carry its defect."""

    kind: DefectKind
    bad: dict[str, Any]
    good: dict[str, Any]
    distance: float
    """Mean per-feature distance on the features both can answer; 0 means indistinguishable."""

    differing: tuple[str, ...]
    detail: str
    """What the independent checker said about the bad decision."""

    @property
    def bad_seq(self) -> int:
        return int(self.bad.get("seq", 0))

    @property
    def good_seq(self) -> int:
        return int(self.good.get("seq", 0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind), "bad_seq": self.bad_seq, "good_seq": self.good_seq,
            "symbol": str(self.bad.get("symbol", "")), "distance": round(self.distance, 4),
            "differing": list(self.differing), "detail": self.detail[:200],
        }


def _spreads(
    records: Sequence[dict[str, Any]], vocabulary: dict[str, Feature]
) -> dict[str, float]:
    """Each numeric feature's range over the window, the scale a difference is measured in.

    Read from the fit window's features only — no labels, no defects — so it tells the pairing how
    large a difference is without telling it which differences matter."""
    lo: dict[str, float] = {}
    hi: dict[str, float] = {}
    for record in records:
        for name, value in features_of(record, vocabulary).items():
            if vocabulary[name].kind != "number" or isinstance(value, bool | str):
                continue
            lo[name] = min(lo.get(name, value), value)
            hi[name] = max(hi.get(name, value), value)
    return {name: hi[name] - lo[name] for name in lo}


def _distance(
    a: dict[str, FeatureValue], b: dict[str, FeatureValue], spreads: dict[str, float],
    vocabulary: dict[str, Feature],
) -> tuple[float, tuple[str, ...]]:
    shared = [name for name in vocabulary if name in a and name in b]
    if not shared:
        return float("inf"), ()
    total = 0.0
    differing: list[str] = []
    for name in shared:
        x, y = a[name], b[name]
        if vocabulary[name].kind == "number" and not isinstance(x, bool | str) \
                and not isinstance(y, bool | str):
            spread = spreads.get(name, 0.0)
            gap = abs(x - y) / spread if spread else float(x != y)
        else:
            gap = float(x != y)
        if gap > 0:
            differing.append(name)
        total += min(gap, 1.0)
    return total / len(shared), tuple(differing)


def pair_contrasts(
    records: Sequence[dict[str, Any]],
    defects: Sequence[Defect],
    kind: DefectKind,
    *,
    graded: set[int] | None = None,
    vocabulary: dict[str, Feature] | None = None,
) -> list[ContrastPair]:
    """Every bad decision of ``kind`` paired with its nearest clean one, sharpest contrast first.

    ``graded`` restricts both sides to decisions on which the defect was observable at all: for a
    LEAN defect only a settled, non-flat lean can be wrong, so a decision that declined to lean is
    not a "good" example of anything. Same symbol is preferred as SIMBA prefers the same input; a
    different symbol is used only when the instrument has no clean decision at all. A bad decision
    with no distinguishable clean neighbour is dropped: nothing in the vocabulary separates it, so
    no rule over the vocabulary could.
    """
    vocab = DESK_FEATURES if vocabulary is None else vocabulary
    pool = [r for r in records if graded is None or int(r.get("seq", 0)) in graded]
    marked: dict[int, str] = {}
    for d in defects:
        if d.kind is kind:
            marked.setdefault(d.seq, d.detail)
    spreads = _spreads(pool, vocab)
    feats = {int(r.get("seq", 0)): features_of(r, vocab) for r in pool}
    clean = [r for r in pool if int(r.get("seq", 0)) not in marked]
    pairs: list[ContrastPair] = []
    for bad in pool:
        seq = int(bad.get("seq", 0))
        if seq not in marked:
            continue
        same = [g for g in clean if g.get("symbol") == bad.get("symbol")] or clean
        best: tuple[float, int, dict[str, Any], tuple[str, ...]] | None = None
        for good in same:
            gseq = int(good.get("seq", 0))
            dist, differing = _distance(feats[seq], feats[gseq], spreads, vocab)
            if not differing:
                continue
            key = (dist, abs(gseq - seq))
            if best is None or key < (best[0], best[1]):
                best = (dist, abs(gseq - seq), good, differing)
        if best is None:
            continue
        pairs.append(ContrastPair(
            kind=kind, bad=bad, good=best[2], distance=best[0], differing=best[3],
            detail=marked[seq],
        ))
    pairs.sort(key=lambda p: (p.distance, p.bad_seq))
    return pairs


def interleave(groups: Sequence[Sequence[ContrastPair]]) -> list[ContrastPair]:
    """Round-robin across defect kinds, then across symbols within each, so a capped budget is
    spent on the breadth of failures rather than on forty grounding misses on one instrument."""
    spread: list[list[ContrastPair]] = []
    for group in groups:
        by_symbol: dict[str, list[ContrastPair]] = {}
        for pair in group:
            by_symbol.setdefault(str(pair.bad.get("symbol", "")), []).append(pair)
        lanes = list(by_symbol.values())
        ordered: list[ContrastPair] = []
        while any(lanes):
            for lane in lanes:
                if lane:
                    ordered.append(lane.pop(0))
        spread.append(ordered)
    out: list[ContrastPair] = []
    while any(spread):
        for lane in spread:
            if lane:
                out.append(lane.pop(0))
    return out


# --- proposals ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Proposal:
    """What a proposer returned for one pair: a rule, or a stated reason there is none."""

    proposer: str
    pair: ContrastPair
    spec: RuleSpec | None
    author: FaultAuthor
    description: str
    """τ-bench's free-text fault description: what went wrong, in one or two sentences."""

    refused: str = ""
    calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "proposer": self.proposer, "pair": self.pair.as_dict(),
            "rule": None if self.spec is None else self.spec.as_dict(),
            "taxonomy": {
                "author": str(self.author), "fault_type": str(self.pair.kind),
                "description": self.description[:400],
            },
            "refused": self.refused, "calls": self.calls,
        }


class Proposer(Protocol):
    name: str

    def propose(self, pair: ContrastPair) -> Proposal: ...


def _slug(text: str) -> str:
    out = "".join(ch if ch.isalnum() else "-" for ch in text.lower())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")[:60] or "rule"


@dataclass
class ContrastProposer:
    """The free baseline: a rule from the pair's own feature differences, no model involved.

    For each feature the two decisions answer differently it forms the comparison that puts the bad
    one on its side — a numeric threshold at the midpoint, a label or flag equal to the bad value —
    and ranks them by how large the difference is relative to the feature's spread in the window.
    The top ``width`` become the conjunction. It sees exactly what the model proposer sees (the
    pair, and the unlabelled spread of the window), so the comparison between them is on the same
    input.

    ``width`` is 2, fixed before any run: one comparison from a single pair is almost always a
    property of half the record, and three describe the bad decision closely enough to memorise it.
    """

    records: Sequence[dict[str, Any]]
    vocabulary: dict[str, Feature] = field(default_factory=lambda: DESK_FEATURES)
    width: int = 2
    journal: bool = False
    name: str = "contrast"
    _spread: dict[str, float] | None = field(default=None, init=False, repr=False)

    def propose(self, pair: ContrastPair) -> Proposal:
        if self._spread is None:
            self._spread = _spreads(self.records, self.vocabulary)
        spreads = self._spread
        bad = features_of(pair.bad, self.vocabulary)
        good = features_of(pair.good, self.vocabulary)
        ranked: list[tuple[float, dict[str, Any]]] = []
        for name in pair.differing:
            x, y = bad[name], good[name]
            kind = self.vocabulary[name].kind
            if kind == "number" and not isinstance(x, bool | str) \
                    and not isinstance(y, bool | str):
                spread = spreads.get(name) or 1.0
                ranked.append((abs(x - y) / spread, {
                    "feature": name, "op": "<" if x < y else ">", "value": round((x + y) / 2, 4),
                }))
            else:
                ranked.append((0.5, {"feature": name, "op": "==", "value": x}))
        ranked.sort(key=lambda kv: -kv[0])
        chosen = [c for _, c in ranked[: min(self.width, MAX_CONDITIONS)]]
        author = author_of([c["feature"] for c in chosen], journal=self.journal)
        conditions = tuple(condition_from(c, self.vocabulary) for c in chosen)
        shown = " and ".join(c.render() for c in conditions)
        spec = RuleSpec(
            name=_slug(f"{pair.kind}-{'-'.join(c.feature for c in conditions)}-{pair.bad_seq}"),
            prompt=f"When {shown}, check for a {pair.kind} defect before accepting the decision.",
            rationale=(
                f"Decision {pair.bad_seq} carried a {pair.kind} defect and the nearest clean "
                f"decision, {pair.good_seq}, differed on these features."
            ),
            targets=frozenset({pair.kind}),
            conditions=conditions,
        )
        return Proposal(
            proposer=self.name, pair=pair, spec=spec, author=author,
            description=f"{DEFECT_MEANING[pair.kind]}; separated from its neighbour by {shown}",
        )


DISCRETE_VALUES = 12
"""A numeric feature with at most this many distinct values in the window is also offered as an
equality: analyst counts, Skill counts and hours of the day, not confidences or source counts."""


def _foil_gain(p0: int, n0: int, p1: int, n1: int) -> float:
    """FOIL's information gain for refining a rule covering (p0, n0) to one covering (p1, n1).

    Quinlan (1990), as RIPPER's grow phase uses it (Cohen 1995, *Fast Effective Rule Induction*,
    §2.2). Written from the published formula; no rule-induction code is vendored here."""
    from math import log2

    if p1 == 0:
        return float("-inf")
    return p1 * (log2(p1 / (p1 + n1)) - log2(p0 / (p0 + n0)))


@dataclass
class InductionProposer:
    """The specialist baseline: classical rule induction, which sees the fit window's labels.

    The two proposers above see one pair. A supervised rule learner sees every labelled decision in
    the fit window, which is strictly more information — so it is the bar the model has to be
    compared with, not merely the free baseline. This is RIPPER's *grow* phase (Cohen 1995,
    §2.2; FOIL gain, Quinlan 1990), seeded the way SIMBA seeds a rule: from the pair's defective
    decision. Starting from the empty rule, it adds the one comparison that keeps the target
    covered and maximises FOIL gain over the fit window, until the rule covers no clean decision
    or has :data:`~argus.desk.review.MAX_CONDITIONS` comparisons. Candidate thresholds are midpoints
    between adjacent observed values, as C4.5 and RIPPER both take them.

    Rejected from RIPPER: the *prune* phase and the MDL stopping rule. Both exist to trade training
    precision for generality on a held-back slice of the training data, and here that job belongs
    to the gate (support) and to the held-out window — duplicating it inside the proposer would
    make the gate's own measurement circular.

    ``graded`` restricts the learner to decisions where the defect was observable, as the gate
    does, so a LEAN rule is not grown against decisions that declined to lean.
    """

    records: Sequence[dict[str, Any]]
    defects: Sequence[Defect]
    graded: set[int] | None = None
    vocabulary: dict[str, Feature] = field(default_factory=lambda: DESK_FEATURES)
    journal: bool = False
    name: str = "induction"
    _table: dict[int, dict[str, FeatureValue]] | None = field(default=None, init=False,
                                                            repr=False)

    def _features(self) -> dict[int, dict[str, FeatureValue]]:
        if self._table is None:
            self._table = {
                int(r.get("seq", 0)): features_of(r, self.vocabulary) for r in self.records
                if self.graded is None or int(r.get("seq", 0)) in self.graded
            }
        return self._table

    def _candidates(
        self, target: dict[str, FeatureValue], rows: dict[int, dict[str, FeatureValue]],
        used: set[str],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name, value in target.items():
            if name in used:
                continue
            kind = self.vocabulary[name].kind
            if kind != "number" or isinstance(value, bool | str):
                out.append({"feature": name, "op": "==", "value": value})
                continue
            seen = sorted({v for f in rows.values() if isinstance(v := f.get(name), float | int)
                           and not isinstance(v, bool)})
            for lo, hi in pairwise(seen):
                cut = round((lo + hi) / 2, 4)
                out.append({"feature": name, "op": "<" if value < cut else ">", "value": cut})
            # A count like `analysts` is often causal at one value, not on one side of a cut, and
            # the representation allows each feature once — so equality is a candidate too.
            if len(seen) <= DISCRETE_VALUES:
                out.append({"feature": name, "op": "==", "value": value})
        return out

    def propose(self, pair: ContrastPair) -> Proposal:
        rows = self._features()
        marked = {d.seq for d in self.defects if d.kind is pair.kind and d.seq in rows}
        target = rows.get(pair.bad_seq) or features_of(pair.bad, self.vocabulary)
        covered = dict(rows)
        chosen: list[dict[str, Any]] = []
        while len(chosen) < MAX_CONDITIONS:
            p0 = sum(1 for s in covered if s in marked)
            n0 = len(covered) - p0
            if n0 == 0 or p0 == 0:
                break
            best: tuple[float, dict[str, Any], dict[int, dict[str, FeatureValue]]] | None = None
            for raw in self._candidates(target, covered, {c["feature"] for c in chosen}):
                cond = condition_from(raw, self.vocabulary)
                kept = {s: f for s, f in covered.items() if cond.holds(f)}
                p1 = sum(1 for s in kept if s in marked)
                gain = _foil_gain(p0, n0, p1, len(kept) - p1)
                if best is None or gain > best[0]:
                    best = (gain, raw, kept)
            if best is None or best[0] <= 0:
                break
            chosen.append(best[1])
            covered = best[2]
        if not chosen:
            return Proposal(
                proposer=self.name, pair=pair, spec=None,
                author=author_of([], journal=self.journal), description="",
                refused="no comparison over the vocabulary gains information on this target",
            )
        conditions = tuple(condition_from(c, self.vocabulary) for c in chosen)
        shown = " and ".join(c.render() for c in conditions)
        spec = RuleSpec(
            name=_slug(f"{pair.kind}-induced-{'-'.join(c.feature for c in conditions)}-"
                       f"{pair.bad_seq}"),
            prompt=f"When {shown}, check for a {pair.kind} defect before accepting the decision.",
            rationale=(
                f"Grown from decision {pair.bad_seq} by FOIL gain over the fit window: these "
                f"comparisons best separate {pair.kind} defects from clean decisions there."
            ),
            targets=frozenset({pair.kind}),
            conditions=conditions,
        )
        return Proposal(
            proposer=self.name, pair=pair, spec=spec,
            author=author_of([c.feature for c in conditions], journal=self.journal),
            description=f"{DEFECT_MEANING[pair.kind]}; induced condition {shown}",
        )


class JsonModel(Protocol):
    def complete_json(
        self, messages: list[dict[str, Any]], *,
        required_keys: tuple[str, ...] = ...,
        validate: Callable[[dict[str, Any]], str] | None = ...,
        max_tokens: int = ...,
        attempts: int = ...,
    ) -> dict[str, Any]: ...


_SYSTEM = """You write ONE checklist rule for a trading desk's review, from a contrast between two \
decisions: one that carried a defect, and the most comparable one that did not.

Rely on the contrast. Avoid boilerplate. The rule must be concrete enough to be checked against a \
future decision and general enough to fire on more than this one decision.

The rule is DATA, not prose: a conjunction of 1 to {max_conditions} comparisons over these \
decision features (nothing else exists; each feature at most once):
{vocabulary}

Comparisons: a number takes <, <=, >, >=, ==, != and a numeric value; a flag takes == with true or \
false; a label takes == or != with a string, or "in" with a list of strings. A decision that does \
not report a feature does not satisfy a comparison on it.

How the rule will be judged (so write for it): it is admitted only if it FIRES on the defective \
decision, stays SILENT on the clean one, fires on at least {min_support} earlier decisions, and \
fires on NO earlier decision that was clean of this defect. Then it is graded on later decisions \
it never saw. A rule that describes only this one decision will be rejected; so will a rule that \
also describes clean decisions.

Return JSON only:
{{"discussion": "<what separates the two decisions, and why that plausibly causes the defect>",
  "author": "desk" | "environment",
  "description": "<one sentence: what went wrong>",
  "rule": {{"name": "<lowercase-slug>", "prompt": "<one imperative sentence the trader follows>",
           "rationale": "<why this condition predicts the defect>",
           "conditions": [{{"feature": "...", "op": "...", "value": ...}}]}}}}
If nothing in the features plausibly explains the defect, return {{"discussion": "...", \
"author": "...", "description": "...", "rule": null, "no_rule": "<why>"}}."""


def _vocabulary_text(vocabulary: dict[str, Feature]) -> str:
    return "\n".join(f"- {f.name} ({f.kind}): {f.description}" for f in vocabulary.values())


def _shown(features: dict[str, FeatureValue]) -> dict[str, FeatureValue]:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in features.items()}


@dataclass
class ModelProposer:
    """SIMBA's ``append_a_rule``, with the rule returned as data and validated before it exists.

    One call per pair, retried at most once with the specific validation complaint fed back
    (`QwenClient.complete_json`'s own retry loop). ``calls`` counts every request actually sent,
    from the client's own counter when it has one, so the cap an evaluation states is the cap it
    kept.
    """

    model: JsonModel
    vocabulary: dict[str, Feature] = field(default_factory=lambda: DESK_FEATURES)
    min_support: int = 5
    attempts: int = 2
    journal: bool = False
    name: str = "model"
    calls: int = 0

    def _sent(self) -> int | None:
        raw = getattr(self.model, "calls", None)
        return raw if isinstance(raw, int) else None

    def messages(self, pair: ContrastPair) -> list[dict[str, Any]]:
        system = _SYSTEM.format(
            max_conditions=MAX_CONDITIONS, vocabulary=_vocabulary_text(self.vocabulary),
            min_support=self.min_support,
        )
        if self.journal:
            system = system.replace('"desk" | "environment"', '"trader"')
        user = json.dumps({
            "defect": {"kind": str(pair.kind), "meaning": DEFECT_MEANING[pair.kind],
                       "checker_said": pair.detail[:300]},
            "defective_decision": {"seq": pair.bad_seq, "symbol": pair.bad.get("symbol"),
                                   "features": _shown(features_of(pair.bad, self.vocabulary))},
            "clean_decision": {"seq": pair.good_seq, "symbol": pair.good.get("symbol"),
                               "features": _shown(features_of(pair.good, self.vocabulary))},
            "features_that_differ": list(pair.differing),
        }, default=str)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _validator(self, pair: ContrastPair) -> Callable[[dict[str, Any]], str]:
        def validate(obj: dict[str, Any]) -> str:
            allowed = {"trader"} if self.journal else {"desk", "environment"}
            if str(obj.get("author", "")).lower() not in allowed:
                return f"author must be one of {sorted(allowed)}"
            if obj.get("rule") is None:
                return "" if str(obj.get("no_rule") or "").strip() else (
                    "rule is null; give the reason in no_rule"
                )
            rule = obj.get("rule")
            if not isinstance(rule, dict):
                return "rule must be an object or null"
            try:
                spec_from({**rule, "targets": [str(pair.kind)]},
                          allowed_targets=frozenset({pair.kind}), vocabulary=self.vocabulary)
            except SpecError as exc:
                return str(exc)
            return ""
        return validate

    def propose(self, pair: ContrastPair) -> Proposal:
        before = self._sent()
        author = FaultAuthor.TRADER if self.journal else FaultAuthor.DESK
        try:
            obj = self.model.complete_json(
                self.messages(pair),
                required_keys=("discussion", "author", "description"),
                validate=self._validator(pair),
                max_tokens=1200,
                attempts=self.attempts,
            )
        except Exception as exc:  # a failed call is a recorded refusal, not a crash of the replay
            spent = self._spent(before, self.attempts)
            return Proposal(
                proposer=self.name, pair=pair, spec=None, author=author, description="",
                refused=f"no valid rule obtained: {str(exc)[:200]}", calls=spent,
            )
        spent = self._spent(before, 1)
        author = FaultAuthor(str(obj["author"]).lower())
        description = str(obj.get("description") or "")
        rule = obj.get("rule")
        if not isinstance(rule, dict):
            return Proposal(
                proposer=self.name, pair=pair, spec=None, author=author,
                description=description,
                refused=f"model declined: {str(obj.get('no_rule') or '')[:300]}", calls=spent,
            )
        spec = spec_from({**rule, "targets": [str(pair.kind)]},
                         allowed_targets=frozenset({pair.kind}), vocabulary=self.vocabulary)
        return Proposal(
            proposer=self.name, pair=pair, spec=spec, author=author, description=description,
            calls=spent,
        )

    def _spent(self, before: int | None, fallback: int) -> int:
        after = self._sent()
        spent = (after - before) if (before is not None and after is not None) else fallback
        self.calls += spent
        return spent


# --- a trader's own journal ---------------------------------------------------------------------

JOURNAL_DEAD_ZONE_BPS = 5.0
"""A trade that moved less than this either way is neither a win nor a loss — `eval/shadow.py`'s
dead zone, applied to a human's trades for the same reason."""


class JournalError(ValueError):
    """A journal line that cannot be read. Raised with the line number rather than skipped."""


@dataclass(frozen=True, slots=True)
class JournalEntry:
    """One trade a trader took, as they logged it. Only ``at``, ``symbol`` and ``side`` are
    required; everything else is optional and a rule simply cannot key on what was not logged."""

    at: datetime
    symbol: str
    side: str
    confidence: float | None = None
    size_pct: float | None = None
    """Position size as a percentage of the book."""
    holding_hours: float | None = None
    setup: str | None = None
    """The trader's own name for the setup: "breakout", "earnings", "mean-reversion", …"""
    followed_plan: bool | None = None
    outcome_bps: float | None = None
    """Signed return of the position in bps; ``None`` while it is open."""


def parse_journal(lines: Sequence[str]) -> list[JournalEntry]:
    """Read a JSON-lines trade journal. Untrusted input: parsed, bounded, never executed."""
    out: list[JournalEntry] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise JournalError("not an object")
            side = str(raw["side"]).lower()
            if side not in {"long", "short"}:
                raise JournalError(f"side {side!r} must be long or short")

            def num(key: str, raw: dict[str, Any] = raw) -> float | None:
                value = raw.get(key)
                return None if value is None else float(value)

            plan = raw.get("followed_plan")
            out.append(JournalEntry(
                at=datetime.fromisoformat(str(raw["at"])),
                symbol=str(raw["symbol"])[:20].upper(), side=side,
                confidence=num("confidence"), size_pct=num("size_pct"),
                holding_hours=num("holding_hours"),
                setup=None if raw.get("setup") is None else str(raw["setup"])[:40].lower(),
                followed_plan=None if plan is None else bool(plan),
                outcome_bps=num("outcome_bps"),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise JournalError(f"journal line {number}: {exc}") from None
    return sorted(out, key=lambda e: e.at)


def _journal(field_name: str) -> Callable[[dict[str, Any]], FeatureValue | None]:
    def read(record: dict[str, Any]) -> FeatureValue | None:
        value = (record.get("journal") or {}).get(field_name)
        if value is None or isinstance(value, bool | str):
            return value
        return float(value)
    return read


JOURNAL_FEATURES: dict[str, Feature] = {f.name: f for f in (
    Feature("side", "label", "long or short", _journal("side")),
    Feature("confidence", "number", "the trader's stated confidence, 0-1", _journal("confidence")),
    Feature("size_pct", "number", "position size as a percentage of the book",
            _journal("size_pct")),
    Feature("holding_hours", "number", "planned holding period in hours",
            _journal("holding_hours")),
    Feature("setup", "label", "the trader's own name for the setup", _journal("setup")),
    Feature("followed_plan", "flag", "the trader says the trade followed their plan",
            _journal("followed_plan")),
    Feature("symbol", "label", "the instrument", lambda r: str(r.get("symbol") or "") or None),
    Feature("hour_utc", "number", "hour the trade was opened, UTC", _journal("hour_utc")),
    Feature("weekday", "number", "day of the week it was opened, Monday = 0",
            _journal("weekday")),
    Feature("trades_earlier_same_day", "number",
            "trades the trader had already opened that UTC day", _journal("trades_today")),
    Feature("after_a_loss", "flag",
            "the trader's most recent trade to have closed before this one opened was a loss",
            _journal("after_a_loss")),
)}
"""The vocabulary for a trader's journal. The two derived features are computed from earlier
trades only — a trade closed after this one opened is not something the trader could have known."""


def journal_records(
    entries: Sequence[JournalEntry],
) -> tuple[list[dict[str, Any]], list[Defect], set[int]]:
    """Records, OUTCOME defects and the graded set for a trader journal.

    A trade is a defect when it lost by more than the dead zone, clean when it gained by more, and
    ungraded otherwise (open, or flat) — the same three-way split `defects_from_leans` makes for
    the desk, so the gate treats a human's record exactly as it treats ARGUS's own.
    """
    records: list[dict[str, Any]] = []
    defects: list[Defect] = []
    graded: set[int] = set()
    ordered = sorted(entries, key=lambda e: e.at)
    for seq, entry in enumerate(ordered, start=1):
        earlier = ordered[: seq - 1]
        today = sum(1 for e in earlier if e.at.date() == entry.at.date())
        closed: list[tuple[datetime, float]] = []
        for e in earlier:
            if e.outcome_bps is None or e.holding_hours is None:
                continue
            closed_at = e.at.timestamp() + e.holding_hours * 3600.0
            if closed_at <= entry.at.timestamp():
                closed.append((datetime.fromtimestamp(closed_at, tz=e.at.tzinfo), e.outcome_bps))
        after_loss: bool | None = (
            max(closed, key=lambda c: c[0])[1] < -JOURNAL_DEAD_ZONE_BPS if closed else None
        )
        records.append({
            "seq": seq, "symbol": entry.symbol, "at": entry.at.isoformat(),
            "journal": {
                "side": entry.side, "confidence": entry.confidence, "size_pct": entry.size_pct,
                "holding_hours": entry.holding_hours, "setup": entry.setup,
                "followed_plan": entry.followed_plan, "hour_utc": entry.at.hour,
                "weekday": entry.at.weekday(), "trades_today": today,
                "after_a_loss": after_loss,
            },
        })
        if entry.outcome_bps is None or abs(entry.outcome_bps) <= JOURNAL_DEAD_ZONE_BPS:
            continue
        graded.add(seq)
        if entry.outcome_bps < 0:
            defects.append(Defect(
                seq=seq, symbol=entry.symbol, kind=DefectKind.OUTCOME,
                detail=f"{entry.side} {entry.symbol} lost {entry.outcome_bps:+.1f}bps",
            ))
    return records, defects, graded


def load_journal(path: Path) -> list[JournalEntry]:
    return parse_journal(path.read_text(encoding="utf-8").splitlines())


__all__ = [
    "DEFECT_MEANING",
    "JOURNAL_DEAD_ZONE_BPS",
    "JOURNAL_FEATURES",
    "ContrastPair",
    "ContrastProposer",
    "FaultAuthor",
    "InductionProposer",
    "JournalEntry",
    "JournalError",
    "ModelProposer",
    "Proposal",
    "Proposer",
    "author_of",
    "interleave",
    "journal_records",
    "load_journal",
    "pair_contrasts",
    "parse_journal",
]
