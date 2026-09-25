"""A proposed review rule is admitted only if it fixes its case and breaks nothing that worked.

`desk/rule_proposer.py` writes one rule from one contrast pair. A rule written from one example is
a hypothesis about one example, and the way that goes wrong is well known from the corpus: DSPy's
SIMBA keeps a rule when the mini-batch it was written from scores better, which is an in-sample
test on the data that produced the rule. This module is the gate that stops that, and the replay
that measures whether the gate itself is worth having.

**The gate is SWE-bench's, applied to checklist rules** (`swebench/harness/grading.py:288-326`,
MIT). A patch resolves an instance only if every FAIL_TO_PASS test now passes (it fixed the bug)
*and* every PASS_TO_PASS test still passes (it broke nothing): ``FULL`` needs both at 1.0,
anything else is ``NO``. Here:

* **FAIL_TO_PASS** is the rule's own target: it must fire on the defective decision it was written
  from, and stay silent on the clean decision it was contrasted with. A rule that cannot separate
  its own pair has not understood it.
* **PASS_TO_PASS** is every *previously-correct* decision on the record the rule was written
  against — every decision in the fit window that did **not** carry the targeted defect (for a
  LEAN rule, every settled lean that was right). A checklist item that fires there sends the trader
  looking for a problem that was not present; that is the regression. The tolerance is
  :data:`MAX_REGRESSIONS`, zero, as in SWE-bench.

**Two departures from SWE-bench, both stated.** SWE-bench scores an empty PASS_TO_PASS set as 1.0
(`grading.py:306-308`, with its own ``# TODO: Don't factor in p2p metrics``); here an empty set is a
refusal, because "broke nothing" over nothing is not evidence. And SWE-bench's tests are fixed while
a rule is free to describe its target so narrowly that nothing else can fail it — a three-way
conjunction naming the one decision passes PASS_TO_PASS trivially. So a rule must also fire on at
least :data:`MIN_SUPPORT` fit-window decisions (review's own ``MIN_FIRINGS``): below that it is an
anecdote, not a rule. And a rule whose target an already-admitted rule catches is not a fix — the
pair is skipped before any model call is spent on it, exactly as SWE-bench would not list an
already-passing test as FAIL_TO_PASS.

**The split is chronological and mandatory.** Decisions are ordered by ledger sequence; the first
half (`eval/reviewoos.py`'s :data:`~argus.eval.reviewoos.SPLIT`, fixed before this module existed)
is the fit window — pairs are drawn from it and the gate replays over it — and the rest is held
out. A LEAN label only exists once the lean settles, 24 to 42 hours later on this record, so a
held-out decision taken before the last fit-window lean settled could not have been reviewed with a
rule written from that lean: those decisions are **embargoed**, dropped from the held-out window
and counted (the embargo of López de Prado's purged cross-validation, the same reason).

**What is published.** Proposals made, admitted and rejected (with the reason for each), and on the
held-out window: the precision, recall, lift and review status of every admitted rule and of the
five hand-written :data:`~argus.desk.review.STANDING_RULES`, the checklist-level precision and
recall per defect kind for each, and — the test of the gate rather than of the rules — the same
held-out numbers for the rules the gate *rejected*. A gate whose admitted rules do no better out of
sample than its rejected ones is a filter that costs proposals and buys nothing, and the report says
so in those words when it happens.

    python -m argus.eval.regression_gate                       # deterministic proposer only
    python -m argus.eval.regression_gate --model qwen --max-calls 30
    python -m argus.eval.regression_gate --journal my_trades.jsonl
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from statistics import fmean
from typing import Any

from argus.desk.review import (
    DESK_FEATURES,
    MIN_DECISIONS,
    MIN_FIRINGS,
    STANDING_RULES,
    Defect,
    DefectKind,
    Feature,
    FeatureValue,
    Rule,
    RulePerformance,
    RuleSpec,
    _corrected,
    _load,
    attach_decisions,
    defects_from_leans,
    defects_from_notes,
    defects_from_risk,
    evaluate,
    features_of,
    lean_settled,
)
from argus.desk.rule_proposer import (
    ContrastPair,
    ContrastProposer,
    InductionProposer,
    ModelProposer,
    Proposal,
    Proposer,
    interleave,
    journal_records,
    pair_contrasts,
)
from argus.eval.artefact import write
from argus.eval.reviewoos import SPLIT, split_records

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "rule_proposals.json"
JOURNAL_REPORT_PATH = DATA / "rule_proposals_journal.json"
NOTES_PATH = DATA / "desk_notes.jsonl"
RISK_PATH = DATA / "risk_records.jsonl"
LEDGER_PATH = DATA / "paper_ledger.jsonl"
"""The same three files `paper/runner.py` names (`:81-93`). Not imported from there: the runner
pulls in the whole live desk, and a replay over recorded files must not depend on it importing."""

MAX_REGRESSIONS = 0
"""PASS_TO_PASS failures tolerated. SWE-bench's own bar: ``p2p == 1``."""

MIN_SUPPORT = MIN_FIRINGS
"""Fewest fit-window firings for a rule to be a rule rather than a description of one decision."""

KINDS = (DefectKind.GROUNDING, DefectKind.CONFLICT, DefectKind.CONTRADICTION, DefectKind.LEAN)
"""The defect kinds with an observed record to write rules against. RISK_INTERVENTION has never
occurred and OUTCOME needs a settled trade; neither has anything to contrast."""

QWEN_CALL_CAP = 30

REASON_CODES = {
    "target_not_fixed": "FAIL_TO_PASS failed: silent on the decision it was written from",
    "fires_on_contrast": "fires on the clean decision it was contrasted with",
    "regression": "PASS_TO_PASS failed: fires on a previously-correct decision",
    "nothing_to_regress": "no previously-correct decision exists to test against",
    "below_support": f"fires on fewer than {MIN_SUPPORT} fit-window decisions",
    "redundant": "catches nothing the admitted checklist does not already catch",
    "no_rule": "the proposer returned no rule",
    "admitted": "FULL: fixes its target and breaks nothing",
}
"""Every reason the gate gives, by code. A rejection can carry several."""


class Resolution(StrEnum):
    """SWE-bench's `ResolvedStatus`, minus PARTIAL: a rule has exactly one FAIL_TO_PASS case, so
    it either fixes it or does not."""

    FULL = "full"
    NO = "no"


def compute_fail_to_pass(report: dict[str, dict[str, list[int]]]) -> float:
    """Share of FAIL_TO_PASS cases fixed (`grading.py:288-295`)."""
    f2p = report["FAIL_TO_PASS"]
    total = len(f2p["success"]) + len(f2p["failure"])
    return 1.0 if total == 0 else len(f2p["success"]) / total


def compute_pass_to_pass(report: dict[str, dict[str, list[int]]]) -> float | None:
    """Share of PASS_TO_PASS cases kept (`grading.py:298-308`). ``None`` on an empty set, where
    SWE-bench returns 1 — see the module docstring for why that is refused here."""
    p2p = report["PASS_TO_PASS"]
    total = len(p2p["success"]) + len(p2p["failure"])
    return None if total == 0 else len(p2p["success"]) / total


@dataclass(frozen=True)
class Window:
    """One side of the split: its decisions, the defects observed within it, and for LEAN which
    decisions had a lean that could be graded at all."""

    records: tuple[dict[str, Any], ...]
    defects: tuple[Defect, ...]
    lean_graded: frozenset[int] | None = None
    vocabulary: dict[str, Feature] = field(default_factory=lambda: DESK_FEATURES)

    @cached_property
    def features(self) -> dict[int, dict[str, FeatureValue]]:
        """Every record's features, read once. A replay tries hundreds of rules on the same
        decisions, and re-parsing each note for each rule made the first version take minutes."""
        return {int(r.get("seq", 0)): features_of(r, self.vocabulary) for r in self.records}

    def fired(self, spec: RuleSpec, kind: DefectKind) -> set[int]:
        feats = self.features
        return {s for s in (int(r.get("seq", 0)) for r in self.eligible(kind))
                if spec.matches(feats[s])}

    def rule(self, spec: RuleSpec) -> Rule:
        """``spec`` compiled against this window's cached features (falls back to reading the
        record for a decision outside the window)."""
        feats = self.features
        vocab = self.vocabulary

        def predicate(record: dict[str, Any]) -> bool:
            got = feats.get(int(record.get("seq", 0)))
            return spec.matches(got) if got is not None else spec.fires(record, vocab)

        return Rule(name=spec.name, prompt=spec.prompt, rationale=spec.rationale,
                    targets=spec.targets, predicate=predicate)

    def eligible(self, kind: DefectKind) -> list[dict[str, Any]]:
        if kind is DefectKind.LEAN and self.lean_graded is not None:
            return [r for r in self.records if int(r.get("seq", 0)) in self.lean_graded]
        return list(self.records)

    def marked(self, kind: DefectKind) -> set[int]:
        seqs = {int(r.get("seq", 0)) for r in self.eligible(kind)}
        return {d.seq for d in self.defects if d.kind is kind and d.seq in seqs}

    def graded_for(self, kind: DefectKind) -> set[int] | None:
        return set(self.lean_graded) if kind is DefectKind.LEAN and self.lean_graded else None


@dataclass(frozen=True, slots=True)
class GateResult:
    """One proposal, judged."""

    proposal: Proposal
    report: dict[str, dict[str, list[int]]]
    """SWE-bench's report shape: ``{FAIL_TO_PASS: {success, failure}, PASS_TO_PASS: {...}}``."""

    support: int
    new_catches: int
    admitted: bool
    reasons: tuple[str, ...]
    codes: tuple[str, ...] = ()
    """One stable code per reason (:data:`REASON_CODES`), so rejections can be counted."""

    @property
    def resolution(self) -> Resolution:
        p2p = compute_pass_to_pass(self.report)
        ok = compute_fail_to_pass(self.report) == 1.0 and p2p is not None and p2p == 1.0
        return Resolution.FULL if ok else Resolution.NO

    @property
    def regressions(self) -> int:
        return len(self.report["PASS_TO_PASS"]["failure"])

    def as_dict(self) -> dict[str, Any]:
        p2p = self.report["PASS_TO_PASS"]
        return {
            **self.proposal.as_dict(),
            "admitted": self.admitted,
            "resolution": str(self.resolution),
            "fail_to_pass": self.report["FAIL_TO_PASS"],
            "pass_to_pass": {"kept": len(p2p["success"]), "broken": p2p["failure"][:40],
                             "broken_count": len(p2p["failure"])},
            "fit_window_support": self.support,
            "new_catches": self.new_catches,
            "reasons": list(self.reasons),
            "reason_codes": list(self.codes),
        }


@dataclass
class RegressionGate:
    """Admits rules one at a time against a fixed fit window, remembering what it admitted."""

    window: Window
    vocabulary: dict[str, Feature] = field(default_factory=lambda: DESK_FEATURES)
    min_support: int = MIN_SUPPORT
    max_regressions: int = MAX_REGRESSIONS
    admitted: list[RuleSpec] = field(default_factory=list)
    results: list[GateResult] = field(default_factory=list)

    def firing(self, spec: RuleSpec, kind: DefectKind) -> set[int]:
        return self.window.fired(spec, kind)

    def covered(self, kind: DefectKind) -> set[int]:
        """Target defects the admitted checklist already catches."""
        marked = self.window.marked(kind)
        out: set[int] = set()
        for spec in self.admitted:
            if kind in spec.targets:
                out |= self.firing(spec, kind) & marked
        return out

    def already_fixed(self, pair: ContrastPair) -> bool:
        return pair.bad_seq in self.covered(pair.kind)

    def judge(self, proposal: Proposal) -> GateResult:
        pair = proposal.pair
        kind = pair.kind
        empty: dict[str, dict[str, list[int]]] = {
            "FAIL_TO_PASS": {"success": [], "failure": [pair.bad_seq]},
            "PASS_TO_PASS": {"success": [], "failure": []},
        }
        if proposal.spec is None:
            result = GateResult(proposal, empty, 0, 0, False,
                                (proposal.refused or "no rule proposed",), ("no_rule",))
            self.results.append(result)
            return result

        spec = proposal.spec
        fired = self.firing(spec, kind)
        marked = self.window.marked(kind)
        eligible = {int(r.get("seq", 0)) for r in self.window.eligible(kind)}
        previously_correct = eligible - marked
        fixed = pair.bad_seq in fired
        report = {
            "FAIL_TO_PASS": {"success": [pair.bad_seq] if fixed else [],
                             "failure": [] if fixed else [pair.bad_seq]},
            "PASS_TO_PASS": {"success": sorted(previously_correct - fired),
                             "failure": sorted(previously_correct & fired)},
        }
        new = len((fired & marked) - self.covered(kind))
        found: list[tuple[str, str]] = []
        if not fixed:
            found.append(("target_not_fixed",
                          f"does not fire on its own target decision {pair.bad_seq}"))
        if pair.good_seq in fired:
            found.append(("fires_on_contrast",
                          f"fires on the clean decision {pair.good_seq} it was contrasted with"))
        if not previously_correct:
            found.append(("nothing_to_regress",
                          "no previously-correct decision to regress against; not evidence"))
        broken = len(report["PASS_TO_PASS"]["failure"])
        if broken > self.max_regressions:
            found.append(("regression",
                          f"breaks {broken} previously-correct decision(s) of "
                          f"{len(previously_correct)} (tolerance {self.max_regressions})"))
        if len(fired) < self.min_support:
            found.append(("below_support",
                          f"fires on {len(fired)} fit-window decision(s), below "
                          f"{self.min_support}: an anecdote, not a rule"))
        if fixed and new == 0:
            found.append(("redundant",
                          "adds no catch the admitted checklist does not already make"))
        admitted = not found
        if admitted:
            self.admitted.append(spec)
            found.append(("admitted",
                          f"fixes its target, keeps all {len(previously_correct)} "
                          f"previously-correct decisions, fires on {len(fired)}"))
        result = GateResult(proposal, report, len(fired), new, admitted,
                            tuple(t for _, t in found), tuple(c for c, _ in found))
        self.results.append(result)
        return result


# --- the replay -----------------------------------------------------------------------------------


def split_windows(
    records: Sequence[dict[str, Any]],
    defects: Sequence[Defect],
    *,
    lean_graded: set[int] | None,
    settled_at: dict[int, datetime] | None = None,
    share: float = SPLIT,
    vocabulary: dict[str, Feature] | None = None,
) -> tuple[Window, Window, int]:
    """Chronological fit / held-out windows, with the held-out side embargoed past the last
    fit-window lean's settlement. Returns the number of decisions the embargo removed."""
    fit, held = split_records(records, share=share)
    fit_seqs = {int(r.get("seq", 0)) for r in fit}
    embargo_until: datetime | None = None
    if settled_at and lean_graded:
        labels = [settled_at[s] for s in fit_seqs & lean_graded if s in settled_at]
        embargo_until = max(labels) if labels else None
    kept: list[dict[str, Any]] = []
    for r in held:
        at = r.get("at")
        if embargo_until is not None and at and datetime.fromisoformat(str(at)) < embargo_until:
            continue
        kept.append(r)
    held_seqs = {int(r.get("seq", 0)) for r in kept}

    def window(seqs: set[int], rows: Sequence[dict[str, Any]]) -> Window:
        return Window(
            records=tuple(rows),
            defects=tuple(d for d in defects if d.seq in seqs),
            lean_graded=None if lean_graded is None else frozenset(lean_graded & seqs),
            vocabulary=DESK_FEATURES if vocabulary is None else vocabulary,
        )

    return window(fit_seqs, fit), window(held_seqs, kept), len(held) - len(kept)


def held_out_performance(
    rules: Sequence[tuple[Rule, frozenset[DefectKind]]], window: Window,
) -> list[RulePerformance]:
    """Grade rules on the held-out window with review's own `evaluate`, then Benjamini-Hochberg
    across them together, exactly as :func:`argus.desk.review.review` does in production.

    A LEAN rule is graded only on held-out decisions whose lean settled, for the same reason the
    gate restricts its PASS_TO_PASS set."""
    graded: list[RulePerformance] = []
    for rule, targets in rules:
        kinds = list(targets)
        records = window.eligible(kinds[0]) if len(kinds) == 1 else list(window.records)
        graded.append(evaluate(rule, records, window.defects, min_decisions=MIN_DECISIONS))
    return _corrected(graded)


def checklist_score(
    rules: Sequence[tuple[Rule, frozenset[DefectKind]]], window: Window, kind: DefectKind,
) -> dict[str, Any]:
    """Precision and recall of a whole checklist for one defect kind: the union of its rules that
    target the kind. A checklist with no rule for the kind catches none of it — recall 0, stated,
    not omitted."""
    records = window.eligible(kind)
    marked = window.marked(kind)
    members = [rule for rule, targets in rules if kind in targets]
    fired: set[int] = set()
    for rule in members:
        for r in records:
            try:
                if rule.predicate(r):
                    fired.add(int(r.get("seq", 0)))
            except Exception:  # a broken predicate must not take the replay down
                continue
    caught = len(fired & marked)
    return {
        "kind": str(kind), "rules": [r.name for r in members], "decisions": len(records),
        "defects": len(marked), "fired": len(fired), "caught": caught,
        "precision": round(caught / len(fired), 4) if fired else None,
        "recall": round(caught / len(marked), 4) if marked else None,
        "base_rate": round(len(marked) / len(records), 4) if records else None,
    }


def _run_proposer(
    proposer: Proposer, pairs: Sequence[ContrastPair], fit: Window,
    vocabulary: dict[str, Feature], *, max_calls: int | None = None,
    max_pairs: int | None = None, attempts_per_call: int = 1,
) -> tuple[RegressionGate, int, int]:
    """Walk the pairs through one proposer and its own gate. Returns the gate, the pairs skipped
    because an admitted rule already fixed them, and the model calls spent."""
    gate = RegressionGate(window=fit, vocabulary=vocabulary)
    skipped = 0
    calls = 0
    tried = 0
    for pair in pairs:
        if max_pairs is not None and tried >= max_pairs:
            break
        if max_calls is not None and calls + attempts_per_call > max_calls:
            break
        if gate.already_fixed(pair):
            skipped += 1
            continue
        proposal = proposer.propose(pair)
        calls += proposal.calls
        tried += 1
        gate.judge(proposal)
    return gate, skipped, calls


def _summary(gate: RegressionGate, skipped: int, calls: int) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    for result in gate.results:
        if result.admitted:
            continue
        for code in result.codes:
            reasons[code] = reasons.get(code, 0) + 1
    return {
        "pairs_tried": len(gate.results),
        "pairs_skipped_already_fixed": skipped,
        "proposals_with_a_rule": sum(1 for r in gate.results if r.proposal.spec is not None),
        "refusals": sum(1 for r in gate.results if r.proposal.spec is None),
        "admitted": sum(1 for r in gate.results if r.admitted),
        "rejected": sum(1 for r in gate.results if not r.admitted),
        "rejection_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "model_calls": calls,
        "results": [r.as_dict() for r in gate.results],
    }


def gate_discrimination(
    gate: RegressionGate, held: Window, vocabulary: dict[str, Feature],
) -> dict[str, Any]:
    """Does the gate pick better rules than it throws away? Held-out numbers for both sides.

    The rejected rules are graded as a diagnostic family of their own — they are not part of any
    checklist, so they do not dilute the admitted rules' multiple-testing correction."""

    def graded(results: Sequence[GateResult]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for res in results:
            spec = res.proposal.spec
            if spec is None:
                continue
            perf = held_out_performance([(held.rule(spec), spec.targets)], held)[0]
            rows.append({"rule": spec.name, "status": str(perf.status), "fired": perf.fired,
                         "precision": perf.precision, "recall": perf.recall, "lift": perf.lift})
        return rows

    admitted = graded([r for r in gate.results if r.admitted])
    rejected = graded([r for r in gate.results if not r.admitted])

    def side(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        lifts = [r["lift"] for r in rows if r["lift"] is not None]
        keeps = sum(1 for r in rows if r["status"] in {"active", "earning"})
        return {
            "rules": len(rows), "fired_out_of_sample": sum(1 for r in rows if r["fired"]),
            "mean_lift": round(fmean(lifts), 4) if lifts else None,
            "share_lift_above_1": round(sum(1 for x in lifts if x > 1) / len(lifts), 4)
            if lifts else None,
            "keeps_its_place": keeps,
        }

    a, r = side(admitted), side(rejected)
    if a["mean_lift"] is None or r["mean_lift"] is None:
        verdict = ("UNDEFINED: one side of the gate has no rule that fired out of sample, so the "
                   "gate cannot be compared with its own rejections on this record")
    elif a["mean_lift"] > r["mean_lift"]:
        verdict = (f"the gate's admitted rules averaged lift {a['mean_lift']:.2f} out of sample "
                   f"against {r['mean_lift']:.2f} for the rules it rejected")
    else:
        verdict = (f"the gate did NOT pick better rules: admitted lift {a['mean_lift']:.2f} out of "
                   f"sample against {r['mean_lift']:.2f} for the rules it rejected — a filter that "
                   f"costs proposals and buys nothing on this record")
    return {"admitted": a, "rejected": r, "verdict": verdict,
            "admitted_detail": admitted, "rejected_detail": rejected}


def run(
    *,
    records: Sequence[dict[str, Any]],
    defects: Sequence[Defect],
    lean_graded: set[int] | None,
    settled_at: dict[int, datetime] | None = None,
    model: Any = None,
    max_calls: int = QWEN_CALL_CAP,
    vocabulary: dict[str, Feature] | None = None,
    kinds: Sequence[DefectKind] = KINDS,
    hand_written: Sequence[Rule] = STANDING_RULES,
    journal: bool = False,
    share: float = SPLIT,
) -> dict[str, Any]:
    """The whole replay: split, pair, propose, gate, grade out of sample, compare."""
    vocab = DESK_FEATURES if vocabulary is None else vocabulary
    fit, held, embargoed = split_windows(
        records, defects, lean_graded=lean_graded, settled_at=settled_at, share=share,
        vocabulary=vocab,
    )
    groups: list[list[ContrastPair]] = []
    available: dict[str, int] = {}
    for kind in kinds:
        if len(fit.marked(kind)) < MIN_SUPPORT:
            available[str(kind)] = 0
            continue
        pairs = pair_contrasts(fit.records, fit.defects, kind,
                               graded=fit.graded_for(kind), vocabulary=vocab)
        available[str(kind)] = len(pairs)
        groups.append(pairs)
    ordered = interleave(groups)

    proposers: dict[str, dict[str, Any]] = {}
    gates: dict[str, RegressionGate] = {}
    model_tried: int | None = None
    if model is not None:
        mp = ModelProposer(model=model, vocabulary=vocab, journal=journal)
        gate, skipped, calls = _run_proposer(
            mp, ordered, fit, vocab, max_calls=max_calls, attempts_per_call=mp.attempts,
        )
        proposers["model"] = _summary(gate, skipped, calls)
        gates["model"] = gate
        model_tried = len(gate.results)
    # The two model-free proposers run on the same pair budget the model had, so each comparison
    # is on the same input; with a model present they also run over every pair, which is what
    # they would do in production at no cost.
    def contrast(name: str) -> Proposer:
        return ContrastProposer(records=fit.records, vocabulary=vocab, journal=journal, name=name)

    def induction(name: str) -> Proposer:
        return InductionProposer(records=fit.records, defects=fit.defects,
                                 graded=set(fit.lean_graded) if fit.lean_graded else None,
                                 vocabulary=vocab, journal=journal, name=name)

    runs: list[tuple[str, Proposer, int | None]] = [
        ("contrast", contrast("contrast"), model_tried),
        ("induction", induction("induction"), model_tried),
    ]
    if model_tried is not None:
        runs += [("contrast_all_pairs", contrast("contrast_all_pairs"), None),
                 ("induction_all_pairs", induction("induction_all_pairs"), None)]
    for name, proposer, budget in runs:
        gate, skipped, calls = _run_proposer(proposer, ordered, fit, vocab, max_pairs=budget)
        proposers[name] = _summary(gate, skipped, calls)
        gates[name] = gate

    # Each checklist is its own multiple-testing family, as it would be in production: the
    # hand-written five together, and each proposer's admitted rules together. Pooling them would
    # let one proposer's many rules raise the bar for another's.
    hand = [(r, r.targets) for r in hand_written]
    admitted_by: dict[str, list[tuple[Rule, frozenset[DefectKind]]]] = {
        name: [(held.rule(s), s.targets) for s in g.admitted] for name, g in gates.items()
    }
    held_out: dict[str, list[dict[str, Any]]] = {
        "hand_written": [p.as_dict() for p in held_out_performance(hand, held)] if hand else [],
    }
    for name, rules in admitted_by.items():
        held_out[name] = [p.as_dict() for p in held_out_performance(rules, held)] if rules else []

    comparison: list[dict[str, Any]] = []
    for kind in kinds:
        row: dict[str, Any] = {"kind": str(kind),
                               "hand_written": checklist_score(
                                   [(r, r.targets) for r in hand_written], held, kind)}
        for name, rules in admitted_by.items():
            row[name] = checklist_score(rules, held, kind)
        comparison.append(row)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "split": share, "max_regressions": MAX_REGRESSIONS, "min_support": MIN_SUPPORT,
            "max_conditions": 3, "qwen_call_cap": max_calls if model is not None else 0,
            "kinds": [str(k) for k in kinds],
            "fixed_before_running": True,
        },
        "windows": {
            "decisions": len(records), "fit": len(fit.records), "held_out": len(held.records),
            "embargoed": embargoed,
            "fit_defects": {str(k): len(fit.marked(k)) for k in kinds},
            "held_out_defects": {str(k): len(held.marked(k)) for k in kinds},
            "held_out_eligible": {str(k): len(held.eligible(k)) for k in kinds},
        },
        "pairs_available": available,
        "proposers": proposers,
        "held_out": held_out,
        "checklist_comparison": comparison,
        "gate_discrimination": {
            name: gate_discrimination(g, held, vocab) for name, g in gates.items()
        },
        "scope_statement": (
            "Rules are written from contrast pairs drawn ONLY from the earlier half of the record "
            "by ledger sequence, gated ONLY on that half, and graded on the later half, with "
            "held-out decisions taken before the last fit-window lean settled removed (embargo). "
            "Every threshold above was fixed before the first run. Defects come from the desk's "
            "own independent checkers and from settled leans, never from the rules. "
            "NOT CLAIMED: that an admitted rule prevents anything in live use — it was graded on "
            "a replay. NOT CLAIMED for OUTCOME or RISK_INTERVENTION: no trade has settled and the "
            "Constitution has never bound, so there is nothing to write a rule against."
        ),
    }


def render(report: dict[str, Any]) -> list[str]:
    w = report["windows"]
    lines = [
        f"RULE PROPOSALS BEHIND A REGRESSION GATE — {w['decisions']} decisions, fit {w['fit']}, "
        f"held out {w['held_out']} ({w['embargoed']} embargoed)",
        f"  pairs available: {report['pairs_available']}",
    ]
    for name, s in report["proposers"].items():
        lines.append(
            f"  {name:20} tried {s['pairs_tried']:3}  rules {s['proposals_with_a_rule']:3}  "
            f"admitted {s['admitted']:3}  rejected {s['rejected']:3}  calls {s['model_calls']}"
        )
        for reason, n in list(s["rejection_reasons"].items())[:4]:
            lines.append(f"      {n:4} x {reason}")
    lines.append("  held out, per defect kind (precision / recall):")
    for row in report["checklist_comparison"]:
        cells = []
        for name, score in row.items():
            if name == "kind":
                continue
            p = "n/a" if score["precision"] is None else f"{score['precision']:.0%}"
            r = "n/a" if score["recall"] is None else f"{score['recall']:.0%}"
            cells.append(f"{name} {p}/{r} ({len(score['rules'])} rules)")
        base = row["hand_written"]["base_rate"]
        shown = "n/a" if base is None else f"{base:.0%}"
        lines.append(f"    {row['kind']:14} base {shown:>4}: " + "; ".join(cells))
    for name, d in report["gate_discrimination"].items():
        lines.append(f"  gate [{name}]: {d['verdict']}")
    return lines


MIN_JOURNAL_TRADES = 2 * MIN_DECISIONS
"""Graded trades a journal needs before it is split: each half must reach review's own floor, or
every rule on the held-out side is PROPOSED by construction and the review says nothing."""


def journal_review(lines: Sequence[str], *, model: Any = None,
                   max_calls: int = QWEN_CALL_CAP) -> dict[str, Any]:
    """The whole loop on a trader's own journal: the entry point the console calls.

    Takes the journal's JSON lines as the trader supplied them and returns either the report or a
    refusal that names what is missing — never a checklist built from too few trades. With no
    model it runs the two model-free proposers only, which cost nothing and are safe to run on
    every upload; a model is the caller's decision and the caller's budget.
    """
    from argus.desk.rule_proposer import JOURNAL_FEATURES, JournalError, parse_journal

    try:
        entries = parse_journal(lines)
    except JournalError as exc:
        return {"refused": str(exc)}
    records, defects, graded = journal_records(entries)
    if len(graded) < MIN_JOURNAL_TRADES:
        return {"refused": (
            f"{len(graded)} closed trade(s) with a result outside the dead zone; "
            f"{MIN_JOURNAL_TRADES} are needed before a rule can be written on half of them and "
            f"tested on the other half"
        )}
    report = run(records=records, defects=defects, lean_graded=graded, model=model,
                 max_calls=max_calls, vocabulary=JOURNAL_FEATURES,
                 kinds=(DefectKind.OUTCOME,), hand_written=(), journal=True)
    report["windows"]["graded_trades"] = len(graded)
    return report


def admitted_checklist(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Every admitted rule in a report with its held-out grade, best-evidenced first — the rows a
    reader is shown. A rule that failed out of sample is listed with that status, not hidden."""
    rows: list[dict[str, Any]] = []
    for name, graded in report.get("held_out", {}).items():
        if name == "hand_written":
            continue
        for perf in graded:
            rows.append({"proposer": name, **perf})
    order = {"active": 0, "earning": 1}
    return sorted(rows, key=lambda r: (order.get(str(r.get("status")), 9),
                                       -(r.get("lift") or 0.0)))


def load_record() -> tuple[list[dict[str, Any]], list[Defect], set[int], dict[int, datetime]]:
    """ARGUS's own recorded decisions, their observed defects, graded leans and settlement times."""
    from argus.paper.ledger import PaperLedger

    entries = [e for e in PaperLedger(path=LEDGER_PATH).entries
               if str(getattr(e, "kind", "decision")) == "decision"]
    notes = _load(NOTES_PATH)
    risk = _load(RISK_PATH)
    records = attach_decisions(notes, entries)
    defects = [*defects_from_notes(notes), *defects_from_risk(risk), *defects_from_leans(entries)]
    settled = {int(e.seq): datetime.fromisoformat(str(e.settled_at))
               for e in entries if getattr(e, "settled_at", None)}
    return records, defects, lean_settled(entries), settled


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="rule proposals behind a regression gate")
    parser.add_argument("--model", choices=["none", "qwen"], default="none")
    parser.add_argument("--max-calls", type=int, default=QWEN_CALL_CAP)
    parser.add_argument("--journal", default="", help="a trader's JSON-lines trade journal")
    args = parser.parse_args(argv)

    model: Any = None
    if args.model == "qwen":
        import os

        from argus.llm.qwen import QwenClient, TokenBudget

        if not os.environ.get("BITGET_QWEN_API_KEY"):
            print("BITGET_QWEN_API_KEY not set; load .secrets/qwen.env into the environment first.")
            return 1
        model = QwenClient(budget=TokenBudget(limit=150_000))

    if args.journal:
        report = journal_review(Path(args.journal).read_text(encoding="utf-8").splitlines(),
                                model=model, max_calls=args.max_calls)
        if "refused" in report:
            print(f"refused: {report['refused']}")
            return 1
        out = JOURNAL_REPORT_PATH
    else:
        records, defects, graded, settled = load_record()
        report = run(records=records, defects=defects, lean_graded=graded, settled_at=settled,
                     model=model, max_calls=args.max_calls)
        out = REPORT_PATH
    if model is not None:
        report["qwen_calls_made"] = int(getattr(model, "calls", 0))
    for line in render(report):
        print(line)
    write(out, report)
    print(f"\nwritten to {out}")
    return 0


__all__ = [
    "KINDS",
    "MAX_REGRESSIONS",
    "MIN_JOURNAL_TRADES",
    "MIN_SUPPORT",
    "QWEN_CALL_CAP",
    "REASON_CODES",
    "REPORT_PATH",
    "GateResult",
    "RegressionGate",
    "Resolution",
    "Window",
    "admitted_checklist",
    "checklist_score",
    "compute_fail_to_pass",
    "compute_pass_to_pass",
    "gate_discrimination",
    "held_out_performance",
    "journal_review",
    "load_record",
    "main",
    "render",
    "run",
    "split_windows",
]


if __name__ == "__main__":
    raise SystemExit(main())
