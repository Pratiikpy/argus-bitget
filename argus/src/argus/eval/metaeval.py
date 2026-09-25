"""Who grades the grader? Agreement of every LLM judge in ARGUS with a labelled set.

Wherever ARGUS lets a language model *judge* something — whether a decision is already refuted,
which kind of question a trader asked, what a headline says — that judgement feeds a number
someone else reads. None of those judges had ever been checked against labels a person wrote.
This module is the harness for doing so, the labelled-set format it reads, and the register of
every judge in the codebase, measured or not.

**What was taken, from where.** openai/evals' model-graded classifier has a ``metaeval`` mode
(``evals/elsuite/modelgraded/classify.py:95-98``, **MIT**): each labelled sample carries the choice
a person made, the judge's choice is compared with it (``metascore = choice ==
test_sample["choice"]``), and the run reports the mean of those as ``metascore``
(``:123-125``). :func:`agreement` computes exactly that figure and adds what the original leaves
out:

* **Cohen's κ**, because raw agreement flatters a judge on a skewed label set — a router that
  answers "none" to everything agrees 90% of the time with a set that is 90% "none", and κ is 0.
* **Who wrote the reference labels.** openai/evals assumes its ``choice`` field is human. Here a
  labelled row must say (``labeller_kind``: ``human``, ``author`` or ``model``), and only a set
  whose every row is ``human`` can make a judge MEASURED. Agreement with the corpus author's own
  expectations is still computed and published — as NOT_HUMAN, never under the human name.
* **A floor on size.** The build plan's bar is κ on at least 50 human labels per judge
  (``research/mypr-teardowns/_SYNTHESIS.md``, row S9). Below that the result is PROVISIONAL.

The four states are combined through ``eval/evaluators.py``'s AND, cheapest rule first, so the
status of a judge is a :class:`~argus.eval.evaluators.Grade` a reader can inspect check by check.

**Labelled-set format** — one JSON object per line in ``data/metaeval/<judge>.jsonl``::

    {"judge": "lui_router", "item_id": "routerbench-00", "input": "...",
     "judge_label": "performance", "reference_label": "performance",
     "labeller": "who wrote the reference", "labeller_kind": "human" | "author" | "model",
     "labelled_at": "2026-09-25", "source": "data/router_bench.json#outcomes[0]"}

``judge_label`` must be what the judge actually returned, read from a recorded output — never
re-generated for the label set. No row here was produced by calling a model.

**First set, and why only one.** The only judge whose recorded outputs sit beside reference labels
is the LUI router: ``data/router_bench.json`` kept every one of its Qwen routings next to the
intent the corpus author expected (``eval/routerbench.py``). :func:`router_labels_from_bench`
converts those, and they are marked ``labeller_kind="author"`` because the corpus was written by
this project's own author and human authorship is NOT VERIFIED. The adversary's two recorded
challenges (``data/desk_notes.jsonl``) keep a truncated counter-case and no evidence pack, so a
label written for them would be a guess about an input nobody can see; the planner and the analysts
have no per-item recorded outputs at all. Those judges are published as UNMEASURED with the count
of recorded outputs available, rather than labelled from imagination.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from argus.eval.evaluators import Grade, combine, rule

PACKAGE = Path(__file__).resolve().parents[3]
DATA = PACKAGE / "data"
LABELS = DATA / "metaeval"
REPORT_PATH = DATA / "metaeval.json"
MIN_HUMAN_LABELS = 50
LABELLER_KINDS = ("human", "author", "model")
FIELDS = ("judge", "item_id", "input", "judge_label", "reference_label", "labeller",
          "labeller_kind", "labelled_at", "source")


class Status(StrEnum):
    MEASURED = "measured"
    """κ against at least :data:`MIN_HUMAN_LABELS` human labels."""

    PROVISIONAL = "provisional"
    """Human labels, but fewer than the floor."""

    NOT_HUMAN = "not_human"
    """Agreement computed, but at least one reference label was not written by a person."""

    UNMEASURED = "unmeasured"
    """No labelled set exists."""


@dataclass(frozen=True)
class Judge:
    """One place ARGUS asks a language model for a judgement."""

    name: str
    module: str
    call_site: str
    judges: str
    label_space: str
    recorded_outputs: str


JUDGES: tuple[Judge, ...] = (
    Judge("adversary", "argus/agents/adversary.py", "argus/agents/adversary.py:229",
          "whether the desk's own decision is already refuted, and how severely it is challenged",
          "outcome in {upheld, reduced, refused} with a severity in [0, 1]",
          "2 challenges in data/desk_notes.jsonl, counter-case truncated, no evidence pack"),
    Judge("lui_router", "argus/lui/router.py", "argus/lui/router.py:278",
          "which kind of question a trader asked, when the patterns cannot tell",
          "one console intent, or none",
          "24 routings in data/router_bench.json, each beside the corpus author's expected intent"),
    Judge("research_planner", "argus/lui/research.py", "argus/lui/research.py:3108",
          "which research engine a free-text question needs, and its inputs",
          "one ResearchKind, or none / record",
          "none per item: data/lui_final_heldout_report.json keeps only per-language totals"),
    Judge("analysts", "argus/agents/analysts.py", "argus/agents/analysts.py:158",
          "the direction and strength an evidence channel supports (event, sentiment, earnings, "
          "cross-asset, factor)",
          "signal in {bullish, bearish, neutral} with a confidence",
          "none per item: data/desk_notes.jsonl keeps the panel's net lean, not each analyst's "
          "view with its evidence"),
    Judge("citation_audit", "argus/eval/document_qa_eval.py", "argus/eval/document_qa_eval.py:458",
          "whether each answer sentence is supported by the filing passages it cites (FACT-style)",
          "verdict in {SUPPORTED, UNSUPPORTED} per sentence",
          "written by a concurrent builder on 2026-09-25; its deterministic checker is a second "
          "grader, not a human label"),
    Judge("unused_evidence_rater", "argus/eval/thesis_quality.py",
          "argus/eval/thesis_quality.py:656",
          "which piece of evidence the thesis did not use is most worth answering",
          "a rating per unused evidence item",
          "written by a concurrent builder on 2026-09-25; no labelled set"),
    Judge("debate_progress", "argus/eval/thesis_quality.py", "argus/eval/thesis_quality.py:834",
          "whether a debate is in a loop and whether progress is being made",
          "is_in_loop and is_progress_being_made, each yes/no",
          "written by a concurrent builder on 2026-09-25; scored against what happened after the "
          "stop, which is an outcome check, not a human label"),
    Judge("refusal_reason", "argus/eval/infeasibilitybench.py",
          "argus/eval/infeasibilitybench.py:1329",
          "whether the console's refusal gave the question's true reason for being unanswerable",
          "right reason / wrong reason / answered with figures, per row",
          "audits the pattern grader, which decides every grade; agreement 71 of 73 offline rows "
          "(97.3%) and 20 of 20 on the model path, data/infeasibility_bench.json; no human labels"),
)
"""Every LLM judge in ARGUS. Found by listing every ``complete_json`` call site in ``src/argus``
(2026-09-25) and keeping the ones whose output is a verdict about something else. Three were added
by other builders the same day; ``tests/test_eval_spine.py`` fails whenever a new call site appears
that is in neither this register nor :data:`NOT_JUDGES`, so a new judge cannot go unregistered."""

NOT_JUDGES: Mapping[str, str] = {
    "argus/agents/meta_pm.py:276,325": "the decision-maker itself; it is graded on outcomes "
                                       "(eval/shadow.py), not on agreement with a label",
    "argus/agents/debate.py:389": "argues a side; convergence is computed arithmetically, no "
                                  "seat issues a verdict",
    "argus/lui/translate.py:145": "translates text; generation, not judgement",
    "argus/eval/leakage.py:753,770": "the model is the subject under test, not the grader",
    "argus/desk/rule_proposer.py:604": "writes a candidate rule; whether it is kept is decided "
                                       "by a replayed regression gate, not by a model",
    "argus/lui/kindmodel.py:74": "a local trained classifier behind the same interface, not an "
                                 "LLM; it is scored on held-out corpora by eval/kindtrain.py",
    "argus/lui/fanout.py:366": "a planner choosing which researchers run; every finding is "
                               "computed by an engine, and against the fixed plan table it lost "
                               "(eval/research_depth.py), so it is off by default",
    "argus/eval/decision_primitives.py:483,491": "the prompt-cache replay of recorded calls; the "
                                                 "cache is measured, the model is not graded",
    "argus/eval/session_arena.py:678": "the rival execution desk (PACE) run as the subject of a "
                                       "comparison, not a grader — held back from publication, "
                                       "unreviewed (Activity/18_READINESS_BACKLOG.md)",
}
"""``complete_json`` call sites that are not judges, with the reason: the register is complete."""


def validate(row: Mapping[str, Any]) -> dict[str, str]:
    """One labelled row, checked. Raises ``ValueError`` naming the first defect."""
    missing = [f for f in FIELDS if f not in row]
    if missing:
        raise ValueError(f"labelled row is missing {missing}")
    if row["labeller_kind"] not in LABELLER_KINDS:
        raise ValueError(f"labeller_kind must be one of {LABELLER_KINDS}, "
                         f"not {row['labeller_kind']!r}")
    for key in ("judge_label", "reference_label", "item_id", "labeller"):
        if not str(row[key]).strip():
            raise ValueError(f"{key} is empty in item {row.get('item_id')!r}")
    return {f: str(row[f]) for f in FIELDS}


def load_labels(path: Path) -> list[dict[str, str]]:
    """Read and validate a labelled set. Duplicate item ids are a defect, not a weight."""
    rows = [validate(json.loads(line)) for line in path.read_text("utf-8").splitlines()
            if line.strip()]
    ids = Counter(r["item_id"] for r in rows)
    duplicated = sorted(i for i, n in ids.items() if n > 1)
    if duplicated:
        raise ValueError(f"duplicate item ids in {path.name}: {duplicated[:5]}")
    return rows


def cohens_kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    """κ = (p_o - p_e) / (1 - p_e). ``None`` when p_e is 1 (every label identical on both sides:
    agreement is then certain by chance and κ is undefined, not 1)."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    judge = Counter(a for a, _ in pairs)
    reference = Counter(b for _, b in pairs)
    expected = sum(judge[k] * reference[k] for k in judge.keys() | reference.keys()) / (n * n)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


@dataclass(frozen=True)
class Metaeval:
    """One judge's agreement with its labelled set."""

    judge: str
    n: int
    metascore: float | None
    kappa: float | None
    labeller_kinds: Mapping[str, int]
    confusion: Mapping[str, Mapping[str, int]]
    status: Status
    grade: Grade | None

    def as_dict(self) -> dict[str, Any]:
        return {"judge": self.judge, "n": self.n, "metascore": self.metascore,
                "kappa": self.kappa, "labeller_kinds": dict(self.labeller_kinds),
                "confusion": {k: dict(v) for k, v in self.confusion.items()},
                "status": self.status.value,
                "grade": self.grade.as_dict() if self.grade is not None else None}


@dataclass(frozen=True)
class _Labelled:
    n: int
    kinds: Mapping[str, int]
    kappa: float | None


def agreement(judge: str, rows: Iterable[Mapping[str, str]]) -> Metaeval:
    """openai/evals' ``metascore`` (mean of judge == reference) plus κ, graded into a status."""
    labelled = [r for r in rows if r["judge"] == judge]
    pairs = [(r["judge_label"], r["reference_label"]) for r in labelled]
    kinds = Counter(r["labeller_kind"] for r in labelled)
    confusion: dict[str, dict[str, int]] = {}
    for got, want in pairs:
        confusion.setdefault(want, {}).setdefault(got, 0)
        confusion[want][got] += 1
    n = len(pairs)
    if n == 0:
        return Metaeval(judge, 0, None, None, {}, {}, Status.UNMEASURED, None)
    metascore = sum(1 for a, b in pairs if a == b) / n
    kappa = cohens_kappa(pairs)
    grade = combine((
        rule("every_reference_label_is_human", lambda s: (
            set(s.kinds) == {"human"}, f"labeller kinds {dict(s.kinds)}")),
        rule(f"at_least_{MIN_HUMAN_LABELS}_labels", lambda s: (
            s.n >= MIN_HUMAN_LABELS, f"{s.n} labelled items")),
        rule("kappa_defined", lambda s: (s.kappa is not None, f"kappa {s.kappa}")),
    ), _Labelled(n, kinds, kappa), stop_at_first_failure=False)
    if grade.passed:
        status = Status.MEASURED
    elif set(kinds) != {"human"}:
        status = Status.NOT_HUMAN
    else:
        status = Status.PROVISIONAL
    return Metaeval(judge, n, metascore, kappa, dict(kinds), confusion, status, grade)


def router_labels_from_bench(path: Path = DATA / "router_bench.json") -> list[dict[str, str]]:
    """Every recorded routing in ``router_bench.json`` beside the label its corpus expected.

    Only cases that reached the router are rows: the others were answered by the patterns and the
    judge never spoke. A routing below the bench's floor of 0.0 does not exist, so ``routed_to`` is
    the judge's answer; ``None`` there is the judge declining, recorded as ``"none"``, and a case
    that must be refused expects ``"none"``.
    """
    bench = json.loads(path.read_text("utf-8"))
    rows: list[dict[str, str]] = []
    for i, case in enumerate(bench["outcomes"]):
        if not case["reached_router"]:
            continue
        rows.append(validate({
            "judge": "lui_router", "item_id": f"routerbench-{i:02d}", "input": case["ask"],
            "judge_label": case["routed_to"] or "none",
            "reference_label": "none" if case["must_refuse"] else (case["expect"] or "none"),
            "labeller": "the routerbench corpus author (eval/routerbench.py); human authorship "
                        "NOT VERIFIED",
            "labeller_kind": "author", "labelled_at": str(bench["as_of"])[:10],
            "source": f"data/router_bench.json#outcomes[{i}]"}))
    return rows


def write_labels(rows: Sequence[Mapping[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(r), ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def run(labels_dir: Path = LABELS) -> dict[str, Any]:
    """Metaeval every judge in :data:`JUDGES` from whatever labelled sets exist."""
    results: list[dict[str, Any]] = []
    for judge in JUDGES:
        path = labels_dir / f"{judge.name}.jsonl"
        rows = load_labels(path) if path.exists() else []
        measured = agreement(judge.name, rows)
        results.append({"judge": judge.name, "module": judge.module,
                        "call_site": judge.call_site, "judges": judge.judges,
                        "label_space": judge.label_space,
                        "recorded_outputs": judge.recorded_outputs,
                        "labelled_set": (path.relative_to(PACKAGE).as_posix()
                                         if path.exists() else None),
                        **measured.as_dict()})
    return {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "method": "openai/evals modelgraded metaeval (classify.py:95-98, 123-125): mean of "
                  "judge == reference, plus Cohen's kappa; MEASURED needs >= "
                  f"{MIN_HUMAN_LABELS} human labels",
        "judges": results,
        "not_judges": dict(NOT_JUDGES),
        "measured": [r["judge"] for r in results if r["status"] == Status.MEASURED.value],
        "qwen_calls": 0,
    }


def main() -> int:  # pragma: no cover - CLI
    from argus.eval import artefact

    write_labels(router_labels_from_bench(), LABELS / "lui_router.jsonl")
    report = run()
    artefact.write(REPORT_PATH, report)
    for r in report["judges"]:
        print(f"{r['judge']:18s} {r['status']:12s} n={r['n']:3d} metascore={r['metascore']} "
              f"kappa={r['kappa']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
