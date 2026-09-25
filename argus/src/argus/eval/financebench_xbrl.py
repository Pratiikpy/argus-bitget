"""Numeric 10-K questions answered from SEC XBRL, on FinanceBench's own questions, against
FinanceBench's own human-graded models on the same questions.

**The rival and the input.** FinanceBench (Patronus AI, arXiv 2311.11944) released 150 of its
questions with the answers sixteen model configurations gave them — GPT-4, GPT-4-Turbo, Claude 2
and Llama 2, closed book, with a vector store, with the page in context, with the oracle page —
each graded by a person as a correct answer, an incorrect answer or a refusal
(``results/*.jsonl`` in the repository). The comparison here is that exact set: the same
question ids, the same wording, scored against the same gold answers. Nothing from the repository
is vendored; it carries no licence file, so the questions and grades are read from the local
clone at run time and the artefact records only ids, ARGUS's answers and the tallies.

**What ARGUS runs.** :class:`argus.research.filing_qa.FilingQA`, which never reads prose: it
resolves the company, the fiscal year and a formula over statement lines, and evaluates it on
the figures the company filed in XBRL. Everything outside that — a quarter, a cause, a segment, a
non-GAAP figure — is an abstention with its reason. Every SEC response is written to a compact
snapshot (:data:`SNAPSHOT_DIR`) so the run replays offline to the same answers.

**How an answer is graded.** A question whose gold answer is a single figure (FinanceBench's
*metrics-generated* class) is graded by number: correct when ARGUS's value is within 1% of the
gold figure, or within 0.011 of it for a figure below 1.1 (a ratio or a per-share amount reported
to two decimals). The FinanceBench grades were given by people reading prose; these are given by a
rule, stated here so a reader can apply it. A question outside that class that ARGUS answers is
graded by reading the gold prose, each grade with its reason (:data:`HAND_GRADES`); one no grade
covers is reported as ungraded.

**What this measurement is not.** Held out. FinanceBench's released questions are all there is,
and both halves have been seen: the 50 metrics-generated questions while the engine was built, the
other 100 on 2026-09-26, when four fixes followed from ten wrong answers in thirteen. The engine
as it stands is scored on questions it was corrected on. The number that would mean more — the
same engine on questions nobody tuned it on — needs new questions, and is not claimed.

    python -m argus.eval.financebench_xbrl            # live SEC, writes the snapshot
    python -m argus.eval.financebench_xbrl --offline  # replay the snapshot
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.eval import artefact
from argus.market.statement_facts import (
    CompanyFactsSource,
    ConceptSource,
    FaceStatementSource,
)
from argus.research.filing_qa import CompanyResolver, FilingQA

PACKAGE = Path(__file__).resolve().parents[3]
ROOT = PACKAGE.parent
BENCH = ROOT / "research" / "repos-themed" / "patronus-ai~financebench"
QUESTIONS = BENCH / "data" / "financebench_open_source.jsonl"
RESULTS = BENCH / "results"
SNAPSHOT_DIR = PACKAGE / "data" / "financebench_xbrl_snapshot"
REPORT_PATH = PACKAGE / "data" / "financebench_xbrl.json"

TARGET_CLASS = "metrics-generated"
RELATIVE_TOLERANCE = 0.01
ABSOLUTE_TOLERANCE = 0.011
SMALL_FIGURE = 1.1

HAND_GRADES: dict[str, tuple[str, str]] = {
    # Questions outside the numeric class that ARGUS answered with a figure, graded against the
    # gold answer's prose by reading it. Each grade names its reason; an id not listed here is
    # reported as ungraded rather than guessed.
    "financebench_id_01351": ("correct", "gold: 24.6% to 21.6%; ARGUS: -3.0 points, 24.6% to "
                                         "21.6%"),
    "financebench_id_00070": ("correct", "gold: negative working capital of -$1,561M; ARGUS: "
                                         "-$1,561M"),
    "financebench_id_00585": ("disputed", "gold: 0.62% in FY2022 against -14.76% in FY2021; "
                                          "ARGUS: -0.6% and 14.8%, the same magnitudes with the "
                                          "opposite sign (income tax over pre-tax income on a "
                                          "pre-tax loss)"),
    "financebench_id_01346": ("correct", "gold: about 20% to 23%; ARGUS: 20.2% to 22.9%"),
    "financebench_id_00005": ("disputed", "gold: positive, $831M counting operating current items "
                                          "only; ARGUS: positive, $2,278M, current assets less "
                                          "current liabilities"),
    "financebench_id_01254": ("correct", "gold: yes, $0.01 a share; ARGUS: $4.05M paid, a yes"),
    "financebench_id_00080": ("disputed", "gold: positive, $1.6B on a narrower definition; ARGUS: "
                                          "positive, $12,416M, current assets less current "
                                          "liabilities"),
    "financebench_id_00302": ("correct", "gold: yes, PP&E grew; ARGUS: +$1,137M, $13,745M to "
                                         "$14,882M"),
}


UNGUARDED_GRADES: dict[str, tuple[str, str]] = {
    # The answers the engine gives only with its coverage guard off (every arithmetic phrase no
    # longer required to be used by the formula), read against the gold prose: what the guard
    # costs and what it prevents.
    "financebench_id_00540": ("wrong", "inventory turnover 12.14 against the gold 9.5"),
    "financebench_id_00799": ("wrong", "one year's quick ratio, 0.57, for a two-year comparison "
                                       "(gold 0.67 to 0.69)"),
    "financebench_id_00684": ("wrong", "one year's gross margin, 18.5%, for 'is it improving' "
                                       "(gold: no, down 0.8 points)"),
    "financebench_id_00222": ("correct", "quick ratio 1.57, the gold figure"),
    "financebench_id_00517": ("wrong", "total revenue for a question about categories over 20% "
                                       "of it"),
    "financebench_id_00678": ("wrong", "one year's gross margin, 5.3%, for 'is it improving' "
                                       "(gold: yes)"),
}


class BenchUnavailable(RuntimeError):
    """The FinanceBench clone is not on this machine, so the comparison cannot run."""


def load_questions(path: Path = QUESTIONS) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BenchUnavailable(f"FinanceBench questions not found at {path}; clone "
                               f"https://github.com/patronus-ai/financebench there")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


_NUMBER = re.compile(r"(-)?\$?\s?(-)?(\d[\d,]*(?:\.\d+)?)")


def gold_numbers(answer: str) -> list[float]:
    """Every figure a gold answer states, in order: "$1577.00" -> [1577.0]; "from 24.6% in FY
    2021 to 21.6%" -> [24.6, 2021.0, 21.6]. Years are kept; callers pick what they need."""
    out = []
    for m in _NUMBER.finditer(answer):
        value = float(m.group(3).replace(",", ""))
        out.append(-value if (m.group(1) or m.group(2)) else value)
    return out


def close(ours: float, gold: float) -> bool:
    """The grading rule of the module docstring."""
    if abs(gold) < SMALL_FIGURE:
        return abs(ours - gold) <= ABSOLUTE_TOLERANCE
    return abs(ours - gold) <= RELATIVE_TOLERANCE * abs(gold)


@dataclass(frozen=True)
class Graded:
    financebench_id: str
    question_type: str
    status: str             # answered | abstained
    grade: str              # correct | wrong | disputed | ungraded | abstained
    answer: str
    reason: str
    formula: str

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.financebench_id, "class": self.question_type, "status": self.status,
                "grade": self.grade, "answer": self.answer, "reason": self.reason,
                "formula": self.formula}


def grade(row: dict[str, Any], status: str, value: float | None, text: str) -> str:
    """One answer's grade: by number for a figure question, from :data:`HAND_GRADES` otherwise."""
    if status != "answered" or value is None:
        return "abstained"
    if row["question_type"] == TARGET_CLASS:
        golds = gold_numbers(row["answer"])
        return "correct" if golds and close(value, golds[0]) else "wrong"
    hand = HAND_GRADES.get(row["financebench_id"])
    return hand[0] if hand else "ungraded"


def run(rows: Sequence[dict[str, Any]], *, offline: bool,
        coverage_guard: bool = True) -> list[Graded]:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    resolver = CompanyResolver(snapshot=SNAPSHOT_DIR / "companies.json", offline=offline)
    qa = FilingQA(
        facts=CompanyFactsSource(snapshot_dir=SNAPSHOT_DIR / "facts", offline=offline),
        companies=resolver,
        faces=FaceStatementSource(snapshot_dir=SNAPSHOT_DIR / "faces", offline=offline),
        concepts=ConceptSource(snapshot_dir=SNAPSHOT_DIR / "concepts", offline=offline),
        coverage_guard=coverage_guard,
    )
    out = []
    for row in rows:
        a = qa.answer(row["question"])
        out.append(Graded(row["financebench_id"], row["question_type"], a.status,
                          grade(row, a.status, a.value, a.text), a.text, a.reason, a.formula))
    if not offline:
        (SNAPSHOT_DIR / "companies.json").write_text(
            json.dumps(resolver.snapshot(), sort_keys=True, separators=(",", ":")),
            encoding="utf-8")
    return out


def rival_tallies(ids_by_class: dict[str, set[str]],
                  results_dir: Path = RESULTS) -> dict[str, dict[str, dict[str, int]]]:
    """FinanceBench's own grades, per configuration, on the same question ids."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for path in sorted(results_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        by: dict[str, dict[str, int]] = {}
        for cls, ids in ids_by_class.items():
            labels = Counter(str(r["label"]) for r in rows if r["financebench_id"] in ids)
            by[cls] = {"correct": labels.get("Correct Answer", 0),
                       "incorrect": labels.get("Incorrect Answer", 0),
                       "refusal": labels.get("Refusal", 0)}
        out[path.stem] = by
    return out


def _tally(graded: Sequence[Graded]) -> dict[str, int]:
    counts = Counter(g.grade for g in graded)
    return {k: counts.get(k, 0)
            for k in ("correct", "wrong", "disputed", "ungraded", "abstained")}


def _commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(BENCH), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build(*, offline: bool) -> dict[str, Any]:
    rows = load_questions()
    started = time.perf_counter()
    graded = run(rows, offline=offline)
    seconds = time.perf_counter() - started
    unguarded = run(rows, offline=offline, coverage_guard=False)
    classes = {TARGET_CLASS: {r["financebench_id"] for r in rows
                              if r["question_type"] == TARGET_CLASS},
               "other": {r["financebench_id"] for r in rows
                         if r["question_type"] != TARGET_CLASS},
               "all": {r["financebench_id"] for r in rows}}
    rivals = rival_tallies(classes)
    target = [g for g in graded if g.question_type == TARGET_CLASS]
    other = [g for g in graded if g.question_type != TARGET_CLASS]
    best_rival = max(rivals.items(), key=lambda kv: kv[1][TARGET_CLASS]["correct"])
    ours = _tally(target)
    guard_other = _tally([g for g in unguarded if g.question_type != TARGET_CLASS])
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": {"repository": "https://github.com/patronus-ai/financebench",
                      "commit": _commit(), "questions": len(rows),
                      "by_class": {k: len(v) for k, v in classes.items() if k != "all"}},
        "grading_rule": {"relative_tolerance": RELATIVE_TOLERANCE,
                         "absolute_tolerance_below": [ABSOLUTE_TOLERANCE, SMALL_FIGURE],
                         "hand_graded_ids": sorted(HAND_GRADES)},
        "argus": {TARGET_CLASS: ours, "other": _tally(other), "seconds": round(seconds, 1),
                  "offline_replay": offline},
        "rivals": rivals,
        "headline": {
            "argus_correct_of_50": ours["correct"],
            "best_rival": best_rival[0],
            "best_rival_correct_of_50": best_rival[1][TARGET_CLASS]["correct"],
            "rivals_with_more_correct": sorted(
                name for name, by in rivals.items()
                if by[TARGET_CLASS]["correct"] > ours["correct"]),
        },
        "ablation_coverage_guard_off": {
            "other": guard_other,
            "note": "the same run with every arithmetic phrase no longer required to be used by "
                    "the formula: answers it gives that the guarded run refuses",
            "extra_answers": [
                {"id": b.financebench_id, "answer": b.answer,
                 "grade": UNGUARDED_GRADES.get(b.financebench_id, ("ungraded", ""))[0],
                 "why": UNGUARDED_GRADES.get(b.financebench_id, ("", "not graded"))[1]}
                for a, b in zip(graded, unguarded, strict=True)
                if a.status == "abstained" and b.status == "answered"],
        },
        "not_held_out": "both classes were seen: the 50 metrics-generated questions while the "
                        "engine was built, the other 100 on 2026-09-26 when four fixes followed "
                        "ten wrong answers in thirteen",
        "answers": [g.as_dict() for g in graded],
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--offline", action="store_true",
                        help="replay the committed SEC snapshot instead of the network")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = build(offline=args.offline)
    artefact.write(REPORT_PATH, report)
    head = report["headline"]
    print(f"ARGUS {head['argus_correct_of_50']}/50 on metrics-generated; best FinanceBench "
          f"configuration {head['best_rival']} {head['best_rival_correct_of_50']}/50")
    print(f"other 100: {report['argus']['other']}")
    print(f"written {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
