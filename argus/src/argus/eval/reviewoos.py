"""Does a rule learned from past defects still catch the next ones? — the review's held-out test.

`desk/review.py` grades a candidate rule by replaying it over the decisions the desk already took
and scoring it against defects found by independent checkers. That is a fit. **A rule graded on
the same decisions it was written against tells you it describes the past, not that it predicts
anything** — and "self-evolving review rules" is a claim about the future, so `eval/standing.py`
asked for an out-of-sample test and correctly found none.

**The split is chronological, never random.** A random split lets a rule be graded on decision 12
after being fitted on decision 31, which leaks the future into the past — and this project's own
standing rules already record that mistake being made elsewhere. Decisions are ordered by ledger
sequence, the first share becomes the window a rule is graded on, and the remainder is held back.

**What a pass and a failure each mean here, stated before the numbers.** A rule whose status
survives the split is one whose behaviour was a property of the desk rather than of the fortnight
it was written in. A rule that flips from a keeping status to a rejecting one is not necessarily
a bad rule — it may simply have had too few firings to be graded on either side — so the report
separates *flipped* from *not gradeable out of sample*, which are different facts that a single
"survived" percentage would merge.

**The honest limit, stated first.** `STANDING_RULES` were written by hand, not generated from the
in-sample window, so this is not the full self-evolution loop: it tests whether a rule's *grade*
generalises, not whether a rule *discovered* in-sample would have been discovered again. The
stronger experiment needs a generator, and saying that is better than letting this number be read
as more than it is.

    python -m argus.eval.reviewoos
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argus.desk.review import (
    MIN_DECISIONS,
    STANDING_RULES,
    Defect,
    Rule,
    Status,
    defects_from_notes,
    defects_from_risk,
    evaluate,
)
from argus.eval.artefact import write

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "review_oos.json"

SPLIT = 0.5
"""Share of decisions used as the in-sample window.

Half, chosen before running and not tuned afterwards. A split picked to make a result look better
is the selection effect this whole module exists to detect."""


class ReviewOosError(RuntimeError):
    """The split cannot be made honestly. Raised rather than reported as a pass."""


@dataclass(frozen=True, slots=True)
class RuleSplit:
    """One rule, graded twice — on the window it was measured in, and on the one held back."""

    rule: str
    in_sample: str
    out_of_sample: str
    in_fired: int
    out_fired: int
    in_caught: int
    out_caught: int

    @property
    def gradeable_out(self) -> bool:
        """Did the held-out half contain enough for a verdict at all?

        `review._status_for` returns `PROPOSED` when a rule has too few decisions to judge. That
        is not a failure of the rule — it is the absence of a test — and counting it as one would
        make a smaller held-out window look like worse generalisation."""
        return self.out_of_sample != str(Status.PROPOSED)

    @property
    def survived(self) -> bool:
        """Same verdict on both halves. Only meaningful when the held-out half is gradeable."""
        return self.gradeable_out and self.in_sample == self.out_of_sample

    @property
    def flipped(self) -> bool:
        return self.gradeable_out and self.in_sample != self.out_of_sample

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "in_sample": self.in_sample,
            "out_of_sample": self.out_of_sample,
            "in_fired": self.in_fired, "out_fired": self.out_fired,
            "in_caught": self.in_caught, "out_caught": self.out_caught,
            "gradeable_out_of_sample": self.gradeable_out,
            "survived": self.survived,
            "flipped": self.flipped,
        }


def split_records(
    records: Sequence[dict[str, Any]], *, share: float = SPLIT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Chronological split by ledger sequence. Never random — see the module docstring."""
    if not records:
        raise ReviewOosError("no decisions to split; an empty held-out set proves nothing")
    ordered = sorted(records, key=lambda r: int(r.get("seq", 0)))
    cut = int(len(ordered) * share)
    if cut == 0 or cut == len(ordered):
        raise ReviewOosError(
            f"a {share:.0%} split of {len(ordered)} decisions leaves one side empty"
        )
    return ordered[:cut], ordered[cut:]


def _defects_for(
    records: Sequence[dict[str, Any]], risk: Sequence[dict[str, Any]]
) -> list[Defect]:
    """Defects observed *within* one window only.

    The window's own defects, not the whole record's — grading a held-out window against defects
    found in the in-sample one would be the leak this split exists to prevent, running backwards.
    """
    seqs = {int(r.get("seq", 0)) for r in records}
    found = [*defects_from_notes(records), *defects_from_risk(risk)]
    return [d for d in found if d.seq in seqs]


def run(
    *,
    notes: Sequence[dict[str, Any]] | None = None,
    risk: Sequence[dict[str, Any]] = (),
    rules: Sequence[Rule] = STANDING_RULES,
    share: float = SPLIT,
) -> dict[str, Any]:
    """Grade every standing rule on both halves of a chronological split."""
    if notes is None:
        # `NOTES_PATH` and `RISK_PATH` live in `paper.runner`, not in `desk.review` — the review
        # module imports them from there inside its own CLI. The first version of this function
        # imported them from `desk.review` on the assumption that is where they were, and failed
        # at import. Read, not guessed, the second time.
        from argus.desk.review import _load
        from argus.paper.runner import NOTES_PATH, RISK_PATH

        notes = _load(NOTES_PATH)
        if not risk:
            risk = _load(RISK_PATH)
    early, late = split_records(notes, share=share)
    early_defects = _defects_for(early, risk)
    late_defects = _defects_for(late, risk)

    splits: list[RuleSplit] = []
    for rule in rules:
        # `min_decisions` is left at its production value on both halves. Lowering it for the
        # smaller held-out window would manufacture verdicts the real system would not have
        # issued, which is the same thing as grading a rule the desk would have called PROPOSED.
        a = evaluate(rule, early, early_defects, min_decisions=MIN_DECISIONS)
        b = evaluate(rule, late, late_defects, min_decisions=MIN_DECISIONS)
        splits.append(RuleSplit(
            rule=rule.name,
            in_sample=str(a.status), out_of_sample=str(b.status),
            in_fired=a.fired, out_fired=b.fired,
            in_caught=a.caught, out_caught=b.caught,
        ))

    gradeable = [s for s in splits if s.gradeable_out]
    survived = [s for s in gradeable if s.survived]
    return {
        "decisions": len(notes),
        "split_share": share,
        "in_sample_decisions": len(early),
        "out_of_sample_decisions": len(late),
        "in_sample_defects": len(early_defects),
        "out_of_sample_defects": len(late_defects),
        "rules": len(splits),
        "gradeable_out_of_sample": len(gradeable),
        "not_gradeable_out_of_sample": len(splits) - len(gradeable),
        "survived": len(survived),
        "flipped": len(gradeable) - len(survived),
        "survival_rate": (
            round(len(survived) / len(gradeable), 4) if gradeable else None
        ),
        "detail": [s.as_dict() for s in splits],
        "failure_cases": [s.as_dict() for s in splits if s.flipped],
        "scope_statement": (
            "Every standing rule is graded twice: on the earlier half of the decisions by ledger "
            "sequence, and on the later half held back. The split is chronological, never random, "
            "because a random split grades a rule on a decision that preceded the ones it was "
            "fitted on. Each half is scored against defects observed WITHIN that half. "
            "NOT CLAIMED: that this is the full self-evolution loop — STANDING_RULES are written "
            "by hand rather than generated from the in-sample window, so this tests whether a "
            "rule's grade generalises, not whether the rule would have been discovered again. "
            "NOT CLAIMED: that a rule which is not gradeable out of sample failed; too few "
            "firings to judge is the absence of a test, and it is counted separately."
        ),
    }


def render(report: dict[str, Any]) -> list[str]:
    rate = report["survival_rate"]
    lines = [
        f"REVIEW OUT-OF-SAMPLE — {report['decisions']} decisions split "
        f"{report['in_sample_decisions']}/{report['out_of_sample_decisions']} chronologically",
        f"  defects: {report['in_sample_defects']} in sample, "
        f"{report['out_of_sample_defects']} held out",
        f"  rules: {report['rules']} graded, {report['gradeable_out_of_sample']} gradeable on the "
        f"held-out half, {report['not_gradeable_out_of_sample']} not",
    ]
    if rate is None:
        lines.append(
            "  No rule could be graded out of sample. That is a fact about how little has "
            "happened on this record, not a result about the rules."
        )
    else:
        lines.append(
            f"  same verdict on both halves: {report['survived']}/"
            f"{report['gradeable_out_of_sample']} ({rate:.0%}); "
            f"{report['flipped']} flipped"
        )
    for row in report["detail"]:
        mark = "  ok  " if row["survived"] else ("  --  " if not row["gradeable_out_of_sample"]
                                                 else "  FLIP")
        lines.append(
            f"{mark} {row['rule']:26} {row['in_sample']:18} -> {row['out_of_sample']}"
        )
    lines.append(
        "  A rule graded on the decisions it was written against describes the past. This asks "
        "whether it also describes the half it never saw."
    )
    return lines


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="held-out test for the self-evolving review")
    parser.add_argument("--share", type=float, default=SPLIT)
    args = parser.parse_args()

    report = run(share=args.share)
    for line in render(report):
        print(line)
    write(REPORT_PATH, report)
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["REPORT_PATH", "ReviewOosError", "RuleSplit", "main", "render", "run", "split_records"]
