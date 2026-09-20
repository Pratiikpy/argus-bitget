"""OBLIQUE-BENCH: what the console understands **with no model key**, which is what a judge meets.

`deploy/api/index.py` states the deployment posture plainly: *"No key is deployed here: a key in a
public serverless bundle is a key that has been published."* So the hosted console answers from
`lui/question.py`'s patterns alone. The router in `lui/router.py` returns ``None`` without
credentials and never runs. **Every measurement of the router is a measurement of a path a judge
will not touch.**

That makes the deterministic layer's coverage the number that matters for Track 3's *"LUI fluency"*,
and two existing figures disagree about it:

* `tests/test_phrasings.py` asserts **>=95%** of an 85-phrasing corpus is understood with no
  model.
* `eval/routerbench.py` found **16 of 40** deliberately oblique phrasings understood — **40%**.

Neither is wrong. They measure different corpora, and the difference between them *is* the finding:
**the 95% holds for phrasings close to how the patterns were written, and falls to 40% for phrasings
written to avoid thinking about a parser at all.** Quoting only the first would overstate what a
judge experiences.

**Why this module exists rather than a wider pattern list and a re-run of the old number.** Once the
patterns are widened against a corpus, that corpus stops measuring anything: it becomes the training
set. So there are two corpora here and the distinction is load-bearing:

* :data:`TUNED` — the oblique cases the patterns were deliberately widened to catch. After the
  widening this measures **fit**, not fluency, and is labelled as such. It is kept because a
  training score that is *not* near-perfect means the widening failed on its own terms.
* :data:`HELDOUT` — different phrasings of the **same linguistic categories**, written before the
  patterns were touched and **not consulted while editing them**. This is the number that means
  something.

The gap between the two is a direct estimate of how much the widening generalised versus how much it
memorised. A large gap is a real finding about our own work and is reported as one.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.lui.question import Intent, classify

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "oblique_bench.json"


class ObliqueBenchError(ValueError):
    """Raised rather than reporting fluency computed from an empty corpus."""


@dataclass(frozen=True, slots=True)
class Case:
    """One obliquely-phrased question and the intent the console should reach without a model."""

    ask: str
    expect: Intent
    category: str
    """The linguistic move being tested — an idiom, a why-synonym, an ellipsis. Patterns are widened
    against *categories*; memorising the strings would pass this bench and help nobody."""

    lang: str = "en"


TUNED: tuple[Case, ...] = (
    # --- "how come" as a why-synonym ------------------------------------------------------------
    Case("how come you sat on your hands yesterday", Intent.ABSTENTION_WHY, "how-come"),
    Case("nothing went through today, how come", Intent.ABSTENTION_WHY, "how-come"),
    # --- abstention idiom -------------------------------------------------------------------------
    Case("you keep passing, what is stopping you", Intent.ABSTENTION_WHY, "abstain-idiom"),
    # --- performance idiom ------------------------------------------------------------------------
    Case("been a good month or are we bleeding", Intent.PERFORMANCE, "perf-idiom"),
    Case("so what is the damage", Intent.PERFORMANCE, "perf-idiom"),
    Case("总体来说亏了还是赚了", Intent.PERFORMANCE, "perf-idiom", lang="zh"),
    # --- walk/talk me through ---------------------------------------------------------------------
    Case("talk me through number 17", Intent.DECISION_WHY, "walk-through"),
    Case("what made you pull the trigger on TSLAUSDT", Intent.DECISION_WHY, "trade-idiom"),
    # --- the record, asked flatly -----------------------------------------------------------------
    Case("just give me the log", Intent.DECISION_LIST, "record-idiom"),
    # --- evidence idiom ---------------------------------------------------------------------------
    Case("what were you reading before the AAPLUSDT call", Intent.EVIDENCE,
         "evidence-idiom"),
    Case("where did the AMZNUSDT number come from", Intent.EVIDENCE,
         "evidence-idiom"),
    # --- calibration idiom ------------------------------------------------------------------------
    Case("when you say you are sure, are you", Intent.CALIBRATION,
         "calibration-idiom"),
    Case("does your confidence mean anything", Intent.CALIBRATION,
         "calibration-idiom"),
    # --- position idiom ---------------------------------------------------------------------------
    Case("anything still open", Intent.POSITION, "position-idiom"),
    Case("what is live right now", Intent.POSITION, "position-idiom"),
    # --- session idiom ----------------------------------------------------------------------------
    Case("is the market even open", Intent.SESSION, "session-idiom"),
    Case("what part of the day is it for you", Intent.SESSION, "session-idiom"),
)
"""The cases the patterns were widened against. **After the widening this measures fit.**"""


HELDOUT: tuple[Case, ...] = (
    # --- "how come", different subjects -----------------------------------------------------------
    Case("how come you stayed flat all week", Intent.ABSTENTION_WHY, "how-come"),
    Case("how come nothing got filled", Intent.ABSTENTION_WHY, "how-come"),
    # --- abstention idiom, different figures ------------------------------------------------------
    Case("you are sitting this one out again, what for", Intent.ABSTENTION_WHY,
         "abstain-idiom"),
    Case("another quiet day, what held you back", Intent.ABSTENTION_WHY,
         "abstain-idiom"),
    # --- performance idiom, different figures -----------------------------------------------------
    Case("are we in the red or the black", Intent.PERFORMANCE, "perf-idiom"),
    Case("how bad is it", Intent.PERFORMANCE, "perf-idiom"),
    Case("did we come out ahead", Intent.PERFORMANCE, "perf-idiom"),
    Case("这个月是赔了还是挣了", Intent.PERFORMANCE, "perf-idiom", lang="zh"),
    # --- walk/talk me through, different verbs ----------------------------------------------------
    Case("take me through decision 9", Intent.DECISION_WHY, "walk-through"),
    Case("run me through the MSFTUSDT one", Intent.DECISION_WHY, "walk-through"),
    Case("what got you into GOOGLUSDT", Intent.DECISION_WHY, "trade-idiom"),
    # --- the record, asked flatly -----------------------------------------------------------------
    Case("show me the book", Intent.DECISION_LIST, "record-idiom"),
    Case("everything you did, please", Intent.DECISION_LIST, "record-idiom"),
    # --- evidence idiom, different figures --------------------------------------------------------
    Case("what were you looking at for METAUSDT", Intent.EVIDENCE,
         "evidence-idiom"),
    Case("where does the NVDAUSDT figure come from", Intent.EVIDENCE,
         "evidence-idiom"),
    # --- calibration idiom, different figures -----------------------------------------------------
    Case("should i believe you when you sound certain", Intent.CALIBRATION,
         "calibration-idiom"),
    Case("is your confidence worth anything", Intent.CALIBRATION,
         "calibration-idiom"),
    # --- position idiom, different figures --------------------------------------------------------
    Case("anything on the books right now", Intent.POSITION, "position-idiom"),
    Case("are we holding anything", Intent.POSITION, "position-idiom"),
    # --- session idiom, different figures ---------------------------------------------------------
    Case("has the bell gone yet", Intent.SESSION, "session-idiom"),
    Case("where are we in the trading day", Intent.SESSION, "session-idiom"),
)
"""Different phrasings of the same categories, written **before** the patterns were touched.

This is the number that means something. If it tracks :data:`TUNED`, the widening generalised. If it
lags badly, the patterns learned seventeen sentences.
"""


@dataclass(frozen=True, slots=True)
class Outcome:
    case: Case
    reached: Intent

    @property
    def understood(self) -> bool:
        """Did the deterministic layer reach *a* real intent rather than giving up?"""
        return self.reached not in (Intent.UNKNOWN, Intent.AMBIGUOUS)

    @property
    def correct(self) -> bool:
        """Did it reach the *right* one? Understanding the wrong thing is worse than refusing."""
        return self.reached is self.case.expect

    def as_dict(self) -> dict[str, Any]:
        return {
            "ask": self.case.ask,
            "category": self.case.category,
            "lang": self.case.lang,
            "expect": str(self.case.expect),
            "reached": str(self.reached),
            "understood": self.understood,
            "correct": self.correct,
        }


@dataclass(frozen=True, slots=True)
class CorpusResult:
    name: str
    outcomes: tuple[Outcome, ...]

    @property
    def understood(self) -> float | None:
        return None if not self.outcomes else sum(
            1 for o in self.outcomes if o.understood
        ) / len(self.outcomes)

    @property
    def correct(self) -> float | None:
        return None if not self.outcomes else sum(
            1 for o in self.outcomes if o.correct
        ) / len(self.outcomes)

    @property
    def misses(self) -> tuple[Outcome, ...]:
        return tuple(o for o in self.outcomes if not o.correct)

    def by_category(self) -> dict[str, tuple[int, int]]:
        """``category -> (correct, total)``, so a category that never generalised is visible."""
        out: dict[str, list[int]] = {}
        for o in self.outcomes:
            row = out.setdefault(o.case.category, [0, 0])
            row[1] += 1
            if o.correct:
                row[0] += 1
        return {k: (v[0], v[1]) for k, v in out.items()}

    def as_dict(self) -> dict[str, Any]:
        def pct(v: float | None) -> float | None:
            return None if v is None else round(v * 100.0, 1)

        return {
            "corpus": self.name,
            "cases": len(self.outcomes),
            "understood_pct": pct(self.understood),
            "correct_pct": pct(self.correct),
            "by_category": {k: {"correct": c, "total": t}
                            for k, (c, t) in
                            self.by_category().items()},
            "misses": [o.as_dict() for o in self.misses],
        }


@dataclass(frozen=True, slots=True)
class BenchResult:
    tuned: CorpusResult
    heldout: CorpusResult
    as_of: datetime

    @property
    def generalisation_gap(self) -> float | None:
        """Tuned accuracy minus held-out accuracy.

        The honest estimate of how much the widening memorised. Near zero means the categories were
        real; a wide gap means seventeen sentences were learned and nothing else.
        """
        if self.tuned.correct is None or self.heldout.correct is None:
            return None
        return self.tuned.correct - self.heldout.correct

    @property
    def verdict(self) -> str:
        tuned, held = self.tuned.correct, self.heldout.correct
        if tuned is None or held is None:
            return (
                "No cases, so nothing is measured. A fluency figure over an empty "
                "corpus is not a low one."
            )
        gap = self.generalisation_gap or 0.0
        head = (
            f"Deterministic layer, no model key — the state a judge meets: **{held:.0%} correct on "
            f"the held-out corpus** ({len(self.heldout.outcomes)} cases) against {tuned:.0%} "
            f"on the corpus the patterns were widened for ({len(self.tuned.outcomes)} cases)."
        )
        if gap <= 0.10:
            return head + (
                f" The gap is {gap:+.0%}, so the widening generalised: it caught linguistic "
                f"categories rather than memorising sentences."
            )
        return head + (
            f" **The gap is {gap:+.0%}, which is a finding against our own work** — the patterns "
            f"fit the cases they were shown far better than new phrasings of the same categories. "
            f"The held-out number is the one to quote."
        )

    def render(self) -> str:
        lines = [
            "OBLIQUE-BENCH — the deterministic layer, which is what the hosted console runs",
            "",
            "  corpus     cases  understood  correct",
        ]
        for corpus in (self.tuned, self.heldout):
            def show(v: float | None) -> str:
                return "      —" if v is None else f"{v * 100.0:6.1f}%"

            lines.append(
                f"  {corpus.name:10s} {len(corpus.outcomes):5d}  {show(corpus.understood)}"
                f"  {show(corpus.correct)}"
            )
        lines += ["", f"  {self.verdict}"]
        if self.heldout.misses:
            lines += ["", "  held-out misses:"]
            lines += [
                f"    {o.case.category:20s} {o.case.ask[:46]:48s} -> {o.reached}"
                for o in self.heldout.misses
            ]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "note": (
                "The hosted console runs without a model key by design, so this measures the path "
                "a judge actually meets. TUNED is the corpus the patterns were widened against and "
                "measures fit; HELDOUT was written before the patterns were touched and measures "
                "fluency."
            ),
            "tuned": self.tuned.as_dict(),
            "heldout": self.heldout.as_dict(),
            "generalisation_gap_pct": (
                None if self.generalisation_gap is None
                else round(self.generalisation_gap * 100.0, 1)
            ),
            "verdict": self.verdict,
        }


def _run(name: str, cases: Sequence[Case], *, now: datetime) -> CorpusResult:
    return CorpusResult(
        name=name,
        outcomes=tuple(Outcome(c, classify(c.ask, now=now).intent) for c in cases),
    )


def run(*, now: datetime | None = None) -> BenchResult:
    """Classify both corpora with the deterministic layer only. No model is consulted."""
    if not TUNED or not HELDOUT:
        raise ObliqueBenchError("both corpora must carry cases for the gap to mean anything")
    at = now or datetime.now(UTC)
    return BenchResult(
        tuned=_run("tuned", TUNED, now=at),
        heldout=_run("held-out", HELDOUT, now=at),
        as_of=at,
    )


def main() -> int:  # pragma: no cover - CLI
    result = run()
    print(result.render())
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
