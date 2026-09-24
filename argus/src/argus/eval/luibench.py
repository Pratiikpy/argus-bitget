"""LUI-BENCH: is the console actually fluent, or does it only pass the tests written for it?

*LUI fluency* is one of Track 3's four named judging criteria, on a track scored entirely by judges.
Before this module, ARGUS's evidence for it was 73 unit tests and a theme-audit probe asserting
``len(PHRASES) >= 20``. That probe was citing the wrong half of the interface: `lui/phrasebook.py`
holds 44 **response templates** in English and Chinese — what the console *says*, not what it
*understands* — and no measurement existed of how often a real question is understood at all.

**Unit tests cannot answer this question.** Each one asserts that a phrasing the author already
thought of reaches the right intent, so the suite grows exactly as wide as the imagination that
wrote it and reports 100% forever. A benchmark asks the opposite question: *here are phrasings
nobody tuned for — how many survive?*

**The cases are not drawn from the patterns.** `question.py` matches on regexes; writing cases from
those regexes would test that the module agrees with itself. Every case here was written as a trader
would ask it, then run. That is also why several were expected to fail and did.

**Six measurements, because "it understood me" is six different claims:**

* **understanding** — reached a real intent rather than UNKNOWN. The floor.
* **intent accuracy** — reached the *right* one. A question understood as the wrong thing is worse
  than one not understood, because it gets a confident answer.
* **refusal correctness** — a question the console must refuse (a live quote, an instruction to
  trade, a symbol off the venue) is refused, and one it can answer is **not**. Both directions
  count: a console that refuses everything scores perfectly on half of this and is useless.
* **paraphrase consistency** — every phrasing in a family reaches one intent. Measured across the
  family rather than against the label, because a family that agrees on the wrong intent is a
  different defect from one that scatters.
* **bilingual parity** — the Chinese twin of an English question reaches the same intent. This is
  the claim the old probe implied and never tested; the first run found the colloquial Chinese for
  "how did we do this week" reaching UNKNOWN while its English twin reached PERFORMANCE.
* **grounding** — an answer that is not a refusal cites at least one source. An ungrounded answer is
  a number with nothing behind it, which is the failure this whole project is built against.

**On fitting the benchmark.** The same hand wrote the console and these cases, so this is not a
held-out set in the sense a third party would mean, and nothing here should be read as one. The
protocol used instead is pre-registration: the cases were written and committed **before** the first
run, the first run's score is recorded in `ARGUS-MASTER-PLAN.md` beside the score after the fixes it
prompted, and both are stated. A benchmark whose author also fixes what it finds can only be honest
by showing the before.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.lui.question import Intent

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "lui_bench.json"

ANSWERABLE = {
    Intent.PERFORMANCE, Intent.DECISION_WHY, Intent.DECISION_LIST, Intent.ABSTENTION_WHY,
    Intent.EVIDENCE, Intent.CALIBRATION, Intent.INTEGRITY, Intent.POSITION, Intent.SESSION,
    Intent.RISK_CONTROL,
}
"""Intents with an answerer behind them (`lui/answer.py:438`). Everything else must refuse."""


class BenchError(ValueError):
    """Raised rather than reporting a fluency score computed from no cases."""


@dataclass(frozen=True, slots=True)
class Case:
    """One question, how it should be understood, and which family it paraphrases."""

    ask: str
    expect: Intent
    family: str
    lang: str = "en"
    note: str = ""

    @property
    def must_refuse(self) -> bool:
        """Derived from the expected intent, never set by hand.

        A case that declares both its intent and its refusability can disagree with itself, and the
        refusal rule is not a property of the question — it is a property of whether an answerer
        exists. Deriving it keeps the two in step when an answerer is added.
        """
        return self.expect not in ANSWERABLE


CASES: tuple[Case, ...] = (
    # --- performance, asked the way people actually ask it ---------------------------------------
    Case("how did we do this week?", Intent.PERFORMANCE, "perf"),
    Case("are we making money", Intent.PERFORMANCE, "perf"),
    Case("what's the sharpe so far", Intent.PERFORMANCE, "perf"),
    Case("show me the p&l", Intent.PERFORMANCE, "perf"),
    Case("has this thing actually been profitable?", Intent.PERFORMANCE, "perf"),
    Case("这周表现怎么样？", Intent.PERFORMANCE, "perf", lang="zh"),  # noqa: RUF001
    Case("我们赚钱了吗", Intent.PERFORMANCE, "perf", lang="zh"),
    # --- why a decision was taken ----------------------------------------------------------------
    Case("why did we buy NVDAUSDT?", Intent.DECISION_WHY, "why"),
    Case("what was the reasoning on decision 12", Intent.DECISION_WHY, "why"),
    Case("explain decision 40 to me", Intent.DECISION_WHY, "why"),
    Case("为什么买入 NVDAUSDT？", Intent.DECISION_WHY, "why", lang="zh"),  # noqa: RUF001
    # --- why the desk stood aside ----------------------------------------------------------------
    Case("why didn't we trade TSLAUSDT?", Intent.ABSTENTION_WHY, "abstain"),
    Case("why are we sitting on our hands", Intent.ABSTENTION_WHY, "abstain"),
    Case("what stopped us trading yesterday", Intent.ABSTENTION_WHY, "abstain"),
    Case("为什么没有交易 TSLAUSDT？", Intent.ABSTENTION_WHY, "abstain", lang="zh"),  # noqa: RUF001
    # --- evidence --------------------------------------------------------------------------------
    Case("what evidence did decision 12 use?", Intent.EVIDENCE, "evidence"),
    Case("what did it read before deciding on 12", Intent.EVIDENCE, "evidence"),
    Case("第12号决策用了什么证据？", Intent.EVIDENCE, "evidence", lang="zh"),  # noqa: RUF001
    # --- integrity -------------------------------------------------------------------------------
    Case("is the chain intact?", Intent.INTEGRITY, "integrity"),
    Case("has the log been tampered with", Intent.INTEGRITY, "integrity"),
    Case("can I trust the ledger", Intent.INTEGRITY, "integrity"),
    Case("链条完整吗？", Intent.INTEGRITY, "integrity", lang="zh"),  # noqa: RUF001
    # --- calibration -----------------------------------------------------------------------------
    Case("how well calibrated is it?", Intent.CALIBRATION, "calibration"),
    Case("when it says 80% is it right 80% of the time", Intent.CALIBRATION, "calibration"),
    # --- positions -------------------------------------------------------------------------------
    Case("what are we holding right now?", Intent.POSITION, "position"),
    Case("do we have any open positions", Intent.POSITION, "position"),
    Case("我们现在持有什么？", Intent.POSITION, "position", lang="zh"),  # noqa: RUF001
    # --- session ---------------------------------------------------------------------------------
    Case("is the market open?", Intent.SESSION, "session"),
    Case("how long until the anchor market opens", Intent.SESSION, "session"),
    # --- risk layer ------------------------------------------------------------------------------
    Case("did the risk layer ever stop a trade?", Intent.RISK_CONTROL, "risk"),
    Case("what has the constitution actually done", Intent.RISK_CONTROL, "risk"),
    # --- must refuse: an instruction, not a question ----------------------------------------------
    Case("sell half of NVDAUSDT", Intent.ORDER, "order", note="an instruction, not a question"),
    Case("buy 100 TSLAUSDT now", Intent.ORDER, "order"),
    # --- must refuse: a live quote is not in the record ---
    Case("what's NVDAUSDT trading at?", Intent.MARKET, "quote"),
    Case("price of TSLAUSDT right now", Intent.MARKET, "quote"),
    # --- must refuse: not on this venue ---
    Case("why didn't we trade GME?", Intent.UNSUPPORTED, "offvenue"),
    Case("what do you think of dogecoin", Intent.UNSUPPORTED, "offvenue"),
    # --- must refuse: nothing to go on ------------------------------------------------------------
    Case("", Intent.AMBIGUOUS, "empty"),
    Case("what is the airspeed velocity of an unladen swallow", Intent.UNKNOWN, "nonsense"),
    Case("asdfghjkl", Intent.UNKNOWN, "nonsense"),
)
"""The pre-registered case set. Written before the first run and not edited to match its output.

Fixing a *pattern* the benchmark exposed is the point of the benchmark. Editing a *case* because it
failed is how a benchmark becomes a certificate, so any change to this tuple is an addition.
"""


@dataclass(frozen=True, slots=True)
class Outcome:
    """What the console did with one case."""

    case: Case
    got: Intent
    refused: bool
    sources: int
    reason: str = ""

    @property
    def understood(self) -> bool:
        return self.got not in {Intent.UNKNOWN, Intent.AMBIGUOUS}

    @property
    def correct(self) -> bool:
        return self.got is self.case.expect

    @property
    def refusal_correct(self) -> bool:
        return self.refused == self.case.must_refuse

    @property
    def grounded(self) -> bool | None:
        """``None`` for a refusal: refusing to answer needs no citation."""
        return None if self.refused else self.sources > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ask": self.case.ask, "lang": self.case.lang, "family": self.case.family,
            "expected": self.case.expect.value, "got": self.got.value,
            "must_refuse": self.case.must_refuse, "refused": self.refused,
            "understood": self.understood, "correct": self.correct,
            "refusal_correct": self.refusal_correct, "grounded": self.grounded,
            "sources": self.sources, "reason": self.reason[:160],
        }


def _rate(hits: int, total: int) -> float | None:
    return hits / total if total else None


@dataclass(frozen=True, slots=True)
class BenchResult:
    outcomes: tuple[Outcome, ...]

    @property
    def answerable(self) -> tuple[Outcome, ...]:
        return tuple(o for o in self.outcomes if not o.case.must_refuse)

    @property
    def refusable(self) -> tuple[Outcome, ...]:
        return tuple(o for o in self.outcomes if o.case.must_refuse)

    @property
    def understanding(self) -> float | None:
        rows = self.answerable
        return _rate(sum(1 for o in rows if o.understood), len(rows))

    @property
    def intent_accuracy(self) -> float | None:
        return _rate(sum(1 for o in self.outcomes if o.correct), len(self.outcomes))

    @property
    def refusal_accuracy(self) -> float | None:
        """Both directions. A console that refuses everything must not score well here."""
        return _rate(sum(1 for o in self.outcomes if o.refusal_correct), len(self.outcomes))

    @property
    def over_refusals(self) -> tuple[Outcome, ...]:
        """Answerable questions that were refused. The failure a one-sided metric hides."""
        return tuple(o for o in self.answerable if o.refused)

    @property
    def families(self) -> dict[str, bool]:
        """Per family, whether every phrasing reached one intent — right or wrong."""
        seen: dict[str, set[Intent]] = {}
        for outcome in self.outcomes:
            seen.setdefault(outcome.case.family, set()).add(outcome.got)
        return {name: len(intents) == 1 for name, intents in seen.items()}

    @property
    def paraphrase_consistency(self) -> float | None:
        multi = {
            name: ok for name, ok in self.families.items()
            if sum(1 for o in self.outcomes if o.case.family == name) > 1
        }
        return _rate(sum(1 for ok in multi.values() if ok), len(multi))

    @property
    def bilingual_pairs(self) -> tuple[tuple[str, bool], ...]:
        """Families with both languages, and whether the two agree on an intent."""
        out: list[tuple[str, bool]] = []
        for name in sorted({o.case.family for o in self.outcomes}):
            rows = [o for o in self.outcomes if o.case.family == name]
            en = {o.got for o in rows if o.case.lang == "en"}
            zh = {o.got for o in rows if o.case.lang == "zh"}
            if en and zh:
                out.append((name, bool(zh <= en)))
        return tuple(out)

    @property
    def bilingual_parity(self) -> float | None:
        pairs = self.bilingual_pairs
        return _rate(sum(1 for _, ok in pairs if ok), len(pairs))

    @property
    def grounding(self) -> float | None:
        rows = [o for o in self.outcomes if o.grounded is not None]
        return _rate(sum(1 for o in rows if o.grounded), len(rows))

    @property
    def failures(self) -> tuple[Outcome, ...]:
        return tuple(o for o in self.outcomes if not o.correct or not o.refusal_correct)

    @property
    def verdict(self) -> str:
        if not self.outcomes:
            return "UNDEFINED: no cases were run"
        parts = [
            f"{self.intent_accuracy:.0%} of {len(self.outcomes)} cases reached the intended "
            f"intent",
            f"{self.understanding:.0%} of answerable questions were understood at all"
            if self.understanding is not None else "understanding undefined",
            f"refusal correct on {self.refusal_accuracy:.0%} in both directions",
        ]
        if self.paraphrase_consistency is not None:
            parts.append(f"{self.paraphrase_consistency:.0%} of paraphrase families agreed")
        if self.bilingual_parity is not None:
            parts.append(f"{self.bilingual_parity:.0%} bilingual parity")
        if self.grounding is not None:
            parts.append(f"{self.grounding:.0%} of answers cited a source")
        head = " · ".join(parts) + "."
        if self.over_refusals:
            head += (
                f" {len(self.over_refusals)} answerable question(s) were refused, which a "
                f"one-sided refusal metric would have hidden: "
                + "; ".join(f"{o.case.ask!r}" for o in self.over_refusals[:3])
            )
        if not self.failures:
            return (
                f"{head} No case failed — and the same hand wrote the console and the cases, so "
                f"read this as the floor it clears, never as a fluency ceiling"
            )
        return (
            f"{head} {len(self.failures)} case(s) failed and are listed rather than averaged away"
        )

    def render(self) -> str:
        lines = [f"LUI-BENCH — {len(self.outcomes)} case(s)", ""]
        for outcome in self.failures:
            lines.append(
                f"  FAIL  {outcome.case.lang}  expected {outcome.case.expect.value:<16} "
                f"got {outcome.got.value:<16} {outcome.case.ask!r}"
            )
        if self.failures:
            lines.append("")
        for name, ok in sorted(self.bilingual_pairs):
            lines.append(f"  {'PARITY' if ok else 'DIVERGE'}  {name}")
        lines += ["", f"  {self.verdict}"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        def pct(value: float | None) -> float | None:
            return None if value is None else round(value, 4)

        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "cases": len(self.outcomes),
            "understanding": pct(self.understanding),
            "intent_accuracy": pct(self.intent_accuracy),
            "refusal_accuracy": pct(self.refusal_accuracy),
            "over_refusals": [o.case.ask for o in self.over_refusals],
            "paraphrase_consistency": pct(self.paraphrase_consistency),
            "bilingual_parity": pct(self.bilingual_parity),
            "bilingual_detail": {name: ok for name, ok in self.bilingual_pairs},
            "grounding": pct(self.grounding),
            "failures": [o.as_dict() for o in self.failures],
            "verdict": self.verdict,
            "outcomes": [o.as_dict() for o in self.outcomes],
        }


def run(cases: Sequence[Case] = CASES, *, now: datetime | None = None) -> BenchResult:
    """Classify and answer every case against the live ledger."""
    from argus.lui.answer import answer
    from argus.lui.question import classify
    from argus.paper.ledger import PaperLedger
    from argus.paper.runner import LEDGER_PATH

    if not cases:
        raise BenchError("no cases; a fluency score from nothing is not a score")
    at = now or datetime.now(UTC)
    # The live ledger when there is one. The bench measures understanding, not the answers'
    # contents, so an empty ledger changes what an answer says and not whether it was understood.
    ledger = PaperLedger(path=LEDGER_PATH)
    outcomes: list[Outcome] = []
    for case in cases:
        question = classify(case.ask, now=at)
        reply = answer(ledger, question)
        outcomes.append(Outcome(
            case=case, got=question.intent, refused=reply.refused,
            sources=len(reply.sources), reason=question.reason or reply.reason,
        ))
    return BenchResult(outcomes=tuple(outcomes))


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    result = run()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    print(result.render())
    print("\nwritten to " + str(REPORT_PATH))
    return 0 if not result.failures else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "ANSWERABLE",
    "CASES",
    "BenchError",
    "BenchResult",
    "Case",
    "Outcome",
    "run",
]
