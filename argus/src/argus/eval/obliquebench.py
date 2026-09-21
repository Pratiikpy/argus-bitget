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

**BURNED FOR TUNING ON 2026-09-21, and kept only as a historical figure.** Its per-case misses were
printed to a terminal while the deterministic layer was being widened, which is exactly the
condition this corpus existed to avoid. Nothing in `lui/question.py` was copied from these
sentences, but "I did not consciously use it" is not a property anybody can check, and a held-out
corpus whose answers the author has read is no longer held out. Quoting it now would be quoting a
training score.

:data:`FRESH` replaces it. This stays so the record shows what was measured when.
"""


FRESH: tuple[Case, ...] = (
    # --- abstention -------------------------------------------------------------------------
    Case("I saw the screen stay blank for hours - why didn't we take any action?",
         Intent.ABSTENTION_WHY, "abstain-idiom"),
    Case("Had any thoughts on why we didn't touch the trade on Tuesday?",
         Intent.ABSTENTION_WHY, "abstain-idiom"),
    Case("It looks like the desk took a day off - what's the deal?",
         Intent.ABSTENTION_WHY, "abstain-idiom"),
    # --- performance ------------------------------------------------------------------------
    Case("最近账户的盈亏大致如何？", Intent.PERFORMANCE, "perf-idiom", lang="zh"),
    Case("Gives me rundown on how the portfolio fared last month.",
         Intent.PERFORMANCE, "perf-idiom"),
    Case("过去一个季度的账户表现到底好吗？", Intent.PERFORMANCE, "perf-idiom", lang="zh"),
    Case("What's the net result on the ledger after the last trade wave?",
         Intent.PERFORMANCE, "perf-idiom"),
    # --- one decision explained ---------------------------------------------------------------
    Case("Why did you decide to lean into TechCorp last Thursday?",
         Intent.DECISION_WHY, "walk-through"),
    Case("我好奇昨天对XXX的卖空决定为什么被认为是“最优”？",
         Intent.DECISION_WHY, "walk-through", lang="zh"),
    Case("What led you to hit the buy on the T-Bill today?", Intent.DECISION_WHY, "trade-idiom"),
    Case("昨晚的决策为什么偏向ETF之类的？", Intent.DECISION_WHY, "walk-through", lang="zh"),
    # --- the whole record ---------------------------------------------------------------------
    Case("Lay out all the moves the desk made over the past year.",
         Intent.DECISION_LIST, "record-idiom"),
    Case("请列出今日所有的操作记录。", Intent.DECISION_LIST, "record-idiom", lang="zh"),
    Case("Show me the entire roster of trades the desk went through today.",
         Intent.DECISION_LIST, "record-idiom"),
    Case("能否把前几天所有决定都做个清单？", Intent.DECISION_LIST, "record-idiom", lang="zh"),
    Case("Deliver the full log of all past decisions from yesterday to now.",
         Intent.DECISION_LIST, "record-idiom"),
    # --- evidence -----------------------------------------------------------------------------
    Case("What data backlit the price jump of XYZ this morning?",
         Intent.EVIDENCE, "evidence-idiom"),
    Case("我要知道你在做那笔交易时参考了哪些信息来源。",
         Intent.EVIDENCE, "evidence-idiom", lang="zh"),
    Case("Did you pull any fundamental reports to support yesterday's long?",
         Intent.EVIDENCE, "evidence-idiom"),
    Case("你在决定A股票时主要看了什么指标？", Intent.EVIDENCE, "evidence-idiom", lang="zh"),
    Case("Mention the sources you consulted before placing that short.",
         Intent.EVIDENCE, "evidence-idiom"),
    # --- calibration --------------------------------------------------------------------------
    Case("Is your confidence percentile for the market steady, or does it shift?",
         Intent.CALIBRATION, "calibration-idiom"),
    Case("我的模型准确率实时靠谱吗？", Intent.CALIBRATION, "calibration-idiom", lang="zh"),
    Case("How reliable are the confidence levels you glimpse during the day?",
         Intent.CALIBRATION, "calibration-idiom"),
    Case("Tell me if the confidence indicator is worth trusting long term.",
         Intent.CALIBRATION, "calibration-idiom"),
    # --- open positions -----------------------------------------------------------------------
    Case("现在的持仓到底有多少？", Intent.POSITION, "position-idiom", lang="zh"),
    Case("Which holdings are live at the moment?", Intent.POSITION, "position-idiom"),
    Case("你们目前持有的资产各是多少？", Intent.POSITION, "position-idiom", lang="zh"),
    Case("Show me the open positions currently in play.", Intent.POSITION, "position-idiom"),
    # --- the trading day ----------------------------------------------------------------------
    Case("Is the bell ringing yet, or should I wait a bit?", Intent.SESSION, "session-idiom"),
    Case("请问现在主场交易时间是几点到几点？", Intent.SESSION, "session-idiom", lang="zh"),
    Case("Know if the market's still on the grind or closed?", Intent.SESSION, "session-idiom"),
    Case("现在正处于交易日的哪一节？", Intent.SESSION, "session-idiom", lang="zh"),
    Case("Give me the break-down of the day's trading session.",
         Intent.SESSION, "session-idiom"),
)
"""**Authored by a model that has never seen `lui/question.py`**, which is what makes it held out.

Generated on 2026-09-21 by ``openai/gpt-oss-20b`` through NVIDIA's free API, asked for oblique,
colloquial phrasings per intent with a fixed share in Simplified Chinese. Raw output is kept at
``data/fresh_gpt-oss-20b.json`` so the curation below can be checked against it.

**Six of the forty were dropped, every one for linguistic validity and none for how the classifier
answers it.** The classifier was not run until after this tuple was frozen, because curating on
"does our parser get it" is how a held-out corpus is quietly turned into a passing one:

* ``我想知道为什么今天准时毫无交易`` and ``昨天午盘的交易一枝不动，这先是怎么回事`` — ``准时``
  is misused and ``一枝不动``/``这先是`` are not Chinese; a native speaker would not type either.
* ``How far popped we are from our yearly target`` — not a grammatical English sentence.
* ``Tell me why the desk tweeted a short on ABC`` — a desk does not tweet an order; the verb makes
  the intent unrecoverable.
* ``你对未来的准确度预估刚好还是糟糕`` — ``刚好`` is misused, leaving no readable question.
* ``What sits in the cabinet right now`` — ``cabinet`` is not a trading term in any register, so
  no parser could fairly be expected to read it as a positions question.

Sentences that are merely *awkward* were kept — "Gives me rundown", "backlit", "glimpse" — because
real traders type badly and a corpus of clean prose would flatter the patterns.
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
    fresh: CorpusResult
    as_of: datetime

    @property
    def generalisation_gap(self) -> float | None:
        """Tuned accuracy minus **fresh** accuracy.

        The honest estimate of how much the widening memorised. Near zero means the categories were
        real; a wide gap means a list of sentences was learned and nothing else.

        Measured against :data:`FRESH` rather than :data:`HELDOUT` since 2026-09-21: held-out was
        read during a debugging session and a corpus whose answers the author has seen cannot
        estimate generalisation, however carefully it was then avoided.
        """
        if self.tuned.correct is None or self.fresh.correct is None:
            return None
        return self.tuned.correct - self.fresh.correct

    @property
    def verdict(self) -> str:
        tuned, fresh = self.tuned.correct, self.fresh.correct
        if tuned is None or fresh is None:
            return (
                "No cases, so nothing is measured. A fluency figure over an empty "
                "corpus is not a low one."
            )
        gap = self.generalisation_gap or 0.0
        head = (
            f"Deterministic layer, no model key — the state a judge meets: **{fresh:.0%} correct "
            f"on the fresh corpus** ({len(self.fresh.outcomes)} cases written by a model that has "
            f"never seen the patterns) against {tuned:.0%} on the corpus the patterns were widened "
            f"for ({len(self.tuned.outcomes)} cases)."
        )
        if gap <= 0.10:
            return head + (
                f" The gap is {gap:+.0%}, so the widening generalised: it caught linguistic "
                f"categories rather than memorising sentences."
            )
        return head + (
            f" **The gap is {gap:+.0%}, which is a finding against our own work** — the patterns "
            f"fit the cases they were shown far better than new phrasings of the same categories. "
            f"The fresh number is the one to quote."
        )

    def render(self) -> str:
        lines = [
            "OBLIQUE-BENCH — the deterministic layer, which is what the hosted console runs",
            "",
            "  corpus     cases  understood  correct",
        ]
        for corpus in (self.tuned, self.heldout, self.fresh):
            def show(v: float | None) -> str:
                return "      —" if v is None else f"{v * 100.0:6.1f}%"

            lines.append(
                f"  {corpus.name:10s} {len(corpus.outcomes):5d}  {show(corpus.understood)}"
                f"  {show(corpus.correct)}"
            )
        lines += [
            "",
            "  'held-out' was read while the patterns were being widened on 2026-09-21 and is a "
            "training score from that day on. It is printed, not quoted.",
            "",
            f"  {self.verdict}",
            "",
            "  SCOPE, since 2026-09-21: this measures the PATTERN LAYER ALONE, which is no longer "
            "the whole console. The +76% gap here is what motivated `lui/ngram.py`; the figure "
            "for what a judge now meets is in `eval/ngrambench.py`, which scores the cascade at "
            "80.9% on a corpus of 351. This file is the before, not the after.",
        ]
        if self.fresh.misses:
            lines += ["", "  fresh-corpus misses:"]
            lines += [
                f"    {o.case.category:20s} {o.case.ask[:44]:46s} -> {o.reached}"
                for o in self.fresh.misses
            ]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "measures": "the deterministic pattern layer alone — NOT the deployed console",
            "superseded_by": "data/ngram_bench.json",
            "note": (
                "TUNED is the corpus the patterns were widened against and measures fit. HELDOUT "
                "was burned on 2026-09-21 — its misses were read during the widening — and is "
                "retained only as a historical figure. FRESH was authored by openai/gpt-oss-20b, "
                "which has never seen lui/question.py. The +76% gap between TUNED and FRESH is "
                "what motivated lui/ngram.py: the patterns memorise and do not generalise. Since "
                "that model was added the hosted console is a cascade, not these patterns alone, "
                "so the figure describing what a judge meets lives in data/ngram_bench.json "
                "(80.9% on 351 questions). This artefact is the before."
            ),
            "tuned": self.tuned.as_dict(),
            "heldout": self.heldout.as_dict(),
            "heldout_status": "BURNED 2026-09-21 — read during tuning; not a held-out score",
            "fresh": self.fresh.as_dict(),
            "fresh_author": "openai/gpt-oss-20b via NVIDIA free API",
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
    """Classify all three corpora with the deterministic layer only. No model is consulted."""
    if not TUNED or not HELDOUT or not FRESH:
        raise ObliqueBenchError("every corpus must carry cases for the gap to mean anything")
    at = now or datetime.now(UTC)
    return BenchResult(
        tuned=_run("tuned", TUNED, now=at),
        heldout=_run("burned", HELDOUT, now=at),
        fresh=_run("fresh", FRESH, now=at),
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
