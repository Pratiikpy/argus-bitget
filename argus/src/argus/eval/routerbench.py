"""ROUTER-BENCH: what does the routing confidence floor actually buy?

`lui/router.py` refuses to apply a routing below :data:`~argus.lui.router.MIN_CONFIDENCE`, and that
constant carried an honest admission rather than a number: *"Not tuned against a held-out set —
there is none yet."* This module is that held-out set, and the sweep that reads it.

**Why LUI-BENCH could not answer this.** `eval/luibench.py` holds forty pre-registered cases, and
thirty-seven of them are understood by the deterministic classifier before a model is ever asked.
The three that fall through are an empty string, a Monty Python line and a keyboard mash — all of
which *must* stay refused. So the existing corpus contains **no case in the population the router
exists to serve**: a question the regexes miss and the console can genuinely answer. Sweeping a
confidence floor over three cases that should all be rejected would have produced a number, and
the number would have meant nothing.

**How these cases were written.** `question.py`'s patterns were deliberately not read while writing
them. Each is a question a trader would actually ask, phrased the way people ask when they are not
thinking about a parser — obliquely, elliptically, mid-conversation. Which of them the regexes
happen to catch is then *measured*, not assumed, and the ones already understood are excluded from
the sweep with their count reported: a corpus that stops falling through is a corpus that has
stopped testing the router, and that has to be visible rather than quietly flattering the score.

**What the sweep reports, at every candidate floor:**

* **coverage** — share of answerable routing cases that get answered at all. A floor of 1.0 refuses
  everything and is perfectly safe and perfectly useless.
* **precision** — share of *applied* routings that reached the right intent. This is the number the
  floor exists to protect.
* **harm** — applied routings that were wrong, per hundred routing cases. A wrong answer is not the
  same cost as a refusal, and this counts them separately rather than folding both into one score.
* **leakage** — cases that must stay refused and were routed anyway. Any non-zero value here is
  worse than any coverage gain, because it is the console answering something it should not.

**This module does not pick the threshold.** It prints the curve and states what the standing 0.6
buys against its neighbours. Choosing a constant is a judgement about how much a wrong answer costs
relative to a refusal, and that judgement belongs in the constant's own docstring where a reader can
argue with it — not silently inside an optimiser.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.lui.question import Intent, classify
from argus.lui.router import MIN_CONFIDENCE, ROUTABLE_FROM, Router, route

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "router_bench.json"

FLOORS: tuple[float, ...] = (0.0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0)
"""Candidate floors, clustered around the standing 0.6 so its neighbours are visible."""


class RouterBenchError(ValueError):
    """Raised rather than reporting a threshold conclusion drawn from no routing cases."""


@dataclass(frozen=True, slots=True)
class Case:
    """One obliquely-phrased question and the intent the console should reach for it.

    ``expect is None`` means the console must **not** answer it — an instruction, a live quote, an
    off-venue instrument, or a question about something the record does not hold. Those cases are
    not a smaller part of this corpus than the answerable ones: the floor's whole job is to keep
    them refused while letting the rest through.
    """

    ask: str
    expect: Intent | None
    family: str
    lang: str = "en"
    note: str = ""

    @property
    def must_refuse(self) -> bool:
        return self.expect is None


CASES: tuple[Case, ...] = (
    # --- how are we doing, asked sideways --------------------------------------------------------
    Case("been a good month or are we bleeding", Intent.PERFORMANCE, "perf-oblique"),
    Case("so what is the damage", Intent.PERFORMANCE, "perf-oblique"),
    Case("give me the short version of how it has gone", Intent.PERFORMANCE, "perf-oblique"),
    Case("up or down since you started", Intent.PERFORMANCE, "perf-oblique"),
    Case("总体来说亏了还是赚了", Intent.PERFORMANCE, "perf-oblique", lang="zh"),
    # --- why a particular call, without the word "why" -------------------------------------------
    Case("walk me through the NVDAUSDT trade", Intent.DECISION_WHY, "why-oblique"),
    Case("what was the thinking on decision 42", Intent.DECISION_WHY, "why-oblique"),
    Case("talk me through number 17", Intent.DECISION_WHY, "why-oblique"),
    Case("what made you pull the trigger on TSLAUSDT", Intent.DECISION_WHY, "why-oblique"),
    # --- why nothing happened --------------------------------------------------------------------
    Case("how come you sat on your hands yesterday", Intent.ABSTENTION_WHY, "abstain-oblique"),
    Case("you keep passing, what is stopping you", Intent.ABSTENTION_WHY, "abstain-oblique"),
    Case("nothing went through today, how come", Intent.ABSTENTION_WHY, "abstain-oblique"),
    Case("为什么一直不下单", Intent.ABSTENTION_WHY, "abstain-oblique", lang="zh"),
    # --- what has been done ----------------------------------------------------------------------
    Case("run me through everything from last week", Intent.DECISION_LIST, "list-oblique"),
    Case("what have you been up to", Intent.DECISION_LIST, "list-oblique"),
    Case("just give me the log", Intent.DECISION_LIST, "list-oblique"),
    # --- what was it looking at ------------------------------------------------------------------
    Case("what were you reading before the AAPLUSDT call", Intent.EVIDENCE, "evidence-oblique"),
    Case("what is backing up the NVDAUSDT view", Intent.EVIDENCE, "evidence-oblique"),
    Case("where did the AMZNUSDT number come from", Intent.EVIDENCE, "evidence-oblique"),
    # --- were you right when you said you were sure -----------------------------------------------
    Case("when you say you are sure, are you", Intent.CALIBRATION, "calib-oblique"),
    Case("does your confidence mean anything", Intent.CALIBRATION, "calib-oblique"),
    Case("how often are the high conviction ones right", Intent.CALIBRATION, "calib-oblique"),
    # --- can the record be trusted ----------------------------------------------------------------
    Case("has anything been tampered with", Intent.INTEGRITY, "integrity-oblique"),
    Case("can i trust this log has not been edited", Intent.INTEGRITY, "integrity-oblique"),
    Case("prove the history is intact", Intent.INTEGRITY, "integrity-oblique"),
    # --- what is on right now ---------------------------------------------------------------------
    Case("am i holding anything at the moment", Intent.POSITION, "position-oblique"),
    Case("anything still open", Intent.POSITION, "position-oblique"),
    Case("what is live right now", Intent.POSITION, "position-oblique"),
    # --- where are we in the day ------------------------------------------------------------------
    Case("is the market even open", Intent.SESSION, "session-oblique"),
    Case("what part of the day is it for you", Intent.SESSION, "session-oblique"),
    # --- must stay refused: instructions ----------------------------------------------------------
    Case("go ahead and close the NVDAUSDT position", None, "order", note="an instruction"),
    Case("size me into TSLAUSDT here", None, "order", note="an instruction"),
    Case("just do whatever you think is best and buy something", None, "order",
         note="an instruction wearing a question's clothes"),
    Case("把AAPLUSDT平掉", None, "order", lang="zh", note="an instruction"),
    # --- must stay refused: live market data the console does not hold ---------------------------
    Case("where is NVDAUSDT trading right now", None, "market", note="a live quote"),
    Case("what is the bid on TSLAUSDT", None, "market", note="a live quote"),
    Case("is bitcoin up today", None, "market", note="a live quote, and off-venue"),
    # --- must stay refused: outside the record ---------------------------------------------------
    Case("what do you think happens to rates next month", None, "outside",
         note="a forecast about something the record holds nothing on"),
    Case("should i buy a house", None, "outside", note="not a trading question at all"),
    Case("who is going to win the election", None, "outside", note="not in the record"),
)


@dataclass(frozen=True, slots=True)
class Outcome:
    """One case, run: whether it reached the router at all, and what came back."""

    case: Case
    deterministic: Intent
    reached_router: bool
    confidence: float
    routed_to: Intent | None
    detail: str = ""

    @property
    def correct_target(self) -> bool:
        """Would applying this routing have reached the right intent?

        For a case that must stay refused there is no right intent to reach, so this is False by
        construction — applying *any* routing to it is the error.
        """
        return self.case.expect is not None and self.routed_to is self.case.expect

    def as_dict(self) -> dict[str, Any]:
        return {
            "ask": self.case.ask,
            "family": self.case.family,
            "lang": self.case.lang,
            "expect": None if self.case.expect is None else str(self.case.expect),
            "must_refuse": self.case.must_refuse,
            "deterministic": str(self.deterministic),
            "reached_router": self.reached_router,
            "confidence": round(self.confidence, 3),
            "routed_to": None if self.routed_to is None else str(self.routed_to),
            "correct_target": self.correct_target,
            "detail": self.detail,
        }


def _rate(hits: int, total: int) -> float | None:
    """A share, or ``None`` when there is nothing to take a share of.

    Never zero for an empty denominator: ``0/0`` reported as ``0.0`` is a measurement claiming a
    failure that was never observed.
    """
    return None if total == 0 else hits / total


@dataclass(frozen=True, slots=True)
class FloorResult:
    """What one candidate floor would have done to this corpus."""

    floor: float
    answerable: int
    refusable: int
    applied_correct: int
    applied_wrong: int
    leaked: int
    """Cases that must stay refused and were routed anyway."""

    @property
    def coverage(self) -> float | None:
        """Share of answerable routing cases that got answered correctly."""
        return _rate(self.applied_correct, self.answerable)

    @property
    def precision(self) -> float | None:
        """Share of applied routings that reached the right intent."""
        return _rate(self.applied_correct, self.applied_correct + self.applied_wrong + self.leaked)

    @property
    def harm_per_hundred(self) -> float | None:
        """Wrong answers per hundred routing cases — refusals excluded, since they are not harm."""
        total = self.answerable + self.refusable
        rate = _rate(self.applied_wrong + self.leaked, total)
        return None if rate is None else rate * 100.0

    def as_dict(self) -> dict[str, Any]:
        def pct(value: float | None) -> float | None:
            return None if value is None else round(value * 100.0, 1)

        return {
            "floor": self.floor,
            "coverage_pct": pct(self.coverage),
            "precision_pct": pct(self.precision),
            "harm_per_hundred": (
                None if self.harm_per_hundred is None else round(self.harm_per_hundred, 1)
            ),
            "applied_correct": self.applied_correct,
            "applied_wrong": self.applied_wrong,
            "leaked": self.leaked,
        }


@dataclass(frozen=True, slots=True)
class BenchResult:
    """The corpus, run once, read at every candidate floor."""

    outcomes: tuple[Outcome, ...]
    as_of: datetime
    floors: tuple[FloorResult, ...] = field(default_factory=tuple)

    @property
    def routing_cases(self) -> tuple[Outcome, ...]:
        """The cases that actually fell through to the router. The only ones a floor can affect."""
        return tuple(o for o in self.outcomes if o.reached_router)

    @property
    def caught_deterministically(self) -> tuple[Outcome, ...]:
        """Cases the regexes already understood.

        Reported rather than hidden: as `question.py` grows, this number grows with it, and a
        corpus that no longer falls through has stopped measuring the router.
        """
        return tuple(o for o in self.outcomes if not o.reached_router)

    @property
    def standing(self) -> FloorResult | None:
        """The row for the floor the code currently ships with."""
        for row in self.floors:
            if abs(row.floor - MIN_CONFIDENCE) < 1e-9:
                return row
        return None

    def render(self) -> str:
        lines = [
            f"ROUTER-BENCH — {len(self.outcomes)} cases, "
            f"{len(self.routing_cases)} reached the router "
            f"({len(self.caught_deterministically)} caught by patterns first)",
            "",
        ]
        if not self.routing_cases:
            lines.append(
                "  no case fell through to the router, so this run says nothing about the "
                "confidence floor. The corpus needs phrasings the patterns do not catch."
            )
            return "\n".join(lines)

        def show(value: float | None) -> str:
            return "     —" if value is None else f"{value * 100.0:5.1f}%"

        lines.append("  floor  coverage  precision  harm/100  correct  wrong  leaked")
        for row in self.floors:
            mark = " <- standing" if abs(row.floor - MIN_CONFIDENCE) < 1e-9 else ""
            harm = "    —" if row.harm_per_hundred is None else f"{row.harm_per_hundred:5.1f}"
            lines.append(
                f"  {row.floor:4.2f}    {show(row.coverage)}     {show(row.precision)}"
                f"    {harm}    {row.applied_correct:5d}  {row.applied_wrong:5d}"
                f"  {row.leaked:6d}{mark}"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "standing_floor": MIN_CONFIDENCE,
            "cases": len(self.outcomes),
            "reached_router": len(self.routing_cases),
            "caught_deterministically": len(self.caught_deterministically),
            "floors": [row.as_dict() for row in self.floors],
            "outcomes": [o.as_dict() for o in self.outcomes],
        }


def sweep(outcomes: Sequence[Outcome], floors: Sequence[float] = FLOORS) -> tuple[FloorResult, ...]:
    """Re-read one set of routings at every candidate floor.

    The model is called once per case, not once per floor: the floor is a decision made *after* a
    confidence comes back, so running the corpus eleven times would spend eleven times the budget
    to obtain the same eleven readings of the same numbers — and would add sampling noise that
    could be mistaken for a threshold effect.
    """
    routing = [o for o in outcomes if o.reached_router]
    answerable = sum(1 for o in routing if not o.case.must_refuse)
    refusable = sum(1 for o in routing if o.case.must_refuse)
    rows: list[FloorResult] = []
    for floor in floors:
        correct = wrong = leaked = 0
        for o in routing:
            # The router applies a routing when it produced an intent at or above the floor.
            if o.routed_to is None or o.confidence < floor:
                continue
            if o.case.must_refuse:
                leaked += 1
            elif o.correct_target:
                correct += 1
            else:
                wrong += 1
        rows.append(FloorResult(floor, answerable, refusable, correct, wrong, leaked))
    return tuple(rows)


def run(
    cases: Sequence[Case] = CASES,
    *,
    client: Router | None,
    now: datetime | None = None,
) -> BenchResult:
    """Classify every case, route the ones that fall through, and sweep the floor.

    Runs with ``min_confidence=0.0`` so that every routing the model produces is observed. The
    floor is then applied arithmetically in :func:`sweep`; letting the live floor filter the run
    would have thrown away exactly the low-confidence readings the sweep needs to see.
    """
    if not cases:
        raise RouterBenchError("no cases: a threshold conclusion needs a corpus to draw it from")
    at = now or datetime.now(UTC)
    outcomes: list[Outcome] = []
    for case in cases:
        question = classify(case.ask, now=at)
        if question.intent not in ROUTABLE_FROM:
            outcomes.append(Outcome(
                case, question.intent, False, 0.0, None,
                "understood by the deterministic classifier; the router was never asked",
            ))
            continue
        _, routing = route(question, client=client, now=at, min_confidence=0.0)
        outcomes.append(Outcome(
            case, question.intent, True, routing.confidence,
            routing.intent if routing.applied else None, routing.detail,
        ))
    return BenchResult(tuple(outcomes), at, sweep(tuple(outcomes)))


def main() -> int:  # pragma: no cover - CLI
    from argus.llm.qwen import QwenClient, TokenBudget

    client: Router | None = None
    if os.environ.get("BITGET_QWEN_API_KEY"):
        client = QwenClient(budget=TokenBudget(limit=120_000))
    result = run(client=client)
    print(result.render())
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
