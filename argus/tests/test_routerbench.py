"""ROUTER-BENCH tests.

The benchmark exists to settle whether `lui/router.py`'s confidence floor earns its place, so the
properties that matter here are the ones that would let it produce a flattering answer:

* a floor must be re-read from one set of routings rather than re-running the model per floor, or
  the curve is measuring sampling noise;
* a case the deterministic classifier already understands must be excluded from the sweep and
  *counted*, because a corpus that stops falling through has stopped testing the router;
* an empty denominator must report ``None``, never ``0.0`` — a coverage of zero and a coverage that
  was never measurable are different findings.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from argus.eval.routerbench import (
    CASES,
    BenchResult,
    Case,
    FloorResult,
    Outcome,
    RouterBenchError,
    run,
    sweep,
)
from argus.lui.question import Intent

AT = datetime(2026, 9, 15, 15, 0, tzinfo=UTC)


class _Stub:
    """A router that answers with a scripted intent and confidence, keyed by question text."""

    def __init__(self, script: dict[str, tuple[str, float]]) -> None:
        self.script = script
        self.asked: list[str] = []

    def complete_json(
        self, messages: list[dict[str, Any]], *, required_keys: tuple[str, ...] = (),
        max_tokens: int = 0, thinking: Any = None,
    ) -> dict[str, Any]:
        text = messages[-1]["content"]
        self.asked.append(text)
        for needle, (intent, confidence) in self.script.items():
            if needle in text:
                return {
                    "intent": intent, "confidence": confidence,
                    "symbols": [], "seq": None, "window": None, "why": "stub",
                }
        return {
            "intent": "unknown", "confidence": 0.0,
            "symbols": [], "seq": None, "window": None, "why": "stub",
        }


def _outcome(
    ask: str, expect: Intent | None, *, confidence: float, routed_to: Intent | None,
    reached: bool = True,
) -> Outcome:
    return Outcome(
        Case(ask, expect, "stub"), Intent.UNKNOWN, reached, confidence, routed_to,
    )


class TestTheCorpusIsAboutTheRouterAndSaysSo:
    def test_cases_the_patterns_already_catch_are_excluded_but_counted(self) -> None:
        """A corpus the regexes swallow is not measuring the router, and must show that."""
        result = run(client=None, now=AT)
        assert len(result.outcomes) == len(CASES)
        assert result.routing_cases
        assert result.caught_deterministically
        assert (
            len(result.routing_cases) + len(result.caught_deterministically) == len(CASES)
        )

    def test_the_corpus_contains_questions_that_must_stay_refused(self) -> None:
        """A corpus of only answerable questions cannot detect a floor that is too low."""
        assert any(case.must_refuse for case in CASES)
        assert any(not case.must_refuse for case in CASES)

    def test_a_refusable_case_declares_no_expected_intent(self) -> None:
        for case in CASES:
            assert case.must_refuse == (case.expect is None)

    def test_an_empty_corpus_raises_rather_than_reporting_a_threshold(self) -> None:
        with pytest.raises(RouterBenchError):
            run((), client=None, now=AT)


class TestTheSweepReadsOneRunAtEveryFloor:
    def test_the_model_is_called_once_per_case_not_once_per_floor(self) -> None:
        stub = _Stub({"anything still open": ("position", 0.8)})
        run(client=stub, now=AT)
        assert len(stub.asked) == len(
            [o for o in run(client=None, now=AT).outcomes if o.reached_router]
        )

    def test_a_routing_below_the_floor_is_not_counted_as_applied(self) -> None:
        rows = sweep(
            [_outcome("q", Intent.POSITION, confidence=0.4, routed_to=Intent.POSITION)],
            floors=(0.3, 0.5),
        )
        assert rows[0].applied_correct == 1
        assert rows[1].applied_correct == 0

    def test_exactly_at_the_floor_counts_as_applied(self) -> None:
        """Matches `route`, which admits a confidence equal to the floor."""
        rows = sweep(
            [_outcome("q", Intent.POSITION, confidence=0.6, routed_to=Intent.POSITION)],
            floors=(0.6,),
        )
        assert rows[0].applied_correct == 1

    def test_a_case_that_never_reached_the_router_affects_no_floor(self) -> None:
        rows = sweep(
            [_outcome(
                "q", Intent.POSITION, confidence=0.9, routed_to=Intent.POSITION, reached=False,
            )],
            floors=(0.0,),
        )
        assert rows[0].answerable == 0
        assert rows[0].applied_correct == 0


class TestWrongAndLeakedAreNotTheSameFailure:
    def test_a_routing_to_the_wrong_intent_is_wrong_not_leaked(self) -> None:
        rows = sweep(
            [_outcome("q", Intent.POSITION, confidence=0.9, routed_to=Intent.SESSION)],
            floors=(0.0,),
        )
        assert (rows[0].applied_wrong, rows[0].leaked) == (1, 0)

    def test_a_routing_applied_to_a_question_that_must_be_refused_is_leaked(self) -> None:
        rows = sweep(
            [_outcome("q", None, confidence=0.9, routed_to=Intent.POSITION)],
            floors=(0.0,),
        )
        assert (rows[0].applied_wrong, rows[0].leaked) == (0, 1)

    def test_a_refusable_case_can_never_be_scored_as_correct(self) -> None:
        """There is no right intent to reach, so any applied routing is the error."""
        assert not _outcome(
            "q", None, confidence=0.9, routed_to=Intent.POSITION,
        ).correct_target


class TestAnUnmeasurableRateIsNeverZero:
    def test_precision_with_nothing_applied_is_none_not_zero(self) -> None:
        row = FloorResult(1.0, answerable=5, refusable=2, applied_correct=0, applied_wrong=0,
                          leaked=0)
        assert row.precision is None
        assert row.as_dict()["precision_pct"] is None

    def test_coverage_with_no_answerable_cases_is_none_not_zero(self) -> None:
        row = FloorResult(0.0, answerable=0, refusable=3, applied_correct=0, applied_wrong=0,
                          leaked=0)
        assert row.coverage is None

    def test_harm_with_no_routing_cases_at_all_is_none_not_zero(self) -> None:
        row = FloorResult(0.0, answerable=0, refusable=0, applied_correct=0, applied_wrong=0,
                          leaked=0)
        assert row.harm_per_hundred is None

    def test_a_run_with_no_routing_cases_says_so_rather_than_scoring_zero(self) -> None:
        text = BenchResult((), AT, ()).render()
        assert "no case fell through" in text


class TestTheRecordIsQueryable:
    def test_the_standing_floor_is_identified_in_the_curve(self) -> None:
        result = run(client=None, now=AT)
        assert result.standing is not None
        assert "<- standing" in result.render()

    def test_every_outcome_serialises_with_what_stopped_it(self) -> None:
        payload = run(client=None, now=AT).as_dict()
        assert payload["cases"] == len(CASES)
        assert payload["reached_router"] + payload["caught_deterministically"] == len(CASES)
        for entry in payload["outcomes"]:
            assert {"ask", "must_refuse", "reached_router", "confidence", "routed_to"} <= set(entry)

    def test_a_stubbed_run_scores_a_correct_routing_as_correct(self) -> None:
        """Scoring is exercised with a **synthetic** case, not one borrowed from `CASES`.

        This test used to pick ``"anything still open"`` out of the live corpus because the
        pattern layer could not classify it, so it was guaranteed to fall through to the router.
        On 2026-09-21 the patterns were widened and caught it — and every other answerable case
        in `CASES` — so the test failed on an *improvement*. A test whose setup depends on the
        system under test still being weak will always eventually break for the wrong reason.
        """
        from argus.eval.routerbench import Case
        from argus.lui.question import Intent

        ask = "zzz qqq vvv"          # deliberately unclassifiable by any pattern, now or later
        case = Case(ask=ask, expect=Intent.POSITION, family="synthetic", lang="en")
        stub = _Stub({ask: ("position", 0.9)})
        result = run((case,), client=stub, now=AT)
        hit = next(o for o in result.outcomes if o.case.ask == ask)
        assert hit.reached_router and hit.correct_target

    def test_the_pattern_layer_now_catches_every_answerable_case_in_the_corpus(self) -> None:
        """A finding, pinned: this corpus no longer reaches the router at all.

        `eval/routerbench.py`'s forty cases were written as phrasings the deterministic layer
        could not handle — it caught 16 of 40. After the widening on 2026-09-21 it catches every
        answerable one, which is why the test above had to stop borrowing from the corpus. If this
        ever fails, the patterns regressed.
        """
        from argus.lui.question import Intent, classify

        fell_through = [
            c for c in CASES
            if not c.must_refuse
            and classify(c.ask, now=AT).intent in (Intent.UNKNOWN, Intent.AMBIGUOUS)
        ]
        assert not fell_through, [c.ask for c in fell_through]
