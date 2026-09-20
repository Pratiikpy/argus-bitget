"""LUI-BENCH tests — a benchmark that cannot fail is a certificate, so most of these make it fail.

The console now scores 100% on all six metrics. That is the least trustworthy moment in a
benchmark's life: it is exactly when a scorer with an inverted comparison, a silently-empty case set
or a one-sided refusal metric looks identical to a working one. Every metric below is therefore
driven to a failing value on purpose and asserted to report it.
"""

from __future__ import annotations

import pytest

from argus.eval.luibench import (
    ANSWERABLE,
    CASES,
    BenchError,
    BenchResult,
    Case,
    Outcome,
    run,
)
from argus.lui.question import Intent


def _outcome(
    expect: Intent, got: Intent, *, family: str = "f", lang: str = "en", refused: bool = False,
    sources: int = 1, ask: str = "q",
) -> Outcome:
    return Outcome(
        case=Case(ask, expect, family, lang=lang), got=got, refused=refused, sources=sources,
    )


class TestTheScorerCanFail:
    def test_a_wrong_intent_is_not_correct(self) -> None:
        got = BenchResult((_outcome(Intent.PERFORMANCE, Intent.SESSION),))
        assert got.intent_accuracy == 0.0
        assert len(got.failures) == 1

    def test_a_right_intent_is_correct(self) -> None:
        got = BenchResult((_outcome(Intent.PERFORMANCE, Intent.PERFORMANCE),))
        assert got.intent_accuracy == 1.0
        assert not got.failures

    def test_unknown_is_not_understood(self) -> None:
        got = BenchResult((_outcome(Intent.PERFORMANCE, Intent.UNKNOWN),))
        assert got.understanding == 0.0

    def test_ambiguous_is_not_understood_either(self) -> None:
        """A question turned into "name a symbol" was not understood; it was deflected."""
        got = BenchResult((_outcome(Intent.EVIDENCE, Intent.AMBIGUOUS),))
        assert got.understanding == 0.0

    def test_understanding_is_measured_only_on_answerable_questions(self) -> None:
        """A question that must be refused cannot count against understanding: reaching UNSUPPORTED
        is the right outcome, not a comprehension failure."""
        got = BenchResult((_outcome(Intent.UNSUPPORTED, Intent.UNSUPPORTED, refused=True),))
        assert got.understanding is None


class TestRefusalIsScoredInBothDirections:
    """A console that refuses everything must not score well. That is the failure a one-sided
    refusal metric hides, and the benchmark reports it by name."""

    def test_refusing_something_answerable_is_wrong(self) -> None:
        got = BenchResult((_outcome(Intent.PERFORMANCE, Intent.UNKNOWN, refused=True),))
        assert got.refusal_accuracy == 0.0
        assert len(got.over_refusals) == 1

    def test_answering_something_that_must_be_refused_is_wrong(self) -> None:
        got = BenchResult((_outcome(Intent.MARKET, Intent.MARKET, refused=False),))
        assert got.refusal_accuracy == 0.0

    def test_a_console_that_refuses_everything_scores_badly(self) -> None:
        outcomes = tuple(
            _outcome(Intent.PERFORMANCE, Intent.UNKNOWN, refused=True, ask=f"q{i}")
            for i in range(10)
        )
        got = BenchResult(outcomes)
        assert got.refusal_accuracy == 0.0
        assert got.understanding == 0.0

    def test_over_refusals_are_named_in_the_verdict(self) -> None:
        got = BenchResult((
            _outcome(Intent.PERFORMANCE, Intent.UNKNOWN, refused=True, ask="are we up?"),
        ))
        assert "are we up?" in got.verdict
        assert "one-sided refusal metric would have hidden" in got.verdict

    def test_must_refuse_is_derived_from_the_answerable_set(self) -> None:
        """Never hand-set, so a case cannot disagree with itself when an answerer is added."""
        assert Case("x", Intent.MARKET, "f").must_refuse is True
        assert Case("x", Intent.PERFORMANCE, "f").must_refuse is False
        for intent in ANSWERABLE:
            assert Case("x", intent, "f").must_refuse is False


class TestParaphraseConsistency:
    def test_a_family_that_scatters_is_inconsistent(self) -> None:
        got = BenchResult((
            _outcome(Intent.PERFORMANCE, Intent.PERFORMANCE, family="perf"),
            _outcome(Intent.PERFORMANCE, Intent.UNKNOWN, family="perf"),
        ))
        assert got.paraphrase_consistency == 0.0

    def test_a_family_that_agrees_is_consistent_even_when_wrong(self) -> None:
        """Consistency and accuracy are different defects: a family agreeing on the wrong intent is
        a mis-specified pattern, while one that scatters is a missing one."""
        got = BenchResult((
            _outcome(Intent.PERFORMANCE, Intent.SESSION, family="perf"),
            _outcome(Intent.PERFORMANCE, Intent.SESSION, family="perf"),
        ))
        assert got.paraphrase_consistency == 1.0
        assert got.intent_accuracy == 0.0

    def test_single_member_families_are_excluded(self) -> None:
        """One phrasing cannot be inconsistent with itself, and counting it would inflate the rate
        with every case that happens to be alone."""
        got = BenchResult((_outcome(Intent.PERFORMANCE, Intent.PERFORMANCE, family="solo"),))
        assert got.paraphrase_consistency is None


class TestBilingualParity:
    def test_a_chinese_question_reaching_a_different_intent_diverges(self) -> None:
        got = BenchResult((
            _outcome(Intent.PERFORMANCE, Intent.PERFORMANCE, family="perf", lang="en"),
            _outcome(Intent.PERFORMANCE, Intent.UNKNOWN, family="perf", lang="zh"),
        ))
        assert got.bilingual_parity == 0.0
        assert ("perf", False) in got.bilingual_pairs

    def test_matching_intents_are_parity(self) -> None:
        got = BenchResult((
            _outcome(Intent.PERFORMANCE, Intent.PERFORMANCE, family="perf", lang="en"),
            _outcome(Intent.PERFORMANCE, Intent.PERFORMANCE, family="perf", lang="zh"),
        ))
        assert got.bilingual_parity == 1.0

    def test_a_family_with_one_language_is_not_a_pair(self) -> None:
        got = BenchResult((
            _outcome(Intent.PERFORMANCE, Intent.PERFORMANCE, family="perf", lang="en"),
        ))
        assert got.bilingual_parity is None

    def test_the_case_set_actually_contains_both_languages(self) -> None:
        """Parity cannot be measured from an all-English corpus, and a benchmark that quietly had
        one would report None forever while looking fine."""
        langs = {c.lang for c in CASES}
        assert {"en", "zh"} <= langs
        zh_families = {c.family for c in CASES if c.lang == "zh"}
        en_families = {c.family for c in CASES if c.lang == "en"}
        assert len(zh_families & en_families) >= 4


class TestGrounding:
    def test_an_answer_with_no_source_is_ungrounded(self) -> None:
        got = BenchResult((_outcome(Intent.PERFORMANCE, Intent.PERFORMANCE, sources=0),))
        assert got.grounding == 0.0

    def test_a_refusal_needs_no_citation(self) -> None:
        got = BenchResult((
            _outcome(Intent.MARKET, Intent.MARKET, refused=True, sources=0),
        ))
        assert got.grounding is None

    def test_grounded_is_none_for_a_refusal_rather_than_false(self) -> None:
        assert _outcome(Intent.MARKET, Intent.MARKET, refused=True).grounded is None


class TestTheCaseSetIsHonest:
    def test_no_case_is_empty_of_intent(self) -> None:
        for case in CASES:
            assert case.family.strip()
            assert isinstance(case.expect, Intent)

    def test_it_contains_cases_that_must_be_refused(self) -> None:
        """A corpus of only answerable questions cannot measure over-answering."""
        assert sum(1 for c in CASES if c.must_refuse) >= 5

    def test_it_contains_more_than_one_phrasing_for_several_intents(self) -> None:
        from collections import Counter

        families = Counter(c.family for c in CASES)
        assert sum(1 for n in families.values() if n > 1) >= 5

    def test_no_case_is_a_verbatim_pattern_from_the_classifier(self) -> None:
        """Cases written from the regexes would test that the module agrees with itself.

        Only lines that *define a pattern* count. A phrasing quoted in a comment is a different
        thing: `question.py` documents "how did we do this week?" as a miss it once had, so that
        case is a regression guard rather than an unfitted probe — which the test below measures
        separately rather than letting it hide inside this one.
        """
        import inspect

        from argus.lui import question as q

        pattern_lines = [
            line for line in inspect.getsource(q).splitlines()
            if "Intent." in line and ("(r\"" in line or "r\"" in line)
        ]
        blob = chr(10).join(pattern_lines)
        for case in CASES:
            if case.ask:
                assert case.ask not in blob, case.ask

    def test_most_cases_are_phrasings_the_source_has_never_seen(self) -> None:
        """The benchmark's whole value is phrasings nobody tuned for.

        A few overlap deliberately: a phrasing the source documents as a past miss earns a permanent
        regression guard. But if most cases were already known to the module, the score would be a
        restatement of the unit tests, so the share is asserted rather than assumed.
        """
        import inspect

        from argus.lui import question as q

        source = inspect.getsource(q)
        novel = [c for c in CASES if c.ask and c.ask not in source]
        assert len(novel) / len([c for c in CASES if c.ask]) > 0.85

    def test_an_empty_case_set_raises(self) -> None:
        with pytest.raises(BenchError, match="not a score"):
            run(cases=())


class TestTheVerdictRefusesToOverclaim:
    def test_a_perfect_score_still_names_the_fitting_limitation(self) -> None:
        got = BenchResult((_outcome(Intent.PERFORMANCE, Intent.PERFORMANCE),))
        assert "the same hand wrote the console and the cases" in got.verdict
        assert "never as a fluency ceiling" in got.verdict

    def test_failures_are_listed_rather_than_averaged(self) -> None:
        got = BenchResult((
            _outcome(Intent.PERFORMANCE, Intent.SESSION),
            _outcome(Intent.PERFORMANCE, Intent.PERFORMANCE),
        ))
        assert "listed rather than averaged away" in got.verdict

    def test_an_empty_result_is_undefined(self) -> None:
        assert "UNDEFINED" in BenchResult(()).verdict

    def test_the_report_serialises(self) -> None:
        import json

        blob = json.loads(json.dumps(BenchResult((
            _outcome(Intent.PERFORMANCE, Intent.PERFORMANCE),
        )).as_dict()))
        assert blob["cases"] == 1


class TestTheLiveConsole:
    """The regression guard. If a pattern change breaks a phrasing, this fails by name."""

    @pytest.fixture(scope="class")
    def result(self) -> BenchResult:
        return run()

    def test_no_case_regresses(self, result: BenchResult) -> None:
        broken = [
            f"{o.case.lang} {o.case.ask!r}: expected {o.case.expect.value}, got {o.got.value}"
            for o in result.failures
        ]
        assert not broken, broken

    def test_every_answer_cites_a_source(self, result: BenchResult) -> None:
        assert result.grounding == 1.0

    def test_bilingual_parity_holds(self, result: BenchResult) -> None:
        diverged = [name for name, ok in result.bilingual_pairs if not ok]
        assert not diverged, diverged

    def test_nothing_answerable_is_refused(self, result: BenchResult) -> None:
        assert not result.over_refusals, [o.case.ask for o in result.over_refusals]
