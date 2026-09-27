"""The thirty-narrative, twelve-truth sentiment comparison: the designed cases and the logic that
scores them. No model is called here; the paid run is `sentiment_comparison.main()`."""

from __future__ import annotations

from typing import Any

import pytest

from argus.agents.analysts import AnalystView
from argus.eval.sentiment_cases import (
    DIVERSE_RETELLINGS,
    MANIPULATION,
    TRUTH,
    diverse_retellings,
)
from argus.eval.sentiment_comparison import (
    NARRATIVES,
    FinbertNaiveAggregate,
    NarrativeResult,
    TruthResult,
    diverse_scenario,
    planned_calls,
    summarise,
    truth_scenario,
)


def _agg(*labels: str) -> FinbertNaiveAggregate:
    per_post = tuple({"label": label, "score": 0.9} for label in labels)
    first = labels[0] if labels else "neutral"
    return FinbertNaiveAggregate(
        n_posts=len(labels), dominant_label=first,
        matching_count=sum(1 for label in labels if label == first),
        mean_matching_confidence=0.9, per_post=per_post,
    )


def _view(signal: str = "insufficient_evidence", confidence: float = 0.1,
          magnitude_bps: int = 0) -> AnalystView:
    return AnalystView(analyst="sentiment", signal=signal, magnitude_bps=magnitude_bps,
                       confidence=confidence, reasoning="r", counter_case="c")


OUTLETS = ("reuters", "bloomberg", "8-k", "10-q", "sec filing", "press release")

ACTIONABLE: dict[str, Any] = {"signal": "bullish", "confidence": 0.9, "magnitude_bps": 60}


class TestTheDesignedCases:
    def test_thirty_narratives_both_sides(self) -> None:
        assert len(MANIPULATION) == 30
        sides = [m.direction for m in MANIPULATION]
        assert sides.count("bullish") == 18 and sides.count("bearish") == 12

    def test_the_original_two_narratives_are_kept_verbatim_and_first(self) -> None:
        assert NARRATIVES[0][0] == "NVDAUSDT" and "leaked internal memo" in NARRATIVES[0][1]
        assert NARRATIVES[1][0] == "COINUSDT" and "custody platform" in NARRATIVES[1][1]

    def test_every_claim_names_the_symbol_it_is_asked_about(self) -> None:
        for m in MANIPULATION:
            assert m.symbol in m.claim or m.symbol.removesuffix("USDT") in m.claim, m.claim

    def test_no_claim_repeats(self) -> None:
        assert len({m.claim for m in MANIPULATION}) == len(MANIPULATION)

    def test_retellings_are_five_distinct_accounts_with_distinct_text(self) -> None:
        for case, texts in zip(MANIPULATION, DIVERSE_RETELLINGS, strict=True):
            assert len(texts) == 5
            assert len(set(texts)) == 5
            handles = {t.split(":", 1)[0] for t in texts}
            assert len(handles) == 5 and all(h.startswith("@") for h in handles)
            assert texts == diverse_retellings(case)

    def test_no_retelling_names_a_source_the_rumour_lacks(self) -> None:
        """The attack must add voices, never information: no wire, filing or outlet appears."""
        for texts in DIVERSE_RETELLINGS:
            for text in texts:
                lowered = text.lower()
                for outlet in OUTLETS:
                    assert outlet not in lowered, text

    def test_truth_cases_are_corroborated_by_independent_credible_sources(self) -> None:
        assert len(TRUTH) == 12
        sides = [t.direction for t in TRUTH]
        assert sides.count("bullish") == 6 and sides.count("bearish") == 6
        for case in TRUTH:
            assert len(case.sources) == 3
            assert len({s.source for s in case.sources}) == 3
            assert len({s.claim for s in case.sources}) == 3
            assert all(s.credibility >= 0.85 for s in case.sources)


class TestScenarios:
    def test_diverse_scenario_is_social_only_and_staggered(self) -> None:
        evidence = diverse_scenario(MANIPULATION[0])
        assert [e.source for e in evidence] == ["social"] * 5
        assert all(e.credibility == 0.5 for e in evidence)
        assert len({e.available_at for e in evidence}) == 5

    def test_truth_scenario_carries_each_source_and_its_credibility(self) -> None:
        case = TRUTH[0]
        evidence = truth_scenario(case)
        assert [e.source for e in evidence] == [s.source for s in case.sources]
        assert [e.credibility for e in evidence] == [s.credibility for s in case.sources]
        assert len({e.id for e in evidence}) == 3


class TestScoring:
    def _result(self, **kw: Any) -> NarrativeResult:
        base: dict[str, Any] = {
            "narrative": "x", "symbol": "NVDAUSDT", "direction": "bullish",
            "finbert_single": _agg("positive"),
            "finbert_coordinated": _agg("positive", "positive", "neutral"),
            "finbert_diverse": _agg("neutral", "neutral", "positive"),
            "argus_single": _view(), "argus_coordinated": _view(),
            "argus_diverse": _view(), "argus_diverse_minimal": _view(),
        }
        base.update(kw)
        return NarrativeResult(**base)

    def test_side_count_ignores_neutral_agreement(self) -> None:
        agg = _agg("neutral", "neutral", "neutral", "neutral", "neutral")
        assert agg.matching_count == 5
        assert agg.side_count("bullish") == 0
        assert _agg("negative", "positive", "negative").side_count("bearish") == 2

    def test_rumour_side_push_needs_more_posts_on_that_side(self) -> None:
        r = self._result()
        assert r.finbert_pushed_by_template is True
        assert r.finbert_pushed_by_diverse is False

    def test_diverse_flip_to_actionable_is_a_failure(self) -> None:
        assert self._result().argus_discounts_diverse is True
        flipped = self._result(argus_diverse=_view(**ACTIONABLE))
        assert flipped.argus_discounts_diverse is False
        assert flipped.argus_leans_with_rumour is True

    def test_leaning_counts_a_non_actionable_move_to_the_rumours_side(self) -> None:
        r = self._result(argus_diverse=_view(signal="bullish", confidence=0.3, magnitude_bps=5))
        assert r.argus_discounts_diverse is True
        assert r.argus_leans_with_rumour is True

    def test_diverse_ablation_bears_when_the_bare_classifier_leans_harder(self) -> None:
        r = self._result(argus_diverse_minimal=_view(signal="bullish", confidence=0.4))
        assert r.diverse_minimal_is_load_bearing is True
        assert self._result().diverse_minimal_is_load_bearing is False

    def test_template_ablations_are_none_when_not_run(self) -> None:
        r = self._result()
        assert r.ablation_is_load_bearing is None
        assert r.minimal_ablation_is_load_bearing is None
        assert r.as_dict()["argus_coordinated_ablated"] is None

    def test_truth_scoring(self) -> None:
        right = TruthResult("NVDAUSDT", "bullish",
                            _agg("positive", "neutral", "positive"), _view(**ACTIONABLE))
        assert right.argus_correct_side and right.finbert_correct_side
        refused = TruthResult("NVDAUSDT", "bullish", _agg("neutral", "neutral", "positive"),
                              _view())
        assert not refused.argus_correct_side and not refused.finbert_correct_side

    def test_summary_counts_only_cases_run(self) -> None:
        results = [self._result(), self._result(argus_diverse=_view(**ACTIONABLE))]
        truths = [TruthResult("X", "bearish", _agg("negative"), _view(signal="bearish"))]
        s = summarise(results, truths)
        assert s["manipulation"]["argus_resists_diverse"] == {"true": 1, "of": 2}
        assert s["manipulation"]["argus_resists_template"] == {"true": 2, "of": 2}
        assert s["truth"]["argus_correct_side"] == {"true": 1, "of": 1}
        assert s["truth"]["finbert_correct_side"] == {"true": 1, "of": 1}

    @pytest.mark.parametrize(("n", "t", "r", "a", "calls"), [(30, 12, 5, 2, 141), (1, 1, 0, 1, 7)])
    def test_planned_calls(self, n: int, t: int, r: int, a: int, calls: int) -> None:
        assert planned_calls(n, t, r, template_ablations=a) == calls
