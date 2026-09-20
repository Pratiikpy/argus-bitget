"""Tests for the ARGUS-vs-finBERT sentiment-integrity comparison.

The finBERT side runs the REAL `ProsusAI/finbert` model (local CPU inference, free, deterministic
in eval mode) — no mocking. ARGUS's `SentimentAnalyst` side needs a real Qwen LLM call (real money
from a limited hackathon budget), so those cases are exercised here with a fake `ChatModel` that
returns fixed, designed responses — proving the AGGREGATION/COMPARISON logic is correct — while the
real end-to-end run happens once, by hand, via `main()`, matching `eval/routerbench.py`'s own
established pattern of keeping paid LLM calls out of the automated suite.
"""

from __future__ import annotations

from typing import Any

import pytest

from argus.agents.analysts import AnalystView, SentimentAnalyst
from argus.eval.baselines.finbert_loader import load_finbert
from argus.eval.sentiment_comparison import (
    NARRATIVES,
    SCOPE_STATEMENT,
    NarrativeResult,
    SentimentComparisonError,
    build_ablated_analyst_role,
    coordinated_scenario,
    measure_finbert_cost,
    run_finbert_naive_aggregate,
    single_source_scenario,
)


@pytest.fixture(scope="module")
def finbert():
    return load_finbert()


class TestScenarioFixtures:
    def test_single_source_is_exactly_one_evidence_item(self) -> None:
        scenario = single_source_scenario("some claim")
        assert len(scenario) == 1
        assert scenario[0].claim == "some claim"

    def test_coordinated_defaults_to_five_rephrasings(self) -> None:
        scenario = coordinated_scenario("some claim")
        assert len(scenario) == 5
        assert all("some claim" in e.claim or "SOME CLAIM" in e.claim for e in scenario)

    def test_coordinated_posts_are_staggered_in_time(self) -> None:
        scenario = coordinated_scenario("some claim")
        timestamps = [e.available_at for e in scenario]
        assert timestamps == sorted(timestamps)
        assert len(set(timestamps)) == len(timestamps)

    def test_coordinated_respects_n(self) -> None:
        assert len(coordinated_scenario("x", n=3)) == 3


class TestAblatedRoleStaysInSyncWithTheRealPrompt:
    def test_the_real_role_contains_the_pinned_instruction(self) -> None:
        assert "Five accounts repeating one article is" in SentimentAnalyst.role

    def test_the_ablated_role_does_not(self) -> None:
        ablated = build_ablated_analyst_role()
        assert "Five accounts" not in ablated

    def test_the_ablated_role_is_shorter_but_not_empty(self) -> None:
        ablated = build_ablated_analyst_role()
        assert 0 < len(ablated) < len(SentimentAnalyst.role)

    def test_a_prompt_that_has_drifted_from_the_pinned_string_is_caught(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(SentimentAnalyst, "role", "a role with no coordination instruction")
        with pytest.raises(SentimentComparisonError, match="no longer present verbatim"):
            build_ablated_analyst_role()


class TestFinbertNaiveAggregateOnRealModel:
    def test_dominant_label_is_the_first_posts_real_label(self, finbert: Any) -> None:
        _symbol, claim = NARRATIVES[0]
        scenario = single_source_scenario(claim)
        agg = run_finbert_naive_aggregate(finbert, scenario)
        assert agg.matching_count == 1

    def test_every_narrative_shows_the_coordinated_aggregate_exceeding_the_single(
        self, finbert: Any,
    ) -> None:
        """The core empirical finding, run for real on finBERT, for every narrative — not just
        one convenient case."""
        for _symbol, claim in NARRATIVES:
            single = run_finbert_naive_aggregate(finbert, single_source_scenario(claim))
            coordinated = run_finbert_naive_aggregate(finbert, coordinated_scenario(claim))
            assert single.matching_count == 1
            assert coordinated.matching_count > single.matching_count, claim

    def test_per_post_carries_every_real_finbert_result(self, finbert: Any) -> None:
        _symbol, claim = NARRATIVES[0]
        scenario = coordinated_scenario(claim, n=5)
        agg = run_finbert_naive_aggregate(finbert, scenario)
        assert len(agg.per_post) == 5
        assert all("label" in r and "score" in r for r in agg.per_post)


class TestCosts:
    def test_finbert_latency_is_measured_and_positive(self, finbert: Any) -> None:
        ms = measure_finbert_cost(finbert, n=5)
        assert ms > 0


def _view(
    *, signal: str = "insufficient_evidence", confidence: float = 0.1, magnitude_bps: int = 20,
) -> AnalystView:
    return AnalystView(
        analyst="sentiment", signal=signal, magnitude_bps=magnitude_bps, confidence=confidence,
        reasoning="test", counter_case="test",
    )


_ACTIONABLE = {"signal": "bullish", "confidence": 0.9, "magnitude_bps": 50}
_NOT_ACTIONABLE = {"signal": "insufficient_evidence", "confidence": 0.1, "magnitude_bps": 0}


class TestNarrativeResultLogic:
    """Exercises the comparison/aggregation logic with a real finBERT aggregate paired against
    FAKE (hand-constructed) AnalystViews — proving the decision logic is correct without needing
    a real, paid LLM call for every combination."""

    def _finbert_pair(self, finbert: Any, narrative: str) -> tuple[Any, Any]:
        return (
            run_finbert_naive_aggregate(finbert, single_source_scenario(narrative)),
            run_finbert_naive_aggregate(finbert, coordinated_scenario(narrative)),
        )

    def test_argus_discounts_coordination_when_both_stay_non_actionable(
        self, finbert: Any,
    ) -> None:
        _symbol, claim = NARRATIVES[0]
        single, coordinated = self._finbert_pair(finbert, claim)
        result = NarrativeResult(
            narrative=claim, finbert_single=single, finbert_coordinated=coordinated,
            argus_single=_view(**_NOT_ACTIONABLE), argus_coordinated=_view(**_NOT_ACTIONABLE),
            argus_coordinated_ablated=_view(**_NOT_ACTIONABLE),
            argus_coordinated_minimal=_view(**_NOT_ACTIONABLE),
        )
        assert result.argus_discounts_coordination is True

    def test_argus_fails_to_discount_when_repetition_alone_flips_to_actionable(
        self, finbert: Any,
    ) -> None:
        """The real manipulation outcome this whole comparison exists to catch: a single
        unsourced source is correctly non-actionable, but repeating it alone makes the SAME
        analyst call it actionable — no new information, just volume."""
        _symbol, claim = NARRATIVES[0]
        single, coordinated = self._finbert_pair(finbert, claim)
        result = NarrativeResult(
            narrative=claim, finbert_single=single, finbert_coordinated=coordinated,
            argus_single=_view(**_NOT_ACTIONABLE), argus_coordinated=_view(**_ACTIONABLE),
            argus_coordinated_ablated=_view(**_ACTIONABLE),
            argus_coordinated_minimal=_view(**_ACTIONABLE),
        )
        assert result.argus_discounts_coordination is False

    def test_discounting_holds_even_if_single_source_was_already_actionable(
        self, finbert: Any,
    ) -> None:
        """Not a manipulation outcome: if the single source alone was already enough to act on,
        the coordinated case being actionable too is not evidence repetition caused anything."""
        _symbol, claim = NARRATIVES[0]
        single, coordinated = self._finbert_pair(finbert, claim)
        result = NarrativeResult(
            narrative=claim, finbert_single=single, finbert_coordinated=coordinated,
            argus_single=_view(**_ACTIONABLE), argus_coordinated=_view(**_ACTIONABLE),
            argus_coordinated_ablated=_view(**_ACTIONABLE),
            argus_coordinated_minimal=_view(**_ACTIONABLE),
        )
        assert result.argus_discounts_coordination is True

    def test_ablation_is_load_bearing_when_the_ablated_analyst_flips_to_actionable(
        self, finbert: Any,
    ) -> None:
        _symbol, claim = NARRATIVES[0]
        single, coordinated = self._finbert_pair(finbert, claim)
        result = NarrativeResult(
            narrative=claim, finbert_single=single, finbert_coordinated=coordinated,
            argus_single=_view(**_NOT_ACTIONABLE), argus_coordinated=_view(**_NOT_ACTIONABLE),
            argus_coordinated_ablated=_view(**_ACTIONABLE),
            argus_coordinated_minimal=_view(**_NOT_ACTIONABLE),
        )
        assert result.ablation_is_load_bearing is True

    def test_ablation_is_load_bearing_on_directional_confidence_short_of_actionable(
        self, finbert: Any,
    ) -> None:
        """Even without crossing into fully actionable, a real directional-confidence increase
        under the ablated prompt is still a real (smaller) signal the instruction was doing
        something — caught by the confidence comparison, not just the binary flip."""
        _symbol, claim = NARRATIVES[0]
        single, coordinated = self._finbert_pair(finbert, claim)
        result = NarrativeResult(
            narrative=claim, finbert_single=single, finbert_coordinated=coordinated,
            argus_single=_view(**_NOT_ACTIONABLE),
            argus_coordinated=_view(signal="bullish", confidence=0.2, magnitude_bps=30),
            argus_coordinated_ablated=_view(signal="bullish", confidence=0.45, magnitude_bps=30),
            argus_coordinated_minimal=_view(**_NOT_ACTIONABLE),
        )
        assert result.ablation_is_load_bearing is True

    def test_ablation_is_not_load_bearing_when_both_behave_identically(
        self, finbert: Any,
    ) -> None:
        _symbol, claim = NARRATIVES[0]
        single, coordinated = self._finbert_pair(finbert, claim)
        result = NarrativeResult(
            narrative=claim, finbert_single=single, finbert_coordinated=coordinated,
            argus_single=_view(**_NOT_ACTIONABLE), argus_coordinated=_view(**_NOT_ACTIONABLE),
            argus_coordinated_ablated=_view(**_NOT_ACTIONABLE),
            argus_coordinated_minimal=_view(**_NOT_ACTIONABLE),
        )
        assert result.ablation_is_load_bearing is False

    def test_minimal_ablation_is_load_bearing_when_the_bare_classifier_flips_to_actionable(
        self, finbert: Any,
    ) -> None:
        """The coarser ablation: with ALL source-independence framing stripped, a bare
        classifier that flips to actionable where the real analyst did not is the decisive
        signal that the whole framing — not any one sentence — is load-bearing."""
        _symbol, claim = NARRATIVES[0]
        single, coordinated = self._finbert_pair(finbert, claim)
        result = NarrativeResult(
            narrative=claim, finbert_single=single, finbert_coordinated=coordinated,
            argus_single=_view(**_NOT_ACTIONABLE), argus_coordinated=_view(**_NOT_ACTIONABLE),
            argus_coordinated_ablated=_view(**_NOT_ACTIONABLE),
            argus_coordinated_minimal=_view(**_ACTIONABLE),
        )
        assert result.minimal_ablation_is_load_bearing is True

    def test_minimal_ablation_is_not_load_bearing_when_the_bare_classifier_still_discounts(
        self, finbert: Any,
    ) -> None:
        _symbol, claim = NARRATIVES[0]
        single, coordinated = self._finbert_pair(finbert, claim)
        result = NarrativeResult(
            narrative=claim, finbert_single=single, finbert_coordinated=coordinated,
            argus_single=_view(**_NOT_ACTIONABLE), argus_coordinated=_view(**_NOT_ACTIONABLE),
            argus_coordinated_ablated=_view(**_NOT_ACTIONABLE),
            argus_coordinated_minimal=_view(**_NOT_ACTIONABLE),
        )
        assert result.minimal_ablation_is_load_bearing is False

    def test_as_dict_serialises_every_field(self, finbert: Any) -> None:
        _symbol, claim = NARRATIVES[0]
        single, coordinated = self._finbert_pair(finbert, claim)
        result = NarrativeResult(
            narrative=claim, finbert_single=single, finbert_coordinated=coordinated,
            argus_single=_view(), argus_coordinated=_view(), argus_coordinated_ablated=_view(),
            argus_coordinated_minimal=_view(),
        )
        d = result.as_dict()
        assert d["narrative"] == claim
        assert "minimal_ablation_is_load_bearing" in d
        assert "finbert_signal_scales_with_repetition" in d
        assert "argus_discounts_coordination" in d
        assert "ablation_is_load_bearing" in d


class TestScopeStatement:
    def test_names_what_is_not_claimed(self) -> None:
        assert "NOT claimed" in SCOPE_STATEMENT
        assert "classification" in SCOPE_STATEMENT
        assert "DEMOTED" in SCOPE_STATEMENT
