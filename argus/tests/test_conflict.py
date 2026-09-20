"""Conflict-record tests.

Two properties matter here. First, disagreement must be *emitted as data*, so a judge asking "did
your analysts agree?" gets an answer from the record rather than from the thesis text. Second,
unanimity must be stated explicitly and labelled honestly: a panel that ran in sequence and agreed
has not demonstrated consensus, it has demonstrated that nobody contradicted the first speaker.
"""

from __future__ import annotations

import pytest

from argus.agents.analysts import ACTIONABLE_CONFIDENCE, AnalystView
from argus.agents.conflict import (
    CONVICTION_GAP,
    MATERIAL_GAP_BPS,
    Kind,
    detect,
    independent_views,
    report,
)


def _view(
    analyst: str,
    signal: str,
    magnitude: int,
    confidence: float,
    sources: tuple[str, ...] = ("a",),
) -> AnalystView:
    return AnalystView(
        analyst=analyst,
        signal=signal,
        magnitude_bps=magnitude,
        confidence=confidence,
        reasoning="fixture",
        counter_case="fixture counter",
        source_ids=sources,
    )


class TestDirectionalSplit:
    """The disagreement that cannot be split the difference on."""

    def test_opposite_signals_are_a_direction_conflict(self) -> None:
        conflicts = detect([
            _view("event", "bullish", 40, 0.8),
            _view("sentiment", "bearish", 35, 0.6),
        ])
        assert len(conflicts) == 1
        assert conflicts[0].kind is Kind.DIRECTION

    def test_a_directional_split_is_flagged_on_the_report(self) -> None:
        rep = report([
            _view("event", "bullish", 40, 0.8),
            _view("sentiment", "bearish", 35, 0.6),
        ])
        assert rep.has_directional_split is True
        assert rep.unanimous is False

    @pytest.mark.parametrize(
        ("left", "right"),
        [("bullish", "bearish"), ("long", "short"), ("buy", "sell"), ("positive", "negative")],
    )
    def test_the_vocabulary_analysts_actually_use_is_recognised(
        self, left: str, right: str
    ) -> None:
        conflicts = detect([_view("a", left, 40, 0.8), _view("b", right, 40, 0.6)])
        assert conflicts and conflicts[0].kind is Kind.DIRECTION

    def test_a_neutral_view_does_not_oppose_a_directional_one(self) -> None:
        """Neutral is not the opposite of bullish; treating it as one invents disagreement."""
        assert detect([_view("a", "bullish", 40, 0.8), _view("b", "neutral", 5, 0.6)]) == ()


class TestMagnitudeAndConviction:
    def test_same_direction_far_apart_on_size_is_a_magnitude_conflict(self) -> None:
        conflicts = detect([
            _view("event", "bullish", 80, 0.7),
            _view("sentiment", "bullish", 20, 0.7, sources=("b", "c")),
        ])
        assert len(conflicts) == 1
        assert conflicts[0].kind is Kind.MAGNITUDE
        assert conflicts[0].gap == 60

    def test_a_small_size_difference_is_agreement_not_conflict(self) -> None:
        conflicts = detect([
            _view("event", "bullish", 40, 0.7),
            _view("sentiment", "bullish", 40 + int(MATERIAL_GAP_BPS) - 1, 0.7),
        ])
        assert conflicts == ()

    def test_same_call_far_apart_on_confidence_is_a_conviction_conflict(self) -> None:
        conflicts = detect([
            _view("event", "bullish", 40, 0.9),
            _view("sentiment", "bullish", 40, 0.4, sources=("b", "c")),
        ])
        assert len(conflicts) == 1
        assert conflicts[0].kind is Kind.CONVICTION

    def test_two_views_straddling_the_actionable_line_conflict_however_close_they_look(
        self,
    ) -> None:
        """0.52 and 0.49 is a gap of 0.03, and one of those views may be traded while the other

        may not. The size of the gap is the wrong question when it falls across the boundary the
        desk actually acts on; this pair used to be reported as no disagreement at all.
        """
        conflicts = detect([
            _view("event", "bullish", 40, ACTIONABLE_CONFIDENCE + 0.02),
            _view("sentiment", "bullish", 40, ACTIONABLE_CONFIDENCE - 0.01, sources=("b", "c")),
        ])
        assert len(conflicts) == 1
        assert conflicts[0].kind is Kind.CONVICTION
        assert conflicts[0].gap < CONVICTION_GAP

    def test_two_views_on_the_same_side_of_the_line_and_close_together_do_not_conflict(
        self,
    ) -> None:
        """The straddle rule must not swallow the ordinary case it sits next to."""
        assert detect([
            _view("event", "bullish", 40, ACTIONABLE_CONFIDENCE + 0.02),
            _view("sentiment", "bullish", 40, ACTIONABLE_CONFIDENCE + 0.05),
        ]) == ()

    def test_a_view_is_never_in_conflict_with_itself(self) -> None:
        one = _view("event", "bullish", 40, 0.8)
        assert detect([one]) == ()


class TestResolution:
    def test_clearly_higher_confidence_dominates(self) -> None:
        conflicts = detect([
            _view("event", "bullish", 40, 0.9),
            _view("sentiment", "bearish", 35, 0.5),
        ])
        assert conflicts[0].dominant == "event"
        assert "confidence" in conflicts[0].grounds

    def test_near_equal_confidence_is_decided_by_evidence_breadth(self) -> None:
        """Two analysts within 0.1 are not meaningfully more sure; picking one reads noise."""
        conflicts = detect([
            _view("event", "bullish", 40, 0.70, sources=("a", "b", "c")),
            _view("sentiment", "bearish", 35, 0.66, sources=("d",)),
        ])
        assert conflicts[0].dominant == "event"
        assert "distinct sources" in conflicts[0].grounds

    def test_an_evenly_matched_disagreement_stands_unresolved(self) -> None:
        """Manufacturing a winner here would be the dishonest option."""
        conflicts = detect([
            _view("event", "bullish", 40, 0.7, sources=("a",)),
            _view("sentiment", "bearish", 35, 0.7, sources=("b",)),
        ])
        assert conflicts[0].dominant == "neither"
        assert "unresolved" in conflicts[0].grounds

    def test_the_report_counts_unresolved_disagreements(self) -> None:
        rep = report([
            _view("event", "bullish", 40, 0.7, sources=("a",)),
            _view("sentiment", "bearish", 35, 0.7, sources=("b",)),
        ])
        assert len(rep.unresolved) == 1
        assert "in spite of them" in " ".join(rep.render())


class TestUnanimityIsStatedNotImplied:
    def test_an_agreeing_panel_produces_an_explicit_note(self) -> None:
        rep = report([_view("event", "bullish", 40, 0.8), _view("sentiment", "bullish", 42, 0.75)])
        assert rep.unanimous is True
        assert "none:" in rep.render()[0]

    def test_sequential_agreement_is_labelled_as_possible_contagion(self) -> None:
        """The honest caveat: nobody contradicting the first speaker is not consensus."""
        rep = report(
            [_view("event", "bullish", 40, 0.8), _view("sentiment", "bullish", 42, 0.75)],
            sequential=True,
        )
        assert "contagion" in rep.render()[0]

    def test_parallel_agreement_carries_no_such_caveat(self) -> None:
        rep = report(
            [_view("event", "bullish", 40, 0.8), _view("sentiment", "bullish", 42, 0.75)],
            sequential=False,
        )
        assert "contagion" not in rep.render()[0]

    def test_the_default_is_the_pessimistic_reading(self) -> None:
        """A caller that has not thought about ordering must not get a free independence claim."""
        assert report([_view("a", "bullish", 40, 0.8)]).sequential is True


class TestIndependenceIsNotAssumed:
    def test_sequential_analysts_are_never_independent(self) -> None:
        views = [_view("a", "bullish", 40, 0.8, ("x",)), _view("b", "bullish", 41, 0.8, ("y",))]
        assert independent_views(views, sequential=True) is False

    def test_parallel_analysts_on_one_source_are_not_independent_either(self) -> None:
        """Parallel analysts reading one article are as correlated as sequential ones."""
        views = [_view("a", "bullish", 40, 0.8, ("x",)), _view("b", "bullish", 41, 0.8, ("x",))]
        assert independent_views(views, sequential=False) is False

    def test_parallel_analysts_on_distinct_sources_are_independent(self) -> None:
        views = [_view("a", "bullish", 40, 0.8, ("x",)), _view("b", "bearish", 41, 0.8, ("y",))]
        assert independent_views(views, sequential=False) is True


class TestTheRecordIsQueryable:
    def test_every_conflict_serialises_with_its_grounds(self) -> None:
        rep = report([
            _view("event", "bullish", 80, 0.9),
            _view("sentiment", "bearish", 20, 0.4),
            _view("cross_asset", "bullish", 25, 0.85),
        ])
        payload = rep.as_dict()
        assert payload["analysts"] == 3
        assert payload["unanimous"] is False
        for entry in payload["conflicts"]:
            assert {"kind", "between", "signals", "gap", "dominant", "grounds"} <= set(entry)
            assert entry["grounds"]

    def test_a_rendered_conflict_names_both_analysts_and_the_winner(self) -> None:
        conflicts = detect([
            _view("event", "bullish", 40, 0.9),
            _view("sentiment", "bearish", 35, 0.5),
        ])
        text = conflicts[0].render()
        assert "event" in text and "sentiment" in text and "dominates" in text

    def test_an_empty_panel_is_unanimous_but_says_so_with_zero_analysts(self) -> None:
        rep = report([])
        assert rep.unanimous is True
        assert rep.as_dict()["analysts"] == 0
