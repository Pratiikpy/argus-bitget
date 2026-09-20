"""The debate: bounded, priced, convergence-measured, and honest about what agreement proves."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from argus.agents.debate import (
    CONVERGENCE_BPS,
    MAX_ROUNDS,
    ROUND_COST_BPS,
    Debate,
    DebateBudget,
    Ending,
    Position,
    Side,
    converged,
    hold,
    signed,
)

EVIDENCE = ("NVDA announced a datacentre agreement", "VIX 15.8, 38th percentile")
RICH = DebateBudget(limit_bps=Decimal("100"))


def _pos(side: Side, direction: str, magnitude: float, *, round_index: int = 0,
         concedes: str = "the other side's best point") -> Position:
    return Position(side=side, round_index=round_index, direction=direction,
                    magnitude_bps=magnitude, case="a case", strongest_opposing_point=concedes)


class ScriptedSeat:
    """Answers from a list. Records what it was shown, so the prompt can be asserted on."""

    def __init__(self, answers: list[dict[str, Any]]) -> None:
        self.answers = answers
        self.seen: list[str] = []
        self.calls = 0

    def complete_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.seen.append(str(messages[-1].get("content", "")))
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return answer


class DeadSeat:
    def complete_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("the seat is down")


def _answer(direction: str, magnitude: float, concedes: str = "their point") -> dict[str, Any]:
    return {"direction": direction, "magnitude_bps": magnitude, "case": "a case",
            "strongest_opposing_point": concedes}


class TestConvergenceIsSignedNotAbsolute:
    def test_opposite_views_of_equal_size_are_the_widest_disagreement(self) -> None:
        """The arithmetic error this guards: comparing unsigned magnitudes would call +30 vs -30
        a perfect agreement, when it is the sharpest possible disagreement."""
        bull = _pos(Side.BULL, "up", 30.0)
        bear = _pos(Side.BEAR, "down", 30.0)
        assert not converged(bull, bear)

    def test_the_same_view_converges(self) -> None:
        assert converged(_pos(Side.BULL, "up", 30.0), _pos(Side.BEAR, "up", 31.0))

    def test_unclear_is_zero_not_a_small_move(self) -> None:
        assert signed(_pos(Side.BULL, "unclear", 40.0)) == 0.0

    def test_the_threshold_is_below_half_a_round_trip(self) -> None:
        """A disagreement smaller than half the fee cannot change what the desk does."""
        assert CONVERGENCE_BPS < 12.0 / 2

    def test_the_boundary_is_inclusive(self) -> None:
        bull = _pos(Side.BULL, "up", CONVERGENCE_BPS)
        bear = _pos(Side.BEAR, "up", 0.0)
        assert converged(bull, bear)


class TestTheDebateIsBounded:
    def test_it_stops_at_the_round_ceiling(self) -> None:
        bulls = ScriptedSeat([_answer("up", 100.0)])
        bears = ScriptedSeat([_answer("down", 100.0)])
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=bulls, bear_seat=bears, budget=RICH)
        assert got.rounds == MAX_ROUNDS
        assert got.ending is Ending.EXHAUSTED_ROUNDS

    def test_the_ceiling_is_three(self) -> None:
        assert MAX_ROUNDS == 3

    def test_it_stops_early_on_convergence(self) -> None:
        bulls = ScriptedSeat([_answer("up", 20.0)])
        bears = ScriptedSeat([_answer("up", 21.0)])
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=bulls, bear_seat=bears, budget=RICH)
        assert got.rounds == 1 and got.ending is Ending.CONVERGED

    def test_a_budget_that_cannot_afford_a_round_holds_no_debate(self) -> None:
        tiny = DebateBudget(limit_bps=Decimal("1"))
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=ScriptedSeat([_answer("up", 1.0)]),
                   bear_seat=ScriptedSeat([_answer("down", 1.0)]), budget=tiny)
        assert got.ending is Ending.EXHAUSTED_BUDGET
        assert "costs more than it could settle" in got.render()

    def test_a_budget_for_exactly_one_round_stops_after_one(self) -> None:
        one = DebateBudget(limit_bps=ROUND_COST_BPS)
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=ScriptedSeat([_answer("up", 90.0)]),
                   bear_seat=ScriptedSeat([_answer("down", 90.0)]), budget=one)
        assert got.rounds == 1 and got.ending is Ending.EXHAUSTED_BUDGET

    def test_the_cost_is_reported_in_basis_points(self) -> None:
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=ScriptedSeat([_answer("up", 90.0)]),
                   bear_seat=ScriptedSeat([_answer("down", 90.0)]), budget=RICH)
        assert got.cost_bps == ROUND_COST_BPS * MAX_ROUNDS

    def test_no_evidence_means_no_debate(self) -> None:
        """A debate with nothing to cite produces two well-written opinions."""
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=(),
                   bull_seat=ScriptedSeat([_answer("up", 1.0)]),
                   bear_seat=ScriptedSeat([_answer("down", 1.0)]), budget=RICH)
        assert got.ending is Ending.NOT_HELD
        assert "most expensive way to learn nothing" in got.render()

    def test_a_missing_seat_means_no_debate(self) -> None:
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=None, bear_seat=ScriptedSeat([]), budget=RICH)
        assert got.ending is Ending.NOT_HELD


class TestAgreementIsNotEvidenceFromOneModel:
    def test_a_shared_model_cannot_corroborate_itself(self) -> None:
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=ScriptedSeat([_answer("up", 20.0)]),
                   bear_seat=ScriptedSeat([_answer("up", 20.0)]), budget=RICH,
                   shared_model=True)
        assert not got.agreement_is_evidence

    def test_the_transcript_says_so_in_words(self) -> None:
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=ScriptedSeat([_answer("up", 20.0)]),
                   bear_seat=ScriptedSeat([_answer("up", 20.0)]), budget=RICH)
        assert "self-consistency and NOT" in got.render()

    def test_two_models_can_corroborate(self) -> None:
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=ScriptedSeat([_answer("up", 20.0)]),
                   bear_seat=ScriptedSeat([_answer("up", 20.0)]), budget=RICH,
                   shared_model=False)
        assert got.agreement_is_evidence
        assert "self-consistency and NOT" not in got.render()


class TestDisagreementSurvivesToThePM:
    def test_an_unresolved_debate_is_marked_unresolved(self) -> None:
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=ScriptedSeat([_answer("up", 80.0)]),
                   bear_seat=ScriptedSeat([_answer("down", 80.0)]), budget=RICH)
        assert got.unresolved
        assert "UNRESOLVED" in got.render()

    def test_the_gap_is_reported_as_a_number(self) -> None:
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=ScriptedSeat([_answer("up", 40.0)]),
                   bear_seat=ScriptedSeat([_answer("down", 40.0)]), budget=RICH)
        assert got.gap_bps == 80.0

    def test_a_converged_debate_is_not_unresolved(self) -> None:
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=ScriptedSeat([_answer("up", 20.0)]),
                   bear_seat=ScriptedSeat([_answer("up", 20.0)]), budget=RICH)
        assert not got.unresolved

    def test_disagreement_is_never_averaged_into_a_consensus(self) -> None:
        """There is no mean anywhere in the result. Averaging +80 and -80 to zero would report a
        neutral view that neither side holds."""
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=ScriptedSeat([_answer("up", 80.0)]),
                   bear_seat=ScriptedSeat([_answer("down", 80.0)]), budget=RICH)
        assert got.final[Side.BULL] is not None and got.final[Side.BEAR] is not None
        assert not hasattr(got, "consensus")


class TestEngagementIsCounted:
    def test_a_side_that_concedes_nothing_is_counted_as_not_engaging(self) -> None:
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=ScriptedSeat([_answer("up", 80.0, concedes="")]),
                   bear_seat=ScriptedSeat([_answer("down", 80.0, concedes="their point")]),
                   budget=RICH)
        assert got.engaged["bull"] == 0
        assert got.engaged["bear"] == MAX_ROUNDS

    def test_the_transcript_explains_what_engagement_means(self) -> None:
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=ScriptedSeat([_answer("up", 80.0)]),
                   bear_seat=ScriptedSeat([_answer("down", 80.0)]), budget=RICH)
        assert "has not argued, it has repeated" in got.render()

    def test_an_empty_concession_renders_visibly(self) -> None:
        assert "did not engage" in _pos(Side.BULL, "up", 1.0, concedes="").render()


class TestTheOpeningRoundHasNothingToRebut:
    def test_the_first_prompt_says_there_is_nothing_to_answer(self) -> None:
        """Without this a seat invents an opposing argument and rebuts it, which is the phantom
        the reference implementation guards against too."""
        bulls = ScriptedSeat([_answer("up", 20.0)])
        hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
             bull_seat=bulls, bear_seat=ScriptedSeat([_answer("up", 20.0)]), budget=RICH)
        assert "nothing to answer yet" in bulls.seen[0]

    def test_the_bear_sees_the_bull_in_the_same_round(self) -> None:
        """Otherwise it is two monologues, not a debate."""
        bears = ScriptedSeat([_answer("down", 20.0)])
        hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
             bull_seat=ScriptedSeat([_answer("up", 55.0)]), bear_seat=bears, budget=RICH)
        assert "55.0bps" in bears.seen[0]

    def test_the_evidence_reaches_both_seats(self) -> None:
        bulls = ScriptedSeat([_answer("up", 20.0)])
        bears = ScriptedSeat([_answer("up", 20.0)])
        hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
             bull_seat=bulls, bear_seat=bears, budget=RICH)
        assert "datacentre agreement" in bulls.seen[0]
        assert "datacentre agreement" in bears.seen[0]


class TestAFailedSeatDoesNotTakeTheDecisionDown:
    def test_a_dead_bull_is_recorded_as_unavailable(self) -> None:
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=DeadSeat(), bear_seat=ScriptedSeat([_answer("down", 1.0)]),
                   budget=RICH)
        assert got.ending is Ending.UNAVAILABLE
        assert "proceeds undebated" in got.render()

    def test_a_dead_bear_keeps_the_bull_position(self) -> None:
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=ScriptedSeat([_answer("up", 30.0)]), bear_seat=DeadSeat(),
                   budget=RICH)
        assert got.ending is Ending.UNAVAILABLE
        assert len(got.positions) == 1


class TestMalformedAnswersAreNotTrusted:
    def test_an_unknown_direction_becomes_unclear(self) -> None:
        seat = ScriptedSeat([{"direction": "sideways-ish", "magnitude_bps": 5,
                              "case": "c", "strongest_opposing_point": "p"}])
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=seat, bear_seat=ScriptedSeat([_answer("up", 0.0)]), budget=RICH)
        assert got.positions[0].direction == "unclear"

    def test_a_non_numeric_magnitude_is_unknown_not_zero(self) -> None:
        """**This test previously asserted the defect.** It was named
        `test_a_non_numeric_magnitude_becomes_zero` and pinned exactly the behaviour that made a
        parse failure look like a stated prediction of no movement.

        Zero is a real answer here: `converged` compares signed moves, so two seats that both
        failed to parse became `abs(0.0 - 0.0) = 0` — inside every tolerance — and a double parse
        failure was recorded as the bull and the bear agreeing.
        """
        seat = ScriptedSeat([{"direction": "up", "magnitude_bps": "a lot",
                              "case": "c", "strongest_opposing_point": "p"}])
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=seat, bear_seat=ScriptedSeat([_answer("up", 0.0)]), budget=RICH)
        assert got.positions[0].magnitude_bps is None

    def test_two_unparseable_seats_do_not_count_as_agreement(self) -> None:
        """The consequence the type change exists to prevent, driven end to end."""
        bad = {"direction": "up", "magnitude_bps": "a lot",
               "case": "c", "strongest_opposing_point": "p"}
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=ScriptedSeat([bad]), bear_seat=ScriptedSeat([dict(bad)]),
                   budget=RICH)
        assert got.gap_bps is None, "no gap is measurable when neither side named a number"
        assert got.unresolved is False, "and an unmeasurable debate is not 'resolved' either"

    def test_a_negative_magnitude_is_taken_as_size_not_direction(self) -> None:
        """Direction is its own field. A bear returning -50 means fifty, downward, not up."""
        seat = ScriptedSeat([_answer("down", -50.0)])
        got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
                   bull_seat=ScriptedSeat([_answer("up", 0.0)]), bear_seat=seat, budget=RICH)
        bear = got.final[Side.BEAR]
        assert bear is not None and bear.magnitude_bps == 50.0 and signed(bear) == -50.0


def test_it_serialises_whole() -> None:
    got = hold(symbol="NVDAUSDT", horizon_hours=24, evidence=EVIDENCE,
               bull_seat=ScriptedSeat([_answer("up", 40.0)]),
               bear_seat=ScriptedSeat([_answer("down", 40.0)]), budget=RICH).as_dict()
    assert got["ending"] == "exhausted_rounds"
    assert got["gap_bps"] == 80.0
    assert got["agreement_is_evidence"] is False
    assert len(got["positions"]) == 2 * MAX_ROUNDS


def test_a_debate_that_never_happened_still_serialises() -> None:
    got = Debate(symbol="X", positions=(), ending=Ending.NOT_HELD, shared_model=True,
                 note="nothing to argue").as_dict()
    # `gap_bps` is None, not 0.0: zero is the value that means the two sides agree exactly, and a
    # debate nobody attended must not be indistinguishable from perfect consensus.
    assert got["rounds"] == 0 and got["gap_bps"] is None
