"""Causal-chain tests — the point is to catch the win a P&L scorer would bank."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from argus.agents.causality import (
    STANDARD_STEPS,
    CausalChain,
    CausalLedger,
    Link,
    LinkGrade,
    chain_from_response,
    grade_magnitude,
)

T0 = datetime(2026, 3, 9, 14, 0, tzinfo=UTC)


def _chain(direction: str = "bearish", realised: str = "bearish") -> CausalChain:
    c = CausalChain(
        event="8-K: FY guidance cut 7%", as_of=T0,
        predicted_direction=direction, predicted_magnitude_bps=60,
        realised_direction=realised, realised_magnitude_bps=55,
    )
    for step in STANDARD_STEPS[:4]:
        c.add(Link(step=step, claim=f"{step} claim", falsifier=f"{step} would be refuted if ..."))
    return c


class TestLuckyWins:
    """The case this module exists to name."""

    def test_right_direction_through_broken_links_is_lucky_not_sound(self) -> None:
        c = _chain()
        c.grade(0, LinkGrade.CORRECT)
        c.grade(1, LinkGrade.WRONG)
        c.grade(2, LinkGrade.WRONG)
        c.grade(3, LinkGrade.WRONG)

        assert c.direction_correct is True
        assert c.chain_accuracy == 0.25
        assert c.was_lucky is True
        assert c.verdict == "lucky"

    def test_right_direction_through_sound_links_is_sound(self) -> None:
        c = _chain()
        for i in range(4):
            c.grade(i, LinkGrade.CORRECT)
        assert c.verdict == "sound"
        assert c.was_lucky is False

    def test_wrong_direction_with_sound_reasoning_is_unlucky(self) -> None:
        """Worth keeping — the process may still be right."""
        c = _chain(direction="bearish", realised="bullish")
        for i in range(4):
            c.grade(i, LinkGrade.CORRECT)
        assert c.direction_correct is False
        assert c.was_unlucky is True
        assert c.verdict == "unlucky"

    def test_wrong_direction_and_broken_links_is_simply_wrong(self) -> None:
        c = _chain(direction="bearish", realised="bullish")
        for i in range(4):
            c.grade(i, LinkGrade.WRONG)
        assert c.verdict == "wrong"


class TestUnfalsifiableLinks:
    def test_a_link_without_a_falsifier_starts_unsupported(self) -> None:
        c = CausalChain(event="e", as_of=T0)
        c.add(Link(step="mechanism", claim="it will go down", falsifier=""))
        assert c.grades[0] is LinkGrade.UNSUPPORTED

    def test_an_unsupported_link_cannot_be_graded_correct_afterwards(self) -> None:
        """Otherwise a chain can launder an unfalsifiable claim into a confirmed one."""
        c = CausalChain(event="e", as_of=T0)
        c.add(Link(step="mechanism", claim="vibes", falsifier=""))
        with pytest.raises(ValueError, match="cannot be graded after"):
            c.grade(0, LinkGrade.CORRECT)

    def test_grading_a_link_that_does_not_exist_raises(self) -> None:
        with pytest.raises(IndexError):
            _chain().grade(99, LinkGrade.CORRECT)


class TestWeakestLink:
    def test_reports_the_first_break(self) -> None:
        c = _chain()
        c.grade(0, LinkGrade.CORRECT)
        c.grade(1, LinkGrade.CORRECT)
        c.grade(2, LinkGrade.WRONG)
        c.grade(3, LinkGrade.WRONG)
        assert c.weakest_link() is not None
        assert c.weakest_link().step == STANDARD_STEPS[2]

    def test_a_sound_chain_has_no_weakest_link(self) -> None:
        c = _chain()
        for i in range(4):
            c.grade(i, LinkGrade.CORRECT)
        assert c.weakest_link() is None


class TestChainConstruction:
    def test_missing_links_are_not_padded(self) -> None:
        """Filling gaps would make every chain look complete and destroy the metric."""
        c = chain_from_response("e", T0, {
            "signal": "bearish", "magnitude_bps": 40,
            "chain": ["guidance cut", "earnings expectations fall"],
            "chain_falsifiers": ["guidance reaffirmed", "consensus unchanged"],
        })
        assert len(c.links) == 2
        assert c.links[0].step == STANDARD_STEPS[0]

    def test_links_without_falsifiers_are_marked_unsupported(self) -> None:
        c = chain_from_response("e", T0, {
            "signal": "bearish", "chain": ["a", "b"], "chain_falsifiers": ["only one"],
        })
        assert c.grades[0] is LinkGrade.UNGRADED
        assert c.grades[1] is LinkGrade.UNSUPPORTED


class TestMagnitudeGrading:
    def test_a_reasonable_miss_counts(self) -> None:
        assert grade_magnitude(50, 35) is LinkGrade.CORRECT

    def test_an_order_of_magnitude_miss_does_not(self) -> None:
        assert grade_magnitude(50, 400) is LinkGrade.WRONG

    def test_a_zero_move_is_uncertain_not_wrong(self) -> None:
        assert grade_magnitude(50, 0) is LinkGrade.UNCERTAIN


class TestLedger:
    def test_surfaces_the_step_that_breaks_most_often(self) -> None:
        """A systematic reasoning weakness neither P&L nor direction accuracy can show."""
        led = CausalLedger()
        for _ in range(4):
            c = _chain()
            c.grade(0, LinkGrade.CORRECT)
            c.grade(1, LinkGrade.WRONG)
            c.grade(2, LinkGrade.CORRECT)
            c.grade(3, LinkGrade.CORRECT)
            led.add(c)
        got = led.summary()
        assert got["most_common_break"] == STANDARD_STEPS[1]
        assert got["direction_accuracy_pct"] == 100.0
        assert got["mean_chain_accuracy_pct"] == 75.0

    def test_reports_the_lucky_win_rate(self) -> None:
        led = CausalLedger()
        for _ in range(4):
            c = _chain()
            for i in range(4):
                c.grade(i, LinkGrade.WRONG if i else LinkGrade.CORRECT)
            led.add(c)
        assert led.summary()["lucky_wins_pct"] == 100.0

    def test_unresolved_chains_report_nothing_rather_than_zero(self) -> None:
        led = CausalLedger()
        led.add(CausalChain(event="e", as_of=T0))
        assert "no outcomes resolved" in led.summary()["note"]


class TestUngradedIsNotZeroPercent:
    """`chain_accuracy` returned **0.0** when nothing had been graded — indistinguishable from a
    chain that was graded and got every link wrong.

    The consequence was not cosmetic. `was_lucky` is `direction_correct and chain_accuracy < 0.5`,
    so **every ungraded chain that happened to call the direction right was labelled a lucky win** —
    the exact accusation this module exists to make carefully, made automatically against decisions
    nobody had examined. `summary` then averaged those zeros into `mean_chain_accuracy_pct`.
    """

    @staticmethod
    def _chain(direction: str = "up") -> CausalChain:
        from datetime import UTC, datetime

        return CausalChain(
            event="probe", as_of=datetime(2026, 6, 1, tzinfo=UTC),
            predicted_direction=direction, predicted_magnitude_bps=50,
        )

    def test_an_ungraded_chain_reports_none_not_zero(self) -> None:
        assert self._chain().chain_accuracy is None

    def test_an_ungraded_chain_is_not_called_lucky(self) -> None:
        chain = self._chain()
        chain.realised_direction = "up"
        assert chain.was_lucky is False, "unmeasured is not the same as reasoning that broke"

    def test_an_ungraded_chain_is_not_called_unlucky_either(self) -> None:
        """Symmetrical: unmeasured is not a defence."""
        chain = self._chain()
        chain.realised_direction = "down"
        assert chain.was_unlucky is False

    def test_as_dict_carries_none_rather_than_a_rounded_zero(self) -> None:
        assert self._chain().as_dict()["chain_accuracy"] is None


class TestAnAbsentMagnitudeIsNotAPredictionOfNoMove:
    """`int(float(response.get("magnitude_bps", 0) or 0))` turned a missing field into a confident
    prediction — *"this event moves the price not at all"* — attributed to a model that never said
    it, then scored that invention against the realised move."""

    def test_a_missing_magnitude_parses_to_none(self) -> None:
        from argus.agents.causality import _magnitude

        assert _magnitude(None) is None
        assert _magnitude("") is None

    def test_an_unparseable_magnitude_is_unknown_not_zero(self) -> None:
        from argus.agents.causality import _magnitude

        assert _magnitude("about fifty") is None

    def test_a_real_magnitude_still_parses(self) -> None:
        from argus.agents.causality import _magnitude

        assert _magnitude("50") == 50
        assert _magnitude(50.7) == 50

    def test_grading_an_absent_magnitude_is_uncertain_not_wrong(self) -> None:
        """Grading our own default is not grading the model."""
        from argus.agents.causality import LinkGrade, grade_magnitude

        assert grade_magnitude(None, 300) is LinkGrade.UNCERTAIN

    def test_a_real_magnitude_still_grades(self) -> None:
        from argus.agents.causality import LinkGrade, grade_magnitude

        assert grade_magnitude(50, 60) is LinkGrade.CORRECT
        assert grade_magnitude(50, 400) is LinkGrade.WRONG

    def test_magnitude_error_is_none_when_nothing_was_predicted(self) -> None:
        from datetime import UTC, datetime

        chain = CausalChain(
            event="probe", as_of=datetime(2026, 6, 1, tzinfo=UTC),
            predicted_direction="up", predicted_magnitude_bps=None,
        )
        chain.realised_magnitude_bps = 300
        assert chain.magnitude_error_bps() is None
