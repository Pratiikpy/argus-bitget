"""Escalation tests — the asymmetry is the property, and it is tested as one.

Two things could go wrong here and only one of them is visible. The obvious failure is a condition
that never fires, which makes the takeover rate a measurement of dead code. The dangerous failure is
an escalation that *does* something — produces size, flips a side, turns a refusal into an action —
because that is the risk layer creating exposure, which is the single invariant this whole project
is built around. So every path through :func:`apply` is checked against the asymmetry, and the
conditions are each tested firing and not firing.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from argus.decision.escalation import (
    DEFAULT_UNATTENDED_FRACTION,
    TRIGGER_REASONS,
    Escalation,
    EscalationError,
    Signals,
    Trigger,
    apply,
    assess,
    takeover_rate,
)
from argus.decision.verdicts import Intent, Side, Verdict


def _intent(
    *, verdict: Verdict = Verdict.TRADE, quantity: str = "1", side: Side = Side.BUY
) -> Intent:
    return Intent(
        symbol="NVDAUSDT",
        side=side,
        quantity=Decimal(quantity),
        verdict=verdict,
        stated_confidence=0.7,
        thesis="a thesis",
        invalidation=("the gap closes",),
    )


class TestEachConditionFires:
    def test_a_split_panel_that_did_not_converge_escalates(self) -> None:
        got = assess(_intent(), Signals(directional_split=True, debate_converged=False))
        assert Trigger.UNRESOLVED_CONFLICT in got.triggers

    def test_a_split_panel_that_converged_does_not(self) -> None:
        """Disagreement that was argued out is a decision, not a deadlock. Escalating it would
        punish the desk for holding a debate."""
        got = assess(_intent(), Signals(directional_split=True, debate_converged=True))
        assert Trigger.UNRESOLVED_CONFLICT not in got.triggers

    def test_an_unconverged_debate_with_no_split_does_not(self) -> None:
        got = assess(_intent(), Signals(directional_split=False, debate_converged=False))
        assert Trigger.UNRESOLVED_CONFLICT not in got.triggers

    def test_a_thesis_refuted_by_its_own_falsifier_escalates(self) -> None:
        got = assess(
            _intent(),
            Signals(
                thesis_refuted_by_own_falsifier=True,
                refuted_condition="the gap already closed",
            ),
        )
        assert Trigger.THESIS_ALREADY_REFUTED in got.triggers
        assert "the gap already closed" in got.detail

    def test_a_refutation_with_no_named_condition_still_escalates_and_says_so(self) -> None:
        got = assess(_intent(), Signals(thesis_refuted_by_own_falsifier=True))
        assert Trigger.THESIS_ALREADY_REFUTED in got.triggers
        assert "not named" in got.detail[0]

    def test_an_unresolved_figure_escalates(self) -> None:
        got = assess(_intent(), Signals(unresolved_figures=1))
        assert Trigger.UNGROUNDED_FIGURE in got.triggers
        assert "1 figure(s)" in got.detail[0]

    def test_fully_grounded_figures_do_not(self) -> None:
        assert assess(_intent(), Signals(unresolved_figures=0)).triggers == ()

    def test_a_halted_underlying_escalates(self) -> None:
        got = assess(_intent(), Signals(underlying_halted=True, halt_reason="LUDP"))
        assert Trigger.UNDERLYING_HALTED in got.triggers
        assert "LUDP" in got.detail

    def test_size_beyond_the_unattended_limit_escalates(self) -> None:
        got = assess(_intent(), Signals(position_fraction_of_book=Decimal("0.30")))
        assert Trigger.SIZE_BEYOND_MANDATE in got.triggers

    def test_size_exactly_at_the_limit_does_not(self) -> None:
        """The limit is what is permitted, not what is refused. An inclusive boundary here would
        make the stated mandate quietly stricter than the number it publishes."""
        got = assess(_intent(), Signals(position_fraction_of_book=DEFAULT_UNATTENDED_FRACTION))
        assert Trigger.SIZE_BEYOND_MANDATE not in got.triggers

    def test_a_tighter_mandate_escalates_a_size_a_looser_one_permits(self) -> None:
        size = Decimal("0.20")
        loose = assess(_intent(), Signals(position_fraction_of_book=size))
        tight = assess(
            _intent(),
            Signals(position_fraction_of_book=size, unattended_fraction=Decimal("0.10")),
        )
        assert not loose.required
        assert Trigger.SIZE_BEYOND_MANDATE in tight.triggers

    def test_every_trigger_has_a_stated_reason(self) -> None:
        for trigger in Trigger:
            assert trigger in TRIGGER_REASONS
            assert len(TRIGGER_REASONS[trigger]) > 40


class TestWhatIsNotEscalated:
    def test_a_clean_decision_is_not_escalated(self) -> None:
        assert not assess(_intent(), Signals()).required

    def test_an_abstention_is_never_escalated_however_bad_the_signals(self) -> None:
        """A desk that says no has already declined to act. Asking a human to approve doing nothing
        converts a decision into an interruption, and would inflate the takeover rate with the very
        decisions that needed no human at all."""
        signals = Signals(
            directional_split=True,
            debate_converged=False,
            thesis_refuted_by_own_falsifier=True,
            unresolved_figures=4,
            underlying_halted=True,
            position_fraction_of_book=Decimal("0.99"),
        )
        for verdict in (Verdict.NO_TRADE, Verdict.DATA_INSUFFICIENT, Verdict.DELAY):
            assert not assess(_intent(verdict=verdict, quantity="0"), signals).required, verdict

    def test_a_decision_already_at_human_review_is_not_escalated_again(self) -> None:
        signals = Signals(underlying_halted=True)
        assert not assess(_intent(verdict=Verdict.HUMAN_REVIEW, quantity="0"), signals).required


class TestAllCausesAreReportedNotJustTheFirst:
    def test_several_conditions_all_appear(self) -> None:
        """Reporting only the first would let a decision be cleared by fixing one condition while
        another still held, and would understate the causes in the benchmark."""
        got = assess(
            _intent(),
            Signals(
                directional_split=True,
                debate_converged=False,
                unresolved_figures=2,
                underlying_halted=True,
            ),
        )
        assert len(got.triggers) == 3
        assert Trigger.UNRESOLVED_CONFLICT in got.triggers
        assert Trigger.UNGROUNDED_FIGURE in got.triggers
        assert Trigger.UNDERLYING_HALTED in got.triggers

    def test_every_trigger_carries_its_own_detail_line(self) -> None:
        got = assess(_intent(), Signals(unresolved_figures=3, underlying_halted=True))
        assert len(got.detail) == len(got.triggers)

    def test_the_report_names_each_cause(self) -> None:
        text = assess(_intent(), Signals(unresolved_figures=1, underlying_halted=True)).render()
        assert "HANDED TO A HUMAN" in text
        assert "ungrounded_figure" in text
        assert "underlying_halted" in text

    def test_no_escalation_renders_as_permission_to_act(self) -> None:
        assert "may act on this unattended" in assess(_intent(), Signals()).render()


class TestTheAsymmetry:
    """An escalation may only reduce. This is the invariant, not a convention."""

    def test_it_zeroes_the_quantity(self) -> None:
        escalation = assess(_intent(quantity="7"), Signals(underlying_halted=True))
        assert apply(_intent(quantity="7"), escalation).quantity == Decimal("0")

    def test_it_produces_a_verdict_that_opens_nothing(self) -> None:
        escalation = assess(_intent(), Signals(underlying_halted=True))
        got = apply(_intent(), escalation)
        assert got.verdict is Verdict.HUMAN_REVIEW
        assert not got.verdict.opens_exposure

    def test_it_never_enlarges_a_position(self) -> None:
        for size in ("0.5", "1", "10", "1000"):
            escalation = assess(_intent(quantity=size), Signals(underlying_halted=True))
            assert apply(_intent(quantity=size), escalation).quantity <= Decimal(size)

    def test_it_never_reverses_a_side(self) -> None:
        """Flipping a side would be the risk layer taking a position of its own."""
        for side in (Side.BUY, Side.SELL):
            escalation = assess(_intent(side=side), Signals(underlying_halted=True))
            assert apply(_intent(side=side), escalation).side is side

    def test_it_preserves_the_thesis_and_the_falsifiers_for_the_human(self) -> None:
        """A human picking this up needs what the desk was thinking, not a blank form."""
        original = _intent()
        got = apply(original, assess(original, Signals(underlying_halted=True)))
        assert got.thesis == original.thesis
        assert got.invalidation == original.invalidation

    def test_no_escalation_returns_the_intent_untouched(self) -> None:
        """So a caller can apply this unconditionally.

        The branch is where a control gets skipped.
        """
        original = _intent()
        assert apply(original, assess(original, Signals())) is original

    def test_escalating_a_decision_that_opens_nothing_raises(self) -> None:
        """There is no action here to stop, so a caller reaching this has a bug, not a control."""
        forced = Escalation(triggers=(Trigger.UNDERLYING_HALTED,), detail=("",))
        with pytest.raises(EscalationError, match="no action here to stop"):
            apply(_intent(verdict=Verdict.NO_TRADE, quantity="0"), forced)


class TestTheTakeoverRate:
    def _clean(self) -> Escalation:
        return Escalation(triggers=(), detail=())

    def _raised(self, trigger: Trigger = Trigger.UNDERLYING_HALTED) -> Escalation:
        return Escalation(triggers=(trigger,), detail=("",))

    def test_the_denominator_is_actionable_decisions_not_all_decisions(self) -> None:
        """A desk that abstains constantly would otherwise report a takeover rate near zero while
        escalating every decision it actually made."""
        outcomes = [
            *[(False, self._clean()) for _ in range(90)],
            *[(True, self._raised()) for _ in range(5)],
            *[(True, self._clean()) for _ in range(5)],
        ]
        got = takeover_rate(outcomes)
        assert got.actionable == 10
        assert got.total_decisions == 100
        assert got.rate == pytest.approx(0.5)

    def test_no_actionable_decision_is_undefined_not_zero(self) -> None:
        """Zero would read as "never needed to ask". This desk has never had anything to ask about,
        which is a different statement and the one that is currently true."""
        got = takeover_rate([(False, self._clean()) for _ in range(126)])
        assert got.rate is None
        assert "UNDEFINED" in got.verdict
        assert "not a takeover rate of zero" in got.verdict

    def test_a_zero_rate_admits_it_cannot_distinguish_dead_code(self) -> None:
        got = takeover_rate([(True, self._clean()) for _ in range(20)])
        assert got.rate == 0.0
        assert "dead code" in got.verdict

    def test_a_high_rate_says_the_desk_is_not_operating_unattended(self) -> None:
        got = takeover_rate([(True, self._raised()) for _ in range(20)])
        assert "not operating unattended" in got.verdict

    def test_causes_are_counted_per_trigger(self) -> None:
        outcomes = [
            (True, self._raised(Trigger.UNDERLYING_HALTED)),
            (True, self._raised(Trigger.UNDERLYING_HALTED)),
            (True, self._raised(Trigger.UNGROUNDED_FIGURE)),
            (True, self._clean()),
        ]
        got = takeover_rate(outcomes)
        assert got.by_trigger == {"underlying_halted": 2, "ungrounded_figure": 1}

    def test_an_escalation_on_a_non_actionable_decision_is_not_counted(self) -> None:
        """It could not have reached the venue, so no human was saved from anything."""
        got = takeover_rate([(False, self._raised()) for _ in range(10)])
        assert got.escalations == 0
        assert got.by_trigger == {}

    def test_the_dict_carries_both_numerator_and_denominator(self) -> None:
        got = takeover_rate([(True, self._raised()), (True, self._clean())]).as_dict()
        assert got["escalations"] == 1
        assert got["actionable_decisions"] == 2
        assert got["rate"] == 0.5

    def test_an_empty_record_is_undefined(self) -> None:
        assert takeover_rate([]).rate is None
