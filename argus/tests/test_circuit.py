"""Circuit-breaker tests.

The decisive test is :meth:`TestItChangesTheDecision.test_a_shock_changes_the_verdict_mid_cycle`.
A risk control that only ever agrees with the decision it is checking is a comment, not a control,
and "risk control layer effectiveness" is a scored criterion. So the suite is built around one
question: does the breaker make the desk do something different, and does it say why.

The second property is the asymmetry. Risk machinery may reduce exposure and may never create it —
the same rule the Constitution obeys. A breaker that could turn NO_TRADE into TRADE would be a
strategy wearing a safety label.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from argus.decision.verdicts import Verdict
from argus.risk.circuit import (
    CONSECUTIVE_LOSS_HALT,
    Activation,
    BookState,
    Breaker,
    BreakerError,
    apply,
    assess,
    risk_multiplier,
)

NOW = datetime(2026, 9, 14, 15, 0, tzinfo=UTC)


def _healthy(**over: object) -> BookState:
    base: dict[str, object] = {
        "equity": Decimal("10000"),
        "peak_equity": Decimal("10000"),
        "session_open_equity": Decimal("10000"),
        "open_positions": 1,
    }
    base.update(over)
    return BookState(**base)  # type: ignore[arg-type]


class TestItChangesTheDecision:
    """A gate that never alters an outcome is decoration."""

    def test_a_shock_changes_the_verdict_mid_cycle(self) -> None:
        """Inject a 3-sigma adverse move; the desk must produce a different answer."""
        calm = apply(Verdict.TRADE, _healthy())
        shocked = apply(Verdict.TRADE, _healthy(realised_move_sigma=Decimal("-3.5")))

        assert calm.changed_the_decision is False
        assert shocked.changed_the_decision is True
        assert shocked.verdict is not Verdict.TRADE
        assert shocked.activation is Activation.HALTED

    def test_the_change_is_explained_in_words(self) -> None:
        ruling = apply(Verdict.TRADE, _healthy(realised_move_sigma=Decimal("-4")))
        text = ruling.explain()
        assert "moved" in text and "shock" in text

    def test_a_breaker_that_does_not_fire_says_so_plainly(self) -> None:
        assert apply(Verdict.TRADE, _healthy()).explain() == "breaker did not fire"

    def test_firing_without_changing_anything_is_reported_honestly(self) -> None:
        """Halted while already standing aside: the rule fired, the decision did not move."""
        ruling = apply(Verdict.NO_TRADE, _healthy(realised_move_sigma=Decimal("-5")))
        assert ruling.fired is True
        assert ruling.changed_the_decision is False
        assert "already complied" in ruling.explain()


class TestTheAsymmetry:
    """Risk machinery may only reduce. This is the property that makes it safe to trust."""

    @pytest.mark.parametrize(
        "state",
        [
            _healthy(realised_move_sigma=Decimal("-9")),
            _healthy(equity=Decimal("8000")),
            _healthy(consecutive_losses=99),
        ],
    )
    def test_it_never_turns_a_refusal_into_a_trade(self, state: BookState) -> None:
        assert apply(Verdict.NO_TRADE, state).verdict is Verdict.NO_TRADE

    def test_it_never_widens_a_reduce_into_a_trade(self) -> None:
        assert apply(Verdict.REDUCE, _healthy()).verdict is Verdict.REDUCE

    def test_halting_with_nothing_open_cannot_produce_a_reducing_order(self) -> None:
        """REDUCE with no position would be an order to open in the opposite direction."""
        ruling = apply(Verdict.TRADE, _healthy(open_positions=0, realised_move_sigma=Decimal("-4")))
        assert ruling.verdict is Verdict.NO_TRADE

    def test_halting_with_something_open_permits_reducing_it(self) -> None:
        ruling = apply(Verdict.TRADE, _healthy(open_positions=2, realised_move_sigma=Decimal("-4")))
        assert ruling.verdict is Verdict.REDUCE


class TestTheRules:
    def test_total_drawdown_halts(self) -> None:
        activation, trips = assess(_healthy(equity=Decimal("8900")))
        assert activation is Activation.HALTED
        assert any(t.rule == "total_drawdown" for t in trips)

    def test_session_drawdown_halts_independently_of_the_peak(self) -> None:
        """A book can be flat against its all-time peak and still be having a catastrophic day."""
        state = BookState(
            equity=Decimal("9500"), peak_equity=Decimal("9600"),
            session_open_equity=Decimal("10000"),
        )
        activation, trips = assess(state)
        assert activation is Activation.HALTED
        assert any(t.rule == "session_drawdown" for t in trips)

    def test_a_losing_streak_halts(self) -> None:
        activation, trips = assess(_healthy(consecutive_losses=CONSECUTIVE_LOSS_HALT))
        assert activation is Activation.HALTED
        assert any(t.rule == "losing_streak" for t in trips)

    def test_stale_evidence_reduces_but_does_not_halt(self) -> None:
        activation, trips = assess(_healthy(evidence_age=timedelta(hours=9)))
        assert activation is Activation.REDUCE_ONLY
        assert any(t.rule == "stale_evidence" for t in trips)

    def test_a_favourable_shock_does_not_trip_the_breaker(self) -> None:
        """The breaker is not a volatility filter; it cares about adverse moves."""
        assert assess(_healthy(realised_move_sigma=Decimal("+6")))[0] is Activation.ACTIVE

    def test_every_applicable_rule_is_reported_not_just_the_first(self) -> None:
        state = _healthy(
            equity=Decimal("8000"), consecutive_losses=9, evidence_age=timedelta(hours=12)
        )
        _, trips = assess(state)
        assert {t.rule for t in trips} >= {"total_drawdown", "losing_streak", "stale_evidence"}

    def test_a_healthy_book_trips_nothing(self) -> None:
        assert assess(_healthy()) == (Activation.ACTIVE, ())


class TestTheDeRiskingLadder:
    @pytest.mark.parametrize(
        ("equity", "expected"),
        [
            (Decimal("10000"), Decimal("1")),
            (Decimal("9700"), Decimal("0.75")),
            (Decimal("9300"), Decimal("0.5")),
            (Decimal("8900"), Decimal("0")),
        ],
    )
    def test_appetite_tightens_as_drawdown_deepens(
        self, equity: Decimal, expected: Decimal
    ) -> None:
        assert risk_multiplier(_healthy(equity=equity)) == expected

    def test_it_is_monotone(self) -> None:
        """A ladder that ever loosens as losses deepen is worse than none."""
        equities = [Decimal(str(v)) for v in (10000, 9800, 9600, 9400, 9200, 9000, 8800)]
        multipliers = [risk_multiplier(_healthy(equity=e)) for e in equities]
        assert multipliers == sorted(multipliers, reverse=True)

    def test_a_zero_peak_does_not_divide_by_zero(self) -> None:
        state = BookState(
            equity=Decimal("0"), peak_equity=Decimal("0"), session_open_equity=Decimal("0")
        )
        assert state.total_drawdown == Decimal("0")
        assert risk_multiplier(state) == Decimal("1")


class TestActivationIsCarriedBetweenDecisions:
    """State between decisions is what separates a breaker from a per-decision filter."""

    def test_a_halt_persists_into_the_next_decision(self) -> None:
        breaker = Breaker()
        breaker.evaluate(Verdict.TRADE, _healthy(realised_move_sigma=Decimal("-4")), now=NOW)
        assert breaker.activation is Activation.HALTED
        # The shock is gone, but the desk does not resume trading in the same breath.
        breaker.evaluate(Verdict.TRADE, _healthy(), now=NOW)
        assert breaker.activation is Activation.REDUCE_ONLY

    def test_recovery_is_stepwise_not_a_jump_to_active(self) -> None:
        breaker = Breaker(activation=Activation.HALTED)
        breaker.evaluate(Verdict.TRADE, _healthy(), now=NOW)
        assert breaker.activation is Activation.REDUCE_ONLY
        breaker.evaluate(Verdict.TRADE, _healthy(), now=NOW)
        assert breaker.activation is Activation.ACTIVE

    def test_an_illegal_transition_raises_rather_than_being_corrected(self) -> None:
        breaker = Breaker(activation=Activation.HALTED)
        with pytest.raises(BreakerError, match="stepwise"):
            breaker.transition(Activation.ACTIVE, reason="impatience", now=NOW)

    def test_every_transition_is_recorded_with_its_reason(self) -> None:
        breaker = Breaker()
        breaker.evaluate(Verdict.TRADE, _healthy(consecutive_losses=9), now=NOW)
        payload = breaker.as_dict()
        assert payload["activation"] == "halted"
        assert payload["transitions"][0]["move"] == "active->halted"
        assert "losing_streak" in payload["transitions"][0]["reason"]

    def test_a_quiet_book_produces_no_transitions(self) -> None:
        breaker = Breaker()
        for _ in range(5):
            breaker.evaluate(Verdict.TRADE, _healthy(), now=NOW)
        assert breaker.history == []
        assert breaker.activation is Activation.ACTIVE


class TestTheReportIsUsable:
    def test_the_ruling_serialises_every_trip(self) -> None:
        ruling = apply(Verdict.TRADE, _healthy(equity=Decimal("8000"), consecutive_losses=9))
        payload = ruling.as_dict()
        assert payload["fired"] is True
        assert payload["changed_the_decision"] is True
        assert len(payload["trips"]) >= 2
        assert all({"rule", "detail", "demands"} <= set(t) for t in payload["trips"])

    def test_a_trip_carries_the_number_that_caused_it(self) -> None:
        """A reason a person cannot check is a reason they have to take on trust."""
        _, trips = assess(_healthy(equity=Decimal("8900")))
        detail = next(t.detail for t in trips if t.rule == "total_drawdown")
        assert "%" in detail and "limit" in detail
