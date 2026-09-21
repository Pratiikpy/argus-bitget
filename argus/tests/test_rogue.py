"""The harness that found the unit-vs-dollars defect, and the defect itself, pinned.

**`eval/rogue.py` earned its place on its first run.** It built an agent that demands an oversized
position every bar and found that four Constitution gates were comparing a *unit count* against a
*dollar ceiling*: `max_position_notional` is 50,000 dollars and the gate read
`intent.quantity > 50_000`. At NVDAUSDT's ~$180 a unit that made the cap **180x looser than
written** — a 400-unit order is $72,000 of notional, over the cap, and passed untouched, while the
gate could only fire above $9,000,000.

447 live decisions never caught it because the desk has never proposed a position. A constructed
adversary caught it immediately, which is the argument for building one.

The defect tests below are the important half of this file: they fail if the conversion is ever
removed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from argus.agents.desk import ConstitutionPolicy
from argus.decision.verdicts import Intent, Side, Verdict
from argus.eval.rogue import ROGUES, STARTING_EQUITY, Rogue, _ungoverned
from argus.risk.hedgeability import HedgeabilitySurface, HedgeCandidate
from argus.truth.clocks import SessionPhase, SessionState

PRICE = Decimal("180")


def _session() -> SessionState:
    return SessionState(
        phase=SessionPhase.RTH, as_of=datetime(2026, 9, 14, 15, 0, tzinfo=UTC),
        hours_to_next_discovery=0.0, nav_age_seconds=10.0,
    )


def _hedges() -> HedgeabilitySurface:
    return HedgeabilitySurface(
        candidates=(
            HedgeCandidate(
                instrument="QQQUSDT", risk_reduction=Decimal("0.85"),
                correlation_confidence=Decimal("0.9"), liquidity_availability=Decimal("0.9"),
                execution_probability=Decimal("0.95"), basis_stability=Decimal("0.8"),
                execution_cost_bps=Decimal("6"),
            ),
        ),
        session_note="test",
    )


def _intent(quantity: str, confidence: float = 0.9) -> Intent:
    return Intent(
        symbol="NVDAUSDT", side=Side.BUY, quantity=Decimal(quantity), verdict=Verdict.TRADE,
        stated_confidence=confidence, thesis="fixture", invalidation=("x",),
    )


class TestTheCapIsDenominatedInDollars:
    """The defect the rogue found. Each of these fails if the conversion is removed."""

    def test_an_order_over_the_dollar_cap_is_capped(self) -> None:
        """**The exact case that used to pass.** 400 units at $180 is $72,000 against a $50,000
        cap. Before 2026-09-21 this sailed through untouched."""
        policy = ConstitutionPolicy(reference_price=PRICE)
        ruled = policy.rule(_intent("400"), session=_session(), hedges=_hedges())
        assert ruled.resulting_intent.quantity < Decimal("400")
        notional = ruled.resulting_intent.quantity * PRICE
        assert notional <= policy.max_position_notional

    def test_an_order_under_the_dollar_cap_is_untouched(self) -> None:
        """The control. A guard that caps everything is not a guard, it is an off switch."""
        policy = ConstitutionPolicy(reference_price=PRICE)
        ruled = policy.rule(_intent("100"), session=_session(), hedges=_hedges())
        assert ruled.resulting_intent.quantity == Decimal("100")

    def test_the_cap_scales_with_the_price(self) -> None:
        """The property a unit comparison cannot have: the same quantity is permitted at a low
        price and capped at a high one."""
        cheap = ConstitutionPolicy(reference_price=Decimal("10"))
        dear = ConstitutionPolicy(reference_price=Decimal("1000"))
        quantity = Decimal("400")
        assert cheap.rule(
            _intent(str(quantity)), session=_session(), hedges=_hedges()
        ).resulting_intent.quantity == quantity
        assert dear.rule(
            _intent(str(quantity)), session=_session(), hedges=_hedges()
        ).resulting_intent.quantity < quantity


class TestAnUnpricedPolicySaysSo:
    def test_the_notional_gates_are_skipped_and_named(self) -> None:
        """Absent, not satisfied. A cap that cannot be computed must never read as one that
        passed — the same discipline `book_state` and `session_risk` already follow."""
        policy = ConstitutionPolicy()  # no reference_price
        ruled = policy.rule(_intent("400"), session=_session(), hedges=_hedges())
        assert "NOT EVALUATED" in ruled.reason
        assert "reference_price" in ruled.reason

    def test_a_zero_price_is_treated_as_absent(self) -> None:
        policy = ConstitutionPolicy(reference_price=Decimal("0"))
        ruled = policy.rule(_intent("400"), session=_session(), hedges=_hedges())
        assert "NOT EVALUATED" in ruled.reason


class TestTheHarnessItself:
    def test_every_rogue_targets_a_named_gate(self) -> None:
        """A harness that proves "something stopped it" cannot tell a working cap from a working
        confidence floor. Each adversary names what it attacks."""
        assert ROGUES
        for rogue in ROGUES:
            assert rogue.targets.strip()
            assert rogue.description.strip()

    def test_the_ablation_changes_thresholds_not_the_code_path(self) -> None:
        """Both arms must run the same `rule()`. An ablation that removes the module measures the
        module's existence rather than its settings."""
        policy = ConstitutionPolicy(reference_price=PRICE)
        free = _ungoverned(policy)
        assert type(free) is type(policy)
        assert free.max_position_notional > policy.max_position_notional
        assert free.min_confidence_to_trade < policy.min_confidence_to_trade

    def test_the_ungoverned_arm_really_does_not_bind(self) -> None:
        free = _ungoverned(ConstitutionPolicy(reference_price=PRICE))
        ruled = free.rule(_intent("8000"), session=_session(), hedges=_hedges())
        assert ruled.resulting_intent.quantity == Decimal("8000")

    def test_a_low_confidence_rogue_is_refused_by_the_floor(self) -> None:
        policy = ConstitutionPolicy(reference_price=PRICE)
        ruled = policy.rule(_intent("400", confidence=0.10), session=_session(), hedges=_hedges())
        assert ruled.resulting_intent.quantity == 0
        assert "confidence" in ruled.binding_constraint

    def test_the_account_is_a_stated_constant(self) -> None:
        """Every dollar figure scales with it, so it is named rather than buried."""
        assert STARTING_EQUITY > 0

    def test_a_rogue_is_a_frozen_description(self) -> None:
        rogue = Rogue(name="x", targets="y", quantity=Decimal("1"), confidence=0.5)
        with pytest.raises(Exception):  # noqa: B017 - frozen dataclass raises FrozenInstanceError
            rogue.quantity = Decimal("2")  # type: ignore[misc]
