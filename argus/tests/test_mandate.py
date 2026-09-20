"""Mandate injection tests.

The criterion is "personalized thesis", and `TraderProfile`'s own docstring states the test: two
profiles get different verdicts on identical market state. So the decisive property here is that a
mandate produces a *different answer*, not a smaller one — a position clipped after the fact still
rests on reasoning written for somebody else.

The second property is that personalisation never becomes a blindfold. Evidence outside a profile's
preferences is reordered, never removed: a trader who cannot see a news item cannot decline it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from argus.agents.mandate import HORIZON_SLACK, Mandate, frame_for, order_evidence
from argus.desk.workbench import TraderProfile
from argus.truth.evidence import Evidence

NOW = datetime(2026, 9, 14, 15, 0, tzinfo=UTC)

CONSERVATIVE = Mandate(TraderProfile.conservative())
AGGRESSIVE = Mandate(TraderProfile.aggressive())


def _evidence() -> list[Evidence]:
    return [
        Evidence(id="e1", claim="8-K filed", source="sec-edgar", available_at=NOW),
        Evidence(id="e2", claim="headline", source="news", available_at=NOW),
        Evidence(id="e3", claim="forum chatter", source="social", available_at=NOW),
        Evidence(id="e4", claim="10-Q", source="filing", available_at=NOW),
    ]


class TestTwoProfilesDiverge:
    """The stated test for this criterion."""

    def test_a_short_horizon_thesis_is_in_mandate_for_one_and_not_the_other(self) -> None:
        small = Decimal("1000")
        assert AGGRESSIVE.out_of_mandate(horizon_hours=36, notional=small) == ()
        assert CONSERVATIVE.out_of_mandate(horizon_hours=36, notional=small) == ()

    def test_a_long_horizon_thesis_breaches_the_event_trader_only(self) -> None:
        small = Decimal("1000")
        assert CONSERVATIVE.out_of_mandate(horizon_hours=600, notional=small) == ()
        breaches = AGGRESSIVE.out_of_mandate(horizon_hours=600, notional=small)
        assert breaches and "different trader's trade" in breaches[0]

    def test_the_position_ceilings_differ(self) -> None:
        assert CONSERVATIVE.max_position_notional == Decimal("5000")
        assert AGGRESSIVE.max_position_notional == Decimal("25000")

    def test_a_notional_can_be_in_mandate_for_one_and_out_for_the_other(self) -> None:
        size = Decimal("12000")
        assert AGGRESSIVE.out_of_mandate(horizon_hours=24, notional=size) == ()
        assert CONSERVATIVE.out_of_mandate(horizon_hours=24, notional=size)

    def test_the_two_mandate_lines_are_materially_different_text(self) -> None:
        """If the frame were the same, the reasoning could not diverge."""
        assert CONSERVATIVE.render() != AGGRESSIVE.render()
        assert "conservative income" in CONSERVATIVE.render()
        assert "aggressive event trader" in AGGRESSIVE.render()


class TestOutOfMandateIsDeclineNotResize:
    def test_the_horizon_breach_says_decline_rather_than_resize(self) -> None:
        text = AGGRESSIVE.render()
        assert "declined rather than resized" in text

    def test_every_breach_is_reported_not_just_the_first(self) -> None:
        breaches = CONSERVATIVE.out_of_mandate(horizon_hours=5_000, notional=Decimal("90000"))
        assert len(breaches) == 2

    def test_a_trade_inside_the_mandate_reports_nothing(self) -> None:
        assert CONSERVATIVE.out_of_mandate(horizon_hours=100, notional=Decimal("100")) == ()


class TestTheHorizonSlack:
    def test_a_rounding_error_is_not_a_policy_breach(self) -> None:
        """A 50-hour thesis for a 48-hour trader is not out of mandate."""
        assert AGGRESSIVE.permits_horizon(50)

    def test_a_materially_longer_thesis_still_breaches(self) -> None:
        assert not AGGRESSIVE.permits_horizon(48 * HORIZON_SLACK + 1)

    def test_the_ceiling_is_stated_in_the_frame(self) -> None:
        assert f"{AGGRESSIVE.horizon_ceiling_hours:.0f}h is out of mandate" in AGGRESSIVE.render()


class TestPersonalisationIsNotABlindfold:
    def test_nothing_is_removed(self) -> None:
        """A trader who cannot see a news item cannot decline it."""
        ordered = order_evidence(_evidence(), mandate=CONSERVATIVE)
        assert len(ordered) == 4
        assert {e.id for e, _ in ordered} == {"e1", "e2", "e3", "e4"}

    def test_preferred_sources_come_first(self) -> None:
        ordered = order_evidence(_evidence(), mandate=CONSERVATIVE)
        assert [e.source for e, _ in ordered][:2] == ["sec-edgar", "filing"]

    def test_the_two_profiles_order_the_same_evidence_differently(self) -> None:
        conservative = [e.id for e, _ in order_evidence(_evidence(), mandate=CONSERVATIVE)]
        aggressive = [e.id for e, _ in order_evidence(_evidence(), mandate=AGGRESSIVE)]
        assert conservative != aggressive

    def test_non_preferred_evidence_is_marked_rather_than_hidden(self) -> None:
        _, rendered = frame_for(CONSERVATIVE, _evidence())
        marked = [line for line in rendered if "outside this mandate" in line]
        assert len(marked) == 2, "news and social are outside the conservative preference"

    def test_the_frame_says_the_ordering_is_not_a_judgement_of_truth(self) -> None:
        assert "less often actionable" in CONSERVATIVE.render()
        assert "not because it is less true" in CONSERVATIVE.render()

    def test_a_profile_with_no_preference_keeps_the_original_order(self) -> None:
        plain = Mandate(
            TraderProfile(
                name="unopinionated", capital=Decimal("10000"),
                max_position_pct=Decimal("10"), max_sector_pct=Decimal("30"),
                holding_horizon_hours=100, loss_tolerance_pct=Decimal("5"),
            )
        )
        ordered = [e.id for e, _ in order_evidence(_evidence(), mandate=plain)]
        assert ordered == ["e1", "e2", "e3", "e4"]

    def test_an_empty_evidence_set_does_not_raise(self) -> None:
        assert order_evidence([], mandate=CONSERVATIVE) == ()


class TestTheFrameIsUsable:
    def test_it_returns_the_mandate_line_and_the_evidence(self) -> None:
        line, rendered = frame_for(AGGRESSIVE, _evidence())
        assert line.startswith("MANDATE —")
        assert len(rendered) == 4

    def test_the_mandate_states_the_numbers_it_binds_with(self) -> None:
        line = CONSERVATIVE.render()
        for fragment in ("100000", "5%", "5000", "720h", "3%"):
            assert fragment in line, f"{fragment} missing from the mandate frame"

    def test_it_serialises_for_the_decision_record(self) -> None:
        payload = CONSERVATIVE.as_dict()
        assert payload["profile"] == "conservative income"
        assert payload["max_position_notional"] == "5000"
        assert payload["preferred_evidence"] == ["filing", "sec-edgar"]

    @pytest.mark.parametrize("mandate", [CONSERVATIVE, AGGRESSIVE])
    def test_every_profile_produces_a_non_empty_frame(self, mandate: Mandate) -> None:
        assert len(mandate.render()) > 100
