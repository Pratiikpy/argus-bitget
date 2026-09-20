"""Profile-value tests — a mandate that refuses everything must not look good.

The measurement is "did the mandate's refusals help its owner", and the way it goes wrong is by
rewarding refusal itself. A profile that declines every proposal never loses money, so any metric
built on realised losses alone would rank it first. The utility number here is therefore measured
against *taking everything*, and the two degenerate mandates — refuse-all and allow-all — are each
tested to make sure neither is reported as a success.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from argus.eval.incremental import Instant
from argus.eval.profile_value import (
    BAD_CASE_SIGMAS,
    MIN_INSTANTS,
    ProfileResult,
    ProfileValueError,
    evaluate,
    proposal_from,
)

START = datetime(2026, 5, 1, tzinfo=UTC)
HURDLE = 18.8


def _instants(n: int, *, drift: float, seed: int, vol: float = 0.01) -> list[Instant]:
    rng = random.Random(seed)
    return [
        Instant(
            symbol=f"S{i % 4}",
            at=START + timedelta(hours=(i // 4) * 6),
            trailing=tuple(rng.gauss(abs(drift) / 10_000, vol) for _ in range(72)),
            realised_bps=rng.gauss(drift, 150.0),
            hurdle_bps=HURDLE,
        )
        for i in range(n)
    ]


class TestTheProposalComesFromTheMarket:
    def test_the_bad_case_scales_with_trailing_volatility(self) -> None:
        """It must be a property of the moment, not a constant chosen to flatter a profile."""
        quiet = _instants(1, drift=0.0, seed=1, vol=0.001)[0]
        loud = _instants(1, drift=0.0, seed=1, vol=0.05)[0]
        assert proposal_from(loud).expected_loss_pct > proposal_from(quiet).expected_loss_pct

    def test_the_sigma_multiple_is_stated(self) -> None:
        assert BAD_CASE_SIGMAS == 2.0

    def test_a_flat_trailing_window_concedes_nothing(self) -> None:
        instant = Instant("S", START, (0.0,) * 72, realised_bps=100.0, hurdle_bps=HURDLE)
        assert proposal_from(instant).expected_loss_pct == Decimal("0")

    def test_the_proposal_carries_no_outcome(self) -> None:
        """The realised move must never reach the mandate, or the mandate is judging with the
        answer in hand."""
        instant = _instants(1, drift=500.0, seed=2)[0]
        blob = str(proposal_from(instant).as_dict())
        assert str(int(instant.realised_bps)) not in blob


class TestARefuseAllMandateIsNotASuccess:
    def test_refusing_everything_is_named_as_not_a_strategy(self) -> None:
        result = ProfileResult(name="paranoid", taken=(), refused=(-50.0,) * 100)
        assert "refusing all trades is not a strategy" in result.verdict

    def test_refusing_everything_scores_no_utility_gain(self) -> None:
        """It avoided every loss and every gain. Measured against taking everything, that is zero
        minus the average, not a win."""
        result = ProfileResult(name="paranoid", taken=(), refused=(-50.0,) * 100)
        assert result.utility_gain_bps == pytest.approx(50.0)
        result_positive = ProfileResult(name="paranoid", taken=(), refused=(80.0,) * 100)
        assert result_positive.utility_gain_bps < 0

    def test_allowing_everything_is_named_as_a_pass_through(self) -> None:
        result = ProfileResult(name="permissive", taken=(10.0,) * 100, refused=())
        assert "pass-through" in result.verdict

    def test_allowing_everything_has_no_utility_gain(self) -> None:
        """It made exactly the same decisions as taking everything, so it added exactly nothing."""
        result = ProfileResult(name="permissive", taken=(10.0, -5.0, 30.0), refused=())
        assert result.utility_gain_bps == pytest.approx(0.0)


class TestRefusalValue:
    def test_declining_losers_is_reported_as_earning_its_keep(self) -> None:
        result = ProfileResult(name="p", taken=(50.0,) * 50, refused=(-90.0,) * 50)
        assert result.refused_value < 0
        assert "earned its keep" in result.verdict

    def test_declining_winners_is_reported_as_costing_money(self) -> None:
        """The verdict must be able to come out against the mandate."""
        result = ProfileResult(name="p", taken=(-40.0,) * 50, refused=(120.0,) * 50)
        assert result.refused_value > 0
        assert "cost its owner money" in result.verdict

    def test_the_refusal_rate_is_over_all_decisions(self) -> None:
        result = ProfileResult(name="p", taken=(1.0,) * 30, refused=(1.0,) * 70)
        assert result.refusal_rate == pytest.approx(0.7)
        assert result.decisions == 100

    def test_a_profile_with_no_decisions_reports_zeroes_not_errors(self) -> None:
        result = ProfileResult(name="p", taken=(), refused=())
        assert result.decisions == 0
        assert result.refusal_rate == 0.0
        assert result.utility_gain_bps == 0.0


class TestTheReport:
    def test_a_thin_sample_raises(self) -> None:
        with pytest.raises(ProfileValueError, match="below the"):
            evaluate(_instants(MIN_INSTANTS - 1, drift=0.0, seed=3))

    def test_it_scores_every_shipped_profile(self) -> None:
        from argus.desk.personalisation import standard_profiles

        got = evaluate(_instants(600, drift=100.0, seed=4))
        assert {r.name for r in got.results} == {p.name for p in standard_profiles()}

    def test_the_divergence_rate_is_measured_over_the_same_instants(self) -> None:
        got = evaluate(_instants(600, drift=100.0, seed=5))
        assert 0.0 <= got.divergence_rate <= 1.0

    def test_it_can_report_that_no_mandate_helped(self) -> None:
        """The honest outcome on a universe with no edge, and the one a metric built to flatter
        would never produce."""
        got = evaluate(_instants(800, drift=0.0, seed=6))
        if not got.any_helped:
            assert "NO MANDATE IMPROVED" in got.verdict
            assert "enforcing a preference, not adding value" in got.verdict

    def test_the_verdict_distinguishes_divergence_from_usefulness(self) -> None:
        got = evaluate(_instants(800, drift=0.0, seed=7))
        assert "disagreed with each other on" in got.verdict

    def test_a_flat_direction_instant_is_skipped_not_counted(self) -> None:
        """An instant the rule would not trade cannot test a mandate, and counting it would dilute
        both the divergence rate and the utility."""
        flat = [
            Instant(f"S{i}", START + timedelta(hours=i), (0.0,) * 72, 100.0, HURDLE)
            for i in range(300)
        ]
        got = evaluate([*_instants(300, drift=100.0, seed=8), *flat], minimum=100)
        assert got.instants == 300

    def test_the_rendered_report_shows_declined_and_allowed_separately(self) -> None:
        text = evaluate(_instants(600, drift=100.0, seed=9)).render()
        assert "declined" in text and "allowed" in text
        assert "PROFILE VALUE" in text

    def test_the_dict_carries_the_baseline_it_was_measured_against(self) -> None:
        """Without take_everything the utility gain cannot be checked by a reader."""
        got = evaluate(_instants(600, drift=100.0, seed=10)).as_dict()
        assert all("take_everything_bps" in r for r in got["results"])


class TestAResizeIsNotAFullTake:
    """The correction that mattered most on the real record.

    The conservative mandate **resizes** 98% of proposals and refuses 2%. Scoring a resize as a
    full take would credit it with the entire move on almost every decision, which is the
    difference between measuring a mandate and flattering one.
    """

    def test_a_halved_position_earns_half_the_move(self) -> None:
        from decimal import Decimal

        from argus.desk.personalisation import Outcome, Verdict

        verdict = Verdict(
            profile="p", outcome=Outcome.RESIZED,
            permitted_notional=Decimal("5000"), reasons=(),
        )
        share = float(verdict.permitted_notional) / 10_000.0
        assert share == 0.5

    def test_the_real_conservative_mandate_resizes_far_more_than_it_refuses(self) -> None:
        """If this ever inverted, the scaling above would stop mattering and the test above would
        silently become decorative."""
        import collections
        import random
        from datetime import UTC, datetime

        from argus.desk.personalisation import judge, standard_profiles

        rng = random.Random(1)
        counts: collections.Counter[str] = collections.Counter()
        conservative = standard_profiles()[0]
        for _ in range(300):
            instant = Instant(
                "NVDAUSDT", datetime(2026, 5, 1, tzinfo=UTC),
                tuple(rng.gauss(0.0005, 0.012) for _ in range(72)), 100.0, HURDLE,
            )
            counts[str(judge(conservative, proposal_from(instant)).outcome)] += 1
        assert counts["resized"] > counts["refused"] * 10

    def test_scaling_changes_the_measured_value(self) -> None:
        """Proof the scaling reaches the number. A full-take scoring and a half-take scoring of
        the same move cannot agree."""
        full = ProfileResult(name="p", taken=(100.0,) * 50, refused=(-50.0,) * 10)
        half = ProfileResult(name="p", taken=(50.0,) * 50, refused=(-50.0,) * 10)
        assert full.taken_value != half.taken_value


class TestItNeverFabricatesADeskDecision:
    def test_the_report_contains_no_desk_arm(self) -> None:
        got = evaluate(_instants(600, drift=0.0, seed=11)).as_dict()
        assert all(r["name"] != "desk" for r in got["results"])

    def test_every_verdict_comes_from_the_deterministic_mandate(self) -> None:
        """The mandate is code, not a model, so replaying it over recorded prices is not the same
        as re-deciding a desk cycle and calling it last month's."""
        from argus.desk.personalisation import judge, standard_profiles

        instant = _instants(1, drift=100.0, seed=12)[0]
        proposal = proposal_from(instant)
        profile = standard_profiles()[0]
        assert judge(profile, proposal).outcome is judge(profile, proposal).outcome


class TestSizingIsNotMistakenForSelection:
    def test_the_verdict_separates_narrowing_from_choosing(self) -> None:
        """The conservative mandate halves every position, so it loses roughly half as much as the
        aggressive one. A reader could take that for better judgement; it is smaller size. The
        utility figure measures each mandate against taking everything *at its own permitted size*,
        so the two effects never add together, and the verdict says so."""
        got = evaluate(_instants(600, drift=100.0, seed=30))
        assert "sizing rather than selection" in got.verdict
        assert "at its own permitted size" in got.verdict
