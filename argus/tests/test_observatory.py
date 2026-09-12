"""Observatory tests — the eight capabilities absent from every harness we tore down."""

from __future__ import annotations

from decimal import Decimal

import pytest

from argus.eval.observatory import (
    AbstentionOutcome,
    ContributionLedger,
    ModelScorecard,
    PitProbe,
    Prediction,
    ReplayResult,
    abstention_quality,
    brier_score,
    expected_calibration_error,
    leaderboard,
    pit_integrity,
    reliability_curve,
)


class TestCalibration:
    """Neither ECE nor Brier appears anywhere in the four evaluation repos in the corpus."""

    def test_perfect_confidence_perfectly_realised_scores_zero(self) -> None:
        preds = [Prediction(1.0, True)] * 10
        assert brier_score(preds) == pytest.approx(0.0)
        assert expected_calibration_error(preds) == pytest.approx(0.0)

    def test_a_coin_flip_claimed_as_certainty_scores_worst(self) -> None:
        preds = [Prediction(1.0, i % 2 == 0) for i in range(10)]
        assert brier_score(preds) == pytest.approx(0.5)

    def test_ece_catches_the_overconfident_model(self) -> None:
        """90% confident, lands 60% of the time — the gap sizing must not ignore."""
        preds = [Prediction(0.9, i < 6) for i in range(10)]
        assert expected_calibration_error(preds) == pytest.approx(0.3, abs=0.01)

    def test_a_well_calibrated_model_scores_near_zero(self) -> None:
        preds = [Prediction(0.7, i < 7) for i in range(10)]
        assert expected_calibration_error(preds) < 0.05

    def test_reliability_curve_exposes_the_bin(self) -> None:
        curve = reliability_curve([Prediction(0.9, i < 6) for i in range(10)])
        assert curve[0]["stated"] == pytest.approx(0.9)
        assert curve[0]["realised"] == pytest.approx(0.6)

    def test_empty_input_raises_rather_than_returning_zero(self) -> None:
        with pytest.raises(ValueError):
            brier_score([])


class TestReplayConsistency:
    def test_identical_verdicts_are_stable(self) -> None:
        r = ReplayResult("h", ("no_trade",) * 5, (Decimal("0"),) * 5)
        assert r.verdict_consistency == 1.0
        assert r.is_stable

    def test_a_flip_on_identical_input_is_a_defect(self) -> None:
        r = ReplayResult("h", ("trade", "no_trade", "trade", "no_trade"), (Decimal("0"),) * 4)
        assert r.verdict_consistency == 0.5
        assert r.is_stable is False

    def test_wild_sizing_is_unstable_even_when_the_side_agrees(self) -> None:
        """Same side, sizes from 10 to 200, is one decision and then a guess."""
        r = ReplayResult(
            "h", ("trade",) * 4,
            (Decimal("10"), Decimal("200"), Decimal("35"), Decimal("180")),
        )
        assert r.verdict_consistency == 1.0
        assert r.size_dispersion > 0.25
        assert r.is_stable is False


class TestAbstentionValue:
    """A NO_TRADE is not automatically good."""

    def test_avoiding_a_loss_scores_positive(self) -> None:
        o = AbstentionOutcome("d", Decimal("-40"), "BUY")
        assert o.value_bps == Decimal("52")   # avoided -40 move and the 12bps fee
        assert o.was_right

    def test_missing_a_gain_scores_negative(self) -> None:
        o = AbstentionOutcome("d", Decimal("60"), "BUY")
        assert o.value_bps == Decimal("-48")
        assert o.was_right is False

    def test_a_move_smaller_than_the_fee_makes_abstention_correct(self) -> None:
        """The measured reality: most moves do not clear 12bps, so standing aside usually wins."""
        o = AbstentionOutcome("d", Decimal("8"), "BUY")
        assert o.was_right

    def test_group_scoring_splits_avoided_from_missed(self) -> None:
        got = abstention_quality([
            AbstentionOutcome("a", Decimal("-40"), "BUY"),
            AbstentionOutcome("b", Decimal("60"), "BUY"),
        ])
        assert got["abstentions"] == 2
        assert got["correct_pct"] == 50.0


class TestContributionAttribution:
    def test_an_analyst_that_never_changes_a_decision_does_not_earn_its_place(self) -> None:
        led = ContributionLedger()
        for i in range(6):
            led.record(decision_id=f"d{i}", analyst="technical",
                       signal="neutral", was_pivotal=False, correct=True)
        assert led.by_analyst()["technical"]["earns_its_place"] is False

    def test_a_pivotal_and_accurate_analyst_earns_its_place(self) -> None:
        led = ContributionLedger()
        for i in range(6):
            led.record(decision_id=f"d{i}", analyst="earnings",
                       signal="bearish", was_pivotal=True, correct=i < 5)
        got = led.by_analyst()["earnings"]
        assert got["pivotal_rate_pct"] == 100.0
        assert got["earns_its_place"] is True

    def test_pivotal_but_wrong_does_not_earn_its_place(self) -> None:
        led = ContributionLedger()
        for i in range(6):
            led.record(decision_id=f"d{i}", analyst="sentiment",
                       signal="bullish", was_pivotal=True, correct=i < 2)
        assert led.by_analyst()["sentiment"]["earns_its_place"] is False


class TestPitIntegrity:
    def test_one_leak_fails_the_whole_suite(self) -> None:
        """90% is a failure — a single successful injection invalidates every prior result."""
        got = pit_integrity([
            PitProbe("future_price", True, "refused"),
            PitProbe("future_filing", True, "refused"),
            PitProbe("restated_figure", False, "returned the restated value"),
        ])
        assert got["integrity_holds"] is False
        assert got["leaked"] == ["restated_figure"]

    def test_all_rejected_holds(self) -> None:
        got = pit_integrity([PitProbe("a", True, ""), PitProbe("b", True, "")])
        assert got["integrity_holds"] is True


class TestLeaderboard:
    def test_ranked_by_calibration_not_return(self) -> None:
        """On a venue where the fee exceeds most effects, knowing what you do not know beats a
        lucky quarter — and calibration cannot be reached by taking more risk."""
        lucky = ModelScorecard(
            "lucky", predictions=[Prediction(0.95, i < 5) for i in range(10)],
            net_pnl_bps=Decimal("500"),
        )
        honest = ModelScorecard(
            "honest", predictions=[Prediction(0.6, i < 6) for i in range(10)],
            net_pnl_bps=Decimal("20"),
        )
        board = leaderboard([lucky, honest])
        assert board["leaderboard"][0]["model"] == "honest"

    def test_no_model_grades_another(self) -> None:
        board = leaderboard([ModelScorecard("m", predictions=[Prediction(0.5, True)])])
        assert "no model grades another" in board["note"]
