"""Restatement tests — anchored on cases whose answer is fixed before the code runs.

Calibration code is unusually easy to get wrong and unusually easy to believe, because every
statistic it produces is a small number between zero and one and they all look plausible. So each
test here pins a value that arithmetic determines rather than one the implementation produces: a
coin scores 0.25, a perfect forecaster scores 0, ForecastBench's index maps 0.25 to exactly 50, and
Murphy's identity reconstructs the Brier score from its own parts to floating-point zero.
"""

from __future__ import annotations

import random
from math import sqrt

import pytest

from argus.eval.forecastbench import (
    BINS,
    Bin,
    RestatementError,
    bin_forecasts,
    brier,
    brier_index,
    brier_skill_score,
    expected_calibration_error,
    maximum_calibration_error,
    murphy,
    restate,
)


def _coin(n: int = 400, *, seed: int = 1) -> tuple[list[float], list[bool]]:
    """A forecaster who says 0.5 forever, on outcomes that happen half the time."""
    rng = random.Random(seed)
    outcomes = [rng.random() < 0.5 for _ in range(n)]
    return [0.5] * n, outcomes


def _oracle(n: int = 400, *, seed: int = 2) -> tuple[list[float], list[bool]]:
    rng = random.Random(seed)
    outcomes = [rng.random() < 0.4 for _ in range(n)]
    return [1.0 if o else 0.0 for o in outcomes], outcomes


def _calibrated(n: int = 6000, *, seed: int = 3) -> tuple[list[float], list[bool]]:
    """Probabilities drawn on a grid, outcomes generated at exactly the stated rate.

    Perfect calibration by construction, so reliability and both calibration errors must come out
    at zero up to sampling noise, and skill against climatology must be positive.
    """
    rng = random.Random(seed)
    probabilities = [rng.choice([0.05, 0.25, 0.45, 0.65, 0.85, 0.95]) for _ in range(n)]
    return probabilities, [rng.random() < p for p in probabilities]


class TestTheBrierScoreItself:
    def test_a_coin_on_coin_flips_scores_a_quarter(self) -> None:
        assert brier(*_coin()) == pytest.approx(0.25)

    def test_perfect_foresight_scores_zero(self) -> None:
        assert brier(*_oracle()) == 0.0

    def test_confident_and_wrong_every_time_scores_one(self) -> None:
        assert brier([1.0, 1.0, 1.0], [False, False, False]) == 1.0

    def test_an_empty_record_raises_rather_than_scoring_zero(self) -> None:
        """Zero is the perfect score. Returning it for "no data" is the worst possible default."""
        with pytest.raises(RestatementError, match="undefined"):
            brier([], [])

    def test_a_mismatched_record_raises(self) -> None:
        with pytest.raises(RestatementError, match="every probability"):
            brier([0.5, 0.5], [True])


class TestTheForecastBenchIndex:
    def test_a_coin_maps_to_fifty(self) -> None:
        """The endpoint that pins the transform. forecastbench.org describes the scale as
        50% maximally uninformed, and only this transform puts 0.25 there."""
        assert brier_index(0.25) == pytest.approx(50.0)

    def test_perfect_maps_to_one_hundred(self) -> None:
        assert brier_index(0.0) == pytest.approx(100.0)

    def test_maximally_wrong_maps_to_zero(self) -> None:
        assert brier_index(1.0) == pytest.approx(0.0)

    def test_it_is_monotone_decreasing_in_the_brier_score(self) -> None:
        values = [brier_index(b) for b in (0.0, 0.05, 0.1, 0.2, 0.25, 0.4, 0.8, 1.0)]
        assert values == sorted(values, reverse=True)

    def test_it_is_the_square_root_transform_and_not_a_linear_one(self) -> None:
        """A linear 100*(1-BS) would also hit 100 and 0 at the endpoints but would put a coin at
        75, not 50. This is the test that tells the two apart."""
        assert brier_index(0.16) == pytest.approx(100.0 * (1.0 - sqrt(0.16)))
        assert brier_index(0.16) != pytest.approx(100.0 * (1.0 - 0.16))

    def test_a_score_outside_the_unit_interval_raises(self) -> None:
        with pytest.raises(RestatementError, match="outside"):
            brier_index(1.5)


class TestSkillScores:
    def test_matching_the_reference_is_zero_skill(self) -> None:
        assert brier_skill_score(0.25, 0.25) == 0.0

    def test_beating_it_is_positive_and_losing_is_negative(self) -> None:
        assert brier_skill_score(0.20, 0.25) > 0
        assert brier_skill_score(0.30, 0.25) < 0

    def test_a_perfect_reference_raises_rather_than_returning_infinity(self) -> None:
        with pytest.raises(RestatementError, match="division by zero"):
            brier_skill_score(0.1, 0.0)

    def test_the_climatology_reference_is_harder_than_the_coin_on_skewed_outcomes(self) -> None:
        """The reason three baselines are reported. At a 10% base rate, always saying 0.10 scores
        0.09 while a coin scores 0.25 — so skill against the coin flatters by a wide margin."""
        probabilities = [0.10] * 1000
        outcomes = [i < 100 for i in range(1000)]
        score = brier(probabilities, outcomes)
        assert brier_skill_score(score, 0.25) > brier_skill_score(score, 0.09)


class TestBinning:
    def test_a_probability_of_one_lands_in_the_top_bin_not_off_the_end(self) -> None:
        bins = bin_forecasts([1.0], [True])
        assert len(bins) == 1
        assert bins[0].lower == pytest.approx(0.9)

    def test_bins_partition_the_record(self) -> None:
        probabilities, outcomes = _calibrated(2000)
        assert sum(b.count for b in bin_forecasts(probabilities, outcomes)) == 2000

    def test_empty_bins_are_dropped(self) -> None:
        bins = bin_forecasts([0.05, 0.06, 0.95], [True, False, True])
        assert len(bins) == 2

    def test_a_probability_outside_the_unit_interval_raises(self) -> None:
        with pytest.raises(RestatementError, match="not in"):
            bin_forecasts([1.4], [True])

    def test_the_default_bin_count_is_the_stated_convention(self) -> None:
        assert BINS == 10


class TestMurphysIdentity:
    """``BS = REL - RES + UNC + WITHIN``, exactly, on any input.

    The three-term version without the within-bin term was written first and these tests refuted
    it: exact on six discrete probabilities, off by -0.0044 once the probabilities were continuous
    and by -0.0186 at two bins. An implementation that cannot rebuild its own input is not
    measuring what it claims to.
    """

    def test_it_reconstructs_the_brier_score_on_a_calibrated_record(self) -> None:
        probabilities, outcomes = _calibrated()
        assert restate(probabilities, outcomes).identity_residual == pytest.approx(0.0, abs=1e-12)

    def test_it_reconstructs_the_brier_score_on_a_miscalibrated_record(self) -> None:
        """The identity is not conditional on the forecaster being any good."""
        rng = random.Random(4)
        outcomes = [rng.random() < 0.3 for _ in range(3000)]
        overconfident = [0.9 if o else 0.85 for o in outcomes]
        overconfident = [min(1.0, p + rng.gauss(0, 0.05)) for p in overconfident]
        assert restate(overconfident, outcomes).identity_residual == pytest.approx(0.0, abs=1e-12)

    def test_it_reconstructs_the_brier_score_at_every_bin_count(self) -> None:
        probabilities, outcomes = _calibrated(3000)
        for bins in (2, 5, 10, 20, 50):
            residual = restate(probabilities, outcomes, bins=bins).identity_residual
            assert residual == pytest.approx(0.0, abs=1e-12), bins

    def test_uncertainty_is_the_base_rate_variance_and_nothing_else(self) -> None:
        probabilities, outcomes = _calibrated(4000)
        result = restate(probabilities, outcomes)
        _, _, uncertainty, _ = result.decomposition
        assert uncertainty == pytest.approx(result.base_rate * (1 - result.base_rate))

    def test_a_perfectly_calibrated_record_has_near_zero_reliability(self) -> None:
        reliability, _, _, _ = restate(*_calibrated()).decomposition
        assert reliability < 0.002, reliability

    def test_a_constant_forecast_has_zero_resolution(self) -> None:
        """Saying the same thing every time cannot discriminate, however well calibrated it is.
        This is the term that catches a forecaster who has simply memorised the base rate."""
        rng = random.Random(5)
        outcomes = [rng.random() < 0.37 for _ in range(4000)]
        _, resolution, _, _ = restate([0.37] * 4000, outcomes).decomposition
        assert resolution == pytest.approx(0.0, abs=1e-9)

    def test_a_discriminating_forecast_has_positive_resolution(self) -> None:
        _, resolution, _, _ = restate(*_calibrated()).decomposition
        assert resolution > 0.05

    def test_the_within_bin_term_vanishes_on_discrete_forecasts(self) -> None:
        """Murphy's original case: one bin per distinct forecast value. The correction term is
        exactly zero there, which is why the three-term identity looked right for so long."""
        rng = random.Random(21)
        probabilities = [rng.choice([0.05, 0.35, 0.65, 0.95]) for _ in range(3000)]
        outcomes = [rng.random() < p for p in probabilities]
        _, _, _, within = restate(probabilities, outcomes, bins=4).decomposition
        assert within == pytest.approx(0.0, abs=1e-12)

    def test_the_within_bin_term_is_non_zero_on_continuous_forecasts(self) -> None:
        """And this is the case that broke the three-term version. If this ever returns zero, the
        correction has been silently disabled and the identity is closing by luck."""
        rng = random.Random(22)
        probabilities = [rng.random() for _ in range(3000)]
        outcomes = [rng.random() < p for p in probabilities]
        _, _, _, within = restate(probabilities, outcomes).decomposition
        assert abs(within) > 1e-6

    def test_narrower_bins_shrink_the_within_term(self) -> None:
        """It is a binning artefact, so it must go to zero as the bins tighten around the values."""
        rng = random.Random(23)
        probabilities = [rng.random() for _ in range(4000)]
        outcomes = [rng.random() < p for p in probabilities]
        coarse = abs(restate(probabilities, outcomes, bins=2).decomposition[3])
        fine = abs(restate(probabilities, outcomes, bins=200).decomposition[3])
        assert fine < coarse

    def test_decomposing_nothing_raises(self) -> None:
        with pytest.raises(RestatementError, match="no forecasts"):
            murphy([], 0.5)


class TestCalibrationError:
    def test_a_perfectly_calibrated_record_has_near_zero_error(self) -> None:
        result = restate(*_calibrated())
        assert result.ece < 0.02
        assert result.mce < 0.06

    def test_the_maximum_is_never_below_the_expected(self) -> None:
        """MCE is a max over the same gaps ECE averages, so this holds by construction — and it is
        the cheapest possible check that the two are computed over the same bins."""
        for seed in range(6, 12):
            result = restate(*_calibrated(1500, seed=seed))
            assert result.mce >= result.ece

    def test_a_systematically_overconfident_record_shows_it(self) -> None:
        """Always claiming 0.9 on an event that happens 40% of the time is a 0.5 gap, and both
        statistics must report it rather than averaging it away."""
        rng = random.Random(12)
        outcomes = [rng.random() < 0.4 for _ in range(2000)]
        result = restate([0.9] * 2000, outcomes)
        assert result.ece > 0.4
        assert result.mce > 0.4

    def test_calibrating_nothing_raises(self) -> None:
        with pytest.raises(RestatementError, match="no forecasts"):
            expected_calibration_error([])
        with pytest.raises(RestatementError, match="no forecasts"):
            maximum_calibration_error([])

    def test_the_gap_is_unsigned(self) -> None:
        under = Bin(lower=0.0, upper=0.1, count=5, mean_probability=0.05, observed_rate=0.30)
        over = Bin(lower=0.0, upper=0.1, count=5, mean_probability=0.30, observed_rate=0.05)
        assert under.gap == pytest.approx(over.gap)


class TestTheRestatementRefusesWhatItCannotMeasure:
    def test_a_record_where_everything_happened_raises(self) -> None:
        """Uncertainty is zero, so skill against climatology is a division by zero. Returning a
        large skill score there would be the single most flattering bug available."""
        with pytest.raises(RestatementError, match="all cleared"):
            restate([0.6] * 50, [True] * 50)

    def test_a_record_where_nothing_happened_raises(self) -> None:
        with pytest.raises(RestatementError, match="none cleared"):
            restate([0.6] * 50, [False] * 50)

    def test_an_empty_record_raises(self) -> None:
        with pytest.raises(RestatementError, match="nothing to restate"):
            restate([], [])

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(RestatementError, match="every probability"):
            restate([0.5, 0.5], [True])


class TestTheVerdictCanComeOutAgainstUs:
    def test_a_base_rate_parrot_is_reported_as_having_no_skill(self) -> None:
        rng = random.Random(13)
        outcomes = [rng.random() < 0.42 for _ in range(3000)]
        verdict = restate([0.42] * 3000, outcomes).verdict
        assert "NO SKILL" in verdict or "WEAKLY DISCRIMINATING" in verdict

    def test_a_genuinely_discriminating_record_is_reported_as_skilled(self) -> None:
        assert "SKILL ESTABLISHED" in restate(*_calibrated()).verdict

    def test_the_comparability_note_refuses_to_claim_a_head_to_head(self) -> None:
        text = restate(*_calibrated()).comparability
        assert "not a rival score" in text
        assert "no leaderboard figure is quoted" in text

    def test_the_persistence_reference_is_scored_from_the_second_observation(self) -> None:
        """The first outcome has nothing before it. Giving it a default would hand the reference a
        free guess it did not earn."""
        result = restate(*_calibrated(1000))
        assert result.persistence_brier is not None
        assert result.skill_vs_persistence is not None

    def test_a_two_row_record_still_forms_a_persistence_reference(self) -> None:
        assert restate([0.5, 0.5], [True, False]).persistence_brier is not None

    def test_the_rendered_report_names_the_reference_that_counts(self) -> None:
        text = restate(*_calibrated()).render()
        assert "skill vs climatology" in text
        assert "the one that counts" in text
        assert "Murphy identity residual" in text

    def test_the_dict_carries_the_identity_residual_unrounded(self) -> None:
        """Rounding the residual to five places would make every implementation look correct."""
        got = restate(*_calibrated()).as_dict()
        assert abs(got["murphy_identity_residual"]) < 1e-12
