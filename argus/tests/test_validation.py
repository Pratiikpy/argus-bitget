"""Validation tests — the guards on the statistics that decide whether a result is believed.

Every test here corresponds to a way this layer could quietly pass a bad strategy. A validation
module that returns a comforting default when it cannot compute is worse than no validation module,
because it produces a number that looks like evidence. So the raising paths are tested as hard as
the arithmetic, and the two headline statistics are checked against cases whose answer is known
before the code runs: a procedure choosing pure noise must score near 0.5, and a procedure with a
real and persistent edge must score near 0.
"""

from __future__ import annotations

import random
from math import comb, sqrt
from statistics import NormalDist, fmean
from typing import ClassVar

import pytest

from argus.backtest.metrics import MetricError
from argus.backtest.validation import (
    DEFAULT_CONFIDENCE,
    MIN_OBSERVATIONS,
    OverfittingResult,
    annualised_track_record_years,
    benjamini_hochberg,
    bonferroni,
    min_track_record_length,
    probability_of_overfitting,
    purged_splits,
)


def _normal(n: int, *, mean: float, sd: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(mean, sd) for _ in range(n)]


def _noise_matrix(rows: int, strategies: int, *, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 0.01) for _ in range(strategies)] for _ in range(rows)]


class TestTrackRecordLengthRefusesTheQuestionsItCannotAnswer:
    def test_a_short_series_raises_rather_than_extrapolating(self) -> None:
        with pytest.raises(MetricError, match="below the"):
            min_track_record_length([0.01] * (MIN_OBSERVATIONS - 1))

    def test_a_flat_series_has_no_track_record(self) -> None:
        """A zero-variance series has an infinite Sharpe, not a computable one."""
        with pytest.raises(MetricError, match="zero-variance"):
            min_track_record_length([0.01] * 50)

    def test_a_losing_strategy_gets_no_length_at_all(self) -> None:
        """The tempting failure is to return a very large number, which implies a long enough
        record would rescue it. No length of record makes a negative edge significant."""
        losing = _normal(200, mean=-0.001, sd=0.01, seed=1)
        with pytest.raises(MetricError, match="does not exceed the benchmark"):
            min_track_record_length(losing)

    def test_a_strategy_that_merely_ties_the_benchmark_also_raises(self) -> None:
        flat = _normal(200, mean=0.0005, sd=0.01, seed=2)
        sharpe = fmean(flat) / (sum((r - fmean(flat)) ** 2 for r in flat) / (len(flat) - 1)) ** 0.5
        with pytest.raises(MetricError, match="does not exceed"):
            min_track_record_length(flat, benchmark_sharpe=sharpe)


class TestTrackRecordLengthArithmetic:
    def test_it_matches_the_closed_form_on_a_normal_sample(self) -> None:
        """With zero skew and kurtosis k, MinTRL is 1 + (1 - skew*S + (k-1)/4*S^2)(z/S)^2.
        Computed here from the sample's own moments so the test is an independent recomputation,
        not a restatement of the implementation."""
        returns = _normal(500, mean=0.0008, sd=0.01, seed=3)
        n = len(returns)
        mean = fmean(returns)
        sd = sqrt(sum((r - mean) ** 2 for r in returns) / (n - 1))
        s = mean / sd
        skew = sum(((r - mean) / sd) ** 3 for r in returns) / n
        kurt = sum(((r - mean) / sd) ** 4 for r in returns) / n
        z = NormalDist().inv_cdf(DEFAULT_CONFIDENCE)
        expected = 1.0 + (1.0 - skew * s + (kurt - 1.0) / 4.0 * s**2) * (z / s) ** 2
        assert min_track_record_length(returns) == pytest.approx(expected)

    def test_a_stronger_edge_needs_a_shorter_record(self) -> None:
        weak = min_track_record_length(_normal(400, mean=0.0004, sd=0.01, seed=4))
        strong = min_track_record_length(_normal(400, mean=0.0030, sd=0.01, seed=4))
        assert strong < weak

    def test_more_confidence_demands_a_longer_record(self) -> None:
        returns = _normal(400, mean=0.001, sd=0.01, seed=5)
        assert min_track_record_length(returns, confidence=0.99) > min_track_record_length(
            returns, confidence=0.90
        )

    def test_a_benchmark_to_beat_lengthens_the_record(self) -> None:
        returns = _normal(400, mean=0.001, sd=0.01, seed=6)
        assert min_track_record_length(returns, benchmark_sharpe=0.05) > min_track_record_length(
            returns
        )

    def test_a_fat_left_tail_lengthens_the_record(self) -> None:
        """Negative skew is exactly the shape that makes a short record flattering, so the
        adjustment must push the required length up, not leave it alone."""
        rng = random.Random(7)
        symmetric = [rng.gauss(0.001, 0.01) for _ in range(400)]
        skewed = list(symmetric)
        for i in range(0, 400, 40):
            skewed[i] = -0.05
        lifted = fmean(symmetric) - fmean(skewed)
        skewed = [r + lifted for r in skewed]  # same mean, worse shape
        assert min_track_record_length(skewed) > min_track_record_length(symmetric)

    def test_years_is_the_length_divided_by_the_frequency(self) -> None:
        returns = _normal(400, mean=0.001, sd=0.01, seed=8)
        hourly = 24 * 365
        assert annualised_track_record_years(
            returns, periods_per_year=hourly
        ) == pytest.approx(min_track_record_length(returns) / hourly)

    def test_the_frequency_argument_is_required(self) -> None:
        with pytest.raises(TypeError):
            annualised_track_record_years(_normal(400, mean=0.001, sd=0.01, seed=9))  # type: ignore[call-arg]


class TestCrossValidationRefusesMalformedInput:
    def test_odd_groups_cannot_be_split_in_half(self) -> None:
        with pytest.raises(MetricError, match="even number"):
            probability_of_overfitting(_noise_matrix(400, 4, seed=10), groups=7)

    def test_fewer_than_two_groups_is_not_a_split(self) -> None:
        with pytest.raises(MetricError, match="even number"):
            probability_of_overfitting(_noise_matrix(400, 4, seed=11), groups=0)

    def test_an_empty_matrix_raises(self) -> None:
        with pytest.raises(MetricError, match="no observations"):
            probability_of_overfitting([])

    def test_one_strategy_has_nothing_to_be_compared_against(self) -> None:
        """PBO is a statement about choosing between strategies. With one there is no choice, and
        returning 0.0 would read as 'this strategy is safe'."""
        with pytest.raises(MetricError, match="at least two"):
            probability_of_overfitting([[0.01] for _ in range(400)])

    def test_a_ragged_matrix_raises(self) -> None:
        rows = _noise_matrix(400, 4, seed=12)
        rows[17] = rows[17][:3]
        with pytest.raises(MetricError, match="every observation"):
            probability_of_overfitting(rows)

    def test_blocks_too_small_to_carry_a_sharpe_raise(self) -> None:
        rows = _noise_matrix(8 * MIN_OBSERVATIONS - 1, 4, seed=13)
        with pytest.raises(MetricError, match="cannot be split"):
            probability_of_overfitting(rows, groups=8)

    def test_the_same_matrix_passes_one_row_larger(self) -> None:
        rows = _noise_matrix(8 * MIN_OBSERVATIONS, 4, seed=13)
        assert 0.0 <= probability_of_overfitting(rows, groups=8).pbo <= 1.0


class TestCrossValidationIsExhaustiveNotSampled:
    def test_it_evaluates_every_balanced_split(self) -> None:
        result = probability_of_overfitting(_noise_matrix(400, 5, seed=14), groups=8)
        assert result.splits == comb(8, 4) == 70
        assert len(result.out_of_sample_ranks) == 70

    def test_more_groups_means_combinatorially_more_splits(self) -> None:
        result = probability_of_overfitting(_noise_matrix(400, 5, seed=15), groups=10)
        assert result.splits == comb(10, 5) == 252

    def test_every_rank_is_a_fraction(self) -> None:
        result = probability_of_overfitting(_noise_matrix(400, 6, seed=16), groups=8)
        assert all(0.0 <= r <= 1.0 for r in result.out_of_sample_ranks)

    def test_the_shape_is_reported_back(self) -> None:
        result = probability_of_overfitting(_noise_matrix(400, 6, seed=17), groups=8)
        assert result.observations == 400
        assert result.strategies == 6

    def test_the_result_is_deterministic(self) -> None:
        """Exhaustive enumeration, not a random sample of splits — so two runs on the same matrix
        must agree exactly, or the statistic depends on which splits were drawn."""
        rows = _noise_matrix(400, 5, seed=18)
        assert probability_of_overfitting(rows).pbo == probability_of_overfitting(rows).pbo


class TestCrossValidationAgainstCasesWhoseAnswerIsKnown:
    def test_pure_noise_scores_near_a_coin_toss(self) -> None:
        """The null. A single PBO estimate has a standard deviation of about 0.25 across
        independent datasets, which is why this averages six of them rather than asserting a band
        around one — the same reason `SINGLE_ESTIMATE_SD` exists."""
        estimates = [
            probability_of_overfitting(_noise_matrix(320, 8, seed=100 + i), groups=8).pbo
            for i in range(6)
        ]
        assert 0.25 <= fmean(estimates) <= 0.75, estimates

    def test_a_real_and_persistent_edge_scores_near_zero(self) -> None:
        rng = random.Random(200)
        rows = []
        for _ in range(320):
            row = [rng.gauss(0.0, 0.01) for _ in range(7)]
            row.append(rng.gauss(0.004, 0.01))  # the one strategy that works
            rows.append(row)
        assert probability_of_overfitting(rows, groups=8).pbo < 0.15

    def test_an_edge_that_stopped_working_scores_worse_than_one_that_did_not(self) -> None:
        """CSCV cuts contiguous blocks precisely so a regime change is visible. A strategy that
        worked in the first half and died in the second is the failure this statistic exists to
        catch, and shuffling rows would hide it."""
        rng = random.Random(300)
        persistent: list[list[float]] = []
        regime: list[list[float]] = []
        for t in range(320):
            noise = [rng.gauss(0.0, 0.01) for _ in range(7)]
            persistent.append([*noise, rng.gauss(0.004, 0.01)])
            drift = 0.008 if t < 160 else 0.0
            regime.append([*noise, rng.gauss(drift, 0.01)])
        assert (
            probability_of_overfitting(regime, groups=8).pbo
            > probability_of_overfitting(persistent, groups=8).pbo
        )

    def test_identical_strategies_sit_exactly_at_the_median(self) -> None:
        """Duplicate columns are not hypothetical: any two variants that never trade both return a
        flat zero series, and a sweep of twenty-five produces several. Scoring a tie by counting
        only strictly-worse strategies puts the whole tie at rank 0 and reports PBO 1.0 — reading
        as total overfitting when the truth is that the strategies are indistinguishable. The
        midrank puts them at 0.5, which is the honest answer."""
        column = _normal(320, mean=0.0005, sd=0.01, seed=400)
        rows = [[v, v, v, v] for v in column]
        result = probability_of_overfitting(rows, groups=8)
        assert all(r == pytest.approx(0.5) for r in result.out_of_sample_ranks)
        assert result.pbo == 0.0

    def test_a_duplicated_column_does_not_drag_the_score(self) -> None:
        """The same defect in its realistic form. Adding a copy of an existing strategy adds no
        information, so it must not move PBO by more than the statistic's own resolution.

        Averaged over six paired matrices because one comparison cannot resolve a shift this small.
        The residual is a discretisation effect and not a tie-handling one: with twelve strategies
        a chosen strategy can sit at eleven distinct ranks, none of them exactly 0.5, so the mass
        that would sit on the median boundary falls to one side. It shrinks as strategies are
        added — measured at +0.067 with five strategies, -0.046 with twelve and -0.032 with twenty
        — which is the signature of a boundary, not a bias in the rank itself."""
        shifts = []
        for seed in range(800, 806):
            rng = random.Random(seed)
            base = [[rng.gauss(0.0, 0.01) for _ in range(12)] for _ in range(320)]
            widened = [[*row, row[0]] for row in base]
            shifts.append(
                probability_of_overfitting(widened, groups=8).pbo
                - probability_of_overfitting(base, groups=8).pbo
            )
        assert abs(fmean(shifts)) < 0.10, shifts
        assert abs(fmean(shifts)) < OverfittingResult.SINGLE_ESTIMATE_SD


class TestTheResultStatesItsOwnPrecision:
    def _result(self, pbo: float, ranks: tuple[float, ...] = ()) -> OverfittingResult:
        return OverfittingResult(
            pbo=pbo, splits=70, strategies=8, observations=320, out_of_sample_ranks=ranks
        )

    def test_a_coin_toss_is_named_as_picking_noise(self) -> None:
        assert "picking noise" in self._result(0.5).verdict

    def test_a_middling_score_is_named_as_degrading(self) -> None:
        assert "degrades substantially" in self._result(0.3).verdict

    def test_a_low_score_is_the_only_one_that_survives(self) -> None:
        assert "survives out of sample" in self._result(0.05).verdict

    def test_the_reliability_line_reports_the_measured_dispersion(self) -> None:
        text = self._result(0.40).reliability
        assert "0.25" in text
        assert "0.15 to 0.65" in text

    def test_the_reliability_band_is_clipped_to_the_unit_interval(self) -> None:
        assert "0.00 to " in self._result(0.05).reliability
        assert " to 1.00" in self._result(0.95).reliability

    def test_the_median_rank_of_an_even_count_is_the_midpoint(self) -> None:
        assert self._result(0.5, (0.2, 0.4, 0.6, 0.8)).median_rank == pytest.approx(0.5)

    def test_the_median_rank_of_an_odd_count_is_the_middle_value(self) -> None:
        assert self._result(0.5, (0.9, 0.1, 0.4)).median_rank == pytest.approx(0.4)

    def test_no_ranks_gives_no_median(self) -> None:
        assert self._result(0.5).median_rank == 0.0

    def test_the_rendered_report_carries_the_caveat_with_the_number(self) -> None:
        text = self._result(0.42, (0.3, 0.5)).render()
        assert "42.0%" in text
        assert "70 balanced split" in text
        assert "standard deviation" in text

    def test_the_dict_carries_the_reliability_too(self) -> None:
        got = self._result(0.42, (0.3, 0.5)).as_dict()
        assert got["pbo"] == 0.42
        assert got["single_estimate_sd"] == OverfittingResult.SINGLE_ESTIMATE_SD
        assert "standard deviation" in got["reliability"]


class TestPurgingRemovesTheLeakAndNothingElse:
    def test_no_training_label_window_reaches_into_the_test_set(self) -> None:
        """The whole point. Recomputed here from the horizon rather than from the implementation:
        a row at i carries a label over [i, i+h], and that interval must not touch the test fold."""
        horizon = 12
        for train, test in purged_splits(300, folds=5, label_horizon=horizon):
            lo, hi = min(test), max(test)
            for i in train:
                assert i + horizon < lo or i > hi + horizon, (i, lo, hi)

    def test_a_naive_split_would_have_leaked(self) -> None:
        """Guards against the purge silently becoming a no-op: the rows it drops must be rows a
        plain k-fold would have trained on."""
        horizon = 12
        train, test = purged_splits(300, folds=5, label_horizon=horizon)[2]
        naive = [i for i in range(300) if i not in set(test)]
        assert set(train) < set(naive)
        assert len(naive) - len(train) == 2 * horizon

    def test_the_embargo_drops_a_further_gap_after_the_test_window(self) -> None:
        train, test = purged_splits(300, folds=5, label_horizon=0, embargo=10)[2]
        assert max(test) + 1 not in train
        assert max(test) + 10 not in train
        assert max(test) + 11 in train

    def test_the_embargo_is_one_sided(self) -> None:
        """Serial correlation leaks forward from the test window, so the embargo goes after it.
        Embargoing before as well would throw away training data for no reason."""
        no_embargo, _ = purged_splits(300, folds=5, label_horizon=0, embargo=0)[2]
        embargoed, test = purged_splits(300, folds=5, label_horizon=0, embargo=10)[2]
        dropped = set(no_embargo) - set(embargoed)
        assert dropped and all(i > max(test) for i in dropped)

    def test_with_no_horizon_and_no_embargo_it_is_plain_k_fold(self) -> None:
        for train, test in purged_splits(100, folds=5, label_horizon=0):
            assert sorted(train + test) == list(range(100))

    def test_the_test_folds_partition_the_sample(self) -> None:
        folds = purged_splits(103, folds=5, label_horizon=3)
        covered = [i for _, test in folds for i in test]
        assert sorted(covered) == list(range(103))

    def test_the_last_fold_absorbs_the_remainder(self) -> None:
        """103 into 5 leaves 3 bars over. Dropping them would quietly shorten the sample."""
        folds = purged_splits(103, folds=5)
        assert len(folds[-1][1]) == 23
        assert max(folds[-1][1]) == 102

    def test_folds_are_contiguous_in_time(self) -> None:
        for _, test in purged_splits(300, folds=6):
            assert test == list(range(min(test), max(test) + 1))

    def test_one_fold_is_not_cross_validation(self) -> None:
        with pytest.raises(MetricError, match="at least two folds"):
            purged_splits(100, folds=1)

    def test_more_folds_than_observations_raises(self) -> None:
        with pytest.raises(MetricError, match="cannot be split"):
            purged_splits(3, folds=5)

    def test_a_negative_horizon_or_embargo_raises(self) -> None:
        with pytest.raises(MetricError, match="cannot be negative"):
            purged_splits(100, folds=5, label_horizon=-1)
        with pytest.raises(MetricError, match="cannot be negative"):
            purged_splits(100, folds=5, embargo=-1)


class TestFalseDiscoveryControl:
    KNOWN: ClassVar[list[float]] = [0.001, 0.008, 0.02, 0.04, 0.2, 0.6]

    def test_the_worked_example(self) -> None:
        """Six p-values at a 5% false-discovery rate. Thresholds are rank/6 * 0.05, so the largest
        rank clearing its own bar is the third."""
        assert benjamini_hochberg(self.KNOWN) == [True, True, True, False, False, False]

    def test_bonferroni_on_the_same_values_rejects_fewer(self) -> None:
        assert bonferroni(self.KNOWN) == [True, True, False, False, False, False]

    def test_it_is_a_step_up_so_a_larger_p_can_carry_a_smaller_one(self) -> None:
        """0.026 fails its own threshold of 0.025 and is still rejected, because 0.030 clears the
        rank-2 threshold of 0.05. Testing each p-value independently would get this wrong."""
        assert benjamini_hochberg([0.026, 0.030]) == [True, True]

    def test_order_is_preserved_not_sorted(self) -> None:
        assert benjamini_hochberg([0.6, 0.001, 0.2, 0.008]) == [False, True, False, True]

    def test_every_bonferroni_rejection_is_also_a_benjamini_hochberg_rejection(self) -> None:
        rng = random.Random(500)
        for _ in range(50):
            p = [rng.random() ** 3 for _ in range(12)]
            bh = benjamini_hochberg(p)
            for kept, strict in zip(bh, bonferroni(p), strict=True):
                assert kept or not strict

    def test_a_looser_rate_rejects_at_least_as_much(self) -> None:
        loose = benjamini_hochberg(self.KNOWN, fdr=0.20)
        tight = benjamini_hochberg(self.KNOWN, fdr=0.01)
        assert sum(loose) >= sum(tight)

    def test_more_trials_makes_the_same_p_value_harder(self) -> None:
        """The reason this module exists: a p-value of 0.03 is a discovery on its own and noise
        among twenty-five variants."""
        assert benjamini_hochberg([0.03]) == [True]
        assert benjamini_hochberg([0.03, *[0.9] * 24]) == [False] * 25

    def test_all_null_p_values_reject_nothing(self) -> None:
        assert benjamini_hochberg([0.4, 0.5, 0.6, 0.9]) == [False] * 4
        assert bonferroni([0.4, 0.5, 0.6, 0.9]) == [False] * 4

    def test_an_empty_sweep_rejects_nothing(self) -> None:
        assert benjamini_hochberg([]) == []
        assert bonferroni([]) == []

    def test_a_p_value_outside_the_unit_interval_raises(self) -> None:
        for bad in ([1.2, 0.3], [-0.1, 0.3], [float("nan"), 0.3]):
            with pytest.raises(MetricError, match="finite number"):
                benjamini_hochberg(bad)
            with pytest.raises(MetricError, match="finite number"):
                bonferroni(bad)

    def test_bonferroni_divides_the_level_by_the_trial_count(self) -> None:
        assert bonferroni([0.05 / 6, 0.05 / 6 + 1e-9]) == [True, True]
        assert bonferroni([0.024, 0.026], alpha=0.05) == [True, False]
