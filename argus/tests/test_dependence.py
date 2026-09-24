"""Dependence tests — each anchored on a process whose true answer is known in closed form.

An autocorrelation estimator, a bootstrap and a bandwidth rule all produce plausible-looking
numbers from any input, so none of them can be checked by inspection. Every test here generates a
process whose answer is determined analytically before the code runs: an AR(1) with parameter phi
has autocorrelations exactly phi^k, a geometric draw with parameter p has mean 1/p, an independent
series has a Newey-West correction of one, and a field of strategies that only differ by noise must
fail a reality check against its own benchmark.
"""

from __future__ import annotations

import random
from math import sqrt
from statistics import fmean

import pytest

from argus.backtest.dependence import (
    DEFAULT_SEED,
    MIN_OBSERVATIONS,
    _geometric,
    auto_lag_truncation,
    autocorrelations,
    bootstrap_sharpe,
    hac_sharpe,
    moving_block_indices,
    newey_west_eta,
    optimal_block_length,
    reality_check,
    stationary_bootstrap_indices,
    suggested_block_length,
)
from argus.backtest.metrics import MetricError


def _ar1(n: int, phi: float, *, seed: int, drift: float = 0.0, sd: float = 1.0) -> list[float]:
    """An AR(1) process. Its population autocorrelation at lag k is exactly phi^k."""
    rng = random.Random(seed)
    x = 0.0
    out = []
    for _ in range(n):
        x = phi * x + rng.gauss(0.0, sd)
        out.append(x + drift)
    return out


def _iid(n: int, *, seed: int, mean: float = 0.0, sd: float = 1.0) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(mean, sd) for _ in range(n)]


class TestTheBandwidthRule:
    def test_it_matches_the_newey_west_formula(self) -> None:
        for n in (100, 252, 1000, 4318):
            assert auto_lag_truncation(n) == int(4.0 * (n / 100.0) ** (2.0 / 9.0))

    def test_the_documented_values_are_the_real_ones(self) -> None:
        """The docstring quotes 9 at 4,318 hourly bars and 4 at 252 daily. A docstring that quotes
        a number the function does not produce is the defect the doc-claims checker exists for, and
        this one was wrong on first writing — it said 6."""
        assert auto_lag_truncation(4318) == 9
        assert auto_lag_truncation(252) == 4

    def test_it_grows_with_the_sample(self) -> None:
        values = [auto_lag_truncation(n) for n in (100, 500, 2000, 10_000, 100_000)]
        assert values == sorted(values)

    def test_a_tiny_sample_gets_no_lags_rather_than_a_fractional_one(self) -> None:
        assert auto_lag_truncation(9) == 0

    def test_the_block_length_suggestion_is_the_square_root(self) -> None:
        assert suggested_block_length(4318) == pytest.approx(sqrt(4318))

    def test_the_optimal_rule_reproduces_the_reference_implementation(self) -> None:
        """Four processes, checked against `arch.bootstrap.optimal_block_length` by running both.
        These are the values that implementation returns, to three decimals, and they are pinned
        here so a refactor of ours cannot drift from it silently."""
        expected = {
            0.0: (1.127, 1.290),
            0.3: (11.440, 13.096),
            0.6: (21.311, 24.395),
            0.85: (32.743, 37.482),
        }
        for phi, (stationary, circular) in expected.items():
            got = optimal_block_length(_ar1(2000, phi, seed=42, sd=1.0))
            assert got[0] == pytest.approx(stationary, abs=0.002), phi
            assert got[1] == pytest.approx(circular, abs=0.002), phi

    def test_more_autocorrelation_demands_a_longer_block(self) -> None:
        lengths = [optimal_block_length(_ar1(2000, phi, seed=42))[0] for phi in (0.0, 0.3, 0.6)]
        assert lengths == sorted(lengths)

    def test_white_noise_asks_for_blocks_of_about_one(self) -> None:
        """An independent series has no dependence to preserve, so the optimal block is a single
        observation and the block bootstrap collapses to an ordinary one."""
        assert optimal_block_length(_iid(2000, seed=77))[0] < 3.0

    def test_the_circular_block_is_always_the_longer_of_the_two(self) -> None:
        """Their variance expansions differ by a constant factor of 3/4 under the cube root, so
        the ordering is fixed. If it ever reversed, the two constants have been swapped."""
        for phi in (0.0, 0.4, 0.8):
            stationary, circular = optimal_block_length(_ar1(1500, phi, seed=9))
            assert circular > stationary

    def test_the_block_is_capped_below_a_third_of_the_sample(self) -> None:
        """A block a third of the sample long leaves three blocks, which is not a resample."""
        n = 400
        assert optimal_block_length(_ar1(n, 0.97, seed=11))[0] <= n / 3 + 1

    def test_a_constant_series_falls_back_rather_than_raising(self) -> None:
        """White noise has no optimal block length and neither does a flat line. Refusing to
        bootstrap at all would be worse than using blocks of one."""
        got = optimal_block_length([0.01] * 500)
        assert got[0] == pytest.approx(suggested_block_length(500))

    def test_a_short_series_raises(self) -> None:
        with pytest.raises(MetricError, match="below the"):
            optimal_block_length(_iid(MIN_OBSERVATIONS - 1, seed=12))

    def test_the_bootstrap_uses_the_rule_rather_than_the_folklore(self) -> None:
        """On an AR(0.6) series of 2,000 points sqrt(n) is 44.7 and the rule gives 21.3, so a
        bootstrap still defaulting to the folklore is roughly twice too long."""
        series = _ar1(2000, 0.6, seed=42, drift=0.02)
        assert bootstrap_sharpe(series, resamples=50).block_length == pytest.approx(
            optimal_block_length(series)[0]
        )


class TestAutocorrelationAgainstAKnownProcess:
    @pytest.mark.parametrize("phi", [0.0, 0.3, 0.6, -0.4])
    def test_an_ar1_gives_phi_to_the_k(self, phi: float) -> None:
        rhos = autocorrelations(_ar1(200_000, phi, seed=7), 4)
        for k, rho in enumerate(rhos, start=1):
            assert rho == pytest.approx(phi**k, abs=0.02), (phi, k)

    def test_an_independent_series_has_autocorrelations_near_zero(self) -> None:
        assert all(abs(r) < 0.02 for r in autocorrelations(_iid(50_000, seed=8), 5))

    def test_no_lags_asks_for_nothing(self) -> None:
        assert autocorrelations(_iid(100, seed=9), 0) == []

    def test_a_lag_past_the_sample_gives_zero_not_an_error(self) -> None:
        assert autocorrelations(_iid(40, seed=10), 60)[-1] == 0.0

    def test_a_constant_series_raises(self) -> None:
        with pytest.raises(MetricError, match="constant series"):
            autocorrelations([0.01] * 200, 4)

    def test_a_one_observation_series_raises(self) -> None:
        with pytest.raises(MetricError, match="at least two"):
            autocorrelations([0.01], 4)


class TestTheNeweyWestCorrection:
    def test_no_lags_is_no_correction(self) -> None:
        assert newey_west_eta([]) == 1.0

    def test_an_independent_series_corrects_by_about_one(self) -> None:
        assert newey_west_eta(autocorrelations(_iid(50_000, seed=11), 6)) == pytest.approx(
            1.0, abs=0.1
        )

    def test_positive_autocorrelation_inflates_the_variance(self) -> None:
        """The direction that matters: it means the naive interval was too narrow."""
        assert newey_west_eta(autocorrelations(_ar1(20_000, 0.6, seed=12), 6)) > 1.5

    def test_negative_autocorrelation_deflates_it(self) -> None:
        """A mean-reverting series makes the naive interval conservative, not optimistic."""
        assert newey_west_eta(autocorrelations(_ar1(20_000, -0.5, seed=13), 6)) < 1.0

    def test_the_bartlett_weights_decay_linearly_to_the_truncation(self) -> None:
        """Recomputed from the kernel rather than from the implementation."""
        rhos = [0.5, 0.4, 0.3]
        expected = 1.0 + 2.0 * (
            (1 - 1 / 4) * 0.5 + (1 - 2 / 4) * 0.4 + (1 - 3 / 4) * 0.3
        )
        assert newey_west_eta(rhos) == pytest.approx(expected)

    def test_the_last_lag_is_weighted_least(self) -> None:
        assert newey_west_eta([0.0, 0.0, 0.4]) < newey_west_eta([0.4, 0.0, 0.0])


class TestTheHacSharpeInterval:
    def test_on_an_independent_series_the_two_errors_nearly_agree(self) -> None:
        result = hac_sharpe(_iid(20_000, seed=14, mean=0.02))
        assert result.widening == pytest.approx(1.0, abs=0.1)

    def test_on_an_autocorrelated_series_the_honest_interval_is_wider(self) -> None:
        result = hac_sharpe(_ar1(5_000, 0.6, seed=15, drift=0.05))
        assert result.widening > 1.5
        assert result.se_hac > result.se_iid

    def test_the_interval_is_centred_on_the_point_estimate(self) -> None:
        result = hac_sharpe(_ar1(2_000, 0.4, seed=16, drift=0.03))
        assert (result.low + result.high) / 2 == pytest.approx(result.sharpe)

    def test_the_sharpe_is_per_observation_not_annualised(self) -> None:
        """Annualising before the interval invites comparing an hourly figure with a daily one."""
        returns = _iid(2_000, seed=17, mean=0.01, sd=1.0)
        assert hac_sharpe(returns).sharpe == pytest.approx(fmean(returns) / 1.0, abs=0.02)

    def test_a_wider_confidence_level_gives_a_wider_interval(self) -> None:
        returns = _ar1(2_000, 0.3, seed=18, drift=0.04)
        tight = hac_sharpe(returns, alpha=0.10)
        wide = hac_sharpe(returns, alpha=0.01)
        assert (wide.high - wide.low) > (tight.high - tight.low)

    def test_it_names_the_case_where_only_the_naive_interval_is_significant(self) -> None:
        """The whole reason for the module. A drift chosen so the independent interval clears zero
        and the corrected one does not."""
        found = False
        for seed in range(30, 90):
            result = hac_sharpe(_ar1(800, 0.7, seed=seed, drift=0.075))
            if "SIGNIFICANT ONLY IF THE RETURNS ARE ASSUMED INDEPENDENT" in result.verdict:
                found = True
                assert not result.excludes_zero
                assert abs(result.sharpe) > 1.96 * result.se_iid
                break
        assert found, "no seed produced the case the verdict is written for"

    def test_a_short_series_raises_rather_than_correcting_nothing(self) -> None:
        with pytest.raises(MetricError, match="below the"):
            hac_sharpe(_iid(MIN_OBSERVATIONS - 1, seed=19))

    def test_a_constant_series_raises(self) -> None:
        with pytest.raises(MetricError, match="constant series"):
            hac_sharpe([0.01] * 200)

    def test_the_lag_count_can_be_overridden_and_is_reported(self) -> None:
        assert hac_sharpe(_iid(1_000, seed=20, mean=0.02), lags=3).lags == 3

    def test_zero_lags_reproduces_the_independent_standard_error_exactly(self) -> None:
        result = hac_sharpe(_ar1(1_000, 0.5, seed=21, drift=0.03), lags=0)
        assert result.eta == 1.0
        assert result.se_hac == pytest.approx(result.se_iid)


class TestTheGeometricDraw:
    def test_its_mean_is_one_over_p(self) -> None:
        rng = random.Random(22)
        for target in (2.0, 10.0, 50.0):
            draws = [_geometric(rng, 1.0 / target) for _ in range(100_000)]
            assert fmean(draws) == pytest.approx(target, rel=0.05), target

    def test_it_never_returns_less_than_one(self) -> None:
        rng = random.Random(23)
        assert min(_geometric(rng, 0.9) for _ in range(10_000)) >= 1

    def test_certainty_gives_blocks_of_one(self) -> None:
        assert _geometric(random.Random(24), 1.0) == 1

    def test_an_impossible_probability_raises(self) -> None:
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(MetricError, match="block probability"):
                _geometric(random.Random(25), bad)


class TestTheStationaryBootstrap:
    def test_a_resample_is_the_length_of_the_original(self) -> None:
        for n in (50, 137, 1000):
            got = stationary_bootstrap_indices(n, block_length=10.0, rng=random.Random(26))
            assert len(got) == n

    def test_every_index_is_in_range(self) -> None:
        got = stationary_bootstrap_indices(200, block_length=8.0, rng=random.Random(27))
        assert all(0 <= i < 200 for i in got)

    def test_the_wrap_reaches_the_tail_of_the_series(self) -> None:
        """Without circular wrapping the last observations are drawn less often than the first,
        which biases every statistic toward the start of the sample."""
        counts = [0] * 60
        rng = random.Random(28)
        for _ in range(400):
            for i in stationary_bootstrap_indices(60, block_length=12.0, rng=rng):
                counts[i] += 1
        assert min(counts) > 0.6 * fmean(counts), counts

    def test_it_preserves_serial_correlation_that_a_shuffle_would_destroy(self) -> None:
        """The point of blocks. A resample of an AR(1) must still look autocorrelated."""
        series = _ar1(4_000, 0.7, seed=29)
        rng = random.Random(30)
        blocked = [series[i] for i in stationary_bootstrap_indices(
            len(series), block_length=50.0, rng=rng
        )]
        shuffled = list(series)
        rng.shuffle(shuffled)
        assert autocorrelations(blocked, 1)[0] > 0.4
        assert abs(autocorrelations(shuffled, 1)[0]) < 0.1

    def test_a_longer_block_preserves_more_of_it(self) -> None:
        series = _ar1(4_000, 0.7, seed=31)
        rng = random.Random(32)
        short = [series[i] for i in stationary_bootstrap_indices(
            len(series), block_length=2.0, rng=rng
        )]
        long = [series[i] for i in stationary_bootstrap_indices(
            len(series), block_length=100.0, rng=rng
        )]
        assert autocorrelations(long, 1)[0] > autocorrelations(short, 1)[0]

    def test_it_is_reproducible_from_the_seed(self) -> None:
        a = stationary_bootstrap_indices(100, block_length=9.0, rng=random.Random(33))
        b = stationary_bootstrap_indices(100, block_length=9.0, rng=random.Random(33))
        assert a == b

    def test_a_degenerate_request_raises(self) -> None:
        with pytest.raises(MetricError, match="fewer than two"):
            stationary_bootstrap_indices(1, block_length=5.0, rng=random.Random(34))
        with pytest.raises(MetricError, match="mean block length"):
            stationary_bootstrap_indices(50, block_length=0.0, rng=random.Random(34))

    def test_the_moving_block_variant_uses_a_fixed_length(self) -> None:
        """The comparison case. Fixed blocks of length 1 are an ordinary bootstrap and must destroy
        the autocorrelation the stationary version keeps."""
        series = _ar1(3_000, 0.7, seed=35)
        rng = random.Random(36)
        single = [series[i] for i in moving_block_indices(len(series), block_length=1, rng=rng)]
        blocked = [series[i] for i in moving_block_indices(len(series), block_length=60, rng=rng)]
        assert abs(autocorrelations(single, 1)[0]) < 0.1
        assert autocorrelations(blocked, 1)[0] > 0.4

    def test_the_moving_block_rejects_a_zero_length(self) -> None:
        with pytest.raises(MetricError, match="at least one"):
            moving_block_indices(50, block_length=0, rng=random.Random(37))


class TestTheBootstrapSharpeInterval:
    def test_it_brackets_the_point_estimate(self) -> None:
        result = bootstrap_sharpe(_iid(1_000, seed=38, mean=0.05), resamples=300)
        assert result.low <= result.statistic <= result.high

    def test_a_clear_edge_produces_an_interval_that_excludes_zero(self) -> None:
        assert bootstrap_sharpe(_iid(2_000, seed=39, mean=0.10), resamples=300).excludes_zero

    def test_no_edge_produces_an_interval_that_covers_zero(self) -> None:
        assert not bootstrap_sharpe(_iid(2_000, seed=40, mean=0.0), resamples=300).excludes_zero

    def test_it_is_reproducible_from_the_seed(self) -> None:
        returns = _iid(500, seed=41, mean=0.04)
        first = bootstrap_sharpe(returns, resamples=200, seed=7)
        second = bootstrap_sharpe(returns, resamples=200, seed=7)
        assert (first.low, first.high) == (second.low, second.high)

    def test_a_different_seed_gives_a_different_interval(self) -> None:
        """If it did not, the seed is not reaching the resampler."""
        returns = _iid(500, seed=42, mean=0.04)
        assert bootstrap_sharpe(returns, resamples=200, seed=1).low != bootstrap_sharpe(
            returns, resamples=200, seed=2
        ).low

    def test_the_default_seed_is_fixed_so_a_published_interval_can_be_rechecked(self) -> None:
        assert isinstance(DEFAULT_SEED, int)
        returns = _iid(400, seed=43, mean=0.04)
        assert bootstrap_sharpe(returns, resamples=150).low == bootstrap_sharpe(
            returns, resamples=150, seed=DEFAULT_SEED
        ).low

    def test_a_short_series_raises(self) -> None:
        with pytest.raises(MetricError, match="below the"):
            bootstrap_sharpe(_iid(MIN_OBSERVATIONS - 1, seed=44), resamples=50)

    def test_a_constant_series_raises(self) -> None:
        with pytest.raises(MetricError, match="constant series"):
            bootstrap_sharpe([0.02] * 200, resamples=50)


class TestTheRealityCheck:
    def _field(self, n: int, strategies: int, *, seed: int, edge: float = 0.0
               ) -> tuple[list[list[float]], list[float]]:
        rng = random.Random(seed)
        benchmark = [rng.gauss(0.0005, 0.01) for _ in range(n)]
        field = [[b + rng.gauss(0.0, 0.01) for b in benchmark] for _ in range(strategies - 1)]
        field.append([b + rng.gauss(edge, 0.01) for b in benchmark])
        return field, benchmark

    def test_a_field_of_noise_fails_to_beat_its_own_benchmark(self) -> None:
        """The null, and the case that matters: twenty strategies that differ from buy-and-hold by
        noise alone must not produce a significant winner, however good the best one looks."""
        field, benchmark = self._field(600, 20, seed=45)
        assert reality_check(field, benchmark, resamples=400).p_value > 0.10

    def test_a_planted_edge_is_found_and_identified(self) -> None:
        field, benchmark = self._field(600, 20, seed=46, edge=0.004)
        result = reality_check(field, benchmark, resamples=400)
        assert result.p_value < 0.05
        assert result.best_index == len(field) - 1

    def test_the_verdict_refuses_to_claim_an_edge_it_did_not_find(self) -> None:
        field, benchmark = self._field(600, 20, seed=47)
        assert "NOT ESTABLISHED" in reality_check(field, benchmark, resamples=400).verdict

    def test_more_strategies_make_the_same_winner_harder_to_believe(self) -> None:
        """The multiple-comparison payload. The identical best strategy must score a larger
        p-value when it was chosen from a wider field."""
        rng = random.Random(48)
        n = 500
        benchmark = [rng.gauss(0.0005, 0.01) for _ in range(n)]
        pool = [[b + rng.gauss(0.0, 0.01) for b in benchmark] for _ in range(40)]
        narrow = reality_check(pool[:3], benchmark, resamples=400, seed=5)
        wide = reality_check(pool, benchmark, resamples=400, seed=5)
        assert wide.p_value >= narrow.p_value

    def test_it_is_reproducible_from_the_seed(self) -> None:
        field, benchmark = self._field(400, 10, seed=49)
        assert reality_check(field, benchmark, resamples=200, seed=3).p_value == reality_check(
            field, benchmark, resamples=200, seed=3
        ).p_value

    def test_an_empty_field_raises(self) -> None:
        with pytest.raises(MetricError, match="no strategies"):
            reality_check([], _iid(100, seed=50))

    def test_a_ragged_field_raises(self) -> None:
        """Scoring one strategy over a different window than the benchmark compares two things
        that never happened together."""
        benchmark = _iid(100, seed=51)
        with pytest.raises(MetricError, match="same periods"):
            reality_check([_iid(99, seed=52)], benchmark)

    def test_a_short_window_raises(self) -> None:
        with pytest.raises(MetricError, match="below the"):
            reality_check([_iid(20, seed=53)], _iid(20, seed=54))
