"""Metric tests — each guards a bug found in a real library.

Track 1 is scored purely on these numbers. A wrong metric here is worse than a wrong strategy,
because it makes a bad strategy look good and nobody checks the arithmetic.
"""

from __future__ import annotations

import random

import pytest

from argus.backtest.metrics import (
    DAILY_PER_YEAR,
    HOURLY_PER_YEAR,
    MetricError,
    deflated_sharpe,
    evaluate,
    max_drawdown,
    out_of_sample_decay,
    probabilistic_sharpe,
    rolling_sharpe,
    sharpe,
    sortino,
    stability,
    turnover,
)


class TestAnnualisation:
    """The frequency factor is where Sharpe is quietly inflated."""

    def test_periods_per_year_is_required(self) -> None:
        with pytest.raises(TypeError):
            sharpe([0.01, -0.005, 0.02])  # type: ignore[call-arg]

    def test_hourly_annualised_as_daily_inflates_by_about_five(self) -> None:
        r = [0.001, -0.0005, 0.002, 0.0001, -0.001] * 20
        as_daily = sharpe(r, periods_per_year=DAILY_PER_YEAR)
        as_hourly = sharpe(r, periods_per_year=HOURLY_PER_YEAR)
        assert as_hourly / as_daily == pytest.approx(5.9, abs=0.2)

    def test_zero_variance_raises_rather_than_returning_infinity(self) -> None:
        with pytest.raises(MetricError, match="zero-variance"):
            sharpe([0.01] * 10, periods_per_year=DAILY_PER_YEAR)


class TestSortino:
    def test_downside_deviation_divides_by_full_sample(self) -> None:
        """Dividing by the count of losing periods inflates the ratio exactly when losses are
        rare — which is when the inflation misleads most."""
        r = [0.01] * 9 + [-0.01]
        got = sortino(r, periods_per_year=DAILY_PER_YEAR)
        # dd = sqrt(0.0001/10) = 0.00316; mean = 0.008 -> 2.53 * sqrt(252)
        assert got == pytest.approx(2.53 * (DAILY_PER_YEAR ** 0.5), rel=0.02)

    def test_no_downside_raises(self) -> None:
        with pytest.raises(MetricError, match="no downside"):
            sortino([0.01, 0.02, 0.03], periods_per_year=DAILY_PER_YEAR)


class TestDrawdown:
    def test_denominator_is_the_running_peak_not_the_start(self) -> None:
        """Using the start understates drawdown for anything that made money first."""
        equity = [1.0, 2.0, 1.0]
        assert max_drawdown(equity) == pytest.approx(0.5)

    def test_monotonic_rise_has_no_drawdown(self) -> None:
        assert max_drawdown([1.0, 1.1, 1.2]) == 0.0

    def test_empty_curve_raises(self) -> None:
        with pytest.raises(MetricError):
            max_drawdown([])


class TestTurnover:
    def test_holding_a_position_costs_nothing(self) -> None:
        """Qlib's gross-notional measure charges for simply holding, overstating cost 2-3x on
        mean-reversion."""
        assert turnover([0.5, 0.5, 0.5, 0.5]) == 0.0

    def test_only_changes_are_counted(self) -> None:
        assert turnover([0.0, 1.0, 0.0]) == pytest.approx(2.0)

    def test_single_weight_is_zero(self) -> None:
        assert turnover([1.0]) == 0.0


class TestDeflatedSharpe:
    """The gate that consumes the trial count."""

    def test_zero_trials_is_refused(self) -> None:
        """An unrecorded trial count voids the gate — the count IS part of the result."""
        with pytest.raises(MetricError, match="voids the gate"):
            deflated_sharpe(2.0, n=500, trials=0, variance_of_trials=0.5)

    def test_never_returns_nan(self) -> None:
        """vectorbt's own formula returns NaN on a single trial — verified empirically against the
        isolated formula (`var_sharpe = np.var([x], ddof=1)` is NaN, 0/0), not asserted from memory.

        The risk this guards against is narrower than an earlier version of this docstring said: a
        gate admitting on ``metric > threshold`` fails CLOSED on NaN (``NaN > x`` is ``False``). It
        is the inverted, negated form — ``not (metric < threshold)`` — that fails OPEN, because
        ``NaN < x`` is also ``False`` and the ``not`` flips it to ``True``. Either shape is a
        plausible real gate; :func:`deflated_sharpe` makes the whole question moot by raising.
        """
        got = deflated_sharpe(2.0, n=500, trials=100, variance_of_trials=0.5)
        assert got == got  # NaN != NaN
        assert 0.0 <= got <= 1.0

    def test_more_trials_makes_the_same_sharpe_less_credible(self) -> None:
        """The whole point: searching harder must raise the bar."""
        few = deflated_sharpe(2.0, n=500, trials=2, variance_of_trials=0.5)
        many = deflated_sharpe(2.0, n=500, trials=1000, variance_of_trials=0.5)
        assert many < few

    def test_a_single_trial_has_nothing_to_deflate(self) -> None:
        one = deflated_sharpe(1.5, n=500, trials=1, variance_of_trials=0.5)
        assert one == pytest.approx(probabilistic_sharpe(1.5, benchmark=0.0, n=500))

    def test_negative_variance_is_refused(self) -> None:
        with pytest.raises(MetricError, match="cannot be negative"):
            deflated_sharpe(2.0, n=500, trials=10, variance_of_trials=-1.0)

    def test_nan_variance_is_refused(self) -> None:
        """`< 0` alone does not catch this: `float('nan') < 0` is `False`, so a NaN
        `variance_of_trials` — exactly what vectorbt's own `np.var(x, ddof=1)` produces on a
        single trial (`eval/dsr_comparison.py`, run against vectorbt's real vendored formula) —
        would have slipped past the pre-existing guard and returned NaN silently, the precise
        defect this whole module exists to design out. Found by running the comparison, not by
        reasoning about it — see this module's own docstring for the full account."""
        with pytest.raises(MetricError, match="cannot be negative or NaN"):
            deflated_sharpe(2.0, n=500, trials=10, variance_of_trials=float("nan"))


class TestProbabilisticSharpe:
    """`deflated_sharpe` reduces to this on a single trial — its own NaN-safety matters directly."""

    def test_nan_observed_is_refused(self) -> None:
        """`skew * observed` is itself NaN whenever `observed` is NaN (`0.0 * nan == nan`), so a
        NaN `observed` would otherwise reach `denom <= 0` as NaN too — and `nan <= 0` is `False`,
        the same fails-every-comparison shape that makes NaN silent in vectorbt's own gate."""
        with pytest.raises(MetricError, match="NaN observed or benchmark"):
            probabilistic_sharpe(float("nan"), benchmark=0.0, n=500)

    def test_nan_benchmark_is_refused(self) -> None:
        with pytest.raises(MetricError, match="NaN observed or benchmark"):
            probabilistic_sharpe(1.5, benchmark=float("nan"), n=500)


class TestOutOfSampleDecay:
    def test_the_handbook_alert_fires_below_half(self) -> None:
        """Bitget's own reference alert: OS < 0.5x IS."""
        got = out_of_sample_decay(2.0, 0.8)
        assert got["breaches_half_alert"] is True
        assert got["retention_ratio"] == pytest.approx(0.4)

    def test_good_retention_does_not_fire(self) -> None:
        assert out_of_sample_decay(2.0, 1.6)["breaches_half_alert"] is False


class TestEvaluate:
    def test_full_result_has_every_scored_field(self) -> None:
        r = [0.01, -0.005, 0.02, -0.01, 0.005] * 12
        w = [0.5] * len(r)
        got = evaluate(r, w, periods_per_year=DAILY_PER_YEAR).as_dict()
        for field in ("sharpe", "sortino", "max_drawdown_pct", "turnover", "win_rate_pct"):
            assert field in got

    def test_empty_returns_raise(self) -> None:
        with pytest.raises(MetricError):
            evaluate([], [], periods_per_year=DAILY_PER_YEAR)


class TestRollingStability:
    """The handbook scores rolling 30-day Sharpe stability as its own criterion."""

    def _steady(self, n: int = 200) -> list[float]:
        rng = random.Random(11)
        return [rng.gauss(0.001, 0.01) for _ in range(n)]

    def test_a_window_is_produced_for_every_position(self) -> None:
        series = rolling_sharpe(self._steady(100), window=30, periods_per_year=252)
        assert len(series) == 100 - 30 + 1

    def test_a_window_shorter_than_two_is_refused(self) -> None:
        with pytest.raises(MetricError, match="at least two periods"):
            rolling_sharpe([0.01] * 10, window=1, periods_per_year=252)

    def test_too_few_returns_for_one_window_is_refused(self) -> None:
        with pytest.raises(MetricError, match="fewer than"):
            rolling_sharpe([0.01, 0.02], window=30, periods_per_year=252)

    def test_a_flat_window_is_skipped_not_scored_as_zero(self) -> None:
        """Giving a flat window a Sharpe would drag the mean toward that invented number."""
        flat = [0.0] * 60
        assert rolling_sharpe(flat, window=30, periods_per_year=252) == []

    def test_stability_over_a_wholly_flat_series_is_undefined(self) -> None:
        with pytest.raises(MetricError, match="undefined rather than zero"):
            stability([0.0] * 60, window=30, periods_per_year=252)

    def test_skipped_windows_are_counted_and_shown(self) -> None:
        lucky = [0.0] * 80 + [0.05] * 20
        got = stability(lucky, window=30, periods_per_year=252)
        assert got.skipped > 0
        assert got.as_dict()["scorable_share"] < 0.5

    def test_a_steady_series_scores_every_window(self) -> None:
        got = stability(self._steady(), window=30, periods_per_year=252)
        assert got.skipped == 0
        assert got.as_dict()["scorable_share"] == 1.0

    def test_the_spread_is_max_minus_min(self) -> None:
        got = stability(self._steady(), window=30, periods_per_year=252)
        assert got.spread == pytest.approx(got.maximum - got.minimum)

    def test_positive_share_is_a_fraction(self) -> None:
        got = stability(self._steady(), window=30, periods_per_year=252)
        assert 0.0 <= got.positive_share <= 1.0

    def test_an_always_rising_series_is_positive_in_every_window(self) -> None:
        rng = random.Random(5)
        rising = [abs(rng.gauss(0.002, 0.0005)) for _ in range(120)]
        got = stability(rising, window=30, periods_per_year=252)
        assert got.positive_share == 1.0

    def test_two_series_with_the_same_headline_sharpe_differ_in_stability(self) -> None:
        """The property the criterion exists for: the headline number hides this."""
        rng = random.Random(2)
        steady = [rng.gauss(0.001, 0.005) for _ in range(300)]
        spiky = [rng.gauss(0.0, 0.002) for _ in range(280)] + [0.02] * 20
        a = stability(steady, window=30, periods_per_year=252)
        b = stability(spiky, window=30, periods_per_year=252)
        assert a.positive_share != b.positive_share or a.spread != b.spread

    def test_it_serialises_every_field_a_judge_asks_about(self) -> None:
        got = stability(self._steady(), window=30, periods_per_year=252).as_dict()
        for key in ("windows", "skipped_flat_windows", "scorable_share", "window_periods",
                    "mean", "stdev", "min", "max", "spread", "positive_share"):
            assert key in got, key


class TestConventionsAreThePinnedOnes:
    """Two places where reference libraries disagree with each other. Ours is pinned by test."""

    def test_the_risk_free_rate_is_annual_and_is_de_annualised(self) -> None:
        """empyrical takes a per-period rate; we take an annual one. Silent when wrong."""
        returns = [0.01, -0.005, 0.02, 0.0, 0.007, -0.002]
        ppy = 252
        annual = 0.0252
        got = sharpe(returns, periods_per_year=ppy, risk_free=annual)
        manual = [r - annual / ppy for r in returns]
        mu = sum(manual) / len(manual)
        var = sum((x - mu) ** 2 for x in manual) / (len(manual) - 1)
        expected = mu / (var ** 0.5) * (ppy ** 0.5)
        assert got == pytest.approx(expected)

    def test_a_zero_rate_is_the_same_under_either_convention(self) -> None:
        returns = [0.01, -0.005, 0.02, 0.0]
        assert sharpe(returns, periods_per_year=252, risk_free=0.0) == pytest.approx(
            sharpe(returns, periods_per_year=252)
        )

    def test_max_drawdown_is_positive_not_negative(self) -> None:
        """empyrical returns -0.25 for the same curve; we return 0.25."""
        assert max_drawdown([100, 120, 90, 95]) == pytest.approx(0.25)

    def test_drawdown_is_measured_from_the_running_peak_not_the_start(self) -> None:
        """Measuring from the start understates any strategy that made money before losing it."""
        assert max_drawdown([100, 200, 150]) == pytest.approx(0.25)
