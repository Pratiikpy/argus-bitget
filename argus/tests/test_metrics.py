"""Metric tests — each guards a bug found in a real library.

Track 1 is scored purely on these numbers. A wrong metric here is worse than a wrong strategy,
because it makes a bad strategy look good and nobody checks the arithmetic.
"""

from __future__ import annotations

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
    sharpe,
    sortino,
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
        """vectorbt returns NaN silently, and a gate comparing NaN with > admits everything."""
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
