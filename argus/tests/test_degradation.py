"""Degradation tests — a fitted series must be caught and a persistent one must not.

The three signals are transcribed thresholds applied to a chronological split, and both halves of
that sentence are places to go wrong. The thresholds are annualised in the source they come from,
and applying them to per-observation Sharpe ratios makes the first signal almost never fire — which
is exactly what happened on first writing, and is why `sharpe_signal` now requires the frequency
rather than defaulting it. The split is by position and must never be shuffled, because a shuffled
test observation sits between two training observations that bracket it.
"""

from __future__ import annotations

import random

import pytest

from argus.backtest.metrics import HOURLY_PER_YEAR
from argus.eval.degradation import (
    DEFAULT_IN_SAMPLE_FRACTION,
    MIN_OBSERVATIONS,
    SHARPE_CRITICAL_DROP,
    SHARPE_WARN_DROP,
    DegradationError,
    Severity,
    assess,
    calibration_signal,
    drawdown_signal,
    max_drawdown,
    sharpe_signal,
)

DAILY = 252


def _fitted(n: int = 1000, *, seed: int = 5, edge: float = 0.004) -> list[float]:
    """Strong for the first 70%, dead for the last 30%. The shape the module exists to catch."""
    rng = random.Random(seed)
    cut = int(n * DEFAULT_IN_SAMPLE_FRACTION)
    return [rng.gauss(edge, 0.01) for _ in range(cut)] + [
        rng.gauss(0.0, 0.01) for _ in range(n - cut)
    ]


def _persistent(n: int = 1000, *, seed: int = 6, edge: float = 0.002) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(edge, 0.01) for _ in range(n)]


class TestTheAnnualisationTheThresholdsAssume:
    def test_the_frequency_is_required_and_not_defaulted(self) -> None:
        """The bug this caught: the thresholds are annualised numbers and were applied to
        per-observation Sharpe ratios, which made the signal almost never fire."""
        with pytest.raises(TypeError):
            sharpe_signal([0.01] * 50, [0.0] * 50)  # type: ignore[call-arg]

    def test_the_same_series_grades_differently_at_different_frequencies(self) -> None:
        """Proof the frequency actually reaches the comparison. If it did not, these would match."""
        series = _fitted()
        cut = int(len(series) * DEFAULT_IN_SAMPLE_FRACTION)
        hourly = sharpe_signal(series[:cut], series[cut:], periods_per_year=HOURLY_PER_YEAR)
        annual = sharpe_signal(series[:cut], series[cut:], periods_per_year=1)
        assert hourly.severity is not annual.severity

    def test_a_fitted_series_is_critical_at_a_realistic_frequency(self) -> None:
        series = _fitted()
        cut = int(len(series) * DEFAULT_IN_SAMPLE_FRACTION)
        got = sharpe_signal(series[:cut], series[cut:], periods_per_year=DAILY)
        assert got.severity is Severity.CRITICAL
        assert got.in_sample is not None and got.in_sample > 1.0

    def test_the_thresholds_are_the_transcribed_ones(self) -> None:
        assert SHARPE_CRITICAL_DROP == 1.5
        assert SHARPE_WARN_DROP == 1.0


class TestTheSharpeSignal:
    def test_a_persistent_edge_is_clean(self) -> None:
        series = _persistent()
        cut = int(len(series) * DEFAULT_IN_SAMPLE_FRACTION)
        assert sharpe_signal(
            series[:cut], series[cut:], periods_per_year=DAILY
        ).severity is Severity.CLEAN

    def test_an_improving_series_is_clean(self) -> None:
        """Getting better out of sample is not degradation. A signal that fired on it would be
        measuring change rather than decay."""
        rng = random.Random(7)
        weak = [rng.gauss(0.0, 0.01) for _ in range(700)]
        strong = [rng.gauss(0.004, 0.01) for _ in range(300)]
        got = sharpe_signal(weak, strong, periods_per_year=DAILY)
        assert got.severity is Severity.CLEAN

    def test_a_flat_half_produces_no_comparison_rather_than_a_flag(self) -> None:
        got = sharpe_signal([0.0] * 100, [0.01, -0.01] * 50, periods_per_year=DAILY)
        assert got.severity is Severity.CLEAN
        assert "no comparison was made" in got.message

    def test_the_message_carries_both_numbers(self) -> None:
        series = _fitted()
        cut = int(len(series) * DEFAULT_IN_SAMPLE_FRACTION)
        got = sharpe_signal(series[:cut], series[cut:], periods_per_year=DAILY)
        assert got.in_sample is not None and got.out_of_sample is not None
        assert f"{got.in_sample:.2f}" in got.message


class TestMaxDrawdown:
    def test_a_rising_curve_has_none(self) -> None:
        assert max_drawdown([0.01] * 50) == 0.0

    def test_a_halving_reads_as_minus_a_half(self) -> None:
        assert max_drawdown([-0.5, *[0.0] * 10]) == pytest.approx(-0.5)

    def test_it_measures_peak_to_trough_not_start_to_end(self) -> None:
        """Up 50%, down 40%, and the drawdown is 40% even though the curve ends above where it
        started. A start-to-end measure would report zero."""
        assert max_drawdown([0.5, -0.4]) == pytest.approx(-0.4)

    def test_an_empty_series_has_no_drawdown(self) -> None:
        assert max_drawdown([]) == 0.0


class TestTheDrawdownSignal:
    def test_a_widening_drawdown_warns(self) -> None:
        shallow = [0.02, -0.01] * 50          # drawdown about -1%
        deep = [*[-0.02] * 20, *[0.01] * 30]  # a sustained bleed, about -33%
        assert drawdown_signal(shallow, deep).severity is Severity.WARN

    def test_a_shallow_out_of_sample_drawdown_is_clean(self) -> None:
        assert drawdown_signal([-0.3, 0.1] * 20, [0.001] * 40).severity is Severity.CLEAN

    def test_it_fires_where_the_sharpe_signal_cannot(self) -> None:
        """The reason it exists, shown the only way that isolates it: **the same returns in a
        different order.** Sharpe is a function of the mean and the standard deviation, so it is
        identical for any permutation. Drawdown depends on the path. Sorting the losses to the
        front changes the tail completely and leaves the Sharpe untouched, so a system relying on
        Sharpe alone cannot see this at all."""
        rng = random.Random(8)
        in_sample = [rng.gauss(0.001, 0.012) for _ in range(300)]
        shuffled = list(in_sample)
        rng.shuffle(shuffled)
        worst_first = sorted(in_sample)

        benign = drawdown_signal(in_sample, shuffled)
        hostile = drawdown_signal(in_sample, worst_first)
        assert hostile.out_of_sample is not None and benign.out_of_sample is not None
        assert hostile.out_of_sample < benign.out_of_sample
        assert hostile.severity is Severity.WARN

        # The Sharpe signal is blind to the reordering: identical on both, by construction.
        a = sharpe_signal(in_sample, shuffled, periods_per_year=DAILY)
        b = sharpe_signal(in_sample, worst_first, periods_per_year=DAILY)
        assert a.out_of_sample == pytest.approx(b.out_of_sample)
        assert a.severity is b.severity


class TestTheCalibrationSignal:
    def test_a_large_rise_past_the_level_warns(self) -> None:
        assert calibration_signal(0.02, 0.30).severity is Severity.WARN

    def test_a_rise_that_stays_below_the_level_is_clean(self) -> None:
        """Both conditions must hold. A rise from 0.01 to 0.12 is a big relative change and still
        leaves the forecaster well calibrated."""
        assert calibration_signal(0.01, 0.12).severity is Severity.CLEAN

    def test_a_high_level_that_did_not_rise_is_clean(self) -> None:
        """Badly calibrated throughout is a calibration problem, not a degradation one, and this
        module measures degradation."""
        assert calibration_signal(0.30, 0.32).severity is Severity.CLEAN

    def test_improving_calibration_is_clean(self) -> None:
        assert calibration_signal(0.30, 0.02).severity is Severity.CLEAN

    def test_a_missing_half_makes_no_comparison(self) -> None:
        assert "no comparison was made" in calibration_signal(None, 0.2).message
        assert "no comparison was made" in calibration_signal(0.2, None).message


class TestTheAssessment:
    def test_a_fitted_series_is_flagged(self) -> None:
        got = assess("fitted", _fitted(), periods_per_year=DAILY)
        assert got.severity is Severity.CRITICAL
        assert "DEGRADATION DETECTED" in got.verdict

    def test_a_persistent_series_is_not(self) -> None:
        got = assess("persistent", _persistent(), periods_per_year=DAILY)
        assert got.severity is Severity.CLEAN

    def test_a_clean_result_admits_it_is_weak_evidence(self) -> None:
        """A strategy with no edge at all also fails to degrade. Without this sentence a clean
        result reads as a pass."""
        got = assess("persistent", _persistent(), periods_per_year=DAILY)
        assert "weak evidence" in got.verdict
        assert "no edge at all also fails to degrade" in got.verdict

    def test_the_worst_signal_sets_the_severity(self) -> None:
        got = assess("fitted", _fitted(), periods_per_year=DAILY, ece_in=0.02, ece_out=0.02)
        assert got.severity is Severity.CRITICAL

    def test_calibration_alone_can_raise_a_warning(self) -> None:
        got = assess(
            "drifting", _persistent(), periods_per_year=DAILY, ece_in=0.01, ece_out=0.40
        )
        assert got.severity is Severity.WARN
        assert "calibration_drift" in got.verdict

    def test_the_split_is_by_position_and_never_shuffled(self) -> None:
        """Reordering the series must change the answer. If it does not, the split is shuffling."""
        series = _fitted()
        reversed_series = list(reversed(series))
        assert (
            assess("a", series, periods_per_year=DAILY).severity
            is not assess("b", reversed_series, periods_per_year=DAILY).severity
        )

    def test_the_in_sample_fraction_is_a_parameter_not_a_constant(self) -> None:
        """Their default of 0.7 is aggressive for a long record, which is why it moves here."""
        series = _fitted()
        assert (
            assess("a", series, periods_per_year=DAILY, in_sample_fraction=0.5).signals[0].in_sample
            != assess("b", series, periods_per_year=DAILY, in_sample_fraction=0.9)
            .signals[0].in_sample
        )

    def test_the_default_fraction_is_the_transcribed_one(self) -> None:
        assert DEFAULT_IN_SAMPLE_FRACTION == 0.7

    def test_a_short_record_raises(self) -> None:
        with pytest.raises(DegradationError, match="below the"):
            assess("thin", [0.01] * (MIN_OBSERVATIONS - 1), periods_per_year=DAILY)

    def test_an_extreme_fraction_raises(self) -> None:
        for fraction in (0.01, 0.99):
            with pytest.raises(DegradationError, match="too small to measure"):
                assess("x", _persistent(), periods_per_year=DAILY, in_sample_fraction=fraction)

    def test_the_report_shows_all_three_signals(self) -> None:
        text = assess("x", _fitted(), periods_per_year=DAILY).render()
        for name in ("sharpe_collapse", "drawdown_widening", "calibration_drift"):
            assert name in text

    def test_the_dict_carries_the_numbers_behind_each_flag(self) -> None:
        got = assess("x", _fitted(), periods_per_year=DAILY).as_dict()
        assert len(got["signals"]) == 3
        assert got["in_sample_fraction"] == DEFAULT_IN_SAMPLE_FRACTION
        assert got["signals"][0]["in_sample"] is not None
