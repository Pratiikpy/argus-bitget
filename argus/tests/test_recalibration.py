"""A miscalibrated confidence is fixed before it is refused — and only if the fix holds out.

`risk/sizing.calibration_gate` used to treat a well-populated record with a large calibration
error exactly like an empty one (research/harvest/23-uncertainty-toolbox.md). These tests pin the
fix: isotonic recalibration, accepted only on its held-out calibration error.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from argus.risk.calibration import Prediction, held_out_ece, isotonic
from argus.risk.sizing import (
    FIXED_FRACTION,
    MIN_FOR_RECALIBRATION,
    calibration_gate,
    size,
)


def _record(n: int, *, confidence: float, hit_rate: float,
            interleaved: bool = True) -> list[Prediction]:
    right = round(n * hit_rate)
    hits = ({round(i * n / right) for i in range(right)} if right else set()) if interleaved \
        else set(range(right))
    return [Prediction(confidence=confidence, correct=i in hits) for i in range(n)]


class TestIsotonic:
    def test_a_single_level_maps_to_its_hit_rate(self) -> None:
        fit = isotonic(_record(40, confidence=0.9, hit_rate=0.55))
        assert fit(0.9) == pytest.approx(0.55) and fit(0.1) == pytest.approx(0.55)

    def test_the_fit_never_falls_as_confidence_rises(self) -> None:
        """0.6s win 70% and 0.8s win 40%: the violators pool into one block at 55%."""
        record = _record(20, confidence=0.6, hit_rate=0.7) + \
            _record(20, confidence=0.8, hit_rate=0.4)
        fit = isotonic(record)
        assert fit(0.6) == pytest.approx(0.55) and fit(0.8) == pytest.approx(0.55)

    def test_an_ordered_record_keeps_its_levels_and_interpolates(self) -> None:
        record = _record(20, confidence=0.6, hit_rate=0.4) + \
            _record(20, confidence=0.8, hit_rate=0.7)
        fit = isotonic(record)
        assert fit(0.6) == pytest.approx(0.4) and fit(0.8) == pytest.approx(0.7)
        assert fit(0.7) == pytest.approx(0.55)
        assert fit(0.95) == pytest.approx(0.7) and fit(0.2) == pytest.approx(0.4)

    def test_an_empty_record_is_refused(self) -> None:
        with pytest.raises(ValueError):
            isotonic([])


class TestHeldOut:
    def test_a_stationary_record_recalibrates_out_of_sample(self) -> None:
        assert held_out_ece(_record(60, confidence=0.9, hit_rate=0.5)) < 0.15

    def test_a_record_that_changed_over_time_does_not(self) -> None:
        """Every win in the first third: a map fitted on some folds is wrong on the others."""
        record = _record(60, confidence=0.9, hit_rate=0.35, interleaved=False)
        assert held_out_ece(record) > 0.15

    def test_too_few_for_the_folds_is_refused(self) -> None:
        with pytest.raises(ValueError):
            held_out_ece(_record(3, confidence=0.5, hit_rate=0.5))


class TestTheGateFixesBeforeItRefuses:
    def test_a_miscalibrated_but_stable_record_is_recalibrated(self) -> None:
        gate = calibration_gate(_record(MIN_FOR_RECALIBRATION, confidence=0.9, hit_rate=0.55))
        assert gate.passed and gate.recalibration is not None
        assert gate.ece is not None and gate.ece > 0.15
        assert gate.held_out_ece is not None and gate.held_out_ece <= 0.15
        assert "recalibrated" in gate.reason

    def test_one_short_of_the_sample_is_still_refused(self) -> None:
        gate = calibration_gate(_record(MIN_FOR_RECALIBRATION - 1, confidence=0.9,
                                        hit_rate=0.55))
        assert not gate.passed and "too few to recalibrate" in gate.reason

    def test_a_fix_that_does_not_hold_out_is_refused(self) -> None:
        gate = calibration_gate(_record(60, confidence=0.9, hit_rate=0.35, interleaved=False))
        assert not gate.passed and gate.recalibration is None
        assert "does not hold out of sample" in gate.reason

    def test_a_calibrated_record_is_not_touched(self) -> None:
        gate = calibration_gate(_record(60, confidence=0.7, hit_rate=0.7))
        assert gate.passed and gate.recalibration is None and gate.held_out_ece is None


class TestSizingReadsTheRecalibratedProbability:
    def test_an_overconfident_desk_is_sized_on_what_it_actually_wins(self) -> None:
        """Says 0.9, wins 55%: at even money half-Kelly on 0.55 is 5%, not the 40% (capped at
        25%) the stated 0.9 would ask for."""
        record = _record(60, confidence=0.9, hit_rate=0.55)
        sized = size(win_probability=0.9, payoff=Decimal("1"), predictions=record)
        assert sized.gate.recalibration is not None
        assert sized.fraction == pytest.approx(Decimal("0.05"), abs=Decimal("1e-9"))
        assert "stated 0.90 -> 0.55" in sized.basis

    def test_a_losing_record_sizes_to_nothing(self) -> None:
        record = _record(60, confidence=0.8, hit_rate=0.3)
        assert size(win_probability=0.8, payoff=Decimal("1"),
                    predictions=record).fraction == 0

    def test_a_refused_fix_falls_back_to_the_fixed_fraction(self) -> None:
        record = _record(60, confidence=0.9, hit_rate=0.35, interleaved=False)
        assert size(win_probability=0.9, payoff=Decimal("1"),
                    predictions=record).fraction == FIXED_FRACTION

    def test_a_non_probability_is_refused_not_clipped(self) -> None:
        record = _record(60, confidence=0.9, hit_rate=0.55)
        with pytest.raises(ValueError, match="not a probability"):
            size(win_probability=1.4, payoff=Decimal("1"), predictions=record)
