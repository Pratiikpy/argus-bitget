"""Statistical machinery tests.

These guard the numbers that decide whether a claim ships. A bug here would let an effect that is
noise pass as a finding, which is the single most damaging failure available to this project.
"""

from __future__ import annotations

import pytest

from argus.research.significance import binomial_p_value, wilson_interval


class TestBinomial:
    def test_coin_flip_result_is_not_significant(self) -> None:
        assert binomial_p_value(50, 100) > 0.05

    def test_overwhelming_result_is_significant(self) -> None:
        assert binomial_p_value(90, 100) < 0.001

    def test_below_null_returns_one(self) -> None:
        """A one-sided test on a result below the null is not evidence of anything."""
        assert binomial_p_value(40, 100) == 1.0

    def test_the_measured_weekend_case(self) -> None:
        """90/156 = 57.7%. p just below 0.05 — which is why the CI check matters."""
        p = binomial_p_value(90, 156)
        assert 0.02 < p < 0.05

    def test_zero_trials_is_safe(self) -> None:
        assert binomial_p_value(0, 0) == 1.0


class TestWilson:
    def test_interval_brackets_the_estimate(self) -> None:
        low, high = wilson_interval(90, 156)
        assert low < 90 / 156 < high

    def test_the_measured_weekend_interval_includes_a_coin_flip(self) -> None:
        """The reason the weekend effect is reported as FAILED despite p < 0.05."""
        low, high = wilson_interval(90, 156)
        assert low < 0.5 < high
        assert low == pytest.approx(0.498, abs=0.005)

    def test_small_samples_give_wide_intervals(self) -> None:
        narrow = wilson_interval(600, 1000)
        wide = wilson_interval(6, 10)
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    def test_zero_trials_spans_everything(self) -> None:
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_a_genuinely_strong_result_excludes_a_coin_flip(self) -> None:
        """The out-of-sample half: 52/78 = 66.7%."""
        low, _ = wilson_interval(52, 78)
        assert low > 0.5
