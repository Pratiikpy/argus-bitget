"""`agents/delay_cost.py`: the deliberation charge read from the book instead of a constant."""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from argus.agents.delay_cost import (
    GAUSSIAN_MEAN_ABS,
    MIN_BAR_RETURNS,
    MIN_WINDOW_MOVES,
    DelayCostError,
    bar_delay_cost_bps,
    empirical_delay_cost_bps,
)


def _zigzag(n: int, step_bps: float, start: float = 100.0) -> list[float]:
    """A mid that moves +step, -step, +step ... one second at a time."""
    out = [start]
    for i in range(n - 1):
        sign = 1 if i % 2 == 0 else -1
        out.append(out[-1] * math.exp(sign * step_bps / 1e4))
    return out


class TestEmpirical:
    def test_a_zigzag_moves_one_step_over_odd_delays_and_nothing_over_even_ones(self) -> None:
        mids = _zigzag(400, 2.0)
        assert float(empirical_delay_cost_bps(mids, delay_s=3)) == pytest.approx(2.0, abs=1e-9)
        assert float(empirical_delay_cost_bps(mids, delay_s=8)) == pytest.approx(0.0, abs=1e-9)

    def test_a_steady_drift_costs_delay_times_the_step(self) -> None:
        mids = [100.0 * math.exp(0.5 * i / 1e4) for i in range(300)]
        assert float(empirical_delay_cost_bps(mids, delay_s=40)) == pytest.approx(20.0, rel=1e-9)

    def test_returns_a_decimal(self) -> None:
        assert isinstance(empirical_delay_cost_bps(_zigzag(200, 1.0), delay_s=3), Decimal)

    @pytest.mark.parametrize("delay", [0, -3])
    def test_refuses_a_non_positive_delay(self, delay: int) -> None:
        with pytest.raises(DelayCostError):
            empirical_delay_cost_bps(_zigzag(200, 1.0), delay_s=delay)

    def test_refuses_too_short_a_window(self) -> None:
        with pytest.raises(DelayCostError, match="at least"):
            empirical_delay_cost_bps(_zigzag(MIN_WINDOW_MOVES + 2, 1.0), delay_s=3)

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
    def test_refuses_a_non_price(self, bad: float) -> None:
        mids = _zigzag(200, 1.0)
        mids[50] = bad
        with pytest.raises(DelayCostError):
            empirical_delay_cost_bps(mids, delay_s=3)


class TestBars:
    def test_constant_absolute_bar_return_scales_by_sqrt_time_and_gaussian_mean(self) -> None:
        closes = _zigzag(31, 10.0)             # every one-minute return is +/-10bps
        got = float(bar_delay_cost_bps(closes, bar_s=60, delay_s=15))
        assert got == pytest.approx(GAUSSIAN_MEAN_ABS * 10.0 * math.sqrt(15 / 60), rel=1e-9)

    def test_gaussian_mean_abs_is_sqrt_two_over_pi(self) -> None:
        assert pytest.approx(0.7978845608, rel=1e-9) == GAUSSIAN_MEAN_ABS

    def test_refuses_too_few_bars(self) -> None:
        with pytest.raises(DelayCostError):
            bar_delay_cost_bps(_zigzag(MIN_BAR_RETURNS, 5.0), bar_s=60, delay_s=40)

    @pytest.mark.parametrize(("bar_s", "delay_s"), [(0, 40.0), (60, 0.0), (-60, 3.0)])
    def test_refuses_non_positive_intervals(self, bar_s: int, delay_s: float) -> None:
        with pytest.raises(DelayCostError):
            bar_delay_cost_bps(_zigzag(40, 5.0), bar_s=bar_s, delay_s=delay_s)

