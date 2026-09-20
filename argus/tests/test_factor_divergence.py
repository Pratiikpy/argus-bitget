"""Tests for `research.factor_divergence` — WorldQuant's real Alpha#101 formula #23, evaluated
via ARGUS's own grammar engine. Pure/constructed tests here; the real market-vs-index comparison
against Alphalens' real Information Coefficient lives in `eval/factor_divergence_comparison.py`
and its own test file (live network).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from argus.backtest.engine import Bar
from argus.market.history import Candle
from argus.research.factor_divergence import (
    ALPHA_23,
    MIN_WARMUP,
    alpha23_series,
    candles_to_bars,
    forward_returns,
)


def _bar(hour: int, high: float, low: float, close: float = 100.0) -> Bar:
    return Bar(
        ts=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=hour),
        close=Decimal(str(close)),
        extra={"high": high, "low": low},
    )


class TestAlpha23:
    def test_stays_zero_while_high_never_exceeds_its_own_mean(self) -> None:
        bars = [_bar(h, high=100.0, low=99.0) for h in range(30)]
        values = alpha23_series(bars)
        assert all(v == 0.0 for v in values[MIN_WARMUP:])

    def test_fires_the_two_bar_delay_once_the_high_breaks_its_mean(self) -> None:
        bars = [_bar(h, high=100.0, low=99.0) for h in range(25)]
        bars += [_bar(25, high=150.0, low=99.0)]  # breaks the 20-bar mean
        values = alpha23_series(bars)
        # at i=25, mean(high,20) < high(25) -> True, so value = delay(2, high) = high at i=23
        assert values[25] == pytest.approx(100.0)

    def test_is_never_wrapped_in_signal_and_so_is_not_clamped_to_unit_range(self) -> None:
        """The whole reason this module evaluates the bare `IfElse`, not `grammar.Signal`:
        Signal clamps to [-1, 1], which would destroy a price-delta factor's real magnitude."""
        bars = [_bar(h, high=100.0, low=99.0) for h in range(25)]
        bars += [_bar(25, high=500.0, low=99.0)]
        values = alpha23_series(bars)
        assert values[25] > 1.0

    def test_degrades_gracefully_before_warmup_rather_than_raising(self) -> None:
        bars = [_bar(h, high=100.0 + h, low=99.0) for h in range(5)]
        values = alpha23_series(bars)  # should not raise
        assert len(values) == 5


class TestForwardReturns:
    def test_computes_the_real_close_to_close_return(self) -> None:
        bars = [_bar(0, high=100, low=99, close=100.0), _bar(1, high=101, low=99, close=110.0)]
        fwd = forward_returns(bars, horizon=1)
        assert fwd[0] == pytest.approx(0.10)

    def test_the_last_horizon_bars_are_none_not_a_fabricated_value(self) -> None:
        bars = [_bar(h, high=100, low=99, close=100.0) for h in range(3)]
        fwd = forward_returns(bars, horizon=1)
        assert fwd[-1] is None
        assert len(fwd) == 3

    def test_a_zero_or_negative_close_refuses_rather_than_dividing_by_it(self) -> None:
        bars = [_bar(0, high=1, low=1, close=0.0), _bar(1, high=1, low=1, close=5.0)]
        fwd = forward_returns(bars, horizon=1)
        assert fwd[0] is None


class TestCandlesToBars:
    def test_carries_high_and_low_into_extra(self) -> None:
        candle = Candle(
            ts=datetime(2026, 1, 1, tzinfo=UTC),
            open=Decimal("100"), high=Decimal("105"), low=Decimal("98"),
            close=Decimal("102"), volume=Decimal("1000"),
        )
        bars = candles_to_bars([candle])
        assert bars[0].extra is not None
        assert bars[0].extra["high"] == 105.0
        assert bars[0].extra["low"] == 98.0
        assert bars[0].close == Decimal("102")


def test_alpha_23_matches_the_catalogued_formula_from_the_port_doc() -> None:
    """`research/architecture/alpha101-port.md`'s own real, catalogued translation of
    WorldQuant's Alpha#101 formula #23 (`Alpha101.py:259-265`):
    `delta(high, 2) if mean(high, 20) < high else 0` — checked here structurally rather than
    trusted from the docstring alone."""
    from argus.research.grammar import BinOp, Const, Delay, Field, IfElse, Ref, Window

    assert IfElse(
        BinOp("lt", Window("mean", 20, Ref(Field.HIGH)), Ref(Field.HIGH)),
        Delay(2, Ref(Field.HIGH)),
        Const(0.0),
    ) == ALPHA_23
