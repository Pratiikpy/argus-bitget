"""Track 1's session-conditioned variants (`strategies/session_alpha.py`), offline.

Every variant is a pure function of a bar sequence and the anchor market's session phase
(`argus.truth.clocks.DualClock`), so each is pinned against hand-picked instants whose phase is
known in advance: a Tuesday during regular hours, the exact 16:00 ET close boundary (which is
EXTENDED, not RTH — the boundary is exclusive), a Saturday, New Year's Day 2026 (a named holiday,
checked ahead of the weekday test), and a Monday night after the close (OVERNIGHT). No network, no
system clock: every instant is stated explicitly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from argus.backtest.engine import Bar
from argus.strategies import session_alpha as sa
from argus.truth.clocks import SessionPhase

# Verified against argus.truth.clocks.DualClock().phase(...) directly:
#   RTH                -> rth        (Tue 2026-01-06, 10:00 ET)
#   EXTENDED_AT_CLOSE   -> extended   (Tue 2026-01-06, 16:00:00 ET exactly — RTH_CLOSE excludes it)
#   WEEKEND             -> weekend    (Sat 2026-01-03)
#   HOLIDAY             -> holiday    (Thu 2026-01-01, New Year's Day)
#   OVERNIGHT           -> overnight  (Mon 2026-01-05, 21:00 ET)
RTH = datetime(2026, 1, 6, 15, 0, tzinfo=UTC)
EXTENDED_AT_CLOSE = datetime(2026, 1, 6, 21, 0, tzinfo=UTC)
WEEKEND = datetime(2026, 1, 3, 15, 0, tzinfo=UTC)
HOLIDAY = datetime(2026, 1, 1, 15, 0, tzinfo=UTC)
OVERNIGHT = datetime(2026, 1, 6, 2, 0, tzinfo=UTC)


def _bar(ts: datetime, close: float) -> Bar:
    return Bar(ts=ts, close=Decimal(str(close)))


def _series(closes: list[float], start: datetime = RTH) -> list[Bar]:
    return [_bar(start + timedelta(hours=i), c) for i, c in enumerate(closes)]


class TestPhase:
    def test_phase_delegates_to_the_dual_clock_for_every_named_instant(self) -> None:
        assert sa._phase(_bar(RTH, 1.0)) is SessionPhase.RTH
        assert sa._phase(_bar(EXTENDED_AT_CLOSE, 1.0)) is SessionPhase.EXTENDED
        assert sa._phase(_bar(WEEKEND, 1.0)) is SessionPhase.WEEKEND
        assert sa._phase(_bar(HOLIDAY, 1.0)) is SessionPhase.HOLIDAY
        assert sa._phase(_bar(OVERNIGHT, 1.0)) is SessionPhase.OVERNIGHT


class TestRet:
    def test_ret_is_zero_before_lookback_bars_of_history_exist(self) -> None:
        bars = _series([100.0, 101.0, 102.0])
        assert sa._ret(bars, 2, 6) == 0.0  # j = 2 - 6 = -4, before the start of the series

    def test_ret_at_exactly_the_start_of_the_series_still_computes(self) -> None:
        bars = _series([100.0] + [110.0] * 6)  # j = 6 - 6 = 0, the boundary, not negative
        assert sa._ret(bars, 6, 6) == pytest.approx(0.10)

    def test_ret_computes_the_percentage_change_over_the_lookback(self) -> None:
        bars = _series([80.0, 0, 0, 0, 0, 0, 100.0])
        assert sa._ret(bars, 6, 6) == pytest.approx(0.25)

    def test_ret_guards_a_zero_close_at_the_start_of_the_window(self) -> None:
        bars = _series([0.0, 0, 0, 0, 0, 0, 100.0])
        assert sa._ret(bars, 6, 6) == 0.0  # division by zero avoided, not an error


class TestTheSessionOnlyVariants:
    """`hold_through_closure`, `trade_only_when_open`, `weekend_only`, `low_turnover_carry`: each
    looks only at the current bar's phase, so a single-bar series pins every one of them."""

    @pytest.mark.parametrize(
        ("instant", "expected"),
        [
            (RTH, 0.0), (EXTENDED_AT_CLOSE, 1.0), (WEEKEND, 1.0), (HOLIDAY, 1.0), (OVERNIGHT, 1.0),
        ],
    )
    def test_hold_through_closure_is_long_exactly_while_the_anchor_is_shut(
        self, instant: datetime, expected: float,
    ) -> None:
        assert sa.hold_through_closure([_bar(instant, 100.0)], 0) == expected

    @pytest.mark.parametrize(
        ("instant", "expected"),
        [
            (RTH, 1.0), (EXTENDED_AT_CLOSE, 0.0), (WEEKEND, 0.0), (HOLIDAY, 0.0), (OVERNIGHT, 0.0),
        ],
    )
    def test_trade_only_when_open_is_the_mirror_image(
        self, instant: datetime, expected: float,
    ) -> None:
        assert sa.trade_only_when_open([_bar(instant, 100.0)], 0) == expected

    def test_hold_through_closure_and_trade_only_when_open_always_sum_to_one(self) -> None:
        for instant in (RTH, EXTENDED_AT_CLOSE, WEEKEND, HOLIDAY, OVERNIGHT):
            bars = [_bar(instant, 100.0)]
            assert (sa.hold_through_closure(bars, 0) + sa.trade_only_when_open(bars, 0)) == 1.0

    @pytest.mark.parametrize(
        ("instant", "expected"),
        [
            (RTH, 0.0), (EXTENDED_AT_CLOSE, 0.0), (WEEKEND, 1.0), (HOLIDAY, 1.0),
            # OVERNIGHT does not count: the anchor is shut but this is not the weekend closure,
            # which is exactly the distinction `hold_through_closure` (1.0 here) does not draw.
            (OVERNIGHT, 0.0),
        ],
    )
    def test_weekend_only_excludes_an_ordinary_overnight_closure(
        self, instant: datetime, expected: float,
    ) -> None:
        assert sa.weekend_only([_bar(instant, 100.0)], 0) == expected

    @pytest.mark.parametrize("instant", [RTH, EXTENDED_AT_CLOSE, WEEKEND, HOLIDAY, OVERNIGHT])
    def test_low_turnover_carry_is_the_complement_of_weekend_only(self, instant: datetime) -> None:
        bars = [_bar(instant, 100.0)]
        assert (sa.weekend_only(bars, 0) + sa.low_turnover_carry(bars, 0)) == 1.0


class TestClosureMomentumAndReversion:
    """`closure_momentum` carries the sign of the 6-bar move into a closed session; the RTH guard
    is checked before the move is even computed, and `closure_reversion` is its exact negation."""

    def _bars(self, last_close: float, *, instant: datetime) -> list[Bar]:
        return _series([100.0, 0, 0, 0, 0, 0, last_close], start=instant - timedelta(hours=6))

    def test_momentum_carries_a_positive_move_into_the_closure(self) -> None:
        bars = self._bars(110.0, instant=WEEKEND)
        assert sa.closure_momentum(bars, 6) == 1.0
        assert sa.closure_reversion(bars, 6) == -1.0

    def test_momentum_carries_a_negative_move_into_the_closure(self) -> None:
        bars = self._bars(90.0, instant=WEEKEND)
        assert sa.closure_momentum(bars, 6) == -1.0
        assert sa.closure_reversion(bars, 6) == 1.0

    def test_momentum_is_exactly_zero_on_an_exactly_flat_move(self) -> None:
        bars = self._bars(100.0, instant=WEEKEND)
        assert sa.closure_momentum(bars, 6) == 0.0
        assert sa.closure_reversion(bars, 6) == 0.0

    def test_momentum_is_zero_during_rth_regardless_of_the_move(self) -> None:
        """The phase is checked before the move is computed: a large move during RTH is not the
        closure trade at all, and must not leak a signal through the guard."""
        bars = self._bars(500.0, instant=RTH)
        assert sa.closure_momentum(bars, 6) == 0.0
        assert sa.closure_reversion(bars, 6) == 0.0


def test_variants_registers_exactly_the_documented_six_functions() -> None:
    assert {
        "hold_through_closure": sa.hold_through_closure,
        "trade_only_when_open": sa.trade_only_when_open,
        "weekend_only": sa.weekend_only,
        "closure_momentum": sa.closure_momentum,
        "closure_reversion": sa.closure_reversion,
        "low_turnover_carry": sa.low_turnover_carry,
    } == sa.VARIANTS


class TestBarsFromCandles:
    def test_builds_bars_from_valid_tz_aware_candles_preserving_order(self) -> None:
        candles: list[dict[str, Any]] = [
            {"ts": datetime(2026, 1, 6, 15, 0, tzinfo=UTC), "close": 101.5},
            {"ts": datetime(2026, 1, 6, 16, 0, tzinfo=UTC), "close": "202.25"},
        ]
        bars = sa.bars_from_candles(candles)
        assert [b.close for b in bars] == [Decimal("101.5"), Decimal("202.25")]
        assert bars[0].ts == datetime(2026, 1, 6, 15, 0, tzinfo=UTC)

    def test_skips_a_string_timestamp(self) -> None:
        candles: list[dict[str, Any]] = [{"ts": "2026-01-06T15:00:00Z", "close": 101.5}]
        assert sa.bars_from_candles(candles) == []

    def test_skips_a_timestamp_with_no_tzinfo_attribute_at_all(self) -> None:
        candles: list[dict[str, Any]] = [{"ts": 1767711600, "close": 101.5}]  # a raw epoch int
        assert sa.bars_from_candles(candles) == []

    def test_keeps_only_the_well_formed_rows_and_drops_the_rest(self) -> None:
        good_a = {"ts": datetime(2026, 1, 6, 15, 0, tzinfo=UTC), "close": 100.0}
        bad = {"ts": "not a datetime", "close": 1.0}
        good_b = {"ts": datetime(2026, 1, 6, 16, 0, tzinfo=UTC), "close": 200.0}
        bars = sa.bars_from_candles([good_a, bad, good_b])
        assert [b.close for b in bars] == [Decimal("100.0"), Decimal("200.0")]

    def test_skips_a_naive_datetime_timestamp(self) -> None:
        candles: list[dict[str, Any]] = [{"ts": datetime(2026, 1, 6, 15, 0), "close": 101.5}]
        assert sa.bars_from_candles(candles) == []
