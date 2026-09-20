"""`BacktestResult.weights` — the per-bar target weight `run()` already computed internally and
never returned. Exists so a caller can reconstruct discrete trade boundaries (a weight crossing
zero, or reversing sign, is an open/close event) from a completed backtest without re-running it —
the prerequisite for comparing ARGUS's continuous-weight research engine against any discrete-trade
risk system (a circuit breaker, `freqtrade`'s Protections) on the same underlying data, which
nothing in this module could do before this field existed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from argus.backtest.engine import Bar, run
from argus.cost.model import CostModel

COST = CostModel.bitget_perp()
_PATTERN = ["0.011", "-0.004", "0.007", "0.019", "-0.013", "0.002", "-0.006", "0.015"]


def _bars(n: int) -> list[Bar]:
    start = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    px = Decimal("100")
    out: list[Bar] = []
    for i in range(n):
        px = px * (Decimal("1") + Decimal(_PATTERN[i % len(_PATTERN)]))
        out.append(Bar(ts=start + timedelta(days=i), close=px))
    return out


def _always_long(bars: object, i: int) -> float:
    return 1.0


def _flat_then_long(bars: object, i: int) -> float:
    """Flat for the first half, long for the second — one clean open event."""
    return 0.0 if i < 10 else 1.0


def _alternating(bars: object, i: int) -> float:
    return 1.0 if i % 2 == 0 else 0.0


class TestWeightsIsPopulated:
    def test_weights_has_one_entry_per_bar_transition(self) -> None:
        bars = _bars(20)
        result = run(
            "always-long", "TEST", bars, _always_long, cost=COST, periods_per_year=365,
        )
        # `run` iterates `range(len(bars) - 1)` — one weight per transition, not per bar.
        assert len(result.weights) == len(bars) - 1

    def test_a_constant_signal_holds_a_constant_weight(self) -> None:
        bars = _bars(20)
        result = run(
            "always-long", "TEST", bars, _always_long, cost=COST, periods_per_year=365,
        )
        assert all(w == 1.0 for w in result.weights)

    def test_a_flat_to_long_transition_is_visible_in_the_series(self) -> None:
        bars = _bars(20)
        result = run(
            "flat-then-long", "TEST", bars, _flat_then_long, cost=COST, periods_per_year=365,
        )
        assert result.weights[0] == 0.0
        assert result.weights[-1] == 1.0
        # The exact bar where it opens is reconstructable — the whole point of exposing this.
        opens_at = next(i for i, w in enumerate(result.weights) if w != 0.0)
        assert opens_at == 10

    def test_default_is_empty_not_missing(self) -> None:
        """The dataclass default (`()`), not this run — pins that the field exists at all and
        that a caller who never asked for it gets an empty tuple, not an AttributeError."""
        from argus.backtest.engine import BacktestResult

        assert BacktestResult.__dataclass_fields__["weights"].default == ()


class TestExtractTrades:
    """Reconstructs discrete open/close events from a continuous weight series — the exact
    correctness bar `desk.book.Position.apply_fill` was held to when it was built, since this
    reuses the same reversal convention (verified against `nautilus_trader`'s `position.rs` then,
    cited here rather than re-derived)."""

    @staticmethod
    def _bars_with_prices(prices: list[str]) -> list[Bar]:
        start = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
        return [
            Bar(ts=start + timedelta(days=i), close=Decimal(p))
            for i, p in enumerate(prices)
        ]

    def test_flat_the_whole_time_produces_no_trades(self) -> None:
        from argus.backtest.engine import extract_trades

        bars = self._bars_with_prices(["100", "101", "102", "103"])
        assert extract_trades(bars, [0.0, 0.0, 0.0]) == []

    def test_a_single_long_open_and_close_is_one_trade(self) -> None:
        from argus.backtest.engine import extract_trades

        # weights held bar 0->1, 1->2, 2->3; flat again at index 2 (held 2->3 is flat).
        bars = self._bars_with_prices(["100", "110", "121", "121"])
        trades = extract_trades(bars, [1.0, 1.0, 0.0])
        assert len(trades) == 1
        trade = trades[0]
        assert trade.entry_bar == 0 and trade.exit_bar == 2
        assert trade.direction == "long"
        assert trade.entry_price == Decimal("100")
        assert trade.exit_price == Decimal("121")
        assert trade.return_pct == pytest.approx(0.21)

    def test_a_single_short_trade_has_inverted_return_sign(self) -> None:
        from argus.backtest.engine import extract_trades

        bars = self._bars_with_prices(["100", "90", "90"])
        trades = extract_trades(bars, [-1.0, 0.0])
        assert len(trades) == 1
        assert trades[0].direction == "short"
        assert trades[0].return_pct == pytest.approx(0.10)  # price fell 10%, short profits

    def test_a_direct_reversal_closes_and_reopens_at_the_same_bar(self) -> None:
        from argus.backtest.engine import extract_trades

        bars = self._bars_with_prices(["100", "110", "99"])
        trades = extract_trades(bars, [1.0, -1.0])
        assert len(trades) == 2
        assert trades[0].direction == "long"
        assert trades[0].entry_bar == 0 and trades[0].exit_bar == 1
        assert trades[1].direction == "short"
        assert trades[1].entry_bar == 1 and trades[1].exit_bar == 2

    def test_a_position_still_open_at_the_series_end_closes_at_the_last_bar(self) -> None:
        from argus.backtest.engine import extract_trades

        bars = self._bars_with_prices(["100", "105", "110"])
        trades = extract_trades(bars, [1.0, 1.0])
        assert len(trades) == 1
        assert trades[0].exit_bar == 2
        assert trades[0].exit_price == Decimal("110")

    def test_a_weight_change_within_the_same_direction_stays_one_trade(self) -> None:
        """0.5 -> 0.8, same sign: one trade at its entry weight, not a resize sequence — the
        simplification `SyntheticTrade.weight`'s own docstring states."""
        from argus.backtest.engine import extract_trades

        bars = self._bars_with_prices(["100", "105", "110", "110"])
        trades = extract_trades(bars, [0.5, 0.8, 0.0])
        assert len(trades) == 1
        assert trades[0].weight == 0.5

    def test_a_mismatched_weights_length_raises_rather_than_silently_misaligning(self) -> None:
        from argus.backtest.engine import extract_trades
        from argus.backtest.metrics import MetricError

        bars = self._bars_with_prices(["100", "105", "110"])
        with pytest.raises(MetricError, match="do not match"):
            extract_trades(bars, [1.0])  # needs exactly len(bars) - 1 = 2

    def test_agrees_with_run_own_weights_on_a_real_backtest(self) -> None:
        """End-to-end: feed `run()`'s own output back into the extractor rather than only
        hand-built fixtures, so the two functions are proven to actually compose.

        Uses an uneven return pattern, not constant drift: a monotonic series has zero downside
        deviation and `run()`'s own Sortino calculation correctly refuses to compute on it
        (`backtest/metrics.py`'s own documented reasoning, matched here rather than worked around
        with a fixture that "tests nothing", the exact phrase `test_passive.py` uses for the same
        pitfall).
        """
        from argus.backtest.engine import extract_trades, run
        from argus.cost.model import CostModel

        bars = _bars(15)
        result = run(
            "steady-long", "TEST", bars, lambda b, i: 1.0,
            cost=CostModel.bitget_perp(), periods_per_year=365,
        )
        trades = extract_trades(bars, list(result.weights))
        assert len(trades) == 1
        assert trades[0].direction == "long"
        assert trades[0].entry_bar == 0
        assert trades[0].exit_bar == len(bars) - 1


class TestWeightsMatchesTheTradeCount:
    def test_every_material_weight_change_is_visible_as_a_delta(self) -> None:
        """Cross-checks the new field against the engine's own pre-existing `trades` counter —
        the same information, two different shapes, must agree."""
        bars = _bars(20)
        result = run(
            "alternating", "TEST", bars, _alternating, cost=COST, periods_per_year=365,
        )
        deltas = sum(
            1 for prev, cur in zip((0.0, *result.weights[:-1]), result.weights, strict=True)
            if abs(cur - prev) > 1e-9
        )
        assert deltas == result.trades
