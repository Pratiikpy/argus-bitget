"""PEAD/SUE study — the pure logic, isolated from the two real network calls (`build()` fetches
live SEC EDGAR and Bitget price history and is exercised separately, not on every test run)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from argus.cost.model import CostModel
from argus.market.history import Candle
from argus.research.pead_study import (
    HOLD_HOURS,
    PeadEvent,
    _build_trade,
    _entry_index,
    _sample_variance,
    _verdict,
)


def _bars(n: int = 400, *, start: datetime | None = None, step_pct: float = 0.001) -> list[Candle]:
    base = start or datetime(2026, 1, 1, tzinfo=UTC)
    out = []
    price = 100.0
    for i in range(n):
        out.append(Candle(
            ts=base + timedelta(hours=i),
            open=Decimal(str(price)), high=Decimal(str(price * 1.001)),
            low=Decimal(str(price * 0.999)), close=Decimal(str(price)),
            volume=Decimal("1000"),
        ))
        price *= (1 + step_pct)
    return out


class TestEntryIndexIsPointInTimeSafe:
    def test_enters_the_first_bar_strictly_after_the_filing_date(self) -> None:
        bars = _bars(start=datetime(2026, 1, 1, tzinfo=UTC))
        idx = _entry_index(bars, date(2026, 1, 1))
        assert idx is not None
        assert bars[idx].ts.date() == date(2026, 1, 2)

    def test_an_event_after_every_fetched_bar_is_not_testable(self) -> None:
        bars = _bars(n=10, start=datetime(2026, 1, 1, tzinfo=UTC))
        assert _entry_index(bars, date(2027, 1, 1)) is None

    def test_an_event_that_predates_the_fetched_history_is_not_testable(self) -> None:
        """The regression this module's own docstring names: a decade-old EDGAR filing must not
        be silently clamped to bar 0 and tested as if the news broke when history begins."""
        bars = _bars(start=datetime(2026, 1, 1, tzinfo=UTC))
        assert _entry_index(bars, date(2015, 1, 1)) is None

    def test_an_empty_bar_sequence_is_never_testable(self) -> None:
        assert _entry_index([], date(2026, 1, 1)) is None

    def test_the_boundary_itself_is_testable_not_off_by_one(self) -> None:
        """A filing dated exactly one day before the very first bar enters at that first bar --
        the earliest legitimate case, distinct from genuinely predating history."""
        bars = _bars(start=datetime(2026, 1, 1, tzinfo=UTC))
        idx = _entry_index(bars, date(2025, 12, 31))
        assert idx == 0


class TestBuildTradeAppliesTheLiteraturesSignNeverAFittedOne:
    def test_positive_sue_goes_long(self) -> None:
        bars = _bars(step_pct=0.002)  # rising
        event = PeadEvent("NVDAUSDT", "NVDA", date(2026, 1, 1), date(2025, 12, 31), sue=2.0)
        trade = _build_trade(
            event, bars, hold_hours=24, cost=CostModel.bitget_perp(), funding_rate_bps=0.0
        )
        assert trade is not None
        assert trade.side == 1
        assert trade.gross_return > 0  # long, rising price -> positive

    def test_negative_sue_goes_short(self) -> None:
        bars = _bars(step_pct=0.002)  # still rising
        event = PeadEvent("NVDAUSDT", "NVDA", date(2026, 1, 1), date(2025, 12, 31), sue=-2.0)
        trade = _build_trade(
            event, bars, hold_hours=24, cost=CostModel.bitget_perp(), funding_rate_bps=0.0
        )
        assert trade is not None
        assert trade.side == -1
        assert trade.gross_return < 0  # short, rising price -> negative

    def test_net_return_is_gross_less_the_full_round_trip_commission(self) -> None:
        bars = _bars(step_pct=0.0)  # flat: isolates the fee
        event = PeadEvent("NVDAUSDT", "NVDA", date(2026, 1, 1), date(2025, 12, 31), sue=1.0)
        cost = CostModel.bitget_perp()
        trade = _build_trade(event, bars, hold_hours=24, cost=cost, funding_rate_bps=0.0)
        assert trade is not None
        assert trade.gross_return == pytest.approx(0.0, abs=1e-12)
        assert trade.net_return == pytest.approx(-float(cost.round_trip_bps()) / 10_000, abs=1e-9)

    def test_a_hold_running_past_the_last_fetched_bar_is_not_realisable(self) -> None:
        bars = _bars(n=30)
        event = PeadEvent("NVDAUSDT", "NVDA", date(2026, 1, 1), date(2025, 12, 31), sue=1.0)
        assert _build_trade(
            event, bars, hold_hours=240, cost=CostModel.bitget_perp(), funding_rate_bps=0.0
        ) is None

    def test_funding_adjusted_is_never_more_favourable_than_net(self) -> None:
        """Funding is a cost or, at best, neutral here -- `abs()` in the estimate means it can
        only ever subtract further, never flatter the commission-only figure."""
        bars = _bars(n=300, step_pct=0.001)
        event = PeadEvent("NVDAUSDT", "NVDA", date(2026, 1, 1), date(2025, 12, 31), sue=1.0)
        trade = _build_trade(
            event, bars, hold_hours=240, cost=CostModel.bitget_perp(), funding_rate_bps=3.0
        )
        assert trade is not None
        assert trade.funding_adjusted <= trade.net_return + 1e-12


class TestSampleVariance:
    def test_matches_the_hand_computed_sample_variance(self) -> None:
        assert _sample_variance([1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_a_single_value_has_no_variance_rather_than_dividing_by_zero(self) -> None:
        assert _sample_variance([5.0]) == 0.0

    def test_an_empty_sequence_has_no_variance_rather_than_dividing_by_zero(self) -> None:
        assert _sample_variance([]) == 0.0


class TestVerdictNamesWhicheverStateActuallyHolds:
    """Each branch is reachable and each message says the thing that is actually true -- the same
    discipline `eval/hurdle.py`'s own `binding_constraint` follows."""

    def test_no_trades_at_all_is_named_no_data(self) -> None:
        assert "NO DATA" in _verdict(
            trades=[], best_hold=None, best_sharpe=None, deflated=None, pbo=None
        )

    def test_trades_below_the_reporting_floor_is_insufficient(self) -> None:
        assert "INSUFFICIENT" in _verdict(
            trades=[object()] * 3, best_hold=None, best_sharpe=None, deflated=None, pbo=None
        )

    def test_a_non_positive_best_sharpe_is_named_no_edge_not_hidden_behind_a_number(self) -> None:
        msg = _verdict(
            trades=[object()] * 20, best_hold=24, best_sharpe=-0.3, deflated=None, pbo=None
        )
        assert "NO EDGE" in msg

    def test_a_positive_sharpe_that_fails_deflation_says_so_by_name(self) -> None:
        msg = _verdict(
            trades=[object()] * 20, best_hold=24, best_sharpe=0.8, deflated=0.4, pbo=None
        )
        assert "DOES NOT SURVIVE DEFLATION" in msg

    def test_deflation_passing_but_pbo_failing_is_not_reported_as_a_clean_survive(self) -> None:
        class _FakePbo:
            pbo = 0.6
        msg = _verdict(
            trades=[object()] * 20, best_hold=24, best_sharpe=0.8, deflated=0.97,
            pbo=_FakePbo(),
        )
        assert "FAILS PBO" in msg
        assert "SURVIVES —" not in msg

    def test_only_deflation_and_pbo_both_passing_is_reported_as_survives(self) -> None:
        class _FakePbo:
            pbo = 0.2
            splits = 70
        msg = _verdict(
            trades=[
                type("T", (), {"event": type("E", (), {"anchor": "NVDA"})()})() for _ in range(20)
            ],
            best_hold=24, best_sharpe=0.8, deflated=0.97, pbo=_FakePbo(),
        )
        assert msg.startswith("SURVIVES —")


class TestHoldHoursStayInsideTheStatedFundingRationale:
    def test_every_hold_is_at_or_under_ten_days(self) -> None:
        assert all(h <= 240 for h in HOLD_HOURS)

    def test_at_least_two_holds_so_a_trial_count_above_one_is_real(self) -> None:
        assert len(HOLD_HOURS) >= 2
