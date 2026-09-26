"""Pins `points()` and `fit()` in `argus.eval.impact_calibration` on hand-built order-book sweeps.

Both functions are pure arithmetic over plain dicts and dataclasses — no console, engine or network
call to stub. `measure()` is the module's own live entry point (`# pragma: no cover - live`, it
fetches Bitget tickers and candle history) and is intentionally left untested here for the same
reason the module itself excludes it from coverage.

Three inputs make `fit()` raise instead of returning a defined result: no rows, rows that all have
zero walk, and rows with exactly one positive-walk row. Each is marked `xfail(strict=True)` below
rather than worked around, and is reported as a bug.
"""

from __future__ import annotations

import pytest

from argus.eval.impact_calibration import SweepPoint, fit, points


class TestPoints:
    def test_only_complete_buy_sweeps_on_a_known_symbol_are_kept(self) -> None:
        depth = {"books": [
            {"symbol": "AUSDT", "spread_bps": 10, "sweeps": [
                {"side": "BUY", "complete": True, "requested_notional": 1000,
                 "slippage_bps": 20},
                {"side": "SELL", "complete": True, "requested_notional": 5000,
                 "slippage_bps": 30},
                {"side": "BUY", "complete": False, "requested_notional": 5000,
                 "slippage_bps": 30},
                {"side": "BUY", "complete": True, "requested_notional": 100,
                 "slippage_bps": 3},
            ]},
            # no ADV for BUSDT, zero ADV for CUSDT, no sigma for DUSDT: each book is dropped whole.
            {"symbol": "BUSDT", "spread_bps": 8, "sweeps": [
                {"side": "BUY", "complete": True, "requested_notional": 2000,
                 "slippage_bps": 25}]},
            {"symbol": "CUSDT", "spread_bps": 6, "sweeps": [
                {"side": "BUY", "complete": True, "requested_notional": 3000,
                 "slippage_bps": 40}]},
            {"symbol": "DUSDT", "spread_bps": 4, "sweeps": [
                {"side": "BUY", "complete": True, "requested_notional": 4000,
                 "slippage_bps": 50}]},
        ]}
        adv = {"AUSDT": 1_000_000.0, "CUSDT": 0.0, "DUSDT": 2_000_000.0}
        sigma_bps = {"AUSDT": 50.0, "CUSDT": 40.0, "BUSDT": 60.0}
        result = points(depth, adv, sigma_bps)
        assert result == [
            SweepPoint("AUSDT", 1000.0, 0.001, 50.0, 15.0),
            # the $100 sweep's slippage (3bps) is inside half the spread (5bps): walk clips to 0.
            SweepPoint("AUSDT", 100.0, 0.0001, 50.0, 0.0),
        ]

    def test_a_book_with_no_sweeps_at_all_gives_no_points(self) -> None:
        assert points({"books": []}, {"AUSDT": 1.0}, {"AUSDT": 1.0}) == []
        assert points({}, {"AUSDT": 1.0}, {"AUSDT": 1.0}) == []


class TestFit:
    def test_b_and_r_squared_and_free_exponent_on_a_perfect_square_root_fit(self) -> None:
        rows = [
            SweepPoint("AUSDT", 1000.0, 0.01, 100.0, 10.0),
            SweepPoint("AUSDT", 4000.0, 0.04, 100.0, 20.0),
        ]
        result = fit(rows)
        # walk == sigma * sqrt(participation) exactly for both rows, so b = 1, the fit is
        # perfect, and the free log-log exponent recovers the textbook square root, 0.5.
        assert result["b"] == pytest.approx(1.0)
        assert result["r_squared"] == pytest.approx(1.0)
        assert result["free_exponent"] == pytest.approx(0.5)
        assert result["median_sigma_daily_bps"] == pytest.approx(100.0)
        assert result["median_walk_bps"] == pytest.approx(15.0)
        assert result["median_old_model_bps"] == pytest.approx(0.045)

    def test_an_off_fit_gives_b_below_one_and_r_squared_below_one(self) -> None:
        rows = [
            SweepPoint("AUSDT", 1000.0, 0.01, 100.0, 10.0),
            SweepPoint("AUSDT", 4000.0, 0.04, 100.0, 20.0),
            SweepPoint("AUSDT", 9000.0, 0.09, 100.0, 25.0),
        ]
        result = fit(rows)
        assert result["b"] == pytest.approx(0.8928571428571429)
        assert result["r_squared"] == pytest.approx(0.923469387755102)
        assert result["free_exponent"] == pytest.approx(0.4259571360984685)
        assert result["median_walk_bps"] == pytest.approx(20.0)
        assert result["median_old_model_bps"] == pytest.approx(0.08)

    def test_no_rows_gives_a_result_instead_of_crashing(self) -> None:
        """Was a ZeroDivisionError (impact_calibration.py, fixed 2026-09-27)."""
        assert set(fit([]).values()) == {None}

    def test_all_zero_walk_rows_gives_a_result_instead_of_crashing(self) -> None:
        rows = [
            SweepPoint("AUSDT", 1000.0, 0.01, 100.0, 0.0),
            SweepPoint("AUSDT", 4000.0, 0.04, 100.0, 0.0),
        ]
        # Was a StatisticsError: no sweep walked the book, so there is no exponent to fit.
        result = fit(rows)
        assert result["free_exponent"] is None and result["b"] == 0.0

    def test_one_positive_walk_row_among_several_gives_a_result_instead_of_crashing(self) -> None:
        rows = [
            SweepPoint("AUSDT", 1000.0, 0.01, 100.0, 10.0),
            SweepPoint("AUSDT", 4000.0, 0.04, 100.0, 0.0),
        ]
        # Was a ZeroDivisionError: one walked sweep is one point, and a line through it has
        # no slope. The through-origin fit over both sweeps is still defined.
        result = fit(rows)
        assert result["free_exponent"] is None and result["b"] is not None
