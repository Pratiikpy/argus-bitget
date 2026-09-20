"""Tests for `argus.research.sue` — checked against QuantConnect's real, vendored SUE factor.

No live network calls here — all cases use constructed quarterly EPS. The live-data comparison
(real SEC EDGAR EPS, real vendored QuantConnect code, run side by side) is
`tests/test_earnings_comparison.py`.
"""

from __future__ import annotations

import pytest

from argus.research.sue import (
    LAG_QUARTERS,
    MIN_QUARTERS,
    N_DELTAS,
    ZERO_VARIANCE_TOLERANCE,
    SueError,
    rank_universe,
    read,
    sue_from_quarters,
)

REAL_NVDA_SHAPED = [2.46, 1.91, 1.5, 0.89, 0.81, 0.78, 0.61, 0.52, 0.4, 0.35, 0.27, 0.18]
"""Shaped like NVDA's real quarterly EPS sequence (see `eval/earnings_comparison.py`'s own real
fetch) — not asserted against a pinned value here since real EDGAR data drifts; the pinned,
exact-agreement assertion against the real vendored function lives in the live comparison test."""


class TestConstants:
    def test_min_quarters_is_lag_plus_deltas(self) -> None:
        assert MIN_QUARTERS == LAG_QUARTERS + N_DELTAS


class TestSueFromQuarters:
    def test_computes_on_real_shaped_data(self) -> None:
        value = sue_from_quarters(REAL_NVDA_SHAPED)
        assert isinstance(value, float)
        assert value == value  # not nan
        assert value not in (float("inf"), float("-inf"))

    def test_refuses_below_the_minimum(self) -> None:
        with pytest.raises(SueError, match="below the 12"):
            sue_from_quarters([1.0] * (MIN_QUARTERS - 1))

    def test_refuses_exact_zero_variance(self) -> None:
        """Integer EPS with a constant year-over-year delta — exact in floating point, so the
        real reference's own denominator is exactly 0.0, not float noise."""
        quarters = [float(12 - i) for i in range(MIN_QUARTERS)]
        with pytest.raises(SueError, match="zero"):
            sue_from_quarters(quarters)

    def test_refuses_all_flat_eps(self) -> None:
        with pytest.raises(SueError, match="zero"):
            sue_from_quarters([0.0] * MIN_QUARTERS)

    def test_refuses_float_noise_near_zero_variance(self) -> None:
        """A decimal (not integer) linear sequence: mathematically zero variance, but binary
        floating-point rounding leaves a non-zero residual of order 1e-17 — below
        ZERO_VARIANCE_TOLERANCE, this must still refuse rather than exploding to an absurd
        finite ratio."""
        quarters = [round(1.2 - 0.1 * i, 2) for i in range(MIN_QUARTERS)]
        with pytest.raises(SueError, match="zero"):
            sue_from_quarters(quarters)

    def test_accepts_variance_just_above_tolerance(self) -> None:
        base = [float(12 - i) for i in range(MIN_QUARTERS)]
        base[0] += ZERO_VARIANCE_TOLERANCE * 100
        value = sue_from_quarters(base)
        assert value == value

    def test_more_than_the_minimum_quarters_is_accepted(self) -> None:
        padded = [*REAL_NVDA_SHAPED, 0.1, 0.2, 0.3]
        value = sue_from_quarters(padded)
        assert value == value


class TestRead:
    def test_carries_the_intermediate_values(self) -> None:
        result = read("NVDA", REAL_NVDA_SHAPED)
        assert result.symbol == "NVDA"
        assert result.quarters_used == len(REAL_NVDA_SHAPED)
        assert result.eps_std > 0

    def test_as_dict_is_json_serialisable(self) -> None:
        import json

        json.dumps(read("NVDA", REAL_NVDA_SHAPED).as_dict())


class TestRankUniverse:
    def test_ranks_by_sue_descending(self) -> None:
        universe = {
            "HIGH": [10.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            "LOW": [5.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        }
        ranked, skipped = rank_universe(universe)
        assert not skipped
        assert ranked[0].symbol in ("HIGH", "LOW")
        assert ranked[0].sue >= ranked[1].sue

    def test_reports_skipped_symbols_with_a_reason_not_silently(self) -> None:
        universe = {
            "OK": REAL_NVDA_SHAPED,
            "TOO_SHORT": [1.0] * 5,
            "ZERO_VARIANCE": [float(12 - i) for i in range(MIN_QUARTERS)],
        }
        ranked, skipped = rank_universe(universe)
        assert {r.symbol for r in ranked} == {"OK"}
        assert set(skipped) == {"TOO_SHORT", "ZERO_VARIANCE"}
        assert all(isinstance(reason, str) and reason for reason in skipped.values())

    def test_empty_universe_returns_empty_results(self) -> None:
        ranked, skipped = rank_universe({})
        assert ranked == []
        assert skipped == {}
