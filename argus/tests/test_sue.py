"""Tests for `argus.research.sue` — checked against QuantConnect's real, vendored SUE factor.

No live network calls here — all cases use constructed quarterly EPS. The live-data comparison
(real SEC EDGAR EPS, real vendored QuantConnect code, run side by side) is
`tests/test_earnings_comparison.py`.
"""

from __future__ import annotations

from datetime import date
from statistics import pstdev

import pytest

from argus.research.sue import (
    LAG_QUARTERS,
    MAX_DELTA_SPAN_DAYS,
    MIN_QUARTERS,
    N_DELTAS,
    SueError,
    noise_floor,
    rank_universe,
    rank_universe_dated,
    read,
    read_dated,
    sue_from_quarters,
    yoy_deltas,
    yoy_window,
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
        the noise floor, this must still refuse rather than exploding to an absurd
        finite ratio. Below `noise_floor` for EPS of that size, so it must refuse."""
        quarters = [round(1.2 - 0.1 * i, 2) for i in range(MIN_QUARTERS)]
        with pytest.raises(SueError, match="zero"):
            sue_from_quarters(quarters)

    def test_accepts_variance_just_above_the_noise_floor(self) -> None:
        base = [float(12 - i) for i in range(MIN_QUARTERS)]
        base[0] += noise_floor(12.0) * 100
        value = sue_from_quarters(base)
        assert value == value

    def test_the_refusal_decision_does_not_depend_on_units(self) -> None:
        """The absolute 1e-9 threshold this replaced refused this very history once expressed
        in units a billion times smaller; SUE is a ratio and its refusal must be scale-free."""
        for scale in (1e-12, 1e-9, 1e-3, 1.0, 1e3, 1e9):
            scaled = [q * scale for q in REAL_NVDA_SHAPED]
            assert sue_from_quarters(scaled) == pytest.approx(
                sue_from_quarters(REAL_NVDA_SHAPED), rel=1e-9
            )

    def test_refuses_a_constant_yearly_step_on_a_high_eps_level(self) -> None:
        """$10 EPS growing exactly $0.10 a year: zero variance in exact arithmetic, rounding
        noise proportional to the $10 level in binary — the case scikit-learn's and SciPy's
        delta-relative guards were measured to let through as a finite SUE of order 1e13."""
        seasonal = (0.0, 0.03, -0.02, 0.01)
        oldest_first = [round(10 + seasonal[t % 4] + 0.1 * (t // 4), 2) for t in range(12)]
        with pytest.raises(SueError, match="zero variance"):
            sue_from_quarters(oldest_first[::-1])

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


def _quarter_ends(newest: date, count: int, *, skip_month: int | None = None) -> list[date]:
    """``count`` quarter-end dates, newest first, three calendar months apart — dropping every
    quarter ending in ``skip_month`` the way SEC XBRL drops a fiscal fourth quarter."""
    out: list[date] = []
    year, month = newest.year, newest.month
    while len(out) < count:
        if month != skip_month:
            out.append(date(year, month, 28))
        month -= 3
        if month <= 0:
            month += 12
            year -= 1
    return out


class TestReadDated:
    """Year-over-year by date, not by list position (module docstring, correction 1)."""

    def test_equals_the_positional_formula_on_consecutive_quarters(self) -> None:
        ends = _quarter_ends(date(2026, 6, 28), MIN_QUARTERS)
        dated = read_dated("NVDA", list(zip(ends, REAL_NVDA_SHAPED, strict=True)))
        positional = read("NVDA", REAL_NVDA_SHAPED)
        assert dated.sue == positional.sue
        assert dated.eps_std == positional.eps_std
        assert dated.quarters_used == MIN_QUARTERS

    def test_pairs_by_date_when_fiscal_q4_is_missing(self) -> None:
        """Twelve facts with every December quarter absent: the positional formula's
        ``quarters[i + 4]`` is five fiscal quarters back for every delta; the dated one pairs
        each quarter with the same quarter a year earlier and needs more history to get eight."""
        ends = _quarter_ends(date(2026, 6, 28), 15, skip_month=12)
        values = [float(v) for v in range(len(ends), 0, -1)]
        values[0] += 0.37
        points = list(zip(ends, values, strict=True))
        window = yoy_window(points)
        assert len(window) == N_DELTAS
        assert all(350 <= (d.end - d.prior_end).days <= 380 for d in window)
        assert all(d.end.month != 12 for d in window)
        positional_gaps = [(ends[i] - ends[i + LAG_QUARTERS]).days for i in range(N_DELTAS)]
        assert all(gap > 380 for gap in positional_gaps)
        assert read_dated("X", points).sue == pytest.approx(
            window[0].delta / pstdev([d.delta for d in window])
        )

    def test_a_53_week_fiscal_year_still_pairs(self) -> None:
        assert yoy_deltas([(date(2026, 1, 3), 2.0), (date(2024, 12, 28), 1.0)])[0].delta == 1.0

    def test_refuses_when_the_newest_quarter_has_no_prior_year_partner(self) -> None:
        ends = _quarter_ends(date(2026, 6, 28), 16)
        points = list(zip(ends, [float(i % 5) for i in range(16)], strict=True))
        del points[4]  # the June 2025 quarter the newest one needs
        with pytest.raises(SueError, match="no same-quarter-last-year"):
            read_dated("X", points)

    def test_refuses_fewer_than_eight_genuine_changes(self) -> None:
        """Ten facts with December missing reach back to June 2023 and hold seven genuine
        year-over-year pairs, one short."""
        ends = _quarter_ends(date(2026, 6, 28), 10, skip_month=12)
        with pytest.raises(SueError, match="only 7"):
            read_dated("X", list(zip(ends, REAL_NVDA_SHAPED[:10], strict=True)))

    def test_refuses_a_volatility_window_reaching_back_too_far(self) -> None:
        recent = [(date(2026, 6, 28), 3.0), (date(2025, 6, 28), 1.0)]
        old_ends = _quarter_ends(date(2019, 6, 28), 12)
        old = list(zip(old_ends, [float(i % 3 + i) for i in range(12)], strict=True))
        with pytest.raises(SueError, match=str(MAX_DELTA_SPAN_DAYS)):
            read_dated("X", recent + old)

    def test_duplicate_period_ends_are_refused_not_guessed_between(self) -> None:
        with pytest.raises(SueError, match="share one period end"):
            yoy_deltas([(date(2026, 6, 28), 1.0), (date(2026, 6, 28), 2.0)])

    def test_one_quarter_reported_twice_is_refused_not_counted_twice(self) -> None:
        """NVDA's real EDGAR history carries quarters ending 2010-07-31 and 2010-08-01."""
        ends = _quarter_ends(date(2026, 6, 28), MIN_QUARTERS)
        points = list(zip(ends, REAL_NVDA_SHAPED, strict=True))
        points.append((date(2025, 6, 27), 0.99))
        with pytest.raises(SueError, match="overlapping periods"):
            read_dated("X", points)

    def test_refuses_zero_variance_by_date_too(self) -> None:
        ends = _quarter_ends(date(2026, 6, 28), MIN_QUARTERS)
        values = [float(12 - i) for i in range(MIN_QUARTERS)]
        with pytest.raises(SueError, match="zero variance"):
            read_dated("X", list(zip(ends, values, strict=True)))

    def test_rank_universe_dated_reports_every_skip(self) -> None:
        ends = _quarter_ends(date(2026, 6, 28), MIN_QUARTERS)
        ranked, skipped = rank_universe_dated({
            "OK": list(zip(ends, REAL_NVDA_SHAPED, strict=True)),
            "EMPTY": [],
        })
        assert [r.symbol for r in ranked] == ["OK"]
        assert "no quarterly EPS" in skipped["EMPTY"]
