"""Tests for the ARGUS-vs-STUMPY path-shape-matching comparison.

Every case runs BOTH systems' real code: STUMPY's real, vendored `_calculate_squared_distance`
and the real, installed `stumpy.stump()` public API, alongside ARGUS's real `desk.shapematch`.
No live network calls — a seeded synthetic high-persistence series stands in for real market data
throughout. See `eval/shapematch_comparison.py`'s module docstring for the four findings these
tests pin.

`stumpy.stump()`'s first call in a fresh process pays a real, one-time numba JIT-compilation cost
(measured directly: ~26s cold, ~2ms warm) — a module-scoped fixture pays it once so the actual
test assertions run fast.
"""

from __future__ import annotations

import pytest

from argus.eval.baselines.stumpy_squared_distance_loader import load_squared_distance_module
from argus.eval.shapematch_comparison import (
    SCOPE_STATEMENT,
    main,
    measure_costs,
    render,
    run_distance_cases,
    run_exclusion_zone_sweep,
    run_lookahead_check,
    run_reproducibility_check,
    run_significance_scan,
    synthetic_high_persistence_series,
)


@pytest.fixture(scope="module")
def surface():
    return load_squared_distance_module()


@pytest.fixture(scope="module", autouse=True)
def _warm_up_stumpy_jit():
    """Pays STUMPY's real, one-time numba JIT-compilation cost once, so every test after the
    first in this module runs at STUMPY's real steady-state speed."""
    import stumpy

    stumpy.stump([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], m=4)


class TestNumericAgreement:
    def test_the_general_case_agrees_to_float_precision(self, surface) -> None:
        cases = {c.name: c for c in run_distance_cases(surface)}
        assert cases["general"].agrees
        assert cases["general"].diff < 1e-9

    def test_both_constant_window_edge_cases_agree_exactly(self, surface) -> None:
        cases = {c.name: c for c in run_distance_cases(surface)}
        assert cases["left_constant"].agrees
        assert cases["left_constant"].diff == 0.0
        assert cases["both_constant"].agrees
        assert cases["both_constant"].diff == 0.0


class TestLookAhead:
    def test_stumpys_real_naive_self_join_leaks_the_future(self) -> None:
        closes, _stamps = synthetic_high_persistence_series(n=80)
        result = run_lookahead_check(closes, window=12)
        assert result.stumpy_leaks_the_future
        assert result.queries_with_future_neighbour > 0
        assert result.queries_checked > 0


class TestExclusionZoneWidth:
    def test_a_real_measured_share_of_causal_queries_diverge(self) -> None:
        closes, stamps = synthetic_high_persistence_series(n=80)
        sweep = run_exclusion_zone_sweep(closes, stamps, window=12, horizon=1)
        assert sweep.queries_checked > 0
        assert sweep.queries_diverged > 0
        assert sweep.example is not None
        # The example's stumpy pick must be temporally closer to the query than ARGUS's own —
        # the exact tautology ARGUS's own module docstring names.
        assert sweep.example["stumpy_offset"] < sweep.example["argus_offset"]


class TestSignificanceScan:
    def test_stumpys_real_source_has_no_significance_machinery(self) -> None:
        result = run_significance_scan()
        assert result.stumpy_has_no_significance_machinery
        assert result.files_matched == ()

    def test_the_scan_actually_searched_real_files(self) -> None:
        """A scan of zero files would also report `files_matched == ()` — assert it genuinely
        found and read source files, not that the directory was empty or missing."""
        from pathlib import Path

        from argus.eval.shapematch_comparison import _STUMPY_SOURCE

        assert _STUMPY_SOURCE.is_dir()
        assert list(Path(_STUMPY_SOURCE).rglob("*.py"))


class TestCosts:
    def test_costs_are_measured_on_both_real_sides(self, surface) -> None:
        closes, _stamps = synthetic_high_persistence_series(n=80)
        costs = measure_costs(surface, closes)
        assert costs["stumpy_squared_distance_seconds_per_call"] > 0
        assert costs["argus_distance_seconds_per_call"] > 0
        assert costs["stumpy_full_profile_seconds_for_n_bars"] > 0


class TestReproducibility:
    def test_both_are_reproducible_on_identical_input(self) -> None:
        closes, stamps = synthetic_high_persistence_series(n=80)
        result = run_reproducibility_check(closes, stamps)
        assert result["argus_reproducible"]
        assert result["stumpy_reproducible"]


class TestMainAndRender:
    def test_main_returns_a_complete_serialisable_report(self) -> None:
        report = main()
        assert report["all_distance_cases_agree"]
        assert report["lookahead"]["stumpy_leaks_the_future"]
        assert report["exclusion_zone_sweep"]["queries_diverged"] > 0
        assert report["significance_scan"]["stumpy_has_no_significance_machinery"]
        assert report["scope_statement"] == SCOPE_STATEMENT

    def test_render_produces_readable_text(self) -> None:
        report = main()
        text = render(report)
        assert "SHAPEMATCH COMPARISON" in text
        assert "look-ahead" in text
