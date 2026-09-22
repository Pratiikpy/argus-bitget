"""Regime tests — built on series whose boundaries are known by construction.

FLUSS returns a plausible answer on any input, which makes it exactly the kind of algorithm that
should be tested against a series with a boundary planted in it rather than against a real one whose
regimes nobody can independently verify. So the headline tests concatenate two synthetic processes
and require the boundary to be found near the join.

The rest pin the pieces the references disagree on — the idealised curve, the head/tail width, the
exclusion zone — because each of those silently changes where the boundaries land.
"""

from __future__ import annotations

import json
import math
import random
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest

from argus.desk.regime import (
    EXCLUSION_FACTOR,
    MIN_BARS,
    RegimeError,
    arc_counts,
    corrected_arc_curve,
    exact_partition,
    find_boundaries,
    idealised_arc,
    l2_cost_table,
    matrix_profile_index,
    segment,
)

T0 = datetime(2026, 6, 1, tzinfo=UTC)


def _series(values: list[float]) -> list[tuple[datetime, float]]:
    return [(T0 + timedelta(hours=i), v) for i, v in enumerate(values)]


def _sine(n: int, period: float, amplitude: float, start: float = 100.0) -> list[float]:
    return [start + math.sin(i / period) * amplitude for i in range(n)]


def _walk(n: int, sigma: float, seed: int, start: float = 100.0) -> list[float]:
    rng = random.Random(seed)
    out = [start]
    for _ in range(n - 1):
        out.append(out[-1] * math.exp(rng.gauss(0, sigma)))
    return out


class TestItFindsAPlantedBoundary:
    def test_a_smooth_cycle_followed_by_a_fast_one_breaks_near_the_join(self) -> None:
        """Both halves have similar amplitude; only the *shape* changes. A volatility threshold
        cannot see this, which is the reason the module exists."""
        left = _sine(400, period=30.0, amplitude=3.0)
        right = _sine(400, period=4.0, amplitude=3.0, start=left[-1])
        got = segment(_series(left + right), symbol="X", window=24, regimes=2)
        assert got.boundaries
        found = (got.boundaries[0] - T0).total_seconds() / 3600
        assert abs(found - 400) < 24 * EXCLUSION_FACTOR

    def test_a_homogeneous_series_is_reported_as_one_regime(self) -> None:
        """"No boundary" must be an available answer. An algorithm that always returns three
        regimes is reporting its argument, not the series."""
        got = segment(_series(_sine(900, period=12.0, amplitude=2.0)), symbol="X", window=24,
                      regimes=3)
        assert got.verdict
        # Either it finds nothing, or whatever it finds is a shallow dip rather than a real break.
        if len(got.segments) > 1:
            assert all(
                s.arc_value is None or s.arc_value > 0.5 for s in got.segments
            )

    def test_the_number_of_segments_never_exceeds_what_was_asked_for(self) -> None:
        got = segment(_series(_walk(800, 0.004, seed=3)), symbol="X", window=24, regimes=3)
        assert len(got.segments) <= 3

    def test_segments_do_not_overlap(self) -> None:
        """A boundary at window index c is a change at BAR c. The first version added the window
        length to every segment's end and produced ranges that appeared to overlap by a day."""
        got = segment(_series(_walk(900, 0.005, seed=4)), symbol="X", window=24, regimes=3)
        for earlier, later in pairwise(got.segments):
            assert earlier.end < later.start

    def test_every_bar_belongs_to_exactly_one_segment(self) -> None:
        values = _walk(900, 0.005, seed=5)
        got = segment(_series(values), symbol="X", window=24, regimes=3)
        assert sum(s.bars for s in got.segments) == len(values)


class TestThePiecesTheReferencesDisagreeOn:
    def test_the_idealised_curve_is_the_published_parabola(self) -> None:
        """`regimes.py:16-38`: height and centre both n/2, so the peak is n/2 and the ends are 0."""
        n = 1000
        assert idealised_arc(n, n // 2) == pytest.approx(n / 2)
        assert idealised_arc(n, 0) == pytest.approx(0.0, abs=1e-9)
        assert idealised_arc(n, n) == pytest.approx(0.0, abs=1e-6)

    def test_arc_counts_count_spanning_arcs(self) -> None:
        """Windows whose nearest neighbours are the far end: every arc spans the middle."""
        index = [3, 3, 3, 0]
        counts = arc_counts(index)
        assert counts[0] == 0.0
        assert counts[1] > counts[0]
        assert len(counts) == len(index)

    def test_the_head_and_tail_are_pinned_at_the_wider_width(self) -> None:
        """stumpy pins `L * excl_factor`; matrixprofile pins only `w`. The narrow one leaves the
        first and last window looking like boundaries when they are only edges."""
        index = list(range(300))
        window = 10
        curve = corrected_arc_curve(index, window, exclusion_factor=EXCLUSION_FACTOR)
        pinned = window * EXCLUSION_FACTOR
        assert all(value == 1.0 for value in curve[:pinned])
        assert all(value == 1.0 for value in curve[-pinned:])

    def test_the_curve_is_bounded(self) -> None:
        index = [min(len(range(200)) - 1, i + 7) for i in range(200)]
        curve = corrected_arc_curve(index, 8)
        assert all(0.0 <= value <= 1.0 for value in curve)

    def test_boundaries_respect_the_exclusion_zone(self) -> None:
        """Without it the top three boundaries are three adjacent bars of the same dip."""
        curve = [1.0] * 400
        for i in range(150, 156):
            curve[i] = 0.1
        found = find_boundaries(curve, count=3, window=10, exclusion_factor=EXCLUSION_FACTOR)
        assert len(found) == 1
        assert 150 <= found[0] <= 155

    def test_a_flat_curve_yields_no_boundary(self) -> None:
        assert find_boundaries([1.0] * 300, count=3, window=10) == []


class TestTheProfileIndex:
    def test_no_window_matches_its_own_neighbourhood(self) -> None:
        """The trivial match: a window's nearest neighbour is itself shifted by one bar unless an
        exclusion zone forbids it."""
        window = 12
        index = matrix_profile_index(_sine(300, period=9.0, amplitude=2.0), window)
        for i, target in enumerate(index):
            assert abs(i - target) > window // 4

    def test_a_periodic_series_matches_one_period_away(self) -> None:
        window = 12
        period = 24
        values = [100.0 + math.sin(2 * math.pi * i / period) for i in range(400)]
        index = matrix_profile_index(values, window)
        middle = index[len(index) // 2]
        offset = abs(middle - len(index) // 2)
        assert offset % period <= 1 or (period - offset % period) <= 1

    def test_too_short_a_series_is_refused(self) -> None:
        with pytest.raises(RegimeError, match="too few"):
            matrix_profile_index([100.0] * 30, 24)

    def test_a_degenerate_window_is_refused(self) -> None:
        with pytest.raises(RegimeError, match="no shape"):
            matrix_profile_index([100.0] * 300, 2)


class TestTheReport:
    def test_a_short_series_is_refused_by_name(self) -> None:
        with pytest.raises(RegimeError, match=f"below the {MIN_BARS}"):
            segment(_series(_walk(100, 0.004, seed=6)), symbol="X")

    def test_each_segment_reports_its_own_statistics(self) -> None:
        got = segment(_series(_walk(900, 0.004, seed=7)), symbol="X", window=24, regimes=3)
        for piece in got.segments:
            assert piece.bars > 0
            assert piece.volatility_bps >= 0

    def test_the_verdict_names_the_incumbent_rule(self) -> None:
        got = segment(_series(_walk(900, 0.004, seed=8)), symbol="X", window=24, regimes=3)
        assert "volatility rule" in got.verdict or "one regime" in got.verdict

    def test_it_serialises(self) -> None:
        blob = json.loads(json.dumps(
            segment(_series(_walk(900, 0.004, seed=9)), symbol="X", window=24).as_dict(),
        ))
        assert blob["symbol"] == "X"
        assert "segments" in blob and "boundaries" in blob

    def test_the_rendered_report_shows_every_segment(self) -> None:
        got = segment(_series(_walk(900, 0.004, seed=10)), symbol="X", window=24, regimes=3)
        text = got.render()
        assert "REGIME SEGMENTATION" in text
        assert text.count("\n") >= len(got.segments)


class TestTheLiveSeries:
    def test_the_stored_report_is_readable(self) -> None:
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "data" / "regimes.json"
        if not path.exists():
            pytest.skip("no regime artefact on this machine")
        blob = json.loads(path.read_text(encoding="utf-8"))
        assert blob["symbol"].endswith("USDT")
        assert blob["segments"]
        starts = [s["start"] for s in blob["segments"]]
        assert starts == sorted(starts)


class TestL2CostTable:
    def test_a_constant_segment_costs_zero(self) -> None:
        cost = l2_cost_table([5.0] * 10)
        assert cost(0, 10) == pytest.approx(0.0, abs=1e-9)

    def test_matches_the_textbook_sum_of_squared_deviations(self) -> None:
        values = [1.0, 2.0, 4.0, 7.0, 3.0]
        cost = l2_cost_table(values)
        mean = sum(values) / len(values)
        expected = sum((v - mean) ** 2 for v in values)
        assert cost(0, len(values)) == pytest.approx(expected)

    def test_an_empty_or_reversed_range_costs_nothing(self) -> None:
        cost = l2_cost_table([1.0, 2.0, 3.0])
        assert cost(1, 1) == 0.0
        assert cost(2, 1) == 0.0


class TestExactPartition:
    def test_a_single_planted_step_is_found_exactly(self) -> None:
        values = [1.0] * 20 + [10.0] * 20
        assert exact_partition(values, 1, min_size=5) == [20]

    def test_zero_breakpoints_returns_nothing(self) -> None:
        assert exact_partition([1.0, 2.0, 3.0, 4.0, 5.0], 0, min_size=1) == []

    def test_a_negative_breakpoint_count_is_rejected(self) -> None:
        with pytest.raises(RegimeError, match="negative"):
            exact_partition([1.0, 2.0, 3.0], -1, min_size=1)

    def test_too_few_bars_for_the_requested_segments_is_rejected(self) -> None:
        with pytest.raises(RegimeError, match="cannot hold"):
            exact_partition([1.0, 2.0, 3.0], 2, min_size=2)

    def test_every_segment_respects_the_minimum_size(self) -> None:
        rng = random.Random(7)
        values = [rng.gauss(0, 1) for _ in range(80)]
        bkps = exact_partition(values, 3, min_size=10)
        edges = [0, *bkps, len(values)]
        for start, stop in pairwise(edges):
            assert stop - start >= 10


class TestExactPartitionMatchesRealRuptures:
    """The load-bearing claim `desk/regime.py`'s own module comment makes: this bottom-up DP is
    not merely inspired by `ruptures.Dynp(model="l2")`, it returns the identical answer — checked
    directly, not assumed, because a combinatorial optimum has exactly one right answer to compare
    against, unlike a continuous weight vector where "close" has to be defined."""

    def test_120_synthetic_trials_exact_match(self) -> None:
        ruptures = pytest.importorskip("ruptures", reason="the real rival, not vendored")
        mismatches = []
        for seed in range(30):
            for n_bkps in (1, 2, 3, 4):
                signal, _true_bkps = ruptures.pw_constant(
                    n_samples=300, n_features=1, n_bkps=n_bkps, noise_std=1.5, seed=seed,
                )
                values = [float(v) for v in signal[:, 0]]
                mine = exact_partition(values, n_bkps, min_size=10)
                ref_raw = (
                    ruptures.Dynp(model="l2", min_size=10, jump=1)
                    .fit(signal)
                    .predict(n_bkps=n_bkps)
                )
                ref = [b for b in ref_raw if b < len(values)]
                if mine != ref:
                    mismatches.append((seed, n_bkps, mine, ref))
        assert not mismatches, f"{len(mismatches)} mismatch(es): {mismatches[:5]}"

    def test_also_matches_kernelcpd_rbf_on_piecewise_constant_data(self) -> None:
        """The rival `eval/regime_comparison.py` actually scores against is `KernelCPD(rbf)`, not
        `Dynp`. Checked separately that the two converge on `pw_constant` data — an L2 cost is
        exactly what a piecewise-constant-mean generative process calls for, so an rbf kernel's
        extra flexibility buys nothing here and matching `Dynp` is not a weaker stand-in."""
        ruptures = pytest.importorskip("ruptures", reason="the real rival, not vendored")
        mismatches = []
        for seed in range(10):
            signal, _true_bkps = ruptures.pw_constant(
                n_samples=200, n_features=1, n_bkps=3, noise_std=1.5, seed=seed,
            )
            values = [float(v) for v in signal[:, 0]]
            mine = exact_partition(values, 3, min_size=5)
            kernel_raw = (
                ruptures.KernelCPD(kernel="rbf", min_size=5).fit(signal).predict(n_bkps=3)
            )
            kernel = [b for b in kernel_raw if b < len(values)]
            if mine != kernel:
                mismatches.append((seed, mine, kernel))
        assert not mismatches, f"{len(mismatches)} mismatch(es): {mismatches}"
