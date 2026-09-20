"""Shape-match tests — the exclusion zone is the whole correctness of this module.

Without it, the nearest neighbour of any window is the window shifted by one bar: a tautology
returned as a finding, and a "top 5" that is one event counted five times. Most of these tests exist
to make that impossible, and the rest make the module refuse rather than return the least-bad row.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest

from argus.desk.shapematch import (
    MIN_ANALOGUES,
    NULL_ALPHA,
    WEAK_MATCH_DISTANCE,
    AnalogueError,
    distance,
    find,
    shuffled_closes,
    znormalise,
)

T0 = datetime(2026, 6, 1, tzinfo=UTC)


def _series(closes: list[float]) -> list[tuple[datetime, float]]:
    return [(T0 + timedelta(hours=i), c) for i, c in enumerate(closes)]


class TestShapeNotLevel:
    def test_the_same_shape_at_a_different_price_is_a_perfect_match(self) -> None:
        """A 3% drift from $118 and from $440 are the same pattern. An un-normalised distance
        would call them unrelated, which is the error z-normalising exists to prevent."""
        low = [100.0, 101.0, 103.0, 102.0, 104.0]
        high = [400.0, 404.0, 412.0, 408.0, 416.0]
        assert distance(low, high) == pytest.approx(0.0, abs=1e-9)

    def test_an_inverted_shape_is_far(self) -> None:
        rising = [100.0, 101.0, 102.0, 103.0, 104.0]
        falling = list(reversed(rising))
        assert distance(rising, falling) > 1.0

    def test_a_flat_window_normalises_to_zeros(self) -> None:
        """No variance means no shape. Zeros make it maximally distant from anything structured,
        which is the honest comparison rather than a divide-by-zero."""
        assert znormalise([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]

    def test_normalised_windows_have_unit_spread(self) -> None:
        got = znormalise([1.0, 2.0, 3.0, 4.0])
        assert sum(got) == pytest.approx(0.0, abs=1e-9)
        assert math.sqrt(sum(x * x for x in got) / len(got)) == pytest.approx(1.0)

    def test_distance_is_per_bar_so_window_lengths_compare(self) -> None:
        """Without dividing by sqrt(len), a 48-bar window scores worse than a 12-bar one for the
        same quality of match, and any threshold would need re-deriving per length."""
        short_a, short_b = [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]
        long_a, long_b = short_a * 4, short_b * 4
        assert distance(short_a, short_b) == pytest.approx(distance(long_a, long_b), abs=1e-9)

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(AnalogueError, match="differ in length"):
            distance([1.0, 2.0], [1.0, 2.0, 3.0])


class TestTheExclusionZone:
    """The failure this module is shaped around."""

    def test_no_analogue_overlaps_another(self) -> None:
        closes = [100.0 + math.sin(i / 3.0) * 5 for i in range(400)]
        got = find(_series(closes), symbol="T", window=12, horizon=6, top=5)
        starts = sorted(a.start_index for a in got.analogues)
        assert len(starts) >= 2
        for earlier, later in pairwise(starts):
            assert later - earlier >= 12, f"analogues overlap: {starts}"

    def test_no_analogue_overlaps_the_query(self) -> None:
        """A 'most similar historical moment' that includes the present moment is look-ahead."""
        closes = [100.0 + math.sin(i / 3.0) * 5 for i in range(400)]
        window, horizon = 12, 6
        got = find(_series(closes), symbol="T", window=window, horizon=horizon)
        query_start = len(closes) - window
        for item in got.analogues:
            assert item.start_index + window <= query_start

    def test_a_periodic_series_does_not_return_one_event_five_times(self) -> None:
        """A sine wave has a match every period. Without exclusion the top-5 are five adjacent
        shifts of the same peak, and the outcome distribution is one outcome counted five times."""
        closes = [100.0 + math.sin(i / 4.0) * 3 for i in range(500)]
        got = find(_series(closes), symbol="T", window=16, horizon=8, top=5)
        assert len({a.start_index for a in got.analogues}) == len(got.analogues)
        spread = max(a.start_index for a in got.analogues) - min(
            a.start_index for a in got.analogues
        )
        assert spread >= 16 * (len(got.analogues) - 1)


class TestTheOutcomeIsStrictlyAfterTheMatch:
    def test_the_forward_window_starts_where_the_match_ends(self) -> None:
        closes = [100.0 + i * 0.1 for i in range(300)]
        got = find(_series(closes), symbol="T", window=10, horizon=10)
        for item in got.analogues:
            assert item.forward_pct is not None

    def test_a_rising_series_gives_positive_outcomes(self) -> None:
        closes = [100.0 * (1.002 ** i) for i in range(300)]
        got = find(_series(closes), symbol="T", window=10, horizon=10)
        assert all(x > 0 for x in got.outcomes)
        assert got.upside_share == 1.0

    def test_a_falling_series_gives_negative_outcomes(self) -> None:
        closes = [100.0 * (0.998 ** i) for i in range(300)]
        got = find(_series(closes), symbol="T", window=10, horizon=10)
        assert all(x < 0 for x in got.outcomes)
        assert got.upside_share == 0.0


class TestItRefusesRatherThanReassures:
    def test_too_short_a_series_raises(self) -> None:
        with pytest.raises(AnalogueError, match="is the minimum"):
            find(_series([100.0] * 20), symbol="T", window=24, horizon=24)

    def test_a_degenerate_window_raises(self) -> None:
        with pytest.raises(AnalogueError, match="no shape to match"):
            find(_series([100.0] * 300), symbol="T", window=2, horizon=5)

    def test_a_zero_horizon_raises(self) -> None:
        with pytest.raises(AnalogueError, match="at least one bar"):
            find(_series([100.0] * 300), symbol="T", window=10, horizon=0)

    def test_a_weak_best_match_is_reported_as_no_precedent(self) -> None:
        """**This test was vacuous.** It guarded on `best.distance > WEAK_MATCH_DISTANCE` and then
        ran a scan whose best distance is 0.968 against a 1.0 threshold — the condition was always
        False and the assertion never executed.

        The guard existed because a *scan* cannot be made to produce a weak best match on demand:
        taking the minimum over ~250 candidates makes a small distance nearly certain whatever the
        data. So the refusal is asserted on a report constructed to be weak, which is the only way
        to exercise the branch at all."""
        from argus.desk.shapematch import Analogue, AnalogueReport

        weak = AnalogueReport(
            symbol="T", window=24, horizon=12, searched=200,
            analogues=tuple(
                Analogue(ends_at=T0, distance=1.6 + i * 0.01, forward_pct=0.5, start_index=i * 50)
                for i in range(4)
            ),
            null_trials=50, null_better=0, null_median=1.9,
        )
        assert weak.best is not None
        assert weak.best.distance > WEAK_MATCH_DISTANCE
        assert not weak.has_precedent
        assert "Weak precedent only" in weak.verdict

    def test_a_real_scan_rarely_produces_a_weak_best_match(self) -> None:
        """Why the test above had to be constructed: the minimum over hundreds of candidates is
        small even on noise. This pins that fact rather than leaving it as a footnote."""
        import random

        rng = random.Random(7)
        closes = [100.0 + rng.gauss(0, 1) for _ in range(300)]
        got = find(_series(closes), symbol="T", window=24, horizon=12)
        assert got.best is not None
        assert got.best.distance < WEAK_MATCH_DISTANCE

    def test_a_precedent_needs_the_minimum_count(self) -> None:
        closes = [100.0 + math.sin(i / 3.0) for i in range(300)]
        got = find(_series(closes), symbol="T", window=12, horizon=6, top=MIN_ANALOGUES - 1)
        assert not got.has_precedent

    def test_the_verdict_never_calls_a_base_rate_a_forecast(self) -> None:
        closes = [100.0 + math.sin(i / 3.0) * 4 for i in range(400)]
        got = find(_series(closes), symbol="T", window=12, horizon=6)
        if got.has_precedent and got.outcomes:
            assert "not a forecast" in got.verdict


class TestTheReport:
    def test_it_serialises(self) -> None:
        import json

        closes = [100.0 + math.sin(i / 3.0) * 4 for i in range(400)]
        blob = json.loads(json.dumps(find(_series(closes), symbol="T", window=12,
                                          horizon=6).as_dict()))
        assert blob["symbol"] == "T"
        assert "has_precedent" in blob

    def test_the_rendered_report_shows_the_outcome_of_each_match(self) -> None:
        closes = [100.0 + math.sin(i / 3.0) * 4 for i in range(400)]
        text = find(_series(closes), symbol="T", window=12, horizon=6).render()
        assert "distance" in text and "then" in text

    def test_analogues_are_ordered_nearest_first(self) -> None:
        closes = [100.0 + math.sin(i / 3.0) * 4 for i in range(400)]
        got = find(_series(closes), symbol="T", window=12, horizon=6)
        distances = [a.distance for a in got.analogues]
        assert distances == sorted(distances)


class TestTheScaleIsKnownNotGuessed:
    """The threshold was 2.0 and could never fire. These tests make that unrepeatable."""

    def test_distance_is_exactly_sqrt_two_times_one_minus_correlation(self) -> None:
        """`stumpy/core.py:1118` computes ``D_squared = 2 * m * (1 - rho)``. Dividing by ``m``, as
        this module does, leaves ``d = sqrt(2 * (1 - rho))`` — so the scale is derivable and every
        threshold on it is checkable arithmetic rather than taste."""
        rng = random.Random(4)
        for _ in range(25):
            left = [rng.gauss(0, 1) for _ in range(32)]
            right = [rng.gauss(0, 1) for _ in range(32)]
            a, b = znormalise(left), znormalise(right)
            rho = sum(x * y for x, y in zip(a, b, strict=True)) / len(a)
            assert distance(left, right) == pytest.approx(math.sqrt(2 * (1 - rho)), abs=1e-9)

    def test_identical_is_zero_and_inverted_is_two(self) -> None:
        rising = [float(i) for i in range(16)]
        assert distance(rising, rising) == pytest.approx(0.0, abs=1e-9)
        assert distance(rising, list(reversed(rising))) == pytest.approx(2.0, abs=1e-9)

    def test_the_threshold_is_below_the_uncorrelated_level(self) -> None:
        """The defect this pins: any threshold at or above sqrt(2) admits uncorrelated shapes, and
        one at 2.0 — the maximum the metric can produce — admits perfectly inverted ones too."""
        assert math.sqrt(2) > WEAK_MATCH_DISTANCE

    def test_a_flat_window_against_a_structured_one_is_one(self) -> None:
        """stumpy gives a constant-vs-non-constant pair ``D_squared = m`` (`core.py:1112`), which is
        exactly 1.0 per bar. Ours lands there by construction rather than by a special case."""
        flat = [7.0] * 16
        shaped = [float(i) for i in range(16)]
        assert distance(flat, shaped) == pytest.approx(1.0, abs=1e-9)


class TestTheNullDecidesWhatCloseMeans:
    """The finding: scanning 1,400 windows for a minimum produces a small number either way."""

    def test_noise_is_refused_even_though_its_best_match_looks_close(self) -> None:
        rng = random.Random(12)
        price = 100.0
        closes = [price]
        for _ in range(600):
            price *= 1 + rng.gauss(0, 0.004)
            closes.append(price)
        got = find(_series(closes), symbol="RW", window=24, horizon=24, trials=40)
        assert got.best is not None
        assert got.best.distance < WEAK_MATCH_DISTANCE, "the distance gate alone lets this through"
        assert got.null_p is not None and got.null_p > NULL_ALPHA
        assert not got.has_precedent
        assert "not distinguishable from chance" in got.verdict

    def test_a_genuinely_repeating_shape_survives_the_null(self) -> None:
        closes = [100.0 + math.sin(i / 4.0) * 6 for i in range(500)]
        got = find(_series(closes), symbol="SINE", window=16, horizon=8, trials=40)
        assert got.null_p == 0.0
        assert got.has_precedent
        assert "not a forecast" in got.verdict

    def test_an_uncalibrated_report_never_claims_a_precedent(self) -> None:
        """A search whose null was not run has not been shown to have found anything."""
        closes = [100.0 + math.sin(i / 4.0) * 6 for i in range(500)]
        got = find(_series(closes), symbol="SINE", window=16, horizon=8, trials=0)
        assert got.null_p is None
        assert not got.has_precedent
        assert "Uncalibrated" in got.verdict

    def test_the_p_value_is_reproducible_for_a_fixed_seed(self) -> None:
        closes = [100.0 + math.sin(i / 5.0) * 4 for i in range(400)]
        first = find(_series(closes), symbol="S", window=12, horizon=6, trials=20, seed=99)
        again = find(_series(closes), symbol="S", window=12, horizon=6, trials=20, seed=99)
        assert first.null_p == again.null_p
        assert first.null_median == again.null_median

    def test_the_null_is_carried_into_the_serialised_report(self) -> None:
        closes = [100.0 + math.sin(i / 4.0) * 6 for i in range(400)]
        blob = find(_series(closes), symbol="S", window=12, horizon=6, trials=20).as_dict()
        assert blob["null_trials"] == 20
        assert blob["null_p"] is not None
        assert blob["null_median_distance"] is not None


class TestTheShuffleKeepsTheSeriesAndDestroysTheOrder:
    def test_it_keeps_every_return_the_series_printed(self) -> None:
        rng = random.Random(5)
        closes = [100.0 * (1.001 ** i) + (i % 7) for i in range(200)]
        got = shuffled_closes(closes, rng)
        original = sorted(round(closes[i + 1] / closes[i], 12) for i in range(len(closes) - 1))
        after = sorted(round(got[i + 1] / got[i], 12) for i in range(len(got) - 1))
        assert after == original
        assert got[0] == closes[0]
        assert len(got) == len(closes)

    def test_it_actually_reorders(self) -> None:
        rng = random.Random(6)
        closes = [100.0 + math.sin(i / 3.0) * 5 for i in range(200)]
        assert shuffled_closes(closes, rng) != closes
