"""Book-calibration tests.

This module had none. It measures the parameters the queue experiment runs on, so an error here
propagates into a capability the standing register tracks — and it shipped a pooled change figure
for a tape whose snapshot gaps spanned 14 seconds to 2.5 hours.

The properties under test are the ones that let a wrong number look like a measurement:

* a change observation is meaningless without the horizon it was measured over, so horizons must
  not be pooled;
* a horizon the tape never sampled reports ``None``, never ``0.0`` — a level nobody watched for ten
  minutes did not hold still for ten minutes;
* a price present in one snapshot and absent from the next is *not* a cancellation to zero; it may
  have fallen outside recorded depth, and counting it would invent the largest moves in the sample.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

from argus.eval.bookcalib import (
    HORIZONS,
    MIN_HORIZON_OBSERVATIONS,
    NEAR_TOUCH,
    Snapshot,
    _changes,
    _horizon_of,
    calibrate,
)

AT = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


def _snap(symbol: str, when: datetime, bids: list[tuple[float, float]]) -> Snapshot:
    return Snapshot.from_dict({
        "taken_at": when.isoformat(),
        "symbol": symbol,
        "bids": [list(b) for b in bids],
        "asks": [],
    })


def _write(tmp_path: Path, snapshots: list[Snapshot]) -> Path:
    path = tmp_path / "tape.jsonl"
    lines = []
    for s in snapshots:
        bids = ",".join(f"[{p},{q}]" for p, q in s.bids)
        lines.append(
            f'{{"taken_at":"{s.taken_at}","symbol":"{s.symbol}","bids":[{bids}],"asks":[]}}'
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestAHorizonIsPartOfTheMeasurement:
    def test_every_gap_lands_in_exactly_one_bucket(self) -> None:
        for seconds in (0.0, 29.9, 30.0, 119.9, 120.0, 599.0, 3599.0, 3600.0, 86_400.0):
            assert _horizon_of(seconds) in {name for name, _, _ in HORIZONS}

    def test_the_buckets_are_contiguous_and_cover_everything(self) -> None:
        """A gap falling between two buckets would be dropped from every distribution."""
        for (_, _, high), (_, low, _) in pairwise(HORIZONS):
            assert high == low
        assert HORIZONS[0][1] == 0.0
        assert HORIZONS[-1][2] == float("inf")

    def test_short_and_long_gaps_are_not_averaged_together(self, tmp_path: Path) -> None:
        """The defect this module shipped: 14s and 2.5h observations pooled into one median."""
        near = [
            _snap("X", AT, [(100.0, 10.0)]),
            _snap("X", AT + timedelta(seconds=10), [(100.0, 9.0)]),
        ]
        far = [
            _snap("Y", AT, [(100.0, 10.0)]),
            _snap("Y", AT + timedelta(hours=2), [(100.0, 2.0)]),
        ]
        report = calibrate(path=_write(tmp_path, near + far))
        rows = report["change_by_horizon"]
        assert rows["under_30s"]["observations"] == 1
        assert rows["over_1h"]["observations"] == 1
        assert (
            rows["under_30s"]["median_share_of_level"]
            != rows["over_1h"]["median_share_of_level"]
        )

    def test_an_unsampled_horizon_reports_none_rather_than_zero(self, tmp_path: Path) -> None:
        pair = [
            _snap("X", AT, [(100.0, 10.0)]),
            _snap("X", AT + timedelta(seconds=5), [(100.0, 9.0)]),
        ]
        rows = calibrate(path=_write(tmp_path, pair))["change_by_horizon"]
        assert rows["10min_to_1h"]["observations"] == 0
        assert rows["10min_to_1h"]["median_share_of_level"] is None
        assert rows["10min_to_1h"]["share_growing"] is None

    def test_the_pooled_figure_survives_but_carries_its_warning(self, tmp_path: Path) -> None:
        """Kept for comparability with earlier artefacts, never without saying what it is."""
        pair = [
            _snap("X", AT, [(100.0, 10.0)]),
            _snap("X", AT + timedelta(seconds=5), [(100.0, 9.0)]),
        ]
        pooled = calibrate(path=_write(tmp_path, pair))["change_pooled"]
        assert "incommensurable" in pooled["warning"]
        assert "Read `change_by_horizon`" in pooled["warning"]

    def test_the_verdict_refuses_to_name_a_figure_below_the_observation_floor(
        self, tmp_path: Path
    ) -> None:
        pair = [
            _snap("X", AT, [(100.0, 10.0)]),
            _snap("X", AT + timedelta(seconds=5), [(100.0, 9.0)]),
        ]
        verdict = calibrate(path=_write(tmp_path, pair))["verdict"]
        assert f"No horizon yet carries {MIN_HORIZON_OBSERVATIONS}" in verdict


class TestAbsenceIsNotACancellation:
    def test_a_price_that_leaves_the_recorded_depth_is_skipped_not_zeroed(self) -> None:
        """Counting it would invent the single largest change in the sample."""
        before = _snap("X", AT, [(100.0, 10.0), (99.0, 5.0)])
        after = _snap("X", AT + timedelta(seconds=5), [(100.0, 10.0)])
        shares, grew, shrank = _changes(before, after)
        assert shares == [0.0]
        assert (grew, shrank) == (0, 0)

    def test_a_level_that_actually_shrank_is_counted(self) -> None:
        before = _snap("X", AT, [(100.0, 10.0)])
        after = _snap("X", AT + timedelta(seconds=5), [(100.0, 4.0)])
        shares, grew, shrank = _changes(before, after)
        assert shares == [0.6]
        assert (grew, shrank) == (0, 1)

    def test_only_levels_near_the_touch_are_counted(self) -> None:
        deep = [(100.0 - i, 10.0) for i in range(NEAR_TOUCH + 5)]
        before = _snap("X", AT, deep)
        after = _snap("X", AT + timedelta(seconds=5), deep)
        shares, _, _ = _changes(before, after)
        assert len(shares) == NEAR_TOUCH


class TestAnEmptyTapeMakesNoClaim:
    def test_no_tape_reports_unmeasured_rather_than_zero(self, tmp_path: Path) -> None:
        report = calibrate(path=tmp_path / "absent.jsonl")
        assert report["snapshots"] == 0
        assert "unmeasured" in report["verdict"]

    def test_a_single_snapshot_measures_size_but_not_change(self, tmp_path: Path) -> None:
        """One snapshot has no elapsed time in it, so change is absent rather than zero."""
        report = calibrate(path=_write(tmp_path, [_snap("X", AT, [(100.0, 10.0)])]))
        assert report["level_size"]["count"] == 1
        assert report["change"] is None
