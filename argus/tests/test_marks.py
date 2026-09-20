"""Short-horizon mark tests.

A mark exists to make a refusal gradeable, and the ways it could quietly fail to are all the same
kind of failure: a number recorded without the thing that makes it interpretable.

* The horizon must be **measured**, never assumed from the cycle schedule — a decision taken in the
  last cycle before the overnight gap is next seen eighteen hours later, not two.
* A decision may be marked **once**. Marking it every cycle would pile up overlapping windows,
  which is the exact defect at 24h that this module was built to route around.
* ``lean: none`` is a real answer — the desk saying it cannot call the direction — and must never be
  scored as a wrong call.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from argus.paper.marks import (
    MAX_MARK_HOURS,
    MIN_MARK_HOURS,
    Mark,
    MarkError,
    already_marked,
    due,
    mark,
    read_marks,
)

AT = datetime(2026, 9, 15, 13, 30, tzinfo=UTC)


@dataclass(frozen=True)
class _Entry:
    """The shape `due` reads off a ledger entry."""

    seq: int
    symbol: str
    decided_at: str
    is_abstention: bool = True
    entry_price: str = "100"
    lean: str = "up"


def _entry(seq: int, *, hours_old: float, **kw: object) -> _Entry:
    decided = (AT - timedelta(hours=hours_old)).isoformat()
    return _Entry(seq=seq, symbol="NVDAUSDT", decided_at=decided, **kw)  # type: ignore[arg-type]


class TestTheHorizonIsMeasuredNotAssumed:
    def test_a_two_hour_gap_is_recorded_as_two_hours(self, tmp_path: Path) -> None:
        row = mark(
            seq=1, symbol="NVDAUSDT", decided_at=AT - timedelta(hours=2), marked_at=AT,
            entry_price=Decimal("100"), mark_price=Decimal("101"), lean="up",
            path=tmp_path / "m.jsonl",
        )
        assert row.horizon_hours == pytest.approx(2.0)
        assert row.move == pytest.approx(100.0)

    def test_an_overnight_gap_records_its_real_length(self, tmp_path: Path) -> None:
        """19:30 UTC is not seen again until 13:30 — an 18h horizon wearing a 2h schedule."""
        row = mark(
            seq=1, symbol="NVDAUSDT", decided_at=AT - timedelta(hours=18), marked_at=AT,
            entry_price=Decimal("100"), mark_price=Decimal("100"), lean="down",
            path=tmp_path / "m.jsonl",
        )
        assert row.horizon_hours == pytest.approx(18.0)

    def test_marking_at_or_before_the_decision_is_refused(self, tmp_path: Path) -> None:
        """That would read back the price the decision already had and call it an outcome."""
        with pytest.raises(MarkError):
            mark(
                seq=1, symbol="NVDAUSDT", decided_at=AT, marked_at=AT,
                entry_price=Decimal("100"), mark_price=Decimal("101"), lean="up",
                path=tmp_path / "m.jsonl",
            )

    def test_a_non_positive_entry_price_is_refused_rather_than_divided_by(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(MarkError):
            mark(
                seq=1, symbol="NVDAUSDT", decided_at=AT - timedelta(hours=2), marked_at=AT,
                entry_price=Decimal("0"), mark_price=Decimal("101"), lean="up",
                path=tmp_path / "m.jsonl",
            )


class TestOneMarkPerDecision:
    def test_a_marked_decision_is_not_offered_again(self, tmp_path: Path) -> None:
        path = tmp_path / "m.jsonl"
        entry = _entry(7, hours_old=2)
        assert due([entry], now=AT, path=path) == [entry]
        mark(
            seq=7, symbol="NVDAUSDT", decided_at=datetime.fromisoformat(entry.decided_at),
            marked_at=AT, entry_price=Decimal("100"), mark_price=Decimal("101"), lean="up",
            path=path,
        )
        assert due([entry], now=AT, path=path) == []
        assert already_marked(path) == {7}

    def test_marks_append_rather_than_replace(self, tmp_path: Path) -> None:
        path = tmp_path / "m.jsonl"
        for seq in (1, 2, 3):
            mark(
                seq=seq, symbol="NVDAUSDT", decided_at=AT - timedelta(hours=2), marked_at=AT,
                entry_price=Decimal("100"), mark_price=Decimal("101"), lean="up", path=path,
            )
        assert [m.seq for m in read_marks(path)] == [1, 2, 3]


class TestWhichDecisionsComeDue:
    def test_a_decision_younger_than_the_floor_is_not_marked(self, tmp_path: Path) -> None:
        assert due([_entry(1, hours_old=MIN_MARK_HOURS - 0.5)], now=AT, path=tmp_path / "m") == []

    def test_a_decision_older_than_the_ceiling_is_left_to_settlement(
        self, tmp_path: Path
    ) -> None:
        """A mark and a settlement at the same horizon would look like corroboration."""
        assert due([_entry(1, hours_old=MAX_MARK_HOURS + 1)], now=AT, path=tmp_path / "m") == []

    def test_a_position_is_not_marked_only_an_abstention(self, tmp_path: Path) -> None:
        entry = _entry(1, hours_old=2, is_abstention=False)
        assert due([entry], now=AT, path=tmp_path / "m") == []

    def test_a_decision_inside_the_window_comes_due(self, tmp_path: Path) -> None:
        entry = _entry(1, hours_old=2)
        assert due([entry], now=AT, path=tmp_path / "m") == [entry]


class TestAProtocolCannotBeAppliedBackwards:
    """The guard that caught a real mistake while this module was being wired in.

    When v3 was committed, 48 decisions taken the previous day were still inside the age window.
    Marking them would have added a second observation to decisions whose governing protocol
    declared only one — chosen after their outcomes were already visible on the tape. That is a
    measurement picked with the answer in view, and it is exactly what the commitment forbids.
    """

    def test_a_decision_taken_before_the_governing_protocol_is_not_marked(
        self, tmp_path: Path
    ) -> None:
        older = _entry(231, hours_old=8)
        assert due([older], now=AT, governs_from_seq=232, path=tmp_path / "m") == []

    def test_a_decision_taken_under_the_governing_protocol_is_marked(
        self, tmp_path: Path
    ) -> None:
        governed = _entry(232, hours_old=2)
        assert due([governed], now=AT, governs_from_seq=232, path=tmp_path / "m") == [governed]

    def test_the_floor_is_inclusive_of_the_sequence_it_names(self, tmp_path: Path) -> None:
        """`governs_from_seq` is the first decision governed, not the last one excluded."""
        at_floor = _entry(232, hours_old=2)
        assert due([at_floor], now=AT, governs_from_seq=232, path=tmp_path / "m") == [at_floor]


class TestALeanOfNoneIsAnAnswerNotAMiss:
    def test_no_stated_direction_grades_as_none_rather_than_wrong(self) -> None:
        row = Mark(
            seq=1, symbol="X", decided_at=AT.isoformat(), marked_at=AT.isoformat(),
            horizon_hours=2.0, entry_price="100", mark_price="90", move_bps="-1000", lean="none",
        )
        assert row.is_directional is False
        assert row.lean_was_right is None

    def test_a_correct_up_call_grades_true(self) -> None:
        row = Mark(
            seq=1, symbol="X", decided_at=AT.isoformat(), marked_at=AT.isoformat(),
            horizon_hours=2.0, entry_price="100", mark_price="101", move_bps="100", lean="up",
        )
        assert row.lean_was_right is True

    def test_a_wrong_up_call_grades_false(self) -> None:
        row = Mark(
            seq=1, symbol="X", decided_at=AT.isoformat(), marked_at=AT.isoformat(),
            horizon_hours=2.0, entry_price="100", mark_price="99", move_bps="-100", lean="up",
        )
        assert row.lean_was_right is False

    def test_a_correct_down_call_grades_true(self) -> None:
        row = Mark(
            seq=1, symbol="X", decided_at=AT.isoformat(), marked_at=AT.isoformat(),
            horizon_hours=2.0, entry_price="100", mark_price="99", move_bps="-100", lean="down",
        )
        assert row.lean_was_right is True


class TestAnAbsentFileIsAbsentNotEmpty:
    def test_reading_a_missing_file_yields_no_marks_and_does_not_create_one(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "never_written.jsonl"
        assert read_marks(path) == []
        assert already_marked(path) == set()
        assert not path.exists()

    def test_a_mark_round_trips_through_its_dict(self) -> None:
        row = Mark(
            seq=9, symbol="TSLAUSDT", decided_at=AT.isoformat(), marked_at=AT.isoformat(),
            horizon_hours=2.5, entry_price="100", mark_price="102", move_bps="200", lean="down",
        )
        assert Mark.from_dict(row.as_dict()) == row
