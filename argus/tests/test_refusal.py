"""Refusal Alpha tests.

This module decides whether a desk that never trades is doing its job, so the ways it could quietly
lie are all ways of turning an absence into a verdict:

* no marks must read **UNDEFINED**, never 0% accuracy — a desk that has not been measured is not a
  desk that scored zero;
* a sample too small to distinguish from a coin flip must say so, rather than quoting a point
  estimate with a decimal point;
* ``lean: none`` must never count as a wrong call, or the honest answer becomes the expensive one;
* horizons must not be pooled, since a 2h window and an 18h window are different measurements.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from argus.eval.refusal import (
    MIN_FOR_A_RATE,
    ROUND_TRIP_BPS,
    HorizonResult,
    orphans,
    score,
    wilson,
)
from argus.paper.marks import Mark

AT = datetime(2026, 9, 15, 13, 30, tzinfo=UTC)


def _mark(*, horizon: float, move: float, lean: str, seq: int = 1) -> Mark:
    return Mark(
        seq=seq, symbol="NVDAUSDT", decided_at=AT.isoformat(), marked_at=AT.isoformat(),
        horizon_hours=horizon, entry_price="100", mark_price="101", move_bps=str(move), lean=lean,
    )


class TestNothingMeasuredIsNotZero:
    def test_no_marks_reports_undefined_rather_than_a_zero_accuracy(self) -> None:
        report = score([], now=AT)
        assert report.total_marks == 0
        assert "UNDEFINED" in report.verdict
        assert "not zero" in report.verdict
        for row in report.horizons:
            assert row.accuracy is None
            assert row.coverage is None

    def test_no_marks_serialises_with_nulls_not_zeroes(self) -> None:
        payload = score([], now=AT).as_dict()
        for row in payload["horizons"]:
            assert row["accuracy_pct"] is None
            assert row["coverage_pct"] is None
            assert row["accuracy_ci95"] is None

    def test_a_horizon_nobody_sampled_stays_none_while_another_has_data(self) -> None:
        report = score([_mark(horizon=2.0, move=50.0, lean="up")], now=AT)
        rows = {r.horizon: r for r in report.horizons}
        assert rows["about_2h"].marks == 1
        assert rows["overnight_12h_plus"].accuracy is None


class TestASampleTooSmallSaysSo:
    def test_a_handful_of_calls_does_not_earn_a_rate(self) -> None:
        marks = [_mark(horizon=2.0, move=50.0, lean="up", seq=i) for i in range(5)]
        row = next(r for r in score(marks, now=AT).horizons if r.horizon == "about_2h")
        assert row.correct == 5
        assert row.accuracy == 1.0
        # Five for five, and still not evidence.
        assert row.beats_a_coin is None
        assert row.as_dict()["enough_for_a_rate"] is False

    def test_the_verdict_names_the_shortfall_rather_than_quoting_a_rate(self) -> None:
        marks = [_mark(horizon=2.0, move=50.0, lean="up", seq=i) for i in range(5)]
        verdict = score(marks, now=AT).verdict
        assert "UNDEFINED" in verdict
        assert str(MIN_FOR_A_RATE) in verdict

    def test_an_adequate_sample_that_is_indistinguishable_from_chance_says_that(self) -> None:
        """A 50/50 split at n=40 must not be reported as 'the lean carries information'."""
        marks = [
            _mark(horizon=2.0, move=50.0 if i % 2 else -50.0, lean="up", seq=i)
            for i in range(40)
        ]
        row = next(r for r in score(marks, now=AT).horizons if r.horizon == "about_2h")
        assert row.directional == 40
        assert row.beats_a_coin is False
        assert "does not show the lean beating a coin flip" in score(marks, now=AT).verdict

    def test_a_strong_adequate_sample_is_allowed_to_claim_information(self) -> None:
        marks = [
            _mark(horizon=2.0, move=50.0 if i < 38 else -50.0, lean="up", seq=i)
            for i in range(40)
        ]
        row = next(r for r in score(marks, now=AT).horizons if r.horizon == "about_2h")
        assert row.beats_a_coin is True
        assert "excludes a coin flip" in score(marks, now=AT).verdict


class TestDecliningToCallIsNotAMiss:
    def test_a_none_lean_is_excluded_from_the_denominator(self) -> None:
        marks = [
            _mark(horizon=2.0, move=50.0, lean="up", seq=1),
            _mark(horizon=2.0, move=-900.0, lean="none", seq=2),
            _mark(horizon=2.0, move=-900.0, lean="none", seq=3),
        ]
        row = next(r for r in score(marks, now=AT).horizons if r.horizon == "about_2h")
        assert row.marks == 3
        assert row.directional == 1
        assert row.correct == 1
        assert row.accuracy == 1.0

    def test_coverage_reports_how_often_a_direction_was_stated_at_all(self) -> None:
        """A desk answering "none" every time cannot be wrong and has said nothing."""
        marks = [
            _mark(horizon=2.0, move=50.0, lean="none", seq=1),
            _mark(horizon=2.0, move=50.0, lean="none", seq=2),
            _mark(horizon=2.0, move=50.0, lean="up", seq=3),
            _mark(horizon=2.0, move=50.0, lean="down", seq=4),
        ]
        row = next(r for r in score(marks, now=AT).horizons if r.horizon == "about_2h")
        assert row.coverage == 0.5


class TestHorizonsAreNotPooled:
    def test_a_two_hour_mark_and_an_overnight_mark_land_in_different_buckets(self) -> None:
        marks = [
            _mark(horizon=2.0, move=50.0, lean="up", seq=1),
            _mark(horizon=18.0, move=-50.0, lean="up", seq=2),
        ]
        rows = {r.horizon: r for r in score(marks, now=AT).horizons}
        assert (rows["about_2h"].marks, rows["about_2h"].correct) == (1, 1)
        assert (rows["overnight_12h_plus"].marks, rows["overnight_12h_plus"].correct) == (1, 0)

    def test_a_mark_outside_every_bucket_is_counted_not_dropped(self) -> None:
        """Silently discarding it would shrink the denominator without saying so."""
        report = score([_mark(horizon=40.0, move=50.0, lean="up")], now=AT)
        assert report.outside_buckets == 1
        assert sum(r.marks for r in report.horizons) == 0
        assert report.total_marks == 1


class TestForgoneEdgeIsReportedEvenWhenItEmbarrasses:
    def test_a_correct_lean_on_a_big_move_shows_refusing_cost_money(self) -> None:
        marks = [_mark(horizon=2.0, move=100.0, lean="up", seq=i) for i in range(4)]
        row = next(r for r in score(marks, now=AT).horizons if r.horizon == "about_2h")
        assert row.median_forgone_bps == pytest.approx(100.0 - ROUND_TRIP_BPS)

    def test_a_wrong_lean_shows_refusing_saved_money(self) -> None:
        marks = [_mark(horizon=2.0, move=-100.0, lean="up", seq=i) for i in range(4)]
        row = next(r for r in score(marks, now=AT).horizons if r.horizon == "about_2h")
        assert row.median_forgone_bps == pytest.approx(-100.0 - ROUND_TRIP_BPS)

    def test_a_down_lean_on_a_falling_price_is_a_gain_forgone(self) -> None:
        """Leaning down and the price falling means the desk was right to want to be short."""
        marks = [_mark(horizon=2.0, move=-100.0, lean="down", seq=i) for i in range(4)]
        row = next(r for r in score(marks, now=AT).horizons if r.horizon == "about_2h")
        assert row.median_forgone_bps == pytest.approx(100.0 - ROUND_TRIP_BPS)

    def test_forgone_edge_is_computed_with_no_sample_size_gate(self) -> None:
        """It is the number most able to embarrass the desk, so it is never withheld."""
        row = next(
            r for r in score([_mark(horizon=2.0, move=500.0, lean="up")], now=AT).horizons
            if r.horizon == "about_2h"
        )
        assert row.median_forgone_bps is not None
        assert row.beats_a_coin is None


class TestTheHurdleClearanceIsALowerBound:
    def test_a_move_below_the_round_trip_does_not_clear(self) -> None:
        row = next(
            r for r in score([_mark(horizon=2.0, move=5.0, lean="up")], now=AT).horizons
            if r.horizon == "about_2h"
        )
        assert row.cleared_hurdle == 0
        assert row.clearance == 0.0

    def test_a_large_move_in_either_direction_clears(self) -> None:
        marks = [
            _mark(horizon=2.0, move=200.0, lean="up", seq=1),
            _mark(horizon=2.0, move=-200.0, lean="up", seq=2),
        ]
        row = next(r for r in score(marks, now=AT).horizons if r.horizon == "about_2h")
        assert row.cleared_hurdle == 2


class TestTheInterval:
    def test_an_empty_sample_has_no_interval(self) -> None:
        assert wilson(0, 0) is None

    def test_the_interval_never_leaves_the_unit_range(self) -> None:
        """The normal approximation returns a bound below zero here; Wilson does not."""
        low, high = wilson(0, 5) or (None, None)
        assert low is not None and high is not None
        assert 0.0 <= low <= high <= 1.0

    def test_a_wider_sample_gives_a_tighter_interval(self) -> None:
        narrow = wilson(80, 100)
        wide = wilson(8, 10)
        assert narrow is not None and wide is not None
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


class TestTheReportIsReadable:
    def test_every_horizon_appears_in_the_rendered_table(self) -> None:
        text = score([], now=AT).render()
        for name in ("about_2h", "4h_to_12h", "overnight_12h_plus"):
            assert name in text

    def test_a_horizon_result_with_no_directional_calls_has_no_accuracy(self) -> None:
        row = HorizonResult("about_2h", marks=3, directional=0, correct=0, cleared_hurdle=1,
                            forgone_bps=())
        assert row.accuracy is None
        assert row.median_forgone_bps is None
        assert row.coverage == 0.0


class TestAMarkNeedsADecisionBehindIt:
    """The guard added after a hand-verification wrote two synthetic rows into the live file.

    They had prices of 100 and 200 and sequence numbers no decision had reached. Nothing in the
    code objected. A fabricated row in the file that feeds a published metric is the exact failure
    this project is built against, so it is now refused rather than scored.
    """

    def test_a_mark_with_no_decision_behind_it_is_named(self, tmp_path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        ledger.write_text('{"seq": 1}\n{"seq": 2}\n', encoding="utf-8")
        stray = orphans(
            [_mark(horizon=2.0, move=10.0, lean="up", seq=1),
             _mark(horizon=2.0, move=10.0, lean="up", seq=999)],
            ledger_path=ledger,
        )
        assert stray == [999]

    def test_marks_that_all_match_the_ledger_leave_no_orphans(self, tmp_path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        ledger.write_text('{"seq": 1}\n{"seq": 2}\n', encoding="utf-8")
        assert orphans([_mark(horizon=2.0, move=10.0, lean="up", seq=2)], ledger_path=ledger) == []

    def test_a_missing_ledger_makes_every_mark_unverifiable_not_valid(self, tmp_path) -> None:
        """Nothing to check against is not the same as everything checking out."""
        stray = orphans(
            [_mark(horizon=2.0, move=10.0, lean="up", seq=5)],
            ledger_path=tmp_path / "absent.jsonl",
        )
        assert stray == [5]

    def test_no_marks_and_no_ledger_yields_no_orphans(self, tmp_path) -> None:
        assert orphans([], ledger_path=tmp_path / "absent.jsonl") == []
