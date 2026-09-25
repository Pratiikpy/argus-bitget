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

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from argus.eval import artefact
from argus.eval.refusal import (
    CONSISTENT,
    CONTRADICTED,
    MIN_FOR_A_RATE,
    ROUND_TRIP_BPS,
    UNGRADEABLE,
    Claim,
    HorizonResult,
    ReasonCheck,
    ReasonReport,
    binomial_upper,
    check_reasons,
    grade_reasons,
    orphans,
    reason_line,
    score,
    wilson,
)
from argus.paper.marks import Mark

AT = datetime(2026, 9, 15, 13, 30, tzinfo=UTC)


def _mark(*, horizon: float, move: float, lean: str, seq: int = 1) -> Mark:
    # Four symbols per decision cycle, cycles two hours apart, as the desk actually runs.
    decided = (AT + timedelta(hours=2 * (seq // 4))).isoformat()
    return Mark(
        seq=seq, symbol="NVDAUSDT", decided_at=decided, marked_at=decided,
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

    def test_leaning_with_the_tape_does_not_count_as_knowing_something(self) -> None:
        """38 of 40 right, every lean "up", in a tape that rose 38 of 40 times: it beats a coin
        and knows nothing the naive always-up call did not. The verdict must say the second."""
        marks = [
            _mark(horizon=2.0, move=50.0 if i < 38 else -50.0, lean="up", seq=i)
            for i in range(40)
        ]
        report = score(marks, now=AT)
        row = next(r for r in report.horizons if r.horizon == "about_2h")
        assert row.naive_accuracy == row.accuracy
        assert row.beats_the_naive_call is False
        assert "no evidence yet that the lean knows more" in report.verdict

    def test_a_lean_that_calls_both_directions_right_beats_the_naive_call(self) -> None:
        marks = [
            _mark(horizon=2.0, move=50.0 if i % 2 else -50.0, lean="up" if i % 2 else "down",
                  seq=i)
            for i in range(80)
        ]
        row = next(r for r in score(marks, now=AT).horizons if r.horizon == "about_2h")
        assert row.accuracy == 1.0
        assert row.beats_the_naive_call is True


class TestTheIntervalRespectsCycles:
    def test_calls_in_one_cycle_are_not_independent_evidence(self) -> None:
        """The same 60 calls read as 15 cycles give a wider interval than a per-call Wilson."""
        marks = [
            _mark(horizon=2.0, move=50.0 if (i // 4) % 3 else -50.0, lean="up", seq=i)
            for i in range(60)
        ]
        row = next(r for r in score(marks, now=AT).horizons if r.horizon == "about_2h")
        low, high = row.interval or (0.0, 0.0)
        wlow, whigh = wilson(row.correct, row.directional) or (0.0, 0.0)
        assert high - low > whigh - wlow
        assert row.as_dict()["interval_method"] == "bootstrap over decision cycles"

    def test_the_interval_is_reproducible(self) -> None:
        marks = [_mark(horizon=2.0, move=50.0 if i % 3 else -50.0, lean="up", seq=i)
                 for i in range(60)]
        first = next(r for r in score(marks, now=AT).horizons if r.horizon == "about_2h")
        second = next(r for r in score(marks, now=AT).horizons if r.horizon == "about_2h")
        assert first.interval == second.interval


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

    def test_a_mark_with_no_decision_behind_it_is_named(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        ledger.write_text('{"seq": 1}\n{"seq": 2}\n', encoding="utf-8")
        stray = orphans(
            [_mark(horizon=2.0, move=10.0, lean="up", seq=1),
             _mark(horizon=2.0, move=10.0, lean="up", seq=999)],
            ledger_path=ledger,
        )
        assert stray == [999]

    def test_marks_that_all_match_the_ledger_leave_no_orphans(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        ledger.write_text('{"seq": 1}\n{"seq": 2}\n', encoding="utf-8")
        assert orphans([_mark(horizon=2.0, move=10.0, lean="up", seq=2)], ledger_path=ledger) == []

    def test_a_missing_ledger_makes_every_mark_unverifiable_not_valid(self, tmp_path: Path) -> None:
        """Nothing to check against is not the same as everything checking out."""
        stray = orphans(
            [_mark(horizon=2.0, move=10.0, lean="up", seq=5)],
            ledger_path=tmp_path / "absent.jsonl",
        )
        assert stray == [5]

    def test_no_marks_and_no_ledger_yields_no_orphans(self, tmp_path: Path) -> None:
        assert orphans([], ledger_path=tmp_path / "absent.jsonl") == []


class TestTheStatedReasonIsCheckedAgainstTheRecord:
    """The fifth question: was the reason a refusal gave true? Each claim is graded only against a
    record that can settle it, and a claim the record cannot reach is ``ungradeable`` — never
    counted as confirmed, which would flatter the desk, nor as contradicted, which would slander
    it."""

    def _entry(self, thesis: str, *, phase: str = "weekend", hours: float = 50.0,
               seq: int = 7) -> dict[str, object]:
        return {"seq": seq, "symbol": "NVDAUSDT", "thesis": thesis, "session_phase": phase,
                "hours_to_discovery": hours, "verdict": "no_trade", "kind": "decision"}

    def _one(self, checks: list[ReasonCheck], claim: Claim) -> ReasonCheck:
        found = [c for c in checks if c.claim is claim]
        assert len(found) == 1, [c.claim for c in checks]
        return found[0]

    def test_a_closed_market_claim_on_a_weekend_is_consistent(self) -> None:
        checks = check_reasons(self._entry("Weekend session with anchor asleep."))
        assert self._one(checks, Claim.SESSION_CLOSED).verdict == CONSISTENT

    def test_a_closed_market_claim_during_regular_hours_is_contradicted(self) -> None:
        checks = check_reasons(self._entry("The anchor market is closed, so stand aside.",
                                           phase="rth", hours=0.0))
        assert self._one(checks, Claim.SESSION_CLOSED).verdict == CONTRADICTED

    def test_a_negated_open_claim_is_not_read_as_an_open_claim(self) -> None:
        """Seqs 139 and 460: "no anchor market open" on a weekend. The first run of this grader
        read both as open-market claims and accused the desk of the error it was looking for."""
        checks = check_reasons(self._entry("a weekend token with no anchor market open"))
        assert not [c for c in checks if c.claim is Claim.SESSION_OPEN]

    def test_a_past_reference_to_the_weekend_is_not_a_claim_about_now(self) -> None:
        checks = check_reasons(self._entry("the weekend memory shows 8 prior passes", phase="rth",
                                           hours=0.0))
        assert not [c for c in checks if c.claim is Claim.SESSION_CLOSED]

    def test_quoted_hours_within_rounding_agree_and_far_off_do_not(self) -> None:
        near = check_reasons(self._entry("52 hours to genuine price discovery", hours=51.99))
        far = check_reasons(self._entry("52 hours to genuine price discovery", hours=20.0))
        assert self._one(near, Claim.HOURS_TO_DISCOVERY).verdict == CONSISTENT
        assert self._one(far, Claim.HOURS_TO_DISCOVERY).verdict == CONTRADICTED

    def test_no_hedge_is_graded_only_where_the_desk_wrote_notes(self) -> None:
        thesis = self._entry("no hedge menu available")
        assert self._one(check_reasons(thesis), Claim.NO_HEDGE).verdict == UNGRADEABLE
        empty = check_reasons(thesis, notes=["hedge menu empty: nothing placeable for 50.0h"])
        assert self._one(empty, Claim.NO_HEDGE).verdict == CONSISTENT
        full = check_reasons(thesis, notes=["panel: 2 analysts"])
        assert self._one(full, Claim.NO_HEDGE).verdict == CONTRADICTED

    def test_blaming_the_risk_layer_when_it_did_not_intervene_is_contradicted(self) -> None:
        thesis = self._entry("The risk layer blocked the position.")
        idle = check_reasons(thesis, risk={"intervened": False,
                                           "binding_constraint": "no_exposure"})
        acted = check_reasons(thesis, risk={"intervened": True,
                                            "binding_constraint": "max_notional"})
        assert self._one(idle, Claim.RISK_LAYER).verdict == CONTRADICTED
        assert self._one(acted, Claim.RISK_LAYER).verdict == CONSISTENT

    def test_news_about_a_kill_switch_is_not_blaming_the_risk_layer(self) -> None:
        checks = check_reasons(self._entry("a Senate AI kill switch proposal weighs on sentiment"))
        assert not [c for c in checks if c.claim is Claim.RISK_LAYER]

    def test_a_forward_size_claim_is_graded_by_the_next_two_hour_mark(self) -> None:
        thesis = self._entry("the binding constraint was direction, not size")
        big = _mark(horizon=2.0, move=40.0, lean="none", seq=7)
        small = _mark(horizon=2.0, move=3.0, lean="none", seq=7)
        late = _mark(horizon=18.0, move=40.0, lean="none", seq=7)
        assert self._one(check_reasons(thesis, mark=big), Claim.MOVES_ENOUGH).verdict == \
            CONSISTENT
        assert self._one(check_reasons(thesis, mark=small), Claim.MOVES_ENOUGH).verdict == \
            CONTRADICTED
        assert self._one(check_reasons(thesis, mark=late), Claim.MOVES_ENOUGH).verdict == \
            UNGRADEABLE

    def test_an_edge_claim_is_not_mistaken_for_a_size_claim(self) -> None:
        checks = check_reasons(self._entry("the remaining edge is unlikely to clear the hurdle"))
        assert not [c for c in checks if c.claim is Claim.MOVE_TOO_SMALL]

    def test_the_report_from_files_and_its_artefact(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        rows = [self._entry("Weekend session with 50 hours to genuine price discovery.", seq=1),
                self._entry("The market is closed.", phase="rth", hours=0.0, seq=2),
                self._entry("No view.", seq=3),
                {"seq": 4, "verdict": "settlement_seal", "kind": "settlement_seal",
                 "thesis": "The market is closed."}]
        ledger.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        report = grade_reasons(ledger_path=ledger, notes_path=tmp_path / "none.jsonl",
                               risk_path=tmp_path / "none.jsonl", marks=[], now=AT)
        assert report.refusals == 3
        assert report.refusals_with_a_claim == 2
        assert [c.seq for c in report.contradicted] == [2]
        assert "1 contradicted it" in report.verdict
        out = tmp_path / "reasons.json"
        artefact.write(out, report.as_dict())
        assert artefact.is_strict(out)

    def test_no_refusals_is_undefined(self, tmp_path: Path) -> None:
        empty = tmp_path / "ledger.jsonl"
        empty.write_text("", encoding="utf-8")
        report = grade_reasons(ledger_path=empty, notes_path=empty, risk_path=empty, marks=[],
                               now=AT)
        assert "UNDEFINED" in report.verdict


class TestAForwardClaimMustBeatTheBaseRate:
    def test_the_exact_upper_tail(self) -> None:
        assert binomial_upper(5, 5, 0.5) == pytest.approx(1 / 32)
        assert binomial_upper(0, 7, 0.3) == pytest.approx(1.0)

    def test_a_claim_right_as_often_as_the_tape_carries_no_information(self) -> None:
        """Four in five right sounds like skill; it is not when four in five moves clear anyway."""
        entries = [{"seq": i, "symbol": "NVDAUSDT", "thesis": "direction, not size",
                    "session_phase": "rth", "hours_to_discovery": 0.0} for i in range(10)]
        marks = [_mark(horizon=2.0, move=40.0 if i < 8 else 2.0, lean="none", seq=i)
                 for i in range(10)]
        checks = [c for e, m in zip(entries, marks, strict=True)
                  for c in check_reasons(e, mark=m)]
        report = ReasonReport(checks=tuple(checks), refusals=10, base_clearance=(8, 10), as_of=AT)
        held, n, base, p = report.against_base(Claim.MOVES_ENOUGH) or (0, 0, 0.0, 0.0)
        assert (held, n, base) == (8, 10, 0.8)
        assert p > 0.05
        assert "no evidence the stated reason says more than the base rate" in report.verdict
        assert report.against_base(Claim.MOVE_TOO_SMALL) is None


class TestTheConsoleLineSaysWhatTheGradingSays:
    def _written(self, tmp_path: Path, *, contradicted: int, p: float) -> Path:
        entries = [{"seq": i, "symbol": "NVDAUSDT",
                    "thesis": "the anchor is asleep; direction, not size",
                    "session_phase": "rth" if i < contradicted else "weekend",
                    "hours_to_discovery": 0.0} for i in range(10)]
        marks = [_mark(horizon=2.0, move=40.0, lean="none", seq=i) for i in range(10)]
        checks = [c for e, m in zip(entries, marks, strict=True)
                  for c in check_reasons(e, mark=m)]
        report = ReasonReport(checks=tuple(checks), refusals=10, base_clearance=(8, 10),
                              as_of=AT).as_dict()
        report["forward_against_base"]["moves_enough"]["p_one_sided"] = p
        out = tmp_path / "reasons.json"
        artefact.write(out, report)
        return out

    def test_contradictions_and_an_uninformative_forward_claim_are_both_said(
            self, tmp_path: Path) -> None:
        line = reason_line(self._written(tmp_path, contradicted=2, p=0.26)) or ""
        assert "of 10 that the record can settle" in line
        assert "8 agreed and 2 contradicted it" in line
        assert "borne out 10 of 10 times (100%) against 80%" in line
        assert "says no more than the tape does" in line
        assert line.endswith("`python -m argus.eval.refusal`.")

    def test_an_informative_claim_is_called_informative(self, tmp_path: Path) -> None:
        line = reason_line(self._written(tmp_path, contradicted=0, p=0.01)) or ""
        assert "10 agreed and 0 contradicted" in line
        assert "carries information beyond the tape" in line

    def test_no_artefact_or_nothing_graded_is_no_line(self, tmp_path: Path) -> None:
        assert reason_line(tmp_path / "missing.json") is None
        empty = tmp_path / "empty.json"
        empty.write_text(json.dumps({"by_claim": {}}), encoding="utf-8")
        assert reason_line(empty) is None
