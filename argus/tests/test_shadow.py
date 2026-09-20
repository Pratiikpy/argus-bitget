"""Shadow-record tests — most of them exist to stop a number being reported.

The module's value is negative capability: it must stay UNDEFINED while the evidence is thin, and
it must never reach for the one field that would produce a figure immediately. The `side` field is
on all 161 rows and is right exactly as often as the base rate, so a fallback to it would turn an
honest "not yet measured" into a measurement of the tape. Several tests below do nothing but assert
that fallback is absent.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from argus.eval.shadow import (
    DEAD_ZONE_BPS,
    MIN_GRADED,
    Call,
    ShadowRecord,
    grade,
    move_of,
)
from argus.paper.ledger import Entry, PaperLedger

T0 = datetime(2026, 9, 13, 3, 0, tzinfo=UTC)


def _call(
    lean: str = "up", move: float = 100.0, seq: int = 1, source: str = "counterfactual",
) -> Call:
    return Call(seq=seq, symbol="rNVDA", lean=lean, confidence=0.6, move_bps=move, source=source)


def _record(
    led: PaperLedger, *, lean: str = "none", confidence: float = 0.0, n: int = 0, qty: str = "0",
) -> Entry:
    return led.record(
        symbol="rNVDA", verdict="no_trade" if qty == "0" else "trade", side="BUY",
        quantity=Decimal(qty), entry_price=Decimal("100"),
        stated_confidence=0.7, thesis="edge does not clear the hurdle",
        invalidation=("guidance reaffirmed",),
        market_state_hash="abc", approved_intent_hash="def",
        session_phase="weekend", hours_to_discovery=30.0,
        decided_at=T0 + timedelta(hours=n),
        lean=lean, lean_confidence=confidence,
    )


class TestItRefusesToReportTooEarly:
    def test_an_empty_record_says_undefined_and_why(self) -> None:
        got = ShadowRecord(calls=(), break_even=0.56)
        assert got.accuracy is None
        assert "UNDEFINED" in got.verdict
        assert "nothing to grade" in got.verdict
        assert "Unmeasured rather than zero" in got.verdict

    def test_one_call_below_the_floor_is_undefined(self) -> None:
        got = ShadowRecord(calls=(_call(),), break_even=0.56)
        assert got.accuracy is None
        assert got.base_rate is None
        assert got.clears_break_even is None

    def test_the_floor_is_exactly_the_stated_one(self) -> None:
        """One below is undefined, the floor itself reports. Off-by-one here silently changes what
        counts as measured."""
        below = ShadowRecord(
            calls=tuple(_call(seq=i) for i in range(MIN_GRADED - 1)), break_even=0.56
        )
        at = ShadowRecord(calls=tuple(_call(seq=i) for i in range(MIN_GRADED)), break_even=0.56)
        assert below.accuracy is None
        assert at.accuracy == pytest.approx(1.0)

    def test_the_undefined_verdict_names_the_two_reasons_a_call_was_not_scored(self) -> None:
        got = ShadowRecord(
            calls=(_call(lean="none"), _call(lean="up", move=1.0)), break_even=0.56
        )
        assert "1 decision(s) declined to lean" in got.verdict
        assert "1 met a tape flatter" in got.verdict


class TestTheSideFieldIsNeverConsulted:
    """The whole reason this module exists. `side` is BUY on every graded row and right 3.8% of the
    time against a 3.8% base rate — a fallback to it would report the tape as the desk."""

    def test_a_row_with_a_side_but_no_lean_scores_nothing(self, tmp_path: Path) -> None:
        led = PaperLedger(path=tmp_path / "p.jsonl")
        for i in range(30):
            entry = _record(led, n=i)
            led.settle_abstention(entry.seq, price_now=Decimal("102"))
        got = grade(led.entries, break_even=0.56)
        assert all(c.lean == "none" for c in got.calls)
        assert got.scored == ()
        assert got.accuracy is None

    def test_every_historical_row_reads_as_declined_rather_than_as_buy(
        self, tmp_path: Path
    ) -> None:
        led = PaperLedger(path=tmp_path / "p.jsonl")
        entry = _record(led)
        led.settle_abstention(entry.seq, price_now=Decimal("102"))
        assert grade(led.entries, break_even=0.56).declined_to_lean == 1


class TestScoringOneCall:
    def test_up_against_a_rise_is_correct(self) -> None:
        assert _call(lean="up", move=120.0).correct is True

    def test_up_against_a_fall_is_wrong(self) -> None:
        assert _call(lean="up", move=-120.0).correct is False

    def test_down_against_a_fall_is_correct(self) -> None:
        assert _call(lean="down", move=-120.0).correct is True

    def test_a_declined_lean_is_neither_right_nor_wrong(self) -> None:
        got = _call(lean="none", move=500.0)
        assert got.scored is False
        assert got.correct is None

    def test_a_flat_tape_is_not_a_direction(self) -> None:
        """Scoring a two-basis-point move as a directional win adds noise to both halves of the
        ratio. The dead zone is symmetric and exclusive at the boundary."""
        assert _call(move=DEAD_ZONE_BPS).scored is False
        assert _call(move=-DEAD_ZONE_BPS).scored is False
        assert _call(move=DEAD_ZONE_BPS + 0.01).scored is True


class TestTheRecord:
    def _mixed(self) -> ShadowRecord:
        # 12 right, 8 wrong out of 20 scored = 60%; 14 of the moves are up.
        calls = [_call(lean="up", move=100.0, seq=i) for i in range(12)]
        calls += [_call(lean="up", move=-100.0, seq=100 + i) for i in range(6)]
        calls += [_call(lean="down", move=100.0, seq=200 + i) for i in range(2)]
        return ShadowRecord(calls=tuple(calls), break_even=0.56)

    def test_accuracy_counts_only_scored_calls(self) -> None:
        assert self._mixed().accuracy == pytest.approx(0.6)

    def test_the_base_rate_is_reported_beside_it(self) -> None:
        assert self._mixed().base_rate == pytest.approx(14 / 20)

    def test_declined_and_flat_calls_stay_in_the_denominator_of_nothing(self) -> None:
        base = self._mixed()
        padded = ShadowRecord(
            calls=(*base.calls, _call(lean="none"), _call(move=0.5)), break_even=0.56
        )
        assert padded.accuracy == base.accuracy
        assert padded.declined_to_lean == 1
        assert padded.flat_tape == 1

    def test_clearing_the_break_even_is_reported_as_a_cost_of_abstaining(self) -> None:
        """90% right against a 70% base rate — high *and* separated from the tape. Both halves are
        load-bearing: an all-up fixture would score 100% and 100%, which is the base-rate case."""
        calls = [_call(lean="up", move=100.0, seq=i) for i in range(13)]
        calls += [_call(lean="down", move=-100.0, seq=100 + i) for i in range(5)]
        calls += [_call(lean="up", move=-100.0, seq=200), _call(lean="down", move=100.0, seq=201)]
        got = ShadowRecord(calls=tuple(calls), break_even=0.56)
        assert got.accuracy == pytest.approx(0.9)
        assert got.base_rate == pytest.approx(0.7)
        assert got.clears_break_even is True
        assert "abstentions are costing its owner money" in got.verdict

    def test_missing_the_break_even_vindicates_standing_aside(self) -> None:
        calls = [_call(lean="up", move=100.0, seq=i) for i in range(4)]
        calls += [_call(lean="up", move=-100.0, seq=100 + i) for i in range(6)]
        calls += [_call(lean="down", move=100.0, seq=200 + i) for i in range(6)]
        calls += [_call(lean="down", move=-100.0, seq=300 + i) for i in range(4)]
        got = ShadowRecord(calls=tuple(calls), break_even=0.56)
        assert got.accuracy == pytest.approx(0.4)
        assert got.base_rate == pytest.approx(0.5)
        assert got.clears_break_even is False
        assert "standing aside is the correct policy" in got.verdict

    def test_an_accuracy_equal_to_the_base_rate_is_called_what_it_is(self) -> None:
        """The finding that produced this module, reproduced as a test: a score matching the base
        rate is a score measuring the tape, and the verdict must say so rather than print a
        respectable-looking percentage."""
        calls = [_call(lean="up", move=100.0, seq=i) for i in range(18)]
        calls += [_call(lean="up", move=-100.0, seq=100 + i) for i in range(2)]
        got = ShadowRecord(calls=tuple(calls), break_even=0.56)
        assert got.accuracy == pytest.approx(got.base_rate)
        assert "indistinguishable from always calling the more common direction" in got.verdict

    def test_the_report_is_json_serialisable(self) -> None:
        blob = json.dumps(self._mixed().as_dict())
        assert json.loads(blob)["accuracy"] == pytest.approx(0.6)

    def test_the_rendered_report_shows_the_three_numbers_together(self) -> None:
        """An accuracy without its base rate and break-even beside it is not interpretable."""
        text = self._mixed().render()
        assert "accuracy" in text and "base rate" in text and "break-even" in text


class TestReadingTheLedger:
    def test_an_unsettled_decision_has_no_outcome_to_grade(self, tmp_path: Path) -> None:
        led = PaperLedger(path=tmp_path / "p.jsonl")
        _record(led, lean="up", confidence=0.7)
        assert grade(led.entries, break_even=0.56).calls == ()

    def test_a_settled_abstention_grades_against_its_counterfactual(self, tmp_path: Path) -> None:
        led = PaperLedger(path=tmp_path / "p.jsonl")
        entry = _record(led, lean="up", confidence=0.7)
        led.settle_abstention(entry.seq, price_now=Decimal("102"))
        got = grade(led.entries, break_even=0.56)
        assert len(got.calls) == 1
        assert got.calls[0].source == "counterfactual"
        assert got.calls[0].move_bps == pytest.approx(200.0)
        assert got.calls[0].correct is True

    def test_a_settled_trade_grades_against_its_fill(self, tmp_path: Path) -> None:
        """Only abstentions carry a counterfactual. Grading those alone would make this a record of
        refusals wearing the name of the desk."""
        led = PaperLedger(path=tmp_path / "p.jsonl")
        entry = _record(led, lean="down", confidence=0.5, qty="10")
        led.settle(entry.seq, exit_price=Decimal("98"))
        got = grade(led.entries, break_even=0.56)
        assert got.calls[0].source == "fill"
        assert got.calls[0].move_bps == pytest.approx(-200.0)
        assert got.calls[0].correct is True

    def test_the_fill_move_is_the_same_arithmetic_the_ledger_writes(self, tmp_path: Path) -> None:
        """If these two ever diverge, the record is grading trades on a different scale from
        abstentions and the accuracy mixes two units."""
        led = PaperLedger(path=tmp_path / "p.jsonl")
        abstained = _record(led, lean="up", n=0)
        traded = _record(led, lean="up", n=1, qty="10")
        led.settle_abstention(abstained.seq, price_now=Decimal("103.5"))
        led.settle(traded.seq, exit_price=Decimal("103.5"))
        rows = grade(led.entries, break_even=0.56).calls
        assert rows[0].move_bps == pytest.approx(rows[1].move_bps)

    def test_the_source_mix_is_reported(self, tmp_path: Path) -> None:
        led = PaperLedger(path=tmp_path / "p.jsonl")
        for i in range(21):
            entry = _record(led, lean="up", n=i, qty="0" if i else "10")
            if i:
                led.settle_abstention(entry.seq, price_now=Decimal("102"))
            else:
                led.settle(entry.seq, exit_price=Decimal("102"))
        got = grade(led.entries, break_even=0.56)
        assert got.by_source == {"fill": 1, "counterfactual": 20}

    def test_a_zero_entry_price_is_skipped_rather_than_dividing(self, tmp_path: Path) -> None:
        assert move_of(_Row(entry_price="0", exit_price="100")) is None

    def test_a_malformed_move_is_skipped_rather_than_crashing(self) -> None:
        assert move_of(_Row(counterfactual_move_bps="not a number")) is None

    def test_a_row_with_neither_shape_is_skipped(self) -> None:
        assert move_of(_Row()) is None


class TestTheLeanIsNormalised:
    def test_case_and_whitespace_do_not_create_a_third_answer(self, tmp_path: Path) -> None:
        led = PaperLedger(path=tmp_path / "p.jsonl")
        entry = _record(led, lean="  UP  ")
        led.settle_abstention(entry.seq, price_now=Decimal("102"))
        assert grade(led.entries, break_even=0.56).calls[0].lean == "up"

    def test_an_unknown_lean_is_not_scored_as_a_direction(self) -> None:
        assert _call(lean="sideways", move=100.0).scored is False


class _Row:
    """A minimal stand-in for a ledger row, for the shapes a real ledger will not produce."""

    def __init__(self, **fields: object) -> None:
        self.__dict__.update(fields)


class TestTheLiveLedger:
    """Run against the real log, not a fixture. This is the claim the master plan will carry."""

    def test_every_recorded_decision_loads_and_grades(self) -> None:
        from argus.paper.runner import LEDGER_PATH

        if not LEDGER_PATH.exists():
            pytest.skip("no live ledger on this machine")
        led = PaperLedger(path=LEDGER_PATH)
        got = grade(led.entries, break_even=0.56)
        assert len(led.entries) > 0
        assert len(got.calls) <= len(led.entries)
        # Whatever the state of the log, the record must be internally consistent.
        assert len(got.scored) + got.declined_to_lean + got.flat_tape <= len(got.calls)
        if got.accuracy is None:
            assert "UNDEFINED" in got.verdict
