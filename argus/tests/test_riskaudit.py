"""Risk-layer audit tests.

Track 2's judged half scores "risk control layer effectiveness", and the honest answer has two
failure modes that both look like success from the outside:

* a layer that **never fires** reads as a clean record, and is in fact untested;
* a layer that fires on **everything** reads as very safe, and means the model is not deciding —
  the positioning failure the whole architecture exists to avoid.

So the tests are mostly about refusing to call either of those a pass, and about the denominator:
a desk that abstained seventy-nine times gave its Constitution nothing to bind on, and dividing by
seventy-nine would report an idle layer where there was never an invocation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from argus.eval.riskaudit import (
    DOMINANT_RATE,
    MIN_DECISIONS_FOR_A_VERDICT,
    Exercise,
    audit,
    from_records,
    read_records,
)


def _row(
    seq: int = 1,
    *,
    intervened: bool = False,
    before: str = "100",
    after: str = "50",
    constraint: str = "unhedgeable_gap",
    responded: bool = True,
) -> dict[str, Any]:
    return {
        "seq": seq, "symbol": "NVDAUSDT", "at": "2026-09-13T12:00:00+00:00",
        "verdict": "trade", "intervened": intervened,
        "binding_constraint": constraint, "reason": "anchor shut",
        "quantity_before": before, "quantity_after": after if intervened else before,
        "model_changed_its_mind": responded,
    }


def _abstention(seq: int) -> dict[str, Any]:
    return {**_row(seq), "verdict": "no_trade", "quantity_before": "0", "quantity_after": "0"}


class TestNeverFiringIsNotAPass:
    def test_a_layer_that_never_bound_is_untested(self) -> None:
        report = from_records([_row(i) for i in range(1, 21)])
        assert report.exercise is Exercise.UNTESTED
        assert not report.exercise.is_healthy

    def test_it_says_untested_rather_than_safe(self) -> None:
        text = " ".join(from_records([_row(i) for i in range(1, 21)]).render())
        assert "This is untested, not safe" in text

    def test_too_few_decisions_is_also_untested_however_it_looks(self) -> None:
        """Two of three is not a 67% rate, it is three decisions."""
        rows = [_row(1, intervened=True), _row(2, intervened=True), _row(3)]
        assert len(rows) < MIN_DECISIONS_FOR_A_VERDICT
        assert from_records(rows).exercise is Exercise.UNTESTED

    def test_an_empty_record_is_untested(self) -> None:
        assert from_records([]).exercise is Exercise.UNTESTED


class TestAlwaysFiringIsAFinding:
    def test_binding_on_everything_is_dominant(self) -> None:
        rows = [_row(i, intervened=True) for i in range(1, 21)]
        assert from_records(rows).exercise is Exercise.DOMINANT

    def test_dominant_is_not_healthy(self) -> None:
        rows = [_row(i, intervened=True) for i in range(1, 21)]
        assert not from_records(rows).exercise.is_healthy

    def test_it_names_the_positioning_failure_rather_than_praising_the_rate(self) -> None:
        rows = [_row(i, intervened=True) for i in range(1, 21)]
        text = " ".join(from_records(rows).render())
        assert "the model is not the decision-maker" in text

    def test_the_boundary_is_the_named_constant(self) -> None:
        total = 20
        firing = int(DOMINANT_RATE * total)
        rows = [_row(i, intervened=i <= firing) for i in range(1, total + 1)]
        assert from_records(rows).exercise is Exercise.DOMINANT


class TestAWorkingLayer:
    ROWS: ClassVar[list[dict[str, Any]]] = [
        _row(i, intervened=i % 3 == 0) for i in range(1, 31)
    ]

    def test_a_middling_rate_is_active(self) -> None:
        assert from_records(self.ROWS).exercise is Exercise.ACTIVE

    def test_active_is_the_only_healthy_verdict(self) -> None:
        assert from_records(self.ROWS).exercise.is_healthy
        for state in (Exercise.UNTESTED, Exercise.LIGHT, Exercise.DOMINANT):
            assert not state.is_healthy

    def test_a_single_intervention_in_thirty_is_light(self) -> None:
        rows = [_row(i, intervened=i == 1) for i in range(1, 31)]
        assert from_records(rows).exercise is Exercise.LIGHT

    def test_the_constraints_are_counted_by_name(self) -> None:
        rows = [
            _row(1, intervened=True, constraint="unhedgeable_gap"),
            _row(2, intervened=True, constraint="unhedgeable_gap"),
            _row(3, intervened=True, constraint="drawdown_ladder"),
        ]
        assert from_records(rows).by_constraint() == {
            "unhedgeable_gap": 2, "drawdown_ladder": 1,
        }

    def test_an_unnamed_constraint_is_labelled_not_dropped(self) -> None:
        rows = [{**_row(1, intervened=True), "binding_constraint": None}]
        assert from_records(rows).by_constraint() == {"unnamed": 1}

    def test_whether_the_model_answered_back_is_counted(self) -> None:
        rows = [
            _row(1, intervened=True, responded=True),
            _row(2, intervened=True, responded=False),
        ]
        assert from_records(rows).responded == 1


class TestTheDenominatorIsTheDecisionsThatOfferedAPosition:
    def test_abstentions_do_not_dilute_the_rate(self) -> None:
        """A desk that stood aside gave the Constitution nothing to bind on."""
        rows = [_row(1, intervened=True)] + [_abstention(i) for i in range(2, 80)]
        report = from_records(rows)
        assert report.positions_offered == 1
        assert report.rate == 1.0

    def test_a_log_of_pure_abstentions_has_no_rate_at_all(self) -> None:
        report = from_records([_abstention(i) for i in range(1, 40)])
        assert report.positions_offered == 0
        assert report.rate is None
        assert report.exercise is Exercise.UNTESTED

    def test_the_decision_count_still_reports_every_row(self) -> None:
        rows = [_abstention(i) for i in range(1, 40)]
        assert from_records(rows).decisions == 39

    def test_a_voided_row_is_not_counted_as_a_position_offered(self) -> None:
        """**The live seq-264 shape.** Until 2026-09-20 `paper/runner.py` wrote `quantity_before`
        from the model's first draft rather than from the intent the Constitution actually ruled
        on, so the two voided rows store `quantity_before: "1"` beside `intervened: false` and
        `reason: "no exposure proposed"`. Counting them made this module report 2 positions offered
        while `eval/autopsy.py` reported 0 proposed exposure off the same record.

        The stored rows are deliberately not edited — see `paper/corrections.py` — so every reader
        carries the correction instead.
        """
        from argus.paper.corrections import VOIDED

        assert VOIDED, "this test is meaningless if the void register is empty"
        voided = [_row(v.seq, intervened=False, before="1", after="0") for v in VOIDED]
        report = from_records(voided + [_abstention(i) for i in range(900, 940)])
        assert report.positions_offered == 0
        assert report.rate is None
        assert report.exercise is Exercise.UNTESTED

    def test_a_voided_row_is_still_counted_as_a_decision(self) -> None:
        """Excluded from the numerator, never from the record. A void row happened; hiding it from
        the decision count would be the deletion this whole correction exists to refuse."""
        from argus.paper.corrections import VOIDED

        voided = [_row(v.seq, intervened=False, before="1", after="0") for v in VOIDED]
        assert from_records(voided).decisions == len(VOIDED)


class TestTheAsymmetryIsChecked:
    def test_increasing_exposure_is_caught(self) -> None:
        rows = [_row(1, intervened=True, before="50", after="100")]
        assert from_records(rows).asymmetry_violations

    def test_it_is_reported_first_because_it_is_a_broken_invariant(self) -> None:
        rows = [_row(1, intervened=True, before="50", after="100")]
        assert "ASYMMETRY VIOLATED" in from_records(rows).render()[0]

    def test_a_reduction_is_not_a_violation(self) -> None:
        rows = [_row(1, intervened=True, before="100", after="50")]
        report = from_records(rows)
        assert report.interventions[0].reduced
        assert not report.asymmetry_violations

    def test_an_unparseable_quantity_is_not_read_as_a_violation(self) -> None:
        rows = [_row(1, intervened=True, before="", after="oops")]
        assert not from_records(rows).asymmetry_violations


class TestWhatItRefusesToClaim:
    def test_correctness_of_an_intervention_is_undefined_without_outcomes(self) -> None:
        report = from_records([_row(i, intervened=i % 3 == 0) for i in range(1, 31)])
        payload = report.as_dict()
        assert payload["outcome_effect"] is None
        assert "undefined rather than zero" in payload["outcome_effect_note"]

    def test_the_note_changes_once_trades_exist(self) -> None:
        report = from_records([_row(1, intervened=True)], settled_trades=12)
        assert "12 settled trade(s)" in report.outcome_effect_note

    def test_the_report_never_prints_a_score(self) -> None:
        """There is a verdict and there are counts. A single number invites over-reading."""
        payload = from_records([_row(i, intervened=i % 3 == 0) for i in range(1, 31)]).as_dict()
        assert "score" not in payload


class TestReadingTheRecord:
    def test_a_missing_file_is_no_records_not_an_error(self, tmp_path: Path) -> None:
        assert read_records(tmp_path / "absent.jsonl") == []

    def test_a_missing_file_audits_as_untested(self, tmp_path: Path) -> None:
        assert audit(tmp_path / "absent.jsonl").exercise is Exercise.UNTESTED

    def test_a_corrupt_line_is_skipped_and_the_rest_survive(self, tmp_path: Path) -> None:
        path = tmp_path / "risk.jsonl"
        path.write_text(
            json.dumps(_row(1, intervened=True)) + "\nnot json\n" + json.dumps(_row(2)) + "\n",
            encoding="utf-8",
        )
        assert len(read_records(path)) == 2

    def test_blank_lines_are_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "risk.jsonl"
        path.write_text("\n\n" + json.dumps(_row(1)) + "\n\n", encoding="utf-8")
        assert len(read_records(path)) == 1

    def test_an_intervention_serialises_everything_needed_to_check_it(self) -> None:
        report = from_records([_row(1, intervened=True)])
        entry = report.interventions[0].as_dict()
        for key in ("seq", "symbol", "binding_constraint", "quantity_before", "quantity_after"):
            assert key in entry


@pytest.mark.parametrize("state", list(Exercise))
def test_every_exercise_state_renders_without_crashing(state: Exercise) -> None:
    assert str(state)
