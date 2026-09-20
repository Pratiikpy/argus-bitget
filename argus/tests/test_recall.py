"""Episodic memory: point-in-time, graded-only lessons, and no pattern from too little."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from argus.agents.recall import (
    MIN_EPISODES_FOR_A_LESSON,
    RECALL_LIMIT,
    recall,
)

NOW = datetime(2026, 9, 14, 15, 0, tzinfo=UTC)
HURDLE = Decimal("18.8")


@dataclass
class Row:
    """The shape `argus.paper.ledger.Entry` presents, built without touching disk."""

    seq: int
    decided_at: str
    symbol: str = "NVDAUSDT"
    verdict: str = "no_trade"
    session_phase: str = "weekend"
    stated_confidence: float = 0.8
    thesis: str = "no edge above the hurdle"
    settled_at: str | None = None
    net_pnl: str | None = None
    direction_correct: bool | None = None
    counterfactual_move_bps: str | None = None


def _row(
    seq: int, *, days_ago: float = 1.0, settled_days_ago: float | None = None, **kw: object
) -> Row:
    decided = NOW - timedelta(days=days_ago)
    settled = (
        (NOW - timedelta(days=settled_days_ago)).isoformat()
        if settled_days_ago is not None else None
    )
    return Row(seq=seq, decided_at=decided.isoformat(), settled_at=settled, **kw)  # type: ignore[arg-type]


class TestPointInTime:
    def test_a_decision_cannot_remember_itself(self) -> None:
        got = recall([_row(1, days_ago=0.0)], symbol="NVDAUSDT", now=NOW)
        assert got.considered == 0

    def test_a_future_decision_is_invisible(self) -> None:
        got = recall([_row(1, days_ago=-3.0)], symbol="NVDAUSDT", now=NOW)
        assert not got.episodes

    def test_an_episode_settled_after_now_contributes_no_outcome(self) -> None:
        """The decisive test. Decided last week, settles tomorrow: the decision is visible and
        the result is not. Reading the result here is the look-ahead that makes every 'learning'
        agent look like it learned."""
        rows = [_row(1, days_ago=7.0, settled_days_ago=-1.0,
                     counterfactual_move_bps="-400.0")]
        got = recall(rows, symbol="NVDAUSDT", now=NOW)
        assert len(got.episodes) == 1
        assert not got.episodes[0].graded
        assert got.episodes[0].counterfactual_move_bps is None

    def test_an_episode_settled_before_now_does_contribute(self) -> None:
        rows = [_row(1, days_ago=7.0, settled_days_ago=6.0,
                     counterfactual_move_bps="-400.0")]
        got = recall(rows, symbol="NVDAUSDT", now=NOW)
        assert got.episodes[0].graded
        assert got.episodes[0].counterfactual_move_bps == Decimal("-400.0")

    def test_moving_now_backwards_can_only_remove_knowledge(self) -> None:
        rows = [_row(i, days_ago=float(i), settled_days_ago=float(i) - 0.5,
                     counterfactual_move_bps="-50") for i in range(1, 9)]
        late = recall(rows, symbol="NVDAUSDT", now=NOW)
        early = recall(rows, symbol="NVDAUSDT", now=NOW - timedelta(days=4))
        assert len(early.graded) < len(late.graded)

    def test_another_symbol_is_never_recalled(self) -> None:
        rows = [_row(1, days_ago=2.0, symbol="TSLAUSDT")]
        assert not recall(rows, symbol="NVDAUSDT", now=NOW).episodes


class TestOnlyGradedEpisodesBecomeLessons:
    def _graded(self, n: int, move: str = "-5.0") -> list[Row]:
        return [
            _row(i, days_ago=float(i + 1), settled_days_ago=float(i) + 0.5,
                 counterfactual_move_bps=move)
            for i in range(1, n + 1)
        ]

    def test_no_lesson_below_the_floor(self) -> None:
        got = recall(self._graded(MIN_EPISODES_FOR_A_LESSON - 1), symbol="NVDAUSDT", now=NOW)
        assert got.lessons(hurdle_bps=HURDLE) == ()
        assert not got.has_enough_to_generalise

    def test_a_lesson_at_the_floor(self) -> None:
        got = recall(self._graded(MIN_EPISODES_FOR_A_LESSON), symbol="NVDAUSDT", now=NOW)
        assert got.lessons(hurdle_bps=HURDLE)

    def test_the_floor_matches_the_calibration_floor(self) -> None:
        assert MIN_EPISODES_FOR_A_LESSON == 5

    def test_ungraded_episodes_never_count_toward_the_floor(self) -> None:
        rows = [_row(i, days_ago=float(i + 1)) for i in range(1, 20)]
        got = recall(rows, symbol="NVDAUSDT", now=NOW)
        assert got.episodes and not got.has_enough_to_generalise

    def test_the_report_says_why_no_pattern_is_stated(self) -> None:
        text = recall(self._graded(2), symbol="NVDAUSDT", now=NOW).render(hurdle_bps=HURDLE)
        assert "no pattern is stated" in text and "not a lesson" in text


class TestTheArithmeticIsCheckable:
    def _passes(self, moves: list[str]) -> object:
        rows = [
            _row(i, days_ago=float(i + 1), settled_days_ago=float(i) + 0.5,
                 counterfactual_move_bps=m)
            for i, m in enumerate(moves, start=1)
        ]
        return recall(rows, symbol="NVDAUSDT", now=NOW, limit=50)

    def test_the_median_passed_move_uses_absolute_size(self) -> None:
        """A pass that avoided -200bps and one that missed +200bps are both evidence the tape
        moved. Signing them would cancel the two to zero."""
        got = self._passes(["-200", "200", "-200", "200", "-200"])
        assert got.median_passed_move_bps() == Decimal("200")  # type: ignore[attr-defined]

    def test_moves_above_the_hurdle_are_counted(self) -> None:
        got = self._passes(["-5", "-5", "-50", "-50", "-50"])
        assert got.abstentions_that_beat(HURDLE) == 3  # type: ignore[attr-defined]

    def test_a_quiet_record_says_the_tape_did_not_pay(self) -> None:
        text = " ".join(
            self._passes(["-1", "-2", "-3", "-4", "-5"]).lessons(hurdle_bps=HURDLE)  # type: ignore[attr-defined]
        )
        assert "did not pay a round trip" in text

    def test_a_moving_record_says_the_problem_was_direction(self) -> None:
        """The distinction the desk could not previously make: a quiet tape and a tape it could
        not call are different failures."""
        text = " ".join(
            self._passes(["-90", "-90", "-90", "-90", "-1"]).lessons(hurdle_bps=HURDLE)  # type: ignore[attr-defined]
        )
        assert "direction, not size" in text

    def test_traded_outcomes_are_counted_separately(self) -> None:
        rows = [
            _row(i, days_ago=float(i + 1), settled_days_ago=float(i) + 0.5,
                 verdict="buy", net_pnl="1.0", direction_correct=(i % 2 == 0))
            for i in range(1, 7)
        ]
        text = " ".join(recall(rows, symbol="NVDAUSDT", now=NOW).lessons(hurdle_bps=HURDLE))
        assert "graded position(s) taken" in text

    def test_a_single_phase_record_says_it_does_not_generalise(self) -> None:
        text = " ".join(
            self._passes(["-5"] * 6).lessons(hurdle_bps=HURDLE)  # type: ignore[attr-defined]
        )
        assert "nothing here generalises to another session phase" in text

    def test_no_lesson_is_written_by_a_model(self) -> None:
        """The asymmetry against TradingAgents' reflection log: every sentence here is arithmetic
        over the ledger, so the module needs no model and cannot import one."""
        from pathlib import Path

        import argus.agents.recall as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")
        for forbidden in ("QwenClient", "complete_json", "from argus.llm"):
            assert forbidden not in source


class TestSelectionIsBoundedAndOrdered:
    def test_the_limit_is_respected(self) -> None:
        rows = [_row(i, days_ago=float(i + 1)) for i in range(1, 40)]
        got = recall(rows, symbol="NVDAUSDT", now=NOW)
        assert len(got.episodes) == RECALL_LIMIT and got.considered == 39

    def test_graded_episodes_are_preferred_over_recent_ungraded_ones(self) -> None:
        rows = [_row(i, days_ago=float(i)) for i in range(1, 9)]
        rows.append(_row(99, days_ago=40.0, settled_days_ago=39.0,
                         counterfactual_move_bps="-12"))
        got = recall(rows, symbol="NVDAUSDT", now=NOW, limit=3)
        assert 99 in [e.seq for e in got.episodes]

    def test_the_same_phase_is_preferred_but_never_filtering(self) -> None:
        """A preference that hides an episode is a blind spot, which is the rule the mandate
        applies to evidence and the same one applies here."""
        rows = [_row(1, days_ago=1.0, session_phase="rth"),
                _row(2, days_ago=2.0, session_phase="weekend")]
        got = recall(rows, symbol="NVDAUSDT", now=NOW, session_phase="weekend")
        assert {e.seq for e in got.episodes} == {1, 2}

    def test_episodes_are_returned_newest_first(self) -> None:
        rows = [_row(i, days_ago=float(i)) for i in range(1, 6)]
        got = recall(rows, symbol="NVDAUSDT", now=NOW)
        stamps = [e.decided_at for e in got.episodes]
        assert stamps == sorted(stamps, reverse=True)

    def test_the_count_considered_is_reported_beside_the_count_returned(self) -> None:
        rows = [_row(i, days_ago=float(i + 1)) for i in range(1, 30)]
        text = recall(rows, symbol="NVDAUSDT", now=NOW).render(hurdle_bps=HURDLE)
        assert "of 29 on record" in text


class TestTheReport:
    def test_an_empty_memory_says_so_plainly(self) -> None:
        text = recall([], symbol="NVDAUSDT", now=NOW).render(hurdle_bps=HURDLE)
        assert "no prior decision" in text and "nothing below is inherited" in text

    def test_it_warns_that_memory_is_not_permission(self) -> None:
        rows = [_row(1, days_ago=1.0)]
        text = recall(rows, symbol="NVDAUSDT", now=NOW).render(hurdle_bps=HURDLE)
        assert "context, never permission" in text

    def test_an_ungraded_episode_is_labelled(self) -> None:
        rows = [_row(1, days_ago=1.0)]
        assert "not yet settled" in recall(rows, symbol="NVDAUSDT", now=NOW).render(
            hurdle_bps=HURDLE
        )

    def test_a_graded_abstention_reports_the_move_it_passed(self) -> None:
        rows = [_row(1, days_ago=2.0, settled_days_ago=1.0,
                     counterfactual_move_bps="-184.6")]
        text = recall(rows, symbol="NVDAUSDT", now=NOW).render(hurdle_bps=HURDLE)
        assert "-184.6bps" in text

    def test_it_serialises(self) -> None:
        rows = [_row(1, days_ago=2.0, settled_days_ago=1.0, counterfactual_move_bps="-5")]
        got = recall(rows, symbol="NVDAUSDT", now=NOW).as_dict()
        assert got["graded"] == 1 and got["considered"] == 1

    def test_a_malformed_timestamp_is_skipped_rather_than_guessed(self) -> None:
        bad = Row(seq=1, decided_at="not-a-date")
        assert not recall([bad], symbol="NVDAUSDT", now=NOW).episodes

    def test_a_malformed_number_is_absent_rather_than_zero(self) -> None:
        """A zero move would read as a correct abstention on a dead tape."""
        rows = [_row(1, days_ago=2.0, settled_days_ago=1.0,
                     counterfactual_move_bps="not-a-number")]
        got = recall(rows, symbol="NVDAUSDT", now=NOW)
        assert got.episodes[0].counterfactual_move_bps is None
