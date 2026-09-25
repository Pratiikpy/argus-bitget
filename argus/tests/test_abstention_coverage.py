"""Tests for risk-coverage scoring of abstention (`eval/abstention_coverage.py`).

The fd-shifts reference values pinned here come from running their real, unmodified
``rc_stats.py``/``rc_stats_utils.py`` (commit c4467aec) on the same inputs — recorded in
`eval/general_abstention_comparison.py:RIVAL_RECORDED["fixtures"]`, which these tests read rather
than restate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from argus.eval.abstention_coverage import (
    MIN_ACTED,
    MIN_CALLS,
    CoverageError,
    Extraction,
    LeanCall,
    augrc,
    augrc_optimal,
    aurc,
    aurc_optimal,
    calls_from_entries,
    rc_points,
    score,
    working_point,
)
from argus.eval.general_abstention_comparison import RIVAL_RECORDED

FIXTURES = RIVAL_RECORDED["fixtures"]


def _call(i: int, *, conf: float, move: float, lean: str = "up", day: int = 0) -> LeanCall:
    at = datetime(2026, 9, 1, 12, tzinfo=UTC) + timedelta(days=day, minutes=i)
    return LeanCall(seq=i, decided_at=at.isoformat(), symbol="NVDAUSDT", lean=lean,
                    confidence=conf, move_bps=move, stated_confidence=0.8)


class TestLeanCall:
    def test_an_up_lean_nets_the_move_less_the_round_trip(self) -> None:
        assert _call(1, conf=0.6, move=50.0).net_bps == pytest.approx(38.0)

    def test_a_down_lean_nets_the_negated_move(self) -> None:
        assert _call(1, conf=0.6, move=-50.0, lean="down").net_bps == pytest.approx(38.0)

    def test_breaking_even_after_fees_is_an_error(self) -> None:
        """A trade that only pays its own fee was not worth taking."""
        assert _call(1, conf=0.6, move=12.0).error == 1
        assert _call(1, conf=0.6, move=12.5).error == 0


class TestParityWithFdShifts:
    @pytest.mark.parametrize("label", sorted(FIXTURES))
    def test_every_quantity_matches_their_real_output(self, label: str) -> None:
        fx = FIXTURES[label]
        conf, res = fx["confids"], fx["residuals"]
        pts = rc_points(conf, res)
        err = sum(res) / len(res)
        theirs = fx["fd_shifts"]
        assert aurc(pts) == pytest.approx(theirs["aurc"], abs=1e-12)
        assert augrc(pts) == pytest.approx(theirs["augrc"], abs=1e-12)
        assert aurc_optimal(err) == pytest.approx(theirs["aurc_optimal"], abs=1e-12)
        assert augrc_optimal(err) == pytest.approx(theirs["augrc_optimal"], abs=1e-12)
        assert aurc(pts) - aurc_optimal(err) == pytest.approx(theirs["eaurc"], abs=1e-12)
        assert len(pts) == theirs["n_rc_points"]

    @pytest.mark.parametrize("label", sorted(FIXTURES))
    def test_the_working_point_matches_theirs(self, label: str) -> None:
        fx = FIXTURES[label]
        wp = working_point(rc_points(fx["confids"], fx["residuals"]), target_risk=0.5)
        assert wp is not None
        assert wp.coverage == pytest.approx(fx["fd_shifts"]["working_point"]["coverage"])
        assert wp.threshold == pytest.approx(fx["fd_shifts"]["working_point"]["threshold"])


class TestTies:
    def test_tied_confidences_make_one_point(self) -> None:
        pts = rc_points([0.6] * 6, [1, 0, 1, 1, 0, 0])
        assert [p.coverage for p in pts] == [1.0, 0.0]

    def test_the_curve_does_not_depend_on_row_order(self) -> None:
        """torch-uncertainty's argsort splits ties by row order; this must not."""
        fx = FIXTURES["ties"]
        rows = list(zip(fx["confids"], fx["residuals"], strict=True))
        expected = aurc(rc_points(fx["confids"], fx["residuals"]))
        rng = random.Random(3)
        for _ in range(50):
            rng.shuffle(rows)
            got = aurc(rc_points([c for c, _ in rows], [r for _, r in rows]))
            assert got == pytest.approx(expected, abs=1e-15)

    def test_the_rival_really_is_order_dependent_on_the_same_rows(self) -> None:
        """Recorded from torch-uncertainty's own run — the reason its ranking was not taken."""
        tied = FIXTURES["all_tied"]
        assert tied["torch_uncertainty"]["aurc"] != pytest.approx(
            tied["torch_uncertainty_reversed_rows"]["aurc"], abs=1e-3)


class TestWorkingPoint:
    def test_none_rather_than_a_crash_when_nothing_meets_the_target(self) -> None:
        """fd-shifts raises ValueError from np.argmax on an empty mask here."""
        pts = rc_points([0.9, 0.8, 0.7, 0.6], [1, 1, 1, 0])
        assert working_point(pts, target_risk=0.2) is None


class TestValueCurve:
    def test_each_point_carries_the_net_of_what_it_acted_on(self) -> None:
        pts = rc_points([0.9, 0.5], [0, 1], [30.0, -100.0])
        assert pts[0].total_net_bps == pytest.approx(-70.0)
        assert pts[1].total_net_bps == pytest.approx(30.0)
        assert pts[-1].coverage == 0.0 and pts[-1].total_net_bps == 0.0

    def test_the_coverage_zero_point_serialises_without_an_infinite_threshold(self) -> None:
        assert rc_points([0.9], [0])[-1].as_dict()["threshold"] is None


@dataclass
class _Row:
    seq: int
    decided_at: str
    symbol: str
    lean: str
    lean_confidence: float
    stated_confidence: float
    counterfactual_move_bps: str | None
    settled_at: str | None
    is_abstention: bool = True


class TestExtraction:
    def _rows(self) -> list[_Row]:
        return [
            _Row(1, "2026-09-01T10:00:00+00:00", "A", "up", 0.6, 0.8, "40",
                 "2026-09-02T10:00:00+00:00"),
            _Row(2, "2026-09-01T12:00:00+00:00", "A", "none", 0.0, 0.8, "40",
                 "2026-09-02T12:00:00+00:00"),
            _Row(3, "2026-09-01T14:00:00+00:00", "A", "down", 0.7, 0.8, None, None),
            _Row(4, "2026-09-03T14:00:00+00:00", "A", "down", 0.7, 0.8, "-5",
                 "2026-09-04T14:00:00+00:00"),
        ]

    def test_a_refusal_without_a_lean_is_counted_not_scored(self) -> None:
        x = calls_from_entries(self._rows())
        assert [c.seq for c in x.calls] == [1, 4]
        assert x.no_lean == 1
        assert x.unsettled == 1

    def test_the_freeze_selects_by_sequence_and_settlement_time(self) -> None:
        x = calls_from_entries(self._rows(), upto_seq=3, settled_by="2026-09-03T00:00:00+00:00")
        assert [c.seq for c in x.calls] == [1]

    def test_an_override_scores_an_unsettled_row(self) -> None:
        x = calls_from_entries(self._rows(), counterfactuals={3: -90})
        assert [c.seq for c in x.calls] == [1, 3, 4]
        assert next(c for c in x.calls if c.seq == 3).net_bps == pytest.approx(78.0)


def _planted(n: int = 120, *, skilled: bool, seed: int = 1) -> Extraction:
    """Calls over six days whose confidence either ranks the outcome or ignores it."""
    rng = random.Random(seed)
    calls = []
    for i in range(n):
        move = rng.gauss(0.0, 80.0)
        net = move - 12.0
        conf = (0.5 + 0.4 * (1 if net > 0 else -1) * rng.random()) if skilled else rng.random()
        calls.append(_call(i, conf=round(conf, 3), move=move, day=i % 6))
    return Extraction(calls=tuple(calls), no_lean=0, unsettled=0)


class TestScore:
    def test_refuses_a_thin_record(self) -> None:
        with pytest.raises(CoverageError, match="below the floor"):
            score(Extraction(calls=tuple(_call(i, conf=0.6, move=1.0)
                                         for i in range(MIN_CALLS - 1)), no_lean=0, unsettled=0))

    def test_a_skilled_gate_is_detected(self) -> None:
        report = score(_planted(skilled=True), resamples=300)
        assert report["gate_skill"] > 0
        assert report["gate_skill_ci95_by_day"][0] > 0
        assert "ranks losing calls below winning ones" in report["gate_skill_verdict"]

    def test_a_random_gate_is_not_credited(self) -> None:
        report = score(_planted(skilled=False), resamples=300)
        lo, hi = report["gate_skill_ci95_by_day"]
        assert lo <= 0 <= hi

    def test_the_interval_reproduces_from_the_same_seed(self) -> None:
        x = _planted(skilled=True)
        assert score(x, resamples=200) == score(x, resamples=200)

    def test_one_lucky_call_is_not_named_the_best_threshold(self) -> None:
        """A cut-off that acts on one winning call is a coin flip, not a policy."""
        calls = [_call(0, conf=0.99, move=200.0)] + [
            _call(i, conf=0.5, move=-40.0, day=i % 3) for i in range(1, MIN_CALLS + 5)]
        report = score(Extraction(calls=tuple(calls), no_lean=0, unsettled=0), resamples=100)
        assert report["best_threshold_in_sample"]["acted"] == 0 or \
            report["best_threshold_in_sample"]["acted"] >= MIN_ACTED
        assert report["abstention_verdict"].startswith("abstaining on every call")

    def test_an_in_sample_threshold_is_tested_on_the_later_half(self) -> None:
        """Early winners at high confidence, later losers there: the pick fails out of sample."""
        early = [_call(i, conf=0.9 if i % 2 else 0.5, move=60.0 if i % 2 else -60.0, day=i % 3)
                 for i in range(40)]
        late = [_call(100 + i, conf=0.9, move=-60.0, day=3 + i % 3) for i in range(40)]
        report = score(Extraction(calls=tuple(early + late), no_lean=0, unsettled=0),
                       resamples=100)
        oos = report["out_of_sample"]
        assert oos["gradeable"] is True
        assert oos["chosen_threshold"] == pytest.approx(0.9)
        assert oos["held_out_total_net_bps"] < 0
        assert oos["held_out_beats_abstaining"] is False

    def test_the_abstention_doubt_ranking_reads_the_stated_confidence(self) -> None:
        x = _planted(skilled=False)
        a = score(x, ranking="lean_confidence", resamples=50)
        b = score(x, ranking="abstention_doubt", resamples=50)
        assert a["ranking"] == "lean_confidence" and b["ranking"] == "abstention_doubt"
        # every stated confidence is 0.8 in `_call`, so the doubt ranking is one tie
        assert b["aurc"] == pytest.approx(b["aurc_random"])

    def test_an_unknown_ranking_is_refused(self) -> None:
        with pytest.raises(CoverageError, match="unknown ranking"):
            score(_planted(skilled=False), ranking="vibes")
