"""Hurdle tests — the verdict has to be able to come out against us, so both directions are tested.

This module answers "was abstaining right?" and the answer is a sentence a reader will quote. The
risk is therefore not that the arithmetic is wrong but that the verdict is written so it can only
say one thing. Every test below that touches the verdict is paired: one case where the hurdle binds
and abstention is vindicated, one where it does not and the desk's stated reason is wrong.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from argus.eval.hurdle import (
    HURDLE_SWEEP_BPS,
    MIN_INSTANTS,
    Frontier,
    HurdleError,
    Instant,
    break_even_accuracy,
    build,
    default_hurdle_bps,
    frontier_point,
    load_instants,
)

START = datetime(2026, 9, 1, 14, 0, tzinfo=UTC)


def _instants(moves: list[float], *, symbols: int = 1, source: str = "replay") -> list[Instant]:
    """One instant per move, spread over distinct hours unless several symbols are asked for."""
    out = []
    for index, move in enumerate(moves):
        out.append(
            Instant(
                symbol=f"SYM{index % symbols}",
                at=START + timedelta(hours=index // symbols),
                move_bps=move,
                source=source,
            )
        )
    return out


class TestBreakEvenAccuracy:
    def test_a_free_trade_breaks_even_at_a_coin_flip(self) -> None:
        """With no hurdle, a directional bet is fair at 50%. Anything else means the formula has
        picked up a cost that is not there."""
        assert break_even_accuracy(100.0, 0.0) == pytest.approx(0.5)

    def test_the_worked_case(self) -> None:
        """``p = (|r| + h) / (2|r|)``. At a 100bps move and a 19bps hurdle, 59.5%."""
        assert break_even_accuracy(100.0, 19.0) == pytest.approx(0.595)

    def test_a_larger_hurdle_demands_more_accuracy(self) -> None:
        values = [break_even_accuracy(150.0, h) for h in (0.0, 10.0, 30.0, 60.0)]
        assert all(a is not None for a in values)
        assert values == sorted(values)  # type: ignore[type-var]

    def test_a_larger_move_demands_less(self) -> None:
        small = break_even_accuracy(50.0, 20.0)
        large = break_even_accuracy(500.0, 20.0)
        assert small is not None and large is not None and large < small

    def test_a_hurdle_at_the_move_size_admits_no_accuracy(self) -> None:
        """At ``h = |r|`` the required accuracy is exactly 1.0. Reporting 100% would imply a
        perfect trader breaks even; reporting 1.4 for a larger hurdle would imply a 140% hit rate
        is a thing that could be attempted. Neither is a number, so neither is returned."""
        assert break_even_accuracy(100.0, 100.0) is None
        assert break_even_accuracy(100.0, 250.0) is None

    def test_a_zero_move_admits_no_accuracy(self) -> None:
        assert break_even_accuracy(0.0, 12.0) is None


class TestOneFrontierPoint:
    def test_everything_clears_a_zero_hurdle(self) -> None:
        point = frontier_point([100.0, -50.0, 20.0], 0.0)
        assert point.clearing_rate == 1.0

    def test_the_clearing_rate_counts_magnitudes_not_signs(self) -> None:
        """A 200bps fall is as tradeable as a 200bps rise. Counting only rises would halve the
        measured opportunity and make abstention look correct for the wrong reason."""
        assert frontier_point([-200.0, -200.0], 50.0).clearing_rate == 1.0

    def test_the_oracle_declines_rather_than_taking_a_loss(self) -> None:
        """Perfect foresight includes foreseeing that a move cannot pay the hurdle. Charging the
        oracle for trades it would never take would understate the ceiling."""
        point = frontier_point([5.0, 5.0, 200.0], 50.0)
        assert point.oracle_net_bps == pytest.approx(150.0 / 3)

    def test_the_oracle_earns_nothing_when_nothing_clears(self) -> None:
        assert frontier_point([10.0, -20.0], 500.0).oracle_net_bps == 0.0

    def test_a_coin_flipper_simply_pays_the_hurdle(self) -> None:
        """The honest null, and it is negative at every hurdle above zero."""
        assert frontier_point([100.0, -100.0], 19.0).coin_flip_net_bps == -19.0

    def test_the_oracle_total_is_the_sum_and_the_net_is_the_mean(self) -> None:
        point = frontier_point([100.0, 300.0], 0.0)
        assert point.oracle_total_bps == pytest.approx(400.0)
        assert point.oracle_net_bps == pytest.approx(200.0)

    def test_an_empty_record_raises(self) -> None:
        with pytest.raises(HurdleError, match="not a frontier"):
            frontier_point([], 12.0)

    def test_a_negative_hurdle_raises(self) -> None:
        with pytest.raises(HurdleError, match="subsidy"):
            frontier_point([100.0], -1.0)


class TestTheVerdictGoesBothWays:
    def test_small_moves_vindicate_the_hurdle(self) -> None:
        """A universe where the median move is barely above the fee. Here the desk's stated reason
        for abstaining is the true one."""
        moves = [20.0, -22.0, 18.0, 25.0, -19.0, 21.0, -24.0, 23.0, 17.0, 26.0, -20.0, 22.0]
        frontier = build(_instants(moves), hurdle_bps=19.0)
        assert "THE HURDLE BINDS" in frontier.binding_constraint

    def test_large_moves_refute_it(self) -> None:
        """Our actual universe. The moves dwarf the hurdle, so blaming the fee is wrong whatever
        the merits of abstaining."""
        moves = [150.0, -200.0, 180.0, -130.0, 220.0, 160.0, -190.0, 140.0,
                 210.0, -170.0, 155.0, 195.0]
        frontier = build(_instants(moves), hurdle_bps=19.0)
        assert "THE CONFIDENCE GATE BINDS" in frontier.binding_constraint
        assert "wrong" in frontier.binding_constraint

    def test_the_boundary_is_stated_in_the_verdict_not_hidden(self) -> None:
        """Whichever side it lands on, the ratio that decided it appears in the sentence."""
        for moves in ([20.0] * 12, [200.0] * 12):
            text = build(_instants(moves), hurdle_bps=19.0).binding_constraint
            assert "19bps hurdle" in text

    def test_too_few_instants_names_no_constraint_at_all(self) -> None:
        frontier = build(_instants([150.0] * (MIN_INSTANTS - 1)), hurdle_bps=19.0)
        assert "INSUFFICIENT" in frontier.binding_constraint

    def test_the_threshold_is_the_stated_one(self) -> None:
        assert "INSUFFICIENT" not in build(
            _instants([150.0] * MIN_INSTANTS), hurdle_bps=19.0
        ).binding_constraint


class TestClusteringIsNotIgnored:
    def test_instants_sharing_an_hour_are_one_cluster(self) -> None:
        frontier = build(_instants([100.0] * 12, symbols=4), hurdle_bps=19.0)
        assert len(frontier.clusters) == 3

    def test_correlated_symbols_reduce_the_effective_count(self) -> None:
        """Four symbols moving together on the same hour are not four independent facts. If the
        effective count ever equals the raw count here, the correction is not wired in."""
        moves = []
        for hour in range(10):
            base = 100.0 + hour * 40.0
            moves.extend([base, base + 1.0, base - 1.0, base + 2.0])
        frontier = build(_instants(moves, symbols=4), hurdle_bps=19.0)
        assert frontier.effective_count < len(frontier.instants)
        assert frontier.icc > 0.5

    def test_independent_instants_keep_their_full_count(self) -> None:
        frontier = build(_instants([100.0, 300.0, 50.0, 200.0] * 5), hurdle_bps=19.0)
        assert frontier.effective_count == len(frontier.instants)

    def test_the_effective_count_is_never_below_one(self) -> None:
        frontier = build(_instants([100.0] * 12, symbols=12), hurdle_bps=19.0)
        assert frontier.effective_count >= 1


class TestTheFrontierItself:
    def test_the_actual_hurdle_is_always_on_it(self) -> None:
        frontier = build(_instants([100.0] * 12), hurdle_bps=17.3)
        assert any(p.hurdle_bps == pytest.approx(17.3) for p in frontier.points)

    def test_a_sweep_level_next_to_the_actual_hurdle_is_not_printed_twice(self) -> None:
        """19.0 is in the sweep and the real hurdle is 18.8. Two rows rounding to the same number
        read as two different answers to one question."""
        frontier = build(_instants([100.0] * 12), hurdle_bps=19.02)
        levels = [p.hurdle_bps for p in frontier.points]
        assert len(levels) == len(set(levels))
        assert sum(1 for level in levels if abs(level - 19.0) < 0.1) == 1

    def test_the_points_are_ordered_by_hurdle(self) -> None:
        frontier = build(_instants([100.0] * 12), hurdle_bps=19.0)
        levels = [p.hurdle_bps for p in frontier.points]
        assert levels == sorted(levels)

    def test_the_clearing_rate_falls_as_the_hurdle_rises(self) -> None:
        frontier = build(_instants([50.0, 120.0, 300.0, 80.0] * 4), hurdle_bps=19.0)
        rates = [p.clearing_rate for p in frontier.points]
        assert rates == sorted(rates, reverse=True)

    def test_the_oracle_ceiling_falls_as_the_hurdle_rises(self) -> None:
        frontier = build(_instants([50.0, 120.0, 300.0, 80.0] * 4), hurdle_bps=19.0)
        ceilings = [p.oracle_net_bps for p in frontier.points]
        assert ceilings == sorted(ceilings, reverse=True)

    def test_the_sweep_reaches_past_the_largest_move(self) -> None:
        """So the frontier is shown all the way to the point where nothing clears, rather than
        stopping where the picture is still flattering."""
        assert max(HURDLE_SWEEP_BPS) >= 400.0

    def test_building_from_nothing_raises(self) -> None:
        with pytest.raises(HurdleError, match="no instants"):
            build([])

    def test_the_default_hurdle_is_the_fee_plus_deliberation(self) -> None:
        assert default_hurdle_bps(deliberation_bps=0.0) == pytest.approx(12.0)
        assert default_hurdle_bps(deliberation_bps=6.8) == pytest.approx(18.8)


class TestLoadingFromBothLedgers:
    def _write(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    def test_it_reads_the_replay_and_the_live_ledger_and_labels_each(self, tmp_path: Path) -> None:
        replay = tmp_path / "replay.jsonl"
        live = tmp_path / "live.jsonl"
        self._write(replay, [
            {"symbol": "NVDAUSDT", "at": START.isoformat(), "realised_bps": 120.0},
        ])
        self._write(live, [
            {"symbol": "TSLAUSDT", "decided_at": START.isoformat(),
             "counterfactual_move_bps": -80.0},
        ])
        got = load_instants(replay_path=replay, live_path=live)
        assert {i.source for i in got} == {"replay", "live"}
        assert {i.move_bps for i in got} == {120.0, -80.0}

    def test_a_row_with_no_recorded_move_is_skipped_not_zeroed(self, tmp_path: Path) -> None:
        """Most live rows have no counterfactual. Treating a missing move as 0bps would drag the
        median toward zero and manufacture the conclusion that the hurdle binds."""
        live = tmp_path / "live.jsonl"
        self._write(live, [
            {"symbol": "A", "decided_at": START.isoformat(), "counterfactual_move_bps": None},
            {"symbol": "B", "decided_at": START.isoformat(), "counterfactual_move_bps": 90.0},
        ])
        got = load_instants(replay_path=tmp_path / "missing.jsonl", live_path=live)
        assert [i.move_bps for i in got] == [90.0]

    def test_a_row_with_no_timestamp_is_skipped(self, tmp_path: Path) -> None:
        live = tmp_path / "live.jsonl"
        self._write(live, [{"symbol": "A", "counterfactual_move_bps": 90.0}])
        assert load_instants(replay_path=tmp_path / "none.jsonl", live_path=live) == []

    def test_a_missing_file_loads_as_empty(self, tmp_path: Path) -> None:
        assert load_instants(
            replay_path=tmp_path / "a.jsonl", live_path=tmp_path / "b.jsonl"
        ) == []

    def test_blank_lines_are_tolerated(self, tmp_path: Path) -> None:
        replay = tmp_path / "replay.jsonl"
        replay.write_text(
            json.dumps({"symbol": "A", "at": START.isoformat(), "realised_bps": 50.0}) + "\n\n",
            encoding="utf-8",
        )
        assert len(load_instants(replay_path=replay, live_path=tmp_path / "x.jsonl")) == 1


class TestTheReport:
    def _frontier(self) -> Frontier:
        return build(_instants([150.0, -200.0, 180.0] * 4), hurdle_bps=18.8)

    def test_the_rendered_report_carries_the_verdict_and_the_sample_size(self) -> None:
        text = self._frontier().render()
        assert "HURDLE FRONTIER" in text
        assert "effective after clustering" in text
        assert "break-even acc" in text

    def test_the_dict_names_where_each_instant_came_from(self) -> None:
        got = build(
            [*_instants([150.0] * 6), *_instants([200.0] * 6, source="live")], hurdle_bps=18.8
        ).as_dict()
        assert got["sources"] == {"live": 6, "replay": 6}

    def test_the_dict_carries_the_frontier_not_just_the_headline(self) -> None:
        got = self._frontier().as_dict()
        assert len(got["frontier"]) == len(HURDLE_SWEEP_BPS) + 1
        assert got["binding_constraint"]

    def test_the_summary_statistics_match_the_instants(self) -> None:
        got = build(_instants([100.0, -300.0, 200.0] * 4), hurdle_bps=18.8).as_dict()
        assert got["max_abs_move_bps"] == pytest.approx(300.0)
        assert got["median_abs_move_bps"] == pytest.approx(200.0)
