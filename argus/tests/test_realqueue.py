"""`eval/realqueue.py` and the hftbacktest-free parts of `eval/hftbacktest_run.py`.

The driver is checked on hand-built episodes whose every queue step is worked out by hand, so a
model that is scored against the truth is scored by a driver that is itself pinned.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from argus.eval.hftbacktest_run import (
    BUY_EVENT,
    EXCH_EVENT,
    LOCAL_EVENT,
    SELL_EVENT,
    events_from_hbt_array,
    schedule_from,
)
from argus.eval.l3queue import (
    ADD,
    FILL,
    STEP_BATCH,
    STEP_DEPTH,
    STEP_THROUGH,
    STEP_TRADE,
    TRADE,
    RealEpisode,
)
from argus.eval.realqueue import (
    RIVAL_DEFAULT,
    ablation_models,
    cluster_bootstrap,
    drive,
    rival_models,
    score_models,
    time_blocks,
)
from argus.execution.queue import RiskAdverseQueue


def episode(steps: list[tuple[int, float]], truth: list[float], true_fill: int = -1, *,
            entry: float = 10.0, t_join: int = 0, dataset: str = "d") -> RealEpisode:
    ep = RealEpisode(dataset=dataset, side="A", px=100, t_join=t_join, entry=entry)
    for kind, value in steps:
        ep.kinds.append(kind)
        ep.vals.append(value)
    for k, v in enumerate(truth):
        ep.truth.append(v)
        ep.batch_ts.append(1000 + k)
    ep.true_fill_batch = true_fill
    return ep


# Entry 10 ahead. A 4-lot print (6 ahead), a depth drop to 3 (3 ahead), a 5-lot print (through
# by 2: filled). The truth agrees at every event end.
WORKED = [(STEP_TRADE, 4.0), (STEP_BATCH, 0.0), (STEP_DEPTH, 3.0), (STEP_BATCH, 0.0),
          (STEP_TRADE, 5.0), (STEP_BATCH, 0.0)]


class TestDrive:
    @pytest.mark.parametrize("view", ["record", "event"])
    def test_a_hand_worked_episode_scores_zero_error(self, view: str) -> None:
        s = drive(episode(WORKED, [6.0, 3.0, 0.0], 2), RiskAdverseQueue(), lot=Decimal(1),
                  view=view)
        assert (s.error, s.samples, s.model_fill, s.true_fill) == (0.0, 3, 2, 2)

    def test_error_is_normalised_by_the_entry(self) -> None:
        s = drive(episode(WORKED, [4.0, 3.0, 0.0], 2), RiskAdverseQueue(), lot=Decimal(1))
        assert s.error == pytest.approx((2.0 / 10.0) / 3)

    def test_a_through_step_fills_at_once(self) -> None:
        ep = episode([(STEP_THROUGH, 0.0), (STEP_BATCH, 0.0)], [0.0], 0)
        s = drive(ep, RiskAdverseQueue(), lot=Decimal(1))
        assert s.model_fill == 0 and s.model_fill_ts(ep) == 1000

    def test_the_event_view_applies_only_the_last_depth_of_an_event(self) -> None:
        # Depth 12 then 2 inside one event. Record view: min(10, 12) then min(10, 2) = 2.
        # Event view: only the final 2 is fed. Risk-adverse lands on 2 either way; a model that
        # reacts to the intermediate rise would not, which is what the event view exists to show.
        steps = [(STEP_DEPTH, 12.0), (STEP_DEPTH, 2.0), (STEP_BATCH, 0.0)]
        for view in ("record", "event"):
            s = drive(episode(steps, [2.0]), RiskAdverseQueue(), lot=Decimal(1), view=view)
            assert s.error == 0.0


class TestScoring:
    def test_false_and_missed_fills_are_counted_against_the_truth(self) -> None:
        filled = episode(WORKED, [6.0, 3.0, 0.0], 2)
        never = episode(WORKED, [6.0, 3.0, 3.0], -1)          # the truth never filled
        empty = episode([], [])                                  # no sample: not scored
        [res] = score_models([filled, never, empty], [("risk", RiskAdverseQueue)],
                             lot=Decimal(1))
        assert (res.episodes, res.false_fills, res.missed_fills, res.agree) == (2, 1, 0, 1)
        row = res.as_dict()
        assert row["false_fill_rate"] == 0.5 and row["cost_bps"] > 0
        assert not res.is_ablation

    def test_the_rival_set_is_what_hftbacktest_ships(self) -> None:
        names = [n for n, _ in rival_models()]
        assert names[0] == RIVAL_DEFAULT and len(names) == len(set(names)) == 8
        assert all(n.startswith("ABLATION") for n, _ in ablation_models())

    def test_cluster_bootstrap(self) -> None:
        a = [1.0, 1.0, 1.0, 1.0]
        b = [2.0, 2.0, 3.0, 3.0]
        assert cluster_bootstrap(a, b, ["x", "x", "x", "x"]) == (1.5, 1.5, 1.5)
        mean, lo, hi = cluster_bootstrap(a, b, ["x", "x", "y", "y"])
        assert mean == 1.5 and 1.0 <= lo <= mean <= hi <= 2.0
        assert cluster_bootstrap(a, b, ["x", "x", "y", "y"]) == (mean, lo, hi)

    def test_time_blocks_key_scored_episodes_by_five_minute_join_block(self) -> None:
        eps = [episode(WORKED, [1.0], t_join=0), episode([], [], t_join=1),
               episode(WORKED, [1.0], t_join=300_000_000_000)]
        assert time_blocks(eps) == ["d|0", "d|1"]


class TestHftbacktestArrays:
    def test_the_rival_event_array_becomes_argus_events(self) -> None:
        dtype = np.dtype([("ev", "u8"), ("exch_ts", "i8"), ("local_ts", "i8"), ("px", "f8"),
                          ("qty", "f8"), ("order_id", "u8"), ("ival", "i8"), ("fval", "f8")])
        rows = [
            (EXCH_EVENT | LOCAL_EVENT | BUY_EVENT | 10, 100, 101, 1.25, 3.0, 7, 0, 0.0),
            (LOCAL_EVENT | BUY_EVENT | 10, 100, 101, 1.25, 3.0, 8, 0, 0.0),   # local only
            (EXCH_EVENT | SELL_EVENT | 13, 100, 101, 1.50, 1.0, 9, 0, 0.0),
            (EXCH_EVENT | 1, 150, 151, 1.50, 1.0, 0, 0, 0.0),                   # depth: skipped
            (EXCH_EVENT | 2, 200, 201, 1.50, 1.0, 0, 0, 0.0),                   # sideless trade
        ]
        data = np.array(rows, dtype=dtype)
        out = events_from_hbt_array(data, tick=0.25)
        # The add shares its exchange timestamp with the fill, so the event is still open after
        # it; the next exchange record (a depth event, not converted) has a new timestamp, so the
        # fill closes the event.
        assert [(e.kind, e.side, e.px, e.order_id, e.batch_end) for e in out] == [
            (ADD, "B", 5, 7, False), (FILL, "A", 6, 9, True), (TRADE, "N", 6, 0, True)]

    def test_schedule_from_orders_joins_in_time(self) -> None:
        eps = [RealEpisode("d", "A", 6, 50, 1.0), RealEpisode("d", "B", 5, 10, 1.0)]
        times, sides, prices = schedule_from(eps, 0.25)
        assert times.tolist() == [11, 51]
        assert sides.tolist() == [1, -1]
        assert prices.tolist() == [1.25, 1.5]
