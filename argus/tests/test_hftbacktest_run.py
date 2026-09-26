"""`eval/hftbacktest_run.py`: hftbacktest's own engine against ARGUS's replay on the ESH4 file.

hftbacktest is not installed where the suite runs, so the engine run itself is not repeated here;
the comparison logic is checked on constructed rows, and the committed result is pinned to what the
module's docstring and the register quote."""

from __future__ import annotations

import json
from array import array
from pathlib import Path

from argus.eval.hftbacktest_run import EngineFill, compare_l2, compare_l3
from argus.eval.l3queue import RealEpisode

ARTEFACT = Path(__file__).resolve().parents[1] / "data" / "hftbacktest_run.json"
FILLED = 3


def _episode(filled_at: int) -> RealEpisode:
    ep = RealEpisode(dataset="t", side="A", px=100, t_join=0, entry=1.0,
                     truth=array("d", [1.0, 0.0]))
    if filled_at >= 0:
        ep.end, ep.fill_ts = "filled", filled_at
    else:
        ep.end = "horizon"
    return ep


class TestTheComparisons:
    def test_l3_counts_agreement_and_same_timestamps(self) -> None:
        eps = [_episode(10), _episode(20), _episode(-1), _episode(30)]
        engine = [EngineFill(FILLED, 10), EngineFill(FILLED, 21), EngineFill(4, -1),
                  EngineFill(4, -1)]
        out = compare_l3(eps, engine, FILLED)
        assert (out["both_filled"], out["argus_only"], out["neither"]) == (2, 1, 1)
        assert out["same_fill_timestamp_of_both_filled"] == 1
        assert out["fill_agreement"] == 0.75

    def test_l2_separates_false_and_missed_fills(self) -> None:
        eps = [_episode(10), _episode(-1), _episode(-1)]
        engine = [EngineFill(4, -1), EngineFill(FILLED, 5), EngineFill(4, -1)]
        out = compare_l2(eps, engine, FILLED)
        assert (out["false_fill_rate"], out["missed_fill_rate"]) == (0.33333, 0.33333)


class TestTheCommittedRun:
    run = json.loads(ARTEFACT.read_text(encoding="utf-8"))

    def test_the_l3_engines_agree_on_every_order(self) -> None:
        l3 = self.run["l3_truth_vs_hftbacktest_l3_fifo"]
        assert l3["orders"] == 1438
        assert l3["argus_only"] == 0
        assert l3["engine_only"] == 0
        assert l3["same_fill_timestamp_of_both_filled"] == l3["both_filled"] == 1130

    def test_the_port_is_within_a_point_of_the_engine_on_every_model(self) -> None:
        for model, pair in self.run["l2_models"].items():
            engine = pair["hftbacktest_engine"]["fill_agreement"]
            port = pair["argus_port"]["fill_agreement"]
            assert abs(engine - port) <= 0.011, model
