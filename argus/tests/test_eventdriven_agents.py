"""eval/eventdriven_agents.py: slimon's own events, gated four ways, traded.

The pure pieces — trigger placement, de-overlapping, one position per name, the day-clustered
score and the chain-defect count — are pinned on hand-built input. The pieces that run a rival's
own code need its clone (neither slimon nor Nicholas-03/trading-bot publishes a licence, so neither
is vendored) and skip, naming the clone, when it is absent; the replay-fidelity test is the one
that says whether anything downstream is slimon's perception at all.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from argus.eval import eventdriven_agents as ea
from argus.eval.baselines import eventdriven_agents_loader as loader

BAR = ea.BAR_MS


def _tape(bars: int, closes: dict[str, list[float]] | None = None) -> ea.Tape:
    """A synthetic tape on a 5-minute grid. Only what the pure functions read is filled in."""
    ms = tuple(1_750_000_000_000 - 1_750_000_000_000 % BAR + k * BAR for k in range(bars))
    closes = closes or {"AAPLUSDT": [100.0 + 0.01 * k for k in range(bars)],
                        ea.MARKET: [200.0] * bars}
    rows = {s: [[float(m), c, c, c, c, 1.0] for m, c in zip(ms, v, strict=True)]
            for s, v in closes.items()}
    return ea.Tape(ms=ms, rows=rows, closes=closes, returns={s: [] for s in closes},
                   stamps=tuple(datetime.fromtimestamp(m / 1000, tz=UTC) for m in ms[1:]),
                   index={m: i for i, m in enumerate(ms)}, source_sha256="", fetched_at="")


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


class TestTriggers:
    def test_direction_reads_each_event_type_its_own_way(self) -> None:
        assert ea._direction({"type": "price_move", "payload": {"return_pct": -1.2}}) == -1
        assert ea._direction({"type": "range_expansion",
                              "payload": {"candle": {"o": 10, "c": 11}}}) == 1
        assert ea._direction({"type": "volume_spike",
                              "payload": {"candle_return_pct": 0.0}}) == 0

    def test_index_perps_flat_events_and_short_history_are_not_triggers(self) -> None:
        bars = ea.ESTIMATION_BARS + ea.GAP_BARS + ea.HOLD_BARS + 1200
        tape = _tape(bars)
        j = ea.ESTIMATION_BARS + ea.GAP_BARS + 100
        at = _iso(tape.ms[j])
        up = {"payload": {"return_pct": 1.5}}
        events = [
            {"type": "price_move", "symbol": "AAPLUSDT", "source_ts": at, **up},
            {"type": "price_move", "symbol": ea.MARKET, "source_ts": at, **up},
            {"type": "price_move", "symbol": "AAPLUSDT", "source_ts": at,
             "payload": {"return_pct": 0.0}},
            {"type": "price_move", "symbol": "AAPLUSDT", "source_ts": _iso(tape.ms[10]), **up},
            {"type": "news", "symbol": "AAPLUSDT", "source_ts": at, **up},
        ]
        got = ea.triggers_from(tape, events)
        assert got == [ea.Trigger("AAPLUSDT", "price_move", 1, j)]
        assert got[0].label == "price_move_up"

    def test_deoverlap_keeps_one_window_per_name_and_class(self) -> None:
        t = [ea.Trigger("A", "price_move", 1, 1000), ea.Trigger("A", "price_move", 1, 1010),
             ea.Trigger("A", "price_move", -1, 1010), ea.Trigger("B", "price_move", 1, 1010),
             ea.Trigger("A", "price_move", 1, 1000 + ea.HOLD_BARS)]
        kept = ea.deoverlap(t)
        assert ea.Trigger("A", "price_move", 1, 1010) not in kept
        assert len(kept) == 4


class TestTrading:
    def test_one_position_per_name_and_the_hurdle_is_charged(self) -> None:
        bars = 400
        tape = _tape(bars, {"AAPLUSDT": [100.0] * 100 + [101.0] * 300, ea.MARKET: [1.0] * bars})
        triggers = [ea.Trigger("AAPLUSDT", "price_move", 1, 50),
                    ea.Trigger("AAPLUSDT", "volume_spike", 1, 60),
                    ea.Trigger("AAPLUSDT", "price_move", 1, 50 + ea.HOLD_BARS)]
        plan = {"price_move_up": 1, "volume_spike_up": 1}
        trades = ea.trade(tape, triggers, plan)
        assert [t["class"] for t in trades] == ["price_move_up", "price_move_up"]
        assert trades[0]["net_bps"] == pytest.approx(100.0 - ea.COST_BPS)
        assert trades[1]["net_bps"] == pytest.approx(-ea.COST_BPS)

    def test_a_class_the_plan_does_not_trade_is_skipped(self) -> None:
        tape = _tape(300)
        assert ea.trade(tape, [ea.Trigger("AAPLUSDT", "price_move", 1, 10)], {}) == []

    def test_score_clusters_the_t_by_day(self) -> None:
        trades: list[dict[str, Any]] = [
            {"at": "2026-09-01T01:00:00+00:00", "net_bps": 10.0},
            {"at": "2026-09-01T09:00:00+00:00", "net_bps": 10.0},
            {"at": "2026-09-02T01:00:00+00:00", "net_bps": -5.0},
            {"at": "2026-09-03T01:00:00+00:00", "net_bps": 15.0},
        ]
        s = ea.score_trades(trades)
        assert s["trades"] == 4 and s["days"] == 3
        assert s["net_bps_total"] == 30.0 and s["hit_rate"] == 0.75
        daily = [20.0, -5.0, 15.0]
        mean = sum(daily) / 3
        sd = (sum((x - mean) ** 2 for x in daily) / 2) ** 0.5
        assert s["day_clustered_t"] == round(mean / (sd / 3 ** 0.5), 3)

    def test_no_trades_scores_as_flat_not_as_an_error(self) -> None:
        assert ea.score_trades([]) == {"trades": 0, "net_bps_total": 0.0, "net_bps_mean": None,
                                       "hit_rate": None, "day_clustered_t": None, "days": 0}


class TestGates:
    def test_ballast_refuses_below_its_own_eight_observation_floor(self) -> None:
        tape = _tape(400)
        call = ea.gate_ballast(tape, [ea.Trigger("AAPLUSDT", "price_move", 1, k)
                                      for k in range(10, 17)])
        assert not call.trade and call.detail == {"refused": "fewer than 8 events"}

    def test_whale_with_no_events_refuses_instead_of_crashing_in_their_code(self) -> None:
        call = ea.gate_whale(_tape(400), [], 1)
        assert not call.trade and call.events == 0

    def test_whale_below_thirty_events_is_their_skip(self) -> None:
        tape = _tape(400)
        call = ea.gate_whale(tape, [ea.Trigger("AAPLUSDT", "price_move", 1, k)
                                    for k in range(10, 20)], 1)
        assert not call.trade and call.detail == {"refused": "fewer than 30 events"}

    def test_argus_refuses_below_its_event_floor(self) -> None:
        call = ea.gate_argus(_tape(400), [])
        assert not call.trade and "refused" in call.detail


class TestChainDefect:
    def test_counts_split_on_the_fix_date(self, tmp_path: Path) -> None:
        rows = [
            {"decided_at": "2026-09-23T10:00:00Z", "symbol": "TSLAUSDT",
             "event": "TSLAUSDT last 250.1", "links": [{"falsifier": ""}], "grades": ["ungraded"]},
            {"decided_at": "2026-09-25T10:00:00Z", "symbol": "TSLAUSDT", "event": "CPI print",
             "links": [{"falsifier": "CPI revised"}, {"falsifier": "x"}],
             "grades": ["supported", "unsupported"]},
        ]
        path = tmp_path / "chains.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        out = ea.chain_defect_evidence(path)
        assert out["before_fix"] == {"chains": 1, "links": 1, "links_with_a_falsifier": 0,
                                     "chains_every_link_falsifiable": 0,
                                     "event_field_is_the_price_line": 1, "graded_links": 0}
        assert out["after_fix"]["chains_every_link_falsifiable"] == 1
        assert out["after_fix"]["graded_links"] == 1


class TestProvenance:
    def test_provenance_carries_no_home_path(self) -> None:
        for row in loader.provenance().values():
            assert not Path(row["clone"]).is_absolute()
            assert "Users" not in row["clone"]

    def test_a_missing_clone_names_the_clone_command(self, tmp_path: Path) -> None:
        with pytest.raises(loader.AgentLoadError, match="git clone"):
            loader._require(tmp_path / "absent.py", loader.SLIMON_URL, tmp_path)


needs_clones = pytest.mark.skipif(
    not (loader.SLIMON_CLONE / "slimon" / "perception.py").is_file()
    or not (loader.NICHOLAS_CLONE / "news" / "filters.py").is_file()
    or not ea.SNAPSHOT_5M.is_file(),
    reason=f"needs `git clone {loader.SLIMON_URL}` and `git clone {loader.NICHOLAS_URL}` under "
           f"research/repos-owned/, and data/{ea.SNAPSHOT_5M.name}",
)


@needs_clones
class TestAgainstTheirOwnLog:
    @pytest.fixture(scope="class")
    def tape(self) -> ea.Tape:
        return ea.load_tape()

    def test_the_replay_reproduces_what_slimon_itself_logged(self, tape: ea.Tape) -> None:
        """Their perception, driven at their own tick times, must find the events their agent
        logged. Measured 2026-09-26 at commit 7d0ae2d: 1,690 of 1,739 (recall 0.972, precision
        0.962) — the misses are 30-minute moves on the boundary of their 1.0% threshold."""
        v = ea.validate_replay(tape)
        assert v["recall_of_their_log"] >= 0.95
        assert v["precision_against_their_log"] >= 0.95
        assert v["by_type"]["us_regular_open"]["matched"] \
            == v["by_type"]["us_regular_open"]["logged_by_slimon"]

    def test_nicholas_gate_runs_their_filters_on_every_logged_headline(
        self, tape: ea.Tape,
    ) -> None:
        out = ea.nicholas_gate(tape)
        assert out["passed"] + out["rejected"] == out["headlines"] > 0
        assert out["abnormal_move_size_ratio"]["passed"]["events"] > 0


def test_artefact_matches_a_fresh_count_of_its_own_trades() -> None:
    if not ea.REPORT_PATH.exists():
        pytest.skip("run python -m argus.eval.eventdriven_agents first")
    report = json.loads(ea.REPORT_PATH.read_text(encoding="utf-8"))
    forward = report["walk_forward"]
    for gate, row in forward["out_of_sample_combined"].items():
        assert row == ea.score_trades(forward["test_trades"][gate]), gate
    assert report["generated_from"]["snapshot_sha256"] == ea.load_tape().source_sha256
