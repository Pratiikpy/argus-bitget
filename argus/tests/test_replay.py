"""Replay: point-in-time or it is worthless, and never mixed with the live chain."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import pytest

from argus.paper.replay import (
    HOLD_HOURS,
    TRADEABLE_PHASES,
    Outcome,
    ReplayResult,
    frame_at,
    instants,
    load,
    realised_bps,
    write,
)
from argus.truth.evidence import Evidence

LIVE = os.environ.get("ARGUS_LIVE_VENUE") == "1"
live_only = pytest.mark.skipif(not LIVE, reason="set ARGUS_LIVE_VENUE=1 to rebuild live sources")

# A Thursday. 13:30-20:00 UTC is the US regular session.
START = datetime(2026, 9, 10, 0, 0, tzinfo=UTC)


@dataclass(frozen=True)
class Bar:
    ts: datetime
    close: float


def _bars(hours: int = 96) -> list[Bar]:
    return [Bar(ts=START + timedelta(hours=i), close=100.0 + i * 0.1) for i in range(hours)]


def _ev(at: datetime, claim: str = "a filing") -> Evidence:
    return Evidence(id=f"e-{at.isoformat()}", claim=claim, source="sec-edgar",
                    available_at=at, credibility=1.0)


def _outcome(**kw: object) -> Outcome:
    base: dict[str, object] = {
        "symbol": "NVDAUSDT", "at": START + timedelta(hours=14), "verdict": "no_trade",
        "side": "flat", "quantity": Decimal("0"), "confidence": 0.8, "thesis": "t",
        "evidence_count": 8, "realised_bps": 50.0, "cost_bps": 12.0,
    }
    base.update(kw)
    return Outcome(**base)  # type: ignore[arg-type]


class TestTheFrameCannotSeeItsOwnAnswer:
    def test_no_bar_after_the_instant_reaches_the_price(self) -> None:
        bars = _bars()
        at = START + timedelta(hours=20)
        frame = frame_at("NVDAUSDT", at, bars)
        assert frame is not None
        assert frame.price == Decimal(str(bars[20].close))

    def test_evidence_dated_after_the_instant_is_refused(self) -> None:
        """The off-by-one that would hand the desk the answer."""
        at = START + timedelta(hours=20)
        future = _ev(at + timedelta(hours=1), "tomorrow's filing")
        frame = frame_at("NVDAUSDT", at, _bars(), macro=(future,))
        assert frame is not None
        assert all("tomorrow" not in e.claim for e in frame.evidence)

    def test_evidence_dated_exactly_at_the_instant_is_admitted(self) -> None:
        at = START + timedelta(hours=20)
        frame = frame_at("NVDAUSDT", at, _bars(), macro=(_ev(at, "right now"),))
        assert frame is not None
        assert any("right now" in e.claim for e in frame.evidence)

    def test_an_instant_before_any_bar_is_not_reconstructible(self) -> None:
        assert frame_at("NVDAUSDT", START - timedelta(hours=1), _bars()) is None

    def test_the_absent_sources_are_named_on_every_frame(self) -> None:
        """A thinner frame biases toward abstention, so the thinness must travel with it."""
        frame = frame_at("NVDAUSDT", START + timedelta(hours=20), _bars())
        assert frame is not None and frame.absent
        assert any("RSS" in a for a in frame.absent)

    def test_truncating_the_future_does_not_change_the_frame(self) -> None:
        at = START + timedelta(hours=20)
        full = frame_at("NVDAUSDT", at, _bars(96))
        short = frame_at("NVDAUSDT", at, _bars(21))
        assert full is not None and short is not None
        assert full.price == short.price


class TestSettlementUsesTheFullHorizon:
    def test_the_move_is_measured_over_the_hold_window(self) -> None:
        bars = _bars()
        at = START + timedelta(hours=20)
        got = realised_bps(bars, at)
        expected = (bars[44].close - bars[20].close) / bars[20].close * 10_000
        assert got == pytest.approx(expected)

    def test_a_horizon_past_the_data_does_not_settle_early(self) -> None:
        """Settling at the last available bar would score a 24-hour call against whatever hours
        happened to remain, and short horizons would look systematically calmer."""
        bars = _bars(30)
        assert realised_bps(bars, START + timedelta(hours=20)) is None

    def test_the_hold_window_matches_the_live_protocol(self) -> None:
        assert HOLD_HOURS == 24


class TestInstantSelection:
    def test_only_tradeable_sessions_are_replayed(self) -> None:
        """Replaying a weekend instant would re-measure what the live ledger already shows."""
        from argus.truth.clocks import DualClock

        clock = DualClock()
        for at in instants(_bars(96)):
            assert str(clock.state(at).phase) in TRADEABLE_PHASES

    def test_instants_are_spaced_so_horizons_do_not_fully_overlap(self) -> None:
        chosen = instants(_bars(96), every_hours=6)
        gaps = [(b - a).total_seconds() / 3600 for a, b in pairwise(chosen)]
        assert all(g >= 6 for g in gaps), gaps

    def test_the_tail_leaves_room_for_the_hold(self) -> None:
        bars = _bars(96)
        chosen = instants(bars)
        assert chosen
        assert max(chosen) + timedelta(hours=HOLD_HOURS) <= max(b.ts for b in bars)

    def test_no_bars_gives_no_instants(self) -> None:
        assert instants([]) == []


class TestAnAbstentionIsNotCreditedWithTheMoveItDeclined:
    def test_an_abstention_nets_zero_however_large_the_move(self) -> None:
        got = _outcome(realised_bps=500.0)
        assert not got.opened and got.net_bps == 0.0

    def test_a_long_earns_the_move_less_the_round_trip(self) -> None:
        got = _outcome(verdict="trade", side="buy", quantity=Decimal("1"), realised_bps=100.0)
        assert got.opened and got.net_bps == pytest.approx(88.0)

    def test_a_short_earns_the_inverse(self) -> None:
        got = _outcome(verdict="trade", side="sell", quantity=Decimal("1"), realised_bps=-100.0)
        assert got.net_bps == pytest.approx(88.0)

    def test_a_zero_quantity_trade_is_not_an_open_position(self) -> None:
        assert not _outcome(verdict="trade", side="buy", quantity=Decimal("0")).opened

    def test_the_counterfactual_is_kept_separately(self) -> None:
        """The abstention's counterfactual must not become its return."""
        got = _outcome(realised_bps=500.0).as_dict()
        assert got["realised_bps"] == 500.0 and got["net_bps"] == 0.0


class TestTheVerdictOnTheHypothesis:
    def test_all_abstentions_is_reported_as_not_supported(self) -> None:
        result = ReplayResult(outcomes=tuple(_outcome() for _ in range(5)), instants=5)
        assert "NOT SUPPORTED" in result.hypothesis_verdict

    def test_it_states_the_caveat_that_weakens_its_own_finding(self) -> None:
        """A replayed frame is thinner than a live one, which biases toward abstention."""
        result = ReplayResult(outcomes=tuple(_outcome() for _ in range(5)), instants=5)
        assert "biases the desk toward abstention" in result.hypothesis_verdict
        assert "weakens the finding" in result.hypothesis_verdict

    def test_one_position_makes_it_supported_in_part(self) -> None:
        outcomes = (_outcome(), _outcome(verdict="trade", side="buy", quantity=Decimal("1")))
        result = ReplayResult(outcomes=outcomes, instants=2)
        assert "SUPPORTED in part" in result.hypothesis_verdict

    def test_no_decisions_leaves_the_hypothesis_untouched(self) -> None:
        assert "untouched" in ReplayResult(outcomes=(), instants=0).hypothesis_verdict

    def test_the_report_says_it_is_not_the_live_chain(self) -> None:
        text = ReplayResult(outcomes=(_outcome(),), instants=1).render()
        assert "NEVER to the live chain" in text
        assert "replay_ledger.jsonl" in text


class TestItNeverTouchesTheLiveLedger:
    def test_it_writes_to_its_own_file(self, tmp_path: Path) -> None:
        path = tmp_path / "replay.jsonl"
        write(ReplayResult(outcomes=(_outcome(),), instants=1), path=path)
        assert path.exists()
        assert load(path)[0]["kind"] == "replay"

    def test_every_row_is_labelled_as_a_replay(self, tmp_path: Path) -> None:
        """So a row can never be mistaken for a live decision by anything reading the file."""
        path = tmp_path / "replay.jsonl"
        write(ReplayResult(outcomes=tuple(_outcome() for _ in range(3)), instants=3), path=path)
        assert all(row["kind"] == "replay" for row in load(path))

    def test_the_default_path_is_not_the_live_ledger(self) -> None:
        from argus.paper.replay import REPLAY_PATH
        from argus.paper.runner import LEDGER_PATH

        assert REPLAY_PATH != LEDGER_PATH
        assert "replay" in REPLAY_PATH.name

    def test_a_missing_file_loads_as_empty(self, tmp_path: Path) -> None:
        assert load(tmp_path / "nothing.jsonl") == []


@live_only
class TestReconstructionAgainstRealSources:
    def test_every_rebuilt_item_predates_the_instant(self) -> None:
        from argus.paper.replay import reconstruct_sources

        at = datetime(2026, 9, 10, 14, 0, tzinfo=UTC)
        rebuilt, _absent = reconstruct_sources("NVDAUSDT", at)
        assert rebuilt, "nothing was reconstructed"
        for item in rebuilt:
            assert item.available_at <= at, item.claim

    def test_it_rebuilds_more_than_one_channel(self) -> None:
        """A single-channel frame is the thin frame this reconstruction exists to avoid."""
        from argus.paper.replay import reconstruct_sources

        rebuilt, _ = reconstruct_sources("NVDAUSDT", datetime(2026, 9, 10, 14, 0, tzinfo=UTC))
        assert len({e.source for e in rebuilt}) >= 2

    def test_short_volume_is_from_a_session_strictly_before(self) -> None:
        """FINRA publishes a session's file after that session closes."""
        from argus.paper.replay import reconstruct_sources

        at = datetime(2026, 9, 10, 14, 0, tzinfo=UTC)
        rebuilt, _ = reconstruct_sources("NVDAUSDT", at)
        shorts = [e for e in rebuilt if "short volume" in e.claim]
        for item in shorts:
            assert item.available_at.date() < at.date()
