"""Tests for the real-MBO adapter -- built and checked ahead of the data it will eventually run
on (`eval/standing.py`'s thirteenth queue-model condition is blocked on a paid CME data purchase
that is the owner's decision, not this project's to make).

No real CME file exists in this repository, so nothing here asserts a result about real market
behaviour. What is checked instead: the adapter's replay logic against the REAL, installed
`databento_dbn.MBOMsg` type (not a hand-rolled mock -- constructed with the same constructor real
data would use), on hand-verified event sequences where the correct `Episode` this project's own
`eval/queueproof.py::score` should see is worked out by hand and asserted exactly.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from argus.eval.mbo_queueproof import (
    JOIN_EVERY,
    PRICE_SCALE,
    MboEvent,
    MboQueueProofError,
    _from_mbo_msg,
    dominant_instrument,
    episodes_from_mbo,
    read_dbn,
)
from argus.eval.queueproof import OUR_QTY, score


def _event(
    action: str, order_id: str, price: str, size: str, *, side: str = "B",
    instrument_id: int = 1,
) -> MboEvent:
    return MboEvent(
        ts_recv=0, action=action, side=side, order_id=order_id, instrument_id=instrument_id,
        price=Decimal(price), size=Decimal(size),
    )


class TestFromMboMsg:
    """`_from_mbo_msg` against the REAL `databento_dbn.MBOMsg` constructor, not a mock -- if the
    real schema ever changes field names or the price scale, this breaks here rather than
    silently mis-reading a real file later."""

    def test_reads_a_real_add_message(self) -> None:
        databento_dbn = pytest.importorskip("databento_dbn", reason="dev-only dependency")
        msg = databento_dbn.MBOMsg(
            publisher_id=1, instrument_id=42, order_id=7, price=4_500_250_000_000, size=3,
            flags=0, channel_id=0, action=databento_dbn.Action.ADD, side=databento_dbn.Side.BID,
            ts_recv=123, ts_in_delta=0, sequence=1, ts_event=123,
        )
        event = _from_mbo_msg(msg)
        assert event.action == "A"
        assert event.side == "B"
        assert event.order_id == "7"
        assert event.instrument_id == 42
        assert event.price == Decimal("4500250000000") * PRICE_SCALE
        assert event.price == Decimal("4500.25")
        assert event.size == Decimal("3")

    def test_price_scale_matches_the_real_sdks_own_pretty_price(self) -> None:
        """The load-bearing claim this module's own docstring makes about PRICE_SCALE, checked
        directly against the real installed package rather than trusted from memory of the DBN
        spec."""
        databento_dbn = pytest.importorskip("databento_dbn", reason="dev-only dependency")
        msg = databento_dbn.MBOMsg(
            publisher_id=1, instrument_id=1, order_id=1, price=100_000_000_000, size=1,
            flags=0, channel_id=0, action=databento_dbn.Action.ADD, side=databento_dbn.Side.ASK,
            ts_recv=0, ts_in_delta=0, sequence=1, ts_event=0,
        )
        assert Decimal(int(msg.price)) * PRICE_SCALE == Decimal(str(msg.pretty_price))


class TestReadDbnGuards:
    def test_a_missing_file_is_refused_not_silently_empty(self) -> None:
        with pytest.raises(MboQueueProofError, match="does not exist"):
            read_dbn(__import__("pathlib").Path("does/not/exist.dbn.zst"))


class TestEpisodesFromMbo:
    """Hand-verified event sequences: the correct `Episode` is worked out by hand before the
    test is written, matching this project's own established discipline for ground-truth code
    (`eval/queueproof.py`'s own docstring: 'the position of every cancellation is known by
    construction')."""

    def test_a_cancellation_ahead_genuinely_advances_the_synthetic_order(self) -> None:
        # Two real adds (30, 20) trigger our synthetic join at join_every=2, behind both --
        # entry_level = 50 (ahead of us, excluding our own 1). Then "a" cancels: true_ahead drops
        # from 50 to 20 (only "b" remains ahead).
        events = [
            _event("A", "a", "100", "30"),
            _event("A", "b", "100", "20"),
            _event("C", "a", "100", "30"),
        ]
        episodes = episodes_from_mbo(events, join_every=2, min_observations=1)
        assert len(episodes) == 1
        ep = episodes[0]
        assert ep.entry_level == pytest.approx(50.0)
        # First observation: right after we join (the join event itself is an Add, so it also
        # calls _observe with traded=0) -- true_ahead is still 50 at that instant.
        assert ep.observations[0].true_ahead == pytest.approx(50.0)
        # Second observation: after "a" cancels, true_ahead is 20.
        assert ep.observations[-1].true_ahead == pytest.approx(20.0)
        assert ep.true_filled == 0.0

    def test_a_trade_that_reaches_our_order_is_a_real_fill(self) -> None:
        # One real add (10) triggers the join at join_every=1, entry_level=10. A trade for 15
        # consumes the 10 ahead of us and then 1 unit of OUR_QTY -- a genuine fill.
        events = [
            _event("A", "a", "100", "10"),
            _event("T", "a", "100", "15"),
        ]
        episodes = episodes_from_mbo(events, join_every=1, min_observations=1)
        assert len(episodes) == 1
        ep = episodes[0]
        assert ep.entry_level == pytest.approx(10.0)
        assert ep.true_filled == pytest.approx(float(OUR_QTY))
        # The episode closes the moment it fills -- no trailing zero-ahead observation appended
        # after the close, matching `eval/queueproof.py::simulate`'s own `if our_leaves <= 0:
        # break`.
        assert ep.observations[-1].traded == pytest.approx(15.0)

    def test_a_size_decrease_modify_preserves_priority(self) -> None:
        """The exact scenario `execution/queue.py::L3FIFOQueue.reduce` was added for: without it,
        this would show "a" vanishing from ahead of us and reappearing behind -- a queue jump
        that never happened on the real venue."""
        events = [
            _event("A", "a", "100", "30"),
            _event("A", "b", "100", "20"),
            _event("M", "a", "100", "12"),  # "a" shrinks from 30 to 12, keeps its place
        ]
        episodes = episodes_from_mbo(events, join_every=2, min_observations=1)
        ep = episodes[0]
        # ahead was 50 (a=30, b=20); after "a" shrinks to 12, ahead is 12 + 20 = 32.
        assert ep.observations[-1].true_ahead == pytest.approx(32.0)

    def test_a_size_increase_modify_loses_priority(self) -> None:
        """A larger size is NOT priority-preserving on a real venue -- modelled as cancel+add,
        which sends "a" to the BACK, behind our own resting order."""
        events = [
            _event("A", "a", "100", "10"),
            _event("A", "b", "100", "5"),
            _event("M", "a", "100", "999"),  # increase -- "a" loses priority, moves to the back
        ]
        episodes = episodes_from_mbo(events, join_every=2, min_observations=1)
        ep = episodes[0]
        # Before the modify: ahead = a(10) + b(5) = 15. After: "a" is removed from the front and
        # re-added at the back (behind us), so only "b" remains ahead: true_ahead = 5.
        assert ep.observations[-1].true_ahead == pytest.approx(5.0)

    def test_a_clear_action_closes_every_open_episode(self) -> None:
        events = [
            _event("A", "a", "100", "10"),
            _event("A", "b", "100", "5"),
            _event("R", "a", "100", "0"),
        ]
        episodes = episodes_from_mbo(events, join_every=2, min_observations=1)
        assert len(episodes) == 1
        assert episodes[0].true_filled == 0.0

    def test_episodes_too_short_at_end_of_file_are_dropped(self) -> None:
        events = [_event("A", "a", "100", "10"), _event("A", "b", "100", "5")]
        # Only one Observation is ever recorded (the join event itself); min_observations=3
        # should drop it as uninformative.
        episodes = episodes_from_mbo(events, join_every=2, min_observations=3)
        assert episodes == []

    def test_the_output_feeds_the_real_scoring_function_without_error(self) -> None:
        """`eval/queueproof.py::score` is not re-implemented for real data -- this is the whole
        point of matching its `Episode`/`Observation` shape exactly."""
        events = []
        for i in range(120):
            events.append(_event("A", f"a{i}", "100", "10"))
            if i % 7 == 0:
                events.append(_event("C", f"a{max(0, i - 3)}", "100", "10"))
            if i % 5 == 0:
                events.append(_event("T", f"a{i}", "100", "4"))
        episodes = episodes_from_mbo(events, join_every=10, min_observations=1)
        assert episodes
        scored = score(episodes)
        assert len(scored) > 0
        assert all(s.episodes == len(episodes) for s in scored)


class TestJoinEveryIsRespected:
    def test_no_synthetic_order_joins_before_join_every_real_adds(self) -> None:
        events = [_event("A", "a", "100", "10")]
        episodes = episodes_from_mbo(events, join_every=JOIN_EVERY, min_observations=1)
        assert episodes == []


class TestInstrumentFiltering:
    """A single `.dbn` file can carry more than one instrument even when its filename names
    only one -- this is the guard that stops two unrelated order books from being replayed as
    if they were the same market."""

    def test_dominant_instrument_is_the_busiest_one(self) -> None:
        events = (
            [_event("A", f"a{i}", "100", "1", instrument_id=1) for i in range(5)]
            + [_event("A", f"b{i}", "200", "1", instrument_id=2) for i in range(2)]
        )
        assert dominant_instrument(events) == 1

    def test_a_second_instruments_price_levels_never_leak_into_the_replay(self) -> None:
        """Instrument 2's price "100" (same numeral as instrument 1's real level) must never be
        treated as the same book -- if it were, this would corrupt `ahead_of` with a stranger's
        resting quantity that was never really in the queue our order joined."""
        events = [
            _event("A", "a", "100", "30", instrument_id=1),
            _event("A", "b", "100", "20", instrument_id=1),
            # A same-priced order on a DIFFERENT instrument -- must be invisible to instrument 1.
            _event("A", "x", "100", "999", instrument_id=2),
        ]
        episodes = episodes_from_mbo(events, instrument_id=1, join_every=2, min_observations=1)
        assert episodes[0].entry_level == pytest.approx(50.0)  # not 50 + 999

    def test_defaults_to_the_dominant_instrument_when_none_is_named(self) -> None:
        events = (
            [_event("A", f"a{i}", "100", "1", instrument_id=7) for i in range(4)]
            + [_event("A", "x", "200", "1", instrument_id=9)]
        )
        episodes = episodes_from_mbo(events, join_every=2, min_observations=1)
        # Only instrument 7's adds (4 of them) count toward join_every=2 -> exactly 2 episodes.
        assert len(episodes) == 2


class TestTheRealArtefact:
    """The one committed record of a real run -- `data/mbo_queue_proof.json`, from a real
    2023-12-25 GLBX.MDP3 ESH4 session (nautilus_trader's own public DataBento-adapter test
    fixture, not a paid purchase). Skipped, not failed, on a machine that never generated it --
    the raw `.dbn.zst` this came from is deliberately not vendored into this repo (data, not
    code; cited by source rather than redistributed)."""

    def test_the_stored_report_is_readable_and_honest(self) -> None:
        from pathlib import Path

        from argus.eval.mbo_queueproof import REPORT_PATH

        if not REPORT_PATH.exists():
            pytest.skip("no real-MBO artefact on this machine")
        blob = json.loads(Path(REPORT_PATH).read_text(encoding="utf-8"))
        assert blob["episodes"] > 0
        assert len(blob["results"]) == 10  # 6 real models + 4 ablations, same as the synthetic run
        # The significance block is the load-bearing honesty check: a real win over the shipped
        # default, reported with its own confidence interval rather than a bare ranking.
        sig = blob.get("significance", {})
        if "best_vs_shipped_default" in sig:
            row = sig["best_vs_shipped_default"]
            assert row["ci_lo"] <= row["mean"] <= row["ci_hi"]
