"""`eval/l3queue.py` and `eval/l3feeds.py`: the market-by-order truth replay, pinned against
hand-worked sequences rather than against its own output.

Every expectation below is worked out from the FIFO rule by hand before the replay runs: which
orders are ahead of the synthetic one, which are behind, and which execution fills it under
hftbacktest's ``L3FIFOQueueModel::fill_market_feed_order`` convention (``queue.rs:974-1070``).
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from argus.eval.l3feeds import (
    LOBSTER_EMPTY_ASK,
    LOBSTER_EMPTY_BID,
    FeedError,
    LobsterSource,
    _book_row,
    _ns,
    databento_events,
    lobster_message_events,
    replay_databento,
    replay_lobster,
)
from argus.eval.l3queue import (
    ADD,
    CANCEL,
    CLEAR,
    EXEC,
    FILL,
    HALT,
    MODIFY,
    REDUCE,
    STEP_BATCH,
    STEP_DEPTH,
    STEP_THROUGH,
    STEP_TRADE,
    TRADE,
    L3Event,
    ReplayConfig,
    TruthReplay,
    _Virtual,
    iter_batches,
)

ONE_JOIN = ReplayConfig(every_ns=10**15, horizon_ns=10**15)
"""One synthetic bid and one synthetic ask at the first event end, then none."""

ESH4 = (Path(__file__).resolve().parents[2] / "research" / "repos-t2"
        / "nautechsystems~nautilus_trader" / "test_data" / "databento"
        / "esh4-glbx-mdp3-20231225.mbo.dbn.zst")


def ev(ts: int, kind: str, side: str, px: int, size: float, oid: int,
       batch_end: bool = True) -> L3Event:
    return L3Event(ts, kind, side, px, size, oid, batch_end)


def seeded(replay: TruthReplay, *events: L3Event) -> None:
    for e in events:
        replay.apply(e)
        if e.batch_end:
            replay.end_batch(e.ts, None)


def the_ask(replay: TruthReplay) -> _Virtual:
    [v] = [v for v in replay.active if v.ep.side == "A"]
    return v


class TestFifo:
    def test_the_synthetic_order_joins_behind_everything_resting(self) -> None:
        r = TruthReplay("t", ONE_JOIN)
        seeded(r, ev(1, ADD, "A", 100, 7, 1), ev(2, ADD, "B", 99, 3, 2))
        v = the_ask(r)
        assert (v.ep.px, v.ep.entry, v.ahead, v.behind) == (100, 7.0, 7.0, 0.0)

    def test_later_adds_queue_behind_and_their_cancels_do_not_move_it(self) -> None:
        r = TruthReplay("t", ONE_JOIN)
        seeded(r, ev(1, ADD, "A", 100, 7, 1, False), ev(2, ADD, "A", 100, 4, 2))
        v = the_ask(r)
        seeded(r, ev(3, ADD, "A", 100, 5, 3))
        assert (v.ahead, v.behind) == (11.0, 5.0)
        seeded(r, ev(4, CANCEL, "A", 100, 5, 3))
        assert (v.ahead, v.behind) == (11.0, 0.0)
        seeded(r, ev(5, CANCEL, "A", 100, 4, 2))
        assert v.ahead == 7.0
        assert r.self_consistent(v)

    def test_a_size_increase_loses_priority_and_a_decrease_keeps_it(self) -> None:
        r = TruthReplay("t", ONE_JOIN)
        seeded(r, ev(1, ADD, "A", 100, 7, 1, False), ev(2, ADD, "A", 100, 4, 2))
        v = the_ask(r)
        seeded(r, ev(3, MODIFY, "A", 100, 2, 1))   # 7 -> 2, keeps its place ahead
        assert (v.ahead, v.behind) == (6.0, 0.0)
        seeded(r, ev(4, MODIFY, "A", 100, 9, 2))   # 4 -> 9, re-queued behind the synthetic
        assert (v.ahead, v.behind) == (2.0, 9.0)
        assert r.self_consistent(v)

    def test_a_partial_cancel_keeps_priority(self) -> None:
        r = TruthReplay("t", ONE_JOIN)
        seeded(r, ev(1, ADD, "A", 100, 7, 1))
        v = the_ask(r)
        seeded(r, ev(2, REDUCE, "A", 100, 3, 1))
        assert v.ahead == 4.0


class TestTheEsh4Match:
    """The first trade of the 2023-12-25 ESH4 session, as the module docstring states it: ``T B 5,
    F A 5, M A 2`` then ``T B 2, F A 2, C A 2`` on one resting ask. The earlier replay counted the
    trade on ``T`` and again on ``F`` and then sent the order to the back on the ``M``."""

    def test_trade_and_fill_records_do_not_touch_the_book(self) -> None:
        r = TruthReplay("t", ONE_JOIN)
        seeded(r, ev(1, ADD, "A", 100, 7, 1, False), ev(2, ADD, "A", 100, 3, 2))
        v = the_ask(r)
        seeded(r, ev(3, ADD, "A", 100, 4, 3))    # behind the synthetic
        seeded(r, ev(4, TRADE, "B", 100, 5, 0, False), ev(4, FILL, "A", 100, 5, 1, False),
               ev(4, MODIFY, "A", 100, 2, 1))
        assert (v.ahead, v.behind, v.filled) == (5.0, 4.0, False)
        assert r.totals[("A", 100)] == 9.0
        seeded(r, ev(5, TRADE, "B", 100, 2, 0, False), ev(5, FILL, "A", 100, 2, 1, False),
               ev(5, CANCEL, "A", 100, 2, 1))
        assert (v.ahead, v.filled) == (3.0, False)
        # The L2 view saw two prints at our price and the depth updates, never a double count.
        kinds = list(v.ep.kinds)
        assert kinds.count(STEP_TRADE) == 2

    def test_an_execution_behind_us_fills_us_and_one_ahead_does_not(self) -> None:
        r = TruthReplay("t", ONE_JOIN)
        seeded(r, ev(1, ADD, "A", 100, 7, 1))
        v = the_ask(r)
        seeded(r, ev(2, ADD, "A", 100, 4, 2))
        seeded(r, ev(3, FILL, "A", 100, 7, 1, False), ev(3, CANCEL, "A", 100, 7, 1))
        assert not v.filled
        seeded(r, ev(4, FILL, "A", 100, 1, 2, False), ev(4, MODIFY, "A", 100, 3, 2))
        assert v.filled
        # A filled order stays open to its horizon so L2 models read the same feed as
        # hftbacktest's engine; it closes as filled.
        assert [e.end for e in r.finish() if e.side == "A"] == ["filled"]
        ep = next(e for e in r.episodes if e.side == "A")
        assert ep.fill_ts == 4 and ep.truth[-1] == 0.0 and ep.true_fill_batch == ep.batches - 1

    def test_fill_caused_reductions_are_not_logged_as_cancellations(self) -> None:
        r = TruthReplay("t", ONE_JOIN)
        r.attribution = []
        seeded(r, ev(1, ADD, "A", 100, 7, 1), ev(2, ADD, "A", 100, 3, 2))
        seeded(r, ev(3, FILL, "A", 100, 5, 1, False), ev(3, MODIFY, "A", 100, 2, 1))
        assert r.attribution == []
        seeded(r, ev(4, CANCEL, "A", 100, 3, 2))    # behind the synthetic
        seeded(r, ev(5, CANCEL, "A", 100, 2, 1))    # ahead of it
        assert [(q, from_back) for _, _, q, from_back, _ in r.attribution] == [
            (3.0, True), (2.0, False)]


class TestThroughAndCrossing:
    def test_an_execution_through_our_price_fills_us(self) -> None:
        r = TruthReplay("t", ONE_JOIN)
        seeded(r, ev(1, ADD, "A", 100, 7, 1), ev(2, ADD, "A", 101, 4, 2))
        v = the_ask(r)
        # A resting ask at 101 executes while the best ask is 100: everything from 100 up to (not
        # including) 101 has traded through.
        seeded(r, ev(3, FILL, "A", 101, 4, 2, False), ev(3, CANCEL, "A", 101, 4, 2))
        assert v.filled

    def test_a_bid_at_our_ask_crosses_it(self) -> None:
        r = TruthReplay("t", ONE_JOIN)
        seeded(r, ev(1, ADD, "A", 100, 7, 1), ev(2, ADD, "B", 98, 1, 2))
        v = the_ask(r)
        seeded(r, ev(3, ADD, "B", 100, 1, 3))
        assert v.filled and STEP_THROUGH in v.ep.kinds
        assert r.stats.crossings == 1

    def test_a_print_through_our_price_is_a_through_step_for_the_l2_view(self) -> None:
        r = TruthReplay("t", ONE_JOIN)
        seeded(r, ev(1, ADD, "A", 100, 7, 1))
        v = the_ask(r)
        seeded(r, ev(2, TRADE, "B", 101, 1, 0))
        assert list(v.ep.kinds)[:1] == [STEP_THROUGH]
        assert not v.filled


class TestEpisodeLifecycle:
    def test_no_join_on_a_locked_book(self) -> None:
        r = TruthReplay("t", ONE_JOIN)
        seeded(r, ev(1, ADD, "A", 100, 7, 1, False), ev(1, ADD, "B", 100, 1, 2))
        assert r.active == [] and r.stats.joins == 0

    def test_min_level_skips_a_thin_touch(self) -> None:
        r = TruthReplay("t", ReplayConfig(every_ns=10**15, horizon_ns=10**15, min_level=5.0))
        seeded(r, ev(1, ADD, "A", 100, 3, 1, False), ev(1, ADD, "B", 99, 9, 2))
        assert [v.ep.side for v in r.active] == ["B"]

    def test_horizon_closes_and_the_join_grid_is_fixed(self) -> None:
        r = TruthReplay("t", ReplayConfig(every_ns=10, horizon_ns=25))
        seeded(r, ev(0, ADD, "A", 100, 3, 1))
        for t in range(1, 60):
            seeded(r, ev(t, ADD, "A", 100, 1, 100 + t))
        joins = sorted({e.t_join for e in r.episodes} | {v.ep.t_join for v in r.active})
        assert joins == [0, 10, 20, 30, 40, 50]
        assert {e.end for e in r.episodes} == {"horizon"}

    @pytest.mark.parametrize(("kind", "end"), [(HALT, "censored:halt"), (CLEAR, "censored:clear")])
    def test_halt_and_clear_censor_every_open_episode(self, kind: str, end: str) -> None:
        r = TruthReplay("t", ONE_JOIN)
        seeded(r, ev(1, ADD, "A", 100, 7, 1, False), ev(1, ADD, "B", 99, 2, 2))
        seeded(r, ev(2, kind, "N", 0, 0, 0))
        assert sorted(e.end for e in r.episodes) == [end, end]
        assert (r.totals == {}) is (kind == CLEAR)

    def test_an_order_resting_before_the_feed_is_ahead_of_us(self) -> None:
        r = TruthReplay("t", ONE_JOIN)
        r.sync_level("A", 100, 10.0)
        seeded(r, ev(1, ADD, "B", 99, 1, 1))
        v = the_ask(r)
        seeded(r, ev(2, CANCEL, "A", 100, 4, 999))   # unknown id: resting before the file
        assert v.ahead == 6.0 and r.stats.unknown_refs == 1

    def test_iter_batches_counts_batch_steps(self) -> None:
        r = TruthReplay("t", ONE_JOIN)
        seeded(r, ev(1, ADD, "A", 100, 7, 1))
        seeded(r, ev(2, ADD, "A", 100, 1, 2), ev(3, TRADE, "B", 100, 1, 0))
        steps = list(iter_batches(the_ask(r).ep))
        assert [k for k, _, _ in steps] == [STEP_DEPTH, STEP_BATCH, STEP_TRADE, STEP_BATCH]
        assert [b for _, _, b in steps] == [0, 0, 1, 1]


class TestLobster:
    def test_message_types(self) -> None:
        def msg(kind: int, direction: int) -> list[str]:
            return ["34200.1", str(kind), "7", "50", "1000000", str(direction)]

        assert [e.kind for e in lobster_message_events(msg(1, 1), 5)] == [ADD]
        assert lobster_message_events(msg(1, 1), 5)[0].side == "B"
        assert [e.kind for e in lobster_message_events(msg(2, -1), 5)] == [REDUCE]
        assert [e.kind for e in lobster_message_events(msg(3, -1), 5)] == [CANCEL]
        execution = lobster_message_events(msg(4, 1), 5)
        # A visible execution: the direction is the resting order's side, the aggressor the other.
        assert [(e.kind, e.side) for e in execution] == [(TRADE, "A"), (EXEC, "B")]
        assert [e.kind for e in lobster_message_events(msg(5, -1), 5)] == [TRADE]
        assert [e.kind for e in lobster_message_events(msg(7, 1), 5)] == [HALT]
        assert lobster_message_events(msg(6, 1), 5) == []

    def test_time_and_book_row(self) -> None:
        assert _ns("1970-01-02", "1.5") == 86_400_000_000_000 + 1_500_000_000
        assert _ns("1970-01-01", "34200.000000001") == 34_200_000_000_001
        full = ["101", "5", "99", "6", "102", "7", "98", "8"]
        asks, bids, deep_ask, deep_bid = _book_row(full, 2)
        assert (asks, bids, deep_ask, deep_bid) == ({101: 5.0, 102: 7.0}, {99: 6.0, 98: 8.0},
                                                    102, 98)
        thin = ["101", "5", "99", "6", str(LOBSTER_EMPTY_ASK), "0", str(LOBSTER_EMPTY_BID), "0"]
        _, _, deep_ask, deep_bid = _book_row(thin, 2)
        assert (deep_ask, deep_bid) == (LOBSTER_EMPTY_ASK, LOBSTER_EMPTY_BID)

    def test_replay_reconciles_against_the_venue_totals(self, tmp_path: Path) -> None:
        # One visible level per side. 200 shares at 101 rest before the file begins.
        messages = [
            "34200.0,1,1,100,10100,-1",   # add 100 at 101 ask behind the 200 already there
            "34201.0,1,2,50,9900,1",      # a bid; nothing changes at 101
            "34202.0,4,1,100,10100,-1",   # order 1 (behind the synthetic) executed in full
        ]
        books = [
            "10100,300,9800,40",
            "10100,300,9900,50",
            "10100,200,9900,50",
        ]
        msg = tmp_path / "m.csv.gz"
        book = tmp_path / "b.csv.gz"
        with gzip.open(msg, "wt", encoding="ascii") as fh:
            fh.write("\n".join(messages) + "\n")
        with gzip.open(book, "wt", encoding="ascii") as fh:
            fh.write("\n".join(books) + "\n")
        source = LobsterSource("X", "2012-06-21", 1, "", "", "m", "b")
        episodes, meta = replay_lobster(msg, book, source,
                                        ReplayConfig(every_ns=10**15, horizon_ns=10**15))
        assert meta["messages"] == 3 and meta["inconsistent_levels"] == 0
        asks = [e for e in episodes if e.side == "A"]
        assert [e.entry for e in asks] == [300.0]
        # The synthetic ask joined behind 200 pre-existing and 100 of order 1, so order 1 is
        # AHEAD of it: its execution does not fill the synthetic order.
        assert asks[0].end == "end_of_data"


class TestDatabento:
    def test_a_missing_file_is_an_error_not_an_empty_book(self, tmp_path: Path) -> None:
        with pytest.raises(FeedError):
            databento_events(tmp_path / "absent.dbn.zst")

    @pytest.mark.skipif(not ESH4.exists(), reason="nautilus_trader test data not cloned")
    def test_the_real_esh4_session_replays_with_no_inconsistency(self) -> None:
        pytest.importorskip("databento")
        episodes, meta = replay_databento(ESH4, "ESH4 2023-12-25", ReplayConfig())
        assert meta["records"] == 68_792
        assert meta["self_consistency_violations"] == 0
        assert meta["inconsistent_levels"] == 0
        assert meta["unknown_order_refs"] == 0
        assert len(episodes) == meta["joins"] == 1454
        assert sum(e.end == "filled" for e in episodes) == 1133


class TestAFilledOrderKeepsItsFeed:
    """After the truth fills, the L2 view keeps recording to the horizon (2026-09-26): an L2 model
    that fills a few events later than the truth, as hftbacktest's engine does, is not a miss."""

    def test_depth_after_the_fill_still_reaches_the_episode(self) -> None:
        r = TruthReplay("t", ONE_JOIN)
        seeded(r, ev(1, ADD, "A", 100, 7, 1))
        v = the_ask(r)
        seeded(r, ev(2, ADD, "A", 100, 4, 2))
        seeded(r, ev(3, FILL, "A", 100, 7, 1, False), ev(3, CANCEL, "A", 100, 7, 1))
        seeded(r, ev(4, FILL, "A", 100, 1, 2, False), ev(4, MODIFY, "A", 100, 3, 2))
        assert v.filled
        assert v in r.active
        steps_at_fill = len(v.ep.kinds)
        seeded(r, ev(5, ADD, "A", 100, 5, 3))
        assert len(v.ep.kinds) > steps_at_fill
        assert v.ep.truth[-1] == 0.0
        assert v.ep.true_fill_batch < v.ep.batches - 1
