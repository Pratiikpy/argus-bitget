"""Queue-position truth from real market-by-order feeds, replayed the way hftbacktest replays it.

`eval/queueproof.py` scores the ported queue models against a simulator whose attribution rule we
chose. `eval/mbo_queueproof.py` was the first attempt at real ground truth, and it carried two
replay defects that this module exists to correct — found by reading a real DataBento record
sequence next to hftbacktest's own L3 engine rather than by trusting the first result:

1. **Trades were counted twice.** A CME match reaches a DataBento MBO file as one ``T`` record
   (the aggressor's print) followed by one ``F`` record per resting order it hit, and *then* the
   book change as ``C`` (fully filled) or ``M`` (partially filled, smaller size). The first replay
   consumed the queue on ``T`` and again on ``F``, then processed the ``C``/``M`` on top.
   hftbacktest never touches the book on either: its L3 exchange ignores ``TRADE_EVENT`` and turns
   ``FILL_EVENT`` into a fill *check* only (``hftbacktest/src/backtest/proc/
   l3_nopartialfillexchange.rs:392-405``); the size change arrives with the following ``C``/``M``.
2. **A partially filled order was sent to the back of its own queue.** After the double
   consumption, the venue's ``M`` (new, smaller size) looked like a size *increase* relative to the
   over-consumed replay, so it was replayed as cancel-then-add — moving an order that was at the
   front of the queue to behind the synthetic one.

Both are visible on the first trade of the 2023-12-25 ESH4 session (records at ``ts_recv``
1703545200107250074-1703545200107482899: ``T B 5``, ``F A 5``, ``M A 2``, then ``T B 2``, ``F A 2``,
``C A 2`` for order 6412777265270).

**The truth rule is hftbacktest's, not ours.** ``L3FIFOQueueModel`` (``queue.rs:481-1050``) keeps
backtest orders in the venue's FIFO without letting them consume real liquidity, and fills one
when *a market-feed order behind it* is filled (``fill_market_feed_order``, ``queue.rs:974-1070``)
— the zero-market-impact convention every L3 backtester uses, because a real order resting
there would have been hit instead. It also fills everything priced through an executed order and
everything the opposite side crosses. That is what :class:`TruthReplay` implements, event for
event, and what ``tests/test_l3queue.py`` pins against hand-worked sequences.

**The L2 view is also hftbacktest's.** Their own L2-versus-L3 study (``examples/Level-3
Backtesting.ipynb``, cell 7, ``convert_l3_to_l2``) derives the L2 feed from the same L3 file:
every Add/Cancel/Modify becomes a depth event carrying the level's new total, every Fill is
dropped, every Trade passes through. That is the ``record`` view here. The ``event`` view collapses
the depth updates of one exchange event into one — what a market-by-price feed publishes — and is
reported alongside, because the two disagree about how often a trade is counted.

Venue adapters live in :mod:`argus.eval.l3feeds`; the scoring and the rival comparison in
:mod:`argus.eval.realqueue`.
"""

from __future__ import annotations

import contextlib
import heapq
from array import array
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

ADD, CANCEL, MODIFY, REDUCE, FILL, EXEC, TRADE, CLEAR, HALT = (
    "A", "C", "M", "D", "F", "E", "T", "R", "H",
)
"""Normalised event kinds.

``A`` add · ``C`` delete (whole order) · ``M`` modify to a new price/size (hftbacktest semantics:
a price change or a size increase loses priority, a decrease keeps it) · ``D`` reduce by ``size``
keeping priority (ITCH/LOBSTER partial cancel) · ``F`` a resting order was filled, book unchanged
(DataBento: the ``C``/``M`` follows) · ``E`` a resting order was filled *and* reduced in one message
(ITCH/LOBSTER execution) · ``T`` a public trade print, ``side`` = aggressor · ``R`` clear the book ·
``H`` trading halt (censors every open episode, keeps the book).
"""

STEP_TRADE, STEP_DEPTH, STEP_THROUGH, STEP_BATCH = 0, 1, 2, 3
"""What an L2 consumer sees at the synthetic order's price, in order.

``TRADE``   a print at our price by an aggressor on the other side (value = size)
``DEPTH``   the level's new total, other participants only (value = total)
``THROUGH`` a print or a quote *through* our price — every L2 exchange model fills here
``BATCH``   the end of one exchange event: the truth is sampled here, never mid-event
"""


@dataclass(frozen=True, slots=True)
class L3Event:
    """One normalised market-by-order record.

    ``px`` is the venue's own integer price (DataBento 1e-9 units, LOBSTER dollars x 10,000,
    Bitstamp dollars x 100) so that equality between levels is exact.
    """

    ts: int
    kind: str
    side: str
    px: int
    size: float
    order_id: int
    batch_end: bool = True


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """How synthetic orders are placed and how long they are followed.

    Placement is *time-driven at the touch*: every ``every_ns`` one synthetic bid joins the back of
    the best bid and one synthetic ask the back of the best ask — the schedule a quoting strategy
    runs, rather than one keyed to the feed's own activity (which would over-sample bursts).
    """

    every_ns: int = 5_000_000_000
    horizon_ns: int = 120_000_000_000
    max_steps: int = 4000
    min_level: float = 0.0
    """Only join a level holding more than this (other participants' quantity)."""
    start_ns: int = 0
    """No synthetic order joins before this timestamp (e.g. after a snapshot or a warm-up)."""
    end_ns: int = 1 << 62


@dataclass(slots=True)
class RealEpisode:
    """One synthetic order's life, as an L2 consumer would have seen it, with the L3 truth."""

    dataset: str
    side: str
    px: int
    t_join: int
    entry: float
    kinds: bytearray = field(default_factory=bytearray)
    vals: array[float] = field(default_factory=lambda: array("d"))
    truth: array[float] = field(default_factory=lambda: array("d"))
    """True quantity ahead at each BATCH step, in order (0.0 once filled)."""
    batch_ts: array[int] = field(default_factory=lambda: array("q"))
    """Timestamp of each BATCH step."""
    true_fill_batch: int = -1
    """Index of the BATCH step at which the truth filled; -1 if it never did."""
    fill_ts: int = -1
    """Timestamp of the event that filled the synthetic order in truth; -1 if none."""
    end: str = "open"
    """``filled`` · ``horizon`` · ``censored:<reason>`` · ``end_of_data``."""

    @property
    def batches(self) -> int:
        return len(self.truth)

    def as_row(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset, "side": self.side, "px": self.px, "t_join": self.t_join,
            "entry": self.entry, "steps": len(self.kinds), "batches": self.batches,
            "true_fill_batch": self.true_fill_batch, "fill_ts": self.fill_ts, "end": self.end,
        }


class _Virtual:
    __slots__ = ("ahead", "behind", "ep", "filled", "seq", "steps", "touched")

    def __init__(self, ep: RealEpisode, seq: int, ahead: float) -> None:
        self.ep = ep
        self.seq = seq
        self.ahead = ahead
        self.behind = 0.0
        self.touched = False
        self.filled = False
        self.steps = 0

    def step(self, kind: int, value: float = 0.0) -> None:
        self.ep.kinds.append(kind)
        self.ep.vals.append(value)
        self.touched = True
        self.steps += 1


UNKNOWN_SEQ = -1
"""Queue seniority of an order that was resting before the feed began. It is ahead of every
synthetic order, because a synthetic order only ever joins after the feed began."""


@dataclass(slots=True)
class ReplayStats:
    events: int = 0
    unknown_refs: int = 0
    crossings: int = 0
    joins: int = 0
    inconsistent: int = 0
    by_end: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "events": self.events, "unknown_order_refs": self.unknown_refs,
            "crossing_fills": self.crossings, "joins": self.joins,
            "inconsistent_levels": self.inconsistent, "episode_ends": dict(self.by_end),
        }


class TruthReplay:
    """A market-by-order book with synthetic orders that never consume real liquidity.

    Each real order carries its queue seniority ``seq``. A synthetic order records the ``seq`` it
    joined at and the quantity resting ahead of it; everything else follows from comparing
    seniorities, so no per-level list is ever walked.

    ``authoritative`` (optional) is how a venue that publishes its own level totals alongside the
    order stream (LOBSTER's orderbook file) keeps the replay honest: orders resting before the file
    began are invisible to an order-level replay, so their quantity is carried as a per-level
    residual, and the residual is re-derived from the venue's totals after every message. A
    synthetic order whose ``ahead + known-behind`` stops matching the venue's own total is censored
    rather than scored — the check that makes a reconstruction error visible instead of silent.
    """

    def __init__(self, dataset: str, config: ReplayConfig | None = None) -> None:
        self.dataset = dataset
        self.cfg = config or ReplayConfig()
        self.orders: dict[int, list[Any]] = {}   # id -> [side, px, size, seq]
        self.totals: dict[tuple[str, int], float] = {}
        self._bid_heap: list[int] = []   # negated prices; lazily cleaned
        self._ask_heap: list[int] = []
        self.virtuals: dict[tuple[str, int], list[_Virtual]] = {}
        self.active: list[_Virtual] = []
        self.episodes: list[RealEpisode] = []
        self.seq = 0
        self.next_join = self.cfg.start_ns
        self.stats = ReplayStats()
        self.best_override: tuple[int | None, int | None] | None = None
        self.attribution: list[tuple[float, float, float, bool, float]] | None = None
        """When a list, every *cancellation* (not a fill) at a level holding a synthetic order is
        logged as ``(front, back, qty, from_back, seconds_since_join)`` — the quantity the
        probability functions exist to predict, observed instead of assumed."""
        self._filled_now: set[int] = set()
        self._cause_fill = False
        self._now = 0

    # --- book primitives -------------------------------------------------------------------

    def _level_add(self, side: str, px: int, qty: float) -> float:
        key = (side, px)
        before = self.totals.get(key, 0.0)
        total = before + qty
        if total <= 1e-12:
            self.totals.pop(key, None)
            return 0.0
        self.totals[key] = total
        if before <= 1e-12:
            self._push_level(side, px)
        return total

    def _push_level(self, side: str, px: int) -> None:
        if side == "B":
            heapq.heappush(self._bid_heap, -px)
        elif side == "A":
            heapq.heappush(self._ask_heap, px)

    def _market_best(self, side: str) -> int | None:
        """Best price among *real* orders (synthetic ones are never in ``totals``) — what
        hftbacktest's ``depth.best_bid_tick()`` reads inside its L3 exchange."""
        heap = self._bid_heap if side == "B" else self._ask_heap
        while heap:
            px = -heap[0] if side == "B" else heap[0]
            if self.totals.get((side, px), 0.0) > 1e-12:
                return px
            heapq.heappop(heap)
        return None

    def _depth_step(self, side: str, px: int) -> None:
        vs = self.virtuals.get((side, px))
        if vs:
            total = self.totals.get((side, px), 0.0)
            for v in vs:
                # Recorded after the truth has filled too: an L2 model has not seen that fill,
                # and hftbacktest's engine keeps reading the feed until the order's horizon.
                v.step(STEP_DEPTH, total)

    def _level_delta(self, side: str, px: int, seq: int, delta: float) -> None:
        """An order of seniority ``seq`` at this level changed by ``delta`` (negative = left). It is
        ahead of every synthetic order that joined after it and behind every one that joined
        before, so this one comparison is the whole FIFO."""
        vs = self.virtuals.get((side, px))
        if vs:
            log = self.attribution if (delta < 0 and not self._cause_fill) else None
            for v in vs:
                if log is not None and not v.filled:
                    log.append((v.ahead, v.behind, -delta, seq > v.seq,
                                (self._now - v.ep.t_join) / 1e9))
                if seq < v.seq:
                    v.ahead = max(v.ahead + delta, 0.0)
                else:
                    v.behind = max(v.behind + delta, 0.0)

    def _fill(self, v: _Virtual, *, through: bool) -> None:
        if v.filled:
            return
        v.filled = True
        v.ahead = 0.0
        v.ep.fill_ts = self._now
        if through:
            v.step(STEP_THROUGH)
        v.touched = True

    def _check_cross(self, side: str, px: int) -> None:
        """A quote on ``side`` at ``px`` crosses synthetic orders resting on the other side —
        hftbacktest's ``fill_*_orders_by_crossing``. Truth and every L2 model fill here alike."""
        for v in self.active:
            if v.ep.side == side:
                continue
            if (v.ep.side == "B" and px <= v.ep.px) or (v.ep.side == "A" and px >= v.ep.px):
                if v.filled:
                    # The truth filled earlier; the L2 view still sees its level crossed.
                    v.step(STEP_THROUGH)
                    v.touched = True
                    continue
                self.stats.crossings += 1
                self._fill(v, through=True)

    # --- event handlers ----------------------------------------------------------------------

    def _add(self, side: str, px: int, size: float, oid: int) -> None:
        self._cause_fill = False
        if oid in self.orders:
            # A duplicate id is a feed defect; the second add replaces the first rather than
            # double-counting a queue that exists once.
            self._cancel(oid, side, px, 0.0)
        self.seq += 1
        self.orders[oid] = [side, px, size, self.seq]
        self._level_add(side, px, size)
        self._level_delta(side, px, self.seq, size)
        self._depth_step(side, px)
        self._check_cross(side, px)

    def _cancel(self, oid: int, side: str, px: int, size: float) -> None:
        self._cause_fill = oid in self._filled_now
        order = self.orders.pop(oid, None)
        if order is None:
            # Resting before the feed began: seniority UNKNOWN_SEQ, size as the message states.
            self.stats.unknown_refs += 1
            if side in ("B", "A") and size > 0:
                self._level_add(side, px, -size)
                self._level_delta(side, px, UNKNOWN_SEQ, -size)
                self._depth_step(side, px)
            return
        o_side, o_px, o_size, o_seq = order
        self._level_add(o_side, o_px, -o_size)
        self._level_delta(o_side, o_px, o_seq, -o_size)
        self._depth_step(o_side, o_px)

    def _reduce(self, oid: int, side: str, px: int, by: float) -> None:
        self._cause_fill = oid in self._filled_now
        order = self.orders.get(oid)
        if order is None:
            self.stats.unknown_refs += 1
            if side in ("B", "A") and by > 0:
                self._level_add(side, px, -by)
                self._level_delta(side, px, UNKNOWN_SEQ, -by)
                self._depth_step(side, px)
            return
        o_side, o_px, o_size, o_seq = order
        by = min(by, o_size)
        order[2] = o_size - by
        self._level_add(o_side, o_px, -by)
        self._level_delta(o_side, o_px, o_seq, -by)
        if order[2] <= 1e-12:
            del self.orders[oid]
        self._depth_step(o_side, o_px)

    def _modify(self, oid: int, side: str, px: int, size: float) -> None:
        self._cause_fill = oid in self._filled_now
        order = self.orders.get(oid)
        if order is None:
            # hftbacktest raises OrderNotFound here; a modify of an unseen order is treated as an
            # add at the back, which is where the venue puts a priority-losing change.
            self.stats.unknown_refs += 1
            self._add(side, px, size, oid)
            return
        o_side, o_px, o_size, o_seq = order
        if px != o_px or size > o_size:
            # queue.rs:877-972 and l3_nopartialfillexchange: a price change or a size increase
            # loses priority — removed from its place, re-queued at the back.
            self._level_add(o_side, o_px, -o_size)
            self._level_delta(o_side, o_px, o_seq, -o_size)
            self._depth_step(o_side, o_px)
            self.seq += 1
            self.orders[oid] = [o_side, px, size, self.seq]
            self._level_add(o_side, px, size)
            self._level_delta(o_side, px, self.seq, size)
            self._depth_step(o_side, px)
            self._check_cross(o_side, px)
            return
        by = o_size - size
        order[2] = size
        if by > 0:
            self._level_add(o_side, o_px, -by)
            self._level_delta(o_side, o_px, o_seq, -by)
        if size <= 1e-12:
            del self.orders[oid]
        self._depth_step(o_side, o_px)

    def _fill_check(self, oid: int, side: str, px: int) -> None:
        """A market-feed order was executed at ``px``. Mirrors hftbacktest
        ``L3FIFOQueueModel::fill_market_feed_order`` (``queue.rs:974-1070``) rule for rule, because
        the truth being scored against has to be the rival's own truth:

        1. **Through.** If ``px`` is strictly worse than the market's best on that side, every
           synthetic order priced between the two (``px`` exclusive, best inclusive) is filled —
           ``fill_bid_between(best_bid_tick, exec_price_tick + 1)``.
        2. **Queue.** In the executed order's *own* queue (its resting price, which for a DataBento
           aggressor-side ``F`` is not the execution price), every synthetic order queued ahead
           of it is filled.

        Rule 2 is applied to the resting price even when the ``F`` belongs to the aggressor of the
        match (CME reports both sides when the aggressor was itself a resting order whose price
        was modified through the market). That is hftbacktest's behaviour as read in
        ``queue.rs``, mirrored rather than corrected. An earlier note here reported a run of
        hftbacktest's own engine against this method on the ESH4 file; that run left no artefact
        and hftbacktest is not installed where this was reviewed, so the engine-level agreement is
        NOT VERIFIED (`eval/hftbacktest_run.py` is the harness that would measure it).
        """
        self._filled_now.add(oid)
        order = self.orders.get(oid)
        if order is not None:
            o_side, o_px, _size, o_seq = order
        else:
            o_side, o_px, o_seq = side, px, UNKNOWN_SEQ
        if o_side not in ("B", "A"):
            return
        best = self.best(o_side)
        for v in self.active:
            if v.filled or v.ep.side != o_side:
                continue
            vpx = v.ep.px
            through = best is not None and (
                (o_side == "B" and px < best and px < vpx <= best)
                or (o_side == "A" and px > best and best <= vpx < px)
            )
            behind_us = vpx == o_px and o_seq > v.seq   # the executed order was queued behind us
            if through or behind_us:
                self._fill(v, through=False)

    def _trade(self, aggressor: str, px: int, size: float) -> None:
        """A public print. Only the L2 view sees it; the book does not change on a print."""
        if aggressor not in ("B", "A"):
            return   # hftbacktest ignores a trade with no side (auction prints)
        resting = "A" if aggressor == "B" else "B"
        for v in self.active:
            if v.ep.side != resting:
                continue
            if v.ep.px == px:
                v.step(STEP_TRADE, size)
            elif (resting == "B" and px < v.ep.px) or (resting == "A" and px > v.ep.px):
                v.step(STEP_THROUGH)

    # --- the loop ------------------------------------------------------------------------------

    def best(self, side: str) -> int | None:
        if self.best_override is not None:
            return self.best_override[0] if side == "B" else self.best_override[1]
        return self._market_best(side)

    def _join(self, ts: int) -> None:
        bid, ask = self.best("B"), self.best("A")
        if bid is not None and ask is not None and bid >= ask:
            return   # crossed or locked book (pre-open, auction): nobody queues here
        for side, px in (("B", bid), ("A", ask)):
            if px is None:
                continue
            level = self.totals.get((side, px), 0.0)
            if level <= self.cfg.min_level:
                continue
            ep = RealEpisode(dataset=self.dataset, side=side, px=px, t_join=ts, entry=level)
            # Strictly senior to nothing already resting and junior to everything that is: every
            # existing order has seq <= the counter, every later one will have seq > it.
            self.seq += 1
            v = _Virtual(ep, self.seq, level)
            self.virtuals.setdefault((side, px), []).append(v)
            self.active.append(v)
            self.stats.joins += 1

    def _close(self, v: _Virtual, end: str) -> None:
        v.ep.end = end
        self.stats.by_end[end.split(":")[0]] = self.stats.by_end.get(end.split(":")[0], 0) + 1
        vs = self.virtuals.get((v.ep.side, v.ep.px))
        if vs is not None:
            with contextlib.suppress(ValueError):
                vs.remove(v)
            if not vs:
                del self.virtuals[(v.ep.side, v.ep.px)]
        self.episodes.append(v.ep)

    def end_batch(self, ts: int, check: Callable[[_Virtual], str | None] | None) -> None:
        self._filled_now.clear()
        still: list[_Virtual] = []
        for v in self.active:
            reason = check(v) if (check is not None and not v.filled) else None
            if v.touched:
                v.step(STEP_BATCH)
                v.ep.truth.append(0.0 if v.filled else v.ahead)
                v.ep.batch_ts.append(ts)
                v.touched = False
                if v.filled and v.ep.true_fill_batch < 0:
                    v.ep.true_fill_batch = len(v.ep.truth) - 1
            expired = (ts - v.ep.t_join >= self.cfg.horizon_ns
                       or v.steps >= self.cfg.max_steps)
            if v.filled:
                # Kept open to the order's horizon after the truth fills, so every L2 model reads
                # the same feed hftbacktest's engine reads before it cancels the order. Closing
                # here scored a model that fills a few events after the truth as a missed fill:
                # 178 of 1,438 orders on the ESH4 file for hftbacktest's default model
                # (data/hftbacktest_run.json, 2026-09-26).
                if expired:
                    self._close(v, "filled")
                else:
                    still.append(v)
            elif reason is not None:
                self.stats.inconsistent += reason.startswith("inconsistent")
                self._close(v, f"censored:{reason}")
            elif expired:
                self._close(v, "horizon")
            else:
                still.append(v)
        self.active = still
        if self.cfg.start_ns <= ts < self.cfg.end_ns and ts >= self.next_join:
            self._join(ts)
            # A fixed grid, not "every_ns after the last join": a quiet spell skips grid points
            # rather than shifting every later one.
            skipped = (ts - self.next_join) // self.cfg.every_ns + 1
            self.next_join += skipped * self.cfg.every_ns

    def apply(self, ev: L3Event) -> None:
        self.stats.events += 1
        self._now = ev.ts
        k = ev.kind
        if k == ADD:
            self._add(ev.side, ev.px, ev.size, ev.order_id)
        elif k == CANCEL:
            self._cancel(ev.order_id, ev.side, ev.px, ev.size)
        elif k == MODIFY:
            self._modify(ev.order_id, ev.side, ev.px, ev.size)
        elif k == REDUCE:
            self._reduce(ev.order_id, ev.side, ev.px, ev.size)
        elif k == FILL:
            self._fill_check(ev.order_id, ev.side, ev.px)
        elif k == EXEC:
            self._fill_check(ev.order_id, ev.side, ev.px)
            self._reduce(ev.order_id, ev.side, ev.px, ev.size)
        elif k == TRADE:
            self._trade(ev.side, ev.px, ev.size)
        elif k == CLEAR:
            for v in list(self.active):
                self._close(v, "censored:clear")
            self.active = []
            self.orders.clear()
            self.totals.clear()
            self._bid_heap.clear()
            self._ask_heap.clear()
        elif k == HALT:
            for v in list(self.active):
                self._close(v, "censored:halt")
            self.active = []

    def run(
        self, events: Iterable[L3Event],
        check: Callable[[_Virtual], str | None] | None = None,
    ) -> list[RealEpisode]:
        for ev in events:
            self.apply(ev)
            if ev.batch_end:
                self.end_batch(ev.ts, check)
        return self.finish()

    def finish(self) -> list[RealEpisode]:
        for v in self.active:
            self._close(v, "filled" if v.filled else "end_of_data")
        self.active = []
        return self.episodes

    def sync_level(self, side: str, px: int, total: float) -> None:
        """Adopt a venue-published level total: the gap to the known orders is quantity that was
        resting before the feed began, and it is senior to every synthetic order."""
        if total <= 1e-12:
            self.totals.pop((side, px), None)
        else:
            if self.totals.get((side, px), 0.0) <= 1e-12:
                self._push_level(side, px)
            self.totals[(side, px)] = total

    def self_consistent(self, v: _Virtual) -> bool:
        """``ahead + behind`` must equal the level's total — the replay's own invariant."""
        total = self.totals.get((v.ep.side, v.ep.px), 0.0)
        return abs(total - (v.ahead + v.behind)) <= 1e-6 * max(1.0, total)


def iter_batches(ep: RealEpisode) -> Iterator[tuple[int, float, int]]:
    """``(kind, value, batch_index)`` for each step, the batch index counting BATCH steps."""
    b = 0
    for kind, value in zip(ep.kinds, ep.vals, strict=True):
        yield kind, value, b
        if kind == STEP_BATCH:
            b += 1


__all__ = [
    "ADD",
    "CANCEL",
    "CLEAR",
    "EXEC",
    "FILL",
    "HALT",
    "MODIFY",
    "REDUCE",
    "STEP_BATCH",
    "STEP_DEPTH",
    "STEP_THROUGH",
    "STEP_TRADE",
    "TRADE",
    "UNKNOWN_SEQ",
    "L3Event",
    "RealEpisode",
    "ReplayConfig",
    "ReplayStats",
    "TruthReplay",
    "iter_batches",
]
