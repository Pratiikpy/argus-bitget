"""The numba-compiled half of `eval/hftbacktest_run.py`: hftbacktest's own L3-to-L2 converter and
the scheduled strategy both engines run.

Kept apart because numba, not Python annotations, types these functions: ``@njit`` and
``@jitclass`` infer every type at compile time, the converter is hftbacktest's notebook code copied
verbatim (MIT), and annotating it to satisfy ``mypy --strict`` would make it no longer the upstream
copy. ``pyproject.toml`` exempts this one module from the strict check for that reason, as it does
the vendored baselines. Nothing here compiles until a builder is called, so importing it needs
neither numba nor hftbacktest.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from argus.eval.hftbacktest_import import import_hftbacktest

# --- convert_l3_to_l2, verbatim from hftbacktest examples/Level-3 Backtesting.ipynb cell 7 (MIT) --


def build_l3_to_l2() -> Any:  # pragma: no cover - requires numba + hftbacktest
    """Compile hftbacktest's own L3->L2 converter. Kept inside a function so importing this module
    never compiles numba code. Only formatting differs from the notebook."""
    hbt = import_hftbacktest()
    from numba import float64, njit, uint64
    from numba.experimental import jitclass
    from numba.typed import Dict
    from numba.types import DictType, int64

    event_dtype = hbt.event_dtype
    depth_event, add_ev, modify_ev, cancel_ev, fill_ev, clear_ev = (
        hbt.DEPTH_EVENT, hbt.ADD_ORDER_EVENT, hbt.MODIFY_ORDER_EVENT, hbt.CANCEL_ORDER_EVENT,
        hbt.FILL_EVENT, hbt.DEPTH_CLEAR_EVENT,
    )
    exch_ev, local_ev, buy_ev, sell_ev = (
        hbt.EXCH_EVENT, hbt.LOCAL_EVENT, hbt.BUY_EVENT, hbt.SELL_EVENT,
    )

    # The notebook declares these fields as class annotations; this module postpones annotations
    # (`from __future__ import annotations`), so numba would read them as strings and fail to
    # resolve the locally imported types. The same fields, passed as jitclass's explicit spec.
    @jitclass([
        ("bid_depth", DictType(int64, float64)),
        ("ask_depth", DictType(int64, float64)),
        ("order_book_px", DictType(uint64, float64)),
        ("order_book_qty", DictType(uint64, float64)),
        ("tick_size", float64),
    ])
    class L3MarketDepth:

        def __init__(self, tick_size):
            self.bid_depth = Dict.empty(int64, float64)
            self.ask_depth = Dict.empty(int64, float64)
            self.order_book_px = Dict.empty(uint64, float64)
            self.order_book_qty = Dict.empty(uint64, float64)
            self.tick_size = tick_size

        def add_order(self, ev):
            if ev.order_id in self.order_book_qty:
                print("add_order: OrderIdExist", ev.order_id)
                raise ValueError
            self.order_book_px[ev.order_id] = ev.px
            l2_ev = np.empty(1, event_dtype)
            l2_ev[0] = ev
            l2_ev[0].ev = (l2_ev[0].ev & ~0xff) | depth_event
            price_tick = int(round(ev.px / self.tick_size))
            if ev.ev & buy_ev == buy_ev:
                self.order_book_qty[ev.order_id] = ev.qty
                if price_tick not in self.bid_depth:
                    self.bid_depth[price_tick] = 0.0
                self.bid_depth[price_tick] += ev.qty
                l2_ev[0].qty = round(self.bid_depth[price_tick])
            elif ev.ev & sell_ev == sell_ev:
                self.order_book_qty[ev.order_id] = -ev.qty
                if price_tick not in self.ask_depth:
                    self.ask_depth[price_tick] = 0.0
                self.ask_depth[price_tick] += ev.qty
                l2_ev[0].qty = round(self.ask_depth[price_tick])
            return l2_ev[0]

        def modify_order(self, ev):
            if ev.order_id not in self.order_book_qty:
                print("modify_order: OrderNotFound", ev.order_id)
                raise ValueError
            prev_px = self.order_book_px[ev.order_id]
            prev_qty = self.order_book_qty[ev.order_id]
            l2_ev = np.empty(2, event_dtype)
            l2_ev[1] = l2_ev[0] = ev
            l2_ev[0].ev = (l2_ev[0].ev & ~0xff) | depth_event
            n = 0
            if prev_qty > 0:
                price_tick = int(round(prev_px / self.tick_size))
                self.bid_depth[price_tick] -= prev_qty
                if int(round(prev_px / self.tick_size)) != int(round(ev.px / self.tick_size)):
                    l2_ev[0].px = prev_px
                    l2_ev[0].qty = round(self.bid_depth[price_tick])
                    n = 1
            elif prev_qty < 0:
                price_tick = int(round(prev_px / self.tick_size))
                self.ask_depth[price_tick] -= np.abs(prev_qty)
                if int(round(prev_px / self.tick_size)) != int(round(ev.px / self.tick_size)):
                    l2_ev[0].px = prev_px
                    l2_ev[0].qty = round(self.ask_depth[price_tick])
                    n = 1
            self.order_book_px[ev.order_id] = ev.px
            price_tick = int(round(ev.px / self.tick_size))
            if ev.ev & buy_ev == buy_ev:
                self.order_book_qty[ev.order_id] = ev.qty
                if price_tick not in self.bid_depth:
                    self.bid_depth[price_tick] = 0.0
                self.bid_depth[price_tick] += ev.qty
                l2_ev[n].qty = round(self.bid_depth[price_tick])
            elif ev.ev & sell_ev == sell_ev:
                self.order_book_qty[ev.order_id] = -ev.qty
                if price_tick not in self.ask_depth:
                    self.ask_depth[price_tick] = 0.0
                self.ask_depth[price_tick] += ev.qty
                l2_ev[n].qty = round(self.ask_depth[price_tick])
            return l2_ev[:n + 1]

        def cancel_order(self, ev):
            if ev.order_id not in self.order_book_qty:
                print("cancel_order: OrderNotFound", ev.order_id, ev)
                raise ValueError
            del self.order_book_px[ev.order_id]
            del self.order_book_qty[ev.order_id]
            l2_ev = np.empty(1, event_dtype)
            l2_ev[0] = ev
            l2_ev[0].ev = (l2_ev[0].ev & ~0xff) | depth_event
            if ev.ev & buy_ev == buy_ev:
                price_tick = int(round(ev.px / self.tick_size))
                self.bid_depth[price_tick] -= ev.qty
                l2_ev[0].qty = round(self.bid_depth[price_tick])
            elif ev.ev & sell_ev == sell_ev:
                price_tick = int(round(ev.px / self.tick_size))
                self.ask_depth[price_tick] -= ev.qty
                l2_ev[0].qty = round(self.ask_depth[price_tick])
            return l2_ev[0]

        def clear(self):
            self.order_book_px.clear()
            self.order_book_qty.clear()
            self.bid_depth.clear()
            self.ask_depth.clear()

    @njit
    def convert_l3_to_l2(data, tick_size):
        result = np.empty(len(data) * 4, event_dtype)
        exch_md = L3MarketDepth(tick_size)
        rn = 0
        for i in range(len(data)):
            if data[i].ev & (exch_ev | local_ev) == exch_ev | local_ev:
                if data[i].ev & 0xff == add_ev:
                    result[rn] = exch_md.add_order(data[i])
                    rn += 1
                elif data[i].ev & 0xff == modify_ev:
                    l2_ev = exch_md.modify_order(data[i])
                    result[rn] = l2_ev[0]
                    rn += 1
                    if len(l2_ev) == 2:
                        result[rn] = l2_ev[1]
                        rn += 1
                elif data[i].ev & 0xff == cancel_ev:
                    result[rn] = exch_md.cancel_order(data[i])
                    rn += 1
                elif data[i].ev & 0xff == fill_ev:
                    continue
                elif data[i].ev & 0xff == clear_ev:
                    exch_md.clear()
                    result[rn] = data[i]
                    rn += 1
                else:
                    result[rn] = data[i]
                    rn += 1
            else:
                raise ValueError
        return result[:rn]

    return convert_l3_to_l2


# --- the scheduled strategy both engines run ----------------------------------------------------


def build_schedule_runner() -> Any:  # pragma: no cover - requires numba + hftbacktest
    """A numba strategy that places the given limit orders at the given times, cancels each one
    ``horizon`` later if still resting, and never reads the book — so hftbacktest and ARGUS hold
    byte-identical orders and only their queue logic can differ."""
    hbt_mod = import_hftbacktest()
    from numba import njit

    gtc, limit = hbt_mod.GTC, hbt_mod.LIMIT

    @njit
    def run(hbt, times, sides, prices, horizon):
        n = len(times)
        k = 0
        c = 0
        big = 9223372036854775807
        # The clock reads INT64_MAX until the first elapse loads the feed; without this the first
        # elapse(t - now) underflows and no order ever reaches the exchange (found by running it).
        hbt.elapse(1)
        while k < n or c < n:
            t_sub = times[k] if k < n else big
            t_can = times[c] + horizon if c < n else big
            t_next = min(t_sub, t_can)
            now = hbt.current_timestamp
            if t_next > now:
                if hbt.elapse(t_next - now) != 0:
                    break
            if t_sub <= t_can:
                if sides[k] == 1:
                    hbt.submit_buy_order(0, k + 1, prices[k], 1.0, gtc, limit, False)
                else:
                    hbt.submit_sell_order(0, k + 1, prices[k], 1.0, gtc, limit, False)
                k += 1
            else:
                order = hbt.orders(0).get(c + 1)
                if order is not None and order.cancellable:
                    hbt.cancel(0, c + 1, False)
                c += 1
        hbt.elapse(1_000_000_000)
        out = np.full((n, 3), -1, np.int64)
        for i in range(n):
            order = hbt.orders(0).get(i + 1)
            if order is not None:
                out[i, 0] = order.status
                out[i, 1] = order.exch_timestamp
                out[i, 2] = order.local_timestamp
        return out

    return run
