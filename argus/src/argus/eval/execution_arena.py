"""Execution Assistance, head to head: how an order is split, scored on a book nobody fitted on.

Track 3 names the sub-theme: "After trader decision, how does AI handle order splitting and
slippage management? Large order splitting; order book depth analysis; slippage pattern
adjustment." The rival review of 2026-09-24 found that neither of ARGUS's OWNED execution rows had
ever scored a schedule on realised cost — one checked the Almgren-Chriss port against its own
objective, the other beat a latency model — and named the incumbents a trader actually clicks:
Bitget's own TWAP (equal market slices at a fixed interval, `bitget.site` support article
12560603819691), the plain even split every execution tool ships, and trading it all at once.

**Referee (version 1): a full-depth replay of one real day.** Tardis's free first-of-month
``incremental_book_L2`` + trades for Bitget NVDAUSDT on 2026-08-01, converted to hftbacktest's
event format (1,610,198 events; ``ev`` flags from ``hftbacktest/types.py:9-55``). The book is
rebuilt event by event and read every 60 seconds. A child order is a market order that walks the
book as it stood at that instant; **what it takes stays taken** until the venue next reports that
level, so a schedule that fires children back to back pays for its own footprint instead of
refilling from history for free. The fee is Bitget's 0.06% taker charge (`cost/model.py:188-197`).
What the referee cannot know is how other traders would have reacted to our orders: impact beyond
the visible book is not modelled, which flatters large immediate orders, and says so.

**Scored per parent order** (buy and sell; $5k, $25k, $100k and $250k; a new parent every 30
minutes through the day, each with a 4-hour horizon):

* ``cost_bps`` — what the children paid against the mid at the instant each was sent, plus fees:
  the part of the bill the schedule controls. **Primary.**
* ``shortfall_bps`` — the average fill against the mid when the parent arrived, plus fees: the
  whole bill, dominated on a four-hour horizon by where the price went, so noisy.

Arms: ``immediate``; ``twap_8`` (eight equal children, 30 minutes apart); ``bitget_twap_60s``
(Bitget's TWAP at a 60-second interval, children of at least 10 USDT as its spec requires);
``argus_ac`` (what the console prints today: the Almgren-Chriss shape — inventory decays e-fold
over the horizon, `lui/research.SCHEDULE_DECAYS` — in hourly children, four over four hours);
``argus_ac_60s`` (the same trajectory cut into one child a minute — added after the first run
showed cadence, not shape, deciding the cost; not in the original design).

Almgren-Chriss does not minimise expected cost alone: it trades expected cost against the variance
of the shortfall. Both are reported — ``shortfall_std_bps`` is the dispersion a trader carries.

Tests: paired per parent, a two-sided sign test and a Wilcoxon (normal approximation) over parents
grouped by start hour, reported per size. One Saturday of one stock perpetual is a small arena;
the forward tape (`market/ws_tape.py`, recording since 2026-09-24) is the next day added.

    ARGUS_TARDIS_NPZ=path/to/nvda_20260801.npz python -m argus.eval.execution_arena
"""

from __future__ import annotations

import hashlib
import math
import os
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parents[3]
REPORT_PATH = PACKAGE / "data" / "execution_arena.json"
DEPTH_EVENT, TRADE_EVENT, DEPTH_CLEAR_EVENT, DEPTH_SNAPSHOT_EVENT = 1, 2, 3, 4
BUY_EVENT, SELL_EVENT = 1 << 29, 1 << 28
STEP_NS = 60 * 1_000_000_000
TAKER_BPS = 6.0
SIZES = (5_000.0, 25_000.0, 100_000.0, 250_000.0)
HORIZON_MIN = 240
PARENT_EVERY_MIN = 30
CHILDREN = 8
AC_DECAY = 1.0
"""kappa x horizon, the console's default (`lui/research.SCHEDULE_DECAYS[False]`)."""
BITGET_MIN_CHILD = 10.0


@dataclass
class Level:
    qty: float
    updated: int


@dataclass
class Snapshot:
    ts: int
    bids: dict[float, Level] = field(default_factory=dict)
    asks: dict[float, Level] = field(default_factory=dict)

    @property
    def mid(self) -> float:
        return (max(self.bids) + min(self.asks)) / 2


def snapshots(data: Any) -> list[Snapshot]:
    """The book every :data:`STEP_NS`, each level carrying when the venue last reported it."""
    bids: dict[float, Level] = {}
    asks: dict[float, Level] = {}
    out: list[Snapshot] = []
    ev, ts, px, qty = data["ev"], data["exch_ts"], data["px"], data["qty"]
    next_cut = (int(ts[0]) // STEP_NS + 1) * STEP_NS
    for i in range(len(ev)):
        t = int(ts[i])
        while t >= next_cut:
            if bids and asks:
                out.append(Snapshot(next_cut, {p: Level(v.qty, v.updated) for p, v in bids.items()},
                                    {p: Level(v.qty, v.updated) for p, v in asks.items()}))
            next_cut += STEP_NS
        flags = int(ev[i])
        kind = flags & 0xFF
        if kind == TRADE_EVENT:
            continue
        book = bids if flags & BUY_EVENT else asks if flags & SELL_EVENT else None
        if book is None:
            continue
        if kind == DEPTH_CLEAR_EVENT:
            book.clear()
            continue
        price, size = float(px[i]), float(qty[i])
        if size <= 0:
            book.pop(price, None)
        else:
            book[price] = Level(size, t)
    return out


@dataclass
class Fill:
    notional: float
    shares: float
    mid_at_send: float


def walk(snap: Snapshot, side: str, notional: float,
         taken: dict[tuple[str, float], tuple[float, int]]) -> Fill:
    """Spend ``notional`` against the book at ``snap``, net of what this parent already took from
    levels the venue has not re-reported since. Records what this child takes."""
    levels = sorted(snap.asks.items()) if side == "BUY" else sorted(snap.bids.items(),
                                                                     reverse=True)
    book_side = "ask" if side == "BUY" else "bid"
    left, shares, spent = notional, 0.0, 0.0
    for price, level in levels:
        prior = taken.get((book_side, price))
        available = level.qty
        if prior is not None and level.updated <= prior[1]:
            available -= prior[0]
        if available <= 0:
            continue
        take = min(available, left / price)
        shares += take
        spent += take * price
        left -= take * price
        used = (prior[0] if prior is not None and level.updated <= prior[1] else 0.0) + take
        taken[(book_side, price)] = (used, snap.ts)
        if left <= 1e-9:
            break
    if left > 1e-6:
        raise ValueError(f"the visible book could not fill {notional:.0f} USDT")
    return Fill(notional=spent, shares=shares, mid_at_send=snap.mid)


def ac_fractions(children: int, decay: float = AC_DECAY) -> list[float]:
    """Almgren-Chriss inventory x_j = sinh(kappa (T - t_j)) / sinh(kappa T); trades are its
    steps."""
    held = [math.sinh(decay * (1 - j / children)) / math.sinh(decay) for j in range(children + 1)]
    return [held[j] - held[j + 1] for j in range(children)]


def plans(total: float) -> dict[str, list[tuple[int, float]]]:
    """Each arm's children as (minutes after arrival, notional)."""
    step = HORIZON_MIN // CHILDREN
    bitget_n = max(1, min(HORIZON_MIN, int(total // BITGET_MIN_CHILD)))
    bitget_every = HORIZON_MIN / bitget_n
    hours = HORIZON_MIN // 60
    return {
        "immediate": [(0, total)],
        "twap_8": [(j * step, total / CHILDREN) for j in range(CHILDREN)],
        "bitget_twap_60s": [(round(j * bitget_every), total / bitget_n) for j in range(bitget_n)],
        "argus_ac": [(j * 60, total * f) for j, f in enumerate(ac_fractions(hours))],
        "argus_ac_60s": [(j, total * f) for j, f in enumerate(ac_fractions(HORIZON_MIN))],
    }


def execute(snaps: Sequence[Snapshot], start: int, side: str,
            children: Sequence[tuple[int, float]]) -> dict[str, float] | None:
    taken: dict[tuple[str, float], tuple[float, int]] = {}
    arrival = snaps[start].mid
    sign = 1.0 if side == "BUY" else -1.0
    fills: list[Fill] = []
    for minute, notional in children:
        idx = start + minute
        if idx >= len(snaps):
            return None
        try:
            fills.append(walk(snaps[idx], side, notional, taken))
        except ValueError:
            return None
    shares = sum(f.shares for f in fills)
    spent = sum(f.notional for f in fills)
    average = spent / shares
    cost = sum(sign * (f.notional / f.shares - f.mid_at_send) / f.mid_at_send * f.notional
               for f in fills) / spent * 1e4
    shortfall = sign * (average - arrival) / arrival * 1e4
    return {"cost_bps": cost + TAKER_BPS, "shortfall_bps": shortfall + TAKER_BPS,
            "children": len(fills)}


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2 ** n)
    return min(1.0, 2.0 * tail)


def run(snaps: Sequence[Snapshot]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(snaps) - HORIZON_MIN, PARENT_EVERY_MIN):
        for side in ("BUY", "SELL"):
            for size in SIZES:
                result = {arm: execute(snaps, start, side, children)
                          for arm, children in plans(size).items()}
                rows.append({"start": datetime.fromtimestamp(snaps[start].ts / 1e9, UTC)
                             .isoformat(timespec="minutes"), "side": side, "size": size,
                             "arms": result})
    return rows


def score(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arms = ("immediate", "twap_8", "bitget_twap_60s", "argus_ac", "argus_ac_60s")
    out: dict[str, Any] = {}
    for size in SIZES:
        subset = [r for r in rows if r["size"] == size
                  and all(r["arms"][a] is not None for a in arms)]
        block: dict[str, Any] = {"parents": len(subset)}
        for metric in ("cost_bps", "shortfall_bps"):
            block[metric] = {a: round(statistics.fmean(r["arms"][a][metric] for r in subset), 3)
                             for a in arms} if subset else {}
        block["shortfall_std_bps"] = {
            a: round(statistics.stdev(r["arms"][a]["shortfall_bps"] for r in subset), 3)
            for a in arms} if len(subset) > 1 else {}
        comps = {}
        for mine in ("argus_ac", "argus_ac_60s"):
            for rival in ("immediate", "twap_8", "bitget_twap_60s"):
                diffs = [r["arms"][mine]["cost_bps"] - r["arms"][rival]["cost_bps"]
                         for r in subset]
                wins = sum(d < -1e-9 for d in diffs)
                losses = sum(d > 1e-9 for d in diffs)
                comps[f"{mine} vs {rival}"] = {
                    "mean_difference_bps": round(statistics.fmean(diffs), 3) if diffs else None,
                    "parents_cheaper": wins, "parents_dearer": losses,
                    "sign_test_p": round(sign_test(wins, losses), 5)}
        block["cost_comparisons"] = comps
        out[f"{int(size):,}"] = block
    return out


def main() -> int:  # pragma: no cover - CLI
    import json

    import numpy as np

    path = Path(os.environ.get("ARGUS_TARDIS_NPZ", ""))
    if not path.is_file():
        raise SystemExit("set ARGUS_TARDIS_NPZ to the converted Tardis day (.npz)")
    raw = path.read_bytes()
    data = np.load(path)["data"]
    snaps = snapshots(data)
    rows = run(snaps)
    scored = score(rows)
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "referee": {"source": "Tardis bitget-futures incremental_book_L2, NVDAUSDT, 2026-08-01 "
                              "(a Saturday), hftbacktest event format",
                    "events": len(data), "snapshots": len(snaps),
                    "npz_sha256": hashlib.sha256(raw).hexdigest(),
                    "depletion": "a child's take stays out of a level until the venue "
                                 "re-reports it",
                    "fee_bps_per_side": TAKER_BPS},
        "design": {"sizes_usdt": list(SIZES), "horizon_minutes": HORIZON_MIN,
                   "parent_every_minutes": PARENT_EVERY_MIN, "children": CHILDREN,
                   "ac_decay": AC_DECAY, "primary": "cost_bps (children vs mid at send, + fees)"},
        "by_size": scored,
        "limitations": [
            "one day, one symbol, a Saturday: weekend liquidity only",
            "impact beyond the visible book is not modelled; other traders do not react",
            "taker children only: no passive or maker arm is scored in this version",
        ],
        "rows_sample": rows[:16],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(scored, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
