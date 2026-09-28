"""Session boundaries, head to head: the schedulers that lead Execution Assistance, run on the same
full-depth Bitget books across the US open and close, against ARGUS's session-aware schedule.

The register graded "Session-aware execution that refuses to solve through a boundary" down from
OWNED on 2026-09-24 for a reason this module accepts in full: `eval/schedule_comparison.py` scored
Almgren-Chriss against TWAP *on Almgren-Chriss's own objective*, which proves the algebra and says
nothing about execution. The rivals the review named — PACE (the LLM parent-order executor), the
S2 entries Egress and zz-0816/bitget-s2-execution-aware-alpha, Bitget's own TWAP — had never been
run on the same fills. This module runs them, on realised fills, and scores what a trader pays.

**The question.** A parent order arrives at ``t0`` with a horizon that contains an anchor open or
close. How should it be split? Every arm answers with child orders ``(minute, notional)``; every
child is a market order that walks the book as it stood at that minute on Tardis's free
first-of-month ``incremental_book_L2`` replay of Bitget (the same depletion rule as
`eval/execution_arena.py`: what a child takes stays taken until the venue re-reports the level).

**Arms.**

* ``immediate``, ``twap_8`` and ``bitget_twap_60s`` — Bitget's native TWAP by its published spec
  (equal market slices, >= 10 USDT each, the 60-second interval), as in `eval/execution_arena.py`.
* ``argus_blind`` — ARGUS's Almgren-Chriss shape as the console prints it (inventory decays e-fold
  over the horizon) in one-minute children, solved with the *arrival* segment's parameters held
  constant: the constant-parameter assumption, stated.
* ``argus_console`` — what `lui/research/execution.py::_optimal_schedule` does today at a boundary:
schedule only
  what fits before the first whole hour at which the anchor opens or closes (DualClock, LEAN
  holidays, no early closes), at <= 10% of an hour's volume, and plan the rest after it.
* ``argus_session`` — `desk/session_schedule.solve`: every minute priced with its own segment's
  measured half-spread, impact slope and volatility, calibrated on training days only.
* ``egress`` — Ritapossible/Egress (MIT, @cc1dda6), its own ``exitcost.sliced`` and
  ``desk._advice`` run unmodified on the arrival book to choose 1, 4 or 12 clips; the clips are
  spread evenly over the horizon (Egress states no timing). Sell parents only — Egress prices
  exits.
* ``zz0816`` — zz-0816's trader rule (`project2/agent_team.py:2320,2530`: slices of $2,000,
  ``max(1, round(q / 2000 + 0.4999))``), spread evenly over the horizon. No licence: the rule is
  run as behaviour, nothing is copied. Its depth cap (0.25 x level-1) would have refused most of
  every parent; the share it would have refused is recorded, and the full parent is executed so
  the costs are comparable.
* ``pace`` — PACE's Planner (arXiv 2607.28410, eq. 5: ``w_n = (1 - lambda c)/N + lambda c
  softmax(a)_n``, lambda = 0.3, 5-minute slots, one-minute aggressive children inside a slot — the
  paper's own "w/o E" ablation, because the Executor needs a model call every minute). Rebuilt from
  the paper: the repository holds a licence file and nothing else. The Planner is Qwen 3.8 Max,
  called once per parent on ten pre-registered parents, every response written to disk before it
  is used.

**Ablations of ARGUS's own arm:** risk aversion removed (cost-only), volatility only, cost only,
and the calendar replaced by a fixed UTC clock (13:30-20:00 UTC every weekday, no holidays, no
early closes — what a time-of-day profile does).

**Metrics**, per parent: ``cost_bps`` (children against the mid when each was sent, plus fee —
what the schedule controls) and ``shortfall_bps`` (average fill against the arrival mid, plus fee —
the whole bill, dominated by where the price went). A risk-averse schedule exists to shrink the
second one's dispersion, so both are reported and compared: paired mean differences with sign and
Wilcoxon tests, and a symbol-day block bootstrap for the dispersion ratio.

    python -m argus.eval.session_arena              # stream the days (cached), run, write
    python -m argus.eval.session_arena --pace-live  # also spend the pre-registered PACE calls
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import random
import statistics
import sys
import tempfile
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

from argus.desk.session_schedule import (
    BookObservation,
    Segment,
    SegmentParams,
    SessionModel,
    SessionRefusal,
    boundaries,
    calibrate,
    segment_at,
    solve,
)
from argus.execution.schedule import ScheduleError
from argus.truth.artefact import write as write_artefact

PACKAGE = Path(__file__).resolve().parents[3]
REPORT_PATH = PACKAGE / "data" / "session_arena.json"
PACE_RECORD = PACKAGE / "data" / "session_arena_pace.jsonl"
RIVALS = PACKAGE.parent / "research" / "repos-owned"
EGRESS_DIR = RIVALS / "Ritapossible~Egress"
ZZ_DIR = RIVALS / "zz-0816~bitget-s2-execution-aware-alpha"
LEAN_DB = (PACKAGE.parent / "research" / "repos" / "Lean-upstream" / "Data" / "market-hours"
           / "market-hours-database.json")
CACHE = Path(os.environ.get("ARGUS_SESSION_CACHE",
                            str(Path(tempfile.gettempdir()) / "argus_session_books")))
TARDIS = "https://datasets.tardis.dev/v1"
USER_AGENT = "tardis-dev/2.0 (+https://github.com/tardis-dev/tardis-python)"
LEVELS = 100
MINUTE_MS = 60_000

PERP, SPOT = "bitget-futures", "bitget"
FEE_BPS = {PERP: 6.0, SPOT: 5.0}
"""Taker fee per side. Perpetual 0.06% (`cost/model.py`, and zz-0816's `execution_cost.py:75-77`
from a real account); rToken spot 0.05% maker and taker (zz-0816 `docs/09`, Bitget's rToken fee
announcement). Identical across arms, so it moves every arm's level and reorders none."""

PERPS = ("NVDAUSDT", "TSLAUSDT", "AAPLUSDT", "QQQUSDT")
SPOTS = ("RNVDAUSDT", "RTSLAUSDT", "RQQQUSDT")
SPOT_SINCE = {"RNVDAUSDT": date(2026, 6, 1), "RTSLAUSDT": date(2026, 6, 8),
              "RQQQUSDT": date(2026, 6, 8)}
"""Tardis's ``availableSince`` for each spot rToken (``api.tardis.dev/v1/exchanges/bitget``)."""
PERP_SINCE = {"NVDAUSDT": date(2025, 8, 20), "TSLAUSDT": date(2025, 8, 20),
              "AAPLUSDT": date(2025, 8, 26), "QQQUSDT": date(2025, 10, 28)}
"""The same for the perpetuals (``api.tardis.dev/v1/exchanges/bitget-futures``, read
2026-09-28). A day before it is answered with HTTP 400; it was never data, so it is skipped the
way a spot day before its listing is, and not counted as unavailable."""

# Pre-registered. Tardis serves the first day of every month without a key; these are every such
# day with both data and a reason to be here. Training days come strictly before test days.
PERP_TRAIN = ("2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01")
"""Labor Day (a holiday), a Wednesday in daylight time, a Saturday, a Monday in standard time."""
PERP_TEST = ("2026-01-01", "2026-04-01", "2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01",
             "2026-09-01")
"""Strictly after every training day: New Year's Day (a holiday), five weekdays, one Saturday."""
SPOT_TRAIN = ("2026-06-01", "2026-07-01", "2026-08-01")
SPOT_TEST = ("2026-09-01",)
DST_TRAIN = ("2025-09-01", "2025-10-01", "2025-11-01", "2026-04-01", "2026-05-01")
DST_TEST = ("2025-12-01",)
"""The daylight-saving experiment, held out rather than forward: fitted on daylight-time days only,
scored on the one first-of-month weekday in standard time, when the anchor opens at 14:30 UTC and
not 13:30."""

SIZES = (5_000.0, 25_000.0, 100_000.0, 250_000.0)
HORIZONS = (60, 240)
PARENT_EVERY_MIN = 30
DECAY = 1.0
"""kappa x horizon — the console's default (`lui/research.SCHEDULE_DECAYS[False]`)."""
BITGET_MIN_CHILD = 10.0
ZZ_SLICE_USD = 2_000.0
ZZ_DEPTH_TAKE = 0.25
CONSOLE_PARTICIPATION = 0.10
PACE_LAMBDA = 0.3
PACE_SLOT_MIN = 5
PACE_PARENTS = (
    ("NVDAUSDT", "2026-06-01", "13:00", "SELL"), ("NVDAUSDT", "2026-06-01", "19:30", "BUY"),
    ("TSLAUSDT", "2026-06-01", "13:00", "SELL"), ("TSLAUSDT", "2026-06-01", "19:30", "BUY"),
    ("NVDAUSDT", "2026-07-01", "13:00", "SELL"), ("NVDAUSDT", "2026-07-01", "19:30", "BUY"),
    ("TSLAUSDT", "2026-07-01", "13:00", "SELL"), ("TSLAUSDT", "2026-07-01", "19:30", "BUY"),
    ("NVDAUSDT", "2026-09-01", "13:00", "SELL"), ("NVDAUSDT", "2026-09-01", "19:30", "BUY"),
)
"""Ten parents, fixed before any PACE call: $100k over 60 minutes, starting half an hour before the
open (13:30 UTC in daylight time) or the close (20:00 UTC), on the three forward test weekdays.
Ten is the Qwen budget for this row; it is a small sample and reported as one."""
PACE_SIZE = 100_000.0
PACE_HORIZON = 60
BOOTSTRAP = 2000
SEED = 20260925


class ArenaError(RuntimeError):
    """The arena cannot be run honestly — surfaced rather than skipped."""


# --- data: Tardis, streamed and cached compactly -------------------------------------------------


@dataclass
class Day:
    """One symbol-day: a book every minute (top :data:`LEVELS` per side) and traded volume."""

    exchange: str
    symbol: str
    day: str
    minute: list[int]
    """Minute of day (0..1439) of each cut."""
    mids: list[float]
    bids: list[list[tuple[float, float, int]]]
    asks: list[list[tuple[float, float, int]]]
    """``(price, quantity, updated_ms_of_day)``, best first."""
    volume: list[float]
    """Base units traded in each minute of the day (1,440 entries)."""
    sources: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.exchange}:{self.symbol}:{self.day}"

    def start_of_day(self) -> datetime:
        return datetime.fromisoformat(self.day).replace(tzinfo=UTC)


class _Hashing(io.RawIOBase):
    def __init__(self, raw: Any) -> None:
        self.raw = raw
        self.sha = hashlib.sha256()
        self.size = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        chunk = self.raw.read(len(buffer))
        self.sha.update(chunk)
        self.size += len(chunk)
        buffer[:len(chunk)] = chunk
        return len(chunk)


def _open(exchange: str, kind: str, symbol: str, day: str) -> Any:
    url = f"{TARDIS}/{exchange}/{kind}/{day.replace('-', '/')}/{symbol}.csv.gz"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=180)


def stream_day(exchange: str, symbol: str, day: str) -> Day:
    """Stream one day of L2 deltas and trades from Tardis without writing the raw files to disk:
    the book is rebuilt in memory and cut every minute. The source bytes are hashed as they pass,
    so the cached compact form can be tied back to what Tardis served."""
    start_ms = int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1000)
    bids: dict[float, tuple[float, int]] = {}
    asks: dict[float, tuple[float, int]] = {}
    out_min: list[int] = []
    out_b: list[list[tuple[float, float, int]]] = []
    out_a: list[list[tuple[float, float, int]]] = []
    hashing = _Hashing(_open(exchange, "incremental_book_L2", symbol, day))
    reader = gzip.GzipFile(fileobj=io.BufferedReader(hashing))
    if next(reader, None) is None:
        raise ArenaError(f"Tardis served no book for {exchange} {symbol} on {day}")
    next_cut: int | None = None
    previous_snapshot = False
    for line in reader:
        parts = line.split(b",")
        ts_ms = int(parts[2]) // 1000 - start_ms
        if next_cut is None:
            next_cut = (ts_ms // MINUTE_MS + 1) * MINUTE_MS
        while ts_ms >= next_cut:
            if bids and asks:
                top_b = sorted(bids.items(), key=lambda kv: -kv[0])[:LEVELS]
                top_a = sorted(asks.items())[:LEVELS]
                out_min.append(next_cut // MINUTE_MS)
                out_b.append([(p, q, u) for p, (q, u) in top_b])
                out_a.append([(p, q, u) for p, (q, u) in top_a])
            next_cut += MINUTE_MS
        snapshot = parts[4] == b"true"
        if snapshot and not previous_snapshot:
            bids.clear()
            asks.clear()
        previous_snapshot = snapshot
        book = bids if parts[5] == b"bid" else asks
        price, qty = float(parts[6]), float(parts[7])
        if qty <= 0:
            book.pop(price, None)
        else:
            book[price] = (qty, ts_ms)
    keep = [i for i, m in enumerate(out_min) if m < 1440]
    volume = [0.0] * 1440
    trades_hash = _Hashing(_open(exchange, "trades", symbol, day))
    treader = gzip.GzipFile(fileobj=io.BufferedReader(trades_hash))
    # Tardis's bitget spot rToken trade files are empty (20 bytes, no header) on every day used
    # here; the volume is then unknown, recorded as such, and never read as zero trading.
    has_trades = next(treader, None) is not None
    for line in treader:
        parts = line.split(b",")
        minute = (int(parts[2]) // 1000 - start_ms) // MINUTE_MS
        if 0 <= minute < 1440:
            volume[minute] += float(parts[7])
    mids = [(out_b[i][0][0] + out_a[i][0][0]) / 2 for i in keep]
    return Day(exchange=exchange, symbol=symbol, day=day, minute=[out_min[i] for i in keep],
               mids=mids, bids=[out_b[i] for i in keep], asks=[out_a[i] for i in keep],
               volume=volume, sources={
                   "incremental_book_L2": {"sha256": hashing.sha.hexdigest(),
                                           "bytes": hashing.size},
                   "trades": {"sha256": trades_hash.sha.hexdigest(), "bytes": trades_hash.size,
                              "empty": not has_trades},
                   "book_minutes": len(keep),
                   "last_book_minute": out_min[keep[-1]] if keep else None})


def _cache_path(exchange: str, symbol: str, day: str) -> Path:
    return CACHE / f"{exchange}_{symbol}_{day}.json.gz"


def _unavailable_path() -> Path:
    return CACHE / "unavailable.json"


def unavailable() -> dict[str, str]:
    """Pre-registered days Tardis served no book for, keyed ``exchange/symbol/day``.

    NVDAUSDT on 2025-09-01 comes back as a 20-byte empty gzip (checked 2026-09-28; the same URL
    for 2025-10-01 carries 13 MB). A source gap is not a result, so the day is recorded here,
    left out of every fit and score, and named in the report's ``not_run`` — the set is not
    re-chosen to replace it."""
    path = _unavailable_path()
    if not path.exists():
        return {}
    loaded: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _mark_unavailable(exchange: str, symbol: str, day: str, reason: str) -> None:
    found = unavailable()
    found[f"{exchange}/{symbol}/{day}"] = reason
    CACHE.mkdir(parents=True, exist_ok=True)
    _unavailable_path().write_text(json.dumps(found, indent=1, sort_keys=True) + "\n",
                                   encoding="utf-8", newline="\n")


def save_day(d: Day) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = _cache_path(d.exchange, d.symbol, d.day)
    blob = {"exchange": d.exchange, "symbol": d.symbol, "day": d.day, "minute": d.minute,
            "bids": d.bids, "asks": d.asks, "volume": d.volume, "sources": d.sources}
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(blob, fh, separators=(",", ":"))
    tmp.replace(path)
    return path


def load_day(exchange: str, symbol: str, day: str, *, fetch: bool = True) -> Day:
    path = _cache_path(exchange, symbol, day)
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            blob = json.load(fh)
        bids = [[(float(p), float(q), int(u)) for p, q, u in lv] for lv in blob["bids"]]
        asks = [[(float(p), float(q), int(u)) for p, q, u in lv] for lv in blob["asks"]]
        out = Day(exchange=exchange, symbol=symbol, day=day, minute=list(blob["minute"]),
                  mids=[(b[0][0] + a[0][0]) / 2 for b, a in zip(bids, asks, strict=True)],
                  bids=bids, asks=asks, volume=list(blob["volume"]), sources=blob["sources"])
        out.sources["cache_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return out
    if not fetch:
        raise ArenaError(f"{path} is not cached and fetching is off")
    d = stream_day(exchange, symbol, day)
    save_day(d)
    return load_day(exchange, symbol, day, fetch=False)


def observations(d: Day) -> Iterator[BookObservation]:
    base = d.start_of_day()
    for i, minute in enumerate(d.minute):
        yield BookObservation(at=base + timedelta(minutes=minute),
                              bids=[(p, q) for p, q, _ in d.bids[i]],
                              asks=[(p, q) for p, q, _ in d.asks[i]])


# --- the referee ---------------------------------------------------------------------------------


@dataclass
class Book:
    minute: int
    ts_ms: int
    mid: float
    bids: list[tuple[float, float, int]]
    asks: list[tuple[float, float, int]]


def books_by_minute(d: Day) -> list[Book | None]:
    grid: list[Book | None] = [None] * 1440
    for i, minute in enumerate(d.minute):
        grid[minute] = Book(minute=minute, ts_ms=minute * MINUTE_MS, mid=d.mids[i],
                            bids=d.bids[i], asks=d.asks[i])
    return grid


def walk(book: Book, side: str, notional: float,
         taken: dict[tuple[str, float], tuple[float, int]]) -> tuple[float, float] | None:
    """`eval/execution_arena.walk`, on pre-sorted levels: spend ``notional`` against the book,
    net of what this parent already took from levels the venue has not re-reported since. Returns
    ``(spent, shares)``, or ``None`` when the visible book cannot fill it. Parity with the original
    is a test (`tests/test_session_arena.py::TestReferee`)."""
    levels = book.asks if side == "BUY" else book.bids
    tag = "ask" if side == "BUY" else "bid"
    left, shares, spent = notional, 0.0, 0.0
    for price, qty, updated in levels:
        prior = taken.get((tag, price))
        stale = prior is not None and updated <= prior[1]
        available = qty - prior[0] if stale and prior is not None else qty
        if available <= 0:
            continue
        take = min(available, left / price)
        shares += take
        spent += take * price
        left -= take * price
        used = (prior[0] if stale and prior is not None else 0.0) + take
        taken[(tag, price)] = (used, book.ts_ms)
        if left <= 1e-9:
            break
    if left > 1e-6:
        return None
    return spent, shares


def execute(grid: Sequence[Book | None], start: int, side: str,
            children: Sequence[tuple[int, float]], fee_bps: float,
            horizon: int) -> dict[str, float] | None:
    arrival = grid[start]
    if arrival is None:
        return None
    sign = 1.0 if side == "BUY" else -1.0
    taken: dict[tuple[str, float], tuple[float, int]] = {}
    cost_num, spent_total, shares_total = 0.0, 0.0, 0.0
    for minute, notional in children:
        if notional <= 1e-9:
            continue
        idx = start + minute
        if idx >= len(grid) or grid[idx] is None:
            return None
        book = grid[idx]
        assert book is not None
        filled = walk(book, side, notional, taken)
        if filled is None:
            return None
        spent, shares = filled
        cost_num += sign * (spent / shares - book.mid) / book.mid * spent
        spent_total += spent
        shares_total += shares
    if shares_total <= 0:
        return None
    window = [b.mid for b in grid[start:start + horizon] if b is not None]
    average = spent_total / shares_total
    twap_price = statistics.fmean(window)
    return {
        "cost_bps": cost_num / spent_total * 1e4 + fee_bps,
        "shortfall_bps": sign * (average - arrival.mid) / arrival.mid * 1e4 + fee_bps,
        "vs_window_twap_bps": sign * (average - twap_price) / twap_price * 1e4 + fee_bps,
        "children": float(sum(1 for _, q in children if q > 1e-9)),
    }


# --- parents -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Parent:
    exchange: str
    symbol: str
    day: str
    start_minute: int
    horizon: int
    size: float
    side: str

    @property
    def start(self) -> datetime:
        return (datetime.fromisoformat(self.day).replace(tzinfo=UTC)
                + timedelta(minutes=self.start_minute))

    @property
    def id(self) -> str:
        return (f"{self.symbol}:{self.day}T{self.start_minute // 60:02d}:"
                f"{self.start_minute % 60:02d}:{self.horizon}m:{int(self.size)}:{self.side}")

    def crosses(self, segmenter: Callable[[datetime], Segment] = segment_at) -> bool:
        return any(b.is_anchor_open_or_close
                   for b in boundaries(self.start, self.horizon, segmenter=segmenter))


def parents_for(d: Day) -> list[Parent]:
    out = []
    for start in range(0, 1440, PARENT_EVERY_MIN):
        for horizon in HORIZONS:
            if start + horizon > 1440:
                continue
            for size in SIZES:
                for side in ("BUY", "SELL"):
                    out.append(Parent(d.exchange, d.symbol, d.day, start, horizon, size, side))
    return out


# --- the arms ------------------------------------------------------------------------------------


def even(total: float, horizon: int, count: int) -> list[tuple[int, float]]:
    count = max(1, min(count, horizon))
    every = horizon / count
    return [(round(j * every), total / count) for j in range(count)]


def bitget_twap(total: float, horizon: int) -> list[tuple[int, float]]:
    return even(total, horizon, max(1, min(horizon, int(total // BITGET_MIN_CHILD))))


def utc_clock_segment(instant: datetime) -> Segment:
    """A fixed time-of-day profile in UTC: weekday regular session 13:30-20:00 UTC, pre-market
    08:00, post-market to 24:00, no holidays, no early closes, no daylight saving. The calendar a
    model gets when it learns an intraday profile on the UTC clock."""
    minute = instant.hour * 60 + instant.minute
    if instant.weekday() >= 5:
        return Segment.WEEKEND
    if 810 <= minute < 1200:
        if minute < 840:
            return Segment.OPEN
        if minute >= 1170:
            return Segment.CLOSE
        return Segment.RTH
    if 480 <= minute < 810:
        return Segment.PRE
    if minute >= 1200:
        return Segment.POST
    return Segment.OVERNIGHT


def mixed_model(model: SessionModel, arrival: SegmentParams, *, keep: str) -> SessionModel:
    """Ablation: each segment keeps only ``keep`` ("sigma" or "cost") of its own parameters and
    takes the rest from the arrival segment."""
    params = {}
    for seg, p in model.params.items():
        if keep == "sigma":
            params[seg] = SegmentParams(arrival.half_spread_bps, arrival.impact_bps_per_usd,
                                        p.sigma_bps, p.observations)
        elif keep == "cost":
            params[seg] = SegmentParams(p.half_spread_bps, p.impact_bps_per_usd,
                                        arrival.sigma_bps, p.observations)
        else:
            raise ArenaError(f"keep={keep!r}")
    return SessionModel(model.symbol, model.venue, model.fee_bps, params, model.calibrated_on)


def session_children(parent: Parent, model: SessionModel, **kwargs: Any) -> list[tuple[int, float]]:
    schedule = solve(parent.size, parent.start, parent.horizon, model, **kwargs)
    return schedule.children()


def console_children(parent: Parent, day_volume_usd: float) -> list[tuple[int, float]]:
    """`lui/research/execution.py::_optimal_schedule` at a boundary, replayed: the first whole
    hour at which the DualClock (LEAN holidays, no early closes) changes open state; the part that
    fits before it at 10% of an hour's volume, as the e-fold schedule over those hours; the rest
    re-planned the same way over what is left of the horizon."""
    clock = _console_clock()

    def is_open(t: datetime) -> bool:
        return bool(clock.phase(t).has_price_discovery)

    hours = max(1, math.ceil(parent.horizon / 60))
    state = is_open(parent.start)
    change = next((h for h in range(1, hours + 1)
                   if is_open(parent.start + timedelta(hours=h)) != state), None)
    if change is None or change * 60 >= parent.horizon:
        return _ac_even(parent.size, parent.horizon, 0)
    before_min = change * 60
    fits = day_volume_usd / 24 * CONSOLE_PARTICIPATION * change
    first = min(parent.size, fits)
    out = _ac_even(first, before_min, 0)
    rest = parent.size - first
    if rest > 1e-9:
        out += _ac_even(rest, parent.horizon - before_min, before_min)
    return out


_CLOCK: list[Any] = []


def _console_clock() -> Any:
    """The console's clock (`lui/research/session.py::_dual_clock`): DualClock with LEAN's
    holidays."""
    if not _CLOCK:
        from argus.eval.baselines.lean_market_holidays_loader import load_usa_equity_holidays
        from argus.truth.clocks import DualClock

        _CLOCK.append(DualClock(holidays=load_usa_equity_holidays()))
    return _CLOCK[0]


def _ac_even(total: float, horizon: int, offset: int) -> list[tuple[int, float]]:
    """The e-fold Almgren-Chriss shape (`eval/execution_arena.ac_fractions`) in one-minute
    children."""
    held = [math.sinh(DECAY * (1 - j / horizon)) / math.sinh(DECAY) for j in range(horizon + 1)]
    return [(offset + j, total * (held[j] - held[j + 1])) for j in range(horizon)]


def _egress(module: str) -> Any:
    """Import a module of the cloned Egress package (MIT, @cc1dda6) exactly as it ships."""
    import importlib

    if str(EGRESS_DIR) not in sys.path:
        sys.path.insert(0, str(EGRESS_DIR))
    return importlib.import_module(f"egress.{module}")


def egress_children(parent: Parent, book: Book) -> tuple[list[tuple[int, float]], dict[str, Any]]:
    """Egress's own split decision on the arrival book, its code run unmodified."""
    egress_desk = _egress("desk")
    exitcost = _egress("exitcost")
    bids = [(p, q) for p, q, _ in book.bids]
    asks = [(p, q) for p, q, _ in book.asks]
    plan = [exitcost.sliced(parent.symbol, parent.size, n, bids, asks,
                            fee_bp=FEE_BPS[parent.exchange]) for n in egress_desk.SLICE_CHOICES]
    advice = egress_desk._advice({"plan": plan}, {})
    quotable = [p for p in plan if p.get("quotable")]
    best = min(quotable, key=lambda p: p["best_case_bp"]) if quotable else None
    one = next((p for p in quotable if p["slices"] == 1), None)
    clips = 1
    # desk._advice's own decision (desk.py:363-377), read off the same plan: split into the best
    # clip count only when the saving over one clip reaches WORTH_SPLITTING_BP.
    if (best is not None and one is not None and best["slices"] != 1
            and one["worst_case_bp"] - best["best_case_bp"] >= egress_desk.WORTH_SPLITTING_BP):
        clips = int(best["slices"])
    return even(parent.size, parent.horizon, clips), {"clips": clips, "advice": advice}


def zz0816_children(parent: Parent, book: Book) -> tuple[list[tuple[int, float]], float]:
    slices = max(1, round(parent.size / ZZ_SLICE_USD + 0.4999))
    level1 = (book.asks[0] if parent.side == "BUY" else book.bids[0])
    cap = level1[0] * level1[1] * ZZ_DEPTH_TAKE
    refused = max(0.0, 1.0 - cap / parent.size)
    return even(parent.size, parent.horizon, slices), refused


# --- PACE ----------------------------------------------------------------------------------------


PACE_SYSTEM = (
    "You are the long-horizon Planner of an execution algorithm. A parent order must be fully "
    "executed inside its window with aggressive (market) child orders. Glossary: Ask1 is the "
    "lowest sell price, Bid1 the highest buy price, the mid-price is their average; TWAP spreads "
    "the quantity evenly over the window. A buy order prefers to trade more when prices are "
    "expected to be lower and less when they are expected to be higher; a sell order the "
    "opposite. Split the window into the given slots. For each slot output a quantity preference "
    "score a_n in [-1, 1] (larger means allocate more to that slot) and one overall confidence c "
    "in [0, 1]. First write a brief trend assessment of the window. Answer with JSON only: "
    '{"trend_assessment": str, "slot_scores": [float, ...], "confidence": float}.')


def pace_prompt(parent: Parent, grid: Sequence[Book | None], volume: Sequence[float]) -> str:
    """The Planner's inputs as the paper states them (§3.2): the parent order, the market history
    over the preceding window length (mid-price and traded volume per minute), and the TWAP curve.
    Stock identifiers and dates are removed, as the paper does to limit leakage; times of day are
    kept, as the paper's own example parent ("from 10:30 to 11:00") keeps them."""
    start, horizon = parent.start_minute, parent.horizon
    rows = []
    for m in range(max(0, start - horizon), start):
        book = grid[m]
        if book is not None:
            rows.append(f"{m // 60:02d}:{m % 60:02d} mid={book.mid:.4f} volume={volume[m]:.4f}")
    arrival = grid[start]
    shares = parent.size / arrival.mid if arrival is not None else 0.0
    slots = horizon // PACE_SLOT_MIN
    end = start + horizon
    slot_lines = [f"slot {n + 1}: {(start + n * PACE_SLOT_MIN) // 60:02d}:"
                  f"{(start + n * PACE_SLOT_MIN) % 60:02d}-"
                  f"{(start + (n + 1) * PACE_SLOT_MIN) // 60:02d}:"
                  f"{(start + (n + 1) * PACE_SLOT_MIN) % 60:02d}" for n in range(slots)]
    return "\n".join([
        f"Parent order: {parent.side} {shares:.4f} shares from {start // 60:02d}:{start % 60:02d} "
        f"to {end // 60:02d}:{end % 60:02d} (times are UTC), decision interval 1 minute.",
        f"TWAP curve: {shares / horizon:.6f} shares every minute for {horizon} minutes.",
        f"Market history, the {horizon} minutes before the order:",
        *rows,
        f"Slots ({slots}, {PACE_SLOT_MIN} minutes each):",
        *slot_lines,
        f"Return exactly {slots} slot_scores."])


def pace_weights(scores: Sequence[float], confidence: float,
                 lam: float = PACE_LAMBDA) -> list[float]:
    """PACE eq. 5."""
    clipped = [min(1.0, max(-1.0, float(s))) for s in scores]
    c = min(1.0, max(0.0, float(confidence)))
    top = max(clipped)
    exps = [math.exp(s - top) for s in clipped]
    total = sum(exps)
    n = len(clipped)
    return [(1 - lam * c) / n + lam * c * e / total for e in exps]


def pace_children(parent: Parent, answer: Mapping[str, Any]) -> list[tuple[int, float]]:
    slots = parent.horizon // PACE_SLOT_MIN
    scores = answer.get("slot_scores")
    if not isinstance(scores, list) or len(scores) != slots:
        raise ArenaError(f"PACE answer for {parent.id} has {len(scores or [])} scores, "
                         f"wanted {slots}")
    weights = pace_weights([float(s) for s in scores], float(answer.get("confidence", 0.0)))
    out = []
    for n, w in enumerate(weights):
        for j in range(PACE_SLOT_MIN):
            out.append((n * PACE_SLOT_MIN + j, parent.size * w / PACE_SLOT_MIN))
    return out


def pace_recorded() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if PACE_RECORD.exists():
        for line in PACE_RECORD.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["parent"]] = row
    return out


def pace_call(parent: Parent, prompt: str) -> dict[str, Any]:  # pragma: no cover - live spend
    """One Planner call, written to disk before it is parsed. Never called by tests."""
    from argus.llm.qwen import QwenClient, Thinking

    client = QwenClient(max_retries=1, cache=False)
    messages = [{"role": "system", "content": PACE_SYSTEM}, {"role": "user", "content": prompt}]
    row: dict[str, Any] = {"parent": parent.id, "prompt_sha256":
                           hashlib.sha256(prompt.encode()).hexdigest(),
                           "at": datetime.now(UTC).isoformat(timespec="seconds")}
    try:
        result = client.complete(messages, json_mode=True, thinking=Thinking.LOW,
                                 max_tokens=4096)
        row.update({"content": result.content, "finish_reason": result.finish_reason,
                    "usage": {"prompt": result.usage.prompt_tokens,
                              "completion": result.usage.completion_tokens,
                              "reasoning": result.usage.reasoning_tokens}})
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:400]
    with PACE_RECORD.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def pace_answer(row: Mapping[str, Any]) -> dict[str, Any] | None:
    content = row.get("content")
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


# --- running -------------------------------------------------------------------------------------


ARMS = ("immediate", "twap_8", "bitget_twap_60s", "argus_blind", "argus_console", "argus_session",
        "egress", "zz0816", "pace")
ABLATIONS = ("session_risk_neutral", "session_sigma_only", "session_cost_only", "utc_clock")


def run_day(d: Day, model: SessionModel, *, utc_model: SessionModel | None = None,
            pace: Mapping[str, Mapping[str, Any]] | None = None,
            only: Callable[[Parent], bool] | None = None) -> list[dict[str, Any]]:
    grid = books_by_minute(d)
    fee = FEE_BPS[d.exchange]
    day_volume_usd = sum(v * m for v, m in zip(d.volume, _minute_mids(grid), strict=True))
    volume_known = not d.sources.get("trades", {}).get("empty", False) and day_volume_usd > 0
    rows = []
    for parent in parents_for(d):
        if only is not None and not only(parent):
            continue
        arrival = grid[parent.start_minute]
        if arrival is None:
            continue
        plans: dict[str, list[tuple[int, float]] | None] = {
            "immediate": [(0, parent.size)],
            "twap_8": even(parent.size, parent.horizon, 8),
            "bitget_twap_60s": bitget_twap(parent.size, parent.horizon),
            "argus_console": (console_children(parent, day_volume_usd)
                              if volume_known else None),
        }
        meta: dict[str, Any] = {"crosses": parent.crosses(),
                                "utc_clock_crosses": parent.crosses(utc_clock_segment)}
        refusals: dict[str, str] = {}
        variants: list[tuple[str, SessionModel, dict[str, Any]]] = [
            ("argus_blind", model, {"decay": DECAY, "parameters": "arrival"}),
            ("argus_session", model, {"decay": DECAY}),
            ("session_risk_neutral", model, {"risk_aversion": 0.0}),
        ]
        try:
            arrival_params = model.lookup(segment_at(parent.start), parent.start)
            variants.append(("session_sigma_only",
                             mixed_model(model, arrival_params, keep="sigma"), {"decay": DECAY}))
            variants.append(("session_cost_only",
                             mixed_model(model, arrival_params, keep="cost"), {"decay": DECAY}))
        except SessionRefusal as exc:
            refusals["arrival"] = str(exc)
        if utc_model is not None:
            variants.append(("utc_clock", utc_model,
                             {"decay": DECAY, "segmenter": utc_clock_segment}))
        for name, m, kwargs in variants:
            try:
                plans[name] = session_children(parent, m, **kwargs)
            except SessionRefusal as exc:
                plans[name] = None
                refusals[name] = str(exc)
            except ScheduleError as exc:
                plans[name] = None
                refusals[name] = f"error: {exc}"
        if parent.side == "SELL":
            plans["egress"], meta["egress"] = egress_children(parent, arrival)
        plans["zz0816"], meta["zz0816_would_refuse"] = zz0816_children(parent, arrival)
        if pace is not None and parent.id in pace:
            answer = pace_answer(pace[parent.id])
            if answer is not None:
                try:
                    plans["pace"] = pace_children(parent, answer)
                    meta["pace_confidence"] = answer.get("confidence")
                except (ArenaError, ValueError, TypeError) as exc:
                    refusals["pace"] = str(exc)
        results = {name: (execute(grid, parent.start_minute, parent.side, children, fee,
                                  parent.horizon) if children is not None else None)
                   for name, children in plans.items()}
        rows.append({"parent": parent.id, "exchange": parent.exchange, "symbol": parent.symbol,
                     "day": parent.day, "start_minute": parent.start_minute,
                     "horizon": parent.horizon, "size": parent.size, "side": parent.side,
                     "arrival_segment": segment_at(parent.start).value, **meta,
                     "refusals": refusals, "arms": results})
    return rows


def _minute_mids(grid: Sequence[Book | None]) -> list[float]:
    last = next((b.mid for b in grid if b is not None), 0.0)
    out = []
    for b in grid:
        if b is not None:
            last = b.mid
        out.append(last)
    return out


# --- statistics ----------------------------------------------------------------------------------


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    # Exact, as a fraction of integers: dividing by float(2 ** n) overflowed once the arena had
    # more than about a thousand paired parents (2026-09-28). A fraction converts to the nearest
    # float at the end, and underflows to 0.0 only when the tail truly is below 1e-308.
    tail = Fraction(sum(math.comb(n, i) for i in range(k + 1)), 2 ** n)
    return min(1.0, float(2 * tail))


def wilcoxon_p(diffs: Sequence[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank, normal approximation with tie correction; ``None`` below
    ten non-zero pairs, where the approximation is not honest."""
    nz = [d for d in diffs if abs(d) > 1e-12]
    n = len(nz)
    if n < 10:
        return None
    ranked = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    ties = 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(nz[ranked[j + 1]]) - abs(nz[ranked[i]])) < 1e-12:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[ranked[k]] = avg
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nz, strict=True) if d > 0)
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - ties / 48
    if var <= 0:
        return None
    z = (w_plus - mean) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2))


def paired(rows: Sequence[Mapping[str, Any]], mine: str, rival: str,
           metric: str) -> dict[str, Any] | None:
    pairs = [(r["arms"][mine][metric], r["arms"][rival][metric], f"{r['symbol']}:{r['day']}")
             for r in rows if r["arms"].get(mine) and r["arms"].get(rival)]
    if not pairs:
        return None
    diffs = [a - b for a, b, _ in pairs]
    wins = sum(d < -1e-9 for d in diffs)
    losses = sum(d > 1e-9 for d in diffs)
    blocks: dict[str, list[float]] = {}
    for (_, _, key), diff in zip(pairs, diffs, strict=True):
        blocks.setdefault(key, []).append(diff)
    block_means = [statistics.fmean(v) for v in blocks.values()]
    return {
        "n": len(pairs), "symbol_days": len(blocks),
        "mean_difference_bps": round(statistics.fmean(diffs), 4),
        "parents_better": wins, "parents_worse": losses,
        "sign_test_p": round(sign_test(wins, losses), 6),
        "wilcoxon_p": _round(wilcoxon_p(diffs)),
        "symbol_days_better": sum(m < 0 for m in block_means),
        "symbol_days_worse": sum(m > 0 for m in block_means),
        "symbol_day_sign_p": round(sign_test(sum(m < 0 for m in block_means),
                                             sum(m > 0 for m in block_means)), 6),
        "block_bootstrap_ci95_bps": _block_ci(blocks),
    }


def _round(v: float | None, digits: int = 6) -> float | None:
    return None if v is None else round(v, digits)


def _block_ci(blocks: Mapping[str, Sequence[float]]) -> list[float] | None:
    keys = sorted(blocks)
    if len(keys) < 2:
        return None
    rng = random.Random(SEED)
    stats = []
    for _ in range(BOOTSTRAP):
        sample: list[float] = []
        for _ in keys:
            sample.extend(blocks[keys[rng.randrange(len(keys))]])
        stats.append(statistics.fmean(sample))
    stats.sort()
    return [round(stats[int(0.025 * BOOTSTRAP)], 4), round(stats[int(0.975 * BOOTSTRAP) - 1], 4)]


def dispersion(rows: Sequence[Mapping[str, Any]], mine: str, rival: str) -> dict[str, Any] | None:
    """Shortfall standard deviation of ``mine`` over ``rival`` on the same parents, with a
    symbol-day block bootstrap interval for the ratio. Below 1 is less risk."""
    pairs = [(r["arms"][mine]["shortfall_bps"], r["arms"][rival]["shortfall_bps"],
              f"{r['symbol']}:{r['day']}")
             for r in rows if r["arms"].get(mine) and r["arms"].get(rival)]
    if len(pairs) < 3:
        return None
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    sb = statistics.stdev(b)
    if sb <= 0:
        return None
    ratio = statistics.stdev(a) / sb
    blocks: dict[str, list[tuple[float, float]]] = {}
    for x, y, key in pairs:
        blocks.setdefault(key, []).append((x, y))
    keys = sorted(blocks)
    ci = None
    if len(keys) >= 2:
        rng = random.Random(SEED)
        stats = []
        for _ in range(BOOTSTRAP):
            sample: list[tuple[float, float]] = []
            for _ in keys:
                sample.extend(blocks[keys[rng.randrange(len(keys))]])
            sy = statistics.stdev(y for _, y in sample)
            if sy > 0:
                stats.append(statistics.stdev(x for x, _ in sample) / sy)
        stats.sort()
        if stats:
            ci = [round(stats[int(0.025 * len(stats))], 4),
                  round(stats[int(0.975 * len(stats)) - 1], 4)]
    return {"n": len(pairs), "std_mine_bps": round(statistics.stdev(a), 4),
            "std_rival_bps": round(sb, 4), "ratio": round(ratio, 4),
            "block_bootstrap_ci95": ci}


def summarise(rows: Sequence[Mapping[str, Any]], arms: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"parents": len(rows)}
    for metric in ("cost_bps", "shortfall_bps", "vs_window_twap_bps"):
        out[metric] = {}
        for arm in arms:
            vals = [r["arms"][arm][metric] for r in rows if r["arms"].get(arm)]
            if vals:
                out[metric][arm] = {"n": len(vals), "mean": round(statistics.fmean(vals), 4)}
    out["shortfall_std_bps"] = {}
    for arm in arms:
        vals = [r["arms"][arm]["shortfall_bps"] for r in rows if r["arms"].get(arm)]
        if len(vals) > 1:
            out["shortfall_std_bps"][arm] = round(statistics.stdev(vals), 4)
    return out


def head_to_head(rows: Sequence[Mapping[str, Any]], mine: str,
                 rivals: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rival in rivals:
        if rival == mine:
            continue
        cost = paired(rows, mine, rival, "cost_bps")
        shortfall = paired(rows, mine, rival, "shortfall_bps")
        disp = dispersion(rows, mine, rival)
        if cost is None:
            continue
        out[rival] = {"cost_bps": cost, "shortfall_bps": shortfall, "shortfall_dispersion": disp,
                      "verdict": verdict(cost, disp)}
    return out


def verdict(cost: Mapping[str, Any], disp: Mapping[str, Any] | None) -> str:
    """Pre-registered. WIN: cheaper with the whole block interval below zero and dispersion not
    worse (ratio interval does not sit wholly above 1), or dispersion lower (interval wholly below
    1) with cost not significantly worse. LOSS: the mirror image. Otherwise TIE."""
    ci = cost.get("block_bootstrap_ci95_bps")
    rci = (disp or {}).get("block_bootstrap_ci95")
    cheaper = ci is not None and ci[1] < 0
    dearer = ci is not None and ci[0] > 0
    less_risk = rci is not None and rci[1] < 1
    more_risk = rci is not None and rci[0] > 1
    if (cheaper and not more_risk) or (less_risk and not dearer):
        return "WIN"
    if (dearer and not less_risk) or (more_risk and not cheaper):
        return "LOSS"
    if (cheaper and more_risk) or (dearer and less_risk):
        return "TRADE-OFF"
    return "TIE"


# --- adversarial: the calendar -------------------------------------------------------------------


def lean_sessions(db: Mapping[str, Any], years: Sequence[int]) -> list[tuple[datetime, datetime]]:
    """Every regular session in ``years`` per LEAN's market-hours database, as UTC intervals: the
    weekday's ``market`` window in America/New_York, minus ``holidays``, cut at ``earlyCloses``.
    LEAN's date strings are unpadded (``1/9/2025``)."""
    from zoneinfo import ZoneInfo

    entry = db["entries"]["Equity-usa-[*]"]
    tz = ZoneInfo(entry["exchangeTimeZone"])
    closed = set(entry["holidays"])
    early_closes = entry.get("earlyCloses", {})
    names = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    out = []
    day = date(min(years), 1, 1)
    last = date(max(years), 12, 31)
    while day <= last:
        stamp = f"{day.month}/{day.day}/{day.year}"
        if stamp not in closed:
            for seg in entry[names[day.weekday()]]:
                if seg["state"] != "market":
                    continue
                start = datetime.strptime(seg["start"], "%H:%M:%S").time()
                end = datetime.strptime(seg["end"], "%H:%M:%S").time()
                if stamp in early_closes:
                    end = min(end, datetime.strptime(early_closes[stamp], "%H:%M:%S").time())
                out.append((datetime.combine(day, start, tzinfo=tz).astimezone(UTC),
                            datetime.combine(day, end, tzinfo=tz).astimezone(UTC)))
        day += timedelta(days=1)
    return out


def calendar_audit(years: Sequence[int] = (2025, 2026, 2027)) -> dict[str, Any]:
    """Minute by minute over ``years``, whose idea of "the anchor is in its regular session"
    disagrees with LEAN's market-hours database — ARGUS's segment clock and the three rival clocks
    that decide a boundary elsewhere."""
    db = json.loads(LEAN_DB.read_text(encoding="utf-8"))
    from argus.eval.baselines.lean_market_holidays_loader import load_usa_equity_holidays
    from argus.truth.clocks import DualClock

    clock = DualClock(holidays=load_usa_equity_holidays())
    clocks: dict[str, Callable[[datetime], bool]] = {
        "argus_session_schedule": lambda t: segment_at(t).anchor_open,
        "argus_dualclock_console": lambda t: clock.phase(t).has_price_discovery,
        "utc_time_of_day": lambda t: utc_clock_segment(t).anchor_open,
    }
    if (EGRESS_DIR / "egress" / "sessions.py").exists():
        egress_sessions = _egress("sessions")
        clocks["egress_sessions_phase"] = lambda t: egress_sessions.phase(t) == "open"
    if (ZZ_DIR / "common" / "market_calendar.py").exists():
        zz = _load_module("zz0816_market_calendar", ZZ_DIR / "common" / "market_calendar.py")
        clocks["zz0816_session_of"] = (
            lambda t: zz.session_of(int(t.timestamp() * 1000)) == "intraday")
    sessions = lean_sessions(db, years)
    wrong: dict[str, int] = dict.fromkeys(clocks, 0)
    examples: dict[str, list[str]] = {k: [] for k in clocks}
    minutes = 0
    cursor = 0
    for year in years:
        t = datetime(year, 1, 1, tzinfo=UTC)
        end = datetime(year + 1, 1, 1, tzinfo=UTC)
        while t < end:
            while cursor < len(sessions) and sessions[cursor][1] <= t:
                cursor += 1
            truth = cursor < len(sessions) and sessions[cursor][0] <= t < sessions[cursor][1]
            for name, fn in clocks.items():
                if fn(t) != truth:
                    wrong[name] += 1
                    seen = examples[name]
                    if len(seen) < 4 and (not seen or seen[-1][:10] != t.date().isoformat()):
                        examples[name].append(t.isoformat(timespec="minutes"))
            minutes += 1
            t += timedelta(minutes=1)
    return {"reference": "QuantConnect LEAN market-hours-database.json, Equity-usa-[*] "
                         "(holidays, earlyCloses, 09:30-16:00 America/New_York)",
            "minutes_checked": minutes, "years": list(years),
            "minutes_wrong": wrong,
            "hours_wrong_per_year": {k: round(v / 60 / len(years), 2) for k, v in wrong.items()},
            "first_wrong_examples": examples}


def _load_module(name: str, path: Path) -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ArenaError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- adversarial: the model ----------------------------------------------------------------------


def placebo_model(model: SessionModel, seed: int = SEED) -> SessionModel:
    """Segment labels shuffled among the measured segments: the same numbers attached to the
    wrong parts of the day. If the session arm's result survives this, the calendar is not what
    produced it."""
    segs = sorted(model.params, key=lambda s: s.value)
    shuffled = segs[:]
    rng = random.Random(seed)
    while len(segs) > 1 and shuffled == segs:
        rng.shuffle(shuffled)
    params = {a: model.params[b] for a, b in zip(segs, shuffled, strict=True)}
    return SessionModel(model.symbol, model.venue, model.fee_bps, params, model.calibrated_on)


def scaled_model(model: SessionModel, factor: float) -> SessionModel:
    """Every segment's impact slope and volatility scaled by ``factor``: a mis-calibration."""
    params = {s: SegmentParams(p.half_spread_bps, p.impact_bps_per_usd * factor,
                               p.sigma_bps * factor, p.observations)
              for s, p in model.params.items()}
    return SessionModel(model.symbol, model.venue, model.fee_bps, params, model.calibrated_on)


# --- the full study ------------------------------------------------------------------------------


def _days(exchange: str, symbols: Sequence[str], days: Sequence[str]) -> list[tuple[str, str]]:
    out = []
    gone = unavailable()
    for symbol in symbols:
        since = SPOT_SINCE.get(symbol) or PERP_SINCE.get(symbol)
        for day in days:
            if since is not None and date.fromisoformat(day) < since:
                continue
            if f"{exchange}/{symbol}/{day}" in gone:
                continue
            out.append((symbol, day))
    return out


_MODELS: dict[tuple[str, str, tuple[str, ...], str], SessionModel] = {}


def fit(exchange: str, symbol: str, days: Sequence[str], *,
        segmenter: Callable[[datetime], Segment] | None = None) -> SessionModel:
    """Calibrate on ``days`` only (memoised: several sections ask for the same fit)."""
    key = (exchange, symbol, tuple(days), getattr(segmenter, "__name__", "session"))
    if key in _MODELS:
        return _MODELS[key]
    obs: list[BookObservation] = []
    for _, day in _days(exchange, (symbol,), days):
        obs.extend(observations(load_day(exchange, symbol, day)))
    model = calibrate(obs, symbol=symbol, venue=exchange, fee_bps=FEE_BPS[exchange],
                      calibrated_on=tuple(days), segmenter=segmenter)
    _MODELS[key] = model
    return model


def study(*, pace: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "design": {
            "referee": "Tardis bitget / bitget-futures incremental_book_L2 rebuilt in memory, top "
                       f"{LEVELS} levels cut every minute; a child walks the book at its minute "
                       "and "
                       "what it takes stays taken until the venue re-reports the level "
                       "(eval/execution_arena.walk semantics)",
            "fees_bps": FEE_BPS, "sizes_usdt": list(SIZES), "horizons_min": list(HORIZONS),
            "parent_every_min": PARENT_EVERY_MIN, "decay": DECAY,
            "perp": {"symbols": list(PERPS), "train": list(PERP_TRAIN), "test": list(PERP_TEST)},
            "spot": {"symbols": list(SPOTS), "train": list(SPOT_TRAIN), "test": list(SPOT_TEST)},
            "dst": {"train": list(DST_TRAIN), "test": list(DST_TEST)},
            "primary": "boundary-crossing parents on test days: paired cost_bps and shortfall "
                       "dispersion, symbol-day block bootstrap, verdict rule in verdict()",
        },
    }
    pace_rows = pace_recorded() if pace else {}
    sources: dict[str, Any] = {}
    models: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for exchange, symbols, train, test in ((PERP, PERPS, PERP_TRAIN, PERP_TEST),
                                           (SPOT, SPOTS, SPOT_TRAIN, SPOT_TEST)):
        for symbol in symbols:
            model = fit(exchange, symbol, train)
            utc_model = fit(exchange, symbol, train, segmenter=utc_clock_segment)
            models[f"{exchange}:{symbol}"] = model.as_dict()
            for sym, day in _days(exchange, [symbol], list(train) + list(test)):
                d = load_day(exchange, sym, day, fetch=True)
                sources[d.key] = d.sources
            for sym, day in _days(exchange, [symbol], test):
                d = load_day(exchange, sym, day)
                all_rows.extend(run_day(d, model, utc_model=utc_model, pace=pace_rows))
    report["sources"] = sources
    report["models"] = models
    report["segment_table"] = _segment_table(models)
    arms = ARMS + ABLATIONS
    crossing = [r for r in all_rows if r["crosses"]]
    report["counts"] = {
        "parents": len(all_rows), "crossing": len(crossing),
        "refused": {name: sum(1 for r in all_rows if name in r["refusals"])
                    for name in ("argus_session", "arrival", "utc_clock", "pace")},
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in crossing:
        groups.setdefault(f"{r['exchange']}|{r['horizon']}m", []).append(r)
    report["crossing"] = {
        "all": {"summary": summarise(crossing, arms),
                "argus_session_vs": head_to_head(crossing, "argus_session", ARMS)},
        "by_venue_horizon": {k: {"summary": summarise(v, arms),
                                 "argus_session_vs": head_to_head(v, "argus_session", ARMS)}
                             for k, v in sorted(groups.items())},
        "by_size": {f"{int(s):,}": {"argus_session_vs": head_to_head(
            [r for r in crossing if r["size"] == s], "argus_session", ARMS)} for s in SIZES},
    }
    control = [r for r in all_rows if not r["crosses"]]
    report["not_crossing"] = {"summary": summarise(control, arms),
                              "argus_session_vs": head_to_head(control, "argus_session",
                                                               ("argus_blind", "bitget_twap_60s"))}
    report["ablation"] = {
        "on": "boundary-crossing test parents",
        "full_vs": head_to_head(crossing, "argus_session", ("argus_blind", *ABLATIONS)),
        "holiday_and_dst_days": head_to_head(
            [r for r in all_rows if r["day"] in ("2026-01-01",) or r["crosses"] !=
             r["utc_clock_crosses"]], "argus_session", ("utc_clock", "argus_blind")),
    }
    report["out_of_sample"] = {
        "train_days": {"perp": list(PERP_TRAIN), "spot": list(SPOT_TRAIN)},
        "test_days": {"perp": list(PERP_TEST), "spot": list(SPOT_TEST)},
        "strictly_forward": min(PERP_TEST) > max(PERP_TRAIN) and min(SPOT_TEST) > max(SPOT_TRAIN),
        "dst": dst_experiment(),
    }
    report["pace"] = _pace_section(all_rows, pace_rows)
    report["adversarial"] = adversarial(all_rows)
    report["costs"] = {"fee_bps_per_side": FEE_BPS,
                       "included_in": "every child of every arm, both metrics"}
    report["reproducibility"] = reproducibility(sources, crossing)
    report["failure_cases"] = failure_cases(report, crossing)
    report["limitations"] = [
        "first-of-month days only (Tardis's free tier): five perpetual test days and one spot "
        "test day; a symbol-day block bootstrap is used because parents inside a day share the "
        "price path",
        "taker children only; no passive or maker arm, so zz-0816's maker/taker decision and "
        "PACE's passive setting are not scored",
        "impact beyond the visible top-100 levels and other traders' reaction to our children "
        "are not modelled",
        "PACE runs its Planner only (the paper's 'w/o E' ablation), on ten parents, one call each "
        "(the paper repeats eight times)",
        "TradeMaster ETEO/PD not run: see not_run",
    ]
    report["unavailable_days"] = unavailable()
    report["not_run"] = {
        "TradeMaster ETEO / PD": "needs ray 1.13, tensorflow 2.11 and mmcv 1.7.1 in a py3.9 "
                                 "environment plus training on Bitget 15-level books; the drive "
                                 "held under 2.5 GB free (at times 1.2 MB) during this run, "
                                 "which rules out the install",
        "Bitget Iceberg / Scaled / BBO": "UI-only algorithms with unpublished repricing rules; "
                                         "only the TWAP spec is replayable",
    }
    return report


def _segment_table(models: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for key, blob in models.items():
        segs = blob["segments"]
        out[key] = {s: {"sigma_bps": v["sigma_bps"], "half_spread_bps": v["half_spread_bps"],
                        "impact_bps_per_kusd": v["impact_bps_per_kusd"]}
                    for s, v in segs.items()}
    return out


def dst_experiment() -> dict[str, Any]:
    """Fitted on daylight-time days only; scored on 2025-12-01, when the anchor opens at 14:30 UTC.
    The session clock moves with New York; the UTC clock does not."""
    rows: list[dict[str, Any]] = []
    for symbol in PERPS:
        model = fit(PERP, symbol, DST_TRAIN)
        utc_model = fit(PERP, symbol, DST_TRAIN, segmenter=utc_clock_segment)
        for _, day in _days(PERP, (symbol,), DST_TEST):
            rows.extend(run_day(load_day(PERP, symbol, day), model, utc_model=utc_model))
    affected = [r for r in rows if r["crosses"] or r["utc_clock_crosses"]]
    return {"parents": len(rows), "boundary_parents": len(affected),
            "session_vs_utc_clock": head_to_head(affected, "argus_session",
                                                 ("utc_clock", "argus_blind", "bitget_twap_60s"))}


def _pace_section(rows: Sequence[Mapping[str, Any]],
                  pace_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r["arms"].get("pace")]
    out: dict[str, Any] = {
        "pre_registered_parents": [f"{s}:{d}T{t}:60m:100000:{side}"
                                   for s, d, t, side in PACE_PARENTS],
        "recorded_calls": len(pace_rows),
        "calls_with_errors": sum(1 for r in pace_rows.values() if r.get("error")),
        "usage": [r.get("usage") for r in pace_rows.values()],
        "scored_parents": len(scored),
        "method": "PACE Planner (arXiv 2607.28410 eq. 5, lambda 0.3, 5-minute slots), uniform "
                  "one-minute aggressive children within each slot (the paper's 'w/o E'); "
                  "Qwen 3.8 Max, thinking LOW, temperature 0, one call per parent",
    }
    if scored:
        out["summary"] = summarise(scored, ARMS)
        out["argus_session_vs"] = {
            rival: {"cost_bps": paired(scored, "argus_session", rival, "cost_bps"),
                    "shortfall_bps": paired(scored, "argus_session", rival, "shortfall_bps")}
            for rival in ("pace", "bitget_twap_60s", "argus_blind")}
        out["pace_vs_twap_home_metric"] = paired(scored, "pace", "twap_8", "vs_window_twap_bps")
    return out


def adversarial(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"calendar": calendar_audit()}
    # Placebo and mis-calibration on the perpetual crossing parents of the forward test days.
    placebo_rows: list[dict[str, Any]] = []
    scaled: dict[str, list[dict[str, Any]]] = {"x0.5": [], "x2": []}
    for symbol in PERPS:
        model = fit(PERP, symbol, PERP_TRAIN)
        for day in ("2026-06-01", "2026-07-01", "2026-09-01"):
            d = load_day(PERP, symbol, day)
            placebo_rows.extend(_variant_rows(d, placebo_model(model), model))
            scaled["x0.5"].extend(_variant_rows(d, scaled_model(model, 0.5), model))
            scaled["x2"].extend(_variant_rows(d, scaled_model(model, 2.0), model))
    out["placebo_segments_shuffled"] = head_to_head(placebo_rows, "argus_session",
                                                    ("variant", "bitget_twap_60s"))
    out["miscalibrated"] = {k: head_to_head(v, "argus_session", ("variant", "bitget_twap_60s"))
                            for k, v in scaled.items()}
    out["refusal"] = refusal_probe()
    return out


def _variant_rows(d: Day, variant: SessionModel, model: SessionModel) -> list[dict[str, Any]]:
    grid = books_by_minute(d)
    fee = FEE_BPS[d.exchange]
    out = []
    for parent in parents_for(d):
        if not parent.crosses():
            continue
        try:
            mine = session_children(parent, model, decay=DECAY)
            other = session_children(parent, variant, decay=DECAY)
        except ScheduleError:
            continue
        arms = {"argus_session": execute(grid, parent.start_minute, parent.side, mine, fee,
                                         parent.horizon),
                "variant": execute(grid, parent.start_minute, parent.side, other, fee,
                                   parent.horizon),
                "bitget_twap_60s": execute(grid, parent.start_minute, parent.side,
                                           bitget_twap(parent.size, parent.horizon), fee,
                                           parent.horizon)}
        out.append({"symbol": d.symbol, "day": d.day, "arms": arms})
    return out


def refusal_probe() -> dict[str, Any]:
    """A spot model has never seen a holiday (spot rTokens listed after the last one in its
    training window): a parent on 2026-11-26 must be refused, not solved with borrowed numbers."""
    model = fit(SPOT, "RNVDAUSDT", SPOT_TRAIN)
    probes = {}
    for label, start in (("thanksgiving_2026", datetime(2026, 11, 26, 15, 0, tzinfo=UTC)),
                         ("early_close_2026_11_27", datetime(2026, 11, 27, 17, 30, tzinfo=UTC)),
                         ("ordinary_close_2026_09_01", datetime(2026, 9, 1, 19, 30, tzinfo=UTC))):
        try:
            sched = solve(100_000.0, start, 60, model, decay=DECAY)
            probes[label] = {"refused": False, "share_by_segment": sched.share_by_segment(),
                             "boundaries": [b.as_dict() for b in sched.boundaries]}
        except SessionRefusal as exc:
            probes[label] = {"refused": True, "segment": exc.segment.value, "reason": str(exc)}
    return {"model_segments": sorted(s.value for s in model.params), "probes": probes}


def reproducibility(sources: Mapping[str, Any],
                    crossing: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256(json.dumps(
        [[r["parent"], r["arms"]] for r in crossing], sort_keys=True).encode()).hexdigest()
    return {"deterministic": True, "seed": SEED,
            "source_streams_sha256": {k: {kind: v[kind]["sha256"]
                                          for kind in ("incremental_book_L2", "trades")}
                                      for k, v in sources.items()},
            "crossing_rows_digest": digest,
            "rerun": "python -m argus.eval.session_arena re-streams any uncached day from Tardis "
                     "and must reproduce every source sha256 and crossing_rows_digest"}


def failure_cases(report: Mapping[str, Any],
                  crossing: Sequence[Mapping[str, Any]]) -> list[str]:
    out = []
    versus = report["crossing"]["all"]["argus_session_vs"]
    for rival, block in versus.items():
        if block["verdict"] in ("LOSS", "TRADE-OFF"):
            c = block["cost_bps"]
            d = block["shortfall_dispersion"] or {}
            out.append(f"argus_session {block['verdict']} vs {rival}: cost difference "
                       f"{c['mean_difference_bps']}bps (CI {c['block_bootstrap_ci95_bps']}), "
                       f"shortfall dispersion ratio {d.get('ratio')} (CI "
                       f"{d.get('block_bootstrap_ci95')})")
    refused = [r for r in crossing if "argus_session" in r["refusals"]]
    if refused:
        out.append(f"{len(refused)} crossing parent(s) refused by argus_session: "
                   f"{refused[0]['refusals']['argus_session']}")
    unfilled = [r for r in crossing if r["arms"].get("argus_session") is None
                and "argus_session" not in r["refusals"]]
    if unfilled:
        out.append(f"{len(unfilled)} crossing parent(s) whose session schedule the visible "
                   f"top-{LEVELS} book could not fill")
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pace-live", action="store_true",
                        help="spend the pre-registered PACE Planner calls (at most ten)")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--save", default=str(REPORT_PATH))
    args = parser.parse_args(argv)
    for exchange, symbols, days in ((PERP, PERPS, PERP_TRAIN + PERP_TEST + DST_TEST),
                                    (SPOT, SPOTS, SPOT_TRAIN + SPOT_TEST)):
        for symbol, day in _days(exchange, symbols, sorted(set(days))):
            try:
                d = load_day(exchange, symbol, day)
            except ArenaError as exc:
                if "served no book" not in str(exc):
                    raise
                _mark_unavailable(exchange, symbol, day, str(exc))
                print(f"unavailable {exchange}/{symbol}/{day}: {exc}", flush=True)
                continue
            print(f"cached {d.key}: {len(d.minute)} minutes", flush=True)
    if args.fetch_only:
        return 0
    if args.pace_live:
        spend_pace()
    report = study()
    write_artefact(Path(args.save), report)
    print(json.dumps(report["crossing"]["all"]["argus_session_vs"], indent=1)[:6000])
    return 0


def spend_pace() -> None:  # pragma: no cover - live spend
    done = pace_recorded()
    for symbol, day, clock, side in PACE_PARENTS:
        hour, minute = (int(x) for x in clock.split(":"))
        parent = Parent(PERP, symbol, day, hour * 60 + minute, PACE_HORIZON, PACE_SIZE, side)
        if parent.id in done:
            continue
        d = load_day(PERP, symbol, day)
        prompt = pace_prompt(parent, books_by_minute(d), d.volume)
        row = pace_call(parent, prompt)
        print(f"PACE {parent.id}: {'error ' + row['error'] if row.get('error') else 'ok'}",
              flush=True)


__all__ = [
    "ARMS",
    "Book",
    "Day",
    "Parent",
    "calendar_audit",
    "dispersion",
    "execute",
    "head_to_head",
    "main",
    "pace_children",
    "pace_weights",
    "paired",
    "run_day",
    "sign_test",
    "study",
    "verdict",
    "walk",
    "wilcoxon_p",
]


if __name__ == "__main__":
    raise SystemExit(main())
