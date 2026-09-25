"""Deliberation cost, refereed by the market: ARGUS's charge against general-purpose forecasters.

**The function, in general terms.** ``agents/meta_pm.py::deliberation_cost_bps`` answers one
question — *how far will the price move while the model thinks?* — and charges the answer against
the hurdle a trade must clear. Stripped of the trading vocabulary that is **short-horizon
probabilistic forecasting of a series' absolute displacement**: given everything known at the
decision instant, predict ``E|p(t+h) - p(t)|`` for the three delays ARGUS actually runs at
(``THINKING_MS``: 3 s, 8 s, 40 s). The owner's rule of 2026-09-25 is that the best implementation
of a function often lives outside trading, and the strongest open systems for this one are the
time-series foundation models, not trading repos.

**Rivals.** Two kinds of general-purpose forecaster, and only the first has been run to the end.

* *The textbook estimators* — trailing one-second realised volatility under a Gaussian reading
  (``trailing_rv_sqrt``) and the model-free trailing mean absolute move (``trailing_empirical``).
  They carry no licence, need no model, and are scored on every instant of all three books.
* *A time-series foundation model.* GIFT-Eval (Salesforce, 97 dataset configurations; results as
  published on 2026-09-25, geometric mean of CRPS relative to seasonal-naive — this aggregation is
  not reproduced in this repository) put TimesFM-3 0.456, TiRex-2 0.478, Toto-2.0-313m 0.481,
  Timer-S1 0.485, **Chronos-2 0.485**, TiRex 0.488, TimesFM-2.5 0.490, Moirai-2 0.516.
  **Chronos-2** (``amazon-science/chronos-forecasting``, Apache-2.0, 120M parameters) was chosen:
  at the top, permissively licensed weights (TimesFM-3's are non-commercial, TiRex's under the NXAI
  community licence), and a 478 MB checkpoint where TimesFM-3's is 1.32 GB. **Its run is not
  finished.** Zero-shot on CPU it managed 1.7 forecasts a second and was stopped at 512 of the
  23,003 instants; no complete forecast file exists, so :func:`run` reports the rival as
  ``absent`` and every verdict it writes is among the contenders that did run. The hand-off is
  kept so the run can be completed without touching this module: :func:`export_contexts` writes
  each instant's 512-second context plus :func:`instants_digest`, a runner in its own environment
  (torch never enters this package) writes :data:`RIVAL_PATH` with that digest echoed back, and
  :func:`head_to_head` scores it only on a digest match.

**The referee is realised price, not a property of the formula.** The earlier comparison
(``eval/deliberation_comparison.py``) checked a structural property on synthetic delays; a
forecaster cannot even be run there, because it needs the market it is forecasting. Here every
contender gets the same input — a real decision instant on a real Bitget stock perpetual, the
mid-price path up to that instant — and is scored on what the mid actually did over the next
3, 8 and 40 seconds. Three real books:

* ``tape-2026-09-24`` — ARGUS's own forward tape (``market/ws_tape.py``), ``books5`` top of book,
  ten stock perpetuals, 14:57-21:01 UTC: RTH and the post-close extended session.
* ``tardis-2026-09-01`` — Tardis's free first-of-month ``quotes`` (top of book), seven stock
  perpetuals, a full Tuesday: overnight, pre-market, RTH, post-market.
* ``tardis-2026-08-01`` — the same for a Saturday: the weekend book.

**Pre-registered before the first run** (the decision rule is fixed here, not chosen after):

* target: ``|ln mid(t+h) - ln mid(t)|`` in bps; one decision instant per symbol every
  :data:`STRIDE_S` seconds, so outcome windows never overlap;
* primary loss: squared error of the predicted mean absolute move. MSE is consistent for a mean
  and, per Patton (2011, *J. Econometrics* 160:246), ranks forecasts correctly even though the
  realised move is only a noisy proxy for the expected one; MAE, the bias ratio (mean predicted /
  mean realised) and Spearman rank correlation (can it tell a calm instant from a busy one) are
  reported beside it, never instead of it;
* significance: a paired block bootstrap over (book, symbol, UTC hour) blocks, 2,000 resamples,
  fixed seed; a contender "beats" ARGUS at a horizon only when the 95% interval of the MSE
  difference excludes zero;
* Chronos-2's point forecast of ``E|move|`` is the integral of ``|Q(u) - p(t)|`` over its
  predicted quantile function (21 levels, flat beyond 1%/99%), the model-free reading of its own
  output.

Every estimator sees only data at or before ``t``. None is fitted on the evaluation set: ARGUS's
constants predate all three books, Chronos-2 is zero-shot, and the two classical estimators read a
trailing window.

**Reproduce:** ``python -m argus.eval.general_delib_comparison --fetch`` downloads the fourteen
free Tardis files (about 17 MB) and scores every contender on all three books; no model, about two
minutes on one core. The tape book is read from ``data/tape/2026-09-24``.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import random
import zlib
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import Any

from argus.agents.delay_cost import bar_delay_cost_bps, empirical_delay_cost_bps
from argus.agents.meta_pm import DEPTH_MULTIPLIER, THINKING_MS, deliberation_cost_bps
from argus.eval import artefact
from argus.eval.baselines.latencybench_reimpl import linear_decay_price
from argus.execution.latency import thinking_budget_cost_bps
from argus.llm.qwen import Thinking
from argus.truth.clocks import DualClock, SessionPhase

PACKAGE = Path(__file__).resolve().parents[3]
REPORT_PATH = PACKAGE / "data" / "general_delib_comparison.json"
RIVAL_PATH = PACKAGE / "data" / "general_delib_rival_forecasts.json"
TAPE_DIR = PACKAGE / "data" / "tape" / "2026-09-24"
QUOTES_DIR_ENV = "ARGUS_DELIB_QUOTES"
"""Directory holding the Tardis ``quotes`` files; :func:`fetch_tardis` fills it."""

TARDIS_URL = "https://datasets.tardis.dev/v1/bitget-futures/quotes/{y}/{m}/{d}/{symbol}.csv.gz"
TARDIS_DAYS = ("2026-09-01", "2026-08-01")
TARDIS_SYMBOLS = ("NVDA", "TSLA", "AAPL", "META", "MSTR", "COIN", "QQQ")
TAPE_SYMBOLS = ("NVDA", "TSLA", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "COIN", "MSTR", "QQQ")

HORIZONS_S: tuple[int, ...] = tuple(ms // 1000 for ms in THINKING_MS.values())
"""ARGUS's own three deliberation delays, in the order OFF, LOW, FULL: (3, 8, 40)."""
HORIZON_THINKING: dict[int, Thinking] = {ms // 1000: t for t, ms in THINKING_MS.items()}

PRODUCTION_VOL = Decimal("0.45")
"""The annualised volatility `agents/desk.py:318` prices deliberation with in production."""

TRAIL_S = 1800
"""Trailing window the classical estimators read: thirty minutes of one-second mids."""
CONTEXT_S = 512
"""Chronos-2's context: the last 512 one-second mids, ending at the decision instant."""
WARMUP_S = max(TRAIL_S, CONTEXT_S)
STRIDE_S = 60
"""One decision instant per symbol per minute. Longer than the 40 s horizon, so no two outcome
windows share a second."""
TAPE_SILENCE_S = 10
"""The tape carries ~90 messages a second across its symbols; ten seconds with none is an outage,
and any instant whose history or outcome window touches one is dropped rather than scored on a
price that was carried forward through a gap."""

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260925

Quote = tuple[int, float, float]
"""(exchange timestamp in ms, best bid, best ask)."""


class DelibComparisonError(ValueError):
    """The comparison was given data it cannot score honestly."""


# =================================================================================================
# Reading the books.
# =================================================================================================


def salvage_gzip(raw: bytes) -> tuple[bytes, int]:
    """Decompress a multi-member gzip file, skipping past corrupt spans instead of stopping.

    The tape recorder appends one gzip member per (re)open, and on 2026-09-24 two recorder
    processes briefly wrote the same hour file (``15.jsonl.gz``), which leaves an undecodable span
    mid-file. Everything before and after that span is intact; ``gzip.open`` gives up at the first
    bad byte and would silently drop the remaining hour. Returns the bytes recovered and the number
    of corrupt spans skipped, so a caller can say how much was lost.
    """
    out: list[bytes] = []
    errors = 0
    pos = 0
    while pos < len(raw):
        decoder = zlib.decompressobj(31)
        cursor = pos
        try:
            while cursor < len(raw) and not decoder.eof:
                out.append(decoder.decompress(raw[cursor:cursor + 65536]))
                cursor += 65536
        except zlib.error:
            errors += 1
            nxt = raw.find(b"\x1f\x8b\x08", pos + 1)
            if nxt == -1:
                break
            pos = nxt
            continue
        if not decoder.eof:
            break
        pos = len(raw) - len(decoder.unused_data)
    return b"".join(out), errors


@dataclass
class TapeRead:
    """Top-of-book quotes per symbol, plus the seconds in which the tape itself was silent."""

    quotes: dict[str, list[Quote]]
    outages: list[tuple[float, float]]
    corrupt_spans: int
    lines: int


def read_tape(directory: Path = TAPE_DIR, symbols: Sequence[str] = TAPE_SYMBOLS) -> TapeRead:
    """Best bid and ask from every kept ``books5`` snapshot on ARGUS's own forward tape."""
    wanted = {f"{s}USDT": s for s in symbols}
    quotes: dict[str, list[Quote]] = {s: [] for s in symbols}
    arrivals: list[float] = []
    disconnects: list[float] = []
    corrupt = 0
    lines = 0
    for path in sorted(directory.glob("*.jsonl.gz")):
        blob, errors = salvage_gzip(path.read_bytes())
        corrupt += errors
        for line in blob.split(b"\n"):
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            lines += 1
            arrived = float(row.get("t", 0.0))
            arrivals.append(arrived)
            if "argus_event" in row:
                disconnects.append(arrived)
                continue
            message = row.get("m", "")
            if '"books5"' not in message:
                continue
            try:
                parsed = json.loads(message)
                symbol = wanted.get(str(parsed["arg"]["symbol"]))
                if symbol is None:
                    continue
                snap = parsed["data"][0]
                bid = float(snap["b"][0][0])
                ask = float(snap["a"][0][0])
                ts = int(snap["ts"])
            except (ValueError, KeyError, IndexError, TypeError):
                continue
            if 0 < bid < ask:
                quotes[symbol].append((ts, bid, ask))
    for series in quotes.values():
        series.sort()
    arrivals.sort()
    outages = [(a, b) for a, b in pairwise(arrivals) if b - a > TAPE_SILENCE_S]
    outages.extend((t - 1.0, t + TAPE_SILENCE_S) for t in disconnects)
    return TapeRead(quotes=quotes, outages=sorted(outages), corrupt_spans=corrupt, lines=lines)


def read_tardis_quotes(path: Path) -> list[Quote]:
    """Tardis ``quotes`` CSV (timestamps in microseconds) to (ms, bid, ask); crossed rows go."""
    out: list[Quote] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                bid = float(row["bid_price"])
                ask = float(row["ask_price"])
                ts = int(row["timestamp"]) // 1000
            except (KeyError, ValueError):
                continue
            if 0 < bid < ask:
                out.append((ts, bid, ask))
    out.sort()
    return out


def quotes_dir() -> Path:
    configured = os.environ.get(QUOTES_DIR_ENV)
    return Path(configured) if configured else PACKAGE / "data" / "tardis_quotes"


def tardis_path(day: str, symbol: str, directory: Path | None = None) -> Path:
    return (directory or quotes_dir()) / f"{day}_{symbol}USDT.csv.gz"


def fetch_tardis(days: Sequence[str] = TARDIS_DAYS, symbols: Sequence[str] = TARDIS_SYMBOLS,
                 directory: Path | None = None) -> list[Path]:  # pragma: no cover - network
    """Download the free first-of-month files that are not already on disk."""
    import urllib.request

    target = directory or quotes_dir()
    target.mkdir(parents=True, exist_ok=True)
    got: list[Path] = []
    for day in days:
        y, m, d = day.split("-")
        for symbol in symbols:
            path = tardis_path(day, symbol, target)
            if not path.exists():
                url = TARDIS_URL.format(y=y, m=m, d=d, symbol=f"{symbol}USDT")
                with urllib.request.urlopen(url, timeout=120) as response:
                    path.write_bytes(response.read())
            got.append(path)
    return got


# =================================================================================================
# One-second mid grid and decision instants.
# =================================================================================================


def mid_grid(quotes: Sequence[Quote], start_s: int, end_s: int) -> list[float | None]:
    """The mid at each whole second in ``[start_s, end_s]``: the last quote at or before it.

    ``None`` until the first quote. A quote stamped inside a second is visible at the *next* grid
    point's instant only if its timestamp is not after it — nothing from the future leaks in.
    """
    stamps = [q[0] for q in quotes]
    out: list[float | None] = []
    for second in range(start_s, end_s + 1):
        i = bisect_right(stamps, second * 1000) - 1
        out.append(None if i < 0 else (quotes[i][1] + quotes[i][2]) / 2.0)
    return out


def _prefix(values: Sequence[float]) -> list[float]:
    out = [0.0]
    for v in values:
        out.append(out[-1] + v)
    return out


@dataclass(frozen=True)
class Book:
    """One symbol on one day's book, gridded."""

    dataset: str
    symbol: str
    start_s: int
    mids: tuple[float | None, ...]
    outages: tuple[tuple[float, float], ...] = ()

    @cached_property
    def _logs(self) -> list[float]:
        # Missing seconds become 0.0 here; every reader first checks the window is complete.
        return [math.log(v) * 1e4 if v is not None else 0.0 for v in self.mids]

    @cached_property
    def _sq_return_prefix(self) -> list[float]:
        logs = self._logs
        return _prefix([(b - a) ** 2 for a, b in pairwise(logs)])

    @cached_property
    def _abs_move_prefix(self) -> dict[int, list[float]]:
        logs = self._logs
        return {h: _prefix([abs(logs[i + h] - logs[i]) for i in range(len(logs) - h)])
                for h in HORIZONS_S}

    def trailing_sq_return_mean(self, end_second: int, length: int) -> float:
        """Mean squared one-second log return (bps^2) over the ``length`` mids ending at
        ``end_second`` — ``length - 1`` returns."""
        hi = end_second - self.start_s
        lo = hi - length + 1
        prefix = self._sq_return_prefix
        return (prefix[hi] - prefix[lo]) / (length - 1)

    def trailing_abs_move_mean(self, end_second: int, length: int, h: int) -> float:
        """Mean ``|ln mid(s+h) - ln mid(s)|`` (bps) over every ``s`` whose move lies wholly inside
        the ``length`` mids ending at ``end_second``."""
        hi = end_second - self.start_s
        lo = hi - length + 1
        prefix = self._abs_move_prefix[h]
        return (prefix[hi - h + 1] - prefix[lo]) / (length - h)

    def at(self, second: int) -> float | None:
        i = second - self.start_s
        return self.mids[i] if 0 <= i < len(self.mids) else None

    def window(self, end_second: int, length: int) -> list[float] | None:
        """The ``length`` mids ending at ``end_second`` inclusive, or None if any is missing."""
        lo = end_second - length + 1 - self.start_s
        hi = end_second + 1 - self.start_s
        if lo < 0 or hi > len(self.mids):
            return None
        values = self.mids[lo:hi]
        if any(v is None for v in values):
            return None
        return [v for v in values if v is not None]


def book_from_quotes(dataset: str, symbol: str, quotes: Sequence[Quote],
                     outages: Sequence[tuple[float, float]] = ()) -> Book | None:
    if len(quotes) < 2:
        return None
    start = quotes[0][0] // 1000 + 1
    end = quotes[-1][0] // 1000
    if end - start < WARMUP_S + max(HORIZONS_S):
        return None
    return Book(dataset=dataset, symbol=symbol, start_s=start,
                mids=tuple(mid_grid(quotes, start, end)), outages=tuple(outages))


@dataclass(frozen=True)
class Instant:
    """One scored decision: where it was, what was known, and what the mid then did."""

    dataset: str
    symbol: str
    second: int
    phase: SessionPhase
    realised_bps: tuple[float, ...]
    """``|ln mid(t+h) - ln mid(t)|`` in bps, one per horizon in :data:`HORIZONS_S`."""

    @property
    def key(self) -> str:
        return f"{self.dataset}|{self.symbol}|{self.second}"

    @property
    def block(self) -> str:
        return f"{self.dataset}|{self.symbol}|{self.second // 3600}"


def _touches(outages: Sequence[tuple[float, float]], lo: float, hi: float) -> bool:
    return any(a < hi and b > lo for a, b in outages)


def decision_instants(book: Book, clock: DualClock | None = None) -> list[Instant]:
    clock = clock or DualClock()
    horizon = max(HORIZONS_S)
    first = book.start_s + WARMUP_S
    last = book.start_s + len(book.mids) - 1 - horizon
    out: list[Instant] = []
    for second in range(first, last + 1, STRIDE_S):
        if book.outages and _touches(book.outages, second - WARMUP_S, second + horizon):
            continue
        now = book.at(second)
        if now is None or book.window(second, WARMUP_S) is None:
            continue
        realised: list[float] = []
        for h in HORIZONS_S:
            later = book.at(second + h)
            if later is None:
                break
            realised.append(abs(math.log(later / now)) * 1e4)
        if len(realised) != len(HORIZONS_S):
            continue
        phase = clock.phase(datetime.fromtimestamp(second, UTC))
        out.append(Instant(book.dataset, book.symbol, second, phase, tuple(realised)))
    return out


# =================================================================================================
# The contenders. Each returns a predicted E|move| in bps per horizon, from data at or before t.
# =================================================================================================


def argus_production(instant: Instant, clock: DualClock | None = None) -> tuple[float, ...]:
    """ARGUS's real, unmodified charge: ``deliberation_cost_bps`` at the production volatility,
    with the session's depth multiplier, for each of its three thinking budgets."""
    clock = clock or DualClock()
    session = clock.state(datetime.fromtimestamp(instant.second, UTC))
    return tuple(
        float(deliberation_cost_bps(session, thinking=HORIZON_THINKING[h],
                                    annualised_vol=PRODUCTION_VOL))
        for h in HORIZONS_S
    )


def latencybench_linear(book: Book, instant: Instant) -> tuple[float, ...]:
    """LatencySensitiveBench's linear decay toward the average price (the specialist this row was
    first measured against), given its native inputs from the trailing window: its high, low and
    average mid. Its cost at delay ``h`` is how far its delayed high moved, in bps."""
    window = book.window(instant.second, TRAIL_S)
    if window is None:
        raise DelibComparisonError(f"no trailing window for {instant.key}")
    high, low, avg = max(window), min(window), sum(window) / len(window)
    out = []
    for h in HORIZONS_S:
        new_high, _ = linear_decay_price(high, low, float(h), average_price=avg, decay_window=1.5)
        out.append(abs(high - new_high) / high * 1e4)
    return tuple(out)


def _require_trailing(book: Book, instant: Instant) -> None:
    if book.window(instant.second, TRAIL_S) is None:
        raise DelibComparisonError(f"no trailing window for {instant.key}")


def trailing_rv_sqrt(book: Book, instant: Instant) -> tuple[float, ...]:
    """Classical: one-second realised volatility over the trailing window, scaled by ``sqrt(h)``,
    times ``sqrt(2/pi)`` — the mean absolute value of a Gaussian with that standard deviation."""
    _require_trailing(book, instant)
    sigma = math.sqrt(book.trailing_sq_return_mean(instant.second, TRAIL_S))
    return tuple(math.sqrt(2.0 / math.pi) * sigma * math.sqrt(h) for h in HORIZONS_S)


def trailing_empirical(book: Book, instant: Instant) -> tuple[float, ...]:
    """Classical, and assumption-free: the mean absolute ``h``-second move actually seen over the
    trailing window. No scaling law, no distribution — the "naive" forecaster of the literature."""
    _require_trailing(book, instant)
    return tuple(book.trailing_abs_move_mean(instant.second, TRAIL_S, h) for h in HORIZONS_S)


def expected_abs_move_from_quantiles(levels: Sequence[float], values: Sequence[float],
                                     current: float, grid: int = 1000) -> float:
    """``E|X - current|`` from a quantile function given at ``levels``, in the units of ``values``.

    ``E|X - c| = integral_0^1 |Q(u) - c| du``. ``Q`` is linearly interpolated between the given
    levels and held flat beyond the outermost two. The two approximations pull in opposite
    directions: flat tails understate ``|Q|``, while linear interpolation overstates it wherever
    ``Q`` is convex above ``c`` or concave below it, as a bell-shaped forecast is. On an exact
    standard normal at Chronos-2's 21 levels (1%, 5%..95%, 99%) the net is a +0.25% overstatement
    (``tests/test_general_delib_comparison.py`` pins it) — small beside the differences the
    comparison scores, and stated so it is not mistaken for a conservative reading. Midpoint rule
    on ``grid`` points.
    """
    if len(levels) != len(values) or len(levels) < 2:
        raise DelibComparisonError("need at least two matching quantile levels and values")
    pairs = sorted(zip(levels, values, strict=True))
    us = [p[0] for p in pairs]
    qs = sorted(p[1] for p in pairs)   # a quantile function is monotone; repair any crossing
    total = 0.0
    for k in range(grid):
        u = (k + 0.5) / grid
        if u <= us[0]:
            q = qs[0]
        elif u >= us[-1]:
            q = qs[-1]
        else:
            j = bisect_right(us, u) - 1
            frac = (u - us[j]) / (us[j + 1] - us[j])
            q = qs[j] + frac * (qs[j + 1] - qs[j])
        total += abs(q - current)
    return total / grid


def chronos_context(book: Book, instant: Instant) -> list[float]:
    """The exact input Chronos-2 receives: the last :data:`CONTEXT_S` one-second mids."""
    window = book.window(instant.second, CONTEXT_S)
    if window is None:
        raise DelibComparisonError(f"no context for {instant.key}")
    return window


def export_contexts(data: Dataset, directory: Path) -> dict[str, Any]:  # pragma: no cover - data
    """Write every instant's Chronos-2 input for the rival runner, which runs in its own venv.

    ``contexts.f32`` is ``N x CONTEXT_S`` little-endian float32 mids, row order as ``index.json``
    lists the keys. The runner echoes :func:`instants_digest` back so a forecast file can never be
    scored against instants it was not made for.
    """
    from array import array

    books = {(b.dataset, b.symbol): b for b in data.books}
    flat = array("f")
    for instant in data.instants:
        flat.extend(chronos_context(books[(instant.dataset, instant.symbol)], instant))
    if flat.itemsize != 4:
        raise DelibComparisonError("platform float is not 32-bit; cannot write the context file")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "contexts.f32").write_bytes(flat.tobytes())
    index = {
        "instants_sha256": instants_digest(data.instants),
        "context_s": CONTEXT_S,
        "horizons_s": list(HORIZONS_S),
        "keys": [i.key for i in data.instants],
    }
    (directory / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return {k: v for k, v in index.items() if k != "keys"} | {"n": len(data.instants)}


def load_rival(path: Path = RIVAL_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    blob: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return blob


# =================================================================================================
# Scoring.
# =================================================================================================


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return out

    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


@dataclass(frozen=True)
class Score:
    n: int
    mse: float
    mae: float
    mean_predicted: float
    mean_realised: float
    spearman: float | None

    @property
    def bias_ratio(self) -> float:
        return self.mean_predicted / self.mean_realised if self.mean_realised else math.inf

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n, "mse": round(self.mse, 4), "rmse": round(math.sqrt(self.mse), 4),
            "mae": round(self.mae, 4), "mean_predicted_bps": round(self.mean_predicted, 4),
            "mean_realised_bps": round(self.mean_realised, 4),
            "bias_ratio": round(self.bias_ratio, 4) if math.isfinite(self.bias_ratio) else None,
            "spearman": None if self.spearman is None else round(self.spearman, 4),
        }


def score(predicted: Sequence[float], realised: Sequence[float]) -> Score:
    if not predicted or len(predicted) != len(realised):
        raise DelibComparisonError("score needs matching, non-empty sequences")
    n = len(predicted)
    return Score(
        n=n,
        mse=sum((p - y) ** 2 for p, y in zip(predicted, realised, strict=True)) / n,
        mae=sum(abs(p - y) for p, y in zip(predicted, realised, strict=True)) / n,
        mean_predicted=sum(predicted) / n,
        mean_realised=sum(realised) / n,
        spearman=_spearman(predicted, realised),
    )


def block_bootstrap_mse_difference(
    blocks: Sequence[str], baseline: Sequence[float], rival: Sequence[float],
    realised: Sequence[float], *, resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Paired block bootstrap of ``MSE(baseline) - MSE(rival)``: positive means the rival's
    squared error is smaller. Blocks are resampled whole, so intra-hour dependence is kept."""
    by_block: dict[str, list[float]] = {}
    for b, base, riv, y in zip(blocks, baseline, rival, realised, strict=True):
        by_block.setdefault(b, []).append((base - y) ** 2 - (riv - y) ** 2)
    keys = sorted(by_block)
    sums = [sum(by_block[k]) for k in keys]
    counts = [len(by_block[k]) for k in keys]
    point = sum(sums) / sum(counts)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        total = 0.0
        count = 0
        for _ in keys:
            j = rng.randrange(len(keys))
            total += sums[j]
            count += counts[j]
        draws.append(total / count)
    draws.sort()
    lo = draws[int(0.025 * resamples)]
    hi = draws[int(0.975 * resamples) - 1]
    return {"point": point, "ci95_low": lo, "ci95_high": hi, "blocks": float(len(keys))}


# =================================================================================================
# The run.
# =================================================================================================


@dataclass
class Dataset:
    books: list[Book] = field(default_factory=list)
    instants: list[Instant] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dataset(tape_dir: Path = TAPE_DIR, quotes_directory: Path | None = None,
                  days: Sequence[str] = TARDIS_DAYS,
                  symbols: Sequence[str] = TARDIS_SYMBOLS) -> Dataset:  # pragma: no cover - data
    data = Dataset()
    clock = DualClock()
    tape = read_tape(tape_dir)
    data.provenance["tape"] = {
        "directory": str(tape_dir.relative_to(PACKAGE)).replace("\\", "/"),
        "files": {p.name: _sha256(p) for p in sorted(tape_dir.glob("*.jsonl.gz"))},
        "lines": tape.lines, "corrupt_gzip_spans_skipped": tape.corrupt_spans,
        "outages": [[round(a, 3), round(b, 3)] for a, b in tape.outages],
    }
    for symbol, quotes in tape.quotes.items():
        book = book_from_quotes(f"tape-{tape_dir.name}", symbol, quotes, tape.outages)
        if book is not None:
            data.books.append(book)
    directory = quotes_directory or quotes_dir()
    tardis: dict[str, str] = {}
    for day in days:
        for symbol in symbols:
            path = tardis_path(day, symbol, directory)
            if not path.exists():
                continue
            tardis[path.name] = _sha256(path)
            book = book_from_quotes(f"tardis-{day}", symbol, read_tardis_quotes(path))
            if book is not None:
                data.books.append(book)
    data.provenance["tardis"] = {"url": TARDIS_URL, "files": tardis}
    for book in data.books:
        data.instants.extend(decision_instants(book, clock))
    return data


def instants_digest(instants: Iterable[Instant]) -> str:
    """A hash of every scored instant's identity and outcome — the rival file must match it."""
    h = hashlib.sha256()
    for i in instants:
        h.update(f"{i.key}:{','.join(f'{v:.6f}' for v in i.realised_bps)};".encode())
    return h.hexdigest()


CONTENDERS = (
    "argus_production", "argus_adapted", "argus_adapted_bars",
    "trailing_rv_sqrt", "latencybench_linear",
)
"""``argus_production`` is the charge the desk runs today. ``argus_adapted`` and
``argus_adapted_bars`` are `agents/delay_cost.py`, the classical estimators the comparison found
better, adopted: the first on one-second mids, the second on one-minute bars (what the desk can
fetch as Bitget candles). ``trailing_rv_sqrt`` is the Gaussian reading of one-second realised
volatility. ``chronos2`` joins wherever the rival file covers the instant."""

ABLATIONS = ("ablation_argus_without_depth", "ablation_measured_vol_with_depth")
"""Which part of ARGUS's charge loses: the session multiplier, or the constant volatility? The
first keeps 0.45 and drops the multiplier; the second keeps the multiplier on a measured
volatility."""

BAR_S = 60


def argus_adapted_bars(book: Book, instant: Instant) -> tuple[float, ...]:
    """`delay_cost.bar_delay_cost_bps` on the one-minute closes of the trailing window."""
    window = book.window(instant.second, TRAIL_S + 1)
    if window is None:
        raise DelibComparisonError(f"no trailing window for {instant.key}")
    closes = window[::-1][::BAR_S][::-1]
    return tuple(float(bar_delay_cost_bps(closes, bar_s=BAR_S, delay_s=h)) for h in HORIZONS_S)


def predictions(data: Dataset) -> dict[str, list[tuple[float, ...]]]:
    books = {(b.dataset, b.symbol): b for b in data.books}
    clock = DualClock()
    names = CONTENDERS + ABLATIONS
    out: dict[str, list[tuple[float, ...]]] = {name: [] for name in names}
    no_depth = [
        float(thinking_budget_cost_bps(think_ms=h * 1000, annualised_vol=PRODUCTION_VOL))
        for h in HORIZONS_S
    ]
    for instant in data.instants:
        book = books[(instant.dataset, instant.symbol)]
        session = clock.state(datetime.fromtimestamp(instant.second, UTC))
        depth = float(DEPTH_MULTIPLIER.get(session.phase, Decimal("3")))
        rv = trailing_rv_sqrt(book, instant)
        out["argus_production"].append(argus_production(instant, clock))
        out["argus_adapted"].append(trailing_empirical(book, instant))
        out["argus_adapted_bars"].append(argus_adapted_bars(book, instant))
        out["trailing_rv_sqrt"].append(rv)
        out["latencybench_linear"].append(latencybench_linear(book, instant))
        out["ablation_argus_without_depth"].append(tuple(no_depth))
        out["ablation_measured_vol_with_depth"].append(tuple(v * depth for v in rv))
    return out


def adapted_parity(data: Dataset, preds: Mapping[str, Sequence[tuple[float, ...]]],
                   every: int = 97) -> dict[str, Any]:
    """``argus_adapted`` is computed from prefix sums for speed; this recomputes a deterministic
    sample of it with `agents/delay_cost.py` itself and records the largest difference, so the
    scored numbers are provably the shipped function's."""
    books = {(b.dataset, b.symbol): b for b in data.books}
    worst = 0.0
    checked = 0
    for j in range(0, len(data.instants), every):
        instant = data.instants[j]
        window = books[(instant.dataset, instant.symbol)].window(instant.second, TRAIL_S)
        if window is None:
            continue
        for k, h in enumerate(HORIZONS_S):
            shipped = float(empirical_delay_cost_bps(window, delay_s=h))
            worst = max(worst, abs(shipped - preds["argus_adapted"][j][k]))
        checked += 1
    return {"instants_checked": checked, "max_abs_diff_bps": worst,
            "matches": worst < 1e-6}


def evaluate(instants: Sequence[Instant], preds: Mapping[str, Sequence[tuple[float, ...]]],
             resamples: int = BOOTSTRAP_RESAMPLES, reference: str = "argus_production",
             by_phase: bool = True) -> dict[str, Any]:
    """Every contender, every horizon, pooled and by session phase, each paired against
    ``reference`` with the block bootstrap."""
    realised = [i.realised_bps for i in instants]
    blocks = [i.block for i in instants]
    phases = sorted({i.phase.value for i in instants})
    report: dict[str, Any] = {"n": len(instants), "reference": reference, "by_horizon": {}}
    for k, h in enumerate(HORIZONS_S):
        ys = [r[k] for r in realised]
        entry: dict[str, Any] = {"pooled": {}, "vs_reference": {}}
        for name, rows in preds.items():
            entry["pooled"][name] = score([r[k] for r in rows], ys).as_dict()
        if by_phase:
            entry["by_phase"] = {}
            for phase in phases:
                idx = [j for j, i in enumerate(instants) if i.phase.value == phase]
                entry["by_phase"][phase] = {
                    name: score([rows[j][k] for j in idx], [ys[j] for j in idx]).as_dict()
                    for name, rows in preds.items()
                }
        base = [r[k] for r in preds[reference]]
        for name, rows in preds.items():
            if name == reference:
                continue
            diff = block_bootstrap_mse_difference(blocks, base, [r[k] for r in rows], ys,
                                                  resamples=resamples)
            entry["vs_reference"][name] = {
                **{key: round(v, 4) for key, v in diff.items()},
                "contender_beats_reference": diff["ci95_low"] > 0,
                "reference_beats_contender": diff["ci95_high"] < 0,
            }
        report["by_horizon"][f"{h}s"] = entry
    return report


def _subset(instants: Sequence[Instant], preds: Mapping[str, Sequence[tuple[float, ...]]],
            keep: Sequence[int]) -> tuple[list[Instant], dict[str, list[tuple[float, ...]]]]:
    return [instants[j] for j in keep], {n: [rows[j] for j in keep] for n, rows in preds.items()}


def head_to_head(data: Dataset, preds: Mapping[str, Sequence[tuple[float, ...]]],
                 rival: Mapping[str, Any], resamples: int = BOOTSTRAP_RESAMPLES,
                 ) -> dict[str, Any]:
    """Every contender and Chronos-2 on exactly the instants the rival file covers."""
    forecasts: Mapping[str, Sequence[float]] = rival["forecasts"]
    keep = [j for j, i in enumerate(data.instants) if i.key in forecasts]
    instants, sub = _subset(data.instants, preds, keep)
    sub["chronos2"] = [tuple(float(v) for v in forecasts[i.key]) for i in instants]
    return {
        "vs_argus_production": evaluate(instants, sub, resamples),
        "vs_argus_adapted": evaluate(instants, sub, resamples, reference="argus_adapted",
                                     by_phase=False),
        "chronos2_worst_overshoot_bps": {
            f"{h}s": round(max(r[k] - i.realised_bps[k]
                               for r, i in zip(sub["chronos2"], instants, strict=True)), 4)
            for k, h in enumerate(HORIZONS_S)
        },
    }


def per_book(data: Dataset, preds: Mapping[str, Sequence[tuple[float, ...]]],
             names: Sequence[str] = CONTENDERS) -> dict[str, Any]:
    """Pooled MSE per book: three separate days, none used to choose anything, so a ranking that
    holds on each is not an artefact of one of them."""
    out: dict[str, Any] = {}
    for dataset in sorted({i.dataset for i in data.instants}):
        keep = [j for j, i in enumerate(data.instants) if i.dataset == dataset]
        instants, sub = _subset(data.instants, preds, keep)
        out[dataset] = {
            f"{h}s": {n: round(score([r[k] for r in sub[n]],
                                     [i.realised_bps[k] for i in instants]).mse, 4)
                      for n in names}
            for k, h in enumerate(HORIZONS_S)
        }
    return out


def after_a_jump(data: Dataset, preds: Mapping[str, Sequence[tuple[float, ...]]],
                 names: Sequence[str] = CONTENDERS, lookback_s: int = 60) -> dict[str, Any]:
    """The adversarial case for any trailing estimator: the instant right after the book lurched.

    Selected on information at ``t`` only — the absolute move over the preceding minute is in the
    top decile of its book — so the subset is not conditioned on the outcome being scored. A
    thirty-minute window reacts slowly to a regime change; this is where it should lose if it is
    going to.
    """
    books = {(b.dataset, b.symbol): b for b in data.books}
    jumps: list[float] = []
    for i in data.instants:
        book = books[(i.dataset, i.symbol)]
        now, before = book.at(i.second), book.at(i.second - lookback_s)
        jumps.append(abs(math.log(now / before)) * 1e4 if now and before else 0.0)
    keep: list[int] = []
    for dataset, symbol in books:
        idx = [j for j, i in enumerate(data.instants)
               if i.dataset == dataset and i.symbol == symbol]
        if len(idx) < 10:
            continue
        cut = sorted(jumps[j] for j in idx)[int(0.9 * len(idx))]
        keep.extend(j for j in idx if jumps[j] >= cut and jumps[j] > 0)
    keep.sort()
    instants, sub = _subset(data.instants, preds, keep)
    return {
        "selection": f"|move over the previous {lookback_s}s| in the top decile of its book",
        "n": len(keep),
        "mse": {
            f"{h}s": {n: round(score([r[k] for r in sub[n]],
                                     [i.realised_bps[k] for i in instants]).mse, 4)
                      for n in names}
            for k, h in enumerate(HORIZONS_S)
        },
    }


def split_halves(data: Dataset, preds: Mapping[str, Sequence[tuple[float, ...]]],
                 names: Sequence[str] = CONTENDERS) -> dict[str, Any]:
    """Each book cut at its own midpoint in time: the ranking in the first half and the second."""
    halves: dict[str, list[int]] = {"first_half": [], "second_half": []}
    for dataset, symbol in {(i.dataset, i.symbol) for i in data.instants}:
        idx = [j for j, i in enumerate(data.instants)
               if i.dataset == dataset and i.symbol == symbol]
        halves["first_half"].extend(idx[:len(idx) // 2])
        halves["second_half"].extend(idx[len(idx) // 2:])
    out: dict[str, Any] = {}
    for label, keep in halves.items():
        instants, sub = _subset(data.instants, preds, sorted(keep))
        out[label] = {
            f"{h}s": {n: round(score([r[k] for r in sub[n]],
                                     [i.realised_bps[k] for i in instants]).mse, 4)
                      for n in names}
            for k, h in enumerate(HORIZONS_S)
        }
    return out


def depth_multiplier_check(data: Dataset) -> dict[str, Any]:
    """ARGUS multiplies the charge by 2 in extended hours and 3 overnight and at weekends. The
    realised ratio of mean absolute move to RTH, on the same symbols, is the direct test of that."""
    out: dict[str, Any] = {}
    for k, h in enumerate(HORIZONS_S):
        per_phase: dict[str, list[float]] = {}
        for i in data.instants:
            per_phase.setdefault(i.phase.value, []).append(i.realised_bps[k])
        rth = per_phase.get(SessionPhase.RTH.value)
        base = sum(rth) / len(rth) if rth else None
        out[f"{h}s"] = {
            phase: {
                "n": len(v), "mean_realised_bps": round(sum(v) / len(v), 4),
                "realised_ratio_to_rth": (round(sum(v) / len(v) / base, 4) if base else None),
                "argus_multiplier": float(DEPTH_MULTIPLIER.get(SessionPhase(phase), Decimal(3))),
            }
            for phase, v in sorted(per_phase.items())
        }
    return out


def hurdle_effect(data: Dataset, preds: Mapping[str, Sequence[tuple[float, ...]]],
                  round_trip_bps: float = 12.0) -> dict[str, Any]:
    """What the charge does to the decision: the FULL-budget hurdle (fee + deliberation) the PM is
    told to clear, per session, under today's charge and the adapted one, beside the move the
    market actually made over those 40 seconds."""
    k = HORIZONS_S.index(max(HORIZONS_S))
    out: dict[str, Any] = {"round_trip_bps": round_trip_bps}
    for phase in sorted({i.phase.value for i in data.instants}):
        idx = [j for j, i in enumerate(data.instants) if i.phase.value == phase]
        mean = {n: sum(preds[n][j][k] for j in idx) / len(idx)
                for n in ("argus_production", "argus_adapted")}
        realised = sum(data.instants[j].realised_bps[k] for j in idx) / len(idx)
        out[phase] = {
            "n": len(idx),
            "hurdle_today_bps": round(round_trip_bps + mean["argus_production"], 3),
            "hurdle_adapted_bps": round(round_trip_bps + mean["argus_adapted"], 3),
            "realised_mean_40s_move_bps": round(realised, 3),
            "overcharge_today_bps": round(mean["argus_production"] - realised, 3),
        }
    return out


def verdict(report: Mapping[str, Any], reference: str = "argus_production") -> dict[str, str]:
    """One line per horizon: who has the lowest pooled MSE, and who beat the reference."""
    out: dict[str, str] = {}
    for label, entry in report["by_horizon"].items():
        pooled = entry["pooled"]
        best = min(pooled, key=lambda n: pooled[n]["mse"])
        beaten = [n for n, d in entry["vs_reference"].items() if d["contender_beats_reference"]]
        out[label] = (
            f"lowest MSE: {best} ({pooled[best]['mse']:.3f} vs {reference} "
            f"{pooled[reference]['mse']:.3f}); significantly better than {reference}: "
            f"{', '.join(beaten) if beaten else 'none'}"
        )
    return out


def reproducibility(data: Dataset) -> dict[str, Any]:
    """Every contender here is a pure function of the book; recompute the whole prediction table
    and compare its hash with the first pass."""
    def digest(preds: Mapping[str, Sequence[tuple[float, ...]]]) -> str:
        return hashlib.sha256(json.dumps(
            {n: [[round(v, 9) for v in row] for row in rows] for n, rows in preds.items()},
            sort_keys=True).encode()).hexdigest()

    first, second = digest(predictions(data)), digest(predictions(data))
    return {"first_sha256": first, "second_sha256": second, "identical": first == second}


def run(resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:  # pragma: no cover - data
    data = build_dataset()
    digest = instants_digest(data.instants)
    rival = load_rival()
    rival_status = "absent"
    if rival is not None:
        rival_status = "matched" if rival.get("instants_sha256") == digest else "stale"
        if rival_status == "stale":
            rival = None
    preds = predictions(data)
    full = evaluate(data.instants, preds, resamples)
    counts: dict[str, int] = {}
    for i in data.instants:
        label = f"{i.dataset}/{i.phase.value}"
        counts[label] = counts.get(label, 0) + 1
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "question": "E|ln mid(t+h) - ln mid(t)| in bps at ARGUS's three thinking delays",
        "horizons_s": list(HORIZONS_S),
        "production_vol": str(PRODUCTION_VOL),
        "trail_s": TRAIL_S, "context_s": CONTEXT_S, "stride_s": STRIDE_S,
        "instants": len(data.instants), "instants_by_book_phase": counts,
        "instants_sha256": digest,
        "provenance": data.provenance,
        "rival": {
            "status": rival_status,
            **({k: v for k, v in rival.items()
                if k not in ("forecasts", "raw_quantile_sample_log_bps")} if rival else {}),
        },
        "full_comparison": full,
        "full_verdict": verdict(full),
        "adapted_parity": adapted_parity(data, preds),
        "per_book_out_of_sample": per_book(data, preds),
        "split_halves": split_halves(data, preds),
        "adversarial_after_a_jump": after_a_jump(data, preds),
        "ablation": {
            f"{h}s": {n: full["by_horizon"][f"{h}s"]["pooled"][n]
                      for n in ("argus_production", *ABLATIONS)}
            for h in HORIZONS_S
        },
        "depth_multiplier_check": depth_multiplier_check(data),
        "hurdle_effect_full_budget": hurdle_effect(data, preds),
        "reproducibility": reproducibility(data),
    }
    if rival is not None:
        h2h = head_to_head(data, preds, rival, resamples)
        report["head_to_head_chronos2"] = h2h
        report["head_to_head_verdict_vs_production"] = verdict(h2h["vs_argus_production"])
        report["head_to_head_verdict_vs_adapted"] = verdict(h2h["vs_argus_adapted"],
                                                            reference="argus_adapted")
        report["rival"]["raw_quantile_sample_log_bps"] = dict(
            list(rival.get("raw_quantile_sample_log_bps", {}).items())[:5])
    return report


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--fetch", action="store_true",
                        help="download the free Tardis first-of-month quotes files first "
                             f"(into ${QUOTES_DIR_ENV}, default data/tardis_quotes)")
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)
    if args.fetch:
        fetch_tardis()
    report = run(args.resamples)
    artefact.write(REPORT_PATH, report)
    for label, line in report["full_verdict"].items():
        print(f"full {label}: {line}")
    for label, line in report.get("head_to_head_verdict_vs_adapted", {}).items():
        print(f"h2h  {label}: {line}")
    print(f"rival: {report['rival']['status']}; instants: {report['instants']}")
    print(f"saved -> {REPORT_PATH}")
    return 0


__all__ = [
    "ABLATIONS",
    "CONTENDERS",
    "HORIZONS_S",
    "Book",
    "DelibComparisonError",
    "Instant",
    "Score",
    "after_a_jump",
    "argus_adapted_bars",
    "argus_production",
    "block_bootstrap_mse_difference",
    "book_from_quotes",
    "build_dataset",
    "chronos_context",
    "decision_instants",
    "depth_multiplier_check",
    "evaluate",
    "expected_abs_move_from_quantiles",
    "head_to_head",
    "hurdle_effect",
    "instants_digest",
    "latencybench_linear",
    "mid_grid",
    "predictions",
    "read_tape",
    "read_tardis_quotes",
    "salvage_gzip",
    "score",
    "trailing_empirical",
    "trailing_rv_sqrt",
    "verdict",
]


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
