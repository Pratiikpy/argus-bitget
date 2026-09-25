"""A US stock's own daily history since listing, and the gaps across the hours it does not trade.

Bitget's stock perpetuals are young — NVDAUSDT has about 400 daily bars — so the weekend record
they carry is some seventy weekends. The company behind each one has traded for decades, and every
Friday-close-to-Monday-open since is an observation of what a closure did to its price. baserate
(Jayanng, a Season 2 desk, proprietary: its idea is used here, none of its code) builds its weekend
risk on 1,227 such episodes per stock from 1999; ARGUS read none of them until 2026-09-25, when the
same leveraged-weekend question was put to both and ARGUS answered a different question.

Prices are Yahoo Finance's daily chart (``/v8/finance/chart``, ``period1=0``, ``interval=1d``),
which is keyless and carries ``adjclose``. A raw gap across a split weekend reads as a 75% crash,
so every open is scaled by that day's ``adjclose / close`` before a gap is taken; dividends move
the factor by a few basis points and are left in.

A closure here is the time from one regular-session close to the next regular-session open with at
least one calendar day in between that had no session: weekends, holidays and the long weekends
they make. ``weekend_only`` keeps the ones that span a Saturday.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

CHART = ("https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
         "?period1=0&period2=9999999999&interval=1d")
CACHE_SECONDS = 6 * 3600.0
MAX_CLOSURE_DAYS = 5
"""The longest real closure is a Friday holiday after a Thursday close, reopening on a Monday
holiday's Tuesday — five calendar days at most. A longer stretch between two bars is a hole in the
data (a halt, a missing row), and counting it as a weekend would invent a gap."""


class HistoryError(RuntimeError):
    """The history could not be read; callers say so rather than answer from nothing."""


@dataclass(frozen=True, slots=True)
class Day:
    day: date
    open: float
    close: float
    """Both split-adjusted (``adjclose`` basis)."""


@dataclass(frozen=True, slots=True)
class Gap:
    closed: date
    reopened: date
    move: float
    """Reopen over the last close, minus one."""


_cache: dict[str, tuple[float, list[Day]]] = {}
_lock = threading.Lock()


def _fetch(ticker: str, *, timeout: float = 20.0) -> dict[str, Any]:
    request = urllib.request.Request(CHART.format(ticker=ticker),
                                     headers={"User-Agent": "Mozilla/5.0 argus-research"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload: dict[str, Any] = json.load(response)
    except Exception as exc:
        raise HistoryError(f"Yahoo daily history for {ticker} did not arrive: "
                           f"{type(exc).__name__}") from exc
    return payload


def parse(payload: dict[str, Any]) -> list[Day]:
    """Adjusted daily bars from a chart payload, oldest first; rows with a missing field dropped."""
    try:
        result = payload["chart"]["result"][0]
        stamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        adj = result["indicators"]["adjclose"][0]["adjclose"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HistoryError(f"unexpected chart shape: {exc}") from exc
    days: list[Day] = []
    for stamp, opened, closed, adjusted in zip(stamps, quote["open"], quote["close"], adj,
                                               strict=False):
        if not opened or not closed or not adjusted or closed <= 0:
            continue
        factor = adjusted / closed
        days.append(Day(day=datetime.fromtimestamp(stamp, tz=UTC).date(),
                        open=opened * factor, close=adjusted))
    return days


def daily(ticker: str, *, fetch: Any = _fetch) -> list[Day]:
    now = time.monotonic()
    with _lock:
        hit = _cache.get(ticker)
        if hit and now - hit[0] < CACHE_SECONDS:
            return hit[1]
    days = parse(fetch(ticker))
    if len(days) < 250:
        raise HistoryError(f"only {len(days)} daily bars for {ticker}")
    with _lock:
        _cache[ticker] = (now, days)
    return days


def closure_gaps(days: list[Day], *, weekend_only: bool = True) -> list[Gap]:
    """Close-to-open moves across every closure longer than a normal overnight."""
    gaps: list[Gap] = []
    for before, after in itertools.pairwise(days):
        span = (after.day - before.day).days
        if span < 2 or span > MAX_CLOSURE_DAYS:
            continue
        spans_saturday = any(
            date.fromordinal(before.day.toordinal() + k).weekday() == 5 for k in range(1, span))
        if weekend_only and not spans_saturday:
            continue
        gaps.append(Gap(closed=before.day, reopened=after.day, move=after.open / before.close - 1))
    return gaps


@dataclass(frozen=True, slots=True)
class GapRecord:
    n: int
    since: date
    up: float
    flat: float
    down_1_5: float
    down_5: float
    worst: Gap
    beyond: int
    """Gaps at or past the stated adverse distance."""

    def as_dict(self) -> dict[str, Any]:
        return {"n": self.n, "since": self.since.isoformat(), "up": self.up, "flat": self.flat,
                "down_1_5": self.down_1_5, "down_5": self.down_5,
                "worst": {"move": self.worst.move, "closed": self.worst.closed.isoformat(),
                          "reopened": self.worst.reopened.isoformat()},
                "beyond": self.beyond}


def record(gaps: list[Gap], *, side: str, adverse: float) -> GapRecord:
    """The closure record from ``side``'s point of view: ``adverse`` is the move against it, as a
    positive fraction, that counts as reached (the liquidation distance)."""
    if not gaps:
        raise HistoryError("no closures to describe")
    sign = -1.0 if side == "short" else 1.0
    moves = [sign * g.move for g in gaps]  # positive = in the position's favour
    n = len(moves)
    worst_index = min(range(n), key=lambda i: moves[i])
    return GapRecord(
        n=n, since=gaps[0].closed,
        up=sum(m > 0.01 for m in moves) / n, flat=sum(abs(m) <= 0.01 for m in moves) / n,
        down_1_5=sum(-0.05 <= m < -0.01 for m in moves) / n,
        down_5=sum(m < -0.05 for m in moves) / n,
        worst=gaps[worst_index], beyond=sum(m <= -adverse for m in moves))
