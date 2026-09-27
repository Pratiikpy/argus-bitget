"""Candles for the answers: live Bitget first, under a deadline, else the frozen fixture — never a
mix."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from argus.desk.portfolio import (
    PortfolioError,
    returns,
)
from argus.lui.answer import Source
from argus.lui.research.kinds import (
    BENCHMARK,
    CACHE_TTL_S,
    CRYPTO_ANCHOR,
    CRYPTO_LINKED,
    FETCH_DEADLINE_S,
    FIXTURE_PATH,
    LOOKBACK_DAYS,
)
from argus.lui.trace import trace_module
from argus.truth.bounded import BoundedDict
from argus.truth.coverage import ContextPool

# --- data -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketData:
    """Per-bar returns for the requested names, and an honest account of where they came from."""

    raw: Mapping[str, Mapping[datetime, float]]
    provenance: str
    source: Source
    live: bool


_SERIES_CACHE: BoundedDict[tuple[str, int], tuple[float, dict[datetime, float]]] = BoundedDict(512)


_CACHE_LOCK = threading.Lock()


_FETCH_SLOTS = threading.Semaphore(3)
"""At most three candle requests in flight. Bitget's history endpoint returns HTTP 429 when a
burst arrives at once — measured on the first run of this module, when eight concurrent fetches
sent a two-name question to the frozen fallback — so concurrency is capped rather than maximised."""


def _candles(symbol: str, days: int) -> list[Any]:
    """Hourly candles covering ``days``, in as few venue calls as the venue allows.

    Up to 1,000 bars (41 days) is one call to the recent-candles endpoint; history-candles pages at
    100, so the same thirty days used to be eight sequential calls — measured 4.2s per name on
    2026-09-23, which put a three-name question past :data:`FETCH_DEADLINE_S` and sent a BTC
    impact question to a frozen file that has no BTC in it. Longer ranges fall back to paging.
    """
    from argus.market.history import RECENT_LIMIT, CandleType, fetch, fetch_range

    with _FETCH_SLOTS:
        if days * 24 <= RECENT_LIMIT:
            return fetch(symbol, interval="1H", candle_type=CandleType.MARKET, recent=True,
                         limit=days * 24)
        return fetch_range(symbol, days=days, interval="1H", candle_type=CandleType.MARKET,
                           pause=0.1)


def _fetch_one(symbol: str, days: int, deadline: float) -> dict[datetime, float]:
    now = time.monotonic()
    with _CACHE_LOCK:
        hit = _SERIES_CACHE.get((symbol, days))
        if hit and now - hit[0] < CACHE_TTL_S:
            return hit[1]
    wait = 0.5
    while True:
        try:
            candles = _candles(symbol, days)
            break
        except Exception as exc:
            if "429" not in str(exc) or time.monotonic() + wait > deadline:
                raise
            time.sleep(wait)
            wait *= 2
    series = returns([(c.ts, float(c.close)) for c in candles])
    if len(series) < 50:
        raise PortfolioError(f"{symbol}: only {len(series)} bars came back")
    with _CACHE_LOCK:
        _SERIES_CACHE[(symbol, days)] = (time.monotonic(), series)
    return series


def _fetch_live(symbols: Sequence[str], days: int) -> dict[str, dict[datetime, float]]:
    deadline = time.monotonic() + FETCH_DEADLINE_S
    out: dict[str, dict[datetime, float]] = {}
    pool = ContextPool(max_workers=min(4, len(symbols)))
    try:
        futures = {pool.submit(_fetch_one, s, days, deadline): s for s in symbols}
        for future in as_completed(futures, timeout=FETCH_DEADLINE_S):
            out[futures[future]] = future.result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return out


def _from_fixture(symbols: Sequence[str], days: int) -> tuple[dict[str, dict[datetime, float]],
                                                               str]:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    candles = fixture.get("candles", {})
    out: dict[str, dict[datetime, float]] = {}
    for symbol in symbols:
        rows = candles.get(symbol)
        if not rows:
            raise PortfolioError(f"{symbol} is not in the frozen history either")
        prices = [(datetime.fromisoformat(ts), float(close)) for ts, close in rows]
        cutoff = prices[-1][0] - timedelta(days=days)
        out[symbol] = returns([p for p in prices if p[0] >= cutoff])
    return out, str(fixture.get("generated_at", "an unrecorded date"))[:10]


def load(symbols: Sequence[str], *, days: int = LOOKBACK_DAYS) -> MarketData:
    """Returns for ``symbols`` plus the benchmark: live if every name arrives in time, else frozen.

    All-or-nothing on purpose — see the module docstring on why live and frozen series are never
    mixed in one answer.
    """
    wanted = tuple(sorted({*symbols, BENCHMARK}))
    anchored = tuple(sorted({*wanted, CRYPTO_ANCHOR})) if CRYPTO_LINKED & set(symbols) else wanted
    try:
        try:
            raw = _fetch_live(anchored, days)
            wanted = anchored
        except Exception:
            if anchored == wanted:
                raise
            # BTC is context for a crypto-linked name, not what was asked: without it the answer
            # loses one line, not the whole risk view.
            raw = _fetch_live(wanted, days)
        data = MarketData(
            raw=raw, live=True,
            provenance=f"live Bitget hourly candles, last {days} days, fetched just now",
            source=Source(kind="venue", ref="bitget /api/v3/market/candles",
                          detail=f"{', '.join(wanted)}; {days}d hourly; live"),
        )
    except Exception as exc:
        raw, frozen_on = _from_fixture(wanted, days)
        data = MarketData(
            raw=raw, live=False,
            provenance=(
                f"Bitget hourly candles frozen on {frozen_on} (the live fetch did not complete: "
                f"{type(exc).__name__}) — the figures are real, but not as of this minute"
            ),
            source=Source(kind="computation", ref="data/risk_layer_candles_fixture.json",
                          detail=f"real Bitget history frozen {frozen_on}; {days}d window"),
        )
    return data


# Every engine here is a traced step from import on (lui/trace.py, trace_module).
trace_module(globals())
