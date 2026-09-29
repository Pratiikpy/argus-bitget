"""How a stock is positioned outside the perpetual: its listed options and its off-exchange flow.

Two feeds the perception comparison against OpenBB's keyless providers found missing here
(`eval/perception_breadth.py`, 2026-09-28): options chains (`market/options.py`, Cboe) and dark-pool
volume (`market/darkpool.py`, FINRA). They are read together because they answer the same
question a sentiment reader asks next — "is the money that is not on Bitget leaning the same
way?" — and they are fetched in parallel with a short cache, because both are public services and
neither changes faster than every few minutes (Cboe's quotes are delayed; FINRA's weeks are
published two to four weeks late).
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from argus.lui.answer import Source
from argus.lui.trace import trace_module

CACHE_TTL_S = 600.0
MISS_TTL_S = 60.0
"""A read in which any feed came back empty is kept for a minute, not ten: a slow FINRA answer or a
busy pool must not blank that feed for every question asked in the next ten minutes."""
DEADLINE_S = 12.0
PENDING_MAX_AGE_S = DEADLINE_S * 2

_CACHE: dict[str, tuple[float, tuple[list[str], list[Source], dict[str, Any]]]] = {}
_PENDING: dict[str, tuple[float, Future[tuple[list[str], list[Source], dict[str, Any]]]]] = {}
_LOCK = threading.Lock()
_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="positioning")


def _options_line(ticker: str) -> tuple[str, Source, dict[str, Any]] | None:
    from argus.market.options import FLAT_SKEW_VOL_POINTS, OptionsError, options_summary
    from argus.truth import http

    try:
        s = options_summary(ticker)
    except (http.RpcError, OptionsError, ValueError, KeyError):
        return None
    bits = []
    if s.implied_move_pct is not None and s.expiry is not None:
        bits.append(f"the at-the-money straddle prices a {s.implied_move_pct:.1f}% move by "
                    f"{s.expiry.isoformat()}")
    if s.iv30 is not None:
        bits.append(f"30-day implied vol {s.iv30:.1f}%")
    if s.skew_25d is not None:
        if abs(s.skew_25d) < FLAT_SKEW_VOL_POINTS:
            bits.append("puts and calls priced alike at 25 delta")
        else:
            side = "puts" if s.skew_25d > 0 else "calls"
            bits.append(f"{side} cost {abs(s.skew_25d):.1f} vol points more at 25 delta"
                        + (" (downside protection is bid)" if s.skew_25d > 0 else ""))
    if s.put_call_volume is not None and s.put_call_open_interest is not None:
        bits.append(f"put/call {s.put_call_volume:.2f} by today's volume, "
                    f"{s.put_call_open_interest:.2f} by open interest")
    if not bits:
        return None
    line = f"Options on {ticker} (Cboe, delayed): " + "; ".join(bits) + "."
    source = Source(kind="venue", ref="cboe delayed_quotes/options",
                    detail=f"{ticker} listed chain, {s.quoted_contracts} of {s.contracts} "
                           f"contracts two-sided, {s.quoted_at} New York")
    return line, source, {"implied_move_pct": s.implied_move_pct, "iv30": s.iv30,
                          "skew_25d": s.skew_25d, "put_call_volume": s.put_call_volume,
                          "put_call_open_interest": s.put_call_open_interest,
                          "expiry": s.expiry.isoformat() if s.expiry else None}


def _dark_pool_line(ticker: str) -> tuple[str, Source, dict[str, Any]] | None:
    from argus.market.darkpool import DarkPoolError, dark_pool
    from argus.truth import http

    try:
        pool = dark_pool(ticker)
    except (http.RpcError, DarkPoolError, ValueError, KeyError):
        return None
    w = pool.latest
    change = pool.vs_baseline
    tail = ("" if change is None else
            f", {abs(change):.0%} {'above' if change > 0 else 'below'} its previous "
            f"{len(pool.weeks) - 1} weeks per trading day")
    late = f", {w.weeks_late} weeks after the week" if w.weeks_late is not None else ""
    line = (f"Dark pools (FINRA ATS, week of {w.week_start.isoformat()}, first published "
            f"{w.published}{late} — FINRA publishes these weeks late by rule): "
            f"{w.shares / 1e6:.1f}M {ticker} shares traded off-exchange{tail}.")
    source = Source(kind="venue", ref="finra otcMarket weeklySummary",
                    detail=f"{ticker} ATS volume, tier {pool.tier}, {len(pool.weeks)} weeks")
    return line, source, {"week": w.week_start.isoformat(), "shares": w.shares,
                          "vs_baseline": change}


def _short_volume_line(ticker: str) -> tuple[str, Source, dict[str, Any]] | None:
    """FINRA's consolidated daily short volume, which the desk already reads each cycle
    (`market/microstructure.py`) and the workbench did not show until the perception comparison
    counted it missing here (2026-09-29)."""
    from argus.market.microstructure import MicrostructureError, fetch_short_volume

    try:
        got = fetch_short_volume(wanted=frozenset({ticker}), timeout=20).get(ticker)
    except (MicrostructureError, ValueError, KeyError):
        return None
    if got is None or got.total <= 0:
        return None
    line = (f"Short volume (FINRA, {got.as_of.isoformat()}): {got.short_share:.0%} of "
            f"{got.total / 1e6:.1f}M {ticker} shares reported that session were sold short — "
            f"a daily flow share, not short interest, and market makers' hedging sits inside it.")
    source = Source(kind="venue", ref="finra regsho daily short volume",
                    detail=f"{ticker} {got.as_of.isoformat()}")
    return line, source, {"as_of": got.as_of.isoformat(), "short_share": round(got.short_share, 4)}


def _fresh(ticker: str, now: float) -> tuple[list[str], list[Source], dict[str, Any]] | None:
    cached = _CACHE.get(ticker)
    if cached is None:
        return None
    at, result = cached
    complete = all(v is not None for v in result[2].values())
    return result if now - at < (CACHE_TTL_S if complete else MISS_TTL_S) else None


def prefetch(ticker: str) -> None:
    """Start reading the feeds now, so a caller that has other work to do first does not wait for
    them afterwards. The sentiment answer calls this before its own fetches: the feeds took about
    eight seconds end to end when read after them (2026-09-29). A finished or abandoned read left
    by a caller that never collected it is dropped, never served later as if it were new."""
    now = time.monotonic()
    with _LOCK:
        if _fresh(ticker, now) is not None:
            return
        pending = _PENDING.get(ticker)
        if pending is not None and (pending[1].done() or now - pending[0] > PENDING_MAX_AGE_S):
            _PENDING.pop(ticker, None)
            pending = None
        if pending is None:
            _PENDING[ticker] = (now, _POOL.submit(_read, ticker))


def listed_positioning(ticker: str) -> tuple[list[str], list[Source], dict[str, Any]]:
    """The feeds' lines (any may be absent) with their sources and figures."""
    prefetch(ticker)
    with _LOCK:
        hit = _fresh(ticker, time.monotonic())
        if hit is not None:
            return hit
        pending = _PENDING.get(ticker)
    if pending is None:
        result = _read(ticker)
    else:
        try:
            # The deadline runs from when the read was submitted, so a queued read that has not
            # started yet is not given a second full deadline, and is not failed before it runs.
            wait = max(1.0, DEADLINE_S + 1.0 - (time.monotonic() - pending[0]))
            result = pending[1].result(timeout=wait)
        except Exception:  # a failed read leaves the rest of the answer standing
            result = ([], [], {"options": None, "dark_pool": None, "short_volume": None})
    with _LOCK:
        _PENDING.pop(ticker, None)
        _CACHE[ticker] = (time.monotonic(), result)
    return result


def _read(ticker: str) -> tuple[list[str], list[Source], dict[str, Any]]:
    lines: list[str] = []
    sources: list[Source] = []
    data: dict[str, Any] = {}
    # Not a `with` block: leaving one waits for every thread, which would turn the deadline
    # below back into however long the slowest feed takes.
    pool = ThreadPoolExecutor(max_workers=3)
    futures = {"options": pool.submit(_options_line, ticker),
               "dark_pool": pool.submit(_dark_pool_line, ticker),
               "short_volume": pool.submit(_short_volume_line, ticker)}
    deadline = time.monotonic() + DEADLINE_S
    try:
        for name, future in futures.items():
            try:
                got = future.result(timeout=max(0.0, deadline - time.monotonic()))
            except Exception:  # a slow or failed public feed leaves the rest of the answer
                got = None
            if got is None:
                data[name] = None
                continue
            lines.append(got[0])
            sources.append(got[1])
            data[name] = got[2]
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return lines, sources, data


trace_module(globals())
