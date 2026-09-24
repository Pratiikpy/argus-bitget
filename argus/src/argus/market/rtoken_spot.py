"""Bitget's spot rTokens, their same-company stock perpetuals, and the nights between sessions.

**What this module corrects.** Everywhere else in ARGUS the instrument is the stock perpetual
(``TSLAUSDT``, USDT-margined, what `market/history` reads). Bitget's **rTokens** are a different
product: spot tokens (``RTSLAUSDT``, base coin ``rTSLA``) that a holder owns outright. A holder of
``RTSLAUSDT`` asking how to hedge the weekend was answered with an index leg that removed 10% of
the variance, because ARGUS did not know the spot token existed and so could not name the one
instrument that removes 99.7% of it: the same company's perpetual (`eval/copilot_hedge.py`).

**Taken from Ballast** (Ritapossible/Ballast, an S2 entry, MIT, read at @5cf6759), whose hedge this
module lets ARGUS offer. Vendored with its notice, and each piece named where it came from:

* the pairing rule — an rToken is a spot symbol whose ``baseCoin`` matches ``^r[A-Z]``, and its
  perp leg must be reported ``symbolType == "stock"`` by the v3 instruments endpoint, because 34
  stock tickers collide with crypto perps (``rF`` is Ford; ``FUSDT`` is a crypto coin)
  (``ballast/universe.py:1-80``);
* the granularity casing — spot takes ``1h``, futures ``1H``, and the wrong case returns an empty
  array with a success code (``ballast/market.py:1-13, 81-89``);
* the NYSE calendar computed by rule — holidays including Good Friday, 13:00 early closes
  (``ballast/holidays.py``) — and the session windows (``ballast/sessions.py:24-90``);
* the overnight return — the session's close to the next session's open, the open snapped up to
  the next whole hour because hourly bars are stamped at their start (``ballast/overnight.py``).

Where this departs: pages are fetched side by side rather than walked one at a time (a question is
waiting on them), nothing is cached on disk, and a pair is refused rather than guessed when the
listing cannot be read.

MIT License — Copyright (c) 2026 Ballast contributors. Permission is hereby granted, free of
charge, to any person obtaining a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following
conditions: The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from __future__ import annotations

import functools
import json
import math
import re
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from datetime import time as clock_time
from typing import Any
from zoneinfo import ZoneInfo

BASE = "https://api.bitget.com"
NEW_YORK = ZoneInfo("America/New_York")
MARKET_OPEN = clock_time(9, 30)
MARKET_CLOSE = clock_time(16, 0)
EARLY_CLOSE = clock_time(13, 0)
HOUR_MS = 3_600_000
PAGE = 200
_RTOKEN = re.compile(r"^r[A-Z]")


class SpotError(RuntimeError):
    """Bitget answered with an error, or no usable spot data came back."""


# --- the NYSE calendar, by rule (ballast/holidays.py) -----------------------------------------


def easter(year: int) -> date:
    """Anonymous Gregorian algorithm; Good Friday is the one movable feast the NYSE observes."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (b - (b + 8) // 25 + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    lm = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lm) // 451
    month = (h + lm - 7 * m + 114) // 31
    day = ((h + lm - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date | None:
    """Saturday holidays are observed the Friday before, Sunday ones the Monday after — except a
    Saturday 1 January, which the exchange does not observe at all."""
    if day.weekday() == 5:
        return None if (day.month, day.day) == (1, 1) else day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


@functools.cache
def holidays(year: int) -> frozenset[date]:
    fixed = [date(year, 1, 1), date(year, 7, 4), date(year, 12, 25)]
    if year >= 2022:
        fixed.append(date(year, 6, 19))
    days: set[date | None] = {_observed(d) for d in fixed}
    days |= {
        _nth_weekday(year, 1, 0, 3), _nth_weekday(year, 2, 0, 3),
        easter(year) - timedelta(days=2), _last_weekday(year, 5, 0),
        _nth_weekday(year, 9, 0, 1), _nth_weekday(year, 11, 3, 4),
    }
    return frozenset(d for d in days if d is not None)


@functools.cache
def early_closes(year: int) -> frozenset[date]:
    out = set()
    for candidate in (date(year, 7, 3), _nth_weekday(year, 11, 3, 4) + timedelta(days=1),
                      date(year, 12, 24)):
        if candidate.weekday() <= 4 and candidate not in holidays(year):
            out.add(candidate)
    return frozenset(out)


def is_trading_day(day: date) -> bool:
    return day.weekday() <= 4 and day not in holidays(day.year)


def close_utc(session: date) -> datetime:
    hour = EARLY_CLOSE if session in early_closes(session.year) else MARKET_CLOSE
    return datetime.combine(session, hour, tzinfo=NEW_YORK).astimezone(UTC)


def open_utc(session: date) -> datetime:
    return datetime.combine(session, MARKET_OPEN, tzinfo=NEW_YORK).astimezone(UTC)


def next_session(session: date) -> date:
    nxt = session + timedelta(days=1)
    for _ in range(10):
        if is_trading_day(nxt):
            return nxt
        nxt += timedelta(days=1)
    raise RuntimeError(f"no trading day within 10 days of {session}")


def overnight_window(session: date) -> tuple[datetime, datetime]:
    """(close of ``session``, open of the next session), UTC. Friday spans the weekend."""
    return close_utc(session), open_utc(next_session(session))


# --- overnight returns (ballast/overnight.py) -------------------------------------------------


def _floor_hour_ms(when: datetime) -> int:
    return int(when.replace(minute=0, second=0, microsecond=0).timestamp() * 1000)


def _ceil_hour_ms(when: datetime) -> int:
    floored = when.replace(minute=0, second=0, microsecond=0)
    if when != floored:
        floored += timedelta(hours=1)
    return int(floored.timestamp() * 1000)


def _lookup(bars: Mapping[int, tuple[float, float]], ts: int, field: int,
            tolerance_hours: int = 2) -> float | None:
    for offset in range(tolerance_hours + 1):
        for step in ((0,) if offset == 0 else (offset, -offset)):
            bar = bars.get(ts + step * HOUR_MS)
            if bar and bar[field] > 0:
                return bar[field]
    return None


def overnight_returns(bars: Mapping[int, tuple[float, float]]) -> dict[date, float]:
    """``{session: log(next session's open / this session's close)}`` from ``{ts_ms: (open,
    close)}`` hourly bars. Both prices are needed: with closes alone the open leg resolves to the
    previous hour's close, which Ballast measured shifting returns by 71-99 bp a night."""
    if not bars:
        return {}
    stamps = sorted(bars)
    first = datetime.fromtimestamp(stamps[0] / 1000, UTC).date()
    last = datetime.fromtimestamp(stamps[-1] / 1000, UTC).date()
    out: dict[date, float] = {}
    day = first
    while day <= last:
        if is_trading_day(day):
            close_at, open_at = overnight_window(day)
            start = _lookup(bars, _floor_hour_ms(close_at), field=1)
            end = _lookup(bars, _ceil_hour_ms(open_at), field=0)
            if start and end:
                out[day] = math.log(end / start)
        day += timedelta(days=1)
    return out


# --- Bitget public endpoints (ballast/market.py, ballast/universe.py) -------------------------


def _get(path: str, retries: int = 4) -> Any:  # pragma: no cover - network
    """One public GET. A 429 or a network fault is retried with a growing pause; any other HTTP
    error is Bitget refusing the request (an unknown symbol, a bad parameter) and is raised at once
    with Bitget's own message, because retrying it cannot change the answer."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as resp:
                payload = json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                try:
                    body = json.loads(exc.read().decode("utf-8", "replace"))
                    message = f"{body.get('code')}: {body.get('msg')}"
                except ValueError:
                    message = f"HTTP {exc.code}"
                raise SpotError(f"Bitget refused the request ({message})") from exc
            last = exc
            time.sleep(2.0 * (attempt + 1))
            continue
        except (OSError, ValueError) as exc:
            last = exc
            time.sleep(1.5 ** attempt)
            continue
        if payload.get("code") != "00000":
            raise SpotError(f"Bitget refused the request ({payload.get('code')}: "
                            f"{payload.get('msg')})")
        return payload.get("data")
    raise SpotError(f"Bitget did not answer after {retries} attempts") from last


@dataclass(frozen=True, slots=True)
class Pair:
    ticker: str
    spot: str
    perp: str


def resolve_pairs() -> dict[str, Pair]:  # pragma: no cover - network
    """Every online rToken whose same-company perp is online and classified as a stock."""
    stock_perps = {i["symbol"] for i in _get("/api/v3/market/instruments?category=USDT-FUTURES")
                   if i.get("symbolType") == "stock" and i.get("status") == "online"}
    out: dict[str, Pair] = {}
    for s in _get("/api/v2/spot/public/symbols"):
        base = str(s.get("baseCoin", ""))
        if s.get("status") != "online" or not _RTOKEN.match(base):
            continue
        ticker = base[1:]
        if f"{ticker}USDT" in stock_perps:
            out[ticker] = Pair(ticker=ticker, spot=str(s["symbol"]), perp=f"{ticker}USDT")
    return out


def spot_rtokens() -> list[str]:  # pragma: no cover - network
    """Every online spot rToken symbol (``RTSLAUSDT``), by the ``baseCoin`` rule above."""
    return sorted(str(s["symbol"]) for s in _get("/api/v2/spot/public/symbols")
                  if s.get("status") == "online" and _RTOKEN.match(str(s.get("baseCoin", ""))))


def hourly_bars(symbol: str, *, spot: bool, days: int = 400,
                end: datetime | None = None, workers: int = 3
                ) -> dict[int, tuple[float, float]]:  # pragma: no cover - network
    """``{ts_ms: (open, close)}`` hourly bars over ``days``, oldest first.

    Every page is addressed by its own ``endTime`` (200 hours apart) and the pages are fetched side
    by side: walking the cursor back one page at a time took 15 s for 400 days of one symbol, too
    slow for a question somebody is waiting on. Pages overlap by nothing and miss nothing, because
    each ends exactly where the next one's window starts."""
    from concurrent.futures import ThreadPoolExecutor

    granularity = "1h" if spot else "1H"
    path = ("/api/v2/spot/market/history-candles" if spot
            else "/api/v2/mix/market/history-candles")
    extra = "" if spot else "&productType=usdt-futures"
    last = int((end or datetime.now(UTC)).timestamp() * 1000)
    stop = last - days * 24 * HOUR_MS
    ends = list(range(last, stop, -PAGE * HOUR_MS))

    def page(cursor: int) -> list[list[str]]:
        rows: list[list[str]] = _get(f"{path}?symbol={symbol}&granularity={granularity}"
                                     f"&endTime={cursor}&limit={PAGE}{extra}") or []
        return rows

    seen: dict[int, tuple[float, float]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rows in pool.map(page, ends):
            for r in rows:
                seen[int(r[0])] = (float(r[1]), float(r[4]))
    if not seen:
        raise SpotError(f"no {'spot' if spot else 'perpetual'} bars for {symbol}")
    return {k: seen[k] for k in sorted(seen) if k >= stop}


__all__ = [
    "Pair", "SpotError", "close_utc", "early_closes", "easter", "holidays", "hourly_bars",
    "is_trading_day", "next_session", "open_utc", "overnight_returns", "overnight_window",
    "resolve_pairs", "spot_rtokens",
]
