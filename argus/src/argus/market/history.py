"""Historical candles, including the basis series that makes rToken research possible.

Spec read from Bitget's own ``agent-sdk/openapi.yaml:1972`` rather than guessed. The v3 endpoint
``/api/v3/market/history-candles`` takes ``category``, ``symbol``, ``interval``, ``startTime``,
``endTime`` and — the important one — ``type``.

**``type`` is the discovery that unlocks Track 1.** Four series are available for the same
instrument and interval, verified live on 2026-09-12:

=========  ==========================================================================
``type``   What it is
=========  ==========================================================================
market     the traded price of the token
index      the underlying reference price — what the anchor is worth
premium    the token's premium/discount as a fraction, signed
mark       the mark price used for liquidation
=========  ==========================================================================

One hour of NVDAUSDT returned market ``219.20``, index ``219.2074``, premium ``-0.0000338``. That
is the rToken basis, published, at hourly resolution, for free. The Arbitrage sub-theme asks about
"NAV premium/discount under mint/redeem mechanics"; the Price-Discovery Twin needs to know what the
anchor was worth while it was shut. Both questions reduce to these three series.

The documented range limit is **90 days**, so :func:`fetch_range` pages backwards through it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Any

BASE_URL = "https://api.bitget.com"
HISTORY_PATH = "/api/v3/market/history-candles"
# Measured, not documented: the openapi spec states no cap, but limit=200 returns
# code 40020 "Parameter limit error" while 100 succeeds.
MAX_LIMIT = 100


class CandleType(StrEnum):
    MARKET = "market"
    INDEX = "index"
    """The underlying reference. What the anchor is worth, published continuously — including
    while the anchor market itself is shut."""

    PREMIUM = "premium"
    """Signed premium/discount as a fraction. Negative means the token trades below its index."""

    MARK = "mark"


class HistoryError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Candle:
    ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True, slots=True)
class BasisPoint:
    """The token, its anchor reference, and the gap between them at one instant.

    This is the object the Arbitrage and After-Hours cells are built on, and the reason it can be
    computed at all is that Bitget publishes the index series continuously — even at 3am on a
    Sunday when the anchor market has no price of its own.
    """

    ts: datetime
    market: Decimal
    index: Decimal
    premium: Decimal

    @property
    def basis_bps(self) -> Decimal:
        """Token price relative to its index, in basis points. Computed, not taken from the feed.

        The published ``premium`` series is kept alongside as a cross-check: two independently
        derived numbers that should agree are worth more than one that cannot be questioned.
        """
        if self.index <= 0:
            return Decimal("0")
        return (self.market - self.index) / self.index * Decimal("10000")

    @property
    def published_premium_bps(self) -> Decimal:
        return self.premium * Decimal("10000")

    @property
    def clears_round_trip(self) -> bool:
        """Does the raw basis even exceed the 12bps round-trip fee?

        Deliberately the first question asked of any apparent arbitrage. A basis that fails this
        is not a small opportunity — it is a loss, and most of them fail it.
        """
        return abs(self.basis_bps) > Decimal("12")


def _dec(value: Any) -> Decimal:
    if value in (None, "", "null"):
        return Decimal("0")
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("0")


def _get(params: dict[str, str], *, timeout: float = 30.0) -> list[list[str]]:
    url = f"{BASE_URL}{HISTORY_PATH}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:200]
        raise HistoryError(f"HTTP {exc.code} for history-candles: {detail}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise HistoryError(f"transport failure: {exc}") from None

    if payload.get("code") not in ("00000", 0, None):
        raise HistoryError(f"history-candles code {payload.get('code')}: {payload.get('msg')}")
    return payload.get("data") or []


def fetch(
    symbol: str,
    *,
    interval: str = "1H",
    candle_type: CandleType = CandleType.MARKET,
    end: datetime | None = None,
    limit: int = MAX_LIMIT,
) -> list[Candle]:
    """One page of history, oldest first."""
    params = {
        "category": "USDT-FUTURES",
        "symbol": symbol,
        "interval": interval,
        "type": str(candle_type),
        "limit": str(min(limit, MAX_LIMIT)),
    }
    if end is not None:
        params["endTime"] = str(int(end.timestamp() * 1000))

    out: list[Candle] = []
    for row in _get(params):
        if len(row) < 5:
            continue
        out.append(Candle(
            ts=datetime.fromtimestamp(int(row[0]) / 1000, tz=UTC),
            open=_dec(row[1]), high=_dec(row[2]), low=_dec(row[3]), close=_dec(row[4]),
            volume=_dec(row[5]) if len(row) > 5 else Decimal("0"),
        ))
    out.sort(key=lambda c: c.ts)
    return out


def fetch_range(
    symbol: str,
    *,
    days: int = 90,
    interval: str = "1H",
    candle_type: CandleType = CandleType.MARKET,
    pause: float = 0.15,
) -> list[Candle]:
    """Page backwards to cover ``days``. The endpoint documents a 90-day maximum range.

    Pages by walking ``endTime`` back to the oldest candle already retrieved. A page that returns
    nothing, or fails to move the window, ends the walk — otherwise a venue that clamps silently
    would spin here forever.
    """
    cutoff = datetime.now(UTC) - timedelta(days=days)
    seen: dict[datetime, Candle] = {}
    end: datetime | None = None

    while True:
        page = fetch(symbol, interval=interval, candle_type=candle_type, end=end)
        if not page:
            break
        new = {c.ts: c for c in page if c.ts not in seen}
        if not new:
            break
        seen.update(new)
        oldest = min(page, key=lambda c: c.ts).ts
        if oldest <= cutoff:
            break
        if end is not None and oldest >= end:
            break  # the window stopped moving; the venue is clamping
        end = oldest
        time.sleep(pause)

    return sorted((c for ts, c in seen.items() if ts >= cutoff), key=lambda c: c.ts)


def fetch_basis(
    symbol: str, *, days: int = 90, interval: str = "1H"
) -> list[BasisPoint]:
    """Market, index and premium joined on timestamp.

    Only timestamps present in all three series are returned. An inner join rather than a forward
    fill: carrying a stale index across a gap would manufacture a basis that never existed, which
    is precisely the kind of number that makes a false arbitrage look real.
    """
    market = {c.ts: c for c in fetch_range(symbol, days=days, interval=interval,
                                           candle_type=CandleType.MARKET)}
    index = {c.ts: c for c in fetch_range(symbol, days=days, interval=interval,
                                          candle_type=CandleType.INDEX)}
    premium = {c.ts: c for c in fetch_range(symbol, days=days, interval=interval,
                                            candle_type=CandleType.PREMIUM)}

    common = sorted(set(market) & set(index) & set(premium))
    return [
        BasisPoint(
            ts=ts,
            market=market[ts].close,
            index=index[ts].close,
            premium=premium[ts].close,
        )
        for ts in common
    ]
