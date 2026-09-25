"""Bitget's own long/short ratios for any USDT perpetual, read from its public market API.

The positioning lines in `market/bitget_positioning.py` carry Binance's ratios, served through
Bitget's data service, for BTC and ETH only — "what is the long/short ratio on SOL?" was answered
with open interest and no ratio at all, and nothing said so (2026-09-25 audit, round 2). Bitget
publishes three hourly series of its own for every USDT perpetual, keyless:

* ``/api/v2/mix/market/account-long-short`` — the share of *accounts* holding a long;
* ``/api/v2/mix/market/position-long-short`` — the share of *open position size* that is long;
* ``/api/v2/mix/market/long-short`` — the venue's headline long ratio.

The first two are the ones a trader reads against each other: many accounts long while position
size is balanced means the crowd leans long in small size and the larger money does not. Each
series is returned oldest first (checked against the live endpoint, 2026-09-25), so the newest row
is the last and the row 24 hours earlier sits 24 places before it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from argus.market.bitget import _get

SOURCE = "Bitget /api/v2/mix/market/account-long-short + position-long-short"


@dataclass(frozen=True, slots=True)
class LongShort:
    symbol: str
    accounts_long: float
    accounts_long_day_ago: float | None
    position_long: float | None
    stamp_ms: int


def _series(path: str, symbol: str) -> list[dict[str, Any]]:
    rows = _get(path, {"symbol": symbol, "period": "1h"}) or []
    return sorted(rows, key=lambda r: int(r.get("ts", 0)))


def read(symbol: str) -> LongShort | None:
    """The newest hourly reading for ``symbol``, or None when Bitget does not answer."""
    try:
        accounts = _series("/api/v2/mix/market/account-long-short", symbol)
        positions = _series("/api/v2/mix/market/position-long-short", symbol)
    except Exception:
        return None
    if not accounts:
        return None
    try:
        latest = accounts[-1]
        now_share = float(latest["longAccountRatio"])
        day_ago = float(accounts[-25]["longAccountRatio"]) if len(accounts) >= 25 else None
        position = float(positions[-1]["longPositionRatio"]) if positions else None
        return LongShort(symbol=symbol, accounts_long=now_share, accounts_long_day_ago=day_ago,
                         position_long=position, stamp_ms=int(latest["ts"]))
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def lines(reading: LongShort, name: str) -> list[str]:
    """One line a trader can act on: the account ratio, its day's change, and the size split."""
    ratio = reading.accounts_long / max(1e-9, 1 - reading.accounts_long)
    text = (f"Long/short on Bitget's {name} perpetual: {reading.accounts_long:.0%} of accounts are "
            f"long (a ratio of {ratio:.2f})")
    if reading.accounts_long_day_ago is not None:
        move = (reading.accounts_long - reading.accounts_long_day_ago) * 100
        text += f", {move:+.1f} points on a day ago"
    if reading.position_long is not None:
        text += f"; by position size {reading.position_long:.0%} is long"
        gap = reading.accounts_long - reading.position_long
        if gap >= 0.10:
            text += (" — many small accounts lean long while the larger positions do not, the "
                     "split a contrarian read starts from")
        elif gap <= -0.10:
            text += (" — the larger positions lean long while most accounts do not")
    return [text + " (Bitget, hourly)."]
