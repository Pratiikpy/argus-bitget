"""Bitget market data — public endpoints, no credentials.

Every fact here was verified against Bitget's own SDK source and the live API on 2026-09-12 rather
than assumed:

* Base URL ``https://api.bitget.com`` — ``agent-sdk/src/config.ts:144``.
* Paper trading is the header ``paptrading: 1``, and it is applied to **private endpoints only**.
  ``agent-sdk/src/client/rest-client.ts:274-280`` documents why: several public endpoints return
  404 "Request URL NOT FOUND" when that header is present. Sending it on market data breaks them.
* Private requests sign ``timestamp + METHOD + endpoint + body`` with HMAC-SHA256 and carry
  ``ACCESS-KEY`` / ``ACCESS-SIGN`` / ``ACCESS-PASSPHRASE`` / ``ACCESS-TIMESTAMP``
  (``rest-client.ts:283-291``). Public market data carries none of that.

**The tokenized equities are real and live.** A call to
``/api/v2/mix/market/tickers?productType=usdt-futures`` returned 787 symbols including NVDAUSDT,
TSLAUSDT, AAPLUSDT, METAUSDT, MSFTUSDT, GOOGLUSDT, AMZNUSDT, COINUSDT, MSTRUSDT, QQQUSDT, SPXUSDT,
TQQQUSDT and SQQQUSDT. These trade continuously while their anchors do not, which is the entire
premise of this system — and it is now checkable rather than asserted.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

BASE_URL = "https://api.bitget.com"

# Tokenized US equities and index products confirmed live on the futures book. Kept explicit
# rather than pattern-matched: a regex over ticker names would sweep in FARTCOINUSDT and
# MARSCOINUSDT, which it did on the first attempt.
# SPXUSDT was here and was WRONG. It trades at $0.49 — it is SPX6900, a memecoin holding the
# ticker, not a tokenized S&P 500. A backtest variant scored 179% and Sharpe 4.32 on it before the
# error was caught. Every one reports symbolType "perpetual", so membership is verified
# behaviourally by session attenuation: see argus.market.validation. Genuine rTokens attenuate
# 3.6-8.8x; SPXUSDT attenuates 1.43x. (Checked again 2026-09-23: the contracts endpoint also
# carries `isRwa`, "NO" for SPXUSDT and "YES" for the tokenised S&P 500, SP500USDT — the venue's
# own flag now agrees with the measurement; `argus.market.universe` reads it.)
RTOKEN_SYMBOLS: tuple[str, ...] = (
    "NVDAUSDT", "TSLAUSDT", "AAPLUSDT", "MSFTUSDT", "METAUSDT",
    "GOOGLUSDT", "AMZNUSDT", "COINUSDT", "MSTRUSDT",
    "QQQUSDT", "TQQQUSDT", "SQQQUSDT",
)

# Excluded by measurement, kept named so the exclusion is visible rather than silent.
REJECTED_SYMBOLS: dict[str, str] = {
    "SPXUSDT": "attenuation 1.43x — no closing anchor; this is SPX6900, a memecoin",
}

ANCHOR_OF: dict[str, str] = {
    "NVDAUSDT": "NVDA", "TSLAUSDT": "TSLA", "AAPLUSDT": "AAPL", "MSFTUSDT": "MSFT",
    "METAUSDT": "META", "GOOGLUSDT": "GOOGL", "AMZNUSDT": "AMZN", "COINUSDT": "COIN",
    "MSTRUSDT": "MSTR", "QQQUSDT": "QQQ",
    "TQQQUSDT": "TQQQ", "SQQQUSDT": "SQQQ",
}


class BitgetError(RuntimeError):
    """A market-data call failed. Carries no credentials — these endpoints have none."""


@dataclass(frozen=True, slots=True)
class Ticker:
    """One instrument's current state, with the fetch time recorded.

    ``fetched_at`` exists so this can become a :class:`~argus.truth.facts.Fact` with an honest
    ``available_at``. A price with no acquisition time cannot be point-in-time bounded later, and
    retrofitting one is how look-ahead gets in.
    """

    symbol: str
    last: Decimal
    bid: Decimal
    ask: Decimal
    high_24h: Decimal
    low_24h: Decimal
    change_24h: Decimal
    base_volume: Decimal
    funding_rate: Decimal
    fetched_at: datetime

    @property
    def spread_bps(self) -> Decimal:
        """Quoted spread in basis points — the first cost any trade must clear.

        Compare against the 12bps round-trip taker fee: on a thin weekend book this alone can
        exceed the entire measured intraday edge.
        """
        if self.bid <= 0 or self.ask <= 0:
            return Decimal("0")
        mid = (self.bid + self.ask) / 2
        return (self.ask - self.bid) / mid * Decimal("10000")

    @property
    def anchor(self) -> str | None:
        """The underlying equity this token tracks, if it is a tokenized equity."""
        return ANCHOR_OF.get(self.symbol)


def _get(path: str, params: dict[str, str] | None = None, *, timeout: float = 30.0) -> Any:
    url = f"{BASE_URL}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    # Deliberately no paptrading header: rest-client.ts:274-280 records that public endpoints
    # 404 when it is present.
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise BitgetError(f"HTTP {exc.code} for {path}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise BitgetError(f"transport failure for {path}: {exc}") from None

    if payload.get("code") not in ("00000", 0, None):
        raise BitgetError(f"{path} returned code {payload.get('code')}: {payload.get('msg')}")
    return payload.get("data")


def maintenance_margin_rate(symbol: str, notional: float) -> float | None:
    """Bitget's maintenance margin rate for a position of ``notional`` USDT in ``symbol``, from the
    public tier table (``/api/v2/mix/market/query-position-lever``: ``startUnit``/``endUnit`` in
    USDT, ``keepMarginRate`` a fraction — NVDAUSDT is 0.50% to $10,000 and 0.66% to $30,000 on
    2026-09-25). None when the table does not answer, and callers then say the distance is before
    maintenance margin rather than assume a rate."""
    try:
        rows = _get("/api/v2/mix/market/query-position-lever",
                    {"symbol": symbol, "productType": "USDT-FUTURES"}, timeout=10.0)
    except BitgetError:
        return None
    for row in rows or []:
        try:
            low, high = float(row["startUnit"]), float(row["endUnit"])
            if low <= notional < high:
                return float(row["keepMarginRate"])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _dec(value: Any, default: str = "0") -> Decimal:
    """Bitget returns numbers as strings, and occasionally as empty strings.

    An empty field becomes the default rather than raising: a missing funding rate should not
    prevent a price from being usable, but it must never silently become something plausible.

    **Use this only for quantities whose zero is a real reading** — a funding rate of zero is what
    the venue pays 81% of the time, and a volume of zero is a quiet hour. For anything that is a
    *price*, use :func:`_price`: see the note there for why the two cannot share a parser.
    """
    if value in (None, "", "null"):
        return Decimal(default)
    try:
        return Decimal(str(value))
    except (ArithmeticError, ValueError, TypeError):
        # Narrowed from `except Exception`. A bare catch would turn a bug in our own call into a
        # default, which is the failure this whole function is supposed to make visible.
        return Decimal(default)


def _price(value: Any, *, field: str, symbol: str) -> Decimal:
    """A price, or an exception. Never zero.

    **The defect this closes.** Every price on a `Ticker` — last, bid, ask, the 24h high and low —
    was parsed by :func:`_dec`, so a field the venue omitted or returned unparseable became
    ``Decimal("0")`` and travelled on as a real quote. Zero is not a price. It is the absence of
    one, and the two must never be the same value, because downstream every one of them reads as a
    number rather than as a gap:

    * ``notional = quantity * entry_price`` silently becomes 0, so a position has no size and no
      cost, and `cost/model.py` charges nothing on it.
    * a 0 entry price makes any subsequent return infinite or undefined, which is why several
      callers already carry ``if prev <= 0: continue`` — those guards are this defect being worked
      around at the far end instead of refused at the near one.
    * `Ticker.spread_bps` returns 0 for a non-positive bid or ask, which reads as *the tightest
      possible book* at exactly the moment we know least about it.

    The last is the one that would have cost money: a missing quote would have made every hurdle
    look clearable. So a missing price raises here, at the boundary, where it can still be
    attributed to the venue rather than to arithmetic forty lines later.
    """
    if value in (None, "", "null"):
        raise BitgetError(f"{symbol}: the venue returned no {field}; a missing price is not zero")
    try:
        got = Decimal(str(value))
    except (ArithmeticError, ValueError, TypeError) as exc:
        raise BitgetError(f"{symbol}: {field}={value!r} is not a number") from exc
    if got <= 0:
        raise BitgetError(f"{symbol}: {field}={got} is not a positive price")
    return got


def fetch_tickers(product_type: str = "usdt-futures") -> dict[str, Ticker]:
    """Every ticker on the book, keyed by symbol. One call, no credentials."""
    now = datetime.now(UTC)
    rows = _get("/api/v2/mix/market/tickers", {"productType": product_type}) or []
    out: dict[str, Ticker] = {}
    for row in rows:
        symbol = row.get("symbol", "")
        if not symbol:
            continue
        try:
            out[symbol] = Ticker(
                symbol=symbol,
                # Prices raise rather than defaulting to zero. See `_price`.
                last=_price(row.get("lastPr"), field="lastPr", symbol=symbol),
                bid=_price(row.get("bidPr"), field="bidPr", symbol=symbol),
                ask=_price(row.get("askPr"), field="askPr", symbol=symbol),
                high_24h=_price(row.get("high24h"), field="high24h", symbol=symbol),
                low_24h=_price(row.get("low24h"), field="low24h", symbol=symbol),
                # These three may legitimately be zero: no change, no volume, no funding due.
                change_24h=_dec(row.get("change24h")),
                base_volume=_dec(row.get("baseVolume")),
                funding_rate=_dec(row.get("fundingRate")),
                fetched_at=now,
            )
        except BitgetError as exc:
            # One unpriceable instrument must not cost the other eleven their cycle, and it must
            # not be silently absent either. It is omitted from the result — callers already treat
            # a missing symbol as "no data for this name", which is exactly what happened — and the
            # reason is printed so a venue that stops quoting something is visible in the log.
            print(f"ticker omitted: {exc}", file=sys.stderr)
    return out


def fetch_rtokens() -> dict[str, Ticker]:
    """Only the tokenized equities and index products — the instruments this system is about."""
    everything = fetch_tickers()
    return {s: everything[s] for s in RTOKEN_SYMBOLS if s in everything}


def fetch_candles(
    symbol: str, *, granularity: str = "1H", limit: int = 200, product_type: str = "usdt-futures"
) -> list[dict[str, Decimal | datetime]]:
    """OHLCV history. The raw material for the overnight and gap studies.

    Bitget returns newest-last arrays of strings:
    ``[ts, open, high, low, close, baseVol, quoteVol]``.
    Converted here rather than at the call site so that no downstream module ever parses a price
    out of a string — a small rule with a large payoff in an arithmetic-sensitive system.
    """
    rows = _get(
        "/api/v2/mix/market/candles",
        {
            "symbol": symbol,
            "productType": product_type,
            "granularity": granularity,
            "limit": str(limit),
        },
    ) or []
    out: list[dict[str, Decimal | datetime]] = []
    for row in rows:
        if len(row) < 6:
            continue
        out.append({
            "ts": datetime.fromtimestamp(int(row[0]) / 1000, tz=UTC),
            # OHLC are prices and raise on absence. A candle with a zero close poisons every
            # return computed from it, and does so silently: the bar looks present.
            "open": _price(row[1], field="open", symbol=symbol),
            "high": _price(row[2], field="high", symbol=symbol),
            "low": _price(row[3], field="low", symbol=symbol),
            "close": _price(row[4], field="close", symbol=symbol),
            "volume": _dec(row[5]),
        })
    return out
