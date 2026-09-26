"""Positioning and mood from Bitget's own data service, for the answers that were missing them.

Track 3 scores "data sources / Skill integration count *and effectiveness*". A toolkit review
(2026-09-24) called every one of the 67 `bitget-mcp-server` entries and found 62 answering with
data, while the console used 8, all of them equity fundamentals. This module reads the ones a
trader's questions need and the answers were missing:

* ``sentiment_market_fear_greed`` — the US stock market's fear & greed score with its week, month
  and year ago. The console's sentiment answer for NVDA quoted the *crypto* index, which says
  nothing about how a stock is being traded.
* ``crypto_sentiment_crypto_fear_greed`` — the crypto score from the same service.
* ``crypto_futures_long_short_ratio`` and ``crypto_futures_long_short_top_position_ratio`` — the
  share of all accounts that are long, against the share of the largest traders' positions: the
  crowd against the whales, the comparison a contrarian read starts from.
* ``crypto_indicators_hyperliquid_whale_sentiment`` — the dollar value large Hyperliquid wallets
  hold long and short.
* ``crypto_futures_liquidations`` — the last day's forced selling on each side.
* ``crypto_institutional_company_flow`` — listed companies' bitcoin treasuries (Strategy, Coinbase).
* ``news_label_search`` — Bitget's own daily US-stock brief ("Bitget UEX Daily").
* ``equity_fundamental_dividends`` — ex-dividend dates, which move a tokenised stock's price.

**What is deliberately not used, and why.** ``crypto_etf_flows`` reports a ``net_flow_1d`` that
is the day's change in each fund's *dollar holding* — IBIT's "inflow" on 2026-09-21 and 22
($3.76bn, $4.47bn) followed bitcoin's price rather than creations, and it read zero on every
weekend day. Quoting it as a flow would present a price move as money arriving, so it is left out
until a series of creations and redemptions is available. The liquidation and long/short series
are Binance's, served through Bitget's service, and every line says so.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from argus.market.bitget_mcp import shared_service

SOURCE = "Bitget's data service, bitget-mcp-server"


def _safe(entry: str, **params: Any) -> list[dict[str, Any]]:
    try:
        return shared_service().results(entry, **params)
    except Exception:
        return []


def stock_mood() -> str | None:
    """The US stock market's fear & greed score, and where it was a week and a month ago."""
    rows = _safe("sentiment_market_fear_greed")
    if not rows:
        return None
    row = rows[0]
    try:
        score = float(row["score"])
        week = float(row["previous_1_week"])
        month = float(row["previous_1_month"])
    except (KeyError, TypeError, ValueError):
        return None
    return (f"US stock-market fear & greed: {score:.0f} ({row.get('rating', '')}); a week ago "
            f"{week:.0f}, a month ago {month:.0f} ({SOURCE}).")


def crypto_mood() -> tuple[int, str, list[int]] | None:
    """Today's crypto fear & greed and the last week of it, from Bitget's service."""
    rows = _safe("crypto_sentiment_crypto_fear_greed", limit=8)
    values: list[int] = []
    label = ""
    for row in sorted(rows, key=lambda r: str(r.get("date", "")), reverse=True):
        try:
            values.append(int(row["value"]))
        except (KeyError, TypeError, ValueError):
            continue
        label = label or str(row.get("classification", ""))
    return (values[0], label, values) if values else None


@dataclass(frozen=True)
class Positioning:
    base: str
    accounts_long: float | None
    whales_long: float | None
    hyperliquid_long_usd: float | None
    hyperliquid_short_usd: float | None
    liquidated_long_usd: float | None
    liquidated_short_usd: float | None
    liquidation_day: str | None

    def lines(self) -> list[str]:
        out: list[str] = []
        if self.accounts_long is not None and self.whales_long is not None:
            gap = self.whales_long - self.accounts_long
            lean = ("the largest traders are longer than the crowd" if gap > 0.05 else
                    "the crowd is longer than the largest traders" if gap < -0.05 else
                    "the crowd and the largest traders lean the same way")
            out.append(f"{self.base} positioning: {self.accounts_long:.0%} of all accounts are "
                       f"long, against {self.whales_long:.0%} of the largest traders' positions "
                       f"— {lean} (Binance, via {SOURCE}).")
        if self.hyperliquid_long_usd and self.hyperliquid_short_usd:
            out.append(f"Large Hyperliquid wallets hold ${self.hyperliquid_long_usd / 1e6:,.0f}m "
                       f"long and ${self.hyperliquid_short_usd / 1e6:,.0f}m short in {self.base} "
                       f"({SOURCE}).")
        if self.liquidated_long_usd is not None and self.liquidated_short_usd is not None:
            heavier = ("longs" if self.liquidated_long_usd > self.liquidated_short_usd
                       else "shorts")
            out.append(f"{self.base} liquidations on {self.liquidation_day}: "
                       f"${self.liquidated_long_usd / 1e6:,.1f}m of longs and "
                       f"${self.liquidated_short_usd / 1e6:,.1f}m of shorts forced out — the "
                       f"{heavier} took the larger flush (Binance, via {SOURCE}).")
        return out


def positioning(base: str) -> Positioning:
    """Crowd against whales, Hyperliquid whale value and the last full day's liquidations, for
    BTC or ETH, fetched side by side."""
    pair = f"{base}USDT"
    with ThreadPoolExecutor(max_workers=4) as pool:
        crowd = pool.submit(_safe, "crypto_futures_long_short_ratio", symbol=pair,
                            interval="1h", limit=1)
        whales = pool.submit(_safe, "crypto_futures_long_short_top_position_ratio", symbol=pair,
                             interval="1h", limit=1)
        hyper = pool.submit(_safe, "crypto_indicators_hyperliquid_whale_sentiment", symbol=base,
                            limit=1)
        liquid = pool.submit(_safe, "crypto_futures_liquidations", symbol=pair,
                             exchange="binance", limit=3)

    def first(rows: list[dict[str, Any]], key: str) -> float | None:
        try:
            return float(rows[0][key]) if rows else None
        except (KeyError, TypeError, ValueError):
            return None

    # The newest row is today's, still filling; the last *full* day is the one before it.
    today = datetime.now(UTC).date().isoformat()
    liquidations = sorted((r for r in liquid.result() if str(r.get("date", ""))[:10] < today),
                          key=lambda r: str(r.get("date", "")))
    full_day = liquidations[-1] if liquidations else None
    return Positioning(
        base=base,
        accounts_long=first(crowd.result(), "long_account"),
        whales_long=first(whales.result(), "long_account"),
        hyperliquid_long_usd=first(hyper.result(), "total_long_position_val"),
        hyperliquid_short_usd=first(hyper.result(), "total_short_position_val"),
        liquidated_long_usd=(float(full_day["long_liquidations"]) if full_day else None),
        liquidated_short_usd=(float(full_day["short_liquidations"]) if full_day else None),
        liquidation_day=(str(full_day.get("date", ""))[:10] if full_day else None),
    )


def us_stock_brief() -> str | None:
    """The title and date of Bitget's latest "UEX Daily" US-stock brief."""
    rows = _safe("news_label_search", label=1, page_size=3)
    for row in rows:
        title = re.sub(r"\s+", " ", str(row.get("title", ""))).strip()
        if title:
            return f"Bitget's own US-stock brief: “{title}” ({SOURCE}, news_label_search)."
    return None


def next_ex_dividend(ticker: str, today: date | None = None) -> str | None:
    """An ex-dividend date in the next 30 days or the last 7, which moves the token's price."""
    now = today or datetime.now(UTC).date()
    for row in sorted(_safe("equity_fundamental_dividends", symbol=ticker),
                      key=lambda r: str(r.get("ex_dividend_date", "")), reverse=True):
        try:
            ex = date.fromisoformat(str(row["ex_dividend_date"])[:10])
            amount = float(row["amount"])
        except (KeyError, TypeError, ValueError):
            continue
        if now - timedelta(days=7) <= ex <= now + timedelta(days=30):
            when = "goes" if ex >= now else "went"
            return (f"{ticker} {when} ex-dividend on {ex:%d %b %Y} (${amount:g} a share, paid "
                    f"{str(row.get('payment_date', ''))[:10]}); the stock drops by about the "
                    f"dividend that morning, and a tokenised price follows it ({SOURCE}).")
        if ex < now - timedelta(days=7):
            return None
    return None


def bitcoin_treasury(ticker: str, market_cap_usd: float | None,
                     bitcoin_usd: float | None) -> str | None:
    """A listed company's bitcoin holding and, with its market value, what the market pays for
    each dollar of that bitcoin (mNAV)."""
    rows = [r for r in _safe("crypto_institutional_company_flow", symbol="BTC")
            if str(r.get("ticker", "")).upper() == ticker.upper()]
    if not rows:
        return None
    latest = max(rows, key=lambda r: (int(str(r.get("ts", "0") or 0)),
                                       float(r.get("holding_balance") or 0)))
    try:
        held = float(latest["holding_balance"])
    except (KeyError, TypeError, ValueError):
        return None
    line = f"{latest.get('company_name', ticker)} holds {held:,.0f} BTC"
    if bitcoin_usd and market_cap_usd and held > 0:
        worth = held * bitcoin_usd
        line += (f", worth ${worth / 1e9:,.1f}bn at ${bitcoin_usd:,.0f}; a "
                 f"${market_cap_usd / 1e9:,.1f}bn market value is {market_cap_usd / worth:.2f}x "
                 f"its bitcoin")
    return line + f" ({SOURCE})."


__all__ = ["Positioning", "bitcoin_treasury", "crypto_mood", "next_ex_dividend", "positioning",
           "stock_mood", "us_stock_brief"]
