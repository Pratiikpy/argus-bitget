"""A stock split the question takes for granted, checked against the record before anything is
priced on it.

"after NVDA's 3-for-1 split last week, what's my 30 share position worth" named a split that never
happened, and the answer neither said so nor valued the 30 shares (a hostile review, 2026-09-30).
A split changes a share count by its ratio, so a position priced on a split that did not happen is
wrong by exactly that ratio. The corporate-actions record is Bitget's data service
(`market/bitget_positioning`, ``equity_fundamental_dividends``, which carries each split's
numerator and denominator), and the answer quotes the last split it holds.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

from argus.lui.answer import Source
from argus.lui.trace import trace_module

SPLIT_CLAIM = re.compile(
    r"\b(?P<num>\d+)\s*(?:-|\s)?\s*(?:for|to|:)\s*(?:-|\s)?\s*(?P<den>\d+)\s+(?:stock\s+)?"
    r"(?:reverse\s+)?split\b|\b(?:stock\s+)?split\s+(?P<num2>\d+)\s*(?:-|\s)?(?:for|to|:)\s*(?:-|\s)?"
    r"(?P<den2>\d+)\b", re.I)
"""A split with its ratio: "3-for-1 split", "a 10:1 stock split", "split 2 for 1"."""

_SPLIT_SAID = re.compile(
    r"\b(?:stock\s+)?split\b[^?.]{0,40}\b(?:happened|took\s+effect|went\s+through|last\s+(?:week|"
    r"month)|yesterday|recently|this\s+(?:week|month)|just)\b|\b(?:just|recently)\s+split\b",
    re.I)
"""A split said to have happened, with no ratio: "Stock split for MSFT happened last week"."""


def claimed(text: str) -> bool:
    """Whether ``text`` states a split, with or without its ratio."""
    return bool(SPLIT_CLAIM.search(text) or _SPLIT_SAID.search(text))


_SHARES = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:-\s*)?shares?\b", re.I)
"""A share count: "my 30 share position", "30 shares"."""

RECENT_DAYS = 45
"""How recent a split must be for "last week" or "this month" to be true of it."""


def _splits(ticker: str, rows: list[dict[str, Any]], today: date
            ) -> list[tuple[date, float, float]]:
    out: list[tuple[date, float, float]] = []
    for row in rows:
        try:
            ex = date.fromisoformat(str(row["ex_dividend_date"])[:10])
        except (KeyError, TypeError, ValueError):
            continue
        if ex <= today and row.get("split_numerator") and row.get("split_denominator"):
            try:
                out.append((ex, float(row["split_numerator"]), float(row["split_denominator"])))
            except (TypeError, ValueError):
                continue
    return sorted(out)


def _service_rows(ticker: str) -> list[dict[str, Any]]:
    from argus.market.bitget_positioning import corporate_actions

    return corporate_actions(ticker)


def check(text: str, symbol: str, *, price: Callable[[str], float] | None = None,
          rows: Callable[[str], list[dict[str, Any]]] | None = None,
          today: date | None = None) -> tuple[list[str], list[Source], dict[str, Any]] | None:
    """The stated split against the record, and the stated shares valued at the live price."""
    claim = SPLIT_CLAIM.search(text)
    if claim is None and not _SPLIT_SAID.search(text):
        return None
    ticker = symbol.removesuffix("USDT")
    now = today or datetime.now(UTC).date()
    record = _splits(ticker, (rows or _service_rows)(ticker), now)
    source = "Bitget's data service (bitget-mcp-server)"
    if not record and rows is None:
        # the service returned no row at all for NVDA's 10-for-1 of June 2024 (2026-10-06), and
        # "holds no split" read as "never split": Yahoo's split events are the second source
        from argus.market.equity_history import HistoryError, split_events

        try:
            record = [s for s in split_events(ticker) if s[0] <= now]
            source = "Yahoo Finance's split events (Bitget's data service returned none)"
        except HistoryError:
            source = "neither Bitget's data service nor Yahoo Finance answered"
    recent = [s for s in record if s[0] >= now - timedelta(days=RECENT_DAYS)]
    if claim is None:
        # "Stock split for MSFT happened last week so what is MSFT trading at?" named no ratio
        # and got a weekly performance with no word on the split (round 45 re-ask): any split in
        # the window confirms it, none denies it
        num, den = (recent[-1][1], recent[-1][2]) if recent else (0.0, 0.0)
    else:
        num = float(claim.group("num") or claim.group("num2"))
        den = float(claim.group("den") or claim.group("den2"))
    same = [s for s in recent if (s[1], s[2]) == (num, den)]
    if same:
        ex, a, b = same[-1]
        lead = (f"Bottom line: the record agrees — {ticker} split {a:g}-for-{b:g} on "
                f"{ex:%d %b %Y}, so share counts from before then are multiplied by {a / b:g}.")
    elif record:
        ex, a, b = record[-1]
        lead = (f"Bottom line: that premise is not on the record — {ticker}'s last split was "
                f"{a:g}-for-{b:g} on {ex:%d %b %Y}, and there is no "
                + (f"{num:g}-for-{den:g} split" if num else "split")
                + f" in the last {RECENT_DAYS} days, so nothing below assumes one.")
    else:
        lead = (f"Bottom line: that premise is not on the record — {source}"
                + ("" if source.startswith("neither") else f" shows no split for {ticker}")
                + ", so nothing below assumes one.")
    lines = [lead]
    shares = _SHARES.search(text)
    data: dict[str, Any] = {"claimed": [num, den] if num else None,
                            "last_split": [str(record[-1][0]), *record[-1][1:]] if record else None}
    if shares is not None and price is not None:
        count = float(shares.group(1).replace(",", ""))
        try:
            last = price(symbol)
        except Exception:
            last = None
        if last is not None:
            lines.append(f"{count:g} shares of {ticker} at the last Bitget price, {last:,.2f}, are "
                         f"worth ${count * last:,.2f}" + ("." if not same else
                         f" — if the {count:g} were counted before the split, they are now "
                         f"{count * num / den:g} shares, worth ${count * num / den * last:,.2f}."))
            data["value"] = count * last
    elif price is not None and re.search(r"\bprice\b|\bwhere\s+is\b|\btrading\s+at\b|\bquote\b",
                                         text, re.I):
        # "Apple's stock split 10-for-1 last week, so what's the AAPL price now?" corrected the
        # split and never gave the price (round 45 hostile, m3)
        try:
            last = price(symbol)
        except Exception:
            last = None
        if last is not None:
            lines.append(f"{ticker} is {last:,.2f} on Bitget now"
                         + (" — no split adjustment applies." if not same else "."))
    lines.append(f"Data: corporate actions from {source}; the price is Bitget's live ticker.")
    sources = [Source("venue", "bitget-mcp-server equity_fundamental_dividends",
                      f"{ticker} splits on record"),
               Source("venue", "bitget /api/v2/mix/market/tickers", f"{symbol} last price")]
    return lines, sources, data


trace_module(globals())
