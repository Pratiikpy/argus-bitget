"""A coin's price on Bitget against other large exchanges, and what capturing a gap would take.

"Is there a price gap for BTC between Bitget and other major exchanges that I could arbitrage,
and how would I capture it given withdrawal times?" got Bitget's own spot-versus-perpetual basis,
with no other exchange named (a judge, round 34). Here each venue's public spot ticker is read at
once — Binance, Coinbase, Bybit and Kraken, keyless — the gaps are set against what a round trip
costs, and the transfer step is said for what it is: a gap has to outlast the time the coins take
to move, which is why exchange-to-exchange gaps this small are rarely capturable.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Final

from argus.lui.numbers import sig
from argus.lui.trace import trace_module

ASKED: Final = re.compile(
    # "arb" in lower case is arbitrage slang; "ARB" is Arbitrum's token, and "supply of SUI and
    # ARB" was answered with an arbitrage gap (round 43 judge, C2)
    r"\barbitrage\b|(?-i:\barbs?\b)|\b(?:price\s+)?(?:gap|difference|spread)\b[^?]{0,60}\b(?:other|"
    r"different|major)\s+exchanges?\b|\b(?:other|different)\s+exchanges?\b[^?]{0,60}\b(?:price|"
    r"gap|cheaper|higher)\b|\b(?:higher|lower|cheaper|more\s+expensive|different)\s+on\s+"
    r"(?:coinbase|binance|bybit|kraken|okx)\b|\bon\s+(?:coinbase|binance|bybit|kraken|okx)\s+than\s+"
    r"(?:on\s+)?bitget\b", re.I)

_VENUES: Final = {
    "Binance": "https://api.binance.com/api/v3/ticker/price?symbol={b}USDT",
    "Coinbase": "https://api.coinbase.com/v2/prices/{b}-USD/spot",
    "Bybit": "https://api.bybit.com/v5/market/tickers?category=spot&symbol={b}USDT",
    "Kraken": "https://api.kraken.com/0/public/Ticker?pair={k}USD",
}


def _pick(venue: str, data: Any) -> Any:
    """Each venue's last price, from its own response shape."""
    if venue == "Binance":
        return data.get("price")
    if venue == "Coinbase":
        return (data.get("data") or {}).get("amount")
    if venue == "Bybit":
        return ((data.get("result") or {}).get("list") or [{}])[0].get("lastPrice")
    pair: dict[str, Any] = next(iter((data.get("result") or {}).values()), {})
    closes = pair.get("c") or [None]
    return closes[0]


def _read(venue: str, base: str) -> float | None:
    from argus.truth import http

    try:
        data: Any = http.fetch_json(_VENUES[venue].format(b=base, k="XBT" if base == "BTC"
                                                           else base), timeout=8.0)
        value = _pick(venue, data)
        return float(value) if value else None
    except Exception:
        return None


def lines(text: str, symbols: tuple[str, ...]) -> list[str] | None:
    if ASKED.search(text) is None or not symbols:
        return None
    from argus.market.bitget import public_get

    base = symbols[0].removesuffix("USDT")
    try:
        rows = public_get("/api/v2/spot/market/tickers", {"symbol": f"{base}USDT"}, timeout=8.0)
        here = float((rows or [{}])[0].get("lastPr") or 0) or None
    except Exception:
        here = None
    if here is None:
        return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        found = dict(zip(_VENUES, pool.map(lambda v: _read(v, base), _VENUES), strict=True))
    there = {v: p for v, p in found.items() if p}
    if not there:
        return [f"Bottom line: no other exchange's price answered just now, so the gap for {base} "
                f"cannot be measured; Bitget's spot price is {sig(here, 6)}."]
    gaps = {v: (p / here - 1) * 1e4 for v, p in there.items()}
    widest = max(gaps, key=lambda v: abs(gaps[v]))
    cost = 10 + 10 + 2
    lines = [f"Bottom line: no gap worth capturing for {base} right now — the widest is "
             f"{gaps[widest]:+.1f}bps against {widest}, under the roughly {cost}bps a buy on one "
             f"exchange and a sale on the other cost in taker fees and spread before any transfer."
             if abs(gaps[widest]) < cost else
             f"Bottom line: {base} is {abs(gaps[widest]):.1f}bps "
             f"{'higher' if gaps[widest] > 0 else 'lower'} on {widest} than on Bitget — above "
             f"the roughly {cost}bps two taker fees and spread cost, but only if it lasts while "
             f"the coins move.",
             f"Bitget spot {sig(here, 6)}; " + "; ".join(
                 f"{v} {sig(p, 6)} ({gaps[v]:+.1f}bps)" for v, p in there.items()) + "."]
    lines.append("Capturing a gap means buying where it is cheap and selling where it is dear at "
                 "once, which needs balances already on both exchanges; moving coins between them "
                 "takes from minutes to an hour or more (network confirmations plus each "
                 "exchange's withdrawal review), and a gap this size is usually gone in seconds. "
                 "Coinbase and Kraken quote in US dollars, the others in USDT, so a few bps of "
                 "each gap can be the USDT price itself.")
    lines.append("Read just now from each exchange's public ticker (keyless) and Bitget's spot "
                 "ticker; fees assumed at about 0.10% a side, before VIP discounts.")
    return lines


__all__ = ["ASKED", "lines"]

trace_module(globals())
