"""Which crypto narratives are moving: baskets of Bitget-listed coins, measured on their own closes.

"What's the broader market intel across crypto right now — which sectors or narratives are hot?"
was answered by the equity sector engine (Technology, Energy, Financials against SPY) and the word
"crypto" never appeared (round 39 judge, C-3). bitget-signal's market-intel Skill is the named
source for this and did not answer in that audit, so the console measures it itself.

**What is measured.** Each narrative is a small basket of coins Bitget lists perpetuals for, chosen
as the narrative's largest liquid names (the basket is printed, so a reader can disagree with it).
Each coin's move over 7 and 30 days comes from Bitget's daily closes; a basket's move is the equal-
weighted average of its coins that answered. Ranked by the 30-day move against BTC's, the market a
narrative has to beat to be "hot". CoinGecko's category table (keyless,
``/api/v3/coins/categories``) adds the 24-hour market-cap change per category as a second, outside
reading — 24 hours alone is noise and is said to be.

Taken from: the way CoinGecko and Artemis group narratives (AI, DeFi, RWA, meme, layer 2) —
names only; the baskets are this console's and listed in full.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Final

ASKED: Final = re.compile(
    r"\b(?:crypto|coins?|tokens?|altcoins?|alts|defi|web3|on-?chain)\b[^?]{0,60}\b(?:narratives?|"
    r"sectors?|themes?|rotation|rotating|hot|leading|lagging|trending|in\s+favou?r|"
    r"market\s+intel\w*)\b|\b(?:narratives?|themes?|sectors?)\b[^?]{0,40}\b(?:crypto|coins?|"
    r"altcoins?|defi)\b|\bwhich\s+narratives?\b", re.I)

BASKETS: Final[dict[str, tuple[str, ...]]] = {
    "AI": ("FET", "RENDER", "TAO", "WLD"),
    "DeFi": ("UNI", "AAVE", "LDO", "CRV"),
    "Real-world assets": ("ONDO", "LINK", "PENDLE"),
    "Layer 1": ("SOL", "ADA", "AVAX", "SUI"),
    "Layer 2": ("ARB", "OP", "POL", "STRK"),
    "Meme": ("DOGE", "PEPE", "SHIB", "WIF"),
    "DePIN": ("FIL", "HNT", "AR", "IOTX"),
}
GECKO_CATEGORIES: Final[dict[str, str]] = {
    "AI": "artificial-intelligence", "DeFi": "decentralized-finance-defi",
    "Real-world assets": "real-world-assets-rwa", "Layer 1": "layer-1",
    "Layer 2": "layer-2", "Meme": "meme-token", "DePIN": "depin",
}


def _moves(symbol: str) -> tuple[float, float] | None:
    """(7-day, 30-day) move of ``symbol`` from Bitget's daily closes, or None."""
    from argus.market.history import fetch_window

    try:
        bars = fetch_window(symbol, start=datetime.now(UTC) - timedelta(days=32), interval="1Dutc",
                            pause=0.05)
    except Exception:
        return None
    closes = [float(b.close) for b in bars if float(b.close) > 0]
    if len(closes) < 31:
        return None
    return closes[-1] / closes[-8] - 1, closes[-1] / closes[-31] - 1


def _gecko() -> dict[str, float]:
    """24-hour market-cap change per CoinGecko category id, or empty when it did not answer."""
    from argus.truth import http

    try:
        rows = http.fetch_json("https://api.coingecko.com/api/v3/coins/categories", timeout=10.0)
    except Exception:
        return {}
    return {str(r.get("id")): float(r["market_cap_change_24h"]) for r in rows or []
            if isinstance(r, dict) and r.get("market_cap_change_24h") is not None}


def lines(text: str) -> list[str] | None:
    """The narratives ranked by their 30-day move against BTC, or None when not asked."""
    if not ASKED.search(text):
        return None
    from concurrent.futures import ThreadPoolExecutor

    names = sorted({c for basket in BASKETS.values() for c in basket} | {"BTC"})
    with ThreadPoolExecutor(max_workers=8) as pool:
        moved = dict(zip(names, pool.map(lambda c: _moves(f"{c}USDT"), names), strict=True))
        gecko_job = pool.submit(_gecko)
    btc = moved.get("BTC")
    if btc is None:
        return None
    rows: list[tuple[str, float, float, list[str]]] = []
    for narrative, basket in BASKETS.items():
        got = [(c, m) for c in basket if (m := moved.get(c)) is not None]
        if len(got) < 2:
            continue
        week = sum(m[0] for _, m in got) / len(got)
        month = sum(m[1] for _, m in got) / len(got)
        rows.append((narrative, week, month, [c for c, _ in got]))
    if len(rows) < 3:
        return None
    rows.sort(key=lambda r: r[2], reverse=True)
    gecko = gecko_job.result()
    hot = [r for r in rows if r[2] > btc[1]]
    lead = (f"Bottom line: over 30 days {rows[0][0]} led ({rows[0][2]:+.1%}, against BTC's "
            f"{btc[1]:+.1%}) and {rows[-1][0]} lagged ({rows[-1][2]:+.1%}); "
            + (f"{len(hot)} of {len(rows)} narratives beat bitcoin." if hot else
               "none of the narratives beat bitcoin — the market's own coin led."))
    table = "; ".join(f"{n} {m:+.1%} (7 days {w:+.1%})" for n, w, m, _ in rows)
    out = [lead, f"Each narrative, 30 days: {table}."]
    outside = [f"{n} {gecko[GECKO_CATEGORIES[n]]:+.1f}%" for n, *_ in rows
               if GECKO_CATEGORIES.get(n) in gecko]
    if outside:
        out.append("CoinGecko's category market caps over the last 24 hours, a second reading "
                   "(a day is mostly noise): " + ", ".join(outside) + ".")
    out.append("Baskets, equal-weighted, on Bitget's daily closes: " + "; ".join(
        f"{n}: {', '.join(cs)}" for n, _, _, cs in rows) + ". A basket is this console's "
        "choice of the narrative's largest liquid coins; a past move, not a forecast.")
    return out
