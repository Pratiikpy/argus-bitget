"""Two chains side by side on what their users pay and build: fees, DEX volume, TVL, fees per
dollar of market value, and commits to each chain's main client.

"What is the thesis for Solana versus Ethereum on real user fees and developer activity, and which
offers better risk-adjusted value?" (asked in Spanish) got a return-and-volatility comparison and
nothing on fees or developers (round 40 judge, Q23), though the thesis engine reads chain fees
(`lui/thesis.chain_activity`) for one chain at a time. Here, for both:

- **Fees** — DeFiLlama's ``overview/fees/{chain}``: the last 30 days in dollars, and the change
  against the 30 before (`thesis.month_change`, with where that change ranks among the chain's own
  past months).
- **DEX volume and TVL** — the same, from ``overview/dexs`` and ``v2/historicalChainTvl``.
- **Value** — 30-day fees annualised over the token's market value (CoinGecko ``simple/price``):
  what a holder's dollar is "earning" in user fees, the closest public analogue of an earnings
  yield. Fees are not all paid to holders, which the answer says.
- **Developers** — commits to each chain's main client repository over the last four weeks
  (GitHub's ``stats/participation``): one repository, so a proxy, said as one.
"""

from __future__ import annotations

import re
from typing import Final, TypedDict

ASKED: Final = re.compile(r"\b(?:fees?|revenue|developer|dev\s+activity|users?|usage|on-?chain\s+"
                          r"activity|tvl)\b", re.I)
_CLIENT: Final = {"Solana": "anza-xyz/agave", "Ethereum": "ethereum/go-ethereum",
                  "Sui": "MystenLabs/sui", "Aptos": "aptos-labs/aptos-core",
                  "Avalanche": "ava-labs/avalanchego", "Near": "near/nearcore"}
_GECKO: Final = {"SOL": "solana", "ETH": "ethereum", "SUI": "sui", "APT": "aptos",
                 "AVAX": "avalanche-2", "NEAR": "near", "BNB": "binancecoin", "TRX": "tron",
                 "ADA": "cardano", "TON": "the-open-network"}
LLAMA: Final = "https://api.llama.fi"


class Row(TypedDict):
    ticker: str
    chain: str
    fees: float | None
    dex: float | None
    cap: float
    yield_: float | None
    fee_change: float | None
    fee_rank: float | None
    tvl_change: float | None
    commits: int | None
    repo: str | None


def _thirty(chain: str, kind: str) -> float | None:
    from argus.truth import http

    try:
        body = http.fetch_json(f"{LLAMA}/overview/{kind}/{chain.lower()}",
                               params={"excludeTotalDataChart": "true",
                                       "excludeTotalDataChartBreakdown": "true"}, timeout=20.0)
        value = (body or {}).get("total30d")
        return float(value) if value else None
    except Exception:
        return None


def _commits(repo: str) -> int | None:
    from argus.truth import http

    try:
        body = http.fetch_json(f"https://api.github.com/repos/{repo}/stats/participation",
                               timeout=20.0)
        weeks = (body or {}).get("all") or []
        return int(sum(weeks[-4:])) if weeks else None
    except Exception:
        return None


def lines(text: str) -> list[str] | None:
    """The comparison, or None when ``text`` does not compare two chains on their activity."""
    from argus.lui.research.parse import research_symbols
    from argus.lui.thesis import CHAINS, chain_activity

    if not ASKED.search(text):
        return None
    named = [s for s in research_symbols(text)[0] if s.removesuffix("USDT") in CHAINS][:2]
    if len(named) < 2:
        return None
    from argus.truth import http

    ids = [_GECKO.get(s.removesuffix("USDT")) for s in named]
    try:
        caps = http.fetch_json("https://api.coingecko.com/api/v3/simple/price",
                               params={"ids": ",".join(i for i in ids if i), "vs_currencies": "usd",
                                       "include_market_cap": "true"}, timeout=20.0) or {}
    except Exception:
        caps = {}
    rows: list[Row] = []
    for symbol, gecko in zip(named, ids, strict=True):
        ticker = symbol.removesuffix("USDT")
        chain = CHAINS[ticker]
        activity = chain_activity(symbol) or {}
        fees = _thirty(chain, "fees")
        dex = _thirty(chain, "dexs")
        cap = float((caps.get(gecko) or {}).get("usd_market_cap") or 0) if gecko else 0.0
        commits = _commits(_CLIENT[chain]) if chain in _CLIENT else None
        rows.append({"ticker": ticker, "chain": chain, "fees": fees, "dex": dex, "cap": cap,
                     "yield_": (fees * 365 / 30 / cap) if fees and cap else None,
                     "fee_change": (activity.get("fees") or {}).get("change"),
                     "fee_rank": (activity.get("fees") or {}).get("percentile"),
                     "tvl_change": (activity.get("tvl") or {}).get("change"),
                     "commits": commits, "repo": _CLIENT.get(chain)})
    if not any(r["fees"] for r in rows):
        return ["Bottom line: DeFiLlama's fee data did not answer just now for "
                + " or ".join(r["chain"] for r in rows) + "; ask again in a minute."]

    def money(x: float | None) -> str:
        if not x:
            return "n/a"
        return f"${x / 1e9:,.2f}bn" if x >= 1e9 else f"${x / 1e6:,.0f}m"

    priced = [r for r in rows if r["yield_"]]
    out = []
    if len(priced) == 2:
        rich = max(priced, key=lambda r: r["yield_"] or 0.0)
        poor = min(priced, key=lambda r: r["yield_"] or 0.0)
        out.append(f"Bottom line: on user fees per dollar of market value, {rich['ticker']} "
                   f"is the better value now — its last 30 days of fees run at "
                   f"{rich['yield_'] or 0.0:.2%} of its market value a year, against "
                   f"{poor['yield_'] or 0.0:.2%} for {poor['ticker']}; neither "
                   f"figure is paid out to holders in full.")
    else:
        out.append("Bottom line: the fee comparison is below; a market value did not answer, so "
                   "fees per dollar of value could not be set side by side.")
    for r in rows:
        change = (f", {r['fee_change']:+.1%} on the 30 days before" +
                  (f" ({r['fee_rank']:.0%} of its past months grew less)" if r["fee_rank"]
                   is not None else "")) if r["fee_change"] is not None else ""
        out.append(f"{r['chain']}: user fees {money(r['fees'])} in the last 30 days{change}; DEX "
                   f"volume {money(r['dex'])}; TVL "
                   + (f"{r['tvl_change']:+.1%} on the month" if r["tvl_change"] is not None
                      else "n/a")
                   + (f"; market value {money(r['cap'])}" if r["cap"] else "") + ".")
    dev = [r for r in rows if r["commits"] is not None]
    if dev:
        out.append("Developers (a proxy — commits in the last four weeks to one main client): "
                   + "; ".join(f"{r['chain']} {r['commits']} in {r['repo']}" for r in dev)
                   + ". One repository is not a chain's whole developer base.")
    out.append("Risk sits beside value: ask \"compare SOL and ETH\" for each one's volatility and "
               "drawdowns. Data: DeFiLlama (fees, DEX volume, TVL), CoinGecko market value, GitHub "
               "participation stats, read now. Analysis, not advice.")
    return out
