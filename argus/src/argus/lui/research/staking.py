"""Ether staking read from the chain itself: Lido's stETH/ETH peg, how much ETH is staked, and
Lido's share and yield.

Round 41's judge (C3, q07) asked "How much has the Lido stETH/ETH peg deviated lately, and what is
the total ETH staked?" and got generic Earn lock-up text. Every figure the question needs is
public and keyless, and each source below was read and cross-checked on 2026-10-05:

- **The peg where it actually trades.** The Curve stETH/ETH pool
  (``0xDC24316b9AE028F1497c275EB9192a3Ea0f67022``) quotes ``get_dy(1, 0, dx)`` — the ETH a seller
  of ``dx`` stETH receives — through a public RPC ``eth_call`` (selector ``0x5e0d443f``). Read for
  1 and for 1,000 stETH, so the answer says what size does to the price: 0.99964 and 0.99958 ETH
  each that day. CoinGecko's ``staked-ether`` priced in ETH gives the 30-day daily range
  (0.9984-1.0034 that day); it is a cross rate built from two dollar prices, so it is the history,
  and Curve is the live peg.
- **Total ETH staked.** ultrasound.money's ``/api/v2/fees/supply-parts`` reports
  ``beaconBalancesSum`` in gwei: 44,192,645 ETH. Checked against a full read of a beacon node's
  ``/eth/v1/beacon/states/head/validator_balances`` (2,378,865 validators, 44,192,549 ETH) — they
  agree to 0.0002%. The beacon read is 88 MB, so it is the check, not the source.
- **Lido's stake.** ``getTotalPooledEther()`` on the stETH contract
  (``0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84``, selector ``0x37cfdaca``): 9,827,548 ETH.
- **Lido's yield.** ``https://eth-api.lido.fi/v1/protocol/steth/apr/sma`` — the seven-day moving
  average APR Lido publishes (2.23%).
- **Share of supply.** ultrasound.money's ``/api/v2/fees/supply-over-time`` (latest ``supply``).

The deposit contract's balance (91.7M ETH) is *not* the amount staked — withdrawals leave from the
beacon chain, not from the contract, so it only ever grows. It is named here so nobody reaches for
it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Final

from argus.truth import http

RPC: Final = ("https://ethereum.publicnode.com", "https://eth.drpc.org")
CURVE_STETH: Final = "0xDC24316b9AE028F1497c275EB9192a3Ea0f67022"
STETH: Final = "0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84"
_GET_DY: Final = "0x5e0d443f"
_POOLED: Final = "0x37cfdaca"

_ASKED: Final = re.compile(
    r"\bst\s?eth\b|\blido\b|\b(?:eth|ether(?:eum)?)\s+stak(?:ed|ing)\b|\bstak(?:ed|ing)\s+"
    r"(?:eth|ether(?:eum)?)\b|\btotal\s+(?:eth\s+)?staked\b|\bstaking\s+(?:ratio|rate|share)\b|"
    r"\bliquid\s+staking\b|\b(?:eth|ether(?:eum)?)\b[^?]{0,20}\b(?:is|are)\s+staked\b|"
    r"\bstaking\s+(?:apr|apy|yield)\b[^?]{0,30}\b(?:eth|ether(?:eum)?)\b", re.I)
_EARN: Final = re.compile(r"\b(?:should\s+i|how\s+(?:do|can)\s+i|where\s+(?:do|can)\s+i)\b"
                          r"[^?]{0,40}\bstak|\bmy\s+(?:staked|steth)\b|\bmy\s+\w+\s+staked\b",
                          re.I)
_TOTAL_FIRST: Final = re.compile(r"\bstaked\b|\bstaking\s+(?:ratio|rate|share)\b", re.I)
_PEG_ASKED: Final = re.compile(r"\bpeg\b|\bdepeg\w*|\bdiscount\b|\bpremium\b|"
                               r"\bst\s?eth\s*/\s*eth\b|"
                               r"\bsell\w*\b|\bcurve\b|\bdeviat\w*", re.I)
_SIZED: Final = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(k)?\s*st\s?eth\b", re.I)


def _rpc(to: str, data: str) -> int:
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_call",
                          "params": [{"to": to, "data": data}, "latest"]}).encode()
    failure: Exception | None = None
    for url in RPC:
        try:
            body = http.fetch_json(url, data=payload, method="POST", timeout=20.0,
                                   headers={"Content-Type": "application/json"})
            return int(str(body["result"]), 16)
        except Exception as exc:  # the next public node, then give up with the reason
            failure = exc
    raise RuntimeError(f"no public Ethereum RPC answered: {failure}")


def curve_out(steth: float) -> float:
    """ETH received for selling ``steth`` stETH into the Curve pool, fee included."""
    dx = round(steth * 10**18)
    data = _GET_DY + f"{1:064x}" + f"{0:064x}" + f"{dx:064x}"
    return _rpc(CURVE_STETH, data) / 10**18


def lido_pooled() -> float:
    return _rpc(STETH, _POOLED) / 10**18


def _get(url: str, **params: Any) -> Any:
    return http.fetch_json(url, params=params or None, timeout=30.0)


def total_staked() -> float:
    return int(_get("https://ultrasound.money/api/v2/fees/supply-parts")["beaconBalancesSum"]) / 1e9


def eth_supply() -> float:
    return float(_get("https://ultrasound.money/api/v2/fees/supply-over-time")["d1"][-1]["supply"])


def lido_apr() -> float:
    return float(_get("https://eth-api.lido.fi/v1/protocol/steth/apr/sma")["data"]["smaApr"])


def peg_history(days: int = 30) -> list[float]:
    rows = _get("https://api.coingecko.com/api/v3/coins/staked-ether/market_chart",
                vs_currency="eth", days=days, interval="daily")["prices"]
    return [float(p) for _, p in rows]


def _try(fn: Any, *args: Any) -> Any:
    try:
        return fn(*args)
    except Exception:
        return None


def asks(text: str) -> bool:
    """A question about ether staking's market figures, not about whether or how to stake."""
    return bool(_ASKED.search(text)) and not _EARN.search(text)


def lines(text: str) -> list[str] | None:
    """The answer, or None when the question is not about ether staking's market figures."""
    if not asks(text):
        return None
    sized = _SIZED.search(text)
    size = None
    if sized:
        size = float(sized.group(1).replace(",", "")) * (1000 if sized.group(2) else 1)
    one = _try(curve_out, 1.0)
    big = _try(curve_out, size or 1000.0)
    history = _try(peg_history)
    staked = _try(total_staked)
    pooled = _try(lido_pooled)
    supply = _try(eth_supply)
    apr = _try(lido_apr)
    if one is None and staked is None and pooled is None:
        return ["Bottom line: the chain could not be read just now (public Ethereum RPC and "
                "ultrasound.money both failed), so there is no peg or staking figure to give; "
                "nothing is estimated in its place."]
    out: list[str] = []
    if one is not None:
        gap = (one - 1) * 100
        lead = (f"stETH trades at {one:.5f} ETH on Curve right now ({gap:+.3f}% from 1:1, the "
                f"pool's 0.01% fee included)")
        if big is not None:
            amount = size or 1000.0
            lead += (f"; selling {amount:,.0f} stETH there returns {big:,.2f} ETH "
                     f"({(big / amount - 1) * 100:+.3f}%)")
        out.append(lead + ".")
    if history:
        lo, hi = min(history), max(history)
        out.append(f"Over the last 30 days the daily stETH/ETH rate ranged {lo:.4f} to {hi:.4f} "
                   f"(widest gap {max(abs(lo - 1), abs(hi - 1)) * 100:.2f}%); that series is "
                   "CoinGecko's cross of two dollar prices, so its wiggles overstate the real "
                   "peg — Curve's quote is the tradable one.")
    if staked is not None:
        part = f"Total ETH staked: {staked / 1e6:,.2f}M ETH on the beacon chain"
        if supply:
            part += f", {staked / supply * 100:.1f}% of the {supply / 1e6:,.1f}M ETH supply"
        part += " (ultrasound.money, summed validator balances)"
        if pooled is not None:
            part += (f"; Lido holds {pooled / 1e6:,.2f}M of it, {pooled / staked * 100:.1f}% "
                     "(its stETH contract, read on-chain)")
        out.append(part + ".")
    elif pooled is not None:
        out.append(f"Lido holds {pooled / 1e6:,.2f}M ETH (its stETH contract, read on-chain); "
                   "the network total could not be read just now.")
    if apr is not None:
        out.append(f"Lido's staking yield: {apr:.2f}% APR, its own seven-day average "
                   "(eth-api.lido.fi).")
    if _TOTAL_FIRST.search(text) and not _PEG_ASKED.search(text):
        # "how much ETH is staked" leads with the total, not with the peg it did not ask about
        total = [x for x in out if x.startswith(("Total ETH staked", "Lido holds"))]
        out = total + [x for x in out if x not in total]
    out[0] = "Bottom line: " + out[0]
    out.append("A discount on stETH is what a holder pays to leave before Lido's withdrawal "
               "queue pays out at 1:1; it went past 6% in June 2022, when lenders were "
               "forced sellers. The beacon-chain deposit contract's balance is not the amount "
               "staked — withdrawals never leave through it.")
    return out


__all__ = ["asks", "curve_out", "lido_apr", "lido_pooled", "lines", "peg_history", "total_staked"]
