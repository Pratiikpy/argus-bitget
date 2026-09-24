"""Every contract Bitget lists on its USDT-futures book, and the names people call them by.

The desk decides on twelve rTokens. The research console is not the desk: a trader asking where
gold is trading, whether PLTR is overbought, or what adding BTC does to a book of US stocks is
asking a research question about a contract Bitget really lists, and the right answer is the
analysis, not a refusal.

**Why this module exists.** Until 2026-09-23 the console told visitors that gold, silver and oil
"are not listed on Bitget". Asked live, ``/api/v2/mix/market/contracts?productType=USDT-FUTURES``
returned 805 contracts including ``XAUUSDT`` (gold, last 4,316), ``XAGUSDT`` (silver), ``CLUSDT``
(WTI crude, 91.09), ``BZUSDT`` (Brent), ``SPYUSDT``, ``PLTRUSDT``, ``AMDUSDT`` and ``NFLXUSDT``. The
refusal table in :mod:`argus.lui.question` had been written from memory, and every one of those
lines was false. The registry below is read from the venue instead.

**What the venue itself says about each contract.** The contracts endpoint carries ``isRwa``:
"YES" on 339 contracts (US and Asian equities, ETFs, commodities, FX pairs, index products) and
"NO" on 466 (crypto). That flag settles the trap recorded in ``argus/market/bitget.py``, where
``SPXUSDT`` was once taken for the S&P 500: it is flagged "NO" — SPX6900, a memecoin — while the
tokenised S&P 500 is ``SP500USDT`` (last 7,757). Seven US stocks whose ticker is already held by
a crypto token are listed under a ``STOCK`` suffix (``CVXSTOCKUSDT`` is Chevron, ``CVXUSDT`` is
Convex); :func:`resolve` reads the bare ticker as the stock and says so.

Live first, frozen second, as everywhere else in this codebase: the registry is fetched once and
cached for :data:`CACHE_TTL_S`; if the venue does not answer, the snapshot in
``data/venue_universe.json`` (written by :func:`freeze`, dated inside the file) is used.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

CONTRACTS_URL = ("https://api.bitget.com/api/v2/mix/market/contracts"
                 "?productType=USDT-FUTURES")
SNAPSHOT_PATH = Path(__file__).resolve().parents[3] / "data" / "venue_universe.json"
CACHE_TTL_S = 6 * 3600.0
FETCH_TIMEOUT_S = 6.0

_CACHE: tuple[float, dict[str, Contract], str] | None = None
_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class Contract:
    symbol: str
    rwa: bool
    """The venue's own ``isRwa`` flag: a real-world asset (equity, ETF, commodity, FX, index)
    rather than a crypto token."""
    funding_hours: int | None = None
    """Hours between funding settlements (``fundInterval``): 8 on most contracts, 4 on 377 of them
    including gold, 1 on two (2026-09-24). Needed to turn a per-interval rate into a daily cost."""

    @property
    def name(self) -> str:
        """The ticker a trader reads: ``PLTR``, ``CVX`` for ``CVXSTOCKUSDT``."""
        base = self.symbol.removesuffix("USDT")
        return base.removesuffix("STOCK") if self.rwa and base.endswith("STOCK") else base


# Names people type for a contract whose ticker they would not guess. Only unambiguous mappings:
# each target was checked against the live book on 2026-09-23 (price shown) so the alias cannot
# point at a memecoin the way SPX once did.
ALIASES: dict[str, str] = {
    "GOLD": "XAUUSDT",          # 4,316
    "XAU": "XAUUSDT",
    "SILVER": "XAGUSDT",        # 65.37
    "XAG": "XAGUSDT",
    "PLATINUM": "XPTUSDT",      # 1,773
    "PALLADIUM": "XPDUSDT",
    "OIL": "CLUSDT",            # WTI, 91.09
    "CRUDE": "CLUSDT",
    "WTI": "CLUSDT",
    "BRENT": "BZUSDT",          # 96.41
    "NATGAS": "NATGASUSDT",
    "COPPER": "COPPERUSDT",
    "SPX": "SP500USDT",         # 7,757 — not SPXUSDT, which is SPX6900 at 0.50
    "SP500": "SP500USDT",
    "NASDAQ": "NDX100USDT",     # 30,662
    "NDX": "NDX100USDT",
    "DOW": "DIASTOCKUSDT",      # the Dow ETF, 515.78 — DIAUSDT is a crypto token at 0.15
    "DIA": "DIASTOCKUSDT",
    "NIKKEI": "JP225USDT",
    "BITCOIN": "BTCUSDT",
    "BTC": "BTCUSDT",
    "ETHEREUM": "ETHUSDT",
    "ETH": "ETHUSDT",
    "SOLANA": "SOLUSDT",
    "SOL": "SOLUSDT",
    "DOGECOIN": "DOGEUSDT",
    "DOGE": "DOGEUSDT",
    "PALANTIR": "PLTRUSDT",
    "NETFLIX": "NFLXUSDT",
    "BROADCOM": "AVGOUSDT",
    "ORACLE": "ORCLUSDT",
    "ROBINHOOD": "HOODUSDT",
    "GAMESTOP": "GMEUSDT",
    "CHEVRON": "CVXSTOCKUSDT",
    "ALIBABA": "BABAUSDT",
    "INTEL": "INTCUSDT",
    "AMAZON": "AMZNUSDT",
    "NVIDIA": "NVDAUSDT",
}

NOT_EQUITY: dict[str, str] = {
    **dict.fromkeys(("XAUUSDT", "XAUTUSDT", "PAXGUSDT", "XAGUSDT", "XPTUSDT", "XPDUSDT", "CLUSDT",
                     "BZUSDT", "NATGASUSDT", "COPPERUSDT"), "commodity"),
    **dict.fromkeys(("EURUSDUSDT", "GBPUSDUSDT", "USDJPYUSDT", "USDBRLUSDT"), "fx"),
    **dict.fromkeys(("SP500USDT", "NDX100USDT", "HSIUSDT", "JP225USDT", "KR200USDT"), "index"),
}
"""Real-world-asset contracts that are not a company's shares. The venue flags them ``isRwa`` like
any stock, and several share a ticker with an unrelated US company — ``CL`` is Colgate-Palmolive on
the NYSE and WTI crude here, ``BZ`` is Kanzhun and Brent — so an equity data server asked about
them answers confidently about the wrong thing. Anything listed here is kept away from earnings,
13F and stock-premium lookups."""


def is_equity(symbol: str) -> bool:
    """A company's shares or an ETF: listed as a real-world asset and not a commodity, FX pair or
    index product."""
    contract = contracts().get(symbol)
    return contract is not None and contract.rwa and symbol not in NOT_EQUITY


CJK_ALIASES: dict[str, str] = {
    "英伟达": "NVDAUSDT", "辉达": "NVDAUSDT", "特斯拉": "TSLAUSDT", "苹果": "AAPLUSDT",
    "微软": "MSFTUSDT", "谷歌": "GOOGLUSDT", "亚马逊": "AMZNUSDT", "脸书": "METAUSDT",
    "英特尔": "INTCUSDT", "奈飞": "NFLXUSDT", "阿里巴巴": "BABAUSDT", "比特币": "BTCUSDT",
    "以太坊": "ETHUSDT", "黄金": "XAUUSDT", "白银": "XAGUSDT", "原油": "CLUSDT",
    "纳斯达克": "NDX100USDT", "标普": "SP500USDT", "微策略": "MSTRUSDT",
}
"""Chinese names for the contracts a Chinese-speaking trader asks about most. "英伟达现在值得买吗?"
(is Nvidia worth buying now?) was answered with the session clock because no Latin ticker appeared
(a judge's probe, 2026-09-24). Matched as substrings, since Chinese has no spaces."""

FUTURES_CODES: dict[str, str] = {
    "GC": "XAUUSDT", "SI": "XAGUSDT", "PA": "XPDUSDT", "HG": "COPPERUSDT", "NG": "NATGASUSDT",
    "ES": "SP500USDT", "NQ": "NDX100USDT",
}
"""CME/COMEX root symbols a futures trader types for the same underlying. Checked free on Bitget's
list on 2026-09-23 — no contract of these names exists there, so the mapping shadows nothing; PL
(platinum) is left out because PLUSDT is Planet Labs. Unlike :data:`ALIASES` these are honoured
only as written in capitals, via the ticker path: "es" is Spanish for "is" and "si" for "if".
Found on the fourth blind corpus, whose author wrote copper as HG and natural gas as NG."""

ALIAS_NOTES: dict[str, str] = {
    "SPX": "SPX read as the S&P 500 (SP500USDT); Bitget's SPXUSDT is SPX6900, a memecoin",
    "DIA": "DIA read as the Dow ETF (DIASTOCKUSDT); Bitget's DIAUSDT is a crypto token",
    "DOW": "the Dow read as its ETF, DIA (DIASTOCKUSDT)",
    "OIL": "oil read as WTI crude (CLUSDT); say Brent for BZUSDT",
    "CRUDE": "crude read as WTI (CLUSDT); say Brent for BZUSDT",
    "NASDAQ": "the Nasdaq read as the Nasdaq-100 index product (NDX100USDT)",
}


def _parse(rows: list[dict[str, object]]) -> dict[str, Contract]:
    out: dict[str, Contract] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "")
        if not symbol.endswith("USDT") or str(row.get("symbolStatus")) != "normal":
            continue
        try:
            hours: int | None = int(str(row.get("fundInterval")))
        except ValueError:
            hours = None
        out[symbol] = Contract(symbol=symbol, rwa=str(row.get("isRwa")) == "YES",
                               funding_hours=hours)
    return out


def _fetch_live() -> dict[str, Contract]:
    req = urllib.request.Request(CONTRACTS_URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("code") not in ("00000", 0, None):
        raise RuntimeError(f"contracts code {payload.get('code')}: {payload.get('msg')}")
    contracts = _parse(payload.get("data") or [])
    if len(contracts) < 100:
        raise RuntimeError(f"only {len(contracts)} contracts came back")
    return contracts


def _from_snapshot() -> tuple[dict[str, Contract], str]:
    snap = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    contracts = {s: Contract(symbol=s, rwa=bool(v["rwa"]), funding_hours=v.get("funding_hours"))
                 for s, v in snap.get("contracts", {}).items()}
    return contracts, str(snap.get("generated_at", "an unrecorded date"))[:10]


def contracts() -> Mapping[str, Contract]:
    """Every listed USDT-futures contract, live within the last :data:`CACHE_TTL_S`, else frozen."""
    global _CACHE
    with _LOCK:
        if _CACHE is not None and time.monotonic() - _CACHE[0] < CACHE_TTL_S:
            return _CACHE[1]
    try:
        found, origin = _fetch_live(), "live"
    except (OSError, ValueError, RuntimeError, urllib.error.URLError):
        found, frozen_on = _from_snapshot()
        origin = f"frozen {frozen_on}"
    with _LOCK:
        # A frozen registry is retried sooner than a live one: it is a stopgap, not an answer.
        stamp = time.monotonic() - (0.0 if origin == "live" else CACHE_TTL_S - 300.0)
        _CACHE = (stamp, found, origin)
    return found


def origin() -> str:
    """Where the current registry came from: "live" or "frozen <date>"."""
    contracts()
    return _CACHE[2] if _CACHE is not None else "unknown"


def resolve(token: str) -> tuple[str, str] | None:
    """A word to a listed contract and a note on how it was read, or None if Bitget lists nothing
    by that name.

    ``PLTR``, ``pltrusdt``, ``palantir`` and ``gold`` all resolve. A US stock whose ticker a crypto
    token already holds resolves to the stock (``CVX`` → ``CVXSTOCKUSDT``), with a note naming the
    other contract, because this is an equities research console; the full symbol ``CVXUSDT``
    still reaches the token.
    """
    upper = token.strip().upper()
    if not upper:
        return None
    listed = contracts()
    if upper in ALIASES and ALIASES[upper] in listed:
        return ALIASES[upper], ALIAS_NOTES.get(upper, "")
    if upper.endswith("USDT") and upper in listed:
        return upper, ""
    if upper in FUTURES_CODES and FUTURES_CODES[upper] in listed:
        target = FUTURES_CODES[upper]
        return target, f"{upper} read as its futures underlying ({target})"
    stock = f"{upper}STOCKUSDT"
    if stock in listed:
        other = f"{upper}USDT"
        note = (f"{upper} read as the stock ({stock}); {other} is a different, crypto contract"
                if other in listed else "")
        return stock, note
    if f"{upper}USDT" in listed:
        return f"{upper}USDT", ""
    return None


def freeze(path: Path = SNAPSHOT_PATH) -> int:
    """Write the live registry to ``path`` so an offline answer still knows what Bitget lists."""
    found = _fetch_live()
    path.write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(),
        "source": CONTRACTS_URL,
        "contracts": {s: {"rwa": c.rwa, "funding_hours": c.funding_hours}
                      for s, c in sorted(found.items())},
    }, indent=1) + "\n", encoding="utf-8", newline="\n")
    return len(found)


def main() -> int:
    count = freeze()
    rwa = sum(1 for c in contracts().values() if c.rwa)
    print(f"{count} contracts written to {SNAPSHOT_PATH} ({rwa} real-world assets)")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ALIASES",
    "ALIAS_NOTES",
    "FUTURES_CODES",
    "NOT_EQUITY",
    "Contract",
    "contracts",
    "freeze",
    "is_equity",
    "origin",
    "resolve",
]
