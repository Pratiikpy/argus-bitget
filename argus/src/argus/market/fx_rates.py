"""Local-currency amounts in US dollars, for the currencies Bitget has no FX contract for.

Bitget lists FX perpetuals for the euro, the pound and the yen, and `lui/research/parse.py` reads
those at Bitget's own price. A newcomer writes in rupiah, dong, rupees or naira: "kalau saya beli 1
juta rupiah untung berapa" was read as one million *dollars* and answered with a $42,059 worst day
(round 45 newcomer, M7). A million rupiah is about fifty-six dollars.

Two keyless sources, read in this order:

1. **The European Central Bank's reference rates**, through Frankfurter
   (``api.frankfurter.dev/v1/latest``), which republishes the ECB's daily 16:00 CET fixing for
   about thirty currencies (IDR, INR, KRW, PHP, THB, MYR, BRL, TRY, ZAR, MXN, …). Official, dated,
   and the rate a reader can check against the ECB's own page.
2. **ExchangeRate-API's open endpoint** (``open.er-api.com/v6/latest/USD``), updated daily, for the
   currencies the ECB does not fix — VND, NGN, PKR, EGP, AED, SAR, BDT and the rest of about 160.

Both were read on 2026-10-06 and agreed within 0.15% on IDR (17,913 against 17,893) and INR
(96.30 against 96.38). A rate is cached for an hour; daily fixings do not move inside one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Final

from argus.truth.http import fetch_json

ECB_URL: Final = "https://api.frankfurter.dev/v1/latest"
OPEN_URL: Final = "https://open.er-api.com/v6/latest/USD"
_TTL: Final = 3600.0


@dataclass(frozen=True)
class FxRate:
    code: str
    per_usd: float
    """Units of the currency one US dollar buys."""
    source: str
    as_of: str

    def to_usd(self, amount: float) -> float:
        return amount / self.per_usd


_CACHE: dict[str, tuple[float, FxRate]] = {}


def _ecb(code: str) -> FxRate | None:
    body = fetch_json(ECB_URL, params={"base": "USD", "symbols": code}, timeout=6.0)
    rate = (body.get("rates") or {}).get(code)
    if not rate:
        return None
    return FxRate(code, float(rate), "the ECB reference rate", str(body.get("date") or ""))


def _open(code: str) -> FxRate | None:
    body = fetch_json(OPEN_URL, timeout=6.0)
    rate = (body.get("rates") or {}).get(code)
    if not rate or body.get("result") != "success":
        return None
    stamp = str(body.get("time_last_update_utc") or "")
    return FxRate(code, float(rate), "ExchangeRate-API's daily rate", stamp[5:16].strip())


def usd_rate(code: str) -> FxRate | None:
    """Units of ``code`` per US dollar, or None when neither source answers."""
    code = code.upper()
    held = _CACHE.get(code)
    if held is not None and time.monotonic() - held[0] < _TTL:
        return held[1]
    for source in (_ecb, _open):
        try:
            found = source(code)
        except Exception:
            found = None
        if found is not None and found.per_usd > 0:
            _CACHE[code] = (time.monotonic(), found)
            return found
    return None


__all__ = ["ECB_URL", "OPEN_URL", "FxRate", "usd_rate"]
