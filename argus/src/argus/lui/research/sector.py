"""A stock's valuation against its sector: the question "is AAPL's P/E high for its sector" had no
benchmark in the answer (stranger QA, 2026-09-29).

The benchmark is the sector's SPDR fund. Yahoo Finance publishes each fund's holdings-weighted
valuation in ``topHoldings.equityHoldings``, stated as yields: ``priceToEarnings`` is earnings over
price (XLK read 0.03029 on 2026-09-29, a P/E of 33.0). The convention was checked rather than
assumed: rebuilt from XLK's ten largest holdings, each weighted by its share and its own trailing
P/E from Yahoo, the harmonic P/E is 35.5 over the 58% of the fund those ten names hold, beside the
33.0 the whole fund implies. The stock's own figures are Yahoo's ``summaryDetail`` and
``defaultKeyStatistics``, so both sides come from one source and one definition.

When the answer already shows the stock's multiples from Bitget's data service, those are passed in
as ``own`` and used here instead, so one answer never prints two different P/Es for the same stock
(NVDA read 29.1 from Yahoo beside 28.6 from Bitget on 2026-09-29, first-user audit). The line then
names both sources.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from argus.lui.answer import Source
from argus.lui.trace import trace_module

SECTOR_FUND: dict[str, str] = {
    "Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
}
"""Yahoo's eleven sector names to the Select Sector SPDR fund that holds that sector of the S&P
500."""


def _raw(block: Any, key: str) -> float | None:
    value = (block or {}).get(key)
    value = value.get("raw") if isinstance(value, dict) else value
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def sector_valuation(ticker: str, fetch: Any = None,
                     own: Mapping[str, float | None] | None = None) -> tuple[str, Source] | None:
    """One line setting ``ticker``'s trailing P/E and P/B beside its sector fund's, or ``None``
    when the sector or either side's figures are not published (a fund, an unlisted sector).

    ``own`` (keys ``pe`` and ``pb``) replaces Yahoo's figures for the stock with the ones the rest
    of the answer already shows; a key that is absent or ``None`` keeps Yahoo's."""
    if fetch is None:
        from argus.market.estimates import EstimatesSource

        fetch = EstimatesSource().summary
    profile = fetch(ticker, "assetProfile,summaryDetail,defaultKeyStatistics")
    sector = str((profile.get("assetProfile") or {}).get("sector") or "")
    fund = SECTOR_FUND.get(sector)
    if fund is None:
        return None
    held = (fetch(fund, "topHoldings").get("topHoldings") or {}).get("equityHoldings") or {}
    own_pe = _raw(profile.get("summaryDetail"), "trailingPE")
    own_pb = _raw(profile.get("defaultKeyStatistics"), "priceToBook")
    given = {k: v for k, v in (own or {}).items() if isinstance(v, (int, float)) and v > 0}
    own_pe, own_pb = given.get("pe", own_pe), given.get("pb", own_pb)
    fund_pe = (1 / y) if (y := _raw(held, "priceToEarnings")) else None
    fund_pb = (1 / y) if (y := _raw(held, "priceToBook")) else None
    parts = []
    for label, stock, bench in (("trailing P/E", own_pe, fund_pe), ("P/B", own_pb, fund_pb)):
        if stock is not None and bench is not None:
            ratio = stock / bench
            rel = f"{ratio:.1f} times" if ratio >= 2 else f"{ratio - 1:+.0%}"
            parts.append(f"{label} {stock:.1f} against {bench:.1f} ({rel})")
    if not parts:
        return None
    # A plain reading for "is it expensive": on trailing earnings, against the sector, with a band
    # of 10% either side called "in line" so a few points of difference are not called a verdict.
    verdict = ""
    if own_pe is not None and fund_pe is not None:
        ratio = own_pe / fund_pe
        verdict = ("priced above its sector on earnings" if ratio > 1.1 else
                   "priced below its sector on earnings" if ratio < 0.9 else
                   "priced in line with its sector on earnings")
        verdict = f"{ticker} is {verdict}: "
    line = (f"Against its sector ({sector}, measured by the SPDR fund {fund}): {verdict}"
            f"{'; '.join(parts)}. "
            + ("The stock's figures are Bitget's data service's, the fund's Yahoo Finance's"
               if given else "Both from Yahoo Finance")
            + "; the fund's figure is its holdings-weighted earnings and book yield, inverted.")
    return line, Source(kind="venue", ref="Yahoo Finance quoteSummary (sector fund)",
                        detail=f"{ticker} against {fund} topHoldings.equityHoldings")


# Every engine here is a traced step from import on (lui/trace.py, trace_module).
trace_module(globals())
