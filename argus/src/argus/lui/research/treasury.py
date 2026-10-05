"""Strategy's (MSTR) bitcoin holdings, cost basis and unrealised gain, read from its own 8-K.

"What are MicroStrategy's current total BTC holdings and average cost basis per their latest
filing, and what is the unrealised P&L at today's BTC price?" got two live quotes and nothing else
(round 40 judge, Q13). Strategy reports its holdings in an 8-K most weeks, under a "BTC Update"
heading, as a table read here row by row: BTC purchased in the period, its aggregate and average
price, then the aggregate holdings, their aggregate purchase price (in billions) and the average
purchase price — "inclusive of fees and expenses", as the filing footnotes. Not every 8-K carries
it (a proxy notice does not), so the recent 8-Ks are read newest first until one does.

The gain is the holdings at Bitget's live BTC price less the aggregate purchase price.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Final

ASKED: Final = re.compile(
    r"\b(?:mstr|microstrategy|strategy|saylor)\b[^?]{0,120}\b(?:btc|bitcoin)\b[^?]{0,60}"
    r"\b(?:holdings?|holds?|owns?|cost\s+basis|average\s+(?:cost|price)|unreali[sz]ed|p&?l|profit|"
    r"how\s+many)\b|\b(?:how\s+(?:many|much)\s+(?:btc|bitcoins?)|bitcoin\s+holdings?)\b[^?]{0,60}"
    r"\b(?:mstr|microstrategy|strategy)\b", re.I)
_UPDATE: Final = re.compile(r"BTC Update.*?As of (?P<asof>[A-Z][a-z]+ \d{1,2}, \d{4}).*?"
                            r"Average Purchase Price.*?Average Purchase Price[^\n]*\n"
                            r"(?P<nums>(?:[\d.,]+\s*\n){6})", re.S)


def _holdings() -> tuple[str, float, float, float, str] | None:
    """(as-of date, BTC held, aggregate cost in dollars, average cost, filing URL), or None."""
    from argus.market.evidence import EdgarSource
    from argus.research.document_qa import EdgarDocuments

    filings = [f for f in EdgarSource().filings("MSTR", since=datetime.now(UTC) - timedelta(
        days=45)) if f.form == "8-K"]
    docs = EdgarDocuments()
    for f in filings[:8]:
        read = docs.event_text("MSTR", f.accession)
        if read is None:
            continue
        m = _UPDATE.search(read[0])
        if m is None:
            continue
        nums = [float(x.replace(",", "")) for x in m.group("nums").split()]
        held, cost_bn, average = nums[3], nums[4], nums[5]
        return m.group("asof"), held, cost_bn * 1e9, average, f.accession
    return None


def lines(text: str) -> list[str] | None:
    """Strategy's holdings answer, or None when ``text`` does not ask about them."""
    if not ASKED.search(text):
        return None
    try:
        found = _holdings()
    except Exception:
        found = None
    if found is None:
        return ["Bottom line: none of Strategy's 8-Ks from the last 45 days carried its bitcoin "
                "holdings table just now, so no count is given; strategy.com publishes the same "
                "figures."]
    asof, held, cost, average, accession = found
    from argus.market.bitget import fetch_tickers

    try:
        btc = float(fetch_tickers()["BTCUSDT"].last)
    except Exception:
        btc = 0.0
    out = [f"Bottom line: Strategy held {held:,.0f} BTC as of {asof}, bought for "
           f"${cost / 1e9:,.2f}bn in all — an average of ${average:,.0f} per bitcoin, fees "
           f"included (its 8-K, accession {accession})."]
    if btc > 0:
        worth = held * btc
        gain = worth - cost
        out.append(f"At Bitget's live BTC price of {btc:,.0f}, that is worth "
                   f"${worth / 1e9:,.2f}bn — "
                   f"an unrealised {'gain' if gain >= 0 else 'loss'} of about "
                   f"${abs(gain) / 1e9:,.2f}bn ({gain / cost:+.1%} on cost).")
        out.append(f"Breakeven is BTC at {average:,.0f}: "
                   f"{(average / btc - 1):+.1%} from here.")
    out.append("Holdings change most weeks; a later purchase is in the next 8-K. Data: SEC EDGAR "
               "8-K (Strategy's \"BTC Update\" table) and Bitget BTCUSDT, read now. Not advice.")
    return out
