"""A covered call against a cash-secured put, priced on the live chain, for a holder of the stock.

"Compare the risk of a covered call strategy versus a cash-secured put strategy on AAPL for
someone who already owns 100 shares — which has worse tail risk?" got AAPL's options statistics and
then its concentration in the book, the two strategies never compared (a judge, round 36).

The comparison rests on put-call parity: at the same strike and expiry, long stock plus a short call
has the same payoff as cash plus a short put, apart from carry (Hull, *Options, Futures and Other
Derivatives*, ch. 11). So for a trader starting from cash the two have the same tail. The question
names a trader who already owns the shares, and that changes the answer: the covered call is
written against the shares held, while a cash-secured put written beside them commits cash to buy
a second hundred at the strike — on a fall, twice the shares. Both are set out with the chain's own
prices: Cboe's delayed chain (`market/options.fetch_chain`), the expiry nearest thirty days out,
the strike nearest the price, each leg at its mid, and the fall that the 30-day implied volatility
puts at two standard deviations over the time to expiry.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from typing import Final

from argus.lui.trace import trace_module

ASKED: Final = re.compile(
    r"\bcovered\s+calls?\b[^?]{0,120}\b(?:cash[\s-]+secured\s+puts?|csps?|short\s+puts?|selling\s+"
    r"puts?)\b|\b(?:cash[\s-]+secured\s+puts?|csps?)\b[^?]{0,120}\bcovered\s+calls?\b", re.I)
SHARES: Final = 100
"""One listed contract's multiplier."""


def lines(text: str) -> list[str] | None:
    if not ASKED.search(text):
        return None
    from argus.lui.research import research_symbols
    from argus.lui.research.parse import is_us_equity
    from argus.market.options import (
        MIN_DAYS_TO_EXPIRY,
        NEW_YORK,
        fetch_chain,
        parse_contract,
    )

    stocks = [s for s in research_symbols(text)[0] if is_us_equity(s)]
    if not stocks:
        return None
    ticker = stocks[0].removesuffix("USDT")
    try:
        payload = fetch_chain(ticker)
    except Exception:
        return [f"Bottom line: the {ticker} option chain (Cboe) did not answer just now, so the "
                f"two strategies cannot be priced; by put-call parity they carry the same payoff "
                f"at the same strike and expiry, apart from the shares you already hold."]
    data = payload.get("data") or {}
    spot = float(data.get("current_price") or data.get("close") or 0.0)
    iv30 = data.get("iv30")
    today = datetime.now(UTC).astimezone(NEW_YORK).date()
    quoted = [c for c in (parse_contract(r) for r in data.get("options") or []) if c
              and c.bid > 0 and c.ask > 0 and (c.expiry - today).days >= MIN_DAYS_TO_EXPIRY]
    if spot <= 0 or not quoted:
        return None
    # the monthly a covered-call writer uses: the expiry nearest thirty days out
    expiry = min({c.expiry for c in quoted}, key=lambda e: abs((e - today).days - 30))
    at = [c for c in quoted if c.expiry == expiry]
    strikes = sorted({c.strike for c in at if c.right == "C"} & {c.strike for c in at
                                                                 if c.right == "P"})
    if not strikes:
        return None
    strike = min(strikes, key=lambda k: abs(k - spot))
    call = next(c for c in at if c.right == "C" and c.strike == strike)
    put = next(c for c in at if c.right == "P" and c.strike == strike)
    days = (expiry - today).days
    sigma = float(iv30) / 100 if iv30 else None
    fall = (spot * math.exp(-2 * sigma * math.sqrt(days / 365)) if sigma else spot * 0.8)
    fall_said = (f"a fall to {fall:,.2f}, two standard deviations of the {sigma:.0%} implied "
                 f"volatility over {days} days" if sigma else f"a 20% fall to {fall:,.2f}")

    def covered(at_price: float) -> float:
        return SHARES * ((min(at_price, strike) - spot) + call.mid)

    def put_beside(at_price: float) -> float:
        # the 100 shares already held, plus a short put on another 100
        return SHARES * ((at_price - spot) + put.mid - max(strike - at_price, 0.0))

    cc, csp = covered(fall), put_beside(fall)
    cash_only = SHARES * (put.mid - max(strike - fall, 0.0))
    worse = "the cash-secured put" if csp < cc else "the covered call"
    return [
        f"Bottom line: for a holder of the 100 shares, {worse} has the worse tail — on "
        f"{fall_said}, the covered call loses about ${-cc:,.0f} and the shares plus a "
        f"cash-secured put lose about ${-csp:,.0f}, because the put commits you to buy a second "
        f"100 at {strike:g} while you keep the first.",
        f"Priced on Cboe's delayed {ticker} chain, {expiry:%d %b %Y} expiry ({days} days), strike "
        f"{strike:g} against a price of {spot:,.2f}: the call bids {call.bid:.2f} / asks "
        f"{call.ask:.2f}, the put {put.bid:.2f} / {put.ask:.2f}; each leg at its mid.",
        f"From cash, the two are the same trade: by put-call parity a covered call and a "
        f"cash-secured put at one strike and expiry pay the same, apart from carry — the put "
        f"alone on that fall loses about ${-cash_only:,.0f} against the covered call's "
        f"${-cc:,.0f} (the gap is the premium difference, {call.mid - put.mid:+.2f} a share, "
        f"less the strike's distance from the price).",
        f"Upside: the covered call caps the shares at {strike:g} (plus {call.mid:.2f} premium); "
        f"the put keeps the shares' upside and adds only its {put.mid:.2f} premium.",
        "A listed option is not a Bitget product; Bitget's own route to a stock is the perpetual "
        "and the rToken, neither of which has options. Data: Cboe delayed quotes.",
    ]


trace_module(globals())
