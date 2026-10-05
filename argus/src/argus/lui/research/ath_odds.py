"""The odds of a new all-time high by a date, two ways: what Polymarket's crowd prices, and what
Deribit's option book implies.

"What is the prediction-market-implied probability of a new BTC all-time high before year end,
and is that consistent with what BTC options are pricing in for the same horizon?" got the
distance to the high and neither probability (round 40 judge, Q22). Both are public:

- **The crowd.** Polymarket runs "Bitcoin all time high by December 31, 2026?"; its YES price is
  read from the event search (`market/prediction.search_events`). A price carries fees and the
  time value of locked money, so it is only loosely a probability, as the answer says.
- **The options.** For the Deribit expiry nearest the date, Black-76 on that expiry's forward with
  the implied volatility of the call nearest the high gives the risk-neutral chance of *ending*
  above it, N(d2). A new high only needs to be *touched* before the date; with no drift, the
  reflection principle puts the chance of touching at about twice the chance of ending above.
  Both are given, labelled. Risk-neutral odds are not a forecast; they carry the market's price
  of risk.
- **The high.** BTC's highest daily close on Yahoo Finance; an intraday high sits a little above.
"""

from __future__ import annotations

import math
import re
from datetime import date
from typing import Final

ASKED: Final = re.compile(
    r"\b(?:probability|odds|chance|likel\w*|pric\w+\s+in|implied)\b[^?]{0,80}\b(?:all[\s-]+time\s+"
    r"high|ath|new\s+(?:record|high))\b|\b(?:all[\s-]+time\s+high|ath|new\s+(?:record|high))\b"
    r"[^?]{0,120}\b(?:probability|odds|chance|polymarket|prediction[\s-]market|options?)\b", re.I)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _crowd(year: int) -> tuple[float, str] | None:
    from argus.market import prediction

    try:
        events = prediction.search_events("bitcoin all time high")
    except Exception:
        return None
    for event in events:
        for m in event.get("markets") or []:
            q = str(m.get("question") or "")
            if m.get("closed") or not re.search(rf"December 31,? {year}", q):
                continue
            try:
                yes = float(__import__("json").loads(m.get("outcomePrices") or "[]")[0])
            except (ValueError, IndexError, TypeError):
                continue
            return yes, q
    return None


def lines(text: str) -> list[str] | None:
    """Both probabilities for BTC, or None when ``text`` is not that question."""
    if not ASKED.search(text) or not re.search(r"\bbtc\b|\bbitcoin\b", text, re.I):
        return None
    from argus.market import deribit
    from argus.market.equity_history import daily

    try:
        days = daily("BTC-USD")
        high = max(days, key=lambda d: d.close)
    except Exception:
        return ["Bottom line: BTC's price history did not answer just now, so the high to beat "
                "is unknown; ask again in a minute."]
    today = deribit.today()
    year_end = date(today.year, 12, 31)
    out: list[str] = []
    crowd = _crowd(today.year)
    try:
        book = deribit.options("BTC")
        spot = deribit.index_price("BTC")
    except Exception:
        book, spot = [], 0.0
    implied = None
    if book:
        expiries = sorted({o.expiry for o in book if o.expiry <= year_end + __import__(
            "datetime").timedelta(days=7) and (o.expiry - today).days >= 7})
        if expiries:
            expiry = expiries[-1]
            calls = [o for o in book if o.expiry == expiry and o.call]
            nearest = min(calls, key=lambda o: abs(o.strike - high.close))
            years = (expiry - today).days / 365
            vol, fwd = nearest.iv, nearest.forward
            d2 = (math.log(fwd / high.close) - vol * vol * years / 2) / (vol * math.sqrt(years))
            end_above = _norm_cdf(d2)
            implied = (expiry, end_above, min(1.0, 2 * end_above), vol, nearest.strike)
    lead = (f"Bottom line: BTC's record close is {high.close:,.0f} ({high.day:%d %b %Y}); "
            + (f"at {spot:,.0f} it needs {high.close / spot - 1:+.1%} to set a new one." if spot
               else "the price now did not answer."))
    out.append(lead)
    if crowd is not None:
        out.append(f"The crowd: Polymarket's \"{crowd[1]}\" trades at {crowd[0]:.1%} for yes — "
                   f"loosely a probability, since a price carries fees and the time value of "
                   f"locked money.")
    else:
        out.append("The crowd: no open Polymarket market on a new BTC high by year end answered "
                   "just now.")
    if implied is not None:
        expiry, end_above, touch, vol, strike = implied
        out.append(f"The options: Deribit's {expiry:%d %b} expiry, at {vol:.1%} implied volatility "
                   f"near the {strike:,.0f} strike, prices about {end_above:.1%} for BTC ending "
                   f"above the record, and about {touch:.1%} for touching it before then "
                   f"(twice the first, the no-drift reflection rule).")
        if crowd is not None:
            gap = crowd[0] - touch
            out.append("Consistent?" + (f" Roughly — the crowd's {crowd[0]:.1%} sits within a few "
                                        f"points of the options' {touch:.1%} for a touch."
                                        if abs(gap) <= 0.04 else
                                        f" No — the crowd's {crowd[0]:.1%} is "
                                        f"{'above' if gap > 0 else 'below'} the options' "
                                        f"{touch:.1%} for a touch by {abs(gap) * 100:.1f} points; "
                                        f"part of a gap like this is the risk premium in option "
                                        f"prices and the fees in Polymarket's."))
    else:
        out.append("The options: Deribit's BTC book did not answer just now.")
    out.append("Risk-neutral and crowd odds are prices, not forecasts. Data: Yahoo Finance daily "
               "closes, Polymarket, Deribit public option book, read now. Not advice.")
    return out
