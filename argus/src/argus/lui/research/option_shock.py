"""What a jump or a crush in implied volatility does to an option position, by Black-Scholes.

"I hold a $50,000 covered call position on AAPL — if implied vol spikes 10 points before expiry,
what happens to my position?" got the chain's current reading and nothing about the spike (round
39 judge, C-8). The answer is arithmetic on the position, so it is worked here.

**How.** The position is read as written: a covered call is the shares plus one short call per
100 of them; a long or short call or put is that leg alone. The shares are the stated dollars over
the live price. The option is taken at the money and about 30 days out unless the question says
otherwise, priced with Black-Scholes (no dividend, a 4% rate — said in the answer) at the chain's
own 30-day implied volatility from Cboe's delayed quotes (`market/options.fetch_chain`), and again
at that volatility plus the stated change. The shares do not move on a volatility change alone;
the option's value does, by about its vega times the change, and the answer gives both the exact
repricing and that vega.
"""

from __future__ import annotations

import math
import re
from typing import Final

ASKED: Final = re.compile(
    r"\b(?P<pos>covered\s+calls?|short\s+(?:calls?|puts?)|long\s+(?:calls?|puts?)|"
    r"(?:calls?|puts?)\s+i\s+(?:sold|wrote|bought)|straddles?)\b[^?]{0,120}?"
    r"\b(?:implied\s+vol(?:atility)?|iv|vol(?:atility)?)\b[^?]{0,30}?\b(?P<dir>spikes?|jumps?|"
    r"rises?|goes\s+up|climbs?|drops?|falls?|goes\s+down|crush(?:es)?|collapses?)\b"
    r"(?:\s+by|\s+of)?\s*(?P<pts>\d{1,2}(?:\.\d+)?)\s*(?:vol\s+)?(?:points?|pts|%)", re.I)
_CRUSH_FIRST: Final = re.compile(r"\b(?:iv|vol(?:atility)?)\s+(?P<dir>crush|spike|jump|drop)\s+"
                                 r"(?:of\s+)?(?P<pts>\d{1,2}(?:\.\d+)?)", re.I)
""""IV crush of 15 points": the change named before its size."""
RATE: Final = 0.04
DAYS: Final = 30


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_scholes(spot: float, strike: float, years: float, vol: float, call: bool,
                  rate: float = RATE) -> float:
    """The Black-Scholes price of a European option, no dividend."""
    if years <= 0 or vol <= 0:
        return max(0.0, (spot - strike) if call else (strike - spot))
    d1 = (math.log(spot / strike) + (rate + vol * vol / 2) * years) / (vol * math.sqrt(years))
    d2 = d1 - vol * math.sqrt(years)
    if call:
        return spot * _norm_cdf(d1) - strike * math.exp(-rate * years) * _norm_cdf(d2)
    return strike * math.exp(-rate * years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def lines(text: str) -> list[str] | None:
    """The repricing for the position and the volatility change ``text`` states, or None."""
    found = ASKED.search(text)
    position = re.search(r"\b(?:covered\s+calls?|short\s+(?:calls?|puts?)|long\s+(?:calls?|puts?)|"
                         r"straddles?)\b", text, re.I)
    crush = _CRUSH_FIRST.search(text)
    if found is not None:
        pos_text, dir_text, pts_text = found.group("pos"), found.group("dir"), found.group("pts")
    elif position is not None and crush is not None:
        pos_text, dir_text, pts_text = position.group(0), crush.group("dir"), crush.group("pts")
    else:
        return None
    from argus.lui.research import research_symbols
    from argus.lui.research.parse import last_price, parse_notional
    from argus.market.options import fetch_chain

    named = research_symbols(text)[0]
    if not named:
        return None
    ticker = named[0].removesuffix("USDT")
    try:
        spot = float(last_price(named[0]) or 0)
        data = fetch_chain(ticker).get("data") or {}
    except Exception:
        return None
    iv30 = data.get("iv30")
    if not spot or not iv30:
        return [f"Bottom line: {ticker} has no listed option chain this console reads (Cboe), so "
                f"the volatility change cannot be priced on it."]
    vol = float(iv30) / 100
    change = float(pts_text) / 100
    if re.match(r"drop|fall|goes\s+down|crush|collapse", dir_text, re.I):
        change = -change
    pos = pos_text.lower()
    call = "put" not in pos
    short = (pos.startswith(("covered", "short"))
             or re.search(r"\b(?:sold|wrote)\b", pos) is not None)
    years = DAYS / 365
    before = black_scholes(spot, spot, years, vol, call)
    after = black_scholes(spot, spot, years, max(vol + change, 0.01), call)
    notional = parse_notional(text)
    shares = float(notional) / spot if notional else 100.0
    contracts = shares / 100
    per_share = after - before
    option_pnl = (-1 if short else 1) * per_share * shares
    d1 = (math.log(1) + (RATE + vol * vol / 2) * years) / (vol * math.sqrt(years))
    vega = spot * math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi) * math.sqrt(years) / 100
    leg = f"{'short' if short else 'long'} {'call' if call else 'put'}"
    sized = (f"${float(notional):,.0f} of {ticker} is {shares:,.0f} shares, {contracts:,.1f} "
             f"contracts of the {leg}" if notional else f"one contract (100 shares) of the {leg}")
    move = f"{'+' if change > 0 else ''}{change * 100:g} points"
    lead = (f"Bottom line: an implied-vol move of {move} {'costs' if option_pnl < 0 else 'makes'} "
            f"the position about ${abs(option_pnl):,.0f} on the option leg — its value goes from "
            f"${before:,.2f} to ${after:,.2f} a share ({sized})")
    out = [lead + ("; the shares themselves do not move on volatility alone." if
                   pos.startswith("covered") else ".")]
    out.append(f"Why: vega — about ${vega:,.2f} a share per volatility point at {ticker} "
               f"{spot:,.2f}, at the money, {DAYS} days out, from {vol:.0%} implied vol (Cboe's "
               f"30-day, delayed). A {'short' if short else 'long'} option "
               f"{'loses' if short == (change > 0) else 'gains'} when volatility "
               f"{'rises' if change > 0 else 'falls'}.")
    if pos.startswith("covered"):
        out.append("Assignment does not change with volatility: it turns on whether the stock "
                   "ends above the strike. Higher volatility makes buying the call back to close "
                   "cost more, and it raises the premium a new call would bring in.")
    out.append(f"Assumed: an at-the-money option {DAYS} days out, Black-Scholes with no dividend "
               f"and a {RATE:.0%} rate; say the strike and expiry for your own contract.")
    return out
