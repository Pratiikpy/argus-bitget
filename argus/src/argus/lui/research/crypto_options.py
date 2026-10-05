"""BTC and ETH options from Deribit's public book: at-the-money implied volatility against the
VIX, the 25-delta skew, and the put strike that pays a stated annualised premium.

Three questions in round 40 (judge, Q4-Q6) had no source: "what is the 25-delta skew on BTC
options this month", "compare the VIX to BTC's at-the-money implied volatility" and "which strike
on a one-week BTC cash-secured put pays about 20% annualised, and where is breakeven". Each is
arithmetic on Deribit's listed book (`market/deribit.py`):

- **At the money** is the strike nearest the expiry's own forward; its mark IV is the answer, and
  the expiry is the one nearest the horizon asked (30 days by default). DVOL, Deribit's 30-day
  index, is given beside it.
- **Delta** is Black-76 on the expiry's forward with no rate — Deribit quotes crypto options on
  the forward, so no carry term is needed. The 25-delta put and call are the listed options whose
  delta is nearest -0.25 and +0.25; skew is the put's IV less the call's. Positive means puts cost
  more: demand for downside protection.
- **Annualised premium** on a cash-secured put is premium over strike, times 365 over the days to
  expiry. Breakeven if assigned is the strike less the premium.
- The VIX is the S&P 500's 30-day implied volatility (Yahoo's ^VIX). The two are compared like for
  like as 30-day annualised implied vol; the ratio is said, not a judgement of value.
"""

from __future__ import annotations

import math
import re
from typing import Any, Final

ASKED: Final = re.compile(
    r"\b(?:btc|bitcoin|eth|ether|ethereum|crypto)\b[^?]{0,80}\b(?:options?|implied\s+vol\w*|iv|"
    r"skew|25[\s-]?delta|strikes?|puts?|calls?|dvol)\b|"
    r"\b(?:options?|implied\s+vol\w*|skew|25[\s-]?delta|strikes?|puts?|calls?|dvol)\b[^?]{0,80}"
    r"\b(?:btc|bitcoin|eth|ether|ethereum)\b", re.I)
_SKEW: Final = re.compile(r"\bskew\b|\b25[\s-]?delta\b|\brisk[\s-]?reversal\b", re.I)
_PREMIUM: Final = re.compile(r"(?P<pct>\d{1,3}(?:\.\d+)?)\s*%\s*(?:annuali[sz]ed|a\s+year|apr|"
                             r"apy|per\s+year)[^?]{0,20}\b(?:premium|yield|return)?|"
                             r"\b(?:premium|yield)\s+of\s+(?:about\s+|roughly\s+)?"
                             r"(?P<pct2>\d{1,3}(?:\.\d+)?)\s*%", re.I)
_VIX: Final = re.compile(r"\bvix\b|\bequit(?:y|ies)\b|\bs&p\b|\bstocks?\b", re.I)
_DAYS: Final = re.compile(r"\b(?P<n>\d{1,3}|one|two|three|four)\s*[-\s]?(?P<unit>days?|weeks?|"
                          r"months?)\s*(?:out|away|to\s+expiry|expiry|expiration)?\b", re.I)
_WORDS: Final = {"one": 1, "two": 2, "three": 3, "four": 4}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def delta(forward: float, strike: float, years: float, vol: float, call: bool) -> float:
    """Black-76 delta on the forward, no rate."""
    if years <= 0 or vol <= 0:
        return (1.0 if forward > strike else 0.0) if call else (-1.0 if forward < strike else 0.0)
    d1 = (math.log(forward / strike) + vol * vol * years / 2) / (vol * math.sqrt(years))
    return _norm_cdf(d1) if call else _norm_cdf(d1) - 1


def _horizon_days(text: str, default: int) -> int:
    m = _DAYS.search(text)
    if m is None:
        return 30 if re.search(r"\bthis\s+month\b", text, re.I) else default
    n = int(m.group("n")) if m.group("n").isdigit() else _WORDS[m.group("n").lower()]
    unit = m.group("unit").lower()
    return n * (7 if unit.startswith("week") else 30 if unit.startswith("month") else 1)


def _expiry_near(rows: list[Any], today: Any, days: int) -> Any:
    expiries = sorted({o.expiry for o in rows if (o.expiry - today).days >= 1})
    return min(expiries, key=lambda e: abs((e - today).days - days))


def _vix() -> float | None:
    from argus.truth import http

    try:
        body = http.fetch_json("https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX",
                               params={"interval": "1d", "range": "5d"},
                               headers={"User-Agent": "Mozilla/5.0 argus-research"}, timeout=15)
        return float(body["chart"]["result"][0]["meta"]["regularMarketPrice"])
    except Exception:
        return None


def lines(text: str) -> list[str] | None:
    """The options answer ``text`` asks for, or None when it is not about BTC or ETH options."""
    if not ASKED.search(text):
        return None
    from argus.market import deribit

    currency = "ETH" if re.search(r"\beth\b|\bether(?:eum)?\b", text, re.I) and not re.search(
        r"\bbtc\b|\bbitcoin\b", text, re.I) else "BTC"
    premium = _PREMIUM.search(text)
    wants_put = bool(re.search(r"\bputs?\b", text, re.I))
    if not (_SKEW.search(text) or premium or re.search(
            r"\bimplied\s+vol\w*|\biv\b|\bdvol\b|\boptions\b|\bstrikes?\b", text, re.I)):
        return None
    try:
        book = deribit.options(currency)
        spot = deribit.index_price(currency)
    except Exception:
        return [f"Bottom line: Deribit's {currency} option book did not answer just now, so no "
                f"implied volatility is given; ask again in a minute."]
    today = deribit.today()
    if premium and wants_put:
        return _put_for_premium(text, book, spot, today, currency,
                                float(premium.group("pct") or premium.group("pct2")))
    if _SKEW.search(text):
        return _skew(text, book, today, currency)
    return _atm(text, book, spot, today, currency, deribit.dvol(currency))


def _atm(text: str, book: list[Any], spot: float, today: Any, currency: str,
         dvol: float | None) -> list[str]:
    days = _horizon_days(text, 30)
    expiry = _expiry_near(book, today, days)
    leg = [o for o in book if o.expiry == expiry]
    forward = leg[0].forward
    atm = min(leg, key=lambda o: (abs(o.strike - forward), not o.call))
    out = [f"Bottom line: {currency}'s at-the-money implied volatility is {atm.iv:.1%} a year for "
           f"the {expiry:%d %b} expiry ({(expiry - today).days} days, strike {atm.strike:,.0f} "
           f"against a forward of {forward:,.0f}), on Deribit's book."]
    if dvol is not None:
        out.append(f"DVOL, Deribit's 30-day {currency} implied-volatility index, reads {dvol:.1f}.")
    if _VIX.search(text):
        vix = _vix()
        if vix is not None:
            out.append(f"Like for like, both as 30-day annualised implied volatility: the VIX "
                       f"(S&P 500) is {vix:.1f} and {currency} is {atm.iv * 100:.1f} — "
                       f"{atm.iv * 100 / vix:.1f} times the stock market's. Whether that is cheap "
                       f"is a judgement against {currency}'s own history, not this one ratio.")
    out.append(f"Data: Deribit public option book and DVOL, read now ({currency} index "
               f"{spot:,.0f})" + ("; VIX from Yahoo Finance" if _VIX.search(text) else "")
               + ". Analysis, not advice.")
    return out


def _skew(text: str, book: list[Any], today: Any, currency: str) -> list[str]:
    days = _horizon_days(text, 30)
    expiry = _expiry_near(book, today, days)
    years = (expiry - today).days / 365
    leg = [o for o in book if o.expiry == expiry]
    put = min((o for o in leg if not o.call),
              key=lambda o: abs(delta(o.forward, o.strike, years, o.iv, False) + 0.25))
    call = min((o for o in leg if o.call),
               key=lambda o: abs(delta(o.forward, o.strike, years, o.iv, True) - 0.25))
    skew = (put.iv - call.iv) * 100
    read = ("puts cost more than calls: buyers are paying up for downside protection"
            if skew > 0.5 else "calls cost more than puts: the demand is for upside"
            if skew < -0.5 else "puts and calls are priced about evenly: no strong lean")
    return [f"Bottom line: the 25-delta skew on {currency} options expiring {expiry:%d %b} "
            f"({(expiry - today).days} days) is {skew:+.1f} vol points — {read}.",
            f"25-delta put: strike {put.strike:,.0f} at {put.iv:.1%} implied volatility; "
            f"25-delta call: strike {call.strike:,.0f} at {call.iv:.1%}; "
            f"forward {leg[0].forward:,.0f}.",
            "Deltas are Black-76 on each expiry's forward from Deribit's mark implied "
            "volatility; the listed strike nearest each delta is used, not an interpolated one.",
            "Data: Deribit public option book, read now. Analysis, not advice."]


def _put_for_premium(text: str, book: list[Any], spot: float, today: Any, currency: str,
                     target: float) -> list[str]:
    days = _horizon_days(text, 7)
    expiry = _expiry_near(book, today, days)
    held = (expiry - today).days
    puts = [o for o in book if o.expiry == expiry and not o.call and o.strike < o.forward * 1.02]
    if not puts:
        return [f"Bottom line: Deribit lists no {currency} puts for the {expiry:%d %b} expiry "
                f"near the money, so no strike can be given."]

    def annualised(o: Any) -> float:
        return float(o.mark_btc * spot / o.strike * 365 / max(held, 1) * 100)

    best = min(puts, key=lambda o: abs(annualised(o) - target))
    paid = best.mark_btc * spot
    return [f"Bottom line: the {best.strike:,.0f} put expiring {expiry:%d %b} ({held} days) pays "
            f"about {annualised(best):.1f}% annualised — ${paid:,.0f} of premium per {currency} of "
            f"cash secured at {best.strike:,.0f} — the listed strike closest to the {target:g}% "
            f"asked" + (f", on the listed expiry nearest the {days} days asked."
                         if held != days else "."),
            f"Breakeven if assigned: {best.strike - paid:,.0f} "
            f"({(best.strike - paid) / spot - 1:+.1%} "
            f"from the {currency} index at {spot:,.0f}); below that the position loses dollar for "
            f"dollar, like owning {currency} bought there.",
            f"Its implied volatility is {best.iv:.1%}; the premium is Deribit's mark, and a fill "
            f"would sit between the bid and the ask.",
            "Data: Deribit public option book, read now. Annualised = premium / strike x 365 / "
            "days. Analysis, not advice."]
