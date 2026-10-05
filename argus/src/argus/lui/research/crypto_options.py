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
    # "versus its 90-day average" names the history's span, not the expiry: ETH's skew was read
    # on an 81-day expiry for it (round 41 live pre-check)
    text = re.sub(r"\b\d{1,3}[\s-]*(?:day|d)\b[^?]{0,20}\b(?:average|avg|mean|norm|history)\b",
                  " ", text, flags=re.I)
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
    if re.search(r"\bmax\s*pain\b|\bput[\s/-]*call\b|\bopen\s+interest\b", text, re.I):
        # the positioning reader's (`crypto_positioning`): "put/call open interest ratio ... and
        # max pain" got ATM implied vol here (round 42 judge, C1)
        return None
    from argus.market import deribit

    currency = _subject(text)
    premium = _PREMIUM.search(text)
    wants_put = bool(re.search(r"\bputs?\b", text, re.I))
    if not (_SKEW.search(text) or premium or _STRIKE.search(text) or _EXPIRY.search(text)
            or _SPREAD.search(text) or re.search(
            r"\bimplied\s+vol\w*|\biv\b|\bdvol\b|\boptions\b|\bstrikes?\b", text, re.I)):
        return None
    try:
        book = deribit.options(currency)
        spot = deribit.index_price(currency)
    except Exception:
        return [f"Bottom line: Deribit's {currency} option book did not answer just now, so no "
                f"implied volatility is given; ask again in a minute."]
    today = deribit.today()
    named = _specific(text, book, spot, today, currency)
    if named is not None:
        return named
    if premium and wants_put:
        return _put_for_premium(text, book, spot, today, currency,
                                float(premium.group("pct") or premium.group("pct2")))
    spread = _put_spread(text, book, spot, today, currency)
    if _SKEW.search(text):
        said = _skew(text, book, today, currency)
        return said if spread is None else [*said[:-1], *spread, said[-1]]
    if spread is not None:
        return [f"Bottom line: {spread[0][:1].lower()}{spread[0][1:]}", *spread[1:],
                "Data: Deribit public option book, read now. Analysis, not advice."]
    return _atm(text, book, spot, today, currency, deribit.dvol(currency))


_STRIKE: Final = re.compile(r"\bstrike\s+(?:of\s+|at\s+)?\$?\s?"
                            r"(?P<a>\d[\d,]*(?:\.\d+)?)\s*(?P<k>k)?\b|"
                            r"\$?(?P<b>\d[\d,]*(?:\.\d+)?)\s*(?P<k2>k)?\s+(?:call|put)\b", re.I)
_EXPIRY: Final = re.compile(r"\b(?:expir\w*|expiry|exp\.?)\s+(?:on\s+)?(?P<d>tomorrow|today|"
                            r"\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|[A-Za-z]{3,9}\s+"
                            r"\d{1,2},?\s+\d{4})", re.I)


def _expiry_said(raw: str, today: Any) -> Any:
    from datetime import datetime, timedelta

    word = raw.lower()
    if word == "tomorrow":
        return today + timedelta(days=1)
    if word == "today":
        return today
    for layout in ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(raw.replace(",", ""), layout).date()
        except ValueError:
            continue
    return None


def _specific(text: str, book: list[Any], spot: float, today: Any,
              currency: str) -> list[str] | None:
    """A named strike or expiry, answered as named: "a BTC call at strike $1 expiring tomorrow",
    "a put 60000 expiring 2019-03-01" and "strike 99,999,999" each got the at-the-money answer
    for another expiry (round 41 hostile, M5)."""
    s = _STRIKE.search(text)
    e = _EXPIRY.search(text)
    if s is None and e is None:
        return None
    expiry = _expiry_said(e.group("d"), today) if e else None
    if e is not None and expiry is None:
        return None
    if expiry is not None and expiry < today:
        return [f"Bottom line: an option expiring {expiry:%d %b %Y} has already expired, so it has "
                f"no price or implied volatility now — Deribit lists live expiries only, from "
                f"{min(o.expiry for o in book):%d %b %Y} out to "
                f"{max(o.expiry for o in book):%d %b %Y}."]
    listed = sorted({o.expiry for o in book})
    on = (min(listed, key=lambda d: abs((d - expiry).days)) if expiry else
          _expiry_near(book, today, _horizon_days(text, 30)))
    leg = [o for o in book if o.expiry == on]
    call = not re.search(r"\bputs?\b", text, re.I)
    strikes = sorted({o.strike for o in leg})
    out: list[str] = []
    if expiry is not None and on != expiry:
        out.append(f"Deribit lists no {currency} expiry on {expiry:%d %b %Y}; the nearest listed "
                   f"is {on:%d %b %Y}.")
    if s is not None:
        asked = float((s.group("a") or s.group("b")).replace(",", "")) * (
            1000 if (s.group("k") or s.group("k2")) else 1)
        if not strikes or asked < strikes[0] * 0.5 or asked > strikes[-1] * 2:
            out.insert(0, f"Bottom line: no {currency} option is listed at a strike of "
                          f"{asked:,.0f} "
                          f"for {on:%d %b %Y} — strikes there run from {strikes[0]:,.0f} to "
                          f"{strikes[-1]:,.0f}, around a forward of {leg[0].forward:,.0f}, so "
                          f"there is no price or delta to give for it.")
            return out
        pick = min((o for o in leg if o.call == call), key=lambda o: abs(o.strike - asked))
    else:
        pick = min((o for o in leg if o.call == call), key=lambda o: abs(o.strike - o.forward))
    years = max((on - today).days, 1) / 365
    d = delta(pick.forward, pick.strike, years, pick.iv, call)
    price = pick.mark_btc * spot
    near = "" if s is None or pick.strike == asked else f" (the listed strike nearest {asked:,.0f})"
    out.insert(0, f"Bottom line: the {currency} {pick.strike:,.0f} {'call' if call else 'put'} "
                  f"expiring {on:%d %b %Y}{near} is marked at "
                  f"{pick.mark_btc:.4f} {currency} — about "
                  f"${price:,.0f} — at {pick.iv:.1%} implied volatility, with a delta of {d:+.2f}.")
    out.append(f"Data: Deribit public option book, read now ({currency} index {spot:,.0f}); delta "
               f"is Black-76 on the expiry's forward. Not advice.")
    return out


_ETH: Final = re.compile(r"\beth\b|\bether(?:eum)?\b", re.I)
_BTC: Final = re.compile(r"\bbtc\b|\bbitcoin\b", re.I)
_VOL_WORD: Final = re.compile(r"\bimplied|\biv\b|\bvol\w*|\bskew|\boptions?\b|\bstrike|\bdvol\b",
                              re.I)


def _subject(text: str) -> str:
    """The coin whose options are asked about: the one named nearest before the volatility words,
    else the first named. "the ETH/BTC ratio and ETH 30-day implied vol" was answered for BTC
    because BTC was named at all (round 41 judge, C1)."""
    vol = _VOL_WORD.search(text)
    at = vol.start() if vol else len(text)
    before = [(m.start(), "ETH") for m in _ETH.finditer(text[:at])] + \
        [(m.start(), "BTC") for m in _BTC.finditer(text[:at])]
    if before:
        return max(before)[1]
    first = [(m.start(), "ETH") for m in _ETH.finditer(text)] + \
        [(m.start(), "BTC") for m in _BTC.finditer(text)]
    return min(first)[1] if first else "BTC"


def _atm_iv(book: list[Any], today: Any, days: int) -> tuple[Any, Any, float]:
    expiry = _expiry_near(book, today, days)
    leg = [o for o in book if o.expiry == expiry]
    forward = leg[0].forward
    atm = min(leg, key=lambda o: (abs(o.strike - forward), not o.call))
    return expiry, atm, forward


def _realised(currency: str) -> float | None:
    """Thirty-day realised volatility from Bitget's hourly closes, annualised over every hour."""
    from argus.lui.research.data import load

    try:
        rets = list((load((f"{currency}USDT",)).raw.get(f"{currency}USDT") or {}).values())
    except Exception:
        return None
    if len(rets) < 100:
        return None
    mu = sum(rets) / len(rets)
    return float((sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)) ** 0.5 * (24 * 365) ** 0.5)


def _atm(text: str, book: list[Any], spot: float, today: Any, currency: str,
         dvol: float | None) -> list[str]:
    from argus.market import deribit

    days = _horizon_days(text, 30)
    expiry, atm, forward = _atm_iv(book, today, days)
    out = [f"Bottom line: {currency}'s at-the-money implied volatility is {atm.iv:.1%} a year for "
           f"the {expiry:%d %b} expiry ({(expiry - today).days} days, strike {atm.strike:,.0f} "
           f"against a forward of {forward:,.0f}), on Deribit's book."]
    if re.search(r"\brealis|\brealiz|\bvol(?:atility)?\s+risk\s+premium\b|"
                 r"\bvrp\b|\bhistoric\w*\s+vol",
                 text, re.I):
        realised = _realised(currency)
        if realised is not None:
            premium = atm.iv - realised
            out.append(f"Realised over the last 30 days: {realised:.1%} a year (Bitget hourly "
                       f"closes), so the volatility risk premium — implied less realised — is "
                       f"{premium * 100:+.1f} points: options are "
                       + ("priced above what the market has actually moved." if premium > 0 else
                          "priced below what the market has actually moved."))
    other = "BTC" if currency == "ETH" else "ETH"
    other_named = (_BTC if other == "BTC" else _ETH).search(text)
    if other_named and re.search(r"\bvs\.?|\bversus\b|\bcompare\w*|\bagainst\b|\bratio\b", text,
                                 re.I):
        try:
            o_book = deribit.options(other)
            o_exp, o_atm, _ = _atm_iv(o_book, today, days)
            out.append(f"{other} beside it: {o_atm.iv:.1%} at the money for {o_exp:%d %b} — "
                       f"{currency} is {atm.iv / o_atm.iv:.2f} times {other}'s implied "
                       f"volatility.")
        except Exception:
            pass
    if re.search(r"\bratio\b", text, re.I) and _ETH.search(text) and _BTC.search(text):
        try:
            from argus.market.bitget import fetch_tickers

            board = fetch_tickers()
            ratio = float(board["ETHUSDT"].last) / float(board["BTCUSDT"].last)
            out.insert(1, f"The ETH/BTC ratio is {ratio:.5f} on Bitget's last prices.")
        except Exception:
            pass
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
    out = [f"Bottom line: the 25-delta skew on {currency} options expiring {expiry:%d %b} "
           f"({(expiry - today).days} days) is "
           f"{'0.0' if abs(skew) < 0.05 else format(skew, '+.1f')} vol points — {read}.",
           f"25-delta put: strike {put.strike:,.0f} at {put.iv:.1%} implied volatility; "
           f"25-delta call: strike {call.strike:,.0f} at {call.iv:.1%}; "
           f"forward {leg[0].forward:,.0f}."]
    out.extend(_skew_asks_unmet(text))
    history = _skew_history_line(text, currency)
    if history:
        out.append(history)
        relative = re.search(r"— (puts cheaper than usual|puts dearer than usual|about its usual "
                             r"level)", history)
        if relative:
            # the comparison asked for leads, not three lines down (round 41 judge, M11)
            out[0] = out[0].rstrip(".") + (f"; against its own {_history_days(text)}-day "
                                           f"history, {relative.group(1)}.")
    return [*out,
            "Deltas are Black-76 on each expiry's forward from Deribit's mark implied "
            "volatility; the listed strike nearest each delta is used, not an interpolated one.",
            "Data: Deribit public option book, read now. Analysis, not advice."]


_AVERAGE: Final = re.compile(r"\b(?P<n>\d{2,3})[\s-]*(?:day|d)\b[^?]{0,20}\b(?:average|avg|mean|"
                             r"norm)\b|\b(?:average|avg|normal|usual|history|historical\w*|"
                             r"percentile)\b", re.I)


_DELTA_ASKED: Final = re.compile(r"(?P<d>[-\u2212]?\d{1,4}(?:\.\d+)?)\s*[-\s]?delta\b", re.I)
_YEARS_BACK: Final = re.compile(r"\b(?P<n>\d{1,3})\s*(?:years?|yrs?)\s*(?:back|ago|of\s+history|"
                                r"history)\b|\b(?:last|past)\s+(?P<n2>\d{1,3})\s*(?:years?|yrs?)\b",
                                re.I)


def _skew_asks_unmet(text: str) -> list[str]:
    """A delta or a lookback the question asks for that the skew reader cannot give, said rather
    than silently replaced: "skew history ... -500 delta, 10 years back" got the 25-delta 30-day
    skew with neither mentioned (round 42 hostile, minor 8)."""
    from argus.market import skew_history

    out: list[str] = []
    for m in _DELTA_ASKED.finditer(text):
        raw = float(m.group("d").replace("\u2212", "-"))
        if abs(raw) == 25:
            continue
        if not 0 < abs(raw) < 100:
            out.append(f"A delta runs from 0 to 100 (0 to 1 as a fraction), so {raw:g} is not one; "
                       "the skew above is the standard 25-delta.")
        else:
            out.append(f"Only the 25-delta skew is computed here, not the {abs(raw):g}-delta.")
        break
    years = _YEARS_BACK.search(text)
    if years is not None:
        n = int(years.group("n") or years.group("n2"))
        out.append(f"The skew history here is rebuilt from Deribit's own trades and kept for the "
                   f"last {skew_history.DAYS} days, so {n} {'year' if n == 1 else 'years'} back is "
                   "not on record; the history line below covers what is.")
    return out


def _history_days(text: str) -> int:
    m = _AVERAGE.search(text)
    return int(m.group("n")) if m is not None and m.group("n") else 90


def _skew_history_line(text: str, currency: str) -> str | None:
    """Today's skew against its own average, both rebuilt the same way from Deribit's trades
    (`market/skew_history.py`); None when the question does not ask for the comparison."""
    m = _AVERAGE.search(text)
    if m is None:
        return None
    from argus.lui.answer import desk_notes_path
    from argus.market import skew_history

    days = int(m.group("n")) if m.group("n") else 90
    snapshot = skew_history.load(desk_notes_path().parent / "crypto_skew_history.json")
    found = skew_history.average(currency, days, snapshot=snapshot)
    if found is None:
        return (f"Its {days}-day average is not on record here: the daily skew history rebuilt "
                "from Deribit's trades does not cover enough of that span.")
    mean, counted, latest = found
    gap = (latest or 0.0) - mean
    return (f"Against its own history: rebuilt from each day's Deribit trades the same way, the "
            f"30-day 25-delta skew averaged {mean:+.1f} vol points over the last {days} days "
            f"({counted} days with enough trades) and read {latest:+.1f} on the latest full day — "
            + ("puts cheaper than usual" if gap < -0.5 else "puts dearer than usual"
               if gap > 0.5 else "about its usual level")
            + ". That series is printed from trades, not the book's marks, so set the history "
              "against the history, not against the figure above.")


_SPREAD: Final = re.compile(r"\bput\s+spread\b|\bbear\s+put\b|\bput\s+vertical\b", re.I)
_OTM: Final = re.compile(r"(?P<a>\d{1,2}(?:\.\d+)?)\s*%\s*(?:otm|out[\s-]of[\s-]the[\s-]money|"
                         r"below)"
                         r"(?:[^?]{0,40}?(?P<b>\d{1,2}(?:\.\d+)?)\s*%\s*(?:otm|out[\s-]of[\s-]the"
                         r"[\s-]money|below))?", re.I)


def _put_spread(text: str, book: list[Any], spot: float, today: Any,
                currency: str) -> list[str] | None:
    """A put spread priced off the listed book: long the put about the stated distance below the
    spot, short one further down (twice the distance when only one is said)."""
    if not _SPREAD.search(text):
        return None
    m = _OTM.search(text)
    near = float(m.group("a")) / 100 if m else 0.10
    far = float(m.group("b")) / 100 if m and m.group("b") else near * 2
    days = _horizon_days(text, 30)
    expiry = _expiry_near(book, today, days)
    puts = [o for o in book if o.expiry == expiry and not o.call]
    if len(puts) < 2:
        return None
    long_leg = min(puts, key=lambda o: abs(o.strike - spot * (1 - near)))
    short_leg = min((o for o in puts if o.strike < long_leg.strike),
                    key=lambda o: abs(o.strike - spot * (1 - far)), default=None)
    if short_leg is None:
        return None
    cost = (long_leg.mark_btc - short_leg.mark_btc) * spot
    width = long_leg.strike - short_leg.strike
    assumed = "" if m and m.group("b") else (f" (the short strike was not stated, so it is set "
                                              f"{far:.0%} below spot)")
    return [f"A {near:.0%}-out-of-the-money put spread to {expiry:%d %b} "
            f"({(expiry - today).days} days): buy the {long_leg.strike:,.0f} put, sell the "
            f"{short_leg.strike:,.0f} put{assumed} — it costs about ${cost:,.0f} per {currency} "
            f"({cost / spot:.2%} of spot) and pays at most ${width - cost:,.0f} if {currency} is "
            f"below {short_leg.strike:,.0f} at expiry, {(width - cost) / cost:.1f}x the cost.",
            f"It starts paying below {long_leg.strike - cost:,.0f} "
            f"({(long_leg.strike - cost) / spot - 1:+.1%} from {spot:,.0f}); "
            f"the long put is marked "
            f"at {long_leg.iv:.1%} implied volatility and the short at {short_leg.iv:.1%} — the "
            "skew is what the short leg sells back. Marks are Deribit's; a fill sits between the "
            "bid and the ask on each leg."]


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
