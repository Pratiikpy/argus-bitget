"""Units, figures and premises a question states, worked as stated or corrected before anything
is answered.

Round 43's hostile audit found the console answering a nearby question instead of the one with the
trap in it. Each reader below takes one shape, works the arithmetic exactly as the words give it,
and says plainly where the premise is wrong:

* **A rate per period** ("0.01% per hour") annualised over that period, not re-read as the usual
  eight-hour funding interval (M1).
* **Cents and pence.** "8576510 cents" is $85,765.10 and "2,500p" is 25 pounds; each is compared
  with the live price in its own currency (M2, M3, M4).
* **False premises.** There was no Bitcoin halving in 2025 (the fourth was block 840,000 on 20 Apr
  2024; the next falls around 2028); a claimed ban on US spot bitcoin ETFs is checked against the
  flows SoSoValue still reports; "50 percentage points" is 5,000 basis points (M5, M6, M7).
* **Figures the trader gives.** A stated taker fee and a stated gas price are used as given, beside
  the live ones, never replaced by them (M8, M9).
* **Stops and targets on the wrong side.** A long's stop above entry, or a short's take-profit
  above entry, is named for what it is and its result worked (M10, M11, M12).
* **Fractions and basis points.** "0.4 ... 40%?" and "0.1 bps versus 0.1%" on a stated notional
  (M14, M15).
* **A currency asked for** ("24h volume in euros") is converted at Bitget's own FX perpetual
  (M16).

Live figures: Bitget's ticker (last, bid, ask, 24-hour change and volume) and Bitget's EURUSD and
GBPUSD perpetuals through `parse.in_us_dollars`. Nothing here is a forecast.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Final

_NUM: Final = r"(\d[\d,]*(?:\.\d+)?)"
_PER_PERIOD: Final = re.compile(
    rf"{_NUM}\s*(?P<unit>%|percent|bps?|basis\s+points?)\s*(?:per|every|an?|each|/)\s*"
    r"(?P<n>\d+)?\s*(?P<period>hours?|hrs?|h|days?|weeks?|months?)\b", re.I)
_ANNUAL_ASK: Final = re.compile(r"\bper\s+year\b|\ba\s+year\b|\bannual\w*|\byearly\b|\bapr\b|"
                                r"\bapy\b|\bper\s+annum\b", re.I)
_CENTS: Final = re.compile(rf"{_NUM}\s*(?:cents?|¢)\b", re.I)
# a figure, not a name: "P2P" is peer-to-peer, not two pence (round 43 stranger pre-check)
_PENCE: Final = re.compile(rf"(?<![A-Za-z\d.,]){_NUM}\s*(?:p\b|pence\b|gbp?x\b)", re.I)
_HALVING_YEAR: Final = re.compile(r"\b(?P<y>20\d\d)\s+(?:bitcoin\s+|btc\s+)?halving\b|"
                                  r"\bhalving\s+(?:of|in)\s+(?P<y2>20\d\d)\b", re.I)
_HALVINGS: Final = (date(2012, 11, 28), date(2016, 7, 9), date(2020, 5, 11), date(2024, 4, 20))
_ETF_BAN: Final = re.compile(r"\b(?:spot\s+)?(?:bitcoin|btc|ether(?:eum)?|eth)\s+etfs?\b[^?.]{0,40}"
                             r"\b(?:banned|ban|delisted|outlawed|shut\s+down)\b|\bban(?:ned)?\s+"
                             r"(?:on\s+)?(?:spot\s+)?(?:bitcoin|btc|crypto)\s+etfs?\b", re.I)
_PP_MOVE: Final = re.compile(rf"\b{_NUM}\s*(?:percentage\s+points?|pp|pts?)\b", re.I)
_FEE_STATED: Final = re.compile(rf"\b{_NUM}\s*(?P<unit>%|bps?|basis\s+points?)\s+(?:taker|maker|"
                                r"trading)?\s*fees?\b|\bfees?\s+(?:of\s+)?" + _NUM
                                + r"\s*(?P<unit2>%|bps?)", re.I)
_GAS: Final = re.compile(rf"\b{_NUM}\s*gwei\b[^?]{{0,60}}?\b{_NUM}\s*(?:k\s+)?(?:units\s+of\s+)?"
                         r"gas\b", re.I)
# a size ("long 1 BTC") is not an entry, and "a stop 5% below" is a distance, not a level
_ENTRY: Final = re.compile(rf"\b(?:entry|entered|bought|sold|short(?:ed)?|long)\s+(?:\w+\s+)?"
                           rf"(?:at|@|of)\s+\$?{_NUM}\b(?!\s*%)|\bentry\s+\$?{_NUM}\b(?!\s*%)",
                           re.I)
_STOP: Final = re.compile(rf"\bstop(?:[\s-]?loss)?\s+(?:at\s+|of\s+)?\$?{_NUM}\b(?![\d.]*\s*%)",
                          re.I)
_TARGET: Final = re.compile(rf"\b(?:take[\s-]?profit|target|tp)\s+(?:at\s+|of\s+)?\$?{_NUM}\b"
                            r"(?![\d.]*\s*%)", re.I)
_FRACTION_PCT: Final = re.compile(rf"\bis\s+(?P<f>0?\.\d+)\b[^?]{{0,60}}?\b{_NUM}\s*%", re.I)
_BPS_VS_PCT: Final = re.compile(rf"\b(?P<a>\d+(?:\.\d+)?)\s*bps?\b[^?]{{0,80}}?\b{_NUM}\s*%|"
                                rf"\b(?P<c>\d+(?:\.\d+)?)\s*%[^?]{{0,80}}?\b(?P<d>\d+(?:\.\d+)?)\s*"
                                r"bps?\b", re.I)
_NOTIONAL: Final = re.compile(rf"\b(?:on\s+)?(?:a\s+)?\$?{_NUM}\s*(?P<k>k)?\s*(?:usdt|usd|\$|"
                              r"dollars?)?\s+(?:trade|order|position|notional)\b", re.I)
_IN_EUROS: Final = re.compile(r"\bin\s+(?P<cur>euros?|eur|pounds?|gbp|sterling|yen|jpy)\b",
                              re.I)


def _n(raw: str) -> float:
    return float(raw.replace(",", ""))


def _ticker(symbol: str) -> Any:
    from argus.market.bitget import fetch_tickers

    return fetch_tickers().get(symbol)


def _symbol(text: str) -> str | None:
    from argus.lui.research.parse import research_symbols

    found = research_symbols(text)[0]
    return found[0] if found else None


def per_period(text: str) -> list[str] | None:
    """A rate per hour, day, week or month, annualised over the period stated."""
    m = _PER_PERIOD.search(text)
    if m is None or not _ANNUAL_ASK.search(text):
        return None
    value = _n(m.group(1))
    unit = m.group("unit").lower()
    rate = value / 100 if unit.startswith(("%", "percent")) else value / 10_000
    n = int(m.group("n") or 1)
    period = m.group("period").lower()
    hours = n * (1 if period.startswith("h") else 24 if period.startswith("d") else
                 168 if period.startswith("w") else 730)
    periods = 365 * 24 / hours
    shown = f"{value:g}{'%' if rate == value / 100 else ' bps'}"
    every = f"every {n} {period.rstrip('s')}s" if n > 1 else f"per {period.rstrip('s')}"
    out = [f"Bottom line: {shown} {every} is {rate * periods:.2%} a year simple "
           f"({periods:,.0f} periods), {(1 + rate) ** periods - 1:.2%} compounded, and "
           f"{rate * 24 / hours * 100:.3f}% a day."]
    if re.search(r"\bfunding\b", text, re.I) and hours != 8:
        out.append("Taken per the period you stated, not per Bitget's usual eight-hour funding "
                   f"interval; at {shown} per eight hours instead it would be "
                   f"{rate * 3 * 365:.2%} a year.")
    out.append("Funding is quoted simple and paid on the position's size, not the margin.")
    return out


def minor_units(text: str) -> list[str] | None:
    """A price written in cents or pence, restated in dollars or pounds and set against the live
    price; the value of a stated share count at the price as written."""
    from argus.lui.research.parse import in_us_dollars

    cents = _CENTS.search(text)
    pence = None if cents else _PENCE.search(text)
    m = cents or pence
    if m is None:
        return None
    symbol = _symbol(text[:m.start()] + " " + text[m.end():])
    major = _n(m.group(1)) / 100
    out: list[str] = []
    if cents:
        out.append(f"Bottom line: {m.group(1)} cents is ${major:,.2f}")
        usd = major
    else:
        restated, said = in_us_dollars(f"{major:.2f} GBP")
        rate_m = re.search(r"\$(\d[\d,]*(?:\.\d+)?)", restated)
        usd = _n(rate_m.group(1)) if rate_m and said else major
        out.append(f"Bottom line: {m.group(1)}p (pence) is £{major:,.2f}, not "
                   f"£{_n(m.group(1)):,.0f} — a hundred pence to the pound; about "
                   f"${usd:,.2f}")
        if re.search(rf"\b{re.escape(m.group(1))}\s+pounds\b", text, re.I):
            out[0] += f", so £{_n(m.group(1)):,.0f} a share is wrong by a factor of 100"
    shares = re.search(r"\b(\d[\d,]*)\s*(?:shares?|-share)\b", text, re.I)
    if shares:
        count = _n(shares.group(1))
        unit = "$" if cents else "£"
        out[0] += f"; {count:,.0f} shares at that price is {unit}{count * major:,.2f}"
    out[0] += "."
    if symbol is not None:
        try:
            ticker = _ticker(symbol)
        except Exception:
            ticker = None
        if ticker is not None:
            last = float(ticker.last)
            gap = usd / last - 1
            name = symbol.removesuffix("USDT")
            verdict = ("within 1% of" if abs(gap) < 0.01 else f"{gap:+.1%} from")
            out.append(f"{name} is {last:,.2f} on Bitget now (in dollars), so the figure as "
                       f"restated is {verdict} the live price"
                       + (" — Bitget's contract tracks the US-listed share, so a London price "
                          "in pounds differs from it by the exchange rate and the listing."
                          if pence else "."))
    out.append("Data: Bitget's ticker and its GBPUSD perpetual, read just now. Not advice.")
    return out


def false_premise(text: str) -> list[str] | None:
    """A halving, an ETF ban or a percentage-point move that did not happen as stated, said first;
    the question's remaining part is left to the readers after this one."""
    h = _HALVING_YEAR.search(text)
    if h is not None and re.search(r"\bbitcoin\b|\bbtc\b", text, re.I):
        year = int(h.group("y") or h.group("y2"))
        if all(d.year != year for d in _HALVINGS):
            before = [d for d in _HALVINGS if d.year < year]
            last = before[-1] if before else _HALVINGS[0]
            dates = ", ".join(f"{d:%d %b %Y}" for d in _HALVINGS)
            return [f"Premise check: there was no Bitcoin halving in {year}. Halvings happen every "
                    f"210,000 blocks, about four years apart: {dates}; "
                    f"the next is expected around 2028. The one before {year} was "
                    f"{last:%d %b %Y}.",
                    f"Ask \"how did BTC do in the year after the {last.year} halving\" for that "
                    "window, or \"BTC in " + str(year) + "\" for the calendar year."]
    if _ETF_BAN.search(text):
        from argus.market import etf_flows

        snap = etf_flows.load()
        latest = str(((snap or {}).get("funds") or {}).get("BTC", {}).get("date") or "")[:10]
        return ["Premise check: this console has no record of a ban on US spot bitcoin ETFs — "
                "they were approved by the SEC on 10 Jan 2024"
                + (f", and SoSoValue still reports their daily flows (latest {latest})"
                   if latest else "")
                + ". A claim of a ban needs a source; the price question is answered on its own "
                  "below."]
    pp = _PP_MOVE.search(text)
    if pp is not None and re.search(r"\byield|\brate\b|\btreasury|\b10[\s-]?year", text, re.I):
        points = _n(pp.group(1))
        out = [f"Premise check: {points:g} percentage points is {points * 100:,.0f} basis points, "
               f"not {points:g} — a basis point is a hundredth of a percentage point, so a "
               f"{points:g}bp move is {points / 100:g} percentage points."]
        level = re.search(r"\bto\s+(\d+(?:\.\d+)?)\s*%?", text[pp.end():], re.I)
        if level:
            from argus.lui.research.macro import _fred

            rows = _fred("DGS10", 20)
            if rows:
                out.append(f"And the 10-year is {rows[-1][1]:.2f}% ({rows[-1][0]}, FRED DGS10), "
                           f"not {level.group(1)}%.")
        return out
    return None


def stated_fee(text: str) -> list[str] | None:
    """A round trip at the fee the trader states, beside the live spread."""
    m = _FEE_STATED.search(text)
    if m is None or not re.search(r"\bbreak[\s-]?even\b|\bround[\s-]?trip\b|\bmove\b", text, re.I):
        return None
    raw = m.group(1) or m.group(3)
    unit = (m.group("unit") or m.group("unit2") or "%").lower()
    fee = _n(raw) / (100 if unit.startswith("%") else 10_000)
    symbol = _symbol(text) or "BTCUSDT"
    spread = None
    try:
        t = _ticker(symbol)
        if t is not None and float(t.bid) > 0:
            spread = (float(t.ask) - float(t.bid)) / ((float(t.ask) + float(t.bid)) / 2)
    except Exception:
        spread = None
    total = 2 * fee + (spread or 0.0)
    name = symbol.removesuffix("USDT")
    return [f"Bottom line: at a {fee:.6%} fee a side the round trip in {name} costs "
            f"{2 * fee:.6%} in fees" + (f" plus the {spread:.4%} bid-ask spread now, {total:.6%} "
                                         "in all" if spread is not None else "")
            + " — the move needed to break even.",
            "That is the fee you stated, used as given; Bitget's standard taker fee is 0.06% on "
            "USDT perpetuals and 0.10% on spot, so check which one applies to your account."]


def stated_gas(text: str) -> list[str] | None:
    """A transfer's cost at the gas price and gas units the trader states."""
    m = _GAS.search(text)
    if m is None:
        return None
    gwei, units = _n(m.group(1)), _n(m.group(2))
    if re.search(rf"{re.escape(m.group(2))}\s*k\b", m.group(0), re.I):
        units *= 1000
    eth = gwei * units / 1e9
    out = [f"Bottom line: {gwei:g} gwei x {units:,.0f} gas = {gwei * units:,.0f} gwei = "
           f"{eth:.6f} ETH."]
    try:
        t = _ticker("ETHUSDT")
        if t is not None:
            out[0] = out[0].rstrip(".") + f", about ${eth * float(t.last):,.2f} at ETH "
            out[0] += f"{float(t.last):,.2f}."
    except Exception:
        pass
    out.append("Worked at the gas price you gave; the live base fee changes block by block, so "
               "ask \"what is gas on Ethereum now\" for today's.")
    return out


def wrong_side(text: str) -> list[str] | None:
    """A long whose stop sits above entry, or a short whose target sits above entry (and the
    mirror cases), named and worked."""
    side_m = re.search(r"\b(long|short)\b", text, re.I)
    entry_m = _ENTRY.search(text)
    stop_m, target_m = _STOP.search(text), _TARGET.search(text)
    if side_m is None or entry_m is None or not (stop_m or target_m):
        return None
    side = side_m.group(1).lower()
    entry = _n(entry_m.group(1) or entry_m.group(2))
    qty_m = re.search(r"\bsize\s+(\d+(?:\.\d+)?)|\b(\d+(?:\.\d+)?)\s*(?:btc|eth|sol|coins?|"
                      r"contracts?|shares?)\b", text, re.I)
    qty = _n(qty_m.group(1) or qty_m.group(2)) if qty_m else 1.0
    out: list[str] = []
    if stop_m:
        stop = _n(stop_m.group(1))
        if (side == "long" and stop > entry) or (side == "short" and stop < entry):
            pnl = (stop - entry) * qty * (1 if side == "long" else -1)
            out.append(f"Bottom line: a stop at {stop:,.2f} is on the wrong side of a {side} "
                       f"entered at {entry:,.2f} — for a {side} it would fire at a "
                       f"{'gain' if pnl > 0 else 'loss'} of {abs(pnl):,.2f} on {qty:g} "
                       f"({abs(stop / entry - 1):.1%}), so it is a take-profit, not a stop. "
                       f"A {side} stop sits {'below' if side == 'long' else 'above'} entry.")
    if target_m and not out:
        target = _n(target_m.group(1))
        if (side == "long" and target < entry) or (side == "short" and target > entry):
            pnl = (target - entry) * qty * (1 if side == "long" else -1)
            out.append(f"Bottom line: a take-profit at {target:,.2f} on a {side} from "
                       f"{entry:,.2f} is a loss of {abs(pnl):,.2f} on {qty:g} "
                       f"({abs(target / entry - 1):.1%}) if hit — for a {side} the target sits "
                       f"{'above' if side == 'long' else 'below'} entry; above it, that order "
                       "is a stop.")
    if not out:
        return None
    out.append("Worked on the figures you gave, before fees and funding.")
    return out


_STOP_PCT: Final = re.compile(r"\bstop(?:[\s-]?loss)?\s+(?:is\s+|at\s+)?(?P<p>\d+(?:\.\d+)?)\s*%\s+"
                             r"(?P<dir>above|below|under|over)\s+(?:my\s+|the\s+)?entry\b", re.I)
_PCT_STOP_AT: Final = re.compile(r"\b(?P<p>\d+(?:\.\d+)?)\s*%\s+stop(?:[\s-]?loss)?\b[^?]{0,40}?"
                                 r"\bat\s+\$?(?P<at>\d[\d,]*(?:\.\d+)?)", re.I)


def stop_percent(text: str, prior: list[str] | None = None) -> list[str] | None:
    """A stop stated as a percentage: "my stop is 5% above entry on that long ... 85,000 entry"
    kept neither the entry nor the side and showed a fresh plan (round 43 hostile, M10); "3% stop
    loss on Bitcoin at 90,000" never computed the level (M13)."""
    turns = [*(prior or [])[-3:], text]
    m = _STOP_PCT.search(text)
    if m is not None:
        side_m = next((s for t in reversed(turns) for s in [re.search(r"\b(long|short)\b", t,
                                                                   re.I)] if s), None)
        entry_m = re.search(r"\b(?:on\s+(?:a\s+)?|at\s+|entry\s+(?:of\s+|at\s+)?)\$?"
                            r"(\d[\d,]*(?:\.\d+)?)\s*(?:entry)?\b", text, re.I)
        if side_m is None or entry_m is None:
            return None
        side, pct = side_m.group(1).lower(), float(m.group("p")) / 100
        entry = _n(entry_m.group(1))
        above = m.group("dir").lower() in ("above", "over")
        level = entry * (1 + pct if above else 1 - pct)
        wrong = (side == "long" and above) or (side == "short" and not above)
        earlier = next((t for t in turns[:-1] if _STOP_PCT.search(t) or re.search(
            r"\bstop\b[^.?]{0,30}\b(?:below|above)\b", t, re.I)), None)
        out = [f"Bottom line: a stop {pct:.0%} {'above' if above else 'below'} an {entry:,.0f} "
               f"entry is {level:,.2f}"
               + (f" — on a {side} that is not a stop: it fires at a gain of "
                  f"{abs(level - entry):,.2f} a unit, so it is a take-profit; a {side} stop sits "
                  f"{'below' if side == 'long' else 'above'} entry" if wrong else
                  f", a loss of {abs(level - entry):,.2f} a unit ({pct:.1%}) if hit") + "."]
        if earlier is not None:
            out.append(f"This changes what you said before (\"{earlier.strip()[:80]}\"); the "
                       "figures above use the new stop and the entry in this message.")
        return out
    p = _PCT_STOP_AT.search(text)
    if p is None:
        return None
    pct, at = float(p.group("p")) / 100, _n(p.group("at"))
    short = bool(re.search(r"\bshort\b", text, re.I))
    level = at * (1 + pct if short else 1 - pct)
    out = [f"Bottom line: a {pct:.0%} stop on a {'short' if short else 'long'} from {at:,.0f} sits "
           f"at {level:,.2f}, {abs(at - level):,.2f} a unit away."]
    symbol = _symbol(text)
    if symbol is not None:
        try:
            t = _ticker(symbol)
            if t is not None:
                gap = at / float(t.last) - 1
                out.append(f"{symbol.removesuffix('USDT')} is {float(t.last):,.2f} now, so "
                           f"{at:,.0f} is {gap:+.1%} from the live price"
                           + (" — an entry that far away is a resting order, not a fill now."
                              if abs(gap) > 0.02 else "."))
        except Exception:
            pass
    return out


_STOP_DISTANCE: Final = re.compile(
    r"\bstop(?:[\s-]?loss)?\s+(?:at\s+|of\s+)?(?P<p>\d+(?:\.\d+)?)\s*%\s+(?:below|under|above|"
    r"over)\b|\b(?P<p2>\d+(?:\.\d+)?)\s*%\s+stop(?:[\s-]?loss)?\b(?![^?]{0,40}\bat\s+\$?\d)", re.I)
_SIZE_ASK: Final = re.compile(r"\bhow\s+(?:many|much)\b|\bsize\b|\bposition\b|\ballow\w*\b",
                              re.I)


def risk_sizing(text: str, memory: str = "") -> list[str] | None:
    """Units a risk rule allows at a stop distance: "If I put a stop 8% below entry on SOL, how many
    coins does 1% risk allow?" was answered with a fresh trade plan, the remembered $50,000
    account and 1% rule unused (round 43 judge, M18). Size = account x risk / stop distance, from
    the figures in the question, else the ones the trader asked to be remembered (said so)."""
    m = _STOP_DISTANCE.search(text)
    if m is None or not _SIZE_ASK.search(text):
        return None
    from argus.lui import memory as kept

    stop = float(m.group("p") or m.group("p2")) / 100
    facts = kept.parse(memory)
    risk_m = re.search(r"\b(\d+(?:\.\d+)?)\s*%\s+(?:risk|of\s+(?:the\s+|my\s+)?account)\b|"
                       r"\brisk(?:ing)?\s+(\d+(?:\.\d+)?)\s*%", text, re.I)
    acct_m = re.search(r"\$?(\d[\d,]*(?:\.\d+)?)\s*(k|m)?\s*(?:usd\s+)?(?:account|capital|"
                       r"portfolio)\b", text, re.I)
    used: list[str] = []
    if risk_m:
        risk = float(risk_m.group(1) or risk_m.group(2)) / 100
    else:
        fact = kept.get(facts, "trade_risk")
        if fact is None:
            return None
        risk = float(fact.value)
        used.append(f"your rule “{fact.text}”")
    if acct_m:
        account = _n(acct_m.group(1)) * {"k": 1e3, "m": 1e6}.get((acct_m.group(2) or "").lower(), 1)
    else:
        fact = kept.get(facts, "capital")
        if fact is None:
            return [f"Bottom line: at a {stop:.0%} stop, {risk:.1%} risk allows a position of "
                    f"{risk / stop:.1%} of the account — say the account size and it is worked "
                    "in dollars and units."]
        account = float(fact.value)
        used.append(f"your account “{fact.text}”")
    budget = account * risk
    notional = budget / stop
    out = [f"Bottom line: {risk:.1%} of ${account:,.0f} is ${budget:,.0f} at risk; with the stop "
           f"{stop:.0%} away that allows ${notional:,.0f} of position"]
    symbol = _symbol(text)
    if symbol is not None:
        try:
            t = _ticker(symbol)
            if t is not None:
                units = notional / float(t.last)
                out[0] += (f", about {units:,.2f} {symbol.removesuffix('USDT')} at "
                           f"{float(t.last):,.2f}")
        except Exception:
            pass
    out[0] += "."
    if notional > account:
        out.append(f"That is {notional / account:.1f}x the account, so it needs leverage; the stop "
                   "loses the risk budget only if it fills at the stop, which a gap can skip.")
    if used:
        out.append("Remembered: " + " and ".join(used) + " — used here as you asked to keep it.")
    return out


def funding_verdict(text: str) -> list[str] | None:
    """"Compare BTC and ETH funding and say which has the higher funding rate" listed one rate and
    never said which was higher (round 43 hostile, minor 1)."""
    if not re.search(r"\bfunding\b", text, re.I) or not re.search(
            r"\b(?:which|who)\b[^?]{0,40}\b(?:higher|lower|highest|lowest|more|less|bigger|"
            r"smaller)\b|\bcompare\b", text, re.I):
        return None
    from argus.lui.research.parse import research_symbols

    named = [s for s in research_symbols(text)[0] if s.endswith("USDT")]
    if len(named) < 2:
        return None
    try:
        from argus.market.bitget import fetch_tickers

        tickers = fetch_tickers()
    except Exception:
        return None
    rates = [(s.removesuffix("USDT"), float(tickers[s].funding_rate)) for s in named
             if s in tickers]
    if len(rates) < 2:
        return None
    ranked = sorted(rates, key=lambda r: -r[1])
    top, bottom = ranked[0], ranked[-1]
    def word(rate: float) -> str:
        return "longs pay shorts" if rate > 0 else "shorts pay longs" if rate < 0 else "no one pays"
    return [f"Bottom line: {top[0]} has the higher funding rate, {top[1]:+.4%} per settlement "
            f"against {bottom[0]}'s {bottom[1]:+.4%}"
            + (f" ({', '.join(f'{n} {r:+.4%}' for n, r in ranked[1:-1])} between)"
               if len(ranked) > 2 else "") + ".",
            "; ".join(f"{n}: {word(r)}, about {r * 3 * 365:+.1%} a year at three settlements a day"
                      for n, r in ranked) + ".",
            "Data: Bitget's public tickers (the rate for the next settlement), read just now. "
            "Not advice."]


_EQUITY_METRIC: Final = re.compile(r"\bp\s*/\s*e\b|\bpe\s+ratio\b|\bprice[\s-]to[\s-]earnings\b|"
                                   r"\beps\b|\bearnings\s+per\s+share\b|\bdividend\w*|"
                                   r"\bpayout\s+ratio\b|\bbuybacks?\b", re.I)
_COINS: Final = {"BTC": "Bitcoin", "ETH": "Ether", "SOL": "Solana", "XRP": "XRP",
                 "DOGE": "Dogecoin", "XAU": "Gold", "XAG": "Silver"}


def crypto_equity_note(text: str, said: str) -> str:
    """A line saying a coin (or gold) has no earnings or dividend when the question asks for an
    equity metric of it and the answer does not say so: "BTC funding and also the P/E of ETH" lost
    the P/E part without a word (round 43 hostile, M17-M19)."""
    m = _EQUITY_METRIC.search(text)
    if m is None:
        return ""
    from argus.lui.research.parse import research_symbols

    # the coin the metric is asked of: named right after it ("the P/E of ETH") or right before
    # it ("Bitcoin's P/E"), not every coin in the question
    coins: list[str] = []
    for hit in _EQUITY_METRIC.finditer(text):
        after = [x.removesuffix("USDT") for x in
                 research_symbols(text[hit.end(): hit.end() + 30])[0]]
        before = [x.removesuffix("USDT") for x in
                  research_symbols(text[max(0, hit.start() - 20): hit.start()])[0]]
        coins += [c for c in (after or before) if c in _COINS]
    named = [x.removesuffix("USDT") for x in research_symbols(text)[0]
             if x.removesuffix("USDT") in _COINS]
    if not coins and len(named) == 1:
        coins = named  # "BTC's beta ... and its EPS growth": the one coin named
    if not coins or re.search(r"\bno\s+(?:p/e|dividend|eps)\b|\bno\s+earnings\b(?!\s+calendar)",
                              said, re.I):
        return ""
    names = " and ".join(_COINS[c] for c in dict.fromkeys(coins))
    one = len(set(coins)) == 1
    return (f"Left out: {names} {'has' if one else 'have'} no earnings and "
            f"{'pays' if one else 'pay'} no dividend, so there is no P/E, EPS or dividend yield "
            "to give; a staking or lending rate is a different thing and is not one.")


def sp500_beta(text: str) -> str:
    """The beta to the S&P 500 when it is asked by name: the profile measures against the
    Nasdaq-100 and swapped the benchmark without saying so (round 43 hostile, M19). Daily closes
    over the last year, the instrument's own against SPY's, on the dates both traded."""
    if not re.search(r"\bbeta\b", text, re.I) or not re.search(
            r"\bs\s*&\s*p(?:\s*500)?\b|\bspx\b|\bsp500\b|\bspy\b", text, re.I):
        return ""
    import statistics

    from argus.lui.research.parse import research_symbols
    from argus.lui.research.rule_test import daily_closes

    symbol = next((s for s in research_symbols(text)[0] if s not in ("SPYUSDT", "SP500USDT")),
                  None)
    if symbol is None:
        return ""
    try:
        s1, c1, _ = daily_closes(symbol)
        s2, c2, _ = daily_closes("SPYUSDT")
    except Exception:
        return "Not measured: the S&P 500 beta could not be read just now."
    a = {t.date(): c for t, c in zip(s1, c1, strict=True)}
    b = {t.date(): c for t, c in zip(s2, c2, strict=True)}
    days = sorted(set(a) & set(b))[-253:]
    ra = [a[days[i]] / a[days[i - 1]] - 1 for i in range(1, len(days))]
    rb = [b[days[i]] / b[days[i - 1]] - 1 for i in range(1, len(days))]
    if len(ra) < 60:
        return ""
    beta = statistics.covariance(ra, rb) / statistics.variance(rb)
    corr = statistics.correlation(ra, rb)
    return (f"Against the S&P 500 (SPY), as asked: beta {beta:.2f} on daily closes over the last "
            f"year ({len(ra)} days both traded), correlation {corr:+.2f}.")


def fraction_or_percent(text: str) -> list[str] | None:
    """"Is 0.4 ... 40%?": a fraction against a percentage, with the live 24-hour change."""
    m = _FRACTION_PCT.search(text)
    if m is None:
        return None
    f, pct = float(m.group("f")), _n(m.group(2))
    if abs(f * 100 - pct) > 1e-9:
        return None
    out = [f"Bottom line: it depends what the {f:g} is. As a fraction, {f:g} is {pct:g}%; but a "
           f"24-hour change shown as {f:g} on a price screen is already in percent, so it means "
           f"{f:g}%, not {pct:g}%."]
    symbol = _symbol(text)
    if symbol is not None:
        try:
            t = _ticker(symbol)
            if t is not None:
                out.append(f"{symbol.removesuffix('USDT')}'s 24-hour change on Bitget now is "
                           f"{float(t.change_24h) * 100:+.2f}%.")
        except Exception:
            pass
    return out


def bps_versus_percent(text: str) -> list[str] | None:
    """A fee in basis points against one in percent, on a stated notional."""
    m = _BPS_VS_PCT.search(text)
    notional_m = _NOTIONAL.search(text)
    if m is None or notional_m is None:
        return None
    bps = _n(m.group("a") or m.group("d"))
    pct = _n(m.group(2) or m.group("c"))
    notional = _n(notional_m.group(1)) * (1000 if notional_m.group("k") else 1)
    a, b = notional * bps / 10_000, notional * pct / 100
    return [f"Bottom line: on ${notional:,.0f}, {bps:g} bps is ${a:,.2f} and {pct:g}% is "
            f"${b:,.2f} — {pct:g}% is {pct * 100:g} bps, {b / a:,.0f} times as much." if a else
            f"Bottom line: on ${notional:,.0f}, {pct:g}% is ${b:,.2f}.",
            "A basis point is a hundredth of a percent: 1 bps = 0.01%."]


def in_currency(text: str, payload_lines: list[str]) -> str:
    """A line restating the dollar figures of an answer in the currency asked, or ""."""
    m = _IN_EUROS.search(text)
    if m is None or not payload_lines:
        return ""
    from argus.lui.research.parse import in_us_dollars

    code = {"e": "EUR", "p": "GBP", "g": "GBP", "s": "GBP", "y": "JPY", "j": "JPY"}[
        m.group("cur")[0].lower()]
    restated, said = in_us_dollars(f"1000000 {code}")
    rate_m = re.search(r"\$(\d[\d,]*(?:\.\d+)?)", restated)
    if not rate_m or not said:
        return f"Not converted: no {code} rate answered just now, so the figures are in dollars."
    usd_per = _n(rate_m.group(1)) / 1_000_000
    first = str(payload_lines[0])
    sums = re.findall(r"\$(\d[\d,]*(?:\.\d+)?)\s*(bn|m|k)?\b", first)
    if not sums:
        return f"In {code}: at Bitget's {code}USD {usd_per:.4f}, divide the dollar figures by it."
    parts = []
    for amount, unit in sums[:3]:
        value = _n(amount) * {"bn": 1e9, "m": 1e6, "k": 1e3}.get(unit, 1.0) / usd_per
        shown = (f"{value / 1e9:,.2f}bn" if value >= 1e9 else f"{value / 1e6:,.1f}m"
                 if value >= 1e6 else f"{value:,.0f}")
        parts.append(f"${amount}{unit} is about {shown} {code}")
    return f"In {code} (Bitget's {code}USD {usd_per:.4f}): " + "; ".join(parts) + "."


READERS: Final = (per_period, minor_units, false_premise, stated_fee, stated_gas, wrong_side,
                  fraction_or_percent, bps_versus_percent)


def lines(text: str) -> list[str] | None:
    """The first unit or premise check that applies to ``text``, or None."""
    for reader in READERS:
        said = reader(text)
        if said is not None:
            return said
    return None


__all__ = [
    "READERS",
    "bps_versus_percent",
    "crypto_equity_note",
    "false_premise",
    "fraction_or_percent",
    "in_currency",
    "lines",
    "minor_units",
    "per_period",
    "sp500_beta",
    "stated_fee",
    "stated_gas",
    "stop_percent",
    "wrong_side",
]
