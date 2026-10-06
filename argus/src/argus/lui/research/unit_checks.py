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
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Final

from argus.lui.numbers import sig

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
    # "Funding on BTC is 0.03 percent per hour, true?" asks no annual figure, but states a rate
    # to be checked (round 45 re-ask): it is checked against the live rate, and annualised too
    checked = bool(re.search(r"\bfunding\b", text, re.I)) and bool(re.search(
        r"\btrue\b|\bright\b|\bcorrect\b|\bisn'?t\s+it\b|\bconfirm\b", text, re.I))
    if m is None or not (_ANNUAL_ASK.search(text) or checked):
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
    if re.search(r"\bfunding\b", text, re.I):
        live = _live_funding_line(text, rate, hours)
        if live is not None:
            lead, extra = live
            if lead:
                out = [lead, *[x.removeprefix("Bottom line: ") for x in out]]
            if extra:
                out.insert(1, extra)
        held = re.search(r"\b(?:hold|holding|on|keep)\s+(?:a\s+)?\$?(?P<v>\d[\d,]*(?:\.\d+)?)\s*"
                         r"(?P<k>k)?\s*(?:usdt|usd|dollars?|\$)?\b", text, re.I)
        if held is not None:
            size = _n(held.group("v")) * (1000 if held.group("k") else 1)
            year = re.search(r"\b(?:a|one|1)\s+year\b|\bper\s+year\b|\bannual", text, re.I)
            if size >= 10 and year:
                # "What does it cost to hold a 10,000 USDT BTC long for a year at 0.01% funding
                # every 8 hours? My exchange says it's free." got percentages only (round 45
                # hostile, M14): the dollars, and the claim answered
                cost = size * rate * periods
                out.insert(1, f"On ${size:,.0f} held a year that is about ${cost:,.0f} paid "
                              f"(simple, if the rate held)"
                              + (" — so not free: funding is paid between traders every "
                                 "interval, on top of trading fees" if re.search(
                                     r"\bfree\b", text, re.I) else "") + ".")
    out.append("Funding is quoted simple and paid on the position's size, not the margin.")
    return out


def _live_funding_line(text: str, rate: float, hours: int) -> tuple[str, str] | None:
    """The named contract's real funding beside the rate stated: "Funding on ETHUSDT is 0.05
    percent per hour right?" was annualised and silently agreed with (round 45 hostile, M12), and
    "is it 0.01 bps or 0.01 percent per 8 hours" never said which (M13)."""
    symbol = _symbol(text)
    if symbol is None:
        return None
    try:
        # the stated rate is still annualised when the live read fails
        ticker = _ticker(symbol)
    except Exception:
        return None
    if ticker is None:
        return None
    try:
        live = float(ticker.funding_rate)
    except (TypeError, ValueError, AttributeError):
        return None
    name = symbol.removesuffix("USDT")
    live_per_hour = live / 8
    stated_per_hour = rate / hours
    checked = re.search(r"\bright\b|\bcorrect\b|\bisn'?t\s+it\b|\btrue\b|\bis\s+(?:the\s+)?"
                        r"funding\b|\bor\b", text, re.I)
    said_now = f"{name}'s funding on Bitget is {live:+.4%} per 8 hours now"
    both = re.search(r"(?P<a>\d+(?:\.\d+)?)\s*(?P<ua>bps|bp|basis\s+points?|%|percent)\s+or\s+"
                     r"(?P<b>\d+(?:\.\d+)?)\s*(?P<ub>bps|bp|basis\s+points?|%|percent)", text, re.I)
    if both is not None:
        # "Is it 0.01 bps or 0.01 percent per 8 hours?" asks which of two units is right
        def as_rate(v: str, u: str) -> float:
            return float(v) / (100 if u.lower() in ("%", "percent") else 10_000)

        a = as_rate(both.group("a"), both.group("ua"))
        b = as_rate(both.group("b"), both.group("ub"))
        nearer = both.group("a") + " " + both.group("ua") if abs(a - abs(live)) <= abs(
            b - abs(live)) else both.group("b") + " " + both.group("ub")
        return (f"Bottom line: {said_now} ({live * 1e4:+.2f} bps), so of the two, {nearer} is "
                f"nearer the real rate; the arithmetic on the rate you wrote first follows.", "")
    if checked and abs(stated_per_hour - live_per_hour) > max(abs(live_per_hour) * 0.5, 1e-7):
        return (f"Bottom line: no — {said_now}, not the rate stated; the arithmetic on the stated "
                f"rate follows.", "")
    if re.search(r"\bbps\b[^?]{0,30}\bor\b[^?]{0,30}\bpercent|\bpercent\b[^?]{0,30}\bor\b"
                 r"[^?]{0,30}\bbps\b", text, re.I):
        return ("", f"Which unit: {said_now}, which is {live * 1e4:.2f} bps — a funding rate "
                    f"is quoted in percent per interval.")
    return ("", f"For comparison, {said_now}.")


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


_RR_ASKED: Final = re.compile(
    r"\breward[\s-]*(?:to|/|vs\.?|versus)?[\s-]*risk\b|\brisk[\s-]*(?:to|/|vs\.?|versus)?[\s-]*"
    r"reward\b|\br\s*:\s*r\b|\brr\b|\brisk[\s-]*reward\s+ratio\b", re.I)
_RR_STOP: Final = re.compile(
    r"\bstop(?:[\s-]*loss)?\b(?:\s+on\s+\w+)?\s+(?:at|@|to|of)\s+\$?(?P<v>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<pct>%)?(?:\s+(?:below|under|away|down))?", re.I)
_RR_TARGET: Final = re.compile(
    r"\b(?:take[\s-]*profit|target|tp|profit\s+target)\b\s*(?:at|@|to|of)?\s+\$?"
    r"(?P<v>\d[\d,]*(?:\.\d+)?)\s*(?P<pct>%)?(?:\s+(?:above|over|away|up|from\s+entry))?", re.I)


def stated_reward_risk(text: str) -> list[str] | None:
    """Reward to risk on the stop and target the trader set, never on levels the console picked:
    "Set my stop on SOL at 0 and take profit 1000% away, what is the reward to risk?" got a fresh
    plan with its own stop and target (round 44 hostile, C3)."""
    if _RR_ASKED.search(text) is None:
        return None
    stop_m, target_m = _RR_STOP.search(text), _RR_TARGET.search(text)
    if stop_m is None or target_m is None:
        return None
    from argus.lui.research.parse import last_price, research_symbols

    named = research_symbols(text)[0]
    entry_m = re.search(r"\b(?:entry|entered|bought|in)\s+(?:at\s+|of\s+)?\$?(\d[\d,]*(?:\.\d+)?)",
                        text, re.I)
    entry = _n(entry_m.group(1)) if entry_m else None
    if entry is None and named:
        try:
            entry = float(last_price(named[0]) or 0) or None
        except Exception:
            entry = None
    if not entry:
        return None
    short = re.search(r"\bshort\b", text, re.I) is not None
    sign = -1 if short else 1
    stop_v, target_v = _n(stop_m.group("v")), _n(target_m.group("v"))
    stop = entry * (1 - sign * stop_v / 100) if stop_m.group("pct") else stop_v
    target = entry * (1 + sign * target_v / 100) if target_m.group("pct") else target_v
    risk, reward = sign * (entry - stop), sign * (target - entry)
    name = named[0].removesuffix("USDT") if named else "the position"
    basis = f"{'entry' if entry_m else 'last price on Bitget'} of {entry:,.2f}"
    if risk <= 0:
        return [f"Bottom line: the stop at {stop:,.2f} is on the wrong side of the {basis} for a "
                f"{'short' if short else 'long'} — it fires at a gain, so there is no risk to "
                f"divide by and no reward-to-risk ratio.",
                f"A {'short' if short else 'long'} stop sits {'above' if short else 'below'} "
                f"entry."]
    if reward <= 0:
        return [f"Bottom line: the target at {target:,.2f} is on the losing side of the {basis}, "
                f"so reaching it is a loss — there is no reward to set against the risk."]
    ratio = reward / risk
    lines = [f"Bottom line: reward to risk is {ratio:,.1f} to 1 on the levels you set — target "
             f"{target:,.2f} ({reward / entry:+.1%}), stop {stop:,.2f} "
             f"({-risk / entry:+.1%}), from the {basis}."]
    if stop <= 0 and not short:
        lines.append(f"A stop at 0 is no stop: {name} would have to become worthless to trigger "
                     f"it, so the whole stake is the risk — the ratio is real arithmetic but the "
                     f"protection is none.")
    lines.append(f"The ratio says nothing about the odds of reaching either level; a "
                 f"{reward / entry:+.0%} target is a question of how often {name} has moved that "
                 f"far, which is a separate ask (\"how often has {name} risen "
                 f"{reward / entry:.0%} in a year\").")
    return lines


_MOVE_FROM_TO: Final = re.compile(
    r"\bfrom\s+\$?(?P<a>\d[\d,]*(?:\.\d+)?)\s+to\s+\$?(?P<b>\d[\d,]*(?:\.\d+)?)\b", re.I)
_PCT_STOP_SET: Final = re.compile(
    r"\b(?P<p>\d+(?:\.\d+)?)\s*%\s+stop(?:[\s-]*loss)?\b[^?.]{0,30}?\b(?:at|@)\s+\$?"
    r"(?P<s>\d[\d,]*(?:\.\d+)?)", re.I)


def move_and_stop(text: str) -> list[str] | None:
    """A move between two stated prices in bps, and whether a stated percent stop sits where it
    says: "How many bps is a move from 2700 to 2727 in ETH, and is the 1% stop I set at 2673
    correct?" got a stop 1% below 2673 instead (round 45 hostile, C6)."""
    move = _MOVE_FROM_TO.search(text)
    if move is None or not re.search(r"\bbps\b|\bbasis\s+points?\b|\bstop\b", text, re.I):
        return None
    a, b = _n(move.group("a")), _n(move.group("b"))
    if a <= 0 or b <= 0:
        return None
    change = b / a - 1
    out = [f"Bottom line: {move.group('a')} to {move.group('b')} is {change * 1e4:+,.0f} bps "
           f"({change:+.2%})."]
    stop = _PCT_STOP_SET.search(text)
    if stop is not None:
        pct, level = float(stop.group("p")) / 100, _n(stop.group("s"))
        short = re.search(r"\bshort\b", text, re.I) is not None
        right = a * (1 + pct) if short else a * (1 - pct)
        off = level / a - 1
        good = abs(level - right) <= max(0.0005 * a, 0.01)
        out[0] += (f" Your {stop.group('p')}% stop at {stop.group('s')} is "
                   + (f"right: exactly {abs(off):.2%} {'above' if short else 'below'} "
                      f"{move.group('a')}, for a {'short' if short else 'long'} from there."
                      if good else
                      f"not where {stop.group('p')}% puts it: {stop.group('p')}% "
                      f"{'above' if short else 'below'} {move.group('a')} is {right:,.2f}, and "
                      f"{stop.group('s')} is {off:+.2%} from {move.group('a')}."))
    return out


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
    out = [f"Bottom line: {top[0]} has the higher funding rate, {top[1]:+.4%} per settlement "
           f"against {bottom[0]}'s {bottom[1]:+.4%}"
           + (f" ({', '.join(f'{n} {r:+.4%}' for n, r in ranked[1:-1])} between)"
              if len(ranked) > 2 else "") + ".",
           "; ".join(f"{n}: {word(r)}, about {r * 3 * 365:+.1%} a year at three settlements a day"
                     for n, r in ranked) + "."]
    data = "Data: Bitget's public tickers (the rate for the next settlement), read just now"
    if re.search(r"\bvol\w*|\bdrawdowns?\b|\bdraw\s*down|\bswings?\b|\brisk\w*", text, re.I):
        # "Compare ETH and SOL: volatility, drawdown, funding" got funding only (round 45 judge,
        # m5): each dimension named is answered, from one year of daily closes
        measured = _vol_and_drawdown(named)
        if measured:
            out.insert(1, measured[0])
            data += "; " + measured[1]
    out.append(data + ". Not advice.")
    return out


def _vol_and_drawdown(symbols: list[str]) -> tuple[str, str] | None:
    """One line ranking ``symbols`` by realised volatility over the last year of daily closes,
    with each one's worst fall from a high in that year; and its data clause."""
    import math
    from itertools import pairwise
    from statistics import stdev

    from argus.lui.research.rule_test import daily_closes

    rows: list[tuple[str, float, float]] = []
    span = ""
    for symbol in symbols:
        try:
            stamps, closes, _ = daily_closes(symbol)
        except Exception:
            continue
        year = closes[-366:]
        if len(year) < 60:
            continue
        returns = [b / a - 1 for a, b in pairwise(year)]
        peak, worst = year[0], 0.0
        for close in year:
            peak = max(peak, close)
            worst = min(worst, close / peak - 1)
        rows.append((symbol.removesuffix("USDT"), stdev(returns) * math.sqrt(365), worst))
        span = f"{stamps[-len(year)]:%d %b %Y} to {stamps[-1]:%d %b %Y}"
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: -r[1])
    return ("Volatility and drawdown over the last year: "
            + "; ".join(f"{n} {v:.0%} a year, worst fall from a high {abs(d):.0%}"
                        for n, v, d in rows)
            + f" — {rows[0][0]} swings the most.",
            f"Bitget daily closes {span} (volatility annualised over 365 days)")


_HOLDING_SAID: Final = re.compile(
    r"\b(?P<q>\d[\d,]*(?:\.\d+)?)\s+(?:shares?\s+(?:of\s+)?)?(?P<sym>[A-Za-z]{2,6})\b"
    r"(?:\s+shares?)?", re.I)
_TOTAL_SAID: Final = re.compile(
    r"\b(?:for|cost(?:\s+me)?|total(?:\s+(?:cost|outlay))?(?:\s+(?:of|is|was))?|worth|valued?\s+at|paid|"
    r"outlay(?:\s+(?:of|is|was))?|spent|invested)\s+"
    r"\$?(?P<t>\d[\d,]*(?:\.\d+)?)\s*(?P<u>k|m)?\b(?:\s+(?:total|in\s+total|all\s+in))?"
    r"(?!\s*(?:each|a\s+(?:share|coin)|per))", re.I)
_UNIT_PRICE_SAID: Final = re.compile(
    r"\b(?:at|bought\s+at|@)\s+(?:a\s+)?\$?(?P<p>\d[\d,]*(?:\.\d+)?)\s*(?P<u>k)?\s*(?:each|a\s+"
    r"(?:share|coin)|per\s+\w+|price)?\b", re.I)


def _signed_usd(x: float) -> str:
    """A dollar change with its sign before the dollar: -$102, +$1,500."""
    return f"{'-' if x < 0 else '+'}${abs(x):,.0f}"


def position_consistency(text: str) -> list[str] | None:
    """A position whose stated figures disagree: "0.5 BTC for $100,000 total at today's price"
    was read as a $100,000 entry per coin, and "100 NVDA bought at $50 each, total cost $100,000"
    and "10 BTC worth $5,000 at a $68,000 price" passed with no word (round 43 hostile, C1, C2,
    M1). Quantity x price, the stated total and the live price are set side by side, the
    disagreement is said, and the loss asked for is worked on the stated cost and today's value
    rather than on a misread figure."""
    from argus.lui.research.parse import research_symbols

    # "I paid $90,000 in total for 0.25 ETH": the first number-and-word is "000 in", not a
    # holding; the first pair whose word is a listed name is (round 44 re-ask)
    held = next((m for m in _HOLDING_SAID.finditer(text)
                 if research_symbols(m.group("sym"))[0]), None)
    if held is None:
        return None
    # "execute the trade for 500 NVDA": the number after "for" is the quantity itself, not a
    # total beside it, and reading it as both fetched a price for every order (suite, round 44)
    total_m = next((m for m in _TOTAL_SAID.finditer(text)
                    if m.start("t") != held.start("q")), None)
    if total_m is None:
        return None
    found = research_symbols(held.group("sym"))[0]
    if not found:
        return None
    symbol = found[0]
    qty = _n(held.group("q"))
    total = _n(total_m.group("t")) * {"k": 1e3, "m": 1e6}.get((total_m.group("u") or "").lower(), 1)
    if qty <= 0 or total <= 0:
        return None
    price_m = _UNIT_PRICE_SAID.search(text, total_m.end()) or _UNIT_PRICE_SAID.search(text)
    unit_price = (_n(price_m.group("p")) * (1000 if price_m.group("u") else 1)
                  if price_m and _n(price_m.group("p")) != total else None)
    try:
        ticker = _ticker(symbol)
        live = float(ticker.last) if ticker is not None else None
    except Exception:
        live = None
    name = symbol.removesuffix("USDT")
    clashes: list[str] = []
    implied = total / qty
    if unit_price is not None and abs(qty * unit_price / total - 1) > 0.05:
        clashes.append(f"{qty:,.10g} {name} at {unit_price:,.2f} is ${qty * unit_price:,.0f}, not "
                       f"the ${total:,.0f} stated")
    if live is not None and abs(implied / live - 1) > 0.10:
        clashes.append(f"${total:,.0f} for {qty:,.10g} {name} is {implied:,.2f} a unit, while "
                       f"{name} is {live:,.2f} on Bitget now, so {qty:,.10g} {name} is worth "
                       f"${qty * live:,.0f} today")
    if not clashes:
        return None
    said_as = ("was worth" if re.search(r"\bworth\b|\bvalued?\b", total_m.group(0), re.I)
               else "cost")
    out = ["Bottom line: these figures disagree — " + "; and ".join(clashes) + "."]
    move = re.search(r"\b(?:falls?|drops?|declines?|rises?|gains?|moves?)\s+(?:by\s+)?"
                     r"(?P<m>\d+(?:\.\d+)?)\s*%", text, re.I)
    if move is not None and live is not None and not re.search(
            r"\bnasdaq|\bs&p|\bmarket\b|\bindex\b|\bqqq\b|\bspy\b", text[:move.start()], re.I):
        pct = float(move.group("m")) / 100 * (-1 if re.search(
            r"\b(?:falls?|drops?|declines?)\b", move.group(0), re.I) else 1)
        after = qty * live * (1 + pct)
        out.append(f"Worked both ways: a {pct:+.0%} move from today's {live:,.2f} takes "
                   f"{qty:,.10g} {name} to ${after:,.0f} — {_signed_usd(after - qty * live)} on "
                   f"today's value, and {_signed_usd(after - total)} against the ${total:,.0f} you "
                   f"said it "
                   f"{said_as}.")
    elif move is not None:
        out.append("A market or index move reaches a stock through its beta, not one for one; "
                   "restate the position and ask again, and the move is worked through the beta.")
    out.append("Say which figure is right — the quantity, the price or the total — and the "
               "answer is worked on that.")
    return out


_LEVERAGES: Final = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*x\b", re.I)
_MARGIN_PCT: Final = re.compile(r"\b(\d+(?:\.\d+)?)\s*%\s+(?:initial\s+)?margin\b|\bmargin\s+"
                                r"(?:is\s+|of\s+)?(\d+(?:\.\d+)?)\s*%", re.I)


def leverage_notes(text: str) -> str:
    """A line saying where the leverage a question states disagrees with itself: "10x leverage
    with 100% margin" (100% margin is 1x), "$1,000 margin on 5x to hold $20,000" ($20,000 on
    $1,000 is 20x), "3x and 2x at once", and "open interest is 50x leverage" (open interest is a
    count of contracts, not a leverage) all passed with one figure silently used (round 44
    hostile, M2-M4)."""
    if not re.search(r"\bleverag\w*|\bmargin\b|\bliquidat\w*", text, re.I):
        return ""
    said: list[str] = []
    levels = [float(m.group(1)) for m in _LEVERAGES.finditer(text)]
    used = levels[0] if levels else None
    if len(set(levels)) > 1:
        said.append(f"two leverages are stated ({' and '.join(f'{x:g}x' for x in levels)}); a "
                    f"position has one, so {used:g}x, the first, is the one worked")
    margin = _MARGIN_PCT.search(text)
    if margin is not None and used:
        pct = float(margin.group(1) or margin.group(2))
        if pct > 0 and abs(100 / pct - used) / used > 0.1:
            said.append(f"{pct:g}% margin means {sig(100 / pct, 2)}x leverage, not {used:g}x — the "
                        f"margin is the share of the position you put up")
    sums = re.search(r"\$?(\d[\d,]*)\s*(?:of\s+)?margin\b[^?.]{0,60}?\$?(\d[\d,]*)\s+(?:of|in|"
                     r"worth)\b|\$(\d[\d,]*)[^?.]{0,40}\bmargin\b[^?.]{0,60}?\$(\d[\d,]*)", text,
                     re.I)
    if sums is not None and used:
        posted = _n(sums.group(1) or sums.group(3))
        position = _n(sums.group(2) or sums.group(4))
        if posted > 0 and position > posted and abs(position / posted - used) / used > 0.1:
            said.append(f"${position:,.0f} held on ${posted:,.0f} of margin is "
                        f"{sig(position / posted, 3)}x, not {used:g}x")
    if re.search(r"\bopen\s+interest\s+is\s+\d+(?:\.\d+)?\s*x\b", text, re.I):
        said.append("open interest is the number of contracts open, not a leverage")
    if not said:
        return ""
    return "Check: " + "; ".join(said) + "."


def stop_versus_liquidation(text: str) -> list[str] | None:
    """Whether a stop fires before liquidation: "long with 20x leverage and put my stop at 1%
    below, will I be liquidated first?" got the liquidation price with the stop never compared
    (round 44 hostile, M14). Isolated margin, Bitget's 0.40% maintenance margin as the leverage
    reader uses; the stop as a market order that can slip in a gap."""
    if not re.search(r"\bliquidat\w*\b[^?]{0,40}\b(?:first|before)\b|\bstop\b[^?]{0,60}\b(?:before|"
                     r"first)\b[^?]{0,30}\bliquidat", text, re.I):
        return None
    lev = _LEVERAGES.search(text)
    stop = re.search(r"\bstop\w*\b[^?.]{0,30}?(\d+(?:\.\d+)?)\s*%\s*(?:below|under|above|"
                     r"over|away)?", text, re.I)
    if lev is None or stop is None:
        return None
    leverage, stop_pct = float(lev.group(1)), float(stop.group(1)) / 100
    short = bool(re.search(r"\bshort\b", text, re.I))
    liq = 1 / leverage - 0.004
    entry_m = re.search(r"\bat\s+\$?(\d[\d,]*(?:\.\d+)?)\b", text)
    entry = _n(entry_m.group(1)) if entry_m else None
    first = "the stop" if stop_pct < liq else "liquidation"
    out = [f"Bottom line: {first} comes first — at {leverage:g}x a {'short' if short else 'long'} "
           f"is liquidated about {liq:.1%} from entry, and your stop is {stop_pct:.1%} away"
           + (f" ({entry * (1 + stop_pct if short else 1 - stop_pct):,.2f} against liquidation "
              f"near {entry * (1 + liq if short else 1 - liq):,.2f})" if entry else "") + "."]
    if first == "the stop":
        out.append(f"Hitting the stop loses about {stop_pct * leverage:.0%} of the margin "
                   f"({stop_pct:.1%} x {leverage:g}), before fees; a fast gap can fill a market "
                   "stop past its level, and the closer it sits to liquidation the less room "
                   "that leaves.")
    else:
        out.append("A stop past the liquidation line never fires: the exchange closes the "
                   "position first and keeps the margin. Move the stop inside, or lower the "
                   "leverage.")
    out.append("Isolated margin with Bitget's 0.40% maintenance margin; cross margin moves the "
               "liquidation line by whatever else the account holds. Not advice.")
    return out


_WEEKDAYS: Final = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def market_open_lines(text: str, today: date | None = None) -> list[str] | None:
    """Whether the US stock market is open on the day asked, from NYSE's holiday calendar
    (`market.rtoken_spot.holidays`, which matches nyse.com's 2026 table exactly, checked
    2026-10-06): "Is the stock market open on Saturday 10 October 2026?" was answered "yes" because
    Bitget's perpetual trades at weekends, and Columbus Day was left unanswered (round 44 hostile,
    M8, M12). A price "then" on a future day does not exist yet, and is said."""
    m = re.search(r"\b(?:is|are|will)\s+(?:the\s+)?(?:us\s+|u\.s\.\s+|stock\s+|nyse\s+|nasdaq\s+)*"
                  r"(?:stock\s+)?(?:market|markets|exchange|nyse|nasdaq)\s+(?:be\s+)?open\b", text,
                  re.I)
    if m is None:
        return None
    from argus.lui.journal import find_date
    from argus.market.rtoken_spot import holidays, is_trading_day

    today = today or date.today()
    day = find_date(text[m.end():], today) or find_date(text, today)
    named = re.search(r"\b(" + "|".join(_WEEKDAYS) + r")\b", text, re.I)
    if day is None and named is not None:
        ahead = (_WEEKDAYS.index(named.group(1).lower()) - today.weekday()) % 7
        day = today + timedelta(days=ahead)
    if day is None:
        return None
    open_ = is_trading_day(day)
    reason = ("a weekend" if day.weekday() >= 5 else
              "an NYSE holiday" if day in holidays(day.year) else "a regular trading day")
    out = [f"Bottom line: {'yes' if open_ else 'no'} — {day:%A %d %B %Y} is {reason}, so US stock "
           f"exchanges are {'open' if open_ else 'closed'}"
           + (" (9:30 to 16:00 New York time)" if open_ else "")
           + ". Bitget's stock perpetuals trade around the clock, but that is not the stock "
             "market being open."]
    if re.search(r"\bcolumbus|\bindigenous", text, re.I) and open_:
        out.append("Columbus Day is a bank and bond-market holiday, not an NYSE one: stocks trade, "
                   "the Treasury market is closed.")
    if day > today and re.search(r"\bprice\b|\bvolume\b|\bclose\b", text, re.I):
        out.append(f"A price or volume for {day:%d %b %Y} does not exist yet — it is in the "
                   "future; ask for today's.")
    out.append("Data: NYSE's published holiday calendar. Not advice.")
    return out


_INTERVAL: Final = re.compile(r"\b(?P<n>-?\d+(?:\.\d+)?)[\s-]*(?P<u>seconds?|secs?|s|minutes?|"
                              r"mins?|m|hours?|h)[\s-]+(?:chart|candles?|timeframe|bars?)\b", re.I)
_CLOCK: Final = re.compile(r"\b(?P<h>\d{1,2}):(?P<m>\d{2})\b")


def premise_notes(text: str, today: date | None = None) -> list[str]:
    """Premises the question states that cannot hold, each said in a line the answer leads with
    (round 44 hostile, M5, M6, M7, M9, M10, M15, M17)."""
    today = today or date.today()
    said: list[str] = []
    iv = _INTERVAL.search(text)
    if iv is not None and re.search(r"\bchart|\bcandle|\btechnical|\brsi\b|\bmacd\b", text, re.I):
        n, unit = float(iv.group("n")), iv.group("u").lower()
        if n <= 0:
            said.append(f"a {iv.group('n')}-{unit} chart is not a timeframe; the standard 4-hour "
                        "chart is used")
        elif unit.startswith("s"):
            said.append(f"{n:g}-second candles are not offered (the shortest is 1 minute); the "
                        "standard 4-hour chart is used")
    clock = _CLOCK.search(text)
    if clock is not None and (int(clock.group("h")) > 23 or int(clock.group("m")) > 59):
        said.append(f"{clock.group(0)} is not a time of day (hours run 00 to 23, minutes 00 to "
                    "59), so no price at that time is given; the day's figures follow")
    year = re.search(r"\b(?:in\s+(?:the\s+year\s+)?)?(?P<y>19\d\d|200\d|201[0-7])\b", text)
    if year is not None and re.search(r"\bbitget\b|\bperpetual|\bperp\b", text, re.I):
        said.append(f"Bitget's perpetuals did not exist in {year.group('y')} (the exchange "
                    "launched in 2018), so there is no Bitget price for that year; today's "
                    "figures follow")
    if re.search(r"\bdominance\b[^?]{0,30}\b(?:1\d\d|[2-9]\d\d)\s*%|\b(?:1\d\d|[2-9]\d\d)\s*%\s+"
                 r"dominance", text, re.I):
        said.append("a market share cannot exceed 100% — a coin is one part of the market it is "
                    "measured against")
    look = re.search(r"\b(?:last|past|previous|prior)\s+-\s*(\d+)\s*(days?|weeks?|months?)",
                     text, re.I)
    if look is not None:
        said.append(f"a look-back cannot be negative; the minus sign is dropped and "
                    f"{look.group(1)} {look.group(2)} back is meant")
    huge = re.search(r"\b10\s*\^\s*\$?(\d{2,})|\b1e\+?(\d{2,})\b", text, re.I)
    if huge is not None and re.search(r"\bbtc\b|\bbitcoin\b", text, re.I) and re.search(
            r"\bmarket\s+cap|\breach|\bif\b", text, re.I):
        power = int(huge.group(1) or huge.group(2))
        said.append(f"at 10^{power} dollars a coin, Bitcoin's roughly 20.1 million coins would be "
                    f"worth about 2 x 10^{power + 7} dollars — a hypothetical with no market "
                    "behind it; today's value follows")
    to_tiny = re.search(r"\b(?:goes|go|falls?|drops?|crash(?:es)?|hits?|reach(?:es)?|trades?)\s+"
                        r"(?:to|at)\s+\$?(?P<p>(?:\d+(?:\.\d+)?)?e-\d+|0\.0{5,}\d+)\s*(?:dollars?"
                        r"|usd|\$)?", text, re.I)
    if to_tiny is not None:
        # "what if BTC goes to 1e-12 dollars" was refused as "a -100% move is not possible"
        # (round 44 hostile, minor 12): the price is possible to write, and the fall to it is a
        # total loss to any precision a book is kept in
        written = Decimal(to_tiny.group("p"))
        said.append(f"a price of {to_tiny.group('p')} dollars is ${format(written, 'f')} — a fall "
                    "of more than 99.9999% from any price a coin trades at, so a position at it "
                    "is worth nothing to the cent; the figures below are the real risk, not that "
                    "price")
    tiny = re.search(r"\b(?:drop(?:s|ped)?|fall(?:s|en)?|fell|los(?:e|es|t)|crash(?:es|ed)?)"
                     r"\s+(?:by\s+)?(99\.9\d*)\s*%", text, re.I)
    if tiny is not None:
        # exact decimal arithmetic: 100 - 99.99999 in floats is 1.0000000003174137e-05
        left = format(Decimal(100) - Decimal(tiny.group(1)), "f")
        said.append(f"a {tiny.group(1)}% fall is not quite a total loss: {left}% of the value "
                    "remains")
    return said


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
    if any(re.search(r"\b(?:EUR|JPY|GBP)\s+\(at\s+Bitget's", str(x)) for x in payload_lines):
        # the answer already converts at Bitget's own rates (round 45): a second restatement
        # beside it only repeats the instruction
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


_SAME_TWICE: Final = re.compile(
    r"\b([A-Z]{2,6})\s+(?:and|vs\.?|versus|or|with|against|to)\s+\1\b(?![-/])")
_PAIR_FORM: Final = re.compile(
    r"\b(?P<a>[A-Z]{2,6})\s+(?:and|vs\.?|versus|or|against)\s+(?P=a)[-/]?(?P<q>USD|USDT|USDC)\b|"
    r"\b(?P<b>[A-Z]{2,6})[-/]?(?P<r>USD|USDT|USDC)\s+(?:and|vs\.?|versus|or|against)\s+(?P=b)\b")
_STOCK_WORD: Final = re.compile(r"\b(?:the\s+)?(?:stock|shares?|equity|company|nyse|nasdaq)\b",
                                re.I)


def name_notes(text: str) -> list[str]:
    """Lines on how the names asked about were read, when the question makes it ambiguous
    (round 44 hostile, minors 3-5):

    - "Compare BTC and BTC" got a one-name profile with no word that the two names are one;
    - "What is SOL vs SOL-USD?" got a risk profile, where the answer is that they are the same
      asset written two ways;
    - "COMP price — same as the stock COMP?" and "UNI ... Uniswap or the stock?" got the coin with
      the stock half unanswered: the token and the US-listed company sharing its ticker are two
      different things, and only the token trades on Bitget."""
    out: list[str] = []
    if re.search(r"\b(?:is|are)\s+(?:ETC|ethereum\s+classic)\s+(?:just|the\s+same\s+as|really)\b|"
                 r"\bETC\b[^?]{0,30}\bjust\s+ETH\b|\bsame\s+(?:as|thing)\b[^?]{0,20}\bETH\b|"
                 r"\b(?:the\s+)?same\s+(?:coin|chain|thing|token|crypto)\b|\bdifferent\s+coins?\b",
                 text, re.I) and re.search(r"\bETC\b|ethereum\s+classic", text, re.I):
        # "Compare ETH vs 'Ethereum Classic' - is ETC just ETH?" got a volatility ranking only
        # (round 45 hostile, m8)
        out.append("ETC is not ETH: Ethereum Classic is the chain that kept the original "
                   "history after Ethereum's July 2016 hard fork (the DAO fork); ETH is the forked "
                   "chain most of the ecosystem followed. Two coins on two networks, which still "
                   "move together much of the time.")
    if re.search(r"\b(?:XAUT|PAXG|XAU)\b", text) and re.search(
            r"\bper\s+(?:ounce|oz|gram|gramme|kilo)\b|\bounce\s+or\s+(?:a\s+)?gram\b|\bunit\b",
            text, re.I):
        # "Is XAUT priced per ounce or per gram?" got a price and no unit (round 45 hostile, m5)
        out.append("Gold is quoted per troy ounce (31.103 grams): XAUT and PAXG are each one fine "
                   "troy ounce of gold per token, and Bitget's XAU perpetual tracks the price of "
                   "one ounce.")
    twice = _SAME_TWICE.search(text)
    if twice is not None:
        name = twice.group(1)
        out.append(f"Both names asked about are {name}: a market compared with itself moves "
                   f"one for one, so there is no difference to show — this is {name} alone.")
    pair = _PAIR_FORM.search(text)
    if pair is not None:
        base = pair.group("a") or pair.group("b")
        quote = pair.group("q") or pair.group("r")
        out.append(f"{base} and {base}-{quote} are the same asset: {base}-{quote} is the "
                   f"{base}/{'US dollar' if quote == 'USD' else quote} pair as data sites and "
                   f"exchanges write it, and Bitget quotes {base} against USDT ({base}USDT), "
                   f"which tracks the dollar to within a fraction of a percent.")
    if _STOCK_WORD.search(text):
        from argus.lui.research.parse import is_us_equity, research_symbols
        from argus.market.company_names import names_for, sec_registered

        for symbol in research_symbols(text)[0]:
            if is_us_equity(symbol):
                continue
            base = symbol.removesuffix("USDT")
            if not re.search(rf"\b(?:stock|shares?|company|equity)\b[^?.]{{0,30}}\b{base}\b|"
                             rf"\b{base}\b[^?.]{{0,30}}\b(?:stock|shares?)\b", text, re.I):
                continue
            company = names_for(base)
            if company or sec_registered(base):
                out.append(f"{base} here is the crypto token Bitget lists ({symbol}); the "
                           f"US-listed stock with the ticker {base}"
                           + (f" ({company[0]})" if company else "")
                           + " is a different company, which Bitget does not list, so no figure "
                             "here is about it.")
            else:
                out.append(f"{base} here is the crypto token Bitget lists ({symbol}); no US-listed "
                           f"company trades under the ticker {base} in SEC's register, so there "
                           f"is no stock of that name to confuse it with.")
    return out


_SAME_UNITS: Final = re.compile(
    r"\b(?P<a>\d+(?:\.\d+)?)\s*(?P<s>[A-Za-z]{2,6})\s+(?:and|plus|\+|,)\s+(?:another\s+)?"
    r"(?P<b>\d+(?:\.\d+)?)\s*(?P=s)\b", re.I)


def same_units_lines(text: str) -> list[str] | None:
    """"I hold 3 ETH and 2 ETH, how much ETH do I have" was summed into a $13,581 book with "5 ETH"
    never said and the question unanswered (round 44 hostile, minor 10): two amounts of one coin
    asked as a total are added and valued."""
    m = _SAME_UNITS.search(text)
    if m is None or not re.search(r"\bhow\s+(?:much|many)\b|\btotal\b|\ball\s+together\b|"
                                  r"\bin\s+all\b|\bsum\b", text, re.I):
        return None
    from argus.lui.research.parse import last_price, research_symbols

    named = research_symbols(m.group("s").upper())[0]
    if not named:
        return None
    a, b = float(m.group("a")), float(m.group("b"))
    total = a + b
    base = named[0].removesuffix("USDT")
    try:
        px = float(last_price(named[0]) or 0)
    except Exception:
        px = 0.0
    lines = [f"Bottom line: {m.group('a')} + {m.group('b')} = {total:g} {base}"
             + (f", worth about ${total * px:,.0f} at Bitget's last price of {px:,.2f}." if px > 0
                else ".")]
    if px > 0:
        lines.append(f"Each part: {a:g} {base} is ${a * px:,.0f}; {b:g} {base} is ${b * px:,.0f}.")
    lines.append("Data: Bitget's live ticker; the value moves with the price.")
    return lines


__all__ = [
    "READERS",
    "bps_versus_percent",
    "crypto_equity_note",
    "false_premise",
    "fraction_or_percent",
    "in_currency",
    "lines",
    "minor_units",
    "name_notes",
    "per_period",
    "same_units_lines",
    "sp500_beta",
    "stated_fee",
    "stated_gas",
    "stop_percent",
    "wrong_side",
]
