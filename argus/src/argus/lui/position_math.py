"""Arithmetic on a position the trader describes: what it costs to hold, its average entry, its
return on margin, its unrealised P&L, and a sum in another currency as a share and a count.

A hostile review (round 31) asked five of these and got a different answer each time:

* "Long 1 BTC at 10x, round-trip taker 0.06%, funding -0.01% against me every 8 hours, hold 5 days
  — total cost?" read the "10" of "10x" as the entry price and answered -$0.007; the real cost at
  ~$84,750 is about $178 (fees $50.85 plus 15 settlements of funding, $127).
* "Bought 1 BTC at 90000, 2 at 80000, 1 at 70000 — average entry?" was answered with a risk report
  on 4 BTC; the answer is $80,000.
* "Shorted 1 ETH at 3000, covered at 2700, 10x — percentage return on margin?" got +$300 and no
  percentage; on $300 of margin that is +100%.
* "My entry was 95000, not 85000 — unrealized P&L at today's price?" got a liquidation price.
* "Allocate EUR 20,000 of a $100,000 book to BTC — what percentage, how many BTC?" got a risk
  report and a request to restate the amount.

Each is computed here from the trader's own figures and Bitget's live last price where "today's
price" is asked, with the arithmetic shown. Nothing is assumed silently: a currency is converted
at a named rate, and a missing figure is asked for by name.
"""

from __future__ import annotations

import re
from typing import Final

from argus.lui.trace import trace_module

_NUM = r"\d(?:[\d,]*\d)?(?:\.\d+)?"
_SYM = r"[A-Za-z]{2,10}"


def _f(raw: str) -> float:
    return float(raw.replace(",", ""))


def _k(raw: str, unit: str | None) -> float:
    return _f(raw) * (1000 if (unit or "").lower() == "k" else 1)


def _last(symbol: str) -> float | None:
    from argus.market.bitget import fetch_tickers

    try:
        ticker = fetch_tickers().get(symbol)
    except Exception:
        return None
    return float(ticker.last) if ticker is not None else None


def _symbol(word: str) -> str | None:
    from argus.lui.research import research_symbols

    found = research_symbols(word)[0]
    return found[0] if found else None


# --- the cost of holding --------------------------------------------------------------------------

_HOLD_DAYS: Final = re.compile(r"\bhold(?:ing)?\s+(?:it\s+)?(?:for\s+)?(?:exactly\s+|about\s+)?"
                               r"(?P<n>\d+(?:\.\d+)?)\s*(?P<u>days?|hours?|weeks?)\b", re.I)
_FUNDING: Final = re.compile(
    r"\bfunding\b[^.?]{0,40}?(?P<sign>[-+])?\s?(?P<r>\d+(?:\.\d+)?)\s*%[^.?]{0,40}?\bevery\s+"
    r"(?P<h>\d+)\s*(?:h|hours?)\b", re.I)
_FEE: Final = re.compile(r"\b(?P<rt>round[\s-]?trip\s+)?(?:taker\s+|maker\s+)?fees?\b[^.?]{0,20}?"
                         r"(?P<r>\d+(?:\.\d+)?)\s*%|(?P<r2>\d+(?:\.\d+)?)\s*%\s+(?P<rt2>round[\s-]?"
                         r"trip\s+)?(?:taker\s+|maker\s+)?fees?\b", re.I)
_QTY: Final = re.compile(rf"\b(?:long|short|holding|hold|own)\s+(?P<q>{_NUM})\s+(?P<s>{_SYM})\b",
                         re.I)


def holding_cost_lines(text: str) -> list[str] | None:
    """Fees and funding over a stated holding period, at today's price."""
    days_m, funding_m = _HOLD_DAYS.search(text), _FUNDING.search(text)
    qty_m = _QTY.search(text)
    if days_m is None or funding_m is None or qty_m is None:
        return None
    symbol = _symbol(qty_m.group("s"))
    if symbol is None:
        return None
    price = _last(symbol)
    if price is None:
        return None
    qty = _f(qty_m.group("q"))
    unit = days_m.group("u").lower()
    hours = _f(days_m.group("n")) * (24 if unit.startswith("d") else 168 if unit.startswith("w")
                                     else 1)
    every = int(funding_m.group("h"))
    settlements = int(hours // every)
    rate = _f(funding_m.group("r")) / 100
    against = bool(re.search(r"\bagainst\s+me\b|\bi\s+pay\b|\bcosts?\s+me\b", text, re.I)) or (
        funding_m.group("sign") == "-" and re.search(r"\bshort\b", qty_m.group(0), re.I) is None)
    notional = qty * price
    fee_m = _FEE.search(text)
    fee_rate = (_f(fee_m.group("r") or fee_m.group("r2")) / 100) if fee_m else 0.0006
    round_trip = bool(fee_m and (fee_m.group("rt") or fee_m.group("rt2")))
    fees = notional * fee_rate * (1 if round_trip else 2)
    funding = notional * rate * settlements * (1 if against else -1)
    name = symbol.removesuffix("USDT")
    total = fees + funding
    return [f"Bottom line: about ${total:,.2f} in all over {hours:g} hours — ${fees:,.2f} in fees "
            f"and ${funding:,.2f} in funding, ignoring any price move.",
            f"The math: {qty:g} {name} at Bitget's last {price:,.2f} is ${notional:,.2f}. Fees: "
            + (f"{fee_rate:.3%} round trip x ${notional:,.2f} = ${fees:,.2f}"
               if round_trip else f"{fee_rate:.3%} a side, twice, x ${notional:,.2f} = ${fees:,.2f}"
               + ("" if fee_m else " (no fee was stated, so Bitget's 0.06% taker rate is used)"))
            + f". Funding: {settlements} settlements (one every {every}h over {hours:g}h) x "
            f"{rate:.3%} x ${notional:,.2f} = ${abs(funding):,.2f} "
            + ("paid." if funding > 0 else "received."),
            "Leverage does not change these dollars — they are charged on the position, not the "
            "margin; it changes how large they are against the margin you put up.",
            "Funding is reset every settlement, so the rate you stated is held flat here; the real "
            "one moves."]


# --- average entry --------------------------------------------------------------------------------

_BUY: Final = re.compile(rf"\b(?:bought|buy|added|add|got|purchased)\s+(?:another\s+)?(?P<q>{_NUM})"
                         rf"\s*(?:more\s+)?(?:(?P<s>{_SYM})\s+)?(?:at|@|for)\s+\$?(?P<p>{_NUM})"
                         r"\s*(?P<k>k)?\b|\bthen\s+(?P<q2>" + _NUM + r")\s*(?:(?P<s2>" + _SYM
                         + r")\s+)?(?:at|@)\s+\$?(?P<p2>" + _NUM + r")\s*(?P<k2>k)?\b", re.I)
AVERAGE_ASKED: Final = re.compile(
    r"\baverage\s+(?:entry|price|cost|buy\s+price)\b|\bbreak[\s-]?even\s+price\b|\bcost\s+"
    r"basis\b|\bdca\b|\b(?:my|the)\s+average\b", re.I)
_SAME_SIZE: Final = re.compile(r"\b(?:same|equal)\s+(?:size|amount|quantity|dollars?|money)\b|"
                               r"\b(?:the\s+)?same\s+each\b", re.I)
"""Buys of equal size with no quantity: "I bought ETH at 2400 and again at 2000, same size each —
where's my average and what's my P&L now?" got a risk table (a live re-ask, round 31)."""
_AT_PRICE: Final = re.compile(rf"(?:\band\s+(?:again\s+)?(?:at\s+|@\s*)?|\bat|@)\s*\$?(?P<p>{_NUM})"
                              r"\s*(?P<k>k)?\b", re.I)
_BOUGHT_NAME: Final = re.compile(rf"\b(?:bought|buy|got|added|purchased|entered)\s+(?P<s>{_SYM})"
                                 r"\s+(?:at|@)", re.I)


def average_entry_lines(text: str) -> list[str] | None:
    """The quantity-weighted average of several buys the trader lists."""
    if not AVERAGE_ASKED.search(text):
        return None
    fills = []
    for m in _BUY.finditer(text):
        q = _f(m.group("q") or m.group("q2"))
        p = _k(m.group("p") or m.group("p2"), m.group("k") or m.group("k2"))
        if q > 0 and p > 0:
            fills.append((q, p))
    if len(fills) < 2:
        return _equal_size_lines(text)
    total_q = sum(q for q, _ in fills)
    cost = sum(q * p for q, p in fills)
    average = cost / total_q
    sym = next((s for m in _BUY.finditer(text) if (s := m.group("s") or m.group("s2"))), None)
    symbol = _symbol(sym) if sym else None
    name = symbol.removesuffix("USDT") if symbol else "units"
    lines = [f"Bottom line: your average entry is {average:,.2f} across {total_q:g} {name}.",
             "The math: (" + " + ".join(f"{q:g} x {p:,.0f}" for q, p in fills)
             + f") / {total_q:g} = {cost:,.0f} / {total_q:g} = {average:,.2f}."]
    if symbol is not None and (price := _last(symbol)) is not None:
        pnl = (price - average) * total_q
        lines.append(f"At Bitget's last {price:,.2f} that is {price / average - 1:+.1%} on the "
                     f"average, {'+' if pnl >= 0 else '-'}${abs(pnl):,.0f} on the {total_q:g} "
                     f"{name}, before fees.")
    return lines


def _equal_size_lines(text: str) -> list[str] | None:
    """Buys of the same size, said by price alone. Same quantity each gives the plain mean of the
    prices; the same dollars each gives their harmonic mean (more units bought where cheaper), and
    the answer says which it used."""
    if not _SAME_SIZE.search(text):
        return None
    prices = [_k(m.group("p"), m.group("k")) for m in _AT_PRICE.finditer(text)]
    prices = [x for x in prices if x > 0]
    if len(prices) < 2:
        return None
    dollars = bool(re.search(r"\b(?:same|equal)\s+(?:dollars?|money|amount\s+of\s+money)\b|\$\s?\d"
                             r"[\d,]*\s+each\b", text, re.I))
    if dollars:
        average = len(prices) / sum(1 / x for x in prices)
        how = (f"the same dollars each buys more where the price is lower, so the average is the "
               f"harmonic mean: {len(prices)} / (" + " + ".join(f"1/{x:,.0f}" for x in prices)
               + f") = {average:,.2f}")
    else:
        average = sum(prices) / len(prices)
        how = ("the same quantity each, so the average is the plain mean: ("
               + " + ".join(f"{x:,.0f}" for x in prices) + f") / {len(prices)} = {average:,.2f}")
    named = _BOUGHT_NAME.search(text)
    symbol = _symbol(named.group("s")) if named else None
    name = symbol.removesuffix("USDT") if symbol else None
    lines = [f"Bottom line: your average entry is {average:,.2f}"
             + (f" on {name}" if name else "") + ".",
             f"The math: {how}." + ("" if dollars else " If each buy was the same dollar amount "
                                    "instead, say so: the average is then a little lower.")]
    if symbol is not None and (price := _last(symbol)) is not None:
        lines.append(f"At Bitget's last {price:,.2f} that is {price / average - 1:+.1%} on the "
                     f"average, before fees; with no size given, that percentage is the P&L on "
                     f"the whole position.")
    return lines


# --- return on margin -----------------------------------------------------------------------------

_ROUND_TRIP: Final = re.compile(
    rf"\b(?P<side>shorted|sold\s+short|short(?:ed)?|bought|went\s+long|long(?:ed)?)\s+(?P<q>{_NUM})"
    rf"\s+(?P<s>{_SYM})\s+at\s+\$?(?P<a>{_NUM})\s*(?P<ka>k)?\b[^.?]{{0,40}}?\b(?:covered|closed|"
    rf"sold|exited|bought\s+back)\s+(?:it\s+|them\s+)?at\s+\$?(?P<b>{_NUM})\s*(?P<kb>k)?\b", re.I)
_LEVERAGE: Final = re.compile(r"\b(?P<x>\d+(?:\.\d+)?)\s*x\b", re.I)
MARGIN_RETURN_ASKED: Final = re.compile(r"\b(?:return|gain|loss|profit|pnl|p&l)\s+(?:on|of)\s+"
                                        r"(?:my\s+)?margin\b|\bpercent(?:age)?\s+return\b|\broe\b|"
                                        r"\breturn\s+in\s+%|\bwhat\s+%", re.I)


def margin_return_lines(text: str) -> list[str] | None:
    """A closed trade's P&L as a share of the margin its leverage required."""
    if not MARGIN_RETURN_ASKED.search(text):
        return None
    trade, lev = _ROUND_TRIP.search(text), _LEVERAGE.search(text)
    if trade is None:
        return None
    short = trade.group("side").lower().startswith(("short", "sold"))
    qty = _f(trade.group("q"))
    entry, exit_ = (_k(trade.group("a"), trade.group("ka")), _k(trade.group("b"),
                                                                 trade.group("kb")))
    pnl = (entry - exit_) * qty if short else (exit_ - entry) * qty
    move = pnl / (entry * qty)
    name = trade.group("s").upper()
    lines = []
    if lev is not None:
        x = _f(lev.group("x"))
        margin = entry * qty / x
        lines.append(f"Bottom line: {pnl / margin:+.0%} on your margin, before fees — "
                     f"{'+' if pnl >= 0 else '-'}${abs(pnl):,.2f} on ${margin:,.2f}.")
        lines.append(f"The math: the {'short' if short else 'long'} made ({entry:,.0f} - "
                     f"{exit_:,.0f}) x {qty:g} = ${pnl:,.2f}"
                     if short else f"The math: ({exit_:,.0f} - {entry:,.0f}) x {qty:g} = "
                                   f"${pnl:,.2f}")
        lines[-1] += (f"; margin at {x:g}x is {entry:,.0f} x {qty:g} / {x:g} = ${margin:,.2f}; "
                      f"${pnl:,.2f} / ${margin:,.2f} = {pnl / margin:+.0%} — {x:g} times the "
                      f"{move:+.1%} move in {name}.")
    else:
        lines.append(f"Bottom line: {move:+.1%} on the position, {'+' if pnl >= 0 else '-'}"
                     f"${abs(pnl):,.2f}; say the leverage for the return on margin.")
    lines.append("Fees and funding are not in this figure: at Bitget's 0.06% taker rate the round "
                 f"trip costs about ${entry * qty * 0.0012:,.2f}.")
    return lines


# --- unrealised P&L -------------------------------------------------------------------------------

_ENTRY: Final = re.compile(rf"\bentry\s+(?:price\s+)?(?:is|was|of|at|=)?\s*\$?(?P<p>{_NUM})\s*"
                           rf"(?P<k>k)?\b(?!\s*(?:x\b|%))", re.I)
_HOLDING: Final = re.compile(rf"\b(?P<side>long|short)\s+(?P<q>{_NUM})\s+(?P<s>{_SYM})\b", re.I)
UNREALISED_ASKED: Final = re.compile(
    r"\bunreali[sz]ed\b|\bpaper\s+(?:loss|gain|p&?l)\b|\bwhat(?:'s|\s+is)\s+my\s+(?:p&?l|pnl|"
    r"profit|loss)\b", re.I)


def unrealised_lines(text: str, prior: list[str]) -> list[str] | None:
    """The open position's P&L at Bitget's last price, from the latest entry the trader gave."""
    if not UNREALISED_ASKED.search(text):
        return None
    said = [*prior[-6:], text]
    entry = None
    for chunk in said:
        for m in _ENTRY.finditer(chunk):
            entry = _k(m.group("p"), m.group("k"))
    corrected = re.search(rf"\bentry\s+was\s+\$?(?P<n>{_NUM})\s*(?P<k>k)?\b[^.?]{{0,20}}\bnot\s+"
                          rf"\$?(?P<o>{_NUM})", text, re.I)
    if corrected is not None:
        entry = _k(corrected.group("n"), corrected.group("k"))
    holding = next((h for chunk in reversed(said) if (h := _HOLDING.search(chunk))), None)
    if entry is None or holding is None:
        return None
    symbol = _symbol(holding.group("s"))
    price = _last(symbol) if symbol else None
    if symbol is None or price is None:
        return None
    qty = _f(holding.group("q"))
    short = holding.group("side").lower() == "short"
    pnl = (entry - price) * qty if short else (price - entry) * qty
    name = symbol.removesuffix("USDT")
    diff = f"{entry:,.0f} - {price:,.2f}" if short else f"{price:,.2f} - {entry:,.0f}"
    lines = [f"Bottom line: {'+' if pnl >= 0 else '-'}${abs(pnl):,.2f} unrealised on your "
             f"{qty:g} {name} {'short' if short else 'long'} from {entry:,.0f}, at Bitget's last "
             f"{price:,.2f}.",
             f"The math: ({diff}) x {qty:g} = {pnl:+,.2f}, which is "
             f"{pnl / (entry * qty):+.1%} of the position."]
    lev = _LEVERAGE.search(" ".join(said))
    if lev is not None:
        x = _f(lev.group("x"))
        margin = entry * qty / x
        lines.append(f"At {x:g}x your margin was about ${margin:,.0f}, so that is "
                     f"{pnl / margin:+.0%} of it" + (" — past the whole margin, which a "
                                                     "liquidation would already have closed."
                                                     if pnl <= -margin else "."))
    if corrected is not None:
        lines.append(f"Used {entry:,.0f}, the entry you corrected to, not "
                     f"{_f(corrected.group('o')):,.0f}.")
    return lines


# --- a sum in another currency --------------------------------------------------------------------

_ALLOCATE: Final = re.compile(
    rf"\b(?:allocate|put|invest|move)\s+(?P<cur>eur|euros?|€|gbp|£|pounds?|jpy|¥|inr|₹|usd|\$)?\s*"
    rf"(?P<a>{_NUM})\s*(?P<ka>k)?\s*(?P<cur2>eur|euros?|gbp|pounds?|jpy|yen|inr|rupees?)?"
    rf"[^.?]{{0,30}}?\bof\s+(?:a|my|the)\s+\$\s?(?P<b>{_NUM})\s*(?P<kb>k)?\s+(?:book|portfolio)"
    rf"[^.?]{{0,20}}?\bto\s+(?P<s>{_SYM})\b", re.I)
_CODES: Final = {"eur": "EUR", "euro": "EUR", "euros": "EUR", "€": "EUR", "gbp": "GBP", "£": "GBP",
                 "pound": "GBP", "pounds": "GBP", "jpy": "JPY", "¥": "JPY", "yen": "JPY",
                 "inr": "INR", "₹": "INR", "rupee": "INR", "rupees": "INR"}


def allocation_lines(text: str) -> list[str] | None:
    """A sum, possibly in another currency, as a share of a dollar book and a count of coins."""
    m = _ALLOCATE.search(text)
    if m is None:
        return None
    symbol = _symbol(m.group("s"))
    price = _last(symbol) if symbol else None
    if symbol is None or price is None:
        return None
    amount = _k(m.group("a"), m.group("ka"))
    book = _k(m.group("b"), m.group("kb"))
    code = _CODES.get((m.group("cur") or m.group("cur2") or "").lower())
    rate_said = ""
    dollars = amount
    if code is not None:
        from argus.market.equity_history import HistoryError, daily

        try:
            per_dollar = float(daily(f"{code}=X")[-1].close)
        except (HistoryError, OSError, ValueError, IndexError):
            return [f"Bottom line: {code} {amount:,.0f} has to be turned into dollars first, and "
                    f"the {code}/USD rate could not be read just now — ask again in a minute, or "
                    "give the amount in dollars."]
        dollars = amount / per_dollar
        rate_said = (f" ({code} {amount:,.0f} at {per_dollar:.4f} {code} per dollar, Yahoo "
                     f"Finance's latest daily close, is ${dollars:,.0f})")
    name = symbol.removesuffix("USDT")
    return [f"Bottom line: that is {dollars / book:.1%} of your ${book:,.0f} book and about "
            f"{dollars / price:,.4f} {name} at Bitget's last {price:,.2f}{rate_said}.",
            f"The math: ${dollars:,.0f} / ${book:,.0f} = {dollars / book:.1%}; ${dollars:,.0f} / "
            f"{price:,.2f} = {dollars / price:,.4f} {name}, before fees."]


def lines(text: str, prior: list[str]) -> list[str] | None:
    for rule in (holding_cost_lines, average_entry_lines, margin_return_lines, allocation_lines):
        found = rule(text)
        if found is not None:
            return found
    return unrealised_lines(text, prior)


__all__ = ["AVERAGE_ASKED", "MARGIN_RETURN_ASKED", "UNREALISED_ASKED", "allocation_lines",
           "average_entry_lines", "holding_cost_lines", "lines", "margin_return_lines",
           "unrealised_lines"]

trace_module(globals())
