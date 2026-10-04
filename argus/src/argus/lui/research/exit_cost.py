"""What it costs to get into a position, out of one, or both — walked on Bitget's live book.

"What does it cost to enter and exit a $250,000 ETH position on Bitget right now?" and "I hold
40 SOL. What would it cost me to close it out right now?" were both answered with a one-day loss on
a worked-example $10,000, the stated size ignored (round 35 build-list check, item 4.1). The order
book walk already existed for a sized round trip (`quote._sized_round_trip`); what was missing was a
reader that knows which leg is asked, takes the size in dollars or in units (or from the holding the
trader already stated), and walks the right side for each leg:

- entering a long takes the asks, exiting it hits the bids; a short is the other way round;
- every leg pays Bitget's taker fee from the contract list (``takerFeeRate``, read live);
- the walk is level by level over 200 visible levels (`market/depth.OrderBook.sweep`), against the
  mid, and cross-checked: the walked average price is set beside the ticker's last;
- a size the visible book cannot hold is priced on the square-root impact law the execution plan
  uses (`cost/model.CostModel.impact_bps`, on the day's volume and 30-day volatility), and said to
  be that estimate rather than a walk.

Reference read before building (EGRESS, the exit-liquidity idea named in the build list): the cost
of leaving is the bid side at size, not the quoted spread; that is the whole point of the walk.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Final

from argus.lui.trace import trace_module

ASKED: Final = re.compile(
    r"\b(?:cost|costs|costing|slippage|fees?|price\s+impact|impact|lose\s+to\s+(?:fees|slippage)|"
    r"pay)\b[^?]{0,80}\b(?:enter\s+and\s+exit|get\s+in\s+and\s+(?:get\s+)?out|in\s+and\s+out|"
    r"round[\s-]?trip|close\s+(?:it|them|this|that|my\s+\w+|the\s+\w+|everything|all)?\s*out|"
    r"close\s+(?:it|them|my\s+\w+(?:\s+position)?)\b|exit(?:ing)?|unwind\w*|get\s+out|sell\s+"
    r"(?:it|them|all|everything|my\s+\w+)|liquidat(?:e|ing)\s+my|dump(?:ing)?\s+(?:it|my))|"
    r"\b(?:enter\s+and\s+exit|close\s+(?:it|them|my\s+\w+)?\s*out|exit(?:ing)?|unwind\w*|"
    # "go in and out of $2M ETH today, what's the round trip cost" (a live re-ask, round 35)
    r"get\s+out\s+of|(?:go|get)\s+in\s+and\s+out|in\s+and\s+out\s+of|round[\s-]?trip)\b[^?]{0,80}"
    r"\b(?:cost|costs|slippage|fees?|impact)\b", re.I)
_BOTH: Final = re.compile(r"\benter\s+and\s+exit|\bin\s+and\s+out\b|\bround[\s-]?trip|\bget\s+in\s+"
                          r"and\b|\bbuy\s+and\s+(?:then\s+)?sell\b|\bopen\s+and\s+close\b", re.I)
_ENTRY_ONLY: Final = re.compile(r"\b(?:enter|entering|get\s+in(?:to)?|open(?:ing)?|buy(?:ing)?)\b"
                                r"(?![^?]{0,40}\b(?:exit|out|close|sell)\b)", re.I)
_DOLLARS: Final = re.compile(r"\$\s?(?P<a>\d[\d,]*(?:\.\d+)?)\s*(?P<u>k|m|mm|million)?\b", re.I)
_UNITS: Final = re.compile(r"(?<![\w.$,])(?P<n>\d[\d,]*(?:\.\d+)?)\s+(?P<name>[A-Za-z]{2,10})\b")


def _size(text: str, book: str, symbol: str) -> tuple[Decimal, str] | None:
    """The size asked about, in dollars, and how it was read; None when no size can be found."""
    dollars = _DOLLARS.search(text)
    if dollars is not None:
        unit = (dollars.group("u") or "").lower()
        amount = Decimal(dollars.group("a").replace(",", "")) * (
            1000 if unit == "k" else 1_000_000 if unit in ("m", "mm", "million") else 1)
        return amount, f"${amount:,.0f} as stated"
    from argus.lui.research import research_symbols
    from argus.lui.research.parse import last_price

    base = symbol.removesuffix("USDT")
    for source, words in ((text, "as stated"), (book, "from the holding you gave")):
        for m in _UNITS.finditer(source or ""):
            named = research_symbols(m.group("name"))[0]
            if named and named[0] == symbol:
                units = Decimal(m.group("n").replace(",", ""))
                try:
                    price = Decimal(str(last_price(symbol)))
                except Exception:
                    return None
                return units * price, (f"{units:g} {base} {words}, at Bitget's last "
                                       f"{price:,.2f} = ${units * price:,.0f}")
    return None


def _px(value: Decimal) -> str:
    return f"{value:,.2f}" if value >= 100 else f"{value:,.4g}"


def lines(text: str, book: str = "") -> list[str] | None:
    """The cost of the leg or legs asked, for the size asked; None when the question is not one."""
    if not ASKED.search(text):
        return None
    from argus.lui.research import research_symbols

    named = research_symbols(text)[0] or research_symbols(book)[0]
    if not named:
        return None
    symbol = named[0]
    sized = _size(text, book, symbol)
    if sized is None:
        return None
    notional, how = sized
    short = re.search(r"\bshort\b", text, re.I) is not None
    both = _BOTH.search(text) is not None
    entry_only = not both and _ENTRY_ONLY.search(text) is not None and not re.search(
        r"\bexit|\bclose|\bout\b|\bsell|\bunwind|\bliquidat|\bdump", text, re.I)
    legs = ([("entry", "SELL" if short else "BUY"), ("exit", "BUY" if short else "SELL")] if both
            else [("entry", "SELL" if short else "BUY")] if entry_only
            else [("exit", "BUY" if short else "SELL")])
    from argus.market.crossasset_feed import fetch_taker_bps
    from argus.market.depth import fetch_orderbook

    try:
        fee_bps = Decimal(str(fetch_taker_bps(symbol)))
        order_book = fetch_orderbook(symbol, limit=200)
    except Exception:
        return None
    name = symbol.removesuffix("USDT")
    parts: list[str] = []
    total_bps = Decimal("0")
    beyond = False
    for leg, direction in legs:
        walked = order_book.sweep(notional, direction=direction)
        if not walked.complete:
            beyond = True
            break
        side = "asks" if direction == "BUY" else "bids"
        leg_bps = walked.slippage_bps + fee_bps
        total_bps += leg_bps
        parts.append(f"{leg} ({'buying' if direction == 'BUY' else 'selling'} into the {side}): "
                     f"{walked.slippage_bps:.1f}bps of slippage over {walked.levels_consumed} "
                     f"level{'s' if walked.levels_consumed != 1 else ''}, average price "
                     f"{_px(walked.average_price)} against a mid of {_px(order_book.mid)}, plus "
                     f"{fee_bps:.0f}bps taker fee = {leg_bps:.1f}bps "
                     f"(${notional * leg_bps / 10_000:,.0f})")
    what = (f"a round trip in ${notional:,.0f} of {name}" if both else
            f"opening ${notional:,.0f} of {name}" + (" short" if short else "") if entry_only else
            f"closing a ${notional:,.0f} {name} short" if short else
            f"closing ${notional:,.0f} of {name}")
    if beyond:
        from argus.cost.model import CostModel
        from argus.lui.research.execution import _daily_volatility_bps
        from argus.market.bitget import fetch_tickers

        try:
            ticker = fetch_tickers()[symbol]
            volume = ticker.base_volume * ticker.last
            if notional > volume:
                # "$900,000,000,000 long BTC" was priced at 4,017bps by the impact law, a curve
                # calibrated on orders a small share of a day's volume (a hostile review, round 36)
                return [f"Bottom line: it cannot be done as one trade — ${notional:,.0f} is "
                        f"{notional / volume:,.0f} times {name}'s whole last 24 hours of volume on "
                        f"Bitget (${volume:,.0f}), so there is no market at that size to price.",
                        "The square-root impact law the console uses is calibrated on orders that "
                        "are a small share of a day's volume; this far past it, any number it gave "
                        "would be invented, so none is given.",
                        f"Size read: {how}. For a size the market can absorb, ask the cost of a "
                        f"round trip in a figure below the day's volume."]
            leg_impact = CostModel.bitget_perp().impact_bps(notional / volume,
                                                            _daily_volatility_bps(symbol))
        except Exception:
            return [f"Bottom line: ${notional:,.0f} of {name} is more than Bitget's 200 visible "
                    f"levels hold on the side it would take, and the impact estimate beyond the "
                    f"book did not load — no cost is given rather than an understated one."]
        per_leg = leg_impact + fee_bps
        total = per_leg * len(legs)
        return [f"Bottom line: {what} costs about {total:.1f}bps "
                f"(${notional * total / 10_000:,.0f}) — the size is more than the 200 visible "
                f"levels hold, so each leg is the square-root impact law's estimate "
                f"({leg_impact:.1f}bps at {notional / volume:.1%} of the day's volume) plus "
                f"{fee_bps:.0f}bps taker fee.",
                f"Size read: {how}.",
                "Splitting the order over time is what brings this down — ask how to split it "
                "for the execution plan."]
    lead = (f"Bottom line: {what} costs about {total_bps:.1f}bps "
            f"(${notional * total_bps / 10_000:,.0f}) on Bitget's book right now"
            + (" — both legs, each walked on its own side" if both else "") + ".")
    out = [lead, *[p[0].upper() + p[1:] + "." for p in parts], f"Size read: {how}."]
    try:
        from argus.lui.research.parse import last_price

        last = Decimal(str(last_price(symbol)))
        checked = (f"; the mid sits {abs(order_book.mid / last - 1) * 10_000:.1f}bps from the "
                   f"ticker's last {_px(last)}")
    except Exception:
        checked = ""
    out.append(f"Book: {name} perpetual, 200 levels a side read just now (touch spread "
               f"{order_book.spread_bps:.1f}bps{checked}); fee from Bitget's contract list. A "
               f"limit order "
               f"can pay less if it fills, but a resting order is filled mostly when the price "
               f"moves against it.")
    return out


trace_module(globals())
