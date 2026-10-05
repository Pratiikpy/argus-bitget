"""Numbers a trader states about a position, checked against the market before anything is built
on them: a share price written in another currency, a stock split applied to a holding, and a
protection level no hedge can give.

Round 42's hostile audit found three questions answered as if the number were not there:

* "NVDA price in euros 2.000,50 EUR per share, 100 shares worth how many USD?" got NVDA's ticker:
  the false price (NVDA is about $238, roughly 212 euros) went unchallenged and no conversion was
  given (minor 5).
* "What would a 10-for-1 NVDA split do to my 100 shares" got the cost of a $238 market order: the
  split was ignored (minor 6).
* "Hedge my 100k book with 250% put protection" was refused as an order, and with "100000 USD"
  the 250% was filed as memory (minor 9). A put pays at most what the asset loses, so cover past
  100% is not protection: the excess is a bet that the market falls.

Every figure is the live one: Bitget's last price for the stock, Bitget's own FX perpetuals for the
currency (`parse.in_us_dollars`), and for the hedge the Cboe chain read by `equity_options`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Final

_PRICE_WRITTEN: Final = re.compile(
    r"(?:(?P<pre>[€£¥$])\s?(?P<a>\d[\d,]*(?:\.\d+)?)|(?P<b>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<code>eur|euros?|gbp|pounds?|jpy|yen|usd|usdt|dollars?|[€£¥]))\s*"
    r"(?:per|a|each|/)\s*share\b", re.I)
_SHARES: Final = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(?:(?-i:[A-Z]{1,5})\s+)?shares?\b",
                            re.I)
_WORTH_ASKED: Final = re.compile(r"\bworth\b|\bvalue\b|\bhow\s+many\s+(?:usd|dollars?)\b|\bin\s+"
                                 r"(?:usd|dollars?)\b", re.I)
_SPLIT: Final = re.compile(r"\b(?P<n>\d{1,3})\s*[-\s]?(?:for|:|-to-|to)[-\s]?(?P<m>\d{1,3})\s+"
                           r"(?:[A-Z]{1,5}\s+)?(?:(?:stock|share|reverse)\s+)*split\b|\bsplit\s+(?P<n2>\d{1,3})\s*"
                           r"(?:for|:|-to-)\s*(?P<m2>\d{1,3})\b", re.I)
_PROTECTION: Final = re.compile(
    r"\b(?P<p>\d{1,4}(?:\.\d+)?)\s*%\s*(?:put\s+)?(?:protection|hedge[ds]?|cover(?:age)?|"
    r"insurance)\b|\b(?:protect|hedge|insure|cover)\w*\s+(?:it\s+|my\s+\w+\s+)?(?:at|by|to|for)\s+"
    r"(?P<p2>\d{1,4}(?:\.\d+)?)\s*%", re.I)
_BOOK_SIZE: Final = re.compile(
    r"\$?(?P<n>\d[\d,]*(?:\.\d+)?)\s*(?P<u>k|m|mn|thousand|million)?\s*(?:usd|dollars?)?\s+"
    r"(?:book|portfolio|account|position)\b", re.I)
_CODES: Final = {"€": "EUR", "£": "GBP", "¥": "JPY", "$": "USD", "eur": "EUR",
                 "euro": "EUR", "euros": "EUR", "gbp": "GBP", "pound": "GBP", "pounds": "GBP",
                 "jpy": "JPY", "yen": "JPY", "usd": "USD", "usdt": "USD", "dollar": "USD",
                 "dollars": "USD"}
SHARES_PER_CONTRACT: Final = 100
PROTECTION_DAYS: Final = 30
"""The horizon a protective put is priced at when the question names none."""


def _stock(text: str, prior: Sequence[str] = ()) -> str | None:
    from argus.lui.research.parse import is_us_equity, research_symbols

    for said in [text, *reversed(list(prior)[-3:])]:
        stocks = [s for s in research_symbols(said)[0] if is_us_equity(s)]
        if stocks:
            return stocks[0]
    return None


def _usd(amount: float, code: str) -> tuple[float, float] | None:
    """(dollars, the rate used) for ``amount`` of ``code``, at Bitget's own FX perpetual."""
    from argus.lui.research.parse import in_us_dollars

    if code == "USD":
        return amount, 1.0
    restated, said = in_us_dollars(f"{amount:.2f} {code}")
    m = re.search(r"\$(\d[\d,]*(?:\.\d+)?)", restated)
    if m is None or not said or "taken as dollars" in said[0]:
        return None
    usd = float(m.group(1).replace(",", ""))
    return usd, usd / amount if amount else 0.0


def share_price_lines(text: str) -> list[str] | None:
    """A share price written in the question, checked against the live one, and what the shares
    are worth in dollars at each."""
    from argus.lui.research.parse import last_price

    m = _PRICE_WRITTEN.search(text)
    if m is None or not _WORTH_ASKED.search(text):
        return None
    symbol = _stock(text)
    if symbol is None:
        return None
    code = _CODES.get((m.group("pre") or m.group("code") or "").lower(), "USD")
    written = float((m.group("a") or m.group("b")).replace(",", ""))
    live = last_price(symbol)
    if live is None or written <= 0:
        return None
    converted = _usd(written, code)
    if converted is None:
        return None
    written_usd, rate = converted
    ticker = symbol.removesuffix("USDT")
    count_m = _SHARES.search(text[:m.start()] + " " + text[m.end():])
    count = float(count_m.group(1).replace(",", "")) if count_m else 1.0
    local = (f" (about {live / rate:,.2f} {code} a share at Bitget's {code}USD "
             f"{rate:,.4f})" if code != "USD" and rate else "")
    out = [f"Bottom line: {count:,.10g} {ticker} {'share is' if count == 1 else 'shares are'} "
           f"worth ${count * live:,.2f} at Bitget's last {live:,.2f}{local}."]
    gap = written_usd / live - 1
    if abs(gap) > 0.05:
        out.append(f"The {written:,.2f} {code} a share written is not {ticker}'s price: it is "
                   f"about ${written_usd:,.2f}, {written_usd / live:,.1f}x the live price "
                   f"({gap:+.0%}). At that price {count:,.10g} "
                   f"{'share' if count == 1 else 'shares'} would be "
                   f"{count * written:,.2f} {code} (about ${count * written_usd:,.0f}) — check "
                   "the figure, or the currency.")
    else:
        out.append(f"The {written:,.2f} {code} written matches the live price within 5% "
                   f"(about ${written_usd:,.2f}).")
    out.append("Data: Bitget's last price for the stock's perpetual and Bitget's FX perpetual "
               "for the currency, read just now. Not advice.")
    return out


def split_lines(text: str, prior: Sequence[str] = ()) -> list[str] | None:
    """What an N-for-M split does to a stated holding: the count scales by N/M, the price by M/N,
    and the value does not move."""
    from argus.lui.research.parse import last_price

    m = _SPLIT.search(text)
    if m is None:
        return None
    n = int(m.group("n") or m.group("n2"))
    d = int(m.group("m") or m.group("m2"))
    if n <= 0 or d <= 0 or n == d:
        return None
    symbol = _stock(text, prior)
    if symbol is None:
        return None
    ticker = symbol.removesuffix("USDT")
    live = last_price(symbol)
    shares_m = _SHARES.search(text)
    shares = float(shares_m.group(1).replace(",", "")) if shares_m else None
    reverse = n < d or bool(re.search(r"\breverse\b", text, re.I))
    if reverse and n > d:
        n, d = d, n
    after = None if shares is None else shares * n / d
    out: list[str] = []
    if shares is not None and after is not None:
        lead = (f"a {n}-for-{d} {'reverse ' if reverse else ''}split turns your {shares:,.10g} "
                f"{ticker} shares into {after:,.10g}")
        if live is not None:
            lead += (f", each priced at about {live * d / n:,.2f} instead of {live:,.2f}, so the "
                     f"holding stays worth ${shares * live:,.2f} — a split changes the count and "
                     "the price, not the value")
        out.append(lead + ".")
        if after != math.floor(after):
            out.append(f"{after:,.10g} is not a whole number of shares: brokers pay the fraction "
                       "out in cash at the post-split price.")
    else:
        out.append(f"a {n}-for-{d} split multiplies every {ticker} holding's share count by "
                   f"{n / d:g} and divides the price by the same, so no holding's value changes"
                   + (f" ({live:,.2f} becomes about {live * d / n:,.2f})" if live else "") + ".")
    out.append("What does change: listed options are adjusted to the new count and strike, "
               "per-share figures (EPS, dividends) are restated, and a lower price can widen "
               "who buys it; none of that is value.")
    out[0] = "Bottom line: " + out[0]
    out.append(f"Data: Bitget's last price for {ticker}'s perpetual, read just now. Not advice.")
    return out


def protection_lines(text: str, *, today: object | None = None) -> list[str] | None:
    """A protection level past 100% said for what it is, and the protective put a book of that
    size would actually need, priced on the index it tracks (SPY when no name is held)."""
    m = _PROTECTION.search(text)
    if m is None:
        return None
    level = float(m.group("p") or m.group("p2"))
    if level <= 100:
        return None
    size_m = _BOOK_SIZE.search(text)
    size = None
    if size_m is not None:
        unit = (size_m.group("u") or "").lower()
        size = float(size_m.group("n").replace(",", "")) * {
            "k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6}.get(unit, 1.0)
    out = [f"Bottom line: {level:g}% protection is not something a hedge can give — a put pays "
           "at most what the holding loses, so cover stops at 100%; puts on "
           f"{level:g}% of the book's value hedge the first 100% and leave you net short the "
           f"other {level - 100:g}%, a bet that the market falls, which loses when it rises."]
    from argus.lui.research.equity_options import ChainUnavailable, load_chain

    try:
        chain = load_chain("SPY")
    except ChainUnavailable:
        chain = None
    if chain is not None and size:
        from datetime import UTC, datetime

        from argus.market.options import NEW_YORK

        day = datetime.now(UTC).astimezone(NEW_YORK).date()
        puts = [c for c in chain.contracts if c.right == "P" and c.quoted
                and (c.expiry - day).days >= 1]
        if puts:
            expiry = min({c.expiry for c in puts},
                         key=lambda e: (abs((e - day).days - PROTECTION_DAYS), e))
            leg = [c for c in puts if c.expiry == expiry]
            put = min(leg, key=lambda c: (abs(c.strike - chain.spot * 0.95), c.strike))
            exact = size / (chain.spot * SHARES_PER_CONTRACT)
            held = max(1, round(exact))
            each = put.mid * SHARES_PER_CONTRACT
            covered = held * chain.spot * SHARES_PER_CONTRACT / size
            out.append(f"What 100% protection of ${size:,.0f} costs, on SPY as the stand-in for "
                       f"a stock book: the book is {exact:.2f} SPY contracts' worth, so {held} SPY "
                       f"{'put' if held == 1 else 'puts'} at the {put.strike:g} strike (about 5% "
                       f"below {chain.spot:,.2f}) expiring {expiry:%d %b %Y} "
                       f"{'covers' if held == 1 else 'cover'} {covered:.0%} of "
                       f"it for about ${held * each:,.0f} at mid "
                       f"({held * each / size:.2%} of the book); the {level:g}% asked would be "
                       f"about {max(1, round(exact * level / 100))} contracts, "
                       f"${max(1, round(exact * level / 100)) * each:,.0f}.")
            out.append("SPY protects a book only as far as the book moves with the S&P 500; name "
                       "the holdings and each one's own put can be priced.")
            out.append(f"Data: Cboe delayed options chain, {chain.quoted_at} New York; mid-quotes, "
                       "not fills. Not advice.")
            return out
    out.append("Name the book's size and holdings (\"hedge my $100,000 book of SPY with puts\") "
               "and the put that covers it is priced from Cboe's chain.")
    return out


__all__ = ["protection_lines", "share_price_lines", "split_lines"]
