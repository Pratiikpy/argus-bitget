"""Claims a question takes as given, checked against the filing or price series that settles them.

A hostile review (round 30) put four false premises to the console, and each was answered around
rather than corrected:

* "Apple's new $5.00 per share quarterly dividend" got "the data source holds no dividend figure",
  while the 10-Q the same answer cited carried $0.27 (SEC EDGAR XBRL,
  ``CommonStockDividendsPerShareDeclared``).
* "Back when BTC was trading at $120,000 last month" got "$120,000 put in at the start would be
  $131,708 now" — a price read as a sum — beside the true range, 74,910 to 87,373.
* "Since Meta Platforms' ticker on Bitget is FB" got META's funding with the model's own reasoning
  repeating "ticker FB on Bitget".
* "Coinbase trades on the NYSE under the ticker COIN" went unchecked; EDGAR lists it on Nasdaq.

Each check returns one "Premise check:" line or None. A check that cannot read its source says
nothing rather than guess: the claim is then neither confirmed nor denied.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Final

from argus.lui.trace import trace_module

DIVIDEND_CLAIM: Final = re.compile(
    r"\$\s?(?P<d>\d+(?:\.\d+)?)\s*(?:(?:per|a)\s+share\s+)?(?:\w+\s+){0,2}?dividends?\b|"
    r"\bdividends?\s+(?:of|at|is|was|to)\s+\$\s?(?P<d2>\d+(?:\.\d+)?)", re.I)
PAST_LEVEL: Final = re.compile(
    r"\b(?:was|were|been)\s+(?:trading\s+|sitting\s+)?(?:at|around|near|above|over|below|under)\s+"
    r"\$\s?(?P<lvl>\d(?:[\d,]*\d)?(?:\.\d+)?)(?![\d,.]*\d)\s*(?P<k>k)?\b[^.?]{0,40}?\b(?P<when>last\s+"
    r"(?:week|month|year)|yesterday|(?P<n>\d+|a|two|three|six)\s+(?P<u>days?|weeks?|months?)\s+"
    r"ago)\b", re.I)
PAST_LEVEL_FIRST: Final = re.compile(
    r"\b(?P<when>last\s+(?:week|month|year)|yesterday|(?P<n>\d+|a|two|three|six)\s+(?P<u>days?|"
    r"weeks?|months?)\s+ago)\b[^.?]{0,20}?\bwhen\s+\w+\s+(?:was|were)\s+(?:trading\s+)?(?:at|around|"
    r"near|above|over|below|under)\s+\$\s?(?P<lvl>\d(?:[\d,]*\d)?(?:\.\d+)?)(?![\d,.]*\d)\s*(?P<k>k)?\b",
    re.I)
"""The same claim with the time first: "Two weeks ago when ETH was at $6,000"."""
TICKER_CLAIM: Final = re.compile(
    r"\b(?:ticker|symbol)\s+(?:on\s+bitget\s+)?(?:is|was)\s+\$?(?P<t>[A-Z]{1,6})\b|"
    r"\b(?:under|as)\s+(?:the\s+)?(?:ticker|symbol)\s+\$?(?P<t2>[A-Z]{1,6})\b", re.I)
EXCHANGE_CLAIM: Final = re.compile(
    r"\b(?:trades?|listed|trading)\s+on\s+(?:the\s+)?(?P<ex>NYSE|New\s+York\s+Stock\s+Exchange|"
    r"Nasdaq|NASDAQ|AMEX|NYSE\s+American)\b", re.I)
_WORDS: Final = {"a": 1, "two": 2, "three": 3, "six": 6}
_UNIT_DAYS: Final = {"day": 1, "week": 7, "month": 31, "year": 366}


def dividend(text: str, symbol: str) -> str | None:
    """A stated dividend per share against the last one declared in the company's filings."""
    m = DIVIDEND_CLAIM.search(text)
    if m is None:
        return None
    stated = float(m.group("d") or m.group("d2"))
    ticker = symbol.removesuffix("USDT").removesuffix("STOCK")
    try:
        from argus.market.fundamentals import FundamentalsSource

        facts, _status = FundamentalsSource().facts(ticker, concept="dividend_per_share",
                                                    as_of=datetime.now(UTC))
    except Exception:
        return None
    if not facts:
        return (f"Premise check: {ticker}'s filings on SEC EDGAR declare no dividend per share, so "
                f"the ${stated:,.2f} the question takes as given is not on its record.")
    last = max(facts, key=lambda f: f.end)
    if last.value > 0 and abs(stated / last.value - 1) <= 0.25:
        return None
    return (f"Premise check: {ticker}'s last declared dividend in its own filings is "
            f"${last.value:,.2f} a share, for the quarter ending {last.end:%d %b %Y} ({last.form} "
            f"filed {last.filed:%d %b %Y}, SEC EDGAR) — not ${stated:,.2f}. A new dividend is "
            f"first filed on an 8-K; this console reads the declared figure, and none at "
            f"${stated:,.2f} is on the record.")


def past_level(text: str, symbol: str, *, now: datetime | None = None) -> tuple[str, float] | None:
    """A price the question says a market traded at in a past window, against that window's
    range on Bitget's daily candles. Returns the line and the stated level when it is outside the
    range, so the caller can drop any reading of that figure as a sum of money."""
    m = PAST_LEVEL.search(text) or PAST_LEVEL_FIRST.search(text)
    if m is None:
        return None
    level = float(m.group("lvl").replace(",", "")) * (1000 if m.group("k") else 1)
    when = m.group("when").lower()
    if when == "yesterday":
        days = 2
    elif m.group("u"):
        n = m.group("n").lower()
        days = (int(n) if n.isdigit() else _WORDS[n]) * _UNIT_DAYS[m.group("u").lower().rstrip("s")]
        days += 3
    else:
        days = _UNIT_DAYS[when.split()[-1]] + 1
    try:
        from argus.market import history

        bars = history.fetch_range(symbol, days=max(days, 2), interval="1Dutc")
    except Exception:
        return None
    clock = now or datetime.now(UTC)
    bars = [b for b in bars if b.ts >= clock - timedelta(days=days)]
    if not bars:
        return None
    low = min(float(b.low) for b in bars)
    high = max(float(b.high) for b in bars)
    if low * 0.97 <= level <= high * 1.03:
        return None
    name = symbol.removesuffix("USDT")
    return (f"Premise check: {name} did not trade at {level:,.0f} in that time — its range on "
            f"Bitget from {bars[0].ts:%d %b} to {bars[-1].ts:%d %b} was {low:,.2f} to "
            f"{high:,.2f} (daily highs and lows), so the question's starting point is not on the "
            f"record; the figures below are the real ones.", level)


def ticker(text: str, symbol: str) -> str | None:
    """A ticker the question says a company has, against Bitget's list and the company's EDGAR
    record."""
    m = TICKER_CLAIM.search(text)
    if m is None:
        return None
    said = (m.group("t") or m.group("t2")).upper()
    real = symbol.removesuffix("USDT").removesuffix("STOCK")
    if said == real:
        return None
    from argus.lui.question import listed_on_bitget

    listed = listed_on_bitget(said)
    name = None
    try:
        from argus.market.evidence import EdgarSource

        found = EdgarSource().listing(real)
        name = found[0] if found else None
    except Exception:
        name = None
    who = name or real
    return (f"Premise check: Bitget lists {who} as {real} ({symbol}), not {said}"
            + (f" — {said} is a different contract on Bitget, and is not what is read below"
               if listed else f"; {said} is not a ticker Bitget lists")
            + ". The figures below are for " + real + ".")


def exchange(text: str, symbol: str) -> str | None:
    """A stock exchange the question says a company lists on, against its EDGAR record."""
    m = EXCHANGE_CLAIM.search(text)
    if m is None:
        return None
    said = m.group("ex")
    real_ticker = symbol.removesuffix("USDT").removesuffix("STOCK")
    try:
        from argus.market.evidence import EdgarSource

        found = EdgarSource().listing(real_ticker)
    except Exception:
        return None
    if not found or not found[2]:
        return None
    name, _tickers, exchanges = found
    norm = {"new york stock exchange": "nyse", "nasdaq": "nasdaq", "nyse": "nyse", "amex": "amex",
            "nyse american": "nyse american"}
    said_n = norm.get(re.sub(r"\s+", " ", said.lower()), said.lower())
    if any(said_n == x.lower() for x in exchanges):
        return None
    return (f"Premise check: SEC EDGAR lists {name} on {', '.join(exchanges)}, not {said}. "
            f"On Bitget it trades as {symbol} either way.")


__all__ = ["DIVIDEND_CLAIM", "EXCHANGE_CLAIM", "PAST_LEVEL", "TICKER_CLAIM", "dividend",
           "exchange", "past_level", "ticker"]

trace_module(globals())
