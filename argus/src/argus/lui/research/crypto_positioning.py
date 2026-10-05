"""Answer lines for crypto positioning questions: perpetual open interest across exchanges, and
Deribit's option put/call open-interest ratio and max pain.

Round 42 (judge): "What is the current Bitcoin perpetual open interest in USD across major
exchanges and has it risen or fallen over the last 7 days?" was answered with a 7-day price move,
and "What is the put/call open interest ratio for BTC options on Deribit right now and where is
max pain for the next monthly expiry?" with at-the-money implied volatility. The data, the unit
checks and the live values are in `market/crypto_oi.py`; this module only recognises the question
and states the figures.

What it answers: anything that asks for open interest of a coin's perpetuals (BTC, ETH, SOL, XRP,
DOGE, BNB, ADA, AVAX, LTC, SUI), and, for BTC and ETH, Deribit's put/call open-interest ratio and
max pain. The first line answers whichever the question names first. What it leaves to the other
readers: funding rates, prices, 25-delta skew, implied volatility, and put/call *volume* — a
question that is none of the above returns None.

Every venue is named in the data line; one that did not answer is named with the reason (Binance
answers HTTP 451 from US hosts, the hosted console's case), and the 7-day change is stated only
over venues that returned both readings.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Final

from argus.market import crypto_oi
from argus.market.crypto_oi import ExpiryKind, OpenInterest, OptionsOI

_OI: Final = re.compile(r"\bopen[\s-]*interest\b|\boi\b", re.I)
_OPTION_WORD: Final = re.compile(r"\boptions?\b", re.I)
_PERP_WORD: Final = re.compile(r"\bperp\w*|\bfutures?\b|\bswaps?\b|\bexchanges?\b|\bvenues?\b"
                               r"|\bbinance\b|\bbybit\b|\bokx\b|\bbitget\b|\bhyperliquid\b", re.I)
_MAX_PAIN: Final = re.compile(r"\bmax[\s-]*pain\b", re.I)
_PUT_CALL: Final = re.compile(r"\bput[\s/-]*(?:to[\s-]*)?call\b|\bputs?\s+(?:vs\.?|versus|to|and)"
                              r"\s+calls?\b", re.I)
_VOLUME: Final = re.compile(r"\bvolume\b", re.I)
_WEEKLY: Final = re.compile(r"\bweekly\b|\b(?:this|next)\s+week\b", re.I)
_QUARTERLY: Final = re.compile(r"\bquarterly\b", re.I)
_COIN_WORDS: Final = (
    ("BTC", r"btc|bitcoin"), ("ETH", r"eth|ether|ethereum"), ("SOL", r"sol|solana"),
    ("XRP", r"xrp|ripple"), ("DOGE", r"doge|dogecoin"), ("BNB", r"bnb"), ("ADA", r"ada|cardano"),
    ("AVAX", r"avax|avalanche"), ("LTC", r"ltc|litecoin"), ("SUI", r"sui"),
)
_MONTHS: Final = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4,
    "june": 6, "jun": 6, "july": 7, "jul": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9,
    "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12,
    "december": 12,
}
_MONTH: Final = re.compile(r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\b", re.I)
NOT_ADVICE: Final = "Not advice."


def _coin(text: str) -> str | None:
    """The first coin named in ``text`` (by position), or None."""
    best: tuple[int, str] | None = None
    for ticker, words in _COIN_WORDS:
        m = re.search(rf"\b(?:{words})\b", text, re.I)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), ticker)
    return best[1] if best else None


def _usd(x: float) -> str:
    if abs(x) >= 1e9:
        return f"${x / 1e9:,.2f}bn"
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.1f}m"
    return f"${x:,.0f}"


def _day(d: date) -> str:
    return f"{d.day} {d:%b %Y}"


def _names(names: list[str] | tuple[str, ...]) -> str:
    return ", ".join(names)


def _move(pct: float) -> str:
    if abs(pct) < 0.0005:
        return "flat (under 0.1%)"
    return f"{'up' if pct > 0 else 'down'} {abs(pct) * 100:.1f}%"


def _intents(text: str) -> tuple[bool, bool]:
    """(asks for perpetual open interest, asks for Deribit options positioning)."""
    oi = bool(_OI.search(text))
    options = bool(_MAX_PAIN.search(text) or _PUT_CALL.search(text)
                   or (oi and _OPTION_WORD.search(text)))
    if _PUT_CALL.search(text) and _VOLUME.search(text) and not oi:
        options = False
    perp = oi and (bool(_PERP_WORD.search(text)) or not options)
    return perp, options


# --------------------------------------------------------------------------------------------
# perpetual open interest


def _oi_lines(reading: OpenInterest) -> list[str]:
    coin = reading.coin
    got = [v for v in reading.venues if v.now is not None]
    if not got:
        why = "; ".join(f"{v} ({e})" for v, e in reading.failed)
        return [f"Bottom line: no exchange returned {coin} perpetual open interest just now, so "
                f"no figure is given ({why}); ask again in a minute.", NOT_ADVICE]
    ch = reading.change
    head = (f"Bottom line: {coin} perpetual open interest is about {_usd(reading.total_usd)} "
            f"across the {len(got)} venue{'s' if len(got) != 1 else ''} that answered "
            f"({_names(reading.answered)})")
    if ch is not None:
        head += (f", and over the last 7 days it is {_move(ch.pct)} "
                 f"({_usd(ch.past_usd)} to {_usd(ch.now_usd)}, over the {len(ch.venues)} "
                 f"venue{'s' if len(ch.venues) != 1 else ''} with a 7-day reading: "
                 f"{_names(ch.venues)}).")
    else:
        head += ", but no venue returned a 7-day history, so the 7-day change is not given."
    out = [head]
    out.append("Now: " + "; ".join(
        f"{v.venue} {_usd(v.now.usd)} ({v.now.coin:,.0f} {coin}, "
        f"{crypto_oi.utc_stamp(v.now.ts_ms)})"
        for v in got if v.now) + ".")
    if ch is not None:
        past = [v for v in got if v.past is not None]
        out.append("7 days ago: " + "; ".join(
            f"{v.venue} {_usd(v.past.usd)} ({v.past.coin:,.0f} {coin}, "
            f"{crypto_oi.utc_stamp(v.past.ts_ms)})" for v in past if v.past) + f". In {coin} "
            f"terms the same venues are {_move(ch.coin_pct)}, so price moved the dollar figure "
            f"by {(ch.pct - ch.coin_pct) * 100:+.1f} points.")
    nohist = [f"{v.venue} ({v.history})" for v in got if v.past is None and v.history]
    if nohist:
        out.append("No 7-day reading from: " + "; ".join(nohist) + ".")
    if reading.failed:
        out.append("Did not answer: " + "; ".join(f"{v} ({e})" for v, e in reading.failed)
                   + ". They are left out of the total, so it is a floor.")
    out.append("Data: " + "; ".join(v.endpoint for v in got) + ", read now.")
    scope = ("Scope: USDT-margined perpetuals on each venue plus Deribit's inverse "
             f"{coin}-PERPETUAL; coin-margined contracts elsewhere are not counted; Bybit is the "
             "single side (its own doubled figure is not used).")
    if coin not in crypto_oi.DERIBIT_COINS:
        scope += " Deribit is included for BTC and ETH only."
    out.append(scope)
    out.append(NOT_ADVICE)
    return out


# --------------------------------------------------------------------------------------------
# Deribit options


def _expiry_args(text: str) -> tuple[ExpiryKind, int | None]:
    month = _MONTH.search(text)
    kind: ExpiryKind = ("weekly" if _WEEKLY.search(text) else
                        "quarterly" if _QUARTERLY.search(text) else "monthly")
    return kind, (_MONTHS[month.group(1).lower()] if month else None)


def _options_lines(text: str, cur: str, book: OptionsOI) -> list[str]:
    exp = book.expiry
    ratio = "n/a" if book.ratio is None else f"{book.ratio:.2f}"
    whole = (f"{ratio} ({book.puts_oi:,.0f} {cur} of puts to "
             f"{book.calls_oi:,.0f} {cur} of calls over {book.expiries} expiries)")
    pain_at = _MAX_PAIN.search(text)
    pc_at = _PUT_CALL.search(text)
    pain_first = pain_at is not None and (pc_at is None or pain_at.start() < pc_at.start())
    if exp is None:
        what = book.asked
        head = (f"Bottom line: Deribit's {cur} put/call open-interest ratio is {ratio} across the "
                f"whole book, but no {what} is listed, so there is no max pain to give.")
    else:
        er = "n/a" if exp.ratio is None else f"{exp.ratio:.2f}"
        gap = ((exp.max_pain / book.spot - 1) * 100) if exp.max_pain else 0.0
        pain = ("" if exp.max_pain is None else
                f"max pain is ${exp.max_pain:,.0f}, {abs(gap):.1f}% "
                f"{'above' if gap > 0 else 'below' if gap < 0 else 'at'} the index of "
                f"${book.spot:,.0f}")
        if pain_first and pain:
            head = (f"Bottom line: for Deribit's {_day(exp.expiry)} {exp.kind} {cur} expiry "
                    f"{pain}; "
                    f"the put/call open-interest ratio is {er} for that expiry and {ratio} across "
                    f"the whole book.")
        elif pc_at is None and pain_at is None:
            total = book.calls_oi + book.puts_oi
            head = (f"Bottom line: Deribit lists {total:,.0f} {cur} of option open interest "
                    f"(about {_usd(total * book.spot)}), put/call ratio {ratio}; for the "
                    f"{_day(exp.expiry)} {exp.kind} expiry it is {er} and {pain}.")
        else:
            head = (f"Bottom line: Deribit's {cur} put/call open-interest ratio is {ratio} across "
                    f"the whole book and {er} for the {_day(exp.expiry)} {exp.kind} expiry, where "
                    f"{pain}.")
    out = [head, f"Whole book: {whole}."]
    if exp is not None:
        out.append(f"{_day(exp.expiry)} expiry ({exp.kind}, {exp.strikes} strikes): "
                   f"{exp.calls_oi:,.0f} {cur} of calls, {exp.puts_oi:,.0f} of puts.")
        if exp.pain_usd is not None:
            out.append(f"Max pain is the strike where option holders collect least if the index "
                       f"settles there: {_usd(exp.pain_usd)} at ${exp.max_pain:,.0f}, against "
                       f"{_usd(exp.pain_at_spot_usd)} if it settled at today's index.")
    out.append(f"Data: Deribit public/get_book_summary_by_currency?currency={cur}&kind=option "
               f"({book.instruments} instruments, open interest in {cur}, read now) and "
               f"public/get_index_price?index_name={cur.lower()}_usd for the index.")
    out.append(NOT_ADVICE)
    return out


# --------------------------------------------------------------------------------------------


def lines(text: str) -> list[str] | None:
    """The positioning answer ``text`` asks for, or None when it is not about open interest,
    put/call open-interest ratio or max pain."""
    perp, options = _intents(text)
    if not (perp or options):
        return None
    coin = _coin(text)
    if coin is None:
        return None
    first_perp = _OI.search(text)
    first_opt = min((m.start() for m in (_MAX_PAIN.search(text), _PUT_CALL.search(text),
                                         _OPTION_WORD.search(text)) if m), default=10**9)
    perp_first = perp and (not options
                           or (first_perp is not None and first_perp.start() < first_opt))
    out: list[str] = []
    blocks: list[list[str]] = []
    if options and coin in crypto_oi.DERIBIT_COINS:
        kind, month = _expiry_args(text)
        try:
            blocks.append(_options_lines(text, coin, crypto_oi.options_read(coin, kind=kind,
                                                                            month=month)))
        except Exception as exc:  # a book that will not load is said, not raised
            why = str(exc).splitlines()[0][:100] if str(exc) else type(exc).__name__
            blocks.append([f"Bottom line: Deribit's {coin} option book did not answer just now "
                           f"({why}), so no put/call ratio or max pain is given; ask again in a "
                           f"minute.", NOT_ADVICE])
    elif options and not perp:
        return None
    if perp and coin in crypto_oi.COINS:
        blocks.append(_oi_lines(crypto_oi.read(coin)))
    if perp_first:
        blocks.reverse()
    for block in blocks:
        out.extend(x for x in block if x != NOT_ADVICE)
    if not out:
        return None
    out.append(NOT_ADVICE)
    return out
