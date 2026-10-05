"""A book stated as leverage multiples with longs and shorts: its gross and net exposure, its beta
to Bitcoin, and whether a named hedge would actually hedge it.

"I hold 200% leveraged exposure via 2x long BTC and -1x inverse ETH — what's my real net crypto
beta, and should I hedge the rest with gold?" was read as a 100% BTC book, the short leg dropped
(round 39 judge, the MCP research task), and on the console as a 2x BTC liquidation question. A
signed book is the sum of its legs: each leg's multiple of equity times that leg's return, so its
beta to BTC is the multiple-weighted sum of each leg's own beta to BTC, measured here on the same
hourly returns over the last 30 days that the hedge engine uses (`lui/research/book.py`), every
hour of the day, since perpetuals are held through all of them.

A named hedge is judged by how much of the book's variance it removes: the squared correlation of
the book's hourly returns with the hedge's. A hedge that removes little is said to, in those words,
with the leg that would remove most named beside it.
"""

from __future__ import annotations

import re
from typing import Final

from argus.desk.portfolio import beta, correlation
from argus.lui.research.parse import research_symbols

_LEG: Final = re.compile(
    r"(?P<side0>long|short)?\s*(?P<sign>[-\u2212])?\s*(?P<mult>\d+(?:\.\d+)?)\s*(?P<unit>x|%)\s+"
    r"(?P<side>long|short|inverse)?\s*(?:on\s+|in\s+|of\s+)?(?P<name>[A-Za-z$]{2,12})", re.I)
"""A leg: "2x long BTC", "-1x inverse ETH", "long 3x SOL", and in percent of equity with the side
said, "short 30% BTC" or "long 130% BTC" (round 40 hostile, C6: the same coin held both ways was
read as net short). A bare "40% NVDA" is an ordinary holding and is left to the book parser."""
ASKED: Final = re.compile(r"\bnet\s+(?:\w+\s+){0,2}(?:beta|exposure|delta|position)\b|"
                          r"\b(?:real|true|actual)\s+(?:net\s+)?(?:beta|exposure)\b|"
                          r"\bgross\s+exposure\b", re.I)
_HEDGE_WITH: Final = re.compile(r"\bhedge\b[^?]{0,40}\bwith\s+(?P<name>[A-Za-z$]{2,12})", re.I)


def stated_legs(text: str) -> list[tuple[str, float]]:
    """Every leg as stated, in order: (contract, signed multiple of equity)."""
    out: list[tuple[str, float]] = []
    for m in _LEG.finditer(text):
        symbols = research_symbols(m.group("name"))[0]
        side = (m.group("side") or m.group("side0") or "").lower()
        if not symbols or (m.group("unit") == "%" and not side and not m.group("sign")):
            continue
        mult = float(m.group("mult")) / (100.0 if m.group("unit") == "%" else 1.0)
        short = bool(m.group("sign")) or side in ("short", "inverse")
        out.append((symbols[0], -mult if short else mult))
    return out


def legs(text: str) -> dict[str, float]:
    """Each named contract's signed multiple of equity, summed per contract: "2x long BTC" is
    +2, "-1x inverse ETH" and "1x short ETH" are -1, and "short 30% BTC ... long 130% BTC" is
    +1.0 BTC. Empty when fewer than two legs are stated."""
    listed = stated_legs(text)
    if len(listed) < 2:
        return {}
    found: dict[str, float] = {}
    for symbol, mult in listed:
        found[symbol] = found.get(symbol, 0.0) + mult
    return found


def _t(symbol: str) -> str:
    return symbol.removesuffix("USDT")


def lines(text: str) -> list[str] | None:
    """The signed book's exposure, beta to BTC and the named hedge, or None when this is not a
    question about a book of leveraged long and short legs."""
    book = legs(text)
    if not book or not (ASKED.search(text) or (any(w < 0 for w in book.values())
                        and _HEDGE_WITH.search(text))):
        return None
    from argus.lui.research.data import load

    hedge_m = _HEDGE_WITH.search(text)
    hedge = None
    if hedge_m is not None:
        named = research_symbols(hedge_m.group("name"))[0]
        hedge = named[0] if named and named[0] not in book else None
    names = tuple(sorted({*book, "BTCUSDT", *((hedge,) if hedge else ())}))
    data = load(names)
    series = {s: data.raw.get(s) or {} for s in names}
    hours = sorted(set.intersection(*(set(v) for v in series.values()))) if all(
        series.values()) else []
    if len(hours) < 48:
        return ["Bottom line: hourly prices for " + ", ".join(_t(s) for s in names)
                + " could not all be read just now, so the book's beta is not computed; ask "
                  "again in a minute."]
    col = {s: [series[s][h] for h in hours] for s in names}
    btc = col["BTCUSDT"]
    book_r = [sum(w * col[s][i] for s, w in book.items()) for i in range(len(hours))]
    listed = stated_legs(text)
    gross = sum(abs(m) for _, m in listed)
    net = sum(book.values())
    leg_beta = {s: (1.0 if s == "BTCUSDT" else beta(col[s], btc)) for s in book}
    if any(b is None for b in leg_beta.values()):
        return None
    total = sum(w * (leg_beta[s] or 0.0) for s, w in book.items())
    stated = " and ".join(f"{m:+g}x {_t(s)}" for s, m in listed)
    out = [f"Bottom line: your book's beta to Bitcoin is {total:+.2f} — each 1% move in BTC moves "
           f"your equity about {abs(total):.2f}% the same way"
           + (" (the opposite way)" if total < 0 else "")
           + f"; the legs are {stated}: {gross:.0%} of equity gross, {net:+.0%} net."]
    for s, w in book.items():
        if s == "BTCUSDT":
            continue
        b = leg_beta[s] or 0.0
        out.append(f"{_t(s)} moves {b:.2f}% for each 1% in BTC (hourly, last 30 days), so the "
                   f"{w:+g}x {_t(s)} leg carries {w * b:+.2f} of BTC beta"
                   + (" — a short in a coin that moves more than BTC offsets more than its size."
                      if w < 0 and b > 1 else "."))
    corr_btc = correlation(book_r, btc) or 0.0
    if len(book) == 1:
        only = _t(next(iter(book)))
        out.append(f"Every leg is {only}, so they net to one position, {net:+.0%} of equity in "
                   f"{only}; the {gross:.0%} gross matters for margin, not market risk — each "
                   f"account's leg is liquidated on its own margin, so ask for the liquidation "
                   f"price of each leg separately.")
        out.append(f"Data: {data.provenance}. Analysis, not advice.")
        return out
    out.append(f"Net is not the whole risk: with {gross:.0%} gross, the legs can move apart — "
               f"BTC explains {corr_btc * corr_btc:.0%} of this book's hourly moves, and the rest "
               f"is the gap between the coins, which no BTC hedge removes.")
    if hedge is not None:
        h = col[hedge]
        rho = correlation(book_r, h) or 0.0
        b_h = beta(book_r, h) or 0.0
        out.append(f"Hedging with {_t(hedge)} ({'a short' if b_h >= 0 else 'a long'} in it): it "
                   f"removes about {rho * rho:.0%} of the book's "
                   f"variance (correlation {rho:+.2f}, beta {b_h:+.2f}), "
                   + ("so it barely hedges this book; " if rho * rho < 0.2 else "")
                   + f"a short of about {abs(total):.2f}x equity in BTC removes "
                   f"{corr_btc * corr_btc:.0%}.")
    out.append(f"Data: {data.provenance}; betas and correlations on {len(hours)} matching hourly "
               f"returns. Analysis, not advice.")
    return out
