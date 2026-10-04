"""Claims about the world that a question takes as given, checked where a fact can be read, and
named as unchecked where none can.

A hostile review (round 32) put six false premises to the console and every one was answered
around without a word: "Bitget delisted USDT last week", "the Fed cut to 0% in an emergency
meeting", "Solana founder Sam Bankman-Fried's token SOL", "Bitcoin's halving just happened this
month", "the SEC approved a 50x leveraged BTC ETF yesterday", "COIN now trades as a spot pair on
Bitget". The checks here, each against its source:

* **Halving.** Bitcoin's issuance halves every 210,000 blocks. The fourth halving was block
  840,000 on 20 April 2024; the next is block 1,050,000, about four years later (expected in
  2028). A fact of the protocol, written here with its block heights.
* **Founders.** A short table of who founded what, for the names the console lists, from each
  project's own history: Solana — Anatoly Yakovenko (Solana Labs, 2017); Ethereum — Vitalik
  Buterin (with others, 2013-2015); FTX — Sam Bankman-Fried; Binance — Changpeng Zhao; Cardano —
  Charles Hoskinson; Tron — Justin Sun; Bitcoin — Satoshi Nakamoto (a pseudonym).
* **USDT on Bitget.** Bitget's own contract list is read: USDT-margined perpetuals listed today
  means USDT has not been delisted as margin.
* **A spot pair.** Bitget's spot symbol list is read for the pair claimed.
* **Approvals, acquisitions and emergency actions said to have happened** that no source here
  can read: said to be unchecked, so the answer is not taken to confirm them.
"""

from __future__ import annotations

import re
from typing import Final

from argus.lui.trace import trace_module

_HALVING: Final = re.compile(
    r"\bhalving\b[^.?]{0,30}\b(?:just\s+)?(?:happened|occurred|took\s+place|was)\b[^.?]{0,20}\b"
    r"(?:this|last)\s+(?:month|week|year)\b|\b(?:this|last)\s+(?:month|week)'?s\s+halving\b|"
    r"\bhalving\s+(?:just\s+)?(?:happened|occurred)\b|\bjust\s+had\s+(?:the|a|its)\s+halving\b",
    re.I)

FOUNDERS: Final = {
    "SOL": ("Solana", "Anatoly Yakovenko"),
    "ETH": ("Ethereum", "Vitalik Buterin"),
    "BNB": ("Binance", "Changpeng Zhao"),
    "ADA": ("Cardano", "Charles Hoskinson"),
    "TRX": ("Tron", "Justin Sun"),
    "BTC": ("Bitcoin", "Satoshi Nakamoto"),
}
_PEOPLE: Final = {
    "sam bankman-fried": "Sam Bankman-Fried", "sam bankman fried": "Sam Bankman-Fried",
    "sbf": "Sam Bankman-Fried", "vitalik": "Vitalik Buterin", "vitalik buterin": "Vitalik Buterin",
    "cz": "Changpeng Zhao", "changpeng zhao": "Changpeng Zhao", "justin sun": "Justin Sun",
    "charles hoskinson": "Charles Hoskinson", "anatoly yakovenko": "Anatoly Yakovenko",
    "do kwon": "Do Kwon", "michael saylor": "Michael Saylor", "saylor": "Michael Saylor",
    "elon musk": "Elon Musk", "elon": "Elon Musk", "satoshi": "Satoshi Nakamoto",
}
_PERSON: Final = re.compile(r"\b(" + "|".join(sorted((re.escape(p) for p in _PEOPLE),
                                                     key=len, reverse=True)) + r")\b", re.I)
_FOUNDED: Final = re.compile(r"\b(?:found\w*|created|creator|invent\w*|launched|built|"
                             r"'s\s+(?:token|coin|chain|project))\b", re.I)

_USDT_GONE: Final = re.compile(
    r"\b(?:bitget|they|the\s+exchange)\s+(?:has\s+|have\s+|just\s+)?(?:delisted|dropped|removed|"
    r"stopped\s+(?:accepting|supporting)|banned|no\s+longer\s+(?:accepts|supports))\s+usdt\b|"
    r"\busdt\s+(?:was|has\s+been|got)\s+(?:delisted|removed|dropped)\b", re.I)
_SPOT_CLAIM: Final = re.compile(
    r"\b(?P<name>(?-i:[A-Z]{2,6}))\b[^.?]{0,40}\b(?:trades?|listed|lists?|available)\s+"
    r"(?:now\s+)?(?:directly\s+)?(?:as\s+)?(?:a\s+)?spot\s+(?:pair|market|trading)\b|\bspot\s+"
    r"(?:pair|market)\s+(?:for\s+)?(?P<name2>(?-i:[A-Z]{2,6}))\b", re.I)
_UNCHECKED_EVENT: Final = re.compile(
    r"\b(?P<who>sec|cftc|fed|federal\s+reserve|bitget|binance|coinbase|blackrock|the\s+us|us\s+"
    r"government|congress)\b[^.?]{0,40}\b(?P<what>approved|approves|acquired|acquires|bought|"
    r"banned|bans|delisted|listed|launched|announced|declared)\b[^.?]{0,80}\b(?P<when>yesterday|"
    r"today|last\s+week|this\s+week|last\s+night|just\s+now|this\s+month)\b|\b(?P<who2>sec|cftc|"
    r"bitget|binance|blackrock)\b\s+(?:just\s+)?(?P<what2>approved|acquired|banned|delisted)\b",
    re.I)


def halving_line(text: str) -> str | None:
    if _HALVING.search(text) is None:
        return None
    return ("Premise check: no Bitcoin halving happened recently — the last was block 840,000 on "
            "20 Apr 2024, and the next comes at block 1,050,000, expected in 2028; the issuance "
            "halves every 210,000 blocks.")


def founder_line(text: str, symbols: tuple[str, ...]) -> str | None:
    person = _PERSON.search(text)
    if person is None or not _FOUNDED.search(text):
        return None
    who = _PEOPLE[person.group(1).lower()]
    for symbol in symbols:
        base = symbol.removesuffix("USDT")
        project = FOUNDERS.get(base)
        if project is None or project[1] == who:
            continue
        window = text[max(0, person.start() - 60):person.end() + 60]
        if not re.search(rf"\b{re.escape(base)}\b|\b{re.escape(project[0])}\b", window, re.I):
            continue
        return (f"Premise check: {project[0]} ({base}) was founded by {project[1]}, not {who}"
                + (" — Sam Bankman-Fried founded FTX and Alameda Research" if who ==
                   "Sam Bankman-Fried" else "") + "; the figures below do not depend on it.")
    return None


def usdt_line(text: str) -> str | None:
    if _USDT_GONE.search(text) is None:
        return None
    from argus.market.bitget import public_get

    try:
        rows = public_get("/api/v2/mix/market/contracts", {"productType": "USDT-FUTURES"},
                          timeout=10.0)
    except Exception:
        return ("Premise check: that USDT was delisted could not be checked just now — Bitget's "
                "contract list did not answer; nothing below assumes it.")
    live = [r for r in rows or [] if isinstance(r, dict)
            and str(r.get("symbolStatus", "normal")).lower() == "normal"]
    if not live:
        return None
    return (f"Premise check: USDT has not been delisted — Bitget's contract list shows "
            f"{len(live)} USDT-margined perpetuals trading now, margined and settled in USDT.")


def spot_line(text: str) -> str | None:
    claim = _SPOT_CLAIM.search(text)
    if claim is None:
        return None
    base = (claim.group("name") or claim.group("name2") or "").upper()
    if base in ("BTC", "ETH", "USDT", "USDC") or not base:
        return None
    from argus.lui.research import research_symbols
    from argus.market.bitget import spot_taker_fee

    named = research_symbols(base)[0]
    if not named:
        return None
    if spot_taker_fee(named[0]) is not None:
        return None
    rtoken = f"R{base}USDT"
    if spot_taker_fee(rtoken) is not None:
        return (f"Premise check: the stock {base} itself is not a spot pair on Bitget — its spot "
                f"market lists r{base} ({rtoken}), a tokenized stock that tracks {base}, and its "
                f"futures market the perpetual {named[0]}; the figures below say which they "
                f"are for.")
    return (f"Premise check: {base} is not a spot pair on Bitget — its spot symbol list has no "
            f"{named[0]}; Bitget lists {base} as a perpetual ({named[0]}), and the figures below "
            f"are the perpetual's.")


def unchecked_line(text: str) -> str | None:
    found = _UNCHECKED_EVENT.search(text)
    if found is None:
        return None
    if re.search(r"\b(?:fed|federal\s+reserve)\b", found.group(0), re.I) and re.search(
            r"\b(?:cut|hike|raise|lower)\w*\b", text, re.I):
        return None
    said = " ".join(found.group(0).split())
    return (f"Not checked: \"{said}\" — no source this console reads can confirm it, so the answer "
            f"below does not take it as true.")


def lines(text: str, symbols: tuple[str, ...]) -> list[str]:
    """Each premise line that applies, in order; empty when the question claims nothing here."""
    said = [halving_line(text), founder_line(text, symbols), usdt_line(text), spot_line(text)]
    out = [x for x in said if x]
    if not out:
        unchecked = unchecked_line(text)
        if unchecked:
            out.append(unchecked)
    return out


__all__ = ["FOUNDERS", "founder_line", "halving_line", "lines", "spot_line", "unchecked_line",
           "usdt_line"]

trace_module(globals())
