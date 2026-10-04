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

Round 33's hostile review added three more, each answered without a word: "Now that Michael Saylor
is CEO of Coinbase" (the model's own trace called it false and answered anyway), "Is BRK.A priced
the same as BRK.B?" (BRK.A dropped), "AAPL's closing price this past Saturday" (a live quote
given). Chief executives come from a small table held against recent 8-K item 5.02 filings; an
unlisted share class and a weekend close are said for what they are.
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


CHIEF_EXECUTIVES: Final = {
    "COINUSDT": "Brian Armstrong", "MSTRUSDT": "Phong Le", "NVDAUSDT": "Jensen Huang",
    "AMDUSDT": "Lisa Su", "TSLAUSDT": "Elon Musk", "AAPLUSDT": "Tim Cook",
    "MSFTUSDT": "Satya Nadella", "METAUSDT": "Mark Zuckerberg", "GOOGLUSDT": "Sundar Pichai",
    "AMZNUSDT": "Andy Jassy",
}
"""Chief executives of the listed names a question is likeliest to mix up, as each company's own
filings name them (read 2026-10-04). A change at the top is filed on an 8-K under item 5.02 within
four business days, so a recent 5.02 filing withholds the line rather than assert a stale name."""
_CEO_CLAIM: Final = re.compile(
    r"\b(?P<who>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:is|became|was\s+(?:named|made|appointed)|"
    r"as|now\s+runs|runs|took\s+over\s+as)\s+(?:the\s+)?(?:new\s+)?(?:ceo|chief\s+executive)\s+"
    r"(?:of|at)\s+(?P<co>[A-Za-z][\w.&-]{1,30})|\b(?P<co2>[A-Za-z][\w.&-]{1,30})(?:'s)?\s+"
    r"(?:new\s+)?(?:ceo|chief\s+executive)\s+(?P<who2>[A-Z][a-z]+\s+[A-Z][a-z]+)", re.I)


def ceo_line(text: str) -> str | None:
    """A chief executive the question names for a company, held against the table above."""
    claim = _CEO_CLAIM.search(text)
    if claim is None:
        return None
    from argus.lui.research import research_symbols

    who = " ".join((claim.group("who") or claim.group("who2") or "").split())
    named = research_symbols(claim.group("co") or claim.group("co2") or "")[0]
    if not named or named[0] not in CHIEF_EXECUTIVES:
        return None
    actual = CHIEF_EXECUTIVES[named[0]]
    if who.lower() == actual.lower() or who.lower() in actual.lower().split():
        return None
    ticker = named[0].removesuffix("USDT")
    try:
        from datetime import UTC, datetime, timedelta

        from argus.lui.watchlist import recent_8k

        filed = recent_8k([ticker], datetime.now(UTC) - timedelta(days=30)).get(ticker, [])
        change = next((f for f in filed if re.search(r"\b5\.02\b", f.items)), None)
        if change is not None:
            return (f"Premise check: that {who} is {ticker}'s chief executive is not confirmed "
                    f"here — {actual} was, as the company's filings named it, and {ticker} filed "
                    f"an 8-K under item 5.02 (an officer or director change) on "
                    f"{change.day:%d %b %Y}; read that filing on SEC EDGAR before acting on the "
                    f"claim. The answer below does not assume it.")
    except Exception:
        pass
    return (f"Premise check: {ticker}'s chief executive is {actual}, not {who}, as the company's "
            f"own filings name it, and no change of officers (8-K item 5.02) was filed in the last "
            f"30 days; the "
            f"answer below does not assume the change.")


_UNLISTED_CLASS: Final = re.compile(r"\b(?P<base>[A-Z]{2,5})\.(?P<cls>[A-Z])\b")


def share_class_line(text: str) -> str | None:
    """A share class the question names that Bitget does not list ("Is BRK.A priced the same as
    BRK.B?" was answered for BRK.B alone, a hostile review, round 33)."""
    from argus.lui.research import research_symbols

    for m in _UNLISTED_CLASS.finditer(text):
        token = m.group(0)
        if research_symbols(token)[0]:
            continue
        listed = next((f"{m.group('base')}.{c}" for c in "ABC" if c != m.group("cls")
                       and research_symbols(f"{m.group('base')}.{c}")[0]), None)
        return (f"Premise check: {token} is not listed on Bitget"
                + (f" — only {listed} is, so the figures below are {listed}'s" if listed else "")
                + (". Berkshire's Class A share converts into 1,500 Class B shares, so the two "
                   "are never priced the same: A trades near 1,500 times B."
                   if m.group("base") == "BRK" else "."))
    return None


_WEEKEND_CLOSE: Final = re.compile(
    r"\b(?:clos\w*|settle\w*|ended)\b[^?]{0,40}\b(?:on\s+|this\s+past\s+|last\s+)?(?P<day>saturday|"
    r"sunday)\b|\b(?:saturday|sunday)'?s?\s+(?:close|closing)", re.I)


def weekend_close_line(text: str, symbols: tuple[str, ...]) -> str | None:
    """A stock's close asked for a weekend day, when its exchange is shut."""
    found = _WEEKEND_CLOSE.search(text)
    if found is None or not symbols:
        return None
    from argus.lui.research.parse import is_us_equity

    stock = next((s for s in symbols if is_us_equity(s)), None)
    if stock is None:
        return None
    day = (found.group("day") or found.group(0).split()[0]).strip("'s").capitalize()
    name = stock.removesuffix("USDT")
    return (f"Premise check: {name}'s stock does not trade on a {day} — the exchange is shut — so "
            f"it has no {day} close; its last close is Friday's. Bitget's {name} perpetual does "
            f"trade at weekends, and any weekend figure below is the perpetual's, not the stock's.")


_CORPORATE_EVENT: Final = re.compile(
    r"\b(?:merged?|merging|acquired|bought\s+out|filed\s+for\s+bankruptcy|went\s+bankrupt|"
    r"bankrupt\w*|chapter\s+11|delisted|de-listed|got\s+delisted|taken\s+private|spun\s+off)\b",
    re.I)
_PERP_DELISTED: Final = re.compile(
    r"\b(?:delist\w*|removed?|stopped\s+(?:trading|listing))\b[^?.]{0,40}?\b(?P<name>[A-Za-z]{2,10})"
    r"\s*(?:usdt\s*)?(?:perp\w*|futures?|contract)\b|\b(?P<name2>[A-Za-z]{2,10})\s*(?:usdt\s*)?"
    r"(?:perp\w*|futures?|contract)\b[^?.]{0,30}\b(?:delisted|removed)\b", re.I)


def corporate_event_line(text: str, symbols: tuple[str, ...]) -> str | None:
    """A merger, bankruptcy or delisting the question states about a listed name or contract.

    "Tesla merged with Twitter to form X Motors", "Coinbase filed for bankruptcy and got
    delisted from Nasdaq" and "did Bitget delist the ETH perpetual last week?" were each answered
    with performance figures and no word on the claim (a hostile review, round 34). A contract
    said to be delisted is checked against Bitget's own contract list; a corporate event is
    checked where Bitget's live listing answers it, and otherwise said to be unchecked."""
    perp = _PERP_DELISTED.search(text)
    if perp is not None:
        from argus.lui.research import research_symbols
        from argus.market.bitget import public_get

        named = research_symbols(perp.group("name") or perp.group("name2") or "")[0]
        if named:
            try:
                rows = public_get("/api/v2/mix/market/contracts",
                                  {"productType": "USDT-FUTURES", "symbol": named[0]},
                                  timeout=10.0)
            except Exception:
                rows = None
            status = str((rows or [{}])[0].get("symbolStatus") or "") if rows else ""
            if status:
                return (f"Premise check: {named[0]} is "
                        + ("still listed and trading" if status == "normal"
                           else f"listed with status \"{status}\"")
                        + " in Bitget's contract list read just now"
                        + (", so it has not been delisted." if status == "normal" else "."))
    event = _CORPORATE_EVENT.search(text)
    if event is None or not symbols:
        return None
    from argus.lui.research.parse import is_us_equity, last_price

    stock = next((s for s in symbols if is_us_equity(s)), None)
    if stock is None:
        return None
    name = stock.removesuffix("USDT")
    try:
        live = last_price(stock)
    except Exception:
        live = None
    trading = (f"Bitget's {name} perpetual is trading normally at {float(live):,.2f}, which "
               f"tracks the listed stock; " if live else "")
    return (f"Premise check: \"{event.group(0)}\" is not confirmed here — {trading}a merger, "
            f"bankruptcy or delisting of {name} would be filed with the SEC on an 8-K, and this "
            f"answer does not assume it happened.")


_SPOT_LEVERAGE: Final = re.compile(
    r"\bspot\b[^?.]{0,50}?\b(?P<x>\d+(?:\.\d+)?)\s*x\b|\b(?P<x2>\d+(?:\.\d+)?)\s*x\b[^?.]{0,50}?"
    r"\bspot\s+market\b", re.I)


def spot_leverage_line(text: str) -> str | None:
    """Leverage asked on the spot market ("buy 1 BTC on the Bitget spot market using 20x — my
    liquidation price?" was priced as a perpetual without a word, a hostile review, round 34)."""
    m = _SPOT_LEVERAGE.search(text)
    if m is None or float(m.group("x") or m.group("x2")) <= 1:
        return None
    lev = m.group("x") or m.group("x2")
    return (f"Premise check: a spot buy has no leverage — you pay for the whole coin and it cannot "
            f"be liquidated. {lev}x is a futures (perpetual) setting, or a separate margin "
            f"product with its own limits on Bitget; the figures below are for a {lev}x "
            f"perpetual, the product that has a liquidation price.")


def lines(text: str, symbols: tuple[str, ...]) -> list[str]:
    """Each premise line that applies, in order; empty when the question claims nothing here."""
    said = [halving_line(text), founder_line(text, symbols), usdt_line(text), spot_line(text),
            ceo_line(text), share_class_line(text), weekend_close_line(text, symbols),
            corporate_event_line(text, symbols), spot_leverage_line(text)]
    out = [x for x in said if x]
    if not out:
        unchecked = unchecked_line(text)
        if unchecked:
            out.append(unchecked)
    return out


__all__ = ["FOUNDERS", "founder_line", "halving_line", "lines", "spot_line", "unchecked_line",
           "usdt_line"]

trace_module(globals())
