"""Amounts written in a reader's own currency — rupiah, dong, rupees, naira, lira — read as what
they are worth in US dollars.

`parse.in_us_dollars` restates euros, pounds and yen at Bitget's own FX perpetuals. Every other
currency used to be read as dollars or not at all: "1 million rupiah of bitcoin, how much profit"
became a $1,000,000 position, and "tôi có 5 triệu đồng" (5 million dong) was dropped (round 45
newcomer, M7). The words people write a sum with differ by language too: Indonesian *juta*
(million) and *ribu* (thousand), Vietnamese *triệu* and *nghìn*, Indian *lakh* (100,000) and
*crore* (10,000,000). The rate comes from :mod:`argus.market.fx_rates` (the ECB's reference rate,
else ExchangeRate-API's daily rate), and the line that converts says which and of what date.

Ambiguous words are left alone rather than guessed: a bare "peso" (Mexican or Philippine), "real",
"won" before a number ("I won 500"), and the baht sign ฿, which many crypto writers use for
bitcoin.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

_CURRENCIES: Final[tuple[tuple[str, str, str], ...]] = (
    # (pattern after the number, ISO code, the name to say it by)
    (r"rupiah|idr", "IDR", "rupiah"),
    (r"(?:vietnamese\s+)?(?:dong|đồng|dongs)|vnd|₫", "VND", "dong"),
    (r"pakistani\s+rupees?|pkr", "PKR", "Pakistani rupees"),
    (r"(?:indian\s+)?rupees?|inr|₹", "INR", "rupees"),
    (r"naira|ngn|₦", "NGN", "naira"),
    (r"(?:turkish\s+)?lira|₺", "TRY", "lira"),
    (r"reais|brl", "BRL", "reais"),
    (r"philippine\s+pesos?|php|₱", "PHP", "Philippine pesos"),
    (r"mexican\s+pesos?|mxn", "MXN", "Mexican pesos"),
    (r"rand|zar", "ZAR", "rand"),
    (r"ringgit|myr", "MYR", "ringgit"),
    (r"baht|thb", "THB", "baht"),
    (r"(?:korean\s+)?won|krw|₩", "KRW", "won"),
    (r"yuan|rmb|cny|renminbi", "CNY", "yuan"),
    (r"(?:uae\s+)?dirhams?|aed", "AED", "dirham"),
    (r"(?:saudi\s+)?riyals?|sar", "SAR", "riyal"),
    (r"taka|bdt", "BDT", "taka"),
    (r"(?:kenyan\s+)?shillings?|kes", "KES", "Kenyan shillings"),
    (r"cedis?|ghs", "GHS", "cedi"),
    (r"(?:egyptian\s+)?pounds?\s+egp|egp", "EGP", "Egyptian pounds"),
    (r"z[łl]oty|pln|zł", "PLN", "złoty"),
    (r"(?:canadian\s+dollars?|cad)", "CAD", "Canadian dollars"),
    (r"(?:australian\s+dollars?|aud)", "AUD", "Australian dollars"),
    (r"(?:singapore\s+dollars?|sgd)", "SGD", "Singapore dollars"),
    (r"(?:hong\s+kong\s+dollars?|hkd)", "HKD", "Hong Kong dollars"),
    (r"(?:swiss\s+francs?|chf)", "CHF", "Swiss francs"),
)
_PREFIXES: Final = {"rp": "IDR", "₹": "INR", "rs": "INR", "₦": "NGN", "₺": "TRY", "r$": "BRL",
                    "₱": "PHP", "₩": "KRW", "₫": "VND", "rm": "MYR"}
_MULTIPLIERS: Final[tuple[tuple[str, float], ...]] = (
    (r"k|thousand|ribu|rb|nghìn|nghin|ngàn|ngan", 1e3),
    (r"lakhs?|lacs?", 1e5),
    (r"m|mn|mio|million|millions|juta|jt|triệu|trieu", 1e6),
    (r"crores?|cr", 1e7),
    (r"bn|billion|miliar|milyar|tỷ|ty", 1e9),
)
_NUMBER: Final = r"\d{1,3}(?:[.,\s]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?"
_MULT_RE: Final = "|".join(f"(?:{m})" for m, _ in _MULTIPLIERS)
_CODE_RE: Final = "|".join(f"(?P<c{i}>{p})" for i, (p, _, _) in enumerate(_CURRENCIES))
_PREFIX_RE: Final = "|".join(re.escape(p) for p in sorted(_PREFIXES, key=len, reverse=True))
_AFTER: Final = re.compile(
    rf"(?<![\w.,])(?P<n>{_NUMBER})\s*(?P<m>{_MULT_RE})?\.?\s*(?:of\s+)?(?:{_CODE_RE})(?![\w])",
    re.I)
_BEFORE: Final = re.compile(
    rf"(?<![\w$])(?P<p>{_PREFIX_RE})\.?\s?(?P<n>{_NUMBER})\s*(?P<m>{_MULT_RE})?(?![\w])", re.I)


@dataclass(frozen=True)
class LocalAmount:
    said: str
    amount: float
    code: str
    name: str
    start: int
    end: int


def _number(raw: str) -> float:
    """"1.000.000" and "1,000,000" and "1 000 000" are a million; "1,5" and "1.5" are one and a
    half: a separator followed by exactly three digits groups thousands, any other is a decimal."""
    cleaned = raw.strip()
    if re.fullmatch(r"\d{1,3}(?:[.,\s]\d{3})+", cleaned):
        return float(re.sub(r"[.,\s]", "", cleaned))
    if re.fullmatch(r"\d{1,3}(?:[.,\s]\d{3})+[.,]\d+", cleaned):
        whole, frac = re.split(r"[.,](?=\d+$)", cleaned)
        return float(re.sub(r"[.,\s]", "", whole) + "." + frac)
    return float(cleaned.replace(",", "."))


def _multiplier(word: str | None) -> float:
    if not word:
        return 1.0
    for pattern, factor in _MULTIPLIERS:
        if re.fullmatch(pattern, word, re.I):
            return factor
    return 1.0


def amounts(text: str) -> list[LocalAmount]:
    """Every amount in ``text`` written in a currency other than the dollar, euro, pound or yen."""
    found: list[LocalAmount] = []
    for m in _AFTER.finditer(text):
        index = next(i for i in range(len(_CURRENCIES)) if m.group(f"c{i}"))
        _, code, name = _CURRENCIES[index]
        value = _number(m.group("n")) * _multiplier(m.group("m"))
        found.append(LocalAmount(m.group(0).strip(), value, code, name, m.start(), m.end()))
    for m in _BEFORE.finditer(text):
        if any(a.start <= m.start() < a.end for a in found):
            continue
        code = _PREFIXES[m.group("p").lower()]
        name = next(n for _, c, n in _CURRENCIES if c == code)
        value = _number(m.group("n")) * _multiplier(m.group("m"))
        found.append(LocalAmount(m.group(0).strip(), value, code, name, m.start(), m.end()))
    return sorted(found, key=lambda a: a.start)


def in_dollars(text: str) -> tuple[str, list[str]]:
    """``text`` with each local-currency amount restated as US dollars, and a line for each saying
    the rate, its source and its date. An amount whose rate no source answers is left as written
    and said so — never read as dollars."""
    from argus.market.fx_rates import usd_rate

    said: list[str] = []
    out, cursor = [], 0
    for amount in amounts(text):
        rate = usd_rate(amount.code)
        out.append(text[cursor:amount.start])
        cursor = amount.end
        if rate is None:
            said.append(f"{amount.amount:,.0f} {amount.name} not converted: no {amount.code} rate "
                        "answered just now, so no dollar figure is given for it")
            out.append(amount.said)
            continue
        usd = rate.to_usd(amount.amount)
        said.append(f"{amount.amount:,.0f} {amount.name} = about ${usd:,.2f} at {rate.source} "
                    f"({rate.per_usd:,.2f} {amount.code} per dollar"
                    + (f", {rate.as_of}" if rate.as_of else "") + ")")
        out.append(f"${usd:.2f}")
    out.append(text[cursor:])
    return "".join(out), said




_IN_CURRENCY: Final = re.compile(
    r"\bin\s+(?:(?:indonesian|vietnamese|nigerian|turkish|brazilian|thai|malaysian|korean|"
    r"chinese|philippine|mexican|south\s+african|kenyan|ghanaian|polish|egyptian)\s+)?"
    r"(?P<c>" + "|".join(p for p, _, _ in _CURRENCIES) + r")\b", re.I)


def price_in_local(text: str, prior: Sequence[str] = ()) -> list[str] | None:
    """A coin's or a stock's price asked in a local currency: "what is the price of BTC in
    rupiah" got the dollar quote alone (round 45 re-ask). USDT is taken at one dollar, which the
    answer says, and the rate is named with its source and date."""
    m = _IN_CURRENCY.search(text)
    if m is None or amounts(text):
        return None
    code, name = next((c, n) for p, c, n in _CURRENCIES if re.fullmatch(p, m.group("c"), re.I))
    if code == "INR":
        return None  # rupees have their own reader, with the rupee rate's own source
    from argus.lui.research.parse import research_symbols

    named = research_symbols(text[:m.start()] + " " + text[m.end():])[0]
    if not named and prior:
        named = research_symbols(prior[-1])[0]
    if len(named) != 1:
        return None
    from argus.market.bitget import fetch_tickers
    from argus.market.fx_rates import usd_rate

    try:
        ticker = fetch_tickers().get(named[0])
    except Exception:
        ticker = None
    rate = usd_rate(code)
    coin = named[0].removesuffix("USDT")
    if ticker is None or rate is None:
        return [f"Bottom line: {coin}'s price in {name} cannot be given just now: "
                + ("the Bitget price" if ticker is None else f"a {code} rate") + " did not answer."]
    last = float(ticker.last)
    local = last * rate.per_usd
    return [f"Bottom line: {coin} is about {local:,.0f} {name} — {last:,.2f} USDT on Bitget times "
            f"{rate.per_usd:,.2f} {code} per dollar ({rate.source}"
            + (f", {rate.as_of}" if rate.as_of else "") + ").",
            "USDT is taken at one US dollar; a local exchange quoting in "
            f"{name} can differ by its own spread and fees.",
            f"Data: Bitget last price for {named[0]}; {rate.source} for {code}."]


__all__ = ["LocalAmount", "amounts", "in_dollars", "price_in_local"]
