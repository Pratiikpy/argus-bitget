"""Company names a trader types ("Micron", "Ford Motor", "Ford"), read as the ticker they mean.

A judge in round 25 asked "When does Micron report next…" and "how did Ford do over the same
stretch?" and got "'the stock' names no instrument" and the previous name's answer: the console
knew fifteen company names by hand (`universe.ALIASES`), and Micron is listed on Bitget as MUUSDT.

The names come from SEC's own register, ``https://www.sec.gov/files/company_tickers.json`` (every
US-listed issuer's ticker and conformed name), frozen into ``data/company_names.json`` by
:func:`freeze` so answers never wait on SEC. Each name gives two keys:

- the **full name** without its legal suffix ("MICRON TECHNOLOGY", "FORD MOTOR") — always kept,
  because a two-word company name is not an English phrase anyone types by accident;
- the **first word** ("MICRON", "FORD") — kept only when it names exactly one issuer, is four
  letters or more, and is not on :data:`_COMMON`, the words that are also ordinary English or
  trading vocabulary ("TARGET", "BLOCK", "STRATEGY", "AMERICAN", "UNITED"). A first word is
  matched only as written with a capital letter, so "target price" never becomes Target.

`listed_name` answers with a Bitget contract when the issuer trades there; `us_ticker` answers
with the US ticker for issuers Bitget does not list, for engines that can read Yahoo Finance
(the period engine) instead.
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Any

from argus.truth.paths import DATA_DIR

NAMES_PATH = DATA_DIR / "company_names.json"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
FAMOUS = 1500
"""SEC orders ``company_tickers.json`` by market value, largest first. A one-word name ("Boeing")
counts only for these issuers: below them the one-word names are words ("Hello", "Total") and
crypto names ("Bitcoin", "Solana") registered by small issuers."""

_SUFFIX = re.compile(
    r"[,.]?\s+(?:INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|PLC|LLC|LP|L\.P|N\.V|NV|"
    r"S\.A|SA|AG|SE|HOLDINGS?|GROUP|TRUST|CLASS\s+[A-C]|THE|/[A-Z]{2}/?|DE|ADR|SPONSORED)\.?\b.*$")
_COMMON = frozenset({
    "AMERICAN", "UNITED", "FIRST", "GENERAL", "NATIONAL", "INTERNATIONAL", "GLOBAL", "NORTHERN",
    "SOUTHERN", "WESTERN", "EASTERN", "CENTRAL", "PACIFIC", "ATLANTIC", "BANK", "CAPITAL", "TRUST",
    "TARGET", "BLOCK", "STRATEGY", "CIRCLE", "UNITY", "SNOW", "SQUARE", "DIGITAL", "ADVANCED",
    "MICRO", "APPLIED", "DATA", "ENERGY", "POWER", "HEALTH", "MEDICAL", "PHARMA", "THERAPEUTICS",
    "TECHNOLOGIES", "TECHNOLOGY", "SYSTEMS", "SOLUTIONS", "PARTNERS", "INDUSTRIES", "RESOURCES",
    "PROPERTIES", "REALTY", "HOLDING", "FINANCIAL", "INVESTMENT", "INVESTORS", "FUND", "INCOME",
    "GROWTH", "VALUE", "EQUITY", "SELECT", "SMART", "BEST", "GOLD", "SILVER", "OIL", "GAS", "COIN",
    "OPEN", "CLOSE", "HIGH", "LOW", "LONG", "SHORT", "CALL", "PUT", "RISK", "SAFE", "STOCK",
    "MARKET", "TRADE", "TRADING", "SPOT", "FUTURE", "FUTURES", "CASH", "MONEY", "PRICE", "RATE",
    "HOME", "LIFE", "WORLD", "NEW", "NEXT", "ONE", "TWO", "BIG", "BLUE", "GREEN", "RED", "BLACK",
    "WHITE", "SUN", "STAR", "SKY", "LIGHT", "CORE", "PRIME", "ROYAL", "STANDARD", "UNION",
    "SECURITY", "SECURITIES", "MORGAN", "STATE", "STREET", "CITY", "COMMUNITY", "FEDERAL",
    "PEOPLE", "PEOPLES", "CHINA", "JAPAN", "INDIA", "CANADA", "TEXAS", "CALIFORNIA", "FLORIDA",
    "NETWORK", "CLOUD", "LABS", "BIO", "GENE", "CELL", "VISION", "SPACE", "ROCKET", "MOTION",
    "ELECTRIC", "MOTORS", "AUTO", "FOOD", "FOODS", "BRANDS", "RETAIL", "STORES", "CHURCH",
    "HUMAN", "ACTIVE", "ALPHA", "BETA", "DELTA", "SIGMA", "OMEGA", "APEX", "SUMMIT", "PEAK",
    "LIBERTY", "FREEDOM", "PIONEER", "FRONTIER", "HORIZON", "VERTEX", "MATRIX", "NOVA", "TERRA",
    "WHAT", "WHEN", "WHERE", "WHICH", "THIS", "THAT", "THESE", "THOSE", "HAVE", "WILL", "SHOULD",
    "WOULD", "COULD", "ABOUT", "AFTER", "BEFORE", "SINCE", "OVER", "UNDER", "MORE", "LESS", "MOST",
    "MAKE", "TAKE", "GIVE", "SHOW", "TELL", "EARNINGS", "REPORT", "REPORTS", "SHARES",
    "PERFORMANCE",
    "DOW", "TOTAL", "HOLD", "WATERS", "CHARLES", "COOPER", "INSPIRE", "THESIS", "CRYPTO", "HELLO",
    "ABOVE", "BITCOIN", "ETHEREUM", "SOLANA", "RIPPLE", "DOGE", "PEPE", "SHIB", "TETHER", "STABLE",
    "NASDAQ", "SAFETY", "CARE", "MATCH", "BOOKING", "CROWN", "EQUITABLE", "PROGRESSIVE",
    "PRINCIPAL", "AFFIRM", "TOAST", "ELASTIC",
    # listed companies whose first word is ordinary English or trading talk: "is that bullish?"
    # was read as Bullish Inc (BLSH) (round 25); their tickers still resolve
    "BULLISH", "BEARISH", "ANALOG", "BENDING", "BLOOM", "CARNIVAL", "COCA", "CREDO", "FLEX",
    "INTUITIVE", "PALO", "PLANET", "TOWER", "TWIST", "VALE", "COHERENT", "CORNING", "SHOP",
    "GLOBE", "FIVE", "FOUR", "THREE", "SEVEN", "TEN", "HUNDRED", "MILLION", "BILLION",
})


def _clean(title: str) -> str:
    upper = re.sub(r"\s+", " ", title.upper().replace("&", " AND ")).strip()
    return _SUFFIX.sub("", upper).strip(" ,.")


def build(rows: dict[str, Any]) -> dict[str, dict[str, str]]:
    """The two key sets from SEC's ``company_tickers.json`` rows: full names and safe first
    words, each to its ticker. Pure, so a frozen snapshot rebuilds the same file."""
    full: dict[str, str] = {}
    single: dict[str, str] = {}
    firsts: dict[str, set[str]] = {}
    primary: dict[str, str] = {}
    for rank, row in enumerate(rows.values()):
        ticker = str(row.get("ticker", "")).upper().replace("-", ".")
        title = str(row.get("title", ""))
        if not ticker or not title:
            continue
        # SEC lists an issuer's share classes as separate rows under one CIK, the main listing
        # first; a name means the issuer, so it maps to that first ticker
        issuer = str(row.get("cik_str", ticker))
        ticker = primary.setdefault(issuer, ticker)
        name = _clean(title)
        if not name:
            continue
        words = name.split()
        if len(words) >= 2:
            full.setdefault(name, ticker)
        famous = rank < FAMOUS
        if not famous or not words[0].isalpha() or len(words[0]) < 4 or words[0] in _COMMON:
            continue
        if len(words) == 1:
            single.setdefault(name, ticker)
        else:
            firsts.setdefault(words[0], set()).add(issuer)
    first_words = {word: primary[next(iter(issuers))] for word, issuers in firsts.items()
                   if len(issuers) == 1 and word not in single}
    first_words.update(single)
    return {"full": dict(sorted(full.items())), "first": dict(sorted(first_words.items()))}


def freeze(path: Path = NAMES_PATH) -> int:  # pragma: no cover - network
    from argus.market.evidence import FEED_USER_AGENT
    from argus.truth import http

    rows = http.fetch_json(SEC_TICKERS, timeout=30, headers={"User-Agent": FEED_USER_AGENT})
    built = build(rows)
    path.write_text(json.dumps({"source": SEC_TICKERS, **built}, indent=0, sort_keys=True),
                    encoding="utf-8", newline="\n")
    return len(built["full"]) + len(built["first"])


@cache
def _names() -> dict[str, dict[str, str]]:
    try:
        raw = json.loads(NAMES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"full": {}, "first": {}}
    return {"full": raw.get("full", {}), "first": raw.get("first", {})}


def us_ticker(text: str) -> tuple[str, str] | None:
    """The first company named in ``text`` and its US ticker, as (name as typed, ticker)."""
    names = _names()
    words = re.findall(r"[A-Za-z][A-Za-z.&'-]*", text)
    for size in (4, 3, 2):
        for i in range(len(words) - size + 1):
            said = " ".join(words[i:i + size])
            ticker = names["full"].get(_clean(said))
            if ticker and len(said) >= 6:
                return said, ticker
    for word in words:
        if len(word) >= 4 and word[0].isupper() and word[1:].islower() and (
                word.upper() not in _COMMON):
            ticker = names["first"].get(word.upper())
            if ticker:
                return word, ticker
    return None


def names_for(ticker: str) -> list[str]:
    """The one-word names a headline uses for ``ticker`` ("NVDA" -> ["Nvidia"]), from the same
    table :func:`us_ticker` reads the other way."""
    wanted = ticker.upper()
    return [word.title() for word, held in _names()["first"].items() if held == wanted]


def ticker_for_full_name(name: str) -> str | None:
    """The SEC ticker registered under a company's full name ("GE Vernova" is GEV), or None."""
    found = _names()["full"].get(_clean(name))
    return str(found) if found else None


def listed_name(word: str) -> str | None:
    """A one-word company name ("Micron") as its Bitget contract, when Bitget lists the issuer."""
    from argus.market import universe

    ticker = _names()["first"].get(word.upper()) or _names()["full"].get(_clean(word))
    if word.upper() in _COMMON:
        return None
    if not ticker:
        return None
    for symbol in (f"{ticker}USDT", f"{ticker}STOCKUSDT"):
        if universe.is_equity(symbol):
            return symbol
    return None


def main() -> int:  # pragma: no cover - CLI
    count = freeze()
    print(f"{count} company-name keys written to {NAMES_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
