"""What prediction markets price about a name: Polymarket's open markets, read keyless.

A research answer lists what the venue, the filings and the tape say. None of those is a crowd
putting money on an outcome — "will Tesla announce a SpaceX merger by December", "will bitcoin
reach $95,000 in September", "gold up or down today". Polymarket prices exactly that, publicly and
without a key. optic-bitget (neromtoobad, Season 2, MIT) carries it as an evidence row; this module
takes its selection rules and leaves its keyword model behind:

- Gamma's ``/public-search`` fuzzy-matches junk to CLOSED markets, so relevance is enforced here:
  the market must be active, not closed, not past its end date, above a volume floor (optic's
  ``VOLUME_FLOOR = 10_000``, `src/lenses/prediction.ts:12`), and its question must name the subject
  as a whole word. A wrong match is worse than none.
- optic asks a model for the search keywords (`predictionTerms`). ARGUS already knows what it is
  asking about — a listed contract — so the terms come from a table of names (ticker, company,
  common name), which costs nothing and cannot hallucinate a subject.

Polymarket also lists daily "<name> Up or Down on <date>" markets — the crowd's price for the
question `desk/odds.py` answers from the record. On 2026-09-25 NVIDIA's and Tesla's had $170 and
$140 traded, so they fall under the floor and are not quoted; one order moves them.

A price is a probability only loosely: it carries fees, the spread and the time value of locked
money, and thin markets move on one order. The lines say "priced at", show the volume, and never
blend the figure into ARGUS's own measurements.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

GAMMA = "https://gamma-api.polymarket.com"
VOLUME_FLOOR = 10_000.0
"""optic-bitget's floor (`src/lenses/prediction.ts:12`): below it a price is one trader's view."""
INFORMATIVE = (0.03, 0.97)
"""A market priced at 1% or 99% says the outcome is settled in the crowd's mind; a ladder of those
("will bitcoin reach $100,000 in September", 1%) buries the one rung that carries information. The
busiest markets inside this band are shown first; the tails only when nothing else is open."""
CACHE_SECONDS = 600.0
MAX_LINES = 3

SEARCH_TERMS: dict[str, tuple[str, ...]] = {
    "BTC": ("bitcoin", "btc"), "ETH": ("ethereum", "ether", "eth"), "SOL": ("solana",),
    "XRP": ("xrp", "ripple"), "DOGE": ("dogecoin", "doge"), "BNB": ("bnb",),
    "XAU": ("gold",), "XAG": ("silver",), "CL": ("crude oil", "oil", "wti"), "BZ": ("brent",),
    "NATGAS": ("natural gas",), "COPPER": ("copper",),
    "SP500": ("s&p 500", "s&p"), "SPY": ("s&p 500", "s&p"), "QQQ": ("nasdaq",),
    "NDX100": ("nasdaq",),
    "NVDA": ("nvidia",), "TSLA": ("tesla",), "AAPL": ("apple",), "MSFT": ("microsoft",),
    "GOOGL": ("google", "alphabet"), "AMZN": ("amazon",), "META": ("meta",),
    "MSTR": ("microstrategy", "strategy (mstr)", "saylor"), "COIN": ("coinbase",),
    "AMD": ("amd",), "PLTR": ("palantir",), "NFLX": ("netflix",), "INTC": ("intel",),
    "HOOD": ("robinhood",), "CRCL": ("circle",), "ORCL": ("oracle",), "AVGO": ("broadcom",),
}
"""What a prediction-market question calls each name. Unlisted names are searched by ticker."""


class PredictionError(RuntimeError):
    """Polymarket could not be read; the answer says so rather than showing nothing."""


@dataclass(frozen=True, slots=True)
class Market:
    question: str
    yes_price: float
    change_24h: float | None
    volume: float
    ends: str
    url: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_lock = threading.Lock()


def _search(term: str, *, timeout: float = 10.0) -> list[dict[str, Any]]:
    now = time.monotonic()
    with _lock:
        hit = _cache.get(term)
        if hit and now - hit[0] < CACHE_SECONDS:
            return hit[1]
    url = f"{GAMMA}/public-search?" + urllib.parse.urlencode({"q": term, "limit_per_type": 8})
    request = urllib.request.Request(url, headers={"Accept": "application/json",
                                                   "User-Agent": "argus-research/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            events = (json.load(response) or {}).get("events") or []
    except Exception as exc:
        raise PredictionError(
            f"Polymarket search for {term!r} failed: {type(exc).__name__}") from exc
    with _lock:
        _cache[term] = (now, events)
    return events


def _yes(market: dict[str, Any]) -> float | None:
    try:
        return float(json.loads(market.get("outcomePrices") or "[]")[0])
    except (ValueError, IndexError, TypeError):
        return None


def relevant(events: list[dict[str, Any]], terms: tuple[str, ...], *,
             now: datetime | None = None, floor: float = VOLUME_FLOOR) -> list[Market]:
    """Open, liquid markets whose question names one of ``terms`` as a whole word, busiest first."""
    clock = now or datetime.now(UTC)
    patterns = [re.compile(rf"(?<![\w&]){re.escape(t)}(?![\w])", re.I) for t in terms]
    seen: set[str] = set()
    found: list[Market] = []
    for event in events:
        for market in event.get("markets") or []:
            question = str(market.get("question") or "")
            if not market.get("active") or market.get("closed") or question in seen:
                continue
            if not any(p.search(question) for p in patterns):
                continue
            volume = float(market.get("volumeNum") or 0.0)
            yes = _yes(market)
            ends = str(market.get("endDate") or "")
            try:
                ended = bool(ends) and datetime.fromisoformat(ends.replace("Z", "+00:00")) <= clock
            except ValueError:
                ended = False
            if volume < floor or yes is None or ended or not 0.0 <= yes <= 1.0:
                continue
            seen.add(question)
            change = market.get("oneDayPriceChange")
            found.append(Market(
                question=question, yes_price=yes,
                change_24h=float(change) if isinstance(change, (int, float)) else None,
                volume=volume, ends=ends[:10],
                url=f"https://polymarket.com/market/{market.get('slug') or ''}"))
    low, high = INFORMATIVE
    found.sort(key=lambda m: (not low <= m.yes_price <= high, -m.volume))
    return found


def markets_for(symbol: str, *, search: Any = _search, now: datetime | None = None,
                floor: float = VOLUME_FLOOR) -> list[Market]:
    """The liquid open markets about the name behind a Bitget symbol."""
    base = symbol.removesuffix("USDT")
    base = base.removesuffix("STOCK") if base.endswith("STOCK") else base
    terms = SEARCH_TERMS.get(base, (base.lower(),))
    events: list[dict[str, Any]] = []
    for term in terms[:2]:
        events.extend(search(term))
    return relevant(events, terms, now=now, floor=floor)


def lines_for(symbol: str, *, search: Any = _search, now: datetime | None = None,
              name: str | None = None) -> list[str]:
    """Up to three lines naming what the busiest relevant markets price, or nothing."""
    try:
        markets = markets_for(symbol, search=search, now=now)
    except PredictionError:
        return []
    label = name or symbol.removesuffix("USDT")
    lines = []
    for m in markets[:MAX_LINES]:
        move = (f", {m.change_24h * 100:+.0f} points in 24h"
                if m.change_24h is not None and abs(m.change_24h) >= 0.005 else "")
        lines.append(f"Prediction market on {label} (Polymarket): \"{m.question}\" priced at "
                     f"{m.yes_price:.0%}{move} — ${m.volume:,.0f} traded, closes {m.ends}.")
    return lines

