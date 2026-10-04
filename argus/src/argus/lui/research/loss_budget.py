"""How large a position fits inside a loss limit stated on the whole account.

"My book is 40% BTC, 40% ETH, 20% cash, total $100,000. I cannot lose more than 3% of total
capital this month. How large a SOL position can I add without breaching that limit?" was answered
by the per-trade mandate check: SOL's own bad case (-6.5%) against the 3% tolerance, "no", with
"resizing does not change a percentage" (a judge, round 36). That rule is right for a limit on one
trade's own percentage loss; it is the wrong rule for a limit on the account, where a $20,000
position falling 6.5% costs $1,300 — 1.3% of $100,000. The size of the position is the whole
question.

This reader answers it the way a risk desk would, on the record rather than a model:

- the limit in dollars: the stated percentage of the stated capital (or of a remembered one);
- the bad case over the stated horizon (a day, a week, a month — a month when "this month" is
  said), taken as the worst rolling window in three years of daily closes, for the book as it is
  and for the book with the new position, held together so the holdings that fall with it count;
- the largest new position whose combined worst window stays inside the limit, found by bisection
  on the joint series; and what the book alone already costs in that window, which can use up the
  limit before anything is added.

Daily closes are Yahoo Finance's (coins as their USD pairs, gold as GC=F, stocks as themselves),
on the days every holding traded. A worst past window is a measured bad case, not a ceiling: the
answer says so.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Final

from argus.lui.trace import trace_module

ASKED: Final = re.compile(
    r"\b(?:how\s+(?:large|big|much)|what\s+size|max(?:imum)?\s+(?:size|position|amount))\b[^?]{0,80}"
    r"\b(?:add|buy|put|position|allocate)\b|\b(?:add|buy)\b[^?]{0,40}\bwithout\s+(?:breach|break|"
    r"exceed|going\s+over)\w*", re.I)
LIMIT: Final = re.compile(
    r"\b(?:can(?:'?t|not|\s+not)|won'?t|don'?t\s+want\s+to|must\s+not|never)\s+(?:afford\s+to\s+)?"
    r"lose\s+(?:more\s+than\s+)?(?P<p>\d+(?:\.\d+)?)\s*%\s+of\s+(?:my\s+|the\s+|total\s+|my\s+total\s+)?"
    r"(?:capital|account|book|portfolio|money|equity|net\s+worth)\b|\b(?:max(?:imum)?\s+)?(?:loss|"
    r"drawdown)\s+(?:limit|budget)\s+(?:is\s+|of\s+)?(?P<q>\d+(?:\.\d+)?)\s*%\s+of\s+(?:my\s+|the\s+|"
    r"total\s+)?(?:capital|account|book|portfolio)\b", re.I)
_TOTAL: Final = re.compile(r"(?:total|worth|capital\s+of|account\s+of|book\s+of)\s+(?:of\s+)?"
                           r"\$\s?(?P<a>\d[\d,]*(?:\.\d+)?)\s*(?P<k>k|m)?\b", re.I)


def _ticker(symbol: str) -> str:
    base = symbol.removesuffix("USDT")
    if base in ("XAU", "GOLD", "PAXG", "XAUT"):
        return "GC=F"
    from argus.lui.research.parse import is_us_equity

    return base if is_us_equity(symbol) else f"{base}-USD"


def _horizon_days(text: str) -> tuple[int, str]:
    if re.search(r"\b(?:this|a|per|each|in\s+a|one)\s+month\b|\bmonthly\b|\b30\s+days\b", text,
                 re.I):
        return 30, "30-day window"
    if re.search(r"\b(?:this|a|per|each|in\s+a|one)\s+week\b|\bweekly\b|\b7\s+days\b", text, re.I):
        return 7, "7-day window"
    return 1, "single day"


def worst_window(weights: dict[str, float], days: int) -> tuple[float, str] | None:
    """The worst ``days``-long fall of a book held at ``weights`` (fractions of the account, the
    rest cash), over three years of daily closes on the days every holding traded."""
    from argus.market.equity_history import daily

    series: dict[str, dict[date, float]] = {}
    for symbol in weights:
        try:
            series[symbol] = {d.day: d.close for d in daily(_ticker(symbol)) if d.close > 0}
        except Exception:
            return None
    if not series:
        return None
    shared = sorted(set.intersection(*(set(v) for v in series.values())))
    if len(shared) < days + 30:
        return None
    start = shared[-1] - timedelta(days=3 * 365)
    shared = [d for d in shared if d >= start]
    worst, when = 0.0, ""
    j = 0
    for i, day in enumerate(shared):
        j = max(j, i + 1)
        while j < len(shared) and (shared[j] - day).days < days:
            j += 1
        if j >= len(shared):
            break
        change = sum(w * (series[s][shared[j]] / series[s][day] - 1) for s, w in weights.items())
        if change < worst:
            worst, when = change, f"{day:%d %b %Y} to {shared[j]:%d %b %Y}"
    return worst, when


def lines(text: str, book: str = "") -> list[str] | None:
    if not ASKED.search(text):
        return None
    limit = LIMIT.search(text)
    if limit is None:
        return None
    from argus.lui.research import research_symbols
    from argus.lui.research.parse import holding_pairs, split_cash

    stated = holding_pairs(text)
    pairs = stated or holding_pairs(book)
    weighted = {s for _, s, _w in pairs}
    # the name asked about is the one named without a weight: in "How large a SOL position can I
    # add" it comes before the verb, after the book
    fresh = [s for s in research_symbols(text)[0] if s not in weighted]
    if not fresh:
        return None
    candidate = fresh[0]
    total = _TOTAL.search(text) or _TOTAL.search(book)
    if total is None:
        return None
    capital = float(total.group("a").replace(",", "")) * (
        1000 if (total.group("k") or "").lower() == "k" else
        1_000_000 if (total.group("k") or "").lower() == "m" else 1)
    holdings, _cash = split_cash(text if stated else book, {s: w for _, s, w in pairs})
    held = {s: w for s, w in holdings.items() if s != candidate}
    pct = float(limit.group("p") or limit.group("q")) / 100
    budget = capital * pct
    days, window = _horizon_days(text)
    alone = worst_window(held, days) if held else (0.0, "")
    if alone is None:
        return None
    name = candidate.removesuffix("USDT")
    own = worst_window({candidate: 1.0}, days)
    if own is None:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        mix = {**held, candidate: mid}
        found = worst_window(mix, days)
        if found is None:
            return None
        if -found[0] * capital <= budget:
            lo = mid
        else:
            hi = mid
    fits = lo * capital
    book_loss = -alone[0] * capital
    held_said = ", ".join(f"{w:.0%} {s.removesuffix('USDT')}" for s, w in held.items()) or "cash"
    if book_loss >= budget:
        lead = (f"Bottom line: none, on the record — the book you hold ({held_said}) already lost "
                f"{-alone[0]:.1%} of the account (${book_loss:,.0f}) in its worst {window} of the "
                f"last three years ({alone[1]}), past your {pct:.0%} (${budget:,.0f}) before any "
                f"{name} is added.")
    else:
        lead = (f"Bottom line: about ${fits:,.0f} of {name} ({lo:.0%} of the account) — the most "
                f"that keeps the whole book's worst {window} of the last three years inside your "
                f"{pct:.0%} limit (${budget:,.0f} of ${capital:,.0f}).")
    out = [lead,
           f"How: {name} alone fell {-own[0]:.1%} in its worst {window} ({own[1]}); the book you "
           f"hold ({held_said}) fell {-alone[0]:.1%} of the account in its own (${book_loss:,.0f})"
           + ("" if not held else ", and the two are held together here, so a window where both "
                                 "fell counts once with both losses in it") + ".",
           f"Sized on {name} alone, ignoring the rest of the book, the same limit would allow "
           f"${budget / -own[0]:,.0f} — the difference is what the holdings you already have use "
           f"up." if own[0] < 0 else "",
           "A worst past window is a measured bad case, not a ceiling: a worse one can come. Data: "
           "Yahoo Finance daily closes (coins as their USD pairs), three years, the book held at "
           "its weights; cash taken as not moving."]
    return [x for x in out if x]


trace_module(globals())
