"""A stress test against a named historical crash: what the asset actually did in that window.

"Stress test a long SOL position against a 30% overnight crash like the May 2022 Terra/Luna
contagion week — does your thesis survive it?" was answered with the capability register in round
30, and the same question on MSTR with a generic QQQ -10% shock in round 31 — four rounds running
the named event was never looked up. Every event here has a fixed window, said in the answer, and
the asset's own daily closes in that window (Yahoo Finance, ``market.equity_history.daily``) give
two figures: the move from the close before the window to its last close, and the deepest close
inside it. When the asset did not trade yet, the answer says so rather than borrowing another's.

The windows are the dates the events are known by, not fitted to make any asset look worse:

* Terra/Luna: 6 to 12 May 2022 (UST lost its peg on 7-9 May; LUNA collapsed by 12 May).
* Celsius and Three Arrows: 10 to 18 June 2022 (Celsius froze withdrawals on 12 June).
* FTX: 4 to 10 November 2022 (the run began on 6 November; FTX paused withdrawals on 8 November).
* Silicon Valley Bank: 8 to 13 March 2023 (SVB failed on 10 March; USDC broke its peg on 11 March).
* COVID: 5 to 23 March 2020 (the fastest bear market on record; "Black Thursday" was 12 March).
* The yen carry unwind: 31 July to 5 August 2024.
* Lehman: 12 September to 10 October 2008.
* The May 2021 crypto crash: 12 to 19 May 2021.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Final

from argus.lui.trace import trace_module

EVENTS: Final[tuple[tuple[re.Pattern[str], str, date, date], ...]] = tuple(
    (re.compile(pattern, re.I), name, start, end) for pattern, name, start, end in (
        (r"\bterra\b|\bluna\b|\bust\s+(?:de-?peg|collapse)|\bmay\s+2022\b",
         "the Terra/Luna collapse", date(2022, 5, 6), date(2022, 5, 12)),
        (r"\bcelsius\b|\bthree\s+arrows\b|\b3ac\b|\bjune\s+2022\b",
         "the Celsius and Three Arrows week", date(2022, 6, 10), date(2022, 6, 18)),
        (r"\bftx\b|\bnovember\s+2022\b|\bnov\s+2022\b", "the FTX collapse",
         date(2022, 11, 4), date(2022, 11, 10)),
        (r"\bsvb\b|\bsilicon\s+valley\s+bank\b|\bmarch\s+2023\b", "the Silicon Valley Bank failure",
         date(2023, 3, 8), date(2023, 3, 13)),
        (r"\bcovid\b|\bcorona\w*\b|\bmarch\s+2020\b|\bblack\s+thursday\b", "the COVID crash",
         date(2020, 3, 5), date(2020, 3, 23)),
        (r"\byen\s+carry\b|\bcarry[\s-]+trade\s+unwind\b|\baugust\s+2024\b|\baug\s+2024\b",
         "the yen carry unwind", date(2024, 7, 31), date(2024, 8, 5)),
        (r"\blehman\b|\b2008\s+(?:crash|crisis)\b|\bfinancial\s+crisis\b", "the Lehman collapse",
         date(2008, 9, 12), date(2008, 10, 10)),
        (r"\bmay\s+2021\b|\bchina\s+(?:crypto\s+)?ban\b", "the May 2021 crypto crash",
         date(2021, 5, 12), date(2021, 5, 19)),
    ))


def named_event(text: str) -> tuple[str, date, date] | None:
    for pattern, name, start, end in EVENTS:
        if pattern.search(text):
            return name, start, end
    return None


def _ticker(symbol: str) -> str:
    from argus.lui.research.parse import is_us_equity

    base = symbol.removesuffix("USDT")
    return base if is_us_equity(symbol) else f"{base}-USD"


def measured(symbol: str, start: date, end: date) -> tuple[float, float, date, date] | None:
    """(move over the window, deepest close in it against the close before, first, last day), or
    None when the asset has no closes before the window."""
    from argus.market.equity_history import daily

    try:
        days = daily(_ticker(symbol))
    except Exception:
        return None
    before = [d for d in days if d.day < start]
    inside = [d for d in days if start <= d.day <= end]
    if not before or not inside:
        return None
    base = before[-1].close
    if base <= 0:
        return None
    return (inside[-1].close / base - 1, min(d.close for d in inside) / base - 1, before[-1].day,
            inside[-1].day)


_SHOCK: Final = re.compile(r"\b(?P<pct>\d{1,2}(?:\.\d+)?)\s*%\s+(?:overnight\s+|one[\s-]day\s+|"
                           r"single[\s-]day\s+)?(?:crash|drop|fall|plunge|decline)\b", re.I)
_DOLLARS: Final = re.compile(r"\$\s?(?P<a>\d(?:[\d,]*\d)?(?:\.\d+)?)\s*(?P<k>k|m)?\b", re.I)


def lines(text: str, symbols: tuple[str, ...]) -> list[str] | None:
    """Each named asset through the named event, with a stated shock and size set beside it."""
    event = named_event(text)
    if event is None or not symbols:
        return None
    name, start, end = event
    side = "short" if re.search(r"\bshort\b", text, re.I) else "long"
    shock = _SHOCK.search(text)
    dollars = _DOLLARS.search(text)
    stake = (float(dollars.group("a").replace(",", "")) * {"k": 1e3, "m": 1e6}.get(
        (dollars.group("k") or "").lower(), 1.0)) if dollars else None
    rows = []
    for symbol in symbols[:3]:
        found = measured(symbol, start, end)
        base = symbol.removesuffix("USDT")
        if found is None:
            rows.append(f"{base} has no daily closes before {start:%d %b %Y} in the history read "
                        f"here, so it cannot be put through {name} — it was not trading yet, or "
                        f"not on this source.")
            continue
        move, deepest, first, last = found
        pnl = move if side == "long" else -move
        worst = deepest if side == "long" else None
        money = (f" — on ${stake:,.0f}, {'+' if pnl >= 0 else '-'}${abs(stake * pnl):,.0f} by the "
                 f"end" + (f" and -${abs(stake * worst):,.0f} at the deepest close"
                           if worst is not None and worst < 0 else "") if stake else "")
        rows.append(f"{base} through {name} ({first:%d %b} close to {last:%d %b %Y}): "
                    f"{move:+.1%} by the end, {deepest:+.1%} at its deepest close; a {side} "
                    f"{'lost' if pnl < 0 else 'made'} {abs(pnl):.1%}{money}.")
    lead_name = symbols[0].removesuffix("USDT")
    first_found = measured(symbols[0], start, end)
    verdict = ""
    if first_found is not None:
        move = first_found[0] if side == "long" else -first_found[0]
        verdict = (f"a {side} in {lead_name} {'lost' if move < 0 else 'made'} {abs(move):.0%} "
                   f"through {name}")
        if shock is not None:
            stated = float(shock.group("pct")) / 100
            actual = abs(min(first_found[1], first_found[0]))
            verdict += (f" — {'worse' if actual > stated else 'milder'} than the {stated:.0%} "
                        f"crash you set beside it")
    lines_out = [f"Bottom line: {verdict or f'{name} measured on its own dates'} (daily closes, "
                 f"Yahoo Finance).", *rows]
    if re.search(r"\b(?:survive|thesis|hold\s+up|still\s+work)\b", text, re.I):
        lines_out.append("Whether a thesis survives is the size question: a position sized so "
                         "that a fall like this is a loss you would accept survives it; one that "
                         "is not gets closed at the bottom. Leverage turns the deepest close above "
                         "into a liquidation line — at 3x, a 33% fall closes a long.")
    lines_out.append("These are what happened in those windows, not a forecast of the next one.")
    return lines_out


__all__ = ["EVENTS", "lines", "measured", "named_event"]

trace_module(globals())
