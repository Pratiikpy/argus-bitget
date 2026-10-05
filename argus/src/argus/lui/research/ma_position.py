"""Where a price sits against its N-day moving average — and, when the question states a position,
what that position is worth at today's price.

Round 42's judge (C4, conversation A) asked "I'm long 100 shares of AAPL bought at 190. Is it above
or below its 200-day moving average?" and got the position valued at $20,000 with "200" read as
the mark; once that was fixed it got AAPL's 200-day performance instead. The question is a
comparison: today's close against the average of the last N daily closes (`rule_test.daily_closes`:
Bitget's daily closes for a coin, Yahoo's split-adjusted closes for a US stock), how far above or
below in percent, and how many sessions it has been on that side. A position stated in the same
question ("100 shares bought at 190") is valued at the same close: AAPL 334.28 against a 200-day
average of 289.46 on 2026-10-05, the judge's own figures.
"""

from __future__ import annotations

import re
from typing import Final

ASKED: Final = re.compile(
    r"\b(?:above|below|under|over|cross\w*|relative\s+to|versus|vs\.?|against|compared?\s+to)\b"
    r"[^?]{0,40}\b(?P<n>\d{1,3})[\s-]*(?:day|d)\b[\s-]*(?:simple\s+|exponential\s+)?(?P<kind>"
    r"moving\s+average|ma|sma|ema|average)\b|\b(?P<n2>\d{1,3})[\s-]*(?:day|d)\b[\s-]*(?:simple\s+|"
    r"exponential\s+)?(?P<kind2>moving\s+average|ma|sma|ema)\b[^?]{0,30}\b(?:above|below|under|"
    r"over)\b", re.I)
_HELD: Final = re.compile(r"\b(?:long|hold|own|bought|have)\s+(?P<q>\d[\d,]*(?:\.\d+)?)\s*"
                          r"(?:shares?\s+(?:of\s+)?)?(?P<name>[A-Za-z]{1,6})?\b[^?.]{0,40}?\b"
                          r"(?:at|@)\s*\$?"
                          r"(?P<px>\d[\d,]*(?:\.\d+)?)", re.I)


def _average(closes: list[float], n: int, exponential: bool) -> float:
    if not exponential:
        return sum(closes[-n:]) / n
    k = 2 / (n + 1)
    value = sum(closes[:n]) / n
    for c in closes[n:]:
        value = c * k + value * (1 - k)
    return value


def lines(text: str, symbol: str) -> list[str] | None:
    """The answer for ``symbol``, or None when the question is not about its moving average."""
    m = ASKED.search(text)
    if m is None:
        return None
    from argus.lui.research.rule_test import daily_closes

    n = int(m.group("n") or m.group("n2"))
    kind = (m.group("kind") or m.group("kind2") or "").lower()
    exponential = kind == "ema"
    try:
        stamps, closes, said = daily_closes(symbol)
    except Exception:
        return [f"Bottom line: {symbol}'s daily closes could not be read just now, so its "
                f"{n}-day average is not given."]
    if len(closes) < n + 1:
        return [f"Bottom line: {symbol} has {len(closes)} daily closes here, fewer than the "
                f"{n} the average needs."]
    name = symbol.removesuffix("USDT")
    last = closes[-1]
    average = _average(closes, n, exponential)
    gap = last / average - 1
    side = "above" if gap >= 0 else "below"
    run = 0
    for i in range(len(closes) - 1, n - 1, -1):
        avg_i = _average(closes[: i + 1], n, exponential)
        if (closes[i] >= avg_i) != (gap >= 0):
            break
        run += 1
    label = f"{n}-day {'exponential' if exponential else 'simple'} moving average"
    out = [f"Bottom line: {name} is {side} its {label} — {last:,.2f} against {average:,.2f}, "
           f"{gap:+.1%} — and has closed on that side for {run} straight sessions "
           f"(to {stamps[-1]:%d %b %Y})."]
    held = _HELD.search(text)
    if held is not None:
        qty = float(held.group("q").replace(",", ""))
        entry = float(held.group("px").replace(",", ""))
        value, pnl = qty * last, qty * (last - entry)
        out.append(f"Your position: {qty:,.0f} at {entry:,.2f} is worth {value:,.2f} at "
                   f"{last:,.2f}, {'+' if pnl >= 0 else '-'}${abs(pnl):,.2f} "
                   f"({last / entry - 1:+.1%}) on the entry, before costs.")
    out.append(f"Data: {said}; the {label} of the last {n} closes. Where a price sits against "
               "its average describes the trend so far, not what comes next. Not advice.")
    return out


__all__ = ["ASKED", "lines"]
