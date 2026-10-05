"""How markets moved around every FOMC decision — holds included — over the Fed's own published
calendar, for a window the question names.

Round 42's hostile audit asked "How did stocks move 99999 days after each FOMC" and got the rates
dashboard; the same question with "5 days" got the same dashboard. `rate_decisions` measures
*changes* (hikes and cuts, from FRED's DFEDTARU), and most meetings change nothing, so a question
about each meeting needs the meeting calendar itself.

**Meetings.** Each decision day (a meeting's last day) from the Federal Reserve's calendar page,
``federalreserve.gov/monetarypolicy/fomccalendars.htm``, kept in ``data/event_calendar.json`` by
`market/calendar.py` (2021 onwards on that page). Only decisions already past are measured.
Each is labelled hike, cut or hold from DFEDTARU's change dated the day after it.

**The move.** The close the day before the decision to the decision day's close (the 14:00 ET
statement lands inside it), and from that close to the close N trading days later. Yahoo Finance
daily closes. The windows asked are used when they fit inside the record; a window longer than
the calendar's span (99,999 days is about 274 years) is said to be impossible and the standard
1-, 5- and 20-day windows are measured instead.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from datetime import UTC, date, datetime
from statistics import mean, median
from typing import Final

ASKED: Final = re.compile(
    r"\b(?:after|around|following|on|before)\s+(?:each|every|all|the|past|recent)?\s*(?:\w+\s+)?"
    r"(?:fomc|fed)\s*(?:meetings?|decisions?|days?|announcements?)?\b|\b(?:fomc|fed)\s+(?:meeting|"
    r"decision)\s+days?\b", re.I)
_REACTION: Final = re.compile(r"\bmov\w*|\breact\w*|\bdo\b|\bperform\w*|\bhistor\w*|\breturn\w*|"
                              r"\btypical\w*|\busual\w*|\bbehav\w*", re.I)
_WINDOW: Final = re.compile(r"\b(?P<n>\d[\d,]*)\s*(?:trading\s+)?(?:days?|sessions?)\s+(?:after|"
                            r"later|following|on)\b", re.I)
DEFAULT_WINDOWS: Final = (1, 5, 20)
_NAMES: Final = {"stocks": ("the S&P 500 (SPY)", "SPY"), "s&p": ("the S&P 500 (SPY)", "SPY"),
                 "spy": ("the S&P 500 (SPY)", "SPY"), "equities": ("the S&P 500 (SPY)", "SPY"),
                 "nasdaq": ("the Nasdaq-100 (QQQ)", "QQQ"), "qqq": ("the Nasdaq-100 (QQQ)", "QQQ"),
                 "tech": ("the Nasdaq-100 (QQQ)", "QQQ"), "btc": ("BTC", "BTC-USD"),
                 "bitcoin": ("BTC", "BTC-USD"), "crypto": ("BTC", "BTC-USD"),
                 "eth": ("ETH", "ETH-USD"), "gold": ("gold", "GC=F"),
                 "dollar": ("the dollar index (DXY)", "DX-Y.NYB"),
                 "bonds": ("long Treasuries (TLT)", "TLT"), "tlt": ("long Treasuries (TLT)", "TLT")}


def _pct(x: float) -> str:
    text = f"{x:+.2%}"
    return text[1:] if float(text.strip("+-%")) == 0 else text


def lines(text: str, *, today: date | None = None) -> list[str] | None:
    """The meeting-by-meeting study, or None when ``text`` does not ask how a market moved around
    FOMC meetings in general (a hike or cut question is `rate_decisions`')."""
    if not ASKED.search(text) or not _REACTION.search(text):
        return None
    from argus.lui.research.parse import about_the_record

    if about_the_record(text):
        return None  # "why did the desk sit out the FOMC" is the desk's own record
    if re.search(r"\b(?:hikes?|hiked|cuts?|cutting|raises?|lowers?)\b", text, re.I):
        return None
    from argus.lui.research.rate_decisions import fed_changes
    from argus.market.calendar import SNAPSHOT
    from argus.market.equity_history import daily

    today = today or datetime.now(UTC).date()
    try:
        meetings = sorted(date.fromisoformat(d) for d in
                          json.loads(SNAPSHOT.read_text(encoding="utf-8"))["fomc"])
    except (OSError, ValueError, KeyError):
        return ["Bottom line: the FOMC calendar snapshot could not be read, so no meeting can be "
                "measured."]
    meetings = [d for d in meetings if d < today]
    if not meetings:
        return None
    asked = [int(m.group("n").replace(",", "")) for m in _WINDOW.finditer(text)]
    span_days = (today - meetings[0]).days
    notes: list[str] = []
    windows = [n for n in asked if 0 < n <= min(250, span_days * 5 // 7)]
    for n in asked:
        if n not in windows:
            years = n / 365.25
            about = f"(about {years:,.0f} years) " if years >= 2 else ""
            notes.append(f"{n:,} days {about}is longer than the record allows — the Fed's "
                         f"calendar here starts {meetings[0]:%b %Y}, so no decision has that "
                         "much history after it; the standard windows are measured instead.")
    windows = windows or list(DEFAULT_WINDOWS)
    named = []
    for word, pair in _NAMES.items():
        if re.search(rf"\b{re.escape(word)}\b", text, re.I) and pair not in named:
            named.append(pair)
    assets = named or [("the S&P 500 (SPY)", "SPY")]
    try:
        changes = {d: s for d, s in fed_changes()}
    except Exception:
        changes = {}
    kind = {d: ("hike" if changes.get(d, 0) > 0 else "cut" if changes.get(d, 0) < 0 else "hold")
            for d in meetings}
    out: list[str] = []
    leads: list[str] = []
    for label, ticker in assets[:3]:
        try:
            series = sorted((d.day, float(d.close)) for d in daily(ticker))
        except Exception:
            out.append(f"{label}: Yahoo Finance did not return its closes just now.")
            continue
        days = [d for d, _ in series]
        closes = [c for _, c in series]
        rows = []
        for meeting in meetings:
            i = bisect_right(days, meeting) - 1  # the decision day's close, or the last before
            if i < 1 or days[i] != meeting:
                continue
            day_move = closes[i] / closes[i - 1] - 1
            later = {n: closes[i + n] / closes[i] - 1 for n in windows if i + n < len(closes)}
            rows.append((meeting, day_move, later))
        if not rows:
            out.append(f"{label}: no closes cover those decision days.")
            continue
        day_moves = [r[1] for r in rows]
        up = sum(x > 0 for x in day_moves)
        parts = [f"on the decision day a median {_pct(median(day_moves))}, up on {up} of "
                 f"{len(rows)}"]
        for n in windows:
            got = [r[2][n] for r in rows if n in r[2]]
            if got:
                after = "the trading day" if n == 1 else f"{n} trading days"
                parts.append(f"{after} after a median {_pct(median(got))} (mean "
                             f"{_pct(mean(got))}, up {sum(x > 0 for x in got)} of {len(got)})")
        by_kind = []
        for k in ("hold", "hike", "cut"):
            sub = [r[1] for r in rows if kind[r[0]] == k]
            if sub:
                plural = "s" if len(sub) != 1 else ""
                by_kind.append(f"{len(sub)} {k}{plural} {_pct(median(sub))}")
        out.append(f"{label[:1].upper()}{label[1:]} over {len(rows)} FOMC decisions "
                   f"({rows[0][0]:%b %Y} to {rows[-1][0]:%b %Y}): " + "; ".join(parts)
                   + ". Decision day by outcome, median: " + ", ".join(by_kind) + ".")
        n0 = windows[0]
        first = [r[2][n0] for r in rows if n0 in r[2]]
        leads.append(f"{label} moved a median {_pct(median(day_moves))} on decision days and "
                     + (f"{_pct(median(first))} over the "
                        f"{'trading day' if n0 == 1 else f'{n0} trading days'} after"
                        if first else "")
                     + f" (up {up} of {len(rows)} decision days)")
    if not leads:
        return ["Bottom line: no price history could be read for the FOMC decision days just "
                "now."]
    out.insert(0, "Bottom line: " + "; ".join(leads) + " — a base rate across every meeting, "
                  "most of them holds the market had priced.")
    out[1:1] = notes
    out.append("Decision days from the Federal Reserve's FOMC calendar (fomccalendars.htm); "
               "hike, cut or hold from FRED DFEDTARU; prices from Yahoo Finance daily closes, the "
               "day before to the decision day and from it to N trading days later. A base rate, "
               "not a forecast; not advice.")
    return out


__all__ = ["ASKED", "DEFAULT_WINDOWS", "lines"]
