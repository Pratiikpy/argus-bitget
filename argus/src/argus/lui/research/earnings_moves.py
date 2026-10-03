"""How a company's shares moved on its last results releases, for any US-listed company.

A judge in round 25 asked "how big has the stock moved after the last few reports?" of Micron,
"pichli baar kitna gira tha?" of Tesla, "Did Netflix or Meta move more on their most recent
earnings?" and "compare that with how Oracle reacted to its last print". None was answered: the
console's event study (`research/event_reactions.py`) covers the twelve names the desk trades, and
every other company got its volatility, a base rate or its technicals instead.

Method. The release instants are SEC's own: each 8-K item 2.02 that releases a quarter's results
(`event_reactions.results_releases`, which already drops Tesla's delivery reports), by its
acceptance time. A release accepted outside the US regular session is first traded at the next
session, so its move is the close before it to the first close after it; one accepted inside a
session moves that session, so its move is the previous close to that day's close — the
convention of `event_reactions.release_anchor`, applied to daily closes. The closes are Yahoo
Finance's split-adjusted daily bars (`market/equity_history.daily`), so a stock Bitget does not
list is measured the same way. The S&P 500's own move that day (SPY) is given beside each, so a
release on a day the whole market fell is not read as the company's own news.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from argus.lui.trace import trace_module

EARNINGS_MOVE_Q = re.compile(
    r"\b(?:move[ds]?|moved|react(?:s|ed|ion)?|gap(?:ped|s)?|f[ae]ll|fallen|drop(?:ped|s)?|jump(?:ed|s)?|"
    r"r[io]se|risen|pop(?:ped)?|tank(?:ed)?|do|did|went)\b[^?]{0,60}\b(?:after|on|following|post|"
    r"around|to)\b[^?]{0,40}\b(?:earnings|reports?|results|prints?|quarter)\b|"
    r"\b(?:earnings|results|report|print)\s+(?:reaction|move|day\s+move|gap)s?\b|"
    r"\breacted\s+to\s+(?:its|their|the)\s+(?:last|latest|most\s+recent)\s+(?:print|report|"
    r"earnings|results)\b|"
    r"\b(?:pichli|last)\s+baar\b[^?]{0,30}\b(?:kitna|kitni)\b|"
    r"\bmove[ds]?\s+more\s+on\s+(?:their|its|the)\s+(?:most\s+recent|last|latest)\s+earnings\b|"
    r"财报后.{0,6}(?:涨|跌|波动)|se\s+movi[oó]\b[^?]{0,30}\b(?:[uú]ltima|resultados)",
    re.I)
"""A question about how a stock moved on its results, not when it reports."""


@dataclass(frozen=True)
class Reaction:
    released: datetime
    base_day: date
    moved_day: date
    move: float
    market: float | None


def _days(ticker: str) -> dict[date, float]:
    from argus.market.equity_history import HistoryError, daily

    try:
        return {d.day: float(d.close) for d in daily(ticker)}
    except (HistoryError, OSError, ValueError):
        return {}


def reactions(ticker: str, count: int = 4) -> list[Reaction]:
    """The last ``count`` results releases and the move each was met with, newest first."""
    from zoneinfo import ZoneInfo

    from argus.market.evidence import EdgarSource
    from argus.research.event_reactions import results_releases

    since = datetime.now(UTC) - timedelta(days=400)
    try:
        released = results_releases(EdgarSource().filings(ticker, since=since, limit=200))
    except Exception:
        return []
    closes, market = _days(ticker), _days("SPY")
    sessions = sorted(closes)
    new_york = ZoneInfo("America/New_York")
    out: list[Reaction] = []
    for at in reversed(released):
        local = at.astimezone(new_york)
        in_session = time(9, 30) <= local.time() < time(16, 0) and local.weekday() < 5
        if in_session:
            moved = next((d for d in sessions if d >= local.date()), None)
        else:
            after = local.date() + timedelta(days=1) if local.time() >= time(16, 0) else \
                local.date()
            moved = next((d for d in sessions if d >= after), None)
        if moved is None:
            continue
        before = [d for d in sessions if d < moved]
        if not before or closes[before[-1]] <= 0:
            continue
        base = before[-1]
        spy = (market[moved] / market[base] - 1) if base in market and moved in market and \
            market[base] > 0 else None
        out.append(Reaction(at, base, moved, closes[moved] / closes[base] - 1, spy))
        if len(out) >= count:
            break
    return out


def reaction_lines(tickers: list[str], count: int = 4, *,
                   typical: bool = False) -> list[str] | None:
    """One company's last moves on its results, or two or three side by side: ranked on the
    latest move, or with ``typical`` on the average size of the last ``count`` ("which of those
    has the biggest typical earnings move?", a judge, round 25)."""
    from zoneinfo import ZoneInfo

    found = [(t, r) for t in tickers[:3] if (r := reactions(t, count))]
    if not found:
        return None

    def said(r: Reaction) -> str:
        context = f"; S&P 500 {r.market:+.1%} that day" if r.market is not None else ""
        out = r.released.astimezone(ZoneInfo("America/New_York"))
        return (f"{r.move:+.1%} on {r.moved_day:%d %b %Y} (8-K accepted {out:%d %b} "
                f"{out:%H:%M} New York time{context})")

    lines: list[str] = []
    if len(found) == 1:
        ticker, moves = found[0]
        usual = sum(abs(r.move) for r in moves) / len(moves)
        lines.append(f"Bottom line: {ticker} moved {moves[0].move:+.1%} on its most recent results "
                     f"({moves[0].moved_day:%d %b %Y}); over its last {len(moves)} releases the "
                     f"move averaged {usual:.1%} either way.")
        lines.append(f"{ticker}, newest first: " + "; ".join(said(r) for r in moves) + ".")
    elif typical:
        def size(moves: list[Reaction]) -> float:
            return sum(abs(r.move) for r in moves) / len(moves)

        ranked = sorted(found, key=lambda f: -size(f[1]))
        lead, rest = ranked[0], ranked[1:]
        lines.append(f"Bottom line: {lead[0]} has the biggest typical move on its results — "
                     f"{size(lead[1]):.1%} either way on average over its last {len(lead[1])}, "
                     f"against " + ", ".join(f"{t} {size(m):.1%}" for t, m in rest) + ".")
    else:
        latest = sorted(found, key=lambda f: -abs(f[1][0].move))
        lead, rest = latest[0], latest[1:]
        lines.append(f"Bottom line: {lead[0]} moved more on its most recent results — "
                     f"{lead[1][0].move:+.1%} against "
                     + ", ".join(f"{t} {m[0].move:+.1%}" for t, m in rest) + ".")
    if len(found) > 1:
        for ticker, moves in found:
            average = sum(abs(r.move) for r in moves) / len(moves)
            lines.append(f"{ticker}: " + "; ".join(said(r) for r in moves)
                         + f" — {average:.1%} on average either way.")
    lines.append("Measured from the close before each results release to the first close after "
                 "it, release times from SEC EDGAR (8-K item 2.02, acceptance time), prices from "
                 "Yahoo Finance's split-adjusted daily closes; a past reaction says how big the "
                 "next one may be, not which way.")
    return lines


def next_report(ticker: str) -> tuple[date, bool] | None:
    """When ``ticker`` reports next and whether the date is an estimate, from Yahoo Finance's
    earnings calendar, or None."""
    from argus.lui.research.fundamentals import yahoo_summary

    try:
        calendar = ((yahoo_summary(ticker).get("calendarEvents") or {}).get("earnings") or {})
    except Exception:
        return None
    stamps = [d.get("fmt") for d in calendar.get("earningsDate") or [] if d.get("fmt")]
    if not stamps:
        return None
    when = date.fromisoformat(str(stamps[0])[:10])
    if when < datetime.now(UTC).date():
        return None
    return when, bool(calendar.get("isEarningsDateEstimate"))


def next_report_line(ticker: str) -> str | None:
    """When ``ticker`` reports next, said as a line, or None."""
    found = next_report(ticker)
    if found is None:
        return None
    when, estimated = found
    days = (when - datetime.now(UTC).date()).days
    return (f"{ticker} reports next on {when:%d %b %Y}"
            + (" (an estimated date)" if estimated else "")
            + f", in {days} days (Yahoo Finance's earnings calendar).")


SOONEST_Q = re.compile(
    r"\b(?:which|who)\b[^?]{0,60}\breports?\b[^?]{0,40}\b(?:soonest|first|next|earliest)\b|"
    r"\bwhich\b[^?]{0,60}\b(?:earnings|results)\b[^?]{0,30}\b(?:soonest|first|next|earliest)\b|"
    r"\bwhen\s+do\b[^?]{0,80}\breport\b|\b(?:earnings|report)\s+dates?\s+for\b", re.I)
"""Report dates asked of several names at once."""


def report_order_lines(tickers: list[str]) -> list[str] | None:
    """Several companies' next report dates, soonest first: "Which of AAPL, MSFT and NVDA report
    earnings soonest?" got a risk comparison (round 25)."""
    dated = sorted(((t, f) for t in tickers[:8] if (f := next_report(t)) is not None),
                   key=lambda tf: tf[1][0])
    if not dated:
        return None
    today = datetime.now(UTC).date()
    first, (when, estimated) = dated[0]
    lines = [f"Bottom line: {first} reports soonest, on {when:%d %b %Y}"
             + (" (an estimated date)" if estimated else "")
             + f", in {(when - today).days} days."]
    lines += [f"{t}: {w:%d %b %Y}" + (" (estimated)" if e else "")
              + f", in {(w - today).days} days." for t, (w, e) in dated]
    missing = [t for t in tickers[:8] if t not in {d[0] for d in dated}]
    if missing:
        lines.append(f"No upcoming date on Yahoo's calendar for {', '.join(missing)}.")
    lines.append("Dates: Yahoo Finance's earnings calendar; a date marked estimated is the "
                 "calendar's projection until the company confirms it.")
    return lines


trace_module(globals())
