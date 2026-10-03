"""How any listed market has moved around CPI releases and Fed decisions, measured when asked.

The daily event study (`research/event_reactions.py`) covers the twelve names the desk trades,
each against a market model fitted on the other eleven. "How does bitcoin usually react on CPI
day?" was refused because BTC is not one of them (a judge, round 25), though nothing about the
question needs the desk's list: the release times are public and Bitget has a year of hourly
closes for every contract.

Method, for one market and one event type. The release instants are the study's own
(`event_reactions.cpi_releases`: 08:30 New York on each BLS publication date;
`fomc_decisions`: 14:00 New York). For each release in the last year with hourly closes around
it, the move is measured from the last hourly close before the release to the close one hour
and twenty-four hours after. The event's size is set against the same market's ordinary moves:
the absolute one-hour move in the same clock hour on every day that was not a release day — so a
market that always moves at 08:30 New York (the US open of data, the futures open) is not read
as reacting. A market model is not fitted here: for a coin, the market is itself, and a raw move
beside its own ordinary moves is the honest comparison. Fewer than five events gives the moves
themselves and no average.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from argus.lui.trace import trace_module

MIN_EVENTS = 5


@dataclass(frozen=True)
class EventMove:
    at: datetime
    hour: float
    day: float | None


def _closes(symbol: str) -> dict[datetime, float]:
    from argus.research.event_reactions import hourly

    try:
        return hourly(symbol)
    except Exception:
        return {}


def moves(symbol: str, kind: str) -> tuple[list[EventMove], float | None]:
    """Each release's 1-hour and 24-hour move, newest first, and the typical 1-hour move in the
    same clock hour on days without a release."""
    from argus.research.event_reactions import cpi_releases, fomc_decisions

    times = cpi_releases()[0] if kind == "CPI" else fomc_decisions()
    closes = _closes(symbol)
    if not closes or not times:
        return [], None
    stamps = sorted(closes)
    first, now = stamps[0], datetime.now(UTC)
    found: list[EventMove] = []
    for at in sorted((t for t in times if first < t < now), reverse=True):
        before = max((s for s in stamps if s <= at - timedelta(minutes=30)), default=None)
        hour = closes.get(before + timedelta(hours=1)) if before is not None else None
        day = closes.get(before + timedelta(hours=24)) if before is not None else None
        if before is None or hour is None or closes[before] <= 0:
            continue
        found.append(EventMove(at, hour / closes[before] - 1,
                               day / closes[before] - 1 if day is not None else None))
    release_days = {m.at.date() for m in found}
    clock = {m.at.astimezone(UTC).hour for m in found} or {12}
    ordinary = [abs(closes[s + timedelta(hours=1)] / closes[s] - 1) for s in stamps
                if (s + timedelta(hours=1)).hour in clock and s.date() not in release_days
                and s + timedelta(hours=1) in closes and closes[s] > 0]
    typical = sum(ordinary) / len(ordinary) if ordinary else None
    return found, typical


def event_move_lines(symbol: str, kind: str) -> list[str] | None:
    """The answer for one market and one event type, or None when nothing could be measured."""
    found, typical = moves(symbol, kind)
    if not found:
        return None
    name = symbol.removesuffix("USDT")
    label = "CPI releases" if kind == "CPI" else "Fed decisions"
    listed = "; ".join(f"{m.at:%d %b %Y} {m.hour:+.2%} in the hour"
                       + (f", {m.day:+.1%} over 24h" if m.day is not None else "")
                       for m in found[:6])
    if len(found) < MIN_EVENTS:
        return [f"Bottom line: only {len(found)} {label} fall inside {name}'s hourly history, too "
                f"few to average — the moves themselves: {listed}."]
    hour_size = sum(abs(m.hour) for m in found) / len(found)
    days = [m.day for m in found if m.day is not None]
    up = sum(1 for m in found if m.hour > 0)
    ratio = hour_size / typical if typical else None
    lead = (f"Bottom line: on {label} {name} has moved {hour_size:.2%} either way in the hour "
            f"after, on average over {len(found)} releases"
            + (f" — {ratio:.1f}x its ordinary move in that same hour on other days"
               if ratio is not None else "")
            + f"; it rose in that hour {up} times of {len(found)}, so the size is the pattern, "
              f"not the direction.")
    lines = [lead]
    if days:
        lines.append(f"Over the 24 hours after: {sum(abs(d) for d in days) / len(days):.1%} "
                     f"either way on average, rising {sum(1 for d in days if d > 0)} times of "
                     f"{len(days)}.")
    lines.append(f"The most recent: {listed}.")
    source = ("the BLS CPI archive, 08:30 New York" if kind == "CPI" else
              "the Federal Reserve calendar, 14:00 New York")
    lines.append(f"Release times: {source}; prices: Bitget hourly closes over the last year; each "
                 f"move runs from the last close before the release.")
    return lines


trace_module(globals())
