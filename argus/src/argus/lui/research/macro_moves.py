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


def hour_move(symbol: str, at: datetime) -> float | None:
    """One release's move in the hour after it, measured as :func:`moves` measures every past
    one: from the last hourly close at least 30 minutes before ``at`` to the close an hour later.
    The daily call record grades the size this module states against it (`eval/call_record`)."""
    from argus.market.history import fetch_window

    try:
        candles = fetch_window(symbol, start=at - timedelta(hours=3), end=at + timedelta(hours=3),
                               interval="1H", pause=0.0)
    except Exception:
        return None
    closes = {c.ts: float(c.close) for c in candles}
    before = max((t for t in closes if t <= at - timedelta(minutes=30)), default=None)
    if before is None or closes[before] <= 0:
        return None
    after = closes.get(before + timedelta(hours=1))
    return after / closes[before] - 1 if after is not None else None


def _around(symbol: str, at: datetime) -> float | None:
    """The move from the last hourly close before ``at`` to the close 24 hours after it."""
    import time

    from argus.market.history import fetch_window

    candles = None
    for attempt in range(3):
        # four windows at once can draw Bitget's 429; a short wait and a retry, so a release is
        # not dropped from the count on a rate limit (counts of 7 and 9 on two runs, round 32)
        try:
            candles = fetch_window(symbol, start=at - timedelta(hours=3),
                                   end=at + timedelta(hours=26), interval="1H", pause=0.0)
            break
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    if candles is None:
        return None
    closes = {c.ts: float(c.close) for c in candles}
    before = max((t for t in closes if t <= at - timedelta(minutes=30)), default=None)
    if before is None or closes[before] <= 0:
        return None
    after = closes.get(before + timedelta(hours=24))
    return after / closes[before] - 1 if after is not None else None


def _surprises(times: list[datetime]) -> dict[datetime, float]:
    """The 2-year Treasury yield's change on each release day, in basis points (FRED DGS2): the
    market's own read of whether the release came in hotter (yields up) or cooler (down) than it
    expected. No consensus forecasts are read here, so the bond market's reaction stands in."""
    from argus.lui.research.macro import _fred

    try:
        rows = dict(_fred("DGS2", 420))
    except Exception:
        return {}
    days = sorted(rows)
    out: dict[datetime, float] = {}
    for at in times:
        day = at.date().isoformat()
        before = [d for d in days if d < day]
        if day in rows and before:
            out[at] = (rows[day] - rows[before[-1]]) * 100
    return out


def _surprise_line(moves: dict[datetime, float], surprise: dict[datetime, float],
                   kind: str) -> str | None:
    """The book's average move after the hotter releases and after the cooler ones."""
    hot = [moves[t] for t, bp in surprise.items() if bp >= 3]
    cool = [moves[t] for t, bp in surprise.items() if bp <= -3]
    if not hot or not cool:
        return None
    word = ("hotter", "cooler") if kind == "CPI" else ("more hawkish", "more dovish")
    return (f"Split by the bond market's read of each release (the 2-year yield rising 3bp or "
            f"more on the day = {word[0]} than expected, falling 3bp or more = {word[1]}): after "
            f"the {len(hot)} {word[0]} ones the book averaged {sum(hot) / len(hot):+.1%} over 24 "
            f"hours, after the {len(cool)} {word[1]} ones {sum(cool) / len(cool):+.1%} — few "
            f"releases, so the split is a description, not a rule.")


def book_event_lines(weights: list[tuple[str, float]], kind: str) -> list[str] | None:
    """What each past release did to a stated book: the weighted 24-hour move of the names held,
    release by release, its worst and its average size, and — where the 2-year yield's move on
    the day sorts them — after the hotter and after the cooler releases (:func:`_surprises`).
    Without that split the worst day stands in for a bad surprise, and the answer says so; a
    name with no hourly history around the releases is left out and named."""
    from concurrent.futures import ThreadPoolExecutor

    from argus.research.event_reactions import cpi_releases, fomc_decisions

    now = datetime.now(UTC)
    times = [t for t in (cpi_releases()[0] if kind == "CPI" else fomc_decisions())
             if now - timedelta(days=365) < t < now - timedelta(hours=25)]
    per: dict[datetime, float] = {}
    covered: dict[datetime, float] = {}
    missing: list[str] = []
    # Only the hours around each release are read — a year of hourly closes per name took most
    # of a minute for a three-name book (round 32), where these windows take seconds
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {(symbol, at): pool.submit(_around, symbol, at) for symbol, _ in weights
                for at in times}
        found = {key: job.result() for key, job in jobs.items()}
    for symbol, weight in weights:
        got = {at: found[(symbol, at)] for at in times if found[(symbol, at)] is not None}
        if not got:
            missing.append(symbol.removesuffix("USDT"))
            continue
        for at, day in got.items():
            per[at] = per.get(at, 0.0) + weight * float(day or 0.0)
            covered[at] = covered.get(at, 0.0) + weight
    present = sum(w for s, w in weights if s.removesuffix("USDT") not in missing)
    complete = {at: move / present for at, move in per.items()
                if present and covered.get(at, 0.0) >= present - 1e-9}
    if len(complete) < 2:
        return None
    label = "CPI release" if kind == "CPI" else "Fed decision"
    surprise = _surprises(list(complete))
    worst_at = min(complete, key=lambda at: complete[at])
    best_at = max(complete, key=lambda at: complete[at])
    size = sum(abs(v) for v in complete.values()) / len(complete)
    held = ", ".join(f"{w:.0%} {s.removesuffix('USDT')}" for s, w in weights)
    lines = [f"Bottom line: on the last {len(complete)} {label}s your book ({held}) moved "
             f"{size:.1%} either way over the next 24 hours on average; its worst was "
             f"{complete[worst_at]:+.1%} after {worst_at:%d %b %Y} and its best "
             f"{complete[best_at]:+.1%} after {best_at:%d %b %Y}.",
             _surprise_line(complete, surprise, kind)
             or f"A surprise is not labelled on the record, so the worst of these is the closest "
                f"measured stand-in: a move like {complete[worst_at]:+.1%} on the whole book, "
                f"before any leverage multiplies it.",
             "Each release: " + "; ".join(f"{at:%d %b %Y} {move:+.1%}"
                                          for at, move in sorted(complete.items(),
                                                                 reverse=True)[:6]) + "."]
    if missing:
        lines.append(f"Left out, no hourly history around the releases: {', '.join(missing)}.")
    lines.append("Release times: " + ("the BLS CPI archive, 08:30 New York" if kind == "CPI"
                                      else "the Federal Reserve calendar, 14:00 New York")
                 + "; prices: Bitget hourly closes, from the last close before each release to "
                   "24 hours after.")
    return lines


trace_module(globals())
