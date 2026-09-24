"""How each traded name has reacted to CPI releases, Fed decisions and its own earnings.

"How does NVDA react to CPI?" is the Event-Driven Agent sub-theme's question in one line, and the
console could not answer it: `research/eventstudy.py` — the MacKinlay pipeline with the Patell,
BMP, Corrado and generalised-sign tests and the Kolari-Pynnonen clustering correction, OWNED
against whale-signals' fixed-null test — had only ever been run inside its comparison. This module
runs it on the three scheduled event types that move tokenised US equities, for every traded name,
and writes one artefact the console reads.

**Event times come from the issuers, not from a vendor calendar.**

* CPI: the Bureau of Labor Statistics release archive
  (https://www.bls.gov/bls/news-release/cpi.htm). Each release links a file named for the day it
  was actually published (``cpi_10242025.htm``), so the September 2025 report delayed by the
  appropriations lapse sits on 24 Oct and the October 2025 report, never published, is absent —
  a scheduled-date calendar would have put an event on a day nothing happened. 08:30 New York.
* FOMC: the Federal Reserve's meeting calendar, already read by `market/calendar.py`; the decision
  is released at 14:00 New York on a meeting's last day.
* Earnings: the company's own 8-K under item 2.02 on SEC EDGAR, timed by ``acceptanceDateTime`` —
  the instant it became public, usually after the close. A tokenised stock trades through that
  evening, so the reaction is measurable hours before the exchange reopens.

**Two figures per event type, and they answer different questions.** The average cumulative
abnormal return and its four tests say whether the name tends to move one way; for a release that
surprises in either direction the answer is usually no, and the module says so. The *size* ratio
says whether it moves more than usual: the mean absolute abnormal return over the event window
against the mean absolute abnormal return over same-length blocks of the estimation window of the
same events. That is what a hedge before CPI is for, and it is reported as a description with its
event count, not as a test.

**Clock and proxy follow `eventstudy.py`.** Hourly bars; the market proxy is the equal-weighted
return of the other traded names, so a CPI-day move shared by all twelve is removed and what is
left is the name's own reaction; the estimation window is purged from the event by a day.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from argus.research.eventstudy import (
    MIN_EVENTS,
    EventStudyError,
    EventWindow,
    build_window,
    study,
)

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "event_reactions.json"
CPI_ARCHIVE = "https://www.bls.gov/bls/news-release/cpi.htm"
READER = "https://r.jina.ai/"
NEW_YORK = ZoneInfo("America/New_York")

SYMBOLS: tuple[str, ...] = (
    "NVDAUSDT", "TSLAUSDT", "AAPLUSDT", "MSFTUSDT", "METAUSDT", "GOOGLUSDT", "AMZNUSDT",
    "COINUSDT", "MSTRUSDT", "QQQUSDT", "TQQQUSDT", "SQQQUSDT",
)
"""The twelve traded names; the proxy for each is the other eleven."""
FUNDS = frozenset({"QQQUSDT", "TQQQUSDT", "SQQQUSDT"})
"""Funds file no earnings."""

EVENT_BARS = 24
"""The reaction window: the 24 hours from the first bar at or after the release."""
ESTIMATION_BARS = 480
GAP_BARS = 24
HISTORY_DAYS = 400
"""Hourly history requested; Bitget's rToken candles begin in August 2025."""


@dataclass(frozen=True, slots=True)
class Reaction:
    """One name's reaction to one event type."""

    symbol: str
    kind: str
    events: int
    average_car_bps: float | None
    size_ratio: float | None
    verdict: str
    tests: dict[str, dict[str, float]]
    dates: tuple[str, ...]
    confounded: tuple[str, ...] = ()
    """Event dates left out because the name's own earnings fell inside the reaction window."""

    def as_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "kind": self.kind, "events": self.events,
                "average_car_bps": self.average_car_bps, "size_ratio": self.size_ratio,
                "verdict": self.verdict, "tests": self.tests, "dates": list(self.dates),
                "confounded": list(self.confounded)}


def _get(url: str, timeout: float = 30.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (argus-research)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return str(response.read().decode("utf-8", errors="replace"))


def parse_cpi_archive(text: str) -> list[date]:
    """Publication dates from the archive's file names (``cpi_MMDDYYYY``), newest first."""
    found = {date(int(y), int(m), int(d))
             for m, d, y in re.findall(r"archives/cpi_(\d\d)(\d\d)(20\d\d)\.htm", text)}
    return sorted(found, reverse=True)


def cpi_releases() -> tuple[list[datetime], str]:
    """Every published CPI release time (08:30 New York, as UTC), and the route that answered."""
    for route, url in (("direct", CPI_ARCHIVE), ("reader", READER + CPI_ARCHIVE)):
        try:
            days = parse_cpi_archive(_get(url))
        except Exception:
            continue
        if days:
            return [datetime.combine(d, time(8, 30), NEW_YORK).astimezone(UTC)
                    for d in days], route
    return [], "unavailable"


def fomc_decisions(snapshot: Path = DATA / "event_calendar.json") -> list[datetime]:
    """Past FOMC decision times (14:00 New York, as UTC) from `market/calendar.py`'s snapshot."""
    try:
        days = json.loads(snapshot.read_text(encoding="utf-8")).get("fomc", [])
    except (OSError, ValueError):
        return []
    return [datetime.combine(date.fromisoformat(d), time(14, 0), NEW_YORK).astimezone(UTC)
            for d in days]


def earnings_releases(ticker: str) -> list[datetime]:
    """The acceptance time of every 8-K item 2.02 (results of operations) on EDGAR."""
    from argus.market.evidence import EdgarSource

    since = datetime.now(UTC) - timedelta(days=HISTORY_DAYS + 30)
    return [f.accepted for f in EdgarSource().filings(ticker, since=since, limit=200)
            if f.form == "8-K" and "2.02" in f.items]


FETCH_ATTEMPTS = 5


def _hourly(symbol: str) -> dict[datetime, float]:
    """A year of hourly closes. Bitget answers HTTP 429 when twelve of these page at once (seen on
    the first run, 2026-09-24), so each attempt backs off and retries rather than failing the
    study."""
    import time as _time

    from argus.market.history import HistoryError, fetch_window

    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            candles = fetch_window(symbol, start=datetime.now(UTC) - timedelta(days=HISTORY_DAYS),
                                   interval="1H", pause=0.25)
            return {c.ts: float(c.close) for c in candles}
        except HistoryError as exc:
            if "429" not in str(exc) or attempt == FETCH_ATTEMPTS:
                raise
            _time.sleep(5.0 * attempt)
    return {}


def _block_sizes(window: EventWindow, bars: int) -> list[float]:
    """Absolute abnormal returns summed over non-overlapping ``bars``-long blocks of the
    estimation window: the name's ordinary 24-hour idiosyncratic move, for scale."""
    series = window.estimation_abnormal
    return [abs(sum(series[i:i + bars])) for i in range(0, len(series) - bars + 1, bars)]


CONFOUND_HOURS = 36
"""An event is confounded for a name when that name's own earnings are released within this many
hours of it — the 24-hour reaction window plus the evening before. Measured on the first full run
(2026-09-24): META reported on the FOMC days of 29 Oct 2025 and 28 Jan 2026, MSFT on the same two,
and their "reaction to the Fed" came out at 5.4x and 4.9x an ordinary day. That was earnings."""


def unconfounded(times: Sequence[datetime], own_events: Sequence[datetime],
                 ) -> tuple[list[datetime], list[datetime]]:
    """``times`` split into those clear of the name's own events and those too close to one."""
    reach = timedelta(hours=CONFOUND_HOURS)
    clear: list[datetime] = []
    close: list[datetime] = []
    for at in times:
        (close if any(abs(at - e) <= reach for e in own_events) else clear).append(at)
    return clear, close


def reaction(symbol: str, kind: str, times: Sequence[datetime],
             timestamps: Sequence[datetime], asset: Sequence[float],
             market: Sequence[float], *, own_events: Sequence[datetime] = ()) -> Reaction:
    """Study one name's reaction to one event type; a refusal, not a figure, below MIN_EVENTS.
    Events within ``CONFOUND_HOURS`` of ``own_events`` (the name's earnings) are left out and
    listed, so a macro reaction is not an earnings reaction in disguise."""
    times, dropped = unconfounded(times, own_events)
    confounded = tuple(t.date().isoformat() for t in sorted(dropped))
    windows: list[EventWindow] = []
    for at in sorted(times):
        try:
            found = build_window(symbol, at, timestamps, asset, market, event_bars=EVENT_BARS,
                                 estimation_bars=ESTIMATION_BARS, gap_bars=GAP_BARS)
        except EventStudyError:
            continue  # a window whose market model cannot be fitted is left out, not guessed at
        if found is not None:
            windows.append(found)
    dates = tuple(w.at.date().isoformat() for w in windows)
    if len(windows) < MIN_EVENTS:
        return Reaction(symbol, kind, len(windows), None, None,
                        f"{len(windows)} {kind} event(s) have enough hourly history around them; "
                        f"the study needs {MIN_EVENTS}, so no figure is given", {}, dates,
                        confounded)
    try:
        result = study(f"{symbol} {kind}", windows, event_bars=EVENT_BARS)
    except EventStudyError as exc:
        return Reaction(symbol, kind, len(windows), None, None, str(exc), {}, dates, confounded)
    event_size = sum(abs(w.car) for w in windows) / len(windows)
    ordinary = [s for w in windows for s in _block_sizes(w, EVENT_BARS)]
    ratio = event_size / (sum(ordinary) / len(ordinary)) if ordinary and sum(ordinary) else None
    return Reaction(symbol, kind, result.events, round(result.average_car_bps, 1),
                    None if ratio is None else round(ratio, 2), result.verdict,
                    result.statistics, dates, confounded)


def aligned(symbol: str, closes: dict[str, dict[datetime, float]],
            ) -> tuple[list[datetime], list[float], list[float]]:
    """``symbol``'s hourly returns and, bar for bar, the equal-weighted return of every other name
    that traded across the same hour.

    `eventstudy.market_proxy` wants one aligned panel, and the first run built it from the hours
    all twelve names share: the latest-listed token starts on 2026-03-31, which cut NVDA's thirteen
    months of history to six and its FOMC and earnings events below the five a study needs. Here
    the proxy is taken over whichever other names existed at each hour, so a name's own history
    sets the span and a late listing only thins the proxy for the hours before it arrived.
    """
    own = closes.get(symbol, {})
    times = sorted(own)
    stamps: list[datetime] = []
    asset: list[float] = []
    proxy: list[float] = []
    hour = timedelta(hours=1)
    for before, at in pairwise(times):
        if at - before != hour or own[before] <= 0:
            continue
        others = [c[at] / c[before] - 1.0 for s, c in closes.items()
                  if s != symbol and at in c and before in c and c[before] > 0]
        if not others:
            continue
        stamps.append(at)
        asset.append(own[at] / own[before] - 1.0)
        proxy.append(sum(others) / len(others))
    return stamps, asset, proxy


def build(now: datetime | None = None) -> dict[str, Any]:
    """Every name against every event type, from live candles and the issuers' own records."""
    stamp = now or datetime.now(UTC)
    with ThreadPoolExecutor(max_workers=2) as pool:
        closes = dict(zip(SYMBOLS, pool.map(_hourly, SYMBOLS), strict=True))
        earnings_jobs = {s: pool.submit(earnings_releases, s.removesuffix("USDT"))
                         for s in SYMBOLS if s not in FUNDS}
        cpi, cpi_route = cpi_releases()
        earnings = {}
        for s, job in earnings_jobs.items():
            try:
                earnings[s] = job.result()
            except Exception:
                earnings[s] = []
    fomc = [t for t in fomc_decisions() if t <= stamp]
    rows: list[dict[str, Any]] = []
    spans: dict[str, list[str]] = {}
    for symbol in SYMBOLS:
        stamps, asset, proxy = aligned(symbol, closes)
        if len(stamps) < ESTIMATION_BARS + GAP_BARS + EVENT_BARS:
            continue
        spans[symbol] = [stamps[0].isoformat(), stamps[-1].isoformat()]
        kinds: list[tuple[str, Sequence[datetime]]] = [("CPI", cpi), ("FOMC", fomc)]
        if symbol not in FUNDS:
            kinds.append(("earnings", earnings.get(symbol, [])))
        own = earnings.get(symbol, [])
        for kind, times in kinds:
            rows.append(reaction(symbol, kind, [t for t in times if t <= stamp], stamps,
                                 asset, proxy,
                                 own_events=own if kind != "earnings" else ()).as_dict())
    return {
        "generated_at": stamp.isoformat(),
        "history": {"interval": "1H", "per_symbol": spans},
        "window": {"event_bars": EVENT_BARS, "estimation_bars": ESTIMATION_BARS,
                   "gap_bars": GAP_BARS, "min_events": MIN_EVENTS},
        "sources": {"cpi": f"BLS CPI release archive ({cpi_route})",
                    "fomc": "Federal Reserve FOMC calendar (market/calendar.py snapshot)",
                    "earnings": "SEC EDGAR 8-K item 2.02, acceptanceDateTime",
                    "prices": "Bitget hourly market candles"},
        "reactions": rows,
    }


def lookup(symbol: str, kind: str, report: dict[str, Any]) -> dict[str, Any] | None:
    for row in report.get("reactions", []):
        if row.get("symbol") == symbol and row.get("kind") == kind:
            return dict(row)
    return None


def write(path: Path = REPORT_PATH) -> dict[str, Any]:
    report = build()
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


MAX_AGE_HOURS = 20.0
"""The desk's cycle runs four times a day; the study needs a year of hourly bars for twelve names,
about eight minutes of paging, and a new CPI or FOMC event arrives at most once a day. So the cycle
refreshes it only when the artefact is older than this."""


def fresh(path: Path = REPORT_PATH, max_age_hours: float = MAX_AGE_HOURS) -> bool:
    try:
        stamp = datetime.fromisoformat(
            json.loads(path.read_text(encoding="utf-8"))["generated_at"])
    except (OSError, ValueError, KeyError):
        return False
    return datetime.now(UTC) - stamp < timedelta(hours=max_age_hours)


def main() -> int:  # pragma: no cover - CLI
    import sys

    if "--if-stale" in sys.argv and fresh():
        print(f"{REPORT_PATH.name} is under {MAX_AGE_HOURS:.0f}h old; not refreshed")
        return 0
    report = write()
    for row in report["reactions"]:
        figure = ("—" if row["average_car_bps"] is None else
                  f"{row['average_car_bps']:+.1f}bps, size x{row['size_ratio']}")
        print(f"{row['symbol']:<10} {row['kind']:<9} {row['events']:>3} events  {figure}")
    print(f"written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["EVENT_BARS", "Reaction", "build", "cpi_releases", "earnings_releases",
           "fomc_decisions", "lookup", "parse_cpi_archive", "reaction", "write"]
