"""How each traded name has reacted to CPI releases, Fed decisions and its own earnings.

"How does NVDA react to CPI?" is the Event-Driven Agent sub-theme's question in one line, and the
console could not answer it: `research/eventstudy.py` — the MacKinlay pipeline with the Patell,
BMP, Corrado and generalised-sign tests and the Kolari-Pynnonen clustering correction, measured
against whale-signals' fixed-null test — had only ever been run inside its comparison. State is
`implemented`, not `OWNED`: the capability's own toml
(`eval/capabilities/10-clustering-corrected-base-rate-honest-event-significance-vs.toml`) was
re-graded down from OWNED on 2026-09-24 because the Event-Driven Agent sub-theme asks for a full
event -> decision -> trade comparison, not significance methodology alone; most of the named
rivals were run 2026-09-26 (TIES Vibe-Trading and whale-signals, beats Ballast) but the
`chain_falsifiers` fix in `agents/analysts.py` still needs re-measuring on freshly recorded chains
before this can be re-graded. This module runs it on the three scheduled event types that move
tokenised US equities, for every traded name, and writes one artefact the console reads.

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
from argus.truth import http
from argus.truth.paths import DATA_DIR

DATA = DATA_DIR
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
    # the BLS archive refuses a client that does not look like a browser
    return http.fetch_text(url, timeout=timeout,
                           headers={"User-Agent": "Mozilla/5.0 (argus-research)"})


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


RESULTS_TO_REPORT_DAYS = 45
"""A quarter's results release precedes its 10-Q or 10-K by at most this many days."""
LATE_IN_QUARTER_DAYS = 14
"""A release this long after its calendar quarter ended can be results; production and delivery
figures land in the first days of the quarter."""


def results_releases(filings: Sequence[Any]) -> list[datetime]:
    """The 8-K item 2.02 filings that release a quarter's results, by acceptance time.

    Item 2.02 ("results of operations") is also used for other figures: Tesla files its quarterly
    production and deliveries under it, on the second day of each quarter, three weeks before its
    results — so its "earnings" study counted eight events, four of them delivery reports (SEC
    EDGAR, checked 2026-09-30). A results release is the last 2.02 before a 10-Q or 10-K, within
    45 days of it; a 2.02 after the latest periodic report counts only when it came at least 14
    days after its calendar quarter ended, so a delivery report is not taken for results before
    the 10-Q that would place it arrives."""
    releases = sorted(f.accepted for f in filings
                      if f.form in ("8-K", "8-K/A") and "2.02" in f.items)
    periodic = sorted(f.accepted for f in filings
                      if f.form in ("10-Q", "10-K", "10-Q/A", "10-K/A"))
    kept: set[datetime] = set()
    for report in periodic:
        before = [r for r in releases
                  if timedelta(0) <= report - r <= timedelta(days=RESULTS_TO_REPORT_DAYS)]
        if before:
            kept.add(before[-1])
    last = periodic[-1] if periodic else None
    for release in releases:
        if last is not None and release <= last:
            continue
        quarter_end = date(release.year, 3 * ((release.month - 1) // 3) + 1, 1) - timedelta(days=1)
        if (release.date() - quarter_end).days >= LATE_IN_QUARTER_DAYS:
            kept.add(release)
    return sorted(kept)


def release_anchor(accepted: datetime) -> datetime:
    """The instant a results release reaches the market, from its 8-K's acceptance time.

    EDGAR accepts an after-close 8-K hours after the press release: Apple's results for the quarter
    to September 2025 went out at 16:30 New York time on 30 Oct and its 8-K was accepted at
    2025-10-31T00:30:35Z (SEC submissions API, checked 2026-10-03), so a window opened at acceptance
    began after the 24-hour perpetual had already moved, and Apple's four "moves" read +0.1%, -0.1%,
    +0.2%, +0.0% (a judge, round 24). An acceptance outside the US regular session (09:30-16:00 New
    York) is anchored at the last regular close before it, where every after-close or pre-open
    release is first traded; an acceptance inside the session is its own anchor."""
    from zoneinfo import ZoneInfo

    new_york = ZoneInfo("America/New_York")
    local = accepted.astimezone(new_york)
    if time(9, 30) <= local.time() < time(16, 0) and local.weekday() < 5:
        return accepted
    day = local.date() if local.time() >= time(16, 0) else local.date() - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return datetime.combine(day, time(16, 0), tzinfo=new_york).astimezone(UTC)


def earnings_releases(ticker: str) -> list[datetime]:
    """Every quarterly results release on EDGAR (:func:`results_releases`), each anchored where the
    market first trades it (:func:`release_anchor`)."""
    from argus.market.evidence import EdgarSource

    since = datetime.now(UTC) - timedelta(days=HISTORY_DAYS + 30)
    return [release_anchor(at) for at in
            results_releases(EdgarSource().filings(ticker, since=since, limit=200))]


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


hourly = _hourly
"""A year of hourly closes for one contract, read by the on-demand study in
`lui/research/macro_moves.py`."""


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
    # the instant, not the day: the move list measures from it (round 24)
    dates = tuple(w.at.astimezone(UTC).isoformat(timespec="minutes") for w in windows)
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
    unavailable: dict[str, str] = {}

    def hourly_or_reason(symbol: str) -> dict[datetime, float]:
        # One name whose history cannot be read is reported and left out; until 2026-09-28 it
        # raised through pool.map and the whole study was lost for the other eleven (the
        # 2026-09-27 19:00 cycle, COINUSDT and TQQQUSDT with no history).
        from argus.market.history import HistoryError

        try:
            return _hourly(symbol)
        except HistoryError as exc:
            unavailable[symbol] = str(exc)[:200]
            return {}

    with ThreadPoolExecutor(max_workers=2) as pool:
        closes = dict(zip(SYMBOLS, pool.map(hourly_or_reason, SYMBOLS), strict=True))
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
        "history": {"interval": "1H", "per_symbol": spans, "unavailable": unavailable},
        "window": {"event_bars": EVENT_BARS, "estimation_bars": ESTIMATION_BARS,
                   "gap_bars": GAP_BARS, "min_events": MIN_EVENTS},
        "sources": {"cpi": f"BLS CPI release archive ({cpi_route})",
                    "fomc": "Federal Reserve FOMC calendar (market/calendar.py snapshot)",
                    "earnings": ("SEC EDGAR 8-K item 2.02 results releases (delivery and "
                                 "pre-announcement filings excluded), acceptanceDateTime"),
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
    path.write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
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
