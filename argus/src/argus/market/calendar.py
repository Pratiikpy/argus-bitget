"""The scheduled macro events a trader hedges around: US CPI releases and FOMC decisions.

"What should I hedge my book with before CPI?" was answered with a hedge and no date (a critic's
probe, 2026-09-24): the question named an event the console could not place in time. This module
places it, from the two official schedules:

* **CPI** — the Bureau of Labor Statistics release schedule,
  https://www.bls.gov/schedule/news_release/cpi.htm (08:30 New York time on each date);
* **FOMC** — the Federal Reserve's meeting calendar,
  https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm (the decision is released on the
  meeting's last day, 14:00 New York time).

The BLS page refuses scripted requests from some networks (HTTP 403), so :func:`refresh` reads it
through a public reader proxy when the direct request is refused, and records which route answered.
The hosted console never fetches at request time: it reads the snapshot :func:`refresh` writes,
which the desk's cycle renews, and every answer states the snapshot's date.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

DATA = Path(__file__).resolve().parents[3] / "data"
SNAPSHOT = DATA / "event_calendar.json"
CPI_URL = "https://www.bls.gov/schedule/news_release/cpi.htm"
FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
READER = "https://r.jina.ai/"
NEW_YORK = ZoneInfo("America/New_York")
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


@dataclass(frozen=True)
class Event:
    kind: str          # "CPI" or "FOMC"
    at: datetime       # UTC
    label: str
    source: str


def _get(url: str, timeout: float = 20.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (argus-research)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return str(response.read().decode("utf-8", errors="replace"))


def parse_cpi(text: str) -> list[date]:
    """Release dates from the BLS schedule table ("| September 2026 | Oct. 14, 2026 | 08:30 AM |",
    or the same cells in HTML)."""
    found: list[date] = []
    for month, day, year in re.findall(
            r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),\s+(20\d\d)"
            r"[^0-9]{0,40}08:30", text):
        found.append(date(int(year), _MONTHS[month.lower()], int(day)))
    return sorted(set(found))


def parse_fomc(html_text: str) -> list[date]:
    """Decision days (each meeting's last day) from the Fed's calendar page."""
    out: list[date] = []
    for year_text in re.findall(r"(20\d\d) FOMC Meetings", html_text):
        year = int(year_text)
        start = html_text.find(f"{year} FOMC Meetings")
        nxt = html_text.find("FOMC Meetings", start + 20)
        segment = html_text[start: nxt if nxt > 0 else start + 20000]
        months = re.findall(r"fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>", segment)
        days = re.findall(r"fomc-meeting__date[^>]*>([^<]+)<", segment)
        for month_text, day_text in zip(months, days, strict=False):
            name = month_text.split("/")[-1].strip()[:3].lower()
            last = re.findall(r"\d+", day_text)
            if name in _MONTHS and last:
                month = _MONTHS[name]
                if "/" in month_text and len(last) == 2 and int(last[-1]) < int(last[0]):
                    month = _MONTHS[month_text.split("/")[-1].strip()[:3].lower()]
                out.append(date(year, month, int(last[-1])))
    return sorted(set(out))


def refresh(out: Path = SNAPSHOT) -> dict[str, Any]:
    """Read both official schedules and write the snapshot; if one fails, the other still lands."""
    record: dict[str, Any] = {"fetched_at": datetime.now(UTC).isoformat(), "cpi": [], "fomc": []}
    try:
        try:
            cpi_text, route = _get(CPI_URL), "bls.gov"
        except Exception:
            cpi_text, route = _get(READER + CPI_URL, timeout=40.0), "bls.gov via r.jina.ai"
        record["cpi"] = [d.isoformat() for d in parse_cpi(cpi_text)]
        record["cpi_source"] = f"{CPI_URL} ({route})"
    except Exception as exc:
        record["cpi_error"] = type(exc).__name__
    try:
        record["fomc"] = [d.isoformat() for d in parse_fomc(_get(FOMC_URL))]
        record["fomc_source"] = FOMC_URL
    except Exception as exc:
        record["fomc_error"] = type(exc).__name__
    previous = _read(out)
    for key in ("cpi", "fomc"):
        if not record[key] and previous and previous.get(key):
            record[key] = previous[key]  # keep the last good schedule rather than an empty one
            record[f"{key}_source"] = previous.get(f"{key}_source", "") + " (previous snapshot)"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def _read(path: Path = SNAPSHOT) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def upcoming(now: datetime | None = None, days: int = 45,
             path: Path = SNAPSHOT) -> tuple[list[Event], str | None]:
    """Events in the next ``days``, soonest first, and the snapshot's date (``None`` if absent)."""
    record = _read(path)
    if record is None:
        return [], None
    moment = now or datetime.now(UTC)
    horizon = moment + timedelta(days=days)
    events: list[Event] = []
    for kind, stamp, label in (("CPI", time(8, 30), "US CPI release"),
                               ("FOMC", time(14, 0), "FOMC rate decision")):
        for text in record.get(kind.lower(), []):
            at = datetime.combine(date.fromisoformat(text), stamp, NEW_YORK).astimezone(UTC)
            if moment <= at <= horizon:
                events.append(Event(kind, at, label, str(record.get(f"{kind.lower()}_source", ""))))
    events.sort(key=lambda e: e.at)
    return events, str(record.get("fetched_at", ""))[:10] or None


def main() -> int:  # pragma: no cover - CLI
    record = refresh()
    print(f"CPI dates: {len(record['cpi'])}, FOMC decisions: {len(record['fomc'])} -> {SNAPSHOT}")
    events, _ = upcoming()
    for event in events:
        print(f"  {event.at:%a %d %b %H:%M} UTC  {event.label}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = ["Event", "parse_cpi", "parse_fomc", "refresh", "upcoming"]
