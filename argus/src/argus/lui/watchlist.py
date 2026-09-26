"""The week ahead for a book: what is scheduled, what it touches, and how the names moved on it.

**Why this module exists, measured before it was written.** The readiness audit of 2026-09-25 asked
the hosted console "what should I watch this week?" and "what earnings and macro events should I
watch this week for my book?" and both were refused. Every piece of the answer already existed in
ARGUS and none of it was reachable by asking: the holdings' report dates
(:func:`argus.market.bitget_mcp.BitgetDataService.next_earnings`, with Yahoo's calendar as the
second source, exactly as `lui/research._fundamentals` reads them), the CPI and FOMC schedules
(`market/calendar.py`), this week's 8-Ks (`market/evidence.EdgarSource`), the MacKinlay event study
of how each traded name moved on CPI, Fed and its own earnings (`research/event_reactions.py`,
artefact `data/event_reactions.json`), and the US holiday calendar behind the session clock
(`lui/research._dual_clock`, QuantConnect Lean's). This module reads them, it does not rebuild
them.

**What was read first, and what was taken.** OpenBB's economic calendar
(`research/repos/OpenBB-upstream/openbb_platform/core/openbb_core/provider/standard_models/
economic_calendar.py:30-59`, AGPL-3.0, so patterns only, no code) models a release as date, country,
category, event, importance, source, consensus, previous, actual; its FRED provider
(`providers/fred/openbb_fred/models/economic_calendar.py:67-126`) scrapes FRED's release calendar
and forward-fills header dates onto event rows. Its earnings calendar
(`standard_models/calendar_earnings.py:25-40`) is report date, symbol, EPS consensus. Taken: one
release is one dated row that names the institution that publishes it. **Rejected**: (1) FRED's
aggregated calendar as the source of dates — the task and the console's own rule is the publisher's
schedule, so BLS, BEA, Census and the Federal Reserve are read directly (FRED is a republisher and
its page is one more place a date can be wrong); (2) the ``importance`` field — OpenBB passes on a
vendor's high/medium/low without saying how it was set, and a label nobody can check is exactly
what this console refuses to print. Neither OpenBB model knows what the trader holds; the part
this module adds is the book: which holdings each event touches, at what weight, and what ARGUS's
own event study measured for those names on that event type.

**Where the dates come from.** The official 2026 schedules, frozen in
`data/macro_calendar_2026.json` by :func:`refresh` with each page's URL, the route that answered
(bls.gov refuses scripted requests with HTTP 403, so it is read through the same public reader
`market/calendar.py` uses) and the retrieval time:

* CPI, PPI, Employment Situation (nonfarm payrolls) — BLS release schedules,
  https://www.bls.gov/schedule/news_release/{cpi,ppi,empsit}.htm, 08:30 New York time;
* GDP and Personal Income and Outlays (PCE) — BEA's release schedule, https://www.bea.gov/news/
  schedule, the time printed on each row;
* Advance Monthly Retail Trade (retail sales) — Census, https://www.census.gov/retail/
  release_schedule.html, 08:30;
* FOMC decisions and minutes — the Federal Reserve's own calendar feed,
  https://www.federalreserve.gov/json/calendar.json (the rows behind
  https://www.federalreserve.gov/newsevents/calendar.htm), 14:00.

The answer never fetches a schedule at request time: it reads the frozen file and states its date.
A release missing from the file is missing from the answer, never estimated.

**How "the biggest event" is chosen, and how far to trust it.** For each event the answer computes
the book's measured excess move: for every touched holding with an event-study row, weight x
(event-day 24h move / its ordinary 24h move - 1). That figure is only defined for CPI, FOMC
decisions and a holding's own earnings — ARGUS has no event study of payrolls, PCE, PPI, retail
sales, GDP or minutes, and says so on the line rather than borrowing a number. Events with a
positive measured excess rank first; unmeasured events follow in a stated convention (FOMC
decision, CPI, payrolls, the holdings' earnings by weight, PCE, PPI, retail sales, GDP, minutes),
labelled "Assumed:" because it is a convention, not a measurement of this book; events measured as
no larger than an ordinary day rank last.

**Weekends and holidays.** Bitget's stock perpetuals trade around the clock while the US stock
market does not. Every day in the window on which the NYSE is shut (weekend or a Lean-calendar
holiday) is named, with the holdings whose perpetual prices without its stock that day, and every
08:30 release is marked as landing before the 09:30 open, when the perpetual is the only price that
can move.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from argus.lui.answer import Source
from argus.lui.research import (
    FILING_LOOKBACK_DAYS,
    _dual_clock,
    _is_equity,
    _pairs,
    _raw_number,
    _strip_budget,
    _t,
    _yahoo_summary,
    book_pricing_note,
    parse_book,
    parse_budget,
    research_symbols,
    split_cash,
)

DATA = Path(__file__).resolve().parents[3] / "data"
MACRO_CALENDAR = DATA / "macro_calendar_2026.json"
DEFAULT_DAYS = 7
MAX_DAYS = 31
KEEP_FROM = "2025-10-01"
"""Older rows are dropped from the frozen file: the Fed feed reaches back to 2021, and the event
study reads its own history, not this calendar."""
NEW_YORK = ZoneInfo("America/New_York")
READER = "https://r.jina.ai/"

BLS_PAGES = {
    "CPI": "https://www.bls.gov/schedule/news_release/cpi.htm",
    "PPI": "https://www.bls.gov/schedule/news_release/ppi.htm",
    "NFP": "https://www.bls.gov/schedule/news_release/empsit.htm",
}
BEA_URL = "https://www.bea.gov/news/schedule"
CENSUS_URL = "https://www.census.gov/retail/release_schedule.html"
FED_URL = "https://www.federalreserve.gov/json/calendar.json"

KIND_NAMES = {
    "FOMC": "FOMC rate decision",
    "CPI": "CPI",
    "NFP": "nonfarm payrolls (Employment Situation)",
    "PCE": "PCE (Personal Income and Outlays)",
    "PPI": "PPI",
    "RETAIL": "retail sales (Advance Monthly Retail Trade)",
    "GDP": "GDP",
    "MINUTES": "FOMC minutes",
}
PUBLISHER = {"FOMC": "Federal Reserve", "MINUTES": "Federal Reserve", "CPI": "BLS", "PPI": "BLS",
             "NFP": "BLS", "PCE": "BEA", "GDP": "BEA", "RETAIL": "Census"}
STUDIED = {"CPI": "CPI", "FOMC": "FOMC"}
"""Macro kinds ARGUS's event study covers, mapped to its row kind. Earnings is the third."""

CONVENTION = ("FOMC", "CPI", "NFP", "EARNINGS", "PCE", "PPI", "RETAIL", "GDP", "MINUTES")
"""The order unmeasured events are ranked in — a stated convention, printed as an assumption."""

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


# --- the frozen official schedules ------------------------------------------------------------


def _get(url: str, timeout: float = 25.0) -> str:
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (argus-research)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return str(response.read().decode("utf-8-sig", errors="replace"))


def _flat(markup: str) -> str:
    """Tags to cell separators, entities decoded: HTML and the reader's Markdown parse alike."""
    text = html.unescape(re.sub(r"<[^>]+>", " | ", markup))
    return re.sub(r"(?:\s*\|\s*)+", " | ", re.sub(r"\s+", " ", text))


def _clock(text: str) -> str:
    """"8:30 AM" / "08:30 AM" / "2:00 p.m." as "08:30" / "14:00"."""
    match = re.match(r"\s*(\d{1,2}):(\d\d)\s*([ap])\.?m\.?", text, re.I)
    if match is None:
        raise ValueError(f"not a clock time: {text!r}")
    hour = int(match.group(1)) % 12 + (12 if match.group(3).lower() == "p" else 0)
    return f"{hour:02d}:{match.group(2)}"


def parse_bls(text: str, kind: str) -> list[dict[str, str]]:
    """Rows of a BLS release schedule: "| September 2026 | Oct. 02, 2026 | 08:30 AM |"."""
    rows: list[dict[str, str]] = []
    for ref, month, day, year, at in re.findall(
            r"\|\s*([A-Z][a-z]+ 20\d\d)\s*\|\s*([A-Z][a-z]{2})[a-z]*\.?\s+(\d{1,2}),\s+(20\d\d)"
            r"\s*\|\s*(\d{1,2}:\d\d\s*[AP]M)", _flat(text)):
        when = date(int(year), _MONTHS[month.lower()], int(day))
        rows.append({"kind": kind, "date": when.isoformat(), "time_et": _clock(at),
                     "title": f"{KIND_NAMES[kind]}, {ref}"})
    return rows


def parse_bea(markup: str) -> list[dict[str, str]]:
    """GDP and Personal Income and Outlays rows from BEA's schedule, under its "Year" header."""
    rows: list[dict[str, str]] = []
    text = _flat(markup)
    year = None
    for token in re.finditer(
            r"Year (20\d\d)|((?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December)) (\d{1,2}) \| (\d{1,2}:\d\d [AP]M) \|(.*?)(?=(?:January|"
            r"February|March|April|May|June|July|August|September|October|November|December) "
            r"\d{1,2} \| \d{1,2}:\d\d|Year 20\d\d|$)", text):
        if token.group(1):
            year = int(token.group(1))
            continue
        if year is None:
            continue
        title = re.sub(r"\s*\|\s*", " ", token.group(5)).strip()
        title = re.sub(r"^(?:N ews|D ata|News|Data)\s+", "", title).strip()
        kind = ("GDP" if re.match(r"GDP \((?:Advance|Second|Third)", title) else
                "PCE" if title.startswith("Personal Income and Outlays") else None)
        if kind is None:
            continue
        when = date(year, _MONTHS[token.group(2)[:3].lower()], int(token.group(3)))
        rows.append({"kind": kind, "date": when.isoformat(), "time_et": _clock(token.group(4)),
                     "title": title})
    unique = {(r["kind"], r["date"]): r for r in rows}
    return sorted(unique.values(), key=lambda r: (r["date"], r["kind"]))


def parse_census(markup: str) -> list[dict[str, str]]:
    """Advance Monthly Retail Trade release dates (08:30) from Census's schedule page."""
    text = _flat(markup)
    start = text.find("Advance Monthly Retail Trade Report")
    end = text.find("Monthly Retail Trade Report", start + 40)
    segment = text[start: end if end > 0 else None] if start >= 0 else ""
    rows: list[dict[str, str]] = []
    for ref, month, day, year in re.findall(
            r"([A-Z][a-z]+ 20\d\d) \| ([A-Z][a-z]+) (\d{1,2}), (20\d\d)", segment):
        when = date(int(year), _MONTHS[month[:3].lower()], int(day))
        rows.append({"kind": "RETAIL", "date": when.isoformat(), "time_et": "08:30",
                     "title": f"{KIND_NAMES['RETAIL']}, {ref}"})
    return rows


def parse_fed(raw: str) -> list[dict[str, str]]:
    """FOMC decisions ("FOMC Meeting", the statement at 2:00 p.m.) and minutes from the Fed feed."""
    rows: list[dict[str, str]] = []
    for event in json.loads(raw).get("events", []):
        title = str(event.get("title", "")).strip()
        if event.get("type") != "FOMC" or title not in ("FOMC Meeting", "FOMC Minutes"):
            continue
        month = str(event.get("month", ""))
        days = re.findall(r"\d+", str(event.get("days", "")))
        if not re.fullmatch(r"20\d\d-\d\d", month) or not days:
            continue
        when = date(int(month[:4]), int(month[5:]), int(days[-1]))
        note = re.sub(r"<[^>]+>", " ", html.unescape(str(event.get("description", ""))))
        meeting = re.search(r"(?:Meeting of|Two-day meeting,)\s*([A-Z][a-z]+ \d+\s*-\s*\d+)", note)
        kind = "FOMC" if title == "FOMC Meeting" else "MINUTES"
        rows.append({"kind": kind, "date": when.isoformat(),
                     "time_et": _clock(str(event.get("time", "2:00 p.m."))),
                     "title": f"{KIND_NAMES[kind]}"
                              + (f", meeting of {meeting.group(1)}" if meeting else "")})
    return sorted(rows, key=lambda r: (r["date"], r["kind"]))


def refresh(out: Path = MACRO_CALENDAR, *,
            fetch: Callable[[str], str] | None = None) -> dict[str, Any]:
    """Read every official schedule and write the frozen calendar. A schedule that fails keeps its
    previous rows (marked as such) rather than landing empty."""
    get = fetch or _get
    now = datetime.now(UTC).isoformat(timespec="seconds")
    releases: list[dict[str, str]] = []
    sources: dict[str, dict[str, Any]] = {}

    def record(kinds: Sequence[str], url: str, route: str, rows: list[dict[str, str]]) -> None:
        for row in rows:
            releases.append({**row, "source": url})
        for kind in kinds:
            count = sum(1 for r in rows if r["kind"] == kind)
            sources[kind] = {"url": url, "route": route, "retrieved_at": now, "rows": count}

    for kind, url in BLS_PAGES.items():
        try:
            try:
                text, route = get(url), "bls.gov"
                rows = parse_bls(text, kind)
            except Exception:
                rows = []
            if not rows:
                text, route = get(READER + url), "bls.gov via r.jina.ai"
                rows = parse_bls(text, kind)
            record((kind,), url, route, rows)
        except Exception as exc:
            sources[kind] = {"url": url, "error": type(exc).__name__, "retrieved_at": now}
    for kinds, url, parser in ((("GDP", "PCE"), BEA_URL, parse_bea),
                               (("RETAIL",), CENSUS_URL, parse_census),
                               (("FOMC", "MINUTES"), FED_URL, parse_fed)):
        try:
            record(kinds, url, url.split("/")[2], parser(get(url)))
        except Exception as exc:
            for kind in kinds:
                sources[kind] = {"url": url, "error": type(exc).__name__, "retrieved_at": now}
    previous = load_calendar(out)
    for kind in KIND_NAMES:
        if not any(r["kind"] == kind for r in releases) and previous is not None:
            kept = [r for r in previous.get("releases", []) if r.get("kind") == kind]
            releases.extend(kept)
            if kept:
                sources[kind] = {**previous.get("sources", {}).get(kind, {}),
                                 "note": "previous snapshot kept; this refresh failed"}
    releases = [r for r in releases if r["date"] >= KEEP_FROM]
    releases.sort(key=lambda r: (r["date"], r["time_et"], r["kind"]))
    payload = {"retrieved_at": now, "timezone": "America/New_York", "sources": sources,
               "releases": releases}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return payload


def load_calendar(path: Path = MACRO_CALENDAR) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


# --- the question -----------------------------------------------------------------------------

_WINDOW = re.compile(
    r"\b(?:this|next|coming)\s+(?:coming\s+)?week\b|\bweek\s+ahead\b|\b(?:next|coming)\s+"
    r"(?:\d{1,2}|few|couple(?:\s+of)?|two|2)\s+(?:days|weeks)\b|\bupcoming\b|\bcoming\s+up\b|"
    r"\bon\s+(?:the\s+)?deck\b|\bahead\s+(?:for|of\s+me|this)\b|\bthe\s+days?\s+ahead\b|"
    r"本周|这周|这个星期|这星期|下周|下个星期|未来\s*\d+\s*天|未来一周|接下来", re.I)
_CUE = re.compile(
    r"\bwatch(?:list|ing)?\b(?!\s+out)|\bkeep\s+(?:an\s+)?eyes?\s+on\b|\bcalendar\b|\bcatalysts?\b|"
    r"\bevents?\b|\bkey\s+dates\b|\b(?:data|macro|economic)\s+(?:releases?|prints?|data)\b|"
    r"\bearnings\b|\breport(?:s|ing)?\b|\bmacro\b|\bfomc\b|\bcpi\b|\bpayrolls?\b|\bnfp\b|"
    r"关注|留意|盯|日历|财报|事件|宏观|数据", re.I)
_ALONE = re.compile(
    r"\bwatch\s*list\b|\b(?:economic|macro|earnings)\s+calendar\b|\bweek\s+ahead\b|"
    r"\bweekly\s+(?:watch|preview|calendar)\b|经济日历|财经日历|财报日历", re.I)
_NOT_OURS = re.compile(
    r"\bthe\s+desk\b|\byou\s+guys\b|\b(?:did|have|has)\s+you\b|\byour\s+(?:trades?|decisions?|"
    r"record)\b|\bsharpe\b|\bdrawdown\b|\bhedg\w*|\bsentiment\b|\bnarrative\b|\bdriving\b|"
    r"\bsupport\b|\bresistance\b|\breact\w*|\bbehave\w*|\btypically\b|\bhistorically\b|"
    r"\busually\b|\bmovie|\bfilm\b|\btv\b|\bnetflix\b|\bshows?\b|\bseries\b|\bfootball\b|"
    r"\bgames?\b|\bhotel\b|\bknee\b|\byoutube\b|\bvideos?\b|\bwhere\s+will\b|\bwill\s+\w+\s+"
    r"(?:go|be|hit|pump|dump|rise|fall|crash)\b|\bprice\s+target\b|交易记录|会涨|会跌", re.I)
_BOOK_WORDS = re.compile(r"\bmy\s+(?:book|portfolio|holdings?|positions?|names|stocks|bag)\b|"
                         r"\bi\s+(?:hold|own)\b|持仓|组合|仓位", re.I)
_EARNINGS_ONLY = re.compile(r"\bearnings\b|\breport(?:s|ing)?\b|财报", re.I)
_WIDE_CUE = re.compile(r"\bwatch|\bcalendar\b|\bcatalysts?\b|\bevents?\b|\bmacro\b|\bdates\b|"
                       r"关注|留意|日历|事件|宏观", re.I)


def asks_for_watchlist(text: str) -> bool:
    """A request for the week (or days) ahead: what is scheduled that touches these names.

    Needs a window and a cue together, or one phrase that is both ("watchlist", "economic
    calendar", "week ahead"). Questions about how a name reacts to an event (the event study's),
    hedging, sentiment, levels, the desk's own record, and a single name's own report date (the
    fundamentals engine answers that one) are left to their engines."""
    if _NOT_OURS.search(text):
        return False
    if _ALONE.search(text):
        return True
    if not (_WINDOW.search(text) and _CUE.search(text)):
        return False
    named, _ = research_symbols(text)
    # "does NVDA report earnings next week" is the fundamentals engine's answer
    return not (len(named) == 1 and _EARNINGS_ONLY.search(text) and not _WIDE_CUE.search(text)
                and not _BOOK_WORDS.search(text))


def window(text: str, now: datetime, days: int | None = None) -> tuple[datetime, datetime, bool]:
    """The window asked for, and whether the question stated it. "next week" is next Monday to
    the following Monday, New York time; "next N days" is N days from now; the default is the next
    seven days, which is what "this week" means to someone asking on a weekend."""
    if days is not None:
        return now, now + timedelta(days=max(1, min(days, MAX_DAYS))), True
    counted = re.search(r"\b(?:next|coming)\s+(\d{1,2})\s+days\b|未来\s*(\d+)\s*天", text, re.I)
    if counted:
        n = int(counted.group(1) or counted.group(2))
        return now, now + timedelta(days=max(1, min(n, MAX_DAYS))), True
    if re.search(r"\b(?:next|coming)\s+(?:two|2)\s+weeks\b", text, re.I):
        return now, now + timedelta(days=14), True
    if re.search(r"\bnext\s+week\b|下周|下个星期", text, re.I):
        local = now.astimezone(NEW_YORK)
        monday = datetime.combine(local.date() + timedelta(days=7 - local.weekday()), time(0),
                                  NEW_YORK)
        return monday.astimezone(UTC), (monday + timedelta(days=7)).astimezone(UTC), True
    return now, now + timedelta(days=DEFAULT_DAYS), False


# --- the book -------------------------------------------------------------------------------


def resolve_book(question: str, book_text: str) -> tuple[dict[str, float], float, list[str]]:
    """The holdings to read the calendar against: the ones the question names (its weights, or
    equal weights said as assumed), else the saved book, else none. Returns weights, cash, lines."""
    named, notes = research_symbols(question)
    if named:
        if _pairs(question):
            book, cash = split_cash(question, parse_book(question))
            return book, cash, [f"Assumed: {n}." for n in notes]
        weight = 1.0 / len(named)
        return ({s: weight for s in named}, 0.0,
                [f"Assumed: no weights were stated, so the {len(named)} names asked about are "
                 f"read at {weight:.0%} each."] + [f"Assumed: {n}." for n in notes])
    if book_text.strip():
        body = _strip_budget(book_text) if parse_budget(book_text) is not None else book_text
        book, cash = split_cash(body, parse_book(body))
        if book:
            shown = ", ".join([f"{w:.0%} {_t(s)}" for s, w in book.items()]
                              + ([f"{cash:.0%} cash"] if cash else []))
            return book, cash, [f"Remembered: your saved book — {shown}"
                                f"{book_pricing_note(body)}."]
    return {}, 0.0, ["Assumed: no holdings were named and no book is saved, so this is the US "
                     "macro calendar alone — name them (\"40% NVDA, 30% MSFT, 30% BTC\") or save "
                     "them in My book to see earnings, filings and what each event touches."]


# --- live lookups (injectable, so tests run offline) ----------------------------------------


@dataclass(frozen=True)
class Report:
    ticker: str
    day: date
    timing: str
    """"after the close", "before the open", or "" when the source does not say."""
    source: str
    estimated: bool = False


def earnings_date(ticker: str, today: date) -> Report | None:
    """The next report on or after ``today``: Bitget's data service first, Yahoo's calendar second
    — the two sources `lui/research._fundamentals` reads, in its order."""
    from argus.market.bitget_mcp import shared_service

    try:
        row = shared_service().next_earnings(ticker)
    except Exception:
        row = {}
    stamp = str(row.get("report_date") or "")[:10]
    if re.fullmatch(r"\d{4}-\d\d-\d\d", stamp) and date.fromisoformat(stamp) >= today:
        timing = {"盘后": "after the close", "盘前": "before the open"}.get(
            str(row.get("is_trading_time") or ""), "")
        return Report(ticker, date.fromisoformat(stamp), timing, "Bitget equity calendar")
    try:
        summary = _yahoo_summary(ticker)
    except Exception:
        return None
    calendar = (summary.get("calendarEvents") or {}).get("earnings") or {}
    for node in calendar.get("earningsDate") or []:
        raw = _raw_number(node)
        if raw is None:
            continue
        day = datetime.fromtimestamp(raw, UTC).date()
        if day >= today:
            return Report(ticker, day, "", "Yahoo Finance earnings calendar",
                          estimated=bool(calendar.get("isEarningsDateEstimate")))
    return None


@dataclass(frozen=True)
class Filed:
    ticker: str
    day: date
    items: str


def recent_8k(tickers: Sequence[str], since: datetime) -> dict[str, list[Filed]]:
    """8-Ks each ticker filed since ``since``, from SEC EDGAR (`market/evidence.EdgarSource`)."""
    from argus.market.evidence import EdgarSource

    edgar = EdgarSource()
    out: dict[str, list[Filed]] = {}
    for ticker in tickers:
        out[ticker] = [Filed(ticker, f.filed.date(), f.item_summary)
                       for f in edgar.filings(ticker, since=since) if f.form.startswith("8-K")]
    return out


def _reactions(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        from argus.lui.answer import _notes_path

        path = _notes_path().parent / "event_reactions.json"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


# --- the answer -------------------------------------------------------------------------------


@dataclass
class Event:
    kind: str
    """A macro kind, or "EARNINGS"."""
    at: datetime
    title: str
    source: str
    touched: dict[str, float]
    timing: str = ""
    history: list[str] | None = None
    excess: float | None = None
    covered: float = 0.0


def _who(touched: Mapping[str, float]) -> str:
    return ", ".join(f"{_t(s)} {w:.0%}" for s, w in sorted(touched.items(), key=lambda x: -x[1]))


def _study(event: Event, report: Mapping[str, Any] | None) -> None:
    """Attach what ARGUS's event study measured for the touched names on this event type."""
    kind = STUDIED.get(event.kind) or ("earnings" if event.kind == "EARNINGS" else None)
    lines: list[str] = []
    if kind is None or report is None:
        name = KIND_NAMES.get(event.kind, event.kind)
        lines.append(f"How these names moved on past {name} releases was not measured — ARGUS "
                     f"has no event study of it." if kind is None else
                     "How these names moved on past events was not measured — the event study "
                     "artefact could not be read.")
        event.history = lines
        return
    rows = {r.get("symbol"): r for r in report.get("reactions", []) if r.get("kind") == kind}
    excess = 0.0
    covered = 0.0
    measured_any = False
    for symbol, weight in sorted(event.touched.items(), key=lambda x: -x[1]):
        row = rows.get(symbol)
        ticker = _t(symbol)
        label = {"CPI": "CPI releases", "FOMC": "Fed decisions",
                 "earnings": "its own results releases"}[kind]
        if row is None or row.get("size_ratio") is None:
            count = None if row is None else row.get("events")
            lines.append(f"{ticker} on {label}: was not measured — "
                         + ("it is not one of the names the event study covers."
                            if row is None else
                            f"only {count} clean events, fewer than the study's minimum of 5."))
            continue
        ratio = float(row["size_ratio"])
        dates = row.get("dates") or []
        car = row.get("average_car_bps")
        verdict = str(row.get("verdict", ""))
        lean = ("a reliable direction" if verdict.startswith("EFFECT ESTABLISHED") else
                "a direction only some tests support" if verdict.startswith("PARTIAL") else
                "no reliable direction")
        lines.append(f"{ticker} on {label}: moved {ratio:.1f}x its ordinary 24 hours over the day "
                     f"after each of {row.get('events')} since {str(dates[0])[:4] if dates else ''}"
                     f"{'-' + str(dates[0])[5:] if dates else ''}; average "
                     f"{float(car):+.0f}bps, {lean}." if car is not None else
                     f"{ticker} on {label}: moved {ratio:.1f}x its ordinary 24 hours; {lean}.")
        excess += weight * (ratio - 1.0)
        covered += weight
        measured_any = True
    event.history = lines
    if measured_any:
        event.excess, event.covered = excess, covered


def _closed_days(start: datetime, end: datetime) -> list[tuple[date, str]]:
    """Days in the window on which the NYSE is shut, from the session clock and Lean's holidays."""
    from argus.truth.clocks import SessionPhase

    clock, _ = _dual_clock()
    out: list[tuple[date, str]] = []
    day = start.astimezone(NEW_YORK).date()
    last = end.astimezone(NEW_YORK).date()
    while day <= last:
        noon = datetime.combine(day, time(12), NEW_YORK)
        if start <= noon.astimezone(UTC) + timedelta(hours=12) and noon.astimezone(UTC) <= end:
            phase = clock.phase(noon)
            if phase is SessionPhase.WEEKEND:
                out.append((day, "weekend"))
            elif phase is SessionPhase.HOLIDAY:
                out.append((day, "US market holiday"))
        day += timedelta(days=1)
    return out


def _span(days: Sequence[tuple[date, str]]) -> list[str]:
    """Consecutive closed days grouped: "Sat 26 to Sun 27 Sep (weekend)"."""
    groups: list[list[tuple[date, str]]] = []
    for item in days:
        if groups and (item[0] - groups[-1][-1][0]).days == 1 and item[1] == groups[-1][-1][1]:
            groups[-1].append(item)
        else:
            groups.append([item])
    return [(f"{g[0][0]:%a %d %b}" + (f" to {g[-1][0]:%a %d %b}" if len(g) > 1 else "")
             + f" ({g[0][1]})") for g in groups]


def _rank_key(event: Event) -> tuple[int, float, float]:
    if event.excess is not None and event.excess > 0:
        return (0, -event.excess, 0.0)
    if event.excess is None:
        order = CONVENTION.index(event.kind) if event.kind in CONVENTION else len(CONVENTION)
        weight = -sum(event.touched.values()) if event.kind == "EARNINGS" else 0.0
        return (1, float(order), weight)
    return (2, -(event.excess or 0.0), 0.0)


def watchlist(question: str, book_text: str = "", *, now: datetime | None = None,
              days: int | None = None, calendar_path: Path = MACRO_CALENDAR,
              reactions_path: Path | None = None,
              earnings_lookup: Callable[[str, date], Report | None] | None = None,
              filings_lookup: Callable[[Sequence[str], datetime], Mapping[str, Sequence[Filed]]]
              | None = None) -> tuple[list[str], list[Source], dict[str, Any]]:
    """The week ahead for a book: every scheduled US macro release and holding's report in the
    window, what each touches and at what weight, how those names moved on it before where ARGUS
    has measured it, this week's 8-Ks, and the days the stock market is shut while the perpetuals
    trade. Returns (lines, sources, data) like the research engines; the first line is the lead."""
    from argus.truth.coverage import ContextPool

    moment = now or datetime.now(UTC)
    start, end, stated = window(question, moment, days)
    book, cash, book_lines = resolve_book(question, book_text)
    equities = [s for s in book if _is_equity(s)]
    sources: list[Source] = []
    lines: list[str] = []

    calendar = load_calendar(calendar_path)
    events: list[Event] = []
    if calendar is None:
        lines.append("The frozen macro calendar (data/macro_calendar_2026.json) could not be read, "
                     "so no macro release is listed — none is estimated in its place.")
    else:
        for row in calendar.get("releases", []):
            hour, minute = (int(x) for x in str(row["time_et"]).split(":"))
            at = datetime.combine(date.fromisoformat(row["date"]), time(hour, minute),
                                  NEW_YORK).astimezone(UTC)
            if start <= at < end:
                events.append(Event(row["kind"], at, str(row["title"]), str(row["source"]),
                                    touched=dict(book)))
        retrieved = str(calendar.get("retrieved_at", ""))[:10]
        sources.append(Source(kind="venue", ref="BLS + BEA + Census + Federal Reserve schedules",
                              detail=f"frozen in {calendar_path.name}, read {retrieved}"))

    lookup = earnings_lookup or earnings_date
    filings_of = filings_lookup or recent_8k
    tickers = [_t(s) for s in equities]
    reports: dict[str, Report | None] = {}
    filed: Mapping[str, Sequence[Filed]] = {}
    filings_failed = False
    with ContextPool(max_workers=max(2, min(8, len(tickers) + 1))) as pool:
        pending = {t: pool.submit(lookup, t, moment.date()) for t in tickers}
        filed_job = pool.submit(filings_of, tickers, moment - timedelta(days=FILING_LOOKBACK_DAYS))
        for ticker, job in pending.items():
            try:
                reports[ticker] = job.result()
            except Exception:
                reports[ticker] = None
        try:
            filed = filed_job.result() if tickers else {}
        except Exception:
            filings_failed = True
    missing_dates: list[str] = []
    for symbol in equities:
        report = reports.get(_t(symbol))
        if report is None:
            missing_dates.append(_t(symbol))
            continue
        clock_at = {"after the close": time(16, 0), "before the open": time(8, 0)}.get(
            report.timing, time(12, 0))
        at = datetime.combine(report.day, clock_at, NEW_YORK).astimezone(UTC)
        if start.astimezone(NEW_YORK).date() <= report.day <= end.astimezone(NEW_YORK).date():
            events.append(Event("EARNINGS", at, f"{_t(symbol)} earnings", report.source,
                                touched={symbol: book[symbol]}, timing=report.timing
                                + (" (an estimated date)" if report.estimated else "")))
    if tickers:
        sources.append(Source(kind="venue", ref="bitget-mcp-server equity_calendar + Yahoo Finance",
                              detail=f"next report for {', '.join(tickers)}"))
        sources.append(Source(kind="venue", ref="SEC EDGAR submissions",
                              detail=f"8-Ks in the last {FILING_LOOKBACK_DAYS} days"))

    study = _reactions(reactions_path)
    for event in events:
        _study(event, study)
    if study is not None:
        sources.append(Source(kind="computation", ref="argus.research.event_reactions",
                              detail="MacKinlay event study, hourly Bitget candles; computed "
                                     + str(study.get("generated_at", ""))[:10]))
    events.sort(key=lambda e: e.at)
    ranked = sorted(events, key=_rank_key)
    closed = _closed_days(start, end)
    last_day = end.astimezone(NEW_YORK) - timedelta(minutes=1)
    span = f"{start.astimezone(NEW_YORK):%a %d %b} to {last_day:%a %d %b}"

    # the lead
    if ranked:
        top = ranked[0]
        what = (f"{_t(next(iter(top.touched)))} reports {top.at.astimezone(NEW_YORK):%a %d %b}"
                + (f" {top.timing}" if top.timing else "") if top.kind == "EARNINGS" else
                f"{KIND_NAMES[top.kind]} on {top.at:%a %d %b} at {top.at:%H:%M} UTC")
        share = sum(top.touched.values())
        if top.excess is not None and top.excess > 0:
            why = (f"it is the event ARGUS has measured moving this book most: the names it "
                   f"touches ({share:.0%} of the book) have moved a weighted "
                   f"{top.excess / max(top.covered, 1e-9) + 1:.1f}x their ordinary day on it"
                   if top.covered else "")
        elif top.excess is None:
            why = ("none of the window's events has a measured effect on these names (ARGUS's "
                   "event study covers CPI, Fed decisions and earnings), so it leads by the "
                   "stated convention" if book else
                   "it leads by the stated convention; no book was given to measure against")
        else:
            why = "every event in the window was measured as no larger than an ordinary day"
        whose = "for this book" if book else "on the US calendar"
        lead = f"Actionable: the biggest event {whose} in {span} is {what} — {why}"
        if book and top.kind != "EARNINGS":
            lead += f"; it touches the whole book ({_who(top.touched)})"
        lines.insert(0, lead + ". Size for the move, not a direction.")
    else:
        lines.insert(0, "Actionable: nothing on the official US macro calendar"
                        + (" or your holdings' earnings calendars" if tickers else "")
                        + f" falls in {span} — the scheduled risk this week is only what the "
                          f"market does on its own"
                        + (f"; {', '.join(missing_dates)} had no report date to check."
                           if missing_dates else "."))

    lines.extend(book_lines)
    if not stated:
        lines.append(f"Assumed: the next {DEFAULT_DAYS} days ({span}), since no window was stated.")
    if cash:
        lines.append(f"Assumed: {cash:.0%} of the book is cash, which no event touches.")

    for event in events:
        local = event.at.astimezone(NEW_YORK)
        if event.kind == "EARNINGS":
            symbol = next(iter(event.touched))
            line = (f"Scheduled: {_t(symbol)} earnings {local:%a %d %b}"
                    + (f", {event.timing}" if event.timing else ", time of day not published")
                    + f" ({event.source}) — {event.touched[symbol]:.0%} of the book"
                    + ("; the perpetual carries the gap overnight while the stock is shut"
                       if "after the close" in event.timing or "before the open" in event.timing
                       else "") + ".")
        else:
            before_open = local.time() < time(9, 30)
            line = (f"Scheduled: {event.title} — {PUBLISHER[event.kind]}, {local:%a %d %b} "
                    f"{local:%H:%M} New York ({event.at:%H:%M} UTC)"
                    + (" — before the 09:30 open, so the stock perpetuals are the first price to "
                       "react" if before_open else "")
                    + (f"; touches the whole book ({_who(event.touched)})" if book else "") + ".")
        lines.append(line)
        lines.extend(event.history or [])

    if closed:
        held = ", ".join(_t(s) for s in equities)
        lines.append("Session: the US stock market is shut " + "; ".join(_span(closed))
                     + (f" — Bitget's {held} perpetuals keep trading, so they price without "
                        f"their stock and any gap is taken there first." if held else
                        " — Bitget's perpetuals keep trading while it is."))

    later = [r for r in reports.values() if r is not None
             and not start.astimezone(NEW_YORK).date() <= r.day <= end.astimezone(NEW_YORK).date()]
    if later:
        lines.append("Next report: " + "; ".join(
            f"{r.ticker} {r.day:%a %d %b}" + (" (estimated)" if r.estimated else "")
            for r in sorted(later, key=lambda r: r.day))
            + " — outside the window, so no holding carries an earnings gap in it"
            + (" beyond those listed above." if any(e.kind == "EARNINGS" for e in events)
               else "."))
    if tickers:
        if filings_failed:
            lines.append("SEC EDGAR could not be read just now, so whether a holding filed an 8-K "
                         "this week was not checked.")
        else:
            found = [f for t in tickers for f in filed.get(t, [])]
            for f in found[:6]:
                lines.append(f"Filed: {f.ticker} 8-K on {f.day:%a %d %b} ({f.items}) — "
                             f"{book[next(s for s in equities if _t(s) == f.ticker)]:.0%} of the "
                             f"book; read it before the week's events.")
            if not found:
                lines.append(f"No 8-K filed in the last {FILING_LOOKBACK_DAYS} days by "
                             f"{', '.join(tickers)} (SEC EDGAR, read now).")
        if missing_dates:
            lines.append(f"No upcoming report date could be read for {', '.join(missing_dates)} "
                         f"from Bitget's calendar or Yahoo's, so an earnings date inside the "
                         f"window was not ruled out.")
    unmeasured = [e for e in events if e.excess is None]
    if unmeasured and len(events) > 1:
        lines.append("Assumed: events without a measured effect on these names are ranked FOMC "
                     "decision, CPI, payrolls, your holdings' earnings (largest weight first), "
                     "PCE, PPI, retail sales, GDP, minutes — a convention, not a measurement.")
    retrieved = str((calendar or {}).get("retrieved_at", ""))[:10]
    lines.append("Data: release dates from the BLS, BEA, Census and Federal Reserve schedules "
                 f"(frozen {retrieved} in data/macro_calendar_2026.json with each page's URL)"
                 + ("; report dates from Bitget's equity calendar, Yahoo's as second source; 8-Ks "
                    "from SEC EDGAR" if tickers else "")
                 + "; past moves from ARGUS's event study; closed days from Lean's US holiday "
                   "calendar. This is analysis, not advice — you make the call.")
    data = {"window": {"start": start.isoformat(), "end": end.isoformat(), "stated": stated},
            "book": dict(book), "cash": cash,
            "events": [{"kind": e.kind, "at": e.at.isoformat(), "title": e.title,
                        "source": e.source, "touched": e.touched, "excess": e.excess,
                        "covered": e.covered} for e in events],
            "ranked": [e.title for e in ranked],
            "closed_days": [{"day": d.isoformat(), "why": why} for d, why in closed],
            "missing_report_dates": missing_dates}
    return lines, sources, data


def main() -> int:  # pragma: no cover - CLI
    record = refresh()
    counts = {k: sum(1 for r in record["releases"] if r["kind"] == k) for k in KIND_NAMES}
    print(f"{MACRO_CALENDAR}: {counts}")
    for kind, meta in record["sources"].items():
        print(f"  {kind}: {meta}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = ["MACRO_CALENDAR", "Filed", "Report", "asks_for_watchlist", "earnings_date",
           "load_calendar", "parse_bea", "parse_bls", "parse_census", "parse_fed", "recent_8k",
           "refresh", "resolve_book", "watchlist", "window"]
