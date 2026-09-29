"""Off-exchange (ATS) volume in an underlying, from FINRA's free weekly OTC transparency data.

The perception comparison against OpenBB (`eval/perception_breadth.py`, 2026-09-28) found dark-pool
volume answering for ten of ten underlyings through OpenBB's ``finra`` provider and nowhere here.
That provider (``openbb_finra/utils/helpers.py``, AGPL-3.0) was read for how FINRA's query API is
shaped; nothing of it is used. The API itself is FINRA's and is public.

**Two requests, because the API insists.** FINRA refuses to sort ``weeklySummary`` unless every
partition key is fixed, so the latest published weeks are read first from
``weeklyDownloadDetails`` (sorted newest first), and each week is then asked for by exact date.

**The lag is the most important thing to say about it.** ATS data for NMS Tier 1 stocks is
published about two weeks after the week ends: on 2026-09-28 the newest week was the one starting
2026-09-07. So this is never a live signal and is not presented as one. What it can say is whether
off-exchange trading in a name is running above or below its own recent weeks, which is what the
claim reports, with the week's date in it.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from argus.truth import http
from argus.truth.clocks import us_equity_holidays
from argus.truth.evidence import Evidence

WEEKS_URL = "https://api.finra.org/data/group/otcMarket/name/weeklyDownloadDetails"
SUMMARY_URL = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"
SUMMARY_TYPE = "ATS_W_SMBL"
"""ATS volume by symbol. ``OTC_W_SMBL`` is non-ATS OTC (internalised retail flow), a different
thing, and not what "dark pool" means."""

TIERS = ("T1", "T2")
"""NMS Tier 1 (the S&P 500, the Russell 1000 and the large ETFs) first, then Tier 2. Ten of the
twelve rTokens' underlyings are Tier 1; TQQQ and SQQQ are Tier 2, which FINRA publishes about four
weeks after the week rather than two (on 2026-09-28 their newest week began 2026-08-24)."""

TIER_2 = frozenset({"TQQQ", "SQQQ"})
"""Read at Tier 2 first. Asking Tier 1 first cost them five empty requests and pushed the whole
read past the workbench's deadline (13.4s for TQQQ, 2026-09-29)."""

WEEKS_TTL_S = 3600.0
"""FINRA publishes weekly; the list of published weeks is read once an hour, not per symbol."""

WEEKS = 5
"""The newest week plus four before it, the baseline the newest is compared against."""

TIMEOUT_S = 20.0
_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}


class DarkPoolError(RuntimeError):
    """FINRA answered, but not with anything this module can report."""


def _post(url: str, body: dict[str, Any], *, timeout: float) -> Any:
    raw = http.fetch(url, data=json.dumps(body).encode("utf-8"), method="POST",
                     headers=_HEADERS, timeout=timeout)
    # FINRA answers a query with no matching rows with an empty body, not with ``[]``.
    return json.loads(raw.decode("utf-8")) if raw.strip() else []


def _eq(field: str, value: str) -> dict[str, str]:
    return {"compareType": "EQUAL", "fieldName": field, "fieldValue": value}


_WEEKS_CACHE: dict[tuple[str, int], tuple[float, list[str]]] = {}
_WEEKS_LOCK = threading.Lock()


def latest_weeks(tier: str = "T1", *, n: int = WEEKS, timeout: float = TIMEOUT_S) -> list[str]:
    """The ``n`` newest published week-start dates, newest first."""
    with _WEEKS_LOCK:
        cached = _WEEKS_CACHE.get((tier, n))
        if cached is not None and time.monotonic() - cached[0] < WEEKS_TTL_S:
            return list(cached[1])
    rows = _post(WEEKS_URL, {
        "compareFilters": [_eq("summaryTypeCode", SUMMARY_TYPE), _eq("tierIdentifier", tier)],
        "fields": ["weekStartDate"], "limit": n, "sortFields": ["-weekStartDate"],
    }, timeout=timeout)
    weeks = [str(r["weekStartDate"]) for r in rows or []]
    if weeks:
        with _WEEKS_LOCK:
            _WEEKS_CACHE[(tier, n)] = (time.monotonic(), weeks)
    return weeks


@dataclass(frozen=True)
class AtsWeek:
    week_start: date
    shares: int
    trades: int
    notional: float
    published: str

    @property
    def trading_days(self) -> int:
        """Weekdays in the week that were not US equity holidays. The week of 2026-09-07 had
        four (Labor Day), and comparing its total with a five-day week reads a holiday as a
        collapse in off-exchange trading."""
        holidays = us_equity_holidays()
        days = (self.week_start + timedelta(days=i) for i in range(5))
        return sum(1 for d in days if d not in holidays)

    @property
    def weeks_late(self) -> int | None:
        """Whole weeks from the end of the week to its first publication."""
        try:
            published = date.fromisoformat(self.published[:10])
        except ValueError:
            return None
        return max(0, (published - (self.week_start + timedelta(days=4))).days // 7)

    @property
    def shares_per_day(self) -> float:
        return self.shares / self.trading_days if self.trading_days else 0.0


def week_for(symbol: str, week_start: str, tier: str, *,
             timeout: float = TIMEOUT_S) -> AtsWeek | None:
    rows = _post(SUMMARY_URL, {
        "compareFilters": [_eq("weekStartDate", week_start), _eq("tierIdentifier", tier),
                           _eq("summaryTypeCode", SUMMARY_TYPE),
                           _eq("issueSymbolIdentifier", symbol.upper())],
        "limit": 5,
    }, timeout=timeout)
    if not rows:
        return None
    row = rows[0]
    return AtsWeek(week_start=date.fromisoformat(str(row["weekStartDate"])),
                   shares=int(row.get("totalWeeklyShareQuantity") or 0),
                   trades=int(row.get("totalWeeklyTradeCount") or 0),
                   notional=float(row.get("totalNotionalSum") or 0.0),
                   published=str(row.get("initialPublishedDate") or ""))


@dataclass(frozen=True)
class DarkPool:
    symbol: str
    tier: str
    weeks: tuple[AtsWeek, ...]
    """Newest first."""

    @property
    def latest(self) -> AtsWeek:
        return self.weeks[0]

    @property
    def vs_baseline(self) -> float | None:
        """The newest week's ATS shares per trading day over the mean of the weeks before it,
        minus one. Per trading day, so a holiday week is not read as a drop."""
        before = [w.shares_per_day for w in self.weeks[1:]]
        if not before or sum(before) == 0:
            return None
        return self.latest.shares_per_day / (sum(before) / len(before)) - 1.0

    def claim(self) -> str:
        w = self.latest
        size = w.shares / w.trades if w.trades else 0.0
        late = w.weeks_late
        parts = [f"{self.symbol} dark-pool (ATS) volume, FINRA week of {w.week_start.isoformat()} "
                 f"(first published {w.published}"
                 + (f", {late} weeks after the week" if late is not None else "")
                 + "; FINRA may revise a week after it is first published): "
                 f"{w.shares:,} shares in {w.trades:,} trades, ${w.notional / 1e9:.2f}bn, "
                 f"average trade {size:.0f} shares"]
        change = self.vs_baseline
        if change is not None:
            parts.append(f"{change:+.0%} per trading day against its previous "
                         f"{len(self.weeks) - 1} weeks' mean ({w.trading_days}-day week)")
        return "; ".join(parts)


def dark_pool(symbol: str, *, timeout: float = TIMEOUT_S) -> DarkPool:
    order = tuple(reversed(TIERS)) if symbol.upper() in TIER_2 else TIERS
    for tier in order:
        weeks = latest_weeks(tier, timeout=timeout)
        # The weeks are independent requests, read side by side.
        with ThreadPoolExecutor(max_workers=max(1, len(weeks))) as pool:
            got = list(pool.map(lambda week, t=tier: week_for(symbol, week, t, timeout=timeout),
                                weeks))
        found = sorted((w for w in got if w is not None), key=lambda w: w.week_start,
                       reverse=True)
        if found:
            return DarkPool(symbol=symbol.upper(), tier=tier, weeks=tuple(found))
    raise DarkPoolError(f"{symbol}: no ATS rows in FINRA's newest {WEEKS} weeks at either tier")


def dark_pool_evidence(symbol: str, *, as_of: datetime,
                       timeout: float = TIMEOUT_S) -> tuple[list[Evidence], str]:
    """One evidence item and a status line. Point in time: a week published after ``as_of`` is
    never shown, so a replay sees what was public then."""
    try:
        pool = dark_pool(symbol, timeout=timeout)
    except (http.RpcError, DarkPoolError, ValueError, KeyError) as exc:
        return [], f"darkpool:{symbol}: unavailable ({http.reason_of(exc)})"
    visible = tuple(w for w in pool.weeks
                    if w.published and date.fromisoformat(w.published) <= as_of.date())
    if not visible:
        return [], f"darkpool:{symbol}: no week published by {as_of.date()}"
    pool = DarkPool(symbol=pool.symbol, tier=pool.tier, weeks=visible)
    # Available from the day it was first published, which is what a point-in-time reader of
    # the record needs; the week itself is a fortnight or more older still.
    published = datetime.fromisoformat(pool.latest.published[:10]).replace(tzinfo=UTC)
    return [Evidence(id=f"darkpool-{symbol}-{pool.latest.week_start.isoformat()}",
                     claim=pool.claim(), source="news", available_at=min(published, as_of),
                     credibility=0.9)], (
        f"darkpool:{symbol}: week of {pool.latest.week_start.isoformat()}")


def pools(symbols: Sequence[str]) -> dict[str, DarkPool | str]:
    out: dict[str, DarkPool | str] = {}
    for symbol in symbols:
        try:
            out[symbol] = dark_pool(symbol)
        except (http.RpcError, DarkPoolError, ValueError, KeyError) as exc:
            out[symbol] = http.reason_of(exc)
    return out


__all__ = [
    "AtsWeek",
    "DarkPool",
    "DarkPoolError",
    "dark_pool",
    "dark_pool_evidence",
    "latest_weeks",
    "pools",
    "week_for",
]
