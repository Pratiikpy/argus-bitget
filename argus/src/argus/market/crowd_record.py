"""How good the Polymarket crowd has been: Brier scores on resolved markets of the same kind.

`market/prediction.py` quotes what money is betting today. A trader reading "Polymarket prices
bitcoin above $120,000 by December at 18%" also needs to know what an 18% from this crowd has been
worth. This module answers that from the record: it pulls resolved markets of the same family
("Bitcoin above ___ on <date>?", "Fed decision in <month>?"), reads each market's YES price a fixed
time before it closed from Polymarket's CLOB, and scores those prices against how the market
resolved.

What was taken, and from where (read before building):

- ``evan-kolberg/prediction-market-backtesting`` (LGPL-3.0 for the files read, so the method is
  rebuilt here, no code taken): Brier is ``(p - y)^2`` per market
  (`analysis/legacy_plot_adapter.py:122-165`, ``prepare_cumulative_brier_advantage``), and a
  market that resolved 50-50 is dropped rather than scored as 0.5, because ``(p - 0.5)^2`` always
  favours a forecast near one half (`adapters/prediction_market/backtest_utils.py:316-320`).
  Their "user" forecast is a rolling mean of the market's own price; here the baselines are the
  two a reader can check by hand instead: a coin (always 50%, Brier 0.25) and the base rate (the
  share of these markets that resolved YES, said as in-sample).
- Gamma ``/public-search`` with ``events_status=closed`` lists resolved events (5,243 for "Bitcoin
  above" on 2026-10-04). CLOB ``/prices-history`` is read over a window around the moment scored
  (``startTs``/``endTs``, hourly): ``interval=max`` at hourly fidelity returned nothing for a market
  closed two months earlier (``bitcoin-above-on-august-1-2026``, probed 2026-10-04) while the
  window returned 12 points.

Where this departs from a plain average, and why:

- A strike ladder inflates the score. "Bitcoin above $58,000" a day before a $110,000 close is
  priced at 99.9% and scores a near-perfect Brier that says nothing about judgement. So the record
  is also scored on the contested markets only — those priced between 20% and 80% at the lead —
  and that is the figure the answer leads with when there are enough of them.
- Reliability is shown by band (the crowd said 60-80%; it happened N%), the calibration table
  forecasters use, so "an 18% from this crowd" can be read against the band it falls in.
- Only markets above Polymarket's $10,000 volume floor (`prediction.VOLUME_FLOOR`) count: below
  it a price is one trader's view.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from argus.market.prediction import CLOB, GAMMA, SEARCH_TERMS, VOLUME_FLOOR, PredictionError
from argus.truth import http
from argus.truth.paths import DATA_DIR

LEAD: Final = timedelta(hours=24)
"""The price is read this long before the market's end: a day ahead is a forecast, the last
minutes are a quote of an outcome already visible."""
CONTESTED: Final = (0.2, 0.8)
MAX_MARKETS: Final = 2000
PAGES: Final = 3
BANDS: Final = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001))
CACHE_SECONDS: Final = 3600.0

_MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|"
           "December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
_DAYS = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"


def family(title: str) -> str:
    """An event title with its dates, times and figures replaced: the "same kind" of question.

    "Bitcoin above ___ on October 3?" and "Bitcoin above ___ on February 23?" are one family;
    "Bitcoin above ___ on April 6, 9PM ET?" (an hourly ladder) is another."""
    out = re.sub(rf"\b(?:{_MONTHS}|{_DAYS})\b\.?", "<d>", title)
    out = re.sub(r"\$?\d[\d,.]*\s*(?:[kKmMbB]|[AP]M)?\b", "#", out)
    out = re.sub(r"#\s*:\s*#", "#", out)
    out = re.sub(r"(?:<d>\s*)+", "<d> ", out)
    return re.sub(r"\s+", " ", out).strip()


@dataclass(frozen=True, slots=True)
class Resolved:
    question: str
    family: str
    token: str
    ends: datetime
    outcome: float
    volume: float


@dataclass(frozen=True, slots=True)
class Scored:
    market: Resolved
    price: float
    """The YES price ``LEAD`` before the end."""

    @property
    def brier(self) -> float:
        return (self.price - self.market.outcome) ** 2


def _outcome(market: dict[str, Any]) -> float | None:
    """1.0 or 0.0 for a cleanly resolved market; None for 50-50, voided or still open."""
    try:
        prices = [float(p) for p in json.loads(market.get("outcomePrices") or "[]")]
    except (ValueError, TypeError):
        return None
    if len(prices) != 2 or not market.get("closed"):
        return None
    if prices == [1.0, 0.0]:
        return 1.0
    if prices == [0.0, 1.0]:
        return 0.0
    return None


def _token(market: dict[str, Any]) -> str:
    try:
        return str(json.loads(market.get("clobTokenIds") or "[]")[0])
    except (ValueError, IndexError, TypeError):
        return ""


def resolved_from(events: list[dict[str, Any]], terms: tuple[str, ...], *,
                  floor: float = VOLUME_FLOOR) -> list[Resolved]:
    """Cleanly resolved markets above the volume floor whose question names one of ``terms``."""
    patterns = [re.compile(rf"(?<![\w&]){re.escape(t)}(?![\w])", re.I) for t in terms]
    seen: set[str] = set()
    out: list[Resolved] = []
    for event in events:
        kind = family(str(event.get("title") or ""))
        for market in event.get("markets") or []:
            question = str(market.get("question") or "")
            outcome = _outcome(market)
            token = _token(market)
            if outcome is None or not token or token in seen:
                continue
            if not any(p.search(question) for p in patterns):
                continue
            volume = float(market.get("volumeNum") or 0.0)
            try:
                ends = datetime.fromisoformat(str(market.get("endDate")).replace("Z", "+00:00"))
            except ValueError:
                continue
            if volume < floor:
                continue
            seen.add(token)
            out.append(Resolved(question=question, family=kind, token=token, ends=ends,
                                outcome=outcome, volume=volume))
    out.sort(key=lambda r: r.ends, reverse=True)
    return out


_cache: dict[str, tuple[float, Any]] = {}
_lock = threading.Lock()


def _cached(key: str, load: Any) -> Any:
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_SECONDS:
            return hit[1]
    value = load()
    with _lock:
        _cache[key] = (now, value)
    return value


def _closed_events(term: str, *, timeout: float = 15.0) -> list[dict[str, Any]]:
    def load() -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for page in range(1, PAGES + 1):
            try:
                found = http.fetch_json(f"{GAMMA}/public-search", timeout=timeout, params={
                    "q": term, "events_status": "closed", "limit_per_type": 50, "page": page})
            except http.RpcError as exc:
                if events:
                    break
                raise PredictionError(f"Polymarket search for resolved {term!r} markets failed: "
                                      f"{http.reason_of(exc)}") from exc
            events.extend((found or {}).get("events") or [])
            if not ((found or {}).get("pagination") or {}).get("hasMore"):
                break
        return events

    result: list[dict[str, Any]] = _cached(f"events:{term}", load)
    return result


def _path(token: str, ends: datetime, lead: timedelta = LEAD, *,
          timeout: float = 15.0) -> list[tuple[int, float]]:
    """The hourly YES prices from twelve hours before the scored moment to one hour after it."""
    at = int((ends - lead).timestamp())

    def load() -> list[tuple[int, float]]:
        try:
            found = http.fetch_json(f"{CLOB}/prices-history", timeout=timeout, params={
                "market": token, "startTs": at - 12 * 3600, "endTs": at + 3600, "fidelity": 60})
        except http.RpcError:
            return []
        out = []
        for point in (found or {}).get("history") or []:
            try:
                out.append((int(point["t"]), float(point["p"])))
            except (KeyError, TypeError, ValueError):
                continue
        return sorted(out)

    result: list[tuple[int, float]] = _cached(f"path:{token}:{at}", load)
    return result


def price_at_lead(path: list[tuple[int, float]], ends: datetime,
                  lead: timedelta = LEAD) -> float | None:
    """The last traded price at or before ``ends - lead``, if one exists within six hours of it.

    A market that only opened after that moment has no forecast to score at this lead."""
    at = int((ends - lead).timestamp())
    before = [(t, p) for t, p in path if t <= at]
    if not before or at - before[-1][0] > 6 * 3600:
        return None
    return before[-1][1]


def score(markets: list[Resolved], *, lead: timedelta = LEAD,
          path: Any = None) -> list[Scored]:
    """Each market's price ``lead`` before its end, paired with its outcome."""
    path = path or _path
    chosen = markets[:MAX_MARKETS]
    with ThreadPoolExecutor(max_workers=16) as pool:
        paths = list(pool.map(lambda m: path(m.token, m.ends, lead), chosen))
    out = []
    for market, history in zip(chosen, paths, strict=True):
        price = price_at_lead(history, market.ends, lead)
        if price is not None:
            out.append(Scored(market=market, price=price))
    return out


@dataclass(frozen=True, slots=True)
class Record:
    family: str
    scored: tuple[Scored, ...]

    @property
    def n(self) -> int:
        return len(self.scored)

    @property
    def brier(self) -> float:
        return sum(s.brier for s in self.scored) / self.n

    @property
    def base_rate(self) -> float:
        return sum(s.market.outcome for s in self.scored) / self.n

    @property
    def base_rate_brier(self) -> float:
        rate = self.base_rate
        return sum((rate - s.market.outcome) ** 2 for s in self.scored) / self.n

    def contested(self) -> Record:
        low, high = CONTESTED
        return Record(self.family, tuple(s for s in self.scored if low <= s.price <= high))

    def bands(self) -> list[tuple[float, float, int, float, float]]:
        """(low, high, count, mean price, share that resolved YES) for each band with markets."""
        out = []
        for low, high in BANDS:
            inside = [s for s in self.scored if low <= s.price < high]
            if inside:
                out.append((low, min(high, 1.0), len(inside),
                            sum(s.price for s in inside) / len(inside),
                            sum(s.market.outcome for s in inside) / len(inside)))
        return out


SERIES: Final = {"BTC": ("bitcoin-above-on", "bitcoin-price-on"),
                  "ETH": ("ethereum-above-on", "ethereum-price-on"),
                  "SOL": ("solana-above-on", "solana-price-on"),
                  "XRP": ("xrp-above-on", "xrp-price-on")}
"""Polymarket's daily series, one event a day under a dated slug: "bitcoin-above-on-october-3-2026"
is an eleven-strike ladder that closes at noon ET. Search returns these in no date order and only a
few days of them (2026-10-04: 11 scored markets, all one day), so the record walks the series back
day by day instead."""
SERIES_DAYS: Final = 45


def _event(slug: str, *, timeout: float = 15.0) -> list[dict[str, Any]]:
    def load() -> list[dict[str, Any]]:
        try:
            found = http.fetch_json(f"{GAMMA}/events", timeout=timeout, params={"slug": slug})
        except http.RpcError:
            return []
        return found if isinstance(found, list) else []

    result: list[dict[str, Any]] = _cached(f"event:{slug}", load)
    return result


def series_events(stem: str, *, days: int = SERIES_DAYS, today: datetime | None = None,
                  event: Any = None) -> list[dict[str, Any]]:
    """The series' events for each of the last ``days`` days, under either slug form Polymarket
    has used ("...-october-3-2026" since 2026, "...-september-20" before)."""
    event = event or _event
    day = (today or datetime.now(UTC)).date()
    slugs = []
    for back in range(1, days + 1):
        d = day - timedelta(days=back)
        month = f"{d:%B}".lower()
        slugs.append((f"{stem}-{month}-{d.day}-{d.year}", f"{stem}-{month}-{d.day}"))
    with ThreadPoolExecutor(max_workers=12) as pool:
        dated = list(pool.map(lambda pair: event(pair[0]) or event(pair[1]), slugs))
    return [e for found in dated for e in found]


SNAPSHOT_PATH: Final = DATA_DIR / "crowd_record.json"
"""Resolved markets never change, so they are scored once and stored: walking 45 days of a series
live took 50 s (313 price paths). The answer reads the snapshot and fetches only the days since."""
SNAPSHOT_DAYS: Final = 90
LIVE_DAYS: Final = 10


def _to_row(item: Scored) -> dict[str, Any]:
    m = item.market
    return {"question": m.question, "family": m.family, "token": m.token,
            "ends": m.ends.isoformat(), "outcome": m.outcome, "volume": m.volume,
            "price": item.price}


def _from_row(row: dict[str, Any]) -> Scored:
    return Scored(market=Resolved(question=row["question"], family=row["family"],
                                  token=row["token"], ends=datetime.fromisoformat(row["ends"]),
                                  outcome=float(row["outcome"]), volume=float(row["volume"])),
                  price=float(row["price"]))


def load_snapshot(path: Any = None) -> dict[str, Any]:
    try:
        found = json.loads((path or SNAPSHOT_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def _series_scored(stem: str, terms: tuple[str, ...], days: int, *, today: datetime | None,
                   event: Any, path: Any, lead: timedelta) -> list[Scored]:
    markets = resolved_from(series_events(stem, days=days, today=today, event=event), terms)
    return score(markets, lead=lead, path=path)


def build_snapshot(*, days: int = SNAPSHOT_DAYS, today: datetime | None = None,
                   event: Any = None, path: Any = None) -> dict[str, Any]:
    """Every series in ``SERIES`` scored over ``days`` days, as the stored snapshot."""
    series: dict[str, list[dict[str, Any]]] = {}
    for base, stems in SERIES.items():
        terms = SEARCH_TERMS.get(base, (base.lower(),))
        for stem in stems:
            series[stem] = [_to_row(s) for s in _series_scored(
                stem, terms, days, today=today, event=event, path=path, lead=LEAD)]
    return {"built": (today or datetime.now(UTC)).isoformat(timespec="seconds"),
            "lead_hours": LEAD.total_seconds() / 3600, "days": days, "series": series}


def record_for(symbol: str, *, question_family: str | None = None, search: Any = None,
               path: Any = None, event: Any = None, lead: timedelta = LEAD,
               today: datetime | None = None, snapshot: dict[str, Any] | None = None,
               ) -> Record | None:
    """The crowd's scored record on resolved markets about the name: the daily "above" ladder
    (or the "price on" range ladder when ``question_family`` names it) from the snapshot plus the
    days since it was built, read live; for a name with no daily series, the busiest family that
    search returns, read live."""
    base = symbol.removesuffix("USDT")
    base = base.removesuffix("STOCK") if base.endswith("STOCK") else base
    terms = SEARCH_TERMS.get(base, (base.lower(),))
    stems = SERIES.get(base, ())
    if stems:
        # "Bitcoin price on <date>?" is the range ladder; anything else reads the "above" ladder
        stem = (stems[1] if question_family and " price on " in question_family.lower()
                else stems[0])
        stored = snapshot if snapshot is not None else load_snapshot()
        rows = [_from_row(r) for r in (stored.get("series") or {}).get(stem, [])]
        clock = today or datetime.now(UTC)
        newest = max((r.market.ends for r in rows), default=None)
        days = LIVE_DAYS if newest is None else max(1, min(LIVE_DAYS, (clock - newest).days + 1))
        fresh = _series_scored(stem, terms, days, today=today, event=event, path=path, lead=lead)
        known = {r.market.token for r in rows}
        merged = rows + [s for s in fresh if s.market.token not in known]
        if merged:
            counts = Counter(s.market.family for s in merged)
            chosen = counts.most_common(1)[0][0]
            return Record(chosen, tuple(s for s in merged if s.market.family == chosen))
    markets = resolved_from(list((search or _closed_events)(terms[0])), terms)
    if not markets:
        return None
    counts = Counter(m.family for m in markets)
    chosen = (question_family if question_family and counts.get(question_family, 0) >= 5
              else counts.most_common(1)[0][0])
    scored = score([m for m in markets if m.family == chosen], lead=lead, path=path)
    if not scored:
        return None
    return Record(chosen, tuple(scored))


def against_coin(record: Record) -> tuple[float, float]:
    """The mean amount by which the crowd beat a coin's 0.25 per market, and its t statistic."""
    gains = [0.25 - s.brier for s in record.scored]
    if len(gains) < 2:
        return (gains[0] if gains else 0.0), 0.0
    mean = sum(gains) / len(gains)
    spread = statistics.stdev(gains)
    return mean, (mean / (spread / math.sqrt(len(gains))) if spread > 0 else 0.0)


def label(kind: str) -> str:
    """A family as a reader would write it: "Bitcoin above ___ on <date>?"."""
    return re.sub(r"<d>\s*#?", "<date>", kind)


def _pct(value: float) -> str:
    return f"{value:.0%}"


def lines(symbol: str, *, question_family: str | None = None, search: Any = None,
          path: Any = None, event: Any = None, snapshot: dict[str, Any] | None = None,
          today: datetime | None = None) -> list[str]:
    """The crowd's record in plain lines; one line saying so when there is no record to score."""
    name = symbol.removesuffix("USDT")
    try:
        record = record_for(symbol, question_family=question_family, search=search, path=path,
                            event=event, snapshot=snapshot, today=today)
    except PredictionError:
        return [f"Bottom line: Polymarket did not answer just now, so the crowd's record on "
                f"{name} cannot be scored — ask again in a minute."]
    if record is None:
        return [f"Bottom line: there are no resolved Polymarket markets on {name} above the "
                f"$10,000 volume floor with a price a day before they closed, so there is no "
                f"record to score."]
    kind = label(record.family)
    hard = record.contested()
    if hard.n >= 10:
        _, t = against_coin(hard)
        verdict = ("better than a coin flip, by more than chance would give "
                   f"(t = {t:.1f})" if t >= 2 else
                   f"worse than a coin flip (t = {t:.1f})" if t <= -2 else
                   f"not distinguishable from a coin flip on this many markets (t = {t:.1f})")
        lead = (f"Bottom line: on close calls — the {hard.n} “{kind}” markets priced "
                f"20-80% a day before they closed — the crowd's Brier score is {hard.brier:.3f} "
                f"against a coin flip's 0.250: {verdict}.")
    else:
        lead = (f"Bottom line: on {record.n} resolved “{kind}” markets, the crowd's "
                f"Brier score a day before the close is {record.brier:.3f} against 0.250 for a "
                f"coin flip — but only {hard.n} of them were close calls (20-80%), too few to say "
                f"how good its judgement is when it matters.")
    out = [lead,
           f"All {record.n} markets: Brier {record.brier:.3f}; always saying the base rate "
           f"({_pct(record.base_rate)} resolved YES, in-sample) scores "
           f"{record.base_rate_brier:.3f}. Lower is better; 0 is perfect. The all-market figure "
           f"is flattered by strikes far from the price, which anyone would call right."]
    band_text = "; ".join(f"said {_pct(low)}-{_pct(high)} (avg {_pct(mean)}): {_pct(hit)} "
                          f"happened, n={count}"
                          for low, high, count, mean, hit in record.bands())
    out.append(f"Calibration by band: {band_text}.")
    newest = max(s.market.ends for s in record.scored)
    oldest = min(s.market.ends for s in record.scored)
    out.append(f"Data: {record.n} resolved Polymarket markets above the $10,000 volume floor, "
               f"{oldest:%d %b %Y} to {newest:%d %b %Y}; each price is the last trade 24 hours "
               f"before the end, from Polymarket's CLOB; 50-50 and voided resolutions are left "
               f"out. A past record, not a promise about the next market.")
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI, live network
    snapshot = build_snapshot()
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=1), encoding="utf-8", newline="\n")
    for stem, rows in snapshot["series"].items():
        print(stem, len(rows))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

