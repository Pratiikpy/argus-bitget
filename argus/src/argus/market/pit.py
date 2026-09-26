"""Point-in-time SEC facts, gated to the second EDGAR accepted the filing — not to its date.

``market/fundamentals.py`` already refuses a fact whose ``filed`` date is after the caller's
``as_of``. That rule is correct to the day and wrong inside it. NVDA's Q2 FY2027 10-Q carries
``filed: 2026-08-26``; EDGAR accepted it at **2026-08-26T20:36:00Z** (16:36 ET, after the close,
fifteen minutes after the 8-K that released the numbers). A date gate shows that quarter to a
caller asking "as of 2026-08-26 14:00 UTC" — six and a half hours before it existed, across the
one boundary in the quarter where the difference is the whole trade. Every row of the XBRL
``companyconcept`` / ``companyfacts`` APIs names its accession (``accn``), and every accession's
exact ``acceptanceDateTime`` is in the issuer's EDGAR submissions index, so the second is
available; this module uses it.

**Availability, in order of preference** (:func:`availability`):

1. ``accepted`` — the accession's own ``acceptanceDateTime`` from
   ``data.sec.gov/submissions/CIK##########.json`` (the ``recent`` block and every historical page
   listed under ``files``). Verified to be real UTC on 2026-09-25: NVDA's earnings 8-K reads
   ``2026-08-26T20:21:19.000Z`` in the JSON and "Accepted 2026-08-26 16:21:19" (Eastern) on the
   EDGAR index page. In the 10-ticker benchmark snapshot every one of the XBRL rows' accessions
   resolved.
2. ``edgar-close`` — if an accession cannot be resolved, 22:00 America/New_York on the filing date.
   The filing-date convention cuts off at 17:30 ET, but in the same snapshot **20 of 3,441**
   10-Q/10-K/8-K accessions were accepted after 17:30 ET and still carry that day's filing date
   (MSFT's FY2020 10-K: accepted 20:44:46 ET, filed 2020-07-30). A 17:30 bound would have leaked
   those; 22:00 ET is the end of EDGAR's accepting day, the latest any same-day acceptance was
   observed to reach (NOT re-verified against the EDGAR Filer Manual text in this session).
3. ``observed`` — a row present in a response fetched at ``t`` was public by ``t``. The bound in (2)
   is capped at the fetch instant, so a live caller asking "as of now" is never denied a filing it
   is looking at merely because the accession index was cached before the filing landed.

**Restatements resolve to the latest knowledge, not the first.** Both rivals read for this module
keep the *earliest* filing of a period in their point-in-time modes — Vibe-Trading's loader
(``agent/backtest/loaders/fundamentals_loader.py:111``, ``keep = "first" if pit else "last"``) and
OpenBB ODP's SEC provider (``openbb_sec/utils/statement_schema/_extraction.py:677-682``,
``pit_mode and filed < ref_map[end_date]``). Keep-first cannot leak, but after a restatement is
filed it keeps serving the superseded number forever; keep-last (their non-PIT modes) is current
but serves the restated number *before* it existed. :func:`resolve` keeps, per period, the most
recently **available as of the cutoff** — current and leak-free at once. ``eval/pit_rivals.py``
measures all four behaviours on the same real facts.

**Alias tags merge per period, by priority.** US-GAAP lets an issuer move a line between tags
(AAPL reported revenue as ``SalesRevenueNet`` to FY2018 and as
``RevenueFromContractWithCustomerExcludingAssessedTax`` after). ``fundamentals.py`` takes the first
tag in :data:`~argus.market.fundamentals.CONCEPTS` that exists at all; here every alias is read and,
for each period, the highest-priority tag with a *visible* fact wins, so an as-of in 2017 still
sees the tag the issuer used in 2017.

**Fiscal Q4 is derived, for additive lines only.** Most issuers no longer tag a three-month Q4 —
the 10-K reports the year. Without a derived Q4 a reader's "latest quarter" is stale for the ~3
months between the 10-K and the next 10-Q. For revenue, net income, operating income and gross
profit, Q4 = FY - (Q1+Q2+Q3), available when the *last* of its four components was, which keeps
it point-in-time (the construction is Vibe-Trading's ``_quarterly_flow_frames``,
``fundamentals_loader.py:137-171``, MIT — adapted, with availability taken to the second rather
than the day). EPS is never derived: per-share figures do not add across quarters when the share
count moves, and a stock split inside the fiscal year turns the subtraction into nonsense — OpenBB
ODP's standardized statement reports NVDA's Q4 FY2025 diluted EPS as **-4.49** (true figure
~0.89) from exactly this subtraction across the June 2024 10-for-1 split (measured in
``eval/pit_rivals.py``). A missing number is recoverable; a confident wrong one is not.
"""

from __future__ import annotations

import json
import threading
import time as _time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from argus.market.evidence import _UA
from argus.market.fundamentals import ANNUAL_DAYS, CONCEPTS, QUARTER_DAYS

EASTERN = ZoneInfo("America/New_York")
EDGAR_CLOSE = time(22, 0)
"""End of EDGAR's accepting day. The fallback bound for an accession whose exact time is unknown."""

ADDITIVE_CONCEPTS: frozenset[str] = frozenset(
    {"revenue", "net_income", "operating_income", "gross_profit"}
)
"""Lines whose four quarters sum to the year, so a fiscal Q4 can be derived by subtraction."""

PER_SHARE_CONCEPTS: frozenset[str] = frozenset({"eps_diluted"})

INDEX_TTL = timedelta(hours=1)
TIMEOUT = 20

Basis = Literal["accepted", "edgar-close", "observed", "derived"]
Gate = Literal["accepted", "filed-date", "none"]
Supersede = Literal["latest", "first"]

FetchJson = Callable[[str], Any]


class PitError(RuntimeError):
    """The point-in-time store cannot answer the question it was asked."""


def edgar_close_utc(filed: date) -> datetime:
    """22:00 America/New_York on ``filed``, in UTC — the latest a same-day acceptance can be."""
    return datetime.combine(filed, EDGAR_CLOSE, tzinfo=EASTERN).astimezone(UTC)


def parse_acceptance(raw: str) -> datetime | None:
    """EDGAR's ``acceptanceDateTime`` (``2026-08-26T20:36:00.000Z``) as an aware UTC datetime."""
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@dataclass(frozen=True)
class AcceptanceIndex:
    """Accession number → the instant EDGAR accepted it, for one filer."""

    cik: int
    accepted: Mapping[str, datetime]
    fetched_at: datetime
    pages: int = 1

    def lookup(self, accn: str) -> datetime | None:
        return self.accepted.get(accn)


def acceptance_from_submissions(
    cik: int, recent: Mapping[str, Any], pages: Sequence[Mapping[str, Any]], *, fetched_at: datetime
) -> AcceptanceIndex:
    """Build the index from a submissions document's ``recent`` block and its historical pages."""
    accepted: dict[str, datetime] = {}
    for block in (recent, *pages):
        numbers = block.get("accessionNumber") or []
        stamps = block.get("acceptanceDateTime") or []
        for accn, stamp in zip(numbers, stamps, strict=False):
            when = parse_acceptance(str(stamp))
            if when is not None:
                accepted[str(accn)] = when
    return AcceptanceIndex(cik=cik, accepted=accepted, fetched_at=fetched_at, pages=1 + len(pages))


def availability(
    accn: str | None,
    filed: date,
    index: AcceptanceIndex | None,
    *,
    observed_at: datetime | None,
) -> tuple[datetime, Basis]:
    """When this row became public, and how that instant is known. See the module docstring."""
    if accn and index is not None:
        exact = index.lookup(accn)
        if exact is not None:
            return exact, "accepted"
    bound = edgar_close_utc(filed)
    if observed_at is not None and observed_at < bound:
        return observed_at, "observed"
    return bound, "edgar-close"


@dataclass(frozen=True)
class PitFact:
    """One reported (or derived) number, stamped with the instant it became knowable."""

    concept: str
    tag: str
    tag_rank: int
    value: float
    unit: str
    start: date | None
    end: date
    filed: date
    form: str
    accn: str
    available_at: datetime
    basis: Basis
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    derived_from: tuple[str, ...] = ()

    @property
    def duration_days(self) -> int | None:
        return None if self.start is None else (self.end - self.start).days

    @property
    def is_quarterly(self) -> bool:
        days = self.duration_days
        return days is not None and QUARTER_DAYS[0] <= days <= QUARTER_DAYS[1]

    @property
    def is_annual(self) -> bool:
        days = self.duration_days
        return days is not None and ANNUAL_DAYS[0] <= days <= ANNUAL_DAYS[1]

    @property
    def is_derived(self) -> bool:
        return bool(self.derived_from)

    def as_dict(self) -> dict[str, Any]:
        return {
            "concept": self.concept,
            "tag": self.tag,
            "value": self.value,
            "unit": self.unit,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat(),
            "filed": self.filed.isoformat(),
            "form": self.form,
            "accn": self.accn,
            "available_at": self.available_at.isoformat(),
            "basis": self.basis,
            "derived_from": list(self.derived_from),
        }


def _as_date(raw: Any) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw))
    except ValueError:
        return None


def _wanted_unit(concept: str, units: Mapping[str, Any]) -> str | None:
    preferred = "USD/shares" if concept in PER_SHARE_CONCEPTS else "USD"
    if preferred in units:
        return preferred
    ranked = sorted(units, key=lambda u: -len(units[u] or []))
    return ranked[0] if ranked else None


def facts_from_units(
    concept: str,
    tag: str,
    tag_rank: int,
    units: Mapping[str, Any],
    index: AcceptanceIndex | None,
    *,
    observed_at: datetime | None,
) -> list[PitFact]:
    """Rows of one tag (the ``units`` block of a companyconcept or companyfacts entry) as facts."""
    unit = _wanted_unit(concept, units)
    if unit is None:
        return []
    out: list[PitFact] = []
    for row in units.get(unit) or []:
        end = _as_date(row.get("end"))
        filed = _as_date(row.get("filed"))
        if end is None or filed is None or "val" not in row:
            continue
        try:
            value = float(row["val"])
        except (TypeError, ValueError):
            continue
        accn = str(row.get("accn") or "")
        when, basis = availability(accn, filed, index, observed_at=observed_at)
        out.append(
            PitFact(
                concept=concept, tag=tag, tag_rank=tag_rank, value=value, unit=unit,
                start=_as_date(row.get("start")), end=end, filed=filed,
                form=str(row.get("form", "")), accn=accn, available_at=when, basis=basis,
                fiscal_year=row.get("fy"), fiscal_period=row.get("fp"),
            )
        )
    return out


@dataclass(frozen=True)
class PitView:
    """What was knowable about one concept at ``as_of``, newest period first."""

    concept: str
    as_of: datetime
    facts: tuple[PitFact, ...]
    withheld: int
    superseded: int
    derived: int

    def latest(self) -> PitFact | None:
        return self.facts[0] if self.facts else None

    def by_end(self) -> dict[date, PitFact]:
        return {f.end: f for f in self.facts}

    def status(self) -> list[str]:
        notes = [f"pit:{self.concept}: {len(self.facts)} period(s) knowable as of "
                 f"{self.as_of.isoformat()}"]
        if self.withheld:
            notes.append(f"pit:{self.concept}: {self.withheld} row(s) accepted after as_of "
                         f"withheld")
        if self.superseded:
            notes.append(f"pit:{self.concept}: {self.superseded} earlier value(s) superseded by "
                         f"a later filing that was already public")
        if self.derived:
            notes.append(f"pit:{self.concept}: {self.derived} fiscal Q4 value(s) derived as FY "
                         f"minus Q1-Q3, available when the last component was")
        return notes


def _visible(fact: PitFact, as_of: datetime, gate: Gate) -> bool:
    if gate == "accepted":
        return fact.available_at <= as_of
    if gate == "filed-date":
        return fact.filed <= as_of.date()
    return True


def _pick(candidates: Sequence[PitFact], supersede: Supersede) -> tuple[PitFact, int]:
    """Best tag first (lowest rank with a visible row), then latest-or-first within that tag."""
    best_rank = min(f.tag_rank for f in candidates)
    same_tag = [f for f in candidates if f.tag_rank == best_rank]
    ordered = sorted(same_tag, key=lambda f: (f.available_at, f.accn))
    chosen = ordered[-1] if supersede == "latest" else ordered[0]
    superseded = sum(1 for f in same_tag if f.value != chosen.value)
    return chosen, superseded


def resolve(
    facts: Sequence[PitFact],
    *,
    concept: str,
    as_of: datetime,
    gate: Gate = "accepted",
    supersede: Supersede = "latest",
    derive_q4: bool | None = None,
) -> PitView:
    """The point-in-time view of one concept. ``gate``/``supersede``/``derive_q4`` exist so the
    benchmark can ablate each mechanism; the defaults are the only production configuration."""
    if as_of.tzinfo is None:
        raise PitError("as_of must be timezone-aware; a naive clock cannot bound a search")
    derive = (concept in ADDITIVE_CONCEPTS) if derive_q4 is None else derive_q4

    visible = [f for f in facts if _visible(f, as_of, gate)]
    withheld = len(facts) - len(visible)

    quarters: dict[date, list[PitFact]] = {}
    years: dict[date, list[PitFact]] = {}
    for fact in visible:
        if fact.is_quarterly:
            quarters.setdefault(fact.end, []).append(fact)
        elif fact.is_annual:
            years.setdefault(fact.end, []).append(fact)

    superseded = 0
    chosen_quarters: dict[date, PitFact] = {}
    for end, group in quarters.items():
        chosen, dropped = _pick(group, supersede)
        chosen_quarters[end] = chosen
        superseded += dropped

    derived: dict[date, PitFact] = {}
    if derive:
        for end, group in years.items():
            if end in chosen_quarters:
                continue
            annual, _ = _pick(group, supersede)
            if annual.start is None:
                continue
            inside = sorted(
                (q for q in chosen_quarters.values()
                 if q.start is not None and q.start >= annual.start and q.end < annual.end),
                key=lambda q: q.end,
            )
            if len(inside) != 3:
                continue
            components = (annual, *inside)
            derived[end] = PitFact(
                concept=concept, tag=annual.tag, tag_rank=annual.tag_rank,
                value=annual.value - sum(q.value for q in inside), unit=annual.unit,
                start=inside[-1].end + timedelta(days=1), end=end,
                filed=max(c.filed for c in components), form=f"{annual.form} (derived Q4)",
                accn=annual.accn, available_at=max(c.available_at for c in components),
                basis="derived", fiscal_year=annual.fiscal_year, fiscal_period="Q4",
                derived_from=tuple(c.accn for c in components),
            )

    resolved = sorted([*chosen_quarters.values(), *derived.values()], key=lambda f: f.end,
                      reverse=True)
    return PitView(concept=concept, as_of=as_of, facts=tuple(resolved), withheld=withheld,
                   superseded=superseded, derived=len(derived))


class PitFundamentals:
    """SEC XBRL facts for the five desk concepts, gated to the accepted second.

    ``fetch_json`` is injectable so the benchmark can serve a frozen snapshot through the exact
    code path a live caller uses; the default reads SEC with the desk's identifying User-Agent.
    """

    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
    PAGE_URL = "https://data.sec.gov/submissions/{name}"

    def __init__(
        self,
        *,
        fetch_json: FetchJson | None = None,
        user_agent: str = _UA,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._ua = user_agent
        self._fetch = fetch_json or self._http_json
        self._clock = clock or (lambda: datetime.now(UTC))
        self._ciks: dict[str, int] | None = None
        self._indexes: dict[int, AcceptanceIndex] = {}
        self._lock = threading.Lock()

    def _http_json(self, url: str) -> Any:
        request = urllib.request.Request(
            url, headers={"User-Agent": self._ua, "Accept": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.loads(response.read().decode())

    def cik_for(self, ticker: str) -> int | None:
        if self._ciks is None:
            raw = self._fetch(self.TICKERS_URL)
            self._ciks = {str(v["ticker"]).upper(): int(v["cik_str"]) for v in raw.values()}
        return self._ciks.get(ticker.upper())

    def acceptance_index(self, cik: int, *, refresh: bool = False) -> AcceptanceIndex:
        """The filer's accession → acceptance map, cached for :data:`INDEX_TTL`."""
        with self._lock:
            held = self._indexes.get(cik)
        now = self._clock()
        if held is not None and not refresh and now - held.fetched_at < INDEX_TTL:
            return held
        sub = self._fetch(self.SUBMISSIONS_URL.format(cik=cik))
        filings = sub.get("filings", {}) if isinstance(sub, dict) else {}
        pages = [self._fetch(self.PAGE_URL.format(name=f["name"]))
                 for f in filings.get("files", []) if f.get("name")]
        index = acceptance_from_submissions(cik, filings.get("recent", {}), pages, fetched_at=now)
        with self._lock:
            self._indexes[cik] = index
        return index

    def rows(self, ticker: str, *, concept: str) -> tuple[list[PitFact], list[str]]:
        """Every row of every alias tag for ``concept``, each stamped with its availability."""
        tags = CONCEPTS.get(concept)
        if not tags:
            raise PitError(f"unknown concept {concept!r}; known: {', '.join(sorted(CONCEPTS))}")
        cik = self.cik_for(ticker)
        if cik is None:
            return [], [f"pit:{ticker}: no CIK on EDGAR"]
        status: list[str] = []
        try:
            index: AcceptanceIndex | None = self.acceptance_index(cik)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
            index = None
            status.append(f"pit:{ticker}: acceptance index unavailable ({type(exc).__name__}); "
                          f"falling back to the 22:00 ET filing-day bound")
        observed_at = self._clock()
        out: list[PitFact] = []
        for rank, tag in enumerate(tags):
            try:
                payload = self._fetch(self.CONCEPT_URL.format(cik=cik, tag=tag))
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    status.append(f"pit:{ticker}:{tag}: HTTP {exc.code}")
                continue
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                status.append(f"pit:{ticker}:{tag}: unavailable ({type(exc).__name__})")
                continue
            if not isinstance(payload, dict) or not payload.get("units"):
                continue
            facts = facts_from_units(concept, tag, rank, payload["units"], index,
                                     observed_at=observed_at)
            unresolved = [f for f in facts if f.basis != "accepted"]
            if unresolved and index is not None:
                # A filing newer than the cached index: refresh once, then restamp.
                index = self.acceptance_index(cik, refresh=True)
                facts = facts_from_units(concept, tag, rank, payload["units"], index,
                                         observed_at=observed_at)
            out.extend(facts)
        return out, status

    def facts(
        self,
        ticker: str,
        *,
        concept: str,
        as_of: datetime,
        derive_q4: bool | None = None,
    ) -> tuple[list[PitFact], list[str]]:
        """Quarterly facts knowable at ``as_of``, newest period first, with a status trail."""
        if as_of.tzinfo is None:
            raise PitError("as_of must be timezone-aware")
        started = _time.perf_counter()
        rows, status = self.rows(ticker, concept=concept)
        view = resolve(rows, concept=concept, as_of=as_of, derive_q4=derive_q4)
        status.extend(view.status())
        status.append(f"pit:{ticker}: resolved in {(_time.perf_counter() - started):.2f}s")
        return list(view.facts), status


__all__ = [
    "ADDITIVE_CONCEPTS",
    "EDGAR_CLOSE",
    "PER_SHARE_CONCEPTS",
    "AcceptanceIndex",
    "PitError",
    "PitFact",
    "PitFundamentals",
    "PitView",
    "acceptance_from_submissions",
    "availability",
    "edgar_close_utc",
    "facts_from_units",
    "parse_acceptance",
    "resolve",
]
