"""XBRL financial statements — and the duration trap that makes naive ingestion wrong.

The earnings sub-theme asks how an agent interprets earnings and executes on them. Until now the
desk saw that a 10-Q *existed*; it never saw a number inside one. SEC's XBRL API closes that,
keyless and free:

* ``https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json``

**The trap, found in the first real response.** Asking NVDA for ``Revenues`` returns, from a single
Q2 10-Q, two facts with the same ``end`` date and the same ``fp: Q2``:

    start=2026-01-26  end=2026-07-26  181 days  177.8B   fp=Q2  frame=None
    start=2026-04-27  end=2026-07-26   90 days   96.2B   fp=Q2  frame=CY2026Q2

The first is six months cumulative, the second is the quarter. Taking the wrong one reports NVDA's
quarterly revenue as **85% too high**, and nothing in ``fp`` or ``form`` distinguishes them. A
reader scanning for the newest Q2 figure gets the cumulative one roughly half the time.

**The discriminator is the duration, and it is exact.** ``end - start`` is 90 days for a quarter and
~365 for a year. :data:`QUARTER_DAYS` and :data:`ANNUAL_DAYS` carry tolerances because fiscal
quarters are 13 weeks and drift by a few days; nothing else about the response is needed. The
``frame`` field correlates but is absent on exactly the rows that most need disambiguating, so it is
reported and not relied on.

**Restatements.** The same period is filed more than once — an original 10-Q and later an amended
one, or an annual report restating a quarter. Facts carry ``filed``, and the latest filing for a
period supersedes earlier ones. Superseded values are counted and reported rather than silently
dropped, because a restatement is itself news about a company.

**Point-in-time.** Every fact carries the date it was *filed*, and :func:`facts` will not return one
filed after the caller's ``as_of``. A backtest that reads today's restatement of a two-year-old
quarter has read the future. This is the same gate ``market/evidence.py`` applies to headlines.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from argus.market.evidence import _UA, EdgarSource
from argus.research.sue import MIN_QUARTERS, SueError, read_dated, yoy_window
from argus.truth.evidence import Evidence

TIMEOUT = 15

QUARTER_DAYS = (80, 100)
"""A fiscal quarter is 13 weeks; the band absorbs 52/53-week calendars and leap years."""

ANNUAL_DAYS = (350, 380)

# The tags that actually matter for a trading thesis, in the order they should be tried. US-GAAP
# lets an issuer choose among several tags for the same line, so each concept lists its synonyms —
# NVDA reports revenue under `Revenues`, many issuers use the longer contract-with-customer tag.
CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "operating_income": ("OperatingIncomeLoss",),
    "eps_diluted": ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"),
    "gross_profit": ("GrossProfit",),
}


class FundamentalsError(RuntimeError):
    """The XBRL source cannot answer the question asked of it."""


@dataclass(frozen=True)
class Fact:
    """One reported number, with everything needed to know whether to trust it."""

    concept: str
    tag: str
    value: float
    unit: str
    start: date | None
    end: date
    filed: date
    form: str
    fiscal_year: int | None
    fiscal_period: str | None
    frame: str | None

    @property
    def duration_days(self) -> int | None:
        if self.start is None:
            return None
        return (self.end - self.start).days

    @property
    def is_quarterly(self) -> bool:
        days = self.duration_days
        return days is not None and QUARTER_DAYS[0] <= days <= QUARTER_DAYS[1]

    @property
    def is_annual(self) -> bool:
        days = self.duration_days
        return days is not None and ANNUAL_DAYS[0] <= days <= ANNUAL_DAYS[1]

    @property
    def is_instant(self) -> bool:
        """A balance-sheet item has no duration — it is a snapshot, not a flow."""
        return self.start is None

    @property
    def period_kind(self) -> str:
        if self.is_instant:
            return "instant"
        if self.is_quarterly:
            return "quarter"
        if self.is_annual:
            return "year"
        return f"{self.duration_days}-day"

    @property
    def period_key(self) -> tuple[str, str]:
        """What makes two facts the *same* period, so a restatement can supersede its original."""
        return (self.period_kind, self.end.isoformat())

    def render(self) -> str:
        scaled = (
            f"{self.value / 1e9:.2f}B" if abs(self.value) >= 1e8
            else f"{self.value:,.2f}"
        )
        return (
            f"{self.concept} {scaled} {self.unit} for the {self.period_kind} ending "
            f"{self.end.isoformat()} (filed {self.filed.isoformat()} on {self.form})"
        )

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
            "period_kind": self.period_kind,
            "duration_days": self.duration_days,
            "fiscal_year": self.fiscal_year,
            "fiscal_period": self.fiscal_period,
            "frame": self.frame,
        }


def _as_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def parse_concept(payload: dict[str, Any], *, concept: str) -> list[Fact]:
    """Turn one companyconcept response into facts, across every unit it reports."""
    tag = str(payload.get("tag", ""))
    out: list[Fact] = []
    for unit, rows in (payload.get("units") or {}).items():
        for row in rows:
            end = _as_date(row.get("end"))
            filed = _as_date(row.get("filed"))
            if end is None or filed is None or "val" not in row:
                continue
            try:
                value = float(row["val"])
            except (TypeError, ValueError):
                continue
            out.append(
                Fact(
                    concept=concept,
                    tag=tag,
                    value=value,
                    unit=unit,
                    start=_as_date(row.get("start")),
                    end=end,
                    filed=filed,
                    form=str(row.get("form", "")),
                    fiscal_year=row.get("fy"),
                    fiscal_period=row.get("fp"),
                    frame=row.get("frame"),
                )
            )
    return out


def latest_per_period(facts: list[Fact]) -> tuple[list[Fact], int]:
    """Keep the most recently filed fact for each period; report how many were superseded.

    A restatement is news about a company, so the count is returned rather than the drop being
    silent. Ties on ``filed`` keep the first seen, which for SEC responses is the earlier row.
    """
    best: dict[tuple[str, str], Fact] = {}
    superseded = 0
    for fact in facts:
        key = fact.period_key
        held = best.get(key)
        if held is None:
            best[key] = fact
            continue
        superseded += 1
        if fact.filed > held.filed:
            best[key] = fact
    return sorted(best.values(), key=lambda f: f.end, reverse=True), superseded


def _sue_evidence(ticker: str, eps_facts: list[Fact]) -> tuple[list[Evidence], list[str]]:
    """The Standardized Unexpected Earnings reading, if the real EPS history covers it.

    Added 2026-09-23. `research.sue` verified the SUE formula to floating-point identity against
    QuantConnect's own reference but was never wired to a live decision: the desk saw only the raw
    quarter-on-quarter EPS change above, which is not standardized by the company's own earnings
    volatility and is not comparable across tickers or across a single ticker's own history — the
    exact gap SUE exists to close. This closes it with the same facts already fetched for the
    quarter-on-quarter figure, at no extra network cost.

    Deliberately makes no return-predictiveness claim in the rendered text — see
    `research.pead_study` for whether one is warranted, and cite it explicitly here only once a
    result is verified, not assumed.

    Corrected 2026-09-25: this used to hand the newest twelve facts to the positional
    `research.sue.read`, whose ``quarters[i + 4]`` is the same quarter a year earlier only when no
    quarter is missing. SEC XBRL carries no standalone fiscal Q4 (it lives in the 10-K's annual
    figure), so for NVDA, AAPL, MSTR and COIN every one of the eight "year-over-year" deltas was
    in fact a fifteen-to-eighteen-month change between different fiscal quarters, and the claim
    below said "year-over-year" over it. `research.sue.read_dated` pairs by period end instead
    (`eval/general_sue_comparison.py`). The reading is available once every fact it used was
    filed, which is the newest quarter's own filing date unless an older one was restated later.
    """
    if len(eps_facts) < MIN_QUARTERS:
        return [], [
            f"xbrl:{ticker}: {len(eps_facts)} quarter(s) of EPS, below the {MIN_QUARTERS} SUE needs"
        ]
    points = [(f.end, f.value) for f in eps_facts]
    try:
        window = yoy_window(points)
        sue = read_dated(ticker, points)
    except SueError as exc:
        return [], [f"xbrl:{ticker}: SUE not computable ({exc})"]
    used_ends = {d.end for d in window} | {d.prior_end for d in window}
    newest = next(f for f in eps_facts if f.end == window[0].end)
    known_on = max(f.filed for f in eps_facts if f.end in used_ends)
    direction = "above" if sue.sue > 0 else "below" if sue.sue < 0 else "in line with"
    magnitude = "many" if abs(sue.sue) >= 3 else "several" if abs(sue.sue) >= 1 else "under one"
    return [
        Evidence(
            id=f"xbrl-{ticker}-sue-{newest.end.isoformat()}",
            claim=(
                f"Standardized Unexpected Earnings for the quarter ending "
                f"{newest.end.isoformat()}: {sue.sue:+.2f} standard deviations "
                f"{direction} this company's own trailing year-over-year EPS-change volatility "
                f"({magnitude} standard deviation(s) of surprise, EPS change "
                f"{sue.eps_change:+.4f} vs a historical deviation of {sue.eps_std:.4f})"
            ),
            source="filing",
            available_at=datetime.combine(known_on, datetime.min.time(), tzinfo=UTC),
            credibility=1.0,
            attributes={
                "concept": "sue",
                "sue": sue.sue,
                "eps_change": sue.eps_change,
                "eps_std": sue.eps_std,
                "period_end": newest.end.isoformat(),
                "quarters_used": sue.quarters_used,
            },
        )
    ], [f"xbrl:{ticker}: SUE {sue.sue:+.2f} from {sue.quarters_used} quarter(s)"]


class FundamentalsSource:
    """SEC XBRL company concepts, keyless."""

    CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json"

    def __init__(self, *, user_agent: str = _UA, edgar: EdgarSource | None = None) -> None:
        self._ua = user_agent
        self._edgar = edgar or EdgarSource(user_agent=user_agent)

    def _get(self, url: str) -> dict[str, Any]:
        request = urllib.request.Request(
            url, headers={"User-Agent": self._ua, "Accept": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body: bytes = response.read()
        loaded: dict[str, Any] = json.loads(body.decode())
        return loaded

    def facts(
        self,
        ticker: str,
        *,
        concept: str,
        as_of: datetime,
        quarterly_only: bool = True,
    ) -> tuple[list[Fact], list[str]]:
        """Facts for one concept, point-in-time, restatements resolved.

        ``quarterly_only`` is the default because the cumulative rows are the trap this module
        exists for: they sit beside the quarterly ones with an identical ``fp`` and a different
        number.
        """
        # Argument validation before any I/O. Both of these are programming errors, and finding
        # out about one by watching a request to SEC fail is a slow and confusing way to learn it —
        # the first draft of this method looked up the CIK first, and an unknown concept surfaced
        # as an HTTP 403 from a network the caller never meant to touch.
        if as_of.tzinfo is None:
            raise FundamentalsError("as_of must be timezone-aware")
        tags = CONCEPTS.get(concept)
        if not tags:
            raise FundamentalsError(
                f"unknown concept {concept!r}; known: {', '.join(sorted(CONCEPTS))}"
            )

        status: list[str] = []
        cik = self._edgar.cik_for(ticker)
        if cik is None:
            return [], [f"xbrl:{ticker}: no CIK on EDGAR"]

        for tag in tags:
            try:
                payload = self._get(self.CONCEPT_URL.format(cik=cik, tag=tag))
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    continue  # this issuer reports the concept under a different tag
                status.append(f"xbrl:{ticker}:{tag}: HTTP {exc.code}")
                continue
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                status.append(f"xbrl:{ticker}:{tag}: unavailable ({type(exc).__name__})")
                continue

            facts = parse_concept(payload, concept=concept)
            cutoff = as_of.date()
            visible = [f for f in facts if f.filed <= cutoff]
            if len(visible) < len(facts):
                status.append(
                    f"xbrl:{ticker}:{tag}: {len(facts) - len(visible)} fact(s) filed after as_of "
                    f"withheld"
                )

            if quarterly_only:
                before = len(visible)
                visible = [f for f in visible if f.is_quarterly]
                cumulative = before - len(visible)
                if cumulative:
                    status.append(
                        f"xbrl:{ticker}:{tag}: {cumulative} non-quarterly row(s) excluded "
                        f"(cumulative and annual durations sit beside the quarterly ones under "
                        f"the same fiscal period)"
                    )

            resolved, superseded = latest_per_period(visible)
            if superseded:
                status.append(
                    f"xbrl:{ticker}:{tag}: {superseded} restated value(s) superseded by a later "
                    f"filing"
                )
            status.append(f"xbrl:{ticker}: {concept} from {tag}, {len(resolved)} period(s)")
            return resolved, status

        status.append(f"xbrl:{ticker}: no us-gaap tag for {concept} among {', '.join(tags)}")
        return [], status

    def evidence(
        self, ticker: str, *, as_of: datetime, periods: int = 2
    ) -> tuple[list[Evidence], list[str]]:
        """The most recent quarters of each headline concept, with growth where it is computable."""
        out: list[Evidence] = []
        status: list[str] = []
        stamp = as_of

        for concept in ("revenue", "net_income", "eps_diluted"):
            facts, st = self.facts(ticker, concept=concept, as_of=as_of)
            status.extend(st)
            if concept == "eps_diluted":
                sue_evidence, sue_status = _sue_evidence(ticker, facts)
                out.extend(sue_evidence)
                status.extend(sue_status)
            recent = facts[:periods]
            for fact in recent:
                out.append(
                    Evidence(
                        id=f"xbrl-{ticker}-{concept}-{fact.end.isoformat()}",
                        claim=fact.render(),
                        source="filing",
                        available_at=datetime.combine(
                            fact.filed, datetime.min.time(), tzinfo=UTC
                        ),
                        credibility=1.0,
                        attributes={
                            "concept": concept,
                            "tag": fact.tag,
                            "value": fact.value,
                            "unit": fact.unit,
                            "period_end": fact.end.isoformat(),
                            "form": fact.form,
                        },
                    )
                )
            if len(recent) >= 2 and recent[1].value:
                change = (recent[0].value - recent[1].value) / abs(recent[1].value)
                out.append(
                    Evidence(
                        id=f"xbrl-{ticker}-{concept}-qoq",
                        claim=(
                            f"{concept} quarter on quarter {change:+.1%} "
                            f"({recent[1].end.isoformat()} to {recent[0].end.isoformat()}), "
                            f"both quarterly durations"
                        ),
                        source="filing",
                        available_at=datetime.combine(
                            recent[0].filed, datetime.min.time(), tzinfo=UTC
                        ),
                        credibility=1.0,
                        # `growing` is the one attribute here a thesis can contradict: a desk that
                        # writes "revenue growth supports the long case" against a filed sequential
                        # decline has made a checkable error. The per-period facts above carry no
                        # such field, so `argus.agents.claims` leaves them alone rather than
                        # guessing which number a sentence meant.
                        attributes={
                            "concept": concept,
                            "change_pct": change * 100.0,
                            "growing": change > 0,
                            "from_period": recent[1].end.isoformat(),
                            "to_period": recent[0].end.isoformat(),
                        },
                    )
                )

        out = [e for e in out if e.available_at <= stamp]
        out.sort(key=lambda e: e.available_at, reverse=True)
        return out, status


__all__ = [
    "ANNUAL_DAYS",
    "CONCEPTS",
    "QUARTER_DAYS",
    "Fact",
    "FundamentalsError",
    "FundamentalsSource",
    "latest_per_period",
    "parse_concept",
]
