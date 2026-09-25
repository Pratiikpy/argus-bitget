"""Annual statement line items from SEC XBRL, read **as reported in one 10-K**.

`market/fundamentals.py` answers "what did this company file for revenue last quarter" — five
quarterly concepts. This module answers the question an analyst actually asks of an annual report:
*what does the FY2018 10-K say capital expenditure, net PP&E, current liabilities or dividends paid
were, and — for a ratio — what do its own comparative columns say about the year before?*

Source: SEC's keyless ``companyfacts`` endpoint, one request per company:

* ``https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json``

**Three traps, each found in a real response, each the reason a naive reader returns a fluent wrong
number.**

1. **``fy`` is the fiscal year of the *filing*, not of the period.** 3M's FY2018 10-K (accession
   0001558370-19-000470) tags its FY2016, FY2017 and FY2018 capital expenditure all as ``fy=2018``.
   Filtering ``fy == 2018`` and taking any row returns 1,420 or 1,373 as often as the correct
   1,577. The year is resolved from the period's *end date* inside the anchor filing instead.
2. **Annual reports carry quarterly rows.** AES's FY2022 10-K reports four quarterly net-income
   figures beside the annual one, under the same tag and the same ``fy``/``fp=FY``. Only rows whose
   duration is a fiscal year (350-380 days) are annual.
3. **Later filings restate.** A value read from the *latest* filing for a period is the restated
   one; an analyst reading the FY2018 10-K sees the as-originally-reported one. Both are legitimate
   and they differ, so this module is explicit about which it returns: the comparative column of
   the **anchor** 10-K (the filing for the latest year the question is about), falling back to that
   year's own 10-K only when the anchor does not carry the period at all (a balance sheet two years
   back). ``edgartools``' recommended ``get_financials()`` path makes the opposite choice — "newest
   filing wins", ``edgar/entity/enhanced_statement.py:1583-1593`` — which is right for a current
   view and wrong for a point-in-time one.

**Tag synonyms.** US-GAAP lets issuers pick among several elements for one line. The synonym lists
below start from ``edgartools``' ``SynonymGroups`` (MIT, ``research/repos-themed/
dgunning~edgartools/edgar/standardization/synonym_groups.py:222-760``, commit 9d6bc98) and
**deliberately drop every synonym that names a different quantity**, because each of those turns a
missing line into a confident wrong answer instead of an honest gap:

* ``PropertyPlantAndEquipmentGross`` / ``AccountsReceivableGross`` / ``InventoryGross`` — gross is
  not net; a question about net PP&E answered with gross PP&E is wrong by the whole accumulated
  depreciation.
* ``Depreciation`` alone and ``AmortizationOfIntangibleAssets`` alone as "D&A" — each is half of
  it. They are accepted only as a *sum*, and only when no combined element exists.
* ``IncomeLossFromContinuingOperations`` as net income (excludes discontinued operations),
  ``IncomeTaxesPaidNet`` as tax expense (cash paid is not expense), ``InterestIncomeExpenseNet`` as
  interest expense (a net figure), ``WeightedAverageNumberOfSharesOutstandingBasic`` as shares
  outstanding (an average is not a count).
* ``ProfitLoss`` as net income attributable to shareholders — it *includes* non-controlling
  interests. Accepted only when the anchor filing reports no non-controlling-interest income for
  that period, which is the one case where the two are the same number.

The ablation in ``eval/financebench_same_input.py`` runs the permissive list against this one on
the same questions, so the cost of each dropped synonym is a measurement, not an opinion.

**Point-in-time.** ``as_of`` hides every filing accepted after it: an anchor 10-K filed later than
``as_of`` does not exist yet, and neither does anything read from it.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from argus.market.evidence import _UA

TIMEOUT = 60

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

ANNUAL_DAYS = (350, 380)
"""A fiscal year, including 52/53-week calendars. Same band as ``fundamentals.ANNUAL_DAYS``."""

DATE_SLACK_DAYS = 7
"""How far an instant may sit from the fiscal year end it is matched to (52/53-week drift)."""

ANNUAL_FORMS = frozenset({"10-K", "10-K405", "10-KT", "10-K/A", "10-K405/A", "10-KT/A"})

Kind = Literal["duration", "instant"]


@dataclass(frozen=True)
class Component:
    """One statement line, the elements that mean exactly it, and how to combine them.

    ``tags`` are tried in order and the first one present for the period wins. ``sum_of`` is a
    fallback used only when no tag in ``tags`` is present: every group in it must resolve, and the
    values are added (D&A reported as depreciation plus amortization, say). ``unit`` is the XBRL
    unit the value must carry.
    """

    name: str
    kind: Kind
    tags: tuple[str, ...]
    description: str
    unit: str = "USD"
    sum_of: tuple[tuple[str, ...], ...] = ()


COMPONENTS: dict[str, Component] = {
    c.name: c
    for c in (
        # --- income statement ---------------------------------------------------------------
        Component(
            "revenue", "duration",
            (
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet",
                "SalesRevenueGoodsNet",
            ),
            "total revenue / net sales",
        ),
        Component(
            "cogs", "duration",
            (
                "CostOfRevenue",
                "CostOfGoodsAndServicesSold",
                "CostOfGoodsSold",
                "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
            ),
            "cost of revenue / cost of goods sold",
        ),
        Component("gross_profit", "duration", ("GrossProfit",), "gross profit"),
        Component("operating_income", "duration", ("OperatingIncomeLoss",), "operating income"),
        Component(
            "net_income", "duration", ("NetIncomeLoss",),
            "net income attributable to the parent's shareholders",
        ),
        Component(
            "pretax_income", "duration",
            (
                "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
            ),
            "income before income taxes",
        ),
        Component("income_tax", "duration", ("IncomeTaxExpenseBenefit",), "income tax expense"),
        Component(
            "interest_expense", "duration",
            ("InterestExpense", "InterestExpenseNonoperating", "InterestAndDebtExpense"),
            "interest expense",
        ),
        Component(
            "sga", "duration", ("SellingGeneralAndAdministrativeExpense",),
            "selling, general and administrative expense",
        ),
        Component(
            "rnd", "duration", ("ResearchAndDevelopmentExpense",), "research and development",
        ),
        Component(
            "eps_diluted", "duration", ("EarningsPerShareDiluted",), "diluted EPS",
            unit="USD/shares",
        ),
        Component(
            "eps_basic", "duration", ("EarningsPerShareBasic",), "basic EPS", unit="USD/shares",
        ),
        # --- cash flow statement ------------------------------------------------------------
        Component(
            "cfo", "duration",
            (
                "NetCashProvidedByUsedInOperatingActivities",
                "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
            ),
            "net cash provided by operating activities",
        ),
        Component(
            "capex", "duration",
            ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
            "capital expenditure (purchases of PP&E)",
        ),
        Component(
            "dna", "duration",
            (
                "DepreciationDepletionAndAmortization",
                "DepreciationAmortizationAndAccretionNet",
                "DepreciationAndAmortization",
            ),
            "depreciation and amortization (cash flow statement)",
            sum_of=(("Depreciation",), ("AmortizationOfIntangibleAssets",)),
        ),
        Component(
            "dividends_paid", "duration",
            (
                "PaymentsOfDividends",
                "PaymentsOfDividendsCommonStock",
                "PaymentsOfOrdinaryDividends",
            ),
            "cash dividends paid",
        ),
        Component(
            "buybacks", "duration", ("PaymentsForRepurchaseOfCommonStock",),
            "repurchases of common stock",
        ),
        # --- balance sheet ------------------------------------------------------------------
        Component("total_assets", "instant", ("Assets",), "total assets"),
        Component("total_liabilities", "instant", ("Liabilities",), "total liabilities"),
        Component("current_assets", "instant", ("AssetsCurrent",), "total current assets"),
        Component(
            "current_liabilities", "instant", ("LiabilitiesCurrent",), "total current liabilities",
        ),
        Component("inventory", "instant", ("InventoryNet",), "inventories"),
        Component(
            "accounts_receivable", "instant",
            ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent"),
            "accounts receivable, net",
        ),
        Component(
            "accounts_payable", "instant",
            ("AccountsPayableCurrent", "AccountsPayableTradeCurrent"),
            "accounts payable",
        ),
        Component(
            "ppe_net", "instant",
            (
                "PropertyPlantAndEquipmentNet",
                "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
            ),
            "property, plant and equipment, net",
        ),
        Component(
            "cash", "instant",
            ("CashAndCashEquivalentsAtCarryingValue", "CashAndDueFromBanks"),
            "cash and cash equivalents",
        ),
        Component(
            "short_term_investments", "instant",
            ("ShortTermInvestments", "MarketableSecuritiesCurrent",
             "AvailableForSaleSecuritiesDebtSecuritiesCurrent"),
            "short-term investments / marketable securities",
        ),
        Component(
            "total_equity", "instant",
            ("StockholdersEquity",),
            "total shareholders' equity attributable to the parent",
        ),
        Component(
            "long_term_debt", "instant",
            ("LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"),
            "long-term debt, non-current",
        ),
    )
}

NCI_INCOME_TAG = "NetIncomeLossAttributableToNoncontrollingInterest"
PROFIT_LOSS_TAG = "ProfitLoss"


def all_tags(components: dict[str, Component] = COMPONENTS) -> frozenset[str]:
    """Every element any component can read — the projection a snapshot needs to keep."""
    out: set[str] = {NCI_INCOME_TAG, PROFIT_LOSS_TAG}
    for comp in components.values():
        out.update(comp.tags)
        for group in comp.sum_of:
            out.update(group)
    return frozenset(out)


class Unresolved(LookupError):
    """The filing does not say this, and the reason why. Never a guess in disguise."""


def _d(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


@dataclass(frozen=True)
class Row:
    """One ``companyfacts`` row, typed."""

    tag: str
    unit: str
    value: float
    start: date | None
    end: date
    accn: str
    form: str
    filed: date
    fy: int | None
    fp: str | None

    @property
    def duration_days(self) -> int | None:
        return None if self.start is None else (self.end - self.start).days

    @property
    def is_annual(self) -> bool:
        days = self.duration_days
        return days is not None and ANNUAL_DAYS[0] <= days <= ANNUAL_DAYS[1]

    @property
    def is_instant(self) -> bool:
        return self.start is None


@dataclass(frozen=True)
class Line:
    """A resolved statement value with everything a reader needs to check it."""

    component: str
    value: float
    unit: str
    fiscal_year: int
    period_end: date
    tags: tuple[str, ...]
    accn: str
    form: str
    filed: date
    from_anchor: bool
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "value": self.value,
            "unit": self.unit,
            "fiscal_year": self.fiscal_year,
            "period_end": self.period_end.isoformat(),
            "tags": list(self.tags),
            "accn": self.accn,
            "form": self.form,
            "filed": self.filed.isoformat(),
            "from_anchor": self.from_anchor,
            "note": self.note,
        }

    def cite(self) -> str:
        where = "" if self.from_anchor else " (not in the anchor 10-K; read from that year's own)"
        return (
            f"{self.component}={self.value:,.0f} {self.unit} FY{self.fiscal_year} "
            f"(period ending {self.period_end.isoformat()}; us-gaap:{'+'.join(self.tags)}; "
            f"{self.form} {self.accn} filed {self.filed.isoformat()}){where}"
        )


@dataclass(frozen=True)
class Anchor:
    """The 10-K a fiscal year is read from, and the fiscal year ends its columns cover."""

    fiscal_year: int
    accn: str
    form: str
    filed: date
    year_ends: tuple[date, ...]
    """Annual period ends reported in this filing, newest first. ``year_ends[0]`` is the fiscal
    year the filing is for; ``year_ends[k]`` is ``k`` years before it."""

    def end_for(self, fiscal_year: int) -> date | None:
        k = self.fiscal_year - fiscal_year
        if k < 0 or k >= len(self.year_ends):
            return None
        return self.year_ends[k]


@dataclass
class CompanyFacts:
    """One company's XBRL facts, restricted to the elements this module reads."""

    cik: int
    entity_name: str
    rows_by_tag: dict[str, list[Row]]
    digest: str = ""
    _anchors: dict[tuple[int, str], Anchor | None] = field(default_factory=dict, repr=False)

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, keep: frozenset[str] | None = None
                     ) -> CompanyFacts:
        keep = keep if keep is not None else all_tags()
        gaap = (payload.get("facts") or {}).get("us-gaap") or {}
        rows_by_tag: dict[str, list[Row]] = {}
        for tag, body in gaap.items():
            if tag not in keep:
                continue
            rows: list[Row] = []
            for unit, raw_rows in (body.get("units") or {}).items():
                for raw in raw_rows:
                    end, filed = _d(raw.get("end")), _d(raw.get("filed"))
                    if end is None or filed is None or "val" not in raw:
                        continue
                    try:
                        value = float(raw["val"])
                    except (TypeError, ValueError):
                        continue
                    rows.append(Row(
                        tag=tag, unit=unit, value=value, start=_d(raw.get("start")), end=end,
                        accn=str(raw.get("accn", "")), form=str(raw.get("form", "")),
                        filed=filed, fy=raw.get("fy"), fp=raw.get("fp"),
                    ))
            rows_by_tag[tag] = rows
        projection = compact(payload, keep=keep)
        digest = hashlib.sha256(
            json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(
            cik=int(payload.get("cik") or 0),
            entity_name=str(payload.get("entityName") or ""),
            rows_by_tag=rows_by_tag,
            digest=digest,
        )

    # --- anchor filing --------------------------------------------------------------------

    def _annual_rows(self) -> list[Row]:
        return [r for rows in self.rows_by_tag.values() for r in rows
                if r.form in ANNUAL_FORMS and r.is_annual]

    def anchor(self, fiscal_year: int, *, as_of: date | None = None) -> Anchor:
        """The 10-K for ``fiscal_year``: the original filing, not a later amendment.

        Selected by the filing's own ``fy``/``fp=FY`` tag (DocumentFiscalYearFocus), then checked
        against its dates: the newest annual period it reports must end within a year either side
        of the named fiscal year, or the tag is not trusted and the method reports why.
        """
        key = (fiscal_year, as_of.isoformat() if as_of else "")
        if key in self._anchors:
            cached = self._anchors[key]
            if cached is None:
                raise Unresolved(f"no 10-K for FY{fiscal_year} on file"
                                 + (f" as of {as_of.isoformat()}" if as_of else ""))
            return cached
        by_accn: dict[str, list[Row]] = {}
        for row in self._annual_rows():
            if row.fy == fiscal_year and row.fp == "FY":
                if as_of is not None and row.filed > as_of:
                    continue
                by_accn.setdefault(row.accn, []).append(row)
        if not by_accn:
            self._anchors[key] = None
            raise Unresolved(
                f"no 10-K tagged fiscal year {fiscal_year} on file"
                + (f" as of {as_of.isoformat()}" if as_of else "")
            )

        def rank(accn: str) -> tuple[int, date]:
            rows = by_accn[accn]
            amended = any(r.form.endswith("/A") for r in rows)
            return (1 if amended else 0, min(r.filed for r in rows))

        accn = min(by_accn, key=rank)
        rows = by_accn[accn]
        ends = sorted({r.end for r in rows}, reverse=True)
        year_ends = _dedupe_year_ends(ends)
        newest = year_ends[0]
        if not (fiscal_year - 1 <= newest.year <= fiscal_year + 1):
            self._anchors[key] = None
            raise Unresolved(
                f"the filing tagged FY{fiscal_year} ({accn}) reports a newest year ending "
                f"{newest.isoformat()}, which does not belong to FY{fiscal_year}"
            )
        anchor = Anchor(
            fiscal_year=fiscal_year, accn=accn, form=rows[0].form,
            filed=min(r.filed for r in rows), year_ends=tuple(year_ends),
        )
        self._anchors[key] = anchor
        return anchor

    # --- values ---------------------------------------------------------------------------

    def _rows_at(self, tag: str, unit: str, accn: str, end: date, kind: Kind) -> list[Row]:
        out = []
        for r in self.rows_by_tag.get(tag, ()):
            if r.accn != accn or r.unit != unit:
                continue
            if abs((r.end - end).days) > DATE_SLACK_DAYS:
                continue
            if kind == "instant" and not r.is_instant:
                continue
            if kind == "duration" and not r.is_annual:
                continue
            out.append(r)
        return out

    def _one(self, tag: str, unit: str, accn: str, end: date, kind: Kind) -> Row | None:
        rows = self._rows_at(tag, unit, accn, end, kind)
        if not rows:
            return None
        values = {r.value for r in rows}
        if len(values) > 1:
            raise Unresolved(
                f"us-gaap:{tag} carries {len(values)} different values for the period ending "
                f"{end.isoformat()} in {accn} ({', '.join(f'{v:,.0f}' for v in sorted(values))})"
            )
        return rows[0]

    def _from_filing(self, comp: Component, anchor: Anchor, end: date, fiscal_year: int,
                     *, from_anchor: bool) -> Line | None:
        for tag in comp.tags:
            row = self._one(tag, comp.unit, anchor.accn, end, comp.kind)
            if row is not None:
                return Line(comp.name, row.value, row.unit, fiscal_year, row.end, (tag,),
                            row.accn, row.form, row.filed, from_anchor)
        if comp.name == "net_income":
            # ProfitLoss includes non-controlling interests. It equals net income attributable to
            # the parent only when there is no NCI income, so that is the only case it is used.
            row = self._one(PROFIT_LOSS_TAG, comp.unit, anchor.accn, end, comp.kind)
            if row is not None:
                nci = self._one(NCI_INCOME_TAG, comp.unit, anchor.accn, end, comp.kind)
                if nci is None or nci.value == 0:
                    return Line(comp.name, row.value, row.unit, fiscal_year, row.end,
                                (PROFIT_LOSS_TAG,), row.accn, row.form, row.filed, from_anchor,
                                note="ProfitLoss used: the filing reports no NCI income")
        if comp.sum_of:
            parts: list[Row] = []
            for group in comp.sum_of:
                found = None
                for tag in group:
                    found = self._one(tag, comp.unit, anchor.accn, end, comp.kind)
                    if found is not None:
                        break
                if found is None:
                    return None
                parts.append(found)
            return Line(comp.name, sum(p.value for p in parts), comp.unit, fiscal_year,
                        parts[0].end, tuple(p.tag for p in parts), anchor.accn, parts[0].form,
                        parts[0].filed, from_anchor, note="sum of components")
        return None

    def value(self, component: str, fiscal_year: int, *, anchor: Anchor,
              as_of: date | None = None, faces: FaceStatementSource | None = None,
              concepts: ConceptSource | None = None) -> Line:
        """``component`` for ``fiscal_year``, read from ``anchor``'s columns where it has them.

        Order: the component's standard elements in the anchor; then (if ``faces`` is given) the
        element the anchor's own face statement prints beside the component's label; then the
        same two steps against that year's own 10-K when the anchor does not carry the year.
        """
        comp = COMPONENTS.get(component)
        if comp is None:
            raise Unresolved(f"unknown statement line {component!r}")

        def attempt(filing: Anchor, end: date, from_anchor: bool) -> Line | None:
            line = self._from_filing(comp, filing, end, fiscal_year, from_anchor=from_anchor)
            if line is None and faces is not None and concepts is not None:
                line = face_value(self, comp, filing, end, fiscal_year, faces=faces,
                                  concepts=concepts, from_anchor=from_anchor)
            return line

        end = anchor.end_for(fiscal_year)
        if end is not None:
            line = attempt(anchor, end, True)
            if line is not None:
                return line
        # The anchor does not carry this year (a balance sheet two years back) or does not carry
        # this line: read that year's own 10-K, and say so on the line.
        if fiscal_year != anchor.fiscal_year:
            try:
                own = self.anchor(fiscal_year, as_of=as_of)
            except Unresolved as exc:
                raise Unresolved(
                    f"{comp.description} for FY{fiscal_year}: not in the FY{anchor.fiscal_year} "
                    f"10-K and {exc}"
                ) from None
            own_end = own.end_for(fiscal_year)
            if own_end is not None:
                line = attempt(own, own_end, False)
                if line is not None:
                    return line
        tried = ", ".join(comp.tags + tuple("+".join(g) for g in comp.sum_of))
        raise Unresolved(
            f"{comp.description} for FY{fiscal_year}: no us-gaap element among {tried} in "
            f"{anchor.accn}"
        )


def _dedupe_year_ends(ends: list[date]) -> list[date]:
    """Collapse ends within a few days of each other (52/53-week drift across elements)."""
    out: list[date] = []
    for end in sorted(ends, reverse=True):
        if out and abs((out[-1] - end).days) <= DATE_SLACK_DAYS:
            continue
        out.append(end)
    return out


def compact(payload: dict[str, Any], *, keep: frozenset[str] | None = None) -> dict[str, Any]:
    """The projection of a ``companyfacts`` payload this module reads — small enough to snapshot.

    Keeps only the us-gaap elements in ``keep`` and only the row fields used above, so a frozen
    copy of every company in an evaluation is a few hundred kilobytes rather than hundreds of
    megabytes, and its SHA-256 is a stable fingerprint of exactly the evidence an answer used.
    """
    keep = keep if keep is not None else all_tags()
    gaap = (payload.get("facts") or {}).get("us-gaap") or {}
    fields = ("start", "end", "val", "accn", "fy", "fp", "form", "filed")
    out_gaap: dict[str, Any] = {}
    for tag in sorted(gaap):
        if tag not in keep:
            continue
        units = {}
        for unit, rows in sorted((gaap[tag].get("units") or {}).items()):
            # Only annual-report rows: every read in this module is anchored to a 10-K, so the
            # 10-Q rows are dead weight in a snapshot (and most of its size).
            kept = [{k: r.get(k) for k in fields if k in r} for r in rows
                    if str(r.get("form", "")) in ANNUAL_FORMS]
            if kept:
                units[unit] = kept
        if units:
            out_gaap[tag] = {"units": units}
    return {
        "cik": payload.get("cik"),
        "entityName": payload.get("entityName"),
        "facts": {"us-gaap": out_gaap},
    }


class CompanyFactsSource:
    """Fetch (or replay) ``companyfacts`` for a CIK. Keyless; one request per company.

    ``snapshot_dir`` makes a run replayable: a compact projection of every payload fetched is
    written there, and later runs read it instead of the network when ``offline`` is set.
    """

    def __init__(self, *, user_agent: str = _UA, snapshot_dir: Path | None = None,
                 offline: bool = False) -> None:
        self._ua = user_agent
        self._snapshot_dir = snapshot_dir
        self._offline = offline
        self._cache: dict[int, CompanyFacts] = {}
        self.requests = 0

    def _path(self, cik: int) -> Path | None:
        return None if self._snapshot_dir is None else self._snapshot_dir / f"CIK{cik:010d}.json"

    def _fetch(self, cik: int) -> dict[str, Any]:
        request = urllib.request.Request(
            COMPANYFACTS_URL.format(cik=cik),
            headers={"User-Agent": self._ua, "Accept": "application/json"},
        )
        self.requests += 1
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body: bytes = response.read()
        loaded: dict[str, Any] = json.loads(body.decode())
        return loaded

    def get(self, cik: int) -> CompanyFacts:
        if cik in self._cache:
            return self._cache[cik]
        path = self._path(cik)
        payload: dict[str, Any]
        if path is not None and path.exists() and self._offline:
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            if self._offline:
                raise Unresolved(f"CIK {cik}: offline and no snapshot at {path}")
            try:
                payload = compact(self._fetch(cik))
            except urllib.error.HTTPError as exc:
                raise Unresolved(f"CIK {cik}: SEC companyfacts HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                raise Unresolved(
                    f"CIK {cik}: SEC companyfacts unavailable ({type(exc).__name__})"
                ) from None
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")),
                                encoding="utf-8")
        facts = CompanyFacts.from_payload(payload)
        self._cache[cik] = facts
        return facts


def utc_today() -> date:
    return datetime.now(UTC).date()


# --- the face of the statements ----------------------------------------------------------------

FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/"
CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json"

FACE_LABELS: dict[str, str] = {
    "revenue": r"^(total )?(net )?(operating )?(revenues?|sales)( and other income)?$"
               r"|^(total )?net sales and (operating )?revenues?$|^net operating revenues$",
    "cogs": r"^(total )?cost of (sales|revenues?|goods sold|products sold|merchandise sold)"
            r"(,? excluding .*)?$",
    "operating_income": r"^(total )?operating (income|profit)( \(loss\))?$"
                        r"|^(income|earnings) from operations$",
    "inventory": r"^(total )?(merchandise |finished goods )?inventor(y|ies)(,? net)?$",
    "accounts_receivable": r"^(trade )?(accounts )?receivables?(,? net)?(,? less allowances?.*)?$"
                           r"|^(trade )?accounts receivable.*net",
    "accounts_payable": r"^(trade )?accounts payable( and accrued.*)?$",
    "ppe_net": r"^(total )?(net )?property,? plant,? (and|&) equipment(,? net.*| - net.*)?$"
               r"|^property,? plant,? (and|&) equipment.*\bnet\b",
    "capex": r"^(capital (expenditures|spending|additions))|^(purchases?|additions?|payments?) "
             r"(of|to|for) property",
    "dna": r"^depreciation(,)?( depletion)? (and|&) amortization",
    "dividends_paid": r"^(cash )?dividends( paid)?( to (common )?(stockholders|shareholders|"
                      r"shareowners))?$",
    "cfo": r"^net cash (provided by|from|generated by|provided from) operating",
    "current_assets": r"^total current assets$",
    "current_liabilities": r"^total current liabilities$",
    "total_assets": r"^total assets$",
}
"""Row labels, as printed on the face of a statement, that name each component.

Used only as a fallback: when none of a component's standard elements is present in the anchor
filing, the element the company itself put beside that printed label is read instead. Nike's
FY2021 balance sheet prints "Inventories" beside ``InventoryFinishedGoodsNetOfReserves`` — every
Nike inventory is finished goods — so the standard ``InventoryNet`` is absent and a synonym list
alone abstains on a line that is plainly on the page."""

_STATEMENT_OF: dict[str, tuple[str, ...]] = {
    "instant": ("balance sheet", "financial position", "financial condition"),
    "cash": ("cash flow",),
    "income": ("income", "operations", "earnings", "profit"),
}
_CASH_COMPONENTS = frozenset({"capex", "dna", "dividends_paid", "cfo", "buybacks"})


def _statement_keys(comp: Component) -> tuple[str, ...]:
    if comp.kind == "instant":
        return _STATEMENT_OF["instant"]
    if comp.name in _CASH_COMPONENTS:
        return _STATEMENT_OF["cash"]
    return _STATEMENT_OF["income"]


@dataclass(frozen=True)
class FaceRow:
    statement: str
    prefix: str
    concept: str
    label: str


def _clean_label(raw: str) -> str:
    text = re.sub(r"<[^>]+>", "", raw)
    text = html.unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip().lower().rstrip(":")


class FaceStatementSource:
    """Which element each printed line of a filing's primary statements carries.

    Reads the filing's own ``FilingSummary.xml`` and the rendered statement pages it lists — the
    same pages EDGAR's viewer shows — and records ``(element, printed label)`` for every row of the
    balance sheet, income statement and cash flow statement. No values are read from HTML: the
    number still comes from SEC's XBRL API, the page only says *which* element is the line.
    """

    def __init__(self, *, user_agent: str = _UA, snapshot_dir: Path | None = None,
                 offline: bool = False) -> None:
        self._ua = user_agent
        self._snapshot_dir = snapshot_dir
        self._offline = offline
        self._cache: dict[str, list[FaceRow]] = {}
        self.requests = 0

    def _get(self, url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": self._ua})
        self.requests += 1
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body: bytes = resp.read()
        return body.decode("utf-8", "replace")

    def rows(self, cik: int, accn: str) -> list[FaceRow]:
        if accn in self._cache:
            return self._cache[accn]
        path = (None if self._snapshot_dir is None
                else self._snapshot_dir / f"face_{accn}.json")
        if path is not None and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            out = [FaceRow(**r) for r in data]
            self._cache[accn] = out
            return out
        if self._offline:
            self._cache[accn] = []
            return []
        base = FILING_INDEX_URL.format(cik=cik, accn=accn.replace("-", ""))
        out = []
        try:
            summary = self._get(base + "FilingSummary.xml")
            for report in re.findall(r"<Report[ >].*?</Report>", summary, re.S):
                def tag(name: str, rep: str = report) -> str:
                    m = re.search(rf"<{name}>(.*?)</{name}>", rep, re.S)
                    return m.group(1).strip() if m else ""
                long_name, short = tag("LongName"), tag("ShortName").lower()
                category = tag("MenuCategory")
                is_statement = category == "Statements" or " - Statement - " in long_name
                if not is_statement or "parenthetical" in short:
                    continue
                if not any(k in short for keys in _STATEMENT_OF.values() for k in keys):
                    continue
                page = tag("HtmlFileName")
                if not page:
                    continue
                html_text = self._get(base + page)
                for prefix, concept, label in re.findall(
                        r"defref_([a-z][a-z0-9\-]*)_(\w+)'[^>]*>(.*?)</a>", html_text, re.S):
                    out.append(FaceRow(short, prefix, concept, _clean_label(label)))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            out = []
        self._cache[accn] = out
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps([r.__dict__ for r in out]), encoding="utf-8")
        return out

    def candidates(self, cik: int, accn: str, comp: Component) -> list[FaceRow]:
        """Face rows whose printed label names ``comp``, on the statement ``comp`` belongs to."""
        pattern = FACE_LABELS.get(comp.name)
        if pattern is None:
            return []
        keys = _statement_keys(comp)
        return [r for r in self.rows(cik, accn)
                if r.prefix == "us-gaap" and any(k in r.statement for k in keys)
                and re.search(pattern, r.label) and not r.concept.endswith("Abstract")]


class ConceptSource:
    """One us-gaap element for one company (``companyconcept``), for elements outside
    :func:`all_tags` that a face statement names. Snapshotted like everything else."""

    def __init__(self, *, user_agent: str = _UA, snapshot_dir: Path | None = None,
                 offline: bool = False) -> None:
        self._ua = user_agent
        self._snapshot_dir = snapshot_dir
        self._offline = offline
        self._cache: dict[tuple[int, str], list[Row]] = {}
        self.requests = 0

    def rows(self, cik: int, tag: str) -> list[Row]:
        key = (cik, tag)
        if key in self._cache:
            return self._cache[key]
        path = (None if self._snapshot_dir is None
                else self._snapshot_dir / f"concept_CIK{cik:010d}_{tag}.json")
        payload: dict[str, Any] | None = None
        if path is not None and path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif not self._offline:
            req = urllib.request.Request(CONCEPT_URL.format(cik=cik, tag=tag),
                                         headers={"User-Agent": self._ua,
                                                  "Accept": "application/json"})
            self.requests += 1
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                    raw = json.loads(resp.read().decode())
                fields = ("start", "end", "val", "accn", "fy", "fp", "form", "filed")
                payload = {"units": {u: [{k: r.get(k) for k in fields if k in r} for r in rows]
                                     for u, rows in (raw.get("units") or {}).items()}}
            except (urllib.error.URLError, TimeoutError, OSError, ValueError):
                payload = None
            if payload is not None and path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")),
                                encoding="utf-8")
        out: list[Row] = []
        for unit, raw_rows in ((payload or {}).get("units") or {}).items():
            for raw_row in raw_rows:
                end, filed = _d(raw_row.get("end")), _d(raw_row.get("filed"))
                if end is None or filed is None or "val" not in raw_row:
                    continue
                out.append(Row(tag=tag, unit=unit, value=float(raw_row["val"]),
                               start=_d(raw_row.get("start")), end=end,
                               accn=str(raw_row.get("accn", "")),
                               form=str(raw_row.get("form", "")), filed=filed,
                               fy=raw_row.get("fy"), fp=raw_row.get("fp")))
        self._cache[key] = out
        return out


def face_value(facts: CompanyFacts, comp: Component, anchor: Anchor, end: date,
               fiscal_year: int, *, faces: FaceStatementSource, concepts: ConceptSource,
               from_anchor: bool) -> Line | None:
    """The value of the element printed beside ``comp``'s label on the anchor's own statement.

    Returns ``None`` when the page has no such row, or the row's element carries no annual (or
    instant) value for the period in that filing. Two different face rows matching the same label
    with different values is an ambiguity, and is reported as one rather than resolved by order.
    """
    found: dict[float, Line] = {}
    for row in faces.candidates(facts.cik, anchor.accn, comp):
        rows = facts.rows_by_tag.get(row.concept) or concepts.rows(facts.cik, row.concept)
        for r in rows:
            if r.accn != anchor.accn or r.unit != comp.unit:
                continue
            if abs((r.end - end).days) > DATE_SLACK_DAYS:
                continue
            if comp.kind == "instant" and not r.is_instant:
                continue
            if comp.kind == "duration" and not r.is_annual:
                continue
            found.setdefault(r.value, Line(
                comp.name, r.value, r.unit, fiscal_year, r.end, (row.concept,), r.accn, r.form,
                r.filed, from_anchor, note=f"face-statement line '{row.label}'"))
            break
    if not found:
        return None
    if len(found) > 1:
        raise Unresolved(
            f"{comp.description}: the face statement prints {len(found)} different lines that "
            f"match its label ({', '.join(sorted({ln.tags[0] for ln in found.values()}))})"
        )
    return next(iter(found.values()))


__all__ = [
    "ANNUAL_DAYS",
    "COMPANYFACTS_URL",
    "COMPONENTS",
    "FACE_LABELS",
    "Anchor",
    "CompanyFacts",
    "CompanyFactsSource",
    "Component",
    "ConceptSource",
    "FaceRow",
    "FaceStatementSource",
    "Line",
    "Row",
    "Unresolved",
    "all_tags",
    "compact",
    "face_value",
    "utc_today",
]
