"""Quarterly capital spending of the four largest AI buyers, from their own SEC filings.

"AI capex is peaking" is a claim about four cash-flow statements: Microsoft, Alphabet, Amazon and
Meta buy most of the world's AI data-centre hardware, and each reports what it paid for property
and equipment every quarter. This module reads those figures from SEC EDGAR's XBRL company facts
and turns them into quarters, so the claim can be tested rather than asserted.

**The tags.** ``PaymentsToAcquirePropertyPlantAndEquipment`` for Microsoft, Alphabet and Meta;
Amazon stopped using it after Q1 2017 and reports ``PaymentsToAcquireProductiveAssets`` instead
(checked in its company facts, 2026-10-05).

**Quarters from year-to-date figures.** A 10-Q's cash-flow statement is year-to-date: six or nine
months, not the quarter. A quarter is therefore the difference between two year-to-date figures
that share a fiscal-year start (Microsoft's 2026 fourth quarter is its 10-K's twelve months,
$115.95bn, less its nine months to March, $80.15bn: $35.80bn), and a three-month figure is taken
as it is.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Final

COMPANIES: Final = (("MSFT", "Microsoft"), ("GOOGL", "Alphabet"), ("AMZN", "Amazon"),
                    ("META", "Meta"))
TAGS: Final = ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")


def quarters(rows: list[dict[str, Any]]) -> dict[date, float]:
    """Quarter-end date -> spending in that quarter, from 10-Q and 10-K duration rows."""
    spans: dict[tuple[date, date], float] = {}
    for row in rows:
        if row.get("form") not in ("10-Q", "10-K") or not row.get("start"):
            continue
        spans[(date.fromisoformat(row["start"]), date.fromisoformat(row["end"]))] = float(
            row["val"])
    out: dict[date, float] = {}
    for (start, end), value in spans.items():
        days = (end - start).days
        if 80 <= days <= 100:
            out[end] = value
            continue
        earlier = [(e, v) for (s, e), v in spans.items() if s == start and e < end
                   and 80 <= (end - e).days <= 100]
        if earlier and end not in out:
            out[end] = value - max(earlier)[1]
    return out


def read(company: str) -> dict[date, float]:
    from argus.market.evidence import EdgarSource

    edgar = EdgarSource()
    cik = edgar.cik_for(company)
    if cik is None:
        return {}
    facts = edgar._get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
    tags = facts["facts"]["us-gaap"]
    best: dict[date, float] = {}
    for tag in TAGS:
        found = quarters(tags.get(tag, {}).get("units", {}).get("USD", []))
        if found and (not best or max(found) > max(best)):
            best = found
    return best


def combined(last: int = 9) -> tuple[list[tuple[str, float]], dict[str, float], list[str]]:
    """The four companies' spending summed by calendar quarter (each company's fiscal quarter is
    filed under the calendar quarter it ends in), the latest quarter per company, and the names
    that could not be read."""
    by_company: dict[str, dict[date, float]] = {}
    missing: list[str] = []
    for ticker, name in COMPANIES:
        try:
            by_company[name] = read(ticker)
        except Exception:
            missing.append(name)
    if not by_company:
        return [], {}, missing

    def label(day: date) -> str:
        return f"{day.year}Q{(day.month - 1) // 3 + 1}"

    totals: dict[str, dict[str, float]] = {}
    for name, series in by_company.items():
        for end, value in series.items():
            totals.setdefault(label(end), {})[name] = value
    complete = sorted(q for q, parts in totals.items() if len(parts) == len(by_company))
    rows = [(q, sum(totals[q].values())) for q in complete][-last:]
    latest = {name: series[max(series)] for name, series in by_company.items() if series}
    return rows, latest, missing


__all__ = ["COMPANIES", "combined", "quarters", "read"]
