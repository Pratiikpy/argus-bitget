"""Run edgartools' real ``EntityFacts``/``FactQuery.as_of`` on a frozen SEC snapshot, in its own
process.

Not imported by ARGUS. edgartools (dgunning/edgartools, MIT) is run from its own clone at
``research/repos-themed/dgunning~edgartools``, installed editable in its own venv
(``research/_venvs/edgartools``). Two real pieces of its code run here, with no network access:

* **``edgar/entity/parser.py::EntityFactsParser.parse_company_facts``** builds a real
  ``EntityFacts`` object directly from the SEC's own companyfacts JSON shape — the snapshot file
  is exactly what ``data.sec.gov/api/xbrl/companyfacts/CIK##########.json`` returns, so this needs
  no monkeypatched transport, unlike the other runners here.
* **``edgar/entity/query.py::FactQuery.as_of(date)``** is edgartools' real point-in-time filter:
  "get facts as of a specific date", implemented as ``f.filing_date <= as_of_date`` — day-granular,
  like Qlib's. Chained with its own ``by_period_type('quarterly')`` and ``latest_periods(n=1,
  annual=False)`` (its documented "facts from the n most recent periods" call), this is the
  ordinary, undecorated way to ask edgartools "the latest quarter's revenue as of X".

Concept selection: ``by_concept(tag, exact=True)`` is run once per tag in the same priority order
``argus/market/fundamentals.py``'s ``CONCEPTS["revenue"]`` uses (copied below, not imported — this
venv does not install ARGUS), and the results merged, mirroring the alias-tag cleanup ARGUS itself
does. A first attempt used ``by_concept("revenue", exact=False)`` (edgartools' free-text fuzzy
search, its most "ask it like an agent would" form) and found a real defect worth recording: the
substring match pulls in ``us-gaap:CostOfRevenue`` (contains "revenue") alongside the real revenue
tags, so a query written the way an agent unfamiliar with ARGUS's tag list naturally would can
silently answer with cost of revenue. Exact-tag querying avoids that here so the measurement is of
edgartools' point-in-time mechanism, not of that separate fuzzy-search defect; the defect itself is
recorded in the runner's output for the report.

Within the selected period, multiple facts can remain (different tags or re-filings); edgartools'
own ``.latest(n)`` convention (sort by ``filing_date`` descending) picks the one to show, applied
here across the merged tags the same way it would within one.

Usage::

    python edgartools_asof_runner.py --clone DIR --snapshot DIR --probes probes.json \
        --tickers NVDA --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REVENUE_TAGS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--tickers", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.clone)))
    from edgar.entity.parser import EntityFactsParser

    snapshot = Path(args.snapshot)
    probes = json.loads(Path(args.probes).read_text(encoding="utf-8"))["tickers"]

    out: dict[str, Any] = {
        "runner": "edgartools_asof_runner", "clone": Path(args.clone).resolve().name,
        "fuzzy_concept_defect": {
            "note": "by_concept('revenue', exact=False) also matches us-gaap:CostOfRevenue "
                    "(substring 'revenue'); not used for scoring, recorded once below",
        },
        "tickers": {},
    }
    fuzzy_checked = False
    for ticker in [t.strip().upper() for t in args.tickers.split(",") if t.strip()]:
        companyfacts = json.loads(
            (snapshot / f"{ticker}.companyfacts.json").read_text(encoding="utf-8"))
        ef = EntityFactsParser.parse_company_facts(companyfacts)
        ticker_probes = probes.get(ticker, [])
        if ef is None or not ticker_probes:
            out["tickers"][ticker] = {"probes": [], "note": "entity facts failed to parse"}
            continue
        if not fuzzy_checked:
            fuzzy = ef.query().by_concept("revenue", exact=False).by_period_type(
                "quarterly").execute()
            out["fuzzy_concept_defect"]["sample_concepts"] = sorted(
                {f.concept for f in fuzzy})
            fuzzy_checked = True
        answered = []
        for probe in ticker_probes:
            as_of = datetime.fromisoformat(probe["as_of"])
            if as_of.tzinfo is None:
                as_of = as_of.replace(tzinfo=UTC)
            as_of_date = as_of.astimezone(UTC).date()
            candidates = []
            for tag in REVENUE_TAGS:
                q = (ef.query().by_concept(f"us-gaap:{tag}", exact=True)
                     .by_period_type("quarterly").as_of(as_of_date)
                     .latest_periods(n=1, annual=False))
                candidates.extend(q.execute())
            shown = max(candidates, key=lambda f: (f.filing_date, f.period_end)) \
                if candidates else None
            answered.append({
                "accn": probe["accn"], "side": probe["side"], "as_of": probe["as_of"],
                "end": shown.period_end.isoformat() if shown else None,
                "value": shown.numeric_value if shown else None,
                "filing_date": shown.filing_date.isoformat() if shown else None,
                "concept": shown.concept if shown else None,
            })
        out["tickers"][ticker] = {"probes": answered}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
