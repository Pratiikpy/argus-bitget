"""Run the REAL, pip-installed edgartools package (not the rebuild in
``eval/general_sue_comparison.py``) on the frozen companyfacts snapshot, and dump its quarterly
diluted-EPS fiscal-label pairing to JSON.

Not imported by ARGUS. edgartools (dgunning/edgartools, MIT) is installed from PyPI into its own
isolated venv (never ARGUS's ``.venv``) and run here with no network access, no ARGUS import, and
no reimplementation of its internals: the three calls below are the package's own public surface.

* ``edgar.entity.parser.EntityFactsParser.parse_company_facts`` (public) builds a real
  ``EntityFacts`` object directly from a companyfacts-shaped JSON dict -- the same shape
  ``data.sec.gov/api/xbrl/companyfacts/CIK##########.json`` serves, and the same shape
  ``eval/general_sue_comparison.py::companyfacts_rows`` extracted the frozen snapshot from. One
  dict per filer is synthesised here from the frozen rows (columns: start, end, val, fy, fp, form,
  filed, accn) under ``facts.us-gaap.EarningsPerShareDiluted.units["USD/shares"]``, offline.
* ``edgar.ttm.calculator.TTMCalculator(facts).quarterize()`` (public) is the real quarterization:
  70-120-day duration facts, YTD/annual derivation skipped for per-share units
  (``_is_additive_concept`` returns False for ``USD/shares``, so diluted EPS is never
  reconstructed from a YTD or FY fact), deduplicated one-fact-per-period-end (periodic form then
  latest filing date) and sorted by period_end ascending.
* ``edgar.entity.enhanced_statement.detect_fiscal_year_end`` and
  ``calculate_fiscal_year_for_label`` (both public, both imported by ``TTMCalculator`` itself for
  ``calculate_ttm_trend``) label each quarterized fact with edgartools' own (fiscal_year,
  fiscal_period), the fiscal year derived from period_end and the most common FY period_end month
  -- not the SEC's raw, filing-year-contaminated ``fy`` tag.

The only code NOT taken from the installed package is the pairing loop below
(``pair_by_label``): "pick the quarter labelled (fiscal_year - 1, same fiscal_period)" is a
user-level convention for computing a year-over-year change from those labels, not a function
edgartools itself exposes (its own ``calculate_ttm_trend`` compares TTM sums positionally, four
quarters back). The loop mirrors ``eval/general_sue_comparison.py::_pair_by_label`` exactly (same
index-with-collision-count, same newest-quarter-must-have-a-partner refusal, same
first-eight-labelled-changes window, same population-std SUE) so the two runs differ only in
whose ``quarterize()``/label functions produced the quarters -- the rebuild's or the real
package's.

This script only dumps the real package's readings to JSON; it does not know the artefact's
correctness rule or run any statistics. Score the output against
``argus/data/general_sue_comparison.json``'s ``edgartools_fiscal`` block, and against
``argus_dated_after``, under ARGUS's own venv (which has ``argus.eval.general_sue_comparison``'s
``TRUTH_YOY_DAYS``, ``is_true_yoy``, ``mcnemar_exact``, ``paired_counts`` importable) -- this venv
does not have ARGUS installed and must not need it.

Usage (the isolated edgartools venv, not ARGUS's)::

    python edgartools_real_runner.py \\
        --companyfacts ../../data/general_sue_frames_snapshot_companyfacts.json \\
        --out real_edgartools_readings.json
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

N_DELTAS = 8
"""``argus.market.sue.N_DELTAS`` -- copied as a literal, not imported (this venv has no ARGUS)."""

EPS_CONCEPT = "EarningsPerShareDiluted"
EPS_UNIT = "USD/shares"
COMPANYFACTS_COLUMNS = ("start", "end", "val", "fy", "fp", "form", "filed", "accn")


def _iso(d: date | None) -> str | None:
    return d.isoformat() if d else None


def companyfacts_payload(cik: str, rows: list[list[str]]) -> dict[str, Any]:
    """One filer's frozen EPS rows, reshaped into the companyfacts JSON
    ``EntityFactsParser.parse_company_facts`` expects: ``facts.us-gaap.<concept>.units[unit]``,
    each row a dict of the SEC's own field names."""
    facts = [dict(zip(COMPANYFACTS_COLUMNS, row, strict=True)) for row in rows]
    return {
        "cik": int(cik),
        "entityName": f"CIK{cik}",
        "facts": {"us-gaap": {EPS_CONCEPT: {"units": {EPS_UNIT: facts}}}},
    }


def population_std(values: list[float]) -> float:
    n = len(values)
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        if numerator == 0.0:
            return math.nan
        return math.inf if numerator > 0 else -math.inf
    return numerator / denominator


def pair_by_label(quarters: list[Any], labels: list[tuple[int, str]]) -> dict[str, Any]:
    """Mirrors ``eval/general_sue_comparison.py::_pair_by_label`` exactly, driven by the REAL
    package's quarterized facts and REAL fiscal labels instead of the rebuild's."""
    index: dict[tuple[int, str], Any] = {}
    collisions = 0
    for quarter, label in zip(quarters, labels, strict=True):
        if label in index:
            collisions += 1
        index[label] = quarter
    if not quarters:
        return {"sue": None, "refusal": "no quarters", "pairs": [], "collisions": collisions}
    changes: list[tuple[Any, Any]] = []
    for quarter, (fy, fp) in reversed(list(zip(quarters, labels, strict=True))):
        partner = index.get((fy - 1, fp))
        if partner is not None:
            changes.append((quarter, partner))
        elif quarter is quarters[-1]:
            return {
                "sue": None, "refusal": "newest quarter has no (fy - 1, fp) partner",
                "pairs": [], "collisions": collisions,
            }
    window = changes[:N_DELTAS]
    if len(window) < N_DELTAS:
        return {
            "sue": None, "refusal": "fewer than eight labelled changes",
            "pairs": [], "collisions": collisions,
        }
    deltas = [float(a.numeric_value) - float(b.numeric_value) for a, b in window]
    value = safe_divide(deltas[0], population_std(deltas))
    pairs = [[_iso(a.period_end), _iso(b.period_end)] for a, b in window]
    return {"sue": value, "refusal": None, "pairs": pairs, "collisions": collisions}


def run_one(EntityFactsParser: Any, TTMCalculator: Any, detect_fiscal_year_end: Any,
            calculate_fiscal_year_for_label: Any, cik: str,
            rows: list[list[str]]) -> dict[str, Any]:
    payload = companyfacts_payload(cik, rows)
    entity_facts = EntityFactsParser.parse_company_facts(payload)
    if entity_facts is None:
        return {"status": "parse_failed", "sue": None, "refusal": "EntityFactsParser returned None",
                "pairs": [], "collisions": 0, "quarters": 0, "fiscal_year_end_month": None}
    facts = [f for f in entity_facts.get_all_facts() if f.concept == f"us-gaap:{EPS_CONCEPT}"]
    if not facts:
        return {"status": "no_facts", "sue": None, "refusal": "no us-gaap:EarningsPerShareDiluted "
                "facts after real parse", "pairs": [], "collisions": 0, "quarters": 0,
                "fiscal_year_end_month": None}
    quarters = TTMCalculator(facts).quarterize()
    fye = detect_fiscal_year_end(facts)
    labels = [(calculate_fiscal_year_for_label(q.period_end, fye), q.fiscal_period)
              for q in quarters]
    result = pair_by_label(quarters, labels)
    result["status"] = "ok"
    result["quarters"] = len(quarters)
    result["fiscal_year_end_month"] = fye
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--companyfacts", required=True,
                         help="path to general_sue_frames_snapshot_companyfacts.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 = every filer in the snapshot")
    args = parser.parse_args()

    from edgar import __version__ as edgar_version
    from edgar.entity.enhanced_statement import (
        calculate_fiscal_year_for_label,
        detect_fiscal_year_end,
    )
    from edgar.entity.parser import EntityFactsParser
    from edgar.ttm.calculator import TTMCalculator

    snapshot = json.loads(Path(args.companyfacts).read_text(encoding="utf-8"))
    anchor_ciks: dict[str, str] = snapshot["anchor_ciks"]
    sample: list[str] = snapshot["sample"]
    ordered = [*sample, *[c for c in anchor_ciks.values() if c not in set(sample)]]
    if args.limit:
        ordered = ordered[: args.limit]

    out: dict[str, Any] = {
        "runner": "edgartools_real_runner", "edgartools_version": edgar_version,
        "method": "real pip-installed edgartools: EntityFactsParser.parse_company_facts -> "
                  "TTMCalculator(facts).quarterize() -> detect_fiscal_year_end + "
                  "calculate_fiscal_year_for_label -> pair_by_label (fiscal_year - 1, same "
                  "fiscal_period), first eight labelled changes, population-std SUE",
        "companyfacts_source": Path(args.companyfacts).name,
        "filers": {},
    }
    for cik in ordered:
        entry = snapshot["filers"].get(cik) or {"status": "not_fetched", "rows": []}
        if entry["status"] != "ok":
            out["filers"][cik] = {"status": entry["status"], "sue": None, "refusal": None,
                                   "pairs": [], "collisions": 0, "quarters": 0,
                                   "fiscal_year_end_month": None}
            continue
        out["filers"][cik] = run_one(
            EntityFactsParser, TTMCalculator, detect_fiscal_year_end,
            calculate_fiscal_year_for_label, cik, entry["rows"],
        )

    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"edgartools {edgar_version}: wrote {len(out['filers'])} filers to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
