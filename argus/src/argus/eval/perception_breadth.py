"""How much the desk can see about one stock, against OpenBB's keyless providers on the same day.

Track 3 is judged on "feature depth (data sources / Skill integration count **and effectiveness**)",
and capability 19 (the perception layer) had only ever been compared on a design property: that
``market/evidence.gather`` never raises where TradingAgents' ``route_to_vendor`` does. Its named
baseline, OpenBB's provider set, was counted statically (32 provider packages in the clone) against
the five evidence *kinds* the desk notes record — which undercounts ARGUS, whose five kinds are fed
by more than a dozen upstreams, and overcounts OpenBB, most of whose providers need a paid key.
This module runs both on the same twelve underlyings on the same day and counts what answered.

**The two sides, and why each is measured the way it is.**

* **OpenBB.** ``scripts/pit_runners/openbb_breadth_runner.py`` calls every per-symbol endpoint of
  every keyless provider through OpenBB's own ``obb`` interface, in its own virtual environment
  (OpenBB is AGPL-3.0; nothing of it is imported here). A cell answers when the call returns at
  least one row. Keyed providers are excluded and named, with the credential that excludes them.
  The run is saved to ``data/openbb_breadth_raw.json`` and this module only reads it.
* **ARGUS.** The desk's own record of its last live cycle on the same day
  (``data/desk_notes.jsonl``): every evidence item the decision-maker actually received, keyed by
  the id each source stamps on it. That is the stricter reading: a feed that was reachable but
  returned nothing relevant to the symbol in its window (an EDGAR sweep with no 8-K this week, an
  RSS feed with no headline naming the company) does not count, where on the OpenBB side a
  historical filings list always does.

**The unit is a data category, not a provider or an endpoint.** Counting endpoints rewards a
provider that splits one statement into ``income``, ``income_growth`` and ``company_facts``;
counting providers rewards three vendors serving the same quote. So each answering cell is mapped
to one of the categories in :data:`CATEGORIES`, and a symbol scores the number of categories at
least one of a system's sources answered. The mapping is published in full in the artefact
(``category_of``) so a reader who would draw the lines differently can recount from the rows.
Derived analytics (Bitget's ``technical_analysis`` Skill; OpenBB's ``technical`` extension, not
installed) are listed but not scored: both compute from a price series either system already has.
Market-wide feeds (rates, VIX, fear and greed, crypto-market mirrors on the ARGUS side; screeners
and calendars on the OpenBB side) are listed separately and not scored per symbol.

NOT claimed: that more categories make better decisions. Effectiveness, what each feed changes in
a decision, needs paired model cycles with and without it and has not been run.

Reproduce::

    research/_venvs/pit-rivals/Scripts/python scripts/pit_runners/openbb_breadth_runner.py \\
        --tickers NVDA,TSLA,AAPL,MSFT,META,GOOGL,AMZN,COIN,MSTR,QQQ,TQQQ,SQQQ \\
        --out data/openbb_breadth_raw.json
    python -m argus.eval.perception_breadth
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from argus.market.bitget import ANCHOR_OF
from argus.truth import artefact

PACKAGE = Path(__file__).resolve().parents[3]
DATA = PACKAGE / "data"
RAW_PATH = DATA / "openbb_breadth_raw.json"
WORKBENCH_PATH = DATA / "perception_workbench_raw.json"
NOTES_PATH = DATA / "desk_notes.jsonl"
REPORT_PATH = DATA / "perception_breadth.json"

CATEGORIES: tuple[str, ...] = (
    "price_quote", "options", "futures", "dark_pool", "short_activity", "estimates",
    "analyst_targets", "fundamentals", "filings", "insider", "ownership", "corporate_actions",
    "company_profile", "news", "social", "etf_composition", "earnings_calendar",
    "venue_derivatives", "prediction_markets",
)
"""Every per-symbol data category either side serves. A category neither side answers for a
symbol simply scores zero for both."""

OPENBB_CATEGORY: dict[str, str] = {
    "equity.price.historical": "price_quote",
    "equity.price.quote": "price_quote",
    "equity.price.performance": "price_quote",
    "etf.historical": "price_quote",
    "etf.price_performance": "price_quote",
    "derivatives.options.chains": "options",
    "derivatives.futures.curve": "futures",
    "equity.darkpool.otc": "dark_pool",
    "equity.shorts.short_interest": "short_activity",
    "equity.shorts.short_volume": "short_activity",
    "equity.shorts.fails_to_deliver": "short_activity",
    "equity.estimates.consensus": "estimates",
    "equity.estimates.forward_eps": "estimates",
    "equity.estimates.forward_sales": "estimates",
    "equity.estimates.price_target": "analyst_targets",
    "equity.fundamental.balance": "fundamentals",
    "equity.fundamental.balance_growth": "fundamentals",
    "equity.fundamental.cash": "fundamentals",
    "equity.fundamental.cash_growth": "fundamentals",
    "equity.fundamental.income": "fundamentals",
    "equity.fundamental.income_growth": "fundamentals",
    "equity.fundamental.metrics": "fundamentals",
    "equity.compare.company_facts": "fundamentals",
    "equity.fundamental.filings": "filings",
    "equity.fundamental.management_discussion_analysis": "filings",
    "equity.ownership.insider_trading": "insider",
    "equity.ownership.form_13f": "ownership",
    "equity.ownership.share_statistics": "ownership",
    "equity.fundamental.dividends": "corporate_actions",
    "equity.fundamental.management": "company_profile",
    "equity.profile": "company_profile",
    "news.company": "news",
    # Market-wide, turned into one row per ticker by the runner: answered when the ticker is
    # listed with an upcoming report date (openbb_breadth_runner._earnings_by_symbol).
    "equity.calendar.earnings": "earnings_calendar",
    "etf.countries": "etf_composition",
    "etf.holdings": "etf_composition",
    "etf.sectors": "etf_composition",
    "etf.info": "etf_composition",
    "etf.nport_disclosure": "etf_composition",
}

# ARGUS evidence ids, as each source stamps them (market/evidence.py, market/insider.py,
# market/fundamentals.py, market/estimates.py, market/microstructure.py, market/skills.py,
# market/skill_mirror.py, paper/runner.py). Order matters: the first pattern that matches wins.
ARGUS_SOURCES: tuple[tuple[str, str, str], ...] = (
    (r"^mkt-", "bitget", "price_quote"),
    (r"^rss-yahoo-", "yahoo-rss", "news"),
    (r"^rss-", "rss", "news"),
    (r"^twitter-", "x", "social"),
    (r"^reddit-", "reddit", "social"),
    (r"^edgar-", "sec-edgar", "filings"),
    (r"^form4-", "sec-form4", "insider"),
    (r"^xbrl-", "sec-xbrl", "fundamentals"),
    (r"^consensus-", "yahoo-estimates", "estimates"),
    (r"^finra-short-", "finra", "short_activity"),
    (r"^skill-", "bitget-skill", "derived"),
    (r"^mirror-", "skill-mirror", "market_wide"),
    (r"^(ust-|fng-|vix-)", "macro", "market_wide"),
    (r"^(feeds-|panel)", "meta", "meta"),
)

NOT_SCORED = ("derived", "market_wide", "meta")

WORKBENCH_KINDS = ("quote", "fundamentals", "news", "sentiment", "technicals", "event")
"""The research kinds that answer about one named stock. Book, stress, hedge and execution answer
about a position, and macro about no stock at all."""

# The research workbench's sources, as each engine cites them (lui/research/*.py). Matched against
# "ref detail"; the first pattern wins. A source whose detail says the figure was withheld is not
# an answer (fundamentals withholds a consensus scraped too long ago).
WORKBENCH_SOURCES: tuple[tuple[str, str, str], ...] = (
    (r"withheld", "withheld", "meta"),
    (r"^bitget /api/v2/mix/market/tickers|^yahoo daily close", "bitget+yahoo", "price_quote"),
    (r"^bitget-mcp-server quote|equity_fundamental_ratios|sue via SEC XBRL",
     "bitget-mcp+sec", "fundamentals"),
    (r"^bitget-mcp-server consensus", "bitget-mcp", "estimates"),
    (r"equity_estimates_price_target", "bitget-mcp", "analyst_targets"),
    (r"13F holdings|inst_position_summary|major_holders", "bitget-mcp", "ownership"),
    (r"insider_trading", "bitget-mcp", "insider"),
    (r"equity_profile", "bitget-mcp", "company_profile"),
    (r"^bitget-mcp-server earnings calendar, consensus", "bitget-mcp", "meta"),
    (r"equity_calendar", "bitget-mcp", "earnings_calendar"),
    (r"crypto_institutional_company_flow", "bitget-mcp", "fundamentals"),
    (r"fear ?& ?greed|sentiment_market_fear_greed", "bitget-mcp", "market_wide"),
    (r"equity_fundamental_dividends", "bitget-mcp", "corporate_actions"),
    (r"8-K exhibit|^SEC EDGAR submissions", "sec-edgar", "filings"),
    (r"^Yahoo Finance quoteSummary calendarEvents .*consensus", "yahoo", "estimates"),
    (r"^Yahoo Finance quoteSummary calendarEvents", "yahoo", "earnings_calendar"),
    (r"^Yahoo Finance quoteSummary financialData", "yahoo", "analyst_targets"),
    (r"^Yahoo Finance quoteSummary topHoldings", "yahoo", "etf_composition"),
    (r"^finra regsho", "finra", "short_activity"),
    (r"RSS \+ Yahoo Finance \+ SEC EDGAR|bitget \+ RSS \+ SEC EDGAR", "rss+edgar", "news"),
    (r"social_pulse", "x+reddit", "social"),
    (r"^cboe ", "cboe", "options"),
    (r"^finra ", "finra", "dark_pool"),
    (r"^Bitget funding|crypto_derivatives", "bitget", "venue_derivatives"),
    (r"^Polymarket", "polymarket", "prediction_markets"),
    (r"^SoSoValue", "sosovalue", "market_wide"),
    (r"^bitget-signal technical_analysis", "bitget-skill", "derived"),
    (r"^argus\.|^data/|^desk notes|^venue|^contracts", "argus", "derived"),
)
_FILINGS_IN_DETAIL = re.compile(r"(\d+) filing\(s\)")
_DATE_IN_DETAIL = re.compile(r"(\d{4}-\d{2}-\d{2})")


class BreadthError(RuntimeError):
    """The inputs cannot support the comparison as specified."""


@dataclass(frozen=True)
class Cell:
    """One (symbol, system, source, category) observation."""

    symbol: str
    system: str
    source: str
    category: str
    answered: bool
    items: int
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "system": self.system, "source": self.source,
                "category": self.category, "answered": self.answered, "items": self.items,
                "detail": self.detail}


def _argus_category(evidence_id: str) -> tuple[str, str] | None:
    for pattern, source, category in ARGUS_SOURCES:
        if re.match(pattern, evidence_id):
            return source, category
    return None


def openbb_cells(raw: Mapping[str, Any]) -> list[Cell]:
    """The runner's per-symbol rows, each mapped to its category."""
    cells: list[Cell] = []
    unknown = sorted({str(r["endpoint"]) for r in raw["per_symbol"]} - set(OPENBB_CATEGORY))
    if unknown:
        raise BreadthError(f"OpenBB endpoint(s) with no category: {', '.join(unknown)}; "
                           "map them in OPENBB_CATEGORY rather than dropping them")
    for r in raw["per_symbol"]:
        cells.append(Cell(
            symbol=str(r["ticker"]), system="openbb", source=str(r["provider"]),
            category=OPENBB_CATEGORY[str(r["endpoint"])], answered=bool(r["answered"]),
            items=int(r.get("rows") or 0),
            detail=f"{r['endpoint']}" + (f": {r['error']}" if r.get("error") else "")))
    return cells


def last_cycle(notes: Iterable[Mapping[str, Any]], *, day: str,
               before: datetime) -> dict[str, Mapping[str, Any]]:
    """The last recorded decision per perpetual on ``day``, at or before ``before``."""
    out: dict[str, Mapping[str, Any]] = {}
    for note in notes:
        at = str(note.get("at", ""))
        if not at.startswith(day) or datetime.fromisoformat(at) > before:
            continue
        symbol = str(note["symbol"])
        if symbol not in out or str(out[symbol]["at"]) <= at:
            out[symbol] = note
    return out


def argus_cells(cycles: Mapping[str, Mapping[str, Any]]) -> list[Cell]:
    """Every evidence item the decision-maker received on its last cycle, grouped by source."""
    cells: list[Cell] = []
    for perp, note in sorted(cycles.items()):
        ticker = ANCHOR_OF.get(perp)
        if ticker is None:
            raise BreadthError(f"{perp} has no underlying in market/bitget.ANCHOR_OF")
        counts: dict[tuple[str, str], int] = {}
        unmatched: list[str] = []
        for line in note.get("evidence", []):
            match = re.match(r"\[([^\]]+)\]", str(line))
            if match is None:
                unmatched.append(str(line)[:40])
                continue
            mapped = _argus_category(match.group(1))
            if mapped is None:
                unmatched.append(match.group(1))
                continue
            counts[mapped] = counts.get(mapped, 0) + 1
        if unmatched:
            raise BreadthError(f"{perp}: evidence id(s) with no source mapping: "
                               f"{', '.join(unmatched[:5])}; map them in ARGUS_SOURCES")
        for (source, category), n in sorted(counts.items()):
            cells.append(Cell(symbol=ticker, system="argus", source=source, category=category,
                              answered=n > 0, items=n, detail=f"seq {note.get('seq')} at "
                              f"{note.get('at')}"))
    return cells


def _workbench_category(ref: str, detail: str) -> tuple[str, str] | None:
    text = f"{ref} {detail}"
    for pattern, source, category in WORKBENCH_SOURCES:
        if re.search(pattern, text):
            return source, category
    return None


def read_workbench(symbols: Sequence[str]) -> dict[str, Any]:  # pragma: no cover - live
    """Run every per-stock research kind for each perpetual and record the sources it cited.
    Live and keyless: no model is called (the engines compute; the planner is not used)."""
    import time as _time
    from datetime import UTC

    from argus.lui.research import dispatch
    from argus.lui.research.kinds import ResearchKind, ResearchRequest

    rows: list[dict[str, Any]] = []
    for perp in symbols:
        for kind in WORKBENCH_KINDS:
            start = _time.monotonic()
            try:
                answer = dispatch.run(kind, ResearchRequest(kind=ResearchKind(kind),
                                                            symbols=(perp,)))
                cited = [s.as_dict() for s in answer.sources]
                error = None
            except Exception as exc:  # a failed engine is recorded, not hidden
                cited, error = [], f"{type(exc).__name__}: {exc}"[:200]
            rows.append({"symbol": ANCHOR_OF[perp], "kind": kind, "sources": cited,
                         "seconds": round(_time.monotonic() - start, 2), "error": error})
    return {"as_of_utc": datetime.now(UTC).isoformat(), "kinds": list(WORKBENCH_KINDS),
            "rows": rows}


def workbench_cells(raw: Mapping[str, Any]) -> list[Cell]:
    """The workbench's cited sources, each mapped to its category, one cell per source kind."""
    cells: list[Cell] = []
    unmatched: set[str] = set()
    for row in raw["rows"]:
        counts: dict[tuple[str, str], int] = {}
        for src in row["sources"]:
            ref, detail = str(src.get("ref", "")), str(src.get("detail", ""))
            mapped = _workbench_category(ref, detail)
            if mapped is None:
                unmatched.add(ref)
                continue
            if mapped[1] == "earnings_calendar":
                # Scored as on the OpenBB side: only an upcoming report date counts. A past one
                # is history, and counting it credited nine names on 2026-09-29 whose answers
                # had no next date at all (fresh-eyes audit).
                found = _DATE_IN_DETAIL.search(detail)
                if found is None or found.group(1) < str(raw["as_of_utc"])[:10]:
                    mapped = (mapped[0], "meta")
            counts[mapped] = counts.get(mapped, 0) + 1
            filings = _FILINGS_IN_DETAIL.search(detail)
            if mapped[1] == "news" and filings and int(filings.group(1)) > 0:
                key = ("sec-edgar", "filings")
                counts[key] = counts.get(key, 0) + 1
        for (source, category), n in sorted(counts.items()):
            cells.append(Cell(symbol=str(row["symbol"]), system="argus_workbench", source=source,
                              category=category, answered=True, items=n,
                              detail=f"kind {row['kind']}"))
    if unmatched:
        raise BreadthError(f"workbench source(s) with no mapping: {', '.join(sorted(unmatched))}; "
                           "map them in WORKBENCH_SOURCES")
    return cells


def per_symbol(cells: Sequence[Cell], symbols: Sequence[str]) -> list[dict[str, Any]]:
    """Scored categories and answering sources per symbol and system."""
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        row: dict[str, Any] = {"symbol": symbol}
        for system in ("argus", "argus_workbench", "openbb"):
            mine = [c for c in cells if c.symbol == symbol and c.system == system and c.answered
                    and c.category not in NOT_SCORED]
            row[f"{system}_categories"] = sorted({c.category for c in mine})
            row[f"{system}_sources"] = sorted({c.source for c in mine})
        row["only_argus"] = sorted(set(row["argus_categories"]) - set(row["openbb_categories"]))
        row["only_openbb"] = sorted(set(row["openbb_categories"]) - set(row["argus_categories"]))
        row["category_difference"] = len(row["argus_categories"]) - len(row["openbb_categories"])
        row["workbench_difference"] = (len(row["argus_workbench_categories"])
                                       - len(row["openbb_categories"]))
        row["only_workbench"] = sorted(set(row["argus_workbench_categories"])
                                       - set(row["openbb_categories"]))
        row["only_openbb_vs_workbench"] = sorted(set(row["openbb_categories"])
                                                 - set(row["argus_workbench_categories"]))
        rows.append(row)
    return rows


def run(raw: Mapping[str, Any], notes: Iterable[Mapping[str, Any]],
        workbench: Mapping[str, Any] | None = None) -> dict[str, Any]:
    as_of = datetime.fromisoformat(str(raw["as_of_utc"]))
    cycles = last_cycle(notes, day=as_of.date().isoformat(), before=as_of)
    if not cycles:
        raise BreadthError(f"no ARGUS decision recorded on {as_of.date()} before {as_of}; the "
                           "two sides must be read on the same day")
    ob = openbb_cells(raw)
    ar = argus_cells(cycles)
    wb = workbench_cells(workbench) if workbench is not None else []
    both = sorted({c.symbol for c in ar} & {c.symbol for c in ob})
    rows = per_symbol(ob + ar + wb, both)
    diffs = [r["category_difference"] for r in rows]
    mean = sum(diffs) / len(diffs)

    def share(system: str, category: str) -> int:
        return sum(1 for r in rows if category in r[f"{system}_categories"])

    by_category = {cat: {"argus": share("argus", cat),
                         "argus_workbench": share("argus_workbench", cat),
                         "openbb": share("openbb", cat)} for cat in CATEGORIES}
    wdiffs = [r["workbench_difference"] for r in rows]
    workbench_summary = None if workbench is None else {
        "as_of": workbench["as_of_utc"],
        "mean_workbench_categories": round(sum(len(r["argus_workbench_categories"])
                                               for r in rows) / len(rows), 3),
        "mean_workbench_difference": round(sum(wdiffs) / len(wdiffs), 3),
        "symbols_workbench_ahead": sum(1 for d in wdiffs if d > 0),
        "symbols_openbb_ahead": sum(1 for d in wdiffs if d < 0),
        "symbols_tied": sum(1 for d in wdiffs if d == 0),
        "verdict": ("rival_better" if all(d < 0 for d in wdiffs) else
                    "argus_better" if all(d > 0 for d in wdiffs) else "mixed"),
    }
    verdict = ("rival_better" if all(d < 0 for d in diffs) else
               "argus_better" if all(d > 0 for d in diffs) else "mixed")
    return {
        "question": "per-symbol data categories that answered with data, same underlyings, same "
                    "day: ARGUS's live desk against every keyless OpenBB provider",
        "openbb_as_of": raw["as_of_utc"],
        "openbb_commit": raw["openbb_commit"],
        "openbb_packages": raw["installed_openbb_packages"],
        "openbb_keyless_tested": raw["keyless_tested_providers"],
        "openbb_excluded_keyed": raw["excluded_keyed_providers"],
        "openbb_keyless_not_tested": raw["keyless_not_tested_no_per_symbol_endpoint"],
        "argus_cycles": {ANCHOR_OF[p]: {"seq": n.get("seq"), "at": n.get("at")}
                         for p, n in sorted(cycles.items())},
        "symbols": both,
        "symbols_without_an_argus_cycle": sorted({c.symbol for c in ob} - set(both)),
        "category_of": {"openbb": OPENBB_CATEGORY,
                        "argus": {p: [s, c] for p, s, c in ARGUS_SOURCES},
                        "argus_workbench": {p: [s, c] for p, s, c in WORKBENCH_SOURCES}},
        "not_scored": list(NOT_SCORED),
        "per_symbol": rows,
        "by_category": by_category,
        "summary": {
            "symbols": len(rows),
            "mean_argus_categories": round(sum(len(r["argus_categories"]) for r in rows)
                                           / len(rows), 3),
            "mean_openbb_categories": round(sum(len(r["openbb_categories"]) for r in rows)
                                            / len(rows), 3),
            "mean_category_difference": round(mean, 3),
            "symbols_argus_ahead": sum(1 for d in diffs if d > 0),
            "symbols_openbb_ahead": sum(1 for d in diffs if d < 0),
            "symbols_tied": sum(1 for d in diffs if d == 0),
            "verdict": verdict,
        },
        "workbench_summary": workbench_summary,
        "baseline": {
            "system": "OpenBB keyless providers, run through OpenBB's own obb interface",
            "calls": len(ob), "answered": sum(1 for c in ob if c.answered),
            "providers_answering": sorted({c.source for c in ob if c.answered}),
            "providers_silent": sorted({c.source for c in ob} - {c.source for c in ob
                                                                 if c.answered}),
        },
        "failure_cases": [
            {"system": "argus_workbench", "symbol": r["symbol"], "missing": r[
                "only_openbb_vs_workbench"]} for r in rows if r["only_openbb_vs_workbench"]
        ] + [{"system": "argus_desk", "symbol": r["symbol"], "missing": r["only_openbb"]}
             for r in rows if r["only_openbb"]] + [
            {"system": "openbb", "excluded_keyed_providers": raw["excluded_keyed_providers"]}],
        "cells": [c.as_dict() for c in ob + ar + wb],
        "scope_statement": (
            "Claimed: on the same underlyings and the same day, the number of per-symbol data "
            "categories that returned data, ARGUS's last live cycle against every keyless OpenBB "
            "provider run through OpenBB's own interface. ARGUS is read strictly (only evidence "
            "the decision-maker received), OpenBB generously (any call returning a row). NOT "
            "claimed: that more categories make better decisions; per-feed effectiveness needs "
            "paired model cycles and was not run. Keyed OpenBB providers are excluded and named."),
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    import sys

    args = sys.argv[1:] if argv is None else list(argv)
    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    notes = [json.loads(line) for line in NOTES_PATH.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if "--workbench" in args:
        perps = sorted(p for p, t in ANCHOR_OF.items() if t in raw["tickers"])
        artefact.write(WORKBENCH_PATH, read_workbench(perps))
    workbench = (json.loads(WORKBENCH_PATH.read_text(encoding="utf-8"))
                 if WORKBENCH_PATH.exists() else None)
    report = run(raw, notes, workbench)
    artefact.write(REPORT_PATH, report)
    s = report["summary"]
    print(f"{s['symbols']} symbols: ARGUS {s['mean_argus_categories']} categories per symbol, "
          f"OpenBB {s['mean_openbb_categories']}; ARGUS ahead on {s['symbols_argus_ahead']}, "
          f"OpenBB on {s['symbols_openbb_ahead']}, tied {s['symbols_tied']} ({s['verdict']})")
    w = report["workbench_summary"]
    if w is not None:
        print(f"workbench: {w['mean_workbench_categories']} categories per symbol; ahead on "
              f"{w['symbols_workbench_ahead']}, behind on {w['symbols_openbb_ahead']}, tied "
              f"{w['symbols_tied']} ({w['verdict']})")
    for cat, n in report["by_category"].items():
        print(f"  {cat:18} desk {n['argus']:2}  workbench {n['argus_workbench']:2}  "
              f"openbb {n['openbb']:2}")
    print(f"saved -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARGUS_SOURCES",
    "CATEGORIES",
    "OPENBB_CATEGORY",
    "BreadthError",
    "Cell",
    "argus_cells",
    "last_cycle",
    "openbb_cells",
    "per_symbol",
    "run",
]
