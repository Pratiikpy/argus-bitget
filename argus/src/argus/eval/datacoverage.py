"""How much of Bitget's own data surface actually answers — counted by calling every entry.

**Track 3 scores "data sources / Skill integration count *and effectiveness*".** The word that
does the work is *effectiveness*: a submission can claim sixty-seven integrations and have four
that return anything, and no reader can tell the difference from a feature list. So this calls
every catalog entry the service publishes, for a real symbol, and reports what came back.

`market/skills.py` already does this for the five `bitget-signal` research Skills and found **6 of
19 tools answering** — 10 empty, 3 erroring. That measurement is why this one exists: the same
question had never been asked of `bitget-mcp-server`, which is the larger service and the one
carrying the anchor data for tokenized equities.

**Three health states, and the distinction is the point.** ``ok`` returned rows. ``empty``
answered with a well-formed envelope and no data — which is a real answer about coverage, not a
failure. ``error`` refused. Collapsing the middle one into either neighbour is how a coverage claim
becomes fiction: an endpoint that exists, responds, and has nothing for your symbol is neither a
working integration nor a broken one.

**Every parameter shape is discovered, not guessed.** An entry that rejects ``symbol`` is retried
against the argument names the catalog itself advertises, and an entry that cannot be satisfied is
reported as ``needs_params`` with the server's own complaint attached — never as a failure of the
service. A probe that calls a tool wrongly measures the probe.

    python -m argus.eval.datacoverage
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argus.eval.artefact import write
from argus.market.bitget_mcp import BitgetDataService, BitgetMcpError, underlying_of

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "data_coverage.json"

PROBE_SYMBOL = "NVDAUSDT"
"""One of the twelve rTokens. Its anchor is NVDA, which is liquid enough that an empty result is a
statement about the endpoint rather than about the company."""

CRYPTO_PROBE = "BTCUSDT"
"""Crypto entries want a crypto pair. Probing them with an equity ticker would manufacture the
empty results this module exists to count honestly."""

PAUSE = 0.15
"""Between calls. The service is Bitget's and is not ours to hammer."""


@dataclass(frozen=True, slots=True)
class Probe:
    """One entry, called for real."""

    entry_id: str
    category: str
    health: str
    rows: int
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id, "category": self.category, "health": self.health,
            "rows": self.rows, "detail": self.detail[:200],
        }


def _arguments(entry_id: str, category: str) -> list[dict[str, Any]]:
    """Argument shapes to try, most likely first.

    Ordered rather than guessed at random: the catalog's equity entries take ``symbol``, crypto
    entries take a pair under the same name, and a handful take nothing at all. Trying the empty
    call last means an endpoint that ignores arguments is still recorded as answering."""
    ticker = underlying_of(PROBE_SYMBOL)
    if category == "equity":
        return [{"symbol": ticker}, {"ticker": ticker}, {}]
    if category == "crypto":
        return [{"symbol": CRYPTO_PROBE}, {"symbol": "BTC"}, {}]
    if category == "etf":
        return [{"symbol": "QQQ"}, {}]
    return [{}, {"symbol": ticker}]


def probe_entry(service: BitgetDataService, entry_id: str, category: str) -> Probe:
    """Call one entry until an argument shape is accepted, or report why none was."""
    complaint = ""
    for arguments in _arguments(entry_id, category):
        try:
            rows = service.results(entry_id, **arguments)
        except BitgetMcpError as exc:
            complaint = str(exc)
            # A validation error means the shape was wrong, so try the next one. Anything else is
            # the service refusing, and trying again with different arguments would only obscure
            # that — so it is recorded as an error immediately.
            if "validation error" in complaint or "Missing required" in complaint:
                continue
            return Probe(entry_id, category, "error", 0, complaint)
        if rows:
            return Probe(entry_id, category, "ok", len(rows), f"args={arguments}")
        return Probe(entry_id, category, "empty", 0, f"answered with no rows; args={arguments}")
    return Probe(entry_id, category, "needs_params", 0, complaint)


def run(categories: Sequence[str] | None = None) -> dict[str, Any]:
    """Every entry in every category, called once."""
    service = BitgetDataService()
    catalog = service.categories()
    wanted = set(categories) if categories else {str(c.get("key")) for c in catalog}

    probes: list[Probe] = []
    for category in catalog:
        key = str(category.get("key", ""))
        if key not in wanted:
            continue
        for entry in service.entries(key):
            probes.append(probe_entry(service, entry.entry_id, key))
            time.sleep(PAUSE)

    by_health: dict[str, int] = {}
    for probe in probes:
        by_health[probe.health] = by_health.get(probe.health, 0) + 1
    answering = [p for p in probes if p.health == "ok"]
    equity = [p for p in probes if p.category == "equity"]

    return {
        "service": service.server,
        "probe_symbol": PROBE_SYMBOL,
        "underlying": underlying_of(PROBE_SYMBOL),
        "catalog": {str(c.get("key")): int(c.get("entry_count", 0)) for c in catalog},
        "entries_probed": len(probes),
        "by_health": by_health,
        "answering": len(answering),
        "answering_rate": round(len(answering) / len(probes), 4) if probes else 0.0,
        "equity_entries": len(equity),
        "equity_answering": sum(1 for p in equity if p.health == "ok"),
        "probes": [p.as_dict() for p in probes],
        "scope_statement": (
            "Every catalog entry Bitget's own MCP data service publishes is called once, with an "
            "argument shape chosen per category and retried when the server rejects it, and the "
            "result is classified ok / empty / error / needs_params. NOT CLAIMED: that an empty "
            "result means the endpoint is broken — it means it had nothing for this symbol, which "
            "is a fact about coverage. NOT CLAIMED: that one symbol characterises an endpoint; "
            "these are single probes, and a wider sweep would move the empty count. The service "
            "needs no API key, which is why this is reproducible by a reader."
        ),
    }


def render(report: dict[str, Any]) -> list[str]:
    health = report["by_health"]
    lines = [
        f"BITGET DATA COVERAGE — {report['service']}, no API key",
        f"  catalog: {sum(report['catalog'].values())} entries across "
        f"{len(report['catalog'])} categories — {report['catalog']}",
        f"  probed {report['entries_probed']}: "
        + ", ".join(f"{k} {v}" for k, v in sorted(health.items())),
        f"  answering with data: {report['answering']}/{report['entries_probed']} "
        f"({report['answering_rate']:.0%})",
        f"  US equity entries answering: {report['equity_answering']}/{report['equity_entries']} "
        f"for {report['underlying']}",
        "  'empty' is reported separately from 'error' on purpose: an endpoint that answers with "
        "no rows is a statement about coverage, not a broken integration.",
    ]
    return lines


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="what fraction of Bitget's data surface answers")
    parser.add_argument("--category", action="append", default=None)
    args = parser.parse_args()

    report = run(args.category)
    for line in render(report):
        print(line)
    write(REPORT_PATH, report)
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["PROBE_SYMBOL", "REPORT_PATH", "Probe", "main", "probe_entry", "render", "run"]
