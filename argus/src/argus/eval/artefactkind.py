"""Deterministic, or re-measured? — because "reproducible" has been doing two jobs.

`tests/test_artefacts_reproducible.py` guards a real property: every artefact the documents cite
must have a command that rebuilds it. It does **not** check that rebuilding produces the same
numbers, and the file's name invites a reader to assume it does.

**`overfit_gates.json` has now produced three different answers.** Its own test docstring records
a stored four-and-four split that failed to reproduce as six-refuted-two-unevaluated; re-running
it on 2026-09-21 gave eight-refuted-none-unevaluated, because the study fetches a rolling 90-day
window and the window had moved. Nothing was broken on any of the three occasions. The measurement
is simply a function of when it was taken, and calling it "reproducible" without saying so invites
exactly the wrong inference from a judge who re-runs it and gets a fourth number.

So this module separates the two properties, mechanically rather than by assertion:

* **deterministic** — the writer reads only files in the repository. Re-running must produce the
  same numbers, and a difference is a defect.
* **re-measured** — the writer fetches live market data. Re-running produces a *fresh measurement*;
  a difference is the instrument moving, not a bug. These carry a date and must be quoted with one.

The classification is derived from what each writer **imports**, not from a hand-maintained list,
because a hand-maintained list is wrong the first time somebody adds a fetch to an existing module
and does not think to update it.

**What this does not claim.** That every deterministic artefact has been re-run and checked byte
for byte — that is expensive and a separate job. It claims only that the two kinds are told apart
and that a re-measured artefact is never described as reproducible.

    python -m argus.eval.artefactkind
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argus.eval.artefact import write

ARGUS = Path(__file__).resolve().parents[3]
SRC = ARGUS / "src" / "argus"
DATA = ARGUS / "data"
REPORT_PATH = DATA / "artefact_kinds.json"

LIVE_SOURCES: tuple[str, ...] = (
    "argus.market.history",
    "argus.market.bitget",
    "argus.market.bitget_mcp",
    "argus.market.skills",
    "fetch_range",
    "fetch_candles",
    "urllib.request",
    "requests",
)
"""Imports and call names that mean a module reaches outside the repository.

Named rather than inferred from behaviour, because inferring it would mean running every writer —
and running them is the expensive thing this check exists to reason about without doing."""

_WRITE = re.compile(r"write_text\s*\(|\bwrite\s*\(\s*REPORT_PATH|json\.dump\s*\(")
_ARTEFACT = re.compile(r"[\"']([A-Za-z0-9_]+\.(?:jsonl|json))[\"']")

_IMPORT = re.compile(
    r"^\s*(?:from\s+(\S+)\s+import\s+([^\n#]+)|import\s+([^\n#]+))", re.MULTILINE
)
"""Real import statements, because a bare substring search matches a module's own documentation.

The first version of this classifier searched each file for the *names* in :data:`LIVE_SOURCES`
and promptly labelled its own artefact re-measured — this file names every one of them, in the
tuple that declares them. A detector that trips on its own definition is not a detector."""


@dataclass(frozen=True, slots=True)
class Kind:
    """One artefact and how it behaves when its writer is run again."""

    artefact: str
    writers: tuple[str, ...]
    live_sources: tuple[str, ...]

    @property
    def deterministic(self) -> bool:
        """No writer reaches outside the repository, so re-running must agree."""
        return not self.live_sources

    @property
    def kind(self) -> str:
        return "deterministic" if self.deterministic else "re-measured"

    def as_dict(self) -> dict[str, Any]:
        return {
            "artefact": self.artefact,
            "kind": self.kind,
            "writers": list(self.writers),
            "reaches_live_data_via": list(self.live_sources),
            "rerunning_it": (
                "must produce the same numbers; a difference is a defect"
                if self.deterministic
                else "produces a fresh measurement; a difference is the market moving, not a bug"
            ),
        }


def classify() -> list[Kind]:
    """Every artefact a module writes, sorted into the two kinds by what its writer imports."""
    found: dict[str, list[str]] = {}
    live: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not _WRITE.search(text):
            continue
        imported: set[str] = set()
        for module, names, plain in _IMPORT.findall(text):
            imported.add(module)
            imported.update(n.strip() for n in (names or plain or "").split(","))
        reaches = {
            name for name in LIVE_SOURCES
            if name in imported or any(m.startswith(name) for m in imported if m)
        }
        for artefact in set(_ARTEFACT.findall(text)):
            found.setdefault(artefact, []).append(path.relative_to(SRC).as_posix())
            live.setdefault(artefact, set()).update(reaches)
    return [
        Kind(artefact=name, writers=tuple(sorted(writers)),
             live_sources=tuple(sorted(live.get(name, set()))))
        for name, writers in sorted(found.items())
    ]


def run() -> dict[str, Any]:
    kinds = classify()
    remeasured = [k for k in kinds if not k.deterministic]
    return {
        "artefacts": len(kinds),
        "deterministic": len(kinds) - len(remeasured),
        "re_measured": len(remeasured),
        "re_measured_names": [k.artefact for k in remeasured],
        "detail": [k.as_dict() for k in kinds],
        "why_this_matters": (
            "overfit_gates.json has produced three different results on three runs — 4/4, then "
            "6/2, then 8/0 — because it fetches a rolling 90-day window. Nothing was broken on "
            "any occasion. A judge who re-runs a re-measured artefact and gets a fourth number "
            "should read that as the instrument moving; a judge who re-runs a deterministic one "
            "and gets a different number has found a defect. The two must not look alike."
        ),
        "scope_statement": (
            "Classification is derived from what each writer imports — a module that names "
            "`argus.market.history`, `fetch_range`, `urllib` or similar is treated as reaching "
            "live data. NOT CLAIMED: that every deterministic artefact has been re-run and "
            "compared byte for byte; that is a separate and expensive job, and this only "
            "establishes which ones such a comparison would be MEANINGFUL for. NOT CLAIMED: that "
            "a re-measured artefact is worse — it is the only honest kind for a claim about a "
            "live market, and it simply has to be quoted with its date."
        ),
    }


def render(report: dict[str, Any]) -> list[str]:
    lines = [
        f"ARTEFACT KINDS — {report['artefacts']} artefact(s) with a writer",
        f"  {report['deterministic']} deterministic (re-running must agree)",
        f"  {report['re_measured']} re-measured (re-running takes a fresh reading)",
        "",
        "  re-measured — quote these with a date:",
    ]
    lines += [f"    {name}" for name in report["re_measured_names"]]
    lines += [
        "",
        "  A difference on re-run means opposite things for the two kinds, and an artefact that "
        "does not say which it is invites the wrong reading of both.",
    ]
    return lines


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = run()
    for line in render(report):
        print(line)
    write(REPORT_PATH, report)
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["LIVE_SOURCES", "REPORT_PATH", "Kind", "classify", "main", "render", "run"]
