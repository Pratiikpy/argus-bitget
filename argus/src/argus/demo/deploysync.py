"""Refresh the deployed console from the working tree — a command, not a copy-paste.

**Why this exists.** The hosted console is a *bundle*: `deploy/api/argus/` is a copy of the package
and `deploy/data/` a copy of the artefacts, both produced by hand. On 2026-09-15 the bundled ledger
held **126 decisions against a live 219** — 43% of the record missing from the one surface a
judge is most likely to open — and there was no command to fix it, only a memory of having
copied files once.

That is the same defect as the orphaned artefacts, wearing a different hat: a thing that must be
regenerated, with nothing that regenerates it. The fix is the same. This module makes the refresh a
command, records what it copied, and **reports the staleness it found rather than silently repairing
it** — because how far behind the shop window had drifted is itself a fact worth seeing.

**What it deliberately does not do.**

* It does not deploy. Publishing is an outward-facing act and stays a human's decision; this
  prepares the bundle and stops.
* It does not copy `.secrets/`, `*.env`, or anything outside the two manifests below. A key in a
  public serverless bundle is a key that has been published.
* It does not invent an artefact that is missing from the working tree. A file absent upstream is
  reported as absent, never carried forward from the previous bundle — a stale copy surviving a
  "refresh" is how the 126 survived three days.
"""

from __future__ import annotations

import filecmp
import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARGUS = Path(__file__).resolve().parents[3]
WORKSPACE = ARGUS.parent
SOURCE_PACKAGE = ARGUS / "src" / "argus"
SOURCE_DATA = ARGUS / "data"
DEPLOY = WORKSPACE / "deploy"
DEPLOY_PACKAGE = DEPLOY / "api" / "argus"
DEPLOY_DATA = DEPLOY / "data"
MANIFEST_PATH = DEPLOY / "sync_manifest.json"

CONSOLE_ARTEFACTS: tuple[str, ...] = (
    "paper_ledger.jsonl",
    "research_report.json",
    "lui_router.json",
    "regime_comparison.json",
    "oblique_bench.json",
    "surface_agreement.json",
    "standing.json",
    "lui_ngram_model.json",
    # The research-kind model (`lui/kindmodel.py`): how the hosted console reads a question when no
    # language model is configured. Without it the console falls back to the patterns alone.
    "lui_kind_model.json",
    # Real Bitget hourly history, frozen on the date it records. `lui/research.py` answers from it
    # only when the live fetch cannot finish inside the serverless deadline, and says so in the
    # answer — without it, a slow venue would turn a research question into a refusal.
    "risk_layer_candles_fixture.json",
    # Bitget's contract list, frozen on the date it records. `market/universe.py` reads it only
    # when the live contracts call fails — without it an offline answer would not know gold trades.
    "venue_universe.json",
    # Read by the "what we got wrong" page: the Skill MACD check (`eval/skillmacd.py`).
    "skill_macd_check.json",
    # FRED series the console answers macro questions from when FRED is unreachable from the host.
    "macro_snapshot.json",
    # How the desk's refusals' leans graded — the track-record answer reads it.
    "refusal_alpha.json",
    # Read by the status page (`lui/status_page.py`): the dated sweeps of every Bitget toolkit
    # surface, shown beside the live checks.
    "bitget_skills_health.json",
    "skill_mirror.json",
    "lui_final_heldout_report.json",
    # X and Reddit crowd read (`market/social_pulse.py`), collected on the desk's machine.
    "social_pulse.json",
    "skill_reliability.json",
    "data_coverage.json",
    "source_health.json",
    # Read by the hedge and macro answers: the scheduled CPI and FOMC dates (`market/calendar.py`).
    "event_calendar.json",
    # Read by the sentiment answer: the coordinated-posting test (finBERT vs the desk's analyst).
    "sentiment_comparison.json",
    # Read by event questions: how each name reacted to CPI, Fed decisions and its earnings.
    "event_reactions.json",
    # Read by add-to-book questions: the post-trade beta's measured record against weekend-copilot.
    "copilot_rivals.json",
    "copilot_stress.json",
    # Read by execution questions: one-minute against hourly children on a full-depth replay.
    "execution_arena.json",
)
"""Artefacts the hosted console's own pages read at request time.

Listed by hand rather than discovered, because discovery would mean importing the page modules and
tracing their reads — and a sync tool that executes the thing it is packaging is a worse idea than
a list somebody has to maintain. A name here that the bundle lacks is reported, never auto-copied:
the rule that a new artefact reaches the public surface only on purpose is the point, and this
check exists so that a *deliberate* omission is distinguishable from a forgotten one."""

SKIP_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules"}
"""``node_modules`` added 2026-09-16: the crypto_sor comparison harness installs real npm
packages under `eval/baselines/crypto_sor_shim/` to run a real Node subprocess, and the first
sync after that harness existed silently copied all ~600 of those third-party files into the
public bundle before this line existed — reinstallable from the harness's own `package.json`,
never something a deploy bundle should carry."""
SKIP_SUFFIXES = {".pyc", ".pyo", ".env"}

NEVER_COPY = {".secrets", "secrets"}
"""Directory names that must never reach a public bundle, checked by name at every level."""


class SyncError(RuntimeError):
    """Raised rather than shipping a bundle this module cannot vouch for."""


@dataclass
class SyncResult:
    """What the refresh found and what it changed."""

    package_files: int = 0
    package_updated: int = 0
    data_files: int = 0
    data_updated: int = 0
    missing_upstream: list[str] = field(default_factory=list)
    needed_by_console: list[str] = field(default_factory=list)
    """Artefacts a console page reads that the bundle does not carry.

    Distinct from `missing_upstream`, which is the opposite direction: a file in the bundle whose
    source is gone. This one is a file the source has and the bundle needs."""
    ledger_before: int = 0
    ledger_after: int = 0
    dry_run: bool = False

    @property
    def ledger_gap(self) -> int:
        """How many decisions the bundle was missing before this ran."""
        return max(0, self.ledger_after - self.ledger_before)

    def as_dict(self) -> dict[str, Any]:
        return {
            "synced_at": datetime.now(UTC).isoformat(),
            "package_files": self.package_files,
            "package_updated": self.package_updated,
            "data_files": self.data_files,
            "data_updated": self.data_updated,
            "missing_upstream": self.missing_upstream,
            "needed_by_console": self.needed_by_console,
            "ledger_before": self.ledger_before,
            "ledger_after": self.ledger_after,
            "ledger_gap_closed": self.ledger_gap,
        }

    def render(self) -> str:
        lines = [
            "DEPLOY SYNC" + ("  (DRY RUN — nothing written)" if self.dry_run else ""),
            "",
            f"  package   {self.package_updated} of {self.package_files} file(s) refreshed",
            f"  artefacts {self.data_updated} of {self.data_files} file(s) refreshed",
            f"  ledger    {self.ledger_before} -> {self.ledger_after} decisions",
        ]
        if self.ledger_gap:
            lines.append(
                f"            the bundle was missing {self.ledger_gap} decision(s) "
                f"({self.ledger_gap / self.ledger_after:.0%} of the record)"
            )
        if self.needed_by_console:
            lines.append(
                f"  ** {len(self.needed_by_console)} artefact(s) a console page READS are not in "
                f"the bundle — those pages will render 'artefact unreadable' **"
            )
            lines += [f"      {name}" for name in self.needed_by_console]
            lines.append(
                "      Copy them into deploy/data deliberately; this tool will not widen the "
                "public surface on its own."
            )
        if self.missing_upstream:
            lines += ["", "  ⚠ present in the bundle but ABSENT upstream — not carried forward:"]
            lines += [f"      {name}" for name in self.missing_upstream]
        return "\n".join(lines)


def _count_lines(path: Path) -> int:
    """Despite the generic name, both call sites are `paper_ledger.jsonl` — the "decisions"
    figure this module prints. A raw line count was exactly that count until settlement seals
    (2026-09-22) added a second row kind; filtering `kind == "decision"` keeps it that way, the
    same fix `eval/docclaims.py::ledger_decisions` needed for the identical reason. A row missing
    the key entirely (every line written before that field existed) defaults to `"decision"`,
    matching `Entry`'s own dataclass default.
    """
    if not path.exists():
        return 0
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("kind", "decision") == "decision"
    )


def _copy_tree(source: Path, target: Path, *, dry_run: bool) -> tuple[int, int]:
    """Mirror ``source`` into ``target``. Returns (files seen, files changed)."""
    seen = changed = 0
    for item in sorted(source.rglob("*")):
        if item.is_dir():
            continue
        parts = set(item.relative_to(source).parts)
        if parts & SKIP_DIRS or parts & NEVER_COPY or item.suffix in SKIP_SUFFIXES:
            continue
        seen += 1
        destination = target / item.relative_to(source)
        if destination.exists() and filecmp.cmp(item, destination, shallow=False):
            continue
        changed += 1
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)
    return seen, changed


def sync(*, dry_run: bool = False) -> SyncResult:
    """Refresh the bundle from the working tree, and report what was behind."""
    if not DEPLOY.exists():
        raise SyncError(f"no deployment directory at {DEPLOY}")
    if not SOURCE_PACKAGE.exists():
        raise SyncError(f"no package to copy at {SOURCE_PACKAGE}")

    result = SyncResult(dry_run=dry_run)
    result.ledger_before = _count_lines(DEPLOY_DATA / "paper_ledger.jsonl")
    result.ledger_after = _count_lines(SOURCE_DATA / "paper_ledger.jsonl")

    result.package_files, result.package_updated = _copy_tree(
        SOURCE_PACKAGE, DEPLOY_PACKAGE, dry_run=dry_run
    )

    # The artefact list is whatever the bundle already carries: the deployment decides what it
    # serves, and this module refreshes exactly that rather than widening the public surface on its
    # own. A new artefact reaches the console by being added to deploy/data deliberately.
    if DEPLOY_DATA.exists():
        for item in sorted(DEPLOY_DATA.rglob("*")):
            if item.is_dir() or item.suffix in SKIP_SUFFIXES:
                continue
            relative = item.relative_to(DEPLOY_DATA)
            upstream = SOURCE_DATA / relative
            result.data_files += 1
            if not upstream.exists():
                result.missing_upstream.append(str(relative).replace("\\", "/"))
                continue
            if filecmp.cmp(upstream, item, shallow=False):
                continue
            result.data_updated += 1
            if not dry_run:
                item.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(upstream, item)

    # **What the console's own pages read, that the bundle does not carry.**
    #
    # The deliberate rule above — a new artefact reaches the bundle only by being added on purpose
    # — is the right posture and it has a blind spot this closes. It protects against *widening*
    # the public surface by accident; it says nothing about a page that ships and then degrades.
    #
    # `/wrong` is built to survive a missing artefact by printing "artefact unreadable" instead of
    # silently shortening. That is correct behaviour and it looked exactly like a broken page:
    # four of its six findings rendered as unreadable from the bundle, because the artefacts had
    # never been seeded. The page did its job and nobody was told. Same trap as the n-gram model
    # a few days earlier, which is twice.
    result.needed_by_console = sorted(
        name for name in CONSOLE_ARTEFACTS
        if not (DEPLOY_DATA / name).is_file() and (SOURCE_DATA / name).is_file()
    )

    if not dry_run:
        MANIFEST_PATH.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    return result


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="refresh the deployed console bundle")
    parser.add_argument("--dry-run", action="store_true", help="report the gap, change nothing")
    args = parser.parse_args()

    result = sync(dry_run=args.dry_run)
    print(result.render())
    if not args.dry_run:
        print(f"\n  manifest -> {MANIFEST_PATH}")
        print("  NOT deployed — publishing stays a human's decision.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "DEPLOY",
    "DEPLOY_DATA",
    "DEPLOY_PACKAGE",
    "MANIFEST_PATH",
    "SyncError",
    "SyncResult",
    "main",
    "sync",
]
