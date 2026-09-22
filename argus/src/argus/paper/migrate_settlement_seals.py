"""Backfill settlement seals for decisions settled before the seal mechanism existed.

**Why this exists.** `paper/ledger.py::Entry.content_hash` deliberately excludes settlement fields
(`net_pnl`, `direction_correct`, ...) so that attaching an outcome is not indistinguishable from
tampering with the decision itself — correct, and documented at `content_hash`'s own docstring.
Adversarial testing on 2026-09-22 found the real gap that left open: nothing else protected those
fields either. A copy of the live ledger was tampered with directly on disk, flipping a settled
decision's `net_pnl` sign, and `PaperLedger.verify()` still reported `chain_intact: True` — a real,
working exploit against the exact claim the console's own landing page invites a judge to test.

The fix (`Entry.settlement_content_hash`, `PaperLedger._seal_settlement`, and the
`"settlement_seal"` row kind) is wired into `settle()`/`settle_abstention()` for every future
decision. It does nothing for a decision that was already settled before this code existed —
this module closes that, one time, for the real committed ledger.

**The honest limit, stated before the result.** A seal backfilled today proves the sealed decision's
outcome has not changed **since today** — it cannot prove the outcome was never touched between its
real settlement date and now, because nothing committed to it at settlement time in the first place.
That is a real, permanent gap for the pre-2026-09-22 rows specifically, and it is the reason this
module exists rather than pretending a retroactive seal is equivalent to a contemporaneous one.
Every decision settled from 2026-09-22 onward gets the contemporaneous guarantee automatically.

    python -m argus.paper.migrate_settlement_seals [--dry-run]
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.paper.ledger import PaperLedger

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "paper_ledger.jsonl"


class MigrationError(RuntimeError):
    """Raised rather than writing a seal this module cannot honestly attribute."""


@dataclass(frozen=True, slots=True)
class Migration:
    """What the backfill did, or would do."""

    path: Path
    total_decisions: int
    already_sealed: int
    backfilled: tuple[int, ...]
    backup: Path | None

    def render(self) -> str:
        lines = [
            f"SETTLEMENT SEAL BACKFILL — {self.path.name}",
            "",
            f"  decisions on record   {self.total_decisions}",
            f"  already sealed        {self.already_sealed}",
            f"  backfilled this run   {len(self.backfilled)}",
        ]
        if self.backfilled:
            lines.append(
                "  **backfilled seals prove unchanged since backfill time, not since the "
                "decision's real settlement date** — see this module's own docstring"
            )
        if self.backup is not None:
            lines += ["", f"  backup written to {self.backup.name}"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "total_decisions": self.total_decisions,
            "already_sealed": self.already_sealed,
            "backfilled": list(self.backfilled),
            "backup": str(self.backup) if self.backup else None,
        }


def migrate(path: Path = DEFAULT_PATH, *, dry_run: bool = False) -> Migration:
    """Seal every already-settled decision on ``path`` that has no seal yet."""
    if not path.exists():
        raise MigrationError(f"{path} does not exist")

    before = PaperLedger(path=path)
    _tampered, unsealed = before._check_settlement_seals()
    if _tampered:
        raise MigrationError(
            f"refusing to backfill: {len(_tampered)} settled decision(s) already fail a "
            f"settlement-hash check before any seal exists for them — seq(s) {_tampered}. That "
            f"can only mean a decision already carries a seal whose hash does not match its "
            f"current fields, which this migration did not write and should not paper over."
        )

    settled_count = len([e for e in before.entries if e.is_settled])
    already_sealed = settled_count - len(unsealed)

    if not unsealed:
        return Migration(
            path=path, total_decisions=len(before.entries),
            already_sealed=already_sealed, backfilled=(), backup=None,
        )

    if dry_run:
        return Migration(
            path=path, total_decisions=len(before.entries),
            already_sealed=already_sealed, backfilled=tuple(unsealed), backup=None,
        )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.stem}.pre-seal-backfill-{stamp}{path.suffix}")
    if backup.exists():
        raise MigrationError(f"{backup.name} already exists; refusing to overwrite a backup")
    shutil.copy2(path, backup)

    ledger = PaperLedger(path=path)
    by_seq = {e.seq: e for e in ledger.entries}
    for seq in unsealed:
        ledger._seal_settlement(by_seq[seq])

    after = PaperLedger(path=path)
    report = after.verify()
    if not report["chain_intact"]:
        raise MigrationError(
            f"the ledger fails verification after backfilling: "
            f"tampered={report['tampered_settlements']} unsealed={report['unsealed_settlements']} "
            f"— the original is preserved at {backup.name}"
        )

    return Migration(
        path=path, total_decisions=len(after.entries),
        already_sealed=already_sealed, backfilled=tuple(unsealed), backup=backup,
    )


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    migration = migrate(args.path, dry_run=args.dry_run)
    print(migration.render())
    return 0


__all__ = ["DEFAULT_PATH", "Migration", "MigrationError", "migrate", "main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
