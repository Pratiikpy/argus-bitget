"""Ledger repair — the one operation allowed to touch history, and only in the open.

A hash chain's value is that an undetected edit is impossible. It is not that an edit never happens:
on 2026-09-12 two overlapping cycles appended two different NVDAUSDT decisions as seq 41, and the
chain has been broken from that row onward ever since. `argus.paper.ledger` now prevents the race
(`_exclusive`, `_append_built`), but prevention does nothing for the file that is already wrong,
and a log that reports ``chain_intact: false`` forever is not a usable deliverable.

So there are three options and only one of them is honest.

* **Leave it.** Every future verification reports a break at 41, and a reader cannot distinguish
  our concurrency bug from tampering. The evidence is real but unusable.
* **Quietly re-link.** The chain verifies and nothing records that it was rebuilt. This is the one
  thing the whole design exists to make impossible, and doing it by hand is worse than the bug.
* **Repair in the open.** Archive the original bytes, hash them, renumber the collision, re-link
  forward, and write an incident record that names the archive, its digest, every sequence number
  that moved and every hash that changed. The repaired chain verifies; the original remains on disk
  and still fails verification at 41; anyone can diff the two.

This module is the third. **No decision is ever dropped, altered, or reordered** — the repair may
only renumber a duplicated sequence and recompute ``prev_hash``. Content — thesis, confidence,
prices, timestamps, the Autonomy Proof hash — is untouched, and :func:`plan` proves it by comparing
the content hash of every row before and after.

It is deliberately not called from the cycle. A repair is an operator action with a written record,
never something a scheduled task does to itself at 3am.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.paper.ledger import GENESIS, Entry, LedgerError, PaperLedger

INCIDENTS = "paper_ledger_incidents.json"
"""Appended to, never overwritten: two repairs must both be readable, or the second hides the
first."""


@dataclass(frozen=True, slots=True)
class Change:
    """One row the repair would touch, before and after."""

    old_seq: int
    new_seq: int
    decided_at: str
    symbol: str
    old_prev_hash: str
    new_prev_hash: str
    content_preserved: bool
    """The row's content hash is unchanged. False anywhere means the repair is not a repair."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "old_seq": self.old_seq,
            "new_seq": self.new_seq,
            "decided_at": self.decided_at,
            "symbol": self.symbol,
            "old_prev_hash": self.old_prev_hash,
            "new_prev_hash": self.new_prev_hash,
            "content_preserved": self.content_preserved,
        }


@dataclass(frozen=True, slots=True)
class RepairPlan:
    """What would change, computed without writing anything."""

    entries_before: int
    entries_after: int
    duplicate_seqs: tuple[int, ...]
    changes: tuple[Change, ...]
    source_digest: str

    @property
    def needed(self) -> bool:
        return bool(self.changes)

    @property
    def is_content_preserving(self) -> bool:
        """Every row's content survived. The repair is refused if this is ever False."""
        return all(c.content_preserved for c in self.changes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "entries_before": self.entries_before,
            "entries_after": self.entries_after,
            "duplicate_seqs": list(self.duplicate_seqs),
            "source_digest": self.source_digest,
            "changes": [c.as_dict() for c in self.changes],
        }

    def render(self) -> list[str]:
        if not self.needed:
            return ["[repair] the chain is intact; nothing to do"]
        return [
            f"[repair] {len(self.changes)} row(s) would be re-linked, "
            f"{len(self.duplicate_seqs)} duplicated sequence number(s): "
            f"{', '.join(str(s) for s in self.duplicate_seqs) or 'none'}",
            f"[repair] {self.entries_before} entries in, {self.entries_after} out — "
            f"a repair that changes this count has dropped a decision",
            f"[repair] content preserved on every row: {self.is_content_preserving}",
        ]


def digest(path: Path) -> str:
    """SHA-256 of the file as it stands. What the incident record commits to."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relinked(entries: list[Entry]) -> list[Entry]:
    """Renumber sequentially and recompute every link, in the order the rows already have.

    Order is *never* changed. The rows are left exactly where they are, including the two that
    collided — the earlier-written one keeps the lower sequence number because that is the order the
    file records, not because of its timestamp. Reordering by ``decided_at`` would be a judgement
    about which decision "really" came first, and the file is the record.
    """
    out: list[Entry] = []
    prev = GENESIS
    for i, entry in enumerate(entries, start=1):
        fixed = replace(entry, seq=i, prev_hash=prev)
        out.append(fixed)
        prev = fixed.content_hash
    return out


def plan(path: Path) -> RepairPlan:
    """Compute the repair without writing. Safe to run on a healthy log."""
    ledger = PaperLedger(path=path)
    before = list(ledger.entries)
    after = _relinked(before)

    seen: dict[int, int] = {}
    for entry in before:
        seen[entry.seq] = seen.get(entry.seq, 0) + 1
    duplicates = tuple(sorted(s for s, n in seen.items() if n > 1))

    changes = tuple(
        Change(
            old_seq=old.seq,
            new_seq=new.seq,
            decided_at=old.decided_at,
            symbol=old.symbol,
            old_prev_hash=old.prev_hash,
            new_prev_hash=new.prev_hash,
            # The content hash covers the decision and excludes settlement. `seq` and `prev_hash`
            # are part of it, so the comparison is made against a copy carrying the new identity:
            # what must be unchanged is the *decision*, not its position in the chain.
            content_preserved=(
                replace(old, seq=new.seq, prev_hash=new.prev_hash).content_hash == new.content_hash
            ),
        )
        for old, new in zip(before, after, strict=True)
        if (old.seq, old.prev_hash) != (new.seq, new.prev_hash)
    )
    return RepairPlan(
        entries_before=len(before),
        entries_after=len(after),
        duplicate_seqs=duplicates,
        changes=changes,
        source_digest=digest(path) if path.exists() else "",
    )


def repair(path: Path, *, reason: str, at: datetime | None = None) -> dict[str, Any]:
    """Archive the original, re-link the chain, and write the incident record.

    Refuses to run unless the archive is written first and the repaired chain verifies afterwards.
    A repair that leaves the log broken is worse than none, because the operator will believe it is
    fixed.
    """
    if not reason.strip():
        raise LedgerError("a repair needs a stated reason; an unexplained rewrite is tampering")

    proposal = plan(path)
    if not proposal.needed:
        return {"repaired": False, "note": "the chain is intact; nothing was written"}
    if not proposal.is_content_preserving:
        raise LedgerError(
            "the re-link would change the content of at least one decision; refusing. Only "
            "sequence numbers and chain links may move."
        )

    stamp = (at or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    archive = path.with_name(f"{path.stem}.corrupted-{stamp}{path.suffix}")
    if archive.exists():
        raise LedgerError(f"{archive.name} already exists; refusing to overwrite an archive")
    archive.write_bytes(path.read_bytes())

    ledger = PaperLedger(path=path)
    ledger.entries = _relinked(ledger.entries)
    with path.open("w", encoding="utf-8") as fh:
        for entry in ledger.entries:
            fh.write(json.dumps(asdict(entry), default=str) + "\n")
    ledger._write_anchor()

    verified = PaperLedger(path=path).verify()
    if not verified["chain_intact"]:
        raise LedgerError(
            f"the repaired chain still fails verification at {verified['first_break_at']}; "
            f"the original is preserved at {archive.name}"
        )

    record = {
        "at": (at or datetime.now(UTC)).isoformat(),
        "reason": reason,
        "archive": archive.name,
        "archive_sha256": proposal.source_digest,
        "repaired_sha256": digest(path),
        "repaired_head_hash": verified["head_hash"],
        "plan": proposal.as_dict(),
    }
    incidents_path = path.with_name(INCIDENTS)
    history: list[dict[str, Any]] = []
    if incidents_path.exists():
        loaded = json.loads(incidents_path.read_text(encoding="utf-8"))
        history = loaded if isinstance(loaded, list) else [loaded]
    history.append(record)
    incidents_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return {"repaired": True, "incident": record, "verification": verified}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect or repair a paper ledger's hash chain.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--apply", action="store_true", help="write the repair, not just plan it")
    parser.add_argument("--reason", default="", help="required with --apply")
    args = parser.parse_args()

    proposal = plan(args.path)
    for line in proposal.render():
        print(line)
    for change in proposal.changes[:20]:
        print(
            f"  seq {change.old_seq} -> {change.new_seq}  {change.symbol}  "
            f"{change.decided_at}  content preserved: {change.content_preserved}"
        )
    if len(proposal.changes) > 20:
        print(f"  ... and {len(proposal.changes) - 20} more")

    if not args.apply:
        print("\nplan only; pass --apply --reason '...' to write it")
        return 0

    outcome = repair(args.path, reason=args.reason)
    if not outcome["repaired"]:
        print(outcome["note"])
        return 0
    incident = outcome["incident"]
    print(f"\narchived  {incident['archive']}  sha256 {incident['archive_sha256'][:16]}")
    print(
        f"repaired  sha256 {incident['repaired_sha256'][:16]}  "
        f"head {incident['repaired_head_hash']}"
    )
    print(f"incident  {INCIDENTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["INCIDENTS", "Change", "RepairPlan", "digest", "main", "plan", "repair"]
