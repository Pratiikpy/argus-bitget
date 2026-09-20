"""Convert risk records from schema 1 to schema 2 — and refuse to guess where it cannot tell.

**Why a migration exists at all.** Schema 1 wrote ``binding_constraint: "none"`` for two different
outcomes of :meth:`argus.agents.desk.ConstitutionPolicy.rule`:

* **gate 1 returned** — the intent carried no quantity, so there was nothing to narrow, and
* **the terminal all-clear** — the intent reached and passed all seven gates.

`eval/autopsy.py` counted every ``"none"`` as gate 1 firing. On a record of the second kind that is
backwards: it decrements ``still_live``, so gates 2-7 report **UNREACHED on the exact decision that
proved them reachable**. Gate reachability is the one claim this project makes that a 787-repo
corpus does not, and the funnel inverted precisely when it finally had something to say.

Schema 2 gives gate 1 its own name, ``"no_exposure"``, exactly as every other gate emits its own,
leaving ``"none"`` to mean only *nothing bound*.

**Why the conversion is safe, and only now.** The two cases are separable after the fact because a
gate-1 record carries ``quantity_before == 0`` — there was no exposure, which is why gate 1
returned — while an all-clear record cannot. This module checks that discriminator on **every** row
rather than assuming it, and refuses the whole file if any row is ambiguous. As of 2026-09-14 all
125 rows on the live record carry ``quantity_before == 0`` and ``intervened: false``, so every one
is genuinely gate 1 and the conversion is lossless.

That window closes the first time a decision proposes exposure. This module is written to fail
loudly rather than silently on the far side of it.

``risk_records.jsonl`` is not hash-chained — it is kept beside the ledger precisely because an
:class:`~argus.paper.ledger.Entry` *is* the decision and its shape is hashed. Rewriting it in place
is therefore an ordinary file edit and breaks no proof; a backup is written regardless.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from argus.eval.autopsy import RECORD_SCHEMA

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "risk_records.jsonl"

GATE_ONE = "no_exposure"
NOTHING_BOUND = "none"


class MigrationError(RuntimeError):
    """Raised rather than writing a file this module cannot convert without guessing."""


@dataclass(frozen=True, slots=True)
class Migration:
    """What the conversion did, or would do."""

    path: Path
    total: int
    converted: int
    already: int
    untouched: int
    backup: Path | None

    def render(self) -> str:
        lines = [
            f"RISK RECORD MIGRATION — {self.path.name}",
            "",
            f"  rows              {self.total}",
            f"  converted         {self.converted}  (\"none\" -> \"{GATE_ONE}\")",
            f"  already schema {RECORD_SCHEMA}  {self.already}",
            f"  untouched         {self.untouched}  (a real gate name, or the all-clear)",
        ]
        if self.backup is not None:
            lines += ["", f"  backup written to {self.backup.name}"]
        return "\n".join(lines)


def _quantity(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError):
        raise MigrationError(
            f"quantity_before {value!r} is not a number, so this row cannot be classified"
        ) from None


def classify(record: dict[str, Any]) -> str:
    """The schema-2 constraint for one schema-1 record.

    Raises rather than guessing when the discriminator does not settle it.
    """
    value = record.get("binding_constraint")
    if value is None:
        raise MigrationError(
            f"row seq={record.get('seq')} has a null binding_constraint. A decision with no "
            f"Constitution ruling has no place in a reachability funnel; fix the writer rather "
            f"than inventing a value here."
        )
    if value != NOTHING_BOUND:
        return str(value)  # a named gate — unambiguous in both schemas

    if "quantity_before" not in record:
        raise MigrationError(
            f"row seq={record.get('seq')} says \"none\" and carries no quantity_before, so gate 1 "
            f"and the all-clear cannot be told apart. This module will not guess."
        )
    return GATE_ONE if _quantity(record["quantity_before"]) <= 0 else NOTHING_BOUND


def migrate(path: Path = DEFAULT_PATH, *, dry_run: bool = False) -> Migration:
    """Convert ``path`` in place, after checking every row can be classified without guessing."""
    if not path.exists():
        raise MigrationError(f"{path} does not exist")

    rows: list[dict[str, Any]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise MigrationError(f"{path.name} line {n} is not JSON: {exc}") from exc

    converted = already = untouched = 0
    out: list[dict[str, Any]] = []
    for record in rows:
        if int(record.get("chain_schema", 1)) == RECORD_SCHEMA:
            already += 1
            out.append(record)
            continue
        was = record.get("binding_constraint")
        now = classify(record)          # raises on anything ambiguous, before a byte is written
        record["chain_schema"] = RECORD_SCHEMA
        record["binding_constraint"] = now
        if was == NOTHING_BOUND and now == GATE_ONE:
            converted += 1
        else:
            untouched += 1
        out.append(record)

    backup: Path | None = None
    if not dry_run and (converted or untouched):
        backup = path.with_suffix(path.suffix + ".schema1.bak")
        shutil.copy2(path, backup)
        path.write_text(
            "".join(json.dumps(r, default=str) + "\n" for r in out), encoding="utf-8",
        )

    return Migration(
        path=path, total=len(rows), converted=converted, already=already,
        untouched=untouched, backup=backup,
    )


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Convert risk records to schema 2")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--dry-run", action="store_true", help="classify, change nothing")
    args = parser.parse_args()

    result = migrate(args.path, dry_run=args.dry_run)
    print(result.render())
    if args.dry_run:
        print("\n  DRY RUN — nothing written")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = ["GATE_ONE", "NOTHING_BOUND", "Migration", "MigrationError", "classify", "migrate"]
