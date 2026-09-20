"""Count what the anchor proofs actually attest, by reading them rather than by trusting a sentence.

**This module exists because the documents claimed something the files did not support.** Until
2026-09-20 the register's proofs were described as "anchored to four independent Bitcoin calendars"
and verifiable "with the reference OpenTimestamps client". Two things were wrong with that:

* the ``.ots`` files held the **raw calendar receipt**, not a wrapped ``DetachedTimestampFile``, so
  the reference client rejected all 48 on the header magic before reading any proof; and
* every one carried only a :class:`PendingAttestation` — a calendar's *promise* to anchor — and not
  a single :class:`BitcoinBlockHeaderAttestation`. Submitted is not anchored.

Both were fixed at the source: the proofs were upgraded against their calendars and rewritten in
the standard format. But a fix that nothing re-checks is a fix with a shelf life, and this project's
entire claim is that its numbers are checked. So the count is computed here, from the bytes, and
registered in ``eval/docclaims.py`` — if a future anchor is written in the wrong format, or the
documents overstate how many are confirmed, the documentation gate fails.

Deliberately reads with the ``opentimestamps`` core library and nothing else. The official ``ots``
client cannot run on this machine — ``python-bitcoinlib`` needs libsecp256k1, which has no Windows
wheel — and a check that cannot run where the work happens is not a check. The core library is pure
Python and parses the same bytes the reference client does, so a file that passes here is a file
``ots verify`` will read.

**Pending is not a failure.** A Bitcoin attestation takes hours to exist. A proof submitted this
cycle that reads *pending* is telling the truth; it becomes a defect only when a document counts it
as confirmed.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# `opentimestamps` ships no py.typed marker, so these three are untyped at the vendor rather than
# here. Ignored narrowly, by code, at the import sites — not silenced project-wide in the mypy
# config, which would also hide a future untyped import somebody adds by accident.
from opentimestamps.core.notary import (  # type: ignore[import-untyped]
    BitcoinBlockHeaderAttestation,
    PendingAttestation,
)
from opentimestamps.core.serialize import (  # type: ignore[import-untyped]
    StreamDeserializationContext,
)
from opentimestamps.core.timestamp import (  # type: ignore[import-untyped]
    DetachedTimestampFile,
    Timestamp,
)

ANCHOR_DIR = Path(__file__).resolve().parents[3] / "data" / "anchors"
REPORT_PATH = ANCHOR_DIR.parent / "anchor_check.json"


def _walk(timestamp: Timestamp) -> Iterator[Timestamp]:
    """Every node in the proof tree. Attestations hang off interior nodes, not only the root."""
    yield timestamp
    for child in timestamp.ops.values():
        yield from _walk(child)


@dataclass(frozen=True)
class Anchor:
    path: str
    readable: bool
    """Whether the reference OpenTimestamps client could parse this file at all.

    The question the old format failed. A proof nobody else can open is not evidence."""

    bitcoin_blocks: tuple[int, ...]
    pending_calendars: tuple[str, ...]
    error: str = ""

    @property
    def confirmed(self) -> bool:
        return bool(self.bitcoin_blocks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "readable": self.readable,
            "bitcoin_blocks": list(self.bitcoin_blocks),
            "pending_calendars": list(self.pending_calendars),
            "error": self.error,
        }


def read_anchor(path: Path) -> Anchor:
    try:
        with path.open("rb") as handle:
            detached = DetachedTimestampFile.deserialize(StreamDeserializationContext(handle))
    except Exception as exc:  # any parse failure is the same finding: nobody else can read it
        return Anchor(path.name, readable=False, bitcoin_blocks=(), pending_calendars=(),
                      error=f"{type(exc).__name__}: {exc}")
    blocks: list[int] = []
    calendars: list[str] = []
    for node in _walk(detached.timestamp):
        for attestation in node.attestations:
            if isinstance(attestation, BitcoinBlockHeaderAttestation):
                blocks.append(attestation.height)
            elif isinstance(attestation, PendingAttestation):
                uri = attestation.uri
                calendars.append(uri.decode() if isinstance(uri, bytes) else str(uri))
    return Anchor(path.name, readable=True, bitcoin_blocks=tuple(sorted(set(blocks))),
                  pending_calendars=tuple(sorted(set(calendars))))


@dataclass(frozen=True)
class AnchorReport:
    anchors: tuple[Anchor, ...]

    @property
    def total(self) -> int:
        return len(self.anchors)

    @property
    def readable(self) -> int:
        return sum(1 for a in self.anchors if a.readable)

    @property
    def confirmed(self) -> int:
        """Proofs carrying a Bitcoin block-header attestation.

        The only ones the word *anchored* is true of. A pending attestation is a calendar's
        promise, and a promise is what the documents were quoting before 2026-09-20."""
        return sum(1 for a in self.anchors if a.confirmed)

    @property
    def pending(self) -> int:
        return sum(1 for a in self.anchors if a.readable and not a.confirmed)

    @property
    def unreadable(self) -> int:
        return self.total - self.readable

    @property
    def block_range(self) -> tuple[int, int] | None:
        blocks = [b for a in self.anchors for b in a.bitcoin_blocks]
        return (min(blocks), max(blocks)) if blocks else None

    @property
    def sound(self) -> bool:
        """Every proof parses. Pending is fine; unreadable is not.

        Deliberately not `confirmed == total`: requiring that would make the check fail every time
        a fresh anchor is written, which trains a reader to ignore it."""
        return self.unreadable == 0 and self.total > 0

    def as_dict(self) -> dict[str, Any]:
        low_high = self.block_range
        return {
            "total": self.total,
            "readable": self.readable,
            "unreadable": self.unreadable,
            "confirmed_on_bitcoin": self.confirmed,
            "calendar_pending": self.pending,
            "bitcoin_block_range": list(low_high) if low_high else None,
            "sound": self.sound,
            "anchors": [a.as_dict() for a in self.anchors],
        }

    def render(self) -> list[str]:
        lines = [
            f"ANCHORS — {self.total} proof(s) in data/anchors/",
            f"  readable by the reference OpenTimestamps client : {self.readable}",
            f"  carrying a Bitcoin block-header attestation     : {self.confirmed}",
            f"  still calendar-pending                          : {self.pending}",
        ]
        low_high = self.block_range
        if low_high:
            lines.append(f"  bitcoin blocks {low_high[0]:,} .. {low_high[1]:,}")
        if self.unreadable:
            lines.append(
                f"  ⚠ {self.unreadable} proof(s) DO NOT PARSE — they cannot be verified by anyone "
                f"but us, which is the same as not being evidence"
            )
            lines += [f"      {a.path}: {a.error}" for a in self.anchors if not a.readable]
        if self.pending:
            lines.append(
                "  pending is not a defect: a Bitcoin attestation takes hours to exist, and a "
                "proof that says pending until then is telling the truth"
            )
        return lines


def check(directory: Path = ANCHOR_DIR) -> AnchorReport:
    return AnchorReport(tuple(read_anchor(p) for p in sorted(directory.glob("*.ots"))))


def confirmed_anchors() -> int:
    """Proofs confirmed on Bitcoin. Registered as a documentation claim."""
    return check().confirmed


def readable_anchors() -> int:
    """Proofs the reference client can read. Was 0 of 48 before 2026-09-20."""
    return check().readable


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = check()
    for line in report.render():
        print(line)
    REPORT_PATH.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    print(f"\nwritten to {REPORT_PATH}")
    return 0 if report.sound else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
