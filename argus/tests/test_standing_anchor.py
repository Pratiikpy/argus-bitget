"""The capability register is timestamped once per distinct register, and anyone can recompute
the digest from the published file (research/harvest/20-slsa.md)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.eval.standing import anchor_register, audit, register_digest, write_report

NOW = datetime(2026, 9, 28, tzinfo=UTC)


@dataclass
class Receipt:
    calendar: str


@dataclass
class Proof:
    anchored: bool
    receipts: tuple[Receipt, ...] = ()
    failures: tuple[str, ...] = ()


@dataclass
class Calendars:
    """Stands in for `paper/anchor.anchor`, recording what it was asked to stamp."""

    refuse: bool = False
    asked: list[str] = field(default_factory=list)

    def __call__(self, digest: str, *, subject: str = "") -> Proof:
        self.asked.append(digest)
        if self.refuse:
            return Proof(anchored=False, failures=("a.pool: timed out",))
        return Proof(anchored=True, receipts=(Receipt("a.pool"), Receipt("b.pool")))


def test_the_digest_is_recomputable_from_the_published_file(tmp_path: Path) -> None:
    report = audit()
    published = tmp_path / "standing.json"
    write_report(published, report=report)
    blob: dict[str, Any] = json.loads(published.read_text(encoding="utf-8"))
    for key in ("transitions_this_write", "transition_log"):
        blob.pop(key)
    canonical = json.dumps(blob, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == register_digest(report)


def test_an_unchanged_register_is_anchored_once(tmp_path: Path) -> None:
    report, calendars = audit(), Calendars()
    record = tmp_path / "standing_anchor.json"
    first = anchor_register(report, path=record, submit=calendars, now=NOW)
    again = anchor_register(report, path=record, submit=calendars, now=NOW)
    assert first["anchored"] and first["calendars"] == ["a.pool", "b.pool"]
    assert again == first and calendars.asked == [register_digest(report)]
    assert len(json.loads(record.read_text(encoding="utf-8"))) == 1


def test_a_refused_submission_is_recorded_and_retried(tmp_path: Path) -> None:
    report = audit()
    record = tmp_path / "standing_anchor.json"
    failed = anchor_register(report, path=record, submit=Calendars(refuse=True), now=NOW)
    assert not failed["anchored"] and failed["failures"] == ["a.pool: timed out"]
    retried = anchor_register(report, path=record, submit=Calendars(), now=NOW)
    assert retried["anchored"]
    assert [row["anchored"] for row in json.loads(record.read_text(encoding="utf-8"))] == [
        False, True]
