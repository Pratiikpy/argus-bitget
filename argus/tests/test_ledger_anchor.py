"""The paper ledger's external commitment (`paper/anchor.py`, `ledger_digest`)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from argus.paper.anchor import ledger_digest
from argus.paper.ledger import PaperLedger

T0 = datetime(2026, 3, 9, 14, 0, tzinfo=UTC)


def _record(led: PaperLedger, n: int) -> None:
    led.record(
        symbol="NVDAUSDT", verdict="trade", side="BUY",
        quantity=Decimal("10"), entry_price=Decimal("220"),
        stated_confidence=0.7, thesis="guidance cut not priced",
        invalidation=("guidance reaffirmed",),
        market_state_hash="abc123", approved_intent_hash="def456",
        session_phase="weekend", hours_to_discovery=30.5,
        decided_at=T0 + timedelta(hours=n),
    )


def _first_rows_digest(path: Path, rows: int) -> str:
    """What a verifier computes from a later copy of the file: the first ``rows`` rows only."""
    led = PaperLedger(path=path)
    joined = "".join(f"{row.content_hash}\n" for row in led._raw_entries[:rows]).encode()
    return hashlib.sha256(joined).hexdigest()


def test_a_later_settlement_leaves_the_anchored_rows_digest_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "paper.jsonl"
    led = PaperLedger(path=path)
    _record(led, 0)
    _record(led, 1)
    digest, rows, _head = ledger_digest(path)
    assert rows == 2
    led.settle(1, exit_price=Decimal("230"), settled_at=T0 + timedelta(days=1))
    _record(led, 2)
    assert _first_rows_digest(path, rows) == digest.hex()


def test_rewriting_an_anchored_decision_changes_the_digest(tmp_path: Path) -> None:
    path = tmp_path / "paper.jsonl"
    led = PaperLedger(path=path)
    _record(led, 0)
    _record(led, 1)
    digest, rows, _head = ledger_digest(path)
    lines = path.read_text("utf-8").splitlines()
    first = json.loads(lines[0])
    first["thesis"] = "rewritten after the fact"
    path.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n", "utf-8")
    assert _first_rows_digest(path, rows) != digest.hex()
