"""Ledger rows known to be wrong, and why — recorded rather than deleted.

**A hash chain proves a record was not altered. It does not prove the record was right.** Those
are different guarantees, and conflating them is how a tamper-evident log becomes a way to launder
a mistake: the chain still verifies, so the wrong number looks authenticated.

On 2026-09-20 an adversarial pre-submission audit found that `paper/runner.py` selected the intent
to record as ``llm_revised_intent or llm_original_intent``. That fallback bypasses the Constitution
entirely and is reached on exactly the decisions the desk chooses **not** to re-put to the model —
the common case, since it deliberately does not buy a second opinion when nothing bound. For two
live decisions the risk layer refused the position and the ledger booked it anyway.

The tempting fix is to delete two rows. That is refused here for the same reason the desk refuses
a trade it cannot justify: it would make the record look better than the system behaved, and the
deletion would be invisible to exactly the person the chain exists to protect. **The rows stay.
They are listed here, with the evidence, and every consumer of the ledger excludes them.**

The underlying defect is fixed in `paper/runner.py::governed_intent` and pinned by
`tests/test_governed_intent.py`, so this list describes a closed bug and must not grow. A new
entry here means the same class of defect recurred and should be treated as such.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VoidedEntry:
    """One ledger row that records something the system did not do."""

    seq: int
    symbol: str
    reason: str
    evidence: str


VOIDED: tuple[VoidedEntry, ...] = (
    VoidedEntry(
        seq=264,
        symbol="COINUSDT",
        reason=(
            "the Constitution refused this position and the ledger recorded it as filled; the "
            "P&L booked against it was never earned"
        ),
        evidence=(
            "risk_records.jsonl seq 264: quantity_after '0', binding_constraint 'no_exposure', "
            "reason 'no exposure proposed; nothing to narrow' — while paper_ledger.jsonl seq 264 "
            "holds verdict 'trade', quantity '1', net_pnl '9.6521'. desk_notes.jsonl seq 264: "
            "'no order: final verdict human_review with quantity 0'"
        ),
    ),
    VoidedEntry(
        seq=265,
        symbol="MSTRUSDT",
        reason=(
            "identical to seq 264 — same cycle, same defect, same contradiction between the risk "
            "record and the ledger"
        ),
        evidence=(
            "risk_records.jsonl seq 265: quantity_after '0', binding_constraint 'no_exposure' — "
            "while paper_ledger.jsonl seq 265 holds verdict 'trade', quantity '1', "
            "net_pnl '5.3339'. desk_notes.jsonl seq 265: 'no order: final verdict human_review "
            "with quantity 0'"
        ),
    ),
)

VOIDED_SEQS: frozenset[int] = frozenset(v.seq for v in VOIDED)


def is_voided(seq: int) -> bool:
    """True if this ledger row records something the system did not actually do."""
    return seq in VOIDED_SEQS


def render() -> list[str]:
    """The correction notice, for any report that quotes ledger-derived figures."""
    if not VOIDED:
        return []
    out = [
        f"[correction] {len(VOIDED)} ledger row(s) are VOID — recorded as filled, never taken. "
        f"They remain in the chain unedited; every figure below excludes them.",
    ]
    out.extend(f"[correction]   seq {v.seq} {v.symbol}: {v.reason}" for v in VOIDED)
    return out


__all__ = ["VOIDED", "VOIDED_SEQS", "VoidedEntry", "is_voided", "render"]
