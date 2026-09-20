"""An independent, clean-room reimplementation of AutonomousTradeAgents' "Ghost P&L" abstention
ledger aggregation — NOT vendored, because there is no license to vendor under.

``eval/standing.py``'s "Abstention scored as a decision" capability's own `best_implementation_
studied` proof previously found nothing in the research corpus that records the counterfactual
outcome of a declined trade. A fresh, targeted search (2026-09-16) found one genuine exception:
``insaneamogh/AutonomousTradeAgents`` — a submitted Alpaca hackathon entry whose entire product
thesis is "The Refusal Ledger." ``apps/api/app/services/council/ghost_service.py``
(``_bucket_from_rows`` and ``build_ghost_summary``, commit not pinned — the clone on this machine
has no `.git` history beyond the working tree; read in full) marks every vetoed/declined trade
proposal to market over a finalization horizon and reports the result in dollars.

**Why this file exists instead of a vendored copy.** ``gh repo view insaneamogh/
AutonomousTradeAgents --json licenseInfo`` returns ``{"licenseInfo": null}`` — no LICENSE file, so
this project's "vendor, byte-verify, pin a hash" playbook does not apply. What follows is an
independent reimplementation of the same described aggregation logic — different structure, no
text copied — verified against reference output computed by running their own real, unmodified
``_bucket_from_rows`` function once locally (via a `sys.modules` shim satisfying its unrelated
SQLAlchemy/FastAPI imports, the same technique this package's other loaders use, but never
registered as a permanent shim here — this is a one-off verification run, not a runtime
dependency)::

    # two synthetic rows reproducing ghost_service.py's OWN documented example
    # (a vetoed bucket holding $30,788 of avoided losses and $32,967 of blocked gains)
    rows = [(-30788.0, "final"), (32967.0, "final")]
    bucket = mod._bucket_from_rows(rows, ("vetoed",), now=now)
    # -> GhostBucket(ghost_pnl=2179.0, loss_avoided_usd=30788.0, upside_blocked_usd=32967.0, ...)
    saved_usd = round(max(0.0, -bucket.ghost_pnl), 2)
    # -> 0.0

which `tests/test_abstention_comparison.py::TestGhostLedgerMatchesTheRealReferenceOutput` pins
exactly.

**The decisive structural property this file makes checkable: the "headline" number their real
`build_ghost_summary` returns (`saved_usd`/`missed_usd`, both used the same `max(0.0, -net)`
pattern) STILL exhibits the exact netting-collapse defect their own code comment on
`GhostBucket.loss_avoided_usd` describes as a real, dated incident ("on 2026-09-04 the vetoed
bucket held $30,788 of avoided losses AND $32,967 of blocked gains, netting +$2,179, and
`max(0.0, -2179)` floored the tile to `$0`") — confirmed by running their own real code on their
own real example numbers, reproduced above, not assumed from reading the comment alone. The split
fields (`loss_avoided_usd`/`upside_blocked_usd`) were added alongside the flawed headline number,
not as a replacement for it — `build_ghost_summary` (source read in full,
`ghost_service.py:249-287`) still computes and returns `saved_usd`/`missed_usd` with the identical
`max(0.0, -net)` pattern their own comment names as the bug.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GhostBucketReimpl:
    """The fields of the real `GhostBucket` this comparison actually exercises — `count`,
    `pending_count` and the trading-day-horizon fields are omitted because they depend on
    `trading_day_offset`, a separate real function not needed to test the netting property."""

    ghost_pnl: float
    loss_avoided_usd: float
    upside_blocked_usd: float


def bucket_from_marks(marks: list[float]) -> GhostBucketReimpl:
    """The same aggregation `_bucket_from_rows` does for a set of already-finalized ghost marks
    (dollar P&L each, signed): sum for the net, split by sign for the two-sided breakdown."""
    net = round(sum(marks), 2)
    loss_avoided = round(-sum(m for m in marks if m < 0), 2)
    upside_blocked = round(sum(m for m in marks if m > 0), 2)
    return GhostBucketReimpl(
        ghost_pnl=net, loss_avoided_usd=loss_avoided, upside_blocked_usd=upside_blocked,
    )


def headline_saved_usd(bucket: GhostBucketReimpl) -> float:
    """`build_ghost_summary`'s real `saved_usd` computation
    (`round(max(0.0, -vetoed.ghost_pnl), 2)`, `ghost_service.py:278`) — the specific pattern their
    own comment on `GhostBucket.loss_avoided_usd` names as the bug, reproduced here unmodified in
    form so the same input produces the same (flawed) output their real code produces."""
    return round(max(0.0, -bucket.ghost_pnl), 2)


__all__ = ["GhostBucketReimpl", "bucket_from_marks", "headline_saved_usd"]
