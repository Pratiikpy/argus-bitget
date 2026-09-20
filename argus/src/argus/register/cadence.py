"""The cadence — continuous registration, so the record is never a frozen batch.

**The defect this exists to fix was in the register's own opening.** All 36 founding claims were
committed at a single 24-hour horizon on 14 September, so every one of them resolved on the 15th.
The handbook puts judge review at **9/22 to 10/7** and public voting at **9/22 to 9/28**. A record
that finished resolving a week before anyone opens it is a record, and a record is not what this
was built to be: the whole argument for a register is that it is *running*, and a page showing the
same thing on the first day of review and the last is indistinguishable from a screenshot.

So registration is a **cadence, not a batch**. Every scheduled cycle registers a small number of
claims across a ladder of horizons, which produces two properties that a single batch cannot:

* **Something is always pending.** At any instant there are claims whose outcome is not yet known,
  which is what makes the commitment meaningful rather than historical.
* **Something has always just resolved.** Whoever opens the page sees claims graded since the last
  time they looked, permanently, none of them editable.

**The ladder is the mechanism.** Horizons of 24h, 72h, 7d, 14d and 30d mean a claim registered
today resolves tomorrow, and another registered today resolves next month, so the stream of
resolutions is continuous rather than lumpy. Registering only at 24h, which is what the opening
batch did, produces a sawtooth: everything known within a day, nothing pending after it.

**Nothing here is chosen.** The symbol rotates by cycle index and the horizons are fixed, so which
claims exist is a function of when the cycle ran and not of what anyone felt like committing to. A
cadence that let a claimant pick today's subject would let them avoid the instruments they expect to
be wrong about, which is the selection effect the whole system is built to refuse.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from pathlib import Path
from typing import Any

from argus.register.claims import (
    REGISTER_PATH,
    Claim,
    Predicate,
    batch_digest,
    make_claim,
    register,
)
from argus.register.open_register import CLAIMANT, SOURCE

HORIZON_LADDER: tuple[timedelta, ...] = (
    timedelta(hours=24),
    timedelta(hours=72),
    timedelta(days=7),
    timedelta(days=14),
    timedelta(days=30),
)
"""Five horizons per registration, so resolutions arrive continuously rather than all at once.

The long end matters more than it looks. A 30-day claim registered in mid-September resolves in
mid-October — inside the review window and after it — which means the record keeps moving for
somebody who checks it a month from now, not only for somebody watching this week.
"""

ANCHOR_EVERY = 4
"""Anchor the chain head every N cycles rather than every cycle.

A calendar submission is a network round trip to four operators, and the chain already binds every
earlier claim into its head — so anchoring the head anchors everything before it. Anchoring on every
cycle would be four times the traffic for no additional proof. Four cycles is roughly daily on the
current schedule.
"""


def _symbol_for(index: int, symbols: tuple[str, ...]) -> str:
    """Rotate by cycle index. Deterministic, and not a choice anybody gets to make."""
    return symbols[index % len(symbols)]


def cycle_index(path: Path = REGISTER_PATH) -> int:
    """How many cadence registrations have happened, derived from the file rather than stored.

    Counting the claims rather than keeping a counter means there is no second piece of state that
    could disagree with the record — and the record is the thing that has to be right.
    """
    from argus.register.claims import _read

    if not path.exists():
        return 0
    return len([c for c in _read(path) if c.claimant == CLAIMANT]) // len(HORIZON_LADDER)


def build_cadence_batch(
    *, now: datetime | None = None, symbols: tuple[str, ...] | None = None,
    index: int | None = None,
) -> list[Claim]:
    """One claim per horizon on one rotating instrument. Five claims per cycle.

    The claim is a volatility claim at every horizon: the absolute move over the horizon exceeds the
    instrument's own median hourly move scaled by the square root of the horizon in hours. That is
    the random-walk scaling, so the threshold is what a driftless market would produce and the claim
    is a statement about whether this instrument moves more than that.

    Direction is deliberately absent from the cadence. `desk/analogue.py`, `desk/shapematch.py` and
    `research/cointegration.py` each measured that we have no directional edge here, the founding
    batch committed direction at exactly 0.50 to say so on the record, and repeating a coin four
    times a day would pad the register with claims that cannot inform anyone.
    """
    from argus.market.bitget import RTOKEN_SYMBOLS
    from argus.market.history import CandleType, fetch_range

    moment = now or datetime.now(UTC)
    universe = symbols or tuple(RTOKEN_SYMBOLS)
    position = cycle_index() if index is None else index
    symbol = _symbol_for(position, universe)

    bars = fetch_range(symbol, days=7, interval="1H", candle_type=CandleType.MARKET)
    moves = sorted(
        abs(float(b.close) / float(a.close) - 1.0) * 10_000
        for a, b in pairwise(bars)
        if float(a.close) > 0
    )
    if len(moves) < 24:
        return []
    median_hourly = moves[len(moves) // 2]

    out: list[Claim] = []
    for horizon in HORIZON_LADDER:
        hours = horizon.total_seconds() / 3600
        threshold = Decimal(str(round(median_hourly * (hours ** 0.5), 2)))
        out.append(make_claim(
            claimant=CLAIMANT, subject=symbol, predicate=Predicate.ABS_MOVE_ABOVE_BPS,
            threshold=str(threshold), resolves_at=moment + horizon, source=SOURCE,
            confidence=0.60, now=moment,
            invalidation="trading halted or the venue stops publishing this symbol",
        ))
    return out


def run_cadence(
    *, now: datetime | None = None, anchor_batch: bool | None = None,
    path: Path = REGISTER_PATH,
) -> dict[str, Any]:
    """Register this cycle's claims, and anchor the head when the cadence says to."""
    position = cycle_index(path)
    batch = build_cadence_batch(now=now, index=position)
    if not batch:
        return {"registered": 0, "reason": "not enough history for this cycle's instrument"}

    committed = register(batch, path=path)
    head = batch_digest(committed)
    should_anchor = (position % ANCHOR_EVERY == 0) if anchor_batch is None else anchor_batch

    result: dict[str, Any] = {
        "cycle": position,
        "registered": len(committed),
        "subject": committed[0].subject,
        "horizons": [c.resolves_at for c in committed],
        "head": head,
        "anchored": False,
        "calendars": [],
    }
    if should_anchor:
        from argus.paper.anchor import anchor as submit_anchor

        proof = submit_anchor(head, subject=f"register cadence cycle {position}")
        result["anchored"] = proof.anchored
        result["calendars"] = [r.calendar for r in proof.receipts]
    return result


def horizon_coverage(
    *, path: Path = REGISTER_PATH, window: tuple[datetime, datetime] | None = None,
) -> dict[str, Any]:
    """How many claims resolve inside a given window — the check that caught the original defect.

    Defaults to the judging window from the handbook: judge review runs 9/22 to 10/7. A register
    with zero resolutions inside it is a register nobody watching will see move.
    """
    from argus.register.claims import _read

    start, end = window or (
        datetime(2026, 9, 22, tzinfo=UTC), datetime(2026, 10, 7, 23, 59, tzinfo=UTC)
    )
    claims = _read(path) if path.exists() else []
    inside = [c for c in claims if start <= datetime.fromisoformat(c.resolves_at) <= end]
    pending_at_start = [
        c for c in claims if datetime.fromisoformat(c.resolves_at) > start
    ]
    return {
        "window": [start.isoformat(), end.isoformat()],
        "claims_total": len(claims),
        "resolving_inside_window": len(inside),
        "still_pending_at_window_open": len(pending_at_start),
        "verdict": (
            f"{len(inside)} claim(s) resolve inside the window and {len(pending_at_start)} are "
            f"still open when it starts"
            if inside else
            "NO claim resolves inside the window — the record would be frozen for anyone watching"
        ),
    }


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="register this cycle's claims")
    parser.add_argument("--no-anchor", action="store_true")
    parser.add_argument(
        "--coverage", action="store_true", help="report resolutions inside the judging window",
    )
    args = parser.parse_args()

    if args.coverage:
        print(json.dumps(horizon_coverage(), indent=2))
        return 0

    result = run_cadence(anchor_batch=False if args.no_anchor else None)
    if not result["registered"]:
        print(f"nothing registered: {result.get('reason')}")
        return 1
    print(
        f"cycle {result['cycle']}: {result['registered']} claim(s) on {result['subject']}, "
        f"head {result['head'][:16]}"
    )
    for stamp in result["horizons"]:
        print(f"    resolves {stamp[:16]}")
    if result["anchored"]:
        print(f"  anchored to {len(result['calendars'])} calendar(s)")

    coverage = horizon_coverage()
    print(f"\n  {coverage['verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "ANCHOR_EVERY",
    "HORIZON_LADDER",
    "build_cadence_batch",
    "cycle_index",
    "horizon_coverage",
    "run_cadence",
]
