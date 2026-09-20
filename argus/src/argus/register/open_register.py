"""Open the register: commit the first claims and anchor them to Bitcoin.

**This is the only step in the whole design where delay is unrecoverable.** Build effort can be
compressed — a year of engineering can be done in a week by someone determined enough. Elapsed time
cannot. A register that opens today and is finished over the following year holds a year of record;
one that is perfect in six months holds six months less, permanently, and never catches up to
itself, because the input is calendar time and there is no market in it.

So this module exists to be run **before** the API, the network, the grading, the scoring page and
everything else that would make it presentable. It is deliberately crude. It cannot be late.

**What it commits.** A batch of falsifiable claims about Bitget's tokenized US equities across the
7x24 window — the structural fact this whole season is built around. Each claim names a predicate
that executes against a point-in-time series, a threshold, a resolution time, and the claimant's
stated probability. The batch's head hash goes to four independent Bitcoin calendars via
`paper/anchor.py`, so the commitment is verifiable by a stranger with no access to this machine and
no reason to trust us.

**The claims are generated from a stated rule, not chosen.** `desk/shapematch.py` found shape
precedent on these instruments to be indistinguishable from reordered noise, and
`research/cointegration.py` found no cointegrated pair survives correction. A register opened with
hand-picked claims we felt good about would be selecting on the answer — the same defect those two
modules were built to detect. So the batch is mechanical: every rToken gets the same three claims
at the same horizons, with confidences set by a rule that is written down here and applied without
exception.
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
    batch_digest,
    make_claim,
    register,
    verify_chain,
)

CLAIMANT = "argus"
SOURCE = "bitget:1H:market"
HORIZON_HOURS = 24

VOLATILITY_CONFIDENCE = 0.60
"""Stated probability that the absolute 24-hour move exceeds the instrument's own median.

**Not a guess, and deliberately not 0.5.** The median is by construction the level an absolute move
exceeds half the time, so a naive claim would sit at exactly 0.50 and score nothing either way. We
state 0.60 because `risk/session_risk.py` measured that the reopen bar carries 1.5-2.2x a
regular-hours bar, and every one of these horizons crosses a reopen — so the 24-hour window ahead is
more volatile than the median hour behind it. That is a real, measured, falsifiable reason to depart
from the coin, and if it is wrong the record will say so within a day.
"""

DIRECTION_CONFIDENCE = 0.50
"""Stated probability on direction: exactly the coin, on purpose.

Every retrieval this project built — `desk/analogue.py`, `desk/shapematch.py`,
`research/cointegration.py` — returned *no precedent* on these instruments. Claiming a directional
edge we have measured ourselves not to have would be the exact dishonesty this register exists to
price. So the direction claims are committed at 0.50 and will score as a coin, which is the honest
number and is itself a falsifiable claim: if our directional calls come back consistently above or
below chance, that is a finding either way.
"""


def build_batch(
    symbols: tuple[str, ...] | None = None, *, now: datetime | None = None,
) -> list[Claim]:
    """Three mechanical claims per instrument. No selection, no discretion.

    The rule, applied to every symbol without exception:

    * **volatility** — the absolute 24-hour move exceeds the instrument's own trailing median hourly
      move times the square root of 24 (the random-walk scaling), stated at
      :data:`VOLATILITY_CONFIDENCE`;
    * **up** — the 24-hour return is above zero, at :data:`DIRECTION_CONFIDENCE`;
    * **down** — the 24-hour return is below zero, at the same confidence.

    The up/down pair is deliberate: they are mutually exclusive and jointly near-exhaustive, so a
    register that mis-scores one will visibly mis-score the other, and the pair acts as a check on
    the resolver itself.
    """
    from argus.market.bitget import RTOKEN_SYMBOLS
    from argus.market.history import CandleType, fetch_range
    from argus.register.claims import Predicate as P

    moment = now or datetime.now(UTC)
    horizon = moment + timedelta(hours=HORIZON_HOURS)
    wanted = symbols or tuple(RTOKEN_SYMBOLS)

    out: list[Claim] = []
    for symbol in wanted:
        try:
            bars = fetch_range(symbol, days=7, interval="1H", candle_type=CandleType.MARKET)
        except Exception as exc:
            print(f"  {symbol}: no history ({type(exc).__name__}); no claim registered")
            continue
        moves = [
            abs(float(b.close) / float(a.close) - 1.0) * 10_000
            for a, b in pairwise(bars)
            if float(a.close) > 0
        ]
        if len(moves) < 24:
            print(f"  {symbol}: only {len(moves)} bars; no claim registered")
            continue
        moves.sort()
        median_hourly = moves[len(moves) // 2]
        threshold = Decimal(str(round(median_hourly * (24 ** 0.5), 2)))

        out.append(make_claim(
            claimant=CLAIMANT, subject=symbol, predicate=P.ABS_MOVE_ABOVE_BPS,
            threshold=str(threshold), resolves_at=horizon, source=SOURCE,
            confidence=VOLATILITY_CONFIDENCE, now=moment,
            invalidation="trading halted or the venue stops publishing this symbol",
        ))
        out.append(make_claim(
            claimant=CLAIMANT, subject=symbol, predicate=P.RETURN_OVER_ABOVE_BPS,
            threshold="0", resolves_at=horizon, source=SOURCE,
            confidence=DIRECTION_CONFIDENCE, now=moment,
            invalidation="trading halted or the venue stops publishing this symbol",
        ))
        out.append(make_claim(
            claimant=CLAIMANT, subject=symbol, predicate=P.RETURN_OVER_BELOW_BPS,
            threshold="0", resolves_at=horizon, source=SOURCE,
            confidence=DIRECTION_CONFIDENCE, now=moment,
            invalidation="trading halted or the venue stops publishing this symbol",
        ))
    return out


def open_register(
    *, symbols: tuple[str, ...] | None = None, anchor_batch: bool = True,
    path: Path = REGISTER_PATH,
) -> dict[str, Any]:
    """Commit the batch, anchor its head, and return what a stranger would need to check it."""
    batch = build_batch(symbols)
    if not batch:
        raise RuntimeError("no claims could be built; nothing was registered")

    committed = register(batch, path=path)
    head = batch_digest(committed)
    intact, note = verify_chain(path)

    result: dict[str, Any] = {
        "opened_at": datetime.now(UTC).isoformat(),
        "claims": len(committed),
        "head": head,
        "chain": note,
        "chain_intact": intact,
        "anchored": False,
        "calendars": [],
        "anchor_failures": [],
    }

    if anchor_batch:
        from argus.paper.anchor import anchor as submit_anchor

        proof = submit_anchor(head, subject=f"register batch of {len(committed)} claim(s)")
        result["anchored"] = proof.anchored
        result["calendars"] = [r.calendar for r in proof.receipts]
        result["anchor_failures"] = list(proof.failures)
    return result


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="open the register: commit and anchor claims")
    parser.add_argument("--symbols", default="")
    parser.add_argument(
        "--no-anchor", action="store_true",
        help="commit without submitting to the Bitcoin calendars (for testing only)",
    )
    args = parser.parse_args()

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip()) or None
    result = open_register(symbols=symbols, anchor_batch=not args.no_anchor)

    print(f"\n{result['claims']} claim(s) committed")
    print(f"  head      {result['head']}")
    print(f"  chain     {result['chain']}")
    if result["anchored"]:
        print(f"  anchored  {len(result['calendars'])} calendar(s): "
              f"{', '.join(c.split('//')[-1] for c in result['calendars'])}")
    else:
        print("  anchored  NO — the commitment exists but is not externally verifiable yet")
    for failure in result["anchor_failures"]:
        print(f"    failed: {failure}")

    out = REGISTER_PATH.parent / "register_opened.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    print("\nthe clock is running. resolve with: python -m argus.register.resolve")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "CLAIMANT",
    "DIRECTION_CONFIDENCE",
    "HORIZON_HOURS",
    "SOURCE",
    "VOLATILITY_CONFIDENCE",
    "build_batch",
    "open_register",
]
