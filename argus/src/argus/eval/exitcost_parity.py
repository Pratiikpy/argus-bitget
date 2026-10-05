"""Exit cost against EGRESS, on the same recorded books (build-list 5.4).

EGRESS (``Ritapossible/Egress``, MIT, a Season 2 AI Trading Desk entry; cloned at
``research/repos-owned/Ritapossible~Egress``) prices leaving a position by walking the displayed
book (``egress/exitcost.py`` ``walk`` and ``quote``, valued at the mid). ARGUS does the same in
`market/depth.OrderBook.sweep`. The two differ in one detail — EGRESS walks a base-unit quantity
fixed at the mid, ARGUS walks the notional — so the expectation is a tie, and this measures it on
every book in ``data/book_tape.jsonl`` (Bitget order books recorded by `market/ws_tape.py`) at two
sizes, with EGRESS's own functions imported from its clone and run unmodified.

**Finding, 2026-10-05.** 1,104 books over twelve symbols. At $10,000 the slippage the two report
differs by a median 0.002 bps (largest 0.027); at $100,000 by a median 0.005 bps (largest 0.15),
and both find the book too thin on the same 92 snapshots. A tie: the method is the same, so
neither is better at it, and EGRESS's fee (an assumed 10 bps spot taker) is a parameter, not a
measurement either side owns.

    python -m argus.eval.exitcost_parity     # writes data/exitcost_parity.json
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

from argus.market.depth import Level, OrderBook
from argus.truth import artefact
from argus.truth.paths import DATA_DIR

RIVAL: Final = Path(__file__).resolve().parents[4] / "research" / "repos-owned" / \
    "Ritapossible~Egress"
TAPE: Final = DATA_DIR / "book_tape.jsonl"
OUT: Final = DATA_DIR / "exitcost_parity.json"
SIZES: Final = (10_000, 100_000)


def compare(rows: list[dict[str, Any]], quote: Any) -> dict[str, Any]:
    """Slippage on a sell of each size, both ways, on every book in ``rows``."""
    out: dict[str, Any] = {}
    for size in SIZES:
        gaps: list[float] = []
        thin = [0, 0]
        for row in rows:
            bids = [(float(p), float(q)) for p, q in row["bids"]]
            asks = [(float(p), float(q)) for p, q in row["asks"]]
            theirs = quote(row["symbol"], size, bids, asks, side="sell")
            book = OrderBook(
                symbol=row["symbol"], fetched_at=datetime.fromisoformat(row["taken_at"]),
                bids=tuple(Level(Decimal(str(p)), Decimal(str(q))) for p, q in bids),
                asks=tuple(Level(Decimal(str(p)), Decimal(str(q))) for p, q in asks))
            ours = book.sweep(Decimal(size), direction="SELL")
            thin[0] += int(theirs.exhausted)
            thin[1] += int(not ours.complete)
            if not theirs.exhausted and ours.complete:
                gaps.append(abs(float(ours.slippage_bps) - theirs.slippage_bp))
        gaps.sort()
        out[str(size)] = {
            "books": len(rows), "compared": len(gaps),
            "median_abs_gap_bps": round(statistics.median(gaps), 4) if gaps else None,
            "largest_gap_bps": round(gaps[-1], 4) if gaps else None,
            "too_thin": {"egress": thin[0], "argus": thin[1]}}
    return out


def run(*, out: Path = OUT) -> dict[str, Any]:
    from importlib import import_module

    sys.path.insert(0, str(RIVAL))
    try:
        exitcost = import_module("egress.exitcost")  # the rival's own code, from its clone
    finally:
        sys.path.remove(str(RIVAL))
    rows = [json.loads(line) for line in TAPE.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    by_size = compare(rows, exitcost.quote)
    blob = {"rival": "Ritapossible/Egress egress/exitcost.py quote", "tape": "data/book_tape.jsonl",
            "symbols": sorted({r["symbol"] for r in rows}),
            # the rival's own function, imported from its clone and run unmodified
            "reference": {"repo": "Ritapossible/Egress", "licence": "MIT",
                          "function": "egress/exitcost.py quote (walk at the mid)"},
            "parity": by_size,
            "reproducibility": {"command": "python -m argus.eval.exitcost_parity",
                                "deterministic": True,
                                "inputs": "the recorded book tape; no network, no model"}}
    artefact.write(out, blob)
    return blob


def main() -> int:  # pragma: no cover - CLI
    print(json.dumps(run(), indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
