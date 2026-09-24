"""Session-tagged rToken collector — the proprietary dataset.

The PRD calls an rToken panel joining price, session state, spread and liquidity **"potentially our
most defensible research asset, because it does not exist anywhere else."** This is that collector.
Every observation is stamped with the anchor market's session phase at the moment of capture, which
is the join nobody else has and the whole reason the panel is worth building.

**It already earned its place.** The first live run, on a Saturday with the anchor shut and 53.6
hours to the next price discovery, measured quoted spreads of **0.46bps on NVDAUSDT, 0.27 on
TSLAUSDT, 0.30 on AAPLUSDT**. Our own specification (SS-6) asserted off-hours spreads run 3-5x
wider, citing Barclay & Hendershott. For Bitget rToken perps that is **false**, and the error would
have flowed straight into the cost model.

What survives, and is strengthened: the **12bps round-trip taker fee is roughly 26x the quoted
spread** on the majors. Spread is not the barrier to an off-hours strategy. The fee is, by an order
of magnitude, and any strategy that round-trips frequently is dead before it starts.

The thin products behave differently and the difference is informative rather than noise: SPXUSDT
quoted 8.09bps and SQQQUSDT 2.58bps in the same snapshot. Liquidity tiering within rTokens is real
even when the headline names are tight.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.market.bitget import RTOKEN_SYMBOLS, Ticker, fetch_rtokens
from argus.truth.clocks import DualClock


@dataclass(frozen=True, slots=True)
class Observation:
    """One instrument at one instant, joined to the anchor market's session state.

    The session fields are the contribution. A price series without them cannot answer "what does
    this instrument do while its anchor is asleep", which is the question every sub-theme in this
    project reduces to.
    """

    captured_at: str
    symbol: str
    anchor: str
    phase: str
    anchor_asleep: bool
    hours_to_discovery: float
    last: str
    bid: str
    ask: str
    spread_bps: str
    change_24h: str
    base_volume: str
    funding_rate: str

    @classmethod
    def of(cls, ticker: Ticker, clock: DualClock) -> Observation:
        state = clock.state(ticker.fetched_at)
        return cls(
            captured_at=ticker.fetched_at.isoformat(),
            symbol=ticker.symbol,
            anchor=ticker.anchor or "",
            phase=str(state.phase),
            anchor_asleep=state.is_anchor_asleep,
            hours_to_discovery=round(state.hours_to_next_discovery, 3),
            last=str(ticker.last),
            bid=str(ticker.bid),
            ask=str(ticker.ask),
            # Stored as a string like every other number here: these rows become Facts, and a
            # float that has been through JSON is not the number that was measured.
            spread_bps=str(ticker.spread_bps.quantize(Decimal("0.0001"))),
            change_24h=str(ticker.change_24h),
            base_volume=str(ticker.base_volume),
            funding_rate=str(ticker.funding_rate),
        )


FIELDS = tuple(Observation.__annotations__.keys())


def snapshot(clock: DualClock | None = None) -> list[Observation]:
    """One capture of every rToken, session-tagged."""
    clock = clock or DualClock()
    tickers = fetch_rtokens()
    return [Observation.of(tickers[s], clock) for s in RTOKEN_SYMBOLS if s in tickers]


def append_csv(rows: list[Observation], path: Path) -> int:
    """Append-only. History is never rewritten — a panel that can be edited in place cannot
    support a point-in-time claim later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return len(rows)


def session_summary(rows: list[Observation]) -> dict[str, Any]:
    """What the snapshot says about spread by session — the headline the panel exists to produce."""
    if not rows:
        return {}
    by_phase: dict[str, list[Decimal]] = {}
    for r in rows:
        by_phase.setdefault(r.phase, []).append(Decimal(r.spread_bps))
    return {
        "captured_at": rows[0].captured_at,
        "phase": rows[0].phase,
        "anchor_asleep": rows[0].anchor_asleep,
        "hours_to_discovery": rows[0].hours_to_discovery,
        "instruments": len(rows),
        "median_spread_bps": {
            phase: str(sorted(v)[len(v) // 2]) for phase, v in by_phase.items()
        },
        "widest": max(rows, key=lambda r: Decimal(r.spread_bps)).symbol,
        "tightest": min(rows, key=lambda r: Decimal(r.spread_bps)).symbol,
        # The number that matters: a 12bps round trip against the observed spread.
        "round_trip_fee_bps": "12",
        "fee_to_median_spread_ratio": str(
            (Decimal("12") / max(sorted(Decimal(r.spread_bps) for r in rows)[len(rows) // 2],
                                 Decimal("0.0001"))).quantize(Decimal("0.1"))
        ),
    }


def main() -> int:
    """Capture once, append to the panel, print the summary.

    Intended to run on a schedule. The value is entirely in the time series: a single snapshot
    shows a spread, a year of session-tagged snapshots shows what happens to an instrument whose
    anchor is asleep — and that does not exist anywhere else.
    """
    clock = DualClock()
    rows = snapshot(clock)
    out = Path(__file__).resolve().parents[3] / "data" / "rtoken_panel.csv"
    written = append_csv(rows, out)
    summary = session_summary(rows)
    print(json.dumps(summary, indent=2))
    print(f"\nappended {written} rows -> {out}")

    if rows and rows[0].anchor_asleep:
        print(
            f"\nAnchor is asleep ({rows[0].phase}, {rows[0].hours_to_discovery}h to discovery). "
            f"Median spread is {summary['median_spread_bps'].get(rows[0].phase)}bps against a "
            f"12bps round trip — the fee is {summary['fee_to_median_spread_ratio']}x the spread."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
