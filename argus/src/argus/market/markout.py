"""Adverse selection, measured: where the mid goes after somebody else's trade prints.

`execution/passive.py` opens with an admission. It models queue position faithfully — ported from
`hftbacktest`'s `queue.rs` — and then refuses to credit the maker side of the spread at all, because
our own replay work turned a +90% local simulation into -0.45% once resting orders were modelled
honestly. The reason a resting order loses is **adverse selection**: it fills when the market is
about to move against it and misses when it is not. That refusal was the right call and it was a
blanket assumption, not a measurement. Nothing in the system had ever asked *how much* the effect is
worth on these instruments.

**The measurement.** Poll the order book at roughly 1Hz for a bounded window, then pull the venue's
public fills (`GET /api/v3/market/fills`, `agent-sdk/src/generated/catalog.ts:118` — public, up to
100 prints, each carrying `price`, `size`, `side` and a millisecond `ts`). Join them on time: for
every print, take the mid at the moment it happened and the mid some seconds later. The signed move
that follows an aggressive buy is what the passive seller on the other side gave up.

That is the standard markout, and its sign convention is the part that is easy to get backwards. A
print with ``side = "buy"`` is an **aggressor lifting the offer**, so the counterparty is a resting
sell. If the mid rises afterwards, the aggressor was right and the resting seller lost. Markout is
therefore reported **from the passive side's point of view**: positive means the passive fill was a
good one, negative means it was picked off.

**Why this is not a backtest.** It measures the venue's own flow, not ours — we have no executed
trades. It is the honest available proxy: if the typical resting order on this book is picked off
by more than the spread it captures, a passive strategy loses here regardless of how good its queue
model is, and `cost/model.py` should keep refusing to credit maker treatment. If the effect is
small, that refusal is costing real edge and should be revisited with a number attached.

**What would make this measurement lie.** A sample taken entirely inside a shut anchor session
would find almost no informed flow and flatter the passive side; one taken across a single news
event would condemn it. So the window, phase and print count are reported with every figure, and
:meth:`MarkoutReport.verdict` refuses to conclude anything below :data:`MIN_PRINTS`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import fmean, median
from typing import Any

from argus.market.bitget import BitgetError, _dec, _get
from argus.market.depth import DepthError, fetch_orderbook

HORIZONS_SECONDS: tuple[int, ...] = (5, 15, 30, 60)
"""Markout horizons. Short by the standards of an equity desk and long by a market maker's.

Chosen to bracket the holding period of a resting order on this venue rather than inherited from a
paper: below five seconds the sample is dominated by the poll interval, and beyond a minute the
measurement is about the market's drift rather than about who traded with whom.
"""

MIN_PRINTS = 20
"""Fewest joined prints before a markout figure is reported as anything but a count."""

DEFAULT_WINDOW_SECONDS = 90
"""How long to sample. Long enough to outlast the longest horizon, short enough to sit inside a
scheduled cycle without delaying the decision it precedes."""

POLL_SECONDS = 1.0
"""Book polling interval. One request per second per symbol is well inside the venue's public rate
limits and fine enough that a five-second horizon is five samples rather than one."""

MAKER_BPS = 2.0
"""Our maker fee, from `cost/model.py:bitget_perp`. The number adverse selection has to beat."""


class MarkoutError(RuntimeError):
    """Raised rather than returning a markout computed from a sample that cannot support one."""


@dataclass(frozen=True, slots=True)
class MidSample:
    """The mid at one instant, from a live book snapshot."""

    ts_ms: int
    mid: float
    spread_bps: float


@dataclass(frozen=True, slots=True)
class Print:
    """One public fill: price, size, and which side was the aggressor."""

    ts_ms: int
    price: float
    size: float
    side: str
    """``buy`` means an aggressor lifted the offer, so the passive counterparty was a seller."""

    rpi: bool
    """Bitget's retail-price-improvement flag, carried through because RPI flow is by construction
    less informed than the rest and pooling the two would understate adverse selection on the
    liquidity a professional maker actually faces."""


@dataclass(frozen=True, slots=True)
class Markout:
    """The passive side's outcome at one horizon."""

    horizon_seconds: int
    prints: int
    median_bps: float
    mean_bps: float
    worst_bps: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon_seconds": self.horizon_seconds,
            "prints": self.prints,
            "median_bps": round(self.median_bps, 4),
            "mean_bps": round(self.mean_bps, 4),
            "worst_bps": round(self.worst_bps, 4),
        }


@dataclass(frozen=True, slots=True)
class MarkoutReport:
    """What the mid did after other people's trades, and what that implies for resting here."""

    symbol: str
    phase: str
    started_at: datetime
    window_seconds: int
    samples: int
    prints_seen: int
    markouts: tuple[Markout, ...]
    rpi_share: float

    @property
    def longest(self) -> Markout | None:
        """The longest horizon that actually joined enough prints to say something.

        Not simply the longest horizon: a 90-second window ends before a 60-second horizon can
        resolve for any print in its final minute, so the 60s row is often empty while the 5s row is
        not. Reporting the empty row as the headline said "0 prints" when five had been measured,
        which read as a quieter book rather than as a shorter usable horizon.
        """
        usable = [m for m in self.markouts if m.prints >= MIN_PRINTS]
        if usable:
            return max(usable, key=lambda m: m.horizon_seconds)
        return max(self.markouts, key=lambda m: (m.prints, -m.horizon_seconds), default=None)

    @property
    def verdict(self) -> str:
        longest = self.longest
        if longest is None or longest.prints < MIN_PRINTS:
            seen = 0 if longest is None else longest.prints
            horizon = "" if longest is None else f" at the {longest.horizon_seconds}s horizon"
            return (
                f"{seen} joined print(s){horizon} over {self.window_seconds}s of {self.symbol} "
                f"({self.phase}) — below the {MIN_PRINTS} this module will draw a conclusion from. "
                f"The sample says the book was quiet, not that resting here is safe"
            )
        cost = -longest.median_bps
        if cost <= 0:
            return (
                f"Over {longest.horizon_seconds}s the passive side of {longest.prints} print(s) "
                f"gained a median {longest.median_bps:+.2f}bps on {self.symbol} ({self.phase}). No "
                f"measurable adverse selection in this window — which is a statement about "
                f"{self.window_seconds} seconds, not about the instrument"
            )
        verdict = (
            f"Resting on {self.symbol} ({self.phase}) cost a median {cost:.2f}bps of adverse "
            f"selection over {longest.horizon_seconds}s across {longest.prints} print(s), worst "
            f"{-longest.worst_bps:.2f}bps."
        )
        if cost > MAKER_BPS:
            return verdict + (
                f" That exceeds the {MAKER_BPS:.0f}bps maker fee it would save, so passive "
                f"execution is not free money here and `cost/model.py`'s refusal to credit the "
                f"maker side stands"
            )
        return verdict + (
            f" That is inside the {MAKER_BPS:.0f}bps maker fee it would save, so the blanket "
            f"refusal to credit maker treatment is now costing measurable edge and deserves a "
            f"number rather than a rule"
        )

    def render(self) -> str:
        lines = [
            f"MARKOUT — {self.symbol}, {self.window_seconds}s of {self.phase}, "
            f"{self.samples} book sample(s), {self.prints_seen} print(s), "
            f"{self.rpi_share:.0%} flagged RPI",
            "",
            f"  {'horizon':>8} {'prints':>7} {'median':>9} {'mean':>9} {'worst':>9}",
        ]
        for markout in self.markouts:
            lines.append(
                f"  {markout.horizon_seconds:7d}s {markout.prints:7d} "
                f"{markout.median_bps:+9.3f} {markout.mean_bps:+9.3f} {markout.worst_bps:+9.3f}"
            )
        lines += ["", f"  {self.verdict}"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "phase": self.phase,
            "started_at": self.started_at.isoformat(),
            "window_seconds": self.window_seconds,
            "samples": self.samples,
            "prints_seen": self.prints_seen,
            "rpi_share": round(self.rpi_share, 4),
            "maker_bps": MAKER_BPS,
            "markouts": [m.as_dict() for m in self.markouts],
            "verdict": self.verdict,
        }


def fetch_prints(symbol: str, *, limit: int = 100, category: str = "USDT-FUTURES") -> list[Print]:
    """The venue's recent public fills. Sorted oldest first, because the feed is not."""
    rows = _get(
        "/api/v3/market/fills",
        {"category": category, "symbol": symbol.upper(), "limit": str(limit)},
    )
    out: list[Print] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        try:
            stamp = int(str(row.get("ts")))
        except (TypeError, ValueError):
            continue
        price = float(_dec(row.get("price")))
        size = float(_dec(row.get("size")))
        side = str(row.get("side", "")).lower()
        if price <= 0 or size <= 0 or side not in ("buy", "sell"):
            continue
        out.append(Print(
            ts_ms=stamp, price=price, size=size, side=side,
            rpi=str(row.get("isRPI", "")).upper() == "YES",
        ))
    return sorted(out, key=lambda p: p.ts_ms)


def _mid_at(samples: Sequence[MidSample], ts_ms: int) -> float | None:
    """The first sampled mid at or after ``ts_ms``.

    At or after, never before: the mid *preceding* a print is contaminated by whatever caused the
    print, and taking the nearest sample in either direction would silently mix the two conventions
    depending on where the poll happened to land.
    """
    for sample in samples:
        if sample.ts_ms >= ts_ms:
            return sample.mid
    return None


def analyse(
    samples: Sequence[MidSample], prints: Sequence[Print], *, symbol: str, phase: str,
    window_seconds: int, horizons: Sequence[int] = HORIZONS_SECONDS,
    started_at: datetime | None = None,
) -> MarkoutReport:
    """Markout from the passive side, by horizon.

    A print with ``side="buy"`` was an aggressor lifting the offer, so the resting counterparty
    sold. The passive outcome is therefore the **negative** of the mid's subsequent move for a buy,
    and the move itself for a sell. Getting that sign backwards produces a beautifully symmetric
    result that says the opposite of the truth, so it is written once here and asserted in tests.
    """
    if not samples:
        raise MarkoutError("no book samples; there is no mid to measure against")
    ordered = sorted(samples, key=lambda s: s.ts_ms)
    results: list[Markout] = []
    for horizon in sorted(horizons):
        moves: list[float] = []
        for fill in prints:
            entry = _mid_at(ordered, fill.ts_ms)
            later = _mid_at(ordered, fill.ts_ms + horizon * 1000)
            if entry is None or later is None or entry <= 0:
                continue
            drift_bps = (later - entry) / entry * 10_000
            passive = -drift_bps if fill.side == "buy" else drift_bps
            moves.append(passive)
        if not moves:
            results.append(Markout(horizon, 0, 0.0, 0.0, 0.0))
            continue
        results.append(Markout(
            horizon_seconds=horizon, prints=len(moves), median_bps=median(moves),
            mean_bps=fmean(moves), worst_bps=min(moves),
        ))
    rpi = sum(1 for p in prints if p.rpi) / len(prints) if prints else 0.0
    return MarkoutReport(
        symbol=symbol, phase=phase, started_at=started_at or datetime.now(UTC),
        window_seconds=window_seconds, samples=len(ordered), prints_seen=len(prints),
        markouts=tuple(results), rpi_share=rpi,
    )


def collect(
    symbol: str, *, window_seconds: int = DEFAULT_WINDOW_SECONDS,
    poll_seconds: float = POLL_SECONDS,
) -> tuple[list[MidSample], list[Print]]:  # pragma: no cover - live sampling
    """Poll the book for a bounded window, then pull the prints that fell inside it.

    The book is polled and the prints are fetched once at the end rather than both being polled:
    the fills feed returns its last hundred with millisecond stamps, so one call at the end covers
    the whole window and cannot double-count a print the way repeated polling would.
    """
    if window_seconds < max(HORIZONS_SECONDS):
        raise MarkoutError(
            f"a {window_seconds}s window cannot support a {max(HORIZONS_SECONDS)}s horizon"
        )
    samples: list[MidSample] = []
    deadline = time.monotonic() + window_seconds
    while time.monotonic() < deadline:
        try:
            book = fetch_orderbook(symbol, limit=5)
        except (BitgetError, DepthError):
            time.sleep(poll_seconds)
            continue
        samples.append(MidSample(
            ts_ms=int(book.fetched_at.timestamp() * 1000),
            mid=float(book.mid),
            spread_bps=float(book.spread_bps),
        ))
        time.sleep(poll_seconds)
    if not samples:
        raise MarkoutError(f"no book sample succeeded for {symbol} in {window_seconds}s")
    first, last = samples[0].ts_ms, samples[-1].ts_ms
    prints = [p for p in fetch_prints(symbol) if first <= p.ts_ms <= last]
    return samples, prints


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import sys
    from pathlib import Path

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from argus.truth.clocks import DualClock

    parser = argparse.ArgumentParser(description="how badly is a resting order picked off here?")
    parser.add_argument("--symbol", default="NVDAUSDT")
    parser.add_argument("--seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    args = parser.parse_args()

    started = datetime.now(UTC)
    phase = DualClock().phase(started).value
    print(f"sampling {args.symbol} for {args.seconds}s ({phase})...")
    try:
        samples, prints = collect(args.symbol, window_seconds=args.seconds)
    except (MarkoutError, BitgetError) as exc:
        print(f"could not sample: {exc}")
        return 1
    report = analyse(
        samples, prints, symbol=args.symbol, phase=phase, window_seconds=args.seconds,
        started_at=started,
    )
    print()
    print(report.render())
    out = Path(__file__).resolve().parents[3] / "data" / "markout.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "DEFAULT_WINDOW_SECONDS",
    "HORIZONS_SECONDS",
    "MAKER_BPS",
    "MIN_PRINTS",
    "Markout",
    "MarkoutError",
    "MarkoutReport",
    "MidSample",
    "Print",
    "analyse",
    "collect",
    "fetch_prints",
]
