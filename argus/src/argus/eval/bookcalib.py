"""Book calibration — the queue experiment's parameters, measured instead of invented.

`eval/queueproof.py` scores hftbacktest's probability functions against a queue whose truth is known
by construction. That answers "is the port right" and "does the model beat a constant", but it
leaves a real objection standing: **the simulated book's parameters were chosen by us.** Level sizes
of `uniform(1, 50)`, three to twelve orders a level, a 35/45/20 split of trades, cancellations and
joins — none of that was measured. A result on a book nobody has seen is a result about our
imagination.

This module measures the parts that *are* observable from Bitget's public feed
(`/api/v3/market/orderbook`, catalogued at `agent-sdk/src/generated/catalog.ts:115` — public, no
auth, up to 200 levels) and hands them back to the simulator.

**What can be measured, and what cannot.** An L2 feed publishes the total resting quantity per
price, and nothing about who is in the queue. So:

* **Measurable** — the distribution of level sizes near the touch, how much a level's total moves
  between snapshots as a share of itself, and how often a level grows rather than shrinks. Those
  set the size of `chg` relative to the level, which is what drives the whole apportionment.
* **Not measurable, ever, from this feed** — whether a decrease came from in front of a particular
  resting order or behind it. That is the exact quantity the probability functions estimate, and no
  amount of L2 recording produces it. It needs market-by-order data, which Bitget does not publish,
  or our own orders resting in the real book.

That second bullet is why `eval/standing.py` still records this capability short of OWNED. This
module narrows the gap honestly rather than closing it by assertion: after calibration the queue
experiment runs on a book whose *dynamics* are Bitget's, while the attribution rule remains a stated
assumption swept across regimes.

**The tape only grows by waiting.** One snapshot measures sizes; change requires two separated in
time, and a distribution of changes requires many. So the recorder appends and never rewrites, and
the scheduled cycle calls it — the measurement improves with elapsed time and cannot be hurried,
which is the same property that makes `register/` worth having.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any

TAPE_PATH = Path(__file__).resolve().parents[3] / "data" / "book_tape.jsonl"
CALIBRATION_PATH = Path(__file__).resolve().parents[3] / "data" / "book_calibration.json"

ENDPOINT = "https://api.bitget.com/api/v3/market/orderbook"
"""Public order book. `agent-sdk/src/generated/catalog.ts:115` — auth "public", isWrite false."""

DEPTH = 50
"""Levels recorded per side. The queue question is about resting near the touch; 50 levels is far
past where a passive order of our size would sit, and keeps the tape small enough to commit."""

HORIZONS: tuple[tuple[str, float, float], ...] = (
    ("under_30s", 0.0, 30.0),
    ("30s_to_2min", 30.0, 120.0),
    ("2min_to_10min", 120.0, 600.0),
    ("10min_to_1h", 600.0, 3600.0),
    ("over_1h", 3600.0, float("inf")),
)
"""Elapsed-time buckets for the change distribution, as ``(name, low, high)`` in seconds.

**A change is only meaningful with the horizon it was measured over, and this module used to throw
that away.** `calibrate` walked `pairwise` over snapshots sorted by time and pooled every result
into one median, regardless of whether the two snapshots were 14 seconds or 2.5 hours apart. On the
live tape that is a 147x range of horizons in one number.

The pooled figure it produced — a level moving a median 71.3% of itself — is not a property of the
book. Split by horizon on the same 961 observations it reads 24.0% under 30s, 77.3% from 30s to 2
minutes and 93.5% beyond an hour. The queue simulator apportions change at the timescale of
individual order events, so handing it the pooled number tells it that levels churn far harder than
they do at the horizon it actually operates on.

Nothing here is a new measurement. The observations were always there; they were being averaged
across incommensurable horizons, which is the same error as averaging a daily return with a monthly
one and reporting the mean.
"""

MIN_HORIZON_OBSERVATIONS = 100
"""Observations a horizon needs before its median is quoted as the simulator's parameter.

Not a tuned number and it is not pretending to be: it is the point below which a median over
per-level changes is being read off a handful of levels on a handful of symbols. The horizon rows
are published whatever their count, so a reader can disagree with this line without losing the data.
"""

NEAR_TOUCH = 10
"""Levels counted as "near the touch" when summarising. Beyond this the book is quote stuffing as
much as it is intent, and including it would flatter the level-size distribution."""

MIN_SNAPSHOTS = 2
"""Below this nothing about *change* can be said, and the report says so rather than guessing."""


@dataclass(frozen=True, slots=True)
class Snapshot:
    """One side-by-side look at a real book."""

    taken_at: str
    symbol: str
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "taken_at": self.taken_at, "symbol": self.symbol,
            "bids": [list(level) for level in self.bids],
            "asks": [list(level) for level in self.asks],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Snapshot:
        return cls(
            taken_at=raw["taken_at"], symbol=raw["symbol"],
            bids=tuple((float(p), float(q)) for p, q in raw["bids"]),
            asks=tuple((float(p), float(q)) for p, q in raw["asks"]),
        )


class BookUnavailable(RuntimeError):
    """The venue did not answer with a book. Absence, never a substituted number."""


def fetch_book(symbol: str, *, category: str = "USDT-FUTURES", timeout: float = 20.0) -> Snapshot:
    """One real snapshot, or an exception.

    There is deliberately no fallback. A synthetic book quietly standing in for a real one is the
    single failure that would invalidate everything downstream, and it would be invisible.
    """
    url = f"{ENDPOINT}?category={category}&symbol={symbol}&limit={DEPTH}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise BookUnavailable(f"{symbol}: {exc}") from exc

    if payload.get("code") != "00000":
        raise BookUnavailable(f"{symbol}: {payload.get('code')} {payload.get('msg')}")
    data = payload.get("data") or {}
    # The v3 response abbreviates the sides to "a" and "b"; the documented long names are not what
    # the endpoint actually returns, and reading `bids` was the first thing that failed here.
    bids = data.get("b") or data.get("bids") or []
    asks = data.get("a") or data.get("asks") or []
    if not bids or not asks:
        raise BookUnavailable(f"{symbol}: the book came back one-sided or empty")
    return Snapshot(
        taken_at=datetime.now(UTC).isoformat(), symbol=symbol,
        bids=tuple((float(p), float(q)) for p, q in bids),
        asks=tuple((float(p), float(q)) for p, q in asks),
    )


def record(symbols: tuple[str, ...], *, path: Path = TAPE_PATH) -> dict[str, Any]:
    """Append one snapshot per symbol. Append-only: the tape is the measurement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written, failed = 0, []
    with path.open("a", encoding="utf-8") as handle:
        for symbol in symbols:
            try:
                snapshot = fetch_book(symbol)
            except BookUnavailable as exc:
                failed.append(str(exc))
                continue
            handle.write(json.dumps(snapshot.as_dict()) + "\n")
            written += 1
    return {"recorded": written, "failed": failed, "path": str(path)}


def read_tape(path: Path = TAPE_PATH) -> list[Snapshot]:
    if not path.exists():
        return []
    out: list[Snapshot] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(Snapshot.from_dict(json.loads(line)))
    return out


def _sizes(snapshot: Snapshot) -> list[float]:
    near = list(snapshot.bids[:NEAR_TOUCH]) + list(snapshot.asks[:NEAR_TOUCH])
    return [qty for _price, qty in near if qty > 0]


def _changes(previous: Snapshot, current: Snapshot) -> tuple[list[float], int, int]:
    """Per-level moves between two snapshots of the same symbol, matched on price.

    A price present in one snapshot and absent in the other is skipped rather than treated as a move
    to zero: the level may simply have fallen outside the recorded depth, and counting that as a
    total cancellation would invent the largest changes in the sample.
    """
    before = dict(list(previous.bids[:NEAR_TOUCH]) + list(previous.asks[:NEAR_TOUCH]))
    after = dict(list(current.bids[:NEAR_TOUCH]) + list(current.asks[:NEAR_TOUCH]))
    shares: list[float] = []
    grew = shrank = 0
    for price, prev_qty in before.items():
        new_qty = after.get(price)
        if new_qty is None or prev_qty <= 0:
            continue
        shares.append(abs(new_qty - prev_qty) / prev_qty)
        if new_qty > prev_qty:
            grew += 1
        elif new_qty < prev_qty:
            shrank += 1
    return shares, grew, shrank


def calibrate(*, path: Path = TAPE_PATH) -> dict[str, Any]:
    """What the real book says about the parameters the queue experiment assumes."""
    tape = read_tape(path)
    if not tape:
        return {
            "snapshots": 0,
            "verdict": "no tape on this machine — the queue experiment's parameters are unmeasured",
        }

    sizes = [qty for snapshot in tape for qty in _sizes(snapshot)]
    by_symbol: dict[str, list[Snapshot]] = {}
    for snapshot in tape:
        by_symbol.setdefault(snapshot.symbol, []).append(snapshot)

    shares: list[float] = []
    grew = shrank = 0
    by_horizon: dict[str, list[float]] = {name: [] for name, _, _ in HORIZONS}
    horizon_growth: dict[str, list[int]] = {name: [0, 0] for name, _, _ in HORIZONS}
    for series in by_symbol.values():
        series.sort(key=lambda s: s.taken_at)
        for previous, current in pairwise(series):
            more, up, down = _changes(previous, current)
            shares.extend(more)
            grew += up
            shrank += down
            # The gap is what makes the observation interpretable; it is recorded rather than
            # collapsed into the pooled distribution above.
            gap = (
                datetime.fromisoformat(current.taken_at)
                - datetime.fromisoformat(previous.taken_at)
            ).total_seconds()
            bucket = _horizon_of(gap)
            by_horizon[bucket].extend(more)
            horizon_growth[bucket][0] += up
            horizon_growth[bucket][1] += down

    ordered = sorted(sizes)
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": ENDPOINT,
        "snapshots": len(tape),
        "symbols": sorted(by_symbol),
        "spans_hours": _span_hours(tape),
        "level_size": {
            "count": len(sizes),
            "median": round(median(sizes), 4) if sizes else 0.0,
            "p10": round(_quantile(ordered, 0.10), 4),
            "p90": round(_quantile(ordered, 0.90), 4),
        },
        "paired_snapshots": sum(max(len(v) - 1, 0) for v in by_symbol.values()),
    }

    if len(tape) < MIN_SNAPSHOTS or not shares:
        report["change"] = None
        report["verdict"] = (
            f"{len(tape)} snapshot(s): level sizes are measured, change is not. A distribution of "
            f"depth changes needs snapshots separated in time, and that is a function of how long "
            f"the recorder has run rather than of anything that can be computed now."
        )
        return report

    report["change_by_horizon"] = _by_horizon(by_horizon, horizon_growth)
    report["change_pooled"] = {
        "observations": len(shares),
        "median_share_of_level": round(median(shares), 5),
        "grew": grew,
        "shrank": shrank,
        "warning": (
            "Pooled across every snapshot gap in the tape, which span "
            f"{_gap_range(by_symbol)}. Read `change_by_horizon` instead: this figure averages "
            "observations taken over incommensurable elapsed times and is kept only so the "
            "earlier artefacts remain comparable."
        ),
    }
    usable = _usable_horizon(report["change_by_horizon"])
    report["verdict"] = (
        f"{len(tape)} snapshots over {report['spans_hours']}h: a near-touch level holds a median "
        f"{report['level_size']['median']} contracts. **Change is reported per horizon, not "
        f"pooled** — a level moves a very different share of itself over 30 seconds than over an "
        f"hour, and one median over both is a number about the sampling schedule rather than "
        f"about the book. {usable} The attribution rule — which side of a resting order a "
        f"decrease came from — is not in this data and cannot be, so it stays a swept assumption "
        f"rather than a measurement."
    )
    return report


def _horizon_of(gap_seconds: float) -> str:
    """Which bucket an elapsed gap falls in. Every gap lands in exactly one."""
    for name, low, high in HORIZONS:
        if low <= gap_seconds < high:
            return name
    return HORIZONS[-1][0]


def _by_horizon(
    shares: dict[str, list[float]], growth: dict[str, list[int]]
) -> dict[str, dict[str, Any]]:
    """The change distribution for each horizon, and ``None`` where there is no observation.

    A horizon the tape never sampled reports ``observations: 0`` and a null median rather than a
    zero. A level that was never watched over ten minutes did not hold still for ten minutes.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, low, high in HORIZONS:
        values = sorted(shares[name])
        up, down = growth[name]
        out[name] = {
            "gap_seconds": [low, None if high == float("inf") else high],
            "observations": len(values),
            "median_share_of_level": round(median(values), 5) if values else None,
            "p90_share_of_level": round(_quantile(values, 0.90), 5) if values else None,
            "grew": up,
            "shrank": down,
            "share_growing": round(up / (up + down), 4) if up + down else None,
        }
    return out


def _usable_horizon(by_horizon: dict[str, dict[str, Any]]) -> str:
    """Name the shortest horizon with enough observations to state, or say none qualifies.

    The queue simulator apportions change per order event, so the shortest sampled horizon is the
    closest available proxy and the long ones are the least relevant — the opposite of what the
    pooled median emphasised, since the long gaps carried the largest moves.
    """
    for name, _, _ in HORIZONS:
        row = by_horizon[name]
        if row["observations"] >= MIN_HORIZON_OBSERVATIONS:
            return (
                f"The shortest horizon with at least {MIN_HORIZON_OBSERVATIONS} observations is "
                f"`{name}`, where a level moves a median "
                f"{row['median_share_of_level'] * 100:.1f}% of itself — that is the figure the "
                f"simulator should take, being nearest the per-event timescale it models."
            )
    return (
        f"No horizon yet carries {MIN_HORIZON_OBSERVATIONS} observations, so no per-horizon change "
        f"figure is stated. That is a function of how long the recorder has run."
    )


def _gap_range(by_symbol: dict[str, list[Snapshot]]) -> str:
    """The shortest and longest gap actually present, so the pooling warning is concrete."""
    gaps: list[float] = []
    for series in by_symbol.values():
        for previous, current in pairwise(series):
            gaps.append(
                (
                    datetime.fromisoformat(current.taken_at)
                    - datetime.fromisoformat(previous.taken_at)
                ).total_seconds()
            )
    if not gaps:
        return "no measurable gaps"
    return f"{min(gaps):.0f}s to {max(gaps):.0f}s"


def _span_hours(tape: list[Snapshot]) -> float:
    stamps = sorted(datetime.fromisoformat(s.taken_at) for s in tape)
    if len(stamps) < 2:
        return 0.0
    return round((stamps[-1] - stamps[0]).total_seconds() / 3600, 3)


def _quantile(ordered: list[float], q: float) -> float:
    """Nearest-rank quantile. No interpolation: with a handful of samples an interpolated tail
    reads as more precision than the sample contains."""
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def measured_level_sizes(*, path: Path = TAPE_PATH) -> tuple[float, float] | None:
    """The (p10, p90) near-touch level size the simulator should draw from, or ``None``.

    ``None`` is the honest answer when no tape exists, and the caller keeps its stated default
    rather than receiving a plausible-looking number with nothing behind it.
    """
    sizes = sorted(qty for snapshot in read_tape(path) for qty in _sizes(snapshot))
    if len(sizes) < 20:
        return None
    return _quantile(sizes, 0.10), _quantile(sizes, 0.90)


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="record and calibrate the real order book")
    parser.add_argument("--record", action="store_true", help="append one snapshot per symbol")
    parser.add_argument("--symbols", default="", help="comma separated; defaults to the rTokens")
    args = parser.parse_args()

    if args.record:
        from argus.market.bitget import RTOKEN_SYMBOLS

        symbols = tuple(s for s in args.symbols.split(",") if s) or tuple(RTOKEN_SYMBOLS)
        result = record(symbols)
        print(f"recorded {result['recorded']} snapshot(s) to {result['path']}")
        for failure in result["failed"]:
            print(f"  unavailable: {failure}")

    report = calibrate()
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  {report['verdict']}")
    print(f"\nwritten to {CALIBRATION_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "CALIBRATION_PATH",
    "DEPTH",
    "ENDPOINT",
    "MIN_SNAPSHOTS",
    "NEAR_TOUCH",
    "TAPE_PATH",
    "BookUnavailable",
    "Snapshot",
    "calibrate",
    "fetch_book",
    "measured_level_sizes",
    "read_tape",
    "record",
]
