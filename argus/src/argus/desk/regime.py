"""Regime segmentation from the shape of the series, not from a volatility threshold.

`strategies/track1_suite.py:206-215` is the incumbent: *fast volatility below slow volatility means
low-volatility regime, otherwise high*. One threshold, one dimension, two states, and the switch
fires whenever a 24-hour window crosses a 240-hour one. It is honest — the docstring says "regime is
measured, not predicted" — and it can only ever notice that the market got louder.

A regime change is not only a change in amplitude. It is a change in the **shape** of what the
series does: a grind becomes a chop, a mean-reverting range becomes a trend, and realised volatility
can be identical on both sides of that boundary.

**FLUSS answers the shape question, and it is a by-product of machinery ARGUS already has.**
`desk/shapematch.py` computes z-normalised Euclidean distance between windows. Extend that to every
window's *nearest neighbour* and you have the matrix profile index; then Gharghabi et al.'s
observation (ICDM 2017, `Segmentation_ICDM.pdf`) is that a regime boundary is a place **few arcs
cross**. Inside a regime, windows find their nearest neighbours nearby; across a boundary they do
not, because the other side looks different. Counting how many nearest-neighbour arcs span each
index, and dividing by how many a homogeneous series would have, gives the corrected arc curve. Its
minima are the boundaries.

**Read before written**, per the standing rule. Two reference implementations, both read:

* `matrix-profile-foundation/matrixprofile/algorithms/regimes.py:66-91` (Apache-2.0) — the arc count
  (`nnmark[small+1] += 1; nnmark[large] -= 1`, then a cumulative sum), the parabolic idealised curve
  (`regimes.py:16-38`: height `n/2`, so `y = -a(i - n/2)^2 + n/2`), the clip to 1, and the head/tail
  correction.
* `stumpy/floss.py:167-183` — the same CAC with two differences: its idealised curve is a **fitted
  beta distribution** (`floss.py:111`, needs scipy) rather than a parabola, and it corrects
  `L * excl_factor` of head and tail where matrixprofile corrects only `w`.

ARGUS follows **matrixprofile's parabola** — it is deterministic, needs no scipy, and the original
paper's Table I defines the idealised curve analytically — and **stumpy's wider head/tail
correction**, because the narrower one leaves the first and last window looking like boundaries when
they are only edges. Both choices are departures from one reference and matches to the other, which
is why both are named here.

**What it does not do.** It does not predict the next regime, and it does not label one. A boundary
is a statement that the series before and after it are shaped differently; calling one "risk-on"
would be a story laid over an index. The report therefore names each segment by its own measured
statistics — length, realised volatility, drift — and leaves the interpretation to the reader.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from statistics import median
from typing import Any

from argus.desk.shapematch import znormalise

MIN_BARS = 240
"""Fewest bars before a segmentation is attempted.

FLUSS needs each regime to repeat internally — the authors say plainly that it "is not able to
segment single gesture patterns" (`regimes.py:105-107`) — so a series that barely contains two
windows cannot be segmented, only split.
"""

EXCLUSION_FACTOR = 5
"""Regime exclusion zone, in window lengths. `regimes.py:127` and `floss.py:142` agree on 5.

The authors' stated reason is the assumption that regimes repeat; two boundaries closer than five
windows are usually the same boundary found twice.
"""

TRIVIAL_MATCH_DENOM = 4
"""The matrix profile's own exclusion zone, `m / 4` (`stumpy/config.py:19`).

Deliberately *narrower* than `desk/shapematch.py`'s full window. The two exclusions answer different
questions: there, no two **reported** analogues may share a bar, because a human reads them as
separate events; here, the arc count wants every window's genuine nearest neighbour, and over-wide
exclusion would push arcs outward and flatten the very curve the boundaries live in.
"""


class RegimeError(ValueError):
    """Raised rather than returning a segmentation of a series too short to segment."""


def matrix_profile_index(values: Sequence[float], window: int) -> list[int]:
    """For each window, the index of its nearest non-trivial neighbour.

    Brute force, O(n^2 m). At 1,400 hourly bars and a 24-bar window that is about 24 million
    multiply-adds — seconds in pure Python, and the alternative is numpy plus numba to save them.
    The distance itself is the same z-normalised Euclidean that `desk/shapematch.py` uses, computed
    here from the correlation identity ``d^2 = 2m(1 - rho)`` (`stumpy/core.py:1118`) so the inner
    loop is one dot product rather than a subtraction and a square per bar.
    """
    if window < 3:
        raise RegimeError(f"a {window}-bar window has no shape to match")
    count = len(values) - window + 1
    if count < window * 2:
        raise RegimeError(
            f"{len(values)} bar(s) give {count} window(s), too few for a {window}-bar profile"
        )
    normalised = [znormalise(list(values[i:i + window])) for i in range(count)]
    exclusion = max(1, window // TRIVIAL_MATCH_DENOM)

    index: list[int] = []
    for i in range(count):
        best_rho = -2.0
        best_j = i
        left = normalised[i]
        for j in range(count):
            if abs(i - j) <= exclusion:
                continue
            right = normalised[j]
            rho = sum(a * b for a, b in zip(left, right, strict=True)) / window
            if rho > best_rho:
                best_rho = rho
                best_j = j
        index.append(best_j)
    return index


def arc_counts(index: Sequence[int]) -> list[float]:
    """How many nearest-neighbour arcs cross each position.

    `regimes.py:69-76`: each arc marks +1 just after its lower endpoint and -1 at its upper one, and
    the running sum is the number of arcs spanning that index. A boundary is where that number
    collapses, because windows on either side stop matching each other.
    """
    n = len(index)
    marks = [0.0] * (n + 1)
    for i, target in enumerate(index):
        low, high = min(i, target), max(i, target)
        marks[low + 1] += 1.0
        marks[high] -= 1.0
    running = 0.0
    out: list[float] = []
    for i in range(n):
        running += marks[i]
        out.append(running)
    return out


def idealised_arc(n: int, i: int) -> float:
    """The parabola a homogeneous series of this length would produce (`regimes.py:16-38`).

    Height and centre are both ``n / 2``, so the curve is ``-a(i - n/2)^2 + n/2`` with
    ``a = height / (n/2)^2``. Dividing the observed count by this is what makes the curve comparable
    across positions: the middle of any series has far more arcs crossing it than the ends do, and
    without the correction every segmentation would put its boundaries at the edges.
    """
    if n <= 0:
        raise RegimeError("an idealised arc curve needs a positive length")
    height = n / 2
    centre = n / 2
    a = height / (n / 2) ** 2
    return -(a * (i - centre) ** 2) + height


def corrected_arc_curve(
    index: Sequence[int], window: int, *, exclusion_factor: int = EXCLUSION_FACTOR,
) -> list[float]:
    """Observed arcs over idealised arcs, clipped to 1, with the ends pinned.

    The ends are pinned to 1 (never a boundary) over ``window * exclusion_factor`` — stumpy's width
    (`floss.py:180-182`), not matrixprofile's narrower ``w``. A window at the very start has no
    neighbours to its left and therefore few arcs crossing it, which is an edge effect and not a
    regime change; the narrow correction lets that artefact win.
    """
    n = len(index)
    if n == 0:
        raise RegimeError("an empty profile index has no arc curve")
    counts = arc_counts(index)
    curve: list[float] = []
    for i in range(n):
        ideal = idealised_arc(n, i)
        value = counts[i] / ideal if ideal > 0 else 1.0
        curve.append(min(1.0, max(0.0, value)))
    pin = min(n, window * max(1, exclusion_factor))
    for i in range(pin):
        curve[i] = 1.0
        curve[n - 1 - i] = 1.0
    return curve


def find_boundaries(
    curve: Sequence[float], *, count: int, window: int,
    exclusion_factor: int = EXCLUSION_FACTOR,
) -> list[int]:
    """The ``count`` deepest minima of the arc curve, each excluding its neighbourhood.

    `regimes.py:133-145` and `floss.py:_rea` do the same thing: take the argmin, blank
    ``window * 5`` either side, repeat. Without the blanking the top three boundaries are three
    adjacent bars of one dip.
    """
    if count < 1:
        raise RegimeError("at least one boundary must be requested")
    working = list(curve)
    n = len(working)
    zone = window * max(1, exclusion_factor)
    found: list[int] = []
    for _ in range(count):
        best = min(range(n), key=lambda i: working[i])
        if working[best] >= 1.0:
            break  # everything left is pinned or flat: there is no further boundary to find
        found.append(best)
        for i in range(max(0, best - zone), min(n, best + zone)):
            working[i] = float("inf")
    return sorted(found)


@dataclass(frozen=True, slots=True)
class Segment:
    """One stretch between boundaries, described by what it measurably did."""

    start: datetime
    end: datetime
    bars: int
    volatility_bps: float
    """Median absolute bar-to-bar move inside the segment."""

    drift_pct: float
    arc_value: float | None
    """The corrected arc value at this segment's opening boundary. Lower means a cleaner break."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "bars": self.bars,
            "volatility_bps": round(self.volatility_bps, 3),
            "drift_pct": round(self.drift_pct, 4),
            "arc_value": None if self.arc_value is None else round(self.arc_value, 4),
        }


@dataclass(frozen=True, slots=True)
class RegimeReport:
    """Where the series changed shape, and what each stretch looked like."""

    symbol: str
    window: int
    bars: int
    segments: tuple[Segment, ...]
    boundaries: tuple[datetime, ...]
    threshold_agrees: bool | None
    """Whether the incumbent fast-vs-slow volatility rule flips at the same places. ``None`` when
    there were not enough bars to evaluate it."""

    @property
    def loudest(self) -> Segment | None:
        return max(self.segments, key=lambda s: s.volatility_bps) if self.segments else None

    @property
    def quietest(self) -> Segment | None:
        return min(self.segments, key=lambda s: s.volatility_bps) if self.segments else None

    @property
    def verdict(self) -> str:
        if len(self.segments) < 2:
            return (
                f"No regime boundary found in {self.bars} bar(s) of {self.symbol}. The series is "
                f"one regime by this measure, which is a finding and not a failure"
            )
        loud, quiet = self.loudest, self.quietest
        assert loud is not None and quiet is not None
        ratio = loud.volatility_bps / quiet.volatility_bps if quiet.volatility_bps > 0 else 0.0
        agreement = (
            "the incumbent fast-vs-slow volatility rule flips at the same places"
            if self.threshold_agrees
            else "the incumbent fast-vs-slow volatility rule does not flip at these points, so "
                 "this segmentation is finding something amplitude alone does not"
        )
        return (
            f"{len(self.segments)} regime(s) in {self.bars} bar(s) of {self.symbol}. The loudest "
            f"runs at {loud.volatility_bps:.1f}bps a bar against {quiet.volatility_bps:.1f} in the "
            f"quietest ({ratio:.1f}x), and {agreement}"
        )

    def render(self) -> str:
        lines = [
            f"REGIME SEGMENTATION — {self.symbol}, {self.window}-bar window, {self.bars} bars",
            "",
            f"  {'from':16} {'to':16} {'bars':>6} {'vol bps':>9} {'drift':>8} {'arc':>7}",
        ]
        for segment in self.segments:
            arc = "—" if segment.arc_value is None else f"{segment.arc_value:.3f}"
            lines.append(
                f"  {segment.start:%Y-%m-%d %H:%M} {segment.end:%Y-%m-%d %H:%M} "
                f"{segment.bars:6d} {segment.volatility_bps:9.1f} {segment.drift_pct:+7.2f}% "
                f"{arc:>7}"
            )
        lines += ["", f"  {self.verdict}"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "symbol": self.symbol,
            "window": self.window,
            "bars": self.bars,
            "boundaries": [b.isoformat() for b in self.boundaries],
            "threshold_agrees": self.threshold_agrees,
            "segments": [s.as_dict() for s in self.segments],
            "verdict": self.verdict,
        }


def _volatility_bps(values: Sequence[float]) -> float:
    moves = [
        abs(b / a - 1.0) * 10_000
        for a, b in pairwise(values)
        if a > 0
    ]
    return median(moves) if moves else 0.0


def _threshold_flips(values: Sequence[float], *, fast: int = 24, slow: int = 240) -> list[int]:
    """Where `strategies/track1_suite.py:206-215`'s rule changes its mind.

    Reimplemented here rather than imported because that function answers per-bar with a weight and
    this needs the transitions; the arithmetic is the same and the comparison would be meaningless
    if it were not.
    """
    flips: list[int] = []
    previous: bool | None = None
    for i in range(slow, len(values)):
        quick = _volatility_bps(values[i - fast:i + 1])
        long_run = _volatility_bps(values[i - slow:i + 1])
        if long_run <= 0:
            continue
        state = quick < long_run
        if previous is not None and state != previous:
            flips.append(i)
        previous = state
    return flips


def segment(
    series: Sequence[tuple[datetime, float]], *, symbol: str, window: int = 24,
    regimes: int = 3, exclusion_factor: int = EXCLUSION_FACTOR,
) -> RegimeReport:
    """Segment a price series into regimes by the shape of its nearest-neighbour arcs."""
    if len(series) < MIN_BARS:
        raise RegimeError(
            f"{len(series)} bar(s) is below the {MIN_BARS} needed to segment anything"
        )
    stamps = [t for t, _ in series]
    values = [v for _, v in series]
    index = matrix_profile_index(values, window)
    curve = corrected_arc_curve(index, window, exclusion_factor=exclusion_factor)
    cuts = find_boundaries(
        curve, count=max(0, regimes - 1), window=window, exclusion_factor=exclusion_factor,
    )

    edges = [0, *cuts, len(curve)]
    segments: list[Segment] = []
    for position, (start, stop) in enumerate(pairwise(edges)):
        if stop - start < 2:
            continue
        # A boundary at window index c is a change at BAR c — the window's first bar — so segments
        # run [start, stop) in bar space and do not overlap. Only the last one carries the trailing
        # window, because those bars belong to no later segment. The first version added `window-1`
        # to every segment's end and produced ranges that appeared to overlap by a day, which reads
        # as an off-by-one in the segmentation rather than as the tail of a sliding window.
        last = position == len(edges) - 2
        chunk = values[start:] if last else values[start:stop]
        opened = curve[start] if position > 0 else None
        segments.append(Segment(
            start=stamps[start],
            end=stamps[len(values) - 1 if last else stop - 1],
            bars=len(chunk),
            volatility_bps=_volatility_bps(chunk),
            drift_pct=(chunk[-1] / chunk[0] - 1.0) * 100 if chunk and chunk[0] > 0 else 0.0,
            arc_value=opened,
        ))

    agrees: bool | None = None
    if len(values) > 240 + 24:
        flips = _threshold_flips(values)
        # "Agrees" means every FLUSS boundary has a threshold flip within one window of it. A rule
        # that flips constantly would otherwise agree with anything by coincidence.
        agrees = bool(cuts) and all(
            any(abs(flip - cut) <= window for flip in flips) for cut in cuts
        )

    return RegimeReport(
        symbol=symbol, window=window, bars=len(values), segments=tuple(segments),
        boundaries=tuple(stamps[c] for c in cuts), threshold_agrees=agrees,
    )


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import sys
    from pathlib import Path

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from argus.market.history import CandleType, fetch_range

    parser = argparse.ArgumentParser(description="where did this series change shape?")
    parser.add_argument("--symbol", default="NVDAUSDT")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--interval", default="1H")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--regimes", type=int, default=3)
    args = parser.parse_args()

    bars = fetch_range(
        args.symbol, days=args.days, interval=args.interval, candle_type=CandleType.MARKET,
    )
    report = segment(
        [(c.ts, float(c.close)) for c in bars], symbol=args.symbol, window=args.window,
        regimes=args.regimes,
    )
    print(report.render())
    out = Path(__file__).resolve().parents[3] / "data" / "regimes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "EXCLUSION_FACTOR",
    "MIN_BARS",
    "TRIVIAL_MATCH_DENOM",
    "RegimeError",
    "RegimeReport",
    "Segment",
    "arc_counts",
    "corrected_arc_curve",
    "find_boundaries",
    "idealised_arc",
    "matrix_profile_index",
    "segment",
]
