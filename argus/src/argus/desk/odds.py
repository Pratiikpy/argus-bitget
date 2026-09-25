"""How often a contract finished higher over a horizon: the answer to "will X be higher?" that is
measured rather than predicted.

"Will MSTR be higher in 48 hours?" has no honest point answer, and the console refused it outright
until 2026-09-25 while a rival desk (optic-bitget) answered the same kind of thesis with a judge's
probability. What can be said without inventing anything is the contract's own record: over every
past window of that length, how often it finished higher, how wide that share's interval is once
overlapping windows are counted as the few independent draws they are, how often the move cleared
the cost of holding it, and whether windows that followed a move like the last one behaved any
differently.

The weekend variant is the one question stock perpetuals raise that crypto does not: the anchor is
shut from Friday's close to Monday's open, and the perpetual keeps trading. Its record is the
Friday-to-Monday windows alone. The weekend framing comes from baserate (Jayanng, a Season 2 desk
built around weekend episodes); its code is proprietary and none of it is used here, only the
question.

Every share carries a Wilson interval on the independent count. Overlapping windows share most of
their path — 480 daily closes give 478 two-day windows but only 239 independent ones — and an
interval computed on the raw count would claim a precision the data does not have.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

WILSON_Z = 1.96
MIN_INDEPENDENT = 8
"""Below this many independent windows the record is called thin, the same floor the analogue
answer uses (`lui/research.MIN_INDEPENDENT_EPISODES`)."""


def wilson(successes: float, n: float, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for a proportion; accepts a fractional effective count."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


@dataclass(frozen=True, slots=True)
class Odds:
    """The record over one horizon. Moves are in basis points; shares are 0-1."""

    windows: int
    independent: float
    higher_share: float
    higher_low: float
    higher_high: float
    median_bps: float
    p10_bps: float
    p90_bps: float
    cost_bps: float
    side: str
    cleared_share: float
    """Share of windows in which the move in ``side``'s direction exceeded ``cost_bps``."""
    last_move_bps: float | None
    after_like_share: float | None
    """Share finishing higher among windows whose preceding window moved the same way as the most
    recent one. None when there is no preceding window to read."""
    after_like_windows: int
    after_like_low: float | None
    after_like_high: float | None
    span_days: int
    adverse_p90_bps: float | None = None
    """The move against ``side`` that 90% of past windows stayed inside, measured on the bars'
    lows (a long) or highs (a short) rather than closes: where a stop is still inside ordinary
    noise. None when the bars carry no intrawindow extremes."""

    @property
    def after_like_differs(self) -> bool:
        """Whether the conditional share's own interval excludes the unconditional share."""
        return (self.after_like_low is not None and self.after_like_high is not None
                and not self.after_like_low <= self.higher_share <= self.after_like_high)

    @property
    def coin_flip(self) -> bool:
        return self.higher_low <= 0.5 <= self.higher_high

    @property
    def thin(self) -> bool:
        return self.independent < MIN_INDEPENDENT

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _moves(closes: list[tuple[datetime, float]], pairs: list[tuple[int, int]]) -> list[float]:
    return [(closes[j][1] / closes[i][1] - 1) * 10_000 for i, j in pairs
            if closes[i][1] > 0 and closes[j][1] > 0]


def directional_odds(closes: list[tuple[datetime, float]], horizon_bars: int, *,
                     cost_bps: float, side: str = "long", weekend: bool = False,
                     extremes: list[tuple[float, float]] | None = None) -> Odds | None:
    """The record of ``horizon_bars``-long windows over ``closes`` (oldest first).

    ``weekend`` reads daily closes stamped with their CLOSE time and keeps only the windows from a
    Friday close to the Monday close three days later, so ``horizon_bars`` is ignored. Bitget's
    daily candles run 16:00 to 16:00 UTC (midnight UTC+8; checked against the live endpoint
    2026-09-25), so the window is Friday 16:00 to Monday 16:00 UTC: it spans the whole US
    closure (Friday 20:00/21:00 to Monday 13:30/14:30 UTC) plus a few trading hours either side,
    which is as close as daily bars get. ``cost_bps`` is the hurdle a position in ``side``'s
    direction must clear: round-trip fees plus the funding it pays over the horizon. ``extremes``
    holds each bar's (low, high), aligned with ``closes``; with it the adverse excursion is
    measured.
    """
    if len(closes) < 3:
        return None
    if weekend:
        index = {ts.date(): k for k, (ts, _) in enumerate(closes)}
        pairs = []
        for i, (ts, _) in enumerate(closes):
            if ts.weekday() == 4:
                j = index.get((ts + timedelta(days=3)).date())
                if j is not None:
                    pairs.append((i, j))
        independent_per_window = 1.0
    else:
        h = max(1, horizon_bars)
        pairs = [(i, i + h) for i in range(len(closes) - h)]
        independent_per_window = 1.0 / h
    moves = _moves(closes, pairs)
    if len(moves) < 2:
        return None
    n_eff = len(moves) * independent_per_window
    higher = sum(1 for m in moves if m > 0) / len(moves)
    low, high = wilson(higher * n_eff, n_eff)
    if side == "short":
        cleared = sum(1 for m in moves if -m > cost_bps) / len(moves)
    else:
        cleared = sum(1 for m in moves if m > cost_bps) / len(moves)

    last_move: float | None = None
    after_share: float | None = None
    after_n = 0
    after_low: float | None = None
    after_high: float | None = None
    if not weekend:
        h = max(1, horizon_bars)
        if len(closes) > h and closes[-1 - h][1] > 0:
            last_move = (closes[-1][1] / closes[-1 - h][1] - 1) * 10_000
            like = [(closes[i + h][1] / closes[i][1] - 1)
                    for i in range(h, len(closes) - h)
                    if closes[i - h][1] > 0 and closes[i][1] > 0
                    and ((closes[i][1] / closes[i - h][1] - 1) > 0) == (last_move > 0)]
            after_n = len(like)
            if like:
                after_share = sum(1 for m in like if m > 0) / len(like)
                after_low, after_high = wilson(after_share * len(like) / h, len(like) / h)
    adverse: float | None = None
    if extremes is not None and len(extremes) == len(closes):
        worst = []
        for i, j in pairs:
            start = closes[i][1]
            window = extremes[i + 1: j + 1]
            if start <= 0 or not window:
                continue
            if side == "short":
                worst.append((max(h for _, h in window) / start - 1) * 10_000)
            else:
                worst.append((min(lo for lo, _ in window) / start - 1) * 10_000)
        if len(worst) >= 2:
            adverse = _quantile(worst, 0.9 if side == "short" else 0.1)
    span = max(1, (closes[-1][0] - closes[0][0]).days)
    return Odds(
        windows=len(moves), independent=round(n_eff, 1), higher_share=higher, higher_low=low,
        higher_high=high, median_bps=_quantile(moves, 0.5), p10_bps=_quantile(moves, 0.1),
        p90_bps=_quantile(moves, 0.9), cost_bps=cost_bps, side=side, cleared_share=cleared,
        last_move_bps=last_move, after_like_share=after_share, after_like_windows=after_n,
        after_like_low=after_low, after_like_high=after_high,
        span_days=span, adverse_p90_bps=adverse,
    )
