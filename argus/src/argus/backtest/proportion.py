"""Win rates and hit rates said with their uncertainty, or not said at all.

A rate like "won 61% of the time (33 cases)" reads as a finding. With 33 cases its 95% interval
runs from about 44% to 76%, which includes a coin flip, and the honest sentence says so. This module
is the one place a count of successes becomes words.

Taken from LuxAlgo's edge-stats (`packages/core/src/stats/stats.ts`, MIT,
github.com/LuxAlgo/edge-stats, read 2026-10-03): the Wilson score interval for a binomial
proportion, the minimum-sample floors (below 30 a result carries a low-sample warning, below 10 no
estimate is shown — `packages/core/src/config.ts:41-47`), and the first-half / second-half
stability split, where the two halves agree when their Wilson intervals overlap. Ours adds the
sentence: :func:`rate_phrase` renders the estimate, the interval, whether the interval excludes
50%, and the floors' verdict in one clause, so no caller can print a bare percentage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

Z95: Final = 1.959963984540054
"""The two-sided 95% normal quantile edge-stats uses."""

WARN_FLOOR: Final = 30
"""Below this many cases a rate is shown with a low-sample warning (edge-stats default)."""

REFUSE_FLOOR: Final = 10
"""Below this many cases no rate is shown at all (edge-stats default)."""


@dataclass(frozen=True, slots=True)
class Wilson:
    estimate: float
    lo: float
    hi: float

    def excludes(self, value: float) -> bool:
        """Whether the interval leaves ``value`` out (e.g. 0.5: better or worse than a coin)."""
        return value < self.lo or value > self.hi


def wilson(k: int, n: int, z: float = Z95) -> Wilson | None:
    """The Wilson score interval for ``k`` successes in ``n`` trials, or ``None`` for ``n = 0``."""
    if n <= 0:
        return None
    if k < 0 or k > n:
        raise ValueError(f"impossible counts: k={k} n={n}")
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    # at k = 0 or k = n the bound is exactly 0 or 1; floating point would leave 1e-16 off it
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return Wilson(estimate=p, lo=lo, hi=hi)


@dataclass(frozen=True, slots=True)
class StabilitySplit:
    first: Wilson | None
    second: Wilson | None
    first_n: int
    second_n: int

    @property
    def agree(self) -> bool | None:
        """The halves' intervals overlap; ``None`` when a half is empty."""
        if self.first is None or self.second is None:
            return None
        return self.first.lo <= self.second.hi and self.second.lo <= self.first.hi


def stability_split(outcomes: list[bool]) -> StabilitySplit:
    """The rate in the first and second half of ``outcomes`` (in time order)."""
    mid = len(outcomes) // 2
    head, tail = outcomes[:mid], outcomes[mid:]
    return StabilitySplit(
        first=wilson(sum(head), len(head)),
        second=wilson(sum(tail), len(tail)),
        first_n=len(head),
        second_n=len(tail),
    )


def rate_phrase(k: int, n: int, *, noun: str = "cases") -> str:
    """``k`` of ``n`` as words a reader cannot misread.

    - fewer than :data:`REFUSE_FLOOR` cases: the count only, and that it is too few for a rate;
    - otherwise the rate, its 95% interval, and whether that interval excludes 50%;
    - fewer than :data:`WARN_FLOOR` cases: marked as a small sample.
    """
    if n < REFUSE_FLOOR:
        return f"{k} of {n} {noun} — too few to state a rate"
    w = wilson(k, n)
    assert w is not None
    coin = (
        "better than a coin flip"
        if w.lo > 0.5
        else "worse than a coin flip"
        if w.hi < 0.5
        else "not distinguishable from a coin flip"
    )
    small = "; a small sample" if n < WARN_FLOOR else ""
    return (
        f"{w.estimate:.0%} ({k} of {n} {noun}; 95% range {w.lo:.0%} to {w.hi:.0%}, {coin}{small})"
    )


def stability_phrase(outcomes: list[bool]) -> str | None:
    """The first-half / second-half rates and whether they agree, or ``None`` when either half is
    below :data:`REFUSE_FLOOR`."""
    split = stability_split(outcomes)
    if split.first_n < REFUSE_FLOOR or split.second_n < REFUSE_FLOOR:
        return None
    assert split.first is not None and split.second is not None
    verdict = "consistent" if split.agree else "not consistent — the rate moved between halves"
    return (
        f"first half {split.first.estimate:.0%} of {split.first_n}, second half "
        f"{split.second.estimate:.0%} of {split.second_n}: {verdict}"
    )


__all__ = [
    "REFUSE_FLOOR",
    "WARN_FLOOR",
    "StabilitySplit",
    "Wilson",
    "rate_phrase",
    "stability_phrase",
    "stability_split",
    "wilson",
]
