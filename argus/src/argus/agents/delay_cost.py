"""What deliberating costs, read from the book the desk is about to trade — not from a constant.

``agents/meta_pm.py::deliberation_cost_bps`` prices a thinking budget as
``0.45 * sqrt(t / year) * 10000 * depth``, with ``depth`` = 1 in RTH, 2 in extended hours and 3
overnight and at weekends. ``eval/general_delib_comparison.py`` is built to referee that charge
against what the mid of Bitget's stock perpetuals actually does over 3, 8 and 40 seconds, against
these estimators and general-purpose forecasters. It has not published an artefact yet, so its
result is NOT VERIFIED and none of its numbers are quoted here. What does not need a run, read at
source:

* **the depth multiplier is in the wrong term.** A thin book is crossed further — that is
  slippage — but how far the price drifts while the model thinks is volatility's alone, the same
  double count `lui/research.py::_waiting_cost` already refuses;
* **it cannot tell a calm minute from a busy one**, because the only input it varies with is the
  session.

What replaces it is the textbook, and it carries no licence at all.

* :func:`empirical_delay_cost_bps` — the mean absolute ``h``-second move actually seen over the
  trailing window of one-second mids. No scaling law and no distribution assumed.
* :func:`bar_delay_cost_bps` — for a caller that only has bars (Bitget's one-minute candles):
  root-mean-square bar return, scaled by ``sqrt(delay / bar)`` and by ``sqrt(2 / pi)``, the mean
  absolute value of a Gaussian with that standard deviation. The random-walk scaling ARGUS already
  used, fed a measured volatility instead of an assumed one, and without the session multiplier.

Both return an *expected magnitude*, as the original did: the model's decision is correct or not
independently of the drift, and a charge that netted drift against itself would be zero.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import Decimal
from itertools import pairwise

GAUSSIAN_MEAN_ABS = math.sqrt(2.0 / math.pi)
"""``E|Z|`` for a standard normal. The original charge used ``sigma * sqrt(t)`` — the standard
deviation of the move — and called it the expected magnitude, which overstates it by 25%."""

MIN_WINDOW_MOVES = 60
"""Fewer ``h``-second moves than this and the empirical mean is too noisy to charge."""
MIN_BAR_RETURNS = 20


class DelayCostError(ValueError):
    """The book given cannot price a delay honestly."""


def _positive(values: Sequence[float], what: str) -> None:
    if any(not (v > 0 and math.isfinite(v)) for v in values):
        raise DelayCostError(f"{what} must be positive, finite prices")


def empirical_delay_cost_bps(mids: Sequence[float], *, delay_s: int) -> Decimal:
    """Mean ``|ln mid(s + delay) - ln mid(s)|`` in bps over every ``s`` in ``mids``.

    ``mids`` is a series sampled once a second, oldest first, ending at the decision instant —
    thirty minutes of it is what was measured. Nothing after the last element is read.
    """
    if delay_s <= 0:
        raise DelayCostError(f"delay_s={delay_s} must be a positive whole number of seconds")
    _positive(mids, "mids")
    count = len(mids) - delay_s
    if count < MIN_WINDOW_MOVES:
        raise DelayCostError(
            f"{len(mids)} one-second mids give {max(count, 0)} {delay_s}s moves; "
            f"at least {MIN_WINDOW_MOVES} are needed"
        )
    logs = [math.log(v) for v in mids]
    total = sum(abs(logs[i + delay_s] - logs[i]) for i in range(count))
    return Decimal(str(total / count * 1e4))


def bar_delay_cost_bps(closes: Sequence[float], *, bar_s: int, delay_s: float) -> Decimal:
    """``sqrt(2/pi) * rms(bar log return) * sqrt(delay / bar)`` in bps.

    ``closes`` are consecutive bar closes, oldest first, ending at or before the decision instant.
    """
    if bar_s <= 0 or delay_s <= 0:
        raise DelayCostError(f"bar_s={bar_s} and delay_s={delay_s} must be positive")
    _positive(closes, "closes")
    if len(closes) - 1 < MIN_BAR_RETURNS:
        raise DelayCostError(
            f"{len(closes) - 1} bar returns; at least {MIN_BAR_RETURNS} are needed"
        )
    returns = [math.log(b / a) for a, b in pairwise(closes)]
    sigma = math.sqrt(sum(r * r for r in returns) / len(returns))
    return Decimal(str(GAUSSIAN_MEAN_ABS * sigma * math.sqrt(delay_s / bar_s) * 1e4))


__all__ = [
    "GAUSSIAN_MEAN_ABS",
    "MIN_BAR_RETURNS",
    "MIN_WINDOW_MOVES",
    "DelayCostError",
    "bar_delay_cost_bps",
    "empirical_delay_cost_bps",
]
