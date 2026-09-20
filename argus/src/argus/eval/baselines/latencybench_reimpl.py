"""An independent, clean-room reimplementation of LatencySensitiveBench's linear price-decay
model — NOT vendored, because there is no license to vendor under.

``eval/standing.py``'s "Deliberation priced as a trading cost" capability previously read the
entire 88-repo corpus and found nothing that prices model inference latency as a real cost — every
harness treats it as free. A fresh, targeted search (2026-09-16) found one genuine exception:
``HaoKang-Timmy/LatencySensitiveBench`` (NeurIPS 2025 poster, "Win Fast or Lose Slow",
arxiv.org/abs/2505.19481), whose ``HFTBench/Environment/TradingEnv.py::_apply_linear_decay``
(commit ``abceb6a374a94fe1b1ea41251ce64d21dbaf6647``, read in full) makes an agent's achievable
execution price degrade toward the day's average price as its own LLM inference delay grows toward
a ``decay_window`` (1.5-2s) — a real, working mechanism, confirmed by reading and running it.

**Why this file exists instead of a vendored copy.** ``gh repo view HaoKang-Timmy/
LatencySensitiveBench --json licenseInfo`` returns ``{"licenseInfo": null}`` — no LICENSE file, so
this project's established "vendor, byte-verify, pin a hash" playbook (used for every other named
specialist baseline in this package) does not apply; copying their file would be redistributing
code without a grant to do so. What follows instead is an independent implementation of the same
described MATHEMATICAL idea (linear interpolation from a price toward a reference price as a delay
approaches a cap, then flat beyond it) — different variable names, different structure, no text
copied — verified against REFERENCE OUTPUT VECTORS computed by running their own real, unmodified
method once (not redistributing it, just calling it locally to check this file's own numbers
against ground truth)::

    python3 -c "
    import sys; sys.path.insert(0, 'HFTBench/Environment')
    from TradingEnv import TradeMarket
    class Fake:
        decay_window = 1.5
        daily_avg_prices = {'TEST': 100.0}
        _apply_linear_decay = TradeMarket._apply_linear_decay
    m = Fake()
    for high, low, delay in [(110.0,90.0,0.0),(110.0,90.0,0.5),(110.0,90.0,0.75),
                              (110.0,90.0,1.5),(110.0,90.0,2.0),(110.0,90.0,-1.0)]:
        print(high, low, delay, '->', m._apply_linear_decay(high, low, delay, 'TEST'))
    "

which, run against their real code in a local clone, printed exactly the six (input, output) pairs
`tests/test_deliberation_comparison.py::TestLinearDecayMatchesTheRealReferenceVectors` pins.

**The structural property this file's docstring below exists to make checkable: their real
function has NO volatility term.** ``_apply_linear_decay(self, high, low, delay, stock_name)``
reads only ``self.daily_avg_prices[stock_name]`` and ``self.decay_window`` — confirmed by reading
the full method (`research/repos/LatencySensitiveBench/HFTBench/Environment/TradingEnv.py:74-92`
on this machine). The decay rate and cap are fixed constants, identical regardless of how volatile
the underlying instrument actually is.
"""

from __future__ import annotations


class LatencyBenchReimplError(ValueError):
    """The reimplementation was called outside the domain it is defined for."""


def linear_decay_price(
    high: float, low: float, delay_seconds: float, *, average_price: float, decay_window: float,
) -> tuple[float, float]:
    """The same idea as `_apply_linear_decay`: as `delay_seconds` grows from 0 toward
    `decay_window`, both `high` and `low` linearly converge toward `average_price`; at or past
    `decay_window` they equal it exactly; at or below zero delay they are unchanged.

    No volatility parameter exists in this signature because none exists in the real function
    being reproduced — that absence is the structural property this whole comparison module
    exists to test, not an oversight here.
    """
    if decay_window <= 0:
        raise LatencyBenchReimplError(f"decay_window={decay_window} must be positive")
    if delay_seconds >= decay_window:
        return average_price, average_price
    if delay_seconds <= 0:
        return high, low
    alpha = delay_seconds / decay_window
    new_high = high * (1 - alpha) + average_price * alpha
    new_low = low * (1 - alpha) + average_price * alpha
    return new_high, new_low


__all__ = ["LatencyBenchReimplError", "linear_decay_price"]
