# Regime detection — from one volatility threshold to the shape of the series

## 1. What ARGUS had

`strategies/track1_suite.py:206-215`, in full:

```python
def rotation_regime_switch(bars, i):
    """Long in low-volatility regimes, flat in high-volatility ones.

    Regime is measured, not predicted: current realised volatility against its own longer history.
    """
    fast = _vol(bars, i, 24)
    slow = _vol(bars, i, 240)
    if slow <= 0:
        return 0.0
    return 1.0 if fast < slow else 0.0
```

One threshold, one dimension, two states. It is honest about what it is, and what it is cannot see a
regime change that does not change amplitude — a grind becoming a chop, a range becoming a trend, a
rally becoming a drawdown at the same realised volatility.

## 2. The two references, read

**`matrix-profile-foundation/matrixprofile/algorithms/regimes.py`** (Apache-2.0):

- `regimes.py:69-76` — the arc count. Each nearest-neighbour arc marks `+1` just after its lower
  endpoint and `-1` at its upper one; the cumulative sum is the number of arcs spanning each index.
- `regimes.py:16-38` — the idealised curve is an explicit **parabola** of height `n/2` centred at
  `n/2`.
- `regimes.py:83-89` — divide, clip to 1, then pin the head and tail over **`w`**.
- `regimes.py:127-145` — regime extraction: argmin, blank `5w` either side, repeat.

**`stumpy/floss.py`**:

- `floss.py:167-183` — the same corrected arc curve, with two differences: the idealised curve is a
  **fitted beta distribution** (`floss.py:111`, `scipy.stats.beta.pdf` plus a least-squares slope
  fit), and the head/tail correction is **`L * excl_factor`**, five times wider.
- `floss.py:_rea` — the same argmin-and-blank extraction, confirming `excl_factor = 5`
  (`floss.py:142`).

## 3. What ARGUS built, and which reference each choice follows

`argus/src/argus/desk/regime.py`, 21 tests, pure Python — no numpy, no scipy.

| Piece | Choice | Following |
|---|---|---|
| arc counting | `+1` after the low end, `-1` at the high end, cumulative sum | both (identical) |
| idealised curve | the analytic **parabola** | matrixprofile — deterministic and scipy-free |
| head/tail correction | **`window * 5`** | stumpy — the narrow `w` leaves the first and last window looking like boundaries when they are only edges |
| regime exclusion | `window * 5` | both |
| profile exclusion | `window / 4` | stumpy's `STUMPY_EXCL_ZONE_DENOM` |

The last row is a deliberate inconsistency with our own `desk/shapematch.py`, which excludes a
**full window**. The two answer different questions: there, no two *reported* analogues may share a
bar because a human reads them as separate events; here the arc count wants every window's genuine
nearest neighbour, and an over-wide exclusion pushes arcs outward and flattens the very curve the
boundaries live in. Both widths are documented at the constant that sets them.

**Cost.** The matrix profile index is brute force, `O(n² m)`: 1,079 hourly bars at a 24-bar window is
about 24 million multiply-adds and runs in **10 seconds**. The inner loop uses the correlation
identity `d² = 2m(1 − ρ)` (`stumpy/core.py:1118`) so each comparison is one dot product rather than
a subtraction and a square per bar. numpy plus numba would save those ten seconds and cost two
dependencies.

## 4. What it found

`python -m argus.desk.regime --symbol NVDAUSDT --days 45` — 1,079 hourly bars:

```
  from             to                 bars   vol bps    drift     arc
  2026-07-31 06:00 2026-08-16 09:00    388       8.6  +14.02%       —
  2026-08-16 10:00 2026-08-26 15:00    246      10.5   -6.36%   0.994
  2026-08-26 16:00 2026-09-14 04:00    445       9.3   +2.43%   0.943
```

Three regimes: a **+14.0% rally**, a **−6.4% drawdown**, and a **+2.4% recovery** — at realised
volatilities of 8.6, 10.5 and 9.3 bps a bar. The loudest stretch is 1.2x the quietest, which is
nothing; amplitude cannot separate these and **the incumbent fast-vs-slow rule does not flip at
either boundary.** The report says so in its own verdict, computed by re-running that rule and
checking whether any of its flips land within one window of a FLUSS boundary.

That comparison is the point of the module. A new detector that merely agreed with the old one would
be a more expensive way to get the same answer.

## 5. Honest limits, stated in the code

- **It does not label a regime.** A boundary says the series before and after are shaped differently.
  Calling one "risk-on" would be a story laid over an index, so each segment is described only by its
  own measured statistics.
- **It does not predict the next one.** FLUSS is retrospective by construction; the arc curve at the
  right-hand edge is pinned to 1 precisely because the data to evaluate it does not exist yet.
- **It needs repetition.** The authors state plainly that FLUSS "is not able to segment single
  gesture patterns" (`regimes.py:105-107`), which is why `MIN_BARS = 240` and why a homogeneous
  series is reported as one regime rather than forced into three. A test asserts that "no boundary"
  is an available answer.
