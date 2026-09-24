# Hedge effectiveness — who measures it, who configures it, and what ARGUS does

**Sources read on this machine, with `file:line`.** Nothing here is from memory.

## 1. The state of the art in the corpus is a config field

`hummingbot` is the most-deployed open-source trading bot in the corpus and its dedicated hedging
strategy does not estimate a hedge ratio at all:

- `hummingbot/hummingbot/strategy/hedge/hedge.py:64` — *"`:param hedge_ratio`: Ratio of total asset
  value to hedge."* It is read from configuration at `hedge.py:77`
  (`self._hedge_ratio = config_map.hedge_ratio`) and used directly in the sizing formulas documented
  at `hedge.py:33` and `hedge.py:39`. The number is whatever the operator typed.

Its statistical-arbitrage controller does estimate one, and then does three things wrong:

- `hummingbot/controllers/generic/stat_arb.py:381-383` — the ratio is fitted by
  `LinearRegression().fit(dominant_cum_returns_reshaped, hedge_cum_returns)`, i.e. an OLS of one
  **cumulative return path** on another. Two integrated series regressed on each other is the
  textbook spurious regression; the slope is not a hedge ratio and its R² is not effectiveness.
- `stat_arb.py:393-395` — the z-score uses `np.mean(spread_pct)` and `np.std(spread_pct)` over the
  whole processed window, the current bar included. The signal is scored against a mean that had not
  happened yet.
- `stat_arb.py:37` and `stat_arb.py:291` — the estimated `beta` is stored in `processed_data` and
  the position is sized with `self.config.pos_hedge_ratio`, a configuration default of `1.0`. The
  regression result never reaches the trade. (`backtrader`'s pairs sample has the same defect; see
  teardown 27.)

**`grep -rli ederington` over 114 repos returns nothing.** No implementation in the corpus computes
hedging effectiveness in the sense the term has had since 1979.

## 2. What ARGUS had before this pass, and why it was worse than it looked

`risk/hedgeability.py` prices a hedge as a product of five factors. Three of them were literals in
the source:

```python
correlation_confidence=Decimal("0.98"),   # open-market candidate
basis_stability=Decimal("0.95"),
...
correlation_confidence=Decimal("0.95"),   # shut-market candidate
basis_stability=Decimal("0.9"),
```

and the callers passed `risk_reduction` as `Decimal("0.98")` / `Decimal("0.95")`
(`paper/runner.py`). Six numbers, no sample behind any of them. This is a worse failure than
hummingbot's config field, because a config field is visibly the operator's assumption while a
literal inside a risk layer reads as a finding — in a project whose standing rule is *never guess,
read the source*.

## 3. What ARGUS does now

`argus/src/argus/risk/effectiveness.py`, 27 tests in `argus/tests/test_effectiveness.py`.

| Field | Was | Is now |
|---|---|---|
| `risk_reduction` | `0.98` literal | Ederington effectiveness: ρ², the share of variance the optimally-sized hedge removes |
| `correlation_confidence` | `0.98` literal | lower bound of the 95% Fisher interval, `tanh(atanh(r) - 1.96/sqrt(n-3))`, floored at 0 |
| `basis_stability` | `0.95` literal | unit-ratio effectiveness, `1 - Var(Δspot - Δhedge) / Var(Δspot)` |
| provenance | — | `MEASURED` or `ASSUMED`, stamped on every candidate, with the sample (`phase, n, window, timestamp`) |

Four decisions worth stating, because each is a place the easy version is wrong:

1. **Confidence is an interval bound, not the estimate.** The field's own docstring always said it
   meant "how much we trust the correlation *in the current regime*" — a statement about sampling
   error. Fisher's transform answers it; a constant cannot, because it cannot know n.
2. **A negative lower bound earns zero, not its absolute value.** An anticorrelated instrument
   hedges only if you flip the sign, and the risk layer does not flip signs. Pinned by
   `test_an_anticorrelated_hedge_is_not_credited_as_confidence`.
3. **Optimal-ratio and unit-ratio effectiveness are reported separately** because their *gap* is the
   basis problem. A hedge that moves twice as hard as the position is perfect at ratio 0.5 and worse
   than nothing at 1.0 — one number cannot say that, and
   `test_a_hedge_that_moves_twice_as_hard_needs_sizing` asserts both halves.
4. **Measured per session phase, never pooled.** `truth/clocks.py` already separates a trading anchor
   from a sleeping one; a correlation measured across both is an average of two different worlds.

**Staleness is absence.** `lookup()` returns `None` past `STALE_AFTER_HOURS = 36` rather than a
slightly-worse number, and the live path then falls back to the old constants *and says so in the
session note*. Nothing silently improves or silently decays.

## 4. What the measurement found

`python -m argus.risk.effectiveness --days 60` — 12 rTokens, hourly market-versus-index, 60
measurements written to `argus/data/hedge_effectiveness.json`:

| Phase | Typical ρ | Ederington R² | n |
|---|---|---|---|
| regular hours | 0.996 – 1.000 | 99.2% – 99.9% | 251 |
| extended | 0.996 – 0.999 | 99.2% – 99.8% | 419 |
| overnight | 0.989 – 0.996 | 97.9% – 99.2% | 335 |
| weekend | 0.989 – 0.998 | 97.7% – 99.5% | 430 |
| pooled | 0.976 – 0.990 | 95.3% – 98.1% | 1,438 |

Two things fall out. The constants were **too pessimistic during regular hours** — 0.98 and 0.95
against a measured 0.998 and 0.998 — and the pooled figure is materially *worse* than any single
phase, which is exactly the averaging artefact that made per-phase measurement necessary.

The minimum-variance ratio sits between 0.983 and 1.009 across all twelve names, so a one-for-one
hedge is very nearly optimal here — a fact worth having rather than assuming, and the first thing
that would change if the token's tracking degraded.

`paper/runner.py:_hedge_surface` reads the file on every cycle; live output on 2026-09-14:

```
awake  | measured | rr 0.998149 | cc 0.998813 | bs 0.99796
    evidence: rth, n=251, 60d, measured 2026-09-14 02:59Z
asleep | measured | rr 0.993925 | cc 0.996323 | bs 0.993913
    evidence: weekend, n=429, 60d, measured 2026-09-14 02:59Z
```

## 5. What we did not build

- **A time-varying (Kalman) hedge ratio.** The measured ratios are stable across phases and close to
  one; a state-space model would add machinery whose whole value is tracking a moving β, and nothing
  in the data says β moves. Revisit if a future measurement shows phase-to-phase drift.
- **An effectiveness measure against the actual equity.** The hedge instrument we can measure is
  Bitget's published index, not the underlying share, because we have no equity data feed and no
  broker. The index is derived from the underlying and tracks it during regular hours, which makes
  it a defensible proxy — and it is **NOT VERIFIED** against the equity's own prints. Stated on the
  candidate rather than buried: `execution_probability` is zero for exactly this reason.
