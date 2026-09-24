# Order-book depth — the endpoint we never called, and what it says

## 1. The gap, stated as it was found

A competitive sweep of this machine's corpora asked which local repos beat ARGUS on execution
realism. The answer came back narrower and more embarrassing than expected: **our fill models are
fine and they had no data.**

- `argus/src/argus/execution/passive.py` and `execution/queue.py` implement queue-position fills
  ported from `hftbacktest`'s `queue.rs`/`latency.rs`, with `BookObservation` requiring
  `level_qty` and `traded_qty` and refusing to grant maker treatment without them.
- `argus/src/argus/sim/book.py` rebuilds a multi-level matching engine from `abides-jpmc-public`.
- And `argus/src/argus/market/bitget.py` fetched **tickers and candles only**. `grep` for
  `orderbook|depth|level_qty` over `market/` returned nothing.

Meanwhile `GET /api/v3/market/orderbook` is documented in Bitget's own SDK catalogue shipped in this
repository — `agent-sdk/src/generated/catalog.ts:115`:

```json
{"operationId":"getOrderbook","method":"GET","path":"/api/v3/market/orderbook","auth":"public",
 "queryParams":[{"name":"category","required":true,...},{"name":"symbol","required":true,...},
                {"name":"limit","required":false,"description":"Level Default: 5. Maximum: 200"}]}
```

Public, keyless, 200 levels. The data had been one HTTP call away for the whole project.

## 2. What was being charged instead

Two constants, both in the live path:

- `paper/ledger.py:record()` was handed `ticker.spread_bps` — the **quoted touch**, which is the
  price of an infinitesimally small trade.
- `paper/ledger.py:settle()` charged a literal `Decimal("0.6")` on **every exit, every instrument,
  every hour**.

Neither is a measurement of what it stands for. The cost of a trade is what you pay to take the size
you want, and that depends on the size resting behind the touch.

## 3. What the book actually says

`python -m argus.market.depth`, 12 rTokens, 50 levels a side, slippage in bps against mid, taking
the ask (2026-09-14):

| symbol | quoted | $1,000 | $5,000 | $25,000 | $100,000 |
|---|---|---|---|---|---|
| QQQUSDT | 0.14 | 0.07 | 0.08 | 0.82 | 2.22 |
| NVDAUSDT | 0.47 | 0.64 | 1.11 | 2.06 | 5.84 |
| MSTRUSDT | 0.76 | 1.21 | 2.62 | 5.88 | 13.36 |
| TSLAUSDT | 0.83 | 1.11 | 2.04 | 4.96 | 9.19 |
| AAPLUSDT | 0.91 | 1.09 | 1.97 | 4.38 | 9.15 |
| GOOGLUSDT | 0.89 | 0.96 | 2.58 | 8.54 | 14.54 |
| MSFTUSDT | 1.42 | 2.90 | 3.84 | 4.09 | 10.03 |
| TQQQUSDT | 1.46 | 2.10 | 3.55 | 11.99 | 25.69 |
| AMZNUSDT | 2.36 | 3.75 | 4.21 | 6.20 | 11.41 |
| METAUSDT | 2.03 | 3.45 | 4.85 | 8.81 | 17.06 |
| COINUSDT | 2.82 | 2.65 | 6.08 | 11.09 | 24.63 |
| SQQQUSDT | 4.96 | 5.38 | 10.05 | 19.43 | 22.64 |

**The important number is not the level, it is the spread of the ratios.** At $25,000, QQQUSDT costs
5.9x its quote and SQQQUSDT 3.9x, but in absolute terms one is 0.82bps and the other 19.43 — a 24x
difference between instruments that the quoted spread shows as only 35x... and the flat 0.6bps
constant showed as no difference at all. A cost model that understates by a constant misprices every
trade equally; one that understates by a varying factor **reorders which instruments look cheap**,
which is the error that actually changes decisions.

For scale: the desk's own hurdle is a 12bps round trip in fees. Taking $25,000 of SQQQUSDT adds
19.43bps on entry and a similar figure on exit, so the true hurdle on that name is roughly four times
the fee-only figure the system had been using.

## 4. What was built

`argus/src/argus/market/depth.py`, 23 tests in `argus/tests/test_depth.py`.

- `OrderBook.sweep(notional, direction)` — walks levels, returns the volume-weighted price, the
  levels consumed, the slippage against mid, and **whether the visible book could absorb the size**.
- `OrderBook.executable_within(budget_bps)` — the inverse: how much size this book will carry at a
  price you are willing to pay. Computed by accumulating levels, so it is exact at a level boundary
  rather than converged-to by bisection.
- `OrderBook.imbalance(levels)` — reported, never traded on: order-book imbalance predicts the next
  tick, and the next tick is not a horizon this desk operates at. It is there so a reader can see
  whether a measured slippage came from an already-lopsided book.
- `measured_spread_bps()` — returns `None` for an unreachable venue **and for a size the book cannot
  absorb**, because a partial sweep prices the cheap half of a trade that could not be done.
- Levels are **sorted on arrival**. The endpoint documents no ordering guarantee and every
  calculation here assumes the touch is at index zero; sorting 200 rows costs nothing and removes a
  silent dependency on an undocumented property of someone else's feed.
- A crossed book raises rather than returning a negative cost: it is a feed error, not an arbitrage.

**Wired into both legs of every paper fill.** `paper/runner.py:_executable_spread_bps` measures the
entry at the size actually being traded, and `ledger.settle(exit_spread_bps=...)` measures the exit
on the opposite side at the position's own notional. Where the book cannot answer, the previous
behaviour applies unchanged — and the fallback is explicit in the code rather than a silent default.
`run_paper_cycle.ps1` refreshes `data/depth.json` before every scheduled cycle.

## 5. The square-root law now has something to be wrong against

`cost/model.py` charges impact as `coefficient * participation ** gamma`, with gamma 1.5 taken from
cvxportfolio (`costs.py:826`) and a coefficient of 10 that nobody on this project measured.
`impact_check()` walks the real book at a series of sizes and prints the modelled figure beside the
measured one. It requires an average daily notional to make participation meaningful and **reports
only the measured column when none is supplied** — inventing a denominator to fill the comparison
would be the exact failure the comparison exists to expose.

## 6. Not done

- **A streaming book.** This is REST snapshots. The WebSocket feed would give queue dynamics over
  time, which is what `execution/queue.py` is actually built for; the snapshot only prices a sweep.
  Stated as absent rather than implied.
- **Markout / adverse-selection measurement.** `GET /api/v3/market/fills` (`catalog.ts:118`, public)
  would support it and is not yet called. `execution/passive.py:20-21` already names this gap in its
  own docstring; it remains open.
- **Recalibrating `impact_coefficient` from the measured curve.** The comparison is now possible; the
  fit is not done, and claiming a calibrated coefficient before fitting one would be exactly the kind
  of unearned number this pass removed.
