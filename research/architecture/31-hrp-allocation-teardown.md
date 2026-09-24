# Portfolio construction — what the libraries solve, and the question they leave out

**Read on this machine, `file:line` throughout.**

- `okx/trading/best-of-the-best/repos/PyPortfolioOpt/pypfopt/hierarchical_portfolio.py` (MIT)
- the same repo's `pypfopt/cla.py`, `pypfopt/efficient_frontier.py`
- `okx/trading/best-of-the-best/repos/hummingbot/controllers/generic/quantum_grid_allocator.py`
- our own `argus/src/argus/desk/portfolio.py` and `argus/src/argus/risk/sizing.py`

## 1. Where ARGUS was

`desk/portfolio.py:assess()` takes `weights_before` and `weights_after` and reports what changed:
beta, risk share, effective positions, the most correlated existing holding. It is a **grader**. The
desk could ask *"is this trade sensible?"* and never *"what should I hold instead?"* — so the shape
of the book was whatever the sequence of individually-acceptable trades happened to produce.
`risk/sizing.py` is single-asset Kelly, which sizes one position and says nothing about the set.

Confirmed by grep over `argus/src`: no HRP, no risk parity, no max-Sharpe, no efficient frontier, no
weight solver of any kind before this pass.

## 2. Why hierarchical risk parity and not the obvious alternative

Mean-variance needs Σ⁻¹. The twelve rTokens are tokenized US equities whose hourly returns correlate
above 0.9; the sample covariance is near-singular, and inverting it turns estimation noise into large
offsetting long/short weights. That failure is the entire motivation of López de Prado's 2016 paper,
and it is not hypothetical here — it is what the correlation structure of this specific universe
guarantees.

HRP never inverts anything. It clusters on a correlation distance, orders the leaves by traversing
the tree, and splits the risk budget recursively. Long-only falls out of the construction: every step
multiplies a positive weight by a split in [0, 1], so a short cannot appear.

PyPortfolioOpt's implementation, the one we reproduced:

- `hierarchical_portfolio.py:188` — `matrix = np.sqrt(np.clip((1.0 - corr) / 2.0, a_min=0.0, a_max=1.0))`.
  The clip is load-bearing: a sample correlation of 1 + 1e-12 gives a negative radicand.
- `hierarchical_portfolio.py:191-194` — `sch.linkage(dist, "single")`, then
  `HRPOpt._get_quasi_diag`, which is `sch.to_tree(link, rd=False).pre_order()` (`:122`).
- `hierarchical_portfolio.py:143-160` — the bisection loop; `:103-105` — cluster variance under
  inverse-variance weights.

**One divergence from the paper, and it is PyPortfolioOpt's, not ours.** De Prado's Chapter 16 code
clusters on the Euclidean distance *between the columns* of the correlation-distance matrix — a
distance of distances. PyPortfolioOpt clusters on the correlation distance directly. We follow
PyPortfolioOpt because that is the implementation we read and can check ourselves against; noted here
rather than left as a silent difference from the citation.

## 3. The reproduction

`argus/src/argus/desk/allocation.py` is pure Python — no numpy, no scipy, including a hand-written
single-linkage clustering and pre-order traversal.

Run PyPortfolioOpt's own code path (numpy 2.4, scipy 1.18, pandas 3.0 in the trading venv) and ours
over the **same twelve real instruments, 400 hourly bars**:

```
AAPLUSDT   ours 0.18879449  pypfopt 0.18879449   diff 2.8e-17
QQQUSDT    ours 0.19985412  pypfopt 0.19985412   diff 3.6e-16
MSTRUSDT   ours 0.01313546  pypfopt 0.01313546   diff 8.7e-18
...        worst difference across all twelve: 3.6e-16
```

The quasi-diagonal leaf order matches exactly too, which matters more than the weights: it decides
every subsequent split, so two implementations agreeing on weights but not on order would be agreeing
by luck. `tests/test_allocation.py::TestItReproducesPyPortfolioOpt` runs both assertions.

A behaviour worth recording rather than hiding: on **three identical series** both implementations
return `{0.5, 0.25, 0.25}`, not thirds — recursive bisection splits an ordered list of three into one
and two and gives each half an equal risk budget. Verified against PyPortfolioOpt on that exact input
and pinned by a test with the explanation attached. It is the honest cost of a method that never
inverts a matrix.

## 4. What neither library has, and it is the decision

PyPortfolioOpt returns a weight vector and stops. So does Riskfolio-Lib, so does skfolio. **A weight
vector is not a decision.** The decision is whether moving from the book you hold to the book the
optimiser wants is worth what the move costs, and here every unit of turnover crosses 6bps.

`optimize_trade()` answers that:

- **turnover priced** — `cost_bps = turnover * 6`.
- **dust dropped** — legs below 1% of the book are not traded. Below that the fee is a larger share
  of the leg than any risk it moves, and a twelve-leg dust rebalance is how a portfolio quietly
  becomes a fee-generating machine. The dropped weight is not reallocated: the plan is a partial move
  and its arithmetic says so.
- **break-even holding period** — with the assumption stated in the same sentence, every time.
  Lowering volatility earns nothing on its own; at constant Sharpe a quieter book earns
  proportionally less. The benefit appears only when the book is scaled to a volatility target, where
  the lower-volatility allocation can be held `vol_before / vol_after` larger. So the gain per bar is
  `s · vol_before · (vol_before/vol_after − 1)` and the break-even is the one-off cost divided by it.
  ARGUS has no measured Sharpe (zero executed trades), so `s` is an explicit input defaulting to 1.0
  annualised and it is named in the verdict on every call.

## 5. What the live run found

`python -m argus.desk.allocation --days 60`, 12 instruments, 1,438 hourly bars, from equal weight:

```
  AAPLUSDT   8.33% -> 24.90%      QQQUSDT   8.33% -> 17.14%
  MSTRUSDT   8.33% ->  1.70%      TQQQUSDT  8.33% ->  1.91%
  diversification ratio 2.224 -> 2.408
  turnover 53.7% costing 3.2bps; volatility 22 -> 16bps per bar, +42.7% of variance
  repays in 43 bars, inside the 168-bar horizon. Worth doing
```

Unlike the last two passes — the shape matcher and the pairs scan, both of which correctly found
nothing — this one finds a real improvement that clears its cost by a wide margin, because turnover
on a *weight* basis is cheap (53.7% of the book at 6bps is 3.2bps) while the variance reduction is
large. The honest reading is that **allocation, not signal, is where the recoverable edge on this
venue lives**: the fee that dominates every intraday trading idea (12bps round trip against ~0bps of
measured intraday edge) is nearly irrelevant to a once-a-week rebalance.

Wired into `desk/research.py:allocation_step`. Live on a three-name book, it moved the chain's verdict
to ADD_SMALLER by objecting to the proposed book:

```
allocation: 3 leg(s) (AAPLUSDT 27.3%->36.1%, NVDAUSDT 9.1%->21.4%, TSLAUSDT 36.4%->16.2%),
turnover 41.3% costing 2.5bps. Portfolio volatility 25 -> 23bps per bar, +16.4% of variance.
It repays in 100 bars, inside the 168-bar horizon, at an assumed annualised Sharpe of 1.0.
```

That is the first step in the chain that can disagree with a trade by proposing a different one.

## 6. Not built

- **Mean-variance / max-Sharpe / CLA** (`pypfopt/cla.py:330-398`). Requires an expected-return vector,
  and ARGUS has no return forecast it would stand behind — the analogue and shape retrievals both
  refused. A frontier built on fabricated μ is a decoration.
- **Ledoit-Wolf shrinkage.** Would matter for an inverting method; HRP does not invert.
- **Black-Litterman.** Same objection as mean-variance, plus a prior nobody here can justify.
- **Transaction costs inside the objective** (rather than evaluated after). The correct form is a
  convex problem with an L1 turnover penalty, which needs a solver; the current design finds the
  unconstrained target and then refuses the move when it does not pay, which reaches the same
  accept/reject conclusion without pretending to a precision the inputs do not support.
