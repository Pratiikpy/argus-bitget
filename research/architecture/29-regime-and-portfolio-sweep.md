# Regime detection and portfolio construction — code-level sweep

ARGUS is pure Python (no numpy/scipy/pandas/cvxpy). Every recommendation below is filtered against
that constraint explicitly. All external claims carry `file:line`; anything not read is marked
NOT VERIFIED. GPL sources are patterns-only, never copied.

---

## Part A — Regime detection

### A0. What ARGUS actually does today (verified)

```
$ grep -rniE "regime|hmm|markov|changepoint|cusum|bocpd|pelt|hurst" src --include=*.py -l
src/argus/agents/novelty.py
src/argus/eval/bench.py
src/argus/eval/forecasts.py
src/argus/eval/incremental.py
src/argus/execution/latency.py
src/argus/market/volatility.py
src/argus/research/factor_lab.py
src/argus/research/grammar.py
src/argus/research/overfit.py
src/argus/risk/hedgeability.py
src/argus/sim/agents.py
src/argus/sim/market.py
src/argus/status.py
src/argus/strategies/track1_suite.py
```

The trading-relevant one is `src/argus/strategies/track1_suite.py:206-215`, `rotation_regime_switch`:

```python
def rotation_regime_switch(bars: Sequence[Bar], i: int) -> float:
    """Long in low-volatility regimes, flat in high-volatility ones.
    Regime is measured, not predicted: current realised volatility against its own longer history.
    """
    fast = _vol(bars, i, 24)
    slow = _vol(bars, i, 240)
    if slow <= 0:
        return 0.0
    return 1.0 if fast < slow else 0.0
```

This is a binary threshold on two rolling-window realised-vol estimates (24 bars vs 240 bars) —
no persistence model, no changepoint statistic, no confidence, no third state. `src/argus/market/
volatility.py:82-119` (`Regime` enum / `classify()`) is a separate, non-trading module: it buckets
CBOE VIX history into NORMAL/ELEVATED/STRESSED/UNKNOWN by percentile — good design (percentile over
absolute threshold) but external-index-only, not applied to ARGUS's own OHLCV series, and still not
a changepoint or state-persistence model. `grep -rn -i "cusum\|structural.break" src` returns
**zero hits** — confirms no structural-break method exists anywhere in ARGUS today. This gap is
real, not a false claim.

### A1. Candidates surveyed, in order of fit

#### A1.1 — FLUSS (matrix-profile regime segmentation) — best fit, already load-bearing in ARGUS

`research/repos-themed/matrix-profile-foundation~matrixprofile/matrixprofile/algorithms/
regimes.py:43-153` (Apache 2.0). Corrected Arc Curve:

```python
# lines ~66-81 (nnmark / arc-crossing count, from the matrix profile index mpi)
for i in range(n):
    mpi_val = mpi[i]
    small, large = int(min(i, mpi_val)), int(max(i, mpi_val))
    nnmark[small + 1] += 1
    nnmark[large] -= 1
cross_count = np.cumsum(nnmark)
idealized = cross_count / idealized_arc_curve(n, ...)   # parabola normaliser
corrected_arc_curve = np.minimum(idealized, 1.0)
corrected_arc_curve[:w] = 1; corrected_arc_curve[-w:] = 1   # edge correction
```

Regime boundaries are the local minima of the CAC with an exclusion zone of `w*5` bars
(`regimes.py:129`) — same file whose exclusion-zone convention ARGUS's own
`desk/shapematch.py` already deviates from deliberately (documented in
`research/architecture/26-matrix-profile-and-analogue-retrieval.md`). Streaming variant: `research/
repos-themed/TDAmeritrade~stumpy/stumpy/floss.py:121-231` (BSD-3) — same CAC math, computed
incrementally as bars arrive (`_nnmark`, `_iac` beta-fit for the idealized curve, lines 167-181).

**Data requirement:** a precomputed matrix profile (self-join distance + nearest-neighbour index)
over the series, window `m` (ARGUS already computes this — `desk/shapematch.py` does a self-join
scan for analogue retrieval, sharing the same O(n·m) core). **Pure Python:** yes — the CAC math
itself is arithmetic on two arrays (cumsum, elementwise divide, argmin with exclusion). At 1,400
hourly bars the self-join is ~1,400² comparisons at window m≈24–48; ARGUS's existing shapematch
scan already does this scale of computation in pure Python without numpy, so FLUSS is a few dozen
lines of arithmetic added on top of code already proven to run at this size.

**Why this is the one to build:** it reuses a distance computation ARGUS already trusts and has
already calibrated with a null-model p-value (`shapematch.py:find`, `null_p <= 0.05` gate — see the
26- note). Nothing else surveyed gets that reuse for free.

#### A1.2 — Hurst exponent (R/S) — cheap persistence signal, pairs well with FLUSS

`research/repos-themed` corpus, Meta's Kats, `kats/tsfeatures/tsfeatures.py:904-928` (MIT):

```python
lags = range(2, min(lag_size, len(x) - 1))
tau = [np.std(np.asarray(x)[lag:] - np.asarray(x)[:-lag]) for lag in lags]
poly = np.polyfit(np.log(lags), np.log(tau), 1)   # slope = Hurst exponent
return poly[0]
```

H>0.5 trending/persistent, H<0.5 mean-reverting, H≈0.5 random walk. Needs only `std` over lagged
differences and a linear regression (polyfit degree 1 — trivially hand-rolled via least squares in
pure Python, no numpy required). Minimum ~`lag_size + 2` samples (default lag_size=30 → ~32 bars);
at 1,400 hourly bars a rolling 240-bar Hurst is cheap and gives a continuous regime-character number
to sit next to FLUSS's discrete boundaries. **Pure Python:** yes, trivially.

#### A1.3 — CUSUM (mean-shift, Gaussian LLR) — usable but ARGUS's own mlfinlab copy is a dead end

Two implementations were checked:

- **mlfinlab-official** (both clones — `research corpus, repos/
  mlfinlab-official\mlfinlab\structural_breaks\cusum.py` and `research/corpus/repos/
  mlfinlab-official/.../cusum.py`) — **verified myself, not from the parallel reader report**: every
  function body in `cusum.py` (`_get_values_diff`, `_get_s_n_for_t`,
  `get_chu_stinchcombe_white_statistics`) is a bare `pass` — 3 of 3 functions stubbed. `sadf.py`
  has 6 stubbed functions, `chow.py` has 2. This is the open-source shell of a package whose real
  structural-break code is paywalled (Hudson & Thames commercial mlfinlab). **mlfinlab is not a
  usable source for this — flag explicitly, since a naive read of the directory listing would
  claim it as "found."**
- **Meta Kats**, `research repos`, `kats/detectors/cusum_detection.py:352-451` (MIT) — a real,
  runnable implementation:

```python
cusum_ts = np.cumsum(ts_int - np.mean(ts_int))          # line 368
changepoint = min(changepoint_func(cusum_ts), len(ts_int) - 2)
while n < max_iter:                                       # lines 373-385, iterate to convergence
    mu0 = np.mean(ts_int[: changepoint + 1]); mu1 = np.mean(ts_int[changepoint + 1 :])
    mean = (mu0 + mu1) / 2
    cusum_ts = np.cumsum(ts_int - mean)
    next_changepoint = max(1, min(changepoint_func(cusum_ts), len(ts_int) - 2))
    if next_changepoint == changepoint: break
# then a log-likelihood-ratio test, lines 425-451:
llr = -2 * (log_llr(left_segment, mu_tilde, sigma_tilde, mu0, scale)
          + log_llr(right_segment, mu_tilde, sigma_tilde, mu1, scale))
p_value = 1 - chi2.cdf(llr, 2)
```

**Pure Python:** mostly — the only non-trivial dependency is `chi2.cdf(llr, 2)`, and the chi-square
CDF with 2 degrees of freedom has a closed form (`1 - exp(-llr/2)`), so this is *fully* hand-rollable
without scipy. Minimum samples: no hard floor, works reasonably from ~20 bars. Detects one
mean-shift breakpoint per call (find multiple by recursive bisection on the two halves — same
pattern as the matrix-profile FLUSS regime extraction above).

#### A1.4 — BOCPD (Bayesian Online Changepoint Detection) — real algorithm, not pure-Python-friendly

`research repos`, Kats, `kats/detectors/bocpd.py:828-969` (MIT), Adams & MacKay (2007) run-length
posterior:

```python
pred_arr = model.pred_prob(t=i, x=this_pt)                 # predictive prob under each run length
message[...] = pred_arr + message[...] + log_om_cp_prior    # growth-probability message (line ~907)
log_change_point_prob = logaddexp.reduce(pred_arr + message + log_cp_prior, axis=0)
log_evidence = logaddexp(log_change_point_prob, log_change_point_prob - log_cp_prior + log_om_cp_prior)
log_posterior = message - log_evidence
model.update_sufficient_stats(x=this_pt)
```

Min 10 samples (`_MIN_POINTS = 10`, line 50). This is the theoretically strongest online method
(full posterior over "bars since last regime change," not a point estimate) but the run-length
distribution is O(n) per step / O(n²) total, and the numerically-stable form leans on
`scipy.special.logsumexp` throughout. Hand-rollable in principle (`logsumexp` is `max(x) +
log(sum(exp(x - max(x))))`, three lines of pure Python) but the full message-passing state (a
growing array of run-length beliefs) is a meaningfully bigger build than FLUSS or CUSUM for the
same 1,400-bar target. **Verdict: buildable in pure Python, but not the first thing to build** —
rank it below FLUSS/Hurst/CUSUM on cost-to-value.

#### A1.5 — Markov-switching / Hamilton filter — verified NOT feasible in pure Python

`site-packages/statsmodels\tsa\regime_switching\
markov_switching.py:110-231` (BSD-3). The Hamilton filter forward pass and the EM parameter update
both call into compiled Cython extensions — confirmed present in the installed package directory:
`_hamilton_filter.cp312-win_amd64.pyd`, `_kim_smoother.cp312-win_amd64.pyd`. The pure-Python
wrapper in `markov_switching.py` only orchestrates calls into those compiled routines; the actual
per-step recursion is not available as readable Python. **Reimplementing Hamilton filtering by hand
is possible in principle** (it is textbook: predict `ξ_{t|t-1} = P'ξ_{t-1|t-1}`, weight by
regime-conditional likelihoods, normalize) but statsmodels itself is not a source to lift code from
here — the report's `A2.2` numbered equations above (for BOCPD) are the closer textbook analogue,
not this file. **Do not cite statsmodels line numbers as "the algorithm" — they are compiled, not
Python.**

#### A1.6 — HMM + Viterbi (offline, known parameters) — buildable but needs a separate EM step

`research repos`, sktime, `sktime/detection/hmm.py:209-282` (BSD-3): pure-Python/numpy Viterbi
decode given a transition matrix and emission functions —

```python
trans_prob[:, 0] = log(initial_probs) + log(emission_probs[:, 0])
for i in range(1, num_obs):
    paths = log(transition_prob_mat) + trans_prob[:, i-1].T + log(emission_probs[:, i])
    trans_id[:, i] = argmax(paths, axis=0)
    trans_prob[:, i] = max(paths, axis=0)
# backtrack trans_id for the MAP state sequence
```

This module assumes the transition matrix and emission distributions are *already known* — it does
not fit them. Fitting (Baum-Welch/EM) is a separate, harder piece not found as a clean pure-Python
reference in this sweep (statsmodels' EM is compiled, see A1.5). **Usable only if ARGUS is willing
to hand-set 2-3 regime emission distributions (e.g. Gaussian vol buckets) rather than learn them** —
viable but strictly more design work than FLUSS/Hurst/CUSUM for a worse-justified payoff at this
data size.

### A2. Ranking for what to build next

1. **FLUSS regime boundaries** on top of ARGUS's existing self-join distance code
   (`desk/shapematch.py`) — cheapest, reuses trusted/calibrated infrastructure, pure Python at
   1,400 bars, gives discrete regime-change events with the same null-model rigor ARGUS already
   applies to analogue matches.
2. **Hurst exponent**, rolling, as a continuous companion signal (trending vs mean-reverting) —
   near-zero implementation cost, pure Python, strengthens the regime read rather than replacing it.
3. **CUSUM (Kats-style mean-shift LLR)**, hand-rolled with the closed-form chi²(2) CDF — a second,
   independent confirmation signal for "did the regime actually change" vs FLUSS's shape-based
   read.
4. BOCPD — worth it later for online, non-binary regime confidence, but scope it as a bigger build.
5. Markov-switching / HMM-EM — not recommended to build from these sources; no clean pure-Python
   reference for the fitting step was found anywhere in the corpus (NOT VERIFIED that one exists
   locally at all — statsmodels' is compiled, sktime's assumes known parameters).

---

## Part B — Portfolio construction

### B0. What ARGUS actually does today (verified, full read of both files)

`src/argus/desk/portfolio.py` (982 lines) — defines: `beta`, `correlation`, `covariance_matrix`,
`decompose` (risk decomposition / effective-N / most-concentrated position), `assess`
(`portfolio.py:452-522`), `stress_by_beta`, `worst_window`, `factor_exposures`. **`assess()` takes
`weights_before` and `weights_after` as *inputs*** — it grades a trade it is handed (beta shift,
risk-share shift, effective-position-count shift, max correlation to existing book) and returns a
`TradeImpact`. It never proposes weights. Confirmed: `grep -n "^def " portfolio.py` shows no
function that returns an allocation — every function either measures an existing book or grades a
proposed delta to one.

`src/argus/risk/sizing.py` (180 lines) — `kelly_fraction()` (line 103): `f* = (p·b - q) / b`,
single-asset, floored at zero. `size()` (line 138): gates the above behind a calibration check
(`calibration_gate`, ECE ≤ 0.15 over ≥20 graded outcomes) and falls back to a fixed 5% fraction
otherwise, capped at 25% of book. **This is single-asset position sizing, not portfolio weight
allocation** — there is no covariance-aware, cross-asset solve anywhere in the file.

`grep -rn -i "hierarchical.risk.parity|risk.parity|\bhrp\b|max.sharpe|mean.variance|efficient.
frontier" src` returns exactly one hit, `execution/schedule.py:9`, which is the Almgren-Chriss
execution-cost efficient frontier (a different, execution-scheduling "efficient frontier," not
portfolio allocation). **Confirmed: ARGUS has zero portfolio-weight solvers. The gap in the prompt
is real, not a repeat of a previously-false claim** — this was checked directly against the grep
output above, not asserted from memory. Also checked: no `mlfinlab` `portfolio_optimization`
module exists in either local mlfinlab clone (`find ... -iname "*.py" -path "*portfolio*"` returns
nothing in both `best-of-the-best/repos/mlfinlab-official` and `research/corpus/repos/
mlfinlab-official`).

### B1. Solvers surveyed

#### B1.1 — HRP (Hierarchical Risk Parity) — the one to build; no convex solver, three independent readable implementations agree on the algorithm

**PyPortfolioOpt** (MIT), `research corpus, repos/PyPortfolioOpt\
pypfopt\hierarchical_portfolio.py:125-162`, `_raw_hrp_allocation`:

```python
def _raw_hrp_allocation(cov, ordered_tickers):
    w = pd.Series(1.0, index=ordered_tickers)
    cluster_items = [ordered_tickers]
    while len(cluster_items) > 0:
        cluster_items = [i[j:k] for i in cluster_items
                          for j, k in ((0, len(i)//2), (len(i)//2, len(i))) if len(i) > 1]
        for i in range(0, len(cluster_items), 2):
            first, second = cluster_items[i], cluster_items[i+1]
            v1 = HRPOpt._get_cluster_var(cov, first)
            v2 = HRPOpt._get_cluster_var(cov, second)
            alpha = 1 - v1 / (v1 + v2)
            w[first]  *= alpha
            w[second] *= 1 - alpha
    return w

@staticmethod
def _get_cluster_var(cov, cluster_items):          # lines 85-105
    cov_slice = cov.loc[cluster_items, cluster_items]
    weights = 1 / np.diag(cov_slice)                 # inverse-variance weights within the cluster
    weights /= weights.sum()
    return np.linalg.multi_dot((weights, cov_slice, weights))
```

Ordering step: quasi-diagonalization via single-linkage hierarchical clustering + matrix seriation
(`hierarchical_portfolio.py` `optimize()`, ~lines 192-197). **Riskfolio-Lib** (BSD-3),
`Riskfolio-Lib/riskfolio/src/HCPortfolio.py:414-519` (`_recursive_bisection`) and `:199-256`
(`_naive_risk`), implements the identical recursive-bisection structure generalized to any risk
measure (not just variance) — same `alpha = 1 - left_risk/(left_risk+right_risk)` split, with
constraint clamping applied to `alpha` (lines 1059-1081). **skfolio** (BSD-3),
`skfolio/src/skfolio/optimization/cluster/hierarchical/_hrp.py:402-487`, same recursive bisection
plus explicit weight-bound clamping on the split factor
(`_apply_weight_constraints_to_split_factor`, lines 439-487) and additive transaction-cost handling
in the expected-return input rather than in the bisection itself.

**Objective:** at each split, minimize the variance (or chosen risk measure) of each side's
naive inverse-risk sub-portfolio, then allocate the parent weight between the two sides in inverse
proportion to their risk. **Constraints:** min/max weight per asset, handled by clamping `alpha` at
each bisection step (skfolio's version, cited above) — no matrix solve needed for bounds.
**Transaction costs:** not inside the bisection recursion in any of the three; applied as an
additive adjustment to expected returns upstream (skfolio) or not modeled at all (PyPortfolioOpt,
Riskfolio HRP mode). **Convex solver required: NO** — confirmed across all three independent
implementations; the only linear-algebra operation is inverse-variance weighting within a cluster
(`1/diag(cov)`, a vector op) and one dot product per cluster to get its aggregate variance — both
trivial in pure Python for the asset counts ARGUS runs.

**Data requirement:** a covariance matrix over the asset universe (ARGUS already computes pairwise
covariance in `portfolio.py:312`, `covariance_matrix`) plus a distance matrix derived from
correlation (`d_ij = sqrt(0.5*(1-ρ_ij))`, standard HRP distance, hierarchical-clustering-ready) —
both buildable from data ARGUS already has.

**Why this is the one to build:** it is the only surveyed solver that (a) needs no convex solver,
(b) is corroborated by three independent codebases agreeing on the same equations (lowers the risk
of copying one author's bug), and (c) consumes exactly the covariance matrix ARGUS's `portfolio.py`
already computes — `decompose()` and `_raw_hrp_allocation` share an input. Building it is adding an
allocation function next to the existing risk-decomposition function, not a new subsystem.

#### B1.2 — Critical Line Algorithm (CLA) — pure Python but the wrong fit for ARGUS's stated preference

**PyPortfolioOpt** (MIT), `pypfopt/cla.py:330-398`. Traces the efficient frontier by finding
"turning points" where a weight bound activates/deactivates, using Lagrange multipliers and
`np.linalg.inv()` on covariance submatrices (lines 342, 357, 374, 386) — analytically exact, no
iterative convex solve, handles box constraints `lB ≤ w ≤ uB` and `Σw=1` natively. Golden-section
search locates the max-Sharpe point along the frontier (lines 289-321). **Convex solver required:
NO** (matrix inversion only) — but it is materially more code than HRP (a full turning-point state
machine) for a benefit (analytically exact mean-variance) that requires *return* forecasts as an
input, which is a harder, noisier estimation problem than the covariance-only input HRP needs
(covariance is famously easier to estimate than expected returns — the well-known reason HRP was
proposed in the first place). Rank below HRP for that reason, not because it fails the "no solver"
bar.

#### B1.3 — Analytic tangency portfolio (`w ∝ Σ^-1 μ`) — real but not found as a standalone local implementation

**NOT VERIFIED as a standalone module** in any surveyed repo — it appears implicitly inside CLA's
unconstrained-case Lagrangian (B1.2) but no repo exposes it as a bare formula. It is textbook and
implementable in pure Python via Gaussian elimination for small N, but carries the same
expected-return estimation problem as CLA, worse (no constraint handling at all unless bounds are
added back in). Not recommended as the primary build.

#### B1.4 — Convex/QP solvers — real, correctly ruled out for ARGUS's stack

- **riskparity.py** (MIT), `riskparityportfolio/sca.py:234-291` — Successive Convex Approximation,
  solves a QP each iteration via `quadprog.solve_qp` (external solver dependency, not pure Python).
- **skfolio** risk budgeting, `skfolio/optimization/convex/_risk_budgeting.py` — formulates via
  `cvxpy` (DCP), solved by ECOS/SCS/CLARABEL.
- **cvxportfolio** (Stanford/BlackRock, **GPL-3.0 — patterns only, no code lifted**) — multi-period
  convex objective with L1 transaction-cost and holding-cost terms baked directly into the
  objective (`return - risk_penalty - trade_cost(Δw) - holding_cost(w)`), solved via cvxpy each
  rebalance. This is the standard for "costs inside the objective" and is the pattern worth
  borrowing conceptually (cost terms as penalty functions added to the objective) even though the
  solve itself is out of reach for a no-cvxpy stack — HRP has no native slot for this, so if ARGUS
  wants costs inside the objective later rather than as a post-hoc filter, that is the one open
  design question HRP alone does not answer. **Convex solver required: YES for both** — correctly
  excluded from the primary recommendation.

### B2. Ranking for what to build next

1. **HRP recursive bisection** — no solver, three corroborating implementations, consumes ARGUS's
   existing covariance matrix, only needs a correlation-distance + single-linkage clustering step
   added (also implementable in pure Python — single-linkage clustering is O(n²) agglomeration,
   trivial at ARGUS's asset counts).
2. CLA, if/when return forecasts are trusted enough to use as an input — more code, same "no
   solver" property, but a harder estimation problem upstream.
3. Convex QP/DCP solvers (riskparity.py SCA, skfolio risk budgeting, cvxportfolio) — correctly out
   of scope for a pure-Python stack; useful only as design references for how transaction costs
   enter an objective function, never as code to run.

---

## What was NOT covered in this sweep

- Did not read mlfinlab's `filters/filters.py` (CUSUM *event-sampling* filter, a different, smaller
  utility than the structural-break test) — the structural_breaks stub finding above is specific to
  `structural_breaks/`, not `filters/`.
- Did not attempt an actual pure-Python performance benchmark of FLUSS or HRP at 1,400 bars — the
  "buildable in pure Python at this size" claims are based on algorithmic complexity and on ARGUS's
  own `shapematch.py` already running a comparable O(n·m) self-join in pure Python, not on a
  timed run. Treat the specific millisecond estimates the parallel readers gave (e.g. "<100ms") as NOT
  VERIFIED — no timing was actually measured in this sweep.
- Did not check FinceptTerminal (named in the prompt) — no such repo was found in
  `best-of-the-best/repos` by either parallel reader or my own listing; NOT VERIFIED / likely absent.
- Did not check darts' HMM/regime classes in detail (parallel reader survey did not return a citation for
  it) — NOT VERIFIED.
- PELT (pruned exact linear time changepoint detection) was in the prompt's list but neither
  parallel reader returned a local file:line for it — NOT VERIFIED / not found in the corpus searched.
