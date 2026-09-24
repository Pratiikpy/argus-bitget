# Pairs Trading / Statistical Arbitrage / Cointegration — Code-Level Teardown

Scope: ARGUS (`argus`, pure Python, pydantic + python-dateutil only — no
numpy/pandas/scipy). This teardown reads the actual source of the reference implementations on
this machine so ARGUS's own pure-Python cointegration/pairs engine can be built to match or beat
them, with every departure stated and justified in the code per STANDING RULE #3.

All claims below carry `file:line`. Where something could not be verified it is marked
**NOT VERIFIED**.

---

## 0. ARGUS gap — confirmed by grep, not assumed

```
grep -rniE "adfuller|cointegrat|half.?life|ornstein|johansen|kalman|z_score|zscore|hedge.?ratio|pairs.?trad|spread" argus/src
```
Every hit is either the English word "spread" (bid-ask spread in `cost/model.py`, `market/bitget.py`,
`market/collector.py`, Treasury-curve `spread_10y2y` in `market/macro.py`) or `desk/stress.py`'s
scenario "spread_treble". Zero hits for `adfuller`, `cointegrat`, `half_life`/`half-life`,
`ornstein`, `johansen`, `kalman`, `z_score`/`zscore`, `hedge_ratio`/`hedge-ratio`, or
`pairs_trad*`. ARGUS has **no** ADF test, no Engle-Granger/Johansen cointegration test, no
half-life estimator, no Kalman filter, and no pairs-trading module today.

```
find argus/src -iname "*pair*" -o -iname "*coint*" -o -iname "*stat_arb*" -o -iname "*statarb*"
```
returns only `argus/paper/repair.py` (a false-positive substring match on "pair" inside "repair")
and its `.pyc`. No pairs/cointegration/stat-arb file exists.

**What ARGUS already has that is directly reusable, found by grep, not assumed:**

- `argus/desk/diversification.py:418-451` — `minimum_variance_hedge()`: closed-form OLS hedge
  ratio `ratio = -cov(book, hedge) / var(hedge)`, computed on **returns**, full-sample, no
  look-ahead guard documented, `n-1` (sample) covariance/variance. This is the right shape of
  calculation for a *returns*-based minimum-variance hedge but is **not** what Engle-Granger needs
  — EG regresses one price **level** series on another (`y0 ~ y1`) to get a residual to test for
  stationarity; regressing returns doesn't produce a spread series to mean-revert. A pairs module
  needs a separate levels-OLS, not a reuse of this function as-is.
- `argus/backtest/validation.py:351-372` — `benjamini_hochberg()` and `bonferroni()` already
  implement FDR and Bonferroni multiple-testing correction in pure Python. `argus/backtest/
  validation.py:130-232` — `pbo` (Probability of Backtest Overfitting) with a documented
  single-estimate standard deviation (`SINGLE_ESTIMATE_SD`, cf. `validation.py:181-201`).
  `argus/backtest/metrics.py:162-166` — `deflated_sharpe()`. **All of this is exactly the
  multiple-testing correction infrastructure a pairs-scanner needs when it tests N candidate pairs
  for cointegration**, and it already exists — a new pairs module must call into it, not
  reimplement it.
- `argus/cost/model.py:240-241,277-285` — `CostModel.round_trip_bps()` and `net_edge_bps(...,
  round_trips=1)` already parameterise cost by number of round trips. A pairs trade opens 2 legs
  and closes 2 legs (4 fills); the existing API supports this exactly by passing
  `round_trips=2` (one round trip per leg) — no new cost machinery is needed, only correct usage.

---

## 1. statsmodels — the reference implementation (verified by reading the installed source)

Environment: `site-packages/statsmodels\tsa\`.

### 1.1 `adfuller()` — `stattools.py:168-372`

Signature (`stattools.py:168-175`): `adfuller(x, maxlag=None, regression="c", autolag="AIC",
store=False, regresults=False)`. Regression types (`stattools.py:189-195`): `"c"` constant only
(default), `"ct"` constant+trend, `"ctt"` constant+linear+quadratic trend, `"n"` none.

**Default maxlag formula** (`stattools.py:288-296`, exact code):
```python
if maxlag is None:
    # from Greene referencing Schwert 1989
    maxlag = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
    # -1 for the diff
    maxlag = min(nobs // 2 - ntrend - 1, maxlag)
```
i.e. `maxlag = min(floor(nobs/2) - ntrend - 1, ceil(12*(nobs/100)^0.25))`, where `ntrend =
len(regression)` if `regression != "n"` else `0` (`stattools.py:287`) — so `ntrend` is 1 for `"c"`,
2 for `"ct"`, 3 for `"ctt"`.

**Regression matrix construction** (`stattools.py:302-306`):
```python
xdiff = np.diff(x)
xdall = lagmat(xdiff[:, None], maxlag, trim="both", original="in")
nobs = xdall.shape[0]
xdall[:, 0] = x[-nobs - 1 : -1]   # replace column 0 (the un-lagged diff) with the level x_{t-1}
xdshort = xdiff[-nobs:]           # the dependent variable, Δx_t
```
This is the standard ADF regression `Δx_t = α + β·t + γ·x_{t-1} + Σ δ_i Δx_{t-i} + ε_t`; `lagmat`
builds the lag matrix of `Δx` up to `maxlag`, then column 0 (which `lagmat(..., original="in")`
would otherwise fill with the contemporaneous `Δx_t`, useless as a regressor) is overwritten with
the **level** `x_{t-1}` — this one line is the entire trick that turns a lag matrix into the ADF
design matrix.

**AIC/BIC lag-selection loop** — delegated to `_autolag()` (`stattools.py:71-155`), called from
`stattools.py:326-337`. It fits OLS once per lag length from `startlag` to `startlag+maxlag`
(`stattools.py:132-134`) and picks the lag minimizing `.aic` or `.bic` (`stattools.py:136-139`);
for `"t-stat"` it starts at `maxlag` and walks the lag count down until the t-stat on the last
included lag exceeds `1.6448536269514722` (the 95% one-sided normal critical value, hardcoded at
`stattools.py:143`), i.e. general-to-specific testing-down (`stattools.py:141-148`). After
`_autolag` returns `bestlag`, the regression is **rebuilt and refit once more** at that lag
(`stattools.py:341-346`) — the final reported ADF statistic is from this refit, not from the
search loop's cached fit.

**Test statistic**: `adfstat = resols.tvalues[0]` (`stattools.py:359`) — the t-stat on the level
coefficient `γ` (column 0), from an OLS fit with `add_trend(xdall[:, :usedlag+1], regression)` when
`regression != "n"` (`stattools.py:349-351`).

**Critical values / p-value** (`stattools.py:367-368`):
```python
pvalue = mackinnonp(adfstat, regression=regression, N=1)
critvalues = mackinnoncrit(N=1, regression=regression, nobs=nobs)
```

### 1.2 MacKinnon tables — `adfvalues.py`

`mackinnonp()` (`adfvalues.py:223-266`) is a **regression-surface response function**, not a
lookup table: response-surface coefficient arrays (`_tau_smallps`/`_tau_largeps`, built above
`adfvalues.py:60` from arrays such as `tau_c_smallp` at `adfvalues.py:52-59`) are evaluated with
`numpy.polyval` at the observed test statistic, then passed through the **standard normal CDF**
(`adfvalues.py:265`: `return norm.cdf(polyval(tau_coef[::-1], teststat))`) to produce the p-value.
Region cutoffs `tau_min_c`/`tau_max_c`/`tau_star_c` etc. (`adfvalues.py:11-20`) return exactly 0.0
or 1.0 outside the fitted range (`adfvalues.py:257-260`) and pick the small-p vs large-p polynomial
by whether `teststat <= starstat[N-1]` (`adfvalues.py:261-264`).

`mackinnoncrit(N=1, regression="c", nobs=inf)` (`adfvalues.py:407-447`) uses the **2010 MacKinnon
response-surface tables** — `tau_c_2010`, `tau_ct_2010`, `tau_ctt_2010`, `tau_nc_2010`
(`adfvalues.py:270-406`, e.g. `tau_c_2010` for N=1 at `adfvalues.py:280-282`:
`[-3.43035, -6.5393, -16.786, -79.433]` for 1%, `[-2.86154, -2.8903, -4.234, -40.040]` for 5%,
`[-2.56677, -1.5384, -2.809, 0]` for 10%). At `nobs=inf` it returns the first (constant) coefficient
of each row directly (`adfvalues.py:445-446`); at finite `nobs` it evaluates
`polyval(coeffs[::-1], 1/nobs)` (`adfvalues.py:447`) — i.e. critical value ≈ `c0 + c1/nobs +
c2/nobs² + c3/nobs³`, a finite-sample correction to the asymptotic value.

### 1.3 `coint()` — Engle-Granger, NOT the plain ADF table — `stattools.py:1702-1839`

Step by step (this is the exact procedure to reimplement):
1. `y1` is reshaped to 2-D (`stattools.py:1797`); if `trend != "n"`, `xx = add_trend(y1,
   trend=trend, prepend=False)` (`stattools.py:1809-1812`) — i.e. **the second series plus a
   constant/trend column becomes the regressor**.
2. `res_co = OLS(y0, xx).fit()` (`stattools.py:1814`) — **`y0` is regressed on `y1`** (first
   argument is the dependent variable). This fixes an asymmetry: `coint(a, b)` and `coint(b, a)`
   are not the same test and can give different p-values, because the hedge ratio and residual
   series differ depending on which leg is the regressand.
3. Collinearity guard (`stattools.py:1816-1828`): if `res_co.rsquared >= 1 - 100*SQRTEPS` the two
   series are (numerically) perfectly collinear; the code sets `res_adf = (-np.inf,)` directly
   rather than running ADF on a degenerate residual, with a `CollinearityWarning`.
4. `res_adf = adfuller(res_co.resid, maxlag=maxlag, autolag=autolag, regression="n")`
   (`stattools.py:1817-1819`) — ADF is run on the OLS **residuals**, and critically with
   `regression="n"` (no constant, no trend) — the residual of an OLS-with-constant already has
   mean zero, so adding a constant back into the ADF regression would be redundant/wrong.
5. **Critical values are NOT `mackinnoncrit(N=1, ...)`.** They are
   `mackinnoncrit(N=k_vars, regression=trend, nobs=nobs-1)` (`stattools.py:1834`), where
   `k_vars = y1.shape[1] + 1` (`stattools.py:1806-1807`) is the **total number of I(1) series in
   the system** (all columns of `y1` plus `y0`). For a two-asset pair, `k_vars=2`, so the critical
   value comes from the **`N=2` row** of the response-surface tables (e.g. `tau_c_2010[1]` at
   `adfvalues.py:283-286`: 1% = `-3.89644`, 5% = `-3.33613`, 10% = `-3.04445` asymptotically) —
   these are the Engle-Granger cointegration critical values, which are **more negative** (harder
   to reject the null of no cointegration) than the plain single-series ADF critical values
   (`N=1`, e.g. 5% = `-2.86154`), because testing a residual from an estimated relationship needs a
   wider band than testing a raw series. Reimplementing "cointegration" as plain ADF with `N=1`
   critical values on the spread residual — a mistake several of the repos below make — silently
   makes the test too liberal (over-rejects the null, finds "cointegration" that isn't there).
   `stattools.py:1832-1833` notes `nobs - 1` is used "to match egranger in Stata", with the comment
   "TODO: check nobs or df = nobs - k" — even statsmodels' own authors flag this as unresolved.
6. `pval_asy = mackinnonp(res_adf[0], regression=trend, N=k_vars)` (`stattools.py:1838`) — same
   `N=k_vars` treatment for the p-value.

### 1.4 Johansen — `coint_johansen()`, `vector_ar/vecm.py:603-737` (verified directly)

`coint_johansen(endog, det_order, k_ar_diff)`. `det_order` (`vecm.py:614-617`): `-1` no
deterministic terms, `0` constant, `1` linear trend — only these three plus `k_vars<=12` have
tabulated critical values (`vecm.py:640-648` warns otherwise). This is the standard **Anderson
(1951) reduced-rank regression** form of the Johansen procedure (Lütkepohl 2005, cited at
`vecm.py:634-635`), not a direct MLE:

1. **Detrend** (`vecm.py:656-662`): `detrend(y, order)` regresses `y` on a Vandermonde basis
   `np.vander(np.linspace(-1,1,len(y)), order+1)` via OLS and keeps the residual — this both
   demeans/detrends the level series and (via the `f` flag, `vecm.py:679-682`) the differenced
   data, depending on `det_order`.
2. **Build `Δx`, its lags, and the level lagged once** (`vecm.py:684-698`): `dx = diff(endog)`;
   `z = lagmat(dx, k_ar_diff)[k_ar_diff:]` (the RHS lag block); `dx = dx[k_ar_diff:]`; the levels
   block is `lx = endog[:len-k_ar_diff][1:]`.
3. **Residualize both `dx` and the levels on `z`** (`vecm.py:688-698`, function `resid(y,x) = y -
   x @ pinv(x) @ y`, `vecm.py:665-668`) — this is exactly the two-step Frisch-Waugh-Lovell
   partialling-out that reduced-rank regression requires: `r0t` = `Δx` purged of its own lags,
   `rkt` = level purged of the same lags.
4. **Canonical correlation eigenproblem** (`vecm.py:699-706`): `skk = rktᵀrkt/n`, `sk0 =
   rktᵀr0t/n`, `s00 = r0tᵀr0t/n` (population covariances, divided by `n` not `n-1`), `sig = sk0 @
   inv(s00) @ sk0ᵀ`, then `eig(inv(skk) @ sig)` — the eigenvalues are the squared canonical
   correlations between the levels and the differences, and the eigenvectors are the cointegrating
   vectors (candidate hedge ratios).
5. **Eigenvectors normalized via Cholesky** of `duᵀ @ skk @ du` (`vecm.py:708-709`) — a specific,
   non-arbitrary but also non-unique-up-to-scale normalization; downstream code that reads off
   `evec[:,0]` as "the" hedge ratio should know this is one valid normalization among several.
6. **Trace and max-eigenvalue statistics** (`vecm.py:722-732`): for rank `i`,
   `lr1[i] = -t*sum(log(1-a[i:]))` (trace, testing rank ≤ i) and `lr2[i] = -t*log(1-a[i])`
   (max-eig, testing rank = i vs i+1), `t = rkt.shape[0]`.
7. **Critical values are a hardcoded table, not a response surface** — `c_sja()` (max-eig) and
   `c_sjt()` (trace), imported from `statsmodels/tsa/coint_tables.py:15`. These are literal
   MacKinnon, Haug & Michelis (1996) numbers, ported from James LeSage's MATLAB `johansen()`
   (`coint_tables.py:1-36`), stored as text blocks (`ss_ejcp0/1/2`, `ss_tjcp0/1/2`,
   `coint_tables.py:42-86,155-196`) and reshaped into `(12,3)` arrays indexed `[n-1, :]` for the
   90/95/99% columns. **No interpolation for `n>12` or `det_order` outside `{-1,0,1}`** — those
   return `NaN` (`coint_tables.py:91-99`).

For a two-asset pair (`k_vars=2`), row `n=2` (i.e. `neqs - i` with `i=0`) of these fixed tables is
the relevant one, and — unlike the MacKinnon 2010 ADF/EG tables — **these do not adjust for sample
size** at all beyond the fixed table; they are asymptotic-only critical values regardless of
`nobs`.

---

## 2. Third-party pairs-trading / stat-arb implementations — read directly, `file:line` cited

Fourteen implementations were read across `okx/trading/best-of-the-best/repos` (114 repos) and
`bitget/research/repos*` (repos, repos-new, repos-themed, repos-t2 — 624 repos combined). Findings
below are pooled from three parallel parallel reader reads plus this review's own spot checks; every
claim carries the citation the reading agent reported.

### 2.1 `mlfinlab-official` — **has no pairs-trading module at all**

Verified directly in this review by listing the package tree:
`research corpus, repos/mlfinlab-official\mlfinlab\` contains
`backtest_statistics/`, `bet_sizing/`, `clustering/`, `codependence/`, `cross_validation/`,
`data_generation/`, `data_structures/`, `ensemble/`, `feature_importance/`, `features/`,
`filters/`, `labeling/`, `microstructural_features/`, `multi_product/`, `networks/`,
`regression/` (only `history_weight_regression.py`), `sample_weights/`, `sampling/`,
`structural_breaks/` (`chow.py`, `cusum.py`, `sadf.py` — explosiveness/bubble tests, not
cointegration), `util/`. **No `cointegration/`, no `pairs_selection/`, no `arbitrage/` directory
anywhere in the tree.** This confirms pairs trading and cointegration (Engle-Granger, Johansen,
Kalman-filter hedge ratios) live in Hudson & Thames's separate **paid** `ArbitrageLab` package, not
in the open-source `mlfinlab`. Treating "mlfinlab" as a pairs-trading reference (a natural
assumption, since it is the best-known quant-finance ML library on this machine) would have been
wrong; the file listing is the proof.

### 2.2 `vectorbt` — `examples/PairsTrading.ipynb` (most complete OSS example found)

- **Hedge ratio**: `rolling_ols_zscore_nb()` / `ols_spread_nb()` — Numba-JIT OLS of log-price A on
  log-price B, `slope, intercept = inv(bᵀb) @ (bᵀa)`, recomputed **every bar on a rolling window**
  (not full-sample, not Kalman/TLS). No look-ahead: the window is trailing.
- **Stationarity/cointegration test**: **none**. The notebook trades PEP/KO on the assumption they
  are cointegrated; no ADF, no Johansen is ever run.
- **Half-life**: not implemented.
- **Entry/exit**: rolling-window z-score, fixed thresholds `UPPER=+1.96`, `LOWER=-1.96` (the 97.5%
  normal quantile via `scipy.stats.norm.ppf`), computed from the same rolling `spread_mean`/
  `spread_std` used for the OLS window (no look-ahead).
- **Costs**: modeled — `fees=COMMPERC=0.5%` per order, applied at execution.
- **Multiple testing**: not applicable — a single hard-coded pair, no scan.
- **Defect**: no cointegration test before trading is the single largest gap; a hedge ratio and a
  z-score are computed for any two series regardless of whether a stationary combination exists.

### 2.3 `backtrader` — `contrib/samples/pair-trading/pair-trading.py`

- **Hedge ratio**: delegates to backtrader's own `btind.OLS_TransformationN(data0, data1,
  period=...)` indicator; the parallel reader could not read the indicator's own source in this review
  (**NOT VERIFIED** what exact OLS variant it runs).
- **Entry/exit**: hardcoded z-score bands `upper=2.1, lower=-2.1`.
- **Costs**: `cerebro.broker.setcommission(commission=args.commperc)` — modeled, but not used to
  adjust the entry threshold.
- **Concrete, provable defect**: position sizing (`x = int(value / data0.close)`) allocates equal
  **dollar** value to each leg and never multiplies by the OLS slope — if the hedge ratio isn't 1,
  the position is not actually hedged, defeating the entire point of running an OLS in the first
  place. This is a real bug, not a design choice, since the strategy computes a slope and then
  ignores it at the one place it matters.

### 2.4 `awesome-systematic-trading` (QuantConnect templates) — two strategies read

**`static/strategies/pairs-trading-with-country-etfs.py`**:
- **Pair selection**: sum-of-squared-deviations "distance" on last-close-normalized prices
  (`norm_i = price_i / price_i[-1]`), top 5 pairs by smallest distance — this is the classic
  Gatev-Goetzmann-Rouwenhorst **distance method**, not a cointegration test. Two independent random
  walks can have a small SSD by pure chance; the method has no null hypothesis or p-value.
- **Spread**: `norm_a - norm_b` — implicitly assumes hedge ratio 1, no regression at all.
- **Entry/exit**: `0.5σ` entry / mean-cross exit / 20-day time stop; mean/std computed over the
  same 120-day **formation** window used to pick the pair (not a fresh, separate calibration
  window) — a form of in-sample reuse, not classic bar-by-bar look-ahead, but the same window
  drives both selection and the trading threshold.
- **Costs**: modeled at 5bps (`fee = price * qty * 0.00005`), unusually low.
- **Multiple testing**: none — 24 ETFs → C(24,2)=276 pairs scanned by distance, no Bonferroni/FDR
  applied to the selection.

**`static/strategies/trading-wti-brent-spread.py`**:
- **Spread**: raw price difference `price1 - price2`, no hedge ratio, no normalization — WTI and
  Brent are similar-priced so this happens to be roughly sane for this one pair, but the code
  generalizes to nothing.
- **Entry/exit**: 20-day SMA mean-cross, no stationarity test, no costs modeled.

### 2.5 `QuantConnect/Lean` — two alpha models

- **`PearsonCorrelationPairsTradingAlphaModel.py`**: selects pairs by **Pearson correlation of
  price changes ≥ 0.5** — this is a materially different (and weaker) criterion than
  cointegration: correlated returns say nothing about whether the price-level spread is stationary,
  and a pair can be highly return-correlated while the spread random-walks away permanently.
- **`BasePairsTradingAlphaModel.py`**: trades the raw price **ratio** `P1/P2` against an EMA(500)
  of itself ± a percentage band — no OLS, no z-score in the statistical sense, no cointegration
  test anywhere in either model.

### 2.6 `sktime` — the one OSS wrapper of a proper cointegration test found in this pass

- `cointegration/_johansen.py:178-180` calls `statsmodels.tsa.vector_ar.vecm.coint_johansen`
  directly (`det_order`, `k_ar_diff` passed through) and exposes `lr1_`/`lr2_` (trace/max-eig) plus
  critical values — a thin, correct wrapper, not a reimplementation.
- `stationarity/_adf_arch.py` wraps the `arch` package's `arch.unitroot.ADF` (a separate,
  independent ADF implementation from statsmodels') with the same trend-type options and returns a
  boolean `stationary_` plus `pvalue_`.

### 2.7 `FinceptTerminal` — `statistical_arbitrage.py`, `mean_reversion.py`

- **Hedge ratio**: full-sample OLS (`statistical_arbitrage.py:185-192`), covariance/variance
  formula equivalent to the textbook slope estimator, computed once over the whole sample (no
  rolling window, no train/test split).
- **"Cointegration" test**: **not an ADF/EG test at all** — `statistical_arbitrage.py:206-211`
  computes the **Hurst exponent** of the spread and scores `max(0, 1 - 2*hurst)` if `hurst < 0.5`.
  Hurst-exponent mean-reversion scoring is a real, used heuristic in the literature, but it carries
  no null hypothesis, no p-value, and no formal distinction between "mean-reverting" and
  "borderline random walk that happens to look mean-reverting in this sample" — presenting it as a
  substitute for a cointegration test is the defect, not the heuristic itself.
- **Half-life**: `mean_reversion.py:35-71` — the textbook OU discretization done correctly: lag-1
  regression `Δy_t = α + β·y_{t-1} + ε`, `half_life = -ln(2)/β`, with a sanity guard rejecting
  `β ≥ 0` (non-mean-reverting) and rejecting `half_life` outside `(0, len(series))`. This is a
  clean, correct half-life implementation worth copying almost verbatim into ARGUS's pure-Python
  stack.
- **Entry/exit — confirmed look-ahead defect**: `statistical_arbitrage.py:199-203` computes
  `spread_mean`/`spread_std` over the **full history including the current bar**, then compares
  `spread[0]` (the current observation) against those same full-sample statistics. This is a
  textbook look-ahead bug: the z-score used to decide whether to enter *today* is partly computed
  *from* today's value and from the future relative to any earlier bar in a backtest that reuses
  the same full-sample stats at every step.
- **Costs**: only used in an "edge estimate" side calculation (`statistical_arbitrage.py:249-269`,
  default 10bps), never actually deducted from entry/exit decisions.

### 2.8 `ml4t-jansen` (Jansen, *Machine Learning for Algorithmic Trading*) — the strongest single implementation found, `research/repos/ml4t-jansen/09_model_based_features/14_panel_features.py`

This is the most complete and most correctly-guarded implementation found across both corpora:

- **Cointegration tests, both**: `coint(dependent, independent, trend="c")` (Engle-Granger, plain
  statsmodels call) **and** `coint_johansen(pair[["dependent","independent"]], det_order=0,
  k_ar_diff=1)` (`14_panel_features.py:168-170`) — run both rather than picking one.
- **Hedge ratio, two methods, both real**:
  - Static: full-sample `LinearRegression().fit(independent, dependent).coef_[0]`
    (`14_panel_features.py:245-247`).
  - **Kalman filter**, 2-D state `[intercept, ratio]` following a random walk
    (`14_panel_features.py:252-276`): `F=I` (random-walk transition), `Q = I * PROCESS_NOISE`
    (`1e-5`), `R = MEASUREMENT_NOISE` (`1e-3`), time-varying design matrix `H_t = [1,
    independent_t]`. This is a real, working Kalman-filter hedge ratio — not asserted, actually
    implemented — and the `PROCESS_NOISE`/`MEASUREMENT_NOISE` ratio (0.01) is the one tunable that
    is not justified anywhere in the file (**defect**: no sensitivity analysis shown).
  - **Explicit correctness note in the code** (`14_panel_features.py:208-221`): when the
    cointegration test *fails to reject* the no-cointegration null, the Johansen eigenvector is
    documented as *not* a valid hedge ratio — "it is the least badly behaved direction in a system
    that has no well behaved one" — i.e. the code itself warns against the single most common
    silent failure mode in every other repo above (using the regression coefficient as a hedge
    ratio without checking the test that licenses doing so).
- **Half-life with an actual train/test split**: AR(1) on the spread
  (`14_panel_features.py:308-319`, same `-ln(2)/ln(1+φ)` form as FinceptTerminal but stated via
  `ln(1+φ)` rather than `φ` directly — the more correct discrete-time form), estimated **only on a
  `TRAIN_FRACTION=0.5` in-sample block** (`14_panel_features.py:322-324`), then used to size the
  trading `LOOKBACK = max(2*half_life_train, MINIMUM_LOOKBACK)` window applied to the full series.
  This is the one implementation in the entire sweep that visibly guards against calibrating the
  mean-reversion speed on the same data it trades on.
- **Entry/exit**: rolling mean/std over `LOOKBACK` sessions, entry at `±2.0σ`, exit on zero-cross.
- **Costs**: **not modeled** — positions are conceptual with no fills, slippage, or commissions
  (same gap as every other repo read).
- **Multiple testing**: **not applied** — Engle-Granger and Johansen are both run on the same pair
  without correction, and there is no evidence in the read file that a Bonferroni/FDR step gates a
  multi-pair scan built on top of this module.

---

## 3. Cross-cutting findings across all fourteen implementations

**Hedge ratio.** Four different approaches were found in the wild: (a) full-sample OLS on levels
(FinceptTerminal, ml4t-jansen static), (b) rolling-window OLS on log-levels (vectorbt), (c) Kalman
filter with a random-walk state (ml4t-jansen only), (d) no hedge ratio at all — equal-dollar or
1:1 sizing (both awesome-systematic-trading strategies, QuantConnect ratio model). **Only
ml4t-jansen** runs a formal cointegration test *before* trusting the resulting ratio, and only it
documents in the code what to do when that test fails.

**Stationarity/cointegration testing.** Of fourteen implementations read: **2** run a real
Engle-Granger or Johansen test (sktime's wrapper, ml4t-jansen); **1** uses a Hurst-exponent proxy
with no hypothesis test (FinceptTerminal); **1** uses return correlation as a stand-in (QuantConnect
Pearson model); the remaining ~10 (vectorbt's example, backtrader's sample, both
awesome-systematic-trading strategies, QuantConnect's ratio model, and every other file the
parallel readers found nothing further in) run **no stationarity or cointegration test whatsoever** and
simply assume the chosen pair mean-reverts.

**Half-life.** Two correct OU-based implementations found (FinceptTerminal, ml4t-jansen), both
using the same `-ln(2)/β` (or equivalently `-ln(2)/ln(1+φ)`) form on a lag-1 AR regression of the
spread. ml4t-jansen's is strictly better because it estimates β on a training split rather than the
full series that also generates the trading signal.

**Entry/exit and look-ahead.** Rolling-window z-scores (vectorbt, ml4t-jansen, most
awesome-systematic-trading code) avoid the obvious look-ahead. **FinceptTerminal has a confirmed,
citable look-ahead bug**: full-sample mean/std computed with the current bar included, then used to
score the current bar (`statistical_arbitrage.py:199-211`). awesome-systematic-trading's ETF
strategy has a subtler in-sample-reuse issue: the same 120-day formation window both selects the
pair and calibrates the trading threshold.

**Transaction costs.** vectorbt and backtrader model fees at execution; awesome-systematic-trading
models a (too-low) 5bps fee; every other implementation (FinceptTerminal's is cosmetic-only,
ml4t-jansen, QuantConnect's two models, WTI-Brent) trades cost-free in the code that was read. **No
implementation in this sweep charges four legs for a round trip** (open 2, close 2) — where costs
are modeled at all, they are modeled per-order, which is the right unit, but nothing in any of
these files explicitly reasons about "a pairs round trip is 4 fills, not 2" the way ARGUS's own
`cost/model.py:277-285` `round_trips` parameter already allows for if used correctly.

**Multiple testing.** **Zero of fourteen implementations correct for multiple testing** when
scanning candidate pairs — not Bonferroni, not FDR, not PBO, not a deflated Sharpe. The
awesome-systematic-trading ETF strategy scans 276 pairs by distance with no correction at all,
which is the largest N found; even a single 5% significance level applied naively across that many
comparisons would produce roughly a dozen false "significant" pairs by chance alone before any
economic reasoning is applied. ARGUS already has `benjamini_hochberg()` and `bonferroni()`
(`argus/backtest/validation.py:351-372`) and PBO (`argus/backtest/validation.py:130-232`) — a new
ARGUS pairs-scanner would be the *first* implementation in this entire 14-repo, 738-repo-searched
sweep to actually gate a pairs scan with a real multiple-testing correction.

**Provable, citable defects, ranked by severity:**
1. `backtrader/contrib/samples/pair-trading/pair-trading.py` — computes an OLS slope and then
   never uses it for position sizing (equal-dollar legs regardless of the ratio). A hedge that
   isn't sized to the hedge ratio isn't a hedge.
2. `FinceptTerminal/statistical_arbitrage.py:199-211` — full-sample mean/std includes the current
   and future bars, used to score the current bar. Classic look-ahead.
3. `vectorbt`'s example, `backtrader`'s sample, both `awesome-systematic-trading` strategies,
   `QuantConnect`'s two alpha models — trade a "pair" with **no cointegration test at all**; only
   correlation, distance, or the raw price ratio license the trade.
4. `awesome-systematic-trading/pairs-trading-with-country-etfs.py` — 276 pairs scanned, zero
   multiple-testing correction, and the formation window that selects pairs is the same window
   that calibrates the entry threshold.
5. Every implementation read — **no round-trip, four-leg transaction-cost accounting** built into
   the entry-threshold decision itself (a few model fees at execution, none subtract them from the
   signal that triggers entry).

---

## 4. Exact numerical procedure to reimplement in pure Python (ARGUS-target summary)

**ADF (`adfuller`)**: default `maxlag = min(floor(nobs/2) - ntrend - 1, ceil(12*(nobs/100)^0.25))`
(`stattools.py:288-296`); build `Δx`, lag it up to `maxlag`, overwrite the unlagged column with
`x_{t-1}` (`stattools.py:302-306`); for each candidate lag from 0 to `maxlag`, OLS-fit
`Δx_t ~ [trend terms] + x_{t-1} + Σ Δx_{t-i}` and pick by AIC/BIC (`_autolag`,
`stattools.py:71-155`) or by testing down on the last lag's t-stat against `1.6448536269514722`;
refit once at the chosen lag; test statistic is the t-stat on `x_{t-1}`'s coefficient
(`stattools.py:359`); p-value and critical values from the MacKinnon response-surface polynomials
(`adfvalues.py:223-266,407-447` — the coefficient arrays there must be ported verbatim, they are
small enough to hardcode in pure Python).

**Engle-Granger (`coint`)**: regress `y0` on `[y1, trend-terms]` by OLS (`stattools.py:1814`);
guard near-perfect collinearity (`stattools.py:1816-1828`); ADF the residuals with
`regression="n"` (`stattools.py:1817-1819`); **use the `N=k_vars` (not `N=1`) row** of the
MacKinnon critical-value/p-value tables, `k_vars` = total series in the system
(`stattools.py:1806-1807,1834,1838`) — this is the step every non-statsmodels implementation in
section 2 skips or gets wrong when it hand-rolls a "cointegration test" as plain ADF on a spread.

**Johansen (`coint_johansen`)**: detrend both levels and differences via OLS-on-Vandermonde
(`vecm.py:656-662`); build `Δx` and its `k_ar_diff` lags; residualize `Δx` and the once-lagged
level against the lag block (Frisch-Waugh-Lovell, `vecm.py:665-668,688-698`); solve the
generalized eigenproblem `inv(Skk) @ Sk0 @ inv(S00) @ Sk0ᵀ` (`vecm.py:699-706`); Cholesky-normalize
eigenvectors (`vecm.py:708-709`); trace/max-eig statistics as in section 1.4 step 6; compare
against the hardcoded MacKinnon-Haug-Michelis (1996) 12×3 tables (`coint_tables.py:42-86,155-196`
— must be ported verbatim; they are not response-surface, they are literal numbers with no formula,
and cap out at 12 series and three fixed `det_order` values).

**Half-life**: lag-1 AR regression of the spread, `Δs_t = α + β·s_{t-1} + ε_t`,
`half_life = -ln(2)/β` (reject if `β ≥ 0`) — verified correct and directly portable from
`FinceptTerminal/mean_reversion.py:35-71` and cross-checked against `ml4t-jansen/
14_panel_features.py:308-319`'s equivalent `-ln(2)/ln(1+φ)` form.

---

## Summary: what to copy, what to avoid

See the closing chat summary for the ranked five-and-five list; every item there cites back to a
section above.

---

## What ARGUS built from this, and what it refused

`argus/src/argus/research/cointegration.py`, 49 tests in `argus/tests/test_cointegration.py`.
Registered in `argus/status.py`, wired as its own step in `desk/research.py:cointegration_step`,
and run live over all twelve rTokens (`argus/data/cointegration.json`).

**Taken, with the reference read first**

| Taken | From | Ours |
|---|---|---|
| ADF regression shape: diff, lag, overwrite column 0 with the level | `statsmodels/tsa/stattools.py:304-307` | `cointegration.py:_design` |
| `maxlag = ceil(12*(n/100)**0.25)`, capped at `n//2 - ntrend - 1` | `stattools.py:289,124` | `cointegration.py:adf` |
| AIC scored on a fixed sample, winner refitted on the longer one | `stattools.py:326-346` | `cointegration.py:adf` |
| MacKinnon 2010 critical-value and p-value tables, incl. the `1e-2`/`1e-1` scalings | `adfvalues.py:42,52-59,87,97-104,282-296` | `cointegration.py:TAU_*` |
| Engle-Granger: OLS then ADF on residuals with `regression="n"`, N=2 table | `stattools.py:1702 (coint)` | `cointegration.py:engle_granger` |
| The unexplained `nobs - 1` in the critical-value lookup | `stattools.py:1835` ("to match egranger in Stata, I do not know why") | copied, and labelled as theirs |
| OU half-life `-ln(2)/β` from the AR(1) of the change on the lagged level | `FinceptTerminal/mean_reversion.py:35-71` | `cointegration.py:half_life` |

**The reproduction is the test, not a claim.** `tests/test_cointegration.py::TestItReproducesStatsmodels`
runs 10 unit-root tests and 3 cointegration tests against values produced by statsmodels 0.14.6 on
this machine and stored in `tests/data/cointegration_expected.json`. Statistic, p-value, selected
lag, sample size and all three critical values agree to **1e-12**. A hand-rolled ADF that is subtly
wrong still returns a number; this is the only way to know ours is not one.

**Refused, and why**

- **Full-sample hedge ratios.** `pair_test` fits β on a training slice and freezes it before the
  held-out spread is tested. Refitting on the test slice makes the out-of-sample ADF a second
  in-sample test with different labels — the defect at `FinceptTerminal/statistical_arbitrage.py:199-211`
  and in the country-ETF scanner.
- **Whole-sample z-scores.** `zscores` uses only bars strictly before each point and returns `None`
  until the lookback is filled, rather than computing an early z from twenty points.
- **Two-leg cost accounting.** `LEGS_PER_ROUND_TRIP = 4`. Backtrader's sample and the QuantConnect
  alphas charge the spread, not the four crossings.
- **Uncorrected scans.** 12 instruments is 66 hypotheses; `ScanReport` carries the hypothesis count
  through `narrow()` so a per-symbol view still corrects against 66, not 11. The correction itself is
  imported from `backtest/validation.py:351,372` rather than reimplemented — a second copy is how two
  parts of one system come to disagree about what survived.
- **Johansen.** Not built. The MacKinnon-Haug-Michelis tables are literal numbers with no response
  surface, and with 12 instruments and no evidence of even pairwise cointegration, a multivariate
  test would add a second family of hypotheses to a search that has already found nothing. Stated
  here as a deliberate omission rather than left as a silent gap.
- **A Kalman-filtered hedge ratio.** Worth building only if a pair first survives the static test.
  None did.

**What the live run found.** 66 pairs, 60 days of hourly closes: **1** reached p ≤ 0.05 naively
against **3.3 expected by chance**, **0** survive Benjamini-Hochberg at q=0.05, **0** survive
Bonferroni, **0** are tradeable. Fewer naive hits than chance predicts is itself the finding: there
is no pairwise cointegration among these instruments at this frequency.

And the binding constraint is the opposite of the one `research/arbitrage_study.py` found for the
basis. There the fee was larger than the entire effect. Here the median pair needs only `|z| >= 0.08`
to clear 24bps, because the spreads are wide — and wide spreads that do not revert are divergence
risk, not edge. `ScanReport.binding_constraint` says so in the artefact, because reporting the low
cost hurdle without that sentence is an invitation to trade the worst pairs hardest.
