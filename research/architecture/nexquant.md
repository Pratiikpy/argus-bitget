# NexQuant: Architecture Teardown for ARGUS
**A Numba/Optuna/LightGBM Explore-Exploit Loop for FX/Crypto Indicator Strategies**

---

## 1. Identity

**Name:** NexQuant

**Authority:** `NicolasBohn/NexQuant` (repo remote shows org `TPTBusiness`; CI badges point to `github.com/TPTBusiness/NexQuant`)

**Domain:** High-speed, non-LLM search over technical-indicator trading rules (FX majors + BTC/USD + XAU/USD), described in its own README as "no LLM required."

**Entry Point (the real engine):** `scripts/nexquant_rd_loop.py:main()` (lines 841-927), invoked as `python scripts/nexquant_rd_loop.py --iterations 10000`.

**Critical scope note, stated by the repo itself (`README.md`, "Project Structure" section):**
> "The LLM-based R&D framework (`rdagent fin_quant`) is part of the codebase but the Qlib/CoSTEER pipeline currently produces zero factors. The primary strategy discovery path is the Numba-based loop in `scripts/`."

The `rdagent/` directory is a vendored fork of Microsoft's RD-Agent (already torn down separately in `rd-agent.md`) and is **dead code for discovery purposes** by the authors' own admission. This teardown therefore treats `scripts/nexquant_rd_loop.py` as the subject — it is what the task description ("explore/exploit loop over Optuna + LightGBM") actually refers to. One exception: `rdagent/components/backtesting/vbt_backtest.py` is live code, imported by the *offline re-validation* scripts (`nexquant_rebacktest_unified.py`, `nexquant_gen_strategies_real_bt.py`) and is materially more rigorous than the search-time engine — this split is important and is covered in §7.

---

## 2. Licence

**SPDX Identifier:** AGPL-3.0

**File:** `LICENSE` at repository root — full GNU Affero General Public License v3, confirmed verbatim (Free Software Foundation boilerplate, network-use copyleft clause present). Confirmed again in `README.md`: "**License** — GNU Affero General Public License v3.0 (AGPL-3.0)."

**Implication for ARGUS:** AGPL-3.0 is copyleft and has a network-service trigger clause (§13) — running a modified version as a network-accessible service obliges source disclosure to users of that service. Nothing from this repo's AGPL-licensed code (`rdagent/components/backtesting/vbt_backtest.py`, `rdagent/components/backtesting/verify.py`, or any of `scripts/`) may be copied into ARGUS verbatim or in derivative form without triggering that obligation. Everything below is therefore read for **ideas and patterns to reimplement independently**, never for code to port. The README itself acknowledges the live-trading module is kept in a separate closed-source `git_ignore_folder/` specifically to avoid this — evidence the authors know AGPL forces disclosure and structured the repo to keep their alpha out of it.

---

## 3. Full Architecture

### 3.1 Module Graph (of the live engine)

```
scripts/
├── nexquant_rd_loop.py          # THE ENGINE — explore/exploit/Optuna/LightGBM loop (931 lines)
├── nexquant_priceaction_loop.py # deterministic grid search variant, not the R&D loop
├── nexquant_portfolio_optimizer.py  # post-hoc greedy correlation-aware selection
├── nexquant_rebacktest_unified.py   # OFFLINE re-validation of already-discovered strategies
│                                     # → imports rdagent/components/backtesting/vbt_backtest.py
└── nexquant_gen_strategies_real_bt.py  # LLM-based strategy generation (separate path, uses same
                                          # vbt_backtest.py validator)

rdagent/components/backtesting/
├── vbt_backtest.py   # backtest_signal(), backtest_signal_risk(), walk_forward_rolling(),
│                       # monte_carlo_trade_pvalue() — used ONLY by the offline re-validators above,
│                       # NEVER by nexquant_rd_loop.py's search loop
└── verify.py         # verify_backtest_result() — fast invariant/sanity checks on a result dict
```

There are, in effect, **two unrelated backtest engines in this repository**: a crude Numba kernel that runs inside the live search loop (`_backtest_numba`, `nexquant_rd_loop.py:43-96`), and a materially more careful vectorised engine (`vbt_backtest.py`) that is applied only *after* the search loop has already picked winners. The search never sees the better engine's numbers. This split matters for every answer below.

### 3.2 Data Flow — One Iteration of the Live Loop

```
main() [nexquant_rd_loop.py:841]
  │
  ├─ load_data() [:776]  → dict {instrument: close_series} for EURUSD/GBPUSD/BTCUSD/XAUUSD
  │
  ├─ loop = ResearchLoop(closes) [:859, class at :435]
  │
  └─ for i in range(iterations):                                    [:862]
       │
       ├─ hp = loop.hypothesize()                                    [:863, method at :448]
       │      → picks one of: random ("explore"), mutate-top-5-SOTA ("exploit"),
       │        force-different-indicator (every 100th), Optuna-refine-#1 (every 500th),
       │        LightGBM-on-top-5 (every 2000th)
       │
       ├─ result = evaluate_multi(closes, hp, use_session=True, use_vola=True)  [:867, fn at :314]
       │      → builds signal, applies session/vola/news/cross-confirm filters,
       │        splits IS/OOS (80/20), runs _backtest_numba on IS, OOS, and FULL series
       │
       ├─ result['hypothesis'] = hp; loop.history.append(result)     [:871-874]
       │
       ├─ is_new_best = loop.feedback(result)                        [:877, method at :614]
       │      → gate: sharpe > 0.3 AND n_trades >= 10                [:616, MIN_SHARPE/MIN_TRADES at :35]
       │      → score = composite_score(result, self.sota_equity)    [:619, fn at :420]
       │      → dedup vs. top-3 SOTA by structural similarity        [:623-629, _similar at :645]
       │      → self.sota.append(result); sort by composite_score DESC; keep top 30  [:632-635]
       │
       └─ every 100 iters: loop.record() writes SOTA (minus equity curves) to JSON  [:894-895, :657]
```

---

## 4. THE RESEARCH LOOP — The Deep Section

### 4.1 Stage 1: Hypothesis Generation

**File:** `scripts/nexquant_rd_loop.py:ResearchLoop.hypothesize()` (lines 448-485)

Five distinct generation modes, chosen by iteration-count modular arithmetic and dice roll — **not** an LLM call anywhere in this path:

1. **`ml`** (every 2000th iteration, if `len(self.sota) >= 5`): returns a synthetic hypothesis wrapping `{'sota': self.sota[:5]}` — line 452-455. Consumed by `_train_ml()`.
2. **`optuna`** (every 500th, if `self.sota` non-empty): copies `self.sota[0]['hypothesis']` (the single top-ranked strategy) — line 458-462. Consumed by `_run_optuna()`.
3. **`explore`-forced-diversity** (every 100th, if `len(self.sota) >= 5`): calls `_top_indicator()` then `_force_different_indicator()` to push a random hypothesis away from whichever indicator dominates the current SOTA list — lines 464-470.
4. **Adaptive `explore`**: if the top indicator dominates more than 80% of the current top-10 SOTA, temporarily raise `effective_rate` by 0.25 — lines 472-479.
5. **`explore` vs. `exploit`** (the base case): `random.random() < effective_rate` → `_random_hypothesis()` (fresh random draw); else → `_mutate_hypothesis(random.choice(self.sota[:5])['hypothesis'])` — lines 481-485.

### 4.2 Stage 2: The Candidate Representation

**File:** same, `_random_hypothesis()` (lines 517-541), `_random_params()` (lines 594-612)

A candidate is a plain, JSON-serialisable Python `dict`. There is no formula search, no symbolic regression, no arbitrary feature construction — the search space is **combinatorial selection over three fixed axes**: a strategy `type`, one or two named indicators from a 15-item pool, one or more of 7 timeframe strings, and a small discrete parameter grid per indicator (hand-coded `random.choice` lists, not continuous ranges — e.g. MACD's `fast` is drawn from literally `[3,5,8,12]`).

Quoted representation, `type='single'` (line 521-522):
```python
{'type': 'single', 'indicator': ind, 'timeframe': tf,
 'params': self._random_params(ind),
 'description': f"{ind} on {tf}", 'generation': 'explore'}
```
`type='multi_role'` (lines 535-540), a trend-filter/entry-trigger pair:
```python
{'type': 'multi_role',
 'trend_ind': trend_ind, 'trend_params': self._random_params(trend_ind), 'trend_tf': trend_tf,
 'entry_ind': entry_ind, 'entry_params': self._random_params(entry_ind), 'entry_tf': entry_tf,
 'description': f"{trend_ind}({trend_tf})→{entry_ind}({entry_tf})", 'generation': 'explore'}
```
`INDICATORS_POOL` (line 31) has 15 entries; `TIMEFRAMES` (line 30) has 7; `STRATEGY_TYPES` (line 32) has 3. Mutation (`_mutate_hypothesis`, lines 543-592) perturbs one field at a time: swap the indicator, swap the timeframe, or scale one numeric parameter by `random.uniform(0.5, 1.5)` (lines 563, 567, 578). This is a textbook micro-genetic-algorithm operating directly on this dict, not a factor-formula DSL.

### 4.3 Stage 3: Execution — `_backtest_numba`

**File:** `scripts/nexquant_rd_loop.py:43-96`, called from `evaluate_multi()` (lines 314-389)

`@jit(nopython=True)` kernel. Enters/exits a single net position on signal flips, tracks equity, drawdown, per-trade returns, and a naive Sharpe (`mean/std * sqrt(trade_count)` — scaled by **trade count**, not by calendar time; line 92). Deterministic, no LLM in this stage — matches RD-Agent's pattern of a deterministic runner, but the runner itself is far less careful (see §6).

### 4.4 Stage 4: Feedback / Selection

**File:** `ResearchLoop.feedback()` (lines 614-643)

```python
def feedback(self, result):
    if result['sharpe'] <= MIN_SHARPE or result['n_trades'] < MIN_TRADES:
        return False
    score = composite_score(result, self.sota_equity)
    result['composite_score'] = float(score)
    is_diverse = True
    if self.sota:
        for existing in self.sota[:3]:
            if self._similar(result, existing):
                is_diverse = False; break
    if is_diverse:
        self.sota.append(result)
        self.sota.sort(key=lambda r: r.get('composite_score', 0), reverse=True)
        self.sota = self.sota[:30]
        self.sota_equity = [s['equity_curves'] for s in self.sota]
        if score > self.best_score:
            self.best_score = score
            return True
    return False
```
There is no separate evaluator anywhere — the same numeric routine that scores a candidate (`composite_score`, §5) is what admits it into the pool that the next generation samples from. This is deterministic, not LLM-mediated — worse in one specific sense than RD-Agent's LLM proposer/evaluator conflation (RD-Agent at least filters its bias through natural-language judgment that occasionally refuses); here the fitness function *is* the selection rule, full stop.

---

## 5. SEARCH/EVALUATION SEPARATION — THE DECISIVE QUESTION

### 5.1 The Question

Does the thing that proposes candidates ever see the scores of previous candidates?

### 5.2 The Finding: **YES — total and structural, worse than an LLM feedback loop**

This is not a subtle information leak mediated by an LLM's judgment (as in RD-Agent). It is a **literal fitness-proportionate genetic algorithm**: the object the proposer samples from (`self.sota`) is a list of full evaluation results — not just hypotheses — kept sorted by the very score being optimised, and every generation path reads that sorted list directly.

**Evidence 1 — elitist parent selection is score-sorted by construction**

`feedback()` line 633: `self.sota.sort(key=lambda r: r.get('composite_score', 0), reverse=True)`. `hypothesize()` line 484: `base = random.choice(self.sota[:5])` — the five hypotheses eligible to be mutated are, by definition, the five with the highest `composite_score` seen so far. The proposer does not merely "see" past scores; the past scores **are the selection mechanism** determining which hypotheses get to reproduce.

**Evidence 2 — the "diversity-forcing" branch reads the rank-#1 result directly**

`_top_indicator()` (lines 487-491): `return self.sota[0]['hypothesis'].get('trend_ind', ...)` — `self.sota[0]` is, after the sort in `feedback()`, the single highest-scoring strategy. This value is used (line 466: `top = self._top_indicator()`) to bias every-100th-iteration proposals away from whatever already won, and again (lines 474-479) to raise the exploration rate when the top scorer's indicator dominates. Both are explicit, score-derived steering of the generator.

**Evidence 3 — Optuna's objective function *is* the metric being searched for**

`_run_optuna()` (lines 671-717), objective at lines 690-701:
```python
def objective(trial):
    params = {...}  # trial.suggest_int / suggest_float over param_ranges
    result = evaluate_multi(closes, hp, use_session=True, use_vola=True)
    return float(result.get('sharpe', 0)) if result.get('sharpe', 0) > 0 else -999.0
```
This is TPE Bayesian optimisation with the in-sample-contaminated Sharpe (see §6) as the *direct* maximisation target, run for 15 trials on the already-rank-1 hypothesis (`hp = dict(self.sota[0]['hypothesis'])`, line 459) every 500 iterations. There is no framing in which this "does not see" past scores — seeing and greedily maximising past scores is the entire mechanism.

**Evidence 4 — LightGBM is trained on the score-selected pool**

`_train_ml()` (lines 720-773): `sota = hypothesis.get('sota', [])` (line 727) is `self.sota[:5]` passed in at line 455 — again, the top-5-by-`composite_score` strategies. Their signals become the classifier's input features (lines 735-743); the label is next-bar direction (line 749). The model is therefore trained to recombine exactly the indicators that already won, using the identical evaluation window that produced their scores.

### 5.3 The Implication

There is no proposer/evaluator boundary in this system at all — describing it as one is generous. It is a single fitness function (`composite_score`) driving (a) elitist selection of mutation parents, (b) anti-domination steering, (c) Bayesian hyperparameter search, and (d) supervised-learning feature selection, all directly and all deterministically. Over the stated 10,000-50,000-iteration runs, this is the maximal-strength version of the defect ARGUS's `factor_lab.py` `ProposerContext` is explicitly built to prevent — NexQuant is what factor_lab.py would look like if the frozen-context field genuinely did carry a performance number and the proposer greedily climbed it.

**Compare to ARGUS:** `argus/research/factor_lab.py`'s `ProposerContext` has no field a score could occupy — this NexQuant loop is the concrete demonstration of what that omission is defending against; every one of its four generation modes is exactly the shape of exploit ARGUS's design forecloses.

---

## 6. Overfitting Controls — Exhaustive Check

| Control | Status | Evidence |
|---|---|---|
| **Purged cross-validation** | **NOT FOUND** | `grep -rniE "purg\|cscv" scripts/ nexquant.py` — zero matches. |
| **Embargo period** | **NOT FOUND** | `grep -rniE "embargo" scripts/ nexquant.py` — zero matches. |
| **Deflated / Probabilistic Sharpe** | **NOT FOUND** | `grep -rniE "deflat\|probabilistic sharpe\|\bpsr\b\|\bdsr\b"` — zero matches anywhere in `scripts/` or `nexquant.py`. The only Sharpe computed is the naive `mean/std*sqrt(trade_count)` at `nexquant_rd_loop.py:92`, never adjusted for the number of trials run. |
| **PBO (Probability of Backtest Overfitting)** | **NOT FOUND** | `grep -rniE "\bpbo\b\|bailey\|overfit"` — the single hit is a comment string in `nexquant.py:625` ("This reduces overfitting risk...") describing an unrelated trade-frequency cap, not a PBO calculation. |
| **Trial counter / multiple-testing correction** | **NOT FOUND as a correction.** `self.iteration` (line 449, incremented every call) is used only for **scheduling** (`% 500`, `% 2000`, `% 100` at lines 452, 458, 464) — never fed into a Bonferroni, Šidák, or deflated-Sharpe adjustment. A run of 50,000 iterations (the README's own example) applies **zero** statistical penalty for having tried 50,000 things. |
| **Placebo / permutation test** | **NOT FOUND in the search loop.** `grep` turns up `mc_n_permutations` only in the *offline* re-validators (`nexquant_gen_strategies_real_bt.py:398`, `nexquant_rebacktest_unified.py:192`), which call into `rdagent/components/backtesting/vbt_backtest.py:monte_carlo_trade_pvalue()` (lines 351-390) — and that function's own docstring admits: *"The `n_permutations` parameter is kept for API compatibility but is unused"* (line 362). It does not permute anything; it runs `scipy.stats.binomtest(n_wins, n_total, p=0.5, alternative="greater")` (line 389) — a one-sided binomial test on trade win-rate, mislabelled as a Monte Carlo permutation test. This is a genuine, checkable defect: the function name and the 200-permutation call-site argument (`nexquant_rebacktest_unified.py:192`) actively promise something the implementation does not do. |
| **Single static OOS split** | **Present but not gating.** `OOS_SPLIT = 0.2` (line 37); `evaluate_multi()` computes `sh_is`/`sh_oos`/`monthly_oos` per instrument (lines 343-370). But the number that actually gates acceptance — `feedback()`'s `result['sharpe'] <= MIN_SHARPE` check (line 616) and the `composite_score()` input (line 422) — is `result['sharpe']`, which traces to `results[inst]["sharpe"] = float(sh_full)` (line 366), i.e. the **full-period** backtest that already contains the nominal OOS segment. `composite_score()` (lines 420-429) only lightly discounts this via `oos_ratio = monthly_oos/monthly_pct` as a multiplicative factor bounded to `[0.3, 1.0]` (line 429) — a strategy that is pure noise out-of-sample is not rejected, at most scored down 70%. The IS/OOS split exists for reporting; it is not enforced as a selection gate. |
| **Rolling walk-forward with multiple windows** | **NOT FOUND in the search loop; present in the separate offline validator.** `walk_forward_rolling()` (`vbt_backtest.py:393-466`) runs true sequential multi-year IS/OOS windows and reports `wf_oos_consistency` (fraction of windows with positive OOS Sharpe, line 464) — a materially better construct than the search loop's single 80/20 split. It is invoked only from `backtest_signal_risk()` (line 469, `wf_rolling=True` default) inside the offline `nexquant_rebacktest_unified.py` path, never from `nexquant_rd_loop.py`. |

**Conclusion:** the live discovery loop that the README calls "the primary strategy discovery path" has **no working overfitting control of any statistical kind**. Its one nominal safeguard (IS/OOS split) is computed but not enforced. A materially better validator exists in the same repository (`vbt_backtest.py`) but is architecturally disconnected from the search — it only ever looks at strategies *after* the uncorrected loop has already picked them, at which point 30 winners have already survived up to 50,000 uncorrected trials of elitist selection on in-sample-contaminated Sharpe.

---

## 7. Cost Modelling

**Search-time engine:** `_backtest_numba(prices, signals, cost=0.000264)` (`nexquant_rd_loop.py:44`) — a hardcoded default of **2.64 bps**, subtracted once per trade close in both directions (lines 55-56, 75-76: `ret = (px - entry_price)/entry_price - cost`). Flat, instrument-agnostic, no slippage, no spread widening under volatility, no funding/carry, no maker/taker distinction — just a constant scalar. Never varied per instrument (EURUSD, GBPUSD, BTCUSD, and XAUUSD all charged the identical 2.64 bps), despite BTCUSD and XAUUSD having structurally different liquidity/spread profiles than EURUSD.

**Offline validator engine:** `DEFAULT_TXN_COST_BPS = 2.14` (`vbt_backtest.py:37`), documented as "≈2.35 pip spread+slippage+commission" on EUR/USD (line 487). Applied properly with a 1-bar-lagged position (`position = signal.shift(1)`, line 150) and cost charged proportional to `|Δposition|` (line 157-159) — the more defensible of the two implementations, but again, it is never in the search loop's path.

**`data_config.yaml`** (repo root) declares `spread_bps: 1.5`, `commission_bps: 0.0` — a *third*, even lower number, attached to the (dead, per README) `rdagent` qlib pipeline, not consumed by either backtest engine described above.

**Verdict:** three different, mutually inconsistent cost assumptions exist in the repository (2.64 bps hardcoded in the live search, 2.14 bps in the offline validator, 1.5 bps in an unused config file), none of them modelling slippage, market impact, or the with/without-cost consistency ARGUS's overfit discipline requires. None reach the ~12 bps round-trip taker fee this desk actually measured — every number here understates real cost by 4.5-8×.

---

## 8. Look-Ahead — A Reproduced, Concrete Defect

**Location:** `build_signal()`, `nexquant_rd_loop.py:102-157`, all three strategy types (lines 128, 136-138, 147-152):
```python
bars = close.resample(tf).last().dropna()
sig = _build_indicator_signal(ind, bars, hypothesis['params'])
signal = sig.reindex(close.index).ffill().fillna(0).astype(int).clip(-1, 1)
```

`pandas.Series.resample()` for sub-daily fixed offsets (`'1h'`, `'30min'`, `'4h'`, etc.) defaults to `label='left'` — a bin timestamped `10:00` aggregates data from `10:00` through `10:59`, and `.last()` returns the value **at the close of that bin** (≈10:59), not at its label time. Reindexing that resampled series back onto the raw per-minute index with `.ffill()` then assigns the 10:59 value to every raw timestamp from `10:00` onward — including `10:00:00` itself, a full hour before that value existed.

**Reproduced empirically** (not asserted from memory, per house rule):
```python
>>> idx = pd.date_range('2024-01-01 09:00', periods=180, freq='1min')
>>> s = pd.Series(range(len(idx)), index=idx)
>>> r = s.resample('1h').last()
2024-01-01 09:00:00     59
2024-01-01 10:00:00    119   # this is the value at 10:59, labeled 10:00
>>> r.reindex(idx).ffill()['2024-01-01 09:58':'2024-01-01 10:05']
09:58:00     59.0
09:59:00     59.0
10:00:00    119.0   # ← known only at 10:59, visible at 10:00:00 sharp
10:01:00    119.0
...
```

This is a systemic, one-bar-forward look-ahead baked into **every** multi-timeframe indicator signal the loop can generate — `single` (line 128), `multi_tf` (lines 136-138), and `multi_role` (lines 147-152) all route through the identical `resample(tf).last()` → `reindex().ffill()` pattern, and no `.shift(1)` or equivalent correction appears anywhere in `build_signal()`. This directly explains why the README's headline numbers diverge so sharply — "+32.0%/month (Numba)" vs. "+24.3%/month (verified independent backtest)" — the independent verifier (`vbt_backtest.py`, which *does* apply `signal.shift(1)` at line 150) removes exactly this leak, and the strategy's edge nearly collapses by a third even before considering that the "independent" run still uses only a single static split, not a purged/embargoed one.

**Also checked and clear:** no global normalisation before splitting (`evaluate_multi()` computes all indicator series from the untouched `close` Series per-instrument, independently — no cross-sectional or dataset-wide scaling that could leak future statistics into a training fold); no `fillna`/`bfill` used anywhere in `build_signal()` (only `ffill`, which is causal by construction *given a correctly-labelled resample* — the defect above is entirely the resample-label issue, not a `bfill` leak).

---

## 9. The Three Mechanisms Genuinely Worth Studying

Being blunt, as instructed: nothing here can be copied outright (AGPL-3.0, §2), and most of what looks clever is entangled with the search/evaluation violation in §5. Three ideas are worth the time to reimplement independently, with caveats attached to each:

1. **Correlation-penalised composite scoring for a *portfolio*, not a search fitness** — `composite_score()` (`nexquant_rd_loop.py:420-429`): `sharpe × (1 − 0.5·|corr to existing SOTA equity curves|) × (0.3 + 0.7·oos_ratio)`. The idea of explicitly penalising a new candidate for correlating with an already-accepted equity curve (`correlation_penalty()`, lines 392-417, computed against every retained SOTA strategy's actual return series) is a genuinely useful diversification signal — but **only** as a post-hoc portfolio-construction step over already-OWNED factors, never as fitness feeding a generator (that reintroduces §5's defect verbatim). ARGUS's `factor_lab.py` proposer context has no such field today; the correlation-vs-portfolio number belongs in a downstream allocation step, not upstream of proposal.
2. **Rolling walk-forward with a reported consistency ratio** — `walk_forward_rolling()` (`vbt_backtest.py:393-466`) runs sequential 1-year-IS/1-year-OOS windows stepped forward across the whole history and reports `wf_oos_consistency` = the fraction of windows with positive OOS Sharpe (line 464), alongside `wf_oos_sharpe_mean`/`std`. This is a more informative single number than a one-shot 80/20 split, and worth comparing directly against `argus/research/overfit.py`'s sub-sample stress gate to confirm ARGUS already reports an equivalent windowed-consistency figure (not established here — NOT VERIFIED, since `overfit.py`'s internals were not read for this teardown) rather than assuming a gap exists.
3. **A negative lesson worth converting into a regression test, not a mechanism**: `monte_carlo_trade_pvalue()`'s docstring admits its own `n_permutations` argument is dead (§6) — it silently runs a binomial test while a caller three files away passes `mc_n_permutations=200` believing 200 permutations are happening. The reusable takeaway for ARGUS is procedural: any function in `argus/research/overfit.py` whose name promises a resampling/permutation count must have a test asserting the RNG is actually invoked `n` times and that varying `n` changes the result distribution — exactly the kind of "does it do what its name says" check this bug would have caught.

---

## 10. WHAT BREAKS — Defects Found by Reading Code

| Defect | File:Line | Severity | Impact |
|---|---|---|---|
| **No search/evaluation separation — deterministic fitness drives selection, hyperparameter search, and ML training** | `nexquant_rd_loop.py:484` (elitist parent choice), `:487-491` (rank-0 read), `:690-701` (Optuna objective = raw Sharpe), `:727-743` (LightGBM trained on top-5) | **CRITICAL** | This is not bias risk, it is the entire mechanism. Every accepted strategy is, by construction, whatever the fitness function could climb highest across up to 50,000 trials. |
| **No multiple-testing correction of any kind** | absent (§6) | **CRITICAL** | 15 indicators × 7 timeframes × per-indicator param grids × 3 strategy types × up to 50,000 iterations, zero correction. At that trial count, a Sharpe > 0.3 gate (`MIN_SHARPE`, line 35) will be cleared by pure noise routinely. |
| **Look-ahead bias in every multi-timeframe signal** | `nexquant_rd_loop.py:128, 136-138, 147-152` | **CRITICAL** | `resample(tf).last()` labels bins by their left (start) edge while `.last()` returns the value at the bin's right (close) edge; `reindex().ffill()` then back-fills that future-dated value onto the raw index from the bin's start. Reproduced empirically in §8. Explains the reported 32%→24.3% gap between the Numba loop and the (partially) corrected external check. |
| **OOS split computed but not enforced as a gate** | `nexquant_rd_loop.py:616, 366, 380, 422` | **HIGH** | `feedback()`'s acceptance threshold and `composite_score()`'s primary term both read the full-period (IS+OOS) Sharpe; OOS performance only lightly discounts the score via a `[0.3, 1.0]`-bounded multiplier, never rejects. |
| **"Monte Carlo" permutation test does not permute** | `rdagent/components/backtesting/vbt_backtest.py:351-390`, esp. `:362` (docstring admission) | **HIGH** | Callers pass `mc_n_permutations=200` (`nexquant_rebacktest_unified.py:192`) believing a permutation test ran; the function performs a one-sided binomial test on win-rate instead and the `n_permutations` argument is unused. Any downstream report or claim that cites this "p-value" as a permutation-test result is false. |
| **Three inconsistent, all-too-low cost assumptions** | `nexquant_rd_loop.py:44` (2.64bps), `vbt_backtest.py:37` (2.14bps), `data_config.yaml` (1.5bps) | **HIGH** | None models slippage or market impact; none is instrument-specific despite covering FX, gold, and BTC; all are 4.5-8× below this desk's measured 12bps round-trip taker fee. |
| **Naive Sharpe scaled by trade count, not calendar time, at search time** | `nexquant_rd_loop.py:92` | **MEDIUM** | `sharpe = mean/std * sqrt(trade_count)` — a strategy that trades more often (regardless of edge) mechanically inflates its own Sharpe under this formula, independent of the annualisation the offline validator does correctly (`vbt_backtest.py:166`, `sqrt(bars_per_year)`). |
| **String comparison used for timeframe ordering** | `nexquant_rd_loop.py:534, 558, 560` | **MEDIUM** | `entry_tf = random.choice([t for t in ENTRY_TFS if t < trend_tf])` compares timeframe strings lexicographically, not by duration. Verified: for every `trend_tf ∈ {'30min','1h','4h'}`, `'5min' < trend_tf` is `False` under Python string ordering (`'5' > '3'/'1'/'4'` as the first character), so `'5min'` can **never** be selected as an entry timeframe in any `multi_role` hypothesis — a third of `ENTRY_TFS` is silently dead regardless of the intended "entry TF shorter than trend TF" invariant. |
| **No capacity/decay/retirement on the SOTA pool beyond a flat top-30 cap** | `nexquant_rd_loop.py:634` | **LOW** | `self.sota = self.sota[:30]` is a hard cap (unlike RD-Agent's unbounded library — a marginal improvement), but there is no age-based decay and no re-validation against newer data; a strategy that was top-30 in iteration 500 and has since decayed stays unless literally outscored. |

---

## 11. Verdict — What ARGUS Takes, What NexQuant Confirms ARGUS Must Never Do

### 11.1 Answering the Task Directly

**Does the thing that proposes candidates ever see the scores of previous candidates?** Yes, unambiguously and by design, at every generation mode (§5). This is a stronger, more literal version of the defect found in RD-Agent — there, an LLM's judgment mediated the leak; here, `composite_score` *is* the selection operator. There is no code path in `nexquant_rd_loop.py` where a candidate is generated without the generator having direct read access to the score-sorted `self.sota` list.

**Does NexQuant do anything `argus/research/factor_lab.py` and `argus/research/overfit.py` do not?**

- Nothing that should be adopted as-is. Its core loop is the concrete embodiment of the exact failure mode ARGUS's frozen `ProposerContext` (no performance field) exists to prevent.
- Its one interesting idea — correlation-penalised diversification (§9.1) — is usable only if relocated to a post-selection portfolio step, never as search fitness, and even then it is a modest addition, not a gap-filler; nothing here suggests ARGUS's four-gate overfit discipline (IC stability, sub-sample stress, 200-shuffle placebo, half-life decay) is missing a capability NexQuant actually has working. NexQuant's own placebo/permutation mechanism (§6, §9.3) is broken by its authors' own admission, which if anything raises confidence that ARGUS's genuinely-shuffling placebo gate is already ahead of the strongest comparable code found in this repo.
- Its look-ahead bug (§8) and cost-inconsistency (§7) are cautionary evidence for what happens *without* a design like ARGUS's fee-aware, purged-CV-oriented backtest metrics module — not techniques to adopt, but a live example of the failure ARGUS's `backtest/metrics.py` is positioned to avoid.

**Would adopting NexQuant's search architecture improve or damage ARGUS's overfitting discipline?** It would damage it outright. Its explore/exploit/Optuna/LightGBM loop is a genetic algorithm maximising an uncorrected, look-ahead-contaminated, under-costed Sharpe proxy across tens of thousands of trials with the fitness function doubling as the selection rule. Every property that makes it fast (Numba JIT, 735M bars/sec per the README) is orthogonal to whether what it finds is real; the architecture searches efficiently for noise.

### 11.2 Nothing to Take on the Core Loop

Per the instructions' own permission to conclude this: on the central question this teardown was commissioned to answer, **there is nothing in `nexquant_rd_loop.py`'s search architecture worth taking.** The three items in §9 are adjacent utilities (portfolio-level correlation scoring, a walk-forward reporting format, and a bug pattern worth writing a regression test against) mined from a different, better-engineered, and structurally disconnected part of the same repository — not from the loop the task asked about.

---

## References

- **Repository:** `NicolasBohn/NexQuant` (CI badges reference `github.com/TPTBusiness/NexQuant`)
- **Primary file analysed:** `scripts/nexquant_rd_loop.py` (931 lines)
- **Secondary files analysed:** `rdagent/components/backtesting/vbt_backtest.py` (666 lines), `rdagent/components/backtesting/verify.py` (112 lines), `README.md`, `data_config.yaml`, `LICENSE`
- **Vendored, dead-per-README code excluded from this teardown's core claims:** `rdagent/app/qlib_rd_loop/*`, `rdagent/scenarios/qlib/*` — already covered by the separate `rd-agent.md` teardown of upstream Microsoft RD-Agent

---

**Document Generated:** 2026-09-13
**Codebase Version:** local clone at `research\repos-suggested\nexquant`, ~715 Python files
**Critical Finding Count:** 3 (no search/evaluation separation, no multiple-testing correction, systemic look-ahead bias)
**High-Severity Defects:** 3 (OOS split not enforced, fake Monte Carlo permutation test, inconsistent under-modelled costs)
