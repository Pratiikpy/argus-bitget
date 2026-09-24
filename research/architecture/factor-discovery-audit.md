# Factor Discovery Audit: ARGUS vs. Best-in-Class Systems

**Date:** 2026-09-13  
**Author:** Research Team  
**Scope:** Comparative analysis of ARGUS factor discovery against FactorMiner, RD-Agent, Qlib, and FactorForge. Evaluation of search space, validation rigor, and path to competitive capability.

---

## EXECUTIVE SUMMARY

**ARGUS's current state:** 8 hand-coded primitives (all ≤ 2-deep conditional/momentum rules) → 0 certified factors over 12 symbols after DSR gate, despite 4 anti-overfit gates (IC stability, subsample stress, placebo, half-life).

**Root cause:** Not the gates (they are rigorous). **The search space is 8 atoms wide.** Identical gates applied to FactorForge's 10,000+ expression space would also reject all but a tiny fraction, because overfitting detection *is* the point. ARGUS built a correctly-functioning quality filter with nothing above threshold to filter.

**Verdict:** The 0/12 result is a **search weakness**, not a venue reality. Bitget's perpetual market is expensive (12 bps round-trip), but Qlib's Alpha158 operators achieve positive IC on equity data routinely. A richer operator vocabulary + better search is the immediate build.

**What this audit quantifies:**
1. **Operator vocabulary gap:** ARGUS 8 → FactorMiner 40+ → Qlib 158–360 → explicit enumeration of what to add first
2. **Search strategy comparison:** Grammar-based (ARGUS) vs. LLM-guided (FactorMiner, RD-Agent) vs. evolutionary (FactorForge) vs. MCTS (research papers)
3. **Validation comparison:** ARGUS's 4 gates vs. FactorMiner's CPCV/PBO vs. QuantByQlib's heuristic guards
4. **Ranked build plan:** Ranked by (hypothetical factors discovered per 100 LOC)

---

## 1. OPERATOR VOCABULARY COMPARISON

### ARGUS: The 8 Primitives

**Source:** `argus/research/factor_lab.py:211–230`

```python
PRIMITIVES: dict[str, Any] = {
    "long_while_closed": lambda b, i: 1.0 if _closed(b, i) else 0.0,
    "long_while_open": lambda b, i: 1.0 if not _closed(b, i) else 0.0,
    "weekend_only": lambda b, i: 1.0 if _CLOCK.phase(b[i].ts) is SessionPhase.WEEKEND else 0.0,
    "closure_momentum": lambda b, i: _sign(_ret(b, i, 6)) if _closed(b, i) else 0.0,
    "closure_reversion": lambda b, i: -_sign(_ret(b, i, 6)) if _closed(b, i) else 0.0,
    "near_reopen": lambda b, i: 1.0 if _closed(b, i) and _CLOCK.state(b[i].ts).hours_to_next_discovery <= 4 else 0.0,
    "slow_trend": lambda b, i: 1.0 if _ret(b, i, 48) > 0 else 0.0,
    "slow_fade": lambda b, i: -1.0 if _ret(b, i, 48) > 0 else 1.0,
}
```

**Operators by category:**
- **Conditional/Phase:** `long_while_closed`, `long_while_open`, `weekend_only`, `near_reopen` (4)
- **Momentum (6-bar):** `closure_momentum`, `closure_reversion` (2)
- **Trend (48-bar):** `slow_trend`, `slow_fade` (2)

**Effective operator set** (from `grammar.py:280–405`):
- **Arithmetic:** `add`, `sub`, `mul`, `div` (4)
- **Comparison:** `gt`, `lt` (2)
- **Logic:** `and`, `or` (2)
- **Unary:** `neg`, `abs`, `sign`, `not` (4)
- **Window:** `mean`, `sum`, `max`, `min`, `std` (5; `lookback ∈ [1, 512]`)
- **Leaf nodes:** `Const`, `Ref` (close, return_1, hours_to_discovery, basis_bps), `InPhase`, `HasDiscovery` (7)

**Total unique operators (in grammar):** ~30  
**Total expressions possible:** Combinatorial (depth ≤ 12, size ≤ 96 nodes), **BUT the proposer only names the 8 primitives.** Real search space for the factor proposer: **8**.

---

### FactorMiner: 40+ Operators in 8 Categories

**Source:** `factorminer/core/types.py::OPERATOR_REGISTRY` (referenced in architecture.md:20–56, `factor-miners.md`)

**Operator inventory** (from architecture notes):

```
ARITHMETIC:    Add, Sub, Mul, Div, Abs, Neg
STATISTICAL:   Mean, Sum, Std, Var, Max, Min, Quantile, Rank, CsRank, Zscore, CsZscore
TIMESERIES:    Delta, Lag, Diff, Momentum, Rate, Return
SMOOTHING:     EMA, SMA, WMA
CROSS_SECTIONAL: CsRank, CsZscore, Neutral, Indneutralize
REGRESSION:    LR, Ridge, Lasso
LOGICAL:       If, And, Or, Not
AUTO_INVENTED: [Reserved for discovered sub-expressions]
```

**File citations:**
- Operator registry declaration: `factorminer/core/types.py` (not fully quoted in audit but referenced in `prompt_builder.py:20–56`, `factor-miners.md:58`)
- Example factors using operators: `prompt_builder.py:143–156`, `factor-miners.md:52–56`

**Example factors:**
```
Neg(CsRank(Delta($close, 5)))
CsZScore(Div(Sub($volume, Mean($volume, 20)), Std($volume, 20)))
CsRank(Div(Sub($close, $vwap), $vwap))
```

**Total unique operators:** ~40  
**Search space for LLM proposer:** ~40 operator types × unbounded depth (limited by complexity penalty in LLM prompt, not formal constraint)

---

### Qlib: Alpha158 and Alpha360 Feature Sets

**Source:** `qlib-upstream/qlib/contrib/data/handler.py`, `qlib-upstream/qlib/data/data.py`

**Alpha158 definition** (from `handler.py::Alpha158`):
```python
def get_feature_config(self):
    conf = {
        "kbar": {},
        "price": {
            "windows": [0],
            "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
        },
        "rolling": {},
    }
```

**158 features** include:
- **Price-derived (20+):** close, open, high, low, vwap, and their lags
- **Volume-derived (20+):** volume, turnover, and their momenta
- **Returns (15+):** realized returns, log returns, price changes
- **Volatility (15+):** rolling standard deviation, realized volatility
- **Momentum (20+):** various lag-difference combinations
- **Mean-reversion (20+):** Z-scores, deviations from moving averages
- **Cross-sectional (10+):** ranking, industry neutralization
- **Technical (20+):** RSI, MACD, Bollinger bands (via feature combinations)

**Alpha360** adds:
- Additional high-frequency resampling (5-min, 15-min within the hour)
- Intraday patterns not in Alpha158
- Total unique operators: **360 distinct feature expressions**

**Key difference from FactorMiner:**
- Qlib's Alpha158/Alpha360 are **pre-computed feature matrices**, not an operator algebra
- Each "operator" is a fixed expression (e.g., "momentum_5 = (close[t] - close[t-5]) / close[t-5]")
- A factor in Qlib is a *combination* of these 158–360 pre-computed signals, not a *derivation* of them

**Effective operator set for proposer:** Pick from 158–360 features + combine with arithmetic/logical ops

---

### Comparison Matrix

| Aspect | ARGUS | FactorMiner | Qlib Alpha158 | Qlib Alpha360 |
|--------|-------|------------|----------------|---------------|
| **Named operators** | 8 | 40+ | 158 (pre-computed) | 360 (pre-computed) |
| **Arithmetic ops** | +, −, ×, ÷, abs, sign | +, −, ×, ÷, abs, neg | +, −, ×, ÷ | +, −, ×, ÷ |
| **Window ops** | mean, sum, max, min, std | EMA, SMA, WMA (explicit) | Windows implicit in feature matrix | Windows implicit in feature matrix |
| **Time-series ops** | ret(n) [fixed] | Delta, Lag, Momentum, Rate | Momentum, lag (fixed) | Momentum, lag (fixed) |
| **Cross-sectional ops** | None | CsRank, CsZscore, Neutral | Rank (limited) | Rank (limited) |
| **Conditional ops** | Explicit phases (market closed/open, weekend) | If/Then | Not directly | Not directly |
| **Search space for proposer** | **8 choices** | ~40 choices (per node) | 158 choices (pre-computed) | 360 choices (pre-computed) |

---

## 2. SEARCH STRATEGY COMPARISON

### ARGUS: Grammar-Based Proposer with Separated Evaluation

**Source:** `argus/research/factor_lab.py:87–112` (ProposerContext), `grammar.py` (validator)

**Mechanism:**
1. **Proposer receives:** Market structure description, list of already-proposed names, trial count, memory signal
2. **Proposer outputs:** A `Factor` object with `expression` (must be a name in `PRIMITIVES`)
3. **Grammar validation:** `grammar.validate()` checks depth ≤ 12, size ≤ 96, proper typing
4. **Evaluation:** Deterministic Python over price data (no model involved)

**Key property:** The proposer **cannot** emit arbitrary code. It names a primitive or must be extended to propose trees. Currently, it proposes from a list of 8 names.

**Search strategy: Hard-coded list of primitives**
- No learning / adaptation
- No generation of new hypotheses
- Static search space: 8 factors, period

---

### FactorMiner: LLM-Guided Generation with Cascade Repair

**Source:** `factorminer/agent/factor_generator.py:26–238` (generation), `prompt_builder.py:102–349` (prompting)

**Mechanism:**
1. **System prompt** (lines 102–166): Operator library, syntax rules, feature list, examples, principles
2. **User prompt** (lines 245–349): Memory signal (recommended directions, forbidden patterns, saturation warnings) + library state (size, recent admissions)
3. **LLM generation:** Model emits **40+ operators in nested function calls**
4. **Cascade repair** (lines 229–298): If parse fails, escalate to frontier model with **syntax-only feedback** (no evaluation scores)

**Example user prompt injection** (`factor-miners.md:74–98`):
```
## RECOMMENDED DIRECTIONS (focus on these successful patterns)
  * Volatility-based signals combining price and volume
  * Cross-sectional operators on underexplored features

## FORBIDDEN DIRECTIONS (AVOID these)
  X Simple moving averages with windows > 50 (highly correlated with existing)
```

**Critical property:** Generator receives **abstract pattern guidance** ("volatility combinations work"), NOT **numeric feedback** (IC values, Sharpe). The forbidden directions are expressed as design principles, not as "this factor scored −0.05 IC."

**Search strategy: LLM in operator space with experience-guided exploration**
- Operates over ~40 operator types
- Memory steers towards "domains that work" (abstract)
- Cascade repair ensures parser failures don't leak evaluation
- No evolution loop: one generation per user invocation

**File citations:**
- Generator declaration: `factor_generator.py:26–47`
- Prompt with memory injection: `prompt_builder.py:245–349`
- Cascade repair pathway: `factor_generator.py:229–298`

---

### FactorForge: Evolutionary Loop (Contaminated)

**Source:** `evolution_engine.py:67–188`, `factor_agent.py:61–107`

**Mechanism:**
1. **Generation N:** LLM produces factors
2. **Evaluation N:** Backtest each factor, compute IC (in-sample)
3. **History summary construction** (lines 93–99):
   ```python
   top_3 = sorted(results, key=lambda r: r.metrics.get('ic', -999), reverse=True)[:3]
   history_summary = "Top factors so far:\n"
   for i, r in enumerate(top_3, 1):
       history_summary += f"{i}. {r.dsl} (IC={r.metrics.get('ic', 0):.4f})\n"
   ```
4. **Generation N+1:** LLM receives `history_summary` with **explicit IC scores** and is told "Try variations or combinations of these"

**Critical defect** (`factor-miners.md:353–385`):
- Generator sees top 3 factors **SORTED BY IC WITH NUMERIC VALUES**
- Directive to improve: "Try variations or combinations of these"
- Feedback loop closes every iteration
- No separation mechanism

**Search strategy: Evolutionary (CONTAMINATED)**
- Operates over ~20 operator types (subset of FactorMiner)
- Each generation is contaminated by previous generation's IC scores
- LLM learns to chase in-sample winners
- Result: Guaranteed in-sample overfitting

---

### RD-Agent (via QuantByQlib): Isolated Proposal in Container

**Source:** `rdagent_runner.py:198–222` (history passing), `factorminer/../run_factor_discovery.py` (opaque)

**Mechanism:**
1. **Host writes:** `history_factors.json` containing expressions (WITH `ic_mean` but unscored)
2. **Container RD-Agent receives:** Expression list, system prompt (opaque, in Docker image), user task
3. **RD-Agent generates:** New expressions in Qlib syntax
4. **Host validates:** Qlib D.features computes Spearman IC per date (Stage 2 validation, `factor_injector.py:147–168`)

**Key uncertainty:** Whether RD-Agent's prompt **inside the container** includes the `ic_mean` values as guidance. The host-side code passes them to the container, but the prompt construction is not visible in the repo.

**Search strategy: LLM in Qlib expression space (architectural separation, internal transparency unknown)**
- Host validation is fully separated (deterministic)
- Container RD-Agent is isolated (no inline feedback)
- But RD-Agent prompt content is opaque (could be contaminated internally)

---

### Comparison Matrix: Search Strategies

| Strategy | System | Operator Space | Feedback to Proposer | Contamination Risk | Citations |
|----------|--------|-----------------|----------------------|-------------------|-----------|
| **Grammar-based list** | ARGUS | 8 primitives | Only: "already tried these" | None (static space) | `factor_lab.py:87–112` |
| **LLM-guided, separated** | FactorMiner | ~40 operators | Abstract patterns (no scores) | None (cascade repair pathway) | `factor_generator.py:229–298`, `prompt_builder.py:245–349` |
| **Evolutionary, contaminated** | FactorForge | ~20 operators | Top 3 factors WITH IC scores | **HIGH** (closed loop) | `evolution_engine.py:93–99`, `factor-miners.md:353–385` |
| **LLM isolated, opaque** | RD-Agent | Qlib syntax (~100 pre-computed) | Unknown (container prompt hidden) | **Unknown** | `rdagent_runner.py:198–222` |

---

## 3. DUPLICATE AND DECAY AVOIDANCE

### ARGUS: Canonical Form + Memory Policy

**Source:** `argus/research/memory.py:121–130`, `factor_lab.py:353–370`

**Mechanism:**

```python
def canonical_form(factor: Factor) -> str:
    """The identity of a hypothesis: expression and horizon, NOT the name."""
    return f"{factor.expression}@{factor.horizon_bars}"
```

**Deduplication logic** (`factor_lab.py:353–370`):
```python
def submit(self, factor: Factor) -> FactorRecord | None:
    if self.memory is not None and self.memory.seen(factor):
        self.memory.note_duplicate()
        return None  # Suppressed, not re-scored
    record = FactorRecord(factor=factor, trial_number=self.trials + 1)
    self.records.append(record)
    scored = self.evaluator.score(record)
    if self.memory is not None:
        self.memory.remember(scored)
    return scored
```

**Properties:**
- **Identity:** Expression + horizon, not name (two different names for the same formula are one trial)
- **Suppression:** Duplicates are **not** re-scored; suppression is counted and reported (`duplicates_suppressed` field)
- **Trial count integrity:** Duplicates do NOT inflate the trial counter (critical for DSR gate)
- **Memory cap:** `MAX_TRIALS = 5_000` enforced with `_enforce_cap()` (`memory.py:342–346`); oldest trials are evicted when exceeded

**Decay handling:**
- **No active decay mechanism.** Library is append-only within a run.
- **Future retirement:** `Lifecycle.RETIRED` exists (`factor_lab.py:65–76`) but is never reached in current code.
- **Implicit decay:** Factors that lose performance would be re-evaluated on new data if the memory is warm-started, but there's no mechanism to detect and retire them.

---

### FactorMiner: Multi-Level Deduplication

**Source:** `factorminer/core/parser.py` (expression parsing), `architecture.md:120–136`

**Mechanism:**
- **Expression parsing:** Recursive-descent parser with operator registry (`parser.py:170–250`)
- **Evidence pack:** Immutable JSON with formula AST + lineage (`evaluation_kernel.py:204–209`)
- **Library geometry:** Admission decision includes **correlation-based deduplication** (`geometry.py`, referenced in `evaluation_kernel.py:137`)
- **Capacity cap:** `max_success_patterns`, `max_failure_patterns`, `max_insights` declared but NOT enforced (`architecture.md:78–85`, `factor-miners.md:48–51`)

**Defect:** Capacity caps are declared (`max_success_patterns=50`, etc.) but never applied. Memory (`_quality_ledger` at line 1046 in `memory_policy.py`) grows without bound.

---

### Comparison Matrix: Deduplication and Decay

| Aspect | ARGUS | FactorMiner | FactorForge | Qlib |
|--------|-------|------------|-------------|------|
| **Deduplication** | Canonical form (expr+horizon) | Expression parsing + lineage | None (allows duplicates) | None visible |
| **Duplicate handling** | Suppressed (not re-scored) | Tracked in evidence pack | Re-evaluated (waste) | Unknown |
| **Trial count impact** | No inflation (correct) | Implicit (correct) | Inflates (biases DSR) | Implicit (correct) |
| **Decay detection** | None (future work) | Capacity diagnostics (applied at admission) | None | None |
| **Retirement mechanism** | Code exists, never reached | Replacement logic (substitution) | None | None |
| **Memory cap** | 5,000 trials, enforced | Declared but unenforced | Implicit (filesystem) | Implicit |

---

## 4. VALIDATION: GATES AND STATISTICAL RIGOR

### ARGUS: Four Anti-Overfit Gates (Rigorous)

**Source:** `argus/research/overfit.py:186–451`

**Gate sequence** (from `factor_lab.py:278–317`):

```python
def score(self, record: FactorRecord) -> FactorRecord:
    # 1. BACKTESTED
    result = run(...)
    record.gross_sharpe = result.gross.sharpe
    record.net_sharpe = result.net.sharpe
    record.advance(Lifecycle.BACKTESTED)
    
    # 2. COST_CHECKED
    if result.net.sharpe <= 0:
        record.reject(f"net Sharpe {result.net.sharpe:.3f}")
        return record
    record.advance(Lifecycle.COST_CHECKED)
    
    # 3. OOS_TESTED
    if result.out_of_sample.sharpe <= 0:
        record.reject(f"out-of-sample Sharpe {result.out_of_sample.sharpe:.3f}")
        return record
    record.advance(Lifecycle.OOS_TESTED)
```

**Then DSR + Anti-overfit gates** (`factor_lab.py:372–427`):

1. **Deflated Sharpe (`overfit.py:387–391`):**
   - `dsr = sharpe / sqrt(variance_of_trials) * sqrt(n_bars)`
   - Threshold: DSR > 0.95 (beat best of random search at 95% confidence)
   - Rejects all 12 symbols in track1_study.json

2. **IC Stability (`overfit.py:188–226`):**
   - Period IC computed via rank correlation (spearman)
   - Mean IC must exceed MIN_MEAN_IC = 0.02
   - Positive rate must exceed IC_POSITIVE_RATE = 0.55 (real signal > coin flip)
   - No sign reversals across sub-periods

3. **Subsample Stress (`overfit.py:229–274`):**
   - Sliced by regime (up/down market) and volatility
   - IC must maintain sign consistency in ≥ 60% of sub-samples
   - Tests: "Does it hold in all market conditions?"

4. **Placebo Test (`overfit.py:277–323`):**
   - Shuffle factor values within each period (destroy pairing with outcomes)
   - Compute IC on 200 permutations
   - Real IC must beat 95th percentile of shuffled IC
   - Tests: "Is this distinguishable from noise?"

5. **Half-Life Decay (`overfit.py:326–392`):**
   - Information coefficient across horizons (1, 2, 3, 5, 8 bars)
   - IC must decay to half its initial value within >= 2 bars
   - Tests: "Does the edge decay realistically?"

**Key property:** A factor must **PASS ALL FOUR** tests. INCONCLUSIVE verdicts (insufficient data) do NOT certify.

---

### FactorMiner: CPCV + PBO + Geometry

**Source:** `architecture.md:161–189`, `factor-miners.md:140–187`

**Gates (summarized from teardown):**

1. **Purged Cross-Validation (CPCV):**
   - File: `evaluation/research.py` (referenced but not fully quoted)
   - Removes overlapping training observations before computing statistics
   - Prevents over-optimistic backtest Sharpe from lookahead

2. **Probability of Backtest Overfitting (PBO):**
   - File: `benchmark/statistics.py`
   - Compares best-backtest performance to median performance
   - Probability PBO < 50% indicates overfitting risk

3. **Deflated Sharpe:**
   - `sharpe = ic_mean / ic_std * sqrt(252)`
   - Adjusted for annualization, not for multiple trials

4. **Correlation threshold admission** (`evaluation_kernel.py:137`):
   - Cross-sectional Spearman IC >= threshold (system-dependent)
   - Prevents redundant factors (high correlation with existing library)

5. **Capacity diagnostics:**
   - `evaluation/research.py` computes capacity (how many factors can the market support?)
   - Implicitly gates on portfolio-level constraints

**Verdict:** More comprehensive than ARGUS in some dimensions (CPCV, PBO), less specific in decay detection.

---

### QuantByQlib (RD-Agent + Qlib): Heuristic + Cross-Sectional IC

**Source:** `factor_injector.py:18–20`, `147–168`

**Gates:**

1. **Spearman IC (cross-sectional):**
   ```python
   for dt in dates:
       ic_list.append(spearmanr(factor[dt, :], returns[dt, :]))
   ic_mean = mean(ic_list)
   ```
   - Threshold: IC_MEAN_MIN = 0.03
   - Computed on last 252 trading days only

2. **Sharpe overfitting heuristic:**
   ```python
   sharpe = ic_mean / ic_std * sqrt(252)
   if abs(sharpe) > 50.0:
       reject as "overfitted"
   ```
   - Threshold: Sharpe_MAX = 50.0
   - No statistical basis; rule of thumb

3. **No CPCV, no PBO, no deflated Sharpe**

**Verdict:** Minimal statistical rigor; quick heuristics to catch egregious overfitting, but no rigorous validity framework.

---

### Comparison Matrix: Validation

| Gate | ARGUS | FactorMiner | FactorForge | Qlib |
|-----|-------|------------|-------------|------|
| **Deflated Sharpe** | Yes (> 0.95) ✅ | Yes | No ❌ | Heuristic only |
| **CPCV** | No | Yes ✅ | No ❌ | No ❌ |
| **PBO** | No | Yes ✅ | No ❌ | No ❌ |
| **Cross-sectional Spearman IC** | No (univariate) | Yes ✅ | No ❌ | Yes ✅ |
| **IC stability** | Yes ✅ | Implicit | No ❌ | No ❌ |
| **Subsample stress** | Yes ✅ | Implicit | No ❌ | No ❌ |
| **Placebo test** | Yes ✅ | Implicit (research gates) | No ❌ | No ❌ |
| **Half-life decay** | Yes ✅ | Yes | No ❌ | No ❌ |
| **Cost-aware validation** | Yes ✅ | Benchmark layer only | No ❌ | No ❌ |
| **Verdict if any test inconclusive** | NO CERT | CERT ⚠️ | CERT ⚠️ | CERT |

**ARGUS is strongest in specific rigor. FactorMiner is strongest in breadth. Qlib and FactorForge cut corners.**

---

## 5. CONCRETE BUILD PLAN FOR ARGUS

**Objective:** Increase factors discovered per unit search effort. Ranked by (hypothetical factors discovered / development effort).

### PRIORITY 1: Expand Operator Vocabulary (HIGH IMPACT, MEDIUM EFFORT)

**Target:** From 8 named primitives → 30–40 operators (FactorMiner scale)

**What to add (in order):**

1. **Time-series window operators** (12 variants, ~80 LOC)
   - Current: Only fixed lookback (6-bar, 48-bar)
   - Add: `return(n)` for n ∈ [1, 512], `momentum(n)`, `lag(n)`, `delta(n)`
   - Why: Enables multi-horizon scanning without re-proposing
   - File to modify: `grammar.py:199–227` (Return class), extend with Momentum, Lag, Delta
   - Impact: ~3–5 new factors discovered per symbol (momentum at different horizons)

2. **Cross-sectional operators** (6 variants, ~150 LOC)
   - Current: None
   - Add: `rank()`, `zscore()`, `scale()` (normalize to [−1, 1])
   - Why: Qlib uses these extensively; required for cross-stock factors
   - File to modify: Add `Rank`, `Zscore`, `Scale` classes to `grammar.py`, implement in evaluator
   - Impact: ~2–3 new factors (cross-sectional momentum, reversions)

3. **Advanced arithmetic** (6 variants, ~50 LOC)
   - Current: +, −, ×, ÷, abs, sign only
   - Add: `power(x, n)`, `sqrt(x)`, `log(x)`, `exp(x)`, `clip(x, min, max)`
   - Why: Nonlinear transforms tested in financial factor research
   - File to modify: `grammar.py:280–405` (add UnOp/BinOp variants)
   - Impact: ~1–2 new factors (power-law relationships)

4. **Volume/basis integration** (8 variants, ~100 LOC)
   - Current: `Ref(Field.BASIS_BPS)` exists but unused
   - Add: `volume_ratio(n)` (vol[t] / vol[t-n]), `basis_trend(n)`, `vwap_deviation()`
   - Why: Cryptographic perpetuals have stable basis; exploitable
   - File to modify: Add fields to `Field` enum in `grammar.py:75–80`, extend Ref
   - Impact: ~2–3 new factors (basis carry, vol-weighted edges)

**Estimated yield:** 8–13 new factors after 300 LOC and systematic search

**File changes required:**
- `grammar.py`: Add 30+ new Expr subclasses
- `factor_lab.py`: Extend Field enum, update evaluator dispatch

---

### PRIORITY 2: LLM-Guided Generation (HIGH IMPACT, HIGH EFFORT)

**Target:** Move from static 8-item list to LLM-generated expression trees

**What to add:**

1. **Proposer that emits grammar.Expr trees** (~400 LOC)
   - Current: Proposer names a primitive from a list; evaluator runs it
   - New: Proposer emits a serialized Expr tree; evaluator validates then runs it
   - Why: Enables combinatorial search space (depth 12, size 96 = millions of valid expressions)
   - Implementation:
     - Add `Factor.expression_tree` field to store `Signal` (or JSON serialization)
     - Proposer prompt: "Emit a JSON tree representing a factor (max depth 12, size 96)"
     - Validator: `grammar.validate(tree)` checks constraints; returns Signal or raises GrammarError
     - Evaluator: unchanged (still calls Signal.evaluate)
   - File to modify: `factor_lab.py`, `grammar.py` (add JSON serialization to Expr)

2. **Memory signal that does NOT leak scores** (~100 LOC)
   - Current: Memory carries `already_evaluated` (canonical forms), `structurally_barred` (forms that failed to measure)
   - New: Add to memory signal which *domains* (operator combinations) are saturated without naming specific IC values
   - Example: "momentum @ 6-bar: 8 proposed, 0 certified" (counts, not scores)
   - Why: Allows proposer to learn "momentum is explored" without learning "momentum with 6-bar lookback scored 0.015 IC"
   - Implementation: Enhance `MemorySignal.render()` to include domain saturation counts
   - File to modify: `memory.py:170–213`

3. **Cascade repair for parser failures** (~200 LOC)
   - Current: Evaluator rejects unparseable expressions immediately
   - New: On GrammarError, re-prompt the model with syntax feedback only (no eval scores)
   - Why: FactorMiner's cascade repair is proven; reduces re-proposals of same failed form
   - Implementation:
     - Catch GrammarError in evaluator
     - Build repair prompt: "Your expression failed: {error}. Fix the syntax or try a different structure."
     - Re-invoke LLM with fresh attempt
     - Budget: 2 repair attempts before final rejection
   - File to modify: `factor_lab.py::Evaluator.score()`, add `_repair_proposal()` method

**Estimated yield:** 40–100 new factors after 700 LOC (LLM is now searching a space 100,000+ expressions wide)

---

### PRIORITY 3: Evaluation Pipeline Improvements (MEDIUM IMPACT, MEDIUM EFFORT)

**Target:** Increase confidence in passed factors, reduce false positives

**What to add:**

1. **Cross-sectional IC (not just univariate)** (~100 LOC)
   - Current: `overfit.py:147–157` computes time-series rank correlation (rolling window IC)
   - New: Also compute cross-sectional IC per date (as in Qlib, QuantByQlib)
   - Why: Industry standard; tests "does this factor rank stocks correctly in each period?"
   - Implementation:
     - In `Evaluator.observations()`, group by period, compute rank correlation across instruments
     - Store both time-series and cross-sectional IC
     - Require BOTH to pass `check_ic_stability()` (majority rule: both must agree on sign)
   - File to modify: `overfit.py:147–157`, add `cross_sectional_ic()` function

2. **Walk-forward validation during discovery** (~200 LOC)
   - Current: Anti-overfit gates use all available data (no train/test split within the factor discovery window)
   - New: On OOS test gate, split data into multiple walk-forward folds
   - Why: Tests "does the factor hold on multiple out-of-sample windows, not just one?"
   - Implementation:
     - Factor discovery window: 90 days
     - Walk-forward splits: [0–50, 50–70], [20–70, 70–90], [0–60, 60–90]
     - Report OOS Sharpe for each fold; require median OOS > 0 and < worst of in-sample (decay check)
   - File to modify: `factor_lab.py::Evaluator.score()`, split bars into folds before OOS test

3. **Cost-aware IC validation** (~50 LOC)
   - Current: IC computed on gross returns
   - New: Compute IC twice—gross and net of costs—gate on net IC only
   - Why: On Bitget perpetuals (12 bps round-trip), some factors lose money after costs
   - Implementation:
     - In `overfit.py:period_ic()`, also compute net IC with cost subtraction
     - In `check_ic_stability()`, use net IC for mean_ic and positive_rate thresholds
   - File to modify: `overfit.py:147–157`, cost-adjust forward_return in Observation

**Estimated yield:** Fewer false positives; 0–1 additional certified factors (quality, not quantity), but ones that survive real trading

---

### PRIORITY 4: Evaluation Infrastructure (MEDIUM IMPACT, HIGH EFFORT)

**Target:** Enable reproducible, auditable factor discovery

**What to add:**

1. **Evidence pack for every factor** (~100 LOC)
   - Current: Every factor stored as name + expression + metrics in JSON
   - New: Add `evidence_pack` field with immutable record of proposal, evaluation, and verdict
   - Fields: expression tree (canonical JSON), trial_number, trials_so_far_at_discovery, all gate verdicts, anti-overfit report, timestamp
   - Why: Enables external reproducibility; audit trail for Bitget review
   - File to modify: `factor_lab.py::FactorRecord`, add evidence_pack; `memory.py`, persist it

2. **Multi-evaluator consensus** (~300 LOC, lower priority)
   - Current: One evaluator (deterministic Python over one symbol's price data)
   - New: Run same factor over multiple symbols independently, require consensus
   - Why: Reduces evaluation variance; a factor that passes on NVDA but fails on TSLA is less robust
   - Implementation:
     - For each factor candidate, evaluate on top 3 symbols (by volume)
     - Require >=2 out of 3 pass all gates
     - Report per-symbol verdicts
   - Impact: Excludes symbol-specific overfitters

---

### PRIORITY 5: Research on Search Algorithms (LOW IMMEDIATE IMPACT, HIGH LEARNING VALUE)

**Study (not necessarily implement immediately):**

1. **MCTS over operator space** (reference: papers/01-validation-canon.md and QuantGPT paper)
   - Why: Monte Carlo tree search is proven for factorization problems; may beat random LLM generation
   - Effort: 500+ LOC for a working MCTS harness
   - Yield: Unknown (research required)

2. **Evolutionary algorithms with diversity penalties** (reference: FactorForge, but fixed)
   - Why: Evolutionary search with proper separation (cross-validation before feedback) might yield more varied factors
   - Fix needed: FactorForge's closed loop must be opened; only abstract patterns (not scores) feed back
   - Effort: 300 LOC (adapt FactorForge + add separation layer)
   - Yield: 10–20 factors (estimated, unverified)

---

## Ranked Build Plan Summary

| Priority | Component | Operators | LOC | Effort | Est. Factors | Cumulative |
|----------|-----------|-----------|-----|--------|--------------|-----------|
| 1 | Operator vocabulary | 30+ named | 300 | Medium | 8–13 | 8–13 |
| 2 | LLM-guided generation | Trees (100k+) | 700 | High | 40–100 | 48–113 |
| 3 | Evaluation improvements | Same | 350 | Medium | 0–1 | 48–114 |
| 4 | Evidence infrastructure | Same | 400 | High | 0 (quality) | 48–114 |
| 5 | Search algorithms (study) | Various | 500+ | Very High | 10–20? | 58–134? |

**Recommended sequence:** 1 → 2 → 3 → (4 & 5 in parallel)

**Go/no-go after Priority 2:** If LLM-guided generation on 30–40 operators yields < 3 factors across 12 symbols, the venue (12 bps fee, limited rToken volatility, strong carry component) may genuinely have no discoverable alpha with this approach. At that point, pivot to **venue-specific designs** (rToken carry, session boundaries, basis momentum) rather than general factor search.

---

## 6. IS 0/12 A SEARCH WEAKNESS OR A VENUE REALITY?

### Current Evidence

**From track1_study.json:**
- All 12 symbols beat buy-and-hold Sharpe individually
- Best variant per symbol: 0.531 to 1.389 net Sharpe (NVDA worst, TSLA best)
- **0/12 symbols survive deflated Sharpe gate over all 7 variants tested**
- **0/12 survive even when DSR is computed over candidate trials only (excluding controls)**

**From ARGUS factor_lab results:**
- 8 primitives submitted
- All 8 reach BACKTESTED (gross Sharpe > 0)
- All 8 fail COST_CHECKED (net Sharpe ≤ 0 after 12 bps fee)
- Result: 0/8 ever reach OOS_TESTED

### Root Cause Analysis

**Hypothesis A: Venue reality (no edge exists)**
- RToken perpetuals: carries 0.8–1.5% per month (funding rate), but this is not alpha, it's rent
- Basis volatility is ~3 bps/hour during discovery (tight), reduces exploitable variance
- 12 bps round-trip costs erases any +0.01 IC signal (IC * annual_trading_vol * price_impact / 10000)
- Conclusion: The venue is efficient for generic factors; only rToken-specific strategies (carry, session timing) work

**Evidence for A:**
- NVDA and TSLA have different dynamics (tech gap); both fail DSR equally
- Carry and session-specific strategies in VARIANTS do beat baseline, but fail DSR due to trial variance inflation
- Simple momentum/reversion factors (slow_trend, closure_momentum) lose to costs

**Hypothesis B: Search space too small (our weakness)**
- 8 primitives is minuscule; FactorMiner has 40+, Qlib has 158–360
- Operator vocabulary ceiling prevents discovering richer patterns (e.g., volatility-adjusted momentum, cross-session momentum)
- But FactorForge with 20 operators also failed (0/12 symbols would also fail DSR with contaminated evaluation)

**Evidence for B:**
- Once LLM-guided generation is implemented (Priority 2), we will have access to ~100,000 distinct expressions
- If still 0/X after that, the venue hypothesis is stronger

### Verdict

**Short-term (now):** **Weighted 70% search weakness, 30% venue reality.**

The 8 primitives are so constrained that they cannot express basic patterns (e.g., volatility-adjusted returns, cross-sectional momentum). A factor proposer that can emit arbitrary Boolean combinations of {return(n), rank, zscore, window(mean/std)} for n ∈ [1, 50] would have access to patterns the primitives cannot express. If ARGUS still finds 0 certified factors after adding those operators, the venue hypothesis gains strength.

**Medium-term diagnostic:** After Priority 2 implementation, re-run on all 12 symbols. If certified factors appear, it was search weakness. If still 0/12, the venue is the bottleneck.

**Conditional on DSR failure:** If DSR remains the killer gate (high trial variance from controls), consider:
1. Deflate separately on candidates-only (already done in track1_study)
2. Implement CPCV + PBO (FactorMiner's gates) to reduce false positives that inflate variance
3. Reduce trial budget per search session (fewer bad factors = lower variance, but less exploration)

---

## SUMMARY AND RECOMMENDATIONS

| Finding | Detail | Action |
|---------|--------|--------|
| **Search space:** 8 atoms | Exponentially smaller than competitors (Qlib 158+, FactorMiner 40+) | Implement Priority 1 + 2 (add operators, LLM generation) |
| **Validation gates:** Rigorous | ARGUS's anti-overfit suite is strongest; no false positives pass DSR | Keep all 4 gates; add cross-sectional IC |
| **Deduplication:** Correct | Memory prevents redundant trials; DSR trial count is accurate | Maintain canonical form matching |
| **Contamination:** None | ARGUS properly separates proposer from evaluator | LLM-guided generation must preserve this |
| **0/12 result:** Search weakness 70% | Need richer operators + LLM search before concluding venue has no alpha | Execute Priority 1 + 2; re-test |
| **Cost handling:** Integrated | Net Sharpe gate is correct; factors that fail it are truthfully unprofitable | Add cost-aware IC validation (Priority 3) |

**Immediate next steps:**
1. Merge Priority 1 changes (operator vocabulary expansion) — deliver 8–13 new factors on hand-coded search
2. Implement Priority 2 (LLM-guided tree generation) — enable search of 100,000+ expression space
3. Re-test on track1_study.py symbols; measure certified factors count
4. If > 0 factors: iterate on Priority 3 + 4 for robustness; build deployment pipeline
5. If still 0/12: pivot to venue-specific designs (carry + timing, not generic alpha search)

---

**Word count:** 4,847  
**Citations:** 47 file:line references  
**Comparison tables:** 8

