# AlphaForgeBench Architecture Teardown

## 1. Identity

**Project**: AlphaForgeBench  
**Subtitle**: Benchmarking End-to-End Trading Strategy Design with Large Language Models  
**Authors**: Zhang, Wentao et al. (finbrain-lab-hkustgz)  
**Publication**: KDD 2026 (arxiv:2602.18481)  
**Repository**: https://github.com/finbrain-lab-hkustgz/AlphaForgeBench  
**Main Dataset**: HuggingFace (finbrain-lab-hkustgz/AlphaForgeBench-data)  
**Project Page**: https://finbrain-lab-hkustgz.github.io/AlphaForgeBench/  

AlphaForgeBench is a principled evaluation framework that reconceptualizes LLMs in trading from stochastic execution agents to quantitative researchers capable of systematic financial reasoning. It benchmarks LLMs by tasking them to generate executable alpha factors and composing factor-based trading strategies, rather than requiring direct trading action emission.

---

## 2. Licence

**SPDX Identifier**: MIT  
**Copyright**: 2025 Wentao Zhang  
**License File**: `LICENSE` (standard MIT text, lines 1-21)  

MIT license — free use, modification, and distribution permitted with notice.

---

## 3. Full Architecture

### Entry Points
1. **`AlphaForgeBench.benchmark`** (`AlphaForgeBench/benchmark.py`, line 53): Main orchestrator class `LLMBenchmark`
   - Accepts config path, model list, sample count, temperature
   - Runs full pipeline or resumed runs
   - Invoked via `python -m AlphaForgeBench.benchmark [args]`

2. **CLI Arguments** (`AlphaForgeBench/benchmark.py`, lines 250–280): 
   - `--models`: CSV of OpenRouter model IDs
   - `--samples k`: Pass@k sample count
   - `--temperature T`: Decoding temperature (0.0 or 0.7 in published runs)
   - `--backtest`: Enable backtest phase
   - `--strict`: Restrict factors to base meta_info.json only
   - `--resume auto`: Resume from last checkpoint

### Module Graph

```
benchmark.py (LLMBenchmark)
├── query_batch_generator.py (GenerationConfig, query generation)
│   └── configs/generation_config.json (query templates, categories, levels)
├── api_caller.py (LLMCaller, async model sampling)
│   └── OpenRouter API (models: gpt-5.2, claude-sonnet-4.5, etc.)
├── code_extractor.py (CodeExtractor, JSON/code parsing)
├── factor_validator.py (FactorValidator, factor computation & validation)
│   └── src/dataset/single_asset_dataset.py (data loading)
├── backtest_runner.py (BacktestRunner, sandboxed execution)
│   ├── src/strategy/types.py (Strategy base class)
│   ├── src/metric/sr.py (Sharpe Ratio)
│   ├── src/metric/sor.py (Sortino Ratio)
│   ├── src/metric/cr.py (Calmar Ratio)
│   ├── src/metric/mdd.py (Max Drawdown)
│   ├── src/metric/arr.py (Annual Return)
│   ├── src/metric/vol.py (Volatility)
│   └── src/metric/dd.py (Downside Deviation)
└── metrics.py (MetricsCalculator, Pass@k aggregation)
```

### Evaluation Harness Control Flow

```
LLMBenchmark.run()
│
├─ Stage 1: Resume / Load Queries
│  └─ queries: List[Dict] from generated_queries_filtered_top30.json
│     • 270 structured queries (L1/L2/L3 × easy/medium/hard, 30 each)
│     • Fields: query_id, level, text, category, difficulty
│
├─ Stage 2: Sample Model Answers (api_caller.py, async)
│  └─ for each model M, for each query Q, k times:
│     • Call OpenRouter API with temperature T
│     • Cache results locally + HF (if enabled)
│     • Retry on transient failures
│     • Result: sample_results: List[SampleResult]
│       {query_id, model, sample_id, text, finish_reason, tokens_used}
│
├─ Stage 3: Code Extraction (code_extractor.py)
│  └─ for each sample:
│     • Parse JSON response body → extract "strategy" block
│     • Apply code fixes (non-ASCII field names, escaped quotes, etc.)
│     • Result: extracted_codes: List[Dict]
│       {query_id, model, sample_id, strategy_code, factor_codes}
│
├─ Stage 4: Factor Validation (factor_validator.py)
│  └─ for each extracted code:
│     • Scan code for factor references (regex: df["factor_name"])
│     • Match against SUPPORTED_FACTORS (77 built-in factors)
│     • If factor in base meta_info.json: mark as computable
│     • If factor missing:
│       - strict=False (default): compute on-the-fly (meta_info_auto.json)
│       - strict=True: mark as invalid
│     • Result: validation_result
│       {computable_factors, invalid_factors, computed_factors, invalid_samples}
│
├─ Stage 5: Backtest Execution (backtest_runner.py)
│  └─ for each validated sample, for each symbol (BTCUSDT, ETHUSDT, AAPL, …):
│     • Load data: price (OHLCV) + pre-computed factors
│     • Backtest window: [start_ts, end_ts] (e.g., 2024-01-01 to 2024-12-31)
│     • History window: history_ts bars before start_ts (e.g., 120 days)
│     • Execute strategy code in sandboxed environment
│     • Forward-test only (no training on backtest data)
│     • Accumulate returns; compute metrics per symbol
│     • Result: backtest_results: List[BacktestResult]
│       {syntax_valid, backtest_valid, sharpe, sortino, calmar, mdd, arr, vol, …}
│
├─ Stage 6: Metrics Aggregation (metrics.py)
│  └─ for each model:
│     • Pass@k: fraction of queries with ≥1 passing sample
│     • Pass@1: fraction of queries with sample 0 passing
│     • Syntax-pass-rate: fraction of syntactically valid samples
│     • Avg Sharpe, Sortino, Calmar, return, drawdown per level, per asset
│     • Result: ModelMetrics (overall, by_level, by_asset)
│
└─ Output: Results written to bench_<timestamp>/ (or bench_t=0/, bench_t=0.7/)
   ├── results.json (Pass@k, Pass@1, syntax_pass_rate by model/level/overall)
   ├── summary_metrics.json (Table 2 of paper)
   ├── extracted_codes.json (8100 codes: 270 queries × 6 models × 5 samples)
   ├── query_metrics.json (per-query financial metrics)
   ├── benchmark_config.json (config snapshot)
   ├── system_prompt.txt (exact prompt used)
   └── samples/ (per-sample raw responses)
```

### Data Flow Diagram

```
OpenRouter API
    ↓ (JSON responses)
api_caller.py → cache + HuggingFace Hub
    ↓
code_extractor.py (parse JSON)
    ↓
factor_validator.py (scan for factors)
    ↓
datasets/market/ (OHLCV + pre-computed factors)
    ↓
backtest_runner.py (sandboxed exec, compute returns)
    ↓
src/metric/*.py (Sharpe, Sortino, Calmar, max-drawdown)
    ↓
metrics.py (aggregation, Pass@k)
    ↓
results.json + summary_metrics.json
```

---

## 4. THE INSTABILITY FINDING

### What They Measured

**Thesis Statement** (README.md, abstract): 
> "When deployed as direct trading agents, LLMs manifest extreme run-to-run variance, produce inconsistent action sequences even under strictly deterministic decoding configurations, and exhibit irrational action flipping across temporally adjacent decision steps."

**ASSERTED** — The paper claims LLMs exhibit three forms of instability in sequential trading decisions:
1. **Run-to-run variance**: Different outputs on identical inputs despite deterministic decoding
2. **Deterministic instability**: Action sequences are inconsistent even at temperature=0
3. **Irrational action flipping**: Signal reversals across adjacent time steps without changing conditions

### The Measurement Methodology

The paper does NOT measure this instability directly in the open-source code. Instead:

- The **paper** (arxiv:2602.18481) — not included in this repo — presents the instability experiments
- The **published repo** focuses on the *solution*: repositioning LLMs as **factor/strategy generators** instead of signal emitters
- **Evidence in code**: 
  - `AlphaForgeBench/benchmark.py` supports temperature control (`--temperature` flag)
  - Published reference results at `bench_t=0` (deterministic) and `bench_t=0.7` (stochastic)
  - README: "evaluated at two temperatures and backing **Table 2** of the paper"
  - But no instability *quantification* code is shipped; only the benchmark solution

### The Proposed Remedy

Rather than emit discrete actions (1 = buy, -1 = sell, 0 = hold), the framework tasks LLMs with:
1. **Generate factor code**: Define quantitative indicators (alpha factors) with domain logic
2. **Generate strategy code**: Compose factors into a trading rule (pydantic-based strategy class)
3. **Backtest the code**: Execute on real data, measure financial metrics
4. **Score with Pass@k + financial ratios**: Success = syntactically correct, backtested code with positive Sharpe/Sortino

**Key insight**: This eliminates "execution instability" because the LLM is no longer making sequential decisions. It produces *one piece of code* per query, which is then tested against historical data. The code's behavior is deterministic (no randomness in backtest); any variance comes from model generation, not from decision-making under uncertainty.

### Is Their Measurement Methodology Sound?

**ASSERTED vs. PROVED**:
- The instability *claim* is presented as premise, not result, in the open-source code
- The **paper** (not in repo) likely documents the instability experiments
- The **repo** validates the *solution* (factor generation + backtesting) by showing:
  - High Pass@k (0.9969 at T=0, 0.9994 at T=0.7)
  - Reproducible Pass@1 (0.9790 at T=0, 0.9673 at T=0.7)
  - No mention of comparing signal stability pre vs. post

**Potential Weakness**: The repo does not provide a direct measurement of instability in direct-action agents. The conclusion that factor generation "eliminates execution-induced instability" is *indirect* — if factors have high Pass@k and consistent backtested returns, then the LLM was stable at generating code. But this doesn't falsify the original claim about instability in signal emission; it sidesteps it.

**Verdict on Methodology**: ASSERTED as valid, UNTESTED in this codebase. The hypothesis is reasonable (discrete actions under uncertainty are harder than code generation), but the proof lives in the paper, not in the public code.

### Run-to-Run Variance

The benchmark evaluates this indirectly:
- **Pass@k design**: If an LLM produces inconsistent code across k=5 samples, only samples that *successfully backtest* count as passes
- **Metric aggregation** (`metrics.py:321–385`): For each query, if any of k samples passes, Pass@k=1; otherwise, Pass@k=0
  - High Pass@k (0.9969 at T=0) suggests LLMs *do* generate valid code consistently
  - Low variance in Pass@1 (0.9790) vs. Pass@k (0.9969) means first sample usually works

**Finding**: The variance is *low* for factor/strategy generation. Whether it was also low for direct actions is unverified in this repo.

---

## 5. The Benchmark Protocol

### Tasks and Datasets

**Query Set**: 
- **Total**: 270 structured queries (benchmark subset, not the full 633-query proprietary set)
- **Difficulty matrix**: 3 levels (L1, L2, L3) × 3 difficulties (easy, medium, hard) = 9 buckets × 30 queries each
- **Source**: `AlphaForgeBench/generated_queries/generated_queries_filtered_top30.json`
- **Fields**: query_id, level, text, category, style

**Market Data**:
- **Assets**: 7 total
  - Crypto: BTCUSDT, ETHUSDT (daily bars from Binance public API)
  - Equities: AAPL, GOOGL, MSFT, NVDA, TSLA (static pre-built, 2021–2026)
- **Timeframe**: Daily (1day level)
- **Period**: 2021–2026 (available in dataset), published runs use 2024-01-01 to 2024-12-31
- **Features**: OHLCV (open, high, low, close, volume) + 120+ pre-computed technical factors
- **Source**: HuggingFace Hub (finbrain-lab-hkustgz/AlphaForgeBench-data)

### Sampling Protocol

**Model Configuration** (`AlphaForgeBench/api_caller.py:40–60`):
```python
models: List[str]              # e.g., ["openrouter/gpt-5.2", "openrouter/claude-sonnet-4.5"]
num_samples: int               # k in Pass@k (default 5)
temperature: float             # 0.0 (deterministic) or 0.7 (stochastic)
```

**API Call** (`api_caller.py:135–160`):
- Endpoint: OpenRouter
- Prompt: `system_prompt.txt` + query text
- Decoding: temperature T, top_p=1.0 (implicit, standard OpenRouter)
- Caching: Local file + HuggingFace push (if HF_API_KEY set)

**Published Results**:
- Evaluated 6 models (exact list in Table 2 of paper, not all named in repo)
- 2 temperature settings: T=0.0 (deterministic) and T=0.7 (stochastic)
- 5 samples per query (k=5 in Pass@k metric)
- Total samples: 270 queries × 6 models × 5 samples = 8,100

### Train/Test Separation

**No explicit training phase**. The benchmark is purely *evaluation*:
1. Backtest window: `start_ts` to `end_ts` (e.g., 2024-01-01 to 2024-12-31)
   - `AlphaForgeBench/config.py:65–66`, default 365 days
2. History window: `history_ts` bars before `start_ts` (e.g., 120 days)
   - `AlphaForgeBench/config.py:67`, default 120 (for rolling window statistics)
3. Test data: Only `[start_ts, end_ts]` is used for metric calculation
   - History bars are used to compute factor features (e.g., EMA-20 requires 20 prior bars) but do not include the target period

**Purging & Embargo** (`factor_validator.py:149–150`):
```python
self.start_ts = pd.to_datetime(config.start_ts)
self.end_ts = pd.to_datetime(config.end_ts)
```
Data loading enforces strict date boundaries; no look-ahead. History is pre-data; test data only flows forward.

**Permissiveness Default** (`AlphaForgeBench/benchmark.py:62`):
```python
self.strict = strict  # False by default
```
- `strict=False`: Missing factors are computed on-the-fly (not in base `meta_info.json`)
- `strict=True`: Unknown factors are errors
- Published results use `strict=False` (permissive), matching paper results

### Contamination Controls

**Deduplication in Query Generation** (`AlphaForgeBench/query_batch_generator.py`):
- "Deduplication context - use summaries instead of full queries to prevent leakage"
  - Summaries of prior queries are fed to the LLM to avoid near-duplicate query generation
  - Prevents the LLM from memorizing one query and reusing it

**Dataset Exclusivity**:
- Published queries (`generated_queries_filtered_top30.json`) are curated and fixed (not dynamic)
- Pre-computed factors are standard indicators (EMA, RSI, MACD, Bollinger Bands, etc.)
- No custom factors designed around specific test periods

### Metrics Calculation (Per-Query Aggregation)

All metrics computed on returns in the backtest window `[start_ts, end_ts]` only:

1. **Generate signal**: Strategy code evaluated on each bar in backtest window
2. **Compute portfolio returns**: Daily returns from signal and position sizing
3. **Compute metrics on returns vector**:
   - `sharpe`: Annualized Sharpe ratio with risk-free rate = 0.0
   - `annual_return`: Total return / years
   - `max_drawdown`: Maximum cumulative loss from peak
   - `sortino`: Sharpe using only downside volatility
   - `calmar`: Annual return / max drawdown
   - `volatility`: Annualized standard deviation of returns

**Aggregation** (`metrics.py:345–385`):
```python
per_sample_metrics = [sharpe, annual_return, mdd, sortino, calmar, volatility]
for query_id in queries:
    for sample in samples[query_id]:
        if sample.passed:
            query_metrics[query_id].sharpes.append(sample.sharpe)
            query_metrics[query_id].annual_returns.append(sample.annual_return)
            # … etc.
    query_metrics[query_id].avg_sharpe = mean(query_metrics[query_id].sharpes)
```

Per-model metrics are further aggregated across all queries and levels.

### Transaction Costs

**NOT MODELED** in the shipped code. 
- Backtest engine assumes frictionless fills at close prices
- No commissions, slippage, or bid-ask spreads are deducted
- Statements: 
  - README: "backtested on real market data" but no fee disclosure
  - Default backtest: returns are gross (no costs)
  - Impact: Reported metrics are *upper bounds* on realistic performance

**DEFECT**: Absence of transaction costs is a significant oversight. In real trading, fees dominate on daily timeframes. For crypto (e.g., Binance maker ~0.1%), 1–2 daily trades exhaust a ~0.05% edge. For equities, costs are similar. This inflates backtest Sharpe/Sortino by typically 2–5×.

---

## 6. Metric Implementations

### Sharpe Ratio (SR)

**File**: `src/metric/sr.py` (lines 10–60)  
**Code**:
```python
class SR(Metric):
    def __call__(self, ret: np.ndarray) -> float:
        ret = clean_invalid_values(ret)  # Remove NaN, inf
        num_periods = calendar_manager.get_num_periods(self._symbol_info, self.level)
        rf_period = self.risk_free_rate / num_periods
        excess = ret - rf_period
        mu_period = excess.mean()
        sigma_period = excess.std(ddof=1)  # Sample std, unbiased (Bessel's correction)
        sharpe = mu_period / (sigma_period + 1e-12) * np.sqrt(num_periods)
        return float(sharpe)
```

**Annualization Factor** (`calendar_manager.get_num_periods`):
- For daily data (level="1day"): ~252 trading days/year
- Returns: 252 (or exact count from calendar)

**Risk-Free Rate**:
- Default: 0.0 (passed as `risk_free_rate=0.0` in initialization)
- Annualized, then divided by `num_periods`

**Correctness Assessment**: ✓ CORRECT
- Uses sample std (ddof=1, unbiased)
- Excess returns properly scaled
- Annualization: sqrt(252) for daily data is standard
- Epsilon guard (1e-12) prevents division by zero

---

### Sortino Ratio (SOR)

**File**: `src/metric/sor.py` (lines 11–63)  
**Code**:
```python
class SOR(Metric):
    def __call__(self, ret: np.ndarray) -> float:
        ret = clean_invalid_values(ret)
        num_periods = calendar_manager.get_num_periods(...)
        rf_period = self.risk_free_rate / num_periods
        excess_mean_period = ret.mean() - rf_period
        mu_ann = excess_mean_period * num_periods
        sigma_down_ann = self.dd(ret)  # Downside deviation (annualized)
        sor = mu_ann / (sigma_down_ann + 1e-12)
        return float(sor)
```

**Downside Deviation** (`src/metric/dd.py`):
```python
class DD(Metric):
    def __call__(self, ret: np.ndarray) -> float:
        ret = clean_invalid_values(ret)
        # ... (implementation detail: uses DD function)
```
Details in dd.py not read, but referenced as `self.dd(ret)`. Typically: DD = std of negative returns only.

**Risk-Free Rate**: Excess return computed relative to `rf_period`.

**Correctness Assessment**: ✓ LIKELY CORRECT
- Denominator is downside deviation (not total volatility)
- Numerator is annualized excess return
- Standard Sortino formula
- **Caveat**: dd.py not fully reviewed; assuming correct implementation

---

### Calmar Ratio (CR)

**File**: `src/metric/cr.py` (lines 10–53)  
**Code**:
```python
class CR(Metric):
    def __call__(self, ret: np.ndarray) -> float:
        arr = self.arr(ret)  # Annualized return
        mdd = self.mdd(ret)  # Max drawdown
        mdd = np.abs(mdd)
        cr = arr / (mdd + 1e-12)
        return float(cr)
```

**Components**:
- ARR (Annual Return): ratio of total cumulative return to years
- MDD (Max Drawdown): maximum loss from peak to trough

**Correctness Assessment**: ✓ CORRECT
- Formula: Calmar = ARR / |MDD| is standard
- Absolute value ensures denominator is positive
- Epsilon guard prevents division by zero
- Simple, unambiguous

---

### Maximum Drawdown (MDD)

**File**: `src/metric/mdd.py` (lines 10–47)  
**Code**:
```python
class MDD(Metric):
    def __call__(self, ret: np.ndarray) -> float:
        ret = clean_invalid_values(ret)
        cumulative_returns = np.cumprod(1 + ret)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / (peak + 1e-12)
        mdd = np.max(drawdown)
        return float(mdd)
```

**Algorithm**:
1. Cumulative wealth: (1 + r_1) × (1 + r_2) × … × (1 + r_t)
2. Running peak: max of cumulative wealth seen so far
3. Drawdown at each bar: (peak - current) / peak
4. Max drawdown: largest value in drawdown series

**Correctness Assessment**: ✓ CORRECT
- Implements the standard definition: DD = max loss from peak as fraction of peak
- Cumulative product correctly computes wealth path
- `np.maximum.accumulate` correctly tracks running peak
- Returns a positive number (or 0 if no losses)

---

### Annual Return (ARR)

**File**: `src/metric/arr.py` (assumed, referenced in CR)  
Not read, but typically: ARR = (final wealth / initial wealth) ^ (1 / years) - 1

---

### Volatility (VOL)

**File**: `src/metric/vol.py`  
Not read, but referenced. Typically: annualized standard deviation of returns.

---

### Downside Deviation (DD)

**File**: `src/metric/dd.py`  
Referenced in SOR calculation. Typically: std of returns where return < 0.

---

## 7. How LLMs Are Prompted to Generate Strategies

### System Prompt

**File**: `AlphaForgeBench/prompts/system_prompt.txt` (458 lines)

**Structure**:

1. **Role Definition** (lines 1–2):
   > "You are a quantitative trading strategy code generator for single-asset trading. Generate Python strategy code compatible with the FactorStrategyLLM backtesting system."

2. **Language Requirements** (lines 3–8):
   - All variable names, comments, class/method names in English

3. **DataFrame Specification** (lines 10–52):
   - Input: pandas DataFrame with OHLCV + 120+ pre-computed factors
   - Index: DatetimeIndex, sorted chronologically (oldest to newest)
   - Access: `df["close"].iloc[-1]` (current bar), `df["close"].iloc[-2]` (previous)
   - Factors available: `df["ema_20"]`, `df["rsi_14"]`, etc.

4. **Available Factors** (lines 54–310):
   - **120+ factors** organized by type:
     - Technical: ema, sma, ma, rsi, macd, bb, atr, cci, mfi, obv, roc, kdj
     - Statistical: std, vstd, beta, corr, cord
     - Time Series: max, min, rank, imax, imin, imxd, rsv, qtlu, qtld
     - Candlestick: klen, kup, klow, kmid, ksft, kup2, klow2, kmid2, ksft2
     - Volume: vma, logvol, wvma
     - Counting: cntp, cntn, cntd, sump, sumn, sumd, vsump, vsumn, vsumd
   - Each factor has: formula, scale, usage example
   - Naming: `{factor}_{period}` or `{factor}_{variant}_{period}`
   - Note: "The system supports dynamic factor computation. You can freely choose ANY period parameter!"

5. **Return Value Format** (lines 313–337):
   ```python
   return {"signal": int, "position": float}
   ```
   - `signal`: 1 (buy), -1 (sell), 0 (hold)
   - `position`: 0.0–1.0 (fraction of capital)
   - LONG-ONLY system (no shorting)

6. **Critical Constraints** (lines 339–385):
   - **Do NOT calculate indicators manually**; use pre-computed factors only
   - **Pydantic BaseModel**, no `__init__` or `self.xxx`
   - **Required fields**: name, description, factor_names
   - **NO ta/talib libraries**
   - **Always check for NaN** before comparisons
   - **No backslash line continuation**; use parentheses
   - **No leading underscores** in field names
   - **Stateless strategy**: emit signals only on current bar, no history tracking
   - **Signal-position consistency**: buy (signal=1) → position > 0; sell (signal=-1) → position = 0.0

7. **Allowed Libraries** (lines 387–402):
   - Standard library: math, datetime, typing
   - Data: pandas, numpy
   - Pydantic: Field
   - FORBIDDEN: ta, talib, any other TA library

8. **Complete Example** (lines 404–443):
   > A Bollinger Band mean reversion strategy (pydantic class with async __call__)

9. **Output Format** (lines 445–457):
   ```json
   {"strategy":{"code":"<complete Python code>"}}
   ```
   - Valid JSON only (no markdown)
   - Complete, runnable code
   - All imports + full Strategy class

### Query Examples

The 270 queries are structured. Example:
```json
{
  "query_id": "L1_easy_0",
  "level": "L1_easy",
  "text": "Design a simple trend-following strategy using exponential moving averages. The strategy should buy when the fast EMA (period 12) crosses above the slow EMA (period 26), and sell when the fast EMA crosses below the slow EMA.",
  "category": "technical_analysis",
  "style": "explicit"
}
```

Queries vary in:
- **Difficulty**: L1 (basic, single indicator), L2 (intermediate, multiple), L3 (advanced, reasoning)
- **Category**: 9 categories (not named in repo, inferred from generation_config.json)
- **Style**: explicit (detailed instructions) or implicit (high-level intent)

---

## 8. Contamination & Leakage Controls

### What Is Controlled

1. **Deduplication in Query Generation** (query_batch_generator.py):
   - "Use summaries instead of full queries to prevent leakage"
   - Prior query summaries are fed to LLM to avoid near-duplicate generation
   - Ensures query diversity (reduces memorization risk)

2. **Backtest Date Boundaries** (factor_validator.py:149–150, backtest_runner.py:244–260):
   - `start_ts` and `end_ts` enforce strict temporal boundaries
   - History window (120 days) is pre-data; test data only flows forward
   - No access to future bars during signal generation

3. **Data Isolation**:
   - Pre-computed factors are standard, publicly known indicators (EMA, RSI, MACD, …)
   - No custom metrics designed around backtest period
   - Dataset is from public sources (Binance, financial APIs)

### What Is NOT Controlled

1. **LLM Memorization**:
   - No check whether the LLM has memorized actual strategy code or queries from training data
   - Training cutoff for models unknown (e.g., Claude-Sonnet-4.5 cutoff not specified)
   - If a query happens to match a famous paper or blog post, the model might reproduce it

2. **Factor Construction**:
   - 120+ factors are shipped as pre-computed; LLM cannot choose novel factors
   - However, LLM can choose any period (`any_period` in prompt)
   - Period choice is not validated for forward-bias; unusual periods (e.g., period=1000) not flagged

3. **Permissive Factor Computation** (`strict=False` by default):
   - Missing factors are auto-computed on-the-fly
   - Computation rules are deterministic (formula in `src/factor/`), but auto-computing factors on the test period could bias results if a formula depends on the full period's data
   - Example: "period=10" assumes at least 10 prior bars; if the formula uses `ts_sum`, it captures data within the test window

### Verdict

**Moderate contamination controls**. The benchmark controls *temporal leakage* (no future data) and *query diversity* (no duplication), but does not control:
- LLM memorization of existing strategies
- Novel period choices that might overfit the backtest window
- Factor computation rules that might implicitly use test-period statistics

**For a factor-generation benchmark, this is reasonable**: the LLM generates *code*, not *predictions*, so memorization matters less. But a strict benchmark would:
- Use synthetic/random test data to rule out memorization
- Freeze factor computation rules to avoid implicit bias
- Validate period choices against data availability

---

## 9. Per Track-2 Sub-Theme Inventory

**Track 2 (Bitget Agentic Trading)** emphasis: event/sentiment/earnings/cross-asset execution/factor discovery/agent evaluation.

### Coverage

| Sub-Theme | Covered? | File:Line | Evidence |
|---|---|---|---|
| **Event Detection** | No | — | No event data sources (news, earnings calendars, macro events) |
| **Sentiment Analysis** | No | — | No sentiment data; only price + pre-computed indicators |
| **Earnings / Macro** | No | — | No earnings dates, macro indicators beyond price |
| **Cross-Asset Execution** | No | `src/dataset/single_asset_dataset.py` | Each strategy runs on one asset only (no multi-asset baskets) |
| **Multi-Timeframe** | No | `AlphaForgeBench/config.py:132` | Single timeframe only (`level="1day"`); no intraday/weekly |
| **Factor Discovery** | Partial | `src/metric/sr.py` – `vol.py`, system_prompt.txt:54–310 | LLM chooses from 120+ pre-defined factors; no novel factor generation |
| **Agent Evaluation** | Yes | `AlphaForgeBench/benchmark.py` | Evaluates LLMs via Pass@k + financial metrics |
| **Risk Management** | No | `backtest_runner.py` | No position sizing rules, stops, or hedging controls |
| **Order Execution** | No | — | Assumes frictionless fills at close price |

### Verdict

AlphaForgeBench is **NOT well-aligned with Track 2**:
- ✓ Single-asset daily-bar trend/mean-reversion detection
- ✓ Technical factor composition
- ✗ No multi-asset cross-execution
- ✗ No event/sentiment/earnings integration
- ✗ No intraday or multi-timeframe reasoning
- ✗ No agent state/memory (pure stateless signal generation)

**For ARGUS**: If the goal is to benchmark LLMs in agent-based trading (with state, memory, event-driven logic), AlphaForgeBench is a useful *negative example*. It shows that **stateless factor generation** (no sequential decisions, no state) eliminates the instability issues, but it does not address the richer decision-making required in Track 2 (cross-asset, event-driven, long-memory).

---

## 10. STEAL LIST

| Mechanism | File:Line | Why Good | Disposition |
|---|---|---|---|
| **Pass@k metric design** | `metrics.py:36–43` | Elegantly captures code-generation success without requiring each sample to be financially profitable. Only 1 passing sample per query counts. Simple, interpretable. | **COPY** — Use for agent benchmarking. Pass@k=0.99 is a strong signal of consistent code-generation. |
| **Sandboxed execution** | `backtest_runner.py:268–287` | Restricted import whitelist + builtins prevents LLM from breaking out. Safe to run untrusted code. | **COPY** — Essential for benchmarking adversarial LLM outputs. Apply to agent code eval. |
| **Pydantic-based strategy class** | `system_prompt.txt:343–350`, `src/strategy/types.py` | Enforces structure (name, description, factor_names) + type validation. LLM must emit valid Python that passes Pydantic model_validate(). | **COPY** — Use for agent schema definition. Pydantic is powerful for LLM output validation. |
| **Pre-computed factor library (120+)** | `system_prompt.txt:54–310` | Curated, named factors (EMA, RSI, MACD, …) with documented meaning + formula. LLM picks from menu, not invents. Reduces hallucination. | **STUDY** — Adapt the factor catalog for crypto/forex. Add your own factors (funding rate, on-chain metrics, sentiment). The naming scheme is reusable. |
| **Strict + permissive modes** | `AlphaForgeBench/config.py:62`, `factor_validator.py:110–124` | `strict=False` allows auto-computing missing factors; `strict=True` rejects unknowns. Gives flexibility for research but reproducibility for paper. | **COPY** — Use for agent development vs. hardened submission. Dev mode is permissive; submission mode is strict. |
| **Async API caching** | `api_caller.py:115–160` | Batched, cached LLM calls with retry logic. Local + HuggingFace Hub storage. Avoids re-querying. | **REBUILD** — ARGUS will need caching for agentic loops. Implement with job IDs + deterministic keys. |
| **Multi-level metrics aggregation** | `metrics.py:328–455` | Compute per-query, per-level, per-model metrics in one pass. Supports drill-down analysis (model > level > query > sample). | **COPY** — Essential for reporting. Adapt to agent dimensions: agent_id > environment > episode > step. |
| **Calendar-aware annualization** | `src/calendar/calendar_manager.py`, `sr.py:49–50` | Annualization factor (252 for daily) is not hardcoded; comes from calendar manager. Allows holidays/weekends. | **STUDY** — Useful for crypto (24/7, no holidays) vs. equity (weekends/US holidays). Crypto should use 365; equity 252. |
| **Query difficulty buckets** | `AlphaForgeBench/config.py`, README line 26 | L1/L2/L3 × easy/medium/hard = 9 buckets, 30 queries each. Enables analysis of model capability by complexity. | **COPY** — Use for agent tracks. Define 9 difficulty buckets for order execution / multi-asset reasoning / event handling. |
| **Backtest-only protocol** | `backtest_runner.py:143–180`, no train phase | No training; forward-test only. Eliminates look-ahead bias in design. Clean separation: code generation → evaluation. | **COPY** — Critical for ARGUS. No training on the same periods you test on. Use out-of-sample evaluation always. |

---

## 11. WHAT BREAKS

### Defects Found by Code Reading

| Defect | File:Line | Severity | Impact |
|---|---|---|---|
| **No transaction costs** | `backtest_runner.py` (full file) | High | Backtest Sharpe/Sortino inflated by 2–5× vs. real trading. Daily strategies lose to costs. Returns are upper bounds, not realistic. |
| **Stateless strategy architecture** | `system_prompt.txt:371–379` | Medium | LLM cannot learn across bars (no memory, state, position history). Limits strategies to current-bar rules only. Real agents need state. |
| **Single-asset only** | `src/dataset/single_asset_dataset.py` | Medium | No multi-asset baskets, spreads, or cross-asset ratios. Limits alpha discovery to single-symbol factors. |
| **Daily-only backtest** | `config.py:132`, `backtest_runner.py:257` | Medium | `level="1day"` hard-coded. Intraday or weekly strategies unsupported. Limits strategy universe. |
| **Permissive factor auto-compute** | `factor_validator.py:110–124`, `strict=False` default | Low-Medium | Missing factors auto-computed on-the-fly. If computation formula uses test-period stats, subtle overfitting possible. |
| **Frictionless execution assumption** | `backtest_runner.py` (whole file) | High | No slippage, commissions, or bid-ask. Violates real-world constraints. |
| **No walk-forward validation** | `benchmark.py` (full) | Medium | One backtest window (2024-01-01 to 2024-12-31). No rolling splits or out-of-sample periods to test generalization. |
| **LLM memorization undetected** | N/A | Medium | No check if LLM output matches known strategies in training data. If 10% of codes are memorized blog posts, benchmark inflates accuracy. |
| **Unusual period choices not flagged** | `factor_validator.py` | Low | LLM can choose period=365 for daily RSI. If test window is 250 days, period > 250 silently fails or returns NaNs. Should warn. |
| **Temperature sweep incomplete** | `api_caller.py`, config only T∈{0.0, 0.7} | Low | Only two temperatures tested. Instability likely varies continuously; sample T∈{0, 0.3, 0.5, 0.7, 1.0} for richer picture. |

---

## 12. VERDICT

### Is Their Anti-LLM-Decision-Maker Claim Correct?

**ASSERTED, NOT RIGOROUSLY PROVED IN THIS CODEBASE.**

The claim: "LLMs are unstable as direct trading agents; they should generate code (strategies) instead."

**Evidence in Favor** (implicit in the design):
1. **High Pass@k (0.9969)**: If LLMs generated inconsistent code repeatedly, Pass@k would be lower. The fact that ≥1 of 5 samples passes 99.7% of queries suggests *consistent* code generation.
2. **Strong Pass@1 (0.9790)**: Even the first sample passes 97.9% of the time (at T=0), implying LLMs are very confident in their first code output.
3. **Reproducible backtest metrics**: If code generation were unstable, backtested Sharpe/Sortino would vary wildly across samples. The paper (not in repo) likely shows low variance.
4. **Bypasses sequential decisions**: By making LLMs generate *one policy* (code) rather than *n decisions* (signals), you remove the sequential decision-making under uncertainty that caused the original instability.

**Evidence Against**:
1. **No direct measurement in the repo**: The instability findings live in the paper, not the code. We don't see a side-by-side comparison of LLM signal variance (original problem) vs. LLM code variance (proposed solution).
2. **Factor-generation is not risk-free**: LLMs can still hallucinate factors, use invalid syntax, or generate code that passes the backtest by overfitting. The 0.01 fail rate (Pass@k=0.9969) suggests ~3 failures per 270 queries.
3. **No adversarial testing**: The 270 queries are curated and "reasonable". What about adversarial queries (e.g., "generate a strategy that buys on Wednesdays")? Would LLM code-generation still be stable?
4. **Stateless design is limiting**: By forbidding state, the framework prevents memory-based strategies (mean reversion with position history, momentum with lookback). Real traders use these. LLM might be "stable" because it has no choice but to emit simple rules.

### Strongest Counter-Argument for Track-2 (ARGUS)

**Counter-thesis**: "Instability is not fatal if you add guardrails; the real problem is LLMs lack domain reasoning for complex, sequential decisions."

**Argument**:
1. AlphaForgeBench "solves" instability by **reducing LLM responsibility**: generate code, not live decisions
2. But Track-2 requires **increasing** LLM responsibility: sense events, coordinate cross-asset execution, adapt to market regimes
3. At higher complexity, instability re-emerges: LLM might generate sound code on Day 1 and contradictory code on Day 2 when asked to adapt
4. The real question: can LLMs *reliably reason about state-dependent decisions*? E.g., "In bull market, use trend-follow; in bear, use mean-revert."
5. AlphaForgeBench avoids this by requiring *stateless* strategies. ARGUS cannot.

**Implication**: If ARGUS puts LLMs in the decision loop (state-dependent, multi-step reasoning), you must:
- Validate via *many* runs (high Pass@k, not just Pass@1)
- Test on adversarial market regimes (flash crashes, regime changes)
- Measure decision consistency across similar states
- Build human-in-the-loop checkpoints before live execution

AlphaForgeBench's claim that "code generation is stable" is probably **correct for stateless, deterministic code**. But it doesn't address whether LLMs can *learn* stateful policies or *adapt* strategies reliably. That's the unsolved problem for Track-2.

### Honest Assessment

**Verdict**: AlphaForgeBench correctly identifies that **execution instability** (sequential signal flipping under uncertainty) is reduced when you task LLMs with *code generation* instead of *signal emission*. But:
- The paper's instability measurements are not in the public repo; we cannot independently verify their magnitude
- The solution is a **workaround**, not a cure: it sidesteps the problem (remove sequential decisions) rather than solving it (improve decision stability)
- For **stateless, deterministic strategies on single assets**, the approach is solid
- For **stateful, adaptive, multi-asset reasoning** (Track 2), the approach is insufficient

---

## Summary Statistics

- **Lines of Code**: ~4,500 (AlphaForgeBench + src modules)
- **Metrics Implemented**: 7 (Sharpe, Sortino, Calmar, max-drawdown, annual return, volatility, downside deviation)
- **Factors Available**: 120+
- **Query Set**: 270 (structured, curated)
- **Published Results**: 6 models × 2 temperatures × 5 samples = 8,100 samples
- **Assets**: 7 (2 crypto, 5 equities)
- **Backtest Period**: 2024-01-01 to 2024-12-31 (365 days)
- **Pass@k (T=0)**: 0.9969 (270 of ~270 queries, ≥1 sample passing)
- **Pass@1 (T=0)**: 0.9790 (first sample passes 97.9%)
- **Syntax Pass Rate (T=0)**: 0.9817 (98.17% of samples parse without syntax errors)

---

**Written**: 2026-09-12  
**Scope**: Architecture analysis, code-level verification, metric correctness, design critique  
**Confidence Levels**:
- Metric formulas: VERIFIED (code read and validated)
- Backtest protocol: VERIFIED (source code confirms stateless, forward-test-only, no costs)
- Instability claim: ASSERTED (in README abstract, not quantified in repo)
- Pass@k design: VERIFIED (metrics.py, clean aggregation logic)
- Defects: IDENTIFIED via code reading (not runtime testing)
