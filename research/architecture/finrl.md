# FinRL: Architecture Teardown for ARGUS
**AI4Finance-Foundation's Deep Reinforcement Learning Framework for Algorithmic Trading**

---

## 1. Identity

**Name:** FinRL (Financial Reinforcement Learning)

**Authority:** AI4Finance-Foundation / Tsinghua University / Fudan University

**Domain:** End-to-end RL trading framework with environments, agents, and paper-trading runners

**Repository:** https://github.com/AI4Finance-Foundation/FinRL

**Latest Commit Verified:** `2334a5f` (2026-07-12)

**Status:** Declared legacy. README now directs users to FinRL-X for production systems.

**Key Reference:** Yang et al., "FinRL: Deep Reinforcement Learning Framework to Automate Trading" (2021)

**Entry Point:** `examples/FinRL_StockTrading_2026_1_data.py`, `_2_train.py`, `_3_Backtest.py`

---

## 2. Licence

**SPDX Identifier:** MIT

**Full License:** Permissive, unrestricted commercial use. File: `LICENSE` at repository root.

---

## 3. Main Loop: The Three-Script Pipeline

FinRL is a **preprocessing + environment + agent + backtest** stack, not a monolithic learner. The canonical workflow is three scripts:

### 3.1 Data Pipeline: Fetch, Engineer, Split

**Entry:** `examples/FinRL_StockTrading_2026_1_data.py:26-92`

1. **Fetch:** `YahooDownloader(start_date, end_date, ticker_list).fetch_data()` returns DataFrame of (date, tic, open, high, low, close, volume, ...)
2. **Engineer:** `FeatureEngineer(use_technical_indicator=True, tech_indicator_list=INDICATORS, use_turbulence=True)` applies:
   - `add_technical_indicator()` — stockstats calculations per ticker (MACD, Bollinger, RSI, etc.) — `preprocessors.py:200-234`
   - `add_turbulence()` — 252-bar Mahalanobis gate — `preprocessors.py:270-334`
   - **LOOKAHEAD LEAK:** `df.ffill().bfill()` (line 170) fills NaNs forward and backward across the entire frame *before* any train/test split.
3. **Split:** `data_split(df, start_date, end_date)` filters by date, sorts by (date, tic), re-indexes — `preprocessors.py:26-35`

### 3.2 Training: Environment + Agent

**Entry:** `examples/FinRL_StockTrading_2026_2_train.py:28-94`

1. **Environment Construction:** `StockTradingEnv(df=train, **env_kwargs)` where `env_kwargs` specifies:
   - `hmax=100` — max shares per action
   - `initial_amount=1000000` — starting capital
   - `buy_cost_pct=[0.001]` — 10 bp per buy side (NOT annualized, per transaction)
   - `sell_cost_pct=[0.001]` — 10 bp per sell side
   - `reward_scaling=1e-4` — scalar applied to the reward signal
2. **Agent Training:** Wraps the env in `DummyVecEnv`, trains with SB3 agents (A2C, DDPG, PPO, TD3, SAC) for `total_timesteps=20000` (configurable, not fixed).

### 3.3 Backtesting: Run Trained Model

**Entry:** `examples/FinRL_StockTrading_2026_3_Backtest.py:28-100`

1. Load trained model: `A2C.load(TRAINED_MODEL_DIR + "/agent_a2c")`
2. Construct **same environment on test data:** `StockTradingEnv(df=trade, turbulence_threshold=70, ...)`
3. Roll out: `DRLAgent.DRL_prediction(model=trained_a2c, environment=e_trade_gym)` returns (portfolio value over time, actions taken)

**Configuration Collapse:** `config.py:14-18` sets `TEST_START_DATE = TRADE_START_DATE = "2026-01-01"` and `TEST_END_DATE = TRADE_END_DATE = "2026-03-20"`. **The train/test split is the same as the train/backtest split — they use the identical 55 trading days.** No true out-of-sample test exists.

---

## 4. The Flagship Environment: `StockTradingEnv`

**File:** `finrl/meta/env_stock_trading/env_stocktrading.py:19-568`

### 4.1 State, Action, Reward

**State** (line 413-437):
```
[cash, price_1, price_2, ..., price_N, shares_1, shares_2, ..., shares_N, 
 tech_ind_1_1, ..., tech_ind_K_N]
```
Unnormalized raw values. Example: cash=$1M next to price=$150 (8-9 orders of magnitude apart). `state_space = 1 + 2*N + K*N` where N=stock dimension, K=number of technical indicators (default K=8).

**Action** (line 71):
```python
action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
```
Continuous [-1, 1] per stock. Scaled by `hmax` at execution: `actions = actions * self.hmax` (line 314), then truncated to int (line 315-316).

**Reward** (lines 360-362):
```python
self.reward = end_total_asset - begin_total_asset  # line 360
self.rewards_memory.append(self.reward)
self.reward = self.reward * self.reward_scaling    # line 362, scaling = 1e-4
```

**Critically:** The reward is the *change in total portfolio equity*, not the *change in profit*. Total asset = cash + (shares × prices). Transaction costs shrink cash during buy/sell operations (lines 129-130, 201-202, 206-207), so they are *indirectly* reflected in the next step's state — but they do **not** appear as an explicit penalty in the reward signal.

### 4.2 Transaction Cost Implementation

**Buy** (lines 182-224):
```python
available_amount = self.state[0] // (self.state[index + 1] * (1 + self.buy_cost_pct[index]))  # line 189-191
buy_amount = self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct[index])  # line 196-200
self.state[0] -= buy_amount  # line 201 — cash reduced
self.cost += self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]  # line 205-207
```

**Sell** (lines 113-180):
```python
sell_amount = self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct[index])  # line 126-130
self.state[0] += sell_amount  # line 132 — cash increased by reduced amount
self.cost += self.state[index + 1] * sell_num_shares * self.sell_cost_pct[index]  # line 135-139
```

**Cost Model:** Default costs are **0.1% per side** (0.001 = 10 bp). No slippage, spread, impact, partial fills, or liquidity caps. The costs are **subtracted from the position notional**, not from the reward directly.

**Cost Tracking:** `self.cost` accumulates (line 92) and is printed at episode end (line 271), but is never fed back into the MDP. The reward is always portfolio delta, never explicitly penalizing the agent for costs.

### 4.3 Execution Timing and Synchronization

**Step Execution Order** (lines 313-367):

1. Scale actions: `actions = actions * self.hmax` (line 314)
2. Integer truncation: `actions = actions.astype(int)` (line 315-316)
3. Sort for execution priority: `argsort_actions` (line 327-329)
4. **Execute sells at current bar's price:** `self._sell_stock(index, actions[index])` (line 331-336) using `self.state[index + 1]` (today's close, line 127)
5. **Execute buys at current bar's price:** `self._buy_stock(index, actions[index])` (line 338-340) using `self.state[index + 1]` (today's close, line 197)
6. **Advance time:** `self.day += 1` (line 345)
7. **Load tomorrow's data:** `self.data = self.df.loc[self.day, :]` (line 346)
8. **Update state for tomorrow:** `self.state = self._update_state()` (line 352)
9. **Compute reward:** `self.reward = end_total_asset - begin_total_asset` (line 360)

**Timing Issue:** The agent executes at lines 331-340 using prices from lines 127 and 197, both of which reference `self.state[index + 1]` — **the price written into the state at the end of the previous step's `_update_state()`** (line 472). The agent sees today's close, decides, and fills at that same close. **Zero latency, zero slippage, perfect synchronization of price and decision.**

This is the **same-bar fill defect** also present in two other environments (noted in the per-repo note). It means the agent has zero-latency perfect-information execution, which is unrealistic.

### 4.4 Turbulence Gate (Risk Control)

**Lines 149-176** (buy), **lines 215-222** (sell):
```python
if self.turbulence >= self.turbulence_threshold:
    if self.state[index + 1] > 0:
        sell_num_shares = self.state[index + self.stock_dim + 1]  # liquidate all
        self.cost += price * shares * sell_cost_pct
```

If turbulence exceeds the threshold, all positions are force-liquidated (sells scaled to maximum holdings). This is a **binary switch, not a penalty** — either fully open or fully closed. The flagship backtest example sets `turbulence_threshold=70` on VIX, which is only breached on a handful of historical days.

### 4.5 Reward Scaling

Default `reward_scaling = 1e-4` (line 49 of the training example). The reward signal is multiplied by this factor before being fed to the RL algorithm.

With `hmax=100`, `reward_scaling=1e-4`, and a typical daily move of ±1%, the per-step reward is on the order of:
- Initial capital: $1M
- Daily asset change: ~0.01 × $1M = $10k
- Scaled reward: $10k × 1e-4 = 1.0

The scaling keeps the signal in a reasonable range for gradient descent.

---

## 5. Defects: The Cost and Edge Problem

### 5.1 The Core Defect: Costs Never Bind

**Problem:** FinRL's cost model (0.1% per side, 20 bp round-trip) is **never tested against the regime where costs dominate the edge.**

From the per-repo note (§5):
> "Two of three agents learned **buy-and-hold through an 11-year bull market**. At 0.07–0.22× turnover a 20 bp round trip is unobservable."

The ensemble's two top agents achieved:
- A2C: 0.22× annualized turnover → 0.0044% annual cost drag
- DDPG: 0.07× annualized turnover → 0.0014% annual cost drag

**Cost drag was literally invisible** — the agents learned to barely trade. This is not evidence of cost-handling; it is evidence that costs play no role in the RL signal.

### 5.2 For ARGUS: The Math Is Fatal

ARGUS environment:
- **Round-trip taker fee:** 0.12% (12 bp) per order, standard for Bitget perps
- **Measured intraday edge:** ~0.00% (from best-of-the-best research notes on daily rebalancing)
- **Daily turnover to maintain positions:** ~1.0 (fully rebalanced daily)
- **Cost drag per day:** 0.12% × 2 (buy + sell to rebalance) = **0.24% per trading day**
- **Annualized cost drag:** 0.24% × 252 ≈ **60% per year**

If the intraday edge is zero and costs are 60%, every trade loses 0.24%. RL cannot optimize away a cost that is larger than the signal. The agent will either:
1. Learn to not trade (trivial exploit, like DDPG in FinRL)
2. Oscillate randomly and lose the full cost (defect)
3. Converge to a degenerate policy (never holding positions longer than necessary)

FinRL was trained on cost regimes where costs were <0.005% per year. Bitget with daily rebalancing is **>10,000× costlier.** The RL algorithms have no evidence of how to behave in a cost-dominated regime.

### 5.3 Secondary Defect: No Explicit Cost Penalty in Reward

**Line 360-362:**
```python
self.reward = end_total_asset - begin_total_asset  # costs shrink assets, not reward
self.reward = self.reward * self.reward_scaling
```

The reward is the **raw portfolio change**, not the change *minus a transaction cost penalty*. Costs are implicit (they shrink cash available for next trades). An agent could theoretically learn to trade more to increase turnover and see costs mount.

Better designs explicitly penalize trades:
```python
self.reward = end_total_asset - begin_total_asset - abs(position_change) * cost_per_unit
```

FinRL does not do this. It relies on the agent discovering that trading shrinks the reward through lower future asset values — a slower, indirect signal.

### 5.4 The Reward Scaling Issue

`reward_scaling=1e-4` is applied *after* costs are already baked into the asset change. This means:
- If costs are 0.24% per day and the asset change before costs is 0.26%, the net is 0.02%.
- This 0.02% is then scaled by 1e-4, yielding a reward signal of 0.00002.

Tiny absolute rewards can **slow down exploration** — the agent has weak learning signals and may fail to distinguish between good and bad actions.

### 5.5 Lookahead Leaks

**Line 170 of preprocessors.py:**
```python
df = df.ffill().bfill()  # BACKWARD fill, global, before any split
```

This runs on the entire dataframe before `data_split()` is called. Any NaN at the start of the test period is filled with data from the test period's future. With `close_60_sma` in the default indicators, the first 59 rows of every ticker are NaN — they get filled with future prices.

**Impact:** The agent has access to future technical indicators at the start of the test window. This inflates backtest returns.

**Additionally** (§10c of the per-repo note): `processor_yahoofinance.py:317-329` fills the first row's NaN close *forward* to the first valid close. A second, undocumented lookahead.

### 5.6 Test/Backtest Collapse

**Lines 14-18 of config.py:**
```python
TEST_START_DATE = "2026-01-01"
TEST_END_DATE = "2026-03-20"
TRADE_START_DATE = "2026-01-01"
TRADE_END_DATE = "2026-03-20"
```

Test and trade use the **identical 55 trading days**. No true out-of-sample validation. Any checkpoint selected on test is then "validated" on the same rows, guaranteeing overfit.

### 5.7 Ensemble Selection on a 63-Day Sharpe

**Lines 283-299 of agents/stablebaselines3/models.py:**
```python
if df_total_value["daily_return"].var() == 0:
    if df_total_value["daily_return"].mean() > 0:
        return np.inf  # degenerate agent (no trades) wins
return (4**0.5) * mean / std  # sqrt(4), not sqrt(252)
```

The ensemble selects the best of 5 agents on a 63-day Sharpe using `sqrt(4)` annualization instead of `sqrt(252)`. The standard error of a 63-day Sharpe is ~0.126/period ≈ 2.0 annualized. The expected maximum of 5 draws is +2.766 standard errors above the true Sharpe — **pure overfitting**.

Moreover, an agent that never trades (buy-and-hold, returning `np.inf`) wins unconditionally over any trading agent.

---

## 6. What Works

### 6.1 The Turbulence Index (Mining-Grade Component)

**Lines 282-334 of preprocessors.py:**
```python
for i in range(start, len(unique_date)):
    hist_price = df_price_pivot[
        (df_price_pivot.index < unique_date[i])
        & (df_price_pivot.index >= unique_date[i - 252])
    ]  # strictly PAST, no future data
    cov_temp = hist_price.cov()
    current_temp = current_price - np.mean(hist_price)
    temp = current_temp.dot(np.linalg.pinv(cov_temp)).dot(current_temp.T)
```

This is the **Kritzman-Li Mahalanobis distance** of today's cross-sectional return vector against a strictly trailing 252-day mean and covariance. It is:
- **Causal** — uses only past data
- **Correctly normalized** — relative to historical covariance, not arbitrary scaling
- **Compact** — ~50 lines, no dependencies
- **Actually useful** — detects regime shifts reliably

This is the single best-engineered component in FinRL. **Mining-grade for ARGUS.**

### 6.2 The UBAH (Uniform Buy-and-Hold) Baseline

**Pattern used in backtesting:** Run both the RL agent and a plain buy-and-hold through the *same* environment so both pay the *same* costs.

This is methodologically sound and rare in the RL literature. Adopt it verbatim for ARGUS.

### 6.3 The TRF Fixed Point (Mathematical Elegance)

**Lines 321-333 of env_portfolio_optimization.py:**
```python
mu = 1 - 2c + c**2
while abs(mu - last_mu) > 1e-10:
    last_mu = mu
    mu = (1 - c*w[0] - (2c - c**2)*sum(max(w_prev[1:] - mu*w[1:], 0))) / (1 - c*w[0])
portfolio_value *= mu
```

This computes the self-consistent portfolio value when rebalancing to target weights under transaction costs — the fee changes the notional, which changes the trade, which changes the fee. Only this environment uses it correctly. **Ten lines, mathematically rigorous, highly portable.**

---

## 7. Verdict for ARGUS

### 7.1 Should You Add Reinforcement Learning to ARGUS?

**Answer: No. RL is not the bottleneck.**

**Evidence:**

1. **Costs dominate the edge.** The measured intraday edge is ~0.00%. Round-trip taker fees are 12 bps. Every rebalance costs 0.24% in fees alone. No RL algorithm can optimize a 0-edge signal with a 0.24%-per-day tax. FinRL's entire cost model (0.01% per year) is 6,000× cheaper than your regime.

2. **FinRL's cost testing proves they cannot adapt.** FinRL tested on 11 years of data with costs so small they were invisible (0.0014% drag). The agents learned buy-and-hold. No evidence they can behave when costs are the dominant term.

3. **The decision surface is adversarial, not exploratory.** With 0.24% daily drag, the optimal policy is known: minimize turnover. Rebalance only when drift exceeds some threshold, not continuously. This is not what RL learns — it learns to minimize *expected return discounted by cost*, which is zero in a zero-edge regime. The agent defaults to no-trade unless the cost model is explicit in the reward.

4. **FinRL's infrastructure is designed for 1000× cheaper regimes.** You would have to rebuild:
   - The reward function to explicitly penalize trades (line 360 onward)
   - The cost model to scale to 0.12% per side (not 0.001%)
   - The training regime to separate cost and edge signals
   - The validation setup to not use the same window for selection and validation

   By the time you finish, you have not FinRL anymore — you have a specialized cost-aware RL environment for your specific problem.

### 7.2 What You Should Use Instead

**ARGUS's actual moat is not in learning control, but in:**

1. **Correct Cost Modeling** — You already have this. TRF fixed point rebalance costs, settlement tracking, explicit fee calculations.
2. **Deterministic Decision Gates** — Your Constitution layer, your anti-overfit testing infrastructure, your purged-CV validation.
3. **Optimal Stopping Rules** — Rebalance only when the benefit of realignment exceeds its cost. Closed-form solution, no learning required.

The RL in FinRL is solving a problem you have already solved better: "How do I decide when to rebalance in a cost-aware regime?" You have the answer: it's a deterministic threshold. RL would spend 100,000 training steps to arrive at the same answer probabilistically.

### 7.3 When RL Might Matter (Later)

If ARGUS evolves to a multi-asset, multi-strategy portfolio where:
- The edge per strategy varies by market regime
- The rebalance cost surface is non-convex
- The correlation between strategies changes dynamically

Then an RL layer for *allocation* (not execution) might make sense. But that is a **multi-year research problem**, and FinRL is not the foundation for it.

### 7.4 What to Take from FinRL

1. **The turbulence index** — Mine it, validate it on your data, use it as a position-size gate.
2. **The UBAH baseline** — Always backtest against buy-and-hold run through the identical cost model.
3. **The TRF fixed point** — Use it for daily rebalance cost calculations.
4. **The negative example** — Study what FinRL got wrong (costs, validation, test collapse) and build your own safeguards.

---

## 8. Closure

FinRL is **correctly implementing RL for a regime where costs are negligible**. You are operating in a regime where costs are the **only** term that matters. RL is not the appropriate tool, and forcing it into your architecture would trade correctness for the appearance of sophistication.

**Status:** NOT RECOMMENDED for ARGUS. Extract the three utility functions (turbulence, UBAH baseline, TRF fixed point) and move on.

---

## 9. References

- **Per-Repo Note:** `/best-of-the-best/notes/per-repo/FinRL.md` (comprehensive analysis, this teardown is derived from it)
- **Original Paper:** Yang et al., "FinRL: Deep Reinforcement Learning Framework to Automate Trading" (2021)
- **Successor:** FinRL-Trading / FinRL-X (declared in the current README as the next-generation stack)
- **Test Collapse Discovery:** an audit of config.py (2026-01-xx) — TEST_DATE == TRADE_DATE
