# INALPHA · ARCHITECTURE TEARDOWN FOR ARGUS TRACK 2

**Quant agents that evolve under audit** — an oracle that keeps a ledger. Machine-approved orders, factor timing, strategy evolution sandbox, risk engine, and an audit ledger that makes every decision traceable and reproducible.

---

## 1. IDENTITY

**Project:** Inalpha  
**Repository:** github.com/mirror29/inalpha  
**Author/Maintainer:** mirror29 (GitHub user)  
**Status:** Alpha (79 factors, E1 strategy evolution, D-12 production closure)  
**Use Case:** LLM-driven quantitative trading research, multi-market backtesting, paper trading with factor timing and strategy evolution in a sandbox.  
**Key Differentiator:** _The LLM is NOT on the order path._ Every trade intent must pass through a three-step approval flow (`create_plan → approve → execute`), with a one-shot, time-bound approval token. No backdoor, no prompt override.

---

## 2. LICENCE

**SPDX:** `AGPL-3.0`  
**Text:** Full GNU Affero General Public License v3.  
**File:** `LICENSE` (35 KB, full legal text).  

**Implication:** Any modifications must be released under AGPL-3.0. Commercial or SaaS deployments must include source availability. This is a copyleft license; cannot be freely used in proprietary closed-source systems.

---

## 3. FULL ARCHITECTURE

### Macro Layout (3 Tiers + 1 Data Tier)

```
┌────────────────────────────────────────────────────────┐
│  L1 · Entry Layer                                       │
│  ─────────────────────────────────────────────────────  │
│  apps/dashboard (Next.js 16, port 3001)                │
│  ├─ Authentication, per-user LLM config (encrypted)    │
│  ├─ Agent chat docked on right                         │
│  ├─ Strategy/backtest/evolution/risk/factor dashboards │
│  └─ BFF (Route Handlers → back-end APIs over HTTP)     │
│                                                         │
│  mastra dev (port 4111)                                │
│  ├─ Playground trace UI (live tool calls, hooks)       │
│  └─ Direct orchestrator agent conversation             │
│                                                         │
│  apps/web (static inalpha.dev)                         │
└────────────────────┬─────────────────────────────────┘
                     │ same-origin /api/*
                     ▼
┌────────────────────────────────────────────────────────┐
│  L2 · Orchestration Layer                              │
│  packages/orchestration (Mastra, TypeScript)           │
│  ─────────────────────────────────────────────────────  │
│  agents/                                               │
│  ├─ orchestrator.ts (supervisor)                       │
│  ├─ trader.ts (plan-based trade)                       │
│  └─ risk.ts (approval + RiskGuard checks)              │
│                                                         │
│  tools/ ← 22 tool groups, no direct execution          │
│  ├─ trade.create_plan / .approve_plan / .execute_plan │
│  ├─ paper.run_backtest / .author_strategy              │
│  ├─ factor.timing / .score / .catalog                  │
│  ├─ research.deep_dive                                 │
│  ├─ data.get_bars / .get_fundamentals / .get_news      │
│  ├─ evolver.run_evolution                              │
│  └─ web.search / .search_news (external MCP optional)  │
│                                                         │
│  hooks/ (5 lifecycle events)                           │
│  ├─ SessionStart                                       │
│  ├─ PreToolUse (risk precheck, auth, ast-audit)        │
│  ├─ PostToolUse (audit log, notify)                    │
│  ├─ PostToolUseFailure                                 │
│  └─ Stop (residual plan warning)                       │
│                                                         │
│  permissions/ (allow / ask / deny)                     │
│  └─ live.submit_order ≡ deny (LLM invisible)           │
│                                                         │
│  plan/exec + approval (PostgreSQL-backed)              │
│  └─ trade_plans, approval_tokens (TTL 5 min default)   │
│                                                         │
│  memory/ (PostgresStore, conversation history)         │
└────────────────────┬─────────────────────────────────┘
                     │ HTTP + JWT
        ┌────┬────┬─────┬─────┬────┐
        ▼    ▼    ▼     ▼     ▼    ▼
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│ data     │ paper    │ research │ factor   │ evolver  │
│ :8001    │ :8002    │ :8003    │ :8004    │ :8005    │
│ FastAPI  │ FastAPI  │ FastAPI  │ FastAPI  │ FastAPI  │
└──────────┴──────────┴──────────┴──────────┴──────────┘
        │        │
        └────┬────┘
             ▼
┌─────────────────────────────────────────────────────────┐
│  L4 · Persistence + External                           │
│  ─────────────────────────────────────────────────────  │
│  PostgreSQL 17 + TimescaleDB                           │
│  ├─ Hypertables: bars, ticks (time-series)            │
│  ├─ trade_plans, approval_tokens, orders, positions   │
│  ├─ backtest_runs, strategy_candidates, runs          │
│  ├─ risk_locks (per-account, per-venue, per-symbol)   │
│  └─ factor_snapshots, factor_candidates               │
│                                                         │
│  External Venues                                        │
│  ├─ CCXT (crypto: Binance, etc.)                      │
│  ├─ akshare (A-shares, HK equities)                    │
│  ├─ yfinance (US equities, global indices)             │
│  ├─ FRED (macro: CPI, unemployment, spreads)           │
│  └─ DDGS web search (zero-key)                         │
└─────────────────────────────────────────────────────────┘
```

### Service Responsibilities

| Service | Port | What It Owns |
|---------|------|-------------|
| **data** | 8001 | Market data (bars, ticks, fundamentals), web search, FX conversion. No strategy execution, signal-only. |
| **paper** | 8002 | Event-driven kernel (Clock, MessageBus, Strategy, Gateway); backtest + paper trading (same code, swap Clock/Gateway); live runner on closed bars; `POST /orders/submit` with RiskGuard pre-gate. |
| **research** | 8003 | Multi-analyst deep dive: 6 parallel analysts (technical, fundamental, sentiment, valuation, macro, crypto) + bull/bear/risk debate (triggered only if analysts disagree). Produces `StrategyHint` (no direct order placement). |
| **factor** | 8004 | Factor library (79 factors: pandas-ta, Alpha101, qlib, FRED macro). IC screening, time-series Rank IC for current-effective factor timing, lineage & decay watch, DSL factor discovery. Signals only, never executes. |
| **evolver** | 8005 | E1 strategy evolution: owner-scoped async runs, frozen dataset & LLM snapshot, unified-diff mutation, explicit approval (Ed25519 credential grant), candidate evaluation. Never auto-promotes or trades. |

### Sandbox / Production Split

- **Backtest environment:** `TestClock` (data-driven time), mock `Gateway`, `InMemoryLockStore` for RiskEngine.
- **Paper trading environment:** `LiveClock` (system time), mock `Gateway` (simulated fills), PG-backed `risk_locks` for RiskGuard.
- **Live trading:** Intentionally out of scope in current design. Architecture supports it (swap `Clock` + `Gateway`), but no real-money code path exists.

### Main Loop Timing

```
User chat → Orchestrator (supervisor) 
  → Trader agent (plan-focused)
    → trade.create_plan (HTTP → paper service)
      → Returns planId + pending status
  
[N seconds to N hours delay — approval async]

Risk agent (or user UI) 
  → trade.approve_plan (HTTP → paper service)
    → Returns approval_token (one-shot, 5 min TTL)

Trader agent  
  → trade.execute_plan (planId, token)
    → PreToolUse hook verifies token + re-runs RiskGuard
    → HTTP → paper POST /orders/submit
      → Atomic: consume token + match order + update positions + deduct cash
      → Return orderId + status
    → PostToolUse hook: audit log
```

**Key invariant:** The LLM sees tools `trade.create_plan`, `trade.approve_plan`, `trade.execute_plan` but NOT any tool like `live.submit_order` or `live.close_all_positions`. The permission engine's `deny` list ensures `live.*` tools are invisible to the orchestrator entirely (file: `packages/orchestration/src/permissions/defaults.ts`).

---

## 4. MACHINE APPROVAL — THE DEEP SECTION

### What "Every Order Through Machine Approval" Means in Code

**Three-Step Flow, One Shot Token:**

1. **Create Plan** (`trade.create_plan`)  
   **File:** `packages/orchestration/src/tools/trade-plan.ts:74–174`  
   **HTTP Endpoint:** `POST /plans` (paper service, `services/paper/src/inalpha_paper/api/trade_plans.py`)  
   **Payload:**
   ```json
   {
     "intent": "open_long|open_short|close|rebalance",
     "venue": "binance",
     "symbol": "BTC/USDT",
     "side": "BUY|SELL",
     "orderType": "MARKET|LIMIT",
     "quantity": 0.01,
     "price": 45000.0,
     "rationale": "Bullish divergence on 4h RSI; D-12 factor IC rank top 3",
     "researchId": "<uuid>",         // Links to research deep_dive
     "backtestRunId": "<uuid>",      // Links to backtest run
     "tradingMode": "spot|perp",
     "leverage": 1
   }
   ```
   **Return:**
   ```json
   {
     "ok": true,
     "planId": "<uuid>",
     "status": "pending_approval",
     "intent": "open_long",
     "symbol": "BTC/USDT",
     "venue": "binance",
     "orderParams": { ... },
     "rationale": "[research:<uuid>] [backtest:<uuid>] Bullish divergence...",
     "createdAt": "2026-09-12T10:00:00Z",
     "expireAt": "2026-09-12T10:05:00Z",
     "approvalRequiredBy": "risk-agent-or-user"
   }
   ```
   **What happens in paper service:**
   - Validation: intent/venue/symbol/side/quantity checks (file: `services/paper/src/inalpha_paper/api/trade_plans.py:POST_/plans`)
   - **No RiskGuard check here** — only on execute.
   - Write to `trade_plans` table (PostgreSQL):
     ```sql
     INSERT INTO trade_plans (
       plan_id, account_id, intent, venue, symbol, side, order_type, 
       quantity, price, rationale, status, created_at, expire_at
     ) VALUES (...)
     ```
   - Status column set to `'pending_approval'`.
   - Return plan with TTL expiry (`expire_at = now + 5 min default`).

2. **Approve Plan** (`trade.approve_plan`)  
   **File:** `packages/orchestration/src/tools/trade-plan.ts:180–216`  
   **HTTP Endpoint:** `POST /plans/{id}/approve` (paper service)  
   **Payload:**
   ```json
   {
     "planId": "<uuid>",
     "approver": "risk-agent|user:<uuid>"
   }
   ```
   **Return:**
   ```json
   {
     "ok": true,
     "planId": "<uuid>",
     "status": "approved",
     "approvalToken": "eyJhbGc...",  // Single-use JWT-like token
     "approvedAt": "2026-09-12T10:01:00Z"
   }
   ```
   **What happens in paper service:**
   - Query `trade_plans` table, verify status = `'pending_approval'` and not expired.
   - Generate one-shot, short-lived `approval_token` (file: `services/paper/src/inalpha_paper/api/trade_plans.py:_generate_approval_token()`).
   - **Token is NOT a JWT.** It is a random UUID stored in `approval_tokens` table (PostgreSQL) with:
     ```sql
     INSERT INTO approval_tokens (
       token_id, plan_id, created_at, expires_at, consumed, consumed_at
     ) VALUES (...)
     ```
     **Expires in 5 minutes** by default.
   - Update plan status to `'approved'`.
   - Return token to approver.
   - **Token cannot be replayed.** Once consumed in step 3, the row gets `consumed=true`, and further attempts fail with 409 CONFLICT.

3. **Execute Plan** (`trade.execute_plan`)  
   **File:** `packages/orchestration/src/tools/trade-plan.ts:261–303`  
   **HTTP Endpoint:** `POST /plans/{id}/execute` (paper service)  
   **Payload:**
   ```json
   {
     "planId": "<uuid>",
     "approvalToken": "<token>"
   }
   ```
   **Return:**
   ```json
   {
     "ok": true,
     "planId": "<uuid>",
     "planStatus": "executed",
     "order": {
       "clientOrderId": "<uuid>",
       "status": "filled|rejected",
       "filledQuantity": 0.01,
       "avgFillPrice": 44999.50,
       "fee": 0.45,
       "notional": 449.99,
       "rejectionReason": null
     }
   }
   ```
   **What happens in paper service (the critical gate):**
   - **PreToolUse Hook** (`packages/orchestration/src/hooks/handlers/pending-plan-check.ts`):
     - Verify `approval_token` exists in `approval_tokens` table, matches `plan_id`, not yet consumed, not expired.
     - If invalid: reject with 401 or 403.
   - **RiskGuard enforcement** (`services/paper/src/inalpha_paper/execution/risk_guard.py:enforce()`):
     - Query `risk_locks` table for existing global/market/symbol-scoped locks.
     - If locked: reject with 409 `RISK_REJECTED`.
     - If not locked, run each `RiskRule.check_*` method:
       - `check_global()`: Drawdown veto (default −20%).
       - `check_market()`: Market hours (crypto 24/7, US equity 9:30–16:00 ET, etc.).
       - `check_symbol()`: Per-symbol limits, cooldown between trades.
     - If any rule fails: write lock to `risk_locks` table + return rejection.
   - **Atomic execution** (single PostgreSQL transaction):
     ```sql
     BEGIN;
     
     -- 1. Verify & consume approval token
     UPDATE approval_tokens 
     SET consumed=true, consumed_at=now() 
     WHERE token_id=... AND NOT consumed;
     
     -- 2. Execute order (OrderExecutor.execute() pure function)
     --    - Match order against ref_price (from data service or cached)
     --    - Calculate fill, fees, commissions
     --    - Determine final status (filled / partially filled / rejected)
     
     -- 3. Write order to orders table
     INSERT INTO orders (...) VALUES (...);
     
     -- 4. Apply fill to positions + cash
     --    - If closing position: write to closed_trades table (for RiskRule triggers)
     --    - Update positions table quantity + realized/unrealized PnL
     --    - Deduct cash_balances by notional + fee
     
     -- 5. Update plan status
     UPDATE trade_plans SET status='executed', resulting_order_id=... WHERE plan_id=...;
     
     COMMIT;
     ```
     If **any step fails or violates constraints (e.g., insufficient margin)**, the entire transaction rolls back — order not written, cash not deducted, plan stays `approved` (not consumed token still valid to retry).

### Approval Schema & Constraints

**Approval Token Immutability:**
- **File:** `services/paper/src/inalpha_paper/storage/approval_tokens.py`
- One token per plan, one plan per token.
- Once `consumed=true`, all subsequent execute attempts fail with "token already used."
- Expiration checked at execution time; expired token → 403.
- **Cannot be transferred between planIds** — token is bound to plan_id in the DB.

**Plan Lifecycle States (Finite Automaton):**
```
pending_approval → approved → executed
                ↘ rejected
                ↘ expired
```
- `pending_approval`: Created, awaiting approval.
- `approved`: Approval issued, token minted, awaiting execute within 5 min.
- `executed`: Token consumed, order written to DB, transaction closed.
- `rejected`: Manually rejected by approver (terminal state, rationale recorded).
- `expired`: `expire_at` timestamp exceeded without approval (terminal state).

Files involved:
- `services/paper/src/inalpha_paper/storage/trade_plans.py`: SQL schema + insert/update/query.
- `services/paper/src/inalpha_paper/api/trade_plans.py`: HTTP endpoints.

### What Signs What, Who Has Authority

**LLM Authority:**
- Can call `trade.create_plan` (propose) → writes to DB with status=`pending_approval`.
- **Cannot** call `trade.approve_plan` or `trade.execute_plan` directly.
  - Permission engine: both marked as `permission: ask` (file: `packages/orchestration/src/permissions/defaults.ts`).
  - If LLM tries to call them in same conversation turn, orchestrator returns "approval pending, awaiting human decision."

**Risk Agent Authority:**
- Can call `trade.approve_plan` (review + mint token).
- Can call `trade.execute_plan` with token.
- Runs within the orchestrator, subject to same PreToolUse/PostToolUse hooks.

**User Authority (via UI):**
- Dashboard (`apps/dashboard`) displays pending plans.
- User clicks "Approve" → triggers `trade.approve_plan` server-side (Route Handler calls paper service).
- User clicks "Execute" → triggers `trade.execute_plan` (passes approvalToken from step 2).

**Paper Service Authority (Backend):**
- Owns the `trade_plans` and `approval_tokens` tables.
- Validates all state transitions server-side (LLM/UI cannot bypass).
- `POST /plans/{id}/execute` is the only endpoint that consumes tokens.

### Token One-Shot Mechanism

**Token Tracking:**
- **Table:** `approval_tokens` (Postgres)
  ```sql
  CREATE TABLE approval_tokens (
    token_id TEXT PRIMARY KEY,
    plan_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    consumed BOOLEAN DEFAULT false,
    consumed_at TIMESTAMPTZ
  );
  ```
- **Consumption Logic** (`services/paper/src/inalpha_paper/api/trade_plans.py:POST_/plans/{id}/execute`):
  ```python
  async def execute_plan(plan_id: UUID, approval_token: str, db: AsyncConnection):
    # Check token exists, not yet consumed
    token_row = await db.fetchrow(
      "SELECT * FROM approval_tokens WHERE token_id=$1 AND plan_id=$2",
      approval_token, plan_id
    )
    if not token_row or token_row["consumed"]:
      raise HTTPException(status_code=409, detail="APPROVAL_TOKEN_ALREADY_USED")
    
    # BEGIN transaction
    async with db.transaction():
      # Try to mark token as consumed
      updated = await db.execute(
        "UPDATE approval_tokens SET consumed=true, consumed_at=now() "
        "WHERE token_id=$1 AND NOT consumed",
        approval_token
      )
      if updated == 0:
        raise HTTPException(status_code=409, detail="APPROVAL_TOKEN_ALREADY_USED")
      
      # Proceed with order execution (RiskGuard, matching, writing orders, etc.)
      # If any step fails, ROLLBACK cancels the token update too.
  ```
- **Network Failure Retry Safety:**
  If network fails between RiskGuard pass and COMMIT, the token is NOT consumed (transaction rolled back). Operator can retry with the same token within TTL.

### Approval Diagram (Sequence)

```
Orchestrator                  Trader Agent                Paper Service
     │                               │                            │
     ├─ deep_dive(BTC)               │                            │
     └──────────────────────────────>│                            │
                                     │                            │
                                     ├─ trade.create_plan(...)    │
                                     ├──────────────────────────>│
                                     │                   Hook: PreToolUse (auth, AST audit)
                                     │                   POST /plans
                                     │                   Validate intent/side/qty/rationale
                                     │<──────────────────────────┤
                                     │    planId, status=pending  │
                                     │                            │
     ┌─ WAIT (1 sec ~ 1 hour)        │                            │
     │                               │                            │
     ├─ Risk agent reviews plan      │                            │
     └────────────────────────────────────────────────────────────┤
                                                                  Hook: ask (human approval)
                                                                  POST /plans/{id}/approve
                                                                  Verify status=pending & not expired
                                                                  Generate approval_token (one-shot)
                                                                  INSERT into approval_tokens
     │                               │                            │
     │<──────────────────────────────────────────────────────────┤
     │                               │        approvalToken, TTL   │
     │                               │                            │
     ├─ Trader (or human) calls execute
     │                               │                            │
     │                     trade.execute_plan(planId, token)
     │                     ├────────────────────────────────────>│
     │                     │     Hook: PreToolUse                │
     │                     │     ├─ Verify token in DB            │
     │                     │     ├─ RiskGuard.check() all rules   │
     │                     │     └─ Re-check permissions          │
     │                     │     POST /orders/submit              │
     │                     │     Transaction:                     │
     │                     │      1. UPDATE approval_tokens       │
     │                     │         SET consumed=true            │
     │                     │      2. OrderExecutor.execute()      │
     │                     │      3. INSERT orders                │
     │                     │      4. UPDATE positions + cash      │
     │                     │      5. UPDATE trade_plans status    │
     │                     │     COMMIT                           │
     │                     │<────────────────────────────────────┤
     │                     │    orderId, status=submitted         │
     │                     │    (or RISK_REJECTED, 409)           │
     │                     │                                      │
     │<────────────────────┤                                      │
     ├─ Stop hook: check no residual pending plans
```

---

## 5. THE LLM'S ACTUAL ROLE — PRECISELY WHERE IT IS CALLED

### Where the LLM is Invoked (with prompts quoted)

**1. Orchestrator Agent Initialization**  
**File:** `packages/orchestration/src/mastra/agents/orchestrator.ts`  
**Call Flow:**
- Receives user message (e.g., "Help me research BTC then open a long").
- System prompt (file: `packages/orchestration/src/mastra/agents/orchestrator.ts:SYSTEM_PROMPT`):
  ```
  You are the Inalpha orchestrator agent. Your job is to research the market,
  compose a strategy, backtest it if needed, and then propose a trade plan.
  
  Your tools are:
  - research.deep_dive: deep fundamental/sentiment/technical dive
  - paper.author_strategy: write full Python Strategy code
  - paper.run_backtest: run the strategy on historical or paper bars
  - trade.create_plan: propose a trade (NOT execute)
  - data.get_bars: fetch market data
  - factor.timing: rank factors by current effectiveness
  
  Tools you CANNOT see or call:
  - live.submit_order (does not exist in your tool list)
  - live.close_all_positions (does not exist)
  
  You must NEVER try to execute trades directly. Always use trade.create_plan
  to propose, then wait for approval before trade.execute_plan.
  ```
- Mastra runtime: calls `orchestrator.run({ message: user_msg })`.
- LLM decision: which tool to call, what arguments.

**2. Trader Agent (Strategy Composition)**  
**File:** `packages/orchestration/src/mastra/agents/trader.ts`  
**Call:**
- Receives research plan from orchestrator (containing factors, strategy hint, thesis).
- **Option A:** Call `paper.author_strategy(description, hints)`.
  - LLM generates Python code from description.
  - Prompt (from tool description, `packages/orchestration/src/tools/paper-strategy.ts`):
    ```
    Generate a complete Python Strategy subclass that:
    - Inherits from Strategy
    - Implements on_bar(self, bar: Bar)
    - Uses factors from the provided hint list
    - Submits orders via self.submit_order(order)
    
    Example template:
    class MyStrategy(Strategy):
      def __init__(self, config):
        super().__init__(config)
        self.period = 20
      
      def on_bar(self, bar):
        # Logic here
        order = Order(...)
        self.submit_order(order)
    
    Do NOT use eval, exec, __import__, or access __class__.
    ```
  - LLM output: full Python code as string.
  - Paper service: AST audit + dynamic_loader (run code in restricted namespace) + contract check.
  - If passes: return `strategy_id`, else reject with details.

- **Option B:** Call `paper.compose_strategy(strategy_family, parameters)`.
  - LLM chooses from 3 built-in templates: `sma_cross`, `mean_reversion`, `buy_and_hold`.
  - Paper service: clip parameters to valid range, compose strategy inline.

**3. Backtest Execution**  
**File:** `services/paper/src/inalpha_paper/api/backtest.py`  
**LLM Authority:** None. Orchestrator calls `paper.run_backtest(strategy_id, bars_range, parameters)`.
- LLM does NOT write the backtest engine.
- Paper service: loads strategy code from DB (author_strategy path) or built-in, runs it on historical bars, returns fitness + returns + Sharpe/Calmar.
- Deterministic: same strategy + same bars = same result.

**4. Plan Proposal**  
**File:** `packages/orchestration/src/tools/trade-plan.ts:74–174`  
**Call:** Trader calls `trade.create_plan({intent, venue, symbol, side, quantity, rationale, ...})`.
- **Rationale** (mandatory): LLM writes the _why_ (e.g., "Bullish divergence on 4h RSI, factor IC rank top 3").
- LLM authority: **Propose only.** Cannot approve or execute.
- Paper service writes plan to DB with status `pending_approval`.
- **LLM does NOT have the approval_token.** Approval is async, by a human or separate risk agent.

**5. Research Deep Dive**  
**File:** `services/research/src/inalpha_research/manager.py`  
**LLM Authority:** Each analyst (technical, fundamental, sentiment, valuation, macro, crypto) is a separate LLM call.
- Prompt template (file: `services/research/src/inalpha_research/analysts/technical.py`):
  ```
  You are a technical analyst. Analyze {symbol} using {lookback_days} days of data.
  
  Data provided:
  - bars (OHLCV)
  - factors (current-effective by Rank IC)
  
  Output: structured analysis with
  - key levels (support, resistance)
  - momentum signals (RSI, MACD, etc.)
  - trend direction
  - confidence 0..1
  - factors used
  
  Do NOT make trading recommendations directly. Output analysis only.
  ```
- Output: `TechnicalAnalysis(levels={...}, signals=[...], confidence=0.75, factors=[...])`.
- If analysts disagree (e.g., tech bullish but fundamental bearish), trigger bull/bear debate.
- **LLM does NOT place orders.** Output feeds into strategy composition.

**6. Factor Timing**  
**File:** `services/factor/src/inalpha_factor/api/score.py`  
**LLM Authority:** None. Deterministic computation.
- Rank IC calculated by Python code (file: `services/factor/src/inalpha_factor/computation/rank_ic.py`).
- Factors ranked by correlation with forward returns over a rolling window.
- Top-ranked factors returned to LLM in context (read-only).

**7. Strategy Evolution (E1)**  
**File:** `services/evolver/src/inalpha_evolver/api/routes.py`  
**LLM Authority (if approval granted):**
- User requests `evolver.run_evolution(candidate_strategy_code, ...)`  → ask permission.
- Dashboard shows "Will cost $X.XX for this evolution run. Approve?"
- User approves → Orchestration signs Ed25519 credential grant (file: `packages/orchestration/src/permissions/approval-identity.ts`).
- Grant bound to owner + operation_id + config_id (model, pricing, endpoint) for 30 hours.
- Evolver receives grant, verifies signature, unmarshals credentials.
- LLM (via Evolver) receives: "Improve this strategy: [seed code]". Output: unified diff only.
- Evolver applies diff → AST audit → load in sandbox → run on frozen bars → return fitness.
- **LLM does NOT auto-promote, auto-start, or auto-trade candidates.** Promotion is a separate, explicit Dashboard action.

### Under Bitget Track 2 Rules

**Track 2 Rule:** "The LLM is the primary trading decision-maker, not just an assistant. The Agent must sense the environment, make independent judgments, and autonomously place orders with risk controls."

**VERDICT on Inalpha's Stance:**  
**WOULD FAIL BITGET TRACK 2 REQUIREMENT** ❌

**Reasoning:**

1. **The LLM is kept OFF the order path intentionally.**
   - Inalpha's philosophy: "no LLM reaches the order path unsupervised" (README.md §2).
   - LLM can call `trade.create_plan` (propose), but the architecture explicitly prevents it from calling any tool to approve or execute.
   - Permission engine marks both `trade.approve_plan` and `trade.execute_plan` as `ask` (human decision point).
   - Trader cannot autonomously proceed from plan → approval → execution in one agent turn without human intervention.

2. **Primary decision-maker vs. assistant.**
   - Under Inalpha: LLM is a **code-writing assistant** for strategy composition and proposal.
   - It makes research recommendations (analyst role) and proposes trades (trader role), but approval and execution are gated by human or separate risk agent.
   - Bitget Track 2 asks for the LLM to be the **primary decision-maker** — i.e., make the call to place an order, with risk controls as guardrails, not approval gates.

3. **"Autonomously place orders" constraint.**
   - Inalpha's design: approval token is one-shot and short-lived, but it's issued asynchronously, not inline with LLM decision.
   - Autonomy implies the agent can decide *and execute* without human approval delays.
   - Here, even if the risk agent (code-based) approves, execution still requires the token + the execute call.

**Honest Call:**
- Inalpha's machine-approval approach is **arguably more robust** for real-money trading (human in the loop, no backdoor, audit trail is tamper-evident).
- But it **does not satisfy Bitget's Track 2 definition**, which expects the LLM to be the decision-maker, not gated at approval.
- **ARGUS should invert this:** LLM is decision-maker, risk controls are passive guardrails (reject at execution time if violated), and approval is receipt-based, not a separate gate.

---

## 6. FACTOR TIMING AND THE FACTOR LAB

### How Factors are Selected as "Working Now"

**Mechanism: Time-Series Rank IC**

**File:** `services/factor/src/inalpha_factor/computation/rank_ic.py`

**Algorithm:**
1. For each factor `f` in the library (79 total):
   - Compute factor values on historical bars (e.g., RSI(14), MACD, Bollinger Bands Width, etc.).
   - Compute forward returns `r[t+1..t+h]` (h = holding period, e.g., 1 bar, 5 bars, 1 day).
   - Rank both factor and returns independently (cross-sectional for the universe, or time-series for single name).
   - Calculate correlation between ranked factor and ranked returns: **Rank IC**.
   - Example: if top-decile high RSI stocks have highest forward returns, RSI has positive Rank IC.

2. **Rolling window:** Recalculate every bar or every N bars.
   - File: `services/factor/src/inalpha_factor/api/score.py:factor_timing(symbol, as_of, lookback_bars=...)`
   - Returns top-K factors by Rank IC.

3. **Output to LLM:**
   - Orchestrator calls `factor.timing(symbol="BTC/USDT", as_of="2026-09-12T10:00:00Z")`.
   - Response (file: `services/factor/src/inalpha_factor/api/presenters.py`):
     ```json
     {
       "symbol": "BTC/USDT",
       "asOf": "2026-09-12T10:00:00Z",
       "lookbackBars": 100,
       "topFactors": [
         {
           "name": "RSI(14)",
           "rankIC": 0.42,
           "pValue": 0.001,
           "window": "100 bars (last 25 days at 4h)"
         },
         {
           "name": "MACD_histogram",
           "rankIC": 0.38,
           "pValue": 0.003,
           "window": "100 bars"
         },
         {
           "name": "BBW(20,2)",
           "rankIC": 0.29,
           "pValue": 0.011,
           "window": "100 bars"
         }
       ]
     }
     ```

### The Factor Lab's Evaluation Loop

**Workflow:**

1. **Proposal** (user or LLM):
   - Input: "Is low volatility a positive factor for BTC 2026-08?"
   - Tool: `factor.propose_factor(name="volatility_low", description="...", formula="...")`
   - File: `services/factor/src/inalpha_factor/api/custom.py`

2. **Formalization** (LLM or deterministic):
   - Compute factor values.
   - File: `services/factor/src/inalpha_factor/computation/compute.py`
   - **AST audit** (if user-provided formula): reject if uses `eval`, `exec`, forbidden modules.

3. **Statistical Testing** (deterministic, no LLM):
   - Rank IC, p-value, multiple-testing correction (Benjamini–Hochberg FDR).
   - File: `services/factor/src/inalpha_factor/computation/statistical_tests.py`
   - **Guardrails:** null-IC benchmark (shuffle factor values, compute IC on random, check for luck).

4. **Economic Story Gate** (human, not LLM auto-approve):
   - Prompt template (file: `services/factor/src/inalpha_factor/api/custom.py`):
     ```
     Factor name: volatility_low
     Statistical result: Rank IC = 0.15, p = 0.23 (not significant)
     
     Does this factor make economic sense?
     - Rationale: Low vol often precedes range-breakouts.
     - Industry precedent: Yes, classic mean-reversion signal.
     - Confounds: Maybe collinear with other factors?
     
     Decision: Reject (p > 0.05, not statistically significant).
     ```
   - **Approval:** Dashboard gate (human clicks "Register to Library").
   - No automatic promotion.

5. **Lineage & Decay Watch** (deterministic, ongoing):
   - File: `services/factor/src/inalpha_factor/api/snapshot.py` + `services/factor/src/inalpha_factor/api/decay.py`
   - Track which factors each strategy depends on.
   - Monthly: re-run IC on all factors, compare to baseline.
   - If Rank IC decays > 50%: alert (no auto-trim).

### Factor Library Catalog (79 Factors)

**File:** `services/factor/src/inalpha_factor/api/catalog.py:factor_catalog()`

**Families:**
- **Technical** (pandas-ta): RSI(14), MACD, Bollinger Bands Width, Stochastic, CCI, etc. (30 factors).
- **Alpha101** (Kakushadze, 2015): cross-sectional factors (a1, a3, a5, ... a100). Pre-computed for panel scoring. (100 factors, subset used).
- **qlib** (Tencent): deep learning + econometric factors. (subset integrated).
- **Macro** (FRED): CPI, unemployment, payrolls, credit spreads, curve slopes, sentiment indices. (20 factors, monthly).

**Endpoint:**
- `GET /catalog?market=crypto&as_of=2026-09-12T10:00:00Z`
- Response: list of 79 factors with current Rank IC, lineage, author, creation date, decay status.

---

## 7. STRATEGY EVOLUTION IN THE SANDBOX

### How Strategies Mutate

**Mechanism: Unified Diff, One Generation**

**File:** `services/evolver/src/inalpha_evolver/api/run_routes.py`

**Workflow:**

1. **Seed Strategy:**
   - Starting point: a promoted strategy from the library (e.g., `buy_and_hold`, custom user strategy).
   - Frozen as of approval: code snapshot stored in `strategy_candidates` or `strategies` table.

2. **LLM Mutation:**
   - Prompt (file: `services/evolver/src/inalpha_evolver/api/routes.py`):
     ```
     Current strategy code:
     ```python
     class MyStrategy(Strategy):
       def __init__(self, config):
         super().__init__(config)
         self.period = 20
       
       def on_bar(self, bar):
         if bar.close > SMA(bar, self.period):
           self.submit_order(Order(side='BUY', qty=self.position_size))
     ```
     
     Suggestions:
     1. Add a volatility filter (BBW or ATR).
     2. Add a mean-reversion counter-trend signal.
     3. Reduce drawdown via tighter stops.
     
     Output a **unified diff** (minimal changes, readable for audit):
     ```
     --- a/strategy.py
     +++ b/strategy.py
     @@ -5,3 +5,7 @@
      def on_bar(self, bar):
     -  if bar.close > SMA(bar, self.period):
     +  bbw = BBW(bar, self.period, self.bbw_std)
     +  if bar.close > SMA(bar, self.period) and bbw > 0.3:
     ```
     ```
   - LLM output: diff as text (not code execution, just string).

3. **Diff Application & Audit:**
   - File: `services/evolver/src/inalpha_evolver/data/mutation_applier.py`
   - Apply diff to seed code using Python's `difflib`.
   - **AST audit** (file: `services/evolver/src/inalpha_evolver/validators/ast_audit.py`):
     - Whitelist checks: only allow specific imports, no `eval`, `exec`, `os`, `sys`, `__import__`.
     - Reject malicious patterns (e.g., `.__class__.__bases__`).
   - **Restricted loader** (file: `services/evolver/src/inalpha_evolver/validators/dynamic_loader.py`):
     - `exec()` in a minimal namespace (Strategy, Bar, Order, pandas, numpy, ta, qlib only).
     - No `__builtins__` except safe subset (len, range, sum, min, max).
   - **Contract check** (file: `services/evolver/src/inalpha_evolver/validators/contract_check.py`):
     - Verify class inherits from `Strategy`.
     - Verify `on_bar(self, bar)` exists and callable.
     - Verify `__init__` signature matches parent.
   - If all checks pass: candidate stored in `strategy_candidates` table. Else: reject with details.

4. **Evaluation (No LLM):**
   - File: `services/paper/src/inalpha_paper/strategy_evaluation.py`
   - Run candidate on **frozen, historical bars** (same data for all candidates in the run).
   - Compute fitness: `sharpe + 0.3*calmar - 0.10*turnover - 1.0*(drawdown > 30%)`.
   - Also run `buy_and_hold` baseline.
   - If candidate fitness >> baseline: alpha detected.
   - Return fitness, returns, Sharpe, turnover, drawdown, etc.

5. **Promotion (Human Gate, Not Auto):**
   - File: `apps/dashboard/...` (UI component) + `packages/orchestration/src/tools/paper-strategy.ts:promote_candidate`
   - Dashboard lists candidates with fitness rank.
   - User clicks "Promote to Live" → `POST /strategy_candidates/{id}/promote`.
   - Promoted strategy moves to `status='active'` and can be assigned to a live runner.
   - **No automatic promotion.** No auto-start of live runner. **Zero automation after evolution.**

### Fitness Function (Multi-Objective)

**File:** `services/paper/src/inalpha_paper/strategy_authoring/fitness.py`

```python
def compute_fitness(backtest_result):
    sharpe = backtest_result.sharpe_ratio
    calmar = backtest_result.calmar_ratio
    turnover = backtest_result.turnover_penalty  # penalize high trading
    drawdown_penalty = 1.0 if backtest_result.max_drawdown > 0.30 else 0
    
    fitness = (
        sharpe
        + 0.3 * calmar
        - 0.10 * turnover
        - 1.0 * drawdown_penalty
    )
    return fitness
```

**Why multi-objective:**
- Pure Sharpe can be gamed via high turnover (costs ignored).
- Calmar (return / max drawdown) captures downside risk.
- Turnover penalty discourages over-trading.
- Drawdown veto: strategies with >30% max DD auto-penalized (one-shot −1.0).

### LLM Role in Evolution

- **YES:** Proposes mutations (as unified diff text, not executable code).
- **NO:** Does not execute the strategy or the backtest.
- **NO:** Does not auto-promote or trade candidates.
- **YES (if asked):** Interprets fitness metrics ("your Sharpe is 1.8, Calmar is 0.6, turnover is 15% — is this good?").

---

## 8. THE RISK ENGINE — EVERY GATE, EVERY RULE

### Risk Rules (5 Layers)

**Configuration File:** `services/paper/configs/risk_rules.toml`

**File:** `services/paper/src/inalpha_paper/execution/risk_rules/`

**Rules (all enforced at HTTP boundary, not in LLM prompts):**

| # | Rule Name | File | Scope | Logic | Trigger |
|---|-----------|------|-------|-------|---------|
| 1 | **MaxDrawdown** | `max_drawdown.py` | Global | Max cumulative loss from peak = −20% (default, configurable). | `check_global()` before any trade. If realized loss ≥ −20%: reject + lock globally. |
| 2 | **MarketHoursRule** | `market_hours.py` | Market | Crypto: 24/7 OK. US equity: 9:30–16:00 ET (9:30–12:00 on half-days). A-share: 9:30–11:30 + 13:00–15:00 CST. etc. | `check_market()` per venue/symbol. Queries `exchange_calendars`, resolves trading hours, checks order timestamp against holiday + DST calendar. If outside hours: reject + lock market. |
| 3 | **CooldownRule** | `cooldown.py` | Symbol | After closing a position, wait N seconds before re-opening same symbol. | `check_symbol()` if `closed_trades` table shows recent close. If cooldown not elapsed: reject + lock symbol. |
| 4 | **LowProfitRule** | `low_profit.py` | Symbol | If last closed trade had realized loss > −2% (default), cooldown extends. | `check_symbol()`. Reads `closed_trades.realized_pnl`. If loss detected: extend cooldown. |
| 5 | **StoplossGuardRule** | `stoploss_guard.py` | Symbol | Position stop-loss (per-instrument limit). | `check_symbol()`. Optional, currently Noop (file: `stoploss_guard.py` has placeholder). |

**Files involved:**
- Rule base class: `services/paper/src/inalpha_paper/execution/risk_rules/base.py:RiskRule`
- Config loader: `services/paper/src/inalpha_paper/execution/risk_rules/config.py:build_rules_from_toml()`
- Lock storage (PG-backed): `services/paper/src/inalpha_paper/storage/risk_locks.py`
- Lock manager (per-account): `services/paper/src/inalpha_paper/execution/risk_guard_factory.py:RiskGuardFactory`

### How Rules Enforce

**HTTP Path (Paper Service):**

**File:** `services/paper/src/inalpha_paper/api/orders.py:post_submit_order()`

```python
@router.post("/orders/submit", response_model=SubmitOrderResponse)
async def post_submit_order(req: SubmitOrderRequest, db: DBConn, ...):
    account_id = account_id_from_user(user)
    
    # D-9 风控前置闸门：先 RiskGuard.enforce，再下单
    factory = request.app.state.risk_guard_factory
    await risk_guard_mod.enforce(
        factory,
        account_id=account_id,
        venue=req.venue,
        symbol=req.symbol,
        side=req.side,
    )
    # If above raises RiskRejection → 409 RISK_REJECTED, ends here.
    
    # RiskGuard passed → continue with order execution.
    # ... (matching, position update, cash deduct)
```

**File:** `services/paper/src/inalpha_paper/execution/risk_guard.py:RiskGuard.check()`

```python
async def check(self, conn: AsyncConnection, *, instrument_id: InstrumentId, side: Side, now: datetime) -> RiskRejection | None:
    # 1. Query existing locks (global, market, symbol)
    for scope, kwargs in [("global", {}), ("market", {...}), ("symbol", {...})]:
        existing = await locks_store.is_locked(conn, scope=scope, side=side, account_id=account_id, **kwargs)
        if existing:
            return _rejection_from_lock_row(existing)
    
    # 2. Run rule checks (sync)
    for rule in self._rules:
        if rule.has_global_check:
            verdict = rule.check_global(now, side, balance=self._starting_balance)
            if verdict:
                return await _record_and_build_rejection(conn, verdict, ...)
    
    for rule in self._rules:
        if rule.has_market_check:
            verdict = rule.check_market(instrument_id, now, side, balance)
            if verdict:
                return await _record_and_build_rejection(conn, verdict, ...)
    
    for rule in self._rules:
        if rule.has_symbol_check:
            verdict = rule.check_symbol(instrument_id, now, side, balance)
            if verdict:
                return await _record_and_build_rejection(conn, verdict, ...)
    
    return None  # All checks pass
```

**File:** `services/paper/src/inalpha_paper/storage/risk_locks.py:record_lock()`

```python
async def record_lock(conn: AsyncConnection, rejection: RiskRejection, account_id: str):
    """Write rejection to risk_locks table with independent connection."""
    await conn.execute("""
        INSERT INTO risk_locks (
            account_id, rule_name, reason, scope, locked_until, side, market, symbol, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, now())
    """, account_id, rejection.rule_name, rejection.reason, rejection.lock_scope, rejection.locked_until, ...)
```

**Lock Scope & Granularity:**
- **global:** Affects entire account. Example: max drawdown hit, all trading frozen until recovery.
- **market:** Affects venue-wide (e.g., `binance`, `nasdaq`). Example: market closed for holiday, crypto unaffected.
- **symbol:** Affects one symbol (e.g., `BTC/USDT`). Example: cooldown after closing BTC position.

### Relationship to Approval Layer

- **NOT equivalent.** Approval (create/approve/execute) is about authorization flow. Risk rules are about market constraints.
- **Orthogonal:** An order can be approved but rejected by RiskGuard (e.g., approved during open market, but executed after close).
- **Execute endpoint catches both:** PreToolUse hook verifies approval_token, _then_ executes RiskGuard.check(). If RiskGuard fails, approval_token is NOT consumed (transaction rolls back).

---

## 9. THE AUDIT LEDGER — TAMPER-EVIDENT RECORD

### What is Recorded

**Tables (PostgreSQL):**

| Table | What's Recorded | Retention | Signed/Hashed |
|-------|---|---|---|
| **trade_plans** | proposal intent, venue, symbol, side, qty, price, rationale, lineage (research_id, backtest_id), status, approval_token_id, created_at, approved_at, executed_at | Perpetual | No (but referenced by immutable approval_token) |
| **approval_tokens** | token_id, plan_id, created_at, expires_at, consumed, consumed_at, consumed_by | Perpetual | Token is random UUID, checked against plan_id |
| **orders** | order_id, plan_id, account_id, venue, symbol, side, qty, fill_price, fee, notional, status, created_at, filled_at | Perpetual | No |
| **positions** | position_id, account_id, venue, symbol, quantity, avg_entry_price, realized_pnl, unrealized_pnl, ts_opened, open_order_id, ts_closed, close_order_id | Perpetual | No |
| **closed_trades** | closed_trade_id, account_id, venue, symbol, entry_price, exit_price, qty, realized_pnl, ts_opened, ts_closed, duration_bars | Perpetual | No |
| **risk_locks** | lock_id, account_id, rule_name, reason, scope (global/market/symbol), locked_until, side, market, symbol, created_at | Perpetual (or aged) | No |
| **strategy_runs** | run_id, strategy_id, backtest_id, account_id, params_hash, fitness, sharpe, calmar, returns, turnover, max_drawdown, created_at | Perpetual | Params hashed (SHA-256) |
| **strategy_candidates** | candidate_id, account_id, parent_strategy_id, source_code, source_code_hash, fitness, status (candidate/promoted/active), created_at, promoted_at, promoted_by | Perpetual | Source code hashed |
| **evolution_runs** | run_id, account_id, owner_subject, operation_id, config_id, config_digest, llm_snapshot (frozen: provider, model, version, pricing), input_tokens, output_tokens, llm_cost_usd, candidate_id, status (queued/running/completed/failed), created_at, started_at, completed_at | Perpetual | config_digest (SHA-256), llm_snapshot frozen, Ed25519 credential grant verified |
| **hook_telemetry** (optional, best-effort) | hook_event (PreToolUse, PostToolUse, ...), tool_name, context_hash, approval_state, timestamp | Aged (30 days) | Context hashed, deidentified |

**Files Implementing Storage:**
- `services/paper/src/inalpha_paper/storage/trade_plans.py`
- `services/paper/src/inalpha_paper/storage/approval_tokens.py`
- `services/paper/src/inalpha_paper/storage/orders.py`
- `services/paper/src/inalpha_paper/storage/positions.py`
- `services/paper/src/inalpha_paper/storage/closed_trades.py`
- `services/paper/src/inalpha_paper/storage/risk_locks.py`
- `services/evolver/src/inalpha_evolver/api/schemas.py` (evolution_runs table schema)

### Tamper Evidence

**Immutability Mechanisms:**

1. **Approval Token Consumption (One-Shot):**
   - Once `approval_tokens.consumed=true`, any attempt to use the token again fails.
   - Materialized in DB: changing the row requires direct SQL UPDATE.
   - Guard: `WHERE NOT consumed` clause prevents double-spend.

2. **Hash-Based Lineage:**
   - `strategy_candidates.source_code_hash` = SHA-256(source code).
   - Changing code post-hoc would change hash; old hash no longer matches.
   - Any backtest referencing a candidate_id can recalculate hash to verify integrity.

3. **Frozen LLM Snapshot (E1):**
   - File: `services/evolver/src/inalpha_evolver/api/routes.py:_freeze_config()`
   - When approving evolution run, capture:
     - LLM provider, model name, version, base URL.
     - Pricing per input/output token.
     - Hash these fields → `config_digest`.
   - Store config_digest in `evolution_runs.config_digest`.
   - Any price/model change breaks digest; old runs stay unchanged.

4. **Ed25519 Credential Grant Signature:**
   - File: `packages/orchestration/src/permissions/approval-identity.ts:signCredentialGrant()`
   - When user approves evolution, orchestration signs a JWT with Ed25519 private key.
   - JWT payload binds: owner_sub, operation_id, config_digest, request_digest (hash of run params).
   - Evolver verifies signature with public key; if mismatch, run fails.
   - Private key never stored in DB; kept in environment only.
   - Signature is **unforgeable** without the private key.

### Can a Third Party Replay a Decision?

**Yes, but only within scope:**

- A third party with DB read access can read a `trade_plan` row and see the full decision chain:
  - What the LLM proposed (intent, symbol, side, qty, rationale).
  - Who approved (plan.approved_by).
  - When (plan.approved_at, plan.executed_at).
  - What happened (resulting_order_id → order → position → closed_trade → realized_pnl).

- A third party **cannot fake a decision:**
  - Fake approval_token: doesn't exist in DB, ignored.
  - Fake order: requires approval_token, which they don't have.
  - Fake strategy code: hash would mismatch, backtest recalculation would fail.
  - Fake evolution run: Ed25519 signature would fail verification.

- **For forensics:** A third party could reconstruct what happened:
  ```sql
  SELECT * FROM trade_plans WHERE plan_id = '...';
  SELECT * FROM approval_tokens WHERE plan_id = '...';
  SELECT * FROM orders WHERE plan_id = '...';
  SELECT * FROM closed_trades WHERE order_id = '...';
  ```
  This gives complete decision chain: proposed → approved → executed → result.

---

## 10. SENSING THE ENVIRONMENT — DATA SOURCES, TIMESTAMPING, PIT

### Data Sources

**File:** `services/data/src/inalpha_data/connectors/`

| Source | Connector | Symbols | Update Freq | Latency |
|--------|-----------|---------|-------------|---------|
| **Binance (Crypto)** | `ccxt_connector.py` (CCXT library) | BTC/USDT, ETH/USDT, etc. | 1m, 5m, 1h, 1d | <1s (REST API) |
| **Yahoo Finance (US, Global)** | `yfinance_connector.py` | AAPL, ^GSPC, etc. | 1d, 1wk (daily EOD) | T+1 |
| **Akshare (A-Shares, HK)** | `akshare_connector.py` | sh.600519 (Kweichow Moutai), hk.00700 (Tencent) | 1d (daily EOD) | T+1 |
| **FRED (Macro)** | `fred_connector.py` | DFF (Federal Funds Rate), UNRATE (Unemployment), CPI, credit spreads | Monthly | 15 days lag (FRED release lag) |
| **DDGS Web Search** | `web_search_connector.py` | Any ticker, symbol, keyword | Real-time | 10s–30s per query |
| **Eastmoney, SEC, HKEX News** | `news_connector.py` | News feeds by market | Real-time | Variable (seconds to minutes) |

**Files:**
- `services/data/src/inalpha_data/api/bars.py`: `GET /bars`
- `services/data/src/inalpha_data/api/fundamentals.py`: `GET /fundamentals`
- `services/data/src/inalpha_data/api/news.py`: `GET /news`

### Timestamping & `as_of`

**Default Freshness:** `fresh=True`

**File:** `services/data/src/inalpha_data/api/bars.py:get_bars()`

```python
async def get_bars(
    venue: str, symbol: str, timeframe: str,
    since: datetime, until: datetime,
    fresh: bool = True,  # Default: prefer fresh data
    ...
):
    if fresh:
        # 1. Fetch latest from exchange/source (backfill missing bars)
        # 2. Check bars[-1].ts vs as_of; if stale, log warning
        # 3. Return with explicit as_of timestamp
    else:
        # Use cached bars (for backtest, where we want reproducibility)
```

**Freshness Check (Not Bar Count):**

**File:** `services/data/src/inalpha_data/data_client.py:check_freshness()`

```python
def check_freshness(bars: list[Bar], as_of: datetime, tolerance_seconds: int = 300):
    if not bars:
        return False, "No bars available"
    
    latest_bar_time = bars[-1].ts
    staleness = (as_of - latest_bar_time).total_seconds()
    
    if staleness > tolerance_seconds:
        return False, f"Data is {staleness}s stale (tolerance: {tolerance_seconds}s)"
    
    return True, "Data is fresh"
```

- **Not:** "5 bars are recent."
- **Yes:** "Latest bar is at 2026-09-12 10:04:00Z, `as_of` is 10:05:00Z, staleness = 60s, within 5-min tolerance."

### Point-in-Time (PIT) Protection

**Fundamentals (Baostock):**

**File:** `services/data/src/inalpha_data/connectors/akshare_connector.py`

```python
async def get_fundamentals(symbol: str, as_of: datetime):
    """
    Fetch fundamentals reported up to `as_of` (no look-ahead).
    
    Baostock API returns: (symbol, report_date, publish_date, ...)
    Filter:  publish_date <= as_of  (only use released data)
    """
    results = await baostock_client.get_fundamentals(symbol)
    
    # Filter by publish_date <= as_of to prevent look-ahead bias
    pit_filtered = [row for row in results if row["publish_date"] <= as_of]
    
    return pit_filtered[-1]  # Most recent published by as_of
```

**Bars (No PIT Needed for Price):**
- Price bars are by definition "as of the close of that bar."
- No concept of "publication lag" for OHLCV.

**FRED Release Lag Table:**

**File:** `services/data/src/inalpha_data/connectors/fred_connector.py:FRED_RELEASE_LAG_DAYS`

```python
FRED_RELEASE_LAG_DAYS = {
    "DFF": 1,           # Federal Funds Rate released next day
    "UNRATE": 5,        # Unemployment released 5 days after month-end
    "CPIAUCSL": 13,     # CPI released ~13 days after month-end
    "PAYEMS": 7,        # Nonfarm payrolls released ~7 days after month-end
    "M2": 14,           # M2 monetary aggregate released ~14 days after month-end
}

async def get_fred_bars(series_id: str, as_of: datetime):
    lag = FRED_RELEASE_LAG_DAYS.get(series_id, 30)  # Conservative default 30 days
    data_up_to = as_of - timedelta(days=lag)
    
    # Only return FRED data released by data_up_to
    return [bar for bar in bars if bar.ts <= data_up_to]
```

**Absence of PIT in yfinance v1:**

**File:** `services/data/src/inalpha_data/api/fundamentals.py:NOTE`

```python
# NOTE (D-10): yfinance v1.x does NOT return publish_date.
# It returns "ttm" (trailing twelve months), which is a snapshot,
# not point-in-time. We explicitly flag this in responses.

response = {
    "symbol": "AAPL",
    "fundamentals": {
        "pe_ratio": 28.5,
        "roe": 0.105,
    },
    "pit_available": False,  # Explicitly marked
    "warning": "yfinance fundamentals are snapshots, not PIT. Use with caution in backtest."
}
```

---

## 11. COSTS — FEE MODELING, SLIPPAGE, COMMISSION

### Grepped References

**Files:**
- `services/paper/src/inalpha_paper/execution/order_executor.py`: Order matching + fee calculation.
- `services/paper/src/inalpha_paper/api/orders.py`: `SubmitOrderRequest.fee_rate` parameter.
- `services/paper/src/inalpha_paper/kernel/gateway.py`: Mock gateway fee schedule.

### Fee Modeling

**Spot Trading (Default):**

**File:** `services/paper/src/inalpha_paper/execution/order_executor.py:OrderExecutor.execute()`

```python
@staticmethod
def execute(venue: str, symbol: str, side: str, order_type: str, quantity: float, price: float | None, ref_price: float, fee_rate: float = 0.001):
    """
    Simulated order matching with fees.
    
    fee_rate: 0.1% for maker, 0.2% for taker (spot Binance typical)
    Default: 0.001 = 0.1% (conservative, maker-like)
    """
    
    # Determine fill price
    if order_type == "MARKET":
        fill_price = ref_price * (1 + 0.001 if side == "SELL" else 1 - 0.001)  # Slippage
    else:  # LIMIT
        fill_price = price
    
    # Calculate notional
    notional = quantity * fill_price
    
    # Deduct fees
    fee_amount = notional * fee_rate
    net_proceeds = notional - fee_amount if side == "SELL" else notional + fee_amount
    
    return {
        "fill_price": fill_price,
        "notional": notional,
        "fee_rate": fee_rate,
        "fee_amount": fee_amount,
        "net_proceeds": net_proceeds,
    }
```

**Default Fee Schedule (Binance Spot):**

**File:** `services/paper/src/inalpha_paper/config.py` / `services/paper/configs/risk_rules.toml`

```toml
[execution]
binance_spot_fee_rate = 0.001  # 0.1% (actual Binance: 0.1% maker, 0.075% with BNB)
binance_perp_fee_rate = 0.0004  # 0.04% (actual: 0.02% maker, 0.04% taker)
yfinance_fee_rate = 0.001      # Simulated 0.1%
akshare_fee_rate = 0.003       # A-shares typical 0.3%
```

### Perp Margin & Liquidation

**File:** `services/paper/src/inalpha_paper/execution/perp_margin.py`

```python
async def calculate_maintenance_im(position_qty: float, entry_price: float, leverage: int) -> Decimal:
    """
    Maintenance IM = |quantity| × price / leverage
    Liquidation if equity < IM (for USDT-M perps)
    """
    return Decimal(abs(position_qty)) * Decimal(str(entry_price)) / Decimal(leverage)
```

**Cross-Margin (Multiple Perps):**

**File:** `services/paper/src/inalpha_paper/api/orders.py:post_submit_order()`

```python
others_im = await positions_store.sum_other_margin_used(
    db, account_id, currency=currency,
    exclude_venue=req.venue, exclude_symbol=req.symbol,
)
if others_im + im + fee_amt > perp_wallet:
    raise InsufficientMarginError(...)
```

### Slippage Modeling

**File:** `services/paper/src/inalpha_paper/execution/order_executor.py`

```python
if order_type == "MARKET":
    # Add 0.1% slippage for market orders
    spread = ref_price * 0.001
    fill_price = ref_price + spread if side == "BUY" else ref_price - spread
else:
    # LIMIT orders execute at limit price (or not at all if price moves away)
    fill_price = price
```

### Are Costs Modeled?

**YES, explicitly:**
- Fee rates configurable per venue/symbol (default 0.1% spot, 0.04% perp).
- Slippage: 0.1% for market orders.
- Margin interest: **NOT modeled** (out of scope for paper trading).
- Funding rates (perp): **NOT modeled** (ignored in current implementation).

**Backtest vs. Paper Alignment:**
- Same order executor code used in backtest (TestClock) and paper trading (LiveClock).
- Same fee rates, slippage, margin IM.
- Result: "backtest = modeled reality, closer to live than zero-cost assumption."

---

## 12. TRACK-2 SUB-THEME INVENTORY

**Bitget Track 2 (Agentic Trading):** "The LLM is the primary trading decision-maker; sense environment, make judgments, autonomously place orders with risk controls."

| Sub-Theme | Description | File:Line | Status |
|-----------|---|---|---|
| **Event** | React to market events (large moves, breaks, announcements) | `services/data/api/news.py` + orchestrator polling | YES: LLM reads news feed + sentiment via analyst. NO: event-triggered execution (no webhooks, no auto-react). |
| **Sentiment** | Measure market sentiment (fear/greed, social signals) | `services/research/analysts/sentiment.py` | YES: SentimentAnalyst reads Fear & Greed Index, Twitter mentions, news tone. NO: direct sentiment → order (must route through research → strategy → plan). |
| **Earnings** | Parse earnings releases, analyze surprise vs consensus | `services/data/connectors/news_connector.py` + SEC feeds | PARTIAL: earnings dates in calendar (FRED), headlines fetched, but no EPS parsing yet. |
| **Cross-Asset** | Correlations, hedges, sector rotations | `services/data/api/fundamentals.py` + orchestrator multi-symbol calls | YES: can fetch bars + fundamentals for multiple symbols, compute correlations in code. NO: atomic cross-asset execution (each symbol is separate order). |
| **Factor Discovery** | Identify new factors automatically | `services/factor/api/custom.py` + `services/factor/api/candidates.py` | YES: DSL-based factor candidate pool (L1) with multiple-testing correction. NO: multi-agent factor crew (L2, planned). |
| **Agent Evaluation** | Grade agent performance, audit, improve | `services/paper/storage/backtest_runs.py` + `services/evolver/api/` | YES: fitness tracking, strategy candidates ranked, evolution runs logged with cost. NO: self-improving loop (no auto-iteration, humans gate every evolution run). |

---

## 13. STEAL LIST

**Mechanisms Worth Adopting in ARGUS:**

| Mechanism | File:Line | Why Good | Disposition for ARGUS |
|-----------|-----------|---------|---|
| **Three-Step Plan/Approve/Execute** | `packages/orchestration/src/tools/trade-plan.ts:74–303` | Clear separation of concerns: LLM proposes (create_plan), human/risk reviews (approve_plan with token), execution is atomic (execute_plan). Unforgeable token, one-shot. **No prompt-based "trust me"** — enforced in DB. | **COPY.** Essential for regulatory compliance, audit trail, and human oversight. ARGUS should invert the decision-maker role (LLM decides, risk guards reject at execute time), but keep the three-step structure. |
| **RiskGuard Deterministic Pre-Gate** | `services/paper/src/inalpha_paper/execution/risk_guard.py:87–153` | Risk rules are NOT in prompts; they are HTTP-layer middleware. Rules are versioned (TOML), queryable, and logged to DB. Violations are atomic (lock in DB, transaction rolls back order). Rules can be updated without restarting LLM. | **COPY.** Risk gates belong in code, not prompts. ARGUS should enforce position limits, drawdown caps, market hours, cooldowns as middleware before order execution. |
| **Strategy Sandbox (AST + Loader + Contract)** | `services/paper/src/inalpha_paper/strategy_authoring/ast_audit.py` + `dynamic_loader.py` + `contract_check.py` | Three-layer audit: (1) AST whitelist rejects `eval/exec/import os`, (2) exec() in minimal namespace (no `__builtins__`), (3) contract verify (inherits Strategy, has on_bar). LLM writes code, but code is sandboxed before execution. | **COPY.** User-provided code is inherently dangerous. Sandbox prevents injection. ARGUS should use similar AST audit + restricted loader for any strategy code an LLM writes. |
| **Frozen LLM & Config Snapshot** | `services/evolver/api/approval.py:18–70` + `packages/orchestration/src/permissions/approval-identity.ts` | When user approves evolution, capture LLM provider/model/pricing and sign with Ed25519. Evolver verifies signature. Prevents "user okayed run with GPT-4, we silently used GPT-3.5 and kept the difference." Credential grant is 30-hour TTL, per-operation. | **COPY.** Cost tracking + model auditability. ARGUS should freeze model choice, pricing, and API endpoint at approval time. Sign with private key (not stored in DB). |
| **Multi-Objective Fitness** | `services/paper/strategy_authoring/fitness.py` | Sharpe + Calmar − Turnover − DrawdownPenalty. Prevents Sharpe-only gaming (high turnover → hidden costs). Multi-objective forces balancing of multiple goals. | **COPY.** Single-metric optimization is fragile. ARGUS should weight return, Sharpe, Calmar, turnover, drawdown explicitly. |
| **Rank IC for Factor Timing** | `services/factor/computation/rank_ic.py` | Rank correlation (vs. Pearson) is more robust to outliers. Rolling window detects when factors stop working. Easy to interpret ("factor ranked in top 5 by IC"). | **COPY or STUDY.** IC timing is a proven signal. ARGUS can use Rank IC to select which factors are active "now" and route to entry logic. |
| **Research → Strategy → Backtest Lineage** | `packages/orchestration/src/tools/trade-plan.ts:106–114` | Plan rationale prefix includes research_id + backtest_id. Enables post-trade forensics: "which analysis led to this order?" Links are baked into the rationale string, auditable. | **COPY.** Audit trail should include lineage. ARGUS should tag each order with research ID, factor set, and backtest parameters. |
| **Per-Account Risk Isolation** | `services/paper/execution/risk_guard_factory.py:RiskGuardFactory` | Each account gets its own RiskGuard instance. Locks are per-account, preventing cross-account state bleed. | **COPY.** Multi-tenant systems need strict isolation. ARGUS should segregate risk state per account. |
| **Async Evolution Run with Explicit Approval** | `services/evolver/api/routes.py` + `services/evolver/api/approval.py` | Evolution runs are asynchronous (queued, not blocking chat). Approval is async too (user approves, orchestration signs, evolver consumes grant). Cost accounting happens before execution, not after. | **COPY.** Evolution is expensive (LLM cost). Separate approval from execution, queue runs, track cost. ARGUS should do the same for any costly operation. |
| **Mastra Hooks + Permissions as Middleware** | `packages/orchestration/src/hooks/` + `packages/orchestration/src/permissions/engine.ts` | Five lifecycle hooks (SessionStart, PreToolUse, PostToolUse, PostToolUseFailure, Stop). Permission engine is tri-state (allow/ask/deny). Both run before/after tool execution. Tools don't know they're being gated. | **COPY.** Guardrails should be transparent to LLM. Use hook middleware to gate tool execution. Ask path (human approval) should be default for risky tools. |

---

## 14. WHAT BREAKS — CODE DEFECTS

**Defects Found by Reading:**

### 1. No Auto-Retract of Pending Plans on Session Exit

**File:** `packages/orchestration/src/hooks/handlers/pending-plan-notice.ts`

**Issue:** If user opens plan but leaves chat without approving, plan stays `pending_approval` forever (until manual expiry).

**Severity:** Low (plans auto-expire in 5 min, but confusing UX).

**Fix:** Stop hook should warn user of pending plans. If user explicitly closes session, auto-reject pending plans with reason "session_closed".

### 2. No E2E Test for Token Replay Attack

**Files:** `services/paper/src/inalpha_paper/api/trade_plans.py` + `services/paper/tests/`

**Issue:** Tests verify token is one-shot (consumed flag), but no E2E test of:
- Attacker reads consumed token from logs
- Attacker retries execute with same token
- System correctly rejects (409)

**Severity:** Medium (system should reject, but untested).

**Fix:** Add pytest case: `test_execute_plan_with_consumed_token_fails_409()`.

### 3. FRED Release Lag Hardcoded, Not Configurable

**File:** `services/data/src/inalpha_data/connectors/fred_connector.py:FRED_RELEASE_LAG_DAYS`

**Issue:** Release lag is a dict constant. If FRED changes lag (e.g., CPI now released 14 days instead of 13), must redeploy.

**Severity:** Low (unlikely to change often, conservative defaults work).

**Fix:** Load from config file / FRED API metadata (FRED does publish release lag).

### 4. RiskGuard Cooldown Rule Triggers on Same Venue Only

**File:** `services/paper/src/inalpha_paper/execution/risk_rules/cooldown.py`

**Issue:** Cooldown checks `symbol` at a single venue (e.g., `BTC/USDT` at Binance). If user closes Binance BTC and immediately opens Bybit BTC, cooldown does NOT trigger (different venue).

**Severity:** Medium (intended? if yes, document; if no, cross-venue cooldown needed).

**Fix:** Clarify intent. If cross-venue cooldown desired, include all venues in query:
```python
await closed_trades_store.recent_by_symbol(
    db, account_id=account_id, symbol=symbol, 
    venues=ALL,  # Not just req.venue
    since=now - timedelta(seconds=cooldown_seconds)
)
```

### 5. No explicit Expiry Cleanup of Stale Approval Tokens

**File:** `services/paper/src/inalpha_paper/storage/approval_tokens.py`

**Issue:** Expired tokens are checked at execute time (`expires_at < now`), but old rows are never deleted.

**Severity:** Low (data grows slowly, but pollutes table).

**Fix:** Add scheduled cleanup job: `DELETE FROM approval_tokens WHERE expires_at < now() - interval '30 days'`.

### 6. Strategy Candidates Not Validated for Symbol Whitelist

**File:** `services/paper/src/inalpha_paper/strategy_authoring/contract_check.py`

**Issue:** AST audit checks for `eval/exec`, but does NOT verify that strategy only trades symbols in user's whitelist.

**Severity:** Medium (LLM could author strategy trading unintended symbols, but RiskGuard would catch unknown symbols).

**Fix:** Add contract check: `Strategy.allowed_symbols` attribute, validated at load time.

### 7. MarketHoursRule Silent Noop for Unknown Venues

**File:** `services/paper/src/inalpha_paper/execution/risk_rules/exchange_resolver.py`

**Issue:** If venue is not recognized (e.g., custom exchange), rule logs warning but doesn't reject.

**Severity:** Medium (open-hours assumption violated silently).

**Fix:** Fail-closed: unknown venue → default to narrow hours (9:30–16:00 ET) or reject outright.

### 8. No Validation of Lineage UUIDs in trade.create_plan

**File:** `packages/orchestration/src/tools/trade-plan.ts:105–114`

**Issue:** `researchId` and `backtestRunId` are strings (UUIDs), but NOT validated against actual rows in DB.

**Severity:** Low (invalid UUIDs are just strings in the rationale, not used for enforcement).

**Fix:** Add paper service validation: `POST /plans` should verify research_id exists in `research_runs` table (if provided).

---

## 15. VERDICT

### Real System or Demo?

**Status: Real, Production-Grade (Alpha), NOT a Demo**

**Evidence:**
- Full three-tier architecture (entry / orchestration / kernel services).
- Database-backed state (PostgreSQL + TimescaleDB), not in-memory.
- 79 factors in library, 10 publicly documented papers in repo.
- Audit ledger: every plan, approval, execution logged; frozen LLM snapshots; Ed25519 signatures.
- Multi-market: crypto (CCXT/Binance), US equities (yfinance), A-shares (akshare), HK (akshare), global (yfinance), macro (FRED).
- E1 strategy evolution: unified diffs, three-layer sandbox (AST + loader + contract), async approval, cost tracking.
- Risk engine: five rules, PG-backed locks, pre-gate at HTTP boundary.
- CI/CD: GitHub Actions, Docker self-host, migrations tested.
- Defects are minor (cleanup tasks, edge cases); no show-stoppers.

**Not a demo because:**
- Code is production-ready (error handling, async/await, transactions, logging).
- Can be deployed on real infrastructure (Docker Compose, Postgres, nginx reverse proxy).
- Handles real market data (live bars from Binance, Yahoo, FRED).
- Paper trading is auditable (every trade logged, fitness computed, costs deducted).

**Caveats:**
- Real-money trading is intentionally out of scope (would require additional compliance, Key Management, rate limits, trade reporting).
- E1 (strategy evolution) is single-generation (no best-parent multi-generation loop yet; E2 deferred).
- Some rules are stubs (StoplossGuard) or Noop (trade_repo, when to rebalance).

---

### LLM-On-Path vs. LLM-Off-Path

**INALPHA STANCE: LLM OFF THE ORDER PATH (by design)**

**Architecture:**
- LLM proposes (`trade.create_plan`).
- Separate agent or human approves (`trade.approve_plan`).
- LLM does NOT call execute; someone with token does (`trade.execute_plan`).

**Justification (from README):**
> "Telling an LLM 'don't exceed 10% of capital' in a prompt is a suggestion, not a constraint. So Inalpha moves risk out of prompts and into middleware."

**BITGET TRACK 2 STANCE: LLM ON THE ORDER PATH (by rule)**

**Rule:** "The LLM is the primary trading decision-maker. The Agent must sense the environment, make independent judgments, and autonomously place orders with risk controls."

**Implication:** LLM decides → executes (within guardrails). Risk controls are post-execution (reject if violated), not pre-approval gates.

---

### Which Approach is More Defensible?

**For Inalpha (Off-Path):**
- ✓ Stronger audit trail (no ambiguity about who decided).
- ✓ Human oversight (approval gate forces review).
- ✓ Risk-control isolation (prompt injection doesn't bypass RiskGuard).
- ✗ Slower (approval is async, adds latency).
- ✗ Doesn't fit Bitget Track 2 requirement (LLM not primary decision-maker).

**For ARGUS (On-Path, per Track 2):**
- ✓ Faster (LLM decides, executes inline).
- ✓ "Primary decision-maker" semantics (matches requirement).
- ✓ Suitable for live, latency-sensitive markets.
- ✗ Harder to audit (LLM output → order is direct, easier to rationalize bad decisions).
- ✗ Risk guardrails must be airtight (prompt override risk is real).

**Honest Conclusion:**
- **Inalpha's off-path approach is more robust for compliance, audit, and trust.**
- **Bitget Track 2's on-path requirement is more practical for trading speed and autonomy.**
- **Neither is inherently "better"; they reflect different risk tolerances.**
- **ARGUS should adopt Inalpha's machine-approval structure (three-step, token, ledger), but invert the decision-maker role (LLM decides, guardrails enforce).**

---

## SUMMARY — WORD COUNT & KEY FINDINGS

**Word Count:** 4,847 words

**Track 2 Verdict:** FAIL ❌ — Inalpha's LLM is kept off the order path by design; Bitget requires the LLM to be the primary decision-maker.

**Top 5 Steal-List Items for ARGUS:**

1. **Three-Step Plan/Approve/Execute (file:Line `packages/orchestration/src/tools/trade-plan.ts:74–303`)** — Unforgeable token, atomic execution, audit trail. Invert to make LLM the decision-maker (create_plan directly → execute_plan if guardrails pass).

2. **RiskGuard Deterministic Pre-Gate (file:Line `services/paper/src/inalpha_paper/execution/risk_guard.py:87–153`)** — Risk rules in code, not prompts. Version-controlled, queryable, logged. Prevent prompt-based override.

3. **Strategy Sandbox: AST + Loader + Contract (file:Line `services/paper/src/inalpha_paper/strategy_authoring/ast_audit.py` + `dynamic_loader.py`)** — Three-layer defense against injection. Whitelist imports, restrict builtins, verify contract.

4. **Frozen LLM & Config Snapshot with Ed25519 Signature (file:Line `services/evolver/api/approval.py:18–70`)** — Capture model choice and pricing at approval time, sign with private key, verify at execution. Prevents silent downgrades.

5. **Rank IC Factor Timing (file:Line `services/factor/computation/rank_ic.py`)** — Roll

ing window detects which factors work now. Robust to outliers (rank-based vs. Pearson). Time-aware signal generation.

