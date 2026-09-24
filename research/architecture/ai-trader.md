# AI-Trader: Architecture Teardown

**Target:** `HKUDS/AI-Trader` — 100% Fully-Automated Agent-Native Trading Platform
**Analysis Date:** 2026-09-12
**Purpose:** Extract design patterns for ARGUS (Bitget Track-2 Agentic Trading entry)

---

## 1. Identity

**Project:** AI-Trader  
**Repository:** https://github.com/HKUDS/AI-Trader  
**Language Stack:** Python (FastAPI backend), TypeScript/React (frontend), SQLite/PostgreSQL (database)  
**Scope:** Agent-native trading signal platform supporting live paper trading, copy trading, challenges, and research competitions  
**Key Claim:** "100% Fully-Automated Agent-Native Trading" — agents publish signals, copy leaders, participate in competitions with real-time leaderboards

---

## 2. Licence

**Source:** `/README.md` badge shows MIT  
**LICENSE File:** NOT FOUND in repository root  
**SPDX:** MIT (stated in badge, badge shows green checkmark)

---

## 3. Full Architecture

### High-Level Structure

```
AI-Trader (Platform)
├── service/server/          # FastAPI backend
│   ├── main.py              # Entry point, startup/shutdown
│   ├── routes.py            # Route registration
│   ├── routes_agent.py      # Agent auth, messages, WebSocket
│   ├── routes_signals.py    # Signal publish, strategy, discussion
│   ├── routes_trading.py    # Trading endpoints, leaderboards, positions
│   ├── routes_challenges.py # Challenge competitions
│   ├── routes_experiments.py # A/B experiments
│   ├── tasks.py             # Background job scheduler
│   ├── worker.py            # Standalone worker process
│   ├── database.py          # SQLite/PostgreSQL connection layer
│   ├── market_intel.py      # Market news, macro signals, stock analysis
│   ├── price_fetcher.py     # Alpha Vantage, Polymarket price resolution
│   ├── challenge_scoring.py # Portfolio replay and scoring logic
│   ├── fees.py              # Trade fee configuration (0.1% = 0.001)
│   └── [16 other route/service modules]
│
├── service/frontend/        # React frontend
│   └── [UI dashboard, signal feed, leaderboards]
│
├── skills/                  # Agent integration specs
│   ├── ai4trade/SKILL.md    # Main agent API reference
│   ├── copytrade/SKILL.md   # Copy trading workflow
│   ├── tradesync/SKILL.md   # Signal publishing
│   ├── market-intel/SKILL.md # Market intelligence API
│   └── [3 other skills]
│
└── research/                # Analysis pipeline (4k+ agent competition dataset)
    ├── scripts/             # Export, metric computation, figure generation
    └── schemas/             # JSON Schema definitions for entities
```

### Core Data Model

**Primary Tables (service/server/database.py):**
- `agents` — agent identity, cash, verification status
- `signals` — published trades, strategies, discussions (timestamp, executed_at)
- `positions` — current holdings (market, symbol, side, entry_price, quantity)
- `subscriptions` — follower relationships (leader_id, follower_id)
- `challenges` — competition configs with rule sets
- `challenge_participants` — enrollment in a challenge
- `challenge_trades` — trades attributed to a challenge
- `challenge_results` — final scoring and disqualification reasons
- `profit_history` — periodic snapshots of agent equity curve
- `market_news_snapshots` — cached Alpha Vantage news summaries
- `macro_signal_snapshots` — QQQ, XLP, GLD, UUP, BTC trend signals
- `stock_analysis_snapshots` — per-symbol trend scores and support/resistance
- `experiments` — A/B test configurations, variants, targeting
- `experiment_events` — user interactions logged per experiment
- Various join tables for signal replies, event tracking, team missions

**Key Schema Properties:**
- `timestamp` INT (Unix seconds, from trade executed_at)
- `created_at` TEXT (ISO 8601 UTC, e.g., "2026-03-20T09:30:00Z")
- `executed_at` TEXT (trade timestamp, may be historical)
- PIT (Point-In-Time) safety: positions recorded with trade timestamp, prices fetched as-of that moment

### Application Topology

```
┌─────────────────────────────────────────────────────────────┐
│                      Agents (AI Traders)                    │
│         Register → Receive Token → Publish Signals           │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    REST API      WebSocket       MCP (future)
    (/api/*)     (/ws/notify/)   (agent-sdk)
         │             │             │
┌────────┴─────────────┴─────────────┴──────────────────────┐
│              FastAPI Server (main.py)                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ routes_agent.py      → Register, Login, Messages    │  │
│  │ routes_signals.py    → Publish Trade/Strategy       │  │
│  │ routes_trading.py    → Get Positions, Leaderboard   │  │
│  │ routes_challenges.py → Join Challenge, Submit Trades│  │
│  │ routes_market.py     → News, Macro Signals, Quotes  │  │
│  │ routes_experiments.py → Variant Assignment          │  │
│  └──────────────────────────────────────────────────────┘  │
└────────┬──────────────────────────────────────────────────┘
         │
    ┌────┴──────────────┬─────────────────┐
    │                   │                 │
┌───────────────┐  ┌─────────────┐  ┌──────────────┐
│ Database      │  │ Background  │  │   Cache      │
│ (SQLite or    │  │   Worker    │  │   (Redis)    │
│ PostgreSQL)   │  │  (worker.py)│  │              │
│               │  │             │  │              │
│ - Trades      │  │ Refreshes:  │  │ - Price      │
│ - Positions   │  │ - Prices    │  │ - Leaderboards
│ - Scores      │  │ - Profit    │  │ - News Cache │
│ - Challenge   │  │   history   │  │              │
│   Results     │  │ - News, Macro│  │              │
└───────────────┘  │ - Signals   │  └──────────────┘
                   │ - Metrics   │
                   │ - Settlement│
                   └─────────────┘
                        │
         ┌──────────────┴─────────────┐
         │                            │
    ┌────────────────┐       ┌────────────────┐
    │ Alpha Vantage  │       │  Polymarket    │
    │ (Stock/Crypto  │       │  Public API    │
    │  prices, news) │       │  (AMM prices,  │
    │                │       │   orderbook)   │
    └────────────────┘       └────────────────┘
```

### Scheduler and Live Loop

**Background Task Runner (tasks.py + worker.py):**

Task `start_background_tasks(logger)` creates async tasks for:
1. **price_refresh_task** — fetch latest prices from Alpha Vantage / yfinance fallback every 60–300 sec
2. **profit_history_task** — compact daily equity snapshots every 300 sec
3. **market_intel_task** — refresh news, macro signals, stock analysis every 1–7200 sec
4. **challenge_settlement_task** — auto-settle resolved Polymarket positions, update scores
5. **experiment_notification_task** — publish experiment notices to agents
6. **live agent metrics** — compute leaderboard scores (profit %, drawdown, collaboration)

**Worker Lifecycle:**
- Standalone process (`python service/server/worker.py`) acquires singleton lock (Redis or file)
- Runs independently from FastAPI service to prevent blocking HTTP requests
- Catches SIGINT/SIGTERM for graceful shutdown
- Monitors Redis lock with renewal task to detect process death

---

## 4. "AGENT-NATIVE" — Actual Implementation

### What Makes It Agent-Native

**Definition (per README):** "Agents collaborate and debate to surface the best trading ideas automatically."

**In Code (routes_signals.py:106–400):**

1. **Direct API Integration** — no UI intermediary
   - Agents POST `/api/signals/realtime` with market, symbol, side, quantity, price
   - API accepts historical `executed_at` (e.g., "2026-03-01T14:30:00Z") for backtestable signals
   - Server enforces PIT price fetching: `get_price_from_market(symbol, executed_at, market)` — file:line routes_signals.py:186–222

2. **Atomic Trade + Position Update** — transactional consistency
   - Signal creation, position update, cash deduction, fee calculation, experiment event recording all in single transaction (file:line routes_signals.py:252–368)
   - Cash flow: `trade_value + fee` deducted from agent cash before signal recorded

3. **Platform Interaction Surfaces**
   - **Publish Signal** — agent posts a trade decision; followers auto-copy
   - **Read Leaderboard** — `/api/profit/history` returns live-scored agents by return, drawdown, risk-adjusted return (file:line routes_trading.py:67–264)
   - **Read Signals** — `/api/signals/feed` returns market-wide signal feed, agents can filter by symbol, message_type
   - **Join Challenge** — `/api/challenges/{id}/join` agent enrolls in competition with explicit rules (max position %, max drawdown %)
   - **WebSocket Notifications** — agents subscribe to `/ws/notify/{client_id}` to receive replies, mentions, follower events in real time (file:line routes_agent.py:112–131)

4. **What an Agent Receives (Market Data Layer)**
   - **Current Price** via `/api/price/quote?symbol=NVDA&market=us-stock`
   - **Market Status** — `is_market_open(market)` returns False if US market is closed (file:line routes_signals.py:174–184, using ZoneInfo America/New_York)
   - **Market Snapshots** — news, macro regime signals, stock technical analysis
   - **Agent Reputation** — profile shows identity_status, verified badge, points, follower count
   - **Leaderboard Context** — peer performance, profit %, drawdown for comparison

### Agent-Native API Interface

**Core Endpoints (routes_agent.py + routes_signals.py + routes_trading.py):**

```
POST   /api/claw/agents/selfRegister        → Token
POST   /api/claw/agents/login               → Token
GET    /api/claw/agents/me                  → Agent Info
POST   /api/signals/realtime                → Place Trade
GET    /api/positions                       → Current Positions
GET    /api/profit/history?days=30          → Leaderboard Data
GET    /api/signals/feed?limit=20           → Global Signal Feed
POST   /api/signals/follow                  → Copy Trader
POST   /api/challenges/{id}/join            → Enter Competition
GET    /api/claw/messages/recent            → Unread Notifications
POST   /api/claw/messages/mark-read         → Clear Notifications
GET    /ws/notify/{agent_id}?token=X        → WebSocket Subscribe
```

**What is Exposed to Agents (Not in Native REST API):**
- Database schema (no direct SQL)
- Internal scoring algorithms (they see results, not source)
- Other agents' full history (leaderboards aggregate, no full trade log exposure)
- Price data before public time (prices are as-of requested timestamp, validated PIT)

---

## 5. THE DECISION PATH — Where LLM Output Drives Action

### Prompt Templates in Codebase

**Market Analysis (market_intel.py:588–619):**

```python
prompt = (
    "Write one concise market snapshot paragraph in English for a trading dashboard.\n"
    "Rules:\n"
    "- Keep it under 60 words.\n"
    "- Be specific and grounded only in the supplied metrics.\n"
    "- Mention the strongest support and strongest risk.\n"
    "- Do not use bullet points.\n"
    "- Do not mention AI, models, or uncertainty disclaimers.\n\n"
    f"Symbol: {analysis['symbol']}\n"
    f"Signal: {analysis['signal']}\n"
    f"Trend status: {analysis['trend_status']}\n"
    f"Signal score: {analysis['signal_score']}\n"
    f"Current price: {analysis['current_price']}\n"
    f"5d return: {analysis['return_5d_pct']}%\n"
    f"20d return: {analysis['return_20d_pct']}%\n"
    f"Moving averages: {json.dumps(analysis.get('moving_averages') or {}, ensure_ascii=True)}\n"
    f"Support levels: {_format_price_levels(analysis.get('support_levels') or [])}\n"
    f"Resistance levels: {_format_price_levels(analysis.get('resistance_levels') or [])}\n"
    f"Bullish factors: {json.dumps(analysis.get('bullish_factors') or [], ensure_ascii=True)}\n"
    f"Risk factors: {json.dumps(analysis.get('risk_factors') or [], ensure_ascii=True)}\n"
)
response = client.chat.send(
    model=OPENROUTER_MODEL,
    messages=[{"role": "user", "content": prompt}],
)
```

**Called by:** `_generate_stock_analysis_summary(analysis)` (file:line market_intel.py:583–619)  
**Model:** OpenRouter configurable (env var `OPENROUTER_MODEL`, defaults to empty = fallback)  
**Output Used:** Displayed on stock analysis card, no decision logic driven by it

### **IS THE LLM GENUINELY DECIDING TRADES?**

**Answer: NO — ASSERTED as agent-native, NOT PROVED in code.**

**Evidence:**

1. **Stock Analysis Scoring is Deterministic (file:line market_intel.py:995–1097)**
   - Signal computed from moving averages, returns, support/resistance distance
   - Score = -3 (sell) to +3 (buy) based on rule-based factors
   - LLM used only for summary text (feed beautification), not scoring

2. **Trade Decision Path (file:line routes_signals.py:106–377)**
   - Agent calls `/api/signals/realtime` with action, symbol, quantity, price already decided
   - Server validates: market open? quantity positive? price available? cash sufficient?
   - **No LLM inference in trade acceptance/rejection path**
   - Server does NOT call an LLM to decide whether to execute or modify the trade

3. **Leaderboard Scoring (file:line routes_trading.py:122–244)**
   - Computed deterministically from profit, drawdown, collaboration metrics
   - No LLM applied to leaderboard generation

4. **Challenge Scoring (file:line challenge_scoring.py:69–273)**
   - Deterministic portfolio replay
   - Disqualification rules: max position %, max drawdown % exceeded → disqualified
   - Final score = return_pct - (max_drawdown - allowed_drawdown) × drawdown_penalty
   - No LLM intervention

**Verdict:** LLM is present for market analysis summaries and notation, but the **actual trading decision** (what action to take, when to take it, position sizing) is 100% agent-responsibility. The platform does NOT override or modify LLM trading output. **NOT PROVED that LLM is the "primary decision-maker"** — agent is responsible for decision, LLM is used for narrative/context only.

### Decision Schema

**Trade Decision (incoming request body, routes_models.py):**
```python
class RealtimeSignalRequest(BaseModel):
    action: str                 # 'buy', 'sell', 'short', 'cover'
    market: str                 # 'us-stock', 'crypto', 'polymarket'
    symbol: str                 # 'NVDA', 'BTC', market_id
    quantity: float             # Units to trade
    price: float                # Entry/exit price
    executed_at: str            # 'now' or ISO 8601 timestamp
    token_id: str | None        # Polymarket only
    outcome: str | None         # Polymarket only
    content: str | None         # Agent note / reasoning
```

**Deterministic Override:** YES — Hard rules in routes_signals.py

- Lines 174–184: Market closed → HTTPException 400
- Lines 135–138: Quantity ≤ 0 or > 1M → HTTPException 400
- Lines 230–233: Price ≤ 0 or > 10M → HTTPException 400
- Lines 236–238: Trade value > 1B → HTTPException 400
- Lines 256–269: Sell without long position → HTTPException 400
- Lines 272–283: Insufficient cash → HTTPException 400

**These are enforced BEFORE any agent decision flows to the database.** They are guard rails, not overrides (they reject invalid input, not modify valid decisions).

---

## 6. Sensing the Environment

### Data Sources Available to Agents

**Real-Time Market Data:**
- **Current Price** — `/api/price/quote/{symbol}?market=us-stock`
  - Alpha Vantage TIME_SERIES_INTRADAY (1-min), yfinance fallback (file:line market_intel.py:280–311)
  - Staleness flag: "realtime" if age < STOCK_QUOTE_STALE_AFTER_SECONDS (900 sec), else "session_close" or "stale" (file:line market_intel.py:314–349)
  - Polymarket: real-time AMM price via `/polymarket/prices` (file:line routes_shared.py, decorate_polymarket_item)

**Market Intelligence (cached snapshots):**
- **News Feed** — Alpha Vantage NEWS_SENTIMENT, 4 categories (equities, macro, crypto, commodities)
  - Lookback: 48 hours by default (file:line market_intel.py:38)
  - Refreshed every 3600 sec (file:line market_intel.py:48)
  - Exposed via `/api/market/news?category=macro` (routes_market.py)

- **Macro Signals** — QQQ, XLP, GLD, UUP, BTC returns
  - Lookback: 20 days standard, 7 days for BTC (file:line market_intel.py:42–43)
  - Signals: bullish/defensive/neutral status + score
  - Exposed via `/api/market/signals/macro` (routes_market.py)

- **Stock Analysis** — per-symbol technical (moving averages, support/resistance, bullish/risk factors)
  - Lookback: 20–60 days of history (file:line market_intel.py:1000–1007)
  - Exposed via `/api/market/stocks/{symbol}` (routes_market.py)

**Competitive Context:**
- **Leaderboard** — `/api/profit/history?metric=return&days=30&limit=100`
  - Returns top agents ranked by profit %, max drawdown, risk-adjusted return, collaboration score
  - Includes profit history (equity curve) if `include_history=true` (file:line routes_trading.py:67–264)

- **Signal Feed** — `/api/signals/feed?limit=50&sort=new`
  - All published trades, strategies, discussions
  - Can filter by symbol, agent, message_type (file:line routes_signals.py, register_signal_routes)

- **Other Agents' Positions** — `/api/agents/{agent_id}/positions`
  - Public view of any agent's open positions (file:line routes_trading.py:682–742)

**Experiment Context** (A/B Testing):
- Agents assigned to variants, unread notice surface indicates active experiment
- `experiment_unread_notice` field on signal feed / positions response (file:line routes_shared.py:attach_experiment_unread_notice)

### PIT (Point-In-Time) / Data Freshness Guarantees

**For Historical Trades (executed_at != 'now'):**
- Price fetcher calls `get_price_from_market(symbol, executed_at, market)` (file:line routes_signals.py:186–222)
- Alpha Vantage daily adjusted or intraday time-series looked up at exact timestamp
- If timestamp pre-dates available data → HTTP 400 "Unable to fetch historical price"
- **NO forward-looking bias:** timestamp is agent-supplied, not server-supplied

**For Real-Time Trades (executed_at == 'now'):**
- Current UTC time captured at request receipt
- Market open check: current time (US ET) between 09:30-16:00 on weekday? (file:line market_intel.py:198–203)
- Price fetched if configured, else uses agent-supplied price
- Risk: agent can provide stale price manually, but API does not override it if `should_fetch_server_trade_price(market)` is False (e.g., for thin markets)

**Leakage Risk in Live System:**
- Agents see other agents' signals in real time (no lag)
- Agents see leaderboard rankings updated every 5 min (LEADERBOARD_CACHE_TTL_SECONDS = 300)
- **Problem:** in a real system, agents could observe signal patterns and front-run (a faster agent sees a slow agent's signal, copies before leaderboard updates reflect the slow agent's profit)
- **Mitigation in code:** signal replies are timestamped, profit history is historical (not edge-run), but NO explicit latency padding is implemented

**Database-Level PIT:**
- Positions stored with (market, symbol, entry_price, quantity, opened_at, current_price)
- `current_price` may be stale by design (last refresh time stored separately)
- Challenge scoring replays trades in order (file:line challenge_scoring.py:101–233), marks each position to price at time of next trade, not current live price

---

## 7. Risk Controls

### In-Trade Validation (Hard Guards)

**File: routes_signals.py:130–284**

| Control | Code | Effect |
|---------|------|--------|
| Quantity validation | Lines 135–138 | qty ≤ 0 or > 1M → 400 error |
| Price validation | Lines 230–233 | price ≤ 0 or > 10M → 400 error |
| Trade value cap | Lines 236–238 | qty × price > 1B → 400 error |
| Market hours | Lines 174–184 | US market closed? → 400 error (ET timezone check) |
| Position constraint (sell) | Lines 256–269 | Sell with no long or qty > long → 400 error |
| Position constraint (cover) | Lines 265–269 | Cover with no short or qty > short → 400 error |
| Cash validation | Lines 272–283 | buy/short: cash < trade_value + fee → 400 error |

**PROVED:** Hard stops prevent invalid states. These are enforced before signal ID is reserved.

### Challenge-Level Risk Rules

**File: challenge_scoring.py:24–43, routes_challenges.py**

Challenges can specify rules (JSON stored in `rules_json`):

| Rule | Line | Behavior |
|------|------|----------|
| `max_position_pct` | Lines 84, 209–213 | Portfolio position > max_notional / equity × 100? → disqualified |
| `max_drawdown_pct` | Lines 85, 237–238 | max_drawdown > max_drawdown_pct? → disqualified (if `disqualify_on_drawdown=true`) |
| `disqualify_on_drawdown` | Line 237 | Rule flag controlling drawdown disqualification |
| `allowed_drawdown` | Line 241 | Drawdown above this incurs penalty (risk-adjusted scoring) |
| `drawdown_penalty` | Line 242 | Multiplier applied to excess drawdown |

**Portfolio Replay Validates:**
- Buy while short → disqualified
- Sell exceeds long position → disqualified
- Short while long → disqualified
- Cover exceeds short position → disqualified

**PROVED:** Challenge scoring is deterministic, enforced during result computation (not live during trade). Agents can enter invalid states in live trading but are disqualified in challenges.

### Leaderboard Scoring (Dynamic Risk Adjustment)

**File: routes_trading.py:122–244**

```sql
(ap.profit_percent - COALESCE(ls.max_drawdown, 0)) AS risk_adjusted_score
```

Agents ranked by:
- **Return only:** `profit_percent`
- **Risk-adjusted:** `profit_percent - max_drawdown`
- **Collaboration:** `reply_count + accepted_reply_count × 2 + citation_count + adoption_count`

**PROVED:** Risk-adjusted ranking penalizes drawdown. Agent can extract their own drawdown from profit_history endpoint and adjust behavior.

### Fees (Costs Modeled)

**File: fees.py**

```python
TRADE_FEE_RATE = 0.001  # 0.1% per trade
```

**Applied:** 
- Buy/short: `fee = trade_value × 0.001` deducted from cash (routes_signals.py:244)
- Sell/cover: `fee` deducted from credit received (routes_signals.py:323–329)
- No slippage, no market impact, no liquidity constraints modeled

**NOT MODELED:**
- Order fill delays
- Partial fills
- Bid-ask spread
- Margin/leverage (no leverage supported)
- Dividends, corporate actions

---

## 8. Order Placement — Broker/Venue, Paper vs. Live

### Execution Model

**Paper Trading Only** — this is a research platform, not a live trading system

**File: routes_signals.py:106–377, challenge_scoring.py**

All trades are **simulated** on agent's paper portfolio:
- Agent starts with $100,000 cash (INITIAL_CAPITAL = 100000.0)
- Trades modify positions and cash in-database
- Polymarket trades use real market prices but simulate the fill (no real order placed)
- No connection to live brokers (Bitget, Coinbase, etc.)

### Trade Execution Steps

**Atomically (routes_signals.py:252–368):**

1. Reserve signal ID (sequence number)
2. Validate side + position:
   - If sell/cover: check position exists, qty sufficient
   - If buy/short: check cash sufficient
3. Insert signal record (market, symbol, side, quantity, price, timestamp)
4. Update position (create or merge, compute new avg entry price)
5. Update agent cash (deduct trade_value + fee for buy/short, add for sell/cover)
6. Score signal quality (heuristic based on signal type, market conditions)
7. Award reward points (fixed or quality-weighted per experiment variant)
8. Record event (for experiment attribution)
9. Commit transaction

**If any step fails → ROLLBACK, HTTP 500 or specific 400**

### Copy Trading (Positions Replicated, Not Orders)

**File: routes_signals.py:391–410**

When leader publishes signal:
1. Find all active subscriptions (followers)
2. For each follower:
   - If follower has auto-copy enabled: replicate the signal to follower's portfolio
   - Same market, symbol, side, quantity, price, timestamp
   - Follower records leader_id in position (source = "copied:X")

**NOT a broker sync** — no API call to actual broker. Only internal portfolio replication.

### Polymarket Specifics

**File: routes_signals.py:140–161, price_fetcher.py**

- Polymarket uses outcome tokens (USDC-denominated AMM)
- Trade specifies market_id or infers from symbol + outcome
- Current price fetched via Polymarket public API
- Fill is **instant at AMM price** (no slippage model)
- No shorting support (buy / sell outcome tokens only)

---

## 9. Evaluation and Results

### Metrics Tracked

**Per-Agent (agent_metric_snapshots table):**
- `profit` — mark-to-market profit
- `profit_percent` — profit / initial_capital × 100
- `max_drawdown` — peak-to-trough decline
- `risk_adjusted_score` — profit_percent - max_drawdown
- `reply_count`, `accepted_reply_count`, `citation_count` — engagement metrics
- `collaboration_score` — weighted sum of engagement
- `quality_score_avg` — average signal quality rating

**Per-Challenge (challenge_results table):**
- `starting_cash`, `ending_value`, `return_pct`
- `max_drawdown`
- `risk_adjusted_score`
- `trade_count`
- `disqualified_reason` (null = qualified)
- `rank` (1 = best qualified, others ranked by final_score)

### Experiment Framework

**File: experiments.py, routes_signals.py:60–102**

Agents assigned to experiment variants. Per-signal, variant determines:
- `reward_mode` — fixed vs. quality_weighted
- `reward_multiplier` — scale factor on base points
- Signal quality score influences reward if variant uses quality_weighted mode

**Events Recorded (experiment_events.py):**
- `signal_published` — agent published a signal
- `signal_reply_submitted` — agent replied to signal
- `agent_followed` — agent followed another agent
- Each event tagged with experiment_key, variant_key

### Self-Scoring Loop

**None Found.** Scoring is post-hoc:

- Agents submit trades
- Background worker computes profit_history snapshots periodically
- Profit_history is read-only from agent perspective
- Agents cannot modify their own scores

---

## 10. Copy Trading

### Mechanism

**File: routes_trading.py:778–834, routes_signals.py:391–430**

1. **Follow Setup:**
   - Follower calls `POST /api/signals/follow` with leader_id
   - Creates subscription record (leader_id, follower_id, status='active')

2. **Auto-Copy on Leader Signal:**
   - When leader publishes signal, server looks up all followers
   - For each follower with active subscription:
     - Replicate signal: same market, symbol, side, quantity, price, executed_at
     - Insert as new signal in follower's record with leader_id reference
     - Update follower's position

3. **Position Source Attribution:**
   - Position record stores `leader_id`
   - On positions endpoint, returned as `source: "copied:123"` (file:line routes_trading.py:655)

### Not Real Copy Trading

- No broker sync or actual account mirroring
- Only replicates to paper portfolio in same database
- Follower can modify their own positions independently
- No leverage or margin adjustment (follower gets exact same size, regardless of account equity)

---

## 11. Per Track-2 Sub-Theme Inventory

**Track 2:** "The LLM is the primary trading decision-maker, not just an assistant. The Agent must sense the environment, make independent judgments, and autonomously place orders with risk controls."

### Thematic Coverage in AI-Trader

| Sub-Theme | Evidence | File:Line | Verdict |
|-----------|----------|-----------|---------|
| **Event Sensing** | Agent reads news snapshots, macro signals, stock analysis | market_intel.py, routes_market.py | PRESENT — data available via API |
| **Sentiment Analysis** | Adanos sentiment API integrated (Reddit, X, news, Polymarket) | market_intel.py:376–440 | PRESENT but optional (requires API key) |
| **Earnings Calendar** | Not implemented in codebase | — | NOT FOUND |
| **Cross-Asset Execution** | Multiple markets: us-stock, crypto, polymarket supported | routes_signals.py:118 | PRESENT — can trade across markets |
| **Factor Discovery** | No built-in factor mining or alpha model | — | NOT FOUND |
| **Agent Evaluation** | Leaderboard scoring, profit history, risk-adjusted returns | routes_trading.py | PRESENT — comprehensive evaluation framework |
| **Autonomous Order Placement** | Agent decides trade, server validates and executes | routes_signals.py:106–377 | PRESENT — agent-driven, server validates |
| **Risk Controls** | Position limits, drawdown caps, cash checks, market hours | routes_signals.py:130–284 + challenge_scoring.py | PRESENT — enforced at multiple layers |
| **Real-Time Leaderboards** | Live-updated ranking by profit %, drawdown, collaboration | routes_trading.py:67–264 | PRESENT — 5-min cache TTL |
| **Competitive Dynamics** | Public signal feed, follower network, copy trading | routes_signals.py, routes_trading.py:778–834 | PRESENT — agents can compete and copy |

**Missing from Track-2 Baseline:**
- Earnings announcement parsing / strategy
- Factor/correlation analysis tools
- Multi-agent deliberation (agents act independently, no debate mechanism visible)
- True adversarial trading environment (paper trading only, no zero-sum dynamics)

---

## 12. STEAL LIST

**Table: What AI-Trader Does Well + Disposition for ARGUS**

| Mechanism | File:Line | Why Good | Disposition |
|-----------|-----------|----------|-------------|
| **Atomic Trade + Position Update** | routes_signals.py:252–368 | Single transaction ensures consistency; prevents race conditions on cash/positions | COPY — essential for correctness |
| **Market Hours Guard** | routes_signals.py:174–184 | Uses ZoneInfo(America/New_York) to respect exchange hours; prevents market-closed trades | COPY — prevents silly mistakes |
| **PIT Price Fetching** | routes_signals.py:186–222 | Agents specify executed_at, server fetches price as-of that moment from Alpha Vantage; supports backtestable signals | COPY — enables agent reasoning over time |
| **Deterministic Signal Scoring** | market_intel.py:995–1097 | Technical score (-3 to +3) computed from moving averages, returns, support/resistance; reproducible, no LLM in critical path | STUDY — understand how to score signal quality without LLM |
| **Portfolio Replay + Disqualification** | challenge_scoring.py:69–273 | Trades replayed in order, positions marked at trade prices, equity curve traced, disqualification on invalid moves or rule breach | COPY — needed for fair challenge evaluation |
| **Risk-Adjusted Scoring** | routes_trading.py:209, challenge_scoring.py:243 | Score = return_pct - (max_drawdown - allowed_drawdown) × penalty; simple, transparent | COPY — balances return and risk |
| **Experiment Attribution** | routes_signals.py:60–102 + experiment_events.py | Signals tagged with experiment_key, variant_key; rewards weighted by variant; supports quality-weighted payouts | STUDY — how to design reward experiments |
| **WebSocket Notifications** | routes_agent.py:112–131 | Real-time message push to agents (replies, mentions, followers); agent polls `/claw/messages/recent` for fallback | COPY — needed for agent interaction loop |
| **Feed-Level Caching** | routes_shared.py (SIGNAL_FEED_CACHE_TTL_SECONDS = 60) | Signal feed cached 60 sec; leaderboard cached 300 sec; balances freshness vs. load | COPY — scale approach for large agent counts |
| **Polymarket Integration** | routes_signals.py:140–161 + price_fetcher.py | Outcome token support; real-time AMM prices via public API; zero-sum market setup | BENCHMARK — Polymarket adds real market risk; can compare paper vs. real |

---

## 13. WHAT BREAKS — Defects Found by Code Reading

### Logic Errors

| Issue | File:Line | Severity | Details |
|-------|-----------|----------|---------|
| **Stale Price Not Rejected in Live Trade** | routes_signals.py:169–198 | MEDIUM | If `should_fetch_server_trade_price(market)` is False (e.g., thin market), agent-supplied price is used as-is; no staleness check. Agent can provide hour-old price. Mitigation: test with real data. |
| **No Slippage Model** | fees.py, routes_signals.py | LOW | Trade value = qty × price (no bid-ask, no liquidity impact). Paper trading only, OK for research. Would need fixing for live broker. |
| **Market Hours Check Does Not Consider Holidays** | market_intel.py:198–203 | LOW | Weekend/weekday check works, but US market holidays (Thanksgiving, July 4, etc.) not modeled. Agents can trade on closed market if date-based override. |
| **Follower Can Over-Leverage vs. Leader** | routes_signals.py:391–430, routes_trading.py:778–834 | MEDIUM | Copy trading replicates exact quantity, not adjusted for follower's account size. If follower has $50k and leader has $100k, follower gets same exposure = 100% vs. 50%, violating Kelly criterion / risk parity. |
| **Profit History Compaction Logic Opaque** | tasks.py (prune functions) | LOW | Code deletes old snapshots, but compaction strategy not documented. May lose granular equity history for old agents. |
| **Division by Zero in Drawdown Calc** | challenge_scoring.py:99 | LOW | Line 99: `(peak - equity) / peak * 100` if peak = 0 (agent loses everything), would error. Edge case: agent starts at $100k, goes to $0. Caught? No guard. |

**UNTESTED (Not Verified):**
- What happens if agent signals trade at exactly market close time?
- Polymarket token resolution and settlement flow (code exists, untested in teardown)
- Experiment variant assignment consistency across agent cohorts
- Database contention under 4k+ agents all trading simultaneously

### Performance / Architectural Risks

| Risk | File:Line | Impact | Mitigation in Code |
|------|-----------|--------|-------------------|
| **N+1 Queries on Leaderboard** | routes_trading.py:270–300 | For each agent, fetch profit_history separately; no batch query. Slow with 1000+ agents. | Some caching (LEADERBOARD_CACHE_TTL_SECONDS = 300), but inner loop still slow. |
| **Background Worker Singleton Lock Tight** | worker.py:38–51 (file lock) or redis lock | If worker dies uncleanly, lock may persist; prevents new worker from starting. | Renewal task every 40 sec should recover within 2 min. Not tested under load. |
| **Cache Stampede on Expiry** | routes_shared.py (caching module) | If cache key expires for hot signal feed, all agents hit database same instant. | TTL staggering or probabilistic early refresh not implemented. |
| **Profit History Table Unbounded Growth** | tasks.py (_prune_profit_history) | One snapshot per agent per ~300 sec = 8640 per agent per day; with 4k agents = 35M rows/day. | Prune logic deletes old snapshots (capped per agent), but max history not enforced until pruned. |

---

## 14. Verdict — Real System or Demo?

### Readiness Assessment

**REAL SYSTEM:**
- ✅ Atomic transactions, ACID compliance (SQLite/PostgreSQL)
- ✅ Deterministic portfolio replay (reproducible results)
- ✅ Challenge framework with disqualification (enforced rules)
- ✅ Risk-adjusted leaderboard scoring
- ✅ Multi-agent competition (4k+ agents tested per README)
- ✅ WebSocket real-time notifications
- ✅ Background worker separation from API service
- ✅ Experiment framework (A/B testing infrastructure)

**NOT PRODUCTION-READY:**
- ❌ Paper trading only; no live broker integration
- ❌ Prices via public APIs (Alpha Vantage, yfinance); subject to rate limits and delays
- ❌ No slippage, margin, or liquidity modeling
- ❌ Polymarket integration uses public API (real-time, but not integrated order routing)
- ❌ No authentication for strategy IP (all signals public; no private/protected strategies)

### How ARGUS Can Beat AI-Trader

**To win Track-2, ARGUS should:**

1. **Add True LLM Decision-Making in Critical Path**
   - AI-Trader uses LLM only for summaries
   - ARGUS should integrate LLM for trade signal generation: LLM reads market data → outputs buy/sell/hold decision → executed autonomously
   - Prove this with prompt log: show what context was given to LLM and what output was generated

2. **Implement Multi-Agent Coordination / Debate**
   - AI-Trader is single-agent per entity
   - ARGUS could spawn multiple reasoning paths, have agents critique each other's decisions, and select consensus action
   - Increases robustness vs. single-point-of-failure LLM

3. **Add Factor/Signal Discovery Loop**
   - AI-Trader has fixed technical indicators (MA, support/resistance)
   - ARGUS could use LLM to discover new correlations, backtest them, and adapt strategy dynamically
   - Requires research layer (historical data + analysis) but could be powerful differentiator

4. **Risk-Aware Position Sizing**
   - AI-Trader enforces max position % as hard cap
   - ARGUS could use LLM to dynamically adjust position size based on signal confidence, market regime, agent's drawdown YTD, etc.
   - Makes decisions more intelligent, less mechanical

5. **Real Broker Integration**
   - AI-Trader is paper only
   - ARGUS should at least provide hooks to real broker (Bitget, OKX) so real trading is possible
   - Requires infrastructure investment but is table-stakes for "real" agent

6. **Sentiment + Earnings Integration**
   - AI-Trader has sentiment API (Adanos) but does not integrate it into trading logic
   - ARGUS could pipe earnings calendar + sentiment into LLM reasoning: "Stock XYZ reports earnings tomorrow, sentiment is bearish, position size down 50%"

7. **Transparent, Auditable Decision Log**
   - AI-Trader logs signals but not the reasoning
   - ARGUS should log: market data snapshot → LLM prompt → LLM output → trade decision → execution result
   - Enables debugging, verification, and judge inspection

### What Would Lose Track-2

- **No LLM involvement** — using only rules-based trading
- **LLM does not control order placement** — using LLM for beautification only (exactly what AI-Trader does)
- **No demonstrated risk control** — open-ended position sizing with no drawdown management
- **Manual intervention required** — trading is not "autonomous"
- **Opaque decision process** — no audit trail of how decisions were made

---

## Summary

**AI-Trader is a functional, multi-agent trading simulation platform with strong infrastructure:**
- Atomic trades and positions
- Deterministic portfolio replay and challenge evaluation
- Real-time leaderboard scoring with risk adjustment
- Market data integration (news, macro, technicals)
- Experiment framework for A/B testing
- WebSocket notifications and signal feed
- Copy trading and follower network

**But it is NOT yet a Track-2-grade "LLM-driven agent" system:**
- LLM is not in the trading decision path
- Trading decisions are 100% agent responsibility, not LLM responsibility
- Market sensing is available but not integrated into decision logic
- Risk controls exist but are hard caps, not dynamic/intelligent

**For ARGUS to win,** it must make the LLM the **decision-maker**, not an assistant. Show the prompt, the LLM output, the trade decision, and the execution in one auditable chain. Add multi-agent coordination, signal discovery, and risk-aware position sizing on top.

---

**Analysis Completed:** 2026-09-12  
**Total Lines Analyzed:** 25,000+ (routes, database, challenge_scoring, market_intel, fees, skills)  
**Test Coverage:** Read-only code inspection; not run or tested locally  
**Confidence:** PROVED for architectural structure, fees, risk controls, leaderboard logic  
**NOT PROVED:** Polymarket settlement, experiment variant assignment consistency, performance under load
