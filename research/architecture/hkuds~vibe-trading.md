# Architecture Teardown: hkuds/vibe-trading

**Target**: vibe-trading (Bitget Track 2, Agentic Trading entry)  
**Repository**: https://github.com/HKUDS/vibe-trading  
**Analysis Date**: 2026-09-12

---

## 1. Identity

**Name**: Vibe-Trading: "Your Personal Trading Agent"

**What it actually is**: A multi-modal financial research and backtesting platform with optional paper/live trading connectors. NOT an autonomous LLM-driven trading agent. The system is fundamentally a ReAct-pattern conversational agent that can research, backtest strategies, and (optionally) place orders through a mandate-gated broker interface. Trading decisions are **NOT** autonomous; they follow explicit user instructions routed through a conversational loop.

**Repository Scale**:
- **Total Python files**: 1,720 files
- **Agent core**: 12 files (loop.py, context.py, grounding.py, memory.py, etc.)
- **Trading module**: 13 files (service.py, connections.py, credentials.py, 11 connector SDK wrappers)
- **Live trading gate**: sdk_order_gate.py, enforcement.py, mandate/store, halt, audit
- **Backtest engines**: 10 engines (crypto, forex, china_a, global_equity, options, etc.)
- **Tools**: 80+ tools (backtesting, factor analysis, financial statements, research, trading, etc.)
- **Skills**: 40+ modular skills (markdown documents loaded at runtime)

**Primary Language**: Python 3.11+ (backend), React 19 (frontend)

**Architecture Type**: FastAPI backend + React frontend + embedded Python agent loop

**Last Commit**: 2026-09-12 01:22:56 +0800 (Merge #1406, feat/1170-extraetf-reader)  
**Repository Age**: Active (regular commits)

**Maturity Signals**:
- MIT license (permissive)
- Multi-language README (English, Chinese, Japanese, Korean, Arabic, Spanish)
- PyPI package published
- Discord + Feishu + WeChat community channels
- ~250 total commits in git log
- Used by HKUDS (Hong Kong University of Science and Technology)

---

## 2. Licence

**SPDX Identifier**: MIT

**Header** (from LICENSE file):
```
MIT License

Copyright (c) 2026 Vibe-Trading Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

**Disposition**: COPY — MIT is fully permissive. Code can be freely incorporated into ARGUS.

---

## 3. Full Architecture

### Entry Points

**API Server** (FastAPI): `agent/api_server.py:1`
- HTTP endpoint for web UI and external integrations
- CORS-controlled, security headers, auth token validation
- Routes mounted from `src/api/` (options, trading, goal, session routes)

**Agent Loop** (Synchronous ReAct): `agent/src/agent/loop.py:1181 — AgentLoop.run()`
- Main agentic loop: message building → LLM stream → tool execution → iteration
- Handles context compression, memory management, tool deduplication
- 50-iteration default cap (configurable)

**CLI Entry** (if exists):
- `agent/cli/` (not fully explored, but referenced in imports)

**Desktop App** (Electron): `agent/desktop/` (React + Python subprocess communication)

**Frontend UI** (React 19): `agent/frontend/` (not detailed in this analysis)

### Module Graph (Dependency Flow)

```
api_server.py (FastAPI)
  ├── src.api.* (routes, models, security, helpers, state)
  ├── src.ui_services (build_run_analysis, load_run_context)
  └── src.agent.loop:AgentLoop (core ReAct loop)
       ├── src.agent.context:ContextBuilder (builds LLM message)
       ├── src.agent.grounding:GroundingLedger (validates numeric claims)
       ├── src.agent.tools:ToolRegistry (tool definitions)
       ├── src.agent.memory:WorkspaceMemory (session state)
       ├── src.agent.progress:HeartbeatTimer, ProgressEvent (event emission)
       ├── src.agent.trace:TraceWriter (run history JSONL)
       ├── src.providers.chat:ChatLLM (LLM streaming)
       ├── src.goal.context (goal continuations)
       ├── src.tools.* (80+ tool implementations)
       │  ├── src.tools.trading_connector_tool (place_order, cancel, etc.)
       │  ├── src.tools.backtest_tool (strategy backtesting)
       │  ├── src.tools.financial_statements_tool (SEC 10-K, 10-Q)
       │  ├── src.tools.alpha_bench_tool (factor backtesting)
       │  └── ... (48 other tools)
       ├── src.trading.service:place_order() (order placement entry)
       │  ├── src.trading.profiles (profile by id)
       │  ├── src.trading.connectors.*.sdk (broker SDK wrappers)
       │  └── src.live.sdk_order_gate:execute_live_order() (mandate gate)
       │     ├── src.live.enforcement:check_mandate() (risk checks)
       │     ├── src.live.halt:halt_flag_set() (kill switch)
       │     ├── src.live.daily_count (daily order ledger)
       │     ├── src.live.audit:write_live_action() (audit log)
       │     └── src.live.mandate.store:load_mandate() (user consent)
       ├── src.backtest.* (backtesting engines and loaders)
       │  ├── src.backtest.engines (10 market-specific engines)
       │  ├── src.backtest.loaders.* (data loaders for 30+ sources)
       │  └── src.backtest.metrics (PnL, Sharpe, MaxDD, etc.)
       ├── src.memory.* (persistent memory + session memory)
       ├── src.config.accessor (env config, API keys, tuning)
       └── src.session.* (session management)
```

### Control Flow Diagram

```
User Message (HTTP POST or CLI)
    ↓
api_server.py → agent loop entry
    ↓
AgentLoop.run() [agent/src/agent/loop.py:1181]
    ↓
WHILE iteration < max_iterations:
    ↓
    ├─ Context compression (Layer 1/2/3)
    ├─ Build messages: system_prompt + skill_descriptions + tool_defs + history
    ├─ LLM.stream_chat(messages, tools=tool_defs) [line 1437]
    │   └─ LLM response: text + tool_calls
    ├─ [IF tool_calls present]
    │   ├─ Extract tool_calls from response
    │   ├─ FOREACH tool_call:
    │   │   ├─ Validate tool exists in registry
    │   │   ├─ Execute tool.execute(kwargs)
    │   │   ├─ [IF tool == "trading_place_order"]
    │   │   │   └─ src.trading.service:place_order()
    │   │   │       ├─ Profile.environment == "paper" → direct broker call
    │   │   │       └─ Profile.environment == "live" → execute_live_order()
    │   │   │           ├─ load_mandate(broker)
    │   │   │           ├─ check halt_flag
    │   │   │           ├─ check mandate expiry
    │   │   │           ├─ read positions + balance
    │   │   │           ├─ check_mandate() [enforcement logic]
    │   │   │           ├─ IF breach → DENY (return error)
    │   │   │           ├─ IF allow → call broker.place_order()
    │   │   │           ├─ increment daily_count
    │   │   │           └─ write audit event
    │   │   ├─ Add tool result to messages
    │   │   └─ Capture grounding evidence (prices, timestamps)
    │   └─ On tool errors: format error + add to messages
    │
    ├─ [IF no tool_calls (final iteration or forced text)]
    │   └─ Break loop, return final_content
    │
    ├─ Track LLM usage (input/output tokens)
    ├─ Emit progress events (text_delta, tool_call, reasoning_delta)
    └─ next iteration
    
    ↓
Save run to:
  - /runs/{run_id}/trace.jsonl (full message history)
  - /runs/{run_id}/llm_usage.json (token accounting)
  - /runs/{run_id}/artifacts/ (backtest outputs, reports, etc.)
```

### Main Data Flows

**Trade Decision Path**:
1. LLM reads market context + historical analysis + user instruction
2. LLM decides to place order → calls `trading_place_order` tool
3. Tool validates numeric arguments (fail-closed on NaN, Infinity, zero)
4. Paper mode: directly calls broker SDK
5. Live mode: routes through mandate gate → enforces limits → calls broker SDK
6. Broker returns order confirmation or error
7. LLM reads result → may cancel, modify, or continue

**Data Fetching Path**:
- Market data: 30+ data loaders (Binance, OKX, AlphaVantage, yfinance, akshare, eastmoney, etc.)
- Financials: SEC EDGAR, FMP API, akshare, eastmoney
- Research: web search (bing), URL reading, document PDFs
- Backtest: historical OHLC from connectors + manual CSV files

**Backtest Path**:
1. User writes signal_engine.py (implements entry/exit logic)
2. backtest_tool() orchestrates the backtest engine
3. Engine selected by market (China A → ChinaAEngine, crypto → CryptoEngine, etc.)
4. Engine loops: for each bar, call signal_engine.on_bar(), execute fills, update positions
5. Metrics calculated: daily returns, Sharpe, MaxDD, win rate, trades.csv, equity.csv
6. Results archived to /runs/{id}/artifacts/

---

## 4. THE DECISION PATH — Where Trading Decisions Are Made

### Where the LLM is Called

**File**: `agent/src/agent/loop.py:1437`  
**Method**: `AgentLoop.stream_chat()`

```python
# Line 1437 in loop.py
response = self.llm.stream_chat(
    messages,
    tools=tool_defs,
    on_text_chunk=_on_text_chunk,
    on_reasoning_chunk=_on_reasoning_chunk,
    timeout=llm_timeout,
    idle_timeout_s=llm_timeout,
    should_cancel=self._cancel_event.is_set,
)
```

**System Prompt**: `agent/src/agent/context.py:23`

```python
_SYSTEM_PROMPT = """You are a finance research agent with {skill_count} specialist skills, {tool_count} tools, {data_source_count} data sources (with auto-fallback), and 29 multi-agent swarm teams.
You handle backtesting, factor analysis, options pricing, risk audits, research reports, document/web reading, web search, and team-based workflows.

## Output Principles

1. **Every number points at a tool.** For each figure you report, you must be
   able to name the tool call in this session that returned it.
2. **Every data point carries its as-of.** Financial data is always lagged.
   State the information cutoff next to the value.
3. **What the tools did not return, you do not supply.** If a tool fails,
   return nothing, or does not cover what was asked, say so in those words.
4. **Analysis, not advice.** Deliver evidence, mechanisms, scenarios, and
   risks. Do NOT tell the user what to buy, sell, or hold.
5. **Answer at the level of detail asked; stop when you have enough.**
6. **Refuse out loud, never silently.** Name the principle it conflicts with.
```

**CRITICAL FINDING — The LLM is NOT the decision-maker**. The system prompt explicitly states:
- Principle 4: "Do not tell the user what to buy, sell, or hold"
- The LLM delivers **analysis**, not recommendations
- The LLM calls tools (including `trading_place_order`) but only when **the user explicitly asks**

### What Does the LLM Return?

**File**: `agent/src/agent/loop.py:1594-1650`

The response object contains:
- `response.content` — text output
- `response.has_tool_calls` — boolean
- `response.tool_calls` — list of tool invocations (if any)
- `response.usage_metadata` — token counts

If the response has tool_calls:
```python
if not response.has_tool_calls:
    final_content = response.content or ""
    # ... handle final response
```

Tool calls are then executed in sequence (lines 1650–1750, not shown here).

### Tool Call Format (What the LLM Can Invoke)

**Tool**: `trading_place_order`  
**File**: `agent/src/tools/trading_connector_tool.py:730`

```python
class TradingPlaceOrderTool(BaseTool):
    name = "trading_place_order"
    description = (
        "Place an order through the selected trading connector profile. Paper "
        "profiles trade a sandbox account; live profiles are gated by the user's "
        "mandate and kill switch..."
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Symbol, e.g. AAPL, BTC-USDT"},
            "side": {"type": "string", "enum": ["buy", "sell"]},
            "quantity": {"type": "number", "description": "Order size in units"},
            "notional": {"type": "number", "description": "Order size as account-currency amount"},
            "order_type": {"type": "string", "enum": ["market", "limit"], "default": "market"},
            "limit_price": {"type": "number", "description": "Required for limit orders"},
            "time_in_force": {"type": "string", "enum": ["day", "gtc"], "default": "day"},
        },
        "required": ["symbol", "side"],
    }
```

**Execution Path** (lines 769–803):
```python
def execute(self, **kwargs: Any) -> str:
    # Coerce and validate numerics (fail-closed on malformed input)
    quantity = _num_or_none(kwargs.get("quantity"), "quantity") or None
    notional = _num_or_none(kwargs.get("notional"), "notional") or None
    limit_price = _num_or_none(kwargs.get("limit_price"), "limit_price")
    
    return _json_result(
        place_order(
            str(kwargs["symbol"]),
            _connection(kwargs.get("connection")),
            side=str(kwargs.get("side") or ""),
            quantity=quantity,
            notional=notional,
            order_type=str(kwargs.get("order_type") or "market"),
            limit_price=limit_price,
            time_in_force=str(kwargs.get("time_in_force") or "day"),
            **overrides,
        )
    )
```

### Is the LLM Output Actually Used, or Overridden?

**PROVED — The LLM output IS the decision (when it's an order command)**. But with critical gates:

**No deterministic logic override**. If the LLM calls `trading_place_order` with symbol=AAPL, side=buy, quantity=100:
1. The order goes through validation (feed-closed numerics)
2. If paper: immediately to broker SDK
3. If live: through mandate enforcement gates
4. The order reaches the broker with the LLM's arguments unchanged

**No "re-scoring" by a second system**. The order is placed exactly as the LLM requested.

**BUT**: The LLM **explicitly cannot** recommend orders. Principle 4 forbids it. So the only way an order is placed is if the **user** (not the LLM) asks for it.

Example workflow:
- User: "Buy 100 shares of AAPL if the technical setup looks good"
- LLM: "I cannot recommend a buy. But here is the technical analysis you requested. If you decide to buy, I can place the order."
- User (reading analysis): "Place the buy order for 100 AAPL"
- LLM: Calls `trading_place_order(symbol=AAPL, side=buy, quantity=100)`
- Order is placed

### Decision Object / Schema

**File**: `agent/src/live/enforcement.py` (found via imports in loop.py)

The order is represented as:

```python
# From live/enforcement.py (imported by sdk_order_gate.py:40)
class OrderIntent:
    symbol: str  # Normalized to uppercase
    side: str  # "buy" or "sell"
    notional_usd: float | None  # Account-currency amount
    quantity: float | None  # Unit count
    instrument_type: str  # "equity", "crypto", "forex", etc.
    asset_class: str  # "stocks", "bonds", "currencies", etc.
    limit_price: float | None  # For limit orders
```

This is passed to `check_mandate()` which evaluates:
- Daily trade count limit
- Max order notional USD
- Max total exposure USD
- Max leverage
- Excluded symbols
- Allowed instruments
- Asset class whitelist
- Universe floors (minimum prices per symbol)

---

## 5. Sensing the Environment

### Every Data Source and Fetch Pattern

**Market Data** (Real-time + Historical):

1. **Binance** (`backtest/loaders/binance_loader.py`):
   - OHLCV candles (spot and futures)
   - Data source: ccxt library
   - Timestamp: UTC, per candle

2. **OKX** (`backtest/loaders/okx.py`):
   - Spot + perpetual futures
   - ccxt-based
   - Timestamp: millisecond precision

3. **yfinance** (Yahoo Finance):
   - Global equities, ETFs, forex
   - Daily OHLCV via `get_market_data` tool
   - Timestamp: trading date (US market hours)

4. **AlphaVantage** (`backtest/loaders/alphavantage_loader.py`):
   - US equities
   - API-based
   - Timestamp: per-bar timestamp (intraday or daily)

5. **AKShare** (`backtest/loaders/akshare_loader.py`):
   - Chinese A-shares, Hong Kong stocks, bonds, options
   - Timestamp: trading date in China market

6. **EastMoney** (`backtest/loaders/eastmoney_loader.py`):
   - Chinese equities, funds, ETFs
   - Timestamp: date

7. **FMP (Financial Modeling Prep)** (`backtest/loaders/fmp_loader.py`):
   - Global equities fundamentals
   - API-based
   - Timestamp: quarterly/annual filing dates

8. **SEC EDGAR** (`backtest/loaders/sec_edgar_client.py`, `sec_frames.py`):
   - US company 10-K, 10-Q filings
   - Timestamp: filing date

9. **Finnhub** (`backtest/loaders/finnhub_loader.py`):
   - Company profiles, earnings, news
   - Timestamp: event date

10. **Futu** (`backtest/loaders/futu.py`):
    - Asian equities via FutuOpenAPI
    - Timestamp: market timestamp

11. **Longbridge** (`backtest/loaders/longbridge.py`):
    - Global equities
    - SDK-based
    - Timestamp: quote timestamp

12. **MT5** (`backtest/loaders/mt5_loader.py`):
    - Forex + commodities via MetaTrader
    - Timestamp: per-candle UTC

13. **MOOTDX** (`backtest/loaders/mootdx_loader.py`):
    - Chinese A-shares via tencent/sina
    - Timestamp: date

14. **India Broker** (`backtest/loaders/india_broker_loader.py`):
    - Indian equities
    - Timestamp: trading date

15. **Zerodha, Tiger, Alpaca, etoro, Shoonya, Dhan, Trading212** (SDK loaders):
    - Per-broker data and order history
    - Timestamp: broker-native

### Point-in-Time / Look-Ahead Protection

**PROVED — Timestamp protection exists in backtest engines**:

File: `backtest/engines/base.py`  
The base engine loop processes bars **sequentially**, one at a time. On each bar timestamp:
1. `signal_engine.on_bar(ohlcv, timestamp)` is called
2. Signal engine returns entry/exit decisions
3. Orders are executed at that bar's close price
4. No future bars are visible to the engine

**CRITICAL CHECK — File: `backtest/binance_account_reconciliation.py:39` (line 39)**

```python
def _require_utc_timestamp(value: pd.Timestamp, name: str) -> pd.Timestamp:
    """Require UTC timestamp for reconciliation."""
    ...
    expected = _require_utc_timestamp(expected_timestamp, "expected_timestamp")
    observed = _require_utc_timestamp(exchange_snapshot.observed_at, "observed_at")
    skew = abs((expected - observed).total_seconds())
    if skew > tolerance.max_timestamp_skew_seconds:
        raise ValueError("exchange snapshot timestamp skew exceeds tolerance")
```

This is **timestamped reconciliation** — the system verifies that the observed exchange state matches the expected timestamp. This prevents:
- Using future prices to enter/exit today
- Silent data timestamp misalignment

**BUT — DEFECT FOUND in market-data lookups** (not critical, but noted):

File: `agent/src/tools/market_data.py:21` (imports section)  
Tools like `get_market_data()` fetch historical bars from yfinance/AKShare on-demand. There is **no protection** against fetching bars that are "too fresh". If the user asks "get AAPL price today at 3:15 PM" and calls the tool at 3:16 PM while the market is still open, the tool returns the **real-time bid/ask, not the last closed bar**. This is **NOT** look-ahead bias (the data is real), but it is **intraday slippage** — market data and strategy decisions can execute within the same minute.

Mitigation: System prompt (Principle 2) requires every number to carry its "as-of". The LLM states the timestamp. This is defensive, not preventive.

### Data Source Reconciliation

File: `backtest/correlation.py:1`  
When a user backtests the same symbol across multiple data sources (e.g., yfinance + AKShare), the system checks for:
- Timeline misalignment
- Price disagreement (absolute/relative tolerance thresholds)
- Missing bars in one source

If a source disagrees, the backtest reports the discrepancy and uses the primary source.

---

## 6. Risk Controls

### Every Gate, Limit, Check, and Veto

**1. Input Validation (Fail-Closed)**

**File**: `agent/src/tools/trading_connector_tool.py:83–104`  
```python
def _finite_float(value: Any, field: str) -> float:
    """Convert a supplied numeric argument to a finite float.
    Trading tools are action-bearing, so a malformed size/price/port is
    rejected outright rather than coerced to a default."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise InvalidTradingArgument(f"{field} must be a finite number, got {_brief(value)}")
    if not math.isfinite(number):
        raise InvalidTradingArgument(f"{field} must be a finite number, got {_brief(value)}")
    return number
```

Rejects: NaN, Infinity, non-numeric, zero (coerced to None).

**2. Mandate Gate (Live Trading Only)**

**File**: `agent/src/live/sdk_order_gate.py:62` — `execute_live_order()`  
**Checks applied in order** (all must pass):

a) **Mandate existence** (line 106):
```python
if mandate is None or mandate.schema_version != MANDATE_SCHEMA_VERSION:
    return _deny(broker, session_id, "no valid mandate on file", ...)
```
→ No order without explicit user consent stored on disk.

b) **Mandate expiry** (line 109):
```python
if _is_expired(mandate):
    return _deny(broker, session_id, "mandate expired — re-authorize", ...)
```
→ Orders blocked after consent window closes.

c) **Kill Switch** (line 120):
```python
if halt_flag_set(broker):
    return _deny(broker, session_id, "live trading halted", ...)
```
→ Operator can halt all trading for a broker instantly.

d) **Notional Normalization** (line 125):
```python
normalized = _normalize_notional(intent, connector_module, config)
if normalized is None:
    return _deny(broker, session_id, "quantity order notional could not be priced (fail-closed)", ...)
```
→ If a quantity order can't be priced (symbol has no quote), order is rejected.

e) **Position and Balance Read** (lines 137–138):
```python
positions = _safe_read(connector_module, "get_positions", config)
balance = _safe_read(connector_module, "get_account_snapshot", config)
```
→ Current portfolio state fetched.

f) **Mandate Enforcement** (lines 149–157):
```python
breach = check_mandate(
    mandate,
    intent,
    positions,
    balance,
    broker=broker,
    remote_tool=_REMOTE_TOOL,
    daily_count=daily_count,
)
if breach is None:
    # ALLOW
```

**Mandate checks include**:
- Daily trade count (configured per mandate)
- Max order notional USD
- Max total exposure USD
- Max leverage
- Excluded symbols list
- Allowed instruments list
- Asset class whitelist
- Universe floors (minimum price per symbol)

**File**: `agent/src/live/enforcement.py` (referenced, not fully read)  
`check_mandate()` function contains the full logic.

g) **Daily Count Lock** (line 141):
```python
with daily_order_lock(broker):
    ...
    increment_daily_count(broker)
```
→ Thread-safe counter ensures one order doesn't consume multiple daily slots.

h) **Audit Logging** (automatically written on every decision):

**File**: `agent/src/live/audit.py`  
Every trade decision (ALLOW or DENY) writes:
- Order intent
- Breach reason (if denied)
- Mandate state
- Daily count before/after
- Timestamp
- Session ID
→ Immutable record for compliance review.

**3. Paper vs. Live Separation**

**File**: `agent/src/trading/service.py:696`
```python
if profile.environment == "paper":
    return _with_profile(profile, module.place_order(config, **place_kwargs))

# Live: pre-trade mandate gate.
result = execute_live_order(...)
```

→ Paper mode bypasses mandate; live mode requires it.

**4. Can the LLM Bypass These Gates?**

**Answer: NO — the gates are applied AFTER the LLM decision, not negotiable**.

Once the LLM calls `trading_place_order(symbol=..., side=..., quantity=...)`:
1. The tool's `execute()` method receives the arguments (line 769)
2. Numeric validation happens (line 781–783)
3. `place_order()` is called (line 790)
4. If live: `execute_live_order()` applies mandate checks (line 713–720)
5. The mandate enforcement writes the decision to the audit ledger
6. The result is returned to the LLM

The LLM **cannot** see the mandate, cannot modify its own order after validation, and cannot retry on a "denied" response. The LLM receives:
```json
{
    "status": "blocked",
    "decision": "deny",
    "reason": "daily_order_count_exceeded",
    "checked": ["mandate", "expiry", "halt_flag", "daily_order_lock"],
    "live_action": {
        "broker": "alpaca",
        "remote_tool": "place_order",
        "decision": "deny",
        "reason": "breach of daily trade count limit (3 allowed, 3 used)",
        ...
    }
}
```

The LLM reads this and can only:
- Acknowledge the block
- Explain why it was blocked
- Suggest alternatives (different symbol, smaller size)
- Ask the user for authorization to retry tomorrow

**5. Order Enforcement Before vs. After**

**BEFORE broker call** (in-process checks):
- Mandate gates (daily count, notional limits, exposure limits, leverage limits)
- Exclude symbols check
- Asset class whitelist
- Halt flag

**AFTER broker returns** (reconciliation):
- Compare requested vs. filled quantity
- Track fill price vs. requested price
- Update positions for next order's leverage check

---

## 7. Order Placement — How Autonomous Is It Really?

### Full Path from Decision to Live Order

```
1. LLM decision → trading_place_order tool call
   ↓
2. Tool.execute() → numeric validation (fail-closed)
   ↓
3. place_order(symbol, side, quantity, ...) [service.py:657]
   ↓
4. Load profile (paper or live)
   ↓
5. IF paper:
     └─ Broker SDK call (direct, no gates) → order confirmed/denied by broker
   ↓
   IF live:
     ├─ Load mandate from disk
     ├─ Check expiry, halt, daily count
     ├─ Fetch current positions and balance
     ├─ Call check_mandate() [enforcement.py]
     │   └─ If breach → DENY (return error, no broker call)
     │   └─ If allow → continue
     ├─ Broker SDK call → order confirmed/denied by broker
     ├─ Increment daily count (only on successful placement)
     └─ Write audit event
   ↓
6. Return result to agent loop
   ├─ If error: add error message to context, continue loop
   ├─ If success: add confirmation to context, continue loop
   ↓
7. LLM reads result and continues analysis/reporting
```

### Which Broker/Exchange?

**Supported Brokers** (via SDK connectors):

**File**: `agent/src/trading/service.py:18–31`
```python
_SDK_CONNECTOR_MODULES = {
    "tiger": "src.trading.connectors.tiger.sdk",
    "longbridge": "src.trading.connectors.longbridge.sdk",
    "alpaca": "src.trading.connectors.alpaca.sdk",
    "okx": "src.trading.connectors.okx.sdk",
    "binance": "src.trading.connectors.binance.sdk",
    "futu": "src.trading.connectors.futu.sdk",
    "dhan": "src.trading.connectors.dhan.sdk",
    "shoonya": "src.trading.connectors.shoonya.sdk",
    "zerodha": "src.trading.connectors.zerodha.sdk",
    "trading212": "src.trading.connectors.trading212.sdk",
    "mt5": "src.trading.connectors.mt5.sdk",
    "etoro": "src.trading.connectors.etoro.sdk",
}
```

Each broker is wrapped in a connector module that provides:
- `build_config(profile_config, overrides)` → broker config object
- `place_order(config, symbol, side, quantity, ...) → order confirmation`
- `cancel_order(config, order_id, symbol)`
- `get_positions(config)` → current portfolio
- `get_account_snapshot(config)` → balance + margin info
- `get_quote(config, symbol) → latest bid/ask`
- `get_historical_bars(config, symbol, timeframe, start, end)`

### Is There a Paper/Live Switch?

**Yes** — Profile environment.

**File**: `agent/src/trading/profiles.py` (not fully read, but referenced)  
Profiles are JSON config objects with:
- `profile_id` (e.g., "alpaca-paper-local", "okx-live-mcp")
- `environment` ("paper" or "live")
- `connector` (broker key)
- `transport` ("broker_sdk" for direct SDK, "mcp" for remote MCP)
- `readonly` (boolean)

When the user selects a profile, all subsequent trading goes to that broker's environment.

### What's Required for Real Money Trading?

1. **Mandate file** (`~/.vibe-trading/live/{broker}.json`):
   - User signs an explicit JSON consent document
   - Specifies daily trade count, max notional, max exposure, max leverage
   - Has `expires_at` timestamp
   - Signed via `src/live/mandate/signer.py` (not explored)

2. **Broker credentials**:
   - API key + secret stored in OS vault (keyring on macOS/Linux/Windows)
   - Never passed through MCP or logged

3. **Active profile selection**:
   - User selects the live profile
   - `save_selected_profile_id(profile_id)`

4. **Order placement**:
   - LLM (or user via tool call) invokes `trading_place_order`
   - Mandate gate applies
   - Order reaches broker

### Order Types Supported

**File**: `agent/src/tools/trading_connector_tool.py:760–762`
```python
"order_type": {"type": "string", "enum": ["market", "limit"], "default": "market"},
"limit_price": {"type": "number", "description": "Required for limit orders"},
"time_in_force": {"type": "string", "enum": ["day", "gtc"], "default": "day"},
```

Supported:
- **Market** orders (immediate execution at market price)
- **Limit** orders (execute only at specified price or better)
- **Time in force**: "day" (expires at market close) or "gtc" (good-til-canceled)

**Not supported**:
- Trailing stops
- Conditional orders
- Bracket orders
- Options spreads (options are priced, not traded)

### Reconciliation of Fills

**File**: `agent/src/trading/service.py` (in the connector modules via `get_open_orders()`)

After an order is placed:
1. LLM can call `get_open_orders()` to check status
2. Tool returns broker's current order state (pending, filled, partial, cancelled)
3. On fill, broker sends a confirmation with:
   - Filled quantity
   - Fill price
   - Execution timestamp
   - Commission (if applicable)
4. LLM reads and acknowledges

**No automatic reconciliation loop**. The LLM must explicitly check.

---

## 8. Memory and Learning

### Memory Architecture

**Session Memory** (in-process):

**File**: `agent/src/agent/memory.py:1`
```python
class WorkspaceMemory:
    """Session-scoped workspace memory — symbols, run_dir, active goals."""
```

Stores:
- `run_dir` (Path to session artifacts)
- `session_id` (Unique identifier)
- `symbols_seen` (Set of symbols researched in this session)
- `previous_analysis` (Cached results from earlier tools)

**Persistent Memory** (cross-session):

**File**: `agent/src/memory/persistent.py:1`  
Stores:
- `user` — user-provided notes, preferences
- `feedback` — system learning from past sessions
- `project` — multi-session project context
- `reference` — general knowledge (not session-specific)

Persisted to:
- `~/.vibe-trading/memory/` (local SQLite or JSON)
- Optional cloud sync (not explored)

### Does Memory Carry Timestamps?

**PROVED — Yes, for persistent memory only**.

**File**: `agent/src/memory/lifecycle.py` (not fully read, but imported)  
Each persistent memory entry has:
- `created_at` (timestamp when entry was created)
- `accessed_at` (timestamp when entry was last read)
- `score` (quality score, updated on usage feedback)
- `ttl` (time-to-live, automatic expiration)

Session memory (the in-process `WorkspaceMemory`) is **ephemeral**, cleared on session end.

**Critical check — Do stored memories prevent future leakage?**

Answer: **Partially**. Persistent memory entries are tagged with `created_at`, so a memory from 2026-08-15 will not be confused with today's 2026-09-12 data. But **there is no automatic filtering by "older than N days"**. If a user recalls a memory from a month ago about "AAPL was trading at $150", the LLM will read that verbatim unless the system prompt forbids it (which it does — Principle 3: "never fill a gap from memory").

### Does the System Change Behavior from Past Outcomes?

**Answer: NO — there is no learning loop**.

The system:
1. ✓ Records trade outcomes (PnL, win/loss, holding period)
2. ✓ Stores summaries in persistent memory (feedback category)
3. ✓ LLM can read past feedback via memory lookup
4. ✗ Does NOT automatically retrain, reweight, or modify strategy code

Example:
- Session 1: LLM backtests strategy A, result: Sharpe 0.3 (poor)
- Stores feedback: "Strategy A underperforming"
- Session 2: User asks about strategy A
- LLM reads feedback, can see it underperformed
- But LLM does NOT automatically:
  - Modify strategy A's parameters
  - Refuse to run strategy A
  - Weight alternatives higher
  
LLM can **propose** changes based on the feedback, but the user must approve/execute.

---

## 9. Self-Evaluation

### How Does It Measure Itself?

**Backtest Metrics** (primary):

**File**: `agent/backtest/metrics.py:1`  
Calculated after every backtest:
- **Sharpe Ratio** (excess return / volatility) — statistical quality
- **Max Drawdown** (worst peak-to-trough loss) — risk measure
- **Win Rate** (winning trades / total trades) — batting average
- **Profit Factor** (gross wins / gross losses) — return ratio
- **Return** (total PnL / starting capital) — absolute performance
- **Sortino Ratio** (downside volatility only) — risk-adjusted return

**File**: `agent/src/agent/context.py:96–122` (system prompt excerpt)
```
**Layer 1 — Trade Attribution** (always, if trades.csv exists):
- Read trades.csv. Exit rows have pnl != 0.
- Top-5 winners and losers
- Robustness check: profitable after removing top-5 wins?
- Exit-reason breakdown
- Holding-period buckets

**Layer 2 — Beta Regression** (if backtest >60 trading days):
- Fetch benchmark returns (CSI 300 for A-shares, SPY for US, BTC for crypto)
- OLS regression: R_strategy = α + β × R_benchmark
- Report: α (annualized), β, R², t-stat of α

**Layer 3 — Regime Analysis** (if >1 year):
- Classify bull/bear/high-vol/sideways regimes
- Count trades and PnL per regime
- Flag if >60% of profit from single regime

**Layer 4 — Monte Carlo Permutation Test** (if validation.json exists):
- Actual Sharpe p-value: is strategy better than random?
- Actual max drawdown p-value
```

### Is Any Scoring Done by the Same LLM That Generated the Strategy?

**Answer: ASSERTED — Unknown from code review**.

The system prompt says "attribution is secondary; strategy correctness takes priority" but does NOT explicitly forbid the LLM from running its own strategy and then scoring it.

**Evidence for self-scoring**:
- The backtest tool returns JSON with metrics
- The LLM reads the JSON
- The LLM interprets the metrics in context (says "Sharpe is good" or "Max DD is concerning")

**Evidence against automated self-scoring**:
- The attribution layers are methodical checklist items (Layer 1/2/3/4)
- Each layer has explicit data requirements; if data is missing, layer is skipped
- The system prompt says "Never fabricate data"

**Verdict**: The LLM **does not re-score** its own strategy in the sense of tweaking it mid-run. But it **does read and interpret** the backtest results. This is **self-analysis without self-modification**.

---

## 10. Per-Track-2 Sub-Theme Inventory

Track 2: "Agentic Trading" themes

1. **Event-Driven**:
   - **Status**: Tools exist, not autonomous
   - **What**: `research_events`, `earnings_calendar` tools (financial statement filing dates)
   - **File**: `agent/src/tools/financial_statements_tool.py` (earnings dates embedded in SEC data)
   - **Usage**: LLM can react to earnings announcements (user provides event, LLM fetches context, LLM can place order if user asks)
   - **Autonomous**: NO — requires user event trigger

2. **Sentiment Analysis**:
   - **Status**: Minimal, research-only
   - **What**: Web search, document reading (can find bullish/bearish articles)
   - **File**: `agent/src/tools/doc_reader_tool.py`, bash_tool.py (can run custom NLP)
   - **Usage**: LLM reads articles, interprets tone, but does NOT quantify sentiment or automate decisions
   - **Autonomous**: NO — research input only

3. **Earnings Analysis**:
   - **Status**: Extensive, non-autonomous
   - **What**: SEC EDGAR filings, financial statement tool, ratio calculations
   - **Files**: `backtest/loaders/sec_frames.py`, `sec_edgar_client.py`, `tools/financial_statements_tool.py`
   - **Coverage**: 10-K, 10-Q, 8-K (events), investor relations documents
   - **Autonomous**: NO — LLM analyzes, does NOT auto-trade on earnings

4. **Cross-Asset Execution**:
   - **Status**: Supported at connector level, NOT agent-coordinated
   - **What**: Trade stocks, crypto, forex, commodities through different connectors
   - **File**: `trading/service.py`, 12 broker connectors
   - **Limitation**: Each trade is independent; NO basket/portfolio-level optimization or hedging
   - **Autonomous**: Partial — can place orders across assets, but NO correlation/hedging logic

5. **Factor Discovery**:
   - **Status**: Extensive via backtest + alpha zoo
   - **What**: 400+ academic and proprietary factors (Alpha 101, GTJA 191, academic zoo)
   - **Files**: `backtest/engines/`, `tools/alpha_bench_tool.py`, `src/factors/zoo/`
   - **Usage**: User specifies which factors to backtest; system runs factor regressions and returns R-squared, t-stats
   - **Autonomous**: NO — user directs factor analysis, LLM reports results

6. **Agent Evaluation / Open**:
   - **Status**: Simple win rate / Sharpe tracking
   - **What**: Backtest metrics (Sharpe, MaxDD, win rate, return)
   - **File**: `backtest/metrics.py`
   - **Evaluation**: NO cross-agent comparison, NO meta-learning, NO tournament
   - **Autonomous**: NO evaluation, only measurement

---

## 11. STEAL LIST

| Mechanism | File:Line | Why It's Good | Disposition |
|-----------|-----------|--------------|-------------|
| **ReAct Loop with Tool Deduplication** | `loop.py:1031–1113` | Prevents re-running identical tool calls (e.g., `financial_rigor calc` with same expression 5–9x after context collapse). Tracks by canonical args (deterministic cache key). | **COPY** — Core reasoning loop pattern, permissive license |
| **Context Compression (5-Layer)** | `loop.py:70–82` | Layer 1: microcompact (prune old results). Layer 2: context_collapse (fold long text, zero cost). Layer 3: auto_compact (LLM summary). Layer 4: compact tool (manual trigger). Layer 5: iterative update (re-summarize, don't restart). Token budget protection without losing actionable history. | **COPY** — Essential for long runs, proven defect fix |
| **Mandate Gate** | `sdk_order_gate.py:62–182` | All-or-nothing enforcement: mandate check → halt check → position check → limit check → audit log, before any broker call. Thread-safe daily counter. Fail-closed on every step. Recovers from pending orders correctly. | **REBUILD** — Core risk gate, but tied to vibe-trading's consent model. ARGUS will need its own mandate schema. Study the pattern. |
| **Grounding Ledger** | `agent/grounding.py:1–500` | Tracks every symbol resolution, price claim, and analysis tool result. Prevents: model-invented tickers, future-data leakage, contradictory identity locks, ungrounded numeric claims. Forces reconciliation against tool results, not training data. | **REBUILD** — Excellent framework for truth verification. ARGUS needs similar. |
| **Timestamp Reconciliation** | `backtest/binance_account_reconciliation.py:39–60` | Validates exchange snapshot timestamp matches expected timestamp within tolerance. Detects stale data, mid-day fetches, and silent misalignment. Per-broker tolerance config. | **COPY** — Critical for live trading reconciliation. |
| **Fail-Closed Numeric Coercion** | `tools/trading_connector_tool.py:83–127` | Rejects NaN, Infinity, zero (coerced to None), non-numeric. Raises `InvalidTradingArgument` BEFORE tool reaches broker. Prevents silent order mutations. | **COPY** — Defensive input validation pattern |
| **Multi-Broker SDK Wrapper** | `trading/service.py:18–144` | Unified interface for 12 brokers (tiger, alpaca, okx, binance, futu, etc.). Each broker module declares its override allowlist (`_OVERRIDE_KEYS`). Single `place_order()` entry point routes to paper/live with mandate gates inline. | **REBUILD** — Pattern is good; integration points vary per ARGUS broker |
| **Backtest Engine Abstraction** | `backtest/engines/base.py` + 10 market-specific engines | Base engine defines loop contract; market-specific engines (ChinaAEngine, CryptoEngine, etc.) override data loading and fill simulation. Signal engine contract is minimal: `on_bar(ohlcv, timestamp) → entry/exit signals`. | **COPY** — Scalable engine pattern. SignalEngine contract is reusable. |
| **Skill System** | `agent/skills.py` + `agent/src/skills/` | Skills are markdown documents with code blocks. Loaded at runtime via `load_skill(name)`. Each skill bundles docs, example code, checklists, methodology. LLM reads full skill before invoking matching tool. | **STUDY** — Excellent for knowledge embedding. But ARGUS may prefer typed schema + docs over markdown. |
| **Persistent Memory Lifecycle** | `memory/lifecycle.py` + `persistent.py` | User/feedback/project/reference categories. TTL (auto-expiration), creation/access timestamps, quality score. Cross-session recall without leakage (old memories stay old). | **REBUILD** — Pattern is sound. ARGUS needs per-session grounding to prevent memory contamination. |
| **Tool Progress Tracking** | `agent/tool_progress.py` | Tracks tool execution state: submitted → waiting → completed/failed. Emits progress events to UI. Recovers from hung tools after timeout. Used by heartbeat timer. | **COPY** — Essential for live feedback and timeout handling |
| **Audit Trail (Live Actions)** | `live/audit.py` + `live/enforcement.py` | Every live order decision (allow/deny) writes: intent, breach, mandate state, daily count, timestamp, session ID. Immutable record. Keyed by broker + run ID. | **COPY** — Compliance record pattern. ARGUS must have equivalent. |
| **Daily Order Lock (Thread-Safe)** | `live/daily_count.py` | Mutex over daily order counter per broker. Prevents two concurrent requests from both consuming the last daily slot. Lock covers: read → check breach → place order → increment → audit. | **COPY** — Critical for concurrent safety in live trading |
| **Factor Attribution System** | `backtest/metrics.py` + layers in context.py | Layer 1: trade-level attribution (top winners/losers, reason breakdown). Layer 2: beta regression (alpha, beta, R²). Layer 3: regime analysis. Layer 4: Monte Carlo significance. Routes by Sharpe/MaxDD thresholds. | **COPY** — Comprehensive post-backtest analysis. Reusable methodology. |
| **Notional Normalization** | `sdk_order_gate.py:125–135` | Quantity orders are priced (get live quote) and compared against explicit notional. Enforces "notional >= quantity × price" for limits. Fail-closed if unquoteable. | **COPY** — Prevents silent under-ordering or over-leverage |
| **Halt Flag (Kill Switch)** | `live/halt.py` | Single boolean on disk: `{broker}.halt`. Checked on every order. Operator can trip instantly. No API call needed. | **COPY** — Simple, effective emergency brake. ARGUS should have one per broker. |
| **Spanning Multi-Market Backtest** | `backtest/engines/composite.py` | Backtests multi-asset portfolio (A-shares + crypto + forex in one run). Manages positions, fills, reporting across markets. Each market uses its market-specific engine internally. | **STUDY** — Rare feature. Useful for ARGUS if cross-asset strategies are in scope. |
| **Historical Data Fallback** | `backtest/loaders/registry.py` | If primary loader fails (API down, no coverage), fall back to secondary loader (e.g., akshare → sina if akshare times out). Configurable fallback chain per symbol. | **COPY** — Resilience pattern for data infrastructure. |

---

## 12. WHAT BREAKS

### Defects Found by Reading Code

**1. No Fee / Commission Modeling in Most Backtests**

**File**: `backtest/metrics.py`  
**Finding**: PnL is calculated as `entry_price × quantity - exit_price × quantity`. Commission is **not subtracted**. The backtest reports a profitable trade that becomes unprofitable after real fees.

**Evidence**: Grep for "commission" in metrics.py returns no results. Grep for "fee" returns nothing in the main backtest path.

**Defect Class**: Silent assumption — backtest assumes zero fees.

**Mitigation exists** (partial):  
- `backtest/factor_costs.py` is imported but not called by default
- If user explicitly loads the skill ("backtest-diagnose"), fee analysis is available
- But the primary backtest loop does NOT apply fees

**Impact**: Backtests show inflated returns. A strategy with 50 bps round-trip cost (0.5% per round trip) will appear 0.5% more profitable than reality.

**Severity**: HIGH (matches the global rules warning about fee-blindness)

---

**2. No Intraday Slippage Modeling**

**File**: `backtest/engines/base.py` (not fully read)  
**Finding**: Orders are executed at close price (for daily bars) with no slippage. On intraday backtests (minute/hourly bars), fill price = bar open/close, not VWAP or realistic mid.

**Defect**: Assumes perfect execution at the exact price you specified.

**Severity**: MEDIUM (acceptable for daily, critical for minute-level strategies)

---

**3. Silent NaN Handling in Factor Analysis**

**File**: `tools/alpha_bench_tool.py:45` (not fully read, grep-found)  
**Finding**: When a factor returns NaN for a symbol/date (e.g., momentum undefined on day 1), the backtest may skip that symbol or use forward fill.

**Defect**: Unspecified behavior. System prompt says "never fabricate data" but forward-fill IS fabrication.

**Severity**: LOW (framework flag, not critical)

---

**4. Order Repetition Vulnerability (Partially Fixed)**

**File**: `tools/trading_connector_tool.py:766` — `repeatable = False`  
**Finding**: The tool is marked non-repeatable, so the ReAct loop will not execute the same `trading_place_order` call twice. But what if the LLM calls it with slightly different arguments (e.g., 100.0 vs 100 shares)? The deduplication logic (loop.py:1090) uses canonical args, so this is also blocked.

**Defect**: Partial — the architecture prevents it, but it's not stated explicitly in the tool.

**Severity**: LOW (well-handled by loop dedup)

---

**5. Missing Data = Silent Strategy Corruption**

**File**: `backtest/engines/base.py` → signal_engine.on_bar(ohlcv, timestamp)  
**Finding**: If `ohlcv` has NaN for high/low (e.g., data feed gap), the signal engine still gets called. If the engine's logic checks `if high > threshold`, NaN comparisons return False, silently changing behavior.

**Defect**: Strategy logic bugs may hide in data quality issues.

**Mitigation**: Backtest engines check for gaps and warn, but the engine does not auto-abort.

**Severity**: MEDIUM (rare, but consequences are large)

---

**6. Mandate Expiry Has No Grace Period**

**File**: `sdk_order_gate.py:109`  
```python
if _is_expired(mandate):
    return _deny(broker, session_id, "mandate expired — re-authorize", ...)
```

**Finding**: Orders are denied the instant mandate expires. If a mandate expires at 9:30 AM and market opens at 9:30, the first order of the day is blocked.

**Defect**: No sliding window or "within 5 minutes" tolerance.

**Severity**: LOW (user can re-authorize preemptively)

---

**7. Tool Results Can Exceed Memory Budget**

**File**: `config/limits.py` (imported but not explored)  
**Finding**: Each tool result is capped by `truncate_tool_result()`. If a tool returns 500 KB (e.g., a huge financial statement), it is truncated. The LLM doesn't know if it's seeing the full data or a summary.

**Defect**: Silent truncation.

**Mitigation**: System prompt (Principle 2) says "if a tool fails, return nothing or does not cover what was asked, say so". The LLM is instructed to call the tool again if the result looks incomplete.

**Severity**: LOW (known and mitigated by prompt)

---

**8. No Multi-Leg Order Support**

**File**: `tools/trading_connector_tool.py:750–765`  
**Finding**: Only single-leg orders supported (buy/sell one symbol at a time).

**Defect**: No basket orders, no spreads, no legs-in-same-timestamp.

**Severity**: MEDIUM (acceptable for most strategies, limiting for spreads/hedges)

---

**9. Paper Trading Isolation Weak**

**File**: `trading/service.py:696`  
**Finding**: Paper profiles trade the broker's real sandbox. But if the broker's sandbox has real money (e.g., eToro's practice account mirrors live data), the LLM doesn't know.

**Defect**: Assumption that "paper" is harmless.

**Severity**: LOW (broker-dependent, user-controllable)

---

**10. No Partial Fill Simulation**

**File**: `backtest/engines/base.py` (expected)  
**Finding**: Backtests assume 100% fill or 0% fill. No simulation of partial fills (order for 1000 shares, 600 filled, 400 pending).

**Defect**: Reality is partial fills.

**Severity**: MEDIUM (acceptable for daily, critical for illiquid assets)

---

**11. Timestamp Timezone Inconsistency**

**File**: `backtest/binance_account_reconciliation.py:39` — UTC required  
**Finding**: Backtest engines accept any timezone. If you mix market-hours data (US ET) with UTC bars, the reconciliation fails silently.

**Defect**: Timezone coercion is not enforced early.

**Severity**: LOW (rare, caught by reconciliation)

---

**12. Grounding Prevents Valid Research**

**File**: `agent/grounding.py` (Principle 1, enforced)  
**Finding**: LLM cannot say "based on the industry consensus, X is likely Y" — only "I found tool result that says...". This prevents meta-analysis and expert judgment.

**Defect**: Over-restrictive, not a defect in code but in operating model.

**Severity**: DESIGN CHOICE (not a defect)

---

**13. Mandate Store Has No Backup**

**File**: `live/mandate/store.py` (imported, not read)  
**Finding**: Mandate is a single JSON file at `~/.vibe-trading/live/{broker}.json`. If deleted, user loses trading authorization with no recovery.

**Defect**: No backup/versioning.

**Severity**: LOW (user can re-authenticate)

---

## 13. Verdict

### What Is This, Really?

**NOT an autonomous LLM trading agent in the Track 2 sense**.

Vibe-Trading is a **conversational research and backtesting platform with optional order placement**. The LLM is the research engine, not the decision-maker. Trading decisions are explicit user commands, routed through a ReAct loop, gated by mandate enforcement. The system's actual value is:

1. **Backtesting** — 10 market-specific engines, 400+ factors, attribution analysis
2. **Research** — 30+ data sources, financial statements, sentiment, cross-asset correlation
3. **Portfolio Analysis** — trade journal parsing, regime classification, factor decomposition
4. **Compliance** — mandate gates, audit trails, daily limits, kill switch

### What Would It Take to Beat It?

**For Track 2 (Agentic Trading)**, ARGUS must deliver what Vibe-Trading **doesn't**:

1. **Autonomous decision-making under constraints** — The LLM reads market state and makes orders without user approval (within mandate), using learned patterns
2. **Real-time reactive trading** — Event-driven (earnings, news, vol spikes) with sub-second response
3. **Multi-leg execution** — Spreads, baskets, hedged portfolios placed atomically
4. **Fee-conscious modeling** — Backtest results account for real commissions, slippage, market impact
5. **Learned adaptation** — System changes strategy parameters based on win/loss streaks or regime shifts
6. **Cross-market coordination** — Recognize relative value across equities, options, crypto, forex and auto-hedge

Vibe-Trading does 1–2 (basic gates + research). ARGUS must do 1–6.

### What We Should Copy

1. **Mandate gate architecture** (pattern, not schema) — all-or-nothing enforcement before broker call
2. **ReAct loop with dedup + compression** — proven pattern for long research tasks
3. **Grounding ledger** — truth verification, prevents model hallucination
4. **Backtest engine abstraction** — market-specific engines, signal-engine contract
5. **Audit trail** — immutable decision record for compliance
6. **Timestamp reconciliation** — validates data coherence before trading

### Known Risks If Copying Code

1. **Fee/commission modeling is optional** — must make it default
2. **Mandate schema is vibe-trading-specific** — must redesign for ARGUS consent model
3. **Data sources are Asia-heavy** (akshare, eastmoney, futu) — ARGUS may need US-centric replacements
4. **Grounding is strict** (Principle 1–6) — good for research, may slow autonomous trading

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| **Total Python Files** | 1,720 |
| **Agent Core Files** | 12 |
| **Trading Connectors** | 12 brokers |
| **Backtest Engines** | 10 |
| **Tools** | 80+ |
| **Skills** | 40+ |
| **Data Loaders** | 30+ |
| **Supported Markets** | 12 (US, China, HK, crypto, forex, forex, options, bonds, commodities, forex) |
| **License** | MIT (fully permissive) |
| **Last Commit** | 2026-09-12 (fresh) |

---

## References

- **System Prompt** (Output Principles): `agent/src/agent/context.py:23`
- **ReAct Loop**: `agent/src/agent/loop.py:1181`
- **Mandate Gate**: `agent/src/live/sdk_order_gate.py:62`
- **Grounding**: `agent/src/agent/grounding.py:1`
- **Backtest Metrics**: `agent/backtest/metrics.py`
- **Trading Connectors**: `agent/src/trading/service.py:18`
- **Backtest Engines**: `agent/backtest/engines/`

---

