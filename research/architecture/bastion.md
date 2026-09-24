# ARCHITECTURE TEARDOWN: Bastion

**Autonomous, verifiable AI fund for tokenized stocks on Robinhood Chain**

A competitor analysis and dissection for ARGUS Track 2 (Agentic Trading) — Bitget Hackathon.

---

## 1. Identity

**Name:** Bastion  
**Author:** adrydevel  
**Repository:** https://github.com/adrydevel/bastion  
**Current Version:** 0.5.1 (Alpha)  
**Active Development:** Yes (as of 2026-07-19)  
**Type:** Autonomous, multi-agent, verifiable trading system for tokenized equities  

**Pitch:** "A swarm of AI agents that researches, debates, sizes and executes — then proves every trade on-chain. Non-custodial. Proven, not trusted."

**Target Asset Class:** Tokenized US stocks (NVDA, AAPL, GOOG, etc.) trading 24/7 on Robinhood Chain (Arbitrum Orbit L2)

---

## 2. Licence

**SPDX Identifier:** MIT  
**Full License:** Apache-compatible, permissive open-source.  
**Source:** `/LICENSE` file, lines 1-21.

MIT is the most permissive license — Bastion's code can be **copied, modified, relicensed, and commercialized** without restriction. ARGUS can freely incorporate any mechanism from Bastion's codebase.

---

## 3. Full Architecture

### 3.1 Data Flow Diagram

```
ORACLE QUOTE
    ↓
[Chainlink Feed] → latestRoundData() ← reads StockTokenQuote (price, ts)
    ↓
REGIME DETECTION (regime/detector.ts)
    ├─ log-return drift over window
    ├─ rolling volatility (stdev of log-returns)
    ├─ classify: trend_up | trend_down | chop | high_vol
    └─ feed to BANDIT for strategy allocation
    ↓
COUNCIL DEBATE (swarm/council.ts)
    ├─ [5 Agents] (swarm/agents/) run in parallel:
    │  ├─ Analyst (price structure, trend analysis)
    │  ├─ Sentiment (news, social flow)
    │  ├─ Risk (downside, liquidity, correlation)
    │  ├─ Contrarian (refute consensus)
    │  └─ Executor (fill quality, slippage, costs)
    ├─ Each agent:
    │  ├─ receives: MarketFrame { ticker, quotes[24], regime }
    │  ├─ calls LLM with system prompt + last 16 prices
    │  ├─ parses response: side (long/short/flat) + confidence [0,1]
    │  └─ returns: AgentOpinion
    ├─ Tally votes by confidence-weighted sum
    ├─ Check quorum (default 3/5)
    └─ Average confidence of winning side
    ↓
RISK KERNEL (risk/)
    ├─ [Kelly Sizing] Kelly fraction: f* = (p·w - (1-p)) / w
    │  └─ Confidence [0.5..0.75] → Win Prob [0.5..0.75]
    │  └─ Payoff fixed at 1.4 (assumed odds)
    │  └─ Half-Kelly applied: position = min(f/2, maxKelly=0.25)
    │
    ├─ [CVaR Tail Cap] Expected shortfall on return distribution
    │  └─ Sorts returns, takes tail below alpha (default 95%)
    │  └─ Returns mean of tail → cvar limit 8% (default)
    │  └─ NOT actively enforced in current code; computed but not applied
    │
    └─ [Circuit Breaker] Max drawdown from peak
       └─ Tracks peak equity, halts if drawdown > 15% (default)
    ↓
PROOF GENERATION (proof/attest.ts)
    ├─ Canonical JSON serialization (keys sorted, undefined dropped)
    ├─ Keccak256 hash of verdict
    ├─ Keccak256 hash of feature window
    └─ Timestamp + optional txHash
    ↓
VERIFICATION / REPLAY (replay/verify.ts)
    ├─ Re-derive decision from cassette (config + quotes + transcript)
    ├─ Check 5 stages:
    │  1. Input hash (tamper detection)
    │  2. Regime classification
    │  3. Verdict (side, size, quorum)
    │  4. Features hash
    │  5. Verdict hash
    └─ Report PASS/FAIL per stage
    ↓
ON-CHAIN ANCHORING (chain/robinhood.ts, proof/onchain.ts)
    ├─ Submit verdictHash + featuresHash to Robinhood Chain
    ├─ CURRENTLY STUBBED: returns hash as-is, no actual write
    ├─ Hosted runtime connects real signer
    └─ Non-custodial: keys stay client-side
    ↓
EXECUTION (chain/robinhood.ts)
    └─ CURRENTLY STUBBED: no actual swaps
       (routed through on-chain DEX aggregator in hosted tier)
```

### 3.2 Module Graph

```
entry: index.ts
  └─ CLI: cli/commands.ts
      ├─ run: fetches quotes → frameFromQuotes → assembleCouncil
      │        → council.decide() → attest() → anchor() → printVerdict
      │        → saveCassette (if --record)
      │
      └─ replay: resolveCassette → replayCassette()
                  ├─ recompute input hash
                  ├─ recompute regime
                  ├─ re-run council with ReplayDeck (transcript lookup)
                  ├─ check verdict, proof hashes
                  └─ print results

types.ts
  └─ Shared: Ticker, StockTokenQuote, MarketFrame, Regime, Side,
             AgentOpinion, Verdict, Proof

config.ts
  └─ ConfigSchema (Zod)
     ├─ rpcUrl (default: https://rpc.robinhood-chain.xyz)
     ├─ chainId (default: 64288)
     ├─ universe (default: [NVDA, AAPL, GOOG, MSFT, AMZN, META, TSLA])
     ├─ Risk params: maxKelly, cvarLimit, maxDrawdown
     ├─ Swarm params: quorum
     ├─ LLM: provider ("hermes" | "offline")
     └─ attestOnchain (boolean)

swarm/
  ├─ debate.ts: assembleCouncil() → wires 5 agents + resolveProvider
  ├─ council.ts: Council class
  │    ├─ decide(frame) → Promise<Verdict>
  │    ├─ Polls all agents in parallel
  │    ├─ Tallies by confidence-weighted sum
  │    ├─ Applies Kelly sizing
  │    └─ Returns Verdict (side, size, quorum, opinions, regime)
  │
  └─ agents/
      ├─ base.ts: Agent abstract class
      │    ├─ opine(frame) → AgentOpinion
      │    ├─ Sends system prompt + last 16 prices to LLM
      │    └─ Parses response for side + confidence
      │
      ├─ analyst.ts: "reads price structure and trend"
      ├─ sentiment.ts: "weighs news and social flow"
      ├─ risk.ts: "argues downside: liquidity, gap, correlation"
      ├─ contrarian.ts: "actively tries to refute consensus"
      └─ executor.ts: "judges fill quality, slippage, costs"

regime/
  ├─ detector.ts: detectRegime(quotes) → Regime
  │    ├─ Computes log-return drift & volatility
  │    ├─ vol > 2% → high_vol
  │    ├─ drift > vol*0.5 → trend_up
  │    ├─ drift < -vol*0.5 → trend_down
  │    └─ else → chop
  │
  └─ bandit.ts: ThompsonBandit
       ├─ Maintains Beta(alpha, beta) for each arm
       ├─ pick(ids) → samples from each arm, returns best
       └─ reward(id, win) → updates alpha or beta

risk/
  ├─ kelly.ts
  │  ├─ kellyFraction(winProb, payoff)
  │  │  └─ f* = (p·w - (1-p)) / w
  │  │
  │  └─ sizePosition(winProb, payoff, maxKelly)
  │     └─ half_kelly = min(f/2, maxKelly)
  │
  ├─ cvar.ts
  │  ├─ valueAtRisk(returns, alpha=0.95)
  │  │  └─ tail quantile
  │  │
  │  └─ cvar(returns, alpha=0.95)
  │     └─ mean of tail below (1-alpha)*len
  │
  └─ circuitBreaker.ts: CircuitBreaker
       └─ update(equity) → { tripped, drawdown }

memory/
  └─ reflexive.ts: ReflexiveMemory
       ├─ remember(episode) → stores verdict + pnl + postmortem + vector
       ├─ recall(vector, k=5) → cosine-ranked top k episodes
       └─ Max 5000 episodes in memory

proof/
  ├─ attest.ts: attest(verdict, features) → Proof
  │  ├─ Hashes verdict with keccak256(canonical(verdict))
  │  ├─ Hashes features with keccak256(canonical(features))
  │  └─ Returns { verdictHash, featuresHash, attestedAt, txHash? }
  │
  └─ onchain.ts: anchor(chain, proof) → Promise<Proof>
       └─ Calls chain.writeProof(verdictHash, featuresHash)

replay/
  ├─ canonical.ts: canonical(value) → string
  │  ├─ Recursive sort of object keys
  │  ├─ Array order preserved
  │  ├─ Undefined members dropped
  │  └─ digest(value) → keccak256(toHex(canonical(value)))
  │
  ├─ cassette.ts: Cassette interface
  │  ├─ schema, version, recordedAt
  │  ├─ config, input (ticker, quotes)
  │  ├─ exchanges[] (agent, system, user, response)
  │  ├─ verdict, proof, inputHash
  │  ├─ saveCassette(c) → JSON file in .bastion/cassettes/
  │  ├─ loadCassette(path) → parse & validate schema
  │  └─ resolveCassette(ref) → by hash prefix or path
  │
  ├─ recorder.ts
  │  ├─ Recorder: wraps LLM, intercepts calls, logs Exchange
  │  └─ ReplayDeck: serves recorded responses, throws on miss
  │
  └─ verify.ts: replayCassette(cassette, opts) → ReplayResult
      ├─ Recomputes input hash
      ├─ Checks regime
      ├─ Re-runs council with ReplayDeck
      ├─ Checks verdict
      ├─ Checks feature hash
      ├─ Checks verdict hash
      └─ Returns { ok, checks[], live }

providers/
  ├─ hermes.ts: HermesProvider (OpenAI-compatible)
  │  ├─ Default model: Hermes-4-70B
  │  ├─ Default endpoint: https://inference-api.nousresearch.com/v1
  │  ├─ Override: BASTION_BASE_URL / BASTION_API_KEY env vars
  │  └─ reason(system, user) → Promise<string> (model response)
  │
  ├─ offline.ts: OfflineProvider (deterministic heuristic)
  │  ├─ No network, no model, no key required
  │  ├─ 5 lenses: momentum, flow, tail, fade, fill
  │  ├─ Score(drift, vol) based on lens
  │  └─ Confidence = 0.5 + min(|score|, 1) * 0.4
  │
  └─ router.ts: resolveProvider(provider) → LLM
      └─ Routes to Hermes or Offline based on config

chain/
  ├─ oracle.ts: readQuote(client, ticker, feed, token) → StockTokenQuote
  │  ├─ Calls Chainlink latestRoundData()
  │  ├─ Decodes: roundId, answer, startedAt, updatedAt, answeredInRound
  │  ├─ Price: answer / 1e8
  │  ├─ Timestamp: updatedAt * 1000 (unix ms)
  │  └─ Returns: { ticker, address, price, ts }
  │
  └─ robinhood.ts: RobinhoodChain class
     ├─ client: PublicClient (viem)
     ├─ writeProof(verdictHash, featuresHash) → txHash (STUBBED)
     └─ execute(ticker, side, size) → void (STUBBED)
```

### 3.3 Agent Swarm Roster

| Agent | Role | System Prompt | What it Reads |
|-------|------|---------------|--------------|
| **Analyst** | Price structure & trend | "reads price structure and trend on the tokenized-stock oracle series" | Last 16 prices |
| **Sentiment** | News & social flow | "weighs news and social flow around the underlying equity" | Last 16 prices (but meant to integrate external sentiment data) |
| **Risk** | Downside arguments | "argues the downside: liquidity, gap risk, correlation to the book" | Last 16 prices |
| **Contrarian** | Consensus refutation | "actively tries to refute the emerging consensus" | Last 16 prices |
| **Executor** | Fill & cost quality | "judges fill quality, slippage and whether the edge survives costs" | Last 16 prices |

**Key observation:** All 5 agents receive the SAME input (last 16 prices + regime). They "disagree" only because they're routed through 5 different system prompts and the LLM sampling is non-deterministic (temp=0.2). The **Sentiment agent in particular is not wired to any news/sentiment data source**—it only gets prices. This is a major gap.

### 3.4 Main Loop

**Entry point:** `cli/commands.ts:buildProgram()`

**`bastion run` flow:**
1. Load config (or use defaults)
2. Load demo quotes (if offline, deterministic; else would read from oracle)
3. Build council: assemble 5 agents + chosen provider
4. Call `council.decide(frame)` → Promise<Verdict>
5. Call `attest(verdict, features)` → Proof
6. Call `anchor(chain, proof)` → Proof with txHash (if `attestOnchain=true`)
7. Print verdict
8. If `--record`: build cassette, save to `.bastion/cassettes/`, print hash

**`bastion replay` flow:**
1. Load cassette (by hash prefix or path)
2. Call `replayCassette(cassette)`:
   - Recompute input hash, check match
   - Recompute regime, check match
   - Re-run council with ReplayDeck (serves recorded LLM responses)
   - Recompute verdict, check match
   - Recompute proof hashes, check match
3. Print checks (each: name, ok, expected, actual)
4. Exit code 0 if all pass, 1 otherwise

---

## 4. THE DECISION PATH: Where the LLM is Called

### 4.1 LLM Invocation Points

**File:** `src/swarm/agents/base.ts:Agent.opine()`

```typescript
async opine(frame: MarketFrame): Promise<AgentOpinion> {
  const user = JSON.stringify({
    ticker: frame.ticker,
    regime: frame.regime,
    window: frame.quotes.slice(-16).map((q) => q.price),
  });
  const raw = await this.llm.reason(this.system(), user);
  return this.parse(raw);
}
```

**System prompt template** (each agent overrides `system()` method):
```
You are the [AGENT NAME] on an autonomous trading desk that trades tokenized
US stocks 24/7 on Robinhood Chain. Your role: [ROLE DESCRIPTION].
Reply with a side (long/short/flat), a confidence in [0,1] as
'confidence: x', and one sentence of evidence. Be decisive but honest;
abstain (flat) when the edge is not there.
```

**User input** (JSON):
```json
{
  "ticker": "NVDA",
  "regime": "trend_up",
  "window": [100.5, 101.2, 100.8, 101.5, ...]  // last 16 prices
}
```

### 4.2 Response Parsing

**File:** `src/swarm/agents/base.ts:Agent.parse()`

```typescript
protected parse(raw: string): AgentOpinion {
  const side = /short/i.test(raw) ? "short" : /long/i.test(raw) ? "long" : "flat";
  const conf = Number((raw.match(/conf(?:idence)?[:=\s]+([01]?\.?\d+)/i) ?? [])[1] ?? 0.5);
  return {
    agent: this.name,
    side,
    confidence: Math.max(0, Math.min(1, conf)),  // clamp to [0,1]
    rationale: raw.slice(0, 280),  // first 280 chars
  };
}
```

**Parsing logic:** REGEX-based, loose. Looks for keywords "short", "long" and a pattern like "confidence: 0.75". No schema enforcement, no structured output, no validation that confidence is a valid number (defaults to 0.5 if missing or unparseable).

### 4.3 Verdict Computation

**File:** `src/swarm/council.ts:Council.decide()`

```typescript
async decide(frame: MarketFrame): Promise<Verdict> {
  const opinions = await Promise.all(this.agents.map((a) => a.opine(frame)));
  const { side, votes } = tally(opinions);

  let size = 0;
  if (votes >= this.cfg.quorum && side !== "flat") {
    const conf = avgConfidence(opinions, side);
    // Confidence [0..1] → winProb [0.5..0.75]
    size = sizePosition(0.5 + conf * 0.25, 1.4, this.cfg.maxKelly);
  }

  return {
    ticker: frame.ticker,
    side: votes >= this.cfg.quorum ? side : "flat",
    size,
    quorum: votes,
    opinions,
    regime: frame.regime,
  };
}

function tally(opinions: AgentOpinion[]): { side: Side; votes: number } {
  const buckets: Record<Side, number> = { long: 0, short: 0, flat: 0 };
  // SUM confidence for each side
  for (const o of opinions) buckets[o.side] += o.confidence;
  // Pick side with highest total confidence
  const side = (Object.keys(buckets) as Side[]).reduce((a, b) =>
    buckets[a] >= buckets[b] ? a : b,
  );
  // Count votes (agents that chose the winning side)
  const votes = opinions.filter((o) => o.side === side).length;
  return { side, votes };
}

function avgConfidence(opinions: AgentOpinion[], side: Side): number {
  const inFavour = opinions.filter((o) => o.side === side);
  if (inFavour.length === 0) return 0;
  return inFavour.reduce((s, o) => s + o.confidence, 0) / inFavour.length;
}
```

**Key detail:** Quorum is **vote count**, not confidence-weighted. Decision requires `votes >= config.quorum` (default 3). If only 2 agents say "long", the desk goes flat—even if both are 100% confident.

### 4.4 Is the LLM Deciding or Narrating?

**VERDICT: The LLM is DECIDING, with deterministic gates.**

- ✓ Each agent's opinion directly influences the final Verdict.side via vote tally
- ✓ No hardcoded logic overrides an LLM's decision
- ✓ The LLM's confidence directly affects Verdict.size (via Kelly formula)
- ✓ BUT the Kelly sizing formula, quorum thresholds, and CVaR/circuit-breaker gates are all DETERMINISTIC policy—they don't call the LLM

**What the policy kernel does:**
1. LLM generates 5 opinions (5 side votes)
2. Tally determines side by confidence-weighted sum
3. **Deterministic logic:** Check quorum, apply Kelly sizing, cap by CVaR/drawdown
4. Return Verdict

The LLM is not overridden by the kernel—rather, the kernel **enforces risk limits after** the LLM decides. If the LLM says "go 50% long" but Kelly sizing caps it at 20%, the desk trades 20%. The LLM is not consulted again.

---

## 5. THE RISK KERNEL: The Deep Dive

### 5.1 Kelly Sizing

**File:** `src/risk/kelly.ts`

```typescript
export function kellyFraction(winProb: number, payoff: number): number {
  if (payoff <= 0) return 0;
  const loseProb = 1 - winProb;
  const f = (winProb * payoff - loseProb) / payoff;
  return Number.isFinite(f) ? Math.max(0, f) : 0;
}

export function sizePosition(
  winProb: number,
  payoff: number,
  maxFraction: number,
): number {
  const raw = kellyFraction(winProb, payoff);
  // Half-Kelly by default
  return Math.min(raw * 0.5, maxFraction);
}
```

**Formula:** 
- **Full Kelly:** f* = (p·w - (1-p)) / w = (p·w - 1 + p) / w = (p·(w+1) - 1) / w
- **Half-Kelly:** f = f* / 2 (in implementation: `raw * 0.5`)
- **Capped:** f_capped = min(f, maxKelly)

**Used in council:**
```typescript
const conf = avgConfidence(opinions, side);  // avg confidence of winning side
size = sizePosition(0.5 + conf * 0.25, 1.4, cfg.maxKelly);
```

**Input mapping:**
- Confidence [0..1] → winProb [0.5..0.75]
- Payoff: always 1.4 (fixed assumption)
- maxFraction: 0.25 (default, 25% of equity per position)

**Example:**
- Conf = 1.0 → p = 0.75, w = 1.4
  - f* = (0.75·1.4 - 0.25) / 1.4 = (1.05 - 0.25) / 1.4 = 0.8 / 1.4 ≈ 0.571
  - Half-Kelly: 0.571 / 2 ≈ 0.286
  - Capped: min(0.286, 0.25) = 0.25 → **25% of equity**

- Conf = 0.0 → p = 0.5, w = 1.4
  - f* = (0.5·1.4 - 0.5) / 1.4 = (0.7 - 0.5) / 1.4 ≈ 0.143
  - Half-Kelly: ≈ 0.071
  - Capped: **7.1% of equity**

**Verification against Rockafellar-Uryasev:** The Kelly formula here is textbook (Poundstone). The half-Kelly application is standard for bias reduction. No verification against a specific research paper claimed; this is bog-standard Kelly sizing.

**Gaps:**
1. **No negative-edge handling:** If `f*` is negative (losing proposition), function returns 0. Correct.
2. **Payoff fixed at 1.4:** Assumes risk-reward is always 1.4:1. No dynamic adjustment for volatility, spread, or actual oracle prices.
3. **No fractional-Kelly justification:** Code applies half-Kelly without citing research or computing optimal fraction for this specific edge distribution.

### 5.2 Conditional Value at Risk (CVaR)

**File:** `src/risk/cvar.ts`

```typescript
export function valueAtRisk(returns: number[], alpha = 0.95): number {
  if (returns.length === 0) return 0;
  const sorted = [...returns].sort((a, b) => a - b);
  const idx = Math.floor((1 - alpha) * sorted.length);
  return -(sorted[idx] ?? sorted[0] ?? 0);
}

export function cvar(returns: number[], alpha = 0.95): number {
  if (returns.length === 0) return 0;
  const sorted = [...returns].sort((a, b) => a - b);
  const cutoff = Math.max(1, Math.floor((1 - alpha) * sorted.length));
  const tail = sorted.slice(0, cutoff);
  const mean = tail.reduce((s, r) => s + r, 0) / tail.length;
  return -mean;
}
```

**Formula:**
- **VaR(α):** The (1-α) quantile of losses (negated for positive number)
  - Sorted returns, take index `floor((1-α)·n)`, negate
  - For α=0.95, n=100: idx=5, so VaR is the 5th-worst return (negated)

- **CVaR(α):** Mean of losses worse than VaR
  - Tail = returns[0..cutoff) in sorted order (most negative)
  - Return -mean(tail)

**Verification against Rockafellar-Uryasev (2000):**
- Rockafellar-Uryasev defines CVaR(α) = E[X | X ≤ VaR(α)]
  - This is the expected value of the tail below the α-quantile
  - Equivalent to mean of the worst (1-α)·n% of returns

- **Bastion's implementation:** Correct for finite samples. The code takes the tail below the quantile and averages—exactly Rockafellar-Uryasev.
- **Assumption:** Discretized distribution (empirical). For continuous distributions, CVaR would require integral; here it's sum/n, which is valid for backtests and historical return samples.

**Gaps:**
1. **CVaR is computed but NOT enforced:** `cvar()` function exists and is tested, but it is **never called** in the trading loop. The risk kernel applies Kelly sizing and circuit breaker (drawdown), but there's no code that says "if CVaR > limit, reduce size" or "skip this trade."
   - File: `src/config.ts` line 11: `cvarLimit: z.number().min(0).max(1).default(0.08),`
   - That's read but never used in `src/swarm/council.ts` or `src/risk/`.
   
   **Status:** UNIMPLEMENTED. CVaR limit is configured but dormant.

2. **No return history:** CVaR requires a sample of historical returns. Bastion doesn't maintain position history per session—a single `run` command makes one decision and exits. There's no rolling return series to compute CVaR on.

### 5.3 Regime Detection

**File:** `src/regime/detector.ts`

```typescript
export function detectRegime(quotes: StockTokenQuote[]): Regime {
  if (quotes.length < 8) return "chop";
  const rets: number[] = [];
  for (let i = 1; i < quotes.length; i++) {
    const p0 = quotes[i - 1]!.price;
    const p1 = quotes[i]!.price;
    if (p0 > 0) rets.push(Math.log(p1 / p0));
  }
  const mean = rets.reduce((s, r) => s + r, 0) / rets.length;
  const variance = rets.reduce((s, r) => s + (r - mean) ** 2, 0) / rets.length;
  const vol = Math.sqrt(variance);

  if (vol > 0.02) return "high_vol";
  if (mean > vol * 0.5) return "trend_up";
  if (mean < -vol * 0.5) return "trend_down";
  return "chop";
}
```

**Classification rules:**
1. `vol > 2%` → high_vol (regardless of drift)
2. `drift > 0.5·vol` → trend_up
3. `drift < -0.5·vol` → trend_down
4. else → chop

**Input:** Window of 24 price quotes (1 hour of data if 1 sample every 2.5 min, or 24 hours if once per hour—not specified).

**Verification:** No research cited. The thresholds (2%, 0.5·vol ratio) are heuristic.

### 5.4 Thompson Sampling Bandit

**File:** `src/regime/bandit.ts`

```typescript
export class ThompsonBandit {
  private arms = new Map<string, Arm>();

  private arm(id: string): Arm {
    let a = this.arms.get(id);
    if (!a) { a = { alpha: 1, beta: 1 }; this.arms.set(id, a); }
    return a;
  }

  private sample(a: Arm, rand: () => number): number {
    const g = (k: number) => {
      let s = 0;
      for (let i = 0; i < k; i++) s += -Math.log(rand() || 1e-9);
      return s;
    };
    const x = g(Math.max(1, Math.round(a.alpha)));
    const y = g(Math.max(1, Math.round(a.beta)));
    return x / (x + y);
  }

  pick(ids: string[], rand: () => number = Math.random): string {
    let best = ids[0]!;
    let bestScore = -1;
    for (const id of ids) {
      const s = this.sample(this.arm(id), rand);
      if (s > bestScore) { bestScore = s; best = id; }
    }
    return best;
  }

  reward(id: string, win: boolean): void {
    const a = this.arm(id);
    if (win) a.alpha += 1; else a.beta += 1;
  }
}
```

**Algorithm:** Thompson sampling on Beta-Bernoulli conjugate prior
- Each arm has Beta(α, β) posterior over win probability
- `sample()` draws from Beta by gamma-ratio trick
- `pick()` samples from each arm, returns arm with highest sample
- `reward(id, win)` updates: wins → α++, losses → β++

**Starting point:** Beta(1, 1) = Uniform [0,1] (uninformative prior)

**Usage in codebase:** The bandit is defined but **NOT CALLED**. No code routes the Sentiment agent to bandit allocation or re-picks strategies per regime.

**Status:** DEFINED but UNINTEGRATED. Like CVaR, the bandit exists as plumbing but is disconnected.

### 5.5 Circuit Breaker

**File:** `src/risk/circuitBreaker.ts`

```typescript
export class CircuitBreaker {
  private peak = 0;
  constructor(private readonly cfg: Config) {}

  update(equity: number): { tripped: boolean; drawdown: number } {
    this.peak = Math.max(this.peak, equity);
    const drawdown = this.peak > 0 ? (this.peak - equity) / this.peak : 0;
    return { tripped: drawdown >= this.cfg.maxDrawdown, drawdown };
  }
}
```

**Rule:** If `(peak - current) / peak >= maxDrawdown`, halt trading.

**Default:** maxDrawdown = 15%

**Usage in codebase:** The CircuitBreaker class is defined but **NOT CALLED** in the trading loop either.

**Status:** DEFINED but UNINTEGRATED.

### Summary: The Working vs. Dormant Pieces

| Mechanism | Implemented? | Integrated? | Status |
|-----------|--------------|-------------|--------|
| **Kelly Sizing** | ✓ | ✓ | **ACTIVE**: Applied to every trade |
| **CVaR** | ✓ (computed) | ✗ | Dormant—never enforced |
| **Regime Detection** | ✓ | ✓ | **ACTIVE**: Classified, fed to agents |
| **Thompson Bandit** | ✓ (defined) | ✗ | Dormant—never called |
| **Circuit Breaker** | ✓ (defined) | ✗ | Dormant—never called |

**Implication:** Bastion's actual risk control is **only Kelly sizing and regime detection**. The Kelly cap is 25% per position, half-Kelly applied, payoff fixed at 1.4:1.

---

## 6. Cryptographic Proof and Deterministic Replay

### 6.1 What is Signed and Hashed

**NOT cryptographically signed.** (Signatures planned for hosted TEE runtime; not in open-source version.)

**What IS hashed with keccak256:**

**File:** `src/proof/attest.ts`

```typescript
export function attest(verdict: Verdict, features: unknown): Proof {
  return {
    verdictHash: digest(verdict),
    featuresHash: digest(features),
    attestedAt: Date.now(),
  };
}
```

**digest() function** (src/replay/canonical.ts):

```typescript
export function digest(value: unknown): `0x${string}` {
  return keccak256(toHex(canonical(value)));
}
```

**Canonical serialization** (src/replay/canonical.ts):

```typescript
export function canonical(value: unknown): string {
  if (value === undefined) return "null";
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value) ?? "null";
  }
  if (Array.isArray(value)) {
    return "[" + value.map(canonical).join(",") + "]";
  }
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([, v]) => v !== undefined)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return (
    "{" +
    entries.map(([k, v]) => JSON.stringify(k) + ":" + canonical(v)).join(",") +
    "}"
  );
}
```

**Two hashes:**

1. **verdictHash = keccak256(canonical(Verdict))**
   - Verdict object: { ticker, side, size, quorum, opinions[], regime }
   - Each AgentOpinion: { agent, side, confidence, rationale }
   - Keys are sorted at every depth
   - No whitespace
   - undefined members dropped
   
2. **featuresHash = keccak256(canonical(features))**
   - features = { window: [prices...] }
   - The price window that was hashed

### 6.2 Deterministic Replay

**Entry point:** `src/replay/verify.ts:replayCassette()`

```typescript
export async function replayCassette(
  cassette: Cassette,
  opts: { llmFor?: (agent: string) => LLM } = {},
): Promise<ReplayResult> {
  const checks: Check[] = [];
  const live = Boolean(opts.llmFor);

  // 1. Cassette integrity
  const recomputedInput = inputHashOf(cassette);
  checks.push({
    name: "cassette integrity",
    ok: recomputedInput === cassette.inputHash,
    expected: cassette.inputHash,
    actual: recomputedInput,
  });

  // 2. Regime
  const frame = frameFromQuotes(cassette.input.quotes);
  checks.push({
    name: "regime",
    ok: frame.regime === cassette.verdict.regime,
    expected: cassette.verdict.regime,
    actual: frame.regime,
  });

  // 3. Verdict
  const llmFor = opts.llmFor ?? ((agent: string) => new ReplayDeck(cassette.exchanges).for(agent));
  const council = assembleCouncil(cassette.config, llmFor);
  let verdict;
  try {
    verdict = await council.decide(frame);
  } catch (err) {
    checks.push({
      name: "verdict",
      ok: false,
      expected: `${cassette.verdict.side} ${cassette.verdict.size.toFixed(4)} q${cassette.verdict.quorum}`,
      actual: err instanceof Error ? err.message : String(err),
    });
    return { ok: false, checks, live };
  }

  checks.push({
    name: "verdict",
    ok: canonical(verdict) === canonical(cassette.verdict),
    expected: `${cassette.verdict.side} ${cassette.verdict.size.toFixed(4)} q${cassette.verdict.quorum}`,
    actual: `${verdict.side} ${verdict.size.toFixed(4)} q${verdict.quorum}`,
  });

  // 4. Proof hashes
  const proof = attest(verdict, { window: frame.quotes.map((q) => q.price) });
  checks.push({
    name: "features hash",
    ok: proof.featuresHash === cassette.proof.featuresHash,
    expected: cassette.proof.featuresHash,
    actual: proof.featuresHash,
  });
  checks.push({
    name: "verdict hash",
    ok: proof.verdictHash === cassette.proof.verdictHash,
    expected: cassette.proof.verdictHash,
    actual: proof.verdictHash,
  });

  return { ok: checks.every((c) => c.ok), checks, live };
}
```

**Five checks performed:**

1. **cassette integrity:** Recompute inputHash from (config, quotes, exchanges)
2. **regime:** Re-classify regime from quotes
3. **verdict:** Re-run council with recorded transcript, compare all fields
4. **features hash:** Re-hash price window
5. **verdict hash:** Re-hash verdict object

**ReplayDeck** (src/replay/recorder.ts): Serves recorded LLM responses

```typescript
export class ReplayDeck {
  constructor(private readonly exchanges: Exchange[]) {}

  for(agent: string): LLM {
    return {
      reason: async (system: string, user: string) => {
        const hit = this.exchanges.find(
          (e) => e.agent === agent && e.system === system && e.user === user,
        );
        if (!hit) {
          throw new Error(
            `no recorded exchange for "${agent}" — this cassette was produced by a different pipeline`,
          );
        }
        return hit.response;
      },
    };
  }
}
```

**Key insight:** Replay looks up exchanges by (agent, system, user). If a cassette was recorded with old prompts or code, replay will fail with "no recorded exchange" rather than silently using a new LLM call. This **prevents tampering by model drift.**

### 6.3 What Can Actually Be Verified

**PROVED by Bastion's verification:**

✓ **Cassette integrity:** No one edited config, prices, or LLM responses after recording (keccak256 hash match).

✓ **Verdict consistency:** Given these inputs and these model outputs, the current pipeline produces the same verdict.

✓ **Proof hash integrity:** The verdict object hashes to the same keccak256 as recorded.

✓ **Feature hash integrity:** The price window hashes to the same keccak256 as recorded.

✓ **Regime classification stability:** The regime detector still classifies the price window the same way.

**NOT proved:**

✗ **Model determinism:** The model will never produce different outputs. (Hermes at temp=0.2 is very stable but not deterministic.)

✗ **On-chain anchoring:** In open-source version, `writeProof()` is stubbed. The hash is computed but not actually written to Robinhood Chain. Hosted version would sign with TEE; open-source doesn't.

✗ **Authenticity:** Who ran the decision? No signature. A TEE-signed version (hosted tier) would certify the enclave that produced it.

### 6.4 Determinism of Replay

**PROVED DETERMINISTIC:** Yes, 100%.

**Given the same cassette, same code version, same machine:**
- Regime classification will re-derive the same regime (deterministic algorithm).
- Council will call ReplayDeck, which returns the same recorded responses (deterministic lookup).
- Verdict will be re-computed identically (deterministic algorithm: tally votes, apply Kelly).
- Proofs will hash identically (keccak256 is deterministic).

**Test coverage:** `src/replay/replay.test.ts` verifies:
- Replay reproduces recorded decision exactly
- Is stable across repeated replays
- Fails when verdict is edited
- Fails when model response is edited
- Fails when market window is edited
- Recomputes the same input hash

---

## 7. TOKENIZED-STOCK SPECIFICS: Session State, Market Hours, Oracle Freshness

### 7.1 Market Hours Handling

**SEARCH RESULT:** No market-hours awareness anywhere in the codebase.

**Grep for keywords:** market, session, hour, close, open, weekend, NYSE, gap, stale, fresh, nav.

**Findings:**

1. **Regime detector** (`src/regime/detector.ts`): Operates on a window of quotes. No check for whether the window spans a market closure, a gap to the next open, or weekend data.

2. **Agent prompts** (`src/swarm/agents/*.ts`): All reference "trade tokenized US stocks 24/7 on Robinhood Chain." The 24/7 claim is **on Robinhood Chain**—not on underlying US equity markets. None of the agent prompts mention market hours, trading halts, or underlying market state.

3. **Oracle reading** (`src/chain/oracle.ts`): Reads Chainlink latestRoundData() and takes whatever timestamp it provides. **No staleness check.** No logic like:
   - "If timestamp is more than X seconds old, wait or reject."
   - "If it's 2am ET and the US markets are closed, what does this price represent?"
   - "Is the NAV stale due to no trading in the underlying?"

4. **Config** (`src/config.ts`): No market-hours configuration. Default universe: `["NVDA", "AAPL", "GOOG", "MSFT", "AMZN", "META", "TSLA"]`—no exclusions for market-hours state.

5. **Circuit breaker, regime, Kelly sizing:** None reference market state.

**Implications for ARGUS:**
- Bastion **assumes oracle prices are always fresh and valid**, even outside US trading hours.
- Bastion **assumes the price window doesn't span a closure**, so drift/vol calculations are correct.
- Bastion **has no awareness of NAV staleness**—if the underlying US stock closed at 3pm ET and it's now 11pm ET, the oracle price may be yesterday's close, but the code treats it as "current."
- Bastion **trades 24/7 on Robinhood Chain** but **does not model the mismatch between 24/7 on-chain prices and 6.5-hour US market openings.**

### 7.2 Oracle Freshness

**Evidence:** None. No staleness checks.

**What could happen:**
1. Oracle provider (Chainlink) publishes a price update at 8am ET (market open).
2. Bastion reads it at 8am ET → decision based on fresh price. ✓
3. Bastion runs again at 4am ET (US market closed, will reopen in 2.5 hours).
4. Oracle still publishes the same price or a slightly stale price.
5. Bastion reads it → **decision based on potentially stale NAV, treating it as live.**
6. Bastion sizes a position for 24/7 trading on Robinhood Chain, but the underlying won't move until 9:30am ET.

### 7.3 Gap Risk and Session State

**NONE MODELED.**

The Risk agent's system prompt mentions "gap risk" (`"argues the downside: liquidity, gap risk, correlation to the book"`) but the agent receives only the last 16 prices—no calendar context, no session markers, no knowledge of whether prices have gapped overnight.

**Example scenario:**
- Last 16 prices: [100.0, 100.5, 101.0, 101.2, ..., 101.8] (intra-session, steady trend)
- Regime → trend_up (positive drift, low vol)
- Agents see trend_up → vote long
- But unknown to agents: the next price bar will open 8 hours from now after a US market close overnight
- Overnight, a news event hits: earnings miss, SEC filing, geopolitical shock
- US market opens with a 5% gap down
- Bastion's long position (sized for trend_up in calm conditions) immediately underwater

Bastion has **zero logic to flag or hedge this risk.**

### 7.4 What Bastion DOES Handle

✓ **On-chain oracle reads:** Correctly reads Chainlink latestRoundData via viem and decodes the 8-decimal fixed-point price.

✓ **24/7 trading model:** The code explicitly supports trading tokenized stocks any time, day or night.

✓ **Non-custodial keys:** Config specifies RPC endpoint; no key storage in code.

### 7.5 Session State Summary

| Aspect | Handled? | Evidence |
|--------|----------|----------|
| **Market hours detection** | ✗ | No NYSE/market-hours logic. |
| **Oracle staleness check** | ✗ | Just reads latestRoundData, no `updatedAt` validation. |
| **NAV freshness** | ✗ | No concept of underlying-market-close time. |
| **Gap risk awareness** | ✗ | Risk agent doesn't know when the next gap might open. |
| **Session state in regime** | ✗ | Regime detector just looks at 24 quotes in isolation. |
| **Intraday vs. overnight** | ✗ | All price bars treated equally. |

**MAJOR GAP FOR ARGUS:** Bastion trades 24/7 on-chain but **does not model the 6.5-hour US market open**. ARGUS must add session awareness, staleness checks, and gap-risk modeling to avoid massive overnight drawdowns.

---

## 8. Memory: Reflexive Recall

**File:** `src/memory/reflexive.ts`

### 8.1 Architecture

```typescript
export interface Episode {
  verdict: Verdict;
  pnl: number;
  postMortem: string;
  vector: number[];  // embedding for similarity search
}

export class ReflexiveMemory {
  private episodes: Episode[] = [];

  remember(ep: Episode): void {
    this.episodes.push(ep);
    if (this.episodes.length > 5000) this.episodes.shift();  // FIFO eviction
  }

  recall(vector: number[], k = 5): Episode[] {
    return [...this.episodes]
      .map((e) => ({ e, s: cosine(e.vector, vector) }))
      .sort((a, b) => b.s - a.s)
      .slice(0, k)
      .map((x) => x.e);
  }
}
```

### 8.2 What is Stored

Each episode captures:
- **verdict:** The decision made (side, size, quorum, opinions, regime)
- **pnl:** Realized P&L from the closed position
- **postMortem:** Text analysis of why the trade succeeded or failed
- **vector:** Embedding for cosine-similarity matching (not generated by code; external dependency)

### 8.3 How Similarity is Computed

Cosine distance between vectors:

```typescript
function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! ** 2;
    nb += b[i]! ** 2;
  }
  const d = Math.sqrt(na) * Math.sqrt(nb);
  return d === 0 ? 0 : dot / d;
}
```

Result: scalar in [-1, 1], higher = more similar.

### 8.4 Integration Gap

**DEFINED but NOT CALLED.** The ReflexiveMemory class is implemented but **never instantiated or used** in the trading loop.

- No code calls `remember(ep)` after a trade closes.
- No code calls `recall(vector)` to retrieve similar past states.
- The postMortem is never generated.
- The vector embedding source is unspecified (no model defined to generate it).

**Status:** Infrastructure without integration, like CVaR and the bandit.

### 8.5 Implications

If integrated, reflexive memory could:
- Feed past similar-outcome episodes back to agents as examples
- Surface "we saw this regime before and here's what happened"
- Enable few-shot learning by analogy

But in open-source Bastion, it's a **placeholder for future work.**

---

## 9. Sensing the Environment: Data Sources and Timestamping

### 9.1 Data Sources

**Oracle:** Robinhood Chain's Chainlink price feed.

**Code:** `src/chain/oracle.ts:readQuote()`

```typescript
const data = (await client.readContract({
  address: feed,
  abi: AGGREGATOR_ABI,
  functionName: "latestRoundData",
})) as readonly [bigint, bigint, bigint, bigint, bigint];
return {
  ticker,
  address: token,
  price: Number(data[1]) / 1e8,
  ts: Number(data[3]) * 1000,
};
```

**Data returned:**
- `roundId` (uint80) — Chainlink round identifier
- **`answer` (int256) — Price, scaled by 1e8** (e.g., $150.00 stored as 15_000_000_000)
- `startedAt` (uint256) — Timestamp (seconds) when aggregation started
- **`updatedAt` (uint256) — Timestamp (seconds) when answer was last updated** ← **Oracle freshness marker**
- `answeredInRound` (uint80) — Which round ID the answer came from

**Scaling:** Price is decoded as `data[1] / 1e8`, so 8 decimal places for fixed-point.

### 9.2 Timestamping

**Sources of timestamps:**

1. **Oracle timestamp** (`updatedAt` from Chainlink): Seconds since Unix epoch, **converted to milliseconds** in the code (`ts: Number(data[3]) * 1000`).

2. **Regime window:** 24 quotes, each with `ts`. No requirement that they're evenly spaced; just assumed to be recent.

3. **Demo mode:** Timestamps are synthetic:
   ```typescript
   ts: Date.now() - (24 - i) * 60000,  // 24 samples, 1 min apart
   ```

### 9.3 Point-in-Time (PIT) Protection

**NONE.** No code checks:
- "Is this oracle price as-of a specific block number?"
- "Is the price from the same block as the config settings?"
- "Could the config have been updated between oracle read and decision?"

The oracle timestamp is purely informational; the code doesn't use it for versioning or validation.

### 9.4 Staleness Validation

**NONE.** No logic like:
```typescript
const now = Date.now();
const age = now - oracleTs;
if (age > 60000) {  // older than 1 min
  throw new Error("oracle price too stale");
}
```

The code happily accepts oracle prices of any age.

### 9.5 Summary

| Aspect | Implemented? | Status |
|--------|--------------|--------|
| **Oracle integration** | ✓ | Reads Chainlink via viem |
| **Timestamp capture** | ✓ | Stored with each quote |
| **Staleness check** | ✗ | No validation of age |
| **PIT versioning** | ✗ | No block number tracking |
| **Config stability** | ✗ | No check that config didn't change mid-run |

---

## 10. Order Placement: Venue, Costs, Paper vs. Live

### 10.1 Order Placement Code

**File:** `src/chain/robinhood.ts:execute()`

```typescript
async execute(ticker: string, side: Side, size: number): Promise<void> {
  if (side === "flat" || size <= 0) return;
  // Routed through the on-chain DEX aggregator in the hosted runtime.
}
```

**Current Status:** **STUBBED.** Empty implementation. No actual orders are placed.

### 10.2 Venue

**Intended:** Robinhood Chain (Arbitrum Orbit L2) DEX aggregator.

**Actual:** None.

### 10.3 Paper Trading vs. Live

**Open-source:** Paper trading only (stubs return success without executing).

**Hosted runtime:** Would execute real orders; keys stay client-side (non-custodial model).

### 10.4 Cost Modeling

**NONE.** No code mentions:
- Maker/taker fees
- Slippage
- Spreads
- Gas costs (for on-chain execution)
- Order impact

The Executor agent's prompt says "judges fill quality, slippage and whether the edge survives costs" but receives only prices. It cannot compute slippage or costs without historical bid-ask data or a price impact model.

### 10.5 Implied Assumption

The Kelly formula uses a fixed payoff of 1.4:1, which implicitly assumes:
- A 40% expected return per unit of risk (e.g., risk $1 to gain $0.40).
- Costs are immaterial, or already factored into the 1.4:1 ratio.

If costs are 0.5% round-trip and average edge is 0.1%, then the true payoff is **negative**—Kelly would return 0. But the code assumes 1.4:1 unconditionally.

---

## 11. Self-Evaluation: Does the LLM Grade Its Own Output?

**SEARCH:** No self-scoring loop in the codebase.

**Evidence:**
- No agent calls the LLM to grade another agent's opinion.
- No meta-agent that evaluates the council's consensus.
- No loop where agents rate confidence in their own rationale.

**Agents generate:** Side + confidence [0..1] + rationale (free text).

**No agent generates:** A grade or critique of its own output.

**Status:** **UNIMPLEMENTED.** Self-evaluation is not a feature of open-source Bastion.

---

## 12. Per Track-2 Sub-Theme Inventory

**Track 2 (Agentic Trading)** sub-themes from hackathon rules:

1. **Event-driven trading** (earnings, announcements, news)
   - **Bastion:** Sentiment agent references "news and social flow" but is not wired to any news source. Receives only prices.
   - **File:line:** `src/swarm/agents/sentiment.ts:8-9`
   - **Evidence:** System prompt mentions news; implementation reads only prices.
   - **Conclusion:** DECLARED but NOT IMPLEMENTED.

2. **Sentiment analysis**
   - **Bastion:** Same as above. Sentiment agent exists; data source missing.
   - **File:line:** `src/swarm/agents/sentiment.ts`
   - **Conclusion:** PARTIALLY IMPLEMENTED (agent exists, data integration missing).

3. **Cross-asset execution**
   - **Bastion:** Single-ticker per run. No portfolio rebalancing, no cross-asset hedging.
   - **File:line:** `src/cli/commands.ts:39` (`--ticker <ticker>`)
   - **Conclusion:** NOT IMPLEMENTED.

4. **Factor discovery**
   - **Bastion:** Agents are hand-coded; no discovery mechanism.
   - **Conclusion:** NOT IMPLEMENTED.

5. **Agent evaluation / performance scoring**
   - **Bastion:** Agents generate opinions; no meta-scoring of agent quality over time.
   - **Conclusion:** NOT IMPLEMENTED.

**Summary:** Bastion addresses 0.5/5 Track-2 sub-themes (sentiment declared, not wired).

---

## 13. STEAL LIST: High-Value Mechanisms

| Mechanism | File:Line | Why Good | Disposition (ARGUS) |
|-----------|-----------|----------|-------------------|
| **Canonical Hashing** | src/replay/canonical.ts:9-24 | Keys sorted at every depth, undefined dropped. Ensures replay proofs don't break under refactoring. Elegant & simple. | **STEAL:** Copy this pattern. Session replay is non-negotiable; canonical JSON is the solution. |
| **Replay Verification** | src/replay/verify.ts:34-98 | Five-stage integrity check (cassette, regime, verdict, features, proof). Hard fail on transcript miss (no silent fallthrough to live model). | **STEAL:** Core audit mechanism. Must-have for autonomous agents. |
| **Thompson Bandit** | src/regime/bandit.ts | Clean implementation of Thompson sampling. Gamma-ratio trick for Beta sampling. Correct conjugate prior. | **STEAL:** If integrating regime-based strategy allocation. Unintegrated in Bastion but code is sound. |
| **Kelly Sizing** | src/risk/kelly.ts:6-10 | Textbook formula, correct. Half-Kelly applied. Cap enforced. | **STEAL:** Standard; check Poundstone. Bastion's fixed payoff of 1.4:1 is the gap to address. |
| **CVaR Implementation** | src/risk/cvar.ts:10-16 | Correct empirical CVaR (mean of tail). Rockafellar-Uryasev verified. | **STEAL:** Code is correct; Bastion just doesn't use it. Integrate into risk checks. |
| **Regime Classification** | src/regime/detector.ts:6-22 | Fast, interpretable regime labels (trend_up, trend_down, chop, high_vol). No HMM overhead. | **STEAL:** Simple and effective. Extend to market-hours state (e.g., add "pre-market", "post-market", "weekend"). |
| **Agent Swarm Voting** | src/swarm/council.ts:14-34 | Quorum requirement (default 3/5). Confidence-weighted tally. No single agent decides. | **STEAL:** Good governance model. Extend: add agent reliability weighting (agents with better historical accuracy get higher votes). |
| **Deterministic Heuristic** | src/providers/offline.ts:13-30 | Five lenses (momentum, flow, tail, fade, fill) rated by drift+vol. Reproducible without a model. | **STEAL:** Valuable for testing/audit. FNV-1a hash ensures agent gets same lens per run. |
| **Cassette Format** | src/replay/cassette.ts:20-30 | Minimal, self-contained. Config + quotes + exchanges + verdict + proof + inputHash. Schema version for forward compatibility. | **STEAL:** Design pattern for audit logs. Add: agent reliability scores, hyperparameter experiments, A/B test tags. |
| **Proof Anchoring Stub** | src/proof/onchain.ts:7-10 | Placeholder that documents the intended on-chain write. TEE-signed version goes here in hosted tier. | **STEAL:** Architecture is right. For ARGUS: implement actual writes to Bitget contracts, with signatures from a trusted oracle (e.g., Chainlink VRF or Pyth). |

---

## 14. WHAT BREAKS: Defects Found by Code Reading

| Defect | File:Line | Severity | Impact |
|--------|-----------|----------|--------|
| **CVaR computed but never used** | src/config.ts:11; src/risk/cvar.ts; src/swarm/council.ts | Medium | Config offers `cvarLimit` but it's ignored. Position sizes are never capped by CVaR. |
| **Circuit breaker defined but never called** | src/risk/circuitBreaker.ts; cli/commands.ts | Medium | Max drawdown limit is configured but never enforced. A 20% drawdown would not halt trading. |
| **Thompson bandit not integrated** | src/regime/bandit.ts; cli/commands.ts | Low | Regime is detected and passed to agents, but not fed into bandit for strategy selection. Bandit never calls `pick()` or `reward()`. |
| **Sentiment agent not wired to sentiment data** | src/swarm/agents/sentiment.ts; src/swarm/agents/base.ts | Medium | Agent references "news and social flow" but receives only prices. No API for news/social integration. |
| **No market-hours awareness** | src/chain/oracle.ts; src/regime/detector.ts | High | Treats prices as valid 24/7. No check if underlying US market is open, closed, gapped, or if oracle is stale. |
| **Oracle staleness never validated** | src/chain/oracle.ts:22-39 | High | Reads `updatedAt` but never checks it. An oracle update from 8 hours ago is treated as fresh. |
| **No NAV-freshness concept** | src/types.ts (StockTokenQuote); src/swarm/agents/*.ts | High | Zero awareness of whether an on-chain token's NAV matches the underlying US stock (which closed 14 hours ago). |
| **Replay LLM miss is hard error, but no recovery** | src/replay/verify.ts:64-73 | Low | If a cassette was made with old prompts, replay fails with "no recorded exchange." Users must re-record. No graceful degradation. |
| **Agent parsing is regex-based and fragile** | src/swarm/agents/base.ts:20-28 | Medium | `confidence: 0.75` is parsed but `confidence=0.75` or `conf: 75%` would fail. Defaults to 0.5 silently. |
| **No input validation on agent responses** | src/swarm/agents/base.ts:20-28 | Low | Agent could return "buy Tesla" and parser would default side to "flat". No schema enforcement. |
| **Payoff is hardcoded to 1.4** | src/swarm/council.ts:22 | High | Kelly formula always assumes 1.4:1 payoff. No adjustment for volatility, realized returns, or cost. Unrealistic assumption. |
| **Price window size is fixed at 16** | src/swarm/agents/base.ts:14 | Low | Last 16 prices are always used. No configurable lookback. Agents can't see longer trends. |
| **Regime classifier needs 8+ quotes to work** | src/regime/detector.ts:6 | Low | If window has <8 quotes, returns "chop" unconditionally. Startup bias. |
| **Reflexive memory evicts by FIFO, not quality** | src/memory/reflexive.ts:16-19 | Low | When 5000 episodes are stored, the oldest is dropped, even if it's the most successful. No score-based eviction. |
| **Proof is not actually anchored on-chain** | src/chain/robinhood.ts:14-20 | Critical | `writeProof()` returns the input hash unchanged. No transaction is sent to Robinhood Chain. Hosted runtime would implement this, but open-source does not. |
| **Execution is not actually executed** | src/chain/robinhood.ts:22-25 | Critical | `execute()` is a stub. No swaps are placed. Only paper trading works. |

---

## 15. HEAD-TO-HEAD: ARGUS vs. Bastion

### Capability Comparison Table

| Capability | Bastion | What ARGUS Must Do to Beat It | ARGUS Status |
|-----------|---------|-------------------------------|-------------|
| **Agent Swarm** | 5 agents (Analyst, Sentiment, Risk, Contrarian, Executor) voting by quorum | Scale swarm with per-agent reliability weighting. Add domain experts (Macro, Micro, Volatility, Vol-of-Vol). Integrate sentiment via real news/social APIs. Add self-critique loop. | Implement + integrate sentiment APIs |
| **Risk Kernel: Kelly Sizing** | Half-Kelly, 25% cap, fixed 1.4:1 payoff | Estimate payoff dynamically from realized returns. Use true edge (current vol vs. model confidence). Implement Bayesian update of expected return after each trade. Add slippage and cost model. | Dynamic payoff + costs |
| **Risk Kernel: CVaR** | Implemented, not enforced | Enforce CVaR limit. Compute on rolling window of closed positions. Adjust position size to keep expected tail loss below limit. | Integrate + enforce |
| **Risk Kernel: Circuit Breaker** | Defined, not enforced | Implement + test. Halt at 15% max drawdown. Add soft circuit (reduce position size) at 10% drawdown. | Integrate + add soft breaker |
| **Regime Detection** | 4 states (trend_up, trend_down, chop, high_vol) based on drift/vol | Expand to 6+ states: add pre-market, post-market, weekend, low-volume, high-uncertainty. Use HMM or regime filter (Bayes). Model state transitions. | Extended regime space + HMM |
| **Regime Allocation** | Thompson bandit defined, not used | Integrate: route strategies per regime, learn arm effectiveness, rebalance. Track per-regime Sharpe ratio. | Integrate Thompson + track Sharpe |
| **Deterministic Replay** | ✓ Full cassette replay, 5-stage verification, canonical hashing | Match this exactly. Add: agent scoring history, hyperparameter audit trail, A/B test labels. | Copy pattern + extend |
| **Cryptographic Proof** | keccak256 hash of verdict + features, not on-chain | Integrate Bitget's chain (or use Pyth/Chainlink for attestation). Implement TEE-signed proof in hosted tier. | On-chain anchoring |
| **Market-Hours Awareness** | ✗ Zero | Add UTC → ET session mapper. Check "is market open?". Reject stale oracle data. Model gap risk. Flag pre-market/post-market trades separately. | Session mapper + gap risk |
| **Oracle Freshness** | Read `updatedAt`, never validate | Reject prices older than 60s (or configurable). Warn if oracle is on low-volume chain. Check Chainlink heartbeat. | Staleness thresholds |
| **NAV Tracking** | ✗ None | Model US stock market open/close. Compute spread between on-chain token NAV and underlying. Flag when spread widens. Hedge if gap > threshold. | NAV tracker + spread hedge |
| **Event-Driven Trading** | Sentiment agent, no data source | Wire Sentiment to: news APIs (Finnhub, NewsAPI), earnings calendars (Polygon), SEC filings, Twitter/X sentiment. Score event impact vs. price momentum. | News APIs + earnings calendar |
| **Cost Modeling** | ✗ None (fixed 1.4:1 payoff) | Measure actual maker/taker fees on Robinhood Chain DEX. Model spreads per symbol. Estimate gas costs. Subtract from Kelly expectancy. | Fee + spread model |
| **Portfolio Rebalancing** | ✗ Single-ticker per run | Support multi-ticker positions. Rebalance weekly/daily. Implement long-short pairs. Track correlation. | Multi-position portfolio |
| **Sentiment Integration** | Not implemented | Real sentiment sources (news, social, options flow). Weight by source reliability. | News + social APIs |
| **Memory & Learning** | Reflexive memory defined, not used | Integrate: store post-mortems, recall similar states, feed back to agents. Score memory by predictive power. | Integrate + score memory |
| **Self-Critique** | ✗ None | Meta-agent that rates agent opinions post-hoc. Adjust agent weights based on forecast accuracy. Hallucination detection. | Meta-evaluation agent |
| **Backtest Harness** | Roadmap item; not implemented | Build deterministic backtest engine. Run 1000+ decisions, compute Sharpe, max DD, skew. Validate strategy before deploy. | Backtest harness |
| **Version Control & Audit** | Cassette format (manual) | Git + merkle tree of all decisions. Automated audit logs. Hyperparameter experiment tracking. | Decision DAG + audit log |

### Summary

**Bastion's Strengths (ARGUS must match or exceed):**
1. **Deterministic replay + cryptographic proof** — This is the gold standard. ARGUS must implement it.
2. **Canonical hashing** — Clever and simple. Copy it.
3. **Quorum voting** — Governance model is sound. Extend with agent reliability weighting.
4. **Regime classification** — Fast, interpretable. Extend to market-hours state.
5. **Offline heuristic** — Valuable for testing without APIs. Build something similar.

**Bastion's Weaknesses (ARGUS can clearly win):**
1. **No market-hours awareness** — CRITICAL GAP. ARGUS adds session state, staleness checks, NAV tracking.
2. **CVaR defined but not enforced** — ARGUS enforces it.
3. **Circuit breaker defined but not enforced** — ARGUS enforces it.
4. **Sentiment agent not wired to sentiment data** — ARGUS integrates real news/sentiment APIs.
5. **No cost modeling** — ARGUS models fees, spreads, gas costs.
6. **Payoff is hardcoded** — ARGUS estimates payoff dynamically.
7. **No self-evaluation** — ARGUS implements meta-agent scoring.
8. **Thompson bandit not integrated** — ARGUS integrates and tracks per-regime performance.
9. **No event-driven trigger** — ARGUS integrates earnings calendar + news alerts.
10. **Execution is stubbed** — ARGUS implements real swaps (or routes through DEX API).

---

## SUMMARY

### Word Count
~4,900 words (excluding tables, section headers, and code blocks)

### Market-Hours Handling
**DO THEY HANDLE MARKET HOURS/SESSION STATE?**  
**No.** Bastion has **zero awareness of US market hours, session state, oracle staleness, NAV freshness, or gap risk.** The code treats all on-chain prices as fresh and valid 24/7. This is a **critical gap** for tokenized-stock trading.

### 5 Highest-Value Steal-List Items for ARGUS

1. **Canonical Hashing + Deterministic Replay** (`src/replay/canonical.ts`, `src/replay/verify.ts`)
   - Why: Non-negotiable audit trail. Protects against post-hoc strategy tweaking. Elegant design.
   - How: Copy the canonical JSON pattern and 5-stage replay verification.

2. **Quorum Voting + Confidence-Weighted Tally** (`src/swarm/council.ts`)
   - Why: Clean governance model. No single agent moves capital. Scalable to more agents.
   - How: Steal the voting logic. Extend with agent reliability weighting (agents with better track records get higher votes).

3. **Thompson Sampling Bandit** (`src/regime/bandit.ts`)
   - Why: Correct implementation. Solves the explore-exploit tradeoff for regime-based strategy allocation.
   - How: Copy the code. Integrate into regime detection (pick best strategy per regime, learn from outcomes).

4. **Offline Deterministic Heuristic** (`src/providers/offline.ts`)
   - Why: Enables testing without APIs or keys. Reproducible across machines. 5 lenses give reasonable diversity.
   - How: Build a similar heuristic for ARGUS. Use it to test replay, prove pipeline correctness, debug production issues.

5. **CVaR Implementation** (`src/risk/cvar.ts`)
   - Why: Correct empirical CVaR (matches Rockafellar-Uryasev). Code is sound but unused in Bastion.
   - How: Copy the implementation. Integrate into position sizing: reject trades where expected tail loss exceeds limit.

---

**Report complete.**

