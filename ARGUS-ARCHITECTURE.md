# ARGUS — System Architecture

## Three products · one engine · all eighteen sub-themes

**Status:** build specification, 2026-09-12. Supersedes `ARGUS-TRACK2-ARCHITECTURE.md`.
**Grounded in:** 25 code-level architecture teardowns (`research/architecture/`, 160,411 words) ·
the consolidated build ledger (129 mechanisms, 103 defects, 24 absent capabilities) · 922-source
research corpus · `ARGUS-MASTER-PRD.md` (18,900 words).

---

## 0. The shape of the whole thing

```text
                         ONE ENGINE
   Truth · Evidence · Intelligence · Risk · Execution · Learning · Authority
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ALPHA FACTORY       TRADING GOVERNOR       RESEARCH OS
     (Track 1)            (Track 2)            (Track 3)
        │                     │                     │
  strongest            strongest             strongest
  STATISTICAL          AUTONOMOUS            RESEARCH
  MACHINE              DECISION-MAKER        EXPERIENCE
        │                     │                     │
  scored 100%          scored 50/50          scored by judges;
  quantitatively       quant/judge           human decides
```

**The engine is shared. The products are not.** Track 1 submits a strategy whose Sharpe must survive
costs. Track 2 submits an agent whose decisions must be provably its own. Track 3 submits a
workspace whose every number must be traceable in one click. Building one and re-pointing it at the
other two is the most likely way to lose all three.

### The six levels — universal across all three products

| Level | Question | Who answers it |
|---|---|---|
| **1 — Trust** | What was knowable at this instant? | Temporal Truth Fabric. A gate, not a feature |
| **2 — Intelligence** | What does it mean? | Six intelligence engines |
| **3 — Decision** | What should be done? | Track 1: the optimiser. Track 2: the LLM. Track 3: the human |
| **4 — Execution** | What actually happened? | Execution Truth Engine |
| **5 — Learning** | Was I right, and *why* — thesis, timing, execution or luck? | Autopsy and attribution |
| **6 — Authority** | Do I deserve to do this again? | Capital authority state machine |

Three invariants, enforced in code:

1. **Level 2 may never write into Level 1.** No intelligence module can revise a timestamp, promote
   a source's credibility, or restate what was knowable — otherwise every leakage defence is one
   prompt injection away from being switched off.
2. **Level 6 is derived, never self-asserted.** An agent cannot argue its way into a larger
   position. It can only be *measured* into one.
3. **A Level 1 failure halts the pipeline.** `DATA_INSUFFICIENT` is a distinct verdict from
   `NO_TRADE`: the first says the evidence was inadequate, the second says it was adequate and the
   answer was no. Conflating them hides the system's most important failure mode.

---

## 1. The shared engine

### 1.1 Level 1 — the Temporal Truth Fabric

Every fact carries five times, not one: `event_time`, `publish_time`, `ingest_time`,
`available_at`, and a revision chain. Retrieval is bounded by `as_of` **at the storage interface**,
not by caller discipline — a caller that omits the bound gets an error, never a silent leak.

**This is the most violated rule in the field**, and the teardowns proved it with `file:line`.
FinMem populates `temp_date_list` at `memorydb.py:169` and `195` and never checks it before ranking,
so a decision on date *t* can retrieve memories from *t+100*. FinAgent does an unbounded
`similarity_search` (`memory/basic_memory.py:62`) and — worse — computes `days_future = now + 14`,
putting future state directly into the observation. Both are widely cited architectures.

**The two-clock model is ours.** Every PIT implementation found across 922 sources assumes a single
market clock. A tokenized equity has two, and they disagree for 65.5 hours a week: a fact can be
tradeable on the token's clock while unpriceable on the anchor's. A single-clock design cannot
represent that.

*Taken:* Qlib's `P` operator for expression-level forward-reference blocking (`pit.py:24–49`);
TradingAgents' FRED vintage-pinning and centralised UTC-normalised date-window filter — genuinely
good, and the correction to our earlier note, which called v0.4.0 leakage-prone when five of its
seven sources are properly protected. Its one real gap is Polymarket, which is real-time only; we
must not reproduce that shape.

### 1.2 Level 1 — Session State Machine

Per instrument, per venue: session phase, NAV freshness, oracle age, hours to next genuine price
discovery, and the live hedgeability surface.

**The fact that justifies the entire system:** Bastion — an autonomous AI fund for tokenized stocks,
the closest thing to ARGUS that exists — has **zero** session awareness. No market hours, no NAV
staleness, no oracle freshness, no gap risk; every on-chain price is treated as fresh 24/7. Its
advertised risk machinery is also largely dormant: CVaR is computed and never enforced in sizing,
the circuit breaker never fires, the sentiment agent is wired to no news source, and execution is
stubbed.

### 1.3 Level 1 — Evidence validity

Every extracted number keeps a locator to its exact position in the source. Docling preserves
cell-level `BoundingBox {l,t,r,b}` in page coordinates (`table_structure_model.py:143–150`, MIT) —
adopted directly, and the foundation for Track 3's evidence lineage.

Citations are **bound, not self-reported**. Agent-Rita builds them from the data actually injected
during the loop, tracked in a `citedWidgets` map keyed by widget UUID plus args
(`round-trip.ts:743–754`, MIT): the model physically cannot cite outside that set. LLM
self-reported citations are the failure mode we refuse.

*Correction to a widely repeated claim:* Docling's "chart understanding" yields **descriptions, not
numeric extraction**. A chart may never sit on an evidence path terminating in a number.

### 1.4 The spine — `DecisionContext`

One append-only object threads every level, in all three products. It is also the audit artefact: a
third party replaying a decision needs this and nothing else.

```text
DecisionContext
├── identity      decision_id · parent_id · created_at · schema_version
├── clocks        TWO always — token_clock, anchor_clock — plus as_of
├── session       per instrument: phase · nav_state · oracle_age · hours_to_discovery
├── evidence[]    claim · source · event/publish/ingest/available_at · credibility
│                 · locator (page, bbox) · lineage_hash
├── hypotheses[]  thesis · evidence_ids · return DISTRIBUTION · falsifier (mandatory)
├── dissent[]     the strongest surviving counter-case, carried intact
├── hedgeability  ranked hedge menu · priced residual · Risk Neutralisation Efficiency
├── risk          marginal VaR/CVaR of THIS trade · gap risk · concentration deltas
├── decision      verdict ∈ {TRADE, REDUCE, HEDGE, DELAY, NO_TRADE,
│                            HUMAN_REVIEW, DATA_INSUFFICIENT}
│                 · size · stated_confidence · invalidation_conditions[]
├── authority     granted envelope · regime · expiry
├── constitution  verdict ∈ {ALLOW, RESIZE, REQUIRE_HEDGE, DELAY, REJECT, FLATTEN}
│                 · binding constraint · machine-readable reason · signature
├── execution     plan · orders[] · fills[] · shortfall · reconciliation state
├── trust         fused: contamination · evidence quality · PIT integrity
│                 · calibration · consistency
└── outcome       interval score · per-link causal grades
                  · edge decomposition {information, timing, execution, luck}
```

Rules: append-only within a decision — a module adds fields, never rewrites another's. Every numeric
field carries the id of the code path that computed it, because *"the LLM interprets, code
calculates"* is only enforceable if you can point at the calculator. The object is content-hashed at
each level boundary, so a divergent replay tells you *which level* diverged.

### 1.5 Levels 4–6 — shared by all three products

**Execution Truth Engine.** A formal order state machine where **a timeout is not a rejection** and
`UNKNOWN` is first-class. Nautilus is the reference — **LGPLv3, so we link the compiled library and
never vendor it**, rebuilding the state machine in simplified form. Fill realism from hftbacktest
(MIT, copy freely): its exact condition for a resting buy limit — best ask drops below the order
price, *or* a trade prints below it, *or* a trade prints at it **and** the queue model shows
`front_qty <= 0` — is the standard we hold ourselves to, with its `ProbQueueModel` family and
`IntpOrderLatency`.

**Two production traps to avoid, both found in shipped engines.** Nautilus backtests at **zero fees**
unless configured, silently overstating P&L. Qlib measures turnover as **gross notional, not net
delta**, overstating cost 2–3× on mean-reversion, and its defaults are A-share shaped (open 0.15%,
close 0.25%, `exchange.py:48–51`) — they must be replaced with Bitget's 0.12% round-trip, never
inherited. **Across 103 catalogued defects, cost-blindness is the single most widespread failure
class in the field** — ahead of leakage, ahead of self-scoring. Our cost model is
*constructed-mandatory*: a zero-fee backtest must be impossible to instantiate.

**Learning.** Post-trade autopsy grading thesis, timing, execution, risk-model correctness, per-link
causal accuracy, agent contribution, calibration and repeated-error detection.

**Authority is probabilistic, not a score.** Not *"agent score = 87"* but *"probability this agent is
competent in this regime = 0.84, from n observations"* — small samples mean wide uncertainty means
low authority, resolving as evidence accumulates. Clawock's beta-binomial shrinkage of stated versus
realised confidence, grouped by driver and regime (`decision/settlement.py`, MIT), is the estimator.

---

## 2. TRACK 2 — the Autonomous Trading Governor

> The AI decides. ARGUS proves whether it should be trusted.

### 2.1 The positioning test, and why most of the field fails it

Bitget's rule: *the LLM is the primary trading decision-maker, not just an assistant; the Agent must
sense the environment, make independent judgments, and autonomously place orders with risk
controls.* Three requirements — **sense**, **judge independently**, **place orders under risk
control**. Most of the recognised field satisfies at most two.

| System | LLM decides? | Evidence | Track 2 |
|---|---|---|---|
| AI-Trader (HKUDS) | **No** | LLM writes market-summary prose; the trading path never consults it (`routes_signals.py`) | **Fails**, despite advertising "100% fully-automated agent-native" |
| Vibe-Trading (HKUDS) | **No** | System prompt forbids recommendations; orders human-gated | **Fails** — a research platform |
| inalpha | **No, by design** | LLM deliberately off the order path; execution needs a separate approval token (`trade-plan.ts:74–303`) | **Fails**, and is honest about it |
| FinRobot | **No** | Verified by grep — no broker client, no order path, no execution | **Fails** — a report generator |
| Bastion | Partial | Council votes; execution stubbed, CVaR unenforced | **Incomplete** |
| TradingAgents | **Yes** | Genuine multi-agent decision, emitted as a recommendation | **Partial** — no autonomous placement |
| **atrx-demo** | **Yes** | Three-tier LLM validation on a live account, 600+ trades since Nov 2025 | **Passes** — our most serious rival on this test |

These systems are not weak — several are far more mature than us. They answer a *different question*
than this track asks. That is the honest framing, and the only defensible one.

### 2.2 The Constitution asymmetry — how both halves of the rule hold at once

Typed intent → policy (OPA) → feasibility (Z3/CVXPY) → signed authorisation. Verdicts: `ALLOW`,
`RESIZE`, `REQUIRE_HEDGE`, `DELAY`, `REJECT`, `FLATTEN`.

**The Constitution may only ever reduce.** It can shrink a trade, demand a hedge, delay it, or
refuse it. It can never create one, never flip a side, never size up. So the economic choice stays
the LLM's and the risk layer stays non-negotiable — simultaneously. Every alternative design either
lets risk logic quietly become the real trader, or leaves risk arguable by prompt.

Risk rules live in versioned configuration, never in prompts — inalpha enforces this at an HTTP
boundary with DB-backed locks (`risk_guard.py:87–153`), and vibe-trading's mandate gate
(`sdk_order_gate.py:62–182`) is fail-closed with the audit write *before* the broker call.

**Licence-driven build decision:** cvxportfolio is **GPL**, as is open-trading-platform — vendoring
either forces our stack open. The Kernel is built on raw CVXPY (Apache). We rebuild two cvxportfolio
ideas: `SimulatorCost` (`costs.py:347–451`), which evaluates the *same* expression with forecast
parameters for the optimiser and realised parameters for the simulator — structurally closing the
optimizer/simulator gap; and γ = 1.5 market impact with convexity enforced at ≥ 1.0
(`costs.py:826`, `845–848`), the classical square-root law.

### 2.3 The six sub-theme engines

Each consumes one `DecisionContext` and returns structured intelligence. **None of them decides.**

| Sub-theme | ARGUS engine | Strongest prior art | Genuinely ours |
|---|---|---|---|
| **Event-driven** | **Event-to-Position Transmission Engine** — event → mechanism → affected variable → industry → asset → magnitude → horizon → *already priced?* → alternative explanation → trade. Each link graded independently afterwards | EventEdge — 12 event classes, correct MacKinlay machinery (*proprietary, all rights reserved: rebuild from MacKinlay 1997*) | Per-link causal grading. A correct call reached through three wrong links is a **failure** our scorer catches and a P&L scorer records as success. EventEdge's LLM is advisory-only and runs once daily |
| **Sentiment** | **Sentiment Integrity Graph** — author, source, account age, historical accuracy, bot probability, coordination, novelty, independent-source count, market confirmation, persistence | FinBERT (baseline); TradingAgents v0.4.0, rewritten upstream after its analyst fabricated Reddit content it never fetched | We do not try to own the best sentiment *model*. We own the best sentiment **trust system**: separating "people are bullish" from "new independent information credibly changed expectations." **Incremental Sentiment Value** is the metric, and if it does not beat raw sentiment after fees, this subsystem is demoted |
| **Earnings** | **Expectation Surface** — seven independent surprises (reported, consensus, guidance, narrative, valuation, management credibility, Q&A) mapped to a forward return *distribution*. Plus the **Earnings Contradiction Detector** | CARAG; VerumTrade peer read-through (`peer_read_through.py:62–146`) | The contradiction test is the killer feature: *EPS +12%, revenue +4%, guidance −7%, margins −250bps, management confidence +2%, consensus expected +8%* must resolve to **"headline beat, fundamental deterioration"** — not "beat → buy" |
| **Cross-asset execution** | **Hedge Optimizer Under Unavailable Markets.** Given rNVDA long, NYSE shut, BTC open, QQQ/SOXX/NVDA all unavailable: what is the cheapest *available* instrument that removes the most relevant risk right now? | hftbacktest · ABIDES · Nautilus · JAX-LOB · RL-LOB · Hummingbot · Almgren-Chriss | The agent's chosen hedge is scored against the **mathematical optimum over the feasible set**. That turns "did the LLM choose well?" into a measurable question — which is exactly what Track 2 is scored on |
| **Factor discovery** | **Factor Lineage Graph** + the 16-state lifecycle ending in `RETIRE`, with a **Factor Cemetery** | RD-Agent · Alpha-Jungle MCTS · FactorMiner | See §2.4 |
| **Open Theme** | **The Observatory** — §2.6 | TraderBench · live-trade-bench · StockBench · KTD-Fin | Eight capabilities absent from every harness torn down |

### 2.4 Search/evaluation separation, and the Factor Cemetery

Three independent instances of the same contamination, in code:

- `mcts-llm-alpha` computes a real IS/OOS overfitting score at `qlib_evaluator.py:143`, then
  **unconditionally overwrites it** with the generating LLM's self-judgment (`comprehensive.py:90–98`).
- **RD-Agent** — Microsoft's, the strongest factor machinery in existence — uses a single
  `APIBackend()` to both propose and judge. No independent evaluator. **No purged CV, no embargo, no
  deflated Sharpe, no PBO, no trial counter.** Capacity, decay and retirement absent. It also shows
  `without_cost` numbers to both proposer and evaluator while feedback reads `with_cost`.
- **FactorForge** feeds the generator the IC of the top 3 factors each round and asks for variations
  (`evolution_engine.py:95–99`).

**FactorMiner is the counter-example and gets the credit**: its generator sees only syntax errors,
never scores (`factor_generator.py:91–112`). We cite it as prior art rather than claiming separation
as ours.

What remains ours: the **trial counter** and **cost-aware IC gating**, which none has. A search that
runs hundreds of trials, reports the best, and never records how many it ran is not a result — the
trial count *is* part of the result, and our DSR gate consumes it.

**The Factor Cemetery makes this demonstrable rather than rhetorical.** The reportable artefact is a
funnel, not a headline:

```text
10,000 discovered → 1,843 failed formalisation → 211 survived OOS
    → 17 survived capacity → 4 deployed → 2 decayed → 1 retired
```

That is far harder to fake than *"our AI generated 5,000 alphas."*

### 2.5 The Five Proof Systems

The architecture is already large enough. What it needs is not more features but **five artefacts
that convert claims into machine-checkable evidence**. These are Track 2's headline deliverables.

#### A. Autonomy Proof — *did the LLM actually decide this trade?*

Every trade emits a signed chain:

```text
market_state_hash
llm_original_intent        ← the model's unconstrained economic choice
constitution_modification  ← what the risk layer changed, and which constraint bound
llm_revised_intent         ← the model's response to being constrained
approved_intent            ← signed
submitted_order
exchange_order_id
fills[]
reconciliation_state
```

The `llm_original_intent` / `llm_revised_intent` pair is the load-bearing part: it shows the model
making an economic choice *and* responding to a constraint, which no amount of architecture diagram
can demonstrate. A judge gets a machine-verifiable answer, not a narrative.

#### B. Counterfactual Twin — *does the LLM add value?*

Every decision runs in **five worlds on identical state**:

| World | Decision-maker |
|---|---|
| A | ARGUS LLM |
| B | Deterministic ARGUS, no LLM |
| C | Human |
| D | A different LLM |
| E | No trade |

**Incremental Agent Value = outcome(A) − outcome(best counterfactual).** World B is the one that
matters: if the LLM does not beat its own deterministic twin on identical data, we report that. An
honest negative on B is more credible than an unexplained positive on A — and it is the only real
proof that the LLM is the decision-maker, which is precisely what §2.1 demands.

#### C. Execution Proof — *is the executed order the approved order?*

Approved intent hash ↔ submitted order ↔ exchange acknowledgement ↔ fills ↔ position delta, with
every mismatch surfaced rather than reconciled away. Partial fills re-plan the residual; a failed
hedge leg resizes the position rather than leaving it naked.

#### D. Competence Proof — *is authority earned?*

The promotion/demotion ledger: every authority change with the evidence that caused it, the regime
it applies to, its expiry, and the posterior competence estimate with its sample size.

#### E. Benchmark Proof — *is it actually better?*

§2.6.

### 2.6 The Observatory (Track 2 Open Theme)

Eight capabilities are **absent from every finance-agent harness torn down** — Microsoft's
FinanceBenchmark, Trata's Hedge-Bench, SUFE's FinEval: probabilistic calibration (ECE / Brier /
reliability), decision consistency under replay, abstention quality, agent-contribution attribution,
point-in-time integrity testing, adversarial resistance, cost-aware evaluation, and live streaming
evaluation. Microsoft's additionally grades with **gpt-52 regardless of the model under test** — an
unguarded LLM judge; Trata at least uses a different family.

**Nine baselines**, D being decisive: buy-and-hold · risk parity · fixed rule · **deterministic
ARGUS** · single LLM · multi-agent without Constitution · full ARGUS · human + ARGUS.

**Model-versus-model, on identical state.** GPT, Claude, Qwen, Gemini and DeepSeek run the same data,
environment, costs, constraints and portfolio — compared on return, Sharpe, max drawdown,
calibration, consistency, abstention quality, risk violations, latency, cost, evidence grounding and
authority progression. That makes ARGUS a **neutral operating system for evaluating trading
intelligence**, which is a much larger claim than being one more agent.

**The Synthetic Market Challenge** removes the "the model already knew what happened" objection
entirely: generated markets no model has seen — bull, bear, sideways, high-volatility, jump risk,
liquidity collapse, correlation breakdown, fake news, delayed news, oracle failure. This is the
direction the literature is moving (KTD-Fin on separating memorised knowledge from skill; SynthFin
on contamination-free sequential evaluation).

**Abstention is scored economically.** A `NO_TRADE` is not automatically good — we compute what would
have happened and report **Abstention Value**: *avoided loss +$412*, or *missed opportunity −$183*.

### 2.7 Decide — the mechanics

**Hypotheses, not directions.** Each carries a thesis, evidence ids, a return *distribution*, and a
mandatory **falsifier**. Levkila's exit-plan schema is the working shape.

**Debate scored on quality.** TradingAgents' bull/bear structure with TraderBench's far better
instrumentation — hallucination, contradiction, concession and new-evidence detection
(`debate.py:150–423`).

**The Source Independence Graph is ours.** Five agents saying "bullish" after reading the same
Reuters article is five opinions and **one** independent source. The system displays and weights it
that way. Nothing in the corpus does this; it is the difference between a debate and an echo.

**Stress runs before the decision.** Hypothesis → stress → *revised* decision, so a CVaR blow-up
under a −10% BTC shock changes the position rather than appearing in a report beneath it.

**Cost discipline on the model itself.** atrx-demo's three-tier validation — cheap pre-filter, full
analysis, portfolio veto — adopted because our Qwen hackathon key has a limited balance and most
candidates never deserve a full analysis.

### 2.8 Challenge Mode

Thirteen attacks as controls a judge presses against the live system on the current decision — not a
CI suite nobody sees. TraderBench is the closest prior art and covers **one** outright (destroy
liquidity), three partially. **Nine are ours.**

ABIDES answers feasibility: **yes**, two `ExchangeAgent`s with independent `mkt_open`/`mkt_close` can
model a live token against a shut equity venue. Two gaps are our build work — it models **no fees at
all**, and shocks must be **pre-configured; there is no mid-run injection**, which is exactly what a
judge pressing a button requires. Midnight wrapping is untested and overnight/weekend is our case,
so it is the first thing to prove.

---

## 3. TRACK 1 — the Autonomous Alpha Factory

> ARGUS does not merely discover strategies. It discovers, falsifies, costs, stress-tests,
> capacity-tests and retires them.

**Track 1 is scored 100% quantitatively** — Sharpe, Sortino, max drawdown, turnover, OOS decay,
rolling 30-day stability. It does not care how elegant the system is. This section is therefore
about producing *numbers that survive costs*, not about architecture.

### 3.1 Six alpha engines

| Cell | Engine | The design decision that matters | Honest grade |
|---|---|---|---|
| **Arbitrage** | **Net Executable Arbitrage** — a signal exists only if `theoretical spread − taker fee − spread crossing − slippage − impact − latency − funding − failed-leg risk − hedge cost > required margin`, multiplied by an **Expected Arbitrage Capture Probability** | 300bps theoretical is not 300bps opportunity; it may be **38bps expected executable edge**. Our own research shows the economics are hostile and Weinberg shows the ordinary-holder redemption channel is structurally absent | **C today.** A only if genuine net executable edge appears. We publish the decomposition either way — that is the contribution |
| **After-hours pricing** | **Overnight Information Absorption Model** — predicts `P(next-open return │ information state)` from after-hours return, volume, spread, order imbalance, event type and surprise, sector and macro state, BTC state, volatility, liquidity, weekday-vs-weekend, NAV distance and time-to-open | Results **must** be split by Monday / Tue–Fri / earnings / macro / geopolitical / crypto-led / low-liquidity / high-volatility. This is what stops one giant Monday effect masquerading as a strategy | **A+** |
| **Cross-market correlation** | **Tradable Cross-Market Hedge Ratio** — not a correlation. *"0.71 ± 0.13 this regime; after session-overlap and stale-print correction, the economically usable ratio is 0.34"* | Clean ablation ladder: raw → session-corrected → DCC → DCC+regime → full ARGUS | **A** |
| **rToken factors** | **rToken State Factor Library** — NAV distance, time-to-open, time since last native price, crypto beta, weekend state, liquidity degradation, spread expansion, oracle state, underlying vol, close-to-open gap history, event intensity, mint/redeem constraints | **Not 1,000 generic factors.** Factors specifically about a *continuous token market with a discontinuous anchor* — a genuinely distinctive research problem, and the thinnest theme in our corpus | **A+ if the dataset is excellent** |
| **Cross-asset rotation** | **Execution-Aware Regime Allocation** — the optimiser sees expected return, risk, correlation, liquidity, cost, capacity, hedgeability and regime | A position with high expected return but an **empty hedge menu is penalised**. This is where Track 2's hedgeability work becomes Track 1 alpha | **A** |
| **Open Theme** | **Execution-aware alpha** — optimise `gross alpha − commissions − spread − impact − slippage − funding − turnover − capacity penalty − regime instability` | Optimise for **real executable P&L**, not predicted return. The handbook names execution-aware alpha explicitly | **S — our strongest Track 1 candidate** |

### 3.2 The Alpha Certification Protocol

Every strategy walks one path, and gets a **certificate**, not a Sharpe:

```text
IDEA → FORMALIZE → BASELINE → BACKTEST → COST → PURGED CV → OOS → DSR → PBO
  → CAPACITY → REGIMES → WALK-FORWARD → PAPER → CERTIFY → DEPLOY → MONITOR
  → DECAY → DEMOTE → RETIRE
```

The certificate records the trial count, the cost assumptions, the OOS window and when it was
unsealed, the DSR with its trial adjustment, PBO, capacity, regime coverage, and the decay curve.
A Sharpe without those is a number; a Sharpe with them is a result. This directly attacks the
field-wide search/evaluation contamination documented in §2.4.

---

## 4. TRACK 3 — the AI Trading Research OS

> A natural-language research question becomes a source-grounded, reproducible, decision-ready
> workspace.

**The warning that governs this track:** Track 3 is now the most competitive of the three. Financial
research agents with SEC and earnings RAG, portfolio dashboards, AI-native wealth workstations and
live agent tool execution already exist, and OpenAI launched a financial-services ChatGPT on
2026-09-10. **A "chat with your stock" product loses immediately.** Our differentiator cannot be data
breadth — it must be *decision quality and evidence lineage*.

| Cell | Product | The mechanism |
|---|---|---|
| **Information extraction** | **Claim → Evidence → Signal Graph** | A claim like *"margins are deteriorating"* is clickable down to: 10-Q page 84 → revenue → COGS → computed gross margin → prior quarter → consensus → expected margin → signal impact. Not citations — **lineage**. Built on docling's cell-level bboxes and Agent-Rita's bound-citation pattern |
| **Review & self-evolution** | **Research Autopsy** → **Personal Error Profile** | After every decision: what did I believe, what supported it, what did I ignore, what changed, which assumption failed, was it the analysis / execution / timing / information? Accumulates into *"you are consistently early on semiconductor reversals; you overweight management commentary; you are overconfident in high-volatility regimes."* This is **0/12 in our corpus** — open territory |
| **Stress testing** | **Decision Simulator** | Not "here are three similar historical examples." A live distribution — bull, base, bear, liquidity shock, BTC shock, correlation breakdown, earnings surprise, valuation compression — answering *"at what price and size does this thesis stop making sense?"* |
| **Personalised workbench** | **Portfolio-aware verdicts** | Same stock, same market, **different verdict** for two users, because the profile binds the Constitution: risk budget, horizon, holdings, sector concentration, preferred evidence, style, authority. The proof sentence a generic desk cannot produce: *"good trade in isolation, bad trade for your book — it lifts your semiconductor concentration from 22% to 31% against a 25% mandate cap"* |
| **Execution assistance** | **Execution Copilot** | Not "buy 100 shares?" — *urgency: low · recommended participation 8% · expected slippage X · expected cost Y · estimated completion Z · alternatives TWAP / liquidity-seeking · risk if delayed*. Track 2's execution engine surfaced as a desk feature |
| **Open Theme** | **Decision Cockpit** | One workspace: question → research → evidence → thesis → counter-thesis → historical analogues → stress → portfolio impact → execution → **human decision**. Every number clickable back to its source |

**Track 3 gets easier once Track 2 is mature**, because almost all of Track 2's hard infrastructure
becomes the desk's backend. That is the sequencing argument, not a reason to defer the design.

---

## 5. Build order

Truth before intelligence, intelligence before authority — each stage is meaningless without the one
beneath it.

1. **Foundation (all tracks)** — Temporal Truth Fabric with two clocks · Session State Machine ·
   evidence graph with bound citations · `DecisionContext` · the cost model that cannot be zero.
2. **Execution truth (all tracks)** — order state machine · fill engine · reconciliation.
   Deliberately early: an intelligence layer that cannot be executed honestly is decoration.
3. **Track 1 alpha engines** — the quantitative cells, because they need the longest data runway and
   are scored on numbers that take time to accumulate.
4. **Track 2 intelligence** — event · sentiment · earnings · cross-asset · factor lab.
5. **Track 2 decision** — hypotheses · source-independence graph · debate · stress · Constitution ·
   the Five Proof Systems.
6. **Track 3 surfaces** — cockpit · lineage · autopsy · simulator · copilot, on the Track 2 backend.
7. **Observatory, Challenge Mode, Synthetic Market Challenge** — the proof surface for all three.

---

## 6. What we refuse to build

Each is easy to build, impossible to defend, and present in the field:

- Bull agent + bear agent + trader, with no ablation showing each role adds value.
- Sentiment score → LLM decision. The edge is the same size as the fees.
- EPS beat → buy.
- "AI discovered a factor" after 20,000 trials with no trial count, DSR, PBO or capacity.
- Backtests without transaction costs — **the most widespread defect class in the entire corpus**.
- LLM arithmetic on financial values. FinRobot's shipped form of this rule is adopted: code computes
  every number, and prompts carry an explicit *"do not estimate"* instruction.
- Memory without `available_at`.
- Production self-editing.
- Fake or backfilled paper trading.
- A Track 3 chatbot that answers questions instead of grounding decisions.
- Competing on counts — most agents, most integrations, most charts, most papers read. All saturated.
  The only remaining differentiator is measured superiority.

---

## 7. The claim this architecture has to earn

Not *"ARGUS has twenty-one innovations."* That is a count, and the rule above bans it.

> We reproduced the strongest systems we could find. Here is where they fail. Here is ARGUS. Here is
> the exact metric where it improves, the ablation showing our component caused it, the attack it
> survived, the out-of-sample result, the replay, and the paper-trading evidence.

**Current status, measured 2026-09-12: built and running.** 53 modules, 317 tests, ruff and mypy
`--strict` clean, seven data artefacts from live Bitget series. Run `python -m argus.status` for a
runtime check rather than a claim.

What that has produced is mostly **negatives, and they are the point**: 0 of 8 factors certified,
0 of 12 symbols surviving the strict deflation gate, and a weekend effect that failed its own
train/test split. A system that reports those is worth more than one that reports a Sharpe.

One step cannot be taken from here: **no order has reached a venue**, because that needs a Bitget
API key. The signed path is built, tested, and probed live — the venue answered `40037 Apikey does
not exist`, confirming request, path, headers and demo routing are correct and only the key is
missing. `STATUS.md` carries the four steps.

The next work is not more design. It is converting §2.5's five proof systems and §3.2's certification
protocol into running acceptance tests.
