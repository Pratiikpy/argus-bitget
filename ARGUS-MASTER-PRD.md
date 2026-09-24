# ARGUS — Master Product Requirements & Build Specification
## Bitget AI Base Camp / Genesis Hackathon Season 2

**Status:** The original build specification, written 2026-09-12, with the measurements and
retractions recorded since. **Read the plan sections as history, not as the entry:** they name
Agentic Trading (Track 2) as the primary submission; the entry actually filed is **Track 3 (AI
Trading Desk) only**, and the README explains why Track 2 was withheld.
**Owner:** the owner (github.com/Pratiikpy)
**Written:** 2026-09-12

---

## 0. How to read this document

This is one document with six jobs:

1. State **what we are building** and why that specific thing.
2. Specify **every feature**, grouped by subsystem.
3. Name **every external source** each feature draws on, with its **licence** and our **legal disposition** (copy / rebuild / benchmark / study).
4. Map our system onto **all 18 hackathon sub-themes**.
5. Define **what we must prove** and how we will prove it.
6. Define **what we actually submit**, mapped field-by-field to the submission form.

It is grounded in three inputs that already exist on disk and should be read alongside it:

| Input | Location | What it gives this document |
|---|---|---|
| 18 sub-theme compendiums | `research/compendium/*.md` (243,297 words) | Every genuine mechanism found across 922 analysed sources, with `file:line` citations |
| Top-2 repo picks per sub-theme | `research/compendium/TOP2-PER-THEME.md` | The two sources that own each sub-theme, and why |
| Corpus calibration | `research/CORPUS_CALIBRATION.md` | Which of those findings are real, measured by hand-judging 216 samples |
| **25 architecture teardowns** | `research/architecture/*.md` (~149,000 words) | One code-level teardown per competing system: decision path, risk controls, PIT integrity, costs, defects — all with `file:line` |
| **Consolidated build ledger** | `research/architecture/_CONSOLIDATED-LEDGER.md` | 129 mechanisms with source, licence and disposition · 103 catalogued defects · 24 capabilities absent from every system read |
| **System architecture** | `ARGUS-ARCHITECTURE.md` | The system design built from all of the above. **That document, not this one, is the build specification.** |

Anything in this document that is **not** traceable to one of those, or to the official handbook, is labelled as original design.

### 0.1 Scope — all three tracks, all eighteen sub-themes

**Confirmed 2026-09-12: we build all three tracks.** An earlier note in this document froze Tracks 1
and 3; that freeze is lifted and this section supersedes it.

The three tracks are **three products on one engine**, and they are deliberately *not* the same
product pointed three ways, because each track is scored on a different thing:

| Track | Product | What it optimises for | How it is scored |
|---|---|---|---|
| **1 — Alpha Factory** | **Autonomous Alpha Factory** — discovers, falsifies, costs, stress-tests, capacity-tests and *retires* strategies | The strongest **statistical machine** | **100% quantitative.** Sharpe, Sortino, max drawdown, turnover, OOS decay, rolling 30-day stability |
| **2 — Agentic Trading** | **Autonomous Trading Governor** — the AI decides; ARGUS proves whether it should be trusted | The strongest **autonomous decision-maker** | 50% quantitative + 50% judge |
| **3 — AI Trading Desk** | **AI Trading Research OS** — a natural-language question becomes a source-grounded, reproducible, decision-ready workspace | The strongest **research experience** | Judge-weighted; the human remains the final decision-maker |

**The distinction that governs every design choice below:** Track 1 does not care how elegant ARGUS
is. A beautiful research system at Sharpe 0.7 loses to a boring strategy at Sharpe 2.0. Track 2 does
not care how high the Sharpe is if the LLM was not genuinely the decision-maker. Track 3 does not
care about either if a judge cannot get from a claim to its evidence in one click. **Optimising one
track's product for another track's scoreboard is the most likely way to lose all three.**

What they share is the engine: one Temporal Truth Fabric, one evidence graph, one risk and portfolio
core, one execution engine, one learning loop. That sharing is what makes eighteen sub-themes
coherent rather than eighteen demos; §7 maps every one of them.

The build specification is `ARGUS-ARCHITECTURE.md`.

---

## 1. The decision, up front

**ARGUS is a modular autonomous trading intelligence system. The closed-market tokenized-equity problem is its proving ground, not its boundary.**

That distinction is the single most important sentence in this document, and it was arrived at deliberately. An earlier draft scoped ARGUS to "a system that owns a set of gaps in the closed-market problem." That framing is too small. It produces a point solution that wins or loses on one thesis. The correct framing is a complete closed-loop trading intelligence system in which each of the 18 sub-themes is a **native capability of one engine**, and in which the 24/7-versus-shut-market asymmetry is the laboratory that forces every subsystem to interact honestly.

The core loop, which every sub-theme is a view into:

> **Observe → Verify → Understand → Hypothesise → Challenge → Stress → Decide → Risk → Execute → Reconcile → Grade → Learn → Re-certify**

Three things make this a product rather than a slogan:

**The problem it proves itself on.** An rToken trades 24/7. Its underlying equity trades 09:30–16:00 ET. Between Friday's close and Monday's open there are **65.5 hours** in which information arrives, the rToken prices it alone on thin liquidity, NAV goes stale behind oracle-freeze rules, **no hedge is placeable**, and the underlying will gap at the next open. Nothing in the 922 sources we analysed solves this. It is the event's own founding premise, unclaimed — and it happens to require evidence, market state, portfolio risk, execution and learning to all work at once, which is why it is the right laboratory for a general system.

**The authority thesis.** Bitget gives an autonomous agent an account. Nothing currently decides whether that agent has *earned* the right to use it. ARGUS makes capital authority a thing an agent earns through measured competence, holds only inside the regimes where it was proven, and loses automatically when the evidence decays.

**The evidence thesis.** A 2026 audit of 77 agentic-trading studies found that only 2 of 19 primary empirical studies had extractable time-consistent splits, only 1 of 19 explicitly modelled transaction costs, only 1 of 19 documented universe and survivorship handling, and **none reached the highest reproducibility category**. The dominant weakness in this field is not architecture — it is that almost nothing is comparably, reproducibly evaluated. ARGUS turns that weakness into a product surface.

The one-line pitch:

> **Bitget built an exchange where agents can trade. ARGUS decides which agents deserve to, how much, and under what conditions — and proves it.**

The longer statement of what the system is:

> **ARGUS converts heterogeneous evidence into point-in-time market state, generates and challenges hypotheses, reasons over cross-asset portfolio consequences, executes within deterministic constraints, and learns only from outcomes it can prove were available at decision time.**

### 1.0 The standard every subsystem must meet

We do not ask *"what is the minimum needed to enter this sub-theme?"* We ask *"what would the best system in the world contain?"* Operationally, every major capability must answer nine questions before it is considered done:

1. Who is the strongest existing implementation?
2. What exactly does it do?
3. What are its assumptions?
4. What does it get wrong?
5. Can we reproduce its result?
6. What does ARGUS add?
7. Does the addition improve the metric?
8. What happens when we remove it (ablation)?
9. Can a hostile judge break it, and can another researcher reproduce it?

**This is deliberately a defence against feature theatre.** Fifty agents, a hundred tools and twenty charts make a README look large and prove nothing. Breadth is only legitimate when each capability carries a baseline, an experiment, an ablation and a failure test.

### 1.1 What we submit

The handbook is explicit: *"Different themed entries from the same team are evaluated and awarded separately"*, and *"Submitting to 2 different themes requires two separate form submissions."* One entry selects one track and one sub-theme. So "own many sub-themes with one project" is not a submission strategy — it is an **engineering** strategy that makes two strong entries cheap to produce.

| | Submission A — primary | Submission B — secondary |
|---|---|---|
| **Track** | Agentic Trading | Alpha Factory |
| **Sub-theme** | Open Theme | Chosen by evidence, see §11.2 |
| **Scoring** | 50% quantitative + 50% judge | **100% quantitative** |
| **Why this one** | The judge half scores decision explainability, agent architecture and risk-control effectiveness — precisely what ARGUS is | Same engine, strategy layer only; a second independent shot at a prize |
| **Hard deliverable** | Runnable demo + event→decision→execution flow + **≥2-week paper-trading log run during the competition** | Strategy code + **≥60-day backtest with ≥30-day out-of-sample** |

**Primary is Agentic Trading, and this is a considered call, not a preference.** Alpha Factory is scored purely on numbers. Our own measured research says the numbers are hostile: 0.12% round-trip taker fees against roughly 0.00% measured intraday edge, 360+ intraday variants tested negative with profit factor never exceeding 0.89, and rToken arbitrage requiring 300–900bps of edge against a best-evidenced dislocation of 150–450bps. Entering a pure-quant scoreboard with an architectural thesis is how good engineering loses. Agentic Trading is where our strength converts into score.

### 1.2 What "owning a sub-theme" means here

Not "we have a button for it." A sub-theme is owned when it has:

a dedicated hypothesis · dedicated data · a named baseline it must beat · the strongest prior art identified and either absorbed or explicitly rejected · an original contribution · point-in-time-safe evaluation · frozen out-of-sample results · costs included · an ablation proving the component earns its place · adversarial testing · and reproducible evidence a stranger can re-run.

Eighteen such cells exist internally. Two get submitted.

---

## 2. Ground truth: what the handbook actually rewards

Read directly from `BITGET_AI_BASE_CAMP_S2_HANDBOOK_EN.md`. These are constraints, not suggestions.

### 2.1 Scoring by track

| Track | Mechanism | Judged on |
|---|---|---|
| Alpha Factory | **Pure quantitative** | Sharpe, Sortino, max drawdown, turnover, out-of-sample Sharpe decay (**alert fires at OS < 0.5×IS**), rolling 30-day Sharpe stability |
| Agentic Trading | **50% quantitative + 50% judge** | Paper-trading Sharpe, max drawdown, win rate; decision explainability; agent architecture quality; **risk-control layer effectiveness** |
| AI Trading Desk | Judge-weighted | Feature/data integration depth, research quality, natural-language interaction, personalisation |

### 2.2 The project description — where most entries lose

One long-form field, six parts. **Judges weigh the first three most.**

1. **Thesis (highest weight)** — why this exists, the core hypothesis, signal sources, decision logic, risk controls.
2. **Target user and product value** — a concrete segment. The handbook states outright that **"all traders" is not accepted**.
3. **Validation data and key metrics** — test period, returns, Sharpe/Sortino, max drawdown, win rate, turnover, **plus fee and slippage costs**, with every figure labelled **observed / estimated / targeted**.
4. Progress · 5. Deliverables · 6. Optional take on AI trading.

Invalid submission: missing a compliant X post, the project description, or accessible materials. Incomplete validation answers do not invalidate but *"noticeably lower your score."*

### 2.3 Dates and mechanics

| Item | Value |
|---|---|
| Submission deadline | **September 21** (UTC+8) |
| Judge review | September 22 – October 7 |
| Public voting | September 22 – 28 (parallel, does not replace judging) |
| Winners | October 8 |
| Open Theme slots | **Top 1 each of 2 open slots** per track — very narrow |
| Prize stacking | Judge side: only highest tier counts. Fan Favorite stacks with everything |
| Qwen credits | 30U-equivalent, separate form, first 300 KYC-passed teams |

### 2.4 The S1 reuse rule, handled honestly

Rule 3 forbids porting an S1 entry with only renaming or minor edits, and requires describing **substantive new additions**, with judging evaluating the new content.

NightDesk (the author's S1 entry, featured on the official S2 landing page) graded *decisions* against the NYSE open with Ed25519 signing and 15 hard gates. ARGUS is not that. ARGUS decomposes *edge* across a closed window into information / timing / execution / luck, and feeds that decomposition back into how much authority an agent holds. Studying our own prior architecture, reusing our own code, and carrying forward what we learned is entirely legitimate; the submission will state plainly what is new. That statement is a scoring asset, not a liability.

---

## 3. The scientific problem

### 3.1 The Sleeping-Anchor Problem

> **How should an intelligent system estimate, trade, hedge and risk-manage an asset that continues trading while its highest-quality native reference market is intermittently unavailable?**

This is the spine of the entire product. Everything else is in service of it.

### 3.2 Why it is genuinely unsolved — measured, not asserted

From `research/CORPUS_CALIBRATION.md` and the 18 compendiums, across 922 sources and 236 S/A-tier + GPT-suggested sources:

| Finding | Evidence |
|---|---|
| Nothing hedges a live rToken against a shut equity market | Searched every multi-leg execution source; Ondo sidesteps with buffer inventory; IBKR's algos *explicitly exclude* extended hours |
| Nothing models gap-against-position risk across a closed session | t3-stress compendium, search recorded per checklist item |
| Nothing computes marginal risk of a **proposed** trade | t3-tdopen: 0 clear passes in 12 sampled, ~0–8% genuine |
| Nothing does post-trade edge attribution or behavioural-bias detection on an agent's own trade log | t3-review: 0 clear passes in 12 sampled |
| Nothing does real execution scheduling for a 24/7 book | t3-execassist: 0 clear passes, 10 of 12 outright false positives |

Corpus-wide, only **~21%** of sources labelled as contributing to a theme genuinely did so; **~61%** were false positives, usually generic utilities claiming relevance through "enables/powers X" language.

### 3.3 The economics that constrain every design choice

These numbers are the reason most obvious ideas fail, and they come from our own prior testing plus the corpus's empirical sources.

| Quantity | Value | Consequence |
|---|---|---|
| Round-trip taker fee, RWA perps | **0.12%** | Any strategy round-tripping daily loses |
| Measured intraday edge | **~0.00%** | Intraday is a fee-donation machine |
| Measured overnight drift | **~0.10%** | The only effect of the right order of magnitude |
| Intraday variants tested | **360+, all negative**, profit factor never > 0.89 | Not a tuning problem |
| Tokenized-equity spreads | 85–150bps normal, **250–400bps off-hours** | Off-hours is 50–100× native equity spreads |
| $25k AMM order impact | **1.85%** | Capacity is tiny |
| Weekend NAV discount | 1.5–4.5% | Real, but see next row |
| Arbitrage breakeven requirement | **300–900bps** | Against 150–450bps available — naive arbitrage is a trap |

**Design consequences, binding on everything below:**

1. **Low turnover or no edge.** Overnight drift is the only measured effect that clears fees.
2. **Cost model is not optional.** It is constructed-mandatory (see §6.4).
3. **Naive arbitrage is out** as a primary thesis. It survives only as a *detector* and a *risk input*.
4. **Capacity must be stated.** A strategy that works at $5k and dies at $50k must say so.

---

### 3.4 Measured ground truth — first real data, 2026-09-12

Everything below replaces a secondary-source estimate with a measurement taken from Bitget's own
published series. Three of our own claims were falsified. Code: `argus/src/argus/research/`,
data: `argus/data/`.

#### The index does not freeze — it attenuates

The specification said price discovery *stops* while the anchor is shut. **False.** Bitget publishes
a continuously updating index for every rToken; on NVDAUSDT it changed in **100% of hours** in every
session phase, weekends included. What actually happens is attenuation — median absolute hourly
index move:

| Phase | Median hourly index move | Relative to RTH |
|---|---|---|
| RTH | **32.17 bps** | 1.00× |
| Extended | 15.59 bps | 0.48× |
| Overnight | 11.97 bps | 0.37× |
| **Weekend** | **4.53 bps** | **0.14×** |

Price discovery during a weekend runs at roughly **one seventh** of RTH intensity. The token tracks
that damped reference closely (market/index move ratio 0.92–1.16 across all phases), which is why
the basis stays narrow.

#### Arbitrage: 28,067 observations, and it does not pay

13 rTokens × 90 days of hourly market/index/premium series:

| Phase | Observations | Median basis | Monetizable after full cost stack |
|---|---|---|---|
| RTH | 5,070 | 6.51 bps | 8.68% |
| Extended | 8,450 | 6.64 bps | 10.0% |
| Overnight | 6,760 | 3.07 bps | 7.44% |
| Weekend | 7,787 | 2.13 bps | **4.87%** |
| **All** | **28,067** | **4.27 bps** | **7.72%** |

**The PRD previously claimed "150–450bps available" from secondary sources. The real median is
4.27bps — two orders of magnitude smaller.** The conclusion that naive rToken arbitrage does not pay
survives and is strengthened; the numbers supporting it were wrong.

Note the inversion: the basis is **widest when the anchor is open** and narrowest at weekends, the
opposite of the "gaps widen outside core hours" claim in the tokenized-equity literature. The
explanation is the attenuation above — with the anchor shut, token and index both track the same
damped reference and agree with each other.

#### The reopening gap: real size, unpredictable direction

832 closed sessions across 13 instruments:

| Phase | Sessions | Median closure | Median reopen move | Net of 12bps fee | Continuation rate |
|---|---|---|---|---|---|
| Weekend | 156 | 66h | 55.40 bps | +43.40 bps | 57.7% |
| Overnight | 676 | 18h | 58.44 bps | +46.44 bps | 50.6% |

The moves are large — **median 58bps, and 89.3% exceed the round-trip fee**. The problem is
entirely directional: overnight continuation is 50.6%, a coin flip.

#### The weekend effect FAILS its own validation — and this is the most important result here

Weekend continuation of 57.7% looked like the strongest signal this project had produced. Under our
own standing rules it does not survive:

| Test | Result | Verdict |
|---|---|---|
| Exact binomial, full sample | p = 0.0326, but 95% CI **[49.8%, 65.2%]** | **FAIL** — the interval includes a coin flip |
| Chronological train/test | train 48.7% · out-of-sample 66.7% (p=0.0022) | **FAIL** — the entire effect is in one half |
| Per-symbol | 8/13 above chance; leave-one-out stable 55.6–60.4% | Pass — not one instrument |

**The full-sample 57.7% is the average of "nothing" (48.7%) and "strong" (66.7%), which makes it
meaningless.** Split the other way and we would have trained on 66.7% and tested into 48.7% — a
catastrophic failure that a full-sample number would have concealed. This is precisely the trap our
own backtest rules record, and the honest report is: **no validated weekend edge.**

**What this buys us.** A rigorously obtained negative, with the machinery that produced it. The
prize is real — 58bps of median reopening movement, 89% of it clearing the fee — and the binding
constraint is now precisely identified as *directional prediction*, not spread, not cost, not
execution. That is exactly what the Price-Discovery Twin exists to attack, and it now has a
measured target instead of an assumption.

#### Measured latency to the venue — and the delay that actually matters

30/30 unauthenticated probes of `api.bitget.com/api/v2/public/time`, spaced 200ms, from this
machine (`argus/data/latency_probe.json`, reproducible with `python -m argus.execution.latency_probe`):

| Leg | Measured |
|---|---|
| Round trip | min **267ms** · median **308ms** · p90 **338ms** · max **353ms** |
| Entry (us → venue) | median **153ms** |
| Response (venue → us) | median **156ms** |
| Rejections | **0 of 30** |
| Clock offset, venue → local | **−459ms** (NTP-style median midpoint) |

Two things follow, and the second is the one that matters.

**First, the clock offset is larger than the entire round trip.** Naively differencing the venue's
`requestTime` against our own clock would have produced negative entry latencies — which are
hftbacktest's *rejection marker* (`latency.rs:199-214`) — and we would have recorded 30 rejections
where there were none. The offset is therefore estimated before any row is built, and rows whose
legs come out impossible are dropped and counted rather than written.

**Second, network latency is not our binding delay — deliberation is, and it is not cheap.** At 45%
annualised vol the worst measured round trip (353ms) prices at **0.48bps**, 4% of the round-trip
fee. But a reasoning model takes **8–40 seconds**, four orders of magnitude more, and
`thinking_budget_cost_bps` prices that against the same walk:

| Thinking budget | RTH (full depth) | Off-hours (≈⅓ depth) |
|---|---|---|
| 8s | 2.27bps | 6.80bps |
| 20s | 3.58bps | 10.75bps |
| 40s | **5.07bps** | **15.20bps** |

**We expected this to come out negligible and it does not.** A 40-second deliberation off-hours
costs **15.2bps of expected adverse move against a 12bps round-trip fee** — the thinking costs more
than the trading. Even at RTH, 40 seconds is 42% of the fee. The result stands the intuition on its
head: on this venue the price of thinking is a first-order term in the cost stack, not a rounding
error, and the two-clock model makes it worst precisely when the desk is most likely to be running
unattended.

The operational consequence is concrete. **Off-hours, a full-reasoning budget needs an edge above
roughly 27bps** (12 fee + 15 latency) before it is worth spending, where a low-thinking budget needs
about 19bps. That is a real trade — accuracy against staleness — and it is now a number the Meta-PM
can be held to rather than a preference. It is also the strongest argument yet for the abstention
machinery: on a thin book, thinking longer about a marginal trade can cost more than the trade was
ever worth.

---

## 4. What the research proved about the tools we planned to use

This section exists because it changed the build plan. Every item was found by reading the actual code.

| Finding | Impact on us |
|---|---|
| **mlfinlab-official is ~90% `pass` stubs** — 420 of 466 functions — behind accurate López de Prado citations | We cannot depend on it for purged CV. Use `purged-cross-validation` and ml4t-jansen instead |
| **Riskfolio-Lib reports swapped EVaR/TG values** in its report module | Any risk number taken from it must be independently recomputed |
| **vectorbt's Deflated Sharpe silently returns NaN** | Cannot be trusted as a validation gate |
| **MIRAI is the only source with a correctly enforced as-of-date firewall** (constructor-raise + independent data-layer filter, reset per query) | Copy this pattern exactly |
| **FinMem, FinAgent, TradeMaster, letta-memgpt, MemGPT all leak** future knowledge via the same omitted-`as_of` retrieval pattern | Do not inherit any of these memory architectures as-is |
| **GDELT's `seendate` is crawl time, not publish time** | Silent look-ahead trap; must be corrected at ingestion |
| **AutoHedge never hedges** — its execution tools are dead code with zero callers | Name recognition is not evidence |
| **live-trade-bench declares fee/funding fields it never increments**; fills at exact prompt price, zero latency | A widely-cited harness with no cost model |
| **FinCon's repo contains no code at all** (README only), yet its paper claims 62–113% returns with **no transaction costs** | Paper-only; claims unverifiable |
| **nofx-real** has 9 live venue adapters and trades tokenized equities as perps on Hyperliquid, but **SL/TP failures are logged and ignored**, leaving positions naked | Steal the multi-venue shape, not the failure handling |
| **Only 1 source of 236 discloses costs** at a level clearing the 0.12% threshold (Alpha-Jungle-MCTS, 0.15%) | Cost honesty is itself a differentiator |

---

## 5. Product definition

### 5.1 System shape

```text
                    EVIDENCE LAYER
        filings · news · macro · social · market · on-chain
                          │
                 TEMPORAL TRUTH FABRIC
        every fact carries: event time · publish time
        · ingest time · available-at time · revision chain
                          │
                 SESSION STATE MACHINE
        per instrument: underlying open/extended/shut ·
        NAV fresh/frozen · hours to next price discovery ·
        HEDGEABILITY SURFACE (ranked menu + priced residual)
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   SLEEPING-ANCHOR    GAP-RISK          UNHEDGEABLE
      TWIN            ENGINE            EXPOSURE LEDGER
   latent native    distribution of    what I can hedge now
   value + its      the open gap       vs. what I must carry
   uncertainty                         to Monday
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                    META-PM (LLM)
        the primary decision-maker, per Bitget's
        Agentic Trading definition
                          │
                CONSTITUTION KERNEL
        typed intent → policy (OPA) → feasibility (Z3/CVXPY)
        → signed authorisation
        verdicts: ALLOW · RESIZE · REQUIRE_HEDGE
                  · DELAY · REJECT · FLATTEN
                          │
              TIDE-AWARE EXECUTION
        participation rate conditioned on session state
                          │
                  BITGET AGENT HUB
                          │
                 EXECUTION TRUTH ENGINE
        formal order state machine + reconciliation
                          │
                  DECISION LEDGER
        graded at next open · edge decomposed into
        information / timing / execution / luck
                          │
                 CAPITAL AUTHORITY
        promote · demote · expire · revoke
```

### 5.1a The six intelligence planes

The system is organised into six planes. Every one of the 18 sub-themes lands in exactly one plane, which is what makes breadth coherent rather than scattered.

| Plane | Owns which sub-themes | Core question |
|---|---|---|
| **1. Information Intelligence** | Event-driven, Sentiment, Earnings, Information Extraction | What happened, when was it knowable, and how much was already priced? |
| **2. Market Intelligence** | Cross-market correlation, After-hours pricing, rToken factors, Cross-asset rotation, Price discovery | What state is the market in, and what is the latent value of an asset whose anchor is asleep? |
| **3. Research Intelligence** | Factor discovery, autonomous hypothesis generation, backtesting, alpha search, strategy retirement | What is a real effect, and what is a survivor of multiple testing? |
| **4. Portfolio & Risk Intelligence** | Marginal risk, CVaR, stress testing, hedgeability, concentration, correlation, collateral, regime exposure | What does *this proposed trade* do to the book? |
| **5. Execution Intelligence** | Cross-asset execution, Execution assistance, scheduling, impact, slippage, queue, partial fills, reconciliation | What order should actually be placed, and what really happened? |
| **6. Meta-Agent Intelligence** | Review & self-evolution, Personalised workbench, Agent evaluation, consistency, calibration, authority, provenance, contamination | Which intelligence deserves influence, and how do we know? |

### 5.1b Three irreversible layers — the ordering constraint above the planes

The six planes say *what* the system knows. They do not say what may not be skipped. That is a separate, stricter structure, and it is the one that makes the architecture defensible rather than merely large:

```text
        LAYER 1 — TRUTH          nothing may trade unless time is correct,
        (gate, not a feature)    evidence is valid, market state is known
                                 and data lineage is resolvable
                    ↓
        LAYER 2 — INTELLIGENCE   event · sentiment · earnings · factor
        (may be wrong)           discovery · cross-asset · portfolio ·
                                 stress · execution · agent debate
                    ↓
        LAYER 3 — AUTHORITY      decision → risk → execution →
        (earned, never assumed)  reconciliation → outcome → attribution →
                                 calibration → authority → next allocation
```

**Truth → Intelligence → Authority.** The direction is one-way and enforced in code, not convention:

- **Layer 2 may never write into Layer 1.** An intelligence module cannot revise a timestamp, promote a source's credibility, or restate what was knowable. If it could, every leakage defence in the system would be one prompt away from being disabled.
- **Layer 3 authority is derived from Layer 1 and 2 outputs and can never be self-asserted.** An agent cannot argue its way into a larger position; it can only be *measured* into one.
- **A Layer 1 failure halts the pipeline.** Unknown session state, unresolved lineage or a disputed `available_at` produces `DATA_INSUFFICIENT`, not a best-effort trade. This is the difference between an abstention and a guess, and §5.2b makes it scoreable.

This is also the honest reply to "you claim eighteen sub-themes." Eighteen capabilities sitting side by side is a feature list. Eighteen capabilities that all sit in Layer 2, all consume one Layer 1, and all answer to one Layer 3, is a system.

### 5.2 The twenty-one original contributions

Open source supplies foundations. These twenty-one are ours. The first five were established in the original specification and each fills a gap we *measured* as empty; the remaining sixteen extend the system from a point solution into a complete closed loop.

**Foundation contributions (1–5)**

| # | Contribution | Fills which proven gap |
|---|---|---|
| 1 | **Hedgeability surface** — see §5.2a. Supersedes the earlier binary "hedgeable fraction" | t2-crossexec: nothing handles trading-hours asymmetry in multi-leg execution |
| 2 | **Gap-risk engine** — marginal VaR/CVaR of a *proposed* trade across a closed session, under 100ms | t3-tdopen: 0/12 sources compute marginal risk of a proposed trade |
| 3 | **Edge decomposition ledger** — every decision graded at next open, attributed to information / timing / execution / luck, feeding back into sizing | t3-review: 0/12 sources do post-trade edge attribution |
| 4 | **Capital authority envelope** — competence certified per regime; authority expires; demotion is automatic | No source ties measured competence to bounded execution permission |
| 5 | **Constructed-mandatory cost model** — a zero-fee backtest is structurally impossible to instantiate | Only 1 of 236 sources is cost-honest at the 0.12% threshold; the 77-study audit found 1/19 modelling costs at all; and across 103 defects catalogued from 61 code-level teardowns, **cost-blindness is the single most widespread defect class** — ahead of leakage and self-scoring. Two production engines fail it by default: Nautilus backtests at **zero fees** unless configured, and Qlib measures turnover as **gross notional rather than net delta**, overstating cost 2–3× on mean-reversion |

**Extended contributions (6–18)**

| # | Contribution | What it does | Why it is needed |
|---|---|---|---|
| 6 | **Event causality graph** | Stores the transmission chain explicitly (Fed speech → rate expectations → real yields → sector → rToken → crypto beta), then **grades each link afterwards** — link-level hit rate, not just terminal direction | Event agents assert transmission; none record and score it. Grading per link makes event reasoning an auditable object: a correct call reached through three wrong links is a *failure* that a P&L-only scorer records as a success |
| 7 | **Sentiment integrity graph** | Sentiment is scored for credibility, bot likelihood, coordination, novelty, cross-platform agreement, price confirmation and persistence before it can influence a decision | The manipulation attack is real and quantified (~50% profit uplift against a naive consumer); no defence exists in the corpus |
| 8 | **Earnings expectation-gap engine** | Decomposes a print into **seven** independent surprises — reported, consensus, guidance, narrative, valuation, management-credibility shift, and Q&A — each carrying its own sign, magnitude and confidence, then maps the vector to an expected-return distribution rather than a direction | "EPS beat → buy" is the field default; earnings land exactly in our closed window. The seven are separated because they routinely disagree — a beat with cut guidance and evasive Q&A is a different asset than a beat with raised guidance, and a single "surprise" scalar destroys that |
| 9 | **Price-discovery twin** | Predicts the **distribution of the next genuine price-discovery event**, not a point price | Nothing models latent value while the anchor is unavailable |
| 10 | **Autonomous factor laboratory** | A factor holds an explicit lifecycle *state*, and every transition is a gate: `DISCOVER → FORMALIZE → BACKTEST → ADVERSARIAL → OOS → DSR → PBO → CAPACITY → REGIME → PAPER → CERTIFY → DEPLOY → MONITOR → DECAY → DEMOTE → RETIRE`. A factor cannot skip a state, and `DEMOTE`/`RETIRE` fire automatically on measured decay rather than on a human noticing | Capacity, decay and retirement are unsolved across the entire pool. **Retirement is the state that matters**: a library that only ever grows is a library that is lying about decay, and nothing in the corpus retires anything |
| 11 | **Historical analogue engine** | "When has the market looked like this before?" via matrix profile over a market-state vector | t3-stress: analogue retrieval confirmed absent |
| 12 | **Counterfactual scenario engine** | A judge alters BTC, liquidity, correlation, news truth, oracle state or latency and the **entire decision recomputes live** | Static stress tables are the norm; interactive counterfactuals are not |
| 13 | **Agent contribution attribution** | Measures which agent actually changed the decision, and its marginal effect on Sharpe and drawdown over many decisions | Most systems stop at "we have eight agents"; none measure which earns its place |
| 14 | **Agent consistency benchmark** | Same state replayed k times; measures decision, reasoning, evidence, risk and execution-plan consistency | A model flipping position on harmless wording variation is a measurable defect nobody measures |
| 15 | **Research survival / retirement engine** | Tracks what fraction of discovered signals survive every gate; the cemetery retains all failures | Publication bias inside our own system is the failure mode |
| 16 | **Evidence contamination score** | Every result carries a contamination confidence, via date masking, ticker masking and synthetic identifiers | LLMs remember historical outcomes; raw returns can be beta in disguise |
| 17 | **Execution-quality intelligence** | Implementation shortfall, fill probability, queue position, latency and failed-leg recovery time as first-class measured outputs | Execution is the weakest-modelled layer in the field |
| 18 | **Regime-specific authority** | Authority is granted per regime, decays with time, freezes on regime change and downgrades instantly on violation | "AI + hard risk gate" is now table stakes; certified, expiring, regime-scoped authority is not |
| 19 | **Two-clock point-in-time model** | Availability is resolved against **two clocks simultaneously** — the rToken's continuous clock and the underlying's session clock — because a fact can be tradeable on one and not yet priced on the other | Verified absent everywhere: every PIT implementation found assumes a single market clock |
| 20 | **Fused per-decision trust score** | One number per decision combining contamination confidence, evidence quality, PIT integrity, calibration and consistency — carried on the DecisionContext | No system fuses these; they exist separately or not at all |
| 21 | **True probabilistic calibration** | ECE, Brier score and reliability diagrams on the agent's own stated confidence, fed back into sizing | **Exhaustively grepped: ECE and Brier appear nowhere in any of the four evaluation repos.** Abstention calibration exists; probability calibration does not |

### 5.2a The hedgeability surface — upgraded from binary to continuous

The original specification treated hedgeability as a single fraction: what share of risk is hedgeable right now. That is directionally right but scientifically weak. It is replaced by a **continuous surface** computed per candidate hedge instrument:

```
                 risk_reduction
               × correlation_confidence
               × liquidity_availability
               × execution_probability
               × basis_stability
               ─────────────────────────
               − execution_cost
               − collateral_cost
               − model_uncertainty
                        ↓
        economically hedgeable risk (per instrument)
                        ↓
        ranked hedge menu + residual unhedgeable risk
```

This produces three things the binary version could not: a **ranked hedge menu** rather than a yes/no, a **residual** that is explicitly priced and carried, and a measurable quantity we call **Risk Neutralisation Efficiency** — marginal risk removed divided by all-in cost of the hedge. The agent's chosen hedge can then be scored against the best feasible hedge, which is an ablation nobody currently runs.

### 5.2b Decision vocabulary — abstention is first-class

The agent does not emit a direction. It emits one of seven verdicts, each scored:

`TRADE` · `REDUCE` · `HEDGE` · `DELAY` · `NO_TRADE` · `HUMAN_REVIEW` · `DATA_INSUFFICIENT`

A system that knows when **not** to trade, and is measured on the quality of its abstentions, is more sophisticated than one that always produces a direction. `DATA_INSUFFICIENT` is distinct from `NO_TRADE`: the first says the evidence was inadequate, the second says the evidence was adequate and the answer was no.

### 5.2c The universal DecisionContext

Every agent in every plane emits the same typed object. This is what allows one engine to express eighteen sub-themes without eighteen bespoke pipelines, and what makes replay and grading uniform.

```text
DecisionContext
  decision_id · event_id(s) · instrument(s) · decision_time
  evidence_manifest · evidence_cutoff · contamination_score
  market_session_state · portfolio_state_hash
  hypotheses[] · opposing_hypotheses[] · historical_analogues[]
  fair_value_distribution · expected_return_distribution
  risk_distribution · hedgeability_surface
  execution_cost_distribution
  proposed_action · proposed_size · holding_horizon
  invalidation_conditions · confidence · uncertainty
  agent_contributions[]            # who moved the decision
  constitution_verdict · execution_plan
  model_hash · prompt_hash · policy_hash · code_hash · signature
```

`invalidation_conditions` deserves emphasis: every position carries the explicit conditions under which its thesis is dead. That converts "we were wrong" from a narrative into a pre-registered, checkable statement.

### 5.2d Agent portfolio — selection as a learned policy

Rather than invoking a fixed roster on every decision, ARGUS selects **which intelligence to consult** based on context. A simple technical move needs no earnings analyst. An earnings event activates earnings, fundamental, sentiment, event and cross-asset. A weekend macro shock activates macro, event, cross-asset, price discovery and gap risk.

Selection is a contextual-bandit problem over `(event class, asset, session state, volatility, liquidity, portfolio state, agent competence, agent uncertainty, correlation to other agents)`. Thompson sampling over measured marginal contribution is the baseline approach. The output is initially an **evidence influence weight**, not capital — it converts to bounded capital authority only after independent validation.

### 5.2e Agent debate quality

Multi-agent disagreement is only useful if it is real. We measure five things most systems never look at:

- **Agreement** — did agents actually agree, or converge trivially?
- **Independence** — were their evidence sources genuinely independent?
- **Information overlap** — did five agents simply re-read the same article?
- **Contradiction quality** — did disagreement expose something useful?
- **Resolution quality** — did the Meta-PM resolve the disagreement correctly?

This connects directly to anchor independence: multiple venues agreeing is not independent information agreeing, and five agents agreeing after reading one source is not consensus.

### 5.2f Three environments, and the separation of search from evaluation

Self-evolution without containment is how a system games its own scoring function. Three hard-separated environments:

| Environment | What is permitted |
|---|---|
| **Research sandbox** | Unlimited experimentation, self-modification of hypotheses and research artefacts |
| **Certification** | Frozen data, frozen evaluator, adversarial checks. The evaluator is **not** modifiable by the searcher |
| **Production** | Only certified artefacts execute |

The governing rule: **an agent may modify hypotheses and research artefacts; it may never modify the authority policy or the safety layer.** And structurally, the **search system and the evaluation system are separated** so the searcher cannot collapse into self-confirmation — the failure mode that sealed-joint-search research exists to prevent.

### 5.3 What we can and cannot guarantee

Stated up front, in the submission, because it is more credible than a performance claim and costs us nothing.

**Guaranteed by construction:**
- No order reaches execution without a valid signed certificate.
- No record unavailable at decision time is exposed through the approved data interface.
- No order violates an encoded deterministic constraint.
- The executed order equals the approved order, or reconciliation flags it.
- Every decision is reproducible from its artifact bundle.

**Not guaranteed:**
- Positive return.
- Correct interpretation of world events.
- Perfect data.
- Absence of software defects.
- Winning a judged competition.

---

## 6. Feature specification

Every feature below carries: **what it does**, **where it comes from**, and its **disposition** — `COPY` (permissive licence, take the code), `REBUILD` (restrictive or absent licence, reimplement from the published method), `BENCHMARK` (keep external, compete against it), `STUDY` (learn, do not import), `ORIGINAL` (ours).

### 6.1 Temporal Truth Fabric

The foundation. If this is wrong, every number above it is a lie.

| ID | Feature | Behaviour | Source | Disposition |
|---|---|---|---|---|
| TT-1 | Five-timestamp fact model | Every fact stores event time, publish time, ingest time, **available-at** time, and revision chain | ORIGINAL, informed by MIRAI | ORIGINAL |
| TT-2 | As-of firewall | Retrieval refuses any record whose available-at > decision time. Enforced twice: constructor-raise **and** an independent data-layer filter, reset per query | MIRAI (`repo-mirai`) — the only correct implementation found in 236 sources | COPY (MIT) |
| TT-3 | Crawl-vs-publish correction | GDELT `seendate` is crawl time; corrected at ingestion or discarded | Our finding, t3-infoextract compendium | ORIGINAL |
| TT-4 | Point-in-time feature joins | Features resolved as of decision time, never latest | Feast | STUDY → REBUILD (thin need; full Feast is heavy) |
| TT-5 | Immutable dataset snapshots | Every evaluation pins an exact dataset version | Apache Iceberg / lakeFS | STUDY (adopt only if scale demands) |
| TT-6 | Revision-aware fundamentals | As-first-reported vs restated kept distinct | `christianpichichero-max~pit-fundamentals`, ml4t-jansen | REBUILD |
| TT-7 | Leakage attack suite | Deliberately attempts to leak future data; must fail | "What Survives Honest Evaluation?" | ORIGINAL |

**Why TT matters more than it looks:** a deliberately future-leaking strategy can post a Sharpe near 35 and survive standard statistical corrections. Leakage must be prevented *structurally*, not detected statistically.

### 6.2 Session State Machine

| ID | Feature | Behaviour | Source | Disposition |
|---|---|---|---|---|
| SS-1 | Per-instrument session state | RTH / extended / overnight / weekend / holiday, per venue | `exchange_calendars`, `pandas_market_calendars`, `maread99~market_prices` | COPY (permissive) |
| SS-2 | Five-state session lifecycle | Including QUIET_PAUSE and price-collar transitions | IBKR overnight-trading spec (`gpt-023`) | REBUILD (spec, not code) |
| SS-3 | Oracle freshness/freeze | NAV staleness thresholds (3600s RTH, 86400s weekend); frozen-oracle flag | Ondo market-hours operational guide (`gpt-029`) | REBUILD (vendor doc) |
| SS-4 | Time-to-next-discovery | Hours until the next real price-discovery event (open/auction) | ORIGINAL | ORIGINAL |
| SS-5 | **Hedgeability surface** | Per candidate instrument: risk reduction × correlation confidence × liquidity × execution probability × basis stability, less execution cost, collateral cost and model uncertainty. Emits a **ranked hedge menu** and an explicitly **priced residual** carried to Monday — never a single fraction. See §5.2a | ORIGINAL — contribution #1 | ORIGINAL |
| SS-6 | Session-conditioned cost curve | Spread/impact expectations switch by session. **The 3–5× off-hours widening this row previously asserted is FALSIFIED for Bitget rToken perps** — measured live on a Saturday with the anchor shut and 53.6h to discovery: NVDAUSDT **0.46bps**, TSLAUSDT **0.27**, AAPLUSDT **0.30**, median across 13 instruments **0.57bps**. Thin products do widen (SPXUSDT 8.09, SQQQUSDT 2.58), so the curve is a *liquidity-tier* effect, not a *session* effect. See `argus/data/rtoken_panel.csv` | Measured by `argus.market.collector`; Barclay & Hendershott retained for the price-discovery method only | ORIGINAL measurement |

### 6.3 Sleeping-Anchor Twin

Estimates the latent native price **distribution** during closure — never a point estimate.

| ID | Feature | Behaviour | Source | Disposition |
|---|---|---|---|---|
| ST-1 | Reopen distribution | P(native open price \| information ≤ t), with 50/80/95% intervals | Darts, NeuralForecast, StatsForecast as baselines | COPY (permissive) + ORIGINAL assembly |
| ST-2 | Calibrated intervals | Conformal prediction; coverage reported honestly, assumptions stated | MAPIE | COPY (permissive) |
| ST-3 | Move decomposition | Splits an rToken move into information / cross-asset transmission / liquidity premium / oracle effect / unresolved | ORIGINAL; must be calibrated, not assumed identifiable | ORIGINAL |
| ST-4 | Weekend crypto → Monday equity signal | The one published formula linking weekend crypto moves to the Monday equity open | IMF "Cryptic Connections" (`gpt-036`) | REBUILD from paper |
| ST-5 | Price-discovery share | Weighted Price Contribution: what fraction of discovery happens in each session | Barclay & Hendershott 2003 (`gpt-021`) | REBUILD from paper |
| ST-6 | Anchor independence check | Two venues agreeing ≠ independent information, if they share a reference | "Cross-Venue Agreement Is Not Price Discovery" | REBUILD |
| ST-7 | Historical analogue retrieval | "When has the market looked like this before?" via matrix profile | STUMPY | COPY (BSD) |
| ST-8 | Structural-break detection | Regime change invalidates fitted relationships | `ruptures` | COPY (BSD) — note its own caveat: usable only if causal |
| ST-9 | Abstention threshold | Below a confidence floor the system declines to trade | ORIGINAL | ORIGINAL |

### 6.4 Cost & Execution Realism

**The constructed-mandatory principle:** the backtest engine cannot be instantiated without a fee model. This is copied directly from hftbacktest, which raises `BuildError::BuilderIncomplete("fee_model")`. A silent zero-fee backtest becomes structurally impossible.

| ID | Feature | Behaviour | Source | Disposition |
|---|---|---|---|---|
| CE-1 | Mandatory fee model | Construction fails without it | hftbacktest | COPY (**MIT**) |
| CE-2 | Queue-position fill model | Tiered: risk-adverse / probabilistic / L3 FIFO | hftbacktest | COPY (**MIT**) — **BUILT**, `argus.execution.queue`; all five probability functions, 55 tests |
| CE-3 | Recorded-latency interpolation | Latency drawn from a *recorded* real round-trip series, not a guessed constant | hftbacktest | COPY (**MIT**) — **BUILT**, `argus.execution.latency`; series recorded live, 30/30 rows (§3.4) |
| CE-4 | Depth-weighted fills with slippage cap | Orderbook-depth-weighted dry-run fill, 5% cap; fee defaults to `max(taker, maker)` | freqtrade (`exchange.py:1251-1296`) | **REBUILD** — GPL, viral |
| CE-5 | Square-root impact law | Power-law decay kernel, τ₀ calibrated (~30s crypto vs 15min+ illiquid equity) | Gatheral (`gpt-066`) | REBUILD from paper |
| CE-6 | Implementation-shortfall decomposition | Delay / temporary / permanent / opportunity cost — matches the brief's own language | Kissell & Glantz TCA handbook (`gpt-067`) | REBUILD from book |
| CE-7 | Almgren-Chriss scheduling | Optimal execution trading impact against volatility risk | Almgren & Chriss 2000; extended with OU order-flow term by Almgren & Li 2016 (`gpt-194`) | REBUILD from papers — **BUILT**, `argus.execution.schedule`; closed-form sinh trajectory, verified at both limits (λ=0 gives exact TWAP, λ→∞ gives immediate) with a monotone mean/variance frontier between. Almgren & Li's OU term is **not** built |
| CE-8 | **Tide-aware participation** | Participation rate conditioned on session state — the gap IBKR's algos explicitly refuse to cover. **Verified original:** AlphaTrade hardcodes continuous NASDAQ hours (`base_env.py:53-54`) and RL-LOB's three market regimes are fixed per-episode, not time-of-day driven. Nothing models a thin-overnight/deep-at-open liquidity tide | ORIGINAL | ORIGINAL |
| CE-14 | **Execution action space** | The schema behind "execute 40% now, 35% passively, 25% opportunistically": a 4-tier price-level space (far-touch / mid / near-touch / deep-passive) combined with a simplex over {market, limit-per-level, hold} | AlphaTrade `exec_env.py:104-341`; RL-LOB `utils.py:36-55` | **REBUILD** — both are unlicensed. **Neither models fees at all** (exhaustive grep), so the cost layer is entirely ours |
| CE-9 | Multi-venue adapter shape | Per-venue order-sync rebuilding local state from exchange truth | nofx-real | STUDY (take the shape; its SL/TP handling is broken) |
| CE-10 | Net-profitability gate | Both legs fire only when spread − (fees + gas + slippage buffer) ≥ minimum | Hummingbot `ArbitrageExecutor` | COPY (**Apache 2.0**) |
| CE-11 | Cross-leg collateral lock | Locks collateral across legs to prevent double-spend | Hummingbot `BudgetChecker` | COPY (**Apache 2.0**) |
| CE-12 | **Single-leg failure handler** | Cancel / Reverse / Proceed when one leg fills and the other doesn't | `bitrinjani~r2` `SingleLegHandler` — the only real answer in the corpus | COPY (**MIT**) |
| CE-13 | Continuous margin chokepoint | Every state change passes one margin check | lfest-rs | **REBUILD** — AGPL, would force our whole stack open |

### 6.5 Gap-Risk Engine — original contribution #2

| ID | Feature | Behaviour | Source | Disposition |
|---|---|---|---|---|
| GR-1 | Marginal VaR/CVaR of a proposed trade | Δrisk from a candidate order, across the closed window, **<100ms** | ORIGINAL — 0/12 sources do this | ORIGINAL |
| GR-2 | Cached covariance | Updated 1–4×/day; marginal computation on demand | cvxpy DPP parameter caching (`problems/problem.py:825-864`) | COPY (Apache 2.0) |
| GR-3 | Covariance denoising | Marchenko-Pastur eigenvalue filtering; Nested Clustered Optimization | López de Prado | REBUILD from published method |
| GR-4 | Shrinkage estimator | Ledoit-Wolf | scikit-learn / PyPortfolioOpt | COPY (**MIT** / BSD) |
| GR-5 | Dynamic correlation | DCC-GARCH: time-varying hedge ratio β_t = ρ_t·σ₁/σ₂ | Engle & Sheppard (`gpt-031`/`gpt-032`) | REBUILD from paper |
| GR-6 | CVaR optimisation | Rockafellar-Uryasev linear program | Riskfolio-Lib as reference | **REBUILD** — licence says "All rights reserved", **and its EVaR/TG report values are swapped** |
| GR-7 | Turbulence circuit-breaker | Mahalanobis-distance covariance anomaly → defensive action on percentile breach | FinRL-Trading | REBUILD (reactive only; not a substitute for GR-1) |
| GR-8 | Session-mismatch correlation correction | Correlation between a 24/7 asset and a session-bound one is biased; 3-day overlapping returns technique | `gpt-046` | REBUILD |
| GR-9 | Ranked hedge suggestions | Candidate hedges ordered by cost-adjusted risk reduction | ORIGINAL | ORIGINAL |
| GR-10 | Collateral haircut model | Haircut-weighted equity across crypto + rToken; staged MMR gates (halt 70% / deleverage 85% / liquidate 100%) | Bitget UTA v3 docs | REBUILD from vendor docs |

### 6.6 Constitution Kernel

The LLM proposes. The Constitution disposes. It can only **ALLOW · RESIZE · REQUIRE_HEDGE · DELAY · REJECT · FLATTEN**.

| ID | Feature | Behaviour | Source | Disposition |
|---|---|---|---|---|
| CK-1 | Typed intent contract | LLM emits a typed object, never free text, never a raw API call | ORIGINAL | ORIGINAL |
| CK-2 | Declarative policy | Machine-readable rules evaluated deterministically | Open Policy Agent | COPY (Apache 2.0) |
| CK-3 | Mathematical feasibility | Constraint satisfiability; detects conflicting policies | Z3 / CVXPY | COPY (MIT / Apache 2.0) |
| CK-4 | Risk schema | `max_position_weight`, `max_sector_weight`, `max_daily_drawdown`, `max_trade_loss`, `max_slippage_bps`, `minimum_liquidity`, `minimum_source_reliability`, `max_data_age_ms` | ARGUS PRIME concept doc | ADOPT |
| CK-5 | Default-deny authorisation | Per-agent opt-in with unguessable tokens | FinceptTerminal's gate — strongest access-control pattern in the corpus | REBUILD |
| CK-6 | Two-phase confirm | Propose → confirm, fails closed if the client lacks the capability | MCP elicitation pattern | REBUILD (MCP itself is a wire protocol, not execution logic) |
| CK-7 | Kill switch | One deterministic path to flatten everything | ORIGINAL | ORIGINAL |

### 6.7 Execution Truth Engine

| ID | Feature | Behaviour | Source | Disposition |
|---|---|---|---|---|
| ET-1 | Formal order state machine | PROPOSED → POLICY_CHECKED → CAPITAL_RESERVED → SUBMITTED → ACKNOWLEDGED → PARTIALLY_FILLED → FILLED; branches REJECTED / CANCEL_PENDING / CANCELLED / **UNKNOWN** / RECONCILING / RECONCILED | ARGUS PRIME concept doc; nautilus_trader's architecture | ADOPT + STUDY (nautilus is **LGPL** — use as a library, do not vendor modified) |
| ET-2 | Timeout ≠ rejection | A network timeout must never be treated as a rejection; the exchange may have accepted | nautilus_trader documentation | ADOPT principle |
| ET-3 | Reconciliation loop | Local state rebuilt from exchange truth, continuously | nofx-real per-venue sync goroutines | STUDY → ORIGINAL |
| ET-4 | Idempotency | `clientOid` tagging with agent-prefixed IDs; survives retries | Bitget UTA v3 | ADOPT |
| ET-5 | Rate-limit awareness | 20 req/s trading, 10 req/s account — a documented cause of leg desync | Bitget UTA v3 | ADOPT |
| ET-6 | Approved-equals-executed check | Fills matched against the signed intent; divergence raises | ORIGINAL | ORIGINAL |

### 6.8 Decision Ledger & Edge Attribution — original contribution #3

| ID | Feature | Behaviour | Source | Disposition |
|---|---|---|---|---|
| DL-1 | Trade certificate | Every order carries evidence manifest, data cutoff, market regime, fair-value distribution, candidate actions, chosen action, contradicting evidence, expected net edge, uncertainty interval, portfolio before/after, stress results, risk verdict, execution plan, model/prompt/policy/code hashes, signature | ARGUS PRIME concept doc | ADOPT |
| DL-2 | Grade at next open | Every closed-window decision scored when the underlying reopens | ORIGINAL; NightDesk did this for decisions, we extend to edge | ORIGINAL |
| DL-3 | **Edge decomposition** | Realised P&L attributed to information / timing / execution / luck | ORIGINAL — 0/12 sources do this | ORIGINAL |
| DL-4 | Behavioural-bias detection | Disposition effect, revenge sizing, overconfidence, anchoring — measured on the agent's own trade log | ORIGINAL — nothing in the corpus detects these | ORIGINAL |
| DL-5 | Confidence calibration | Stated confidence vs realised hit rate; ECE and reliability diagram | `repo-agent-backtest-lab` — the only calibration code found | COPY (permissive) + ORIGINAL feedback into sizing |
| DL-6 | Strategy cemetery | Every failed hypothesis retained permanently | "What Survives Honest Evaluation?" | ADOPT |
| DL-7 | One-click replay | Any decision reproducible from its artifact bundle | ORIGINAL | ORIGINAL |

### 6.9 Research & Validation Harness

| ID | Feature | Behaviour | Source | Disposition |
|---|---|---|---|---|
| RV-1 | Purged cross-validation + embargo | Embargo **must be explicitly set**; the default-zero footgun is removed | `repo-purged-cross-validation` | COPY — **not mlfinlab (90% stubs)** |
| RV-2 | Deflated Sharpe Ratio | With **logged trial counts**, making it externally verifiable — which no corpus source does | Bailey & López de Prado; ml4t-jansen's wired chain | REBUILD from paper |
| RV-3 | Probability of Backtest Overfitting | CSCV, recommended gate PBO < 0.25 | Bailey/Borwein/López de Prado/Zhu (`gpt-062`) | REBUILD from paper |
| RV-4 | Full multiple-testing chain | BH-FDR → DSR → PBO → MinTRL, with synthetic ground-truth calibration | ml4t-jansen — the only actually-wired chain in the pool | COPY (permissive) |
| RV-5 | Vectorised factor engine with lookback guard | Code-enforced, so a factor cannot silently see the future | Qlib; zipline-reloaded `SimplePipelineEngine` | COPY (**MIT** / Apache 2.0) |
| RV-6 | Independent second engine | A result must reproduce on an engine that did not generate it | LEAN or zipline-reloaded | BENCHMARK |
| RV-7 | Trial ledger | Every attempted configuration recorded; search budget accounted | "What Survives Honest Evaluation?" | ADOPT |
| RV-8 | Factor evaluation | IC, decay, turnover, quantile spreads | alphalens-reloaded | COPY (Apache 2.0) |
| RV-9 | Contamination-resistant evaluation | BRIGHT / ticker-blind / date-blind / fully-blind scoring | KTD-Fin | ADOPT |
| RV-10 | pass^k consistency | Repeated-run decision consistency, deterministic state-diff reward, policy ablations | tau-bench / tau2-bench | COPY (**MIT**) |
| RV-11 | Capacity & decay monitoring | **Confirmed unsolved across the entire pool** | ORIGINAL | ORIGINAL |
| RV-12 | **Search/evaluation separation, enforced in code** | The searcher cannot write to, influence, or overwrite the evaluator's verdict. See the cautionary evidence below | ORIGINAL — and now demonstrably necessary | ORIGINAL |

#### Why RV-12 exists: a self-scoring loop found in the wild

`dtbtc/mcts-llm-alpha` — a reproduction of the Alpha Jungle LLM+MCTS factor-mining paper — **computes a genuine IS/OOS statistical overfitting score at `qlib_evaluator.py:143`, then unconditionally overwrites it with the same generating LLM's own self-judgment at `comprehensive.py:90-98`.** The searcher grades its own work and the statistical test is discarded. This is a confirmed, code-traced instance of exactly the failure mode §5.2f exists to prevent, in one of the strongest factor-mining implementations available.

Two further cost-blindness findings from the same teardown:

- **`toininoi/alphaagent`** has a genuinely wired AST-duplication regulator (`factor_regulator.py`), but its autonomous keep/reject decision reads only `without_cost` metrics at `feedback.py:40-45`, while the Qlib backtest computes `with_cost` metrics that path never reads. Its shipped US template sets `open_cost: 0.0`.
- **`Nunchi-trade/auto-researchtrading`** has the most honest cost model of the six execution/alpha repos examined — taker fee plus slippage, never assuming maker fills — but its test-set isolation is a *prompt convention*, not a code control: `load_data("test")` is freely callable mid-loop.
- **`cheer932041235/FinAgent-RAG`** is non-functional: 26 of its 27 methods raise `NotImplementedError`.

**And the headline negative, verified by direct grep rather than assumed: no repository in the alpha-discovery group implements purged or embargoed cross-validation, deflated Sharpe with trial counts, PBO, capacity estimation, or factor retirement.** The entire validation chain in §9 is ours to build.

### 6.10 Evidence & Extraction

| ID | Feature | Behaviour | Source | Disposition |
|---|---|---|---|---|
| EV-1 | SEC filing access | Typed access to 10-K/10-Q/8-K, XBRL facts, Forms 3/4/5, 13F | EdgarTools | COPY (permissive) |
| EV-2 | Independent XBRL validation | Structured facts parsed deterministically, never by LLM | Arelle (`gpt-152`) | COPY (Apache 2.0) |
| EV-3 | Document structure extraction | PDF/DOCX/PPTX/tables → structured representation | Docling | COPY (MIT) |
| EV-4 | Numeric reasoning | Retrieve → program → execute, so arithmetic is deterministic | FinQA / ConvFinQA | COPY (**MIT**) |
| EV-5 | Filing-timestamp leakage check | Signal timestamp vs SEC filing timestamp | `finance-broski~backtest-bias` | COPY |
| EV-6 | Two-stage news extraction | Headline discovery, then per-article content + byline | FNSPID | STUDY (**has a sign-inverted timezone bug — 10-hour error**) |
| EV-7 | Financial sentiment baseline | 3-class classifier with documented scoring formula | FinBERT | COPY (Apache 2.0) — as a **baseline**, not the thesis |
| EV-8 | Event-study machinery | CAR, abnormal returns, SUE, event windows, t-tests | MacKinlay 1997; `LemaireJean-Baptiste/EventStudy`; `jsmidt~quantpy` `event_profiler.py` | COPY / REBUILD |
| EV-9 | Deterministic-vs-interpretive split | **LLM interprets; code calculates.** No accounting number is ever produced by a language model | ORIGINAL principle | ORIGINAL |

### 6.11 Agent Layer

Per Bitget's definition, the LLM must be the **primary decision-maker** — not a narrator for a deterministic strategy.

| ID | Feature | Behaviour | Source | Disposition |
|---|---|---|---|---|
| AG-1 | Stateful reasoning graph | Durable agent state, checkpointing, human intervention points | LangGraph (Pregel checkpointing) | COPY (**MIT**) |
| AG-2 | Durable workflow | Crash-resistant long-running processes. **LangGraph owns reasoning state; Temporal owns durable operational state** | Temporal | STUDY → adopt if scale demands |
| AG-3 | Role specialisation | Real division of responsibility, not agent theatre | TradingAgents | BENCHMARK (permanent challenger) |
| AG-4 | Memory with as-of filtering | Retrieval carries a knowledge timestamp enforced at query time | MIRAI pattern; **not** FinMem/FinAgent/MemGPT, which all leak | REBUILD |
| AG-5 | Abstention | "No trade" is a first-class output, scored as a decision | ORIGINAL | ORIGINAL |
| AG-6 | Contradiction surfacing | The agent must present evidence against its own thesis | ORIGINAL | ORIGINAL |

### 6.12 Adversarial & Security

Non-optional: this system touches capital.

| ID | Feature | Behaviour | Source | Disposition |
|---|---|---|---|---|
| SEC-1 | Indirect prompt-injection tests | Hostile tool environments with measurable attack-success rates | AgentDojo | BENCHMARK |
| SEC-2 | Red-team pipeline | Automated adversarial regression tests in CI | Promptfoo, PyRIT | COPY (permissive) |
| SEC-3 | Financial attack suite | Fake filing, injected exhibit, malicious MCP tool, stale quote, ticker collision, duplicate order, unknown order state, memory poisoning, oracle failure, stablecoin depeg | ORIGINAL (ARGUS Dojo concept) | ORIGINAL |
| SEC-4 | Sentiment-manipulation defence | The attack is real and quantified (~50% profit uplift against a naive consumer) and **no defence exists anywhere in the corpus** | ORIGINAL | ORIGINAL |
| SEC-5 | Governance framing | Deterministic policy enforcement around model I/O rather than trusting agent judgment | NIST AI RMF, OWASP AI Agent Security | ADOPT as framing |
| SEC-6 | **Challenge Mode** — the attack suite as a judge-operable surface | SEC-3 exists as a CI regression suite, which no judge will ever see. SEC-6 exposes the same thirteen attacks as live controls a judge presses during the demo, against the running system, on the real current decision. See §6.12a | ORIGINAL | ORIGINAL |

#### 6.12a Challenge Mode — turning the security layer into the product

A security section nobody can operate is a claim. The thirteen attacks below are buttons, and each one mutates live state and forces the full pipeline to recompute in front of the judge:

| Control | What it injects | What ARGUS must demonstrate |
|---|---|---|
| **Leak future information** | A fact stamped `available_at` after the decision time | Layer 1 refuses it; the decision is unchanged; the refusal is logged with the offending timestamp |
| **Poison memory** | A fabricated "lesson" in the reflection store | Provenance check rejects it, or it is quarantined and its influence on the decision is shown as zero |
| **Inject fake news** | A plausible unsourced claim from a low-credibility origin | Sentiment integrity graph scores it down; the decision shift is bounded and displayed |
| **Change the ticker** | Same narrative, different symbol | Ticker-collision defence fires; nothing routes to the wrong instrument |
| **Duplicate the order** | Replay of a signed authorisation | Idempotency key rejects the second; execution truth engine shows one fill, not two |
| **Force a partial fill** | Execution returns 40% of requested size | Residual is re-planned, not abandoned; the hedge leg is re-solved against the real filled quantity |
| **Break the hedge leg** | The hedge instrument rejects | Unhedgeable exposure ledger takes the residual, position is resized down, authority is not spent on an unhedged book |
| **Destroy liquidity** | Depth collapses 90% | Tide-aware participation drops; expected impact re-prices; the trade may become `DELAY` |
| **Break the correlation** | DCC estimate inverts | Hedgeability surface re-ranks; Risk Neutralisation Efficiency collapses and the hedge is withdrawn rather than held on faith |
| **Change the oracle** | NAV goes stale, then wrong | Freshness gate freezes; `DATA_INSUFFICIENT` rather than a trade against a bad print |
| **Delay the network** | 3s added latency | Order state enters UNKNOWN; timeout is not treated as rejection; reconciliation resolves it |
| **Move the market open** | Session calendar mutated | Time-to-next-discovery, gap risk and hedgeability all recompute; nothing is hardcoded to 09:30 ET |
| **Alter one source** | A single upstream fact is silently edited | Lineage diff detects the change and names which conclusions depended on it |

**Why this is a product feature and not a demo trick.** Every competitor can narrate that their system is safe. None can hand a judge the controls. The asymmetry is the point: a judge who breaks a system learns nothing about it, and a judge who *fails* to break one learns everything. Challenge Mode is also self-scoring — each control records whether the expected defence actually fired, so the suite doubles as the G7 evidence for every sub-theme in §7.2a.

---

## 7. Sub-theme ownership map

All 18. For each: our component, the two sources that own it, and whether the theme is **contested** (real prior art exists) or **open** (proven near-empty, so originality is cheap and defensible).

### Track 1 — Alpha Factory

| Sub-theme | ARGUS component | Owning sources | State |
|---|---|---|---|
| **Arbitrage** | **Monetizable-spread decomposition** — a pipeline, not a threshold: `opportunity → tradability → executable spread → capacity → fill probability → all-in cost → residual risk → verdict`. Its headline output is never "spread > threshold" but *"212bp apparent, 61bp economically monetizable after fee, slippage, capacity and failed-leg probability"* | ArbitrageLab (`gpt-013`), Mafrur RWA illiquidity study (`gpt-048`), Weinberg redemption-gap (§8.2c) | **Contested + economically hostile.** Needs 300–900bps vs 150–450bps available, and Weinberg shows the ordinary-holder redemption channel is structurally absent. We ship it as a *detector and risk input*, and the decomposition itself is the contribution — it is what lets us say *why* the naive version fails with a number instead of a shrug |
| **After-hours pricing** | Sleeping-Anchor Twin (§6.3) | Barclay & Hendershott 2003 (WPC) + 2004 (spread decomposition) | **Open in code.** Both owners are *papers* — no repo owns this. Methodology exists, rToken application does not |
| **Cross-market correlation** | Session-mismatch-corrected correlation + DCC hedge ratios. The output is a **usable hedge ratio, not a correlation** — *"estimate 0.71 ± 0.13 in the current regime; after correcting for session overlap and stale equity observations, the economically usable ratio is 0.34"* — because a correlation computed across a shut session is measuring a stale print, not a relationship | ArbitrageLab (VECM), Diebold-Yilmaz spillover (`gpt-035`), Engle-Sheppard DCC (`gpt-032`) | **Contested.** Real machinery exists; the correction for mismatched sessions does not. Tested across six regimes: normal, high-vol, weekend, post-event, liquidity shock, correlation breakdown |
| **rToken factors** | Session-boundary factor family | Mafrur (`gpt-048`), Ondo 24/7 product (`gpt-030`) | **Thinnest theme in the corpus** (6/290 at strict baseline). Mostly original |
| **Cross-asset rotation** | Regime-conditioned allocation constrained by the hedgeability surface — an asset whose hedge menu is empty carries a higher effective risk charge than its volatility implies | Engle-Sheppard DCC-GARCH (`gpt-032`), TVP-VAR connectedness (`gpt-039`) | **Contested.** Note: crypto-only rotation was repeatedly mislabelled as cross-asset |
| **Open Theme** | Execution-aware alpha — the whole cost stack of §6.4 | hftbacktest, Gatheral (`gpt-066`) | **Our strongest Alpha candidate.** The handbook names "execution-aware alpha" explicitly as an example direction |

### Track 2 — Agentic Trading

| Sub-theme | ARGUS component | Owning sources | State |
|---|---|---|---|
| **Event-driven** | Closed-window event agent — a 3am event hits a 24/7 token with the hedge shut | MacKinlay 1997, EventStudy (`gpt-085`) | **Contested.** Event-study machinery is solid; the closed-window angle is not |
| **Sentiment** | Manipulation-resistant sentiment with credibility weighting | FinBERT, RavenPack signal combination (`gpt-089`) | **Trap theme.** Edges are the same size as the fees that eat them; the *defence* is the open niche |
| **Earnings** | Expectation-gap engine — earnings land exactly in the closed window | ConvFinQA, EventStudy | **Near-empty** (1/12 genuine) yet the most rToken-native event type. High value |
| **Cross-asset execution** | Hedgeability surface + Risk Neutralisation Efficiency (§5.2a, §6.5) | Hummingbot (BudgetChecker), Barclay & Hendershott 2004 | **The sharpest gap.** Nothing hedges a live rToken against a shut market |
| **Factor discovery** | Trial-logged discovery loop with a verifiable DSR gate | Alpha-Jungle-MCTS, ml4t-jansen | **Strongest machinery in the corpus** — but capacity/decay/retirement is unsolved, and no agent wires trial counts into its own gate |
| **Open Theme** | **← SUBMISSION A.** The full authority system | tau-bench (pass^k), agent-backtest-lab (PIT firewall, DSR/PSR) | **Deepest evaluation machinery available.** Only 2 open slots per track |

### Track 3 — AI Trading Desk

| Sub-theme | ARGUS component | Owning sources | State |
|---|---|---|---|
| **Information extraction** | Evidence graph with as-of enforcement | ConvFinQA, Arelle (`gpt-152`) | **Contested.** But MIRAI is the *only* source with a correct as-of firewall |
| **Review & self-evolution** | Edge decomposition ledger (§6.8) | Generative Agents, Voyager — *scaffolding only* | **Open. 0/12 sources.** Neither pick actually owns it |
| **Stress testing** | Gap-against-position engine | STUMPY (matrix profiles), ABIDES | **Open.** 1/12 genuine, and that one was a connectivity-failure source |
| **Workbench** | Session-aware desk | LangGraph (Pregel checkpointing), Haystack | **Contested-ish.** Saved re-runnable workflows exist but were never applied to finance |
| **Execution assistance** | Tide-aware execution (§6.4) | hftbacktest, Almgren & Chriss | **Worst-covered theme in 922 sources.** 0/12, 10 outright false positives |
| **Open Theme** | Pre-trade marginal-risk copilot (§6.5) | Riskfolio-Lib, cvxportfolio (`gpt-201`) | **Second-worst.** 0/12; 11 of 12 were data connectors with no risk mechanism |

**Reading the map:** eight of eighteen are effectively open territory. Every one of those eight sits in the closed-market problem. That is not a coincidence — it is the same unsolved phenomenon showing up under eight different names, which is exactly why one engine can legitimately claim them all.

### 7.1 The cross-theme research graph

The sub-themes are not parallel features. They are a chain, and each link feeds the next. This is the structural argument for why one product can own many themes without being eighteen demos in a trench coat:

```text
Event detection → Sentiment change → Earnings/fundamental interpretation
   → Cross-asset transmission → Factor hypothesis → Historical analogue
   → Stress test → Portfolio marginal risk → Execution optimisation
   → Trade → Post-trade attribution → Factor & agent update
                          ↑______________________|
```

### 7.2 The per-sub-theme proof standard

Every sub-theme gets an identical directory, and a capability is not "done" until all of it is populated:

```text
/subtheme
    /prior-art      — who is strongest today, and what exactly they do
    /papers         — the literature that defines the problem
    /repos          — the implementations, with licence dispositions
    /datasets       — what we evaluate on
    /baselines      — the strongest realistic existing approach
    /replications   — their result, reproduced by us
    /argus          — our mechanism
    /ablations      — the same system with our mechanism removed
    /adversarial    — deliberate attempts to break it
    /out-of-sample  — frozen, untouched until the end
    /failures       — everything that did not work, retained
    /results        — the comparison, reproducible by a stranger
```

**The minimum proof package, per subsystem, is four artefacts:**

| Artefact | Question it answers |
|---|---|
| **Baseline** | What does the strongest realistic existing approach achieve? |
| **ARGUS** | What does our mechanism achieve? |
| **Ablation** | What happens when our mechanism is removed? |
| **Failure test** | What happens when someone deliberately attacks it? |

Worked example for sentiment: baseline is FinBERT; ARGUS is integrity-weighted sentiment; ablation removes the integrity layer; the attack is a coordinated bot narrative. That pattern repeats for every module without exception.

### 7.2a The ownership acceptance matrix — 18 × 10, and nothing is graded on opinion

§7.2 says what a finished sub-theme *contains*. This says when we are allowed to use the word **OWNED**, and it is a gate, not a narrative. A sub-theme is owned when all ten cells are green. Nine green is not owned; it is nine green.

**The ten gates:**

| # | Gate | Passing evidence |
|---|---|---|
| G1 | Best existing implementation identified | Named repo, commit SHA, licence, and a code-level teardown — not a README summary |
| G2 | Best paper identified | The paper that *defines* the problem, read in full, method extracted |
| G3 | Best dataset identified | Named, obtainable, with its known defects written down |
| G4 | **Their result reproduced by us** | We ran their code or method and got their number, or we recorded precisely where it failed to reproduce |
| G5 | ARGUS beats the baseline | On the metric that theme is actually judged on, net of the 0.12% cost model |
| G6 | Ablation proves our component is what did it | Same system, our mechanism removed, measurable degradation |
| G7 | Adversarial test survived | The theme-specific attack from §6.12 run against it |
| G8 | Out-of-sample result survived | Frozen slice, touched once, at the end |
| G9 | Cost included | Not a footnote — instantiated through the constructed-mandatory cost model (CE-5) |
| G10 | A stranger can reproduce it | One command, pinned seeds, pinned data, from a clean checkout |

**G4 is the gate that will hurt, and it is deliberately placed before G5.** Claiming to beat a baseline you never actually ran is the single most common failure in this field — the 77-study audit found **2 of 19** primary studies with extractable time-consistent splits and **none** at the highest reproducibility level. We do not get to be the twentieth. Where a baseline genuinely cannot be reproduced (dead dependency, absent data, unlicensed code), that is recorded as a **documented non-reproduction with the reason**, which passes G4; silently skipping it does not.

**Current state — measured, 2026-09-12.** The system is built and running: **53 modules, 317
tests, ruff and mypy `--strict` clean**, with seven data artefacts produced from live Bitget series.
G1–G3 were discharged by the research corpus; **G4–G6 are now discharged in several cells by code
and measurement**, and what remains open is stated per cell below rather than in aggregate.

What exists and runs today:

| Layer | Module | Evidence on disk |
|---|---|---|
| Truth — two-clock PIT, session state, as-of store | `argus.truth` | leakage suite, 100% of probes refused |
| Cost — zero-fee unconstructible | `argus.cost` | every study is net of 12bps |
| Market — public API, basis series, universe validation | `argus.market` | `rtoken_panel.csv`, 12 verified rTokens |
| Research — arbitrage, gap, significance, Track-1 sweep, factor lab | `argus.research` | 5 JSON reports, 28,067 + 832 + 2,159 observations |
| Backtest — lagged engine, correct metrics | `argus.backtest` | `track1_study.json`, 16 strategies × 12 symbols |
| Agents — 5 analysts, Meta-PM, causality, earnings | `argus.agents` | live-verified against Qwen |
| Decision — Constitution that may only reduce | `argus.decision` | invariant enforced and tested |
| Execution — 14-state order machine, signed Bitget client | `argus.execution` | signing verified 3 ways from source |
| Proof — Autonomy Proof, Execution Proof | `argus.proof` | `track2_desk_run.json` |
| Paper — hash-chained ledger, scheduled runner | `argus.paper` | `paper_ledger.jsonl`, accumulating |
| Eval — Observatory, Challenge Mode, model bake-off | `argus.eval` | 13/13 controls, 2 models at consistency 1.0 |
| Desk — all six Track-3 sub-themes | `argus.desk` | cockpit published |

**What is still genuinely open**, stated plainly rather than folded into a status colour:

* **No certified alpha.** The factor lab proposed 8 and certified **0** — 7 died on the cost gate,
  1 on DSR. The Track-1 sweep found 11 of 12 symbols beating buy-and-hold on Sharpe and **0 of 12**
  surviving the strict deflation gate. That is the honest result, not a pending one.
* **The weekend effect failed validation** (train 48.7%, out-of-sample 66.7%) and is reported as
  failed.
* **No order has reached a venue.** The signed path is built, tested, and probed against the live
  API — Bitget answered `40037 Apikey does not exist`, which confirms the request, path, headers and
  demo routing are all correct and the key itself is absent. Creating that key is the one step that
  cannot be done from here.
* **The paper log is young.** It accumulates on a schedule; the ledger refuses to draw a pattern
  below five graded outcomes.

| # | Sub-theme | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 | G9 | G10 | Position today |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| 1 | t1-arbitrage | ✅ | ✅ | ◐ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | Weakest Track-1 cell. The honest target is a *decomposition nobody else publishes*, not a profitable arb |
| 2 | t1-afterhours | ✅ | ✅ | ◐ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | Strong. Both G1 owners are papers — no repo holds this, so G4 means reproducing Barclay-Hendershott WPC |
| 3 | t1-crossmarket | ✅ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | Machinery exists and is reproducible; the session correction is the whole delta |
| 4 | t1-rtokenfactor | ◐ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | Thinnest theme in the corpus. **G3 is the blocker** — the dataset does not exist and we must build it |
| 5 | t1-crossasset | ✅ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | G4 is cheap here: PyPortfolioOpt and skfolio both run. No excuse for skipping it |
| 6 | t1-afopen | ✅ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | Strongest Alpha candidate. G9 is the entire thesis, so it is the first gate attempted, not the last |
| 7 | t2-event | ✅ | ✅ | ◐ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | G5 needs a link-level metric that does not yet exist in the literature — we define and publish it |
| 8 | t2-sentiment | ✅ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | FinBERT reproduces easily. **G7 is the real gate** — the manipulation attack is the product |
| 9 | t2-earnings | ✅ | ✅ | ◐ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | CARAG is now the baseline to beat, not a citation. G4 means running it |
| 10 | t2-crossexec | ✅ | ✅ | ◐ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | Sharpest gap. G5's metric is Risk Neutralisation Efficiency, which no baseline reports — so we must compute it *for* the baseline too |
| 11 | t2-factordiscovery | ✅ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | Best machinery available. G6 must isolate the *evaluator separation*, since that is our claim, not MCTS |
| 12 | t2-atopen | ✅ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | **← Submission A.** G10 carries disproportionate weight: the authority system is worthless if unauditable |
| 13 | t3-infoextract | ✅ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | FinAgentBench grades document *and* passage selection, so G5 is measurable rather than vibes |
| 14 | t3-review | ✅ | ◐ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | 0/12 sources. G5 = measured behaviour change after N lessons, not the existence of a memory store |
| 15 | t3-stress | ✅ | ✅ | ◐ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | Open. G7 and G5 partly coincide — the attack surface *is* the feature |
| 16 | t3-workbench | ✅ | ◐ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | **Weakest cell in the whole matrix.** G5 requires a UX baseline, and we have not named one. See §7.4a |
| 17 | t3-execassist | ✅ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | Worst-covered theme in 922 sources, best-defined gates. G4 against TWAP/VWAP/POV/Almgren-Chriss is mechanical |
| 18 | t3-tdopen | ✅ | ✅ | ◐ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | 0/12. G5 needs a portfolio-aware baseline; cvxportfolio is the only honest candidate |

✅ discharged · ◐ partial · ⬜ open

**Two rules that stop this becoming decoration.** First, the matrix lives in the repo as `OWNERSHIP.md`, generated from the per-sub-theme directories in §7.2 — a cell turns green because an artefact exists on disk, never because someone typed a tick. Second, **a red cell is reported, not hidden**: the submission states which sub-themes are owned and which are merely implemented. Eighteen implemented and six owned, stated plainly, is a stronger claim than eighteen asserted, because the first is checkable and a judge knows it.

### 7.3 The baseline matrix

A number without a comparison is not evidence. Every headline claim runs against eight baselines:

| | Baseline | Purpose |
|---|---|---|
| A | Buy-and-hold | Did we beat doing nothing? |
| B | Risk parity / portfolio baseline | Did we beat naive diversification? |
| C | Fixed-rule strategy | Did we beat a simple rule? |
| D | **Deterministic ARGUS, no LLM** | **Does the LLM add anything at all?** |
| E | Single LLM agent | Does multi-agent add anything? |
| F | Multi-agent, no Constitution | Does the risk layer add anything? |
| G | Full ARGUS | The system as submitted |
| H | Human + ARGUS | Does the human-in-loop mode beat autonomy? |

**Baseline D is the one that matters most and the one most projects avoid.** If the LLM does not beat its own deterministic twin on identical data, we report that finding rather than hiding it. An honest negative on D is more credible than an unexplained positive on G.

### 7.4 The twelve workbenches

One interface, twelve surfaces. A judge moves from evidence to reasoning to risk to execution to outcome without ever changing products:

**1.** Live Market · **2.** Event Radar · **3.** Sentiment Integrity · **4.** Earnings Lab · **5.** Cross-Asset Map · **6.** Price Discovery Twin · **7.** Factor Lab · **8.** Pre-Trade Risk · **9.** Execution Planner · **10.** Decision Ledger · **11.** Agent Evaluation · **12.** Capital Authority · **13.** Challenge Mode (§6.12a)

Each workbench is the visible surface of a subsystem specified in §6 — they are not separate products, and none of them exists without a mechanism behind it.

### 7.4a The workbench is a graded deliverable, not a skin

`t3-workbench` is the weakest cell in the acceptance matrix (§7.2a) and it is weak for a specific reason: a backend can be world-class and still lose to a team with a more convincing workstation, and G5 there requires beating a *UX* baseline we had not previously named. We name it now.

**Baseline:** the current class of public AI trading desks — multi-agent research, portfolio view, market context, SEC RAG, sentiment, paper trading, decision history. That is the bar, and it is already met by shipped open-source projects. A chat box in front of charts loses to it.

**What makes ARGUS's desk measurably better rather than prettier** — personalisation must change *decisions*, not only layout. The persistent user profile is an input to the Constitution Kernel, not a theme setting: risk budget, holding horizon, preferred evidence classes, portfolio constraints, deployable capital, sector exposure limits, strategy preferences and granted authority all bind. The output that proves it is the sentence a generic desk cannot produce:

> *"This is a good trade in isolation and a bad trade for your book: it raises your semiconductor concentration from 22% to 31%, and your mandate caps it at 25%."*

**Graded on:** time-to-first-insight for an unprimed judge, number of products they must leave to complete one decision (target: zero), whether every number on screen is traceable to a source in one click, and whether the same profile produces demonstrably different decisions for two different users on identical market state. Those are measurable; "looks institutional" is not.

### 7.5 First-class metrics

Metrics are grouped by what they actually measure, because collapsing everything into one number destroys information.

| Group | Metrics |
|---|---|
| **Agent** | Paper Sharpe, Sortino, max drawdown, win rate, turnover, decision consistency, calibration, intervention rate, abstention quality |
| **Information** | Extraction accuracy, source-grounding accuracy, timestamp validity, contradiction detection, event classification accuracy, **event attribution accuracy** (did we identify the part of the event that actually explained the repricing?) |
| **Risk** | Expected shortfall, marginal CVaR, risk-forecast calibration, violations prevented, false-positive vetoes |
| **Execution** | Implementation shortfall, slippage, market impact, fill probability, latency, partial-fill rate, failed-leg recovery time, **Risk Neutralisation Efficiency** |
| **Research** | Hypothesis survival rate, DSR, PBO, IC, turnover, factor decay, capacity, cross-regime performance |
| **Authority** | Return per unit risk, calibration, policy compliance, stability, evidence quality, unresolved incidents |
| **Sentiment** | **Incremental Sentiment Value** — how much better the integrity model performs than raw sentiment after fees, controlling for price and volume |

Two of these are original metrics worth naming explicitly because they do not exist in the literature: **Risk Neutralisation Efficiency** and **Incremental Sentiment Value**.

---

## 8. Source & licence ledger

**This is a legal constraint, not a preference.** Licences below were read from the actual `LICENSE` files in our local clones on 2026-09-12, not from memory or README badges.

### 8.1 Verified licences

| Repository | Licence | Disposition | Note |
|---|---|---|---|
| **hftbacktest** | **MIT** | **COPY** | Queue models, recorded-latency interpolation, mandatory fee model |
| **qlib-official** | **MIT** | **COPY** | Factor engine with lookback guard |
| **RD-Agent** | **MIT** | **COPY** | Hypothesis→backtest→feedback loop, bandit, IC dedup |
| **langgraph** | **MIT** | **COPY** | Pregel checkpointing, durable agent state |
| **ConvFinQA** | **MIT** | **COPY** | Retrieve→program→execute numeric reasoning |
| **tau-bench** | **MIT** | **COPY** | pass^k, deterministic state-diff reward |
| **bitrinjani~r2** | **MIT** | **COPY** | `SingleLegHandler` — the single most valuable small component we found |
| **PyPortfolioOpt** | **MIT** | **COPY** | Shrinkage, Black-Litterman, HRP |
| **hummingbot** | **Apache 2.0** | **COPY** (+ NOTICE) | ArbitrageExecutor, BudgetChecker collateral locking |
| **zipline-reloaded** | **Apache 2.0** | **COPY** (+ NOTICE) | `SimplePipelineEngine` |
| **cvxpy** | **Apache 2.0** | **COPY** (+ NOTICE) | PSD verification, DPP caching, strict OPTIMAL-only solving |
| **finBERT** | **Apache 2.0** | **COPY** (+ NOTICE) | Sentiment baseline |
| **skfolio** | **BSD** | **COPY** | Portfolio science |
| **ABIDES** (both clones) | **BSD 3-Clause** | **COPY** | Agent-based market simulation; `EtfArbAgent` quote-guard |
| **vectorbt** | **Apache 2.0** | COPY with care | **Its Deflated Sharpe silently returns NaN — do not use as a gate** |
| **nautilus_trader** | **LGPL** | **LINK, DO NOT VENDOR** | Usable as a library; modifications to Nautilus itself carry share-alike duty |
| **freqtrade** | **GPL** | **REBUILD** | Viral on distribution. Reimplement the depth-weighted fill logic |
| **lfest-rs** | **AGPL** | **REBUILD** | **Most dangerous licence here** — network use would force our entire stack open |
| **Riskfolio-Lib** | **"All rights reserved"** | **REBUILD** | Not permissively licensed **and** its EVaR/TG report values are swapped |
| **arbitragelab** | **NO LICENCE FILE** | **REBUILD** | No licence = all rights reserved by default. **We may not copy this code.** Its methods — Johansen, Engle-Granger, Leung & Li optimal double-stopping — are published academic work and freely reimplementable from the papers |

#### 8.1a Second licence sweep — 25 Track-2 repositories, read from the LICENSE files (2026-09-12)

Verified by reading each `LICENSE` on disk, not from READMEs or search results. **Thirteen of
twenty-five are restricted**, which is materially more than any secondary source reported.

| Repository | Licence | Disposition | Consequence |
|---|---|---|---|
| **cvxportfolio** | **GPL-3.0** | **REBUILD** | Previously assumed permissive and used as a design reference. It is copyleft — vendoring forces the whole stack open. **The Constitution Kernel is therefore built on raw CVXPY (Apache), not on cvxportfolio.** Its two ideas are rebuilt: `SimulatorCost` (same expression, forecast parameters for the optimiser and realised for the simulator — `costs.py:347–451`) and γ = 1.5 market impact with convexity enforced at ≥ 1.0 (`costs.py:826`, `845–848`) |
| **open-trading-platform** | **GPL-3.0** | **REBUILD** | Parent/child order model studied, never copied |
| **eventedge** | **Proprietary — "Copyright 2026 Pedro Otalora. All rights reserved."** | **REBUILD** | The strongest event-driven system found, and explicitly proprietary. Its event taxonomy and event-study machinery are re-derived from MacKinlay 1997, which is method, not code |
| **live-trade-bench** | **PolyForm Noncommercial 1.0.0** | **REBUILD** | **Cannot be embedded in the product at all.** Its synchronised-distribution and allocation-normalisation mechanisms must be rebuilt, not copied |
| **traderbench** · **trata-hedge-bench** · **factorforge** · **quantbyqlib** · **levkila-trade** · **crypto-trading-agent** · **moss-trade-bot-skills** | **NO LICENCE FILE** (7 repos) | **REBUILD** | Absence of a licence is not permission. Every mechanism is rebuilt from observed behaviour with the source named as prior art |
| **bastion** · **hftbacktest** · **docling** · **clawock** · **vibe-trading** · **agent-rita** · **finmem** · **alphaforgebench** · **factorminer** · **financebenchmark** · **llm_trader** · **atrx-demo** | **MIT** | **COPY** | Clean |
| **fineval** · **FinRobot** | **Apache 2.0** | **COPY** (+ NOTICE) | Clean |

**The wider corpus adds eight more no-licence repositories**: `Meridian402/meridian`,
`RedGnad/Neutrino`, `0ncharted/Storkshield`, `KangOxford/AlphaTrade`, `cmarvinzurich/RL-LOB`,
`HowardLiYH/PopAgent`, `Nunchi-trade/auto-researchtrading`. Two of these — AlphaTrade and RL-LOB —
are the source of our execution action-space design, which is consequently REBUILD.

### 8.2 Rules that follow

1. **MIT / BSD / Apache** → copy freely. Apache requires preserving the NOTICE file. Attribution always.
2. **LGPL (nautilus_trader)** → use as an unmodified library dependency. Do not fork-and-embed.
3. **GPL / AGPL (freqtrade, lfest-rs)** → **never** copy code. Read, understand the mechanism, write our own. AGPL in particular would force us to open-source the entire ARGUS stack the moment it serves anything over a network.
4. **No licence or restrictive (arbitragelab, Riskfolio-Lib)** → treat as all-rights-reserved. Reimplement from the underlying published papers, which are not copyrightable as methods.
5. **Every copied file** keeps its original licence header, and every third-party component appears in a public `THIRD-PARTY-NOTICES.md`.
6. **Originality ledger** — a public table stating, per subsystem, the external inspiration and our specific addition. Transparency here strengthens the submission; pretending open-source work is ours would be fatal if noticed.

### 8.2a Newly acquired sources (2026-09-12)

Fourteen repositories were cloned to `research/repos-new/` after the initial specification, because they name capabilities our corpus did not cover. Licences read from actual files.

| Repository | Licence | Why it matters | Disposition |
|---|---|---|---|
| `adrydevel~bastion` | **MIT** | **Direct competitor** — autonomous verifiable AI fund for tokenized stocks on Robinhood Chain; claims agent swarm, risk kernel, CVaR, Kelly, regime allocation, cryptographic proof, deterministic replay | STUDY + COPY where useful |
| `Meridian402~meridian` | **NONE** | **Direct competitor** — live autonomous tokenized-equity market making, portfolio breakers, P&L attribution, on-chain transparency, x402 agent payments | STUDY ONLY — all rights reserved |
| `RedGnad~Neutrino` | **NONE** | **Direct competitor** — AI proposal → deterministic policy → risk validation/override → canonical receipt → RWA execution | STUDY ONLY |
| `0ncharted~Storkshield` | **NONE** | **Direct competitor** — tokenized-stock perps, explicitly built for overnight gaps, earnings volatility and market halts | STUDY ONLY |
| `barj28~llm-leakage-instrument` | **MIT** | Measures whether a model recalls realised financial facts — separates capacity from genuine skill | **COPY** |
| `flysheep-ai~NoAlpha` | **MIT** | Point-in-time and leakage discipline | COPY |
| `DimaMerc~TieOutBench` | **MIT** | Earnings workflows scored with source-grounded rubrics, gate failures and calibrated uncertainty rather than answer similarity | COPY |
| `toininoi~alphaagent` | **MIT** | Idea → Factor → Evaluation decomposition with anti-duplication regularisation | COPY |
| `dtbtc~mcts-llm-alpha` | **MIT** | Reproduction of LLM+MCTS factor mining with frequent-subtree avoidance | COPY |
| `cheer932041235~FinAgent-RAG` | **MIT** | Financial retrieval workflow | COPY |
| `KangOxford~AlphaTrade` | **NONE** | JAX-LOB, GPU-accelerated order-book simulation for RL execution | STUDY ONLY |
| `cmarvinzurich~RL-LOB` | **NONE** | Actor-critic optimal execution with TWAP/SL baselines | STUDY ONLY |
| `HowardLiYH~PopAgent` | **NONE** | Agents learning *which methods to use* via Thompson sampling — the basis for §5.2d | STUDY ONLY, reimplement the algorithm |
| `Nunchi-trade~auto-researchtrading` | **NONE** | Self-modifying strategy loop: edit, backtest, keep or revert | STUDY ONLY |

**Four of these are live competitors occupying our exact territory.** That materially changes the differentiation argument and is analysed in §8.2b.

### 8.2b Competitive landscape — what is no longer differentiating

This is the most important strategic update since the original specification. Four systems already attack tokenized-equity autonomous trading:

- **Bastion** — agent swarm plus risk kernel, CVaR, Kelly sizing, regime allocation, memory, cryptographic proof and deterministic replay, on tokenized stocks.
- **Meridian** — live autonomous market making for tokenized equities, with risk caps, portfolio breakers, execution, P&L attribution and public on-chain transparency.
- **Neutrino** — explicitly AI proposal → deterministic policy → risk validation → canonical receipt → RWA execution.
- **StockShield** — tokenized-stock perps, explicitly targeting overnight gaps, earnings volatility and market halts.

**These four were torn down against their actual source code. Full evidence with `file:line` in `research/COMPETITOR-TEARDOWN.md`.** The results materially change what we may claim.

#### What survives as genuinely ours — verified

**Zero of the four compute hedgeable-now versus carried, in any form — binary or continuous. Zero compute marginal VaR/CVaR of a proposed trade.** Both foundation differentiators (§5.2 #1 and #2) survive intact against the closest competition in the world. This was checked in code, not inferred from READMEs.

#### What is now table stakes — stop claiming it

| Claim we can no longer make | Who already ships it |
|---|---|
| "The market closes overnight but tokenized stocks don't" | This is **Neutrino's entire pitch** and **Storkshield's entire mechanism** — `StockOracle.sol` freezes price and `LiquidationBrake.sol` pauses rebalancing when the underlying is shut, with a 15%/40% gap-severity ladder |
| "We have a portfolio circuit breaker" | Meridian's `portfolioBreaker.ts` is real, incident-driven and wired into **every** entry path |
| "AI proposes, a deterministic risk layer disposes" | Neutrino's whole architecture; Bastion claims it too |
| "Autonomous, transparent, with replay" | Meridian is live with on-chain transparency |

Observing the asymmetry is not a contribution. **Quantifying and pricing it is.**

#### 8.2b-ii Second competitive sweep — 25 code-level architecture teardowns (2026-09-12)

`research/architecture/` now holds 25 teardowns totalling ~149,000 words, one per system, each with
`file:line` evidence. Three findings change the strategic picture.

**Finding 1 — Bastion has no session awareness at all.** The closest system to ARGUS in the world —
an autonomous AI fund for tokenized stocks — has **zero** market-hours modelling: no session state,
no NAV staleness, no oracle freshness, no gap risk. It treats every on-chain price as fresh 24/7.
Furthermore, much of its advertised risk machinery is dormant: **CVaR is computed and never enforced
in sizing**, the circuit breaker never fires, the sentiment agent is wired to no news source, and
execution is stubbed — the open-source build places no real trades. The marketing is ahead of the
code. This is now the strongest available evidence that the Sleeping-Anchor problem is unclaimed.

**Finding 2 — most of the field fails Bitget's own Track-2 positioning rule.** The rule requires the
LLM to be the *primary decision-maker* and to *autonomously place orders under risk control*.

| System | LLM decides? | Evidence | Track 2 |
|---|---|---|---|
| AI-Trader (HKUDS) | **No** | LLM writes market-summary prose; the trading path never consults it (`routes_signals.py`) | **Fails**, despite advertising "100% fully-automated agent-native" |
| Vibe-Trading (HKUDS) | **No** | System prompt forbids recommendations; orders are human-gated | **Fails** — a research platform |
| inalpha | **No, by design** | LLM deliberately kept off the order path; execution needs a separate approval token (`trade-plan.ts:74–303`) | **Fails**, and is honest about it |
| FinRobot | **No** | Verified by grep — no broker client, no order placement, no execution path | **Fails** — a report generator |
| Bastion | Partial | Council votes; execution stubbed, CVaR unenforced | **Incomplete** |
| TradingAgents | **Yes** | Genuine multi-agent decision, emitted as a recommendation | **Partial** — no autonomous placement |
| **atrx-demo** | **Yes** | Three-tier LLM validation on a live account, 600+ trades since Nov 2025 | **Passes** — our most serious rival on this specific test |

The correct claim is not that these systems are weak — several are far more mature than us. It is
that **four of them answer a different question than the one this track asks.**

**Finding 3 — cost-blindness is the most widespread defect class in the corpus**, ahead of leakage
and ahead of self-scoring. Across 103 catalogued defects it recurs more than any other failure. That
is the direct empirical justification for the constructed-mandatory cost model (CE-5).

#### 8.2b-iii Corrections to our own earlier claims

Honesty requires retracting three things this document previously asserted, and recording one
defect we shipped and then found in our own code.

**TradingAgents v0.4.0 does not leak the way we said.** Verified per source: FRED is vintage-pinned
to `curr_date`; social and news retrieval is date-windowed through a centralised UTC-normalised
filter; fundamentals filter `fiscal_date <= end_date`; memory retrieval filters
`resolution_date <= as_of`. **Five of seven sources are genuinely protected.** Two remain open:
market OHLCV defers to yfinance (unverified), and **Polymarket is real-time only — a live leak**.
It is a serious benchmark and this document must stop implying otherwise.

**RD-Agent is weaker than its reputation, in a way that helps us.** Microsoft's automated quant R&D
loop uses a single `APIBackend()` to both propose factors and judge them — no independent evaluator.
It has **no purged cross-validation, no embargo, no deflated Sharpe, no PBO, and no trial counter at
all**. Capacity, decay and retirement are absent. It also displays `without_cost` numbers to both
proposer and evaluator while feedback reads `with_cost` — the same bug class found in `alphaagent`.

**We shipped the cost-blindness defect we spent this whole document criticising.** `plan_execution`
in the Track-3 workbench labelled slices "passive limit" and charged `maker_bps` on them, with no
book data behind the claim and no fill simulation anywhere near it. That is exactly the defect
class §8.2b-ii names as the most widespread in the corpus — *"cost-blindness, ahead of leakage and
ahead of self-scoring"* — and we had it in our own Track-3 execution planner while the Track-1
backtest engine was refusing the same thing outright.

It was found by auditing our 18 sub-themes against the goal file, not by a test, and the reason no
test caught it is instructive: every test asserted the planner's *shape* — three slices, correct
fractions, the right style labels — and none asserted that a quoted fee was **earned**. A test
suite can be complete against an interface and blind to the claim the interface is making.

Fixed: a passive slice is now priced through `argus.execution.passive`, and with no book supplied it
is quoted at the **taker** rate with the label saying so. Four tests were added, including one
asserting the un-evidenced case produces an upper bound rather than a forecast.

**The general lesson, which is the part worth keeping.** A defect our own architecture explicitly
forbids can still ship, in a module written before the enforcement existed, because nothing
re-audits old code when a new invariant arrives. The engine gained the rule; the planner predated
it. That is now a standing check rather than a hope: `argus.status.subtheme_coverage()` resolves
every sub-theme by import at runtime, so the sub-themes are at least enumerated in one place where
a sweep like this one can start.

**Search/evaluation contamination is a field-wide pattern, now with three citations:**
`mcts-llm-alpha` (`qlib_evaluator.py:143` computes a real overfitting score, `comprehensive.py:90–98`
overwrites it with the generating LLM's self-judgment), RD-Agent (shared backend), and FactorForge
(`evolution_engine.py:95–99` feeds the generator the IC of the top 3 factors and asks for
variations). **FactorMiner is the counter-example and deserves credit** — its generator sees only
syntax errors, never scores (`factor_generator.py:91–112`). We cite it as prior art rather than
claiming separation as ours. What remains ours is the **trial counter** and **cost-aware IC gating**,
which none of them has.

#### 8.2b-iv What no evaluation harness does

Across Microsoft's FinanceBenchmark, Trata's Hedge-Bench and SUFE's FinEval, **eight capabilities are
absent from all three**: probabilistic calibration (ECE / Brier / reliability), decision consistency
under replay, abstention quality, agent-contribution attribution, point-in-time integrity testing,
adversarial resistance, cost-aware evaluation, and live streaming evaluation. Microsoft's harness
additionally grades with **gpt-52 regardless of which model is under test** — an unguarded LLM judge;
Trata at least uses a different model family. This is the evidential basis for Submission A.

**Challenge Mode survives contact with prior art.** TraderBench is the closest existing adversarial
finance-agent benchmark and covers **one** of our thirteen controls outright (destroy liquidity) and
three partially. Nine are genuinely ours.

#### What a competitor does better than our current plan

**Meridian's cost model is empirically measured, not assumed.** It derives ~3% fee plus ~4.5% impact from real simulated fills against live liquidity, dated and sized. Our "constructed-mandatory cost model" (§5.2 #5) is more rigorous *architecturally* — it cannot be bypassed — but a mandatory model populated with assumed numbers is weaker evidence than a measured one. **Action: our cost model must be grounded in measured Bitget rToken fills, not a published fee schedule.** Meridian also justifies its entire risk stack with dated real-incident postmortems carrying dollar figures, a discipline our Constitution Kernel has not yet demonstrated.

#### Two claims that need hardening before demo

1. **Edge decomposition into information / timing / execution / luck has no precedent anywhere in this field.** That is simultaneously our opportunity and our exposure — with no prior art there is no accepted methodology, so an improvised attribution will be attacked. We must state a defensible method up front: **counterfactual baselines**, where each component's contribution is the difference between the realised outcome and the outcome under a baseline that lacks it. Stated, pre-registered, and applied identically to every decision.

2. **Competence-gated capital authority needs its distinction stated explicitly.** Meridian's venue-admission and churn gates are the only precedent found — but they gate **venue selection** on realised P&L, not an **agent's own expiring execution mandate**. None of the four systems enforce authority *expiry*. Our pitch must draw that line clearly, and the code must actually enforce expiry rather than describe it.

#### Quality verdict on the competition — useful context

- **Bastion** has correct CVaR, Kelly, circuit-breaker and bandit mathematics in `src/risk/cvar.ts` and `src/regime/bandit.ts` — but **it is dead code**, never called by `Council.decide()` or `commands.ts`. `RobinhoodChain.writeProof()` and `.execute()` are stubs that never touch a chain. Its risk-kernel README is largely aspirational. This is the same pattern we found in AutoHedge and FinRL-Meta: impressive machinery with no callers.
- **Storkshield's** headline "4-stage Liquidation Brake" does not exist — grepping `executeStage1-4` returns zero hits and only one stage is implemented. Keeper rewards are never paid. Position-opening never touches the deployed contract; `getCurrentPrice()` is `base * (1 + random*0.02)` written into a database row. Contracts and product are disconnected.
- **Meridian is the one genuinely live, production-grade system** in the batch — 93 tests, dated incident postmortems. But its differentiators are venue selection and cost discipline, **not the RWA gap-risk problem itself**.

**Strategic read:** the territory is occupied but thinly. Two competitors are substantially demo-ware; one is genuinely good at a different problem. The specific gap we identified — pricing risk you cannot hedge because the anchor market is shut — remains open, and we now have code-level proof of that rather than an assumption.

### 8.2c Tokenized-equity literature — directly on our thesis

Four 2026 papers address our exact market structure and must be read before the After-Hours and Arbitrage cells are finalised. These strengthen the thesis and also constrain it.

| Work | Finding that matters to us |
|---|---|
| **Tokenized Stocks** — Cong, Landsman, Rabetti, Zhang & Zhao | Studies hundreds of tokenized stocks and examines price discovery under continuous fractional trading |
| **Fractional and around the clock** — *J. Int. Fin. Markets, Institutions & Money*, 2026 | Although tokenized assets trade 24/7, activity remains far heavier during primary-market hours and **price gaps widen outside core hours** — consistent with limits to arbitrage. This directly corroborates our cost findings |
| **Tokenized Stock Premiums, Yield Pass-Through Rights, and the Limits of Off-Hours Price Discovery** — Arshadi & Dombrowski | Modest weekday price discovery but **stronger weekend predictability** in some tokens. This is the single most encouraging empirical result for our after-hours thesis |
| **Trading in the Dark: Price Discovery and the Redemption Gap in 24/7 Tokenized Equities** — Weinberg | Robinhood Chain tokenized equities and the **absence of a direct ordinary-holder arbitrage channel** — which is precisely why naive arbitrage fails and why the redemption gap matters |

**Read together, these say: the after-hours effect is real but concentrated at weekends, the arbitrage channel is structurally blocked for ordinary holders, and gaps widen exactly when liquidity is worst.** That is simultaneously the strongest support for our thesis and the clearest warning against the naive version of it.

### 8.2d Mandatory research library

Ordered by reading priority. Everything here is either already in `research/` or newly cloned.

**Phase 1 — core systems.** TradingAgents · AI-Trader · FinAgent · FinCon · FinRobot · ABIDES · Qlib · RD-Agent · FinRL · hftbacktest · PyPortfolioOpt · skfolio · Hummingbot · nautilus_trader (link, LGPL)

**Phase 2 — finance foundations.** *Trading and Exchanges* (Harris) · *Market Microstructure Theory* (O'Hara) · *Advances in Financial Machine Learning* (López de Prado) · *Machine Learning for Asset Managers* (López de Prado) · *Algorithmic Trading* (Chan) · *Inside the Black Box* (Narang) · Almgren-Chriss · Gatheral impact · MacKinlay event studies · Engle DCC-GARCH · Diebold-Yilmaz connectedness · TVP-VAR · Johansen & Engle-Granger cointegration · VECM · Black-Litterman · HRP · Ledoit-Wolf · Marchenko-Pastur · CVaR (Rockafellar-Uryasev) · Deflated Sharpe · PBO/CSCV · MinTRL

**Phase 3 — agent intelligence.** ReAct · Toolformer · Reflexion · MemGPT/Letta · AgentBench · tau-bench / tau2-bench · AgentDojo · LangGraph (Pregel checkpointing)

**Phase 4 — financial reasoning.** FinQA · TAT-QA · ConvFinQA · FinBERT · FinGPT · FinAgentBench · InvestorBench · TieOutBench · CARAG (agentic RAG over earnings calls → post-earnings shock; the direct t2-earnings baseline) · Arelle · EdgarTools · Docling · SEC EDGAR

**Phase 5 — autonomous quant.** RD-Agent · AlphaAgent · Navigating the Alpha Jungle (MCTS) · Chain-of-Alpha · Cognitive Alpha Mining · FinRL

**Phase 5a — execution specialists, read as benchmarks not references.** hftbacktest · ABIDES · Almgren-Chriss · DeepLOB · JAX-LOB / AlphaTrade · RL-LOB · Hummingbot ArbitrageExecutor. Every one of these is something `t3-execassist` must be measured *against* under G4, not merely cited.

**Phase 5b — the evaluation canon, weighted above its apparent size.** StockBench (dynamic, contamination-resistant sequential trading) · Agent Market Arena (live multi-market comparison) · AI-Trader (live autonomous evaluation) · FinAgentBench (document *and* passage selection) · InvestorBench · KTD-Fin · *Agentic Trading: When LLM Agents Meet Financial Markets* · *Beyond Agent Architecture: Execution Assumptions and Reproducibility in LLM-Based Trading Systems* (audits execution timing, turnover and cost assumptions across 30 trade-relevant studies). **This phase gets disproportionate effort relative to a normal hackathon**, because the field's unsolved problem has moved: it is no longer "can an LLM trade" but "can anyone prove the resulting intelligence is real, point-in-time valid, executable, risk-aware and reproducible." That is the question we are choosing to answer.

**Phase 6 — tokenized equity.** The four papers in §8.2c, then build our own proprietary dataset: rToken price · underlying price · NAV · crypto · news · earnings · macro · liquidity · session state · oracle state · reopening. **This dataset is potentially our most defensible research asset**, because it does not exist anywhere else.

**Red-alert reading — deep, not skimmed.** *Agentic Trading: When LLM Agents Meet Financial Markets* (the 77-study audit) · KTD-Fin · Navigating the Alpha Jungle · Almgren-Chriss · the four tokenized-equity papers.

**A correction to the earlier specification, now verified in code (§8.2b-iii):** TradingAgents v0.4.0 ships point-in-time fixes across macro, social sentiment and decision-log memory, plus structured outputs and checkpointing. Five of its seven data sources are genuinely protected; only market OHLCV (unverified, defers to yfinance) and Polymarket (real-time only — a live leak) remain open. It is no longer fair to treat it as the leakage-prone baseline described earlier. It should be treated as a **serious permanent challenger**, and the fact that multi-agent decomposition is now well-executed elsewhere means that decomposition alone is not our differentiator.

### 8.4 Head-to-head against the named systems — where we win, and where we do not

"Track-level best" is a claim, and our own standard says a claim requires proof or a plain statement
of what is missing. This section is that, measured against the systems the build was pointed at.
**Where ARGUS loses, it says so.**

#### Where ARGUS is genuinely ahead, with evidence

| Against | What we do that it does not | Evidence |
|---|---|---|
| **Bastion** (closest tokenized-equity competitor) | Session awareness of any kind. It has **none** — no market hours, no NAV staleness, no oracle freshness, no gap risk; every on-chain price treated as fresh 24/7. Its CVaR is computed and never enforced, its circuit breaker never fires, execution is stubbed | `research/architecture/bastion.md`; our `argus.truth.clocks` + 12-instrument attenuation validator |
| **RD-Agent** (strongest factor machinery in existence) | Search/evaluation separation, a trial counter, and a cost gate. It shares one `APIBackend()` between proposer and judge and has **no purged CV, no embargo, no DSR, no PBO, no trial count** | `research/architecture/rd-agent.md`; our `ProposerContext` has no field that can hold a score |
| **TradingAgents** (our permanent benchmark) | Costs in the prompt and risk enforced in code. Its own teardown records risk management as *"via prompt guidance, not code enforcement"*, and costs never reach the trader | `research/architecture/tradingagents.md`; our Constitution is a typed invariant, tested |
| **Every evaluation harness** (FinanceBenchmark, Hedge-Bench, FinEval) | Eight capabilities absent from all three: ECE/Brier, replay consistency, abstention value, contribution attribution, PIT probes, adversarial resistance, cost-aware scoring, live evaluation. Microsoft's grades with gpt-52 **regardless of subject** | `research/architecture/finance-benchmarks.md`; `argus.eval.observatory` + `scorecard` |
| **TraderBench** (closest adversarial benchmark) | Nine of thirteen Challenge Mode controls | 13/13 passing, `argus.eval.challenge` |
| **FinMem / FinAgent** | Point-in-time retrieval that cannot be forgotten. Both leak: FinMem never checks `temp_date_list`; FinAgent runs an unbounded `similarity_search` and puts `now + 14` in the observation | `argus.truth.facts` — omitting `as_of` is a `TypeError` |

#### Where ARGUS is behind, plainly

| Against | What it has that we do not | Honest position |
|---|---|---|
| **Qlib** | A mature data layer, a 54-operator expression DSL, nested daily/intraday execution, and years of production hardening. Our backtest engine is deliberately small | We do not compete on breadth. Ours guarantees costs are charged and signals are lagged; Qlib does far more, and for a large factor programme it is the better base |
| **NautilusTrader** | A production Rust event engine, full venue adapters, real order emulation. Our order machine is a simplified 14-state rebuild | LGPL, so linking is the intended path. We have not done that integration — the state machine is ours, the engine is not |
| **hftbacktest** | Tick-level replay against L2/L3 feeds. We advance a queue once per bar | **Largely closed, and one part inverted.** `argus.execution.queue` ports all three queue models (risk-adverse, probabilistic with all five probability functions, L3 FIFO) from `queue.rs`; `argus.execution.passive` makes a maker fee payable in fills — a passive run must supply `level_qty` and `traded_qty` per bar or it raises, and the engine reports the realised fill rate beside the returns; `argus.execution.latency` ports `ConstantLatency` and `IntpOrderLatency` from `latency.rs`, including the `exch_ts == 0` rejection convention. **The latency part is now ours more than theirs:** hftbacktest models microseconds of network delay, and our binding delay is *deliberation* — a reasoning model takes 8–40s, four orders of magnitude larger — so `DecisionLatency` and `thinking_budget_cost_bps` price a thinking budget in basis points against realised volatility. What remains genuinely theirs is **granularity**: they advance the queue per tick, we per bar. We also do not model adverse selection, so a passive result of ours is still optimistic, which `passive.py` states in its own docstring |
| **ABIDES** | A calibrated agent-based market simulator. We use it for nothing yet | Our counterfactual and stress work is scenario arithmetic, not simulation. Building on ABIDES remains the correct next step and is unstarted |
| **atrx-demo** | A live account with 600+ real trades since Nov 2025 | **The single most important gap.** Our paper ledger holds six abstentions and no order has reached a venue. Everything of ours is verified except the part that only running proves |
| **OpenBB** | A vastly larger data platform and an established user base | We compete on evidence lineage, not data breadth, and that choice is deliberate |

#### What is unproven rather than lost

* **No certified alpha exists.** 0 of 8 factors, 0 of 12 strategies past the deflation gate. We
  report that as the finding; it is not a claim of superiority over anything.
* **No venue has accepted an order.** The signed path is verified three ways from source and probed
  live to credential lookup (`40037`), but signature *acceptance* is untested.
* **Calibration is unmeasured.** The machinery is built and the ledger is too young to feed it.

**The honest summary.** ARGUS is ahead on *epistemics* — knowing what was knowable, refusing to
score what cannot be scored, separating search from evaluation, and grading reasoning rather than
outcomes. It is behind on *execution realism and operating history*, and those gaps are named above
rather than papered over. On the sub-themes where the corpus is genuinely empty — session state,
hedgeability under a shut anchor, causal-link grading, abstention value — nothing found in 922
sources does the work at all, and that is where the claim is strongest.

### 8.3 Bitget-native layer — build above it, never re-create it

`agent_hub` (89 UTA v3 operations behind 14 intent verbs), `agent-sdk`, `agent-cli`, `agent-mcp`, `agent-skill`, `bitget-signal`. All MIT. Safety flags that matter: `--read-only`, `--paper-trading` (this is what produces the paper-trading log the Agentic track requires), and `dryRun` previews.

---

## 9. Validation plan — what we must prove

The handbook says validation answers that are incomplete *"noticeably lower your score."* This section is therefore a scoring asset, and it is where most entries will be weak.

### 9.1 Every figure carries a label

**observed** (measured on real data we can show) · **estimated** (modelled, with the model stated) · **targeted** (a goal, not a result). Mislabelling here is the fastest way to lose a judge's trust.

### 9.2 Required evidence per track

**Agentic Trading (Submission A)**
- Paper-trading log, genuinely run during the competition, ≥2 weeks.
- Paper Sharpe, max drawdown, win rate — labelled observed.
- **pass^k consistency**: the same scenario replayed k times, reporting decision consistency rather than one lucky run.
- **Ablation results, not ablation designs**: system with and without the gap-risk engine, with and without the Constitution, with and without memory.
- **Risk-layer effectiveness**: count of proposed violations, final violations, and losses prevented — this is explicitly judged.
- Latency per decision, and LLM cost per decision.
- Adversarial results: attack-success rate from the §6.12 suite.

**Alpha Factory (Submission B)**
- ≥60 days total, ≥30 days out-of-sample, frozen before results are seen.
- Sharpe, Sortino, max drawdown, turnover.
- **OS/IS Sharpe ratio reported explicitly** — judges alert below 0.5×, so we report it ourselves rather than letting them discover it.
- Rolling 30-day Sharpe stability.
- Fees and slippage included, with the cost model stated.
- Deflated Sharpe **with the trial count that produced it**.
- PBO below 0.25.
- Per-symbol contribution — to pre-empt the single-instrument artefact our own prior testing found.
- Capacity estimate: the position size at which the edge dies.

### 9.2a Evaluation machinery — verified against source

Four evaluation repositories were torn down in code. Full evidence in `research/EVAL-REPOS-TEARDOWN.md`. What each genuinely supplies:

| Repo | Licence | What is real, with citation | Disposition |
|---|---|---|---|
| **llm-leakage-instrument** | MIT | The strongest of the four. Capacity probe via NLL ranking (`vr_harness.py:116-159`), a black-box MCQ analogue (`vr_C_api.py`), capacity→realisation transmission across models (`vr_04_transmission.py`), multi-cutoff regression discontinuity (`vr_11_multicutoff_rd.py`), and a from-scratch two-way fixed-effects confound check (`vr_10b_panel.py`) | **COPY the methodology.** Caveat: it requires a private licensed DuckDB (CRSP/Compustat/Polygon) absent from the repo, so it is **not independently re-runnable** — we must supply our own data layer |
| **NoAlpha** | MIT | `filter_evidence_as_of` (`pit.py:7-16`) is structurally wired into the pipeline at `pipeline.py:222-232` and **fails closed** on `event_time`. This is the second correct as-of implementation we have found, after MIRAI | **COPY the pattern.** Note the flaw: its fail-loud twin `assert_available_as_of` has **zero production callers** — the loud check exists but is never invoked |
| **TieOutBench** | MIT | A materialise → grade → gate → score pipeline, with a self-documented fail-closed fix for an 18-atom default-credit leak worth 14% of total points | COPY the rubric and gating design |
| **PopAgent** | **NONE** | Genuine Beta-posterior Thompson sampling with a contextual bonus — the basis for our agent-selection policy (§5.2d) | **STUDY ONLY**, reimplement the algorithm |

**Two defects found in PopAgent that we must not inherit:** its Jaccard-diversity anti-collapse guard only mutates `exploration_rate`, which is **never read in the Thompson-sampling branch** — the guard is dead code under the project's own default configuration. And its backtest reward is a hand-authored function of method *names* with hardcoded synergy constants rather than real strategy execution; `engine.py:406` calls it "simulated" outright. The algorithm is sound; the validation is not.

**NoAlpha's own self-audit is worth reading directly.** Its authors found and fixed three bugs of exactly the class this specification warns about: three shadow-portfolio arms that were secretly two objects, a re-compounding return bug, and a "daily loss" limit that was actually measuring all-time P&L. That a careful team shipped all three is the argument for our own adversarial self-audit.

**What none of the four supply — and therefore where our original work goes:**

1. **True probabilistic calibration.** Exhaustive grep confirms ECE and Brier score appear in none of them. TieOutBench's FailSafeQA R/G/F-beta is calibrated *abstention*, which is a different thing from a model's stated confidence matching its realised hit rate.
2. **A two-clock point-in-time model.** Every implementation assumes one market clock. An rToken fact can be tradeable on the continuous clock while remaining unpriced on the underlying's session clock — availability must be resolved against both.
3. **A fused per-decision trust score.** Contamination, evidence quality, PIT integrity, calibration and consistency exist as separate concerns or not at all; nothing combines them into one number attached to a decision.

### 9.3 Honest-evaluation protections

Because a deliberately leaking strategy can post Sharpe ≈ 35 and survive standard corrections:

- Leakage prevented **structurally** at the data interface, not detected statistically afterwards.
- Complete trial ledger; search budget accounted.
- Strategy cemetery — failures retained and reported.
- Pre-registration of the hypothesis before the out-of-sample window is touched.
- Contamination testing: BRIGHT / ticker-blind / date-blind / fully-blind, since the model may simply remember what happened.

### 9.4 Baselines we must beat

A result is meaningless without them: buy-and-hold, equal-weight, risk parity, a fixed-rule non-LLM version of the same strategy, and — for the agent — a deterministic baseline that uses identical data with no LLM. If the LLM does not beat its own deterministic twin, we report that.

---

## 10. The demonstration

One scenario exercises the entire system, and it is the scenario the hackathon is about.

> **Sunday, 03:00 ET.** A semiconductor company issues materially changed guidance. NYSE is shut and will not open for 30 hours. The rToken is trading. BTC is down 3.7%. Liquidity is thin. A viral post exaggerates the news; the underlying filing is real.

1. **Temporal truth** — shows exactly what was knowable at 03:00, and refuses everything else.
2. **Evidence** — authenticates the filing, downgrades the social claim, surfaces the contradiction.
3. **Session state** — NYSE shut, NAV frozen, next discovery in 30h, **hedgeability surface returns an empty menu**: every candidate hedge fails on execution probability or basis stability, so 100% of the risk is priced as residual and carried.
4. **Sleeping-Anchor Twin** — reopen distribution with 50/80/95% intervals, decomposed into information / transmission / liquidity / oracle / unresolved.
5. **Gap-risk engine** — marginal CVaR of the proposed trade across the closed window, in under 100ms.
6. **The epistemic panel — shown *before* the decision, not after.** Six statements, on screen, while the judge still has no idea what ARGUS will do: **what it knows** (the authenticated filing, the session state, the book), **what it does not know** (the reopening auction, whether the viral claim is independently sourced), **what it thinks** (the thesis and its size), **what disagrees** (the dissenting agent, stated in full, not summarised away), **what would break the thesis** (the specific falsifier — a guidance reaffirmation at the open), and **what can and cannot be hedged, at what price**. This is the moment the demo stops resembling every other entry. A system willing to state its own counter-case *in advance* is making a falsifiable claim; one that narrates confidently after the fact is not.
7. **Meta-PM (LLM)** — the decision, chosen against that panel.
8. **Constitution** — requested `SELL $31k rNVDA`; approved `SELL $16.5k + required hedge`, because concentration binds and the hedge menu is empty. Verdict and reason are machine-readable.
9. **Execution** — Agent Hub receives only the signed, approved instruction.
10. **Reconciliation** — fills matched to approved intent.
11. **Then the judge changes the world.** BTC −3.7% becomes −10%; or the viral report becomes true; or they open Challenge Mode (§6.12a) and break the hedge leg, poison the memory, or move the market open. Risk, hedge menu, position, confidence, execution plan and authority all recompute live, in front of them. **This is the step that has to land.** Everything before it is a system explaining itself; this is the system being contradicted and having to answer.
12. **Grading at the open** — prediction interval scored, each causality link marked hit or miss, edge decomposed into information / timing / execution / luck, authority updated.

Steps 3, 5, 6, 8, 11 and 12 are things nothing in 922 sources does.

---

## 10.1 Capital authority as a measurable state machine

Authority is the most original idea in this specification, and it must not remain conceptual. It is a state machine with an explicit score, explicit transitions and automatic demotion.

**The ladder:**

```text
OBSERVE → SIMULATE → PAPER → RESTRICTED → REGIME-SPECIFIC → QUALIFIED
```

**The score:**

```text
Authority = performance_quality
          + calibration
          + risk_compliance
          + evidence_quality
          + execution_quality
          + regime_coverage
          − instability
          − drawdown_behaviour
          − policy_violations
          − contamination_score
```

**The transitions that make it real, rather than a diagram:**

| Trigger | Consequence |
|---|---|
| Sustained measured competence in a regime | Promotion, **scoped to that regime only** |
| Time passing without re-certification | **Automatic expiry** — authority decays by default |
| Entering a regime with no certified competence | **Authority freeze** |
| Any policy violation | **Immediate downgrade** |
| Unresolved incident | Promotion blocked until closed |

An agent may therefore be trusted for NVDA earnings, untrusted for macro events, and hold zero authority on a thin weekend book — simultaneously. This is what turns "AI safety" from a posture into an economic mechanism, and it is the specific thing Bastion and Neutrino's static risk gates do not do.

## 10.2 What we deliberately do not copy

The field has recognisable anti-patterns. Each is banned by name, because each is easy to build and impossible to defend.

| Anti-pattern | Why it is banned |
|---|---|
| "Bull agent + bear agent + trader" with no measured incremental value | TradingAgents already does this well. Without ablations showing each role adds value, it is decoration |
| "Sentiment score → LLM decision" | Too shallow; the edge is the same size as the fees |
| "EPS beat → buy" | Ignores expectation, guidance and narrative — the parts that actually move price |
| "AI discovered a factor" after 20,000 trials, with no DSR, PBO or capacity | Statistically meaningless. The trial count *is* the result |
| Backtests without transaction costs | Unacceptable. 1 of 19 studies in the audit modelled costs; we will not be the twentieth |
| LLM arithmetic on financial values | Numbers are computed by code. The model interprets them and never produces them |
| Memory without `available_at` timestamps | This is look-ahead bias wearing a hat. Five widely-used memory architectures leak exactly this way |
| Production self-editing | Unsafe and uncertifiable. Self-modification is confined to the research sandbox |
| Fake or backfilled paper trading | Fatal reputational risk, and trivially detectable |
| Fifty agents and a hundred tools to make the README look large | Feature theatre. Every capability carries a baseline, an experiment, an ablation and a failure test, or it does not ship |
| Competing on *counts* — most agents, most integrations, most charts, most papers read | The landscape already contains systems with large agent counts, debate, RAG, risk kernels, live trading, on-chain proofs, self-reflection and portfolio tooling. None of those is a differentiator any more. **The only remaining differentiator is measured superiority**, and a count is not a measurement |

**The sentence the submission has to earn.** Not *"ARGUS has twenty-one innovations."* That is a count, and the row above bans it. Instead:

> *We reproduced the strongest systems we could find. Here is where they fail. Here is ARGUS. Here is the exact metric where it improves, the ablation showing our component caused it, the attack it survived, the out-of-sample result, the replay, and the paper-trading evidence.*

We do not claim to own a sub-theme because we implemented its API. We claim it because we benchmarked the best prior art and demonstrated a better system — and §7.2a is where a judge checks whether that is true.

## 11. Submission plan

### 11.1 Submission A — Agentic Trading, Open Theme

**Thesis (the highest-weighted field):** An autonomous agent trading tokenized equities faces a structural condition no existing system handles — for 65.5 hours a week it holds risk it cannot hedge, priced against a market that is shut, on liquidity 3–5× thinner than normal. ARGUS makes that condition explicit and computable: it measures what fraction of risk is hedgeable right now, prices the gap risk of the unhedgeable remainder before the order is placed, and constrains the LLM's decision accordingly. The LLM is the decision-maker; the Constitution is the boundary; every decision is graded at the next open and decomposed into the edge that caused it.

**Target user (must be specific — "all traders" is rejected):** Active crypto-native traders holding tokenized US equity exposure through weekends, with $10k–$250k of deployable capital, trading weekly-to-daily frequency, whose primary market is Bitget rTokens and whose specific unsolved problem is carrying unhedgeable overnight and weekend exposure.

**Validation:** §9.2, all labelled.

### 11.2 Submission B — Alpha Factory

Do **not** pick the sub-theme by narrative. Run all six internal Alpha cells and submit whichever produces the strongest genuinely frozen, point-in-time, cost-inclusive evidence. Current expectation, given the measured economics:

1. **Open Theme (execution-aware alpha)** — most likely. The handbook names it explicitly, and our cost stack is the strongest asset we have.
2. **After-hours information pricing** — the most rToken-native named theme, and overnight drift (~0.10%) is the only measured effect of the right size.
3. **Arbitrage** — only if a real, capacity-adjusted, cost-clearing spread is actually found. Current evidence says it will not be.

### 11.3 Non-negotiable submission mechanics

- Compliant X post link — **its absence invalidates the entry entirely**.
- Separate form submission per theme, different project name, same Bitget UID.
- "Role of the LLM" field: state precisely where Qwen was used and whether it met the need.
- Apply for Demo Day (checkbox), K3 subsidy, and the Qwen credit form.
- All materials in "Submission Materials Link" — GitHub links do not substitute for the description field.

---

## 12. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| **No real edge is found** | High | The thesis is the *risk and authority system*, not a return claim. Submission A is judged 50% on architecture. We report honest negative results rather than fabricating a number — this is defensible; a fake Sharpe is not |
| **Paper-trading log too short** | Fatal for Track 2 | Start immediately. ≥2 weeks is the stated minimum |
| **Only 2 Open Theme slots per track** | High | Open Theme is the right fit for A, but a named sub-theme is the fallback if the field looks crowded |
| **AGPL/GPL contamination** | Fatal, legally | §8.2 rules. `lfest-rs` and `freqtrade` are rebuild-only, no exceptions |
| **arbitragelab has no licence** | Legal | Reimplement from the papers. Never vendor its code |
| **Scope collapse — building 18 things badly** | High | Two submissions. One engine. Eight open sub-themes are facets of one problem, not eight products |
| **Judges see through architecture without evidence** | High | §9 exists precisely for this. Every claim labelled; ablations run; negative results published |
| **Qwen budget ($30 ≈ 7 single runs)** | Medium | pass^k evaluation at k=8 costs ~$4,352. Use k=2 plus audit (~$1,088), or self-fund; state the constraint openly |
| **S1 reuse challenge** | Medium | §2.4. State the new contribution explicitly and early |
| **LLM is decorative** | Fatal for Track 2 | Bitget requires the LLM to be the *primary* decision-maker. The Constitution constrains it; it must not merely narrate a deterministic strategy |

---

## 13. Build sequence

Ordered by dependency, and by what fails loudest if left late.

**Phase 0 — start the clock.** Paper-trading log begins now; it has a hard minimum duration and cannot be backfilled. Qwen credits applied for. X build-in-public posts begin.

**Phase 1 — foundation.** Temporal Truth Fabric (§6.1) and Session State Machine (§6.2). Nothing above these can be trusted until they are correct. The as-of firewall is copied from MIRAI's double-enforced pattern.

**Phase 2 — cost reality.** The constructed-mandatory cost model (§6.4). Building this second means every later number is honest by construction rather than corrected afterwards.

**Phase 3 — the three original engines.** Hedgeability surface, gap-risk engine, edge-decomposition ledger. These are the contributions; they are also small — the t3-tdopen compendium estimates 1,500–3,000 lines for the risk copilot.

**Phase 4 — Constitution and execution truth.** Policy, feasibility, signed authorisation, order state machine, reconciliation.

**Phase 5 — agent layer.** LangGraph-based Meta-PM with as-of-filtered memory. The LLM becomes the decision-maker inside the boundaries built in Phase 4.

**Phase 6 — validation harness.** Purged CV, DSR with trial counts, PBO, contamination tests, ablations, adversarial suite.

**Phase 7 — Alpha cells compete.** All six run against the same harness; the winner becomes Submission B.

**Phase 8 — demonstration and packaging.** The §10 scenario, the evidence bundle, the two form submissions.

---

## 14. Appendix — document lineage

| Source | What it contributed |
|---|---|
| `research/compendium/*.md` (18 files, 243,297 words) | Every mechanism, gap and false positive, with citations |
| `research/compendium/TOP2-PER-THEME.md` | The 36 owning sources |
| `research/CORPUS_CALIBRATION.md` | The ~21% genuine rate that forced honest reading of the corpus |
| `BITGET_AI_BASE_CAMP_S2_HANDBOOK_EN.md` | All rules, scoring and mechanics in §2 |
| ARGUS PRIME concept document | Capital Passport/Licence framing, Constitution Kernel, Trade Certificate schema, order state machine, AnchorBench, the guarantee/non-guarantee framing |
| Early design notes (not published) | The repository canon, BUILD/BENCH/STUDY/WATCH taxonomy, the acceptance-condition formalism |
| Local licence audit, 2026-09-12 | §8, read from actual LICENSE files |
| NightDesk, binance-governor, horos | Prior art in gating, signed decisions, and public accuracy records |

**On the ARGUS PRIME concept document specifically:** its central framing — that capital authority should be earned rather than granted — is the strongest idea in the input material and is carried through here largely intact. What this document adds is the measured evidence for *which* gaps are real, the licence constraints that determine what can actually be copied, and a scope narrow enough to finish.


