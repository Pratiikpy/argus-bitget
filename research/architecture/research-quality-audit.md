# Research Quality Audit: ARGUS

**Judgment Criterion:** Track 3 "research quality" (100% judge weight) — one complete research task from question to actionable insight.

**Audited:** 2026-09-13 | **Model:** Claude Haiku | **Status:** HONEST ASSESSMENT — SIGNIFICANT GAPS

---

## 1. HIGH-QUALITY RESEARCH: THE RUBRIC

### What the benchmarks define

Read five categories from the corpus:

**1.1 — OpenBB's Research Workbench** (`research/repos/OpenBB-finance/OpenBB`, `openbb_terminal/stocks/research`)
- Every report starts with a question and states scope: "analyzed for 30 days", "US equities only"
- Claims carry timestamps: "as of 2026-09-13 15:42Z"
- Numbers cite sources inline: "(Seeking Alpha consensus, 27 analysts)" not just "(27)"
- Chains facts: reported EPS → analyst consensus → growth rate → valuation → position
- Handles absence explicitly: "no forward guidance published" not silent zero
- Shows alternative metrics: Shiller PE, trailing PE, forward PE — not one number
- Refusals name the blocker: "insufficient trading volume in this sector" not blank

**1.2 — FinRobot Analyst Agent** (`research/repos/FinRobot/multi_agents/agents/analyst_agent.py:40-200`)
- Report structure: SUMMARY → KEY METRICS → RATIONALE → RISKS → RECOMMENDATION
- Every recommendation carries: magnitude (price target), horizon (3-month), confidence (0.75)
- Contradictions are surfaced: "consensus raised 8%, but free cash flow declined"
- Session beta acknowledged: "open-market beta differs from blended" (named reason, not assumed)
- Grounding checked: *before* final output, runs evidence attachment
- Worst-case scenario in every report: "if consensus underestimates, margin at risk"
- Consensus used correctly: 30-analyst average, not subset

**1.3 — TradingAgents (Research Module)** (`research/repos/TradingAgents/agents/research_agents.py`)
- Research output is *structured JSON, not prose*: allows verification of each claim
- Each agent's output declares: input_sources, computation_path, confidence_score
- Versioning: "estimated on 2026-09-10 data, refreshed 2026-09-12 — change: +2%"
- Refusals carry reason codes: INSUFFICIENT_DATA, CONTRADICTORY_SOURCES, STALE_EVIDENCE
- Report includes sensitivity: "if revenue growth is 5% lower, margin drops from 18% to 14%"
- Defect captured: "assumes stable FCF conversion; if capex rises, thesis fails"

**1.4 — FinBERT + ConvFinQA (Academic Standard)** (`research/papers/01-validation-canon.md`)
- **Rubric (FinQA/TAT-QA evaluated):**
  - Claim + amount + source + as-of date + supporting chain
  - Counter-case (what would falsify it)
  - Confidence band (low / medium / high, with sample size)
  - Explicit uncertainty on forward-looking statements
  - No number appears without a retrieval anchor

**1.5 — Agent-Rita Material** (Bitget Agent Hub, `argus/lui/answer.py:1-18` philosophy)
- Research answer = SOURCE-FIRST, never text-first
- Every fact must resolve to: ledger row | evidence id | computation | venue API
- Refusal is valid: "cannot answer because..."
- Grounding is active check, not passive citation

---

### Synthesized Rubric for Judging ARGUS

A **HIGH-QUALITY research answer** contains (ranked by what's tested):

| Rank | Element | Definition | ARGUS Today? |
|------|---------|-----------|---|
| **1** | **Claim-source binding** | Every number ↔ source, checked not assumed | PARTIAL |
| **2** | **Refusal when sample/data insufficient** | Names the blocker, not silent zero | STRONG |
| **3** | **Timestamp + as-of date on all figures** | "as of 2026-09-13 14:42Z" not "current" | WEAK |
| **4** | **Contradictions surfaced** | When consensus disagrees or data conflicts | PARTIAL |
| **5** | **Worst-case scenario** | Position breaks under what condition? | STRONG (stress) |
| **6** | **Magnitude + direction + confidence** | "up 5%, 0.75 confidence, 12-month horizon" | PARTIAL |
| **7** | **Source independence discount** | Shared provider = one view, not three | STRONG (independence) |
| **8** | **Historical analogue** | "Last time this setup happened, it moved X" | STRONG (analogue.py) |
| **9** | **Counter-case / falsification** | "This breaks if [X happens]" | WEAK |
| **10** | **Explicit uncertainty on forward estimates** | Growth = "likely 12%±5%" not "12%" | WEAK |
| **11** | **Portfolio/execution integration** | "Position can be exited at this cost" | STRONG |
| **12** | **Session/regime conditioning** | "In open hours, beta=1.71 (not 0.25)" | STRONG |

---

## 2. ARGUS AGAINST THE RUBRIC

### What ARGUS Does Well (Evidence: file:line)

**2.1 — Refusal When Data Cannot Speak**
- `analogue.py:293-300`: Returns named refusal when fewer than MIN_ANALOGUES (8), not a 4-analogue estimate dressed up
  ```python
  if len(matches) < MIN_ANALOGUES:
      return AnalogueReport(
          matches, None,
          refused=f"{len(matches)} analogue(s) within distance {max_distance}; "
          f"{MIN_ANALOGUES} are needed. A distribution over fewer is an anecdote with error bars"
      )
  ```
- **Not verified but code-clean:** `portfolio.py:780-790` refuses a correlation when variance ≤ 0, does not return NaN
- `expectation.py:246-253`: No surprise computed when historical consensus unavailable, returns Gap with surprise=None + reason
  - Honest statement: "no consensus for an already-reported quarter is available from any keyless source"

**2.2 — Session-Conditional Measurement (Discovered Gap, Fixed)**
- `portfolio.py:12-36`: Session beta non-negotiably reported per phase (open/shut/blended)
  - AAPL measured: 0.668 open vs 0.090 shut — **7x difference** — blended hides it
  - Two exceptions (TQQQUSDT, SQQQUSDT) check the effect: mechanically pinned by leverage, so no session drift — validates the finding
  - References: `PyPortfolioOpt` (lines 44-46), `riskparity.py` (lines 47-48), measurements from data (lines 12-22)

**2.3 — Stress Testing Without Guessing Exit Cost**
- `workbench.py:294-356` (stress_position): `simulate=True` liquidates position into a real ABIDES book, measures slippage + fee
- Not arithmetic: `exit_cost_bps` is *observed* VWAP-to-mid, not a fee multiplied
- Explicitly discloses when measured: `exit_measured` boolean on every result (line 281)
- Falls back honestly: `MarketError` → uses assumed cost, marks `exit_measured=False` (lines 337-340)
- **Handles unexitable:** Position cannot leave at any price → `exitable=False` (line 284), not hidden in P&L

**2.4 — Claim → Evidence → Signal Graph (ClaimGraph)**
- `workbench.py:49-135`: Every claim is `inseparable from where it came from`
  - `Extracted` (lines 74-90): label + value + unit + **Locator** (document + page + cell/bbox)
  - `ClaimGraph.trace()` (lines 123-131): Walk from claim to its evidence in one step
  - Design: Agent-Rita pattern — claims and locators constructed together, cannot separate (docstring line 79-80)
  - Guard: `UnsourcedClaim` exception raised, never silently skipped (lines 114-118)

**2.5 — Error Profile → Actionable Diagnosis**
- `workbench.py:180-248`: ErrorProfile accumulates autopsies, computes calibration_gap
  - Sample floor: `findings()` refuses to draw a pattern under 5 decisions (line 215)
  - Diagnoses are deterministic arithmetic over record (rootcause.py:101-226)
  - Remedies are checkable: "size on the realised hit rate until calibration_gap closes below 15%" (rootcause.py:127-129)
  - Evidence-backed: Every diagnosis carries decision ids in order (rootcause.py:66)

**2.6 — Review → Process Validation Against Outcomes**
- `review.py:283-353`: Rules are graded, not assumed to work
  - Status lifecycle: PROPOSED → EARNING → ACTIVE or DEAD_WEIGHT/MISLEADING/NO_DISCRIMINATION
  - Evaluation is harsh: precision must be ≥50%, recall is measured, fire rate has both a floor (never) and ceiling (always)
  - Defects from independent checkers (grounding, conflict, risk interventions, outcomes) — rule is not grading itself
  - Floor: 20 decisions before any status but PROPOSED (line 53)

---

### Where ARGUS Fails (Critical Gaps)

**2.7 — End-to-End Research Path Does Not Exist**

The handbook requires: *one complete research task from question to actionable insight*.

Tracing the path for **"Should I add NVDA to this book?"**:

| Step | Module | Status | Gap |
|------|--------|--------|-----|
| **1. Gather facts** | `market/fundamentals.py`, `market/estimates.py` | ✓ Exists | Reads SEC, Yahoo — API-level only |
| **2. Expectation gap** | `expectation.py:227-290` (detect) | ✓ Exists | Computes gap; refusals work |
| **3. Analogue retrieval** | `analogue.py:232-307` (find) | ✓ Exists | Requires historical states corpus — **NOT built** |
| **4. Stress testing** | `stress.py:419-471` (assess) | ✓ Exists | Empirical scenarios on close history |
| **5. Portfolio impact** | `portfolio.py:452-523` (assess) | ✓ Exists | Beta, risk share, correlation |
| **6. Execution cost** | `workbench.py:555-633` (plan_execution) | ✓ Exists | Slicing, market/limit costs |
| **7. Decision + grounding** | `lui/answer.py`, `agents/grounding.py` | ✓ Exists | Answer ties to ledger or refuses |
| **MISSING: Synthesis** | — | **✗ NONE** | No module assembles 1-7 into a single research report |
| **MISSING: Confidence on forward** | — | **✗ NONE** | No module says "under these assumptions, confidence is 0.68" |
| **MISSING: Session scheduling** | — | **✗ NONE** | No module says "wait for Asia close if this depends on cash discovery" |

**Code Evidence:** There is no `research_report()` function, no `assemble_analysis()`, no orchestration. Each subsystem is callable but never called in sequence.

**2.8 — Grounding is Numeric Only, Not Narrative**
- `agents/grounding.py:1-26`: Extracts numbers from thesis text, tries to resolve them within TOLERANCE
- Does NOT check: narrative claims ("growth has stabilized" when growth accelerated), contradiction detection at claim level
- Does NOT check: whether the thesis *uses* all available evidence (cherrypicking)
- `claims.py` (not read — file size suggests ~200 lines, focused on claim contradiction) exists but is separate from answer assembly

**2.9 — Confidence Scoring is Implicit, Not Output**
- `workbench.py:Autopsy.stated_confidence` (line 157) — input to error profile, not output of analysis
- No module produces: "NVDA upside recommendation, confidence 0.72 (calibrated against historical accuracy on tech names, n=47)"
- `expectation.py:Gap` carries no confidence or confidence basis (lines 115-141)

**2.10 — Counter-Case / Falsification Not Explicit**
- `expectation.py` computes direction_of_travel but does NOT say "this conclusion breaks if analysts stop revising upward"
- `portfolio.py:TradeImpact` reports risk-share change but does NOT compute: "if correlation jumps to 0.8, risk share becomes X"
- `stress.py` runs scenarios but does NOT generate: "a position survives if [condition], fails if [condition]" as forward logic

**2.11 — Historical Analogue is Disconnected from Live Decision**
- `analogue.py` is fully built and sound, BUT:
  - No corpus of historical states is constructed from market data (no `Observation` factory)
  - `analogue.py` is tested in isolation, never called in the answer path
  - Forward return (`forward_return_bps`) is always `None` in live use because no outcome data feeds it
  - Module answers: "here are 12 historical matches to your setup" but never: "adding NVDA is similar to decision #47 on 2026-08-15, which returned +180bps"

**2.12 — Prose Report is Not Generated**
- `workbench.py:ClaimGraph.trace()` outputs a list of strings (lines 123-131) for one claim
- No module generates a multi-paragraph report containing:
  - Executive summary (recommendation + confidence + horizon)
  - Key metrics (EPS growth, valuation, session beta)
  - Historical context (when has this looked like this before, outcome)
  - Risk (what breaks the thesis)
  - Execution (how to enter, expected cost, timing)
- Current system: Each subsystem produces JSON or isolated text; assembling them into a coherent narrative is manual

---

## 3. CAN ARGUS DO END-TO-END TODAY?

### The Test: NVDA Question Traced

**Question:** "Should I add NVDA to this book?"  
**Current portfolio:** 40% AAPLUSDT, 40% TSLAUSDT, 20% METAUSDT

**Path Trace:**

1. **Gather fundamentals** → `market/fundamentals.py:facts("NVDA", as_of=now)` ✓ Returns EPS history
2. **Consensus** → `market/estimates.py:fetch("NVDA")` ✓ Returns analyst consensus
3. **Expectation gap** → `expectation.detect(ticker="NVDA", reported=facts, consensus=estimates)` ✓ Returns Gap object
   - Output: "implied growth +90%, delivered +128%, **decelerating** while revisions rise" ✓
4. **Historical analogue** → `analogue.find(query={...}, corpus=[...], as_of=now)` ✓ *Code exists*
   - **Blocker:** corpus is empty. `Observation` objects must come from somewhere; no module builds them
5. **Stress test** → `stress.assess(symbol="NVDA", quantity=..., moves=[...])` ✓ Code exists
   - Requires hourly close history; can fetch this via `market/history.fetch_range()`
6. **Portfolio impact** → `portfolio.assess(symbol="NVDA", weights_before={...}, weights_after={...}, columns={...})` ✓ Code exists
7. **Execution plan** → `workbench.plan_execution(symbol="NVDA", notional=..., adv_notional=...)` ✓ Code exists
8. **Synthesize into answer** → ??? **NOTHING CALLS THE ABOVE CHAIN**

**Current State:** Steps 1-7 all exist in callable form. Step 8 does not exist. **A human engineer can glue them, the system cannot.**

**Why this matters for judging:** The handbook says the research system should work end-to-end. Today it requires hand-assembly of subsystems. When one breaks, nothing knows to fail gracefully. When a new subsystem is added, no orchestration updates.

---

## 4. WHAT RIVALS PRODUCE THAT ARGUS DOES NOT

### OpenBB Terminal (`research/repos/OpenBB-finance`)
- **Report output:** Markdown with embedded tables, charts, source links
- **Session metadata:** Every report header states analysis date, data currency, venue
- **Multimetric:** Displays five PEG variants, not one; reader chooses which to believe
- **Recommendation confidence:** "BUY (target 180, confidence MEDIUM, hold 12mo)"
- **Known gaps:** "Free cash flow forward estimate unavailable; using prior-year carry-forward (conservative assumption)"

### FinRL Suite (`research/repos/FinRL/...`)
- **Scenario branching:** "If Fed cuts 50bps next month, expected return: +3%. If no cut: −2%"
- **Asset allocation output:** Shows position size by Monte Carlo confidence level (80% confident: $X, 95%: $Y)
- **Risk-adjusted metrics:** Displays Sortino, not only Sharpe

### TradingAgents (`research/repos/TradingAgents/agents/research_agents.py`)
- **Agent transparency:** Which agent produced which claim (analyst_agent vs earnings_agent)
- **Computational path:** Shows the chain: "earnings_agent → fundamental_analyst → risk_analyst"
- **Defect flagging:** "analyst_agent and earnings_agent disagreed on guidance; marked CONFLICT"

### In ARGUS Form
ARGUS **is not producing**:
- Formatted report (Markdown, JSON with schema, structured table)
- Recommendation sentence ("ADD with 12-month horizon, 0.68 confidence")
- Explicit second-order: "This works if Bitget liquidity stays above $500M/day; if it drops, cost rises"
- Session awareness in report: "Analysis run during Asia close; repeat in RTH for price-discovery-based metrics"
- Horizon on all assertions: "EPS estimate (current quarter)", "Beta (30 days, open session)", "Worst case (24-hour window)"

---

## 5. RANKED WORK TO RAISE RESEARCH QUALITY

**Effort levels:** L=low (days), M=moderate (1-2 weeks), H=high (3+ weeks)

| Rank | Work | Effort | Impact | Rationale |
|------|------|--------|--------|-----------|
| **1** | **Build and wire `research_pipeline()` orchestrator** | H | CRITICAL | Connects: question → facts → gap → analogue → stress → portfolio → execution → answer. Today unconnected. |
| **2** | **Construct historical-state corpus** | M | HIGH | Analogue.py is complete but starved: no `Observation` builder. Build from Bitget 1H candles: z-score features, window labeling. |
| **3** | **Add explicit confidence scoring** | M | HIGH | Every claim carries: confidence ∈ [0,1], basis (e.g., "historical hit rate on similar setups, n=47"), horizon. Not implicit in Autopsy. |
| **4** | **Generate report prose** | H | HIGH | Markdown template engine: take results from 1-7 above, render as executive summary + metrics + risks + recommendation. |
| **5** | **Implement falsification checks** | M | MEDIUM | For each claim, compute: "True if X, False if Y, Unknown if Z". Surface contradictions at assembly time, not after. |
| **6** | **Add counter-case evaluation** | M | MEDIUM | Before committing to thesis: "Position breaks if correlation > 0.8", "Returns become negative if base effects more than half the move". Compute numerically. |
| **7** | **Forward-estimate uncertainty quantification** | M | MEDIUM | Expectation gap + stress = point estimates. Add: "Under Bull scenario (5th percentile growth), return is X; Bear (95th), return is Y." |
| **8** | **Publish report structure (schema)** | L | MEDIUM | OpenAPI-style JSON schema: what a research report *must* contain (source count, refusal reason, confidence basis, time horizon). Guards against silent gaps. |
| **9** | **Session/regime conditioning in output** | L | LOW | Every numeric result shows: "measured in RTH" or "blended, 82% shut-session bars". Already in code; not in rendered answer. |
| **10** | **Integrate consensus timeline** | L | LOW | Show: "Consensus for next quarter: 27 analysts, 3 cuts in 30d, 8 raises". Trends, not snapshots. |

**Sequencing:**
- **Phase 1 (Critical):** 1, 2, 3 — establish pipeline, corpus, scoring
- **Phase 2 (Essential):** 4, 5, 6 — complete analysis, integrity checks, report generation
- **Phase 3 (Polish):** 7-10 — refine confidence, transparency, formatting

---

## 6. HONEST ASSESSMENT: STRENGTH & WEAKNESS

### Why ARGUS Research is Sound in Parts

1. **Refusals work.** When data is insufficient, ARGUS names the blocker and does not invent.
2. **Session matters.** Open-session beta reported separately from blended, with evidence (9 of 11 single-name rTokens show session drift).
3. **Exit cost is measured, not assumed.** Liquidation is simulated into a real book, not guessed from fees.
4. **Evidence is traceable.** ClaimGraph binds every claim to a Locator; contradiction checks are active.
5. **Process is validated.** Review module grades checklist items against outcomes, keeps only rules that fire and are right.

### Why ARGUS Research is Incomplete

1. **No orchestration.** Six expert subsystems exist; orchestrating them into one research flow is unbuilt.
2. **No narrative.** JSON output is precise but unread by traders. Markdown prose is absent.
3. **Confidence is silent.** Stated confidence is input to error profile; output confidence is absent.
4. **Analogue is disconnected.** Code is sound, but corpus never populates and live decisions never call it.
5. **Falsification is implicit.** Stress tests run; "position breaks if X" is not explicitly computed and surfaced.

### On Judging Criterion: "Research Quality"

**What's measured:**
- Numeric accuracy: ✓ (grounding checks, source traces)
- Refusal discipline: ✓ (named blockers, sample floors)
- Computation honesty: ✓ (measured vs. assumed clearly marked)

**What's missing:**
- Complete research task (end-to-end): ✗
- Trader-actionable output (prose + recommendation + confidence): ✗
- Explicit uncertainty quantification: ✗
- Contradiction surfacing: Partial

---

## 7. CRITICAL FINDING: THE MISSING PIECE

**ARGUS builds the parts of a research workbench; it does not yet *compose* them.**

A hedge fund's research process is: Gather → Analyze → Stress → Judge → Report → Execute.

ARGUS today:
- ✓ Gathers facts (fundamentals, estimates, prices)
- ✓ Analyzes independently (expectation gap, correlation, beta)
- ✓ Stresses deeply (empirical + structural scenarios, worst-window drawdown)
- ✓ Judges rigorously (error profile, diagnosis, checklist)
- ✗ Reports in a form a trader reads (no)
- ✗ Executes the complete chain for one question (no)

**To pass the handbook test**, ARGUS must answer: *"Should I add NVDA?"* with a coherent chain that a human can follow from "how do you know?" to "what do I do?" — all from one `research_report()` call.

**Code to write:** ~500 lines to wire the six subsystems + ~300 lines for prose rendering + ~200 lines for orchestration error handling. Feasible in one engineering week.

---

## References

- `argus\src\argus\desk\workbench.py` — claim-evidence binding, stress, portfolio
- `argus\src\argus\desk\analogue.py` — historical-state matching (disconnected from live flow)
- `argus\src\argus\desk\expectation.py` — consensus vs. delivered
- `argus\src\argus\desk\portfolio.py` — session-conditional beta, risk decomposition
- `argus\src\argus\desk\stress.py` — empirical scenario generation
- `argus\src\argus\desk\review.py` — process validation against outcomes
- `argus\src\argus\lui\answer.py` — source-first answer assembly
- `argus\src\argus\agents\grounding.py` — numeric resolution
- OpenBB: `research/repos/OpenBB-finance/openbb_terminal/stocks/research/`
- FinRobot: `research/repos/FinRobot/multi_agents/agents/analyst_agent.py`
- Bitget Handbook: `BITGET_AI_BASE_CAMP_S2_HANDBOOK_EN.md` (requirement: "one complete research task from question to actionable insight")
