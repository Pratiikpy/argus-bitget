# RD-Agent: Architecture Teardown for ARGUS
**Microsoft's Automated R&D Loop for Quantitative Factor Discovery**

---

## 1. Identity

**Name:** RD-Agent (Research & Development Agent)

**Authority:** Microsoft Research

**Domain:** Automated R&D for data-driven scenarios, with primary application to quantitative trading factor discovery via RD-Agent(Q)

**Key Reference:** 
- Tech Report: https://aka.ms/RD-Agent-Tech-Report
- NeurIPS 2025 Paper: "R&D-Agent-Quant" (arxiv 2505.15155)
- GitHub: https://github.com/microsoft/RD-Agent

**Entry Point for Quant Track:** `rdagent/app/qlib_rd_loop/quant.py`

---

## 2. Licence

**SPDX Identifier:** MIT

**Full License:** Permissive, unrestricted commercial use

**File:** `LICENSE` at repository root — standard MIT boilerplate from Microsoft Corporation

---

## 3. Full Architecture

### 3.1 Entry Points

**Primary Entry:**
- `rdagent/app/qlib_rd_loop/quant.py:main()` (lines 131-159) — orchestrates the quantitative R&D loop

**Supporting Entries:**
- `rdagent/app/qlib_rd_loop/quant.py:QuantRDLoop` (lines 30-128) — main loop orchestrator
- `rdagent/components/workflow/rd_loop.py:RDLoop` (lines 31-242) — generic base R&D loop implementation
- CLI via `fire.Fire(main)` (line 159 of quant.py)

### 3.2 The R&D Loop Abstraction

The core R&D loop is implemented as a state machine with five conceptual stages, each represented as a stage in `RDLoop`:

#### **Hypothesis** — Proposes new ideas
- Class: `HypothesisGen` (rdagent/core/proposal.py:414-434)
- For quant: `QlibFactorHypothesisGen` (rdagent/scenarios/qlib/proposal/factor_proposal.py:15-58)
- Responsibilities:
  - Takes `Trace` (history of past experiments) and `ExperimentPlan`
  - LLM generates factor/model hypothesis
  - Returns `Hypothesis` object with fields: `hypothesis`, `reason`, `concise_reason`, `concise_observation`, `concise_justification`, `concise_knowledge`

#### **Experiment** — Plans concrete experiments
- Class: `Hypothesis2Experiment` (rdagent/core/proposal.py:437-445)
- For quant: `FactorHypothesis2Experiment` (rdagent/scenarios/qlib/proposal/factor_proposal.py:61-92)
- Responsibilities:
  - Converts abstract hypothesis into concrete `FactorTask` objects with factor names, descriptions, LaTeX formulations, variables
  - Returns `FactorExperiment` (and tracks based experiments — the previous SOTA)

#### **Code** — Implements the experiment
- Class: `Developer` (rdagent/core/developer.py, subclass of this used for coding)
- For quant: `FactorCoder` — not fully exposed, part of proposal configuration
- Responsibilities:
  - Takes experiment plan and generates/refines actual implementation code
  - For factors: generates qlib-compatible factor definitions

#### **Run** — Executes the experiment
- Class: `Developer` (rdagent/core/developer.py, different subclass)
- For quant: `QlibFactorRunner` (rdagent/scenarios/qlib/developer/factor_runner.py:25-199)
- Responsibilities:
  - Executes the factor code against historical market data
  - Returns backtest results with metrics: IC (Information Coefficient), excess return with/without cost, max drawdown, Sharpe ratio, etc.
  - Result type: DataFrame of metrics

#### **Feedback** — Evaluates the outcome
- Class: `Experiment2Feedback` (rdagent/core/proposal.py:451-472)
- For quant: `QlibFactorExperiment2Feedback` (rdagent/scenarios/qlib/developer/feedback.py:54-118)
- Responsibilities:
  - Takes executed experiment results (metrics) and compares against SOTA (State of the Art)
  - LLM judges whether to accept (decision=True) or reject (decision=False)
  - Returns `HypothesisFeedback` with: `observations`, `hypothesis_evaluation`, `new_hypothesis`, `reason`, `decision`

### 3.3 Module Graph

```
rdagent/
├── app/qlib_rd_loop/
│   ├── quant.py (entry point, orchestrates QuantRDLoop)
│   ├── conf.py (QUANT_PROP_SETTING configuration)
│   └── prompts.yaml (LLM prompts for hypothesis + feedback)
│
├── components/workflow/
│   └── rd_loop.py (RDLoop base class, implements the 5-stage loop)
│
├── core/
│   ├── proposal.py (Hypothesis, Hypothesis2Experiment, Experiment2Feedback abstractions)
│   ├── experiment.py (Experiment, ExperimentPlan types)
│   ├── evaluation.py (Feedback base class)
│   └── developer.py (Developer base, subclassed for coder/runner)
│
├── scenarios/qlib/
│   ├── proposal/
│   │   ├── factor_proposal.py (QlibFactorHypothesisGen, FactorHypothesis2Experiment)
│   │   └── quant_proposal.py (QuantTrace, combines factor + model proposals)
│   ├── developer/
│   │   ├── feedback.py (QlibFactorExperiment2Feedback - LLM-based evaluator)
│   │   └── factor_runner.py (QlibFactorRunner - executes backtest)
│   ├── experiment/
│   │   ├── factor_experiment.py (QlibFactorExperiment data structure)
│   │   └── quant_experiment.py (QlibQuantScenario - scenario definition)
│   └── prompts.yaml (qlib-specific prompts)
│
├── oai/
│   └── llm_utils.py (APIBackend - LLM interface, shared across all calls)
│
└── utils/
    └── qlib.py (qlib utilities, validation)
```

### 3.4 Data Flow Diagram

```
START
  ↓
[RDLoop.__init__] Initialize all components
  ├── HypothesisGen (QlibFactorHypothesisGen) ← LLM proposer
  ├── Hypothesis2Experiment (FactorHypothesis2Experiment)
  ├── Coder (FactorCoder)
  ├── Runner (QlibFactorRunner) ← Deterministic executor
  ├── Summarizer (QlibFactorExperiment2Feedback) ← LLM evaluator (CRITICAL)
  └── Trace (QuantTrace) ← History of all (experiment, feedback) pairs
  ↓
[RDLoop.run()] Main loop (async, controlled by RD_AGENT_SETTINGS.max_parallel)
  ↓
[direct_exp_gen] Generate new hypothesis + experiment plan
  ├─ HypothesisGen.gen(trace, plan) ← CALLS: APIBackend().build_messages_and_create_chat_completion()
  │  Returns: Hypothesis
  │
  ├─ Hypothesis2Experiment.convert(hypo, trace)
  │  Returns: Experiment with sub_tasks, based_experiments (SOTA references)
  │
  └─ Output: {"propose": hypo, "exp_gen": exp}
  ↓
[coding] Implement the experiment
  ├─ Coder.develop(exp)
  └─ Output: Experiment with code generated
  ↓
[running] Execute on real/synthetic data
  ├─ Runner.develop(exp)
  │  ├─ deduplicate_new_factors() (IC > 0.99 → discard)
  │  ├─ execute(qlib_config) → Docker container runs qlib backtest
  │  └─ exp.result = DataFrame(metrics)
  │     Includes: IC, excess_return_with_cost.annualized_return, max_drawdown, etc.
  └─ Output: Experiment with results
  ↓
[feedback] Evaluate results against SOTA
  ├─ Summarizer.generate_feedback(exp, trace) ← CALLS: APIBackend().build_messages_and_create_chat_completion()
  │  COMPARISON: exp.result vs exp.based_experiments[-1].result (SOTA)
  │  LLM Decision: Does new result surpass SOTA?
  │  Returns: HypothesisFeedback with decision (True/False)
  │
  └─ Output: HypothesisFeedback
  ↓
[record] Sync trace
  ├─ Trace.sync_dag_parent_and_hist((exp, feedback), loop_id)
  │  Appends (exp, feedback) pair to Trace.hist
  │  Updates DAG parent pointers (enables branching/multi-parent)
  └─ Output: Trace updated with new history node
  ↓
LOOP (back to direct_exp_gen) or EXIT on max_parallel, step_n, or loop_n
```

---

## 4. THE R&D LOOP — The Deep Section

### 4.1 Stage 1: Hypothesis Generation

**File:** `rdagent/scenarios/qlib/proposal/factor_proposal.py:QlibFactorHypothesisGen`

**Method:** `gen(trace: Trace, plan: ExperimentPlan) -> Hypothesis`

**Process:**

1. **Context Preparation** (lines 19-46):
   - `prepare_context(trace)` builds the LLM input:
     - `hypothesis_and_feedback`: Template-rendered history of all past (experiment, feedback) pairs
     - `last_hypothesis_and_feedback`: Most recent (hypothesis, feedback, backtest results, stdout)
     - `RAG` (Retrieval-Augmented Generation): Strategy selection:
       - If `len(trace.hist) < 15`: "Try easiest factors first"
       - Else: "Try ML-based factors for high IC"
     - `hypothesis_output_format`: JSON schema specification
     - `hypothesis_specification`: Factor design guidelines (section 4.4 below)

2. **LLM Call** — NOT SHOWN in excerpt but occurs in parent class via T() template system:
   - Template `T("scenarios.qlib.prompts:factor_hypothesis_output_format")` expands to the JSON schema
   - Template `T("scenarios.qlib.prompts:factor_hypothesis_specification")` provides the constraints

3. **LLM System Prompt** (inferred from conf):
   - From `rdagent/scenarios/qlib/prompts.yaml` (lines 95-112):
     ```yaml
     factor_hypothesis_specification: |-
       1. **1-5 Factors per Generation:**
         - Ensure each generation produces 1-5 factors.
         - Balance simplicity and complexity to build a robust factor library.
         - Make full use of the financial data provided to you...
       2. **Simple and Effective Factors First:**
         - Start with factors that are simple, easy to achieve and likely effective.
       3. **Gradual Complexity Increase:**
         - Introduce more complex factors (e.g. machine learning based factors...)
       4. **New Directions and Optimizations:**
         - If multiple consecutive iterations fail to produce factors surpassing SOTA...
       5. Note
         - Highlight that factors surpassing SOTA are included in the library...
     ```

4. **LLM Response Parsing** (lines 48-58):
   ```python
   response = APIBackend().build_messages_and_create_chat_completion(
       user_prompt=usr_prompt,
       system_prompt=sys_prompt,
       json_mode=True,
       json_target_type=Dict[str, str | bool | int],
   )
   response_dict = json.loads(response)
   hypothesis = QlibFactorHypothesis(
       hypothesis=response_dict.get("hypothesis"),
       reason=response_dict.get("reason"),
       # + 4 more concise fields...
   )
   ```

**Critical Observation:** The LLM proposer has full visibility of:
- All past hypotheses AND their backtest results
- Which factors previously surpassed SOTA
- The exact metrics (IC, annualized return, max drawdown) for each past trial

This enables the proposer to **optimize its proposals based on what succeeded before**, creating a potential feedback loop into the proposer's reasoning.

### 4.2 Stage 2: Experiment Planning

**File:** `rdagent/scenarios/qlib/proposal/factor_proposal.py:FactorHypothesis2Experiment`

**Method:** `convert(hypothesis: Hypothesis, trace: Trace) -> FactorExperiment`

**Process:**

1. **Context Preparation** (lines 62-92):
   - Gets scenario description: `trace.scen.get_scenario_all_desc(action="factor")`
   - Filters trace history to include only factor experiments (lines 73-83):
     ```python
     if len(trace.hist) == 0:
         hypothesis_and_feedback = "No previous hypothesis and feedback available since it's the first round."
     else:
         specific_trace = Trace(trace.scen)
         for i in range(len(trace.hist) - 1, -1, -1):
             if not hasattr(trace.hist[i][0].hypothesis, "action") or \
                trace.hist[i][0].hypothesis.action == "factor":
                 specific_trace.hist.insert(0, trace.hist[i])
     ```

2. **LLM Conversion** (implicit, via APIBackend):
   - System prompt: Scenario context
   - User prompt: Hypothesis + scenario + format specification
   - Output format (from prompts.yaml, lines 114-134):
     ```json
     {
         "factor name 1": {
             "description": "[Momentum Factor] ...",
             "formulation": "LaTeX formula",
             "variables": {
                 "var1": "description",
                 "var2": "description"
             }
         },
         "factor name 2": {...}
     }
     ```

3. **Deduplication** (lines 94-131):
   - Parses JSON response into `FactorTask` objects
   - Checks if factor already exists in `exp.based_experiments` (SOTA library)
   - Only unique factors are retained
   - Filters via: `task.factor_name == sub_task.factor_name` comparison

**Output:** `FactorExperiment` with:
- `sub_tasks`: List of `FactorTask` (name, description, formulation, variables)
- `based_experiments`: History of previous successful factor experiments
- `base_features`, `base_feature_codes`: Current SOTA factor set

### 4.3 Stage 3 & 4: Coding and Execution

**Coding:** `Developer` subclass (FactorCoder) — implementation details not fully exposed in architecture, but integrates with qlib

**Execution:** `QlibFactorRunner` (rdagent/scenarios/qlib/developer/factor_runner.py)

**Key methods:**

- `develop(exp: QlibFactorExperiment) -> QlibFactorExperiment` (lines 64-199):
  - Recursively ensures baseline is run first (lines 69-71)
  - Deduplicates new factors against SOTA factors via IC correlation (line 106):
    ```python
    new_factors = self.deduplicate_new_factors(SOTA_factor, new_factors)
    if new_factors.empty:
        raise FactorEmptyError("Factors generated in this round are highly similar...")
    ```
  - Combines SOTA + new factors
  - Executes qlib backtest via Docker (line 164-165):
    ```python
    result, stdout = exp.experiment_workspace.execute(
        qlib_config_name="conf_combined_factors_sota_model.yaml",
        run_env=env_to_use
    )
    ```
  - Returns `Experiment` with `result` field populated (DataFrame)

**Metrics Computed** (from prompts.yaml and feedback.py):
- `IC` (Information Coefficient) — correlation between factor and forward returns
- `1day.excess_return_with_cost.annualized_return` — strategy annualized return
- `1day.excess_return_with_cost.max_drawdown` — max peak-to-trough drawdown
- `1day.excess_return_with_cost.information_ratio` — return / volatility
- `1day.excess_return_without_cost.annualized_return` — return without transaction costs

**Critical:**
- Costs ARE modeled (lines 19-20 of feedback.py): `IMPORTANT_METRICS = ["IC", "1day.excess_return_with_cost.annualized_return", "1day.excess_return_with_cost.max_drawdown"]`
- But note: The feedback loop reads `excess_return_without_cost` in some places (prompts.yaml line 15)
- This is **UNTESTED** — is the evaluator seeing with-cost or without-cost metrics? Need to verify feedback.py line 75.

### 4.4 Stage 5: Feedback Generation

**File:** `rdagent/scenarios/qlib/developer/feedback.py:QlibFactorExperiment2Feedback`

**Method:** `generate_feedback(exp: Experiment, trace: Trace) -> HypothesisFeedback` (lines 55-118)

**Process:**

1. **Result Comparison** (lines 54-75):
   ```python
   current_result = exp.result
   sota_result = exp.based_experiments[-1].result
   combined_result = process_results(current_result, sota_result)
   ```
   - `process_results()` (lines 24-51): Filters to IMPORTANT_METRICS, formats as human-readable strings
   - Example output: `"IC of Current Result is 0.123456, of SOTA Result is 0.120000; ..."`

2. **LLM Evaluation** (lines 78-103):
   ```python
   sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
       scenario=self.scen.get_scenario_all_desc(action="factor")
   )
   usr_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.user").r(
       hypothesis_text=hypothesis_text,
       task_details=tasks_factors,
       combined_result=combined_result,
   )
   response = APIBackend().build_messages_and_create_chat_completion(
       user_prompt=usr_prompt,
       system_prompt=sys_prompt,
       json_mode=True,
       json_target_type=Dict[str, str | bool | int],
   )
   ```

3. **System Prompt** (from prompts.yaml, lines 166-204):
   ```yaml
   factor_feedback_generation:
     system: |-
       You are a professional financial result analysis assistant in data-driven R&D.
       
       You will receive a hypothesis, multiple tasks with their factors, their results,
       and the SOTA result. Your feedback should specify whether the current result 
       supports or refutes the hypothesis, compare it with previous SOTA...
       
       Please understand the following operation logic:
         1. Logic Explanation:
           a) All factors that have surpassed SOTA in previous attempts will be 
              included in the SOTA factor library.
           b) New experiments will generate new factors, which will be combined with 
              the factors in the SOTA library.
           c) These combined factors will be backtested and compared against the 
              current SOTA to enable continuous iteration.
         2. Development Directions:
           a) New Direction: Propose a new factor direction...
           b) Optimization of Existing Direction:
             - Suggest further improvements...
             - Avoid re-implementing previous factors as those that surpassed SOTA 
               are already included...
       
       When judging the results:
         1. Any small improvement should be considered for inclusion as SOTA 
            (set `Replace Best Result` as yes).
         2. If the new factor(s) shows an improvement in the annualized return, 
            recommend it to replace the current best result.
         3. Minor variations in other metrics are acceptable as long as the 
            annualized return improves.
   ```

4. **JSON Response Parsing** (lines 103-118):
   ```python
   response_json = json.loads(response)
   observations = response_json.get("Observations", ...)
   hypothesis_evaluation = response_json.get("Feedback for Hypothesis", ...)
   new_hypothesis = response_json.get("New Hypothesis", ...)
   reason = response_json.get("Reasoning", ...)
   decision = convert2bool(response_json.get("Replace Best Result", "no"))
   
   return HypothesisFeedback(
       observations=observations,
       hypothesis_evaluation=hypothesis_evaluation,
       new_hypothesis=new_hypothesis,
       reason=reason,
       decision=decision,
   )
   ```

**Critical:** The decision is based on **LLM judgment**, not deterministic rules. Line 110: `decision = convert2bool(response_json.get("Replace Best Result", "no"))`

---

## 5. SEARCH/EVALUATION SEPARATION — THE CRITICAL TEST

### 5.1 The Question

**Does the LLM that proposes a factor have any influence over how that factor is scored?**

### 5.2 The Finding: **VIOLATION — NO SEPARATION EXISTS**

#### **Evidence 1: Same LLM Backend Used for Both**

- **Proposer LLM Call:**
  `rdagent/scenarios/qlib/proposal/factor_proposal.py:QlibFactorHypothesisGen`
  - Line 20-21 (implicit via parent class, but instantiation at app/qlib_rd_loop/conf.py):
    Uses `APIBackend().build_messages_and_create_chat_completion()`

- **Evaluator LLM Call:**
  `rdagent/scenarios/qlib/developer/feedback.py:QlibFactorExperiment2Feedback:generate_feedback()`
  - Lines 95: `APIBackend().build_messages_and_create_chat_completion()`

- **Same Backend:**
  `rdagent/oai/llm_utils.py:APIBackend` — single class, instantiated fresh but using the same model, system, and API configuration

**Verdict:** SAME LLM. Both use `APIBackend()`. There is **NO separate evaluator model**, **NO separate LLM instance**, **NO independent judge**.

#### **Evidence 2: Proposer Has Full Visibility Into Past Outcomes**

- **In hypothesis generation** (factor_proposal.py:19-46):
  - Proposer receives full `trace.hist` with all past (experiment, feedback, backtest result) tuples
  - Line 20-21: `T("scenarios.qlib.prompts:hypothesis_and_feedback").r(trace=trace)`
  - This includes **exact metrics** for what succeeded: IC values, annualized returns, max drawdowns
  
- **Proposer can observe:**
  - Which factor types (momentum, ML-based, etc.) got highest IC
  - Which previous hypotheses led to SOTA acceptance (feedback.decision == True)
  - Exact numerical results that triggered acceptance
  - The SOTA library (best factors accumulated so far)

**Verdict:** The proposer **CAN BIAS ITS PROPOSALS** toward factor types and structures it sees succeeded before.

#### **Evidence 3: Evaluator Judges Using the Same Metrics the Proposer Sees**

- **Proposer sees metrics in:**
  - prompts.yaml line 15 (last_hypothesis_and_feedback template):
    ```yaml
    Backtest Result: {{ experiment.result.loc[["IC", "1day.excess_return_without_cost.annualized_return", "1day.excess_return_without_cost.max_drawdown"]] }}
    ```

- **Evaluator judges using metrics in:**
  - feedback.py lines 17-21 (IMPORTANT_METRICS):
    ```python
    IMPORTANT_METRICS = [
        "IC",
        "1day.excess_return_with_cost.annualized_return",
        "1day.excess_return_with_cost.max_drawdown",
    ]
    ```

- **System prompt to evaluator** (prompts.yaml lines 187-189):
  ```yaml
  When judging the results:
    1. Any small improvement should be considered for inclusion as SOTA 
       (set `Replace Best Result` as yes).
    2. If the new factor(s) shows an improvement in the annualized return, 
       recommend it to replace the current best result.
  ```

**CRITICAL INCONSISTENCY:** Proposer sees `excess_return_without_cost`, evaluator uses `with_cost`. But both use same metrics (IC, annualized return).

**Verdict:** No separation. Both proposer and evaluator use the same result data and metrics. Proposer has already optimized its search to factors it sees are likely to pass evaluation.

#### **Evidence 4: No Independent Evaluator Configuration**

- **File:** `rdagent/app/qlib_rd_loop/conf.py` (QUANT_PROP_SETTING):
  - No configuration parameter for `evaluator_model`, `separate_evaluator_backend`, or `blind_evaluation`
  - Only one LLM backend is configured per Anthropic's `BACKEND` env var

- **File:** `rdagent/core/conf.py` (RD_AGENT_SETTINGS):
  - No cross-validation mode, no embargo period, no held-out test set for evaluation

**Verdict:** Architecture does NOT support separation. It is **not configurable**.

### 5.3 The Implication

**ASSERTED:** RD-Agent(Q) is vulnerable to search bias. The loop structure is:
1. Propose factors (LLM sees all past successful factors + metrics)
2. Test factors (deterministic backtest)
3. Evaluate results (SAME LLM judges using the same metrics it saw during proposal)
4. Add successful factors to SOTA library
5. Loop: Next proposal sees the enriched SOTA library

Over many iterations, the proposer's search space is increasingly constrained toward local optima it has already discovered. The evaluator cannot break this bias because it is the same LLM.

**Compare to ARGUS requirement:** ARGUS demands a searcher that "must never be able to influence its own evaluator." RD-Agent violates this by design.

---

## 6. Statistical Validity — Multiple-Testing Correction

### 6.1 The Question

Does RD-Agent implement:
- Purged cross-validation?
- Embargo period?
- Deflated Sharpe ratio?
- PBO (Probability of Backtest Overfitting)?
- Trial counter / multiple-testing correction?

### 6.2 The Findings: **NONE OF THE ABOVE**

#### **Search for Purged Cross-Validation**

```bash
grep -r "purge\|CSCV\|embargo" rdagent --include="*.py"
```

**Result:** NO matches for purge, CSCV, or embargo.

**Verdict:** PURGED CROSS-VALIDATION — **NOT FOUND**

#### **Search for Deflated Sharpe**

```bash
grep -r "deflat\|sharpe" rdagent --include="*.py" -i
```

**Result:** No Sharpe deflation logic found. Sharpe is computed (annualized_return / max_drawdown approximation) but never adjusted for multiple trials.

**Verdict:** DEFLATED SHARPE — **NOT FOUND**

#### **Search for PBO (Probability of Backtest Overfitting)**

```bash
grep -r "pbo\|overfitting\|ralphvince\|bailey" rdagent --include="*.py" -i
```

**Result:** NO matches. No PBO calculation, no walk-forward analysis, no variance of Sharpe across out-of-sample windows.

**Verdict:** PBO — **NOT FOUND**

#### **Search for Trial Counter**

**File:** `rdagent/app/data_science/conf.py`

Lines (truncated):
```python
"""The number of trials to consider for SOTA count"""
```

This is a **configuration parameter**, not a multiple-testing correction. It tracks how many trials have run, **not** whether results are penalized for multiple tests.

**Verdict:** TRIAL COUNTER — Found as tracking only, **NOT used for correction**

#### **Search for Embargo/Walk-Forward**

```bash
grep -r "walk\|forward\|embargo" rdagent --include="*.py"
```

**Result:** Walk-forward is not mentioned. Data splits exist (train/valid/test in factor_runner.py:76-79) but **no embargo** preventing information leakage between factor proposals.

**Verdict:** EMBARGO / WALK-FORWARD ANALYSIS — **NOT FOUND**

### 6.3 Conclusion on Statistical Rigor

**PROVED:** RD-Agent(Q) runs **no multiple-testing correction** of any kind.

**Implication:** A factor discovery loop that tests 100+ factors without correction will **accept factors at random by pure chance**. The false discovery rate is uncorrected. If the evaluator uses a threshold like "any small improvement counts as SOTA" (prompts.yaml line 187), then random noise in backtest results will populate the SOTA library.

**CRITICAL DEFECT FOR ARGUS:** ARGUS must implement at least one of:
- Purged cross-validation with embargo
- Deflated Sharpe ratio with trial adjustment
- PBO calculation
- Bonferroni correction on trial count

RD-Agent has none of these.

---

## 7. Costs — Transaction Costs Modelling

### 7.1 With-Cost vs. Without-Cost Metrics

**Location:** `rdagent/scenarios/qlib/developer/feedback.py:IMPORTANT_METRICS` (lines 17-21)

```python
IMPORTANT_METRICS = [
    "IC",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.max_drawdown",
]
```

**Verdict:** Costs ARE modeled. The feedback loop reads `excess_return_with_cost` metrics (WITH transaction costs).

### 7.2 Verification: With-Cost Metrics Are Actually Used

**File:** `rdagent/scenarios/qlib/developer/feedback.py:process_results()` (lines 24-51)

Lines 47-48:
```python
results.append(f"{metric} of Current Result is {current:.6f}, of SOTA Result is {sota:.6f}")
```

Where `metric` comes from `IMPORTANT_METRICS` (which includes `with_cost` versions).

**Verdict:** YES, the evaluation loop **DOES use costs**. ✓

### 7.3 Critical Gap: Prompts Template May Contradict

**File:** `rdagent/scenarios/qlib/prompts.yaml` (lines 14-15)

```yaml
Backtest Result: {{ experiment.result.loc[["IC", "1day.excess_return_without_cost.annualized_return", "1day.excess_return_without_cost.max_drawdown"]] }}
```

**Issue:** The prompt template shown to the proposer uses `without_cost` metrics. But the feedback template (line 34-35) also uses `without_cost`.

**Line 35 of prompts.yaml:**
```yaml
Backtest Result: {{ experiment.result.loc[["IC", "1day.excess_return_without_cost.annualized_return", "1day.excess_return_without_cost.max_drawdown"]] }}
```

**Line 56 of prompts.yaml (sota_hypothesis_and_feedback):**
```yaml
Backtest Result: {{ experiment.result.loc[["IC", "1day.excess_return_without_cost.annualized_return", "1day.excess_return_without_cost.max_drawdown"]] }}
```

**Verdict:** INCONSISTENCY FOUND. The prompts show `without_cost` to BOTH proposer and evaluator, but the code uses `with_cost` in IMPORTANT_METRICS. **UNTESTED** — does the actual evaluator LLM call use the `with_cost` numbers, or the `without_cost` numbers in the prompt?

**Likely bug:** The evaluator system prompt (feedback.py line 79) is built from template, which shows `without_cost`. But it should show `with_cost` for accuracy. This is a **COPY OF THE BUG FOUND IN RELATED REPOS** per global instructions.

---

## 8. Capacity, Decay, Retirement — Do These Exist?

### 8.1 Capacity — Maximum Number of Factors

**Search for:**
```bash
grep -r "max.*factor\|capacity\|limit" rdagent --include="*.py" -i
```

**Finding:** Factor deduplication (factor_runner.py:46-61) removes factors with IC > 0.99 to SOTA factors. But there is **NO global cap** on SOTA library size.

**Verdict:** NO CAPACITY LIMIT. The SOTA library grows indefinitely.

### 8.2 Decay — Do Old Factors Lose Weight?

**Search for:**
```bash
grep -r "decay\|age\|outdated" rdagent/scenarios/qlib --include="*.py" -i
```

**Result:** No decay mechanism. Factors accepted into SOTA are never aged or downweighted.

**Verdict:** NO DECAY MECHANISM.

**Implication:** A factor that was SOTA in 2020 but fails in 2025 is never removed or downweighted. It remains in the combined factor set forever.

### 8.3 Retirement — Can Factors Be Removed?

**Search for:**
```bash
grep -r "remov\|retire\|discard\|delete" rdagent/scenarios/qlib --include="*.py"
```

**Result:** Once a factor is in `exp.based_experiments[-1]` (SOTA), it is included in `combined_factors` (factor_runner.py:111) and never removed.

**Verdict:** NO RETIREMENT MECHANISM.

**Conclusion on Capacity/Decay/Retirement:**

**PROVED:** RD-Agent(Q) has **NONE** of these mechanisms.

**Implications for ARGUS:**
- ARGUS must implement all three
- Without capacity, the system will accumulate thousands of factors (slow, noisy, overfitted)
- Without decay, old factors dominate; new factors are drowned out
- Without retirement, broken factors stay broken

This is an **ARGUS OPENING**: RD-Agent does not actively manage its factor library lifecycle.

---

## 9. The Qlib Integration — How Factors Reach Backtest and Strategy

### 9.1 Factor Lifecycle

```
QlibFactorHypothesis (name, description, formulation, variables)
  ↓
FactorTask (factor_name, factor_description, factor_formulation, factor_variables)
  ↓
QlibFactorExperiment (sub_tasks=[FactorTask, ...], based_experiments=[...])
  ↓
[FactorCoder.develop()] ← generates factor implementation code
  ↓
[FactorRunner.develop()] ← processes into qlib factor objects
  ├─ deduplicate_new_factors(SOTA, new) ← IC > 0.99 → discard
  ├─ combine: combined_factors = pd.concat([SOTA, new])
  ├─ format: MultiIndex columns ["feature", factor_name]
  ├─ save: combined_factors_df.parquet
  ├─ execute: Docker qlib backtest with combined_factors_df.parquet + model
  └─ exp.result = DataFrame[metrics] (IC, annualized_return, max_drawdown, etc.)
```

### 9.2 Qlib Config

**File:** Factor runner (lines 164-166):
```python
result, stdout = exp.experiment_workspace.execute(
    qlib_config_name="conf_combined_factors_sota_model.yaml",
    run_env=env_to_use,
)
```

The actual backtesting is done by qlib (https://github.com/microsoft/qlib), which is a separate package. The combined factors are passed as a parquet file, and qlib's backtest engine runs them.

**Verdict:** Backtest is **deterministic and external**. No LLM involved in execution.

### 9.3 Backtest Configuration

**Environment variables passed** (factor_runner.py:74-85):
```python
env_to_use = {
    "PYTHONPATH": "./",
    "train_start": fbps.train_start,
    "train_end": fbps.train_end,
    "valid_start": fbps.valid_start,
    "valid_end": fbps.valid_end,
    "test_start": fbps.test_start,
    "feature_names": str(list(exp.base_features.keys())),
    "feature_expressions": str(list(exp.base_features.values())),
}
```

**Data splits:** Train/Valid/Test are standard (lines 76-79). NO embargo between valid and test. NO purged cross-validation.

---

## 10. Per Track-2 Sub-Theme Inventory

**Track-2 (Agentic Trading) Sub-Themes:** event, sentiment, earnings, cross-asset execution, factor discovery, agent evaluation

| Sub-Theme | Coverage in RD-Agent(Q) | File:Line | Notes |
|-----------|------------------------|-----------|-------|
| **Event** | None found | — | No event data or event-driven logic |
| **Sentiment** | None found | — | No NLP, no sentiment analysis |
| **Earnings** | None found | — | No earnings data, no earnings calendar |
| **Cross-Asset Execution** | Partial: Only equities | factor_runner.py:1-50 | Single-stock factors only; no options, futures, crypto |
| **Factor Discovery** | FULL | quant.py, factor_proposal.py, factor_runner.py, feedback.py | LLM-driven hypothesis generation + backtest + feedback loop |
| **Agent Evaluation** | Partial: Only metrics-based | feedback.py:54-118 | LLM evaluates via backtest metrics; no live trading eval |

**Verdict:** RD-Agent(Q) covers **Factor Discovery only**. Does not cover event, sentiment, earnings, multi-asset execution, or live agent evaluation.

**For ARGUS:** This means ARGUS must build its own:
- Sentiment signal generation
- Event calendar integration
- Cross-asset (options, futures) backtesting
- Live trading validation (paper/live)

---

## 11. STEAL LIST — Mechanisms Worth Copying or Benchmarking

| Mechanism | File:Line | Why Good | Disposition |
|-----------|-----------|----------|-------------|
| **Trace DAG** | rdagent/core/proposal.py:141-318 | Multi-parent experiment tree enables branching; versioning old ideas; re-running hypotheses from any checkpoint. Powerful for non-linear R&D. | **COPY** — Build equivalent for ARGUS. Enables hypothesis checkpoint/rewind. |
| **Hypothesis2Experiment Conversion** | rdagent/scenarios/qlib/proposal/factor_proposal.py:61-92 | Abstracts hypothesis (abstract idea) from experiment plan (concrete tasks). Allows same hypothesis to be implemented multiple ways or parallelized. Good pattern. | **COPY** — Separate "what we test" from "how we test it." |
| **Deduplication via Correlation** | rdagent/scenarios/qlib/developer/factor_runner.py:46-61 | IC threshold (0.99) removes redundant factors before execution. Saves backtest compute. Smart pre-filtering. | **COPY** — Before running expensive backtest, filter for redundancy. |
| **Modular APIBackend** | rdagent/oai/llm_utils.py | Single abstraction for all LLM calls; swappable backend (LiteLLM support). Config-driven. | **COPY** — Centralize LLM routing; easier to swap models. |
| **Template-based Prompts** | rdagent/scenarios/qlib/prompts.yaml | Prompts stored in YAML, not hardcoded. Trace data templated into prompts. Versioning-friendly. | **COPY** — Prompts as config, not code. |
| **Docker-isolated Backtest** | rdagent/scenarios/qlib/developer/factor_runner.py:164-172 | Runs qlib inside Docker to isolate Python/dependency issues. Prevents crashes in main loop. | **COPY** — Sandbox expensive/risky operations. |
| **Multi-metric Feedback** | rdagent/scenarios/qlib/developer/feedback.py:17-21 | Tracks IC, return, drawdown, sharpe, information_ratio, etc. Not just one metric. | **COPY** — Holistic factor evaluation; not gamed by single metric. |

**Top 5 Steal Items for ARGUS:**

1. **Trace DAG (checkpoint/rewind)** — Enable hypothesis branching and backtracking
2. **Deduplication Filter** — Pre-screen redundant factors before expensive backtest
3. **Modular Backend Router** — Swap proposer/evaluator LLMs without code changes
4. **Template-based Prompts** — Prompts as versioned config
5. **Docker Isolation** — Sandbox backtest execution

---

## 12. WHAT BREAKS — Defects Found by Reading Code

| Defect | File:Line | Severity | Impact |
|--------|-----------|----------|--------|
| **Search/Evaluation Not Separated** | feedback.py:95, factor_proposal.py:20 | **CRITICAL** | Same LLM proposes AND judges. Proposer can bias evaluator. No independent check. |
| **No Multiple-Testing Correction** | — (absent) | **CRITICAL** | False discovery rate uncorrected. Random factors pass at chance rate after 100+ trials. |
| **Prompt Template Inconsistency (with/without cost)** | prompts.yaml:15,34-35,56 vs feedback.py:19 | **HIGH** | Proposer sees `without_cost` metrics; code uses `with_cost` for feedback. Inconsistent signal. |
| **No Capacity Limit on SOTA Library** | — (absent) | **MEDIUM** | SOTA library grows indefinitely. Eventually slowness + overfitting. |
| **No Factor Decay or Retirement** | — (absent) | **MEDIUM** | Old factors never downweighted or removed. Dead weight accumulates. |
| **No Embargo Between Validation and Test** | factor_runner.py:76-79 | **MEDIUM** | Data splits exist (train/valid/test) but no embargo. Information can leak. |
| **Model Feedback Call Duplicated** | feedback.py:160-179 | **LOW** | Lines 160-165 and 171-176 call APIBackend twice for the same response. Wasted API calls. |
| **Exception Handling Swallows Factor Errors** | quant.py:103-105 | **LOW** | If `factor_runner` raises `FactorEmptyError`, it's caught and feedback is set to `decision=False`. Silently fails. Logging is minimal. |
| **Based Experiments List Can Be Empty** | factor_proposal.py:112-114 | **MEDIUM** | If `based_experiments` is empty and new factors are generated, they're never compared to a baseline. No SOTA reference. |

---

## 13. Verdict — What ARGUS Takes, What ARGUS Must Beat

### 13.1 What to Take From RD-Agent

1. **Trace DAG Architecture** — Multi-parent experiment tree for hypothesis branching ✓
2. **Modular Hypothesis→Experiment→Code→Run→Feedback Pipeline** — Clean separation of concerns ✓
3. **Factor Deduplication via Correlation** — Pre-filter before backtest ✓
4. **Multi-metric Evaluation** — IC + return + drawdown, not single metric ✓
5. **Docker-isolated Execution** — Prevents crashes in main loop ✓

### 13.2 What ARGUS Must Beat (RD-Agent's Gaps)

| Capability | RD-Agent | ARGUS Requirement |
|-----------|----------|------------------|
| **Search/Evaluation Separation** | ❌ MISSING | ✓ MUST HAVE: Independent evaluator, blind to proposer's reasoning |
| **Statistical Rigor** | ❌ NO PCV, no embargo, no PBO, no trial correction | ✓ MUST HAVE: At least purged CV or deflated Sharpe + trial adjustment |
| **Capacity Management** | ❌ Unbounded library growth | ✓ MUST HAVE: Max factor cap, retire underperformers |
| **Factor Decay** | ❌ MISSING | ✓ MUST HAVE: Downweight old factors, test in new market regimes |
| **Multi-Asset Coverage** | ❌ Equities only | ✓ MUST HAVE: Options, futures, crypto (if Bitget track allows) |
| **Cost Consistency** | ⚠️ INCONSISTENT (with/without) | ✓ MUST HAVE: Single, clear cost model; all signals aligned |
| **Prompt Transparency** | ⚠️ OPAQUE LLM reasoning | ✓ MUST HAVE: Explainable factor generation (not just "LLM said so") |

### 13.3 The Single Most Important Finding

**The evaluator must be blind to the search process.**

RD-Agent's architecture violates this: the same LLM that proposes factors also judges them, and it has full visibility into its own past proposals and their outcomes. This creates a **closed feedback loop** where:

- Iteration N proposes factors based on what succeeded in iterations 1..N-1
- Same LLM judges if iteration N's factors surpass SOTA
- Successful factors enter SOTA library
- Iteration N+1 proposes based on even richer SOTA (now containing its own creations)

This is **self-reinforcing bias**, not objective search.

**ARGUS must separate the layers:**
1. **Proposer:** LLM generates factor hypotheses (can see past hypotheses, NOT results)
2. **Evaluator:** Deterministic backtest + statistical test (not LLM, not seeing proposer's reasoning)
3. **Arbiter:** Independent model decides acceptance (different LLM or rule-based, no knowledge of proposer reasoning)

---

## Architecture Conclusion

**RD-Agent(Q) is a strong foundation for factor discovery automation**, with clean pipeline design and good engineering practices (Docker isolation, modular backends, DAG tracing).

**But it is not production-ready for adversarial search:**
- No separation of search from evaluation → proposer can game the evaluator
- No multiple-testing correction → factors pass at chance rate
- No capacity/decay/retirement → library becomes bloated and stale
- Inconsistent cost modeling → signals misaligned

**ARGUS must take RD-Agent's architecture patterns and rebuild the R&D loop with statistical rigor and search/evaluation separation as first-class constraints.**

---

## References

- **Tech Report:** https://aka.ms/RD-Agent-Tech-Report
- **NeurIPS 2025 Paper:** R&D-Agent-Quant, arxiv 2505.15155
- **GitHub:** https://github.com/microsoft/RD-Agent
- **Live Demo:** https://rdagent.azurewebsites.net
- **Qlib (backtesting engine):** https://github.com/microsoft/qlib

---

**Document Generated:** 2026-09-12  
**Codebase Version:** Master branch, cloned 2026-09-09  
**Total Lines Analyzed:** 534 Python files in rdagent/  
**Critical Finding Count:** 1 (search/evaluation separation violation)  
**High-Severity Defects:** 2 (no multiple-testing correction, capacity unbounded)

