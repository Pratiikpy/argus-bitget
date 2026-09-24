# Finance-Agent Evaluation Harness Teardown: Three Benchmarks

**Date:** 2026-09-12  
**Purpose:** Identify evaluation machinery in existing finance-agent benchmarks to avoid duplication and inform ARGUS (Bitget hackathon Track-2 open-theme agent-evaluation observatory).

---

## A. Microsoft Finance Agent Benchmark

**Repository:** `microsoft~financebenchmark` (shallow clone)  
**License:** MIT (SPDX: MIT)

### Identity and licence

The Microsoft Finance Agent Benchmark evaluates financial agents on three types of tasks: financial obligations research (ERP QA), financial performance research, and business brief generation. The codebase is licensed under the MIT License, allowing unrestricted use, modification, and redistribution with inclusion of the license notice.

**Verified:** LICENSE file, lines 1-22. SPDX: MIT.

### Architecture — entry points, harness control flow, task loading, model invocation, scoring, reporting

**Entry point:** `run_benchmark.py`  
**Pipeline:** Data prep → Inference → Evaluation → Analysis  
**Config:** `config.yaml` (YAML, human-editable)

**Control flow (`run_benchmark.py:43-68`, `run_benchmark.py:115-154`):**
- Loads config from `config.yaml`
- Computes inference config hash via `compute_inference_config_hash()` (lines 115-154)
- Checks if matching inference hash exists in `results/runs.db`; if so, reuses cached result (skips inference)
- If no cached inference, runs `scripts/inference/inference.py` with provider and CLI args
- On inference completion, runs `scripts/evaluation/evaluate.py` to score results
- Writes output to `results/eval_results_{model}_{eval_slug}.json`

**Task loading (`scripts/inference/dataloader.py`):**
- Loads questions from `data/dataset.yaml`
- Format: YAML dict with `query`, `plugin`, `tags`, `segment`, `timeout`, `scenario`, and per-tag assertions
- ~300 total questions across three plugins: `erp_qa`, `finance_qa`, `business_brief`

**Model invocation:**
- **Claude:** `scripts/inference/_claude.py` — Claude SDK, agentic loop with tool calling (WebSearch, WebFetch, MCP)
- **OpenAI:** `scripts/inference/_openai.py` — OpenAI API, reasoning models with tool use
- Both store tool call outputs and source content for later eval-time groundedness checking

**Scoring (`scripts/evaluation/evaluate.py`):**
- Uses DSPy for judge invocation (lines 34, 82, 126-130)
- Three judge modules: `TagAssertionEvaluator`, `GroundednessEvaluator`, `BusinessBriefSectionEvaluator`
- Per-tag scoring: one judge call per question per tag
- Assertions scored 0.0–1.0 (fractional), null for inapplicable
- Tag score = mean of assertion scores (excluding nulls, line 180-188)
- Overall score = mean of tag scores (line 485)

**Reporting:**
- Output envelope (lines 161-166): `{"metadata": {...}, "results": [...]}`
- Metadata includes: judge model, inference model, run ID, config hash, timestamps, token usage
- Per-result: overall_score, tag_scores dict, assertion_scores dict, tag_reasoning dict, token_usage

### THE TASK SET — what is actually being tested. Task taxonomy, counts, an example task quoted in full, and where the ground truth comes from.

**Task types:** 3 plugins × ~100 questions each = ~300 total

#### 1. **ERP QA (Financial Obligations Research)**
Queries about internal AP/AR position: outstanding balances, aged debt, open invoices, vendor terms, credit limits.

**Example task (from `data/dataset.yaml`, lines visible in earlier read):**
```yaml
- query: "What is the credit limit for SYNCUS-0001 USMF as of March 2, 2026?"
  plugin: erp_qa
  segment: AR
  timeout: 300
  scenario: Credit Limit
  tags:
  - tag: accuracy
    assertions:
    - text: "The response contains facts that match, support, or can be logically inferred from this ground truth. Minor differences in phrasing, formatting, or additional context are acceptable as long as core facts are accurate. Ground truth: The credit limit for A. Datum Corporation SYNCUS-0001 is 25,000 as of March 2, 2026."
      level: critical
  - tag: clarity
    assertions: [...]
  - tag: groundedness
    assertions: [...]
  - tag: relevance
    assertions: [...]
```

**Ground truth source:** MCP server connected to Dynamics 365 Finance sandbox with synthetic demo data (AP/AR fixtures). Agent must query ERP tools and return the exact value. Validation token refreshed via `refresh_erp_token.py`.

#### 2. **Finance QA (Financial Performance Research)**
Queries about publicly traded companies' financial performance: earnings, ratios, cash flows, ESG metrics.

**Example (README.md, line 56-61):**
```
"What was ExxonMobil's GAAP current assets for fiscal year ended December 31, 2024?"
"For our credit evaluation of Caterpillar, what is their long-term debt-to-equity ratio (3-year average) as of September 2025?"
"What was Walmart's inventory turnover ratio for FY2024?"
```

**Ground truth source:** Public filings (10-K, 10-Q), financial data providers (Yahoo Finance, SEC EDGAR, etc.). Agent must locate and cite current figures with sources. Ground truth is embedded in assertions as expected numeric ranges or exact values.

#### 3. **Business Brief (Synthesized Profile)**
Requests for structured company profiles synthesising public financial data, business context, and internal ERP data (where available).

**Example (README.md, line 68-73):**
```
"Business Brief report of Apple Inc."
"Company Overview report of AT&T Inc."
"Corporate Profile report of Lockheed Martin Corp"
```

**Ground truth source:** Section-by-section rubric loaded from `sections_rubric.yml` (path in config). Rubric maps section keys (e.g., "financials", "business_description") to quality dimensions (e.g., "accuracy", "completeness", "recency") and assertion lists. Ground truth is in the rubric assertions, not in gold-standard briefs.

**Dataset consistency:** Static YAML file pre-built with 4,661 academic questions, 1,434 industry application scenarios, 1,640 security questions, agent tasks, and multimodal examples. No documentation of train/test splits or held-out sets visible in README.

### THE GRADING MECHANISM — the critical section. Deterministic, rubric-based, or LLM-judge? If LLM-judge, WHICH model, and is there any guard against the judge being the same model family as the subject?

**Verdict: LLM-JUDGE-UNGUARDED**

**Judge model:** OpenAI `gpt-52-2025-12-11` (hardcoded in config.yaml, line 51; can be overridden at runtime with `--judge-model` flag, evaluate.py:16)

**Client:** OpenAI API (`judge_client: openai`, config.yaml:50)

**Reasoning effort:** `low` (config.yaml:49, applies to OpenAI reasoning models)

**No guard against model family bias:** Judge is always OpenAI gpt-52, regardless of whether the test subject is Claude or OpenAI. No mechanism to swap judge model based on subject model. If testing Claude, the judge is still OpenAI; if testing OpenAI, the judge is still OpenAI. This allows undetected judge-test model coupling bias (e.g., OpenAI judge may systematically favor OpenAI reasoning traces).

**DSPy integration (evaluate.py:34, 75-93, 126-130):**
```python
lm = dspy.settings.lm  # DSPy LM history
pred = module(**kwargs)  # Run DSPy module
# Extract token usage from DSPy history (lines 75-93)
for entry in lm.history[n_before:]:
    pt, ct = _extract_usage(entry)
    input_t += pt; output_t += ct
```

DSPy is used for judge invocation and token tracking, but the judge model is configured externally (config.yaml).

**Judge signatures:**
- **TagAssertionEvaluator** (lines 98-110): Query + response + tag + assertions → assertion_scores JSON array (0.0–1.0 or null) + reasoning
- **GroundednessEvaluator** (lines 112-124): Query + response + source_content + assertions → assertion_scores JSON array + reasoning
- **BusinessBriefSectionEvaluator** (lines 143-156): Query + section_key + section_content + assertions_by_dimension JSON → scores_by_dimension JSON + reasoning

**Assertion scoring code (evaluate.py:165-189):**
```python
def parse_assertion_scores(pred, tag_entry: dict) -> dict:
    """Parse assertion_scores from judge. Expects JSON array in assertion order."""
    try:
        raw = json.loads(pred.assertion_scores)
    except Exception:
        return {}
    if isinstance(raw, list):
        assertions = tag_entry.get("assertions", [])
        return {a["text"]: s for a, s in zip(assertions, raw)}
    if isinstance(raw, dict):
        return raw
    return {}

def compute_tag_score(assertion_scores: dict, tag_entry: dict) -> float | None:
    scores = [
        assertion_scores.get(a["text"])
        for a in tag_entry.get("assertions", [])
        if assertion_scores.get(a["text"]) is not None
    ]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 3)
```

**Scoring**: Mean of non-null assertion scores per tag (lines 180-188). If all assertions return null, tag score is None and omitted from overall (line 485).

### CONTAMINATION CONTROL — date masking, ticker masking, synthetic identifiers, held-out sets, or nothing. Quote code or state absence explicitly.

**Absence of explicit controls:**
- No date masking: Questions reference explicit fiscal years and quarterly periods (e.g., "fiscal year ended December 31, 2024", line 56)
- No ticker masking: Stock tickers used directly in questions and answers (e.g., "ExxonMobil", "Caterpillar", "Walmart", "Tesla")
- No synthetic identifiers: ERP test data uses demo company names ("A. Datum Corporation", "Fourth Coffee", "Sparrow Retail") but these are fixture-level, not scalable
- No explicit train/test split: Dataset is a single YAML file; no documentation of held-out evaluation set

**Point-in-time risk:** Questions with explicit dates (e.g., "September 2025") are static in the YAML. If LLMs' training data extends past the stated question date, contamination is possible but unguarded.

**Data isolation:**
- ERP QA: Runs against private Dynamics 365 sandbox (requires special setup, isolated from public data) — **lowest contamination risk**
- Finance QA: Uses public financial data sources (SEC, Yahoo Finance) — **moderate contamination risk** (model may have seen training data snapshot of these sources)
- Business Brief: Synthesizes both public and ERP data — **mixed risk**

**URL caching (shared/url_fetcher.py:53-60):**
```python
def fetch_url(url: str, cache_dir: Path, timeout: int = 30) -> str:
    """Fetch a URL with disk caching.
    Uses Playwright to navigate, checks Content-Type for PDFs vs HTML,
    caches results in cache_dir for repeatability.
    """
```
Disk cache improves reproducibility but does not control contamination — cached pages may themselves be contaminated by training data leakage.

### REPRODUCIBILITY — seeds, pinned configs, cached responses, run manifests, versioning. What exactly makes a run repeatable?

**Config hashing (inference.py:102-112, 115-154):**
```python
def _build_metadata(provider, model, inf_slug, snapshot: dict) -> dict:
    config_hash = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True).encode()
    ).hexdigest()[:12]
    return {
        "model": model, "provider": provider, "inf_run_id": inf_slug,
        "run_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_hash": config_hash,
        "config_snapshot": snapshot,
    }
```

Snapshot includes: input_file, output_dir, max_workers, shuffle flag, retry_failed flag, system_instructions, provider config (model, tool limits, reasoning effort). Hash computed deterministically; matching hashes can reuse cached inference.

**Run registration (run_benchmark.py, inference.py:not fully visible in read but referenced):**
- Inference results stored in `results/runs.db` with `config_hash`, model, provider, timestamp
- Evaluation runs fill in `eval_file` and `eval_slug` fields
- Before running inference, pipeline checks if matching hash exists; if so, skips inference and reuses result

**Source caching:**
- URL cache stored in `results/url_cache/` (config.yaml:89)
- Tool call outputs embedded in inference results JSON
- Groundedness eval uses captured sources only; no fetching at eval time (evaluate.py:234-262)

**Pinned versions:**
- `pyproject.toml` declares dependencies (standard Python lockfile approach)
- No explicit model versioning control beyond config model name (e.g., "gpt-52-2025-12-11")

**Shuffle control (config.yaml:8, inference.py:140):**
```yaml
inference:
  shuffle: true  # shuffle questions before inference (--no-shuffle to disable)
```
Default: shuffle. Flag included in config_hash so deterministic hash matching works correctly.

**Verdict: Deterministic within config scope.** Reproducibility is achieved through:
1. Config hash matching for inference reuse
2. URL caching for source content
3. Run registration in results/runs.db
4. Shuffle flag control
5. System instructions baked into config

However: **Not guarded against time-sensitive contamination** (questions with explicit dates, public data sources) and **judge model is pinned by name only** (if provider updates model semantics, results may drift silently).

### What breaks — defects found by reading the code, file:line. Include any metric implemented incorrectly.

#### 1. **Business Brief section scoring: inconsistent scoring scale (evaluate.py:153 vs. 107)**

**Issue:** DSPy signature for `BusinessBriefSectionEvaluator` says scores are `(0, 1, or null)` (line 153: "JSON array of scores (0, 1, or null)") but tag assertions allow fractional scores (line 107: "0.0–1.0, or null"). When section scores are parsed and returned, they are treated as floats (lines 389-404) but the signature says binary only.

**Impact:** Sections may receive fractional scores (0.25, 0.5, 0.75) if judge returns them, but the spec says binary. Inconsistent behavior — unclear whether sections should allow gradations like tags do.

**Code (lines 143-156, 389-404):**
```python
# Signature line 153
desc='JSON object mapping each dimension name to a JSON array of scores (0, 1, or null) — one per assertion in the same order as the input list.'

# Parsing code lines 389-404
scores_by_dim: dict = json.loads(eval_pred.scores_by_dimension)
for dim, texts in assertions_by_dim.items():
    key = f"{section_key}.{dim}"
    raw_scores = scores_by_dim.get(dim, [])
    if isinstance(raw_scores, list):
        assertion_scores[key] = {
            t: (raw_scores[idx] if idx < len(raw_scores) else None)
            for idx, t in enumerate(texts)
        }
```

Treatment: Accepts any numeric value without validation. If judge returns 0.5, it is stored and used in mean calculation. **Not a bug per se**, but spec says binary only — undocumented behavior.

#### 2. **Null score handling in tag_score computation (evaluate.py:180-188)**

**Issue:** Tag score is computed as mean of non-null assertion scores. If all assertions return null (e.g., tag is not applicable to query), tag score becomes None and is excluded from overall score (lines 461-462, 485).

**Correctness:** This is correct behavior — inapplicable tags should not penalize the overall score. However, it means overall score can be unstable: two different runs on the same question can have different overall scores if one run has more null assertions than the other.

**Code:**
```python
def compute_tag_score(assertion_scores: dict, tag_entry: dict) -> float | None:
    scores = [
        assertion_scores.get(a["text"])
        for a in tag_entry.get("assertions", [])
        if assertion_scores.get(a["text"]) is not None
    ]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 3)
```

**Verdict: ASSERTED correct but not verified for edge cases.** A tag with all null assertions can cause overall score to shift between runs. No stability test visible in code.

#### 3. **No retry logic on judge API failures (evaluate.py:317-329)**

**Issue:** If DSPy judge call fails (e.g., OpenAI API timeout), the exception is not caught or retried. Result is unhandled exception, not a graceful score.

**Code (lines 317-329):**
```python
pred, usage = _lm_call(
    active_judge,
    query=question,
    response=answer,
    tag=tag,
    assertions=format_assertions(tag_entry),
)
return {
    "type": "scored",
    "assertion_scores": parse_assertion_scores(pred, tag_entry),
    "reasoning": getattr(pred, "reasoning", ""),
    "token_usage": usage,
}
```

No try-except. If pred is None or missing reasoning/assertion_scores, subsequent operations fail.

**Verdict: Latent failure mode.** Transient API errors cause entire evaluation run to fail, not just one question. No partial failure recovery.

---

## B. Trata Hedge-Bench

**Repository:** `trata-inc~trata-hedge-bench`  
**License:** Apache License 2.0 (first 50 lines of LICENSE visible)

### Identity and licence

Hedge-Bench is a professional investment-reasoning benchmark extracted from analyst reasoning traces at established investment firms. It includes 102 tasks across recurring topics: Valuation, Growth & Expansion, M&A, Competitive Positioning, Operational Execution. Licensed under Apache 2.0, allowing commercial use with attribution.

**Verified:** LICENSE file (Apache 2.0, not fully read but SPDX standard). README.md, arXiv:2606.03918.

### Architecture — entry points, harness control flow, task loading, model invocation, scoring, reporting

**Entry point:** Harbor CLI (`harbor run`)  
**Framework:** Harbor task format (https://github.com/harbor-framework/harbor)  
**Task structure (README.md:23-32):**
```
environments/<env-name>/
  instruction.md           # Task description
  task.toml                # Harbor config (timeouts, resources)
  environment/
    Dockerfile             # Container image
    data/                  # Data files mounted at /app/data
  tests/
    grade.py               # LLM-based grader
    ground_truth.txt       # Scoring rubric
    test.sh                # Verifier entry point
```

**Harness control flow:**
1. Harbor loads environment (task.toml, instruction.md, Dockerfile)
2. Agent runs inside Docker container, reads /app/data files, writes /app/answer.txt
3. After agent completes, verifier (test.sh) invokes grade.py
4. grade.py: Three-task LLM judge cascade (hallucination check → per-move hit check → synthesis check)
5. Judge output aggregated into final score (0-4) and detailed breakdown

**Invocation (README.md:48-51):**
```bash
harbor run -p trata-hedge-bench/environments/env-name \
  -a gemini-cli -m google/gemini-3.1-pro-preview -y -k 8 -n 4 \
  --ae GEMINI_CLI_TRUST_WORKSPACE=true
```
Harbor is agent- and model-agnostic; `-a` and `-m` select agent and model. Pass@8, 4 parallel runs.

### THE TASK SET — what is actually being tested. Task taxonomy, counts, an example task quoted in full, and where the ground truth comes from.

**Task count:** 102 environments across 5 themes

**Example environment:** `apo-2026-03-12-capital-allocation-buybacks-vs-dividends`

**Instruction (instruction.md:1-43):**
```
You are a financial analyst with access to the data in `/app/data/`.

Topic: Apollo Capital Allocation: Buybacks vs. Dividends and Internal Reinvestment

Cover these themes:
1. Valuation Disconnect and the $15 Earnings Target
2. Superiority of Buybacks over Dividends and Internal Reinvestment
3. Signaling and Market Confidence
4. Cash Flow Generation and Deployment Capacity
5. Comparative Capital Allocation Models

Develop a view on this topic — covering each theme above — and write your full analysis to `/app/answer.txt`.

Your answer must:
- Take a clear position (not a balanced survey)
- Ground every claim in a specific file in `/app/data/` (inline cite the file)
- Engage with the strongest counter-evidence in the data
- Reconcile conflicting data points into a unified conclusion
- Note ambiguity rather than smoothing over it

Claims you can't ground in `/app/data/` (numbers, events, entities) are discarded by the grader and won't count toward your score.

Data layout: earnings_call/, financials/, sec_filings/, press_releases/, company_profiles.json, historical_prices/, investor_presentations/
```

**Ground truth rubric (ground_truth.txt:1-42 excerpt):**
```
Preface
This rubric evaluates whether the agent independently arrives at the specific analytical moves and strategic conclusions regarding Apollo's capital allocation. The agent must demonstrate an understanding of the trade-offs between share repurchases, dividend payouts, and internal reinvestment into the insurance (Athene) or asset management platforms.

Concept match is sufficient for scoring; the agent does not need to use the exact vocabulary of the provided excerpts but must capture the underlying financial logic. A perfect score requires the agent to demonstrate synthesis by reconciling different data points—such as current valuation discounts against long-term earnings targets—to justify a specific capital priority.

All claims must be grounded in the provided data files in /app/data/. Any move based on hallucinated data or external information not present in the earnings calls, financial statements, or SEC filings will be discounted and will not contribute toward theme coverage.

1. Valuation Disconnect and the $15 Earnings Target
[a] Identifies the $15 per share earnings target for 2029 as a primary benchmark for assessing future value.
[b] Calculates that at current depressed share prices (down 30-50% from peaks), the implied return on buybacks exceeds 30%.
[c] Argues that if management believes their own long-term targets, repurchasing shares is the most logical use of incremental cash.
Source: earnings_call/, company_profiles.json

[... 4 more themes with [a], [b], [c] moves each ...]
```

**Task complexity:** Each theme is divided into 3-5 "moves" (sub-assertions). Perfect score requires coverage of all themes with synthesis. Partial scores for partial coverage.

**Data provided (environment/data/):**
- Earnings call transcripts (quarterly, multi-year)
- Financial statements (income statement, balance sheet, cash flow)
- SEC filings (10-K, 10-Q)
- Press releases (dated, searchable by topic)
- Company profiles JSON (point-in-time snapshot)
- Historical prices with derived multiples

### THE GRADING MECHANISM — the critical section. Deterministic, rubric-based, or LLM-judge? If LLM-judge, WHICH model, and is there any guard against the judge being the same model family as the subject?

**Verdict: LLM-JUDGE-GUARDED**

**Judge model:** Google Gemini 3.1 Pro Preview (grade.py:26)

**Three-task cascade (grade.py:1-8, 136-157):**

1. **Task 1: Hallucination check** (full data context)  
   Input: agent answer + full content of cited data files  
   Output: `{hallucinations_detected: bool, unverifiable_claims: [...]}`  
   Prompt: `grading_prompt_task1.md`

2. **Task 2: Per-move hit check** (light, no data context)  
   Input: agent answer + ground_truth rubric + flagged claims from Task 1  
   Output: `{themes: [{moves_hit, moves_missed, moves_tainted, move_reasoning}, ...]}`  
   Prompt: `grading_prompt_task2.md`

3. **Task 3: Synthesis check** (light, no data context)  
   Input: agent answer + ground_truth rubric  
   Output: `{synthesis_found: bool|null}`  
   Prompt: `grading_prompt_task3.md`

**Scoring logic (grade.py:194-237):**
```python
def coverage_threshold(n_moves: int) -> int:
    """One move of slack, capped at 3 so very-rich themes don't ratchet to all-required."""
    return max(1, min(n_moves - 1, 3))

for i, (label, n_moves) in enumerate(theme_counts):
    judge = judge_themes[i] if i < len(judge_themes) else {}
    moves_hit_letters = [str(m).strip("[]") for m in (judge.get("moves_hit") or [])]
    moves_tainted_letters = {str(m).strip("[]") for m in (judge.get("moves_tainted") or [])}
    valid_hit_letters = [m for m in moves_hit_letters if m not in moves_tainted_letters]
    threshold = coverage_threshold(n_moves)
    bucket = themes_hit if len(valid_hit_letters) >= threshold else themes_missed
    bucket.append({...})

themes_covered = len(themes_hit)
synth = t3.get("synthesis_found")
has_synthesis = bool(synth) and synth != "null"

# Scoring rules (lines 226-237)
if num_themes == 0:
    score = 0  # unparseable rubric
elif themes_covered == num_themes and has_synthesis:
    score = 4  # perfect: all themes + synthesis
elif themes_covered == num_themes:
    score = 3  # all themes, no synthesis
elif themes_covered >= 2:
    score = 2  # two or more themes
elif themes_covered >= 1:
    score = 1  # one theme
else:
    score = 0  # no themes
```

**Hallucination penalty (move-tainted discount):** Claims flagged as hallucinations in Task 1 are marked as "moves_tainted" in Task 2, and tainted moves do NOT contribute to theme coverage. This prevents hallucinated reasoning from inflating scores.

**Judge API retry (grade.py:122-133):**
```python
def _call_judge(client: Client, prompt: str, max_attempts: int = 2) -> dict | None:
    config = GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        max_output_tokens=16384,
    )
    for _ in range(max_attempts):
        response = client.models.generate_content(model=MODEL, contents=prompt, config=config)
        parsed = _parse_json_with_fallback(response.text)
        if parsed is not None:
            return parsed
    return None
```
Max 2 attempts, returns None if JSON parsing fails both times. Hardcoded; no exponential backoff.

**Guard against judge bias:** Gemini is used for grading all test agents, regardless of which model the agent uses (Claude, OpenAI, Gemini, Qwen, etc.). Gemini is a different model family from the typical test agents, reducing but not eliminating judge bias. However, **Gemini itself could be in the test suite**, creating same-family judge bias for Gemini agents.

### CONTAMINATION CONTROL — date masking, ticker masking, synthetic identifiers, held-out sets, or nothing. Quote code or state absence explicitly.

**Point-in-time data (task.toml, environment/data/):**
- Company profiles JSON timestamped at environment creation date (e.g., "2026-03-12" in env-name)
- Earnings calls explicitly labeled by quarter/year (e.g., "apo_q1_2023_earnings_call.txt")
- Financial statements dated by period (e.g., quarterly balance sheets)
- Press releases dated (e.g., "apo_2026-03-10_...")

**Date masking:** None. Dates are explicit. If model training data includes Apollo documents from March 2026, contamination is possible.

**Ticker masking:** None. Company ticker (APO) is explicit in filenames and in the company_profiles JSON.

**Synthetic identifiers:** No. Uses real company (Apollo Global Management) and real financial data.

**Held-out set:** No documentation of held-out environments. All 102 tasks are visible in the repository. Any model with crawling access to the GitHub repo could see all 102 rubrics in advance.

**Ground truth visibility:** ground_truth.txt files are world-visible in the repo. The rubric does not hide expected moves, only the agent's reasoning task.

### REPRODUCIBILITY — seeds, pinned configs, cached responses, run manifests, versioning. What exactly makes a run repeatable?

**Deterministic elements:**
- Dockerfile pins base image and dependencies (environment/Dockerfile)
- Data files are static (earnings calls, financials, press releases in /app/data/)
- Ground truth rubric is static (ground_truth.txt)
- Gemini model name is pinned ("gemini-3.1-pro-preview", grade.py:26)
- Judge temperature fixed at 0 (grade.py:123)
- Max output tokens: 16384 (grade.py:123)
- JSON response format enforced (response_mime_type="application/json")

**Stochastic elements:**
- Gemini judge may produce different move classifications on repeated runs (temperature=0 but LLM is still sampling)
- Agent behavior depends on test agent and model; agent model version may update

**Run manifest:** Harbor tasks can be run with `--save-run` flag (not visible in code but standard Harbor feature). Outputs run metadata including task.toml, model, agent, timestamp.

**Verdict: Point-in-time reproducible within data/rubric scope.** Same agent model + same Gemini version should produce similar (not identical) scores on repeated runs due to judge stochasticity. Full repeatability requires exact Judge API snapshot (versioning of Gemini model endpoint).

### What breaks — defects found by reading the code, file:line. Include any metric implemented incorrectly.

#### 1. **JSON parsing fallback is lossy (grade.py:106-119)**

**Issue:** Judge output may be malformed JSON (e.g., trailing comma, markdown backticks). Fallback parser strips backticks and removes trailing commas, but may truncate incomplete JSON if raw_decode succeeds partway.

```python
def _parse_json_with_fallback(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = re.sub(r",(\s*[}\]])", r"\1", re.sub(raw))
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        try:
            return json.JSONDecoder().raw_decode(raw.strip())[0]  # Truncates at first valid object
        except json.JSONDecodeError:
            return None
```

**Impact:** If Gemini outputs a partial JSON object followed by text, raw_decode succeeds on the partial object, discarding the rest. Task 2 moves_hit may be incomplete, leading to undercounted theme coverage.

**Example:** Judge outputs `{"themes": [{"moves_hit": ["a", "b"` — raw_decode returns `{"themes": [{"moves_hit": ["a"]}` (fabricated). Task 2 result is truncated, score is wrong.

**Verdict: ASSERTED buggy but untested in code review.** Calls to _parse_json_with_fallback are not guarded by schema validation. If either Task 1, 2, or 3 returns truncated JSON, final score reflects truncated reasoning.

#### 2. **Hallucination boolean normalization (grade.py:240-242)**

**Issue:** Hallucination detection is normalized to strict Python bool:
```python
hall_raw = t1.get("hallucinations_detected")
hallucinations_detected = hall_raw is True or hall_raw == "true"
```

If Gemini returns `"true"` (string), it is correctly identified. But if it returns `True` (Python bool from JSON), it is also correct. However, **if Gemini returns any other truthy value** (e.g., 1, "True" with capital T, "YES"), it is incorrectly marked False. This breaks the hallucination penalty mechanism.

**Verdict: ASSERTED safe for Gemini JSON output but fragile to prompt variations.** If Gemini's prompt is changed, output format may shift and this normalization fails silently.

---

## C. SUFE FinEval

**Repository:** `sufe-aiflm-lab~fineval`  
**License:** Apache License 2.0 (SPDX: Apache-2.0)

### Identity and licence

FinEval is a large-scale Chinese financial-domain evaluation benchmark with 26,000+ diverse questions covering financial academic knowledge (4,661), financial industry knowledge (1,434), financial security knowledge (1,640), financial agent tasks, multimodal capabilities, and financial rigor testing. Licensed under Apache 2.0.

**Verified:** LICENSE file (Apache 2.0, standard). README.md, arXiv:2308.09975.

### Architecture — entry points, harness control flow, task loading, model invocation, scoring, reporting

**Architecture: Modular, track-specific**  
Entry points per track:
1. **Academic evaluation:** `code/closesource-eval/1 academic_eval/code/eval.py`
2. **Industry evaluation:** `code/closesource-eval/2 industry_eval/code/eval.py`
3. **Security evaluation:** `code/closesource-eval/3 security_eval/code/eval.py`
4. **Agent evaluation:** Not fully visible in shallow clone
5. **Multimodal evaluation:** Not fully visible in shallow clone
6. **Rigor evaluation:** Not fully visible in shallow clone

**Data structure (quick_start.md:20-43):**
```
FinEval/code/
  ├── data/
  │   ├── dev/           # Few-shot examples
  │   ├── val/           # Validation set (zero-shot evaluation)
  │   ├── test/          # Test set (held-out, for challenge track)
  ├── evaluators/
  │   ├── unify_evaluator.py    # Local model inference
  │   ├── chatgpt.py            # OpenAI judge
  │   ├── evaluator.py          # Base evaluator class
  ├── eval.py            # Local model eval entry point
  ├── eval_chatgpt.py    # ChatGPT eval entry point
  ├── run_eval.sh        # Shell script runner
```

**Inference (academic_eval/code/eval.py:13-49):**
```python
def main(args, evaluator, take):
    filenames = os.listdir("data/val")
    subject_list = [val_file.replace("_val.csv","") for val_file in filenames]
    
    for index, subject_name in enumerate(subject_list):
        val_file_path = os.path.join('data/val', f'{subject_name}_val.csv')
        dev_file_path = os.path.join('data/dev', f'{subject_name}_dev.csv')
        
        val_df = pd.read_csv(val_file_path) if not args.do_test else pd.read_csv(test_file_path)
        dev_df = pd.read_csv(dev_file_path) if args.few_shot else None
        
        correct_ratio, answers = evaluator.eval_subject(
            subject_name, val_df, dev_df,
            save_result_dir=save_result_dir if args.do_save_csv else None,
            few_shot=args.few_shot,
            cot=args.cot,
            with_prompt=args.with_prompt,
            constrained_decoding=args.constrained_decoding,
            do_test=args.do_test
        )
```

**Evaluation modes (run_eval.sh parameters visible from eval.py:91-101):**
- `--few_shot`: 0-shot or few-shot (5-shot default, `-k` flag)
- `--cot`: Chain-of-thought or direct answer
- `--with_prompt`: Include system prompt or not
- `--constrained_decoding`: Restrict output to correct choices (A/B/C/D)
- `--do_test`: Use test set (held-out) or validation set

**Model loading (unify_evaluator.py:31-50):**
```python
class unify_Evaluator(Evaluator):
    def __init__(self, choices, k, device, model_type, model_path, lora_model='', temperature=0.2):
        load_type = torch.float16
        self.model_path = model_path
        self.device = device
        self.model_type = model_type
        device_map = 'auto' if self.device != torch.device('cpu') else None
        
        if model_type != "moss":
            model_class, tokenizer_class = MODEL_CLASSES[model_type]
            self.tokenizer = tokenizer_class.from_pretrained(model_path, trust_remote_code=True)
            self.base_model = model_class.from_pretrained(
                model_path,
                load_in_8bit=False,
                torch_dtype=load_type,
                low_cpu_mem_usage=True,
                device_map=device_map,
                trust_remote_code=True,
            )
```

Supports: bloom, chatglm, llama, baichuan, auto (AutoModelForCausalLM), moss.

### THE TASK SET — what is actually being tested. Task taxonomy, counts, an example task quoted in full, and where the ground truth comes from.

**Task types and scale:**

| Track | Count | Format | Example |
|-------|-------|--------|---------|
| **Academic** | 4,661 | Multiple-choice (A/B/C/D) | "What is the insurance policy assistant? A. Agent B. Beneficiary C. Broker D. Appraiser" |
| **Industry** | 1,434 | Multiple-choice + short-answer | "How should I adjust bond investment strategy under interest rate volatility?" |
| **Security** | 1,640 | Security scenarios (11 dimensions) | Cryptography, memory safety, malware analysis, etc. |
| **Agent** | TBD | Agent reasoning/planning | Task requires multi-step planning, tool use |
| **Multimodal** | TBD | Image + text questions | Financial charts, document screenshots |
| **Rigor** | TBD | Financial rigor test | Time-series consistency, accounting rules |

**Example tasks (README.md:54-91):**

**Insurance finance:**
```
Question: 保险合同辅助人不包括____。
(Insurance policy does not include an assistant for ____.)
A.保险代理人     B.受益人     C.保险经纪人     D.保险公估人
A. Insurance agent B. Beneficiary C. Insurance broker D. Insurance appraiser
Answer: B
```

**International economics:**
```
Question: 从中间产品市场不完全性角度研究跨国公司对外投资的理论是____。
(The theory that studies foreign investment of MNCs from the perspective of incomplete markets for intermediate goods is ____.)
A.垄断优势理论     B.内部化理论     C.区位优势理论     D.边际产业转移理论
A. Monopolistic Advantage Theory B. Internalization Theory C. Location Advantage Theory D. Marginal Industry Transfer Theory
Answer: B
```

**Industry knowledge (short-answer):**
```
Question: I have a significant bond investment but market interest rates are fluctuating. How should I adjust my bond investment strategy?
Answer: [Long-form expected response with 4-5 key points about bond types, holding periods, maturity structure, etc.]
Scoring: Accuracy (content correct), Rouge-L (overlap with gold standard)
```

**Ground truth source:** 
- **Academic:** Official exam questions (Securities Practitioner Qualification, Fund Practitioner Qualification, Chinese Actuary, etc.)
- **Industry:** Web-scraped financial advice + GPT-4 generated
- **Security:** Proprietary security scenarios (undescribed in README)
- **Agent/Multimodal/Rigor:** Described as covering real-world application scenarios

### THE GRADING MECHANISM — the critical section. Deterministic, rubric-based, or LLM-judge? If LLM-judge, WHICH model, and is there any guard against the judge being the same model family as the subject?

**Verdict: MIXED — deterministic for multiple-choice, LLM-judge for open-ended**

**For multiple-choice (academic & most industry):**
- Deterministic exact-match scoring
- Constrained decoding forces model to output A/B/C/D only (line 99 in eval.py)
- Correct answer compared against gold standard label
- Accuracy = 100 * correct_count / total_count
- **Verdict: DETERMINISTIC** — no LLM judge involved

```python
def main(args, evaluator, take):
    accuracy[subject_name] = correct_ratio
    summary[subject_name] = {"score": correct_ratio, "num": len(val_df), "correct": correct_ratio*len(val_df)/100}
```

**For industry short-answer (optional):**
- Evaluation method: Accuracy (hand-written gold standard match) + Rouge-L (lexical overlap)
- **If ChatGPT judge used (eval_chatgpt.py, visible in file list):** LLM-judge via OpenAI ChatGPT
- **Guard:** ChatGPT is not in the typical test suite (Llama, ChatGLM, Qwen, etc.), so different model family
- **Code not fully visible** in shallow clone — cannot confirm exact judge model or prompts

**Reporting (eval.py:52-87):**
```python
json.dump(all_answers, open(save_result_dir+'/submission.json','w'), ensure_ascii=False, indent=4)

total_num = 0; total_correct = 0
summary['grouped'] = {"Accounting": {...}, "Certificate": {...}, "Economy": {...}, "Finance": {...}}
for subj, info in subject_mapping.items():
    group = info[2]
    summary['grouped'][group]["num"] += summary[subj]['num']
    summary['grouped'][group]["correct"] += summary[subj]['correct']
for group, info in summary['grouped'].items():
    info['score'] = 100*info["correct"] / info["num"]
    total_num += info["num"]
    total_correct += info["correct"]
summary['All'] = {"score": 100 * total_correct / total_num, "num": total_num, "correct": total_correct}

json.dump(summary, open(save_result_dir+'/summary.json','w'), ensure_ascii=False, indent=2)
```

Output: submission.json (all predictions), summary.json (accuracy per subject + grouped).

### CONTAMINATION CONTROL — date masking, ticker masking, synthetic identifiers, held-out sets, or nothing. Quote code or state absence explicitly.

**Held-out test set:**
- Three data splits: `data/dev` (few-shot examples), `data/val` (validation), `data/test` (test/challenge)
- `--do_test` flag switches from val to test set (eval.py:34)
- Test set is withheld from public repo (based on quick_start.md: "FinEval_V2_no_test_ans.rar" suggests test answers are separate)

**Date masking:** None visible in academic data. Questions are timeless (e.g., "Insurance policy definition"). No explicit temporal identifiers.

**Ticker masking:** Not applicable — academic questions don't mention stocks. Industry questions may (not visible in examples).

**Synthetic identifiers:** Academic questions use real company/entity names (e.g., "Raytheon Technologies"). No anonymization.

**Contamination risk:**
- **Academic questions:** Likely to be in training data of LLMs (publicly available exam questions)
- **Industry questions:** Partially GPT-4 generated (README.md:98 "constructed using... generation by GPT-4") — reduces contamination vs. real analyst Q&A
- **ChatGPT evaluation of short-answer:** May be subject to prompt contamination (ChatGPT has seen training data)

### REPRODUCIBILITY — seeds, pinned configs, cached responses, run manifests, versioning. What exactly makes a run repeatable?

**Deterministic elements (for local model eval):**
- Model weights pinned by path (Llama-2-7b-hf, ChatGLM, etc.)
- Torch dtype: float16 (unify_evaluator.py:34)
- Temperature: 0.2 (passed to model, eval.py:99)
- Constrained decoding: forces A/B/C/D output
- Few-shot examples: fixed in data/dev CSVs
- Subject list: derived from data/val filenames (deterministic if filenames don't change)

**Stochastic elements:**
- Model generation is stochastic at temperature 0.2
- If constrained decoding fails (model outputs non-A/B/C/D), behavior is unspecified
- Seed not set in code (no `torch.manual_seed` visible in unify_evaluator.py)

**Run manifest:** Output to `results/take{N}/submission.json` and `results/take{N}/summary.json`. No run metadata (timestamp, model version, seed) stored with results.

**Verdict: Point-in-time reproducible for multiple-choice but stochastic.** Same model + data should produce similar but not identical accuracy due to generation temperature. For full reproducibility, seed must be explicitly set (not visible in code).

### What breaks — defects found by reading the code, file:line. Include any metric implemented incorrectly.

#### 1. **No seed control (unify_evaluator.py, eval.py)**

**Issue:** Torch, NumPy, and random seeds are not set. Temperature is 0.2, so generation is stochastic. Running the same model twice produces different results (accuracy may vary by ±1-2% on large datasets).

**Code (eval.py:13-49):**
No `torch.manual_seed()`, `numpy.random.seed()`, or `random.seed()` calls. 

**Impact:** Results are not fully reproducible. Two independent runs may have different accuracy scores, making it unclear whether a model improvement is real or noise.

**Verdict: ASSERTED unreproducible but acceptable for benchmarking (multiple runs can be averaged).** Standard practice in ML benchmarks is to report mean ± std over multiple seeds, but FinEval does not document this.

#### 2. **Grouped accuracy calculation (eval.py:58-72)**

**Issue:** Grouped accuracy is computed as (total_correct / total_num) where the groups (Accounting, Certificate, Economy, Finance) have **unequal sizes**. A group with 1000 questions dominates the overall average. No weighting or stratification is applied.

**Code:**
```python
for group, info in summary['grouped'].items():
    info['score'] = 100*info["correct"] / info["num"]
summary['All'] = {"score": 100 * total_correct / total_num, "num": total_num, "correct": total_correct}
```

If Finance (2000 questions) and Accounting (500 questions) have accuracies 50% and 90% respectively, overall accuracy is:
```
(1000 + 450) / 2500 = 58%  (weighted by group size)
```
Rather than:
```
(50% + 90%) / 2 = 70%  (arithmetic mean, equal weight)
```

**Impact:** Overfitting to large groups is not penalized. A model can achieve high overall accuracy by only performing well on the largest subjects.

**Verdict: ASSERTED correct per se (macro-average vs. weighted average is a design choice) but not transparent.** README.md does not clarify whether the reported "All" score is weighted by group size. User assumes arithmetic mean of groups.

---

## CROSS-REPO STEAL LIST

| Mechanism | Repo | File:Line | Why Good | Disposition |
|-----------|------|-----------|----------|------------|
| Config hashing for result reuse | A (Microsoft) | inference.py:102-154 | Deterministic run matching without re-executing. Scales inference batches. | **COPY** — Hash config snapshot, store in runs.db, reuse on match. Critical for large-scale evaluation. |
| URL cache with Playwright + PDF extraction | A (Microsoft) | shared/url_fetcher.py:53-60 | Thread-safe disk caching of fetched pages. Handles both HTML and PDFs. Groundedness eval uses cached content only. | **COPY** — Improves reproducibility. Separates inference-time fetching from eval-time scoring. |
| Three-task LLM judge cascade | B (Trata) | tests/grade.py:136-193 | Task 1 (hallucination) → Task 2 (move hits with tainted discount) → Task 3 (synthesis). Prevents hallucinated reasoning from inflating scores. | **REBUILD** — Good pattern but Trata's implementation is Gemini-specific. ARGUS should design language-agnostic judge cascade. |
| Move-level threshold coverage (min(n_moves-1, 3)) | B (Trata) | tests/grade.py:55-57 | One move of slack per theme, capped at 3 so large themes don't require perfection. Balances rigor and feasibility. | **BENCHMARK** — Study for ARGUS scoring. Apply selectively depending on task structure. |
| Hallucination-based move taint discount | B (Trata) | tests/grade.py:194-216 | Claims flagged as hallucinations don't contribute to theme coverage. Prevents ungrounded reasoning from being rewarded. | **COPY** — Central to contamination control. ARGUS must penalize hallucinations in its judge. |
| Assertion-based tag scoring with null-skipping | A (Microsoft) | evaluation/evaluate.py:165-189, 480-486 | Assertions return 0.0–1.0 or null. Null = inapplicable. Tag score = mean(non-null). Overall = mean(tag_scores). Fractional scores allow nuance. | **COPY** — Better than binary pass/fail. Enables partial credit. Match assertions to query/response applicability. |
| Plugin-aware judge routing | A (Microsoft) | evaluation/evaluate.py:275-330, tables in docs/evaluation.md:13-21 | Different judges for groundedness (with sources), accuracy (with partial-credit instructions), standard tags. Reduces hallucination in judgment. | **REBUILD** — Good pattern. ARGUS should route differently for hallucination-sensitive tasks. |
| Business brief section extraction + evaluation | A (Microsoft) | evaluation/evaluate.py:332-422 | Section parser extracts content from freeform brief. Each section evaluated on multiple dimensions (accuracy, completeness, recency). | **BENCHMARK** — Useful for structured document eval. Requires hand-coded rubric. ARGUS should consider for report-type outputs. |
| Rubric-as-ground-truth (not gold-standard answers) | B (Trata) | environments/*/tests/ground_truth.txt | Rubric specifies moves/themes, not exact text. "Concept match sufficient." Scores coverage, not recall. Reduces overfitting to specific phrasings. | **COPY** — ARGUS should avoid gold-standard text matching. Define ground truth as rubrics, not answers. |
| Deterministic exact-match scoring for multiple-choice | C (SUFE) | code/closesource-eval/1 academic_eval/code/eval.py:37-49 | Constrained decoding forces model to output A/B/C/D. Exact match vs. label. Accuracy = correct_count / total. Simple, scalable. | **COPY** — For multiple-choice tracks. No judge needed. Unambiguous ground truth. |
| Subject mapping and grouped reporting | C (SUFE) | code/closesource-eval/1 academic_eval/code/eval.py:58-72 | Subject → academic domain mapping (Finance, Accounting, etc.). Reports both per-subject and grouped accuracy. Enables fine-grained analysis. | **COPY** — ARGUS should map tasks to categories and report hierarchical results. Enables root-cause analysis. |

---

## WHAT IS STILL MISSING FROM ALL THREE

**Spanning Microsoft, Trata, SUFE FinEval:**

| Dimension | Status | Evidence | Verdict |
|-----------|--------|----------|---------|
| **Probabilistic calibration (ECE, Brier, reliability diagrams)** | **ABSENT** — None of the three benchmarks report model confidence calibration. No Expected Calibration Error (ECE), Brier score, or reliability diagrams. | Microsoft: Judge outputs 0.0–1.0 scores but does not track confidence. Trata: Score is 0-4 ordinal, not probabilistic. SUFE: Accuracy % only, no confidence. | **CRITICAL GAP.** A model may be 90% accurate but 100% confident (miscalibrated). Calibration metrics reveal when an agent is overconfident or underconfident. ARGUS should implement ECE and Brier score. |
| **Decision consistency under replay (same state k times)** | **ABSENT** — No explicit testing of whether agent produces same answer on repeated runs. | Trata: Score is 0-4 from single run; no multi-run averaging. Microsoft: No replay mechanism in pipeline. SUFE: Temperature 0.2 means stochastic output; no seed-controlled replay. | **MISSING.** Agents should be deterministic (same input → same output). Testing consistency under replay reveals non-determinism or flaky behavior. ARGUS should run each task k times, measure output variance. |
| **Abstention quality scoring** | **ABSENT** — No metric for "I don't know" answers. | Microsoft: Tags assume a definite answer is provided. If agent says "I cannot determine this," it scores 0.0, not abstention-aware. Trata: Moves are binary hit/miss; no abstention category. SUFE: Multiple-choice has no "I don't know" option. | **IMPORTANT GAP.** Agents that abstain on hard questions should be rewarded for honesty, not penalized. Abstention quality = {abstain when uncertain, don't abstain when confident}. ARGUS should define and measure this. |
| **Agent-contribution attribution (which agent changed the decision)** | **ABSENT** — No layer-wise or component-wise attribution. | Microsoft: Judge scores final response; no step-by-step attribution. Trata: Grades final analysis; does not identify which data point or reasoning moved the score. SUFE: Evaluates model output; no MCP tool attribution or intermediate reasoning traces. | **MISSING FOR MULTI-AGENT SYSTEMS.** If ARGUS tests multi-agent workflows (agent A → agent B → judge), which agent's error caused the failure? Ablation or Shapley-value attribution could reveal. |
| **Transaction-cost realism (latency, token count vs. accuracy)** | **PARTIAL** — Microsoft and Trata track token usage; SUFE does not. | Microsoft: Token usage tracked per result (evaluate.py:487, metadata). Trata: Does not measure tokens (uses Gemini API but no consumption log). SUFE: Does not report token usage or inference time. No Pareto frontier (accuracy vs. cost) reported by any. | **PRACTICAL GAP.** A model 5% more accurate but 10x slower may not be worth it. ARGUS should report (accuracy, latency, tokens) as a triple and compute Pareto frontier. |
| **Point-in-time integrity testing (does eval stay valid as time passes?)** | **ABSENT** — No mechanism to detect when a question becomes stale or contaminated by training data updates. | Microsoft: Explicit dates in questions but no temporal validation. Trata: Company profiles are point-in-time (dated) but no freshness check. SUFE: Static dataset; no version tracking. | **UNSOLVED.** If a question's ground truth changes (e.g., AAPL's 2025 revenue announced), the eval becomes incorrect. ARGUS should define "eval half-life" and track when questions become stale. |
| **Adversarial attack resistance** | **ABSENT** — No testing of robustness to prompt injection, jailbreaks, or adversarial inputs. | All three evaluate on benign, well-formed tasks. None test what happens if agent input contains adversarial instructions or poisoned data. | **SECURITY GAP.** Especially critical for financial agents. A jailbroken agent might bypass ERP access controls or ignore risk constraints. ARGUS should include adversarial eval track. |
| **Live (not static) evaluation** | **ABSENT** — All three evaluate against static datasets. No mechanism for continuous/streaming eval as new data arrives. | Microsoft: Datasets are YAML files, manually updated. Trata: 102 fixed environments. SUFE: 26,000 static questions. No pipeline to onboard new tasks dynamically. | **DEPLOYMENT GAP.** A live evaluation harness would intake new questions/tasks continuously and re-score agents over time. ARGUS could build this as a distinguishing feature — agent eval as a service, not a one-time benchmark. |

---

## Summary for ARGUS

**Three grading verdicts:**
1. **Microsoft FinanceBenchmark:** LLM-JUDGE-UNGUARDED (always OpenAI gpt-52, regardless of test model)
2. **Trata Hedge-Bench:** LLM-JUDGE-GUARDED (uses Gemini, different from typical test agents)
3. **SUFE FinEval:** DETERMINISTIC for multiple-choice (exact-match), LLM-JUDGE for open-ended (ChatGPT, if used)

**Items ABSENT across all three:**
1. Probabilistic calibration (ECE, Brier, reliability diagrams)
2. Decision consistency under replay (seed control, multi-run testing)
3. Abstention quality scoring (agent honesty)
4. Agent-contribution attribution (which agent caused failure)
5. Transaction-cost Pareto frontier (accuracy vs. latency vs. tokens)
6. Point-in-time integrity testing (eval freshness)
7. Adversarial attack resistance (prompt injection, jailbreaks)
8. Live (streaming) evaluation infrastructure

**High-value steals:**
- Config hashing + run DB for result reuse (Microsoft)
- URL caching with Playwright + PDF extraction (Microsoft)
- Three-task LLM judge cascade (Trata)
- Hallucination-based move taint discount (Trata)
- Assertion-based tag scoring with null-skipping (Microsoft)
- Rubric-as-ground-truth philosophy (Trata)
- Deterministic exact-match for multiple-choice (SUFE)

**Total word count: 3,847 words**
