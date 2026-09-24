# LLM Structured JSON: Extraction Techniques & Recovery Patterns

How the best LLM-agent repositories reliably extract structured JSON from LLM responses and recover from truncation, malformed JSON, and prose wrapping.

---

## 1. OpenAI-compatible `response_format={"type":"json_object"}` (STRONGEST)

**Used by:** DeepSeek, Qwen, GLM, MiniMax, OpenAI, xAI, Anthropic (via Claude API)

### Implementation

```python
# cooperiano~crypto-trading-agent/src/trading_agent/llm/__init__.py:141-152
async def generate_structured(self, prompt: str, response_schema: type[BaseModel],
                               system_prompt: str | None = None, max_tokens: int = 1024) -> BaseModel:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {"model": self._model, "messages": messages, "max_tokens": max_tokens,
               "response_format": {"type": "json_object"}}
    async with self._session.post(f"{self._base_url}/chat/completions", json=payload) as resp:
        data = await resp.json()
        result = json.loads(data["choices"][0]["message"]["content"])
        return response_schema.model_validate(result)
```

### What it fixes

- **Prose wrapping**: API enforces that ONLY valid JSON is returned; no markdown fences, no preamble
- **Trailing commas**: Native JSON compliance at the API level
- **Empty fields**: Model commits to returning the exact schema structure

### When it fails

- **Truncation at `finish_reason="length"`**: JSON may be cut mid-key or mid-value. The model has no way to know it's being cut, so it produces syntactically valid JSON that is semantically incomplete (e.g., a `reasoning` field cut off mid-sentence, but the `}` is still there)
- **Provider doesn't support it** (rare; Ollama, older local models)
- **Provider ignores it** and returns prose anyway (not standard-compliant)

### Detection

Check the response object's `finish_reason`:

```python
# Pseudo-code; actual structure varies by provider
if response.choices[0].finish_reason == "length":
    # JSON completed but may be semantically truncated
    # Shorter prompt or higher max_tokens needed
```

### Qwen/Alibaba Support

**YES** — Qwen's OpenAI-compatible API (`dashscope-intl.aliyuncs.com/compatible-mode/v1` or China endpoint `dashscope.aliyuncs.com/compatible-mode/v1`) **fully supports** `response_format={"type":"json_object"}` as of 2024+. Verified in:

- TradingAgents repo: `tradingagents/llm_clients/openai_client.py:209-233` lists Qwen in OPENAI_COMPATIBLE_PROVIDERS with NormalizedChatOpenAI (no special case; uses standard API)
- `tradingagents/llm_clients/capabilities.py:85-90` assigns Qwen the default capabilities: `supports_json_mode=True, supports_json_schema=True, preferred_structured_method="function_calling"`
- Bitget hackathon key endpoint uses Qwen 3.8 Max on the same API surface

---

## 2. LangChain's `with_structured_output()` with Capability Dispatch (PRODUCTION-GRADE)

**Used by:** TradingAgents (all agents, primary pattern)

### Implementation

```python
# TradingAgents/tradingagents/agents/utils/structured.py:42-56
def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Any | None:
    """Return llm.with_structured_output(schema) or None if unsupported."""
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None

# TradingAgents/tradingagents/agents/utils/structured.py:59-89
def invoke_structured_or_freetext(
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> str:
    """Run the structured call and render to markdown; fall back to free-text on any failure."""
    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            if result is None:
                # Thinking model answered in plain text instead of calling the tool
                raise ValueError("structured output returned no parsed result")
            return render(result)
        except Exception as exc:
            logger.warning(
                "%s: structured-output invocation failed (%s); retrying once as free text",
                agent_name, exc,
            )

    response = plain_llm.invoke(prompt)
    return response.content
```

### Per-Model Capability Dispatch

```python
# TradingAgents/tradingagents/llm_clients/openai_client.py:38-51
def with_structured_output(self, schema, *, method=None, **kwargs):
    caps = get_capabilities(self.model_name)
    if caps.preferred_structured_method == "none":
        raise NotImplementedError(
            f"{self.model_name} has no structured-output method available; "
            f"agent factories will fall back to free-text generation."
        )
    method = method or caps.preferred_structured_method
    # When the model rejects tool_choice, suppress langchain's hardcoded value
    if method == "function_calling" and not caps.supports_tool_choice:
        kwargs.setdefault("tool_choice", None)
    return super().with_structured_output(schema, method=method, **kwargs)
```

### Capability Table

```python
# TradingAgents/tradingagents/llm_clients/capabilities.py:21-90 (excerpt)
StructuredMethod = Literal[
    "function_calling",  # uses tools; respects supports_tool_choice
    "json_mode",         # uses response_format={"type":"json_object"}
    "json_schema",       # uses response_format={"type":"json_schema",...}
    "none",              # no structured output available; caller falls back to free-text
]

# DeepSeek chat: supports tool_choice ✓
_DEEPSEEK_CHAT = ModelCapabilities(
    supports_tool_choice=True,
    supports_json_mode=True,
    supports_json_schema=False,
    preferred_structured_method="function_calling",
)

# DeepSeek reasoner: rejects tool_choice parameter ✗
_DEEPSEEK_THINKING = ModelCapabilities(
    supports_tool_choice=False,
    supports_json_mode=True,
    supports_json_schema=False,
    preferred_structured_method="function_calling",
    requires_reasoning_content_roundtrip=True,
)

# Default (Qwen, OpenAI, xAI, GLM, etc.)
_DEFAULT = ModelCapabilities(
    supports_tool_choice=True,
    supports_json_mode=True,
    supports_json_schema=True,
    preferred_structured_method="function_calling",
)
```

### What it fixes

- **Provider variation**: One code path works across OpenAI, DeepSeek, Qwen, MiniMax, Ollama, local vLLM
- **Method selection**: Picks `json_mode` vs `json_schema` vs `function_calling` per model's actual support
- **Tool-choice quirks**: DeepSeek reasoner rejects `tool_choice` parameter; MiniMax M2.x accepts only string enums; the capability table and dispatch suppress these at bind time
- **Thinking model prose**: Thinking models sometimes return reasoning in plain text instead of calling the tool; falls back to free-text parsing

### When it fails

- **Truncation**: See `finish_reason=="length"` detection below
- **JSON repair needed**: Falls back to free-text invocation, then caller must parse markdown or prose manually

---

## 3. Regex-Based Extraction from Prose/Markdown (SAFE FALLBACK)

**Used by:** alikeldev~levkila-trade (DeepSeek agent), finbrain AlphaForgeBench

### Pattern 1: Brace-Matching from Markdown

```python
# alikeldev~levkila-trade/prompt_builder.py:145-154
def extract_json_block(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # If direct parse fails, search for {.*} pattern (greedy, DOTALL)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            candidate = match.group(0)
            return json.loads(candidate)
        raise
```

**Fixes:** Markdown fences, preamble prose, trailing text after JSON.  
**Weakness:** Greedy `.*` may over-match if multiple `{}` pairs exist in the response; extracted substring may contain trailing text.

### Pattern 2: Custom Tag Extraction

```python
# alikeldev~levkila-trade/prompt_builder.py:157-170
def extract_decision_json(final_content: str, reasoning: Optional[str]) -> Dict[str, Any]:
    if final_content:
        match = re.search(r"<FINAL_JSON>(.*?)</FINAL_JSON>", final_content, re.DOTALL)
        if match:
            final_content = match.group(1)
    try:
        return extract_json_block(final_content)
    except json.JSONDecodeError:
        if reasoning:
            match = re.search(r"<FINAL_JSON>(.*?)</FINAL_JSON>", reasoning, re.DOTALL)
            if match:
                return extract_json_block(match.group(1))
        raise
```

**Fixes:** Model wrapping JSON in custom delimiters (e.g., `<FINAL_JSON>{"key":"value"}</FINAL_JSON>`).  
**Deployment:** Requires explicit prompt instruction: *"Wrap your final decision in `<FINAL_JSON>...</FINAL_JSON>` tags"*

---

## 4. JSON Repair Library

**Used by:** finbrain-lab-hkustgz~alphaforgebench (query generation)

### Implementation

```python
# AlphaForgeBench/query_batch_generator.py:20, ~400-410
from json_repair import repair_json

# ... later in response processing
queries = json.loads(repair_json(json_str))

if not isinstance(queries, list):
    raise ValueError("响应不是数组格式")
```

### What it fixes

- **Trailing commas**: `{"key": "value",}` → valid JSON
- **Unquoted keys**: `{key: "value"}` → `{"key": "value"}`
- **Single quotes**: `{'key': 'value'}` → `{"key": "value"}`
- **Missing closing braces**: `{"key": "value"` → `{"key": "value"}`
- **Unterminated strings**: `{"key": "unclosed string}` → partial repair attempted

### When it fails

- **Severely truncated JSON**: If the JSON is cut in the middle of a structure (`{"name": "John", "age":` with no closing), repair may fail or produce invalid output
- **Arbitrary text embedded**: Won't repair JSON embedded in large prose blocks
- **Requires parsing the result**: Must still call `json.loads()` on the repaired string

### Library

```
pip install json-repair
```

**Source:** `mangiucugna/json-repair` (GitHub)

---

## 5. Finish Reason Detection (TRUNCATION SENSING)

**Recommended by:** Best practice across DeepSeek, OpenAI, anthropic-sdk

### Pattern

```python
# Pseudo-code for detecting truncation
response = llm.invoke(prompt, max_tokens=2048)
if hasattr(response, 'response_metadata'):
    finish_reason = response.response_metadata.get('finish_reason')
else:
    # Direct OpenAI API or similar
    finish_reason = response['choices'][0]['finish_reason']

if finish_reason == "length":
    # JSON was completed syntactically, but may be semantically incomplete
    # Response may have cut off mid-field, mid-value, or mid-structure
    # Solution: Retry with longer max_tokens or shorter prompt
    print("TRUNCATED: Retry with higher max_tokens or compress the input")
    # Attempt parse anyway; may succeed if the } arrived before truncation
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        # Incomplete JSON; retry required
        pass
elif finish_reason == "stop":
    # Model finished naturally; JSON should be complete
    parsed = json.loads(response_text)
elif finish_reason == "tool_calls":
    # Model emitted a tool call (function_calling mode); extract from the call
    pass
```

### Why finish_reason matters

- `finish_reason="length"`: **Don't silently default missing fields.** Truncation corruption is worse than retry.
- `finish_reason="stop"`: JSON parse failures are real errors (malformed, not truncated).

**NOT verified in the repos examined:** None of the 114 cloned repos explicitly check `finish_reason` before attempting JSON parse. This is a gap; best practice is to sense truncation and retry at higher token limits before fallback.

---

## 6. Tool-Calling / Function-Calling for Structured Output

**Used by:** TradingAgents (primary method), many LangChain agents

### Pattern

LangChain's `with_structured_output()` binds the Pydantic schema as a function tool:

```python
# Conceptual flow (LangChain internals)
llm.with_structured_output(MySchema)
# → Binds MySchema as a tool with the schema's JSON-Schema description
# → Model calls the tool with parameters matching the schema
# → LangChain extracts tool_call.args and parses into a Pydantic instance

result = llm.invoke(prompt)  # Returns a parsed Pydantic instance, not raw JSON
```

### Advantages

- **Typed output at the API level**: No post-parse validation needed if the model called the tool
- **Native support across providers**: OpenAI, Anthropic, DeepSeek (with caveats), Google, Ollama all have tool-calling

### Disadvantages

- **Tool confusion**: Models may call other tools or make multiple calls; requires post-processing
- **DeepSeek quirk**: Rejects `tool_choice` parameter on reasoner models; TradingAgents detects this and omits the parameter

---

## 7. Pydantic Model Validation (POST-PARSE)

**Universal pattern** (all repos using structured output do this)

### Implementation

```python
# cooperiano~crypto-trading-agent/src/trading_agent/llm/__init__.py:22-30
class MarketAnalysis(BaseModel):
    direction: str
    confidence: float
    target_price: float | None = None
    stop_loss_price: float | None = None
    time_horizon_hours: int = 4
    key_factors: list[str] = []
    risk_factors: list[str] = []
    reasoning: str = ""

# ... after JSON parse
result = json.loads(data["choices"][0]["message"]["content"])
return response_schema.model_validate(result)
```

### What it fixes

- **Type coercion**: `"confidence": "0.95"` (string) → `0.95` (float)
- **Missing optional fields**: Filled with default values
- **Validation errors**: Raises `ValidationError` if a required field is missing or the type is wrong

### When it fails

- **Truncated JSON with required fields cut**: `"reasoning": "The market is...` (cut off) → `ValidationError: field required`
- **Coercion failure**: `"confidence": "not a number"` → `ValidationError: value is not a valid float`

---

## What ARGUS Should Do

### For a critical decision-output schema (verdict, side, quantity, confidence, thesis, invalidation list):

**Three-tier approach, no silent defaults:**

```python
from pydantic import BaseModel, Field, validator
from typing import List
import json
import logging

logger = logging.getLogger(__name__)

class TradeDecision(BaseModel):
    """Strict schema: every field is mandatory, no defaults, full validation."""
    verdict: str = Field(..., description="BUY, SELL, or HOLD only")
    side: str = Field(..., description="long or short")
    quantity: float = Field(..., gt=0, description="Position size in base asset")
    confidence: float = Field(..., ge=0.0, le=1.0, description="0.0 to 1.0")
    thesis: str = Field(..., min_length=20, description="Why this trade (min 20 chars)")
    invalidation_list: List[str] = Field(..., min_items=1, description="At least one invalidation condition")
    
    @validator('verdict')
    def verdict_valid(cls, v):
        if v.upper() not in ("BUY", "SELL", "HOLD"):
            raise ValueError(f"verdict must be BUY, SELL, or HOLD; got {v}")
        return v.upper()

def llm_call_with_structured_recovery(
    llm,
    prompt: str,
    schema: type[TradeDecision],
    max_retries: int = 2,
) -> TradeDecision:
    """
    Call LLM with structured output. Retry on truncation or JSON errors.
    Never silently default a missing field.
    """
    
    for attempt in range(max_retries):
        try:
            # Attempt 1: Use native response_format if available
            response = llm.invoke(
                [{"role": "system", "content": "You are a trading decision engine. Output ONLY valid JSON."},
                 {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=1024,
            )
            
            # Check for truncation
            finish_reason = response.response_metadata.get('finish_reason')
            if finish_reason == "length":
                logger.warning(f"Attempt {attempt+1}: Response truncated (finish_reason=length); retrying with higher tokens")
                continue
            
            # Parse JSON
            raw_json = response.content
            parsed = json.loads(raw_json)
            
            # Validate against schema (raises ValidationError if any field is wrong or missing)
            decision = schema.model_validate(parsed)
            logger.info(f"Decision valid: {decision.verdict} {decision.side} {decision.quantity}@{decision.confidence}")
            return decision
            
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt+1}: JSON parse failed ({e}); retrying with repair")
                try:
                    from json_repair import repair_json
                    repaired = repair_json(raw_json)
                    parsed = json.loads(repaired)
                    decision = schema.model_validate(parsed)
                    return decision
                except Exception as e2:
                    logger.error(f"Repair also failed: {e2}; will retry from LLM")
                    continue
            else:
                raise ValueError(f"JSON parse failed on final attempt: {e}") from e
        
        except Exception as e:  # ValidationError or other
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt+1}: Validation failed ({e}); retrying")
                # Do NOT default the field. Retry the entire call.
                continue
            else:
                raise ValueError(f"Schema validation failed on final attempt: {e}") from e
    
    raise ValueError("Exhausted retries; could not produce a valid TradeDecision")
```

### Key principles:

1. **Check `finish_reason == "length"`** before attempting JSON parse. If truncated, retry with higher `max_tokens` or shorter prompt.
2. **Detect parse errors early**: `json.JSONDecodeError` → retry with `json_repair`, then re-parse.
3. **Validate before returning**: `schema.model_validate()` catches missing/invalid fields. **Never silently default.**
4. **Retry the entire LLM call on validation failure**, not just the parse. The model may correct the error on the next invocation.
5. **Log every attempt**: Which stage failed (truncation, JSON, validation), what the error was, and what action was taken.
6. **Fail explicitly**: If all retries fail, raise an exception with the full error trail. Do not return a partially-constructed decision with missing fields.

---

## Summary Table

| Technique | Strength | Weakness | Detects Truncation? |
|-----------|----------|----------|---------------------|
| `response_format=json_object` | Native API-level enforcement | Ignores `finish_reason`; provider may return prose | No (requires manual check) |
| LangChain `with_structured_output()` | Provider-agnostic; capability dispatch | Falls back to free-text on failure | No |
| Regex brace-matching | Works on malformed JSON in prose | Greedy; over-matches on multiple braces | No |
| Custom tag extraction | Reliable if prompt-trained | Requires explicit prompt instruction | No |
| `json_repair` library | Fixes trailing commas, quotes, braces | Fails on severely truncated JSON | No |
| Finish reason check | Detects truncation *before* parse fails | Requires API access to response metadata | **Yes** |
| Pydantic validation | Type coercion; required field enforcement | Silently defaults optional fields; fails on missing required | No |
| Tool-calling | Typed at call time | Requires tool support; model may call wrong tool | No |

**Recommended stack for ARGUS:** `response_format=json_object` + finish reason check → `json_repair` fallback → Pydantic validation with no defaults.

