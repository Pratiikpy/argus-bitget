"""Qwen 3.8 Max client — the seat the decision-maker sits in.

Every capability below was **verified by calling the live endpoint on 2026-09-12**, not assumed
from the provider being "OpenAI-compatible":

======================  ======  ==========================================================
Capability              Works   Evidence from the live response
======================  ======  ==========================================================
``/chat/completions``   yes     ``{"object": "chat.completion", "model": "qwen3.8-max"}``
``response_format``     yes     ``json_object`` returned exactly ``{"verdict": ...}``
tool calling            yes     ``finish_reason: "tool_calls"`` + standard ``tool_calls[]``
``reasoning_content``   yes     separate field alongside ``content``
======================  ======  ==========================================================

**The billing fact that shapes this module.** Qwen 3.8 Max is a reasoning model and returns
``reasoning_content`` as a distinct field. In the smallest possible probe — a six-word prompt
answered with the single token ``OK`` — usage reported ``completion_tokens: 26`` of which
``reasoning_tokens: 23``. Reasoning was **88% of the completion cost of a one-word answer.**

This is a hackathon key with a limited balance, so that ratio is a budget constraint, not trivia:

* every call's reasoning tokens are counted separately and reported (:class:`Usage`),
* a session-wide :class:`TokenBudget` can hard-stop before the key is exhausted,
* identical requests are cached, because re-asking a deterministic question is pure waste.

Credentials are read from the environment only. The key never appears in code, a default argument,
a log line, a repr, or an exception message.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Thinking(StrEnum):
    """Reasoning budget. All three tiers measured against the live endpoint on 2026-09-12.

    On one identical decision prompt:

    =========  ======================  ===========  ==========  ============================
    Tier       Request parameter       Completion   Reasoning   Use
    =========  ======================  ===========  ==========  ============================
    ``OFF``    ``enable_thinking:F``          116           0   cheap pre-filter / triage
    ``LOW``    ``reasoning_effort:low``       828         652   routine decisions
    ``FULL``   (default)                    4,264       4,043   the decision that matters
    =========  ======================  ===========  ==========  ============================

    **FULL is 37x the cost of OFF**, and on a hackathon key with a finite balance that ratio is the
    whole reason this enum exists. It is also atrx-demo's three-tier validation pattern — cheap
    pre-filter, full analysis, portfolio veto — arrived at from the same pressure.

    A caveat measured rather than assumed: at ``OFF`` the model returned ``"quantity": "reduced"``
    and ``"confidence": "high"`` where numbers were asked for. It is fine for triage and unfit for
    a decision that sizes a position.

    ``extra_body`` nesting does **not** work here (it is an SDK-side concept) — passing it left
    reasoning fully enabled at 2,457 tokens.
    """

    OFF = "off"
    LOW = "low"
    FULL = "full"

    def apply_to(self, payload: dict[str, Any]) -> None:
        if self is Thinking.OFF:
            payload["enable_thinking"] = False
        elif self is Thinking.LOW:
            payload["reasoning_effort"] = "low"
        # FULL: send nothing; the endpoint reasons by default.


class QwenError(RuntimeError):
    """A call failed. Never carries the API key, even when echoing server detail."""


class BudgetExhausted(QwenError):
    """The configured token budget is spent. Raised *before* the call, never after."""


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    cached_tokens: int = 0

    @property
    def reasoning_share(self) -> float:
        """Fraction of completion tokens spent thinking rather than answering.

        Worth watching: it ran at 0.88 on a one-word answer in our probe, so a prompt that invites
        deliberation is far more expensive than its output length suggests.
        """
        return self.reasoning_tokens / self.completion_tokens if self.completion_tokens else 0.0


@dataclass
class TokenBudget:
    """A hard ceiling for the session. The key has a finite balance and no overdraft."""

    limit: int
    spent: int = 0

    def check(self, projected: int = 0) -> None:
        if self.spent + projected >= self.limit:
            raise BudgetExhausted(
                f"token budget exhausted: {self.spent}/{self.limit} spent. "
                f"Raise the limit deliberately rather than by accident."
            )

    def record(self, usage: Usage) -> None:
        self.spent += usage.total_tokens

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.spent)


@dataclass(frozen=True, slots=True)
class Completion:
    """One model response, with thinking kept separate from the answer.

    ``reasoning`` is deliberately *not* concatenated into ``content``. It is the model's scratch
    work: useful for the decision ledger and for judging debate quality, but it is not the answer
    and must never be parsed as one.
    """

    content: str
    reasoning: str
    usage: Usage
    finish_reason: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw_id: str = ""

    @property
    def wants_tool_call(self) -> bool:
        return self.finish_reason == "tool_calls" and bool(self.tool_calls)


class QwenClient:
    """Minimal, dependency-free client for the Bitget hackathon Qwen endpoint.

    Deliberately uses ``urllib`` rather than the ``openai`` SDK: this is the one component every
    agent depends on, and a hard dependency on a large client library for four fields of JSON is a
    liability we do not need. The wire format was verified against the live endpoint.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        budget: TokenBudget | None = None,
        # 120s is the endpoint's own gateway timeout; matching it leaves no headroom for a
        # streamed full-reasoning call, which measured 112s. Read timeouts must outlast it.
        timeout: float = 600.0,
        max_retries: int = 3,
        cache: bool = True,
    ) -> None:
        self._api_key = api_key or os.environ.get("BITGET_QWEN_API_KEY", "")
        if not self._api_key:
            raise QwenError(
                "BITGET_QWEN_API_KEY is not set. Load it from .secrets/qwen.env into the "
                "environment; never pass it as a literal."
            )
        self._base_url = (base_url or os.environ.get("BITGET_QWEN_BASE_URL", "")).rstrip("/")
        self._model = model or os.environ.get("BITGET_QWEN_MODEL", "qwen3.8-max")
        if not self._base_url:
            raise QwenError("BITGET_QWEN_BASE_URL is not set")

        self.budget = budget
        self._timeout = timeout
        self._max_retries = max_retries
        self._cache: dict[str, Completion] | None = {} if cache else None
        self.calls = 0
        self.cache_hits = 0

    def __repr__(self) -> str:
        # Explicit: the default dataclass-ish repr would be fine today, but a future field holding
        # the key must never leak through a log line or a traceback.
        return f"QwenClient(model={self._model!r}, base_url={self._base_url!r})"

    def _apply_thinking(self, payload: dict[str, Any], thinking: Thinking) -> None:
        """How this provider spells reasoning control. Overridden per provider — the spellings
        are not interchangeable and passing the wrong one silently does nothing."""
        thinking.apply_to(payload)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
        seed: int | None = None,
        thinking: Thinking = Thinking.LOW,
        stream: bool | None = None,
    ) -> Completion:
        """One completion. ``temperature=0`` by default because a trading decision is not a place
        for sampling variance — and because decision-consistency-under-replay is a metric we are
        graded on.

        ``thinking`` defaults to :attr:`Thinking.LOW` rather than FULL. Measured: FULL costs 37x
        OFF and 5x LOW on an identical prompt, and a full-reasoning call took 112s — close enough
        to the endpoint's **120-second gateway timeout** that non-streamed FULL calls return
        HTTP 504. Choose FULL deliberately, for the decision that matters.

        ``stream`` defaults to on for FULL, because streaming is what keeps a long generation under
        the gateway timeout. Measured: the same prompt that 504'd unstreamed completed in 112s
        streamed.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        self._apply_thinking(payload, thinking)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = tools
        if seed is not None:
            payload["seed"] = seed

        use_stream = (thinking is Thinking.FULL) if stream is None else stream
        if use_stream:
            payload["stream"] = True

        key = self._cache_key(payload)
        if self._cache is not None and key in self._cache:
            self.cache_hits += 1
            return self._cache[key]

        if self.budget is not None:
            self.budget.check(max_tokens)

        result = self._post(payload)

        if self.budget is not None:
            self.budget.record(result.usage)
        if self._cache is not None:
            self._cache[key] = result
        return result

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        *,
        required_keys: tuple[str, ...] = (),
        max_tokens: int = 2048,
        attempts: int = 3,
        thinking: Thinking = Thinking.LOW,
    ) -> dict[str, Any]:
        """A completion parsed as JSON, retried with the parse error fed back to the model.

        ``json_object`` mode makes valid JSON very likely but not certain, and a decision schema
        has required fields that JSON validity alone does not guarantee. Retrying with the specific
        failure is markedly more effective than retrying blind.
        """
        convo = list(messages)
        last: str = ""
        for attempt in range(attempts):
            got = self.complete(
                convo,
                json_mode=True,
                max_tokens=max_tokens,
                thinking=thinking,
                # Vary the seed per attempt: an identical request would otherwise be served from
                # cache and every retry would reproduce the same failure.
                seed=attempt if attempt else None,
            )
            # Empty content with reasoning present means the token budget was spent thinking
            # before any answer was emitted. Measured on DeepSeek-via-NIM: 1,668 characters of
            # reasoning against a 900-token cap produced an empty string, which reads as a parse
            # failure and is really a truncation. Retrying at the same size would fail identically.
            if not got.content.strip() and got.reasoning.strip():
                max_tokens = min(max_tokens * 3, 8192)
                convo = list(messages)
                last = (
                    f"reasoning consumed the budget before any answer "
                    f"({got.usage.completion_tokens} tokens); retrying at max_tokens={max_tokens}"
                )
                continue

            try:
                parsed: dict[str, Any] = json.loads(_strip_fences(got.content))
            except json.JSONDecodeError as exc:
                last = f"invalid JSON ({exc})"
            else:
                missing = [k for k in required_keys if k not in parsed]
                if not missing:
                    return parsed
                last = f"missing required keys: {missing}"

            convo = [
                *messages,
                {"role": "assistant", "content": got.content},
                {"role": "user", "content": f"That response failed validation: {last}. "
                                            f"Return only valid JSON with all required keys."},
            ]
        raise QwenError(f"could not obtain valid JSON after {attempts} attempts — last: {last}")

    # --- internals -------------------------------------------------------------------------

    def _cache_key(self, payload: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _post(self, payload: dict[str, Any]) -> Completion:
        body = json.dumps(payload).encode()
        url = f"{self._base_url}/chat/completions"
        last_error = ""

        for attempt in range(self._max_retries):
            # Fixed https endpoint from config; no user-controlled scheme.
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    self.calls += 1
                    if payload.get("stream"):
                        return _parse_stream(resp)
                    return _parse(json.loads(resp.read().decode()))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                last_error = f"HTTP {exc.code}: {detail}"
                # 4xx other than rate-limiting will not fix themselves; fail fast.
                if exc.code not in (408, 429) and exc.code < 500:
                    raise QwenError(last_error) from None
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = f"transport: {exc}"

            if attempt < self._max_retries - 1:
                time.sleep(2.0 ** attempt)

        raise QwenError(f"failed after {self._max_retries} attempts — {last_error}")


def _strip_fences(text: str) -> str:
    """Remove markdown fences a model sometimes wraps JSON in despite json_object mode."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


def _parse_stream(resp: Any) -> Completion:
    """Assemble a Completion from a Server-Sent Events stream.

    Streaming exists here for one measured reason: the endpoint's gateway closes a non-streamed
    request at ~120s with HTTP 504, and a full-reasoning decision prompt took 112s. Streaming keeps
    bytes flowing so the gateway stays open.

    Wire shape verified live: each ``data:`` line carries a ``chat.completion.chunk`` whose
    ``choices[0].delta`` holds incremental ``content`` and ``reasoning_content``. The **final**
    chunk carries ``choices: []`` and the authoritative ``usage`` block, so usage must be taken
    from the last chunk rather than accumulated. The stream ends with ``data: [DONE]``.
    """
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    finish_reason = ""
    raw_id = ""
    usage_block: dict[str, Any] = {}

    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            break
        try:
            chunk = json.loads(body)
        except json.JSONDecodeError:
            continue  # a truncated keep-alive line is not fatal

        raw_id = chunk.get("id") or raw_id
        if chunk.get("usage"):
            usage_block = chunk["usage"]

        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
            if delta.get("tool_calls"):
                tool_calls.extend(delta["tool_calls"])
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    details = usage_block.get("completion_tokens_details", {}) or {}
    prompt_details = usage_block.get("prompt_tokens_details", {}) or {}
    return Completion(
        content="".join(content),
        reasoning="".join(reasoning),
        usage=Usage(
            prompt_tokens=usage_block.get("prompt_tokens", 0),
            completion_tokens=usage_block.get("completion_tokens", 0),
            reasoning_tokens=details.get("reasoning_tokens", 0),
            total_tokens=usage_block.get("total_tokens", 0),
            cached_tokens=prompt_details.get("cached_tokens", 0),
        ),
        finish_reason=finish_reason or "stop",
        tool_calls=tool_calls,
        raw_id=raw_id,
    )


def _parse(data: dict[str, Any]) -> Completion:
    try:
        choice = data["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError) as exc:
        raise QwenError(f"unexpected response shape: missing {exc}") from None

    raw_usage = data.get("usage", {})
    details = raw_usage.get("completion_tokens_details", {}) or {}
    prompt_details = raw_usage.get("prompt_tokens_details", {}) or {}

    usage = Usage(
        prompt_tokens=raw_usage.get("prompt_tokens", 0),
        completion_tokens=raw_usage.get("completion_tokens", 0),
        reasoning_tokens=details.get("reasoning_tokens", 0),
        total_tokens=raw_usage.get("total_tokens", 0),
        cached_tokens=prompt_details.get("cached_tokens", 0),
    )

    return Completion(
        content=message.get("content") or "",
        reasoning=message.get("reasoning_content") or "",
        usage=usage,
        finish_reason=choice.get("finish_reason", ""),
        tool_calls=message.get("tool_calls") or [],
        raw_id=data.get("id", ""),
    )
