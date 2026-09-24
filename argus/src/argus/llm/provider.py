"""Provider abstraction — so the Observatory can put different models in the same seat.

ARGUS's Track 2 Open Theme claim is that it is a *neutral operating system for evaluating trading
intelligence*, not one more agent. That claim only holds if swapping the model is a one-line change
and every other input — data, tools, costs, constraints, portfolio, prompt — is provably identical.
This module is what makes that true.

Three providers are wired:

===========  ==============================  =========  ==========
Provider     Model                           Latency    Thinking
===========  ==============================  =========  ==========
Qwen         ``qwen3.8-max``                 ~110s      3 tiers
0G Qwen      ``qwen3.8-max``                 n/a        3 tiers
NVIDIA NIM   ``z-ai/glm-5.3`` / ``-flash``   n/a        on/off
===========  ==============================  =========  ==========

**Qwen** is the Bitget hackathon key, verified live 2026-09-12. **0G Qwen** is the same model
routed through 0G instead of Bitget — verified 2026-09-22 to authenticate; the account had
insufficient balance and was unfunded as of that date, so it did not answer a real completion.
**NVIDIA NIM** was verified live on 2026-09-12 against a different model
(``deepseek-v4-pro``, since retired from the catalog); the current models were checked
2026-09-22 and the key authenticates (``/v1/models`` answers instantly) but every
``/v1/chat/completions`` call timed out with zero bytes back across three real attempts (60s,
100s, 240s; both ``z-ai/glm-5.3`` and its ``-flash`` variant; streaming both on and off) — a
real, external, tested-not-assumed block on the inference endpoint specifically, not the key.

The providers disagree in ways that matter and are worth recording rather than smoothing over:

* **Reasoning control is spelled differently.** Qwen (and 0G, serving the same model) takes
  ``enable_thinking`` / ``reasoning_effort`` as top-level fields. NIM's chat models take
  ``chat_template_kwargs: {"thinking": false}``. Passing one to the other silently does nothing —
  which is exactly how a "fair" comparison quietly stops being fair.
* **Qwen bills reasoning separately** (``reasoning_tokens``); NIM reported none.
* **Qwen's gateway closes at 120s**, so long reasoning must stream. NIM answered a 2-token reply in
  85s on 2026-09-12, so its latency is a property of the endpoint rather than the work — though as
  of 2026-09-22 the endpoint does not answer chat completions at all, only ``/v1/models``.

A comparison that ignored those differences would be measuring plumbing, not intelligence.

**Fallback ordering — the project owner's explicit decision, 2026-09-22.** Two different orders
for two different purposes: testing conserves the metered Bitget key by trying free/other options
first; submission leads with the Bitget key because that is the one actually tied to the hackathon.
See :data:`FALLBACK_ORDER`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from argus.llm.base import ChatModel
from argus.llm.qwen import Completion, QwenClient, QwenError, Thinking, TokenBudget


class Provider(StrEnum):
    QWEN = "qwen"
    OG_QWEN = "og_qwen"
    DEEPSEEK_NIM = "deepseek_nim"


class Mode(StrEnum):
    """Which fallback order applies. See :data:`FALLBACK_ORDER`."""

    TESTING = "testing"
    SUBMISSION = "submission"


class NvidiaClient(QwenClient):
    """DeepSeek v4 Pro via NVIDIA NIM.

    Subclasses the Qwen client because the wire format is otherwise identical OpenAI shape — only
    the reasoning-control spelling differs, which is overridden below. Inheriting rather than
    duplicating keeps caching, budget enforcement, retry and streaming behaviour bit-identical
    across providers, which is a precondition for the comparison being fair.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("api_key", os.environ.get("NVIDIA_API_KEY", ""))
        kwargs.setdefault("base_url", os.environ.get("NVIDIA_BASE_URL", ""))
        kwargs.setdefault("model", os.environ.get("NVIDIA_MODEL", ""))
        super().__init__(**kwargs)

    def _apply_thinking(self, payload: dict[str, Any], thinking: Thinking) -> None:
        """NIM spells it ``chat_template_kwargs``. Qwen's top-level fields are silently ignored
        here, so sending them would produce a comparison that only *looked* controlled."""
        payload.pop("enable_thinking", None)
        payload.pop("reasoning_effort", None)
        payload["chat_template_kwargs"] = {"thinking": thinking is not Thinking.OFF}


class OgQwenClient(QwenClient):
    """The real ``qwen3.8-max`` model, routed through 0G's endpoint instead of Bitget's.

    No wire-format override needed, unlike :class:`NvidiaClient` — 0G serves the identical model
    behind an identical OpenAI-compatible shape (confirmed from 0G's own published request
    example, which matches Qwen's own field names exactly), so this class exists only to read the
    ``OG_QWEN_*`` environment variables instead of ``BITGET_QWEN_*``.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("api_key", os.environ.get("OG_QWEN_API_KEY", ""))
        kwargs.setdefault("base_url", os.environ.get("OG_QWEN_BASE_URL", ""))
        kwargs.setdefault("model", os.environ.get("OG_QWEN_MODEL", "qwen3.8-max"))
        super().__init__(**kwargs)


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    provider: Provider
    label: str
    default_thinking: Thinking
    stream_by_default: bool


SPECS: dict[Provider, ProviderSpec] = {
    Provider.QWEN: ProviderSpec(
        provider=Provider.QWEN,
        label="qwen3.8-max (Bitget)",
        default_thinking=Thinking.FULL,
        # Its gateway closes at 120s and a full-reasoning decision measured 112s.
        stream_by_default=True,
    ),
    Provider.OG_QWEN: ProviderSpec(
        provider=Provider.OG_QWEN,
        label="qwen3.8-max (0G)",
        default_thinking=Thinking.FULL,
        stream_by_default=True,
    ),
    Provider.DEEPSEEK_NIM: ProviderSpec(
        provider=Provider.DEEPSEEK_NIM,
        label="z-ai/glm-5.3 (NVIDIA NIM)",
        default_thinking=Thinking.FULL,
        stream_by_default=False,
    ),
}


def build(provider: Provider, *, budget_limit: int = 200_000, **kwargs: Any) -> ChatModel:
    """One line to swap the intelligence in the seat. Everything else stays fixed."""
    budget = TokenBudget(limit=budget_limit)
    if provider is Provider.QWEN:
        return QwenClient(budget=budget, **kwargs)
    if provider is Provider.OG_QWEN:
        return OgQwenClient(budget=budget, **kwargs)
    if provider is Provider.DEEPSEEK_NIM:
        return NvidiaClient(budget=budget, **kwargs)
    raise ValueError(f"unknown provider: {provider}")


def available() -> tuple[Provider, ...]:
    """Providers whose credentials are actually present.

    The Observatory reports which models it could *not* reach rather than quietly comparing a
    smaller field — a leaderboard missing a competitor with no explanation is misleading.
    """
    found: list[Provider] = []
    if os.environ.get("BITGET_QWEN_API_KEY"):
        found.append(Provider.QWEN)
    if os.environ.get("OG_QWEN_API_KEY"):
        found.append(Provider.OG_QWEN)
    if os.environ.get("NVIDIA_API_KEY"):
        found.append(Provider.DEEPSEEK_NIM)
    return tuple(found)


# ---------------------------------------------------------------------------------------------
# Fallback chain — the project owner's decision, 2026-09-22
# ---------------------------------------------------------------------------------------------

FALLBACK_ORDER: dict[Mode, tuple[Provider, ...]] = {
    # Testing conserves the metered Bitget key: try the free/other options first, and only spend
    # the hackathon balance if both alternatives fail.
    Mode.TESTING: (Provider.OG_QWEN, Provider.DEEPSEEK_NIM, Provider.QWEN),
    # Submission leads with the key actually tied to the hackathon.
    Mode.SUBMISSION: (Provider.QWEN, Provider.OG_QWEN, Provider.DEEPSEEK_NIM),
}


class AllProvidersFailed(QwenError):
    """Every provider in the fallback chain failed. Carries what each one said, never a key."""


@dataclass(frozen=True, slots=True)
class Attempt:
    """One provider's outcome within a fallback call — kept so a caller can see what was tried,
    not just what finally worked."""

    provider: Provider
    ok: bool
    detail: str


class FallbackClient:
    """Tries each provider in :data:`FALLBACK_ORDER` for ``mode``, in order, until one answers.

    Implements the same :class:`~argus.llm.base.ChatModel` surface the decision layer already
    depends on, so a caller that takes a `ChatModel` can take this in its place with no other
    change. A provider with no credentials present is skipped without being counted as a failure
    (:func:`available` decides that, not a raised exception); a provider that raises
    :class:`~argus.llm.qwen.QwenError` (covers auth, balance, timeout, malformed response — every
    failure mode `qwen.py` itself surfaces) is recorded and the chain moves to the next one. All
    of them failing raises :class:`AllProvidersFailed` naming every attempt, rather than only the
    last error, so a genuine outage is diagnosable from one exception instead of one grep through
    logs per provider.

    **``timeout``/``max_retries`` default far below `QwenClient`'s own 600s/3, deliberately.**
    Verified 2026-09-22 against NVIDIA NIM's real inference endpoint: at `QwenClient`'s own
    defaults, one broken provider in the chain would block for up to ~30 minutes (600s x 3
    attempts, with backoff) before falling through — defeating the entire purpose of a fallback,
    which exists to move on from a broken provider quickly. 60s x 2 attempts bounds a single
    broken provider to about two minutes before the chain moves on, while still covering Qwen's
    own measured LOW-thinking latency; FULL-thinking calls (measured ~110s) should request a
    longer `timeout` explicitly via `client_kwargs`, since the default here is sized for the
    common case, not the slowest one.
    """

    def __init__(
        self,
        mode: Mode,
        *,
        budget_limit: int = 200_000,
        timeout: float = 60.0,
        max_retries: int = 2,
        on_attempt: Callable[[Attempt], None] | None = None,
        client_kwargs: dict[Provider, dict[str, Any]] | None = None,
    ) -> None:
        self._order = FALLBACK_ORDER[mode]
        self._budget_limit = budget_limit
        self._timeout = timeout
        self._max_retries = max_retries
        self._on_attempt = on_attempt
        self._client_kwargs = client_kwargs or {}
        self.last_used: Provider | None = None
        self.attempts: list[Attempt] = []
        self._clients: dict[Provider, ChatModel] = {}

    def _client_for(self, provider: Provider) -> ChatModel:
        if provider not in self._clients:
            overrides = self._client_kwargs.get(provider, {})
            kwargs: dict[str, Any] = {"timeout": self._timeout, "max_retries": self._max_retries}
            kwargs.update(overrides)
            self._clients[provider] = build(provider, budget_limit=self._budget_limit, **kwargs)
        return self._clients[provider]

    @property
    def budget(self) -> Any:
        """The last provider that actually answered — the one whose spend matters right now.
        Before any call succeeds, the first available provider's, so the attribute is never
        undefined."""
        provider = self.last_used or next(
            (p for p in self._order if p in available()), self._order[0]
        )
        return self._client_for(provider).budget

    def _run(self, name: str, fn: Callable[[ChatModel], Any]) -> Any:
        self.attempts = []
        present = set(available())
        errors: list[str] = []
        for provider in self._order:
            if provider not in present:
                attempt = Attempt(provider, ok=False, detail="no credentials present")
                self.attempts.append(attempt)
                if self._on_attempt:
                    self._on_attempt(attempt)
                continue
            try:
                result = fn(self._client_for(provider))
            except QwenError as exc:
                detail = str(exc)
                attempt = Attempt(provider, ok=False, detail=detail)
                self.attempts.append(attempt)
                if self._on_attempt:
                    self._on_attempt(attempt)
                errors.append(f"{provider.value}: {detail}")
                continue
            attempt = Attempt(provider, ok=True, detail=f"{name} succeeded")
            self.attempts.append(attempt)
            if self._on_attempt:
                self._on_attempt(attempt)
            self.last_used = provider
            return result
        raise AllProvidersFailed(
            f"every provider failed for {name}: " + "; ".join(errors) if errors
            else f"no provider in the fallback chain has credentials present ({name})"
        )

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> Completion:
        result: Completion = self._run("complete", lambda c: c.complete(messages, **kwargs))
        return result

    def complete_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = self._run(
            "complete_json", lambda c: c.complete_json(messages, **kwargs)
        )
        return result


def build_fallback(
    mode: Mode,
    *,
    budget_limit: int = 200_000,
    timeout: float = 60.0,
    max_retries: int = 2,
    on_attempt: Callable[[Attempt], None] | None = None,
    client_kwargs: dict[Provider, dict[str, Any]] | None = None,
) -> FallbackClient:
    """The entry point: ``build_fallback(Mode.TESTING)`` during development,
    ``build_fallback(Mode.SUBMISSION)`` for whatever the submitted demo actually runs."""
    return FallbackClient(
        mode,
        budget_limit=budget_limit,
        timeout=timeout,
        max_retries=max_retries,
        on_attempt=on_attempt,
        client_kwargs=client_kwargs,
    )
