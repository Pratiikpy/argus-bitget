"""Provider abstraction — so the Observatory can put different models in the same seat.

ARGUS's Track 2 Open Theme claim is that it is a *neutral operating system for evaluating trading
intelligence*, not one more agent. That claim only holds if swapping the model is a one-line change
and every other input — data, tools, costs, constraints, portfolio, prompt — is provably identical.
This module is what makes that true.

Two providers are wired and both verified live on 2026-09-12:

===========  ==============================  =========  ==========  ==========================
Provider     Model                           Latency    Thinking    Note
===========  ==============================  =========  ==========  ==========================
Qwen         ``qwen3.8-max``                 ~110s      3 tiers     Bitget hackathon key
NVIDIA NIM   ``deepseek-ai/deepseek-v4-pro`` ~90s       on/off      free tier
===========  ==============================  =========  ==========  ==========================

The two disagree in ways that matter and are worth recording rather than smoothing over:

* **Reasoning control is spelled differently.** Qwen takes ``enable_thinking`` /
  ``reasoning_effort`` as top-level fields. DeepSeek-on-NIM takes
  ``chat_template_kwargs: {"thinking": false}``. Passing one to the other silently does nothing —
  which is exactly how a "fair" comparison quietly stops being fair.
* **Qwen bills reasoning separately** (``reasoning_tokens``); NIM reported none.
* **Qwen's gateway closes at 120s**, so long reasoning must stream. NIM answered a 2-token reply in
  85s, so its latency is a property of the endpoint rather than the work.

A comparison that ignored those differences would be measuring plumbing, not intelligence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from argus.llm.base import ChatModel
from argus.llm.qwen import QwenClient, Thinking, TokenBudget


class Provider(StrEnum):
    QWEN = "qwen"
    DEEPSEEK_NIM = "deepseek_nim"


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


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    provider: Provider
    label: str
    default_thinking: Thinking
    stream_by_default: bool


SPECS: dict[Provider, ProviderSpec] = {
    Provider.QWEN: ProviderSpec(
        provider=Provider.QWEN,
        label="qwen3.8-max",
        default_thinking=Thinking.FULL,
        # Its gateway closes at 120s and a full-reasoning decision measured 112s.
        stream_by_default=True,
    ),
    Provider.DEEPSEEK_NIM: ProviderSpec(
        provider=Provider.DEEPSEEK_NIM,
        label="deepseek-v4-pro",
        default_thinking=Thinking.FULL,
        stream_by_default=False,
    ),
}


def build(provider: Provider, *, budget_limit: int = 200_000, **kwargs: Any) -> ChatModel:
    """One line to swap the intelligence in the seat. Everything else stays fixed."""
    budget = TokenBudget(limit=budget_limit)
    if provider is Provider.QWEN:
        return QwenClient(budget=budget, **kwargs)
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
    if os.environ.get("NVIDIA_API_KEY"):
        found.append(Provider.DEEPSEEK_NIM)
    return tuple(found)
