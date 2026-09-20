"""The narrow interface the decision layer depends on.

Kept deliberately small and in its own module. The Meta-PM must be typed against *a model*, not
against one vendor's client, or the Observatory's model-swap claim is false at the type level —
and a wider interface would let provider-specific behaviour leak into the agent, which is the same
failure one layer down.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from argus.llm.qwen import Completion, Thinking


class ChatModel(Protocol):
    """What a decision-maker needs from a model. Nothing more."""

    budget: Any

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = ...,
        max_tokens: int = ...,
        json_mode: bool = ...,
        tools: list[dict[str, Any]] | None = ...,
        seed: int | None = ...,
        thinking: Thinking = ...,
        stream: bool | None = ...,
    ) -> Completion: ...

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        *,
        required_keys: tuple[str, ...] = ...,
        validate: Callable[[dict[str, Any]], str] | None = ...,
        max_tokens: int = ...,
        attempts: int = ...,
        thinking: Thinking = ...,
    ) -> dict[str, Any]: ...
    """``validate`` returns an empty string when the object is acceptable, or the complaint to feed
    back to the model. Presence of a key is not validity of its value, and the difference was
    measured: the decision prompt returned a *side* in the verdict field on roughly one call in
    five, and `required_keys` passed it."""
