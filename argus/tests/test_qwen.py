"""Qwen client tests.

Offline tests run always. Live tests run only when BITGET_QWEN_API_KEY is in the environment and
ARGUS_LIVE_LLM=1 is set, because the hackathon key has a finite balance and an accidental
test-suite loop against it is exactly the kind of waste CLAUDE.md warns about.

    set -a && . .secrets/qwen.env && set +a && ARGUS_LIVE_LLM=1 pytest tests/test_qwen.py
"""

from __future__ import annotations

import json
import os

import pytest

from argus.llm.qwen import (
    BudgetExhausted,
    Completion,
    QwenClient,
    QwenError,
    TokenBudget,
    Usage,
    _parse,
    _strip_fences,
)

LIVE = os.environ.get("ARGUS_LIVE_LLM") == "1" and bool(os.environ.get("BITGET_QWEN_API_KEY"))
live_only = pytest.mark.skipif(not LIVE, reason="set ARGUS_LIVE_LLM=1 and load .secrets/qwen.env")


# --- response parsing, against the real wire shape recorded on 2026-09-12 --------------------

REAL_RESPONSE = {
    "choices": [{
        "finish_reason": "stop",
        "index": 0,
        "message": {
            "content": "OK",
            "reasoning_content": "We need to respond to user. Need final exactly OK.",
            "role": "assistant",
        },
    }],
    "created": 1789196289,
    "id": "chatcmpl-c58829d6",
    "model": "qwen3.8-max",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 26,
        "completion_tokens_details": {"reasoning_tokens": 23, "text_tokens": 26},
        "prompt_tokens": 66,
        "prompt_tokens_details": {"cached_tokens": 0, "text_tokens": 66},
        "total_tokens": 92,
    },
}

REAL_TOOL_RESPONSE = {
    "choices": [{
        "finish_reason": "tool_calls",
        "index": 0,
        "message": {
            "content": "I'll look that up for you.\n\n",
            "reasoning_content": "The user asks for session state for rNVDA.",
            "role": "assistant",
            "tool_calls": [{
                "function": {"arguments": '{"symbol": "rNVDA"}', "name": "get_session_state"},
                "id": "call_13f2f852020d4010ac9c30d2",
                "index": 0,
                "type": "function",
            }],
        },
    }],
    "id": "chatcmpl-f4bb0911",
    "usage": {"completion_tokens": 40, "prompt_tokens": 100, "total_tokens": 140},
}


class TestParsing:
    def test_reasoning_is_kept_separate_from_the_answer(self) -> None:
        """Scratch work is not the answer and must never be parsed as one."""
        got = _parse(REAL_RESPONSE)
        assert got.content == "OK"
        assert "We need to respond" in got.reasoning
        assert "We need to respond" not in got.content

    def test_reasoning_tokens_are_counted(self) -> None:
        """The measured billing fact: 23 of 26 completion tokens on a one-word answer."""
        got = _parse(REAL_RESPONSE)
        assert got.usage.reasoning_tokens == 23
        assert got.usage.completion_tokens == 26
        assert got.usage.reasoning_share > 0.85

    def test_tool_calls_are_surfaced(self) -> None:
        got = _parse(REAL_TOOL_RESPONSE)
        assert got.wants_tool_call is True
        assert got.tool_calls[0]["function"]["name"] == "get_session_state"
        assert json.loads(got.tool_calls[0]["function"]["arguments"]) == {"symbol": "rNVDA"}

    def test_plain_completion_does_not_want_a_tool_call(self) -> None:
        assert _parse(REAL_RESPONSE).wants_tool_call is False

    def test_missing_usage_details_do_not_crash(self) -> None:
        """The tool response carried no completion_tokens_details. Absent fields must degrade."""
        got = _parse(REAL_TOOL_RESPONSE)
        assert got.usage.reasoning_tokens == 0
        assert got.usage.reasoning_share == 0.0

    def test_null_content_becomes_empty_string(self) -> None:
        """A pure tool call can return content: null."""
        data = json.loads(json.dumps(REAL_TOOL_RESPONSE))
        data["choices"][0]["message"]["content"] = None
        assert _parse(data).content == ""

    def test_malformed_response_raises_clearly(self) -> None:
        with pytest.raises(QwenError, match="unexpected response shape"):
            _parse({"choices": []})


class TestFenceStripping:
    def test_plain_json_untouched(self) -> None:
        assert _strip_fences('{"a": 1}') == '{"a": 1}'

    def test_fenced_json_is_unwrapped(self) -> None:
        assert json.loads(_strip_fences('```json\n{"a": 1}\n```')) == {"a": 1}

    def test_bare_fence_is_unwrapped(self) -> None:
        assert json.loads(_strip_fences('```\n{"a": 1}\n```')) == {"a": 1}


class TestBudget:
    def test_budget_stops_before_the_call_not_after(self) -> None:
        """A budget that notices after spending is not a budget."""
        b = TokenBudget(limit=1000, spent=990)
        with pytest.raises(BudgetExhausted, match="deliberately"):
            b.check(projected=50)

    def test_spending_accumulates(self) -> None:
        b = TokenBudget(limit=1000)
        b.record(Usage(66, 26, 23, 92))
        assert b.spent == 92
        assert b.remaining == 908

    def test_remaining_never_goes_negative(self) -> None:
        b = TokenBudget(limit=50)
        b.record(Usage(100, 100, 0, 200))
        assert b.remaining == 0


class TestCredentialHygiene:
    def test_missing_key_raises_without_inventing_a_default(self, monkeypatch) -> None:
        monkeypatch.delenv("BITGET_QWEN_API_KEY", raising=False)
        with pytest.raises(QwenError, match="never pass it as a literal"):
            QwenClient()

    def test_repr_does_not_contain_the_key(self, monkeypatch) -> None:
        monkeypatch.setenv("BITGET_QWEN_API_KEY", "super-secret-value")
        monkeypatch.setenv("BITGET_QWEN_BASE_URL", "https://example.invalid/v1")
        client = QwenClient()
        assert "super-secret-value" not in repr(client)
        assert "qwen" in repr(client).lower()

    def test_missing_base_url_raises(self, monkeypatch) -> None:
        monkeypatch.setenv("BITGET_QWEN_API_KEY", "k")
        monkeypatch.delenv("BITGET_QWEN_BASE_URL", raising=False)
        with pytest.raises(QwenError, match="BASE_URL"):
            QwenClient()


class TestCaching:
    def test_identical_requests_hit_the_cache(self, monkeypatch) -> None:
        """The key has a finite balance; re-asking a deterministic question is pure waste."""
        monkeypatch.setenv("BITGET_QWEN_API_KEY", "k")
        monkeypatch.setenv("BITGET_QWEN_BASE_URL", "https://example.invalid/v1")
        client = QwenClient()

        calls = {"n": 0}

        def fake_post(payload):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            return Completion("x", "r", Usage(1, 1, 0, 2), "stop")

        monkeypatch.setattr(client, "_post", fake_post)
        msgs = [{"role": "user", "content": "same question"}]
        client.complete(msgs)
        client.complete(msgs)

        assert calls["n"] == 1
        assert client.cache_hits == 1


# --- live tests, opt-in ----------------------------------------------------------------------

@live_only
class TestLive:
    def test_basic_completion(self) -> None:
        client = QwenClient(budget=TokenBudget(limit=20_000))
        got = client.complete(
            [{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=32
        )
        assert "OK" in got.content
        assert got.usage.total_tokens > 0

    def test_json_mode_returns_a_valid_decision_shape(self) -> None:
        client = QwenClient(budget=TokenBudget(limit=20_000))
        got = client.complete_json(
            [{
                "role": "user",
                "content": (
                    "You are a trading agent. The US market is shut and no hedge is placeable. "
                    'Return JSON with keys "verdict" and "confidence". '
                    "verdict must be one of TRADE, REDUCE, HEDGE, DELAY, NO_TRADE, "
                    "HUMAN_REVIEW, DATA_INSUFFICIENT."
                ),
            }],
            required_keys=("verdict", "confidence"),
            max_tokens=256,
        )
        assert got["verdict"] in {
            "TRADE", "REDUCE", "HEDGE", "DELAY", "NO_TRADE", "HUMAN_REVIEW", "DATA_INSUFFICIENT",
        }

    def test_tool_calling_round_trip(self) -> None:
        client = QwenClient(budget=TokenBudget(limit=20_000))
        got = client.complete(
            [{"role": "user", "content": "Get the session state for rNVDA. Use the tool."}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_session_state",
                    "description": "Session state for a symbol",
                    "parameters": {
                        "type": "object",
                        "properties": {"symbol": {"type": "string"}},
                        "required": ["symbol"],
                    },
                },
            }],
            max_tokens=256,
        )
        assert got.wants_tool_call
        assert got.tool_calls[0]["function"]["name"] == "get_session_state"
