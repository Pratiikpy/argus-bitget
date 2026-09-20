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
    MAX_COMPLETION_TOKENS,
    BudgetExhausted,
    Completion,
    QwenClient,
    QwenError,
    TokenBudget,
    Usage,
    _parse,
    _strip_fences,
    extract_json_object,
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


class TestJsonObjectExtraction:
    """The outermost balanced object, found without a regex."""

    def test_clean_json_is_returned_unchanged(self) -> None:
        assert extract_json_object('{"a": 1}') == '{"a": 1}'

    def test_json_wrapped_in_prose_is_recovered(self) -> None:
        got = extract_json_object('Here is my analysis: {"a": 1} Hope that helps!')
        assert got == '{"a": 1}'

    def test_nested_objects_are_not_cut_at_the_first_brace(self) -> None:
        assert extract_json_object('x {"a": {"b": 2}} y') == '{"a": {"b": 2}}'

    def test_a_brace_inside_a_string_does_not_end_the_object(self) -> None:
        raw = '{"thesis": "a } inside prose", "x": 1}'
        assert extract_json_object("noise " + raw) == raw

    def test_an_escaped_quote_does_not_flip_string_state(self) -> None:
        raw = '{"t": "he said \\" and } too", "y": 2}'
        assert extract_json_object(raw) == raw

    def test_fences_are_removed_before_matching(self) -> None:
        assert extract_json_object('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_text_with_no_object_is_handed_back_for_the_real_error(self) -> None:
        """Returning the original keeps the caller's parse error truthful."""
        assert extract_json_object("no object here") == "no object here"

    def test_an_unbalanced_object_is_not_silently_closed(self) -> None:
        """A truncated object must stay truncated: closing it here would invent content."""
        assert extract_json_object('{"a": 1, "b": "unterm') == '{"a": 1, "b": "unterm'


class TestTruncationIsDetectedFromTheWire:
    """`finish_reason` says the answer was cut off. A parse error only says it did not parse."""

    def _client(self, monkeypatch, replies):  # type: ignore[no-untyped-def]
        monkeypatch.setenv("BITGET_QWEN_API_KEY", "k")
        monkeypatch.setenv("BITGET_QWEN_BASE_URL", "https://example.invalid/v1")
        client = QwenClient()
        seen: list[dict] = []

        def fake_post(payload):  # type: ignore[no-untyped-def]
            seen.append(payload)
            return replies[min(len(seen) - 1, len(replies) - 1)]

        monkeypatch.setattr(client, "_post", fake_post)
        return client, seen

    def test_a_cut_off_answer_retries_with_a_bigger_budget(self, monkeypatch) -> None:
        cut = Completion('{"verdict": "tr', "", Usage(10, 300, 0, 310), "length")
        whole = Completion('{"verdict": "no_trade"}', "", Usage(10, 50, 0, 60), "stop")
        client, seen = self._client(monkeypatch, [cut, whole])
        got = client.complete_json([{"role": "user", "content": "q"}],
                                   required_keys=("verdict",), max_tokens=300)
        assert got == {"verdict": "no_trade"}
        assert seen[1]["max_tokens"] > seen[0]["max_tokens"]

    def test_the_truncated_text_is_not_fed_back_into_the_prompt(self, monkeypatch) -> None:
        """Appending the cut-off answer is what makes the retry larger than the attempt that
        already did not fit. Every repository surveyed does exactly that."""
        cut = Completion('{"verdict": "tr', "", Usage(10, 300, 0, 310), "length")
        whole = Completion('{"verdict": "no_trade"}', "", Usage(10, 50, 0, 60), "stop")
        client, seen = self._client(monkeypatch, [cut, whole])
        client.complete_json([{"role": "user", "content": "q"}],
                             required_keys=("verdict",), max_tokens=300)
        assert len(seen[1]["messages"]) == 1
        assert all(m.get("role") != "assistant" for m in seen[1]["messages"])

    def test_a_truncated_answer_is_never_repaired_into_a_decision(self, monkeypatch) -> None:
        """A completed thesis the model never wrote is worse than no decision."""
        cut = Completion('{"verdict": "trade", "thesis": "NVDA looks',
                         "", Usage(10, 8192, 0, 8202), "length")
        client, _ = self._client(monkeypatch, [cut])
        with pytest.raises(QwenError, match="not repaired"):
            client.complete_json([{"role": "user", "content": "q"}],
                                 required_keys=("verdict", "thesis"),
                                 max_tokens=MAX_COMPLETION_TOKENS)

    def test_growth_stops_at_the_ceiling_and_says_so(self, monkeypatch) -> None:
        cut = Completion('{"a": "x', "", Usage(10, 8192, 0, 8202), "length")
        client, seen = self._client(monkeypatch, [cut])
        with pytest.raises(QwenError, match="ceiling"):
            client.complete_json([{"role": "user", "content": "q"}],
                                 required_keys=("a",), max_tokens=MAX_COMPLETION_TOKENS)
        assert len(seen) == 1

    def test_empty_content_with_reasoning_is_still_treated_as_truncation(
        self, monkeypatch
    ) -> None:
        """Measured on DeepSeek-via-NIM: the budget went entirely on reasoning."""
        spent = Completion("", "long reasoning", Usage(10, 900, 900, 910), "stop")
        whole = Completion('{"verdict": "no_trade"}', "", Usage(10, 50, 0, 60), "stop")
        client, seen = self._client(monkeypatch, [spent, whole])
        got = client.complete_json([{"role": "user", "content": "q"}],
                                   required_keys=("verdict",), max_tokens=300)
        assert got == {"verdict": "no_trade"}
        assert seen[1]["max_tokens"] > seen[0]["max_tokens"]

    def test_a_complete_but_malformed_answer_does_feed_the_error_back(self, monkeypatch) -> None:
        """The opposite branch: the model finished and got it wrong, so tell it what was wrong."""
        bad = Completion("not json at all", "", Usage(10, 20, 0, 30), "stop")
        whole = Completion('{"verdict": "no_trade"}', "", Usage(10, 50, 0, 60), "stop")
        client, seen = self._client(monkeypatch, [bad, whole])
        client.complete_json([{"role": "user", "content": "q"}],
                             required_keys=("verdict",), max_tokens=300)
        assert len(seen[1]["messages"]) == 3
        assert "failed validation" in seen[1]["messages"][-1]["content"]
        assert seen[1]["max_tokens"] == seen[0]["max_tokens"]

    def test_a_missing_required_key_is_never_defaulted(self, monkeypatch) -> None:
        partial = Completion('{"verdict": "trade"}', "", Usage(10, 20, 0, 30), "stop")
        client, _ = self._client(monkeypatch, [partial])
        with pytest.raises(QwenError, match="missing required keys"):
            client.complete_json([{"role": "user", "content": "q"}],
                                 required_keys=("verdict", "quantity"), max_tokens=300)

    def test_a_json_array_is_not_accepted_as_a_decision(self, monkeypatch) -> None:
        arr = Completion('[{"verdict": "trade"}]', "", Usage(10, 20, 0, 30), "stop")
        client, _ = self._client(monkeypatch, [arr])
        with pytest.raises(QwenError, match="expected a JSON object"):
            client.complete_json([{"role": "user", "content": "q"}],
                                 required_keys=("verdict",), max_tokens=300)

    def test_prose_wrapped_json_is_accepted_without_a_retry(self, monkeypatch) -> None:
        wrapped = Completion('Sure! {"verdict": "no_trade"} Done.', "",
                             Usage(10, 20, 0, 30), "stop")
        client, seen = self._client(monkeypatch, [wrapped])
        got = client.complete_json([{"role": "user", "content": "q"}],
                                   required_keys=("verdict",), max_tokens=300)
        assert got == {"verdict": "no_trade"} and len(seen) == 1


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


class TestUnmeasuredSpendIsNotFreeSpend:
    """`usage_block.get("prompt_tokens", 0)` made a response with no `usage` block report **zero
    tokens spent**, and nothing distinguished that from a genuinely free call.

    On a finite hackathon key that drifts in the one direction a budget must never drift: the burn
    looks smaller than it is, and `eval/cyclecheck.cumulative_tokens` inherits the understatement.
    A measurement that fails toward *"cheaper than reality"* on a budget is the worst way for it to
    fail.
    """

    def test_a_venue_reported_zero_is_measured_but_not_a_spend(self) -> None:
        from argus.llm.qwen import Usage

        got = Usage(0, 0, 0, 0)
        assert got.reported is True, "the venue did answer"
        assert got.is_measured is False, "but zero total is not a measurement of spend"

    def test_an_absent_usage_block_is_unmeasured(self) -> None:
        from argus.llm.qwen import Usage

        got = Usage(0, 0, 0, 0, reported=False)
        assert got.reported is False
        assert got.is_measured is False

    def test_a_real_call_is_measured(self) -> None:
        from argus.llm.qwen import Usage

        assert Usage(100, 50, 20, 150).is_measured is True

    def test_the_budget_does_not_count_an_unreported_call_as_free(self) -> None:
        from argus.llm.qwen import TokenBudget, Usage

        budget = TokenBudget(limit=1_000)
        budget.record(Usage(10, 5, 2, 15))
        budget.record(Usage(0, 0, 0, 0, reported=False))
        budget.record(Usage(20, 10, 4, 30))
        assert budget.spent == 45
        assert budget.unreported_calls == 1, "the call the budget cannot see must still be counted"

    def test_no_estimate_is_invented_for_an_unreported_call(self) -> None:
        """The tokens really are unknown. Guessing a number here would put a fabricated figure into
        the one measurement that guards a finite key."""
        from argus.llm.qwen import TokenBudget, Usage

        budget = TokenBudget(limit=1_000)
        before = budget.spent
        budget.record(Usage(0, 0, 0, 0, reported=False))
        assert budget.spent == before, "spend must not move on an unmeasured call"

    def test_unreported_calls_start_at_zero(self) -> None:
        from argus.llm.qwen import TokenBudget

        assert TokenBudget(limit=10).unreported_calls == 0
