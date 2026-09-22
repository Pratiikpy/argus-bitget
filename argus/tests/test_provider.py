"""Tests for the provider fallback chain.

Fakes `provider.build` — the one seam `FallbackClient` calls through — rather than the network,
so these test the fallback LOGIC (order, skip-on-missing-credentials, fall-through-on-error,
all-fail raises) without re-verifying `QwenClient`'s own internals, which `test_qwen.py` already
covers. `available()` decides which providers have credentials at all; `_client_for` decides what
each one does when asked — the two are tested separately below for exactly that reason.
"""

from __future__ import annotations

from typing import Any

import pytest

from argus.llm.provider import (
    FALLBACK_ORDER,
    AllProvidersFailed,
    Attempt,
    FallbackClient,
    Mode,
    Provider,
    available,
    build_fallback,
)
from argus.llm.qwen import Completion, QwenError, TokenBudget, Usage


class _FakeClient:
    """A `ChatModel` that either always succeeds or always raises `QwenError`."""

    def __init__(self, *, fails: bool, detail: str = "boom") -> None:
        self.budget = TokenBudget(limit=1000)
        self._fails = fails
        self._detail = detail
        self.calls = 0

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> Completion:
        self.calls += 1
        if self._fails:
            raise QwenError(self._detail)
        return Completion("ok", "", Usage(1, 1, 0, 2), "stop")

    def complete_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        if self._fails:
            raise QwenError(self._detail)
        return {"verdict": "ok"}


def _set_all_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BITGET_QWEN_API_KEY", "k")
    monkeypatch.setenv("OG_QWEN_API_KEY", "k")
    monkeypatch.setenv("NVIDIA_API_KEY", "k")


class TestFallbackOrder:
    def test_testing_mode_tries_og_before_the_metered_bitget_key(self) -> None:
        """Conserve the metered key: free/other options first during development."""
        order = FALLBACK_ORDER[Mode.TESTING]
        assert order.index(Provider.OG_QWEN) < order.index(Provider.QWEN)
        assert order[-1] == Provider.QWEN

    def test_submission_mode_leads_with_the_bitget_key(self) -> None:
        """The key actually tied to the hackathon goes first for whatever is submitted."""
        assert FALLBACK_ORDER[Mode.SUBMISSION][0] == Provider.QWEN

    def test_both_modes_name_all_three_providers_exactly_once(self) -> None:
        for mode in Mode:
            assert sorted(FALLBACK_ORDER[mode]) == sorted(Provider)
            assert len(set(FALLBACK_ORDER[mode])) == 3


class TestAvailable:
    def test_no_credentials_means_nothing_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("BITGET_QWEN_API_KEY", "OG_QWEN_API_KEY", "NVIDIA_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        assert available() == ()

    def test_each_credential_independently_toggles_its_provider(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        for var in ("BITGET_QWEN_API_KEY", "OG_QWEN_API_KEY", "NVIDIA_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("OG_QWEN_API_KEY", "k")
        assert available() == (Provider.OG_QWEN,)


class TestFallbackClientFallsThrough:
    def test_first_provider_success_never_touches_the_rest(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _set_all_credentials(monkeypatch)
        built: dict[Provider, _FakeClient] = {
            Provider.OG_QWEN: _FakeClient(fails=False),
            Provider.DEEPSEEK_NIM: _FakeClient(fails=True),
            Provider.QWEN: _FakeClient(fails=True),
        }
        monkeypatch.setattr(
            "argus.llm.provider.build", lambda provider, **kw: built[provider]
        )
        client = FallbackClient(Mode.TESTING)
        result = client.complete([{"role": "user", "content": "hi"}])
        assert result.content == "ok"
        assert client.last_used is Provider.OG_QWEN
        assert built[Provider.DEEPSEEK_NIM].calls == 0
        assert built[Provider.QWEN].calls == 0

    def test_first_two_failures_fall_through_to_the_third(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The exact scenario verified live 2026-09-22: 0G insufficient-balance, NVIDIA timeout,
        Bitget Qwen succeeds."""
        _set_all_credentials(monkeypatch)
        built: dict[Provider, _FakeClient] = {
            Provider.OG_QWEN: _FakeClient(fails=True, detail="insufficient_balance"),
            Provider.DEEPSEEK_NIM: _FakeClient(fails=True, detail="timed out"),
            Provider.QWEN: _FakeClient(fails=False),
        }
        monkeypatch.setattr(
            "argus.llm.provider.build", lambda provider, **kw: built[provider]
        )
        client = FallbackClient(Mode.TESTING)
        result = client.complete([{"role": "user", "content": "hi"}])
        assert result.content == "ok"
        assert client.last_used is Provider.QWEN
        assert [a.provider for a in client.attempts] == [
            Provider.OG_QWEN, Provider.DEEPSEEK_NIM, Provider.QWEN,
        ]
        assert [a.ok for a in client.attempts] == [False, False, True]

    def test_a_provider_with_no_credentials_is_skipped_not_counted_as_a_failure(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("OG_QWEN_API_KEY", raising=False)
        monkeypatch.setenv("BITGET_QWEN_API_KEY", "k")
        monkeypatch.setenv("NVIDIA_API_KEY", "k")
        built: dict[Provider, _FakeClient] = {
            Provider.DEEPSEEK_NIM: _FakeClient(fails=False),
            Provider.QWEN: _FakeClient(fails=False),
        }
        monkeypatch.setattr(
            "argus.llm.provider.build", lambda provider, **kw: built[provider]
        )
        client = FallbackClient(Mode.TESTING)
        result = client.complete([{"role": "user", "content": "hi"}])
        assert result.content == "ok"
        skipped = next(a for a in client.attempts if a.provider is Provider.OG_QWEN)
        assert skipped.ok is False
        assert "no credentials" in skipped.detail

    def test_every_provider_failing_raises_all_providers_failed_naming_each(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _set_all_credentials(monkeypatch)
        built: dict[Provider, _FakeClient] = {
            Provider.OG_QWEN: _FakeClient(fails=True, detail="og-reason"),
            Provider.DEEPSEEK_NIM: _FakeClient(fails=True, detail="nvidia-reason"),
            Provider.QWEN: _FakeClient(fails=True, detail="qwen-reason"),
        }
        monkeypatch.setattr(
            "argus.llm.provider.build", lambda provider, **kw: built[provider]
        )
        client = FallbackClient(Mode.TESTING)
        with pytest.raises(AllProvidersFailed) as excinfo:
            client.complete([{"role": "user", "content": "hi"}])
        message = str(excinfo.value)
        assert "og-reason" in message
        assert "nvidia-reason" in message
        assert "qwen-reason" in message

    def test_complete_json_falls_through_the_same_way_as_complete(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _set_all_credentials(monkeypatch)
        built: dict[Provider, _FakeClient] = {
            Provider.OG_QWEN: _FakeClient(fails=True),
            Provider.DEEPSEEK_NIM: _FakeClient(fails=False),
            Provider.QWEN: _FakeClient(fails=False),
        }
        monkeypatch.setattr(
            "argus.llm.provider.build", lambda provider, **kw: built[provider]
        )
        client = FallbackClient(Mode.TESTING)
        result = client.complete_json([{"role": "user", "content": "hi"}])
        assert result == {"verdict": "ok"}
        assert client.last_used is Provider.DEEPSEEK_NIM

    def test_on_attempt_callback_fires_for_every_provider_tried(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _set_all_credentials(monkeypatch)
        built: dict[Provider, _FakeClient] = {
            Provider.OG_QWEN: _FakeClient(fails=True),
            Provider.DEEPSEEK_NIM: _FakeClient(fails=True),
            Provider.QWEN: _FakeClient(fails=False),
        }
        monkeypatch.setattr(
            "argus.llm.provider.build", lambda provider, **kw: built[provider]
        )
        seen: list[Attempt] = []
        client = FallbackClient(Mode.TESTING, on_attempt=seen.append)
        client.complete([{"role": "user", "content": "hi"}])
        assert [a.provider for a in seen] == [
            Provider.OG_QWEN, Provider.DEEPSEEK_NIM, Provider.QWEN,
        ]

    def test_budget_reports_the_provider_that_actually_answered(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _set_all_credentials(monkeypatch)
        og = _FakeClient(fails=True)
        qwen = _FakeClient(fails=False)
        built: dict[Provider, _FakeClient] = {
            Provider.OG_QWEN: og,
            Provider.DEEPSEEK_NIM: _FakeClient(fails=True),
            Provider.QWEN: qwen,
        }
        monkeypatch.setattr(
            "argus.llm.provider.build", lambda provider, **kw: built[provider]
        )
        client = FallbackClient(Mode.TESTING)
        client.complete([{"role": "user", "content": "hi"}])
        assert client.budget is qwen.budget


class TestBuildFallback:
    def test_returns_a_fallback_client_for_the_requested_mode(self) -> None:
        client = build_fallback(Mode.SUBMISSION)
        assert isinstance(client, FallbackClient)
        assert client._order == FALLBACK_ORDER[Mode.SUBMISSION]

    def test_short_timeout_default_bounds_a_hanging_provider(self) -> None:
        """Verified live 2026-09-22: at QwenClient's own 600s/3-retry defaults, one broken
        provider would block the whole chain for up to ~30 minutes. The default here must stay
        far below that or the fallback defeats its own purpose."""
        client = build_fallback(Mode.TESTING)
        assert client._timeout <= 120.0
        assert client._max_retries <= 3
