"""The console's model client behind a circuit breaker (`lui/router.py`): a spent balance, a
revoked key, rate limiting or a gateway outage pauses the model so each question stops paying the
client's retry ladder, and the pause is visible on /status."""

from __future__ import annotations

from typing import Any

import pytest

from argus.llm.qwen import QwenError
from argus.lui import router


class _Inner:
    def __init__(self, failure: str | None) -> None:
        self.failure = failure
        self.calls = 0

    def complete_json(self, messages: list[dict[str, Any]], **_kw: Any) -> dict[str, Any]:
        self.calls += 1
        if self.failure:
            raise QwenError(self.failure)
        return {"ok": True}


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.parametrize("failure", ["HTTP 402: insufficient balance", "HTTP 401: ",
                                     "failed after 2 attempts — HTTP 503: gateway",
                                     "transport: <urlopen error timed out>"])
def test_a_lasting_failure_pauses_the_model(failure: str) -> None:
    inner, clock = _Inner(failure), _Clock()
    guarded = router.Breaker(inner, pause_s=600, clock=clock)
    with pytest.raises(QwenError):
        guarded.complete_json([])
    assert guarded.state()["paused"] is True and failure[:20] in guarded.state()["reason"]
    with pytest.raises(QwenError, match="model paused"):
        guarded.complete_json([])
    assert inner.calls == 1  # the second question cost no call and no wait
    clock.now += 601
    inner.failure = None
    assert guarded.complete_json([]) == {"ok": True}
    assert guarded.state()["paused"] is False and inner.calls == 2


def test_a_malformed_reply_does_not_pause() -> None:
    inner = _Inner("no JSON object in the reply")
    guarded = router.Breaker(inner, clock=_Clock())
    with pytest.raises(QwenError):
        guarded.complete_json([])
    assert guarded.state()["paused"] is False
    with pytest.raises(QwenError):
        guarded.complete_json([])
    assert inner.calls == 2


def test_other_attributes_pass_through() -> None:
    inner = _Inner(None)
    assert router.Breaker(inner).calls == 0


def test_the_console_seat_has_a_short_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def seat(**kwargs: Any) -> _Inner:
        seen.update(kwargs)
        return _Inner(None)

    from argus.llm import provider

    monkeypatch.setattr(provider, "seat", seat)
    built = router.build_router(budget_tokens=1000)
    assert isinstance(built, router.Breaker)
    assert seen == {"budget_limit": 1000, "timeout": router.CONSOLE_TIMEOUT_S,
                    "max_retries": router.CONSOLE_RETRIES}
