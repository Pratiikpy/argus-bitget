"""Telegram approvals: a real held decision, answered by button, and every way to forge a tap
refused (research/harvest/49-telegram-deep.md, decision 1, points a-f)."""

from __future__ import annotations

import threading
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from argus.decision.pause import PauseStore
from argus.eval.pausedrill import by_key, run_paused_half
from argus.lui import pause_bot
from argus.lui.pause_bot import (
    ApprovalConfigError,
    Approvals,
    announce,
    callback_data,
    handle_callback,
    keyboard,
)

REVIEWER, STRANGER = 111, 999
SECRET = "s" * 40


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARGUS_PAUSE_REVIEWERS", "alice,bob")
    monkeypatch.setenv(pause_bot.REVIEWERS_BY_ID_ENV, f"{REVIEWER}:alice,222:bob")
    monkeypatch.setenv(pause_bot.SECRET_ENV, SECRET)


def _held(tmp_path: Path) -> tuple[PauseStore, Any]:
    store = PauseStore(tmp_path)
    run_paused_half(by_key("underlying_halted/approve"), store, decision_id="d-1")
    [request] = store.pending()
    return store, request


def _tap(data: str, user: int = REVIEWER, qid: str = "q1") -> dict[str, Any]:
    return {
        "update_id": 1,
        "callback_query": {
            "id": qid,
            "data": data,
            "from": {"id": user},
            "message": {"message_id": 7, "chat": {"id": user}},
        },
    }


def _soon(request: Any) -> Any:
    """Inside the request's answer window: the drill's decision is dated in the past."""
    return request.as_of + timedelta(minutes=1)


def _approvals() -> Approvals:
    loaded = Approvals.from_env()
    assert loaded is not None
    return loaded


class TestConfiguration:
    def test_unconfigured_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(pause_bot.REVIEWERS_BY_ID_ENV)
        monkeypatch.delenv(pause_bot.SECRET_ENV)
        assert Approvals.from_env() is None

    def test_the_store_allowlist_is_mandatory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARGUS_PAUSE_REVIEWERS")
        with pytest.raises(ApprovalConfigError, match="ARGUS_PAUSE_REVIEWERS must be set"):
            Approvals.from_env()

    def test_a_mapped_name_off_the_allowlist_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(pause_bot.REVIEWERS_BY_ID_ENV, f"{REVIEWER}:alice,333:mallory")
        with pytest.raises(ApprovalConfigError, match="mallory"):
            Approvals.from_env()

    @pytest.mark.parametrize("missing", [pause_bot.REVIEWERS_BY_ID_ENV, pause_bot.SECRET_ENV])
    def test_half_configured_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, missing: str
    ) -> None:
        monkeypatch.delenv(missing)
        with pytest.raises(ApprovalConfigError, match="as well"):
            Approvals.from_env()

    def test_a_short_secret_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(pause_bot.SECRET_ENV, "short")
        with pytest.raises(ApprovalConfigError, match="32 characters"):
            Approvals.from_env()


class TestTheButtons:
    def test_every_payload_fits_telegrams_limit_whatever_the_request_id(
        self, tmp_path: Path
    ) -> None:
        store = PauseStore(tmp_path)
        run_paused_half(by_key("underlying_halted/approve"), store, decision_id="d" * 200)
        [request] = store.pending()
        for row in keyboard(_approvals(), request)["inline_keyboard"]:
            for button in row:
                assert len(button["callback_data"].encode()) <= 64

    def test_announce_reaches_only_registered_reviewers_once(self, tmp_path: Path) -> None:
        store, request = _held(tmp_path)
        sent: list[int] = []
        assert announce(
            store, _approvals(), lambda chat, text, markup: sent.append(chat), now=_soon(request)
        ) == [request.request_id]
        assert sorted(sent) == [REVIEWER, 222]
        assert announce(store, _approvals(), lambda *a: sent.append(0), now=_soon(request)) == []
        assert 0 not in sent

    def test_a_failed_send_is_retried_next_sweep(self, tmp_path: Path) -> None:
        store, request = _held(tmp_path)

        def fail(*_: Any) -> None:
            raise OSError("network")

        assert announce(store, _approvals(), fail, now=_soon(request)) == []
        assert announce(store, _approvals(), lambda *a: None, now=_soon(request)) == [
            request.request_id
        ]


class TestATap:
    def test_approve_is_recorded_through_the_store_with_the_tappers_name(
        self, tmp_path: Path
    ) -> None:
        store, request = _held(tmp_path)
        out = handle_callback(
            _tap(callback_data(_approvals(), request, "a")), store, _approvals(), now=_soon(request)
        )
        assert out is not None and out.toast.startswith("Recorded: approved")
        assert out.text is not None and "alice approved" in out.text
        answered = [e for e in store.events() if e["event"] == "answered"]
        assert len(answered) == 1 and answered[0]["reviewer"] == "alice"
        assert answered[0]["action"] == "approve"

    def test_half_size_records_half_the_proposal(self, tmp_path: Path) -> None:
        store, request = _held(tmp_path)
        handle_callback(
            _tap(callback_data(_approvals(), request, "h")), store, _approvals(), now=_soon(request)
        )
        [event] = [e for e in store.events() if e["event"] == "answered"]
        assert event["action"] == "modify_size"
        assert event["quantity"] == str(pause_bot.half_of(request))

    def test_a_stranger_cannot_answer_with_a_valid_button(self, tmp_path: Path) -> None:
        store, request = _held(tmp_path)
        out = handle_callback(
            _tap(callback_data(_approvals(), request, "a"), user=STRANGER),
            store,
            _approvals(),
            now=_soon(request),
        )
        assert out is not None and "not a reviewer" in out.toast
        assert store.status(request.request_id) == "paused"

    def test_a_flipped_signature_is_refused(self, tmp_path: Path) -> None:
        store, request = _held(tmp_path)
        data = callback_data(_approvals(), request, "a")
        forged = data[:-1] + ("0" if data[-1] != "0" else "1")
        out = handle_callback(_tap(forged), store, _approvals(), now=_soon(request))
        assert out is not None and "does not verify" in out.toast
        assert store.status(request.request_id) == "paused"

    def test_a_reject_signature_cannot_be_relabelled_approve(self, tmp_path: Path) -> None:
        store, request = _held(tmp_path)
        _, _, token, tag = callback_data(_approvals(), request, "r").split(":")
        out = handle_callback(_tap(f"pz:a:{token}:{tag}"), store, _approvals(), now=_soon(request))
        assert out is not None and "does not verify" in out.toast

    def test_a_button_signed_with_another_secret_matches_nothing(self, tmp_path: Path) -> None:
        store, request = _held(tmp_path)
        other = Approvals(reviewers={REVIEWER: "alice"}, secret=b"x" * 40)
        out = handle_callback(
            _tap(callback_data(other, request, "a")), store, _approvals(), now=_soon(request)
        )
        assert out is not None and "No held decision" in out.toast

    def test_a_replayed_tap_is_refused_by_the_store(self, tmp_path: Path) -> None:
        store, request = _held(tmp_path)
        data = callback_data(_approvals(), request, "a")
        handle_callback(_tap(data), store, _approvals(), now=_soon(request))
        again = handle_callback(_tap(data, qid="q2"), store, _approvals(), now=_soon(request))
        assert again is not None and again.toast.startswith("Already answered")
        assert len([e for e in store.events() if e["event"] == "answered"]) == 1

    def test_two_reviewers_tapping_at_once_record_one_answer(self, tmp_path: Path) -> None:
        store, request = _held(tmp_path)
        approvals = _approvals()
        outcomes: list[Any] = []
        taps = [
            threading.Thread(
                target=lambda u=u: outcomes.append(
                    handle_callback(
                        _tap(callback_data(approvals, request, "a"), user=u),
                        store,
                        approvals,
                        now=_soon(request),
                    )
                )
            )
            for u in (REVIEWER, 222)
        ]
        for t in taps:
            t.start()
        for t in taps:
            t.join()
        assert len([e for e in store.events() if e["event"] == "answered"]) == 1
        assert sorted(o.toast.split(":")[0] for o in outcomes) == ["Already answered", "Recorded"]

    def test_a_late_tap_is_refused_as_expired(self, tmp_path: Path) -> None:
        store, request = _held(tmp_path)
        late = request.expires_at + timedelta(minutes=1)
        out = handle_callback(
            _tap(callback_data(_approvals(), request, "a")), store, _approvals(), now=late
        )
        assert out is not None and out.toast.startswith("Not recorded")
        assert not [e for e in store.events() if e["event"] == "answered"]

    def test_an_ordinary_update_is_not_a_tap(self, tmp_path: Path) -> None:
        store, _ = _held(tmp_path)
        assert handle_callback({"message": {"text": "hi"}}, store, _approvals()) is None
        assert handle_callback(_tap("w:open:1"), store, _approvals()) is None


class TestFloodControl:
    """A 429 is waited out once for the time Telegram named (decision 2 of the study)."""

    @staticmethod
    def _limited(retry_after: float) -> Exception:
        import json

        from argus.truth.failures import classify_http
        from argus.truth.http import HttpError

        body = json.dumps({"ok": False, "error_code": 429,
                           "parameters": {"retry_after": retry_after}}).encode()
        return HttpError(classify_http(429, body), body)

    def test_a_429_is_waited_out_and_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import telegram_bot

        calls: list[int] = []
        waited: list[float] = []

        def fetch(*_: Any, **__: Any) -> Any:
            calls.append(1)
            if len(calls) == 1:
                raise self._limited(3)
            return {"ok": True, "result": {"message_id": 5}}

        monkeypatch.setattr(telegram_bot.http, "fetch_json", fetch)
        monkeypatch.setattr(telegram_bot, "_sleep", waited.append)
        assert telegram_bot._call("t", "sendMessage", {}) == {"message_id": 5}
        assert waited == [3.0] and len(calls) == 2

    def test_a_long_wait_is_raised_not_slept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import telegram_bot

        def fetch(*_: Any, **__: Any) -> Any:
            raise self._limited(600)

        monkeypatch.setattr(telegram_bot.http, "fetch_json", fetch)
        monkeypatch.setattr(telegram_bot, "_sleep", lambda s: pytest.fail("slept"))
        with pytest.raises(Exception):  # noqa: B017 - the typed HttpError, re-raised
            telegram_bot._call("t", "sendMessage", {})

    def test_other_failures_are_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import telegram_bot

        calls: list[int] = []

        def fetch(*_: Any, **__: Any) -> Any:
            calls.append(1)
            raise OSError("down")

        monkeypatch.setattr(telegram_bot.http, "fetch_json", fetch)
        with pytest.raises(OSError):
            telegram_bot._call("t", "sendMessage", {})
        assert len(calls) == 1


def test_half_size_is_written_as_a_person_writes_it() -> None:
    """300, not 3E+2, on the reviewer's button (QA surfaces pass, 2026-09-28)."""
    from decimal import Decimal
    from types import SimpleNamespace

    for held, half in (("600", "300"), ("5", "2.5"), ("0.30", "0.15"), ("1000000", "500000")):
        assert str(pause_bot.half_of(SimpleNamespace(max_quantity=Decimal(held)))) == half
