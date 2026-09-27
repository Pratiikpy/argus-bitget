"""The Telegram chat store: a restart resumes a conversation, and the remote record is sealed."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import pytest

from argus.lui import chat_store, telegram_bot
from argus.lui.chat_store import FileChatStore, RestChatStore, from_env, parse_key
from argus.lui.telegram_bot import ChatState, handle_update, hydrate, persist


def _message(chat_id: int, text: str) -> dict[str, Any]:
    return {"update_id": 1, "message": {"chat": {"id": chat_id}, "text": text}}


def _ask(text: str, prior: list[str], **_: Any) -> dict[str, Any]:
    return {"answer": f"seen {len(prior)} earlier", "kind": "test"}


class FakeUpstash:
    """Upstash's REST contract as its docs state it: a JSON command array, ``{"result": ...}``."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def __call__(self, url: str, *, data: bytes, method: str, headers: dict[str, str],
                 timeout: float) -> Any:
        assert method == "POST" and headers["Authorization"] == "Bearer tok"
        command = json.loads(data)
        verb = command[0]
        if verb == "SET":
            assert command[3:] == ["EX", chat_store.TTL_SECONDS]
            self.data[command[1]] = command[2]
            return {"result": "OK"}
        if verb == "GET":
            return {"result": self.data.get(command[1])}
        if verb == "DEL":
            return {"result": int(self.data.pop(command[1], None) is not None)}
        return {"error": f"ERR unknown command {verb}"}


KEY = base64.urlsafe_b64encode(bytes(range(32))).decode()


@pytest.fixture
def upstash(monkeypatch: pytest.MonkeyPatch) -> FakeUpstash:
    fake = FakeUpstash()
    monkeypatch.setattr("argus.lui.chat_store.http.fetch_json", fake)
    return fake


def test_a_restarted_bot_resumes_the_conversation(tmp_path: Path) -> None:
    store = FileChatStore(tmp_path)
    first: dict[int, ChatState] = {}
    for text in ("/book 50% NVDA, 50% MSFT", "is NVDA overbought?"):
        hydrate(first, 7, store)
        handle_update(_message(7, text), first, ask=_ask, now=100.0)
        persist(first, 7, store)
    restarted: dict[int, ChatState] = {}
    hydrate(restarted, 7, store)
    seen: list[list[str]] = []

    def ask(text: str, prior: list[str], **kwargs: Any) -> dict[str, Any]:
        seen.append(prior)
        return _ask(text, prior)

    handle_update(_message(7, "and MSFT?"), restarted, ask=ask, now=101.0)
    assert seen == [["is NVDA overbought?"]]
    assert restarted[7].book == "50% NVDA, 50% MSFT"
    assert restarted[7].asked_at == [100.0, 101.0]


def test_the_hourly_allowance_survives_cold_starts(tmp_path: Path) -> None:
    store = FileChatStore(tmp_path)
    states: dict[int, ChatState] = {}
    for i in range(telegram_bot.HOURLY_LIMIT):
        hydrate(states, 3, store)
        handle_update(_message(3, f"q{i}"), states, ask=_ask, now=10.0 + i)
        persist(states, 3, store)
        states.clear()
    hydrate(states, 3, store)
    reply = handle_update(_message(3, "one more"), states, ask=_ask, now=100.0)
    assert "questions this hour" in reply[0][1]


def test_clear_forgets_the_stored_chat(tmp_path: Path) -> None:
    store = FileChatStore(tmp_path)
    states: dict[int, ChatState] = {}
    handle_update(_message(5, "/book 100% TSLA"), states, ask=_ask)
    persist(states, 5, store)
    handle_update(_message(5, "/clear"), states, ask=_ask)
    persist(states, 5, store)
    fresh: dict[int, ChatState] = {}
    hydrate(fresh, 5, store)
    assert fresh.get(5, ChatState()).book == ""


def test_an_empty_chat_is_deleted_not_stored(tmp_path: Path) -> None:
    store = FileChatStore(tmp_path)
    store.save(9, {"book": "x"})
    persist({9: ChatState()}, 9, store)
    assert store.load(9) is None


def test_a_damaged_record_is_bounded_on_the_way_in() -> None:
    state = ChatState.from_record({"turns": ["a" * 900] * 40, "book": "b" * 900,
                                   "asked_at": ["1", None, "x", 2.5], "memory": None})
    assert len(state.turns) == telegram_bot.MAX_TURNS and len(state.turns[0]) == 500
    assert len(state.book) == 300 and state.asked_at == [1.0, 2.5] and state.memory == ""


def test_an_unreachable_store_keeps_the_local_copy() -> None:
    class Down:
        def load(self, chat_id: int) -> Any:
            raise OSError("down")

        def save(self, chat_id: int, state: Any) -> None:
            raise OSError("down")

        def delete(self, chat_id: int) -> None:
            raise OSError("down")

    states = {1: ChatState(book="kept")}
    hydrate(states, 1, Down())
    persist(states, 1, Down())
    assert states[1].book == "kept"


def test_a_chat_cleared_by_another_instance_is_dropped_here(tmp_path: Path) -> None:
    states = {4: ChatState(book="stale")}
    hydrate(states, 4, FileChatStore(tmp_path))
    assert 4 not in states


def test_the_remote_record_is_encrypted_and_names_no_chat(upstash: FakeUpstash) -> None:
    store = RestChatStore("https://db.example", "tok", parse_key(KEY))
    store.save(123456789, {"book": "40% NVDA", "turns": ["my stop is 170"]})
    [(name, value)] = upstash.data.items()
    assert "123456789" not in name
    assert b"NVDA" not in base64.b64decode(value) and b"stop" not in base64.b64decode(value)
    assert store.load(123456789) == {"book": "40% NVDA", "turns": ["my stop is 170"]}
    store.delete(123456789)
    assert store.load(123456789) is None


def test_a_record_moved_under_another_chat_does_not_open(upstash: FakeUpstash) -> None:
    from cryptography.exceptions import InvalidTag

    store = RestChatStore("https://db.example", "tok", parse_key(KEY))
    store.save(1, {"book": "mine"})
    store.save(2, {"book": "theirs"})
    upstash.data[store._name(2)] = upstash.data[store._name(1)]
    with pytest.raises(InvalidTag):
        store.load(2)


def test_another_key_finds_no_record(upstash: FakeUpstash) -> None:
    RestChatStore("https://db.example", "tok", parse_key(KEY)).save(1, {"book": "x"})
    other = base64.urlsafe_b64encode(os.urandom(32)).decode()
    assert RestChatStore("https://db.example", "tok", parse_key(other)).load(1) is None


def test_an_upstash_error_is_raised_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("argus.lui.chat_store.http.fetch_json",
                        lambda *a, **k: {"error": "WRONGPASS invalid password"})
    store = RestChatStore("https://db.example", "tok", parse_key(KEY))
    with pytest.raises(RuntimeError, match="WRONGPASS"):
        store.load(1)


@pytest.mark.parametrize("bad", ["", "not base64!!", base64.urlsafe_b64encode(b"short").decode()])
def test_a_bad_key_is_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_key(bad)


def test_from_env_picks_the_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN", "KV_REST_API_URL",
                 "KV_REST_API_TOKEN", chat_store.KEY_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ARGUS_CHAT_STORE", str(tmp_path))
    assert from_env(remote=True) is None
    assert isinstance(from_env(remote=False), FileChatStore)
    monkeypatch.setenv("KV_REST_API_URL", "https://db.example")
    monkeypatch.setenv("KV_REST_API_TOKEN", "tok")
    assert from_env(remote=True) is None  # configured without a key: refused, never plaintext
    monkeypatch.setenv(chat_store.KEY_ENV, KEY)
    assert isinstance(from_env(remote=True), RestChatStore)


def test_the_webhook_loads_and_saves_through_the_store(monkeypatch: pytest.MonkeyPatch,
                                                       tmp_path: Path) -> None:
    store = FileChatStore(tmp_path)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "s")
    monkeypatch.setattr(telegram_bot, "_WEBHOOK_STORE", [store])
    monkeypatch.setattr(telegram_bot, "_WEBHOOK_STATES", {})
    monkeypatch.setattr(telegram_bot, "_call", lambda *a, **k: {})
    monkeypatch.setattr(telegram_bot, "send", lambda *a, **k: None)
    body = json.dumps(_message(11, "/book 100% AAPL")).encode()
    assert telegram_bot.handle_webhook(body, "s", ask=_ask)[0] == 200
    telegram_bot._WEBHOOK_STATES.clear()
    telegram_bot.handle_webhook(json.dumps(_message(11, "/book")).encode(), "s", ask=_ask)
    assert telegram_bot._WEBHOOK_STATES[11].book == "100% AAPL"
