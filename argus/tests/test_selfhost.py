"""Run-your-own ARGUS in Telegram (`lui/selfhost.py`): setup, pairing, persistence, offline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from argus.lui import selfhost
from argus.lui.selfhost import BotConfig


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARGUS_HOME", str(tmp_path))
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)


def _update(text: str, user: int = 7, chat: int | None = None) -> dict[str, Any]:
    return {"update_id": 1, "message": {"chat": {"id": chat or user},
                                        "from": {"id": user, "first_name": "Ada"},
                                        "text": text}}


def _echo(update: dict[str, Any], states: dict[int, Any], **_: Any) -> list[tuple[int, str]]:
    chat = update["message"]["chat"]["id"]
    return [(chat, f"answered: {update['message']['text']}")]


def test_setup_checks_the_token_and_keeps_it_locally(tmp_path: Path) -> None:
    reply = selfhost.setup("123:abc", "pairing", [], get_me=lambda t: {"username": "my_desk"})
    assert reply.startswith("Ready: https://t.me/my_desk (pairing).")
    saved = BotConfig.load()
    assert saved is not None and saved.token == "123:abc" and saved.username == "my_desk"
    assert (tmp_path / "bot.json").exists()


def test_a_refused_token_and_an_empty_allowlist_are_said_plainly() -> None:
    def refuse(token: str) -> dict[str, Any]:
        raise RuntimeError("401 Unauthorized")

    assert "refused that token" in selfhost.setup("bad", "pairing", [], get_me=refuse)
    assert "needs at least one numeric user id" in selfhost.setup(
        "123:abc", "allowlist", [], get_me=lambda t: {})
    assert BotConfig.load() is None


def test_a_stranger_gets_a_code_and_nothing_else_until_approved() -> None:
    config = BotConfig(token="t")
    first = selfhost.handle(_update("is NVDA overbought?"), config, {}, watches=None,
                            respond=_echo, now=1000.0)
    assert "private ARGUS desk" in first[0][1] and "user id is 7" in first[0][1]
    code = next(iter(config.pending))
    again = selfhost.handle(_update("hello"), config, {}, watches=None, respond=_echo,
                            now=1010.0)
    assert code in again[0][1]  # the same request, not a new code each message
    assert selfhost.approve(config, code.lower(), 1020.0) == "Approved Ada (7) as the owner."
    answered = selfhost.handle(_update("is NVDA overbought?"), config, {}, watches=None,
                               respond=_echo, now=1030.0)
    assert answered == [(7, "answered: is NVDA overbought?")]


def test_codes_expire_and_only_the_owner_approves_from_telegram() -> None:
    config = BotConfig(token="t", owner=7, allow_from=[7])
    selfhost.handle(_update("hi", user=9), config, {}, watches=None, respond=_echo, now=0.0)
    code = next(iter(config.pending))
    refused = selfhost.handle(_update(f"/approve {code}", user=9), config, {}, watches=None,
                              respond=_echo, now=10.0)
    assert refused[0][1] == "Only this desk's owner can approve pairing requests."
    by_owner = selfhost.handle(_update(f"/approve {code}", user=7), config, {}, watches=None,
                               respond=_echo, now=20.0)
    assert by_owner[0][1] == "Approved Ada (9)."
    assert 9 in config.allow_from and config.owner == 7
    selfhost.handle(_update("hi", user=11), config, {}, watches=None, respond=_echo, now=0.0)
    late = next(c for c, r in config.pending.items() if r["user_id"] == 11)
    assert "expire" in selfhost.approve(config, late, selfhost.PAIRING_SECONDS + 1)


def test_allowlist_and_open_policies() -> None:
    locked = BotConfig(token="t", policy="allowlist", allow_from=[5])
    assert "argus-bot allow 6" in selfhost.handle(
        _update("hi", user=6), locked, {}, watches=None, respond=_echo)[0][1]
    assert selfhost.handle(_update("hi", user=5), locked, {}, watches=None,
                           respond=_echo) == [(5, "answered: hi")]
    public = BotConfig(token="t", policy="open")
    assert selfhost.handle(_update("hi", user=99), public, {}, watches=None,
                           respond=_echo) == [(99, "answered: hi")]


def test_a_chat_keeps_its_book_and_memory_across_restarts() -> None:
    from argus.lui.telegram_bot import ChatState

    states = {7: ChatState(turns=["q1"], book="40% NVDA, 60% MSFT", memory='[{"kind":"x"}]')}
    selfhost.save_states(states)
    back = selfhost.load_states()
    assert back[7].book == "40% NVDA, 60% MSFT" and back[7].turns == ["q1"]
    assert back[7].memory == '[{"kind":"x"}]'
