"""The console in Telegram: commands, history, the saved book, limits, the webhook's secret."""

from __future__ import annotations

import json
from typing import Any

import pytest

from argus.lui import telegram_bot as tg


class FakeDesk:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, text: str, prior: list[str], *, visitor: str, book: str) -> dict[str, Any]:
        self.calls.append({"text": text, "prior": prior, "visitor": visitor, "book": book})
        return {"lines": ["Actionable: NVDA <is> fine & well.", "Second line."],
                "sources": [{"ref": "bitget tickers"}, {"ref": "bitget tickers"},
                            {"ref": "SEC EDGAR"}]}


def _msg(text: str, chat: int = 42) -> dict[str, Any]:
    return {"update_id": 1, "message": {"chat": {"id": chat}, "text": text}}


def test_a_question_is_answered_through_the_console_with_its_sources_and_a_link() -> None:
    desk, states = FakeDesk(), {}
    [(chat, text)] = tg.handle_update(_msg("is NVDA overbought?"), states, ask=desk)
    assert chat == 42 and desk.calls[0]["visitor"] == "tg-42"
    assert text.startswith("<b>Actionable:</b> NVDA &lt;is&gt; fine &amp; well.")
    assert "<i>Sources: bitget tickers, SEC EDGAR.</i>" in text
    assert "?q=is+NVDA+overbought%3F" in text


def test_history_carries_follow_ups_and_the_saved_book_is_sent() -> None:
    desk, states = FakeDesk(), {}
    tg.handle_update(_msg("/book 40% NVDA, 60% AAPL"), states, ask=desk)
    tg.handle_update(_msg("what does adding 10% TSLA do?"), states, ask=desk)
    tg.handle_update(_msg("and MSFT?"), states, ask=desk)
    assert desk.calls[1]["prior"] == ["what does adding 10% TSLA do?"]
    assert desk.calls[1]["book"] == "40% NVDA, 60% AAPL"
    [(_, shown)] = tg.handle_update(_msg("/book"), states, ask=desk)
    assert "40% NVDA, 60% AAPL" in shown
    tg.handle_update(_msg("/clear"), states, ask=desk)
    assert states[42].turns == [] and states[42].book == ""


def test_chats_do_not_share_state() -> None:
    desk, states = FakeDesk(), {}
    tg.handle_update(_msg("/book 100% NVDA", chat=1), states, ask=desk)
    tg.handle_update(_msg("how risky is my book?", chat=2), states, ask=desk)
    assert desk.calls[0]["book"] == "" and desk.calls[0]["prior"] == []


def test_help_unknown_commands_and_non_text_updates() -> None:
    desk, states = FakeDesk(), {}
    assert "/book" in tg.handle_update(_msg("/start"), states, ask=desk)[0][1]
    assert "only know" in tg.handle_update(_msg("/buy NVDA"), states, ask=desk)[0][1]
    assert tg.handle_update({"update_id": 2, "edited_message": {}}, states, ask=desk) == []
    assert desk.calls == []


def test_the_hourly_allowance_holds_per_chat() -> None:
    desk, states = FakeDesk(), {}
    for i in range(tg.HOURLY_LIMIT):
        tg.handle_update(_msg(f"q {i}"), states, ask=desk, now=1000.0 + i)
    [(_, text)] = tg.handle_update(_msg("one more"), states, ask=desk, now=1100.0)
    assert "questions this hour" in text and len(desk.calls) == tg.HOURLY_LIMIT
    tg.handle_update(_msg("later"), states, ask=desk, now=1000.0 + 3700)
    assert len(desk.calls) == tg.HOURLY_LIMIT + 1


def test_a_desk_failure_is_said_not_guessed() -> None:
    def broken(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise TimeoutError

    [(_, text)] = tg.handle_update(_msg("is NVDA overbought?"), {}, ask=broken)
    assert "could not answer just now (TimeoutError)" in text


def test_long_answers_split_under_telegrams_limit() -> None:
    body = "\n\n".join("x" * 900 for _ in range(10))
    parts = tg.split_message(body)
    assert len(parts) == 3 and all(len(p) <= tg.MAX_MESSAGE for p in parts)
    assert "".join(parts).replace("\n", "") == body.replace("\n", "")


def test_the_webhook_needs_configuration_and_the_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    assert tg.handle_webhook(b"{}", "s")[0] == 503
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "right")
    assert tg.handle_webhook(b"{}", "wrong")[0] == 403
    assert tg.handle_webhook(b"{}", None)[0] == 403
    assert tg.handle_webhook(b"not json", "right")[0] == 400
    sent: list[tuple[int, str]] = []
    monkeypatch.setattr(tg, "send", lambda token, chat, text: sent.append((chat, text)))
    status, body = tg.handle_webhook(json.dumps(_msg("/help")).encode(), "right")
    assert status == 200 and json.loads(body)["send_failures"] == 0 and sent[0][0] == 42
