"""The console in Telegram: the same questions and engines, answered where traders already are.

Two ways in, one core. ``handle_update`` turns one Telegram update into replies; it is reached
either by the webhook (`POST /telegram` on the console's own server, which Telegram calls when a
message arrives) or by ``poll`` (long polling from any machine, no public URL needed). Both answer
through `server.handle_ask` — nothing here reads a question, computes a figure or decides what is
research; this module only carries text in and out.

What a chat keeps: its last twelve questions, so "and what about TSLA?" resolves the way it does in
the browser, and a saved book (``/book 40% NVDA, 30% MSFT, 30% AAPL``) so portfolio questions need
not repeat it. Both live in the process's memory. Under long polling that is one process and it
holds; behind the webhook on a serverless host each warm instance holds its own, so a cold start
forgets them — the reply says so when a book-dependent question arrives without one.

Security: the webhook is accepted only with Telegram's ``X-Telegram-Bot-Api-Secret-Token`` header
matching ``TELEGRAM_WEBHOOK_SECRET`` (compared in constant time), set when the webhook is
registered (``--set-webhook``). The bot token is read from ``TELEGRAM_BOT_TOKEN`` and never logged.
The console refuses orders, and so does the bot: it is the same code path.

Telegram's Bot API, as used here: ``getUpdates`` (long polling, ``offset`` = last update_id + 1),
``sendMessage`` (text up to 4096 characters, ``parse_mode=HTML`` needs ``<``, ``>`` and ``&``
escaped), ``setWebhook`` with ``secret_token``. https://core.telegram.org/bots/api
"""

from __future__ import annotations

import argparse
import hmac
import html
import json
import os
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

API = "https://api.telegram.org/bot{token}/{method}"
CONSOLE = "https://deploy-topaz-seven-64.vercel.app/"
MAX_MESSAGE = 4000
"""Telegram's limit is 4096 characters of text; the margin covers the HTML tags added."""
HOURLY_LIMIT = 30
"""Questions per chat per hour, the same allowance the web console gives a visitor."""
MAX_TURNS = 12

HELP = (
    "<b>ARGUS research desk</b> — ask about any contract Bitget lists: stocks, ETFs, gold, oil, "
    "crypto. Every figure is computed from live data and sourced; nothing is a forecast, and no "
    "order is ever placed.\n\n"
    "Try:\n"
    "• is NVDA overbought?\n"
    "• what does adding 15% TSLA do to my risk?\n"
    "• will MSTR be higher in 48 hours?\n"
    "• if the Nasdaq drops 10%, what happens to my book?\n"
    "• 英伟达的资金费率贵吗？\n\n"  # noqa: RUF001 - a real Chinese question
    "/book 40% NVDA, 30% MSFT, 30% AAPL — save your holdings for portfolio questions\n"
    "/book — show them · /clear — forget this chat's history and book"
)


@dataclass
class ChatState:
    turns: list[str] = field(default_factory=list)
    book: str = ""
    asked_at: list[float] = field(default_factory=list)


Ask = Callable[..., dict[str, Any]]


def _default_ask(text: str, prior: list[str], *, visitor: str, book: str) -> dict[str, Any]:
    """The console's answer, with the signed translation offer when the question was not English
    (`server.offer_translation`)."""
    from argus.lui.server import handle_ask, offer_translation

    payload = handle_ask(text, prior, visitor=visitor, book=book)
    offer_translation(payload, text)
    return payload


def _translated(payload: dict[str, Any], question: str, book: str, visitor: str) -> str | None:
    """The same answer in the question's language, or None when there is no offer or nothing
    was translated. A translation measured 9 to 32 seconds on the hackathon Qwen, so the bot sends
    the English at once and edits it into this when it arrives (`lui/translate.py`)."""
    offer = payload.get("translate")
    if not offer:
        return None
    from argus.lui import translate
    from argus.lui.server import _model_for

    lines = [str(line) for line in payload.get("lines") or []]
    skip = int(offer["skip"])
    done = translate.translate(lines[skip:], str(offer["lang"]), _model_for(visitor))
    if len(done["kept_english"]) == len(lines) - skip:
        return None
    head = [done["note"]] if done.get("note") else lines[:skip]
    lead = {len(head) + i for i, line in enumerate(lines[skip:])
            if line.startswith("Actionable:")}
    from argus.lui.provenance import labels as provenance_labels

    tags = [None] * len(head) + provenance_labels(lines[skip:])
    return format_answer({**payload, "lines": [*head, *done["lines"]], "line_labels": tags},
                         question, book, lead)


def split_message(text: str, limit: int = MAX_MESSAGE) -> list[str]:
    """Break at paragraph boundaries, then at line ends, never mid-tag for the tags used here
    (each tag opens and closes inside one paragraph)."""
    parts: list[str] = []
    current = ""
    for block in text.split("\n\n"):
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            parts.append(current)
        while len(block) > limit:
            cut = block.rfind("\n", 0, limit)
            cut = cut if cut > limit // 2 else block.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            parts.append(block[:cut])
            block = block[cut:].lstrip()
        current = block
    if current:
        parts.append(current)
    return parts


def format_answer(payload: dict[str, Any], question: str, book: str,
                  lead: set[int] | None = None) -> str:
    """The console's answer as a Telegram message: its lines, its sources by name, and a link to
    the same question on the web console where every source is expandable."""
    from argus.lui.provenance import labels as provenance_labels

    lines = [str(line) for line in payload.get("lines") or []]
    tags = payload.get("line_labels")
    if not isinstance(tags, list) or len(tags) != len(lines):
        tags = provenance_labels(lines)
    body = []
    for index, line in enumerate(lines):
        text = html.escape(line, quote=False)
        if text.startswith("Actionable:"):
            text = "<b>Actionable:</b>" + text[len("Actionable:"):]
        elif lead and index in lead:
            # A translated lead line no longer starts with "Actionable:"; it stays bold.
            text = f"<b>{text}</b>"
        if tags[index]:
            text += f" <i>· {tags[index]}</i>"
        body.append(text)
    refs: list[str] = []
    for source in payload.get("sources") or []:
        ref = str(source.get("ref", "")).strip()
        if ref and ref not in refs:
            refs.append(ref)
    if refs:
        # In code tags: Telegram otherwise links "technical_analysis.support" as a web address
        # and "/api/v3/market/candles" as a bot command (seen in the chat, 2026-09-25).
        shown = ", ".join(f"<code>{html.escape(r, quote=False)}</code>" for r in refs[:6])
        more = f" and {len(refs) - 6} more" if len(refs) > 6 else ""
        body.append(f"<i>Sources: {shown}{more}.</i>")
    query = {"q": question[:500], **({"book": book[:300]} if book else {})}
    link = CONSOLE + "?" + urllib.parse.urlencode(query)
    body.append(f'<a href="{html.escape(link)}">Open this answer on the console</a>')
    return "\n\n".join(body)


def handle_update(update: dict[str, Any], states: dict[int, ChatState], *,
                  ask: Ask = _default_ask, now: float | None = None) -> list[tuple[int, str]]:
    """Replies for one update, as (chat_id, html_text) pairs. Updates that are not a text message
    from a chat (edits, joins, stickers) get no reply."""
    message = update.get("message") or {}
    chat = message.get("chat") or {}
    text = str(message.get("text") or "").strip()
    chat_id = chat.get("id")
    if not isinstance(chat_id, int) or not text:
        return []
    state = states.setdefault(chat_id, ChatState())
    clock = time.time() if now is None else now
    command, _, rest = text.partition(" ")
    command = command.split("@", 1)[0].lower()
    if command in ("/start", "/help"):
        return [(chat_id, HELP)]
    if command == "/clear":
        states[chat_id] = ChatState(asked_at=state.asked_at)
        return [(chat_id, "Forgotten: this chat's earlier questions and its saved book.")]
    if command == "/book":
        if rest.strip():
            state.book = rest.strip()[:300]
            return [(chat_id, f"{SAVED_PREFIX}{html.escape(state.book)}. Portfolio questions "
                              f"in this chat now use it.")]
        return [(chat_id, f"Your saved book: {html.escape(state.book)}." if state.book else
                 "No book saved. Send, for example: /book 40% NVDA, 30% MSFT, 30% AAPL")]
    if command.startswith("/"):
        return [(chat_id, "I only know /start, /help, /book and /clear — anything else, just "
                          "ask it as a question.")]
    state.asked_at = [t for t in state.asked_at if clock - t < 3600.0]
    if len(state.asked_at) >= HOURLY_LIMIT:
        return [(chat_id, f"That is {HOURLY_LIMIT} questions this hour from this chat, the same "
                          f"allowance the web console gives. Ask again in a while, or use "
                          f'<a href="{CONSOLE}">the console</a>.')]
    state.asked_at.append(clock)
    try:
        payload = ask(text[:500], list(state.turns), visitor=f"tg-{chat_id}", book=state.book)
    except Exception as exc:
        return [(chat_id, f"The desk could not answer just now ({html.escape(type(exc).__name__)})"
                          f". Nothing was guessed; try again in a minute.")]
    state.turns = [*state.turns, text[:500]][-MAX_TURNS:]
    if payload.get("translate"):
        PENDING[chat_id] = (payload, text[:500], state.book)
    return [(chat_id, part) for part in split_message(format_answer(payload, text, state.book))]


PENDING: dict[int, tuple[dict[str, Any], str, str]] = {}
"""An answer sent in English whose translation is still to come, by chat."""


def follow_up(token: str, chat_id: int, message_id: int | None) -> None:
    """Edit the English answer just sent into the question's language, or send the translation
    as a new message when it is too long for one. Nothing happens when there is none."""
    pending = PENDING.pop(chat_id, None)
    if pending is None:
        return
    payload, question, book = pending
    text = _translated(payload, question, book, f"tg-{chat_id}")
    if text is None:
        return
    parts = split_message(text)
    if message_id is not None and len(parts) == 1:
        _call(token, "editMessageText", {"chat_id": chat_id, "message_id": message_id,
                                         "text": parts[0], "parse_mode": "HTML",
                                         "link_preview_options": {"is_disabled": True}},
              timeout=30.0)
        return
    for part in parts:
        send(token, chat_id, part)


# --- Telegram transport -------------------------------------------------------------------------

def _call(token: str, method: str, params: dict[str, Any], *, timeout: float = 70.0) -> Any:
    request = urllib.request.Request(
        API.format(token=token, method=method), data=json.dumps(params).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram {method} failed: {payload.get('description')}")
    return payload.get("result")


def send(token: str, chat_id: int, text: str) -> int | None:
    sent = _call(token, "sendMessage", {"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                                        "link_preview_options": {"is_disabled": True}},
                 timeout=30.0)
    return sent.get("message_id") if isinstance(sent, dict) else None


_WEBHOOK_STATES: dict[int, ChatState] = {}

SAVED_PREFIX = "Saved your book: "


def recall_book(token: str, chat_id: int, states: dict[int, ChatState]) -> None:
    """Fill a chat's book from its pinned "Saved your book" message when this process does not
    hold it. Behind the webhook each serverless instance keeps its own memory, so a book saved in
    one request was unknown to the next; the chat itself is the one store every instance can read
    (`getChat` returns the pinned message)."""
    state = states.setdefault(chat_id, ChatState())
    if state.book:
        return
    try:
        chat = _call(token, "getChat", {"chat_id": chat_id}, timeout=10.0)
    except Exception:
        return
    pinned = str(((chat or {}).get("pinned_message") or {}).get("text") or "")
    if pinned.startswith(SAVED_PREFIX):
        state.book = pinned[len(SAVED_PREFIX):].split(". Portfolio questions", 1)[0].strip()


def remember_book(token: str, chat_id: int, message_id: int | None, text: str) -> None:
    """Pin the confirmation of a saved book, silently, so any later request can recall it; a
    cleared book unpins everything the bot pinned."""
    try:
        if text.startswith(SAVED_PREFIX) and message_id is not None:
            _call(token, "pinChatMessage", {"chat_id": chat_id, "message_id": message_id,
                                            "disable_notification": True}, timeout=10.0)
        elif text.startswith("Forgotten:"):
            _call(token, "unpinAllChatMessages", {"chat_id": chat_id}, timeout=10.0)
    except Exception:
        return


def handle_webhook(body: bytes, secret_header: str | None, *,
                   ask: Ask = _default_ask) -> tuple[int, bytes]:
    """The ``POST /telegram`` endpoint. 403 without the registered secret; 200 otherwise, after
    the replies are sent — Telegram retries an update until it gets a 2xx, so a failure to send
    is logged in the response body but still acknowledged rather than answered twice."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
    if not token or not secret:
        return 503, b'{"error": "the Telegram bot is not configured on this server"}'
    if not hmac.compare_digest((secret_header or "").encode(), secret.encode()):
        return 403, b'{"error": "bad secret token"}'
    try:
        update = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return 400, b'{"error": "not JSON"}'
    failures = 0
    last: dict[int, int | None] = {}
    chat = ((update.get("message") or {}).get("chat") or {}).get("id")
    if isinstance(chat, int):
        recall_book(token, chat, _WEBHOOK_STATES)
    for chat_id, text in handle_update(update, _WEBHOOK_STATES, ask=ask):
        try:
            last[chat_id] = send(token, chat_id, text)
            remember_book(token, chat_id, last[chat_id], text)
        except Exception:
            failures += 1
    for chat_id, message_id in last.items():
        try:
            follow_up(token, chat_id, message_id)
        except Exception:
            failures += 1
    return 200, json.dumps({"ok": True, "send_failures": failures}).encode()


def poll(token: str, *, ask: Ask = _default_ask) -> None:  # pragma: no cover - network loop
    """Long polling: no public URL, one process, chat state held for its lifetime. Deletes any
    registered webhook first, because Telegram refuses getUpdates while one is set."""
    _call(token, "deleteWebhook", {"drop_pending_updates": False})
    states: dict[int, ChatState] = {}
    offset = 0
    while True:
        try:
            updates = _call(token, "getUpdates", {"offset": offset, "timeout": 50,
                                                  "allowed_updates": ["message"]})
        except Exception as exc:
            print(f"getUpdates failed ({type(exc).__name__}); retrying in 5s", flush=True)
            time.sleep(5)
            continue
        for update in updates:
            offset = max(offset, int(update["update_id"]) + 1)
            last: dict[int, int | None] = {}
            chat = ((update.get("message") or {}).get("chat") or {}).get("id")
            if isinstance(chat, int):
                recall_book(token, chat, states)
            for chat_id, text in handle_update(update, states, ask=ask):
                try:
                    last[chat_id] = send(token, chat_id, text)
                    remember_book(token, chat_id, last[chat_id], text)
                except Exception as exc:
                    print(f"sendMessage failed ({type(exc).__name__})", flush=True)
            for chat_id, message_id in last.items():
                try:
                    follow_up(token, chat_id, message_id)
                except Exception as exc:
                    print(f"translation follow-up failed ({type(exc).__name__})", flush=True)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="ARGUS in Telegram")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--poll", action="store_true", help="answer by long polling")
    group.add_argument("--set-webhook", metavar="URL",
                       help="register URL (ending /telegram) with TELEGRAM_WEBHOOK_SECRET")
    args = parser.parse_args(argv)
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("TELEGRAM_BOT_TOKEN is not set: create a bot with @BotFather and export its token.")
        return 2
    if args.set_webhook:
        secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
        if not secret:
            print("TELEGRAM_WEBHOOK_SECRET is not set: choose a random string (A-Z a-z 0-9 _ -) "
                  "and set it both here and on the server.")
            return 2
        _call(token, "setWebhook", {"url": args.set_webhook, "secret_token": secret,
                                    "allowed_updates": ["message"]})
        _call(token, "setMyCommands", {"commands": [
            {"command": "help", "description": "what the desk answers"},
            {"command": "book", "description": "save or show your holdings"},
            {"command": "clear", "description": "forget this chat's history and book"}]})
        print("webhook registered")
        return 0
    poll(token)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
