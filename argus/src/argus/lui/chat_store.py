"""Where a Telegram chat's state lives between messages, so a restart does not wipe a conversation.

**Why this exists.** The bot held each chat's last questions, saved book, remembered facts and
hourly allowance in a dict inside the process (audit finding 176). Under long polling a restart
forgot all of it; behind the webhook every cold start did, so "and what about TSLA?" lost its
referent and the thirty-questions-an-hour allowance reset whenever a new instance answered.

**What was taken, and from where.** LangGraph's checkpointers (``langchain-ai/langgraph``, MIT):
state is keyed by a thread id, written after every step and read back before the next one
(``libs/checkpoint/langgraph/checkpoint/base/__init__.py:183-201``, ``put``/``get_tuple``), with an
in-memory saver its own docstring calls dev-only (``checkpoint/memory/__init__.py:40-44``) and
durable backends for production. We take that shape — one record per chat id, loaded before an
update and saved after it — and not the object model: a chat is one small dict, so there are no
channel versions, pending writes or history to keep (the "shallow" trade-off
``checkpoint/postgres/shallow.py`` makes). Their ``EncryptedSerializer`` (``serde/encrypted.py``,
AES with a key from the environment) is the reason the remote store here encrypts: once a chat's
questions leave the process they are personal data at rest.

**Two stores, chosen by where the bot runs** (:func:`from_env`):

- :class:`FileChatStore` — one JSON file per chat under ``ARGUS_CHAT_STORE`` (default
  ``~/.argus/chats``), written atomically. For the always-on bot, next to ``lui/watch.py``'s file.
- :class:`RestChatStore` — an Upstash Redis database over its REST API, for the webhook, whose
  serverless instances share no disk. Configured by ``UPSTASH_REDIS_REST_URL`` and
  ``UPSTASH_REDIS_REST_TOKEN`` (Upstash's own names, docs: upstash.com/docs/redis/quickstarts/
  vercel-python-runtime) or ``KV_REST_API_URL``/``KV_REST_API_TOKEN`` (the names Vercel's storage
  integration has used; NOT VERIFIED against a current integration — both are read). It refuses to
  start without ``ARGUS_CHAT_STATE_KEY``: records are AES-256-GCM encrypted with the record's key
  as associated data, so a record copied under another chat's key fails to open, and the key names
  themselves are an HMAC of the chat id, so the database never holds a chat id in the clear. Each
  record expires :data:`TTL_SECONDS` after its last write.

REST calls follow upstash.com/docs/redis/features/restapi: a POST of ``["SET", key, value, "EX",
ttl]`` with ``Authorization: Bearer <token>``, answered ``{"result": ...}`` or ``{"error": ...}``.

A store that cannot be read or written never stops an answer: a failed load is a fresh chat and a
failed save is logged. What is lost then is memory, never correctness.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol

from argus.truth import http

_LOG = logging.getLogger(__name__)

TTL_SECONDS = 30 * 24 * 3600
"""A chat nobody has written to for thirty days is forgotten by the remote store."""
KEY_ENV = "ARGUS_CHAT_STATE_KEY"
_NONCE = 12


class ChatStore(Protocol):
    def load(self, chat_id: int) -> dict[str, Any] | None: ...
    def save(self, chat_id: int, state: dict[str, Any]) -> None: ...
    def delete(self, chat_id: int) -> None: ...


class FileChatStore:
    """Each chat in its own JSON file, replaced whole so a crash mid-write leaves the old one."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(os.environ.get("ARGUS_CHAT_STORE", "")
                                 or Path.home() / ".argus" / "chats")

    def _path(self, chat_id: int) -> Path:
        return self.root / f"{int(chat_id)}.json"

    def load(self, chat_id: int) -> dict[str, Any] | None:
        try:
            state = json.loads(self._path(chat_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return state if isinstance(state, dict) else None

    def save(self, chat_id: int, state: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(state, out)
        os.replace(name, self._path(chat_id))

    def delete(self, chat_id: int) -> None:
        self._path(chat_id).unlink(missing_ok=True)


def parse_key(text: str) -> bytes:
    """The 32-byte key from its URL-safe base64 form. Make one with
    ``python -c "import os, base64; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"``."""
    try:
        key = base64.urlsafe_b64decode(text.strip().encode())
    except ValueError as exc:
        raise ValueError(f"{KEY_ENV} is not URL-safe base64") from exc
    if len(key) != 32:
        raise ValueError(f"{KEY_ENV} must decode to 32 bytes, not {len(key)}")
    return key


class RestChatStore:
    """Chats in Upstash Redis, encrypted, keyed by an HMAC of the chat id."""

    def __init__(self, url: str, token: str, key: bytes, *, timeout: float = 5.0) -> None:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        self.url = url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        self._aead = AESGCM(key)
        self._name_key = hmac.new(key, b"argus chat record name", hashlib.sha256).digest()
        self.timeout = timeout

    def _name(self, chat_id: int) -> str:
        digest = hmac.new(self._name_key, str(int(chat_id)).encode(), hashlib.sha256).hexdigest()
        return f"argus:chat:{digest}"

    def _command(self, *args: object) -> Any:
        body = http.fetch_json(self.url, data=json.dumps(list(args)).encode(), method="POST",
                               headers=self._headers, timeout=self.timeout)
        if not isinstance(body, dict):
            raise RuntimeError(f"Upstash answered {args[0]} with {type(body).__name__}, not JSON")
        if "error" in body:
            raise RuntimeError(f"Upstash refused {args[0]}: {body['error']}")
        return body.get("result")

    def load(self, chat_id: int) -> dict[str, Any] | None:
        name = self._name(chat_id)
        sealed = self._command("GET", name)
        if not isinstance(sealed, str):
            return None
        raw = base64.b64decode(sealed)
        plain = self._aead.decrypt(raw[:_NONCE], raw[_NONCE:], name.encode())
        state = json.loads(plain.decode("utf-8"))
        return state if isinstance(state, dict) else None

    def save(self, chat_id: int, state: dict[str, Any]) -> None:
        name = self._name(chat_id)
        nonce = os.urandom(_NONCE)
        sealed = nonce + self._aead.encrypt(nonce, json.dumps(state).encode(), name.encode())
        self._command("SET", name, base64.b64encode(sealed).decode(), "EX", TTL_SECONDS)

    def delete(self, chat_id: int) -> None:
        self._command("DEL", self._name(chat_id))


def from_env(*, remote: bool) -> ChatStore | None:
    """The store for this process. ``remote`` is the webhook's call: there only the REST store
    helps, and without one configured the answer is ``None`` (state held by the instance, as
    before). The always-on bot gets the file store. A REST store configured without a valid key,
    or without the ``cryptography`` package, is refused with a warning rather than run
    unencrypted."""
    url = os.environ.get("UPSTASH_REDIS_REST_URL") or os.environ.get("KV_REST_API_URL") or ""
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN") or os.environ.get("KV_REST_API_TOKEN") or ""
    if url and token:
        try:
            return RestChatStore(url, token, parse_key(os.environ.get(KEY_ENV, "")))
        except (ValueError, ImportError) as exc:
            _LOG.warning("chat store not used: %s", exc)
            return None
    return None if remote else FileChatStore()
