"""Run your own ARGUS in Telegram: your bot, your token, your machine.

The hosted console answers at @argusbitgetbot for anyone who wants to try it. A trader who wants
the desk as their own — alerts that keep running, a book and memory that stay on their hardware, a
bot nobody else can use — runs it themselves, the way OpenClaw and Hermes Agent are run (read
2026-09-27: ``docs.openclaw.ai/channels/telegram``, its setup and access-control pages). Three
commands:

    argus-bot setup --token <token from @BotFather>     # checks the token, writes ~/.argus/bot.json
    argus-bot run                                        # long polling; no public URL, no server
    argus-bot pairing approve <CODE>                     # let someone in (the first is the owner)

**Access, as OpenClaw does it.** ``pairing`` (the default): a stranger who messages the bot gets
their Telegram user id and a code that expires in an hour, and nothing else; the owner approves the
code from the terminal, or with ``/approve CODE`` in their own chat, and the first approval makes
that person the owner. ``allowlist``: only the numeric user ids given with ``--allow``. ``open``: a
public bot, which is how @argusbitgetbot runs for judges; chosen explicitly, never by default.

**What stays local.** The token, the approved ids, each chat's saved book, what it told the desk
and its watches are kept under ``ARGUS_HOME`` (default ``~/.argus``), written atomically and
readable by the owner only where the system allows it. Questions are answered by this machine's own
copy of the desk (`server.handle_ask`); no ARGUS server is involved. A language-model key is
optional: without one the desk reads questions with its own classifier and patterns.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import secrets
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

POLICIES = ("pairing", "allowlist", "open")
PAIRING_SECONDS = 3600.0
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
"""No 0/O or 1/I: a code read off a screen and typed into a terminal."""


def home() -> Path:
    return Path(os.environ.get("ARGUS_HOME", "") or Path.home() / ".argus")


@dataclass
class BotConfig:
    token: str
    username: str = ""
    policy: str = "pairing"
    allow_from: list[int] = field(default_factory=list)
    owner: int | None = None
    pending: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Pairing code -> ``{"user_id", "name", "at"}``."""

    @classmethod
    def load(cls, path: Path | None = None) -> BotConfig | None:
        target = path or home() / "bot.json"
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            return cls(token=token) if token else None
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: Path | None = None) -> None:
        target = path or home() / "bot.json"
        _atomic_write(target, json.dumps(dataclasses.asdict(self), indent=1))


def _atomic_write(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        out.write(text)
    os.replace(name, target)
    # a system without POSIX modes keeps the file in the user's own home directory
    with contextlib.suppress(OSError):
        os.chmod(target, 0o600)


# --- access ---------------------------------------------------------------------------------------

def _prune(config: BotConfig, now: float) -> None:
    config.pending = {code: row for code, row in config.pending.items()
                      if now - float(row.get("at", 0)) < PAIRING_SECONDS}


def admit(config: BotConfig, user_id: int, name: str, now: float) -> str | None:
    """None when ``user_id`` may use the bot; otherwise the one reply a refused sender gets."""
    if config.policy == "open" or user_id in config.allow_from or user_id == config.owner:
        return None
    if config.policy == "allowlist":
        return ("This is a private ARGUS desk. Your Telegram user id is "
                f"{user_id}; its owner can add it with: argus-bot allow {user_id}")
    _prune(config, now)
    code = next((c for c, row in config.pending.items() if row.get("user_id") == user_id), None)
    if code is None:
        code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(6))
        config.pending[code] = {"user_id": user_id, "name": name[:64], "at": now}
    return (f"This is a private ARGUS desk. Your Telegram user id is {user_id}; your pairing "
            f"code is {code}, valid for an hour. The owner lets you in with: "
            f"argus-bot pairing approve {code}")


def approve(config: BotConfig, code: str, now: float) -> str:
    """Admit the sender a pairing code names; the first one admitted becomes the owner."""
    _prune(config, now)
    row = config.pending.pop(code.strip().upper(), None)
    if row is None:
        return f"No pairing request with code {code.strip().upper()} (codes expire after an hour)."
    user_id = int(row["user_id"])
    if user_id not in config.allow_from:
        config.allow_from.append(user_id)
    owner_note = ""
    if config.owner is None:
        config.owner = user_id
        owner_note = " as the owner"
    return f"Approved {row.get('name') or user_id} ({user_id}){owner_note}."


# --- chats kept across restarts -------------------------------------------------------------------

def load_states(path: Path | None = None) -> dict[int, Any]:
    from argus.lui.telegram_bot import ChatState

    target = path or home() / "chats.json"
    try:
        rows = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[int, Any] = {}
    for chat, row in rows.items() if isinstance(rows, dict) else []:
        try:
            out[int(chat)] = ChatState(turns=list(row.get("turns") or []),
                                       book=str(row.get("book") or ""),
                                       memory=str(row.get("memory") or ""))
        except (TypeError, ValueError):
            continue
    return out


def save_states(states: dict[int, Any], path: Path | None = None) -> None:
    target = path or home() / "chats.json"
    _atomic_write(target, json.dumps({str(chat): {"turns": s.turns, "book": s.book,
                                                  "memory": s.memory}
                                      for chat, s in states.items()}, ensure_ascii=False))


# --- one update -----------------------------------------------------------------------------------

Handle = Callable[..., list[tuple[int, str]]]


def handle(update: dict[str, Any], config: BotConfig, states: dict[int, Any], *,
           watches: Any, ask: Any = None, now: float | None = None,
           respond: Handle | None = None) -> list[tuple[int, str]]:
    """Replies for one update under the bot's access policy. ``/approve CODE`` from the owner
    admits a pending sender without leaving Telegram."""
    from argus.lui import telegram_bot

    clock = time.time() if now is None else now
    message = update.get("message") or {}
    sender = message.get("from") or {}
    chat_id = (message.get("chat") or {}).get("id")
    user_id = sender.get("id")
    text = str(message.get("text") or "").strip()
    if not isinstance(chat_id, int) or not isinstance(user_id, int) or not text:
        return []
    if text.split(" ", 1)[0].split("@", 1)[0].lower() == "/approve":
        if user_id != config.owner:
            return [(chat_id, "Only this desk's owner can approve pairing requests.")]
        return [(chat_id, approve(config, text.partition(" ")[2], clock))]
    refused = admit(config, user_id, str(sender.get("first_name") or ""), clock)
    if refused is not None:
        return [(chat_id, refused)]
    respond = respond or telegram_bot.handle_update
    kwargs: dict[str, Any] = {"now": clock, "watches": watches}
    if ask is not None:
        kwargs["ask"] = ask
    return respond(update, states, **kwargs)


# --- commands -------------------------------------------------------------------------------------

def _get_me(token: str) -> dict[str, Any]:
    from argus.lui.telegram_bot import _call

    me: dict[str, Any] = _call(token, "getMe", {}, timeout=20.0)
    return me


def setup(token: str, policy: str, allow: list[int],
          get_me: Callable[[str], dict[str, Any]] = _get_me) -> str:
    if policy not in POLICIES:
        return f"policy must be one of {', '.join(POLICIES)}"
    if policy == "allowlist" and not allow:
        return "an allowlist bot needs at least one numeric user id: --allow 123456789"
    try:
        me = get_me(token)
    except Exception as exc:
        return f"Telegram refused that token ({type(exc).__name__}); copy it again from @BotFather"
    config = BotConfig.load() or BotConfig(token=token)
    config.token, config.username, config.policy = token, str(me.get("username") or ""), policy
    config.allow_from = sorted(set(config.allow_from) | set(allow))
    config.save()
    return (f"Ready: https://t.me/{config.username} ({policy}). Start it with: argus-bot run"
            + (" — then message it and approve your own code; you become the owner."
               if policy == "pairing" and config.owner is None else ""))


def run(config: BotConfig) -> None:  # pragma: no cover - network loop
    from argus.lui import telegram_bot
    from argus.lui.watch import SWEEP_SECONDS, WatchStore

    token = config.token
    telegram_bot._call(token, "deleteWebhook", {"drop_pending_updates": False})
    telegram_bot._call(token, "setMyCommands", {"commands": [
        {"command": "help", "description": "what the desk answers"},
        {"command": "book", "description": "save or show your holdings"},
        {"command": "watch", "description": "a message when a level trades"},
        {"command": "watches", "description": "list your watches"},
        {"command": "clear", "description": "forget this chat's history and book"}]})
    states = load_states()
    store = WatchStore(home() / "watches.json")
    swept, offset = 0.0, 0
    print(f"ARGUS is answering at https://t.me/{config.username} ({config.policy}); "
          f"state in {home()}. Ctrl+C stops it.", flush=True)
    while True:
        if time.monotonic() - swept >= SWEEP_SECONDS:
            swept = time.monotonic()
            telegram_bot._sweep_watches(token, store)
        try:
            updates = telegram_bot._call(token, "getUpdates", {
                "offset": offset, "timeout": 50, "allowed_updates": ["message"]})
        except Exception as exc:
            print(f"getUpdates failed ({type(exc).__name__}); retrying in 5s", flush=True)
            time.sleep(5)
            continue
        for update in updates:
            offset = max(offset, int(update["update_id"]) + 1)
            for chat_id, text in handle(update, config, states, watches=store):
                try:
                    message_id = telegram_bot.send(token, chat_id, text)
                    telegram_bot.follow_up(token, chat_id, message_id)
                except Exception as exc:
                    print(f"sendMessage failed ({type(exc).__name__})", flush=True)
            config.save()
            save_states(states)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(prog="argus-bot",
                                     description="run your own ARGUS desk in Telegram")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("setup", help="check a @BotFather token and save it locally")
    s.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    s.add_argument("--policy", choices=POLICIES, default="pairing")
    s.add_argument("--allow", type=int, action="append", default=[])
    sub.add_parser("run", help="answer by long polling")
    st = sub.add_parser("status", help="the saved configuration, and the token checked live")
    st.add_argument("--probe", action="store_true")
    p = sub.add_parser("pairing", help="list or approve pairing requests")
    p.add_argument("action", choices=("list", "approve"))
    p.add_argument("code", nargs="?")
    a = sub.add_parser("allow", help="admit a numeric Telegram user id")
    a.add_argument("user_id", type=int)
    args = parser.parse_args(argv)

    if args.command == "setup":
        if not args.token:
            print("Give the token from @BotFather: argus-bot setup --token 123:abc")
            return 2
        print(setup(args.token, args.policy, args.allow))
        return 0
    config = BotConfig.load()
    if config is None or not config.token:
        print("No bot configured: argus-bot setup --token <token from @BotFather>")
        return 2
    if args.command == "run":
        run(config)
        return 0
    if args.command == "status":
        print(f"bot @{config.username or '?'} · policy {config.policy} · owner "
              f"{config.owner or 'none yet'} · {len(config.allow_from)} admitted · "
              f"{len(config.pending)} pending · state in {home()}")
        if args.probe:
            try:
                me = _get_me(config.token)
                print(f"token: live, @{me.get('username')}")
            except Exception as exc:
                print(f"token: refused ({type(exc).__name__})")
                return 1
        return 0
    if args.command == "allow":
        if args.user_id not in config.allow_from:
            config.allow_from.append(args.user_id)
        config.save()
        print(f"Admitted {args.user_id}.")
        return 0
    now = time.time()
    if args.action == "list":
        _prune(config, now)
        for code, row in config.pending.items():
            print(f"{code}  {row.get('name') or '-'}  user {row['user_id']}  "
                  f"{int((now - float(row['at'])) // 60)} min ago")
        if not config.pending:
            print("No pairing requests.")
        config.save()
        return 0
    if not args.code:
        print("Say which: argus-bot pairing approve <CODE>")
        return 2
    print(approve(config, args.code, now))
    config.save()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))


__all__ = ["POLICIES", "BotConfig", "admit", "approve", "handle", "home", "load_states", "main",
           "save_states", "setup"]
