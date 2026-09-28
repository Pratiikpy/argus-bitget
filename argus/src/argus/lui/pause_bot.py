"""A held decision, answered from Telegram: approve, reject or half size, by a named reviewer.

**Why this exists.** `decision/pause.py` holds an escalated decision for a human and resumes it from
the answer, and until now the only way to answer was the command line on the desk's machine. A judge
watching the bot saw nothing when the desk escalated — an unexplained decision. Here the held
proposal goes to each registered reviewer as a message with three buttons, the tap is
recorded through the store's own :meth:`~argus.decision.pause.PauseStore.answer`, and the message
is edited to show what was recorded, so the chat itself is the audit trail
(research/harvest/49-telegram-deep.md, decision 1).

**What was taken, and from where.** Freqtrade's force-exit confirmation
(`freqtrade/rpc/telegram.py`, GPL-3.0, pattern only, nothing copied): one button per action, and
the confirmation written into the same message. Rook's callback routing
(`lib/telegram/keyboards.ts:115`, MIT): a short prefix naming the handler, then the action and the
subject. Rook's payload is unsigned because it has no server
secret binding a tap to an identity; this one is signed, because a tap here releases an order.

**Security, point by point** (the design the study's independent audit required before building):

* *Who answered* is ``callback_query.from.id``, which Telegram fills for whoever tapped and which
  no other user can forge, mapped to a reviewer name through ``TELEGRAM_PAUSE_REVIEWERS``
  (``"<telegram user id>:<name>,..."``). Nothing in the button's payload names a reviewer.
* *The payload* is ``pz:<a|r|h>:<token>:<tag>``: the action, a 10-character token standing for the
  request (an HMAC of its id, so a long request id still fits Telegram's 64-byte limit), and a
  16-character HMAC over action, request id and quantity under ``TELEGRAM_PAUSE_SECRET`` — a secret
  of its own, not the webhook's — checked with :func:`hmac.compare_digest`.
* *Who is asked*: only the chats of registered reviewers (a private chat's id is its user's id);
  never a chat that merely talks to the research bot.
* *Mandatory allowlist*: approvals refuse to start unless ``ARGUS_PAUSE_REVIEWERS`` is set and names
  every reviewer mapped here — unset means "any name accepted", safe behind a shell, not behind a
  button anyone in Telegram can press.
* *One path*: every answer goes through ``PauseStore.answer``, so the allowlist and its
  ``answer_refused`` event, the first-answer-wins claim file and the expiry check all apply
  unchanged. A second tap, a replayed update or a late one is refused by the store, not here.

**Where it runs.** In the always-on bot (``telegram_bot.poll``), on the machine that holds the
desk's pause store (``paper/runner.pause_root``). The serverless webhook has no such store, so it
offers no approvals rather than pretending to.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.decision.pause import (
    REVIEWERS_ENV,
    AlreadyResolved,
    HumanAction,
    HumanResponse,
    PauseError,
    PauseRequest,
    PauseStore,
    reviewer_allowlist,
)

REVIEWERS_BY_ID_ENV = "TELEGRAM_PAUSE_REVIEWERS"
SECRET_ENV = "TELEGRAM_PAUSE_SECRET"
PREFIX = "pz"
HALF = Decimal("0.5")
_ACTIONS = {"a": HumanAction.APPROVE, "r": HumanAction.REJECT, "h": HumanAction.MODIFY_SIZE}


class ApprovalConfigError(RuntimeError):
    """Telegram approvals were asked for but cannot be run safely as configured."""


@dataclass(frozen=True)
class Approvals:
    """Who may answer, and the key their buttons are signed with."""

    reviewers: dict[int, str]
    """Telegram user id -> reviewer name, every name on ``ARGUS_PAUSE_REVIEWERS``."""
    secret: bytes

    @classmethod
    def from_env(cls) -> Approvals | None:
        """``None`` when approvals are not configured at all; an error when half-configured or
        when the store's own allowlist would accept a name this map does not bind."""
        raw = os.environ.get(REVIEWERS_BY_ID_ENV, "").strip()
        secret = os.environ.get(SECRET_ENV, "").strip()
        if not raw and not secret:
            return None
        if not raw or not secret:
            missing = REVIEWERS_BY_ID_ENV if not raw else SECRET_ENV
            raise ApprovalConfigError(f"Telegram approvals need {missing} as well")
        if len(secret) < 32:
            raise ApprovalConfigError(f"{SECRET_ENV} must be at least 32 characters")
        reviewers: dict[int, str] = {}
        for item in raw.split(","):
            user, _, name = item.strip().partition(":")
            if not user.strip().lstrip("-").isdigit() or not name.strip():
                raise ApprovalConfigError(
                    f"{REVIEWERS_BY_ID_ENV} entries are <telegram user id>:<name>, got {item!r}")
            reviewers[int(user)] = name.strip()
        allowed = reviewer_allowlist()
        if allowed is None:
            raise ApprovalConfigError(
                f"{REVIEWERS_ENV} must be set before approvals are offered in Telegram: unset, the "
                f"store accepts any reviewer name, and a button anyone can press is not a shell")
        stray = sorted(set(reviewers.values()) - allowed)
        if stray:
            raise ApprovalConfigError(
                f"{', '.join(stray)} answer from Telegram but are not on {REVIEWERS_ENV}")
        return cls(reviewers=reviewers, secret=secret.encode())

    def token(self, request_id: str) -> str:
        return hmac.new(self.secret, f"id:{request_id}".encode(), hashlib.sha256).hexdigest()[:10]

    def tag(self, code: str, request_id: str, quantity: Decimal | None) -> str:
        message = f"{code}:{request_id}:{'' if quantity is None else quantity}".encode()
        return hmac.new(self.secret, message, hashlib.sha256).hexdigest()[:16]


def half_of(request: PauseRequest) -> Decimal:
    """Half the held quantity, written as a person writes it. ``normalize()`` alone turned 300
    into ``3E+2`` on the reviewer's button and confirmation (QA surfaces pass, 2026-09-28): a whole
    number keeps its digits, a fraction loses only its trailing zeros."""
    half = request.max_quantity * HALF
    return half.quantize(Decimal(1)) if half == half.to_integral_value() else half.normalize()


def _quantity(code: str, request: PauseRequest) -> Decimal | None:
    return half_of(request) if code == "h" else None


def callback_data(approvals: Approvals, request: PauseRequest, code: str) -> str:
    data = (f"{PREFIX}:{code}:{approvals.token(request.request_id)}:"
            f"{approvals.tag(code, request.request_id, _quantity(code, request))}")
    assert len(data.encode()) <= 64, data  # Telegram's limit; the format is fixed-width
    return data


def keyboard(approvals: Approvals, request: PauseRequest) -> dict[str, Any]:
    labels = {"a": f"Approve {request.max_quantity}", "r": "Reject",
              "h": f"Half size ({half_of(request)})"}
    offered = [c for c in ("a", "h", "r") if _ACTIONS[c] in request.allowed_actions]
    return {"inline_keyboard": [[{"text": labels[c], "callback_data": callback_data(
        approvals, request, c)} for c in offered]]}


def message_text(request: PauseRequest, footer: str = "") -> str:
    shown = request.render()
    if len(shown) > 3400:  # Telegram caps a message at 4096 characters, tags included
        shown = shown[:3400].rsplit("\n", 1)[0] + "\n  ... (the full request: python -m " \
            "argus.decision.pause list)"
    body = f"<b>A decision is held for you</b>\n<pre>{html.escape(shown)}</pre>"
    return body + (f"\n{footer}" if footer else "")


@dataclass(frozen=True)
class Outcome:
    """What one tap produced: the toast shown to the tapper, and the message edit, if any."""

    callback_id: str
    toast: str
    chat_id: int | None = None
    message_id: int | None = None
    text: str | None = None


def handle_callback(update: dict[str, Any], store: PauseStore, approvals: Approvals, *,
                    now: datetime | None = None) -> Outcome | None:
    """Answer one tap. ``None`` for an update that is not a pause button."""
    query = update.get("callback_query") or {}
    data = str(query.get("data") or "")
    if not data.startswith(f"{PREFIX}:"):
        return None
    clock = now or datetime.now(UTC)
    cid = str(query.get("id") or "")
    message = query.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    message_id = message.get("message_id")
    user_id = (query.get("from") or {}).get("id")
    reviewer = approvals.reviewers.get(user_id) if isinstance(user_id, int) else None
    if reviewer is None:
        return Outcome(cid, "You are not a reviewer for this desk; nothing was recorded.")
    parts = data.split(":")
    if len(parts) != 4 or parts[1] not in _ACTIONS:
        return Outcome(cid, "That button is not one this desk issued; nothing was recorded.")
    _, code, token, tag = parts
    request = next((store.load_request(rid) for rid in store.request_ids()
                    if hmac.compare_digest(approvals.token(rid), token)), None)
    if request is None:
        return Outcome(cid, "No held decision matches that button; nothing was recorded.")
    expected = approvals.tag(code, request.request_id, _quantity(code, request))
    if not hmac.compare_digest(expected, tag):
        return Outcome(cid, "That button's signature does not verify; nothing was recorded.")
    action = _ACTIONS[code]
    response = HumanResponse(request_id=request.request_id, action=action, reviewer=reviewer,
                             answered_at=clock, quantity=_quantity(code, request),
                             note=f"answered from Telegram by user {user_id}")
    try:
        store.answer(response, now=clock)
    except AlreadyResolved as exc:
        return Outcome(cid, f"Already answered: {exc}", chat_id, message_id,
                       message_text(request, f"<i>{html.escape(str(exc))}</i>"))
    except PauseError as exc:  # expired, or refused by the store's own checks
        return Outcome(cid, f"Not recorded: {exc}", chat_id, message_id,
                       message_text(request, f"<i>Not recorded: {html.escape(str(exc))}</i>"))
    what = {"a": f"approved at {request.max_quantity}", "r": "rejected",
            "h": f"approved at half size, {half_of(request)}"}[code]
    footer = (f"<b>{html.escape(reviewer)} {what}</b> at {clock.isoformat(timespec='seconds')}. "
              f"The desk resumes it on its next cycle.")
    return Outcome(cid, f"Recorded: {what}.", chat_id, message_id, message_text(request, footer))


def announced_path(store: PauseStore) -> Path:
    return store.root / "telegram_announced.json"


def announce(store: PauseStore, approvals: Approvals,
             send: Callable[[int, str, dict[str, Any]], Any], *,
             now: datetime | None = None) -> list[str]:
    """Send every pending request not yet sent to every reviewer; returns the ids sent. A request
    is marked sent only once every reviewer's send succeeded, so a failure is retried next sweep."""
    path = announced_path(store)
    try:
        done = set(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        done = set()
    sent: list[str] = []
    for request in store.pending(now=now or datetime.now(UTC)):
        if request.request_id in done:
            continue
        ok = True
        for user_id in approvals.reviewers:
            try:
                send(user_id, message_text(request), keyboard(approvals, request))
            except Exception:
                ok = False
        if ok:
            done.add(request.request_id)
            sent.append(request.request_id)
    if sent:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(done)), encoding="utf-8", newline="\n")
    return sent


__all__ = [
    "PREFIX",
    "REVIEWERS_BY_ID_ENV",
    "SECRET_ENV",
    "ApprovalConfigError",
    "Approvals",
    "Outcome",
    "announce",
    "callback_data",
    "half_of",
    "handle_callback",
    "keyboard",
    "message_text",
]
