"""Anonymous usage events: how many people asked, what the console did, and whether it helped.

The submission form asks a research tool for "test users, task completion rate, usage data", and
the honest answer before this module was *none recorded*. This is the smallest record that answers
it without keeping anything a visitor typed:

* **One line per answered question** — which engine answered, whether it refused, how long it took,
  how many lines were marked `missing` — and **one line per "did this answer your question?"
  click**. The question, the book, the memory and the answer are never in the event.
* **Visitors are a daily hash**, ``sha256(salt | UTC day | address)`` cut to 12 hex characters. The
  same visitor on the same day counts once; nothing links two days, and without the salt (a server
  environment variable, never in the repository) the hash cannot be reversed by enumerating IPv4.
* **Our own traffic is marked, not trusted to be absent.** A browser that has opened the console
  with ``?internal=1`` sends that flag on every question, and a scripted client (curl, Python,
  headless Chrome) is classed ``automated`` from its User-Agent. The report counts only
  ``browser`` events without the flag as real use.

Events go to standard output as one JSON object with an ``argus_usage`` key. On the hosted console
that is Vercel's function log, which `eval/usage_report.py` pulls, de-duplicates and aggregates; the
logs are kept for a limited time, so the pull has to run while they are.

Taken from: Plausible's published visitor-counting method (its data policy: a hash of a daily
rotating salt with the address, no cookies, no stored identifiers) — the method only; Plausible's
code is AGPL and none of it is used. Not taken: its page-view model; the unit here is a question and
its outcome, because that is what "task completion" measures.

Also why the page sends questions by POST: the hosting runtime's access log records every GET
query string, so a question sent as ``/ask?q=…&memory=…`` put the visitor's words and their
"kept in this browser" memory into a log the console does not control (found reading that log on
2026-09-25 while building this module).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sys
from datetime import UTC, datetime
from typing import Any

_AUTOMATED = re.compile(r"curl|python|wget|httpx|aiohttp|go-http|headless|playwright|"
                        r"puppeteer|bot\b|crawler|spider|node-fetch|axios|postman", re.I)

_FALLBACK_SALT = secrets.token_hex(16)
"""Used when ``ARGUS_USAGE_SALT`` is unset (a local console). Random per process, so hashes from two
processes never match — visitors are then over-counted, never identifiable."""


def visitor_hash(address: str, now: datetime | None = None) -> str:
    salt = os.environ.get("ARGUS_USAGE_SALT", "").strip() or _FALLBACK_SALT
    day = (now or datetime.now(UTC)).strftime("%Y-%m-%d")
    return hashlib.sha256(f"{salt}|{day}|{address}".encode()).hexdigest()[:12]


def client_class(user_agent: str) -> str:
    """``browser`` or ``automated``, from the User-Agent alone. A missing one is automated."""
    ua = user_agent.strip()
    return "automated" if not ua or _AUTOMATED.search(ua) else "browser"


def answer_id() -> str:
    """The id a feedback click refers back to. Random: it names an answer, not a person."""
    return secrets.token_hex(6)


def emit(kind: str, **fields: Any) -> dict[str, Any]:
    """Write one event line to standard output and return it."""
    event = {"argus_usage": 1, "kind": kind,
             "at": datetime.now(UTC).isoformat(timespec="seconds"), **fields}
    sys.stdout.write(json.dumps(event, separators=(",", ":"), default=str) + "\n")
    sys.stdout.flush()
    return event


def ask_event(payload: dict[str, Any], *, answer: str, visitor: str, client: str,
              internal: bool) -> dict[str, Any]:
    """The event for one answered question — the outcome only, never its text."""
    labels = [str(x) for x in payload.get("line_labels") or [] if x]
    return emit(
        "ask", id=answer, v=visitor, client=client, internal=internal,
        route=str(payload.get("classified_by") or ""), intent=str(payload.get("intent") or ""),
        refused=bool(payload.get("refused")), ms=round(float(payload.get("elapsed_ms") or 0)),
        lines=len(payload.get("lines") or []), missing=labels.count("missing"),
        sources=len(payload.get("sources") or []), remembered=len(payload.get("remembered") or []),
        translated=bool(payload.get("translate")))


def feedback_event(answer: str, useful: bool, *, visitor: str, client: str,
                   internal: bool) -> dict[str, Any] | None:
    """The event for one "did this answer your question?" click, or None for a malformed id."""
    if not re.fullmatch(r"[0-9a-f]{12}", answer):
        return None
    return emit("feedback", id=answer, useful=useful, v=visitor, client=client,
                internal=internal)


__all__ = ["answer_id", "ask_event", "client_class", "emit", "feedback_event", "visitor_hash"]
