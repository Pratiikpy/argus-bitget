"""The consent gate: no order that can spend real money leaves ARGUS without a per-run token.

**The gap, found by reading the client rather than its docstring.** `execution/bitget_client.py`
says paper trading "is the default here and turning it off takes an explicit argument", and that is
true: ``BitgetTradingClient(paper_trading=False)`` is one keyword away, and nothing after it asked
again. The explicit argument was the whole safeguard, and an argument is exactly the thing a
refactor, a copied snippet or a config file sets without anybody deciding to spend money. Worse,
the client's own ``trades_real_money`` predicate was wrong in one corner: ``paper_trading=True``
with an explicit live ``product_type`` sent a real-money order while the predicate read ``False``
(fixed in the client on 2026-09-25, and now refused at construction).

What was taken, from which file, under which licence
----------------------------------------------------

MLE-bench (``openai/mle-bench``, MIT, Copyright (c) 2024 OpenAI; the MIT grant covers the code
only, not the competition data, none of which is used here; local clone
``mypr/06_agent_evaluation_and_benchmarks/mle_bench``):

- **A dangerous mode refuses unless consent is present in code, not in a README.**
  ``run_agent.py:97-104`` raises before any container starts when an agent needs a privileged
  container and ``I_ACCEPT_RUNNING_PRIVILEGED_CONTAINERS`` is not set to a true value. Adapted
  here as :func:`grant_live_consent` and :func:`require_live_consent`. Changed, deliberately, in
  three ways, each closing a hole the original leaves open for this use:

  1. **Bound to one run, not to a shell.** MLE-bench accepts ``true``; an ``export`` from last week
     would then authorise every run after it. Here the token must be the exact phrase
     :func:`consent_phrase` builds from *this* run's id, so a token for one run never matches
     another.
  2. **Spent on use.** A granted token is recorded in an append-only ledger and refused if
     presented again, so re-running the same command with the same environment does not
     re-authorise itself.
  3. **Expires.** A token is valid for :data:`DEFAULT_LIFETIME` from its grant. A run that is
     still sending real orders an hour after a human agreed to it is not the run they agreed to.

- **The capability cannot be forged.** :class:`LiveOrderConsent` is constructible only through
  :func:`grant_live_consent`, which holds the module-private sentinel — the same pattern
  :class:`argus.decision.verdicts.Authorised` uses (itself read from Ritapossible/Ballast's
  ``ballast/enforcer.py``). A caller cannot build a token by hand to get past the gate.

**The default is refusal.** No token, a token for another run, a spent token or an expired one all
raise :class:`ConsentRefused` before a byte reaches the venue. Paper and demo orders need no token:
the gate exists for the orders that cost money, and gating the ones that do not would teach
everyone to paste the phrase by reflex, which is the failure a consent gate must not create.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

CONSENT_ENV = "ARGUS_LIVE_ORDER_CONSENT"
"""Where :func:`grant_live_consent` looks for the token when none is passed explicitly."""

CONSENT_PREFIX = "I-ACCEPT-REAL-MONEY-ORDERS-FOR-RUN"
"""The fixed part of the phrase. Long and specific on purpose: nobody types it by accident."""

DEFAULT_LIFETIME = timedelta(minutes=15)
"""How long a grant stays valid. A policy choice, stated as one; long enough for one live run."""

SPENT_PATH = Path(__file__).resolve().parents[3] / "data" / "live_consent_spent.jsonl"
"""Append-only record of every run id whose consent was granted. A spent run id is refused."""

_GRANT = object()
"""The sentinel only :func:`grant_live_consent` holds."""


class ConsentRefused(RuntimeError):
    """A real-money order was asked for without a valid, unspent, unexpired consent for this run."""


def new_run_id(now: datetime | None = None) -> str:
    """A run id nobody could have prepared a token for in advance: timestamp plus 32 random bits."""
    at = now or datetime.now(UTC)
    return f"run-{at.strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(4)}"


def consent_phrase(run_id: str) -> str:
    """The exact token that consents to real-money orders for ``run_id`` and for nothing else."""
    if not run_id.strip() or any(c.isspace() for c in run_id):
        raise ConsentRefused(f"{run_id!r} is not a run id; a consent must name one exact run")
    return f"{CONSENT_PREFIX}:{run_id}"


class LiveOrderConsent:
    """Proof that a human consented to real-money orders for one run. Unconstructable by hand."""

    __slots__ = ("expires_at", "granted_at", "run_id")

    def __init__(
        self, run_id: str, granted_at: datetime, expires_at: datetime, *, _token: object
    ) -> None:
        if _token is not _GRANT:
            raise ConsentRefused(
                "a LiveOrderConsent cannot be constructed directly; it is minted only by "
                "grant_live_consent(), which is the whole point of the type"
            )
        self.run_id = run_id
        self.granted_at = granted_at
        self.expires_at = expires_at

    def __repr__(self) -> str:
        return f"LiveOrderConsent(run={self.run_id}, until={self.expires_at.isoformat()})"

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "granted_at": self.granted_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }


def _spent(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            out.add(str(json.loads(raw)["run_id"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            # A torn line records a grant that may or may not have committed. Its run id cannot be
            # read, so it cannot be honoured either; nothing is lost by skipping it.
            continue
    return out


def grant_live_consent(
    run_id: str,
    *,
    token: str | None = None,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
    lifetime: timedelta = DEFAULT_LIFETIME,
    spent_path: Path | None = None,
) -> LiveOrderConsent:
    """Mint the consent for ``run_id``, or refuse. Refusal is the default.

    ``token`` is the phrase a human supplied; when ``None`` it is read from :data:`CONSENT_ENV`.
    The run id is spent the moment the grant succeeds, so the same phrase never grants twice.
    """
    at = now or datetime.now(UTC)
    if at.tzinfo is None:
        raise ConsentRefused("a consent is never naive: the grant instant carries no timezone")
    if lifetime <= timedelta(0):
        raise ConsentRefused(f"a consent lifetime must be positive, got {lifetime}")
    expected = consent_phrase(run_id)
    source = os.environ if env is None else env
    supplied = (token if token is not None else source.get(CONSENT_ENV, "")).strip()
    if not supplied:
        raise ConsentRefused(
            f"real-money orders need an explicit consent for this run and none was given. To "
            f"consent, set {CONSENT_ENV} to exactly: {expected}"
        )
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise ConsentRefused(
            f"the consent supplied does not name run {run_id}; a token consents to one run only. "
            f"The phrase for this run is: {expected}"
        )
    ledger = SPENT_PATH if spent_path is None else spent_path
    if run_id in _spent(ledger):
        raise ConsentRefused(
            f"the consent for run {run_id} was already used; a token is spent on its first grant, "
            f"so start a new run and consent to that one"
        )
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "run_id": run_id, "granted_at": at.isoformat(), "pid": os.getpid(),
        }) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return LiveOrderConsent(run_id, at, at + lifetime, _token=_GRANT)


def require_live_consent(consent: LiveOrderConsent | None, *, now: datetime | None = None) -> None:
    """Raise unless ``consent`` is a real, unexpired grant. Called before every real-money order."""
    at = now or datetime.now(UTC)
    if consent is None:
        raise ConsentRefused(
            "refusing a real-money order: this client holds no consent. Paper and demo orders "
            "need none; a live order needs grant_live_consent() for this run"
        )
    if not isinstance(consent, LiveOrderConsent):  # pragma: no cover - structural guard
        raise ConsentRefused(f"{type(consent).__name__} is not a LiveOrderConsent")
    if at > consent.expires_at:
        raise ConsentRefused(
            f"the consent for run {consent.run_id} expired at {consent.expires_at.isoformat()}; "
            f"a run still sending real orders after that is not the run that was agreed to"
        )


__all__ = [
    "CONSENT_ENV",
    "CONSENT_PREFIX",
    "DEFAULT_LIFETIME",
    "SPENT_PATH",
    "ConsentRefused",
    "LiveOrderConsent",
    "consent_phrase",
    "grant_live_consent",
    "new_run_id",
    "require_live_consent",
]
