# Vendored, verbatim, unmodified below the "VENDORED FROM HERE" marker.
#
# Source:  https://github.com/SerenityTn/serenity-guardrails
# Path:    etoro_trading/guards.py
# Commit:  13d46dc1c6204f27fe321bd66023691918017cce (2026-08-28)
# Licence: Apache License 2.0 (Copyright SerenityTn — see the sibling vendored
#          `serenity_journal.py`'s header for the full attribution note; the upstream LICENSE
#          file itself carries an unfilled template) — full text in that sibling file's header,
#          not repeated verbatim here to avoid a second full-license block for one small file from
#          the same repository and commit.
#
# `GuardError` is a dependency of the sibling vendored `serenity_journal.py`
# (`ChainedJournal.append`/`.verify` both raise it) — vendored alongside it, whole file rather than
# just the one class, so that file runs completely unmodified rather than needing this one name
# stubbed out, and because the file's other guards are genuinely part of the same "fail-closed
# operational guards" module the journal ships inside.
#
# ============================== VENDORED FROM HERE ==============================
"""Fail-closed operational guards.

Every guard here fails toward "no execution": an absent environment variable,
a missing state file, a corrupt record, or an unexpected type all resolve to
the most restrictive state. This closes the Gate 2 rejection class where an
absent ``ETORO_ORDER_TRANSMISSION`` or an uppercase ``FALSE`` was accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import json
import os


TRANSMISSION_ENV = "ETORO_ORDER_TRANSMISSION"


class GuardError(ValueError):
    """Stable rejection for any guard violation."""


def require_transmission_disabled() -> None:
    """The variable must be present and exactly ``false`` (case-sensitive).

    Absent, empty, ``FALSE``, ``0``, or any other value is a violation: an
    environment that cannot prove transmission is off must not run anything
    that could route toward execution.
    """
    if os.environ.get(TRANSMISSION_ENV) != "false":
        raise GuardError(f"{TRANSMISSION_ENV} must be present and exactly 'false'")


class ActivationState(Enum):
    HALTED = "HALTED"
    REDUCE_ONLY = "REDUCE_ONLY"
    ACTIVE = "ACTIVE"


_ALLOWED_TRANSITIONS = {
    ActivationState.HALTED: {ActivationState.REDUCE_ONLY},
    ActivationState.REDUCE_ONLY: {ActivationState.HALTED, ActivationState.ACTIVE},
    ActivationState.ACTIVE: {ActivationState.HALTED, ActivationState.REDUCE_ONLY},
}


@dataclass(frozen=True, slots=True)
class ActivationRecord:
    state: ActivationState
    changed_at: str
    reason: str
    policy_version: str


class ActivationStore:
    """Persisted promotion state machine. Missing or corrupt state is HALTED.

    Promotion above HALTED additionally requires an owner-authorization file
    whose ``policy_version`` matches the requested transition; halting never
    requires anything.
    """

    __slots__ = ("_path", "_owner_path")

    def __init__(self, path: str | Path, owner_authorization: str | Path) -> None:
        self._path = Path(path)
        self._owner_path = Path(owner_authorization)

    def read(self) -> ActivationRecord:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            state = ActivationState(payload["state"])
            record = ActivationRecord(
                state=state,
                changed_at=str(payload["changed_at"]),
                reason=str(payload["reason"]),
                policy_version=str(payload["policy_version"]),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return ActivationRecord(
                state=ActivationState.HALTED,
                changed_at=_utc_now(),
                reason="state file missing or unreadable; failing closed",
                policy_version="",
            )
        return record

    def transition(
        self,
        target: ActivationState,
        *,
        reason: str,
        policy_version: str,
    ) -> ActivationRecord:
        if type(target) is not ActivationState:
            raise GuardError("target state has an invalid runtime type")
        if type(reason) is not str or not reason.strip():
            raise GuardError("a transition requires a non-empty reason")
        current = self.read()
        if target is current.state:
            return current
        if target not in _ALLOWED_TRANSITIONS[current.state]:
            raise GuardError(
                f"transition {current.state.value} -> {target.value} is not allowed"
            )
        if target is not ActivationState.HALTED:
            self._require_owner_authorization(target, policy_version)
        record = ActivationRecord(
            state=target,
            changed_at=_utc_now(),
            reason=reason.strip(),
            policy_version=policy_version,
        )
        serialized = json.dumps(
            {
                "state": record.state.value,
                "changed_at": record.changed_at,
                "reason": record.reason,
                "policy_version": record.policy_version,
            },
            indent=2,
            sort_keys=True,
        )
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(serialized + "\n", encoding="utf-8")
        os.replace(tmp, self._path)
        return record

    def _require_owner_authorization(
        self, target: ActivationState, policy_version: str
    ) -> None:
        try:
            payload = json.loads(self._owner_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise GuardError(
                "owner authorization file is missing or unreadable; "
                "promotion above HALTED is blocked"
            ) from None
        if type(payload) is not dict:
            raise GuardError("owner authorization is malformed")
        states = payload.get("authorized_states")
        if (
            type(states) is not list
            or target.value not in states
            or payload.get("policy_version") != policy_version
            or type(payload.get("owner")) is not str
            or not payload["owner"].strip()
        ):
            raise GuardError(
                f"owner authorization does not cover {target.value} "
                f"at policy version {policy_version!r}"
            )


class KillSwitch:
    """File-based kill switch: if the file exists, everything halts.

    The check is existence-only so a partially written or empty file still
    kills. ``engage`` records who and why for the audit trail.
    """

    __slots__ = ("_path",)

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def engaged(self) -> bool:
        return self._path.exists()

    def require_clear(self) -> None:
        if self.engaged:
            raise GuardError("kill switch is engaged")

    def engage(self, *, by: str, reason: str) -> None:
        payload = json.dumps(
            {"engaged_at": _utc_now(), "by": by, "reason": reason},
            indent=2,
            sort_keys=True,
        )
        self._path.write_text(payload + "\n", encoding="utf-8")

    def clear(self, *, by: str) -> None:
        if type(by) is not str or not by.strip():
            raise GuardError("clearing the kill switch requires an operator name")
        if self._path.exists():
            self._path.unlink()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
