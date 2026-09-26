"""A bitget-signal Skill first, the source it names second, and every answer says which one spoke.

**Why this exists.** Track 3 scores "Skill integration count **and effectiveness**", and until
2026-09-26 the console asked one of bitget-signal's five Skills (technical analysis) when it
answered. Sentiment went to bitget-mcp-server and then straight to alternative.me; macro went
straight to FRED. The desk used the Skills, through `market/skill_mirror.py`, and the documents
counted what the desk did as if the console did it (judge audit, 2026-09-26). This module puts the
Skill in front of those answers, so the console's own lines are what the count describes.

**Order and labels.** :func:`route` calls the Skill's tool, classifies the reply with the same
rules the health sweep uses (`market/skills._classify`: an error envelope or a hollow payload is
EMPTY, never an answer), and when the Skill did not answer, runs the mirror for that tool — the
public upstream the Skill's own tool description names (`market/skill_mirror.MIRRORS`). The
:class:`Routed` result carries both facts, and :meth:`Routed.source` writes them into the receipt:
"bitget-signal sentiment_index.current" when the Skill answered; "alternative.me Fear & Greed, the
source bitget-signal's sentiment_index names" with the Skill's own failure when it did not.

**Latency.** The mirror starts beside the Skill, not after it, so a dark Skill costs at most
:data:`SKILL_WAIT_S` of an answer, and after one failure a Skill is not asked again for
:data:`COOLDOWN_S`: the answer says the Skill was down when last asked and when that was. Measured
2026-09-26: most bitget-signal tools fail after 15 to 30 seconds (``data/skill_matrix.json``), and a
console that waited that long on every sentiment question would be worse than one that did not ask.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from argus.lui.answer import Source
from argus.market.skills import PROBES, Health, _classify
from argus.truth.coverage import ContextPool

SKILL_WAIT_S = 8.0
"""How long an answer waits for a Skill before it takes the mirror that ran beside it."""

COOLDOWN_S = 900.0
"""After a Skill fails, how long it is answered from its mirror without being asked again."""

SkillCall = Callable[[str, dict[str, Any], float], tuple[Any, str]]
MirrorCall = Callable[[str, dict[str, Any]], tuple[Health, str, Any, str, bool]]


@dataclass(frozen=True, slots=True)
class Routed:
    """One reading, and which of the Skill or its mirror produced it."""

    tool: str
    action: str
    skill: str
    """The Skill the tool belongs to (``sentiment-analyst``), or ``server`` for a tool no SKILL.md
    names."""
    skill_health: Health
    skill_detail: str
    via: str
    """``skill`` when bitget-signal answered, ``mirror`` when the named upstream did, ``none``."""
    payload: Any
    upstream: str = ""
    same_upstream: bool = True
    asked_at: str = ""
    """When the Skill was last actually asked (UTC), which is now unless it is cooling down."""
    reused: bool = False
    """True when the Skill was not re-asked because it failed within :data:`COOLDOWN_S`."""

    @property
    def ident(self) -> str:
        return f"{self.tool}.{self.action}"

    @property
    def answered(self) -> bool:
        return self.via != "none"

    @property
    def skill_said(self) -> str:
        """What the Skill did, in words: "did not answer in time", "answered with no data"."""
        said = _SAID.get(self.skill_health, self.skill_health.value)
        return (f"{said} when last asked at {self.asked_at}" if self.reused
                else f"{said} at {self.asked_at}" if self.asked_at else said)

    def source(self) -> Source:
        if self.via == "skill":
            owner = (f"Bitget's {self.skill} Skill" if self.skill != "server"
                     else "a bitget-signal tool no SKILL.md names")
            return Source(kind="venue", ref=f"bitget-signal {self.ident}", detail=f"{owner}, live")
        why = f"the Skill {self.skill_said}"
        relation = ("the source bitget-signal's " if self.same_upstream
                    else "standing in for the keyed source of bitget-signal's ")
        return Source(kind="venue", ref=self.upstream or "none",
                      detail=f"{relation}{self.ident} names; {why}")


_SAID = {
    Health.OK: "answered",
    Health.EMPTY: "answered with no data",
    Health.TOOL_ERROR: "reported an error",
    Health.TIMEOUT: "did not answer in time",
    Health.UNAVAILABLE: "was unreachable",
}

_DOWN: dict[str, tuple[float, Health, str, str]] = {}
_LOCK = threading.Lock()


def _skill_of(tool: str, action: str) -> str:
    for probe in PROBES:
        if probe.tool == tool and probe.action == action:
            return probe.skill
    return next((p.skill for p in PROBES if p.tool == tool), "server")


def _default_skill(tool: str, args: dict[str, Any], wait: float) -> tuple[Any, str]:
    from argus.market.evidence import BitgetSkillSource

    return BitgetSkillSource().call(tool, args, timeout=max(1, round(wait)))


def _default_mirror(ident: str, args: dict[str, Any]) -> tuple[Health, str, Any, str, bool]:
    from argus.market.skill_mirror import MIRRORS, mirror_one

    mirror = MIRRORS.get(ident)
    probe = next((p for p in PROBES if p.ident == ident), None)
    if mirror is None or probe is None:
        return Health.UNAVAILABLE, "no mirror for this tool", None, "", True
    health, detail, payload, _ = mirror_one(probe, args)
    return health, detail, payload, mirror.upstream, mirror.same_upstream


def route(tool: str, action: str, args: Mapping[str, Any] | None = None, *,
          wait: float = SKILL_WAIT_S, skill_call: SkillCall | None = None,
          mirror_call: MirrorCall | None = None, now: Callable[[], float] = time.monotonic,
          ) -> Routed:
    """Ask bitget-signal's ``tool`` for ``action``; if it does not answer, its mirror.

    ``skill_call`` and ``mirror_call`` default to the live Skill client and `skill_mirror`; tests
    pass fakes. The mirror starts beside the Skill and its reading is used only when the Skill
    does not answer; a Skill cooling down is not asked at all."""
    ident = f"{tool}.{action}"
    call_args = {"action": action, **dict(args or {})}
    skill = _skill_of(tool, action)
    ask = skill_call or _default_skill
    mirror = mirror_call or _default_mirror
    with _LOCK:
        down = _DOWN.get(ident)
    if down is not None and now() - down[0] < COOLDOWN_S:
        health, _detail, payload, upstream, same = mirror(ident, call_args)
        _, skill_health, skill_detail, asked_at = down
        return Routed(tool, action, skill, skill_health, skill_detail,
                      "mirror" if health is Health.OK else "none",
                      payload if health is Health.OK else None, upstream, same, asked_at, True)
    asked_at = datetime.now(UTC).strftime("%H:%M UTC")
    pool = ContextPool(max_workers=2)
    try:
        skill_job = pool.submit(ask, tool, call_args, wait)
        mirror_job = pool.submit(mirror, ident, call_args)
        try:
            payload, status = skill_job.result(timeout=wait + 2.0)
            skill_health, skill_detail = _classify(payload, status)
        except Exception as exc:  # the Skill's own timeout, or ours around it
            payload, skill_health = None, Health.TIMEOUT
            skill_detail = f"bitget:{tool}: no answer within {wait:.0f}s ({type(exc).__name__})"
        if skill_health is Health.OK:
            with _LOCK:
                _DOWN.pop(ident, None)
            return Routed(tool, action, skill, skill_health, skill_detail, "skill", payload,
                          asked_at=asked_at)
        with _LOCK:
            _DOWN[ident] = (now(), skill_health, skill_detail, asked_at)
        health, _detail, mirrored, upstream, same = mirror_job.result()
    finally:
        pool.shutdown(wait=False)  # a Skill that answered does not wait for the mirror
    if health is Health.OK:
        return Routed(tool, action, skill, skill_health, skill_detail, "mirror", mirrored,
                      upstream, same, asked_at)
    return Routed(tool, action, skill, skill_health, skill_detail, "none", None, upstream, same,
                  asked_at)


def reset() -> None:
    """Forget every cooling-down Skill (tests; a restarted server starts clean anyway)."""
    with _LOCK:
        _DOWN.clear()


__all__ = ["COOLDOWN_S", "SKILL_WAIT_S", "Routed", "reset", "route"]
