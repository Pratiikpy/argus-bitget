"""Named risk modes and the precedence stack that picks one — the desk's posture, made legible.

Before this module the desk's posture was implicit. Whether it could open a position right now
depended on the circuit breaker's activation, on whether the venue's rules had loaded, and on the
fact that nothing in ARGUS routes to a live venue — three facts in three files, none of them stated
anywhere a reader could see at once. A judge asking "what is this desk allowed to do right now, and
who decided?" had to reconstruct it from code.

**Taken from letta-code** (Apache-2.0, ``letta-ai/letta-code``; attribution in
``licenses/letta_code-APACHE-2.0.txt``):

* **Named modes with one-line descriptions** — ``src/permissions/mode.ts:3-18`` defines
  ``standard`` / ``acceptEdits`` / ``unrestricted`` / ``strict``, and
  ``src/reminders/engine.ts:337-343`` gives each one sentence. Here: :class:`RiskMode`, four
  postures from the most permissive to the safest, each with the sentence
  :attr:`RiskMode.description` returns.
* **A fixed, documented precedence of stages** — ``src/permissions/checker.ts:88-105`` numbers its
  stages 0 to 10, with an unbypassable guard first. Here: :data:`PRECEDENCE`, four layers in a
  fixed order, the operator first.
* **A reminder that fires only on change** — ``buildPermissionModeReminder``
  (``src/reminders/engine.ts:345-377``) keeps the last mode it announced and emits a line only when
  the mode differs, and on the first turn only when the mode is not the default. Here:
  :class:`ModeNotifier`, one line, same rule; :func:`persisted_notice` keeps its state on disk,
  because a paper cycle is a fresh process every time and has no conversation to keep it in.

**Changed, deliberately: most restrictive wins, not first match.** letta-code resolves permissions
by first match: the earliest stage with an opinion decides, so an early *allow* beats a later
*deny*. For a risk layer that is the wrong composition — an operator typing "normal" must not be
able to override a breaker that has halted the book. So every layer states the mode it demands,
the **effective mode is the most restrictive of them**, and precedence only decides which layer is
named as the binding one when two demand the same mode. A lower layer can therefore never loosen a
higher one, and no layer can loosen any other: the stack only ever tightens. Lifting a halt is the
breaker's own stepwise transition (`risk/circuit.py`, ``Breaker.transition``), not a mode.

**Unknown resolves to halted.** An unreadable operator setting, an unrecognised breaker state or
venue rules that failed to load each demand :attr:`RiskMode.HALTED` — the same default
`risk/circuit.py` gives an unreadable activation, for the same reason: the case where the state is
lost is the case where the control is most needed.

**The mode gate agrees with the breaker by construction.** :func:`admit` refuses exactly what
``circuit.apply`` narrows — ``halted`` refuses trade and hedge, ``reduce_only`` refuses trade — and
narrows to the same fallback (reduce if positions are open, else no trade). What the mode adds is
the venue: anything that opens exposure goes to the paper venue unless the mode is ``normal``, and
``normal`` is reachable only by a posture that says live trading is enabled, which in ARGUS today
nothing does. A reduction is admitted in every mode and on either venue: a mode may never trap a
position.

`risk/certify.py` checks every resolution and admission against its own written table
(``MODE_SPEC``); `eval/risk_certify.py` runs that over every combination of layer demands,
verdicts, venues and open positions, and breaks this module on purpose to confirm the certifier
notices.

    python -m argus.risk.modes            # the stack as the next paper cycle would see it
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

from argus.execution.guard import (
    GUARD_GATES,
    Denial,
    GateVerdict,
    Guard,
    PermissionCheckTrace,
    Ruling,
    TraceEvent,
    not_reached,
)
from argus.risk.circuit import Activation, BookState, assess

OPERATOR_ENV = "ARGUS_RISK_MODE"
"""The operator's switch. Unset or empty is no demand; a mode name demands that mode; anything else
demands :attr:`RiskMode.HALTED`, because a typo in a kill switch must not read as "carry on"."""


class RiskMode(StrEnum):
    """What the desk may do, from the most permissive posture to the safest."""

    NORMAL = "normal"
    PAPER = "paper"
    REDUCE_ONLY = "reduce_only"
    HALTED = "halted"

    @property
    def rank(self) -> int:
        """Restrictiveness. The effective mode is the highest rank any layer demands."""
        return _RANK[self]

    @property
    def description(self) -> str:
        return _DESCRIPTIONS[self]


_RANK = {RiskMode.NORMAL: 0, RiskMode.PAPER: 1, RiskMode.REDUCE_ONLY: 2, RiskMode.HALTED: 3}

_DESCRIPTIONS = {
    RiskMode.NORMAL: "Every gate applies; orders may open, add, hedge or reduce on the live venue.",
    RiskMode.PAPER: "Orders may open, add, hedge or reduce, on the paper venue only.",
    RiskMode.REDUCE_ONLY: "No new directional risk: hedges and reductions only, on paper.",
    RiskMode.HALTED: "Nothing that opens exposure, hedges included; reductions only.",
}

DEFAULT_MODE = RiskMode.PAPER
"""ARGUS's baseline posture. Nothing in this repository routes an order to a live venue."""


class Layer(StrEnum):
    """Who may demand a mode. :data:`PRECEDENCE` is their order."""

    OPERATOR = "operator"
    VENUE = "venue"
    CIRCUIT = "circuit"
    POSTURE = "posture"


PRECEDENCE: tuple[Layer, ...] = (Layer.OPERATOR, Layer.VENUE, Layer.CIRCUIT, Layer.POSTURE)
"""The operator first and unbypassable, the configured posture last. Order names the binding layer
when two demand the same mode; it never lets one loosen another."""


@dataclass(frozen=True, slots=True)
class Demand:
    """One layer's demand: a mode, or ``None`` for no restriction from this layer, and why."""

    layer: Layer
    mode: RiskMode | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"layer": str(self.layer), "mode": None if self.mode is None else str(self.mode),
                "reason": self.reason}


class ModeError(ValueError):
    """A malformed stack. Raised rather than resolved, because a guess here is a posture."""


@dataclass(frozen=True, slots=True)
class ModeStack:
    """Every layer's demand, in :data:`PRECEDENCE` order, and the mode they resolve to."""

    demands: tuple[Demand, ...]

    def __post_init__(self) -> None:
        # letta-code's assertSharedReminderCoverage (engine.ts:510-527) checks at start-up that the
        # catalogue and its providers match one to one. The same structural check here: every
        # layer exactly once, in order, and a posture that always states a mode.
        layers = tuple(demand.layer for demand in self.demands)
        if layers != PRECEDENCE:
            raise ModeError(f"a mode stack needs {[str(x) for x in PRECEDENCE]} in that order, "
                            f"got {[str(x) for x in layers]}")
        if self.demands[-1].mode is None:
            raise ModeError("the posture layer must demand a mode; the desk has no baseline")

    @property
    def effective(self) -> RiskMode:
        return max((d.mode for d in self.demands if d.mode is not None), key=lambda m: m.rank)

    @property
    def binding(self) -> Demand:
        """The first layer, in precedence order, demanding the effective mode."""
        mode = self.effective
        return next(d for d in self.demands if d.mode is mode)

    def headline(self) -> str:
        binding = self.binding
        return f"risk mode {self.effective}, set by {binding.layer}: {binding.reason}"

    def render(self) -> str:
        lines = [f"[mode] {self.headline()}", f"  {self.effective.description}"]
        binding = self.binding
        for demand in self.demands:
            marker = "*" if demand is binding else " "
            asks = "no demand" if demand.mode is None else f"demands {demand.mode}"
            lines.append(f"  {marker} {demand.layer:<9} {asks:<20} {demand.reason}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "effective": str(self.effective), "binding": str(self.binding.layer),
            "description": self.effective.description,
            "demands": [d.as_dict() for d in self.demands],
        }


def resolve(demands: Sequence[Demand]) -> ModeStack:
    """Order the demands by :data:`PRECEDENCE` and resolve them. A layer may appear only once."""
    by_layer: dict[Layer, Demand] = {}
    for demand in demands:
        if demand.layer in by_layer:
            raise ModeError(f"layer {demand.layer} demanded twice")
        by_layer[demand.layer] = demand
    ordered = tuple(
        by_layer.get(layer, Demand(layer, None, "no demand")) for layer in PRECEDENCE
    )
    return ModeStack(ordered)


# --- the four layers ------------------------------------------------------------------------------


def operator_demand(environ: Mapping[str, str] | None = None) -> Demand:
    """Read the operator's switch, :data:`OPERATOR_ENV`."""
    raw = (os.environ if environ is None else environ).get(OPERATOR_ENV, "").strip()
    if not raw:
        return Demand(Layer.OPERATOR, None, "no operator override")
    try:
        mode = RiskMode(raw.lower())
    except ValueError:
        return Demand(Layer.OPERATOR, RiskMode.HALTED,
                      f"{OPERATOR_ENV}={raw!r} is not a mode; an unreadable switch halts")
    return Demand(Layer.OPERATOR, mode, f"{OPERATOR_ENV}={mode}")


def venue_demand(*, rules_loaded: bool, detail: str = "") -> Demand:
    """Venue rules that did not load mean no order can be validated, so nothing may be sent."""
    if rules_loaded:
        return Demand(Layer.VENUE, None, detail or "venue rules loaded")
    return Demand(Layer.VENUE, RiskMode.HALTED,
                  "venue rules unavailable, so no order can be validated"
                  + (f" ({detail})" if detail else ""))


def circuit_demand(activation: str, trips: Sequence[str] = ()) -> Demand:
    """The circuit breaker's activation, as a demand. An unrecognised state halts."""
    because = ", ".join(trips) or "no rule fired"
    try:
        state = Activation(str(activation))
    except ValueError:
        return Demand(Layer.CIRCUIT, RiskMode.HALTED,
                      f"unrecognised breaker state {activation!r}; unknown resolves to halted")
    if state is Activation.ACTIVE:
        return Demand(Layer.CIRCUIT, None, "breaker active: no rule fired")
    mode = RiskMode.HALTED if state is Activation.HALTED else RiskMode.REDUCE_ONLY
    return Demand(Layer.CIRCUIT, mode, f"breaker {state}: {because}")


def posture_demand(*, live_enabled: bool = False) -> Demand:
    """The configured baseline. Live only when the caller states it; ARGUS never does."""
    if live_enabled:
        return Demand(Layer.POSTURE, RiskMode.NORMAL, "live trading enabled by configuration")
    return Demand(Layer.POSTURE, RiskMode.PAPER, "paper desk: no live venue is configured")


def stack_for_cycle(
    *, rules_loaded: bool, rules_detail: str = "", book: BookState | None,
    environ: Mapping[str, str] | None = None, live_enabled: bool = False,
) -> ModeStack:
    """The stack a paper cycle runs under, from facts the cycle already has.

    ``book`` is the book the breaker judges (`paper/runner.py` ``_book_state``). ``None`` means the
    book could not be read, which demands a halt rather than a pass.
    """
    if book is None:
        breaker = Demand(Layer.CIRCUIT, RiskMode.HALTED,
                         "the book could not be read, so the breaker cannot judge it")
    else:
        activation, trips = assess(book)
        breaker = circuit_demand(activation, [f"{t.rule} ({t.detail})" for t in trips])
    return resolve((
        operator_demand(environ),
        venue_demand(rules_loaded=rules_loaded, detail=rules_detail),
        breaker,
        posture_demand(live_enabled=live_enabled),
    ))


# --- the mode gate --------------------------------------------------------------------------------

_CARRIES = frozenset({"trade", "hedge", "reduce"})
_VENUES = frozenset({"paper", "live"})


@dataclass(frozen=True, slots=True)
class Admission:
    """The mode gate's ruling on one order."""

    mode: RiskMode
    layer: Layer
    verdict: str
    venue: str
    open_positions: int
    admitted: bool
    narrowed_to: str | None
    """For a refused trade or hedge, what the breaker's narrowing would leave: the same fallback
    `risk/circuit.py` ``apply`` uses."""

    reason: str

    def event(self) -> TraceEvent:
        """This ruling as the first event of the order's :class:`PermissionCheckTrace`."""
        given = (f"mode {self.mode} (set by {self.layer}), verdict {self.verdict}, "
                 f"venue {self.venue}, open positions {self.open_positions}")
        return TraceEvent("risk_mode", given,
                          GateVerdict.PASS if self.admitted else GateVerdict.DENY, self.reason)

    def as_dict(self) -> dict[str, Any]:
        return {"mode": str(self.mode), "layer": str(self.layer), "verdict": self.verdict,
                "venue": self.venue, "open_positions": self.open_positions,
                "admitted": self.admitted, "narrowed_to": self.narrowed_to,
                "reason": self.reason}


def admit(stack: ModeStack, *, verdict: str, venue: str = "paper",
          open_positions: int = 0) -> Admission:
    """May an order with this verdict go to this venue under the stack's effective mode?"""
    mode, layer = stack.effective, stack.binding.layer
    kind = str(verdict).lower()
    place = str(venue).lower()

    def ruling(admitted: bool, reason: str, narrowed: str | None = None) -> Admission:
        return Admission(mode, layer, kind, place, open_positions, admitted, narrowed, reason)

    if kind not in _CARRIES:
        return ruling(False, f"verdict {kind} carries no quantity; there is no order to admit")
    if place not in _VENUES:
        return ruling(False, f"venue {venue!r} is neither paper nor live")
    if kind == "reduce":
        return ruling(True, f"{mode} admits reductions on any venue; a mode never traps a "
                            f"position")
    if place == "live" and mode is not RiskMode.NORMAL:
        return ruling(False, f"{mode} admits orders that open exposure on the paper venue only")
    fallback = "reduce" if open_positions > 0 else "no_trade"
    if mode is RiskMode.HALTED:
        return ruling(False, f"halted refuses {kind}: nothing that opens exposure", fallback)
    if mode is RiskMode.REDUCE_ONLY and kind == "trade":
        return ruling(False, "reduce_only refuses new directional risk", fallback)
    return ruling(True, f"{mode} admits {kind} on the {place} venue")


def check_order(
    guard: Guard, stack: ModeStack, *, symbol: str, verdict: str, quantity: Decimal, side: str,
    venue: str = "paper", open_positions: int = 0, price: Decimal | None = None,
    reference_price: Decimal | None = None, available_balance: Decimal | None = None,
    now: float | None = None,
) -> Ruling:
    """The mode gate, then the venue gates: one :class:`Ruling` and one trace for the whole path.

    A refusal by the mode spends no rate-window slot and lists every venue gate as not reached.
    """
    admission = admit(stack, verdict=verdict, venue=venue, open_positions=open_positions)
    event = admission.event()
    if not admission.admitted:
        return Ruling(
            Denial.RISK_MODE, admission.reason, Decimal("0"), price, False,
            PermissionCheckTrace((event, *not_reached(GUARD_GATES, "risk_mode"))),
        )
    ruling = guard.check(
        symbol, quantity=quantity, price=price, reference_price=reference_price, side=side,
        available_balance=available_balance, now=now,
    )
    return replace(ruling, trace=ruling.trace.prepend(event))


# --- the change notice ----------------------------------------------------------------------------


@dataclass
class ModeNotifier:
    """One line when the mode changes, and silence when it does not.

    After letta-code's ``buildPermissionModeReminder`` (``src/reminders/engine.ts:345-377``): the
    last announced mode is kept; the first observation is announced only when it is not the
    default; every later one only when the mode differs. Hold one per conversation or per cycle log.
    """

    last_notified: RiskMode | None = None
    default: RiskMode = DEFAULT_MODE

    def notice(self, stack: ModeStack) -> str | None:
        current = stack.effective
        previous = self.last_notified
        self.last_notified = current
        if previous is None and current is self.default:
            return None
        if previous is current:
            return None
        binding = stack.binding
        if previous is None:
            return (f"Risk mode active: {current}. {current.description} "
                    f"Set by {binding.layer}: {binding.reason}.")
        return (f"Risk mode changed to {current} (was {previous}). {current.description} "
                f"Set by {binding.layer}: {binding.reason}.")

    def as_dict(self) -> dict[str, Any]:
        return {"last_notified": None if self.last_notified is None else str(self.last_notified),
                "default": str(self.default)}

    @classmethod
    def from_dict(cls, blob: Mapping[str, Any]) -> ModeNotifier:
        """Restore a notifier. An unreadable last mode restores as never-notified, so the next
        stack is announced rather than silently assumed unchanged."""
        try:
            last = None if blob.get("last_notified") is None else RiskMode(blob["last_notified"])
        except ValueError:
            last = None
        try:
            default = RiskMode(blob.get("default", DEFAULT_MODE))
        except ValueError:
            default = DEFAULT_MODE
        return cls(last_notified=last, default=default)


NOTICE_STATE = Path(__file__).resolve().parents[3] / "data" / "risk_mode_notice.json"
"""Where the paper cycle keeps the last mode it announced, so a notice fires once per change across
cycles rather than once per process. letta-code keeps ``lastNotifiedPermissionMode`` in the
conversation's reminder state (``src/reminders/engine.ts:349,362``); a cycle that runs every two
hours in a fresh process has no such state, so it lives on disk."""


def persisted_notice(stack: ModeStack, path: Path = NOTICE_STATE) -> str | None:
    """:meth:`ModeNotifier.notice` with the notifier's state read from and written back to ``path``.

    A missing, unreadable or malformed state file restores as never-notified, so the stack is
    announced rather than assumed unchanged — silence is the claim that needs the evidence. The new
    state is written to a sibling file and moved into place, so an interrupted write leaves either
    the old state or the new one and never half of either.
    """
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        blob = {}
    notifier = ModeNotifier.from_dict(blob if isinstance(blob, dict) else {})
    notice = notifier.notice(stack)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f"{path.name}.tmp")
    staged.write_text(json.dumps(notifier.as_dict(), indent=2) + "\n", encoding="utf-8")
    os.replace(staged, path)
    return notice


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="the risk-mode stack the next cycle would see")
    parser.add_argument("--rules-unavailable", action="store_true",
                        help="show the stack as if the venue rules had failed to load")
    args = parser.parse_args(argv)
    from argus.paper.ledger import PaperLedger
    from argus.paper.runner import LEDGER_PATH, _book_state

    book = _book_state(PaperLedger(LEDGER_PATH))
    stack = stack_for_cycle(
        rules_loaded=not args.rules_unavailable, book=book if isinstance(book, BookState) else None,
    )
    print(stack.render())
    notice = ModeNotifier().notice(stack)
    print(notice or f"(no notice: {stack.effective} is the default posture)")
    return 0


__all__ = [
    "DEFAULT_MODE",
    "NOTICE_STATE",
    "OPERATOR_ENV",
    "PRECEDENCE",
    "Admission",
    "Demand",
    "Layer",
    "ModeError",
    "ModeNotifier",
    "ModeStack",
    "RiskMode",
    "admit",
    "check_order",
    "circuit_demand",
    "operator_demand",
    "persisted_notice",
    "posture_demand",
    "resolve",
    "stack_for_cycle",
    "venue_demand",
]


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
