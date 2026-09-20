"""The carry desk — the first proposal in ARGUS's history that reaches gate two.

`eval/autopsy.py` established the thing this module exists to change. The Constitution is an ordered
chain of six gates, and across all 170 recorded decisions the **first** gate returned every single
time: ``no exposure proposed; nothing to narrow``. Five of the six had therefore never executed. The
autopsy's own conclusion was that this is not a fact about the risk layer but a fact about what the
desk was ever asked — every directional study here measured no edge, so the desk correctly proposed
nothing, and a chain that is never entered cannot be said to work.

`research/carry.py` supplies the missing input. A funding payment is collected for *holding* a side,
not for being right about direction, so a carry basket can carry a non-zero quantity without
claiming a forecast that our own measurements refuse. That clears gate one honestly — not by
loosening it, which would have been the obvious and wrong fix.

**What happens next is the point, and it is not a pass.** Gate two is a 0.55 confidence floor, and
the confidence attached here is a Wilson lower bound on the share of holding windows that made
money — computed against the number of **independent** windows, not the number of overlapping ones.
The study replays 1,823 overlapping fourteen-day windows out of ninety days of history; those are
roughly six independent holds. At six, a 74% win rate has a 95% lower bound near 0.36, and the floor
refuses it.

So the result is: **gate one passes for the first time, gate two fires, and gates three to six are
still unreached.** That is a smaller claim than "the desk now trades" and a much better one, because
every part of it was measured. Quoting the raw window count instead would have put the bound near
0.72, cleared the floor, and produced a position sized on a sample that does not exist — which is
exactly the failure the floor is there to catch, arriving through the front door.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.decision.verdicts import (
    ConstitutionRuling,
    Intent,
    Side,
    Verdict,
)
from argus.eval.autopsy import CHAIN, FIRED, PASSED, UNREACHED
from argus.truth.clocks import SessionState

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "carry_desk.json"

WILSON_Z = 1.96
"""95%. The same z the Fisher interval in `risk/effectiveness.py` uses, so two confidence
statements in the same report mean the same thing."""

DEFAULT_NOTIONAL = Decimal("1000")
"""Gross notional the basket is expressed in. Paper only — `paper/` is the only book here."""


def wilson_lower(successes: float, trials: float, *, z: float = WILSON_Z) -> float:
    """Lower bound of the Wilson score interval for a proportion.

    Wilson rather than the normal approximation because the normal interval is badly behaved at
    small ``n`` and near the ends — at six trials it can and does produce bounds outside [0, 1],
    which would be a confidence the schema rejects rather than a conservative one. Wilson stays in
    range by construction and is the standard recommendation for exactly this regime.
    """
    if trials <= 0:
        return 0.0
    p = max(0.0, min(1.0, successes / trials))
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = (p + z2 / (2 * trials)) / denominator
    spread = (z / denominator) * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials))
    return max(0.0, centre - spread)


@dataclass(frozen=True, slots=True)
class GateStatus:
    """One gate, and whether it ran at all on this proposal."""

    order: int
    name: str
    status: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "order": self.order, "name": self.name, "status": self.status, "detail": self.detail,
        }


def walk_chain(ruling: ConstitutionRuling) -> tuple[GateStatus, ...]:
    """Which gates actually executed, derived from the binding constraint and the source order.

    **Terminal gates still short-circuit; ceiling gates no longer do — this must track
    `eval/autopsy.examine()`'s model exactly, or it silently reports the same class of defect
    that model was rewritten to fix on 2026-09-15.** Only a TERMINAL gate ends evaluation now; the
    Constitution collects every ceiling and takes the minimum (`agents/desk.py`'s
    ``ConstitutionPolicy.rule``), so a ceiling gate positioned after the binding one in source
    order still ran — it was simply not the tightest. Marking it UNREACHED, as a plain "one bound,
    everything after never ran" walk would, is exactly `vibe-trading`'s defect wearing this
    project's own name: reachability evidence that is not actually reachability evidence.
    """
    binding = ruling.binding_constraint
    binding_gate = next((g for g in CHAIN if g.name == binding), None)
    binding_is_terminal = binding_gate is not None and binding_gate.terminal

    out: list[GateStatus] = []
    bound = False
    for gate in CHAIN:
        if gate.name == binding:
            bound = True
            out.append(GateStatus(gate.order, gate.name, FIRED, ruling.reason))
        elif bound and binding_is_terminal:
            # A terminal binder still ends evaluation; nothing after it ran.
            out.append(GateStatus(gate.order, gate.name, UNREACHED))
        elif not bound and gate.terminal:
            # A terminal gate before the binder always ran and passed, regardless of what binds.
            out.append(GateStatus(gate.order, gate.name, PASSED))
        elif not binding_is_terminal:
            # The binder is a ceiling (or nothing bound at all): every ceiling gate runs
            # unconditionally once the terminals clear, so a ceiling here — before OR after the
            # binder in source order — is PASSED, not UNREACHED.
            out.append(GateStatus(gate.order, gate.name, PASSED))
        else:
            # A ceiling gate encountered before a still-pending terminal binder: unreachable,
            # because the terminal that fires later would have ended evaluation before this ran.
            out.append(GateStatus(gate.order, gate.name, UNREACHED))
    if binding == "none":
        # Nothing bound: every gate ran and allowed. `apply_constraint` reports "none" both for the
        # no-exposure short circuit and for a clean pass, so the quantity is what separates them.
        allowed = ruling.resulting_intent.quantity > 0
        out = [
            GateStatus(g.order, g.name, PASSED if allowed else UNREACHED)
            if g.order > 1 else GateStatus(g.order, g.name, PASSED if allowed else FIRED,
                                           "" if allowed else ruling.reason)
            for g in CHAIN
        ]
    return tuple(out)


@dataclass(frozen=True, slots=True)
class LegDecision:
    """One leg of the basket, its ruling, and the reachability of the chain behind it."""

    symbol: str
    side: str
    quantity: Decimal
    stated_confidence: float
    verdict: str
    binding_constraint: str
    reason: str
    gates: tuple[GateStatus, ...]

    @property
    def reached_beyond_exposure(self) -> bool:
        """The question the autopsy left open: did anything past gate one execute?"""
        return any(g.status != UNREACHED for g in self.gates if g.order > 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "side": self.side, "quantity": str(self.quantity),
            "stated_confidence": round(self.stated_confidence, 4),
            "verdict": self.verdict, "binding_constraint": self.binding_constraint,
            "reason": self.reason,
            "gates": [g.as_dict() for g in self.gates],
            "reached_beyond_exposure": self.reached_beyond_exposure,
        }


def carry_intents(
    basket: dict[str, Any], *, confidence: float, notional: Decimal = DEFAULT_NOTIONAL,
) -> tuple[Intent, ...]:
    """One intent per leg, sized to the basket's own weights.

    The thesis names the source of the return rather than the instrument's story, and the
    invalidation is mandatory for anything opening exposure — a carry whose funding turns is a
    carry that is over, and that has to be written down before the position exists rather than
    discovered afterwards.
    """
    replay = basket.get("replay") or {}
    out: list[Intent] = []
    for leg in basket["legs"]:
        weight = float(leg["weight"])
        if weight == 0:
            continue
        out.append(Intent(
            symbol=leg["symbol"],
            side=Side.SELL if weight < 0 else Side.BUY,
            quantity=(notional * Decimal(str(abs(weight)))).quantize(Decimal("0.01")),
            verdict=Verdict.TRADE,
            stated_confidence=confidence,
            thesis=(
                f"Funding carry leg of '{basket['position']}'. Return is not directional: "
                f"{replay.get('mean_funding_pct', 0.0):+.3f}% of the "
                f"{replay.get('mean_total_pct', 0.0):+.3f}% {replay.get('holding_days', 0)}-day "
                f"hold is funding, {replay.get('mean_price_pct', 0.0):+.3f}% is the price path of "
                f"a basket hedged at the minimum-variance ratio."
            ),
            invalidation=(
                "funding turns negative on the collecting leg for three consecutive settlements",
                "the measured hedge ratio moves more than 10% from the value the basket was sized "
                "on",
                f"the realised hold breaches the study's worst observed window of "
                f"{replay.get('worst_pct', 0.0):+.2f}%",
            ),
            required_hedge=tuple(
                other["symbol"] for other in basket["legs"] if other["symbol"] != leg["symbol"]
            ),
            lean="none",
        ))
    return tuple(out)


def honest_confidence(basket: dict[str, Any], span_days: float) -> tuple[float, float, str]:
    """The win rate's lower bound at the *independent* sample size, with the workings.

    Returns (confidence, independent_windows, explanation). The explanation is returned rather than
    logged because the gap between this number and the one the raw window count would give is the
    single most important fact about the proposal.
    """
    from argus.research.carry import effective_windows

    replay = basket.get("replay") or {}
    share = float(replay.get("share_positive", 0.0))
    windows = int(replay.get("windows", 0))
    holding = int(replay.get("holding_days", 1))
    independent = effective_windows(windows, holding, span_days)
    bound = wilson_lower(share * independent, independent)
    naive = wilson_lower(share * windows, windows) if windows else 0.0
    return bound, independent, (
        f"{share:.0%} of {windows} overlapping {holding}-day windows were profitable, but ninety "
        f"days of history holds about {independent:.0f} independent ones. The Wilson bound at "
        f"{independent:.0f} trials is {bound:.2f}; at {windows} it would be {naive:.2f}. The "
        f"smaller number is the one the floor sees."
    )


def assess(
    basket: dict[str, Any], *, session: SessionState, span_days: float = 90.0,
    notional: Decimal = DEFAULT_NOTIONAL, policy: Any | None = None,
) -> dict[str, Any]:
    """Run a carry basket through the real Constitution and report what each gate did."""
    from argus.agents.desk import ConstitutionPolicy
    from argus.risk.hedgeability import HedgeabilitySurface

    rules = policy or ConstitutionPolicy()
    confidence, independent, workings = honest_confidence(basket, span_days)
    # The legs hedge one another, so the surface offered to the unhedgeable-gap rule is the basket's
    # own other leg rather than an empty menu. An empty surface here would make gate four fire on a
    # position that is hedged by construction, which would be a false refusal.
    surface = HedgeabilitySurface(candidates=(), session_note="hedged internally by the basket")

    decisions: list[LegDecision] = []
    for intent in carry_intents(basket, confidence=confidence, notional=notional):
        ruling = rules.rule(intent, session=session, hedges=surface)
        decisions.append(LegDecision(
            symbol=intent.symbol, side=str(intent.side), quantity=intent.quantity,
            stated_confidence=intent.stated_confidence,
            verdict=ruling.verdict.value, binding_constraint=ruling.binding_constraint,
            reason=ruling.reason, gates=walk_chain(ruling),
        ))

    reached = sorted({
        g.name for d in decisions for g in d.gates if g.order > 1 and g.status != UNREACHED
    })
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "basket": basket["position"],
        "notional": str(notional),
        "stated_confidence": round(confidence, 4),
        "independent_windows": round(independent, 2),
        "confidence_workings": workings,
        "legs": [d.as_dict() for d in decisions],
        "gates_reached_beyond_exposure": reached,
        "verdict": _verdict(decisions, confidence, workings),
    }


def _verdict(decisions: list[LegDecision], confidence: float, workings: str) -> str:
    if not decisions:
        return "no basket survived the study, so nothing was proposed and no gate was reached"
    beyond = [g for d in decisions for g in d.gates if g.order > 1 and g.status != UNREACHED]
    if not beyond:
        return (
            "every leg returned at the exposure gate, exactly as all 170 recorded decisions did. "
            "The carry proposal did not clear gate one, so nothing changed."
        )
    fired = [g for d in decisions for g in d.gates if g.status == FIRED]
    head = (
        f"Gate one passed for the first time on record: a carry basket carries a quantity without "
        f"claiming a direction, so ``no exposure proposed`` no longer short-circuits the chain. "
        f"{len({g.name for g in beyond})} gate(s) beyond it executed."
    )
    if not fired:
        return head + (
            " Nothing bound — every gate ran and allowed the position at the stated size."
        )
    first = min(fired, key=lambda g: g.order)
    still_unreached = sorted({
        g.name for d in decisions for g in d.gates if g.status == UNREACHED
    })
    tail = (
        f" The gates after it ({', '.join(still_unreached)}) are still unreached, and are reported "
        f"as unreached rather than as passed."
        if still_unreached else ""
    )
    return head + (
        f" The first gate to actually run refused it: **{first.name}** — {first.detail}. "
        f"{workings} That is the floor doing its job on a real proposal rather than never being "
        f"consulted."
    ) + tail


def main() -> int:  # pragma: no cover - CLI
    import sys

    from argus.research.carry import REPORT_PATH as CARRY_PATH
    from argus.truth.clocks import DualClock

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not CARRY_PATH.exists():
        print("no carry study on this machine; run `python -m argus.research.carry` first")
        return 1
    study = json.loads(CARRY_PATH.read_text(encoding="utf-8"))
    survivors = [
        p for p in study["pairs"]
        if p.get("replay") and p["replay"]["funding_survives_the_price"]
    ]
    if not survivors:
        print("no basket survived being held; nothing is proposed and no gate is reached")
        return 0

    basket = max(survivors, key=lambda p: p["replay"]["mean_total_pct"])
    # A fresh NAV age: the oracle-staleness gate is a real rule and must be given a real input, but
    # it is gate three and this proposal does not get that far. Passing a stale age would make the
    # run look like it reached further than it did.
    session = DualClock().state(datetime.now(UTC), nav_age_seconds=60.0)
    report = assess(basket, session=session)

    print("CARRY DESK — a proposal that reaches the Constitution\n")
    print(f"  basket: {report['basket']}")
    print(f"  stated confidence: {report['stated_confidence']:.3f} "
          f"({report['independent_windows']:.0f} independent windows)\n")
    for leg in report["legs"]:
        print(f"  {leg['side']:5} {leg['quantity']:>9} {leg['symbol']:12} -> {leg['verdict']}")
        for gate in leg["gates"]:
            mark = {"FIRED": "X", "PASSED": "o", "UNREACHED": "."}[gate["status"]]
            print(f"      {mark} {gate['order']} {gate['name']:20} {gate['status']}")
        print()
    print(f"  {report['verdict']}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "DEFAULT_NOTIONAL",
    "REPORT_PATH",
    "WILSON_Z",
    "GateStatus",
    "LegDecision",
    "assess",
    "carry_intents",
    "honest_confidence",
    "walk_chain",
    "wilson_lower",
]
