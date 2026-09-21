"""Pre-trade review for an order somebody else is about to place.

**What this is.** A verdict on a Bitget v3 `placeOrder` payload — ALLOW, NARROW or REFUSE — naming
the one rule that bound and reporting the reachability of every rule behind it. It is the
Constitution pointed outward: the same `ConstitutionPolicy` the desk runs on itself, applied to an
order it did not author.

**The baseline it has to beat, read rather than remembered.** Bitget's own safety layer is
`agent-sdk/src/tools/safety.ts:78-107` (`executeWithSafety`), and it is **three checks**:

1. ``dryRun`` — return a preview instead of sending;
2. ``readOnly && op.isWrite`` — refuse writes when the session is read-only;
3. ``riskLevel === "high" && !confirm`` — require confirmation on a high-risk operation.

**None of the three looks at the order.** They are about the *session* and the *operation class*.
Size, price, instrument limits, staleness of the evidence, drawdown state and session volatility are
all outside their scope, by design — that layer exists to stop an accidental write, not an
ill-judged one. So this module is not a reimplementation of it; it answers a different question,
and the two compose.

**Why this shape and not a score.** `research/overfit.py` refuses a 0-100 score for a stated
reason, and the same reasoning applies here: a review that returns 73/100 invites a caller to ship
an order that failed a hard limit because it passed enough soft ones. There is exactly one binding
constraint and it is named.

**⚠️ The honest limitation, stated where a caller will read it.** Gates 2-7 of the Constitution have
**never bound on a live ARGUS decision** — every one of the 231 decisions on record returned at gate
1, because none proposed exposure. `eval/riskproof.py` now sweeps 60,480 states and reports that
every rule binds *somewhere in that domain*, which is a statement about the sweep, not about
production. A caller installing this is getting a gate chain proven over a synthetic domain and
unexercised in the wild, and :func:`review` says so in every verdict it returns. A safety belt
marketed as tested and never shown to hold is worse than no belt, because someone installs it and
stops looking.

The v3 contract is read from `agent-sdk/src/generated/catalog.ts` (`operationId: placeOrder`,
``POST /api/v3/trade/place-order``): ``category``, ``symbol``, ``qty``, ``side`` and ``orderType``
are required; ``price`` is required for a limit order and not applicable to a market order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from argus.decision.verdicts import Intent, Side, Verdict

CATEGORIES = ("SPOT", "MARGIN", "USDT-FUTURES", "COIN-FUTURES", "USDC-FUTURES")
SIDES = ("buy", "sell")
ORDER_TYPES = ("limit", "market")

REQUIRED = ("category", "symbol", "qty", "side", "orderType")

ALLOW = "ALLOW"
NARROW = "NARROW"
REFUSE = "REFUSE"
MALFORMED = "MALFORMED"
"""The payload could not be read as an order. **Not the same as a refusal** — a refusal is a
judgement about a trade, and this is the absence of one to judge."""

UNEXERCISED_NOTE = (
    "gates 2-7 have never bound on a live decision (231 of 231 returned at gate 1); their "
    "permissiveness is proven over eval/riskproof.py's swept domain, not in production"
)


class ReviewError(RuntimeError):
    """Raised rather than returning a verdict this module cannot stand behind."""


@dataclass(frozen=True, slots=True)
class ReviewVerdict:
    """One order, reviewed."""

    verdict: str
    binding_constraint: str
    reason: str
    submitted_qty: Decimal | None = None
    permitted_qty: Decimal | None = None
    chain: tuple[dict[str, Any], ...] = ()
    caveats: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()

    @property
    def may_send(self) -> bool:
        """Whether the caller may send the order **as submitted**.

        A NARROW is not a yes. It is a different order.
        """
        return self.verdict == ALLOW

    def as_dict(self) -> dict[str, Any]:
        return {
            "reviewed_at": datetime.now(UTC).isoformat(),
            "verdict": self.verdict,
            "may_send_as_submitted": self.may_send,
            "binding_constraint": self.binding_constraint,
            "reason": self.reason,
            "submitted_qty": None if self.submitted_qty is None else str(self.submitted_qty),
            "permitted_qty": None if self.permitted_qty is None else str(self.permitted_qty),
            "chain": list(self.chain),
            "caveats": list(self.caveats),
            "problems": list(self.problems),
        }

    def render(self) -> str:
        lines = [f"{self.verdict}  —  {self.reason}", ""]
        narrowed = (
            self.submitted_qty is not None
            and self.permitted_qty is not None
            and self.permitted_qty != self.submitted_qty
        )
        if narrowed:
            lines.append(f"  submitted {self.submitted_qty} -> permitted {self.permitted_qty}")
        for row in self.chain:
            lines.append(f"  {row['status']:<9} {row['order']}. {row['gate']}")
        if self.problems:
            lines += ["", "  problems:"] + [f"    {p}" for p in self.problems]
        if self.caveats:
            lines += ["", "  caveats:"] + [f"    {c}" for c in self.caveats]
        return "\n".join(lines)


@dataclass
class _Payload:
    """A `placeOrder` body, validated against the published contract."""

    category: str
    symbol: str
    qty: Decimal
    side: str
    order_type: str
    price: Decimal | None
    problems: list[str] = field(default_factory=list)


def _number(value: Any, *, field_name: str, problems: list[str]) -> Decimal | None:
    if value is None:
        return None
    try:
        got = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        problems.append(f"{field_name}: {value!r} is not a number")
        return None
    if got <= 0:
        problems.append(f"{field_name}: {got} is not a positive size")
        return None
    return got


def parse(payload: dict[str, Any]) -> _Payload:
    """Read a v3 `placeOrder` body, collecting every problem rather than raising on the first.

    A caller fixing one field at a time learns nothing about the second; returning the whole list is
    the difference between a reviewer and a gate.
    """
    problems: list[str] = []
    for name in REQUIRED:
        if payload.get(name) in (None, ""):
            problems.append(f"{name}: required by /api/v3/trade/place-order and absent")

    category = str(payload.get("category", ""))
    if category and category not in CATEGORIES:
        problems.append(f"category: {category!r} is not one of {', '.join(CATEGORIES)}")

    side = str(payload.get("side", "")).lower()
    if side and side not in SIDES:
        problems.append(f"side: {side!r} is not one of {', '.join(SIDES)}")

    order_type = str(payload.get("orderType", "")).lower()
    if order_type and order_type not in ORDER_TYPES:
        problems.append(f"orderType: {order_type!r} is not one of {', '.join(ORDER_TYPES)}")

    qty = _number(payload.get("qty"), field_name="qty", problems=problems)
    price = _number(payload.get("price"), field_name="price", problems=problems)

    # The contract's own conditional rule, not an invention of ours.
    if order_type == "limit" and price is None:
        problems.append("price: required when orderType is limit")
    if order_type == "market" and payload.get("price") not in (None, ""):
        problems.append("price: not applicable when orderType is market")

    return _Payload(
        category=category, symbol=str(payload.get("symbol", "")),
        qty=qty or Decimal("0"), side=side, order_type=order_type, price=price,
        problems=problems,
    )


def review(
    payload: dict[str, Any],
    *,
    policy: Any | None = None,
    session: Any | None = None,
    hedges: Any | None = None,
    confidence: float = 1.0,
    reference_price: Decimal | None = None,
) -> ReviewVerdict:
    """Rule on one order and report the whole chain behind the answer.

    ``confidence`` defaults to 1.0 — **a caller's order is not ours to doubt on conviction.** Gate 2
    is the desk's own hurdle about its own edge, and applying it to somebody else's thesis would be
    substituting our judgement for theirs. The structural gates are what this service is for.
    """
    from argus.agents.desk import ConstitutionPolicy
    from argus.eval.autopsy import CHAIN
    from argus.risk.hedgeability import HedgeabilitySurface
    from argus.truth.clocks import DualClock, SessionState

    read = parse(payload)
    if read.problems:
        return ReviewVerdict(
            verdict=MALFORMED,
            binding_constraint="contract",
            reason=(
                f"the payload does not satisfy /api/v3/trade/place-order: "
                f"{len(read.problems)} problem(s)"
            ),
            problems=tuple(read.problems),
            caveats=(UNEXERCISED_NOTE,),
        )

    # **Every notional gate in this service was dead, and this is the third place that defect
    # appeared.** `ConstitutionPolicy` denominates `max_position_notional`, `max_gross_exposure`
    # and the rest in *money*, and computes them from `reference_price`; left at ``None`` it skips
    # them and says so. So a default-constructed policy ran the structural gates and silently
    # ignored every size limit — on the one service that exists to point the Constitution outward
    # at somebody else's order.
    #
    # A limit order carries its own price and that is the right one to use: it is the price the
    # caller has committed to, so it bounds the notional exactly. A market order carries none, and
    # rather than invent a quote the gates stay skipped and `reference_price` stays ``None`` — the
    # Constitution already reports that honestly, and a made-up price would produce a limit that
    # looks enforced and is not. `caveats` says which of the two happened.
    #
    # Precedence: an explicit `reference_price` from the caller, else the order's own limit price,
    # else nothing. The caller's wins because they may be reviewing a market order against a quote
    # only they have — this service takes no market-data dependency by design, and inventing a
    # price here would produce a limit that looks enforced and is not.
    if policy is not None:
        rules = policy
    else:
        priced = reference_price if reference_price is not None else read.price
        rules = ConstitutionPolicy(reference_price=priced) if priced is not None \
            else ConstitutionPolicy()
    now = datetime.now(UTC)
    clock = DualClock()
    state = session if session is not None else SessionState(
        phase=clock.phase(now), as_of=now, hours_to_next_discovery=0.0, nav_age_seconds=0.0,
    )
    surface = hedges if hedges is not None else HedgeabilitySurface(candidates=())

    intent = Intent(
        symbol=read.symbol,
        side=Side.BUY if read.side == "buy" else Side.SELL,
        quantity=read.qty,
        verdict=Verdict.TRADE,
        stated_confidence=confidence,
        thesis=f"reviewed order: {read.order_type} {read.side} {read.qty} {read.symbol}",
        invalidation=("the submitting caller's own falsifier, not ours",),
    )

    ruling = rules.rule(intent, session=state, hedges=surface)
    bound = str(ruling.binding_constraint)
    permitted = Decimal(str(ruling.resulting_intent.quantity))

    # **Terminal gates short-circuit; ceiling gates do not — matching `eval/autopsy.examine()`'s
    # model exactly.** The Constitution evaluates every ceiling gate and takes the minimum
    # (`agents/desk.ConstitutionPolicy.rule`), so a ceiling positioned after the binding one in
    # source order still ran; only a terminal binder actually ends evaluation. Found by this same
    # module's own test suite (`desk/carrydesk.walk_chain` carried the identical defect,
    # independently) — a third copy of the same short-circuit-everything assumption, this one
    # never updated when the Constitution was restructured on 2026-09-15.
    bound_gate = next((g for g in CHAIN if g.name == bound), None)
    bound_is_terminal = bound_gate is not None and bound_gate.terminal
    chain: list[dict[str, Any]] = []
    fired = False
    for gate in CHAIN:
        if gate.name == bound:
            status = "FIRED"
            fired = True
        elif fired and bound_is_terminal:
            status = "UNREACHED"
        elif (not fired and gate.terminal) or not bound_is_terminal:
            status = "PASSED"
        else:
            status = "UNREACHED"
        chain.append({"order": gate.order, "gate": gate.name, "status": status})

    if bound in ("none", "no_exposure"):
        verdict, reason = ALLOW, "within every configured limit"
    elif permitted <= 0:
        verdict, reason = REFUSE, ruling.reason
    elif permitted < read.qty:
        verdict, reason = NARROW, ruling.reason
    else:
        verdict, reason = ALLOW, ruling.reason

    # A caller is entitled to know that the size limits did not run, rather than reading an ALLOW
    # and assuming they did.
    caveats = [UNEXERCISED_NOTE]
    if getattr(rules, "reference_price", None) is None:
        caveats.append(
            "no reference price was available for this order, so the notional limits "
            "(position size, gross exposure) were SKIPPED rather than passed. Send a limit "
            "order, whose price bounds the notional exactly, to have them evaluated."
        )
    return ReviewVerdict(
        verdict=verdict,
        binding_constraint=bound,
        reason=reason,
        submitted_qty=read.qty,
        permitted_qty=permitted,
        chain=tuple(chain),
        caveats=tuple(caveats),
    )


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="review a Bitget v3 placeOrder payload before it is sent"
    )
    parser.add_argument("--payload", help="the order as JSON; omit to read stdin")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    raw = args.payload if args.payload else sys.stdin.read()
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"not JSON: {exc}", file=sys.stderr)
        return 2

    got = review(body)
    print(json.dumps(got.as_dict(), indent=2) if args.json else got.render())
    return 0 if got.may_send else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "ALLOW",
    "CATEGORIES",
    "MALFORMED",
    "NARROW",
    "REFUSE",
    "REQUIRED",
    "UNEXERCISED_NOTE",
    "ReviewError",
    "ReviewVerdict",
    "main",
    "parse",
    "review",
]
