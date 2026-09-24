"""Proving the risk layer — by sweeping its whole state space rather than waiting for it to fire.

Track 2 scores "risk control layer effectiveness", and on the live record ours has an effectiveness
of nothing: twenty risk records, zero interventions, every binding constraint ``none``. That is not
a defect — the desk has abstained on every decision so far and an abstention proposes no exposure to
narrow — but it leaves the claim unevidenced, and "our risk layer is sound" with no firing behind it
is exactly the assertion this project refuses to make about anything else.

Waiting is not the answer. A risk layer is a *function*, and a function can be proved over its
domain instead of sampled by whatever the market happens to do. This module enumerates the
Constitution's inputs — proposed verdict, side, size, stated confidence, hedge availability, NAV
staleness, session phase — takes the cross product, and applies the policy to every point. Each
point is then checked against the invariants that make the layer a risk layer at all:

* it may **reduce** exposure, never create or enlarge it;
* it may never **reverse** a side;
* it may never turn an **abstention into a trade**;
* an ``ALLOW`` must leave the intent untouched.

Those four are already guarded structurally in :func:`~argus.decision.verdicts.apply_constraint`,
which raises rather than logs. Re-checking them here is not duplication: the guards prove that *one
ruling* is well formed, and this proves that the *policy* never asks for a malformed one anywhere in
its domain. A rule with a sign error would raise in production and pass every unit test that never
happened to reach it.

**The finding this module exists to produce is rule shadowing.** The Constitution's rules are
evaluated in order and the first match wins, so a rule placed after a broader one can be
unreachable: the system appears to carry five protections and actually carries three, and nothing in
the code or the tests says which. :attr:`RiskProof.unreachable` names any rule that never binds
anywhere in the swept space, and :attr:`RiskProof.precedence` records, for every state where more
than one rule *could* have applied, which one actually did. Both are derived from the sweep, not
read off the source, so a reordering that silently disables a protection turns a test red.

    python -m argus.eval.riskproof
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import product
from pathlib import Path
from typing import Any

from argus.agents.desk import ConstitutionPolicy
from argus.decision.verdicts import (
    ConstitutionRuling,
    ConstitutionVerdict,
    ConstitutionViolation,
    Intent,
    Side,
    Verdict,
)
from argus.risk.hedgeability import HedgeabilitySurface, HedgeCandidate
from argus.truth.clocks import SessionPhase, SessionState

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "risk_proof.json"

AT = datetime(2026, 9, 14, 15, 0, tzinfo=UTC)
"""A fixed instant, so a sweep is reproducible. Monday, mid-RTH."""


@dataclass(frozen=True, slots=True)
class State:
    """One point in the Constitution's input domain."""

    verdict: Verdict
    side: Side
    quantity: Decimal
    confidence: float
    hedges_empty: bool
    nav_stale: bool
    phase: SessionPhase

    drawdown: Decimal = Decimal("0")
    """Fraction below peak equity, driving the circuit breaker and therefore **gate 6**.

    Added 2026-09-14. Without it the sweep built one policy with ``book_state=None``, which
    *structurally disables* `risk_budget` — so it could not bind anywhere in the domain, and
    because `_applicable` did not name the rule either, nothing was ever reported unreachable. The
    proof published ``unreachable: []`` while two of seven gates never ran. A domain that cannot
    reach a rule is not a proof about that rule.
    """

    session_vol_bps: float = 0.0
    """Measured path volatility for the phase, driving the throttle and therefore **gate 7**.

    Zero means "not measured", which is how the policy reads ``session_risk=None``: the throttle
    does nothing and says so. Absence is not a volatility of zero.
    """

    hedge_at_risk: bool = False
    """Whether the swept book carries a hedge-linked position matching the intent's own symbol,
    driving **gate 7**, ``hedge_integrity``.

    ``False`` builds no hedge-linked position at all, which is the honest "not applicable" case:
    most swept states carry no REDUCE verdict on a hedge-linked position, so the gate should be
    structurally inert for them, the same discipline :attr:`drawdown` and :attr:`session_vol_bps`
    already follow. ``True`` builds exactly the case the gate exists to catch: a LONG at the
    intent's own symbol (NVDAUSDT — see `_intent`), hedge-linked to an open SHORT elsewhere, so a
    REDUCE/SELL on the intent's symbol finds a still-open cluster partner.
    """

    extra_risk: str = "none"
    """One of ``"none"``, ``"margin"``, ``"factor"``, ``"scenario"``, ``"liquidation"``,
    ``"underperformance"`` — which of **gate 8** (``margin_usage``), **gate 9**
    (``factor_exposure``), **gate 10** (``scenario_loss``), **gate 11** (``liquidation_cost``),
    **gate 12** (``per_symbol_underperformance``) is built over its cap in this state, or none of
    the five.

    **Replaced an earlier design that reused `existing_gross`/`hedge_at_risk` for the first three
    of these gates and shipped a real bug**, caught by this file's own `unreachable == ()`
    invariant: tying ``margin_usage`` to the same `existing_gross` values chosen to saturate
    `gross_exposure`'s own headroom meant the two always tied at ``permitted == 0``, and
    `ConstitutionPolicy.rule`'s `min(ceilings, key=...)` deterministically picks the
    first-appended tie — `gross_exposure` every time — so `margin_usage` could never be the
    *reported* binding constraint anywhere in the domain, however correctly it computed.
    `factor_exposure`/`scenario_loss` shared the same flaw against each other via `hedge_at_risk`.
    A **mutually-exclusive** categorical, independent of every other dimension, is the fix: exactly
    one of the four (or none) is ever active in a given state, so each gets states where it is the
    *sole* ceiling and must win outright — a shared boolean/scalar reused across gates that fire on
    unrelated conditions is the anti-pattern this field exists to stop repeating. `liquidation_cost`
    (added same session, same day) went straight into this categorical rather than repeating the
    mistake a fourth time.
    """

    existing_gross: Decimal = Decimal("0")
    """Gross notional the book already carries, driving :attr:`ConstitutionPolicy.book` and
    therefore **gate 5** (``gross_exposure``) and **gate 6** (``signed_exposure``) — two gates,
    one knob, for the reuse-over-multiply reason stated below.

    One knob deliberately drives both: `_policy_for` synthesises the swept book as a single
    one-directional LONG position, so its gross notional and its signed notional are numerically
    identical by construction (nothing offsets it). A genuinely separate ``existing_signed``
    dimension — letting the sweep explore a hedged book with high gross and low net — was
    considered and rejected for this pass: it would triple the domain again on top of the tripling
    this field already caused, for coverage this project's own doctrine does not yet demand
    (neither gate has a live caller to validate against — see `Activity/PROGRESS.md`). What this
    single knob does cover: both gates' zero case (``0``), and both gates' cap-straddling cases at
    values chosen so headroom brackets zero for each cap independently (``130000`` sits above
    ``signed``'s 100,000 cap and below ``gross``'s 150,000 cap; ``150000`` sits at ``gross``'s cap
    and further above ``signed``'s).

    Same discipline as :attr:`drawdown` and :attr:`session_vol_bps` otherwise: zero means "no book
    at all" (``policy.book=None``, both gates structurally cannot fire), not "a book measured at
    zero gross/signed".
    """

    def label(self) -> str:
        return (
            f"{self.verdict}/{self.side}/q={self.quantity}/c={self.confidence}/"
            f"hedges={'empty' if self.hedges_empty else 'available'}/"
            f"nav={'stale' if self.nav_stale else 'fresh'}/{self.phase}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": str(self.verdict), "side": str(self.side),
            "quantity": str(self.quantity), "confidence": self.confidence,
            "hedges_empty": self.hedges_empty, "nav_stale": self.nav_stale,
            "phase": str(self.phase),
        }


def _session(state: State) -> SessionState:
    """A session whose staleness actually matches the requested flag.

    ``nav_is_stale`` has phase-dependent thresholds — an hour old is a broken feed during RTH and
    entirely normal on a Saturday — so a fixed age would silently mean different things across the
    sweep. The age is chosen per phase and then asserted, because a sweep that thinks it is testing
    stale NAV while feeding fresh NAV proves nothing and looks thorough.
    """
    asleep = not state.phase.has_price_discovery
    hours = 0.0 if not asleep else 18.0
    for age in (10.0, 600.0, 3600.0, 86_400.0, 604_800.0):
        candidate = SessionState(
            phase=state.phase, as_of=AT, hours_to_next_discovery=hours, nav_age_seconds=age
        )
        if candidate.nav_is_stale() is state.nav_stale:
            return candidate
    raise RiskProofError(
        f"no NAV age makes staleness {state.nav_stale} reachable in {state.phase}; the sweep "
        f"cannot honestly claim to have covered that combination"
    )


def _hedges(state: State) -> HedgeabilitySurface:
    if state.hedges_empty:
        return HedgeabilitySurface(candidates=(), session_note="swept: nothing placeable")
    return HedgeabilitySurface(
        candidates=(
            HedgeCandidate(
                instrument="QQQUSDT",
                risk_reduction=Decimal("0.85"),
                correlation_confidence=Decimal("0.9"),
                liquidity_availability=Decimal("0.9"),
                execution_probability=Decimal("0.95"),
                basis_stability=Decimal("0.8"),
                execution_cost_bps=Decimal("6"),
            ),
        ),
        session_note="swept: one placeable hedge",
    )


class RiskProofError(RuntimeError):
    """The sweep cannot be run honestly. Raised rather than silently narrowed."""


QUANTITIES: tuple[Decimal, ...] = (
    Decimal("1"),
    Decimal("19999"),
    Decimal("20000"),
    Decimal("20001"),
    Decimal("49999"),
    Decimal("50000"),
    Decimal("50001"),
    Decimal("250000"),
)
"""Sizes chosen to straddle every configured threshold, plus one either side.

Boundary values rather than a uniform grid: a limit at 20,000 is wrong in exactly one place, and a
sweep of round thousands would step over it. Each of ``max_unhedged_notional`` and
``max_position_notional`` is bracketed below, at, and above."""

CONFIDENCES: tuple[float, ...] = (0.0, 0.54, 0.55, 0.56, 0.9, 1.0)
"""Straddles ``min_confidence_to_trade``, including the boundary itself."""

DRAWDOWNS: tuple[Decimal, ...] = (Decimal("0"), Decimal("0.06"), Decimal("0.16"))
"""Fraction below peak equity, straddling the circuit breaker's ladder — **gate 6**, `risk_budget`.

Zero is the un-drawn book, where `_policy_for` supplies ``book_state=None`` and the gate is
correctly inert. The other two sit either side of the breaker's rungs so the gate has somewhere to
bind. Before these existed the sweep ran one policy with no book state at all, and `risk_budget`
could not fire for any input in the domain."""

SESSION_VOLS: tuple[float, ...] = (0.0, 1.0, 3.0)
"""Path volatility as a **multiple of the regular-hours baseline**, driving the throttle — gate 7,
`session_volatility`.

Not an absolute bps figure, because the throttle is a *ratio*: it narrows when the phase ahead is
more violent than regular hours, and an absolute number says nothing without the baseline beside
it. ``0`` means unmeasured (``session_risk=None`` — the throttle does nothing and says so, because
absence is not a volatility of zero); ``1`` is a phase no worse than RTH, which must NOT narrow;
``3`` narrows to 0.46-0.87 depending on phase, verified by probing `throttle` directly.

The middle value is the one that matters: without it the sweep could not tell "the gate binds" from
"the gate binds on everything it sees"."""

EXISTING_GROSS: tuple[Decimal, ...] = (Decimal("0"), Decimal("130000"), Decimal("150000"))
"""Gross (and, by construction, signed — see :attr:`State.existing_gross`) notional the book
already carries — gate 5 `gross_exposure` (cap 150,000) and gate 6 `signed_exposure` (cap
100,000) together.

``0`` is the un-traded book, where `_policy_for` supplies ``book=None`` and both gates are
correctly inert. ``130000`` leaves exactly 20,000 of headroom against the gross cap (deliberately
reusing the same boundary :data:`QUANTITIES` already straddles) while sitting *past* the signed
cap already — a BUY has zero signed headroom there, a SELL has plenty, exercising the
direction-dependent branch `signed_exposure` has and `gross_exposure` does not. ``150000`` is the
gross cap itself: headroom 0, so any positive quantity binds it; the signed cap is further behind
it still."""

EXTRA_RISK: tuple[str, ...] = (
    "none", "margin", "factor", "scenario", "liquidation", "underperformance",
)
"""Which one of gate 8/9/10/11/12 is built over its own cap — see :attr:`State.extra_risk` for why
this is a mutually-exclusive categorical rather than independent booleans reusing another
dimension: the earlier design tied multiple gates to the same trigger and they permanently tied
against each other at ``permitted == 0``, so none but the first-appended could ever be the
*reported* binding constraint. Each value here builds exactly one gate past its cap, with the
other four left `None` (structurally inert), so each gets swept states where it is the sole
ceiling and must win outright."""


def states(
    *,
    verdicts: Sequence[Verdict] | None = None,
    quantities: Sequence[Decimal] = QUANTITIES,
    confidences: Sequence[float] = CONFIDENCES,
    phases: Sequence[SessionPhase] | None = None,
    drawdowns: Sequence[Decimal] = DRAWDOWNS,
    session_vols: Sequence[float] = SESSION_VOLS,
    existing_grosses: Sequence[Decimal] = EXISTING_GROSS,
    hedge_at_risks: Sequence[bool] = (False, True),
    extra_risks: Sequence[str] = EXTRA_RISK,
) -> Iterator[State]:
    """Every combination of inputs the Constitution can be handed.

    Intents the domain forbids are skipped rather than forced: ``Intent`` refuses a zero quantity on
    a verdict that carries one, and manufacturing an illegal intent to feed the policy would test a
    state the system cannot reach.
    """
    verdict_pool = list(verdicts if verdicts is not None else Verdict)
    phase_pool = list(phases if phases is not None else SessionPhase)
    # Nested rather than one flat `product()` call: mypy's typeshed stub for `product` only has
    # typed overloads up to ten iterables, and the eleventh (added for `hedge_at_risk`) collapsed
    # every element's type to `object`. Three nested calls — kept inside the typed range at every
    # level — produce the identical cross product.
    base = product(
        verdict_pool, (Side.BUY, Side.SELL), quantities, confidences,
        (True, False), (True, False), phase_pool, drawdowns, session_vols,
    )
    extra = product(existing_grosses, hedge_at_risks, extra_risks)
    for (verdict, side, quantity, confidence, empty, stale, phase, drawdown, vol), (
        gross, hedge, extra_risk,
    ) in product(base, extra):
        if not verdict.carries_quantity:
            # An abstention proposes no size, so it is swept once at zero rather than once per
            # value in the grid. The first version of this loop skipped the state entirely
            # whenever the quantity was positive — and since every value in QUANTITIES is
            # positive, no abstention was ever swept at all. The "never creates a trade"
            # invariant therefore had nothing to run against, in the module whose whole purpose is
            # to find exactly that kind of hole. Caught by its own tests.
            if quantity != QUANTITIES[0]:
                continue
            yield State(
                verdict=verdict, side=side, quantity=Decimal("0"), confidence=confidence,
                hedges_empty=empty, nav_stale=stale, phase=phase,
                drawdown=drawdown, session_vol_bps=vol, existing_gross=gross,
                hedge_at_risk=hedge, extra_risk=extra_risk,
            )
            continue
        yield State(
            verdict=verdict, side=side, quantity=quantity, confidence=confidence,
            hedges_empty=empty, nav_stale=stale, phase=phase,
            drawdown=drawdown, session_vol_bps=vol, existing_gross=gross, hedge_at_risk=hedge,
            extra_risk=extra_risk,
        )


def _intent(state: State) -> Intent | None:
    """Build the proposed intent, or ``None`` if the domain forbids this combination."""
    try:
        return Intent(
            symbol="NVDAUSDT", side=state.side, quantity=state.quantity,
            verdict=state.verdict, stated_confidence=state.confidence,
            thesis="swept state",
            invalidation=("swept falsifier",) if state.verdict.opens_exposure else (),
        )
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class Violation:
    """An invariant the policy broke at one point in its domain."""

    state: str
    invariant: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"state": self.state, "invariant": self.invariant, "detail": self.detail}


def check_invariants(
    original: Intent, ruling: ConstitutionRuling, state: State
) -> list[Violation]:
    """The seven properties that make this a risk layer rather than a strategy.

    **Four of these were here from the start and two were added on 2026-09-13, because an external
    review found that the phrase "the risk layer may only reduce" was running three different
    invariants together and this prover checked one of them.** The four original checks are all
    about the *primary leg's quantity and side*. REQUIRE_HEDGE does not touch either — it attaches
    a hedge instrument (`decision/verdicts.py:184`) — so a hedge leg that was in fact a
    free-standing directional position of the risk layer's own would have passed every one of
    them.

    The two added properties are named for what they actually assert:

    * **Intent monotonicity** — the Constitution may not originate an economic thesis. Any leg it
      attaches must be a declared hedge against the position the model proposed, and in a
      *different* instrument: a "hedge" in the same symbol is not a hedge, it is more of the same
      trade wearing a risk layer's label.
    * **Risk monotonicity** — under the declared risk functional (net directional exposure in the
      proposed instrument, which is the only functional computable from a ruling alone), exposure
      after the intervention is never greater than before.

    Gross notional is deliberately **not** asserted to be monotone, because it is not: a hedge leg
    raises it. That is the honest statement and it is the one the documents now make.
    """
    out: list[Violation] = []
    result = ruling.resulting_intent

    if result.quantity > original.quantity:
        out.append(Violation(
            state.label(), "never_increases",
            f"{original.quantity} -> {result.quantity} under {ruling.verdict}",
        ))
    if original.quantity > 0 and result.quantity > 0 and result.side is not original.side:
        out.append(Violation(
            state.label(), "never_reverses",
            f"{original.side} -> {result.side} under {ruling.verdict}",
        ))
    if not original.verdict.opens_exposure and result.verdict.opens_exposure:
        out.append(Violation(
            state.label(), "never_creates_a_trade",
            f"{original.verdict} -> {result.verdict}",
        ))
    if ruling.verdict is ConstitutionVerdict.ALLOW and (
        result.quantity != original.quantity or result.verdict is not original.verdict
    ):
        out.append(Violation(
            state.label(), "allow_is_untouched",
            f"ALLOW changed {original.verdict}/{original.quantity} to "
            f"{result.verdict}/{result.quantity}",
        ))

    # --- intent monotonicity: the Constitution may not originate a thesis ---
    added_legs = tuple(h for h in result.required_hedge if h not in original.required_hedge)
    if added_legs and ruling.verdict is not ConstitutionVerdict.REQUIRE_HEDGE:
        out.append(Violation(
            state.label(), "intent_monotonicity",
            f"{ruling.verdict} attached {added_legs} without being a hedge ruling",
        ))
    for leg in added_legs:
        if leg == original.symbol:
            out.append(Violation(
                state.label(), "intent_monotonicity",
                f"a hedge leg in {leg} is the same instrument as the position it claims to "
                f"hedge, which is more of the same trade rather than a hedge",
            ))

    # --- risk monotonicity under the declared functional ---
    # The functional is net directional exposure in the proposed instrument. It is the only one
    # computable from a ruling alone: a ruling carries quantities and instrument names, not
    # correlations or prices. A richer functional would need a portfolio, and is named as a
    # limitation rather than approximated here.
    if _net_exposure(result) > _net_exposure(original):
        out.append(Violation(
            state.label(), "risk_monotonicity",
            f"net directional exposure rose from {_net_exposure(original)} to "
            f"{_net_exposure(result)} under {ruling.verdict}",
        ))

    # --- authorisation: what the ruling permits to reach a venue ---
    #
    # **The six properties above are about what the Constitution DID. This one is about what can
    # be done with its answer**, and it is the property this prover previously could not see at
    # all: it drives `ConstitutionPolicy`, so a caller that never consulted the Constitution was
    # outside the swept domain by construction. Adding the check here means the capability is not
    # merely unit-tested on a handful of fixtures — it is asserted on every one of the swept
    # states, which is the combination Ritapossible/Ballast's capability-token design does not
    # have and ours now does.
    #
    # Two things are asserted. A REJECT must be unable to mint an authorisation at all; and any
    # other ruling must authorise *its own* resulting intent and refuse a tampered one. The
    # tampered order used below is the live seq-264 shape: the same decision, one unit larger.
    out.extend(_authorisation_violations(state, ruling))
    return out


@dataclass(frozen=True, slots=True)
class _ProbeOrder:
    """The minimum an authorisation is checked against. Structural, so the prover stays in the
    decision layer and never imports execution."""

    symbol: str
    side: str
    quantity: Decimal


def _authorisation_violations(state: State, ruling: ConstitutionRuling) -> list[Violation]:
    approved = ruling.resulting_intent
    faithful = _ProbeOrder(approved.symbol, str(approved.side), approved.quantity)
    tampered = _ProbeOrder(approved.symbol, str(approved.side), approved.quantity + Decimal("1"))
    out: list[Violation] = []

    if ruling.verdict is ConstitutionVerdict.REJECT:
        try:
            ruling.authorise(faithful)
        except ConstitutionViolation:
            return out
        return [Violation(
            state.label(), "authorisation_is_required",
            "a REJECT ruling minted an authorisation, so a refused order could reach a venue",
        )]

    try:
        ruling.authorise(faithful)
    except ConstitutionViolation as exc:
        out.append(Violation(
            state.label(), "authorisation_is_required",
            f"the ruling refused to authorise its own resulting intent: {exc}",
        ))
    try:
        ruling.authorise(tampered)
    except ConstitutionViolation:
        pass
    else:
        out.append(Violation(
            state.label(), "authorisation_is_required",
            f"an order of {tampered.quantity} was authorised by a ruling for "
            f"{approved.quantity} — the capability is not bound to the size it approved",
        ))
    return out


def _net_exposure(intent: Intent) -> Decimal:
    """The declared risk functional: directional size in the proposed instrument.

    An intent that opens nothing carries no exposure whatever its stated quantity, which is why
    this reads the verdict and not the number alone — a REDUCE of 200 is not 200 of risk.
    """
    if not intent.verdict.opens_exposure:
        return Decimal("0")
    return intent.quantity


@dataclass
class RiskProof:
    """What sweeping the Constitution's domain showed."""

    generated_at: datetime
    swept: int = 0
    skipped: int = 0
    """States the intent domain forbids. Counted, never silently dropped."""

    violations: list[Violation] = field(default_factory=list)
    bindings: dict[str, int] = field(default_factory=dict)
    reduced: int = 0
    """States where the layer actually narrowed the proposal."""

    rules: tuple[str, ...] = ()
    precedence: dict[str, str] = field(default_factory=dict)
    """For each state where more than one rule could apply, which rule won."""

    errors: list[str] = field(default_factory=list)

    @property
    def sound(self) -> bool:
        return not self.violations and not self.errors

    @property
    def unreachable(self) -> tuple[str, ...]:
        """Rules that never bound anywhere in the swept domain.

        Either dead code or shadowed by an earlier rule. Both mean a protection that is believed to
        exist and does not.
        """
        return tuple(r for r in self.rules if self.bindings.get(r, 0) == 0)

    @property
    def intervention_rate(self) -> float:
        return self.reduced / self.swept if self.swept else 0.0

    def render(self) -> list[str]:
        lines = [
            f"[riskproof] swept {self.swept} state(s), skipped {self.skipped} the intent domain "
            f"forbids",
            f"[riskproof] the layer narrowed the proposal in {self.reduced} "
            f"({self.intervention_rate:.0%})",
        ]
        lines.append("[riskproof] binding constraint by frequency:")
        for name, count in sorted(self.bindings.items(), key=lambda kv: -kv[1]):
            lines.append(f"[riskproof]   {name}: {count}")
        if self.unreachable:
            lines.append(
                f"[riskproof] UNREACHABLE rule(s): {', '.join(self.unreachable)} — believed to "
                f"protect something, never binds anywhere in the domain"
            )
        else:
            lines.append("[riskproof] every configured rule binds somewhere; none is shadowed")
        if self.precedence:
            lines.append("[riskproof] precedence where rules overlap:")
            for combo, winner in sorted(self.precedence.items()):
                lines.append(f"[riskproof]   {combo} -> {winner}")
        if self.violations:
            lines.append(f"[riskproof] {len(self.violations)} INVARIANT VIOLATION(S):")
            lines.extend(
                f"[riskproof]   {v.invariant}: {v.detail} at {v.state}"
                for v in self.violations[:20]
            )
        else:
            lines.append(
                "[riskproof] no invariant violated across all seven properties: the layer never "
                "increased exposure, reversed a side, turned an abstention into a trade, altered "
                "an ALLOW, originated a leg of its own, raised net directional exposure, or "
                "authorised an order it had not approved — anywhere in the domain. Gross notional "
                "is deliberately not asserted monotone, because a hedge leg raises it while "
                "lowering net risk"
            )
        lines.extend(f"[riskproof] error: {e}" for e in self.errors)
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "swept": self.swept,
            "skipped": self.skipped,
            "sound": self.sound,
            "reduced": self.reduced,
            "intervention_rate": self.intervention_rate,
            "bindings": dict(self.bindings),
            "rules": list(self.rules),
            "unreachable": list(self.unreachable),
            "precedence": dict(self.precedence),
            "violations": [v.as_dict() for v in self.violations],
            "errors": list(self.errors),
        }


def _policy_for(base: ConstitutionPolicy, state: State) -> ConstitutionPolicy:
    """The policy as it would actually be constructed at this state.

    **The sweep used to hold one policy constant across the whole domain**, and that policy had
    ``book_state=None`` and ``session_risk=None`` — the two fields ``risk_budget`` and
    ``session_volatility`` are guarded on. So those gates could not fire for any input, and the
    sweep reported a clean bill of health over a domain in which two sevenths of the Constitution
    was switched off.

    All three are injected here exactly as `paper/runner.py` injects ``book_state``/``session_risk``
    live, from the state's own dimensions, so the swept policy is the policy. ``book`` (added
    2026-09-15, for ``gross_exposure``) has no live injection site yet to match, because nothing in
    the desk constructs a `desk.book.Book` from real fills yet either — see `Activity/PROGRESS.md`.
    """
    from argus.desk.book import Book, HedgeLink, Lot, PositionSide, VenueMarginSnapshot
    from argus.desk.portfolio import FactorExposure, StressOutcome
    from argus.market.depth import Sweep
    from argus.risk.circuit import BookState
    from argus.risk.session_risk import SessionRisk

    # **A price of 1, stated rather than defaulted.** `ConstitutionPolicy` gained
    # `reference_price` on 2026-09-21, when a rogue harness found four gates comparing a unit
    # count against a dollar ceiling. An unpriced policy now skips those four gates and says so —
    # which is correct in production and would silently disable a quarter of the Constitution
    # here, across all 2,177,280 states. The sweep's own `unreachable` check caught exactly that:
    # `unhedgeable_gap`, `gross_exposure`, `signed_exposure` and `max_position` all became
    # unreachable, which is the rule-shadowing defect this module exists to detect, turned on
    # itself.
    #
    # One is the right value, not a placeholder: QUANTITIES straddles the dollar limits directly
    # (19999/20000/20001 against a 20,000 cap) and the synthetic book below is already priced at
    # unity for the same reason, so quantity and notional coincide and the swept thresholds mean
    # what they say.
    equity = Decimal("100000")
    book_state = BookState(
        equity=equity,
        peak_equity=(
            equity / (Decimal("1") - state.drawdown) if state.drawdown < 1 else equity
        ),
        session_open_equity=equity,
    ) if state.drawdown > 0 else None

    # Unit price on a synthetic symbol: quantity alone then equals the target gross notional
    # exactly, with no rounding from decomposing it into a realistic (quantity, price) pair —
    # `total_gross_notional` only ever multiplies the two back together.
    real_book = None
    if state.existing_gross > 0:
        real_book = Book()
        real_book.apply_fill(
            symbol="SWEEP", side=PositionSide.LONG,
            lot=Lot(
                order_id="sweep", approved_intent_hash="sweep", side="buy",
                quantity=state.existing_gross, price=Decimal("1"), commission=Decimal("0"),
                ts_filled=AT,
            ),
        )

    # A hedge-linked pair at the intent's own symbol (`_intent` always proposes NVDAUSDT), for
    # gate 7, `hedge_integrity`: a LONG at NVDAUSDT, hedge-linked to an open SHORT elsewhere, so a
    # REDUCE/SELL on NVDAUSDT finds a cluster partner still open. Built on top of `real_book`
    # rather than replacing it, so this dimension composes with `existing_gross` instead of
    # silently disabling it.
    if state.hedge_at_risk:
        real_book = real_book or Book()
        real_book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            lot=Lot(
                order_id="sweep-hedge-a", approved_intent_hash="sweep", side="buy",
                quantity=Decimal("1"), price=Decimal("1"), commission=Decimal("0"), ts_filled=AT,
            ),
        )
        real_book.apply_fill(
            symbol="SWEEP-HEDGE-PARTNER", side=PositionSide.SHORT,
            lot=Lot(
                order_id="sweep-hedge-b", approved_intent_hash="sweep", side="sell",
                quantity=Decimal("1"), price=Decimal("1"), commission=Decimal("0"), ts_filled=AT,
            ),
        )
        real_book.link_hedge(HedgeLink(
            source=("NVDAUSDT", PositionSide.LONG),
            target=("SWEEP-HEDGE-PARTNER", PositionSide.SHORT),
            relationship="HEDGE", hedge_ratio=None, ts_created=AT,
        ))

    # `margin_usage` (gate 8), `factor_exposure` (gate 9), `scenario_loss` (gate 10),
    # `per_symbol_underperformance` (gate 12) — driven by `state.extra_risk`, a mutually-exclusive
    # categorical. The first version of this tied three of these to dimensions ALREADY calibrated
    # to saturate other gates (`existing_gross`, `hedge_at_risk`), and every one of them
    # permanently tied at `permitted == 0` against whichever gate was appended first in
    # `ConstitutionPolicy.rule` — caught by this file's own `unreachable == ()` test, which is
    # exactly what it exists to catch. `extra_risk` is independent of every other dimension, so
    # exactly one of these gates (or none) is ever built past its cap in a given state, and each
    # gets states where it is the sole ceiling.
    if state.extra_risk == "margin":
        real_book = real_book or Book()
        real_book.set_margin(VenueMarginSnapshot(
            account_equity=Decimal("100000"), unrealised_pnl=Decimal("0"),
            imr=Decimal("0"), mmr=Decimal("0"), mgn_ratio=Decimal("0.9"),
            position_mgn_ratio=Decimal("0"), leverage=Decimal("1"), fetched_at=AT,
        ))

    # **The throttle compares the current phase against a regular-hours baseline**, so a profile
    # carrying only one phase can never narrow anything: `throttle` returns 1.0 with the reason
    # "no regular-hours baseline to compare against". The first version of this builder supplied
    # exactly that, and `session_volatility` stayed unreachable no matter how violent the number
    # was — a sweep
    # that looked like it was testing volatility while testing nothing. Verified by probing
    # `throttle` directly: at 3x the RTH baseline it returns 0.46-0.87 depending on phase; at 1x it
    # returns 1.0 for every phase including RTH itself.
    baseline = 50.0
    risk = SessionRisk(
        symbol="SWEEP",
        phase_bps={"rth": baseline, str(state.phase): baseline * state.session_vol_bps},
        phase_counts={"rth": 500, str(state.phase): 500},
        reopen_bps=baseline * state.session_vol_bps,
        reopens=20,
        measured_at=AT,
        window_days=30,
    ) if state.session_vol_bps > 1 else None

    factor_exposures = (
        FactorExposure(factor="test-factor", exposure=0.9, observations=100),
    ) if state.extra_risk == "factor" else None
    stress_outcomes = (
        StressOutcome(shock="test-shock", portfolio_move_pct=-9.0, worst_position=None),
    ) if state.extra_risk == "scenario" else None
    liquidation_cost_estimates = {
        "NVDAUSDT": Sweep(
            side="SELL", requested_notional=Decimal("10000"), filled_notional=Decimal("10000"),
            average_price=Decimal("100"), levels_consumed=5, slippage_bps=Decimal("900"),
            complete=True,
        ),
    } if state.extra_risk == "liquidation" else None

    # `per_symbol_underperformance` (gate 12) must be built on NVDAUSDT specifically — `_intent`
    # always proposes that symbol, and the gate reads `book.symbol_realized_pnl_since(intent.
    # symbol, ...)`, so building the loss on any other symbol would make the gate permanently
    # unreachable in the sweep (the exact `unreachable == ()` failure mode this module exists to
    # catch). Safe to share the LONG NVDAUSDT position `hedge_at_risk` also builds: that fill
    # never closes, so it contributes zero realized PnL regardless of whether both dimensions are
    # active in the same swept state.
    if state.extra_risk == "underperformance":
        real_book = real_book or Book()
        real_book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            lot=Lot(
                order_id="sweep-loss-open", approved_intent_hash="sweep", side="buy",
                quantity=Decimal("10"), price=Decimal("100"), commission=Decimal("0"),
                ts_filled=AT,
            ),
        )
        real_book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            lot=Lot(
                order_id="sweep-loss-close", approved_intent_hash="sweep", side="sell",
                quantity=Decimal("10"), price=Decimal("50"), commission=Decimal("0"),
                ts_filled=AT,
            ),
        )  # realized_pnl_since = 10 * (50 - 100) = -500, past the -100 floor below

    return replace(
        base,
        reference_price=Decimal("1"), book_state=book_state, session_risk=risk, book=real_book,
        factor_exposures=factor_exposures,
        factor_exposure_limits={"test-factor": 0.5} if state.extra_risk == "factor" else {},
        stress_outcomes=stress_outcomes,
        max_scenario_loss_pct=-5.0 if state.extra_risk == "scenario" else None,
        liquidation_cost_estimates=liquidation_cost_estimates,
        max_liquidation_cost_bps=(
            Decimal("500") if state.extra_risk == "liquidation" else None
        ),
        min_symbol_realized_pnl=(
            Decimal("-100") if state.extra_risk == "underperformance" else None
        ),
    )


_THROTTLES: dict[tuple[Any, ...], Any] = {}
"""Session throttles by (profile, start, horizon). The throttle is a pure function of those three,
and the sweep asks it about the same few combinations two million times over. The profile is keyed
by its values, not its identity: `_policy_for` builds a fresh one per state, and a recycled `id()`
would hand one state another's throttle."""


def _throttle(policy: ConstitutionPolicy, start: datetime) -> Any:
    from argus.risk.session_risk import throttle as session_throttle

    risk = policy.session_risk
    assert risk is not None
    key = (risk.symbol, tuple(sorted(risk.phase_bps.items())),
           tuple(sorted(risk.phase_counts.items())), risk.reopen_bps, risk.reopens,
           risk.measured_at, risk.window_days, start, policy.session_horizon_bars)
    if key not in _THROTTLES:
        if len(_THROTTLES) > 200_000:
            _THROTTLES.clear()
        _THROTTLES[key] = session_throttle(policy.session_risk, start=start,
                                           horizon_bars=policy.session_horizon_bars)
    return _THROTTLES[key]


def _applicable(policy: ConstitutionPolicy, state: State, intent: Intent) -> set[str]:
    """Which rules *could* have bound at this state, evaluated independently of order.

    This is what makes shadowing detectable: the policy reports the one rule that won, and this
    reports every rule whose condition held. Where the two sets differ by more than one element,
    precedence is doing real work and is recorded.
    """
    could: set[str] = set()
    if not intent.verdict.carries_quantity or intent.quantity <= 0:
        return {"no_exposure"}
    if intent.stated_confidence < policy.min_confidence_to_trade:
        could.add("min_confidence")
    session = _session(state)
    if session.nav_is_stale() and session.is_anchor_asleep:
        could.add("oracle_stale")
    if state.hedges_empty and intent.quantity > policy.max_unhedged_notional:
        could.add("unhedgeable_gap")

    # **`risk_budget` and `session_volatility` were missing from this oracle entirely, and the
    # omission was invisible.** With no rule named, nothing could ever be reported unreachable —
    # so `data/risk_proof.json` published `unreachable: []`, i.e. "every rule binds somewhere",
    # while two of seven gates never bound at all. Shadowing is the exact thing this module's
    # docstring says it exists to detect, and it could not detect its own.
    #
    # `gross_exposure` — added 2026-09-15 alongside `desk.book.Book`, Foundation 3 — is below,
    # after `unhedgeable_gap`, in the same source order it appears in `ConstitutionPolicy.rule`.
    if policy.book is not None:
        existing_gross = policy.book.total_gross_notional()
        headroom = max(Decimal("0"), policy.max_gross_exposure_notional - existing_gross)
        if intent.quantity > headroom:
            could.add("gross_exposure")

        direction = Decimal("1") if intent.side is Side.BUY else Decimal("-1")
        existing_signed = policy.book.total_signed_notional()
        signed_headroom = max(
            Decimal("0"), policy.max_signed_exposure_notional - direction * existing_signed
        )
        if intent.quantity > signed_headroom:
            could.add("signed_exposure")

        if intent.verdict is Verdict.REDUCE:
            from argus.desk.book import PositionSide

            being_reduced = (
                intent.symbol, PositionSide.LONG if intent.side is Side.SELL
                else PositionSide.SHORT,
            )
            position = policy.book.positions.get(being_reduced)
            if position is not None and not position.is_flat:
                cluster = policy.book.hedge_cluster(*being_reduced)
                if any(
                    other != being_reduced
                    and (p := policy.book.positions.get(other)) is not None and not p.is_flat
                    for other in cluster
                ):
                    could.add("hedge_integrity")

        if policy.book.margin is not None and policy.book.margin.mgn_ratio >= (
            policy.max_margin_usage_ratio
        ):
            could.add("margin_usage")

    if policy.factor_exposures is not None:
        for exposure in policy.factor_exposures:
            limit = policy.factor_exposure_limits.get(exposure.factor)
            if limit is not None and exposure.exposure is not None and (
                abs(exposure.exposure) >= limit
            ):
                could.add("factor_exposure")
                break

    if policy.stress_outcomes is not None and policy.max_scenario_loss_pct is not None:
        scored = [o for o in policy.stress_outcomes if o.portfolio_move_pct is not None]
        worst = min(scored, key=lambda o: o.portfolio_move_pct, default=None)
        if worst is not None and worst.portfolio_move_pct <= policy.max_scenario_loss_pct:
            could.add("scenario_loss")

    if (
        policy.liquidation_cost_estimates is not None
        and policy.max_liquidation_cost_bps is not None
    ):
        worst_sweep = max(
            (s for s in policy.liquidation_cost_estimates.values()),
            key=lambda s: s.slippage_bps, default=None,
        )
        if worst_sweep is not None and worst_sweep.slippage_bps >= policy.max_liquidation_cost_bps:
            could.add("liquidation_cost")

    if policy.book is not None and policy.min_symbol_realized_pnl is not None:
        cutoff = session.as_of - timedelta(
            minutes=policy.symbol_underperformance_window_minutes
        )
        windowed_pnl = policy.book.symbol_realized_pnl_since(intent.symbol, cutoff)
        if windowed_pnl < policy.min_symbol_realized_pnl:
            could.add("per_symbol_underperformance")

    if policy.book_state is not None:
        from argus.risk.circuit import risk_multiplier as circuit_multiplier
        from argus.risk.sizing import size as size_position

        sizing = size_position(
            win_probability=intent.stated_confidence,
            payoff=policy.assumed_payoff,
            predictions=policy.graded_predictions or (),
            risk_multiplier=circuit_multiplier(policy.book_state),
            session_multiplier=Decimal("1"),
        )
        if policy.book_state.equity * sizing.fraction < intent.quantity:
            could.add("risk_budget")

    if policy.session_risk is not None:
        scaled = _throttle(policy, session.as_of)
        if scaled.multiplier < 1 and intent.quantity * scaled.multiplier < intent.quantity:
            could.add("session_volatility")

    if intent.quantity > policy.max_position_notional:
        could.add("max_position")
    return could or {"none"}


def sweep(
    policy: ConstitutionPolicy | None = None,
    *,
    state_pool: Sequence[State] | None = None,
    now: datetime | None = None,
) -> RiskProof:
    """Apply the policy across its whole input domain and check every result."""
    policy = policy or ConstitutionPolicy()
    proof = RiskProof(
        generated_at=now or datetime.now(UTC),
        # The full chain in source order. This tuple previously listed FIVE names and omitted
        # `risk_budget` and `session_volatility` — which is why nothing was ever reported
        # unreachable: a rule with no name cannot be missed.
        rules=(
            "no_exposure", "min_confidence", "oracle_stale", "unhedgeable_gap", "gross_exposure",
            "signed_exposure", "hedge_integrity", "margin_usage", "factor_exposure",
            "scenario_loss", "liquidation_cost", "per_symbol_underperformance", "risk_budget",
            "session_volatility", "max_position", "none",
        ),
    )
    pool = list(state_pool) if state_pool is not None else list(states())

    for state in pool:
        intent = _intent(state)
        if intent is None:
            proof.skipped += 1
            continue
        at_state = _policy_for(policy, state)
        try:
            ruling = at_state.rule(intent, session=_session(state), hedges=_hedges(state))
        except ConstitutionViolation as exc:
            # The structural guard fired, which means the policy asked for a malformed ruling.
            # That is the most serious finding available here and is recorded as a violation
            # rather than allowed to abort the sweep.
            proof.swept += 1
            proof.violations.append(Violation(
                state.label(), "policy_requested_an_illegal_ruling", str(exc)
            ))
            continue
        except RiskProofError as exc:
            proof.errors.append(str(exc))
            continue

        proof.swept += 1
        proof.bindings[ruling.binding_constraint] = (
            proof.bindings.get(ruling.binding_constraint, 0) + 1
        )
        if ruling.resulting_intent.quantity < intent.quantity:
            proof.reduced += 1
        proof.violations.extend(check_invariants(intent, ruling, state))

        could = _applicable(at_state, state, intent)
        if len(could) > 1:
            key = "+".join(sorted(could))
            proof.precedence[key] = ruling.binding_constraint

    return proof


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARGUS risk-layer proof by state sweep")
    parser.add_argument("--save", default=str(REPORT_PATH))
    args = parser.parse_args(argv)

    proof = sweep()
    for line in proof.render():
        print(line)
    if args.save:
        Path(args.save).write_text(
            json.dumps(proof.as_dict(), indent=2, default=str), encoding="utf-8"
        )
        print(f"saved -> {args.save}")
    return 0 if proof.sound else 1


__all__ = [
    "AT",
    "CONFIDENCES",
    "QUANTITIES",
    "REPORT_PATH",
    "RiskProof",
    "RiskProofError",
    "State",
    "Violation",
    "check_invariants",
    "main",
    "states",
    "sweep",
]


if __name__ == "__main__":
    raise SystemExit(main())
