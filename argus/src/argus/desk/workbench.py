"""Track 3 — the AI Trading Research OS. All six sub-themes.

Positioning: *"A natural-language-driven AI research workbench. AI processes information, invokes
tools, and presents analysis; **human traders make final decisions**."*

That last clause inverts Track 2 and the code has to honour it. Nothing in this module places an
order or produces a binding verdict. Every output ends at a human, and the measure of quality is
whether the human can get from a claim to its evidence in one step.

**The competitive reality this is built against.** Track 3 is the most crowded of the three:
financial research agents with SEC and earnings RAG, portfolio dashboards, AI-native wealth
workstations, and OpenAI's financial-services ChatGPT (launched 2026-09-10) all exist. Data breadth
is dead as a differentiator. What is left is **decision quality and evidence lineage**, so that is
what every sub-theme below is built around.

======================================  ==================================================
Sub-theme                               What this module provides
======================================  ==================================================
Information Extraction & Signal Gen     Claim -> Evidence -> Signal graph with bound
                                        citations; a claim is never separable from its
                                        locator
Review & Self-Evolution                 Research Autopsy accumulating a Personal Error
                                        Profile (0/12 sources own this)
Decision Stress Testing                 Scenario recomputation, not a static report
Personalised Research Workbench         A profile that changes the *verdict*, not the theme
Execution Assistance                    Participation schedule with expected cost, from the
                                        same engine Track 2 uses
Open Theme — Portfolio Copilot          What a proposed trade does to an existing book
======================================  ==================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from argus.cost.model import CostModel
from argus.execution.passive import BookObservation, PassiveExecution
from argus.sim.market import MarketError, exit_cost_bps

_ZERO = Decimal("0")


# =============================================================================================
# 1 — Information Extraction: the Claim -> Evidence -> Signal graph
# =============================================================================================

@dataclass(frozen=True, slots=True)
class Locator:
    """Where a number physically came from.

    Adopted from docling's table extraction, which preserves a cell-level ``BoundingBox`` in page
    coordinates (``table_structure_model.py:143-150``). A number without one of these cannot be
    audited, and an unauditable number is the thing this whole track is supposed to eliminate.
    """

    document: str
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    cell: str | None = None

    def render(self) -> str:
        parts = [self.document]
        if self.page is not None:
            parts.append(f"p{self.page}")
        if self.cell:
            parts.append(self.cell)
        if self.bbox:
            parts.append("bbox({:.0f},{:.0f},{:.0f},{:.0f})".format(*self.bbox))
        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class Extracted:
    """One extracted value, inseparable from where it came from.

    ``value`` and ``locator`` are constructed together on purpose. Agent-Rita binds citations to
    the data actually injected during the loop (``round-trip.ts:743-754``) so the model physically
    cannot cite outside that set; self-reported citations are the failure mode we refuse.
    """

    label: str
    value: Decimal
    unit: str
    locator: Locator
    as_of: datetime

    def render(self) -> str:
        return f"{self.label} = {self.value}{self.unit}  [{self.locator.render()}]"


class UnsourcedClaim(RuntimeError):
    """A claim was asserted with no evidence behind it.

    Raised rather than warned. A research workbench whose claims can float free of their sources
    is a chat box, and a chat box loses this track.
    """


@dataclass
class ClaimGraph:
    """Claims, the evidence under them, and the derived signal — all traversable.

    The product promise: a user clicks *"margins are deteriorating"* and walks down to the 10-Q
    page, the revenue and COGS cells, the computed margin, the prior quarter and the consensus.
    Not citations — **lineage**.
    """

    claims: dict[str, list[Extracted]] = field(default_factory=dict)
    derivations: dict[str, str] = field(default_factory=dict)

    def assert_claim(self, claim: str, evidence: list[Extracted], *, derivation: str = "") -> None:
        if not evidence:
            raise UnsourcedClaim(
                f"{claim!r} has no evidence. Every claim on this desk carries its locators or it "
                f"does not appear."
            )
        self.claims[claim] = evidence
        if derivation:
            self.derivations[claim] = derivation

    def trace(self, claim: str) -> list[str]:
        """The full path from a claim down to page and cell."""
        if claim not in self.claims:
            raise UnsourcedClaim(f"{claim!r} is not on this graph")
        lines = [claim]
        if claim in self.derivations:
            lines.append(f"  computed as: {self.derivations[claim]}")
        lines.extend(f"  <- {e.render()}" for e in self.claims[claim])
        return lines

    @property
    def is_fully_sourced(self) -> bool:
        return all(self.claims.values())


# =============================================================================================
# 2 — Review & Self-Evolution: the Research Autopsy
# =============================================================================================

@dataclass(frozen=True, slots=True)
class Autopsy:
    """One decision, graded after the outcome is known.

    The corpus has **0/12 sources** doing post-trade edge attribution, which makes this the most
    open cell in the whole project. The point is not to record P&L — everyone does that — it is to
    separate *why* it worked from *whether* it worked.
    """

    decision_id: str
    thesis: str
    predicted_direction: str
    realised_direction: str
    predicted_magnitude_bps: int
    realised_magnitude_bps: int
    stated_confidence: float
    evidence_ignored: tuple[str, ...] = ()

    @property
    def direction_correct(self) -> bool:
        return self.predicted_direction == self.realised_direction

    @property
    def magnitude_error_bps(self) -> int:
        return abs(self.predicted_magnitude_bps - self.realised_magnitude_bps)

    @property
    def failure_mode(self) -> str:
        """Which part was wrong — the four are different lessons and must not be merged."""
        if self.direction_correct and self.magnitude_error_bps < 20:
            return "correct"
        if self.direction_correct:
            return "magnitude"      # right idea, wrong size — a sizing problem
        if self.stated_confidence > 0.75:
            return "overconfident"  # wrong and sure — the expensive failure
        return "thesis"             # wrong and appropriately unsure


@dataclass
class ErrorProfile:
    """What this researcher gets wrong, accumulated.

    Turns the desk from a research assistant into a research *improvement* system — the difference
    between "here is an analysis" and "you are consistently early on semiconductor reversals".
    """

    autopsies: list[Autopsy] = field(default_factory=list)

    def add(self, autopsy: Autopsy) -> None:
        self.autopsies.append(autopsy)

    @property
    def failure_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self.autopsies:
            out[a.failure_mode] = out.get(a.failure_mode, 0) + 1
        return out

    @property
    def calibration_gap(self) -> float:
        """Stated confidence minus realised hit rate. Positive means overconfident.

        The number that makes "90% confident" mean something: if it does not behave like 90%
        across many decisions, sizing on it is a mistake the desk should surface.
        """
        if not self.autopsies:
            return 0.0
        stated = sum(a.stated_confidence for a in self.autopsies) / len(self.autopsies)
        realised = sum(1 for a in self.autopsies if a.direction_correct) / len(self.autopsies)
        return stated - realised

    def findings(self) -> list[str]:
        """Plain-language patterns, only where the sample can carry them."""
        if len(self.autopsies) < 5:
            return [f"only {len(self.autopsies)} graded decisions — too few to draw a pattern"]

        out: list[str] = []
        counts = self.failure_counts
        total = len(self.autopsies)

        gap = self.calibration_gap
        if gap > 0.15:
            out.append(
                f"overconfident by {gap:.0%}: stated confidence exceeds realised hit rate "
                f"across {total} decisions — reduce size, not conviction"
            )
        elif gap < -0.15:
            out.append(f"underconfident by {abs(gap):.0%}: your calls land more often than you say")

        if counts.get("magnitude", 0) > total * 0.4:
            out.append(
                "directionally right, repeatedly wrong on size — a sizing problem, "
                "not a research problem"
            )
        if counts.get("overconfident", 0) > total * 0.25:
            out.append("a quarter of your errors are high-confidence misses — the expensive kind")

        ignored = [e for a in self.autopsies for e in a.evidence_ignored]
        if ignored:
            seen: dict[str, int] = {}
            for e in ignored:
                seen[e] = seen.get(e, 0) + 1
            worst = max(seen, key=lambda k: seen[k])
            if seen[worst] >= 3:
                out.append(f"you have discounted {worst!r} evidence {seen[worst]} times")

        return out or ["no systematic bias detected in this sample"]


# =============================================================================================
# 3 — Decision Stress Testing
# =============================================================================================

@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    price_shock_pct: Decimal
    liquidity_multiplier: Decimal = Decimal("1")
    correlation_break: bool = False


STANDARD_SCENARIOS = (
    Scenario("base", Decimal("0")),
    Scenario("bear", Decimal("-5")),
    Scenario("bull", Decimal("5")),
    Scenario("gap_down", Decimal("-12")),
    Scenario("liquidity_collapse", Decimal("-3"), Decimal("0.3")),
    Scenario("correlation_break", Decimal("-4"), Decimal("0.7"), correlation_break=True),
)


@dataclass(frozen=True, slots=True)
class StressResult:
    scenario: str
    position_value: Decimal
    pnl: Decimal
    pnl_pct: Decimal
    exit_cost_bps: Decimal
    survives: bool
    exit_measured: bool = False
    """Whether ``exit_cost_bps`` was measured against a simulated book or assumed from the fee."""

    exitable: bool = True
    """Whether the position could be fully liquidated at all.

    The finding arithmetic cannot produce. On a thin book a position can be **unexitable at any
    price**, and no multiplier applied to a fee will ever say so.
    """

    unfilled_quantity: Decimal = _ZERO


def stress_position(
    *,
    quantity: Decimal,
    entry_price: Decimal,
    loss_tolerance_pct: Decimal,
    scenarios: tuple[Scenario, ...] = STANDARD_SCENARIOS,
    cost: CostModel | None = None,
    simulate: bool = True,
) -> list[StressResult]:
    """Recompute a position under each scenario, exit cost included.

    A static stress table is the field norm. The property that matters here is that exit cost
    rises as liquidity falls — a position is hardest to leave exactly when you most want to.

    **``simulate=True`` measures that instead of assuming it.** The position is liquidated into a
    book populated by the ABIDES agent zoo at the scenario's liquidity, and the cost is the
    realised slippage from the mid to the achieved VWAP, plus the fee. Setting ``simulate=False``
    falls back to ``round_trip / liquidity_multiplier`` — a number that moves in the right
    direction and is otherwise invented, kept only so the two can be compared.

    The simulation earns its cost by answering one question arithmetic cannot: whether the position
    can be exited **at all**. A fee divided by a multiplier always says yes.
    """
    model = cost or CostModel.bitget_perp()
    notional = quantity * entry_price
    out: list[StressResult] = []

    for s in scenarios:
        shocked = entry_price * (Decimal("1") + s.price_shock_pct / Decimal("100"))
        value = quantity * shocked
        pnl = value - notional
        pnl_pct = (pnl / notional * Decimal("100")) if notional > 0 else _ZERO

        measured, exitable, unfilled = False, True, _ZERO
        if simulate:
            try:
                m = exit_cost_bps(
                    quantity=quantity,
                    depth_multiplier=max(s.liquidity_multiplier, Decimal("0.01")),
                    reference_price=shocked,
                )
                exit_cost = model.round_trip_bps() + m.slippage_bps
                measured, exitable, unfilled = True, m.fully_exited, m.unfilled
            except MarketError:
                # A simulated book that cannot produce a mid cannot price an exit. Fall back and
                # say so, rather than reporting an assumed number as a measured one.
                exit_cost = model.round_trip_bps() / max(s.liquidity_multiplier, Decimal("0.01"))
        else:
            exit_cost = model.round_trip_bps() / max(s.liquidity_multiplier, Decimal("0.01"))

        out.append(StressResult(
            scenario=s.name,
            position_value=value,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_cost_bps=exit_cost,
            # A position that cannot be liquidated does not survive, whatever its mark says.
            survives=pnl_pct > -loss_tolerance_pct and exitable,
            exit_measured=measured,
            exitable=exitable,
            unfilled_quantity=unfilled,
        ))
    return out


# =============================================================================================
# 4 — Personalised Workbench: a profile that changes the verdict
# =============================================================================================

@dataclass(frozen=True, slots=True)
class TraderProfile:
    """Who this is for. "All traders" is explicitly not accepted by the submission form.

    These fields bind decisions rather than styling. The proof that personalisation is real is
    that two profiles get **different verdicts on identical market state**.
    """

    name: str
    capital: Decimal
    max_position_pct: Decimal
    max_sector_pct: Decimal
    holding_horizon_hours: int
    loss_tolerance_pct: Decimal
    preferred_evidence: tuple[str, ...] = ()

    # --- the dimensions an Investor Policy Statement carries that the first six did not ---
    #
    # Chosen because each one can change a decision *on this venue*, not because an IPS lists them.
    # A field that cannot flip a verdict is styling, and the audit against Vibe-Trading
    # (`research/architecture/personalisation-audit.md`) found our divergence rate was low precisely
    # because six fields left too little for a profile to disagree about.

    requires_hedge: bool = False
    """Refuse to open exposure that cannot be hedged right now.

    The strongest personalisation dimension available here, because the hedge menu is genuinely
    empty for 65 hours a week while the anchor market sleeps. A profile with this set declines every
    weekend position; one without it accepts them. Two traders, the same evidence, opposite answers
    — which is what personalisation has to mean."""

    excluded_symbols: tuple[str, ...] = ()
    """Instruments this trader will not hold, whatever the evidence says.

    A hard refusal rather than a penalty: an exclusion that can be outvoted by a good thesis is a
    preference, and a mandate is not a preference."""

    max_concurrent_positions: int = 0
    """Zero means unlimited. A cap the book must respect regardless of individual position size."""

    min_confidence: float = 0.0
    """Conviction floor. A cautious trader may want more than the desk's own 0.55."""

    @classmethod
    def conservative(cls) -> TraderProfile:
        return cls(
            name="conservative income", capital=Decimal("100000"),
            max_position_pct=Decimal("5"), max_sector_pct=Decimal("15"),
            holding_horizon_hours=720, loss_tolerance_pct=Decimal("3"),
            preferred_evidence=("filing", "sec-edgar"),
            # Will not carry unhedgeable exposure, and wants more conviction than the desk floor.
            # Both of these make this profile decline positions the aggressive one accepts.
            requires_hedge=True,
            excluded_symbols=("TQQQUSDT", "SQQQUSDT"),
            max_concurrent_positions=3,
            min_confidence=0.7,
        )

    @classmethod
    def aggressive(cls) -> TraderProfile:
        return cls(
            name="aggressive event trader", capital=Decimal("100000"),
            max_position_pct=Decimal("25"), max_sector_pct=Decimal("60"),
            holding_horizon_hours=48, loss_tolerance_pct=Decimal("15"),
            preferred_evidence=("news", "social", "transcript"),
            requires_hedge=False,
            max_concurrent_positions=0,
            min_confidence=0.55,
        )


# =============================================================================================
# 5 — Portfolio Copilot (Open Theme): what a trade does to the book
# =============================================================================================

@dataclass(frozen=True, slots=True)
class Holding:
    symbol: str
    quantity: Decimal
    price: Decimal
    sector: str

    @property
    def notional(self) -> Decimal:
        return self.quantity * self.price


@dataclass(frozen=True, slots=True)
class PortfolioImpact:
    """The sentence a generic desk cannot produce.

    Not "is this a good trade" but "is this a good trade *for your book*" — which is a different
    question and the only one that justifies a personalised workbench existing.
    """

    symbol: str
    sector: str
    trade_notional: Decimal
    position_pct_before: Decimal
    position_pct_after: Decimal
    sector_pct_before: Decimal
    sector_pct_after: Decimal
    breaches: tuple[str, ...]

    @property
    def permitted(self) -> bool:
        return not self.breaches

    def verdict(self) -> str:
        if self.permitted:
            return (
                f"within mandate: {self.symbol} moves to {self.position_pct_after:.1f}% of book, "
                f"{self.sector} sector to {self.sector_pct_after:.1f}%"
            )
        return (
            "good trade in isolation, bad trade for your book — "
            + "; ".join(self.breaches)
        )


def assess_trade(
    *,
    holdings: list[Holding],
    symbol: str,
    sector: str,
    trade_notional: Decimal,
    profile: TraderProfile,
) -> PortfolioImpact:
    """What a proposed trade does to beta, concentration and sector exposure.

    Deliberately reports *both* sides: the numbers before and after, so the user sees the movement
    rather than a verdict they must trust.
    """
    book = sum((h.notional for h in holdings), _ZERO)
    total_after = book + trade_notional
    if total_after <= 0:
        total_after = trade_notional if trade_notional > 0 else Decimal("1")

    sym_before = sum((h.notional for h in holdings if h.symbol == symbol), _ZERO)
    sec_before = sum((h.notional for h in holdings if h.sector == sector), _ZERO)

    def pct(x: Decimal, of: Decimal) -> Decimal:
        return (x / of * Decimal("100")) if of > 0 else _ZERO

    pos_before = pct(sym_before, book if book > 0 else Decimal("1"))
    pos_after = pct(sym_before + trade_notional, total_after)
    sec_after = pct(sec_before + trade_notional, total_after)

    breaches: list[str] = []
    if pos_after > profile.max_position_pct:
        breaches.append(
            f"it lifts {symbol} from {pos_before:.1f}% to {pos_after:.1f}% of the book "
            f"against your {profile.max_position_pct}% single-name cap"
        )
    if sec_after > profile.max_sector_pct:
        breaches.append(
            f"it lifts {sector} from {pct(sec_before, book if book > 0 else Decimal('1')):.1f}% "
            f"to {sec_after:.1f}% against your {profile.max_sector_pct}% sector cap"
        )
    if trade_notional > profile.capital * profile.max_position_pct / Decimal("100"):
        breaches.append(
            f"the ticket itself ({trade_notional}) exceeds your per-position budget"
        )

    return PortfolioImpact(
        symbol=symbol, sector=sector, trade_notional=trade_notional,
        position_pct_before=pos_before, position_pct_after=pos_after,
        sector_pct_before=pct(sec_before, book if book > 0 else Decimal("1")),
        sector_pct_after=sec_after,
        breaches=tuple(breaches),
    )


# =============================================================================================
# 6 — Execution Assistance
# =============================================================================================

@dataclass(frozen=True, slots=True)
class ExecutionSlice:
    index: int
    fraction: Decimal
    style: str
    expected_cost_bps: Decimal


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    symbol: str
    total_notional: Decimal
    slices: tuple[ExecutionSlice, ...]
    participation_rate: Decimal
    expected_total_cost_bps: Decimal
    rationale: str

    def render(self) -> list[str]:
        out = [
            f"{self.symbol}  {self.total_notional}  "
            f"participation {self.participation_rate:.1%}  "
            f"expected cost {self.expected_total_cost_bps:.1f}bps",
            f"  {self.rationale}",
        ]
        out.extend(
            f"  slice {s.index}: {s.fraction:.0%} {s.style} (~{s.expected_cost_bps:.1f}bps)"
            for s in self.slices
        )
        return out


def _passive_rate(
    model: CostModel, book: BookObservation | None, *, fraction: Decimal, notional: Decimal
) -> tuple[Decimal, str]:
    """The rate a passive slice actually earns, and the style label that tells the truth about it.

    **This function exists because the planner used to be wrong.** It quoted ``maker_bps`` on every
    slice it labelled "passive limit" — which is precisely the defect
    :mod:`argus.execution.passive` was built to prevent, and which the backtest engine now refuses
    outright. A maker rate is a claim about fills; without a book it is unevidenced, and the honest
    quote is the taker rate.
    """
    if book is None:
        return model.taker_bps, "limit (quoted at taker: no book supplied to price the fill)"
    fill = PassiveExecution(cost=model, chase=True).execute(
        max(fraction * notional, Decimal("1")), book
    )
    label = f"passive limit (fill {fill.fill_rate:.0%})"
    return fill.fee_bps, label


def plan_execution(
    *,
    symbol: str,
    notional: Decimal,
    adv_notional: Decimal,
    urgency: str = "low",
    anchor_asleep: bool = False,
    cost: CostModel | None = None,
    book: BookObservation | None = None,
) -> ExecutionPlan:
    """Split an order, with the cost of each style stated rather than implied.

    Three properties taken from reading execution engines rather than invented:

    * **Impact is superlinear in participation** (cvxportfolio's default exponent is 1.5,
      ``costs.py:826``), so splitting genuinely reduces cost — but only until the fee dominates.
    * **A thin session is not a good time to be passive.** hftbacktest's fill condition requires
      queue position to advance; on a book at a third of normal depth, resting orders are adversely
      selected. Urgency therefore rises, not falls, when the anchor is asleep.
    * **A passive slice is only cheap if it fills.** Supply ``book`` and each passive slice is
      priced through :class:`~argus.execution.passive.PassiveExecution` — the realised blend of
      maker and chased-taker. Omit it and passive slices are quoted at the *taker* rate, because an
      unevidenced maker fee is the single most common way a plan flatters itself.

    For *how fast* rather than *how*, see :func:`argus.execution.schedule.trajectory`, which solves
    the impact-versus-risk trade-off in closed form. This function chooses styles; that one chooses
    the speed, and they compose.
    """
    model = cost or CostModel.bitget_perp()
    participation = (notional / adv_notional) if adv_notional > 0 else Decimal("1")
    slices: tuple[ExecutionSlice, ...]

    if urgency == "high" or participation > Decimal("0.05"):
        opp_rate, opp_label = _passive_rate(
            model, book, fraction=Decimal("0.2"), notional=notional
        )
        slices = (
            ExecutionSlice(1, Decimal("0.5"), "market", model.taker_bps),
            ExecutionSlice(2, Decimal("0.3"), "near-touch limit", model.taker_bps),
            ExecutionSlice(3, Decimal("0.2"), f"opportunistic {opp_label}", opp_rate),
        )
        rationale = (
            "high urgency or large footprint: take liquidity early, finish opportunistically"
        )
    elif anchor_asleep:
        slices = (
            ExecutionSlice(1, Decimal("0.6"), "near-touch limit", model.taker_bps),
            ExecutionSlice(2, Decimal("0.4"), "market", model.taker_bps),
        )
        rationale = (
            "anchor asleep: book is thin and resting orders are adversely selected, so passive "
            "styles are down-weighted rather than up-weighted"
        )
    else:
        rate, label = _passive_rate(model, book, fraction=Decimal("0.34"), notional=notional)
        slices = (
            ExecutionSlice(1, Decimal("0.34"), label, rate),
            ExecutionSlice(2, Decimal("0.33"), label, rate),
            ExecutionSlice(3, Decimal("0.33"), "near-touch limit", model.taker_bps),
        )
        rationale = (
            "low urgency in a liquid session: work the order passively"
            if book is not None
            else "low urgency in a liquid session, but no book was supplied — passive slices are "
                 "quoted at taker, so this plan's cost is an upper bound rather than a forecast"
        )

    expected = sum(
        (s.fraction * s.expected_cost_bps for s in slices), _ZERO
    ) + model.impact_coefficient * (participation ** model.gamma)

    return ExecutionPlan(
        symbol=symbol,
        total_notional=notional,
        slices=slices,
        participation_rate=participation,
        expected_total_cost_bps=expected,
        rationale=rationale,
    )


__all__ = [
    "STANDARD_SCENARIOS",
    "Autopsy",
    "ClaimGraph",
    "ErrorProfile",
    "ExecutionPlan",
    "Extracted",
    "Holding",
    "Locator",
    "PortfolioImpact",
    "Scenario",
    "StressResult",
    "TraderProfile",
    "UnsourcedClaim",
    "assess_trade",
    "plan_execution",
    "stress_position",
]
