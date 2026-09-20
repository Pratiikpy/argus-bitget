"""Decision stress testing — scenarios drawn from what actually happened, not from round numbers.

Track 3's sub-theme asks: *before opening a position, how does AI retrieve historically similar
scenarios and run preset stress tests?* This desk already had the second half —
:func:`~argus.desk.workbench.stress_position` shocks a position and liquidates it into a simulated
book, so it can say whether the position is exitable at all rather than only what it would be worth.
What it did not have is the first half, and the gap matters more than it looks.

**The field's stress table is a table of round numbers.** ``-5%``, ``-10%``, ``-20%``: every
published example uses them, and nothing about a particular instrument produced them. They are a
convention. A trader reading "you lose 8% in the bear case" has no way to know whether the bear case
is a Tuesday or a once-a-decade event, and the table does not say, because the table does not know.

This module answers the question the table cannot: **how often has that actually happened to this
instrument, and when was the last time?** A shock is derived from the realised distribution of the
instrument's own moves over a chosen horizon, at a chosen severity, and carries its own frequency
with it. "A 5.1% two-hour fall" becomes "a 5.1% two-hour fall, the 1st percentile of 718 observed
windows, which occurred 7 times in 90 days, most recently on 2026-08-29." That is a sentence a
person can act on.

**Everything is conditioned on session phase, because our own measurements say it must be.** Open
and shut markets are different distributions for the same rToken: session-conditional beta differs
by a factor of seven on AAPL (0.668 open against 0.090 shut, ``data/session_beta.json``), and the
two-hour move clears the cost hurdle 58-86% of the time in regular hours and materially less often
outside them (``data/hurdle_clearance.json``). A tail estimated from blended bars is therefore an
average of two populations and describes neither. :func:`empirical_scenarios` takes a phase and
estimates within it, and refuses to report a percentile it does not have the observations to
support.

**Measured and assumed are never mixed.** Two kinds of scenario appear in a report. An *empirical*
one is a quantile of an observed distribution and carries its sample size. A *structural* one —
the hedge market closing, the venue going dark, evidence going stale, the spread widening — has no
frequency we have measured, because these are conditions rather than price moves, and each is
labelled with the assumption it rests on. A reader can always tell which is which, and
:attr:`StressReport.measured_share` states the proportion outright.

    python -m argus.desk.stress --symbol NVDAUSDT --quantity 1 --loss-tolerance 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.cost.model import CostModel
from argus.desk.workbench import Scenario, StressResult, stress_position
from argus.sim.market import MarketError, exit_cost_bps
from argus.truth.clocks import SessionPhase

_ZERO = Decimal("0")

MIN_OBSERVATIONS = 30
"""Fewest windows from which a quantile will be reported at all.

Below this the estimate is noise wearing a decimal point. Thirty is the conventional floor for a
sample mean to be worth quoting and is already generous for a *tail*: a 1st percentile of thirty
observations is the single worst one, which is a maximum, not a percentile. :func:`quantile_move`
refuses rather than returning a number that would be read as an estimate.
"""

SEVERITIES: tuple[tuple[str, Decimal], ...] = (
    ("typical", Decimal("50")),
    ("adverse", Decimal("10")),
    ("severe", Decimal("5")),
    ("extreme", Decimal("1")),
)
"""Named severities and the percentile of the move distribution each corresponds to.

Percentiles are taken from the *left* of the distribution, so a lower number is a worse outcome;
"typical" at the 50th is the median move rather than a loss, and is included so a reader can see
the ordinary case beside the tail.

Names rather than raw percentiles because a report is read by a person, and "extreme" carries the
meaning that "the 1st percentile" only carries to a statistician. The mapping is stated here so a
reader can convert back.
"""


REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "stress_report.json"
"""Where `--save` writes.

**This constant exists because the artefact did not have one.** `data/stress_report.json` is cited
across the docs, and this module computed every figure in it and then *printed the JSON to stdout* —
so the file could only ever have been produced by a human copying a terminal, and no command
regenerated it. The computation was never the missing piece; the last line was.
"""


def ordinal(value: Decimal) -> str:
    """1 -> "1st", 5 -> "5th", 50 -> "50th".

    Exists because "the 1th percentile" appeared in the first live output, and a report that reads
    as though nobody proofread it is a report a reader trusts less than they should.
    """
    whole = int(value)
    text = str(value.normalize()) if value != whole else str(whole)
    if 11 <= (whole % 100) <= 13:
        return f"{text}th"
    return text + {1: "st", 2: "nd", 3: "rd"}.get(whole % 10, "th")


class StressError(RuntimeError):
    """A stress test that cannot be run honestly. Raised rather than approximated."""


@dataclass(frozen=True, slots=True)
class Move:
    """One observed move over the horizon, kept with the time it ended and the phase it was in."""

    at: datetime
    pct: Decimal
    phase: str

    def as_dict(self) -> dict[str, Any]:
        return {"at": self.at.isoformat(), "pct": str(self.pct), "phase": self.phase}


def horizon_moves(
    closes: Sequence[tuple[datetime, Decimal]], *, bars: int, phase_of: Any = None
) -> list[Move]:
    """Percentage moves over a rolling window of ``bars``, one per window.

    Overlapping windows are used deliberately: a 24-hour horizon on hourly bars sampled
    non-overlapping would give 3 observations a day and throw away the other 23, and the tail is
    precisely where observations are scarce. The cost is that neighbouring observations are not
    independent, which matters for a significance test and not for "how bad did it get" — and the
    tail statistics here are descriptive, never used to claim significance.

    ``phase_of`` maps a timestamp to a session-phase string. ``None`` marks every move
    ``"unknown"``, which is honest and makes the result unusable for phase-conditioned work rather
    than quietly blending phases.
    """
    if bars < 1:
        raise StressError("a horizon of fewer than one bar is not a horizon")
    out: list[Move] = []
    for i in range(bars, len(closes)):
        start_price = closes[i - bars][1]
        if start_price <= 0:
            continue
        at, end_price = closes[i]
        pct = (end_price - start_price) / start_price * Decimal("100")
        out.append(Move(at=at, pct=pct, phase=(phase_of(at) if phase_of else "unknown")))
    return out


def quantile_move(moves: Sequence[Move], *, percentile: Decimal) -> Decimal:
    """The move at a percentile of the observed distribution, by nearest rank.

    Nearest rank rather than interpolation: an interpolated tail invents a value between two
    observations, and in a tail the two observations are often far apart. Every number this returns
    is a move that genuinely occurred.
    """
    if len(moves) < MIN_OBSERVATIONS:
        raise StressError(
            f"{len(moves)} observation(s) is below the {MIN_OBSERVATIONS} needed to quote a "
            f"percentile; a tail from this few points is a maximum, not an estimate"
        )
    if not Decimal("0") < percentile <= Decimal("100"):
        raise StressError(f"percentile must be in (0, 100], got {percentile}")
    ordered = sorted(m.pct for m in moves)
    rank = int((percentile / Decimal("100") * Decimal(len(ordered))).to_integral_value(
        rounding="ROUND_CEILING"
    ))
    return ordered[max(0, min(rank - 1, len(ordered) - 1))]


@dataclass(frozen=True, slots=True)
class EmpiricalScenario:
    """A shock that happened, with how often and how recently.

    The distinction from :class:`~argus.desk.workbench.Scenario` is the provenance: that one is a
    number someone chose, this one is a number the instrument produced.
    """

    name: str
    severity: str
    percentile: Decimal
    shock_pct: Decimal
    observations: int
    occurrences: int
    """How many observed windows were at least this bad. The frequency the round-number table
    cannot supply."""

    last_seen: datetime | None
    phase: str
    horizon_bars: int

    @property
    def frequency_pct(self) -> Decimal:
        return (
            Decimal(self.occurrences) / Decimal(self.observations) * Decimal("100")
            if self.observations else _ZERO
        )

    def to_scenario(self, *, liquidity_multiplier: Decimal = Decimal("1")) -> Scenario:
        return Scenario(
            name=self.name, price_shock_pct=self.shock_pct,
            liquidity_multiplier=liquidity_multiplier,
        )

    def render(self) -> str:
        when = f", most recently {self.last_seen.date()}" if self.last_seen else ""
        return (
            f"{self.severity}: {self.shock_pct:.2f}% over {self.horizon_bars} bar(s) in "
            f"{self.phase} — the {ordinal(self.percentile)} percentile of {self.observations} "
            f"observed windows, reached {self.occurrences} time(s) "
            f"({self.frequency_pct:.1f}%){when}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "severity": self.severity, "percentile": str(self.percentile),
            "shock_pct": str(self.shock_pct), "observations": self.observations,
            "occurrences": self.occurrences, "frequency_pct": str(self.frequency_pct),
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "phase": self.phase, "horizon_bars": self.horizon_bars, "measured": True,
        }


def empirical_scenarios(
    moves: Sequence[Move], *, phase: SessionPhase | str | None = None, horizon_bars: int,
    severities: Sequence[tuple[str, Decimal]] = SEVERITIES,
) -> list[EmpiricalScenario]:
    """Turn observed moves into named downside scenarios for one session phase.

    Filtering by phase before estimating is the whole point; see the module docstring. A phase with
    too few observations yields nothing rather than a blended figure, and the caller is expected to
    report that absence rather than substitute the blended one.
    """
    label = str(phase) if phase is not None else "all"
    pool = [m for m in moves if phase is None or m.phase == str(phase)]
    if len(pool) < MIN_OBSERVATIONS:
        return []

    out: list[EmpiricalScenario] = []
    for severity, percentile in severities:
        try:
            shock = quantile_move(pool, percentile=percentile)
        except StressError:
            continue
        at_least = [m for m in pool if m.pct <= shock]
        out.append(EmpiricalScenario(
            name=f"{severity}_{label}",
            severity=severity,
            percentile=percentile,
            shock_pct=shock,
            observations=len(pool),
            occurrences=len(at_least),
            last_seen=max((m.at for m in at_least), default=None),
            phase=label,
            horizon_bars=horizon_bars,
        ))
    return out


@dataclass(frozen=True, slots=True)
class StructuralScenario:
    """A condition rather than a price move, carried with the assumption it rests on.

    These are the ways a position goes wrong that a return distribution cannot express: the hedge
    venue is shut, the exchange is down, the evidence is hours stale, the spread has trebled. None
    of them has a frequency we have measured on this venue, and every one of them says so.
    """

    name: str
    description: str
    price_shock_pct: Decimal
    liquidity_multiplier: Decimal
    assumption: str
    """Where the numbers come from. Never blank — an unstated assumption reads as a measurement."""

    def to_scenario(self) -> Scenario:
        return Scenario(
            name=self.name, price_shock_pct=self.price_shock_pct,
            liquidity_multiplier=self.liquidity_multiplier,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "shock_pct": str(self.price_shock_pct),
            "liquidity_multiplier": str(self.liquidity_multiplier),
            "assumption": self.assumption, "measured": False,
        }


STRUCTURAL: tuple[StructuralScenario, ...] = (
    StructuralScenario(
        "anchor_shut", "the underlying market closes while the position is open",
        Decimal("0"), Decimal("0.33"),
        "liquidity multiplier 0.33 from the measured off-hours depth ratio used throughout "
        "(agents/meta_pm.py:56-63); no price shock is asserted because closure is not itself "
        "directional",
    ),
    StructuralScenario(
        "hedge_unavailable", "the anchor hedge cannot be reached, so the position is naked",
        Decimal("0"), Decimal("0.5"),
        "ASSUMPTION, NOT MEASURED: hedge failure is modelled as halved depth. The real cost is "
        "the residual risk the hedgeability surface prices, which this scenario does not attempt",
    ),
    StructuralScenario(
        "spread_treble", "the quoted spread widens to three times its normal width",
        Decimal("0"), Decimal("0.33"),
        "ASSUMPTION, NOT MEASURED: a trebled spread is represented as a third of the depth, which "
        "is a proxy for cost rather than a model of the book",
    ),
    StructuralScenario(
        "venue_outage", "the exchange stops accepting orders while the position is open",
        Decimal("0"), Decimal("0.01"),
        "ASSUMPTION, NOT MEASURED, AND THE PROXY IS KNOWN NOT TO BITE. Thin depth stands in for "
        "'no orders accepted', and that substitution does not hold: a rejected order fills "
        "NEVER, while a thin book still fills a small position cheaply. This scenario therefore "
        "PASSES for any position small enough to clear the remaining depth, and it passed here. "
        "Its previous wording said 'the exit test should fail; that failing is the finding' — it "
        "did not fail, and an adversarial audit found this exit reported at 12.2bps, CHEAPER than "
        "the normal-market baseline, i.e. the report implied an outage is a good moment to exit. "
        "The flooring that made the book un-thinnable is fixed (sim/market.py seeds scale with "
        "depth_multiplier), but the proxy itself is still the wrong mechanism: modelling "
        "rejection needs an order path that can refuse, which this simulator does not have. "
        "TREAT THIS SCENARIO AS UNTESTED, and do not read 'survives' here as evidence",
    ),
    StructuralScenario(
        "stale_evidence", "the evidence behind the thesis is hours old and no longer true",
        Decimal("0"), Decimal("1"),
        "no price or liquidity effect is asserted. Listed so a stress report cannot omit the "
        "failure mode that has no market shock at all",
    ),
)
"""Named structural conditions. Every one carries its assumption, and three say NOT MEASURED."""


@dataclass(frozen=True, slots=True)
class ReverseStressResult:
    """The real breaking point a forward stress test can only assume — added 2026-09-15, the plan's
    own named gap ("Needs reverse stress tests and hedge-leg failure").

    A forward test asks "how bad is scenario X for this position". A reverse test asks the more
    useful question the ``hedge_unavailable`` structural scenario admits it cannot answer: *this
    position's own* ``hedge_unavailable`` assumes liquidity halves for every position, regardless
    of size — its own docstring: "ASSUMPTION, NOT MEASURED... which this scenario does not
    attempt". This finds it, per position, by bisecting on the REAL simulated book
    (`sim.market.exit_cost_bps`, the same ABIDES-based simulation `stress_position` already uses
    with ``simulate=True``) rather than asserting one flat multiplier for every position size.
    """

    price_shock_pct: Decimal
    breaking_liquidity_multiplier: Decimal | None
    """The liquidity multiplier (fraction of normal book depth) at which this exact position first
    becomes NOT fully exitable, found by bisection between ``searched_floor`` and
    ``searched_ceiling``. ``None`` means the position stayed fully exitable even at the floor —
    honestly reported as "did not break within the searched range", never rounded to a number that
    was never actually observed."""

    always_breaks: bool
    """True when the position could not be fully exited even at ``searched_ceiling`` — the
    position is oversized for this book regardless of any liquidity assumption within range."""

    searched_floor: Decimal
    searched_ceiling: Decimal
    iterations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "price_shock_pct": str(self.price_shock_pct),
            "breaking_liquidity_multiplier": (
                None if self.breaking_liquidity_multiplier is None
                else str(self.breaking_liquidity_multiplier)
            ),
            "always_breaks": self.always_breaks,
            "searched_floor": str(self.searched_floor),
            "searched_ceiling": str(self.searched_ceiling),
            "iterations": self.iterations,
        }

    def render(self) -> str:
        if self.always_breaks:
            return (
                f"reverse stress ({self.price_shock_pct}% shock): position cannot be fully "
                f"exited even at the healthiest liquidity tested ({self.searched_ceiling}x) — "
                f"oversized for this book regardless of liquidity assumption"
            )
        if self.breaking_liquidity_multiplier is None:
            return (
                f"reverse stress ({self.price_shock_pct}% shock): stayed fully exitable down to "
                f"{self.searched_floor}x depth — no breaking point found in the searched range"
            )
        return (
            f"reverse stress ({self.price_shock_pct}% shock): breaks at "
            f"{self.breaking_liquidity_multiplier:.4f}x normal depth ({self.iterations} probes) — "
            f"the measured answer `hedge_unavailable`'s flat 0.5x assumption could not give"
        )


def reverse_stress_liquidity(
    *,
    quantity: Decimal,
    entry_price: Decimal,
    price_shock_pct: Decimal = Decimal("-10"),
    search_floor: Decimal = Decimal("0.01"),
    search_ceiling: Decimal = Decimal("1.0"),
    tolerance: Decimal = Decimal("0.005"),
    max_iterations: int = 24,
) -> ReverseStressResult:
    """Bisect on the real simulated book to find the liquidity level at which THIS position stops
    being fully exitable, at a fixed price shock.

    **Verified monotonic before trusting a bisection on it, not assumed.** Probed
    `sim.market.exit_cost_bps` directly across a size sweep (qty 500->4000 at fixed depth) and a
    depth sweep (depth 0.05->1.0 at a fixed qty straddling the boundary) before writing this:
    ``unfilled`` strictly decreases as depth rises, and ``fully_exited`` flips cleanly from False
    to True exactly once, never back — a bisection is valid over this range for this simulator.

    ``exit_cost_bps`` always exits via a SELL (`sim/market.py:273`), the same assumption
    `stress_position` already carries — this function does not introduce a new one.
    """
    shocked = entry_price * (Decimal("1") + price_shock_pct / Decimal("100"))

    def _exitable(depth_multiplier: Decimal) -> bool:
        try:
            m = exit_cost_bps(
                quantity=quantity, depth_multiplier=depth_multiplier, reference_price=shocked,
            )
        except MarketError:
            # A simulated book that cannot even produce a mid cannot exit anything through it.
            return False
        return m.fully_exited

    if not _exitable(search_ceiling):
        return ReverseStressResult(
            price_shock_pct=price_shock_pct, breaking_liquidity_multiplier=search_ceiling,
            always_breaks=True, searched_floor=search_floor, searched_ceiling=search_ceiling,
            iterations=0,
        )
    if _exitable(search_floor):
        return ReverseStressResult(
            price_shock_pct=price_shock_pct, breaking_liquidity_multiplier=None,
            always_breaks=False, searched_floor=search_floor, searched_ceiling=search_ceiling,
            iterations=0,
        )

    lo, hi = search_floor, search_ceiling  # invariant: lo is not exitable, hi is exitable
    iterations = 0
    while hi - lo >= tolerance and iterations < max_iterations:
        mid = (lo + hi) / Decimal("2")
        iterations += 1
        if _exitable(mid):
            hi = mid
        else:
            lo = mid

    return ReverseStressResult(
        price_shock_pct=price_shock_pct, breaking_liquidity_multiplier=hi, always_breaks=False,
        searched_floor=search_floor, searched_ceiling=search_ceiling, iterations=iterations,
    )


@dataclass
class StressReport:
    """One trade idea, stressed against history and against structure."""

    symbol: str
    quantity: Decimal
    entry_price: Decimal
    loss_tolerance_pct: Decimal
    generated_at: datetime
    horizon_bars: int
    phase: str
    empirical: list[EmpiricalScenario] = field(default_factory=list)
    structural: list[StructuralScenario] = field(default_factory=list)
    results: list[StressResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reverse_stress: ReverseStressResult | None = None
    """The real, measured liquidity breaking point at this report's worst tested price shock —
    ``None`` only when the report has no scenarios to take a worst shock from at all."""

    @property
    def failures(self) -> list[StressResult]:
        return [r for r in self.results if not r.survives]

    @property
    def unexitable(self) -> list[StressResult]:
        """Scenarios in which the position could not be fully liquidated at any price."""
        return [r for r in self.results if not r.exitable]

    @property
    def measured_share(self) -> Decimal:
        """Proportion of scenarios whose shock came from observed data.

        Stated outright so a reader never has to count. A report that is mostly assumption is a
        weaker report, and hiding that would make it look stronger than it is.
        """
        total = len(self.empirical) + len(self.structural)
        return (
            Decimal(len(self.empirical)) / Decimal(total) * Decimal("100") if total else _ZERO
        )

    UNTESTED_SCENARIOS = ("venue_outage",)
    """Scenarios whose stated proxy is known not to exercise what they name.

    `venue_outage` substitutes thin depth for 'no orders accepted'. A rejected order never fills;
    a thin book still fills a small position. The scenario therefore cannot fail for a position
    small enough to clear the residual depth, and reporting its pass as evidence would be
    counting a test that cannot fail. Named here rather than deleted, because the failure mode is
    real even though this proxy does not reach it.
    """

    @property
    def survives_all(self) -> bool:
        """True only if every scenario that CAN fail did not.

        **Excludes `UNTESTED_SCENARIOS`, and that exclusion is the point.** Before 2026-09-20 this
        counted `venue_outage`'s unearned pass toward an all-clear, so the headline safety claim of
        the risk tool rested partly on a check that could not fail.
        """
        testable = [r for r in self.results if r.scenario not in self.UNTESTED_SCENARIOS]
        if not testable:
            return False
        return not [r for r in self.failures if r.scenario not in self.UNTESTED_SCENARIOS]

    def render(self) -> list[str]:
        lines = [
            f"[stress] {self.quantity} {self.symbol} at {self.entry_price}, tolerance "
            f"{self.loss_tolerance_pct}%, horizon {self.horizon_bars} bar(s), phase {self.phase}",
        ]
        if self.empirical:
            lines.append("[stress] from this instrument's own history:")
            lines.extend(f"[stress]   {s.render()}" for s in self.empirical)
        else:
            lines.append(
                f"[stress] no empirical scenario: fewer than {MIN_OBSERVATIONS} observed windows "
                f"in {self.phase}, so no percentile is quoted rather than one being blended "
                f"across sessions"
            )
        lines.append(
            f"[stress] {len(self.structural)} structural condition(s) tested; "
            f"{self.measured_share:.0f}% of scenarios are measured, the rest are stated assumptions"
        )
        for r in self.results:
            verdict = "survives" if r.survives else "FAILS"
            exit_note = "" if r.exitable else " — POSITION NOT FULLY EXITABLE"
            measured = "measured" if r.exit_measured else "assumed"
            lines.append(
                f"[stress]   {r.scenario}: {r.pnl_pct:+.2f}%, exit {r.exit_cost_bps:.1f}bps "
                f"({measured}) — {verdict}{exit_note}"
            )
        if self.unexitable:
            lines.append(
                f"[stress] {len(self.unexitable)} scenario(s) leave a position that cannot be "
                f"fully liquidated; mark-to-market survival is not survival"
            )
        if self.reverse_stress is not None:
            lines.append(f"[stress] {self.reverse_stress.render()}")
        lines.extend(f"[stress] {n}" for n in self.notes)
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": str(self.quantity),
            "entry_price": str(self.entry_price),
            "loss_tolerance_pct": str(self.loss_tolerance_pct),
            "generated_at": self.generated_at.isoformat(),
            "horizon_bars": self.horizon_bars,
            "phase": self.phase,
            "measured_share_pct": str(self.measured_share),
            "survives_all": self.survives_all,
            "empirical": [s.as_dict() for s in self.empirical],
            "structural": [s.as_dict() for s in self.structural],
            "results": [
                {
                    "scenario": r.scenario, "pnl_pct": str(r.pnl_pct),
                    "exit_cost_bps": str(r.exit_cost_bps), "survives": r.survives,
                    "exitable": r.exitable, "exit_measured": r.exit_measured,
                    "unfilled_quantity": str(r.unfilled_quantity),
                }
                for r in self.results
            ],
            "reverse_stress": (
                None if self.reverse_stress is None else self.reverse_stress.as_dict()
            ),
            "notes": list(self.notes),
        }


def assess(
    *,
    symbol: str,
    quantity: Decimal,
    entry_price: Decimal,
    loss_tolerance_pct: Decimal,
    moves: Sequence[Move],
    phase: SessionPhase | str | None = None,
    horizon_bars: int = 2,
    structural: Sequence[StructuralScenario] = STRUCTURAL,
    cost: CostModel | None = None,
    simulate: bool = True,
    now: datetime | None = None,
) -> StressReport:
    """Stress one trade idea against its own history and against structural failure.

    The empirical half can be empty — a new listing, or a phase with too few windows — and the
    report then says so in those words. The structural half always runs, because a condition does
    not need a price history to be possible.
    """
    if quantity <= 0:
        raise StressError("a stress test of a zero position tests nothing")
    if entry_price <= 0:
        raise StressError("entry price must be positive")

    label = str(phase) if phase is not None else "all"
    empirical = empirical_scenarios(moves, phase=phase, horizon_bars=horizon_bars)
    scenarios = tuple(
        [s.to_scenario() for s in empirical] + [s.to_scenario() for s in structural]
    )
    results = stress_position(
        quantity=quantity, entry_price=entry_price,
        loss_tolerance_pct=loss_tolerance_pct, scenarios=scenarios,
        cost=cost, simulate=simulate,
    )
    # The real, measured breaking point at this report's own worst tested shock — reusing the
    # scenario set already built above rather than a separately-chosen severity, so the reverse
    # stress test answers "how bad, really, at the shock this report already considers worst"
    # instead of a number picked independently of everything else in the report.
    reverse: ReverseStressResult | None = None
    if scenarios:
        worst_shock = min(s.price_shock_pct for s in scenarios)
        reverse = reverse_stress_liquidity(
            quantity=quantity, entry_price=entry_price, price_shock_pct=worst_shock,
        )
    report = StressReport(
        symbol=symbol, quantity=quantity, entry_price=entry_price,
        loss_tolerance_pct=loss_tolerance_pct,
        generated_at=now or datetime.now(UTC),
        horizon_bars=horizon_bars, phase=label,
        empirical=empirical, structural=list(structural), results=results,
        reverse_stress=reverse,
    )
    if not empirical:
        report.notes.append(
            "the empirical half is absent, so every shock in this report is an assumption someone "
            "chose; treat the survival verdict accordingly"
        )
    if report.unexitable:
        report.notes.append(
            "exit failure is the finding a return table cannot produce: the position is worth "
            "something on paper and cannot be sold"
        )
    return report


def phase_of_timestamp(at: datetime) -> str:
    """Session phase for a timestamp, using the same clock the desk decides on.

    Imported lazily and wrapped so a stress report never fails because a clock lookup did; an
    unknown phase is marked unknown, and an unknown phase simply does not match a phase filter.
    """
    try:
        from argus.truth.clocks import DualClock

        return str(DualClock().state(at, nav_age_seconds=0.0).phase)
    except Exception:  # a clock failure must not take the report down
        return "unknown"


def _load_moves(symbol: str, *, horizon_bars: int, days: int) -> tuple[list[Move], list[str]]:
    """Fetch candles and turn them into phase-tagged moves. Returns the moves and any status."""
    from argus.market.history import CandleType, fetch_range

    status: list[str] = []
    try:
        candles = fetch_range(symbol, interval="1H", days=days, candle_type=CandleType.MARKET)
    except Exception as exc:
        status.append(f"history unavailable ({type(exc).__name__}: {exc}); no empirical scenarios")
        return [], status
    closes = [(c.ts, c.close) for c in candles]
    status.append(f"{len(closes)} hourly bar(s) over {days} day(s)")
    return horizon_moves(closes, bars=horizon_bars, phase_of=phase_of_timestamp), status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARGUS decision stress testing")
    parser.add_argument("--symbol", default="NVDAUSDT")
    parser.add_argument("--quantity", type=Decimal, default=Decimal("1"))
    parser.add_argument("--entry-price", type=Decimal, default=None,
                        help="defaults to the latest close")
    parser.add_argument("--loss-tolerance", type=Decimal, default=Decimal("5"),
                        help="percent of notional the trader will accept losing")
    parser.add_argument("--horizon-bars", type=int, default=2)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--phase", default=str(SessionPhase.RTH),
                        help="session phase to condition on, or 'all'")
    parser.add_argument(
        "--save", action="store_true",
        help="write the report to data/stress_report.json instead of only printing it",
    )
    parser.add_argument("--no-simulate", action="store_true",
                        help="use the arithmetic exit cost instead of a simulated liquidation")
    args = parser.parse_args(argv)

    moves, status = _load_moves(
        args.symbol, horizon_bars=args.horizon_bars, days=args.days
    )
    entry = args.entry_price
    if entry is None:
        from argus.market.bitget import fetch_rtokens

        try:
            entry = fetch_rtokens()[args.symbol].last
        except Exception as exc:
            print(f"could not read a price for {args.symbol}: {exc}", file=sys.stderr)
            return 1

    phase: str | None = None if args.phase == "all" else args.phase
    report = assess(
        symbol=args.symbol, quantity=args.quantity, entry_price=entry,
        loss_tolerance_pct=args.loss_tolerance, moves=moves, phase=phase,
        horizon_bars=args.horizon_bars, simulate=not args.no_simulate,
    )
    report.notes.extend(status)
    for line in report.render():
        print(line)

    blob = report.as_dict()
    if args.save:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(blob, indent=2, default=str), encoding="utf-8")
        print(f"\nwritten to {REPORT_PATH}")
    else:
        print(json.dumps(blob, indent=2, default=str))
    return 0 if report.survives_all else 1


__all__ = [
    "MIN_OBSERVATIONS",
    "REPORT_PATH",
    "SEVERITIES",
    "STRUCTURAL",
    "EmpiricalScenario",
    "Move",
    "ReverseStressResult",
    "StressError",
    "StressReport",
    "StructuralScenario",
    "assess",
    "empirical_scenarios",
    "horizon_moves",
    "main",
    "ordinal",
    "phase_of_timestamp",
    "quantile_move",
    "reverse_stress_liquidity",
]


if __name__ == "__main__":
    raise SystemExit(main())
