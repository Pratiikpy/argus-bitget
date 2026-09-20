"""ARGUS's ``Mandate`` vs. Vibe-Trading's real ``check_mandate()`` — same input, both systems.

``eval/standing.py``'s "Per-profile mandate that changes the verdict" capability names
``hkuds/vibe-trading``'s ``agent/src/live/enforcement.py:458-609`` as its baseline. Reading that
file (in full — all 797 lines, this module's docstring cites exact ranges below) is
``best_method_studied``; this module is ``baseline_reproduced`` and ``same_input_comparison``: it
runs Vibe-Trading's own ``check_mandate()``, vendored byte-verified in ``eval/baselines/`` (see that
package's docstring), against the same constructed scenarios ARGUS's own
:func:`argus.agents.mandate.Mandate.out_of_mandate` sees.

**The two systems check different things, and this module says so rather than hiding it.**
Vibe-Trading's gate (``check_mandate``, ``enforcement.py:455-617``) enforces, in a fixed fail-fast
order: exclude-list, instrument-type allowlist, asset-class allowlist, single-order notional,
post-trade total exposure, post-trade gross leverage, daily order count, a funding-ceiling
defense-in-depth check, and (skipped here — see :mod:`argus.eval.baselines`) market-cap/liquidity
floors. ARGUS's ``Mandate.out_of_mandate`` checks holding horizon, position-size percentage of
capital, an excluded-symbol list, hedge availability, a confidence floor, and a concurrent-position
cap, and — unlike Vibe-Trading — collects every applicable reason rather than stopping at the
first. Four dimensions overlap (notional/position-size, excluded symbols) or nearly so; the rest are
each system's own. ``no_specialist_capability_superior`` at the bottom of this module states the
scope this is honestly claimed over, not "ARGUS wins everything a pre-trade gate could check" —
portfolio-wide leverage and gross exposure are a *different* ARGUS capability's job
(``agents.desk.ConstitutionPolicy.max_gross_exposure_notional`` /
``max_signed_exposure_notional`` — the class at ``agents/desk.py:753``, its ``rule()`` method
that enforces the exposure ceiling at ``agents/desk.py:958`` — checked directly before this claim
was written), not this one's.

Also see ``research/architecture/personalisation-audit.md`` for the corrected record of two
FABRICATED citations this project previously carried about Vibe-Trading (a mandate-injection
mechanism attributed to a real file that does not contain it, and an evidence-ordering function
attributed to a Vibe-Trading path that does not exist at all) — this module is the source-verified
replacement for both.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from argus.agents.mandate import Mandate
from argus.desk.workbench import TraderProfile
from argus.eval.baselines.loader import BaselineLoadError, VibeTradingSymbols, load_baseline

_HUNDRED = Decimal("100")


class MandateComparisonError(RuntimeError):
    """The comparison could not run — the baseline failed to load or a scenario was malformed."""


@dataclass(frozen=True)
class Scenario:
    """One order, proposed by one profile's trader, against one account state.

    A superset of both systems' native inputs. Fields meaningful to only one side are held at a
    neutral non-triggering value on scenarios that are not specifically probing that dimension
    (documented per-scenario below), never omitted — a field silently defaulted away would hide
    which system's blind spot a given disagreement comes from.
    """

    name: str
    profile: TraderProfile
    symbol: str
    side: str
    notional_usd: Decimal
    thesis_horizon_hours: float
    hedge_available: bool | None = None
    confidence: float | None = None
    open_positions: int | None = None
    # Vibe-Trading-only dimensions (no ARGUS Mandate equivalent).
    max_leverage: float = 1000.0
    """Effectively unlimited unless a scenario is specifically probing leverage."""
    max_total_exposure_usd: float = 10_000_000.0
    """Effectively unlimited unless a scenario is specifically probing total exposure."""
    max_trades_per_day: int = 1_000_000
    daily_count: int = 0
    existing_positions: tuple[dict[str, Any], ...] = ()

    def account_funding_usd(self) -> float:
        return float(self.profile.capital)


@dataclass(frozen=True)
class SideVerdict:
    """One system's answer to one scenario, in a shape the other side did not have to produce."""

    allowed: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class ComparisonResult:
    scenario_name: str
    argus: SideVerdict
    vibe_trading: SideVerdict

    @property
    def agree_on_allow(self) -> bool:
        """Both sides reached the same allow/refuse verdict, for whatever reason each has."""
        return self.argus.allowed == self.vibe_trading.allowed

    @property
    def only_argus_refuses(self) -> bool:
        return (not self.argus.allowed) and self.vibe_trading.allowed

    @property
    def only_vibe_trading_refuses(self) -> bool:
        return self.argus.allowed and (not self.vibe_trading.allowed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "argus": self.argus.as_dict(),
            "vibe_trading": self.vibe_trading.as_dict(),
            "agree_on_allow": self.agree_on_allow,
            "only_argus_refuses": self.only_argus_refuses,
            "only_vibe_trading_refuses": self.only_vibe_trading_refuses,
        }


def run_argus(scenario: Scenario) -> SideVerdict:
    """Run the scenario through ARGUS's real ``Mandate.out_of_mandate`` — no reimplementation."""
    mandate = Mandate(profile=scenario.profile)
    reasons = mandate.out_of_mandate(
        horizon_hours=scenario.thesis_horizon_hours,
        notional=scenario.notional_usd,
        symbol=scenario.symbol,
        hedge_available=scenario.hedge_available,
        confidence=scenario.confidence,
        open_positions=scenario.open_positions,
    )
    return SideVerdict(allowed=not reasons, reasons=reasons)


def run_vibe_trading(scenario: Scenario, baseline: VibeTradingSymbols) -> SideVerdict:
    """Run the scenario through Vibe-Trading's real, vendored ``check_mandate`` — no paraphrase."""
    caps = baseline.hard_caps(
        account_funding_usd=scenario.account_funding_usd(),
        max_order_notional_usd=float(
            scenario.profile.capital * scenario.profile.max_position_pct / _HUNDRED
        ),
        max_total_exposure_usd=scenario.max_total_exposure_usd,
        max_leverage=scenario.max_leverage,
        allowed_instruments=(baseline.instrument_type.EQUITY,),
        max_trades_per_day=scenario.max_trades_per_day,
    )
    universe = baseline.universe_constraint(
        asset_classes=(baseline.asset_class.US_EQUITY,),
        min_market_cap_usd=None,
        min_avg_daily_volume_usd=None,
        exclude_symbols=tuple(s.upper() for s in scenario.profile.excluded_symbols),
    )
    consent = baseline.consent_meta(
        created_at="2026-01-01T00:00:00+00:00",
        consent_token_sha256="mandate_comparison_scenario",
        broker="argus_comparison",
        account_ref="argus_comparison_account",
        expires_at="2099-01-01T00:00:00+00:00",
    )
    mandate = baseline.mandate(schema_version=1, hard_caps=caps, universe=universe, consent=consent)
    intent = baseline.order_intent(
        symbol=scenario.symbol.upper(),
        side=scenario.side,
        notional_usd=float(scenario.notional_usd),
        quantity=None,
        instrument_type=baseline.instrument_type.EQUITY,
        asset_class=baseline.asset_class.US_EQUITY,
    )
    breach = baseline.check_mandate(
        mandate,
        intent,
        positions=list(scenario.existing_positions),
        balance={"equity": scenario.account_funding_usd()},
        broker="argus_comparison",
        remote_tool="place_order",
        daily_count=scenario.daily_count,
    )
    if breach is None:
        return SideVerdict(allowed=True, reasons=())
    reason = (
        breach.detail
        or f"{breach.limit} breached: {breach.attempted_value} > {breach.limit_value}"
    )
    return SideVerdict(allowed=False, reasons=(f"[{breach.kind}/{breach.limit}] {reason}",))


def compare(scenario: Scenario, baseline: VibeTradingSymbols) -> ComparisonResult:
    return ComparisonResult(
        scenario_name=scenario.name,
        argus=run_argus(scenario),
        vibe_trading=run_vibe_trading(scenario, baseline),
    )


# =============================================================================================
# Hand-designed scenarios — one per dimension, each built to probe a specific boundary.
# =============================================================================================


def _conservative() -> TraderProfile:
    return TraderProfile.conservative()


def _aggressive() -> TraderProfile:
    return TraderProfile.aggressive()


def designed_scenarios() -> tuple[Scenario, ...]:
    """Hand-built boundary cases, one per checked dimension, using ARGUS's real shipped profiles.

    Not a random sample — each one is deliberately constructed to sit just past one specific limit
    with every other limit held comfortably clear, so a disagreement is attributable to exactly one
    dimension. :func:`swept_scenarios` is the out-of-sample counterpart: a grid nobody hand-tuned.
    """
    conservative = _conservative()
    aggressive = _aggressive()
    return (
        Scenario(
            name="clean_allow",
            profile=aggressive,
            symbol="NVDAUSDT",
            side="buy",
            notional_usd=Decimal("5000"),
            thesis_horizon_hours=4.0,
        ),
        Scenario(
            name="excluded_symbol_both_should_refuse",
            profile=conservative,
            symbol="TQQQUSDT",
            side="buy",
            notional_usd=Decimal("1000"),
            thesis_horizon_hours=4.0,
        ),
        Scenario(
            name="oversized_notional_both_should_refuse",
            profile=conservative,
            symbol="NVDAUSDT",
            side="buy",
            notional_usd=Decimal("50000"),
            thesis_horizon_hours=4.0,
            hedge_available=True,
            confidence=0.9,
        ),
        Scenario(
            name="horizon_breach_argus_only",
            profile=aggressive,
            symbol="NVDAUSDT",
            side="buy",
            notional_usd=Decimal("5000"),
            thesis_horizon_hours=200.0,
        ),
        Scenario(
            name="hedge_unavailable_argus_only",
            profile=conservative,
            symbol="NVDAUSDT",
            side="buy",
            notional_usd=Decimal("1000"),
            thesis_horizon_hours=4.0,
            hedge_available=False,
            confidence=0.9,
        ),
        Scenario(
            name="confidence_floor_argus_only",
            profile=conservative,
            symbol="NVDAUSDT",
            side="buy",
            notional_usd=Decimal("1000"),
            thesis_horizon_hours=4.0,
            hedge_available=True,
            confidence=0.5,
        ),
        Scenario(
            name="concurrent_positions_cap_argus_only",
            profile=conservative,
            symbol="NVDAUSDT",
            side="buy",
            notional_usd=Decimal("1000"),
            thesis_horizon_hours=4.0,
            hedge_available=True,
            confidence=0.9,
            open_positions=3,
        ),
        Scenario(
            name="leverage_breach_vibe_trading_only",
            profile=aggressive,
            symbol="NVDAUSDT",
            side="buy",
            notional_usd=Decimal("5000"),
            thesis_horizon_hours=4.0,
            max_leverage=0.01,
        ),
        Scenario(
            name="daily_count_breach_vibe_trading_only",
            profile=aggressive,
            symbol="NVDAUSDT",
            side="buy",
            notional_usd=Decimal("500"),
            thesis_horizon_hours=4.0,
            max_trades_per_day=3,
            daily_count=3,
        ),
        Scenario(
            name="total_exposure_breach_vibe_trading_only",
            profile=aggressive,
            symbol="NVDAUSDT",
            side="buy",
            notional_usd=Decimal("500"),
            thesis_horizon_hours=4.0,
            max_total_exposure_usd=1000.0,
            existing_positions=({"symbol": "NVDAUSDT", "quantity": 10, "price": 90.0},),
        ),
        Scenario(
            name="funding_ceiling_defense_in_depth_vibe_trading_only",
            profile=aggressive,
            symbol="NVDAUSDT",
            side="buy",
            notional_usd=Decimal("500"),
            thesis_horizon_hours=4.0,
            max_total_exposure_usd=10_000_000.0,
            existing_positions=(
                {"symbol": "NVDAUSDT", "quantity": 2000, "price": 90.0},
            ),
        ),
    )


_EXPECTED_DESIGNED = {
    "clean_allow": (True, True),
    "excluded_symbol_both_should_refuse": (False, False),
    "oversized_notional_both_should_refuse": (False, False),
    "horizon_breach_argus_only": (False, True),
    "hedge_unavailable_argus_only": (False, True),
    "confidence_floor_argus_only": (False, True),
    "concurrent_positions_cap_argus_only": (False, True),
    "leverage_breach_vibe_trading_only": (True, False),
    "daily_count_breach_vibe_trading_only": (True, False),
    "total_exposure_breach_vibe_trading_only": (True, False),
    "funding_ceiling_defense_in_depth_vibe_trading_only": (True, False),
}
"""(argus_allowed, vibe_trading_allowed) this module's own design intends for each scenario.

Checked by :func:`run_designed` against what actually happened — a scenario that does not match its
intent is a bug in the scenario, not a footnote, and the comparison must say so rather than report
the mismatch as if it were the finding."""


@dataclass(frozen=True)
class DesignedRun:
    results: tuple[ComparisonResult, ...]
    mismatches: tuple[str, ...]
    """Scenario names whose actual verdict pair did not match this module's own stated intent."""

    @property
    def design_is_sound(self) -> bool:
        return not self.mismatches

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "mismatches": list(self.mismatches),
            "design_is_sound": self.design_is_sound,
        }


def run_designed(baseline: VibeTradingSymbols) -> DesignedRun:
    results = tuple(compare(s, baseline) for s in designed_scenarios())
    mismatches = tuple(
        r.scenario_name
        for r in results
        if (r.argus.allowed, r.vibe_trading.allowed) != _EXPECTED_DESIGNED[r.scenario_name]
    )
    return DesignedRun(results=results, mismatches=mismatches)


# =============================================================================================
# Out-of-sample sweep — a parameter grid nobody hand-tuned to hit the boundaries above.
# =============================================================================================


def swept_scenarios() -> tuple[Scenario, ...]:
    """A deterministic grid over realistic values, none chosen to land on a specific limit.

    Deterministic (``itertools.product`` over fixed grids), not RNG-seeded: a seed changes meaning
    across Python versions and a hand-picked seed value is itself a form of hand-tuning. Every
    combination here is generated the same way regardless of whether it happens to breach anything,
    which is the actual definition of "not tuned to the boundary" this needs.

    **A first version of this grid held ``max_trades_per_day`` fixed at 100 against a
    ``daily_count`` grid topping out at 50, held ``max_total_exposure_usd`` fixed at 10,000,000
    with ``existing_positions`` always empty, and never swept a ``max_leverage`` below 1.0 against
    notional fractions that top out at 40% of capital.** All three meant the corresponding checks
    in Vibe-Trading's ``check_mandate`` could never trip — running it produced
    ``vibe_trading_limits_seen == ['max_order_notional_usd']`` only, one dimension out of six, which
    would have UNDERSTATED Vibe-Trading's real coverage and correspondingly overstated the
    "blind spot" asymmetry between the two systems. Caught by actually running the sweep and
    reading its own output before treating the numbers as evidence, not assumed correct on write.
    Fixed by pairing ``max_trades_per_day`` with ``daily_count`` and ``max_total_exposure_usd``
    with ``existing_positions`` as explicit (restrictive, loose) pairs — a full independent cross
    would also work but adds scenarios that are redundant in what they can newly reveal — and by
    adding a sub-1.0 ``max_leverage`` value that the notional grid can actually breach.
    """
    profiles = (_conservative(), _aggressive())
    notional_fractions = (Decimal("0.01"), Decimal("0.08"), Decimal("0.20"), Decimal("0.40"))
    horizons = (2.0, 96.0, 400.0)
    hedge_states: tuple[bool | None, ...] = (None, True, False)
    confidences: tuple[float | None, ...] = (None, 0.5, 0.8)
    open_positions_states: tuple[int | None, ...] = (None, 0, 5)
    # 0.1 is restrictive enough that even the smallest notional_fraction (1% of capital = 0.01
    # post-leverage) still clears it, while the larger fractions (8%/20%/40%) do not — so this one
    # value alone produces both trips and clean passes depending on the paired notional fraction,
    # rather than needing its own separate trip/loose pairing the way the two grids below do.
    leverages = (0.1, 1.0, 1000.0)
    # (max_trades_per_day, daily_count) as explicit pairs, not an independent cross: a "restrictive
    # cap, small count" pair and a "small cap, at-cap count" pair both actually trip; "loose cap,
    # any realistic count" never trips. An independent cross would add cells like (loose cap, huge
    # count) that reveal nothing a smaller grid doesn't already cover.
    daily_pairs = ((100, 0), (100, 50), (3, 0), (3, 3), (3, 5))
    # (max_total_exposure_usd, existing_positions) as explicit pairs, same reasoning: one loose+
    # empty pair that never trips, one restrictive+already-large-book pair where the EXISTING
    # position alone (30 * 90.0 = 2,700 usd) already exceeds the 2,000 cap before the new order's
    # own notional is even added — so this pair specifically exercises "total exposure distinct
    # from single-order notional" (Vibe-Trading's own §5-6 distinction), not just a bigger notional.
    exposure_pairs: tuple[tuple[float, tuple[dict[str, Any], ...]], ...] = (
        (10_000_000.0, ()),
        (2_000.0, ({"symbol": "NVDAUSDT", "quantity": 30, "price": 90.0},)),
    )

    out: list[Scenario] = []
    combos = itertools.product(
        profiles, notional_fractions, horizons, hedge_states, confidences,
        open_positions_states, leverages, daily_pairs, exposure_pairs,
    )
    for i, combo in enumerate(combos):
        profile, frac, horizon, hedge, conf, openp, lev, daily_pair, exposure_pair = combo
        notional = (profile.capital * frac).quantize(Decimal("1"))
        max_trades, daily = daily_pair
        max_exposure, existing = exposure_pair
        out.append(
            Scenario(
                name=f"sweep_{i:05d}",
                profile=profile,
                symbol="NVDAUSDT",
                side="buy",
                notional_usd=notional,
                thesis_horizon_hours=horizon,
                hedge_available=hedge,
                confidence=conf,
                open_positions=openp,
                max_leverage=lev,
                max_trades_per_day=max_trades,
                daily_count=daily,
                max_total_exposure_usd=max_exposure,
                existing_positions=existing,
            )
        )
    return tuple(out)


@dataclass(frozen=True)
class SweepSummary:
    total: int
    agree_on_allow: int
    only_argus_refuses: int
    only_vibe_trading_refuses: int
    both_refuse: int
    """Distinct from ``agree_on_allow``: specifically both-refuse, not both-allow."""
    argus_reasons_seen: tuple[str, ...]
    """Which of ARGUS's own reason-templates fired at least once in the sweep, by keyword."""
    vibe_trading_limits_seen: tuple[str, ...]
    """Which of Vibe-Trading's own ``limit``/``kind`` breach names actually fired, by keyword."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "agree_on_allow": self.agree_on_allow,
            "only_argus_refuses": self.only_argus_refuses,
            "only_vibe_trading_refuses": self.only_vibe_trading_refuses,
            "both_refuse": self.both_refuse,
            "argus_reasons_seen": list(self.argus_reasons_seen),
            "vibe_trading_limits_seen": list(self.vibe_trading_limits_seen),
        }


_ARGUS_REASON_KEYWORDS = (
    "different trader's trade",  # horizon
    "exceeds the",  # position pct
    "exclusion list",  # excluded symbol
    "unhedgeable exposure",  # hedge
    "confidence",  # confidence floor
    "already open",  # concurrent positions
)
_VIBE_LIMIT_KEYWORDS = (
    "max_order_notional_usd", "max_total_exposure_usd", "max_leverage",
    "max_trades_per_day", "account_funding_usd", "exclude_symbols",
)


def run_sweep(baseline: VibeTradingSymbols) -> tuple[tuple[ComparisonResult, ...], SweepSummary]:
    scenarios = swept_scenarios()
    results = tuple(compare(s, baseline) for s in scenarios)

    argus_reasons_all = " | ".join(r for res in results for r in res.argus.reasons)
    vibe_reasons_all = " | ".join(r for res in results for r in res.vibe_trading.reasons)
    summary = SweepSummary(
        total=len(results),
        agree_on_allow=sum(1 for r in results if r.agree_on_allow),
        only_argus_refuses=sum(1 for r in results if r.only_argus_refuses),
        only_vibe_trading_refuses=sum(1 for r in results if r.only_vibe_trading_refuses),
        both_refuse=sum(
            1 for r in results if not r.argus.allowed and not r.vibe_trading.allowed
        ),
        argus_reasons_seen=tuple(k for k in _ARGUS_REASON_KEYWORDS if k in argus_reasons_all),
        vibe_trading_limits_seen=tuple(k for k in _VIBE_LIMIT_KEYWORDS if k in vibe_reasons_all),
    )
    return results, summary


# =============================================================================================
# Ablation — each check, independently, actually changes the verdict.
# =============================================================================================


@dataclass(frozen=True)
class AblationCase:
    dimension: str
    system: str
    """"argus" or "vibe_trading" — which side this check belongs to."""
    tripped: ComparisonResult
    """The scenario with this dimension pushed past its limit."""
    cleared: ComparisonResult
    """The identical scenario with only this dimension pulled back to a passing value."""

    @property
    def check_is_load_bearing(self) -> bool:
        """True when tripping this ONE dimension, and only this one, flips this system's verdict."""
        side = "argus" if self.system == "argus" else "vibe_trading"
        tripped_side = getattr(self.tripped, side)
        cleared_side = getattr(self.cleared, side)
        return bool(tripped_side.allowed is False and cleared_side.allowed is True)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "system": self.system,
            "tripped": self.tripped.as_dict(),
            "cleared": self.cleared.as_dict(),
            "check_is_load_bearing": self.check_is_load_bearing,
        }


def ablation_cases(baseline: VibeTradingSymbols) -> tuple[AblationCase, ...]:
    """One tripped/cleared pair per check on each side — every check independently pulls weight."""
    conservative = _conservative()
    aggressive = _aggressive()

    def pair(
        name: str, system: str, base: Scenario, cleared_overrides: dict[str, Any]
    ) -> AblationCase:
        cleared = _replace(base, **cleared_overrides)
        return AblationCase(
            dimension=name, system=system,
            tripped=compare(base, baseline), cleared=compare(cleared, baseline),
        )

    def _replace(scenario: Scenario, **overrides: Any) -> Scenario:
        from dataclasses import replace
        return replace(scenario, **overrides)

    cases: list[AblationCase] = []

    horizon_trip = Scenario(
        name="ablation_horizon", profile=aggressive, symbol="NVDAUSDT", side="buy",
        notional_usd=Decimal("5000"), thesis_horizon_hours=200.0,
    )
    cases.append(pair("horizon", "argus", horizon_trip, {"thesis_horizon_hours": 4.0}))

    notional_trip = Scenario(
        name="ablation_notional", profile=conservative, symbol="NVDAUSDT", side="buy",
        notional_usd=Decimal("50000"), thesis_horizon_hours=4.0,
        hedge_available=True, confidence=0.9,
    )
    cases.append(
        pair("position_notional_pct", "argus", notional_trip, {"notional_usd": Decimal("1000")})
    )

    exclusion_trip = Scenario(
        name="ablation_exclusion", profile=conservative, symbol="TQQQUSDT", side="buy",
        notional_usd=Decimal("1000"), thesis_horizon_hours=4.0,
        hedge_available=True, confidence=0.9,
    )
    cases.append(pair("excluded_symbol", "argus", exclusion_trip, {"symbol": "NVDAUSDT"}))

    hedge_trip = Scenario(
        name="ablation_hedge", profile=conservative, symbol="NVDAUSDT", side="buy",
        notional_usd=Decimal("1000"), thesis_horizon_hours=4.0,
        hedge_available=False, confidence=0.9,
    )
    cases.append(pair("hedge_required", "argus", hedge_trip, {"hedge_available": True}))

    confidence_trip = Scenario(
        name="ablation_confidence", profile=conservative, symbol="NVDAUSDT", side="buy",
        notional_usd=Decimal("1000"), thesis_horizon_hours=4.0,
        hedge_available=True, confidence=0.5,
    )
    cases.append(pair("confidence_floor", "argus", confidence_trip, {"confidence": 0.9}))

    concurrent_trip = Scenario(
        name="ablation_concurrent", profile=conservative, symbol="NVDAUSDT", side="buy",
        notional_usd=Decimal("1000"), thesis_horizon_hours=4.0,
        hedge_available=True, confidence=0.9, open_positions=3,
    )
    cases.append(pair("concurrent_positions", "argus", concurrent_trip, {"open_positions": 0}))

    leverage_trip = Scenario(
        name="ablation_leverage", profile=aggressive, symbol="NVDAUSDT", side="buy",
        notional_usd=Decimal("5000"), thesis_horizon_hours=4.0, max_leverage=0.01,
    )
    cases.append(pair("leverage", "vibe_trading", leverage_trip, {"max_leverage": 1000.0}))

    daily_trip = Scenario(
        name="ablation_daily_count", profile=aggressive, symbol="NVDAUSDT", side="buy",
        notional_usd=Decimal("500"), thesis_horizon_hours=4.0,
        max_trades_per_day=3, daily_count=3,
    )
    cases.append(pair("daily_trade_count", "vibe_trading", daily_trip, {"daily_count": 0}))

    exposure_trip = Scenario(
        name="ablation_total_exposure", profile=aggressive, symbol="NVDAUSDT", side="buy",
        notional_usd=Decimal("500"), thesis_horizon_hours=4.0,
        max_total_exposure_usd=1000.0,
        existing_positions=({"symbol": "NVDAUSDT", "quantity": 10, "price": 90.0},),
    )
    cases.append(
        pair(
            "total_exposure", "vibe_trading", exposure_trip,
            {"max_total_exposure_usd": 10_000_000.0},
        )
    )

    vibe_exclusion_trip = Scenario(
        name="ablation_vibe_exclusion", profile=aggressive, symbol="TQQQUSDT", side="buy",
        notional_usd=Decimal("500"), thesis_horizon_hours=4.0,
    )
    # aggressive() carries no excluded_symbols, so this pair demonstrates the check via the
    # conservative profile's own exclusion list, applied to an aggressive-profile-shaped order.
    cases.append(
        pair(
            "exclude_list_both_systems", "vibe_trading",
            Scenario(
                name="ablation_vibe_exclusion", profile=conservative, symbol="TQQQUSDT", side="buy",
                notional_usd=Decimal("500"), thesis_horizon_hours=4.0,
                hedge_available=True, confidence=0.9,
            ),
            {"symbol": "NVDAUSDT"},
        )
    )
    del vibe_exclusion_trip  # superseded by the pair() call above; kept unused var out of the list

    return tuple(cases)


# =============================================================================================
# Costs — the measurable size of each system's blind spot, not just its existence.
# =============================================================================================


@dataclass(frozen=True)
class BlindSpotCosts:
    """How often, across the sweep, each system silently allows what the other would refuse.

    Not a defect in either system — vibe-trading was never built to know about a trader's holding
    horizon, and ARGUS's Mandate was deliberately not built to police leverage (that is
    ``agents.desk.ConstitutionPolicy``'s job). The point of measuring it is that "structural blind
    spot" is currently an assertion; this is what makes it a number — the ``costs_included``
    condition asks for a real cost, and for a compliance gate the real cost of an unrepresented
    dimension is exactly how often it would have silently let something through.
    """

    total_scenarios: int
    argus_blind_spot_count: int
    """Vibe-Trading refuses, ARGUS allows — ARGUS's blind spot size on THIS sweep."""
    argus_blind_spot_rate: float
    vibe_trading_blind_spot_count: int
    """ARGUS refuses, Vibe-Trading allows — Vibe-Trading's blind spot size on THIS sweep."""
    vibe_trading_blind_spot_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_scenarios": self.total_scenarios,
            "argus_blind_spot_count": self.argus_blind_spot_count,
            "argus_blind_spot_rate": round(self.argus_blind_spot_rate, 4),
            "vibe_trading_blind_spot_count": self.vibe_trading_blind_spot_count,
            "vibe_trading_blind_spot_rate": round(self.vibe_trading_blind_spot_rate, 4),
        }


def blind_spot_costs(results: tuple[ComparisonResult, ...]) -> BlindSpotCosts:
    total = len(results)
    argus_blind = sum(1 for r in results if r.only_vibe_trading_refuses)
    vibe_blind = sum(1 for r in results if r.only_argus_refuses)
    return BlindSpotCosts(
        total_scenarios=total,
        argus_blind_spot_count=argus_blind,
        argus_blind_spot_rate=argus_blind / total if total else 0.0,
        vibe_trading_blind_spot_count=vibe_blind,
        vibe_trading_blind_spot_rate=vibe_blind / total if total else 0.0,
    )


# =============================================================================================
# Scope statement — what "no specialist capability superior" is and is not being claimed over.
# =============================================================================================

SCOPE_STATEMENT = """\
Claimed: within the scope `agents.mandate.Mandate` actually covers — per-trader personalisation \
that changes a decision (holding horizon, position size as a percentage of THIS trader's capital, \
a hard per-symbol exclusion list, hedge-availability refusal, a confidence floor, a concurrent-\
position cap) AND reaches the reasoning layer before a thesis is written (`agents/desk.py`'s \
`mandate_block` + `ordered_evidence`, read inside `MarketFrame.to_prompt_block()`) — no studied \
specialist capability is superior. Vibe-Trading's `check_mandate()` has no representation for \
five of ARGUS's six dimensions (only the exclude-list and a notional cap overlap) and, verified \
directly across every file in its repository that could plausibly carry it, no mechanism \
anywhere that reaches an LLM's reasoning at all — see `eval/baselines/` and `research/\
architecture/personalisation-audit.md`'s correction notice for what was checked.

NOT claimed: that ARGUS's Mandate is a complete pre-trade risk gate. Vibe-Trading's leverage, \
total-exposure, daily-trade-count, instrument/asset-class allowlist and funding-ceiling checks \
have no ARGUS equivalent INSIDE `agents.mandate.Mandate` — by design, not oversight: \
portfolio-wide leverage and gross/signed exposure are `agents.desk.ConstitutionPolicy`'s job \
(`max_gross_exposure_notional` / `max_signed_exposure_notional`, `agents/desk.py:753`, gate \
logic in `rule()` at `agents/desk.py:958`), a separate, already-built ARGUS capability this \
module does not re-litigate. Whether splitting \
"whose trade is this" from "how much risk can the book carry" across two modules is better \
architecture than Vibe-Trading's one bundled gate is a real, currently open question — not decided \
here, and not needed to be, since the scope of THIS capability's claim is the personalisation \
dimension alone, stated plainly rather than smuggled in as "wins everything.\"
"""


def main() -> dict[str, Any]:
    """Run the whole comparison and return a serialisable summary.

    Raises:
        MandateComparisonError: the vendored baseline failed to load.
    """
    try:
        baseline = load_baseline()
    except BaselineLoadError as exc:
        raise MandateComparisonError(f"could not load the vendored baseline: {exc}") from exc

    designed = run_designed(baseline)
    sweep_results, sweep_summary = run_sweep(baseline)
    ablations = ablation_cases(baseline)
    costs = blind_spot_costs(sweep_results)

    return {
        "designed": designed.as_dict(),
        "sweep_summary": sweep_summary.as_dict(),
        "ablations": [a.as_dict() for a in ablations],
        "ablation_all_load_bearing": all(a.check_is_load_bearing for a in ablations),
        "blind_spot_costs": costs.as_dict(),
        "scope_statement": SCOPE_STATEMENT,
    }


def render(report: dict[str, Any]) -> str:
    lines = ["MANDATE COMPARISON — ARGUS vs. Vibe-Trading (real, vendored check_mandate)", ""]
    d = report["designed"]
    lines.append(
        f"designed scenarios: {len(d['results'])}, design sound (matched stated intent): "
        f"{d['design_is_sound']}"
        + ("" if d["design_is_sound"] else f" -- MISMATCHES: {d['mismatches']}")
    )
    s = report["sweep_summary"]
    lines.append(
        f"sweep: {s['total']} scenario(s), agree {s['agree_on_allow']}, "
        f"only-ARGUS-refuses {s['only_argus_refuses']}, "
        f"only-Vibe-Trading-refuses {s['only_vibe_trading_refuses']}, "
        f"both-refuse {s['both_refuse']}"
    )
    lines.append(f"ARGUS reason dimensions seen in sweep: {s['argus_reasons_seen']}")
    lines.append(f"Vibe-Trading limit dimensions seen in sweep: {s['vibe_trading_limits_seen']}")
    lines.append(
        f"ablation: {len(report['ablations'])} check(s), all independently load-bearing: "
        f"{report['ablation_all_load_bearing']}"
    )
    c = report["blind_spot_costs"]
    lines.append(
        f"blind spots on the sweep: ARGUS misses "
        f"{c['argus_blind_spot_count']}/{c['total_scenarios']} "
        f"({c['argus_blind_spot_rate']:.1%}) that Vibe-Trading would catch; Vibe-Trading misses "
        f"{c['vibe_trading_blind_spot_count']}/{c['total_scenarios']} "
        f"({c['vibe_trading_blind_spot_rate']:.1%}) that ARGUS would catch"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    from pathlib import Path

    result = main()
    print(render(result))
    out_path = Path(__file__).resolve().parents[3] / "data" / "mandate_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_path}")


__all__ = [
    "SCOPE_STATEMENT",
    "AblationCase",
    "BlindSpotCosts",
    "ComparisonResult",
    "DesignedRun",
    "MandateComparisonError",
    "Scenario",
    "SideVerdict",
    "SweepSummary",
    "ablation_cases",
    "blind_spot_costs",
    "compare",
    "designed_scenarios",
    "main",
    "render",
    "run_argus",
    "run_designed",
    "run_sweep",
    "run_vibe_trading",
    "swept_scenarios",
]
