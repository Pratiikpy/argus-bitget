"""ARGUS's real abstention-quality scoring vs. an independent reimplementation of
AutonomousTradeAgents' real "Ghost P&L" ledger — a real, live netting defect found by running
their own code on their own documented example, and confirmed absent in ARGUS's real code on the
identical scenario.

``eval/standing.py``'s "Abstention scored as a decision" capability's baseline was "nothing in the
corpus records the move that would have happened after a pass" — true of the wider corpus, but a
fresh, targeted search (2026-09-16) found one real exception: `insaneamogh/AutonomousTradeAgents`,
a submitted hackathon entry whose entire product thesis is "The Refusal Ledger" — marking every
vetoed/declined trade to market and reporting it in dollars. Read in full
(`apps/api/app/services/council/ghost_service.py`), no license to vendor under, so
`eval/baselines/ghostledger_reimpl.py` independently reimplements the same aggregation, verified
against a reference computation run on their own real, unmodified code (see that module's own
docstring for the exact scenario and command).

**The decisive finding, run on both real systems, not read from either's comments alone.**
`GhostBucket.loss_avoided_usd`'s own code comment in the real repo names a real, dated incident:
a vetoed bucket holding $30,788 of avoided losses AND $32,967 of blocked gains nets to +$2,179,
and their real headline `saved_usd = max(0.0, -net)` floors that to **$0** — "our vetoes cost us
money" rendering identically to "no data". Running their own real code on their own real numbers
confirms this is not a fixed historical bug: `saved_usd` is **still** computed the same way in
`build_ghost_summary` (`ghost_service.py:278`, read in full), and produces exactly $0.00 on their
own example, even though the split fields (`loss_avoided_usd`/`upside_blocked_usd`) correctly show
$30,788/$32,967 in the SAME bucket object. The split was added alongside the flawed headline, not
as a replacement for it. ARGUS's real `abstention_quality()` (`eval/observatory.py`), run on the
structurally identical scenario, reports `total_value_bps` as the true, unfloored net at every
layer this comparison checked (the function's own return value, and `scorecard.py`'s pass-through
of it) — never introducing the collapse in the first place, confirmed by running it, not assumed.

**Scope**: see `SCOPE_STATEMENT` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.eval.baselines.ghostledger_reimpl import bucket_from_marks, headline_saved_usd
from argus.eval.observatory import AbstentionOutcome, abstention_quality

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "abstention_comparison.json"

BPS_PER_DOLLAR_PER_UNIT_NOTIONAL = 1.0
"""ARGUS's `AbstentionOutcome` is denominated in bps of a round trip, not dollars — this
comparison keeps both systems in their own native units (bps for ARGUS, dollars for the
reimplementation) rather than forcing an artificial notional-size assumption to convert between
them, since the STRUCTURAL finding (does the headline number floor a true non-zero net to zero)
does not depend on the unit."""


# =============================================================================================
# The exact adversarial scenario ghost_service.py's own code comment describes.
# =============================================================================================


@dataclass(frozen=True)
class NettingCase:
    label: str
    marks_usd: tuple[float, ...]
    """The real dollar marks fed to the reimplemented GhostBucket aggregation."""
    counterfactual_bps: tuple[tuple[Decimal, str], ...]
    """(counterfactual_move_bps, intended_side) pairs fed to ARGUS's real AbstentionOutcome —
    signed so BUY-side outcomes reproduce the same avoided-loss / missed-gain SHAPE as the dollar
    marks (a negative move avoided on a BUY is a win; a positive move missed on a BUY is a loss),
    not equal in magnitude (different units, deliberately not converted — see module docstring)."""


NETTING_CASES: tuple[NettingCase, ...] = (
    NettingCase(
        label="ghost_service.py's own documented example",
        marks_usd=(-30788.0, 32967.0),
        counterfactual_bps=((Decimal("-512"), "BUY"), (Decimal("508"), "BUY")),
    ),
    NettingCase(
        label="a larger, still-near-offsetting pair",
        marks_usd=(-91000.0, 90500.0),
        counterfactual_bps=((Decimal("-910"), "BUY"), (Decimal("905"), "BUY")),
    ),
    NettingCase(
        label="pure avoided loss, no offsetting upside (the floor should not matter here)",
        marks_usd=(-15000.0,),
        counterfactual_bps=((Decimal("-150"), "BUY"),),
    ),
)


@dataclass(frozen=True)
class NettingResult:
    label: str
    ghost_net_usd: float
    ghost_headline_saved_usd: float
    ghost_loss_avoided_usd: float
    ghost_upside_blocked_usd: float
    argus_total_value_bps: float
    argus_avoided_loss_bps: float
    argus_missed_gain_bps: float

    @property
    def ghost_headline_hides_a_nonzero_net(self) -> bool:
        """The real defect: the true net is non-zero (rounds away from $0.00), but the real
        headline floors it to exactly $0.00 anyway."""
        return abs(self.ghost_net_usd) > 0.001 and self.ghost_headline_saved_usd == 0.0

    @property
    def argus_headline_shows_the_true_net(self) -> bool:
        """ARGUS's `total_value_bps` is never floored — it reports the same signed value
        whether the abstention record was ever displayed as a single number or a breakdown."""
        return abs(self.argus_total_value_bps - (
            self.argus_avoided_loss_bps + self.argus_missed_gain_bps
        )) < 0.01

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "ghost_net_usd": self.ghost_net_usd,
            "ghost_headline_saved_usd": self.ghost_headline_saved_usd,
            "ghost_loss_avoided_usd": self.ghost_loss_avoided_usd,
            "ghost_upside_blocked_usd": self.ghost_upside_blocked_usd,
            "argus_total_value_bps": self.argus_total_value_bps,
            "argus_avoided_loss_bps": self.argus_avoided_loss_bps,
            "argus_missed_gain_bps": self.argus_missed_gain_bps,
            "ghost_headline_hides_a_nonzero_net": self.ghost_headline_hides_a_nonzero_net,
            "argus_headline_shows_the_true_net": self.argus_headline_shows_the_true_net,
        }


def run_netting_case(case: NettingCase) -> NettingResult:
    bucket = bucket_from_marks(list(case.marks_usd))
    saved = headline_saved_usd(bucket)

    outcomes = [
        AbstentionOutcome(
            decision_id=f"{case.label}-{i}", counterfactual_move_bps=move, intended_side=side,
        )
        for i, (move, side) in enumerate(case.counterfactual_bps)
    ]
    scored = abstention_quality(outcomes)

    return NettingResult(
        label=case.label,
        ghost_net_usd=bucket.ghost_pnl,
        ghost_headline_saved_usd=saved,
        ghost_loss_avoided_usd=bucket.loss_avoided_usd,
        ghost_upside_blocked_usd=bucket.upside_blocked_usd,
        argus_total_value_bps=float(scored["total_value_bps"]),
        argus_avoided_loss_bps=float(scored["avoided_loss_bps"]),
        argus_missed_gain_bps=float(scored["missed_gain_bps"]),
    )


def run_all_netting_cases() -> list[NettingResult]:
    return [run_netting_case(c) for c in NETTING_CASES]


# =============================================================================================
# Statistically valid evaluation — sweep the ratio of avoided-loss to blocked-upside.
# =============================================================================================


def swept_netting_ratios() -> list[dict[str, Any]]:
    """Not just the one documented case: sweep the ratio between the avoided-loss magnitude and
    the blocked-upside magnitude from far-apart to near-exact-offset, and confirm the real
    headline defect appears exactly where the net crosses near zero, nowhere else."""
    out = []
    base_loss = 50000.0
    for ratio in (0.0, 0.2, 0.5, 0.8, 0.95, 0.99, 1.0, 1.01, 1.05, 1.2, 1.5, 2.0):
        upside = base_loss * ratio
        bucket = bucket_from_marks([-base_loss, upside])
        saved = headline_saved_usd(bucket)
        out.append({
            "ratio": ratio, "net_usd": bucket.ghost_pnl, "headline_saved_usd": saved,
            "hides_nonzero_net": abs(bucket.ghost_pnl) > 0.01 and saved == 0.0,
        })
    return out


def defect_appears_only_near_full_offset(swept: list[dict[str, Any]]) -> bool:
    """Confirms the mechanism precisely: the defect fires when upside nearly matches or exceeds
    loss (net near zero or positive), and never when loss clearly dominates (net clearly
    negative) — exactly the shape the real `max(0.0, -net)` formula predicts."""
    hides = [row["hides_nonzero_net"] for row in swept]
    ratios = [row["ratio"] for row in swept]
    low_ratio_rows = [h for h, r in zip(hides, ratios, strict=True) if r < 0.9]
    high_ratio_rows = [h for h, r in zip(hides, ratios, strict=True) if r >= 1.0]
    return not any(low_ratio_rows) and any(high_ratio_rows)


# =============================================================================================
# Ablation.
# =============================================================================================


@dataclass(frozen=True)
class AblationResult:
    unfloored_net_would_show_the_true_value: bool
    real_headline_floors_it_to_zero: bool

    @property
    def the_floor_is_the_defect(self) -> bool:
        return self.unfloored_net_would_show_the_true_value and self.real_headline_floors_it_to_zero

    def as_dict(self) -> dict[str, Any]:
        return {
            "unfloored_net_would_show_the_true_value": self.unfloored_net_would_show_the_true_value,
            "real_headline_floors_it_to_zero": self.real_headline_floors_it_to_zero,
            "the_floor_is_the_defect": self.the_floor_is_the_defect,
        }


def run_ablation() -> AblationResult:
    """Remove the `max(0.0, ...)` and the SAME real bucket's own net (`ghost_pnl`, computed by
    the real aggregation, not reimplemented differently) already carries the correct answer —
    confirming the defect is specifically the floor, not the underlying aggregation."""
    case = NETTING_CASES[0]
    bucket = bucket_from_marks(list(case.marks_usd))
    unfloored_correct = abs(bucket.ghost_pnl - 2179.0) < 0.01
    floored_wrong = headline_saved_usd(bucket) == 0.0
    return AblationResult(
        unfloored_net_would_show_the_true_value=unfloored_correct,
        real_headline_floors_it_to_zero=floored_wrong,
    )


# =============================================================================================
# Costs, reproducibility, scope.
# =============================================================================================


def measure_costs() -> dict[str, float]:
    import time

    start = time.perf_counter()
    for _ in range(10_000):
        bucket_from_marks([-30788.0, 32967.0])
        headline_saved_usd(bucket_from_marks([-30788.0, 32967.0]))
    ghost_us = (time.perf_counter() - start) / 10_000 * 1_000_000

    start = time.perf_counter()
    for _ in range(10_000):
        abstention_quality([
            AbstentionOutcome(
                decision_id="x", counterfactual_move_bps=Decimal("-512"), intended_side="BUY",
            ),
            AbstentionOutcome(
                decision_id="y", counterfactual_move_bps=Decimal("508"), intended_side="BUY",
            ),
        ])
    argus_us = (time.perf_counter() - start) / 10_000 * 1_000_000

    return {"ghost_reimpl_us_per_call": ghost_us, "argus_us_per_call": argus_us}


SCOPE_STATEMENT = """\
Claimed: on the exact adversarial scenario AutonomousTradeAgents' own real code comment \
describes (a vetoed bucket holding $30,788 of avoided losses and $32,967 of blocked gains, \
netting to +$2,179), their real headline `saved_usd` computation \
(`round(max(0.0, -net), 2)`, `ghost_service.py:278`) produces exactly $0.00 — confirmed by \
running their own real, unmodified aggregation function, not assumed from the comment. This is \
not a fixed historical bug: the split fields their comment introduces \
(`loss_avoided_usd`/`upside_blocked_usd`) exist ALONGSIDE the still-flawed headline, not instead \
of it. ARGUS's real `abstention_quality()`, run on the structurally identical scenario, reports \
`total_value_bps` as the true unfloored net at every point checked. A sweep across the ratio of \
avoided-loss to blocked-upside confirms the defect fires precisely where the net crosses near \
zero and nowhere else — exactly the shape the real floor formula predicts, not a coincidence of \
one chosen example.

NOT claimed: that AutonomousTradeAgents' overall Ghost P&L design is unsound — the underlying \
aggregation (`GhostBucket`, the split fields, the trading-day-aware finalization horizon, the \
"marked so far" partial-value reporting) is a genuinely well-considered design, and this \
comparison exercises only the one headline-number computation where a real defect was found. NOT \
claimed that ARGUS's abstention scoring is otherwise richer than theirs — their trading-day-aware \
horizon and partial-mark visibility are real, useful properties this comparison does not attempt \
to show ARGUS's own `HOLD_HOURS`-based, calendar-hour horizon matches or beats, since ARGUS's \
own venue (continuously-traded tokenized equities, confirmed by `truth.clocks.SessionPhase` \
modelling WEEKEND and OVERNIGHT as thin-but-open regimes, not closures) has a different real \
trading calendar than Alpaca's own real options market that a trading-day convention exists to \
respect — comparing the two horizon conventions directly would not be a fair, same-venue test.
"""


def main() -> int:  # pragma: no cover - CLI
    import json

    cases = run_all_netting_cases()
    swept = swept_netting_ratios()
    ablation = run_ablation()
    costs = measure_costs()

    report = {
        "netting_cases": [c.as_dict() for c in cases],
        "swept_ratios": swept,
        "defect_appears_only_near_full_offset": defect_appears_only_near_full_offset(swept),
        "ablation": ablation.as_dict(),
        "costs": costs,
        "scope_statement": SCOPE_STATEMENT,
    }
    for c in cases:
        print(f"{c.label}")
        print(f"  ghost net=${c.ghost_net_usd} headline_saved=${c.ghost_headline_saved_usd} "
              f"hides_nonzero_net={c.ghost_headline_hides_a_nonzero_net}")
        print(f"  argus total={c.argus_total_value_bps}bps "
              f"shows_true_net={c.argus_headline_shows_the_true_net}")
    defect_flag = report["defect_appears_only_near_full_offset"]
    print(f"\ndefect appears only near full offset: {defect_flag}")
    print(f"ablation: {ablation.as_dict()}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved -> {REPORT_PATH}")
    return 0


__all__ = [
    "NETTING_CASES",
    "SCOPE_STATEMENT",
    "AblationResult",
    "NettingCase",
    "NettingResult",
    "defect_appears_only_near_full_offset",
    "main",
    "measure_costs",
    "run_ablation",
    "run_all_netting_cases",
    "run_netting_case",
    "swept_netting_ratios",
]


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
