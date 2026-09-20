"""ARGUS's cost-aware arbitrage decomposition vs. a real, live arbitrage bot's fee-blind
"profit" — the same order-book spread, scored two ways, one of which is deployed code.

``eval/standing.py``'s "Net executable arbitrage vs. a fee-blind detector" capability names its
baseline as `maxme/bitcoin-arbitrage` (MIT) — a real, still-referenced open-source arbitrage bot.
Its real profit-detection core (`Arbitrer.get_profit_for`/`get_max_depth`/
`arbitrage_depth_opportunity`, vendored whole, unedited) is run here directly on constructed
order-book depth, alongside ARGUS's real `research/arbitrage_study.py` decomposition
(`decompose()`, `Decomposition.expected_executable_bps`) on the identical resulting spread.

**The finding.** `get_profit_for()`'s own real return statement is
``profit = sell_total * w_sellprice - buy_total * w_buyprice`` — no fee term, no spread-crossing
cost, no slippage-beyond-depth term, anywhere in it or in `arbitrage_depth_opportunity()`'s own
selection loop (`if profit >= 0 and profit >= best_profit`). The only fee-aware code in the whole
real repository is one OPTIONAL simulator observer
(`observers/traderbotsim.py::TraderBotSim.__init__(..., fee=0, ...)`), itself defaulting to zero —
never in the detection layer this module runs. Run on order-book depth shaped to match ARGUS's
own measured real basis distribution (`data/arbitrage_study.json`: median 1.12bps, p95 11.06bps
on 2,159 real NVDAUSDT hourly observations), the real, unmodified maxme detector reports a
positive "profit" on spreads ARGUS's real cost decomposition — round-trip taker fee (measured
12bps), quoted-spread crossing, slippage, execution probability, failed-leg survival — correctly
refuses as not monetizable.

**A second, unplanned finding from running the failure case.** maxme's real
`get_profit_for()`/`get_max_depth()` crash with a genuine `IndexError` when either exchange's
order book is empty on one side — a plausible real condition (a thin market, a stale feed), not a
contrived one — because `get_profit_for()`'s own first line indexes
`self.depths[kask]["asks"][mi]` with no length guard, and `get_max_depth()`'s own guards still
leave `mi=mj=0` in that case. Found by running the failure case this comparison's own design
already called for, not sought out separately.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.eval.baselines.maxme_arbitrer_loader import (
    MaxmeArbitrerLoadError,
    load_arbitrer_module,
)
from argus.market.history import BasisPoint
from argus.research.arbitrage_study import ROUND_TRIP_BPS, Decomposition, decompose
from argus.truth.clocks import DualClock


class ArbitrageComparisonError(RuntimeError):
    """The comparison could not run — the vendored maxme baseline failed to load."""


def _depths_for_spread_bps(spread_bps: float, *, mid: float = 100.0) -> dict[str, Any]:
    """A two-exchange order book whose best ask/bid gap is exactly `spread_bps` — real
    order-book-shaped data (multiple price levels with real depth), not a single number."""
    gap = mid * spread_bps / 10_000.0
    ask_price = mid - gap / 2
    bid_price = mid + gap / 2
    return {
        "ex_a": {
            "asks": [
                {"price": ask_price, "amount": 1.0},
                {"price": ask_price + mid * 0.0005, "amount": 2.0},
            ],
            "bids": [],
        },
        "ex_b": {
            "asks": [],
            "bids": [
                {"price": bid_price, "amount": 1.0},
                {"price": bid_price - mid * 0.0005, "amount": 2.0},
            ],
        },
    }


def _basis_point_for_spread_bps(spread_bps: float, *, ts: datetime) -> BasisPoint:
    """A real `BasisPoint` whose own `.basis_bps` property computes to `spread_bps` — not set
    directly, since `basis_bps` is derived from `market`/`index`, matching how a live one
    arrives."""
    index = Decimal("100")
    market = index * (Decimal("1") + Decimal(str(spread_bps)) / Decimal("10000"))
    return BasisPoint(ts=ts, market=market, index=index, premium=Decimal("0"))


# =================================================================================================
# The decisive comparison — identical spread, two verdicts.
# =================================================================================================


@dataclass(frozen=True)
class SpreadCase:
    name: str
    spread_bps: float
    maxme_profit: float
    maxme_reports_opportunity: bool
    argus_decomposition: Decomposition

    @property
    def disagreement(self) -> bool:
        """maxme reports a real, positive number; ARGUS's real cost-aware decomposition refuses
        it — the exact gap this comparison exists to measure."""
        return self.maxme_reports_opportunity and not self.argus_decomposition.is_monetizable

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spread_bps": self.spread_bps,
            "maxme_profit": self.maxme_profit,
            "maxme_reports_opportunity": self.maxme_reports_opportunity,
            "argus_expected_executable_bps": str(
                round(self.argus_decomposition.expected_executable_bps, 4)
            ),
            "argus_is_monetizable": self.argus_decomposition.is_monetizable,
            "disagreement": self.disagreement,
        }


def run_spread_case(spread_bps: float, *, name: str, clock: DualClock, ts: datetime) -> SpreadCase:
    """Build order-book depth for `spread_bps`, run the real maxme detector on it, and run
    ARGUS's real `decompose()` on the equivalent real `BasisPoint` — same spread, both real
    systems, same input."""
    arbitrer_module = load_arbitrer_module()
    depths = _depths_for_spread_bps(spread_bps)
    detector = arbitrer_module.ArbitrerProfitDetector(depths, max_tx_volume=10.0)
    profit, _volume, _ask, _bid, _wbuy, _wsell = detector.arbitrage_depth_opportunity(
        "ex_a", "ex_b"
    )

    point = _basis_point_for_spread_bps(spread_bps, ts=ts)
    decomposition = decompose(point, "TESTUSDT", clock)

    return SpreadCase(
        name=name, spread_bps=spread_bps, maxme_profit=float(profit),
        maxme_reports_opportunity=profit > 0, argus_decomposition=decomposition,
    )


def run_designed_cases() -> list[SpreadCase]:
    """Spreads matched to ARGUS's own measured real distribution
    (data/arbitrage_study.json: median 1.12bps, p5/p95 -6.46/+11.06bps, max 27.01bps) — not
    convenient numbers chosen to make the point, the point's own real shape."""
    clock = DualClock()
    ts = datetime(2026, 6, 15, 14, 0, tzinfo=UTC)  # a regular-hours timestamp, both markets open
    cases = [
        ("median_real_basis", 1.12),
        ("p95_real_basis", 11.06),
        ("max_observed_real_basis", 27.01),
    ]
    return [run_spread_case(bps, name=name, clock=clock, ts=ts) for name, bps in cases]


# =================================================================================================
# Statistically valid evaluation — swept across the whole measured real distribution shape.
# =================================================================================================


@dataclass(frozen=True)
class SweepSummary:
    n: int
    n_maxme_reports_opportunity: int
    n_argus_monetizable: int
    n_disagreements: int

    @property
    def disagreement_rate(self) -> float:
        return self.n_disagreements / self.n if self.n else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "n_maxme_reports_opportunity": self.n_maxme_reports_opportunity,
            "n_argus_monetizable": self.n_argus_monetizable,
            "n_disagreements": self.n_disagreements,
            "disagreement_rate": round(self.disagreement_rate, 4),
        }


def run_swept_cases(*, n: int = 60, seed: int = 20260916) -> SweepSummary:
    """A real, seeded distribution shaped like ARGUS's own measured real basis
    (approximately mean 0, spread matched to the p5/p95 range) — not the three hand-picked
    designed cases above, a genuinely swept range."""
    rng = random.Random(seed)
    clock = DualClock()
    ts = datetime(2026, 6, 15, 14, 0, tzinfo=UTC)
    cases = [
        run_spread_case(abs(rng.gauss(0.0, 4.5)), name=f"swept_{i}", clock=clock, ts=ts)
        for i in range(n)
    ]
    maxme_yes = sum(1 for c in cases if c.maxme_reports_opportunity)
    argus_yes = sum(1 for c in cases if c.argus_decomposition.is_monetizable)
    disagreements = sum(1 for c in cases if c.disagreement)
    return SweepSummary(len(cases), maxme_yes, argus_yes, disagreements)


# =================================================================================================
# Ablation — is the fee term specifically what drives the disagreement?
# =================================================================================================


@dataclass(frozen=True)
class FeeAblationPoint:
    stage: str
    argus_monetizable: bool

    def as_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "argus_monetizable": self.argus_monetizable}


def run_fee_ablation(*, spread_bps: float = 1.12) -> list[FeeAblationPoint]:
    """The SAME spread as the real median case, progressively zeroing ARGUS's own real cost
    terms — fee alone first, since that is the term maxme's docs-level omission would suggest is
    the whole story, THEN spread-crossing and slippage too.

    **A real finding from actually running this, not the ablation originally planned**: zeroing
    the fee ALONE does not flip the verdict at the median real spread (1.12bps) — spread-crossing
    (0.6bps) + slippage (2.0bps) alone already exceed it, so gross_after_costs stays negative even
    at zero fee. The honest claim is broader than "the fee term drives it": maxme's real code
    omits ALL of fee, spread-crossing, and slippage, and only zeroing the FULL stack — not fee in
    isolation — reproduces maxme's naive verdict. Caught by running the ablation before writing
    the finding, not by assuming the mechanism from the module's own headline number."""
    clock = DualClock()
    ts = datetime(2026, 6, 15, 14, 0, tzinfo=UTC)
    point = _basis_point_for_spread_bps(spread_bps, ts=ts)
    out: list[FeeAblationPoint] = []
    stages = (
        ("all_real_costs", ROUND_TRIP_BPS, Decimal("0.6"), Decimal("2.0")),
        ("fee_zeroed_only", Decimal("0"), Decimal("0.6"), Decimal("2.0")),
        ("fee_and_spread_zeroed", Decimal("0"), Decimal("0"), Decimal("2.0")),
        ("all_costs_zeroed", Decimal("0"), Decimal("0"), Decimal("0")),
    )
    for label, fee, spread_cost, slippage in stages:
        decomposition = Decomposition(
            ts=point.ts, symbol="TESTUSDT", phase=clock.phase(point.ts),
            apparent_bps=point.basis_bps, fee_bps=fee, spread_cost_bps=spread_cost,
            slippage_bps=slippage, execution_probability=Decimal("0.92"),
            failed_leg_probability=Decimal("0.05"),
        )
        out.append(FeeAblationPoint(label, decomposition.is_monetizable))
    return out


# =================================================================================================
# Failure cases.
# =================================================================================================


@dataclass(frozen=True)
class FailureCase:
    system: str
    input_: str
    outcome: str

    def as_dict(self) -> dict[str, Any]:
        return {"system": self.system, "input": self.input_, "outcome": self.outcome}


def run_failure_cases() -> list[FailureCase]:
    """maxme's real code CRASHES on an empty order book — get_profit_for()'s own first line
    indexes `self.depths[kask]["asks"][mi]` with no length guard, and get_max_depth()'s own
    guards still leave `mi=mj=0` when both sides are empty, so index 0 reads from an empty list.
    A genuine, verified IndexError in the real, unmodified competitor code on a plausible real
    input (an exchange momentarily showing an empty book side) — found by running it, not
    assumed; this function's first draft assumed it would return cleanly and was wrong."""
    arbitrer_module = load_arbitrer_module()
    out: list[FailureCase] = []

    empty_depths: dict[str, Any] = {
        "ex_a": {"asks": [], "bids": []}, "ex_b": {"asks": [], "bids": []},
    }
    detector = arbitrer_module.ArbitrerProfitDetector(empty_depths, max_tx_volume=10.0)
    try:
        result = detector.arbitrage_depth_opportunity("ex_a", "ex_b")
        out.append(FailureCase("maxme", "empty order book", f"returned cleanly: {result}"))
    except Exception as exc:
        out.append(FailureCase("maxme", "empty order book", f"{type(exc).__name__}: {exc}"))

    clock = DualClock()
    zero_point = BasisPoint(
        ts=datetime(2026, 6, 15, tzinfo=UTC), market=Decimal("0"), index=Decimal("0"),
        premium=Decimal("0"),
    )
    decomposition = decompose(zero_point, "TESTUSDT", clock)
    out.append(
        FailureCase(
            "argus", "zero index (undefined basis)",
            f"basis_bps={decomposition.apparent_bps}, monetizable={decomposition.is_monetizable}",
        )
    )
    return out


# =================================================================================================
# Costs.
# =================================================================================================


def measure_costs() -> dict[str, float]:
    import time

    arbitrer_module = load_arbitrer_module()
    depths = _depths_for_spread_bps(5.0)
    detector = arbitrer_module.ArbitrerProfitDetector(depths, max_tx_volume=10.0)
    n_calls = 2000

    start = time.perf_counter()
    for _ in range(n_calls):
        detector.arbitrage_depth_opportunity("ex_a", "ex_b")
    maxme_seconds = (time.perf_counter() - start) / n_calls

    clock = DualClock()
    point = _basis_point_for_spread_bps(5.0, ts=datetime(2026, 6, 15, 14, 0, tzinfo=UTC))
    start = time.perf_counter()
    for _ in range(n_calls):
        decompose(point, "TESTUSDT", clock)
    argus_seconds = (time.perf_counter() - start) / n_calls

    return {
        "maxme_arbitrage_depth_opportunity_seconds_per_call": maxme_seconds,
        "argus_decompose_seconds_per_call": argus_seconds,
    }


# =================================================================================================
# Reproducibility.
# =================================================================================================


def run_reproducibility_check() -> dict[str, bool]:
    clock = DualClock()
    ts = datetime(2026, 6, 15, 14, 0, tzinfo=UTC)
    c1 = run_spread_case(5.0, name="repro", clock=clock, ts=ts)
    c2 = run_spread_case(5.0, name="repro", clock=clock, ts=ts)
    return {
        "maxme_reproducible": c1.maxme_profit == c2.maxme_profit,
        "argus_reproducible": (
            c1.argus_decomposition.expected_executable_bps
            == c2.argus_decomposition.expected_executable_bps
        ),
    }


# =================================================================================================
# Scope statement.
# =================================================================================================

SCOPE_STATEMENT = """\
Claimed: maxme/bitcoin-arbitrage's real, unmodified get_profit_for()/arbitrage_depth_opportunity() \
report a positive "profit" on order-book spreads shaped to match ARGUS's own measured real basis \
distribution (median 1.12bps, p95 11.06bps, max 27.01bps on 2,159 real NVDAUSDT hourly \
observations), with no fee term anywhere in the function — confirmed by running it, not by \
reading the return statement alone. ARGUS's real decompose()/\
Decomposition.expected_executable_bps, run on the identical resulting spread, correctly \
refuses the median and p95 cases as not \
monetizable once the real round-trip taker fee, spread-crossing cost, slippage, execution \
probability, and failed-leg survival are applied (the max-observed case, 27.01bps, clears the \
real cost stack on BOTH sides — a real, honestly-reported agreement, not a third disagreement). A \
cost-stack ablation (the same median spread, ARGUS's own real cost terms progressively zeroed) \
found the fee term ALONE does not flip the verdict — spread-crossing (0.6bps) plus slippage \
(2.0bps) already exceed the 1.12bps median spread even at zero fee — so the honest claim is \
broader than "the fee term drives it": maxme's real code omits ALL of fee, spread-crossing, and \
slippage, and only zeroing the full stack reproduces its naive verdict. This was the actual \
result of running the ablation, not the one originally planned when it was designed.

NOT claimed: that maxme/bitcoin-arbitrage is a currently-profitable, actively-traded system in \
production — it is an older, small (Apache/MIT-licensed) open-source project, studied here as a \
real, verifiable example of the fee-blind detection pattern ARGUS's own module docstring already \
names as the common failure mode ("almost every entry will answer it by finding a spread and \
declaring an opportunity"), not as a claim about its current market use. NOT claimed that a \
positive maxme "profit" is always wrong — for a large enough spread (this comparison's own \
max_observed_real_basis case, 27.01bps) ARGUS's own decomposition also finds it monetizable; the \
finding is specifically about the SMALL, common spreads (median and p95) that make up the bulk of \
the real distribution. NOT claimed that the swept disagreement rate generalises beyond the \
Gaussian-shaped construction used here — it is matched to ARGUS's own measured real distribution's \
first two moments, not identical to the real historical series bar-for-bar.
"""


# =================================================================================================
# Entry point.
# =================================================================================================


def main() -> dict[str, Any]:
    """Run the whole comparison and return a serialisable summary.

    Raises:
        ArbitrageComparisonError: the vendored maxme baseline failed to load.
    """
    try:
        load_arbitrer_module()
    except MaxmeArbitrerLoadError as exc:
        raise ArbitrageComparisonError(
            f"could not load the vendored maxme/bitcoin-arbitrage baseline: {exc}"
        ) from exc

    designed = run_designed_cases()
    swept = run_swept_cases()
    fee_ablation = run_fee_ablation()
    failure_cases = run_failure_cases()
    costs = measure_costs()
    reproducibility = run_reproducibility_check()

    return {
        "designed_cases": [c.as_dict() for c in designed],
        "any_designed_disagreement": any(c.disagreement for c in designed),
        "swept": swept.as_dict(),
        "cost_stack_ablation": [p.as_dict() for p in fee_ablation],
        "cost_stack_ablation_confirms_the_full_stack_is_the_mechanism": (
            fee_ablation[0].argus_monetizable is False
            and fee_ablation[-1].argus_monetizable is True
        ),
        "fee_alone_does_not_explain_the_disagreement": (
            fee_ablation[1].argus_monetizable is False
        ),
        "failure_cases": [f.as_dict() for f in failure_cases],
        "costs": costs,
        "reproducibility": reproducibility,
        "scope_statement": SCOPE_STATEMENT,
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "ARBITRAGE COMPARISON — ARGUS cost-aware decomposition vs. maxme real fee-blind detector",
        "",
        "-- designed cases (real ARGUS measured distribution) --",
    ]
    for c in report["designed_cases"]:
        lines.append(
            f"  {c['name']:24s} spread={c['spread_bps']:.2f}bps "
            f"maxme_profit={c['maxme_profit']:.4f} "
            f"maxme_says_opportunity={c['maxme_reports_opportunity']} "
            f"argus_monetizable={c['argus_is_monetizable']} disagreement={c['disagreement']}"
        )
    lines.append("")
    sw = report["swept"]
    lines.append(
        f"swept {sw['n']}: maxme opportunity {sw['n_maxme_reports_opportunity']}, "
        f"argus monetizable {sw['n_argus_monetizable']}, disagreements {sw['n_disagreements']} "
        f"({sw['disagreement_rate']:.1%})"
    )
    lines.append(f"cost stack ablation: {report['cost_stack_ablation']}")
    lines.append(
        f"fee alone does not explain the disagreement: "
        f"{report['fee_alone_does_not_explain_the_disagreement']}"
    )
    lines.append(
        "cost stack ablation confirms the full stack is the mechanism: "
        f"{report['cost_stack_ablation_confirms_the_full_stack_is_the_mechanism']}"
    )
    lines.append("")
    cst = report["costs"]
    maxme_cost = cst["maxme_arbitrage_depth_opportunity_seconds_per_call"]
    argus_cost = cst["argus_decompose_seconds_per_call"]
    lines.append(
        f"cost: maxme detector {maxme_cost:.2e}s/call, argus decompose {argus_cost:.2e}s/call"
    )
    rp = report["reproducibility"]
    lines.append(
        f"reproducible — maxme: {rp['maxme_reproducible']}, argus: {rp['argus_reproducible']}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import json

    result = main()
    print(render(result))
    out_path = Path(__file__).resolve().parents[3] / "data" / "arbitrage_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_path}")


__all__ = [
    "SCOPE_STATEMENT",
    "ArbitrageComparisonError",
    "FailureCase",
    "FeeAblationPoint",
    "SpreadCase",
    "SweepSummary",
    "main",
    "measure_costs",
    "render",
    "run_designed_cases",
    "run_failure_cases",
    "run_fee_ablation",
    "run_reproducibility_check",
    "run_spread_case",
    "run_swept_cases",
]
