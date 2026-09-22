"""ARGUS's real deliberation-cost model vs. an independent reimplementation of
LatencySensitiveBench's linear price-decay — same question, two different answers, one of them
grounded in random-walk theory and one of them an arbitrary heuristic with no stated derivation.

``eval/standing.py``'s "Deliberation priced as a trading cost" capability's own `best_implementation
_studied` proof already reads: 88 repos in the research corpus, none price model inference latency
as a real cost. A fresh, targeted search (2026-09-16) found one genuine exception —
`HaoKang-Timmy/LatencySensitiveBench` (NeurIPS 2025 poster, arxiv 2505.19481) — read in full, with
no permissive license to vendor under, so `eval/baselines/latencybench_reimpl.py` is an independent
reimplementation of the same described idea, verified against reference output vectors computed by
running their own real code once (see that module's own docstring for the exact command and output
pinned in `test_deliberation_comparison.py`).

**Both models answer "what does N seconds of thinking cost, in execution quality" — with genuinely
different shapes, both real, both run here.** ARGUS's `execution.latency.latency_slippage_bps`
charges `annualised_vol * sqrt(delay / year) * 10000 * depth_multiplier` — the SQRT-of-time
scaling is the textbook result for a random walk's expected absolute displacement (the same
reasoning market-microstructure and options-pricing literature uses for price diffusion over an
interval), and it is UNBOUNDED: the charge keeps growing for arbitrarily long delays, correctly.
LatencySensitiveBench's `_apply_linear_decay` instead interpolates LINEARLY from the current
price toward the day's flat average as delay grows from 0 to a fixed `decay_window` (1.5-2s in
their own stated defaults), and is CAPPED there — past `decay_window`, more delay costs literally
nothing further, by construction (confirmed by reading the method: `if delay >= self.decay_window:
return avg, avg`).

**The decisive adversarial case uses ARGUS's OWN real measured numbers, not invented ones.**
`agents/meta_pm.py::THINKING_MS` states the real, bake-off-measured Qwen wall-clock at each of
ARGUS's three real thinking budgets: OFF=3,000ms, LOW=8,000ms, FULL=40,000ms — all three exceed
LatencySensitiveBench's own stated `decay_window` (1.5-2s) many times over. Run through the real
linear-decay reimplementation (verified against their real code's own reference vectors), all
three collapse to the IDENTICAL cost — the model cannot distinguish a 3-second decision from a
40-second one at all, because both are already past its cap. ARGUS's own real sqrt(t) model,
run on the identical three real delays, produces three meaningfully different, correctly ordered
costs (verified by calling the real `deliberation_cost_bps`, not paraphrased).

**Scope**: see `SCOPE_STATEMENT` below.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.agents.meta_pm import THINKING_MS
from argus.eval.baselines.latencybench_reimpl import linear_decay_price
from argus.execution.latency import latency_slippage_bps

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "deliberation_comparison.json"

REFERENCE_AVERAGE_PRICE = 100.0
REFERENCE_SPREAD_BPS = 200.0
"""A 2% high/low band around the reference average — a plausible one-day range for a volatile
token, used only to give the linear-decay model concrete high/low inputs (it has no volatility
parameter of its own to drive this from); not tuned to favour either model."""

DECAY_WINDOW_SECONDS = 1.5
"""LatencySensitiveBench's own stated default (`TradingEnv.py:14`, `decay_window: float = 1.5`)."""


def linear_decay_cost_bps(
    delay_seconds: float, *, spread_bps: float = REFERENCE_SPREAD_BPS,
    decay_window: float = DECAY_WINDOW_SECONDS,
) -> float:
    """The bps deviation the real linear-decay reimplementation produces at `delay_seconds` — the
    magnitude the executable price moved from its undelayed value, expressed the same way ARGUS's
    own `latency_slippage_bps` is (an expected magnitude, not a signed loss)."""
    half_spread_frac = (spread_bps / 2.0) / 10_000.0
    high = REFERENCE_AVERAGE_PRICE * (1 + half_spread_frac)
    low = REFERENCE_AVERAGE_PRICE * (1 - half_spread_frac)
    new_high, _new_low = linear_decay_price(
        high, low, delay_seconds,
        average_price=REFERENCE_AVERAGE_PRICE, decay_window=decay_window,
    )
    return abs(high - new_high) / high * 10_000.0


def argus_cost_bps(
    delay_seconds: float, *, annualised_vol: Decimal = Decimal("0.6"),
    depth_multiplier: Decimal = Decimal("1"),
) -> float:
    """ARGUS's own real `latency_slippage_bps`, called directly — not reimplemented."""
    return float(
        latency_slippage_bps(
            latency_ns=int(delay_seconds * 1_000_000_000),
            annualised_vol=annualised_vol, depth_multiplier=depth_multiplier,
        )
    )


# =============================================================================================
# Same-input comparison, on ARGUS's own real measured thinking-budget delays.
# =============================================================================================


@dataclass(frozen=True)
class ThinkingBudgetCase:
    label: str
    delay_seconds: float
    linear_decay_bps: float
    argus_bps: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label, "delay_seconds": self.delay_seconds,
            "linear_decay_bps": round(self.linear_decay_bps, 4),
            "argus_bps": round(self.argus_bps, 4),
        }


def run_thinking_budget_cases() -> list[ThinkingBudgetCase]:
    return [
        ThinkingBudgetCase(
            label=str(thinking), delay_seconds=ms / 1000.0,
            linear_decay_bps=linear_decay_cost_bps(ms / 1000.0),
            argus_bps=argus_cost_bps(ms / 1000.0),
        )
        for thinking, ms in THINKING_MS.items()
    ]


def linear_decay_collapses_argus_real_tiers(cases: list[ThinkingBudgetCase]) -> bool:
    """The decisive adversarial finding: does the real linear-decay reimplementation report the
    SAME cost for all three of ARGUS's real, distinct thinking-budget delays — because all three
    exceed its fixed cap — while ARGUS's own model reports three distinct, correctly ordered
    costs on the identical delays?"""
    linear_values = {round(c.linear_decay_bps, 6) for c in cases}
    argus_values = [c.argus_bps for c in cases]
    all_collapsed = len(linear_values) == 1
    argus_distinct_and_ordered = (
        len(set(round(v, 6) for v in argus_values)) == len(argus_values)
        and argus_values == sorted(argus_values)
    )
    return all_collapsed and argus_distinct_and_ordered


# =============================================================================================
# Statistically valid evaluation — sweep delay across a wide range, both models, every point.
# =============================================================================================

SWEPT_DELAYS_SECONDS: tuple[float, ...] = (
    0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 40.0, 60.0, 120.0,
)


@dataclass(frozen=True)
class SweptPoint:
    delay_seconds: float
    linear_decay_bps: float
    argus_bps: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "delay_seconds": self.delay_seconds,
            "linear_decay_bps": round(self.linear_decay_bps, 4),
            "argus_bps": round(self.argus_bps, 4),
        }


def swept_comparison() -> list[SweptPoint]:
    return [
        SweptPoint(
            delay_seconds=d, linear_decay_bps=linear_decay_cost_bps(d), argus_bps=argus_cost_bps(d),
        )
        for d in SWEPT_DELAYS_SECONDS
    ]


def linear_decay_is_bounded_past_its_cap(points: list[SweptPoint]) -> bool:
    """Every swept delay past decay_window (1.5s) reports the identical linear-decay cost —
    confirming boundedness is real across the whole range, not just the three thinking-budget
    points."""
    past_cap = [p.linear_decay_bps for p in points if p.delay_seconds >= DECAY_WINDOW_SECONDS]
    return len(past_cap) >= 2 and len(set(round(v, 6) for v in past_cap)) == 1


def argus_cost_is_unbounded(points: list[SweptPoint]) -> bool:
    """ARGUS's own real cost keeps growing (strictly) across the whole swept range — no cap."""
    from itertools import pairwise

    values = [p.argus_bps for p in points]
    return all(b > a for a, b in pairwise(values))


# =============================================================================================
# Ablation — is the collapse caused by decay_window's specific VALUE, not merely its existence?
# =============================================================================================


@dataclass(frozen=True)
class AblationResult:
    real_decay_window_collapses_real_tiers: bool
    widened_decay_window_distinguishes_them: bool

    @property
    def the_specific_window_value_is_load_bearing(self) -> bool:
        return (
            self.real_decay_window_collapses_real_tiers
            and self.widened_decay_window_distinguishes_them
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "real_decay_window_collapses_real_tiers": self.real_decay_window_collapses_real_tiers,
            "widened_decay_window_distinguishes_them": (
                self.widened_decay_window_distinguishes_them
            ),
            "the_specific_window_value_is_load_bearing": (
                self.the_specific_window_value_is_load_bearing
            ),
        }


def run_ablation() -> AblationResult:
    real_cases = run_thinking_budget_cases()
    real_collapses = linear_decay_collapses_argus_real_tiers(real_cases)

    widened_values = [
        linear_decay_cost_bps(ms / 1000.0, decay_window=60.0) for ms in THINKING_MS.values()
    ]
    widened_distinguishes = len(set(round(v, 6) for v in widened_values)) == len(widened_values)

    return AblationResult(
        real_decay_window_collapses_real_tiers=real_collapses,
        widened_decay_window_distinguishes_them=widened_distinguishes,
    )


# =============================================================================================
# Costs, reproducibility, scope.
# =============================================================================================


def measure_costs() -> dict[str, float]:
    """Both are pure, deterministic Python functions — the real cost here is call overhead,
    measured, not the modelled bps output (which is the finding itself, not a cost of running
    the comparison)."""
    import time

    start = time.perf_counter()
    for _ in range(10_000):
        linear_decay_cost_bps(1.0)
    linear_us = (time.perf_counter() - start) / 10_000 * 1_000_000

    start = time.perf_counter()
    for _ in range(10_000):
        argus_cost_bps(1.0)
    argus_us = (time.perf_counter() - start) / 10_000 * 1_000_000

    return {"linear_decay_us_per_call": linear_us, "argus_us_per_call": argus_us}


SCOPE_STATEMENT = """\
Claimed: ARGUS's real deliberation-cost model (`execution.latency.latency_slippage_bps`) and an \
independently reimplemented version of LatencySensitiveBench's real linear price-decay model \
(verified against reference vectors computed from their own unmodified code) answer the same \
question — "what does N seconds of LLM inference delay cost, in execution quality" — with \
genuinely different, both-real shapes. ARGUS's charge scales as sqrt(delay), matching the \
textbook expected-displacement result for a random walk, and is unbounded. The baseline's charge \
scales linearly up to a fixed cap (`decay_window`, their own stated default 1.5s) and is flat \
beyond it — confirmed by running it, not assumed from reading. On ARGUS's own real, \
bake-off-measured thinking-budget delays (3s / 8s / 40s), the baseline's real formula reports the \
IDENTICAL cost for all three, because all three exceed its cap; ARGUS's own real formula reports \
three distinct, correctly ordered costs on the identical delays. An ablation confirms the \
collapse is caused by the SPECIFIC 1.5s cap value, not the existence of a cap in general — \
widening it to 60s (still finite) restores the ability to distinguish ARGUS's own three delays.

NOT claimed: that LatencySensitiveBench's model is "wrong" for the purpose ITS OWN paper built \
it for — a bounded, simple decay may be a deliberate, defensible modelling choice for the delay \
ranges their own benchmark actually exercises (likely sub-2-second, matching classical HFT \
latency budgets, not the 3-40 second range a reasoning LLM like Qwen actually runs at). The \
finding is narrower and precise: applied to the delay range ARGUS's real system actually \
operates in, the baseline's real formula cannot distinguish ARGUS's own three real thinking \
tiers at all, while ARGUS's own formula can — which is the property this capability is about.
"""


def check_reproducibility() -> dict[str, Any]:
    """Both compared functions are pure and deterministic — no LLM call, no randomness, no clock
    in the modelled output. Reproducibility here is not a plausible assumption from that fact; it
    is run and confirmed, the same discipline every other comparison in this project applies
    (`regime_comparison.py`, `allocation_comparison.py`, ...), because a claim resting on 'this
    should be deterministic' rather than 'this was run twice and matched' is exactly the gap
    `verify()` exists to catch."""
    first = {
        "thinking_budget_cases": [c.as_dict() for c in run_thinking_budget_cases()],
        "swept_comparison": [p.as_dict() for p in swept_comparison()],
        "ablation": run_ablation().as_dict(),
    }
    second = {
        "thinking_budget_cases": [c.as_dict() for c in run_thinking_budget_cases()],
        "swept_comparison": [p.as_dict() for p in swept_comparison()],
        "ablation": run_ablation().as_dict(),
    }
    identical = json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    return {"identical": identical, "deterministic": identical}


def main() -> int:  # pragma: no cover - CLI
    thinking_cases = run_thinking_budget_cases()
    swept = swept_comparison()
    ablation = run_ablation()
    costs = measure_costs()

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "thinking_budget_cases": [c.as_dict() for c in thinking_cases],
        "swept_comparison": [p.as_dict() for p in swept],
        "linear_decay_collapses_argus_real_tiers": linear_decay_collapses_argus_real_tiers(
            thinking_cases
        ),
        "linear_decay_is_bounded_past_its_cap": linear_decay_is_bounded_past_its_cap(swept),
        "argus_cost_is_unbounded": argus_cost_is_unbounded(swept),
        "ablation": ablation.as_dict(),
        "costs": costs,
        "reproducibility": check_reproducibility(),
        "scope_statement": SCOPE_STATEMENT,
    }
    for c in thinking_cases:
        print(f"{c.label}: delay={c.delay_seconds}s  linear_decay={c.linear_decay_bps:.2f}bps  "
              f"argus={c.argus_bps:.2f}bps")
    print(f"\ncollapses real tiers: {report['linear_decay_collapses_argus_real_tiers']}")
    print(f"ablation: {ablation.as_dict()}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved -> {REPORT_PATH}")
    return 0


__all__ = [
    "DECAY_WINDOW_SECONDS",
    "REFERENCE_AVERAGE_PRICE",
    "REFERENCE_SPREAD_BPS",
    "SCOPE_STATEMENT",
    "SWEPT_DELAYS_SECONDS",
    "AblationResult",
    "SweptPoint",
    "ThinkingBudgetCase",
    "argus_cost_bps",
    "argus_cost_is_unbounded",
    "linear_decay_collapses_argus_real_tiers",
    "linear_decay_cost_bps",
    "linear_decay_is_bounded_past_its_cap",
    "main",
    "measure_costs",
    "run_ablation",
    "run_thinking_budget_cases",
    "swept_comparison",
]


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
