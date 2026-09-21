"""What the risk layer is worth, measured by building the agent that makes it fire.

**This module exists because the Constitution has never intervened.** It is proved sound over
2,177,280 states — it can only ever reduce a proposed position, never create, enlarge or reverse
one — and across 365 live risk records it shows **zero interventions**, because the desk abstained
on every decision. So the layer's *correctness* is established and its *value* is not, and "our risk
layer is sound" with nothing behind it is precisely the unevidenced claim this project refuses to
make about anything else.

Waiting for the market to supply a reckless decision is not a plan. The answer, taken from
`narutopyy/agent-arena`'s `scripts/firewall_value.py` (MIT), is to stop waiting and **construct the
adversary**: an agent that demands an oversized position on every bar, run twice over the *same*
real price path — once governed by the real Constitution, once by one whose limits are effectively
infinite — with the difference in final equity reported as the guard's value in dollars.

**What is taken from that reference and what is not.** The design is theirs: rogue agent, paired
run, ablated limits, dollar delta. Three things are added here because the original could not
answer an adversarial reading of its own number:

* **The path is real, not synthetic.** Equity is walked over actual Bitget hourly closes for a real
  rToken, so the number is not a property of a generated series somebody chose.
* **Several rogues, not one.** A single greedy long measures the position cap and nothing else.
  Each rogue here is built to attack a *different* gate — size, direction-flipping, confidence,
  hedge-dependence — so the report attributes the saving to the rule that produced it rather than
  to "the firewall" as an undifferentiated object.
* **The worst window, and the best one.** A guard that only helps when prices fall is a guard that
  costs money the rest of the time. Every rogue is run over the worst *and* the best stretch of the
  same history, and the favourable number is published beside the favourable one. A risk layer's
  honest description is "it cost this much in the good case and saved this much in the bad".

**What this does not claim.** It is a measurement of the *rule set* against a *constructed*
adversary, not evidence about the desk's own model, which has never proposed exposure. It says what
the Constitution would have prevented had something reckless reached it. It does not say the desk
would ever have been reckless.

    python -m argus.eval.rogue
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.agents.desk import ConstitutionPolicy
from argus.decision.verdicts import Intent, Side, Verdict
from argus.eval.artefact import write
from argus.risk.hedgeability import HedgeabilitySurface, HedgeCandidate
from argus.truth.clocks import SessionPhase, SessionState

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "rogue_value.json"

STARTING_EQUITY = Decimal("10000")
"""The account the rogue is let loose on. A round number; every result scales with it."""

UNGOVERNED = Decimal("1e12")
"""The ablated limit. Large enough that no gate binds, small enough to stay exact in Decimal.

The ablation is of the *limits*, not of the code path: both arms run the same
:meth:`ConstitutionPolicy.rule`, so a difference between them cannot come from one arm skipping
logic the other ran. That matters — an ablation that removes the module rather than its thresholds
measures the module's existence, not its settings."""

WINDOW_BARS = 168
"""One week of hourly closes. Long enough for a drawdown to develop, short enough that several
disjoint windows fit inside a 90-day history."""


class RogueError(RuntimeError):
    """The measurement cannot be made honestly. Raised rather than reported as a zero."""


@dataclass(frozen=True, slots=True)
class Rogue:
    """An adversary built to make one specific gate fire.

    Each carries the gate it targets so the report can attribute a saving to a rule rather than to
    the risk layer as a whole. A harness that only proves "something stopped it" cannot tell a
    working cap from a working confidence floor.
    """

    name: str
    targets: str
    quantity: Decimal
    confidence: float
    flips_side: bool = False
    """Alternate BUY/SELL every bar. Attacks nothing by itself — it exists to confirm the layer
    never *reverses* a side, and to make the equity path churn fees the way a real rogue would."""

    description: str = ""


ROGUES: tuple[Rogue, ...] = (
    Rogue(
        name="size_maximiser",
        targets="max_position / gross_exposure",
        quantity=Decimal("8000"),
        confidence=0.99,
        description="demands an 8,000-unit position every bar — 8x the account at $10 a unit",
    ),
    Rogue(
        name="confident_fool",
        targets="min_confidence",
        quantity=Decimal("400"),
        confidence=0.99,
        description="a size the cap allows, asserted at maximum confidence on every bar",
    ),
    Rogue(
        name="doubtful_plunger",
        targets="min_confidence",
        quantity=Decimal("4000"),
        confidence=0.10,
        description="a large position it openly does not believe in; the floor should refuse it",
    ),
    Rogue(
        name="whipsawer",
        targets="max_position / never_reverses",
        quantity=Decimal("6000"),
        confidence=0.95,
        flips_side=True,
        description="flips direction every bar at size, churning fees and inverting exposure",
    ),
)

TAKER_BPS = Decimal("6")
"""One side. `cost/model.py` charges 12bps round trip; a rogue that re-states its position every
bar pays this on every change in size, which is most of what ruins it."""


def _session() -> SessionState:
    """Regular hours with fresh NAV — the *most permissive* session the Constitution offers.

    Chosen deliberately. A shut anchor or a stale NAV would let the session gates do the work and
    flatter the result; measuring in the friendliest conditions means any saving reported is
    attributable to the exposure rules rather than to the clock.
    """
    return SessionState(
        phase=SessionPhase.RTH,
        as_of=datetime(2026, 9, 14, 15, 0, tzinfo=UTC),
        hours_to_next_discovery=0.0,
        nav_age_seconds=10.0,
    )


def _hedges() -> HedgeabilitySurface:
    """One placeable hedge, so `unhedgeable_gap` does not fire and mask the size rules."""
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
        session_note="rogue harness: one placeable hedge",
    )


def _intent(rogue: Rogue, symbol: str, bar: int) -> Intent:
    side = Side.SELL if (rogue.flips_side and bar % 2) else Side.BUY
    return Intent(
        symbol=symbol,
        side=side,
        quantity=rogue.quantity,
        verdict=Verdict.TRADE,
        stated_confidence=rogue.confidence,
        thesis=f"rogue {rogue.name}: {rogue.description}",
        invalidation=("the harness ends",),
    )


@dataclass(frozen=True, slots=True)
class Arm:
    """One side of the paired run."""

    governed: bool
    final_equity: Decimal
    peak_equity: Decimal
    trough_equity: Decimal
    max_drawdown_pct: float
    orders_capped: int
    orders_refused: int
    orders_allowed: int
    binding_constraints: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "governed": self.governed,
            "final_equity": float(self.final_equity),
            "peak_equity": float(self.peak_equity),
            "trough_equity": float(self.trough_equity),
            "max_drawdown_pct": self.max_drawdown_pct,
            "orders_capped": self.orders_capped,
            "orders_refused": self.orders_refused,
            "orders_allowed": self.orders_allowed,
            "binding_constraints": dict(self.binding_constraints),
        }


def _walk(
    rogue: Rogue, closes: Sequence[float], policy: ConstitutionPolicy, *, symbol: str,
    governed: bool,
) -> Arm:
    """Run one rogue over one price path under one policy, marking to market each bar.

    Exposure is whatever the Constitution *permitted* — so the ungoverned arm carries the rogue's
    full demand and the governed arm carries the narrowed quantity. Position changes pay the taker
    fee, which is the mechanism by which the whipsawer destroys itself even when prices are kind.
    """
    equity = STARTING_EQUITY
    peak = trough = equity
    held = Decimal("0")
    held_side = Side.BUY
    capped = refused = allowed = 0
    binding: dict[str, int] = {}
    session, hedges = _session(), _hedges()

    for bar in range(len(closes) - 1):
        intent = _intent(rogue, symbol, bar)
        # The policy is re-bound to *this bar's* price. The notional caps are denominated in
        # dollars and the intent carries units, so a harness that fixed one price for the whole
        # window would be measuring the caps against a stale conversion — and the defect this
        # module found in the first place was exactly a unit mismatch.
        priced = replace(policy, reference_price=Decimal(str(closes[bar])))
        ruling = priced.rule(intent, session=session, hedges=hedges)
        permitted = ruling.resulting_intent
        binding[ruling.binding_constraint] = binding.get(ruling.binding_constraint, 0) + 1
        if permitted.quantity <= 0:
            refused += 1
        elif permitted.quantity < intent.quantity:
            capped += 1
        else:
            allowed += 1

        # Signed exposure: a SELL is negative, so a flip is a full round trip in fee terms.
        wanted = permitted.quantity if permitted.side is Side.BUY else -permitted.quantity
        current = held if held_side is Side.BUY else -held
        traded = abs(wanted - current)
        price = Decimal(str(closes[bar]))
        equity -= traded * price * TAKER_BPS / Decimal("10000")

        move = Decimal(str(closes[bar + 1])) - price
        equity += wanted * move

        held, held_side = abs(wanted), (Side.BUY if wanted >= 0 else Side.SELL)
        peak, trough = max(peak, equity), min(trough, equity)

    drawdown = float((peak - trough) / peak * 100) if peak > 0 else 0.0
    return Arm(
        governed=governed, final_equity=equity, peak_equity=peak, trough_equity=trough,
        max_drawdown_pct=round(drawdown, 2), orders_capped=capped, orders_refused=refused,
        orders_allowed=allowed, binding_constraints=binding,
    )


def _ungoverned(policy: ConstitutionPolicy) -> ConstitutionPolicy:
    """The same rulebook with every exposure ceiling raised out of reach.

    `min_confidence_to_trade` goes to 0.0 rather than being deleted, for the same reason the
    notionals become 1e12 rather than `None`: the ablated arm must run the identical code path, so
    a difference between arms is attributable to the thresholds and not to a branch one arm skipped.
    """
    return replace(
        policy,
        max_position_notional=UNGOVERNED,
        max_unhedged_notional=UNGOVERNED,
        max_gross_exposure_notional=UNGOVERNED,
        max_signed_exposure_notional=UNGOVERNED,
        min_confidence_to_trade=0.0,
    )


def _windows(closes: Sequence[float], *, bars: int = WINDOW_BARS) -> dict[str, list[float]]:
    """The worst and best contiguous stretches of the history, by simple return.

    Both, always. A harness that reports only the crash is measuring a guard in the conditions
    chosen to flatter it, and the honest description of any risk layer includes what it costs when
    the market goes the way the rogue bet.
    """
    if len(closes) < bars + 1:
        raise RogueError(
            f"{len(closes)} closes is fewer than the {bars + 1} a window needs; refusing to "
            f"report a value measured over a shorter path than claimed"
        )
    best_i = worst_i = 0
    best = worst = (closes[bars] - closes[0]) / closes[0]
    for i in range(1, len(closes) - bars):
        change = (closes[i + bars] - closes[i]) / closes[i]
        if change < worst:
            worst, worst_i = change, i
        if change > best:
            best, best_i = change, i
    return {
        "worst": list(closes[worst_i: worst_i + bars + 1]),
        "best": list(closes[best_i: best_i + bars + 1]),
    }


def run(symbol: str = "NVDAUSDT", *, days: int = 90) -> dict[str, Any]:
    """Every rogue, both arms, both windows, over real venue history."""
    from argus.market.history import fetch_range

    candles = fetch_range(symbol, days=days, interval="1H")
    closes = [float(c.close) for c in candles]
    if len(closes) < WINDOW_BARS + 1:
        raise RogueError(f"{symbol}: only {len(closes)} closes returned; need {WINDOW_BARS + 1}")
    windows = _windows(closes)
    policy = ConstitutionPolicy()
    ablated = _ungoverned(policy)

    results: list[dict[str, Any]] = []
    for rogue in ROGUES:
        for window_name, path in windows.items():
            governed = _walk(rogue, path, policy, symbol=symbol, governed=True)
            free = _walk(rogue, path, ablated, symbol=symbol, governed=False)
            saved = governed.final_equity - free.final_equity
            results.append({
                "rogue": rogue.name,
                "targets": rogue.targets,
                "description": rogue.description,
                "window": window_name,
                "window_return_pct": round((path[-1] - path[0]) / path[0] * 100, 2),
                "governed": governed.as_dict(),
                "ungoverned": free.as_dict(),
                "saved_usd": float(saved),
                "saved_pct_of_account": float(saved / STARTING_EQUITY * 100),
                "drawdown_avoided_pct": round(
                    free.max_drawdown_pct - governed.max_drawdown_pct, 2
                ),
            })

    worst = [r for r in results if r["window"] == "worst"]
    best = [r for r in results if r["window"] == "best"]
    return {
        "symbol": symbol,
        "days_of_history": days,
        "bars_per_window": WINDOW_BARS,
        "starting_equity": float(STARTING_EQUITY),
        "windows": {k: {"bars": len(v), "return_pct": round((v[-1] - v[0]) / v[0] * 100, 2)}
                    for k, v in windows.items()},
        "results": results,
        "headline": {
            "worst_window_median_saved_usd": statistics.median(r["saved_usd"] for r in worst),
            "worst_window_total_saved_usd": sum(r["saved_usd"] for r in worst),
            "best_window_median_cost_usd": statistics.median(r["saved_usd"] for r in best),
            "rogues": len(ROGUES),
            "every_rogue_was_constrained": all(
                r["governed"]["orders_allowed"] == 0 for r in results
            ),
        },
        "scope_statement": (
            "The Constitution's rule set is measured against constructed adversaries over real "
            "Bitget hourly closes, in the most permissive session it offers (regular hours, fresh "
            "NAV, one placeable hedge), with both arms running the identical code path and "
            "differing only in their thresholds. NOT CLAIMED: that ARGUS's own model would ever "
            "propose these positions — it has proposed exposure twice in 447 decisions and the "
            "risk layer has intervened zero times. This measures what the rules would have "
            "prevented, not what the desk would have done."
        ),
    }


def render(report: dict[str, Any]) -> list[str]:
    head = report["headline"]
    lines = [
        f"ROGUE VALUE — {head['rogues']} adversar(ies) against the Constitution, "
        f"{report['symbol']}, {report['bars_per_window']}-bar windows of real closes",
        f"  worst window ({report['windows']['worst']['return_pct']:+.1f}%): "
        f"median ${head['worst_window_median_saved_usd']:,.0f} saved, "
        f"${head['worst_window_total_saved_usd']:,.0f} across all rogues",
        f"  best window  ({report['windows']['best']['return_pct']:+.1f}%): "
        f"median ${head['best_window_median_cost_usd']:,.0f} — a negative number here is the "
        f"upside the guard gave up, and it is published beside the saving on purpose",
        "",
    ]
    for row in report["results"]:
        if row["window"] != "worst":
            continue
        g, u = row["governed"], row["ungoverned"]
        lines.append(
            f"  {row['rogue']:18} vs {row['targets']:32} "
            f"${row['saved_usd']:>10,.0f} saved  "
            f"(capped {g['orders_capped']}, refused {g['orders_refused']}, "
            f"allowed {g['orders_allowed']} | ungoverned drawdown {u['max_drawdown_pct']:.0f}%)"
        )
    if head["every_rogue_was_constrained"]:
        lines.append(
            "\n  Every rogue order was capped or refused — none passed through untouched, which is "
            "the property the exhaustive sweep proves and this measures the price of."
        )
    return lines


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="what the risk layer is worth, in dollars")
    parser.add_argument("--symbol", default="NVDAUSDT")
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()

    report = run(args.symbol, days=args.days)
    for line in render(report):
        print(line)
    write(REPORT_PATH, report)
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["REPORT_PATH", "ROGUES", "Rogue", "RogueError", "main", "render", "run"]
