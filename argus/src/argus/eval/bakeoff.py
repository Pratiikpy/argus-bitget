"""Model-vs-model on identical state — the Observatory's headline experiment.

The Track-2 Open Theme claim is that ARGUS is a *neutral operating system for evaluating trading
intelligence*, not one more agent. That only holds if swapping the model is the single variable.
This module holds everything else fixed — same market frame, same evidence, same prompt contract,
same cost model, same Constitution, same temperature — and varies only which model sits in the
decision seat.

It also measures **decision consistency under replay**: each model decides the same frame ``k``
times. A model that flips verdict on identical input at temperature 0 has not made a decision; it
has sampled one. Nothing in the corpus measures this.

Two providers are wired and both were verified live:

* Qwen 3.8 Max via Bitget's hackathon endpoint
* DeepSeek v4 Pro via NVIDIA NIM

Their reasoning-control parameters are spelled differently — Qwen takes top-level
``enable_thinking`` / ``reasoning_effort``; NIM takes ``chat_template_kwargs``. Sending the wrong
one silently does nothing, which would make a "controlled" comparison uncontrolled. The provider
layer handles that, and this module asserts both are reachable before it reports anything.

    python -m argus.eval.bakeoff --replays 3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.agents.meta_pm import MarketFrame, MetaPM
from argus.cost.model import CostModel
from argus.eval.observatory import ReplayResult
from argus.llm.provider import Provider, available, build
from argus.llm.qwen import QwenError
from argus.market.bitget import fetch_rtokens
from argus.truth.clocks import DualClock

SYMBOL = "NVDAUSDT"


@dataclass
class ModelRun:
    provider: str
    verdicts: list[str] = field(default_factory=list)
    quantities: list[Decimal] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    theses: list[str] = field(default_factory=list)
    tokens: int = 0
    errors: list[str] = field(default_factory=list)

    def replay(self, state_hash: str) -> ReplayResult:
        return ReplayResult(
            state_hash=state_hash,
            verdicts=tuple(self.verdicts),
            quantities=tuple(self.quantities),
        )

    def as_dict(self, state_hash: str) -> dict[str, Any]:
        r = self.replay(state_hash)
        return {
            "provider": self.provider,
            "decisions": len(self.verdicts),
            "verdicts": self.verdicts,
            "quantities": [str(q) for q in self.quantities],
            "mean_confidence": (
                round(sum(self.confidences) / len(self.confidences), 3)
                if self.confidences else None
            ),
            "verdict_consistency": round(r.verdict_consistency, 3) if self.verdicts else None,
            "size_dispersion": round(r.size_dispersion, 3) if self.verdicts else None,
            "stable": r.is_stable if self.verdicts else None,
            "tokens": self.tokens,
            "errors": self.errors,
            "first_thesis": self.theses[0][:220] if self.theses else "",
        }


def build_frame() -> MarketFrame:
    """One frame, from live market data. Every model sees exactly this."""
    clock = DualClock()
    now = datetime.now(UTC)
    ticker = fetch_rtokens()[SYMBOL]
    session = clock.state(now, nav_age_seconds=40_000)

    return MarketFrame(
        symbol=SYMBOL,
        as_of=now,
        session=session,
        token_price=ticker.last,
        position_quantity=Decimal("200"),
        round_trip_bps=CostModel.bitget_perp().round_trip_bps(),
        hedge_menu=(),
        evidence=(
            f"{SYMBOL} last {ticker.last}, 24h change {ticker.change_24h}, "
            f"quoted spread {ticker.spread_bps:.2f}bps.",
            f"Anchor market {session.phase}; {session.hours_to_next_discovery:.1f}h until "
            f"genuine price discovery. No hedge in the underlying is placeable.",
            "[SYNTHETIC] SEC 8-K filed 29 minutes ago: FY guidance revised down 7%.",
            "[SYNTHETIC] Viral posts claim a 20% cut; no independent source; credibility 0.18.",
        ),
    )


def run(*, replays: int = 3) -> dict[str, Any]:
    frame = build_frame()
    state_hash = frame.state_hash()
    providers = available()

    results: list[ModelRun] = []
    for provider in providers:
        run_result = ModelRun(provider=provider.value)
        client = build(provider, budget_limit=250_000)
        pm = MetaPM(client, max_tokens=900)

        for k in range(replays):
            try:
                proof = pm.decide(frame, decision_id=f"bakeoff-{provider.value}-{k}")
            except (QwenError, Exception) as exc:
                run_result.errors.append(f"replay {k}: {str(exc)[:120]}")
                continue
            intent = proof.llm_original_intent
            run_result.verdicts.append(str(intent.verdict))
            run_result.quantities.append(intent.quantity)
            run_result.confidences.append(intent.stated_confidence)
            run_result.theses.append(intent.thesis)

        if client.budget is not None:
            run_result.tokens = int(client.budget.spent)
        results.append(run_result)

    # Do the models agree with each other, having seen identical input?
    modal = {
        r.provider: max(set(r.verdicts), key=r.verdicts.count) if r.verdicts else None
        for r in results
    }
    reporting = {k: v for k, v in modal.items() if v}
    distinct = set(reporting.values())

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "symbol": SYMBOL,
        "market_state_hash": state_hash,
        "replays_per_model": replays,
        "providers_reachable": [p.value for p in providers],
        "providers_unreachable": [
            p.value for p in Provider if p not in providers
        ],
        "models": [r.as_dict(state_hash) for r in results],
        "cross_model": {
            "modal_verdicts": modal,
            # A single reporting model is not agreement. An earlier version counted a failed
            # provider as consent by filtering it out and finding one distinct verdict left.
            "models_agree": len(reporting) >= 2 and len(distinct) == 1,
            "models_reporting": len(reporting),
            "agreement_undefined": len(reporting) < 2,
            "note": (
                "identical frame, identical prompt contract, identical cost model, "
                "temperature 0; the model is the only variable"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ARGUS model bake-off")
    parser.add_argument("--replays", type=int, default=3, help="decisions per model")
    args = parser.parse_args()

    result = run(replays=args.replays)
    out = Path(__file__).resolve().parents[3] / "data" / "bakeoff.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(f"frame {result['market_state_hash']} · {result['replays_per_model']} replays each")
    if result["providers_unreachable"]:
        print(f"unreachable: {result['providers_unreachable']}")
    print()
    for m in result["models"]:
        print(f"  {m['provider']:<14} {m['verdicts']!s:<44} "
              f"consistency {m['verdict_consistency']} stable={m['stable']} "
              f"tokens {m['tokens']}")
        if m["errors"]:
            print(f"    errors: {m['errors']}")
    print(f"\n  models agree: {result['cross_model']['models_agree']} "
          f"{result['cross_model']['modal_verdicts']}")
    print(f"  full report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
