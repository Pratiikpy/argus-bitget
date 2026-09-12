"""The Sleeping-Anchor demo: one decision, end to end, against the live model.

    set -a && . ../.secrets/qwen.env && set +a && python demo_sleeping_anchor.py

Sunday 03:00 ET. A real 8-K cut guidance 26 minutes ago. The rToken trades; NYSE has been shut for
35 hours and will not open for another 30.5. No hedge is placeable. A viral post exaggerates the
news. The round trip costs 12bps.

The model decides. The Constitution narrows. The model decides again. The proof is printed.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from decimal import Decimal

from argus.agents.meta_pm import MarketFrame, MetaPM
from argus.cost.model import CostModel
from argus.decision.verdicts import ConstitutionVerdict, apply_constraint
from argus.llm.qwen import QwenClient, QwenError, TokenBudget
from argus.risk.hedgeability import HedgeabilitySurface, shut_market_candidate
from argus.truth.clocks import ET, DualClock

SUNDAY_3AM = datetime(2026, 3, 8, 3, 0, tzinfo=ET)


def main() -> int:
    clock = DualClock()
    session = clock.state(SUNDAY_3AM, nav_age_seconds=40_000)

    # Every hedge that would work is on a venue that is shut. The surface keeps them so the
    # absence is visible rather than silently omitted.
    surface = HedgeabilitySurface(
        (
            shut_market_candidate("NVDA", Decimal("0.95")),
            shut_market_candidate("SOXX", Decimal("0.70")),
        ),
        session_note="NYSE shut; next discovery in 30.5h",
    )

    print("=" * 78)
    print("LEVEL 1 — TRUTH".center(78))
    print("=" * 78)
    print(f"  as of                  {SUNDAY_3AM.isoformat()}")
    print(f"  anchor phase           {session.phase}")
    print(f"  anchor asleep          {session.is_anchor_asleep}")
    print(f"  hours to discovery     {session.hours_to_next_discovery:.1f}")
    print(f"  NAV stale              {session.nav_is_stale()}")
    print(f"  token clock            continuous (knowable)")
    print(f"  anchor clock           shut (not priceable, not hedgeable)")
    print(f"\n  hedge menu             {'EMPTY' if surface.is_empty else surface.menu}")
    print(f"  residual carried       {surface.residual_after(None)} of position")

    frame = MarketFrame(
        symbol="rNVDA",
        as_of=SUNDAY_3AM,
        session=session,
        token_price=Decimal("118.40"),
        position_quantity=Decimal("200"),
        round_trip_bps=CostModel.bitget_perp().round_trip_bps(),
        hedge_menu=(),
        evidence=(
            "SEC 8-K filed 02:31 ET: FY guidance revised down 7%. (available_at 02:34 ET)",
            "Viral social post claims a 20% cut; no independent source. Credibility 0.18.",
            "BTC -3.7% over 6h. Token liquidity roughly a third of RTH depth.",
        ),
    )
    print(f"  market state hash      {frame.state_hash()}")

    try:
        client = QwenClient(budget=TokenBudget(limit=80_000))
        pm = MetaPM(client, max_tokens=900)

        print("\n" + "=" * 78)
        print("LEVEL 2 — THE LLM DECIDES (unconstrained)".center(78))
        print("=" * 78)
        proof = pm.decide(frame, decision_id="demo-1")
        o = proof.llm_original_intent
        print(f"  verdict                {o.verdict}")
        print(f"  side / quantity        {o.side} {o.quantity}")
        print(f"  confidence             {o.stated_confidence}")
        print(f"  thesis                 {o.thesis}")
        print(f"  invalidation           {list(o.invalidation)}")
        print(f"  reasoning              {proof.llm_original_reasoning[:200]}")

        print("\n" + "=" * 78)
        print("LEVEL 3 — THE CONSTITUTION NARROWS (it may only reduce)".center(78))
        print("=" * 78)
        if o.quantity > 0:
            ruling = apply_constraint(
                o,
                verdict=ConstitutionVerdict.RESIZE,
                binding_constraint="unhedgeable_gap",
                reason=(
                    "no hedge is placeable for 30.5h; the entire position is carried as priced "
                    "residual and gap CVaR exceeds the weekend budget"
                ),
                resized_quantity=max(Decimal("1"), o.quantity / Decimal("2")),
            )
        else:
            ruling = apply_constraint(
                o,
                verdict=ConstitutionVerdict.ALLOW,
                binding_constraint="none",
                reason="no exposure proposed; nothing to narrow",
            )
        print(f"  verdict                {ruling.verdict}")
        print(f"  binding constraint     {ruling.binding_constraint}")
        print(f"  reason                 {ruling.reason}")
        print(f"  permitted quantity     {ruling.resulting_intent.quantity}")

        print("\n" + "=" * 78)
        print("LEVEL 2 AGAIN — THE LLM RESPONDS TO BEING CONSTRAINED".center(78))
        print("=" * 78)
        pm.revise(proof, frame, ruling)
        r = proof.llm_revised_intent
        assert r is not None
        print(f"  verdict                {r.verdict}")
        print(f"  side / quantity        {r.side} {r.quantity}")
        print(f"  thesis                 {r.thesis}")
        print(f"  changed its mind       {proof.llm_changed_its_mind}")

        proof.approve()

        print("\n" + "=" * 78)
        print("AUTONOMY PROOF".center(78))
        print("=" * 78)
        print(f"  constitution intervened     {proof.constitution_intervened}")
        print(f"  constitution ONLY reduced   {proof.constitution_only_reduced()}")
        print(f"  LLM genuinely decided       {proof.attests_llm_decided()}")
        print(f"  approved intent hash        {proof.approved_intent_hash}")

        print("\n--- full record ---")
        print(json.dumps(proof.to_record(), indent=2, default=str))

        usage = client.budget
        if usage is not None:
            print(f"\n  tokens spent {usage.spent} / {usage.limit}  "
                  f"({client.calls} calls, {client.cache_hits} cache hits)")
        return 0

    except QwenError as exc:
        print(f"\n[LLM unavailable] {exc}", file=sys.stderr)
        print("Levels 1 and 3 above ran without the model; they are deterministic.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
