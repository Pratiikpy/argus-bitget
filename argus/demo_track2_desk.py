"""Track 2 end-to-end: five analysts, a real decision, a signed order, a proof.

    set -a && . ../.secrets/qwen.env && set +a && python demo_track2_desk.py

Runs on **live Bitget market data** with the anchor market in whatever state it is actually in when
you run it. The scenario evidence is synthetic and labelled as such; everything else — price,
session phase, hours to discovery, spread, fee — is measured.

This is the "event -> decision -> execution flow" the track requires, and it produces the Autonomy
Proof that answers the one question a judge cannot check from a diagram: did the LLM actually
decide this trade?
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from argus.agents.analysts import Evidence
from argus.agents.desk import ConstitutionPolicy, TradingDesk
from argus.cost.model import CostModel
from argus.llm.qwen import QwenClient, QwenError, TokenBudget
from argus.market.bitget import fetch_rtokens
from argus.risk.hedgeability import HedgeabilitySurface, shut_market_candidate
from argus.truth.clocks import DualClock

SYMBOL = "NVDAUSDT"


def main() -> int:
    clock = DualClock()
    now = datetime.now(UTC)

    print("=" * 78)
    print("LIVE MARKET".center(78))
    print("=" * 78)
    try:
        tickers = fetch_rtokens()
    except Exception as exc:  # noqa: BLE001
        print(f"market data unavailable: {exc}", file=sys.stderr)
        return 1

    ticker = tickers[SYMBOL]
    session = clock.state(now, nav_age_seconds=40_000)
    print(f"  symbol                 {SYMBOL} (anchor {ticker.anchor})")
    print(f"  last / spread          {ticker.last}  /  {ticker.spread_bps:.2f} bps")
    print(f"  anchor phase           {session.phase}  (asleep: {session.is_anchor_asleep})")
    print(f"  hours to discovery     {session.hours_to_next_discovery:.1f}")
    print(f"  round trip fee         {CostModel.bitget_perp().round_trip_bps()} bps "
          f"= {CostModel.bitget_perp().round_trip_bps() / max(ticker.spread_bps, Decimal('0.01')):.0f}x "
          f"the spread")

    # Hedges that would work are on venues that are shut. Kept on the surface so the absence is
    # visible rather than silently omitted.
    hedges = HedgeabilitySurface(
        (
            shut_market_candidate("NVDA", Decimal("0.95")),
            shut_market_candidate("SOXX", Decimal("0.70")),
        ),
        session_note=f"{session.phase}; {session.hours_to_next_discovery:.1f}h to discovery",
    ) if session.is_anchor_asleep else HedgeabilitySurface(())

    # Synthetic scenario evidence, clearly labelled. Timestamps are before `now`, so the
    # point-in-time bound holds.
    evidence = [
        Evidence("e1", "[SYNTHETIC] SEC 8-K: FY guidance revised down 7%.",
                 "sec-edgar", now - timedelta(minutes=29), credibility=0.99),
        Evidence("s1", "[SYNTHETIC] Viral posts claim a 20% cut. No independent source.",
                 "social", now - timedelta(minutes=14), credibility=0.18),
        Evidence("f1", "[SYNTHETIC] EPS +12% vs consensus +8%; revenue +4% vs +6%.",
                 "filing", now - timedelta(minutes=31), credibility=0.99),
        Evidence("t1", "[SYNTHETIC] Management declined three questions on datacentre orders.",
                 "transcript", now - timedelta(minutes=25), credibility=0.95),
    ]

    try:
        client = QwenClient(budget=TokenBudget(limit=200_000))
        desk = TradingDesk(client)

        print("\n" + "=" * 78)
        print("THE PANEL — four sub-theme analysts".center(78))
        print("=" * 78)

        run = desk.run(
            symbol=SYMBOL,
            session=session,
            token_price=ticker.last,
            position=Decimal("200"),
            evidence=evidence,
            hedges=hedges,
            decision_id="t2-demo",
            constitution=ConstitutionPolicy(),
        )

        for view in run.panel.views:
            flag = "ACTIONABLE" if view.is_actionable else "not actionable"
            print(f"\n  {view.analyst.upper():<12} {view.signal:<22} "
                  f"{view.magnitude_bps:>5}bps  conf {view.confidence:.2f}  [{flag}]")
            print(f"    {view.reasoning[:150]}")
            if view.counter_case:
                print(f"    counter: {view.counter_case[:130]}")

        panel = run.panel.as_dict()
        print(f"\n  independence      {panel['distinct_sources']} distinct sources across "
              f"{panel['analysts']} analysts = {panel['independence_ratio']}")
        print(f"  consensus         {panel['consensus_signal']} at "
              f"{panel['provenance_discounted_confidence']} (after provenance discount)")

        print("\n" + "=" * 78)
        print("THE DECISION".center(78))
        print("=" * 78)
        o = run.proof.llm_original_intent
        print(f"  LLM proposed      {o.verdict} {o.side} {o.quantity}  conf {o.stated_confidence}")
        print(f"    thesis          {o.thesis[:160]}")
        for inv in o.invalidation[:3]:
            print(f"    invalidated if  {inv[:140]}")

        if run.ruling:
            print(f"\n  CONSTITUTION      {run.ruling.verdict}  "
                  f"[{run.ruling.binding_constraint}]")
            print(f"    reason          {run.ruling.reason[:150]}")
            print(f"    permitted       {run.ruling.resulting_intent.quantity}")

        r = run.proof.llm_revised_intent
        if r:
            print(f"\n  LLM revised       {r.verdict} {r.side} {r.quantity}")
            print(f"    thesis          {r.thesis[:160]}")

        print("\n" + "=" * 78)
        print("PROOF".center(78))
        print("=" * 78)
        print(f"  constitution intervened      {run.proof.constitution_intervened}")
        print(f"  constitution ONLY reduced    {run.proof.constitution_only_reduced()}")
        print(f"  LLM changed its mind         {run.proof.llm_changed_its_mind}")
        print(f"  LLM genuinely decided        {run.proof.attests_llm_decided()}")
        print(f"  approved intent hash         {run.proof.approved_intent_hash}")
        if run.order:
            print(f"  order                        {run.order.client_order_id} "
                  f"{run.order.side} {run.order.quantity} -> {run.order.state}")
        for note in run.notes:
            print(f"  note                         {note}")

        out = "data/track2_desk_run.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(run.as_dict(), fh, indent=2, default=str)
        print(f"\n  full record -> {out}")

        if client.budget:
            print(f"  tokens {client.budget.spent}/{client.budget.limit} "
                  f"({client.calls} calls, {client.cache_hits} cached)")
        return 0

    except QwenError as exc:
        print(f"\n[LLM unavailable] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
