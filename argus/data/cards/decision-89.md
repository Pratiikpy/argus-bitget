# Decision 89 — AAPLUSDT — no_trade

**Decided** 2026-09-13T03:22:02.799291+00:00 during the weekend session, 34.13h from the next price discovery.
**Size** 0 BUY at 333.63; entry cost 0.000bps.
**Stated confidence** 0.82

## Why
Weekend session with 34+ hours until genuine price discovery, no hedge menu available, flat fundamentals (QoQ EPS +0.5%), mixed-to-negative consensus revisions (14 down vs 7 up for current quarter), and contradictory technical signals (RSI overbought at 73.56 vs MACD golden cross) provide no edge exceeding the 18.80 bps total hurdle.

## What would make this wrong
- AAPLUSDT gaps up more than 2% at Monday open on a material catalyst not yet visible in current evidence
- Consensus revisions flip net positive before next session with a credible upward catalyst

## What the checkers found
1 finding(s):
- [conflict:conviction] earnings (neutral) vs cross_asset (neutral), gap 0.60 — cross_asset dominates: confidence 0.95 against 0.35

<details><summary>every check that ran</summary>

- [panel] 2 of 3 analysts run: sentiment, earnings
- [panel] event not run — relevance 0.10 below 0.35 — its evidence is stale, low-credibility, or carries nothing this analyst reads. 8 piece(s) of evidence it would have read went unexamined; that is a judgement about cost, not about their worth
- [panel] 2.267bps of deliberation not spent, against a 12bps round trip
- hedge menu empty: nothing placeable for 34.1h; 100% of risk carried as priced residual
- panel: 3 analysts, 11 distinct sources, independence 3.67 -> neutral at 0.65 after provenance discount
- [conflict:conviction] earnings (neutral) vs cross_asset (neutral), gap 0.60 — cross_asset dominates: confidence 0.95 against 0.35
- [grounding] all 5 figure(s) resolve to a computed value or a cited fact
- [claim] no checkable claim found in the thesis (17 structured record(s) available)
- no order: final verdict no_trade with quantity 0
- [skills] 6 of 6 official-Skill calls answered for AAPLUSDT; 1 of 5 Skills reached (technical-analysis)

</details>

## What the risk layer did
nothing to narrow — no exposure proposed; nothing to narrow. The Constitution may only reduce, so an untouched decision is the model's own.

## The rules in force
— not recorded — — this decision predates the pre-registered protocol and is reported as ungoverned rather than as compliant

## What happened next
not settled yet — it is held for 24 hours and graded against a price fetched in a later cycle, never one available when it was decided

## Integrity
market state `51f93b0414a40dc1`, approved intent `7c8f084696cc0dfd`, chained to `4d05b8e59fe41258`.

## Where each part came from
- the decision: data/paper_ledger.jsonl
- the checks: data/desk_notes.jsonl
- the risk ruling: data/risk_records.jsonl