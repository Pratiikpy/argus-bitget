# Decision 88 — TSLAUSDT — no_trade

**Decided** 2026-09-13T03:22:02.799291+00:00 during the weekend session, 34.13h from the next price discovery.
**Size** 0 BUY at 367.5; entry cost 0.000bps.
**Stated confidence** 0.82

## Why
Weekend session with anchor market asleep for 34+ hours, no hedge available, consensus estimates showing heavy downward revisions (15 down vs 4 up current quarter, 16 down vs 3 up next quarter), and no actionable catalyst to clear the 18.80 bps total hurdle. The strong Q2 sequential recovery is already priced in at $367.50, and the forward revision skew is bearish.

## What would make this wrong
- If TSLA gaps up more than 2% at Monday open on a material catalyst not yet in evidence (e.g., new product announcement, major contract win, or positive regulatory development), the bearish revision thesis would be overtaken.
- If consensus revisions flip net positive (more upward than downward) before the next earnings report, the forward earnings deterioration thesis is invalidated.

## What the checkers found
2 finding(s):
- [conflict:conviction] sentiment (neutral) vs earnings (neutral), gap 0.37 — sentiment dominates: confidence 0.72 against 0.35
- [conflict:conviction] earnings (neutral) vs cross_asset (neutral), gap 0.60 — cross_asset dominates: confidence 0.95 against 0.35

<details><summary>every check that ran</summary>

- [panel] 2 of 3 analysts run: sentiment, earnings
- [panel] event not run — relevance 0.10 below 0.35 — its evidence is stale, low-credibility, or carries nothing this analyst reads. 9 piece(s) of evidence it would have read went unexamined; that is a judgement about cost, not about their worth
- [panel] 2.267bps of deliberation not spent, against a 12bps round trip
- hedge menu empty: nothing placeable for 34.1h; 100% of risk carried as priced residual
- panel: 3 analysts, 12 distinct sources, independence 4.00 -> neutral at 0.67 after provenance discount
- [conflict:conviction] sentiment (neutral) vs earnings (neutral), gap 0.37 — sentiment dominates: confidence 0.72 against 0.35
- [conflict:conviction] earnings (neutral) vs cross_asset (neutral), gap 0.60 — cross_asset dominates: confidence 0.95 against 0.35
- [grounding] all 6 figure(s) resolve to a computed value or a cited fact
- [claim] no checkable claim found in the thesis (18 structured record(s) available)
- no order: final verdict no_trade with quantity 0
- [skills] 6 of 6 official-Skill calls answered for TSLAUSDT; 1 of 5 Skills reached (technical-analysis)

</details>

## What the risk layer did
nothing to narrow — no exposure proposed; nothing to narrow. The Constitution may only reduce, so an untouched decision is the model's own.

## The rules in force
— not recorded — — this decision predates the pre-registered protocol and is reported as ungoverned rather than as compliant

## What happened next
not settled yet — it is held for 24 hours and graded against a price fetched in a later cycle, never one available when it was decided

## Integrity
market state `cca83eb192f30547`, approved intent `40442223516a4382`, chained to `42cc4782d359c4b7`.

## Where each part came from
- the decision: data/paper_ledger.jsonl
- the checks: data/desk_notes.jsonl
- the risk ruling: data/risk_records.jsonl