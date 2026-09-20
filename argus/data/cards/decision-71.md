# Decision 71 — TSLAUSDT — no_trade

**Decided** 2026-09-13T01:22:03.015248+00:00 during the weekend session, 36.13h from the next price discovery.
**Size** 0 BUY at 368.01; entry cost 0.000bps.
**Stated confidence** 0.82

## Why
Weekend session with anchor market asleep for 36+ hours, no hedge menu available, no actionable catalyst, and consensus neutral. The 18.80 bps total hurdle cannot be cleared without a genuine edge, and none is present.

## What would make this wrong
- A material news event (e.g., regulatory action, major delivery data, or Musk announcement) that creates a directional gap exceeding the 18.80 bps hurdle before Monday open

## What the checkers found
1 finding(s):
- [conflict:conviction] earnings (neutral) vs cross_asset (neutral), gap 0.60 — cross_asset dominates: confidence 0.95 against 0.35

<details><summary>every check that ran</summary>

- [panel] 1 of 3 analysts run: earnings
- [panel] event not run — relevance 0.10 below 0.35 — its evidence is stale, low-credibility, or carries nothing this analyst reads. 9 piece(s) of evidence it would have read went unexamined; that is a judgement about cost, not about their worth
- [panel] sentiment not run — no evidence on its channels (social)
- [panel] 4.534bps of deliberation not spent, against a 12bps round trip
- hedge menu empty: nothing placeable for 36.1h; 100% of risk carried as priced residual
- panel: 2 analysts, 6 distinct sources, independence 3.00 -> neutral at 0.65 after provenance discount
- [conflict:conviction] earnings (neutral) vs cross_asset (neutral), gap 0.60 — cross_asset dominates: confidence 0.95 against 0.35
- [grounding] all 2 figure(s) resolve to a computed value or a cited fact
- [claim] no checkable claim found in the thesis (10 structured record(s) available)
- no order: final verdict no_trade with quantity 0

</details>

## What the risk layer did
— not recorded — (no risk record for this sequence)

## The rules in force
— not recorded — — this decision predates the pre-registered protocol and is reported as ungoverned rather than as compliant

## What happened next
not settled yet — it is held for 24 hours and graded against a price fetched in a later cycle, never one available when it was decided

## Integrity
market state `4ea8d3c0ebb4423c`, approved intent `e2c3571ce7ea79dd`, chained to `21a872e4b5d7dc1e`.

## Where each part came from
- the decision: data/paper_ledger.jsonl
- the checks: data/desk_notes.jsonl