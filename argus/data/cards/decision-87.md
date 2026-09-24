# Decision 87 — NVDAUSDT — no_trade

**Decided** 2026-09-13T03:22:02.799291+00:00 during the weekend session, 34.13h from the next price discovery.
**Size** 0 BUY at 217.91; entry cost 0.000bps.
**Stated confidence** 0.88

## Why
Weekend session with 34 hours until genuine price discovery, no NVDA-specific catalyst, no actionable edge exceeding the 18.80 bps hurdle, and all panel assessments neutral. The technical oversold reading is already priced in and lacks independent confirmation.

## What would make this wrong
- A material NVDA-specific catalyst (e.g., export restriction news, major contract announcement, or guidance revision) emerges before Monday open that would move the stock more than 19 bps
- Token price gaps significantly away from fair value relative to underlying NVDA futures when they reopen

## What the checkers found
3 finding(s):
- [conflict:conviction] event (neutral) vs sentiment (neutral), gap 0.47 — event dominates: confidence 0.82 against 0.35
- [conflict:conviction] sentiment (neutral) vs cross_asset (neutral), gap 0.60 — cross_asset dominates: confidence 0.95 against 0.35
- [conflict:conviction] earnings (neutral) vs cross_asset (neutral), gap 0.40 — cross_asset dominates: confidence 0.95 against 0.55

<details><summary>every check that ran</summary>

- [panel] 3 of 3 analysts run: event, sentiment, earnings
- causal chain: 8 links stated, gradable at the next price discovery
- hedge menu empty: nothing placeable for 34.1h; 100% of risk carried as priced residual
- panel: 4 analysts, 17 distinct sources, independence 4.25 -> neutral at 0.67 after provenance discount
- [conflict:conviction] event (neutral) vs sentiment (neutral), gap 0.47 — event dominates: confidence 0.82 against 0.35
- [conflict:conviction] sentiment (neutral) vs cross_asset (neutral), gap 0.60 — cross_asset dominates: confidence 0.95 against 0.35
- [conflict:conviction] earnings (neutral) vs cross_asset (neutral), gap 0.40 — cross_asset dominates: confidence 0.95 against 0.55
- [grounding] all 1 figure(s) resolve to a computed value or a cited fact
- [claim] no checkable claim found in the thesis (19 structured record(s) available)
- no order: final verdict no_trade with quantity 0
- [skills] 6 of 6 official-Skill calls answered for NVDAUSDT; 1 of 5 Skills reached (technical-analysis)

</details>

## What the risk layer did
nothing to narrow — no exposure proposed; nothing to narrow. The Constitution may only reduce, so an untouched decision is the model's own.

## The rules in force
— not recorded — — this decision predates the pre-registered protocol and is reported as ungoverned rather than as compliant

## What happened next
not settled yet — it is held for 24 hours and graded against a price fetched in a later cycle, never one available when it was decided

## Integrity
market state `27c84d601a66b2b7`, approved intent `ce1acb2e7f66de3e`, chained to `b788b68158e90fb2`.

## Where each part came from
- the decision: data/paper_ledger.jsonl
- the checks: data/desk_notes.jsonl
- the risk ruling: data/risk_records.jsonl