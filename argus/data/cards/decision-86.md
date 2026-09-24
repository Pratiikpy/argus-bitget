# Decision 86 — NVDAUSDT — no_trade

**Decided** 2026-09-13T02:32:29.487840+00:00 during the weekend session, 34.96h from the next price discovery.
**Size** 0 BUY at 217.98; entry cost 0.000bps.
**Stated confidence** 0.72

## Why
Weekend session with 35 hours until genuine price discovery, no hedge available, and no NVDA-specific catalyst that could move the token more than the 18.80 bps total hurdle before Monday's open. Technicals are bearish (MACD death cross, price below all MAs) but RSI is oversold, creating conflicting signals with no clear directional edge above cost.

## What would make this wrong
- NVDA token price gaps more than 50 bps in either direction within the first hour of Monday's open, indicating a weekend catalyst was missed
- A material NVDA-specific news event (earnings guidance revision, major product announcement, regulatory action) breaks before Monday's open

## What the checkers found
2 finding(s):
- [conflict:conviction] event (neutral) vs earnings (neutral), gap 0.37 — event dominates: confidence 0.72 against 0.35
- [conflict:conviction] earnings (neutral) vs cross_asset (neutral), gap 0.60 — cross_asset dominates: confidence 0.95 against 0.35

<details><summary>every check that ran</summary>

- [panel] 3 of 3 analysts run: event, sentiment, earnings
- causal chain: 7 links stated, gradable at the next price discovery
- hedge menu empty: nothing placeable for 35.0h; 100% of risk carried as priced residual
- panel: 4 analysts, 17 distinct sources, independence 4.25 -> neutral at 0.67 after provenance discount
- [conflict:conviction] event (neutral) vs earnings (neutral), gap 0.37 — event dominates: confidence 0.72 against 0.35
- [conflict:conviction] earnings (neutral) vs cross_asset (neutral), gap 0.60 — cross_asset dominates: confidence 0.95 against 0.35
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
market state `b354c1fdce4a305d`, approved intent `970c21848de2c982`, chained to `5cb22d662b1e7c31`.

## Where each part came from
- the decision: data/paper_ledger.jsonl
- the checks: data/desk_notes.jsonl
- the risk ruling: data/risk_records.jsonl