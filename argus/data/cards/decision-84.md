# Decision 84 — TSLAUSDT — no_trade

**Decided** 2026-09-13T01:45:26.486373+00:00 during the weekend session, 35.74h from the next price discovery.
**Size** 0 BUY at 368.19; entry cost 0.000bps.
**Stated confidence** 0.92

## Why
Weekend session with anchor market asleep for 35.7 hours, no hedge menu available, no actionable catalyst, and technical signals (MACD golden cross, bullish MAs) are insufficient to clear the 18.80 bps total hurdle given the absence of genuine price discovery until Monday.

## What would make this wrong
- A material news catalyst emerges over the weekend that would gap the token significantly at open
- Token price breaks below ATR-based stop at 364.04 indicating the bullish structure has failed

## What the checkers found
1 finding(s):
- [conflict:conviction] sentiment (neutral) vs cross_asset (neutral), gap 0.60 — cross_asset dominates: confidence 0.95 against 0.35

<details><summary>every check that ran</summary>

- [panel] 1 of 3 analysts run: sentiment
- [panel] event not run — relevance 0.10 below 0.35 — its evidence is stale, low-credibility, or carries nothing this analyst reads. 8 piece(s) of evidence it would have read went unexamined; that is a judgement about cost, not about their worth
- [panel] earnings not run — no evidence on its channels (filing, transcript)
- [panel] 4.534bps of deliberation not spent, against a 12bps round trip
- hedge menu empty: nothing placeable for 35.7h; 100% of risk carried as priced residual
- panel: 2 analysts, 4 distinct sources, independence 2.00 -> neutral at 0.65 after provenance discount
- [conflict:conviction] sentiment (neutral) vs cross_asset (neutral), gap 0.60 — cross_asset dominates: confidence 0.95 against 0.35
- [grounding] all 2 figure(s) resolve to a computed value or a cited fact
- [claim] no checkable claim found in the thesis (6 structured record(s) available)
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
market state `c61e82f84374e5b3`, approved intent `9b06103324abc05d`, chained to `1423049fdded9eed`.

## Where each part came from
- the decision: data/paper_ledger.jsonl
- the checks: data/desk_notes.jsonl
- the risk ruling: data/risk_records.jsonl