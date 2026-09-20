# Decision 64 — NVDAUSDT — no_trade

**Decided** 2026-09-12T18:13:31.953607+00:00 during the weekend session, 43.27h from the next price discovery.
**Size** 0 BUY at 219.25; entry cost 0.000bps.
**Stated confidence** 0.92

## Why
Weekend session with 43+ hours until genuine price discovery, flat 24h price action, no actionable catalysts in the news flow (all recycled analyst commentary), empty hedge menu, and no position to manage. The total hurdle of 18.80 bps cannot be cleared by any identifiable edge.

## What would make this wrong
- A material, credible news event (e.g., unexpected earnings guidance, major contract announcement, or regulatory action) emerges before Monday open that would shift fair value by more than 19 bps
- Token price gaps significantly from anchor market price upon reopening, indicating information leakage not yet reflected

## What the checkers found
2 finding(s):
- [grounding] 1 of 3 figure(s) do not resolve to anything the desk was given: -0.055%
- [grounding] an unattributable number in a thesis is the easiest place for an unsupported fact to enter the record

<details><summary>every check that ran</summary>

- [panel] 1 of 3 analysts run: event
- [panel] sentiment not run — no evidence on its channels (social). 0 piece(s) of evidence it would have read went unexamined; that is a judgement about cost, not about their worth
- [panel] earnings not run — no evidence on its channels (filing, transcript). 0 piece(s) of evidence it would have read went unexamined; that is a judgement about cost, not about their worth
- [panel] 4.534bps of deliberation not spent, against a 12bps round trip
- causal chain: 7 links stated, gradable at the next price discovery
- hedge menu empty: nothing placeable for 43.3h; 100% of risk carried as priced residual
- panel: 2 analysts, 5 distinct sources, independence 2.50 -> neutral at 0.90 after provenance discount
- [conflict] none: 2 analysts agreed on direction and size — but they ran in sequence, so agreement may be contagion rather than consensus
- [grounding] 1 of 3 figure(s) do not resolve to anything the desk was given: -0.055%
- [grounding] an unattributable number in a thesis is the easiest place for an unsupported fact to enter the record
- [claim] no structured evidence to check the thesis against
- no order: final verdict no_trade with quantity 0

</details>

## What the risk layer did
— not recorded — (no risk record for this sequence)

## The rules in force
— not recorded — — this decision predates the pre-registered protocol and is reported as ungoverned rather than as compliant

## What happened next
not settled yet — it is held for 24 hours and graded against a price fetched in a later cycle, never one available when it was decided

## Integrity
market state `01b7c6c4ad49f49d`, approved intent `87a19e78406c1327`, chained to `dab7c7c20fe317d0`.

## Where each part came from
- the decision: data/paper_ledger.jsonl
- the checks: data/desk_notes.jsonl