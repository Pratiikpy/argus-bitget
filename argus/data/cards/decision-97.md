# Decision 97 — TSLAUSDT — no_trade

**Decided** 2026-09-13T05:22:03.626841+00:00 during the weekend session, 32.13h from the next price discovery.
**Size** 0 BUY at 366.85; entry cost 0.000bps.
**Stated confidence** 0.82

## Why
Weekend session with anchor market asleep for 32+ hours, no hedge available, consensus estimates showing aggressive downward revisions (4 up / 15 down for Q3, 3 up / 16 down for Q4), and no actionable catalyst. The 18.80 bps hurdle cannot be cleared with the available evidence.

## What would make this wrong
- Q3 consensus revisions reverse sharply upward (net positive) before Monday open
- A material positive catalyst (e.g., major contract win, regulatory approval) emerges over the weekend with credible sourcing

## What the checkers found
no check reported a problem. Every check that ran:
- [panel] 2 of 3 analysts run: sentiment, earnings
- [panel] event not run — relevance 0.10 below 0.35 — its evidence is stale, low-credibility, or carries nothing this analyst reads. 9 piece(s) of evidence it would have read went unexamined; that is a judgement about cost, not about their worth
- [panel] 2.267bps of deliberation not spent, against a 12bps round trip
- hedge menu empty: nothing placeable for 32.1h; 100% of risk carried as priced residual
- panel: 3 analysts, 13 distinct sources, independence 4.33 -> neutral at 0.83 after provenance discount
- [conflict] none: 3 analysts agreed on direction and size — but they ran in sequence, so agreement may be contagion rather than consensus
- [grounding] all 6 figure(s) resolve to a computed value or a cited fact
- [claim] no checkable claim found in the thesis (18 structured record(s) available)
- no order: final verdict no_trade with quantity 0
- [skills] 6 of 6 official-Skill calls answered for TSLAUSDT; 1 of 5 Skills reached (technical-analysis); 1 of 5 reachable
- [protocol] within the committed protocol; nothing changed (argus-paper-v1 v1: opens no position)

## What the risk layer did
nothing to narrow — no exposure proposed; nothing to narrow. The Constitution may only reduce, so an untouched decision is the model's own.

## The rules in force
`argus-paper-v1` v1, digest `sha256:a194a27e89568a98c9160ec798b2e7126e5df638c793cbb7a8dd6ae02ae5dc87`, committed 2026-09-13T05:10:23.077364+00:00 against ledger head `4c382231cf77e3d3` at 94 entries — so it was frozen before this decision existed.

## What happened next
not settled yet — it is held for 24 hours and graded against a price fetched in a later cycle, never one available when it was decided

## Integrity
market state `8031741403dbd943`, approved intent `3263878037676a4d`, chained to `102c8579725d571c`.

## Where each part came from
- the decision: data/paper_ledger.jsonl
- the checks: data/desk_notes.jsonl
- the risk ruling: data/risk_records.jsonl
- the rules: data/protocol_commitments.jsonl