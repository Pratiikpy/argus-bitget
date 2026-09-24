# Decision 98 — AAPLUSDT — no_trade

**Decided** 2026-09-13T05:22:03.626841+00:00 during the weekend session, 32.13h from the next price discovery.
**Size** 0 BUY at 333.57; entry cost 0.000bps.
**Stated confidence** 0.82

## Why
Weekend session with 32+ hours until genuine price discovery, no hedge available, flat QoQ earnings with no beat/miss context, contradictory technical signals, and a total hurdle of 18.80 bps that no identifiable edge can clear. The correct decision is to stand aside.

## What would make this wrong
- A material catalyst (e.g., product announcement, regulatory action, or macro shock) emerges before Monday's open that creates a directional gap exceeding the 18.80 bps hurdle
- Consensus revisions shift decisively positive (e.g., >70% upward) ahead of the September 30 quarter-end

## What the checkers found
no check reported a problem. Every check that ran:
- [panel] 2 of 3 analysts run: sentiment, earnings
- [panel] event not run — relevance 0.10 below 0.35 — its evidence is stale, low-credibility, or carries nothing this analyst reads. 8 piece(s) of evidence it would have read went unexamined; that is a judgement about cost, not about their worth
- [panel] 2.267bps of deliberation not spent, against a 12bps round trip
- hedge menu empty: nothing placeable for 32.1h; 100% of risk carried as priced residual
- panel: 3 analysts, 12 distinct sources, independence 4.00 -> neutral at 0.83 after provenance discount
- [conflict] none: 3 analysts agreed on direction and size — but they ran in sequence, so agreement may be contagion rather than consensus
- [grounding] all 2 figure(s) resolve to a computed value or a cited fact
- [claim] no checkable claim found in the thesis (17 structured record(s) available)
- no order: final verdict no_trade with quantity 0
- [skills] 6 of 6 official-Skill calls answered for AAPLUSDT; 1 of 5 Skills reached (technical-analysis); 1 of 5 reachable
- [protocol] within the committed protocol; nothing changed (argus-paper-v1 v1: opens no position)

## What the risk layer did
nothing to narrow — no exposure proposed; nothing to narrow. The Constitution may only reduce, so an untouched decision is the model's own.

## The rules in force
`argus-paper-v1` v1, digest `sha256:a194a27e89568a98c9160ec798b2e7126e5df638c793cbb7a8dd6ae02ae5dc87`, committed 2026-09-13T05:10:23.077364+00:00 against ledger head `4c382231cf77e3d3` at 94 entries — so it was frozen before this decision existed.

## What happened next
not settled yet — it is held for 24 hours and graded against a price fetched in a later cycle, never one available when it was decided

## Integrity
market state `4dcd64ff81a26a00`, approved intent `f4035ada2e672754`, chained to `646ffb5b97a0d18b`.

## Where each part came from
- the decision: data/paper_ledger.jsonl
- the checks: data/desk_notes.jsonl
- the risk ruling: data/risk_records.jsonl
- the rules: data/protocol_commitments.jsonl