# Decision 99 — MSFTUSDT — no_trade

**Decided** 2026-09-13T05:22:03.626841+00:00 during the weekend session, 32.13h from the next price discovery.
**Size** 0 BUY at 495.44; entry cost 0.000bps.
**Stated confidence** 0.92

## Why
Weekend session with anchor market asleep for 32+ hours, no MSFT-specific catalyst, conflicting technical signals netting to zero, and a total hurdle of 18.80 bps that cannot be cleared by any identifiable edge. The 24h change of +0.11% is well inside transaction costs.

## What would make this wrong
- A material MSFT-specific catalyst emerges before Monday's cash open (e.g., acquisition announcement, regulatory action, or major partnership disclosure)
- Token price gaps more than 50 bps from current level on thin weekend volume, creating a mean-reversion opportunity exceeding the 18.80 bps hurdle

## What the checkers found
no check reported a problem. Every check that ran:
- [panel] 3 of 3 analysts run: event, sentiment, earnings
- causal chain: 6 links stated, gradable at the next price discovery
- hedge menu empty: nothing placeable for 32.1h; 100% of risk carried as priced residual
- panel: 4 analysts, 17 distinct sources, independence 4.25 -> neutral at 0.86 after provenance discount
- [conflict] none: 4 analysts agreed on direction and size — but they ran in sequence, so agreement may be contagion rather than consensus
- [grounding] all 2 figure(s) resolve to a computed value or a cited fact
- [claim] no checkable claim found in the thesis (17 structured record(s) available)
- no order: final verdict no_trade with quantity 0
- [skills] 6 of 6 official-Skill calls answered for MSFTUSDT; 1 of 5 Skills reached (technical-analysis); 1 of 5 reachable
- [protocol] within the committed protocol; nothing changed (argus-paper-v1 v1: opens no position)

## What the risk layer did
nothing to narrow — no exposure proposed; nothing to narrow. The Constitution may only reduce, so an untouched decision is the model's own.

## The rules in force
`argus-paper-v1` v1, digest `sha256:a194a27e89568a98c9160ec798b2e7126e5df638c793cbb7a8dd6ae02ae5dc87`, committed 2026-09-13T05:10:23.077364+00:00 against ledger head `4c382231cf77e3d3` at 94 entries — so it was frozen before this decision existed.

## What happened next
not settled yet — it is held for 24 hours and graded against a price fetched in a later cycle, never one available when it was decided

## Integrity
market state `74e6ba64d6dc5991`, approved intent `4a85b5d8a9cd3003`, chained to `24e7943b58da19bd`.

## Where each part came from
- the decision: data/paper_ledger.jsonl
- the checks: data/desk_notes.jsonl
- the risk ruling: data/risk_records.jsonl
- the rules: data/protocol_commitments.jsonl