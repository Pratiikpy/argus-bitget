# Decision 96 — NVDAUSDT — no_trade

**Decided** 2026-09-13T05:22:03.626841+00:00 during the weekend session, 32.13h from the next price discovery.
**Size** 0 BUY at 217.52; entry cost 0.000bps.
**Stated confidence** 0.82

## Why
Weekend session with 32+ hours to genuine price discovery, no hedge available, and no edge exceeding the 18.80 bps total hurdle. Technicals are bearish but oversold conditions on a 4h timeframe during illiquid weekend trading do not constitute a tradeable edge against the cost structure.

## What would make this wrong
- Price gaps up more than 2% at Monday open on a material catalyst not yet in evidence
- RSI(14) on 4h diverges positively with volume confirmation before next session

## What the checkers found
2 finding(s):
- [conflict:conviction] event (neutral) vs earnings (neutral), gap 0.47 — event dominates: confidence 0.82 against 0.35
- [conflict:conviction] earnings (neutral) vs cross_asset (neutral), gap 0.60 — cross_asset dominates: confidence 0.95 against 0.35

<details><summary>every check that ran</summary>

- [panel] 3 of 3 analysts run: event, sentiment, earnings
- causal chain: 7 links stated, gradable at the next price discovery
- hedge menu empty: nothing placeable for 32.1h; 100% of risk carried as priced residual
- panel: 4 analysts, 16 distinct sources, independence 4.00 -> neutral at 0.71 after provenance discount
- [conflict:conviction] event (neutral) vs earnings (neutral), gap 0.47 — event dominates: confidence 0.82 against 0.35
- [conflict:conviction] earnings (neutral) vs cross_asset (neutral), gap 0.60 — cross_asset dominates: confidence 0.95 against 0.35
- [grounding] all 3 figure(s) resolve to a computed value or a cited fact
- [claim] no checkable claim found in the thesis (19 structured record(s) available)
- no order: final verdict no_trade with quantity 0
- [skills] 6 of 6 official-Skill calls answered for NVDAUSDT; 1 of 5 Skills reached (technical-analysis); 1 of 5 reachable
- [protocol] within the committed protocol; nothing changed (argus-paper-v1 v1: opens no position)

</details>

## What the risk layer did
nothing to narrow — no exposure proposed; nothing to narrow. The Constitution may only reduce, so an untouched decision is the model's own.

## The rules in force
`argus-paper-v1` v1, digest `sha256:a194a27e89568a98c9160ec798b2e7126e5df638c793cbb7a8dd6ae02ae5dc87`, committed 2026-09-13T05:10:23.077364+00:00 against ledger head `4c382231cf77e3d3` at 94 entries — so it was frozen before this decision existed.

## What happened next
not settled yet — it is held for 24 hours and graded against a price fetched in a later cycle, never one available when it was decided

## Integrity
market state `2b74037228f6faf4`, approved intent `1701a37deeb76434`, chained to `28a9ffdff82727d4`.

## Where each part came from
- the decision: data/paper_ledger.jsonl
- the checks: data/desk_notes.jsonl
- the risk ruling: data/risk_records.jsonl
- the rules: data/protocol_commitments.jsonl