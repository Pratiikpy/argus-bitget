# Decision 95 — NVDAUSDT — no_trade

**Decided** 2026-09-13T05:18:56.259437+00:00 during the weekend session, 32.18h from the next price discovery.
**Size** 0 BUY at 217.54; entry cost 0.000bps.
**Stated confidence** 0.82

## Why
Weekend session with 32+ hours to genuine price discovery, no hedge menu available, and no catalyst with edge exceeding the 18.80 bps total hurdle. Technical signals are conflicting (oversold RSI vs. bearish trend), fundamentals are strong but already priced, and the panel consensus is neutral.

## What would make this wrong
— not recorded — (an abstention opens no exposure, so it needs no falsifier)

## What the checkers found
2 finding(s):
- [conflict:conviction] event (neutral) vs sentiment (neutral), gap 0.37 — event dominates: confidence 0.72 against 0.35
- [conflict:conviction] sentiment (neutral) vs cross_asset (neutral), gap 0.60 — cross_asset dominates: confidence 0.95 against 0.35

<details><summary>every check that ran</summary>

- [panel] 3 of 3 analysts run: event, sentiment, earnings
- hedge menu empty: nothing placeable for 32.2h; 100% of risk carried as priced residual
- panel: 4 analysts, 12 distinct sources, independence 3.00 -> neutral at 0.67 after provenance discount
- [conflict:conviction] event (neutral) vs sentiment (neutral), gap 0.37 — event dominates: confidence 0.72 against 0.35
- [conflict:conviction] sentiment (neutral) vs cross_asset (neutral), gap 0.60 — cross_asset dominates: confidence 0.95 against 0.35
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
market state `9c9a711ccd99ff51`, approved intent `b925d3c1bb3d5d73`, chained to `4c382231cf77e3d3`.

## Where each part came from
- the decision: data/paper_ledger.jsonl
- the checks: data/desk_notes.jsonl
- the risk ruling: data/risk_records.jsonl
- the rules: data/protocol_commitments.jsonl