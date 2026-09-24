# Decision 91 — METAUSDT — no_trade

**Decided** 2026-09-13T03:22:02.799291+00:00 during the weekend session, 34.13h from the next price discovery.
**Size** 0 BUY at 648.98; entry cost 0.000bps.
**Stated confidence** 0.82

## Why
Weekend session with 34 hours until genuine price discovery, no placeable hedges, no discrete catalyst exceeding the 18.8 bps total hurdle, and internally inconsistent technical signals make any directional bet a negative-expectancy coin flip.

## What would make this wrong
- A material, credible news event (e.g., regulatory action, major acquisition announcement, or data breach) breaks before Monday open that would move META more than 19 bps
- Token price gaps significantly away from 648.98 on thin weekend volume, creating a mean-reversion opportunity exceeding the hurdle

## What the checkers found
no check reported a problem. Every check that ran:
- [panel] 3 of 3 analysts run: event, sentiment, earnings
- causal chain: 6 links stated, gradable at the next price discovery
- hedge menu empty: nothing placeable for 34.1h; 100% of risk carried as priced residual
- panel: 4 analysts, 19 distinct sources, independence 4.75 -> neutral at 0.83 after provenance discount
- [conflict] none: 4 analysts agreed on direction and size — but they ran in sequence, so agreement may be contagion rather than consensus
- [grounding] all 1 figure(s) resolve to a computed value or a cited fact
- [claim] 1 checkable claim(s) agree with the 18 structured record(s) behind them
- no order: final verdict no_trade with quantity 0
- [skills] 6 of 6 official-Skill calls answered for METAUSDT; 1 of 5 Skills reached (technical-analysis)

## What the risk layer did
nothing to narrow — no exposure proposed; nothing to narrow. The Constitution may only reduce, so an untouched decision is the model's own.

## The rules in force
— not recorded — — this decision predates the pre-registered protocol and is reported as ungoverned rather than as compliant

## What happened next
not settled yet — it is held for 24 hours and graded against a price fetched in a later cycle, never one available when it was decided

## Integrity
market state `507153d3df36c5d2`, approved intent `47463a9f62ec3d9a`, chained to `2331d6a2a6942484`.

## Where each part came from
- the decision: data/paper_ledger.jsonl
- the checks: data/desk_notes.jsonl
- the risk ruling: data/risk_records.jsonl