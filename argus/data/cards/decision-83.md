# Decision 83 — NVDAUSDT — no_trade

**Decided** 2026-09-13T01:45:26.486373+00:00 during the weekend session, 35.74h from the next price discovery.
**Size** 0 BUY at 218.14; entry cost 0.000bps.
**Stated confidence** 0.89

## Why
Weekend session with 35.7 hours to genuine price discovery, no hedge menu, no NVDA-specific catalyst, and a 24h move of -3.5bps well inside the 18.80bps total hurdle. Technical bearishness (RSI 28.5, MACD death cross, price below all MAs) is real but already extended, and without institutional flow or a catalyst to act on, any position opened now faces pure weekend drift risk with no ability to manage it.

## What would make this wrong
- NVDA-specific material news (earnings pre-announcement, regulatory action, major contract win/loss) breaks before Monday open that would gap the token significantly
- Token price moves more than 50bps from 218.14 during the weekend session indicating genuine repricing rather than noise

## What the checkers found
2 finding(s):
- [grounding] 1 of 3 figure(s) do not resolve to anything the desk was given: -3.5bps
- [grounding] an unattributable number in a thesis is the easiest place for an unsupported fact to enter the record

<details><summary>every check that ran</summary>

- [panel] 2 of 3 analysts run: event, sentiment
- [panel] earnings not run — no evidence on its channels (filing, transcript)
- [panel] 2.267bps of deliberation not spent, against a 12bps round trip
- causal chain: 6 links stated, gradable at the next price discovery
- hedge menu empty: nothing placeable for 35.7h; 100% of risk carried as priced residual
- panel: 3 analysts, 8 distinct sources, independence 2.67 -> neutral at 0.89 after provenance discount
- [conflict] none: 3 analysts agreed on direction and size — but they ran in sequence, so agreement may be contagion rather than consensus
- [grounding] 1 of 3 figure(s) do not resolve to anything the desk was given: -3.5bps
- [grounding] an unattributable number in a thesis is the easiest place for an unsupported fact to enter the record
- [claim] no checkable claim found in the thesis (6 structured record(s) available)
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
market state `4c6adb61b3ec9832`, approved intent `380793a79d147ef3`, chained to `35fb3bb17b3c1bdb`.

## Where each part came from
- the decision: data/paper_ledger.jsonl
- the checks: data/desk_notes.jsonl
- the risk ruling: data/risk_records.jsonl