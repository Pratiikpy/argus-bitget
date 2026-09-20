# Decision 68 — TSLAUSDT — no_trade

**Decided** 2026-09-12T18:21:17.738098+00:00 during the weekend session, 43.15h from the next price discovery.
**Size** 0 BUY at 366.11; entry cost 0.000bps.
**Stated confidence** 0.92

## Why
Weekend session with 43 hours until genuine price discovery, no hedge available, mixed and non-actionable news flow, and the 24h move of +16bps is already inside the 18.80bps total hurdle. There is no edge to capture.

## What would make this wrong
- A material, high-credibility catalyst emerges before Monday open that creates a directional edge exceeding 18.80bps
- Token price gaps significantly on thin weekend volume creating an exploitable dislocation

## What the checkers found
2 finding(s):
- [grounding] 1 of 2 figure(s) do not resolve to anything the desk was given: +16bps
- [grounding] an unattributable number in a thesis is the easiest place for an unsupported fact to enter the record

<details><summary>every check that ran</summary>

- [panel] 1 of 3 analysts run: event
- [panel] sentiment not run — no evidence on its channels (social)
- [panel] earnings not run — no evidence on its channels (filing, transcript)
- [panel] 4.534bps of deliberation not spent, against a 12bps round trip
- [panel] nothing cleared its own cost; the strongest candidate was run anyway because it held evidence it could act on and a near miss is not a reason to look away
- causal chain: 1 links stated, gradable at the next price discovery
- hedge menu empty: nothing placeable for 43.1h; 100% of risk carried as priced residual
- panel: 2 analysts, 4 distinct sources, independence 2.00 -> neutral at 0.90 after provenance discount
- [conflict] none: 2 analysts agreed on direction and size — but they ran in sequence, so agreement may be contagion rather than consensus
- [grounding] 1 of 2 figure(s) do not resolve to anything the desk was given: +16bps
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
market state `ab1e120c4468a66a`, approved intent `e91da26427f86974`, chained to `af20bcd757a570a1`.

## Where each part came from
- the decision: data/paper_ledger.jsonl
- the checks: data/desk_notes.jsonl