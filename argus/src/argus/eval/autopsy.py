"""Why has this desk never traded? Answered from the record, with the unreached gates named.

Track 2 is 50% quantitative and ARGUS has no Sharpe, no max drawdown and no win rate, because zero
positions have settled. Every report that touches this says so. None of them said **why**, and the
two explanations that got published were both wrong:

* *the fee* — refuted by the hurdle frontier, which measured a median absolute move of 137bps
  against an 18.8bps total hurdle. Moves clear the cost seven times over.
* *the confidence gate* — refuted here. `agents/desk.py:651` returns on ``quantity <= 0`` before
  reaching the 0.55 floor, so the floor **has never executed**. All 88 risk records read
  ``binding_constraint: none``. A constraint that never runs cannot be the binding one, and naming
  it as such pointed an iteration at loosening a gate that is not in the path.

**The distinction this module exists to make is between a gate that passed and a gate that never
ran.** They look identical in every counter that only tallies rejections: both contribute zero. But
"the confidence floor allowed 170 decisions through" and "the confidence floor was never evaluated"
support opposite conclusions, and only the second is true here. A funnel that reports survival rates
without reporting *reachability* will call an unreachable gate a permissive one every time.

**~~The measured cause is in the sampling, not the desk.~~ REFUTED 2026-09-14.** This module
published a third explanation, and it has now failed its own test:

> *"All 170 recorded decisions were taken in a ``weekend`` session, between 18 and 52 hours from
> price discovery… The desk has never been offered a session in which its own stated policy would
> trade. 'It refuses to trade' has therefore never been an observation about the desk — it is an
> observation about when we asked it."*

On 2026-09-14 the 13:30 UTC cycle ran **inside US regular hours with the anchor market open**. The
desk was offered exactly the session this module said it had never been offered, and it proposed
**nothing on all twelve instruments**. The falsifier named below fired on the first attempt.
:attr:`Autopsy.falsifier_has_fired` computes it from the record rather than leaving it to a reader.

**So three published explanations have now been wrong: the fee, the confidence gate, and the
sampling.** What the record actually shows, after the RTH session:

* **It is not the confidence floor.** All twelve RTH decisions state confidence between 0.58 and
  0.88 — every one *above* the 0.55 floor. The floor still never executes, because gate 1 returns
  first on a zero quantity.
* **It is not the fee.** `eval/hurdle.py` is unchanged and still right: a median absolute move of
  137bps against an 18.8bps hurdle, 95% of instants clearing it, and trading beating abstaining at
  **56.0% directional accuracy**.
* **It is the desk's own conviction about direction.** The RTH theses say so in their own words —
  *"the edge is not clearly above the 16.27bps total hurdle"*. Read against the frontier that is not
  a claim about cost, which is cleared seven times over; it is the desk declining to bet that it can
  call direction better than 56%.

**And whether that refusal is correct is still UNDEFINED, which is the honest end of this.**
`eval/shadow.py` has **0 scored calls** against a floor of 20: all 69 settled decisions declined to
lean. The twelve RTH decisions are the first that carry a directional lean (up/down, 0.38-0.62
confidence) and they settle 24 hours later. Until they score, "the desk correctly refuses to bet on
an edge it does not have" and "the desk is too timid" are both consistent with the record, and this
module will not choose between them.

**Prior art, read before this was written** (`research/architecture/gate-attribution.md`). Four
systems in the corpus record why an order was refused, and **none distinguishes a gate that allowed
from a gate that never ran**. `nautilus_trader`'s `OrderDeniedReason`
(`crates/model/src/events/order/denied_reason.rs:69`) is the best taxonomy available — 37 typed
variants carrying their own comparison operands — and records only the reason that fired.
`freqtrade` returns the first protection that locks and nothing about the rest
(`plugins/protections/iprotection.py:17-22`). `factorminer` carries one reason string with good
context (`evaluation/admission.py:21-85`).

The near miss is `vibe-trading`, which puts a ``checked_limits`` list in every decision envelope
(`agent/src/live/sdk_order_gate.py:398`) — and builds it as a **static literal** at `:283`, `:469`
and `:544` that is never mutated. An order denied by its first gate is recorded as having had all
twelve limits checked. The field reads like reachability evidence and is a constant, which is worse
than `freqtrade` recording nothing. Ours is derived by walking :data:`CHAIN` in source order and
subtracting what each gate took — the same short-circuit the policy performs — so reachability is
computed rather than declared.

**A second near miss, found 2026-09-15 by searching all 667 cloned repositories rather than the
four this section was written from** (`eval/corpusclaim.py`, `data/corpus_claims.json`).
`AlgoVaultLabs~crypto-quant-signal-mcp/src/lib/scorer-input-codes.ts:82` declares
``NOT_EVALUATED: 0`` and comments it *"Distinct from NEUTRAL"* — **the same distinction this module
makes, between a rule that ran and allowed and one that never ran at all.** It is applied to a
signal adjustment rather than to an order-gate chain, so the claim above stands as scoped; but the
idea is not ours alone, and saying so is cheaper than being told. The narrower claim — that no
system in the corpus reports it *for a gate chain* — is what the search actually supports.

**What would falsify the rewritten conclusion.** The sampling falsifier has already fired, so the
open question has moved: it is no longer *why* the desk abstains but *whether it should*. Two
observations would settle it, and both arrive on their own:

* **The twelve RTH leans settle.** If they score materially above the 56.0% break-even, the desk had
  a directional edge and declined to use it — the abstention was timidity and the conviction was
  miscalibrated. If they score at or below it, the refusal was correct.
* **A cycle in which the desk proposes exposure.** Gates 2-7 have still never executed, so their
  permissiveness remains unmeasured rather than demonstrated.

The original falsifier worked exactly as intended: it was written down before the evidence existed,
it was checked automatically every cycle, and when it fired the conclusion was rewritten rather than
defended. That is the only part of this module's history worth keeping.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

# The analysis lives in the risk layer (`risk/gate_autopsy.py`), where product code may run it;
# every name is re-exported so this module's readers and tests are unchanged.
from argus.risk.gate_autopsy import (
    CHAIN,
    FIRED,
    NOTHING_BOUND,
    PASSED,
    RECORD_SCHEMA,
    UNREACHED,
    Autopsy,
    AutopsyError,
    Gate,
    GateOutcome,
    SessionCoverage,
    examine,
)
from argus.truth.paths import DATA_DIR

DATA = DATA_DIR
REPORT_PATH = DATA / "abstention_autopsy.json"


def main() -> int:  # pragma: no cover - CLI
    from argus.paper.ledger import PaperLedger
    from argus.paper.runner import LEDGER_PATH, RISK_PATH
    from argus.truth.clocks import DualClock

    ledger = PaperLedger(path=LEDGER_PATH)
    records: list[dict[str, Any]] = []
    if RISK_PATH.exists():
        records = [
            json.loads(x) for x in RISK_PATH.read_text(encoding="utf-8").splitlines() if x.strip()
        ]
    state = DualClock().state(datetime.now(UTC), nav_age_seconds=60)
    report = examine(
        ledger.entries, records,
        hours_to_first_discovery=(
            0.0 if state.phase.has_price_discovery else state.hours_to_next_discovery
        ),
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    print(report.render())
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "CHAIN",
    "FIRED",
    "NOTHING_BOUND",
    "PASSED",
    "RECORD_SCHEMA",
    "UNREACHED",
    "Autopsy",
    "AutopsyError",
    "Gate",
    "GateOutcome",
    "SessionCoverage",
    "examine",
]
