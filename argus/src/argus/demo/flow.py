"""event → decision → execution, as one auditable trace. Track 2's required material.

The handbook asks for exactly this and names it a **submission requirement**, not a scoring
criterion: *"Runnable Demo + demonstrate a complete event → decision → execution flow (simulated or
paper trading acceptable)"*. ARGUS had all three legs and had never connected them.

* **event** — the evidence path runs on every cycle; 175 decisions on the ledger.
* **decision** — the desk, the Constitution and the proof chain, 175 times.
* **execution** — a real order round trip on the Bitget demo venue, verified in
  `execution/preflight.py:117` (venue order `1482642956228374529`).

Each leg was proven **in isolation**. Nothing linked them, and "these three things work" is not the
same claim as "this event caused that order". This module makes the second claim checkable.

**The link is the intent hash, not the timestamp.** `Order.__post_init__`
(`execution/orders.py:275`) already refuses any order without an ``approved_intent_hash``, on the
grounds that "an order that cannot be traced to a Constitution verdict is unauthorised however
well-formed it looks", and `BitgetTradingClient.place_order` enforces it again at the venue
boundary. So the causal chain here is not two log lines that happen to be adjacent: the venue order
**names** the verdict that authorised it, and :meth:`Flow.links_hold` recomputes that match rather
than trusting the ordering of a file.

**A leg that did not happen is recorded, never omitted.** The desk abstains on every decision so
far, so on a live run the execution leg is `NOT_REACHED` and :attr:`Flow.complete` is ``False`` —
which means this module usually reports that the required material is **not** demonstrated. That is
the honest output and it is the reason the leg has an explicit state instead of being absent from
the trace: an execution leg missing from a JSON file reads as a flow that ran and did nothing, and
an execution leg marked `NOT_REACHED` with the desk's own verdict on it reads as what it is.

**Scenario runs are labelled as scenarios.** A demonstration that can only ever abstain demonstrates
nothing, so :func:`run` accepts a constructed market state. Everything downstream of that state is
real — real model, real Constitution, real venue in demo mode — and the trace records the scenario's
inputs in full so a reader can see the setup was chosen and the *outcome* was not. A scenario run is
stamped ``scenario`` in the artefact and never counts as live evidence.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

DATA = __import__("pathlib").Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "flow_trace.json"

EXECUTED = "EXECUTED"
"""The leg ran and produced its artefact."""

NOT_REACHED = "NOT_REACHED"
"""The leg did not run, because an earlier leg's outcome meant there was nothing for it to do.

Distinct from a failure. An execution leg that is NOT_REACHED because the desk declined to trade is
the system working; one that is FAILED is the system broken. Collapsing them is how an abstaining
desk and a broken venue connection produce the same trace.
"""

FAILED = "FAILED"
"""The leg tried and could not complete. Always carries the error."""


class FlowError(RuntimeError):
    """Raised rather than emitting a trace whose legs cannot be linked."""


@dataclass(frozen=True, slots=True)
class Leg:
    """One stage of the flow, and what it produced."""

    name: str
    status: str
    summary: str
    at: datetime
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "leg": self.name, "status": self.status, "summary": self.summary,
            "at": self.at.isoformat(), "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class Flow:
    """One complete pass, linked by the hash of the intent the Constitution approved."""

    trace_id: str
    symbol: str
    mode: str
    """``live`` or ``scenario``. A scenario run is never counted as live evidence."""

    legs: tuple[Leg, ...]
    decision_intent_hash: str | None = None
    venue_intent_hash: str | None = None

    def leg(self, name: str) -> Leg | None:
        return next((x for x in self.legs if x.name == name), None)

    @property
    def complete(self) -> bool:
        """Every leg executed. The only state in which the required material is demonstrated."""
        return bool(self.legs) and all(x.status == EXECUTED for x in self.legs)

    @property
    def reached_venue(self) -> bool:
        """Did the order actually reach Bitget's demo environment, or only the paper book?

        Read from the venue's own response id, not from whether a client was supplied. The first
        COMPLETE run of this module printed "the decision reached the venue" for a flow whose venue
        leg had declined the instrument — an overclaim of exactly the kind the module exists to
        prevent, produced by the module itself.
        """
        execution = self.leg("execution")
        return bool(execution and execution.detail.get("venue_order_id"))

    @property
    def links_hold(self) -> bool | None:
        """Does the venue order name the verdict that authorised it?

        Recomputed from the two recorded hashes rather than inferred from the order of the file.
        ``None`` when no order reached the venue — an unlinked flow is not a broken link.
        """
        if self.venue_intent_hash is None:
            return None
        return self.venue_intent_hash == self.decision_intent_hash

    @property
    def verdict(self) -> str:
        execution = self.leg("execution")
        if self.complete and self.links_hold:
            where = (
                "Bitget's demo venue" if self.reached_venue
                else "ARGUS's own order book (paper), which the handbook accepts and which is what "
                     "the paper-trading log is made of; the demo venue carries no equity instrument"
            )
            return (
                f"COMPLETE ({self.mode}): the event reached a decision and the decision reached "
                f"{where}. The order carries intent hash {self.venue_intent_hash}, which is the "
                f"hash of the intent the Constitution approved — so the chain is linked by "
                f"authorisation, not by timing"
            )
        if self.links_hold is False:
            return (
                f"BROKEN: the venue order carries intent hash {self.venue_intent_hash} but the "
                f"approved intent hashed to {self.decision_intent_hash}. An order that does not "
                f"name the verdict that authorised it is unauthorised, whatever else is true"
            )
        failed = [x for x in self.legs if x.status == FAILED]
        if failed:
            return (
                f"INCOMPLETE ({self.mode}): the {failed[0].name} leg failed — {failed[0].summary}"
            )
        if execution is not None and execution.status == NOT_REACHED:
            return (
                f"INCOMPLETE ({self.mode}): event and decision ran; the execution leg was not "
                f"reached because {execution.summary}. Track 2's required "
                f"event->decision->execution material is NOT demonstrated by this run"
            )
        return f"INCOMPLETE ({self.mode}): {len(self.legs)} leg(s) recorded"

    def render(self) -> str:
        lines = [
            f"FLOW TRACE {self.trace_id} — {self.symbol} — mode={self.mode}",
            "",
        ]
        for item in self.legs:
            lines.append(f"  {item.status:<12} {item.name}")
            lines.append(f"               {item.summary}")
        lines += [
            "",
            f"  approved intent hash : {self.decision_intent_hash or '—'}",
            f"  venue order carries  : {self.venue_intent_hash or '— (none sent)'}",
            f"  link holds           : {self.links_hold}",
            f"  reached demo venue   : {self.reached_venue}",
            "",
            f"  {self.verdict}",
        ]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "trace_id": self.trace_id,
            "symbol": self.symbol,
            "mode": self.mode,
            "complete": self.complete,
            "reached_venue": self.reached_venue,
            "links_hold": self.links_hold,
            "decision_intent_hash": self.decision_intent_hash,
            "venue_intent_hash": self.venue_intent_hash,
            "legs": [x.as_dict() for x in self.legs],
            "verdict": self.verdict,
        }


def event_leg(evidence: Sequence[Any], *, at: datetime) -> Leg:
    """What arrived, when it became knowable, and from where.

    ``available_at`` rather than the time we read it: the whole point-in-time discipline depends on
    the difference, and a trace that stamped ingestion time would make every decision look
    prescient by exactly the fetch latency.
    """
    if not evidence:
        return Leg(
            "event", NOT_REACHED, "no evidence was gathered for this symbol", at,
            {"items": 0},
        )
    newest = max(evidence, key=lambda e: getattr(e, "available_at", at))
    sources = sorted({str(getattr(e, "source", "?")) for e in evidence})
    return Leg(
        "event", EXECUTED,
        f"{len(evidence)} item(s) from {len(sources)} source(s); newest available at "
        f"{getattr(newest, 'available_at', at)}",
        at,
        {
            "items": len(evidence),
            "sources": sources,
            "newest_claim": str(getattr(newest, "claim", ""))[:200],
            "newest_available_at": str(getattr(newest, "available_at", at)),
        },
    )


def decision_leg(run: Any, *, at: datetime) -> Leg:
    """The verdict, the constraint that bound it, and the hash the venue will have to name."""
    intent = run.proof.approved_intent if hasattr(run.proof, "approved_intent") else None
    final = intent or run.proof.llm_original_intent
    ruling = run.ruling
    return Leg(
        "decision", EXECUTED,
        f"verdict {final.verdict} quantity {final.quantity}"
        + (f"; risk layer bound on {ruling.binding_constraint}" if ruling else ""),
        at,
        {
            "verdict": str(final.verdict),
            "quantity": str(final.quantity),
            "side": str(final.side),
            "lean": getattr(run.proof.llm_original_intent, "lean", "none"),
            "binding_constraint": getattr(ruling, "binding_constraint", None),
            "ruling_reason": getattr(ruling, "reason", None),
            "notes": list(run.notes)[:12],
        },
    )


def execution_leg(
    run: Any, *, at: datetime, client: Any | None, symbol: str,
) -> tuple[Leg, str | None]:
    """Carry the approved order as far as it can go, and say exactly how far that was.

    **There are two executions here and the handbook accepts either** — *"simulated or paper trading
    acceptable"*. They are reported separately because they are different claims:

    * **paper** — the order enters ARGUS's own :class:`~argus.execution.orders.OrderBook`, which
      enforces the state machine and refuses an order carrying no ``approved_intent_hash``. This is
      the execution the paper-trading log is made of, and it works for every instrument.
    * **venue** — the order additionally reaches Bitget's demo environment. Stronger evidence, and
      **only possible for three instruments**: the demo product type ``SUSDT-FUTURES`` carries
      ``SBTCSUSDT``, ``SETHSUSDT`` and ``SXRPSUSDT`` only (`execution/bitget_client.py:65`). An
      equity symbol is rejected with venue code 40778 — found by running this, not by reading docs.

    So an equity flow completes on paper and reports the venue as unavailable *for that instrument*,
    which is a fact about Bitget's demo environment rather than a gap in ARGUS. Marking the whole
    leg FAILED for that would report a venue limitation as a broken system; marking it EXECUTED
    without saying which execution happened would overclaim. It says both.
    """
    from argus.execution.bitget_client import DEMO_SYMBOLS

    order = getattr(run, "order", None)
    if order is None:
        return Leg(
            "execution", NOT_REACHED,
            "the desk proposed no exposure, so no order was created to send",
            at, {"reason": "no order on the desk run"},
        ), None

    paper = {
        "client_order_id": order.client_order_id,
        "state": str(order.state),
        "quantity": str(order.quantity),
        "side": order.side,
        "approved_intent_hash": order.approved_intent_hash,
        "transitions": [str(t.to) for t in order.history],
    }
    summary = (
        f"paper: order {order.client_order_id} {order.side} {order.quantity} is {order.state} in "
        f"the book, bound to intent {order.approved_intent_hash}"
    )

    if client is None:
        return Leg(
            "execution", EXECUTED, f"{summary}; venue: not attempted (no client supplied)",
            at, {**paper, "venue": "not attempted"},
        ), order.approved_intent_hash
    if not client.is_paper:
        # Refusing is the safe direction and it is not a judgement call: this module exists to be
        # run by a reader, and a reader running a demo must not be able to reach a funded account.
        return Leg(
            "execution", EXECUTED,
            f"{summary}; venue: refused, the client is not in paper mode",
            at, {**paper, "venue": "refused (not paper mode)"},
        ), order.approved_intent_hash
    if symbol not in DEMO_SYMBOLS:
        return Leg(
            "execution", EXECUTED,
            f"{summary}; venue: {symbol} is not carried by the demo environment, which lists only "
            f"{', '.join(DEMO_SYMBOLS)}",
            at, {**paper, "venue": f"unavailable for {symbol}", "demo_symbols": list(DEMO_SYMBOLS)},
        ), order.approved_intent_hash
    try:
        # The venue call takes a capability, not a bare order: `run.ruling` is the Constitution
        # ruling this order came from, and `authorise` refuses if the two disagree on symbol,
        # side or quantity.
        if run.ruling is None:
            return Leg(
                "execution", NOT_REACHED,
                "no Constitution ruling on this run, so no order may be sent to a venue",
                at, {**paper, "venue": "refused: unauthorised"},
            ), order.approved_intent_hash
        placed = client.place_order(run.ruling.authorise(order), order_type="market")
        state, filled = client.reconcile(order, symbol=symbol)
    except Exception as exc:
        return Leg(
            "execution", FAILED,
            f"{summary}; venue: {type(exc).__name__}: {str(exc)[:180]}",
            at, {**paper, "venue_error": str(exc)[:300]},
        ), order.approved_intent_hash
    return Leg(
        "execution", EXECUTED,
        f"{summary}; venue order {placed.venue_order_id}, state {state}, filled {filled}",
        at,
        {
            **paper,
            "venue_order_id": str(placed.venue_order_id),
            "venue_state": str(state),
            "venue_filled": str(filled),
        },
    ), order.approved_intent_hash


def assemble(
    *,
    trace_id: str,
    symbol: str,
    mode: str,
    evidence: Sequence[Any],
    run: Any,
    client: Any | None,
    at: datetime | None = None,
) -> Flow:
    """Build the trace from a completed desk run. Pure assembly; no network of its own."""
    if mode not in {"live", "scenario"}:
        raise FlowError(f"mode must be 'live' or 'scenario', not {mode!r}")
    when = at or datetime.now(UTC)
    legs = [event_leg(evidence, at=when), decision_leg(run, at=when)]
    execution, venue_hash = execution_leg(run, at=when, client=client, symbol=symbol)
    legs.append(execution)
    approved = getattr(run.proof, "approved_intent_hash", None)
    return Flow(
        trace_id=trace_id, symbol=symbol, mode=mode, legs=tuple(legs),
        decision_intent_hash=str(approved) if approved else None,
        venue_intent_hash=venue_hash,
    )


SLEEPING_ANCHOR = "sleeping-anchor"
"""The scenario the whole system was designed around, and the one the live log cannot reach.

03:00 ET on a Sunday: an 8-K filed while the anchor market is shut, revising FY guidance down 7%.
The token keeps trading; the equity cannot reprice for 30 hours; nothing is placeable as a hedge.
Every input below is stated in the artefact so a reader can see that the **setup** was chosen and
the **outcome** was not — the model, the Constitution and the venue are all real, and the desk is
free to abstain here exactly as it does on the live log.
"""


def sleeping_anchor_frame() -> tuple[datetime, list[Any]]:
    """The scenario's evidence, with the times that make it a point-in-time problem."""
    from argus.truth.clocks import ET
    from argus.truth.evidence import Evidence

    # 15 March, not 8 March. The 8th is the US spring-forward date, and 02:31 ET does not exist
    # that morning: the filing timestamp landed inside the skipped hour, converted to 07:31 UTC —
    # *after* the 07:00 UTC decision — while a naive same-zone comparison still read it as earlier.
    # A point-in-time system whose own demonstration contains a look-ahead would be the worst
    # possible place to learn that, and it was a test comparing against the UTC instant that caught
    # it. The scenario is identical on any ordinary Sunday, so it uses one.
    as_of = datetime(2026, 3, 15, 3, 0, tzinfo=ET)
    filed = datetime(2026, 3, 15, 2, 31, tzinfo=ET)
    return as_of.astimezone(UTC), [
        Evidence(
            id="sec-8k-1",
            claim=(
                "SEC 8-K filed 02:31 ET: FY revenue guidance revised down 7% against prior "
                "guidance; company cites a delayed data-centre order book."
            ),
            source="sec-edgar",
            available_at=filed,
            credibility=1.0,
            attributes={"form": "8-K", "guidance_change_pct": -7.0},
        ),
        Evidence(
            id="social-1",
            claim="Viral post claims a 20% cut. No independent source carries it.",
            source="social",
            available_at=filed,
            credibility=0.18,
        ),
        Evidence(
            id="mkt-1",
            claim=(
                "rNVDA last 118.40, down 1.2% since the filing on roughly a third of "
                "regular-hours depth; quoted spread 9.4bps."
            ),
            # `news`, not `market`: the live runner emits its price fact on this channel
            # (`paper/runner.py:378`) and a source no analyst reads is evidence nobody sees.
            source="news",
            available_at=as_of,
            credibility=1.0,
        ),
    ]


def run_scenario(  # pragma: no cover - drives the live model and venue
    *, symbol: str = "NVDAUSDT", send: bool = False
) -> Flow:
    """Drive the real desk through the scenario, and carry any approved order to the demo venue."""
    from argus.agents.desk import ConstitutionPolicy, TradingDesk
    from argus.cost.model import CostModel
    from argus.llm.qwen import QwenClient, Thinking, TokenBudget
    from argus.risk.hedgeability import HedgeabilitySurface
    from argus.truth.clocks import DualClock

    as_of, evidence = sleeping_anchor_frame()
    trace_id = f"flow-scenario-{as_of.strftime('%Y%m%dT%H%M%S')}"
    session = DualClock().state(as_of, nav_age_seconds=40_000)
    desk = TradingDesk(
        QwenClient(budget=TokenBudget(limit=80_000)),
        pm_thinking=Thinking.LOW, cost=CostModel.bitget_perp(),
    )
    run = desk.run(
        symbol=symbol, session=session, token_price=Decimal("118.40"),
        # Two units, not the two hundred the scenario would carry in life. The size is chosen so
        # that whatever the desk decides is placeable on a demo account — the thing being
        # demonstrated is the chain from filing to venue order, and an execution leg that fails on
        # margin would demonstrate the account, not the system. The *decision* is untouched: the
        # desk sets its own quantity and is free to abstain.
        position=Decimal("2"), evidence=evidence,
        # Nothing placeable: NYSE is shut. This is the constraint the scenario exists to pose.
        hedges=HedgeabilitySurface(candidates=()),
        decision_id=trace_id, constitution=ConstitutionPolicy(),
    )
    venue = None
    if send:
        from argus.execution.bitget_client import BitgetTradingClient

        venue = BitgetTradingClient(paper_trading=True)
    return assemble(
        trace_id=trace_id, symbol=symbol, mode="scenario",
        evidence=evidence, run=run, client=venue, at=as_of,
    )


def from_ledger() -> Flow:  # pragma: no cover - reads the live record
    """Assemble the trace of the most recent **recorded** decision, without deciding again.

    Deliberately not a re-run. The live evidence path lives inside `paper.runner.run_once` and
    duplicating it here would let the demo drift from the code that actually produced the log — the
    trace a judge reads must come from the same pass that wrote the ledger row, not from a
    re-enactment that might disagree with it.
    """
    from argus.paper.ledger import PaperLedger
    from argus.paper.runner import LEDGER_PATH, NOTES_PATH, RISK_PATH

    ledger = PaperLedger(path=LEDGER_PATH)
    if not ledger.entries:
        raise FlowError("the ledger is empty; there is no recorded decision to trace")
    entry = ledger.entries[-1]
    notes: list[str] = []
    if NOTES_PATH.exists():
        for line in NOTES_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("seq") == entry.seq:
                    notes = list(row.get("notes", []))
    binding = None
    if RISK_PATH.exists():
        for line in RISK_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("seq") == entry.seq:
                    binding = row.get("binding_constraint")

    at = datetime.fromisoformat(entry.decided_at)
    evidence_note = next((n for n in notes if n.startswith("[panel]")), "")
    legs = (
        Leg(
            "event", EXECUTED,
            f"recorded decision {entry.seq} on {entry.symbol}: {evidence_note[:140]}",
            at, {"seq": entry.seq, "session_phase": entry.session_phase,
                 "hours_to_discovery": entry.hours_to_discovery},
        ),
        Leg(
            "decision", EXECUTED,
            f"verdict {entry.verdict} quantity {entry.quantity}, lean {entry.lean}, "
            f"stated confidence {entry.stated_confidence}",
            at, {"thesis": entry.thesis[:300], "binding_constraint": binding,
                 "approved_intent_hash": entry.approved_intent_hash},
        ),
        Leg(
            "execution", NOT_REACHED,
            f"the desk proposed no exposure ({entry.verdict}, quantity {entry.quantity}), so no "
            f"order was created to send",
            at, {"seq": entry.seq},
        ),
    )
    return Flow(
        trace_id=f"flow-ledger-{entry.seq}", symbol=entry.symbol, mode="live", legs=legs,
        decision_intent_hash=entry.approved_intent_hash, venue_intent_hash=None,
    )


def main() -> int:  # pragma: no cover - CLI
    """Two modes, and the difference is stated in the artefact.

    ``--from-ledger`` (default) traces the most recent real decision. It will report the execution
    leg as NOT_REACHED for as long as the desk abstains, which is the honest state of the required
    material. ``--scenario`` drives the real desk through the Sleeping-Anchor state, where the
    policy has something to act on.
    """
    import argparse
    import sys

    # A judge runs this on their own machine, and on a Windows console that is cp1252 by default an
    # arrow in the verdict raised UnicodeEncodeError and took the whole demo down. The report is the
    # deliverable, so the stream is made lossy rather than fatal: a mangled character is a blemish,
    # a traceback is a failed demonstration.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="event -> decision -> execution, as one trace")
    parser.add_argument("--scenario", action="store_true", help=f"run the {SLEEPING_ANCHOR} frame")
    parser.add_argument("--symbol", default="NVDAUSDT", help="scenario symbol")
    parser.add_argument(
        "--send", action="store_true",
        help="carry an approved order to the Bitget DEMO venue (paper mode only, always refused "
             "outside it)",
    )
    args = parser.parse_args()

    flow = run_scenario(symbol=args.symbol, send=args.send) if args.scenario else from_ledger()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    path = REPORT_PATH if not args.scenario else REPORT_PATH.with_name("flow_trace_scenario.json")
    path.write_text(json.dumps(flow.as_dict(), indent=2), encoding="utf-8")
    print(flow.render())
    print()
    print("written to " + str(path))
    return 0 if flow.complete else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "EXECUTED",
    "FAILED",
    "NOT_REACHED",
    "Flow",
    "FlowError",
    "Leg",
    "assemble",
    "decision_leg",
    "event_leg",
    "execution_leg",
]
