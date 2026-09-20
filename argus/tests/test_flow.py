"""Flow-trace tests — the module's job is to refuse to overclaim, so that is what is tested.

Track 2 names the event→decision→execution flow a **submission requirement**. A module that reports
that requirement as met is the last thing that should be generous, and this one has already
overclaimed once: its first COMPLETE run printed "the decision reached the venue" for a flow whose
venue leg had declined the instrument. Most of the tests below are about that class of mistake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from argus.demo.flow import (
    EXECUTED,
    FAILED,
    NOT_REACHED,
    Flow,
    FlowError,
    Leg,
    assemble,
    event_leg,
    execution_leg,
)

AT = datetime(2026, 3, 8, 7, 0, tzinfo=UTC)


@dataclass
class _Evidence:
    id: str
    claim: str
    source: str
    available_at: datetime


@dataclass
class _Order:
    client_order_id: str = "t-1"
    symbol: str = "NVDAUSDT"
    side: str = "SELL"
    quantity: Decimal = Decimal("2")
    approved_intent_hash: str = "abc123"
    state: str = "submitted"
    history: list[Any] = field(default_factory=list)


@dataclass
class _Intent:
    verdict: str = "reduce"
    quantity: Decimal = Decimal("2")
    side: str = "sell"
    lean: str = "down"


@dataclass
class _Proof:
    llm_original_intent: _Intent = field(default_factory=_Intent)
    approved_intent: _Intent = field(default_factory=_Intent)
    approved_intent_hash: str = "abc123"


@dataclass
class _Ruling:
    binding_constraint: str = "none"
    reason: str = "within every configured limit"


@dataclass
class _Run:
    proof: _Proof = field(default_factory=_Proof)
    ruling: _Ruling | None = field(default_factory=_Ruling)
    order: _Order | None = field(default_factory=_Order)
    notes: list[str] = field(default_factory=list)


class _Client:
    def __init__(self, *, paper: bool = True, venue_id: str = "v-99", raises: bool = False) -> None:
        self.is_paper = paper
        self._venue_id = venue_id
        self._raises = raises

    def place_order(self, order: Any, **_: Any) -> Any:
        if self._raises:
            raise RuntimeError("HTTP 400, venue code 40778")
        return type("Placed", (), {"venue_order_id": self._venue_id})()

    def reconcile(self, order: Any, **_: Any) -> tuple[str, Decimal]:
        return "filled", Decimal("2")


def _evidence() -> list[_Evidence]:
    return [
        _Evidence("sec-1", "8-K: FY guidance cut 7%", "sec-edgar", AT),
        _Evidence("news-1", "price line", "news", AT),
    ]


class TestCompleteMeansCompleteAndNothingLess:
    def test_an_abstention_is_incomplete_and_says_the_requirement_is_unmet(self) -> None:
        """The state of the live log, and it must read as a gap rather than as a quiet success."""
        run = _Run(order=None)
        flow = assemble(
            trace_id="t", symbol="NVDAUSDT", mode="live",
            evidence=_evidence(), run=run, client=None, at=AT,
        )
        assert not flow.complete
        assert "NOT demonstrated" in flow.verdict

    def test_a_failed_execution_is_incomplete(self) -> None:
        flow = assemble(
            trace_id="t", symbol="SBTCSUSDT", mode="scenario", evidence=_evidence(),
            run=_Run(order=_Order(symbol="SBTCSUSDT")), client=_Client(raises=True), at=AT,
        )
        assert not flow.complete
        assert "leg failed" in flow.verdict

    def test_a_paper_only_flow_is_complete_but_does_not_claim_the_venue(self) -> None:
        """The overclaim that actually happened. An equity order reaches the paper book and the
        demo venue carries no equity instrument; saying "reached the venue" is false."""
        flow = assemble(
            trace_id="t", symbol="NVDAUSDT", mode="scenario", evidence=_evidence(),
            run=_Run(), client=_Client(), at=AT,
        )
        assert flow.complete
        assert flow.reached_venue is False
        assert "Bitget's demo venue" not in flow.verdict
        assert "paper" in flow.verdict

    def test_a_venue_flow_says_so(self) -> None:
        flow = assemble(
            trace_id="t", symbol="SBTCSUSDT", mode="scenario", evidence=_evidence(),
            run=_Run(order=_Order(symbol="SBTCSUSDT")), client=_Client(), at=AT,
        )
        assert flow.reached_venue is True
        assert "Bitget's demo venue" in flow.verdict

    def test_reached_venue_reads_the_venue_response_not_the_client(self) -> None:
        """Supplying a client is not evidence the venue accepted anything."""
        flow = assemble(
            trace_id="t", symbol="NVDAUSDT", mode="scenario", evidence=_evidence(),
            run=_Run(), client=_Client(), at=AT,
        )
        assert flow.leg("execution") is not None
        assert flow.leg("execution").detail.get("venue_order_id") is None  # type: ignore[union-attr]
        assert flow.reached_venue is False


class TestTheLinkIsAuthorisationNotTiming:
    def test_a_matching_hash_links(self) -> None:
        flow = assemble(
            trace_id="t", symbol="NVDAUSDT", mode="scenario", evidence=_evidence(),
            run=_Run(), client=_Client(), at=AT,
        )
        assert flow.links_hold is True
        assert "linked by authorisation, not by timing" in flow.verdict

    def test_an_order_naming_a_different_verdict_is_reported_as_broken(self) -> None:
        """The failure mode a correlation id cannot catch: an id can be copied onto any order, but
        the intent hash is the hash of the approved intent's own content."""
        run = _Run(order=_Order(approved_intent_hash="forged"))
        flow = assemble(
            trace_id="t", symbol="NVDAUSDT", mode="scenario", evidence=_evidence(),
            run=run, client=_Client(), at=AT,
        )
        assert flow.links_hold is False
        assert "BROKEN" in flow.verdict
        assert "unauthorised" in flow.verdict

    def test_no_order_means_no_link_rather_than_a_broken_one(self) -> None:
        flow = assemble(
            trace_id="t", symbol="NVDAUSDT", mode="live", evidence=_evidence(),
            run=_Run(order=None), client=None, at=AT,
        )
        assert flow.links_hold is None
        assert "BROKEN" not in flow.verdict


class TestALegThatDidNotHappenIsRecorded:
    def test_no_order_gives_not_reached_not_absence(self) -> None:
        leg, hash_ = execution_leg(_Run(order=None), at=AT, client=None, symbol="NVDAUSDT")
        assert leg.status == NOT_REACHED
        assert hash_ is None
        assert "proposed no exposure" in leg.summary

    def test_not_reached_and_failed_are_different_states(self) -> None:
        """An abstaining desk and a broken venue must never produce the same trace."""
        declined, _ = execution_leg(_Run(order=None), at=AT, client=None, symbol="NVDAUSDT")
        broken, _ = execution_leg(
            _Run(order=_Order(symbol="SBTCSUSDT")), at=AT,
            client=_Client(raises=True), symbol="SBTCSUSDT",
        )
        assert declined.status == NOT_REACHED
        assert broken.status == FAILED
        assert declined.status != broken.status

    def test_every_leg_appears_in_the_trace_even_when_it_did_nothing(self) -> None:
        flow = assemble(
            trace_id="t", symbol="NVDAUSDT", mode="live", evidence=(), run=_Run(order=None),
            client=None, at=AT,
        )
        assert [x.name for x in flow.legs] == ["event", "decision", "execution"]

    def test_a_live_venue_client_is_refused(self) -> None:
        """A reader running a demo must not be able to reach a funded account."""
        leg, _ = execution_leg(
            _Run(), at=AT, client=_Client(paper=False), symbol="SBTCSUSDT",
        )
        assert "not in paper mode" in leg.summary
        assert leg.detail["venue"] == "refused (not paper mode)"


class TestTheEventLeg:
    def test_it_reports_availability_not_ingestion(self) -> None:
        """Stamping fetch time would make every decision look prescient by the fetch latency."""
        leg = event_leg(_evidence(), at=AT)
        assert leg.status == EXECUTED
        assert str(AT.date()) in leg.detail["newest_available_at"]

    def test_no_evidence_is_not_reached(self) -> None:
        assert event_leg([], at=AT).status == NOT_REACHED

    def test_it_counts_distinct_sources(self) -> None:
        leg = event_leg(_evidence(), at=AT)
        assert leg.detail["sources"] == ["news", "sec-edgar"]


class TestModesAreLabelled:
    def test_a_scenario_is_stamped_as_one(self) -> None:
        flow = assemble(
            trace_id="t", symbol="NVDAUSDT", mode="scenario", evidence=_evidence(),
            run=_Run(), client=None, at=AT,
        )
        assert flow.mode == "scenario"
        assert "(scenario)" in flow.verdict
        assert flow.as_dict()["mode"] == "scenario"

    def test_an_unknown_mode_is_refused(self) -> None:
        """A trace with no stated provenance is worse than no trace."""
        with pytest.raises(FlowError, match="must be 'live' or 'scenario'"):
            assemble(
                trace_id="t", symbol="X", mode="demo", evidence=(), run=_Run(),
                client=None, at=AT,
            )


class TestTheScenarioEvidenceIsRoutable:
    """The scenario abstained for the wrong reason once: its 8-K carried ``source="sec"``, which no
    analyst reads, so the filing reached nobody and the desk decided on a social post alone."""

    def test_every_scenario_source_reaches_an_analyst(self) -> None:
        from argus.agents.selection import SOURCES
        from argus.demo.flow import sleeping_anchor_frame

        known = set().union(*SOURCES.values())
        _, evidence = sleeping_anchor_frame()
        orphans = [e.source for e in evidence if e.source not in known]
        assert not orphans, f"scenario evidence nobody reads: {orphans}"

    def test_the_filing_is_on_the_event_analysts_channel(self) -> None:
        from argus.agents.selection import SOURCES
        from argus.demo.flow import sleeping_anchor_frame

        _, evidence = sleeping_anchor_frame()
        filing = next(e for e in evidence if "8-K" in e.claim)
        assert filing.source in SOURCES["event"]

    def test_the_filing_is_available_before_the_decision(self) -> None:
        from argus.demo.flow import sleeping_anchor_frame

        as_of, evidence = sleeping_anchor_frame()
        filing = next(e for e in evidence if "8-K" in e.claim)
        assert filing.available_at <= as_of


class TestTheReport:
    def test_it_serialises(self) -> None:
        import json

        flow = assemble(
            trace_id="t", symbol="NVDAUSDT", mode="scenario", evidence=_evidence(),
            run=_Run(), client=_Client(), at=AT,
        )
        blob = json.loads(json.dumps(flow.as_dict()))
        assert blob["complete"] is True and blob["links_hold"] is True

    def test_the_rendered_trace_shows_both_hashes_for_the_reader_to_match(self) -> None:
        flow = assemble(
            trace_id="t", symbol="NVDAUSDT", mode="scenario", evidence=_evidence(),
            run=_Run(), client=_Client(), at=AT,
        )
        text = flow.render()
        assert "approved intent hash" in text and "venue order carries" in text
        assert "reached demo venue" in text

    def test_a_leg_lookup_for_something_absent_is_none(self) -> None:
        flow = Flow(trace_id="t", symbol="X", mode="live", legs=(
            Leg("event", EXECUTED, "s", AT),
        ))
        assert flow.leg("execution") is None
