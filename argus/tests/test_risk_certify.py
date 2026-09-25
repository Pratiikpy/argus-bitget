"""The independent certifier: agrees with every correct gate, and catches every broken one."""

from __future__ import annotations

import ast
import itertools
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from argus.decision.verdicts import Verdict
from argus.eval import risk_certify, risk_shadow
from argus.execution import guard
from argus.risk import certify as c
from argus.risk import circuit, modes

CERTIFY_SOURCE = Path(c.__file__)
PRE_S20 = risk_shadow.BASELINES / "guard_2026-09-25_pre_s20.py.txt"
NVDA_ROW = risk_shadow.variant_row("nvda")
NVDA = guard.Instrument.from_payload(NVDA_ROW)
RULES = c.VenueRules.from_row(NVDA_ROW)


def _certify(ruling: guard.Ruling, **order: object) -> c.Certificate:
    return c.certify_venue(c.VenueOrder(**order), RULES, c.observe_venue(ruling))  # type: ignore[arg-type]


class TestIndependence:
    def test_the_certifier_imports_none_of_the_gates_it_checks(self) -> None:
        """A certifier that imported the guard would be the self-check again, by another name."""
        tree = ast.parse(CERTIFY_SOURCE.read_text(encoding="utf-8"))
        imported = {node.module for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module}
        imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                     for alias in node.names}
        assert not {m for m in imported if m.startswith("argus")}, imported

    def test_the_circuit_thresholds_are_restated_not_imported_and_agree_today(self) -> None:
        assert c.TOTAL_DRAWDOWN_HALT == circuit.TOTAL_DRAWDOWN_HALT
        assert c.SESSION_DRAWDOWN_HALT == circuit.DAILY_DRAWDOWN_HALT
        assert c.LADDER_FROM == circuit.REDUCE_ONLY_DRAWDOWN
        assert c.SHOCK_SIGMA == circuit.SIGMA_SHOCK
        assert c.LOSING_STREAK == circuit.CONSECUTIVE_LOSS_HALT
        assert c.STALE_AFTER == circuit.STALE_EVIDENCE

    def test_the_specification_names_the_guards_gates_in_the_guards_order(self) -> None:
        assert tuple(spec.gate for spec in c.VENUE_SPEC) == guard.GUARD_GATES


class TestAgreementWithTheGuard:
    def test_every_probe_and_recorded_order_is_certified(self) -> None:
        report = risk_certify.guard_agreement(guard, risk_shadow.probe_intents(), {})
        assert report["probe"]["violated"] == 0, report["probe"]["examples_violated"]
        assert report["probe"]["orders"] == len(risk_shadow.PROBES)

    def test_a_slice_of_the_swept_space_is_certified_with_trace_and_rate_effects(self) -> None:
        sample = list(itertools.islice(risk_shadow.swept_intents(), 0, 190_944, 11))
        report = risk_certify.guard_agreement(guard, sample, {})["swept"]
        assert report["orders"] == len(sample) > 17_000
        assert report["violated"] == 0, report["examples_violated"]

    def test_a_refusal_is_localised_in_words(self) -> None:
        ruling = guard.validate(NVDA, quantity=Decimal("0.004"), reference_price=Decimal("200"))
        cert = _certify(ruling, quantity=Decimal("0.004"), reference_price=Decimal("200"))
        assert cert.certified
        assert cert.first_unmet is not None
        assert cert.first_unmet.startswith("quantity_floor (step 10 of 12): unmet precondition")
        assert "below the venue minimum 0.01" in cert.first_unmet

    def test_an_allowed_order_has_no_unmet_precondition(self) -> None:
        ruling = guard.validate(NVDA, quantity=Decimal("1"), reference_price=Decimal("200"))
        cert = _certify(ruling, quantity=Decimal("1"), reference_price=Decimal("200"))
        assert cert.certified and cert.first_unmet is None and cert.expected == "allowed 1"

    def test_a_dishonest_trace_is_caught_even_when_the_ruling_is_right(self) -> None:
        ruling = guard.validate(NVDA, quantity=Decimal("1"), reference_price=Decimal("200"))
        events = list(ruling.trace.events)
        events[3] = guard.TraceEvent("rate_window", "", guard.GateVerdict.PASS, "invented")
        forged = guard.Ruling(ruling.denial, ruling.reason, ruling.quantity, ruling.price,
                              ruling.adjusted, guard.PermissionCheckTrace(tuple(events)))
        cert = _certify(forged, quantity=Decimal("1"), reference_price=Decimal("200"))
        assert not cert.certified
        assert "rate_window (step 4)" in cert.violations[0]

    def test_an_exception_instead_of_a_ruling_is_a_violation(self) -> None:
        cert = c.certify_venue(c.VenueOrder(quantity=Decimal("NaN")), RULES,
                               c.raised(ArithmeticError("boom")))
        assert not cert.certified and "raised ArithmeticError" in cert.violations[0]


@pytest.fixture(scope="module")
def before() -> risk_shadow.Engine:
    if not PRE_S20.exists():
        pytest.skip("the frozen pre-S20 guard is not present")
    return risk_shadow.load_engine(PRE_S20, "guard before S20")


class TestTheGuardBeforeS20:
    """The certifier, pointed at the guard as it stood on the morning of 2026-09-25."""

    def test_it_flags_the_three_defects_and_nothing_on_the_swept_space(
        self, before: risk_shadow.Engine,
    ) -> None:
        flagged = set()
        for intent in risk_shadow.probe_intents():
            row = risk_shadow.rows_for(intent, {})
            cert = risk_certify.certify_run(risk_shadow.run_guard(before.module, intent, row),
                                            row)
            if not cert.certified:
                flagged.add(intent.ident)
        assert {"side neither buy nor sell", "empty side",
                "sub-tick limit price, no reference",
                "31-digit quantity just under two steps",
                "notional a hair under the floor", "notional a hair over the balance",
                "limit on the tick, band edge from a 31-digit reference"} <= flagged
        assert "exact odd step" not in flagged and "limit exactly on the band edge" not in flagged
        sample = list(itertools.islice(risk_shadow.swept_intents(), 0, 190_944, 97))
        swept = risk_certify.guard_agreement(before.module, sample, {})["swept"]
        assert swept["violated"] == 0


class TestTheRecordedChain:
    def _chain(self, **over: object) -> c.RecordedChain:
        base: dict[str, object] = {
            "seq": 1, "symbol": "COINUSDT", "proposed": Decimal("1"), "approved": Decimal("1"),
            "constitution_verdict": "trade", "protocol_changed": False,
            "venue_consulted": True, "venue_allowed": Decimal("1"), "venue_adjusted": False,
            "ledger_verdict": "trade", "ledger_quantity": Decimal("1"),
        }
        base.update(over)
        return c.RecordedChain(**base)  # type: ignore[arg-type]

    def test_a_consistent_chain_is_certified(self) -> None:
        assert c.certify_chain(self._chain()).certified

    def test_an_order_the_constitution_refused_is_caught_at_the_venue_gate(self) -> None:
        cert = c.certify_chain(self._chain(approved=Decimal("0"),
                                           constitution_verdict="human_review"))
        assert cert.status is c.Status.VIOLATED
        assert cert.first_unmet is not None and cert.first_unmet.startswith("venue_gate (step 3")

    def test_a_decision_without_a_risk_record_is_uncertifiable_not_passed(self) -> None:
        cert = c.certify_chain(self._chain(proposed=None, approved=None))
        assert cert.status is c.Status.UNCERTIFIABLE and not cert.certified

    def test_an_order_booked_without_a_venue_ruling_is_caught(self) -> None:
        cert = c.certify_chain(self._chain(venue_consulted=False, venue_allowed=None))
        assert "no venue ruling was recorded" in cert.violations[0]

    def test_on_the_live_record_it_finds_exactly_the_rows_the_human_audit_voided(self) -> None:
        from argus.paper.corrections import VOIDED_SEQS

        report = risk_certify.chain_report()
        assert report["caught_every_voided_row"]
        assert set(report["flagged_seqs"]) == set(VOIDED_SEQS)
        assert report["certified"] > 500


class TestTheOtherGates:
    def test_the_circuit_breaker_and_its_transitions_are_certified(self) -> None:
        report = risk_certify.tally(risk_certify.circuit_cases(circuit))
        assert report["violated"] == 0, report["examples_violated"]
        assert report["cases"] > 2_000

    def test_a_breaker_that_misses_a_rule_is_caught(self) -> None:
        state = circuit.BookState(equity=Decimal("90"), peak_equity=Decimal("100"),
                                  session_open_equity=Decimal("90"))
        honest = circuit.apply(Verdict.TRADE, state)
        forged = SimpleNamespace(
            activation=honest.activation, trips=honest.trips[:0],
            risk_multiplier=honest.risk_multiplier, original_verdict=honest.original_verdict,
            verdict=honest.verdict,
        )
        cert = c.certify_circuit(state, "trade", forged)
        assert not cert.certified
        assert any("total_drawdown must fire" in v for v in cert.violations)

    def test_sizing_the_throttle_and_the_modes_are_certified(self) -> None:
        from argus.risk import session_risk, sizing

        for cases in (risk_certify.sizing_cases(sizing),
                      risk_certify.throttle_cases(session_risk),
                      risk_certify.mode_cases(modes, guard)):
            report = risk_certify.tally(cases)
            assert report["violated"] == 0, report["examples_violated"]
            assert report["cases"] > 100

    def test_the_stepwise_recovery_rule(self) -> None:
        assert c.certify_transition("halted", "active", "reduce_only").certified
        assert not c.certify_transition("halted", "active", "active").certified


class TestMutationTesting:
    def test_every_mutant_is_caught_and_the_equivalent_one_survives(self) -> None:
        intents = [*risk_shadow.probe_intents()]
        report = risk_certify.mutation_report(intents, {}, guard)
        survivors = [h for h in report["hunts"] if h["outcome"] == "SURVIVED"]
        assert not survivors, survivors
        assert report["caught"] == report["mutants"] >= 50
        assert report["equivalent_mutants_survived_as_expected"]
        assert set(report["by_target"]) == {"guard", "circuit", "sizing", "session_risk", "modes"}

    def test_a_mutant_whose_target_has_moved_is_refused_not_skipped(self) -> None:
        stale = risk_certify.Mutant("X", "guard", "none", "nothing", "no such text", "")
        with pytest.raises(risk_certify.StaleMutant):
            risk_certify.mutate(risk_certify.source_of("guard"), stale)

    def test_every_mutant_still_applies_to_the_current_source(self) -> None:
        for mutant in risk_certify.MUTANTS:
            risk_certify.mutate(risk_certify.source_of(mutant.target), mutant)


def test_the_book_grid_meets_every_threshold_exactly() -> None:
    drawdowns = {c.BookFacts.read(SimpleNamespace(**book)).drawdown
                 for book, _ in risk_shadow.book_grid()}
    assert {c.TOTAL_DRAWDOWN_HALT, c.LADDER_FROM, c.LADDER_HALF} <= drawdowns
    sessions = {c.BookFacts.read(SimpleNamespace(**book)).session_drawdown
                for book, _ in risk_shadow.book_grid()}
    assert c.SESSION_DRAWDOWN_HALT in sessions
    ages = {book["evidence_age"] for book, _ in risk_shadow.book_grid()}
    assert ages == {timedelta(hours=6), timedelta(hours=6, seconds=1)}
