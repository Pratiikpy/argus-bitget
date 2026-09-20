"""Autopsy tests — the whole value is one distinction, so most of these defend it.

A gate that evaluated and allowed, and a gate that never executed, both contribute zero to a
rejection counter. Every system in `research/architecture/gate-attribution.md` collapses them; the
closest prior art (`vibe-trading`) records a hardcoded list of "checked" limits that is identical
whether the first gate refused or the twelfth did. These tests exist so ours cannot drift into that.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from argus.eval.autopsy import (
    CHAIN,
    FIRED,
    PASSED,
    RECORD_SCHEMA,
    UNREACHED,
    AutopsyError,
    examine,
)


@dataclass(frozen=True)
class _Entry:
    session_phase: str = "weekend"
    hours_to_discovery: float = 30.0
    stated_confidence: float = 0.88
    is_abstention: bool = True


def _risk(constraint: str) -> dict[str, object]:
    """One persisted risk record.

    **Stamped with the schema, because the vocabulary changed and silence would be worse than a
    failure.** Schema 1 wrote ``"none"`` for both gate 1 returning and the terminal all-clear, so
    these fixtures had to invent their own tokens — ``"nothing_bound"`` and ``"nothing"`` — neither
    of which the Constitution ever emits. Those tests passed only because the funnel silently
    ignored values it did not recognise, which is the very defect they were meant to guard.
    """
    return {"binding_constraint": constraint, "chain_schema": RECORD_SCHEMA}


class TestUnreachedIsNotPassed:
    """The distinction the module exists for."""

    def test_a_gate_behind_a_short_circuit_is_unreached_not_passed(self) -> None:
        got = examine([_Entry()] * 5, [_risk("no_exposure")] * 5)
        assert got.gates[0].status == FIRED
        assert [g.status for g in got.gates[1:]] == [UNREACHED] * (len(CHAIN) - 1)

    def test_a_gate_that_evaluated_and_allowed_is_passed(self) -> None:
        """Nothing bound at all, so every gate ran and let it through. That is a real permission and
        must read differently from never running."""
        got = examine([_Entry(is_abstention=False)] * 5, [_risk("none")] * 5)
        assert all(g.status == PASSED for g in got.gates)

    def test_the_two_states_produce_different_prose(self) -> None:
        unreached = examine([_Entry()] * 3, [_risk("no_exposure")] * 3).gates[1]
        passed = examine([_Entry()] * 3, [_risk("none")] * 3).gates[1]
        assert "never executed" in unreached.note
        assert "allowed every one" in passed.note
        assert unreached.note != passed.note

    def test_an_unreached_gate_reports_zero_reached_not_zero_fired(self) -> None:
        """A reader who sees fired=0 and reached=0 knows it was unreachable. fired=0 alone is the
        ambiguity every prior implementation ships."""
        got = examine([_Entry()] * 4, [_risk("no_exposure")] * 4).gates[2]
        assert got.fired == 0 and got.reached == 0

    def test_partial_short_circuit_leaves_later_gates_unreached(self) -> None:
        """Half the population stopped at gate 1; the rest reached gate 2 and bound there. Gates 3+
        saw nobody and must say so."""
        records = [_risk("no_exposure")] * 5 + [_risk("min_confidence")] * 5
        got = examine([_Entry()] * 10, records)
        by_name = {g.gate.name: g for g in got.gates}
        assert by_name["no_exposure"].status == FIRED
        assert by_name["min_confidence"].status == FIRED
        assert by_name["min_confidence"].reached == 5
        assert by_name["oracle_stale"].status == UNREACHED


class TestReachabilityIsComputedNotDeclared:
    """`vibe-trading` declares a static list of checked limits; an order denied at the first gate
    is recorded as having passed all twelve (`sdk_order_gate.py:283`, never mutated). Ours
    subtracts."""

    def test_each_gate_sees_only_what_survived_the_one_before(self) -> None:
        records = (
            [_risk("no_exposure")] * 2 + [_risk("min_confidence")] * 3
            + [_risk("max_position")] * 5
        )
        got = examine([_Entry()] * 10, records)
        reached = [g.reached for g in got.gates]
        # Ten decisions; `no_exposure` takes two, `min_confidence` three, and everything after sees
        # the same five. The tail is as long as the chain, so adding a gate lengthens it here too.
        assert reached == [10, 8] + [5] * (len(CHAIN) - 2)

    def test_reached_never_increases_down_the_chain(self) -> None:
        records = [_risk("no_exposure")] * 3 + [_risk("oracle_stale")] * 4 + [_risk("none")] * 3
        got = examine([_Entry()] * 10, records)
        reached = [g.reached for g in got.gates]
        assert reached == sorted(reached, reverse=True)

    def test_the_chain_is_recorded_in_source_order(self) -> None:
        assert [g.order for g in CHAIN] == list(range(1, len(CHAIN) + 1))
        assert CHAIN[0].name == "no_exposure"

    def test_the_chain_matches_the_constitution_it_describes(self) -> None:
        """If a rule is added to, removed from or reordered in the policy, this file is stale and
        the funnel silently attributes hits to the wrong gate."""
        import inspect

        from argus.agents.desk import ConstitutionPolicy

        source = inspect.getsource(ConstitutionPolicy.rule)
        # **Every gate, gate 1 included.** It used to be excluded from this loop because it returned
        # the shared "none" token instead of its own name — the exclusion in this test was the
        # defect's own footprint, sitting in the file written to catch it.
        #
        # The two kinds are now spelled differently in the source, because they behave differently:
        # a terminal gate returns with `binding_constraint="<name>"`, while a ceiling gate appends
        # `("<name>", cap, reason)` to the list the minimum is taken over. Both forms are checked,
        # so a gate that disappears from either is still caught here.
        for gate in CHAIN:
            spelling = (
                f'binding_constraint="{gate.name}"' if gate.terminal else f'"{gate.name}",'
            )
            assert spelling in source, gate.name
        assert 'reason="no exposure proposed; nothing to narrow"' in source

    def test_the_two_gate_kinds_are_declared_the_way_the_source_behaves(self) -> None:
        """The distinction that stopped a resized order from bypassing the circuit breaker.

        A terminal gate ends the evaluation; a ceiling gate contributes a cap and every one of them
        runs. If `CHAIN` mislabelled a ceiling as terminal, the funnel would report gates UNREACHED
        that demonstrably executed.
        """
        import inspect

        from argus.agents.desk import ConstitutionPolicy

        source = inspect.getsource(ConstitutionPolicy.rule)
        terminal = [g.name for g in CHAIN if g.terminal]
        ceilings = [g.name for g in CHAIN if not g.terminal]
        assert terminal == ["no_exposure", "min_confidence", "oracle_stale"]
        assert ceilings == [
            "unhedgeable_gap", "gross_exposure", "signed_exposure", "hedge_integrity",
            "margin_usage", "factor_exposure", "scenario_loss", "liquidation_cost",
            "per_symbol_underperformance", "risk_budget", "session_volatility", "max_position",
        ]
        # Every ceiling must actually be collected rather than returned early.
        assert "ceilings.append((" in source
        assert "min(ceilings, key=" in source, "the minimum ceiling must be the binding one"

    def test_the_terminal_chain_order_matches_the_order_in_the_source(self) -> None:
        """Order still matters for the terminal gates, because they still short-circuit.

        It deliberately no longer matters for the ceilings: a minimum does not care what order its
        arguments arrive in, which is one fewer thing to get wrong when a gate is added.
        """
        import inspect

        from argus.agents.desk import ConstitutionPolicy

        source = inspect.getsource(ConstitutionPolicy.rule)
        terminal = [g for g in CHAIN if g.terminal]
        positions = [source.index(f'binding_constraint="{g.name}"') for g in terminal[1:]]
        assert positions == sorted(positions), "the terminal gates are out of order versus desk.py"


class TestSessionCoverage:
    def test_a_single_phase_record_is_flagged_as_unevaluated(self) -> None:
        got = examine([_Entry(session_phase="weekend")] * 8, [_risk("no_exposure")] * 8)
        assert got.coverage.is_single_phase
        assert "has not been evaluated" in got.coverage.note

    def test_a_record_with_price_discovery_is_not_flagged(self) -> None:
        entries = [_Entry(session_phase="regular", hours_to_discovery=0.0)] * 8
        got = examine(entries, [_risk("no_exposure")] * 8)
        assert got.coverage.with_price_discovery == 8
        assert "has not been evaluated" not in got.coverage.note

    def test_the_phase_is_read_from_the_record_not_recomputed(self) -> None:
        """The ledger stores what the clock said at decision time. Recomputing from the timestamp
        would re-date every historical decision under today's calendar."""
        got = examine([_Entry(session_phase="regular")] * 3, [_risk("no_exposure")] * 3)
        assert got.coverage.by_phase == {"regular": 3}

    def test_mixed_phases_are_counted_separately(self) -> None:
        entries = [_Entry(session_phase="weekend")] * 6 + [_Entry(session_phase="regular")] * 4
        got = examine(entries, [_risk("no_exposure")] * 10)
        assert got.coverage.by_phase == {"weekend": 6, "regular": 4}
        assert got.coverage.with_price_discovery == 4
        assert not got.coverage.is_single_phase


class TestTheSecondaryFactStaysSecondary:
    """Stated confidence never fell below the floor. That is evidence the floor is not a hidden
    cause — it is not the *reason* the floor is not the cause. The reason is that it never ran."""

    def test_the_verdict_gives_the_reason_before_the_confidence_aside(self) -> None:
        got = examine([_Entry(stated_confidence=0.88)] * 6, [_risk("no_exposure")] * 6)
        verdict = got.verdict
        assert verdict.index("never executed") < verdict.index("Stated confidence")
        assert "that is a second fact, not the reason" in verdict

    def test_it_reports_whether_the_floor_would_have_bound(self) -> None:
        high = examine([_Entry(stated_confidence=0.88)] * 4, [_risk("no_exposure")] * 4)
        low = examine([_Entry(stated_confidence=0.30)] * 4, [_risk("no_exposure")] * 4)
        assert high.confidence_would_have_bound is False
        assert low.confidence_would_have_bound is True

    def test_a_low_confidence_record_does_not_change_the_unreached_status(self) -> None:
        """Even if the floor *would* have bound, it still did not run. The status is about
        execution, never about what the outcome would have been."""
        got = examine([_Entry(stated_confidence=0.30)] * 4, [_risk("no_exposure")] * 4)
        assert got.gates[1].status == UNREACHED


class TestThePopulationIsStated:
    def test_a_funnel_over_fewer_rows_than_decisions_says_so(self) -> None:
        """The risk log began after the ledger. Quoting the full decision count over a subset funnel
        is a denominator swap."""
        got = examine([_Entry()] * 20, [_risk("no_exposure")] * 8)
        assert got.gate_population == 8
        assert "fewer than the 20 on the ledger" in got.verdict

    def test_matching_populations_add_no_caveat(self) -> None:
        got = examine([_Entry()] * 8, [_risk("no_exposure")] * 8)
        assert "fewer than" not in got.verdict

    def test_no_risk_records_falls_back_to_the_ledger_count(self) -> None:
        got = examine([_Entry()] * 6, [])
        assert got.gate_population == 6


class TestTheConclusionCarriesItsFalsifier:
    def test_a_falsifier_is_always_stated(self) -> None:
        got = examine([_Entry()] * 5, [_risk("no_exposure")] * 5)
        assert "refuted" in got.falsifier or "would move the cause" in got.falsifier

    def test_it_reports_how_far_away_the_test_is(self) -> None:
        got = examine([_Entry()] * 5, [_risk("no_exposure")] * 5, hours_to_first_discovery=17.6)
        assert "17.6h away" in got.falsifier

    def test_it_names_what_would_move_the_cause_back_to_the_desk(self) -> None:
        got = examine([_Entry()] * 5, [_risk("no_exposure")] * 5, hours_to_first_discovery=3.0)
        assert "the cause is the desk" in got.falsifier


class TestItRefusesToExplainNothing:
    def test_an_empty_record_raises_rather_than_diagnosing(self) -> None:
        with pytest.raises(AutopsyError, match="nothing to explain"):
            examine([], [])

    def test_a_record_that_did_propose_exposure_says_so_instead(self) -> None:
        """When the desk starts trading this module must stop explaining why it does not."""
        got = examine([_Entry(is_abstention=False)] * 5, [_risk("max_position")] * 5)
        assert got.proposed_exposure == 5
        assert "proposed exposure, so" in got.verdict
        assert "never executed" not in got.verdict


class TestTheReport:
    def test_it_serialises_with_every_gate(self) -> None:
        got = examine([_Entry()] * 5, [_risk("no_exposure")] * 5).as_dict()
        assert len(got["gates"]) == len(CHAIN)
        assert got["unreached_gates"] == [
            "min_confidence", "oracle_stale", "unhedgeable_gap", "gross_exposure",
            "signed_exposure", "hedge_integrity", "margin_usage", "factor_exposure",
            "scenario_loss", "liquidation_cost", "per_symbol_underperformance", "risk_budget",
            "session_volatility", "max_position",
        ]

    def test_the_rendered_report_shows_the_chain_and_the_falsifier(self) -> None:
        text = examine([_Entry()] * 5, [_risk("no_exposure")] * 5).render()
        assert "CONSTITUTION CHAIN, in source order" in text
        assert "FALSIFIER" in text
        assert UNREACHED in text


class TestTheLiveRecord:
    def test_the_real_log_is_explained_without_raising(self) -> None:
        import json

        from argus.paper.ledger import PaperLedger
        from argus.paper.runner import LEDGER_PATH, RISK_PATH

        if not LEDGER_PATH.exists():
            pytest.skip("no live ledger on this machine")
        records = []
        if RISK_PATH.exists():
            records = [
                json.loads(x)
                for x in RISK_PATH.read_text(encoding="utf-8").splitlines() if x.strip()
            ]
        got = examine(PaperLedger(path=LEDGER_PATH).entries, records)
        assert got.decisions > 0
        assert got.verdict
        # Whatever the state of the log, reached counts must be monotone down the chain.
        reached = [g.reached for g in got.gates]
        assert reached == sorted(reached, reverse=True)


class TestTheFalsifierFired:
    """The falsifier fired on 2026-09-14 and the published conclusion was rewritten.

    These pin the rewrite. The failure mode they guard against is the one that matters most for a
    module like this: quietly restoring a refuted explanation because it reads better than the
    honest one.
    """

    def test_an_rth_record_with_no_exposure_fires_the_falsifier(self) -> None:
        entries = [_Entry(session_phase="rth", hours_to_discovery=0.0)] * 12
        got = examine(entries, [_risk("no_exposure")] * 12)
        assert got.falsifier_has_fired

    def test_a_weekend_only_record_does_not_fire_it(self) -> None:
        """The condition is price discovery, not merely 'a cycle ran'."""
        got = examine([_Entry(session_phase="weekend")] * 12, [_risk("no_exposure")] * 12)
        assert not got.falsifier_has_fired

    def test_proposing_exposure_does_not_fire_it(self) -> None:
        """The falsifier is about abstaining *despite* discovery. A desk that trades has not
        refuted anything — it has moved past the question."""
        entries = [_Entry(session_phase="rth", hours_to_discovery=0.0, is_abstention=False)] * 6
        got = examine(entries, [_risk("none")] * 6)
        assert not got.falsifier_has_fired

    def test_the_verdict_says_refuted_rather_than_explaining_it_away(self) -> None:
        entries = [_Entry(session_phase="rth", hours_to_discovery=0.0)] * 12
        verdict = examine(entries, [_risk("no_exposure")] * 12).verdict
        assert "REFUTED" in verdict
        assert "conviction about direction" in verdict

    def test_the_verdict_admits_the_refusal_may_still_be_correct(self) -> None:
        """Refuting the sampling explanation does not establish that the desk is wrong to abstain.
        Claiming it did would replace one overreach with another."""
        entries = [_Entry(session_phase="rth", hours_to_discovery=0.0)] * 12
        verdict = examine(entries, [_risk("no_exposure")] * 12).verdict
        assert "UNDEFINED" in verdict

    def test_the_falsifier_line_reports_firing_rather_than_a_countdown(self) -> None:
        entries = [_Entry(session_phase="rth", hours_to_discovery=0.0)] * 12
        assert "FIRED" in examine(entries, [_risk("no_exposure")] * 12).falsifier

    def test_the_refuted_claim_survives_only_as_a_quotation(self) -> None:
        """The old conclusion is kept in the docstring **on purpose** — a module that silently
        deletes what it got wrong teaches a reader nothing. What must not survive is the claim
        stated as current fact. So every line carrying it has to be a blockquote, and the refutation
        has to come first.
        """
        from argus.eval import autopsy

        doc = (autopsy.__doc__ or "").splitlines()
        claim = "observation about when we asked it"
        carrying = [i for i, line in enumerate(doc) if claim in line]
        assert carrying, "the refuted claim should be preserved as a quotation, not deleted"
        for i in carrying:
            assert doc[i].lstrip().startswith(">"), (
                f"line {i} states the refuted claim as fact rather than quoting it: {doc[i]!r}"
            )
        refuted_at = next(i for i, line in enumerate(doc) if "REFUTED" in line)
        assert refuted_at < min(carrying), "the refutation must precede the claim it refutes"


class TestTheAllClearIsNotGateOne:
    """The defect this vocabulary change exists for, driven from both sides.

    Schema 1 had `ConstitutionPolicy.rule` return ``binding_constraint="none"`` from two different
    places — gate 1's no-exposure short circuit and the terminal all-clear — and `examine` counted
    every ``"none"`` as gate 1 firing. An intent that reached and cleared all seven gates was
    therefore recorded as having been stopped by the first one, and **gates 2-7 reported UNREACHED
    on the exact decision that proved them reachable.**

    It stayed invisible because no decision in 207 has ever proposed exposure. These tests are the
    reason it cannot come back silently.
    """

    def test_passing_every_gate_reads_passed_not_unreached(self) -> None:
        got = examine([_Entry(is_abstention=False)] * 6, [_risk("none")] * 6)
        assert [g.status for g in got.gates] == [PASSED] * len(CHAIN)

    def test_gate_one_returning_still_leaves_the_rest_unreached(self) -> None:
        got = examine([_Entry()] * 6, [_risk("no_exposure")] * 6)
        assert got.gates[0].status == FIRED
        assert all(g.status == UNREACHED for g in got.gates[1:])

    def test_the_two_are_not_the_same_record(self) -> None:
        """If these ever collapse to one token again, this is the test that says so."""
        assert _risk("no_exposure") != _risk("none")

    def test_a_mixed_population_attributes_each_correctly(self) -> None:
        records = [_risk("no_exposure")] * 4 + [_risk("none")] * 6
        got = examine([_Entry()] * 10, records)
        by_name = {g.gate.name: g for g in got.gates}
        assert by_name["no_exposure"].fired == 4
        # The six that passed everything must still be counted as having reached the last gate.
        assert by_name["max_position"].reached == 6
        assert by_name["max_position"].status == PASSED


class TestItRefusesAVocabularyItCannotRead:
    """A funnel that ignores a value it does not recognise reports a larger `reached` than it
    measured — and `reached` is the denominator of every PASSED claim this module makes.

    This module's own docstring convicts `vibe-trading` of a `checked_limits` field that is a static
    literal. Silently dropping constraint families would be the same defect wearing our name.
    """

    def test_an_unknown_constraint_raises_rather_than_being_ignored(self) -> None:
        with pytest.raises(AutopsyError, match="cannot interpret"):
            examine([_Entry()] * 3, [_risk("mandate")] * 3)

    def test_the_error_names_the_offending_value(self) -> None:
        with pytest.raises(AutopsyError, match="adversary_severe"):
            examine([_Entry()] * 3, [_risk("adversary_severe")] * 3)

    def test_a_null_constraint_raises(self) -> None:
        """`paper/runner.py` writes null when the Constitution produced no ruling at all. The old
        reader turned that into the string "None" via `str(None)`, which matched no gate and fell
        through the walk without decrementing anything."""
        with pytest.raises(AutopsyError, match="null binding_constraint"):
            null = {"binding_constraint": None, "chain_schema": RECORD_SCHEMA}
            examine([_Entry()] * 3, [null] * 3)

    def test_a_schema_one_record_is_refused_not_reinterpreted(self) -> None:
        """Reading a schema-1 file under schema-2 rules would report the **opposite** of the truth,
        so it is refused and the migration is named."""
        with pytest.raises(AutopsyError, match="migrate_risk_records"):
            examine([_Entry()] * 3, [{"binding_constraint": "none"}] * 3)
