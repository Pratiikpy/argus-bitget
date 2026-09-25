"""Shadow evaluation: old and new risk logic on the same orders, every changed ruling listed."""

from __future__ import annotations

import itertools
from decimal import Decimal

import pytest

from argus.eval import risk_certify, risk_shadow
from argus.execution import guard
from argus.risk import circuit

PRE_S20 = risk_shadow.BASELINES / "guard_2026-09-25_pre_s20.py.txt"
CIRCUIT_BASELINE = risk_shadow.BASELINES / "circuit_2026-09-25.py.txt"


def _sample() -> list[risk_shadow.Intent]:
    return [*risk_shadow.probe_intents(),
            *itertools.islice(risk_shadow.swept_intents(), 0, 190_944, 53)]


class TestTheShadow:
    def test_a_version_against_itself_changes_nothing(self) -> None:
        report = risk_shadow.shadow_guard(guard, guard, _sample(), {})
        assert all(stream["changed"] == 0 for stream in report.values()), report

    def test_a_one_character_change_is_listed_with_both_rulings(self) -> None:
        source = risk_certify.source_of("guard").replace(
            "    if stepped_qty < instrument.min_order_qty:\n",
            "    if stepped_qty <= instrument.min_order_qty:\n",
        )
        candidate = risk_shadow.load_source(source, "candidate floor")
        at_minimum = [
            risk_shadow.Intent("probe", "one contract, minimum one", "coarse", "1",
                               reference="100", window="empty"),
            risk_shadow.Intent("probe", "0.01 at 600, minimum 0.01", "nvda", "0.01",
                               reference="600", window="empty"),
            risk_shadow.Intent("probe", "well above the minimum", "nvda", "2",
                               reference="600", window="empty"),
        ]
        report = risk_shadow.shadow_guard(guard, candidate, at_minimum, {})["probe"]
        assert report["changed"] == 2 and report["unchanged"] == 1
        assert report["transitions"] == {"allowed -> denied quantity_below_minimum": 2}
        first = report["changes"][0]
        assert first["intent"]["id"] == "one contract, minimum one"
        assert first["old"]["outcome"] == "allowed"
        assert first["new"]["outcome"] == "denied quantity_below_minimum"

    def test_the_pre_s20_guard_differs_only_on_the_probes_aimed_at_its_defects(self) -> None:
        if not PRE_S20.exists():
            pytest.skip("the frozen pre-S20 guard is not present")
        before = risk_shadow.load_engine(PRE_S20, "before S20").module
        report = risk_shadow.shadow_guard(before, guard, _sample(), {})
        assert report["swept"]["changed"] == 0
        changed = {c["intent"]["id"] for c in report["probe"]["changes"]}
        assert "side neither buy nor sell" in changed
        assert "31-digit quantity just under the maximum" in changed
        assert "notional a hair over the balance" in changed
        assert "limit exactly on the band edge" not in changed
        assert report["probe"]["changes_complete"]

    def test_an_exception_is_an_outcome_not_a_crash(self) -> None:
        intent = risk_shadow.Intent("probe", "nan", "nvda", "NaN")
        source = risk_certify.source_of("guard").replace(
            "if value is not None and not value.is_finite():", "if False:")
        broken = risk_shadow.load_source(source, "no finiteness check")
        outcome = risk_shadow.Outcome.of(risk_shadow.run_guard(broken, intent,
                                                               risk_shadow.variant_row("nvda")))
        assert outcome.raised == "InvalidOperation" and outcome.label() == "raised InvalidOperation"

    def test_sizes_compare_by_value(self) -> None:
        one = risk_shadow.Outcome("none", Decimal("1"), None, False, "a", None)
        also_one = risk_shadow.Outcome("none", Decimal("1.00"), None, False, "b", None)
        assert one.same_ruling(also_one)


class TestTheStreams:
    def test_the_recorded_stream_is_the_orders_the_live_cycle_sent(self) -> None:
        intents, stats = risk_shadow.recorded_intents()
        ids = {intent.ident for intent in intents}
        assert {"seq 264", "seq 265"} <= ids
        assert stats["reached_venue_gate"] >= stats["replayable"] >= 2
        assert all(intent.lookup and intent.window == "empty" for intent in intents)

    def test_recorded_and_counterfactual_orders_are_unchanged_by_s20(self) -> None:
        if not (PRE_S20.exists() and risk_shadow.SPECS_PATH.exists()):
            pytest.skip("needs the frozen pre-S20 guard and the frozen venue rows")
        intents, rows, _ = risk_shadow.streams()
        realistic = [i for i in intents if i.stream in ("recorded", "counterfactual")]
        assert len(realistic) > 6_000
        before = risk_shadow.load_engine(PRE_S20, "before S20").module
        report = risk_shadow.shadow_guard(before, guard, realistic, rows)
        assert report["recorded"]["changed"] == 0
        assert report["counterfactual"]["changed"] == 0

    def test_a_note_carrying_the_whole_trace_replays_like_the_one_line_note(self) -> None:
        nvda = guard.Instrument.from_payload(risk_shadow.variant_row("nvda"))
        for quantity in ("1", "1.005"):
            ruling = guard.validate(nvda, quantity=Decimal(quantity), price=Decimal("100"),
                                    reference_price=Decimal("100"))
            assert "\n" in ruling.explain()
            short = risk_shadow.ALLOWED_NOTE.match(risk_shadow.ruling_line(ruling.render()))
            full = risk_shadow.ALLOWED_NOTE.match(risk_shadow.ruling_line(ruling.explain()))
            assert short is not None and full is not None
            assert short.groupdict() == full.groupdict()
            assert risk_shadow.ALLOWED_NOTE.match(ruling.explain()) is None

    def test_every_probe_is_distinct_and_labelled(self) -> None:
        labels = [label for label, _ in risk_shadow.PROBES]
        assert len(labels) == len(set(labels)) == len(risk_shadow.probe_intents())


class TestTheCircuitShadow:
    def test_the_frozen_breaker_and_the_current_one_agree(self) -> None:
        if not CIRCUIT_BASELINE.exists():
            pytest.skip("the frozen breaker is not present")
        before = risk_shadow.load_engine(CIRCUIT_BASELINE, "circuit baseline").module
        report = risk_shadow.shadow_circuit(before, circuit)
        assert report["changed"] == 0 and report["books_x_verdicts"] > 1_000

    def test_a_moved_threshold_is_listed(self) -> None:
        source = risk_certify.source_of("circuit").replace(
            "    if state.total_drawdown >= TOTAL_DRAWDOWN_HALT:\n",
            "    if state.total_drawdown > TOTAL_DRAWDOWN_HALT:\n",
        )
        moved = risk_shadow.load_source(source, "circuit strict threshold")
        report = risk_shadow.shadow_circuit(circuit, moved)
        assert report["changed"] > 0
        assert all(change["book"]["equity"] == "90" for change in report["changes"])
