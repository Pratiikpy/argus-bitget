"""Named risk modes: most restrictive wins, unknown halts, and one line when the posture changes."""

from __future__ import annotations

import itertools
import json
from decimal import Decimal
from pathlib import Path

import pytest

from argus.eval import risk_shadow
from argus.execution.guard import GUARD_GATES, Denial, GateVerdict, Guard, Instrument
from argus.risk import certify, modes
from argus.risk.circuit import BookState
from argus.risk.modes import Demand, Layer, ModeError, ModeNotifier, RiskMode

NVDA = Instrument.from_payload(risk_shadow.variant_row("nvda"))
CALM = BookState(equity=Decimal("100000"), peak_equity=Decimal("100000"),
                 session_open_equity=Decimal("100000"))
LADDER = BookState(equity=Decimal("97000"), peak_equity=Decimal("100000"),
                   session_open_equity=Decimal("97000"))
HALTING = BookState(equity=Decimal("89000"), peak_equity=Decimal("100000"),
                    session_open_equity=Decimal("89000"))


def _stack(**over: object) -> modes.ModeStack:
    facts: dict[str, object] = {"rules_loaded": True, "book": CALM, "environ": {}}
    facts.update(over)
    return modes.stack_for_cycle(**facts)  # type: ignore[arg-type]


class TestPrecedence:
    def test_a_calm_paper_desk_is_in_paper_mode_set_by_its_posture(self) -> None:
        stack = _stack()
        assert stack.effective is RiskMode.PAPER
        assert stack.binding.layer is Layer.POSTURE

    def test_the_most_restrictive_demand_wins(self) -> None:
        assert _stack(book=LADDER).effective is RiskMode.REDUCE_ONLY
        assert _stack(book=HALTING).effective is RiskMode.HALTED
        assert _stack(rules_loaded=False).effective is RiskMode.HALTED

    def test_an_operator_cannot_loosen_a_breaker_halt(self) -> None:
        stack = _stack(book=HALTING, environ={modes.OPERATOR_ENV: "normal"})
        assert stack.effective is RiskMode.HALTED
        assert stack.binding.layer is Layer.CIRCUIT

    def test_the_first_layer_demanding_the_effective_mode_is_named(self) -> None:
        stack = _stack(book=HALTING, environ={modes.OPERATOR_ENV: "halted"})
        assert stack.binding.layer is Layer.OPERATOR

    def test_a_malformed_stack_is_refused(self) -> None:
        with pytest.raises(ModeError):
            modes.ModeStack((Demand(Layer.POSTURE, RiskMode.PAPER, "x"),))
        with pytest.raises(ModeError):
            modes.resolve([Demand(Layer.OPERATOR, None, "a"), Demand(Layer.OPERATOR, None, "b"),
                           modes.posture_demand()])
        with pytest.raises(ModeError):
            modes.resolve([Demand(Layer.POSTURE, None, "no baseline")])

    def test_resolution_matches_the_written_specification_everywhere(self) -> None:
        choices = (None, *RiskMode)
        for operator, venue, breaker, posture in itertools.product(
            choices, choices, choices, (RiskMode.NORMAL, RiskMode.PAPER),
        ):
            stack = modes.resolve([
                Demand(Layer.OPERATOR, operator, "o"), Demand(Layer.VENUE, venue, "v"),
                Demand(Layer.CIRCUIT, breaker, "c"), Demand(Layer.POSTURE, posture, "p"),
            ])
            wanted = certify.expect_mode([(str(d.layer), None if d.mode is None else str(d.mode))
                                          for d in stack.demands])
            assert (str(stack.effective), str(stack.binding.layer)) == wanted


class TestTheLayers:
    @pytest.mark.parametrize(("raw", "mode"), [
        (None, None), ("", None), ("  ", None), ("paper", RiskMode.PAPER),
        (" HALTED ", RiskMode.HALTED), ("reduce_only", RiskMode.REDUCE_ONLY),
        ("hlated", RiskMode.HALTED), ("off", RiskMode.HALTED),
    ])
    def test_the_operator_switch(self, raw: str | None, mode: RiskMode | None) -> None:
        environ = {} if raw is None else {modes.OPERATOR_ENV: raw}
        assert modes.operator_demand(environ).mode is mode

    def test_unknown_breaker_state_and_missing_rules_halt(self) -> None:
        assert modes.circuit_demand("bogus").mode is RiskMode.HALTED
        assert modes.circuit_demand("active").mode is None
        assert modes.venue_demand(rules_loaded=False).mode is RiskMode.HALTED
        assert modes.venue_demand(rules_loaded=True).mode is None

    def test_an_unreadable_book_halts(self) -> None:
        assert _stack(book=None).effective is RiskMode.HALTED

    def test_live_only_when_configured(self) -> None:
        assert modes.posture_demand().mode is RiskMode.PAPER
        assert modes.posture_demand(live_enabled=True).mode is RiskMode.NORMAL


class TestTheModeGate:
    def test_admission_matches_the_written_specification_everywhere(self) -> None:
        for mode, verdict, venue, open_positions in itertools.product(
            RiskMode, ("trade", "hedge", "reduce", "no_trade", "delay", "human_review",
                       "data_insufficient"), ("paper", "live", "moon"), (0, 1),
        ):
            stack = modes.resolve([modes.posture_demand(live_enabled=mode is RiskMode.NORMAL),
                                   Demand(Layer.OPERATOR, mode, "test")])
            got = modes.admit(stack, verdict=verdict, venue=venue,
                              open_positions=open_positions)
            wanted, _ = certify.expect_admission(str(mode), verdict, venue, open_positions)
            assert got.admitted is wanted, (mode, verdict, venue, open_positions)

    def test_a_refused_trade_names_the_breakers_fallback(self) -> None:
        stack = _stack(book=LADDER)
        assert modes.admit(stack, verdict="trade", open_positions=0).narrowed_to == "no_trade"
        assert modes.admit(stack, verdict="trade", open_positions=2).narrowed_to == "reduce"
        assert modes.admit(stack, verdict="hedge").admitted

    def test_a_reduction_is_never_trapped(self) -> None:
        for stack in (_stack(), _stack(book=LADDER), _stack(book=HALTING)):
            for venue in ("paper", "live"):
                assert modes.admit(stack, verdict="reduce", venue=venue).admitted


class TestCheckOrder:
    def test_a_mode_refusal_spends_nothing_and_lists_every_venue_gate(self) -> None:
        guard = Guard(instruments={"NVDAUSDT": NVDA})
        ruling = modes.check_order(guard, _stack(book=HALTING), symbol="NVDAUSDT",
                                   verdict="trade", quantity=Decimal("1"), side="buy",
                                   reference_price=Decimal("100"), now=1000.0)
        assert ruling.denial is Denial.RISK_MODE and ruling.quantity == 0
        assert ruling.trace.gates == ("risk_mode", *GUARD_GATES)
        assert all(e.verdict is GateVerdict.NOT_REACHED for e in ruling.trace.events[1:])
        assert guard.limiter.in_window == 0

    def test_an_admitted_order_carries_one_trace_for_the_whole_path(self) -> None:
        guard = Guard(instruments={"NVDAUSDT": NVDA})
        ruling = modes.check_order(guard, _stack(), symbol="NVDAUSDT", verdict="trade",
                                   quantity=Decimal("1"), side="buy",
                                   reference_price=Decimal("100"), now=1000.0)
        assert ruling.allowed
        assert ruling.trace.gates == ("risk_mode", *GUARD_GATES)
        assert ruling.trace.events[0].verdict is GateVerdict.PASS
        assert "paper admits trade" in ruling.explain()
        assert guard.limiter.in_window == 1


class TestTheNotice:
    def test_the_default_posture_is_not_announced_but_a_change_is(self) -> None:
        notifier = ModeNotifier()
        assert notifier.notice(_stack()) is None
        changed = notifier.notice(_stack(book=LADDER))
        assert changed is not None and "\n" not in changed
        assert changed.startswith("Risk mode changed to reduce_only (was paper).")
        assert "Set by circuit: breaker reduce_only: drawdown_ladder" in changed
        assert notifier.notice(_stack(book=LADDER)) is None
        back = notifier.notice(_stack())
        assert back is not None and back.startswith("Risk mode changed to paper (was reduce_only)")

    def test_a_first_non_default_mode_is_announced_as_active(self) -> None:
        notice = ModeNotifier().notice(_stack(rules_loaded=False))
        assert notice is not None and notice.startswith("Risk mode active: halted.")

    def test_the_notifier_round_trips_and_an_unreadable_state_reannounces(self) -> None:
        notifier = ModeNotifier()
        notifier.notice(_stack(book=LADDER))
        restored = ModeNotifier.from_dict(notifier.as_dict())
        assert restored.last_notified is RiskMode.REDUCE_ONLY
        broken = ModeNotifier.from_dict({"last_notified": "sideways"})
        assert broken.last_notified is None
        assert broken.notice(_stack(book=LADDER)) is not None

    def test_the_persisted_notice_fires_once_per_change_across_processes(
        self, tmp_path: Path,
    ) -> None:
        state = tmp_path / "data" / "risk_mode_notice.json"
        assert modes.persisted_notice(_stack(), state) is None
        changed = modes.persisted_notice(_stack(book=HALTING), state)
        assert changed is not None and changed.startswith("Risk mode changed to halted (was paper)")
        assert modes.persisted_notice(_stack(book=HALTING), state) is None
        assert json.loads(state.read_text(encoding="utf-8"))["last_notified"] == "halted"
        back = modes.persisted_notice(_stack(), state)
        assert back is not None and back.startswith("Risk mode changed to paper (was halted)")
        assert sorted(p.name for p in state.parent.iterdir()) == ["risk_mode_notice.json"]

    @pytest.mark.parametrize("damage", ["{not json", "[]", '{"last_notified": "sideways"}'])
    def test_a_damaged_state_file_reannounces_rather_than_assuming_unchanged(
        self, tmp_path: Path, damage: str,
    ) -> None:
        state = tmp_path / "risk_mode_notice.json"
        state.write_text(damage, encoding="utf-8")
        notice = modes.persisted_notice(_stack(book=LADDER), state)
        assert notice is not None and notice.startswith("Risk mode active: reduce_only.")
        assert json.loads(state.read_text(encoding="utf-8"))["last_notified"] == "reduce_only"

    def test_the_default_state_file_is_under_the_projects_data_directory(self) -> None:
        assert modes.NOTICE_STATE == risk_shadow.DATA / "risk_mode_notice.json"

    def test_the_stack_renders_with_the_binding_layer_marked(self) -> None:
        text = _stack(book=LADDER).render()
        assert text.splitlines()[0].startswith("[mode] risk mode reduce_only, set by circuit")
        assert any(line.startswith("  * circuit") for line in text.splitlines())
        blob = _stack(book=LADDER).as_dict()
        assert blob["effective"] == "reduce_only" and blob["binding"] == "circuit"
