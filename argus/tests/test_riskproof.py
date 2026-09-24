"""Proving the risk layer by sweeping it — and proving the prover by breaking the layer.

A sweep that reports "no violations" is worthless unless it can be shown to find one. Half of this
file feeds the harness deliberately broken policies and requires it to catch each.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from argus.agents.desk import ConstitutionPolicy
from argus.decision.verdicts import (
    ConstitutionRuling,
    ConstitutionVerdict,
    Intent,
    Side,
    Verdict,
    apply_constraint,
)
from argus.eval.riskproof import (
    CONFIDENCES,
    QUANTITIES,
    RiskProof,
    State,
    check_invariants,
    states,
    sweep,
)
from argus.risk.hedgeability import HedgeabilitySurface
from argus.truth.clocks import SessionPhase, SessionState

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def proof() -> RiskProof:
    """The full sweep, once. It covers 2,177,280 states and takes minutes; eleven tests used to run
    it again each, which made this one file most of a fresh clone's test time."""
    return sweep(now=NOW)


def _state(**over: object) -> State:
    base: dict[str, object] = {
        "verdict": Verdict.TRADE, "side": Side.BUY, "quantity": Decimal("100"),
        "confidence": 0.9, "hedges_empty": False, "nav_stale": False,
        "phase": SessionPhase.RTH,
    }
    base.update(over)
    return State(**base)  # type: ignore[arg-type]


def _intent(quantity: str = "100", side: Side = Side.BUY,
            verdict: Verdict = Verdict.TRADE) -> Intent:
    return Intent(
        symbol="NVDAUSDT", side=side, quantity=Decimal(quantity), verdict=verdict,
        stated_confidence=0.9, thesis="t",
        invalidation=("f",) if verdict.opens_exposure else (),
    )


class TestTheDomainIsCoveredHonestly:
    def test_the_sweep_visits_thousands_of_states(self) -> None:
        assert len(list(states())) > 1000

    def test_boundaries_are_straddled_on_both_configured_limits(self) -> None:
        policy = ConstitutionPolicy()
        for limit in (policy.max_unhedged_notional, policy.max_position_notional):
            assert any(q < limit for q in QUANTITIES)
            assert limit in QUANTITIES
            assert any(q > limit for q in QUANTITIES)

    def test_the_confidence_boundary_itself_is_swept(self) -> None:
        assert ConstitutionPolicy().min_confidence_to_trade in CONFIDENCES

    def test_every_session_phase_appears(self) -> None:
        assert {s.phase for s in states()} == set(SessionPhase)

    def test_both_sides_appear(self) -> None:
        assert {s.side for s in states()} == {Side.BUY, Side.SELL}

    def test_abstentions_are_swept_and_are_swept_at_zero(self) -> None:
        """They were skipped entirely at first, so the no-new-trade invariant never ran."""
        abstentions = [s for s in states() if not s.verdict.carries_quantity]
        assert abstentions
        assert {s.quantity for s in abstentions} == {Decimal("0")}

    def test_the_no_new_trade_invariant_has_states_to_run_against(self, proof: RiskProof) -> None:
        assert proof.swept > 0
        assert any(not s.verdict.opens_exposure for s in states())


class TestTheRealPolicyIsSound:
    def test_no_invariant_is_violated_anywhere_in_the_domain(self, proof: RiskProof) -> None:
        assert proof.violations == [], proof.violations[:5]
        assert proof.sound

    def test_the_sweep_produces_no_errors(self, proof: RiskProof) -> None:
        assert proof.errors == []

    def test_every_configured_rule_binds_somewhere(self, proof: RiskProof) -> None:
        """The shadowing check: a rule that never fires is a protection that does not exist."""
        assert proof.unreachable == ()

    def test_the_layer_actually_narrows_a_substantial_share_of_proposals(
        self, proof: RiskProof
    ) -> None:
        assert 0.1 < proof.intervention_rate < 0.95

    def test_precedence_is_recorded_where_rules_overlap(self, proof: RiskProof) -> None:
        assert proof.precedence
        assert proof.precedence["min_confidence+oracle_stale"] == "min_confidence"

    def test_confidence_outranks_every_other_rule(self, proof: RiskProof) -> None:
        """Documented from the sweep, not from reading the source."""
        for combo, winner in proof.precedence.items():
            if "min_confidence" in combo.split("+"):
                assert winner == "min_confidence", combo

    def test_the_report_renders_and_serialises(self, proof: RiskProof) -> None:
        text = " ".join(proof.render())
        assert "no invariant violated" in text
        assert "none is shadowed" in text
        got = proof.as_dict()
        assert got["sound"] is True and got["swept"] == proof.swept


class TestPerSymbolUnderperformance:
    """Gate 12, the ARGUS-native ``LowProfitPairs``. A fast, targeted check (one swept state, not
    the whole domain) that the gate actually binds when built — complementing the slow full-domain
    `test_every_configured_rule_binds_somewhere`, which already proves it is not unreachable but
    costs a full sweep to say so."""

    def test_the_gate_binds_when_the_symbol_is_underperforming(self) -> None:
        proof = sweep(state_pool=[_state(extra_risk="underperformance")], now=NOW)
        assert proof.violations == []
        assert proof.bindings.get("per_symbol_underperformance", 0) == 1


# --- proving the prover -------------------------------------------------------------------------

@dataclass(frozen=True)
class _Enlarging(ConstitutionPolicy):
    """A policy that quietly doubles the position. The cardinal sin."""

    def rule(
        self, intent: Intent, *, session: SessionState, hedges: HedgeabilitySurface
    ) -> ConstitutionRuling:
        if intent.verdict.carries_quantity and intent.quantity > 0:
            bigger = Intent(
                symbol=intent.symbol, side=intent.side, quantity=intent.quantity * 2,
                verdict=intent.verdict, stated_confidence=intent.stated_confidence,
                thesis=intent.thesis, invalidation=intent.invalidation,
            )
            return ConstitutionRuling(
                ConstitutionVerdict.ALLOW, "none", "enlarged on purpose", bigger
            )
        return apply_constraint(
            intent, verdict=ConstitutionVerdict.ALLOW, binding_constraint="none", reason="ok",
        )


@dataclass(frozen=True)
class _Reversing(ConstitutionPolicy):
    """A policy that flips the side — a strategy wearing a risk layer's clothes."""

    def rule(
        self, intent: Intent, *, session: SessionState, hedges: HedgeabilitySurface
    ) -> ConstitutionRuling:
        if intent.verdict.carries_quantity and intent.quantity > 0:
            flipped = Intent(
                symbol=intent.symbol,
                side=Side.SELL if intent.side is Side.BUY else Side.BUY,
                quantity=intent.quantity, verdict=intent.verdict,
                stated_confidence=intent.stated_confidence, thesis=intent.thesis,
                invalidation=intent.invalidation,
            )
            return ConstitutionRuling(
                ConstitutionVerdict.ALLOW, "none", "reversed on purpose", flipped
            )
        return apply_constraint(
            intent, verdict=ConstitutionVerdict.ALLOW, binding_constraint="none", reason="ok",
        )


@dataclass(frozen=True)
class _Shadowed(ConstitutionPolicy):
    """A policy whose confidence rule swallows every state, hiding the others."""

    def rule(
        self, intent: Intent, *, session: SessionState, hedges: HedgeabilitySurface
    ) -> ConstitutionRuling:
        if not intent.verdict.carries_quantity or intent.quantity <= 0:
            return apply_constraint(
                intent, verdict=ConstitutionVerdict.ALLOW, binding_constraint="none",
                reason="nothing to narrow",
            )
        return apply_constraint(
            intent, verdict=ConstitutionVerdict.REJECT, binding_constraint="min_confidence",
            reason="swallows everything",
        )


@pytest.fixture(scope="module")
def slice_of_the_domain() -> list[State]:
    """Every 25th state. The prover's teeth are about detection, not coverage, and each broken
    policy below used to be swept over all 2,177,280 states — five full sweeps to show five
    violations that the first few hundred states already exhibit."""
    from itertools import islice

    return list(islice(states(), 0, None, 25))


class TestTheProverHasTeeth:
    def test_a_policy_that_enlarges_a_position_is_caught(
        self, slice_of_the_domain: list[State]
    ) -> None:
        proof = sweep(_Enlarging(), state_pool=slice_of_the_domain, now=NOW)
        assert not proof.sound
        assert any(v.invariant == "never_increases" for v in proof.violations)

    def test_a_policy_that_reverses_a_side_is_caught(
        self, slice_of_the_domain: list[State]
    ) -> None:
        proof = sweep(_Reversing(), state_pool=slice_of_the_domain, now=NOW)
        assert not proof.sound
        assert any(v.invariant == "never_reverses" for v in proof.violations)

    def test_a_shadowing_policy_is_reported_as_unreachable_rules(
        self, slice_of_the_domain: list[State]
    ) -> None:
        proof = sweep(_Shadowed(), state_pool=slice_of_the_domain, now=NOW)
        assert set(proof.unreachable) >= {"oracle_stale", "unhedgeable_gap", "max_position"}
        assert "UNREACHABLE" in " ".join(proof.render())

    def test_a_shadowing_policy_still_obeys_the_asymmetry_invariants(
        self, slice_of_the_domain: list[State]
    ) -> None:
        """Shadowing is a coverage bug, not an asymmetry bug; the two must be reported apart."""
        proof = sweep(_Shadowed(), state_pool=slice_of_the_domain, now=NOW)
        assert proof.violations == []
        assert proof.unreachable

    def test_the_violation_names_the_state_it_was_found_at(
        self, slice_of_the_domain: list[State]
    ) -> None:
        proof = sweep(_Enlarging(), state_pool=slice_of_the_domain, now=NOW)
        assert "q=" in proof.violations[0].state and "c=" in proof.violations[0].state


class TestTheInvariantsThemselves:
    def test_an_untouched_allow_passes(self) -> None:
        original = _intent()
        ruling = apply_constraint(
            original, verdict=ConstitutionVerdict.ALLOW, binding_constraint="none", reason="ok",
        )
        assert check_invariants(original, ruling, _state()) == []

    def test_a_reduction_passes(self) -> None:
        original = _intent("100")
        ruling = apply_constraint(
            original, verdict=ConstitutionVerdict.RESIZE, binding_constraint="max_position",
            reason="cut", resized_quantity=Decimal("10"),
        )
        assert check_invariants(original, ruling, _state()) == []

    def test_a_rejection_passes(self) -> None:
        original = _intent("100")
        ruling = apply_constraint(
            original, verdict=ConstitutionVerdict.REJECT, binding_constraint="min_confidence",
            reason="no",
        )
        assert check_invariants(original, ruling, _state()) == []

    def test_an_enlargement_is_flagged(self) -> None:
        original = _intent("10")
        bigger = Intent(
            symbol="NVDAUSDT", side=Side.BUY, quantity=Decimal("100"), verdict=Verdict.TRADE,
            stated_confidence=0.9, thesis="t", invalidation=("f",),
        )
        ruling = ConstitutionRuling(ConstitutionVerdict.ALLOW, "none", "bad", bigger)
        found = check_invariants(original, ruling, _state())
        assert {v.invariant for v in found} >= {"never_increases"}

    def test_an_allow_that_changed_anything_is_flagged(self) -> None:
        original = _intent("100")
        smaller = Intent(
            symbol="NVDAUSDT", side=Side.BUY, quantity=Decimal("50"), verdict=Verdict.TRADE,
            stated_confidence=0.9, thesis="t", invalidation=("f",),
        )
        ruling = ConstitutionRuling(ConstitutionVerdict.ALLOW, "none", "bad", smaller)
        assert any(v.invariant == "allow_is_untouched"
                   for v in check_invariants(original, ruling, _state()))

    def test_a_legitimate_hedge_leg_passes(self) -> None:
        """REQUIRE_HEDGE attaching a different instrument is the case the two new invariants are
        designed to permit. If this ever failed, the prover would forbid hedging."""
        original = _intent("100")
        ruling = apply_constraint(
            original, verdict=ConstitutionVerdict.REQUIRE_HEDGE,
            binding_constraint="unhedgeable_gap", reason="hedge it",
            required_hedge=("QQQUSDT",),
        )
        assert check_invariants(original, ruling, _state()) == []

    def test_a_hedge_in_the_same_instrument_is_flagged(self) -> None:
        """A "hedge" in the symbol it claims to hedge is more of the same trade wearing a risk
        layer's label. This is the mutant the intent invariant exists to catch."""
        original = _intent("100")
        ruling = apply_constraint(
            original, verdict=ConstitutionVerdict.REQUIRE_HEDGE,
            binding_constraint="unhedgeable_gap", reason="not really a hedge",
            required_hedge=("NVDAUSDT",),
        )
        assert any(v.invariant == "intent_monotonicity"
                   for v in check_invariants(original, ruling, _state()))

    def test_a_leg_attached_outside_a_hedge_ruling_is_flagged(self) -> None:
        """The Constitution originating a position of its own under cover of ALLOW. Nothing in the
        four original invariants could see this, because the primary quantity never moves."""
        original = _intent("100")
        smuggled = Intent(
            symbol="NVDAUSDT", side=Side.BUY, quantity=Decimal("100"), verdict=Verdict.TRADE,
            stated_confidence=0.9, thesis="t", invalidation=("f",),
            required_hedge=("TSLAUSDT",),
        )
        ruling = ConstitutionRuling(ConstitutionVerdict.ALLOW, "none", "bad", smuggled)
        assert any(v.invariant == "intent_monotonicity"
                   for v in check_invariants(original, ruling, _state()))

    def test_raising_net_exposure_is_flagged(self) -> None:
        """Risk monotonicity under the declared functional. Caught here even though the four
        original checks also catch the enlargement, because the two answer different questions and
        a future functional that is not simply quantity must still be checked."""
        original = _intent("10")
        bigger = Intent(
            symbol="NVDAUSDT", side=Side.BUY, quantity=Decimal("50"), verdict=Verdict.TRADE,
            stated_confidence=0.9, thesis="t", invalidation=("f",),
        )
        ruling = ConstitutionRuling(ConstitutionVerdict.ALLOW, "none", "bad", bigger)
        assert any(v.invariant == "risk_monotonicity"
                   for v in check_invariants(original, ruling, _state()))

    def test_an_abstention_carries_no_exposure_whatever_its_quantity(self) -> None:
        """A REJECT that leaves a stale quantity behind must not read as risk. The functional
        reads the verdict, not the number alone."""
        original = _intent("100")
        ruling = apply_constraint(
            original, verdict=ConstitutionVerdict.REJECT,
            binding_constraint="min_confidence", reason="no",
        )
        assert not any(v.invariant == "risk_monotonicity"
                       for v in check_invariants(original, ruling, _state()))

    def test_turning_an_abstention_into_a_trade_is_flagged(self) -> None:
        original = _intent("0", verdict=Verdict.NO_TRADE)
        trade = Intent(
            symbol="NVDAUSDT", side=Side.BUY, quantity=Decimal("5"), verdict=Verdict.TRADE,
            stated_confidence=0.9, thesis="t", invalidation=("f",),
        )
        ruling = ConstitutionRuling(ConstitutionVerdict.ALLOW, "none", "bad", trade)
        assert any(v.invariant == "never_creates_a_trade"
                   for v in check_invariants(original, ruling, _state()))


def test_an_empty_sweep_is_sound_but_says_nothing() -> None:
    """Vacuous soundness must be visible: nothing swept is not the same as nothing wrong."""
    proof = sweep(state_pool=[], now=NOW)
    assert proof.sound and proof.swept == 0
    assert proof.unreachable == proof.rules


def test_the_proof_is_reproducible() -> None:
    """Two runs from cold over the same every-25th slice of the domain agree exactly. A slice,
    because the property is determinism, not coverage — the full domain is swept once above."""
    from itertools import islice

    from argus.eval import riskproof

    pool = list(islice(states(), 0, None, 25))
    riskproof._THROTTLES.clear()
    first = sweep(state_pool=pool, now=NOW)
    riskproof._THROTTLES.clear()  # from cold again, so the cache is not what makes them agree
    second = sweep(state_pool=pool, now=NOW)
    assert first.as_dict()["bindings"] == second.as_dict()["bindings"]
    assert first.swept == second.swept


def test_a_riskproof_with_no_states_reports_a_zero_rate_not_a_crash() -> None:
    assert RiskProof(generated_at=NOW).intervention_rate == 0.0


@pytest.mark.parametrize("phase", list(SessionPhase))
def test_every_phase_can_express_both_staleness_states(phase: SessionPhase) -> None:
    """If a phase could not express stale NAV, the sweep would be quietly incomplete there."""
    pool = [_state(phase=phase, nav_stale=stale) for stale in (True, False)]
    proof = sweep(state_pool=pool, now=NOW)
    assert proof.errors == []
    assert proof.swept == 2
