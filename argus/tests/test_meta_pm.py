"""Meta-PM and Autonomy Proof — end-to-end, including against the live Qwen endpoint.

The live test is the one that matters: it demonstrates the full Track 2 loop on the actual
Sleeping-Anchor scenario, with a real model making a real economic choice, being constrained, and
responding. Run it with:

    set -a && . .secrets/qwen.env && set +a && ARGUS_LIVE_LLM=1 pytest tests/test_meta_pm.py -v
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from argus.agents.meta_pm import (
    MarketFrame,
    MetaPM,
    _to_intent,
    deliberation_cost_bps,
)
from argus.cost.model import CostModel
from argus.decision.verdicts import (
    ConstitutionVerdict,
    Intent,
    Side,
    Verdict,
    apply_constraint,
)
from argus.llm.qwen import QwenClient, Thinking, TokenBudget
from argus.proof.autonomy import AutonomyProof, ProofIncomplete, hash_intent
from argus.truth.clocks import ET, DualClock, SessionPhase, SessionState

LIVE = os.environ.get("ARGUS_LIVE_LLM") == "1" and bool(os.environ.get("BITGET_QWEN_API_KEY"))
live_only = pytest.mark.skipif(not LIVE, reason="set ARGUS_LIVE_LLM=1 and load .secrets/qwen.env")

# 03:00 ET Sunday — the scenario the whole system exists for.
SUNDAY_3AM = datetime(2026, 3, 8, 3, 0, tzinfo=ET)


def _frame(position: str = "200") -> MarketFrame:
    clock = DualClock()
    return MarketFrame(
        symbol="rNVDA",
        as_of=SUNDAY_3AM,
        session=clock.state(SUNDAY_3AM, nav_age_seconds=40_000),
        token_price=Decimal("118.40"),
        position_quantity=Decimal(position),
        round_trip_bps=CostModel.bitget_perp().round_trip_bps(),
        hedge_menu=(),  # nothing placeable: NYSE is shut
        evidence=(
            "SEC 8-K filed 02:31 ET: FY guidance revised down 7%. (available_at 02:34 ET)",
            "Viral social post claims a 20% cut; no independent source. Credibility 0.18.",
            "BTC -3.7% over 6h. Token liquidity is roughly a third of RTH depth.",
        ),
    )


def _original() -> Intent:
    return Intent(
        symbol="rNVDA", side=Side.SELL, quantity=Decimal("200"),
        verdict=Verdict.TRADE, stated_confidence=0.71,
        thesis="guidance cut not yet reflected in the token",
        invalidation=("guidance reaffirmed at the open",),
    )


class TestResponseParsing:
    def test_opening_exposure_without_a_falsifier_is_downgraded_not_fabricated(self) -> None:
        """We never invent an invalidation condition on the model's behalf. The decision is real
        but not actionable, so it goes to a human."""
        got = _to_intent(
            {"verdict": "TRADE", "side": "SELL", "quantity": 100,
             "confidence": 0.8, "thesis": "feels right", "invalidation": []},
            "rNVDA",
        )
        assert got.verdict is Verdict.HUMAN_REVIEW
        assert got.quantity == Decimal("0")

    def test_abstention_carries_no_quantity(self) -> None:
        got = _to_intent(
            {"verdict": "NO_TRADE", "side": "SELL", "quantity": 999,
             "confidence": 0.9, "thesis": "edge does not clear 12bps"},
            "rNVDA",
        )
        assert got.quantity == Decimal("0")

    def test_data_insufficient_is_parsed_distinctly(self) -> None:
        got = _to_intent(
            {"verdict": "DATA_INSUFFICIENT", "side": "BUY", "quantity": 0,
             "confidence": 0.2, "thesis": "NAV is frozen and stale"},
            "rNVDA",
        )
        assert got.verdict is Verdict.DATA_INSUFFICIENT
        assert got.verdict.is_abstention


class TestMarketFrame:
    def test_prompt_states_plainly_that_no_hedge_exists(self) -> None:
        block = _frame().to_prompt_block()
        assert "no hedge is placeable" in block
        assert "anchor asleep: True" in block
        assert "12 bps" in block

    def test_state_hash_is_stable_and_sensitive(self) -> None:
        """Replay determinism depends on this: same state, same hash; any change, new hash."""
        a = _frame("200").state_hash()
        assert a == _frame("200").state_hash()
        assert a != _frame("201").state_hash()

    def test_hours_to_discovery_is_computed_not_guessed(self) -> None:
        """Sunday 03:00 -> Monday 09:30 is 30.5 hours."""
        assert _frame().session.hours_to_next_discovery == pytest.approx(30.5, abs=0.1)


class TestAutonomyProof:
    def test_proof_refuses_to_answer_before_it_is_whole(self) -> None:
        """A partial proof that answers 'yes' is worse than no proof — it looks like evidence."""
        proof = AutonomyProof(
            decision_id="d1", as_of=SUNDAY_3AM, market_state_hash="abc",
            llm_original_intent=_original(), llm_original_reasoning="because",
        )
        with pytest.raises(ProofIncomplete, match="does not reach an order"):
            proof.attests_llm_decided()

    def test_revision_before_a_ruling_is_refused(self) -> None:
        proof = AutonomyProof(
            decision_id="d1", as_of=SUNDAY_3AM, market_state_hash="abc",
            llm_original_intent=_original(), llm_original_reasoning="because",
        )
        with pytest.raises(ProofIncomplete, match="before a constitution ruling"):
            proof.record_revision(_original(), "revised")

    def test_full_chain_attests_the_llm_decided(self) -> None:
        proof = AutonomyProof(
            decision_id="d1", as_of=SUNDAY_3AM, market_state_hash="abc",
            llm_original_intent=_original(),
            llm_original_reasoning="guidance cut is not priced; counter-case: it may be stale news",
        )
        ruling = apply_constraint(
            _original(),
            verdict=ConstitutionVerdict.RESIZE,
            binding_constraint="unhedgeable_gap",
            reason="no hedge placeable for 30.5h; gap CVaR exceeds budget",
            resized_quantity=Decimal("105"),
        )
        proof.record_ruling(ruling)
        proof.record_revision(
            Intent(
                symbol="rNVDA", side=Side.SELL, quantity=Decimal("105"),
                verdict=Verdict.TRADE, stated_confidence=0.64,
                thesis="accept the reduced size; edge still clears cost",
                invalidation=("guidance reaffirmed at the open",),
            ),
            "reduced size still clears the 12bps round trip",
        )
        proof.approve()

        assert proof.attests_llm_decided() is True
        assert proof.constitution_intervened is True
        assert proof.constitution_only_reduced() is True
        assert proof.llm_changed_its_mind is True

    def test_the_asymmetry_is_checkable_from_the_artefact_alone(self) -> None:
        """A judge should not have to trust our code — the proof carries both sides."""
        proof = AutonomyProof(
            decision_id="d1", as_of=SUNDAY_3AM, market_state_hash="abc",
            llm_original_intent=_original(), llm_original_reasoning="r",
        )
        proof.record_ruling(apply_constraint(
            _original(), verdict=ConstitutionVerdict.RESIZE,
            binding_constraint="c", reason="r", resized_quantity=Decimal("50"),
        ))
        assert proof.constitution_only_reduced() is True

    def test_record_is_serialisable_and_carries_the_attestations(self) -> None:
        proof = AutonomyProof(
            decision_id="d1", as_of=SUNDAY_3AM, market_state_hash="abc",
            llm_original_intent=_original(), llm_original_reasoning="r",
        )
        proof.record_ruling(apply_constraint(
            _original(), verdict=ConstitutionVerdict.ALLOW,
            binding_constraint="none", reason="within limits",
        ))
        proof.approve()
        record = proof.to_record()

        assert record["attestations"]["constitution_only_reduced"] is True
        assert record["approved_intent_hash"]
        assert record["llm_original_intent"]["hash"] == hash_intent(_original())

    def test_hash_changes_when_the_intent_changes(self) -> None:
        a = _original()
        b = Intent(
            symbol=a.symbol, side=a.side, quantity=Decimal("99"), verdict=a.verdict,
            stated_confidence=a.stated_confidence, thesis=a.thesis,
            invalidation=a.invalidation,
        )
        assert hash_intent(a) != hash_intent(b)


# --- the live end-to-end loop ---------------------------------------------------------------

@live_only
class TestLiveDecisionLoop:
    """The full Track 2 claim, demonstrated against the real model."""

    def test_sleeping_anchor_decision_and_revision(self) -> None:
        client = QwenClient(budget=TokenBudget(limit=60_000))
        pm = MetaPM(client, max_tokens=900)
        frame = _frame()

        proof = pm.decide(frame, decision_id="live-1")

        # The model made an economic choice with a thesis of its own.
        assert proof.llm_original_intent.thesis
        assert proof.llm_original_intent.verdict in set(Verdict)

        ruling = apply_constraint(
            proof.llm_original_intent,
            verdict=ConstitutionVerdict.RESIZE,
            binding_constraint="unhedgeable_gap",
            reason=(
                "no hedge is placeable for 30.5 hours; gap CVaR on the unhedged residual "
                "exceeds the weekend budget"
            ),
            resized_quantity=max(
                Decimal("1"), proof.llm_original_intent.quantity / Decimal("2")
            ),
        ) if proof.llm_original_intent.quantity > 0 else apply_constraint(
            proof.llm_original_intent,
            verdict=ConstitutionVerdict.ALLOW,
            binding_constraint="none",
            reason="no exposure proposed",
        )

        pm.revise(proof, frame, ruling)
        proof.approve()

        assert proof.constitution_only_reduced() is True
        assert proof.attests_llm_decided() is True
        record = proof.to_record()
        assert record["approved_intent_hash"]
        print("\nAUTONOMY PROOF:\n", record)

    def test_model_can_abstain_when_cost_dominates(self) -> None:
        """A model that can only pick a direction will always pick one. Ours must be able to
        decline — and on a 12bps round trip against a thin weekend book, declining is often right.
        """
        client = QwenClient(budget=TokenBudget(limit=40_000))
        pm = MetaPM(client, max_tokens=700)

        clock = DualClock()
        frame = MarketFrame(
            symbol="rNVDA",
            as_of=SUNDAY_3AM,
            session=clock.state(SUNDAY_3AM, nav_age_seconds=90_000),
            token_price=Decimal("118.40"),
            position_quantity=Decimal("0"),
            round_trip_bps=Decimal("12"),
            hedge_menu=(),
            evidence=(
                "No filings, no news, no macro events in the last 48h.",
                "Token spread 34bps, roughly 3x the RTH level. Depth is thin.",
                "NAV oracle is 25h stale and frozen until Monday open.",
            ),
        )
        proof = pm.decide(frame, decision_id="live-2")
        assert proof.llm_original_intent.verdict.is_abstention, (
            f"expected an abstention on a no-information, high-cost, stale-oracle frame; "
            f"got {proof.llm_original_intent.verdict} — {proof.llm_original_intent.thesis}"
        )


class TestDeliberationIsInTheHurdle:
    """The model is told what its own reasoning costs.

    Measured in PRD §3.4: a full budget off-hours is 15.2bps against a 12bps round trip, so a frame
    that quotes the fee alone understates the hurdle by more than half in exactly the sessions the
    desk runs unattended.
    """

    @staticmethod
    def _frame(phase: SessionPhase, deliberation: Decimal) -> MarketFrame:
        return MarketFrame(
            symbol="rNVDA",
            as_of=datetime(2026, 3, 2, 21, 0, tzinfo=UTC),
            session=SessionState(
                as_of=datetime(2026, 3, 2, 21, 0, tzinfo=UTC),
                phase=phase,
                hours_to_next_discovery=12.0,
            ),
            token_price=Decimal("100"),
            position_quantity=Decimal("0"),
            round_trip_bps=Decimal("12"),
            deliberation_bps=deliberation,
        )

    def test_the_hurdle_adds_the_cost_of_deciding(self) -> None:
        frame = self._frame(SessionPhase.OVERNIGHT, Decimal("15.2"))
        assert frame.total_hurdle_bps == Decimal("27.2")

    def test_the_prompt_quotes_the_total_not_the_fee(self) -> None:
        block = self._frame(SessionPhase.OVERNIGHT, Decimal("15.2")).to_prompt_block()
        assert "TOTAL HURDLE: 27.2 bps" in block
        assert "cost of deciding: 15.2 bps" in block

    def test_deliberation_changes_the_state_hash(self) -> None:
        """Two decisions differing only in what thinking cost are different decisions."""
        a = self._frame(SessionPhase.OVERNIGHT, Decimal("15.2")).state_hash()
        b = self._frame(SessionPhase.OVERNIGHT, Decimal("5.07")).state_hash()
        assert a != b

    def test_off_hours_costs_three_times_rth(self) -> None:
        vol = Decimal("0.45")
        rth = deliberation_cost_bps(
            self._frame(SessionPhase.RTH, Decimal("0")).session,
            thinking=Thinking.FULL, annualised_vol=vol,
        )
        overnight = deliberation_cost_bps(
            self._frame(SessionPhase.OVERNIGHT, Decimal("0")).session,
            thinking=Thinking.FULL, annualised_vol=vol,
        )
        assert overnight == rth * 3

    def test_a_cheaper_budget_costs_less(self) -> None:
        session = self._frame(SessionPhase.OVERNIGHT, Decimal("0")).session
        vol = Decimal("0.45")
        low = deliberation_cost_bps(session, thinking=Thinking.LOW, annualised_vol=vol)
        full = deliberation_cost_bps(session, thinking=Thinking.FULL, annualised_vol=vol)
        assert Decimal("0") < low < full

    def test_an_unknown_phase_defaults_to_the_thin_book(self) -> None:
        """The conservative default: an unmapped session is assumed illiquid, never liquid."""
        session = self._frame(SessionPhase.HOLIDAY, Decimal("0")).session
        assert deliberation_cost_bps(
            session, thinking=Thinking.FULL, annualised_vol=Decimal("0.45")
        ) == deliberation_cost_bps(
            self._frame(SessionPhase.OVERNIGHT, Decimal("0")).session,
            thinking=Thinking.FULL, annualised_vol=Decimal("0.45"),
        )


class TestPmThinkingBudgetIsThreaded:
    """The budget is a cost. A desk quoted the FULL-budget hurdle (27.2bps off-hours) correctly
    abstains every cycle, which produced 12 no_trades and an undefined paper-trading Sharpe —
    half of the Track-2 score. LOW drops the hurdle to 18.8bps. This test pins that the budget
    reaches both the PM call and the hurdle the PM is told about, so they cannot drift apart."""

    def test_desk_prices_the_hurdle_at_its_own_pm_budget(self) -> None:
        from argus.agents.desk import TradingDesk

        class _Silent:
            def complete_json(self, *a, **k):  # pragma: no cover - never reached
                raise AssertionError("no model call expected")

        low = TradingDesk(_Silent(), pm_thinking=Thinking.LOW)  # type: ignore[arg-type]
        full = TradingDesk(_Silent(), pm_thinking=Thinking.FULL)  # type: ignore[arg-type]
        assert low.pm.thinking is Thinking.LOW
        assert full.pm.thinking is Thinking.FULL

        session = SessionState(
            as_of=datetime(2026, 9, 13, 3, 0, tzinfo=UTC),
            phase=SessionPhase.WEEKEND, hours_to_next_discovery=30.5,
        )
        vol = Decimal("0.45")
        h_low = deliberation_cost_bps(session, thinking=low.pm_thinking, annualised_vol=vol)
        h_full = deliberation_cost_bps(session, thinking=full.pm_thinking, annualised_vol=vol)
        assert h_low < h_full
        assert Decimal("12") + h_low < Decimal("20"), "LOW must bring the weekend hurdle under 20"
        assert Decimal("12") + h_full > Decimal("27"), "FULL weekend hurdle exceeds the fee"

    def test_the_runner_uses_low_for_routine_cycles(self) -> None:
        import pathlib

        src = pathlib.Path(__file__).resolve().parents[1] / "src/argus/paper/runner.py"
        assert "pm_thinking=Thinking.LOW" in src.read_text(encoding="utf-8")


class TestTheLeanIsAskedForAndParsed:
    """A view the desk states while refusing to act on it.

    The `side` field looked like this and was not: over the 53 settled abstentions on the live
    ledger it was BUY in all 53, directionally right 3.8% of the time against a base rate of
    up-moves of exactly 3.8%. That is a schema being filled in, not a judgement — so the lean is
    asked for separately, normalised here rather than trusted, and graded by `argus.eval.shadow`.
    """

    def test_an_abstention_still_carries_a_direction(self) -> None:
        """The point of the field. A refusal that states no view can never be scored, and a desk
        that cannot be scored while abstaining cannot show its abstentions were right."""
        got = _to_intent(
            {"verdict": "NO_TRADE", "side": "BUY", "quantity": 0, "confidence": 0.3,
             "thesis": "edge does not clear the hurdle", "lean": "DOWN", "lean_confidence": 0.62},
            "rNVDA",
        )
        assert got.verdict.is_abstention
        assert got.quantity == Decimal("0")
        assert got.lean == "down"
        assert got.lean_confidence == pytest.approx(0.62)

    def test_a_response_with_no_lean_declines_rather_than_defaulting_to_a_direction(self) -> None:
        got = _to_intent(
            {"verdict": "NO_TRADE", "side": "BUY", "quantity": 0, "confidence": 0.3,
             "thesis": "no view"},
            "rNVDA",
        )
        assert got.lean == "none"
        assert got.lean_confidence == 0.0

    def test_an_unrecognised_lean_becomes_none_rather_than_being_guessed(self) -> None:
        """'sideways', 'flat', 'neutral' and a hallucinated fourth option all mean the same thing
        for grading: nothing to score. Mapping any of them onto up or down would invent a call."""
        for raw in ("sideways", "flat", "neutral", "BUY", ""):
            assert _to_intent(
                {"verdict": "NO_TRADE", "quantity": 0, "thesis": "x", "lean": raw}, "rNVDA"
            ).lean == "none"

    def test_case_and_whitespace_are_normalised(self) -> None:
        assert _to_intent(
            {"verdict": "NO_TRADE", "quantity": 0, "thesis": "x", "lean": "  Up  "}, "rNVDA"
        ).lean == "up"

    def test_a_confidence_outside_the_unit_interval_is_clamped(self) -> None:
        """A model that answers 85 when asked for a probability has answered 0.85 in the wrong
        units, or has not answered at all. Either way it must not enter the record as 85."""
        high = _to_intent(
            {"verdict": "NO_TRADE", "quantity": 0, "thesis": "x", "lean": "up",
             "lean_confidence": 85}, "rNVDA",
        )
        low = _to_intent(
            {"verdict": "NO_TRADE", "quantity": 0, "thesis": "x", "lean": "up",
             "lean_confidence": -2}, "rNVDA",
        )
        assert high.lean_confidence == 1.0
        assert low.lean_confidence == 0.0

    def test_a_non_numeric_confidence_is_zero_rather_than_a_crash(self) -> None:
        got = _to_intent(
            {"verdict": "NO_TRADE", "quantity": 0, "thesis": "x", "lean": "up",
             "lean_confidence": "quite sure"}, "rNVDA",
        )
        assert got.lean_confidence == 0.0

    def test_a_trade_carries_a_lean_too(self) -> None:
        got = _to_intent(
            {"verdict": "TRADE", "side": "SELL", "quantity": 100, "confidence": 0.8,
             "thesis": "guidance cut not priced", "invalidation": ["reaffirmed at the open"],
             "lean": "down", "lean_confidence": 0.8},
            "rNVDA",
        )
        assert got.verdict is Verdict.TRADE
        assert got.lean == "down"

    def test_the_prompt_asks_for_it_and_says_why(self) -> None:
        """If the field is in the schema but the prompt never explains that declining to trade is
        not declining to have a view, the model fills it in the way it filled in `side`."""
        from argus.agents.meta_pm import SYSTEM_PROMPT

        assert '"lean"' in SYSTEM_PROMPT
        assert "lean_confidence" in SYSTEM_PROMPT
        assert "NOT saying you have no view" in SYSTEM_PROMPT


class TestTheLeanCannotBeEditedAfterTheFact:
    """A directional call that could be rewritten once the move is known is worth nothing.

    The ledger does not hash the lean directly — adding a field to `Entry.content_hash` would rehash
    all 161 existing rows and break a chain already anchored. It does not need to: `hash_intent`
    hashes the whole Intent through `asdict`, and that hash *is* in the row's payload. These tests
    assert the claim rather than repeating it.
    """

    def test_changing_the_lean_changes_the_intent_hash(self) -> None:
        up = Intent(
            symbol="rNVDA", side=Side.BUY, quantity=Decimal("0"), verdict=Verdict.NO_TRADE,
            stated_confidence=0.3, thesis="x", invalidation=(), lean="up", lean_confidence=0.6,
        )
        down = Intent(
            symbol="rNVDA", side=Side.BUY, quantity=Decimal("0"), verdict=Verdict.NO_TRADE,
            stated_confidence=0.3, thesis="x", invalidation=(), lean="down", lean_confidence=0.6,
        )
        assert hash_intent(up) != hash_intent(down)

    def test_changing_only_the_lean_confidence_changes_it_too(self) -> None:
        base = Intent(
            symbol="rNVDA", side=Side.BUY, quantity=Decimal("0"), verdict=Verdict.NO_TRADE,
            stated_confidence=0.3, thesis="x", invalidation=(), lean="up", lean_confidence=0.6,
        )
        louder = Intent(
            symbol="rNVDA", side=Side.BUY, quantity=Decimal("0"), verdict=Verdict.NO_TRADE,
            stated_confidence=0.3, thesis="x", invalidation=(), lean="up", lean_confidence=0.9,
        )
        assert hash_intent(base) != hash_intent(louder)

    def test_an_intent_written_before_the_lean_existed_still_hashes(self) -> None:
        """Defaults exist so the 161 rows already on the chain keep loading. If this ever raised,
        the whole history would be unreadable."""
        legacy = Intent(
            symbol="rNVDA", side=Side.BUY, quantity=Decimal("0"), verdict=Verdict.NO_TRADE,
            stated_confidence=0.3, thesis="x", invalidation=(),
        )
        assert legacy.lean == "none"
        assert len(hash_intent(legacy)) == 16


class TestAMalformedVerdictIsADecisionNotACrash:
    """The live model put a *side* in the verdict field and took the whole cycle down.

    `_to_intent` called `Verdict("sell")` directly and `ValueError: 'sell' is not a valid Verdict`
    propagated out of `MetaPM.decide`, out of `desk.run`, and killed the run. In the paper runner
    that is a lost symbol; in the flow demo it was a lost demonstration. A malformed answer is
    unactionable, which is a decision outcome the system already has a state for.
    """

    def test_an_unknown_verdict_routes_to_human_review(self) -> None:
        got = _to_intent(
            {"verdict": "sell", "side": "SELL", "quantity": 2, "confidence": 0.8,
             "thesis": "guidance cut", "invalidation": ["reaffirmed"]},
            "NVDAUSDT",
        )
        assert got.verdict is Verdict.HUMAN_REVIEW

    def test_it_carries_no_quantity(self) -> None:
        """An unparseable verdict must not reach the risk layer carrying size."""
        got = _to_intent(
            {"verdict": "sell", "side": "SELL", "quantity": 999, "thesis": "x",
             "invalidation": ["y"]},
            "NVDAUSDT",
        )
        assert got.quantity == Decimal("0")

    def test_the_raw_value_is_recorded_rather_than_swallowed(self) -> None:
        """A trail that shows HUMAN_REVIEW without saying why cannot be debugged."""
        got = _to_intent({"verdict": "sell", "thesis": "guidance cut"}, "NVDAUSDT")
        assert "unparseable verdict" in got.thesis
        assert "'sell'" in got.thesis
        assert "guidance cut" in got.thesis

    def test_a_missing_verdict_is_handled_the_same_way(self) -> None:
        got = _to_intent({"thesis": "no verdict key at all"}, "NVDAUSDT")
        assert got.verdict is Verdict.HUMAN_REVIEW
        assert "(empty)" in got.thesis

    def test_it_is_not_guessed_into_a_trade_or_a_refusal(self) -> None:
        """Mapping "sell" to TRADE invents a decision; mapping it to NO_TRADE invents a refusal.
        Neither is what the model said, and HUMAN_REVIEW is the honest third answer."""
        got = _to_intent({"verdict": "sell", "side": "SELL", "quantity": 2}, "NVDAUSDT")
        assert got.verdict not in {Verdict.TRADE, Verdict.NO_TRADE, Verdict.DATA_INSUFFICIENT}

    def test_every_real_verdict_still_parses(self) -> None:
        for verdict in Verdict:
            got = _to_intent(
                {"verdict": verdict.value, "side": "BUY", "quantity": 1, "thesis": "t",
                 "invalidation": ["i"]},
                "NVDAUSDT",
            )
            assert "unparseable" not in got.thesis, verdict
