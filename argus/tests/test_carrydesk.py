"""Tests for the carry desk.

Two things have to be true for this module to mean anything, and they pull in opposite directions:
the Constitution chain must actually be **reachable** past gate one, and the confidence handed to it
must be the one computed from independent windows rather than overlapping ones. A test suite that
only checked the second would be satisfied by a chain that is still a wall.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from argus.decision.verdicts import ConstitutionVerdict, Side
from argus.desk.carrydesk import (
    assess,
    carry_intents,
    honest_confidence,
    walk_chain,
    wilson_lower,
)
from argus.eval.autopsy import CHAIN, FIRED, PASSED, UNREACHED
from argus.truth.clocks import DualClock

NOW = datetime(2026, 9, 14, 18, tzinfo=UTC)
SESSION = DualClock().state(NOW, nav_age_seconds=60.0)


def basket(*, share_positive: float = 0.74, windows: int = 1823) -> dict:
    return {
        "position": "short 1.000 TQQQUSDT + short 0.969 SQQQUSDT",
        "legs": [
            {"symbol": "TQQQUSDT", "weight": -1.0},
            {"symbol": "SQQQUSDT", "weight": -0.9688},
        ],
        "replay": {
            "holding_days": 14, "windows": windows, "mean_total_pct": 0.221,
            "mean_funding_pct": 0.046, "mean_price_pct": 0.295,
            "share_positive": share_positive, "worst_pct": -3.30, "best_pct": 3.0,
            "funding_survives_the_price": True,
        },
    }


class TestTheWilsonBound:
    def test_it_stays_inside_the_unit_interval_at_tiny_samples(self) -> None:
        """The reason Wilson is used rather than the normal approximation: at six trials the normal
        interval runs outside [0, 1], and a confidence outside [0, 1] is rejected by `Intent`."""
        for successes in range(7):
            got = wilson_lower(successes, 6)
            assert 0.0 <= got <= 1.0

    def test_certainty_on_a_small_sample_is_still_not_certainty(self) -> None:
        assert wilson_lower(6, 6) < 1.0
        assert wilson_lower(6, 6) > wilson_lower(4, 6)

    def test_more_trials_at_the_same_rate_raise_the_bound(self) -> None:
        assert wilson_lower(740, 1000) > wilson_lower(74, 100) > wilson_lower(7.4, 10)

    def test_no_trials_is_no_confidence(self) -> None:
        assert wilson_lower(0, 0) == 0.0

    def test_zero_successes_is_zero(self) -> None:
        assert wilson_lower(0, 20) == 0.0


class TestTheOverlapCorrectionChangesTheAnswer:
    def test_the_independent_bound_is_far_below_the_naive_one(self) -> None:
        """The whole point. Quoting 1,823 overlapping windows would clear the 0.55 floor; quoting
        the ~6 independent ones does not, and the position is refused."""
        confidence, independent, workings = honest_confidence(basket(), span_days=90.0)
        naive = wilson_lower(0.74 * 1823, 1823)
        assert independent == pytest.approx(90 / 14)
        assert confidence < 0.55 < naive
        assert "independent" in workings

    def test_the_workings_quote_both_numbers(self) -> None:
        _confidence, _independent, workings = honest_confidence(basket(), span_days=90.0)
        assert "0.37" in workings and "0.72" in workings

    def test_a_longer_record_earns_a_higher_bound(self) -> None:
        """Not a knob: more history is more independent windows, which is the only honest way this
        confidence rises."""
        short, _n1, _w1 = honest_confidence(basket(), span_days=90.0)
        long, _n2, _w2 = honest_confidence(basket(), span_days=1000.0)
        assert long > short


class TestReachability:
    def test_a_fired_gate_leaves_everything_after_it_unreached(self) -> None:
        from argus.decision.verdicts import ConstitutionRuling, Intent, Verdict

        intent = Intent(
            symbol="TQQQUSDT", side=Side.SELL, quantity=Decimal("1000"),
            verdict=Verdict.TRADE, stated_confidence=0.37, thesis="t",
            invalidation=("x",),
        )
        ruling = ConstitutionRuling(
            verdict=ConstitutionVerdict.REJECT, binding_constraint="min_confidence",
            reason="below the floor", resulting_intent=intent,
        )
        gates = walk_chain(ruling)
        assert [g.status for g in gates] == [PASSED, FIRED] + [UNREACHED] * (len(CHAIN) - 2)

    def test_a_ceiling_binder_leaves_gates_after_it_passed_not_unreached(self) -> None:
        """The Constitution evaluates every ceiling and takes the minimum — a ceiling binder does
        not short-circuit the gates after it in source order. This exact case was untested here
        before the 2026-09-15 restructure, and `walk_chain` carried the old short-circuit-only
        assumption silently until the full suite caught it against `execution/review.py`'s
        equivalent test."""
        from argus.decision.verdicts import ConstitutionRuling, Intent, Verdict

        intent = Intent(
            symbol="TQQQUSDT", side=Side.SELL, quantity=Decimal("1000"),
            verdict=Verdict.TRADE, stated_confidence=0.9, thesis="t", invalidation=("x",),
        )
        ruling = ConstitutionRuling(
            verdict=ConstitutionVerdict.RESIZE, binding_constraint="unhedgeable_gap",
            reason="ceiling bound", resulting_intent=intent,
        )
        gates = walk_chain(ruling)
        statuses = [g.status for g in gates]
        binder_index = next(i for i, g in enumerate(CHAIN) if g.name == "unhedgeable_gap")
        assert statuses[binder_index] == FIRED
        assert statuses[:binder_index] == [PASSED] * binder_index
        assert statuses[binder_index + 1:] == [PASSED] * (len(CHAIN) - binder_index - 1)

    def test_the_chain_matches_the_constitution_in_source_order(self) -> None:
        assert [g.name for g in CHAIN] == [
            "no_exposure", "min_confidence", "oracle_stale", "unhedgeable_gap", "gross_exposure",
            "signed_exposure", "hedge_integrity", "margin_usage", "factor_exposure",
            "scenario_loss", "liquidation_cost", "per_symbol_underperformance", "risk_budget",
            "session_volatility", "max_position",
        ]


class TestTheProposalItself:
    def test_a_negative_weight_becomes_a_sell(self) -> None:
        intents = carry_intents(basket(), confidence=0.37)
        assert [i.side for i in intents] == [Side.SELL, Side.SELL]

    def test_quantities_follow_the_basket_weights(self) -> None:
        intents = carry_intents(basket(), confidence=0.37, notional=Decimal("1000"))
        assert intents[0].quantity == Decimal("1000.00")
        assert intents[1].quantity == Decimal("968.80")

    def test_every_leg_names_the_others_as_its_required_hedge(self) -> None:
        intents = carry_intents(basket(), confidence=0.37)
        assert intents[0].required_hedge == ("SQQQUSDT",)
        assert intents[1].required_hedge == ("TQQQUSDT",)

    def test_every_leg_carries_a_falsifier(self) -> None:
        """Mandatory for anything opening exposure: a position with no falsifier can only be hoped
        over. The study's own worst window is one of them, so the position is refutable by the
        number it was sized on."""
        for intent in carry_intents(basket(), confidence=0.37):
            assert len(intent.invalidation) >= 3
            assert any("-3.30%" in reason for reason in intent.invalidation)

    def test_the_desk_states_no_lean(self) -> None:
        """A carry has no directional view, and the lean field exists so that can be said rather
        than inferred from a side the schema forced."""
        assert all(i.lean == "none" for i in carry_intents(basket(), confidence=0.37))


class TestTheEndToEndRun:
    def test_gate_one_passes_and_gate_two_refuses(self) -> None:
        report = assess(basket(), session=SESSION, span_days=90.0)
        assert report["gates_reached_beyond_exposure"] == ["min_confidence"]
        for leg in report["legs"]:
            assert leg["gates"][0]["status"] == PASSED
            assert leg["gates"][1]["status"] == FIRED
            assert leg["reached_beyond_exposure"]
            assert leg["verdict"] == "reject"

    def test_the_chain_is_not_a_wall(self) -> None:
        """The test that keeps the previous one honest. If gate two refused everything, "reachable"
        would be a word rather than a property — so a proposal confident enough to clear the floor
        must reach further, and this proves the gates behind it execute."""
        confident = basket(share_positive=1.0)
        confident["replay"]["windows"] = 100_000
        report = assess(confident, session=SESSION, span_days=100_000.0)
        assert report["stated_confidence"] > 0.55
        reached = report["gates_reached_beyond_exposure"]
        assert "min_confidence" in reached
        assert len(reached) > 1, "nothing past the confidence floor ever ran"

    def test_the_verdict_reports_the_unreached_gates_as_unreached(self) -> None:
        report = assess(basket(), session=SESSION, span_days=90.0)
        assert "unreached" in report["verdict"]
        assert "for the first time" in report["verdict"]

    def test_the_confidence_the_floor_saw_is_the_one_recorded(self) -> None:
        """No gap between the number in the report and the number handed to the rule."""
        report = assess(basket(), session=SESSION, span_days=90.0)
        for leg in report["legs"]:
            assert leg["stated_confidence"] == report["stated_confidence"]
