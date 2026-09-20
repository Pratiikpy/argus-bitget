"""The standing adversary: it may falsify a decision, and may never author one."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from argus.agents.adversary import (
    REDUCE_FLOOR,
    SEVERE,
    SYSTEM_PROMPT,
    Outcome,
    build_critic,
    challenge,
    explain,
)
from argus.decision.verdicts import Intent, Side, Verdict
from argus.truth.evidence import Evidence

AT = datetime(2026, 9, 14, 15, 0, tzinfo=UTC)


def _intent(**over: Any) -> Intent:
    base: dict[str, Any] = {
        "symbol": "NVDAUSDT", "side": Side.BUY, "quantity": Decimal("10"),
        "verdict": Verdict.TRADE, "stated_confidence": 0.8,
        "thesis": "guidance raised above consensus",
        "invalidation": ("the guidance is withdrawn", "the print is restated"),
    }
    base.update(over)
    return Intent(**base)


def _evidence() -> list[Evidence]:
    return [
        Evidence(id="e1", claim="Q3 guidance raised 2% above consensus", source="sec-edgar",
                 available_at=AT, credibility=1.0),
    ]


class FakeCritic:
    def __init__(self, payload: dict[str, Any] | Exception) -> None:
        self.payload = payload
        self.calls: list[list[dict[str, Any]]] = []

    def complete_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.calls.append(messages)
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def _payload(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "counter_case": "the move may already be priced",
        "weakest_link": "the guidance figure",
        "severity": 0.1,
        "already_refuted": False,
        "refuted_condition": "",
    }
    base.update(over)
    return base


class TestItOnlyEverNarrows:
    def test_a_sound_decision_is_untouched(self) -> None:
        ruling, record = challenge(_intent(), _evidence(), critic=FakeCritic(_payload()))
        assert ruling is None and record.outcome is Outcome.UPHELD

    def test_a_counter_case_is_recorded_even_when_the_decision_stands(self) -> None:
        _, record = challenge(_intent(), _evidence(), critic=FakeCritic(_payload()))
        assert record.counter_case and record.weakest_link

    def test_moderate_doubt_halves_the_position(self) -> None:
        ruling, record = challenge(
            _intent(), _evidence(), critic=FakeCritic(_payload(severity=0.5))
        )
        assert record.outcome is Outcome.REDUCED
        assert ruling is not None
        assert ruling.resulting_intent.quantity == Decimal("5")

    def test_severe_doubt_refuses_the_trade(self) -> None:
        ruling, record = challenge(
            _intent(), _evidence(), critic=FakeCritic(_payload(severity=0.95))
        )
        assert record.outcome is Outcome.REFUSED
        assert ruling is not None and ruling.resulting_intent.quantity == Decimal("0")

    def test_it_can_never_enlarge_a_position(self) -> None:
        for severity in (0.0, 0.2, REDUCE_FLOOR, 0.6, SEVERE, 1.0):
            ruling, _ = challenge(
                _intent(), _evidence(), critic=FakeCritic(_payload(severity=severity))
            )
            got = ruling.resulting_intent.quantity if ruling else Decimal("10")
            assert got <= Decimal("10"), severity

    def test_it_can_never_reverse_a_side(self) -> None:
        for severity in (0.1, 0.5, 0.9):
            ruling, _ = challenge(
                _intent(), _evidence(), critic=FakeCritic(_payload(severity=severity))
            )
            if ruling is not None and ruling.resulting_intent.quantity > 0:
                assert ruling.resulting_intent.side is Side.BUY

    def test_it_never_runs_on_a_decision_that_opens_nothing(self) -> None:
        flat = _intent(verdict=Verdict.NO_TRADE, quantity=Decimal("0"), invalidation=())
        critic = FakeCritic(_payload(severity=0.9))
        ruling, record = challenge(flat, _evidence(), critic=critic)
        assert ruling is None and record.outcome is Outcome.NOT_RUN
        assert critic.calls == [], "an abstention must not cost a model call"

    def test_a_position_too_small_to_halve_is_refused_rather_than_rounded_up(self) -> None:
        tiny = _intent(quantity=Decimal("0.00000001"))
        ruling, _record = challenge(
            tiny, _evidence(), critic=FakeCritic(_payload(severity=0.5))
        )
        assert ruling is not None
        assert ruling.resulting_intent.quantity <= tiny.quantity


class TestSelfRefutation:
    def test_a_falsifier_the_thesis_stated_and_is_already_true_kills_the_trade(self) -> None:
        """The check no reference system performs."""
        ruling, record = challenge(
            _intent(), _evidence(),
            critic=FakeCritic(_payload(
                already_refuted=True, refuted_condition="the guidance is withdrawn"
            )),
        )
        assert record.outcome is Outcome.SELF_REFUTED
        assert ruling is not None and ruling.resulting_intent.quantity == Decimal("0")
        assert ruling.binding_constraint == "adversary_self_refuted"

    def test_a_falsifier_the_thesis_never_stated_is_ignored(self) -> None:
        """Otherwise a critic could veto anything by inventing a condition."""
        ruling, record = challenge(
            _intent(), _evidence(),
            critic=FakeCritic(_payload(
                already_refuted=True, refuted_condition="mars is in retrograde"
            )),
        )
        assert ruling is None and record.outcome is Outcome.UPHELD

    def test_a_refutation_claim_with_no_condition_named_is_ignored(self) -> None:
        ruling, record = challenge(
            _intent(), _evidence(),
            critic=FakeCritic(_payload(already_refuted=True, refuted_condition="")),
        )
        assert ruling is None and record.outcome is Outcome.UPHELD

    def test_a_partial_match_against_a_stated_condition_counts(self) -> None:
        ruling, _ = challenge(
            _intent(), _evidence(),
            critic=FakeCritic(_payload(
                already_refuted=True, refuted_condition="the print is restated by the company"
            )),
        )
        assert ruling is not None and ruling.binding_constraint == "adversary_self_refuted"

    def test_the_rendered_line_names_the_condition(self) -> None:
        _, record = challenge(
            _intent(), _evidence(),
            critic=FakeCritic(_payload(
                already_refuted=True, refuted_condition="the guidance is withdrawn"
            )),
        )
        assert "already refuted" in record.render()
        assert "guidance is withdrawn" in record.render()


class TestItNeverTakesTheDeskDown:
    def test_no_critic_leaves_the_decision_alone(self) -> None:
        ruling, record = challenge(_intent(), _evidence(), critic=None)
        assert ruling is None and record.outcome is Outcome.NOT_RUN
        assert "no adversary is configured" in record.detail

    def test_a_failing_critic_leaves_the_decision_alone_and_says_so(self) -> None:
        ruling, record = challenge(
            _intent(), _evidence(), critic=FakeCritic(RuntimeError("endpoint down"))
        )
        assert ruling is None and record.outcome is Outcome.UNAVAILABLE
        assert "RuntimeError" in record.detail
        assert "unchallenged" in record.render()

    def test_an_unreadable_severity_is_treated_as_no_objection(self) -> None:
        ruling, record = challenge(
            _intent(), _evidence(), critic=FakeCritic(_payload(severity="very"))
        )
        assert ruling is None and record.severity == 0.0

    def test_a_severity_above_one_is_clamped(self) -> None:
        _, record = challenge(
            _intent(), _evidence(), critic=FakeCritic(_payload(severity=9.0))
        )
        assert record.severity == 1.0

    def test_build_critic_never_raises_without_credentials(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.delenv("BITGET_QWEN_API_KEY", raising=False)
        assert build_critic() is None


class TestWhatTheCriticIsShown:
    def test_it_is_given_the_thesis_and_its_falsifiers(self) -> None:
        critic = FakeCritic(_payload())
        challenge(_intent(), _evidence(), critic=critic)
        body = critic.calls[0][-1]["content"]
        assert "guidance raised above consensus" in body
        assert "the guidance is withdrawn" in body
        assert "the print is restated" in body

    def test_it_is_given_the_same_evidence(self) -> None:
        critic = FakeCritic(_payload())
        challenge(_intent(), _evidence(), critic=critic)
        assert "2% above consensus" in critic.calls[0][-1]["content"]

    def test_the_evidence_shown_is_bounded(self) -> None:
        many = [
            Evidence(id=f"e{i}", claim=f"item {i}", source="news", available_at=AT,
                     credibility=0.5)
            for i in range(200)
        ]
        critic = FakeCritic(_payload())
        challenge(_intent(), many, critic=critic)
        assert critic.calls[0][-1]["content"].count("item ") <= 30

    def test_the_prompt_forbids_arguing_a_side(self) -> None:
        assert "NOT arguing a side" in SYSTEM_PROMPT
        assert "not the bear" in SYSTEM_PROMPT.lower()

    def test_the_prompt_warns_against_manufactured_objections(self) -> None:
        assert "manufactured objection" in SYSTEM_PROMPT


def test_a_challenge_serialises() -> None:
    _, record = challenge(_intent(), _evidence(), critic=FakeCritic(_payload()))
    got = record.as_dict()
    assert got["outcome"] == "upheld"
    assert "counter_case" in explain(record)


@pytest.mark.parametrize("outcome", list(Outcome))
def test_every_outcome_knows_whether_it_changed_the_decision(outcome: Outcome) -> None:
    changed = outcome.changed_the_decision
    assert changed == (outcome in {Outcome.REDUCED, Outcome.REFUSED, Outcome.SELF_REFUTED})
