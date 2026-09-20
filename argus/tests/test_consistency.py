"""Consistency tests — the scorer must be able to report disagreement, so it is made to.

The live measurement found 80% action agreement and 100% lean agreement. A scorer that always
returned 100% would look identical, so every level below is driven to a failing value on purpose.
The cache guard gets the most attention: with the client's cache left on, replays two onwards are
served from replay one and agree perfectly having asked the model nothing.
"""

from __future__ import annotations

import pytest

from argus.eval.consistency import (
    MIN_RUNS,
    ConsistencyError,
    Decision,
    Replay,
    Report,
    decision_of,
    replay,
    report_from,
)


def _decision(
    verdict: str = "no_trade", quantity: str = "0", lean: str = "down", thesis: str = "t",
    intent_hash: str = "h",
) -> Decision:
    return Decision(
        verdict=verdict, side="sell", quantity=quantity, lean=lean,
        lean_confidence=0.8, intent_hash=intent_hash, thesis=thesis,
    )


def _replay(decisions: tuple[Decision, ...], *, cached: bool = False) -> Replay:
    return Replay(label="s", decisions=decisions, cached=cached)


class TestTheScorerCanReportDisagreement:
    def test_identical_decisions_agree_completely(self) -> None:
        got = _replay(tuple(_decision() for _ in range(3)))
        assert got.action_agreement == 1.0
        assert got.lean_agreement == 1.0
        assert got.verbatim_agreement == 1.0
        assert got.unanimous_action

    def test_a_different_action_lowers_action_agreement(self) -> None:
        got = _replay((_decision(), _decision(), _decision(verdict="reduce", quantity="2")))
        assert got.action_agreement == pytest.approx(2 / 3)
        assert not got.unanimous_action

    def test_agreement_is_modal_not_pairwise(self) -> None:
        """Three answers, two the same: two of three agree, not one of three pairs."""
        got = _replay((_decision(), _decision(), _decision(verdict="human_review")))
        assert got.action_agreement == pytest.approx(2 / 3)

    def test_every_answer_different_gives_the_floor(self) -> None:
        got = _replay((
            _decision(verdict="no_trade"),
            _decision(verdict="reduce"),
            _decision(verdict="human_review"),
        ))
        assert got.action_agreement == pytest.approx(1 / 3)

    def test_distinct_actions_are_listed_for_the_reader(self) -> None:
        got = _replay((_decision(), _decision(verdict="reduce", quantity="2"), _decision()))
        assert len(got.distinct_actions) == 2


class TestTheThreeLevelsAreSeparate:
    def test_a_stable_lean_survives_an_unstable_action(self) -> None:
        """The live finding: the desk knew what it thought and wavered on what to do about it.
        A single 'consistency' number would have averaged that away."""
        got = _replay((
            _decision(verdict="no_trade", lean="down"),
            _decision(verdict="reduce", quantity="2", lean="down"),
            _decision(verdict="no_trade", lean="down"),
        ))
        assert got.lean_agreement == 1.0
        assert got.action_agreement == pytest.approx(2 / 3)

    def test_wording_can_vary_while_the_action_is_pinned(self) -> None:
        """Verbatim identity is the strictest level and the least important one: a desk that always
        sells the same size and phrases it differently is consistent where it matters."""
        got = _replay((
            _decision(intent_hash="a"), _decision(intent_hash="b"), _decision(intent_hash="c"),
        ))
        assert got.action_agreement == 1.0
        assert got.verbatim_agreement == pytest.approx(1 / 3)

    def test_the_verdict_names_the_stable_view_when_the_action_moves(self) -> None:
        got = report_from([_replay((
            _decision(verdict="no_trade"), _decision(verdict="reduce"), _decision(),
        ))])
        assert "knows what it thinks" in got.verdict

    def test_it_does_not_claim_a_stable_view_when_the_lean_moves_too(self) -> None:
        got = report_from([_replay((
            _decision(verdict="no_trade", lean="down"),
            _decision(verdict="reduce", lean="up"),
            _decision(lean="none"),
        ))])
        assert "knows what it thinks" not in got.verdict


class TestTheCacheIsNotTheDesk:
    """With the client's cache on, replay two onwards are served from replay one."""

    def test_a_cached_replay_is_excluded_from_the_headline(self) -> None:
        got = report_from([_replay(tuple(_decision() for _ in range(3)), cached=True)])
        assert got.sampled == ()
        assert got.unanimity is None

    def test_the_verdict_refuses_to_speak_for_the_desk_from_a_cached_run(self) -> None:
        got = report_from([_replay(tuple(_decision() for _ in range(3)), cached=True)])
        assert "UNDEFINED as a statement about the desk" in got.verdict
        assert "the agreement measured is the cache's" in got.verdict

    def test_an_uncached_replay_counts(self) -> None:
        got = report_from([_replay(tuple(_decision() for _ in range(3)), cached=False)])
        assert len(got.sampled) == 1
        assert got.unanimity == 1.0

    def test_the_live_scenario_disables_the_cache(self) -> None:
        """If this ever ran with the cache on, the measurement would be of the dict."""
        import inspect

        from argus.eval import consistency

        source = inspect.getsource(consistency.scenario_run)
        assert "cache=False" in source


class TestItRefusesTooFewRuns:
    def test_two_replays_raise(self) -> None:
        with pytest.raises(ConsistencyError, match="below the floor"):
            replay(lambda i: None, label="s", runs=2)

    def test_agreement_is_none_below_the_floor(self) -> None:
        got = _replay(tuple(_decision() for _ in range(MIN_RUNS - 1)))
        assert got.action_agreement is None
        assert got.lean_agreement is None

    def test_the_floor_is_at_least_three(self) -> None:
        """Two runs report 0% or 100% and nothing between."""
        assert MIN_RUNS >= 3

    def test_an_empty_report_is_undefined(self) -> None:
        assert "UNDEFINED" in Report(replays=()).verdict


class TestReadingADeskRun:
    def test_it_reads_the_approved_intent_not_the_proposal(self) -> None:
        """On a constrained decision the original is a proposal the system did not act on."""
        from decimal import Decimal

        from argus.decision.verdicts import Intent, Side, Verdict
        from argus.proof.autonomy import AutonomyProof, hash_intent

        original = Intent(
            symbol="X", side=Side.SELL, quantity=Decimal("10"), verdict=Verdict.TRADE,
            stated_confidence=0.8, thesis="original", invalidation=("i",), lean="down",
        )
        revised = Intent(
            symbol="X", side=Side.SELL, quantity=Decimal("2"), verdict=Verdict.REDUCE,
            stated_confidence=0.8, thesis="revised", invalidation=("i",), lean="down",
        )
        proof = AutonomyProof(
            decision_id="d", as_of=__import__("datetime").datetime.now(
                __import__("datetime").UTC
            ),
            market_state_hash="m", llm_original_intent=original, llm_original_reasoning="r",
        )
        proof.llm_revised_intent = revised
        proof.approved_intent_hash = hash_intent(revised)

        got = decision_of(type("Run", (), {"proof": proof})())
        assert got.quantity == "2"
        assert got.thesis == "revised"

    def test_the_lean_comes_from_the_original_intent(self) -> None:
        """The risk layer resizes and refuses; it does not form opinions, so the lean is the
        model's own and is read from the intent it first stated."""
        import inspect

        from argus.eval import consistency

        source = inspect.getsource(consistency.decision_of)
        assert "original" in source


class TestTheReport:
    def test_it_serialises_with_every_level(self) -> None:
        import json

        blob = json.loads(json.dumps(report_from([
            _replay(tuple(_decision() for _ in range(3)))
        ]).as_dict()))
        first = blob["replays"][0]
        assert {"lean_agreement", "action_agreement", "verbatim_agreement"} <= set(first)

    def test_the_rendered_report_shows_all_three(self) -> None:
        text = report_from([_replay(tuple(_decision() for _ in range(3)))]).render()
        assert "lean" in text and "action" in text and "verbatim" in text

    def test_a_state_that_moved_is_named_rather_than_averaged(self) -> None:
        got = report_from([Replay(
            label="sleeping-anchor", cached=False,
            decisions=(_decision(), _decision(verdict="reduce"), _decision()),
        )])
        assert "sleeping-anchor" in got.verdict
        assert "named rather than averaged away" in got.verdict
