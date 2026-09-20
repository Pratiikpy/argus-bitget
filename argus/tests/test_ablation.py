"""Ablation: the harness must be able to say 'remove it' and must refuse to guess."""

from __future__ import annotations

import math

import pytest

from argus.eval.ablation import (
    MIN_PAIRS,
    SIGNIFICANT,
    Outcome,
    Verdict,
    deterministic,
    pair_outcomes,
    paired,
    sign_test,
)


def _outcomes(deltas: list[float], *, base: float = 0.0) -> list[Outcome]:
    """One pair per delta: the 'with' arm beats 'without' by exactly that much."""
    out: list[Outcome] = []
    for i, d in enumerate(deltas):
        out.append(Outcome(frame_id=f"f{i}", arm="with", value=base + d))
        out.append(Outcome(frame_id=f"f{i}", arm="without", value=base))
    return out


class TestTheSignTestIsExact:
    def test_a_coin_split_is_not_significant(self) -> None:
        assert sign_test(15, 15) == 1.0

    def test_no_trials_is_no_evidence_not_an_error(self) -> None:
        assert sign_test(0, 0) == 1.0

    def test_it_matches_the_binomial_by_hand(self) -> None:
        """5 of 5 one-sided is 1/32; two-sided doubles it."""
        assert sign_test(5, 0) == pytest.approx(2 * (1 / 32))

    def test_the_documented_minimum_actually_reaches_significance(self) -> None:
        """MIN_PAIRS is defended in the docstring by this exact number. Check it."""
        assert sign_test(21, 9) < SIGNIFICANT
        assert sign_test(20, 10) > SIGNIFICANT

    def test_it_is_symmetric_in_its_arguments(self) -> None:
        assert sign_test(21, 9) == sign_test(9, 21)

    def test_it_never_exceeds_one(self) -> None:
        for w in range(0, 12):
            for losses in range(0, 12):
                assert 0.0 <= sign_test(w, losses) <= 1.0

    def test_it_agrees_with_a_direct_summation(self) -> None:
        n, k = 17, 5
        expected = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
        assert sign_test(k, n - k) == pytest.approx(expected)


class TestItRefusesToAnswerOnTooLittleEvidence:
    def test_a_single_run_is_insufficient_not_a_finding(self) -> None:
        got = paired(_outcomes([5.0]), component="sentiment", metric="bps")
        assert got.verdict is Verdict.INSUFFICIENT

    def test_a_clean_sweep_below_the_minimum_is_still_insufficient(self) -> None:
        """Ten wins and no losses is p=0.002 and still refused. The threshold is on evidence
        volume, not on how pretty the split is."""
        got = paired(_outcomes([1.0] * 10), component="c", metric="bps")
        assert got.p_value < SIGNIFICANT
        assert got.verdict is Verdict.INSUFFICIENT

    def test_the_report_says_insufficient_is_not_no_effect(self) -> None:
        text = paired(_outcomes([1.0] * 3), component="c", metric="bps").render()
        assert "NOT a finding of no effect" in text

    def test_ties_do_not_count_toward_the_minimum(self) -> None:
        """Forty frames where the component changed nothing is not forty pieces of evidence."""
        got = paired(_outcomes([0.0] * 40), component="c", metric="bps")
        assert got.n == 40 and got.differing == 0
        assert got.verdict is Verdict.INSUFFICIENT

    def test_the_minimum_is_configurable_upward_and_reported(self) -> None:
        got = paired(_outcomes([1.0] * 40), component="c", metric="bps", minimum_pairs=100)
        assert got.verdict is Verdict.INSUFFICIENT and got.minimum_pairs == 100


class TestItCanSayRemoveIt:
    def test_a_component_that_helps_is_named_as_helping(self) -> None:
        got = paired(_outcomes([1.0] * 31), component="c", metric="bps")
        assert got.verdict is Verdict.HELPS

    def test_a_component_that_hurts_is_named_as_hurting(self) -> None:
        got = paired(_outcomes([-1.0] * 31), component="c", metric="bps")
        assert got.verdict is Verdict.HURTS

    def test_a_coin_flip_over_enough_pairs_is_no_effect(self) -> None:
        got = paired(_outcomes([1.0] * 20 + [-1.0] * 20), component="c", metric="bps")
        assert got.verdict is Verdict.NO_EFFECT
        assert got.differing == 40

    def test_no_effect_and_insufficient_are_never_the_same_verdict(self) -> None:
        small = paired(_outcomes([1.0, -1.0]), component="c", metric="bps")
        large = paired(_outcomes([1.0] * 20 + [-1.0] * 20), component="c", metric="bps")
        assert small.verdict is not large.verdict


class TestTheMagnitudeNeverOverridesTheDirection:
    def test_one_huge_win_does_not_make_a_losing_component_helpful(self) -> None:
        """The failure this harness exists to prevent: a positive mean carried by one frame."""
        deltas = [-1.0] * 30 + [1000.0]
        got = paired(_outcomes(deltas), component="c", metric="bps")
        assert got.mean_delta > 0
        assert got.verdict is Verdict.HURTS

    def test_a_result_carried_by_one_frame_is_flagged(self) -> None:
        got = paired(_outcomes([-1.0] * 30 + [1000.0]), component="c", metric="bps")
        assert got.carried_by_one_frame
        assert "carrying this result" in got.render()

    def test_a_broad_result_is_not_flagged(self) -> None:
        assert not paired(_outcomes([1.0] * 31), component="c", metric="bps").carried_by_one_frame

    def test_the_median_is_reported_next_to_the_mean(self) -> None:
        got = paired(_outcomes([-1.0] * 30 + [1000.0]), component="c", metric="bps")
        assert got.median_delta < 0 < got.mean_delta


class TestPairingIsByFrame:
    def test_frames_present_in_one_arm_only_are_reported_not_dropped(self) -> None:
        outcomes = [
            Outcome(frame_id="a", arm="with", value=1.0),
            Outcome(frame_id="a", arm="without", value=0.0),
            Outcome(frame_id="b", arm="with", value=99.0),
        ]
        pairs, orphaned = pair_outcomes(outcomes, with_arm="with", without_arm="without")
        assert len(pairs) == 1 and orphaned == ("b",)

    def test_an_arm_that_skipped_hard_frames_is_visible_in_the_report(self) -> None:
        outcomes = _outcomes([1.0] * 3)
        outcomes.append(Outcome(frame_id="crashed", arm="with", value=5.0))
        got = paired(outcomes, component="c", metric="bps")
        assert "ran in only one arm" in got.render()

    def test_pairing_does_not_depend_on_input_order(self) -> None:
        outcomes = _outcomes([1.0, -2.0, 3.0])
        forward = paired(outcomes, component="c", metric="bps")
        backward = paired(list(reversed(outcomes)), component="c", metric="bps")
        assert forward.as_dict() == backward.as_dict()


class TestLowerIsBetterMetrics:
    def test_a_cost_reduction_counts_as_helping(self) -> None:
        """With the component, cost is lower. On a lower-is-better metric that is a win."""
        outcomes: list[Outcome] = []
        for i in range(31):
            outcomes.append(Outcome(frame_id=f"f{i}", arm="with", value=8.0))
            outcomes.append(Outcome(frame_id=f"f{i}", arm="without", value=12.0))
        got = paired(outcomes, component="guard", metric="cost_bps", higher_is_better=False)
        assert got.verdict is Verdict.HELPS

    def test_forgetting_the_flag_would_have_reversed_the_verdict(self) -> None:
        outcomes: list[Outcome] = []
        for i in range(31):
            outcomes.append(Outcome(frame_id=f"f{i}", arm="with", value=8.0))
            outcomes.append(Outcome(frame_id=f"f{i}", arm="without", value=12.0))
        assert paired(
            outcomes, component="guard", metric="cost_bps", higher_is_better=True
        ).verdict is Verdict.HURTS


class TestTheDeterministicPath:
    def test_it_counts_exactly_and_offers_no_p_value(self) -> None:
        frames = [(f"f{i}", i) for i in range(10)]
        got = deterministic(
            frames,
            component="gate",
            with_component=lambda n: "block" if n % 2 else "allow",
            without_component=lambda n: "allow",
        )
        assert got.frames == 10 and len(got.changes) == 5
        assert not hasattr(got, "p_value")

    def test_a_component_that_changes_nothing_is_called_inert(self) -> None:
        frames = [(f"f{i}", i) for i in range(6)]
        got = deterministic(
            frames, component="gate",
            with_component=lambda n: "allow", without_component=lambda n: "allow",
        )
        assert got.inert and "changed nothing on any frame" in got.render()

    def test_inert_is_not_reported_as_useless(self) -> None:
        """A gate that never fired is untested, not proven worthless, and the text must say so."""
        got = deterministic(
            [("f0", 0)], component="gate",
            with_component=lambda n: "allow", without_component=lambda n: "allow",
        )
        assert "Either the frames do not reach it" in got.render()

    def test_each_change_is_named_not_just_counted(self) -> None:
        got = deterministic(
            [("weekend-NVDA", 1)], component="constitution",
            with_component=lambda n: "no_trade", without_component=lambda n: "buy",
        )
        assert got.changes[0].frame_id == "weekend-NVDA"
        assert "no_trade" in got.render() and "buy" in got.render()

    def test_no_frames_does_not_divide_by_zero(self) -> None:
        got = deterministic([], component="c", with_component=lambda x: x,
                            without_component=lambda x: x)
        assert got.share_changed == 0.0 and got.inert

    def test_it_serialises(self) -> None:
        got = deterministic(
            [("f0", 1)], component="c",
            with_component=lambda n: "a", without_component=lambda n: "b",
        ).as_dict()
        assert got["changed"] == 1 and got["changes"][0]["with"] == "a"


def test_the_default_minimum_is_the_documented_one() -> None:
    assert MIN_PAIRS == 30
    assert paired(_outcomes([1.0] * 29), component="c", metric="b").verdict is Verdict.INSUFFICIENT
    assert paired(_outcomes([1.0] * 30), component="c", metric="b").verdict is Verdict.HELPS
