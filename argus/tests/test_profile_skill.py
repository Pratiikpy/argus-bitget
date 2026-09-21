"""Did the mandate pick *which* trades to decline, or did it just decline some?

`profile_value.py` computed a utility gain and called any positive number "earned its keep". That
is a check that cannot fail: the first run of the module reported **+0.35bps off 72 refusals out of
4,528** and blessed it, and a permutation test says refusing 72 at random matches or beats that in
most draws. The gain was noise wearing a conclusion, and nothing in the module could tell.

These tests hold the replacement to the standard the old verdict failed: it must be **able to say
yes**, able to say no, and unmoved by how the outcomes happen to be ordered.
"""

from __future__ import annotations

import random

import pytest

from argus.eval.profile_value import SKILL_ALPHA, ProfileResult


def _result(name: str, taken: list[float], refused: list[float]) -> ProfileResult:
    return ProfileResult(name=name, taken=tuple(taken), refused=tuple(refused))


class TestTheTestCanSayYes:
    """**A significance test that never passes is as useless as one that never fails.**

    The real mandate fails this test, so nothing in the shipped artefact demonstrates that a pass
    is reachable. These construct one.
    """

    def test_a_mandate_that_declines_only_losers_is_credited(self) -> None:
        rng = random.Random(11)
        taken = [rng.gauss(5.0, 10.0) for _ in range(900)]
        refused = [rng.gauss(-90.0, 10.0) for _ in range(100)]
        result = _result("oracle", taken, refused)
        assert result.utility_gain_bps > 0
        assert result.skill_p_value is not None and result.skill_p_value < SKILL_ALPHA
        assert result.beat_random_refusal
        assert "earned its keep" in result.verdict

    def test_a_mandate_declining_at_random_is_not_credited(self) -> None:
        """Same refusal rate, same outcomes, chosen without skill. It must not pass."""
        rng = random.Random(12)
        pool = [rng.gauss(0.0, 25.0) for _ in range(1000)]
        rng.shuffle(pool)
        result = _result("coin flip", pool[100:], pool[:100])
        assert not result.beat_random_refusal

    def test_a_mandate_that_declines_only_winners_is_called_out(self) -> None:
        rng = random.Random(13)
        taken = [rng.gauss(-5.0, 10.0) for _ in range(900)]
        refused = [rng.gauss(90.0, 10.0) for _ in range(100)]
        result = _result("inverted", taken, refused)
        assert result.utility_gain_bps < 0
        assert not result.beat_random_refusal
        assert "cost its owner money" in result.verdict


class TestAGainIsNotEnough:
    """The exact defect: a positive gain that random refusal reproduces."""

    def test_a_gain_that_random_refusal_matches_is_refused_the_word_earned(self) -> None:
        rng = random.Random(14)
        pool = [rng.gauss(0.0, 30.0) for _ in range(4000)]
        pool.sort()
        # Decline 40 of the very worst, which is a real but tiny edge on a large sample.
        result = _result("thin edge", pool[40:], pool[:40])
        if result.utility_gain_bps > 0 and not result.beat_random_refusal:
            assert "earned its keep" not in result.verdict
            assert "not evidence of selection" in result.verdict

    def test_the_p_value_reaches_the_artefact(self) -> None:
        result = _result("x", [1.0, 2.0, 3.0], [-4.0])
        row = result.as_dict()
        assert "skill_p_value" in row
        assert "beat_random_refusal" in row


class TestUndefinedIsNotPassed:
    @pytest.mark.parametrize(
        ("taken", "refused"),
        [([1.0, 2.0], []), ([], [1.0, 2.0])],
    )
    def test_no_verdict_of_skill_without_both_sides(
        self, taken: list[float], refused: list[float]
    ) -> None:
        result = _result("degenerate", taken, refused)
        assert result.skill_p_value is None
        assert not result.beat_random_refusal
        assert result.as_dict()["skill_p_value"] is None


class TestItReproduces:
    def test_two_calls_agree(self) -> None:
        """Seeded. A significance figure that moves between runs is one more thing to trust."""
        rng = random.Random(15)
        result = _result(
            "stable",
            [rng.gauss(0.0, 20.0) for _ in range(500)],
            [rng.gauss(-30.0, 20.0) for _ in range(60)],
        )
        assert result.skill_p_value == result.skill_p_value

    def test_order_does_not_change_the_answer(self) -> None:
        rng = random.Random(16)
        taken = [rng.gauss(0.0, 20.0) for _ in range(400)]
        refused = [rng.gauss(-25.0, 20.0) for _ in range(50)]
        straight = _result("a", taken, refused).skill_p_value
        shuffled_taken = list(taken)
        random.Random(17).shuffle(shuffled_taken)
        assert _result("a", shuffled_taken, refused).skill_p_value == straight


class TestRiskIsMeasuredAsWellAsReturn:
    """**A conservative mandate is not supposed to be alpha.**

    It sells expected return for a smaller loss tail, and judging it only on bps asks it to be
    something it never claimed to be. The shipped conservative profile costs -2.09bps and cuts the
    downside tail by 35.4%; reporting only the first number calls a working mandate a failure.
    """

    def test_refusing_losers_cuts_the_tail_and_is_credited(self) -> None:
        rng = random.Random(21)
        taken = [rng.gauss(2.0, 8.0) for _ in range(900)]
        refused = [rng.gauss(-120.0, 15.0) for _ in range(100)]
        result = _result("shield", taken, refused)
        assert result.tail_reduction > 0
        assert result.tail_p_value is not None and result.tail_p_value < SKILL_ALPHA
        assert result.delivered_its_risk_promise
        assert "cut the downside tail" in result.verdict

    def test_refusing_at_random_does_not_earn_the_risk_claim(self) -> None:
        """**Refusing anything cuts variance, so this is the claim easiest to win by cheating.**

        A zero has no dispersion. Without the permutation null, every mandate that declines
        anything at all would appear to deliver risk reduction.
        """
        rng = random.Random(22)
        pool = [rng.gauss(0.0, 30.0) for _ in range(1200)]
        rng.shuffle(pool)
        result = _result("coin flip", pool[150:], pool[:150])
        assert result.tail_reduction > 0, "zeros must mechanically reduce the tail"
        assert not result.delivered_its_risk_promise, "and that alone must not earn the claim"

    def test_a_mandate_that_refuses_winners_still_reports_the_tail_honestly(self) -> None:
        rng = random.Random(23)
        taken = [rng.gauss(-4.0, 20.0) for _ in range(800)]
        refused = [rng.gauss(60.0, 10.0) for _ in range(120)]
        result = _result("inverted", taken, refused)
        assert result.utility_gain_bps < 0
        assert "cost its owner money" in result.verdict
        assert "downside tail" in result.verdict

    def test_the_two_streams_are_paired_decision_for_decision(self) -> None:
        result = _result("x", [1.0, -2.0], [-5.0])
        assert result.mandate_stream == [1.0, -2.0, 0.0]
        assert result.everything_stream == [1.0, -2.0, -5.0]
        assert len(result.mandate_stream) == len(result.everything_stream) == result.decisions

    def test_gains_are_not_counted_as_risk(self) -> None:
        """Semi-deviation, not standard deviation: a mandate must not be penalised for upside."""
        from argus.eval.profile_value import _downside

        assert _downside([10.0, 20.0, 30.0]) == 0.0
        assert _downside([-10.0, 0.0, 10.0]) > 0.0

    def test_the_risk_fields_reach_the_artefact(self) -> None:
        row = _result("x", [1.0, -2.0, 3.0], [-4.0]).as_dict()
        for field in ("downside_deviation_bps", "downside_taking_everything_bps",
                      "tail_reduction_pct", "tail_p_value", "delivered_its_risk_promise"):
            assert field in row, field


class TestTheReportedPValueIsPossible:
    def test_an_unreachable_p_is_not_printed_as_zero(self) -> None:
        """``(hits+1)/(trials+1)`` makes 0 unreachable, so ``p=0.000`` would be a rendering bug."""
        from argus.eval.profile_value import SKILL_TRIALS, _p

        floor = 1.0 / (SKILL_TRIALS + 1)
        assert _p(floor).startswith("<")
        assert _p(0.5) == "0.500"

    def test_no_shipped_verdict_says_p_equals_zero(self) -> None:
        rng = random.Random(24)
        result = _result(
            "shield",
            [rng.gauss(2.0, 8.0) for _ in range(600)],
            [rng.gauss(-150.0, 10.0) for _ in range(80)],
        )
        assert "p=0.000" not in result.verdict
