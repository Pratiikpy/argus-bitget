"""Earnings decomposition tests — the seven must not collapse into one."""

from __future__ import annotations

from decimal import Decimal

import pytest

from argus.agents.earnings import (
    HEADLINE_BEAT_ROTTEN_CORE,
    SURPRISES,
    WEIGHTS,
    EarningsError,
    Surprises,
    decompose,
    from_response,
)


class TestTheCaseTheFieldGetsWrong:
    """EPS beat with cut guidance and evasive Q&A is a bearish print."""

    def test_a_headline_beat_with_a_rotten_core_reads_bearish(self) -> None:
        got = decompose("rNVDA", HEADLINE_BEAT_ROTTEN_CORE)
        assert got.surprises.headline > 0, "the headline genuinely beat"
        assert got.surprises.forward < 0, "the forward view genuinely deteriorated"
        assert got.direction == "bearish", f"read as {got.direction}: {got.explanation}"

    def test_the_contradiction_is_named_not_averaged(self) -> None:
        got = decompose("rNVDA", HEADLINE_BEAT_ROTTEN_CORE)
        assert got.surprises.is_contradictory is True
        assert "contradicts itself" in got.explanation

    def test_a_contradiction_lowers_confidence_not_the_signal(self) -> None:
        """Shrinking the signal would turn a disagreement into a weak consensus."""
        clean = Surprises(consensus=Decimal("0.6"), guidance=Decimal("0.6"))
        conflicted = Surprises(consensus=Decimal("0.6"), guidance=Decimal("-0.6"))
        a, b = decompose("x", clean), decompose("x", conflicted)
        assert b.confidence < a.confidence
        assert b.surprises.guidance == Decimal("-0.6")  # the score itself is untouched


class TestWeighting:
    def test_guidance_outweighs_the_reported_number(self) -> None:
        """A forward cut reprices every future period; a reported beat prices one that passed."""
        assert WEIGHTS["guidance"] > WEIGHTS["reported"]

    def test_weights_sum_to_one(self) -> None:
        assert sum(WEIGHTS.values()) == Decimal("1.00")

    def test_every_surprise_has_a_weight(self) -> None:
        assert set(WEIGHTS) == set(SURPRISES)

    def test_the_dominant_dimension_is_reported(self) -> None:
        got = decompose("x", Surprises(guidance=Decimal("-0.9")))
        assert got.dominant == "guidance"


class TestCostGate:
    def test_a_move_below_the_round_trip_is_neutral(self) -> None:
        """12bps is the floor; below it there is no trade, whatever the sign."""
        tiny = decompose("x", Surprises(consensus=Decimal("0.01")))
        assert tiny.clears_round_trip is False
        assert tiny.direction == "neutral"

    def test_a_large_clean_surprise_clears_it(self) -> None:
        big = decompose("x", Surprises(
            reported=Decimal("0.8"), consensus=Decimal("0.8"), guidance=Decimal("0.8"),
        ))
        assert big.clears_round_trip is True
        assert big.direction == "bullish"


class TestValidation:
    def test_scores_outside_the_range_are_refused(self) -> None:
        with pytest.raises(EarningsError, match=r"outside \[-1, 1\]"):
            Surprises(guidance=Decimal("2.5"))

    def test_a_missing_surprise_is_zero_not_imputed(self) -> None:
        """Imputing would let one strong dimension manufacture six others."""
        got = from_response("x", {"surprises": {"guidance": -0.8}})
        assert got.surprises.guidance == Decimal("-0.8")
        assert got.surprises.qa == Decimal("0")
        assert got.surprises.narrative == Decimal("0")

    def test_a_malformed_score_becomes_zero_rather_than_crashing(self) -> None:
        got = from_response("x", {"surprises": {"guidance": "very bad", "consensus": 0.5}})
        assert got.surprises.guidance == Decimal("0")
        assert got.surprises.consensus == Decimal("0.5")

    def test_out_of_range_values_from_a_model_are_clamped(self) -> None:
        got = from_response("x", {"surprises": {"guidance": -9}})
        assert got.surprises.guidance == Decimal("-1")


class TestDispersion:
    def test_an_ambiguous_print_is_less_confident(self) -> None:
        """High dispersion deserves a smaller position, not a louder opinion."""
        agreed = Surprises(**{n: Decimal("0.5") for n in SURPRISES})
        split = Surprises(
            reported=Decimal("1"), consensus=Decimal("-1"), guidance=Decimal("1"),
            narrative=Decimal("-1"), valuation=Decimal("1"),
            management_credibility=Decimal("-1"), qa=Decimal("1"),
        )
        assert split.dispersion > agreed.dispersion
        assert decompose("x", split).confidence < decompose("x", agreed).confidence

    def test_a_unanimous_print_has_no_dispersion(self) -> None:
        agreed = Surprises(**{n: Decimal("0.5") for n in SURPRISES})
        assert agreed.dispersion == Decimal("0")
        assert agreed.is_contradictory is False
