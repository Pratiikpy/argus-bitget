"""Constitution asymmetry tests — the property that holds both halves of Track 2's rule.

The claim under test: **the Constitution may only reduce.** If any of these fails, the risk layer
can author an economic decision, the LLM becomes a narrator, and ARGUS fails the positioning rule
the same way four of the systems we tore down do.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from argus.decision.verdicts import (
    ConstitutionVerdict,
    ConstitutionViolation,
    Intent,
    Side,
    Verdict,
    apply_constraint,
)


def _intent(qty: str = "100", side: Side = Side.SELL) -> Intent:
    return Intent(
        symbol="rNVDA",
        side=side,
        quantity=Decimal(qty),
        verdict=Verdict.TRADE,
        stated_confidence=0.72,
        thesis="guidance cut is not yet priced into the token",
        invalidation=("guidance reaffirmed at the open", "BTC recovers above 92k"),
    )


class TestTheConstitutionCanOnlyReduce:
    def test_resize_upward_is_refused(self) -> None:
        """The single most important invariant in the system."""
        with pytest.raises(ConstitutionViolation, match="may only reduce"):
            apply_constraint(
                _intent("100"),
                verdict=ConstitutionVerdict.RESIZE,
                binding_constraint="concentration_cap",
                reason="sector concentration would reach 31% against a 25% cap",
                resized_quantity=Decimal("150"),
            )

    def test_resize_downward_is_allowed(self) -> None:
        ruling = apply_constraint(
            _intent("100"),
            verdict=ConstitutionVerdict.RESIZE,
            binding_constraint="concentration_cap",
            reason="sector concentration would reach 31% against a 25% cap",
            resized_quantity=Decimal("53"),
        )
        assert ruling.resulting_intent.quantity == Decimal("53")
        assert ruling.binding_constraint == "concentration_cap"

    def test_side_is_never_flipped_by_any_verdict(self) -> None:
        """A risk layer that can reverse a position has authored the trade."""
        original = _intent(side=Side.SELL)
        for verdict, extra in [
            (ConstitutionVerdict.ALLOW, {}),
            (ConstitutionVerdict.RESIZE, {"resized_quantity": Decimal("10")}),
            (ConstitutionVerdict.REQUIRE_HEDGE, {"required_hedge": ("BTCUSDT",)}),
            (ConstitutionVerdict.DELAY, {}),
            (ConstitutionVerdict.REJECT, {}),
            (ConstitutionVerdict.FLATTEN, {}),
        ]:
            ruling = apply_constraint(
                original, verdict=verdict, binding_constraint="c", reason="r", **extra
            )
            assert ruling.resulting_intent.side is Side.SELL

    def test_no_verdict_increases_quantity(self) -> None:
        original = _intent("100")
        for verdict, extra in [
            (ConstitutionVerdict.ALLOW, {}),
            (ConstitutionVerdict.RESIZE, {"resized_quantity": Decimal("40")}),
            (ConstitutionVerdict.REQUIRE_HEDGE, {"required_hedge": ("BTCUSDT",)}),
            (ConstitutionVerdict.DELAY, {}),
            (ConstitutionVerdict.REJECT, {}),
            (ConstitutionVerdict.FLATTEN, {}),
        ]:
            ruling = apply_constraint(
                original, verdict=verdict, binding_constraint="c", reason="r", **extra
            )
            assert ruling.resulting_intent.quantity <= original.quantity

    def test_require_hedge_cannot_increase_quantity(self) -> None:
        with pytest.raises(ConstitutionViolation, match="may not increase"):
            apply_constraint(
                _intent("100"),
                verdict=ConstitutionVerdict.REQUIRE_HEDGE,
                binding_constraint="unhedged_gap",
                reason="carrying naked exposure across a shut session",
                resized_quantity=Decimal("120"),
                required_hedge=("BTCUSDT",),
            )

    def test_allow_cannot_quietly_change_quantity(self) -> None:
        with pytest.raises(ConstitutionViolation, match="use RESIZE"):
            apply_constraint(
                _intent("100"),
                verdict=ConstitutionVerdict.ALLOW,
                binding_constraint="none",
                reason="within limits",
                resized_quantity=Decimal("90"),
            )

    def test_resize_to_zero_must_be_expressed_as_reject(self) -> None:
        """Otherwise a zero-size 'allowed' trade hides a refusal from the attribution count."""
        with pytest.raises(ConstitutionViolation, match="REJECT"):
            apply_constraint(
                _intent("100"),
                verdict=ConstitutionVerdict.RESIZE,
                binding_constraint="x",
                reason="y",
                resized_quantity=Decimal("0"),
            )

    def test_require_hedge_must_name_an_instrument(self) -> None:
        """'Hedge it somehow' is not a constraint anyone can execute or audit."""
        with pytest.raises(ConstitutionViolation, match="at least one hedge"):
            apply_constraint(
                _intent(),
                verdict=ConstitutionVerdict.REQUIRE_HEDGE,
                binding_constraint="unhedged_gap",
                reason="naked across the weekend",
            )


class TestRulingsPreserveWhatMatters:
    def test_reject_zeroes_quantity_and_records_no_trade(self) -> None:
        ruling = apply_constraint(
            _intent("100"),
            verdict=ConstitutionVerdict.REJECT,
            binding_constraint="daily_loss_limit",
            reason="daily loss limit already breached",
        )
        assert ruling.resulting_intent.quantity == Decimal("0")
        assert ruling.resulting_intent.verdict is Verdict.NO_TRADE

    def test_invalidation_conditions_survive_a_downgrade(self) -> None:
        """They are why the trade was refused; dropping them loses the reason on the way to
        the ledger."""
        ruling = apply_constraint(
            _intent(),
            verdict=ConstitutionVerdict.DELAY,
            binding_constraint="oracle_stale",
            reason="NAV is 4h old during RTH",
        )
        assert ruling.resulting_intent.invalidation == (
            "guidance reaffirmed at the open",
            "BTC recovers above 92k",
        )

    def test_thesis_survives_every_ruling(self) -> None:
        """The LLM's reasoning must reach the ledger whatever the risk layer decided."""
        ruling = apply_constraint(
            _intent(),
            verdict=ConstitutionVerdict.REJECT,
            binding_constraint="x",
            reason="y",
        )
        assert "guidance cut" in ruling.resulting_intent.thesis

    def test_binding_constraint_is_machine_readable(self) -> None:
        """'Which constraint bound, how often' is the evidence the risk layer does real work."""
        ruling = apply_constraint(
            _intent(),
            verdict=ConstitutionVerdict.RESIZE,
            binding_constraint="concentration_cap",
            reason="human-readable detail",
            resized_quantity=Decimal("50"),
        )
        assert ruling.binding_constraint == "concentration_cap"
        assert " " not in ruling.binding_constraint


class TestIntentValidation:
    def test_opening_exposure_requires_an_invalidation_condition(self) -> None:
        """A position whose thesis cannot be falsified cannot be monitored, only hoped over."""
        with pytest.raises(ValueError, match="invalidation condition"):
            Intent(
                symbol="rNVDA", side=Side.SELL, quantity=Decimal("10"),
                verdict=Verdict.TRADE, stated_confidence=0.6, thesis="feels right",
            )

    def test_abstaining_does_not_require_invalidation(self) -> None:
        Intent(
            symbol="rNVDA", side=Side.SELL, quantity=Decimal("0"),
            verdict=Verdict.NO_TRADE, stated_confidence=0.9,
            thesis="edge does not clear the 12bps round trip",
        )

    def test_a_thesis_is_mandatory(self) -> None:
        with pytest.raises(ValueError, match="thesis"):
            Intent(
                symbol="rNVDA", side=Side.SELL, quantity=Decimal("0"),
                verdict=Verdict.NO_TRADE, stated_confidence=0.5, thesis="   ",
            )

    def test_confidence_must_be_a_probability(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            Intent(
                symbol="rNVDA", side=Side.SELL, quantity=Decimal("0"),
                verdict=Verdict.NO_TRADE, stated_confidence=87.0, thesis="t",
            )

    def test_trade_requires_positive_quantity(self) -> None:
        with pytest.raises(ValueError, match="positive quantity"):
            Intent(
                symbol="rNVDA", side=Side.SELL, quantity=Decimal("0"),
                verdict=Verdict.TRADE, stated_confidence=0.5, thesis="t",
                invalidation=("x",),
            )


class TestVerdictSemantics:
    def test_data_insufficient_is_not_no_trade(self) -> None:
        """'I looked and the answer is no' versus 'I could not see' are different failures."""
        assert Verdict.DATA_INSUFFICIENT is not Verdict.NO_TRADE
        assert Verdict.DATA_INSUFFICIENT.is_abstention
        assert Verdict.NO_TRADE.is_abstention

    def test_only_trade_and_hedge_open_exposure(self) -> None:
        assert {v for v in Verdict if v.opens_exposure} == {Verdict.TRADE, Verdict.HEDGE}


class TestReduceCarriesQuantity:
    """Regression: found by a live model run, not by design review.

    The model returned REDUCE with a real size and the parser zeroed it, because an earlier
    version conflated "opens exposure" with "has a quantity". REDUCE lowers risk rather than
    creating it — but "reduce" with no number is a sentiment, not an instruction.
    """

    def test_reduce_carries_quantity_but_does_not_open_exposure(self) -> None:
        assert Verdict.REDUCE.carries_quantity is True
        assert Verdict.REDUCE.opens_exposure is False

    def test_reduce_requires_a_positive_quantity(self) -> None:
        with pytest.raises(ValueError, match="positive quantity"):
            Intent(
                symbol="rNVDA", side=Side.SELL, quantity=Decimal("0"),
                verdict=Verdict.REDUCE, stated_confidence=0.6,
                thesis="cut exposure into the close",
            )

    def test_reduce_does_not_require_an_invalidation_condition(self) -> None:
        """It lowers risk. Demanding a falsifier to de-risk would punish caution."""
        got = Intent(
            symbol="rNVDA", side=Side.SELL, quantity=Decimal("80"),
            verdict=Verdict.REDUCE, stated_confidence=0.6,
            thesis="cut exposure ahead of a 30.5h unhedgeable window",
        )
        assert got.quantity == Decimal("80")

    def test_abstentions_still_carry_no_quantity(self) -> None:
        for v in (Verdict.NO_TRADE, Verdict.DELAY, Verdict.DATA_INSUFFICIENT):
            assert v.carries_quantity is False
