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


class TestAResizedOrderIsStillCheckedByEveryOtherGate:
    """The defect an external review predicted and a demonstration confirmed, 2026-09-15.

    The chain returned on the first gate that bound, so a gate that *resized* an order and returned
    left every later gate unrun on the resized size. Demonstrated: a book in deep drawdown, a
    45,000 proposal and no placeable hedge — `unhedgeable_gap` capped it at 20,000 and returned,
    `risk_budget` never ran, and the circuit breaker was sizing the book to **zero**. A 20,000 order
    was approved and the record read `binding_constraint: unhedgeable_gap`, i.e. a gate that
    appeared to be working.

    "A risk layer may only reduce what you choose" was true of each gate alone and false of the
    chain. These tests hold the fix in place.
    """

    @staticmethod
    def _policy_in_deep_drawdown() -> object:
        from argus.agents.desk import ConstitutionPolicy
        from argus.risk.circuit import BookState

        return ConstitutionPolicy(
            book_state=BookState(
                equity=Decimal("30000"),
                peak_equity=Decimal("100000"),
                session_open_equity=Decimal("30000"),
                consecutive_losses=3,
            )
        )

    @staticmethod
    def _weekend_no_hedge() -> tuple[object, object]:
        from datetime import UTC, datetime

        from argus.truth.clocks import SessionPhase, SessionState

        session = SessionState(
            phase=SessionPhase.WEEKEND,
            as_of=datetime(2026, 9, 12, 12, 0, tzinfo=UTC),
            hours_to_next_discovery=40.0,
            nav_age_seconds=10.0,
        )

        class _NoHedge:
            is_empty = True
            menu: tuple[object, ...] = ()

        return session, _NoHedge()

    def test_the_tightest_ceiling_binds_not_the_first_one_encountered(self) -> None:
        policy = self._policy_in_deep_drawdown()
        session, hedges = self._weekend_no_hedge()
        ruling = policy.rule(_intent("45000"), session=session, hedges=hedges)
        assert ruling.binding_constraint == "risk_budget", (
            f"{ruling.binding_constraint} bound; a looser ceiling must never win over a tighter one"
        )

    def test_a_zero_risk_budget_refuses_rather_than_approving_the_unhedged_cap(self) -> None:
        """A circuit breaker sizing to zero means trade nothing. It must not be bypassable."""
        policy = self._policy_in_deep_drawdown()
        session, hedges = self._weekend_no_hedge()
        ruling = policy.rule(_intent("45000"), session=session, hedges=hedges)
        assert ruling.resulting_intent.quantity == Decimal("0"), (
            "an order was approved while the risk budget allowed nothing"
        )

    def test_the_reason_names_every_ceiling_that_was_computed(self) -> None:
        """Naming one gate hid that another had also produced a cap."""
        policy = self._policy_in_deep_drawdown()
        session, hedges = self._weekend_no_hedge()
        ruling = policy.rule(_intent("45000"), session=session, hedges=hedges)
        assert "also computed" in ruling.reason
        assert "unhedgeable_gap" in ruling.reason

    def test_an_order_under_every_ceiling_is_still_allowed(self) -> None:
        """The fix must not turn a healthy desk into one that refuses everything."""
        from argus.agents.desk import ConstitutionPolicy

        session, hedges = self._weekend_no_hedge()
        ruling = ConstitutionPolicy().rule(_intent("1000"), session=session, hedges=hedges)
        assert ruling.binding_constraint == "none"
        assert ruling.resulting_intent.quantity == Decimal("1000")


class TestHedgeIntegrity:
    """"Smaller quantity is not safer" in its sharpest form — reducing a hedge leg while its
    partner stays open can raise portfolio risk even though every other gate reads it as strictly
    safer. `Verdict.REDUCE` plus `intent.side` disambiguate which position is being reduced (a
    SELL reduces a LONG, a BUY reduces a SHORT) with no change to `Intent` — see `agents/desk.py`'s
    gate comment for why that mapping is a Constitution-level fact, not the venue's `tradeSide`."""

    @staticmethod
    def _fresh_session_and_available_hedges() -> tuple[object, object]:
        from datetime import UTC, datetime

        from argus.truth.clocks import SessionPhase, SessionState

        session = SessionState(
            phase=SessionPhase.RTH,
            as_of=datetime(2026, 9, 15, 15, 0, tzinfo=UTC),
            hours_to_next_discovery=0.0, nav_age_seconds=10.0,
        )

        class _AvailableHedge:
            is_empty = False
            menu: tuple[object, ...] = (object(),)

        return session, _AvailableHedge()

    @staticmethod
    def _reduce_intent(*, symbol: str = "NVDAUSDT", side: Side = Side.SELL,
                        qty: str = "1") -> Intent:
        return Intent(
            symbol=symbol, side=side, quantity=Decimal(qty), verdict=Verdict.REDUCE,
            stated_confidence=0.9, thesis="closing a leg",
        )

    @staticmethod
    def _book_with_open_hedge_pair() -> object:
        from datetime import UTC, datetime

        from argus.desk.book import Book, HedgeLink, Lot, PositionSide

        book = Book()
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            lot=Lot(order_id="o1", approved_intent_hash="a1", side="buy",
                    quantity=Decimal("10"), price=Decimal("100"), commission=Decimal("0"),
                    ts_filled=datetime(2026, 9, 15, tzinfo=UTC)),
        )
        book.apply_fill(
            symbol="SQQQUSDT", side=PositionSide.SHORT,
            lot=Lot(order_id="o2", approved_intent_hash="a2", side="sell",
                    quantity=Decimal("5"), price=Decimal("20"), commission=Decimal("0"),
                    ts_filled=datetime(2026, 9, 15, tzinfo=UTC)),
        )
        book.link_hedge(HedgeLink(
            source=("NVDAUSDT", PositionSide.LONG), target=("SQQQUSDT", PositionSide.SHORT),
            relationship="HEDGE", hedge_ratio=None, ts_created=datetime(2026, 9, 15, tzinfo=UTC),
        ))
        return book

    def test_fires_when_reducing_a_hedge_linked_open_position(self) -> None:
        from argus.agents.desk import ConstitutionPolicy

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(book=self._book_with_open_hedge_pair())
        ruling = policy.rule(self._reduce_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint == "hedge_integrity"
        assert ruling.resulting_intent.quantity == Decimal("0")

    def test_does_not_fire_without_a_book(self) -> None:
        from argus.agents.desk import ConstitutionPolicy

        session, hedges = self._fresh_session_and_available_hedges()
        ruling = ConstitutionPolicy().rule(self._reduce_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint != "hedge_integrity"

    def test_does_not_fire_for_an_opening_verdict_on_the_same_symbol(self) -> None:
        from argus.agents.desk import ConstitutionPolicy

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(book=self._book_with_open_hedge_pair())
        intent = Intent(
            symbol="NVDAUSDT", side=Side.SELL, quantity=Decimal("1"), verdict=Verdict.TRADE,
            stated_confidence=0.9, thesis="new short", invalidation=("x",),
        )
        ruling = policy.rule(intent, session=session, hedges=hedges)
        assert ruling.binding_constraint != "hedge_integrity"

    def test_does_not_fire_once_the_hedge_partner_is_closed(self) -> None:
        from argus.agents.desk import ConstitutionPolicy
        from argus.desk.book import Lot, PositionSide

        session, hedges = self._fresh_session_and_available_hedges()
        book = self._book_with_open_hedge_pair()
        book.apply_fill(
            symbol="SQQQUSDT", side=PositionSide.SHORT,
            lot=Lot(order_id="o3", approved_intent_hash="a3", side="buy",
                    quantity=Decimal("5"), price=Decimal("20"), commission=Decimal("0"),
                    ts_filled=book.positions[("NVDAUSDT", PositionSide.LONG)].ts_last),
        )
        assert book.positions[("SQQQUSDT", PositionSide.SHORT)].is_flat
        policy = ConstitutionPolicy(book=book)
        ruling = policy.rule(self._reduce_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint != "hedge_integrity"

    def test_does_not_fire_with_no_hedge_link_recorded(self) -> None:
        from datetime import UTC, datetime

        from argus.agents.desk import ConstitutionPolicy
        from argus.desk.book import Book, Lot, PositionSide

        session, hedges = self._fresh_session_and_available_hedges()
        book = Book()
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            lot=Lot(order_id="o1", approved_intent_hash="a1", side="buy",
                    quantity=Decimal("10"), price=Decimal("100"), commission=Decimal("0"),
                    ts_filled=datetime(2026, 9, 15, tzinfo=UTC)),
        )
        policy = ConstitutionPolicy(book=book)
        ruling = policy.rule(self._reduce_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint != "hedge_integrity"

    def test_reducing_the_unlinked_side_is_unaffected(self) -> None:
        """A BUY on NVDAUSDT would reduce a SHORT there, not the hedge-linked LONG."""
        from argus.agents.desk import ConstitutionPolicy

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(book=self._book_with_open_hedge_pair())
        ruling = policy.rule(
            self._reduce_intent(side=Side.BUY), session=session, hedges=hedges,
        )
        assert ruling.binding_constraint != "hedge_integrity"


class TestMarginUsage:
    """A binary block, not a headroom — see `agents/desk.py`'s gate comment for why a partial
    number cannot honestly be computed against Bitget's proprietary cross-margin engine."""

    @staticmethod
    def _fresh_session_and_available_hedges() -> tuple[object, object]:
        from datetime import UTC, datetime

        from argus.truth.clocks import SessionPhase, SessionState

        session = SessionState(
            phase=SessionPhase.RTH,
            as_of=datetime(2026, 9, 15, 15, 0, tzinfo=UTC),
            hours_to_next_discovery=0.0, nav_age_seconds=10.0,
        )

        class _AvailableHedge:
            is_empty = False
            menu: tuple[object, ...] = (object(),)

        return session, _AvailableHedge()

    @staticmethod
    def _open_intent(qty: str = "1") -> Intent:
        return Intent(
            symbol="NVDAUSDT", side=Side.BUY, quantity=Decimal(qty), verdict=Verdict.TRADE,
            stated_confidence=0.9, thesis="t", invalidation=("x",),
        )

    @staticmethod
    def _book_with_margin(mgn_ratio: str) -> object:
        from datetime import UTC, datetime

        from argus.desk.book import Book, VenueMarginSnapshot

        book = Book()
        book.set_margin(VenueMarginSnapshot(
            account_equity=Decimal("10000"), unrealised_pnl=Decimal("0"),
            imr=Decimal("0"), mmr=Decimal("0"), mgn_ratio=Decimal(mgn_ratio),
            position_mgn_ratio=Decimal("0"), leverage=Decimal("1"),
            fetched_at=datetime(2026, 9, 15, tzinfo=UTC),
        ))
        return book

    def test_fires_at_the_policy_cap(self) -> None:
        from argus.agents.desk import ConstitutionPolicy

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(book=self._book_with_margin("0.5"))
        ruling = policy.rule(self._open_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint == "margin_usage"
        assert ruling.resulting_intent.quantity == Decimal("0")

    def test_fires_above_the_policy_cap(self) -> None:
        from argus.agents.desk import ConstitutionPolicy

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(book=self._book_with_margin("0.9"))
        ruling = policy.rule(self._open_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint == "margin_usage"

    def test_does_not_fire_below_the_policy_cap(self) -> None:
        from argus.agents.desk import ConstitutionPolicy

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(book=self._book_with_margin("0.1"))
        ruling = policy.rule(self._open_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint != "margin_usage"

    def test_does_not_fire_without_a_margin_snapshot(self) -> None:
        """A book that exists but has never fetched a margin snapshot must not read as zero
        usage — `None` means unmeasured, not safe."""
        from argus.agents.desk import ConstitutionPolicy
        from argus.desk.book import Book

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(book=Book())
        ruling = policy.rule(self._open_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint != "margin_usage"

    def test_does_not_fire_without_a_book(self) -> None:
        from argus.agents.desk import ConstitutionPolicy

        session, hedges = self._fresh_session_and_available_hedges()
        ruling = ConstitutionPolicy().rule(self._open_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint != "margin_usage"


class TestFactorExposure:
    """A binary block — `FactorExposure` is a whole-book regression slope, not decomposed per
    symbol, so there is no per-symbol beta to compute a partial headroom against (design brief,
    2026-09-15)."""

    @staticmethod
    def _fresh_session_and_available_hedges() -> tuple[object, object]:
        from datetime import UTC, datetime

        from argus.truth.clocks import SessionPhase, SessionState

        session = SessionState(
            phase=SessionPhase.RTH,
            as_of=datetime(2026, 9, 15, 15, 0, tzinfo=UTC),
            hours_to_next_discovery=0.0, nav_age_seconds=10.0,
        )

        class _AvailableHedge:
            is_empty = False
            menu: tuple[object, ...] = (object(),)

        return session, _AvailableHedge()

    @staticmethod
    def _open_intent(qty: str = "1") -> Intent:
        return Intent(
            symbol="NVDAUSDT", side=Side.BUY, quantity=Decimal(qty), verdict=Verdict.TRADE,
            stated_confidence=0.9, thesis="t", invalidation=("x",),
        )

    def test_fires_when_a_named_factor_is_at_or_above_its_cap(self) -> None:
        from argus.agents.desk import ConstitutionPolicy
        from argus.desk.portfolio import FactorExposure

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(
            factor_exposures=(FactorExposure(factor="market", exposure=0.9, observations=200),),
            factor_exposure_limits={"market": 0.5},
        )
        ruling = policy.rule(self._open_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint == "factor_exposure"
        assert ruling.resulting_intent.quantity == Decimal("0")

    def test_negative_exposure_is_checked_by_magnitude(self) -> None:
        from argus.agents.desk import ConstitutionPolicy
        from argus.desk.portfolio import FactorExposure

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(
            factor_exposures=(FactorExposure(factor="market", exposure=-0.9, observations=200),),
            factor_exposure_limits={"market": 0.5},
        )
        ruling = policy.rule(self._open_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint == "factor_exposure"

    def test_does_not_fire_below_the_cap(self) -> None:
        from argus.agents.desk import ConstitutionPolicy
        from argus.desk.portfolio import FactorExposure

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(
            factor_exposures=(FactorExposure(factor="market", exposure=0.1, observations=200),),
            factor_exposure_limits={"market": 0.5},
        )
        ruling = policy.rule(self._open_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint != "factor_exposure"

    def test_a_factor_with_no_configured_limit_is_uncapped_not_zero_capped(self) -> None:
        from argus.agents.desk import ConstitutionPolicy
        from argus.desk.portfolio import FactorExposure

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(
            factor_exposures=(FactorExposure(factor="unlisted", exposure=5.0, observations=200),),
            factor_exposure_limits={"market": 0.5},
        )
        ruling = policy.rule(self._open_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint != "factor_exposure"

    def test_does_not_fire_without_precomputed_exposures(self) -> None:
        from argus.agents.desk import ConstitutionPolicy

        session, hedges = self._fresh_session_and_available_hedges()
        ruling = ConstitutionPolicy().rule(self._open_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint != "factor_exposure"


class TestScenarioLoss:
    """A binary block, same reasoning as `TestFactorExposure` — `stress_by_beta`'s output is the
    book's whole market-driven move under a shock, not this order's marginal contribution."""

    @staticmethod
    def _fresh_session_and_available_hedges() -> tuple[object, object]:
        from datetime import UTC, datetime

        from argus.truth.clocks import SessionPhase, SessionState

        session = SessionState(
            phase=SessionPhase.RTH,
            as_of=datetime(2026, 9, 15, 15, 0, tzinfo=UTC),
            hours_to_next_discovery=0.0, nav_age_seconds=10.0,
        )

        class _AvailableHedge:
            is_empty = False
            menu: tuple[object, ...] = (object(),)

        return session, _AvailableHedge()

    @staticmethod
    def _open_intent(qty: str = "1") -> Intent:
        return Intent(
            symbol="NVDAUSDT", side=Side.BUY, quantity=Decimal(qty), verdict=Verdict.TRADE,
            stated_confidence=0.9, thesis="t", invalidation=("x",),
        )

    def test_fires_when_the_worst_shock_is_past_the_floor(self) -> None:
        from argus.agents.desk import ConstitutionPolicy
        from argus.desk.portfolio import StressOutcome

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(
            stress_outcomes=(
                StressOutcome(shock="-10%", portfolio_move_pct=-9.0, worst_position=None),
                StressOutcome(shock="-2%", portfolio_move_pct=-1.0, worst_position=None),
            ),
            max_scenario_loss_pct=-5.0,
        )
        ruling = policy.rule(self._open_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint == "scenario_loss"
        assert ruling.resulting_intent.quantity == Decimal("0")

    def test_does_not_fire_when_every_shock_is_inside_the_floor(self) -> None:
        from argus.agents.desk import ConstitutionPolicy
        from argus.desk.portfolio import StressOutcome

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(
            stress_outcomes=(
                StressOutcome(shock="-2%", portfolio_move_pct=-1.0, worst_position=None),
            ),
            max_scenario_loss_pct=-5.0,
        )
        ruling = policy.rule(self._open_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint != "scenario_loss"

    def test_unavailable_shocks_are_excluded_not_treated_as_zero(self) -> None:
        from argus.agents.desk import ConstitutionPolicy
        from argus.desk.portfolio import StressOutcome

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(
            stress_outcomes=(
                StressOutcome(shock="-10%", portfolio_move_pct=None, worst_position=None,
                               reason="no estimable beta"),
            ),
            max_scenario_loss_pct=-5.0,
        )
        ruling = policy.rule(self._open_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint != "scenario_loss"

    def test_does_not_fire_without_a_configured_floor(self) -> None:
        from argus.agents.desk import ConstitutionPolicy
        from argus.desk.portfolio import StressOutcome

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(
            stress_outcomes=(
                StressOutcome(shock="-10%", portfolio_move_pct=-9.0, worst_position=None),
            ),
        )
        ruling = policy.rule(self._open_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint != "scenario_loss"


class TestLiquidationCost:
    """A proxy for the venue's own bankruptcy-price/ADL cost, not the literal thing — real live
    order-book depth (`market.depth.OrderBook.sweep`) instead, stated plainly in the gate's own
    reason string (design brief, 2026-09-15). Binary block, same reasoning as the three gates
    above: `Sweep` prices exiting one position, not this order's own marginal contribution."""

    @staticmethod
    def _fresh_session_and_available_hedges() -> tuple[object, object]:
        from datetime import UTC, datetime

        from argus.truth.clocks import SessionPhase, SessionState

        session = SessionState(
            phase=SessionPhase.RTH,
            as_of=datetime(2026, 9, 15, 15, 0, tzinfo=UTC),
            hours_to_next_discovery=0.0, nav_age_seconds=10.0,
        )

        class _AvailableHedge:
            is_empty = False
            menu: tuple[object, ...] = (object(),)

        return session, _AvailableHedge()

    @staticmethod
    def _open_intent(qty: str = "1") -> Intent:
        return Intent(
            symbol="NVDAUSDT", side=Side.BUY, quantity=Decimal(qty), verdict=Verdict.TRADE,
            stated_confidence=0.9, thesis="t", invalidation=("x",),
        )

    @staticmethod
    def _sweep(slippage_bps: str, *, complete: bool = True) -> object:
        from argus.market.depth import Sweep

        return Sweep(
            side="SELL", requested_notional=Decimal("10000"), filled_notional=Decimal("10000"),
            average_price=Decimal("100"), levels_consumed=5,
            slippage_bps=Decimal(slippage_bps), complete=complete,
        )

    def test_fires_at_or_above_the_cap(self) -> None:
        from argus.agents.desk import ConstitutionPolicy

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(
            liquidation_cost_estimates={"NVDAUSDT": self._sweep("500")},
            max_liquidation_cost_bps=Decimal("500"),
        )
        ruling = policy.rule(self._open_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint == "liquidation_cost"
        assert ruling.resulting_intent.quantity == Decimal("0")

    def test_the_worst_symbol_across_the_book_is_the_one_that_binds(self) -> None:
        from argus.agents.desk import ConstitutionPolicy

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(
            liquidation_cost_estimates={
                "NVDAUSDT": self._sweep("50"), "TQQQUSDT": self._sweep("900"),
            },
            max_liquidation_cost_bps=Decimal("500"),
        )
        ruling = policy.rule(self._open_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint == "liquidation_cost"

    def test_does_not_fire_below_the_cap(self) -> None:
        from argus.agents.desk import ConstitutionPolicy

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(
            liquidation_cost_estimates={"NVDAUSDT": self._sweep("10")},
            max_liquidation_cost_bps=Decimal("500"),
        )
        ruling = policy.rule(self._open_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint != "liquidation_cost"

    def test_does_not_fire_without_a_configured_cap(self) -> None:
        from argus.agents.desk import ConstitutionPolicy

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(
            liquidation_cost_estimates={"NVDAUSDT": self._sweep("900")},
        )
        ruling = policy.rule(self._open_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint != "liquidation_cost"

    def test_does_not_fire_without_estimates(self) -> None:
        from argus.agents.desk import ConstitutionPolicy

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(max_liquidation_cost_bps=Decimal("500"))
        ruling = policy.rule(self._open_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint != "liquidation_cost"

    def test_an_empty_estimates_mapping_does_not_fire(self) -> None:
        from argus.agents.desk import ConstitutionPolicy

        session, hedges = self._fresh_session_and_available_hedges()
        policy = ConstitutionPolicy(
            liquidation_cost_estimates={}, max_liquidation_cost_bps=Decimal("500"),
        )
        ruling = policy.rule(self._open_intent(), session=session, hedges=hedges)
        assert ruling.binding_constraint != "liquidation_cost"
