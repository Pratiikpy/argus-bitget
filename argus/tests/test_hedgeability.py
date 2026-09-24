"""Hedgeability surface tests.

The claim: a ranked menu and a priced residual, never a fraction — and the empty-menu case is the
normal state of the world for 65.5 hours a week, not an edge case.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from argus.paper.runner import _hedge_surface
from argus.risk.hedgeability import (
    HedgeabilitySurface,
    HedgeCandidate,
    open_market_candidate,
    shut_market_candidate,
)


def _candidate(
    name: str = "BTCUSDT",
    *,
    risk_reduction: str = "0.40",
    execution_probability: str = "0.97",
    liquidity: str = "0.85",
    cost_bps: str = "21",
    correlation_confidence: str = "0.70",
    basis_stability: str = "0.80",
) -> HedgeCandidate:
    return HedgeCandidate(
        instrument=name,
        risk_reduction=Decimal(risk_reduction),
        correlation_confidence=Decimal(correlation_confidence),
        liquidity_availability=Decimal(liquidity),
        execution_probability=Decimal(execution_probability),
        basis_stability=Decimal(basis_stability),
        execution_cost_bps=Decimal(cost_bps),
    )


class TestEffectiveRiskReduction:
    def test_every_way_of_disappointing_compounds(self) -> None:
        """Independent failure modes multiply. A hedge that looks 40% effective is not."""
        c = _candidate()
        # 0.40 * 0.70 * 0.85 * 0.97 * 0.80
        assert c.effective_risk_reduction == pytest.approx(Decimal("0.184688"), abs=Decimal("1e-6"))
        assert c.effective_risk_reduction < c.risk_reduction

    def test_a_shut_venue_removes_no_risk_however_good_the_correlation(self) -> None:
        """The Sleeping-Anchor case. A perfect hedge you cannot reach is not a hedge."""
        c = shut_market_candidate("NVDA", Decimal("0.95"))
        assert c.effective_risk_reduction == Decimal("0")
        assert c.is_placeable is False

    def test_factors_must_be_probabilities(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            _candidate(risk_reduction="1.4")

    def test_costs_may_not_be_negative(self) -> None:
        with pytest.raises(ValueError, match="may not be negative"):
            HedgeCandidate(
                instrument="X", risk_reduction=Decimal("0.5"),
                correlation_confidence=Decimal("0.5"), liquidity_availability=Decimal("0.5"),
                execution_probability=Decimal("0.5"), basis_stability=Decimal("0.5"),
                execution_cost_bps=Decimal("-1"),
            )


class TestRiskNeutralisationEfficiency:
    def test_efficiency_is_risk_removed_per_bp_of_cost(self) -> None:
        cheap = _candidate("BTC", cost_bps="10")
        dear = _candidate("SOXX", cost_bps="58")
        assert cheap.risk_neutralisation_efficiency > dear.risk_neutralisation_efficiency

    def test_all_in_cost_includes_collateral_and_model_uncertainty(self) -> None:
        """Model uncertainty is charged in bps so it competes with real costs on one scale
        instead of being argued about qualitatively."""
        c = HedgeCandidate(
            instrument="X", risk_reduction=Decimal("0.5"),
            correlation_confidence=Decimal("1"), liquidity_availability=Decimal("1"),
            execution_probability=Decimal("1"), basis_stability=Decimal("1"),
            execution_cost_bps=Decimal("20"),
            collateral_cost_bps=Decimal("5"),
            model_uncertainty_bps=Decimal("15"),
        )
        assert c.all_in_cost_bps == Decimal("40")

    def test_a_free_hedge_that_removes_nothing_scores_zero(self) -> None:
        """Irrelevant, not efficient."""
        c = HedgeCandidate(
            instrument="X", risk_reduction=Decimal("0"),
            correlation_confidence=Decimal("1"), liquidity_availability=Decimal("1"),
            execution_probability=Decimal("1"), basis_stability=Decimal("1"),
            execution_cost_bps=Decimal("0"),
        )
        assert c.risk_neutralisation_efficiency == Decimal("0")


class TestTheMenu:
    def test_menu_is_ranked_by_efficiency_not_raw_risk_reduction(self) -> None:
        """The biggest risk reducer is not automatically the right hedge."""
        big_and_dear = _candidate("SOXX", risk_reduction="0.60", cost_bps="90")
        small_and_cheap = _candidate("BTC", risk_reduction="0.30", cost_bps="8")
        surface = HedgeabilitySurface((big_and_dear, small_and_cheap))
        assert surface.best is not None
        assert surface.best.instrument == "BTC"

    def test_unplaceable_candidates_are_excluded_from_the_menu(self) -> None:
        surface = HedgeabilitySurface((
            shut_market_candidate("NVDA", Decimal("0.95")),
            _candidate("BTCUSDT"),
        ))
        assert [c.instrument for c in surface.menu] == ["BTCUSDT"]

    def test_shut_instruments_are_still_carried_on_the_surface(self) -> None:
        """Omitting them would lose the fact that a good hedge exists and cannot be reached —
        which is exactly what the Sleeping-Anchor problem turns on."""
        shut = shut_market_candidate("NVDA", Decimal("0.95"))
        surface = HedgeabilitySurface((shut,))
        assert shut in surface.candidates
        assert surface.is_empty is True


class TestTheEmptyMenu:
    """For 65.5 hours a week this is the normal state, not an edge case."""

    surface = HedgeabilitySurface(
        (shut_market_candidate("NVDA", Decimal("0.95")),
         shut_market_candidate("SOXX", Decimal("0.70"))),
        session_note="NYSE shut; next discovery in 30.5h",
    )

    def test_menu_is_empty_and_says_so(self) -> None:
        assert self.surface.is_empty
        assert self.surface.best is None

    def test_residual_is_the_whole_position(self) -> None:
        """The honest description of a token held naked across a shut anchor session."""
        assert self.surface.residual_after(None) == Decimal("1")

    def test_declining_to_hedge_scores_perfectly_when_nothing_is_placeable(self) -> None:
        """Not hedging is the only correct choice, so it must not be penalised."""
        assert self.surface.efficiency_of_choice(None) == Decimal("1")


class TestScoringTheAgentsChoice:
    """The ablation nobody runs: score the decision, not the outcome."""

    best = _candidate("BTC", risk_reduction="0.40", cost_bps="10")
    worse = _candidate("QQQ", risk_reduction="0.35", cost_bps="34")
    surface = HedgeabilitySurface((best, worse))

    def test_choosing_the_best_scores_one(self) -> None:
        assert self.surface.efficiency_of_choice(self.best) == Decimal("1")

    def test_choosing_a_worse_hedge_scores_below_one(self) -> None:
        score = self.surface.efficiency_of_choice(self.worse)
        assert Decimal("0") < score < Decimal("1")

    def test_failing_to_hedge_when_a_hedge_existed_scores_zero(self) -> None:
        assert self.surface.efficiency_of_choice(None) == Decimal("0")

    def test_residual_reflects_the_hedge_actually_chosen(self) -> None:
        assert self.surface.residual_after(self.best) < Decimal("1")
        assert self.surface.residual_after(self.worse) > self.surface.residual_after(self.best)


class TestTheOpenMarketCandidate:
    """The regular-hours branch used to hand the desk an empty surface.

    That put a false sentence into the record of every regular-hours decision — "no hedge
    placeable" — when what was true is narrower: the venue is open, the liquidity is there, and
    ARGUS has no equity broker. The distinction matters in the direction that flatters us, so it is
    pinned here: "the market gave me no hedge" excuses an unhedged position and "I have not built
    the connection" does not.
    """

    def test_an_open_anchor_is_carried_on_the_surface(self) -> None:
        surface = _hedge_surface(False, "NVDAUSDT")
        assert len(surface.candidates) == 1

    def test_it_is_not_placeable_because_we_have_no_broker(self) -> None:
        """No capability is invented: the menu is still empty, for the true reason."""
        surface = _hedge_surface(False, "NVDAUSDT")
        assert surface.is_empty
        assert surface.candidates[0].execution_probability == Decimal("0")

    def test_the_liquidity_is_recorded_as_present(self) -> None:
        """Zero liquidity would say the market had none, which is false while it is open."""
        assert _hedge_surface(False, "NVDAUSDT").candidates[0].liquidity_availability == Decimal(
            "1"
        )

    def test_the_note_names_the_capability_limit_not_a_market_state(self) -> None:
        note = _hedge_surface(False, "NVDAUSDT").session_note
        assert "no equity broker" in note
        assert "capability limit, not a market state" in note

    def test_the_shut_branch_still_says_the_market_is_shut(self) -> None:
        surface = _hedge_surface(True, "NVDAUSDT")
        assert surface.is_empty
        assert "anchor shut" in surface.session_note
        assert surface.candidates[0].liquidity_availability == Decimal("0")

    def test_the_two_states_do_not_share_a_reason(self) -> None:
        assert (
            _hedge_surface(True, "NVDAUSDT").session_note
            != _hedge_surface(False, "NVDAUSDT").session_note
        )

    def test_the_underlying_is_named_not_the_token(self) -> None:
        assert _hedge_surface(False, "NVDAUSDT").candidates[0].instrument == "NVDA"

    def test_a_broker_route_would_make_the_menu_live_with_no_other_change(self) -> None:
        """The day an execution route exists, one argument turns the candidate on."""
        live = open_market_candidate("NVDA", Decimal("0.98"), execution_probability=Decimal("0.9"))
        assert HedgeabilitySurface((live,)).menu

    def test_the_open_hedge_carries_a_real_cost_rather_than_a_flattering_zero(self) -> None:
        assert open_market_candidate("NVDA", Decimal("0.98")).execution_cost_bps > 0
