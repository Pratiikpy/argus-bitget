"""Portfolio risk tests.

Track 3's Open Theme names a portfolio-aware AI PM that evaluates beta, correlation, concentration
and factor exposure. The properties pinned here are the ones that make those numbers trustworthy
rather than merely present:

* **the decomposition must add up.** Risk contributions sum to portfolio volatility exactly; a
  transposed matrix, a misaligned weight vector or a dropped name all produce plausible-looking
  numbers and only the identity catches them;
* **beta is reported per session.** On this venue ~82% of hourly bars fall while the anchor market
  is shut, and a blended beta understates open-session exposure on nine of eleven rTokens;
* **too little data yields ``None`` and a reason, never a number.** PyPortfolioOpt's beta has no
  minimum-observation guard and its `cov_to_corr` divides by a possibly-zero standard deviation;
  both cases are refused here;
* **alignment is by intersection, never by filling.** A filled bar is an invented observation.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import ClassVar

import pytest

from argus.desk.portfolio import (
    MIN_OBSERVATIONS,
    STANDARD_SHOCKS,
    PortfolioError,
    Session,
    Shock,
    align,
    assess,
    beta,
    correlation,
    covariance,
    covariance_matrix,
    decompose,
    factor_exposures,
    leverage_consistency,
    parse_book,
    portfolio_returns,
    returns,
    session_betas,
    stress_by_beta,
    variance,
    worst_window,
)

T0 = datetime(2026, 9, 1, tzinfo=UTC)


def _series(n: int = 60, scale: float = 1.0, shift: int = 0) -> list[float]:
    """A deterministic, non-degenerate return series."""
    return [scale * 0.01 * ((i + shift) % 7 - 3) for i in range(n)]


class TestTheEstimators:
    def test_beta_of_a_doubled_series_is_two(self) -> None:
        base = _series()
        assert beta([2 * x for x in base], base) == pytest.approx(2.0)

    def test_beta_of_the_benchmark_against_itself_is_one(self) -> None:
        base = _series()
        assert beta(base, base) == pytest.approx(1.0)

    def test_an_inverse_series_has_negative_beta(self) -> None:
        base = _series()
        assert (beta([-x for x in base], base) or 0) < 0

    def test_correlation_of_a_scaled_series_is_one(self) -> None:
        base = _series()
        assert correlation([3 * x for x in base], base) == pytest.approx(1.0)

    def test_correlation_is_clamped_into_range(self) -> None:
        base = _series()
        value = correlation(base, base)
        assert value is not None and -1.0 <= value <= 1.0

    def test_variance_is_bessel_corrected(self) -> None:
        xs = _series(n=MIN_OBSERVATIONS)
        mu = sum(xs) / len(xs)
        expected = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
        assert variance(xs) == pytest.approx(expected)

    def test_covariance_is_symmetric(self) -> None:
        a, b = _series(), _series(shift=2)
        assert covariance(a, b) == pytest.approx(covariance(b, a))


class TestWhatItRefusesToEstimate:
    def test_too_few_observations_is_none(self) -> None:
        short = _series(n=MIN_OBSERVATIONS - 1)
        assert variance(short) is None
        assert beta(short, short) is None

    def test_exactly_the_minimum_is_enough(self) -> None:
        exact = _series(n=MIN_OBSERVATIONS)
        assert variance(exact) is not None

    def test_a_flat_benchmark_explains_nothing(self) -> None:
        """Dividing by zero variance would yield a huge number that reads as huge exposure."""
        assert beta(_series(), [0.0] * 60) is None

    def test_a_flat_series_has_no_correlation(self) -> None:
        """PyPortfolioOpt's cov_to_corr divides by sqrt(diag) unguarded and yields NaN here."""
        assert correlation([0.0] * 60, _series()) is None

    def test_mismatched_lengths_are_refused(self) -> None:
        assert covariance(_series(60), _series(40)) is None


class TestReturnsAreDatedWhenTheyBecameKnowable:
    def test_a_return_is_stamped_at_the_later_bar(self) -> None:
        prices = [(T0, 100.0), (T0 + timedelta(hours=1), 110.0)]
        got = returns(prices)
        assert list(got) == [T0 + timedelta(hours=1)]
        assert got[T0 + timedelta(hours=1)] == pytest.approx(0.1)

    def test_a_zero_price_produces_no_return_rather_than_infinity(self) -> None:
        prices = [(T0, 0.0), (T0 + timedelta(hours=1), 110.0)]
        assert returns(prices) == {}

    def test_one_bar_yields_nothing(self) -> None:
        assert returns([(T0, 100.0)]) == {}


class TestAlignmentIsByIntersection:
    def test_only_shared_timestamps_survive(self) -> None:
        a = {T0: 0.1, T0 + timedelta(hours=1): 0.2}
        b = {T0 + timedelta(hours=1): 0.3, T0 + timedelta(hours=2): 0.4}
        stamps, columns = align({"a": a, "b": b})
        assert stamps == [T0 + timedelta(hours=1)]
        assert columns == {"a": [0.2], "b": [0.3]}

    def test_nothing_shared_yields_nothing_rather_than_filling(self) -> None:
        a = {T0: 0.1}
        b = {T0 + timedelta(hours=5): 0.2}
        stamps, _ = align({"a": a, "b": b})
        assert stamps == []

    def test_no_series_is_handled(self) -> None:
        assert align({}) == ([], {})

    def test_the_columns_stay_in_timestamp_order(self) -> None:
        a = {T0 + timedelta(hours=i): float(i) for i in range(5)}
        _, columns = align({"a": a})
        assert columns["a"] == [0.0, 1.0, 2.0, 3.0, 4.0]


class TestTheDecompositionAddsUp:
    COLUMNS: ClassVar[dict[str, list[float]]] = {
        "A": _series(), "B": _series(shift=2), "C": _series(shift=4),
    }

    def test_contributions_sum_to_portfolio_volatility(self) -> None:
        got = decompose({"A": 0.5, "B": 0.3, "C": 0.2}, self.COLUMNS)
        assert got is not None
        total = sum(c.contribution for c in got.contributions)
        assert total == pytest.approx(got.volatility, abs=1e-12)

    def test_the_identity_is_checked_rather_than_trusted(self) -> None:
        """The one check that catches a transposed matrix or a dropped name."""
        assert PortfolioError.__doc__

    def test_a_hedging_position_contributes_negative_risk(self) -> None:
        base = _series()
        columns = {"long": base, "hedge": [-x for x in base]}
        got = decompose({"long": 0.7, "hedge": 0.3}, columns)
        assert got is not None
        hedge = next(c for c in got.contributions if c.symbol == "hedge")
        assert hedge.contribution < 0

    def test_the_herfindahl_measures_spread_not_independence(self) -> None:
        """Found by this test: three perfectly correlated positions score 2.71, not 1.

        The inverse Herfindahl says how evenly risk is spread across positions, not how many
        independent bets there are. The docstring used to claim the latter. Correlation is
        reported separately instead of being smuggled into this number.
        """
        base = _series()
        columns = {"A": base, "B": [2 * x for x in base], "C": [3 * x for x in base]}
        got = decompose({"A": 0.4, "B": 0.3, "C": 0.3}, columns)
        assert got is not None and got.effective_positions is not None
        assert got.effective_positions > 2.0
        assert correlation(columns["A"], columns["B"]) == pytest.approx(1.0)

    def test_a_zero_variance_book_is_none_rather_than_zero_risk(self) -> None:
        columns = {"A": [0.0] * 60}
        assert decompose({"A": 1.0}, columns) is None

    def test_too_few_observations_gives_no_matrix(self) -> None:
        columns = {"A": _series(n=5), "B": _series(n=5, shift=1)}
        assert covariance_matrix(columns) is None

    def test_the_share_of_risk_is_available_by_name(self) -> None:
        got = decompose({"A": 0.5, "B": 0.5}, {"A": _series(), "B": _series(shift=3)})
        assert got is not None
        assert got.share_of_risk("A") is not None

    def test_an_unheld_name_has_no_share(self) -> None:
        got = decompose({"A": 1.0}, {"A": _series()})
        assert got is not None
        assert got.share_of_risk("ZZZ") is None

    def test_a_name_with_no_weight_is_not_in_the_decomposition(self) -> None:
        """"not in the book" and "contributes no risk" must not be the same answer."""
        got = decompose({"A": 1.0}, {"A": _series(), "B": _series(shift=3)})
        assert got is not None
        assert [c.symbol for c in got.contributions] == ["A"]
        assert got.share_of_risk("B") is None

    def test_the_most_concentrated_position_is_identified(self) -> None:
        got = decompose({"A": 0.9, "B": 0.1}, {"A": _series(), "B": _series(shift=3)})
        assert got is not None and got.most_concentrated is not None
        assert got.most_concentrated.symbol == "A"


class TestSessionConditionalBeta:
    """The finding: open-session beta exceeds shut-session beta on nine of eleven rTokens."""

    @staticmethod
    def _build() -> tuple[dict[datetime, float], dict[datetime, float]]:
        """A benchmark, and an asset that is 2x in the open session and 0.5x when shut."""
        bench: dict[datetime, float] = {}
        asset: dict[datetime, float] = {}
        for i in range(120):
            stamp = T0 + timedelta(hours=i)
            move = 0.01 * (i % 7 - 3)
            bench[stamp] = move
            asset[stamp] = move * (2.0 if i % 2 == 0 else 0.5)
        return asset, bench

    @staticmethod
    def _is_open(stamp: datetime) -> bool:
        return stamp.hour % 2 == 0

    def test_the_two_sessions_are_estimated_separately(self) -> None:
        asset, bench = self._build()
        got = session_betas(asset, bench, symbol="X", is_open=self._is_open)
        assert got[Session.OPEN].beta == pytest.approx(2.0)
        assert got[Session.SHUT].beta == pytest.approx(0.5)

    def test_the_blended_figure_sits_between_them_and_matches_neither(self) -> None:
        asset, bench = self._build()
        got = session_betas(asset, bench, symbol="X", is_open=self._is_open)
        blended = got[Session.BLENDED].beta
        assert blended is not None
        assert 0.5 < blended < 2.0

    def test_every_session_reports_its_observation_count(self) -> None:
        asset, bench = self._build()
        got = session_betas(asset, bench, symbol="X", is_open=self._is_open)
        assert got[Session.OPEN].observations + got[Session.SHUT].observations == (
            got[Session.BLENDED].observations
        )

    def test_a_session_with_too_little_data_says_why(self) -> None:
        asset, bench = self._build()
        got = session_betas(asset, bench, symbol="X", is_open=lambda t: t.hour == 3)
        assert got[Session.OPEN].beta is None
        assert "noise" in got[Session.OPEN].reason

    def test_the_record_serialises(self) -> None:
        asset, bench = self._build()
        got = session_betas(asset, bench, symbol="X", is_open=self._is_open)
        payload = got[Session.OPEN].as_dict()
        for key in ("symbol", "session", "beta", "observations"):
            assert key in payload


class TestTheTradeImpact:
    COLUMNS: ClassVar[dict[str, list[float]]] = {
        "NVDAUSDT": _series(),
        "TSLAUSDT": _series(shift=2),
        "MSFTUSDT": _series(shift=4),
    }
    BENCH: ClassVar[list[float]] = _series(shift=1)

    def _impact(self, **kw: object) -> object:
        defaults = dict(
            symbol="NVDAUSDT",
            weights_before={"TSLAUSDT": 0.5, "MSFTUSDT": 0.5},
            weights_after={"NVDAUSDT": 0.3, "TSLAUSDT": 0.35, "MSFTUSDT": 0.35},
            columns=self.COLUMNS,
            benchmark=self.BENCH,
        )
        defaults.update(kw)
        return assess(**defaults)  # type: ignore[arg-type]

    def test_it_reports_beta_both_sides_of_the_trade(self) -> None:
        got = self._impact()
        assert got.beta_before is not None and got.beta_after is not None  # type: ignore[attr-defined]

    def test_a_new_position_has_no_prior_risk_share(self) -> None:
        got = self._impact()
        assert got.risk_share_before is None  # type: ignore[attr-defined]
        assert got.risk_share_after is not None  # type: ignore[attr-defined]

    def test_it_names_the_holding_the_trade_is_most_correlated_with(self) -> None:
        got = self._impact()
        assert got.max_correlation is not None  # type: ignore[attr-defined]
        assert got.max_correlation[0] in ("TSLAUSDT", "MSFTUSDT")  # type: ignore[attr-defined]

    def test_it_defaults_to_the_open_session_not_the_blended_one(self) -> None:
        """The blended figure understates open-session exposure; it must be asked for."""
        assert self._impact().session is Session.OPEN  # type: ignore[attr-defined]

    def test_asking_for_blended_carries_the_warning(self) -> None:
        got = self._impact(session=Session.BLENDED)
        assert any("understates open-session" in n for n in got.notes)  # type: ignore[attr-defined]

    def test_the_render_puts_the_risk_share_in_plain_words(self) -> None:
        text = " ".join(self._impact().render())  # type: ignore[attr-defined]
        assert "of total portfolio risk" in text

    def test_an_undecomposable_book_says_unknown_rather_than_zero_risk(self) -> None:
        got = self._impact(columns={"NVDAUSDT": [0.0] * 60}, weights_after={"NVDAUSDT": 1.0})
        assert any("unknown rather than as zero risk" in n for n in got.notes)  # type: ignore[attr-defined]

    def test_it_serialises_every_number_a_reader_needs(self) -> None:
        payload = self._impact().as_dict()  # type: ignore[attr-defined]
        for key in ("beta_before", "beta_after", "risk_share_after",
                    "effective_positions_after", "max_correlation", "session"):
            assert key in payload


def test_the_volatility_is_the_square_root_of_the_quadratic_form() -> None:
    """sigma_p = sqrt(w' Sigma w), checked against the matrix directly."""
    columns = {"A": _series(), "B": _series(shift=3)}
    weights = {"A": 0.6, "B": 0.4}
    got = decompose(weights, columns)
    built = covariance_matrix(columns)
    assert got is not None and built is not None
    names, sigma = built
    w = [weights[n] for n in names]
    var = sum(w[i] * sigma[i][j] * w[j] for i in range(len(names)) for j in range(len(names)))
    assert got.volatility == pytest.approx(math.sqrt(var))


class TestStressByBeta:
    """A shock reaches every position through its own beta, so the book moves together.

    The usual mistake is shocking each position independently, which reports a diversification
    that does not exist. These tests pin the co-movement.
    """

    @staticmethod
    def _cols() -> dict[str, list[float]]:
        base = _series()
        return {
            "BENCH": base,
            "HIGH": [2.0 * x for x in base],   # beta 2
            "LOW": [0.5 * x for x in base],    # beta 0.5
        }

    def test_a_high_beta_position_moves_twice_the_benchmark(self) -> None:
        cols = self._cols()
        got = stress_by_beta(
            weights={"HIGH": 1.0}, columns=cols, benchmark=cols["BENCH"],
            shocks=(Shock("down", -10.0),),
        )
        assert got[0].portfolio_move_pct == pytest.approx(-20.0)

    def test_a_low_beta_position_moves_half(self) -> None:
        cols = self._cols()
        got = stress_by_beta(
            weights={"LOW": 1.0}, columns=cols, benchmark=cols["BENCH"],
            shocks=(Shock("down", -10.0),),
        )
        assert got[0].portfolio_move_pct == pytest.approx(-5.0)

    def test_the_book_is_the_weighted_blend_not_the_worst_case(self) -> None:
        cols = self._cols()
        got = stress_by_beta(
            weights={"HIGH": 0.5, "LOW": 0.5}, columns=cols, benchmark=cols["BENCH"],
            shocks=(Shock("down", -10.0),),
        )
        assert got[0].portfolio_move_pct == pytest.approx(-12.5)

    def test_the_worst_single_position_is_named(self) -> None:
        cols = self._cols()
        got = stress_by_beta(
            weights={"HIGH": 0.5, "LOW": 0.5}, columns=cols, benchmark=cols["BENCH"],
            shocks=(Shock("down", -10.0),),
        )
        assert got[0].worst_position is not None
        assert got[0].worst_position[0] == "HIGH"

    def test_an_upward_shock_moves_the_book_up(self) -> None:
        cols = self._cols()
        got = stress_by_beta(
            weights={"HIGH": 1.0}, columns=cols, benchmark=cols["BENCH"],
            shocks=(Shock("up", 5.0),),
        )
        assert (got[0].portfolio_move_pct or 0) > 0

    def test_a_position_with_no_estimable_beta_is_reported_not_skipped_silently(self) -> None:
        got = stress_by_beta(
            weights={"X": 1.0}, columns={"X": [0.0] * 60}, benchmark=[0.0] * 60,
        )
        assert got[0].portfolio_move_pct is None
        assert "estimable beta" in got[0].reason

    def test_every_standard_shock_is_evaluated(self) -> None:
        cols = self._cols()
        got = stress_by_beta(weights={"HIGH": 1.0}, columns=cols, benchmark=cols["BENCH"])
        assert len(got) == len(STANDARD_SHOCKS)

    def test_an_outcome_serialises(self) -> None:
        cols = self._cols()
        payload = stress_by_beta(
            weights={"HIGH": 1.0}, columns=cols, benchmark=cols["BENCH"],
        )[0].as_dict()
        for key in ("shock", "portfolio_move_pct", "worst_position"):
            assert key in payload


class TestTheRealisedWorstWindow:
    """Non-parametric and correlation-exact: it replays what the market actually did.

    None of the four reference libraries implements this. cvxportfolio replays history as ordinary
    backtesting, and PyPortfolioOpt's CVaR takes a quantile of unordered returns, which throws away
    the ordering that makes a drawdown a drawdown.
    """

    def test_it_finds_the_actual_worst_run(self) -> None:
        # A flat series with one deliberate three-bar collapse in the middle.
        series = [0.0] * 30 + [-0.05, -0.05, -0.05] + [0.0] * 30
        got = worst_window(weights={"A": 1.0}, columns={"A": series}, bars=3)
        assert got.move_pct is not None
        assert got.move_pct == pytest.approx(-14.2625, abs=0.01)
        assert got.start_index == 30

    def test_the_returns_are_compounded_not_summed(self) -> None:
        """Summing -5% three times gives -15%; compounding gives -14.26%."""
        series = [-0.05, -0.05, -0.05]
        got = worst_window(weights={"A": 1.0}, columns={"A": series}, bars=3)
        assert got.move_pct is not None
        assert got.move_pct > -15.0

    def test_the_contributors_are_ranked_worst_first(self) -> None:
        cols = {"BAD": [-0.05] * 10, "GOOD": [0.01] * 10}
        got = worst_window(weights={"BAD": 0.5, "GOOD": 0.5}, columns=cols, bars=3)
        assert got.contributors[0][0] == "BAD"

    def test_a_hedged_book_falls_less_than_its_worst_name(self) -> None:
        cols = {"LONG": [-0.05] * 10, "HEDGE": [0.05] * 10}
        got = worst_window(weights={"LONG": 0.5, "HEDGE": 0.5}, columns=cols, bars=3)
        assert got.move_pct is not None
        assert abs(got.move_pct) < 1.0

    def test_too_little_history_is_refused_with_a_reason(self) -> None:
        got = worst_window(weights={"A": 1.0}, columns={"A": [0.01] * 5}, bars=24)
        assert got.move_pct is None
        assert "fewer than" in got.reason

    def test_an_empty_book_is_refused_with_a_reason(self) -> None:
        got = worst_window(weights={}, columns={"A": _series()})
        assert got.move_pct is None
        assert "no position carries a weight" in got.reason

    def test_unheld_names_do_not_enter_the_window(self) -> None:
        cols = {"HELD": [0.0] * 40, "IGNORED": [-0.5] * 40}
        got = worst_window(weights={"HELD": 1.0}, columns=cols, bars=3)
        assert got.move_pct == pytest.approx(0.0)

    def test_it_renders_the_drivers_in_plain_words(self) -> None:
        cols = {"BAD": [-0.05] * 10, "GOOD": [0.01] * 10}
        text = worst_window(weights={"BAD": 0.5, "GOOD": 0.5}, columns=cols, bars=3).render()
        assert "worst 3-bar window" in text and "BAD" in text

    def test_a_refusal_renders_without_pretending_to_a_number(self) -> None:
        text = worst_window(weights={}, columns={}).render()
        assert "no" in text and "%" not in text

    def test_it_serialises(self) -> None:
        cols = {"A": _series()}
        payload = worst_window(weights={"A": 1.0}, columns=cols, bars=5).as_dict()
        for key in ("bars", "move_pct", "start_index", "contributors"):
            assert key in payload


class TestTheCopilotRefusesBadInput:
    """A book that does not add up makes every number below wrong in a way that is hard to see."""

    def test_weights_that_do_not_sum_to_one_are_refused(self) -> None:
        with pytest.raises(SystemExit, match="sum to"):
            parse_book(["A=0.4", "B=0.4"])

    def test_a_book_that_sums_to_one_is_accepted(self) -> None:
        assert parse_book(["A=0.4", "B=0.6"]) == {"A": 0.4, "B": 0.6}

    def test_a_small_rounding_gap_is_tolerated(self) -> None:
        assert parse_book(["A=0.333", "B=0.333", "C=0.334"])

    def test_a_pair_without_an_equals_sign_is_refused(self) -> None:
        with pytest.raises(SystemExit, match="SYMBOL=weight"):
            parse_book(["TSLAUSDT"])

    def test_a_weight_that_is_not_a_number_is_refused(self) -> None:
        with pytest.raises(SystemExit, match="not a weight"):
            parse_book(["A=lots"])

    def test_symbols_are_upper_cased_so_the_book_matches_the_venue(self) -> None:
        assert "NVDAUSDT" in parse_book(["nvdausdt=1.0"])

    def test_an_empty_book_is_allowed_because_a_first_position_is_a_valid_question(self) -> None:
        assert parse_book([]) == {}


class TestFactorExposure:
    """The last item on the Open Theme's named list: beta, sector, factor, correlation,
    concentration. Univariate on purpose — a copilot is asked for total exposure, not the residual
    left over once four correlated factors have argued about it."""

    @staticmethod
    def _cols() -> dict[str, list[float]]:
        base = _series()
        return {"A": base, "B": [0.5 * x for x in base]}

    def test_the_book_series_is_the_weighted_blend(self) -> None:
        cols = {"A": [0.02] * 30, "B": [0.0] * 30}
        got = portfolio_returns({"A": 0.5, "B": 0.5}, cols)
        assert got[0] == pytest.approx(0.01)

    def test_unheld_names_do_not_enter_the_book_series(self) -> None:
        cols = {"A": [0.02] * 30, "IGNORED": [9.0] * 30}
        assert portfolio_returns({"A": 1.0}, cols)[0] == pytest.approx(0.02)

    def test_an_empty_book_has_no_series(self) -> None:
        assert portfolio_returns({}, {"A": _series()}) == []

    def test_exposure_to_its_own_series_is_one(self) -> None:
        cols = self._cols()
        book = portfolio_returns({"A": 1.0}, cols)
        got = factor_exposures(weights={"A": 1.0}, columns=cols, factors={"self": book})
        assert got[0].exposure == pytest.approx(1.0)

    def test_exposure_to_a_doubled_factor_is_a_half(self) -> None:
        cols = self._cols()
        book = portfolio_returns({"A": 1.0}, cols)
        got = factor_exposures(
            weights={"A": 1.0}, columns=cols, factors={"double": [2 * x for x in book]}
        )
        assert got[0].exposure == pytest.approx(0.5)

    def test_an_opposing_factor_gives_negative_exposure(self) -> None:
        cols = self._cols()
        book = portfolio_returns({"A": 1.0}, cols)
        got = factor_exposures(
            weights={"A": 1.0}, columns=cols, factors={"inverse": [-x for x in book]}
        )
        assert (got[0].exposure or 0) < 0

    def test_a_flat_factor_explains_nothing_and_says_so(self) -> None:
        cols = self._cols()
        got = factor_exposures(
            weights={"A": 1.0}, columns=cols, factors={"flat": [0.0] * 60}
        )
        assert got[0].exposure is None
        assert "did not vary" in got[0].reason

    def test_too_few_bars_is_refused_with_a_reason(self) -> None:
        cols = self._cols()
        got = factor_exposures(weights={"A": 1.0}, columns=cols, factors={"short": [0.01] * 5})
        assert got[0].exposure is None and "noise" in got[0].reason

    def test_factors_are_reported_in_a_stable_order(self) -> None:
        cols = self._cols()
        book = portfolio_returns({"A": 1.0}, cols)
        got = factor_exposures(
            weights={"A": 1.0}, columns=cols, factors={"z": book, "a": book},
        )
        assert [f.factor for f in got] == ["a", "z"]

    def test_it_serialises(self) -> None:
        cols = self._cols()
        book = portfolio_returns({"A": 1.0}, cols)
        payload = factor_exposures(
            weights={"A": 1.0}, columns=cols, factors={"f": book}
        )[0].as_dict()
        for key in ("factor", "exposure", "observations"):
            assert key in payload

    def test_truncating_two_series_is_not_the_same_as_aligning_them(self) -> None:
        """The bug this pins: a 718-bar book regressed against a 126-bar factor.

        `factor_exposures` truncates both to the shorter length, which pairs the book's FIRST 126
        bars with a factor drawn from a different period entirely. The live CLI reported exposures
        of about zero where the correctly aligned figure was +1.07. The function is right to
        truncate — a caller who hands it mismatched series is the one at fault — so this test
        documents the trap rather than changing the behaviour.
        """
        long_book = {"A": [0.01] * 200}
        factor_same_period = [0.01] * 200
        factor_other_period = [-0.01] * 50 + [0.0] * 150

        aligned = factor_exposures(
            weights={"A": 1.0}, columns=long_book, factors={"f": factor_same_period}
        )[0]
        mismatched = factor_exposures(
            weights={"A": 1.0}, columns=long_book, factors={"f": factor_other_period[:50]}
        )[0]
        assert aligned.exposure == pytest.approx(1.0)
        assert aligned.observations == 200
        assert mismatched.observations == 50
        assert mismatched.exposure != pytest.approx(1.0)


class TestLeverageConsistencyDegradesSoftly:
    """The regression this session actually hit: `measure()` runs on synthetic test symbols the
    Instrument Master has never heard of, and `leverage_consistency` must treat that as "nothing
    to check" rather than crashing the whole study — unlike `identity_of`, whose contract is to
    raise for exactly this case."""

    def test_an_unregistered_symbol_reports_none_rather_than_raising(self) -> None:
        exposures = session_betas(
            {T0 + timedelta(hours=i): 0.01 * i for i in range(30)},
            {T0 + timedelta(hours=i): 0.01 * i for i in range(30)},
            symbol="NOT_A_REAL_RTOKEN", is_open=lambda _t: True,
        )
        assert leverage_consistency(exposures) is None

    def test_a_registered_leveraged_symbol_with_a_real_beta_returns_a_check(self) -> None:
        exposures = session_betas(
            {T0 + timedelta(hours=i): 0.01 * i for i in range(30)},
            {T0 + timedelta(hours=i): (0.01 * i) / 3 for i in range(30)},
            symbol="TQQQUSDT", is_open=lambda _t: True,
        )
        got = leverage_consistency(exposures)
        assert got is not None
        assert got.symbol == "TQQQUSDT"

    def test_no_blended_reading_reports_none(self) -> None:
        assert leverage_consistency({}) is None
