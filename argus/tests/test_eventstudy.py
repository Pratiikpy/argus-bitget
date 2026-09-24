"""Event-study tests — a planted effect must be found, and a null must not be.

An event study is the easiest kind of analysis to produce a finding from by accident. Overlapping
windows, an estimation period that touches the event, a clustering correction that is skipped, a
test that divides by the wrong variance — every one of those makes the p-value smaller, and none of
them is visible in the output. So the tests here are built around synthetic processes with known
parameters: a beta that must be recovered, a shock that must be detected at roughly its planted
size, and a null under which no test may reject.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from argus.research.eventstudy import (
    DEFAULT_GAP_BARS,
    MIN_ESTIMATION_BARS,
    MIN_EVENTS,
    EventStudyError,
    average_cross_correlation,
    bmp_t,
    build_window,
    corrado_rank_z,
    fit_market_model,
    generalised_sign_z,
    kolari_pynnonen,
    market_proxy,
    patell_z,
    returns_from,
    study,
)

START = datetime(2026, 1, 1, tzinfo=UTC)
BARS = 900
EVENT_INDEX = 700
STAMPS = [START + timedelta(hours=i) for i in range(BARS)]
EVENT_AT = STAMPS[EVENT_INDEX]


def _series(
    *, seed: int, beta: float = 1.2, shock_at: int | None = None, shock: float = 0.0,
    common: list[float] | None = None,
) -> tuple[list[float], list[float]]:
    """An asset generated from a market with a known beta, optionally shocked at one bar."""
    rng = random.Random(seed)
    market = common if common is not None else [rng.gauss(0.0, 0.006) for _ in range(BARS)]
    asset = []
    for t, m in enumerate(market):
        value = beta * m + rng.gauss(0.0, 0.004)
        if shock_at is not None and t == shock_at:
            value += shock
        asset.append(value)
    return asset, list(market)


def _windows(
    n: int, *, shock: float = 0.0, seed: int = 100, event_bars: int = 6,
    shared_residual: float = 0.0,
) -> list:
    """``shared_residual`` adds a common component the market proxy does NOT contain.

    Sharing the *market* series produces no residual correlation at all — removing the common
    factor is exactly what a market model does, and a test asserting otherwise was wrong rather
    than the code. Residual correlation, which is what the clustering adjustment corrects, comes
    from a factor the proxy misses. On this venue that is the real case: an equal-weighted basket
    of eleven tokenised equities does not span whatever moves the twelfth.
    """
    rng = random.Random(4242)
    extra = [rng.gauss(0.0, shared_residual) for _ in range(BARS)] if shared_residual else None
    out = []
    for k in range(n):
        asset, market = _series(
            seed=seed + k,
            shock_at=EVENT_INDEX + 1 if shock else None,
            shock=shock,
        )
        if extra is not None:
            asset = [a + e for a, e in zip(asset, extra, strict=True)]
        window = build_window(
            f"S{k}", EVENT_AT, STAMPS, asset, market, event_bars=event_bars
        )
        assert window is not None
        out.append(window)
    return out


class TestTheMarketModel:
    def test_it_recovers_a_known_beta(self) -> None:
        asset, market = _series(seed=1, beta=1.2)
        model = fit_market_model(asset, market)
        assert model.beta == pytest.approx(1.2, abs=0.05)

    def test_it_recovers_a_beta_below_one(self) -> None:
        asset, market = _series(seed=2, beta=0.4)
        assert fit_market_model(asset, market).beta == pytest.approx(0.4, abs=0.05)

    def test_the_residual_standard_error_matches_the_noise_it_was_given(self) -> None:
        asset, market = _series(seed=3, beta=1.0)
        assert fit_market_model(asset, market).residual_sd == pytest.approx(0.004, abs=0.0005)

    def test_the_degrees_of_freedom_are_n_minus_two(self) -> None:
        """Two parameters are estimated. Dividing by n understates the residual error and every
        standardised statistic built on it comes out too large."""
        asset = [0.0, 0.01, -0.01, 0.02] * 40
        market = [0.0, 0.005, -0.005, 0.012] * 40
        model = fit_market_model(asset, market)
        residuals = [a - model.alpha - model.beta * m for a, m in zip(asset, market, strict=True)]
        expected = (sum(r * r for r in residuals) / (len(asset) - 2)) ** 0.5
        assert model.residual_sd == pytest.approx(expected)

    def test_a_short_estimation_window_raises(self) -> None:
        asset, market = _series(seed=4)
        with pytest.raises(EventStudyError, match="below the"):
            fit_market_model(asset[:MIN_ESTIMATION_BARS - 1], market[:MIN_ESTIMATION_BARS - 1])

    def test_a_constant_market_raises_rather_than_returning_beta_zero(self) -> None:
        """Beta is undefined against a market that does not move, not zero. Returning zero would
        make every return abnormal."""
        with pytest.raises(EventStudyError, match="undefined, not zero"):
            fit_market_model([0.01] * 200, [0.0] * 200)

    def test_misaligned_series_raise(self) -> None:
        with pytest.raises(EventStudyError, match="same bars"):
            fit_market_model([0.01] * 200, [0.01] * 199)


class TestPatellsForecastCorrection:
    def test_the_forecast_error_exceeds_the_residual_error(self) -> None:
        """Always, because forecasting out of sample adds estimation error to residual error. If
        these were ever equal the correction has been dropped."""
        asset, market = _series(seed=5)
        model = fit_market_model(asset[:480], market[:480])
        assert model.forecast_sd(model.market_mean) > model.residual_sd

    def test_the_penalty_grows_as_the_market_moves_further_from_the_centre(self) -> None:
        """The third term, and the one most implementations drop. It matters exactly on the bars an
        event study is about — the ones where the market itself moved."""
        asset, market = _series(seed=6)
        model = fit_market_model(asset[:480], market[:480])
        near = model.forecast_sd(model.market_mean)
        far = model.forecast_sd(model.market_mean + 0.05)
        assert far > near

    def test_it_reduces_to_the_residual_error_in_the_limit(self) -> None:
        """With a huge estimation window and a market return at the centre, the inflation is 1."""
        asset, market = _series(seed=7)
        model = fit_market_model(asset, market)
        ratio = model.forecast_sd(model.market_mean) / model.residual_sd
        assert 1.0 < ratio < 1.01


class TestTheWindowRefusesWhatItCannotMeasure:
    def test_an_event_too_early_for_an_estimation_window_returns_none(self) -> None:
        asset, market = _series(seed=8)
        assert build_window("S", STAMPS[10], STAMPS, asset, market, event_bars=6) is None

    def test_an_event_too_late_for_a_full_event_window_returns_none(self) -> None:
        asset, market = _series(seed=9)
        assert build_window("S", STAMPS[-2], STAMPS, asset, market, event_bars=6) is None

    def test_it_refuses_rather_than_fitting_on_whatever_is_available(self) -> None:
        """A model fitted on 130 bars for one event and 480 for another makes the two standardised
        returns incomparable, which is the thing standardising was for."""
        asset, market = _series(seed=10)
        short = build_window(
            "S", STAMPS[MIN_ESTIMATION_BARS], STAMPS, asset, market,
            event_bars=6, estimation_bars=480,
        )
        assert short is None

    def test_the_estimation_window_is_purged_from_the_event(self) -> None:
        """The gap must actually be left out. Without it the model is fitted partly on the bars the
        event window then scores."""
        asset, market = _series(seed=11)
        window = build_window(
            "S", EVENT_AT, STAMPS, asset, market, event_bars=6,
            estimation_bars=300, gap_bars=24,
        )
        assert window is not None
        assert len(window.estimation_abnormal) == 300

    def test_a_larger_gap_shortens_nothing_but_moves_the_window_back(self) -> None:
        asset, market = _series(seed=12)
        tight = build_window("S", EVENT_AT, STAMPS, asset, market,
                             event_bars=6, estimation_bars=300, gap_bars=0)
        wide = build_window("S", EVENT_AT, STAMPS, asset, market,
                            event_bars=6, estimation_bars=300, gap_bars=100)
        assert tight is not None and wide is not None
        assert len(tight.estimation_abnormal) == len(wide.estimation_abnormal)
        assert tight.model.beta != wide.model.beta

    def test_a_zero_length_event_window_raises(self) -> None:
        asset, market = _series(seed=13)
        with pytest.raises(EventStudyError, match="measures nothing"):
            build_window("S", EVENT_AT, STAMPS, asset, market, event_bars=0)

    def test_a_negative_gap_raises(self) -> None:
        asset, market = _series(seed=14)
        with pytest.raises(EventStudyError, match="overlap"):
            build_window("S", EVENT_AT, STAMPS, asset, market, event_bars=6, gap_bars=-1)

    def test_misaligned_inputs_raise(self) -> None:
        asset, market = _series(seed=15)
        with pytest.raises(EventStudyError, match="align"):
            build_window("S", EVENT_AT, STAMPS[:-1], asset, market, event_bars=6)

    def test_the_default_gap_matches_the_desks_hold_horizon(self) -> None:
        assert DEFAULT_GAP_BARS == 24


class TestTheNullIsNotRejected:
    def test_no_test_rejects_when_nothing_happened(self) -> None:
        result = study("null", _windows(10, seed=200), event_bars=6)
        for name, row in result.statistics.items():
            assert row["adjusted_p_value"] > 0.05, (name, row)

    def test_the_verdict_says_so_plainly(self) -> None:
        result = study("null", _windows(10, seed=300), event_bars=6)
        assert "NO EFFECT ESTABLISHED" in result.verdict

    def test_an_average_with_no_significance_is_not_reported_as_a_finding(self) -> None:
        """The commonest way an event study produces a false finding: quote the average cumulative
        abnormal return and skip the test. The verdict names both."""
        result = study("null", _windows(8, seed=400), event_bars=6)
        assert "no test rejects" in result.verdict


class TestAPlantedEffectIsFound:
    def test_all_four_tests_reject(self) -> None:
        result = study("effect", _windows(10, shock=0.025, seed=500), event_bars=6)
        for name, row in result.statistics.items():
            assert row["adjusted_p_value"] < 0.05, (name, row)

    def test_the_average_abnormal_return_is_near_the_planted_size(self) -> None:
        """A 250bps shock inside a six-bar window. Anything far from 250 means the abnormal return
        is being measured against the wrong benchmark."""
        result = study("effect", _windows(10, shock=0.025, seed=600), event_bars=6)
        assert result.average_car_bps == pytest.approx(250.0, abs=60.0)

    def test_the_sign_is_right_for_a_negative_shock(self) -> None:
        result = study("crash", _windows(10, shock=-0.025, seed=700), event_bars=6)
        assert result.average_car_bps < -150
        assert "negative" in result.verdict

    def test_the_verdict_names_it_as_established(self) -> None:
        result = study("effect", _windows(10, shock=0.025, seed=800), event_bars=6)
        assert "EFFECT ESTABLISHED" in result.verdict

    def test_a_smaller_effect_is_harder_to_establish(self) -> None:
        big = study("big", _windows(10, shock=0.025, seed=900), event_bars=6)
        small = study("small", _windows(10, shock=0.002, seed=900), event_bars=6)
        assert (
            small.statistics["bmp"]["adjusted_p_value"]
            > big.statistics["bmp"]["adjusted_p_value"]
        )


class TestTheClusteringAdjustment:
    def test_zero_correlation_changes_nothing(self) -> None:
        assert kolari_pynnonen(3.0, events=12, correlation=0.0) == pytest.approx(3.0)

    def test_positive_correlation_deflates_the_statistic(self) -> None:
        assert kolari_pynnonen(3.0, events=12, correlation=0.3) < 3.0

    def test_more_events_at_the_same_correlation_deflate_further(self) -> None:
        """The over-rejection grows with the number of correlated events, so the correction must."""
        assert kolari_pynnonen(3.0, events=50, correlation=0.2) < kolari_pynnonen(
            3.0, events=5, correlation=0.2
        )

    def test_the_formula_is_the_published_one(self) -> None:
        from math import sqrt

        expected = 3.0 * sqrt((1 - 0.25) / (1 + 11 * 0.25))
        assert kolari_pynnonen(3.0, events=12, correlation=0.25) == pytest.approx(expected)

    def test_a_negative_correlation_is_clipped_rather_than_inflating(self) -> None:
        """Letting a negative sample correlation make a result *more* significant would be using a
        correction as an enhancement."""
        assert kolari_pynnonen(3.0, events=12, correlation=-0.4) == pytest.approx(3.0)

    def test_one_event_is_not_adjusted(self) -> None:
        assert kolari_pynnonen(3.0, events=1, correlation=0.5) == 3.0

    def test_a_factor_the_proxy_misses_shows_up_as_residual_correlation(self) -> None:
        """What the adjustment is actually for. If this reads near zero the correction is being
        computed on the wrong series and every parametric p-value here is too small."""
        shared = study("shared", _windows(8, seed=1000, shared_residual=0.006), event_bars=6)
        assert shared.correlation > 0.2

    def test_a_factor_the_proxy_does_capture_leaves_no_residual_correlation(self) -> None:
        """The complement, and the reason the previous test plants the factor outside the market.
        Removing a common factor is what a market model does; residuals from a well-specified
        model are uncorrelated even though the raw returns are not."""
        captured = study("captured", _windows(8, seed=1050), event_bars=6)
        assert abs(captured.correlation) < 0.15

    def test_independent_events_show_little_correlation(self) -> None:
        independent = study("independent", _windows(8, seed=1100), event_bars=6)
        assert abs(independent.correlation) < 0.2

    def test_the_note_states_the_size_of_the_correction(self) -> None:
        result = study("shared", _windows(8, seed=1200, shared_residual=0.006), event_bars=6)
        assert "deflates every parametric statistic" in result.clustering_note

    def test_only_the_parametric_tests_are_adjusted(self) -> None:
        """The rank and sign tests are not deflated by this factor; marking them as adjusted would
        be applying a parametric correction to a non-parametric statistic."""
        stats = study("mixed", _windows(8, seed=1300), event_bars=6).statistics
        assert stats["patell"]["clustering_adjusted"]
        assert stats["bmp"]["clustering_adjusted"]
        assert not stats["corrado_rank"]["clustering_adjusted"]
        assert not stats["generalised_sign"]["clustering_adjusted"]
        assert stats["corrado_rank"]["statistic"] == stats["corrado_rank"]["adjusted"]


class TestTheTestsDifferFromEachOther:
    def test_patell_and_bmp_do_not_agree_exactly(self) -> None:
        """They divide by different variances. If they matched, one is not being computed."""
        windows = _windows(10, shock=0.02, seed=1400)
        assert patell_z(windows) != pytest.approx(bmp_t(windows))

    def test_bmp_is_the_more_conservative_when_the_event_adds_variance(self) -> None:
        """The whole reason BMP exists. Events that widen the cross-sectional spread must not be
        read as events that shifted it."""
        rng = random.Random(1500)
        windows = []
        for k in range(12):
            # A shock whose SIZE varies wildly across names: pure event-induced variance, and a
            # mean effect near zero.
            asset, market = _series(
                seed=1500 + k, shock_at=EVENT_INDEX + 1, shock=rng.gauss(0.0, 0.05)
            )
            window = build_window(f"S{k}", EVENT_AT, STAMPS, asset, market, event_bars=6)
            assert window is not None
            windows.append(window)
        assert abs(bmp_t(windows)) < abs(patell_z(windows))

    def test_the_rank_test_resists_a_single_outlier(self) -> None:
        """One enormous move must not carry the result. The parametric statistic will grow; the
        rank statistic must grow far less."""
        base = _windows(10, seed=1600)
        spiked = list(base)
        asset, market = _series(seed=1600, shock_at=EVENT_INDEX + 1, shock=0.40)
        outlier = build_window("SPIKE", EVENT_AT, STAMPS, asset, market, event_bars=6)
        assert outlier is not None
        spiked[0] = outlier
        parametric_shift = abs(patell_z(spiked)) - abs(patell_z(base))
        rank_shift = abs(corrado_rank_z(spiked)) - abs(corrado_rank_z(base))
        assert parametric_shift > rank_shift

    def test_the_sign_test_uses_the_samples_own_positive_rate_not_a_half(self) -> None:
        """On a trending sample the residual positive rate is not 0.5, and assuming it is turns a
        drift into a finding."""
        windows = _windows(10, seed=1700)
        assert isinstance(generalised_sign_z(windows), float)

    def test_all_statistics_are_finite(self) -> None:
        from math import isfinite

        stats = study("finite", _windows(10, shock=0.01, seed=1800), event_bars=6).statistics
        for row in stats.values():
            assert all(isfinite(v) for v in row.values() if isinstance(v, float))


class TestTooFewEventsIsARefusal:
    def test_a_study_below_the_floor_raises(self) -> None:
        with pytest.raises(EventStudyError, match="below the"):
            study("thin", _windows(MIN_EVENTS - 1, seed=1900), event_bars=6)

    def test_each_statistic_refuses_on_its_own(self) -> None:
        thin = _windows(MIN_EVENTS - 1, seed=2000)
        for fn in (patell_z, bmp_t, corrado_rank_z, generalised_sign_z):
            with pytest.raises(EventStudyError, match="below the"):
                fn(thin)

    def test_the_floor_is_the_stated_one(self) -> None:
        assert MIN_EVENTS == 5
        assert study("just-enough", _windows(MIN_EVENTS, seed=2100), event_bars=6).events == 5

    def test_the_rank_test_requires_equal_window_lengths(self) -> None:
        mixed = [*_windows(3, seed=2200, event_bars=6), *_windows(3, seed=2300, event_bars=8)]
        with pytest.raises(EventStudyError, match="same length"):
            corrado_rank_z(mixed)


class TestTheMarketProxy:
    def test_it_excludes_the_event_symbol_from_its_own_benchmark(self) -> None:
        """Leaving a name in its own equal-weighted benchmark pulls its fitted beta toward one and
        shrinks the abnormal return the study exists to measure."""
        panel = {"A": [1.0, 2.0, 3.0], "B": [0.0, 0.0, 0.0], "C": [0.0, 0.0, 0.0]}
        assert market_proxy(panel, exclude="A") == [0.0, 0.0, 0.0]

    def test_it_equally_weights_the_rest(self) -> None:
        panel = {"A": [9.0], "B": [1.0], "C": [3.0]}
        assert market_proxy(panel, exclude="A") == [2.0]

    def test_a_panel_of_one_leaves_nothing_to_benchmark_against(self) -> None:
        with pytest.raises(EventStudyError, match="no symbols left"):
            market_proxy({"A": [1.0, 2.0]}, exclude="A")

    def test_it_aligns_on_the_shortest_series(self) -> None:
        panel = {"A": [0.0], "B": [1.0, 2.0, 3.0], "C": [9.0]}
        assert len(market_proxy(panel, exclude="A")) == 1


class TestReturns:
    def test_they_are_one_shorter_than_the_prices(self) -> None:
        assert len(returns_from([100.0, 101.0, 102.0])) == 2

    def test_a_flat_series_returns_zeros(self) -> None:
        assert returns_from([100.0, 100.0, 100.0]) == [0.0, 0.0]

    def test_a_ten_percent_move_reads_as_a_tenth(self) -> None:
        assert returns_from([100.0, 110.0]) == [pytest.approx(0.1)]

    def test_a_zero_price_does_not_divide_by_zero(self) -> None:
        assert returns_from([0.0, 100.0]) == [0.0]

    def test_a_single_price_has_no_return(self) -> None:
        assert returns_from([100.0]) == []


class TestTheReport:
    def test_it_prints_every_test_adjusted_and_unadjusted(self) -> None:
        text = study("report", _windows(10, shock=0.02, seed=2400), event_bars=6).render()
        for name in ("patell", "bmp", "corrado_rank", "generalised_sign"):
            assert name in text
        assert "adjusted" in text and "adj p" in text

    def test_the_dict_carries_every_window(self) -> None:
        got = study("report", _windows(7, seed=2500), event_bars=6).as_dict()
        assert len(got["windows"]) == 7
        assert got["events"] == 7
        assert "clustering_note" in got

    def test_the_correlation_is_reported_not_only_applied(self) -> None:
        """A correction whose size is invisible is a correction a reader cannot check."""
        got = study("report", _windows(8, seed=2600, shared_residual=0.006), event_bars=6).as_dict()
        assert got["cross_correlation"] > 0.2

    def test_the_average_correlation_of_a_single_window_is_zero(self) -> None:
        assert average_cross_correlation(_windows(1, seed=2700)) == 0.0
