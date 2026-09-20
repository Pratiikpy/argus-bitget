"""Historical analogue retrieval tests.

The decisive property is the look-ahead gate. Selecting past states by what happened next is the
purest form of look-ahead and the easiest to commit by accident, because "find me days like this
that rallied" is a question that answers itself. So the matcher must be structurally unable to see
the outcome it is being asked to predict.

The second property is the refusal. A distribution over four analogues is an anecdote with error
bars, and reporting it as a distribution is how a retrieval system becomes a confidence machine.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from argus.desk.analogue import (
    MIN_ANALOGUES,
    OVERLAP_WINDOW,
    AnalogueError,
    Observation,
    find,
)

NOW = datetime(2026, 9, 14, 15, 0, tzinfo=UTC)


def _corpus(n: int = 40, *, seed: int = 7, spread: float = 1.0) -> list[Observation]:
    rng = random.Random(seed)
    return [
        Observation(
            as_of=NOW - timedelta(days=n - i),
            symbol="NVDAUSDT",
            features={
                "vol": 20 + rng.gauss(0, spread),
                "momentum": rng.gauss(0, spread),
                "spread_bps": 4 + rng.gauss(0, spread * 0.4),
            },
            forward_return_bps=rng.gauss(5, 40),
        )
        for i in range(n)
    ]


class TestTheLookaheadGate:
    """The property that separates retrieval from wishful thinking."""

    def test_observations_at_or_after_the_decision_instant_are_invisible(self) -> None:
        corpus = _corpus()
        future = [
            Observation(
                as_of=NOW + timedelta(days=i),
                symbol="NVDAUSDT",
                features={"vol": 20, "momentum": 0.0, "spread_bps": 4},
                forward_return_bps=9_999,
            )
            for i in range(1, 6)
        ]
        report = find(
            query={"vol": 20, "momentum": 0.0, "spread_bps": 4},
            corpus=[*corpus, *future],
            as_of=NOW,
        )
        assert report.usable
        assert 9_999 not in report.distribution.returns_bps  # type: ignore[union-attr]
        assert all(m.observation.as_of < NOW for m in report.matches)

    def test_the_outcome_is_not_a_matchable_feature(self) -> None:
        """Distance is computed only over `features`; `forward_return_bps` cannot enter it."""
        corpus = _corpus()
        report = find(
            query={"vol": 20, "momentum": 0.0, "spread_bps": 4}, corpus=corpus, as_of=NOW
        )
        for match in report.matches:
            assert "forward_return_bps" not in match.contributions

    def test_a_naive_clock_is_refused(self) -> None:
        with pytest.raises(AnalogueError, match="timezone-aware"):
            find(query={"vol": 20}, corpus=_corpus(), as_of=datetime(2026, 9, 14, 15, 0))

    def test_a_naive_observation_is_refused_at_construction(self) -> None:
        with pytest.raises(AnalogueError, match="timezone-aware"):
            Observation(
                as_of=datetime(2026, 9, 1, 12, 0), symbol="NVDAUSDT",
                features={"vol": 20}, forward_return_bps=10,
            )


class TestTheRefusal:
    def test_too_few_analogues_refuses_rather_than_reporting(self) -> None:
        report = find(
            query={"vol": 20, "momentum": 0.0, "spread_bps": 4},
            corpus=_corpus(6), as_of=NOW,
        )
        assert report.usable is False
        assert "are needed" in report.refused

    def test_a_distant_query_refuses_rather_than_stretching(self) -> None:
        """Nothing resembles the query, so there is no distribution to report."""
        report = find(
            query={"vol": 500.0, "momentum": 90.0, "spread_bps": 300.0},
            corpus=_corpus(), as_of=NOW,
        )
        assert report.usable is False
        assert "anecdote with error bars" in report.refused

    def test_an_empty_corpus_refuses_with_the_count(self) -> None:
        report = find(query={"vol": 20}, corpus=[], as_of=NOW)
        assert report.usable is False

    def test_a_query_feature_absent_from_the_corpus_refuses(self) -> None:
        report = find(query={"unheard_of": 1.0}, corpus=_corpus(), as_of=NOW)
        assert report.usable is False
        assert "no query feature" in report.refused

    def test_the_refusal_renders_as_a_line_the_desk_can_record(self) -> None:
        report = find(query={"vol": 20}, corpus=_corpus(3), as_of=NOW)
        assert report.render()[0].startswith("[analogue] no distribution:")


class TestItReportsADistributionNotANumber:
    def _report(self):  # type: ignore[no-untyped-def]
        return find(
            query={"vol": 20, "momentum": 0.0, "spread_bps": 4},
            corpus=_corpus(60), as_of=NOW, max_distance=5.0,
        )

    def test_the_whole_shape_comes_back(self) -> None:
        payload = self._report().as_dict()["distribution"]
        for key in ("median_bps", "mean_bps", "p25_bps", "p75_bps", "worst_bps",
                    "best_bps", "iqr_bps", "hit_rate"):
            assert key in payload

    def test_the_worst_case_is_reported_because_that_is_what_a_position_survives(self) -> None:
        report = self._report()
        assert report.usable
        worst = min(report.distribution.returns_bps)  # type: ignore[union-attr]
        assert report.as_dict()["distribution"]["worst_bps"] == pytest.approx(round(worst, 2))

    def test_the_rendered_lines_lead_with_the_spread_not_the_average(self) -> None:
        text = " ".join(self._report().render())
        assert "IQR" in text and "spread, not the average" in text

    def test_the_hit_rate_is_separate_from_the_magnitude(self) -> None:
        """Direction and size are different questions; one number cannot answer both."""
        report = self._report()
        assert report.usable
        assert 0.0 <= report.distribution.hit_rate <= 1.0  # type: ignore[union-attr]


class TestClusteredAnaloguesAreNotIndependent:
    def test_overlapping_matches_collapse_into_one_episode(self) -> None:
        """Twenty analogues from three clustered episodes are not twenty observations."""
        base = NOW - timedelta(days=200)
        clustered = [
            Observation(
                as_of=base + timedelta(hours=6 * i),
                symbol="NVDAUSDT",
                features={"vol": 20.0, "momentum": 0.0, "spread_bps": 4.0},
                forward_return_bps=30.0,
            )
            for i in range(12)
        ]
        report = find(
            query={"vol": 20.0, "momentum": 0.0, "spread_bps": 4.0},
            corpus=clustered, as_of=NOW, max_distance=5.0,
        )
        assert report.usable
        assert report.distribution.effective_n < report.distribution.count  # type: ignore[union-attr]

    def test_the_collapse_is_reported_rather_than_applied_silently(self) -> None:
        base = NOW - timedelta(days=200)
        clustered = [
            Observation(
                as_of=base + timedelta(hours=6 * i), symbol="NVDAUSDT",
                features={"vol": 20.0, "momentum": 0.0, "spread_bps": 4.0},
                forward_return_bps=30.0,
            )
            for i in range(12)
        ]
        report = find(
            query={"vol": 20.0, "momentum": 0.0, "spread_bps": 4.0},
            corpus=clustered, as_of=NOW, max_distance=5.0,
        )
        assert any("collapsed" in line for line in report.render())

    def test_well_separated_matches_stay_independent(self) -> None:
        spaced = [
            Observation(
                as_of=NOW - timedelta(days=10 * (i + 1)), symbol="NVDAUSDT",
                features={"vol": 20.0, "momentum": 0.0, "spread_bps": 4.0},
                forward_return_bps=float(i),
            )
            for i in range(12)
        ]
        report = find(
            query={"vol": 20.0, "momentum": 0.0, "spread_bps": 4.0},
            corpus=spaced, as_of=NOW, max_distance=5.0,
        )
        assert report.usable
        assert report.distribution.effective_n == report.distribution.count  # type: ignore[union-attr]

    def test_the_same_instant_on_two_symbols_is_two_observations(self) -> None:
        """Overlap is about one episode, not one moment; different names are different bets."""
        at = NOW - timedelta(days=50)
        rows = [
            Observation(at, sym, {"vol": 20.0, "momentum": 0.0, "spread_bps": 4.0}, 10.0)
            for sym in ("NVDAUSDT", "TSLAUSDT", "AAPLUSDT", "MSFTUSDT",
                        "METAUSDT", "AMZNUSDT", "COINUSDT", "MSTRUSDT")
        ]
        report = find(
            query={"vol": 20.0, "momentum": 0.0, "spread_bps": 4.0},
            corpus=rows, as_of=NOW, max_distance=5.0,
        )
        assert report.usable
        assert report.distribution.effective_n == 8  # type: ignore[union-attr]


class TestSimilarityIsInspectable:
    def test_every_match_says_which_feature_drove_it(self) -> None:
        report = find(
            query={"vol": 20, "momentum": 0.0, "spread_bps": 4},
            corpus=_corpus(60), as_of=NOW, max_distance=5.0,
        )
        for match in report.matches:
            assert match.dominant_feature in {"vol", "momentum", "spread_bps"}
            assert sum(match.contributions.values()) == pytest.approx(1.0, abs=1e-6)

    def test_matches_come_back_nearest_first(self) -> None:
        report = find(
            query={"vol": 20, "momentum": 0.0, "spread_bps": 4},
            corpus=_corpus(60), as_of=NOW, max_distance=5.0,
        )
        distances = [m.distance for m in report.matches]
        assert distances == sorted(distances)

    def test_a_feature_with_a_wide_range_does_not_dominate_by_unit_choice(self) -> None:
        """Standardisation is what stops "spread in bps" outvoting "momentum in units"."""
        rng = random.Random(3)
        corpus = [
            Observation(
                as_of=NOW - timedelta(days=i + 1), symbol="NVDAUSDT",
                features={"tiny": rng.gauss(0, 0.001), "huge": rng.gauss(0, 1000.0)},
                forward_return_bps=rng.gauss(0, 20),
            )
            for i in range(40)
        ]
        report = find(query={"tiny": 0.0, "huge": 0.0}, corpus=corpus, as_of=NOW,
                      max_distance=5.0)
        assert report.usable
        dominant = [m.dominant_feature for m in report.matches]
        assert "tiny" in dominant, "a small-scale feature must still be able to drive a match"

    def test_weights_shift_which_feature_matters(self) -> None:
        corpus = _corpus(60)
        query = {"vol": 20, "momentum": 0.0, "spread_bps": 4}
        unweighted = find(query=query, corpus=corpus, as_of=NOW, max_distance=5.0)
        weighted = find(
            query=query, corpus=corpus, as_of=NOW, max_distance=5.0,
            weights={"vol": 10.0, "momentum": 0.1, "spread_bps": 0.1},
        )
        assert [m.observation.as_of for m in unweighted.matches] != [
            m.observation.as_of for m in weighted.matches
        ]


class TestTheWindowConstant:
    def test_overlap_window_is_stated_not_inlined(self) -> None:
        assert timedelta(0) < OVERLAP_WINDOW

    def test_the_floor_is_stated_not_inlined(self) -> None:
        assert MIN_ANALOGUES >= 5
