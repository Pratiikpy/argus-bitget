"""Diversification tests — anchored on books whose answer is known before the code runs.

The measure this replaces was an inverse Herfindahl that scored three perfectly correlated
positions at 2.71, and `desk/portfolio.py` says so in its own docstring. So the tests that matter
most are the degenerate ones: N independent equal-weight positions must score exactly N, and N
perfectly correlated ones must score exactly 1. A measure that gets those wrong is not
correlation-aware whatever it reports in between.

The eigensolver is checked against `numpy.linalg.eigh` where numpy is available, and against
analytic eigenvalues where it is not, so "pure Python" is a statement about dependencies rather
than about accuracy.
"""

from __future__ import annotations

import math
import random

import pytest

from argus.desk.diversification import (
    MIN_EIGENVALUE,
    Diversification,
    DiversificationError,
    Hedge,
    effective_bets,
    jacobi_eigen,
    minimum_variance_hedge,
    suggest_hedges,
)

NAMES = ["A", "B", "C", "D"]
EQUAL = dict.fromkeys(NAMES, 0.25)


def _identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _all_ones(n: int) -> list[list[float]]:
    return [[1.0] * n for _ in range(n)]


def _block(n: int, size: int) -> list[list[float]]:
    """Perfectly correlated blocks of ``size``, independent across blocks."""
    return [[1.0 if (i // size) == (j // size) else 0.0 for j in range(n)] for i in range(n)]


class TestTheEigensolver:
    def test_the_identity_has_unit_eigenvalues(self) -> None:
        values, _ = jacobi_eigen(_identity(5))
        assert values == pytest.approx([1.0] * 5)

    def test_a_diagonal_matrix_returns_its_diagonal_sorted(self) -> None:
        matrix = [[3.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 1.0]]
        values, _ = jacobi_eigen(matrix)
        assert values == pytest.approx([7.0, 3.0, 1.0])

    def test_a_known_two_by_two(self) -> None:
        """Eigenvalues of [[2,1],[1,2]] are 3 and 1, by hand."""
        values, _ = jacobi_eigen([[2.0, 1.0], [1.0, 2.0]])
        assert values == pytest.approx([3.0, 1.0])

    def test_the_all_ones_matrix_has_one_nonzero_eigenvalue(self) -> None:
        """Rank one: a single direction carries everything. This is the case the whole module is
        about — it is what perfectly correlated positions look like."""
        values, _ = jacobi_eigen(_all_ones(4))
        assert values[0] == pytest.approx(4.0)
        assert all(abs(v) < 1e-9 for v in values[1:])

    def test_eigenvalues_come_back_descending(self) -> None:
        rng = random.Random(1)
        raw = [[rng.gauss(0, 1) for _ in range(6)] for _ in range(6)]
        symmetric = [[(raw[i][j] + raw[j][i]) / 2 for j in range(6)] for i in range(6)]
        values, _ = jacobi_eigen(symmetric)
        assert values == sorted(values, reverse=True)

    def test_the_trace_is_preserved(self) -> None:
        """The sum of eigenvalues equals the trace, for any symmetric matrix. A rotation that
        loses mass fails this and nothing else would catch it."""
        rng = random.Random(2)
        raw = [[rng.gauss(0, 1) for _ in range(7)] for _ in range(7)]
        symmetric = [[(raw[i][j] + raw[j][i]) / 2 for j in range(7)] for i in range(7)]
        values, _ = jacobi_eigen(symmetric)
        assert sum(values) == pytest.approx(sum(symmetric[i][i] for i in range(7)))

    def test_the_eigenvectors_reconstruct_the_matrix(self) -> None:
        """``sum_k lambda_k v_k v_k'`` must rebuild the input. This is the check that catches a
        transposed eigenvector array, which leaves the eigenvalues correct and everything built on
        the vectors wrong."""
        rng = random.Random(3)
        n = 5
        raw = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(n)]
        symmetric = [[(raw[i][j] + raw[j][i]) / 2 for j in range(n)] for i in range(n)]
        values, vectors = jacobi_eigen(symmetric)
        for i in range(n):
            for j in range(n):
                rebuilt = sum(
                    values[k] * vectors[k][i] * vectors[k][j] for k in range(n)
                )
                assert rebuilt == pytest.approx(symmetric[i][j], abs=1e-9)

    def test_the_eigenvectors_are_orthonormal(self) -> None:
        rng = random.Random(4)
        n = 5
        raw = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(n)]
        symmetric = [[(raw[i][j] + raw[j][i]) / 2 for j in range(n)] for i in range(n)]
        _, vectors = jacobi_eigen(symmetric)
        for i in range(n):
            for j in range(n):
                dot = sum(vectors[i][k] * vectors[j][k] for k in range(n))
                assert dot == pytest.approx(1.0 if i == j else 0.0, abs=1e-9)

    def test_an_asymmetric_matrix_raises_rather_than_being_symmetrised(self) -> None:
        """An asymmetric covariance matrix is a transposition bug upstream. Averaging the halves
        here would hide it behind a plausible answer."""
        with pytest.raises(DiversificationError, match="not symmetric"):
            jacobi_eigen([[1.0, 2.0], [3.0, 1.0]])

    def test_a_ragged_matrix_raises(self) -> None:
        with pytest.raises(DiversificationError, match="square"):
            jacobi_eigen([[1.0, 0.0], [0.0]])

    def test_an_empty_matrix_raises(self) -> None:
        with pytest.raises(DiversificationError, match="no eigenvalues"):
            jacobi_eigen([])

    def test_it_matches_numpy_where_numpy_is_available(self) -> None:
        """Pure Python is a dependency choice, not an accuracy compromise."""
        numpy = pytest.importorskip("numpy")
        rng = random.Random(5)
        for n in (2, 3, 5, 8):
            raw = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(n)]
            symmetric = [[(raw[i][j] + raw[j][i]) / 2 for j in range(n)] for i in range(n)]
            ours, _ = jacobi_eigen(symmetric)
            reference = sorted(numpy.linalg.eigvalsh(numpy.array(symmetric)).tolist(), reverse=True)
            assert ours == pytest.approx(reference, abs=1e-10)


class TestEffectiveBetsOnKnownBooks:
    def test_independent_equal_positions_score_the_position_count(self) -> None:
        for n in (2, 3, 4, 6):
            names = [f"S{i}" for i in range(n)]
            weights = dict.fromkeys(names, 1.0 / n)
            got = effective_bets(weights, names, _identity(n))
            assert got.effective_bets == pytest.approx(float(n)), n

    def test_perfectly_correlated_positions_score_one(self) -> None:
        """The case the inverse Herfindahl gets wrong: it scores 2.71 for three of these."""
        for n in (2, 3, 4, 6):
            names = [f"S{i}" for i in range(n)]
            weights = dict.fromkeys(names, 1.0 / n)
            got = effective_bets(weights, names, _all_ones(n))
            assert got.effective_bets == pytest.approx(1.0, abs=1e-6), n

    def test_two_correlated_pairs_score_two(self) -> None:
        got = effective_bets(EQUAL, NAMES, _block(4, 2))
        assert got.effective_bets == pytest.approx(2.0, abs=1e-6)

    def test_more_correlation_means_fewer_bets_on_an_unequal_book(self) -> None:
        """Measured on unequal weights, because the equally-weighted equicorrelated book is the
        knife-edge case where both rotations degenerate — see `TestTheTwoRotationsDisagree`."""
        unequal = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
        scores = []
        for rho in (0.0, 0.3, 0.6, 0.9):
            matrix = [[1.0 if i == j else rho for j in range(4)] for i in range(4)]
            scores.append(effective_bets(unequal, NAMES, matrix).effective_bets)
        assert scores == sorted(scores, reverse=True)
        assert scores[0] > 2.5 and scores[-1] < 1.1

    def test_an_unequal_book_lands_strictly_inside_the_range(self) -> None:
        unequal = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
        partial = [[1.0 if i == j else 0.5 for j in range(4)] for i in range(4)]
        got = effective_bets(unequal, NAMES, partial)
        assert 1.0 < got.effective_bets < 4.0

    def test_the_distribution_sums_to_one(self) -> None:
        got = effective_bets(EQUAL, NAMES, _block(4, 2))
        assert sum(got.distribution) == pytest.approx(1.0)

    def test_the_distribution_is_returned_not_only_the_summary(self) -> None:
        """A single number hides the shape. One direction at 90% and one at 10% is a different
        book from two at 50%, and both can produce a similar entropy."""
        got = effective_bets(EQUAL, NAMES, _all_ones(4))
        assert got.largest_share == pytest.approx(1.0)


class TestWhatItRefusesToMeasure:
    def test_one_position_is_not_a_diversification_question(self) -> None:
        with pytest.raises(DiversificationError, match="at least two held"):
            effective_bets({"A": 1.0}, NAMES, _identity(4))

    def test_an_empty_book_raises(self) -> None:
        with pytest.raises(DiversificationError, match="at least two held"):
            effective_bets({}, NAMES, _identity(4))

    def test_a_riskless_book_raises_rather_than_scoring_infinite_diversity(self) -> None:
        with pytest.raises(DiversificationError, match="no variance"):
            effective_bets(EQUAL, NAMES, [[0.0] * 4 for _ in range(4)])

    def test_a_mismatched_matrix_raises(self) -> None:
        with pytest.raises(DiversificationError, match="must match"):
            effective_bets(EQUAL, NAMES, _identity(3))

    def test_unheld_names_are_excluded_from_the_count(self) -> None:
        """A zero-weight name is not a position and must not dilute the measure."""
        weights = {"A": 0.5, "B": 0.5, "C": 0.0, "D": 0.0}
        got = effective_bets(weights, NAMES, _identity(4))
        assert got.positions == 2
        assert got.effective_bets == pytest.approx(2.0)

    def test_a_near_zero_eigenvalue_does_not_inflate_the_score(self) -> None:
        """The trap this guards: log of a tiny share is a large negative number that dominates the
        entropy, so a degenerate book would read as maximally diversified."""
        got = effective_bets(EQUAL, NAMES, _all_ones(4))
        assert got.effective_bets < 1.001
        assert MIN_EIGENVALUE > 0


class TestTheHedge:
    def test_a_perfect_inverse_is_sized_at_one_and_removes_everything(self) -> None:
        rng = random.Random(6)
        book = [rng.gauss(0, 1) for _ in range(400)]
        mirror = [-b for b in book]
        got = minimum_variance_hedge(book, mirror, instrument="INV")
        assert got.ratio == pytest.approx(1.0)
        assert got.variance_reduction == pytest.approx(1.0)

    def test_a_perfect_copy_is_sized_short(self) -> None:
        """Hedging a book with the same thing means shorting it. A positive ratio here would be
        doubling the position while calling it a hedge."""
        rng = random.Random(7)
        book = [rng.gauss(0, 1) for _ in range(400)]
        got = minimum_variance_hedge(book, list(book), instrument="SAME")
        assert got.ratio == pytest.approx(-1.0)

    def test_an_uncorrelated_instrument_removes_almost_nothing(self) -> None:
        rng = random.Random(8)
        book = [rng.gauss(0, 1) for _ in range(2000)]
        noise = [rng.gauss(0, 1) for _ in range(2000)]
        got = minimum_variance_hedge(book, noise, instrument="NOISE")
        assert got.variance_reduction < 0.02
        assert not got.useful

    def test_the_reduction_is_the_squared_correlation(self) -> None:
        """Both come from the same covariance, so the promised reduction cannot disagree with the
        recommendation. Recomputed here independently."""
        rng = random.Random(9)
        book = [rng.gauss(0, 1) for _ in range(500)]
        partial = [0.6 * b + rng.gauss(0, 0.8) for b in book]
        got = minimum_variance_hedge(book, partial, instrument="PART")
        assert got.variance_reduction == pytest.approx(got.correlation**2)

    def test_a_half_scale_hedge_is_sized_at_two(self) -> None:
        """If the hedge moves half as much, twice as much of it is needed."""
        rng = random.Random(10)
        book = [rng.gauss(0, 1) for _ in range(500)]
        half = [-0.5 * b for b in book]
        got = minimum_variance_hedge(book, half, instrument="HALF")
        assert got.ratio == pytest.approx(2.0)

    def test_a_motionless_instrument_raises_rather_than_sizing_infinitely(self) -> None:
        with pytest.raises(DiversificationError, match="does not move"):
            minimum_variance_hedge([1.0, -1.0, 2.0], [0.0, 0.0, 0.0], instrument="FLAT")

    def test_a_riskless_book_has_nothing_to_hedge(self) -> None:
        with pytest.raises(DiversificationError, match="nothing to hedge"):
            minimum_variance_hedge([0.0, 0.0, 0.0], [1.0, -1.0, 2.0], instrument="X")

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(DiversificationError, match="same periods"):
            minimum_variance_hedge([1.0, 2.0, 3.0], [1.0, 2.0], instrument="X")

    def test_the_verdict_names_the_side_to_take(self) -> None:
        rng = random.Random(11)
        book = [rng.gauss(0, 1) for _ in range(300)]
        got = minimum_variance_hedge(book, list(book), instrument="SAME")
        assert "short" in got.verdict

    def test_a_useless_hedge_says_it_does_not_pay_for_a_round_trip(self) -> None:
        rng = random.Random(12)
        book = [rng.gauss(0, 1) for _ in range(2000)]
        noise = [rng.gauss(0, 1) for _ in range(2000)]
        assert "does not pay for a round trip" in minimum_variance_hedge(
            book, noise, instrument="NOISE"
        ).verdict


class TestSuggestions:
    def test_candidates_are_ranked_by_what_they_remove(self) -> None:
        rng = random.Random(13)
        book = [rng.gauss(0, 1) for _ in range(500)]
        got = suggest_hedges(book, {
            "strong": [-0.95 * b + rng.gauss(0, 0.1) for b in book],
            "weak": [-0.2 * b + rng.gauss(0, 1.0) for b in book],
            "none": [rng.gauss(0, 1) for _ in range(500)],
        })
        assert [h.instrument for h in got] == ["strong", "weak", "none"]

    def test_unusable_candidates_are_kept_and_marked(self) -> None:
        """Dropping them would leave a reader unable to tell "we checked and none works" from "we
        only checked the ones that do"."""
        rng = random.Random(14)
        book = [rng.gauss(0, 1) for _ in range(2000)]
        got = suggest_hedges(book, {"none": [rng.gauss(0, 1) for _ in range(2000)]})
        assert len(got) == 1
        assert not got[0].useful

    def test_a_candidate_that_cannot_be_sized_is_skipped_not_crashed(self) -> None:
        rng = random.Random(15)
        book = [rng.gauss(0, 1) for _ in range(300)]
        got = suggest_hedges(book, {"flat": [0.0] * 300, "real": [-b for b in book]})
        assert [h.instrument for h in got] == ["real"]

    def test_the_limit_is_respected(self) -> None:
        rng = random.Random(16)
        book = [rng.gauss(0, 1) for _ in range(300)]
        candidates = {f"c{i}": [rng.gauss(0, 1) for _ in range(300)] for i in range(8)}
        assert len(suggest_hedges(book, candidates, limit=3)) == 3


class TestTheReport:
    def test_a_concentrated_book_is_named_as_one_trade_in_many_names(self) -> None:
        got = effective_bets(EQUAL, NAMES, _all_ones(4))
        assert "one trade wearing 4 names" in got.verdict

    def test_a_diversified_book_is_named_as_close_to_independent(self) -> None:
        got = effective_bets(EQUAL, NAMES, _identity(4))
        assert "close to independent" in got.verdict

    def test_the_dict_carries_the_distribution_and_the_ratio(self) -> None:
        got = effective_bets(EQUAL, NAMES, _block(4, 2)).as_dict()
        assert len(got["distribution"]) == 4
        assert got["concentration_ratio"] == pytest.approx(0.5, abs=1e-6)

    def test_the_concentration_ratio_is_bets_over_positions(self) -> None:
        got = Diversification(
            effective_bets=2.0, torsion_bets=2.0, distribution=(0.5, 0.5),
            eigenvalues=(1.0, 1.0), positions=4,
        )
        assert got.concentration_ratio == pytest.approx(0.5)

    def test_a_hedge_serialises_its_size_and_its_promise(self) -> None:
        got = Hedge(instrument="X", ratio=-0.5, variance_reduction=0.4, correlation=0.632).as_dict()
        assert got["ratio"] == -0.5
        assert got["variance_reduction"] == 0.4
        assert got["useful"]

    def test_the_entropy_identity_holds(self) -> None:
        """``effective_bets == exp(entropy(distribution))``, recomputed from the returned shape."""
        got = effective_bets(EQUAL, NAMES, _block(4, 2))
        entropy = -sum(p * math.log(p) for p in got.distribution if p > 0)
        assert got.effective_bets == pytest.approx(math.exp(entropy))


class TestTheTwoRotationsDisagree:
    """Both were implemented because each is blind exactly where the other is informative.

    Measured, not assumed. On an equally-weighted equicorrelated book the principal-component
    reading collapses to 1.00 at a correlation of 0.0001 and stays there; the minimum-torsion
    reading stays at 4.00 all the way to 0.99. Shipping either alone would have been a plausible
    number that cannot see the thing it claims to measure.
    """

    def test_the_pca_reading_collapses_on_an_equally_weighted_correlated_book(self) -> None:
        for rho in (0.0001, 0.05, 0.3, 0.6, 0.9):
            got = effective_bets(
                EQUAL, NAMES, [[1.0 if i == j else rho for j in range(4)] for i in range(4)]
            )
            assert got.effective_bets == pytest.approx(1.0, abs=1e-6), rho

    def test_the_torsion_reading_is_insensitive_on_the_same_book(self) -> None:
        for rho in (0.0001, 0.05, 0.3, 0.6, 0.9):
            got = effective_bets(
                EQUAL, NAMES, [[1.0 if i == j else rho for j in range(4)] for i in range(4)]
            )
            assert got.torsion_bets == pytest.approx(4.0, abs=1e-6), rho

    def test_they_agree_exactly_when_there_is_no_correlation(self) -> None:
        """With nothing to rotate, the two rotations are the same rotation. If they ever disagreed
        here, one of them is not a rotation of the book it was given."""
        got = effective_bets(EQUAL, NAMES, _identity(4))
        assert got.torsion_bets == pytest.approx(got.effective_bets)

    def test_they_agree_at_zero_correlation_on_an_unequal_book_too(self) -> None:
        unequal = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
        got = effective_bets(unequal, NAMES, _identity(4))
        assert got.torsion_bets == pytest.approx(got.effective_bets)

    def test_the_torsion_is_undefined_under_perfect_correlation(self) -> None:
        """No invertible rotation exists. The PCA reading still works, so the result carries one
        number and says which — rather than reporting agreement between one measure and nothing."""
        got = effective_bets(EQUAL, NAMES, _all_ones(4))
        assert got.torsion_bets is None
        assert got.disagreement is None
        assert got.effective_bets == pytest.approx(1.0, abs=1e-6)

    def test_the_verdict_prints_both_and_says_which_to_plan_against(self) -> None:
        unequal = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
        got = effective_bets(
            unequal, NAMES, [[1.0 if i == j else 0.6 for j in range(4)] for i in range(4)]
        )
        assert "Minimum torsion reads" in got.verdict
        assert "the lower number is the one to plan against" in got.verdict

    def test_the_verdict_names_the_singular_case_rather_than_omitting_it(self) -> None:
        got = effective_bets(EQUAL, NAMES, _all_ones(4))
        assert "perfectly correlated" in got.verdict

    def test_the_disagreement_grows_with_correlation(self) -> None:
        """The gap between the two readings is itself the signal that the book is correlated."""
        unequal = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
        gaps = []
        for rho in (0.05, 0.3, 0.6, 0.9):
            got = effective_bets(
                unequal, NAMES, [[1.0 if i == j else rho for j in range(4)] for i in range(4)]
            )
            assert got.disagreement is not None
            gaps.append(got.disagreement)
        assert gaps == sorted(gaps)

    def test_the_dict_carries_both_readings(self) -> None:
        unequal = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
        got = effective_bets(unequal, NAMES, _identity(4)).as_dict()
        assert got["torsion_bets"] is not None
        assert got["disagreement"] == pytest.approx(0.0, abs=1e-6)


class TestTheTorsionItself:
    def test_it_is_the_identity_when_there_is_nothing_to_decorrelate(self) -> None:
        from argus.desk.diversification import minimum_torsion

        torsion = minimum_torsion(_identity(4))
        for i in range(4):
            for j in range(4):
                assert torsion[i][j] == pytest.approx(1.0 if i == j else 0.0, abs=1e-9)

    def test_its_factors_are_uncorrelated(self) -> None:
        """The defining property: ``t Sigma t'`` must be diagonal. If it is not, the rotation did
        not decorrelate anything and every share built on it is wrong."""
        from argus.desk.diversification import minimum_torsion

        sigma = [[1.0 if i == j else 0.4 for j in range(4)] for i in range(4)]
        t = minimum_torsion(sigma)
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                value = sum(
                    t[i][a] * sigma[a][b] * t[j][b] for a in range(4) for b in range(4)
                )
                assert abs(value) < 1e-9, (i, j, value)

    def test_a_singular_correlation_matrix_raises(self) -> None:
        from argus.desk.diversification import DiversificationError, minimum_torsion

        with pytest.raises(DiversificationError, match="singular"):
            minimum_torsion(_all_ones(4))

    def test_a_riskless_asset_cannot_be_rotated(self) -> None:
        from argus.desk.diversification import DiversificationError, minimum_torsion

        sigma = [[1.0, 0.0], [0.0, 0.0]]
        with pytest.raises(DiversificationError, match="zero variance"):
            minimum_torsion(sigma)
