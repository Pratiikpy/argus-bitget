"""Diversity-aware analogue retrieval: MMR, the score breakdown, and the recall/accuracy split.

Three properties decide whether the re-ranking can be trusted. The default must be untouched —
both matchers back OWNED capabilities, so ``diversity=None`` has to return exactly what it did.
The MMR loop must be paper-qa's loop, not a lookalike, so it is pinned against a line-for-line
numpy transcription of `src/paperqa/llms.py:151-166`. And diversity must actually do the thing it
is for: fewer near-duplicates in the retrieved set when the corpus is clustered.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from argus.desk import analogue, shapematch
from argus.desk.analogue import (
    AnalogueError,
    Observation,
    find,
    mmr_order,
    relevance,
)
from argus.eval.retrieval_diversity import (
    Retrieval,
    arm_key,
    diagnostics,
    recall_cap,
)

NOW = datetime(2026, 9, 14, 15, 0, tzinfo=UTC)


def _paperqa_mmr(embeddings: np.ndarray, scores: np.ndarray, k: int, lam: float) -> list[int]:
    """paper-qa's `max_marginal_relevance_search` body, `llms.py:147-166`, with its cosine."""
    norms = np.linalg.norm(embeddings, axis=1)
    similarity_matrix = embeddings @ embeddings.T / np.outer(norms, norms)
    selected_indices = [0]
    remaining_indices = list(range(1, len(embeddings)))
    while len(selected_indices) < k:
        selected_similarities = similarity_matrix[:, selected_indices]
        max_sim_to_selected = selected_similarities.max(axis=1)
        mmr_scores = lam * scores - (1 - lam) * max_sim_to_selected
        mmr_scores[selected_indices] = -np.inf
        max_mmr_index = int(mmr_scores.argmax())
        selected_indices.append(max_mmr_index)
        remaining_indices.remove(max_mmr_index)
    return selected_indices


class TestMMROrderIsPaperQAs:
    @pytest.mark.parametrize("lam", [0.0, 0.3, 0.5, 0.7, 0.9])
    @pytest.mark.parametrize("seed", range(5))
    def test_matches_the_reference_loop(self, lam: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        embeddings = rng.normal(size=(30, 6))
        query = rng.normal(size=6)
        norms = np.linalg.norm(embeddings, axis=1)
        scores = embeddings @ query / (norms * np.linalg.norm(query))
        order = np.argsort(-scores)
        embeddings, scores = embeddings[order], scores[order]
        sims = embeddings @ embeddings.T / np.outer(norms[order], norms[order])
        ours = [i for i, _, _ in mmr_order(
            list(scores), lambda i, j: float(sims[i, j]), k=10, mmr_lambda=lam)]
        assert ours == _paperqa_mmr(embeddings, scores, 10, lam)

    def test_lambda_one_is_relevance_order(self) -> None:
        rels = [0.9, 0.8, 0.7, 0.6]
        picks = mmr_order(rels, lambda i, j: 1.0, k=3, mmr_lambda=1.0)
        assert [i for i, _, _ in picks] == [0, 1, 2]

    def test_rejects_lambda_outside_unit_interval(self) -> None:
        with pytest.raises(AnalogueError):
            mmr_order([1.0, 0.5, 0.2], lambda i, j: 0.0, k=2, mmr_lambda=1.5)

    def test_lambda_zero_picks_the_most_dissimilar_second(self) -> None:
        # candidate 1 is a copy of 0; candidate 2 is unrelated
        sim = [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        picks = mmr_order([1.0, 0.99, 0.5], lambda i, j: sim[i][j], k=2, mmr_lambda=0.0)
        assert [i for i, _, _ in picks] == [0, 2]


def _clustered_corpus() -> list[Observation]:
    """Hourly states in tight episodes of six: the case diversity exists for."""
    rng = random.Random(3)
    out = []
    t = NOW - timedelta(days=400)
    for _ in range(60):
        centre = (rng.gauss(0, 1), rng.gauss(0, 1))
        for h in range(6):
            out.append(Observation(
                as_of=t + timedelta(hours=h), symbol="X",
                features={"a": centre[0] + rng.gauss(0, 0.02), "b": centre[1] + rng.gauss(0, 0.02)},
                forward_return_bps=rng.gauss(0, 50),
            ))
        t += timedelta(days=6)
    return out


class TestAnalogueDiversity:
    def test_default_is_unchanged(self) -> None:
        corpus = _clustered_corpus()
        plain = find(query={"a": 0.1, "b": 0.0}, corpus=corpus, as_of=NOW, limit=20)
        again = find(query={"a": 0.1, "b": 0.0}, corpus=corpus, as_of=NOW, limit=20,
                     diversity=None, explain=False)
        explicit_one = find(query={"a": 0.1, "b": 0.0}, corpus=corpus, as_of=NOW, limit=20,
                            diversity=1.0)
        assert plain.as_dict() == again.as_dict()
        assert [m.observation for m in plain.matches] == [
            m.observation for m in explicit_one.matches]
        assert "diversity_lambda" not in plain.as_dict()
        assert all("why" not in m for m in plain.as_dict()["matches"])
        distances = [m.distance for m in plain.matches]
        assert distances == sorted(distances)

    def test_diversity_raises_independent_episodes(self) -> None:
        corpus = _clustered_corpus()
        plain = find(query={"a": 0.1, "b": 0.0}, corpus=corpus, as_of=NOW, limit=12)
        diverse = find(query={"a": 0.1, "b": 0.0}, corpus=corpus, as_of=NOW, limit=12,
                       diversity=0.3)
        assert plain.distribution is not None and diverse.distribution is not None
        assert diverse.distribution.effective_n > plain.distribution.effective_n
        assert diverse.matches[0].observation == plain.matches[0].observation

    def test_explain_carries_the_whole_breakdown(self) -> None:
        corpus = _clustered_corpus()
        report = find(query={"a": 0.1, "b": 0.0}, corpus=corpus, as_of=NOW, limit=10,
                      diversity=0.5, explain=True)
        rows = report.as_dict()["matches"]
        assert report.as_dict()["diversity_lambda"] == 0.5
        assert [r["why"]["selection_rank"] for r in rows] == list(range(1, 11))
        assert rows[0]["why"]["diversity_penalty"] == 0.0
        assert rows[0]["why"]["pool"] == 20
        for match in report.matches:
            d = match.details
            assert d is not None
            assert d.relevance == pytest.approx(relevance(d.distance, d.bandwidth))
            assert d.final_score == pytest.approx(
                d.mmr_lambda * d.relevance - (1 - d.mmr_lambda) * d.diversity_penalty)
            assert d.distance <= d.max_distance
        assert any("re-ranked for diversity" in line for line in report.render())

    def test_explain_lines_only_when_explained(self) -> None:
        corpus = _clustered_corpus()
        plain = find(query={"a": 0.1, "b": 0.0}, corpus=corpus, as_of=NOW, limit=10)
        assert analogue.explain_lines(plain) == []
        explained = find(query={"a": 0.1, "b": 0.0}, corpus=corpus, as_of=NOW, limit=10,
                         diversity=0.5, explain=True)
        lines = analogue.explain_lines(explained, top=2)
        assert len(lines) == 2
        assert lines[0].startswith("Why ") and "pick 1 of 20" in lines[0]
        assert "MMR score" in lines[1]

    def test_threshold_gates_before_diversity(self) -> None:
        corpus = _clustered_corpus()
        report = find(query={"a": 0.1, "b": 0.0}, corpus=corpus, as_of=NOW, limit=50,
                      max_distance=0.8, diversity=0.0)
        assert all(m.distance <= 0.8 for m in report.matches)

    def test_invalid_lambda_is_refused(self) -> None:
        with pytest.raises(AnalogueError):
            find(query={"a": 0.0}, corpus=_clustered_corpus(), as_of=NOW, diversity=-0.1)

    def test_fetch_pool_is_twice_the_limit(self) -> None:
        assert analogue.FETCH_MULTIPLIER == 2


def _series(n: int = 900, seed: int = 11) -> list[tuple[datetime, float]]:
    rng = random.Random(seed)
    price = 100.0
    out = []
    for i in range(n):
        # a repeating motif plus noise, so several windows share one shape
        price *= 1 + 0.004 * math.sin(i / 3.0) + rng.gauss(0, 0.002)
        out.append((NOW - timedelta(hours=n - i), price))
    return out


class TestShapeDiversity:
    def test_default_is_unchanged(self) -> None:
        series = _series()
        plain = shapematch.find(series, symbol="X", window=24, horizon=12, top=8, trials=0)
        again = shapematch.find(series, symbol="X", window=24, horizon=12, top=8, trials=0,
                                diversity=None)
        one = shapematch.find(series, symbol="X", window=24, horizon=12, top=8, trials=0,
                              diversity=1.0)
        assert plain.as_dict()["analogues"] == again.as_dict()["analogues"]
        assert [a.start_index for a in plain.analogues] == [a.start_index for a in one.analogues]

    def test_exclusion_zone_survives_diversity(self) -> None:
        series = _series()
        report = shapematch.find(series, symbol="X", window=24, horizon=12, top=8, trials=0,
                                 diversity=0.3, explain=True)
        starts = [a.start_index for a in report.analogues]
        assert all(abs(a - b) >= 24 for i, a in enumerate(starts) for b in starts[:i])
        assert report.analogues[0].why is not None
        assert report.analogues[0].why["selection_rank"] == 1.0
        assert "why" in report.as_dict()["analogues"][0]

    def test_null_calibration_is_untouched(self) -> None:
        series = _series()
        plain = shapematch.find(series, symbol="X", window=24, horizon=12, top=5, trials=20)
        diverse = shapematch.find(series, symbol="X", window=24, horizon=12, top=5, trials=20,
                                  diversity=0.5)
        assert (plain.null_better, plain.null_median) == (diverse.null_better, diverse.null_median)
        assert plain.best is not None and diverse.best is not None
        assert plain.best.start_index == diverse.best.start_index

    def test_diversity_lowers_shape_redundancy(self) -> None:
        series = _series()
        closes = [c for _, c in series]

        def redundancy(report: shapematch.AnalogueReport) -> float:
            wins = [closes[a.start_index:a.start_index + 24] for a in report.analogues]
            pairs = [shapematch._correlation(a, b) for i, a in enumerate(wins) for b in wins[:i]]
            return sum(pairs) / len(pairs)

        plain = shapematch.find(series, symbol="X", window=24, horizon=12, top=6, trials=0)
        diverse = shapematch.find(series, symbol="X", window=24, horizon=12, top=6, trials=0,
                                  diversity=0.2)
        assert redundancy(diverse) < redundancy(plain)


class TestRecallAccuracySplit:
    def test_recall_cap_is_the_range_of_the_first_k(self) -> None:
        outcomes = (0.01, -0.02, 0.05, 0.10)
        assert recall_cap(outcomes, 0.0, 2)
        assert not recall_cap(outcomes, 0.04, 2)
        assert recall_cap(outcomes, 0.04, 3)
        assert not recall_cap((), 0.0, 5)

    def test_arm_keys(self) -> None:
        assert arm_key("argusShape", 0.7) == "argusShapeMMR70"

    def test_diagnostics_split_retriever_from_answer(self) -> None:
        from argus.eval.analogstress_comparison import score

        rows = []
        retrieved: dict[tuple[str, int], dict[str, Retrieval | None]] = {}
        rng = random.Random(1)
        for q in range(60):
            y = rng.gauss(0, 0.02)
            era = "calibration" if q < 30 else "test"
            outcomes = tuple(rng.gauss(0, 0.02) for _ in range(50))
            rows.append({"sym": "X", "q": q, "date": f"2024-01-{1 + q % 28:02d}", "era": era,
                         "y": y, "arm": {"centre": 0.0, "hw": 0.02}})
            retrieved[("X", q)] = {"arm": Retrieval(outcomes, 0.1, 0.9)}
        calib = [r for r in rows if r["era"] == "calibration"]
        test = [r for r in rows if r["era"] == "test"]
        scored = score(calib, test, "arm", 0.8)
        out = diagnostics(test, "arm", scored, retrieved)
        assert out["queries"] == 30
        assert set(out["retriever"]["recall_cap"]) == {"@5", "@10", "@20", "@50"}
        assert out["retriever"]["recall_cap"]["@50"] >= out["retriever"]["recall_cap"]["@5"]
        assert out["answer"]["recall_misses"] + round(
            out["retriever"]["recall_cap"]["@50"] * 30) == 30
        assert out["retriever"]["near_duplicate_share"] == pytest.approx(0.1)
