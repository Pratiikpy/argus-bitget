"""Tests for the real-cluster sentiment-integrity measurement (capability 20, route (c).2).

Entirely offline: a synthetic ledger (no read of the real T2 repository), a stubbed finBERT
classifier and a stubbed Qwen client, and a stubbed post fetcher. Nothing here spends a token or
opens a socket — ``fetch_x_post``/``fetch_reddit_post``/the real ``fetch_post`` are never called.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from argus.agents.analysts import AnalystView
from argus.eval import sentiment_integrity_real as sir
from argus.eval.sentiment_comparison import FinbertNaiveAggregate
from argus.llm.qwen import Completion


def _snapshot(cluster: dict[str, Any]) -> str:
    payload = {"snapshot": {"crowd": {"clusters": [cluster]}}}
    return json.dumps({"kind": "snapshot", "payload": payload})


def _view(*, signal: str, confidence: float, magnitude: int = 50) -> AnalystView:
    return AnalystView(
        analyst="sentiment", signal=signal, magnitude_bps=magnitude, confidence=confidence,
        reasoning="r", counter_case="c",
    )


def _always(label: str, score: float = 0.9) -> Any:
    def classify(texts: list[str]) -> list[dict[str, Any]]:
        return [{"label": label, "score": score} for _ in texts]
    return classify


def _by_text(mapping: dict[str, str], default: str = "neutral") -> Any:
    def classify(texts: list[str]) -> list[dict[str, Any]]:
        return [{"label": mapping.get(t, default), "score": 0.9} for t in texts]
    return classify


class TestAggregateClusters:
    def test_two_snapshots_of_the_same_story_merge(self) -> None:
        lines = [
            _snapshot({"cluster_id": "s1", "coordinated": False, "item_ids": ["x:1", "x:2"],
                      "sources": ["@a", "@b"], "symbols": ["NVDAUSDT"],
                      "first_seen": "2026-09-25T00:00:00Z", "last_seen": "2026-09-25T00:05:00Z",
                      "representative": "short"}),
            _snapshot({"cluster_id": "s1", "coordinated": True, "item_ids": ["x:2", "x:3"],
                      "sources": ["@b", "@c"], "symbols": ["NVDAUSDT", "AAPLUSDT"],
                      "first_seen": "2026-09-25T00:00:00Z", "last_seen": "2026-09-25T00:10:00Z",
                      "representative": "a longer representative text"}),
        ]
        out = sir.aggregate_clusters(lines)
        rec = out["s1"]
        assert rec.item_ids == ("x:1", "x:2", "x:3")
        assert rec.sources == ("@a", "@b", "@c")
        assert rec.symbols == ("AAPLUSDT", "NVDAUSDT")
        assert rec.coordinated is True  # flagged in either snapshot -> True forever
        assert rec.last_seen == "2026-09-25T00:10:00Z"
        assert rec.representative == "a longer representative text"
        assert rec.day == "2026-09-25"

    def test_non_snapshot_events_are_ignored(self) -> None:
        lines = ['{"kind": "note", "payload": {}}', "", "  "]
        assert sir.aggregate_clusters(lines) == {}

    def test_two_distinct_cluster_ids_stay_separate(self) -> None:
        lines = [
            _snapshot({"cluster_id": "a", "item_ids": ["x:1"], "sources": ["@a"],
                      "symbols": ["NVDAUSDT"], "coordinated": False,
                      "first_seen": "2026-09-25T00:00:00Z", "last_seen": "2026-09-25T00:00:00Z",
                      "representative": "x"}),
            _snapshot({"cluster_id": "b", "item_ids": ["x:2"], "sources": ["@b"],
                      "symbols": ["AAPLUSDT"], "coordinated": False,
                      "first_seen": "2026-09-25T00:00:00Z", "last_seen": "2026-09-25T00:00:00Z",
                      "representative": "y"}),
        ]
        assert set(sir.aggregate_clusters(lines)) == {"a", "b"}


def _rec(cid: str, *, coordinated: bool, day: str, symbols: tuple[str, ...],
        item_ids: tuple[str, ...]) -> sir.ClusterRecord:
    return sir.ClusterRecord(
        cluster_id=cid, item_ids=item_ids, sources=tuple(f"@s{i}" for i in range(len(item_ids))),
        symbols=symbols, coordinated=coordinated, first_seen=f"{day}T00:00:00Z",
        last_seen=f"{day}T01:00:00Z", representative="rep",
    )


class TestMatchUncoordinated:
    def test_prefers_same_day_and_shared_symbol(self) -> None:
        coord = [_rec("c1", coordinated=True, day="2026-09-25", symbols=("NVDAUSDT",),
                      item_ids=("x:1", "x:2", "x:3"))]
        pool = [
            _rec("u_wrong_day", coordinated=False, day="2026-09-26", symbols=("NVDAUSDT",),
                item_ids=("x:10", "x:11")),
            _rec("u_wrong_symbol", coordinated=False, day="2026-09-25", symbols=("AAPLUSDT",),
                item_ids=("x:20", "x:21")),
            _rec("u_match", coordinated=False, day="2026-09-25", symbols=("NVDAUSDT",),
                item_ids=("x:30", "x:31")),
        ]
        pairs = sir.match_uncoordinated(coord, pool)
        assert len(pairs) == 1
        assert pairs[0].control is not None
        assert pairs[0].control.cluster_id == "u_match"
        assert pairs[0].match_basis == "day+symbol"

    def test_falls_back_to_day_only_then_symbol_only(self) -> None:
        coord = [_rec("c1", coordinated=True, day="2026-09-25", symbols=("NVDAUSDT",),
                      item_ids=("x:1", "x:2"))]
        pool = [_rec("u1", coordinated=False, day="2026-09-25", symbols=("AAPLUSDT",),
                    item_ids=("x:10", "x:11"))]
        pairs = sir.match_uncoordinated(coord, pool)
        assert pairs[0].control is not None and pairs[0].control.cluster_id == "u1"
        assert pairs[0].match_basis == "day_only"

        pool2 = [_rec("u2", coordinated=False, day="2026-09-27", symbols=("NVDAUSDT",),
                     item_ids=("x:20", "x:21"))]
        pairs2 = sir.match_uncoordinated(coord, pool2)
        assert pairs2[0].control is not None and pairs2[0].control.cluster_id == "u2"
        assert pairs2[0].match_basis == "symbol_only"

    def test_no_candidate_at_all_records_no_match(self) -> None:
        coord = [_rec("c1", coordinated=True, day="2026-09-25", symbols=("NVDAUSDT",),
                      item_ids=("x:1", "x:2"))]
        assert sir.match_uncoordinated(coord, [])[0].control is None
        assert sir.match_uncoordinated(coord, [])[0].match_basis == "no_match"

    def test_controls_are_not_reused_and_prefers_larger(self) -> None:
        coord = [
            _rec("c1", coordinated=True, day="2026-09-25", symbols=("NVDAUSDT",),
                item_ids=("x:1", "x:2")),
            _rec("c2", coordinated=True, day="2026-09-25", symbols=("NVDAUSDT",),
                item_ids=("x:3", "x:4")),
        ]
        pool = [
            _rec("u_big", coordinated=False, day="2026-09-25", symbols=("NVDAUSDT",),
                item_ids=("x:10", "x:11", "x:12")),
            _rec("u_small", coordinated=False, day="2026-09-25", symbols=("NVDAUSDT",),
                item_ids=("x:20", "x:21")),
        ]
        pairs = sir.match_uncoordinated(coord, pool)
        chosen = {p.coordinated.cluster_id: p.control.cluster_id for p in pairs if p.control}
        assert chosen == {"c1": "u_big", "c2": "u_small"}

    def test_a_control_needs_at_least_min_cluster_items(self) -> None:
        coord = [_rec("c1", coordinated=True, day="2026-09-25", symbols=("NVDAUSDT",),
                      item_ids=("x:1", "x:2"))]
        pool = [_rec("u1", coordinated=False, day="2026-09-25", symbols=("NVDAUSDT",),
                    item_ids=("x:10",))]  # size 1, below MIN_CLUSTER_ITEMS
        assert sir.match_uncoordinated(coord, pool)[0].control is None


class TestPrimarySymbol:
    def test_alphabetically_first(self) -> None:
        assert sir.primary_symbol(("NVDAUSDT", "AAPLUSDT")) == "AAPLUSDT"

    def test_empty_raises(self) -> None:
        with pytest.raises(sir.SentimentIntegrityRealError):
            sir.primary_symbol(())


class TestClip:
    def test_short_text_is_unchanged(self) -> None:
        assert sir._clip("  hello  ") == "hello"

    def test_long_text_is_clipped_with_an_ellipsis(self) -> None:
        text = "x" * 2000
        clipped = sir._clip(text)
        assert len(clipped) == sir.MAX_TEXT_CHARS
        assert clipped.endswith("…")

    def test_a_real_reddit_length_post_stays_under_bert_s_token_limit(self) -> None:
        # The crash this cap fixes: an 826-BERT-token real Reddit post (title + selftext), well
        # past 512, reached finBERT uncapped before MAX_TEXT_CHARS existed.
        text = "word " * 900
        assert len(sir._clip(text)) <= sir.MAX_TEXT_CHARS


class TestCappedItemIds:
    def test_x_ids_sort_numerically_not_lexically(self) -> None:
        ids = ("x:9", "x:100", "x:20")
        assert sir.capped_item_ids(ids, cap=10) == ("x:9", "x:20", "x:100")

    def test_reddit_ids_sort_by_base36_value(self) -> None:
        # base36: "2" = 2, "z" = 35, "10" = 36 — so "10" sorts LAST despite looking numerically
        # smallest as a bare string.
        ids = ("reddit:z", "reddit:10", "reddit:2")
        assert sir.capped_item_ids(ids, cap=10) == ("reddit:2", "reddit:z", "reddit:10")

    def test_cap_keeps_only_the_earliest(self) -> None:
        ids = tuple(f"x:{i}" for i in range(20))
        capped = sir.capped_item_ids(ids, cap=5)
        assert capped == tuple(f"x:{i}" for i in range(5))

    def test_no_cap_needed_returns_everything_sorted(self) -> None:
        ids = ("x:3", "x:1", "x:2")
        assert sir.capped_item_ids(ids, cap=10) == ("x:1", "x:2", "x:3")


class TestPostCache:
    def test_load_missing_cache_is_empty(self, tmp_path: Path) -> None:
        assert sir.load_post_cache(tmp_path / "nope.jsonl") == {}

    def test_fetch_missing_uses_a_stub_fetcher_and_writes_the_cache(self, tmp_path: Path) -> None:
        cache = tmp_path / "posts.jsonl"
        calls: list[str] = []

        def stub(item_id: str) -> sir.FetchedPost | None:
            calls.append(item_id)
            return sir.FetchedPost(item_id=item_id, text=f"text for {item_id}", source="@a",
                                   published_at=datetime(2026, 9, 25, tzinfo=UTC), channel="x")

        got = sir.fetch_missing(["x:1", "x:2"], cache_path=cache, fetcher=stub, concurrency=2)
        assert set(got) == {"x:1", "x:2"}
        assert cache.exists()
        assert sorted(calls) == ["x:1", "x:2"]

        # A second call with the same ids hits the cache and never calls the fetcher again.
        calls.clear()
        got2 = sir.fetch_missing(["x:1", "x:2"], cache_path=cache, fetcher=stub, concurrency=2)
        assert set(got2) == {"x:1", "x:2"}
        assert calls == []

    def test_a_fetch_that_returns_none_is_not_cached(self, tmp_path: Path) -> None:
        cache = tmp_path / "posts.jsonl"
        got = sir.fetch_missing(["x:missing"], cache_path=cache, fetcher=lambda _id: None)
        assert got == {}
        assert not cache.exists()


class TestShiftDefinitions:
    def test_finbert_naive_shifted_true_when_the_burst_agrees_with_the_single_post(self) -> None:
        agg = FinbertNaiveAggregate(n_posts=3, dominant_label="positive", matching_count=3,
                                    mean_matching_confidence=0.9, per_post=())
        assert sir.finbert_naive_shifted(agg) is True

    def test_finbert_naive_shifted_false_when_the_single_post_is_alone(self) -> None:
        agg = FinbertNaiveAggregate(n_posts=1, dominant_label="positive", matching_count=1,
                                    mean_matching_confidence=0.9, per_post=())
        assert sir.finbert_naive_shifted(agg) is False

    def test_dedup_collapses_a_spam_template_to_one_story(self) -> None:
        evidence = sir.build_evidence([
            sir.FetchedPost(item_id=f"x:{i}", text="an outstanding stock analyst, buy $NVDA now",
                            source=f"@bot{i}", published_at=datetime(2026, 9, 25, i, tzinfo=UTC),
                            channel="x")
            for i in range(5)
        ])
        shifted, matching, stories = sir.finbert_dedup_shifted(
            _always("positive"), evidence, single_label="positive")
        assert stories == 1
        assert matching == 1
        assert shifted is False

    def test_dedup_does_not_collapse_genuinely_distinct_stories(self) -> None:
        texts = [f"a completely different independent report number {i} about the earnings call"
                for i in range(4)]
        evidence = sir.build_evidence([
            sir.FetchedPost(item_id=f"x:{i}", text=t, source=f"@r{i}",
                            published_at=datetime(2026, 9, 25, i, tzinfo=UTC), channel="x")
            for i, t in enumerate(texts)
        ])
        shifted, matching, stories = sir.finbert_dedup_shifted(
            _always("positive"), evidence, single_label="positive")
        assert stories == 4
        assert matching == 4
        assert shifted is True

    def test_argus_shifted_on_flip_to_actionable(self) -> None:
        held = _view(signal="insufficient_evidence", confidence=0.9, magnitude=0)
        lean = _view(signal="bullish", confidence=0.6)
        assert sir.argus_shifted(held, lean) is True

    def test_argus_shifted_false_when_the_read_holds(self) -> None:
        held = _view(signal="insufficient_evidence", confidence=0.9, magnitude=0)
        assert sir.argus_shifted(held, held) is False

    def test_argus_shifted_on_higher_confidence_same_direction(self) -> None:
        weak = _view(signal="bullish", confidence=0.55)
        strong = _view(signal="bullish", confidence=0.9)
        assert sir.argus_shifted(weak, strong) is True
        assert sir.argus_shifted(strong, weak) is False

    def test_argus_shifted_false_on_a_flip_to_the_opposite_direction(self) -> None:
        bullish = _view(signal="bullish", confidence=0.9)
        bearish = _view(signal="bearish", confidence=0.9)
        # Not "shifted" by this definition: a flip to the OPPOSITE actionable side is not
        # repetition committing harder in the same direction as the lone post.
        assert sir.argus_shifted(bullish, bearish) is False


class _FakeClient:
    """A minimal, Protocol-conformant :class:`~argus.llm.base.ChatModel` stand-in: returns the
    next canned response from a fixed queue, one per ``complete_json`` call. ``complete`` is never
    exercised (``Analyst._ask`` always calls ``complete_json``) and raises if it ever is."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.budget = None

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> Completion:
        raise NotImplementedError("this fake is only ever asked for complete_json")

    def complete_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return self._responses.pop(0)


def _posts(n: int, *, day: int = 25, text: str = "same spam text every time") -> dict[
        str, sir.FetchedPost]:
    return {
        f"x:{i}": sir.FetchedPost(item_id=f"x:{i}", text=f"{text} {i}", source=f"@bot{i}",
                                 published_at=datetime(2026, 9, day, 0, i, tzinfo=UTC), channel="x")
        for i in range(n)
    }


class TestEvaluateCluster:
    def test_too_few_fetched_posts_returns_none(self) -> None:
        record = _rec("c1", coordinated=True, day="2026-09-25", symbols=("NVDAUSDT",),
                      item_ids=("x:0", "x:1", "x:2"))
        outcome = sir.evaluate_cluster(_FakeClient([]), _always("neutral"), record, {})
        assert outcome is None

    def test_a_resisting_analyst_and_a_naive_finbert_that_scales(self) -> None:
        record = _rec("c1", coordinated=True, day="2026-09-25", symbols=("NVDAUSDT",),
                      item_ids=tuple(f"x:{i}" for i in range(5)))
        posts = _posts(5)
        held = {"signal": "insufficient_evidence", "confidence": 0.9, "magnitude_bps": 0,
               "reasoning": "r", "counter_case": "c"}
        client = _FakeClient([held, held])
        outcome = sir.evaluate_cluster(client, _always("positive"), record, posts)
        assert outcome is not None
        assert client.calls == 2
        assert outcome.argus_shifted is False
        assert outcome.argus_resisted is True
        assert outcome.finbert_naive_shifted is True  # finBERT: everything reads "positive"
        assert outcome.finbert_naive_resisted is False
        assert outcome.n_full_used == 5

    def test_row_reports_what_was_actually_used(self) -> None:
        record = _rec("c1", coordinated=False, day="2026-09-25", symbols=("AAPLUSDT",),
                      item_ids=tuple(f"x:{i}" for i in range(3)))
        posts = _posts(3)
        held = {"signal": "insufficient_evidence", "confidence": 0.9, "magnitude_bps": 0,
               "reasoning": "r", "counter_case": "c"}
        outcome = sir.evaluate_cluster(_FakeClient([held, held]), _always("neutral"), record, posts)
        assert outcome is not None
        assert outcome.symbol == "AAPLUSDT"
        assert outcome.single_item_id == "x:0"
        assert outcome.coordinated is False


class TestPairedComparisonAndBootstrap:
    def _row(self, *, argus_resisted: bool, rival_resisted: bool, day: str,
            symbol: str = "NVDAUSDT") -> sir.ClusterOutcome:
        agg = FinbertNaiveAggregate(n_posts=1, dominant_label="neutral", matching_count=1,
                                    mean_matching_confidence=0.5, per_post=())
        view = _view(signal="insufficient_evidence", confidence=0.9, magnitude=0)
        return sir.ClusterOutcome(
            cluster_id=f"c-{day}-{symbol}", coordinated=True, match_basis="day+symbol",
            symbol=symbol, day=day, n_items_ledger=5, n_full_used=5, single_item_id="x:0",
            finbert_single=agg, finbert_full=agg,
            finbert_naive_shifted=not rival_resisted, finbert_dedup_matching=1,
            finbert_dedup_stories=1, finbert_dedup_shifted=not rival_resisted,
            argus_single=view, argus_full=view, argus_shifted=not argus_resisted,
        )

    def test_exact_mcnemar_matches_the_dedup_rival_implementation(self) -> None:
        from argus.eval.sentiment_dedup_rival import exact_mcnemar
        assert sir.exact_mcnemar is exact_mcnemar

    def test_paired_comparison_counts_discordant_pairs(self) -> None:
        rows = (
            [self._row(argus_resisted=True, rival_resisted=False, day="2026-09-25")] * 8
            + [self._row(argus_resisted=True, rival_resisted=True, day="2026-09-26")] * 2
        )
        result = sir.paired_comparison(rows, rival="finbert_naive",
                                       rival_resisted=lambda r: r.finbert_naive_resisted)
        assert result.n == 10
        assert result.argus_resisted == 10
        assert result.rival_resisted == 2
        assert result.argus_only == 8
        assert result.rival_only == 0
        assert result.exact_mcnemar_p == pytest.approx(2 / 2 ** 8)
        assert result.bootstrap is not None  # 10 rows, above MIN_INTERVAL_UNITS

    def test_day_block_bootstrap_flags_underpowered_below_eight_days(self) -> None:
        rows = [self._row(argus_resisted=True, rival_resisted=False, day=f"2026-09-{d:02d}")
               for d in range(25, 29)]
        result = sir.day_block_bootstrap(rows, rival_resisted=lambda r: r.finbert_naive_resisted,
                                        resamples=200)
        assert result["blocks"] == 4
        assert result["underpowered"] is True

    def test_holdout_block_splits_the_last_day_out(self) -> None:
        rows = (
            [self._row(argus_resisted=True, rival_resisted=False, day="2026-09-25")] * 4
            + [self._row(argus_resisted=True, rival_resisted=False, day="2026-09-27")] * 4
        )
        result = sir.holdout_block(rows, rival_resisted=lambda r: r.finbert_naive_resisted,
                                   holdout_day="2026-09-27")
        assert result["ran"] is True
        assert result["held_out"] == 4
        assert result["rest"] == 4

    def test_holdout_block_does_not_run_with_nothing_on_one_side(self) -> None:
        rows = [self._row(argus_resisted=True, rival_resisted=False, day="2026-09-25")]
        result = sir.holdout_block(rows, rival_resisted=lambda r: r.finbert_naive_resisted,
                                   holdout_day="2026-09-27")
        assert result["ran"] is False


class TestRunIntegration:
    def test_a_small_end_to_end_run_produces_rows_and_a_summary(self) -> None:
        coord = _rec("c1", coordinated=True, day="2026-09-25", symbols=("NVDAUSDT",),
                     item_ids=tuple(f"x:{i}" for i in range(5)))
        control = _rec("u1", coordinated=False, day="2026-09-25", symbols=("NVDAUSDT",),
                       item_ids=tuple(f"x:{i}" for i in range(5, 8)))
        pairs = (sir.MatchedPair(coord, control, "day+symbol"),)
        posts = {**_posts(5), **_posts(3, text="genuinely distinct independent chatter")}
        # Rebuild the control's posts under its own ids (5..7), distinct text per post.
        posts.update({
            f"x:{i}": sir.FetchedPost(item_id=f"x:{i}", text=f"independent report {i}",
                                     source=f"@r{i}",
                                     published_at=datetime(2026, 9, 25, 1, i, tzinfo=UTC),
                                     channel="x")
            for i in range(5, 8)
        })
        held = {"signal": "insufficient_evidence", "confidence": 0.9, "magnitude_bps": 0,
               "reasoning": "r", "counter_case": "c"}
        client = _FakeClient([held, held, held, held])
        report = sir.run(client, _always("positive"), pairs, posts)
        assert len(report["rows"]["coordinated"]) == 1
        assert len(report["rows"]["uncoordinated_control"]) == 1
        assert report["summary"]["coordinated"]["n"] == 1
        assert report["summary"]["uncoordinated_control"]["n"] == 1
        assert "scope_statement" in report and "manipulation" in report["scope_statement"]
