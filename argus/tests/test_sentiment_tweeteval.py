"""The TweetEval evaluation: its metrics against their definitions (and scikit-learn, which
TweetEval's own script calls), its refusal to score misaligned data or publish benchmark text,
and the pulse regression check on text written for these tests. No model is called here; the
real run is ``python -m argus.eval.sentiment_tweeteval``."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from argus.eval import sentiment_tweeteval as st
from argus.eval.artefact import is_strict
from argus.market import social_pulse
from argus.market.abuse import DEFAULT_SCREEN

NOW = datetime(2026, 9, 25, 12, tzinfo=UTC)
NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def _split(texts: list[str], labels: list[int], names: dict[int, str] | None = None,
           task: str = "sentiment") -> st.Split:
    return st.Split(task=task, split="test", texts=tuple(texts), labels=tuple(labels),
                    names=names or NAMES)


class TestMetrics:
    def test_a_trivial_classifier_scores_one_third_whatever_the_imbalance(self) -> None:
        gold = [1] * 60 + [0] * 30 + [2] * 10
        always_neutral = [1] * 100
        assert st.macro_recall(gold, always_neutral, (0, 1, 2)) == pytest.approx(1 / 3)
        assert st.accuracy(gold, always_neutral) == pytest.approx(0.6)

    def test_macro_recall_by_hand(self) -> None:
        gold = [0, 0, 1, 1, 2, 2]
        pred = [0, 1, 1, 1, 2, 0]
        assert st.macro_recall(gold, pred, (0, 1, 2)) == pytest.approx((0.5 + 1 + 0.5) / 3)

    def test_an_undefined_precision_counts_as_zero(self) -> None:
        scores = st.per_class(st.confusion([0, 1], [0, 0], (0, 1)))
        assert scores[1] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_agrees_with_scikit_learn_as_tweeteval_calls_it(self) -> None:
        metrics = pytest.importorskip("sklearn.metrics")
        rng = random.Random(7)
        gold = [rng.choice((0, 1, 2)) for _ in range(500)]
        pred = [g if rng.random() < 0.6 else rng.choice((0, 1, 2)) for g in gold]
        report: dict[str, Any] = metrics.classification_report(gold, pred, output_dict=True)
        assert st.macro_recall(gold, pred, (0, 1, 2)) == pytest.approx(
            report["macro avg"]["recall"], abs=1e-12)
        assert st.macro_f1(gold, pred, (0, 1, 2)) == pytest.approx(
            report["macro avg"]["f1-score"], abs=1e-12)

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError):
            st.confusion([0, 1], [0], (0, 1))

    def test_bootstrap_is_seeded_and_brackets_the_estimate(self) -> None:
        rng = random.Random(3)
        gold = [rng.choice((0, 1, 2)) for _ in range(300)]
        pred = [g if rng.random() < 0.7 else 1 for g in gold]
        first = st.bootstrap_ci(gold, pred, (0, 1, 2), st.macro_recall, resamples=200)
        again = st.bootstrap_ci(gold, pred, (0, 1, 2), st.macro_recall, resamples=200)
        assert first == again
        assert first[0] <= st.macro_recall(gold, pred, (0, 1, 2)) <= first[1]

    def test_paired_bootstrap_reports_the_real_difference(self) -> None:
        gold = [0, 1, 2] * 50
        better = list(gold)
        worse = [1] * len(gold)
        out = st.paired_bootstrap(gold, better, worse, (0, 1, 2), st.macro_recall,
                                  resamples=100)
        assert out["difference"] == pytest.approx(1 - 1 / 3)
        assert out["share_of_resamples_first_not_ahead"] == 0.0


class TestData:
    def _write(self, root: Path, texts: str, labels: str) -> None:
        folder = root / "sentiment"
        folder.mkdir(parents=True)
        (folder / "test_text.txt").write_bytes(texts.encode("utf-8"))
        (folder / "test_labels.txt").write_bytes(labels.encode("utf-8"))
        (folder / "mapping.txt").write_bytes(b"0\tnegative\n1\tneutral\n2\tpositive\n")

    def test_crlf_files_load_and_a_line_separator_inside_a_tweet_does_not_split_it(
            self, tmp_path: Path) -> None:
        crlf = chr(13) + chr(10)
        texts = crlf.join(["first", f"second has a {chr(0x2028)} inside", "third", ""])
        self._write(tmp_path, texts, crlf.join(["0", "1", "2", ""]))
        split = st.load_split(tmp_path, "sentiment", "test")
        assert len(split.texts) == 3 and split.labels == (0, 1, 2)
        assert split.class_counts() == {"negative": 1, "neutral": 1, "positive": 1}

    def test_misaligned_files_are_refused(self, tmp_path: Path) -> None:
        self._write(tmp_path, "a\nb\nc\n", "0\n1\n")
        with pytest.raises(st.TweetEvalDataError, match="misaligned"):
            st.load_split(tmp_path, "sentiment", "test")

    def test_a_label_outside_the_mapping_is_refused(self, tmp_path: Path) -> None:
        self._write(tmp_path, "a\n", "7\n")
        with pytest.raises(st.TweetEvalDataError, match="not in mapping"):
            st.load_split(tmp_path, "sentiment", "test")

    def test_missing_data_names_the_fix(self, tmp_path: Path) -> None:
        with pytest.raises(st.TweetEvalDataError, match="ARGUS_TWEETEVAL_DIR"):
            st.load_split(tmp_path, "hate", "test")

    def test_the_blob_hash_is_git_s_and_ignores_crlf(self, tmp_path: Path) -> None:
        lf = tmp_path / "lf.txt"
        crlf = tmp_path / "crlf.txt"
        lf.write_bytes(b"hello\n")
        crlf.write_bytes(b"hello\r\n")
        assert st.git_blob_sha1(lf) == "ce013625030ba8dba906f756967f9e9ca394464a"
        assert st.git_blob_sha1(crlf) == st.git_blob_sha1(lf)

    def test_reference_predictions_must_align_with_the_gold_labels(self, tmp_path: Path) -> None:
        datasets = tmp_path / "datasets"
        split = _split(["a", "b", "c"], [0, 1, 2])
        with pytest.raises(st.TweetEvalDataError, match="missing"):
            st.reference_predictions(datasets, split)
        (tmp_path / "predictions").mkdir()
        (tmp_path / "predictions" / "sentiment.txt").write_text("0\n1\n", encoding="utf-8")
        with pytest.raises(st.TweetEvalDataError, match="misaligned"):
            st.reference_predictions(datasets, split)
        (tmp_path / "predictions" / "sentiment.txt").write_text("0\n1\n2\n", encoding="utf-8")
        assert st.reference_predictions(datasets, split) == [0, 1, 2]


class TestNoBenchmarkTextInTheArtefact:
    def test_a_report_quoting_a_tweet_is_refused(self) -> None:
        tweet = "this sentence stands in for a benchmark tweet"
        with pytest.raises(st.TweetEvalDataError):
            st.assert_no_text({"scores": {"note": f"e.g. {tweet}"}}, [tweet])

    def test_counts_and_short_strings_pass(self) -> None:
        st.assert_no_text({"macro_recall": 0.5, "names": ["neutral"]},
                          ["neutral", "a longer benchmark tweet that is not in the report"])


class TestSelection:
    def _row(self, recall: float, clean: float, ordinary: float) -> dict[str, Any]:
        return {"mean_abusive_withheld_share": recall, "sentiment_withheld_share": ordinary,
                "offensive": {"clean_withheld_share": clean}}

    def test_the_most_abuse_withheld_within_the_limits_wins(self) -> None:
        table = {"narrow": self._row(0.30, 0.02, 0.03), "broad": self._row(0.50, 0.10, 0.09),
                 "middle": self._row(0.42, 0.05, 0.05)}
        assert st.select(table) == "middle"

    def test_nothing_within_the_limits_is_an_error(self) -> None:
        with pytest.raises(st.TweetEvalDataError):
            st.select({"broad": self._row(0.5, 0.2, 0.2)})

    def test_the_allowlist_is_not_a_selection_candidate(self) -> None:
        assert all(screen.allowlist for screen in st.CANDIDATES)
        assert not st.WITHOUT_ALLOWLIST.allowlist


def _keyword_tone(text: str) -> float:
    """A stand-in scorer so these tests do not depend on the VADER clone."""
    lowered = text.lower()
    return 0.6 if "moon" in lowered or "great" in lowered else (
        -0.6 if "awful" in lowered or "idiot" in lowered else 0.0)


class TestCrowdRead:
    def test_each_tweet_gets_the_class_the_pulse_published_and_withheld_reads_neutral(
            self) -> None:
        split = _split(["NVDA to the moon", "awful earnings", "you are a fucking idiot",
                        "flat day on the tape"], [2, 0, 0, 1])
        read = st.crowd_read(split, NOW, tone=_keyword_tone)
        assert read.pred == [2, 0, 1, 1]
        assert read.unscreened == [2, 0, 0, 1]
        assert read.kept == [0, 1, 3]
        assert read.extra["withheld_by_the_abuse_screen"] == 1
        assert read.extra["withheld_share_by_sentiment_class"]["negative"] == pytest.approx(0.5)
        bias = read.extra["tone_bias_from_the_abuse_screen"]
        assert bias["withheld_tweets_as_vader_would_have_read_them"]["negative"] == 1
        assert bias["gold_share_all_tweets"]["negative"] == pytest.approx(0.5)
        assert bias["gold_share_tweets_the_screen_kept"]["negative"] == pytest.approx(1 / 3)
        screen = bias["per_sweep_share_difference"]["negative_screened_minus_unscreened"]
        assert screen["mean"] == pytest.approx(1 / 3 - 1 / 2) and screen["sweeps"] == 1

    def test_a_pulse_that_stops_scoring_tone_stops_the_run(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        def without_tone(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"posts": 1, "withheld_abusive": 0, "tone": {"status": "not scored: gone"},
                    "post_tone": []}

        monkeypatch.setattr(social_pulse, "pulse", without_tone)
        with pytest.raises(st.TweetEvalDataError, match="scored no tone"):
            st.crowd_read(_split(["one tweet"], [1]), NOW)

    def test_a_kept_post_without_a_tone_stops_the_run(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        def missing_one(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"posts": 2, "withheld_abusive": 0, "tone": {"status": "ok"},
                    "post_tone": [{"id": "te0", "label": "neutral"}]}

        monkeypatch.setattr(social_pulse, "pulse", missing_one)
        with pytest.raises(st.TweetEvalDataError, match="every post it kept"):
            st.crowd_read(_split(["one tweet", "two tweets"], [1, 1]), NOW)

    def test_the_bias_verdict_follows_the_interval(self) -> None:
        def cell(mean: float, low: float, high: float) -> dict[str, Any]:
            return {"mean": mean, "ci95": [low, high], "sweeps": 205}

        bias: dict[str, Any] = {"gold_share_all_tweets": {"negative": 0.323},
                                "gold_share_tweets_the_screen_kept": {"negative": 0.305},
                                "withheld_tweets_as_vader_would_have_read_them": {"negative": 400},
                                "coordinated_stories_counted_once": 3,
                                "per_sweep_share_difference": {
            "negative_screened_minus_unscreened": cell(-0.04, -0.045, -0.035),
            "negative_screened_minus_gold": cell(-0.08, -0.09, -0.07),
            "negative_unscreened_minus_gold": cell(-0.04, -0.05, -0.03),
            "negative_weighted_minus_unweighted": cell(0.0, 0.0, 0.0)}}
        assert "lowers the negative share" in st.tone_bias_verdict(bias)
        bias["per_sweep_share_difference"]["negative_screened_minus_unscreened"] = cell(
            0.0, -0.01, 0.01)
        assert "does not measurably move" in st.tone_bias_verdict(bias)

    def test_mean_ci_brackets_the_mean(self) -> None:
        out = st.mean_ci([0.1, 0.2, 0.3, 0.4], resamples=200)
        assert out["mean"] == pytest.approx(0.25) and out["sweeps"] == 4
        assert out["ci95"][0] <= 0.25 <= out["ci95"][1]


class TestPipelineRegression:
    def _split(self) -> st.Split:
        texts = ["the devs are fucking idiots, rug incoming", "kys bagholders",
                 "earnings next week, guidance matters", "rates cut odds rising into FOMC"]
        return _split(texts, [1, 1, 0, 0], {0: "not-hate", 1: "hate"}, task="hate")

    def test_before_the_screen_a_brigade_was_counted_and_quoted(self) -> None:
        split = self._split()
        run = st._pipeline_run(split.texts, split.labels, brigade=True, screened=False, now=NOW)
        assert run["abusive_posts_counted"] == run["abusive_posts"] == 6
        assert run["coordinated_abusive_stories_in_snapshot"] == 2
        assert run["abusive_texts_quoted_to_reader"] == 1

    def test_with_the_screen_nothing_abusive_is_counted_or_quoted(self) -> None:
        out = st.pipeline_check(self._split(), NOW)
        for scenario in ("organic", "brigade"):
            screened = out[scenario]["screened"]
            assert screened["abusive_posts_counted"] == 0
            assert screened["abusive_texts_in_snapshot"] == 0
            assert screened["abusive_texts_quoted_to_reader"] == 0
        assert out["organic"]["screened"]["clean_posts_withheld"] == 0

    def test_screen_scores_report_the_rates_the_pulse_runs_on(self) -> None:
        scores = st.screen_scores(DEFAULT_SCREEN, self._split())
        assert scores["abusive_withheld_share"] == 1.0 and scores["clean_withheld_share"] == 0.0
        assert scores["macro_f1"] == 1.0


class TestReportPieces:
    def test_the_analyst_run_size_comes_from_the_measured_spread(self) -> None:
        out = st.analyst_not_run([0.60, 0.62], 12284)
        assert out["status"] == "not_run" and out["qwen_calls"] == 0
        assert out["tweets_needed"] == int(12284 * (0.01 / 0.03) ** 2) + 1

    def test_the_gap_closers_are_priced_by_measured_or_cited_numbers(self) -> None:
        def row(recall: float) -> dict[str, Any]:
            return {"macro_recall": recall}

        rows = st.paper_rows("sentiment")
        scorers = {"argus_crowd_read": row(0.56), "vader": row(0.57), "finbert": row(0.50),
                   "argus_sentiment_analyst": st.analyst_not_run([0.60, 0.62], 12284)}
        transformer, analyst, finbert, vader = st.gap_closers(scorers, rows)
        best = max(rows.values())
        # Closures are measured from the floor the crowd read stood at before it scored tone.
        assert vader["closes_of_the_gap_to_best"] == pytest.approx(
            (0.57 - 1 / 3) / (best - 1 / 3))
        assert vader["crowd_read_closes_of_the_gap_to_best"] == pytest.approx(
            (0.56 - 1 / 3) / (best - 1 / 3))
        assert "adopted" in vader["status"] and vader["argus_crowd_read_with_it"] == 0.56
        assert "SVM" in vader["cost"] and "BLSTM" in vader["cost"]
        assert transformer["macro_recall"]["best_reported"] == best
        assert transformer["closes_of_the_gap_to_best"] == 1.0
        assert analyst["macro_recall"] is None and "Qwen" in analyst["option"]
        assert finbert["macro_recall"] == 0.50
        verdict = st.sentiment_verdict(scorers, rows)
        assert "not run" in verdict and verdict.startswith("ARGUS LOSES")
        assert "0.560" in verdict and "0.333" in verdict

    def test_the_paper_numbers_are_table_3(self) -> None:
        assert st.PAPER_TABLE3_TEST["RoBERTa-Retrained (RoB-RT)"]["sentiment"] == 72.6
        assert st.PAPER_TABLE3_TEST["SVM"]["hate"] == 36.7
        assert st.README_LEADERBOARD["TimeLMs-2021"]["offensive"] == 82.2
        rows = st.paper_rows("sentiment")
        assert min(rows.values()) == pytest.approx(0.583)


class _FakePipeline:
    """Stands in for the transformers pipeline: labels by a keyword, records what it was sent."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, texts: list[str], **_: Any) -> list[dict[str, Any]]:
        self.seen.extend(texts)
        return [{"label": "positive" if "up" in t else "negative" if "down" in t else "neutral",
                 "score": 0.9} for t in texts]


class TestFinbertProgress:
    TEXTS = ("up only", "down bad", "flat day", "going up", "sideways")

    def _run(self, monkeypatch: pytest.MonkeyPatch, cache: Path,
             blob: str = "blob") -> tuple[list[int], dict[str, Any], _FakePipeline]:
        from argus.eval.baselines import finbert_loader

        fake = _FakePipeline()
        monkeypatch.setattr(finbert_loader, "load_finbert", lambda: fake)
        split = _split(list(self.TEXTS), [2, 0, 1, 2, 1])
        labels, extra = st.finbert_predictions(split, check_batching=2, cache=cache,
                                               text_blob=blob)
        return labels, extra, fake

    def test_an_interrupted_run_resumes_and_writes_through(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        cache = tmp_path / "progress.json"
        cache.write_text(json.dumps({"text_blob": "blob", "labels": [2, -1, 1, -1, -1],
                                     "seconds": 1.5}), encoding="utf-8")
        labels, extra, fake = self._run(monkeypatch, cache)
        assert labels == [2, 0, 1, 2, 1]
        assert extra["tweets_resumed_from_an_interrupted_run"] == 2
        assert sorted(fake.seen[:3]) == sorted(["down bad", "going up", "sideways"])
        saved = json.loads(cache.read_text(encoding="utf-8"))
        assert saved["labels"] == labels and saved["seconds"] >= 1.5
        assert not any(isinstance(v, str) for v in saved["labels"])
        assert extra["batching_label_agreement"] == {"tweets": 2, "agree": 2}

    def test_progress_for_other_text_is_ignored(self, monkeypatch: pytest.MonkeyPatch,
                                                tmp_path: Path) -> None:
        cache = tmp_path / "progress.json"
        cache.write_text(json.dumps({"text_blob": "other", "labels": [0, 0, 0, 0, 0]}),
                         encoding="utf-8")
        labels, extra, fake = self._run(monkeypatch, cache)
        assert labels == [2, 0, 1, 2, 1] and extra["tweets_resumed_from_an_interrupted_run"] == 0
        assert len(fake.seen) == len(self.TEXTS) + 2


@pytest.fixture(scope="module")
def root() -> Path:
    """The research corpus's copy of TweetEval, read in place, when it is on this machine."""
    found = st.tweeteval_dir()
    if not (found / "sentiment" / "test_text.txt").is_file():
        pytest.skip(f"TweetEval not present at {found}")
    return found


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    if not st.REPORT_PATH.is_file():
        pytest.skip("run python -m argus.eval.sentiment_tweeteval first")
    loaded: dict[str, Any] = json.loads(st.REPORT_PATH.read_text(encoding="utf-8"))
    return loaded


class TestOnTheRealData:

    def test_the_local_copy_is_the_upstream_one(self, root: Path) -> None:
        checked = st.provenance(root, st.UPSTREAM_BLOBS)
        assert all(entry["matches_upstream"] for entry in checked.values()), checked

    def test_the_sentiment_test_split_is_the_imbalanced_one_the_docstring_describes(
            self, root: Path) -> None:
        split = st.load_split(root, "sentiment", "test")
        assert split.class_counts() == {"negative": 3972, "neutral": 5937, "positive": 2375}

    def test_the_published_rob_rt_predictions_rescore_close_to_the_paper(
            self, root: Path) -> None:
        checked = st.provenance(root.parent, st.REFERENCE_BLOBS, st.REFERENCE_BLOBS)
        assert all(entry["matches_upstream"] for entry in checked.values()), checked
        split = st.load_split(root, "sentiment", "test")
        rescored = st.macro_recall(split.labels, st.reference_predictions(root, split),
                                   split.classes)
        # One run against Table 3's mean of three (72.6) and the README's 72.8.
        assert rescored == pytest.approx(0.726, abs=0.01)


class TestTheArtefact:
    def test_it_is_strict_json_and_made_no_qwen_call(self, report: dict[str, Any]) -> None:
        assert is_strict(st.REPORT_PATH)
        assert report["qwen_calls"] == 0
        assert report["sentiment"]["scorers"]["argus_sentiment_analyst"]["status"] == "not_run"

    def test_every_scorer_read_the_whole_split_and_the_loss_is_recorded(
            self, report: dict[str, Any]) -> None:
        sentiment = report["sentiment"]
        for name in ("argus_crowd_read", "finbert", "vader", "majority_class"):
            assert sum(sentiment["scorers"][name]["predicted_counts"].values()) == 12284, name
        crowd = sentiment["scorers"]["argus_crowd_read"]
        vader = sentiment["scorers"]["vader"]
        # The crowd read scores tone now: above the floor, at most VADER alone (the screen can
        # only take tweets away from it), and the unscreened pulse is VADER exactly.
        assert 1 / 3 < crowd["macro_recall"] <= vader["macro_recall"]
        agree = crowd["unscreened_pulse_agrees_with_vader_alone"]
        assert agree["agree"] == agree["tweets"] == 12284
        assert "argus_crowd_read_minus_vader" in sentiment["paired_bootstrap_macro_recall"]
        assert "tone_bias_from_the_abuse_screen" in sentiment
        assert len(sentiment["what_would_close_the_gap"]) == 4

    def test_the_selected_screen_is_the_production_one(self, report: dict[str, Any]) -> None:
        assert report["abuse_screen"]["selection"]["selected_is_production"] is True

    def test_the_screen_removed_every_quote_of_a_flagged_brigade(
            self, report: dict[str, Any]) -> None:
        for task in ("hate", "offensive"):
            brigade = report["abuse_screen"]["pipeline_regression"][task]["brigade"]
            before = brigade["unscreened_as_before_2026_09_25"]
            after = brigade["screened"]
            assert before["abusive_posts_counted"] == before["abusive_posts"]
            assert after["abusive_posts_counted"] < before["abusive_posts_counted"]
            assert before["abusive_texts_quoted_to_reader"] > 0
            assert after["abusive_texts_quoted_to_reader"] == 0

    def test_it_carries_no_benchmark_text(self, report: dict[str, Any], root: Path) -> None:
        texts = [*st.load_split(root, "sentiment", "test").texts,
                 *st.load_split(root, "hate", "test").texts,
                 *st.load_split(root, "offensive", "test").texts]
        st.assert_no_text(report, texts)
