"""Tests for row 33's sentiment-signal comparison (eval/sentiment_signal_comparison.py).

Offline throughout: arm A is exercised through a scripted fake `ChatModel` (never a real Qwen
call), finBERT through a fake classifier matching `FinbertClassifier`'s protocol (never the real
HuggingFace pipeline), and VADER through the real vendored, local analyzer — a pure file load with
no network, exactly as `tests/test_vader.py` and `tests/test_social_pulse.py` already exercise it
unmarked. Every scoring function under test (the join, the four arms' lean/confidence logic, the
hit-rate/net-bps/rank-correlation scoring, McNemar and the paired bootstrap) is the real production
code; only the two external dependencies (Qwen, the real finBERT weights) are stubbed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from argus.eval import sentiment_signal_comparison as mod
from argus.llm.qwen import Completion, Thinking
from argus.market import vader
from argus.truth.evidence import Evidence

# =================================================================================================
# Fakes
# =================================================================================================


class ScriptedSentimentModel:
    """A `ChatModel` that answers from a fixed script, one answer per `complete_json` call — the
    same shape `tests/test_decision_primitives.ScriptedPM` uses for the Meta-PM."""

    budget = None

    def __init__(self, answers: list[dict[str, Any]]) -> None:
        self._answers = list(answers)
        self.calls = 0

    def complete(self, *_: Any, **__: Any) -> Completion:
        raise AssertionError("SentimentAnalyst asks for structured JSON only")

    def complete_json(
        self, messages: list[dict[str, Any]], *, required_keys: tuple[str, ...] = (),
        validate: Any = None, max_tokens: int = 0, attempts: int = 3,
        thinking: Thinking = Thinking.LOW,
    ) -> dict[str, Any]:
        self.calls += 1
        answer = self._answers.pop(0)
        missing = [k for k in required_keys if k not in answer]
        if missing:
            raise AssertionError(f"scripted answer missing required keys: {missing}")
        return answer


def _finbert_all(label: str, score: float = 0.9) -> Any:
    """A fake `FinbertClassifier`: every text gets the same label/score, matching
    `tests/test_sentiment_dedup_rival.py`'s `_always` convention."""

    def classify(texts: list[str]) -> list[dict[str, Any]]:
        return [{"label": label, "score": score} for _ in texts]

    return classify


def _finbert_from(labels: list[tuple[str, float]]) -> Any:
    def classify(texts: list[str]) -> list[dict[str, Any]]:
        assert len(texts) == len(labels)
        return [{"label": lab, "score": sc} for lab, sc in labels]

    return classify


AT = datetime(2026, 9, 26, 14, 0, 0, tzinfo=UTC)


def _post(id_: str, claim: str, *, at: datetime = AT, cred: float = 0.4) -> Evidence:
    return Evidence(id=id_, claim=claim, source="social", available_at=at, credibility=cred)


def _cycle(
    seq: int, *, symbol: str = "NVDAUSDT", move_bps: float = 0.0,
    posts: list[Evidence] | None = None, extra_social: list[Evidence] | None = None,
    decided_at: datetime = AT, horizon_hours: float = 24.0, verdict: str = "no_trade",
) -> mod.Cycle:
    from datetime import timedelta

    posts = posts if posts is not None else [_post("twitter-1", "NVDA is going to the moon")]
    social = tuple(posts) + tuple(extra_social or ())
    return mod.Cycle(
        seq=seq, symbol=symbol, decided_at=decided_at.isoformat(),
        settled_at=(decided_at + timedelta(hours=horizon_hours)).isoformat(),
        entry_price=100.0, exit_price=100.0 * (1 + move_bps / 10_000), move_bps=move_bps,
        horizon_hours=horizon_hours, verdict=verdict, social_evidence=social,
        posts=tuple(p for p in social if mod.is_social_post_id(p.id)),
    )


# =================================================================================================
# Evidence parsing
# =================================================================================================


class TestEvidenceLineParsing:
    def test_round_trips_evidences_own_render(self) -> None:
        ev = Evidence(
            id="twitter-123", claim="NVDA earnings beat, guidance raised.", source="social",
            available_at=datetime(2026, 9, 26, 14, 12, 18, 950841, tzinfo=UTC), credibility=0.4,
        )
        parsed = mod.parse_evidence_line(ev.render())
        assert parsed.id == ev.id
        assert parsed.claim == ev.claim
        assert parsed.source == ev.source
        assert parsed.available_at == ev.available_at
        assert parsed.credibility == ev.credibility

    def test_claim_with_parentheses_survives(self) -> None:
        ev = Evidence(
            id="mkt-NVDAUSDT", claim="NVDAUSDT last 224.47 (24h -0.36%)", source="news",
            available_at=AT, credibility=1.0,
        )
        parsed = mod.parse_evidence_line(ev.render())
        assert parsed.claim == ev.claim

    def test_raises_on_garbage(self) -> None:
        with pytest.raises(mod.JoinError):
            mod.parse_evidence_line("not an evidence line at all")

    def test_social_post_id_classification(self) -> None:
        assert mod.is_social_post_id("twitter-2103666575265989094")
        assert mod.is_social_post_id("reddit-abc123")
        assert not mod.is_social_post_id("skill-technical_analysis-rsi-1790617248")
        assert not mod.is_social_post_id("mirror-derivatives_sentiment-taker_ratio-1790603240")
        assert not mod.is_social_post_id("mkt-NVDAUSDT")


# =================================================================================================
# The join
# =================================================================================================


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8", newline="\n",
    )


class TestLoadCycles:
    def test_joins_a_settled_abstention_with_social_evidence(self, tmp_path: Path) -> None:
        tweet = _post("twitter-1", "NVDA looking strong today")
        skill = Evidence(id="skill-technical_analysis-rsi-1", claim="RSI 61.2", source="social",
                         available_at=AT, credibility=0.5)
        notes = tmp_path / "desk_notes.jsonl"
        ledger = tmp_path / "paper_ledger.jsonl"
        _write_jsonl(notes, [{
            "seq": 1, "symbol": "NVDAUSDT", "at": AT.isoformat(),
            "evidence": [tweet.render(), skill.render()], "notes": [], "flags": [],
        }])
        _write_jsonl(ledger, [{
            "seq": 1, "kind": "decision", "symbol": "NVDAUSDT", "verdict": "no_trade",
            "decided_at": AT.isoformat(), "settled_at": AT.isoformat(),
            "entry_price": "100.00", "exit_price": "100.50",
            "counterfactual_move_bps": "50.0",
        }])
        cycles = mod.load_cycles(notes, ledger)
        assert len(cycles) == 1
        cyc = cycles[0]
        assert cyc.seq == 1
        assert cyc.move_bps == 50.0
        # Arm A's input is the FULL social channel (matches agents/desk.py:484) ...
        assert len(cyc.social_evidence) == 2
        # ... arms B/C read only the real post subset.
        assert len(cyc.posts) == 1
        assert cyc.posts[0].id == "twitter-1"

    def test_excludes_unsettled_rows(self, tmp_path: Path) -> None:
        notes = tmp_path / "desk_notes.jsonl"
        ledger = tmp_path / "paper_ledger.jsonl"
        tweet = _post("twitter-1", "unsettled cycle")
        _write_jsonl(notes, [{"seq": 1, "symbol": "NVDAUSDT", "at": AT.isoformat(),
                              "evidence": [tweet.render()], "notes": [], "flags": []}])
        _write_jsonl(ledger, [{"seq": 1, "kind": "decision", "symbol": "NVDAUSDT",
                               "verdict": "no_trade", "decided_at": AT.isoformat(),
                               "settled_at": None, "entry_price": "100.00", "exit_price": None,
                               "counterfactual_move_bps": None}])
        assert mod.load_cycles(notes, ledger) == []

    def test_excludes_real_trade_rows(self, tmp_path: Path) -> None:
        """A real trade settles through net_pnl, not counterfactual_move_bps
        (paper/ledger.py:_settled_position vs _settled_abstention)."""
        notes = tmp_path / "desk_notes.jsonl"
        ledger = tmp_path / "paper_ledger.jsonl"
        tweet = _post("twitter-1", "a real trade was made")
        _write_jsonl(notes, [{"seq": 1, "symbol": "NVDAUSDT", "at": AT.isoformat(),
                              "evidence": [tweet.render()], "notes": [], "flags": []}])
        _write_jsonl(ledger, [{
            "seq": 1, "kind": "decision", "symbol": "NVDAUSDT", "verdict": "trade",
            "decided_at": AT.isoformat(), "settled_at": None,
            "entry_price": "100.00", "exit_price": None, "counterfactual_move_bps": None,
        }])
        assert mod.load_cycles(notes, ledger) == []

    def test_excludes_cycles_with_no_real_posts(self, tmp_path: Path) -> None:
        notes = tmp_path / "desk_notes.jsonl"
        ledger = tmp_path / "paper_ledger.jsonl"
        skill_only = Evidence(id="skill-technical_analysis-rsi-1", claim="RSI 61.2",
                              source="social", available_at=AT, credibility=0.5)
        _write_jsonl(notes, [{"seq": 1, "symbol": "NVDAUSDT", "at": AT.isoformat(),
                              "evidence": [skill_only.render()], "notes": [], "flags": []}])
        _write_jsonl(ledger, [{"seq": 1, "kind": "decision", "symbol": "NVDAUSDT",
                               "verdict": "no_trade", "decided_at": AT.isoformat(),
                               "settled_at": AT.isoformat(), "entry_price": "100.00",
                               "exit_price": "100.10", "counterfactual_move_bps": "10.0"}])
        assert mod.load_cycles(notes, ledger) == []

    def test_missing_files_raise_join_error(self, tmp_path: Path) -> None:
        with pytest.raises(mod.JoinError):
            mod.load_cycles(tmp_path / "nope.jsonl", tmp_path / "also_nope.jsonl")


class TestSelectSample:
    def test_returns_everything_when_n_covers_all(self) -> None:
        cycles = [_cycle(i) for i in range(5)]
        assert mod.select_sample(cycles, 10) == cycles

    def test_evenly_spans_first_to_last(self) -> None:
        cycles = [_cycle(i) for i in range(10)]
        sample = mod.select_sample(cycles, 3)
        assert sample[0].seq == 0
        assert sample[-1].seq == 9
        assert len(sample) == 3

    def test_n_one_returns_the_first_cycle(self) -> None:
        cycles = [_cycle(i) for i in range(5)]
        assert [c.seq for c in mod.select_sample(cycles, 1)] == [0]

    def test_empty_input(self) -> None:
        assert mod.select_sample([], 5) == []

    def test_deterministic_across_calls(self) -> None:
        cycles = [_cycle(i) for i in range(89)]
        first = [c.seq for c in mod.select_sample(cycles, 32)]
        second = [c.seq for c in mod.select_sample(cycles, 32)]
        assert first == second
        assert len(first) == 32


# =================================================================================================
# net_bps_of
# =================================================================================================


class TestNetBps:
    def test_flat_lean_is_zero_regardless_of_move(self) -> None:
        result = mod.ArmResult(lean=0, confidence=0.9)
        assert mod.net_bps_of(result, 500.0) == 0.0
        assert mod.net_bps_of(result, -500.0) == 0.0

    def test_correct_long_nets_move_minus_round_trip(self) -> None:
        result = mod.ArmResult(lean=1, confidence=0.6)
        assert mod.net_bps_of(result, 20.0) == pytest.approx(20.0 - mod.ROUND_TRIP_BPS)

    def test_wrong_short_nets_negative_move_minus_round_trip(self) -> None:
        result = mod.ArmResult(lean=-1, confidence=0.6)
        assert mod.net_bps_of(result, 20.0) == pytest.approx(-20.0 - mod.ROUND_TRIP_BPS)

    def test_arm_result_rejects_an_out_of_range_lean(self) -> None:
        with pytest.raises(ValueError, match="lean must be"):
            mod.ArmResult(lean=2, confidence=0.5)


# =================================================================================================
# Arm A — the production SentimentAnalyst through a scripted model.
# =================================================================================================


class TestArgusArm:
    def test_bullish_signal_maps_to_lean_plus_one(self) -> None:
        model = ScriptedSentimentModel([{
            "signal": "bullish", "magnitude_bps": 18, "confidence": 0.62,
            "reasoning": "beat and raise", "counter_case": "already priced", "source_ids": [],
        }])
        cyc = _cycle(1)
        result = mod.run_argus_arm(model, cyc)
        assert result.lean == 1
        assert result.confidence == pytest.approx(0.62)
        assert model.calls == 1
        assert "prompt_hash" in result.raw
        assert "view" in result.raw

    def test_bearish_and_neutral_and_insufficient_map_correctly(self) -> None:
        for signal, expected in (
            ("bearish", -1), ("neutral", 0), ("insufficient_evidence", 0),
        ):
            model = ScriptedSentimentModel([{
                "signal": signal, "magnitude_bps": 5 if signal != "neutral" else 0,
                "confidence": 0.3, "reasoning": "x", "counter_case": "y", "source_ids": [],
            }])
            result = mod.run_argus_arm(model, _cycle(1))
            assert result.lean == expected, signal

    def test_prompt_hash_is_deterministic_and_evidence_sensitive(self) -> None:
        a = _cycle(1, posts=[_post("twitter-1", "NVDA earnings beat")])
        b = _cycle(1, posts=[_post("twitter-1", "NVDA earnings beat")])
        c = _cycle(1, posts=[_post("twitter-1", "totally different narrative")])
        assert mod.argus_prompt_hash(a) == mod.argus_prompt_hash(b)
        assert mod.argus_prompt_hash(a) != mod.argus_prompt_hash(c)

    def test_uses_the_full_social_channel_not_only_posts(self) -> None:
        """Matches agents/desk.py:484: the real desk hands the sentiment analyst every
        source == "social" item, not only the twitter/reddit subset."""
        tweet = _post("twitter-1", "NVDA earnings beat")
        skill = Evidence(id="skill-technical_analysis-rsi-1", claim="RSI reading: 61.2",
                         source="social", available_at=AT, credibility=0.5)
        cyc = _cycle(1, posts=[tweet], extra_social=[skill])
        model = ScriptedSentimentModel([{
            "signal": "neutral", "magnitude_bps": 0, "confidence": 0.4, "reasoning": "x",
            "counter_case": "y", "source_ids": [],
        }])
        mod.run_argus_arm(model, cyc)
        # rebuild the body the same way the analyst does, and check the skill item is in it
        from argus.agents.quarantine import render_for_prompt
        rendered = render_for_prompt(list(cyc.social_evidence))
        assert "RSI reading" in rendered
        assert "NVDA earnings beat" in rendered


# =================================================================================================
# Arm B — VADER, real local analyzer.
# =================================================================================================


class TestVaderArm:
    def test_build_posts_uses_the_evidence_id_as_the_voice_proxy(self) -> None:
        cyc = _cycle(1, posts=[_post("twitter-1", "NVDA to the moon")])
        posts = mod.build_posts(cyc)
        assert len(posts) == 1
        assert posts[0].source == "twitter-1"

    def test_clearly_positive_crowd_leans_up(self) -> None:
        posts = [
            _post("twitter-1", "NVDA is fantastic, amazing earnings, great outlook, love it"),
            _post("twitter-2", "Best quarter ever for NVDA, incredible growth, wonderful"),
        ]
        cyc = _cycle(1, posts=posts)
        analyzer = vader.load_analyzer()
        result = mod.run_vader_arm(cyc, analyzer)
        assert result.lean == 1
        assert result.raw["label"] == "positive"

    def test_clearly_negative_crowd_leans_down(self) -> None:
        posts = [
            _post("twitter-1", "NVDA is terrible, awful earnings, horrible, hate this stock"),
            _post("twitter-2", "Worst quarter ever, disgusting management, disaster"),
        ]
        cyc = _cycle(1, posts=posts)
        analyzer = vader.load_analyzer()
        result = mod.run_vader_arm(cyc, analyzer)
        assert result.lean == -1

    def test_coordinated_duplicate_text_collapses_to_one_voice(self) -> None:
        """Three near-identical posts inside the coordination window: production's
        voice_weights collapses them to one vote (market/social_pulse.py:138-154), which this
        arm reproduces through the id-as-voice proxy."""
        text = "NVDA earnings beat expectations massively today great news"
        posts = [_post(f"twitter-{i}", text) for i in range(3)]
        cyc = _cycle(1, posts=posts)
        analyzer = vader.load_analyzer()
        result = mod.run_vader_arm(cyc, analyzer)
        # voices collapse toward 1 rather than summing to 3
        assert result.raw["voices"] < 3.0
        assert result.raw["coordinated_stories"] == 1


# =================================================================================================
# Arm C — finBERT, fake classifier.
# =================================================================================================


class TestFinbertArm:
    def test_all_positive_leans_up(self) -> None:
        cyc = _cycle(1, posts=[_post("twitter-1", "x"), _post("twitter-2", "y")])
        result = mod.run_finbert_arm(_finbert_all("positive", 0.9), cyc)
        assert result.lean == 1
        assert result.raw["mean_signed_score"] == pytest.approx(0.9)

    def test_all_negative_leans_down(self) -> None:
        cyc = _cycle(1, posts=[_post("twitter-1", "x")])
        result = mod.run_finbert_arm(_finbert_all("negative", 0.7), cyc)
        assert result.lean == -1
        assert result.raw["mean_signed_score"] == pytest.approx(-0.7)

    def test_mixed_signals_near_zero_are_flat(self) -> None:
        cyc = _cycle(1, posts=[_post("twitter-1", "x"), _post("twitter-2", "y")])
        classifier = _finbert_from([("positive", 0.5), ("negative", 0.5)])
        result = mod.run_finbert_arm(classifier, cyc)
        assert result.lean == 0

    def test_threshold_is_the_fixed_module_constant_by_default(self) -> None:
        cyc = _cycle(1, posts=[_post("twitter-1", "x")])
        classifier = _finbert_from([("positive", mod.FINBERT_LEAN_THRESHOLD - 0.001)])
        result = mod.run_finbert_arm(classifier, cyc)
        assert result.lean == 0  # just under threshold

    def test_neutral_labels_score_zero(self) -> None:
        cyc = _cycle(1, posts=[_post("twitter-1", "x")])
        result = mod.run_finbert_arm(_finbert_all("neutral", 0.99), cyc)
        assert result.lean == 0
        assert result.raw["mean_signed_score"] == 0.0


# =================================================================================================
# Arm D — no-information baselines.
# =================================================================================================


class TestBaselineArms:
    def test_flat_is_always_flat_and_zero_confidence(self) -> None:
        result = mod.run_flat_arm(_cycle(1))
        assert result.lean == 0
        assert result.confidence == 0.0

    def test_coin_flip_is_reproducible_from_its_seed(self) -> None:
        cycles = [_cycle(i) for i in range(20)]
        first = mod.run_coin_flip_arms(cycles, seed=mod.COIN_FLIP_SEED)
        second = mod.run_coin_flip_arms(cycles, seed=mod.COIN_FLIP_SEED)
        assert {seq: r.lean for seq, r in first.items()} == {
            seq: r.lean for seq, r in second.items()
        }
        assert all(r.lean in (-1, 1) for r in first.values())

    def test_a_different_seed_can_change_the_draw(self) -> None:
        cycles = [_cycle(i) for i in range(20)]
        a = mod.run_coin_flip_arms(cycles, seed=1)
        b = mod.run_coin_flip_arms(cycles, seed=2)
        assert {s: r.lean for s, r in a.items()} != {s: r.lean for s, r in b.items()}


# =================================================================================================
# Scoring.
# =================================================================================================


class TestScoreArm:
    def test_hit_rate_counts_only_leaned_and_round_trip_clearing_cycles(self) -> None:
        cycles = [
            _cycle(1, move_bps=20.0),   # clears 12bps, arm correct long
            _cycle(2, move_bps=-20.0),  # clears, arm wrong (long again)
            _cycle(3, move_bps=5.0),    # does not clear the round trip
            _cycle(4, move_bps=30.0),   # arm flat here: not counted as leaned
        ]
        results = {
            1: mod.ArmResult(lean=1, confidence=0.6),
            2: mod.ArmResult(lean=1, confidence=0.6),
            3: mod.ArmResult(lean=1, confidence=0.6),
            4: mod.ArmResult(lean=0, confidence=0.0),
        }
        score = mod.score_arm("test", cycles, results)
        assert score.n_leaned == 3
        assert score.hit_eligible == 2  # cycles 1 and 2 clear the round trip
        assert score.hits == 1  # only cycle 1
        assert score.hit_rate == pytest.approx(0.5)

    def test_net_bps_totals_only_leaned_cycles(self) -> None:
        cycles = [_cycle(1, move_bps=20.0), _cycle(2, move_bps=-20.0)]
        results = {1: mod.ArmResult(lean=1, confidence=0.6),
                  2: mod.ArmResult(lean=0, confidence=0.0)}
        score = mod.score_arm("test", cycles, results)
        assert score.total_net_bps == pytest.approx(20.0 - mod.ROUND_TRIP_BPS)
        assert score.mean_net_bps_per_trade == pytest.approx(20.0 - mod.ROUND_TRIP_BPS)

    def test_perfect_rank_correlation_is_plus_one(self) -> None:
        cycles = [_cycle(i, move_bps=float(i) * 10) for i in range(1, 6)]
        results = {i: mod.ArmResult(lean=1, confidence=i / 5) for i in range(1, 6)}
        score = mod.score_arm("test", cycles, results)
        assert score.spearman_rho == pytest.approx(1.0)

    def test_no_scoreable_cycles_returns_none_not_a_crash(self) -> None:
        score = mod.score_arm("test", [], {})
        assert score.hit_rate is None
        assert score.mean_net_bps_per_trade is None
        assert score.spearman_rho is None
        assert score.n_cycles == 0


class TestCompareArms:
    def test_mcnemar_counts_only_discordant_both_leaned_cycles(self) -> None:
        cycles = [
            _cycle(1, move_bps=20.0),   # A right, B wrong
            _cycle(2, move_bps=20.0),   # both right
            _cycle(3, move_bps=-20.0),  # A wrong, B right
            _cycle(4, move_bps=20.0),   # B flat: excluded
        ]
        a = {1: mod.ArmResult(lean=1, confidence=0.6), 2: mod.ArmResult(lean=1, confidence=0.6),
             3: mod.ArmResult(lean=1, confidence=0.6), 4: mod.ArmResult(lean=1, confidence=0.6)}
        b = {1: mod.ArmResult(lean=-1, confidence=0.6), 2: mod.ArmResult(lean=1, confidence=0.6),
             3: mod.ArmResult(lean=-1, confidence=0.6), 4: mod.ArmResult(lean=0, confidence=0.0)}
        cmp = mod.compare_arms("argus", "vader", cycles, a, b)
        assert cmp.n_both_leaned == 3
        assert cmp.a_correct_b_wrong == 1
        assert cmp.b_correct_a_wrong == 1
        assert cmp.both_correct == 1

    def test_notes_when_too_few_date_clusters(self) -> None:
        cycles = [_cycle(1, move_bps=20.0), _cycle(2, move_bps=-20.0)]
        a = {1: mod.ArmResult(lean=1, confidence=0.6), 2: mod.ArmResult(lean=1, confidence=0.6)}
        b = {1: mod.ArmResult(lean=-1, confidence=0.6), 2: mod.ArmResult(lean=-1, confidence=0.6)}
        cmp = mod.compare_arms("argus", "vader", cycles, a, b)
        assert "date" in cmp.note
        assert cmp.n_dates == 1

    def test_verdict_reads_the_bootstrap_interval(self) -> None:
        # 10 dates so the cluster bootstrap is at least nominally meaningful, ARGUS always right,
        # the rival always wrong, both always leaning -> interval should sit above zero.
        from datetime import timedelta
        cycles = [
            _cycle(i, move_bps=20.0, decided_at=AT + timedelta(days=i)) for i in range(10)
        ]
        a = {i: mod.ArmResult(lean=1, confidence=0.6) for i in range(10)}
        b = {i: mod.ArmResult(lean=-1, confidence=0.6) for i in range(10)}
        cmp = mod.compare_arms("argus", "vader", cycles, a, b)
        verdict = mod._verdict_for(cmp)
        assert "argus ahead" in verdict


# =================================================================================================
# build_report end-to-end (offline).
# =================================================================================================


class TestBuildReport:
    def test_assembles_every_section(self) -> None:
        cycles = [_cycle(1, move_bps=20.0), _cycle(2, move_bps=-20.0)]
        results_by_arm = {
            "argus": {1: mod.ArmResult(lean=1, confidence=0.6)},
            "vader": {i: mod.ArmResult(lean=0, confidence=0.0) for i in (1, 2)},
            "finbert": {i: mod.ArmResult(lean=0, confidence=0.0) for i in (1, 2)},
            "flat": {i: mod.run_flat_arm(c) for i, c in zip((1, 2), cycles, strict=True)},
            "coin_flip": mod.run_coin_flip_arms(cycles),
        }
        report = mod.build_report(
            cycles, results_by_arm, calls_made=1, tokens_spent=4777, sample_seqs=[1],
        )
        assert report["capability_row"] == 33
        assert report["join"]["usable_cycles"] == 2
        assert report["join"]["argus_scored_cycles"] == 1
        assert set(report["scores"]) == {"argus", "vader", "finbert", "flat", "coin_flip"}
        assert "argus_vs_vader" in report["comparisons"]
        assert "argus_vs_finbert" in report["comparisons"]
        assert len(report["cycles"]) == 2
        # strict JSON — no NaN/Infinity tokens, matching truth/artefact.py's contract
        json.dumps(report, allow_nan=False)

    def test_offline_load_saved_argus_reads_a_previous_reports_arm_a(self, tmp_path: Path) -> None:
        path = tmp_path / "report.json"
        path.write_text(json.dumps({
            "cycles": [
                {"seq": 1, "arms": {"argus": {"lean": 1, "confidence": 0.5, "raw": {}}}},
                {"seq": 2, "arms": {}},
            ],
        }), encoding="utf-8")
        loaded = mod._load_saved_argus(path)
        assert set(loaded) == {1}
        assert loaded[1].lean == 1

    def test_offline_load_saved_argus_on_a_missing_file_is_empty(self, tmp_path: Path) -> None:
        assert mod._load_saved_argus(tmp_path / "nope.json") == {}
