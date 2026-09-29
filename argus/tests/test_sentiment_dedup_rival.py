"""Dedup-then-finBERT, the fair general rival for sentiment integrity
(eval/sentiment_dedup_rival.py; capability 20). Offline: a stand-in classifier, and the recorded
artefact read back."""

from __future__ import annotations

import json
from typing import Any

import pytest

from argus.eval import sentiment_dedup_rival as rival


def _always(label: str) -> Any:
    def classify(texts: list[str]) -> list[dict[str, Any]]:
        return [{"label": label, "score": 0.9} for _ in texts]
    return classify


class TestTheSharedQuestion:
    def test_rumour_votes_count_only_the_rumours_side(self) -> None:
        assert rival.rumour_votes(["positive", "neutral", "negative", "positive"], "bullish") == 2
        assert rival.rumour_votes(["positive", "negative"], "bearish") == 1

    def test_argus_moves_only_toward_the_rumour(self) -> None:
        held = {"signal": "insufficient_evidence", "confidence": 0.9}
        lean = {"signal": "bullish", "confidence": 0.3}
        assert rival.argus_moved(held, lean, "bullish")
        assert not rival.argus_moved(held, held, "bullish")
        assert not rival.argus_moved(held, {"signal": "bearish", "confidence": 0.9}, "bullish")
        assert rival.argus_moved(lean, {"signal": "bullish", "confidence": 0.6}, "bullish")

    def test_the_exact_mcnemar_is_two_sided(self) -> None:
        assert rival.exact_mcnemar(0, 0) == 1.0
        assert rival.exact_mcnemar(21, 0) == pytest.approx(2 / 2 ** 21)
        assert rival.exact_mcnemar(3, 3) == 1.0


def test_template_copies_collapse_to_one_story() -> None:
    from argus.eval.sentiment_cases import MANIPULATION
    from argus.eval.sentiment_comparison import coordinated_scenario

    labels, stories = rival.dedup_labels(_always("positive"),
                                         coordinated_scenario(MANIPULATION[0].claim))
    assert stories == 1 and labels == ["positive"]


def test_the_published_report_matches_its_recorded_inputs() -> None:
    if not rival.REPORT_PATH.exists():
        pytest.skip("sentiment_dedup_rival.json is not on this machine")
    published = json.loads(rival.REPORT_PATH.read_text(encoding="utf-8"))
    recorded = json.loads(rival.SOURCE_PATH.read_text(encoding="utf-8"))
    by_claim = {(n["symbol"], n["narrative"]): n for n in recorded["narratives"]}
    # The ARGUS arm is re-derived from the recorded answers, and must agree row by row.
    for row in published["rows"]:
        rec = by_claim[(row["symbol"], row["claim"])]
        assert row["template"]["argus_moved"] == rival.argus_moved(
            rec["argus_single"], rec["argus_coordinated"], row["direction"])
        assert row["diverse"]["argus_moved"] == rival.argus_moved(
            rec["argus_single"], rec["argus_diverse"], row["direction"])
