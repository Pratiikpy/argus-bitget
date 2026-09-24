"""Analyst suite tests.

Offline tests run always. Live tests (ARGUS_LIVE_LLM=1) exercise all five sub-theme analysts
against the real Qwen endpoint.
"""

from __future__ import annotations

import os
from datetime import datetime
from decimal import Decimal

import pytest

from argus.agents.analysts import (
    AnalystView,
    CrossAssetAnalyst,
    EarningsAnalyst,
    EventAnalyst,
    Evidence,
    SentimentAnalyst,
    SourceIndependenceGraph,
    _parse_view,
)
from argus.llm.qwen import QwenClient, TokenBudget
from argus.truth.clocks import ET, DualClock

LIVE = os.environ.get("ARGUS_LIVE_LLM") == "1" and bool(os.environ.get("BITGET_QWEN_API_KEY"))
live_only = pytest.mark.skipif(not LIVE, reason="set ARGUS_LIVE_LLM=1")

SUNDAY = datetime(2026, 3, 8, 3, 0, tzinfo=ET)


def _view(signal: str, mag: int, conf: float, sources: tuple[str, ...] = ("e1",)) -> AnalystView:
    return AnalystView(
        analyst="t", signal=signal, magnitude_bps=mag, confidence=conf,
        reasoning="r", counter_case="c", source_ids=sources,
    )


class TestParsing:
    def test_directional_call_without_magnitude_is_downgraded(self) -> None:
        """A direction with no size is not a signal, and we do not invent one."""
        got = _parse_view("x", {"signal": "bullish", "magnitude_bps": 0, "confidence": 0.9})
        assert got.signal == "neutral"

    def test_unknown_signal_becomes_insufficient_evidence(self) -> None:
        got = _parse_view("x", {"signal": "very bullish!!", "confidence": 0.9})
        assert got.signal == "insufficient_evidence"

    def test_confidence_is_clamped(self) -> None:
        assert _parse_view("x", {"signal": "neutral", "confidence": 87}).confidence == 1.0
        assert _parse_view("x", {"signal": "neutral", "confidence": -3}).confidence == 0.0

    def test_garbage_magnitude_does_not_crash(self) -> None:
        got = _parse_view("x", {"signal": "neutral", "magnitude_bps": "lots", "confidence": 0.5})
        assert got.magnitude_bps == 0


class TestFeeAwareness:
    """On this venue the fee is larger than most effects. An analyst claiming less than 12bps has
    not found an opportunity."""

    def test_a_move_smaller_than_the_round_trip_is_not_actionable(self) -> None:
        assert _view("bullish", 8, 0.9).is_actionable is False

    def test_a_move_larger_than_the_round_trip_can_be_actionable(self) -> None:
        assert _view("bullish", 40, 0.9).is_actionable is True

    def test_low_confidence_is_not_actionable_however_large(self) -> None:
        assert _view("bullish", 400, 0.2).is_actionable is False

    def test_neutral_is_never_actionable(self) -> None:
        assert _view("neutral", 0, 0.99).is_actionable is False


class TestSourceIndependenceGraph:
    """Five agents agreeing after reading one article is one piece of evidence, not five."""

    def test_unanimity_on_one_source_is_discounted_hard(self) -> None:
        g = SourceIndependenceGraph()
        for _ in range(5):
            g.add(_view("bullish", 50, 0.9, sources=("reuters-1",)))
        signal, confidence = g.consensus()
        assert signal == "bullish"
        assert g.independence_ratio == pytest.approx(0.2)
        assert confidence == pytest.approx(0.18, abs=0.01)  # 0.9 * 0.2

    def test_independent_sources_are_not_discounted(self) -> None:
        g = SourceIndependenceGraph()
        for i in range(5):
            g.add(_view("bullish", 50, 0.9, sources=(f"src-{i}",)))
        _, confidence = g.consensus()
        assert g.independence_ratio == pytest.approx(1.0)
        assert confidence == pytest.approx(0.9)

    def test_a_diverse_panel_beats_an_echo_chamber(self) -> None:
        echo = SourceIndependenceGraph()
        diverse = SourceIndependenceGraph()
        for i in range(4):
            echo.add(_view("bullish", 50, 0.95, sources=("one-article",)))
            diverse.add(_view("bullish", 50, 0.70, sources=(f"s{i}",)))
        assert diverse.consensus()[1] > echo.consensus()[1]

    def test_empty_graph_is_insufficient_evidence(self) -> None:
        assert SourceIndependenceGraph().consensus() == ("insufficient_evidence", 0.0)

    def test_majority_signal_wins(self) -> None:
        g = SourceIndependenceGraph()
        g.add(_view("bullish", 50, 0.8, ("a",)))
        g.add(_view("bearish", 50, 0.8, ("b",)))
        g.add(_view("bearish", 50, 0.8, ("c",)))
        assert g.consensus()[0] == "bearish"


@live_only
class TestLiveAnalysts:
    """All five sub-theme analysts against the real model."""

    def _client(self) -> QwenClient:
        return QwenClient(budget=TokenBudget(limit=120_000))

    def test_event_analyst_builds_a_transmission_chain(self) -> None:
        a = EventAnalyst(self._client())
        got = a.analyse(
            "rNVDA",
            DualClock().state(SUNDAY, nav_age_seconds=40_000),
            [Evidence("e1", "SEC 8-K: FY guidance revised down 7%.", "sec-edgar",
                      SUNDAY, credibility=0.99)],
        )
        assert got.signal in ("bullish", "bearish", "neutral", "insufficient_evidence")
        assert got.reasoning

    def test_sentiment_analyst_discounts_an_unsourced_narrative(self) -> None:
        """The manipulation defence: loud and unsourced must not become a signal."""
        a = SentimentAnalyst(self._client())
        got = a.analyse(
            "rNVDA",
            [Evidence("s1", "Anonymous posts claim a 20% guidance cut. No source given.",
                      "social", SUNDAY, credibility=0.15)],
            social_volume_z=4.2,
        )
        assert got.signal in ("neutral", "insufficient_evidence") or got.confidence < 0.6, (
            f"unsourced viral claim produced {got.signal} at {got.confidence}: {got.reasoning}"
        )

    def test_earnings_analyst_sees_through_a_headline_beat(self) -> None:
        """EPS beat with cut guidance and evasive Q&A is a bearish print."""
        a = EarningsAnalyst(self._client())
        got = a.analyse("rNVDA", [
            Evidence("q1", "EPS +12% vs consensus +8%.", "filing", SUNDAY),
            Evidence("q2", "Revenue +4%, below consensus +6%.", "filing", SUNDAY),
            Evidence("q3", "FY guidance cut 7%.", "filing", SUNDAY),
            Evidence("q4", "Gross margin -250bps.", "filing", SUNDAY),
            Evidence("q5", "Management declined three questions on datacentre orders.",
                     "transcript", SUNDAY),
        ])
        assert got.signal in ("bearish", "neutral"), (
            f"headline beat with deteriorating fundamentals read as {got.signal}: {got.reasoning}"
        )

    def test_cross_asset_analyst_accepts_an_empty_hedge_menu(self) -> None:
        """For ~65 hours a week nothing is placeable. Declining must be a correct answer."""
        a = CrossAssetAnalyst(self._client())
        got = a.analyse(
            "rNVDA",
            DualClock().state(SUNDAY, nav_age_seconds=40_000),
            [],
            position_notional=Decimal("24000"),
        )
        assert got.reasoning


class TestGradableOutputs:
    """The graders are only useful if the analysts actually feed them."""

    def test_a_view_keeps_the_full_response(self) -> None:
        """Discarding it would leave the causality and earnings graders with nothing to read."""
        got = _parse_view("event", {
            "signal": "bearish", "magnitude_bps": 40, "confidence": 0.7,
            "reasoning": "r", "chain": ["a", "b"], "surprises": {"guidance": -0.7},
        })
        assert got.raw["chain"] == ["a", "b"]
        assert got.raw["surprises"]["guidance"] == -0.7

    def test_an_event_view_converts_to_a_gradable_chain(self) -> None:
        from argus.agents.causality import LinkGrade, chain_from_response

        view = _parse_view("event", {
            "signal": "bearish", "magnitude_bps": 60, "confidence": 0.7, "reasoning": "r",
            "chain": ["guidance cut", "forward earnings fall", "multiple compresses"],
            "chain_falsifiers": ["guidance reaffirmed", "consensus unchanged", "multiple holds"],
        })
        chain = chain_from_response("8-K", SUNDAY, view.raw)
        assert len(chain.links) == 3
        assert all(g is LinkGrade.UNGRADED for g in chain.grades)
        chain.grade(0, LinkGrade.CORRECT)
        assert chain.links_correct == 1

    def test_an_earnings_view_converts_to_seven_surprises(self) -> None:
        from argus.agents.earnings import from_response

        view = _parse_view("earnings", {
            "signal": "bearish", "magnitude_bps": 50, "confidence": 0.6, "reasoning": "r",
            "surprises": {
                "reported": 0.8, "consensus": 0.6, "guidance": -0.7,
                "narrative": -0.4, "qa": -0.6,
            },
        })
        read = from_response("rNVDA", view.raw)
        assert read.surprises.is_contradictory is True
        assert read.direction == "bearish"
        assert read.dominant == "guidance"

    def test_a_view_without_structure_still_converts_safely(self) -> None:
        """A model that omits the detail produces an empty chain, not a crash or a fake one."""
        from argus.agents.causality import chain_from_response
        from argus.agents.earnings import from_response

        view = _parse_view("event", {"signal": "neutral", "confidence": 0.3, "reasoning": "r"})
        assert chain_from_response("e", SUNDAY, view.raw).links == []
        assert from_response("x", view.raw).direction == "neutral"


class TestTheChainIsGradable:
    """All 138 recorded chains named the price line as their event and carried no falsifier, so
    no link was ever graded (the rival review, 2026-09-24). Both halves are pinned here."""

    def test_the_prompt_asks_for_a_falsifier_per_link(self) -> None:
        from argus.agents.analysts import EventAnalyst

        assert '"chain_falsifiers"' in EventAnalyst.role
        assert "same length" in EventAnalyst.role

    def test_the_event_is_the_filing_or_headline_not_the_price(self) -> None:
        from datetime import UTC, datetime

        from argus.agents.analysts import chain_event
        from argus.truth.evidence import Evidence

        at = datetime(2026, 9, 24, tzinfo=UTC)
        price = Evidence(id="mkt-NVDAUSDT", claim="NVDAUSDT last 180", source="news",
                         available_at=at)
        vix = Evidence(id="vix-1", claim="VIX 14.2", source="macro", available_at=at)
        headline = Evidence(id="rss-1", claim="Nvidia wins a contract", source="news",
                            available_at=at)
        filing = Evidence(id="sec-1", claim="NVDA filed 8-K item 2.02", source="filing",
                          available_at=at)
        assert chain_event("NVDAUSDT", [price, vix, headline, filing]) == filing.claim
        assert chain_event("NVDAUSDT", [price, vix, headline]) == headline.claim
        assert chain_event("NVDAUSDT", [price, vix]) == vix.claim
        assert chain_event("NVDAUSDT", [price]) == price.claim
        assert chain_event("NVDAUSDT", []) == "NVDAUSDT"
