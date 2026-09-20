"""Episode-grouping tests.

The property under test is that a repeated view is counted once. The second property, equally
important, is that the grouping never quietly merges two genuinely different judgements — a rule
that collapses too much flatters the log just as surely as one that collapses too little, only in
the other direction.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from argus.eval.episodes import (
    EPISODE_WINDOW,
    group,
    invalidation_similarity,
    summarise,
)
from argus.paper.ledger import Entry

START = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)


def _entry(
    seq: int,
    *,
    symbol: str = "NVDAUSDT",
    verdict: str = "no_trade",
    minutes: int = 0,
    phase: str = "weekend",
    invalidation: tuple[str, ...] = ("a catalyst emerges before the open",),
    net_pnl: str | None = None,
    confidence: float = 0.9,
) -> Entry:
    settled = net_pnl is not None
    return Entry(
        seq=seq,
        decided_at=(START + timedelta(minutes=minutes)).isoformat(),
        symbol=symbol,
        verdict=verdict,
        side="flat" if verdict == "no_trade" else "BUY",
        quantity="0" if verdict == "no_trade" else "1",
        entry_price="100",
        stated_confidence=confidence,
        thesis="fixture",
        invalidation=invalidation,
        market_state_hash="m" * 16,
        approved_intent_hash="a" * 16,
        session_phase=phase,
        hours_to_discovery=46.0,
        entry_cost_bps="6",
        prev_hash="0" * 16,
        settled_at=(START + timedelta(days=1)).isoformat() if settled else None,
        net_pnl=net_pnl,
        gross_pnl=net_pnl,
        direction_correct=(Decimal(net_pnl) > 0) if settled and net_pnl else None,
    )


class TestARepeatedViewIsCountedOnce:
    def test_consecutive_identical_refusals_collapse(self) -> None:
        episodes = group([_entry(i, minutes=90 * i) for i in range(1, 6)])
        assert len(episodes) == 1
        assert episodes[0].restatements == 4

    def test_the_representative_is_the_first_decision(self) -> None:
        """The episode is dated from when the view was taken, not when it was last repeated."""
        episodes = group([_entry(i, minutes=90 * i) for i in range(1, 4)])
        assert episodes[0].representative.seq == 1

    def test_the_inflation_ratio_is_reported(self) -> None:
        summary = summarise([_entry(i, minutes=90 * i) for i in range(1, 7)])
        assert summary.episode_count == 1
        assert summary.inflation == 6.0
        assert summary.restatements == 5

    def test_both_numbers_are_shown_so_the_correction_is_visible(self) -> None:
        payload = summarise([_entry(i, minutes=90 * i) for i in range(1, 5)]).as_dict()
        assert payload["decisions"] == 4
        assert payload["episodes"] == 1
        assert "overstates" in payload["note"]


class TestDifferentJudgementsAreNotMerged:
    def test_a_verdict_change_starts_a_new_episode(self) -> None:
        rows = [
            _entry(1, minutes=0),
            _entry(2, minutes=90),
            _entry(3, minutes=180, verdict="open_long"),
        ]
        assert len(group(rows)) == 2

    def test_a_session_change_starts_a_new_episode(self) -> None:
        """A view carried from the weekend into the open is about a different market."""
        rows = [_entry(1, minutes=0), _entry(2, minutes=90, phase="rth")]
        assert len(group(rows)) == 2

    def test_a_long_gap_starts_a_new_episode(self) -> None:
        beyond = int(EPISODE_WINDOW.total_seconds() // 60) + 60
        rows = [_entry(1, minutes=0), _entry(2, minutes=beyond)]
        assert len(group(rows)) == 2

    def test_different_symbols_never_share_an_episode(self) -> None:
        rows = [_entry(1), _entry(2, symbol="TSLAUSDT", minutes=10)]
        assert len(group(rows)) == 2

    def test_interleaved_symbols_do_not_break_each_other(self) -> None:
        """Decisions on different symbols interleave in the log; that must not split episodes."""
        rows = [
            _entry(1, symbol="NVDAUSDT", minutes=0),
            _entry(2, symbol="TSLAUSDT", minutes=10),
            _entry(3, symbol="NVDAUSDT", minutes=90),
            _entry(4, symbol="TSLAUSDT", minutes=100),
        ]
        episodes = group(rows)
        assert len(episodes) == 2
        assert all(e.restatements == 1 for e in episodes)


class TestTheOutcomeIsTheMeanNotAChosenMember:
    def test_the_episode_carries_the_mean_pnl(self) -> None:
        """clawock records that picking a member moves their headline across the 50% line."""
        rows = [
            _entry(1, verdict="open_long", minutes=0, net_pnl="100"),
            _entry(2, verdict="open_long", minutes=90, net_pnl="-40"),
            _entry(3, verdict="open_long", minutes=180, net_pnl="60"),
        ]
        episode = group(rows)[0]
        assert episode.mean_net_pnl == pytest.approx(Decimal("40"))

    def test_an_unsettled_episode_has_no_mean(self) -> None:
        assert group([_entry(1), _entry(2, minutes=90)])[0].mean_net_pnl is None

    def test_confidence_is_averaged_across_the_restatements(self) -> None:
        rows = [_entry(1, confidence=0.6), _entry(2, minutes=90, confidence=0.8)]
        assert group(rows)[0].mean_confidence == pytest.approx(0.7)


class TestInvalidationSimilarityIsReportedNotEnforced:
    """The measurement that changed the design, kept as a regression guard."""

    def test_rewording_the_same_condition_does_not_split_an_episode(self) -> None:
        """The real failure: the model rewords its invalidation every cycle."""
        rows = [
            _entry(1, invalidation=("anchor market reopens before the window closes",)),
            _entry(2, minutes=90, invalidation=("a genuine price-discovery event occurs",)),
        ]
        assert len(group(rows)) == 1, "a reworded condition is not a change of mind"

    def test_the_similarity_is_attached_for_a_reader_to_weigh(self) -> None:
        rows = [
            _entry(1, invalidation=("anchor market reopens before the window closes",)),
            _entry(2, minutes=90, invalidation=("a genuine price-discovery event occurs",)),
        ]
        agreement = group(rows)[0].invalidation_agreement
        assert agreement is not None and agreement < 0.5

    def test_a_single_decision_episode_has_no_agreement_to_report(self) -> None:
        assert group([_entry(1)])[0].invalidation_agreement is None

    def test_identical_text_scores_one(self) -> None:
        phrase = ("a catalyst emerges before the open",)
        assert invalidation_similarity(phrase, phrase) == pytest.approx(1.0)

    def test_unrelated_text_scores_near_zero(self) -> None:
        assert invalidation_similarity(("earnings beat raises guidance",),
                                       ("liquidity dries up overnight",)) == 0.0

    def test_two_empty_invalidations_are_identical_not_undefined(self) -> None:
        assert invalidation_similarity((), ()) == 1.0

    def test_one_empty_against_one_populated_shares_nothing(self) -> None:
        assert invalidation_similarity((), ("a catalyst emerges",)) == 0.0


class TestTheRealLedgerShape:
    def test_an_empty_log_produces_no_episodes(self) -> None:
        summary = summarise([])
        assert summary.episode_count == 0
        assert summary.inflation == 0.0

    def test_ordering_is_by_the_first_decision_of_each_episode(self) -> None:
        rows = [
            _entry(1, symbol="NVDAUSDT"),
            _entry(2, symbol="TSLAUSDT", minutes=5),
            _entry(3, symbol="AAPLUSDT", minutes=10),
        ]
        assert [e.representative.seq for e in group(rows)] == [1, 2, 3]
