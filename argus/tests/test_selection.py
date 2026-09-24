"""Analyst selection tests.

The corpus sweep found no finance system that conditions its panel on the evidence, and none that
skips an agent because running it is not worth the cost. So the tests here are mostly about the ways
such a router goes wrong rather than about it working:

* it must never **drop evidence silently** — a skip is recorded, with the count of what went unread;
* it must not discard a **near miss** — an analyst holding evidence it can act on that fell just
  under the bar still runs, and the record marks that it was a floor rather than a choice;
* a skip must read as a statement about **cost**, not as a verdict that the evidence was worthless,
  because the decision record is the thing a judge reads;
* it must be **deterministic**, so the same evidence always produces the same panel.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from argus.agents.selection import (
    MIN_RELEVANCE,
    SOURCES,
    STALE_AFTER,
    Assessment,
    Selection,
    assess,
    select,
)
from argus.truth.evidence import Evidence

AS_OF = datetime(2026, 9, 12, 18, 0, tzinfo=UTC)
BPS = Decimal("15.2")


def _ev(
    source: str,
    claim: str = "NVDA filed 8-K — acquisition",
    *,
    credibility: float = 1.0,
    age: timedelta = timedelta(hours=1),
    ident: str = "e1",
) -> Evidence:
    return Evidence(
        id=ident, claim=claim, source=source,
        available_at=AS_OF - age, credibility=credibility,
    )


class TestItReadsTheEvidenceNotJustTheChannel:
    def test_a_material_filing_summons_the_event_analyst(self) -> None:
        assert "event" in select([_ev("sec-edgar")], as_of=AS_OF, deliberation_bps=BPS).run

    def test_a_generic_headline_does_not(self) -> None:
        """The distinction the corpus does not make: same channel, different content."""
        chatter = _ev("news", claim="Ten stocks to watch this week")
        selection = select([chatter], as_of=AS_OF, deliberation_bps=BPS)
        assert "event" not in selection.run

    def test_an_earnings_claim_summons_the_earnings_analyst(self) -> None:
        filing = _ev("filing", claim="revenue quarter on quarter +12.4%")
        assert "earnings" in select([filing], as_of=AS_OF, deliberation_bps=BPS).run

    def test_a_filing_with_no_earnings_content_does_not(self) -> None:
        filing = _ev("filing", claim="NVDA filed a change of registered agent")
        assert "earnings" not in select([filing], as_of=AS_OF, deliberation_bps=BPS).run

    def test_sentiment_is_gated_on_credibility_not_on_a_keyword_list(self) -> None:
        """Mood has no vocabulary; encoding one would be a guess about what sentiment looks like."""
        strong = _ev("social", claim="Crypto Fear & Greed index 71 (greed)", credibility=0.9)
        assert "sentiment" in select([strong], as_of=AS_OF, deliberation_bps=BPS).run

    def test_a_feed_health_line_summons_nobody(self) -> None:
        """The live cycle appends a 0.1-credibility feed-health line to every decision."""
        health = _ev("news", claim="evidence feeds: rss ok; edgar ok", credibility=0.1)
        selection = select([health], as_of=AS_OF, deliberation_bps=BPS)
        assert selection.run == ()
        assert not selection.floor_applied


class TestStaleness:
    def test_recent_evidence_counts_at_full_weight(self) -> None:
        got = assess("event", [_ev("sec-edgar")], as_of=AS_OF, cost_bps=BPS)
        assert got.relevance == pytest.approx(1.0)

    def test_old_evidence_is_discounted_rather_than_discarded(self) -> None:
        """A Form 4 filed last week is still the most recent thing that insider chose to do."""
        old = _ev("sec-edgar", age=STALE_AFTER + timedelta(hours=1))
        got = assess("event", [old], as_of=AS_OF, cost_bps=BPS)
        assert 0 < got.relevance < 1.0

    def test_enough_stale_evidence_still_clears_the_bar(self) -> None:
        old = [
            _ev("sec-edgar", age=STALE_AFTER + timedelta(hours=1), ident=f"e{i}")
            for i in range(3)
        ]
        assert assess("event", old, as_of=AS_OF, cost_bps=BPS).run


class TestNothingIsDroppedSilently:
    def test_a_skip_names_the_analyst_and_the_reason(self) -> None:
        chatter = _ev("news", claim="Ten stocks to watch")
        selection = select([chatter], as_of=AS_OF, deliberation_bps=BPS)
        skipped = [a for a in selection.assessments if not a.run]
        assert skipped and all(a.reason for a in skipped)

    def test_the_evidence_a_skipped_analyst_would_have_read_is_counted(self) -> None:
        chatter = [_ev("news", claim="Ten stocks to watch", ident=f"e{i}") for i in range(4)]
        strong = _ev("social", claim="Fear & Greed 71", credibility=0.9, ident="s1")
        selection = select([*chatter, strong], as_of=AS_OF, deliberation_bps=BPS)
        event = next(a for a in selection.assessments if a.analyst == "event")
        assert not event.run
        assert event.evidence_count == 4

    def test_the_count_is_of_everything_on_the_channel_not_only_what_matched(self) -> None:
        """A skip must state the full extent of what went unexamined."""
        mixed = [
            _ev("news", claim="Ten stocks to watch", ident="a"),
            _ev("news", claim="weather in Santa Clara", ident="b"),
        ]
        got = assess("event", mixed, as_of=AS_OF, cost_bps=BPS)
        assert got.evidence_count == 2 and got.relevance == 0.0

    def test_the_record_says_a_skip_is_about_cost_not_worth(self) -> None:
        chatter = _ev("news", claim="Ten stocks to watch")
        text = " ".join(select([chatter], as_of=AS_OF, deliberation_bps=BPS).render())
        assert "judgement about cost, not about their worth" in text

    def test_an_empty_channel_is_not_described_as_evidence_going_unexamined(self) -> None:
        """Live wording defect: "0 piece(s) of evidence went unexamined; a judgement about cost"."""
        selection = select([_ev("sec-edgar")], as_of=AS_OF, deliberation_bps=BPS)
        empty = [line for line in selection.render() if "sentiment not run" in line]
        assert empty and "0 piece(s)" not in empty[0]
        assert "judgement about cost" not in empty[0]

    def test_unread_evidence_is_totalled(self) -> None:
        chatter = [_ev("news", claim="stocks to watch", ident=f"e{i}") for i in range(3)]
        social = [_ev("social", claim="mood", credibility=0.05, ident=f"s{i}") for i in range(2)]
        selection = select([*chatter, *social], as_of=AS_OF, deliberation_bps=BPS)
        assert selection.evidence_not_read >= 2


class TestTheFloor:
    def test_a_near_miss_runs_anyway(self) -> None:
        """Evidence it can act on that fell just under the bar is not thrown away."""
        weak = _ev("news", claim="NVDA guidance cut", credibility=MIN_RELEVANCE - 0.05)
        selection = select([weak], as_of=AS_OF, deliberation_bps=BPS)
        assert selection.run == ("event",) and selection.floor_applied

    def test_evidence_with_nothing_the_analyst_reads_does_not_trigger_the_floor(self) -> None:
        """Paying for a certain shrug is the cost this module exists to avoid."""
        chatter = _ev("news", claim="Ten stocks to watch", credibility=1.0)
        selection = select([chatter], as_of=AS_OF, deliberation_bps=BPS)
        assert selection.run == () and not selection.floor_applied

    def test_the_floor_is_marked_so_it_is_not_mistaken_for_a_chosen_panel(self) -> None:
        weak = _ev("news", claim="NVDA guidance cut", credibility=MIN_RELEVANCE - 0.05)
        text = " ".join(select([weak], as_of=AS_OF, deliberation_bps=BPS).render())
        assert "a near miss is not a reason to look away" in text

    def test_a_chosen_panel_is_not_marked_as_floored(self) -> None:
        assert not select([_ev("sec-edgar")], as_of=AS_OF, deliberation_bps=BPS).floor_applied

    def test_no_evidence_at_all_leaves_the_panel_empty_rather_than_inventing_one(self) -> None:
        """With nothing to read there is no candidate, and pretending otherwise is worse."""
        selection = select([], as_of=AS_OF, deliberation_bps=BPS)
        assert selection.run == () and not selection.floor_applied

    def test_an_empty_panel_explains_itself_and_names_what_still_ran(self) -> None:
        chatter = _ev("news", claim="Ten stocks to watch")
        text = " ".join(select([chatter], as_of=AS_OF, deliberation_bps=BPS).render())
        assert "no analyst held evidence it could act on" in text
        assert "cross-asset analyst still ran" in text


class TestTheCostSide:
    def test_the_deliberation_cost_is_split_across_the_candidates(self) -> None:
        selection = select([_ev("sec-edgar")], as_of=AS_OF, deliberation_bps=Decimal("15.0"))
        assert all(a.cost_bps == Decimal("5.000") for a in selection.assessments)

    def test_skipping_saves_that_share(self) -> None:
        selection = select([_ev("sec-edgar")], as_of=AS_OF, deliberation_bps=Decimal("15.0"))
        assert selection.bps_saved == Decimal("10.000")

    def test_the_saving_is_stated_against_the_round_trip(self) -> None:
        text = " ".join(
            select([_ev("sec-edgar")], as_of=AS_OF, deliberation_bps=Decimal("15.0")).render()
        )
        assert "12bps round trip" in text

    def test_a_full_panel_saves_nothing_and_says_nothing_about_saving(self) -> None:
        full = [
            _ev("sec-edgar", ident="a"),
            _ev("social", claim="Fear & Greed 71", credibility=0.9, ident="b"),
            _ev("filing", claim="revenue +12% this quarter", ident="c"),
        ]
        selection = select(full, as_of=AS_OF, deliberation_bps=BPS)
        assert len(selection.run) == 3
        assert selection.bps_saved == Decimal("0")
        assert not any("not spent" in line for line in selection.render())


class TestItIsDeterministicAndAuditable:
    def test_the_same_evidence_gives_the_same_panel(self) -> None:
        rows = [_ev("sec-edgar"), _ev("social", claim="mood", credibility=0.9, ident="s")]
        first = select(rows, as_of=AS_OF, deliberation_bps=BPS)
        second = select(rows, as_of=AS_OF, deliberation_bps=BPS)
        assert first.as_dict() == second.as_dict()

    def test_every_analyst_is_accounted_for_whether_it_ran_or_not(self) -> None:
        selection = select([_ev("sec-edgar")], as_of=AS_OF, deliberation_bps=BPS)
        assert {a.analyst for a in selection.assessments} == {"event", "sentiment", "earnings"}

    def test_the_record_carries_the_reason_for_each(self) -> None:
        payload = select([_ev("sec-edgar")], as_of=AS_OF, deliberation_bps=BPS).as_dict()
        for row in payload["assessments"]:
            assert row["reason"] and "relevance" in row

    def test_the_source_map_matches_what_the_desk_routes_on(self) -> None:
        """Selection that disagreed with routing would skip an analyst over unseen evidence."""
        assert SOURCES["event"] == frozenset({"sec-edgar", "news", "macro"})
        assert SOURCES["sentiment"] == frozenset({"social"})
        assert SOURCES["earnings"] == frozenset({"filing", "transcript"})

    def test_an_analyst_with_no_channel_evidence_says_which_channels_it_wanted(self) -> None:
        got = assess("sentiment", [_ev("sec-edgar")], as_of=AS_OF, cost_bps=BPS)
        assert "social" in got.reason

    def test_the_threshold_is_a_named_constant_not_a_literal(self) -> None:
        weak = _ev("social", claim="mood", credibility=MIN_RELEVANCE - 0.01)
        assert not assess("sentiment", [weak], as_of=AS_OF, cost_bps=BPS).run

    def test_a_selection_of_nothing_renders_without_crashing(self) -> None:
        assert Selection(assessments=(), floor_applied=False).render()

    def test_an_assessment_serialises_every_field_a_reader_needs(self) -> None:
        got: Assessment = assess("event", [_ev("sec-edgar")], as_of=AS_OF, cost_bps=BPS)
        for key in ("analyst", "evidence_count", "relevance", "cost_bps", "run", "reason"):
            assert key in got.as_dict()
