"""The deterministic ablations: exact counts, named changes, and no borrowed frames."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from argus.eval.ablations import (
    ANALYSTS,
    LOOKBACK,
    MACRO_FEEDS,
    FrameSet,
    MacroCoverage,
    _completeness,
    as_of_gate_ablation,
    selection_ablation,
)
from argus.truth.evidence import Evidence

LIVE = os.environ.get("ARGUS_LIVE_VENUE") == "1"
live_only = pytest.mark.skipif(not LIVE, reason="set ARGUS_LIVE_VENUE=1 to gather live frames")

NOW = datetime(2026, 9, 14, 15, 0, tzinfo=UTC)
DELIBERATION = Decimal("6.8")


def _ev(source: str, claim: str, *, minutes_ago: float = 30.0,
        credibility: float = 0.9) -> Evidence:
    return Evidence(
        id=f"{source}-{claim[:8]}-{minutes_ago}",
        claim=claim,
        source=source,
        available_at=NOW - timedelta(minutes=minutes_ago),
        credibility=credibility,
    )


class TestSelectionIsMeasuredNotAsserted:
    def test_a_frame_with_nothing_to_read_runs_fewer_analysts(self) -> None:
        frames = [("QUIET", (_ev("macro", "US Treasury par curve 2026-09-11: 10y 4.9%"),))]
        got = selection_ablation(frames, now=NOW, deliberation_bps=DELIBERATION)
        assert got.changes, "a macro-only frame should not need all three analysts"

    def test_the_change_names_both_panels(self) -> None:
        frames = [("QUIET", (_ev("macro", "curve steepened"),))]
        got = selection_ablation(frames, now=NOW, deliberation_bps=DELIBERATION)
        change = got.changes[0]
        assert change.frame_id == "QUIET"
        assert set(change.without_component) == set(ANALYSTS)

    def test_the_ablated_arm_always_runs_everyone(self) -> None:
        """The 'without' arm is what the desk did before the layer existed, and what most systems
        in the corpus still do — not an invented alternative."""
        frames = [("X", (_ev("news", "NVIDIA announced a datacentre agreement"),))]
        got = selection_ablation(frames, now=NOW, deliberation_bps=DELIBERATION)
        for change in got.changes:
            assert tuple(sorted(change.without_component)) == tuple(sorted(ANALYSTS))

    def test_an_identical_panel_is_not_counted_as_a_change(self) -> None:
        rich = (
            _ev("news", "NVIDIA announced an acquisition of a datacentre business"),
            _ev("social", "chatter about NVIDIA is elevated versus its 30-day average"),
            _ev("filing", "Q2 results show revenue up 12% and guidance raised"),
        )
        got = selection_ablation([("RICH", rich)], now=NOW, deliberation_bps=DELIBERATION)
        assert got.frames == 1
        assert got.share_changed in (0.0, 1.0)

    def test_it_reports_an_exact_count_with_no_p_value(self) -> None:
        frames = [(f"S{i}", (_ev("macro", "curve"),)) for i in range(7)]
        got = selection_ablation(frames, now=NOW, deliberation_bps=DELIBERATION)
        assert got.frames == 7
        assert not hasattr(got, "p_value")

    def test_no_frames_does_not_divide_by_zero(self) -> None:
        got = selection_ablation([], now=NOW, deliberation_bps=DELIBERATION)
        assert got.frames == 0 and got.share_changed == 0.0 and got.inert


class TestTheAsOfGate:
    def test_evidence_from_the_future_is_dropped(self) -> None:
        frames = [("X", (_ev("news", "tomorrow's headline", minutes_ago=-60.0),))]
        got = as_of_gate_ablation(frames, now=NOW)
        assert len(got.changes) == 1
        assert got.changes[0].with_component == 0
        assert got.changes[0].without_component == 1

    def test_a_well_behaved_feed_makes_the_gate_inert(self) -> None:
        """Inert is a finding, not a pass. It means the protection is untested in production."""
        frames = [("X", (_ev("news", "a headline from an hour ago"),))]
        got = as_of_gate_ablation(frames, now=NOW)
        assert got.inert

    def test_inert_is_reported_as_untested_not_useless(self) -> None:
        frames = [("X", (_ev("news", "a headline"),))]
        text = as_of_gate_ablation(frames, now=NOW).render()
        assert "Either the frames do not reach it" in text

    def test_evidence_exactly_at_the_instant_is_kept(self) -> None:
        """The boundary: `available_at <= now` is inclusive, so a headline published at the
        decision instant is visible. Excluding it would drop a catalyst for being punctual."""
        frames = [("X", (_ev("news", "right now", minutes_ago=0.0),))]
        assert as_of_gate_ablation(frames, now=NOW).inert


class TestTheModuleIsHonestAboutItsFrames:
    def test_the_lookback_matches_the_desk(self) -> None:
        assert timedelta(days=14) == LOOKBACK

    def test_the_analyst_list_matches_the_desk(self) -> None:
        """A selection ablation over a different analyst set would measure a different system."""
        from argus.agents.selection import SOURCES

        assert set(ANALYSTS) == set(SOURCES)

    def test_the_risk_appetite_index_no_longer_triggers_sentiment(self) -> None:
        """The defect this harness found: one market-wide index on the `social` channel made a
        demoted analyst run on every symbol, and charged deliberation for it each time."""
        from argus.market.macro import FearGreed, fear_greed_evidence

        item = fear_greed_evidence(
            FearGreed(as_of=NOW, value=61, classification="Greed"), as_of=NOW
        )[0]
        got = selection_ablation([("X", (item,))], now=NOW, deliberation_bps=DELIBERATION)
        with_panel = got.changes[0].with_component if got.changes else ()
        assert "sentiment" not in with_panel


class TestNothingLeavesTheExperimentSilently:
    """Every exclusion has to arrive with the result, not be inferred from a smaller number."""

    def test_a_missing_macro_feed_is_named_with_its_error(self) -> None:
        coverage = MacroCoverage(("treasury_curve",), (("fear_greed", "URLError: timed out"),))
        assert coverage.complete is False
        payload = coverage.as_dict()
        assert payload["failed"] == [{"feed": "fear_greed", "error": "URLError: timed out"}]
        assert payload["expected"] == list(MACRO_FEEDS)

    def test_complete_coverage_says_so_without_inventing_a_feed(self) -> None:
        assert MacroCoverage(MACRO_FEEDS, ()).complete is True

    def test_the_completeness_note_warns_when_the_run_is_thinner_than_claimed(self) -> None:
        """The old note asserted all three macro items were present and could not notice a gap."""
        thin = FrameSet(
            (("X", ()),), MacroCoverage((), (("implied_volatility", "VolatilityError: no level"),)),
            (), (),
        )
        note = _completeness(thin)
        assert "This run is thinner than that" in note
        assert "implied_volatility" in note and "VolatilityError" in note

    def test_the_completeness_note_states_full_coverage_when_it_holds(self) -> None:
        full = FrameSet((("X", ()),), MacroCoverage(MACRO_FEEDS, ()), (), ())
        note = _completeness(full)
        assert "All three instrument-independent macro items" in note
        assert "thinner" not in note

    def test_a_dropped_symbol_carries_its_reason(self) -> None:
        got = FrameSet((), MacroCoverage(MACRO_FEEDS, ()), (("NVDAUSDT", "HTTPError: 503"),), ())
        assert got.as_dict()["dropped"] == [{"symbol": "NVDAUSDT", "reason": "HTTPError: 503"}]

    def test_a_symbol_with_no_evidence_is_not_reported_as_a_failed_one(self) -> None:
        """A feed that answered with nothing and a feed that raised are different observations."""
        got = FrameSet((), MacroCoverage(MACRO_FEEDS, ()), (), ("TSLAUSDT",))
        payload = got.as_dict()
        assert payload["empty"] == ["TSLAUSDT"]
        assert payload["dropped"] == []


@live_only
class TestAgainstLiveFrames:
    def test_it_gathers_a_usable_universe(self) -> None:
        from argus.eval.ablations import frames as live_frames

        got = live_frames()
        assert len(got) >= 6, "fewer than six symbols returned evidence"
        for _symbol, evidence in got.rows:
            assert evidence

    def test_a_live_run_reports_which_macro_feeds_answered(self) -> None:
        from argus.eval.ablations import frames as live_frames

        got = live_frames()
        named = set(got.macro.obtained) | {feed for feed, _ in got.macro.failed}
        assert named == set(MACRO_FEEDS), "a macro feed neither answered nor reported a failure"

    def test_the_artefact_records_what_the_frames_do_not_contain(self) -> None:
        """A lower-bound frame reported as a complete one would overstate every result."""
        from argus.eval.ablations import run

        report = run()
        assert "lower bound" in report["frame_completeness"]
        assert report["macro"]["expected"] == list(MACRO_FEEDS)
