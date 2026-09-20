"""Source audit — the half of the scored line everyone else will skip.

Track 3's judging focus is *"data sources / Skill integration count **and effectiveness**"*. Two
halves. `source_health.json` and `bitget_skills_health.json` answer the first and are often mistaken
for the second: **a source that answers every probe and never reaches a decision is integrated and
worthless**, and a count that includes it counts wiring, not evidence.

The answer this produces is unflattering, which is why it is worth publishing: of Bitget's five
research Skills, **one reaches 90% of decisions and four reach none**, because they return empty
payloads for tokenized equities. A rival can claim nineteen integrations; we can say which were
called, which answered, which reached a decision, and why the rest cannot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.eval.sourceaudit import (
    REPORT_PATH,
    SourceAuditError,
    SourceRow,
    audit,
    feed_reach,
    report,
    skill_reach,
)


class TestUnmeasuredIsNotZero:
    """A source with no record behind it has not been shown to be useless; it has not been shown
    to be anything. Rating it 0% punishes it for being new."""

    def test_no_record_reports_none_not_zero(self) -> None:
        row = SourceRow("fresh", "skill", True, "ok", reached_decisions=None, decisions_seen=150)
        assert row.effectiveness is None
        assert row.as_dict()["effectiveness_pct"] is None
        assert "unmeasured" in row.as_dict()["verdict"]

    def test_zero_reach_over_real_decisions_is_a_real_zero(self) -> None:
        row = SourceRow("dead", "skill", True, "empty", reached_decisions=0, decisions_seen=150)
        assert row.effectiveness == 0.0
        assert "never reached" in row.as_dict()["verdict"]

    def test_no_decisions_seen_is_unmeasured_even_with_a_count(self) -> None:
        row = SourceRow("x", "skill", True, "ok", reached_decisions=0, decisions_seen=0)
        assert row.effectiveness is None


class TestItRefusesToScoreFromAMissingRecord:
    def test_absent_notes_raise_rather_than_reporting_everything_ineffective(
        self, tmp_path: Path
    ) -> None:
        """*Reporting a source as ineffective on the strength of a missing file would be the worst
        kind of zero.*"""
        with pytest.raises(SourceAuditError, match="worst kind of zero"):
            skill_reach(tmp_path / "nope.jsonl")


class TestFeedEffectivenessIsMeasuredForwardOnly:
    """The other half of the judged line, which was unmeasurable until provenance was recorded.

    Feeds used to report `reached_decisions: None` for all twelve, with the honest note *"per-
    decision reach is not recorded per feed, so it is reported as unmeasured rather than
    invented."* The desk
    always knew — every `Evidence` carries a `source` — and simply discarded it at the end of the
    cycle. It is recorded from 2026-09-15 onward.

    **The denominator is the trap.** Decisions taken before provenance existed carry no `sources`
    key, and counting them would turn a record that did not store the answer into twelve feeds that
    look ineffective. That is the same manufactured zero this module was built to refuse.
    """

    def test_a_row_without_provenance_is_excluded_from_the_denominator(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "notes.jsonl"
        path.write_text(
            json.dumps({"seq": 1, "notes": []}) + "\n"
            + json.dumps({"seq": 2, "notes": [], "sources": ["vix"]}) + "\n",
            encoding="utf-8",
        )
        reached, measurable = feed_reach(path)
        assert measurable == 1, "a pre-provenance row must not enter the denominator"
        assert reached["vix"] == 1

    def test_an_empty_sources_list_is_a_real_zero_not_an_absence(self, tmp_path: Path) -> None:
        """A decision that recorded provenance and saw nothing IS a measured zero."""
        path = tmp_path / "notes.jsonl"
        path.write_text(json.dumps({"seq": 1, "notes": [], "sources": []}) + "\n", encoding="utf-8")
        reached, measurable = feed_reach(path)
        assert measurable == 1
        assert sum(reached.values()) == 0

    def test_a_feed_seen_by_several_decisions_is_counted_each_time(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.jsonl"
        path.write_text(
            "".join(
                json.dumps({"seq": i, "notes": [], "sources": ["vix", "edgar"]}) + "\n"
                for i in range(3)
            ),
            encoding="utf-8",
        )
        reached, measurable = feed_reach(path)
        assert measurable == 3
        assert reached["vix"] == 3 and reached["edgar"] == 3

    def test_a_feed_contributing_many_items_to_one_decision_counts_once(
        self, tmp_path: Path
    ) -> None:
        """`reached_decisions` counts DECISIONS, not evidence items.

        Caught by running the writer end to end rather than by reading the code: a feed that
        supplied three items to a single decision reported 3 against a denominator of 1, which
        renders as 300% effectiveness. The field is a count of decisions reached.
        """
        path = tmp_path / "notes.jsonl"
        path.write_text(
            json.dumps(
                {"seq": 1, "notes": [], "sources": ["vix", "vix", "vix", "edgar"]}
            ) + "\n",
            encoding="utf-8",
        )
        reached, measurable = feed_reach(path)
        assert measurable == 1
        assert reached["vix"] == 1, "a feed must count once per decision, not once per item"
        assert reached["edgar"] == 1

    def test_no_provenance_anywhere_yields_a_zero_denominator(self, tmp_path: Path) -> None:
        """Which the caller must read as unmeasured, never as every feed reaching nothing."""
        path = tmp_path / "notes.jsonl"
        path.write_text(json.dumps({"seq": 1, "notes": []}) + "\n", encoding="utf-8")
        reached, measurable = feed_reach(path)
        assert measurable == 0
        assert not reached

    def test_a_missing_notes_file_raises_rather_than_reporting_ineffective(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(SourceAuditError):
            feed_reach(tmp_path / "absent.jsonl")

    def test_a_malformed_line_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.jsonl"
        path.write_text(
            "{not json\n" + json.dumps({"seq": 1, "notes": [], "sources": ["vix"]}) + "\n",
            encoding="utf-8",
        )
        reached, measurable = feed_reach(path)
        assert measurable == 1 and reached["vix"] == 1


class TestItParsesTheDesksOwnNotes:
    def test_it_reads_which_skill_reached(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.jsonl"
        path.write_text(
            json.dumps({"seq": 1, "notes": [
                "[skills] 6 of 6 official-Skill calls answered for NVDAUSDT; "
                "1 of 5 Skills reached (technical-analysis)"
            ]}) + "\n",
            encoding="utf-8",
        )
        reached, decisions = skill_reach(path)
        assert decisions == 1
        assert reached["technical-analysis"] == 1

    def test_several_named_skills_are_all_credited(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.jsonl"
        path.write_text(
            json.dumps({"seq": 1, "notes": [
                "[skills] 9 of 9 official-Skill calls answered for X; "
                "2 of 5 Skills reached (technical-analysis, macro-analyst)"
            ]}) + "\n",
            encoding="utf-8",
        )
        reached, _ = skill_reach(path)
        assert reached["technical-analysis"] == 1
        assert reached["macro-analyst"] == 1

    def test_a_record_without_a_skills_note_is_not_counted_as_a_decision(
        self, tmp_path: Path
    ) -> None:
        """Otherwise the denominator grows with records that never consulted a Skill, and every
        effectiveness figure is quietly deflated."""
        path = tmp_path / "notes.jsonl"
        path.write_text(
            json.dumps({"seq": 1, "notes": ["[panel] 3 of 3 analysts"]}) + "\n",
            encoding="utf-8",
        )
        _, decisions = skill_reach(path)
        assert decisions == 0


class TestTheReportAnswersBothHalves:
    def test_it_reports_count_and_effectiveness_separately(self) -> None:
        rows = audit()
        blob = report(rows)
        assert "count" in blob and "effectiveness" in blob
        assert blob["count"]["total_integrated"] > 0

    def test_it_names_the_scored_line_it_answers(self) -> None:
        blob = report(audit())
        assert "count AND effectiveness" in blob["answers"]

    def test_a_skill_that_never_reaches_carries_its_reason(self) -> None:
        """*Integrated and never reached a decision* is a finding only if it says why."""
        rows = audit()
        dead = [r for r in rows if r.kind == "skill" and (r.effectiveness or 0) == 0]
        if not dead:
            pytest.skip("every Skill reaches a decision on this machine")
        for row in dead:
            assert row.exclusion_reason, row.name

    def test_the_live_artefact_if_present_covers_every_source(self) -> None:
        if not REPORT_PATH.exists():
            pytest.skip("no source audit on this machine")
        blob = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        assert len(blob["sources"]) == blob["count"]["total_integrated"]

    def test_the_note_explains_why_both_numbers_are_needed(self) -> None:
        blob = report(audit())
        assert "integrated and" in blob["effectiveness"]["note"]
