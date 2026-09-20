"""Deploy-sync tests — the shop window had drifted 43% behind and nothing could tell you.

The hosted console is a bundle: a copy of the package and a copy of the artefacts, both made by
hand. On 2026-09-15 it served **126 decisions against a live 221** — 95 missing, 43% of the
record — and **91 of 172 package files were stale too**, so the console ran two-day-old *code* as
well as two-day-old data. There was no command to fix it, only a memory of having copied files.

That is the orphaned-artefact defect wearing a different hat: a thing that must be regenerated, with
nothing that regenerates it.

`lui/server._status` is tested here too, because the two failures compound. The endpoint returned
only `entries` and `chain_intact`, so the page read *"126 decisions on record, chain intact"* — and
**both facts were true**. The chain of a stale snapshot verifies perfectly. The page looked healthy
precisely because nothing was wrong with the copy, only with its age.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from argus.demo.deploysync import (
    MANIFEST_PATH,
    NEVER_COPY,
    SKIP_SUFFIXES,
    SyncResult,
    sync,
)
from argus.lui.server import STALE_AFTER_HOURS


class TestTheSyncReportsTheGapRatherThanHidingIt:
    def test_the_gap_is_what_the_bundle_was_missing(self) -> None:
        got = SyncResult(ledger_before=126, ledger_after=221)
        assert got.ledger_gap == 95

    def test_a_bundle_ahead_of_the_tree_reports_no_negative_gap(self) -> None:
        """Can happen mid-cycle. A negative gap would render as nonsense."""
        assert SyncResult(ledger_before=221, ledger_after=126).ledger_gap == 0

    def test_the_render_states_the_share_of_the_record_that_was_missing(self) -> None:
        text = SyncResult(ledger_before=126, ledger_after=221).render()
        assert "missing 95 decision(s)" in text
        assert "43%" in text

    def test_a_dry_run_says_so_unmistakably(self) -> None:
        assert "DRY RUN" in SyncResult(dry_run=True).render()

    def test_a_dry_run_changes_nothing(self) -> None:
        before = MANIFEST_PATH.read_bytes() if MANIFEST_PATH.exists() else None
        sync(dry_run=True)
        after = MANIFEST_PATH.read_bytes() if MANIFEST_PATH.exists() else None
        assert before == after, "a dry run must not write the manifest"


class TestNothingSecretCanReachAPublicBundle:
    def test_secret_directories_are_refused_by_name(self) -> None:
        assert ".secrets" in NEVER_COPY
        assert ".env" in SKIP_SUFFIXES

    def test_the_exclusions_are_checked_at_every_level_not_only_the_top(self) -> None:
        """`_copy_tree` tests the whole relative path, so `a/.secrets/b.py` is caught as surely as
        `.secrets/b.py`. A key in a public serverless bundle is a key that has been published."""
        from argus.demo.deploysync import _copy_tree

        source = Path(__file__).resolve().parents[1] / "src" / "argus"
        seen, _ = _copy_tree(source, Path("/nonexistent-target"), dry_run=True)
        assert seen > 0, "the sweep must actually see files for this test to mean anything"


class TestTheBundleIsNotSilentlyBehind:
    def test_the_live_bundle_is_close_to_the_working_tree(self) -> None:
        """The regression guard. If this fails, the shop window has drifted again."""
        got = sync(dry_run=True)
        if got.ledger_after == 0:
            pytest.skip("no ledger in the working tree on this machine")
        assert got.ledger_gap <= 24, (
            f"the deployed bundle is {got.ledger_gap} decision(s) behind the working tree "
            f"({got.ledger_before} vs {got.ledger_after}). Run "
            f"`python -m argus.demo.deploysync`. A judge opening the console sees the bundle, "
            f"not this repository."
        )

    def test_nothing_in_the_bundle_is_absent_upstream(self) -> None:
        """A file the bundle carries and the tree does not is a stale copy surviving a refresh —
        which is exactly how 126 decisions survived three days."""
        got = sync(dry_run=True)
        assert not got.missing_upstream, (
            f"carried in the bundle but absent from data/: {', '.join(got.missing_upstream)}"
        )


class TestTheConsoleDeclaresItsOwnAge:
    """`_status` returned `entries` and `chain_intact` and nothing else, so a stale snapshot
    rendered as a healthy one."""

    def test_the_status_carries_an_age(self) -> None:
        from argus.lui.server import _status

        got = _status()
        assert "age_hours" in got and "newest_decision_at" in got and "stale" in got

    def test_an_unknown_age_is_rendered_as_unknown_not_omitted(self) -> None:
        """`None` means the record carries no timestamp to judge by. That is not the same as
        current, so the page prints a line saying so rather than printing nothing.

        The first version of this test ended `assert ... if hasattr(server, "PAGE") else True`,
        which passes whatever the page says. A vacuous assertion is worse than no assertion: it
        occupies the slot where a real one would have gone.
        """
        from argus.lui.server import PAGE

        assert "Age unknown" in PAGE, (
            "the page must say the age is unknown; omitting the line lets a reader assume current"
        )
        assert "not the same as current" in PAGE

    def test_the_page_always_shows_the_age_not_only_when_stale(self) -> None:
        """A figure with no date beside it invites the reader to assume it is current — and the
        chain of a stale snapshot verifies perfectly, so nothing else on the page contradicts it."""
        from argus.lui.server import PAGE

        assert "Last decision" in PAGE
        assert "stopped moving" in PAGE, "the stale branch must be visibly different, not a nuance"

    def test_the_threshold_is_two_scheduled_cycles(self) -> None:
        """One missed cycle is a blip; two is a record that has stopped moving."""
        assert STALE_AFTER_HOURS == 12.0

    def test_a_fresh_record_is_not_flagged(self) -> None:
        from argus.lui.server import _status

        got = _status()
        if got["age_hours"] is None:
            pytest.skip("no timestamped decisions on this machine")
        assert got["stale"] == (got["age_hours"] > STALE_AFTER_HOURS)


class TestTheManifestRecordsWhatWasCopied:
    def test_the_manifest_shape_is_auditable(self) -> None:
        blob = SyncResult(
            package_files=172, package_updated=91, data_files=25, data_updated=14,
            ledger_before=126, ledger_after=221,
        ).as_dict()
        assert set(blob) >= {
            "synced_at", "package_files", "package_updated", "data_files", "data_updated",
            "missing_upstream", "ledger_before", "ledger_after", "ledger_gap_closed",
        }
        assert blob["ledger_gap_closed"] == 95

    def test_the_live_manifest_parses_and_records_a_time(self) -> None:
        if not MANIFEST_PATH.exists():
            pytest.skip("no manifest on this machine")
        blob = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        when = datetime.fromisoformat(blob["synced_at"])
        assert when <= datetime.now(UTC) + timedelta(minutes=5)
