"""Build-status tests — the submission's own claims, checked at runtime.

The three tracks name eighteen sub-themes between them, and the submission claims all eighteen. A
coverage table written in prose is a claim; this resolves each one's implementing symbol by import
and each one's test file on disk, so a rename or a deleted test turns the claim red here rather
than in front of a judge.
"""

from __future__ import annotations

from argus.status import MODULES, SUBTHEMES, check, subtheme_coverage


class TestSubthemeCoverage:
    def test_all_eighteen_resolve(self) -> None:
        coverage = subtheme_coverage()
        assert coverage["uncovered"] == {}, coverage["uncovered"]
        assert coverage["covered"] == "18/18"

    def test_six_per_track(self) -> None:
        """The tracks are equal in the handbook and must be equal here."""
        assert subtheme_coverage()["by_track"] == {"track_1": 6, "track_2": 6, "track_3": 6}

    def test_the_map_itself_is_complete(self) -> None:
        assert len(SUBTHEMES) == 18
        assert {t for t, _, _ in SUBTHEMES.values()} == {1, 2, 3}

    def test_a_broken_mapping_is_caught_not_ignored(self) -> None:
        """The check must be capable of failing, or it is decoration."""
        original = SUBTHEMES["earnings"]
        SUBTHEMES["earnings"] = (2, "argus.agents.earnings:no_such_function", "test_earnings.py")
        try:
            assert "earnings" in subtheme_coverage()["uncovered"]
        finally:
            SUBTHEMES["earnings"] = original
        assert subtheme_coverage()["uncovered"] == {}

    def test_a_missing_test_file_is_caught(self) -> None:
        original = SUBTHEMES["earnings"]
        SUBTHEMES["earnings"] = (2, "argus.agents.earnings:decompose", "test_nothing.py")
        try:
            assert "test file" in subtheme_coverage()["uncovered"]["earnings"]
        finally:
            SUBTHEMES["earnings"] = original


class TestBuildStatus:
    def test_every_module_imports(self) -> None:
        status = check()
        assert status["modules_broken"] == {}, status["modules_broken"]
        assert status["modules_importable"] == f"{len(MODULES)}/{len(MODULES)}"

    def test_no_artefact_is_missing(self) -> None:
        """Every artefact the status claims is on disk really is."""
        assert check()["artefacts_missing"] == []

    def test_the_blocker_is_stated_when_credentials_are_absent(self) -> None:
        """An untested path must announce itself rather than read as passing."""
        status = check()
        if not status["credentials"]["bitget_trading"]:
            assert status["blocked"], "a missing credential must surface as a named blocker"
