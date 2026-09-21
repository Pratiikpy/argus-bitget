"""Deterministic vs re-measured: the distinction "reproducible" was hiding.

`overfit_gates.json` has produced three different results on three runs — 4/4, then 6/2, then
8/0 — because it fetches a rolling 90-day window. Nothing was broken on any occasion. A judge who
re-runs a *re-measured* artefact and gets a fourth number has watched the market move; one who
re-runs a *deterministic* artefact and gets a different number has found a defect. The two must
not look alike, and until now nothing told them apart.
"""

from __future__ import annotations

from typing import Any

import pytest

from argus.eval.artefactkind import Kind, classify, run


class TestTheClassificationIsRight:
    @pytest.fixture(scope="class")
    def kinds(self) -> dict[str, Kind]:
        return {k.artefact: k for k in classify()}

    def test_a_writer_that_fetches_market_data_is_re_measured(
        self, kinds: dict[str, Kind]
    ) -> None:
        """The artefact whose drift motivated this whole module."""
        assert kinds["overfit_gates.json"].kind == "re-measured"

    @pytest.mark.parametrize(
        "artefact", ["standing.json", "ngram_bench.json", "doc_claims.json"]
    )
    def test_repo_only_writers_are_deterministic(
        self, kinds: dict[str, Kind], artefact: str
    ) -> None:
        """These read files in the repository and nothing else, so re-running must agree."""
        assert kinds[artefact].kind == "deterministic"

    def test_a_re_measured_artefact_names_what_it_reaches(
        self, kinds: dict[str, Kind]
    ) -> None:
        """"It moves" is not actionable; "it fetches history" is."""
        assert kinds["overfit_gates.json"].live_sources

    def test_every_artefact_lands_in_exactly_one_kind(self, kinds: dict[str, Kind]) -> None:
        for kind in kinds.values():
            assert kind.kind in {"deterministic", "re-measured"}
            assert kind.deterministic is (kind.kind == "deterministic")


class TestTheDetectorDoesNotTripOnItself:
    """**The first version labelled its own artefact re-measured.**

    It searched each file for the *names* in `LIVE_SOURCES` — and this module names every one of
    them, in the tuple that declares them. A detector that matches its own definition is not a
    detector, and the failure was silent: one extra row in a list of sixty-seven.
    """

    def test_its_own_artefact_is_deterministic(self) -> None:
        kinds = {k.artefact: k for k in classify()}
        assert kinds["artefact_kinds.json"].kind == "deterministic", (
            "the classifier matched its own LIVE_SOURCES declaration"
        )

    def test_a_module_that_only_mentions_a_source_in_prose_is_not_flagged(
        self, tmp_path, monkeypatch
    ) -> None:
        """Documentation naming a fetcher must not count as importing one."""
        import argus.eval.artefactkind as mod

        module = tmp_path / "prose.py"
        module.write_text(
            '"""This module deliberately does NOT use argus.market.history or fetch_range."""\n'
            'import json\n'
            'def go() -> None:\n'
            '    json.dump({}, open("prose_artefact.json", "w"))\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(mod, "SRC", tmp_path)
        kinds = {k.artefact: k for k in mod.classify()}
        assert kinds["prose_artefact.json"].kind == "deterministic"

    def test_a_module_that_really_imports_one_is_flagged(
        self, tmp_path, monkeypatch
    ) -> None:
        """And the detector must still fire on a genuine import, or it is useless."""
        import argus.eval.artefactkind as mod

        module = tmp_path / "fetcher.py"
        module.write_text(
            "import json\n"
            "from argus.market.history import fetch_range\n"
            "def go() -> None:\n"
            '    json.dump(fetch_range(), open("fetched_artefact.json", "w"))\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(mod, "SRC", tmp_path)
        kinds = {k.artefact: k for k in mod.classify()}
        assert kinds["fetched_artefact.json"].kind == "re-measured"


class TestTheReport:
    @pytest.fixture(scope="class")
    def report(self) -> dict[str, Any]:
        return run()

    def test_the_two_counts_add_up(self, report: dict[str, Any]) -> None:
        assert report["deterministic"] + report["re_measured"] == report["artefacts"]

    def test_both_kinds_exist(self, report: dict[str, Any]) -> None:
        """A classification where everything lands in one bucket has classified nothing."""
        assert report["deterministic"] > 0
        assert report["re_measured"] > 0

    def test_it_says_what_it_has_not_checked(self, report: dict[str, Any]) -> None:
        """It does not claim the deterministic ones were re-run and compared byte for byte."""
        assert "NOT CLAIMED" in report["scope_statement"]
        assert "byte for byte" in report["scope_statement"]

    def test_it_does_not_treat_re_measured_as_inferior(self, report: dict[str, Any]) -> None:
        """It is the only honest kind for a claim about a live market."""
        assert "not that" in report["scope_statement"].lower() or \
               "NOT CLAIMED: that" in report["scope_statement"]
