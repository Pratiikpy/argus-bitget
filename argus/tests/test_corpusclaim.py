"""Corpus-claim tests — the universal negatives, and the discipline they need.

A universal negative cannot be proven. *"Nothing in the corpus does X"* is the strongest sentence on
its row and the most attackable one in the project: a rival destroys it with a single repository
link, and a falsified superlative discredits the fifty careful claims beside it.

So the honest form of the claim is not the sentence, it is the search — the corpus, the query, the
result, and **the closest things the search did find**. These tests defend that discipline rather
than the sentences: a claim that reports no near-misses has probably been searched too narrowly, and
a claim reported as surviving a search that never ran is worse than an unchecked claim, because it
carries a receipt.

**It caught one of ours on its first run.** The planning documents said *"787 cloned repos"*
throughout. Counting `.git` directories gives **667**. Nobody had counted; the number had been
repeated until it read as measured.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.eval.corpusclaim import (
    CLAIMS,
    CORPUS_DIRS,
    REPORT_PATH,
    Claim,
    ClaimResult,
    CorpusError,
    repo_count,
    report,
    search,
)


class TestAClaimIsASearchNotASentence:
    def test_every_claim_carries_the_query_behind_it(self) -> None:
        for claim in CLAIMS:
            assert claim.pattern, claim.name
            assert claim.sentence, claim.name

    def test_every_claim_carries_a_near_miss_query(self) -> None:
        """Reporting the near-misses is what separates *"we looked and found nothing"* from
        *"we only looked for the thing we do"*."""
        for claim in CLAIMS:
            assert claim.near_miss, claim.name
            assert claim.near_miss != claim.pattern, claim.name

    def test_the_near_miss_query_actually_matches_common_idiom(self) -> None:
        """A near-miss pattern that matches nothing would report zero near-misses and make every
        claim look unassailable.

        The first version of this test ended `assert ... or True`, which passes whatever the
        patterns do — the second vacuous assertion I have written this session. It is replaced with
        one probe per claim, each a phrase the corpus demonstrably contains.
        """
        import re

        probes = {
            "gate_reachability": "self.checked_limits = True",
            "panel_conditioned_on_evidence": "agents = [Analyst(), Critic()]",
            "market_open_boundary": "if market_open(now):",
            "abstention_graded": "return Decision.no_trade",
        }
        for claim in CLAIMS:
            probe = probes[claim.name]
            assert re.search(claim.near_miss, probe, re.IGNORECASE), (
                f"{claim.name}: the near-miss pattern does not match {probe!r}, so it will report "
                f"no near-misses and the claim will look unassailable for the wrong reason"
            )
            assert not re.search(claim.pattern, probe, re.IGNORECASE), (
                f"{claim.name}: the CLAIM pattern matches ordinary idiom, so it would report a "
                f"false hit and refute a true claim"
            )

    def test_both_patterns_compile(self) -> None:
        import re

        for claim in CLAIMS:
            re.compile(claim.pattern, re.IGNORECASE)
            re.compile(claim.near_miss, re.IGNORECASE)


class TestTheVerdictIsPhrasedAsASearchResult:
    def test_a_surviving_claim_says_not_found_by_this_query(self) -> None:
        """Never *"does not exist"*. A grep finds the idiom it was given and misses a system that
        implements the same idea under different words."""
        got = ClaimResult(claim=CLAIMS[0], files_searched=100).as_dict()
        assert got["holds"] is True
        assert "not found by this query" in got["verdict"]
        assert "does not exist" not in got["verdict"]

    def test_a_hit_refutes_the_claim_loudly(self) -> None:
        result = ClaimResult(claim=CLAIMS[0], files_searched=100)
        result.hits.append("repos/somebody/does_it.py")
        got = result.as_dict()
        assert got["holds"] is False
        assert "REFUTED" in got["verdict"]

    def test_the_hit_count_is_reported_even_when_the_list_is_truncated(self) -> None:
        result = ClaimResult(claim=CLAIMS[0], files_searched=100)
        result.hits.extend(f"f{i}.py" for i in range(50))
        got = result.as_dict()
        assert got["hit_count"] == 50
        assert len(got["hits"]) <= 20, "the list is capped but the count is not"


class TestItRefusesToReportASearchThatDidNotRun:
    def test_an_empty_corpus_raises(self, tmp_path: Path) -> None:
        """*A claim reported as surviving a search that never ran is worse than an unchecked claim,
        because it carries a receipt.*"""
        with pytest.raises(CorpusError, match="no searchable files"):
            search(root=tmp_path)

    def test_the_report_carries_its_own_caveat(self) -> None:
        results = [ClaimResult(claim=c, files_searched=10) for c in CLAIMS]
        blob = report(results, repos=667)
        assert "does not exist" in blob["caveat"], "the caveat must name what it is NOT claiming"
        assert "different words" in blob["caveat"]

    def test_the_report_states_its_method(self) -> None:
        blob = report([ClaimResult(claim=CLAIMS[0], files_searched=1)], repos=1)
        assert "regex" in blob["method"]
        assert "holds" in blob["method"]


class TestTheCorpusSizeIsCountedNotRemembered:
    def test_repo_count_returns_a_real_number(self) -> None:
        got = repo_count()
        if got == 0:
            pytest.skip("no cloned corpus on this machine")
        assert got > 100, "the corpus should be substantial if it is present at all"

    def test_it_is_667_not_787(self) -> None:
        """The figure the planning documents repeated for weeks was **787**. Counting `.git`
        directories gives **667**. Nobody had counted."""
        got = repo_count()
        if got == 0:
            pytest.skip("no cloned corpus on this machine")
        assert got != 787, "787 was the remembered number; this must be the counted one"

    def test_every_corpus_directory_is_named(self) -> None:
        assert len(CORPUS_DIRS) >= 6
        assert "repos-rivals" in CORPUS_DIRS, "rivals' repos are part of what was searched"


class TestTheArtefactIsUsable:
    def test_the_report_round_trips_through_json(self) -> None:
        results = [ClaimResult(claim=c, files_searched=5) for c in CLAIMS]
        blob = json.loads(json.dumps(report(results, repos=667)))
        assert len(blob["claims"]) == len(CLAIMS)
        assert blob["repos_searched"] == 667

    def test_the_live_artefact_if_present_names_every_claim(self) -> None:
        if not REPORT_PATH.exists():
            pytest.skip("no corpus-claim artefact on this machine")
        blob = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        assert {c["name"] for c in blob["claims"]} == {c.name for c in CLAIMS}

    def test_the_live_artefact_reports_near_misses_for_at_least_one_claim(self) -> None:
        """If a full corpus search finds zero near-misses across every claim, the queries are too
        narrow to be evidence of anything."""
        if not REPORT_PATH.exists():
            pytest.skip("no corpus-claim artefact on this machine")
        blob = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        assert any(c["near_miss_count"] > 0 for c in blob["claims"])

    def test_a_refuted_claim_would_be_visible_in_the_artefact(self) -> None:
        """The artefact must be able to carry bad news, not only good."""
        result = ClaimResult(claim=CLAIMS[0], files_searched=10)
        result.hits.append("repos/rival/exactly_this.py")
        blob = report([result], repos=667)
        assert blob["claims"][0]["holds"] is False


class TestTheClaimsMatchWhatTheDocumentsSay:
    @pytest.mark.parametrize("claim", CLAIMS, ids=lambda c: c.name)
    def test_each_sentence_reads_as_a_universal_negative(self, claim: Claim) -> None:
        lowered = claim.sentence.lower()
        assert any(
            token in lowered for token in ("no ", "nothing", "none", "not one")
        ), claim.sentence
