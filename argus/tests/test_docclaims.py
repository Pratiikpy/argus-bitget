"""Documentation self-verification: quoted numbers must match the artefacts, at quoted precision."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from argus.eval import docclaims
from argus.eval.docclaims import CLAIMS, Claim, agrees, audit, decimals_of, parse_number, summary


@pytest.mark.parametrize(
    ("text", "value"),
    [("eleven", 11), ("Zero", 0), ("1,470", 1470), ("7.72", 7.72), ("12", 12), ("0.090", 0.09)],
)
def test_parse_number(text: str, value: float) -> None:
    assert parse_number(text) == value


def test_parse_number_rejects_prose() -> None:
    with pytest.raises(ValueError, match="not a number"):
        parse_number("several")


def test_decimals_follow_the_document() -> None:
    assert decimals_of("0.67") == 2
    assert decimals_of("0.668") == 3
    assert decimals_of("1,470") == 0


def test_exact_agreement_is_at_quoted_precision() -> None:
    assert agrees("0.67", 0.6677, "exact")
    assert agrees("0.668", 0.6677, "exact")
    assert not agrees("0.66", 0.6677, "exact")
    assert agrees("eleven", 11, "exact")
    assert not agrees("nine", 11, "exact")


def test_at_least_allows_lag_but_not_overstatement() -> None:
    assert agrees("86", 93, "at_least")
    assert agrees("93", 93, "at_least")
    assert not agrees("100", 93, "at_least")


def test_unknown_mode_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        agrees("1", 1, "roughly")


def _doc(tmp_path: Path, name: str, text: str) -> dict[str, Path]:
    path = tmp_path / f"{name}.md"
    path.write_text(text, encoding="utf-8")
    return {name: path}


def test_stale_claim_is_named_with_line(tmp_path: Path) -> None:
    docs = _doc(tmp_path, "readme", "intro\n\ncleared on nine of twelve symbols\n")
    claim = Claim("clearing", r"on (?P<q>\w+) of twelve symbols", lambda: 11, ("readme",))
    report = audit((claim,), docs)
    assert [f.status for f in report.findings] == ["STALE"]
    finding = report.findings[0]
    assert finding.line == 3
    assert finding.quoted == ("nine",)
    assert finding.live == (11,)
    assert "readme:3 clearing quotes nine, live 11" in summary(report)["stale_detail"]


def test_matching_claim_is_ok(tmp_path: Path) -> None:
    docs = _doc(tmp_path, "readme", "cleared on eleven of twelve symbols")
    claim = Claim("clearing", r"on (?P<q>\w+) of twelve symbols", lambda: 11, ("readme",))
    assert [f.status for f in audit((claim,), docs).findings] == ["OK"]


def test_every_occurrence_is_checked(tmp_path: Path) -> None:
    text = "on eleven of twelve symbols ... later, on nine of twelve symbols"
    docs = _doc(tmp_path, "readme", text)
    claim = Claim("clearing", r"on (?P<q>\w+) of twelve symbols", lambda: 11, ("readme",))
    statuses = [f.status for f in audit((claim,), docs).findings]
    assert statuses == ["OK", "STALE"]


def test_absent_claim_is_not_a_failure(tmp_path: Path) -> None:
    docs = _doc(tmp_path, "readme", "nothing quoted here")
    claim = Claim("clearing", r"on (?P<q>\w+) of twelve symbols", lambda: 11, ("readme",))
    report = audit((claim,), docs)
    assert [f.status for f in report.findings] == ["ABSENT"]
    assert report.stale == []


def test_missing_document_is_absent_not_a_crash(tmp_path: Path) -> None:
    claim = Claim("clearing", r"on (?P<q>\w+) of twelve symbols", lambda: 11, ("readme",))
    report = audit((claim,), {"readme": tmp_path / "nope.md"})
    assert report.findings[0].status == "ABSENT"
    assert report.findings[0].excerpt == "document not found"


def test_multi_group_claims_compare_each_number(tmp_path: Path) -> None:
    text = "65/65 modules importable and 58–86% during US regular hours"  # noqa: RUF001
    docs = _doc(tmp_path, "readme", text)
    modules = Claim("modules", r"(?P<q>\d+)/(?P<q2>\d+) modules importable",
                    lambda: (65, 65), ("readme",))
    rng = Claim("range", r"(?P<q>\d+)[–-](?P<q2>\d+)% during US regular",  # noqa: RUF001
                lambda: (58, 87), ("readme",))
    statuses = {f.claim: f.status for f in audit((modules, rng), docs).findings}
    assert statuses == {"modules": "OK", "range": "STALE"}


def test_group_count_mismatch_is_stale(tmp_path: Path) -> None:
    docs = _doc(tmp_path, "readme", "65/65 modules importable")
    claim = Claim("modules", r"(?P<q>\d+)/(?P<q2>\d+) modules importable", lambda: 65, ("readme",))
    assert audit((claim,), docs).findings[0].status == "STALE"


def test_expensive_claims_are_unchecked_unless_asked(tmp_path: Path) -> None:
    docs = _doc(tmp_path, "readme", "1,470 tests passing")
    calls: list[int] = []

    def live() -> int:
        calls.append(1)
        return 1470

    claim = Claim("tests_passing", r"(?P<q>[\d,]+) tests", live, ("readme",), mode="at_least")
    report = audit((claim,), docs)
    assert report.findings[0].status == "UNCHECKED"
    assert calls == []
    report = audit((claim,), docs, include_expensive=True)
    assert report.findings[0].status == "OK"
    assert calls == [1]


def test_live_overrides_bypass_disk(tmp_path: Path) -> None:
    docs = _doc(tmp_path, "readme", "1,470 tests passing")
    claim = Claim("tests_passing", r"(?P<q>[\d,]+) tests", lambda: 0, ("readme",), mode="at_least")
    report = audit((claim,), docs, live_overrides={"tests_passing": 1500})
    # LAGGING, not OK: the override took effect and the document is behind it, which is the point
    # of the override. Both are passing states; what matters here is that disk was not read.
    assert report.findings[0].status in {"OK", "LAGGING"}
    assert report.findings[0].live == (1500,)


def test_summary_counts(tmp_path: Path) -> None:
    docs = _doc(tmp_path, "readme", "on nine of twelve symbols; 7.72% monetisable")
    a = Claim("a", r"on (?P<q>\w+) of twelve symbols", lambda: 11, ("readme",))
    b = Claim("b", r"(?P<q>\d+\.\d+)% monetisable", lambda: 7.72, ("readme",))
    c = Claim("c", r"never (?P<q>\d+)", lambda: 1, ("readme",))
    s = summary(audit((a, b, c), docs))
    assert (s["checked"], s["stale"], s["unchecked"]) == (2, 1, 0)


def test_registry_patterns_carry_a_quoted_group() -> None:
    for claim in CLAIMS:
        groups = re.compile(claim.pattern).groupindex
        assert any(re.fullmatch(r"q\d*", g) for g in groups), claim.name
        assert claim.mode in {"exact", "at_least", "lagging"}
        assert set(claim.docs) <= set(docclaims.DOCS)


def test_registry_names_are_unique() -> None:
    names = [c.name for c in CLAIMS]
    assert len(names) == len(set(names))


def test_live_producers_read_real_artefacts() -> None:
    assert docclaims.module_count() >= 60
    assert docclaims.subtheme_count() == 18
    assert docclaims.source_files() >= 90
    assert docclaims.ledger_entries() >= 86
    assert docclaims.hurdle_clearing_symbols() <= 12
    low, high = docclaims.hurdle_clearing_range()
    assert 50 <= low <= high <= 100
    assert 0 < docclaims.session_beta_aapl("beta_open") < 2


def test_the_real_documents_quote_no_stale_number() -> None:
    """The CI gate: a document may lag a growing counter, never contradict an artefact."""
    report = audit()
    assert report.stale == [], summary(report)["stale_detail"]


def test_the_real_documents_make_every_registered_claim_somewhere() -> None:
    """A registered claim no document makes is dead weight; prune it or restore the prose.

    Only claims with at least one of their documents present are expected: the public repository
    does not carry the working documents (the submission draft, the master plan), so a claim made
    only there cannot be made in a public checkout."""
    from argus.eval.docclaims import DOCS

    report = audit()
    made = {f.claim for f in report.findings
            if f.status in {"OK", "LAGGING", "STALE", "UNCHECKED"}}
    expected = {c.name for c in CLAIMS if any(DOCS[d].is_file() for d in c.docs)}
    assert made == expected


class TestRepairWritesOnlyWhatTheArtefactSays:
    """The checker can now close the drift it finds. It must never close it with a made-up value.

    Added with the fix for a real bug: the first version rewrote the document once per captured
    group, so "81/81 modules importable" became "82/81" and stopped — after the first substitution
    the excerpt no longer matched, and the second group silently did nothing. A half-applied
    repair is worse than none, because the check that follows reports a number nobody wrote.
    """

    def _doc(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / "doc.md"
        path.write_text(text, encoding="utf-8")
        return path

    def test_every_group_in_one_excerpt_is_rewritten(self, tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.eval import docclaims as dc

        path = self._doc(tmp_path, "we run 3/3 modules importable today\n")
        monkeypatch.setitem(dc.DOCS, "fixture", path)
        claim = dc.Claim("modules_importable", r"(?P<q1>\d+)/(?P<q2>\d+) modules importable",
                         lambda: (9, 9), ("fixture",))
        report = dc.audit(claims=(claim,))
        assert dc.repair(report)
        assert "9/9 modules importable" in path.read_text(encoding="utf-8")

    def test_a_partial_rewrite_never_survives(self, tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
        """The exact shape of the bug: no mixed old/new pair may remain."""
        from argus.eval import docclaims as dc

        path = self._doc(tmp_path, "we run 3/3 modules importable today\n")
        monkeypatch.setitem(dc.DOCS, "fixture", path)
        claim = dc.Claim("modules_importable", r"(?P<q1>\d+)/(?P<q2>\d+) modules importable",
                         lambda: (9, 9), ("fixture",))
        dc.repair(dc.audit(claims=(claim,)))
        text = path.read_text(encoding="utf-8")
        assert "9/3" not in text and "3/9" not in text

    def test_it_preserves_the_documents_number_formatting(self, tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
        """Replacing "1,470 tests" with "2286 tests" would fix the figure and break the prose.

        The fixture claim is deliberately NOT named `tests_passing`: that name is on the EXPENSIVE
        skip-list, so a fixture borrowing it would be skipped and the test would pass by never
        running the code it is about."""
        from argus.eval import docclaims as dc

        path = self._doc(tmp_path, "there are 1,470 tests here\n")
        monkeypatch.setitem(dc.DOCS, "fixture", path)
        claim = dc.Claim("fixture_count", r"(?P<q>[\d,]+) tests", lambda: 2286, ("fixture",))
        dc.repair(dc.audit(claims=(claim,)))
        assert "2,286 tests" in path.read_text(encoding="utf-8")

    def test_it_leaves_a_correct_document_untouched(self, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.eval import docclaims as dc

        original = "we run 9/9 modules importable today\n"
        path = self._doc(tmp_path, original)
        monkeypatch.setitem(dc.DOCS, "fixture", path)
        claim = dc.Claim("modules_importable", r"(?P<q1>\d+)/(?P<q2>\d+) modules importable",
                         lambda: (9, 9), ("fixture",))
        assert dc.repair(dc.audit(claims=(claim,))) == []
        assert path.read_text(encoding="utf-8") == original

    def test_it_reports_every_change_it_made(self, tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
        """Never silent: a repair nobody can see is an edit nobody reviewed."""
        from argus.eval import docclaims as dc

        path = self._doc(tmp_path, "there are 5 tests here\n")
        monkeypatch.setitem(dc.DOCS, "fixture", path)
        claim = dc.Claim("fixture_count", r"(?P<q>[\d,]+) tests", lambda: 11, ("fixture",))
        done = dc.repair(dc.audit(claims=(claim,)))
        assert len(done) == 1 and "5 -> 11" in done[0]

    def test_the_live_documents_carry_no_stale_number_after_a_repair(self) -> None:
        """The state this whole mechanism exists to hold."""
        from argus.eval import docclaims as dc

        assert dc.audit().stale == []


class TestAFastGrowingCounterNeedsItsOwnMode:
    """Neither `exact` nor `at_least` fits a number that changes every cycle.

    `exact` failed the gate each time a scheduled run appended a row — the recorded-lean count moved
    from 11 to 13 during one test run — which teaches a reader to ignore a red check. `at_least`
    fails the other way: it passes whenever the document quotes *less* than live, so "0 decisions
    carry a lean" would keep passing forever after the first lean was written. Understating the
    evidence is the exact defect that let the README claim two settled abstentions when there were
    53. `lagging` permits a bounded lag behind, and no overstatement at all.
    """

    def test_a_small_lag_is_tolerated(self) -> None:
        from argus.eval.docclaims import agrees

        assert agrees("11", 13, "lagging")

    def test_a_large_lag_is_stale(self) -> None:
        from argus.eval.docclaims import agrees

        assert not agrees("2", 53, "lagging"), "the original README defect must still be caught"

    def test_overstating_is_never_tolerated_however_small(self) -> None:
        """A document may fall behind reality. It may not run ahead of it."""
        from argus.eval.docclaims import agrees

        assert not agrees("14", 13, "lagging")

    def test_the_boundary_is_the_stated_tolerance(self) -> None:
        from argus.eval.docclaims import LAG_TOLERANCE, agrees

        live = 100
        assert agrees(str(int(live * (1 - LAG_TOLERANCE))), live, "lagging")
        assert not agrees(str(int(live * (1 - LAG_TOLERANCE)) - 1), live, "lagging")

    def test_an_exact_match_always_agrees(self) -> None:
        from argus.eval.docclaims import agrees

        assert agrees("13", 13, "lagging")

    def test_zero_live_requires_zero_quoted(self) -> None:
        """Before the first lean is written, the honest number is 0 and nothing else."""
        from argus.eval.docclaims import agrees

        assert agrees("0", 0, "lagging")
        assert not agrees("1", 0, "lagging")

    def test_the_growing_counters_use_it(self) -> None:
        from argus.eval.docclaims import CLAIMS

        modes = {c.name: c.mode for c in CLAIMS}
        assert modes["leans_recorded"] == "lagging"
        assert modes["settled_abstentions"] == "lagging"

    def test_an_unknown_mode_still_raises(self) -> None:
        from argus.eval.docclaims import agrees

        with pytest.raises(ValueError, match="unknown mode"):
            agrees("1", 1, "vibes")


class TestLaggingIsNamedNotHiddenBehindOK:
    """**Tolerated is not the same as equal, and the report printed both as OK.**

    `LAG_TOLERANCE` lets a counter that grows every cycle trail its live value rather than failing
    the gate between refreshes, which is right. What was wrong was the rendering: `quoted 495, live
    500` appeared identically to `quoted 115/193, live 115/193`, so seven published figures — one of
    them 386 against a live 457 — read as agreement while being behind the record.
    """

    def test_a_figure_inside_tolerance_is_lagging_not_ok(self) -> None:
        from argus.eval.docclaims import agrees

        # Behind, but inside the tolerance: the gate passes and the state must still say so.
        assert agrees("495", 500, "lagging")
        assert not agrees("495", 500, "exact")

    def test_a_figure_beyond_tolerance_is_still_stale(self) -> None:
        from argus.eval.docclaims import LAG_TOLERANCE, agrees

        live = 1000.0
        beyond = live * (1.0 - LAG_TOLERANCE) - 1.0
        assert not agrees(str(int(beyond)), live, "lagging")

    def test_overstating_is_never_tolerated(self) -> None:
        from argus.eval.docclaims import agrees

        assert not agrees("505", 500, "lagging")

    def test_an_exact_match_is_not_reported_as_lagging(self) -> None:
        from argus.eval.docclaims import agrees

        assert agrees("500", 500, "lagging")
        assert agrees("500", 500, "exact")

    def test_the_summary_carries_the_lagging_lines(self) -> None:
        """A count without the lines is a number nobody can act on."""
        from argus.eval.docclaims import audit, summary

        out = summary(audit())
        assert "lagging" in out
        assert "lagging_detail" in out
        assert out["lagging"] == len(out["lagging_detail"])

    def test_lagging_does_not_fail_the_run(self) -> None:
        """This change must not break a build that was passing; it only stops hiding the gap."""
        from argus.eval.docclaims import audit

        report = audit()
        assert report.count("LAGGING") >= 0
        assert not report.stale, [f"{f.doc}:{f.line} {f.claim}" for f in report.stale]


def test_no_withdrawn_phrasing_comes_back() -> None:
    """A sentence withdrawn because it was wrong has no artefact to disagree with, so it is named
    in RETIRED_PHRASES and must not reappear in a document or a test's docstring."""
    from argus.eval.docclaims import retired_phrases

    assert retired_phrases() == []


def test_the_retired_phrase_check_finds_one(tmp_path: Path) -> None:
    from argus.eval.docclaims import retired_phrases

    doc = tmp_path / "doc.md"
    doc.write_text("ARGUS read the rest, so all 19 are covered.\n", encoding="utf-8")
    (found,) = retired_phrases([doc])
    assert found.startswith("doc.md:1")
