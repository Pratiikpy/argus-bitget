"""Theme-audit tests — the harness must be harder on us than a reader would be.

An audit that grades its own subject has one failure mode that matters: quietly rounding up. So the
tests here are mostly about the ways this file could flatter ARGUS — a probe that raises counting as
a pass, a WEAK finding averaged into a headline, a bar written after the result was known — and
about the one that actually happened, which was a probe printing a confident false sentence because
it matched the wrong note shape.
"""

from __future__ import annotations

import json

import pytest

from argus.eval.themeaudit import (
    ABSENT,
    RUNS,
    UNDEFINED,
    WEAK,
    Audit,
    Finding,
    Probe,
    _exposure_shape,
    _read_jsonl,
    audit,
    probes,
)


def _probe(key: str, status: str, track: int = 2, kind: str = "subtheme") -> Probe:
    return Probe(
        key=key, track=track, kind=kind, question="q", bar="b",
        run=lambda: Finding(status, f"{key} said {status}"),
    )


class TestAProbeThatFailsIsNeverRoundedUp:
    def test_a_raising_probe_becomes_a_finding_not_a_crash(self) -> None:
        """One bad probe must not take the whole audit down — a judge would see nothing at all."""
        def explode() -> Finding:
            raise RuntimeError("the artefact moved")

        got = Probe(
            key="x", track=2, kind="subtheme", question="q", bar="b", run=explode
        ).execute()
        assert got.finding.status == ABSENT
        assert "RuntimeError" in got.finding.evidence
        assert "the artefact moved" in got.finding.evidence

    def test_a_raising_probe_is_absent_not_weak(self) -> None:
        """WEAK means it ran and produced too little. ABSENT means nothing ran. Conflating them
        would let a broken probe read as a working capability with a thin result."""
        def explode() -> Finding:
            raise KeyError("nope")

        assert Probe(
            key="x", track=3, kind="judging", question="q", bar="b", run=explode
        ).execute().finding.status == ABSENT

    def test_only_runs_counts_toward_the_headline(self) -> None:
        got = Audit(results=tuple(
            p.execute() for p in (
                _probe("a", RUNS), _probe("b", WEAK),
                _probe("c", ABSENT), _probe("d", UNDEFINED),
            )
        ))
        assert "1/4 probes met their bar" in got.verdict

    def test_every_shortfall_is_named_in_the_verdict(self) -> None:
        """Not a count. A judge must be able to read the weaknesses without opening the JSON."""
        got = Audit(results=tuple(
            p.execute() for p in (_probe("a", RUNS), _probe("bad-one", WEAK))
        ))
        assert "bad-one (WEAK)" in got.verdict

    def test_the_weakest_are_ordered_worst_first(self) -> None:
        got = Audit(results=tuple(
            p.execute() for p in (
                _probe("w", WEAK), _probe("u", UNDEFINED), _probe("a", ABSENT), _probe("r", RUNS),
            )
        ))
        assert [r.probe.key for r in got.weakest] == ["a", "u", "w"]

    def test_a_clean_sweep_still_refuses_to_claim_superiority(self) -> None:
        """The plan's four states: a probe that runs is IMPLEMENTED, never OWNED. If this audit
        ever reads as a ranking, it has become the thing it was built to replace."""
        got = Audit(results=(_probe("a", RUNS).execute(),))
        assert "not about ranking" in got.verdict
        assert "OWNED" not in got.verdict or "never OWNED" in got.verdict

    def test_a_mixed_audit_says_owned_is_not_claimed(self) -> None:
        got = Audit(results=tuple(p.execute() for p in (_probe("a", RUNS), _probe("b", WEAK))))
        assert "never OWNED" in got.verdict


class TestTheBarIsWrittenBeforeTheResult:
    def test_every_probe_declares_a_bar_and_a_question(self) -> None:
        """A probe with no stated bar can be graded against whatever it produced."""
        for probe in probes():
            assert probe.bar.strip(), probe.key
            assert probe.question.strip(), probe.key

    def test_the_bar_is_not_the_question_restated(self) -> None:
        for probe in probes():
            assert probe.bar != probe.question, probe.key

    def test_the_question_is_the_handbook_wording(self) -> None:
        """Spot-checked against BITGET_AI_BASE_CAMP_S2_HANDBOOK_EN.md, quoted not paraphrased."""
        by_key = {p.key: p for p in probes()}
        assert by_key["event-driven"].question == (
            "How do news / announcements / macro events drive autonomous Agent trading?"
        )
        assert by_key["execution-assistance"].question == (
            "After trader decision, how does AI handle order splitting and slippage management?"
        )
        assert by_key["t2-sharpe-mdd-winrate"].question == (
            "Paper trading Sharpe, max drawdown, win rate"
        )


class TestEveryNamedThemeAndCriterionIsCovered:
    """The handbook names five sub-themes plus an Open Theme per track, and four judged criteria
    for Track 2 and four for Track 3. A missing probe is a theme nobody checked."""

    def test_track_two_has_all_six_subthemes(self) -> None:
        keys = {p.key for p in probes() if p.track == 2 and p.kind == "subtheme"}
        assert keys == {
            "event-driven", "sentiment", "earnings", "cross-asset-execution",
            "factor-discovery", "t2-open-evaluation",
        }

    def test_track_three_has_all_six_subthemes(self) -> None:
        keys = {p.key for p in probes() if p.track == 3 and p.kind == "subtheme"}
        assert keys == {
            "info-extraction", "review-self-evolution", "stress-testing",
            "personal-workbench", "execution-assistance", "portfolio-copilot",
        }

    def test_track_two_judging_focus_is_covered(self) -> None:
        """"Paper trading Sharpe, max drawdown, win rate; decision explainability; Agent
        architecture quality; risk control layer effectiveness." """
        keys = {p.key for p in probes() if p.track == 2 and p.kind == "judging"}
        # Equality, not a subset. The subset form passed with three of the four criteria the
        # docstring above quotes: "Agent architecture quality" had no probe, and the test written
        # to guard the coverage let the gap through for as long as it existed.
        assert keys == {
            "t2-sharpe-mdd-winrate", "t2-explainability", "t2-risk-control", "t2-architecture",
        }

    def test_every_judging_criterion_the_handbook_names_has_a_probe(self) -> None:
        """Four for Track 2, four for Track 3. Counted, so a fifth criterion cannot be missed the
        way the fourth was."""
        for track in (2, 3):
            assert len([p for p in probes() if p.track == track and p.kind == "judging"]) == 4

    def test_the_lui_probe_cites_fluency_not_the_phrasebook(self) -> None:
        """It used to assert len(PHRASES) >= 20 — the console's *output* templates — and describe
        that as evidence about question handling. The evidence must be the benchmark."""
        got = next(p for p in probes() if p.key == "t3-lui").execute()
        assert "LUI-BENCH" in got.finding.evidence
        assert got.finding.artefact == "data/lui_bench.json"
        assert "both directions" in got.finding.evidence

    def test_track_three_judging_focus_is_covered(self) -> None:
        """"Feature depth (data sources / Skill integration count and effectiveness), research
        quality, LUI fluency, personalized thesis." """
        keys = {p.key for p in probes() if p.track == 3 and p.kind == "judging"}
        assert keys == {
            "t3-source-depth", "t3-research-quality", "t3-lui", "t3-personal-thesis",
        }

    def test_track_one_is_deliberately_absent(self) -> None:
        """Track 1 is scored purely quantitatively from a backtest. A capability probe there would
        be answering a question nobody asked."""
        assert not [p for p in probes() if p.track == 1]

    def test_no_probe_key_is_duplicated(self) -> None:
        keys = [p.key for p in probes()]
        assert len(keys) == len(set(keys))


class TestTheOpenThemesFollowTheHandbooksOwnList:
    def test_the_track_two_open_theme_names_every_measure_it_was_given(self) -> None:
        """The handbook lists six by name. Reporting four and staying quiet about two would be the
        easiest possible way to look complete."""
        got = next(p for p in probes() if p.key == "t2-open-evaluation").execute()
        evidence = got.finding.evidence.lower()
        for measure in (
            "decision consistency", "risk-violation rate", "max drawdown",
            "stress behaviour", "human-takeover rate", "incremental value",
        ):
            assert measure in evidence, measure

    def test_decision_consistency_is_not_the_autonomy_round_trip(self) -> None:
        """Two different quantities were reported under one name.

        "Decision consistency" as a judge means it is *same state, same answer*. What was reported
        was `llm_changed_its_mind` — revising after being constrained — which read 96 of 96
        on the live log because `revise()` ran unconditionally and told the model it had been
        constrained when it had not. The probe must not conflate them again.
        """
        got = next(p for p in probes() if p.key == "t2-open-evaluation").execute()
        evidence = got.finding.evidence
        assert "decision consistency:" in evidence
        # Whatever it reports, it must never again be the count of revisions after a constraint.
        assert "revised its own answer" not in evidence
        assert "cache disabled" in evidence or "UNDEFINED" in evidence

    def test_an_unmeasurable_one_is_undefined_with_its_reason_not_omitted(self) -> None:
        got = next(p for p in probes() if p.key == "t2-open-evaluation").execute()
        assert "UNDEFINED" in got.finding.evidence
        assert "zero positions have settled" in got.finding.evidence

    def test_the_portfolio_copilot_bar_demands_one_answer_not_seven(self) -> None:
        """The Open Theme asks how a trade changes beta, correlation and concentration *together*.
        Seven separate tools that each answer one of them is a different product."""
        probe = next(p for p in probes() if p.key == "portfolio-copilot")
        assert "ONE answer" in probe.bar


class TestTheQuantitativeGapIsNotSoftened:
    def test_sharpe_is_undefined_rather_than_zero_while_nothing_has_settled(self) -> None:
        got = next(p for p in probes() if p.key == "t2-sharpe-mdd-winrate").execute()
        assert got.finding.status in {UNDEFINED, RUNS}
        if got.finding.status == UNDEFINED:
            assert "zero positions have settled" in got.finding.evidence
            assert "no amount of judged quality substitutes" in got.finding.evidence

    def test_undefined_is_not_counted_as_meeting_the_bar(self) -> None:
        got = Audit(results=(_probe("q", UNDEFINED).execute(),))
        assert "0/1 probes met their bar" in got.verdict


class TestTheAuditReportsItself:
    def test_the_report_serialises(self) -> None:
        got = Audit(results=tuple(p.execute() for p in (_probe("a", RUNS), _probe("b", WEAK))))
        blob = json.loads(json.dumps(got.as_dict()))
        assert blob["probes"] == 2 and blob["runs"] == 1 and blob["weak"] == 1

    def test_each_result_carries_its_question_and_bar_into_the_artefact(self) -> None:
        """So a reader of the JSON can check the bar was not fitted to the outcome."""
        got = Audit(results=(_probe("a", RUNS).execute(),)).as_dict()
        assert got["results"][0]["question"] == "q"
        assert got["results"][0]["bar"] == "b"

    def test_the_rendered_report_separates_subthemes_from_judging(self) -> None:
        got = Audit(results=tuple(
            p.execute() for p in (
                _probe("a", RUNS, track=2, kind="subtheme"),
                _probe("b", RUNS, track=2, kind="judging"),
            )
        ))
        text = got.render()
        assert "subthemes" in text and "judgings" in text

    def test_selecting_a_subset_runs_only_those(self) -> None:
        got = audit([_probe("only", RUNS)])
        assert [r.probe.key for r in got.results] == ["only"]


class TestTheLiveAudit:
    """Run the real probes against the real record. Slow, and the point of the module."""

    @pytest.fixture(scope="class")
    def live(self) -> Audit:
        return audit()

    def test_every_probe_returns_a_known_status(self, live: Audit) -> None:
        for result in live.results:
            assert result.finding.status in {RUNS, WEAK, ABSENT, UNDEFINED}, result.probe.key

    def test_no_probe_crashes_on_its_own_api(self, live: Audit) -> None:
        """An ABSENT caused by `probe raised TypeError` is a bug in this file, not a finding about
        ARGUS — six of them were, on the first run, and each was a wrong keyword or module path."""
        broken = [
            r.probe.key for r in live.results if r.finding.evidence.startswith("probe raised")
        ]
        assert not broken, f"probes calling the wrong API: {broken}"

    def test_every_finding_carries_evidence(self, live: Audit) -> None:
        for result in live.results:
            assert len(result.finding.evidence) > 40, result.probe.key

    def test_the_verdict_names_whatever_fell_short(self, live: Audit) -> None:
        for result in live.weakest:
            assert result.probe.key in live.verdict


class TestFourFailuresThatAreOneFailure:
    """Reading the Track 2 gaps as four problems invites four fixes. They are one.

    No Sharpe, no drawdown, an unexercised risk layer and an undefined takeover rate all follow
    from the desk never proposing a position. The audit says so once rather than leaving a reader
    to notice, and it names the constraint that is actually binding — which is the confidence gate,
    not the fee: the measured median absolute move is 137bps against an 18.8bps hurdle.
    """

    def test_the_cluster_is_diagnosed_once_rather_than_four_times(self) -> None:
        got = Audit(results=tuple(p.execute() for p in (
            _probe("t2-sharpe-mdd-winrate", UNDEFINED, kind="judging"),
            _probe("t2-risk-control", WEAK, kind="judging"),
        )))
        assert got.common_cause is not None
        assert "never proposed a position" in got.common_cause

    def test_it_blames_neither_the_fee_nor_the_confidence_floor(self) -> None:
        """Both were wrong, and one of them was published before it was checked.

        The fee was ruled out by the hurdle frontier: a median absolute move of 137bps against an
        18.8bps hurdle clears the cost seven times over. The confidence floor was then named as the
        binding constraint — and it has never run. `agents/desk.py:651` returns on ``quantity <= 0``
        before reaching the floor, so all 88 risk records read ``binding_constraint: none``. A
        constraint that never executes cannot be the binding one, and naming it sent an earlier
        iteration toward loosening a gate that was not in the path.
        """
        got = Audit(results=tuple(p.execute() for p in (
            _probe("t2-sharpe-mdd-winrate", UNDEFINED, kind="judging"),
            _probe("t2-open-evaluation", WEAK),
        )))
        assert got.common_cause is not None
        # Asserted as a property, not as a sentence. These three used to pin the exact prose, which
        # is how the diagnosis went stale unnoticed: the string kept passing while the ledger it
        # described had moved on.
        assert "confidence floor is not the cause" in got.common_cause
        assert "has never run" in got.common_cause
        assert "no-exposure gate" in got.common_cause

    def test_it_names_the_session_coverage_from_the_ledger_rather_than_from_a_string(
        self,
    ) -> None:
        """This test used to assert the word "weekend" and the phrase "never been offered".

        Both were true when written and false by 2026-09-14, when the desk took 48 decisions in an
        `rth` session. The diagnosis is now computed, so what is asserted here is that it says one
        of the two possible things — never that it says a particular one, which is the assertion
        that let a stale claim keep passing.
        """
        got = Audit(results=tuple(p.execute() for p in (
            _probe("t2-sharpe-mdd-winrate", UNDEFINED, kind="judging"),
            _probe("t2-risk-control", WEAK, kind="judging"),
        )))
        assert got.common_cause is not None
        assert (
            "never been offered" in got.common_cause
            or "HAS been offered tradeable sessions" in got.common_cause
        )

    def test_it_separates_a_proven_risk_layer_from_an_exercised_one(self) -> None:
        """6,720 swept states with no unreachable rule is a proof; 0 live interventions is a gap.
        Reporting only the second reads as a broken risk layer, which it is not."""
        got = Audit(results=tuple(p.execute() for p in (
            _probe("t2-sharpe-mdd-winrate", UNDEFINED, kind="judging"),
            _probe("t2-risk-control", WEAK, kind="judging"),
        )))
        assert got.common_cause is not None
        assert "proven over the swept domain" in got.common_cause
        assert "unexercised live" in got.common_cause

    def test_one_failure_alone_is_not_diagnosed_as_a_pattern(self) -> None:
        """Two of the cluster is a pattern; one is a single finding, and calling it a root cause
        would be reading a story into a single data point."""
        got = Audit(results=(_probe("t2-risk-control", WEAK, kind="judging").execute(),))
        assert got.common_cause is None

    def test_a_clean_track_two_reports_no_root_cause(self) -> None:
        got = Audit(results=tuple(p.execute() for p in (
            _probe("t2-sharpe-mdd-winrate", RUNS, kind="judging"),
            _probe("t2-risk-control", RUNS, kind="judging"),
        )))
        assert got.common_cause is None

    def test_the_diagnosis_reaches_the_rendered_report_and_the_artefact(self) -> None:
        got = Audit(results=tuple(p.execute() for p in (
            _probe("t2-sharpe-mdd-winrate", UNDEFINED, kind="judging"),
            _probe("t2-risk-control", WEAK, kind="judging"),
        )))
        assert "ROOT CAUSE" in got.render()
        assert got.as_dict()["common_cause"] == got.common_cause


def _stub_ledger(tmp_path, monkeypatch, rows) -> None:
    """Point the module's DATA at a temp directory holding just these ledger rows."""
    (tmp_path / "paper_ledger.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8"
    )
    (tmp_path / "risk_records.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr("argus.eval.themeaudit.DATA", tmp_path)


class TestTheRootCauseIsReadNotAsserted:
    """It was a hardcoded sentence, and it went stale the moment the desk ran on a weekday.

    The two readings below are opposite findings. Only the second is a statement about the desk,
    and a fixed string cannot tell them apart.
    """

    def test_no_tradeable_session_is_reported_as_a_sampling_fact(
        self, tmp_path, monkeypatch
    ) -> None:
        _stub_ledger(tmp_path, monkeypatch, [
            {"session_phase": "weekend", "lean": "none"},
            {"session_phase": "weekend", "lean": "none"},
        ])
        shape = _exposure_shape()
        assert shape.tradeable == 0
        assert "never been offered" in shape.sentence
        assert "about the sampling, not about the desk" in shape.sentence

    def test_a_tradeable_session_with_a_lean_is_reported_as_a_fact_about_the_desk(
        self, tmp_path, monkeypatch
    ) -> None:
        _stub_ledger(tmp_path, monkeypatch, [
            {"session_phase": "weekend", "lean": "none"},
            {"session_phase": "rth", "lean": "up"},
            {"session_phase": "rth", "lean": "down"},
            {"session_phase": "rth", "lean": "none"},
        ])
        shape = _exposure_shape()
        assert (shape.decisions, shape.tradeable, shape.with_lean) == (4, 3, 2)
        assert "HAS been offered tradeable sessions" in shape.sentence
        assert "statement about the desk" in shape.sentence

    def test_an_extended_session_counts_as_tradeable(self, tmp_path, monkeypatch) -> None:
        """The committed protocol names both `rth` and `extended`; counting only one understates."""
        _stub_ledger(tmp_path, monkeypatch, [{"session_phase": "extended", "lean": "up"}])
        assert _exposure_shape().tradeable == 1

    def test_an_empty_ledger_makes_no_claim_about_the_desk(self, tmp_path, monkeypatch) -> None:
        _stub_ledger(tmp_path, monkeypatch, [])
        shape = _exposure_shape()
        assert shape.decisions == 0
        assert "nothing here is a statement about the desk" in shape.sentence

    def test_a_missing_ledger_reads_as_absent_not_as_zero_decisions_with_a_verdict(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr("argus.eval.themeaudit.DATA", tmp_path)
        assert _read_jsonl(tmp_path / "nothing.jsonl") == []
