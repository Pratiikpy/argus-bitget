"""The cycle must surface what its own checks found.

Three checks run on every decision — numeric grounding, claim grounding, analyst conflict — and
until 2026-09-12 the runner discarded all three. The seq-41 hallucination was found by reading a
thesis by hand while the checker that catches it was already running and reporting to nobody.

So two properties are pinned here. The notes are **written**, beside the ledger rather than into it.
And a note that reports a *finding* is **flagged**, so it reaches the cycle summary rather than only
a sidecar file. The flag list matches on prose, which is fragile by nature — these tests take each
marker from the module that writes it, so a reworded check fails a test instead of quietly ceasing
to be flagged. The first draft of that list was written from memory and missed a real grounding
failure on the next live cycle.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from argus.agents.analysts import AnalystView
from argus.agents.claims import check as check_claims
from argus.agents.conflict import report as conflict_report
from argus.agents.grounding import check as check_grounding
from argus.paper import runner
from argus.paper.runner import _is_flag, _write_notes

AT = datetime(2026, 9, 12, 18, 0, tzinfo=UTC)


def _view(analyst: str, signal: str, magnitude: int, confidence: float) -> AnalystView:
    return AnalystView(
        analyst=analyst, signal=signal, magnitude_bps=magnitude, confidence=confidence,
        reasoning="r", counter_case="c", source_ids=(f"{analyst}-1",),
    )


class TestAFindingIsFlagged:
    """Each marker is taken from the module that writes it, not from memory."""

    def test_a_contradicted_claim_is_flagged(self) -> None:
        record = ("form4-1-S", {"pre_arranged": False})
        report = check_claims("These sales are pre-arranged 10b5-1 plans.", records=[record])
        assert not report.sound
        assert any(_is_flag(line) for line in report.render())

    def test_an_unresolved_figure_is_flagged(self) -> None:
        """The live miss: 'do not resolve to anything the desk was given' matched nothing."""
        report = check_grounding("The move was -0.046% today.", facts={"hurdle_bps": 12.0})
        assert not report.grounded
        assert any(_is_flag(line) for line in report.render())

    def test_a_real_disagreement_is_flagged(self) -> None:
        report = conflict_report(
            [_view("event", "bullish", 60, 0.9), _view("sentiment", "bearish", 55, 0.4)],
            sequential=False,
        )
        assert report.conflicts
        assert any(_is_flag(line) for line in report.render())


class TestLookingIsNotAFinding:
    """A check that ran and found nothing must not raise a flag, or every cycle is flagged."""

    def test_a_sound_claim_report_is_not_flagged(self) -> None:
        report = check_claims("Insider buying.", records=[("e", {"acquired": True})])
        assert report.sound
        assert not any(_is_flag(line) for line in report.render())

    def test_a_thesis_with_no_checkable_claim_is_not_flagged(self) -> None:
        report = check_claims("Weekend session, no edge.", records=[])
        assert not any(_is_flag(line) for line in report.render())

    def test_fully_grounded_figures_are_not_flagged(self) -> None:
        report = check_grounding("The hurdle is 12.0bps.", facts={"hurdle_bps": 12.0})
        assert report.grounded
        assert not any(_is_flag(line) for line in report.render())

    def test_agreement_is_not_flagged_even_when_labelled_possible_contagion(self) -> None:
        """The sequential caveat is information about method, not a finding about the market."""
        report = conflict_report(
            [_view("event", "bullish", 60, 0.9), _view("sentiment", "bullish", 58, 0.88)],
            sequential=True,
        )
        lines = report.render()
        assert any("contagion" in line for line in lines)
        assert not any(_is_flag(line) for line in lines)


class TestTheNotesAreWritten:
    @pytest.fixture(autouse=True)
    def _redirect(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runner, "NOTES_PATH", tmp_path / "desk_notes.jsonl")

    def _rows(self) -> list[dict]:
        return [
            json.loads(line)
            for line in runner.NOTES_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_a_decision_writes_one_row(self) -> None:
        _write_notes(7, "NVDAUSDT", AT, ["[grounding] the thesis states no figures"])
        assert len(self._rows()) == 1

    def test_the_row_is_keyed_to_the_ledger_entry(self) -> None:
        _write_notes(7, "NVDAUSDT", AT, ["note"])
        row = self._rows()[0]
        assert row["seq"] == 7 and row["symbol"] == "NVDAUSDT" and row["at"] == AT.isoformat()

    def test_every_note_is_kept_not_only_the_flagged_ones(self) -> None:
        notes = ["[grounding] all 2 figure(s) resolve to a computed value or a cited fact", "b"]
        _write_notes(7, "NVDAUSDT", AT, notes)
        assert self._rows()[0]["notes"] == notes

    def test_the_flags_are_separated_out(self) -> None:
        notes = [
            "[claim] x is contradicted by form4-1 (pre_arranged=False) - because",
            "[grounding] the thesis states no figures",
        ]
        _write_notes(7, "NVDAUSDT", AT, notes)
        assert self._rows()[0]["flags"] == [notes[0]]

    def test_a_second_decision_appends_rather_than_replaces(self) -> None:
        _write_notes(7, "NVDAUSDT", AT, ["a"])
        _write_notes(8, "TSLAUSDT", AT, ["b"])
        assert [r["seq"] for r in self._rows()] == [7, 8]

    def test_non_ascii_survives_the_round_trip(self) -> None:
        """The conflict note contains an em-dash, and these files are read on a cp1252 console."""
        note = "[conflict] none: 2 analysts agreed - but they ran in sequence — contagion"
        _write_notes(7, "NVDAUSDT", AT, [note])
        assert self._rows()[0]["notes"] == [note]


class TestTheCycleDegradesInsteadOfDying:
    """Every scheduled run from 08:52 on 2026-09-13 wrote nothing at all.

    The budget guard was doing exactly what it was told — a flat 150,000 tokens, against a
    twelve-symbol cycle that costs about 240,000 — so it raised partway through and the whole
    cycle was lost. Two separate defects: a limit that did not fit the job, and a failure mode
    that threw away the work already done.
    """

    def test_the_default_budget_is_derived_from_the_work_requested(self) -> None:
        import inspect

        from argus.paper.runner import PER_SYMBOL_TOKENS, run_once

        assert inspect.signature(run_once).parameters["budget"].default == 0
        assert PER_SYMBOL_TOKENS >= 20_000

    def test_the_per_symbol_figure_is_measured_not_guessed(self) -> None:
        """The marginal cost of one more symbol, solved from two live cycles: 20,838."""
        from argus.paper.runner import PER_SYMBOL_TOKENS

        assert PER_SYMBOL_TOKENS > 20_838, "must carry the measured marginal cost plus headroom"

    def test_a_twelve_symbol_cycle_gets_a_budget_that_fits_it(self) -> None:
        from argus.market.bitget import RTOKEN_SYMBOLS
        from argus.paper.runner import CYCLE_OVERHEAD_TOKENS, PER_SYMBOL_TOKENS

        needed = CYCLE_OVERHEAD_TOKENS + PER_SYMBOL_TOKENS * len(RTOKEN_SYMBOLS)
        assert needed > 150_000, (
            "the old flat limit was smaller than a full cycle, which is why every scheduled "
            "run died"
        )
        # Three live twelve-symbol cycles spent 269,809 / 280,182 / 284,372.
        assert needed > 284_372, "must cover the worst observed full cycle"


class TestTheBudgetHasAFixedTermAndNotOnlyAProportionalOne:
    """A proportional budget starves short cycles, and a one-symbol cycle proved it.

    ``PER_SYMBOL_TOKENS * len(symbols)`` assumes a cycle costs nothing until a symbol is added. Two
    live cycles on identical code say otherwise — 42,478 for one symbol and 104,991 for four, which
    solves to 20,838 marginal and 21,640 fixed. At twelve symbols the missing fixed term is inside
    the headroom and nobody notices; at one symbol it budgeted 25,000 against a true 42,478 and the
    guard fired 17,478 tokens short, killing the cycle with nothing written.
    """

    def test_a_single_symbol_cycle_is_funded_above_its_measured_cost(self) -> None:
        """The exact failure, pinned. 25,000 against a measured 42,478 wrote zero decisions."""
        from argus.paper.runner import CYCLE_OVERHEAD_TOKENS, PER_SYMBOL_TOKENS

        assert CYCLE_OVERHEAD_TOKENS + PER_SYMBOL_TOKENS > 42_478

    def test_the_fixed_term_is_not_zero(self) -> None:
        """If this is ever dropped back to a pure multiple, the one-symbol cycle dies again."""
        from argus.paper.runner import CYCLE_OVERHEAD_TOKENS

        assert CYCLE_OVERHEAD_TOKENS >= 21_640

    def test_the_model_predicts_the_cycles_it_was_not_fitted_on(self) -> None:
        """Fitted on n=1 and n=4; checked against three twelve-symbol cycles it never saw."""
        marginal, fixed = 20_838.0, 21_640.0
        for observed in (269_809, 280_182, 284_372):
            predicted = fixed + marginal * 12
            assert abs(predicted - observed) / observed < 0.05

    def test_the_default_scales_with_the_symbol_count(self) -> None:
        from argus.paper.runner import CYCLE_OVERHEAD_TOKENS, PER_SYMBOL_TOKENS

        one = CYCLE_OVERHEAD_TOKENS + PER_SYMBOL_TOKENS
        four = CYCLE_OVERHEAD_TOKENS + PER_SYMBOL_TOKENS * 4
        assert four - one == PER_SYMBOL_TOKENS * 3

    def test_the_stop_early_threshold_is_the_marginal_cost_not_the_whole_cycle(self) -> None:
        """The guard asks "can I afford one more symbol", not "can I afford a cycle". Using the
        overhead-inclusive figure would stop a run that had budget for several more decisions."""
        import inspect

        from argus.paper import runner

        source = inspect.getsource(runner.run_once)
        assert "remaining < PER_SYMBOL_TOKENS" in source
        assert "remaining < CYCLE_OVERHEAD_TOKENS" not in source
