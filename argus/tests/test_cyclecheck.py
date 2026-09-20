"""Cycle-check tests — an alarm that has never sounded is indistinguishable from a broken one.

This module exists to watch one cycle that has not happened yet: the first with price discovery. It
will run unattended, and if its falsifier detection is wrong nobody finds out until the evidence is
gone. So every check here is driven to FAIL on purpose, and the falsifier is fired on a synthetic
RTH cycle rather than waited for.

The first version of the session check read the phase from the log's `decisions_written` rows, which
carry seq, symbol, verdict, quantity, confidence and flags — and no session phase. It reported
"none recorded" and passed, so the falsifier could never have fired. That is the failure this file
is shaped around.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from argus.eval.cyclecheck import (
    EXPECTED_SYMBOLS,
    FAIL,
    PASS,
    UNKNOWN,
    CycleCheckError,
    check,
    latest_log,
)


@dataclass
class _Entry:
    seq: int
    session_phase: str = "weekend"
    lean: str = "down"


def _log(
    tmp_path: Path, *, rows: list[dict] | None = None, exit_code: int = 0,
    stopped: str = "", spent: int = 200_000, budget: int = 330_000,
    intact: bool = True, extra: str = "",
) -> Path:
    decisions = rows if rows is not None else [
        {"seq": 100 + i, "symbol": f"S{i}", "verdict": "no_trade", "quantity": "0"}
        for i in range(EXPECTED_SYMBOLS)
    ]
    body = {
        "decisions_written": decisions,
        "settled_this_cycle": [],
        "tokens_spent": spent,
        "token_budget": budget,
        "stopped_early": stopped,
        "chain": {"chain_intact": intact},
    }
    path = tmp_path / "cycle_2026-09-15_1330.log"
    path.write_text(
        "=== ARGUS paper cycle ===\n"
        + json.dumps(body, indent=1)
        + f"\n{extra}\nexit={exit_code}\n",
        encoding="utf-8",
    )
    return path


def _rows(base: int) -> list[dict]:
    return [
        {"seq": base + i, "symbol": f"S{i}", "verdict": "no_trade", "quantity": "0"}
        for i in range(EXPECTED_SYMBOLS)
    ]


def _entries(rows: list[dict], phase: str = "weekend") -> list[_Entry]:
    return [_Entry(seq=r["seq"], session_phase=phase) for r in rows]


def _records(rows: list[dict], leans: list[str] | None = None) -> list[dict]:
    """The **persisted** ledger rows, which is where a dropped key is still visible.

    Deliberately dicts and not `_Entry`: the bug these tests were rewritten around is that a loaded
    entry fills `lean` from its default, so absence and an honest ``none`` become the same object.
    """
    picked = leans or ["down"] * len(rows)
    return [
        {"seq": r["seq"], "symbol": r["symbol"], "lean": lean, "lean_confidence": 0.6}
        for r, lean in zip(rows, picked, strict=True)
    ]


class TestTheFalsifierFires:
    """`eval/autopsy.py`'s named observation: an RTH cycle where the desk proposes nothing."""

    def test_an_rth_cycle_with_no_exposure_fires_it(self, tmp_path: Path) -> None:
        rows = [
            {"seq": 200 + i, "symbol": f"S{i}", "verdict": "no_trade", "quantity": "0"}
            for i in range(EXPECTED_SYMBOLS)
        ]
        got = check(_log(tmp_path, rows=rows), entries=_entries(rows, "regular"))
        assert got.saw_price_discovery
        assert got.falsifier_fired
        assert "FALSIFIER FIRED" in got.verdict
        assert "must be rewritten, not defended" in got.verdict

    def test_a_weekend_cycle_never_fires_it(self, tmp_path: Path) -> None:
        rows = [
            {"seq": 300 + i, "symbol": f"S{i}", "verdict": "no_trade", "quantity": "0"}
            for i in range(EXPECTED_SYMBOLS)
        ]
        got = check(_log(tmp_path, rows=rows), entries=_entries(rows, "weekend"))
        assert not got.falsifier_fired
        assert "says nothing about whether this desk trades" in got.verdict

    def test_an_rth_cycle_that_traded_does_not_fire_it(self, tmp_path: Path) -> None:
        rows = [
            {"seq": 400 + i, "symbol": f"S{i}", "verdict": "no_trade", "quantity": "0"}
            for i in range(EXPECTED_SYMBOLS - 1)
        ] + [{"seq": 499, "symbol": "NVDAUSDT", "verdict": "trade", "quantity": "2"}]
        got = check(_log(tmp_path, rows=rows), entries=_entries(rows, "regular"))
        assert got.proposed_exposure == 1
        assert not got.falsifier_fired
        assert "first time it has done so on the live record" in got.verdict

    def test_human_review_is_not_counted_as_exposure(self, tmp_path: Path) -> None:
        """A malformed verdict routed to HUMAN_REVIEW carries no quantity and is not a position;
        counting it would mask the falsifier with a decision the desk never made."""
        rows = [
            {"seq": 500 + i, "symbol": f"S{i}", "verdict": "human_review", "quantity": "0"}
            for i in range(EXPECTED_SYMBOLS)
        ]
        got = check(_log(tmp_path, rows=rows), entries=_entries(rows, "regular"))
        assert got.proposed_exposure == 0
        assert got.falsifier_fired

    def test_an_unmatched_ledger_is_unknown_not_weekend(self, tmp_path: Path) -> None:
        """An unknown session must never be read as a shut market — that would silently disarm the
        alarm on exactly the cycle it was built for."""
        got = check(_log(tmp_path), entries=[])
        session = next(c for c in got.checks if c.name == "session")
        assert session.status == UNKNOWN
        assert "cannot be read as a weekend one" in session.detail


class TestEveryCheckCanFail:
    def test_a_nonzero_exit_fails(self, tmp_path: Path) -> None:
        got = check(_log(tmp_path, exit_code=1))
        assert next(c for c in got.checks if c.name == "exit_code").status == FAIL
        assert not got.sound

    def test_a_short_cycle_fails(self, tmp_path: Path) -> None:
        rows = [{"seq": 1, "symbol": "S", "verdict": "no_trade", "quantity": "0"}]
        got = check(_log(tmp_path, rows=rows))
        assert next(c for c in got.checks if c.name == "symbols_decided").status == FAIL

    def test_stopping_early_fails_and_carries_the_reason(self, tmp_path: Path) -> None:
        got = check(_log(tmp_path, stopped="stopped after 7 of 12 symbol(s): 15628 token(s) left"))
        found = next(c for c in got.checks if c.name == "ran_to_completion")
        assert found.status == FAIL
        assert "7 of 12" in found.detail

    def test_spending_the_whole_budget_fails(self, tmp_path: Path) -> None:
        got = check(_log(tmp_path, spent=330_000, budget=330_000))
        assert next(c for c in got.checks if c.name == "token_budget").status == FAIL

    def test_an_exhausted_guard_fails(self, tmp_path: Path) -> None:
        got = check(_log(tmp_path, extra="LLM unavailable: token budget exhausted: 33600/25000"))
        assert next(c for c in got.checks if c.name == "budget_not_exhausted").status == FAIL

    def test_a_budget_truncated_cycle_cannot_report_budget_not_exhausted(
        self, tmp_path: Path
    ) -> None:
        """**The live shape from cycle_2026-09-20_0100, which this check passed.** `runner.py`
        stops pre-emptively when the remaining budget is under one symbol's cost, so the guard in
        `qwen.py` never fires and its wording never reaches the log — and the old substring search
        therefore reported PASS beside `ran_to_completion FAIL` in the same report. A cycle stopped
        early with less than a symbol's budget left was stopped by the budget."""
        got = check(_log(
            tmp_path, spent=316_686, budget=330_000,
            stopped=(
                "stopped after 11 of 12 symbol(s): 13314 token(s) left of 330000, "
                "below the 25000 a symbol costs."
            ),
        ))
        found = next(c for c in got.checks if c.name == "budget_not_exhausted")
        assert found.status == FAIL
        assert "before the guard had to" in found.detail

    def test_stopping_early_with_budget_to_spare_is_not_blamed_on_the_budget(
        self, tmp_path: Path
    ) -> None:
        """The converse, so the new check cannot become a blanket FAIL on every early stop: a
        cycle that stopped for some other reason with 130,000 tokens left is not budget-starved."""
        got = check(_log(tmp_path, stopped="stopped after 7 of 12 symbol(s): venue unreachable"))
        assert next(c for c in got.checks if c.name == "budget_not_exhausted").status == PASS

    def test_a_broken_chain_fails_loudly(self, tmp_path: Path) -> None:
        got = check(_log(tmp_path, intact=False))
        found = next(c for c in got.checks if c.name == "chain_intact")
        assert found.status == FAIL
        assert "DID NOT VERIFY" in found.detail

    def test_a_dropped_lean_key_fails(self, tmp_path: Path) -> None:
        """The failure the check exists for, and the one the previous version could not see.

        `Entry.lean` defaults to ``"none"``, so a row that lost the key on disk loads as a row that
        answered ``none``. Checking the loaded object could never distinguish them; this reads the
        persisted row, where absence is still absence.
        """
        rows = _rows(600)
        records = _records(rows)
        del records[3]["lean"]
        got = check(_log(tmp_path, rows=rows), entries=_entries(rows), records=records)
        found = next(c for c in got.checks if c.name == "leans_recorded")
        assert found.status == FAIL
        assert "seq 603" in found.detail and "key absent" in found.detail

    def test_a_dropped_lean_confidence_fails(self, tmp_path: Path) -> None:
        """A lean with no confidence is not gradeable against the 56% break-even."""
        rows = _rows(610)
        records = _records(rows)
        del records[0]["lean_confidence"]
        got = check(_log(tmp_path, rows=rows), entries=_entries(rows), records=records)
        assert next(c for c in got.checks if c.name == "leans_recorded").status == FAIL

    def test_an_illegal_lean_value_fails(self, tmp_path: Path) -> None:
        """`agents/meta_pm.py:435` normalises to three values. A fourth is schema drift."""
        rows = _rows(620)
        records = _records(rows)
        records[5]["lean"] = "sideways"
        got = check(_log(tmp_path, rows=rows), entries=_entries(rows), records=records)
        found = next(c for c in got.checks if c.name == "leans_recorded")
        assert found.status == FAIL
        assert "sideways" in found.detail

    def test_one_honest_none_does_not_fail_the_cycle(self, tmp_path: Path) -> None:
        """The regression. Fired live on 2026-09-14: TQQQUSDT answered ``none`` and the harness
        failed a cycle in which the machinery had done its job.

        `decision/verdicts.py:128` documents ``none`` as a legitimate answer, and
        `agents/meta_pm.py:107` offers it to the model as one of three. Marking a correct answer
        wrong is worse than not checking, because it trains a reader to ignore the alarm.
        """
        rows = _rows(630)
        leans = ["none"] + ["up"] * (EXPECTED_SYMBOLS - 1)
        got = check(
            _log(tmp_path, rows=rows), entries=_entries(rows), records=_records(rows, leans),
        )
        assert next(c for c in got.checks if c.name == "leans_recorded").status == PASS
        assert next(c for c in got.checks if c.name == "leans_gradeable").status == PASS
        assert not got.failures

    def test_a_cycle_that_grades_nothing_fails(self, tmp_path: Path) -> None:
        """Every symbol answering ``none`` is the state the lean was introduced to end — all 156
        decisions before seq 157 read ``none``. The recording is intact and the cycle still produced
        nothing scoreable, which is why the two properties are separate checks."""
        rows = _rows(640)
        got = check(
            _log(tmp_path, rows=rows),
            entries=_entries(rows),
            records=_records(rows, ["none"] * EXPECTED_SYMBOLS),
        )
        assert next(c for c in got.checks if c.name == "leans_recorded").status == PASS
        found = next(c for c in got.checks if c.name == "leans_gradeable")
        assert found.status == FAIL
        assert "12 none" in found.detail

    def test_without_the_persisted_rows_both_are_unknown_not_pass(self, tmp_path: Path) -> None:
        """A check that cannot be evaluated says so. Folding it into PASS is how a harness reports
        success by default, which is the thing this module was written against."""
        rows = _rows(650)
        got = check(_log(tmp_path, rows=rows), entries=_entries(rows))
        assert {c.status for c in got.checks if c.name.startswith("leans_")} == {UNKNOWN}

    def test_an_unparseable_verdict_fails(self, tmp_path: Path) -> None:
        got = check(_log(tmp_path, extra="[unparseable verdict 'sell'; routed to human review]"))
        found = next(c for c in got.checks if c.name == "verdicts_parsed")
        assert found.status == FAIL
        assert "validate-and-retry" in found.detail

    def test_a_clean_cycle_passes_everything(self, tmp_path: Path) -> None:
        rows = _rows(700)
        got = check(_log(tmp_path, rows=rows), entries=_entries(rows), records=_records(rows))
        assert got.sound
        assert not got.failures
        assert not got.unknowns, "a clean cycle leaving a check unevaluated is not a clean cycle"


class TestUnknownIsNotPass:
    def test_a_stage_marker_is_not_mistaken_for_the_process_exit(self, tmp_path: Path) -> None:
        """The cycle script emits a marker per stage — `bookcalib_exit=`, `markout_exit=`,
        `effectiveness_exit=`, `check_exit=`. The unanchored regex matched the first of those and
        reported it as the process exit; a complete log carries ten such strings and only one is
        real. Here a stage FAILED and the process succeeded, and the check must say 0."""
        got = check(_log(tmp_path, exit_code=0, extra="markout_exit=0\ncheck_exit=1"))
        found = next(c for c in got.checks if c.name == "exit_code")
        assert found.status == PASS
        assert "0" in found.detail and "1" not in found.detail

    def test_a_failing_process_is_not_masked_by_passing_stages(self, tmp_path: Path) -> None:
        """The inverse, and the one that would cost us a silent failure: every stage exited 0 and
        the process did not."""
        got = check(_log(tmp_path, exit_code=1, extra="bookcalib_exit=0\nmarkout_exit=0"))
        found = next(c for c in got.checks if c.name == "exit_code")
        assert found.status == FAIL
        assert "1" in found.detail

    def test_an_in_flight_cycle_is_unknown_not_passed(self, tmp_path: Path) -> None:
        """A cycle still running has stage markers but no process exit. Reporting PASS there is how
        the most load-bearing check in this harness said "process exited 0" about a cycle that had
        not exited — observed live on cycle_2026-09-14_2300.log."""
        path = tmp_path / "cycle_2026-09-15_2300.log"
        path.write_text("=== ARGUS paper cycle ===\nbookcalib_exit=0\nmarkout_exit=0\n",
                        encoding="utf-8")
        found = next(c for c in check(path).checks if c.name == "exit_code")
        assert found.status == UNKNOWN
        assert "still running" in found.detail

    def test_a_missing_exit_code_is_unknown(self, tmp_path: Path) -> None:
        path = tmp_path / "cycle_x.log"
        path.write_text("nothing useful here", encoding="utf-8")
        got = check(path)
        assert next(c for c in got.checks if c.name == "exit_code").status == UNKNOWN

    def test_unknowns_are_named_in_the_verdict(self, tmp_path: Path) -> None:
        path = tmp_path / "cycle_x.log"
        path.write_text("nothing useful here", encoding="utf-8")
        assert "could not be evaluated" in check(path).verdict

    def test_an_unknown_does_not_make_the_cycle_unsound(self, tmp_path: Path) -> None:
        """UNKNOWN means the check could not run, which is not the same as the cycle being wrong.
        Reported, never counted as a failure and never counted as a pass."""
        got = check(_log(tmp_path), entries=None)
        assert any(c.status == UNKNOWN for c in got.checks)
        assert got.sound


class TestReadingTheRealRun:
    def test_a_missing_log_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CycleCheckError, match="does not exist"):
            check(tmp_path / "nope.log")

    def test_an_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CycleCheckError, match="no cycle logs"):
            latest_log(tmp_path)

    def test_the_newest_log_is_chosen(self, tmp_path: Path) -> None:
        for name in ("cycle_2026-09-13_1900.log", "cycle_2026-09-14_0100.log"):
            (tmp_path / name).write_text("x", encoding="utf-8")
        assert latest_log(tmp_path).name == "cycle_2026-09-14_0100.log"

    def test_the_live_latest_cycle_is_checkable(self) -> None:
        from argus.eval.cyclecheck import RUNS_DIR
        from argus.paper.ledger import PaperLedger
        from argus.paper.runner import LEDGER_PATH

        if not RUNS_DIR.exists() or not list(RUNS_DIR.glob("cycle_*.log")):
            pytest.skip("no cycle logs on this machine")
        entries = PaperLedger(path=LEDGER_PATH).entries if LEDGER_PATH.exists() else None
        records = (
            [
                json.loads(line)
                for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if LEDGER_PATH.exists()
            else None
        )
        got = check(latest_log(), entries=entries, records=records)
        assert got.checks
        assert got.verdict
        for found in got.checks:
            assert found.status in {PASS, FAIL, UNKNOWN}

    def test_the_report_serialises(self, tmp_path: Path) -> None:
        blob = json.loads(json.dumps(check(_log(tmp_path)).as_dict()))
        assert "falsifier_fired" in blob and "saw_price_discovery" in blob


class TestTheCycleScriptRunsTheCheck:
    """The monitor must not depend on a Claude session being alive.

    The cron watcher is session-only and dies when the session does. The scheduled Windows task is
    not, so the verification is wired into the cycle script itself — whether or not anyone is
    watching, `data/cycle_check.json` is written while the evidence is fresh.
    """

    def test_the_scheduled_script_invokes_cyclecheck(self) -> None:
        script = Path(__file__).resolve().parents[1] / "run_paper_cycle.ps1"
        if not script.exists():
            pytest.skip("the cycle script is not on this machine")
        text = script.read_text(encoding="utf-8", errors="replace")
        assert "argus.eval.cyclecheck" in text
        # After the runner, never before: it reads the log that run produces.
        assert text.index("argus.paper.runner") < text.index("argus.eval.cyclecheck")

    def test_the_script_records_the_checks_exit_code(self) -> None:
        script = Path(__file__).resolve().parents[1] / "run_paper_cycle.ps1"
        if not script.exists():
            pytest.skip("the cycle script is not on this machine")
        assert "check_exit=" in script.read_text(encoding="utf-8", errors="replace")


class TestTheDrainOnTheKeyIsTracked:
    """`token_budget` and `budget_not_exhausted` are both **per-cycle**.

    They answer *"did this run stay inside its cap"*, which says nothing about how much of a finite
    hackathon key is left. A key that exhausts mid-judging stops the scheduled cycle, freezes the
    ledger, and silently degrades the console to its deterministic layer — on the one surface judges
    are watching. Nothing measured that until 2026-09-15.
    """

    def test_it_sums_across_every_log(self, tmp_path: Path) -> None:
        from argus.eval.cyclecheck import cumulative_tokens

        for i, spend in enumerate((1000, 2000, 3000)):
            (tmp_path / f"cycle_2026-09-1{i}_1300.log").write_text(
                f'{{"tokens_spent": {spend}}}', encoding="utf-8",
            )
        total, cycles = cumulative_tokens(tmp_path)
        assert (total, cycles) == (6000, 3)

    def test_a_log_without_a_token_figure_is_not_counted_as_zero(self, tmp_path: Path) -> None:
        """A cycle whose spend was never written is unmeasured, not free."""
        from argus.eval.cyclecheck import cumulative_tokens

        (tmp_path / "cycle_a_1300.log").write_text('{"tokens_spent": 500}', encoding="utf-8")
        (tmp_path / "cycle_b_1300.log").write_text("no figures here", encoding="utf-8")
        total, cycles = cumulative_tokens(tmp_path)
        assert (total, cycles) == (500, 1), "the second log must not lower the average"

    def test_an_empty_directory_reports_nothing_rather_than_zero(self, tmp_path: Path) -> None:
        from argus.eval.cyclecheck import cumulative_tokens

        assert cumulative_tokens(tmp_path) == (0, 0)

    def test_the_check_reports_the_burn_rate_not_a_runway(self, tmp_path: Path) -> None:
        """The key's remaining balance is not observable from here, and dividing by a number we do
        not have would be exactly the invented figure this project refuses."""
        got = check(_log(tmp_path))
        found = next(c for c in got.checks if c.name == "cumulative_tokens")
        assert "burn rate, not a runway" in found.detail
        assert "not observable" in found.detail

    def test_it_is_reported_as_a_fact_not_graded_against_an_invented_threshold(
        self, tmp_path: Path
    ) -> None:
        got = check(_log(tmp_path))
        found = next(c for c in got.checks if c.name == "cumulative_tokens")
        assert found.status in (PASS, UNKNOWN), "there is no honest threshold to FAIL against"
