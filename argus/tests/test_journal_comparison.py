"""Tests for the ARGUS-vs-serenity-guardrails journal comparison.

Every case here runs BOTH systems' real code (the vendored serenity-guardrails
``ChainedJournal`` and ARGUS's real ``PaperLedger``) — nothing is mocked or paraphrased. See
``eval/journal_comparison.py``'s module docstring for the real, reproducible, Windows-specific
finding these tests pin: serenity-guardrails' own ``verify()`` rejects a genuinely clean,
untampered chain due to a raw-disk-bytes hashing bug; ARGUS's ``PaperLedger`` is immune by
construction.
"""

from __future__ import annotations

import pytest

from argus.eval.baselines.serenity_loader import load_chained_journal_class
from argus.eval.journal_comparison import (
    SCOPE_STATEMENT,
    SWEPT_ENTRY_COUNTS,
    main,
    measure_costs,
    render,
    run_ablation,
    run_argus_crlf_case,
    run_argus_tamper_case,
    run_out_of_sample_case,
    run_serenity_crlf_case,
    run_serenity_tamper_case,
    run_tamper_comparison,
    run_truncation_comparison,
    swept_entry_counts,
)


@pytest.fixture
def chained_journal_class() -> type:
    return load_chained_journal_class()


class TestCrlfCase:
    def test_serenitys_own_real_code_writes_crlf_on_this_platform(
        self, chained_journal_class: type
    ) -> None:
        case = run_serenity_crlf_case(chained_journal_class)
        assert case.disk_bytes_contain_crlf is True

    def test_serenitys_own_real_verify_rejects_its_own_untampered_write(
        self, chained_journal_class: type
    ) -> None:
        """The core finding: zero tampering, real code on both ends, and verify() still raises."""
        case = run_serenity_crlf_case(chained_journal_class)
        assert case.verify_reports_clean is False
        assert "mismatch" in case.detail or "GuardError" in case.detail

    def test_argus_paper_ledger_also_writes_crlf_on_this_platform(self) -> None:
        case = run_argus_crlf_case()
        assert case.disk_bytes_contain_crlf is True

    def test_argus_paper_ledger_verifies_clean_under_the_identical_condition(self) -> None:
        case = run_argus_crlf_case()
        assert case.verify_reports_clean is True


class TestTamperComparison:
    def test_serenity_detects_a_genuine_content_edit(self, chained_journal_class: type) -> None:
        assert run_serenity_tamper_case(chained_journal_class) is True

    def test_argus_detects_a_genuine_content_edit(self) -> None:
        assert run_argus_tamper_case() is True

    def test_both_agree_on_genuine_tampering(self, chained_journal_class: type) -> None:
        result = run_tamper_comparison(chained_journal_class)
        assert result.both_detect_genuine_tampering is True


class TestAblation:
    def test_the_hashing_design_choice_is_load_bearing(self, chained_journal_class: type) -> None:
        result = run_ablation(chained_journal_class)
        assert result.hashing_raw_disk_bytes_is_fragile is True
        assert result.hashing_parsed_content_is_robust is True
        assert result.the_design_choice_is_load_bearing is True


class TestSweptEntryCounts:
    def test_serenity_fails_at_every_swept_entry_count(self, chained_journal_class: type) -> None:
        """The CRLF bug is not a fluke of N=2 — it fires from N=1 through N=50."""
        results = swept_entry_counts(chained_journal_class)
        assert [r.entry_count for r in results] == list(SWEPT_ENTRY_COUNTS)
        assert all(not r.serenity_clean for r in results)

    def test_argus_is_clean_at_every_swept_entry_count(self, chained_journal_class: type) -> None:
        results = swept_entry_counts(chained_journal_class)
        assert all(r.argus_clean for r in results)


class TestOutOfSample:
    def test_the_real_production_ledger_verifies_clean(self) -> None:
        """Not a synthetic fixture: the actual paper-trading ledger this session has been
        writing to. If it is absent in some environment, `checked` says so rather than the
        result silently reading as a pass."""
        result = run_out_of_sample_case()
        if result.checked:
            assert result.chain_intact is True
            assert result.entries > 0
        else:
            pytest.skip("real production ledger not present in this environment")


class TestTruncationComparison:
    def test_both_systems_detect_a_dropped_tail_entry(self, chained_journal_class: type) -> None:
        result = run_truncation_comparison(chained_journal_class)
        assert result["serenity_detects_truncation"] is True
        assert result["argus_detects_truncation"] is True


class TestCosts:
    def test_costs_are_measured_and_positive(self) -> None:
        costs = measure_costs()
        assert costs["write_ms_per_entry"] > 0
        assert costs["verify_seconds_for_250_entries"] >= 0


class TestMainAndRender:
    def test_main_runs_the_whole_comparison(self) -> None:
        report = main()
        assert report["serenity_crlf_case"]["verify_reports_clean"] is False
        assert report["argus_crlf_case"]["verify_reports_clean"] is True
        assert report["tamper_comparison"]["both_detect_genuine_tampering"] is True
        assert report["ablation"]["the_design_choice_is_load_bearing"] is True
        assert report["swept_serenity_always_fails"] is True
        assert report["swept_argus_always_clean"] is True
        assert report["truncation_comparison"]["argus_detects_truncation"] is True
        assert report["costs"]["write_ms_per_entry"] > 0

    def test_render_names_both_systems_and_the_disagreement(self) -> None:
        report = main()
        text = render(report)
        assert "ARGUS PaperLedger" in text
        assert "serenity-guardrails" in text
        assert "serenity clean: False" in text
        assert "ARGUS clean: True" in text

    def test_scope_statement_names_the_platform_boundary(self) -> None:
        assert "Windows" in SCOPE_STATEMENT
        assert "NOT claimed" in SCOPE_STATEMENT
        assert "not re-tested on a non-Windows platform" in SCOPE_STATEMENT
