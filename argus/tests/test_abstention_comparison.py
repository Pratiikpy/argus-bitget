"""Tests for the ARGUS-vs-AutonomousTradeAgents abstention-scoring comparison.

`bucket_from_marks`/`headline_saved_usd` are an independent, clean-room reimplementation (no
permissive license to vendor under — see `eval/baselines/ghostledger_reimpl.py`'s own docstring),
proven correct here by pinning the reference output computed by running their own real,
unmodified `_bucket_from_rows` once locally via a `sys.modules` shim (the exact scenario and
command are in that module's own docstring). ARGUS's real `abstention_quality()` is called
directly, never reimplemented.
"""

from __future__ import annotations

import pytest

from argus.eval.abstention_comparison import (
    NETTING_CASES,
    SCOPE_STATEMENT,
    defect_appears_only_near_full_offset,
    measure_costs,
    run_ablation,
    run_all_netting_cases,
    run_netting_case,
    swept_netting_ratios,
)
from argus.eval.baselines.ghostledger_reimpl import bucket_from_marks, headline_saved_usd


class TestGhostLedgerMatchesTheRealReferenceOutput:
    """The exact scenario `ghost_service.py`'s own code comment describes, computed by running
    their real, unmodified `_bucket_from_rows` once locally: `GhostBucket(ghost_pnl=2179.0,
    loss_avoided_usd=30788.0, upside_blocked_usd=32967.0, ...)`, `saved_usd=0.0`."""

    def test_matches_the_real_reference_net(self) -> None:
        bucket = bucket_from_marks([-30788.0, 32967.0])
        assert bucket.ghost_pnl == pytest.approx(2179.0)

    def test_matches_the_real_reference_split(self) -> None:
        bucket = bucket_from_marks([-30788.0, 32967.0])
        assert bucket.loss_avoided_usd == pytest.approx(30788.0)
        assert bucket.upside_blocked_usd == pytest.approx(32967.0)

    def test_matches_the_real_reference_headline_defect(self) -> None:
        bucket = bucket_from_marks([-30788.0, 32967.0])
        assert headline_saved_usd(bucket) == 0.0

    def test_a_clean_negative_net_is_not_floored_incorrectly(self) -> None:
        """The floor is only wrong when it hides a genuinely non-zero net — a real, purely
        negative net correctly produces a positive `saved_usd`, matching the real code's
        intended behaviour for the case it WAS designed for."""
        bucket = bucket_from_marks([-15000.0])
        assert headline_saved_usd(bucket) == pytest.approx(15000.0)


class TestNettingCases:
    def test_the_documented_case_reproduces_the_real_defect(self) -> None:
        result = run_netting_case(NETTING_CASES[0])
        assert result.ghost_headline_hides_a_nonzero_net is True
        assert result.argus_headline_shows_the_true_net is True

    def test_every_declared_case_runs(self) -> None:
        results = run_all_netting_cases()
        assert len(results) == len(NETTING_CASES)

    def test_argus_never_hides_the_true_net_on_any_case(self) -> None:
        results = run_all_netting_cases()
        assert all(r.argus_headline_shows_the_true_net for r in results)

    def test_as_dict_serialises_every_field(self) -> None:
        result = run_netting_case(NETTING_CASES[0])
        d = result.as_dict()
        assert "ghost_headline_hides_a_nonzero_net" in d
        assert "argus_headline_shows_the_true_net" in d


class TestSweptRatios:
    def test_the_defect_appears_only_near_full_offset_on_the_real_sweep(self) -> None:
        swept = swept_netting_ratios()
        assert defect_appears_only_near_full_offset(swept) is True

    def test_a_far_from_offset_ratio_never_hides_the_net(self) -> None:
        swept = swept_netting_ratios()
        low_ratio_rows = [row for row in swept if row["ratio"] < 0.5]
        assert low_ratio_rows
        assert not any(row["hides_nonzero_net"] for row in low_ratio_rows)


class TestAblation:
    def test_the_floor_is_confirmed_as_the_defect(self) -> None:
        result = run_ablation()
        assert result.unfloored_net_would_show_the_true_value is True
        assert result.real_headline_floors_it_to_zero is True
        assert result.the_floor_is_the_defect is True


class TestCosts:
    def test_both_costs_are_measured_and_positive(self) -> None:
        costs = measure_costs()
        assert costs["ghost_reimpl_us_per_call"] > 0
        assert costs["argus_us_per_call"] > 0


class TestScopeStatement:
    def test_names_what_is_not_claimed(self) -> None:
        assert "NOT claimed" in SCOPE_STATEMENT
        assert "genuinely well-considered design" in SCOPE_STATEMENT
