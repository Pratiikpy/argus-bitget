"""The PBO cross-check tests — closing the exact defect this module shipped for one session.

`track1_study._dsr` corrects for the variants tried on one symbol. It does not correct for testing
the same recipe across 12 symbols and reporting whichever one happened to clear 0.95 — a second,
uncorrected layer of search. On live data this produced a real false positive: TQQQUSDT/
`weekend_only` cleared `dsr_candidates_only` at 1.0, while the sibling PBO/FDR study — which does
correct for the full grid — found that exact pair the single most overfit result in the study
(PBO 0.771, "picking noise") and 0 of 300 grid trials significant under Benjamini-Hochberg or
Bonferroni.

These tests pin the two-part filter that prevents that headline from shipping again: a DSR survivor
must clear BOTH the per-symbol PBO verdict (not "picking noise") AND the grid-wide FDR/Bonferroni
survivor count (at least one significant result anywhere in the corrected grid) — because a survivor
can dodge the word "noise" in its own per-symbol reading while the properly-corrected grid still
finds nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from argus.research.track1_study import _pbo_cross_check


def _write_study(tmp_path: Path, rows: list[dict[str, object]], *, grid: dict[str, object]) -> Path:
    path = tmp_path / "overfitting_study.json"
    path.write_text(
        json.dumps({"symbols": rows, "grid": grid}), encoding="utf-8"
    )
    return path


class TestThePerSymbolVerdict:
    def test_a_noise_verdict_is_flagged_true(self, tmp_path: Path) -> None:
        path = _write_study(
            tmp_path,
            [{
                "symbol": "TQQQUSDT", "pbo_all_trials": 0.771,
                "verdict_all": "the selection procedure is picking noise: ...",
            }],
            grid={"survivors_benjamini_hochberg": 0, "survivors_bonferroni": 0, "trials": 300},
        )
        got = _pbo_cross_check("TQQQUSDT", path)
        assert got is not None
        assert got["picking_noise"] is True

    def test_a_non_noise_verdict_is_flagged_false(self, tmp_path: Path) -> None:
        path = _write_study(
            tmp_path,
            [{
                "symbol": "TSLAUSDT", "pbo_all_trials": 0.386,
                "verdict_all": "the selection carries real information but degrades substantially "
                "out of sample",
            }],
            grid={"survivors_benjamini_hochberg": 0, "survivors_bonferroni": 0, "trials": 300},
        )
        got = _pbo_cross_check("TSLAUSDT", path)
        assert got is not None
        assert got["picking_noise"] is False

    def test_a_symbol_absent_from_the_sibling_study_returns_none(self, tmp_path: Path) -> None:
        path = _write_study(
            tmp_path, [{"symbol": "OTHERUSDT", "pbo_all_trials": 0.1, "verdict_all": "x"}],
            grid={"survivors_benjamini_hochberg": 0, "survivors_bonferroni": 0, "trials": 300},
        )
        assert _pbo_cross_check("NVDAUSDT", path) is None

    def test_a_missing_sibling_study_returns_none_not_a_fabricated_agreement(
        self, tmp_path: Path
    ) -> None:
        assert _pbo_cross_check("NVDAUSDT", tmp_path / "absent.json") is None

    def test_a_malformed_sibling_study_returns_none_rather_than_raising(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "overfitting_study.json"
        path.write_text("{not valid json", encoding="utf-8")
        assert _pbo_cross_check("NVDAUSDT", path) is None


class TestTheGridWideCountIsCarried:
    """The number that made the two-part filter necessary: a per-symbol verdict avoiding the word
    "noise" is not the same claim as the corrected grid finding a real significant result."""

    def test_the_grid_survivor_counts_reach_the_check(self, tmp_path: Path) -> None:
        path = _write_study(
            tmp_path,
            [{"symbol": "TSLAUSDT", "pbo_all_trials": 0.386, "verdict_all": "carries real "
              "information but degrades substantially out of sample"}],
            grid={"survivors_benjamini_hochberg": 0, "survivors_bonferroni": 0, "trials": 300},
        )
        got = _pbo_cross_check("TSLAUSDT", path)
        assert got is not None
        assert got["grid_survivors_benjamini_hochberg"] == 0
        assert got["grid_trials"] == 300

    def test_a_nonzero_grid_count_is_carried_faithfully_too(self, tmp_path: Path) -> None:
        """The check must not always report zero — it reads what is actually there."""
        path = _write_study(
            tmp_path,
            [{"symbol": "AAPLUSDT", "pbo_all_trials": 0.05, "verdict_all": "survives"}],
            grid={"survivors_benjamini_hochberg": 3, "survivors_bonferroni": 1, "trials": 300},
        )
        got = _pbo_cross_check("AAPLUSDT", path)
        assert got is not None
        assert got["grid_survivors_benjamini_hochberg"] == 3
        assert got["grid_survivors_bonferroni"] == 1


class TestLiveArtefactRegressionGuard:
    """Pins the actual live finding, so a future re-run that silently drops the cross-check again
    is caught. Skips cleanly if the live artefacts are not present in this environment."""

    def test_the_live_tqqq_pair_is_flagged_picking_noise(self) -> None:
        import pytest

        from argus.research.track1_study import OVERFIT_STUDY_PATH

        if not OVERFIT_STUDY_PATH.exists():
            pytest.skip("overfitting_study.json not present in this environment")
        got = _pbo_cross_check("TQQQUSDT")
        if got is None:
            pytest.skip("TQQQUSDT not present in the live overfitting_study.json")
        assert got["picking_noise"] is True, (
            "the live TQQQUSDT/weekend_only pair was found 'picking noise' on 2026-09-15; if this "
            "assertion now fails, either the underlying data genuinely changed (fine, update this "
            "test with the new evidence) or the cross-check silently broke (not fine)"
        )
