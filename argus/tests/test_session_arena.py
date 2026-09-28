"""The session-execution arena's own logic, offline: the pre-registered verdict rule, the exact sign
test at the arena's sample sizes, and how days the source never served are kept out of every fit
and score (eval/session_arena.py; tracker 252)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from argus.eval import session_arena as arena


class TestTheVerdictRule:
    @pytest.mark.parametrize(("ci", "ratio", "expected"), [
        ([-3.0, -0.5], [0.9, 1.05], "WIN"),       # cheaper, dispersion not worse
        ([0.01, 1.53], [0.95, 1.02], "LOSS"),     # dearer: Bitget's TWAP on 2026-09-28
        ([-3.6, 0.38], [1.006, 1.031], "LOSS"),   # cost inside, dispersion worse: zz-0816
        ([-11.6, -3.2], [2.9, 3.6], "TRADE-OFF"),  # cheaper but riskier: immediate
        ([-1.0, 1.0], [0.95, 1.05], "TIE"),
    ])
    def test_each_outcome(self, ci: list[float], ratio: list[float], expected: str) -> None:
        cost = {"block_bootstrap_ci95_bps": ci}
        assert arena.verdict(cost, {"block_bootstrap_ci95": ratio}) == expected


class TestTheSignTest:
    def test_small_samples_are_exact(self) -> None:
        assert arena.sign_test(3, 1) == pytest.approx(0.625)
        assert arena.sign_test(0, 0) == 1.0

    def test_the_arenas_sample_sizes_do_not_overflow(self) -> None:
        # 2 ** n as a float overflowed past n ~ 1,000 parents (2026-09-28)
        p = arena.sign_test(2000, 1900)
        assert 0.0 < p < 1.0 and math.isfinite(p)
        # exact down to the far tail rather than rounded to nothing on the way
        assert 0.0 < arena.sign_test(1500, 500) < 1e-100


class TestDaysTheSourceNeverServed:
    def test_an_empty_day_is_recorded_and_skipped(self, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(arena, "CACHE", tmp_path)
        days = ("2025-09-01", "2025-10-01")
        assert arena._days(arena.PERP, ("NVDAUSDT",), days) == [
            ("NVDAUSDT", "2025-09-01"), ("NVDAUSDT", "2025-10-01")]
        arena._mark_unavailable(arena.PERP, "NVDAUSDT", "2025-09-01", "20-byte empty gzip")
        assert arena._days(arena.PERP, ("NVDAUSDT",), days) == [("NVDAUSDT", "2025-10-01")]
        assert arena.unavailable() == {f"{arena.PERP}/NVDAUSDT/2025-09-01": "20-byte empty gzip"}
        assert (tmp_path / "unavailable.json").read_bytes().endswith(b"}\n")

    def test_a_day_before_the_listing_is_not_asked_for(self, tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(arena, "CACHE", tmp_path)
        assert arena._days(arena.PERP, ("QQQUSDT",), ("2025-10-01", "2025-11-01")) == [
            ("QQQUSDT", "2025-11-01")]
