"""Session-beta tests — the third orphan, whose docstring already called it reproducible.

`desk/portfolio.py` described `data/session_beta.json` as *"reproducible"* while nothing could
produce it. The arithmetic existed, the clock existed, and the step that joins them and saves the
answer did not. A file called reproducible that no command regenerates is the most expensive kind of
claim, because it reads as settled.

The rebuild lands on the stored figures: AAPLUSDT's open beta recomputes to +0.670 against a stored
0.6677, its shut beta to +0.105 against 0.0898, with 125 open bars against 126 — the residual is the
two days the window moved, not a difference in method.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from argus.research.session_beta_study import (
    BENCHMARK,
    MIN_SESSION_BARS,
    REPORT_PATH,
    SessionBetaError,
    SymbolBeta,
    measure,
    report_of,
)

AT = datetime(2026, 6, 1, tzinfo=UTC)


def _closes(values: list[float], *, start: datetime = AT) -> list[tuple[datetime, float]]:
    return [(start + timedelta(hours=i), v) for i, v in enumerate(values)]


def _always_open(_: datetime) -> bool:
    return True


def _alternating(stamp: datetime) -> bool:
    return stamp.hour % 2 == 0


def _linked(n: int, multiple: float) -> tuple[list[float], list[float]]:
    """A benchmark with *varying* returns, and an asset that is exactly ``multiple`` times each one.

    Varying deliberately: a constant-growth series has constant returns and therefore zero variance,
    and a beta regressed on zero variance is undefined. Building the asset from the benchmark's own
    per-bar return is what makes the expected beta exactly ``multiple``, not approximately it.
    """
    bench, asset = [100.0], [100.0]
    for i in range(1, n):
        step = 0.004 * ((i % 7) - 3)  # deterministic, both signs, no repeats within a cycle
        bench.append(bench[-1] * (1 + step))
        asset.append(asset[-1] * (1 + multiple * step))
    return asset, bench


class TestTheArithmetic:
    def test_a_symbol_that_moves_twice_the_benchmark_has_beta_two(self) -> None:
        asset, bench = _linked(60, 2.0)
        got = measure("X", _closes(asset), _closes(bench), is_open=_always_open)
        assert got.beta_blended is not None
        assert got.beta_blended == pytest.approx(2.0, abs=0.05)

    def test_an_inverse_symbol_has_negative_beta(self) -> None:
        asset, bench = _linked(60, -1.0)
        got = measure("X", _closes(asset), _closes(bench), is_open=_always_open)
        assert got.beta_blended is not None
        assert got.beta_blended == pytest.approx(-1.0, abs=0.05)

    def test_returns_are_bar_to_bar_so_n_is_one_less_than_the_bars(self) -> None:
        asset, bench = _linked(60, 1.5)
        got = measure("X", _closes(asset), _closes(bench), is_open=_always_open)
        assert got.n_open == 59


class TestTheSessionSplitIsRealNotDecorative:
    def test_the_two_sessions_partition_the_observations(self) -> None:
        asset, bench = _linked(120, 1.6)
        got = measure("X", _closes(asset), _closes(bench), is_open=_alternating)
        assert got.n_open + got.n_shut == 119
        assert got.n_open > 0 and got.n_shut > 0

    def test_a_session_with_too_few_bars_reports_absence_not_a_number(self) -> None:
        """`portfolio.beta` returns None there, and that None is carried through rather than being
        backfilled with the blended figure wearing a different label."""
        def open_only_at_the_start(stamp: datetime) -> bool:
            return stamp < AT + timedelta(hours=MIN_SESSION_BARS - 5)

        asset, bench = _linked(120, 1.6)
        got = measure("X", _closes(asset), _closes(bench), is_open=open_only_at_the_start)
        assert got.n_open < MIN_SESSION_BARS
        assert got.beta_open is None, "a beta from a handful of bars must not be published"
        assert got.beta_shut is not None, "the well-populated session still reports"

    def test_an_absent_session_is_not_counted_as_a_comparison(self) -> None:
        absent = SymbolBeta("X", 100, 1.0, None, 3, 0.5, 97)
        assert not absent.open_exceeds_shut, "None > 0.5 is not a comparison, it is an absence"


class TestItRefusesRatherThanInventing:
    def test_unaligned_series_raise_rather_than_producing_a_number(self) -> None:
        a, b = _linked(60, 1.0)
        bench = _closes(b)
        asset = _closes(a, start=AT + timedelta(days=400))
        with pytest.raises(SessionBetaError, match="share no timestamps"):
            measure("X", asset, bench, is_open=_always_open)

    def test_a_non_positive_close_raises(self) -> None:
        a, b = _linked(60, 1.0)
        bench = _closes(b)
        asset = _closes([0.0, *a[1:]])
        with pytest.raises(SessionBetaError, match="not a price"):
            measure("X", asset, bench, is_open=_always_open)


class TestTheArtefactIsReproducible:
    def test_report_of_emits_the_shape_the_docs_cite(self) -> None:
        rows = [SymbolBeta("A", 700, 0.25, 0.67, 126, 0.09, 592)]
        blob = report_of(rows)
        assert set(blob) >= {
            "generated_at", "benchmark", "interval", "lookback_days", "min_session_bars",
            "symbols_compared", "symbols_higher_open", "shut_bar_share_pct", "finding", "rows",
        }
        assert blob["symbols_higher_open"] == 1

    def test_the_finding_counts_only_comparable_symbols(self) -> None:
        rows = [
            SymbolBeta("A", 700, 0.25, 0.67, 126, 0.09, 592),   # comparable, open higher
            SymbolBeta("B", 700, 0.25, None, 3, 0.09, 592),     # open not estimated
        ]
        blob = report_of(rows)
        assert blob["symbols_compared"] == 1, "a symbol missing a session is not a comparison"
        assert blob["symbols_higher_open"] == 1

    def test_the_live_artefact_matches_this_module_s_shape(self) -> None:
        if not REPORT_PATH.exists():
            pytest.skip("no live artefact on this machine")
        blob = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        assert blob["benchmark"] == BENCHMARK
        assert blob["rows"], "the artefact must carry its per-symbol rows"
        for row in blob["rows"]:
            assert row["symbol"] != BENCHMARK, "the benchmark's beta to itself says nothing"
            assert row["n_open"] + row["n_shut"] <= row["observations"] + 1

    def test_the_live_finding_agrees_with_its_own_rows(self) -> None:
        """The sentence and the table must not be able to disagree."""
        if not REPORT_PATH.exists():
            pytest.skip("no live artefact on this machine")
        blob = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        comparable = [
            r for r in blob["rows"] if r["beta_open"] is not None and r["beta_shut"] is not None
        ]
        higher = [r for r in comparable if r["beta_open"] > r["beta_shut"]]
        assert blob["symbols_compared"] == len(comparable)
        assert blob["symbols_higher_open"] == len(higher)
        assert f"{len(higher)} of {len(comparable)}" in blob["finding"]
