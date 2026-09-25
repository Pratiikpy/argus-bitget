"""Tests for the general-purpose rival comparison: HiGHS (via SciPy) and networkx against ARGUS's
arbitrage decomposition, on the capability's own constructed books and on real Bitget
spot-rToken and perpetual books captured to `data/general_arb_books*.json`.

No network: every case runs on the saved captures. The full run takes about a minute, most of
it HiGHS, so it is computed once per module.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from argus.eval import general_arb_comparison as module
from argus.eval.arbitrage_comparison import run_swept_cases
from argus.eval.artefact import SCRATCH_DIR
from argus.eval.general_arb_comparison import (
    ARTEFACT_PATH,
    BOOKS_PATH,
    HOLDOUT_PATH,
    Confusion,
    TwoBooks,
    highs_leg,
    load,
    main,
    networkx_cycle,
    swept_spreads,
)
from argus.market.depth import Level, OrderBook
from argus.research.executable_arb import leg_optimum


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return main()


def _held(report: dict[str, Any]) -> dict[str, Any]:
    held: dict[str, Any] | None = report["out_of_sample"]["held_out"]
    assert held is not None, f"{HOLDOUT_PATH} is missing"
    return held


class TestBaselineReproduced:
    def test_highs_and_the_exact_walk_agree_on_every_real_snapshot(
        self, report: dict[str, Any]
    ) -> None:
        for rep in (report["real_books"], _held(report)):
            agree = rep["baseline_agreement"]
            assert agree["verdict_mismatches"] == 0
            assert agree["highs_vs_argus_exact_max_abs_diff_net"] < 1e-9
            assert agree["highs_vs_argus_exact_capped_max_abs_diff_net"] < 1e-9

    def test_networkx_matches_the_optimum_on_existence(self, report: dict[str, Any]) -> None:
        for rep in (report["real_books"], _held(report)):
            assert rep["comparison"]["networkx_bellman_ford_touch"]["errors"] == 0


class TestSameInputComparison:
    def test_the_deployed_decomposition_accepts_spreads_the_book_cannot_pay(
        self, report: dict[str, Any]
    ) -> None:
        dep = report["real_books"]["comparison"]["argus_deployed_decompose"]
        assert dep["false_accept"] > 10 * dep["true_accept"]
        assert dep["precision"] < 0.1

    def test_the_adapted_exact_walk_makes_no_errors(self, report: dict[str, Any]) -> None:
        assert report["real_books"]["comparison"]["argus_exact_walk"]["errors"] == 0

    def test_every_scored_snapshot_uses_the_venues_published_fees(
        self, report: dict[str, Any]
    ) -> None:
        assert report["real_books"]["input"]["fee_schedules"] == ["spot 0.001 / perp 0.0006"]


class TestOutOfSample:
    def test_the_held_out_capture_repeats_the_finding(self, report: dict[str, Any]) -> None:
        held = _held(report)["comparison"]
        assert held["argus_deployed_decompose"]["false_accept"] > 0
        assert held["argus_deployed_decompose"]["precision"] < 0.5
        assert held["argus_exact_walk"]["errors"] == 0


class TestStatisticallyValid:
    def test_precision_upper_bound_stays_low_when_resampled_by_ticker(
        self, report: dict[str, Any]
    ) -> None:
        boot = report["real_books"]["ticker_bootstrap"]
        assert boot["clusters"] >= 50
        assert boot["precision_ci95"][1] < 0.5
        assert boot["false_accept_rate_ci95"][0] > 0


class TestAblation:
    def test_the_flat_spread_constant_carries_most_of_the_error(
        self, report: dict[str, Any]
    ) -> None:
        arms = report["real_books"]["ablation"]
        deployed = arms["deployed_constants"]["false_accept"]
        assert arms["measured_touch_only"]["false_accept"] < deployed / 10
        assert arms["venue_fees_only"]["false_accept"] < deployed
        assert arms["venue_fees_and_measured_touch"]["errors"] == 0


class TestConstructedInput:
    def test_the_swept_sequence_is_the_existing_comparisons_own(self) -> None:
        assert len(swept_spreads()) == run_swept_cases().n

    def test_the_deployed_decomposition_refuses_a_band_the_touch_clears(
        self, report: dict[str, Any]
    ) -> None:
        grid = report["constructed_input"]["grid_0_to_40bps_step_0_1"]
        assert grid["false_refuse_band_bps"] == [12.1, 14.6]
        assert grid["confusion"]["false_accept"] == 0

    def test_one_swept_disagreement_with_maxme_is_an_argus_error(
        self, report: dict[str, Any]
    ) -> None:
        refused = report["constructed_input"]["swept_false_refusals_bps"]
        assert refused == [pytest.approx(13.4657, abs=1e-3)]

    def test_the_designed_cases_agree(self, report: dict[str, Any]) -> None:
        assert report["constructed_input"]["designed_confusion"]["errors"] == 0


class TestAdversarial:
    def test_a_wide_spot_touch_fools_the_flat_decomposition_only(
        self, report: dict[str, Any]
    ) -> None:
        case = {c["name"]: c for c in report["adversarial"]}[
            "wide_spot_touch_behind_a_wide_mid_gap"]
        assert case["argus_deployed_monetizable"]
        assert case["truth_net_usdt"] == 0.0
        assert case["argus_exact_net_usdt"] == 0.0
        assert not case["networkx_cycle"]

    def test_a_one_hundredth_share_touch_is_sized_not_just_flagged(
        self, report: dict[str, Any]
    ) -> None:
        case = {c["name"]: c for c in report["adversarial"]}[
            "profitable_touch_one_hundredth_of_a_share_deep"]
        assert case["argus_exact_quantity"] == pytest.approx(0.01)
        assert case["argus_exact_net_usdt"] == pytest.approx(case["truth_net_usdt"])
        assert case["argus_exact_net_usdt"] < 0.01


class TestFailureCases:
    def test_each_failure_case_has_its_recorded_outcome(self, report: dict[str, Any]) -> None:
        got = {(f["system"], f["input"]): f["outcome"] for f in report["failure_cases"]}
        assert got[("highs", "empty ask ladder")] == "returned (0.0, 0.0)"
        assert got[("argus_exact", "negative fee")].startswith("ExecutableArbError")
        assert got[("order_book", "crossed book")].startswith("DepthError")

    def test_empty_spot_books_are_counted_not_dropped(self, report: dict[str, Any]) -> None:
        skipped = report["real_books"]["input"]["skipped"]
        assert sum(skipped.values()) > 0


class TestCosts:
    def test_both_real_costs_are_measured(self, report: dict[str, Any]) -> None:
        costs = report["costs"]
        assert costs["highs_both_directions_seconds"] > 0
        assert costs["argus_exact_both_directions_seconds"] > 0
        assert costs["argus_exact_both_directions_seconds"] < costs[
            "highs_both_directions_seconds"]


class TestReproducibility:
    def test_a_rerun_on_the_same_books_is_identical(self, report: dict[str, Any]) -> None:
        assert report["reproducibility"]["real_books_rerun_identical"]

    def test_the_committed_artefact_matches_a_fresh_run(self, report: dict[str, Any]) -> None:
        saved = json.loads(ARTEFACT_PATH.read_text(encoding="utf-8"))
        assert saved["reproducibility"]["digest"] == report["reproducibility"]["digest"]
        assert saved["real_books"]["comparison"] == report["real_books"]["comparison"]


class TestScope:
    def test_the_scope_statement_says_what_is_not_claimed(
        self, report: dict[str, Any]
    ) -> None:
        text = report["scope_statement"]
        assert "NOT claimed" in text
        assert "tie with the general tool" in text


class TestPerSnapshotRecord:
    def test_every_verdict_is_recorded_and_sums_to_the_confusions(
        self, report: dict[str, Any]
    ) -> None:
        """The groupwise audit needs each observation, not only the counts."""
        for rep in (report["real_books"], _held(report)):
            rows = rep["per_snapshot"]
            assert len(rows) == rep["input"]["scored_snapshots"]
            dep = rep["comparison"]["argus_deployed_decompose"]
            assert sum(r["argus_deployed"] and not r["truth_monetizable"] for r in rows) == dep[
                "false_accept"]
            assert sum(r["truth_monetizable"] for r in rows) == rep["comparison"][
                "truth_monetizable"]
            assert {r["phase"] for r in rows} == set(rep["input"]["session_phases"])


class TestSmallBooks:
    """The two exact methods and the graph test on books small enough to check by hand."""

    def _pair(self) -> TwoBooks:
        ts = datetime(2026, 9, 25, 16, 0, tzinfo=UTC)
        spot = OrderBook("RXUSDT", ts, (Level(Decimal("99.0"), Decimal("5")),),
                         (Level(Decimal("100.0"), Decimal("1")),
                          Level(Decimal("100.5"), Decimal("2"))))
        perp = OrderBook("XUSDT", ts, (Level(Decimal("101.0"), Decimal("1.5")),
                                       Level(Decimal("100.2"), Decimal("4"))),
                         (Level(Decimal("101.2"), Decimal("3")),))
        return TwoBooks("X", 0, spot, perp, Decimal("0.001"), Decimal("0.0006"), 0)

    def test_highs_and_the_walk_agree_on_net_and_size(self) -> None:
        tb = self._pair()
        net, qty = highs_leg(tb.spot.asks, tb.perp.bids, buy_fee=tb.spot_fee,
                             sell_fee=tb.perp_fee)
        exact = leg_optimum(tb.spot.asks, tb.perp.bids, buy_fee=tb.spot_fee,
                            sell_fee=tb.perp_fee)
        # 1 @ 100.0 -> 101.0 nets 101*0.9994 - 100*1.001 = 0.8394; 0.5 @ 100.5 -> 101.0 nets
        # 0.5*(100.9394 - 100.6005) = 0.16945; the next unit (100.5 -> 100.2) loses.
        assert net == pytest.approx(0.8394 + 0.16945, abs=1e-9)
        assert float(exact.net) == pytest.approx(net, abs=1e-9)
        assert qty == pytest.approx(1.5) and float(exact.quantity) == pytest.approx(1.5)

    def test_networkx_sees_the_cycle_at_the_touch(self) -> None:
        assert networkx_cycle(self._pair()) is True

    def test_a_notional_cap_binds_the_same_way_in_both(self) -> None:
        tb = self._pair()
        cap = Decimal("50")
        net, qty = highs_leg(tb.spot.asks, tb.perp.bids, buy_fee=tb.spot_fee,
                             sell_fee=tb.perp_fee, max_notional=cap)
        exact = leg_optimum(tb.spot.asks, tb.perp.bids, buy_fee=tb.spot_fee,
                            sell_fee=tb.perp_fee, max_notional=cap)
        assert float(exact.net) == pytest.approx(net, abs=1e-9)
        assert exact.capped is True and qty < 1

    def test_the_confusion_counts_a_false_accept_as_a_loss(self) -> None:
        c = Confusion()
        for said, truth in ((True, True), (True, False), (False, True), (False, False)):
            c.add(said, truth)
        d = c.as_dict()
        assert (d["true_accept"], d["false_accept"], d["false_refuse"], d["true_refuse"]) == (
            1, 1, 1, 1)
        assert d["precision"] == 0.5 and d["errors"] == 2


class TestLoading:
    def test_unscoreable_snapshots_are_counted_with_the_reason(self, tmp_path: Path) -> None:
        good = {"round": 0, "ticker": "X", "spot": "RXUSDT", "perp": "XUSDT",
                "spot_ts": "1758816000000", "perp_ts": "1758816000050",
                "spot_fee": "0.001", "perp_fee": "0.0006",
                "spot_bids": [["99", "1"]], "spot_asks": [["100", "1"]],
                "perp_bids": [["99.5", "1"]], "perp_asks": [["100.5", "1"]]}
        empty = good | {"ticker": "E", "spot_bids": [], "spot_asks": []}
        no_fee = good | {"ticker": "F", "perp_fee": None}
        path = tmp_path / "books.json"
        path.write_text(json.dumps({"captured_from": "a", "captured_to": "b",
                                    "snapshots": [good, empty, no_fee]}), encoding="utf-8")
        got = load(path)
        assert [b.ticker for b in got.books] == ["X"]
        assert got.books[0].skew_ms == 50
        assert sum(got.skipped.values()) == 2
        assert any("empty on both sides" in k for k in got.skipped)
        assert any("taker fee" in k for k in got.skipped)


class TestPublicReadiness:
    def test_no_local_path_reaches_the_code_or_the_artefact(self) -> None:
        paths = [Path(module.__file__), ARTEFACT_PATH, BOOKS_PATH, HOLDOUT_PATH]
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for marker in (SCRATCH_DIR, "AppData", "C:/Users", "C:\\Users", "/c/Users",
                           "best-of-the-best", "research/repos"):
                assert marker not in text, f"{marker!r} in {path.name}"

    def test_the_second_capture_is_not_called_a_guard_against_tuning(
        self, report: dict[str, Any]
    ) -> None:
        assert "code was frozen" not in report["scope_statement"]
        assert "Second capture, ten minutes later" in report["scope_statement"]
