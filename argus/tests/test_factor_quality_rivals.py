"""The FactorMiner comparison: the panel it reads, the rank-to-weight step, and ARGUS's gates.

Everything here is offline and synthetic except the last class, which drives FactorMiner's own
interpreter when it has been created (`research/_venvs/factorminer`) and is skipped otherwise.
"""

from __future__ import annotations

import json
import math
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from argus.backtest.engine import Bar
from argus.eval import factor_quality_rivals as fqr
from argus.eval import factor_split_half as fsh
from argus.research.factor_lab import PRIMITIVES, Evaluator, Factor, FactorLab, Lifecycle

START = datetime(2026, 6, 1, tzinfo=UTC)


def _rows(n: int, seed: int, *, drift: float = 0.0) -> list[list[str]]:
    rng = random.Random(seed)
    price = 100.0
    rows = []
    for i in range(n):
        opened = price
        price *= 1.0 + drift + rng.gauss(0, 0.004)
        high, low = max(opened, price) * 1.001, min(opened, price) * 0.999
        volume = 100.0 + 50.0 * rng.random()
        rows.append([(START + timedelta(hours=i)).isoformat(), f"{opened:.6f}", f"{high:.6f}",
                     f"{low:.6f}", f"{price:.6f}", f"{volume:.4f}", f"{volume * price:.4f}"])
    return rows


def _series(n: int = 1500, symbols: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD")
            ) -> dict[str, list[list[str]]]:
    return {s: _rows(n, seed) for seed, s in enumerate(symbols)}


class TestPanel:
    def test_schema_and_amount_is_quote_volume(self) -> None:
        series = _series(20)
        panel = fqr.to_panel(series)
        assert fqr.PANEL_COLUMNS == ("datetime", "asset_id", "open", "high", "low", "close",
                                     "volume", "amount")
        assert len(panel) == 80
        first = panel[0]
        source = series["AAA"][0]
        assert first == [source[0], "AAA", *source[1:7]]
        assert [r[1] for r in panel[:20]] == ["AAA"] * 20

    def test_ragged_timestamps_are_refused(self) -> None:
        series = _series(20)
        series["BBB"] = series["BBB"][1:]
        with pytest.raises(fqr.RivalRunError, match="timestamps"):
            fqr.to_panel(series)

    def test_missing_quote_volume_is_refused(self) -> None:
        series = _series(5)
        series["AAA"] = [r[:6] for r in series["AAA"]]
        with pytest.raises(fqr.RivalRunError, match="quote volume"):
            fqr.to_panel(series)


class TestWeights:
    def test_ranks_map_onto_minus_one_to_one(self) -> None:
        w = fqr.cross_sectional_weights(np.array([[3.0], [1.0], [2.0]]))
        assert w[:, 0].tolist() == [1.0, -1.0, 0.0]

    def test_ties_share_the_average_rank(self) -> None:
        w = fqr.cross_sectional_weights(np.array([[1.0], [1.0], [2.0], [0.0]]))
        assert w[:, 0].tolist() == pytest.approx([0.0, 0.0, 1.0, -1.0])
        assert fqr.cross_sectional_weights(np.ones((5, 1)))[:, 0].tolist() == [0.0] * 5

    def test_missing_scores_are_flat_and_do_not_count(self) -> None:
        w = fqr.cross_sectional_weights(np.array([[np.nan, 5.0], [1.0, np.nan], [2.0, np.nan]]))
        assert w[:, 0].tolist() == [0.0, -1.0, 1.0]
        assert w[:, 1].tolist() == [0.0, 0.0, 0.0]  # one finite score: no cross-section

    def test_each_bar_uses_only_its_own_column(self) -> None:
        rng = np.random.default_rng(3)
        scores = rng.normal(size=(12, 50))
        full = fqr.cross_sectional_weights(scores)
        early = fqr.cross_sectional_weights(scores[:, :20])
        assert np.array_equal(full[:, :20], early)

    def test_rejects_a_flat_vector(self) -> None:
        with pytest.raises(ValueError):
            fqr.cross_sectional_weights(np.ones(4))


class TestGates:
    def test_primitives_through_gate_equal_the_lab(self) -> None:
        # The claim "identical gates" rests on this: every primitive through `gate` gets the lab's
        # cost+OOS verdict and the lab's split-half verdict, to the last field.
        for seed, drift in ((1, 0.0), (2, 0.0004)):
            bars = fsh.bars_from_rows(_rows(1500, seed, drift=drift))
            evaluator = Evaluator(bars)
            lab = FactorLab(evaluator=evaluator)
            for name in PRIMITIVES:
                lab.submit(Factor(name=name, expression=name, rationale=""))
            reached = {r.factor.name for r in lab.records if r.state is Lifecycle.OOS_TESTED}
            for name in PRIMITIVES:
                got = fqr.gate(bars, fqr.primitive_values(bars, name))
                assert got.passes_cost_and_oos == (name in reached), name
                factor = Factor(name=name, expression=name, rationale="")
                assert got.split_half == evaluator.split_half(factor).as_dict(), name

    def test_non_finite_weights_are_flat(self) -> None:
        bars = fsh.bars_from_rows(_rows(1500, 4))
        w = [1.0 if i % 7 else math.nan for i in range(len(bars) - 1)]
        clean = [0.0 if not math.isfinite(v) else v for v in w]
        assert fqr.gate(bars, w).as_dict() == fqr.gate(bars, clean).as_dict()

    def test_length_must_be_one_per_bar_but_the_last(self) -> None:
        bars = fsh.bars_from_rows(_rows(200, 5))
        with pytest.raises(ValueError, match="one per bar"):
            fqr.gate(bars, [0.0] * len(bars))

    def test_a_flat_factor_passes_nothing(self) -> None:
        bars = fsh.bars_from_rows(_rows(1500, 6))
        result = fqr.gate(bars, [0.0] * (len(bars) - 1))
        assert not result.passes_all and not result.passes_cost

    def test_a_perfect_foresight_factor_fails_cost_only_if_costs_bite(self) -> None:
        # Sanity: the gates can be passed. A weight that knows the next return's sign passes the
        # split-half gate on any series; the cost gate depends on turnover against 12 bps.
        bars = fsh.bars_from_rows(_rows(1500, 7))
        returns = fsh.next_returns(bars)
        result = fqr.gate(bars, [1.0 if r > 0 else -1.0 for r in returns])
        assert result.passes_split_half
        assert result.passes_all == (result.passes_cost and result.passes_oos)


class TestParity:
    def _published(self, row: dict[str, Any]) -> dict[str, Any]:
        return {"libraries": [{"symbol": "AAA", "factors": [
            {"factor": "slow_trend", "passes_cost_and_oos": row["passes_cost_and_oos"],
             "split_half": row["split_half"]}]}]}

    def test_identical_rows_pass(self) -> None:
        bars = fsh.bars_from_rows(_rows(1500, 8))
        mine = fqr.gate(bars, fqr.primitive_values(bars, "slow_trend")).as_dict()
        rows = [{"symbol": "AAA", "factor": "slow_trend", "as_written": mine}]
        assert fqr.check_parity(rows, self._published(mine)) == 1

    def test_a_different_split_half_raises(self) -> None:
        bars = fsh.bars_from_rows(_rows(1500, 8))
        mine = fqr.gate(bars, fqr.primitive_values(bars, "slow_trend")).as_dict()
        other = dict(mine, split_half=dict(mine["split_half"], p_value=0.0001))
        rows = [{"symbol": "AAA", "factor": "slow_trend", "as_written": mine}]
        with pytest.raises(fqr.RivalRunError, match="not identical"):
            fqr.check_parity(rows, self._published(other))


class TestStatistics:
    def test_binomial_tail_matches_direct_sum(self) -> None:
        n, p = 30, 0.052
        for k in (0, 1, 2, 5, 12, 30, 31):
            direct = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))
            assert fqr.binomial_tail(k, n, p) == pytest.approx(direct, rel=1e-9, abs=1e-15)

    def _summary(self, argus: tuple[int, int], rival: tuple[int, int]) -> dict[str, Any]:
        def rows(family: str, n: int, k: int) -> list[dict[str, Any]]:
            gate = {"passes_cost": True, "passes_cost_and_oos": True, "passes_split_half": True}
            return [{"family": family, "factor": f"f{i}", "symbol": "AAA", "category": "c",
                     **{arm: dict(gate, passes_all=i < k)
                        for arm in (("as_written",) if family == "argus" else fqr.ARMS)}}
                    for i in range(n)]
        all_rows = rows("argus", *argus) + rows("paper", *rival)
        factors = [{"family": "paper", "name": f"f{i}", "has_signal": True, "causal": True,
                    "stage_passed": 0, "ic_paper_mean": 0.0, "ic_paper_icir": 0.0}
                   for i in range(rival[0])]
        return fqr.summarise(all_rows, 0.052, factors)

    def test_verdict_tied_when_everyone_is_at_noise(self) -> None:
        decision = fqr.verdict(self._summary((96, 2), (1300, 60)), 0.052, power_80pct_at="0.62")
        assert decision["state"] == "TIED" and "0.62" in decision["reason"]

    def test_verdict_lost_when_a_rival_family_clears_noise_and_argus(self) -> None:
        decision = fqr.verdict(self._summary((96, 0), (1300, 200)), 0.052)
        assert decision["state"] == "LOST"
        assert "paper/as_written" in decision["families_above_noise_and_argus"]

    def test_verdict_names_the_case_the_rule_does_not_cover(self) -> None:
        decision = fqr.verdict(self._summary((96, 40), (1300, 60)), 0.052)
        assert decision["state"] == "NOT_DECIDED_BY_RULE"


class TestCompare:
    def test_end_to_end_offline_with_injected_signals(self) -> None:
        series = _series(1500)
        assets = sorted(series)
        periods = len(series["AAA"])
        rng = np.random.default_rng(9)
        signals = rng.normal(size=(2, len(assets), periods))
        verdicts = {
            "meta": {"asset_ids": assets, "periods": periods},
            "factors": [
                {"index": 0, "family": "paper", "name": "noise", "category": "Momentum",
                 "has_signal": True, "causal": True, "in_sample_ic_mean": -0.01,
                 "stage_passed": 0},
                {"index": 1, "family": "paper", "name": "peeks", "category": "Momentum",
                 "has_signal": True, "causal": False, "in_sample_ic_mean": 0.2,
                 "stage_passed": 1},
            ],
        }
        rows = fqr.compare(series, verdicts, signals, symbols=tuple(assets))
        argus = [r for r in rows if r["family"] == "argus"]
        rival = [r for r in rows if r["family"] == "paper"]
        assert len(argus) == len(PRIMITIVES) * len(assets)
        assert {r["factor"] for r in rival} == {"noise"}  # the non-causal factor is excluded
        assert all(r["orientation"] == -1.0 for r in rival)
        weights = fqr.cross_sectional_weights(signals[0])
        bars = fsh.bars_from_rows(series["AAA"])
        expected = fqr.gate(bars, weights[0, :-1].tolist()).as_dict()
        assert next(r for r in rival if r["symbol"] == "AAA")["as_written"] == expected

    def test_placebo_is_the_factor_rotated_against_its_returns(self) -> None:
        series = _series(1500)
        assets = sorted(series)
        periods = len(series["AAA"])
        signals = np.random.default_rng(4).normal(size=(1, len(assets), periods))
        verdicts = {"meta": {"asset_ids": assets, "periods": periods}, "factors": [
            {"index": 0, "family": "paper", "name": "noise", "category": "Momentum",
             "has_signal": True, "causal": True, "in_sample_ic_mean": 0.01, "stage_passed": 0}]}
        rows = fqr.compare(series, verdicts, signals, symbols=tuple(assets))
        row = next(r for r in rows if r["family"] == "paper" and r["symbol"] == "AAA")
        rolled = np.roll(fqr.cross_sectional_weights(signals[0]),
                         int(periods * fqr.PLACEBO_SHIFT), axis=1)
        bars = fsh.bars_from_rows(series["AAA"])
        assert row["placebo"] == fqr.gate(bars, rolled[0, :-1].tolist()).as_dict()
        assert row["placebo"] != row["as_written"]

    def test_asset_order_must_match_the_panel(self) -> None:
        series = _series(200)
        verdicts = {"meta": {"asset_ids": ["BBB", "AAA", "CCC", "DDD"], "periods": 200},
                    "factors": []}
        with pytest.raises(fqr.RivalRunError, match="asset order"):
            fqr.compare(series, verdicts, np.zeros((0, 4, 200)), symbols=("AAA",))


class TestFixture:
    def _write(self, path: Path, series: dict[str, list[list[str]]]) -> None:
        digests = {s: fsh._digest(r) for s, r in series.items()}
        blob: dict[str, Any] = {
            "series": {s: {"sha256": digests[s], "rows": r} for s, r in series.items()},
            "sha256": fsh._digest({s: digests[s] for s in sorted(series)}),
        }
        path.write_text(json.dumps(blob), encoding="utf-8")

    def test_round_trip_and_conversion(self, tmp_path: Path) -> None:
        series = _series(30)
        path = tmp_path / "bars.json"
        self._write(path, series)
        loaded, digest = fsh.load_fixture(path)
        assert loaded == series and len(digest) == 64
        bars = fsh.bars_from_rows(loaded["AAA"])
        row = series["AAA"][3]
        assert bars[3] == Bar(ts=datetime.fromisoformat(row[0]), close=Decimal(row[4]),
                              extra={"volume": float(row[5]), "high": float(row[2]),
                                     "low": float(row[3])})

    def test_an_edited_row_is_refused(self, tmp_path: Path) -> None:
        series = _series(30)
        path = tmp_path / "bars.json"
        self._write(path, series)
        blob = json.loads(path.read_text("utf-8"))
        blob["series"]["BBB"]["rows"][5][4] = "999.0"
        path.write_text(json.dumps(blob), encoding="utf-8")
        with pytest.raises(fsh.FixtureError, match="BBB"):
            fsh.load_fixture(path)

    def test_a_dropped_series_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "bars.json"
        self._write(path, _series(30))
        blob = json.loads(path.read_text("utf-8"))
        del blob["series"]["CCC"]
        path.write_text(json.dumps(blob), encoding="utf-8")
        with pytest.raises(fsh.FixtureError, match="set of series"):
            fsh.load_fixture(path)

    def test_missing_fixture_names_the_command(self, tmp_path: Path) -> None:
        with pytest.raises(fsh.FixtureError, match="--fetch"):
            fsh.load_fixture(tmp_path / "absent.json")


@pytest.mark.skipif(not (fqr.FACTORMINER_PYTHON.exists()
                         and (fqr.FACTORMINER_ROOT / "factorminer").is_dir()),
                    reason="FactorMiner's interpreter or clone is not on this machine")
class TestFactorMinerEngine:
    def test_the_panel_runs_through_factorminers_own_pipeline(self) -> None:
        series = _series(400, ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"))
        verdicts, signals = fqr.run_factorminer(fqr.to_panel(series))
        meta = verdicts["meta"]
        assert meta["asset_ids"] == sorted(series) and meta["periods"] == 400
        assert meta["paper_factors"] == 110 and meta["alpha101_classic"] == 12
        assert len(verdicts["factors"]) == 110 + 12 + 60
        assert signals.shape == (182, 6, 400)
        assert sum(f["has_signal"] for f in verdicts["factors"]) >= 170
        assert all(f["causal"] is not False for f in verdicts["factors"])
