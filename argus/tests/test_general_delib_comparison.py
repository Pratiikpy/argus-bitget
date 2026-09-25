"""Tests for the market-refereed deliberation-cost comparison.

Everything here runs on small synthetic books built in the test, so no tape, no Tardis download and
no model is needed. The properties pinned are the ones the comparison's honesty rests on: nothing
after the decision instant reaches a forecaster, outcome windows never overlap, every fast
prefix-sum path equals its brute-force definition, and the contenders called "ARGUS" are exactly
the shipped functions.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import random
import zlib
from datetime import UTC, datetime
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import pytest

from argus.agents.delay_cost import bar_delay_cost_bps, empirical_delay_cost_bps
from argus.agents.meta_pm import DEPTH_MULTIPLIER, deliberation_cost_bps
from argus.eval.general_delib_comparison import (
    ABLATIONS,
    CONTENDERS,
    HORIZONS_S,
    STRIDE_S,
    TRAIL_S,
    WARMUP_S,
    Book,
    Dataset,
    DelibComparisonError,
    Instant,
    adapted_parity,
    after_a_jump,
    argus_adapted_bars,
    argus_production,
    block_bootstrap_mse_difference,
    book_from_quotes,
    chronos_context,
    decision_instants,
    depth_multiplier_check,
    evaluate,
    expected_abs_move_from_quantiles,
    head_to_head,
    hurdle_effect,
    instants_digest,
    latencybench_linear,
    mid_grid,
    per_book,
    predictions,
    read_tape,
    read_tardis_quotes,
    salvage_gzip,
    score,
    split_halves,
    trailing_empirical,
    trailing_rv_sqrt,
    verdict,
)
from argus.llm.qwen import Thinking
from argus.truth.clocks import DualClock, SessionPhase

RTH_START = int(datetime(2026, 9, 24, 15, 0, tzinfo=UTC).timestamp())
"""Thursday 11:00 New York time: regular trading hours on both sides of every test book."""


def _quotes(
    start_s: int, seconds: int, seed: int, *, scale: float = 2e-4
) -> list[tuple[int, float, float]]:
    """One quote per second, stamped half-way through it, on a seeded log random walk."""
    rng = random.Random(seed)
    mid = 100.0
    out = []
    for k in range(seconds):
        mid *= math.exp(rng.gauss(0.0, scale))
        out.append(((start_s + k) * 1000 + 500, mid - 0.01, mid + 0.01))
    return out


def _book(seed: int = 1, seconds: int = WARMUP_S + 900, dataset: str = "synthetic",
          symbol: str = "NVDA") -> Book:
    book = book_from_quotes(dataset, symbol, _quotes(RTH_START, seconds, seed))
    assert book is not None
    return book


def _dataset(*seeds: int) -> Dataset:
    data = Dataset()
    for n, seed in enumerate(seeds):
        book = _book(seed, symbol=f"S{n}")
        data.books.append(book)
        data.instants.extend(decision_instants(book))
    return data


class TestReadingTheBooks:
    def test_salvage_skips_a_corrupt_span_and_keeps_both_members(self) -> None:
        first = gzip.compress(b'{"a":1}\n')
        second = gzip.compress(b'{"b":2}\n')
        blob, errors = salvage_gzip(first + b"\x1f\x8b\x08garbage-not-deflate" + second)
        assert errors == 1
        assert blob == b'{"a":1}\n{"b":2}\n'

    def test_salvage_of_a_clean_multi_member_file_reports_no_loss(self) -> None:
        blob, errors = salvage_gzip(gzip.compress(b"x\n") + gzip.compress(b"y\n"))
        assert (blob, errors) == (b"x\ny\n", 0)

    def test_gzip_module_would_have_stopped_where_salvage_continues(self) -> None:
        raw = gzip.compress(b"kept\n") + b"\x1f\x8b\x08junk" + gzip.compress(b"also kept\n")
        with pytest.raises((OSError, EOFError, zlib.error)):
            gzip.decompress(raw)
        assert salvage_gzip(raw)[0] == b"kept\nalso kept\n"

    def test_read_tape_takes_top_of_book_and_marks_disconnects(self, tmp_path: Path) -> None:
        def books5(symbol: str, ts: int, bid: str, ask: str) -> str:
            return json.dumps({"arg": {"channel": "books5", "symbol": symbol},
                               "data": [{"b": [[bid, "1"]], "a": [[ask, "1"]], "ts": str(ts)}]})

        rows = [
            {"t": 1000.0, "m": books5("NVDAUSDT", 1_000_000, "10", "11")},
            {"t": 1000.5, "m": books5("TSLAUSDT", 1_000_400, "20", "21")},
            {"t": 1001.0, "m": books5("NVDAUSDT", 1_001_000, "12", "11")},  # crossed: dropped
            {"t": 1001.2, "m": books5("XYZUSDT", 1_001_100, "1", "2")},  # not asked for
            {"t": 1002.0, "argus_event": "reconnect"},
            {"t": 1030.0, "m": books5("NVDAUSDT", 1_030_000, "10.5", "10.7")},
        ]
        text = "".join(json.dumps(r) + "\n" for r in rows)
        (tmp_path / "15.jsonl.gz").write_bytes(gzip.compress(text.encode()))
        tape = read_tape(tmp_path, symbols=("NVDA", "TSLA"))
        assert tape.quotes["NVDA"] == [(1_000_000, 10.0, 11.0), (1_030_000, 10.5, 10.7)]
        assert tape.quotes["TSLA"] == [(1_000_400, 20.0, 21.0)]
        assert tape.lines == len(rows)
        assert (1001.0, 1012.0) in tape.outages  # the disconnect, padded
        assert (1002.0, 1030.0) in tape.outages  # 28 s of silence

    def test_read_tardis_converts_microseconds_and_drops_crossed_rows(
        self, tmp_path: Path
    ) -> None:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["exchange", "symbol", "timestamp", "local_timestamp", "ask_amount",
                         "ask_price", "bid_price", "bid_amount"])
        writer.writerow(["bitget-futures", "NVDAUSDT", "2000000", "0", "1", "10.2", "10.0", "1"])
        writer.writerow(["bitget-futures", "NVDAUSDT", "1000000", "0", "1", "10.1", "9.9", "1"])
        writer.writerow(["bitget-futures", "NVDAUSDT", "3000000", "0", "1", "9.0", "9.5", "1"])
        path = tmp_path / "q.csv.gz"
        path.write_bytes(gzip.compress(buffer.getvalue().encode()))
        assert read_tardis_quotes(path) == [(1000, 9.9, 10.1), (2000, 10.0, 10.2)]


class TestGrid:
    def test_a_quote_is_invisible_until_its_own_timestamp_has_passed(self) -> None:
        quotes = [(1_000, 9.0, 11.0), (2_500, 19.0, 21.0)]
        assert mid_grid(quotes, 0, 3) == [None, 10.0, 10.0, 20.0]

    def test_too_short_a_book_is_not_built(self) -> None:
        assert book_from_quotes("d", "s", _quotes(RTH_START, WARMUP_S, 1)) is None
        assert book_from_quotes("d", "s", _quotes(RTH_START, 1, 1)) is None

    def test_window_refuses_a_gap_and_the_edges(self) -> None:
        book = Book("d", "s", 100, (1.0, None, 3.0, 4.0))
        assert book.window(103, 2) == [3.0, 4.0]
        assert book.window(103, 3) is None  # contains the missing second
        assert book.window(104, 1) is None  # past the end
        assert book.window(100, 2) is None  # before the start

    def test_prefix_sums_equal_their_definitions(self) -> None:
        book = _book(3)
        end = book.start_s + WARMUP_S + 120
        window = book.window(end, TRAIL_S)
        assert window is not None
        logs = [math.log(v) * 1e4 for v in window]
        sq = sum((b - a) ** 2 for a, b in pairwise(logs)) / (TRAIL_S - 1)
        assert book.trailing_sq_return_mean(end, TRAIL_S) == pytest.approx(sq, rel=1e-9)
        for h in HORIZONS_S:
            direct = sum(abs(logs[i + h] - logs[i]) for i in range(TRAIL_S - h)) / (TRAIL_S - h)
            assert book.trailing_abs_move_mean(end, TRAIL_S, h) == pytest.approx(direct, rel=1e-9)


class TestDecisionInstants:
    def test_instants_are_strided_and_outcomes_never_overlap(self) -> None:
        book = _book(4)
        instants = decision_instants(book)
        assert instants
        seconds = [i.second for i in instants]
        assert seconds[0] == book.start_s + WARMUP_S
        assert all(b - a == STRIDE_S for a, b in pairwise(seconds))
        assert max(HORIZONS_S) < STRIDE_S
        assert all(i.phase is SessionPhase.RTH for i in instants)

    def test_realised_move_is_the_absolute_log_move_in_bps(self) -> None:
        book = _book(5)
        instant = decision_instants(book)[2]
        now = book.at(instant.second)
        assert now is not None
        for k, h in enumerate(HORIZONS_S):
            later = book.at(instant.second + h)
            assert later is not None
            assert instant.realised_bps[k] == pytest.approx(abs(math.log(later / now)) * 1e4)

    def test_an_instant_whose_history_or_outcome_touches_an_outage_is_dropped(self) -> None:
        clean = _book(6)
        instants = decision_instants(clean)
        target = instants[3].second
        outage = ((float(target + 10), float(target + 12)),)
        broken = Book(clean.dataset, clean.symbol, clean.start_s, clean.mids, outage)
        kept = {i.second for i in decision_instants(broken)}
        assert target not in kept  # its 40 s outcome window crosses the outage
        # every instant whose warm-up history covers the outage goes too, and only those
        expected = {
            i.second for i in instants
            if not (i.second - WARMUP_S < target + 12 and i.second + max(HORIZONS_S) > target + 10)
        }
        assert kept == expected
        assert instants[0].second in kept  # finished before the outage began
        assert instants[4].second not in kept  # the outage sits inside its 30-minute history

    def test_digest_identifies_the_scored_instants(self) -> None:
        instants = decision_instants(_book(7))
        assert instants_digest(instants) == instants_digest(list(instants))
        assert instants_digest(instants) != instants_digest(instants[1:])


class TestContenders:
    def test_argus_production_is_the_shipped_charge_for_each_budget(self) -> None:
        instant = Instant("d", "s", RTH_START, SessionPhase.RTH, (1.0, 1.0, 1.0))
        session = DualClock().state(datetime.fromtimestamp(RTH_START, UTC))
        expected = tuple(
            float(deliberation_cost_bps(session, thinking=t, annualised_vol=Decimal("0.45")))
            for t in (Thinking.OFF, Thinking.LOW, Thinking.FULL)
        )
        assert HORIZONS_S == (3, 8, 40)
        assert argus_production(instant) == expected

    def test_argus_production_multiplies_off_hours(self) -> None:
        saturday = int(datetime(2026, 9, 26, 15, 0, tzinfo=UTC).timestamp())
        weekend = Instant("d", "s", saturday, SessionPhase.WEEKEND, (1.0, 1.0, 1.0))
        rth = Instant("d", "s", RTH_START, SessionPhase.RTH, (1.0, 1.0, 1.0))
        ratio = float(DEPTH_MULTIPLIER[SessionPhase.WEEKEND])
        for a, b in zip(argus_production(weekend), argus_production(rth), strict=True):
            assert a == pytest.approx(ratio * b)

    def test_argus_adapted_is_the_shipped_empirical_estimator(self) -> None:
        book = _book(8)
        instant = decision_instants(book)[1]
        window = book.window(instant.second, TRAIL_S)
        assert window is not None
        shipped = [float(empirical_delay_cost_bps(window, delay_s=h)) for h in HORIZONS_S]
        assert list(trailing_empirical(book, instant)) == pytest.approx(shipped, rel=1e-9)

    def test_bar_estimator_reads_minute_closes_ending_at_the_instant(self) -> None:
        book = _book(9)
        instant = decision_instants(book)[1]
        window = book.window(instant.second, TRAIL_S + 1)
        assert window is not None
        closes = window[::-1][::60][::-1]
        assert closes[-1] == book.at(instant.second)
        assert len(closes) == TRAIL_S // 60 + 1
        expected = [float(bar_delay_cost_bps(closes, bar_s=60, delay_s=h)) for h in HORIZONS_S]
        assert list(argus_adapted_bars(book, instant)) == expected

    def test_trailing_rv_is_the_gaussian_mean_absolute_move(self) -> None:
        book = _book(10)
        instant = decision_instants(book)[0]
        sigma = math.sqrt(book.trailing_sq_return_mean(instant.second, TRAIL_S))
        got = trailing_rv_sqrt(book, instant)
        for value, h in zip(got, HORIZONS_S, strict=True):
            assert value == pytest.approx(math.sqrt(2 / math.pi) * sigma * math.sqrt(h))

    def test_latencybench_saturates_past_its_decay_window(self) -> None:
        book = _book(11)
        instant = decision_instants(book)[0]
        costs = latencybench_linear(book, instant)
        assert costs[0] == costs[1] == costs[2] > 0  # 1.5 s window: every ARGUS delay is past it

    def test_no_contender_reads_after_the_instant(self) -> None:
        """Rewrite every mid after the instant; no prediction may change."""
        book = _book(12)
        instant = decision_instants(book)[2]
        cut = instant.second - book.start_s + 1
        future_changed = Book(book.dataset, book.symbol, book.start_s,
                              book.mids[:cut] + tuple(v * 1.5 if v else v
                                                      for v in book.mids[cut:]))
        for fn in (trailing_empirical, trailing_rv_sqrt, argus_adapted_bars, latencybench_linear):
            assert fn(book, instant) == fn(future_changed, instant)
        assert chronos_context(book, instant) == chronos_context(future_changed, instant)

    def test_missing_history_is_refused_not_guessed(self) -> None:
        book = _book(13)
        early = Instant("d", "s", book.start_s + 10, SessionPhase.RTH, (1.0, 1.0, 1.0))
        for fn in (trailing_empirical, trailing_rv_sqrt, argus_adapted_bars, latencybench_linear,
                   chronos_context):
            with pytest.raises(DelibComparisonError):
                fn(book, early)


class TestQuantileReading:
    def test_gaussian_quantiles_give_the_gaussian_mean_absolute_move(self) -> None:
        """Chronos-2's 21 levels on an exact standard normal. The two approximations pull in
        opposite directions — linear interpolation between knots overstates ``|Q|`` (the normal
        quantile function is convex above the median, concave below), flat tails understate it —
        and the net, measured here, is a small overstatement."""
        from statistics import NormalDist

        levels = [0.01, *[k / 20 for k in range(1, 20)], 0.99]
        values = [NormalDist(0.0, 1.0).inv_cdf(u) for u in levels]
        got = expected_abs_move_from_quantiles(levels, values, 0.0, grid=20_000)
        exact = math.sqrt(2 / math.pi)
        assert exact < got < exact * 1.005

    def test_a_point_mass_away_from_the_current_price(self) -> None:
        assert expected_abs_move_from_quantiles([0.1, 0.9], [5.0, 5.0], 2.0) == pytest.approx(3.0)

    def test_crossed_quantiles_are_repaired_to_a_monotone_function(self) -> None:
        crossed = expected_abs_move_from_quantiles([0.1, 0.5, 0.9], [1.0, -1.0, 0.0], 0.0)
        repaired = expected_abs_move_from_quantiles([0.1, 0.5, 0.9], [-1.0, 0.0, 1.0], 0.0)
        assert crossed == pytest.approx(repaired)

    def test_malformed_input_is_refused(self) -> None:
        with pytest.raises(DelibComparisonError):
            expected_abs_move_from_quantiles([0.5], [1.0], 0.0)
        with pytest.raises(DelibComparisonError):
            expected_abs_move_from_quantiles([0.1, 0.9], [1.0], 0.0)


class TestScoring:
    def test_score_is_mse_mae_bias_and_rank(self) -> None:
        s = score([1.0, 2.0, 3.0], [2.0, 2.0, 5.0])
        assert s.mse == pytest.approx((1 + 0 + 4) / 3)
        assert s.mae == pytest.approx(1.0)
        assert s.bias_ratio == pytest.approx(6.0 / 9.0)
        assert s.spearman == pytest.approx(0.8660254, rel=1e-6)  # the tie is mid-ranked

    def test_constant_forecast_has_no_rank_correlation(self) -> None:
        assert score([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]).spearman is None

    def test_empty_or_mismatched_input_is_refused(self) -> None:
        with pytest.raises(DelibComparisonError):
            score([], [])
        with pytest.raises(DelibComparisonError):
            score([1.0], [1.0, 2.0])

    def test_bootstrap_of_identical_forecasts_is_exactly_zero(self) -> None:
        blocks = [f"b{j % 5}" for j in range(50)]
        ys = [float(j % 7) for j in range(50)]
        preds = [1.0] * 50
        d = block_bootstrap_mse_difference(blocks, preds, preds, ys, resamples=200)
        assert d["point"] == d["ci95_low"] == d["ci95_high"] == 0.0
        assert d["blocks"] == 5.0

    def test_bootstrap_finds_a_forecaster_that_is_better_everywhere(self) -> None:
        rng = random.Random(2)
        blocks = [f"b{j // 10}" for j in range(200)]
        ys = [abs(rng.gauss(0.0, 1.0)) for _ in range(200)]
        good = [y + rng.gauss(0.0, 0.05) for y in ys]
        bad = [5.0] * 200
        d = block_bootstrap_mse_difference(blocks, bad, good, ys, resamples=500)
        assert d["point"] > 0 and d["ci95_low"] > 0
        reverse = block_bootstrap_mse_difference(blocks, good, bad, ys, resamples=500)
        assert reverse["ci95_high"] < 0

    def test_bootstrap_is_seeded(self) -> None:
        blocks = [f"b{j % 9}" for j in range(90)]
        ys = [float(j % 4) for j in range(90)]
        a = [float(j % 3) for j in range(90)]
        b = [1.5] * 90
        assert block_bootstrap_mse_difference(blocks, a, b, ys, resamples=300) == (
            block_bootstrap_mse_difference(blocks, a, b, ys, resamples=300)
        )


@pytest.fixture(scope="module")
def data() -> Dataset:
    return _dataset(21, 22, 23)


@pytest.fixture(scope="module")
def preds(data: Dataset) -> dict[str, list[tuple[float, ...]]]:
    return predictions(data)


class TestTheRun:
    def test_every_contender_and_ablation_predicts_every_instant(
        self, data: Dataset, preds: dict[str, list[tuple[float, ...]]]
    ) -> None:
        assert set(preds) == set(CONTENDERS + ABLATIONS)
        assert all(len(rows) == len(data.instants) for rows in preds.values())
        assert all(len(r) == len(HORIZONS_S) for rows in preds.values() for r in rows)

    def test_adapted_parity_holds_against_the_shipped_function(
        self, data: Dataset, preds: dict[str, list[tuple[float, ...]]]
    ) -> None:
        parity = adapted_parity(data, preds, every=3)
        assert parity["instants_checked"] > 0 and parity["matches"] is True

    def test_evaluate_pairs_every_contender_against_the_reference(
        self, data: Dataset, preds: dict[str, list[tuple[float, ...]]]
    ) -> None:
        report = evaluate(data.instants, preds, resamples=100)
        assert set(report["by_horizon"]) == {"3s", "8s", "40s"}
        for entry in report["by_horizon"].values():
            assert set(entry["vs_reference"]) == set(preds) - {"argus_production"}
            assert set(entry["by_phase"]) == {"rth"}
        lines = verdict(report)
        assert set(lines) == {"3s", "8s", "40s"}
        assert all("lowest MSE:" in line for line in lines.values())

    def test_the_constant_charge_loses_on_a_calm_synthetic_book(
        self, data: Dataset, preds: dict[str, list[tuple[float, ...]]]
    ) -> None:
        """At 2 bps a second the 0.45-vol constant is far off; the estimator that reads the book
        must beat it at every horizon, or the scoring is wired backwards."""
        report = evaluate(data.instants, preds, resamples=200, by_phase=False)
        for entry in report["by_horizon"].values():
            assert entry["pooled"]["argus_adapted"]["mse"] < entry["pooled"]["argus_production"][
                "mse"
            ]

    def test_diagnostics_run_on_the_same_instants(
        self, data: Dataset, preds: dict[str, list[tuple[float, ...]]]
    ) -> None:
        assert set(per_book(data, preds)) == {"synthetic"}
        halves = split_halves(data, preds)
        assert set(halves) == {"first_half", "second_half"}
        jump = after_a_jump(data, preds)
        assert 0 < jump["n"] < len(data.instants)
        depth = depth_multiplier_check(data)
        assert depth["40s"]["rth"]["realised_ratio_to_rth"] == 1.0
        hurdle = hurdle_effect(data, preds)
        assert hurdle["rth"]["n"] == len(data.instants)

    def test_head_to_head_scores_only_the_instants_the_rival_covered(
        self, data: Dataset, preds: dict[str, list[tuple[float, ...]]]
    ) -> None:
        covered = data.instants[::2]
        rival = {"forecasts": {i.key: list(i.realised_bps) for i in covered}}
        h2h = head_to_head(data, preds, rival, resamples=100)
        by = h2h["vs_argus_production"]["by_horizon"]
        assert by["3s"]["pooled"]["chronos2"]["n"] == len(covered)
        assert by["3s"]["pooled"]["chronos2"]["mse"] == 0.0  # a perfect "rival" by construction
        assert all(v == 0.0 for v in h2h["chronos2_worst_overshoot_bps"].values())
