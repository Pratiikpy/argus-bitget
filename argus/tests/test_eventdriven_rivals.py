"""eval/eventdriven_rivals.py and its vendored baselines: the pins, the draws, and the scorers.

The vendored rival bodies must still hash to the upstream blobs they cite; the placebo draws must
have the shapes the module docstring names; one draw must be scored by every system on identical
input; and the same seed must give the same p-values twice.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import pytest

from argus.eval import eventdriven_rivals as er
from argus.eval.baselines import vibe_trading_eventstudy_loader as loader


def test_vendored_bodies_hash_to_the_pinned_upstream_blobs() -> None:
    assert loader.vendored_body_sha256("vibe_trading_eventstudy.py") \
        == loader.VIBE_EVENTSTUDY_SHA256
    assert loader.vendored_body_sha256("ballast_stats.py") == loader.BALLAST_STATS_SHA256


def test_an_edited_vendored_body_is_refused(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    here = Path(loader.__file__).resolve().parent
    copy = tmp_path / "vibe_trading_eventstudy.py"
    shutil.copy(here / "vibe_trading_eventstudy.py", copy)
    copy.write_bytes(copy.read_bytes() + b"\n# an edit\n")
    monkeypatch.setattr(loader, "_BASELINES_DIR", tmp_path)
    with pytest.raises(loader.EventBaselineLoadError, match="edited"):
        loader._load("vibe_trading_eventstudy.py", "argus_test_tampered_vibe",
                     loader.VIBE_EVENTSTUDY_SHA256)


@pytest.fixture(scope="module")
def panel() -> er.Panel:
    return er.load_hourly_panel()


def test_the_snapshot_aligns_every_name(panel: er.Panel) -> None:
    assert panel.bars > er.ESTIMATION_BARS + er.GAP_BARS + er.EVENT_BARS + 200
    assert set(panel.returns) == set(er.NAMES)
    assert all(len(r) == panel.bars for r in panel.returns.values())
    assert len(panel.market) == panel.bars


@pytest.mark.parametrize("shape,dates", [("independent", 108), ("clustered_3", 36),
                                         ("clustered_9", 12), ("us_open_clustered", 12),
                                         ("crypto_pair", 30)])
def test_draw_shapes(panel: er.Panel, shape: str, dates: int) -> None:
    low, high = er.eligible(panel)
    events = er.draw_events(shape, random.Random(1), panel, low, high)
    assert len({i for _, i in events}) == dates
    assert all(low <= i < high for _, i in events)


def test_near_clustered_jitters_within_two_bars(panel: er.Panel) -> None:
    low, high = er.eligible(panel)
    exact = er.draw_events("clustered_9", random.Random(4), panel, low, high)
    near = er.draw_events("near_clustered", random.Random(4), panel, low, high)
    assert len(near) == len(exact)
    assert len({i for _, i in near}) > len({i for _, i in exact})


def test_unknown_shape_is_refused(panel: er.Panel) -> None:
    with pytest.raises(er.RivalStudyError):
        er.draw_events("nope", random.Random(0), panel, *er.eligible(panel))


def test_one_draw_is_scored_by_every_system(panel: er.Panel) -> None:
    scorers = er.load_scorers(panel)
    run = er.run_shape(panel, scorers, "clustered_9", n_draws=2, seed=3)
    for draw in run.draws:
        for key in er.DECISION_KEYS:
            assert 0.0 <= draw[key] <= 1.0, key
        assert draw["vibe.flagged_shared_dates"] == 1.0  # identical timestamps are flagged


def test_injected_shock_moves_every_system_toward_rejection(panel: er.Panel) -> None:
    scorers = er.load_scorers(panel)
    null = er.run_shape(panel, scorers, "independent", n_draws=1, seed=9).draws[0]
    shocked = er.run_shape(panel, scorers, "independent", n_draws=1, seed=9,
                           shock=0.05).draws[0]
    for key in ("argus.verdict", "vibe.bmp", "ballast.pooled_t"):
        assert shocked[key] <= null[key], key
        assert shocked[key] < 0.05, key


def test_same_seed_same_p_values(panel: er.Panel) -> None:
    out = er.reproducibility(panel, n_draws=2, seed=5)
    assert out["identical"] is True


def test_the_linear_cross_correlation_equals_the_pairwise_loop(panel: er.Panel) -> None:
    """`research/eventstudy.py:average_cross_correlation` replaced an O(N^2 L) pairwise loop with a
    sum-of-unit-vectors identity so studies of thousands of events are practical; its docstring
    says the two are pinned equal here. The loop below is the original, written out again."""
    from math import sqrt

    from argus.research.eventstudy import average_cross_correlation

    low, high = er.eligible(panel)
    events = er.draw_events("clustered_3", random.Random(11), panel, low, high)[:30]
    windows = er.argus_windows(panel, events, panel.returns)
    length = min(len(w.estimation_abnormal) for w in windows)
    series = []
    for w in windows:
        tail = w.estimation_abnormal[-length:]
        mean = sum(tail) / length
        centred = [x - mean for x in tail]
        norm = sqrt(sum(x * x for x in centred))
        if norm > 0:
            series.append([x / norm for x in centred])
    pairs = [sum(a * b for a, b in zip(series[i], series[j], strict=True))
             for i in range(len(series)) for j in range(i + 1, len(series))]
    assert average_cross_correlation(windows) == pytest.approx(sum(pairs) / len(pairs), abs=1e-12)


def test_wilson_and_critical_p() -> None:
    lo, hi = er.wilson(10, 200)
    assert lo < 0.05 < hi
    draws = [{"k": i / 100} for i in range(100)]
    assert er.critical_p(draws, "k") == 0.04


def test_artefact_headline_matches_its_own_tables() -> None:
    if not er.REPORT_PATH.exists():
        pytest.skip("run python -m argus.eval.eventdriven_rivals first")
    import json

    report = json.loads(er.REPORT_PATH.read_text(encoding="utf-8"))
    assert report["generated_from"]["snapshot_sha256"] == er.load_hourly_panel().source_sha256
    for shape, row in report["headline_false_positive_rates"].items():
        rates = report["size_under_null"][shape]["rejection_rates"]
        assert row["argus_verdict"] == rates["argus.verdict"]["rate"]
        assert row["vibe_bmp"] == rates["vibe.bmp"]["rate"]
    assert report["reproducibility"]["identical"] is True
