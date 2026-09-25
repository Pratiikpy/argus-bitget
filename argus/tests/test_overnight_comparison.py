"""The implied open: gloaming's overnight fair value against the perpetual, and the console line."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

import pytest

from argus.eval import overnight_comparison as oc


def _inputs(nights: int = 30) -> dict[str, Any]:
    """Synthetic sessions where the perpetual carries the true gap and the proxies carry half of
    it plus noise, so the scorer has a known answer."""
    hourly: dict[str, list[tuple[float, float]]] = {k: [] for k in
                                                    (*oc.PROXIES, *oc.CRYPTO, "NVDAUSDT")}
    from argus.eval.baselines.lean_market_holidays_loader import load_usa_equity_holidays

    # Session days only: the console's clock carries the NYSE calendar (Juneteenth and 3 July fall
    # in this window), so a synthetic session on a holiday would be anchored on the day before.
    shut = load_usa_equity_holidays()
    sessions = []
    price, day = 100.0, date(2026, 6, 1)
    for i in range(nights):
        while day.weekday() >= 5 or day in shut:
            day += timedelta(days=1)
        gap = (0.01 if i % 2 else -0.012) * (1 + i % 3)
        close_at = oc._ny(day, time(16))
        nxt = day + timedelta(days=1)
        while nxt.weekday() >= 5 or nxt in shut:
            nxt += timedelta(days=1)
        predict_at = oc._ny(nxt, time(9))
        for stamp, factor in ((predict_at - 24 * 3600, 1.0), (close_at, 1.0),
                              (predict_at - 3600, 1.0), (predict_at, 1.0 + gap)):
            hourly["NVDAUSDT"].append((stamp, price * factor))
            # Half the gap plus noise the gap does not explain, so no fit recovers it exactly.
            noise = 0.004 * ((i * 7) % 5 - 2)
            hourly["NQ=F"].append((stamp, 1000 * (1 + (gap / 2 + noise) * (factor != 1.0))))
            hourly["ES=F"].append((stamp, 1000.0))
            hourly["DX-Y.NYB"].append((stamp, 100.0))
            for c in oc.CRYPTO:
                hourly[c].append((stamp, 50.0 * (1 + (0.02 if i % 5 == 0 else 0) * (factor != 1))))
        sessions.append({"day": day.isoformat(), "open": price, "close": price})
        price *= 1 + gap
        sessions.append({"day": nxt.isoformat(), "open": price, "close": price})
        day = nxt + timedelta(days=1)
    for key in hourly:
        hourly[key].sort()
    # Pairs are adjacent sessions; keep only the constructed ones.
    return {"fetched": "test", "hourly": hourly, "sessions": {"NVDA": sessions}, "failures": {}}


@pytest.fixture(autouse=True)
def only_nvda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oc, "UNIVERSE", {"NVDA": "NQ=F"})


def test_each_input_is_read_at_its_moment_and_stale_prices_are_refused() -> None:
    series = [(100.0, 1.0), (200.0, 2.0)]
    assert oc._at(series, 199.0) == 1.0 and oc._at(series, 200.0) == 2.0
    assert oc._at(series, 50.0) is None
    assert oc._at(series, 200.0 + 4 * 3600) is None


def test_the_scorer_finds_the_estimator_that_carries_the_gap() -> None:
    report = oc.score(_inputs())
    summary = report["summary"]
    assert summary["argus_perp"]["mae_bps"] < 1.0
    assert summary["gloaming_prior"]["mae_bps"] > 10.0
    assert summary["argus_perp"]["direction_hit_rate"] == 1.0
    ours_vs_theirs = report["paired"][0]
    assert ours_vs_theirs["a"] == "argus_perp" and ours_vs_theirs["verdict"] == "a better"


def test_calibrated_candidates_only_use_earlier_nights() -> None:
    rows = oc.predict(oc.nights(_inputs(14)))
    assert [r["fitted"] for r in rows][: oc.MIN_FIT] == [False] * oc.MIN_FIT
    assert all(r["estimates"]["gloaming_ols"] == r["estimates"]["gloaming_prior"]
               for r in rows[: oc.MIN_FIT])


def test_the_console_states_the_implied_open_with_its_record(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    import importlib

    from argus.lui import research
    from argus.market import history

    answer = importlib.import_module("argus.lui.answer")
    report = {"best_gloaming": "gloaming_ols", "nights": 88,
              "summary": {"argus_perp": {"mae_bps": 30.4, "direction_hit_rate": 0.93},
                          "gloaming_ols": {"mae_bps": 84.4}, "zero": {"mae_bps": 92.0}},
              "per_stock": {"QQQ": {"nights": 88, "argus_perp": 13.0, "gloaming_ols": 13.5,
                                    "zero": 71.7, "argus_perp_direction_hit_rate": 1.0,
                                    "argus_perp_vs_best_gloaming": {"verdict": "not separable"}}}}
    (tmp_path / "overnight_comparison.json").write_text(json.dumps(report), "utf-8")
    monkeypatch.setattr(answer, "_notes_path", lambda: tmp_path / "notes.json")
    close = datetime(2026, 9, 24, 20, tzinfo=UTC)
    monkeypatch.setattr(research, "_last_regular_close", lambda now: close)
    monkeypatch.setattr(research, "_yahoo_close", lambda symbol, at: 200.0)
    monkeypatch.setattr(history, "fetch", lambda *a, **k: [history.Candle(
        ts=close - timedelta(hours=1), open=Decimal(1), high=Decimal(1), low=Decimal(1),
        close=Decimal("100"), volume=Decimal(0))])

    line, source = research._implied_open_line("NVDAUSDT", Decimal("101"))
    assert "moved +1.00% since the Thu 20:00 UTC close" in line
    assert "near 202.00 at the next open (last close 200)" in line
    assert "on 88 nights across 1 stocks, the open was missed by 30bps" in line
    assert source.ref == "argus.eval.overnight_comparison"

    tied, _ = research._implied_open_line("QQQUSDT", Decimal("101"))
    assert "QQQ on 88 nights" in tied and "(a difference too small to call)" in tied
    assert research._implied_open_line("BTCUSDT", Decimal("101")) is None


def test_no_close_bar_means_no_line(monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.lui import research
    from argus.market import history

    monkeypatch.setattr(research, "_last_regular_close",
                        lambda now: datetime(2026, 9, 24, 20, tzinfo=UTC))
    monkeypatch.setattr(research, "_yahoo_close", lambda symbol, at: 200.0)
    monkeypatch.setattr(history, "fetch", lambda *a, **k: [])
    assert research._implied_open_line("NVDAUSDT", Decimal("101")) is None


def test_the_last_regular_close_skips_weekends_and_holidays() -> None:
    from argus.lui import research

    saturday = datetime(2026, 9, 26, 12, tzinfo=UTC)
    assert research._last_regular_close(saturday) == datetime(2026, 9, 25, 20, tzinfo=UTC)
    thanksgiving_evening = datetime(2026, 11, 26, 23, tzinfo=UTC)
    assert research._last_regular_close(thanksgiving_evening) == datetime(
        2026, 11, 25, 21, tzinfo=UTC)


def test_nocturnes_claim_is_scored_on_the_stock_at_its_own_moment() -> None:
    """Fridays into Mondays only; the perpetual's weekend move against the last regular close."""
    inputs = _inputs(60)
    assert oc.sunday_evening({**inputs, "sessions": {"NVDA": inputs["sessions"]["NVDA"][:4]}}) \
        is None
    from argus.eval.baselines.lean_market_holidays_loader import load_usa_equity_holidays

    friday, monday = date(2026, 6, 5), date(2026, 6, 8)
    hourly: dict[str, list[tuple[float, float]]] = {k: [] for k in inputs["hourly"]}
    sessions = []
    # Holiday Fridays (19 June, 3 July) are not sessions: the console anchors on Thursday then.
    weeks = [w for w in range(24)
             if friday + timedelta(weeks=w) not in load_usa_equity_holidays()]
    for week in weeks:
        fri, mon = friday + timedelta(weeks=week), monday + timedelta(weeks=week)
        move = 0.01 * (1 if week % 2 else -1)
        hourly["NVDAUSDT"] += [(oc._ny(fri, time(16)), 100.0),
                               (oc._ny(mon - timedelta(days=1), time(20)), 100.0 * (1 + move))]
        sessions += [{"day": fri.isoformat(), "open": 100.0, "close": 100.0},
                     {"day": mon.isoformat(), "open": 100.0 * (1 + move), "close": 100.0}]
    report = oc.sunday_evening({"hourly": hourly, "sessions": {"NVDA": sessions}})
    assert report is not None and report["weekends"] == len(weeks) == 22
    assert report["slope_gap_on_perp_weekend_move"] == 1.0
    assert report["paired"]["verdict"] == "a better"


@pytest.mark.skipif(not (oc.ROOT.parent / "research" / "repos-rivals" / "nocturne").exists(),
                    reason="nocturne's clone is not on this machine")
def test_nocturnes_published_walk_forward_reproduces_from_its_clone() -> None:
    from argus.eval import void_comparison as vc

    preds = vc.predictions()
    assert vc._score(preds, "M0")["mae_pct"] == 2.117
    assert vc._score(preds, "MF")["mae_pct"] == 1.922


@pytest.mark.parametrize(("text", "routed"), [
    ("where will NVDA open?", True), ("what will TSLA open at monday", True),
    ("what is NVDA worth right now?", True), ("fair value of AAPL", True),
    ("NVDA opening price", True), ("英伟达开盘价会是多少", True),
    ("what is the open interest on BTC", False), ("should I open a position in NVDA", False),
])
def test_implied_open_questions_are_recognised_and_open_interest_is_not(
        text: str, routed: bool) -> None:
    from argus.lui.research import IMPLIED_OPEN_QUESTION

    assert bool(IMPLIED_OPEN_QUESTION.search(text)) is routed


@pytest.mark.skipif(not oc.GLOAMING_ENGINE.exists(), reason="gloaming's clone is not here")
def test_the_vendored_gloaming_model_is_its_clone_but_for_the_one_import() -> None:
    from pathlib import Path

    vendored = (Path(oc.__file__).parent / "baselines" / "gloaming_fairvalue.py").read_text("utf-8")
    original = (oc.GLOAMING_ENGINE / "fairvalue" / "model.py").read_text("utf-8")
    body = original.replace("from fairvalue.config import FAIRVALUE_WEIGHTS\n", "")
    for line in body.splitlines():
        assert line in vendored.splitlines()


def test_while_shut_only_the_regular_close_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """An extended-hours quote is not a close: without the daily bar there is no line."""
    from argus.lui import research

    monkeypatch.setattr(research, "_last_regular_close",
                        lambda now: datetime(2026, 9, 24, 20, tzinfo=UTC))
    monkeypatch.setattr(research, "_yahoo_close", lambda symbol, at: None)
    monkeypatch.setattr(research, "_stock_last", lambda symbol, service: 226.36)
    assert research._implied_open_line("NVDAUSDT", Decimal("226.5")) is None


def test_the_scored_argus_estimate_is_the_consoles_printed_implied_open(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """argus_perp is read back from `_implied_open_line`'s sentence: breaking the console breaks
    the harness, and the former in-module formula agrees with it to the printed cent."""
    from argus.lui import research

    report = oc.score(_inputs())
    replay = report["console_replay"]
    assert replay["nights_the_console_declined"] == []
    assert replay["max_abs_diff_vs_former_formula_bps"] < 1.0
    assert "puts the stock near" in replay["sample_line"]

    def broken(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("console broken")

    monkeypatch.setattr(research, "_implied_open_line", broken)
    with pytest.raises(RuntimeError, match="console broken"):
        oc.score(_inputs())


def test_the_replay_feed_is_restored_and_refuses_what_it_does_not_hold() -> None:
    from argus.market import equity_history, history, universe

    real = (history.fetch, equity_history.daily, universe.contracts)
    with oc.console_feed({"NVDAUSDT": [(7200.0, 101.0)]}, {"NVDA": [(date(2026, 1, 2), 99.0)]}):
        bars = history.fetch("NVDAUSDT", interval="1H",
                             start=datetime.fromtimestamp(0, UTC),
                             end=datetime.fromtimestamp(3600, UTC))
        assert [(b.ts, b.close) for b in bars] == [(datetime.fromtimestamp(3600, UTC),
                                                    Decimal("101.0"))]
        assert equity_history.daily("NVDA")[0].close == 99.0
        with pytest.raises(history.HistoryError):
            history.fetch("NVDAUSDT", interval="1D", start=None, end=None)
    assert (history.fetch, equity_history.daily, universe.contracts) == real


def test_claimcheck_grades_the_consoles_own_premise_check(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The replay calls `_claim_check` on the saved sentences and tape; its verdicts are the ones
    the console gave that morning, and a broken `_claim_check` breaks the harness."""
    from argus.eval import claimcheck_comparison as cc
    from argus.lui import research

    claims = json.loads((cc.DATA / "claims.json").read_text("utf-8"))
    tape = json.loads((cc.DATA / "tape.json").read_text("utf-8"))
    recorded = json.loads((cc.DATA / "argus_raw.json").read_text("utf-8"))
    replay = cc.console_answers(claims, tape)
    assert cc._agreement(claims, replay, recorded)["differ"] == []
    session = replay["NVDA"][-1]
    assert session["symbol"] == "" and "at 04:36 UTC" in session["lines"][0]
    # No claim in the set reads a price, so the zeroed price fields of the replayed tape are safe.
    for sentences in claims.values():
        for sentence in sentences:
            assert not research._FUNDING_CLAIM.search(sentence)
            assert not research._PREMIUM_CLAIM.search(sentence)
            assert not research._LIQUIDITY_CLAIM.search(sentence)

    def broken(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("console broken")

    monkeypatch.setattr(research, "_claim_check", broken)
    with pytest.raises(RuntimeError, match="console broken"):
        cc.console_answers(claims, tape)
