"""The Polymarket crowd's record (`market/crowd_record.py`, build-list 4.6), scored by hand on
stubbed resolved markets: no network."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from argus.lui import server
from argus.market import crowd_record as cr

END = datetime(2026, 10, 3, 16, tzinfo=UTC)


def _market(question: str, outcome: str, token: str, *, volume: float = 50_000.0,
            ends: datetime = END) -> dict[str, Any]:
    return {"question": question, "outcomePrices": outcome, "closed": True,
            "clobTokenIds": json.dumps([token, token + "n"]), "volumeNum": volume,
            "endDate": ends.isoformat().replace("+00:00", "Z")}


def test_a_family_drops_dates_and_figures() -> None:
    assert cr.family("Bitcoin above ___ on October 3?") == cr.family(
        "Bitcoin above ___ on February 23?")
    assert cr.label(cr.family("Bitcoin above ___ on October 3?")) == \
        "Bitcoin above ___ on <date>?"


def test_only_a_clean_resolution_is_scored() -> None:
    events = [{"title": "Bitcoin above ___ on October 3?", "markets": [
        _market("Will Bitcoin be above $74,000 on October 3?", '["1", "0"]', "a"),
        _market("Will Bitcoin be above $76,000 on October 3?", '["0", "1"]', "b"),
        # a 50-50 resolution would favour every forecast near one half (kolberg,
        # backtest_utils.py:316-320), so it is left out rather than scored as 0.5
        _market("Will Bitcoin be above $78,000 on October 3?", '["0.5", "0.5"]', "c"),
        _market("Will Bitcoin be above $80,000 on October 3?", '["1", "0"]', "d", volume=500),
        _market("Will Ethereum be above $4,000 on October 3?", '["1", "0"]', "e")]}]
    got = cr.resolved_from(events, ("bitcoin",))
    assert [(m.token, m.outcome) for m in got] == [("a", 1.0), ("b", 0.0)]


def test_the_price_a_day_before_the_end() -> None:
    at = int((END - timedelta(hours=24)).timestamp())
    path = [(at - 7200, 0.40), (at - 60, 0.55), (at + 60, 0.90)]
    assert cr.price_at_lead(path, END) == 0.55
    # a market that only traded after the lead has no forecast to score
    assert cr.price_at_lead([(at + 60, 0.9)], END) is None
    assert cr.price_at_lead([(at - 7 * 3600, 0.9)], END) is None


def _snapshot(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    rows = [{"question": f"Will Bitcoin be above ${70 + i},000 on October 3?",
             "family": cr.family("Bitcoin above ___ on October 3?"), "token": f"t{i}",
             "ends": (END - timedelta(days=i)).isoformat(), "outcome": y, "volume": 20_000.0,
             "price": p} for i, (p, y) in enumerate(pairs)]
    return {"series": {"bitcoin-above-on": rows}}


def test_brier_and_calibration_match_a_hand_count() -> None:
    pairs = [(0.3, 0.0), (0.6, 1.0), (0.7, 1.0), (0.4, 1.0), (0.99, 1.0), (0.02, 0.0)]
    record = cr.record_for("BTCUSDT", snapshot=_snapshot(pairs), event=lambda slug: [],
                           today=END + timedelta(days=1))
    assert record is not None and record.n == 6
    by_hand = (0.09 + 0.16 + 0.09 + 0.36 + 0.0001 + 0.0004) / 6
    assert record.brier == pytest.approx(by_hand)
    assert record.contested().n == 4
    assert record.base_rate == pytest.approx(4 / 6)
    assert record.base_rate_brier == pytest.approx((4 * (1 / 3) ** 2 + 2 * (2 / 3) ** 2) / 6)
    bands = {(lo, hi): (n, hit) for lo, hi, n, _, hit in record.bands()}
    assert bands[(0.3, 0.5)] == (2, 0.5) and bands[(0.9, 1.0)] == (1, 1.0)


def test_a_close_call_record_is_tested_against_a_coin() -> None:
    # twelve contested markets the crowd called at 70% and got right ten times
    pairs = [(0.7, 1.0)] * 10 + [(0.7, 0.0)] * 2
    snapshot = _snapshot(pairs)
    record = cr.record_for("BTCUSDT", snapshot=snapshot, event=lambda slug: [],
                           today=END + timedelta(days=1))
    assert record is not None
    gains = [0.25 - 0.09] * 10 + [0.25 - 0.49] * 2
    mean = sum(gains) / 12
    sd = math.sqrt(sum((g - mean) ** 2 for g in gains) / 11)
    assert cr.against_coin(record)[1] == pytest.approx(mean / (sd / math.sqrt(12)))
    said = cr.lines("BTCUSDT", snapshot=snapshot, event=lambda slug: [],
                    today=END + timedelta(days=1))
    assert said[0].startswith("Bottom line: on close calls — the 12 “Bitcoin above ___ on "
                              "<date>?” markets")
    assert "Brier score is 0.157" in said[0]  # (10 x 0.09 + 2 x 0.49) / 12
    assert "better than a coin flip, by more than chance would give (t = 2.1)" in said[0]
    # eight close calls are too few to judge, and the answer says so instead of a verdict
    few = cr.lines("BTCUSDT", snapshot=_snapshot(pairs[4:]), event=lambda slug: [],
                   today=END + timedelta(days=1))
    assert "only 8 of them were close calls" in few[0]


def test_new_days_are_read_past_the_snapshot() -> None:
    asked: list[str] = []

    def event(slug: str) -> list[dict[str, Any]]:
        asked.append(slug)
        if slug == "bitcoin-above-on-october-4-2026":
            return [{"title": "Bitcoin above ___ on October 4?", "markets": [_market(
                "Will Bitcoin be above $74,000 on October 4?", '["1", "0"]', "new",
                ends=END + timedelta(days=1))]}]
        return []

    at = int((END + timedelta(days=1) - timedelta(hours=24)).timestamp())
    record = cr.record_for("BTCUSDT", snapshot=_snapshot([(0.5, 1.0)]), event=event,
                           path=lambda token, ends, lead: [(at - 60, 0.8)],
                           today=END + timedelta(days=2))
    assert record is not None and record.n == 2
    assert {s.market.token for s in record.scored} == {"t0", "new"}
    assert "bitcoin-above-on-october-4-2026" in asked


def test_a_name_with_no_series_is_said(monkeypatch: pytest.MonkeyPatch) -> None:
    said = server._crowd_record_lines("how accurate has Polymarket been on Nvidia?", [])
    assert said is not None and "NVDA has no such series" in said[0]


def test_the_console_routes_the_question(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, Any]] = []

    def lines(symbol: str, *, question_family: str | None = None) -> list[str]:
        seen.append((symbol, question_family))
        return ["Bottom line: scored.", "Data: stub."]

    monkeypatch.setattr(cr, "lines", lines)
    assert server._crowd_record_lines("How accurate has Polymarket been on ETH price calls?",
                                      []) is not None
    assert seen[-1] == ("ETHUSDT", None)
    # "can I trust those odds?" after a Polymarket answer scores that name's crowd
    assert server._crowd_record_lines(
        "can I trust those odds?",
        ["What's Polymarket saying about bitcoin hitting 150k this year?"]) is not None
    assert seen[-1][0] == "BTCUSDT"
    assert server._crowd_record_lines("can I trust those odds?", ["what is BTC at?"]) is None
    assert server._crowd_record_lines("how accurate is your BTC forecast?", []) is None
    assert server._crowd_record_lines("is the prediction market crowd any good at calling where "
                                      "SOL closes?", []) is not None
    assert seen[-1][0] == "SOLUSDT"


def test_the_rung_nearest_an_unlisted_level_leads(monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.market import prediction

    def market(question: str, price: float) -> prediction.Market:
        return prediction.Market(question=question, yes_price=price, change_24h=None,
                                 volume=50_000.0, ends="2026-11-01", url="", token="")

    monkeypatch.setattr(prediction, "markets_for", lambda symbol: [
        market("Will Ethereum dip to $2,100 in October?", 0.12),
        market("Will Ethereum reach $3,600 in October?", 0.05),
        market("Will Ethereum reach $3,000 in October?", 0.40)])
    said = server._polymarket_lines("What's Polymarket saying about ETH hitting 5k?", [])
    assert said is not None and "reach $3,600" in said[0]
    assert said[1].startswith("No open market above the volume floor names 5,000")
