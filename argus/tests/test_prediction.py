"""Prediction markets: only open, liquid, on-subject markets, informative ones first."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from argus.market import prediction as pm

NOW = datetime(2026, 9, 25, 4, tzinfo=UTC)


def _market(question: str, yes: float, volume: float, *, ends: str = "2026-12-31T00:00:00Z",
            active: bool = True, closed: bool = False, change: float | None = None
            ) -> dict[str, Any]:
    return {"question": question, "outcomePrices": json.dumps([str(yes), str(1 - yes)]),
            "volumeNum": volume, "endDate": ends, "active": active, "closed": closed,
            "oneDayPriceChange": change, "slug": question.lower().replace(" ", "-")[:30]}


EVENTS = [{"markets": [
    _market("Will Bitcoin reach $100,000 in September?", 0.009, 1_354_873),
    _market("Will Bitcoin reach $87,500 in September?", 0.32, 924_937, change=-0.03),
    _market("Will Bitcoin reach $90,000 in September?", 0.12, 917_138),
    _market("Will Bitcoin dip to $55,000 in September?", 0.001, 1_011_782),
    _market("Bitcoin Up or Down on September 25?", 0.5, 170),
    _market("Will bitcoiners win the election?", 0.4, 50_000),
    _market("Will Bitcoin reach $80,000 in August?", 0.99, 2_000_000, closed=True),
    _market("Will Bitcoin reach $70,000 on September 24?", 0.99, 50_000,
            ends="2026-09-24T16:00:00Z"),
]}]


def test_only_open_liquid_on_subject_markets_informative_first() -> None:
    found = pm.relevant(EVENTS, ("bitcoin", "btc"), now=NOW)
    questions = [m.question for m in found]
    assert questions[:2] == ["Will Bitcoin reach $87,500 in September?",
                             "Will Bitcoin reach $90,000 in September?"]
    # the settled-looking tails follow, busiest first
    assert questions[2:] == ["Will Bitcoin reach $100,000 in September?",
                             "Will Bitcoin dip to $55,000 in September?"]
    assert not any("Up or Down" in q or "bitcoiners" in q or "August" in q
                   or "September 24" in q for q in questions)


def test_lines_name_the_price_volume_and_close() -> None:
    lines = pm.lines_for("BTCUSDT", search=lambda term: EVENTS, now=NOW, name="BTC")
    assert lines[0] == ('Prediction market on BTC (Polymarket): "Will Bitcoin reach $87,500 in '
                        'September?" priced at 32%, -3 points in 24h — $924,937 traded, closes '
                        '2026-12-31.')
    assert len(lines) == pm.MAX_LINES


def test_a_failed_search_adds_nothing() -> None:
    def broken(term: str) -> list[dict[str, Any]]:
        raise pm.PredictionError("down")

    assert pm.lines_for("BTCUSDT", search=broken, now=NOW) == []


def test_symbols_map_to_the_names_markets_use() -> None:
    asked: list[str] = []

    def spy(term: str) -> list[dict[str, Any]]:
        asked.append(term)
        return []

    pm.markets_for("XAUUSDT", search=spy, now=NOW)
    pm.markets_for("TSLAUSDT", search=spy, now=NOW)
    pm.markets_for("CVXSTOCKUSDT", search=spy, now=NOW)
    assert asked == ["gold", "tesla", "cvx"]
