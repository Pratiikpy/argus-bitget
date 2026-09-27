"""What's trending, and which of it trades on Bitget (`lui/trending.py`), offline."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pytest

from argus.lui import trending as tr
from argus.lui.skillroute import Routed
from argus.market.skills import Health


@dataclass
class _Ticker:
    last: Decimal
    change_24h: Decimal
    funding_rate: Decimal


def _routed(tool: str, via: str, payload: Any, health: Health = Health.TIMEOUT) -> Routed:
    return Routed(tool, "trending", "market-intel", Health.OK if via == "skill" else health,
                  "detail", via, payload, upstream="CoinGecko", asked_at="03:44 UTC")


COINS = {"trending": [
    {"symbol": "qnt", "name": "Quant", "market_cap_rank": 44, "price_usd": 170.0},
    {"symbol": "edel", "name": "Edel", "market_cap_rank": 920, "price_usd": 0.1},
    {"symbol": "pons", "name": "Pons", "market_cap_rank": 123, "price_usd": 2.0},
]}
BOARD = {"QNTUSDT": _Ticker(Decimal("171"), Decimal("0.72"), Decimal("-0.00142")),
         "PONSUSDT": _Ticker(Decimal("0.05"), Decimal("-0.02"), Decimal("0.0001"))}


def _route(coins: Routed, social: Routed) -> Any:
    def route(tool: str, action: str, args: Any = None, **kwargs: Any) -> Routed:
        return coins if tool == "crypto_market" else social
    return route


@pytest.mark.parametrize("text", ["what's trending right now?", "what are the hot coins",
                                  "what is the crowd talking about", "trending coins today",
                                  "现在什么币热门"])
def test_trending_questions_are_recognised(text: str) -> None:
    assert tr.asks_for_trending(text)


@pytest.mark.parametrize("text", ["is NVDA trending up?", "should I add 15% TSLA",
                                  "what's the trend in QQQ"])
def test_other_questions_are_left_alone(text: str) -> None:
    assert not tr.asks_for_trending(text)


def test_both_reply_shapes_are_read() -> None:
    gecko = {"coins": [{"item": {"symbol": "zec", "name": "Zcash", "market_cap_rank": 9,
                                 "data": {"price": 40.5}}}]}
    assert tr.coins_from(gecko) == [{"symbol": "ZEC", "name": "Zcash", "rank": 9,
                                     "price": 40.5}]
    assert tr.coins_from(COINS)[0]["symbol"] == "QNT"
    assert tr.coins_from({"unexpected": 1}) == [] and tr.coins_from([1, 2]) == []


def test_listing_moves_and_a_ticker_collision_are_stated() -> None:
    lines, sources, data = tr.trending(
        "what's trending", route=_route(_routed("crypto_market", "mirror", COINS),
                                        _routed("social_trending", "none", None)),
        tickers=lambda: BOARD, mirror=lambda a: {})
    assert lines[0].startswith("Bottom line: 1 of the 3 coins most searched on CoinGecko trade "
                               "as USDT perpetuals on Bitget; the biggest 24-hour move among "
                               "them is QNT at +72.0%.")
    assert "QNT (Quant, market-cap rank 44): on Bitget +72.0% in 24h, funding -14.2bp." in lines
    assert "EDEL (Edel, market-cap rank 920): no Bitget USDT perpetual." in lines
    assert any(line.startswith("Not matched: PONS (CoinGecko $2, Bitget 0.05)") for line in lines)
    assert any("social_trending did not answer in time" in line for line in lines)
    assert data["listed"] == 1 and data["coins_via"] == "mirror"
    assert sources[0].ref == "CoinGecko"


def test_a_skill_reply_in_an_unknown_shape_falls_to_coingecko_and_says_so() -> None:
    lines, sources, _ = tr.trending(
        "hot coins", route=_route(_routed("crypto_market", "skill", {"odd": []}),
                                  _routed("social_trending", "none", None)),
        tickers=lambda: BOARD, mirror=lambda a: COINS)
    assert sources[0].ref == "CoinGecko /search/trending"
    assert "shape this console does not read" in sources[0].detail
    assert lines[0].startswith("Bottom line: 1 of the 3")


def test_nothing_readable_is_said_plainly() -> None:
    lines, _, data = tr.trending(
        "what's trending", route=_route(_routed("crypto_market", "none", None),
                                        _routed("social_trending", "none", None)),
        tickers=lambda: {}, mirror=lambda a: {})
    assert lines[0].startswith("Missing: the trending list could not be read just now")
    assert any(line.startswith("Missing: Bitget's ticker board did not answer") for line in lines)
    assert data["trending"] == []


def test_xueqiu_titles_show_when_the_skill_answers() -> None:
    social = _routed("social_trending", "skill", {"items": [{"title": "茅台"}, {"x": 1}]})
    lines, sources, _ = tr.trending(
        "what's trending", route=_route(_routed("crypto_market", "mirror", COINS), social),
        tickers=lambda: BOARD, mirror=lambda a: {})
    assert "Xueqiu's hot list (bitget-signal social_trending): 茅台." in lines
    assert any(s.ref == "bitget-signal social_trending.trending" for s in sources)


@pytest.mark.parametrize("text", ["what is hot in the stock market", "what is trending in tech "
                                  "stocks", "what's trending on the Nasdaq"])
def test_a_stock_question_is_not_answered_with_a_crypto_list(text: str) -> None:
    assert not tr.asks_for_trending(text)
