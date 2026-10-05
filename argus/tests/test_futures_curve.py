"""Commodity futures curve answers (`lui/research/futures_curve.py`): contract-month symbols, the
freshness filter, spreads and roll yield, the month-ago comparison, the Bitget perpetuals and the
wiring condition, from fake Yahoo charts, offline."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from argus.lui.research import futures_curve as fc
from argus.truth.failures import ErrorKind, RpcError

M = fc.MINUS
TODAY = date(2026, 10, 6)
LAST = date(2026, 10, 5)
Book = dict[str, tuple[float, date, float | None]]


def _stamp(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, 20, tzinfo=UTC).timestamp())


def _payload(price: float, last: date, then: float | None = None) -> dict[str, Any]:
    """A chart whose closes are ``then`` more than 20 days before ``last`` and ``price`` after."""
    days = [last - timedelta(days=n) for n in range(45, -1, -1)]
    days = [d for d in days if d.weekday() < 5]
    closes = [(then if then is not None and (last - d).days > 20 else price) for d in days]
    closes[-1] = price
    return {"chart": {"result": [{"meta": {"regularMarketPrice": price,
                                           "regularMarketTime": _stamp(last)},
                                  "timestamp": [_stamp(d) for d in days],
                                  "indicators": {"quote": [{"close": closes}]}}]}}


_GONE: dict[str, Any] = {"chart": {"result": None, "error": {"code": "Not Found"}}}


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    fc._cache.clear()
    yield
    fc._cache.clear()


def _world(monkeypatch: pytest.MonkeyPatch, book: Book) -> list[str]:
    asked: list[str] = []

    def chart(symbol: str) -> dict[str, Any]:
        asked.append(symbol)
        if symbol not in book:
            return _GONE
        price, last, then = book[symbol]
        return _payload(price, last, then)

    monkeypatch.setattr(fc, "_chart", chart)
    return asked


def _wti_book() -> Book:
    prices = {"X26": 89.0, "Z26": 87.0, "F27": 86.0, "G27": 85.0, "H27": 84.0, "J27": 83.0,
              "K27": 82.0, "M27": 81.0, "N27": 80.0, "Q27": 79.0, "U27": 78.0, "V27": 77.0}
    book: Book = {f"CL{k}.NYM": (v, LAST, None) for k, v in prices.items()}
    book["CLX26.NYM"] = (89.0, LAST, 88.0)
    book["CLJ27.NYM"] = (83.0, LAST, 76.5)
    book["CLQ26.NYM"] = (84.99, date(2026, 7, 21), None)  # an expired month still answering
    return book


def _perps(symbols: Sequence[str]) -> dict[str, fc.Perp]:
    table = {"CLUSDT": fc.Perp("CLUSDT", 89.1, 0.0, 4),
             "BZUSDT": fc.Perp("BZUSDT", 100.0, -0.00004, 4),
             "XAUUSDT": fc.Perp("XAUUSDT", 4143.3, 0.000148, 4),
             "NATGASUSDT": fc.Perp("NATGASUSDT", 3.068, 0.0004, 4)}
    return {s: table[s] for s in symbols if s in table}


def _line(out: list[str] | None, start: str) -> str:
    assert out is not None
    return next(line for line in out if line.startswith(start))


def test_symbols_use_the_month_codes_and_exchange() -> None:
    wti, gold = fc.COMMODITIES["wti"], fc.COMMODITIES["gold"]
    assert fc.symbol_for(wti, 2026, 11) == "CLX26.NYM"
    assert fc.symbol_for(wti, 2027, 1) == "CLF27.NYM"
    assert fc.symbol_for(gold, 2026, 12) == "GCZ26.CMX"
    assert fc.CODES == "FGHJKMNQUVXZ"


def test_chain_skips_expired_and_stale_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = _world(monkeypatch, _wti_book())
    legs = fc.chain(fc.COMMODITIES["wti"], TODAY)
    labels = [leg.label for leg in legs]
    assert labels[0] == "Nov 2026"  # Oct 2026 is gone
    assert "CLV26.NYM" in asked
    assert all(leg.series.last == LAST for leg in legs)
    stale = _wti_book()
    stale["CLZ26.NYM"] = (87.0, date(2026, 9, 28), None)
    fc._cache.clear()
    _world(monkeypatch, stale)
    assert "Dec 2026" not in [leg.label for leg in fc.chain(fc.COMMODITIES["wti"], TODAY)]


def test_backwardation_spreads_and_roll_yield(monkeypatch: pytest.MonkeyPatch) -> None:
    _world(monkeypatch, _wti_book())
    monkeypatch.setattr(fc, "perps_for", _perps)
    out = fc.lines("Is WTI crude in backwardation or contango?", today=TODAY)
    assert out is not None
    assert out[0].startswith("Bottom line: WTI crude is in backwardation")
    curve = _line(out, "WTI crude curve")
    assert f"2nd Dec 2026 87.00 ({M}2.00, {M}2.2%)" in curve
    assert f"6th Apr 2027 83.00 ({M}6.00, {M}6.7%)" in curve
    assert f"12th Oct 2027 77.00 ({M}12.00, {M}13.5%)" in curve
    roll = (89.0 / 87.0) ** 12 - 1
    assert f"2nd month +{roll * 100:.1f}%" in _line(out, "WTI crude roll yield")
    assert out[-1].startswith("Data: ") and out[-1].endswith("Not advice.")


def test_month_ago_reads_the_same_two_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    _world(monkeypatch, _wti_book())
    monkeypatch.setattr(fc, "perps_for", _perps)
    out = fc.lines("wti crude curve", today=TODAY)
    ago = _line(out, "A month ago")
    # Nov 2026 was 88.0 and Apr 2027 76.5 a month ago: -13.1%, now -6.7%: flatter.
    assert f"{M}13.1% (same two contracts), now {M}6.7%" in ago
    assert "flattening" in ago


def test_contango_and_flip(monkeypatch: pytest.MonkeyPatch) -> None:
    book: Book = {f"GC{k}.CMX": (v, LAST, None) for k, v in
                  {"V26": 4100.0, "X26": 4110.0, "Z26": 4130.0, "G27": 4150.0, "J27": 4160.0,
                   "M27": 4190.0, "Q27": 4230.0}.items()}
    book["GCV26.CMX"] = (4100.0, LAST, 4090.0)
    book["GCG27.CMX"] = (4150.0, LAST, 4060.0)  # a month ago: below the front, backwardated
    _world(monkeypatch, book)
    monkeypatch.setattr(fc, "perps_for", _perps)
    out = fc.lines("Is the gold futures curve in contango?", today=TODAY)
    assert out is not None and out[0].startswith("Bottom line: gold is in contango")
    assert f"roll yield about {M}" in out[0]
    assert "flipped from backwardation to contango" in _line(out, "A month ago")


def test_mixed_shape_is_said_not_forced(monkeypatch: pytest.MonkeyPatch) -> None:
    prices = {"X26": 3.0, "Z26": 3.3, "F27": 3.7, "G27": 3.4, "H27": 2.7, "J27": 2.6, "K27": 2.7}
    _world(monkeypatch, {f"NG{k}.NYM": (v, LAST, None) for k, v in prices.items()})
    monkeypatch.setattr(fc, "perps_for", _perps)
    out = fc.lines("natural gas curve shape", today=TODAY)
    assert out is not None and "is not one shape" in out[0]
    assert "+10.0% to Dec 2026" in out[0] and f"{M}13.3% to Apr 2027" in out[0]
    assert any("heating months" in line for line in out)


def test_off_month_gap_uses_the_nearest_traded_month(monkeypatch: pytest.MonkeyPatch) -> None:
    book: Book = {f"SI{k}.CMX": (v, LAST, None) for k, v in
                  {"V26": 61.0, "Z26": 61.4, "H27": 62.1, "N27": 63.8}.items()}
    _world(monkeypatch, book)
    monkeypatch.setattr(fc, "perps_for", _perps)
    out = fc.lines("silver term structure", today=TODAY)
    curve = _line(out, "silver curve")
    assert "2nd Dec 2026" in curve and "2 months out, nearest traded month" in curve
    assert "6th Mar 2027 62.10 (+1.10, +1.8%)" in curve  # Mar is exactly five months out
    assert "12th" not in curve  # nothing traded within a month of Sep 2027


def test_bitget_perps_and_funding(monkeypatch: pytest.MonkeyPatch) -> None:
    _world(monkeypatch, _wti_book())
    monkeypatch.setattr(fc, "perps_for", _perps)
    out = fc.lines("Which Bitget contracts could I use to trade oil?", today=TODAY)
    assert out is not None
    assert out[0].startswith("Bottom line: Bitget lists these USDT perpetuals for WTI crude and "
                             "Brent crude: CLUSDT, BZUSDT")
    wti = _line(out, "Bitget perpetuals for WTI crude")
    assert "CLUSDT (WTI) 89.10" in wti and "tracks the front month" in wti
    brent = _line(out, "Bitget perpetuals for Brent crude")
    assert f"funding {M}0.0040% per 4h, {M}8.8% a year" in brent
    assert any("no expiry" in line and "funding rate plays that role" in line for line in out)
    # The fake book holds no Brent contracts, and the answer says the curve could not be read.
    assert "the Brent crude futures curve could not be read just now" in out[0]


def test_unreadable_source_is_one_honest_line(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(symbol: str) -> dict[str, Any]:
        raise RpcError(ErrorKind.TRANSPORT, "down")

    monkeypatch.setattr(fc, "_chart", down)
    monkeypatch.setattr(fc, "perps_for", _perps)
    out = fc.lines("Is the gold futures curve in contango?", today=TODAY)
    assert out is not None
    assert out[0] == "Bottom line: the gold futures curve could not be read just now."
    assert out[-1].endswith("Not advice.")


def test_missing_perpetual_is_said(monkeypatch: pytest.MonkeyPatch) -> None:
    _world(monkeypatch, _wti_book())
    monkeypatch.setattr(fc, "perps_for", lambda symbols: {})
    out = fc.lines("wti backwardation", today=TODAY)
    assert any(line.startswith("Bitget lists no USDT perpetual for WTI crude")
               for line in out or [])


@pytest.mark.parametrize("text", [
    "How is the oil market positioned: is crude in backwardation or contango?",
    "Is the gold futures curve in contango?",
    "natural gas curve shape",
    "Which Bitget contracts could I use to trade oil?",
    "copper term structure",
    "what is the brent futures curve doing",
    "silver roll yield",
])
def test_wiring_accepts_commodity_curve_questions(text: str) -> None:
    assert fc.asked(text)


@pytest.mark.parametrize("text", [
    "What is the yield curve doing?",
    "Is the Treasury curve inverted?",
    "bond curve steepening",
    "Is the Bund yield curve in contango",
    "VIX term structure contango",
    "Is bitcoin futures in contango?",
    "gold price today",
    "oil price",
    "What is the Fed going to do?",
    "natural gas storage report",
    "how do I trade stocks on Bitget?",
    "",
])
def test_wiring_refuses_other_questions(text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    assert not fc.asked(text)

    def forbidden(symbol: str) -> dict[str, Any]:
        raise AssertionError("a refused question must not fetch")

    monkeypatch.setattr(fc, "_chart", forbidden)
    assert fc.lines(text, today=TODAY) is None


def test_commodity_words() -> None:
    def names(text: str) -> list[str]:
        return [c.key for c in fc.commodities_asked(text)]

    assert names("oil curve") == ["wti", "brent"]
    assert names("brent contango") == ["brent"]
    assert names("WTI and gold") == ["wti", "gold"]
    assert names("commodity curves") == list(fc.COMMODITIES)
    assert names("Neo GAS token") == []
