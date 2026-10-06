"""Amounts in a reader's own currency read at a named rate (round 45 newcomer, M7)."""

from __future__ import annotations

from typing import Any

import pytest

from argus.lui.research import local_money
from argus.market import fx_rates


@pytest.fixture(autouse=True)
def rates(monkeypatch: pytest.MonkeyPatch) -> None:
    table = {"IDR": 17_913.0, "VND": 25_966.23, "INR": 96.3, "NGN": 1_331.0, "TRY": 49.16}
    monkeypatch.setattr(fx_rates, "usd_rate", lambda code: fx_rates.FxRate(
        code, table[code], "the ECB reference rate", "2026-10-05") if code in table else None)


@pytest.mark.parametrize(("text", "amount", "code"), [
    ("If I buy 1 million rupiah of bitcoin, how much profit?", 1e6, "IDR"),
    ("kalau saya beli 1 juta rupiah untung berapa", 1e6, "IDR"),
    ("tôi có 5 triệu đồng", 5e6, "VND"),
    ("Rp 500.000 di BTC", 5e5, "IDR"),
    ("Rp1,5 juta", 1.5e6, "IDR"),
    ("I have 2 lakh rupees", 2e5, "INR"),
    ("₹50,000 in ETH", 5e4, "INR"),
    ("5 crore inr", 5e7, "INR"),
    ("50000 naira", 5e4, "NGN"),
    ("3.5 million VND", 3.5e6, "VND"),
])
def test_amounts_and_their_multipliers(text: str, amount: float, code: str) -> None:
    found = local_money.amounts(text)
    assert [(a.amount, a.code) for a in found] == [(amount, code)]


@pytest.mark.parametrize("text", [
    "I won 500 dollars", "buy 100 USDT of BTC", "give it 10 try", "a 5 mil budget",
    "₿0.5 is how much", "100 pesos",
])
def test_ambiguous_words_are_left_alone(text: str) -> None:
    assert local_money.amounts(text) == []


def test_in_dollars_names_the_rate_and_never_reads_an_unrated_sum_as_dollars() -> None:
    said, lines = local_money.in_dollars("If I buy 1 juta rupiah of BTC")
    assert said == "If I buy $55.83 of BTC"
    assert lines == ["1,000,000 rupiah = about $55.83 at the ECB reference rate "
                     "(17,913.00 IDR per dollar, 2026-10-05)"]
    kept, why = local_money.in_dollars("I have 100 ringgit")
    assert kept == "I have 100 ringgit" and "not converted" in why[0]


def test_parse_restates_local_amounts_too() -> None:
    from argus.lui.research.parse import in_us_dollars

    said, lines = in_us_dollars("I have 5 triệu đồng in BTC")
    assert "$192.56" in said and any("dong" in x for x in lines)


class TestSources:
    def test_ecb_first_then_the_open_rate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.undo()
        fx_rates._CACHE.clear()
        calls: list[str] = []

        def fetch(url: str, **kw: Any) -> Any:
            calls.append(url)
            if url == fx_rates.ECB_URL:
                raise OSError("not fixed by the ECB")
            return {"result": "success", "rates": {"VND": 25_966.2},
                    "time_last_update_utc": "Tue, 06 Oct 2026 00:02:31 +0000"}

        monkeypatch.setattr(fx_rates, "fetch_json", fetch)
        rate = fx_rates.usd_rate("vnd")
        assert rate is not None and rate.per_usd == 25_966.2 and rate.as_of == "06 Oct 2026"
        assert "ExchangeRate-API" in rate.source and calls == [fx_rates.ECB_URL, fx_rates.OPEN_URL]
        assert fx_rates.usd_rate("VND") is rate  # cached
        fx_rates._CACHE.clear()

    def test_no_source_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.undo()
        fx_rates._CACHE.clear()

        def down(url: str, **kw: Any) -> Any:
            raise OSError("offline")

        monkeypatch.setattr(fx_rates, "fetch_json", down)
        assert fx_rates.usd_rate("IDR") is None
