"""A saved book written as amounts is priced, not read as an equal-weight list of names.

Found on the live console on 2026-09-26: My book "long 2 NVDAUSDT, long 1 TSLAUSDT" came back as
50% NVDA and 50% TSLA, so every risk figure rested on a book the trader does not hold. Prices are
pinned here; the live path reads Bitget's tickers once per book text.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from argus.lui import provenance, research
from argus.lui.research import (
    book_pricing_note,
    parse_book,
    priced_book,
    saved_book_lines,
    split_cash,
)

PRICES = {"NVDAUSDT": 200.0, "TSLAUSDT": 400.0, "AAPLUSDT": 250.0, "BTCUSDT": 100_000.0,
          "ETHUSDT": 4_000.0}


@pytest.fixture(autouse=True)
def pinned_prices(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    calls = [0]

    def prices() -> dict[str, float]:
        calls[0] += 1
        return dict(PRICES)

    monkeypatch.setattr(research, "_last_prices", prices)
    research._PRICED.clear()
    yield calls
    research._PRICED.clear()


def book(text: str) -> tuple[dict[str, float], float]:
    held, cash = split_cash(text, parse_book(text))
    return {s: round(w, 4) for s, w in held.items()}, round(cash, 4)


class TestAmountsArePriced:
    def test_contract_counts_are_valued_at_the_last_price(self) -> None:
        # 2 x 200 = 400 and 1 x 400 = 400: equal value, so equal weight for the right reason
        assert book("long 2 NVDAUSDT, long 1 TSLAUSDT") == ({"NVDAUSDT": 0.5, "TSLAUSDT": 0.5},
                                                              0.0)

    def test_unequal_counts_are_not_read_as_equal_weight(self) -> None:
        held, _ = book("long 4 NVDAUSDT, long 1 TSLAUSDT")
        assert held == {"NVDAUSDT": pytest.approx(0.6667, abs=1e-4),
                        "TSLAUSDT": pytest.approx(0.3333, abs=1e-4)}

    def test_dollar_values_need_no_price(self, pinned_prices: list[int]) -> None:
        assert book("$20k NVDA, $10k TSLA") == ({"NVDAUSDT": 0.6667, "TSLAUSDT": 0.3333}, 0.0)
        assert pinned_prices[0] == 0

    def test_cash_stated_as_an_amount_stays_cash(self) -> None:
        assert book("$20k NVDA, $10k TSLA, $10k cash") == (
            {"NVDAUSDT": 0.5, "TSLAUSDT": 0.25}, 0.25)
        assert book("0.1 BTC, 10,000 USDT") == ({"BTCUSDT": 0.5}, 0.5)

    def test_share_counts_and_units(self) -> None:
        assert book("200 shares of AAPL, 50 shares TSLA") == (
            {"AAPLUSDT": 0.7143, "TSLAUSDT": 0.2857}, 0.0)
        assert book("2 contracts of ETHUSDT and 0.08 BTC") == (
            {"ETHUSDT": 0.5, "BTCUSDT": 0.5}, 0.0)

    def test_a_short_is_a_negative_weight(self) -> None:
        assert book("long 2 NVDAUSDT, short 1 TSLAUSDT") == (
            {"NVDAUSDT": 0.5, "TSLAUSDT": -0.5}, 0.0)

    def test_leverage_is_not_a_holding(self) -> None:
        assert book("3x leverage on 2 NVDAUSDT") == ({"NVDAUSDT": 1.0}, 0.0)


class TestWhatDidNotChange:
    def test_weights_still_win(self, pinned_prices: list[int]) -> None:
        assert book("40% NVDA, 60% TSLA") == ({"NVDAUSDT": 0.4, "TSLAUSDT": 0.6}, 0.0)
        assert book("NVDA 50%, cash 50%") == ({"NVDAUSDT": 0.5}, 0.5)
        assert pinned_prices[0] == 0

    def test_bare_names_are_still_equal_weight(self) -> None:
        assert book("NVDA, TSLA") == ({"NVDAUSDT": 0.5, "TSLAUSDT": 0.5}, 0.0)
        assert priced_book("NVDA, TSLA") is None

    def test_a_budget_is_not_an_amount(self) -> None:
        assert book("40% NVDA, 30% MSFT, 30% AAPL, risk budget 20%")[0] == {
            "NVDAUSDT": 0.4, "MSFTUSDT": 0.3, "AAPLUSDT": 0.3}


class TestNothingIsSilent:
    def test_an_unpriceable_count_is_left_out_and_said(self, monkeypatch: pytest.MonkeyPatch
                                                       ) -> None:
        monkeypatch.setattr(research, "_last_prices", lambda: {})
        research._PRICED.clear()
        # stated amounts that cannot be priced leave the book empty, never equal weight
        assert parse_book("long 2 NVDAUSDT, long 1 TSLAUSDT") == {}
        priced = priced_book("long 2 NVDAUSDT, long 1 TSLAUSDT")
        assert priced is not None
        assert any("no live price for NVDA" in line for line in priced.lines)

    def test_the_saved_book_answer_shows_each_conversion(self) -> None:
        lines = saved_book_lines("long 2 NVDAUSDT, long 1 TSLAUSDT, $5k cash")
        assert lines[0].startswith("Actionable: your saved book reads as 7% NVDA, 7% TSLA")
        assert "with 86% in cash" in lines[0]
        assert lines[1] == ("Priced at Bitget's last price: 2 NVDA = $400 (200.00), "
                            "1 TSLA = $400 (400.00), $5,000 cash.")
        assert provenance.label(lines[1]) == "live"

    def test_every_saved_book_note_carries_the_prices(self) -> None:
        note = book_pricing_note("long 2 NVDAUSDT")
        assert note == "; priced at Bitget's last price: 2 NVDA = $400 (200.00)"
        assert book_pricing_note("40% NVDA, 60% TSLA") == ""

    def test_one_book_is_priced_once_per_minute(self, pinned_prices: list[int]) -> None:
        for _ in range(5):
            parse_book("long 2 NVDAUSDT, long 1 TSLAUSDT")
        assert pinned_prices[0] == 1
