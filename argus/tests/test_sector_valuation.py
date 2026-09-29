"""A stock's valuation against its sector fund (`lui/research/sector.py`)."""

from __future__ import annotations

from typing import Any

from argus.lui.research.sector import SECTOR_FUND, sector_valuation


def _fetch(pe: Any = 39.08, pb: Any = 46.0, sector: str = "Technology") -> Any:
    def fetch(ticker: str, modules: str) -> dict[str, Any]:
        if ticker == "XLK":
            return {"topHoldings": {"equityHoldings": {
                "priceToEarnings": {"raw": 0.03029}, "priceToBook": {"raw": 0.08767}}}}
        return {"assetProfile": {"sector": sector},
                "summaryDetail": {"trailingPE": {"raw": pe} if pe else {}},
                "defaultKeyStatistics": {"priceToBook": {"raw": pb}}}
    return fetch


def test_the_fund_yield_is_inverted_and_set_beside_the_stock() -> None:
    found = sector_valuation("AAPL", fetch=_fetch())
    assert found is not None
    line, source = found
    assert line.startswith("Against its sector (Technology, measured by the SPDR fund XLK)")
    assert "trailing P/E 39.1 against 33.0 (+18%)" in line
    assert "P/B 46.0 against 11.4 (4.0 times)" in line
    assert "XLK" in source.detail


def test_a_missing_pe_leaves_the_other_figure() -> None:
    found = sector_valuation("COIN", fetch=_fetch(pe=None))
    assert found is not None and "trailing P/E" not in found[0] and "P/B" in found[0]


def test_an_unmapped_sector_or_a_fund_gives_nothing() -> None:
    assert sector_valuation("QQQ", fetch=_fetch(sector="")) is None
    assert sector_valuation("X", fetch=_fetch(sector="Crypto")) is None


def test_every_yahoo_sector_has_a_fund() -> None:
    assert len(SECTOR_FUND) == 11 and len(set(SECTOR_FUND.values())) == 11
