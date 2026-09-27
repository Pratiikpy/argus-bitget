"""DeFi TVL and Ethereum gas, Skill first (`lui/onchain.py`), offline."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pytest

from argus.lui import onchain
from argus.lui.skillroute import Routed
from argus.market.skills import Health


@dataclass
class _Ticker:
    last: Decimal
    change_24h: Decimal


def _routed(tool: str, action: str, via: str, payload: Any) -> Routed:
    return Routed(tool, action, "market-intel", Health.OK if via == "skill" else Health.TIMEOUT,
                  "detail", via, payload, upstream="DeFiLlama", asked_at="03:50 UTC")


TVL = {"protocols": [
    {"name": "Lido", "category": "Liquid Staking", "symbol": "LDO", "tvl_usd": 26.5e9,
     "change_1d_pct": 0.4},
    {"name": "WBTC", "category": "Bridge", "symbol": "-", "tvl_usd": 9.8e9,
     "change_1d_pct": None},
]}
GAS = {"base_fee_gwei": 0.06, "priority_fee_gwei": {"low": 0.0, "median": 0.01, "high": 1.0},
       "gas_used_ratio": [0.5, 0.52, 0.54]}
BOARD = {"LDOUSDT": _Ticker(Decimal("1.2"), Decimal("0.031")),
         "ETHUSDT": _Ticker(Decimal("2700"), Decimal("0.01"))}


@pytest.mark.parametrize(("text", "tvl", "gas"), [
    ("what are the top DeFi protocols by TVL?", True, False),
    ("total value locked ranking", True, False),
    ("how much is ETH gas right now?", False, True),
    ("gas fees today", False, True),
    ("should I add 15% TSLA", False, False),
    ("natural gas price", False, False),
])
def test_questions_are_recognised(text: str, tvl: bool, gas: bool) -> None:
    assert onchain.asks_for_tvl(text) is tvl
    assert onchain.asks_for_gas(text) is gas


def test_tvl_ranks_and_checks_the_board() -> None:
    lines, sources, data = onchain.tvl(
        "top defi", route=lambda *a, **k: _routed("defi_analytics", "tvl_rank", "mirror", TVL),
        tickers=lambda: BOARD)
    assert lines[0] == ("Bottom line: the ten largest DeFi protocols hold $36bn, Lido the most "
                        "at $26.5bn; 1 of them have a token under the same ticker on Bitget's "
                        "perpetual board (LDO).")
    assert ("Lido (Liquid Staking): $26.5bn, +0.4% TVL in a day; a Bitget perpetual under the "
            "same ticker (LDOUSDT) moved +3.1% in 24h.") in lines
    assert "WBTC (Bridge): $9.8bn." in lines
    assert sources[0].ref == "DeFiLlama" and data["tradable"] == ["LDO"]


def test_gas_prices_a_transfer_in_eth_and_dollars() -> None:
    lines, _, data = onchain.gas(
        "gas now", route=lambda *a, **k: _routed("network_status", "eth_gas", "mirror", GAS),
        tickers=lambda: BOARD)
    assert data["transfer_eth"] == pytest.approx(0.07e-9 * 21_000)
    assert lines[0].startswith("Bottom line: Ethereum's base fee is 0.06 gwei and the median tip "
                               "0.01 gwei, so a plain ETH transfer costs about 0.00000147 ETH "
                               "(under a cent, $0.0040 at Bitget's ETHUSDT 2,700)")
    assert any("52% full" in line and "rising" in line for line in lines)


def test_a_skill_reply_in_another_shape_falls_to_the_mirror_and_says_so() -> None:
    lines, sources, _ = onchain.gas(
        "gas now", route=lambda *a, **k: _routed("network_status", "eth_gas", "skill",
                                                 {"odd": 1}),
        tickers=lambda: {}, direct=lambda a: GAS)
    assert "shape this console does not read" in sources[0].detail
    assert lines[0].startswith("Bottom line: Ethereum's base fee is 0.06 gwei")
    assert "$" not in lines[0]


def test_nothing_readable_is_said_plainly() -> None:
    lines, sources, _ = onchain.tvl(
        "tvl", route=lambda *a, **k: _routed("defi_analytics", "tvl_rank", "none", None),
        tickers=lambda: {})
    assert lines[0].startswith("Missing: the TVL ranking could not be read just now")
    assert sources == []
