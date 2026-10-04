"""Round 35's judge and hostile-reviewer audits, run offline: each fix on the phrasing that
exposed it, with every live figure stubbed."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from argus.lui import account_math, premise_facts, server
from argus.lui.research import parse

CHECKS = ("dividend", "ticker", "exchange", "earnings_day", "leadership", "index_membership",
          "fee_promotion")


class TestNotional:
    def test_a_dollar_amount_is_not_a_coin_count(self) -> None:
        got = parse._unit_notional("I hold $10,000 BTC", "BTCUSDT")
        assert got is None or got[0] != Decimal(10_000) or got[1] != "units"


class TestAccountMath:
    def test_basis_points_are_not_percent(self) -> None:
        said = account_math.bps_lines("a 40bps spread on $50,000 means I'm paying 40% in costs, "
                                      "right?")
        assert said is not None and said[0].startswith("Bottom line: no — 40bps is 0.4%")
        assert "$200" in said[0]
        assert account_math.bps_lines("40bps is 0.4%, right?") is None

    def test_a_zero_volatility_sharpe_is_undefined(self) -> None:
        said = account_math.sharpe_lines("my strategy made a return of 5% last month with exactly "
                                         "0% volatility — what is my Sharpe?")
        assert said is not None and "undefined" in said[0]

    def test_a_stated_sharpe_is_computed(self) -> None:
        said = account_math.sharpe_lines("annual return 30% at 20% vol, what's the Sharpe?")
        assert said is not None and "about 1.50" in said[0]

    def test_sats_to_eth_with_gas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import position_math

        monkeypatch.setattr(position_math, "_last",
                            lambda s: {"BTCUSDT": 100_000.0, "ETHUSDT": 2_500.0}[s])
        said = account_math.sats_lines("I have 50000 sats and gas is 30 gwei — if I convert my "
                                       "sats to ETH, how much will I pay in fees?")
        assert said is not None and "about $50" in said[0] and "0.02 ETH" in said[0]
        assert "4,500,000 gwei" in said[1] and "more than the swap is worth" not in said[1]


class TestPremises:
    def test_an_index_rebalance_said_in_the_present(self) -> None:
        assert premise_facts.INDEX_CLAIM.search("the Nasdaq-100 is adding MSTR this Friday, "
                                                "replacing INTC")

    def test_a_stated_earnings_date_is_checked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import claims

        dated = {"lines": ["Bottom line: HOOD reports in 23 days, on 27 Oct, after the close."]}
        monkeypatch.setattr(server, "_answer", lambda *a, **k: dict(dated))
        monkeypatch.setattr(claims, "lines", lambda *a, **k: [])
        for name in CHECKS:
            monkeypatch.setattr(premise_facts, name, lambda *a, **k: None)
        monkeypatch.setattr(premise_facts, "past_level", lambda *a, **k: None)
        said = server._premise_lines("Robinhood (HOOD) reports Q3 earnings on November 15th — "
                                     "should I hold through it?", None, "")
        assert any("next report is on 27 Oct" in s and "not Nov 15" in s for s in said)

    def test_a_matching_date_says_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import claims

        dated: dict[str, Any] = {"lines": ["Bottom line: HOOD reports in 23 days, on 27 Oct."]}
        monkeypatch.setattr(server, "_answer", lambda *a, **k: dict(dated))
        monkeypatch.setattr(claims, "lines", lambda *a, **k: [])
        for name in CHECKS:
            monkeypatch.setattr(premise_facts, name, lambda *a, **k: None)
        monkeypatch.setattr(premise_facts, "past_level", lambda *a, **k: None)
        said = server._premise_lines("HOOD reports earnings on Oct 27, hold through it?", None,
                                     "")
        assert not any("next report" in s for s in said)
