"""Crypto market-structure answers (`lui/research/crypto_structure.py`): the six round-43 judge
questions (bitcoin dominance, total stablecoin supply, per-chain stablecoin change, Aave against
Compound, spot against perpetual liquidity on Bitget, crypto event calendar), read from fake
sources shaped like the ones called live on 2026-10-06, offline. Every expected figure is worked
by hand in the comment beside it."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from argus.lui.research import crypto_structure as cs
from argus.truth import http
from argus.truth.failures import ErrorKind, RpcError

DAY = 86400
NOW = 1_791_000_000.0


def _text(out: list[str] | None) -> str:
    assert out is not None
    return "\n".join(out)


def _frame(out: list[str] | None) -> None:
    """The shape every answer promises: a bottom line first, the sources and the advice line
    last."""
    assert out is not None
    assert out[0].startswith("Bottom line: ")
    assert out[-1].startswith("Data: ")
    assert out[-1].endswith("Not advice.")


# -- the period a question asks about ----------------------------------------------------------


@pytest.mark.parametrize(("question", "days"), [
    ("what happened this month", 30), ("over the last 30 days", 30), ("in the past week", 7),
    ("the next two weeks", 14), ("over 90 days", 90), ("this quarter", 90),
    ("this year", 365), ("in the last 24 hours", 1), ("in 3 weeks", 21), ("how is it", 30),
])
def test_window_days(question: str, days: int) -> None:
    assert cs.window_days(question) == days


# -- 1. bitcoin dominance ----------------------------------------------------------------------


def _market_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin", "market_cap": 600.0,
         "circulating_supply": 100.0, "price_change_percentage_30d_in_currency": 10.0},
        {"id": "tether", "symbol": "usdt", "name": "Tether", "market_cap": 100.0,
         "price_change_percentage_30d_in_currency": 0.0},
        {"id": "ethereum", "symbol": "eth", "name": "Ethereum", "market_cap": 200.0,
         "price_change_percentage_30d_in_currency": 20.0},
    ]
    for i in range(60):
        rows.append({"id": f"alt{i}", "symbol": f"a{i}", "name": f"Alt {i}", "market_cap": 5.0,
                     "price_change_percentage_30d_in_currency": 30.0 if i < 40 else -10.0})
    rows.append({"id": "wrapped-bitcoin", "symbol": "wbtc", "name": "Wrapped Bitcoin",
                 "market_cap": 0.0, "price_change_percentage_30d_in_currency": 50.0})
    return rows


def _llama_chart(then_total: float, now_total: float) -> list[dict[str, Any]]:
    last = int(NOW) - int(NOW) % DAY
    return [{"date": str(last - d * DAY),
             "totalCirculating": {"peggedUSD": now_total if d == 0 else
                                  then_total if d == 30 else (then_total + now_total) / 2}}
            for d in range(60, -1, -1)]


def _dominance_get(url: str, params: dict[str, Any] | None = None,
                   timeout: float = 40.0) -> Any:
    if url.endswith("/global"):
        return {"data": {"total_market_cap": {"usd": 1200.0},
                         "market_cap_percentage": {"btc": 50.0}}}
    if url.endswith("/coins/markets"):
        assert params is not None and params["price_change_percentage"] == "30d"
        return _market_rows()
    if "market_chart" in url:
        ts = (NOW - 30 * DAY) * 1000
        return {"market_caps": [[ts, 600.0]], "prices": [[ts, 6.0]]}
    if "stablecoincharts" in url:
        return _llama_chart(900.0, 1000.0)
    if url.endswith("/stablecoins"):
        return {"peggedAssets": [{"gecko_id": "tether", "symbol": "USDT", "pegType": "peggedUSD"}]}
    raise AssertionError(url)


def test_dominance_rebuilds_the_start_of_the_period(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "_get", _dominance_get)
    monkeypatch.setattr(cs.time, "time", lambda: NOW)
    out = cs.lines("What happened to bitcoin dominance this month, and what does it say about "
                   "altcoin season?")
    _frame(out)
    text = _text(out)
    # Bitcoin then: 600 / 1.10 = 545.45 (supply 100 then and now). Tether then: 100 * 0.9 = 90
    # (stablecoin history). Ethereum 200 / 1.2 = 166.67; 40 alts 5 / 1.3 = 153.85; 20 alts
    # 5 / 0.9 = 111.11. Sum then 1067.08, sum now 1200: total then 1067.08.
    # Share now 600 / 1200 = 50.0%, then 545.45 / 1067.08 = 51.12%: a fall of 1.12 points.
    assert "50.0% now against about 51.1% 30 days ago, -1.1 percentage points (down)" in text
    # Bitcoin +10.0%; the rest (1200 - 600) / (1067.08 - 545.45) = +15.0%; the whole +12.5%.
    assert "Bitcoin +10.0%, everything else +15.0%, the whole market +12.5%" in text
    assert "the other coins gained more than Bitcoin" in out[0]
    # Alts beating Bitcoin's +10%: Ethereum and 40 alts at +30% are 41 of the first 50 (Tether
    # and the wrapped copy are left out, the 9 weakest alts that fit make up the rest).
    assert "41 of the top 50 coins" in text
    assert "an altcoin season by the usual bar" in text
    assert "differs from the published one by 0.00 points" in text


def test_dominance_says_when_the_window_is_not_the_one_asked(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def get(url: str, params: dict[str, Any] | None = None, timeout: float = 40.0) -> Any:
        if url.endswith("/coins/markets") and params:
            params = {**params, "price_change_percentage": "200d"}
            return [{**r, "price_change_percentage_200d_in_currency":
                     r["price_change_percentage_30d_in_currency"]} for r in _market_rows()]
        return _dominance_get(url, params, timeout)

    monkeypatch.setattr(cs, "_get", get)
    monkeypatch.setattr(cs.time, "time", lambda: NOW)
    monkeypatch.setattr(cs, "_btc_supply_then", lambda days: 100.0)
    monkeypatch.setattr(cs, "_stable_ratio", lambda days: 0.9)
    text = _text(cs.lines("How has bitcoin dominance changed over 90 days?"))
    assert "200 days stands in for the 90 asked" in text


@pytest.mark.parametrize("question", [
    "What is the dominance of USDT among stablecoins?",
    "Who has dominance in the cloud market?",
    "What is the price of bitcoin?",
])
def test_dominance_is_strict(question: str) -> None:
    assert cs.dominance_lines(question) is None


def test_dominance_reports_a_failed_source(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(url: str, params: dict[str, Any] | None = None, timeout: float = 40.0) -> Any:
        raise RpcError(ErrorKind.TRANSPORT, "down")

    monkeypatch.setattr(cs, "_get", down)
    out = cs.lines("What is bitcoin dominance now?")
    _frame(out)
    assert "could not be read just now" in _text(out)


def test_one_429_is_waited_out_then_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def get(url: str, params: dict[str, Any] | None = None, timeout: float = 40.0) -> Any:
        calls.append(url)
        if len(calls) == 1:
            raise RpcError(ErrorKind.RATE_LIMIT, "HTTP 429", http_status=429)
        return {"ok": True}

    monkeypatch.setattr(cs, "_get", get)
    monkeypatch.setattr(cs, "RATE_WAIT", 0.0)
    assert cs._gecko("/global") == {"ok": True}
    assert len(calls) == 2
    assert http.RpcError is RpcError


# -- 2. total stablecoin supply and 3. per-chain change ----------------------------------------


def _peg(now: float, day: float | None, week: float | None, month: float | None,
         ) -> dict[str, Any]:
    block = {"circulating": {"peggedUSD": now}}
    for name, value in (("circulatingPrevDay", day), ("circulatingPrevWeek", week),
                        ("circulatingPrevMonth", month)):
        block[name] = {} if value is None else {"peggedUSD": value}
    return block


def _chain(now: float, month: float | None) -> dict[str, Any]:
    return {"current": {"peggedUSD": now}, "circulatingPrevDay": {"peggedUSD": now},
            "circulatingPrevWeek": {"peggedUSD": now},
            "circulatingPrevMonth": {} if month is None else {"peggedUSD": month}}


def _assets() -> list[dict[str, Any]]:
    return [
        {"id": "1", "name": "Tether", "symbol": "USDT", "pegType": "peggedUSD",
         **_peg(100e9, 99.9e9, 99.5e9, 98e9),
         "chainCirculating": {"Ethereum": _chain(40e9, 41e9), "Tron": _chain(60e9, 57e9)}},
        {"id": "2", "name": "USD Coin", "symbol": "USDC", "pegType": "peggedUSD",
         **_peg(50e9, 50e9, 51e9, 52e9),
         "chainCirculating": {"Ethereum": _chain(30e9, 33.5e9), "Tron": _chain(0.03e9, 0.03e9)}},
        {"id": "3", "name": "USDD", "symbol": "USDD", "pegType": "peggedUSD",
         **_peg(2e9, None, None, None), "chainCirculating": {"Tron": _chain(1.5e9, None)}},
        {"id": "4", "name": "Sky Dollar", "symbol": "USDS", "pegType": "peggedUSD",
         **_peg(7e9, 7e9, 6.5e9, 6e9), "chainCirculating": {"Ethereum": _chain(5e9, 4.2e9)}},
        {"id": "5", "name": "Euro Coin", "symbol": "EURC", "pegType": "peggedEUR",
         **_peg(1e9, 1e9, 1e9, 1e9), "chainCirculating": {"Ethereum": _chain(1e9, 1e9)}},
    ]


def _stable_get(url: str, params: dict[str, Any] | None = None, timeout: float = 40.0) -> Any:
    if "stablecoincharts" in url:
        return _llama_chart(155e9, 157.5e9)
    if url.endswith("/stablecoins"):
        return {"peggedAssets": _assets()}
    raise AssertionError(url)


def test_total_stablecoin_supply_and_its_change(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "_get", _stable_get)
    out = cs.lines("What is total stablecoin supply now and has it grown or shrunk over the "
                   "last 30 days?")
    _frame(out)
    text = _text(out)
    # Daily series: 157.5 now against 155.0 thirty days back: +2.5bn, 2.5 / 155 = +1.6%.
    assert "$157.50bn" in out[0]
    assert "+$2.50bn (+1.6%) over 30 days" in out[0]
    assert "it has grown" in out[0]
    # Like for like over the coins that have a month-ago figure (USDT, USDC, USDS): now 157,
    # then 156: +1.0bn, 1 / 156 = +0.6%. USDD (no month-ago figure) is not in it.
    assert "+$1.00bn (+0.6%)" in text
    # The table adds up to 100 + 50 + 2 + 7 = 159bn; the euro coin is not counted.
    assert "$159.00bn today" in text
    assert "USDT $100.00bn (+2.0%)" in text  # 100 / 98 - 1
    assert "Biggest dollar gain: USDT +$2.00bn" in text
    assert "biggest fall: USDC -$2.00bn" in text  # 50 - 52
    assert "EURC" not in text


def test_total_stablecoin_supply_over_a_longer_window(monkeypatch: pytest.MonkeyPatch) -> None:
    def get(url: str, params: dict[str, Any] | None = None, timeout: float = 40.0) -> Any:
        if "stablecoincharts" in url:
            last = int(NOW) - int(NOW) % DAY
            return [{"date": str(last - d * DAY), "totalCirculating": {
                "peggedUSD": 300e9 - d * 0.1e9}} for d in range(400, -1, -1)]
        return _stable_get(url, params, timeout)

    monkeypatch.setattr(cs, "_get", get)
    out = cs.lines("Has total stablecoin supply grown or shrunk over the last year?")
    _frame(out)
    assert out is not None
    # 300bn now; 365 days back 300 - 36.5 = 263.5bn: +36.5bn, 36.5 / 263.5 = +13.9%.
    assert "+$36.50bn (+13.9%) over 365 days" in out[0]
    # the per-coin table has no 365-day column, so it is not compared
    assert "like for like" not in _text(out)
    assert "Largest coins: USDT $100.00bn, USDC $50.00bn" in _text(out)


@pytest.mark.parametrize("question", [
    "How much USDT versus USDC supply is there?",
    "What is the USDC supply on Base?",
    "What is the yield on stablecoin supply?",
    "Is USDT holding its peg?",
])
def test_total_supply_is_strict(question: str) -> None:
    assert cs.stable_total_lines(question) is None


def test_stablecoin_change_per_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "_get", _stable_get)
    out = cs.lines("Which stablecoin has seen the biggest supply change on Ethereum versus "
                   "Tron this month?")
    _frame(out)
    text = _text(out)
    # Ethereum: USDT 40 - 41 = -1.0bn; USDC 30 - 33.5 = -3.5bn (-10.4%); USDS 5 - 4.2 = +0.8bn.
    # Tron: USDT 60 - 57 = +3.0bn (+5.3%); USDC unchanged; USDD has no month-ago figure and is
    # left out rather than counted as new.
    assert "on Ethereum USDC -$3.50bn (-10.4%)" in out[0]
    assert "on Tron USDT +$3.00bn (+5.3%)" in out[0]
    assert "largest single move of the two is USDC on Ethereum" in out[0]
    assert "Ethereum: all dollar stablecoins $75.00bn, -$3.70bn net" in text
    assert "Tron: all dollar stablecoins $60.03bn, +$3.00bn net" in text
    assert "USDD" not in text
    assert "EURC" not in text


def test_stablecoin_change_names_a_chain_with_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "_get", _stable_get)
    text = _text(cs.lines("Which stablecoin had the biggest supply change on Ethereum or Aptos "
                          "this week?"))
    assert "no dollar stablecoins on Aptos" in text
    assert "7 days" in text


def test_a_window_the_chain_table_lacks_is_said(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "_get", _stable_get)
    text = _text(cs.lines("Which stablecoin had the biggest supply change on Ethereum over "
                          "the last year?"))
    assert "30 days stands in for the 365 days asked" in text


def test_a_chain_question_without_a_chain_is_left_alone() -> None:
    assert cs.stable_chain_lines("Which stablecoin has the biggest supply change this month?") \
        is None


# -- 4. lending comparison and safety ----------------------------------------------------------


def _pool(project: str, chain: str, apy: float, tvl: float, base: float | None = None,
          reward: float | None = None, symbol: str = "USDC") -> dict[str, Any]:
    return {"project": project, "chain": chain, "symbol": symbol, "apy": apy, "tvlUsd": tvl,
            "apyBase": apy if base is None else base, "apyReward": reward}


def _lending_get(url: str, params: dict[str, Any] | None = None, timeout: float = 40.0) -> Any:
    if url.endswith("/pools"):
        return {"data": [
            _pool("aave-v3", "Ethereum", 6.0, 100e6), _pool("aave-v3", "Ethereum", 7.7, 57e6,
                                                             base=6.0, reward=1.7),
            _pool("compound-v3", "Ethereum", 3.5, 40e6), _pool("compound-v3", "Arbitrum", 4.0,
                                                                2e6),
            _pool("sparklend", "Ethereum", 3.4, 4e6), _pool("morpho-blue", "Ethereum", 0.0, 0.2e6),
        ]}
    if url.endswith("/protocols"):
        return [
            {"id": "1599", "slug": "aave-v3", "parentProtocol": "parent#aave", "tvl": 18e9,
             "listedAt": 1648776877, "audits": "2"},
            {"id": "111", "slug": "aave-v2", "parentProtocol": "parent#aave", "tvl": 0.1e9,
             "listedAt": None, "audits": "2"},
            {"id": "3", "slug": "compound-v3", "parentProtocol": "parent#compound-finance",
             "tvl": 1.5e9, "listedAt": 1663169029, "audits": "2"},
            {"id": "114", "slug": "compound-v2", "parentProtocol": "parent#compound-finance",
             "tvl": 0.1e9, "listedAt": None, "audits": "2"},
            {"id": "9", "slug": "sparklend", "parentProtocol": "parent#spark", "tvl": 5e9,
             "listedAt": 1683144119, "audits": "2"},
            {"id": "77", "slug": "unrelated", "parentProtocol": None, "tvl": 1e9,
             "listedAt": 1, "audits": "0"},
        ]
    if url.endswith("/hacks"):
        return [
            {"name": "Compound V2", "date": 1632873600, "amount": 147e6,
             "parentProtocolId": "parent#compound-finance", "defillamaId": "114",
             "returnedFunds": None},
            {"name": "Compounder Finance", "date": 1606867200, "amount": 12e6,
             "defillamaId": None, "returnedFunds": None},
            {"name": "Aave V3", "date": 1773273600, "amount": 862000,
             "parentProtocolId": "parent#aave", "defillamaId": "1599", "returnedFunds": 862000},
            {"name": "Aave", "date": 1724803200, "amount": 56000, "defillamaId": "1",
             "returnedFunds": None},
        ]
    raise AssertionError(url)


def test_aave_against_compound_with_measurable_safety(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "_get", _lending_get)
    out = cs.lines("What is the yield and TVL on Aave USDC versus Compound, and which is safer?")
    _frame(out)
    text = _text(out)
    # The largest USDC pool on Ethereum for each: Aave 6.00% in $100.0m (the rewarded $57m pool
    # is smaller), Compound 3.50% in $40.0m (its $2m Arbitrum pool is another chain).
    assert "Aave v3 pays 6.00% (all from borrowers' interest) in a $100.0m pool" in out[0]
    assert "Compound v3 pays 3.50% (all from borrowers' interest) in a $40.0m pool" in out[0]
    # Family TVL: Aave 18.0 + 0.1 = $18.10bn over 2 versions; Compound 1.5 + 0.1 = $1.60bn.
    assert "$18.10bn across all 2 versions" in text
    assert "$1.60bn across all 2 versions" in text
    # Aave: $862k returned and $56k not = $918k in all; "Compounder Finance" is not Compound.
    assert "2 hacks recorded, $918k in all ($862k returned)" in text
    assert "1 hack recorded, $147.0m in all ($0 returned)" in text
    assert "Compounder" not in text
    # Aave wins value locked, hack losses and time listed; audits tie at 2: 3 of 4.
    assert "Aave v3 looks safer" in text
    assert "(3 of 4 tests)" in text
    assert "does not rate smart-contract, oracle or governance risk" in text


def test_lending_without_a_safety_question_gives_no_verdict(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "_get", _lending_get)
    text = _text(cs.lines("Compare Aave and Compound USDC yields on Ethereum"))
    assert "looks safer" not in text


def test_lending_names_a_missing_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "_get", _lending_get)
    text = _text(cs.lines("Aave versus Morpho USDC yield, which is safer?"))
    # Morpho's only pool is $0.2m, under the $1m floor.
    assert "Morpho Blue: DeFiLlama lists no USDC pool over $1m on Ethereum" in text


def test_lending_on_a_named_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "_get", _lending_get)
    text = _text(cs.lines("Aave vs Compound USDC rates on Arbitrum"))
    assert "Compound v3 pays 4.00%" in text
    assert "Aave v3: DeFiLlama lists no USDC pool over $1m on Arbitrum" in text


@pytest.mark.parametrize("question", [
    "What is Aave's TVL?", "Is Compound safe?", "Compare Aave to a savings account",
    "Which is better, Uniswap or Curve?",
])
def test_lending_needs_two_lenders(question: str) -> None:
    assert cs.lending_lines(question) is None


def test_the_safety_tally_can_tie() -> None:
    cards = [{"name": "A", "family_tvl": 1.0, "lost": 0.0, "audits": 2, "listed": 5},
             {"name": "B", "family_tvl": 2.0, "lost": 0.0, "audits": 2, "listed": 9}]
    # A is older, B is bigger; hack losses and audits tie: 1 test each.
    assert "no clear difference" in cs._safer_line(cards)


# -- 5. spot against perpetual liquidity on Bitget ---------------------------------------------


class Bitget:
    """A fake Bitget: tickers, funding and merged books that only reach 1% at a coarser step."""

    def __init__(self, funding: str = "0.0002", fine_complete: bool = False) -> None:
        self.funding = funding
        self.fine_complete = fine_complete
        self.precisions: list[str] = []

    def __call__(self, path: str, params: dict[str, str]) -> Any:
        if path == "/api/v2/spot/market/tickers":
            return [{"bidPr": "100", "askPr": "100.2", "usdtVolume": "1000000"}]
        if path == "/api/v2/mix/market/ticker":
            return [{"bidPr": "100.1", "askPr": "100.2", "usdtVolume": "20000000",
                     "holdingAmount": "500", "markPrice": "100.15", "fundingRate": self.funding}]
        if path == "/api/v2/mix/market/current-fund-rate":
            return [{"fundingRate": self.funding, "fundingRateInterval": "8"}]
        if path.endswith("/merge-depth"):
            self.precisions.append(params["precision"])
            if params["precision"] == "scale0" and not self.fine_complete:
                # the finest step stops at 99.9 / 100.4: short of 1% on both sides
                return {"bids": [["100.0", "5"], ["99.9", "5"]],
                        "asks": [["100.2", "4"], ["100.4", "4"]]}
            if path.startswith("/api/v2/spot"):
                return {"bids": [["100.0", "5"], ["99.5", "5"], ["99.0", "5"]],
                        "asks": [["100.2", "4"], ["100.8", "4"], ["101.2", "4"]]}
            return {"bids": [["100.1", "10"], ["99.6", "10"], ["99.0", "10"]],
                    "asks": [["100.2", "10"], ["101.0", "10"], ["101.3", "10"]]}
        raise AssertionError(path)


def test_eth_perpetual_against_spot_and_the_funding_sign(
        monkeypatch: pytest.MonkeyPatch) -> None:
    fake = Bitget()
    monkeypatch.setattr(cs, "_bitget", fake)
    out = cs.lines("How much liquidity is there in ETH perpetuals versus spot on Bitget, and is "
                   "the funding paying longs or shorts?")
    _frame(out)
    text = _text(out)
    # 24-hour volume $20.0m against $1.0m: 20.0 times.
    assert "ETH perpetuals traded $20.0m in 24 hours against $1.0m on spot (20.0 times)" in text
    # Spot mid 100.1, band 99.099 to 101.101: bids 100.0 * 5 + 99.5 * 5 = 997.5 (99.0 is outside),
    # asks 100.2 * 4 + 100.8 * 4 = 804. Perpetual mid 100.15, band 99.1485 to 101.1515: bids
    # 100.1 * 10 + 99.6 * 10 = 1,997, asks 100.2 * 10 + 101.0 * 10 = 2,012.
    assert "Spot book within 1% of the mid: bids $998, asks $804." in text
    assert "Perpetual book within 1% of the mid: bids $2k, asks $2k." in text
    # the finest step did not reach 1%, so the next one was read (spot and perpetual each)
    assert fake.precisions == ["scale0", "scale1", "scale0", "scale1"]
    # +0.0002 every 8 hours: 0.0002 * 3 * 365 = 21.9% a year; positive means longs pay shorts.
    assert "Funding on the ETH perpetual is +0.0200% every 8 hours (+21.9% a year" in text
    assert "longs are paying shorts" in text
    assert "Open interest in the ETH perpetual: $50k." in text  # 500 * 100.15 = 50,075


def test_negative_and_zero_funding_are_said_in_words(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "_bitget", Bitget(funding="-0.0001"))
    assert "shorts are paying longs" in _text(cs.lines(
        "Is the BTC funding rate paying longs or shorts on the perpetual?"))
    monkeypatch.setattr(cs, "_bitget", Bitget(funding="0"))
    assert "funding is zero" in _text(cs.lines(
        "Is the BTC funding rate paying longs or shorts on the perpetual?"))


def test_funding_question_leads_with_funding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "_bitget", Bitget())
    out = cs.lines("Who is paying funding on SOL perps right now, longs or shorts?")
    _frame(out)
    assert out is not None and out[0].startswith("Bottom line: Funding on the SOL perpetual")


def test_a_lower_bound_is_marked(monkeypatch: pytest.MonkeyPatch) -> None:
    class Short(Bitget):
        def __call__(self, path: str, params: dict[str, str]) -> Any:
            if path.endswith("/merge-depth"):
                return {"bids": [["100.0", "5"], ["99.9", "5"]],
                        "asks": [["100.2", "4"], ["100.4", "4"]]}
            return super().__call__(path, params)

    monkeypatch.setattr(cs, "_bitget", Short())
    text = _text(cs.lines("ETH perpetual versus spot liquidity on Bitget"))
    assert "(lower bound)" in text
    assert "at least" in text


def test_coin_comes_from_the_earlier_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "_bitget", Bitget())
    out = cs.lines("And how does perpetual liquidity compare with spot on Bitget?",
                   prior=["I hold some SOL", "thanks"])
    assert out is not None and "SOL perpetuals traded" in out[0]
    assert cs.lines("And how does perpetual liquidity compare with spot on Bitget?") is None


def test_a_coin_with_no_spot_market_cannot_be_compared(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoSpot(Bitget):
        def __call__(self, path: str, params: dict[str, str]) -> Any:
            if path == "/api/v2/spot/market/tickers":
                return []
            return super().__call__(path, params)

    monkeypatch.setattr(cs, "_bitget", NoSpot())
    text = _text(cs.lines("ETH perpetuals versus spot volume on Bitget"))
    assert "so the two cannot be compared" in text


def test_a_failed_bitget_read_is_one_honest_line(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(path: str, params: dict[str, str]) -> Any:
        raise RpcError(ErrorKind.TRANSPORT, "down")

    monkeypatch.setattr(cs, "_bitget", down)
    out = cs.lines("ETH perpetuals versus spot volume on Bitget")
    assert out is not None and "could not be read just now" in out[0]


@pytest.mark.parametrize("question", [
    "Is the funding rate positive?", "What is ETH trading at on spot?",
    "How deep is the BTC order book?", "What is the funding rate on my savings account?",
])
def test_perp_spot_is_strict(question: str) -> None:
    assert cs.perp_spot_lines(question) is None


# -- 6. crypto event calendar ------------------------------------------------------------------


def _option_rows(expiry: Any) -> list[dict[str, Any]]:
    return [{"expiry": expiry, "strike": 100.0, "call": True, "oi": 10.0},
            {"expiry": expiry, "strike": 100.0, "call": False, "oi": 20.0},
            {"expiry": expiry, "strike": 110.0, "call": True, "oi": 5.0},
            {"expiry": expiry, "strike": 90.0, "call": False, "oi": 5.0}]


def test_max_pain_is_the_strike_that_costs_holders_least() -> None:
    # Settling at 90 pays the strike-100 puts 20 * 10 = 200; at 100 nothing; at 110 the
    # strike-100 calls 10 * 10 = 100. The minimum is 100.
    assert cs.max_pain(_option_rows(datetime(2026, 10, 9, tzinfo=UTC).date())) == 100.0


def test_expiries_count_notional_and_skip_the_settled_and_the_far() -> None:
    now = datetime(2026, 10, 6, 9, 0, tzinfo=UTC)  # after 08:00: today's expiry has settled
    rows = (_option_rows(now.date()) + _option_rows((now + timedelta(days=3)).date())
            + _option_rows((now + timedelta(days=40)).date()))
    out = cs.expiries(rows, 100.0, now, 14)
    assert len(out) == 1
    expiry, notional, put_call, pain = out[0]
    assert expiry == (now + timedelta(days=3)).date()
    # open interest 10 + 20 + 5 + 5 = 40 contracts at an index of 100: $4,000; puts 25 / calls 15
    assert notional == 4000.0
    assert put_call == pytest.approx(25 / 15)
    assert pain == 100.0


def _series(now: float, jumps: dict[int, float], start: float = 1000.0) -> list[dict[str, Any]]:
    value, points = start, []
    for k in range(-2, 14):
        value += jumps.get(k, 0.0)
        points.append({"timestamp": int(now) + k * DAY, "unlocked": value})
    return [{"label": "Investors", "data": points}]


def test_upcoming_unlocks_rank_by_dollars_and_read_cliffs_and_streams(
        monkeypatch: pytest.MonkeyPatch) -> None:
    bodies: dict[str, dict[str, Any] | None] = {
        # a cliff of 500 tokens on day 5, and 100 in the same window before it starts
        "cliffy": {"documentedData": {"data": _series(NOW, {5: 500.0})}},
        # 10 tokens a day on each of days 1 to 13
        "streamy": {"documentedData": {"data": _series(NOW, {k: 10.0 for k in range(1, 14)})}},
        "none": None,
        "tether": {"documentedData": {"data": _series(NOW, {3: 9e9})}},
    }
    monkeypatch.setattr(cs, "_emissions", lambda slug: bodies.get(slug))
    markets = [
        {"id": "cliffy", "symbol": "clf", "current_price": 2.0, "market_cap": 100_000.0},
        {"id": "streamy", "symbol": "str", "current_price": 1.0, "market_cap": 10_000.0},
        {"id": "none", "symbol": "non", "current_price": 1.0, "market_cap": 1.0},
        {"id": "tether", "symbol": "usdt", "current_price": 1.0, "market_cap": 1e11},
        {"id": "bitcoin", "symbol": "btc", "current_price": 1.0, "market_cap": 1e12},
    ]
    found, scheduled = cs.upcoming_unlocks(markets, {"tether"}, 14, NOW)
    # stablecoins and mining coins are not unlocks; "none" has no schedule
    assert scheduled == 2
    assert [u["symbol"] for u in found] == ["CLF", "STR"]
    clf, stream = found
    assert clf["tokens"] == 500.0
    assert clf["usd"] == 1000.0  # 500 tokens at $2
    assert clf["share"] == pytest.approx(0.01)  # 1,000 / 100,000
    assert clf["peak_share"] == 1.0  # all of it on one day: a cliff
    assert stream["tokens"] == 130.0  # 13 days at 10
    assert stream["peak_share"] == pytest.approx(10 / 130)  # a stream, not a cliff


def _event_get(url: str, params: dict[str, Any] | None = None, timeout: float = 40.0) -> Any:
    if url.endswith("/stablecoins"):
        return {"peggedAssets": [{"gecko_id": "tether"}]}
    if url.endswith("/coins/markets"):
        return [{"id": "cliffy", "symbol": "clf", "current_price": 2.0, "market_cap": 100_000.0}]
    raise AssertionError(url)


def test_event_calendar_answers_with_what_is_not_covered(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now(UTC)
    soon = (now + timedelta(days=3)).date()
    cpi = datetime(2026, 10, 14, 12, 30, tzinfo=UTC)
    monkeypatch.setattr(cs, "_get", _event_get)
    monkeypatch.setattr(cs, "_deribit", lambda cur: (_option_rows(soon), 100.0))
    monkeypatch.setattr(cs, "_macro", lambda days: [(cpi, "US CPI release")])
    monkeypatch.setattr(cs, "_emissions", lambda slug: {"documentedData": {"data": _series(
        time.time(), {5: 500.0})}})
    out = cs.lines("Which major crypto event risks are scheduled in the next two weeks that "
                   "could move BTC?")
    _frame(out)
    text = _text(out)
    assert ("in the next 14 days $4k of BTC options expire on Deribit across 1 date, the "
            "largest") in text
    assert "US CPI release Wed 14 Oct 12:30 UTC" in text
    assert "put/call 1.67, max pain 100" in text
    assert "CLF $1k (1.0% of its market cap)" in text
    assert "Not covered: ETF decisions, protocol upgrades" in text
    assert "not an all-clear" in text


def test_unlock_question_leads_with_unlocks_and_names_the_heaviest(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "_get", _event_get)
    monkeypatch.setattr(cs, "_deribit", lambda cur: (_option_rows(
        (datetime.now(UTC) + timedelta(days=3)).date()), 100.0))
    monkeypatch.setattr(cs, "_macro", lambda days: [])
    # 500 tokens at $2 is only $1,000: under the $10m floor, so no release stands out
    monkeypatch.setattr(cs, "_emissions", lambda slug: {"documentedData": {"data": _series(
        time.time(), {5: 500.0})}})
    out = cs.lines("What are the next big token unlocks in the coming 30 days and which "
                   "would worry you most?")
    _frame(out)
    assert out is not None
    assert out[0].startswith("Bottom line: in the next 30 days the largest scheduled token "
                             "releases are CLF")
    assert "None of the scheduled releases is over $10m" in out[0]
    assert "no US CPI or FOMC date falls in the window" in out[0]


def test_event_calendar_survives_every_source_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(*args: Any, **kwargs: Any) -> Any:
        raise RpcError(ErrorKind.TRANSPORT, "down")

    monkeypatch.setattr(cs, "_get", down)
    monkeypatch.setattr(cs, "_deribit", down)
    monkeypatch.setattr(cs, "_macro", down)
    out = cs.lines("Which crypto events are scheduled in the next week?")
    _frame(out)
    text = _text(out)
    assert "Deribit's expiries could not be read just now" in text
    assert "token-release schedules could not be read just now" in text
    assert "Not covered" in text


@pytest.mark.parametrize("question", [
    "When is the next CPI release?", "Which events are scheduled for the Olympics?",
    "What is the next unlock of ARB?", "Is bitcoin a good buy?",
    "How many events did bitcoin have last year?",
])
def test_events_are_strict(question: str) -> None:
    assert cs.event_lines(question) is None


@pytest.mark.parametrize(("question", "days"), [
    ("crypto events in the next two weeks", 14), ("crypto events next month", 30),
    ("crypto events this week", 7), ("crypto events coming up", 14),
    ("crypto events in the next 400 days", 90),
])
def test_event_horizon(question: str, days: int) -> None:
    assert cs.event_horizon(question) == days


# -- the entry point ---------------------------------------------------------------------------


@pytest.mark.parametrize("question", [
    "What is the weather in Paris?", "Rank Exxon, Shell and BP by dividend yield.",
    "What is the circulating versus max supply of SUI and ARB?",
    "Which tokens in the top 20 have the highest share of supply still locked?",
    "How is the S&P 500 doing this month?", "",
])
def test_unrelated_questions_are_left_to_other_readers(question: str) -> None:
    assert cs.lines(question) is None


def test_total_supply_also_answers_how_big_and_is_it_growing(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "_get", _stable_get)
    out = cs.lines("How big is the stablecoin market and is it growing?")
    _frame(out)
    assert out is not None and "it has grown" in out[0]
