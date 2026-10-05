"""Token supply and unlock answers (lui/research/token_supply.py): every figure recomputed by hand
on made-up sources, the questions each reader branch answers, and the failures that must read as an
honest one-liner."""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from argus.lui.research import token_supply as ts
from argus.truth.failures import ErrorKind, RpcError

NOW = 1_800_000_000.0
DAY = 86400.0


def _row(cid: str, sym: str, circ: float, total: float | None, cap: float | None, mcap: float,
         fdv: float | None = None, volume: float = 1e6, name: str | None = None) -> dict[str, Any]:
    return {"id": cid, "symbol": sym, "name": name or cid.title(), "circulating_supply": circ,
            "total_supply": total, "max_supply": cap, "market_cap": mcap,
            "fully_diluted_valuation": fdv, "total_volume": volume}


def _chart(start_supply: float, end_supply: float, days: int = 365) -> dict[str, Any]:
    """Daily price 2.0 throughout, so market cap / price is the supply, linear in between."""
    caps, prices = [], []
    for i in range(days + 1):
        t = (NOW - (days - i) * DAY) * 1000
        supply = start_supply + (end_supply - start_supply) * i / days
        caps.append([t, supply * 2.0])
        prices.append([t, 2.0])
    return {"market_caps": caps, "prices": prices, "total_volumes": []}


class Net:
    """Routes ``_get`` by URL to the made-up source answers, and records what was read."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.rows: dict[str, dict[str, Any]] = {}
        self.top: list[dict[str, Any]] = []
        self.stable: list[dict[str, Any]] | None = []
        self.charts: dict[str, dict[str, Any]] = {}
        self.index: list[dict[str, Any]] | None = None
        self.down: set[str] = set()

    def __call__(self, url: str, params: dict[str, Any] | None = None, data: bytes | None = None,
                 timeout: float = 25.0) -> Any:
        self.calls.append(url)
        params = params or {}
        for marker in self.down:
            if marker in url or (marker == "rpc" and url == ts.SOLANA_RPC):
                raise RpcError(ErrorKind.TRANSPORT, "down")
        if url.endswith("/coins/markets"):
            if params.get("category") == "stablecoins":
                if self.stable is None:
                    raise RpcError(ErrorKind.TRANSPORT, "down")
                return self.stable
            if "ids" in params:
                return [self.rows[i] for i in str(params["ids"]).split(",") if i in self.rows]
            return self.top
        if "/market_chart" in url:
            return self.charts[url.split("/coins/")[1].split("/")[0]]
        if url == ts.LLAMA_INDEX:
            if self.index is None:
                raise RpcError(ErrorKind.TRANSPORT, "down")
            return {"data": self.index}
        if url.endswith("/supply-over-time"):
            return {"since_merge": [
                {"timestamp": _iso(NOW - 365 * DAY), "supply": 100_000_000.0},
                {"timestamp": _iso(NOW - 90 * DAY), "supply": 100_500_000.0},
                {"timestamp": _iso(NOW - 30 * DAY), "supply": 100_800_000.0},
                {"timestamp": _iso(NOW), "supply": 101_000_000.0}]}
        if url.endswith("/issuance-estimate"):
            return {"issuance_per_slot_gwei": 400_000_000.0}
        if url == ts.SOLANA_RPC:
            method = json.loads(data or b"{}")["method"]
            if method == "getInflationRate":
                return {"result": {"total": 0.04, "epoch": 1000, "validator": 0.04}}
            return {"result": {"initial": 0.08, "taper": 0.15, "terminal": 0.015}}
        raise AssertionError(f"unexpected read {url}")


def _iso(t: float) -> str:
    from datetime import UTC, datetime
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture
def net(monkeypatch: pytest.MonkeyPatch) -> Net:
    n = Net()
    monkeypatch.setattr(ts, "_get", n)
    monkeypatch.setattr(ts, "_index_cache", {})
    monkeypatch.setattr(time, "sleep", lambda s: None)
    return n


# ------------------------------------------------------------------- which question is this


@pytest.mark.parametrize("text, kind", [
    ("What is the circulating versus max supply of SUI and ARB, and how inflationary are they?",
     "supply"),
    ("Which tokens in the top 20 have the highest share of supply still locked or vesting?",
     "locked"),
    ("What are the next big token unlocks in the coming 30 days and which would worry you most?",
     "unlocks"),
    ("What is Solana's inflation rate compared with Ethereum's?", "supply"),
    ("What is the fully diluted valuation of Aptos compared with its market cap?", "supply"),
    ("How much of ARB supply is still locked in the next 7 days of unlocks?", "unlocks"),
])
def test_asks_picks_the_right_branch(text: str, kind: str) -> None:
    assert ts.asks(text) == kind


@pytest.mark.parametrize("text", [
    "Is bitcoin an inflation hedge?",
    "How will US inflation and CPI affect Solana?",
    "What is the price of SUI?",
    "How do I unlock my Bitget Earn product?",
    "What is the money supply doing to ETH?",
    "Show me ETH supply on exchanges",
    "Is the supply chain a risk for chip stocks?",
    "Explain how staking works",
    "What is the circulating supply of cheese?",
])
def test_other_questions_are_left_to_other_readers(text: str) -> None:
    assert ts.lines(text) is None


def test_tickers_are_not_taken_from_ordinary_words() -> None:
    assert ts._named("what is the total supply of near the end of the year") == []
    assert ts._named("total supply of NEAR and Optimism and polygon") == ["NEAR", "OP", "POL"]
    assert ts._named("how inflationary is ether's issuance, and sui") == ["ETH", "SUI"]


# ------------------------------------------------------------------- the arithmetic


def test_locked_shares_by_hand() -> None:
    row = _row("x", "x", 4.0e9, 10e9, 12e9, 1.0)
    vs_total, vs_max = ts.locked_shares(row)
    assert vs_total == pytest.approx(0.6)
    assert vs_max == pytest.approx(1 - 4 / 12)
    assert ts.locked_shares(_row("y", "y", 5.0, None, None, 1.0)) == (None, None)
    # a circulating figure above total is floored, not reported as negative locked supply
    assert ts.locked_shares(_row("z", "z", 11.0, 10.0, None, 1.0))[0] == 0.0


def test_growth_is_between_real_points_and_annualised() -> None:
    series = [(NOW - (365 - i) * DAY, 100.0 * (1.0 + i / 365)) for i in range(366)]
    total, annual, span = ts.growth(series, 365)  # type: ignore[misc]
    assert total == pytest.approx(1.0) and span == pytest.approx(365.0)
    assert annual == pytest.approx(1.0)
    t90, a90, s90 = ts.growth(series, 90)  # type: ignore[misc]
    start = series[-91][1]
    assert t90 == pytest.approx(series[-1][1] / start - 1)
    assert a90 == pytest.approx((series[-1][1] / start) ** (365 / s90) - 1)


def test_growth_is_none_when_the_series_is_too_short() -> None:
    short = [(NOW - (40 - i) * DAY, 100.0 + i) for i in range(41)]
    assert ts.growth(short, 365) is None and ts.growth(short, 90) is None
    assert ts.growth(short[:1], 30) is None


def test_circulating_series_is_market_cap_over_price(net: Net) -> None:
    net.charts["sui"] = _chart(100.0, 150.0, days=10)
    out = ts.circulating_series("sui")
    assert out[0][1] == pytest.approx(100.0) and out[-1][1] == pytest.approx(150.0)
    assert out[-1][0] == pytest.approx(NOW)


# ------------------------------------------------------------------- named coins


def test_sui_and_arb_answer(net: Net) -> None:
    net.rows = {"sui": _row("sui", "sui", 4.0e9, 10e9, 10e9, 5e9, 12e9),
                "arbitrum": _row("arbitrum", "arb", 6.0e9, 10e9, 10e9, 1.5e9, 2.5e9)}
    net.charts = {"sui": _chart(3.5e9, 4.0e9), "arbitrum": _chart(4.8e9, 6.0e9)}
    out = ts.lines("What is the circulating versus max supply of SUI and ARB, and how "
                   "inflationary are they?")
    assert out is not None
    assert out[0].startswith("Bottom line: SUI circulating 4.00B of a 10.00B maximum (40.0%)")
    assert "ARB circulating 6.00B of a 10.00B maximum (60.0%)" in out[0]
    assert "SUI's circulating supply grew 14.3% in 365 days" in out[0]   # 4.0/3.5 - 1
    assert "ARB's circulating supply grew 25.0% in 365 days" in out[0]   # 6.0/4.8 - 1
    joined = "\n".join(out)
    assert "60.0% of total supply is not yet circulating, 60.0% of the maximum" in joined
    assert "fully diluted $12.00B (2.40x the market cap)" in joined
    assert "not a vesting schedule" in joined
    assert out[-1].startswith("Data: CoinGecko") and out[-1].endswith("Not advice.")


def test_supply_only_question_does_not_read_a_history(net: Net) -> None:
    net.rows = {"aptos": _row("aptos", "apt", 8.0e8, 1.1e9, None, 4e9, 5e9)}
    out = ts.lines("What is the fully diluted valuation of Aptos compared with its market cap?")
    assert out is not None and "grew" not in out[0]
    assert not any("market_chart" in c for c in net.calls)
    assert "no maximum set" in out[0] and "maximum none set" in "\n".join(out)
    assert "fully diluted $5.00B (1.25x the market cap)" in "\n".join(out)


def test_coin_without_a_maximum_or_fdv_says_so(net: Net) -> None:
    net.rows = {"ethereum": _row("ethereum", "eth", 1.2e8, 1.2e8, None, 3e11, None)}
    out = ts.lines("What is the circulating and total supply of ETH?")
    assert out is not None
    assert "no fully diluted value" in "\n".join(out)


def test_ethereum_and_solana_use_their_own_chains(net: Net) -> None:
    out = ts.lines("What is Solana's inflation rate compared with Ethereum's?")
    assert out is not None
    # 400,000,000 gwei a slot = 0.4 ETH x 7,200 x 365 = 1,051,200 ETH of 101,000,000 = 1.0408%
    assert "Ethereum's 1.04% gross issuance" in out[0]
    assert "Solana's inflation rate is 4.00% a year (epoch 1000)" in out[0]
    assert "3.8 times" in out[0]                      # 0.04 / 0.010408
    # net 30 days: 101.0/100.8 over 30 days, annualised
    net30 = (101.0 / 100.8) ** (365 / 30) - 1
    assert f"{net30 * 100:.2f}% annualised over the last 30 days" in out[0]
    joined = "\n".join(out)
    assert "8.0% initial, tapering 15% a year toward 1.5%" in joined
    assert "1,051,200 ETH a year" in joined
    assert not any("coins/markets" in c for c in net.calls)   # nothing from CoinGecko needed
    assert out[-1] == "Data: ultrasound.money, Solana public RPC. Not advice."


def test_a_native_source_failure_is_one_honest_line(net: Net) -> None:
    net.down = {"ultrasound", "rpc"}
    out = ts.lines("What is Solana's inflation rate compared with Ethereum's?")
    assert out is not None
    assert out[0].startswith("Bottom line: the supply figures could not be read just now")
    assert out[-1].endswith("Not advice.")


def test_markets_failure_keeps_the_native_answer(net: Net) -> None:
    net.down = {"coins/markets"}
    out = ts.lines("How inflationary is Solana, and what is its circulating supply?")
    assert out is not None
    assert "Solana's protocol inflation is 4.00% a year" in out[0]
    assert any("did not return it just now" in x for x in out)


# ------------------------------------------------------------------- top 20 locked


def _top_table() -> list[dict[str, Any]]:
    return [
        _row("bitcoin", "btc", 19.9e6, 19.9e6, 21e6, 2e12),
        {**_row("tether", "usdt", 1.8e11, 1.9e11, None, 1.8e11), "current_price": 1.0004},
        _row("hyperliquid", "hype", 2.2e8, 9.5e8, 1e9, 2e10),
        _row("wrapped-bitcoin", "wbtc", 1e5, 1e5, None, 1e10, name="Wrapped Bitcoin"),
        _row("dogecoin", "doge", 1.5e11, 1.7e11, None, 3e10),
        _row("uniswap", "uni", 6.0e8, 8.8e8, 1e9, 6e9),
        _row("figure-heloc", "figr_heloc", 2.3e10, 2.3e10, None, 2.3e10),
        _row("unknown", "nodata", 0, None, None, 1e9),
    ]


def test_top_n_locked_ranking(net: Net) -> None:
    net.top = _top_table()
    net.stable = [{"id": "tether"}]
    ranked, skipped = ts.locked_ranking(20)
    assert [r["symbol"] for r in ranked] == ["hype", "uni", "doge", "btc"]
    # against the larger of total and maximum supply
    assert ranked[0]["locked"] == pytest.approx(1 - 2.2e8 / 1e9)
    assert ranked[1]["locked"] == pytest.approx(1 - 6.0e8 / 1e9)
    assert ranked[3]["locked"] == pytest.approx(1 - 19.9e6 / 21e6)
    assert ranked[2]["locked_base"] == "total"
    assert ranked[2]["locked"] == pytest.approx(1 - 1.5e11 / 1.7e11)
    assert set(skipped) == {"USDT", "WBTC", "FIGR_HELOC", "NODATA"}


def test_stablecoin_list_down_falls_back_to_the_price_rule_and_says_so(net: Net) -> None:
    net.top = _top_table()
    net.stable = None
    out = ts.lines("Which of the top 20 tokens have the most supply still locked or vesting?")
    assert out is not None
    joined = chr(10).join(out)
    assert "stablecoin list could not be read just now" in joined
    assert "USDT" in joined and "(price rule)" not in joined
    assert "DOGE" in joined


def test_top_n_locked_answer(net: Net) -> None:
    net.top = _top_table()
    net.stable = [{"id": "tether"}]
    out = ts.lines("Which tokens in the top 20 have the highest share of supply still locked "
                   "or vesting?")
    assert out is not None
    assert out[0].startswith("Bottom line: of the top 4 coins by market cap (stablecoins "
                             "excluded), "
                             "HYPE has the highest share of supply not yet circulating at 78.0%, "
                             "then UNI at 40.0%")
    assert out[1].startswith("1. HYPE 78.0% not yet circulating (220.00M of 1.00B maximum")
    joined = "\n".join(out)
    assert "Left out of the ranking" in joined and "USDT" in joined
    assert "It is not a vesting schedule" in joined
    assert out[-1].startswith("Data: CoinGecko coins/markets") and out[-1].endswith("Not advice.")
    assert ts._topn_n("top ten tokens still locked") == 10
    assert ts._topn_n("top 3 locked") == 5 and ts._topn_n("top 500 locked") == 50


def test_top_n_failure_is_honest(net: Net) -> None:
    net.down = {"coins/markets"}
    out = ts.lines("Which of the top 20 tokens have the most supply still locked?")
    assert out is not None and "could not be read just now" in out[0]


# ------------------------------------------------------------------- unlock calendar


def _proto(name: str, gid: str | None, price: float, circ: float, events: list[dict[str, Any]],
           per_day: float = 0.0, priced: bool = True) -> dict[str, Any]:
    return {"name": name, "gecko_id": gid, "token": f"coingecko:{gid}", "circSupply": circ,
            "tokenPrice": [{"price": price, "timestamp": NOW - 600}] if priced else [],
            "unlockEvents": events, "unlocksPerDay": per_day}


def _cliff(at: float, *amounts: float) -> dict[str, Any]:
    return {"timestamp": at, "cliffAllocations": [
        {"recipient": r, "category": "insiders", "unlockType": "cliff", "amount": a}
        for r, a in zip(["Team", "Investors", "Foundation"], amounts, strict=False)],
        "linearAllocations": [], "summary": {}}


def _index() -> list[dict[str, Any]]:
    return [
        # 100M of 1B circulating at $2: 10% of circulating, $200M; 10M also burns (ignored)
        _proto("Alpha", "alpha", 2.0, 1e9, [
            _cliff(NOW - 5 * DAY, 500e6),                     # already past
            _cliff(NOW + 10 * DAY, 60e6, 40e6),
            _cliff(NOW + 12 * DAY, -10e6),                    # a burn
            _cliff(NOW + 40 * DAY, 900e6)]),                  # outside the window
        # a continuous flow only: 1M a day for 30 days of 2B circulating at $1
        _proto("Beta", "beta", 1.0, 2e9, [], per_day=1e6),
        # large dollar value, small share
        _proto("Gamma", "gamma", 50.0, 2e8, [_cliff(NOW + 3 * DAY, 4e6)]),
        # micro-cap, 50% unlocking: left out of the ranking
        _proto("Dust", "dust", 0.01, 1e7, [_cliff(NOW + 2 * DAY, 5e6)]),
        _proto("NoPrice", "noprice", 1.0, 1e9, [_cliff(NOW + 2 * DAY, 5e8)], priced=False),
        _proto("Quiet", "quiet", 1.0, 1e9, [_cliff(NOW + 90 * DAY, 1e8)]),
    ]


def test_unlock_calendar_by_hand() -> None:
    cal = {c["name"]: c for c in ts.unlock_calendar(_index(), NOW, 30)}
    assert set(cal) == {"Alpha", "Beta", "Gamma", "Dust"}
    alpha = cal["Alpha"]
    assert alpha["cliff"] == 100e6 and alpha["linear"] == 0.0
    assert alpha["value"] == pytest.approx(200e6) and alpha["share"] == pytest.approx(0.10)
    ts_big, amount, who = alpha["biggest"]
    assert ts_big == NOW + 10 * DAY and amount == 100e6 and who == ["Team", "Investors"]
    beta = cal["Beta"]
    assert beta["linear"] == 30e6 and beta["cliff"] == 0.0
    assert beta["share"] == pytest.approx(30e6 / 2e9) and beta["value"] == pytest.approx(30e6)
    assert cal["Gamma"]["value"] == pytest.approx(200e6)
    assert ts.unlock_calendar(_index(), NOW, 100)[0]["name"]  # a wider window sees more
    assert {c["name"] for c in ts.unlock_calendar(_index(), NOW, 100)} >= {"Quiet", "Alpha"}


def test_unlock_answer(net: Net) -> None:
    net.index = _index()
    net.rows = {"alpha": _row("alpha", "a", 1e9, 2e9, None, 2e9, volume=500e6),
                "gamma": _row("gamma", "g", 2e8, 3e8, None, 1e10, volume=1e9)}
    out = ts.lines("What are the next big token unlocks in the coming 30 days and which would "
                   "worry you most?", now=NOW)
    assert out is not None
    assert out[0].startswith("Bottom line: the largest unlock in the next 30 days relative to what "
                             "circulates is Alpha: 100.00M tokens ($200.00M), 10.0% of its "
                             "circulating supply and 40% of its 24-hour trading volume")
    assert "the largest in dollars is " in out[0]
    assert out[2].startswith("1. Alpha: 100.00M in dated cliffs, $200.00M at $2.00 a token, 10.0% "
                             "of circulating supply, 40% of its 24-hour volume; largest cliff "
                             "100.00M on ")
    assert "to Team, Investors" in out[2]
    joined = "\n".join(out)
    assert "Dust" not in joined and "1 smaller tokens" in joined
    assert "not opinion" in joined and "not modelled" in joined
    assert out[-1].startswith("Data: DeFiLlama emissions index") and out[-1].endswith("Not advice.")


def test_unlock_window_follows_the_question(net: Net) -> None:
    net.index = _index()
    out = ts.lines("Which token unlocks are coming in the next 100 days?", now=NOW)
    assert out is not None and "in the next 100 days" in out[0]
    assert ts._unlock_days("next week's unlocks") == 7
    assert ts._unlock_days("any big unlocks coming") == 30


def test_unlock_for_a_named_token_only(net: Net) -> None:
    net.index = _index()
    net.rows = {}
    out = ts.lines("When is the next ARB unlock coming up in the next 30 days?", now=NOW)
    assert out is not None
    assert "no unlock for the named token" in out[0]


def test_unlock_source_down_gives_the_locked_ranking_and_names_the_need(net: Net) -> None:
    net.index = None
    net.top = _top_table()
    net.stable = [{"id": "tether"}]
    out = ts.lines("What are the next big token unlocks in the coming 30 days?", now=NOW)
    assert out is not None
    assert out[0].startswith("Bottom line: the token-unlock calendar could not be read just now")
    assert "Tokenomist" in out[0] and "none is estimated" in out[0]
    assert out[1].startswith("Locked-share ranking instead: of the top 4 coins")
    assert out[-1].endswith("Not advice.")


def test_unlock_volume_missing_is_said(net: Net) -> None:
    net.index = _index()
    net.down = {"coins/markets"}
    out = ts.lines("What are the next big token unlocks in the coming 30 days?", now=NOW)
    assert out is not None
    assert any("24-hour trading volume could not be read" in x for x in out)
    assert "trading volume" not in out[0]


# ------------------------------------------------------------------- plumbing


def test_a_rate_limit_is_waited_out_once(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []
    answers = [RpcError(ErrorKind.RATE_LIMIT, "429", http_status=429), [{"id": "sui"}]]

    def fake(url: str, params: dict[str, Any] | None = None, data: bytes | None = None,
             timeout: float = 25.0) -> Any:
        item = answers.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(ts, "_get", fake)
    monkeypatch.setattr(time, "sleep", waits.append)
    assert ts.markets(["sui"]) == [{"id": "sui"}]
    assert waits == [ts.RATE_WAIT]


def test_other_failures_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(url: str, params: dict[str, Any] | None = None, data: bytes | None = None,
             timeout: float = 25.0) -> Any:
        raise RpcError(ErrorKind.TRANSPORT, "down")

    monkeypatch.setattr(ts, "_get", fake)
    monkeypatch.setattr(time, "sleep", lambda s: pytest.fail("slept"))
    with pytest.raises(RpcError):
        ts.markets(["sui"])


def test_an_unlisted_capital_ticker_is_looked_up(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(url: str, params: dict[str, Any] | None = None, data: bytes | None = None,
             timeout: float = 25.0) -> Any:
        assert url.endswith("/search") and params == {"query": "XYZ"}
        return {"coins": [{"id": "xyz-fake", "symbol": "XYZ", "name": "Fake",
                           "market_cap_rank": 900},
                          {"id": "xyz", "symbol": "XYZ", "name": "Xyz", "market_cap_rank": 30},
                          {"id": "other", "symbol": "XYZA", "name": "Other"}]}

    monkeypatch.setattr(ts, "_get", fake)
    assert ts.coin_ids("What is the max supply of XYZ?") == [("XYZ", "xyz", "Xyz")]
