"""Premises stated in a question are measured and given a verdict, not left to ride along.

Found by running optic-bitget on "Long NVDA perp into earnings — funding looks cheap": it reported
the funding rate, ARGUS answered the earnings half and never looked at the funding claim.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from argus.lui import research
from argus.market import bitget, universe
from argus.market.bitget import Ticker
from argus.research import carry

NOW = datetime(2026, 9, 25, 3, tzinfo=UTC)


def _ticker(symbol: str, funding: str, last: str = "225") -> Ticker:
    d = Decimal
    return Ticker(symbol=symbol, last=d(last), bid=d(last), ask=d(last), high_24h=d(last),
                  low_24h=d(last), change_24h=d(0), base_volume=d(1), funding_rate=d(funding),
                  fetched_at=NOW)


@pytest.fixture
def venue(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"tickers": {}, "history": []}
    monkeypatch.setattr(bitget, "fetch_tickers", lambda: state["tickers"])
    monkeypatch.setattr(carry, "fetch_funding", lambda symbol, pages=1: [
        carry.Settlement(at=NOW, rate_bps=r) for r in state["history"]])
    monkeypatch.setattr(universe, "contracts",
                        lambda: {"MSTRUSDT": universe.Contract("MSTRUSDT", True, funding_hours=8)})
    return state


def test_cheap_funding_that_is_dear_for_this_contract_does_not_hold(venue: dict[str, Any]) -> None:
    venue["tickers"] = {"MSTRUSDT": _ticker("MSTRUSDT", "0.000395")}
    venue["history"] = [0.0] * 90 + [1.0] * 10
    lines, sources = research._claim_check("Long MSTR perp into earnings — funding looks cheap",
                                           "MSTRUSDT") or ([], [])
    assert lines[0].startswith("Your premise that funding looks cheap: does not hold.")
    assert "+0.0395% per 8h" in lines[0] and "43.3% a year" in lines[0]
    assert "higher than 100% of its last 100 settlements" in lines[0]
    assert sources[0].ref == "bitget /api/v3/market/history-fund-rate"


def test_high_funding_that_is_below_its_usual_does_not_hold(venue: dict[str, Any]) -> None:
    venue["tickers"] = {"MSTRUSDT": _ticker("MSTRUSDT", "0.00004")}
    venue["history"] = [0.64] * 80 + [0.1] * 20
    lines, _ = research._claim_check("MSTR funding is high, fade it?", "MSTRUSDT") or ([], [])
    assert "funding looks high: does not hold" in lines[0]


def test_zero_funding_gets_one_verdict_whatever_the_ties(venue: dict[str, Any]) -> None:
    venue["tickers"] = {"MSTRUSDT": _ticker("MSTRUSDT", "0")}
    for history in ([0.0] * 100, [0.0] * 60 + [-0.2] * 34 + [0.3] * 6):
        # Two contracts both at zero, one with more negative settlements, got "holds" and
        # "unclear" on the same run before the rule stopped depending on tied ranks.
        venue["history"] = history
        lines, _ = research._claim_check("cheap funding on MSTR, go long?", "MSTRUSDT") or ([], [])
        assert "holds in absolute terms — a long pays nothing" in lines[0]
        lines, _ = research._claim_check("is MSTR funding expensive?", "MSTRUSDT") or ([], [])
        assert "does not hold — the rate is at or below this contract's usual" in lines[0]


def test_a_rate_in_the_middle_of_its_history_is_unclear(venue: dict[str, Any]) -> None:
    venue["tickers"] = {"MSTRUSDT": _ticker("MSTRUSDT", "0.00005")}
    venue["history"] = [0.2] * 40 + [0.5] * 20 + [0.8] * 40
    lines, _ = research._claim_check("MSTR funding looks cheap", "MSTRUSDT") or ([], [])
    assert "unclear — the rate is ordinary for this contract" in lines[0]


def test_missing_data_says_so_instead_of_judging(venue: dict[str, Any]) -> None:
    lines, _ = research._claim_check("funding looks cheap on MSTR", "MSTRUSDT") or ([], [])
    assert "not checkable" in lines[0]
    venue["tickers"] = {"MSTRUSDT": _ticker("MSTRUSDT", "0.0001")}
    lines, _ = research._claim_check("funding looks cheap on MSTR", "MSTRUSDT") or ([], [])
    assert "unclear — the contract's settlement history did not arrive" in lines[0]


def test_a_premium_claim_is_checked_against_the_stock(
        venue: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    venue["tickers"] = {"NVDAUSDT": _ticker("NVDAUSDT", "0", last="225.0")}
    monkeypatch.setattr(research, "_premium_line", lambda s, last, open_: (
        "Versus the stock: NVDA 224.6 → the perpetual trades at a 17.8bps discount (both live).",
        research.Source(kind="venue", ref="bitget-mcp-server quote", detail="NVDA")))
    lines, _ = research._claim_check("the NVDA perp trades at a premium, short it?",
                                     "NVDAUSDT") or ([], [])
    assert lines[0].startswith("Your premise that the contract trades at a premium: does not hold")


def test_a_question_with_no_premise_is_left_alone() -> None:
    assert research._claim_check("what are NVDA's earnings expectations?", "NVDAUSDT") is None
    assert research._claim_check("funding history for BTC", "BTCUSDT") is None


def _moved(symbol: str, change: str) -> Ticker:
    t = _ticker(symbol, "0")
    return Ticker(**{**{f: getattr(t, f) for f in t.__slots__}, "change_24h": Decimal(change)})


@pytest.mark.parametrize(("text", "change", "verdict"), [
    ("why did TSLA drop today", "0.0022", "does not hold — TSLA is +0.22% over 24 hours, "
                                         "essentially flat and if anything up"),
    ("why did TSLA drop today", "-0.041", "holds — TSLA is -4.10% over 24 hours"),
    ("TSLA is pumping, chase it?", "-0.02", "does not hold — TSLA is -2.00% over 24 hours, down"),
    ("TSLA is up a bit today", "0.001", "barely — TSLA is +0.10% over 24 hours"),
])
def test_a_claimed_move_is_checked_against_the_day(
        venue: dict[str, Any], text: str, change: str, verdict: str) -> None:
    venue["tickers"] = {"TSLAUSDT": _moved("TSLAUSDT", change)}
    lines, _ = research._claim_check(text, "TSLAUSDT") or ([], [])
    assert verdict in lines[0]


def test_a_stated_cause_is_never_confirmed_by_the_price(venue: dict[str, Any]) -> None:
    venue["tickers"] = {"NVDAUSDT": _moved("NVDAUSDT", "0.03")}
    lines, _ = research._claim_check("NVDA is up today because of AI earnings, right?",
                                     "NVDAUSDT") or ([], [])
    assert lines[0].startswith("Your premise that NVDA is up: holds")
    assert "not something a price can confirm" in lines[0]


def test_a_future_direction_is_not_read_as_a_move_that_happened(venue: dict[str, Any]) -> None:
    venue["tickers"] = {"NVDAUSDT": _moved("NVDAUSDT", "0.03")}
    assert research._claim_check("will NVDA go down tomorrow?", "NVDAUSDT") is None
    assert research._claim_check("is NVDA going to drop next week", "NVDAUSDT") is None


@pytest.mark.parametrize(("text", "change", "verdict"), [
    ("META rallied today because of strong earnings", "0.058", "holds — META is +5.80%"),
    ("META dropped today because of a downgrade", "0.058", "does not hold — META is +5.80%"),
])
def test_a_past_tense_claim_with_a_cause_is_read(
        venue: dict[str, Any], text: str, change: str, verdict: str) -> None:
    # MirrorLine read these and ARGUS did not, on the first head-to-head run (2026-09-25).
    venue["tickers"] = {"METAUSDT": _moved("METAUSDT", change)}
    lines, _ = research._claim_check(text, "METAUSDT") or ([], [])
    assert verdict in lines[0] and "not something a price can confirm" in lines[0]


def test_a_scenario_is_not_a_claim_about_the_tape(venue: dict[str, Any]) -> None:
    venue["tickers"] = {"NVDAUSDT": _moved("NVDAUSDT", "0.03")}
    assert research._claim_check("If NVDA dropped 10% what happens to my book?",
                                 "NVDAUSDT") is None


def test_a_session_claim_is_checked_against_the_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(research, "_anchor_open_now", lambda: False)
    lines, _ = research._claim_check("The US stock market is open right now", "") or ([], [])
    assert lines[0].startswith("Your premise that the US market is open: does not hold")
    lines, _ = research._claim_check("NVDA is trading after hours", "") or ([], [])
    assert "holds" in lines[0]


class _Book:
    def __init__(self, slip: str, complete: bool = True, levels: int = 5) -> None:
        from argus.market.depth import Sweep

        self.fetched_at = NOW
        self._sweep = Sweep(side="BUY", requested_notional=Decimal(50_000),
                            filled_notional=Decimal(50_000), average_price=Decimal(1),
                            levels_consumed=levels, slippage_bps=Decimal(slip), complete=complete)

    def sweep(self, notional: Decimal, *, direction: str = "BUY") -> Any:
        return self._sweep


@pytest.mark.parametrize(("text", "book", "verdict"), [
    ("NVDA is liquid", _Book("3.8"), "Your premise that NVDA is liquid: holds — a $50,000"),
    ("NVDA is liquid", _Book("9.0"), "Your premise that NVDA is liquid: unclear"),
    ("NVDA is liquid", _Book("0", complete=False),
     "Your premise that NVDA is liquid: does not hold — the visible 50 levels cannot absorb"),
    ("the NVDA book looks thin", _Book("1.0"), "Your premise that NVDA is thin: does not hold"),
    ("is NVDA liquid enough for a $250k buy?", _Book("12.0"), "Is NVDA liquid enough? Borderline"),
])
def test_a_liquidity_claim_is_measured_on_the_live_book(
        monkeypatch: pytest.MonkeyPatch, text: str, book: _Book, verdict: str) -> None:
    from argus.market import depth

    monkeypatch.setattr(depth, "fetch_orderbook", lambda symbol, **_: book)
    lines, _ = research._claim_check(text, "NVDAUSDT") or ([], [])
    assert lines[0].startswith(verdict)
