"""The trader's stated reasons, each tested (lui/thesis.py; research/h2h/RESULTS.md §4). Inputs
are the shapes the engines return; the SOL figures are the ones read live on 2026-09-28."""

from __future__ import annotations

from typing import Any

import pytest

from argus.lui import thesis
from argus.lui.thesis import Kind, Result, check, month_change, reasons


def long_run(trailing: float, *, up: float = 0.47, base_up: float = 0.51, z: float = -0.3,
             episodes: int = 14) -> dict[str, Any]:
    return {"trailing_return_pct": trailing, "window": 20, "horizons": [
        {"days": 5, "up": up, "base_up": base_up, "z": z, "episodes": episodes,
         "median_bps": 10.0}]}


def quote(*, last: float = 120.288, high: float = 124.933, low: float = 120.072,
          change: float = -0.00907, funding: float = 0.000052) -> dict[str, Any]:
    return {"quotes": {"SOLUSDT": {"last": str(last), "high_24h": str(high), "low_24h": str(low),
                                   "change_24h": str(change), "funding_rate": str(funding)}}}


def data(**parts: Any) -> dict[str, dict[str, Any]]:
    return {"analogue": {"long_run": parts.get("long_run") or long_run(17.9)},
            "quote": parts.get("quote") or quote(),
            "technicals": {"technicals": parts.get("tech") or {"macd_histogram": -0.05,
                                                               "rsi": 45, "timeframe": "4h"}},
            "fundamentals": {"fundamentals": parts.get("fund") or {}}}


class TestReadingTheReasons:
    def test_killmythesis_own_example(self) -> None:
        got = reasons("long SOL, ecosystem activity strengthening, this pullback looks temporary")
        assert [(r.kind, r.text) for r in got] == [
            (Kind.ACTIVITY, "ecosystem activity strengthening"),
            (Kind.REVERSION, "this pullback looks temporary")]

    def test_the_question_and_the_stance_are_not_reasons(self) -> None:
        got = reasons("Should I buy NVDA? It looks cheap and momentum is strong")
        assert [r.kind for r in got] == [Kind.VALUATION, Kind.MOMENTUM]
        assert reasons("Should I enter TSLA?") == ()
        assert reasons("long ETH") == ()

    @pytest.mark.parametrize(("text", "kind"), [
        ("short ETH because funding is crowded long", Kind.POSITIONING),
        ("the rally will fade", Kind.REVERSION),
        ("everyone is too fearful", Kind.SENTIMENT),
        ("earnings will beat", Kind.EARNINGS),
        ("the fed will cut rates", Kind.MACRO),
        ("the new CEO is a proven operator", Kind.OTHER),
    ])
    def test_each_kind_is_recognised(self, text: str, kind: Kind) -> None:
        assert [r.kind for r in reasons(text)] == [kind]


class TestAMoveReversing:
    def test_a_dip_inside_a_rise_is_a_pullback(self) -> None:
        [got] = check(reasons("this pullback looks temporary"), name="SOL", data=data())
        assert got.result is Result.NOT_MEASURABLE
        assert "3.7% below its 24-hour high" in got.line and "+17.9% over 20 days" in got.line

    def test_no_pullback_at_all_contradicts_the_premise(self) -> None:
        flat = quote(last=124.9, high=124.933, change=0.004)
        [got] = check(reasons("this pullback looks temporary"), name="SOL",
                      data=data(quote=flat))
        assert got.result is Result.CONTRADICTED and "no pullback to reverse" in got.line

    def test_history_backing_a_rise_supports_it(self) -> None:
        backed = long_run(-8.0, up=0.72, base_up=0.52, z=2.6)
        [got] = check(reasons("the dip is overdone"), name="SOL", data=data(long_run=backed))
        assert got.result is Result.SUPPORTED and "2.6 standard errors" in got.line

    def test_history_against_a_rise_contradicts_it(self) -> None:
        against = long_run(-8.0, up=0.30, base_up=0.52, z=-2.3)
        [got] = check(reasons("this pullback looks temporary"), name="SOL",
                      data=data(long_run=against))
        assert got.result is Result.CONTRADICTED and "points the other way" in got.line

    def test_a_fading_rally_reads_the_other_direction(self) -> None:
        falls = long_run(12.0, up=0.30, base_up=0.52, z=-2.3)
        [got] = check(reasons("the rally will fade"), name="SOL", data=data(long_run=falls))
        assert got.result is Result.SUPPORTED and "a fall from here" in got.line

    def test_too_few_episodes_is_not_tested(self) -> None:
        thin = long_run(-8.0, episodes=3)
        [got] = check(reasons("the dip is overdone"), name="SOL", data=data(long_run=thin))
        assert got.result is Result.NOT_TESTED


class TestNetworkActivity:
    def test_month_change_places_the_change_in_its_history(self) -> None:
        rows = [(86400 * i, 100.0) for i in range(400)]
        rows += [(86400 * (400 + i), 130.0) for i in range(30)]
        got = month_change(rows)
        assert got is not None and got["change"] == pytest.approx(0.30)
        assert got["percentile"] == 1.0

    def test_strong_fees_and_tvl_support_it(self) -> None:
        strong = {"chain": "Solana", "fees": {"change": 0.327, "percentile": 0.75, "months": 36,
                                              "as_of": "2026-09-28"},
                  "tvl": {"change": 0.191, "percentile": 0.75, "months": 36,
                          "as_of": "2026-09-28"}}
        [got] = check(reasons("ecosystem activity strengthening"), name="SOL", data=data(),
                      activity=strong)
        assert got.result is Result.SUPPORTED
        assert got.line.startswith("The data agrees: Solana daily fees: +32.7%")
        assert "in dollars" in got.line

    def test_a_claim_of_weakening_is_read_the_other_way(self) -> None:
        falling = {"chain": "Solana", "fees": {"change": -0.2, "percentile": 0.1, "months": 36,
                                               "as_of": "2026-09-28"}}
        [got] = check(reasons("network activity is declining"), name="SOL", data=data(),
                      activity=falling)
        assert got.result is Result.SUPPORTED

    def test_mixed_readings_are_not_measurable(self) -> None:
        mixed = {"chain": "Solana", "fees": {"change": 0.3, "percentile": 0.8, "months": 36,
                                             "as_of": "2026-09-28"},
                 "tvl": {"change": -0.05, "percentile": 0.3, "months": 36,
                         "as_of": "2026-09-28"}}
        [got] = check(reasons("adoption is growing"), name="SOL", data=data(), activity=mixed)
        assert got.result is Result.NOT_MEASURABLE

    def test_a_stock_has_no_network(self) -> None:
        [got] = check(reasons("adoption is growing"), name="NVDA", data=data(), activity=None)
        assert got.result is Result.NOT_TESTED and "not a chain's own token" in got.line
        assert thesis.chain_activity("NVDAUSDT") is None


class TestTheFiguresTheTaskHolds:
    def test_momentum_is_read_off_the_tape_and_called_no_edge(self) -> None:
        [got] = check(reasons("momentum is strong"), name="NVDA", data=data())
        assert got.result is Result.CONTRADICTED and "not an edge" in got.line

    def test_funding_tells_which_side_is_crowded(self) -> None:
        crowded = quote(funding=0.0001)
        [got] = check(reasons("funding is crowded long"), name="ETH", data=data(quote=crowded))
        assert got.result is Result.SUPPORTED and "longs pay shorts" in got.line
        [squeeze] = check(reasons("a short squeeze is coming"), name="ETH",
                          data=data(quote=crowded))
        assert squeeze.result is Result.CONTRADICTED

    def test_flat_funding_is_not_measurable(self) -> None:
        [got] = check(reasons("shorts are crowded"), name="ETH", data=data(quote=quote(funding=0)))
        assert got.result is Result.NOT_MEASURABLE

    def test_sentiment_reads_fear_and_greed(self) -> None:
        [got] = check(reasons("everyone is too fearful"), name="BTC", data=data(),
                     fear_greed={"value": 22, "classification": "Extreme Fear"})
        assert got.result is Result.SUPPORTED and "22 (Extreme Fear)" in got.line

    def test_valuation_reads_the_analysts_target_as_opinion(self) -> None:
        fund = {"target_mean": 327.7, "price": 225.07}
        [got] = check(reasons("It looks cheap"), name="NVDA", data=data(fund=fund))
        assert got.result is Result.SUPPORTED and "an opinion, not a measurement" in got.line

    def test_what_no_engine_reads_is_said_to_be_untested(self) -> None:
        [got] = check(reasons("the new CEO is a proven operator"), name="NVDA", data=data())
        assert got.result is Result.NOT_TESTED and "yours to judge" in got.line


def test_the_page_and_the_json_show_each_reason_with_its_result() -> None:
    from argus.lui.task import STEPS, Step, Task, as_dict
    from argus.lui.task_page import render_task

    tested = check(reasons("long SOL, this pullback looks temporary"), name="SOL", data=data())
    steps = [Step(title=t, engine=e, lines=["Bottom line: x"]) for t, _, e in STEPS]
    task = Task(question="q", name="SOL", size_pct=20.0, book={}, steps=steps, seconds=1.0,
                asked="long SOL, this pullback looks temporary", tested=tested)
    page = render_task(task, "")
    assert "Your reasons, tested" in page
    assert "“this pullback looks temporary”" in page
    assert "<span class='r not_measurable'>not measurable</span>" in page
    assert as_dict(task)["reasons_tested"][0]["result"] == "not measurable"
