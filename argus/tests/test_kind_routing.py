"""The console's engine routing on the blind question sets (`eval/kind_routing.py`).

The three 2026-09-27 fixes are pinned on phrasings written for this file, not taken from the
scored sets, so the tests say whether the fix generalises rather than whether it memorised a row.
No market read and no model call: every route runs inside the evaluator's offline context.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from argus.eval import kind_routing
from argus.eval.record_routing import _offline


@pytest.fixture(scope="module")
def offline() -> Iterator[None]:
    with _offline():
        yield


@pytest.mark.parametrize(("question", "kind"), [
    # an instruction to trade is refused, however it is phrased
    ("execute a sell order on all my TSLA now", "refuse"),
    ("execute the trade for 500 NVDA", "refuse"),
    # ... while a question about execution still reaches the execution engine
    ("how well do you execute large orders", "execution"),
    # a hedge named against a risk asks for the hedge
    ("hedge my NVDA against a tech selloff", "hedge"),
    ("please hedge my portfolio against inflation", "hedge"),
    # "why is it pumping" asks what moved it; "pumped on twitter" asks about the crowd
    ("why is ETH pumping today", "news"),
    ("why did MSTR pump this morning", "news"),
    ("is COIN getting pumped on twitter", "sentiment"),
    ("why is the market so bearish today", "sentiment"),
    ("stress test my book for a 20% nasdaq crash, what's the drawdown", "stress"),
    ("what's the max drawdown of the desk so far", "record"),
])
def test_a_question_reaches_the_engine_it_asks_for(offline: None, question: str,
                                                   kind: str) -> None:
    assert kind_routing.route(question)[0] == kind


def test_a_refusal_is_read_from_the_consoles_own_reply(offline: None) -> None:
    """An unlisted ticker is answered with a refusal under the market intent; the flag decides."""
    assert kind_routing.route("what's the price of QWXZ coin")[0] == "refuse"
    assert kind_routing.route("asdkjh qwe zzz")[0] == "refuse"


def test_a_coin_named_by_its_ticker_is_not_coinbase() -> None:
    from argus.lui.research import detect

    request = detect("what's the price of QWXZ coin")
    assert request is None or "COINUSDT" not in request.symbols
    macd = detect("macd on coin")
    assert macd is not None and macd.symbols == ("COINUSDT",)


def test_the_score_counts_rows_by_kind_and_language(offline: None) -> None:
    rows = [{"id": 1, "lang": "en", "text": "why is ETH pumping today", "expected": "news"},
            {"id": 2, "lang": "en", "text": "execute the trade for 500 NVDA",
             "expected": "execution"}]
    report = kind_routing.score(rows)
    assert (report["correct"], report["rows"]) == (1, 2)
    assert report["by_language"] == {"en": [1, 2]}
    assert report["by_kind"] == {"execution": [0, 1], "news": [1, 1]}
    assert report["wrong_by_layer"] == {"patterns": 1}


def test_the_published_artefact_adds_up() -> None:
    """Every total in the committed report is the count of its own rows."""
    report = json.loads(kind_routing.REPORT_PATH.read_text(encoding="utf-8"))
    assert list(report) == [name for name, _ in kind_routing.CORPORA]
    for blob in report.values():
        results = blob["results"]
        assert blob["rows"] == len(results)
        assert blob["correct"] == sum(r["right"] for r in results)
        assert all(r["right"] == (r["got"] == r["expected"]) for r in results)
        assert sum(total for _, total in blob["by_kind"].values()) == len(results)
