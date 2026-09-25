"""A held name set to a final weight is a resize, not an add; a stated add size is kept."""

from __future__ import annotations

from typing import Any

import pytest

from argus.desk.portfolio import PortfolioError, rebalance, resize
from argus.lui import research
from argus.lui.kindmodel import LocalPlanner, kind_model
from argus.lui.research import ResearchKind, _detect, _resize, _resolve_resize, pattern_reading_wins
from argus.market import history, universe


@pytest.fixture(autouse=True)
def frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("live fetch disabled in tests")

    monkeypatch.setattr(research, "_fetch_live", _fail)
    monkeypatch.setattr(history, "fetch", _fail)
    monkeypatch.setattr(universe, "_fetch_live", _fail)
    monkeypatch.setattr(universe, "_CACHE", None)


BOOK = {"TSLAUSDT": 0.3, "NVDAUSDT": 0.4, "AAPLUSDT": 0.3}


def test_resize_sets_the_final_weight_and_keeps_the_book_whole() -> None:
    after = resize(BOOK, "TSLAUSDT", 0.1)
    assert after["TSLAUSDT"] == pytest.approx(0.1)
    assert sum(after.values()) == pytest.approx(1.0)
    assert after["NVDAUSDT"] / after["AAPLUSDT"] == pytest.approx(0.4 / 0.3)
    assert resize(BOOK, "TSLAUSDT", 0.0)["TSLAUSDT"] == 0.0
    with pytest.raises(PortfolioError):
        resize({"TSLAUSDT": 1.0}, "TSLAUSDT", 0.5)
    # the add semantics stay what they were: 15% on top of a scaled 30%
    assert rebalance(BOOK, "TSLAUSDT", 0.15)["TSLAUSDT"] == pytest.approx(0.3 * 0.85 + 0.15)


@pytest.mark.parametrize(("text", "expected"), [
    ("trimming my TSLA weight to 10% of book — is that enough", (0.10, None, None)),
    ("resizing NVDA from 8% to 20% of the book", (0.20, 0.08, None)),
    ("把英伟达从15%降到5%,对整体beta有多大帮助", (0.05, 0.15, None)),
    ("如果我把比特币仓位从10%提到25%,整体风险怎么变化", (0.25, 0.10, None)),
    ("trim TSLA by 30%, what happens to my risk?", (None, None, -0.30)),
    ("voy a cerrar 30% de mi posicion en ETHUSDT", (None, None, -0.30)),
    ("raise my NVDA position by 50%", (None, None, 0.50)),
])
def test_resize_phrasings_are_read(text: str, expected: tuple[Any, ...]) -> None:
    got = _resize(text)
    assert got is not None
    for value, want in zip(got, expected, strict=True):
        assert value == (pytest.approx(want) if want is not None else None)


@pytest.mark.parametrize("text", [
    "what does adding 15% TSLA do to my risk?",
    "if the Nasdaq drops 10%, what happens to my book?",
    "keep any name under 15% of my risk",
])
def test_adds_and_shocks_are_not_resizes(text: str) -> None:
    request = _detect(text)
    assert request is None or (request.target is None and request.resize_by is None)


def test_a_resize_request_wins_over_the_models_reading() -> None:
    request = _detect("trimming my TSLA weight to 10% of book — is that enough")
    assert request is not None and request.kind is ResearchKind.IMPACT
    assert request.target == pytest.approx(0.1)
    assert pattern_reading_wins(request, "trimming my TSLA weight to 10% of book")


def test_resolving_needs_the_book_and_honours_a_stated_from() -> None:
    bare = research.ResearchRequest(kind=ResearchKind.IMPACT, symbols=("TSLAUSDT",), target=0.1)
    assert isinstance(_resolve_resize(bare, {}), str)
    stated = research.ResearchRequest(kind=ResearchKind.IMPACT, symbols=("NVDAUSDT",),
                                      target=0.2, resize_from=0.08)
    resolved = _resolve_resize(stated, dict(BOOK))
    assert not isinstance(resolved, str) and resolved is not None
    book, target, line = resolved
    assert book["NVDAUSDT"] == pytest.approx(0.08) and sum(book.values()) == pytest.approx(1.0)
    assert target == 0.2 and line.startswith("Resizing NVDA from 8% to 20%")
    by = research.ResearchRequest(kind=ResearchKind.IMPACT, symbols=("TSLAUSDT",),
                                  resize_by=-0.3)
    resolved = _resolve_resize(by, dict(BOOK))
    assert not isinstance(resolved, str) and resolved is not None
    assert resolved[1] == pytest.approx(0.21) and "a 30% cut of the position" in resolved[2]
    whole = research.ResearchRequest(kind=ResearchKind.IMPACT, symbols=("TSLAUSDT",),
                                     resize_by=3.0)
    assert "leaves nothing else" in str(_resolve_resize(whole, dict(BOOK)))


def test_the_local_planner_keeps_a_stated_add_size() -> None:
    model = kind_model()
    if model is None:
        pytest.skip("kind model export not present")
    planned, _ = research.plan_with_model("what does adding 15% TSLA do to my risk?",
                                          LocalPlanner(model))
    assert planned is not None and planned.kind is ResearchKind.IMPACT
    assert planned.size == pytest.approx(0.15) and planned.size_stated
