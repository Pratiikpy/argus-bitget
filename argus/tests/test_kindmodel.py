"""The research-kind model's runtime: it chooses the engine, never a number."""

from __future__ import annotations

from typing import Any

import pytest

from argus.lui import research
from argus.lui.kindmodel import KindModel, LocalPlanner
from argus.lui.ngram import NgramClassifier
from argus.lui.research import ResearchKind, plan_with_model
from argus.market import history, universe


@pytest.fixture(autouse=True)
def frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("live fetch disabled in tests")

    monkeypatch.setattr(research, "_fetch_live", _fail)
    monkeypatch.setattr(history, "fetch", _fail)
    monkeypatch.setattr(universe, "_fetch_live", _fail)
    monkeypatch.setattr(universe, "_CACHE", None)


class _Fixed(KindModel):
    """A model that answers one label at one confidence, so the planner is tested alone."""

    def __init__(self, label: str | None, confidence: float = 0.9) -> None:
        self._label, self._confidence = label, confidence

    def predict(self, text: str) -> tuple[str | None, float]:
        return self._label, self._confidence


def _plan(label: str | None, text: str) -> Any:
    return plan_with_model(text, LocalPlanner(_Fixed(label)))


def test_the_kind_comes_from_the_model_and_the_figures_from_the_text() -> None:
    request, audit = _plan("stress", "tech dumps 12% tomorrow, I hold 60% NVDA and 40% MSFT")
    assert request is not None and request.kind is ResearchKind.STRESS
    assert audit["applied"] is True
    assert request.shock_pct == -12.0
    assert dict(request.book) == pytest.approx({"NVDAUSDT": 0.6, "MSFTUSDT": 0.4})


def test_a_compare_takes_every_named_instrument() -> None:
    request, _ = _plan("compare", "nvda or tsla, which one swings harder")
    assert request is not None and request.kind is ResearchKind.COMPARE
    assert set(request.symbols) >= {"NVDAUSDT", "TSLAUSDT"}


def test_refuse_and_record_build_no_research_request() -> None:
    assert _plan("refuse", "what's a good pasta recipe")[0] is None
    assert _plan("record", "how did your trades do last week")[0] is None
    assert _plan(None, "asdf qwer")[0] is None


def test_the_planner_never_supplies_a_number_the_text_does_not_contain() -> None:
    planner = LocalPlanner(_Fixed("impact"))
    plan = planner.complete_json([{"role": "user", "content": "should I add some TSLA"}])
    numbers = {k: v for k, v in plan.items() if k in ("size_percent", "shock_percent")}
    assert numbers == {}
    assert plan["holdings"] == {}


def test_the_threshold_decides_whether_the_model_answers() -> None:
    blob = {"classes": ["impact", "refuse"], "min_n": 1, "max_n": 3, "threshold": 0.99,
            "vocabulary": {"a": 0}, "idf": [1.0], "coef": [[1.0], [0.0]], "intercept": [0.0, 0.0]}
    model = KindModel(NgramClassifier(blob))
    label, confidence = model.predict("a")
    assert label is None and 0.0 < confidence < 0.99


def test_the_shipped_model_carries_the_fourteen_labels_and_a_cv_chosen_threshold() -> None:
    from argus.eval.kindtrain import LABELS
    from argus.lui.kindmodel import kind_model

    model = kind_model()
    assert model is not None
    assert sorted(model._clf.classes) == sorted(LABELS)
    assert 0.0 < model.threshold <= 0.5


@pytest.mark.parametrize(("text", "label"), [
    ("how should I split a $200k NVDA buy to keep slippage down", "execution"),
    ("what's the funding rate on BTC right now", "quote"),
    ("why did the risk layer block your last TSLA trade", "record"),
    ("write me a python script to sort a list", "refuse"),
    ("英伟达的RSI现在超买了吗", "technicals"),
])
def test_the_shipped_model_reads_plain_questions(text: str, label: str) -> None:
    from argus.lui.kindmodel import kind_model

    model = kind_model()
    assert model is not None
    assert model.predict(text)[0] == label


def test_the_threshold_rule_prefers_abstaining_to_confident_errors() -> None:
    from argus.eval.kindtrain import choose_threshold

    oof = [("a", "a", 0.9)] * 10 + [("a", "b", 0.2)] * 5 + [("a", "a", 0.2)] * 2
    threshold, stats = choose_threshold(oof)
    # Answering the 0.2 band gains 2 correct and costs 5 x 3: the rule stops above it.
    assert threshold > 0.2 and stats["wrong"] == 0
