"""The research-kind model at runtime, and a planner that lets it stand in for the language model.

`eval/kindtrain.py` fits a character n-gram linear model over fourteen classes — the thirteen
research kinds, ``record`` and ``refuse`` — and exports it as JSON. Scoring reuses
`lui/ngram.NgramClassifier` (stdlib only), so the deployed bundle gains no dependency.

**How it is used.** The console already has a validated path for a model's reading of a question:
`research.plan_with_model` takes a JSON plan (kind, confidence, names, holdings, shock, size),
checks every field and builds the request. :class:`LocalPlanner` produces that same plan with no
network call: the *kind* from this model, and every *figure* — names, weights, the shock — from the
deterministic extractors the patterns already use. So the model decides which engine answers and
nothing else, exactly as the language model is allowed to, and the hosted console reads questions
the patterns cannot place without spending a token.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from argus.lui.ngram import NgramClassifier

MODEL_PATH = Path(__file__).resolve().parents[3] / "data" / "lui_kind_model.json"
REFUSE = "refuse"
RECORD = "record"


class KindModel:
    """Predicts one of the fourteen labels, or None below the fitted threshold."""

    def __init__(self, classifier: NgramClassifier) -> None:
        self._clf = classifier

    @classmethod
    def load(cls, path: Path | None = None) -> KindModel:
        return cls(NgramClassifier.load(path or MODEL_PATH))

    @property
    def threshold(self) -> float:
        return self._clf.threshold

    def predict(self, text: str) -> tuple[str | None, float]:
        ranked = self._clf.probabilities(text)
        if not ranked:
            return None, 0.0
        confidence, label = ranked[0]
        return (label if confidence >= self._clf.threshold else None), confidence


_CACHED: KindModel | None = None
_LOADED = False


def kind_model() -> KindModel | None:
    """The shipped model, loaded once; None when the file is absent (the patterns then stand)."""
    global _CACHED, _LOADED
    if not _LOADED:
        _LOADED = True
        try:
            _CACHED = KindModel.load()
        except (OSError, ValueError, KeyError):
            _CACHED = None
    return _CACHED


class LocalPlanner:
    """A drop-in for the language-model router's ``complete_json``, answered by :class:`KindModel`
    and the pattern extractors. It never invents a number: every figure it returns was read from
    the question by the same code the patterns use."""

    def __init__(self, model: KindModel) -> None:
        self._model = model

    def complete_json(self, messages: list[dict[str, str]], **_: Any) -> dict[str, Any]:
        from argus.lui import research

        text = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        label, confidence = self._model.predict(text)
        if label is None:
            return {"kind": "none", "confidence": 0.0, "why": "below the kind model's threshold"}
        if label in (REFUSE, RECORD):
            # Full confidence because the model already cleared its own threshold; the console's
            # "declined by the model" rule reads this field on the same terms as the LLM's.
            return {"kind": "none" if label == REFUSE else "record",
                    "confidence": 1.0, "why": f"kind model: {label} at {confidence:.2f}"}
        # A percentage beside a shock word or signed ("a -10% NVDA shock", "qqq -10% scenario")
        # is the shock, not a holding weight; the patterns draw the same line
        # (`research._named_shock_request`).
        pairs = [(pos, symbol, weight) for pos, symbol, weight in research._pairs(text)
                 if not research._SHOCK_WEIGHT.search(text[max(0, pos - 2): pos + 16])
                 and not text[max(0, pos - 1): pos + 1].startswith("-")]
        holdings: dict[str, float] = {}
        for _pos, symbol, weight in pairs:
            holdings[symbol] = holdings.get(symbol, 0.0) + weight * 100.0
        named = list(research.research_symbols(text)[0])
        candidate = next((s for s in named if s not in holdings), named[0] if named else None)
        plan: dict[str, Any] = {
            "kind": label,
            # The plan passes `plan_with_model`'s confidence floor only because the kind model
            # already cleared its own, CV-chosen threshold; its softmax is on a different scale.
            "confidence": 1.0,
            "why": f"kind model: {label} at {confidence:.2f}",
            "names": named,
            "holdings": holdings,
        }
        if candidate is not None:
            plan["candidate"] = candidate
        # The raw reading, before cash and leverage adjustments: `plan_with_model` applies those
        # to the plan once, and taking them here too applied a 3x multiple twice (90% for 10%).
        patterned = research._detect(text)
        if (patterned is not None and patterned.size_stated and patterned.symbols
                and patterned.symbols[0] == candidate):
            # "what does adding 15% TSLA do to my risk?" was answered for 20%: the 15% was read as
            # a holding, and the plan carried no size, so the default applied (found through the
            # Telegram bot's own help examples, 2026-09-25). The patterns' size reading is exact.
            plan["size_percent"] = patterned.size * 100.0
            holdings.pop(candidate, None)
        notional = research._parse_notional(text)
        if label == "execution" and notional is not None:
            plan["order_usd"] = str(notional)
        # The shock is a percentage that is not a holding weight: "I hold 50% BTC, 30% ETH, how
        # risky is my book" was stressed with a +50% Nasdaq move read from "50% BTC".
        weights = {pos for pos, _, _ in pairs}
        shock = next((m for m in research._SHOCK_NUMBER.finditer(text)
                      if not any(abs(pos - m.start()) < 16 for pos in weights)), None)
        if label == "stress" and shock is not None:
            value = abs(float(shock.group(1)))
            down = research._DOWN_WORDS.search(text) or shock.group(1).startswith("-")
            plan["shock_percent"] = -value if down else value
        return plan


__all__ = ["KindModel", "LocalPlanner", "kind_model"]
