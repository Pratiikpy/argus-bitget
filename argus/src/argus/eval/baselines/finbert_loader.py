"""Loads the REAL, published `ProsusAI/finbert` inference model — not vendored source.

Unlike every other module in this package, there is no upstream `.py` file to copy byte-for-byte:
FinBERT is a HuggingFace Hub *model* (fine-tuned BERT weights), and the standard, documented way
to use it is exactly what this module does — `transformers.pipeline("sentiment-analysis",
model="ProsusAI/finbert")`, the same call the model's own README (fetched and read at
`https://huggingface.co/ProsusAI/finbert/blob/main/README.md`) shows as its usage example. Nothing
is copied into this repository; the weights are downloaded at runtime from the Hub the same way
any `transformers` user would get them, and the model card carries no license field to vendor
against — the reasonable posture here is standard runtime USE of a published public inference
model (50k+ monthly downloads, explicitly published for exactly this), not redistribution, so the
"vendor, byte-verify, pin a hash" playbook this package's other loaders follow does not apply.

Source: https://github.com/ProsusAI/finBERT (Apache-2.0 — the training/eval CODE repository) and
https://huggingface.co/ProsusAI/finbert (the published weights this module actually loads).
Read at commit 44995e0c5870c4ab37a189d756550654ae87cdf0 for the paper/architecture context
(`eval/sentiment_comparison.py`'s own docstring cites what was taken from it).
"""

from __future__ import annotations

from typing import Any, Protocol

_PIPELINE: Any = None


class FinbertClassifier(Protocol):
    def __call__(self, texts: list[str]) -> list[dict[str, Any]]: ...


def load_finbert() -> FinbertClassifier:
    """The real `ProsusAI/finbert` sentiment pipeline, loaded once per process and cached.

    Returns a callable: `classifier(["some financial text", ...])` ->
    `[{"label": "positive"|"negative"|"neutral", "score": float}, ...]`, exactly finBERT's own
    real three-way softmax output — not remapped or reinterpreted here.
    """
    global _PIPELINE
    if _PIPELINE is None:
        from transformers import pipeline

        # "text-classification" is the canonical task name; "sentiment-analysis" (the name in
        # finBERT's own README example) is a runtime-only alias for the identical pipeline that
        # transformers' own type stubs do not declare as a Literal overload — using the
        # canonical name keeps this file mypy --strict clean without changing behaviour.
        _PIPELINE = pipeline("text-classification", model="ProsusAI/finbert")
    result: FinbertClassifier = _PIPELINE
    return result


__all__ = ["FinbertClassifier", "load_finbert"]
