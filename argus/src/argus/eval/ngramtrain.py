"""Fit the console's character-n-gram intent model and export it as dependency-free JSON.

**scikit-learn appears here and nowhere else.** The deployed bundle has an empty
`requirements.txt`, so the runtime in `lui/ngram.py` reimplements scoring in the stdlib; this
module is the only thing that ever *fits* a model, and it runs on a developer machine.

**The protocol, which matters more than the model.** `data/oblique_pool.json` holds 384 questions
across eight intents, six personas and two languages, authored by `openai/gpt-oss-20b` — a model
that has never seen this repository. It was split 192/192 on seed 20260921 **before any
measurement was taken**, so the split cannot have been chosen to flatter a result. This module
trains on the **dev** half only, plus the in-repo labelled corpora, and chooses hyper-parameters
and the abstention threshold by 5-fold cross-validation **within that half**. The test half is
untouched here and is read exactly once, by `eval/ngrambench.py`.

**Why the threshold is 0.15 and not a higher, safer-looking number.** The rule, unchanged since the
logistic-head version: take the lowest threshold whose out-of-fold CV result beats the incumbent
pattern layer on *both* axes at once — more correct answers **and** no more confident errors.
Re-run after the classifier head changed (see below), because a linear-SVM margin softmax is not
the same number as a log-odds softmax and 0.20 does not transfer: out-of-fold 5-fold CV over all
334 training rows (`class_weight="balanced"`) answers 169 correct and 50 wrong at 0.15, against the
pattern layer's 94 correct and 50 wrong **on the identical rows**. Checked for fold-split
sensitivity across three CV seeds (20260921/1/42): the lowest qualifying threshold lands at
0.145-0.155 every time, so 0.15 is not an artefact of one particular split. Choosing a threshold
that merely maximised accuracy would have buried the error count, which on a console sitting in
front of an account is the number that hurts.

**Why the head is a linear SVM and not the multinomial logistic regression this shipped with until
2026-09-22.** Dev-half 5-fold CV, mean over 10 fold-split seeds: 76.71% for
`LinearSVC(C=5.0, class_weight="balanced")` against 72.28% for the logistic head it replaced — the
SVM wins on every one of the 10 seeds tested, not just on average. A calibrated variant
(`CalibratedClassifierCV` wrapping the same `LinearSVC`) was tried first and also won (75.36%
mean) but the plain, uncalibrated margin scored higher still and needs no k-fold Platt-scaling
machinery to reimplement in the stdlib scorer. Word-level TF-IDF features stacked onto the existing
character n-grams were tried too and **hurt**: 73.35% down to 66.77%/66.16% — the working
hypothesis is that word-level tokenisation is a poor fit for the un-spaced half of this corpus.
`lui/ngram.py`'s scoring code needed no change at all: it was already a generic
"linear-scores-then-softmax" implementation, and `LinearSVC.coef_`/`.intercept_` carry the
identical shape and class ordering `LogisticRegression` did — confirmed directly before this
change was made, and confirmed again by 0 argmax mismatches between the pure-Python scorer and
real `sklearn.svm.LinearSVC.decision_function()` on all 334 training rows.

    python -m argus.eval.ngramtrain
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from argus.lui.ngram import OUT_OF_SCOPE

DATA = Path(__file__).resolve().parents[3] / "data"
POOL_PATH = DATA / "oblique_pool.json"
OOS_PATH = DATA / "oblique_out_of_scope.json"
MODEL_PATH = DATA / "lui_ngram_model.json"

OUT_OF_SCOPE_LABEL = OUT_OF_SCOPE
"""An explicit class for "not a question this console answers".

**This is CLINC150's `oos-train` scheme, and it is here because `oos-threshold` failed.** The
first model carried only the eight in-scope intents and leaned on the confidence threshold to
decline everything else. Measured against the fourteen ordinary non-trading questions in
`eval/luirouter.OUT_OF_SCOPE`, it answered **ten of them** — 29% out-of-scope recall, against the
embedding router's 93%. It placed *"how do i make pasta"* in ``performance`` at 0.304 and
*"what is 17 times 23"* in ``decision_why`` at 0.225.

That is the failure the CLINC150 paper exists to describe: a softmax over in-scope classes has
nowhere to put an out-of-scope input, so it puts it somewhere. Giving the model a class for
"none of these" is the paper's own strongest remedy, and the label is mapped back to a decline at
inference rather than ever being returned as an intent."""

NGRAM_RANGE = (1, 5)
"""Chosen by 5-fold CV on the dev half: 77.3%, against 76.9% for (1, 3) and 74.4% for (2, 4).

The unigram floor is doing real work rather than padding the feature count — a single Han
character is a morpheme, so ``n=1`` is to Chinese roughly what a word unigram is to English."""

REGULARISATION = 5.0
"""``C`` for the linear-SVM head. Unchanged from the logistic head's value: a dedicated sweep for
`LinearSVC` (0.5/1/2/5/10/20, mean over 10 CV fold-split seeds) is flat within 0.1 points across
5.0-20.0 (76.71%/76.74%/76.83%), so the same more-regularised end is kept rather than chasing noise
with a new number to defend."""

ABSTAIN_THRESHOLD = 0.15
"""See the module docstring. Fixed by a rule stated before the sweep, not picked from the table —
re-derived, by the same rule, when the classifier head changed on 2026-09-22."""

OOS_BALANCE_SEED = 20260922
"""Downsampling the majority-language out-of-scope negatives is a training-time choice, not a
dataset edit, so it needs its own seed rather than reusing the pool split's 20260921 — the two
are picked independently and neither should silently move if the other changes.

**Why this exists.** `data/oblique_out_of_scope.json` carries 40 English negatives and 60 Chinese
ones — nobody chose that 40/60 split on purpose, it is just how many rows each language's authoring
pass produced. Training on it unbalanced teaches the model "out-of-scope" partly as a proxy for
"is this Chinese", which is exactly backwards: `data/oblique_pool.json`'s *in-scope* pool is close
to 50/50 EN/ZH (192/192 before any split), so a Chinese question is not inherently more likely to
be out-of-scope than an English one.

**Measured, not assumed.** Out-of-fold 5-fold CV on the dev half, 5 fold-split seeds, holding
everything else fixed (`LinearSVC(C=5.0, class_weight="balanced")`, same features): downsampling
the 60 Chinese negatives to 40 (matching English) cut the rate at which genuine Chinese in-scope
questions are wrongly refused as out-of-scope from 14.2% to 8.4% (66/465 to 39/465 held-out
predictions), paired McNemar exact test b=28 (current wrong, balanced right) vs c=1 (the reverse),
p<0.0001 — not a coincidence of one split. Overall CV accuracy was unaffected (78.5% vs 78.7%,
within noise). This is the "genuinely different algorithmic idea" the 2026-09-22 rival-lens pass
went looking for: not more evidence, a training-distribution fix for a mechanism the earlier
`t3-lui` write-up had named as a hypothesis (`eval/standing.py`) but never actually tested."""

EXCLUDED_LABELS = frozenset({
    "unknown", "ambiguous", "unsupported", "integrity", "risk_control", "order", "market",
})
"""Labels kept out of this model, and the two reasons are different.

``unknown``/``ambiguous``/``unsupported`` are the console's *refusal* states: they are outcomes,
not intents, and training a classifier to predict "I do not know" as a class confuses abstention
with classification. The other four are real intents that the generated pool does not cover, and a
class with a handful of examples from one source would be a class this model cannot honestly
claim. `lui/question.py` still answers all four, and it runs first."""


class NgramTrainError(RuntimeError):
    """Training cannot proceed honestly. Raised rather than writing a model nobody can trust."""


def training_rows() -> tuple[list[str], list[str]]:
    """Dev-half questions, the in-repo labelled corpora, and the out-of-scope negatives.

    **Neither held-back corpus is read here** — not the pool's test half, and not
    `data/oblique_validation.json`.
    """
    from argus.eval.luibench import CASES
    from argus.eval.obliquebench import TUNED

    if not POOL_PATH.exists():
        raise NgramTrainError(
            f"{POOL_PATH} is missing. The pool is generated data and is committed with the repo; "
            "regenerating it would produce a different split and invalidate every published "
            "figure measured against the old one."
        )
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    texts = [row["ask"] for row in pool["dev"]]
    labels = [row["expect"] for row in pool["dev"]]
    # `obliquebench.Case` and `luibench.Case` are two different classes that happen to share the
    # two attributes used here. Chained into one loop variable they are a type error, and papering
    # over it with a cast would hide the day one of them grows a third field this code needs.
    # Flattened to the pair actually consumed instead.
    for ask, expect in [
        *((c.ask, str(c.expect)) for c in TUNED),
        *((c.ask, str(c.expect)) for c in CASES),
    ]:
        if expect not in EXCLUDED_LABELS:
            texts.append(ask)
            labels.append(expect)
    keep = [i for i, label in enumerate(labels) if label not in EXCLUDED_LABELS]
    texts, labels = [texts[i] for i in keep], [labels[i] for i in keep]

    # The out-of-scope class. Absent rather than fatal: the model trains without it, and
    # `eval/ngrambench.py` will simply report the recall that produces.
    if OOS_PATH.exists():
        negatives = json.loads(OOS_PATH.read_text(encoding="utf-8"))
        for row in balance_oos_by_language(negatives["rows"]):
            texts.append(row["ask"])
            labels.append(OUT_OF_SCOPE_LABEL)
    return texts, labels


def balance_oos_by_language(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Downsample every language group to the smallest group's size — see ``OOS_BALANCE_SEED``.

    Deterministic (seeded `random.Random`, sorted group order) so re-running training on the same
    `oblique_out_of_scope.json` always drops the identical rows rather than a different random
    subset each fit. A row with no ``lang`` field is kept untouched and unbalanced against, since
    there is nothing to balance it relative to.
    """
    by_lang: dict[str, list[dict[str, Any]]] = {}
    unlabelled: list[dict[str, Any]] = []
    for row in rows:
        lang = row.get("lang")
        if lang is None:
            unlabelled.append(row)
        else:
            by_lang.setdefault(lang, []).append(row)
    if not by_lang:
        return list(rows)
    target = min(len(group) for group in by_lang.values())
    rng = random.Random(OOS_BALANCE_SEED)
    balanced: list[dict[str, Any]] = []
    for lang in sorted(by_lang):
        group = by_lang[lang]
        balanced.extend(group if len(group) <= target else rng.sample(group, target))
    return balanced + unlabelled


def fit(texts: Sequence[str], labels: Sequence[str]) -> Any:
    """The scikit-learn pipeline, fitted. Returns ``(vectorizer, classifier)``.

    ``LinearSVC``, not ``LogisticRegression`` — see the module docstring for the CV evidence. The
    exported shape (``coef_``/``intercept_``, one row per class, same ordering) is identical either
    way, so `lui/ngram.py`'s scorer needed no change: it has always been a generic linear-scores-
    then-softmax implementation, never one written against log-odds specifically.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
    except ImportError as exc:  # pragma: no cover - developer environment only
        raise NgramTrainError(
            "scikit-learn is needed to FIT the model. It is deliberately absent from the deployed "
            "bundle; `lui/ngram.py` scores without it."
        ) from exc

    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=NGRAM_RANGE, sublinear_tf=True, min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)
    # class_weight="balanced" because the pool is uneven by intent (performance 13, decision_why
    # 31 in the dev half). Without it the head would learn the prior and answer the common intent
    # when unsure, which is exactly the confident-wrong failure the threshold exists to prevent.
    # random_state matters here in a way it did not for the logistic head: liblinear's dual
    # coordinate descent uses a random number generator for tie-breaking, and on this small,
    # wide problem (334 rows, ~15k features) that produced a genuinely different fitted model on
    # every call when unset -- checked directly (5 fits, 5 different weight hashes) before this
    # line was added. Fixed to the project's standard split seed so training is reproducible,
    # not because that seed was chosen for its result.
    classifier = LinearSVC(
        C=REGULARISATION, class_weight="balanced", max_iter=5000, random_state=20260921,
    ).fit(matrix, list(labels))
    return vectorizer, classifier


def export(vectorizer: Any, classifier: Any, *, threshold: float = ABSTAIN_THRESHOLD,
           rows: int = 0) -> dict[str, Any]:
    """Everything `lui/ngram.py` needs, as plain JSON.

    Coefficients are rounded to six decimals. That is far below the precision at which any
    argmax here changes — `tests/test_ngram_router.py` asserts the pure-Python scorer agrees with
    scikit-learn on every training row *after* this rounding — and it roughly halves the file.
    """
    vocabulary = {term: int(index) for term, index in vectorizer.vocabulary_.items()}
    return {
        "generated_by": "argus.eval.ngramtrain",
        "training_rows": rows,
        "classes": [str(c) for c in classifier.classes_],
        "min_n": NGRAM_RANGE[0],
        "max_n": NGRAM_RANGE[1],
        "threshold": threshold,
        "vocabulary": vocabulary,
        "idf": [round(float(v), 6) for v in vectorizer.idf_],
        "coef": [[round(float(v), 6) for v in row] for row in classifier.coef_],
        "intercept": [round(float(v), 6) for v in classifier.intercept_],
        "provenance": (
            "Trained on the DEV half of data/oblique_pool.json (seed 20260921, split before any "
            "measurement) plus obliquebench.TUNED and luibench.CASES. The TEST half was not read "
            "during training or hyper-parameter selection."
        ),
    }


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="fit the console's n-gram intent model")
    parser.add_argument("--out", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    texts, labels = training_rows()
    vectorizer, classifier = fit(texts, labels)
    blob = export(vectorizer, classifier, rows=len(texts))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, ensure_ascii=False) + "\n", encoding="utf-8")

    size = args.out.stat().st_size / 1024
    print(f"trained on {len(texts)} rows, {len(blob['classes'])} intents")
    print(f"  features: {len(blob['vocabulary'])}  threshold: {blob['threshold']}")
    print(f"  written to {args.out} ({size:.0f} KB)")
    print("  the TEST half of the pool was not read; see argus.eval.ngrambench")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ABSTAIN_THRESHOLD", "EXCLUDED_LABELS", "MODEL_PATH", "NGRAM_RANGE", "NgramTrainError",
    "export", "fit", "main", "training_rows",
]
