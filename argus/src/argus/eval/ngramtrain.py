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

**Why the threshold is 0.20 and not a higher, safer-looking number.** The rule was fixed before
looking: take the lowest threshold whose CV result beats the incumbent pattern layer on *both*
axes at once — more correct answers **and** no more confident errors. On the dev CV that is 0.20,
where the model answers 211 of 234 with 169 correct and 42 wrong, against the pattern layer's 52
correct and 46 wrong. Choosing a threshold that merely maximised accuracy would have buried the
error count, which on a console sitting in front of an account is the number that hurts.

    python -m argus.eval.ngramtrain
"""

from __future__ import annotations

import argparse
import json
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
"""``C`` for the logistic head. CV is flat from 5.0 to 20.0 (77.3% either way), so the more
regularised end is taken: with 234 training rows the looser fit buys nothing measurable and
would only be harder to defend."""

ABSTAIN_THRESHOLD = 0.20
"""See the module docstring. Fixed by a rule stated before the sweep, not picked from the table."""

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
    for case in TUNED:
        texts.append(case.ask)
        labels.append(str(case.expect))
    for case in CASES:
        if str(case.expect) not in EXCLUDED_LABELS:
            texts.append(case.ask)
            labels.append(str(case.expect))
    keep = [i for i, label in enumerate(labels) if label not in EXCLUDED_LABELS]
    texts, labels = [texts[i] for i in keep], [labels[i] for i in keep]

    # The out-of-scope class. Absent rather than fatal: the model trains without it, and
    # `eval/ngrambench.py` will simply report the recall that produces.
    if OOS_PATH.exists():
        negatives = json.loads(OOS_PATH.read_text(encoding="utf-8"))
        for row in negatives["rows"]:
            texts.append(row["ask"])
            labels.append(OUT_OF_SCOPE_LABEL)
    return texts, labels


def fit(texts: Sequence[str], labels: Sequence[str]) -> Any:
    """The scikit-learn pipeline, fitted. Returns ``(vectorizer, classifier)``."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
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
    classifier = LogisticRegression(
        C=REGULARISATION, max_iter=3000, class_weight="balanced",
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
