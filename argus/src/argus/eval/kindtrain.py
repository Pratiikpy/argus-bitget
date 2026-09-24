"""Fit the console's research-kind model: which engine a free-form question needs.

**Why this exists.** The hosted console reads questions with hand-written patterns when no language
model is configured. On 2026-09-25 a question set written blind by a second agent — 200 rows in
English, Chinese, Hinglish, Spanish, Korean and more, never inspected — measured the patterns at
**55.0%** before that day's fixes and **63.5%** after them, while the corpus the fixes were made
against read 85%. Patching patterns fits the questions you have read; it does not generalise. The
fix that generalises is a model trained on many writers' phrasings, which is what this is.

**The model.** The same character n-gram TF-IDF + linear SVM as `eval/ngramtrain.py` (its
docstring carries the CV evidence for the head and the n-gram range), so the stdlib scorer in
`lui/ngram.py` runs it unchanged and the deployed bundle still has no dependencies. Fourteen
classes: the thirteen research kinds and record, plus ``refuse`` — CLINC150's out-of-scope-as-a-
class scheme, for the same reason the intent model uses it.

**The data, and what is never read.** Training rows are ``data/lui_kind_train_{A..E}.jsonl``
(five writers who were forbidden to read this repository: English/Chinese; ten other languages;
confusable boundaries; record/refuse-heavy; twelve more languages) plus the 2026-09-25 dev corpus,
which already informed the patterns and so cannot be a test.

**Held-out sets, stated honestly.** ``data/lui_heldout_corpus_2026-09-25.jsonl`` was never trained
on; it was scored once (model alone: 88.5%) and has since been used to fix the console's plumbing
around the model, so it is now a development set. The final figure is
``data/lui_final_heldout_2026-09-25.jsonl``, written by a third blind writer and scored once.

**Choosing the knobs.** ``C`` and the n-gram range by stratified 5-fold CV on the training rows.
The abstention threshold by a rule fixed before the sweep: maximise ``correct - 3 x wrong`` over
the answered out-of-fold predictions, because a confident wrong answer on a console in front of an
account costs more than a question passed back to the patterns.

    python -m argus.eval.kindtrain          # CV, fit, export data/lui_kind_model.json
    python -m argus.eval.kindtrain --heldout  # score the export once on the held-out set
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parents[3] / "data"
TRAIN_FILES = (*(DATA / f"lui_kind_train_{k}.jsonl" for k in "ABCDE"),
               DATA / "lui_blind_corpus_2026-09-25.jsonl",
               # Scored once as held-out (88.5%), then used to fix the plumbing: now training data.
               DATA / "lui_heldout_corpus_2026-09-25.jsonl")
FINAL_HELDOUT = DATA / "lui_final_heldout_2026-09-25.jsonl"
HELDOUT = DATA / "lui_heldout_corpus_2026-09-25.jsonl"
MODEL_PATH = DATA / "lui_kind_model.json"
REPORT_PATH = DATA / "lui_kind_model_report.json"

LABELS = ("impact", "stress", "compare", "execution", "quote", "technicals", "fundamentals",
          "event", "hedge", "macro", "sentiment", "news", "record", "refuse")
SEED = 20260925
C_GRID = (1.0, 2.0, 5.0, 10.0)
NGRAM_GRID = ((1, 4), (1, 5))
ERROR_COST = 3.0


class KindTrainError(RuntimeError):
    """Training cannot proceed honestly."""


def _read(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def training_rows() -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for path in TRAIN_FILES:
        if not path.exists():
            raise KindTrainError(f"{path.name} is missing")
        for row in _read(path):
            label = str(row.get("label") or row.get("expected") or "").strip().lower()
            text = str(row.get("text") or "").strip()
            if label in LABELS and text:
                texts.append(text)
                labels.append(label)
    return texts, labels


def _fit(texts: Sequence[str], labels: Sequence[str], c: float,
         ngrams: tuple[int, int]) -> tuple[Any, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngrams, sublinear_tf=True,
                                 min_df=1, lowercase=True)
    matrix = vectorizer.fit_transform(texts)
    classifier = LinearSVC(C=c, class_weight="balanced", max_iter=10000,
                           random_state=SEED).fit(matrix, list(labels))
    return vectorizer, classifier


def _softmax_top(scores: Sequence[float]) -> tuple[int, float]:
    import math

    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    best = max(range(len(exps)), key=exps.__getitem__)
    return best, exps[best] / sum(exps)


def _folds(labels: Sequence[str], k: int = 5) -> list[list[int]]:
    """Stratified folds, seeded, so every label is spread across all five."""
    rng = random.Random(SEED)
    by_label: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        by_label.setdefault(label, []).append(i)
    folds: list[list[int]] = [[] for _ in range(k)]
    for label in sorted(by_label):
        idx = by_label[label]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(i)
    return folds


def cross_validate(texts: Sequence[str], labels: Sequence[str], c: float,
                   ngrams: tuple[int, int]) -> list[tuple[str, str, float]]:
    """Out-of-fold (truth, predicted, confidence) for every training row."""
    out: list[tuple[str, str, float]] = []
    for fold in _folds(labels):
        held = set(fold)
        train = [i for i in range(len(texts)) if i not in held]
        vec, clf = _fit([texts[i] for i in train], [labels[i] for i in train], c, ngrams)
        scores = clf.decision_function(vec.transform([texts[i] for i in fold]))
        for i, row in zip(fold, scores, strict=True):
            best, confidence = _softmax_top(list(row))
            out.append((labels[i], str(clf.classes_[best]), confidence))
    return out


def choose_threshold(oof: Sequence[tuple[str, str, float]]) -> tuple[float, dict[str, float]]:
    """The threshold maximising ``correct - ERROR_COST x wrong`` over answered predictions."""
    best: tuple[float, float, dict[str, float]] | None = None
    for step in range(0, 51):
        t = step / 100
        answered = [(truth, got) for truth, got, conf in oof if conf >= t]
        correct = sum(truth == got for truth, got in answered)
        wrong = len(answered) - correct
        utility = correct - ERROR_COST * wrong
        stats = {"threshold": t, "answered": len(answered), "correct": correct, "wrong": wrong,
                 "coverage": len(answered) / len(oof), "accuracy_answered":
                 correct / len(answered) if answered else 0.0}
        if best is None or utility > best[0]:
            best = (utility, t, stats)
    assert best is not None
    return best[1], best[2]


def export(vectorizer: Any, classifier: Any, *, threshold: float, rows: int,
           ngrams: tuple[int, int], c: float) -> dict[str, Any]:
    return {
        "generated_by": "argus.eval.kindtrain",
        "training_rows": rows,
        "classes": [str(x) for x in classifier.classes_],
        "min_n": ngrams[0],
        "max_n": ngrams[1],
        "threshold": threshold,
        "regularisation": c,
        "vocabulary": {term: int(i) for term, i in vectorizer.vocabulary_.items()},
        "idf": [round(float(v), 6) for v in vectorizer.idf_],
        "coef": [[round(float(v), 6) for v in row] for row in classifier.coef_],
        "intercept": [round(float(v), 6) for v in classifier.intercept_],
        "provenance": ("Trained on data/lui_kind_train_{A,B,C,D}.jsonl and the 2026-09-25 dev "
                       "corpus. data/lui_heldout_corpus_2026-09-25.jsonl was not read in "
                       "training, CV or threshold selection."),
    }


def train() -> dict[str, Any]:
    texts, labels = training_rows()
    grid: list[dict[str, Any]] = []
    best: tuple[float, float, tuple[int, int], list[tuple[str, str, float]]] | None = None
    for ngrams in NGRAM_GRID:
        for c in C_GRID:
            oof = cross_validate(texts, labels, c, ngrams)
            accuracy = sum(t == g for t, g, _ in oof) / len(oof)
            grid.append({"ngrams": list(ngrams), "C": c, "cv_accuracy": round(accuracy, 4)})
            if best is None or accuracy > best[0]:
                best = (accuracy, c, ngrams, oof)
    assert best is not None
    accuracy, c, ngrams, oof = best
    threshold, at = choose_threshold(oof)
    vectorizer, classifier = _fit(texts, labels, c, ngrams)
    MODEL_PATH.write_text(json.dumps(export(vectorizer, classifier, threshold=threshold,
                                            rows=len(texts), ngrams=ngrams, c=c)),
                          encoding="utf-8")
    per_label = Counter(labels)
    report = {
        "training_rows": len(texts), "per_label": dict(sorted(per_label.items())),
        "grid": grid, "chosen": {"ngrams": list(ngrams), "C": c, "cv_accuracy": accuracy},
        "threshold": at, "error_cost": ERROR_COST,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def heldout(path: Path = FINAL_HELDOUT) -> dict[str, Any]:
    """Score the exported model once on the held-out set, through the stdlib scorer."""
    from argus.lui.kindmodel import KindModel

    model = KindModel.load(MODEL_PATH)
    rows = _read(path)
    correct = answered = 0
    by_lang: dict[str, list[int]] = {}
    for row in rows:
        label, _confidence = model.predict(str(row["text"]))
        hit = label == row["expected"]
        answered += label is not None
        correct += hit
        tally = by_lang.setdefault(str(row.get("lang", "?")), [0, 0])
        tally[0] += hit
        tally[1] += 1
    return {"rows": len(rows), "correct": correct, "answered": answered,
            "accuracy": correct / len(rows),
            "by_lang": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_lang.items())}}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="fit the console's research-kind model")
    parser.add_argument("--heldout", action="store_true", help="score the export once")
    args = parser.parse_args(argv)
    if args.heldout:
        print(json.dumps(heldout(), indent=2))
        return 0
    print(json.dumps(train(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["LABELS", "KindTrainError", "choose_threshold", "cross_validate", "export", "heldout",
           "main", "train", "training_rows"]
