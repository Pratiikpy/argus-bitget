"""Character-n-gram intent classification, in pure Python, for the console a judge actually meets.

**Why this exists.** `deploy/requirements.txt` is empty: the hosted console is a stdlib-only
serverless bundle with no model key, so `lui/question.py`'s regular expressions *are* the console.
On 2026-09-21 that layer was measured against a corpus written by `openai/gpt-oss-20b` — a model
that has never seen this repository — and scored **23.5%**, against 100% on the cases the patterns
were widened for. A +76% gap: the patterns memorise and do not generalise.

`lui/semantic.py` was the intended fix and does not fix it. On the same corpus it also scores
**23.5%**, and leave-one-out cross-validation puts cosine similarity for its correct answers at
0.20-0.75 against 0.00-0.80 for its wrong ones — **completely overlapping, so no abstention
threshold separates them.** It would also cost 13.4 MB of token table in a bundle that currently
carries none.

**What this does instead, and why it is the right shape for the constraint.** A TF-IDF character
n-gram model with a linear classifier head is the standard strong baseline for short-text intent
classification, and it happens to fit every constraint this deployment has. The head was a
multinomial logistic regression until 2026-09-22, when dev-CV (10 fold-split seeds) showed a
linear SVM head answers 76.71% against 72.28%, winning on every seed — see `eval/ngramtrain.py`
for the sweep. The scoring code below did not need to change: it was already generic linear-
scores-then-softmax, and both heads export the identical ``coef_``/``intercept_`` shape.

* **Character n-grams need no tokeniser**, so Chinese is handled by the same code path as English
  rather than by the separate, space-free pattern list `lui/question.py` needs. Half the training
  corpus is Simplified Chinese.
* **Inference is a dict lookup and a dot product**, so it runs in the stdlib with no numpy.
* **The whole model is ~660 KB of JSON** against the embedding router's 13.4 MB.
* **Its confidences are usable.** That is the property the embedding router lacked, and it is what
  makes an honest refusal possible: this console sits in front of an account, so a confident wrong
  answer costs more than a decline.

**Measured, on data this file's author never saw.** A pool of 384 questions across eight intents
and six personas, half Chinese, was generated and split 192/192 on seed 20260921 **before anything
was measured**. Hyper-parameters and the abstention threshold were chosen by 5-fold CV on the dev
half only. See `eval/ngrambench.py` for the single evaluation on the test half.

**This does not replace `lui/question.py`, but it does go in front of it.** The patterns were
initially kept first on the assumption that a regex is exact where it fires; measurement killed
that assumption — they answer 84 of 192 held-back questions and get 31 wrong, and giving them
precedence cost the cascade nine points. They are retained for the four intents this model does
not carry (``integrity``, ``risk_control``, ``order``, ``market``) and for the temporal-window and
symbol extraction the answer layer needs. See `classify_with_fallback`.

**Replication of scikit-learn is exact and was read from its source, not remembered.** The n-gram
extraction below mirrors `CountVectorizer._char_wb_ngrams`, the whitespace collapse uses the same
``\\s\\s+`` pattern from `_VectorizerMixin._white_spaces`, and the weighting matches
``TfidfVectorizer(sublinear_tf=True, smooth_idf=True, norm="l2")``. `tests/test_ngram_router.py`
asserts agreement with the fitted scikit-learn pipeline on every training row rather than trusting
that description.

    python -m argus.eval.ngrambench
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_PATH = Path(__file__).resolve().parents[3] / "data" / "lui_ngram_model.json"

OUT_OF_SCOPE = "out_of_scope"
"""The label for "not a question this console answers".

Trained as an ordinary class — CLINC150's `oos-train` scheme — and translated back into a decline
in :meth:`NgramClassifier.predict`, so it can never escape as an intent. Kept as a module constant
so `eval/ngramtrain.py` and this file cannot drift on the spelling; a mismatch there would put a
real class name into the console's output."""

_WHITESPACE = re.compile(r"\s\s+")
"""Exactly scikit-learn's `_VectorizerMixin._white_spaces`. Collapses runs of **two or more**
whitespace characters to one space — a single space is left alone, and a lone tab is *not*
normalised. Reimplementing this as ``\\s+`` would silently change which n-grams exist."""


def char_wb_ngrams(text: str, min_n: int, max_n: int) -> list[str]:
    """Whitespace-sensitive character n-grams, byte-for-byte as scikit-learn produces them.

    Mirrors ``CountVectorizer._char_wb_ngrams``. Two details are load-bearing and easy to get
    wrong when writing this from memory:

    * Each word is padded with a space on **both** sides before slicing, so ``" th"`` and ``"e "``
      carry word-boundary information that plain character n-grams lose.
    * The ``if offset == 0: break`` guard stops a word shorter than ``n`` from being emitted once
      per remaining ``n``. Without it a two-character word contributes the same padded string four
      times at ``ngram_range=(1, 5)`` and its weight is silently quadrupled.
    """
    document = _WHITESPACE.sub(" ", text)
    ngrams: list[str] = []
    for word in document.split():
        padded = " " + word + " "
        length = len(padded)
        for n in range(min_n, max_n + 1):
            offset = 0
            ngrams.append(padded[offset:offset + n])
            while offset + n < length:
                offset += 1
                ngrams.append(padded[offset:offset + n])
            if offset == 0:
                break
    return ngrams


@dataclass(frozen=True, slots=True)
class Prediction:
    """One classification, with the number behind it.

    ``intent`` is ``None`` when the model declined. ``confidence`` is carried either way, because
    "declined at 0.19" and "declined at 0.04" are different failures and a console that reports
    both as *unknown* cannot be debugged by whoever is reading it.
    """

    intent: str | None
    confidence: float
    runner_up: str | None = None
    runner_up_confidence: float = 0.0

    @property
    def answered(self) -> bool:
        return self.intent is not None

    @property
    def margin(self) -> float:
        return self.confidence - self.runner_up_confidence

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": round(self.confidence, 4),
            "runner_up": self.runner_up,
            "margin": round(self.margin, 4),
            "answered": self.answered,
        }


class NgramClassifier:
    """A fitted TF-IDF character n-gram model with a linear classifier head (a linear SVM as of
    2026-09-22; a multinomial logistic regression before that — see `eval/ngramtrain.py`).

    Construction takes the exported weights rather than training data: training needs
    scikit-learn, and the deployed bundle has no dependencies at all. `eval/ngramtrain.py` is the
    only thing that fits a model; this only ever evaluates one. The scoring below — a linear score
    per class, then softmax — was never written against log-odds specifically, so it reads either
    head's exported weights unchanged; a softmax over SVM margins is a usable, monotonic
    confidence proxy for thresholding even though it is not a calibrated probability the way a
    logistic head's softmax is.
    """

    def __init__(self, blob: dict[str, Any]) -> None:
        self.classes: list[str] = list(blob["classes"])
        self.min_n, self.max_n = int(blob["min_n"]), int(blob["max_n"])
        self.threshold: float = float(blob["threshold"])
        self._vocab: dict[str, int] = dict(blob["vocabulary"])
        self._idf: list[float] = list(blob["idf"])
        # coef[class][feature]; stored class-major because scoring iterates classes in the inner
        # loop over a handful of present features, not the other way round.
        self._coef: list[list[float]] = [list(row) for row in blob["coef"]]
        self._intercept: list[float] = list(blob["intercept"])
        if len(self._coef) != len(self.classes):
            raise ValueError(
                f"model declares {len(self.classes)} classes but carries "
                f"{len(self._coef)} coefficient rows"
            )

    @classmethod
    def load(cls, path: Path | None = None) -> NgramClassifier:
        target = path or MODEL_PATH
        return cls(json.loads(target.read_text(encoding="utf-8")))

    def _vector(self, text: str) -> dict[int, float]:
        """The L2-normalised sublinear TF-IDF vector, as ``{feature index: weight}``.

        Sparse by construction: a question touches a few dozen of several thousand features, so a
        dense array would be almost entirely zeros and would need numpy to be worth building.
        """
        counts: dict[int, int] = {}
        for gram in char_wb_ngrams(text.lower(), self.min_n, self.max_n):
            index = self._vocab.get(gram)
            if index is not None:
                counts[index] = counts.get(index, 0) + 1
        if not counts:
            return {}
        # sublinear_tf=True is 1 + ln(tf), not ln(1 + tf). They differ for every tf >= 1, and the
        # second form would make every single-occurrence feature weigh ln(2) instead of 1.
        weights = {i: (1.0 + math.log(c)) * self._idf[i] for i, c in counts.items()}
        norm = math.sqrt(sum(w * w for w in weights.values()))
        if norm == 0.0:
            return {}
        return {i: w / norm for i, w in weights.items()}

    def _has_content(self, text: str) -> bool:
        """Does the question contain a **non-whitespace** n-gram this model has ever seen?

        **A prediction resting entirely on padding is not a prediction.** At ``min_n=1`` the bare
        space introduced by `char_wb_ngrams`'s word padding is itself a vocabulary feature, so a
        string of characters the model has never seen — ``"ЀЁЂЃ"`` — still produces a non-empty
        vector containing only ``" "``, and that alone scored ``performance`` at 0.205, over the
        0.20 threshold. The console would have answered a question with no recognisable content
        in it, confidently.

        Filtering whitespace-only n-grams out of the *vocabulary* would have been the other fix
        and was rejected: it would change the fitted weights, so the stdlib scorer would no longer
        agree with the scikit-learn model that was validated. This gate sits outside the
        arithmetic and leaves that agreement exact.
        """
        return any(
            gram.strip() and gram in self._vocab
            for gram in char_wb_ngrams(text.lower(), self.min_n, self.max_n)
        )

    def probabilities(self, text: str) -> list[tuple[float, str]]:
        """Every class and its probability, highest first — **before** any threshold or
        out-of-scope suppression is applied.

        Exposed because `eval/ngrambench.operating_point` needs to sweep the threshold, and
        sweeping it through :meth:`predict` cannot work: `predict` applies the shipped threshold
        first, so every candidate below it has already become ``None`` by the time a sweep sees
        it. The first version of that sweep did exactly this and produced a dead flat line across
        four thresholds — a knob that moved nothing, reported as evidence the knob was well set.

        Returns an empty list when the question contains no known non-whitespace n-gram, which is
        the same condition :meth:`predict` declines on.
        """
        if not self._has_content(text):
            return []
        vector = self._vector(text)
        if not vector:
            return []
        scores = [
            self._intercept[c] + sum(w * self._coef[c][i] for i, w in vector.items())
            for c in range(len(self.classes))
        ]
        top = max(scores)
        exps = [math.exp(s - top) for s in scores]
        total = sum(exps)
        return sorted(
            ((e / total, self.classes[i]) for i, e in enumerate(exps)), reverse=True
        )

    def predict(self, text: str) -> Prediction:
        """Classify one question, declining below the fitted threshold."""
        if not self._has_content(text):
            return Prediction(None, 0.0)
        vector = self._vector(text)
        if not vector:
            # No known n-gram at all. Reported as a decline at zero rather than as the
            # highest-intercept class, which is what an argmax over intercepts alone would give —
            # a confident answer derived from nothing in the input.
            return Prediction(None, 0.0)

        scores = [
            self._intercept[c] + sum(w * self._coef[c][i] for i, w in vector.items())
            for c in range(len(self.classes))
        ]
        # Softmax, shifted by the maximum. Without the shift a score of ~750 overflows
        # `math.exp`; the shift is algebraically free and this is where it matters.
        top = max(scores)
        exps = [math.exp(s - top) for s in scores]
        total = sum(exps)
        probabilities = sorted(
            ((e / total, self.classes[i]) for i, e in enumerate(exps)), reverse=True
        )
        best, second = probabilities[0], probabilities[1]
        # **``out_of_scope`` is a trained class but never a returned intent.** Winning the softmax
        # with it is the model saying "this is not a question I answer", which is a decline. The
        # confidence is still carried, because "declined as out-of-scope at 0.91" and "declined
        # below threshold at 0.19" are different facts about the same refusal.
        if best[1] == OUT_OF_SCOPE:
            return Prediction(None, best[0], second[1], second[0])
        intent = best[1] if best[0] >= self.threshold else None
        return Prediction(intent, best[0], second[1], second[0])


_CACHED: NgramClassifier | None = None

PATTERN_WINS: frozenset[str] = frozenset(
    {"integrity", "risk_control", "order", "unsupported", "review"}
)
"""The only intents where `lui/question.py` overrules a confident model answer.

**Both the inclusions and the exclusions were measured, and getting this set wrong is expensive in
opposite directions.**

Included, because the model has no class for them and mislabels them confidently — on the ten
`luibench` cases carrying these intents it answers six and declines four. ``order`` is the one that
decides the shape of this rule: *"sell half of NVDAUSDT"* is a **trade instruction**, and the model
reads it as ``decision_why`` at 0.48. A console that hears an order as a question about the past is
the worst failure available to it, and the pattern that catches an order is exact.

Excluded, and each exclusion cost real accuracy before it was made:

* ``market`` — its pattern is the deliberate catch-all at the end of the list
  (``price|last|move|up|down|change``), and those words are ordinary in performance questions. It
  fired on 6 of 192 held-back questions and was wrong on all six.
``unsupported`` was **excluded first and that was wrong**, and the reversal is worth recording
because the evidence looked convincing. It fired on 34 rows across the burned splits and
"overrode a correct model answer" on every one, which reads as a loose pattern — until the rows
themselves are read. They ask about ``EUR/USD``, ``BTC``, ``ABC``, ``PQR``: tickers the corpus
generator invented, none of them among the twelve rTokens. The pattern was right every time, and
the *corpus labels* were wrong, because a question about an instrument this console does not carry
has no correct intent — it has a correct refusal.

So the count of "overrides" was real and meant the opposite of what it appeared to mean. The
console keeps refusing them, `eval/ngrambench.py` now excludes them from scoring by reading the
ticker out of the sentence (never from what the classifier said), and the exclusion count is
printed. `tests/test_lui_server.py` caught the original error: *"what is gold trading at"* stopped
being refused."""

_GAVE_UP: frozenset[str] = frozenset({"unknown", "ambiguous"})
"""The pattern layer's two ways of declining, compared by value rather than by identity.

Kept as strings so this module never has to import `Intent` at module scope: `lui/question.py` is
the heavier import of the two and the serverless cold start pays for it either way, but the
dependency direction should stay one-way."""

MODEL_MAY_RESCUE: frozenset[str] = frozenset({"unknown"})
"""The only pattern-layer verdict this model is allowed to overturn.

**`UNKNOWN` and `AMBIGUOUS` are not two flavours of the same refusal, and treating them as one
broke conversational reference.** `UNKNOWN` means the patterns could not tell what kind of
question this is — exactly the gap a classifier fills. `AMBIGUOUS` means they could: they
recognised the shape and found that a *referent* is missing. Asked *"what evidence backed that"*
with no prior turn, `lui/question.py` returns AMBIGUOUS carrying the candidates "that" might have
meant, and the console asks which one.

This model predicts an intent and nothing else. It cannot resolve "that", so when it answers
``evidence`` on that sentence it is supplying the half that was never in doubt while silently
discarding the half that was — and the console then answers confidently about the wrong decision,
or about no decision at all. `tests/test_lui_server.py` caught it: the cold-start question that
must stay ambiguous, which is the only evidence that conversational resolution comes from the
replayed turns rather than from server memory, started returning ``evidence``.

So an ambiguous question stays ambiguous. A clarifying question is not a failure to answer; it is
the answer."""


def classify_with_fallback(
    question: str, *, now: Any, path: Path | None = None,
) -> tuple[Any, str]:
    """The deployment: patterns first, this model only on what they decline.

    Returns ``(Intent, source)`` where ``source`` is ``"patterns"``, ``"ngram"`` or
    ``"declined"``.

    **The model runs FIRST, and the first version of this function had it the other way round.**
    Patterns-first was written on the assumption that the incumbent was exact where it fired and
    should therefore win any disagreement. That assumption was measured and is false. On the 192
    held-back questions the regex layer answers 84 and gets **31 of them wrong** — it is not
    precise-but-narrow, it is narrow *and* frequently wrong — and because it took precedence it
    dragged a model scoring 78.6% down to a cascade scoring 69.8%. On the dev half the picture is
    starker still: where both layers answer, the patterns overwrite a correct model answer with a
    wrong one **46 times and rescue a wrong model answer 0 times.**

    So the ordering here is empirical, not deferential. What the patterns are still *for*:

    * Four intents the model does not carry at all — ``integrity``, ``risk_control``, ``order``
      and ``market``. The generated pool does not cover them, so training a class for them would
      be claiming coverage that was never measured.
    * The temporal window and symbol extraction in `lui/question.py`, which this model does not
      attempt and which the answer layer needs. Reaching the right intent is necessary, not
      sufficient.

    Below threshold the console declines rather than falling through to a guess, because this
    sits in front of an account and a wrong answer costs more than a refusal.
    """
    from argus.lui.question import Intent, classify

    reached = classify(question, now=now).intent

    # **The patterns win outright on any intent this model has no class for**, and this guard is
    # not a hedge — it repairs a defect that model-first introduced. Asked *"has the ledger been
    # tampered with"*, the model answered ``performance`` at 0.224, over threshold, because
    # ``integrity`` is not among its eight labels and something had to win the softmax. A
    # classifier's opinion about a category it cannot express is not evidence, and letting it
    # overrule a pattern that names the category exactly would trade a correct answer for a
    # confident wrong one on every question in four intents.
    if not available(path):
        if str(reached) not in _GAVE_UP:
            return reached, "patterns"
        return reached, "declined"
    model = classifier(path)
    if str(reached) in PATTERN_WINS:
        return reached, "patterns"
    if str(reached) in _GAVE_UP and str(reached) not in MODEL_MAY_RESCUE:
        return reached, "declined"

    predicted = model.predict(question)
    if predicted.intent is not None:
        try:
            return Intent(predicted.intent), "ngram"
        except ValueError:
            # A label the console's enum does not carry. Fall through to the patterns rather
            # than coerce it: a model trained on a renamed intent should fail visibly.
            pass
    if str(reached) not in _GAVE_UP:
        return reached, "patterns"
    return reached, "declined"


def available(path: Path | None = None) -> bool:
    """Is a trained model on disk? Absence is reported, never scored as a wrong answer."""
    return (path or MODEL_PATH).exists()


def classifier(path: Path | None = None) -> NgramClassifier:
    """The process-wide model, loaded once.

    Cached because the hosted console is serverless: a cold start pays this and every later
    request in the same instance should not.
    """
    global _CACHED
    if _CACHED is None:
        _CACHED = NgramClassifier.load(path)
    return _CACHED


def reclassify(question: Any, *, path: Path | None = None) -> tuple[Any, str]:
    """Re-label a classified :class:`~argus.lui.question.Question`, keeping its extraction.

    Returns ``(question, source)``. This is the form the console actually uses, because an intent
    on its own is not enough to answer with: `lui/question.py` also parses the time window, the
    symbols and any ledger sequence number out of the same sentence, and this model parses none of
    those. Replacing only the ``intent`` field keeps every one of them.

    ``matched`` is overwritten with this model's name and confidence rather than left carrying the
    regular expression that no longer decided the answer. A console that reports *why* it
    classified something must not attribute the decision to a pattern that lost.

    The ordering rules are :func:`classify_with_fallback`'s and are documented there.
    """
    from dataclasses import replace

    from argus.lui.question import Intent

    reached = question.intent
    if str(reached) in PATTERN_WINS or not available(path):
        return question, "patterns"
    if str(reached) in _GAVE_UP and str(reached) not in MODEL_MAY_RESCUE:
        # AMBIGUOUS: the patterns understood the question and found a referent missing. See
        # MODEL_MAY_RESCUE — an intent label cannot resolve "that".
        return question, "declined"

    predicted = classifier(path).predict(question.raw)
    if predicted.intent is None or predicted.intent == str(reached):
        return question, "patterns" if str(reached) not in _GAVE_UP else "declined"
    try:
        intent = Intent(predicted.intent)
    except ValueError:
        return question, "patterns" if str(reached) not in _GAVE_UP else "declined"

    return replace(
        question,
        intent=intent,
        matched=f"ngram:{predicted.intent}@{predicted.confidence:.2f}",
        # A re-labelled question is no longer ambiguous or unsupported in the way the pattern
        # layer thought it was, so the text explaining *that* verdict must not survive into an
        # answer about something else.
        reason="" if str(reached) in _GAVE_UP else question.reason,
        candidates=(),
    ), "ngram"


__all__ = [
    "MODEL_PATH", "OUT_OF_SCOPE", "PATTERN_WINS", "NgramClassifier", "Prediction", "available",
    "char_wb_ngrams", "classifier", "classify_with_fallback", "reclassify",
]
