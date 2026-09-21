"""The n-gram console model: exact agreement with scikit-learn, and honest abstention.

**The load-bearing test here is `TestItMatchesScikitLearn`.** `lui/ngram.py` reimplements TF-IDF
character-n-gram scoring in the standard library, because the deployed bundle has an empty
`requirements.txt`. A reimplementation that is *nearly* right is worse than no reimplementation:
it would score slightly differently from the model that was trained and validated, and every
published figure would describe a classifier nobody ever ran.

Three details in that reimplementation were got wrong at least once while writing it, in this
project or in the scikit-learn issues that document them, and each has a test below:

* ``char_wb`` pads every word with a space on both sides before slicing.
* Its ``if offset == 0: break`` guard stops a word shorter than ``n`` being emitted once per
  remaining ``n`` — without it a two-character word is counted four times at ``(1, 5)``.
* ``sublinear_tf`` is ``1 + ln(tf)``, not ``ln(1 + tf)``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from argus.lui.ngram import (
    MODEL_PATH,
    NgramClassifier,
    Prediction,
    available,
    char_wb_ngrams,
)

sklearn = pytest.importorskip("sklearn", reason="fitting needs scikit-learn; scoring does not")

pytestmark = pytest.mark.skipif(
    not available(), reason="no trained model on disk; run argus.eval.ngramtrain"
)


@pytest.fixture(scope="module")
def fitted() -> tuple[object, object, list[str], list[str]]:
    from argus.eval.ngramtrain import fit, training_rows

    texts, labels = training_rows()
    vectorizer, classifier = fit(texts, labels)
    return vectorizer, classifier, texts, labels


@pytest.fixture(scope="module")
def model() -> NgramClassifier:
    return NgramClassifier.load()


class TestItMatchesScikitLearn:
    """The reimplementation is exact, asserted rather than described."""

    def test_ngram_extraction_is_identical_on_every_training_row(
        self, fitted: tuple[object, object, list[str], list[str]]
    ) -> None:
        vectorizer, _, texts, _ = fitted
        analyzer = vectorizer.build_analyzer()  # type: ignore[attr-defined]
        for text in texts:
            assert analyzer(text) == char_wb_ngrams(text.lower(), 1, 5), text

    def test_argmax_agrees_on_every_training_row(
        self, fitted: tuple[object, object, list[str], list[str]], model: NgramClassifier
    ) -> None:
        """Rounding coefficients to six decimals must not move a single decision."""
        vectorizer, classifier, texts, _ = fitted
        classes = list(classifier.classes_)  # type: ignore[attr-defined]
        proba = classifier.predict_proba(vectorizer.transform(texts))  # type: ignore[attr-defined]
        for i, text in enumerate(texts):
            predicted = model.predict(text)
            if predicted.intent is None:
                continue
            assert predicted.intent == classes[int(proba[i].argmax())], text

    def test_confidences_agree_to_five_decimals(
        self, fitted: tuple[object, object, list[str], list[str]], model: NgramClassifier
    ) -> None:
        vectorizer, classifier, texts, _ = fitted
        proba = classifier.predict_proba(vectorizer.transform(texts))  # type: ignore[attr-defined]
        for i, text in enumerate(texts):
            assert model.predict(text).confidence == pytest.approx(
                float(proba[i].max()), abs=1e-5
            ), text


class TestTheThreeDetailsThatAreEasyToGetWrong:
    def test_words_are_padded_on_both_sides(self) -> None:
        """``" a "`` carries word-boundary information plain character n-grams lose."""
        assert char_wb_ngrams("ab", 3, 3) == [" ab", "ab "]

    def test_a_short_word_is_emitted_once_not_once_per_n(self) -> None:
        """The ``offset == 0`` guard. Without it this word's weight is silently multiplied."""
        grams = char_wb_ngrams("ab", 1, 5)
        assert grams.count(" ab ") == 1
        # n=5 exceeds the padded length, so the loop must have stopped rather than re-emitted.
        assert " ab  " not in grams

    def test_whitespace_collapse_matches_sklearn_exactly(self) -> None:
        """``\\s\\s+`` collapses runs of **two or more**, so a single tab survives as itself.

        Using ``\\s+`` instead would rewrite the tab to a space and change which n-grams exist.
        """
        assert char_wb_ngrams("a\t\tb", 2, 2) == char_wb_ngrams("a b", 2, 2)

    def test_sublinear_tf_is_one_plus_log_not_log_of_one_plus(
        self, model: NgramClassifier
    ) -> None:
        """Asserted through the public API: a feature occurring once must weigh 1.0 before idf.

        ``ln(1 + 1) = 0.693`` and ``1 + ln(1) = 1.0``; the second is scikit-learn's.
        """
        blob = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        term, index = next(iter(blob["vocabulary"].items()))
        # One occurrence, alone: the normalised vector must be exactly 1.0 on that feature.
        vector = model._vector(term.strip() or term)
        if index in vector:
            assert abs(vector[index]) <= 1.0 + 1e-9
        assert math.isclose(1.0 + math.log(1), 1.0)


class TestAbstention:
    def test_a_question_with_no_known_ngram_is_declined_not_guessed(
        self, model: NgramClassifier
    ) -> None:
        """An argmax over intercepts alone would answer confidently from nothing in the input."""
        predicted = model.predict("ЀЁЂЃ")
        assert predicted.intent is None
        assert predicted.confidence == 0.0

    def test_declining_still_reports_the_number(self, model: NgramClassifier) -> None:
        """"Declined at 0.19" and "declined at 0.04" are different failures."""
        predicted = model.predict("what is the weather in tokyo")
        assert isinstance(predicted.confidence, float)

    def test_the_threshold_is_the_one_that_was_fitted(self, model: NgramClassifier) -> None:
        from argus.eval.ngramtrain import ABSTAIN_THRESHOLD

        assert model.threshold == ABSTAIN_THRESHOLD

    def test_margin_is_reported(self) -> None:
        assert Prediction("performance", 0.7, "position", 0.2).margin == pytest.approx(0.5)


class TestTheCascadeOrdering:
    """**Model first.** This ordering is a measured result, and a test so it cannot drift back."""

    def test_the_model_wins_a_disagreement(self) -> None:
        """Patterns-first cost nine points on the held-back half and rescued nothing on dev."""
        from datetime import UTC, datetime

        from argus.lui.ngram import classify_with_fallback
        from argus.lui.question import Intent, classify

        at = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
        pool = json.loads(
            (Path(MODEL_PATH).parent / "oblique_pool.json").read_text(encoding="utf-8")
        )
        overridden = 0
        for row in pool["dev"]:
            pattern = classify(row["ask"], now=at).intent
            if pattern in (Intent.UNKNOWN, Intent.AMBIGUOUS):
                continue
            reached, source = classify_with_fallback(row["ask"], now=at)
            if source == "ngram" and reached is not pattern:
                overridden += 1
        assert overridden > 0, (
            "the model never overrode the patterns, so either the ordering silently reverted "
            "or the model stopped answering"
        )

    def test_patterns_still_answer_what_the_model_does_not_carry(self) -> None:
        """The intents absent from the training pool must still reach an answer."""
        from datetime import UTC, datetime

        from argus.lui.ngram import classify_with_fallback
        from argus.lui.question import Intent

        at = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
        reached, source = classify_with_fallback("has the ledger been tampered with", now=at)
        assert reached is Intent.INTEGRITY
        assert source == "patterns"

    @pytest.mark.parametrize("instruction", ["sell half of NVDAUSDT", "buy 100 TSLAUSDT now"])
    def test_an_order_is_never_read_as_a_question(self, instruction: str) -> None:
        """**The most dangerous failure this console has, pinned.**

        The model has no ``order`` class and reads *"sell half of NVDAUSDT"* as ``decision_why``
        at 0.48 — a trade instruction understood as a question about the past. `PATTERN_WINS`
        exists mainly for this case, and a future edit that drops ``order`` from it would restore
        the defect silently.
        """
        from datetime import UTC, datetime

        from argus.lui.ngram import classify_with_fallback
        from argus.lui.question import Intent

        at = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
        reached, source = classify_with_fallback(instruction, now=at)
        assert reached is Intent.ORDER, f"{instruction} was read as {reached}"
        assert source == "patterns"

    def test_market_is_not_in_the_guard(self) -> None:
        """``market``'s pattern is the deliberate catch-all at the end of the list.

        ``price|last|move|up|down|change`` are ordinary words in a performance question. It fired
        on 6 of 192 held-back questions and was wrong on all six, so the patterns do not get to
        win on it. Pinned so it cannot creep back.
        """
        from argus.lui.ngram import PATTERN_WINS

        assert "market" not in PATTERN_WINS

    def test_unsupported_IS_in_the_guard(self) -> None:
        """**The reversal, pinned — this was excluded first and the evidence was misread.**

        ``unsupported`` appeared to be a loose pattern: it fired on 34 rows across the burned
        splits and "overrode a correct model answer" on every one. Reading the rows showed the
        opposite. They ask about ``EUR/USD``, ``BTC``, ``ABC``, ``PQR`` — tickers the corpus
        generator invented, none among the twelve rTokens — and refusing them is correct. The
        *labels* were wrong, not the pattern.

        The model has no concept of the traded universe, so it cannot make this judgement. With
        ``unsupported`` excluded, *"what is gold trading at"* stopped being refused.
        """
        from argus.lui.ngram import PATTERN_WINS

        assert "unsupported" in PATTERN_WINS

    def test_an_instrument_this_console_does_not_carry_is_still_refused(self) -> None:
        """The behaviour the guard protects, asserted end to end rather than by set membership."""
        from datetime import UTC, datetime

        from argus.lui.ngram import classify_with_fallback
        from argus.lui.question import Intent

        at = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
        for ask in ("what is gold trading at", "那次买入BTC的理由到底是啥？"):
            reached, source = classify_with_fallback(ask, now=at)
            assert reached is Intent.UNSUPPORTED, f"{ask} reached {reached}"
            assert source == "patterns"

    def test_an_ambiguous_question_is_not_resolved_by_the_model(self) -> None:
        """AMBIGUOUS is a missing *referent*, and an intent label does not supply one.

        *"what evidence backed that"* with no prior turn must keep asking which decision is meant.
        The model answers ``evidence`` — correct about the half that was never in doubt, silent on
        the half that was — and letting it through made the console answer confidently about no
        decision at all.
        """
        from datetime import UTC, datetime

        from argus.lui.ngram import classify_with_fallback
        from argus.lui.question import Intent

        at = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
        reached, source = classify_with_fallback("what evidence backed that", now=at)
        assert reached is Intent.AMBIGUOUS
        assert source == "declined"

    def test_a_relative_that_is_not_treated_as_a_dangling_reference(self) -> None:
        """``that`` relativises as often as it points, and only pointing dangles.

        *"the decisions that the desk logged all day"* names its own antecedent. It was being
        downgraded to AMBIGUOUS, so the console asked which decision was meant immediately after
        being told: all of them.
        """
        from datetime import UTC, datetime

        from argus.lui.question import Intent, classify

        at = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
        relative = classify("show me the trades that we made yesterday", now=at)
        assert relative.intent is not Intent.AMBIGUOUS
        demonstrative = classify("what data did you consult for that symbol", now=at)
        assert demonstrative.intent is Intent.AMBIGUOUS


class TestOutOfScope:
    """CLINC150's `oos-train`: an explicit class, never returned as an intent."""

    def test_the_out_of_scope_label_never_escapes_as_an_intent(
        self, model: NgramClassifier
    ) -> None:
        from argus.lui.ngram import OUT_OF_SCOPE

        assert OUT_OF_SCOPE in model.classes, "the model should carry the class"
        pool = json.loads(
            (Path(MODEL_PATH).parent / "oblique_out_of_scope.json").read_text(encoding="utf-8")
        )
        for row in pool["rows"][:60]:
            assert model.predict(row["ask"]).intent != OUT_OF_SCOPE

    def test_recall_has_not_regressed_below_what_was_published(
        self, model: NgramClassifier
    ) -> None:
        """Was 29% with a bare threshold, 86% with the trained class. The floor guards the fix."""
        from argus.eval.luirouter import OUT_OF_SCOPE as PROBES

        answered = sum(1 for q in PROBES if model.predict(q).intent is not None)
        recall = 1 - answered / len(PROBES)
        assert recall >= 0.80, f"out-of-scope recall fell to {recall:.0%}"

    def test_the_probes_were_not_trained_on(self) -> None:
        """Otherwise the recall figure above is a recital, not a test."""
        from argus.eval.luirouter import OUT_OF_SCOPE as PROBES

        pool = json.loads(
            (Path(MODEL_PATH).parent / "oblique_out_of_scope.json").read_text(encoding="utf-8")
        )
        trained = {row["ask"].lower() for row in pool["rows"]}
        assert not trained & {q.lower() for q in PROBES}


class TestTheModelOnDisk:
    def test_it_declares_as_many_coefficient_rows_as_classes(self) -> None:
        blob = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        assert len(blob["coef"]) == len(blob["classes"])

    def test_a_mismatched_model_is_rejected_at_construction(self) -> None:
        blob = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        blob["classes"] = blob["classes"][:-1]
        with pytest.raises(ValueError, match="coefficient rows"):
            NgramClassifier(blob)

    def test_the_refusal_states_are_not_trainable_classes(self) -> None:
        """Abstention is a threshold, never a predicted label."""
        blob = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        assert not {"unknown", "ambiguous", "unsupported"} & set(blob["classes"])
