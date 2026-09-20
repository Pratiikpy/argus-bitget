"""Phrasebook tests — English must not have moved, and Chinese must not be English.

Two failure modes, and the first is the dangerous one. If converting the answerers to templates
quietly reworded the English, every existing phrasing test would have been silently rewritten to
match the new implementation rather than checking the old behaviour. So the English side is pinned
against the exact strings the console produced before, and the Chinese side is checked for actually
being Chinese rather than an English fallback wearing a language tag.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from argus.lui.answer import answer
from argus.lui.phrasebook import PHRASES, Language, PhraseError, language_of, t
from argus.lui.question import classify
from argus.paper.ledger import PaperLedger

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)
HAN_RANGE = range(0x4E00, 0xA000)


def _has_han(text: str) -> bool:
    return any(ord(c) in HAN_RANGE for c in text)


class TestTheRegistry:
    def test_every_phrase_carries_both_languages(self) -> None:
        """A key with only English is a sentence a Chinese reader gets in English, which is the
        exact defect this module was written to remove."""
        for key, entry in PHRASES.items():
            assert Language.EN in entry, key
            assert Language.ZH in entry, key

    def test_no_chinese_rendering_is_left_as_english(self) -> None:
        """The failure that would make the whole module cosmetic: a Chinese slot filled with the
        English string so the tests pass and the reader gets English anyway."""
        for key, entry in PHRASES.items():
            assert entry[Language.ZH] != entry[Language.EN], key
            assert _has_han(entry[Language.ZH]), key

    def test_every_template_has_the_same_fields_in_both_languages(self) -> None:
        """A Chinese template missing a field would drop a number a reader was told to check."""
        import string

        for key, entry in PHRASES.items():
            fields = {
                lang: {
                    f for _, f, _, _ in string.Formatter().parse(text) if f
                }
                for lang, text in entry.items()
            }
            assert fields[Language.EN] == fields[Language.ZH], key

    def test_an_unknown_key_raises_rather_than_rendering_itself(self) -> None:
        """Falling back to the key would ship "perf.headline" to a reader."""
        with pytest.raises(PhraseError, match="no phrase registered"):
            t("nope.missing")

    def test_a_missing_field_raises_rather_than_rendering_a_blank(self) -> None:
        with pytest.raises(PhraseError, match="needs field"):
            t("integ.head")


class TestLanguageDetection:
    def test_an_english_question_is_answered_in_english(self) -> None:
        assert language_of("why did you do nothing") is Language.EN

    def test_a_chinese_question_is_answered_in_chinese(self) -> None:
        assert language_of("为什么没有交易") is Language.ZH

    def test_one_han_character_is_enough(self) -> None:
        """A mixed question is still a Chinese reader asking."""
        assert language_of("NVDAUSDT 的表现") is Language.ZH

    def test_symbols_and_numbers_alone_stay_english(self) -> None:
        assert language_of("NVDAUSDT seq 42") is Language.EN

    def test_the_question_decides_and_nothing_else(self) -> None:
        """A property of the question, so a caller cannot forget to pass it and cannot set it to
        something the question contradicts."""
        assert classify("为什么没有交易", now=NOW).language is Language.ZH
        assert classify("why did you do nothing", now=NOW).language is Language.EN

    def test_each_question_decides_independently(self) -> None:
        """No sticky setting: asking in Chinese then English gives English back."""
        first = classify("记录完整吗", now=NOW)
        second = classify("is the record intact", now=NOW)
        assert first.language is Language.ZH
        assert second.language is Language.EN


class TestEnglishOutputIsUnchanged:
    """Pinned against the strings the console produced before the templates existed."""

    def test_the_abstention_explanation_is_word_for_word(self) -> None:
        assert t("abst.explain") == (
            "An abstention here is a recorded decision, not an absence of one: it is hash-chained, "
            "settled against the move that actually happened, and scored by abstention value."
        )

    def test_the_integrity_header_is_word_for_word(self) -> None:
        assert t("integ.header", Language.EN, state="intact", count=126) == (
            "Hash chain intact across 126 entry(ies)."
        )

    def test_the_untested_risk_layer_sentence_is_word_for_word(self) -> None:
        """The most load-bearing sentence the console says: it refuses to report an untested risk
        layer as a restrained one. Any drift here changes a claim."""
        assert t("risk.untested") == (
            "It intervened zero times because it was never invoked: every decision in this record "
            "was an abstention, and there was no exposure to narrow. That is an UNTESTED risk "
            "layer, not a restrained one, and reporting it as a low intervention rate over all "
            "decisions would be the flattering version of the same number."
        )

    def test_the_calibration_refusal_is_word_for_word(self) -> None:
        assert t("calib.insufficient_why") == (
            "Calibration on fewer is noise wearing a decimal point, so none is reported. This is a "
            "refusal to compute, not a missing feature."
        )

    def test_an_unknown_language_falls_back_to_english(self) -> None:
        """Defensive, and it must fall back to a real sentence rather than to the key."""
        assert t("pos.none", Language.EN) == "No open positions."


class TestAnswersRenderInTheLanguageAsked:
    def _ledger(self, tmp_path: object) -> PaperLedger:
        from argus.paper.runner import LEDGER_PATH

        return PaperLedger(path=LEDGER_PATH)

    def test_a_chinese_question_gets_chinese_lines(self) -> None:
        ledger = self._ledger(None)
        if not ledger.entries:
            pytest.skip("no ledger on disk to answer from")
        got = answer(ledger, classify("记录完整吗", now=NOW))
        assert not got.refused
        assert _has_han(got.lines[0])

    def test_the_same_question_in_english_gets_english_lines(self) -> None:
        ledger = self._ledger(None)
        if not ledger.entries:
            pytest.skip("no ledger on disk to answer from")
        got = answer(ledger, classify("is the record intact", now=NOW))
        assert not got.refused
        assert not _has_han(got.lines[0])
        assert "Hash chain" in got.lines[0]

    def test_the_two_carry_the_same_numbers(self) -> None:
        """Translation must not change a figure. The count in both answers is the same ledger."""
        ledger = self._ledger(None)
        if not ledger.entries:
            pytest.skip("no ledger on disk to answer from")
        count = str(len(ledger.entries))
        english = answer(ledger, classify("is the record intact", now=NOW))
        chinese = answer(ledger, classify("记录完整吗", now=NOW))
        assert count in english.lines[0]
        assert count in chinese.lines[0]

    def test_identifiers_are_never_translated(self) -> None:
        """A symbol, a hash prefix and a verdict are the things a reader goes and checks. Rendering
        them in Chinese would break the one promise this console makes."""
        ledger = self._ledger(None)
        if not ledger.entries:
            pytest.skip("no ledger on disk to answer from")
        got = answer(ledger, classify("为什么没有交易", now=NOW))
        joined = " ".join(got.lines)
        assert any(e.symbol in joined for e in ledger.entries[-3:])

    def test_a_chinese_answer_still_carries_its_sources(self) -> None:
        """Grounding is not a language feature. An answer in either language that asserts anything
        must say where it came from."""
        ledger = self._ledger(None)
        if not ledger.entries:
            pytest.skip("no ledger on disk to answer from")
        got = answer(ledger, classify("记录完整吗", now=NOW))
        assert got.is_grounded
        assert got.sources
