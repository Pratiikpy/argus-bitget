"""Tests for `eval/ngramtrain.py`'s out-of-scope language-balancing step.

Added 2026-09-22 alongside `balance_oos_by_language` itself, the RIVAL LENS fix for the
Chinese-negative-dominance hypothesis `eval/standing.py` had named but never tested: dev-CV
evidence (documented on `OOS_BALANCE_SEED`) showed downsampling the 60 Chinese out-of-scope
negatives to match English's 40 cuts wrong-OOS errors on genuine Chinese in-scope questions from
14.2% to 8.4%, McNemar p<0.0001, with no cost to overall accuracy.
"""

from __future__ import annotations

from argus.eval.ngramtrain import OOS_BALANCE_SEED, balance_oos_by_language


def _row(ask: str, lang: str) -> dict:
    return {"ask": ask, "lang": lang}


class TestBalanceOosByLanguage:
    def test_downsamples_the_majority_language_to_match_the_minority(self) -> None:
        rows = [_row(f"en{i}", "en") for i in range(40)] + [_row(f"zh{i}", "zh") for i in range(60)]
        balanced = balance_oos_by_language(rows)
        by_lang: dict[str, int] = {}
        for row in balanced:
            by_lang[row["lang"]] = by_lang.get(row["lang"], 0) + 1
        assert by_lang == {"en": 40, "zh": 40}

    def test_is_deterministic_across_repeated_calls(self) -> None:
        rows = [_row(f"en{i}", "en") for i in range(40)] + [_row(f"zh{i}", "zh") for i in range(60)]
        first = balance_oos_by_language(rows)
        second = balance_oos_by_language(rows)
        assert [row["ask"] for row in first] == [row["ask"] for row in second]

    def test_every_kept_row_was_actually_in_the_input(self) -> None:
        rows = [_row(f"en{i}", "en") for i in range(40)] + [_row(f"zh{i}", "zh") for i in range(60)]
        balanced = balance_oos_by_language(rows)
        input_asks = {row["ask"] for row in rows}
        assert all(row["ask"] in input_asks for row in balanced)

    def test_a_language_already_at_or_below_the_minimum_is_kept_whole(self) -> None:
        rows = [_row(f"en{i}", "en") for i in range(5)] + [_row(f"zh{i}", "zh") for i in range(60)]
        balanced = balance_oos_by_language(rows)
        en_kept = [row for row in balanced if row["lang"] == "en"]
        assert len(en_kept) == 5
        assert {row["ask"] for row in en_kept} == {f"en{i}" for i in range(5)}

    def test_rows_with_no_lang_field_are_kept_unbalanced(self) -> None:
        rows = [_row("en0", "en"), _row("zh0", "zh"), {"ask": "no-lang-row"}]
        balanced = balance_oos_by_language(rows)
        assert {"ask": "no-lang-row"} in balanced

    def test_empty_input_returns_empty(self) -> None:
        assert balance_oos_by_language([]) == []

    def test_a_single_language_is_returned_whole_not_dropped(self) -> None:
        rows = [_row(f"en{i}", "en") for i in range(17)]
        balanced = balance_oos_by_language(rows)
        assert len(balanced) == 17

    def test_the_seed_is_fixed_not_a_placeholder(self) -> None:
        assert isinstance(OOS_BALANCE_SEED, int)
