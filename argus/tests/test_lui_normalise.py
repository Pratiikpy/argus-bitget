"""The character folds the console applies before reading a question (`lui/normalise.py`)."""

from __future__ import annotations

import json
from pathlib import Path

from argus.lui.normalise import fold, t2s_available

DATA = Path(__file__).resolve().parents[1] / "data"


def test_full_width_letters_and_digits_become_ascii() -> None:
    assert fold("ＮＶＤＡ 下跌３０％怎么办") == "NVDA 下跌30％怎么办"


def test_full_width_punctuation_is_left_as_the_training_rows_carry_it() -> None:
    assert fold("英伟达？，") == "英伟达？，"


def test_traditional_characters_become_simplified() -> None:
    assert t2s_available()
    assert fold("爲什麼買入這個倉位") == "为什么买入这个仓位"


def test_fold_is_idempotent_and_the_identity_on_plain_text() -> None:
    for text in ("what is the sharpe", "英伟达现在值得买吗", "ＮＶＤＡ爲什麼"):
        assert fold(fold(text)) == fold(text)
    assert fold("how risky is NVDA? 50% AAPL") == "how risky is NVDA? 50% AAPL"


def test_the_fold_changes_nothing_but_the_characters_it_maps() -> None:
    """On every held-out question, each character the fold changes is a full-width letter or digit
    or a Traditional character in its table — nothing else is touched (routing on those corpora is
    unchanged with the fold on: 210/240, 190/200, 236/240 on 2026-09-26)."""
    from argus.lui.normalise import _table

    table = _table()
    for path in sorted(DATA.glob("lui_*_2026-09-25.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            text = str(json.loads(line)["text"])
            for before, after in zip(text, fold(text), strict=True):
                assert before == after or ord(before) in table, (text, before)
