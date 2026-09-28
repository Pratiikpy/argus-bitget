"""English plurals resolved against the number in front of them: "1 trade", "789 entries".

The console wrote ``{n} trade(s)`` in some forty places and its translation templates did the same
(``"Hash chain {state} across {count} entry(ies)."``), so every answer a trader read said
"1 decision(s)" (research/harvest/53-i18next.md). i18next solves this with a plural suffix per key
chosen by ``Intl.PluralRules`` (``src/PluralResolver.js:40-90``); its fallback when that API is
absent is the rule English needs, ``count === 1 ? 'one' : 'other'`` (``:12-17``, MIT). Chinese marks
no plural at all, so English is the only language this touches.

Rather than rewrite every template, this resolves the marker where an answer leaves the system: a
number, a word and ``(s)``/``(es)``/``(ies)`` become the singular for exactly one and the plural
otherwise. A marker with no number directly before it ("decision(s) 12, 14") is left as written,
because the count it refers to is not in the sentence.
"""

from __future__ import annotations

import re

_MARKED = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(\s+)([A-Za-z]+(?:[ -][A-Za-z]+)?)"
                     r"\((s|es|ies)\)")


def _one(number: str) -> bool:
    try:
        return float(number.replace(",", "")) == 1
    except ValueError:
        return False


def _form(word: str, suffix: str, one: bool) -> str:
    if suffix == "ies":
        # "entry(ies)": the marker is written after the singular, which ends in "y".
        return word if one else (word[:-1] + "ies" if word.endswith("y") else word + "ies")
    return word if one else word + suffix


def resolve_plurals(text: str) -> str:
    """``text`` with every counted ``word(s)`` marker resolved against its count."""
    return _MARKED.sub(lambda m: m.group(1) + m.group(2)
                       + _form(m.group(3), m.group(4), _one(m.group(1))), text)


__all__ = ["resolve_plurals"]
