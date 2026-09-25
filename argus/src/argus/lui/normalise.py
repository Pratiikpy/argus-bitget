"""Fold the character variants a bilingual console receives onto the forms its layers were built on.

**Found by an adversarial test, not assumed.** `eval/lui_rematch.py` perturbed the sealed intent
corpus the way real users do without trying, and scored the deployed router before anything was
changed (`data/lui_rematch_argus_before_fixes.json`). Two perturbations that preserve a question's
meaning exactly broke it:

* **Full-width Latin letters and digits**, which a Chinese input method in full-width mode types
  for "NVDA" or "30%": ``ＮＶＤＡ`` shares no code point with ``NVDA``, so the n-gram model saw no
  known feature and the patterns' ``[A-Za-z]`` classes saw no letters. 15 of 122 such rows were
  routed correctly.
* **Traditional characters** (``爲什麼買入這個倉位`` for ``为什么买入这个仓位``), which is how Hong
  Kong and Taiwan write. Every training row and every Chinese pattern is Simplified.

Both are folded here, and only these two: ASCII full-width forms U+FF01..U+FF5E to U+0021..U+007E
(the Unicode Standard's own compatibility mapping for that block, which NFKC applies), the
ideographic space U+3000 to a space, and Traditional characters to Simplified by OpenCC's
character table (``TSCharacters.txt``, Apache-2.0, vendored as `data/lui_t2s_characters.json`).

**Deliberately not NFKC.** NFKC also rewrites full-width *punctuation* — ``？`` and ``，`` — which
the Chinese training rows carry, so it would change the features of text the model was fitted on
and the stdlib scorer would stop agreeing with the fitted scikit-learn model.
`tests/test_lui_normalise.py` asserts that nothing but a mapped character ever changes, on every
held-out question; routing on those corpora was the same with the fold on (210/240, 190/200,
236/240, 2026-09-26).

**Character-level, not phrase-level.** OpenCC's full conversion also consults a phrase table for
the few words whose character mapping is ambiguous; a character n-gram model and a regular
expression both work below the word, so the character table is the layer that matters here, and it
keeps this module a dict lookup with no segmentation.

If the table is missing — a deployment that did not bundle it — Traditional text passes through
unchanged rather than failing: the console loses this robustness, never its answer.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

T2S_PATH = Path(__file__).resolve().parents[3] / "data" / "lui_t2s_characters.json"

_FULLWIDTH = {
    code: code - 0xFEE0 for code in range(0xFF01, 0xFF5F) if chr(code - 0xFEE0).isalnum()
}
"""Full-width letters and digits only. Full-width punctuation is left alone on purpose (see the
module docstring): the Chinese training rows carry it, and folding it would change the features the
model was fitted on."""
_FULLWIDTH[0x3000] = 0x20


@lru_cache(maxsize=1)
def _table() -> dict[int, int]:
    """Full-width and Traditional folds as one ``str.translate`` table, loaded once."""
    table = dict(_FULLWIDTH)
    try:
        blob = json.loads(T2S_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return table
    for trad, simp in blob.get("map", {}).items():
        if len(trad) == 1 and len(simp) == 1:
            table[ord(trad)] = ord(simp)
    return table


def fold(text: str) -> str:
    """``text`` with full-width letters and digits made ASCII and Traditional characters made
    Simplified. Idempotent, and the identity on text that carries neither."""
    return text.translate(_table())


def t2s_available() -> bool:
    """Is the Traditional-to-Simplified table on disk? Reported by the rematch, never assumed."""
    return T2S_PATH.exists()


__all__ = ["T2S_PATH", "fold", "t2s_available"]
