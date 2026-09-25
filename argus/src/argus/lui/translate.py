"""An answer in the reader's language, with every figure locked to the English it came from.

The research engines write English. A question asked in Chinese was answered in English under a
one-line note saying so, because putting a model between a computed figure and the reader is the
one thing this console refused to do (`server._language_note`). That rule stays; what changes is
how it is kept. A language model translates the lines, and each translated line is then checked
against its English source: the multiset of numbers in it (every digit run, so prices, percentages,
dates, counts and basis points) must be identical, and every URL must survive character for
character. A line that fails keeps its English. So the model can change the words around a figure
and never the figure — baserate (a Season 2 desk) guards its narration with the same idea, an
allow-list of numbers taken from the computed dossier.

Measured on 2026-09-25 against the hackathon Qwen with thinking off: an eight-line quote answer
translated in 9.3 seconds for 723 tokens, every figure intact. Too slow to hold an answer back for,
so the page shows the English at once and swaps the translation in when it arrives; the request is
signed by the server that produced the lines (HMAC over the language and the lines), so the
endpoint translates only answers this console wrote and cannot be used as a free translator.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
from collections import Counter
from typing import Any

LANGUAGES: dict[str, str] = {
    "zh": "Simplified Chinese", "es": "Spanish", "pt": "Portuguese", "fr": "French",
    "de": "German", "ja": "Japanese", "ko": "Korean", "vi": "Vietnamese", "zh-Hant":
    "Traditional Chinese",
}
MAX_CHARS = 8000
"""An answer larger than this is not translated; none measured so far comes near it."""

NOTES: dict[str, str] = {
    "zh": ("以下由 Qwen 翻译成中文；每一个数字都已与引擎的英文原文逐一核对，"  # noqa: RUF001
           "数字不一致的行保留英文。"),
    "zh-Hant": ("以下由 Qwen 翻譯成中文；每一個數字都已與引擎的英文原文逐一核對，"  # noqa: RUF001
                "數字不一致的行保留英文。"),
    "es": "Traducido por Qwen; cada cifra se ha comprobado contra el original en inglés, y las "
          "líneas cuyas cifras no coinciden quedan en inglés.",
    "pt": "Traduzido pelo Qwen; cada número foi conferido com o original em inglês, e as linhas "
          "cujos números não batem ficam em inglês.",
    "fr": "Traduit par Qwen ; chaque chiffre a été vérifié contre l'original anglais, et les "
          "lignes dont les chiffres ne correspondent pas restent en anglais.",
    "de": "Von Qwen übersetzt; jede Zahl wurde mit dem englischen Original abgeglichen, Zeilen "
          "mit abweichenden Zahlen bleiben englisch.",
    "ja": ("Qwen による翻訳です。すべての数字はエンジンの英語原文と照合済みで、"
           "一致しない行は英語のまま表示します。"),
    "ko": "Qwen이 번역했습니다. 모든 숫자는 엔진의 영어 원문과 대조했으며, 숫자가 다른 줄은 영어로 "
          "남겨 두었습니다.",
    "vi": "Bản dịch của Qwen; mọi con số đã được đối chiếu với bản tiếng Anh gốc, dòng nào số "
          "không khớp thì giữ nguyên tiếng Anh.",
}

_NUMBER = re.compile(r"\d+")
_URL = re.compile(r"https?://\S+")
_KANA = re.compile(r"[぀-ヿ]")
_HANGUL = re.compile(r"[가-힯]")
_TRADITIONAL = re.compile(
    r"[體圖點價這個會與為資費貴嗎輝達們說買賣對"
    r"開關過時間風險請問還應該歷當從後幾結]")
"""Characters written only in Traditional Chinese, common in trading questions."""
_HAN = re.compile(r"[一-鿿]")


def target_language(text: str) -> str | None:
    """The language to answer in, or None for English. Only scripts and languages the console
    already notes are detected; anything else is answered in English as before."""
    if _KANA.search(text):
        return "ja"
    if _HANGUL.search(text):
        return "ko"
    if _HAN.search(text):
        return "zh-Hant" if _TRADITIONAL.search(text) else "zh"
    lowered = f" {text.lower()} "
    for code, words in (("es", (" qué ", " cómo ", " mi cuenta", " si el ", " debería ", "¿")),
                        ("pt", (" minha ", " carteira", " se o ", " ações", " você ")),
                        ("fr", (" pourquoi ", " est-ce ", " mon portefeuille", " si le ")),
                        ("de", (" wenn ", " mein ", " meine ", " warum ", " passiert "))):
        if any(w in lowered for w in words):
            return code
    return None


def figures_match(source: str, translated: str) -> bool:
    """Every digit run and every URL of ``source`` is in ``translated``, and nothing numeric was
    added: the multisets of numbers are equal."""
    if Counter(_NUMBER.findall(source)) != Counter(_NUMBER.findall(translated)):
        return False
    return all(url in translated for url in _URL.findall(source))


def _secret() -> bytes:
    global _PROCESS_SECRET
    configured = os.environ.get("ARGUS_SIGNING_SECRET", "")
    if configured:
        return configured.encode()
    if _PROCESS_SECRET is None:
        _PROCESS_SECRET = secrets.token_bytes(32)
    return _PROCESS_SECRET


_PROCESS_SECRET: bytes | None = None


def sign(lang: str, lines: list[str]) -> str:
    body = lang + "\n" + json.dumps(lines, ensure_ascii=False)
    return hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest()[:40]


def verify(lang: str, lines: list[str], token: str) -> bool:
    return hmac.compare_digest(sign(lang, lines), token or "")


_CACHE: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def translate(lines: list[str], lang: str, client: Any) -> dict[str, Any]:
    """``{"lines", "kept_english", "note"}``. Lines whose figures do not survive keep their
    English; a model failure keeps them all, and says so."""
    if lang not in LANGUAGES or client is None:
        return {"lines": lines, "kept_english": list(range(len(lines))), "note": None}
    key = sign(lang, lines)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    from argus.llm.qwen import Thinking

    messages = [
        {"role": "system", "content":
            f"Translate each line into {LANGUAGES[lang]} for a trader reading a research desk's "
            f"answer. Keep every number, percentage, ticker, date, time and URL exactly as "
            f"written, in Arabic digits. Do not add, drop or merge lines, and add nothing. Reply "
            f"as JSON {{\"lines\": [...]}} with one translated line per input line, same order."},
        {"role": "user", "content": json.dumps({"lines": lines}, ensure_ascii=False)},
    ]
    try:
        out = client.complete_json(messages, required_keys=("lines",), max_tokens=4000,
                                   thinking=Thinking.OFF)
        translated = out.get("lines")
    except Exception:
        translated = None
    if not isinstance(translated, list) or len(translated) != len(lines):
        return {"lines": lines, "kept_english": list(range(len(lines))), "note": None}
    result_lines: list[str] = []
    kept: list[int] = []
    for i, (src, dst) in enumerate(zip(lines, translated, strict=True)):
        if isinstance(dst, str) and dst.strip() and figures_match(src, dst):
            result_lines.append(dst.strip())
        else:
            result_lines.append(src)
            kept.append(i)
    result = {"lines": result_lines, "kept_english": kept, "note": NOTES.get(lang)}
    with _LOCK:
        _CACHE[key] = result
    return result
