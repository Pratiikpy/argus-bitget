"""The frozen inputs for the LUI rematch against Rasa's DIET classifier, built once and committed.

`eval/lui_comparison.py` measured ARGUS's intent router against Rasa on 293 sealed rows and 14
out-of-scope probes, with Rasa trained once, unseeded, on the training rows as they stood on
2026-09-21. Its own register entry names what that left unproven: no variance estimate for either
system, an out-of-scope set too small to carry a safety claim (Wilson intervals of [0.69, 0.99]
against [0.21, 0.67]), no adversarial test, and a Rasa side trained on 334 rows while ARGUS's
side has since moved to 314. This module freezes the inputs that close those gaps, so both systems
are scored on byte-identical strings and a reader can re-run either side without re-deriving them.

**Every choice below was fixed before either system was scored on any of it.** The probe sets
come from two public corpora neither system was trained on, filtered by rules written here rather
than by looking at predictions:

* **MASSIVE** (FitzGerald et al. 2022, arXiv 2206.08729; Amazon, CC BY 4.0) — parallel utterances
  across 51 languages. 4 test-split ids per intent, seeded, and the *same ids* taken in ``en-US``,
  ``zh-CN`` and ``zh-TW``, so the English, Simplified and Traditional thirds are translations of
  each other and a language gap cannot be a content gap. The two finance intents (``qa_stock``,
  ``qa_currency``) are excluded by label, because a stock-price question is in scope for a
  trading console.
* **CLINC150** ``oos_test`` (Larson et al. 2019, EMNLP; CC BY 3.0) — the 1,000 English queries the
  standard out-of-scope benchmark holds out as belonging to none of its 150 intents.

Both are then filtered by :data:`FINANCE_LEXICON`, a list written for this module and deliberately
**not** ARGUS's own topic gate (`lui/server._DOMAIN`): reusing the system under test's vocabulary
to choose its test set would let it pick the rows it is judged on. A probe that mentions a stock,
a market or a currency is dropped because its correct answer is arguably not a refusal, and the
count dropped is recorded rather than hidden.

**Adversarial rows** are label-preserving perturbations of the sealed rows, in the style of
CheckList's invariance tests (Ribeiro et al. 2020, ACL; `marcotcr/checklist`, MIT). The typo
perturbation is CheckList's ``Perturb.add_typos`` (``checklist/perturb.py:148-170``) — adjacent
characters swapped at seeded positions — reimplemented on the stdlib ``random`` module instead of
``numpy.random`` so the seed alone reproduces it. The others are the ones a real user of a
bilingual console produces without trying: shouting in capitals, dropping punctuation, typing
Latin letters through a Chinese IME (full-width forms), writing in Traditional rather than
Simplified characters, and wrapping a question in greetings. Traditional characters come from
``opencc-python-reimplemented`` (Apache-2.0), needed only when this builder runs; the converted
strings are stored, so scoring never needs it.

**Two adversarial suites, and the second exists because the first was used.** ``perturbations``
applies the transforms to the sealed rows and was scored before anything in this pass changed.
Its failures were then read and fixed, which spends it as a blind test of the fixed code. So
``confirmation_perturbations`` applies the same transforms, with a different seed
(:data:`CONFIRMATION_SEED`), to rows none of the fixes was designed on — the ``validation`` and
``final`` splits — and it is the suite a post-fix robustness claim may quote. Both suites are in
this file before either system is scored on the second.

**Hard negatives** are out-of-scope requests built around the very names this console trades —
"tell me a joke about NVDA" — because an out-of-scope set made only of pasta recipes measures the
easy half. Their templates are fixed below and every one is a request no reading of the console's
scope can answer (jokes, poems, songs, translation, pronunciation, drawing).

    python -m argus.eval.lui_rematch_inputs --massive-dir <dir> --clinc <data_full.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from argus.eval.artefact import write

DATA = Path(__file__).resolve().parents[3] / "data"
INPUTS_PATH = DATA / "lui_rematch_inputs.json"

SEED = 20260925
"""One seed for every sampled or perturbed choice in this file. Chosen as the build date, not for
any result: nothing had been scored on these inputs when it was set."""

CONFIRMATION_SEED = 20260926
"""The confirmation suite's own seed, so its typo positions are not the first suite's."""

CONFIRMATION_SPLITS = ("validation", "final")

MASSIVE_PER_INTENT = 4
MASSIVE_EXCLUDED_INTENTS = frozenset({"qa_stock", "qa_currency"})
MASSIVE_INTENTS: tuple[str, ...] = (
    # Verbatim from the dataset's own loader (huggingface.co/datasets/AmazonScience/massive,
    # massive.py `_INTENTS`), whose index order is what the parquet's integer `intent` column uses.
    "datetime_query", "iot_hue_lightchange", "transport_ticket", "takeaway_query", "qa_stock",
    "general_greet", "recommendation_events", "music_dislikeness", "iot_wemo_off",
    "cooking_recipe", "qa_currency", "transport_traffic", "general_quirky", "weather_query",
    "audio_volume_up", "email_addcontact", "takeaway_order", "email_querycontact",
    "iot_hue_lightup", "recommendation_locations", "play_audiobook", "lists_createoradd",
    "news_query", "alarm_query", "iot_wemo_on", "general_joke", "qa_definition", "social_query",
    "music_settings", "audio_volume_other", "calendar_remove", "iot_hue_lightdim",
    "calendar_query", "email_sendemail", "iot_cleaning", "audio_volume_down", "play_radio",
    "cooking_query", "datetime_convert", "qa_maths", "iot_hue_lightoff", "iot_hue_lighton",
    "transport_query", "music_likeness", "email_query", "play_music", "audio_volume_mute",
    "social_post", "alarm_set", "qa_factoid", "calendar_set", "play_game", "alarm_remove",
    "lists_remove", "transport_taxi", "recommendation_movies", "iot_coffee", "music_query",
    "play_podcasts", "lists_query",
)

FINANCE_LEXICON = re.compile(
    r"\b(?:stocks?|shares?|share\s+price|stock\s+market|markets?|dow|nasdaq|s\s*&\s*p|"
    r"invest\w*|portfolio\w*|trad(?:e|es|ed|er|ers|ing)|crypto\w*|bitcoin|ethereum|btc|eth|"
    r"currenc\w*|exchange\s+rates?|forex|dividends?|bonds?|etfs?|hedg\w*|brokers?|"
    r"futures|options\s+chain|ticker|equit\w*|nyse|interest\s+rates?|mortgage|loan|"
    r"bank\w*|money|dollars?|euros?|yen|yuan|pounds?\s+sterling)\b"
    r"|股票|股市|股价|基金|投资|交易|汇率|比特币|加密|债券|期货|外汇|证券|指数|道琼斯|纳斯达克|"
    r"行情|货币|美元|欧元|日元|人民币|银行|贷款|利率|理财|钱",
    re.I,
)
"""What makes a probe arguably in scope for a trading console, written for this file.

**Deliberately not `lui/server._DOMAIN`.** That regex is part of the system under test — it is
ARGUS's own topic gate — and selecting the test set with it would guarantee the gate passes every
probe that survives. This list is wider on money words and narrower on everything else, and it is
applied identically to what both systems see."""

HARD_NEGATIVE_TEMPLATES_EN: tuple[str, ...] = (
    "tell me a joke about {name}",
    "write a poem about {name}",
    "sing me a song about {name}",
    "how do you pronounce {name}",
    "translate the word {name} into french",
    "draw me a picture of the {name} logo",
)
HARD_NEGATIVE_TEMPLATES_ZH: tuple[str, ...] = (
    "讲一个关于{name}的笑话",
    "写一首关于{name}的诗",
    "给我唱一首关于{name}的歌",
    "{name}这个词怎么读",
    "把{name}翻译成法语",
    "画一张{name}的标志给我看",
)
HARD_NEGATIVE_NAMES: tuple[str, ...] = (
    "NVDA", "TSLA", "AAPL", "MSFT", "COIN", "MSTR", "Nvidia", "Tesla", "Apple", "Microsoft",
)

POLITE_WRAP = {
    "en": ("hi, quick question: ", " thanks"),
    # Full-width comma and colon on purpose: they are what a Chinese IME types.
    "zh": ("你好，请问一下：", " 谢谢"),  # noqa: RUF001
}


class RematchInputError(RuntimeError):
    """An input source is missing or malformed. Raised rather than building a smaller set."""


def digest(blob: Any) -> str:
    """SHA-256 of a canonical JSON encoding — the identity a reader checks the inputs against."""
    text = json.dumps(blob, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- perturbations -----------------------------------------------------------------------------


def add_typos(text: str, typos: int, rng: random.Random) -> str:
    """CheckList's ``Perturb.add_typos`` (`checklist/perturb.py:148-170`, MIT), on stdlib random.

    Same algorithm: draw ``typos`` positions from ``range(len - 1)`` *with replacement* (their
    ``np.random.choice(len(string) - 1, typos)``) and swap each with its right neighbour, in draw
    order. Only the random source differs, so a seed reproduces the output without numpy.
    """
    chars = list(text)
    if len(chars) < 2:
        return text
    for swap in [rng.randrange(len(chars) - 1) for _ in range(typos)]:
        chars[swap], chars[swap + 1] = chars[swap + 1], chars[swap]
    return "".join(chars)


def strip_punctuation(text: str) -> str:
    """Remove every Unicode punctuation character (categories P*), ASCII and CJK alike."""
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))


def to_fullwidth(text: str) -> str:
    """ASCII ``!``..``~`` to their full-width forms (U+FF01..U+FF5E), as a Chinese IME in
    full-width mode types them. Spaces are left alone."""
    return "".join(chr(ord(ch) + 0xFEE0) if "!" <= ch <= "~" else ch for ch in text)


def polite_wrap(text: str, lang: str) -> str:
    prefix, suffix = POLITE_WRAP.get(lang, POLITE_WRAP["en"])
    return f"{prefix}{text}{suffix}"


def perturbations(
    rows: Sequence[dict[str, str]], *, to_traditional: Callable[[str], str] | None,
    seed: int = SEED,
) -> dict[str, list[dict[str, str]]]:
    """Every label-preserving perturbation of the sealed rows, only where it changes the string.

    A row the perturbation leaves identical is dropped from that set rather than counted: an
    unchanged string is not a robustness test, and counting it would inflate both systems'
    invariance by the same meaningless amount.
    """
    rng = random.Random(seed)
    out: dict[str, list[dict[str, str]]] = {
        "typo1": [], "typo2": [], "upper": [], "no_punctuation": [], "fullwidth": [],
        "polite_wrap": [], "traditional": [],
    }
    for row in rows:
        ask, lang = row["ask"], row.get("lang", "en")
        candidates = {
            "typo1": add_typos(ask, 1, rng),
            "typo2": add_typos(ask, 2, rng),
            "upper": ask.upper(),
            "no_punctuation": strip_punctuation(ask),
            "fullwidth": to_fullwidth(ask),
            "polite_wrap": polite_wrap(ask, lang),
        }
        if to_traditional is not None and lang == "zh":
            candidates["traditional"] = to_traditional(ask)
        for name, perturbed in candidates.items():
            if perturbed != ask and perturbed.strip():
                out[name].append({
                    "ask": perturbed, "original": ask, "expect": row["expect"], "lang": lang,
                })
    return out


def hard_negatives() -> list[dict[str, str]]:
    rows = [
        {"ask": t.format(name=n), "lang": "en", "template": t}
        for t in HARD_NEGATIVE_TEMPLATES_EN for n in HARD_NEGATIVE_NAMES
    ]
    rows += [
        {"ask": t.format(name=n), "lang": "zh", "template": t}
        for t in HARD_NEGATIVE_TEMPLATES_ZH for n in HARD_NEGATIVE_NAMES
    ]
    return rows


# --- external out-of-scope corpora -------------------------------------------------------------


def _read_massive(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found, import-untyped, unused-ignore]
    except ImportError as exc:  # pragma: no cover - builder-only dependency
        raise RematchInputError("pyarrow is needed to read the MASSIVE parquet files") from exc
    if not path.exists():
        raise RematchInputError(f"{path} is missing")
    rows: list[dict[str, Any]] = pq.read_table(path).to_pylist()
    return rows


def massive_probes(
    en_rows: Iterable[dict[str, Any]], zh_rows: Iterable[dict[str, Any]],
    known: frozenset[str], tw_rows: Iterable[dict[str, Any]] = (),
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Seeded, intent-stratified, parallel probes. Returns ``(probes, exclusion counts)``.

    Ids are chosen on the English and Simplified Chinese pair; the Traditional Chinese utterance of
    the same id is added when the file carries it and it passes the same filters, so the choice of
    ids never depends on the third language.
    """
    en = {str(r["id"]): r for r in en_rows}
    zh = {str(r["id"]): r for r in zh_rows}
    tw = {str(r["id"]): r for r in tw_rows}
    excluded = {"finance_intent": 0, "finance_lexicon": 0, "seen_in_training": 0,
                "missing_translation": 0}
    by_intent: dict[str, list[str]] = {}
    for rid in sorted(en, key=lambda x: int(x)):
        intent = MASSIVE_INTENTS[int(en[rid]["intent"])]
        if intent in MASSIVE_EXCLUDED_INTENTS:
            excluded["finance_intent"] += 1
            continue
        if rid not in zh:
            excluded["missing_translation"] += 1
            continue
        pair = (str(en[rid]["utt"]), str(zh[rid]["utt"]))
        if any(FINANCE_LEXICON.search(u) for u in pair):
            excluded["finance_lexicon"] += 1
            continue
        if any(u.casefold() in known for u in pair):
            excluded["seen_in_training"] += 1
            continue
        by_intent.setdefault(intent, []).append(rid)
    rng = random.Random(SEED)
    probes: list[dict[str, str]] = []
    for intent in sorted(by_intent):
        ids = by_intent[intent]
        for rid in sorted(rng.sample(ids, min(MASSIVE_PER_INTENT, len(ids))), key=int):
            probes.append({"ask": str(en[rid]["utt"]), "lang": "en", "id": rid,
                           "source_intent": intent})
            probes.append({"ask": str(zh[rid]["utt"]), "lang": "zh", "id": rid,
                           "source_intent": intent})
            if rid in tw:
                hant = str(tw[rid]["utt"])
                if FINANCE_LEXICON.search(hant) or hant.casefold() in known:
                    excluded["zh_hant_filtered"] = excluded.get("zh_hant_filtered", 0) + 1
                else:
                    probes.append({"ask": hant, "lang": "zh-Hant", "id": rid,
                                   "source_intent": intent})
    return probes, excluded


def clinc_probes(
    blob: dict[str, Any], known: frozenset[str],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    excluded = {"finance_lexicon": 0, "seen_in_training": 0, "duplicate": 0}
    seen: set[str] = set()
    probes: list[dict[str, str]] = []
    for text, label in blob["oos_test"]:
        if label != "oos":
            continue
        ask = str(text)
        if FINANCE_LEXICON.search(ask):
            excluded["finance_lexicon"] += 1
            continue
        if ask.casefold() in known:
            excluded["seen_in_training"] += 1
            continue
        if ask.casefold() in seen:
            excluded["duplicate"] += 1
            continue
        seen.add(ask.casefold())
        probes.append({"ask": ask, "lang": "en"})
    return probes, excluded


# --- assembly -----------------------------------------------------------------------------------


def build(massive_dir: Path, clinc_path: Path) -> dict[str, Any]:
    from argus.eval.luirouter import OUT_OF_SCOPE
    from argus.eval.ngrambench import _rows
    from argus.eval.ngramtrain import training_rows

    texts, labels = training_rows()
    known = frozenset(t.casefold() for t in texts)
    sealed = [{"ask": r["ask"], "expect": r["expect"], "lang": r.get("lang", "en")}
              for r in _rows("sealed")]

    try:
        import opencc  # type: ignore[import-not-found, unused-ignore]

        converter = opencc.OpenCC("s2t")
        to_traditional: Callable[[str], str] | None = converter.convert
    except ImportError:  # pragma: no cover - builder-only dependency
        to_traditional = None

    massive, massive_excluded = massive_probes(
        _read_massive(massive_dir / "massive_en-US_test.parquet"),
        _read_massive(massive_dir / "massive_zh-CN_test.parquet"), known,
        _read_massive(massive_dir / "massive_zh-TW_test.parquet"),
    )
    confirmation_base = [
        {"ask": r["ask"], "expect": r["expect"], "lang": r.get("lang", "en"), "split": split}
        for split in CONFIRMATION_SPLITS for r in _rows(split)
    ]
    if not clinc_path.exists():
        raise RematchInputError(f"{clinc_path} is missing")
    clinc, clinc_excluded = clinc_probes(json.loads(clinc_path.read_text(encoding="utf-8")), known)

    blob: dict[str, Any] = {
        "built_by": "argus.eval.lui_rematch_inputs",
        "seed": SEED,
        "training": [{"text": t, "label": lab} for t, lab in zip(texts, labels, strict=True)],
        "sealed": sealed,
        "oos_probes_14": list(OUT_OF_SCOPE),
        "massive_oos": massive,
        "clinc_oos": clinc,
        "hard_negatives": hard_negatives(),
        "perturbations": perturbations(sealed, to_traditional=to_traditional),
        "confirmation_base": confirmation_base,
        "confirmation_perturbations": perturbations(
            confirmation_base, to_traditional=to_traditional, seed=CONFIRMATION_SEED,
        ),
        "excluded": {"massive": massive_excluded, "clinc": clinc_excluded},
        "sources": {
            "massive": {
                "citation": "FitzGerald et al. 2022, MASSIVE, arXiv 2206.08729",
                "files": (
                    "AmazonScience/massive, refs/convert/parquet "
                    "{en-US,zh-CN,zh-TW}/test/0000.parquet"
                ),
                "licence": "CC BY 4.0",
                "per_intent": MASSIVE_PER_INTENT,
                "excluded_intents": sorted(MASSIVE_EXCLUDED_INTENTS),
            },
            "clinc150": {
                "citation": "Larson et al. 2019, An Evaluation Dataset for Intent Classification "
                            "and Out-of-Scope Prediction, EMNLP",
                "files": "clinc/oos-eval data/data_full.json, oos_test",
                "licence": "CC BY 3.0",
            },
            "typos": "marcotcr/checklist checklist/perturb.py:148-170 (MIT), reimplemented",
            "traditional": (
                "opencc-python-reimplemented 0.1.7 s2t (Apache-2.0)"
                if to_traditional is not None else "not available at build time"
            ),
        },
    }
    blob["digests"] = {
        key: digest(blob[key])
        for key in ("training", "sealed", "oos_probes_14", "massive_oos", "clinc_oos",
                    "hard_negatives", "perturbations", "confirmation_base",
                    "confirmation_perturbations")
    }
    return blob


def load(path: Path = INPUTS_PATH) -> dict[str, Any]:
    if not path.exists():
        raise RematchInputError(
            f"{path} is missing. It is committed with the repo; rebuilding it needs the MASSIVE "
            "and CLINC150 source files (see this module's docstring)."
        )
    blob: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return blob


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="freeze the LUI rematch inputs")
    parser.add_argument("--massive-dir", type=Path, required=True)
    parser.add_argument("--clinc", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=INPUTS_PATH)
    args = parser.parse_args()
    blob = build(args.massive_dir, args.clinc)
    write(args.out, blob)
    print(f"training {len(blob['training'])}  sealed {len(blob['sealed'])}  "
          f"massive {len(blob['massive_oos'])}  clinc {len(blob['clinc_oos'])}  "
          f"hard negatives {len(blob['hard_negatives'])}")
    print("perturbations: " + ", ".join(f"{k} {len(v)}" for k, v in blob["perturbations"].items()))
    print("confirmation: " + ", ".join(
        f"{k} {len(v)}" for k, v in blob["confirmation_perturbations"].items()))
    print(f"excluded: {blob['excluded']}")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CONFIRMATION_SEED", "FINANCE_LEXICON", "INPUTS_PATH", "SEED", "RematchInputError", "add_typos",
    "build", "clinc_probes", "digest", "hard_negatives", "load", "massive_probes", "perturbations",
    "polite_wrap", "strip_punctuation", "to_fullwidth",
]
