"""ARGUS's injection rules against a model-based classifier, on attacks neither was written for.

`eval/garak_quarantine.py` measured `agents/quarantine.py`'s rules on NVIDIA garak's corpora and
against the rules' own earlier version. The rival that matters was never run
(Activity/28_CAPABILITY_CLOSE_PLAN_2.md §47): a trained prompt-injection classifier. The one run
here is ProtectAI's `deberta-v3-base-prompt-injection-v2` (Apache-2.0, local CPU inference through
`transformers`, archived by its authors but still the most-used open checkpoint). Meta's Prompt
Guard 2 is gated behind a licence click and Lakera Guard is a paid API; both are named as not run.

**The corpora, and what each tests.**

* ``deepset``: `deepset/prompt-injections` (Apache-2.0, notice at
  licenses/deepset_prompt_injections-APACHE-2.0.txt; 662 labelled prompts, English and German),
  sampled by its authors rather than templated here, and not among the classifier's listed
  training sets. It is direct injection and role-play jailbreaking; ARGUS's rules were written for
  injection hidden in evidence (news, filings, posts), so this is the hardest fair test for them.
* ``garak_development`` / ``garak_held_out``: the latent-injection and promptinject corpora of
  `eval/garak_quarantine.py`, the first of which ARGUS's rules were written from.
* ``clean``: text the desk really reads and must not withhold, the paper ledger's theses and
  README lines from the cloned repositories (`eval/quarantine_generalisation`).

Every item gets both verdicts, so the comparison is paired, and the deepset items are kept as rows.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from argus.agents.quarantine import inspect, withholds
from argus.truth import artefact
from argus.truth.paths import DATA_DIR

REPORT_PATH = DATA_DIR / "injection_classifier_rival.json"
DEEPSET_PATH = DATA_DIR / "deepset_prompt_injections.json"
MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
ROWS_URL = ("https://datasets-server.huggingface.co/rows?dataset=deepset/prompt-injections"
            "&config=default&split={split}&offset={offset}&length=100")
THRESHOLD = 0.5
"""The classifier's own decision rule: the INJECTION label wins the two-way softmax."""

Detector = Callable[[str], bool]


def fetch_deepset() -> list[dict[str, Any]]:  # pragma: no cover - network
    """Every row of both splits, through Hugging Face's public rows API (no key)."""
    out: list[dict[str, Any]] = []
    for split in ("train", "test"):
        offset = 0
        while True:
            request = urllib.request.Request(ROWS_URL.format(split=split, offset=offset),
                                             headers={"User-Agent": "argus-research"})
            with urllib.request.urlopen(request, timeout=30) as response:
                rows = json.load(response).get("rows") or []
            if not rows:
                break
            out.extend({"split": split, "text": str(r["row"]["text"]),
                        "label": int(r["row"]["label"])} for r in rows)
            offset += len(rows)
    return out


def argus_rules(text: str) -> bool:
    return bool(withholds(inspect(text)))


def load_classifier() -> Detector:  # pragma: no cover - loads a model
    from transformers import pipeline

    classify = pipeline("text-classification", model=MODEL, truncation=True, max_length=512)

    def detect(text: str) -> bool:
        result = classify(text)[0]
        return bool(result["label"] == "INJECTION" and result["score"] >= THRESHOLD)

    return detect


def exact_mcnemar(b: int, c: int) -> float:
    from fractions import Fraction
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    tail = sum(Fraction(comb(n, k), 2 ** n) for k in range(0, min(b, c) + 1))
    return float(min(Fraction(1), 2 * tail))


def score(items: Sequence[dict[str, Any]], arms: dict[str, Detector]) -> dict[str, Any]:
    """Recall on attacks, withholds on clean text, per arm, with the per-item verdicts."""
    verdicts = [{arm: detect(str(item["text"])) for arm, detect in arms.items()} for item in items]
    out: dict[str, Any] = {"items": len(items)}
    for arm in arms:
        attacks = [v[arm] for v, i in zip(verdicts, items, strict=True) if i["label"] == 1]
        clean = [v[arm] for v, i in zip(verdicts, items, strict=True) if i["label"] == 0]
        out[arm] = {"caught": sum(attacks), "attacks": len(attacks),
                    "recall": round(sum(attacks) / len(attacks), 4) if attacks else None,
                    "clean_withheld": sum(clean), "clean": len(clean)}
    names = list(arms)
    if len(names) == 2:
        a, b = names
        for label, what in ((1, "attacks"), (0, "clean")):
            only_a = sum(1 for v, i in zip(verdicts, items, strict=True)
                         if i["label"] == label and v[a] and not v[b])
            only_b = sum(1 for v, i in zip(verdicts, items, strict=True)
                         if i["label"] == label and v[b] and not v[a])
            out[f"discordant_{what}"] = {f"only_{a}": only_a, f"only_{b}": only_b,
                                         "exact_mcnemar_p": exact_mcnemar(only_a, only_b)}
    out["verdicts"] = verdicts
    return out


def run(arms: dict[str, Detector], deepset: Sequence[dict[str, Any]],
        corpora: dict[str, Any] | None, clean: Sequence[str]) -> dict[str, Any]:
    started = time.perf_counter()
    blocks: dict[str, Any] = {"deepset": score(deepset, arms)}
    if corpora is not None:
        for name in ("development", "held_out"):
            rows = [{"text": r["text"], "label": 1 if r["kind"] == "injected" else 0}
                    for r in corpora[name]]
            blocks[f"garak_{name}"] = score(rows, arms)
    blocks["clean"] = score([{"text": t, "label": 0} for t in clean], arms)
    # Keep the per-item verdicts only for the sampled corpus; the others are summarised.
    for name, block in blocks.items():
        if name != "deepset":
            block.pop("verdicts", None)
    rows = [{"split": item["split"], "label": item["label"], **verdict}
            for item, verdict in zip(deepset, blocks["deepset"].pop("verdicts"), strict=True)]
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rival": {"model": MODEL, "licence": "Apache-2.0", "threshold": THRESHOLD,
                  "not_run": ["Meta Llama Prompt Guard 2 (gated licence)",
                              "Lakera Guard (paid API)", "Rebuff (archived; needs OpenAI)"]},
        "corpora": blocks,
        "deepset_rows": rows,
        "seconds": round(time.perf_counter() - started, 1),
        "scope_statement": (
            "Claimed: on the same items, how many attacks each detector withholds and how much "
            "clean text it wrongly withholds, on a sampled external corpus (deepset, direct "
            "injection, never used to write ARGUS's rules nor listed in the classifier's "
            "training data), garak's corpora, and real desk text. NOT claimed: that deepset is "
            "the desk's threat model; its attacks arrive as user prompts, where the desk's arrive "
            "inside evidence."),
    }


def main() -> int:  # pragma: no cover - CLI
    import sys

    from argus.eval import garak_quarantine as gq
    from argus.eval import quarantine_generalisation as qg

    if "--fetch" in sys.argv or not DEEPSET_PATH.exists():
        artefact.write(DEEPSET_PATH, {"source": "huggingface.co/datasets/deepset/prompt-injections",
                                      "licence": "Apache-2.0", "rows": fetch_deepset()})
    deepset = json.loads(DEEPSET_PATH.read_text(encoding="utf-8"))["rows"]
    corpora = gq.load_corpora() if gq.CORPUS_PATH.exists() else None
    theses = qg._paper_ledger_prose()
    readme, _repos = qg._readme_prose()
    report = run({"argus": argus_rules, "deberta": load_classifier()}, deepset, corpora,
                 [*theses, *readme])
    artefact.write(REPORT_PATH, report)
    for name, block in report["corpora"].items():
        print(name, {k: v for k, v in block.items() if k != "items"})
    print(f"saved -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["MODEL", "argus_rules", "exact_mcnemar", "run", "score"]
