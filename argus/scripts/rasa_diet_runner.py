"""Train and score Rasa's real DIET classifier on the frozen LUI rematch inputs, one seed per run.

Runs under a separate Python 3.10 environment holding ``rasa==3.6.21`` (Apache-2.0) — Rasa pins
TensorFlow 2.12 and does not install on the project's own Python 3.11 runtime, and a 1.9 GB
TensorFlow venv does not belong in ARGUS's dependencies. So this script lives outside the
``argus`` package, imports nothing from it, and talks to it only through two JSON files:

    <rasa-python> scripts/rasa_diet_runner.py --inputs data/lui_rematch_inputs.json \\
        --config rasa_default --seed 1 --workdir <scratch> --out <predictions.json>

**Rasa is run unmodified.** Two configurations, both chosen before any result was seen:

* ``rasa_default`` — Rasa's own shipped default pipeline, verbatim from
  ``rasa/engine/recipes/config_files/default_config.yml`` in 3.6.21 (WhitespaceTokenizer,
  RegexFeaturizer, LexicalSyntacticFeaturizer, word and ``char_wb`` 1-4 CountVectorsFeaturizers,
  ``DIETClassifier`` 100 epochs with ``constrain_similarities``, EntitySynonymMapper,
  ResponseSelector, and ``FallbackClassifier`` at 0.3 / 0.1). The FallbackClassifier is Rasa's
  own out-of-scope mechanism; its ``nlu_fallback`` output is scored as a decline.
* ``legacy_20260921`` — the exact pipeline `eval/baselines/rasa_diet_sealed_predictions.json` was
  produced with on 2026-09-21 (the default minus RegexFeaturizer, LexicalSyntacticFeaturizer,
  ResponseSelector and FallbackClassifier), kept so the earlier single-run figure can be placed
  inside this run's seed distribution rather than compared against a different model.

The only change to either is ``random_seed`` on ``DIETClassifier``, which Rasa ships as ``None``
(``diet_classifier.py:189``) — the reason the 2026-09-21 run cannot be reproduced exactly.

One Windows-only workaround is applied, carried over unchanged from the 2026-09-21 run: Rasa's
``LocalModelStorage._extract_archive_to_directory`` prefixes the extraction directory with the
``\\\\?\\`` long-path escape, and Python 3.10's ``tarfile`` then rejects the mixed-slash path with
WinError 123 when loading any trained model. The prefix is dropped; every path here is far below
MAX_PATH. It changes where bytes are unpacked, never the pipeline, the weights or a prediction.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

CONFIGS = {
    "rasa_default": """recipe: default.v1
assistant_id: argus_rematch
language: en
pipeline:
  - name: WhitespaceTokenizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: "char_wb"
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: 100
    constrain_similarities: true
    random_seed: {seed}
  - name: EntitySynonymMapper
  - name: ResponseSelector
    epochs: 100
    constrain_similarities: true
    random_seed: {seed}
  - name: FallbackClassifier
    threshold: 0.3
    ambiguity_threshold: 0.1
policies: []
""",
    "legacy_20260921": """recipe: default.v1
assistant_id: argus_rematch
language: en
pipeline:
  - name: WhitespaceTokenizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: char_wb
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: 100
    constrain_similarities: true
    random_seed: {seed}
policies: []
""",
}


def _patch_windows_longpath() -> None:
    """See the module docstring. A no-op off Windows."""
    if sys.platform != "win32":
        return
    from rasa.engine.storage.local_model_storage import LocalModelStorage
    from tarsafe import TarSafe

    def _extract(model_archive_path: Any, temporary_directory: Any) -> None:
        with TarSafe.open(model_archive_path, mode="r:gz") as tar:
            tar.extractall(str(temporary_directory))
        LocalModelStorage._assert_not_rasa2_archive(temporary_directory)

    LocalModelStorage._extract_archive_to_directory = staticmethod(_extract)
    original = shutil.rmtree
    shutil.rmtree = lambda p, *a, **k: original(str(p).replace("\\\\?\\", ""), *a, **k)


def _nlu_yaml(training: list[dict[str, str]]) -> str:
    by_label: dict[str, list[str]] = {}
    for row in training:
        by_label.setdefault(row["label"], []).append(row["text"])
    lines = ['version: "3.1"', "nlu:"]
    for label in sorted(by_label):
        lines += [f"- intent: {label}", "  examples: |"]
        lines += ["    - " + t.replace("\n", " ").strip() for t in by_label[label]]
    return "\n".join(lines) + "\n"


def _all_texts(inputs: dict[str, Any]) -> list[str]:
    texts: list[str] = [r["ask"] for r in inputs["sealed"]]
    texts += list(inputs["oos_probes_14"])
    texts += [r["ask"] for r in inputs["massive_oos"]]
    texts += [r["ask"] for r in inputs["clinc_oos"]]
    texts += [r["ask"] for r in inputs["hard_negatives"]]
    texts += [r["ask"] for r in inputs.get("confirmation_base", [])]
    for suite in ("perturbations", "confirmation_perturbations"):
        for rows in inputs.get(suite, {}).values():
            texts += [r["ask"] for r in rows]
    seen: set[str] = set()
    unique = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


async def _predict(agent: Any, texts: list[str]) -> tuple[dict[str, list[Any]], list[float]]:
    out: dict[str, list[Any]] = {}
    latencies: list[float] = []
    for text in texts:
        started = time.perf_counter()
        parsed = await agent.parse_message(text)
        latencies.append((time.perf_counter() - started) * 1000.0)
        intent = parsed.get("intent") or {}
        # [top name, top confidence, second name, second confidence]. With the FallbackClassifier
        # in the pipeline the top entry can be `nlu_fallback`, and the second is then DIET's own
        # best guess — kept so the fallback's contribution can be ablated without retraining.
        ranking = parsed.get("intent_ranking") or []
        second = ranking[1] if len(ranking) > 1 else {}
        out[text] = [intent.get("name"), round(float(intent.get("confidence") or 0.0), 6),
                     second.get("name"), round(float(second.get("confidence") or 0.0), 6)]
    return out, latencies


def main() -> int:
    parser = argparse.ArgumentParser(description="train + score Rasa DIET on the rematch inputs")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--training", type=Path, default=None,
                        help="optional JSON list of {text,label} replacing the inputs' training rows")
    args = parser.parse_args()

    process_started = time.perf_counter()
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    training = inputs["training"]
    if args.training is not None:
        training = json.loads(args.training.read_text(encoding="utf-8"))

    project = args.workdir / f"{args.config}_seed{args.seed}"
    if project.exists():
        shutil.rmtree(project)
    (project / "data").mkdir(parents=True)
    (project / "data" / "nlu.yml").write_text(_nlu_yaml(training), encoding="utf-8")
    config_text = CONFIGS[args.config].format(seed=args.seed)
    (project / "config.yml").write_text(config_text, encoding="utf-8")
    labels = sorted({r["label"] for r in training})
    (project / "domain.yml").write_text(
        'version: "3.1"\nintents:\n' + "".join(f"- {lab}\n" for lab in labels), encoding="utf-8",
    )

    import_started = time.perf_counter()
    import rasa
    from rasa.core.agent import Agent
    from rasa.model_training import train_nlu

    _patch_windows_longpath()
    import_seconds = time.perf_counter() - import_started

    train_started = time.perf_counter()
    model_path = train_nlu(
        config=str(project / "config.yml"), nlu_data=str(project / "data" / "nlu.yml"),
        output=str(project / "models"), fixed_model_name="model",
        domain=str(project / "domain.yml"),
    )
    train_seconds = time.perf_counter() - train_started
    if not model_path:
        raise SystemExit("rasa training produced no model")

    load_started = time.perf_counter()
    agent = Agent.load(model_path)
    load_seconds = time.perf_counter() - load_started

    texts = _all_texts(inputs)
    first_started = time.perf_counter()
    asyncio.run(agent.parse_message(texts[0]))
    first_seconds = time.perf_counter() - first_started
    predictions, latencies = asyncio.run(_predict(agent, texts))

    archive = Path(model_path)
    blob = {
        "rasa_version": rasa.__version__,
        "config_name": args.config,
        "config": config_text,
        "seed": args.seed,
        "training_rows": len(training),
        "training_digest": hashlib.sha256(
            json.dumps(training, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")).encode("utf-8")).hexdigest(),
        "inputs_digests": inputs.get("digests", {}),
        "predictions": predictions,
        "predictions_digest": hashlib.sha256(
            json.dumps(predictions, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "costs": {
            "import_seconds": round(import_seconds, 3),
            "train_seconds": round(train_seconds, 3),
            "model_load_seconds": round(load_seconds, 3),
            "first_prediction_seconds": round(first_seconds, 3),
            "process_to_ready_seconds": round(
                import_seconds + load_seconds + first_seconds, 3),
            "latency_ms_median": round(statistics.median(latencies), 4),
            "latency_ms_p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 4),
            "latency_ms_mean": round(statistics.fmean(latencies), 4),
            "model_archive_bytes": archive.stat().st_size,
            "predictions_timed": len(latencies),
        },
        "wall_seconds": round(time.perf_counter() - process_started, 3),
        "python": sys.version.split()[0],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{args.config} seed {args.seed}: trained {train_seconds:.1f}s, "
          f"{len(predictions)} predictions, median {blob['costs']['latency_ms_median']}ms "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
