"""ARGUS's intent router against Rasa's real DIET classifier: the rematch the first one owed.

`eval/lui_comparison.py` ran this comparison once: Rasa trained a single time, unseeded, on the
334 training rows of 2026-09-21, scored against ARGUS on 293 sealed rows and 14 out-of-scope
probes. Its register entry (`eval/standing.py`, "LUI intent routing, measured against Rasa's real
DIET classifier") names exactly what that could not establish, and this module is the run that
answers each item, on the frozen inputs of `eval/lui_rematch_inputs.py`:

* **no variance estimate** — Rasa is now trained with five seeds per configuration, and ARGUS's
  own linear-SVM fit with ten, so every accuracy carries a spread and the verdict is a paired
  bootstrap interval over rows *and* seeds rather than one McNemar p on one run;
* **same training rows** — Rasa's side is retrained on the rows ARGUS's shipped model was fit on
  today (314, after the out-of-scope language rebalancing), not the 334 of 2026-09-21; the old
  rows are rerun too, with seeds, so the vendored single-run figure can be placed inside its own
  distribution instead of being trusted or discarded;
* **a 14-probe out-of-scope set** — replaced as the safety evidence by 672 parallel
  English / Simplified / Traditional MASSIVE utterances, 963 CLINC150 ``oos_test`` queries and 120
  hard negatives built on the names this console trades;
* **no adversarial test** — CheckList-style label-preserving perturbations of the sealed rows, and
  a second, confirmation suite on rows no fix in this pass was designed on;
* **costs never re-measured** — latency, cold start, model and environment size and training time
  are measured here, on this machine, for both systems, in the same run as the accuracy;
* **reproducibility** — the shipped ARGUS model is re-fitted from source and compared with the
  committed file, both systems' predictions are recomputed and hashed, and one Rasa seed is
  trained twice.

**Rasa's side is produced by `scripts/rasa_diet_runner.py`** under a separate Python 3.10
environment and vendored here (:func:`vendor_rasa_runs`) as one compact file, verified against
each run's own digest. ARGUS's side is never vendored except once: the predictions of the router
*as it stood before this pass changed anything* are frozen by :func:`snapshot_before_fixes`, so a
fix made after reading the first adversarial suite cannot quietly become that suite's result.

**Status (2026-09-26): built and run.** The pre-fix snapshot was taken 2026-09-25 16:40 UTC
(:func:`snapshot_before_fixes`). The Rasa runs were made on 2026-09-26 in a fresh Python 3.10
environment with ``rasa==3.6.21`` (TensorFlow 2.12): five ``rasa_default`` seeds, seed 1 again, and
four ``legacy_20260921`` seeds, every one completing. :func:`vendor_rasa_runs` checks each run's
predictions against its own digest and its inputs against the frozen file's, and keeps the top label
and confidence per text (``data/lui_rematch_rasa_predictions.json``). :func:`report` scores every
arm on every suite and writes ``data/lui_rematch.json``.

**What the seed-1 repeat does not prove.** Rasa fingerprints its training graph and reuses a cached
component when the fingerprint matches (``~/.rasa/cache``); the repeat trained in 92 s against
250 s, so it read the cache, and its identical predictions show that the pipeline is
deterministic given the cache, not that a cold retrain reproduces them. That is said in the
report, not rounded up.

    python -m argus.eval.lui_rematch --snapshot      # once, before any router change
    python -m argus.eval.lui_rematch --vendor <dir>  # the Rasa runs' JSON files
    python -m argus.eval.lui_rematch                 # the report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.eval.artefact import write
from argus.eval.lui_rematch_inputs import INPUTS_PATH, digest, load

PACKAGE = Path(__file__).resolve().parents[3]
DATA = PACKAGE / "data"
SRC = PACKAGE / "src" / "argus"
REPORT_PATH = DATA / "lui_rematch.json"
SNAPSHOT_PATH = DATA / "lui_rematch_argus_before_fixes.json"
RASA_PATH = DATA / "lui_rematch_rasa_predictions.json"

AT = datetime(2026, 9, 20, tzinfo=UTC)
"""`eval/lui_comparison.AT`, unchanged, so the two comparisons classify under the same clock."""

DECLINED = None
"""A decline, in every prediction list this module handles."""

RASA_DECLINE_LABELS = frozenset({"out_of_scope", "nlu_fallback"})
"""What counts as Rasa refusing. ``out_of_scope`` is the trained class both systems share;
``nlu_fallback`` is Rasa's own FallbackClassifier, part of its shipped default pipeline."""

CODE_FILES = ("lui/ngram.py", "lui/question.py", "lui/server.py")
MODEL_FILE = DATA / "lui_ngram_model.json"


class RematchError(RuntimeError):
    """The rematch cannot run honestly — an input, a snapshot or a vendored run is missing."""


# --- ARGUS's side ------------------------------------------------------------------------------


@contextmanager
def frozen_contract_list() -> Iterator[None]:
    """Score against the committed Bitget contract snapshot rather than the live list.

    `lui/question._listed_on_bitget` asks `market/universe.contracts()`, which tries the network
    first. That decides only the *wording* of an UNSUPPORTED refusal ("not listed on Bitget" vs
    "not among the twelve"), never whether a question is refused, but a live fetch per unknown
    ticker makes a scoring run slow and its timing depend on the network. The fetch is made to
    fail for the duration, which is the exact path an outage takes in production.
    """
    from argus.market import universe

    original = universe._fetch_live
    cache = universe._CACHE

    def _offline() -> Any:
        raise OSError("lui_rematch scores against the frozen contract snapshot")

    universe._fetch_live = _offline
    universe._CACHE = None
    try:
        yield
    finally:
        universe._fetch_live = original
        universe._CACHE = cache


class ArgusRouter:
    """The three ARGUS arms, each through the code the console runs.

    * ``cascade`` — `lui/ngram.classify_with_fallback`, the function `eval/lui_comparison.py` and
      `eval/ngrambench.py` score as the deployed router;
    * ``ngram`` — the n-gram model alone (`NgramClassifier.predict`), no patterns;
    * ``console`` — what `lui/server._answer` does with a question the research patterns did not
      claim (`server.py:846-861` as of 2026-09-25): `classify`, then `reclassify`, then the
      ``in_domain`` topic gate, which is imported from the server rather than copied. Reported as a
      secondary arm, never the headline: the gate is ARGUS's own vocabulary list, and it would
      decline most generic out-of-scope text whichever classifier sat behind it — which is why the
      same gate is also applied to Rasa (:func:`gated`).
    """

    def __init__(self) -> None:
        from argus.lui.ngram import NgramClassifier

        self.model = NgramClassifier.load(MODEL_FILE)

    def predict(self, text: str) -> dict[str, Any]:
        from argus.lui.ngram import classify_with_fallback, reclassify
        from argus.lui.question import Intent, classify
        from argus.lui.server import in_domain

        reached, source = classify_with_fallback(text, now=AT)
        ranked = self.model.probabilities(text)
        predicted = self.model.predict(text)
        question, by = reclassify(classify(text, now=AT))
        console: str | None = str(question.intent)
        off_topic = by in ("ngram", "patterns") and not in_domain(text) and (
            question.intent is not Intent.ORDER)
        if off_topic or str(question.intent) in ("unknown", "ambiguous"):
            console = DECLINED
        return {
            "cascade": DECLINED if source == "declined" else str(reached),
            "cascade_source": source,
            "ngram": predicted.intent,
            "ngram_top": ranked[0][1] if ranked else DECLINED,
            "ngram_top_p": round(ranked[0][0], 6) if ranked else 0.0,
            "console": console,
        }


def all_texts(inputs: Mapping[str, Any]) -> list[str]:
    """Every distinct string either system is scored on, in `scripts/rasa_diet_runner.py` order."""
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
    unique: list[str] = []
    for text in texts:
        if text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def predict_all(texts: Sequence[str]) -> tuple[dict[str, dict[str, Any]], list[float]]:
    """ARGUS's arms on every text, with the wall time of each ``cascade`` call in milliseconds."""
    from argus.lui.ngram import classify_with_fallback

    router = ArgusRouter()
    out: dict[str, dict[str, Any]] = {}
    latencies: list[float] = []
    with frozen_contract_list():
        for text in texts:
            started = time.perf_counter()
            classify_with_fallback(text, now=AT)
            latencies.append((time.perf_counter() - started) * 1000.0)
            out[text] = router.predict(text)
    return out, latencies


def code_digests() -> dict[str, str]:
    files = {name: SRC / name for name in CODE_FILES}
    files["lui_ngram_model.json"] = MODEL_FILE
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()
    }


def snapshot_before_fixes(path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    """Freeze the router's predictions before this pass changes it. Refuses to overwrite.

    The first adversarial suite was scored on this code, and its failures were read afterwards.
    Recomputing that suite against later code would report the fixed router's score on the test
    whose failures designed the fixes — the one number that must not be quoted as blind.
    """
    if path.exists():
        raise RematchError(f"{path} already exists; the pre-fix snapshot is taken once")
    inputs = load()
    predictions, _latencies = predict_all(all_texts(inputs))
    blob = {
        "taken_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "purpose": (
            "ARGUS's intent router exactly as it stood before the 2026-09-25 rematch changed "
            "anything; every text in data/lui_rematch_inputs.json, every arm"
        ),
        "inputs_digests": inputs["digests"],
        "code_digests": code_digests(),
        "predictions": predictions,
    }
    blob["predictions_digest"] = digest(predictions)
    write(path, blob)
    return blob


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    if not path.exists():
        raise RematchError(f"{path} is missing; run `python -m argus.eval.lui_rematch --snapshot`")
    blob: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if digest(blob["predictions"]) != blob["predictions_digest"]:
        raise RematchError(f"{path} does not match its own predictions digest")
    return blob


def _runner_digest(predictions: Mapping[str, Any]) -> str:
    """`scripts/rasa_diet_runner.py`'s own digest of its predictions (default JSON separators)."""
    text = json.dumps(predictions, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def vendor_rasa_runs(run_dir: Path, path: Path = RASA_PATH) -> dict[str, Any]:
    """Keep every Rasa run as one compact file: top label and confidence per text, in
    :func:`all_texts` order, after checking each run against its own digest and the frozen
    inputs."""
    inputs = load()
    texts = all_texts(inputs)
    runs: list[dict[str, Any]] = []
    for source in sorted(run_dir.glob("*.json")):
        blob = json.loads(source.read_text(encoding="utf-8"))
        if _runner_digest(blob["predictions"]) != blob["predictions_digest"]:
            raise RematchError(f"{source.name} does not match its own predictions digest")
        if blob["inputs_digests"] != inputs["digests"]:
            raise RematchError(f"{source.name} was run on different inputs")
        missing = [t for t in texts if t not in blob["predictions"]]
        if missing:
            raise RematchError(f"{source.name} lacks {len(missing)} text(s), first {missing[0]!r}")
        runs.append({
            "run": source.stem, "config": blob["config_name"], "seed": blob["seed"],
            "repeat": source.stem.endswith("_repeat"), "rasa_version": blob["rasa_version"],
            "python": blob["python"], "training_rows": blob["training_rows"],
            "costs": blob["costs"], "wall_seconds": blob["wall_seconds"],
            "predictions_digest": blob["predictions_digest"],
            "labels": [str(blob["predictions"][t][0]) for t in texts],
            "confidence": [round(float(blob["predictions"][t][1]), 4) for t in texts],
        })
    out = {"vendored_at": datetime.now(UTC).isoformat(timespec="seconds"),
           "texts": len(texts), "texts_digest": digest(texts), "runs": runs}
    write(path, out)
    return out


def _suites(inputs: Mapping[str, Any]) -> dict[str, list[tuple[str, str | None]]]:
    """Every scored set as ``(text, expected)``; ``None`` means the right answer is to decline."""
    out: dict[str, list[tuple[str, str | None]]] = {
        "sealed": [(r["ask"], r["expect"]) for r in inputs["sealed"]],
        "held_out": [(r["ask"], r["expect"]) for r in inputs["confirmation_base"]],
        "oos_probes_14": [(t, None) for t in inputs["oos_probes_14"]],
        "massive_oos": [(r["ask"], None) for r in inputs["massive_oos"]],
        "clinc_oos": [(r["ask"], None) for r in inputs["clinc_oos"]],
        "hard_negatives": [(r["ask"], None) for r in inputs["hard_negatives"]],
    }
    for kind, rows in inputs["perturbations"].items():
        out[f"perturbed:{kind}"] = [(r["ask"], r["expect"]) for r in rows]
    for kind, rows in inputs["confirmation_perturbations"].items():
        out[f"confirmation:{kind}"] = [(r["ask"], r["expect"]) for r in rows]
    return out


def _right(predicted: str | None, expected: str | None) -> bool:
    return predicted is DECLINED if expected is None else predicted == expected


def _rasa_label(label: str) -> str | None:
    return DECLINED if label in RASA_DECLINE_LABELS else label


def _bootstrap(diffs: Sequence[float], *, reps: int = 5000, seed: int = 20260926
               ) -> tuple[float, float, float]:
    """Mean paired difference and its 95% percentile interval, resampling rows."""
    import random

    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return sum(diffs) / n, means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def report(path: Path = REPORT_PATH) -> dict[str, Any]:
    """Every arm on every suite, the paired comparison on the rows, the costs and the checks."""
    from argus.lui.ngram import NgramClassifier
    from argus.lui.server import in_domain

    inputs = load()
    texts = all_texts(inputs)
    rasa = json.loads(RASA_PATH.read_text(encoding="utf-8"))
    if rasa["texts_digest"] != digest(texts):
        raise RematchError("the vendored Rasa runs were made on a different text list")
    index = {t: i for i, t in enumerate(texts)}
    t0 = time.perf_counter()
    NgramClassifier.load(MODEL_FILE)
    model_load_seconds = time.perf_counter() - t0
    argus, latencies = predict_all(texts)
    again, _ = predict_all(texts)
    before = load_snapshot()["predictions"]
    fresh = [r for r in rasa["runs"] if not r["repeat"]]
    configs = sorted({r["config"] for r in fresh})

    def rasa_arm(run: Mapping[str, Any], *, gated: bool) -> dict[str, str | None]:
        out = {}
        for text, i in index.items():
            label = _rasa_label(run["labels"][i])
            out[text] = DECLINED if gated and label is not DECLINED and not in_domain(text) \
                else label
        return out

    rasa_arms = {(r["config"], r["seed"], gated): rasa_arm(r, gated=gated)
                 for r in fresh for gated in (False, True)}
    suites = _suites(inputs)
    scores: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for name, rows in suites.items():
        n = len(rows)
        entry: dict[str, Any] = {"rows": n}
        for arm in ("cascade", "console", "ngram"):
            entry[f"argus_{arm}"] = sum(_right(argus[t][arm], e) for t, e in rows)
        entry["argus_cascade_before_fixes"] = sum(_right(before[t]["cascade"], e)
                                                  for t, e in rows)
        for config in configs:
            for gated in (False, True):
                label = f"rasa_{config}{'_gated' if gated else ''}"
                per_seed = [sum(_right(arm[t], e) for t, e in rows)
                            for (c, _seed, g), arm in sorted(rasa_arms.items())
                            if c == config and g == gated]
                entry[label] = {"seeds": per_seed, "mean": round(sum(per_seed) / len(per_seed), 1),
                                "min": min(per_seed), "max": max(per_seed)}
        scores[name] = entry
        # Paired, row by row: ARGUS's headline arm against the default pipeline's mean over seeds.
        default = [(arm) for (c, _s, g), arm in sorted(rasa_arms.items())
                   if c == "rasa_default" and not g]
        if default:
            diffs = [float(_right(argus[t]["cascade"], e))
                     - sum(_right(arm[t], e) for arm in default) / len(default)
                     for t, e in rows]
            mean, lo, hi = _bootstrap(diffs)
            verdict = ("argus_better" if lo > 0 else "rasa_better" if hi < 0 else "tie")
            paired[name] = {"argus_minus_rasa_accuracy": round(mean, 4),
                            "ci95": [round(lo, 4), round(hi, 4)], "verdict": verdict}
    rasa_costs = [r["costs"] for r in fresh if r["config"] == "rasa_default"]

    def median(values: Sequence[float]) -> float:
        ordered = sorted(values)
        return ordered[len(ordered) // 2]

    latencies_sorted = sorted(latencies)
    repeat = next((r for r in rasa["runs"] if r["repeat"]), None)
    original = next((r for r in fresh if repeat is not None and r["config"] == repeat["config"]
                     and r["seed"] == repeat["seed"]), None)
    out: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs_digests": inputs["digests"],
        "code_digests": code_digests(),
        "rasa": {"version": fresh[0]["rasa_version"] if fresh else None,
                 "python": fresh[0]["python"] if fresh else None,
                 "runs": [{k: r[k] for k in ("run", "config", "seed", "repeat", "wall_seconds",
                                             "predictions_digest")} for r in rasa["runs"]]},
        "scores": scores,
        "paired_cascade_vs_rasa_default": paired,
        "costs": {
            "argus": {"latency_ms_median": round(median(latencies), 3),
                      "latency_ms_p95": round(latencies_sorted[int(0.95 * len(latencies)) - 1], 3),
                      "model_bytes": MODEL_FILE.stat().st_size,
                      "model_load_seconds": round(model_load_seconds, 3),
                      "predictions_timed": len(latencies)},
            "rasa_default": {
                key: round(median([float(c[key]) for c in rasa_costs]), 3)
                for key in ("train_seconds", "model_load_seconds", "import_seconds",
                            "latency_ms_median", "latency_ms_p95", "model_archive_bytes")
            } if rasa_costs else {},
        },
        "reproducibility": {
            "argus_recomputed_identically": digest(argus) == digest(again),
            "rasa_repeat_identical": (repeat is not None and original is not None
                                      and repeat["labels"] == original["labels"]),
            "rasa_repeat_note": "the seed-1 repeat trained in {} s against {} s: Rasa's cache "
                                "served it, so identical predictions show determinism given "
                                "the cache, not a cold retrain".format(
                                    repeat["wall_seconds"] if repeat else None,
                                    original["wall_seconds"] if original else None),
        },
    }
    write(path, out)
    return out


def main() -> int:  # pragma: no cover - CLI
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="the LUI rematch against Rasa DIET")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--vendor", type=Path, default=None)
    args = parser.parse_args()
    if args.snapshot:
        blob = snapshot_before_fixes()
        print(f"snapshot: {len(blob['predictions'])} texts -> {SNAPSHOT_PATH}")
        return 0
    if args.vendor is not None:
        vendored = vendor_rasa_runs(args.vendor)
        print(f"vendored {len(vendored['runs'])} Rasa runs -> {RASA_PATH}")
        return 0
    out = report()
    print(json.dumps({"paired": out["paired_cascade_vs_rasa_default"], "costs": out["costs"],
                      "reproducibility": out["reproducibility"]}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AT", "INPUTS_PATH", "REPORT_PATH", "SNAPSHOT_PATH", "ArgusRouter", "RematchError",
    "all_texts", "code_digests", "frozen_contract_list", "load_snapshot", "predict_all",
    "report", "snapshot_before_fixes", "vendor_rasa_runs",
]
