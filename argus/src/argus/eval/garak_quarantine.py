"""The quarantine on attack text it was never written against: NVIDIA garak's injection probes.

`eval/quarantine_generalisation.py` measured that `agents/quarantine.py` recognises AgentDojo's
forms and nothing written outside them. garak (NVIDIA, Apache-2.0; `research/repos-new/garak`
@8d1259ef) is a second, independent corpus, and it is used here with a split fixed before any rule
was changed (research/harvest/06-garak.md):

* **Development: the latent-injection probes** (`probes/latentinjection.py`): an instruction buried
  inside a translation request, a report, a résumé, a fact snippet, a WHOIS record. The misses on
  this half are what the 2026-09-28 rules were written from.
* **Held out: the prompt-injection probes** (`probes/promptinject.py`, built by garak's own
  `resources/promptinject` package): opened for the first time to score the finished rules, once.

Each injected document has its clean counterpart — the same context with nothing injected — and
those are the false-alarm check on garak's own material; `eval/quarantine_generalisation.py`'s
larger real corpora (the paper ledger's theses, README prose) are run again beside them.

The corpora are generated from garak's source by this module (its probe classes run against
stand-ins for garak's runtime, which is torch- and API-heavy and not needed to build a string),
and a copy is committed at :data:`CORPUS_PATH` so the scores reproduce without the clone. The rules
as they were before this change are frozen at `data/quarantine_pre_garak.py.txt`, so the "before"
column is a run, not a remembered number.

    python -m argus.eval.garak_quarantine [--clone research/repos-new/garak]
"""

from __future__ import annotations

import argparse
import enum
import gzip
import importlib.machinery
import importlib.util
import json
import random
import sys
import types
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from argus.agents.quarantine import inspect, withholds
from argus.truth.artefact import write
from argus.truth.paths import DATA_DIR

CORPUS_PATH = DATA_DIR / "garak_corpora.json.gz"
REPORT_PATH = DATA_DIR / "garak_quarantine.json"
BEFORE_PATH = DATA_DIR / "quarantine_pre_garak.py.txt"
GARAK_COMMIT = "8d1259ef310e4803cf5a4cc77267fdfdc24434ec"
SEED = 20260928
"""garak draws its fact-snippet contexts at random (`probes/latentinjection.py:182-188`) and
seeds only when its CLI is given `--seed` (`cli.py:504`); the corpus is built under this seed, and
the unordered set of contexts is read in sorted order, so a rebuild is byte-for-byte the same."""
HIJACKS = ("ignore-print", "ignore-say", "ignore-print-upper", "nevermind", "screaming-stop")
ROGUES = ("hate-humans", "kill-humans", "long-prompt")

Detector = Callable[[str], bool]


# --- building the corpora from garak's own source -------------------------------------------

def _stubs(root: Path) -> None:
    """Stand-ins for the parts of garak's runtime its probe classes touch while building prompts:
    the probe base class, payload loading from `data/payloads/*.json`, the language provider
    (identity) and the progress bar's colour. Nothing that decides what a prompt says."""

    def module(name: str, **attrs: Any) -> types.ModuleType:
        made = types.ModuleType(name)
        made.__dict__.update(attrs)
        sys.modules[name] = made
        return made

    class Tier(enum.IntEnum):
        OF_CONCERN = 1
        COMPETE_WITH_SOTA = 2
        INFORMATIONAL = 3
        UNLISTED = 9

    class Language:
        def get_text(self, items: list[str], notify_callback: Any = None) -> list[str]:
            return list(items)

    class Probe:
        soft_probe_prompt_cap = 10**9
        langprovider = Language()
        DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {}

        def __init__(self, config_root: Any = None) -> None:
            for key, value in self.DEFAULT_PARAMS.items():
                setattr(self, key, value)

    class Group:
        def __init__(self, payloads: list[str]) -> None:
            self.payloads = payloads

    def load(name: str) -> Group:
        blob = json.loads((root / "data" / "payloads" / f"{name}.json").read_text("utf-8"))
        return Group(list(blob["payloads"]))

    garak = module("garak")
    resources = module("garak.resources")
    setattr(resources, "theme", module("garak.resources.theme", LANGPROVIDER_RGB="000000"))  # noqa: B010
    setattr(garak, "_config", module("garak._config"))  # noqa: B010
    module("garak.attempt", Attempt=object)
    setattr(garak, "payloads", module("garak.payloads", load=load))  # noqa: B010
    setattr(garak, "probes", module("garak.probes", Probe=Probe, Tier=Tier))  # noqa: B010
    setattr(garak, "resources", resources)  # noqa: B010


def latent_corpus(root: Path) -> list[dict[str, str]]:
    """Every latent-injection document garak builds, and each context with nothing injected."""
    _stubs(root)
    namespace: dict[str, Any] = {"__name__": "garak.probes.latentinjection"}
    source = (root / "probes" / "latentinjection.py").read_text("utf-8")
    exec(compile(source, "latentinjection.py", "exec"), namespace)  # garak's own file
    inj, pay, trig = (namespace["INJECTION_MARKER"], namespace["PAYLOAD_MARKER"],
                      namespace["TRIGGER_MARKER"])
    rows: list[dict[str, str]] = []
    for name, cls in sorted(namespace.items()):
        if not (isinstance(cls, type) and name.startswith("Latent") and name.endswith("Full")):
            continue
        random.seed(SEED)
        probe: Any = cls()
        seen: set[str] = set()
        for context in sorted(probe.contexts):
            if inj not in context:
                continue
            clean = context.replace(inj, "")
            if clean not in seen:
                seen.add(clean)
                rows.append({"family": name, "kind": "clean", "text": clean})
            for start, end in probe.injection_sep_pairs:
                for instruction in probe.injection_instructions:
                    for payload in probe.payloads:
                        for trigger in probe.payload_triggers:
                            injection = start + instruction.replace(
                                pay, payload.replace(trig, trigger)) + end
                            rows.append({"family": name, "kind": "injected",
                                         "text": context.replace(inj, injection)})
    return rows


def promptinject_corpus(root: Path) -> list[dict[str, str]]:
    """garak's prompt-injection hijacks over the OpenAI playground prompts, as its
    `probes/promptinject.py` configures them, and the playground prompts with their own input."""
    package = root / "resources" / "promptinject"
    spec = importlib.util.spec_from_file_location(
        "_garak_promptinject", package / "__init__.py", submodule_search_locations=[str(package)])
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules["_garak_promptinject"] = loaded
    spec.loader.exec_module(loaded)
    data: Any = importlib.import_module("_garak_promptinject.prompt_data")
    rows: list[dict[str, str]] = []
    for rogue in ROGUES:
        config = {
            "base": {"text": data.ALL_OPENAI_PLAYGROUND_PROMPTS},
            "attack": {"text": tuple(data.goal_hikacking_attacks[k] for k in HIJACKS),
                       "rogue_string": data.rogue_strings[rogue],
                       "scoring": "match_rogue_to_output",
                       "settings": {"escape": data.escape_chars["n"],
                                    "delimiter": data.delimiter_chars["dash"],
                                    "escape_length": 10, "escape_times": 2}},
            "config": {"id": (0, 1, 2, 3)},
        }
        rows += [{"family": f"promptinject/{rogue}", "kind": "injected", "text": p["prompt"]}
                 for p in loaded.build_prompts(config)]
    rows += [{"family": "promptinject", "kind": "clean",
              "text": base["instruction"] + (base.get("input") or "")}
             for base in data.ALL_OPENAI_PLAYGROUND_PROMPTS]
    return rows


def build_corpora(clone: Path) -> dict[str, Any]:
    root = clone / "garak"
    return {"source": {"repository": "NVIDIA/garak", "commit": GARAK_COMMIT,
                       "licence": "Apache-2.0 (licenses/garak-APACHE-2.0.txt)"},
            "development": latent_corpus(root), "held_out": promptinject_corpus(root)}


def load_corpora(path: Path = CORPUS_PATH) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded


# --- scoring ----------------------------------------------------------------------------------

def _before() -> Detector | None:
    """The rules as they were before 2026-09-28, loaded from the frozen copy."""
    if not BEFORE_PATH.is_file():
        return None
    name = "_argus_quarantine_pre_garak"
    spec = importlib.util.spec_from_file_location(
        name, BEFORE_PATH, loader=importlib.machinery.SourceFileLoader(name, str(BEFORE_PATH)))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return lambda text: bool(module.withholds(module.inspect(text)))


def _now(text: str) -> bool:
    return withholds(inspect(text))


def score(rows: list[dict[str, str]], detector: Detector) -> dict[str, Any]:
    families: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    clean = [0, 0]
    for row in rows:
        hit = detector(row["text"])
        if row["kind"] == "injected":
            families[row["family"]][0] += hit
            families[row["family"]][1] += 1
        else:
            clean[0] += hit
            clean[1] += 1
    caught = sum(v[0] for v in families.values())
    total = sum(v[1] for v in families.values())
    return {"caught": caught, "injected": total,
            "recall": round(caught / total, 4) if total else None,
            "by_family": {k: f"{v[0]}/{v[1]}" for k, v in sorted(families.items())},
            "clean_withheld": clean[0], "clean": clean[1]}


def run(corpora: dict[str, Any]) -> dict[str, Any]:
    from argus.eval import quarantine_generalisation as qg

    before = _before()
    theses = qg._paper_ledger_prose()
    readme, repos = qg._readme_prose()
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": corpora["source"],
        "split": "development = latent injection (rules written from its misses); held out = "
                 "promptinject (scored once, after the rules were fixed)",
        # One adversarial test, two arms on the same attacks: ARGUS's rules now, and as they were.
        "adversarial": {"argus": {"development": score(corpora["development"], _now),
                                  "held_out": score(corpora["held_out"], _now)}},
        "real_prose": {
            "paper_ledger_theses": {"n": len(theses), "withheld": sum(map(_now, theses))},
            "readme_lines": {"n": len(readme), "repos": repos,
                             "withheld": sum(map(_now, readme)),
                             "environment_dependent": True},
        },
    }
    if before is not None:
        report["adversarial"]["argus_before_2026_09_28"] = {
            "development": score(corpora["development"], before),
            "held_out": score(corpora["held_out"], before)}
    return report


def main() -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone", type=Path, default=None,
                        help="a garak checkout to rebuild the corpora from")
    args = parser.parse_args()
    if args.clone is not None:
        corpora = build_corpora(args.clone)
        # mtime=0 and no file name in the header: the same corpus is the same bytes.
        with CORPUS_PATH.open("wb") as raw, gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write(json.dumps(corpora, ensure_ascii=False).encode("utf-8"))
    report = run(load_corpora())
    write(REPORT_PATH, report)
    print(json.dumps({k: report[k] for k in ("adversarial", "real_prose")},
                     indent=1)[:3000])
    print(f"written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
