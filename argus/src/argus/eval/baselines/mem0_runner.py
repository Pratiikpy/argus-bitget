"""Runs real ``mem0ai`` (Apache-2.0) inside its own isolated venv, never inside ARGUS's own.

**Why this file exists and cannot be imported by ARGUS directly.** `argus/CLAUDE.md`'s setup rule
for `eval/memory_comparison.py` is explicit: mem0ai is installed only in an isolated venv at
`~/.venvs/mem0`, never in the environment that has `argus` installed. So this
process boundary is load-bearing, not a style choice — the same discipline
`eval/baselines/crypto_sor_loader.py` already uses for a different-language rival (a real Node
subprocess instead of a Python import), applied here to a same-language rival that cannot share a
Python environment with ARGUS for a policy reason rather than a language one.

This script is executed with `~/.venvs/mem0/Scripts/python.exe` by
``mem0_loader.run_chains`` (same package, the ARGUS-side caller). It reads one JSON request from
stdin, runs real `mem0.Memory` operations against it, and writes exactly one JSON response to
stdout — nothing else may reach stdout, so every mem0/HuggingFace/qdrant progress bar and warning
is redirected to stderr before `mem0` is imported.

**What real mem0 mechanism this exercises**, `mem0/memory/main.py` (installed copy, mem0ai 2.2.1,
matches `research/harvest/04-mem0.md`'s line numbers for the same functions within a few lines —
`_add_to_vector_store` at :881, the "V3 PHASED BATCH PIPELINE" comment at :918, confirmed by
reading the installed package, not assumed from the note): ``Memory.add(infer=True)`` runs the
real single-pass ADD-only extraction (one `generate_response` call per `add()`, confirmed by
`grep -c generate_response` over the function body — exactly one hit), the real hash-based dedup,
and the real entity extraction/linking. ``Memory.search()`` runs the real multi-signal retrieval
(`_search_vector_store`, semantic + BM25 + entity boost, `mem0/utils/scoring.py`). ``get_all()``
lists everything a user's store holds, used here to confirm mem0's real "never delete, both persist"
contradiction model rather than assuming it from the note.

**Backends, chosen by reading the installed package's own config code, not guessed:**
- Vector store: `mem0.vector_stores.qdrant.Qdrant` in **local embedded mode** (`path=<temp dir>`,
  no `host`/`port`/`url` given) — `mem0/vector_stores/qdrant.py:59-77` builds a bare
  `QdrantClient(path=path)` whenever no server params are supplied, which is qdrant-client's
  fully local, no-server, on-disk mode. `qdrant-client` is a hard dependency of the `mem0ai` wheel
  itself (`pip show mem0ai` → `Requires: ... qdrant-client ...`), so this needed no extra install —
  it is the only vector store mem0 supports out of the box without an additional package.
- Embedder: `mem0.embeddings.huggingface.HuggingFaceEmbedding`, local `sentence-transformers`
  (`mem0/embeddings/huggingface.py:26-29` — no `huggingface_base_url` set, so it loads
  `SentenceTransformer(model)` in-process; no network call per embed after the one-time model
  download). Model `multi-qa-MiniLM-L6-cos-v1` is mem0's own documented default for this provider
  (same file, line 26), 384 dimensions.
- LLM: `mem0.llms.openai.OpenAILLM` pointed at the Qwen hackathon endpoint via
  `OpenAIConfig(api_key=..., openai_base_url=...)` — `mem0/llms/openai.py:50-52` reads
  `self.config.api_key`/`self.config.openai_base_url` first, falling back to
  `OPENAI_API_KEY`/`OPENAI_BASE_URL` only if those are unset. This script passes them explicitly
  and never relies on the generic `OPENAI_API_KEY` fallback, so nothing here depends on — or sets
  — an ambient `OPENAI_API_KEY` in the environment.
- `fastembed` (BM25 sparse vectors) and `spacy`+`en_core_web_sm` (entity extraction, real mem0
  mechanism `main.py:1128-1210`) are installed in the isolated venv too, so mem0 runs with its real
  hybrid-retrieval and entity-linking machinery live, not a hobbled semantic-only fallback — the
  first smoke run without them logged "fastembed not installed - BM25 keyword search disabled" and
  "Failed to load spaCy ... model", which would have understated mem0's real recall/retrieval
  mechanism. Both are free, local, pip-installable packages; no paid API was added.

**Qwen key handling.** Loaded from `.secrets/qwen.env` at the workspace root — and ONLY that
file — straight into local variables, never left in `os.environ` under the generic `OPENAI_API_KEY`
name and never included in any JSON this script writes; :func:`_redact` scrubs the literal key
value out of any exception text before it can reach stdout. No other file under `.secrets/` is
opened.

**Concurrency model.** One `mem0.Memory` instance is expensive to construct (~15-30s: loading the
sentence-transformers model, spaCy, qdrant's local client) and cheap to reuse. `run_request` shards
the caller's independent "chains" (a chain is a same-user sequence of ops that must run in stated
order — e.g. state-a-fact, then ask, then ask-again) across a small thread pool; each worker builds
exactly ONE `Memory` instance, backed by its OWN temp qdrant directory (never shared across
threads — qdrant's embedded client is not documented as safe for concurrent multi-thread writers,
and `mem0/memory/main.py`'s own comment on `entity_store` warns of "RocksDB lock contention" when
two clients open the same local path at once), and runs its assigned chains sequentially. Each
chain still gets its own `user_id` even though a worker's chains share one qdrant collection,
because within one worker everything runs on a single thread — there is no concurrent access to
guard against there, only cross-chain data leakage, which `user_id` filtering already prevents
(the same scoping mechanism `research/harvest/04-mem0.md` §5 cites for mem0's three closed
cross-tenant issues, #4490/#6277/#6655).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Silence every non-JSON thing that could land on stdout before mem0/its deps are imported.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
# mem0's own telemetry path (`Memory.__init__`, gated on this) opens a SECOND qdrant client at a
# fixed, shared path (`~/.mem0/migrations_qdrant`) on every `Memory()` construction, regardless of
# this script's own per-worker temp directory. qdrant's embedded client takes an exclusive file
# lock on whatever path it opens, so two `Memory()` instances built concurrently on different
# worker threads — the whole point of this script's thread pool — collided on that ONE shared path
# and the second one failed with "already accessed by another instance of Qdrant client" (found by
# running the two-worker smoke test, not anticipated). Telemetry is also just not wanted here: this
# is a controlled local experiment against a hackathon-budget Qwen key, not a deployment that should
# phone home. `MEM0_TELEMETRY=false` is mem0's own documented off-switch
# (`mem0/memory/telemetry.py:14`), read before `mem0` is imported anywhere below.
os.environ["MEM0_TELEMETRY"] = "false"
warnings.filterwarnings("ignore")
for _name in ("mem0", "qdrant_client", "sentence_transformers", "transformers", "httpx", "openai",
              "posthog", "urllib3", "fastembed", "spacy"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)
logging.disable(logging.CRITICAL)

QWEN_ENV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 5, ".secrets", "qwen.env")
# src/argus/eval/baselines/ -> the workspace root; this file runs outside the argus package, so
# it cannot import argus.truth.paths.
_QWEN_KEYS = ("BITGET_QWEN_API_KEY", "BITGET_QWEN_BASE_URL", "BITGET_QWEN_MODEL")
MAX_QWEN_CALLS = 400
"""Hard cap enforced here too (belt), not just by the caller (braces): a malformed request that
somehow asked for more `add()` calls than the experiment's own budget must fail loudly rather than
spend past the hackathon key's limited balance."""

EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"
EMBEDDING_DIMS = 384


def _load_qwen_env() -> dict[str, str]:
    """The three named keys, and only those, read from the one named file, and only that file."""
    values: dict[str, str] = {}
    with open(QWEN_ENV_PATH, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key in _QWEN_KEYS:
                values[key] = value.strip()
    missing = [k for k in _QWEN_KEYS if k not in values or not values[k]]
    if missing:
        raise RuntimeError(f"qwen.env is missing required key(s): {missing}")
    for key, value in values.items():
        os.environ[key] = value
    return values


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "<redacted>") if secret else text


class _UsageLog:
    """One per worker (never shared across threads): captures token usage from mem0's OpenAI
    client via `response_callback` (`mem0/configs/llms/openai.py:35`), which mem0 calls
    synchronously inside `generate_response` right after the one real HTTP call `add()` makes."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any] | None] = []

    def callback(self, _llm: Any, response: Any, _params: dict[str, Any]) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            self.entries.append(None)
            return
        self.entries.append({
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        })

    def take_latest(self) -> dict[str, Any] | None:
        return self.entries[-1] if self.entries else None


def _build_memory(qdrant_path: str, collection: str, qwen: dict[str, str],
                  usage_log: _UsageLog) -> Any:
    # mem0 lives in its own venv (~/.venvs/mem0); this file runs there as a
    # subprocess, so the argus environment's type checker cannot resolve it.
    from mem0 import Memory  # type: ignore[import-not-found]
    from mem0.configs.base import MemoryConfig  # type: ignore[import-not-found]

    cfg = MemoryConfig(
        vector_store={
            "provider": "qdrant",
            "config": {
                "collection_name": collection,
                "embedding_model_dims": EMBEDDING_DIMS,
                "path": qdrant_path,
                "on_disk": False,
            },
        },
        embedder={
            "provider": "huggingface",
            "config": {"model": EMBEDDING_MODEL, "embedding_dims": EMBEDDING_DIMS},
        },
        llm={
            "provider": "openai",
            "config": {
                "model": qwen["BITGET_QWEN_MODEL"],
                "api_key": qwen["BITGET_QWEN_API_KEY"],
                "openai_base_url": qwen["BITGET_QWEN_BASE_URL"],
                "temperature": 0.0,
                "response_callback": usage_log.callback,
            },
        },
    )
    return Memory(cfg)


def _run_op(memory: Any, user_id: str, op: dict[str, Any], usage_log: _UsageLog,
            secret: str) -> dict[str, Any]:
    kind = op["op"]
    started = time.perf_counter()
    try:
        if kind == "add":
            before = len(usage_log.entries)
            result = memory.add(op["text"], user_id=user_id, infer=True)
            usage = usage_log.take_latest() if len(usage_log.entries) > before else None
            payload = {
                "memories": [
                    {"id": r.get("id"), "memory": r.get("memory"), "event": r.get("event")}
                    for r in result.get("results", [])
                ],
                "qwen_call": True,
                "usage": usage,
            }
        elif kind == "search":
            found = memory.search(
                op["query"], filters={"user_id": user_id}, top_k=int(op.get("top_k", 10)),
            )
            rows = sorted(found.get("results", []), key=lambda r: r.get("score") or 0.0,
                          reverse=True)
            payload = {
                "results": [
                    {"id": r.get("id"), "memory": r.get("memory"), "score": r.get("score"),
                     "created_at": r.get("created_at")}
                    for r in rows
                ],
                "qwen_call": False,
            }
        elif kind == "get_all":
            found = memory.get_all(filters={"user_id": user_id}, limit=int(op.get("limit", 200)))
            payload = {
                "results": [
                    {"id": r.get("id"), "memory": r.get("memory"),
                     "created_at": r.get("created_at")}
                    for r in found.get("results", [])
                ],
                "qwen_call": False,
            }
        else:
            raise ValueError(f"unknown op {kind!r}")
        payload["ok"] = True
    except Exception as exc:  # a per-op failure must not abort the whole chain/batch
        payload = {"ok": False, "error": _redact(f"{type(exc).__name__}: {exc}", secret),
                   "qwen_call": kind == "add"}
    payload["op"] = kind
    payload["elapsed_s"] = time.perf_counter() - started
    return payload


def _run_chain(chain: dict[str, Any], qwen: dict[str, str], secret: str) -> dict[str, Any]:
    tmp_dir = tempfile.mkdtemp(prefix="argus_mem0_")
    usage_log = _UsageLog()
    try:
        memory = _build_memory(tmp_dir, f"c_{abs(hash(chain['chain_id'])) % 10**8}", qwen,
                                usage_log)
        results = [_run_op(memory, chain["user_id"], op, usage_log, secret)
                   for op in chain["ops"]]
        return {"chain_id": chain["chain_id"], "ok": True, "results": results}
    except Exception as exc:
        return {"chain_id": chain["chain_id"], "ok": False,
                "error": _redact(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", secret),
                "results": []}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_request(request: dict[str, Any]) -> dict[str, Any]:
    qwen = _load_qwen_env()
    secret = qwen["BITGET_QWEN_API_KEY"]

    chains = request["chains"]
    planned_adds = sum(1 for c in chains for op in c["ops"] if op["op"] == "add")
    if planned_adds > MAX_QWEN_CALLS:
        raise RuntimeError(
            f"request plans {planned_adds} add() calls, over the {MAX_QWEN_CALLS} Qwen-call cap"
        )

    max_workers = max(1, min(int(request.get("max_workers", 6)), len(chains) or 1))
    chain_results: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()

    def worker(chain: dict[str, Any]) -> None:
        result = _run_chain(chain, qwen, secret)
        with lock:
            chain_results[chain["chain_id"]] = result

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(worker, chains))

    ordered = [chain_results[c["chain_id"]] for c in chains]
    total_qwen_calls = sum(
        1 for c in ordered for r in c["results"] if r.get("qwen_call") and r.get("ok")
    )
    total_tokens = sum(
        (r.get("usage") or {}).get("total_tokens") or 0
        for c in ordered for r in c["results"] if r.get("qwen_call") and r.get("usage")
    )
    return {
        "chains": ordered,
        "total_qwen_calls": total_qwen_calls,
        "total_usage_tokens": total_tokens,
        "mem0_version": _mem0_version(),
    }


def _mem0_version() -> str:
    try:
        import mem0
        return str(getattr(mem0, "__version__", "unknown"))
    except Exception:
        return "unknown"


def main() -> int:
    raw = sys.stdin.read()
    try:
        request = json.loads(raw)
        response = run_request(request)
    except Exception as exc:
        secret = ""
        with contextlib.suppress(Exception):
            secret = os.environ.get("BITGET_QWEN_API_KEY", "")
        response = {
            "fatal_error": _redact(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                                    secret),
        }
        sys.stdout.write(json.dumps(response))
        sys.stdout.flush()
        return 1
    sys.stdout.write(json.dumps(response))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
