"""Exact-match prompt cache — one primitive for "this exact question was already asked".

**Why a primitive, when the Qwen client already caches.** :class:`~argus.llm.qwen.QwenClient` keeps
a per-instance dict of completions, which is right and stays. It has three limits this module
removes, each observed rather than supposed:

* **It dies with the process.** Replaying an evaluation re-bills every call the last run already
  paid for, so evaluations grew their own on-disk answer stores independently —
  `eval/thesis_quality.py` (a recording keyed by the payload hash), `eval/document_qa_eval.py` (its
  own ``_cache`` dict, rewritten whole as JSON after each answer), and `eval/infeasibilitybench.py`
  and `eval/research_depth.py` (write-through logs of paid calls) — each with its own format and
  its own write discipline. :class:`PromptCache` is one store any of them can use, appended to on
  every answer.
* **It belongs to one client.** A :class:`~argus.llm.provider.FallbackClient` holds one cache per
  provider, and any other :class:`~argus.llm.base.ChatModel` holds none. :class:`CachedModel` wraps
  any of them.
* **It ignored whether the request was deterministic.** It cached at any temperature, so a caller
  asking for a fresh sample twice would have been handed the first one twice. No caller does that
  today; the client now refuses to cache such a request, and so does this module.

**Taken from Tree of Thoughts** (``princeton-nlp/tree-of-thought-llm``, MIT): ``get_value``
(``src/tot/methods/bfs.py:6-14``) looks the exact prompt string up in ``task.value_cache``
(created at ``src/tot/tasks/game24.py:34``) before calling the model and stores the parsed value
after, so two search branches that reach the same textual state share one evaluation. Adapted,
changed in four ways:

1. **The key is the whole request, not the prompt string.** Messages *and* every parameter that can
   change the answer (temperature, token cap, thinking tier, JSON mode, tools, seed) and the model
   behind the wrapper (:attr:`CachedModel.namespace`). ToT's key is the prompt alone because its
   sampling parameters are fixed for a run; ARGUS's callers vary them — `complete_json` varies the
   seed per retry precisely so a retry is *not* served the answer that just failed.
2. **Only deterministic requests are cached** — temperature 0, or a fixed seed. ToT samples at
   temperature 0.7 and caches anyway, which in ARGUS would silently collapse deliberate sampling
   (``eval/consistency.py`` measures exactly that variance and turns the client cache off to do so).
3. **A cached structured answer is re-validated on the way out.** ``complete_json`` takes a
   validator, a callable that cannot be part of a key; a stored object that fails the caller's
   validator is treated as a miss rather than returned, so two callers with different standards
   cannot hand each other an answer only one of them accepts.
4. **Persistent and write-through**, one JSON line per answer, appended the moment it arrives. A
   previous evaluation lost paid calls to an end-of-run save; this one cannot.

**Rejected from Tree of Thoughts:** the in-batch duplicate guard (``bfs.py:18-24``), which scores
the second copy of an identical candidate 0 without asking. That is a *selection* rule — it biases a
beam away from duplicates — not a cache, and ARGUS's own ranking already settles a tie in favour of
the earlier answer (`agents/meta_pm.py`'s ``rank_key``), so a duplicate cannot win on score anyway.

**What it is for, measured rather than promised.** `eval/decision_primitives.py` replays the
recorded thesis-quality run through this cache and reports the share of its requests that were
exact repeats, and reads the live cost ledger for the same figure on real desk traffic. The live
desk rarely repeats itself — every decision prompt carries its own instant — so the value is in
replays, evaluations and repair loops, and the report says so in numbers. On 2026-09-26: within one
run of the recorded stream, 1 of 38 lookups was a repeat (2.6%); the same stream run again from the
persisted cache, as a new process would, hit 36 of 38 (94.7%; the two misses are the requests the
recording itself never answered), reusing 126,428 reported tokens; the live cost ledger shows 2
cache hits in 371 requests (0.5%). A replay is where the store pays; live decisions barely repeat.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argus.llm.base import ChatModel
from argus.llm.ledger import CallShape, CostLedger, Outcome, new_entry
from argus.llm.qwen import Completion, Thinking, Usage

CACHE_SCHEMA = 1
"""Written on every stored line; a line of another schema is skipped on load, never guessed at."""

_WRITE_LOCK = threading.Lock()


def request_key(kind: str, namespace: str, messages: list[dict[str, Any]],
                params: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical request: what is asked, of which model, with which parameters.

    Canonical means sorted keys and no whitespace, so the key depends on the values and not on how
    a dict happened to be built.
    """
    body = json.dumps(
        {"kind": kind, "namespace": namespace, "messages": messages, "params": dict(params)},
        sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def deterministic(*, temperature: float, seed: int | None) -> bool:
    """A request whose answer may be reused: greedy decoding, or a fixed seed.

    A seed is treated as a promise of reproducibility because that is what a caller passing one
    asks for; whether a hosted endpoint keeps the promise is the endpoint's business, and
    `eval/consistency.py` measures it.
    """
    return temperature == 0.0 or seed is not None


def completion_to_dict(completion: Completion) -> dict[str, Any]:
    usage = completion.usage
    return {
        "content": completion.content,
        "reasoning": completion.reasoning,
        "finish_reason": completion.finish_reason,
        "tool_calls": list(completion.tool_calls),
        "raw_id": completion.raw_id,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "total_tokens": usage.total_tokens,
            "cached_tokens": usage.cached_tokens,
            "reported": usage.reported,
        },
    }


def completion_from_dict(blob: Mapping[str, Any]) -> Completion:
    usage = dict(blob.get("usage") or {})
    return Completion(
        content=str(blob.get("content", "")),
        reasoning=str(blob.get("reasoning", "")),
        usage=Usage(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            reasoning_tokens=int(usage.get("reasoning_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            cached_tokens=int(usage.get("cached_tokens", 0)),
            reported=bool(usage.get("reported", False)),
        ),
        finish_reason=str(blob.get("finish_reason", "")),
        tool_calls=list(blob.get("tool_calls") or []),
        raw_id=str(blob.get("raw_id", "")),
    )


@dataclass
class CacheStats:
    """What the cache did. A hit rate over requests it could not have served would flatter it, so
    bypassed requests are counted apart from lookups."""

    hits: int = 0
    misses: int = 0
    bypassed: int = 0
    """Requests that were not deterministic and went straight to the model."""

    revalidation_misses: int = 0
    """Stored structured answers the caller's validator refused; each was re-asked."""

    tokens_saved: int = 0
    """Tokens the model reported for the answers that hits reused — what a re-ask would have cost
    by the only measure on record. Hits whose stored answer carried no usage block add nothing
    here and are counted in :attr:`unmeasured_hits`, never estimated."""

    unmeasured_hits: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float | None:
        """Hits over lookups; ``None`` before the first lookup, which is unknown rather than 0."""
        return self.hits / self.lookups if self.lookups else None

    def as_dict(self) -> dict[str, Any]:
        rate = self.hit_rate
        return {
            "lookups": self.lookups, "hits": self.hits, "misses": self.misses,
            "bypassed": self.bypassed, "revalidation_misses": self.revalidation_misses,
            "hit_rate": None if rate is None else round(rate, 4),
            "tokens_saved": self.tokens_saved, "unmeasured_hits": self.unmeasured_hits,
        }


class PromptCache:
    """Request key → stored answer, in memory and, when ``path`` is set, appended to disk.

    Loading skips a torn last line (a killed process) and any line of another schema, and counts
    them in :attr:`skipped_lines`; neither is fatal and neither is repaired by guessing.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._store: dict[str, dict[str, Any]] = {}
        self.stats = CacheStats()
        self.skipped_lines = 0
        self.write_failures = 0
        if path is not None and path.exists():
            self._load(path)

    def _load(self, path: Path) -> None:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                self.skipped_lines += 1
                continue
            if (not isinstance(row, dict) or row.get("schema") != CACHE_SCHEMA
                    or not isinstance(row.get("key"), str)
                    or not isinstance(row.get("value"), dict)):
                self.skipped_lines += 1
                continue
            self._store[row["key"]] = row["value"]

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def get(self, key: str) -> dict[str, Any] | None:
        return self._store.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Keep the answer and, if persistent, append it now. A failed write never raises into the
        model call — the answer is still returned and kept in memory — and is counted."""
        self._store[key] = value
        if self.path is None:
            return
        line = json.dumps({"schema": CACHE_SCHEMA, "key": key, "value": value},
                          ensure_ascii=False, allow_nan=False, default=str) + "\n"
        try:
            with _WRITE_LOCK:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
        except (OSError, ValueError):
            self.write_failures += 1

    def count_hit(self, stored: Mapping[str, Any], *, namespace: str, thinking: Thinking,
                   stream: bool, json_mode: bool, tools: bool, ledger: CostLedger | None) -> None:
        self.stats.hits += 1
        usage = dict(stored.get("usage") or {})
        if usage.get("reported") and int(usage.get("total_tokens", 0)) > 0:
            self.stats.tokens_saved += int(usage["total_tokens"])
        else:
            self.stats.unmeasured_hits += 1
        if ledger is not None:
            ledger.record(new_entry(outcome=Outcome.CACHE_HIT, shape=CallShape(
                model=namespace, host="prompt-cache", thinking=str(thinking),
                stream=stream, json_mode=json_mode, tools=tools,
            )))

    def complete(
        self,
        ask: Callable[[], Completion],
        messages: list[dict[str, Any]],
        *,
        namespace: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        tools: list[dict[str, Any]] | None,
        seed: int | None,
        thinking: Thinking,
        stream: bool | None,
        ledger: CostLedger | None = None,
    ) -> Completion:
        """Serve a completion from the store, or call ``ask`` and store what it returns.

        ``ask`` must make exactly the request described by the other arguments; they are what the
        key is built from. Kept separate from :class:`CachedModel` so a client subclass can route
        its own ``complete`` through the same store without being wrapped.
        """
        if not deterministic(temperature=temperature, seed=seed):
            self.stats.bypassed += 1
            return ask()
        key = request_key("complete", namespace, messages, {
            "temperature": temperature, "max_tokens": max_tokens, "json_mode": json_mode,
            "tools": tools, "seed": seed, "thinking": str(thinking), "stream": stream,
        })
        stored = self.get(key)
        if stored is not None:
            self.count_hit(stored, namespace=namespace, thinking=thinking, stream=bool(stream),
                            json_mode=json_mode, tools=bool(tools), ledger=ledger)
            return completion_from_dict(stored)
        self.stats.misses += 1
        result = ask()
        self.put(key, completion_to_dict(result))
        return result


def default_namespace(model: object) -> str:
    """The model behind a wrapper, as far as it says: class, model id and host.

    Two different models must never share an answer, so a model that cannot name itself gets its
    object identity — which never matches across processes, so a persistent cache simply never
    hits for it. That fails towards a paid re-ask, never towards a wrong answer.
    """
    name = str(getattr(model, "_model", "") or "")
    host = str(getattr(model, "_base_url", "") or "")
    if name:
        return f"{type(model).__name__}:{name}@{host}"
    return f"{type(model).__name__}#{id(model)}"


class CachedModel:
    """A :class:`~argus.llm.base.ChatModel` that asks its :class:`PromptCache` before its model."""

    def __init__(self, inner: ChatModel, cache: PromptCache | None = None, *,
                 namespace: str | None = None, ledger: CostLedger | None = None) -> None:
        self.inner = inner
        self.cache = cache if cache is not None else PromptCache()
        self.namespace = namespace if namespace is not None else default_namespace(inner)
        self.ledger = ledger
        """When set, every hit is recorded as a ``cache_hit`` line, so the cost ledger's "Role of
        the LLM" summary counts repeats this wrapper absorbed."""

    @property
    def budget(self) -> Any:
        return self.inner.budget

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
        seed: int | None = None,
        thinking: Thinking = Thinking.LOW,
        stream: bool | None = None,
    ) -> Completion:
        def ask() -> Completion:
            return self.inner.complete(
                messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
                tools=tools, seed=seed, thinking=thinking, stream=stream,
            )

        return self.cache.complete(
            ask, messages, namespace=self.namespace, temperature=temperature,
            max_tokens=max_tokens, json_mode=json_mode, tools=tools, seed=seed,
            thinking=thinking, stream=stream, ledger=self.ledger,
        )

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        *,
        required_keys: tuple[str, ...] = (),
        validate: Callable[[dict[str, Any]], str] | None = None,
        max_tokens: int = 2048,
        attempts: int = 3,
        thinking: Thinking = Thinking.LOW,
    ) -> dict[str, Any]:
        """Structured answers, cached as the parsed object and re-validated when reused.

        ``complete_json`` decodes greedily on the underlying client (it never sets a temperature),
        so every such request is deterministic by construction and cacheable.
        """
        key = request_key("complete_json", self.namespace, messages, {
            "required_keys": list(required_keys), "max_tokens": max_tokens,
            "attempts": attempts, "thinking": str(thinking),
        })
        stored = self.cache.get(key)
        if stored is not None:
            obj = stored.get("object")
            acceptable = isinstance(obj, dict) and all(k in obj for k in required_keys) and not (
                validate(obj) if validate is not None else ""
            )
            if acceptable and isinstance(obj, dict):
                self.cache.count_hit(stored, namespace=self.namespace, thinking=thinking,
                                      stream=False, json_mode=True, tools=False,
                                      ledger=self.ledger)
                return dict(obj)
            self.cache.stats.revalidation_misses += 1
        self.cache.stats.misses += 1
        result = self.inner.complete_json(
            messages, required_keys=required_keys, validate=validate, max_tokens=max_tokens,
            attempts=attempts, thinking=thinking,
        )
        self.cache.put(key, {"object": result})
        return result


__all__ = [
    "CACHE_SCHEMA",
    "CacheStats",
    "CachedModel",
    "PromptCache",
    "completion_from_dict",
    "completion_to_dict",
    "default_namespace",
    "deterministic",
    "request_key",
]
