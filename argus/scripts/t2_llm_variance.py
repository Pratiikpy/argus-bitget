r"""G7 -- LLM decision variance under identical inputs (21_T2_QUANT_SEARCH.md row G7).

Answers: for a real decision cycle run 2 already made, if the *identical* request (same system
and user prompt text, same model, same thinking tier, same temperature, max_tokens and seed) is
sent to Qwen again, how often does the answer change -- the action (act / no act), the side and
size of each target, and the stated confidence -- and would that change the paper-trading record
(some repeats trading while others do not)?

Run 2 is FROZEN (paper-traded on Bitget UTA Demo): this script never imports it as a running
service, never restarts anything, and places no order. It reads run 2's own ledger
(``var/ledger/paper.jsonl``) and blob store (``var/blobs/``) as plain files -- JSON and SHA-256
only, no pydantic, no ``sentiment_agent`` import -- so nothing here depends on run 2's virtualenv
or touches its process. The only network calls this script makes are the repeat Qwen calls
themselves, through ``argus.llm.qwen.QwenClient`` (the same client ARGUS's own agents use), which
writes one row per call to ``argus/data/qwen_cost_ledger.jsonl`` automatically.

Event selection, objective and stated (not hand-picked): among every ``DECISION`` ledger event
whose ``outcome`` is ``"decided"`` (a real answer was obtained; the one ``transport_error`` event
in the log is excluded because it produced no decision to compare against) and whose recorded
input is complete (``call.request_blob`` is present, and one of ``call.response_blobs`` is an
``application/vnd.t2sa.decision-attempt+json`` blob with ``verdict == "accepted"``, whose
``request`` field is the exact JSON payload the live client sent to Qwen for the accepted answer)
-- take the single most recent such event with ``decision.stance == "act"`` (at least one acted
case) and the two most recent such events with ``decision.stance != "act"`` (at least one
not-acted case, and one extra for breadth), sorted by ``decided_at`` descending.

Why 3 events x 2 repeats (6 new calls), not 5 x 5 (25 calls). The owner approved at most 30 calls
and about 150k tokens for this check. Run 2's decision prompt is a dense, portfolio-wide fact
sheet (DESIGN.md: "one token per prompt byte" of the daily budget) -- measured directly from the
five most recent decided events' own recorded usage, one call alone costs 18.7k-30.0k tokens
(mean ~23.5k), not the few thousand a generic chat call would cost. 25 such calls would cost
roughly 470k-590k tokens on that same measured basis: three to four times the approved token
spend, even though it is only 25 of the approved 30 calls. This design's three events (measured
original cost 79,258 tokens combined) at 2 repeats each is expected to cost about 159k tokens on
the same basis -- within about 6% of the ~150k estimate -- while every metric the task asks for
(action agreement, side, size, confidence spread, agreement with the original recording) is still
computed from a real repeated sample. ``--n-repeats`` and ``--max-events`` are exposed for a
future run with a larger approved budget.

Usage::

    python argus/scripts/t2_llm_variance.py --dry-run   # build and print the plan, no spend
    python argus/scripts/t2_llm_variance.py                      # spend the calls, write the report
    python argus/scripts/t2_llm_variance.py --out argus/data/t2_llm_variance.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]  # .../bitget
WORKSPACE_ROOT = REPO_ROOT.parent  # the parent that also holds the sibling checkouts below

# Sibling checkout, not part of this repo. Overridable by environment variable so this script has
# no hardcoded personal path baked in (same pattern as scripts/t2_stress_windows.py).
RUN2_ROOT = Path(os.environ.get("T2_RUN2_ROOT", WORKSPACE_ROOT / "t2-run2"))
LEDGER_PATH = RUN2_ROOT / "var" / "ledger" / "paper.jsonl"
BLOBS_ROOT = RUN2_ROOT / "var" / "blobs"

QWEN_ENV_PATH = REPO_ROOT / ".secrets" / "qwen.env"
"""The only key source for this script (project CLAUDE.md): never run2's own .secrets/qwen.env,
never .secrets/bitget.env. The key is read into a local variable and never printed or written."""

DEFAULT_OUT = REPO_ROOT / "argus" / "data" / "t2_llm_variance.json"
ARGUS_SRC = REPO_ROOT / "argus" / "src"

ATTEMPT_MEDIA_TYPE = "application/vnd.t2sa.decision-attempt+json"

DEFAULT_N_ACT = 1
DEFAULT_N_NO_ACT = 2
DEFAULT_N_REPEATS = 2

MAX_CALLS = 30
"""Hard ceiling: the owner approved at most 30 calls for this check. Enforced before every call."""
TOKEN_SAFETY_CEILING = 200_000
"""A coded backstop, not a guess: about 30% above the ~159k tokens this design (3 events x 2
repeats) is expected to cost on the measured basis in the module docstring, so ordinary per-call
variance in completion length cannot silently run the spend past the approved ~150k by much. If
crossed, remaining repeats are skipped and marked so in the report rather than sent."""


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise SystemExit(
            f"{what} not found at {path}. This script reads run 2's frozen checkout read-only; "
            "set T2_RUN2_ROOT if it lives somewhere other than a sibling of this repo."
        )
    return path


# ================================================================================================
# The Qwen key (project CLAUDE.md: loaded inside Python from .secrets/qwen.env, never printed)
# ================================================================================================

_ENV_LINE = re.compile(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)")


def _load_env_file(path: Path) -> dict[str, str]:
    """``NAME=value`` lines; ``#`` comments and matching quotes allowed. No value is logged."""
    _require(path, "the Qwen credentials file")
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENV_LINE.fullmatch(line)
        if match is None:
            continue
        name, value = match.group(1), match.group(2).strip()
        if value[:1] in {"'", '"'} and len(value) >= 2 and value[-1] == value[0]:
            value = value[1:-1]
        values[name] = value
    return values


def load_qwen_credentials(path: Path) -> tuple[str, str, str]:
    """``(api_key, base_url, model)`` from ``path``. Raises if the key is missing."""
    values = _load_env_file(path)
    api_key = values.get("BITGET_QWEN_API_KEY", "")
    if not api_key:
        raise SystemExit(f"BITGET_QWEN_API_KEY is missing or empty in {path}")
    base_url = values.get("BITGET_QWEN_BASE_URL") or "https://hackathon.bitgetops.com/v1"
    model = values.get("BITGET_QWEN_MODEL") or "qwen3.8-max"
    return api_key, base_url, model


# ================================================================================================
# Reading run 2's ledger and blob store (plain JSON; no sentiment_agent import)
# ================================================================================================


def _iter_decision_records(ledger_path: Path) -> list[dict[str, Any]]:
    """Every ``DECISION`` event's ``payload["record"]`` (a ``DecisionRecord``), in file order."""
    records: list[dict[str, Any]] = []
    with ledger_path.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("kind") == "decision":
                records.append(event["payload"]["record"])
    return records


def _blob(blobs_root: Path, sha256_hex: str) -> bytes:
    path = blobs_root / sha256_hex
    data = path.read_bytes()
    digest = sha256(data).hexdigest()
    if digest != sha256_hex:
        raise ValueError(f"blob {sha256_hex} does not hash to its name (got {digest})")
    return data


def _accepted_attempt(record: dict[str, Any], blobs_root: Path) -> dict[str, Any] | None:
    """The one attempt blob of ``record.call`` whose ``verdict`` is ``"accepted"``, or ``None``.

    This is the exact request the live client sent for the answer that was kept -- for a
    multi-attempt call (a truncation or a complaint-fed retry) this is *not* necessarily the first
    request (``call.request_blob``), which is why every accepted request is read from here.
    """
    accepted: dict[str, Any] | None = None
    for ref in record["call"]["response_blobs"]:
        if ref["media_type"] != ATTEMPT_MEDIA_TYPE:
            continue
        data = _blob(blobs_root, ref["sha256"])
        if len(data) != ref["size"]:
            raise ValueError(f"attempt blob {ref['sha256']} size does not match its reference")
        attempt = json.loads(data)
        if attempt.get("verdict") == "accepted":
            if accepted is not None:
                raise ValueError(f"{record['decision_id']}: more than one accepted attempt blob")
            accepted = attempt
    return accepted


def _complete_input(record: dict[str, Any], blobs_root: Path) -> dict[str, Any] | None:
    """The accepted attempt when this record's input is complete and replayable, else ``None``."""
    if record["outcome"] != "decided" or record.get("call", {}).get("request_blob") is None:
        return None
    try:
        return _accepted_attempt(record, blobs_root)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def select_events(
    records: list[dict[str, Any]],
    blobs_root: Path,
    *,
    n_act: int = DEFAULT_N_ACT,
    n_no_act: int = DEFAULT_N_NO_ACT,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """``[(record, accepted_attempt), ...]``, most recent first: see the module docstring's rule.

    ``n_act`` most recent complete ``stance == "act"`` events, plus ``n_no_act`` most recent
    complete ``stance != "act"`` events, de-duplicated and sorted by ``decided_at`` descending.
    """
    complete: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in records:
        attempt = _complete_input(record, blobs_root)
        if attempt is not None:
            complete.append((record, attempt))
    complete.sort(key=lambda pair: pair[0]["decided_at"], reverse=True)

    acts = [pair for pair in complete if pair[0]["decision"]["stance"] == "act"][:n_act]
    no_acts = [pair for pair in complete if pair[0]["decision"]["stance"] != "act"][:n_no_act]
    chosen: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for pair in (*acts, *no_acts):
        chosen[pair[0]["decision_id"]] = pair
    return sorted(chosen.values(), key=lambda pair: pair[0]["decided_at"], reverse=True)


# ================================================================================================
# The request payload, and the completion JSON, exactly as run 2 reads them
# ================================================================================================


def thinking_tier(payload: dict[str, Any]) -> str:
    """``"off"`` / ``"low"`` / ``"full"`` from a request payload, as run 2 spells them on the wire
    (``sentiment_agent/llm/client.py:_thinking_from_payload``)."""
    has_off = "enable_thinking" in payload
    has_low = "reasoning_effort" in payload
    if has_off and has_low:
        raise ValueError("a payload sets both enable_thinking and reasoning_effort")
    if has_off:
        return "off"
    if has_low:
        return "low"
    return "full"


def _strip_fences(text: str) -> str:
    """Adapted from run 2 ``decision/contract.py:_strip_fences`` (itself ported from ARGUS)."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else stripped
        stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()


def _extract_json_object(text: str) -> str:
    """Adapted from run 2 ``decision/contract.py:extract_json_object``: the outermost balanced
    ``{...}`` in ``text``, string-aware so a brace inside a thesis does not end the object."""
    stripped = _strip_fences(text)
    start = stripped.find("{")
    if start == -1:
        return stripped
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]
    return stripped


def parse_decision_content(content: str) -> dict[str, Any]:
    """``{"stance": ..., "targets": [{"symbol", "target", "confidence"}, ...]}`` from a completion.

    Deliberately lighter than run 2's ``decision/contract.py:parse_decision``: this script measures
    *variance*, not contract legality, and has no book/policy in hand to check a repeat against.
    Raises ``ValueError`` with the parse problem; the caller records it rather than failing the run.
    """
    text = _strip_fences(content)
    if not text:
        raise ValueError("empty answer")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        extracted = _extract_json_object(text)
        parsed = json.loads(extracted)  # let this raise if extraction also fails
    if not isinstance(parsed, dict):
        raise ValueError(f"answer is a JSON {type(parsed).__name__}, not an object")
    stance = parsed.get("stance")
    if not isinstance(stance, str):
        raise ValueError("answer has no string 'stance'")
    targets_raw = parsed.get("targets")
    if not isinstance(targets_raw, list):
        raise ValueError("answer has no 'targets' list")
    targets: list[dict[str, Any]] = []
    for entry in targets_raw:
        if not isinstance(entry, dict) or "symbol" not in entry or "target" not in entry:
            raise ValueError("a targets entry is missing 'symbol' or 'target'")
        targets.append(
            {
                "symbol": entry["symbol"],
                "target": float(entry["target"]),
                "confidence": float(entry["confidence"]) if "confidence" in entry else None,
            }
        )
    return {"stance": stance, "targets": targets}


def original_decision(record: dict[str, Any]) -> dict[str, Any]:
    """The same shape as :func:`parse_decision_content`, read from the ledger's own record."""
    decision = record["decision"]
    return {
        "stance": decision["stance"],
        "targets": [
            {
                "symbol": t["symbol"],
                "target": float(t["target"]),
                "confidence": float(t["confidence"]),
            }
            for t in decision["targets"]
        ],
    }


# ================================================================================================
# Calling Qwen (argus.llm.qwen.QwenClient -- auto-logs to argus/data/qwen_cost_ledger.jsonl)
# ================================================================================================


def _import_argus_llm() -> Any:
    if str(ARGUS_SRC) not in sys.path:
        sys.path.insert(0, str(ARGUS_SRC))
    import argus.llm.ledger as argus_ledger
    import argus.llm.qwen as argus_qwen

    return argus_qwen, argus_ledger


def build_client(api_key: str, base_url: str, model: str) -> Any:
    """A fresh, uncached ``QwenClient``: caching must be off, or repeats 2-5 of an identical,
    temperature-0, fixed-seed request would be served from the client's own cache rather than
    reaching the endpoint, which would make every repeat trivially identical by construction."""
    argus_qwen, _ = _import_argus_llm()
    return argus_qwen.QwenClient(
        api_key=api_key, base_url=base_url, model=model, cache=False, timeout=600.0
    )


def call_once(
    client: Any, request: dict[str, Any], tier: str, *, step_label: str
) -> tuple[dict[str, Any] | None, str | None]:
    """One live call reproducing ``request`` exactly. ``(result, error)``; exactly one is not
    ``None``. Logged to the Qwen cost ledger by the client itself under ``step_label``."""
    argus_qwen, argus_ledger = _import_argus_llm()
    tier_map = {
        "off": argus_qwen.Thinking.OFF,
        "low": argus_qwen.Thinking.LOW,
        "full": argus_qwen.Thinking.FULL,
    }
    started = time.perf_counter()
    try:
        with argus_ledger.cost_step(step_label):
            completion = client.complete(
                request["messages"],
                temperature=request["temperature"],
                max_tokens=request["max_tokens"],
                json_mode="response_format" in request,
                seed=request.get("seed"),
                thinking=tier_map[tier],
                stream=request.get("stream"),
            )
    except argus_qwen.QwenError as exc:
        return None, str(exc)
    latency_ms = int((time.perf_counter() - started) * 1000)
    result = {
        "content": completion.content,
        "reasoning": completion.reasoning,
        "finish_reason": completion.finish_reason,
        "usage": {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "reasoning_tokens": completion.usage.reasoning_tokens,
            "total_tokens": completion.usage.total_tokens,
            "cached_tokens": completion.usage.cached_tokens,
            "reported": completion.usage.reported,
        },
        "latency_ms": latency_ms,
    }
    return result, None


# ================================================================================================
# Metrics
# ================================================================================================


def _side(target: float) -> str:
    if target > 0:
        return "long"
    if target < 0:
        return "short"
    return "flat"


def event_metrics(
    original: dict[str, Any], repeat_decisions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Agreement of action, side and size across ``original`` and every successfully parsed
    repeat, plus the spread of confidence and size per symbol. ``repeat_decisions`` excludes
    failed or unparsed repeats (each is reported separately in the event's own record)."""
    observations = [
        ("original", original),
        *[(f"repeat{i + 1}", d) for i, d in enumerate(repeat_decisions)],
    ]
    stances = [name_and_dec[1]["stance"] for name_and_dec in observations]
    action_agreement = len(set(stances)) <= 1
    n_repeats = len(repeat_decisions)
    flips_vs_original = sum(1 for d in repeat_decisions if d["stance"] != original["stance"])

    symbols = sorted({t["symbol"] for _, dec in observations for t in dec["targets"]})
    per_symbol: dict[str, Any] = {}
    symbol_side_disagreement = False
    for symbol in symbols:
        sides: list[str] = []
        sizes: list[float] = []
        confidences: list[float] = []
        present_in: list[str] = []
        for name, dec in observations:
            match = next((t for t in dec["targets"] if t["symbol"] == symbol), None)
            if match is None:
                continue
            present_in.append(name)
            sides.append(_side(match["target"]))
            sizes.append(match["target"])
            if match["confidence"] is not None:
                confidences.append(match["confidence"])
        side_agreement = len(set(sides)) <= 1
        present_everywhere = len(present_in) == len(observations)
        if not side_agreement or not present_everywhere:
            symbol_side_disagreement = True
        per_symbol[symbol] = {
            "present_in": present_in,
            "present_in_all_observations": present_everywhere,
            "sides": sides,
            "side_agreement": side_agreement,
            "size_values": sizes,
            "size_range": (max(sizes) - min(sizes)) if sizes else None,
            "size_pstdev": statistics.pstdev(sizes) if len(sizes) > 1 else 0.0 if sizes else None,
            "confidence_values": confidences,
            "confidence_range": (max(confidences) - min(confidences)) if confidences else None,
            "confidence_pstdev": (statistics.pstdev(confidences) if len(confidences) > 1 else 0.0)
            if confidences
            else None,
        }

    any_disagreement = (not action_agreement) or symbol_side_disagreement
    return {
        "n_observations": len(observations),
        "n_repeats_parsed": n_repeats,
        "stances": {name: dec["stance"] for name, dec in observations},
        "action_agreement": action_agreement,
        "flips_vs_original": flips_vs_original,
        "per_symbol": per_symbol,
        "any_disagreement": any_disagreement,
        "metric_impact": any_disagreement,
        "metric_impact_reason": (
            "stance differs across observations (some would/would not place an order)"
            if not action_agreement
            else (
                "a symbol's side, presence or size differs across observations"
                if symbol_side_disagreement
                else "none: every observation agreed on action, side and size"
            )
        ),
    }


# ================================================================================================
# Orchestration
# ================================================================================================


def build_plan(
    n_act: int, n_no_act: int, n_repeats: int
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    _require(LEDGER_PATH, "run 2's paper ledger")
    _require(BLOBS_ROOT, "run 2's blob store")
    records = _iter_decision_records(LEDGER_PATH)
    chosen = select_events(records, BLOBS_ROOT, n_act=n_act, n_no_act=n_no_act)
    if not chosen:
        raise SystemExit("no complete decided DECISION events found in run 2's paper ledger")
    has_act = any(r["decision"]["stance"] == "act" for r, _ in chosen)
    has_no_act = any(r["decision"]["stance"] != "act" for r, _ in chosen)
    projected_tokens = 0
    for _record, attempt in chosen:
        original_total = attempt["completion"]["usage"].get("total_tokens", 0)
        projected_tokens += original_total * n_repeats
    plan_meta = {
        "n_events": len(chosen),
        "n_repeats_per_event": n_repeats,
        "n_new_calls": len(chosen) * n_repeats,
        "has_act_event": has_act,
        "has_no_act_event": has_no_act,
        "projected_tokens_measured_basis": projected_tokens,
        "decision_ids": [record["decision_id"] for record, _ in chosen],
    }
    return chosen, plan_meta


def run(
    *,
    n_act: int,
    n_no_act: int,
    n_repeats: int,
    out_path: Path,
    qwen_env_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    chosen, plan_meta = build_plan(n_act, n_no_act, n_repeats)

    print(
        f"[plan] {plan_meta['n_events']} events x {n_repeats} repeats = "
        f"{plan_meta['n_new_calls']} new calls; projected "
        f"~{plan_meta['projected_tokens_measured_basis']:,} "
        f"tokens on the original calls' own measured basis (safety ceiling "
        f"{TOKEN_SAFETY_CEILING:,}, hard call cap {MAX_CALLS})."
    )
    for record, attempt in chosen:
        tier = thinking_tier(attempt["request"])
        print(
            f"  - {record['decision_id']} {record['decided_at']} "
            f"stance={record['decision']['stance']!r} "
            f"thinking={tier} original_tokens={attempt['completion']['usage'].get('total_tokens')}"
        )

    if dry_run:
        return {"dry_run": True, "plan": plan_meta}

    api_key, base_url, model = load_qwen_credentials(qwen_env_path)
    client = build_client(api_key, base_url, model)

    calls_made = 0
    calls_failed = 0
    tokens_spent = 0
    prompt_tokens_spent = 0
    completion_tokens_spent = 0
    reasoning_tokens_spent = 0
    budget_stopped = False

    events_out: list[dict[str, Any]] = []
    for record, attempt in chosen:
        request = attempt["request"]
        tier = thinking_tier(request)
        original_dec = original_decision(record)
        original_completion = attempt["completion"]

        repeats_out: list[dict[str, Any]] = []
        repeat_decisions_for_metrics: list[dict[str, Any]] = []
        for repeat_index in range(1, n_repeats + 1):
            if budget_stopped:
                repeats_out.append({"repeat": repeat_index, "skipped": "budget_guard"})
                continue
            if calls_made >= MAX_CALLS:
                budget_stopped = True
                repeats_out.append({"repeat": repeat_index, "skipped": "max_calls_reached"})
                continue
            if tokens_spent >= TOKEN_SAFETY_CEILING:
                budget_stopped = True
                repeats_out.append({"repeat": repeat_index, "skipped": "token_ceiling_reached"})
                continue

            label = f"t2_llm_variance:{record['decision_id']}:repeat{repeat_index}"
            result, error = call_once(client, request, tier, step_label=label)
            calls_made += 1
            if error is not None:
                calls_failed += 1
                repeats_out.append({"repeat": repeat_index, "error": error})
                print(f"    [{record['decision_id']} repeat {repeat_index}] ERROR: {error}")
                continue

            usage = result["usage"]
            if usage.get("reported"):
                tokens_spent += usage["total_tokens"]
                prompt_tokens_spent += usage["prompt_tokens"]
                completion_tokens_spent += usage["completion_tokens"]
                reasoning_tokens_spent += usage["reasoning_tokens"]
            print(
                f"    [{record['decision_id']} repeat {repeat_index}] "
                f"total_tokens={usage.get('total_tokens')} "
                f"finish_reason={result['finish_reason']!r} "
                f"(running spend {tokens_spent:,} tokens, {calls_made} calls)"
            )

            entry: dict[str, Any] = {"repeat": repeat_index, "completion": result}
            try:
                parsed = parse_decision_content(result["content"])
            except ValueError as exc:
                entry["parse_error"] = str(exc)
            else:
                entry["parsed"] = parsed
                repeat_decisions_for_metrics.append(parsed)
            repeats_out.append(entry)

        metrics = event_metrics(original_dec, repeat_decisions_for_metrics)
        events_out.append(
            {
                "decision_id": record["decision_id"],
                "decided_at": record["decided_at"],
                "settings": {
                    "model": request["model"],
                    "thinking": tier,
                    "temperature": request["temperature"],
                    "max_tokens": request["max_tokens"],
                    "seed": request.get("seed"),
                    "json_mode": "response_format" in request,
                    "stream": request.get("stream", False),
                    "n_messages": len(request["messages"]),
                },
                "original": {
                    "decision": original_dec,
                    "completion": original_completion,
                },
                "repeats": repeats_out,
                "metrics": metrics,
            }
        )

    events_with_disagreement = [e for e in events_out if e["metrics"]["any_disagreement"]]
    total_flips = sum(e["metrics"]["flips_vs_original"] for e in events_out)
    total_repeats_parsed = sum(e["metrics"]["n_repeats_parsed"] for e in events_out)

    summary = {
        "n_events": len(events_out),
        "n_events_with_any_disagreement": len(events_with_disagreement),
        "share_events_with_any_disagreement": (
            round(len(events_with_disagreement) / len(events_out), 4) if events_out else None
        ),
        "events_with_metric_impact": [e["decision_id"] for e in events_with_disagreement],
        "total_repeats_attempted": sum(len(e["repeats"]) for e in events_out),
        "total_repeats_parsed": total_repeats_parsed,
        "total_flips_vs_original": total_flips,
        "flip_rate_vs_original": (
            round(total_flips / total_repeats_parsed, 4) if total_repeats_parsed else None
        ),
        "budget_stopped_early": budget_stopped,
    }

    spend = {
        "calls_made": calls_made,
        "calls_failed": calls_failed,
        "tokens_total": tokens_spent,
        "prompt_tokens": prompt_tokens_spent,
        "completion_tokens": completion_tokens_spent,
        "reasoning_tokens": reasoning_tokens_spent,
        "max_calls_allowed": MAX_CALLS,
        "token_safety_ceiling": TOKEN_SAFETY_CEILING,
    }

    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rule": {
            "source": "var/ledger/paper.jsonl in run 2's frozen paper-mode checkout",
            "event_selection": (
                "the most recent DECISION event with outcome=='decided' and decision.stance=="
                f"'act' (n_act={n_act}), plus the n_no_act={n_no_act} most recent DECISION "
                "events with "
                "outcome=='decided' and decision.stance!='act'; every candidate requires a "
                "complete recorded input (call.request_blob present, and an "
                "application/vnd.t2sa.decision-attempt+json response blob with "
                "verdict=='accepted' giving the exact request payload sent for the kept answer); "
                "sorted by decided_at descending."
            ),
            "repeat_rule": (
                f"each selected event's exact accepted request (identical messages, model, "
                f"thinking tier, temperature, max_tokens, seed) sent {n_repeats} more time(s) "
                "through a fresh, uncached Qwen client."
            ),
            "why_not_5x5": (
                "measured original cost of these decision prompts is 18.7k-30.0k tokens per "
                "call; 5 events x 5 repeats (25 calls) would cost roughly 470k-590k tokens on "
                "that basis, 3-4x the owner's ~150k-token approval, even though 25 calls is "
                "within the 30-call cap. This run's design is expected to cost about 159k tokens "
                "on the same measured basis -- within about 6% of the ~150k estimate."
            ),
        },
        "plan": plan_meta,
        "events": events_out,
        "summary": summary,
        "spend": spend,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--qwen-env", type=Path, default=QWEN_ENV_PATH)
    parser.add_argument("--n-act", type=int, default=DEFAULT_N_ACT)
    parser.add_argument("--n-no-act", type=int, default=DEFAULT_N_NO_ACT)
    parser.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS)
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan (events, settings) and spend nothing"
    )
    args = parser.parse_args(argv)

    if args.n_act * args.n_repeats + args.n_no_act * args.n_repeats > MAX_CALLS:
        raise SystemExit(
            f"(n_act + n_no_act) * n_repeats = "
            f"{(args.n_act + args.n_no_act) * args.n_repeats} exceeds the approved "
            f"{MAX_CALLS}-call cap; lower --n-repeats, --n-act or --n-no-act"
        )

    report = run(
        n_act=args.n_act,
        n_no_act=args.n_no_act,
        n_repeats=args.n_repeats,
        out_path=args.out,
        qwen_env_path=args.qwen_env,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(json.dumps(report, indent=2, default=str))
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    print(f"[done] wrote {args.out}")
    print(
        f"[spend] calls_made={report['spend']['calls_made']} "
        f"tokens_total={report['spend']['tokens_total']:,} "
        f"calls_failed={report['spend']['calls_failed']}"
    )
    print(
        f"[summary] events={report['summary']['n_events']} "
        f"with_disagreement={report['summary']['n_events_with_any_disagreement']} "
        f"flip_rate_vs_original={report['summary']['flip_rate_vs_original']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
