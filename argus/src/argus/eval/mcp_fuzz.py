"""Fuzzing the MCP server's wire: malformed JSON, broken envelopes, hostile names and arguments.

The synthesis (`research/mypr-teardowns/_SYNTHESIS.md`, row S15) sets the bar for the MCP surface:
*malformed params, oversize names and unknown tools — zero unstructured exceptions reach the
wire.* That is a claim about every body a client can send, so it is measured over a corpus built
to hit each place a hand-written JSON-RPC layer can go wrong, and judged by an oracle that does not
share code with the server it judges.

**The corpus** (:func:`corpus`), by family:

* ``parse`` — bytes that are not JSON, not UTF-8, nested past the recursion limit, or JSON that
  Python's parser accepts and the JSON grammar does not (``NaN``, ``1e999`` → ``inf``).
* ``envelope`` — valid JSON that is not a JSON-RPC 2.0 request: wrong ``jsonrpc``, a method that is
  not a string, an ``id`` that is a boolean, object or array, ``params`` that is not an object.
* ``batch`` — the empty batch (JSON-RPC 2.0 §6: one Invalid Request error, not silence), batches
  of junk, nested batches, and a batch large enough to be a denial of service.
* ``call`` — ``tools/call`` with the name missing, not a string, a megabyte long, unknown, or
  carrying control characters; ``arguments`` that is not an object.
* ``args`` — every tool, with each argument given the wrong type, a non-finite number, an empty or
  oversize value, a required argument missing, and an unknown argument added.

**The oracle** (:func:`judge`), per case:

1. The transport returned. An exception out of ``handle_body`` is the unstructured failure the bar
   forbids — in `lui/server.py` it becomes an HTTP 500 with no JSON-RPC body.
2. Every reply is strict JSON (no ``NaN``) and a conformant JSON-RPC 2.0 response: ``jsonrpc``,
   an ``id`` of an allowed type, exactly one of ``result`` / ``error``, an integer error code.
3. A ``tools/call`` result has ``content`` text blocks and a boolean ``isError``.
4. No reply reflects an oversize input: a reply over :data:`REFLECTION_LIMIT` bytes to a request
   that carried a megabyte string is an amplifier, not an error message.
5. **No argument that violates the tool's declared ``inputSchema`` reaches an engine.** The engine
   is replaced by a probe that records what it received, and each received argument object is
   checked against the schema — by this module's own validator and, when the ``jsonschema``
   package is importable, by the reference Draft 2020-12 implementation as well.

    python -m argus.eval.mcp_fuzz
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from argus.eval import artefact

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "mcp_fuzz.json"

REFLECTION_LIMIT = 16_384
"""Bytes. The largest honest error reply here is a few hundred; sixteen kilobytes is generous."""

MEGABYTE = 1_048_576

Transport = Callable[[bytes], tuple[int, bytes]]


@dataclass(frozen=True, slots=True)
class Case:
    label: str
    family: str
    body: bytes
    oversize: bool = False
    expect_error: bool = False
    """JSON-RPC requires an error reply here (the empty batch, §6); silence is a violation."""


def _rpc(method: Any, params: Any = None, *, id_: Any = 1, omit_params: bool = False,
         **extra: Any) -> dict[str, Any]:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": id_, "method": method}
    if not omit_params:
        message["params"] = {} if params is None else params
    message.update(extra)
    return message


def _call(name: Any, arguments: Any = None, *, omit_arguments: bool = False) -> dict[str, Any]:
    params: dict[str, Any] = {"name": name}
    if not omit_arguments:
        params["arguments"] = {} if arguments is None else arguments
    return _rpc("tools/call", params)


def _enc(message: Any) -> bytes:
    return json.dumps(message).encode()


_BAD_VALUES: tuple[tuple[str, Any], ...] = (
    ("null", None), ("true", True), ("int", 7), ("float", 2.5), ("string", "abc"),
    ("empty-string", ""), ("list", [1, 2]), ("object", {"a": 1}),
)
"""Each argument is sent as each of these in turn; the ones that match its declared type are
skipped when the case is built, so every case is a genuine type violation."""


def _matches(value: Any, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    return False


VALID_ARGUMENTS: dict[str, dict[str, Any]] = {
    "argus_ask": {"question": "what is NVDA doing", "book": "50% NVDA, 50% AAPL",
                  "memory": "[]"},
    "argus_quote": {"symbols": ["NVDA", "BTCUSDT"]},
    "argus_portfolio_impact": {"add": "TSLA", "size_percent": 15,
                               "book": {"NVDA": 50, "AAPL": 50}},
    "argus_stress": {"book": {"NVDA": 50, "COIN": 50}, "shock_percent": -10, "shocked": "oil"},
    "argus_execution_plan": {"symbol": "NVDA", "usd": 5000},
    "argus_scoreboard": {},
}
"""One well-formed call per tool — the baseline each ``args`` case mutates one field of."""


def corpus(tools: tuple[Mapping[str, Any], ...]) -> list[Case]:
    """Every case, built from the tools' own declared schemas so a new tool is fuzzed too."""
    cases: list[Case] = [
        Case("empty body", "parse", b""),
        Case("truncated object", "parse", b'{"jsonrpc": "2.0", "id": 1, "method": "ping"'),
        Case("not json", "parse", b"not json"),
        Case("invalid utf-8", "parse", b'\xff\xfe{"jsonrpc": "2.0"}'),
        Case("utf-8 bom", "parse", b'\xef\xbb\xbf{"jsonrpc": "2.0", "id": 1, "method": "ping"}'),
        Case("nested past the recursion limit", "parse", b"[" * 200_000 + b"]" * 200_000),
        Case("NaN literal as params", "parse",
             b'{"jsonrpc": "2.0", "id": 1, "method": "ping", "params": NaN}'),
        Case("overflowing number as id", "parse",
             b'{"jsonrpc": "2.0", "id": 1e999, "method": "ping"}'),
        Case("json null", "envelope", b"null"),
        Case("json number", "envelope", b"42"),
        Case("json string", "envelope", b'"tools/list"'),
        Case("json true", "envelope", b"true"),
        Case("jsonrpc 1.0", "envelope", _enc({"jsonrpc": "1.0", "id": 1, "method": "ping"})),
        Case("jsonrpc as number", "envelope", _enc({"jsonrpc": 2.0, "id": 1, "method": "ping"})),
        Case("method missing", "envelope", _enc({"jsonrpc": "2.0", "id": 1})),
        Case("method is a number", "envelope", _enc(_rpc(123))),
        Case("method is a list", "envelope", _enc(_rpc(["tools/list"]))),
        Case("method is null", "envelope", _enc(_rpc(None))),
        Case("id is true", "envelope", _enc(_rpc("ping", id_=True))),
        Case("id is an object", "envelope", _enc(_rpc("ping", id_={"x": 1}))),
        Case("id is a list", "envelope", _enc(_rpc("ping", id_=[1]))),
        Case("id is a float", "envelope", _enc(_rpc("ping", id_=1.5))),
        Case("id is a megabyte string", "envelope", _enc(_rpc("ping", id_="i" * MEGABYTE)),
             oversize=True),
        Case("params is a list", "envelope", _enc(_rpc("tools/list", [1, 2]))),
        Case("params is a string", "envelope", _enc(_rpc("initialize", "hello"))),
        Case("params is a number", "envelope", _enc(_rpc("tools/call", 5))),
        Case("params is true", "envelope", _enc(_rpc("ping", True))),
        Case("initialize with a list version", "envelope",
             _enc(_rpc("initialize", {"protocolVersion": ["2025-06-18"]}))),
        Case("unknown method", "envelope", _enc(_rpc("resources/list"))),
        Case("megabyte method name", "envelope", _enc(_rpc("m" * MEGABYTE)), oversize=True),
        Case("empty batch", "batch", b"[]", expect_error=True),
        Case("batch of numbers", "batch", b"[1, 2, 3]"),
        Case("batch of junk and one ping", "batch",
             _enc([None, "x", {"jsonrpc": "2.0", "id": 9, "method": "ping"}])),
        Case("nested batch", "batch", _enc([[_rpc("ping")]])),
        Case("batch of 10,000 pings", "batch",
             _enc([_rpc("ping", id_=i) for i in range(10_000)])),
        Case("batch of only notifications", "batch",
             _enc([{"jsonrpc": "2.0", "method": "notifications/initialized"}] * 3)),
        Case("call without params", "call", _enc(_rpc("tools/call", omit_params=True))),
        Case("call without a name", "call", _enc(_rpc("tools/call", {"arguments": {}}))),
        Case("call name is a number", "call", _enc(_call(123))),
        Case("call name is a list", "call", _enc(_call(["argus_scoreboard"]))),
        Case("call name is an object", "call", _enc(_call({"name": "argus_scoreboard"}))),
        Case("call name is null", "call", _enc(_call(None))),
        Case("call name is empty", "call", _enc(_call(""))),
        Case("unknown tool", "call", _enc(_call("delete_everything"))),
        Case("megabyte tool name", "call", _enc(_call("n" * MEGABYTE)), oversize=True),
        Case("tool name with control characters", "call",
             _enc(_call("argus_quote\u0000\r\n<script>"))),
        Case("tool name differs only in case", "call", _enc(_call("ARGUS_SCOREBOARD"))),
        Case("arguments is a list", "call", _enc(_call("argus_scoreboard", [1]))),
        Case("arguments is a string", "call", _enc(_call("argus_ask", "what is NVDA"))),
        Case("arguments is null", "call", _enc(_rpc("tools/call", {
            "name": "argus_scoreboard", "arguments": None}))),
        Case("arguments omitted", "call", _enc(_call("argus_scoreboard", omit_arguments=True))),
        Case("megabyte question", "args",
             _enc(_call("argus_ask", {"question": "q" * MEGABYTE})), oversize=True),
        Case("ten-thousand-name book", "args", _enc(_call("argus_stress", {
            "book": {f"X{i}": 1 for i in range(10_000)}}))),
        Case("NaN shock", "args", b'{"jsonrpc": "2.0", "id": 1, "method": "tools/call", '
             b'"params": {"name": "argus_stress", "arguments": '
             b'{"book": {"NVDA": 100}, "shock_percent": NaN}}}'),
        Case("infinite usd", "args", b'{"jsonrpc": "2.0", "id": 1, "method": "tools/call", '
             b'"params": {"name": "argus_execution_plan", "arguments": '
             b'{"symbol": "NVDA", "usd": 1e999}}}'),
    ]
    for tool in tools:
        name = str(tool["name"])
        schema = tool["inputSchema"]
        valid = VALID_ARGUMENTS.get(name, {})
        properties: Mapping[str, Any] = schema.get("properties", {})
        cases.append(Case(f"{name}: unknown argument", "args",
                          _enc(_call(name, {**valid, "zz_unknown": 1}))))
        for required in schema.get("required", []):
            partial = {k: v for k, v in valid.items() if k != required}
            cases.append(Case(f"{name}: missing {required}", "args", _enc(_call(name, partial))))
        for prop, spec in properties.items():
            kind = str(spec.get("type", ""))
            for tag, value in _BAD_VALUES:
                if _matches(value, kind):
                    continue
                cases.append(Case(f"{name}.{prop} as {tag}", "args",
                                  _enc(_call(name, {**valid, prop: value}))))
            items = spec.get("items")
            if kind == "array" and isinstance(items, Mapping):
                cases.append(Case(f"{name}.{prop} items as numbers", "args",
                                  _enc(_call(name, {**valid, prop: [1, 2]}))))
            extra = spec.get("additionalProperties")
            if kind == "object" and isinstance(extra, Mapping):
                cases.append(Case(f"{name}.{prop} values as strings", "args",
                                  _enc(_call(name, {**valid, prop: {"NVDA": "abc"}}))))
    return cases


# --- the oracle ------------------------------------------------------------------------------


def schema_violations(schema: Mapping[str, Any], value: Any, path: str = "$") -> list[str]:
    """The subset of JSON Schema the ARGUS tools declare, checked independently of the server.

    ``type`` (object / string / number / array), ``properties``, ``required``, ``items`` and
    ``additionalProperties`` as a schema. Numbers must also be finite: JSON has no NaN, so a
    non-finite number reaching an engine only got there through Python's lenient parser.
    """
    kind = schema.get("type")
    if kind is not None and not _matches(value, str(kind)):
        return [f"{path}: expected {kind}, got {type(value).__name__}"]
    if kind == "number" and not math.isfinite(float(value)):
        return [f"{path}: non-finite number"]
    out: list[str] = []
    if kind == "object":
        props: Mapping[str, Any] = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                out.append(f"{path}.{required}: required")
        extra = schema.get("additionalProperties")
        for key, item in value.items():
            if key in props:
                out += schema_violations(props[key], item, f"{path}.{key}")
            elif isinstance(extra, Mapping):
                out += schema_violations(extra, item, f"{path}.{key}")
    if kind == "array" and isinstance(schema.get("items"), Mapping):
        for i, item in enumerate(value):
            out += schema_violations(schema["items"], item, f"{path}[{i}]")
    return out


def _reference_violations(schema: Mapping[str, Any], value: Any) -> list[str] | None:
    """The same question put to the reference implementation, when it is installed."""
    try:
        reference: Any = importlib.import_module("jsonschema")
    except ImportError:
        return None
    validator: Any = reference.Draft202012Validator(dict(schema))
    return [str(error.message) for error in validator.iter_errors(value)]


@dataclass
class Probe:
    """Stands in for the engines: records what reached it, answers nothing of substance."""

    received: list[tuple[str, Any]] = field(default_factory=list)

    def __call__(self, name: str, arguments: Mapping[str, Any]) -> tuple[str, bool]:
        self.received.append((name, arguments))
        return f"probe: {name}", False


_ALLOWED_ID = (str, int, type(None))


def _reply_violations(reply: Any) -> list[str]:
    if not isinstance(reply, dict):
        return [f"reply is {type(reply).__name__}, not an object"]
    out: list[str] = []
    if reply.get("jsonrpc") != "2.0":
        out.append("jsonrpc is not '2.0'")
    if "id" not in reply:
        out.append("reply has no id")
    elif isinstance(reply["id"], bool) or not isinstance(reply["id"], _ALLOWED_ID):
        out.append(f"id of type {type(reply['id']).__name__}")
    if ("result" in reply) == ("error" in reply):
        out.append("not exactly one of result / error")
    error = reply.get("error")
    if error is not None and (not isinstance(error, dict)
                              or not isinstance(error.get("code"), int)
                              or isinstance(error.get("code"), bool)
                              or not isinstance(error.get("message"), str)):
        out.append("malformed error object")
    result = reply.get("result")
    if isinstance(result, dict) and "content" in result:
        content = result["content"]
        if not isinstance(content, list) or not all(
                isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
                for b in content):
            out.append("tools/call content is not a list of text blocks")
        if not isinstance(result.get("isError"), bool):
            out.append("tools/call isError is not a boolean")
    return out


@dataclass(frozen=True, slots=True)
class Verdict:
    label: str
    family: str
    status: int | None
    raised: str
    violations: tuple[str, ...]
    reply_bytes: int

    @property
    def clean(self) -> bool:
        return not self.raised and not self.violations


def judge(case: Case, transport: Transport, probe: Probe,
          schemas: Mapping[str, Mapping[str, Any]]) -> Verdict:
    """Send one case and apply every oracle rule to what came back."""
    before = len(probe.received)
    try:
        status, body = transport(case.body)
    except RecursionError:
        return Verdict(case.label, case.family, None, "RecursionError", (), 0)
    except Exception as exc:  # the whole point: anything escaping the transport is a finding
        return Verdict(case.label, case.family, None, type(exc).__name__, (), 0)

    violations: list[str] = []
    if body:
        def reject(token: str) -> Any:
            raise ValueError(token)
        try:
            payload = json.loads(body.decode("utf-8"), parse_constant=reject)
        except (UnicodeDecodeError, ValueError) as exc:
            violations.append(f"reply is not strict JSON ({exc})")
        else:
            replies = payload if isinstance(payload, list) else [payload]
            if isinstance(payload, list) and not payload:
                violations.append("reply is an empty array")
            for reply in replies:
                violations += _reply_violations(reply)
    elif status not in (202, 204):
        violations.append(f"empty body with status {status}")
    if case.expect_error and b'"error"' not in body:
        violations.append("no error reply where JSON-RPC requires one")
    if case.oversize and len(body) > REFLECTION_LIMIT:
        violations.append(f"reflected an oversize input ({len(body):,} bytes)")

    for name, arguments in probe.received[before:]:
        schema = schemas.get(name)
        if schema is None:
            violations.append(f"engine reached for undeclared tool {name!r}")
            continue
        found = schema_violations(schema, arguments)
        reference = _reference_violations(schema, arguments)
        if found or reference:
            violations.append(f"engine {name} received invalid arguments: "
                              f"{(found or reference or [''])[0]}")
    return Verdict(case.label, case.family, status, "", tuple(violations), len(body))


def fuzz(transport_for: Callable[[Probe], Transport],
         tools: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    """Run the whole corpus through one server and summarise."""
    probe = Probe()
    transport = transport_for(probe)
    schemas = {str(t["name"]): t["inputSchema"] for t in tools}
    verdicts = [judge(case, transport, probe, schemas) for case in corpus(tools)]
    by_family: dict[str, dict[str, int]] = {}
    for verdict in verdicts:
        slot = by_family.setdefault(verdict.family, {"cases": 0, "clean": 0})
        slot["cases"] += 1
        slot["clean"] += verdict.clean
    return {
        "cases": len(verdicts),
        "clean": sum(v.clean for v in verdicts),
        "unstructured_exceptions": sum(bool(v.raised) for v in verdicts),
        "cases_with_violations": sum(bool(v.violations) for v in verdicts),
        "engine_calls": len(probe.received),
        "reference_validator": _reference_violations({"type": "object"}, {}) is not None,
        "by_family": by_family,
        "failures": [
            {"label": v.label, "family": v.family, "status": v.status, "raised": v.raised,
             "violations": list(v.violations)[:3]}
            for v in verdicts if not v.clean
        ],
    }


def hand_rolled_transport(probe: Probe) -> Transport:
    from argus.lui import mcp_server

    def send(body: bytes) -> tuple[int, bytes]:
        return mcp_server.handle_body(body, tool=probe)

    return send


def run(path: Path = REPORT_PATH) -> dict[str, Any]:
    from argus.lui import mcp_server

    report = {
        "what": "MCP wire fuzz: zero unstructured exceptions, conformant replies, no reflection, "
                "no schema-invalid argument reaching an engine",
        "server": "argus.lui.mcp_server (hand-rolled, hardened 2026-09-25)",
        "result": fuzz(hand_rolled_transport, mcp_server.TOOLS),
    }
    artefact.write(path, report)
    return report


def main() -> int:  # pragma: no cover - CLI
    report = run()
    result = report["result"]
    print(f"cases {result['cases']}  clean {result['clean']}  unstructured exceptions "
          f"{result['unstructured_exceptions']}  with violations "
          f"{result['cases_with_violations']}  engine calls {result['engine_calls']}  "
          f"reference validator {result['reference_validator']}")
    for failure in result["failures"][:40]:
        print(f"  [{failure['family']}] {failure['label']}: {failure['raised']} "
              f"{failure['violations']}")
    return 0 if result["clean"] == result["cases"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
