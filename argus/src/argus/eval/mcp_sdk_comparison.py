"""The official MCP Python SDK against the hand-rolled server, on the same fuzz corpus.

The synthesis (row S15) proposed replacing `lui/mcp_server.py` with the SDK's server
(modelcontextprotocol/python-sdk, MIT; ``src/mcp/server/lowlevel/server.py``,
``src/mcp/server/mcpserver/server.py``, ``src/mcp/shared/jsonrpc_dispatcher.py``) — *only if the
SDK installs and is compatible*, keeping every tool and its schema exactly. This module is the
experiment that decides it, run rather than argued.

**The SDK server under test** is built the one way that can keep our schemas byte-for-byte: the
low-level ``Server`` with ``on_list_tools`` / ``on_call_tool`` handlers returning our own
``TOOLS``. The high-level ``MCPServer`` was read and ruled out before running anything: its
``Tool.from_function`` (``mcpserver/tools/base.py:63-117``) *derives* each ``inputSchema`` from the
Python signature through pydantic's ``model_json_schema``, so the schemas an agent sees would be
pydantic's rendering (``title`` keys, ``anyOf`` for optionals), not the ones published today.

It is served through the SDK's own Streamable HTTP app in stateless JSON mode
(``Server.streamable_http_app(stateless_http=True, json_response=True)``) and driven with
Starlette's ``TestClient``, so every fuzz body crosses the SDK's real transport, its
``JSONRPCDispatcher`` exception boundary (``jsonrpc_dispatcher.py:701-770``) and its pydantic
request validation — the parts the synthesis credited it for. The engine is the same
:class:`~argus.eval.mcp_fuzz.Probe` the hand-rolled server is fuzzed with.

**And the interoperability check the synthesis asked for**: the SDK's own *client* against the
console's real ``/mcp`` handler over HTTP (:func:`external_client_round_trip`).

**Beyond the fuzz, it records what adoption would cost:** whether the SDK imports in the
project's own environment, how many distributions it pulls that the project does not already
have, and whether it can be called from the console's existing ``ThreadingHTTPServer`` handler
without a second server — `lui/server.py` hands ``/mcp`` bodies to a synchronous function, and the
SDK's transport is an ASGI app on an ``anyio`` event loop.

    python -m argus.eval.mcp_sdk_comparison     # needs `mcp` importable (it is not a project dep)
"""

from __future__ import annotations

import importlib
import importlib.metadata
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from argus.eval import artefact, mcp_fuzz

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "mcp_sdk_comparison.json"

BEFORE_HARDENING: dict[str, Any] = {
    "measured": "2026-09-25, same corpus and oracle, against lui/mcp_server.py as it stood "
                "before the hardening in this change (that source is not kept, so this row is "
                "a record, not a re-run)",
    "cases": 145, "clean": 46, "unstructured_exceptions": 3, "cases_with_violations": 96,
    "engine_calls": 98,
}

_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": "2025-06-18",
}


def sdk_available() -> str:
    """The installed SDK version, or ``""`` when it cannot be imported here."""
    try:
        importlib.import_module("mcp.server.lowlevel")
    except ImportError:
        return ""
    return importlib.metadata.version("mcp")


def sdk_transport(tools: tuple[Mapping[str, Any], ...]) -> Any:
    """A transport factory serving ``tools`` through the SDK's low-level server over HTTP."""
    types: Any = importlib.import_module("mcp_types")
    lowlevel: Any = importlib.import_module("mcp.server.lowlevel")
    exceptions: Any = importlib.import_module("mcp.shared.exceptions")
    testclient: Any = importlib.import_module("starlette.testclient")
    names = {str(t["name"]) for t in tools}

    def factory(probe: mcp_fuzz.Probe) -> mcp_fuzz.Transport:
        async def list_tools(ctx: Any, params: Any) -> Any:
            return types.ListToolsResult(tools=[
                types.Tool(name=t["name"], description=t["description"],
                           input_schema=t["inputSchema"]) for t in tools])

        async def call_tool(ctx: Any, params: Any) -> Any:
            if params.name not in names:
                raise exceptions.MCPError(-32602, f"unknown tool {params.name[:64]!r}")
            text, is_error = probe(params.name, params.arguments or {})
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=text)], is_error=is_error)

        server = lowlevel.Server("argus-research-desk", version="1.0.0",
                                 on_list_tools=list_tools, on_call_tool=call_tool)
        app = server.streamable_http_app(stateless_http=True, json_response=True)
        # The SDK turns on DNS-rebinding protection for a localhost bind and answers any other
        # Host header with 421, so the client must present itself as 127.0.0.1 — as the
        # console's own clients would.
        client = testclient.TestClient(app, base_url="http://127.0.0.1:8000",
                                       raise_server_exceptions=False)
        client.__enter__()

        def send(body: bytes) -> tuple[int, bytes]:
            response = client.post("/mcp", content=body, headers=_HEADERS)
            return int(response.status_code), bytes(response.content)

        return send

    return factory


def schema_round_trip(tools: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    """Do our schemas come back from the SDK's ``tools/list`` exactly as published?"""
    factory = sdk_transport(tools)
    send = factory(mcp_fuzz.Probe())
    status, body = send(b'{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}')
    import json

    listed = json.loads(body)["result"]["tools"] if status == 200 else []
    by_name = {t["name"]: t for t in listed}
    mismatched = [str(t["name"]) for t in tools
                  if by_name.get(t["name"], {}).get("inputSchema") != t["inputSchema"]]
    return {"status": status, "tools_listed": len(listed), "schemas_mismatched": mismatched}


def external_client_round_trip() -> dict[str, Any]:
    """The official SDK *client* against this project's own server, over real HTTP.

    The synthesis asks that "the same tools list round-trips through an external client". The
    console's real handler (`lui/server.py`'s ``Handler`` on a ``ThreadingHTTPServer``, bound to
    an ephemeral localhost port) serves ``/mcp``; the SDK's ``streamable_http_client`` and
    ``ClientSession`` initialise, list the tools, call one tool properly and one with a wrong
    argument type. Every schema the client parsed is compared with the published one.
    """
    import asyncio
    import threading
    from http.server import ThreadingHTTPServer

    from argus.lui import mcp_server
    from argus.lui.server import Handler

    mcp: Any = importlib.import_module("mcp")
    client_http: Any = importlib.import_module("mcp.client.streamable_http")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/mcp"
    published = {str(t["name"]): t["inputSchema"] for t in mcp_server.TOOLS}

    async def session() -> dict[str, Any]:
        async with (
            client_http.streamable_http_client(url) as (read, write),
            mcp.ClientSession(read, write) as client,
        ):
            init = await client.initialize()
            listed = await client.list_tools()
            good = await client.call_tool("argus_scoreboard", {})
            bad = await client.call_tool("argus_execution_plan", {"symbol": "NVDA", "usd": "x"})
            parsed = {t.name: t.model_dump(by_alias=True, exclude_none=True)["inputSchema"]
                      for t in listed.tools}
            return {
                "negotiated_version": str(init.protocol_version),
                "tools_listed": sorted(parsed),
                "schemas_identical": parsed == published,
                "good_call_is_error": bool(good.is_error),
                "bad_call_is_error": bool(bad.is_error),
                "bad_call_text": str(bad.content[0].text),
            }

    try:
        return asyncio.run(session())
    finally:
        httpd.shutdown()


def missing_distributions() -> list[str]:
    """Distributions the SDK requires, recursively, that the ARGUS environment does not have.

    Read from the metadata of the environment this runs in (where the SDK is installed), then
    each is looked up in the project's own ``.venv``.
    """
    import re

    requirements: Any = importlib.import_module("packaging.requirements")
    venv = Path(__file__).resolve().parents[3] / ".venv" / "Lib" / "site-packages"
    present = {re.sub(r"[-_.]+", "-", p.name.split("-")[0]).lower()
               for p in venv.glob("*.dist-info")} if venv.exists() else set()
    seen: set[str] = set()
    queue = ["mcp"]
    while queue:
        name = queue.pop()
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            requires = importlib.metadata.requires(name) or []
        except importlib.metadata.PackageNotFoundError:
            continue
        for requirement in requires:
            parsed: Any = requirements.Requirement(requirement)
            if parsed.marker is not None and not parsed.marker.evaluate({"extra": ""}):
                continue
            queue.append(str(parsed.name))
    return sorted(seen - present)


def latency(transport: mcp_fuzz.Transport, rounds: int = 200) -> float:
    """Median milliseconds for one ``tools/list`` round trip."""
    body = b'{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}'
    samples: list[float] = []
    for _ in range(rounds):
        started = time.perf_counter()
        transport(body)
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    return round(samples[len(samples) // 2], 3)


def run(path: Path = REPORT_PATH) -> dict[str, Any]:
    from argus.lui import mcp_server

    version = sdk_available()
    report: dict[str, Any] = {
        "what": "official MCP SDK (low-level Server, Streamable HTTP, stateless JSON) vs the "
                "hand-rolled lui/mcp_server.py, same fuzz corpus and oracle",
        "sdk_version": version,
        "hand_rolled_before_hardening": BEFORE_HARDENING,
        "hand_rolled": mcp_fuzz.fuzz(mcp_fuzz.hand_rolled_transport, mcp_server.TOOLS),
        "hand_rolled_tools_list_ms": latency(
            mcp_fuzz.hand_rolled_transport(mcp_fuzz.Probe())),
    }
    if version:
        report["sdk"] = mcp_fuzz.fuzz(sdk_transport(mcp_server.TOOLS), mcp_server.TOOLS)
        report["sdk_schema_round_trip"] = schema_round_trip(mcp_server.TOOLS)
        report["sdk_tools_list_ms"] = latency(sdk_transport(mcp_server.TOOLS)(mcp_fuzz.Probe()))
        report["sdk_distributions_missing_from_argus_venv"] = missing_distributions()
        report["sdk_client_against_argus_server"] = external_client_round_trip()
    artefact.write(path, report)
    return report


def main() -> int:  # pragma: no cover - CLI
    report = run()
    for side in ("hand_rolled", "sdk"):
        result = report.get(side)
        if result is None:
            print(f"{side}: not run (SDK not importable here)")
            continue
        print(f"{side}: cases {result['cases']}  clean {result['clean']}  unstructured "
              f"{result['unstructured_exceptions']}  violations "
              f"{result['cases_with_violations']}  engine calls {result['engine_calls']}")
        for failure in result["failures"][:60]:
            print(f"    [{failure['family']}] {failure['label']}: {failure['status']} "
                  f"{failure['raised']} {failure['violations'][:1]}")
    for key in ("sdk_schema_round_trip", "hand_rolled_tools_list_ms", "sdk_tools_list_ms",
                "sdk_distributions_missing_from_argus_venv", "sdk_client_against_argus_server"):
        if key in report:
            print(f"{key}: {report[key]}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
