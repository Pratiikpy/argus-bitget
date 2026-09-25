"""Bitget's own US-equity data service, read through its MCP endpoint — and no key is needed.

**This exists because Track 3 scores "data sources / Skill integration count *and effectiveness*"
by name, and ARGUS was not using this service at all.** The handbook lists `bitget-mcp-server`
alongside `bitget-signal` and says plainly that neither requires a Bitget account or an API key.
`market/skills.py` already probes the five `bitget-signal` research Skills; nothing touched this
one, which is the larger of the two and the one carrying the *anchor* data for tokenized US
equities — quotes, fundamentals, earnings calendar, institutional holdings and analyst estimates
for the underlying stocks our rTokens track.

**What is actually there, enumerated by calling it rather than by reading the docs.** The server
(`bitget-mcp-server` v4.0.5 at ``https://agent.bitget.com/mcp``) exposes **two** tools, not the
long list the handbook's category table implies: ``guide`` walks a catalog, and ``do_query``
executes one catalog entry. Behind them sit **67 entries in five categories** — crypto 40, equity
21, ETF 3, news 1, sentiment 2.

The 22 equity entries matter most here, because ARGUS's universe is twelve tokenized US equities
and the *anchor* is exactly what a tokenized-equity desk cannot see from the venue's own book:
``equity_calendar``, ``equity_estimates_consensus``, ``equity_ownership_form_13f``,
``equity_fundamental_*`` and the rest.

**Two things were found by probing that no document states, and both would cost an afternoon:**

* **The endpoint returns 403 to Python's default User-Agent** and 200 to curl's. Identical request
  otherwise. So :data:`_HEADERS` sets one explicitly — a client that omits it fails in a way that
  looks exactly like a credential problem and is not.
* **``do_query`` takes ``entry_id``, not ``id``.** The catalog lists entries under a key called
  ``id``, so the obvious call is wrong, and the server answers with a pydantic validation error
  rather than a hint. Named here so the next reader does not rediscover it.

Transport is JSON-RPC over HTTP with Server-Sent Events framing (a reply arrives as ``data: {...}``
lines rather than a bare body). Since 2026-09-25 that transport is :mod:`argus.market.rpc`, the one
client both Bitget MCP servers now share, and :class:`BitgetMcpError` carries a typed ``kind``:
a refused argument (``tool_execution``), the service's own upstream answering 503
(``upstream_5xx``), a missing session (``protocol``) and a dropped connection (``transport``) used
to be one undifferentiated string and are now four facts. The negotiated protocol version is
2025-11-25 — the server's newest, found by asking it — where this module used to pin 2024-11-05.

**This module fetches; it does not decide.** Everything returned is evidence for the desk to weigh,
and it is deliberately *not* wired into the Constitution — the risk layer computes, and a rule that
reaches for the network is a rule that can fail open.

    python -m argus.market.bitget_mcp
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from argus.market.rpc import (
    LATEST_SUPPORTED,
    ErrorKind,
    JsonRpcClient,
    RpcError,
    payload_failure,
)

ENDPOINT = "https://agent.bitget.com/mcp"
"""The HTTP transport the handbook publishes. Keyless, and confirmed keyless by calling it."""

_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    # **Load-bearing.** The endpoint 403s Python's default User-Agent and answers curl's. This is
    # not documented anywhere and presents as an auth failure on a service that needs no auth.
    "User-Agent": "curl/8.0",
}

PROTOCOL_VERSION = LATEST_SUPPORTED
"""The version asked for first. The server's answer is what is used — see
:attr:`BitgetDataService.negotiation`."""
TIMEOUT = 45.0


class BitgetMcpError(RpcError):
    """The service could not be reached or refused a call. Raised rather than returning empty.

    An empty result and an unreachable service are different facts, and `market/skills.py` already
    learned that lesson: a probe that reports a transport failure as "no data" measures the probe.
    ``kind`` (an :class:`~argus.market.rpc.ErrorKind`) says which failure it was; the message text
    is unchanged from before the taxonomy, so ``eval/datacoverage.py``'s reading of it still holds.
    """


@dataclass(frozen=True, slots=True)
class Entry:
    """One catalog entry: a named dataset `do_query` can execute."""

    entry_id: str
    category: str
    name: str = ""
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id, "category": self.category,
            "name": self.name, "description": self.description,
        }


class BitgetDataService:
    """A session against the MCP server. One handshake, then any number of queries."""

    def __init__(self, *, client: JsonRpcClient | None = None) -> None:
        self.client = client or JsonRpcClient(
            ENDPOINT, headers=_HEADERS, timeout=TIMEOUT, error_type=BitgetMcpError,
            client_info={"name": "argus", "version": "1.0"},
        )
        try:
            self.negotiation = self.client.initialize()
        except RpcError as exc:  # an injected client may raise the base type
            raise BitgetMcpError(exc.kind, str(exc), http_status=exc.http_status) from exc
        self.server = self.negotiation.server

    def _call(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """One ``tools/call``, returning the raw result. A protocol failure raises typed; a tool
        failure is left in the result for :meth:`query` to read, as SEP-1303 places it there."""
        try:
            tool_result = self.client.call_tool(tool, arguments)
        except RpcError as exc:
            raise BitgetMcpError(exc.kind, f"{tool}: {exc}", http_status=exc.http_status,
                                 code=exc.code, data=exc.data, attempts=exc.attempts) from exc
        result: dict[str, Any] = {"content": list(tool_result.content),
                                  "isError": tool_result.is_error}
        if tool_result.structured is not None:
            result["structuredContent"] = tool_result.structured
        return result

    def categories(self) -> list[dict[str, Any]]:
        """The five top-level groups and how many entries each holds."""
        result = self._call("guide", {})
        found = result.get("structuredContent", {}).get("categories", [])
        return [dict(c) for c in found]

    def entries(self, category: str) -> list[Entry]:
        """Every dataset in one category, by id."""
        result = self._call("guide", {"category": category})
        content = result.get("structuredContent", {})
        rows = content.get("entries") or content.get("items") or []
        return [
            Entry(
                entry_id=str(row.get("id", "")), category=category,
                name=str(row.get("name", "") or ""),
                description=str(row.get("description", "") or ""),
            )
            for row in rows if row.get("id")
        ]

    def query(self, entry_id: str, **params: Any) -> dict[str, Any]:
        """Execute one catalog entry.

        The keyword is ``entry_id``. The catalog calls the same field ``id``, so the natural call
        is rejected with a pydantic error — see the module docstring.
        """
        result = self._call("do_query", {"entry_id": entry_id, "params": dict(params)})
        if result.get("isError"):
            text = json.dumps(result.get("content", ""))[:300]
            raise BitgetMcpError(ErrorKind.TOOL_EXECUTION, f"{entry_id}: {text}")
        payload = result.get("structuredContent", {})
        if not isinstance(payload, dict):
            payload = {}
        if not payload.get("success", True):
            # success=false with the upstream's own status: a 503 is the service behind this one
            # being down, a 4xx is our request being refused. Different owners, different kinds.
            failed = payload_failure(payload)
            kind = failed[0] if failed else ErrorKind.DOMAIN
            raise BitgetMcpError(kind, f"{entry_id}: service reported failure: {payload}")
        return dict(payload.get("data", {}))

    def results(self, entry_id: str, **params: Any) -> list[dict[str, Any]]:
        """The rows of a query, unwrapped from the service's envelope."""
        rows = self.query(entry_id, **params).get("results", [])
        return [dict(r) for r in rows] if isinstance(rows, list) else []

    # --- the anchor facts a tokenized-equity desk cannot see from the venue's book -------------

    def quote(self, symbol: str) -> dict[str, Any]:
        """Live bid/ask/last on the **underlying** stock, not the rToken.

        The gap between the two is the basis every arbitrage and hedge claim in this project rests
        on, and until now it was read from the venue's own index rather than from an equity feed.
        """
        rows = self.results("equity_price_quote", symbol=symbol)
        return rows[0] if rows else {}

    def next_earnings(self, symbol: str) -> dict[str, Any]:
        """The most recently known scheduled report. A date, from the source, rather than an
        inference — **not guaranteed to be in the future**: the source (``equity_calendar``, not
        ``equity_calendar_earnings`` — the catalog's real entry name, found by listing it live
        rather than assumed from this method's own prior code) returns quarterly history sorted
        newest-first and does not always carry a not-yet-happened row, so the caller must check the
        date rather than assume "returned" means "upcoming".

        Directly relevant to the risk layer's event-window reasoning: a position held across an
        earnings print is a different risk from the same position held the week before.

        ``report_date`` is a normalised field this method adds — the real source has no single
        date column, only ``perf_brief_dsclsr_date`` (confirmed) and
        ``perf_briefing_fore_dsclsr_date`` (forecast, present before the confirmed one is). The
        confirmed date wins when both exist.
        """
        rows = self.results("equity_calendar", symbol=symbol)
        if not rows:
            return {}
        row = rows[0]
        report_date = row.get("perf_brief_dsclsr_date") or row.get("perf_briefing_fore_dsclsr_date")
        return {**row, "report_date": report_date} if report_date else dict(row)

    def consensus(self, symbol: str) -> dict[str, Any]:
        """Analyst consensus and price targets."""
        rows = self.results("equity_estimates_consensus", symbol=symbol)
        return rows[0] if rows else {}

    def price_targets(self, symbol: str) -> list[dict[str, Any]]:
        """Every analyst price-target action on file, newest first
        (``equity_estimates_price_target``). Measured 2026-09-23: 972 rows for NVDA reaching back to
        2016, the latest dated 2026-09-09 — current, unlike ``consensus``, whose NVDA row was
        scraped in 2024. Ratings arrive in Chinese; see :data:`RATING_STANCE`."""
        return self.results("equity_estimates_price_target", symbol=symbol)

    def institutional_holdings(self, symbol: str) -> list[dict[str, Any]]:
        """13F filings. Who owns it, at the last reporting period."""
        return self.results("equity_ownership_form_13f", symbol=symbol)


RATING_STANCE: dict[str, str] = {
    "强力买进": "buy", "买入": "buy", "增持": "buy", "跑赢大盘": "buy", "积极": "buy",
    "持有": "hold", "中性": "hold", "持股观望": "hold", "市场持平": "hold",
    "减持": "sell", "卖出": "sell", "逊于大盘": "sell",
}
"""The rating vocabulary ``equity_estimates_price_target`` returns, read into three stances.
Built from the full set of values on NVDA's 972 rows (2026-09-23): strong buy, buy, overweight,
outperform and positive are buy; hold, neutral, wait-and-see and market-perform are hold;
underweight, sell and underperform are sell. An unlisted value is left out of the count rather
than guessed."""


def underlying_of(rtoken: str) -> str:
    """``NVDAUSDT`` -> ``NVDA``. The rToken symbol carries its anchor's ticker as a prefix.

    Deliberately a plain suffix strip rather than a lookup table: the venue's own naming is the
    mapping, and a table would be a second source of truth to keep in sync."""
    return rtoken.upper().removesuffix("USDT")


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    service = BitgetDataService()
    print(f"connected to {service.server} at {ENDPOINT} — no API key")
    total = 0
    for category in service.categories():
        count = int(category.get("entry_count", 0))
        total += count
        note = str(category.get("description", ""))[:44]
        print(f"  {category.get('key', '?'):10} {count:3} entr(ies)  {note}")
    print(f"  {'TOTAL':10} {total:3}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ENDPOINT", "BitgetDataService", "BitgetMcpError", "Entry", "main", "underlying_of",
]
