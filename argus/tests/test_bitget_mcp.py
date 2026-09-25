"""Bitget's own data service, and the measurement of how much of it answers.

**Track 3 scores "data sources / Skill integration count *and effectiveness*" by name**, and
before 2026-09-21 ARGUS used `bitget-signal`'s five research Skills and had never touched
`bitget-mcp-server` — the larger of the two services and the one carrying anchor data for
tokenized equities: quotes, earnings dates, analyst consensus and 13F holdings for the underlying
stocks. Neither service needs an API key.

These are live-network tests. That is deliberate: the claim being made is that a *real* Bitget
service answers, and a mocked version of it would test the mock. They skip rather than fail when
the network is unavailable, because an unreachable endpoint is an environment fact and not a defect
in this repository. The same holds one layer down: the MCP host can answer while the data service
behind it returns a 503 (it did for most of 2026-09-25), and that outage belongs to Bitget. Only
those kinds skip (`_OUTAGE`); a 4xx, a tool error or a wrong shape still fails, because those would
be ours.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import pytest

from argus.market.bitget_mcp import (
    ENDPOINT,
    BitgetDataService,
    BitgetMcpError,
    underlying_of,
)
from argus.market.rpc import ErrorKind

T = TypeVar("T")

# The failures that are the service being down rather than the request being wrong.
_OUTAGE = frozenset({ErrorKind.UPSTREAM_5XX, ErrorKind.TIMEOUT, ErrorKind.TRANSPORT})


def _live(call: Callable[[], T]) -> T:
    """Run one live data call; skip, naming the kind, when Bitget's side is down."""
    try:
        return call()
    except BitgetMcpError as exc:
        if exc.kind in _OUTAGE:
            pytest.skip(f"Bitget data service down ({exc.kind}): {str(exc)[:120]}")
        raise


@pytest.fixture(scope="module")
def service() -> BitgetDataService:
    try:
        return BitgetDataService()
    except BitgetMcpError as exc:
        pytest.skip(f"Bitget MCP unreachable: {exc}")


class TestTheServiceIsReachableWithoutACredential:
    def test_it_connects_with_no_api_key(self, service: BitgetDataService) -> None:
        """The whole reason this is worth integrating: a judge can reproduce it."""
        assert "bitget-mcp-server" in service.server

    def test_the_endpoint_is_the_published_one(self) -> None:
        assert ENDPOINT == "https://agent.bitget.com/mcp"

    def test_the_catalog_is_larger_than_the_skills_surface(
        self, service: BitgetDataService
    ) -> None:
        """67 entries against `bitget-signal`'s 19 probed tools. Size is not quality, but a
        service this large being entirely unused was the gap."""
        catalog = service.categories()
        assert sum(int(c.get("entry_count", 0)) for c in catalog) > 50

    def test_the_equity_category_carries_the_anchor_data(
        self, service: BitgetDataService
    ) -> None:
        """rTokens track US equities. The venue's book cannot show an earnings date.

        ``equity_calendar``, not ``equity_calendar_earnings`` — the earlier name was never real,
        caught only once this test and `next_earnings` were actually run against the live catalog
        (2026-09-23) rather than trusted from an earlier reading."""
        ids = {e.entry_id for e in service.entries("equity")}
        assert "equity_calendar" in ids
        assert "equity_estimates_consensus" in ids
        assert "equity_ownership_form_13f" in ids


class TestTheAnchorFactsComeBack:
    def test_a_live_quote_on_the_underlying(self, service: BitgetDataService) -> None:
        """The basis every hedge and arbitrage claim rests on is rToken minus *this*."""
        quote = _live(lambda: service.quote("NVDA"))
        assert quote.get("symbol") == "NVDA"
        assert float(quote.get("last_price") or 0) > 0

    def test_the_next_earnings_date_is_a_date_from_the_source(
        self, service: BitgetDataService
    ) -> None:
        """A position held across a print is a different risk from one held the week before, and
        this is that date from the issuer's calendar rather than inferred."""
        earnings = _live(lambda: service.next_earnings("NVDA"))
        assert earnings.get("report_date")

    def test_institutional_holdings_are_real_filings(
        self, service: BitgetDataService
    ) -> None:
        rows = _live(lambda: service.institutional_holdings("NVDA"))
        assert rows and rows[0].get("period_ending")


class TestTheTrapsThatCostAnAfternoon:
    def test_the_user_agent_is_set_explicitly(self) -> None:
        """**The endpoint 403s Python's default User-Agent and answers curl's.** Identical request
        otherwise. Without this header the service looks like it needs a credential, on a service
        that needs none — so the header is asserted rather than left to chance."""
        from argus.market.bitget_mcp import _HEADERS

        assert _HEADERS.get("User-Agent")

    def test_a_bad_entry_id_raises_rather_than_returning_empty(
        self, service: BitgetDataService
    ) -> None:
        """An empty result and a refused call are different facts. `market/skills.py` learned this
        the same way: a probe reporting a transport failure as "no data" measures the probe."""
        with pytest.raises(BitgetMcpError):
            service.query("equity_this_entry_does_not_exist", symbol="NVDA")

    def test_the_rtoken_to_underlying_mapping(self) -> None:
        assert underlying_of("NVDAUSDT") == "NVDA"
        assert underlying_of("tslausdt") == "TSLA"
        assert underlying_of("QQQUSDT") == "QQQ"
