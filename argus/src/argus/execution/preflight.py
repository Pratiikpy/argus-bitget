"""Credential preflight — proves the signed path end to end the moment keys exist.

Everything in :mod:`argus.execution.bitget_client` is built and unit-tested, but a signature is only
genuinely correct when a venue accepts it. This module is the one command that establishes that,
and it is deliberately ordered so a failure names its own cause instead of returning a generic
signature error:

1. **credentials present** — which of the three are missing, by name
2. **public reachability** — is it the network, before blaming the signature
3. **signature accepted** — a private read (``/api/v2/mix/account/accounts``); this is the step
   that proves base64 digest + query-string-inclusive signing are both right
4. **demo routing** — confirms ``paptrading: 1`` reached the demo environment
5. **order round trip** — place the smallest possible order, read it back, cancel it

Step 5 places a real order **on the demo environment**. It is gated behind an explicit flag and
never runs by default, because "it defaulted to live" is not a mistake worth being one command away
from.

    python -m argus.execution.preflight            # steps 1-4, read-only
    python -m argus.execution.preflight --order    # adds step 5, demo only
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from argus.execution.bitget_client import (
    BitgetAuthError,
    BitgetOrderError,
    BitgetTradingClient,
    credentials_present,
)
from argus.execution.orders import Order, OrderBook, OrderState

# Bitget's own names, from agent-sdk/src/config.ts:192-194 and both official READMEs.
REQUIRED = ("BITGET_API_KEY", "BITGET_SECRET_KEY", "BITGET_PASSPHRASE")
# Accepted alternates, so a .env written either way works.
ALTERNATES = {
    "BITGET_SECRET_KEY": "BITGET_API_SECRET",
    "BITGET_PASSPHRASE": "BITGET_API_PASSPHRASE",
}


@dataclass(frozen=True, slots=True)
class Step:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"step": self.name, "passed": self.passed, "detail": self.detail}


def _credentials() -> Step:
    missing = [
        k for k in REQUIRED
        if not (os.environ.get(k) or os.environ.get(ALTERNATES.get(k, ""), ""))
    ]
    if missing:
        return Step(
            "credentials", False,
            f"missing {', '.join(missing)}. Copy .secrets/bitget.env.example to "
            f".secrets/bitget.env, fill it, then: set -a && . .secrets/bitget.env && set +a",
        )
    return Step("credentials", True, "all three present in the environment")


def _public_reachable() -> Step:
    """Check the venue answers at all, before any failure gets blamed on the signature."""
    from argus.market.bitget import BitgetError, fetch_rtokens

    try:
        got = fetch_rtokens()
    except BitgetError as exc:
        return Step("public_reachable", False, f"public endpoint failed: {exc}")
    return Step("public_reachable", True, f"{len(got)} rTokens quoting")


def _signature(client: BitgetTradingClient) -> Step:
    """The step that proves base64 digest and query-string signing are both right."""
    try:
        got = client.account()
    except BitgetAuthError as exc:
        return Step("signature", False, str(exc))
    except BitgetOrderError as exc:
        return Step("signature", False, f"request failed: {exc}")
    accounts = got.get("accounts") or []
    return Step("signature", True, f"accepted; {len(accounts)} account rows returned")


def _demo_routing(client: BitgetTradingClient) -> Step:
    headers = client._headers("GET", "/api/v2/mix/account/accounts", "", private=True)
    on = headers.get("paptrading") == "1"
    return Step(
        "demo_routing",
        on if client.is_paper else not on,
        f"paptrading header {'set' if on else 'absent'} in "
        f"{'paper' if client.is_paper else 'live'} mode",
    )


def _order_round_trip(client: BitgetTradingClient, *, symbol: str, size: Decimal) -> Step:
    """Place the smallest sensible order on the demo environment, read it, reconcile it."""
    if not client.is_paper:
        return Step("order_round_trip", False, "refusing to place a probe order outside paper mode")

    book = OrderBook()
    order = Order(
        client_order_id=f"argus-preflight-{int(size * 1000)}",
        symbol=symbol, side="BUY", quantity=size,
        approved_intent_hash="preflight-probe",
    )
    try:
        placed = client.place_order(order, order_type="market")
    except BitgetOrderError as exc:
        return Step("order_round_trip", False, f"place failed: {exc}")

    book.submit(order, at=datetime.now(UTC))
    try:
        state, filled = client.reconcile(order, symbol=symbol)
    except BitgetOrderError as exc:
        return Step(
            "order_round_trip", False,
            f"placed as {placed.venue_order_id} but reconcile failed: {exc}",
        )

    return Step(
        "order_round_trip",
        state is not OrderState.UNKNOWN,
        f"venue id {placed.venue_order_id}, state {state}, filled {filled}",
    )


def run(*, place_order: bool = False, symbol: str = "NVDAUSDT",
        size: Decimal = Decimal("0.01")) -> dict[str, Any]:
    steps: list[Step] = [_credentials()]

    if not steps[0].passed:
        return _report(steps, blocked=True)

    steps.append(_public_reachable())

    try:
        client = BitgetTradingClient(paper_trading=True)
    except BitgetAuthError as exc:
        steps.append(Step("client", False, str(exc)))
        return _report(steps, blocked=True)

    steps.append(_demo_routing(client))
    steps.append(_signature(client))

    if place_order and steps[-1].passed:
        steps.append(_order_round_trip(client, symbol=symbol, size=size))

    return _report(steps, blocked=False)


def _report(steps: list[Step], *, blocked: bool) -> dict[str, Any]:
    failed = [s for s in steps if not s.passed]
    return {
        "steps": [s.as_dict() for s in steps],
        "all_passed": not failed,
        "blocked_on_credentials": blocked,
        "first_failure": failed[0].name if failed else None,
        "next_step": (
            failed[0].detail if failed
            else "signed path verified end to end; the desk can place demo orders"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ARGUS Bitget credential preflight")
    parser.add_argument(
        "--order", action="store_true",
        help="also place, read back and reconcile one minimal DEMO order",
    )
    parser.add_argument("--symbol", default="NVDAUSDT")
    parser.add_argument("--size", default="0.01")
    args = parser.parse_args()

    result = run(place_order=args.order, symbol=args.symbol, size=Decimal(args.size))
    print(json.dumps(result, indent=2, default=str))

    for step in result["steps"]:
        mark = "PASS" if step["passed"] else "FAIL"
        print(f"  {mark}  {step['step']:<18} {step['detail'][:96]}")
    print(f"\n  -> {result['next_step']}")
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REQUIRED", "Step", "credentials_present", "run"]
