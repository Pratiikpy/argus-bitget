"""Bitget private trading client — the full signed path to a demo-environment order.

Every signing detail below was read from Bitget's own SDK rather than guessed, and two of them are
the kind that silently produce a 40009 signature error:

* **The digest is base64, not hex.** ``agent-sdk/src/utils/signature.ts:3-5`` —
  ``createHmac("sha256", secretKey).update(payload).digest("base64")``.
* **The signed endpoint includes the query string.** ``rest-client.ts:145`` builds
  ``endpoint = path?query`` and ``rest-client.ts:285`` signs
  ``timestamp + METHOD + endpoint + body``. Signing the bare path fails on every GET with params.
* Headers are ``ACCESS-KEY`` / ``ACCESS-SIGN`` / ``ACCESS-PASSPHRASE`` / ``ACCESS-TIMESTAMP``
  (``rest-client.ts:287-290``).
* **Paper trading is the header ``paptrading: 1`` on private endpoints only**
  (``rest-client.ts:274-280``). Public market data 404s when it is present, which is why
  :mod:`argus.market.bitget` never sends it.

**Paper trading is the default here and turning it off takes an explicit argument.** A client whose
default is real money is one typo from a real loss.

Credentials come from the environment only, using **Bitget's own variable names** —
``BITGET_API_KEY``, ``BITGET_SECRET_KEY``, ``BITGET_PASSPHRASE``
(``agent-sdk/src/config.ts:192-194`` and both official READMEs). An earlier draft of this module
invented ``BITGET_API_SECRET`` / ``BITGET_API_PASSPHRASE``; anyone following Bitget's own
documentation would have exported the real names and been told their credentials were missing.
The documented names are canonical and the invented ones are accepted as a fallback, so neither
spelling silently fails.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from argus.execution.orders import Order, OrderState

BASE_URL = "https://api.bitget.com"

LIVE_PRODUCT_TYPE = "USDT-FUTURES"
DEMO_PRODUCT_TYPE = "SUSDT-FUTURES"
"""Bitget's demo environment is a separate **productType**, not a header.

Found by probe, not by documentation: `GET /api/v2/mix/market/tickers?productType=SUSDT-FUTURES`
returns `SBTCSUSDT`, `SETHSUSDT`, `SXRPSUSDT` — three simulated instruments settling in `SUSDT`,
which is exactly what the web simulator trades. The `paptrading: 1` header belongs to Bitget's
*classic* demo mechanism and returns **40099 "exchange environment is incorrect"** on a v2 account:
three separate live keys were probed and all three behaved identically.

**The consequence for this project is not small.** Demo carries three crypto perpetuals and **no
rTokens**, so a demo order can never be a tokenized-equity order. Track 2's paper-trading log
therefore cannot be produced by routing real rToken decisions to this venue, and the hash-chained
internal ledger stays the system of record. That is a property of the venue, not a gap in the
build, and it is recorded here rather than discovered at submission.
"""

DEMO_SYMBOLS = ("SBTCSUSDT", "SETHSUSDT", "SXRPSUSDT")

MARGIN_COIN = {LIVE_PRODUCT_TYPE: "USDT", DEMO_PRODUCT_TYPE: "SUSDT"}
"""The demo environment settles in **SUSDT**, not USDT.

Found the same way as the productType: a demo order with `marginCoin=USDT` is refused with
**40778 "SBTCSUSDT does not support USDT currency as margin"**. Defaulting the margin coin to USDT
everywhere is the kind of hardcode that passes every unit test and fails the one live call.
"""


class BitgetAuthError(RuntimeError):
    """Credentials are absent or rejected. Never contains the secret."""


class BitgetOrderError(RuntimeError):
    """The venue refused an order. Carries the venue's own code so it can be acted on."""


@dataclass(frozen=True, slots=True)
class PlacedOrder:
    """What the venue said it did."""

    client_order_id: str
    venue_order_id: str
    raw: dict[str, Any]


def sign(payload: str, secret: str) -> str:
    """HMAC-SHA256, base64-encoded.

    Base64 rather than hex. Bitget returns a generic signature error for a hex digest, which is
    indistinguishable from a wrong secret and costs an afternoon to diagnose.
    """
    digest = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


class BitgetTradingClient:
    """Signed private client. Paper trading unless explicitly disabled."""

    def __init__(
        self,
        *,
        paper_trading: bool = True,
        base_url: str = BASE_URL,
        timeout: float = 30.0,
        product_type: str | None = None,
        legacy_paptrading: bool = False,
    ) -> None:
        self._key = _from_env("BITGET_API_KEY")
        self._secret = _from_env("BITGET_SECRET_KEY", "BITGET_API_SECRET")
        self._passphrase = _from_env("BITGET_PASSPHRASE", "BITGET_API_PASSPHRASE")

        missing = [
            name for name, value in (
                ("BITGET_API_KEY", self._key),
                ("BITGET_SECRET_KEY", self._secret),
                ("BITGET_PASSPHRASE", self._passphrase),
            ) if not value
        ]
        if missing:
            raise BitgetAuthError(
                f"missing {', '.join(missing)} in the environment. Put them in "
                f".secrets/bitget.env and load it; never pass them as literals. "
                f"Everything else in this client is built and tested."
            )

        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._paper = paper_trading
        # Demo is selected by productType. `legacy_paptrading` re-enables the classic header for
        # an account that still uses it; on a v2 account it produces 40099, so it is off by
        # default and turning it on is a deliberate, named choice.
        self._legacy_paptrading = legacy_paptrading
        self._product_type = product_type or (
            DEMO_PRODUCT_TYPE if paper_trading else LIVE_PRODUCT_TYPE
        )
        self._pos_mode: str | None = None

    def __repr__(self) -> str:
        mode = "PAPER" if self._paper else "LIVE"
        return f"BitgetTradingClient(mode={mode}, product_type={self._product_type!r})"

    @property
    def product_type(self) -> str:
        return self._product_type

    @property
    def margin_coin(self) -> str:
        """The settlement currency this environment actually uses.

        Derived from the productType rather than defaulted, because the two must agree and a
        caller should not have to know that demo settles in SUSDT.
        """
        return MARGIN_COIN.get(self._product_type, "USDT")

    @property
    def trades_real_money(self) -> bool:
        """The single question that matters before any order goes out.

        True only when the client is live *and* pointed at a real productType. Kept as one
        property so a caller never has to reason about two flags agreeing.
        """
        return not self._paper and self._product_type != DEMO_PRODUCT_TYPE

    @property
    def is_paper(self) -> bool:
        return self._paper

    # --- signing ---------------------------------------------------------------------------

    def _headers(self, method: str, endpoint: str, body: str, *, private: bool) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "locale": "en-US",
        }
        if not private:
            return headers

        # Paper trading routes to the demo environment via this header, and ONLY on private
        # endpoints — public market data 404s when it is present.
        if self._paper and self._legacy_paptrading:
            headers["paptrading"] = "1"

        timestamp = str(int(time.time() * 1000))
        payload = f"{timestamp}{method.upper()}{endpoint}{body}"
        headers.update({
            "ACCESS-KEY": self._key,
            "ACCESS-SIGN": sign(payload, self._secret),
            "ACCESS-PASSPHRASE": self._passphrase,
            "ACCESS-TIMESTAMP": timestamp,
        })
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
        private: bool = True,
    ) -> Any:
        query = urllib.parse.urlencode(params) if params else ""
        endpoint = f"{path}?{query}" if query else path
        body_json = json.dumps(body, separators=(",", ":")) if body else ""

        req = urllib.request.Request(
            f"{self._base_url}{endpoint}",
            data=body_json.encode() if body_json else None,
            headers=self._headers(method, endpoint, body_json, private=private),
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            # Bitget returns its own error code in a JSON body alongside a 4xx status. Raising on
            # the status alone throws that away: a probe with a fabricated key came back as
            # "HTTP 400" when the body said 40037 "Apikey does not exist" — which is the whole
            # diagnostic. Parse the body first and route on the venue's code.
            raw = exc.read().decode(errors="replace")
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                raise BitgetOrderError(f"HTTP {exc.code}: {raw[:300]}") from None
            self._raise_for_code(body, http_status=exc.code)
            raise BitgetOrderError(f"HTTP {exc.code}: {raw[:300]}") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            raise BitgetOrderError(f"transport failure: {exc}") from None

        self._raise_for_code(payload)
        return payload.get("data")

    def _raise_for_code(self, payload: dict[str, Any], *, http_status: int | None = None) -> None:
        """Route on Bitget's own error code, which is more informative than the HTTP status.

        The distinction that matters when diagnosing a failed setup:

        * **40037** "Apikey does not exist" — the request, path, headers and demo routing were all
          fine; the key itself is unknown. Signing was never reached.
        * **40009** signature error — the key exists and the signature did not match. *This* is the
          one that means the digest encoding or the signed path is wrong.
        * **40099** "exchange environment is incorrect" — the key is real and the signature was
          never even consulted; the key simply belongs to the *other* environment. A live key sent
          with ``paptrading: 1``, or a demo key sent without it.

        Collapsing these into "auth failed" sends someone hunting a signing bug when they have a
        typo'd key, or the reverse.

        **40099 was found by live probe, not by reading the docs.** Two real live keys returned
        ``40012`` against production and ``40099`` against demo routing, which is what proved they
        were live keys rather than demo ones. Without this branch the message would have been the
        generic "venue code 40099" and the diagnosis would have been a guess.
        """
        code = str(payload.get("code", ""))
        if code in ("00000", "0", ""):
            return
        msg = payload.get("msg", "")

        if code == "40037":
            raise BitgetAuthError(
                f"the API key is not recognised by Bitget (40037: {msg}). The request reached the "
                f"venue and its headers parsed, so the path and demo routing are fine — check the "
                f"key value itself, and that it was created on the account you expect."
            )
        if code in ("40009", "40012"):
            raise BitgetAuthError(
                f"signature rejected ({code}: {msg}). The key exists, so this is the signing "
                f"itself: the digest must be base64 (not hex) and the signed path must include "
                f"the query string."
            )
        if code == "40099":
            wanted = "a DEMO key" if self._paper else "a LIVE key"
            got = "live" if self._paper else "demo"
            raise BitgetAuthError(
                f"wrong environment for this key ({code}: {msg}). The key is real and reached the "
                f"venue, but it belongs to the {got} environment. This client is configured for "
                f"{wanted}. A demo key is a separate key created while the account is in Demo "
                f"mode — it is not a permission toggle on a live key."
            )
        if code in ("40010", "40014"):
            raise BitgetAuthError(
                f"timestamp or permission rejected ({code}: {msg}). Check the clock is in sync "
                f"and the key carries Trade permission."
            )
        status = f"HTTP {http_status}, " if http_status else ""
        raise BitgetOrderError(f"{status}venue code {code}: {msg}")

    # --- the operations the desk needs -------------------------------------------------------

    def account(self, margin_coin: str | None = None) -> dict[str, Any]:
        """Account state. The first call to make — it proves the signature works."""
        got = self._request(
            "GET", "/api/v2/mix/account/accounts",
            params={"productType": self._product_type.lower(),
                    "marginCoin": margin_coin or self.margin_coin},
        )
        return {"accounts": got, "mode": "paper" if self._paper else "live"}

    def position_mode(self, symbol: str) -> str:
        """``one_way_mode`` or ``hedge_mode``, read from the venue and cached.

        **Asked rather than assumed.** A hedge-mode account refuses an order with no ``tradeSide``
        (**40774** "The order type for unilateral position must also be the unilateral position
        type"), and a one-way account refuses one that has it. Hardcoding either is a coin flip
        that only fails on the live call, which is exactly how it was found here.
        """
        if self._pos_mode is None:
            got = self._request(
                "GET", "/api/v2/mix/account/account",
                params={"symbol": symbol, "productType": self._product_type.lower(),
                        "marginCoin": self.margin_coin},
            ) or {}
            self._pos_mode = str(got.get("posMode") or "one_way_mode")
        return self._pos_mode

    def place_order(
        self,
        order: Order,
        *,
        margin_coin: str | None = None,
        order_type: str = "market",
        price: Decimal | None = None,
        trade_side: str = "open",
    ) -> PlacedOrder:
        """Send an approved order.

        Refuses anything without an ``approved_intent_hash``. An order that cannot be traced to a
        Constitution verdict must not reach a venue, however well-formed it looks — and this is the
        last place that can be enforced.
        """
        if not order.approved_intent_hash.strip():
            raise BitgetOrderError(
                f"{order.client_order_id} carries no approved_intent_hash; refusing to send an "
                f"order that cannot be traced to a Constitution verdict"
            )
        if order_type == "limit" and price is None:
            raise BitgetOrderError("a limit order requires a price")

        body: dict[str, Any] = {
            "symbol": order.symbol,
            "productType": self._product_type.lower(),
            "marginMode": "isolated",
            "marginCoin": margin_coin or self.margin_coin,
            "size": str(order.quantity),
            "side": order.side.lower(),
            "orderType": order_type,
            "clientOid": order.client_order_id,
        }
        if price is not None:
            body["price"] = str(price)

        # Hedge mode needs to be told whether this opens or closes; one-way mode refuses the field
        # outright. The venue is the authority on which this account is.
        if self.position_mode(order.symbol) == "hedge_mode":
            if trade_side not in ("open", "close"):
                raise BitgetOrderError(
                    f"trade_side={trade_side!r} must be 'open' or 'close' on a hedge-mode account"
                )
            body["tradeSide"] = trade_side

        got = self._request("POST", "/api/v2/mix/order/place-order", body=body) or {}
        return PlacedOrder(
            client_order_id=order.client_order_id,
            venue_order_id=str(got.get("orderId", "")),
            raw=got,
        )

    def order_status(self, *, symbol: str, client_order_id: str) -> dict[str, Any]:
        """What the venue holds. The only legitimate way out of UNKNOWN."""
        got = self._request(
            "GET", "/api/v2/mix/order/detail",
            params={
                "symbol": symbol,
                "productType": self._product_type.lower(),
                "clientOid": client_order_id,
            },
        )
        return got or {}

    def reconcile(self, book_order: Order, *, symbol: str) -> tuple[OrderState, Decimal]:
        """Map the venue's state onto ours.

        An unrecognised venue state maps to UNKNOWN rather than to a guess. Assuming a state we do
        not recognise is exactly how a position stops being watched.
        """
        detail = self.order_status(symbol=symbol, client_order_id=book_order.client_order_id)
        venue_state = str(detail.get("state", "")).lower()
        filled = Decimal(str(detail.get("baseVolume", "0") or "0"))

        mapping = {
            "live": OrderState.ACCEPTED,
            "new": OrderState.ACCEPTED,
            "partially_filled": OrderState.PARTIALLY_FILLED,
            "filled": OrderState.FILLED,
            "cancelled": OrderState.CANCELLED,
            "canceled": OrderState.CANCELLED,
            "rejected": OrderState.REJECTED,
            "expired": OrderState.EXPIRED,
        }
        return mapping.get(venue_state, OrderState.UNKNOWN), filled


def _from_env(canonical: str, *fallbacks: str) -> str:
    """Read a credential, preferring Bitget's own variable name.

    Fallbacks exist because this module briefly used invented names; accepting both means a
    ``.env`` written against either spelling works rather than reporting a credential as missing
    when it is sitting right there under another name.
    """
    for name in (canonical, *fallbacks):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def credentials_present() -> bool:
    """Can a private client be built at all? Used to skip live tests cleanly."""
    return bool(
        _from_env("BITGET_API_KEY")
        and _from_env("BITGET_SECRET_KEY", "BITGET_API_SECRET")
        and _from_env("BITGET_PASSPHRASE", "BITGET_API_PASSPHRASE")
    )
