"""Bitget trading client tests.

The signing tests are the important ones: both details verified below are the kind that produce a
generic signature error indistinguishable from a wrong secret.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from decimal import Decimal

import pytest

from argus.execution.bitget_client import (
    BitgetAuthError,
    BitgetOrderError,
    BitgetTradingClient,
    credentials_present,
    sign,
)
from argus.execution.orders import Order

# Bitget's own names — agent-sdk/src/config.ts:192-194 and both official READMEs.
CREDS = {
    "BITGET_API_KEY": "test-key",
    "BITGET_SECRET_KEY": "test-secret",
    "BITGET_PASSPHRASE": "test-pass",
}
ALTERNATE_CREDS = {
    "BITGET_API_KEY": "test-key",
    "BITGET_API_SECRET": "test-secret",
    "BITGET_API_PASSPHRASE": "test-pass",
}


@pytest.fixture
def client(monkeypatch) -> BitgetTradingClient:
    for k, v in CREDS.items():
        monkeypatch.setenv(k, v)
    return BitgetTradingClient()


class TestSigning:
    def test_digest_is_base64_not_hex(self) -> None:
        """agent-sdk/src/utils/signature.ts:3-5 — .digest("base64").

        A hex digest returns a generic signature error that looks exactly like a wrong secret.
        """
        got = sign("1700000000GET/api/v2/mix/account/accounts", "secret")
        expected = base64.b64encode(
            hmac.new(b"secret", b"1700000000GET/api/v2/mix/account/accounts",
                     hashlib.sha256).digest()
        ).decode()
        assert got == expected
        assert got != hmac.new(b"secret", b"1700000000GET/api/v2/mix/account/accounts",
                               hashlib.sha256).hexdigest()

    def test_signature_is_deterministic(self) -> None:
        assert sign("payload", "k") == sign("payload", "k")

    def test_different_payloads_sign_differently(self) -> None:
        assert sign("a", "k") != sign("b", "k")


class TestHeaders:
    def test_private_requests_carry_all_four_access_headers(self, client) -> None:
        h = client._headers("GET", "/api/v2/mix/account/accounts", "", private=True)
        for key in ("ACCESS-KEY", "ACCESS-SIGN", "ACCESS-PASSPHRASE", "ACCESS-TIMESTAMP"):
            assert key in h

    def test_paper_trading_routes_by_product_type_not_header(self, client) -> None:
        """Corrected against the venue, not the SDK docs.

        `rest-client.ts:274-280` sends `paptrading: 1` for demo, and that is Bitget's *classic*
        mechanism. On a v2 account it returns **40099 "exchange environment is incorrect"** — three
        separate real keys were probed and all three behaved identically, while the public endpoint
        showed demo living at `productType=SUSDT-FUTURES`. So paper mode routes by productType and
        sends no header.
        """
        from argus.execution.bitget_client import DEMO_PRODUCT_TYPE

        assert client.product_type == DEMO_PRODUCT_TYPE
        assert "paptrading" not in client._headers("GET", "/x", "", private=True)
        assert client.trades_real_money is False

    def test_the_classic_header_is_available_but_must_be_asked_for(self, monkeypatch) -> None:
        """Kept for any account still on the classic mechanism — off unless named."""
        for k, v in CREDS.items():
            monkeypatch.setenv(k, v)
        legacy = BitgetTradingClient(paper_trading=True, legacy_paptrading=True)
        assert legacy._headers("GET", "/x", "", private=True)["paptrading"] == "1"

    def test_live_mode_trades_real_money_and_paper_does_not(self, monkeypatch) -> None:
        """One property, so a caller never has to reason about two flags agreeing."""
        for k, v in CREDS.items():
            monkeypatch.setenv(k, v)
        assert BitgetTradingClient(paper_trading=False).trades_real_money is True
        assert BitgetTradingClient(paper_trading=True).trades_real_money is False

    def test_demo_carries_no_rtokens(self, monkeypatch) -> None:
        """A venue property with a real consequence for Track 2.

        Demo has three crypto perpetuals and no tokenized equities, so a demo order can never be
        an rToken order. The hash-chained internal ledger stays the system of record, and this
        test exists so that stops being a surprise.
        """
        from argus.execution.bitget_client import DEMO_SYMBOLS

        assert DEMO_SYMBOLS == ("SBTCSUSDT", "SETHSUSDT", "SXRPSUSDT")
        assert not any("NVDA" in s or "TSLA" in s for s in DEMO_SYMBOLS)

    def test_paper_trading_header_is_absent_on_public_requests(self, client) -> None:
        """Public endpoints 404 when it is present — the SDK documents this explicitly."""
        assert "paptrading" not in client._headers("GET", "/x", "", private=False)

    def test_live_mode_sends_no_paper_header(self, monkeypatch) -> None:
        for k, v in CREDS.items():
            monkeypatch.setenv(k, v)
        live = BitgetTradingClient(paper_trading=False)
        assert "paptrading" not in live._headers("GET", "/x", "", private=True)

    def test_the_signed_endpoint_includes_the_query_string(self, client) -> None:
        """rest-client.ts:145 builds endpoint = path?query, and :285 signs it.

        Signing the bare path fails on every GET that carries parameters.
        """
        with_query = client._headers("GET", "/api/v2/x?a=1&b=2", "", private=True)
        bare = client._headers("GET", "/api/v2/x", "", private=True)
        assert with_query["ACCESS-SIGN"] != bare["ACCESS-SIGN"]


class TestSafetyDefaults:
    def test_paper_trading_is_the_default(self, client) -> None:
        """A client whose default is real money is one typo from a real loss."""
        assert client.is_paper is True

    def test_repr_shows_the_mode_and_hides_the_secret(self, client) -> None:
        assert "PAPER" in repr(client)
        assert "test-secret" not in repr(client)

    def test_missing_credentials_name_exactly_what_is_missing(self, monkeypatch) -> None:
        for k in (*CREDS, *ALTERNATE_CREDS):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("BITGET_API_KEY", "present")
        with pytest.raises(BitgetAuthError) as exc:
            BitgetTradingClient()
        assert "BITGET_SECRET_KEY" in str(exc.value)
        assert "BITGET_PASSPHRASE" in str(exc.value)

    def test_bitgets_documented_names_are_what_the_error_asks_for(self, monkeypatch) -> None:
        """Following Bitget's own README must not produce a 'missing credentials' error."""
        for k in (*CREDS, *ALTERNATE_CREDS):
            monkeypatch.delenv(k, raising=False)
        for k, v in CREDS.items():
            monkeypatch.setenv(k, v)
        assert BitgetTradingClient().is_paper is True

    def test_the_invented_names_still_work_as_a_fallback(self, monkeypatch) -> None:
        """A .env written against the earlier draft must not silently fail."""
        for k in (*CREDS, *ALTERNATE_CREDS):
            monkeypatch.delenv(k, raising=False)
        for k, v in ALTERNATE_CREDS.items():
            monkeypatch.setenv(k, v)
        assert credentials_present() is True


class TestOrderGuards:
    def test_an_unauthorised_order_never_reaches_the_venue(self, client) -> None:
        """The last place this can be enforced."""
        order = Order("c-1", "NVDAUSDT", "SELL", Decimal("1"), "hash")
        object.__setattr__(order, "approved_intent_hash", "   ")
        with pytest.raises(BitgetOrderError, match="Constitution verdict"):
            client.place_order(order)

    def test_a_limit_order_without_a_price_is_refused(self, client) -> None:
        order = Order("c-2", "NVDAUSDT", "SELL", Decimal("1"), "hash-ok")
        with pytest.raises(BitgetOrderError, match="requires a price"):
            client.place_order(order, order_type="limit")


class TestCredentialDetection:
    def test_detects_absence(self, monkeypatch) -> None:
        for k in (*CREDS, *ALTERNATE_CREDS):
            monkeypatch.delenv(k, raising=False)
        assert credentials_present() is False

    def test_detects_presence(self, monkeypatch) -> None:
        for k, v in CREDS.items():
            monkeypatch.setenv(k, v)
        assert credentials_present() is True


def _router(*, paper: bool = True):
    """A client whose only purpose is to expose the error router.

    `_raise_for_code` became an instance method when it gained the 40099 branch: telling a live key
    from a demo one requires knowing which environment *this client* is configured for, and a
    staticmethod cannot know that.
    """
    import os

    from argus.execution.bitget_client import BitgetTradingClient

    for name, value in (
        ("BITGET_API_KEY", "probe-key"),
        ("BITGET_SECRET_KEY", "probe-secret"),
        ("BITGET_PASSPHRASE", "probe-pass"),
    ):
        os.environ[name] = value
    return BitgetTradingClient(paper_trading=paper)


class TestVenueErrorRouting:
    """Bitget's own code is more informative than the HTTP status, and the distinction decides
    where someone looks when their setup fails.

    Verified against the live venue on 2026-09-12 with a fabricated key: the response was
    HTTP 400 carrying {"code":"40037","msg":"Apikey does not exist"}. Raising on the status alone
    reported "HTTP 400" and threw the diagnostic away.
    """

    def test_an_unknown_key_is_not_reported_as_a_signing_problem(self) -> None:
        """40037 means signing was never reached, so pointing at the digest is wrong."""
        from argus.execution.bitget_client import BitgetAuthError

        with pytest.raises(BitgetAuthError) as exc:
            _router()._raise_for_code(
                {"code": "40037", "msg": "Apikey does not exist"}, http_status=400
            )
        text = str(exc.value)
        assert "not recognised" in text
        assert "path and demo routing are fine" in text
        assert "base64" not in text, "an unknown key must not point at the digest encoding"

    def test_a_signature_rejection_does_point_at_the_digest(self) -> None:
        """40009 means the key exists and the signature did not match — now it IS the signing."""
        from argus.execution.bitget_client import BitgetAuthError

        with pytest.raises(BitgetAuthError, match="base64"):
            _router()._raise_for_code({"code": "40009", "msg": "sign error"})

    def test_a_clock_or_permission_error_is_named_as_such(self) -> None:
        from argus.execution.bitget_client import BitgetAuthError

        with pytest.raises(BitgetAuthError, match="clock is in sync"):
            _router()._raise_for_code({"code": "40010", "msg": "timestamp expired"})

    def test_success_codes_pass_through(self) -> None:

        for code in ("00000", "0", ""):
            _router()._raise_for_code({"code": code})

    def test_an_unrecognised_code_keeps_the_status_and_message(self) -> None:
        from argus.execution.bitget_client import BitgetOrderError

        with pytest.raises(BitgetOrderError, match="43012"):
            _router()._raise_for_code(
                {"code": "43012", "msg": "insufficient balance"}, http_status=400
            )

    def test_a_live_key_on_demo_routing_names_the_environment(self) -> None:
        """40099, found by live probe: two real keys returned it under `paptrading: 1`.

        That is what proved they were live keys rather than demo ones — and without this branch the
        message would have been a bare "venue code 40099" and the diagnosis a guess.
        """
        from argus.execution.bitget_client import BitgetAuthError

        with pytest.raises(BitgetAuthError, match="belongs to the live environment"):
            _router(paper=True)._raise_for_code(
                {"code": "40099", "msg": "exchange environment is incorrect"}
            )

    def test_a_demo_key_on_live_routing_names_it_the_other_way(self) -> None:
        from argus.execution.bitget_client import BitgetAuthError

        with pytest.raises(BitgetAuthError, match="belongs to the demo environment"):
            _router(paper=False)._raise_for_code(
                {"code": "40099", "msg": "exchange environment is incorrect"}
            )

    def test_40099_says_a_demo_key_is_a_separate_key(self) -> None:
        """The misconception that cost real time: it is not a toggle on a live key."""
        from argus.execution.bitget_client import BitgetAuthError

        with pytest.raises(BitgetAuthError, match="not a permission toggle"):
            _router(paper=True)._raise_for_code({"code": "40099", "msg": "wrong env"})
