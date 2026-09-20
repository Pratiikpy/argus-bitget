"""Preflight tests — the last mile before a real demo order.

The point of preflight is that a failure names its own cause. These assert that ordering and
messaging, because a generic "signature error" is the thing it exists to prevent.
"""

from __future__ import annotations

from decimal import Decimal

from argus.execution.preflight import REQUIRED, run

CREDS = {
    "BITGET_API_KEY": "k",
    "BITGET_SECRET_KEY": "s",
    "BITGET_PASSPHRASE": "p",
}
ALL_NAMES = (*REQUIRED, "BITGET_API_SECRET", "BITGET_API_PASSPHRASE")


class TestWithoutCredentials:
    def test_stops_at_the_first_step_and_says_which_are_missing(self, monkeypatch) -> None:
        for k in ALL_NAMES:
            monkeypatch.delenv(k, raising=False)
        got = run()
        assert got["blocked_on_credentials"] is True
        assert got["first_failure"] == "credentials"
        for k in REQUIRED:
            assert k in got["next_step"]

    def test_names_only_the_missing_ones(self, monkeypatch) -> None:
        """Reporting a credential you already set as missing sends you hunting the wrong thing."""
        for k in ALL_NAMES:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("BITGET_API_KEY", "present")
        got = run()
        assert "BITGET_SECRET_KEY" in got["next_step"]
        assert "BITGET_API_KEY" not in got["next_step"]

    def test_does_not_attempt_a_network_call(self, monkeypatch) -> None:
        """No point asking a venue anything when we have nothing to sign with."""
        for k in ALL_NAMES:
            monkeypatch.delenv(k, raising=False)
        assert len(run()["steps"]) == 1

    def test_gives_a_runnable_next_step(self, monkeypatch) -> None:
        for k in ALL_NAMES:
            monkeypatch.delenv(k, raising=False)
        step = run()["next_step"]
        assert ".secrets/bitget.env.example" in step
        assert "set -a" in step


class TestOrderGating:
    def test_a_probe_order_never_runs_by_default(self, monkeypatch) -> None:
        """'It defaulted to live' is not a mistake worth being one command away from."""
        for k, v in CREDS.items():
            monkeypatch.setenv(k, v)

        import argus.execution.preflight as pf
        from argus.execution.preflight import Step

        monkeypatch.setattr(pf, "_public_reachable", lambda: Step("public_reachable", True, "stub"))
        monkeypatch.setattr(pf, "_signature", lambda c: Step("signature", True, "stub"))

        placed: list[bool] = []
        monkeypatch.setattr(
            pf, "_order_round_trip",
            lambda c, **kw: (placed.append(True), Step("order_round_trip", True, "stub"))[1],
        )

        run()
        assert placed == []
        run(place_order=True)
        assert placed == [True]

    def test_probe_order_is_refused_outside_paper_mode(self, monkeypatch) -> None:
        for k, v in CREDS.items():
            monkeypatch.setenv(k, v)
        from argus.execution.bitget_client import BitgetTradingClient
        from argus.execution.preflight import _order_round_trip

        live = BitgetTradingClient(paper_trading=False)
        got = _order_round_trip(live, symbol="NVDAUSDT", size=Decimal("0.01"))
        assert got.passed is False
        assert "outside paper mode" in got.detail


class TestDemoRouting:
    def test_paper_mode_points_at_the_demo_product_type(self, monkeypatch) -> None:
        for k, v in CREDS.items():
            monkeypatch.setenv(k, v)
        from argus.execution.bitget_client import BitgetTradingClient
        from argus.execution.preflight import _demo_routing

        got = _demo_routing(BitgetTradingClient(paper_trading=True))
        assert got.passed is True
        assert "SUSDT-FUTURES" in got.detail
        assert "real money: False" in got.detail
