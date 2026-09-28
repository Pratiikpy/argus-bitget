"""The out-of-band kill switch: a person's file stops every order at both venue doors
(research/harvest/27-resilience4j-breakers.md, decision 2)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from argus.execution import kill

NOW = datetime(2026, 9, 28, 12, tzinfo=UTC)


@pytest.fixture
def switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "KILL"
    monkeypatch.setenv(kill.ENV, str(target))
    monkeypatch.setattr(kill, "EVENTS", tmp_path / "kill_events.jsonl")
    return target


def _authorised() -> object:
    from argus.decision.verdicts import (
        ConstitutionVerdict,
        Intent,
        Side,
        Verdict,
        apply_constraint,
    )
    from argus.execution.orders import Order

    intent = Intent(symbol="NVDAUSDT", side=Side.BUY, quantity=Decimal("1"),
                    verdict=Verdict.TRADE, stated_confidence=0.8, thesis="t", invalidation=("x",))
    ruling = apply_constraint(intent, verdict=ConstitutionVerdict.ALLOW,
                              binding_constraint="none", reason="none")
    return ruling.authorise(Order(client_order_id="c-1", symbol="NVDAUSDT", side="BUY",
                                  quantity=Decimal("1"), approved_intent_hash="h"))


def test_engage_release_and_the_log(switch: Path) -> None:
    kill.engage("alice", "fills we did not send", events=kill.EVENTS, now=NOW)
    assert kill.engaged()["by"] == "alice"  # type: ignore[index]
    with pytest.raises(kill.KillSwitchEngaged, match="alice"):
        kill.check()
    kill.release("alice", "reconciled", events=kill.EVENTS, now=NOW)
    assert kill.engaged() is None
    kill.check()
    events = kill.EVENTS.read_text(encoding="utf-8").splitlines()
    assert [e.split('"event": "')[1].split('"')[0] for e in events] == ["engaged", "released"]


def test_an_engagement_must_say_who_and_why(switch: Path) -> None:
    with pytest.raises(ValueError):
        kill.engage("", "reason")
    with pytest.raises(ValueError):
        kill.engage("alice", "  ")


def test_an_unreadable_kill_file_still_stops(switch: Path) -> None:
    switch.write_bytes(b"\xff\xfe not json")
    with pytest.raises(kill.KillSwitchEngaged, match="unreadable"):
        kill.check()


def test_the_order_book_refuses_while_engaged(switch: Path) -> None:
    from argus.execution.orders import OrderBook

    kill.engage("alice", "stop", events=kill.EVENTS, now=NOW)
    with pytest.raises(kill.KillSwitchEngaged):
        OrderBook().submit(_authorised(), at=NOW)  # type: ignore[arg-type]
    kill.release("alice", "go", events=kill.EVENTS, now=NOW)
    assert OrderBook().submit(_authorised(), at=NOW).symbol == "NVDAUSDT"  # type: ignore[arg-type]


def test_the_venue_client_refuses_before_any_request(switch: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.execution.bitget_client import BitgetTradingClient

    for k, v in {"BITGET_API_KEY": "k", "BITGET_SECRET_KEY": "s", "BITGET_PASSPHRASE": "p"}.items():
        monkeypatch.setenv(k, v)
    client = BitgetTradingClient(paper_trading=True)
    monkeypatch.setattr(client, "_request", lambda *a, **k: pytest.fail("a request was made"))
    kill.engage("alice", "stop", events=kill.EVENTS, now=NOW)
    with pytest.raises(kill.KillSwitchEngaged):
        client.place_order(_authorised())  # type: ignore[arg-type]
