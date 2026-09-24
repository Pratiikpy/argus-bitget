"""The /agent page renders the Track 2 agent's own record and never invents one."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from argus.lui import agent_page

SUMMARY: dict[str, Any] = {
    "mode": "paper", "generated_at": "2026-09-24T19:00:38Z",
    "ledger": {"events": 41, "head_hash": "4c923b9013423cc9ed05",
               "first_ts": "2026-09-24T17:06:25Z",
               "genesis_hash": "ebbf607a8fe199bccc0f612ededf2a6902e4884a"},
    "metrics": {"sharpe_ann": None, "max_drawdown": -0.0042, "win_rate": 0.5,
                "n_closed_trades": 2, "total_return": 0.0013, "fees_paid": 3.1},
    "expected_envelope": {"sharpe_ann": "median -2.3, p05 -20.2, p95 +16.8"},
    "counts": {"decisions": 3, "orders_sent": 2, "fills": 2, "kernel_changed_decisions": 1},
}
DECISIONS: dict[str, Any] = {"decisions": [
    {"decided_at": "2026-09-24T19:30:02Z", "stance": "act", "seq": 40,
     "summary": "Crowd long, funding hot: fade.", "changed_by_kernel": True,
     "targets": [{"symbol": "BTCUSDT", "target": "short", "proposed_weight": 0.3,
                  "approved_weight": 0.2, "binding_guard": "gross_cap"}]},
    {"decided_at": "2026-09-24T20:00:00Z", "stance": "flat", "seq": 44,
     "summary": "<b>nothing</b>",
     "changed_by_kernel": False, "flat_reasons": ["no edge after costs"], "targets": []},
]}


def _fetch(files: Mapping[str, Any]) -> agent_page.Fetch:
    return lambda name: files.get(name)


def test_renders_the_record_metrics_envelope_and_decisions() -> None:
    html = agent_page.render(_fetch({"summary.json": SUMMARY, "decisions.json": DECISIONS}))
    assert "The agent that trades." in html
    assert "41 events" in html and "genesis ebbf607a8fe1" in html
    assert "-0.42%" in html and "50%" in html and "+0.13%" in html
    assert "undefined" in html  # a null Sharpe is printed as undefined, never as zero
    assert "median -2.3, p05 -20.2, p95 +16.8" in html
    assert "kernel changed it" in html and "bound by gross_cap" in html
    assert "proposed 30%, approved 20%" in html
    assert "Flat: no edge after costs" in html
    assert "<b>nothing</b>" not in html and "&lt;b&gt;nothing&lt;/b&gt;" in html
    # newest first
    assert html.index("seq 44") < html.index("seq 40")


def test_an_unreadable_record_shows_nothing_stale() -> None:
    html = agent_page.render(_fetch({}))
    assert "could not be read just now" in html
    assert "Scored so far" not in html


def test_no_decisions_yet_explains_the_heartbeats() -> None:
    html = agent_page.render(_fetch({"summary.json": SUMMARY}))
    assert "No decision yet" in html and "heartbeats" in html


def test_the_nav_links_the_page() -> None:
    from argus.lui import design

    assert ("/agent", "Live agent") in design.LINKS
    assert 'href="/agent" class="on"' in design.nav("/agent")
