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
    assert "undefined" not in html
    assert "n/a" in html  # a null Sharpe is printed as n/a with its reason, never as zero
    assert "median -2.3, p05 -20.2, p95 +16.8" in html
    assert "kernel changed it by up to 10.00 pt of weight" in html and "bound by gross_cap" in html
    assert "proposed 30.00%, approved 20.00%" in html
    # The logged reasons are kept, behind a control (first-user audit, 2026-09-29).
    assert "<details><summary>The logged reasons" in html and "no edge after costs" in html
    assert "<b>nothing</b>" not in html and "&lt;b&gt;nothing&lt;/b&gt;" in html
    # newest first
    assert html.index("seq 44") < html.index("seq 40")


def test_an_unreadable_record_shows_nothing_stale() -> None:
    html = agent_page.render(_fetch({}))
    assert "could not be read just now" in html
    assert "Scored so far" not in html
    # The disambiguation is part of the page's own frame, not the fetched body — it must survive
    # even when the record itself cannot be read.
    assert "separate project" in html


def test_no_decisions_yet_explains_the_heartbeats() -> None:
    html = agent_page.render(_fetch({"summary.json": SUMMARY}))
    assert "No decision yet" in html and "heartbeats" in html


def test_the_track3_nav_does_not_carry_the_track2_entry() -> None:
    """Each entry is an independent project (handbook, Basic Competition Rules 2): the page stays
    at its URL, but the Track 3 console's own navigation does not link the Track 2 agent."""
    from argus.lui import design

    assert all(path != "/agent" for path, _ in design.LINKS)
    assert 'href="/agent"' not in design.nav("/")


def test_the_page_says_plainly_it_is_a_separate_project_before_the_reader_can_confuse_it() -> None:
    """First-user audit, 2026-09-29: the headline "The agent that trades." on the Track 3
    console's own domain read as ARGUS itself placing orders. The h1 stays (it is kept content,
    not removed); a plain notice now sits directly under it."""
    html = agent_page.render(_fetch({"summary.json": SUMMARY, "decisions.json": DECISIONS}))
    assert "The agent that trades." in html  # unchanged
    assert "separate project" in html
    assert "same team" in html
    assert "Bitget's Demo (paper) environment" in html
    assert "no real money" in html
    assert "ARGUS" in html and "places no order on any venue" in html
    assert html.index("The agent that trades.") < html.index("separate project")


def test_the_no_edge_envelope_is_shown_in_the_agent_columns_units() -> None:
    # Return is kept in basis points and win rate as a fraction; both stood bare beside the
    # agent's percentages (a judge, round 20, row 716).
    assert agent_page._in_units("median -3.5, p05 -75.9, p95 +93.6", "%", 0.01) == (
        "median -0.04%, p05 -0.76%, p95 +0.94%")
    assert agent_page._in_units("median 0.45", "%", 100.0) == "median 45.0%"


def test_refused_orders_are_counted_and_explained() -> None:
    orders = {"counts": {"sent": 4}, "orders": [
        *({"symbol": "METAUSDT", "purpose": "protective_exit", "state": "rejected",
           "submitted_at": f"2026-09-28T09:{m:02d}:00Z",
           "rejection": {"message": "HTTP 400 from Bitget: Parameter METAUSDT_UMCBL does "
                                    "not exist"}}
          for m in (10, 20, 30, 40)),
        {"symbol": "METAUSDT", "purpose": "close", "state": "filled",
         "submitted_at": "2026-09-28T16:06:00Z"}]}
    count, note = agent_page.rejections(orders)
    assert count == 4
    assert "4 protective exits on METAUSDT" in note and "METAUSDT_UMCBL" in note
    assert "later order filled" in note and "sent the same one 4 times" in note
    html = agent_page.render(_fetch({"summary.json": SUMMARY, "decisions.json": DECISIONS,
                                     "orders.json": orders}))
    assert "refused by the venue" in html and "The venue refused 4 of the 4 orders" in html
