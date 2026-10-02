"""``/agent`` — the Track 2 agent's paper run, read live from its own public record.

ARGUS is the research desk; `t2-sentiment-agent` is the entry that trades. They are separate
projects on purpose — the agent's run is pinned to its own genesis commit and must not change
because the desk did — but a judge reading one should be able to see the other without leaving.

**Nothing on this page is computed here.** The agent publishes its record every hour, computed from
its hash-chained log (`t2-sentiment-agent/src/sentiment_agent/site/export.py`), to
:data:`RECORD`. This page fetches ``summary.json`` and ``decisions.json`` from there and renders
them. A figure this page re-derived could disagree with the record it summarises; a figure it only
reads cannot. When the record cannot be fetched the page says so and shows nothing stale.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from html import escape
from typing import Any

from argus.lui import design
from argus.lui.agent_record import (  # noqa: F401 — re-exported for the page's callers and tests
    CACHE_SECONDS,
    RECORD,
    REPO,
    RUN_1_RECORD,
    Fetch,
    fetch_json,
)


def _num(value: Any, fmt: str, missing: str = "n/a") -> str:
    if value is None:
        return missing
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return missing


def _metrics(metrics: Mapping[str, Any], envelope: Mapping[str, Any]) -> str:
    # A ratio over closed trades has no value before the first one closes; the page printed
    # "undefined", which reads as a bug rather than as the fact (readiness backlog L43).
    closed = metrics.get("n_closed_trades", 0)
    per_trade = f"n/a — {closed} closed trades" if not closed else "n/a"
    rows = (
        ("Sharpe (annualised)", _num(metrics.get("sharpe_ann"), ".2f", per_trade),
         envelope.get("sharpe_ann")),
        ("Max drawdown", _num(metrics.get("max_drawdown"), ".2%"),
         envelope.get("max_drawdown_pct")),
        ("Win rate", _num(metrics.get("win_rate"), ".0%", per_trade), envelope.get("win_rate")),
        ("Closed trades", str(metrics.get("n_closed_trades", 0)), envelope.get("closed_trades")),
        ("Return", _num(metrics.get("total_return"), "+.2%"),
         envelope.get("return_on_equity_bps")),
        ("Fees paid (USDT)", _num(metrics.get("fees_paid"), ".2f"), None),
    )
    body = "".join(
        f"<tr><td>{escape(label)}</td><td class='n'>{escape(value)}</td>"
        f"<td class='env'>{escape(str(env)) if env else '—'}</td></tr>"
        for label, value, env in rows)
    # "Win rate 0%" on three closed trades stood with nothing beside it (a first-time user, round
    # 10): on a handful of trades one more moves it by tens of points, and the Sharpe is read from
    # the whole book's equity, open positions included, so the two can point opposite ways.
    few = ""
    if isinstance(closed, int) and 0 < closed < SAMPLE_TO_READ:
        few = (f"<p class='dim few'>On {closed} closed trade{'' if closed == 1 else 's'} these "
               f"figures say little yet: one more trade moves the win rate by "
               f"{100 / (closed + 1):.0f} points or more. The Sharpe is read from the whole "
               f"book's equity, open positions included; the win rate from closed trades "
               f"alone.</p>")
    return ("<div class='tw'><table class='m'><thead><tr><th>Metric</th><th>Agent</th>"
            "<th>A no-edge book over the same window</th></tr></thead><tbody>" + body
            + "</tbody></table></div>" + few)


SAMPLE_TO_READ = 30
"""Closed trades below which the page says the figures are too few to read."""


def _funnel(counts: Mapping[str, Any]) -> str:
    order = (("decisions", "decisions"), ("flat_with_reasons", "flat, with reasons"),
             ("kernel_changed_decisions", "changed by the risk kernel"),
             ("orders_sent", "orders sent"), ("fills", "fills"),
             ("closed_trades", "closed trades"), ("protective_rulings", "protective rulings"))
    return "".join(f"<div class='stat'><b>{escape(str(counts.get(key, 0)))}</b>"
                   f"<span>{escape(label)}</span></div>" for key, label in order)


_DRAWDOWN_SAID = re.compile(
    r"drawdown\s+(?:beyond|above|over|past|exceed\w*)\s+(\d+(?:\.\d+)?)\s*%", re.I)
_DRAWDOWN_HELD = re.compile(r"book\.drawdown_pct\s*=\s*(-?\d+(?:\.\d+)?)")


def narration_check(row: Mapping[str, Any]) -> str | None:
    """A drawdown the agent's summary claims, checked against the drawdown its own record holds.

    Seq 2638 says "Circuit breaker active (4 consecutive losses, drawdown beyond 2.5%)" while the
    same row's facts read `book.drawdown_pct = -0.0479769`: the breaker tripped on the losing
    streak alone, and the narration overstated the drawdown fifty-fold (a judge, round 19, row
    664). The agent's run is not edited mid-window; the page says what the record shows."""
    said = _DRAWDOWN_SAID.search(str(row.get("summary") or ""))
    held = _DRAWDOWN_HELD.search(str(row.get("mandate_response") or ""))
    if said is None or held is None:
        return None
    claimed, actual = float(said.group(1)), abs(float(held.group(1)))
    if actual >= claimed:
        return None
    return (f"Narration check: the summary says the drawdown is beyond {claimed:g}%, but the "
            f"book's own record for this decision holds {actual:.3g}% — under that line. The "
            f"breaker's other trip, the losing streak, is what the record supports.")


def _decision(row: Mapping[str, Any]) -> str:
    targets = row.get("targets") or []
    legs = "; ".join(
        f"{t.get('symbol')} {t.get('target')} "
        f"(proposed {_num(t.get('proposed_weight'), '.0%', '—')}, "
        f"approved {_num(t.get('approved_weight'), '.0%', '—')}"
        + (f", bound by {t.get('binding_guard')}" if t.get("binding_guard") else "") + ")"
        for t in targets)
    reasons = "; ".join(str(r) for r in row.get("flat_reasons") or [])
    kernel = row.get("changed_by_kernel")
    pill = ("<span class='pill red'>kernel changed it</span>" if kernel
            else "<span class='pill'>kernel passed it</span>" if kernel is not None else "")
    # Targets read plainly and stay in view; the logged reasons for standing flat are the
    # kernel's own field names and values, so they sit behind a control rather than in the prose
    # (first-user audit, 2026-09-29: "raw telemetry with no plain summary").
    detail = (f"<p class='dim'>{escape(legs)}</p>" if legs else
              f"<details><summary>The logged reasons, as the kernel records them</summary>"
              f"<p class='dim mono'>{escape(reasons)}</p></details>" if reasons else "")
    when = escape(str(row.get("decided_at", ""))[:16])
    stance = escape(str(row.get("stance") or row.get("outcome")).replace("_", " "))
    seq = escape(str(row.get("seq", "—")))
    return (f"<li><div class='dh'><span class='mono'>{when}Z</span>"
            f"<span class='pill proof'>{stance}</span>{pill}"
            f"<span class='mono dim'>seq {seq}</span></div>"
            f"<p>{escape(str(row.get('summary') or ''))}</p>"
            + (f"<p class='dim'>{escape(note)}</p>" if (note := narration_check(row)) else "")
            + detail + "</li>")


def render(fetch: Fetch = fetch_json) -> str:
    summary = fetch("summary.json")
    decisions = fetch("decisions.json")
    if summary is None:
        body = (f"<div class='card'><p><strong>The agent's public record could not be read just "
                f"now.</strong> Nothing is shown rather than a stale copy. The record itself is at "
                f"<a href='{RECORD}'>{RECORD}</a>.</p></div>")
    else:
        ledger = summary.get("ledger") or {}
        envelope_source = (summary.get("expected_envelope") or {}).get("source")
        genesis = str(ledger.get("genesis_hash") or "")
        rows = list((decisions or {}).get("decisions") or [])[-8:][::-1]
        recent = ("<ol class='dec'>" + "".join(_decision(r) for r in rows) + "</ol>" if rows else
                  "<p class='dim'>No decision yet. The agent decides on pre-registered heartbeats "
                  "(the US open and each funding settlement) and on event triggers; every tick "
                  "between them is in the log.</p>")
        body = (
            f"<div class='facts'><span class='pill proof'>{escape(str(summary.get('mode')))}"
            f" · Bitget Demo</span><span class='mono'>{escape(str(ledger.get('events')))} "
            f"events · head {escape(str(ledger.get('head_hash', ''))[:12])}…</span>"
            f"<span class='mono'>genesis {escape(genesis[:12])}… · "
            f"{escape(str(ledger.get('first_ts', ''))[:16])}Z</span>"
            f"<span class='mono dim'>published {escape(str(summary.get('generated_at', ''))[:16])}"
            f"Z</span></div>"
            f"<h2>Scored so far</h2>"
            f"{_metrics(summary.get('metrics') or {}, summary.get('expected_envelope') or {})}"
            f"<p class='dim'>The right-hand column was committed in the genesis event before the "
            f"first decision: what a book with no edge produces over the same window"
            + (f" ({escape(str(envelope_source))})" if envelope_source else "")
            + ". The agent is judged against it, not against zero.</p>"
            f"<h2>What it has done</h2><div class='funnel'>{_funnel(summary.get('counts') or {})}"
            f"</div><h2>Latest decisions</h2>{recent}")
    return f"""<!doctype html><html lang="en"><head>{design.head(
        "ARGUS · the Track 2 agent, live",
        "The Track 2 sentiment agent's paper run on Bitget Demo, read from its public record.")}
<style>{design.TOKENS_CSS}
  .facts {{ display:flex; flex-wrap:wrap; gap:10px 16px; align-items:center; margin:18px 0 8px }}
  .mono {{ font:12.5px/1.4 var(--mono) }} .dim {{ color:var(--dim) }}
  h2 {{ font-size:18px; margin:34px 0 10px }}
  table.m {{ width:100%; border-collapse:collapse; font-size:14.5px }}
  table.m th {{ text-align:left; font:600 11px/1.4 var(--mono); letter-spacing:.1em;
    text-transform:uppercase; color:var(--dim); padding:8px 10px;
    border-bottom:1px solid var(--line) }}
  table.m td {{ padding:10px; border-bottom:1px solid var(--line); vertical-align:top }}
  table.m td.n {{ font-family:var(--mono); white-space:nowrap }}
  table.m td.env {{ color:var(--dim); font-size:13px }}
  .tw {{ overflow-x:auto }}
  .funnel {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:10px }}
  .stat {{ border:1px solid var(--line); border-radius:12px; padding:14px;
    background:var(--panel) }}
  .stat b {{ display:block; font:700 24px/1 var(--sans) }}
  .stat span {{ color:var(--dim); font-size:13px }}
  ol.dec {{ list-style:none; padding:0; margin:0; display:grid; gap:10px }}
  ol.dec li {{ border:1px solid var(--line); border-radius:12px; padding:14px 16px }}
  .dh {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center }}
  ol.dec p {{ margin:8px 0 0 }}
  .card {{ border:1px solid var(--line); border-radius:12px; padding:16px;
    background:var(--panel) }}
  .notice {{ border:1px solid var(--line); border-left:3px solid var(--ink); border-radius:12px;
    padding:14px 16px; margin:16px 0 20px; background:var(--panel); font-size:14.5px }}
  .notice b {{ font-weight:600 }}
{design.BASE_CSS}</style></head><body>{design.nav('/agent')}<div class="wrap">
<p class="kicker">Track 2 · Agentic Trading · Market Sentiment Agent</p>
<h1>The agent that trades.</h1>
<div class="notice"><b>This is a separate project, by the same team</b> — not ARGUS. It trades on
Bitget's Demo (paper) environment, with no real money. ARGUS, the console you are reading right
now, places no order on any venue: its own paper desk fills its decisions in its own ledger, at the
recorded price, and nowhere else.</div>
<p class="sub">ARGUS researches; its sibling project trades. Qwen decides, a risk kernel that can
only reduce stands between it and the venue, and Bitget's Agent Hub places every order on Bitget
Demo. Everything below is read live from the agent's own record, recomputed hourly from its
hash-chained log — <a href="{RECORD}">full record</a> · <a href="{REPO}">source</a>. This is run 2,
the scored run; run 1, which came before it, is disclosed and not scored —
<a href="{RUN_1_RECORD}">its record</a>.</p>
{body}
</div>{design.footer()}</body></html>"""


__all__ = ["RECORD", "RUN_1_RECORD", "fetch_json", "render"]
