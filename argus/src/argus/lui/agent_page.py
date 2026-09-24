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

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from html import escape
from typing import Any

from argus.lui import design

RECORD = "https://t2-sentiment-agent-live.vercel.app"
REPO = "https://github.com/Pratiikpy/t2-sentiment-agent"
CACHE_SECONDS = 300.0
"""The record republishes hourly; five minutes keeps a busy page from refetching on every view
while never showing a record more than one publish behind."""

Fetch = Callable[[str], Mapping[str, Any] | None]
_CACHE: dict[str, tuple[float, Mapping[str, Any] | None]] = {}


def fetch_json(name: str) -> Mapping[str, Any] | None:
    """One file of the public record, or None when it cannot be read. Cached briefly."""
    hit = _CACHE.get(name)
    if hit is not None and time.monotonic() - hit[0] < CACHE_SECONDS:
        return hit[1]
    request = urllib.request.Request(f"{RECORD}/{name}", headers={"User-Agent": "ARGUS console"})
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            loaded = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        loaded = None
    value = loaded if isinstance(loaded, dict) else None
    if value is not None:
        _CACHE[name] = (time.monotonic(), value)
    return value


def _num(value: Any, fmt: str, missing: str = "undefined") -> str:
    if value is None:
        return missing
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return missing


def _metrics(metrics: Mapping[str, Any], envelope: Mapping[str, Any]) -> str:
    rows = (
        ("Sharpe (annualised)", _num(metrics.get("sharpe_ann"), ".2f"), envelope.get("sharpe_ann")),
        ("Max drawdown", _num(metrics.get("max_drawdown"), ".2%"),
         envelope.get("max_drawdown_pct")),
        ("Win rate", _num(metrics.get("win_rate"), ".0%"), envelope.get("win_rate")),
        ("Closed trades", str(metrics.get("n_closed_trades", 0)), envelope.get("closed_trades")),
        ("Return", _num(metrics.get("total_return"), "+.2%"),
         envelope.get("return_on_equity_bps")),
        ("Fees paid (USDT)", _num(metrics.get("fees_paid"), ".2f"), None),
    )
    body = "".join(
        f"<tr><td>{escape(label)}</td><td class='n'>{escape(value)}</td>"
        f"<td class='env'>{escape(str(env)) if env else '—'}</td></tr>"
        for label, value, env in rows)
    return ("<div class='tw'><table class='m'><thead><tr><th>Metric</th><th>Agent</th>"
            "<th>A no-edge book over the same window</th></tr></thead><tbody>" + body
            + "</tbody></table></div>")


def _funnel(counts: Mapping[str, Any]) -> str:
    order = (("decisions", "decisions"), ("flat_with_reasons", "flat, with reasons"),
             ("kernel_changed_decisions", "changed by the risk kernel"),
             ("orders_sent", "orders sent"), ("fills", "fills"),
             ("closed_trades", "closed trades"), ("protective_rulings", "protective rulings"))
    return "".join(f"<div class='stat'><b>{escape(str(counts.get(key, 0)))}</b>"
                   f"<span>{escape(label)}</span></div>" for key, label in order)


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
    detail = legs or (f"Flat: {reasons}" if reasons else "")
    when = escape(str(row.get("decided_at", ""))[:16])
    stance = escape(str(row.get("stance") or row.get("outcome")))
    seq = escape(str(row.get("seq", "—")))
    return (f"<li><div class='dh'><span class='mono'>{when}Z</span>"
            f"<span class='pill proof'>{stance}</span>{pill}"
            f"<span class='mono dim'>seq {seq}</span></div>"
            f"<p>{escape(str(row.get('summary') or ''))}</p>"
            + (f"<p class='dim'>{escape(detail)}</p>" if detail else "") + "</li>")


def render(fetch: Fetch = fetch_json) -> str:
    summary = fetch("summary.json")
    decisions = fetch("decisions.json")
    if summary is None:
        body = (f"<div class='card'><p><strong>The agent's public record could not be read just "
                f"now.</strong> Nothing is shown rather than a stale copy. The record itself is at "
                f"<a href='{RECORD}'>{RECORD}</a>.</p></div>")
    else:
        ledger = summary.get("ledger") or {}
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
            f"first decision: what a book with no edge produces over the same window, measured "
            f"across 17,400 72-hour windows. The agent is judged against it, not against zero.</p>"
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
{design.BASE_CSS}</style></head><body>{design.nav('/agent')}<div class="wrap">
<p class="kicker">Track 2 · Agentic Trading · Market Sentiment Agent</p>
<h1>The agent that trades.</h1>
<p class="sub">ARGUS researches; its sibling project trades. Qwen decides, a risk kernel that can
only reduce stands between it and the venue, and Bitget's Agent Hub places every order on Bitget
Demo. Everything below is read live from the agent's own record, recomputed hourly from its
hash-chained log — <a href="{RECORD}">full record</a> · <a href="{REPO}">source</a>.</p>
{body}
</div>{design.footer()}</body></html>"""


__all__ = ["RECORD", "fetch_json", "render"]
