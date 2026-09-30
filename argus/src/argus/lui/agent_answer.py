"""The Track 2 agent's record, answered from the record it publishes — never from this console's own
desk.

"The agent page shows a Sharpe of 1.88 but a win rate of 0% on 3 closed trades — how can both be
true?" was answered from the research desk's ledger, a different system with one settled trade,
without a word that the two are different (a judge's audit, 2026-09-30). The Track 2 agent is a
separate project (t2-sentiment-agent) that paper-trades on Bitget's demo venue and publishes its
scored summary; /agent renders it (`lui/agent_page.py`), and this answers questions about it from
the same file, with the figure's uncertainty beside it and the no-edge envelope it is judged
against.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from argus.lui.agent_record import RECORD, fetch_json
from argus.lui.trace import trace_module
from argus.truth.source import Source

AGENT_Q = re.compile(
    r"\b(?:track\s*2|t2|trading\s+agent|sentiment\s+agent|the\s+agent'?s?|agent\s+page|your\s+agent|"
    r"run\s*2|paper[\s-]+trading\s+(?:agent|run|account))\b", re.I)
"""A question about the Track 2 agent rather than about the research desk or a market."""

_ABOUT_ITS_RECORD = re.compile(
    r"\b(?:sharpe|sortino|win\s+rate|drawdown|return|trades?|fills?|orders?|record|performance|"
    r"results?|doing|pnl|p&l|profit|loss|metrics?|numbers?|rejected|how\s+(?:is|has)\s+it)\b", re.I)


def asks_about_the_agent(text: str) -> bool:
    """A question about the Track 2 agent's own record."""
    return bool(AGENT_Q.search(text) and _ABOUT_ITS_RECORD.search(text))


def answer(fetch: Callable[[str], Any] = fetch_json
           ) -> tuple[list[str], list[Source], dict[str, Any]]:
    """The agent's scored metrics, its activity counts, and the envelope a no-edge book lands in."""
    try:
        summary = fetch("summary.json") or {}
    except Exception:
        summary = {}
    metrics = summary.get("metrics") or {}
    counts = summary.get("counts") or {}
    envelope = summary.get("expected_envelope") or {}
    if not metrics:
        return ([f"Bottom line: the Track 2 agent's published record ({RECORD}/summary.json) did "
                 f"not answer just now, so no figure is given rather than a stale one — /agent "
                 f"shows it when it loads."], [], {})
    closed = int(metrics.get("n_closed_trades") or 0)
    sharpe = float(metrics.get("sharpe_ann") or 0.0)
    se = metrics.get("sharpe_se_ann")
    ci = (metrics.get("ci90") or {}).get("sharpe_ann") or []
    ret = float(metrics.get("total_return") or 0.0)
    win = float(metrics.get("win_rate") or 0.0)
    drawdown = float(metrics.get("max_drawdown") or 0.0)
    hours = int(metrics.get("n_hours") or 0)
    lines = [
        f"Bottom line: the Track 2 agent — a separate project that paper-trades on Bitget's demo "
        f"venue, not this console's research desk — is {hours} hours into its scoring window: "
        f"return {ret:+.2%}, Sharpe {sharpe:.2f}"
        + (f" with a standard error of {float(se):.1f}" if se is not None else "")
        + (f" (90% interval {float(ci[0]):.1f} to {float(ci[1]):.1f})" if len(ci) == 2 else "")
        + f", win rate {win:.0%} on {closed} closed trade{'' if closed == 1 else 's'}, max "
          f"drawdown {drawdown:.2%} — too few trades for any of it to show skill either way.",
        f"How the two are measured: the Sharpe is read from {hours} hourly marks of the whole "
        f"book, open positions included, and the win rate from closed trades alone"
        + (" — which is how a positive Sharpe can sit beside a 0% win rate" if sharpe > 0
           and closed and win == 0 else "")
        + f"; with {closed} closed, each figure rests on a handful of outcomes.",
        f"What it has done: {counts.get('decisions', 0)} decisions, "
        f"{counts.get('orders_sent', 0)} orders sent, {counts.get('fills', 0)} filled, "
        f"{counts.get('protective_rulings', 0)} protective rulings by its risk kernel.",
    ]
    if envelope.get("sharpe_ann"):
        lines.append(f"A no-edge book over the same window lands at Sharpe "
                     f"{envelope['sharpe_ann']} and {envelope.get('closed_trades', '?')} closed "
                     f"trades, so this run is judged against that band, not against zero.")
    lines.append(f"Read from the agent's published record, {RECORD}/summary.json (generated "
                 f"{str(summary.get('generated_at', ''))[:16]} UTC); /agent shows it with its "
                 f"latest decisions. This console's own desk keeps a separate ledger — ask \"what "
                 f"is your track record\" for that.")
    sources = [Source("venue", f"{RECORD}/summary.json",
                      "the Track 2 agent's scored metrics and counts")]
    return lines, sources, {"agent": {"closed_trades": closed, "sharpe": sharpe, "return": ret}}


trace_module(globals())
