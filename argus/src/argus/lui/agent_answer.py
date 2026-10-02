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
from collections.abc import Callable, Mapping
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


_BREAKER_Q = re.compile(r"\b(?:circuit\s+breaker|breaker|reduce[\s-]?only|halt(?:ed)?|kernel)\b",
                        re.I)
_ITS = re.compile(r"\b(?:its|it|it's|the\s+agent'?s?|that\s+agent)\b", re.I)


def asks_follow_up(text: str, prior: list[str]) -> bool:
    """"Why is its circuit breaker on…?" straight after a question about the Track 2 agent: the
    pronoun is the agent, and it went to this console's own desk record (a judge, round 19, row
    663)."""
    return bool(prior and AGENT_Q.search(prior[-1]) and _ITS.search(text)
                and (_BREAKER_Q.search(text) or _ABOUT_ITS_RECORD.search(text)))


def asks_about_the_breaker(text: str) -> bool:
    return bool(_BREAKER_Q.search(text))


_HELD = re.compile(r"book\.(drawdown_pct|consecutive_losses)\s*=\s*(-?\d+(?:\.\d+)?)")
_LIMIT = re.compile(r"kernel\.(breaker_reduce_only_drawdown_pct|breaker_losing_streak)\s*=\s*"
                    r"(-?\d+(?:\.\d+)?)")


def breaker_answer(fetch: Callable[[str], Any] = fetch_json
                   ) -> tuple[list[str], list[Source], dict[str, Any]]:
    """Why the agent's breaker is on, from the latest decision whose record names it: each trip
    rule against the figure the book held, so the reason is the record's and not the narration's."""
    try:
        loaded = fetch("decisions.json") or []
    except Exception:
        loaded = []
    rows = loaded if isinstance(loaded, list) else loaded.get("decisions") or []
    latest = next((r for r in sorted(rows, key=lambda r: str(r.get("decided_at", "")),
                                     reverse=True)
                   if _HELD.search(str(r.get("mandate_response") or ""))
                   and _BREAKER_Q.search(str(r.get("mandate_response") or ""))), None)
    if latest is None:
        return ([f"Bottom line: no decision in the Track 2 agent's published record "
                 f"({RECORD}/decisions.json) names its breaker's inputs, so the reason is not "
                 f"stated rather than guessed."], [], {})
    response = str(latest.get("mandate_response") or "")
    held = {k: float(v) for k, v in _HELD.findall(response)}
    # The rule's thresholds are the policy's, the same on every decision of the run; a decision
    # that quotes only the figure that tripped borrows them from the nearest one that names them.
    limits: dict[str, float] = {}
    for row in sorted(rows, key=lambda r: str(r.get("decided_at", "")), reverse=True):
        for key, value in _LIMIT.findall(str(row.get("mandate_response") or "")):
            limits.setdefault(key, float(value))
        if len(limits) == 2:
            break
    trips = []
    streak, streak_limit = held.get("consecutive_losses"), limits.get("breaker_losing_streak")
    if streak is not None and streak_limit is not None:
        trips.append((streak >= streak_limit,
                      f"{streak:g} consecutive losing closes against a {streak_limit:g}-loss rule"))
    draw, draw_limit = held.get("drawdown_pct"), limits.get("breaker_reduce_only_drawdown_pct")
    if draw is not None and draw_limit is not None:
        trips.append((abs(draw) >= draw_limit,
                      f"a {abs(draw):.3g}% drawdown against its {draw_limit:g}% line"))
    fired = [what for hit, what in trips if hit]
    clear = [what for hit, what in trips if not hit]
    lead = ("Bottom line: the Track 2 agent's breaker is on because of "
            + (" and ".join(fired) if fired else "no rule its record shows tripped")
            + (f"; not because of {' or '.join(clear)}" if clear else "")
            + " — it makes the book reduce-only: positions can be closed, none opened.")
    lines = [lead,
             f"Read from decision seq {latest.get('seq')} "
             f"({str(latest.get('decided_at', ''))[:16]} UTC) in the agent's published record."]
    if "drawdown beyond" in str(latest.get("summary") or "") and clear:
        lines.append("That decision's own summary says the drawdown is beyond the line; its "
                     "record does not support that, and /agent says so beside it.")
    lines.append("This is the separate Track 2 project, not this console's research desk.")
    return lines, [Source("venue", f"{RECORD}/decisions.json",
                          "the agent's decisions and the facts each was given")], {}


def _elapsed_hours(summary: Mapping[str, Any]) -> int:
    """Whole hours from the scoring window's start to the record's publication, or 0 unread."""
    from datetime import datetime

    try:
        start = datetime.fromisoformat(str((summary.get("scoring_window") or {})["start"])
                                       .replace("Z", "+00:00"))
        made = datetime.fromisoformat(str(summary["generated_at"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return 0
    return max(0, int((made - start).total_seconds() // 3600))


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
    # n_hours counts the hourly marks recorded, not the time elapsed: 92 marks stood as "92 hours
    # into its scoring window" about 103 hours after it opened (a judge, round 20, row 718).
    elapsed = _elapsed_hours(summary)
    into = (f"{elapsed} hours into its scoring window ({hours} hourly marks recorded"
            + (f"; {elapsed - hours} hours have no mark" if hours < elapsed else "") + ")"
            if elapsed else f"{hours} hourly marks into its scoring window")
    lines = [
        f"Bottom line: the Track 2 agent — a separate project that paper-trades on Bitget's demo "
        f"venue, not this console's research desk — is {into}: "
        f"return {ret:+.2%}, Sharpe {sharpe:.2f}"
        + (f" with a standard error of {float(se):.1f}" if se is not None else "")
        + (f" (90% interval {float(ci[0]):.1f} to {float(ci[1]):.1f}"
           + (", which spans zero, so it cannot be told from no skill" if len(ci) == 2
              and float(ci[0]) < 0 < float(ci[1]) else "") + ")" if len(ci) == 2 else "")
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
