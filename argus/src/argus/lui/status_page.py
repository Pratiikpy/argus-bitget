"""The record and its sources, checked now — the status page a person opens.

`/status` answered with raw JSON, and a judge who clicked it saw the one screen of the product that
looked unfinished (a Track 3 judge audit, 2026-09-24). JSON stays for any client that asks for it;
a browser gets this page.

It also answers the question Track 3 scores by name — "data sources / Skill integration count
*and effectiveness*" — on the product itself rather than in a file a reader has to find. Two kinds
of line, never mixed:

* **checked now** — a live, keyless call to each Bitget surface this console answers from, made
  when the page is requested (at most once every ten minutes per server instance), with how long
  it took and what came back;
* **last full sweep** — the slow measurements (all nineteen `bitget-signal` tools, three attempts
  each; every `bitget-mcp-server` entry) read from their dated artefacts, because a sweep that takes
  minutes cannot run inside a page load. Each says when it ran.

A surface that does not answer is shown as not answering. The page never substitutes a remembered
value for a live one without saying which it is.
"""

from __future__ import annotations

import html
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argus.lui import design

PROBE_TIMEOUT_S = 12.0
CACHE_SECONDS = 600.0
_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class Check:
    surface: str
    what: str
    ok: bool
    detail: str
    ms: float


def _timed(surface: str, what: str, call: Callable[[], str]) -> Check:
    began = time.perf_counter()
    try:
        detail = call()
        ok = True
    except Exception as exc:  # a probe that fails is the finding, not an error in the page
        detail, ok = f"did not answer ({type(exc).__name__})", False
    return Check(surface, what, ok, detail, (time.perf_counter() - began) * 1000)


def _ticker() -> str:
    from argus.market.bitget import fetch_tickers

    tickers = fetch_tickers()
    nvda = tickers.get("NVDAUSDT")
    return (f"{len(tickers)} USDT-margined contracts quoted"
            + (f"; NVDA last {nvda.last}" if nvda is not None else ""))


def _book() -> str:
    from argus.market.depth import fetch_orderbook

    book = fetch_orderbook("NVDAUSDT", limit=50)
    return f"{len(book.bids)} bid and {len(book.asks)} ask levels for NVDA"


def _candles() -> str:
    from argus.market.history import CandleType, fetch

    bars = fetch("NVDAUSDT", interval="1H", candle_type=CandleType.MARKET, recent=True, limit=24)
    return f"{len(bars)} hourly NVDA candles, newest {bars[-1].ts:%d %b %H:%M} UTC"


def _mcp_quote() -> str:
    from argus.market.bitget_mcp import shared_service

    quote = shared_service().quote("NVDA")
    price = quote.get("price") or quote.get("close") or quote.get("last")
    return f"US equity quote for NVDA{f': {price}' if price else ''}"


def _skill_ta() -> str:
    from argus.market.skills import indicators

    found = indicators("NVDAUSDT")
    if not found:
        raise RuntimeError("no indicators returned")
    rsi = found.get("rsi") or found.get("RSI")
    return f"technical-analysis Skill answered for NVDA{f', RSI {float(rsi):.1f}' if rsi else ''}"


CHECKS: tuple[tuple[str, str, Callable[[], str]], ...] = (
    ("Bitget market API", "all tickers", _ticker),
    ("Bitget market API", "order book, 50 levels", _book),
    ("Bitget market API", "hourly candles", _candles),
    ("bitget-mcp-server", "equity quote (agent.bitget.com/mcp)", _mcp_quote),
    ("bitget-signal", "technical-analysis Skill", _skill_ta),
)


def live_checks(*, force: bool = False) -> tuple[list[Check], float]:
    """Every live check, run in parallel; cached so a busy page does not hammer the venue."""
    now = time.time()
    if not force and _CACHE.get("at") and now - float(_CACHE["at"]) < CACHE_SECONDS:
        return list(_CACHE["checks"]), float(_CACHE["at"])
    results: list[Check] = []
    # Not a `with` block: leaving one waits for every probe, so one hung upstream held the page past
    # its own timeout (audit, 2026-09-26). One deadline for all of them, and a probe still running
    # at it is reported as not answering and abandoned.
    pool = ThreadPoolExecutor(max_workers=len(CHECKS))
    deadline = time.monotonic() + PROBE_TIMEOUT_S
    try:
        jobs = [(s, w, pool.submit(_timed, s, w, fn)) for s, w, fn in CHECKS]
        for surface, what, job in jobs:
            try:
                results.append(job.result(timeout=max(0.0, deadline - time.monotonic())))
            except FutureTimeout:
                results.append(Check(surface, what, False,
                                     f"no answer within {PROBE_TIMEOUT_S:.0f}s",
                                     PROBE_TIMEOUT_S * 1000))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    _CACHE.update(at=now, checks=results)
    return results, now


def _load(data: Path, name: str) -> dict[str, Any] | None:
    try:
        loaded = json.loads((data / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def sweep_lines(data: Path) -> list[tuple[str, str]]:
    """The slow measurements, each with the date it was taken."""
    out: list[tuple[str, str]] = []
    skills = _load(data, "bitget_skills_health.json")
    if skills:
        # Counted in calls, with the tools they reach: the desk's set asks six actions of one
        # tool, and "19 tools" once described 19 calls to 12 (judge audit, 2026-09-26).
        tools = len({str(r.get("tool")) for r in skills.get("results") or []})
        out.append(("bitget-signal, the desk's calls",
                    f"{skills.get('tools_answered')} of {skills.get('tools_probed')} calls "
                    f"({tools} tools) returned data; {len(skills.get('skills_reached') or [])} of "
                    f"{len(skills.get('skills_available') or [])} Skills returned data from at "
                    f"least one tool — swept {str(skills.get('checked_at', ''))[:10]}"))
    understanding = _load(data, "lui_final_heldout_report.json")
    if understanding:
        after = understanding.get("console_after") or {}
        before = understanding.get("console_before") or {}
        alone = understanding.get("kind_model_alone") or {}
        qwen = understanding.get("console_with_qwen") or {}
        out.append(("Understanding questions",
                    f"{understanding.get('written_by', 'a blind writer')}: this console, with no "
                    f"language model, read {before.get('no_book', '?')} correctly before "
                    f"2026-09-25 and {after.get('no_book', '?')} after "
                    f"({after.get('saved_book', '?')} with a saved book); its question "
                    f"classifier alone {alone.get('correct', '?')}/{alone.get('rows', '?')}; "
                    f"with Qwen reading first "
                    f"{qwen.get('no_book', 'not measured')} — "
                    f"measured {understanding.get('measured_at', '')}"))
    figures = _load(data, "figurecheck.json")
    if figures:
        asked = figures.get("questions_with_numbers", 0)
        lost = figures.get("with_a_dropped_number", 0)
        out.append(("Numbers read from questions",
                    f"of {asked} blind-corpus questions that state a percentage, amount or "
                    f"multiple, {asked - lost} carry every one into the analysis; {lost} "
                    f"{'loses' if lost == 1 else 'lose'} one — each listed in "
                    f"data/figurecheck.json, none silently"))
    mirror = _load(data, "skill_mirror.json")
    if mirror:
        substitutes = (mirror.get("answered_by_mirror", 0)
                       - mirror.get("answered_by_mirror_same_upstream", 0))
        out.append(("bitget-signal, covered",
                    f"of the desk's {mirror.get('tools_probed')} calls, Bitget's server answered "
                    f"{mirror.get('answered_by_bitget')}; ARGUS read "
                    f"{mirror.get('answered_by_mirror')} more straight from the sources those "
                    f"Skills name ({mirror.get('answered_by_mirror_same_upstream')} the same "
                    f"source, "
                    f"{substitutes} a named substitute), labelled as that source, never as the "
                    f"Skill — swept {str(mirror.get('checked_at', ''))[:10]}"))
    reliability = _load(data, "skill_reliability.json")
    if reliability:
        verdicts = reliability.get("by_verdict") or {}
        named = ", ".join(reliability.get("reliable_tools") or [])
        out.append(("bitget-signal, reliability",
                    f"{verdicts.get('reliable', 0)} of {reliability.get('tools')} tools answered "
                    f"all {reliability.get('attempts_per_tool')} attempts"
                    + (f" ({named})" if named else "") + "; "
                    f"{reliability.get('skills_with_a_reliable_tool')} of "
                    f"{reliability.get('skills_total')} Skills have a reliable tool — "
                    + (f"swept {str(reliability['checked_at'])[:10]}"
                       if reliability.get("checked_at") else "sweep date not recorded")))
    coverage = _load(data, "data_coverage.json")
    if coverage:
        from argus.market.bitget_mcp import ANSWER_ENTRIES

        healthy = {str(p.get("entry_id")) for p in coverage.get("probes") or []
                   if p.get("health") == "ok"}
        out.append(("bitget-mcp-server, used by answers",
                    f"{len(ANSWER_ENTRIES)} of {coverage.get('entries_probed')} catalog entries "
                    f"are read by the console's answers; "
                    f"{sum(1 for e in ANSWER_ENTRIES if e in healthy)} of those answered in the "
                    f"sweep below"))
        rate = coverage.get("answering_rate")
        out.append(("bitget-mcp-server, every entry",
                    f"{coverage.get('answering')} of {coverage.get('entries_probed')} catalog "
                    f"entries answered"
                    + (f" ({float(rate):.0%})" if isinstance(rate, (int, float)) else "")
                    + (f" — swept {str(coverage['checked_at'])[:10]}"
                       if coverage.get("checked_at") else " — sweep date not recorded")))
    macd = _load(data, "skill_macd_check.json")
    if macd:
        checked = int(macd.get("swapped") or 0) + int(macd.get("straight") or 0)
        out.append(("Corrected, not trusted",
                    f"bitget-signal returns MACD's signal line and histogram swapped on "
                    f"{macd.get('swapped')} of {checked} symbols checked; every technicals "
                    f"answer here recomputes from Bitget candles and corrects it"))
    sources = _load(data, "source_health.json")
    if sources:
        out.append(("Research sources beyond Bitget",
                    f"{sources.get('live')} of {sources.get('total')} (SEC EDGAR, FRED, news "
                    f"feeds and others) answered — probed "
                    f"{str(sources.get('probed_at', ''))[:10]}"))
    calls = _load(data, "call_grades.json")
    if calls:
        # The console's own answers, kept before the outcome and graded after it
        # (`eval/call_record.py`); a record with nothing graded yet says so, not a score.
        risk, direction = calls.get("risk_graded", 0), calls.get("direction_graded", 0)
        graded = []
        if risk:
            graded.append(f"beta off by {float(calls['mean_beta_error']):.2f} on average over "
                          f"{risk} call{'s' if risk != 1 else ''} four weeks on")
        if direction:
            graded.append(f"direction Brier {float(calls['brier']):.3f} against "
                          f"{float(calls['brier_base_rate']):.3f} for the plain base rate, "
                          f"{direction} horizon{'s' if direction != 1 else ''}")
        out.append(("Our own calls, graded",
                    f"{calls.get('calls', 0)} answers recorded since "
                    f"{str(calls.get('first_call') or '')[:10]}, hash-chained; "
                    + ("; ".join(graded) if graded else "none has reached its horizon yet")
                    + f" — graded {str(calls.get('graded_at', ''))[:10]}"))
        weekend = calls.get("weekend") or {}
        if weekend.get("recorded"):
            scored = int(weekend.get("graded") or 0)
            out.append(("Our weekend bands, graded",
                        f"{weekend['recorded']} weekend bands recorded before the reopen; "
                        + (f"{scored} graded by Monday's open, {float(weekend['covered']):.0%} "
                           f"inside their 10-90 band (80% is calibrated)" if scored else
                           "none graded yet: the first reopen has not printed")
                        + (f", {weekend['pending']} pending" if weekend.get("pending") else "")))
    return out


def _uptime_rows(history: list[dict[str, Any]] | None, now: float) -> str:
    """The uptime table from `lui/status_history.py`'s recorded rounds, or a line saying there
    are none yet — never a figure invented from the live check alone."""
    from datetime import UTC, datetime

    from argus.lui.status_history import WINDOWS, uptime, verify

    if not history:
        return ("<p class='sub'>No probe rounds recorded yet: the local cycle records one each "
                "time it runs (<code>python -m argus.lui.status_page --record</code>).</p>")
    esc = html.escape
    broken = verify(history)
    rows = "".join(
        f"<tr><td data-label='Surface'>{esc(u['surface'])}</td>"
        f"<td data-label='Checked'>{esc(u['what'])}</td>"
        + "".join(
            f"<td data-label='{esc(label)}' class='n'>"
            + ("—" if u[label]['uptime'] is None
               else f"{u[label]['uptime']}% of {u[label]['probes']}") + "</td>"
            for label, _ in WINDOWS) + "</tr>"
        for u in uptime(history, now=datetime.fromtimestamp(now, UTC)))
    heads = "".join(f"<th scope='col'>{esc(label)}</th>" for label, _ in WINDOWS)
    newest = esc(str(history[-1].get("at", ""))[:16].replace("T", " "))
    chain = ("the history's hash chain holds" if broken is None
             else f"the history's hash chain is BROKEN ({esc(broken)})")
    return (f"<div class='tbl'><table><thead><tr><th scope='col'>Surface</th>"
            f"<th scope='col'>Checked</th>{heads}</tr></thead><tbody>{rows}</tbody></table></div>"
            f"<p class='sub' style='margin-top:10px'>Share of recorded probes that answered; the "
            f"newest round was {newest} UTC, and {chain}.</p>")


def render(status: dict[str, Any], checks: list[Check], checked_at: float,
           sweeps: list[tuple[str, str]], favicon: str,
           history: list[dict[str, Any]] | None = None) -> str:
    esc = html.escape
    live = sum(1 for c in checks if c.ok)
    rows = "".join(
        f"<tr><td data-label='Surface'>{esc(c.surface)}</td>"
        f"<td data-label='Checked'>{esc(c.what)}</td>"
        f"<td data-label='Result' class='{'ok' if c.ok else 'bad'}'>"
        f"{'answered' if c.ok else 'no answer'}</td>"
        f"<td data-label='Detail'>{esc(c.detail)}</td>"
        f"<td data-label='Time' class='n'>{c.ms:,.0f} ms</td></tr>" for c in checks)
    sweep_rows = "".join(f"<tr><td data-label='Sweep'>{esc(a)}</td>"
                         f"<td data-label='Result'>{esc(b)}</td></tr>" for a, b in sweeps)
    age = status.get("age_hours")
    stale = status.get("stale")
    checked = time.strftime("%d %b %Y %H:%M UTC", time.gmtime(checked_at))
    chain = "intact" if status.get("chain_intact") else "BROKEN"
    age_text = "—" if age is None else f"{age:.1f}h"
    stale_text = " — stale" if stale else ""
    due = str(status.get("next_cycle_at") or "")
    try:
        from datetime import datetime

        due = datetime.fromisoformat(due).strftime("%d %b %H:%M UTC")
    except ValueError:
        due = due or "on schedule"
    del favicon  # the head carries the mark itself (design.head)
    head = design.head("ARGUS — status",
                       "What the desk can see right now: every data source it reads, checked "
                       "live and dated.", "/status")
    return f"""<!doctype html><html lang="en"><head>{head}
<style>{design.TOKENS_CSS}
 body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,sans-serif }}
 h2 {{ font-size:15px; margin:26px 0 8px }}
 .sub {{ margin:0 0 16px }}
 .cards {{ display:flex; gap:10px; flex-wrap:wrap }}
 .card {{ background:var(--panel); border:1px solid var(--line);
   padding:12px 14px; flex:1 1 150px }}
 .card b {{ display:block; font-size:20px; font-variant-numeric:tabular-nums }}
 .card span {{ color:var(--dim); font-size:12.5px }}
 .tbl {{ overflow-x:auto; background:var(--panel); border:1px solid var(--line);
   border-radius:10px }}
 table {{ border-collapse:collapse; width:100%; font-size:13.5px }}
 td {{ padding:8px 10px; border-top:1px solid var(--line); vertical-align:top }}
 th {{ padding:6px 10px; text-align:left; font:600 11px/1.3 var(--mono); letter-spacing:.08em;
   text-transform:uppercase; color:var(--dim); border-bottom:1px solid var(--line) }}
 tbody tr:first-child td {{ border-top:0 }}
 .ok {{ color:var(--good); font-weight:600 }} .bad {{ color:var(--bad); font-weight:600 }}
 .n {{ font-family:var(--mono); color:var(--dim); white-space:nowrap; text-align:right }}
 a {{ color:var(--accent) }}
 @media (max-width:640px) {{
   table, tbody, tr, td {{ display:block; width:100% }}
   thead {{ position:absolute; left:-9999px }}
   tr {{ border-top:1px solid var(--line); padding:8px 0 }} tr:first-child {{ border-top:0 }}
   td {{ border:0; padding:2px 12px }} .n {{ text-align:left }}
   td::before {{ content:attr(data-label) " · "; color:var(--dim); font:11px var(--mono) }}
 }}
{design.BASE_CSS}</style></head><body>{design.nav('/status')}<div class="wrap">
<h1>Status — the record and its sources</h1>
<p class="sub">What the desk's hash-chained record holds, and which of Bitget's data surfaces
this console answers from are answering right now. Live checks ran {esc(checked)}; the slow
sweeps say when they ran. <a href="/">Console</a> · <a href="/research">Research task</a> ·
<a href="/wrong">What we got wrong</a> · <a href="/status?format=json">JSON</a></p>
<div class="cards">
 <div class="card"><b>{status.get("entries")}</b><span>decisions on record</span></div>
 <div class="card"><b>{chain}</b><span>hash chain</span></div>
 <div class="card"><b>{age_text}</b><span>since the newest decision{stale_text}</span></div>
 <div class="card"><b>{live} of {len(checks)}</b><span>Bitget surfaces answering now</span></div>
</div>
<h2>Checked now</h2>
<div class="tbl"><table><thead><tr><th scope="col">Surface</th><th scope="col">Checked</th>
<th scope="col">Result</th><th scope="col">Detail</th><th scope="col">Time</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<h2>Over time</h2>
{_uptime_rows(history, checked_at)}
<h2>Last full sweep</h2>
<div class="tbl"><table><thead><tr><th scope="col">Sweep</th><th scope="col">Result</th></tr>
</thead><tbody>{sweep_rows}</tbody></table></div>
<p class="sub" style="margin-top:14px">The desk decides four times a day during US market hours;
the next cycle is due {esc(due)}.</p>
</div>{design.footer()}</body></html>"""


def record_round() -> int:  # pragma: no cover - CLI, live network
    """Run every live check once and append the round to the status history
    (`lui/status_history.py`); the scheduled local cycle calls this."""
    from datetime import UTC, datetime

    from argus.lui.status_history import read, record, verify

    checks, _ = live_checks(force=True)
    at = datetime.now(UTC)
    written = record(checks, at=at)
    answered = sum(1 for c in checks if c.ok)
    print(f"recorded {written} probe(s) at {at:%Y-%m-%d %H:%M} UTC: {answered} answered")
    broken = verify(read())
    if broken:
        print(f"history chain broken: {broken}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    import sys

    if sys.argv[1:] != ["--record"]:
        print("usage: python -m argus.lui.status_page --record")
        raise SystemExit(2)
    raise SystemExit(record_round())


__all__ = ["CHECKS", "Check", "live_checks", "record_round", "render", "sweep_lines"]
