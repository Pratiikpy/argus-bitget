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
    from argus.market.bitget_mcp import BitgetDataService

    quote = BitgetDataService().quote("NVDA")
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
    with ThreadPoolExecutor(max_workers=len(CHECKS)) as pool:
        jobs = [(s, w, pool.submit(_timed, s, w, fn)) for s, w, fn in CHECKS]
        for surface, what, job in jobs:
            try:
                results.append(job.result(timeout=PROBE_TIMEOUT_S))
            except FutureTimeout:
                results.append(Check(surface, what, False,
                                     f"no answer within {PROBE_TIMEOUT_S:.0f}s",
                                     PROBE_TIMEOUT_S * 1000))
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
        out.append(("bitget-signal, every tool",
                    f"{skills.get('tools_answered')} of {skills.get('tools_probed')} tools "
                    f"returned data; {len(skills.get('skills_reached') or [])} of "
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
    mirror = _load(data, "skill_mirror.json")
    if mirror:
        substitutes = (mirror.get("answered_by_mirror", 0)
                       - mirror.get("answered_by_mirror_same_upstream", 0))
        out.append(("bitget-signal, covered",
                    f"Bitget's server answered {mirror.get('answered_by_bitget')} of "
                    f"{mirror.get('tools_probed')}; ARGUS read "
                    f"{mirror.get('answered_by_mirror')} more straight from the sources those "
                    f"Skills name ({mirror.get('answered_by_mirror_same_upstream')} the same "
                    f"source, "
                    f"{substitutes} a named substitute) — {mirror.get('answered_total')} of "
                    f"{mirror.get('tools_probed')} answered, "
                    f"{len(mirror.get('skills_answered_in_total') or [])} of 5 Skills — swept "
                    f"{str(mirror.get('checked_at', ''))[:10]}"))
    reliability = _load(data, "skill_reliability.json")
    if reliability:
        verdicts = reliability.get("by_verdict") or {}
        out.append(("bitget-signal, reliability",
                    f"{verdicts.get('reliable', 0)} tools answered all "
                    f"{reliability.get('attempts_per_tool')} attempts; "
                    f"{reliability.get('skills_with_a_reliable_tool')} of "
                    f"{reliability.get('skills_total')} Skills have a reliable tool"))
    coverage = _load(data, "data_coverage.json")
    if coverage:
        rate = coverage.get("answering_rate")
        out.append(("bitget-mcp-server, every entry",
                    f"{coverage.get('answering')} of {coverage.get('entries_probed')} catalog "
                    f"entries answered"
                    + (f" ({float(rate):.0%})" if isinstance(rate, (int, float)) else "")))
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
    return out


def render(status: dict[str, Any], checks: list[Check], checked_at: float,
           sweeps: list[tuple[str, str]], favicon: str) -> str:
    esc = html.escape
    live = sum(1 for c in checks if c.ok)
    rows = "".join(
        f"<tr><td>{esc(c.surface)}</td><td>{esc(c.what)}</td>"
        f"<td class='{'ok' if c.ok else 'bad'}'>{'answered' if c.ok else 'no answer'}</td>"
        f"<td>{esc(c.detail)}</td><td class='n'>{c.ms:,.0f} ms</td></tr>" for c in checks)
    sweep_rows = "".join(f"<tr><td>{esc(a)}</td><td>{esc(b)}</td></tr>" for a, b in sweeps)
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
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">{design.FONTS}
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="{favicon}">
<title>ARGUS — status</title>
<style>{design.TOKENS_CSS}
 body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,sans-serif }}
 .wrap {{ max-width:900px; margin:0 auto; padding:28px 18px 64px }}
 h1 {{ font-size:21px; margin:0 0 6px }} h2 {{ font-size:15px; margin:26px 0 8px }}
 .sub {{ color:var(--dim); font-size:13.5px; margin:0 0 16px; max-width:72ch }}
 .cards {{ display:flex; gap:10px; flex-wrap:wrap }}
 .card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
   padding:12px 14px; flex:1 1 150px }}
 .card b {{ display:block; font-size:20px; font-variant-numeric:tabular-nums }}
 .card span {{ color:var(--dim); font-size:12.5px }}
 .tbl {{ overflow-x:auto; background:var(--panel); border:1px solid var(--line);
   border-radius:10px }}
 table {{ border-collapse:collapse; width:100%; font-size:13.5px }}
 td {{ padding:8px 10px; border-top:1px solid var(--line); vertical-align:top }}
 tr:first-child td {{ border-top:0 }}
 .ok {{ color:var(--good); font-weight:600 }} .bad {{ color:var(--bad); font-weight:600 }}
 .n {{ font-family:var(--mono); color:var(--dim); white-space:nowrap; text-align:right }}
 a {{ color:var(--accent) }}
 @media (max-width:640px) {{
   table, tbody, tr, td {{ display:block; width:100% }}
   tr {{ border-top:1px solid var(--line); padding:8px 0 }} tr:first-child {{ border-top:0 }}
   td {{ border:0; padding:2px 12px }} .n {{ text-align:left }}
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
<div class="tbl"><table>{rows}</table></div>
<h2>Last full sweep</h2>
<div class="tbl"><table>{sweep_rows}</table></div>
<p class="sub" style="margin-top:14px">The desk decides four times a day during US market hours;
the next cycle is due {esc(due)}.</p>
</div>{design.footer()}</body></html>"""


__all__ = ["CHECKS", "Check", "live_checks", "render", "sweep_lines"]
