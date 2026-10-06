"""The research task as a page: the question, the conclusion, each step and its charts.

Split from `lui/task.py` on 2026-09-27 (audit finding 169): the engine composes the task from the
engines' figures and keeps them as data — the quote step's price path, the impact step's report —
and this module is the only one that turns them into HTML and SVG. The charts are drawn from the
same numbers the lines state.
"""

from __future__ import annotations

import html
from typing import Any
from urllib.parse import urlencode

from argus.lui import design
from argus.lui.answer import LEAD
from argus.lui.task import ASKED_TITLE, DEFAULT_BOOK_VALUE, IMPACT_TITLE, Step, Task
from argus.lui.thesis import Tested


def _chart(task: Task, step: Step) -> str:
    """The step's chart, drawn from its own data: the price path on the quote step, the risk shares
    on the impact step, nothing elsewhere."""
    if step.path:
        return price_chart(list(step.path), task.name)
    if step.title == IMPACT_TITLE and not step.refused:
        return risk_chart(step.data.get("report") or {})
    return ""


def price_chart(points: list[tuple[Any, float]], name: str) -> str:
    """The name's last thirty days as a line with its range and last close marked."""
    if len(points) < 2:
        return ""
    width, height, pad = 640.0, 150.0, 6.0
    closes = [c for _, c in points]
    low, high = min(closes), max(closes)
    span = (high - low) or 1.0

    def x(i: int) -> float:
        return pad + (width - 2 * pad) * i / (len(points) - 1)

    def y(value: float) -> float:
        return pad + (height - 2 * pad) * (1 - (value - low) / span)

    line = " ".join(f"{x(i):.1f},{y(c):.1f}" for i, c in enumerate(closes))
    area = f"{x(0):.1f},{height - pad:.1f} {line} {x(len(closes) - 1):.1f},{height - pad:.1f}"
    first, last = points[0][0], points[-1][0]
    change = closes[-1] / closes[0] - 1
    label = (f"{name}, 4-hour closes {first:%d %b} to {last:%d %b}: {closes[0]:,.2f} to "
             f"{closes[-1]:,.2f} ({change:+.1%}); range {low:,.2f} to {high:,.2f}")
    return (
        f"<figure class='chart'><svg viewBox='0 0 {width:.0f} {height:.0f}' role='img' "
        f"aria-label='{html.escape(label)}' preserveAspectRatio='none'>"
        f"<polygon class='area' points='{area}'/>"
        f"<polyline class='path' points='{line}'/>"
        f"<circle class='dot' cx='{x(len(closes) - 1):.1f}' cy='{y(closes[-1]):.1f}' r='3.5'/>"
        f"</svg><figcaption>{html.escape(label)}</figcaption></figure>"
    )


def risk_chart(report: dict[str, Any]) -> str:
    """Each holding's share of the money beside its share of the risk, after the trade."""
    risk = report.get("risk_after") or {}
    vol = float(risk.get("volatility") or 0.0)
    rows = risk.get("contributions") or []
    if vol <= 0 or not rows:
        return ""
    data = sorted(((str(r["symbol"]).removesuffix("USDT"), float(r["weight"]),
                    float(r["contribution"]) / vol) for r in rows), key=lambda r: -r[2])
    top = max(max(w, abs(k)) for _, w, k in data) or 1.0
    row_h, label_w, width = 30, 70, 640
    height = row_h * len(data) + 8
    bar_w = width - label_w - 60
    parts = []
    for i, (name, weight, share) in enumerate(data):
        y0 = 4 + i * row_h
        parts.append(
            f"<text class='lbl' x='0' y='{y0 + 16}'>{html.escape(name)}</text>"
            f"<rect class='w' x='{label_w}' y='{y0 + 3}' width='{bar_w * weight / top:.1f}' "
            f"height='9' rx='2'/>"
            f"<rect class='r' x='{label_w}' y='{y0 + 14}' width='{bar_w * max(share, 0) / top:.1f}'"
            f" height='9' rx='2'/>"
            f"<text class='val' x='{label_w + bar_w * max(weight, share, 0) / top + 6:.1f}' "
            f"y='{y0 + 17}'>{weight:.0%} / {share:.0%}</text>")
    summary = "; ".join(f"{n} {w:.0%} of the money, {k:.0%} of the risk" for n, w, k in data)
    return (
        f"<figure class='chart'><svg viewBox='0 0 {width} {height}' role='img' "
        f"aria-label='{html.escape(summary)}'>{''.join(parts)}</svg>"
        f"<figcaption><span class='key w'></span>share of the money "
        f"<span class='key r'></span>share of the risk, after the trade</figcaption></figure>"
    )


def _line_class(line: str) -> str:
    if bool(LEAD.match(line)):
        return "act"
    return "fine" if line.startswith(("Data:", "Assumed:")) else "l"



def _card(n: int, step: Step, chart: str = "") -> str:
    """One engine's step, folded to its header and its Bottom line; the chart and every other
    line open on a tap (round 45 visual audit, minor 4: eight expanded sections ran /research to
    fifteen phone screens). The engine's own "Corrected:" note is a development record of a
    provider bug, not part of the answer, so it is left out here and kept in the JSON."""
    esc = html.escape
    lines = [line for line in step.lines if not line.startswith("Corrected:")]
    lead = next((line for line in lines if LEAD.match(line)), None)
    rest = [line for line in lines if line is not lead]
    # with no Bottom line the first line stands in for it, so a folded card still says something
    shown = lead if lead is not None else (rest.pop(0) if rest else None)
    top = (f"<p class='{_line_class(shown)}'>{design.linked(shown)}</p>" if shown else "")
    body = "".join(f"<p class='{_line_class(line)}'>{design.linked(line)}</p>" for line in rest)
    classes = "s" + (" r" if step.refused else "") + ("" if step.applicable else " na")
    inner = chart + body
    more = (f"<div class='more'>{inner}</div>" if inner else "")
    return (f"<details class='{classes}'><summary><div class='h'><span class='n'>{n}"
            f"</span><h2>{esc(step.title)}</h2><span class='e'>{esc(step.engine)} · "
            f"{step.seconds:.1f}s</span></div>{top}</summary>{more}</details>")


def _top(asked: str, name: str, size_pct: float, book_text: str) -> str:
    """Everything above the task itself: head, navigation, introduction and the form."""
    esc = html.escape
    head = design.head("ARGUS — one research task, live",
                       "Ask a question about adding a name to your book; eight engines answer "
                       "from live data and the page ends in one verdict.", "/research")
    return f"""<!doctype html><html lang="en"><head>{head}
<style>{design.TOKENS_CSS}
 body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,sans-serif }}
 .sub {{ margin:0 0 18px }}
 form {{ margin:0 0 20px }}
 .ask {{ display:flex; gap:8px; flex-wrap:wrap; align-items:flex-start }}
 form textarea, form input {{ padding:9px 11px; border:1px solid var(--line); border-radius:8px;
   background:var(--panel); color:var(--ink); font:15px/1.45 system-ui,sans-serif }}
 form textarea {{ flex:1 1 320px; min-width:0; resize:vertical; min-height:44px;
   box-sizing:border-box }}
 form details {{ margin-top:8px; font-size:13.5px; color:var(--dim) }}
 form details summary {{ cursor:pointer }}
 .edit {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:8px }}
 .edit label {{ display:flex; flex-direction:column; gap:3px; font-size:12px }}
 form .name {{ width:110px }} form .size {{ width:90px }}
 .edit .wide {{ flex:1 1 260px }} form .book {{ width:100%; box-sizing:border-box }}
 .read {{ margin:0 0 4px }} .notes {{ margin:0 0 14px; padding-left:20px; color:var(--dim);
   font-size:13px }}
 form input:focus-visible, form textarea:focus-visible, form button:focus-visible {{
   outline:2px solid var(--accent);
   outline-offset:2px }}
 form button {{ padding:9px 16px; border:1px solid var(--accent); background:var(--accent);
   color:var(--on-accent); border-radius:8px; font-size:14px; cursor:pointer }}
 form button:disabled {{ opacity:.7; cursor:progress }}
 .running {{ margin:8px 0 0; font-size:13.5px; color:var(--dim) }}
 .example {{ margin:0 0 8px; font-size:13.5px; color:var(--dim) }}
 .example .pill {{ margin-right:8px; vertical-align:1px }}
 .q {{ font-size:17px; font-weight:600; margin:0 0 12px }}
 .concl {{ background:var(--panel); border:1px solid var(--accent); border-radius:10px;
   padding:14px 18px; margin:0 0 20px }}
 .concl h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:.08em; color:var(--accent);
   margin:0 0 8px }}
 .concl ol {{ margin:0; padding-left:20px }} .concl li {{ margin:4px 0 }}
 .verdict {{ background:var(--panel); border:2px solid var(--accent); border-radius:10px;
   padding:14px 18px; margin:0 0 14px }}
 .verdict h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:.08em;
   color:var(--accent); margin:0 0 6px }}
 .verdict .call {{ font-size:19px; font-weight:700; margin:0 0 6px }}
 .verdict p {{ margin:4px 0 }}
 .verdict h3 {{ font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--dim);
   margin:14px 0 4px }}
 .verdict .call2 {{ font-size:16px; font-weight:700; margin:0 0 4px }}
 .verdict ul {{ margin:4px 0; padding-left:20px }}
 .verdict li {{ margin:3px 0 }}
 .verdict .r {{ font-weight:700 }} .verdict .r.supported {{ color:var(--good) }}
 .verdict .r.contradicted {{ color:var(--bad) }}
 .verdict .r.not_measurable, .verdict .r.not_tested {{ color:var(--dim) }}
 .verdict ul.ev {{ margin:4px 0 6px; font-size:13.5px; color:var(--dim) }}
 .verdict .src {{ white-space:nowrap }} .verdict .src::before {{ content:"· " }}
 .verdict .imp {{ font-weight:400; color:var(--dim); font-size:13px }}
 .s {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
   padding:14px 16px; margin-bottom:12px }}
 .s.r {{ border-left:3px solid var(--warn) }}
 .s.na {{ color:var(--dim) }}
 .s.wait {{ color:var(--dim); border-style:dashed }}
 .h {{ display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; margin-bottom:6px }}
 .n {{ font:12px var(--mono); color:var(--dim) }}
 .h h2 {{ font-size:15.5px; margin:0 }}
 .e {{ font:12px var(--mono); color:var(--dim); margin-left:auto }}
 .s p {{ margin:4px 0; overflow-wrap:anywhere }}
 details.s > summary {{ list-style:none; cursor:pointer }}
 details.s > summary::-webkit-details-marker {{ display:none }}
 details.s > summary .h h2::after {{ content:"  +"; color:var(--accent); font-weight:600 }}
 details.s[open] > summary .h h2::after {{ content:"  -" }}
 details.s .more {{ margin-top:6px }}
 .act {{ font-weight:600; color:var(--accent) }} .fine {{ color:var(--dim); font-size:13px }}
 a {{ color:var(--accent) }}
 .chart {{ margin:6px 0 10px }} .chart svg {{ width:100%; height:auto; display:block }}
 .chart figcaption {{ font-size:12px; color:var(--dim); margin-top:4px }}
 .chart .path {{ fill:none; stroke:var(--accent); stroke-width:1.6 }}
 .chart .area {{ fill:var(--accent); opacity:.09 }} .chart .dot {{ fill:var(--accent) }}
 .chart .lbl, .chart .val {{ font:12px var(--sans); fill:var(--dim) }}
 .chart .lbl {{ fill:var(--ink); font-weight:600 }}
 .chart rect.w, .key.w {{ fill:var(--dim); background:var(--dim) }}
 .chart rect.r, .key.r {{ fill:var(--accent); background:var(--accent) }}
 .key {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin:0 5px 0 10px;
   vertical-align:-1px }}
{design.BASE_CSS}</style></head><body>{design.nav('/research')}<div class="wrap">
<h1>One research task, question to actionable insight — run live</h1>
<p class="sub">Ask it the way you would ask a colleague. Eight engines answer in parallel, each
the same engine the <a href="/">console</a> uses, every figure from live Bitget, SEC, Yahoo or
news data and every line naming its source. It ends in one verdict and each engine's bottom line —
nothing on this page is written by a language model, and your question is read without one.</p>
<form method="post" action="/research">
 <input type="hidden" name="saved"><input type="hidden" name="memory">
 <div class="ask">
  <textarea name="q" rows="2" aria-label="your question">{esc(asked)}</textarea>
  <button>Run</button>
 </div>
 <p class="running" role="status" aria-live="polite" hidden></p>
 <details><summary>Or set the name, size and book yourself</summary>
  <div class="edit">
   <label>Name<input class="name" name="name" value="{esc(name)}"></label>
   <label>Size, %<input class="size" name="size" value="{size_pct:g}"></label>
   <label class="wide">Your book<input class="book" name="book" value="{esc(book_text)}"></label>
  </div>
  <p class="fine">A question in the box above wins; clear it to run these fields.</p>
 </details>
</form>
"""


def _reason_item(tested: Tested) -> str:
    """One reason: its verdict and the test behind it, then each other figure with its source
    and a link a reader can open to check it."""
    esc = html.escape
    said = ("<b>" + esc(tested.reason) + "</b> <span class='imp'>(implied, not stated)</span>"
            if tested.implied else f"<b>“{esc(tested.reason)}”</b>")
    found = "".join(
        f"<li>{esc(f.text)} <span class='src'>"
        + (f"<a href='{esc(f.url)}' rel='noopener' target='_blank'>{esc(f.source)}</a>"
           if f.url else esc(f.source))
        + "</span></li>" for f in tested.evidence)
    return (f"<li>{said} — <span class='r {tested.result.name.lower()}'>"
            f"{esc(tested.result.value)}</span>. {esc(tested.line)}"
            + (f"<ul class='ev'>{found}</ul>" if found else "") + "</li>")


def _conclusion(task: Task) -> str:
    """Whether to enter (`lui/weigh.py`), then how much (`lui/task.verdict`), then where the
    engines disagree, what would change the call and what only the trader can answer."""
    esc = html.escape
    call, weighed = task.verdict, task.weighing
    asked = next((s for s in task.steps if s.title == ASKED_TITLE), None)
    if call is None and weighed is None and not task.tested and asked is None:
        return ""

    def listed(title: str, items: tuple[str, ...]) -> str:
        return (f"<h3>{title}</h3><ul>" + "".join(f"<li>{esc(i)}</li>" for i in items)
                + "</ul>") if items else ""

    parts = ["<section class='verdict'><h2>Conclusion</h2>"]
    if asked is not None:
        # The question asked is answered first; the add-to-book call below is what the research
        # around it says about owning the name, and is headed so (judge's probe, 2026-09-29).
        parts.append(f"<h3>Your question</h3><p class='call'>"
                     f"{design.linked(asked.actionable or (asked.lines[0] if asked.lines else ''))}"
                     f"</p>" + (f"<h3>If you add {esc(task.name)} to your book</h3>"
                                if task.name else ""))
    if weighed is not None:
        parts.append(f"<p class='call'>{esc(weighed.call)}</p><p>{esc(weighed.reason)}</p>")
    if task.tested:
        parts.append("<h3>Your reasons, tested</h3><ul class='reasons'>" + "".join(
            _reason_item(t) for t in task.tested) + "</ul>")
    if call is not None:
        # "No measured edge" followed by a bare "How much: Add, at 15%" read to a newcomer as an
        # instruction to add (a first-time user, round 27): the sizing is conditional, and its
        # heading says so
        lead = (f"<h3>If you do add it: the size your risk budget allows</h3>"
                f"<p class='call2'>{esc(call.call)}</p>" if weighed is not None
                else f"<p class='call'>{esc(call.call)}</p>")
        parts.append(lead + "".join(f"<p>{esc(line)}</p>" for line in call.lines))
    if weighed is not None:
        parts += [listed("Where the engines disagree", weighed.disagreements),
                  listed("What would change the call", weighed.triggers),
                  listed("Questions only you can answer", weighed.questions)]
    return "".join(parts) + "</section>"


def _main(task: Task) -> str:
    """The task itself: what the question was read as, the verdict, each engine's bottom line and
    every step's card."""
    esc = html.escape
    conclusion = "".join(
        f"<li><b>{esc(title.rstrip('?'))}:</b> {esc(text[:1].upper() + text[1:])}</li>"
        for title, text in task.conclusion) or (
        "<li>Nothing to act on: " + esc(task.steps[0].lines[0] if task.steps and task.steps[0].lines
                                         else "no engine answered") + "</li>")
    cards = "".join(_card(n, step, _chart(task, step)) for n, step in enumerate(task.steps, 1))
    called = _conclusion(task)
    json_link = ("/research?" + urlencode({"q": task.asked, "format": "json"})
                 if task.asked else "/research?format=json")
    read = ""
    if task.reading is not None:
        notes = "".join(f"<li>{esc(n)}</li>" for n in task.reading.notes)
        read = (f"<p class='read'><b>Read as:</b> {esc(task.reading.summary)}</p>"
                + (f"<ul class='notes'>{notes}</ul>" if notes else ""))
    # This page auto-runs a worked example the instant it loads with nothing typed (`asked` is
    # empty only on that path — `_research_route` never reaches `research_task()` with an empty
    # `asked` any other way). Without a label the first thing a visitor sees is a full answer to a
    # question they never asked, read as though it were theirs (first-user audit, 2026-09-29).
    example = ("<p class='example'><span class='pill'>Example</span> Run live when this page "
               "loaded &mdash; not your question. Ask your own above.</p>"
               if not task.asked else "")
    return f"""{example}<p class="q">{esc(task.question)}</p>
{read}
{called}<section class="concl"><h2>What to do, engine by engine</h2><ol>{conclusion}</ol>
</section>
{cards}
<p class="sub">Ran at {task.ran_at:%H:%M} UTC on {task.ran_at:%d %b}, in {task.seconds:.1f}s.
Execution is sized on a ${DEFAULT_BOOK_VALUE:,.0f} book.
This is analysis, not advice — you make the call. <a href="{esc(json_link)}">JSON</a></p>
"""


_SCRIPT = """
<script>
// The page is one blocking POST: the engines run before a byte comes back, so without this the
// Run button looked dead for as long as the slowest engine took (audit, 2026-09-26).
(() => {
  const form = document.querySelector('form'), box = form.querySelector('textarea');
  // The trader's saved book and memory, kept by the console in this browser, travel with the
  // question; the server keeps neither.
  try {
    form.elements.saved.value = localStorage.getItem('argus.book') || '';
    form.elements.memory.value = localStorage.getItem('argus.memory') || '';
  } catch (e) {}
  const run = form.querySelector('button'), note = form.querySelector('.running');
  let timer = 0;
  form.addEventListener('submit', () => {
    const started = Date.now();
    run.disabled = true; run.textContent = 'Running';
    note.hidden = false;
    const tick = () => { note.textContent = 'Eight engines are reading live data: '
      + Math.round((Date.now() - started) / 1000) + ' s'; };
    tick(); timer = setInterval(tick, 1000);
  });
  // Enter asks and Shift+Enter starts a new line, as in the console; never mid-composition.
  box.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && box.value.trim()) {
      e.preventDefault(); form.requestSubmit();
    }
  });
  // Coming back with the browser's Back button restores this page from its cache as it was.
  addEventListener('pageshow', () => {
    clearInterval(timer); run.disabled = false; run.textContent = 'Run'; note.hidden = true;
  });
})();
</script>"""


def _bottom() -> str:
    return f"</div>{design.footer()}{_SCRIPT}</body></html>"


def _book_text(book: dict[str, float]) -> str:
    return ", ".join(f"{w:.0%} {s.removesuffix('USDT')}" for s, w in book.items())


def render_task(task: Task, favicon: str) -> str:
    """The task as a page: the question, the conclusion, then every step with its engine."""
    del favicon  # the head carries the mark itself (design.head)
    return (_top(task.asked or task.question, task.name, task.size_pct, _book_text(task.book))
            + _main(task) + _bottom())


# --- the same page, sent as the steps finish -----------------------------------------------------

_SWAP = """<script>
function argusStep(n) {
  const t = document.getElementById('st-' + n), slot = document.getElementById('slot-' + n);
  if (t && slot) slot.replaceWith(t.content.cloneNode(true));
}
function argusDone() {
  const t = document.getElementById('st-done'), live = document.getElementById('live');
  if (t && live) live.replaceWith(t.content.cloneNode(true));
}
</script>"""


def stream_start(asked: str, name: str, size_pct: float, book: dict[str, float],
                 titles: list[tuple[str, str]]) -> str:
    """The page before any engine has answered: the form, the question, and one waiting card per
    step. Sent at once, so the reader sees what is running instead of a blank tab
    (research/harvest/48-open-webui.md)."""
    esc = html.escape
    waiting = "".join(
        f"<article class='s wait' id='slot-{n}'><div class='h'><span class='n'>{n}</span>"
        f"<h2>{esc(title)}</h2><span class='e'>{esc(engine)} · running</span></div></article>"
        for n, (title, engine) in enumerate(titles, 1))
    return (_top(asked, name, size_pct, _book_text(book)) + _SWAP
            + f"<div id='live'><p class='q'>{esc(asked)}</p><p class='sub'>The verdict appears "
              f"once every engine has answered; each card fills in as its engine does.</p>"
              f"{waiting}</div>")


def stream_step(n: int, step: Step) -> str:
    """One finished step, swapped into its waiting card. Charts wait for the full page."""
    return f"<template id='st-{n}'>{_card(n, step)}</template><script>argusStep({n})</script>"


def stream_end(task: Task) -> str:
    """The finished task, replacing everything shown while it ran: the page a reader is left with
    is exactly the one :func:`render_task` draws."""
    return (f"<template id='st-done'>{_main(task)}</template><script>argusDone()</script>"
            + _bottom())


__all__ = ["price_chart", "render_task", "risk_chart", "stream_end", "stream_start",
           "stream_step"]
