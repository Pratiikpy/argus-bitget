"""The demo surface — the research workbench's console over HTTP.

``python -m argus.lui.server`` serves the console on http://127.0.0.1:8765.

**Why the standard library and nothing else.** Both tracks treat a broken demo as an invalidator,
so the demo is the one component where a dependency that fails to resolve costs the entry rather
than an afternoon. ``http.server`` ships with Python, has no build step, and cannot go stale. The
page is a single string in this file for the same reason: there is no bundler to run, no lockfile
to drift, and nothing to install before a judge can open it.

**What it deliberately does not do.** It never writes. The handler exposes exactly one verb — ask a
question, read the ledger, render the answer — and the ledger is opened fresh per request so the
page reflects the file on disk rather than a cached copy. There is no order path, no settings, no
authentication to misconfigure, because there is nothing behind it that could be changed. That is
also why binding is to loopback by default: this is a window onto a record, and a record that can
be read from anywhere is a decision for its owner to make, not a default to inherit.

**Concurrency.** Each request builds its own conversation unless the client sends one back, so two
people reading at once cannot resolve each other's "that". The client holds the turn history and
returns it; the server keeps no session state.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import threading
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from argus.lui import design
from argus.lui.answer import EMPTY_QUESTION_HINT, Answer, answer, unlead
from argus.lui.arbiter import arbitrate, gate_ledger_reading
from argus.lui.cli import BUDGET_MS
from argus.lui.ngram import reclassify
from argus.lui.provenance import labels as provenance_labels
from argus.lui.question import Conversation, Intent, classify, resolve_window
from argus.lui.research import (
    BARE_FOLLOW,
    ResearchKind,
    ResearchRequest,
    about_the_desk,
    about_the_record,
    follow_up,
    research_symbols,
    resolved_previous,
    with_book,
    worth_asking_the_model,
)
from argus.lui.research import detect as detect_research
from argus.lui.research import run as run_research
from argus.lui.research.parse import PRICE_FORECAST
from argus.lui.router import Router, build_router, route
from argus.paper.ledger import PaperLedger
from argus.truth.bounded import BoundedDict

HOST = "127.0.0.1"
PORT = 8765

FAVICON = design.favicon()
OG_PNG = Path(__file__).with_name("og.png")
"""The link-preview card, `design.OG_IMAGE`."""
"""An inline SVG, not a bundled file. Every route in this server is one string in one module —
adding a binary asset and a route to serve it would be the first exception to that, for a mark a
judge never consciously looks at. Its absence was not silent, though: an unstyled 404 for
`/favicon.ico` was the one console error on an otherwise-clean page, found driving the console as
a first-time judge would (2026-09-22) rather than by reading the code and assuming it was fine."""

PAGE = """<!doctype html>
<html lang="en"><head>__HEAD__
<style>__TOKENS__
  body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,sans-serif; }
  .sub { margin:0 0 20px }
  .bar { display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap }
  input[type=text], textarea#q { flex:1 1 320px; min-width:0; padding:11px 13px;
    border:1px solid var(--line); border-radius:8px; background:var(--panel); color:var(--ink);
    font-size:15px }
  textarea#q { font:15px/1.4 system-ui,sans-serif; resize:none; overflow-y:auto; max-height:320px;
    box-sizing:border-box }
  /* On a phone the example question wraps to two lines; one row cut it in half before the
     reader typed anything (QA screen pass, 2026-09-28). */
  @media (max-width: 640px) { textarea#q { min-height:calc(2 * 1.4em + 24px) } }
  .deep { margin-top:10px }
  .deep button { padding:0; border:0; background:none; color:var(--accent); font-size:13.5px;
    font-weight:600; cursor:pointer; text-decoration:underline; text-underline-offset:3px }
  button { padding:11px 18px; font-size:15px; cursor:pointer }
  button:disabled { opacity:.55; cursor:default }
  .chips { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:22px }
  .chip { font:inherit; border:1px solid var(--line); background:var(--panel); color:var(--dim);
    border-radius:999px; padding:5px 11px; font-size:12.5px; line-height:1.4; cursor:pointer;
    text-align:left }
  .chip:hover, .chip:focus-visible { color:var(--ink); border-color:var(--accent) }
  .chip:focus-visible { outline:2px solid var(--accent); outline-offset:2px }
  .card { scroll-margin-top:84px; background:var(--panel); border:1px solid var(--line);
    padding:14px 16px; margin-bottom:12px }
  .q { font-weight:600; margin-bottom:8px }
  .meta { font:11.5px/1.4 var(--mono); color:var(--dim); margin-bottom:9px;
    display:flex; gap:10px; flex-wrap:wrap }
  .tag { border:1px solid var(--line); border-radius:4px; padding:1px 6px }
  .over { color:var(--warn); border-color:var(--warn) }
  .refused .q { color:var(--warn) }
  .line { margin:3px 0; white-space:pre-wrap; overflow-wrap:anywhere }
  .src { margin-top:14px; padding:12px 14px; border:1px solid var(--line); border-radius:10px;
    background:var(--bg); font:11.5px/1.6 var(--mono); color:var(--dim); overflow-wrap:anywhere }
  .src .rk { display:block; font:600 10.5px/1 var(--mono); letter-spacing:.14em;
    text-transform:uppercase; color:var(--proof); margin-bottom:8px }
  .src b { color:var(--ink); font-weight:600 }
  .empty { color:var(--dim); font-size:13.5px }
  .book { display:flex; gap:8px; align-items:center; margin:-6px 0 16px; flex-wrap:wrap }
  .book label { font-size:12.5px; color:var(--dim); white-space:nowrap }
  .book input { flex:1 1 280px; min-width:0; padding:7px 10px; border:1px solid var(--line);
    border-radius:7px; background:var(--panel); color:var(--ink); font:13px var(--mono) }
  .book .saved { font-size:12px; color:var(--ok) }
  .book .saved .clear { background:none; border:0; padding:0; font:inherit; color:var(--accent);
    text-decoration:underline; cursor:pointer }
  .group { font-size:11.5px; letter-spacing:.06em; text-transform:uppercase; color:var(--dim);
    margin:0 0 6px }
  .line.act { font-weight:600; color:var(--ink); border-left:2px solid var(--ink);
    padding-left:8px }
  .line.hedge { font-weight:600 }
  .line.fine { color:var(--dim); font-size:13px }
  /* Lowercase and light: the tag says where a line comes from without shouting over the
     sentence it labels (first-user audit, 2026-09-29: "reads as technical noise"). */
  .pv { display:inline-block; min-width:4.6em; margin-right:.5em; padding:0 .35em;
        border-radius:3px;
        font:500 10.5px/1.6 var(--mono); letter-spacing:0;
        vertical-align:1px; text-align:center;
        color:var(--dim); border:1px solid color-mix(in srgb, var(--dim) 45%, transparent) }
  .pv-live { color:var(--accent); border-color:color-mix(in srgb, var(--accent) 55%, transparent) }
  .pv-record { font-style:italic }
  .pv-memory { color:var(--ink); border-style:dotted }
  .pv-explained { color:var(--dim) }
  .mem { display:flex; gap:6px; flex-wrap:wrap; align-items:center; margin:-8px 0 16px;
    font-size:12.5px; color:var(--dim) }
  .mem .fact { border:1px dotted var(--line); border-radius:999px; padding:3px 9px;
    background:var(--panel); color:var(--ink) }
  .mem button { padding:0 0 0 6px; border:0; background:none; color:var(--dim); font-size:12.5px;
    cursor:pointer }
  .pv-missing, .pv-assumed { border-style:dashed }
  .fb { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-top:12px;
    font-size:12.5px; color:var(--dim) }
  .fb button { padding:3px 11px; border:1px solid var(--line); background:var(--panel);
    color:var(--ink); border-radius:999px; font-size:12.5px; cursor:pointer }
  .fb button:hover, .fb button:focus-visible { border-color:var(--accent) }
  .more { display:block; margin:8px 0 2px; padding:6px 12px; border:1px solid var(--line);
    background:var(--panel); color:var(--accent); font:600 12.5px system-ui,sans-serif;
    border-radius:8px; cursor:pointer }
  .more:hover, .more:focus-visible { border-color:var(--accent) }
  .stat { font:12px/1.6 var(--mono); color:var(--dim); margin:0 0 14px }
  .jump { display:flex; gap:10px 22px; flex-wrap:wrap; margin:0 0 30px; font-weight:500 }
  .jump a { text-decoration:none; color:var(--ink); border-bottom:1px solid var(--line);
    padding-bottom:2px }
  .jump a:hover { border-color:var(--ink) }
__BASE__</style></head><body>__NAV__<div class="wrap">
<p class="kicker">ARGUS · research workbench for Bitget</p>
<h1>Ask the desk. Every number comes with its source.</h1>
<p class="sub">Ask about anything Bitget lists — stocks, ETFs, gold, oil, crypto. The desk's own
engines compute every figure from live data and name the source; the language model only reads your
question and never writes a number. No source, no answer: you get a refusal and the reason.</p>
<p class="stat"><span id="stat"></span></p>
<p class="jump"><a href="/research">Run a full research task &rarr;</a>
<a href="/proof">What we beat &rarr;</a><a href="/wrong">What we got wrong &rarr;</a>
<a href="/status">What the desk can see &rarr;</a>
<a href="/materials">Every deliverable on one page &rarr;</a></p>

<form class="bar" id="f">
  <textarea id="q" rows="1" autocomplete="off" aria-label="your question"
    placeholder="I hold 50% NVDA, 50% AAPL — what does adding 20% TSLA do to my risk?"></textarea>
  <button id="go">Ask</button>
</form>
<div class="book">
  <label for="book">My book</label>
  <input type="text" id="book" autocomplete="off"
    placeholder="optional, e.g. 40% NVDA, 30% MSFT, 30% AAPL, risk budget 20%"
    title="Used by every research question that does not name its own holdings">
  <span class="saved" id="saved"></span>
</div>
<div class="mem" id="mem" hidden></div>
<p class="group">Research</p>
<div class="chips" id="chips-research"></div>
<p class="group">The desk's own record</p>
<div class="chips" id="chips"></div>
<div id="out" role="log" aria-live="polite" aria-label="Answers">
<p class="empty">Ask something, or pick one of the suggestions above.</p></div>
</div><script>
// "sell half of that" is deliberately a question the console REFUSES — an order — so a visitor who
// only clicks chips still meets a refusal rather than only the happy path. (A second chip, "what
// is gold trading at", used to be refused too; gold trades on Bitget and is now answered, so it
// moved to research as a non-desk example.)
// "what did the risk layer block" is here because Track 2 scores risk-control effectiveness and a
// capability nobody can find is a capability nobody credits.
// Research first: Track 3 judges a question-to-insight workbench, and these are the questions its
// Open Theme names — trade impact on a book, stress, comparison, execution — then the Bitget
// Skills and data server by name: technicals, the earnings calendar, and past analogues.
// A pasted table of fills is a question too (`lui/journal.py` reads it); this chip puts an
// example in the box so a trader sees the shape, and the box keeps its line breaks.
const REVIEW_CHIP = "review my trades (paste a table of fills)";
const REVIEW_EXAMPLE = ["review my trades", "time,symbol,side,price,qty",
  "2026-09-22 14:05,NVDAUSDT,buy,176.40,20", "2026-09-23 15:10,NVDAUSDT,sell,181.20,20",
  "2026-09-24 14:30,TSLAUSDT,buy,262.10,10", "2026-09-25 16:45,TSLAUSDT,sell,255.30,10"]
  .join("\\n");
const RESEARCH = ["I hold 50% NVDA, 50% AAPL — what does adding 20% TSLA do to my risk?",
  "what if the Nasdaq drops 10%? I hold 40% MSFT, 30% META, 30% GOOGL",
  "is TSLA riskier than NVDA", "should I buy MSTR", "where is NVDA trading right now",
  "how should I split a $50k order in NVDA", "is TSLA overbought",
  "when does NVDA report earnings", "what did NVDA's latest 10-Q say drove data center revenue",
  "has COIN been here before", "compare gold and bitcoin", REVIEW_CHIP,
  // The newest engines, reachable from the first screen (audit, 2026-09-26).
  "what are my sector and factor exposures? I hold 40% NVDA, 30% MSFT, 30% AAPL",
  "what should I keep an eye on this week? I hold 50% NVDA, 50% BTC",
  "where is NVDA trading and when does it report earnings",
  "I can't lose more than 10% and I'm a swing trader", "英伟达什么时候发布财报"];
const SUGGEST = ["why did you do nothing all weekend","what is the sharpe","show me decision 25",
  "what did the risk layer block","what evidence backed that","is the log tamper-evident",
  "are you well calibrated","what bad decision patterns do you have","what is my position",
  "sell half of that"];
const out = document.getElementById('out'), qEl = document.getElementById('q');
// The box grows with what is typed or pasted; Enter asks, Shift+Enter starts a new line, and an
// input method still composing (Chinese, Japanese) keeps its Enter.
const grow = () => { qEl.style.height = 'auto'; qEl.style.height = qEl.scrollHeight + 2 + 'px'; };
qEl.addEventListener('input', grow);
qEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault(); document.getElementById('f').requestSubmit();
  }
});
const bookEl = document.getElementById('book'), savedEl = document.getElementById('saved');
let turns = [], first = true, askSeq = 0;
// Our own visits are marked so the usage count (`lui/usage.py`) is of other people: opening the
// console once with ?internal=1 sets the flag in this browser.
let internal = '';
try {
  if (new URLSearchParams(location.search).get('internal') === '1')
    localStorage.setItem('argus.internal', '1');
  internal = localStorage.getItem('argus.internal') === '1' ? '1' : '';
} catch (e) {}
// Questions, books and memory go in a POST body: a query string is written to the host's access
// log, and what a visitor types is theirs.
const post = (path, fields, signal) => fetch(path, {method: 'POST', signal,
  headers: {'Content-Type': 'application/x-www-form-urlencoded'},
  body: new URLSearchParams({...fields, internal})});
const ASK_LIMIT_MS = 90000;

// The book is the visitor's own and stays in their browser; it is sent with each question and
// never stored on the server.
// A restored book says so, with a way to clear it: it came back silently on every visit with no
// sign it was remembered and no control to forget it (a first-time-user audit, 2026-09-30).
const showSaved = () => {
  savedEl.innerHTML = bookEl.value.trim()
    ? 'saved in this browser · <button type="button" class="clear" id="clearbook">clear</button>'
    : '';
};
try { bookEl.value = localStorage.getItem('argus.book') || ''; } catch (e) {}
showSaved();
bookEl.addEventListener('change', () => {
  try { localStorage.setItem('argus.book', bookEl.value.trim()); } catch (e) {}
  showSaved();
});
savedEl.addEventListener('click', e => {
  if (!e.target.closest('#clearbook')) return;
  bookEl.value = '';
  try { localStorage.removeItem('argus.book'); } catch (err) {}
  showSaved();
  bookEl.focus();
});

// What the trader has told the console ("I can't lose more than 10%", "I think NVDA runs on AI
// capex"): kept in this browser, sent with each question, shown here, forgettable one by one.
const memEl = document.getElementById('mem');
let memory = [];
try { memory = JSON.parse(localStorage.getItem('argus.memory') || '[]'); }
catch (e) { memory = []; }
function saveMemory() {
  try { localStorage.setItem('argus.memory', JSON.stringify(memory)); } catch (e) {}
  memEl.hidden = !memory.length;
  memEl.innerHTML = memory.length ? '<span>Remembered:</span>' + memory.map((f, i) =>
    `<span class="fact">${esc0((f.kind === 'check' ? 'Checklist: ' : '') + f.text)}` +
      `<button data-i="${i}" title="forget this" ` +
      `aria-label="forget: ${esc0(f.text).replace(/"/g, '&quot;')}">&times;</button>` +
    `</span>`).join('') : '';
}
memEl.addEventListener('click', e => {
  if (e.target.dataset.i === undefined) return;
  memory.splice(Number(e.target.dataset.i), 1); saveMemory();
});
saveMemory();

for (const [id, list] of [['chips-research', RESEARCH], ['chips', SUGGEST]]) {
  const el = document.getElementById(id);
  // Buttons, not spans: the suggestions were reachable by mouse only, never by Tab or a screen
  // reader (first-user audit, 2026-09-29). A button is focusable and fires on Enter and Space.
  el.innerHTML = list.map(s => `<button type="button" class="chip">${esc0(s)}</button>`)
    .join('');
  el.addEventListener('click', e => {
    if (!e.target.classList.contains('chip')) return;
    if (e.target.textContent === REVIEW_CHIP) {
      qEl.value = REVIEW_EXAMPLE; grow(); qEl.focus(); return;
    }
    // A question typed and not yet asked survives a suggestion: tabbing from the box to a chip and
    // pressing Enter asked the chip and threw the typed words away (a first-time user, round 10).
    // The chip is asked, and the draft is put back in the box to ask next.
    const draft = qEl.value.trim() && qEl.value !== e.target.textContent ? qEl.value : '';
    qEl.value = e.target.textContent; document.getElementById('f').requestSubmit();
    if (draft) { qEl.value = draft; grow(); }
  });
}
function esc0(s) {
  return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}
const PV = {
  live: 'read from a source just now',
  computed: 'worked out for this answer from live data',
  record: 'a measured past record, with its sample named',
  desk: "quoted from the desk's own logged decision",
  assumed: 'a default applied because the question did not say',
  memory: 'something you told the console earlier, kept in this browser',
  missing: 'could not be read or checked',
  explained: 'written into the console to explain a term or a rule; nothing in it is ' +
    'computed for this answer'};
const pv = t => t ? `<span class="pv pv-${t}" title="${PV[t]}">${t}</span>` : '';
const lineClass = l => /^(Actionable|Bottom line)(?: \\([^)]*\\))?:/.test(l) ? 'line act'
  : l.startsWith('Hedge:') ? 'line hedge'
  : (l.startsWith('Assumed:') || l.startsWith('Data:')) ? 'line fine' : 'line';

// A long answer (research and portfolio questions run past eight lines routinely) used to land as
// one wall of text; a first-time reader had no way to tell the lead from the supporting detail
// (first-user audit, 2026-09-29). Past eight lines, only the first six show — the lead and its
// closest support — behind a real button rather than a link, so it is reachable by keyboard and
// announced by a screen reader as expandable. Source badges (`pv()`) are baked into every line,
// shown or hidden, so nothing loses its provenance by being collapsed. A refusal is never
// collapsed: the reason a question was refused is the one thing this console must never bury.
let cardSeq = 0;
const collapseLines = (divs, refused) => {
  if (refused || divs.length <= 8) return divs.join('');
  const id = 'more-' + (++cardSeq);
  const label = `Show all ${divs.length} lines`;
  // A source link in the folded lines is repeated under the button, so the answer's citation is
  // one click away whatever is folded (first-user audit, 2026-09-29: the only link sat at line 20
  // of 20, hidden until expanded).
  const folded = [...new Set([...divs.slice(6).join('')
    .matchAll(/<a href="[^"]+"[^>]*>[^<]+<\\/a>/g)].map(m => m[0]))];
  const cite = folded.length
    ? `<div class="line fine">Sources in the folded lines: ${folded.join(' · ')}</div>` : '';
  return divs.slice(0, 6).join('') +
    `<div class="rest" id="${id}" hidden>${divs.slice(6).join('')}</div>` +
    `<button type="button" class="more" aria-expanded="false" aria-controls="${id}" ` +
    `data-label="${escA(label)}">${label}</button>` + cite;
};

fetch('status').then(r => r.json()).then(s => {
  // The age is shown unconditionally, not only when stale. A figure with no date beside it invites
  // the reader to assume it is current, and the chain of a stale snapshot verifies perfectly.
  let age = '';
  if (s.age_hours === null || s.age_hours === undefined) {
    age = ' Age unknown \u2014 which is not the same as current.';
  } else if (s.stale) {
    age = ` \u26a0 Last decision ${s.age_hours}h ago \u2014 a scheduled cycle was missed, ` +
      `so this record is behind.`;
  } else {
    const next = s.next_cycle_at ? new Date(s.next_cycle_at) : null;
    const until = next ? Math.max(0, Math.round((next - Date.now()) / 36e5 * 10) / 10) : null;
    age = ` Last decision ${s.age_hours}h ago` + (until !== null ? `, next in ${until}h.` : '.');
  }
  // Plain language before the jargon: a first-time reader met "836 decisions on record, chain
  // intact" with no idea what a "decision" was or whose money was on the line (first-user audit,
  // 2026-09-29). What it is, then the live count.
  document.getElementById('stat').textContent =
    `This is ARGUS's own paper desk: it decides four times a day during US market hours, with ` +
    `no real money, and every decision is written into a tamper-evident record. ${s.entries} ` +
    `decisions logged so far, chain ${s.chain_intact ? 'unbroken' : 'BROKEN'}.` + age;
}).catch(() => {});

const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const escA = s => esc(s).replace(/"/g, '&quot;');
// A source's address becomes a link reading its host (first-user audit, 2026-09-29: every
// citation was plain text, so "every number comes with its source" could be read, not opened).
const linked = s => esc(s).replace(
  /https?:\\/\\/[^\\s<>"']*[^\\s<>"'.,;:!?)\\]]/g,
  u => `<a href="${u}" rel="noopener noreferrer" target="_blank">` +
    `${u.replace(/^https?:\\/\\/(?:www\\.)?([^\\/?#]+).*$/, '$1')} &#8599;</a>`)
  // "bps" explained where it stands: a newcomer met it in every cost line with no definition
  // anywhere on the page (a first-time-user audit, 2026-09-29).
  .replace(/(\\d)bps\\b/g,
    '$1<abbr title="basis points: hundredths of a percent, so 100bps is 1%">bps</abbr>');

document.getElementById('f').addEventListener('submit', async ev => {
  ev.preventDefault();
  const text = qEl.value.trim(); if (!text) return;
  qEl.value = ''; grow(); document.getElementById('go').disabled = true;
  if (first) { out.innerHTML = ''; first = false; }
  const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), ASK_LIMIT_MS);
  // The question stays on screen with a running count while the desk works: a slow answer showed
  // only a greyed-out button and an empty box, and read as stuck (first-user audit, 2026-09-29).
  // Each question holds its own place: the answer replaces this card where it stands, so two
  // questions in flight keep the order they were asked in. A slow answer used to be inserted at
  // the top when it finished, above a newer fast one (a first-user audit, 2026-09-30).
  const slotId = `pending-${++askSeq}`;
  out.insertAdjacentHTML('afterbegin', `<div class="card pending" id="${slotId}">` +
    `<div class="q">${esc(text)}</div><div class="meta"><span class="tag">working · ` +
    `<span class="tick">0</span> s</span></div></div>`);
  const place = html => {
    const slot = document.getElementById(slotId);
    if (!slot) { out.insertAdjacentHTML('afterbegin', html); return out.firstElementChild; }
    slot.insertAdjacentHTML('beforebegin', html);
    const card = slot.previousElementSibling;
    slot.remove();
    return card;
  };
  let placed = null;
  // Brought into view when it lands in the lower half of the screen, as it does under the examples
  // on a phone and on a portrait tablet; the answer replaces it in place, so the reader is already
  // looking at where it arrives. The line used to be "below the screen", which on an 820x1180
  // tablet left the answer starting 119px from the bottom edge (a first-user audit, 2026-09-30).
  const inFlight = document.getElementById(slotId);
  const bringIntoView = card => {
    if (card && card.getBoundingClientRect().top > innerHeight * 0.5) {
      card.scrollIntoView({block: 'start', behavior:
        matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'});
    }
  };
  bringIntoView(inFlight);
  const t0 = Date.now(), ticker = setInterval(() => {
    const tick = document.querySelector(`#${slotId} .tick`);
    if (tick) tick.textContent = String(Math.round((Date.now() - t0) / 1000));
  }, 1000);
  try {
    const r = await post('ask', {q: text, turns: JSON.stringify(turns),
      book: bookEl.value.trim(), memory: JSON.stringify(memory)}, ctl.signal);
    if (!r.ok) throw new Error(`http ${r.status}`);
    const a = await r.json();
    turns = a.turns || turns;
    if (a.memory) { try { memory = JSON.parse(a.memory); } catch (e) {} saveMemory(); }
    // Plain elapsed time. The per-intent budget is an engineering target, and stamping a correct
    // answer "OVER BUDGET" in the warning colour read as a failure (audit, 2026-09-26).
    placed = place(`
      <div class="card ${a.refused ? 'refused' : ''}">
        <div class="q">${esc(text)}</div>
        <div class="meta">
          <span class="tag">answered in ${(a.elapsed_ms / 1000).toFixed(1)} s</span>
          ${a.refused ? '<span class="tag over">refused</span>' : ''}
          ${a.model_left !== undefined && a.model_left <= 10 ? `<span class="tag" title="` +
            `Counted per network address. After these, the console's own readers answer, in ` +
            `English, until the hour is up.">` +
            (a.model_left > 0 ? `language model: ${a.model_left} question` +
              `${a.model_left === 1 ? '' : 's'} left this hour` :
              `language model paused` + (a.model_back_in ? ` for ~${a.model_back_in} min` : '') +
              ` — answers still computed live`) + `</span>` : ''}
        </div>
        <div class="lines">${collapseLines(a.lines.map((l, i) =>
          `<div class="${lineClass(l)}">${pv((a.line_labels || [])[i])}${linked(l)}</div>`),
          a.refused)}</div>
        ${a.sources.length ? `<div class="src"><span class="rk">Receipt · ${a.sources.length}` +
          ` source${a.sources.length === 1 ? '' : 's'}</span>` +
          a.sources.map(s => `&nbsp;&nbsp;<b>${esc(s.kind)}</b>:${linked(s.ref)}` +
            (s.detail ? ' — ' + linked(s.detail) : '')).join('<br>') + `</div>` : ''}
        ${a.research_task ? `<form class="deep" method="post" action="/research">` +
          `<input type="hidden" name="q" value="${escA(text)}">` +
          `<input type="hidden" name="book" value="${escA(bookEl.value.trim())}">` +
          `<button>Run the full eight-engine research task on this &rarr;</button></form>` : ''}
        ${a.answer_id ? `<div class="fb" data-id="${esc(a.answer_id)}">` +
          `<span>Did this answer your question?</span><button type="button" data-u="1">Yes` +
          `</button><button type="button" data-u="0">No</button></div>` : ''}
      </div>`);
    if (a.translate) {
      // English first, then the reader's language: every figure in the translation was checked
      // against the English by the server before it was sent (`lui/translate.py`).
      const card = placed;
      const body = a.lines.slice(a.translate.skip);
      post('translate', {lang: a.translate.lang, token: a.translate.token,
        lines: JSON.stringify(body)})
        .then(r => r.ok ? r.json() : null).then(t => {
          if (!t || !t.lines) return;
          // Each translated line keeps the styling of the English line it came from.
          const note = t.note ? [t.note] : a.lines.slice(0, a.translate.skip);
          const lines = card.querySelector('.lines');
          lines.lang = a.translate.lang;
          lines.innerHTML =
            note.map(l => `<div class="line fine">${linked(l)}</div>`).join('') +
            collapseLines(t.lines.map((l, i) => `<div class="${lineClass(body[i])}">` +
              `${pv((a.line_labels || [])[a.translate.skip + i])}${linked(l)}</div>`), a.refused);
        }).catch(() => {});
    }
  } catch (e) {
    // Said plainly, and the question is put back so it is not lost: a raw exception ("Failed to
    // fetch", "Unexpected token <") told the reader nothing they could act on.
    const why = e.name === 'AbortError'
      ? `The desk did not answer within ${ASK_LIMIT_MS / 1000} seconds, so the request was stopped.`
      : String(e.message).startsWith('http ')
        ? `The desk hit an error answering this (${esc(e.message.toUpperCase())}).`
        : 'The console could not reach the desk: the connection dropped or the server is down.';
    placed = place(
      `<div class="card refused"><div class="q">${esc(text)}</div>
       <div class="line">${why} Your question is back in the box — ask again, or ask something
       narrower.</div></div>`);
    if (!qEl.value) { qEl.value = text; grow(); }
  } finally {
    clearTimeout(timer); clearInterval(ticker);
    const leftover = document.getElementById(slotId); if (leftover) leftover.remove();
    // Again once the answer is in: while it was pending the page was often too short to scroll
    // that far. The box is refocused only where that opens no on-screen keyboard over the answer.
    bringIntoView(placed);
    document.getElementById('go').disabled = false;
    if (!matchMedia('(pointer: coarse)').matches) qEl.focus({preventScroll: true});
  }
});
// "Show all N lines": a real button toggled by a click or, natively, by Enter/Space on it, with
// aria-expanded kept true to what is on screen for a screen reader.
out.addEventListener('click', e => {
  const more = e.target.closest('.more');
  if (more) {
    const rest = document.getElementById(more.getAttribute('aria-controls'));
    if (!rest) return;
    const expand = rest.hidden;
    rest.hidden = !expand;
    more.setAttribute('aria-expanded', String(expand));
    more.textContent = expand ? 'Show fewer lines' : more.dataset.label;
    return;
  }
  const b = e.target.closest('.fb button'); if (!b) return;
  const box = b.parentElement;
  post('feedback', {id: box.dataset.id, useful: b.dataset.u}).catch(() => {});
  box.textContent = b.dataset.u === '1' ? 'Thanks \u2014 noted as answered.'
    : 'Thanks \u2014 noted as not answered. Rephrasing, or naming the ticker, often helps.';
});
// A link can carry a question (and a book) so it demonstrates an engine in one click: /proof
// links every capability the console runs this way. The book fills the field for this visit
// only; it is saved only if the visitor edits it.
const asked = new URLSearchParams(location.search);
if (asked.get('book')) bookEl.value = asked.get('book').slice(0, 300);
if (asked.get('q')) { qEl.value = asked.get('q').slice(0, 500);
  document.getElementById('f').requestSubmit(); }
qEl.focus();
</script>__FOOT__</body></html>"""
for _token, _value in (("__HEAD__", design.head(
                            "ARGUS — ask the desk",
                            "A research workbench for Bitget: ask about stocks, ETFs, gold or "
                            "crypto and every number comes with its source.")),
                       ("__FAVICON__", FAVICON), ("__TOKENS__", design.TOKENS_CSS),
                       ("__BASE__", design.BASE_CSS), ("__NAV__", design.nav("/")),
                       ("__FOOT__", design.footer()), ("__FONTS__", design.FONTS)):
    PAGE = PAGE.replace(_token, _value)
# `.replace()`, not an f-string or `.format()`: the page above is thousands of characters of
# literal CSS and JS, and both of those already use `{`/`}` constantly — an f-string would need
# every one of them doubled, and a plain `.format()` call would raise on the first unescaped
# brace. `__FAVICON__` is a token that cannot collide with anything CSS or JS would ever contain.


def _ledger_path() -> Path:
    """Where the record lives.

    ``ARGUS_DATA_DIR`` wins when set. The default derives the path from this source tree's layout,
    which is correct when running from a checkout and wrong the moment the package is copied into
    a deployment bundle — the hosted console would otherwise look for the ledger in a directory
    that does not exist and report an empty record as though the desk had never decided anything.
    An empty ledger is a claim, so it must not be producible by a path mistake.
    """
    override = os.environ.get("ARGUS_DATA_DIR", "").strip()
    if override:
        return Path(override) / "paper_ledger.jsonl"

    from argus.paper.runner import LEDGER_PATH

    return LEDGER_PATH


def repair_mojibake(text: str) -> str:
    """Undo a latin-1 round trip on a query string, when one actually happened.

    **Honest provenance, because the first version of this docstring was wrong.** It was written to
    explain why Chinese questions were refused on the hosted deployment while working in process.
    They were not. The `/echo` route, added to stop the guessing, showed the hosted runtime
    receiving ``%3F%3F%3F%3F%3F%3F%3F`` — seven literal question marks for seven Han characters —
    and the loss was happening in the *test harness*: a Windows shell was replacing the characters
    before curl ever encoded them. Driven from Python, every Chinese phrasing answered correctly on
    the live URL. A harness bug reported as a product bug is its own kind of failure, and this
    paragraph exists so nobody re-derives the wrong cause from the code.

    The function is kept because the class of fault is real even though this instance was not: a
    runtime that decodes a path as latin-1 is a normal thing to meet, and the repair is free when
    there is nothing to repair. **It has never fired in production.**

    The repair itself is the standard one. UTF-8 bytes decoded as latin-1 produce a string whose
    characters all sit below U+0100, and re-encoding to latin-1 recovers the original bytes. Two
    guards keep it from damaging text that was never broken:

    * A string that is pure ASCII is returned untouched — there is nothing to repair.
    * The repair is kept **only if it produces characters outside latin-1**. Genuine European text
      ("café", "naïve", "Ölpreis") survives the round trip unchanged and so is rejected by this
      test, while mojibake turns into the Han, Cyrillic or emoji characters it was meant to be.

    Without the second guard this would mangle every accented word a European judge types, which is
    a larger bug than the one it was written to fix.
    """
    if text.isascii():
        return text
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    # Only a repair that reaches beyond latin-1 is evidence that a latin-1 round trip happened.
    return repaired if any(ord(c) > 0xFF for c in repaired) else text


_PRONOUN = re.compile(r"\b(?:that|it|this|those|them|its)\b", re.I)
_NAME_SWAP = re.compile(r"^\s*(?:and\s+)?(?:what|how)\s+(?:about|abt|bout)\s+\S+\s*\??\s*$|"
                        r"^\s*and\s+(?:for\s+)?\S+\s*\??\s*$|^\s*(?:(?:actually|sorry|no|oops|"
                        r"wait)[,\s]+)*(?:i\s+)?(?:meant|mean)\b", re.I)
_ANOTHER_PERIOD = re.compile(
    r"^\s*(?:and\s+|so\s+)?(?:what|how)?\s*(?:about\s+)?(?:over\s+|for\s+|if\s+)?(?:the\s+"
    r"(?:last|past)\s+)?(?P<n>\d+|a|one|two|three|four|five|ten)\s+(?P<unit>years?|months?)"
    r"(?:\s+ago)?\s*[?.!]*\s*$", re.I)
"""A follow-up that changes only the period: "what about 5 years?"."""

def _sentence_case(text: str) -> str:
    """``text`` continued mid-sentence: its first letter lowered unless the first word is a name or
    a ticker ("TSLA carries..." stays, "The book..." becomes "the book...")."""
    first = text.split(" ", 1)[0]
    return text if any(c.isupper() for c in first[1:]) else text[:1].lower() + text[1:]


def riskexplain_asked(text: str) -> bool:
    """Whether the question is about the risk layer's mechanism, which has its own answer."""
    from argus.lui import riskexplain

    return bool(riskexplain.RISK_HOW_Q.search(text))


_ALLOWANCE_ASKED = re.compile(
    r"\bhow\s+many\s+(?:more\s+)?(?:questions|queries|asks|messages)\b[^?]{0,30}\b(?:left|remaining|"
    r"do\s+i\s+get|can\s+i\s+ask)|\b(?:question|model|usage|rate)\s+(?:limit|allowance|quota)\b|"
    r"\bquestions?\s+(?:do\s+i\s+have\s+)?left\b", re.I)
"""A question about the hourly allowance itself."""

_WHY_THAT = re.compile(
    # "why not?" after "why did you skip NVDA?" read the whole record's abstentions (round 10)
    r"^\s*(?:but\s+|and\s+|so\s+|ok\s+|okay\s+)?why(?:\s+not)?\s*[?.!]*\s*$|"
    # "are you sure?" after a thesis answer was declined (a first-time user, round 12)
    r"^\s*(?:but\s+|and\s+|so\s+)?(?:are\s+you\s+sure|you\s+sure|really|is\s+that\s+(?:right|true|"
    r"correct)|how\s+(?:sure|confident)\s+are\s+you(?:\s+about\s+(?:that|this|it))?)\s*[?.!]*\s*$|"
    r"^\s*(?:but\s+|and\s+)?why\s+(?:do|did|would)\s+you\s+(?:say|think|conclude|recommend|suggest)"
    r"\s+(?:that|this|so)\b[^?]*[?.!]*\s*$|"
    r"^\s*(?:how|why)\s+(?:do|did)\s+you\s+(?:know|get|work\s+(?:that|it)\s+out)\b[^?]*[?.!]*\s*$|"
    r"^\s*(?:what'?s|what\s+is)\s+(?:that|this)\s+based\s+on\s*[?.!]*\s*$|"
    r"^\s*(?:explain|show)\s+(?:me\s+)?(?:your|the)\s+(?:reasoning|working|evidence)\s*[?.!]*\s*$|"
    # "Explain your previous answer: why does ETH carry 58% of the risk?" reached the Ethereum
    # concept card (a judge, round 14)
    r"^\s*(?:please\s+)?(?:explain|justify|defend)\s+(?:your|the)\s+(?:previous|last|prior|earlier)"
    r"\s+(?:answer|reply|response|number|figure)s?\b[^?]*\??[^?]*[?.!]*\s*$",
    re.I)
"""A request for the reasons behind the previous answer."""

_EXPLAIN_PREV_Q = re.compile(
    r"^\s*(?:please\s+)?(?:explain|justify|defend)\s+(?:your|the)\s+(?:previous|last|prior|earlier)"
    r"\s+(?:answer|reply|response|number|figure)s?\s*[:\-—]\s*(?P<q>[^\n]{12,}\?)\s*$", re.I)
"""A request to explain the last answer that carries its own question after a colon."""

_TRUST = re.compile(
    r"\bwhich\s+(?:of\s+)?(?:the\s+|these\s+|those\s+)?(?:numbers?|figures?|answers?|claims?|"
    r"results?)\b[^?]{0,80}\b(?:trust|rely\s+on|believe|weak(?:est)?|least\s+(?:reliable|solid)|"
    r"most\s+(?:reliable|solid))\b|\bwhich\s+of\s+(?:the\s+)?(?:things|numbers|figures)\s+you'?ve\s+"
    r"(?:told|said|given)\b[^?]{0,80}\b(?:trust|weak|reliable)", re.I)
""""Which numbers should I trust and which are weakest?" — asked of the whole conversation."""

_SIMPLER = re.compile(
    r"^\s*(?:(?:can\s+you\s+|please\s+)?(?:explain|say|put)\s+(?:it|that|this)?\s*(?:again\s+)?"
    r"(?:simpler|more\s+simply|simply|in\s+simple(?:r)?\s+(?:terms|words|english)|in\s+plain\s+"
    r"(?:english|words|terms)|like\s+i'?m\s+(?:5|five|new))|simpler(?:\s+please)?|eli5|"
    r"in\s+plain\s+english|i\s+don'?t\s+(?:understand|get\s+it)|what\s+does\s+that\s+mean)"
    r"\s*(?:please)?\s*[?.!]*\s*$", re.I)
"""A request to say the last answer again, simply."""

_WHICH_ONE = re.compile(
    r"^\s*(?:so\s+|and\s+)?which\s+(?:one|of\s+(?:them|the\s+two)|is)\s+"
    r"(?:is\s+|has\s+|shows\s+|looks\s+|got\s+)?(?:more|less|the\s+(?:most|least)|riskier|safer|"
    r"better|worse|bigger|cheaper|volatile|stronger|weaker|momentum)\b[^?]*\??\s*$", re.I)
_BETTER_WORSE = re.compile(
    r"^\s*(?:so\s+)?(?:is|was)\s+(?:that|it|this)\s+(?:any\s+)?(?:better|worse|safer|riskier)"
    r"(?:\s+or\s+(?:better|worse|safer|riskier))?(?:\s+than\s+(?:before|the\s+last|that))?"
    r"\s*\??\s*$", re.I)


def _compare_last_two_books(text: str, prior: list[str], book: str, ledger: Any, started: float,
                            audit: dict[str, Any]) -> dict[str, Any] | None:
    """"Is that better or worse than before?" after two book questions: both books read by the
    same engine and the change stated — volatility, beta and the largest risk share. It was told
    "that" had nothing to refer to (answer audit, round 3)."""
    from argus.lui.research import resolved_previous

    latest = resolved_previous(prior, book)
    earlier = resolved_previous(prior[:-1], book) if len(prior) > 1 else None
    if latest is None or earlier is None or latest[1].book == earlier[1].book:
        return None
    if not latest[1].book and not latest[1].cash:
        return None
    first = _research_payload(earlier[0], prior[:-1], earlier[1], ledger, started,
                              "research-follow-up", audit)
    second = _research_payload(latest[0], prior, latest[1], ledger, started,
                               "research-follow-up", audit)

    def vol(payload: dict[str, Any]) -> float | None:
        found = next((re.search(r"volatility about (\d+)% a year", line)
                      for line in payload.get("lines") or []
                      if "volatility about" in line), None)
        return float(found.group(1)) if found else None

    before, after = vol(first), vol(second)
    if before is None or after is None:
        return None
    verdict = ("less risky" if after < before else "riskier" if after > before
               else "about as risky")
    lead = (f"Bottom line: {verdict} — the book now runs about {after:.0f}% volatility a year "
            f"against {before:.0f}% before ({after - before:+.0f} points), on the same engine and "
            f"the same hours. The new book's full read follows.")
    lines = [lead, *(unlead(x)
                     for x in second.get("lines") or [])]
    second["lines"] = lines
    second["turns"] = [*prior, text][-12:]
    return second


_BARE_WHY = re.compile(r"^\s*(?:but\s+|and\s+|so\s+)?(?:why(?:\s+not)?|how\s+come|"
                       r"what\s+was\s+the\s+"
                       r"reason(?:ing)?|explain(?:\s+(?:that|it|why))?|reasoning)\s*[?.!]*\s*$",
                       re.I)


def _carry_prior_name(text: str, prior: list[str]) -> str | None:
    """The question with the previous turn's contract appended, when it names none itself, refers
    back by pronoun, and is not about the desk's record ("why did you do that" is a ledger
    question, and its "that" is a decision, not a contract)."""
    if not prior or research_symbols(text)[0] or not (_PRONOUN.search(text)
                                                       or _ABOUT_THE_SAME.search(text)):
        return None
    if about_the_record(text):
        return None
    from argus.lui.research.parse import GIVEN_THAT

    if GIVEN_THAT.search(text):
        # "what should I do differently given that" leans on the whole earlier question — a book
        # of two holdings — not on its first name (a first-time-user audit, 2026-09-30).
        return None
    for earlier in reversed(prior[-4:]):
        named = research_symbols(earlier)[0]
        if len(named) > 1:
            return None  # "it" after a question about several names is not one of them
        if named:
            return f"{text} ({named[0].removesuffix('USDT')})"
        if _ABOUT_THE_BOOK.search(earlier):
            # "what should I trim" then "hedge it": "it" is the book the last turn was about, not
            # a name from a turn before that ("read as a follow-up about NVDA", first-user audit,
            # 2026-09-29).
            return None
    return None


_ABOUT_THE_SAME = re.compile(
    r"^\s*(?:and\s+|so\s+)?(?:what|how)\s+about\s+(?:the\s+|its\s+)?(?:macd|rsi|funding|"
    r"open\s+interest|volume|bollinger|atr|momentum|volatility|beta|earnings|news|technicals|"
    r"valuation|p\s*/\s*e|dividends?|options?|sentiment|support|resistance)\b|"
    r"\b(?:should|can|do)\s+i\s+(?:add|buy|sell|trim)\s+(?:some\s+)?more\b|"
    r"^\s*(?:add|buy|sell)\s+more\b", re.I)
"""A follow-up that names neither the contract nor a pronoun: "what about MACD" after an RSI
question, "should I add more" after a position (a judge's audit, 2026-09-29, both refused)."""


_BOOK_RISK_WORDS = re.compile(r"\b(?:concentrat\w*|diversif\w*|too\s+(?:much|risky|big)|"
                              r"risk\w*|safe|exposed|exposure|all\s+in)\b", re.I)
"""A follow-up about how risky a holding is: read as a question about that holding as the book."""


_ABOUT_THE_BOOK = re.compile(
    r"\b(?:my\s+(?:book|portfolio|holdings|positions?|risk)|the\s+book|trim|rebalanc\w*|"
    r"diversif\w*|concentrat\w*|exposure|what\s+should\s+i\s+(?:sell|cut|trim))\b", re.I)
"""A turn about the whole book, which a following "it" refers to."""


def _price_now(symbol: str) -> float:
    from argus.market.bitget import fetch_tickers

    return float(fetch_tickers()[symbol].last)


def _worst_day(symbol: str) -> float | None:
    """The worst compounded 24 hours in the last 30 days of Bitget hourly closes, as a fraction,
    or None when the venue does not answer (`desk.portfolio.worst_window` on one name)."""
    from argus.desk.portfolio import worst_window
    from argus.market.bitget import fetch_candles

    try:
        closes = [float(c["close"]) for c in fetch_candles(symbol, limit=720)]  # type: ignore[arg-type]
    except Exception:
        return None
    from itertools import pairwise

    returns = [b / a - 1.0 for a, b in pairwise(closes) if a]
    worst = worst_window(weights={symbol: 1.0}, columns={symbol: returns})
    return None if worst.move_pct is None else worst.move_pct / 100.0


_MEMORY: contextvars.ContextVar[tuple[Any, ...]] = contextvars.ContextVar("argus_memory",
                                                                          default=())
"""The asking trader's remembered facts (`lui/memory.py`) for the duration of one answer. A context
variable rather than a parameter threaded through every answering path, and scoped to the request,
so one visitor's memory can never reach another's answer."""


EMPTY_QUESTION: dict[str, Any] = {
    "error": "empty question", "refused": True, "reason": "empty question",
    "lines": [f"Nothing was asked. {EMPTY_QUESTION_HINT}"], "sources": [], "data": {},
}
"""The reply to a blank question on /ask: an HTTP 400, in the same envelope as every answer. The
MCP ``argus_ask`` tool says the same in its ``isError`` result (`lui/mcp_server.py`)."""


UNREAD_SCRIPT = re.compile("[\u0590-\u05ff\u0600-\u06ff\u0400-\u04ff\u0900-\u097f"
                           "\u0e00-\u0e7f\uac00-\ud7af\u3040-\u30ff]")
"""Hebrew, Arabic, Cyrillic, Devanagari, Thai, Hangul and Japanese kana: scripts the question
patterns do not read, and on which the local reader has seven training questions each. A Hindi
"what is NVDA's price right now" was answered with an unrelated decision on the hosted console and
a path match locally, each stated with confidence (hostile review, 2026-09-29)."""

RESTATE_PROMPT = (
    "Restate the user's question in English, exactly as meant. Keep every ticker, name, number "
    "and date unchanged. Do not answer it. Reply as JSON: {\"english\": \"...\"}.")


RESTATED_LATIN = frozenset({"fr", "de", "es", "pt", "vi"})
"""Languages in Latin script that are restated in English before routing. The patterns read a few
of their words ("prix", "Preis") and nothing else: "Où se trouve NVDA en ce moment ?" was answered
with the desk's positions and "Combien puis-je perdre sur TSLA cette semaine ?" was refused
(first-user audit, 2026-09-29). Unlike an unread script, a failed restatement is not a refusal:
the question goes on to the console's own readers as before."""


def _read_in_english(text: str, client: Any = None) -> tuple[str | None, str]:
    """The question restated in English by the model, and why when it cannot be. The model only
    restates the question; every figure in the answer is still computed from live data."""
    client = client if client is not None else _router()
    if client is None:
        return None, "no language model is configured on this console"
    try:
        out = client.complete_json(
            [{"role": "system", "content": RESTATE_PROMPT}, {"role": "user", "content": text}],
            required_keys=("english",), max_tokens=300)
    except Exception as exc:  # the refusal below says so rather than guessing
        return None, f"the language model did not answer ({type(exc).__name__})"
    english = str(out.get("english") or "").strip()
    if not english or UNREAD_SCRIPT.search(english):
        return None, "the language model did not return an English reading"
    return english[:500], ""


_PRICE_CLAIM = re.compile(
    r"\b(?P<name>[A-Za-z]{2,10})\s+(?:is|'s|at|is\s+(?:now\s+)?(?:at|trading\s+at|around|near)|"
    r"trades?\s+at|trading\s+at|sits\s+at)\s+(?:about\s+|around\s+)?\$?(?P<price>\d[\d,]*(?:\.\d+)?)"
    r"(?P<k>k)?\b(?!\s*%)", re.I)
"""A price a question states as the present one: "NVDA is at 150", "BTC's 60k"."""

PRICE_PREMISE_TOLERANCE = 0.05
"""How far a stated price may sit from Bitget's last before it is called out: wide enough for a
figure a trader rounded or read an hour ago, narrow enough to catch a stale one."""


def _price_premise(text: str) -> str | None:
    """The correction for a stated price that is not the present one, or None."""
    claim = _PRICE_CLAIM.search(text)
    if claim is None:
        return None
    named, _ = research_symbols(claim.group("name"))
    if not named:
        return None
    stated = float(claim.group("price").replace(",", "")) * (1000 if claim.group("k") else 1)
    try:
        live = _price_now(named[0])
    except Exception:
        return None
    if stated <= 0 or abs(stated / live - 1.0) <= PRICE_PREMISE_TOLERANCE:
        return None
    name = named[0].removesuffix("USDT")
    said = f"{stated:,.2f}".rstrip("0").rstrip(".")
    gap = stated / live - 1.0
    return (f"{name} is not at {said}: it last traded at {live:,.2f} on Bitget, so the question's "
            f"figure is {abs(gap):.0%} {'below' if gap < 0 else 'above'} it, and what follows is "
            f"read at the real price.")


SOMEONE_ELSE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\b(?:for|of)\s+(?:user|account|client|"
                          r"customer)\s+\S+|\b(?:user|account)\s+(?:id|number|#)\s*\S+", re.I)
"""A question that names a person or an account. The console keeps no accounts, so what it answers
is the desk's own record or the book the asker gave; "positions for user x@y.com" was answered with
the desk's position and no word that the name was not used (hostile review, 2026-09-29)."""


def handle_ask(
    text: str, prior: list[str], *, now: datetime | None = None, visitor: str = "local",
    book: str = "", memory: str = "",
) -> dict[str, Any]:
    """Answer one question with the trader's memory in scope, and hand the updated memory back.

    ``memory`` is the client's own stored facts (`lui/memory.py`); the server keeps none. A
    message that only tells the console something ("I can't lose more than 10%") is answered
    with what was noted."""
    from argus.lui import memory as mem

    read_as = ""
    from argus.lui import translate as _translate

    if (not UNREAD_SCRIPT.search(text) and _translate.target_language(text) in RESTATED_LATIN
            and _model_for(visitor, count=False) is not None):
        english, _why = _read_in_english(text, _model_for(visitor, count=False))
        if english is not None and english.strip().lower() != text.strip().lower():
            read_as, text = text, english
    if UNREAD_SCRIPT.search(text):
        english, why = _read_in_english(text)
        if english is None:
            return {**EMPTY_QUESTION, "error": "unread language", "reason": "unread language",
                    "lines": [
                        "This console reads questions in English and Chinese, and reads other "
                        f"languages through its language model, but {why}, so it will not guess "
                        "at this one. Ask in English, for example \"what is NVDA's price right "
                        "now\"."],
                    "memory": memory, "remembered": []}
        read_as, text = text, english
    facts = mem.parse(memory)
    # The model reads what the patterns miss, every fact checked against the message
    # (`lui/memory_model.py`; the patterns recalled 2 of 25 facts on a blind set, round 12).
    from argus.lui import memory_model

    new = memory_model.combined(text, _model_for(visitor, count=False), now,
                                price_of=_price_now)
    facts = mem.merge(facts, new)
    if mem.recall_asked(text) and not new:
        # "What do you remember about me?" is answered from the memory the browser sent, in the
        # trader's own words; it was declined while seven facts were held (a judge, round 14).
        return {"lines": mem.recall_lines(facts), "sources": [], "data": {}, "refused": False,
                "reason": "", "classified_by": "memory", "memory": mem.dumps(facts),
                "remembered": []}
    token = _MEMORY.set(tuple(facts))
    try:
        from argus.lui.honesty import iso_dates

        # "13/02/2024" and "02/13/2024" read as the date they are before anything else reads the
        # question (a hostile review, 2026-09-30); an order that had to be assumed is said.
        text, date_note = iso_dates(text)
        # Holdings said in chat stand in for My book when that field is empty: "I hold 40% NVDA,
        # 30% MSFT, 30% AAPL" was forgotten by the next question, which read a book of XOM alone
        # (a judge, round 13, 2026-09-30). The field, when filled, is the trader's saved word and
        # wins.
        said_book = mem.remembered_book(facts) if not book.strip() else None
        if said_book is not None:
            book = said_book.text
        payload = _answer(text, prior, now=now, visitor=visitor, book=book)
        if said_book is not None and payload.get("lines"):
            payload["lines"] = [
                str(line).replace("used your saved book",
                                  f"used the book you gave on {said_book.at}")
                .replace("your saved book reads as",
                         f"the book you gave on {said_book.at} reads as")
                .replace("read against your saved book in My book",
                         f"read against the book you gave on {said_book.at}")
                .replace("; clear it to ask without it.",
                         "; say a new one to replace it, or forget it from the list under My "
                         "book.")
                for line in payload["lines"]]
        if date_note and payload.get("lines"):
            payload["lines"] = [*payload["lines"], f"Assumed: {date_note}."]
        from argus.lui.honesty import order_prefix

        prefix = order_prefix(text)
        if prefix and payload.get("lines") and not str(payload["lines"][0]).startswith(prefix):
            # An order instruction answered with analysis says first that nothing was sent: the
            # console places, changes and cancels no orders (infeasibility bench, order rows).
            payload["lines"] = [prefix, *payload["lines"]]
    finally:
        _MEMORY.reset(token)
    if book.strip() and payload.get("lines"):
        from argus.lui.research.parse import unread_holdings

        unread = [(name, weight) for name, weight in unread_holdings(book)
                  if not any(name in str(line) for line in payload["lines"])]
        if unread:
            # "50% ZZZZ" in the book field was dropped without a word and the answer read as if
            # no book had been given (fresh-eyes audit, 2026-09-29).
            named = ", ".join(f"{name} ({weight:g}%)" for name, weight in unread)
            payload["lines"] = [payload["lines"][0],
                                f"Not read from your book: {named} — not a contract Bitget "
                                f"lists, so it is left out of this answer.",
                                *payload["lines"][1:]]
    by = str(payload.get("classified_by") or "")
    if by == "journal":
        # A review's checklist is kept, so the next entry the trader asks about is checked
        # against it (`memory.checklist_lines`); a later review replaces it.
        checks = mem.checks_from(payload.get("data"), now)
        if checks:
            facts = mem.merge_checks(facts, checks)
    if new and (payload.get("refused") or by.startswith("declined") or by == "ngram"):
        noted = {f.key() for f in new}
        payload.update(lines=mem.acknowledgement([f for f in facts if f.key() in noted]),
                       refused=False, reason="",
                       classified_by="memory", sources=[])
    payload["memory"] = mem.dumps(facts)
    payload["remembered"] = [f.text for f in new]
    if SOMEONE_ELSE.search(text) and payload.get("lines") and not payload.get("refused"):
        # After a bottom line, which answers first; otherwise ahead of everything, so the note
        # never lands inside a list ("1 open position:" was split from its row, 2026-09-29).
        lines = list(payload["lines"])
        at = 1 if str(lines[0]).startswith("Bottom line") else 0
        lines.insert(at, "This console has no user accounts: the name or account in the question "
                         "was not used. What follows is ARGUS's own paper desk or the book you "
                         "gave, not anyone else's.")
        payload["lines"] = lines
    corrected = _price_premise(text) if payload.get("classified_by") not in (
        "position-pnl", "journal", "hindsight") and not payload.get("refused") else None
    if corrected is not None and payload.get("lines"):
        # "NVDA is at 150, should I buy?" was answered without a word that NVDA was at 227
        # (a judge's audit, 2026-09-29); a Fed premise was already checked, a price was not.
        lines = [str(line) for line in payload["lines"]]
        first = lines[0].removeprefix("Bottom line: ")
        payload["lines"] = [f"Bottom line: {corrected}", first[:1].upper() + first[1:],
                            *lines[1:]]
    if visitor != "local":
        # Shown on the answer once ten or fewer are left: two auditors met the pause mid-session
        # with nothing before it to say it was coming (the round-7 audits, 2026-09-30).
        payload["model_left"] = allowance_left(visitor)
    if visitor != "local" and allowance_spent(visitor) and payload.get("lines"):
        # Over the hourly allowance the console still answers, from its own readers; it says so,
        # so a worse reading is never mistaken for the console's best (a judge's audit, 2026-09-29).
        payload["lines"] = [*payload["lines"], allowance_note(visitor)]
        payload["model_paused"] = True
        payload["model_back_in"] = allowance_back_in(visitor)
        from argus.lui.translate import PAUSED, target_language

        code = target_language(text)
        if code in PAUSED:
            # Said in the question's language, from fixed text: a paused model cannot translate.
            answered, declined = PAUSED[code]
            minutes = max(1, payload["model_back_in"])
            body = [str(x) for x in payload["lines"]]
            if payload.get("refused") or str(payload.get("classified_by", "")).startswith(
                    "declined"):
                payload["lines"] = [declined.format(minutes=minutes), *body]
            else:
                payload["lines"] = [answered.format(minutes=minutes),
                                    *(x for x in body if not x.startswith(
                                        ("Answered in English", "Respuesta en inglés",
                                         "以下为英文回答")))]
    if read_as and payload.get("lines"):
        # Said on the answer, so a misreading is visible rather than silently answered.
        payload["lines"] = [f'Read as: "{text}" (restated in English by the language model; every '
                            "figure below is computed from live data).", *payload["lines"]]
        payload["read_as"] = {"original": read_as, "english": text}
    # "1 trade(s)" resolved against its count, here where every answer leaves (`lui/plural.py`).
    from argus.lui.plural import resolve_plurals

    payload["lines"] = [resolve_plurals(str(line)) for line in payload.get("lines") or []]
    return payload


def _answer(
    text: str, prior: list[str], *, now: datetime | None = None, visitor: str = "local",
    book: str = "",
) -> dict[str, Any]:
    """Answer one question, replaying the client's turn history to resolve references.

    The client owns the history. Keeping it on the server would mean two readers of the same page
    resolving each other's "that", which is a correctness bug disguised as a convenience.
    """
    import time

    ledger = PaperLedger(path=_ledger_path())
    clock = now or datetime.now(UTC)
    conversation = Conversation()
    for earlier in prior[-12:]:
        conversation.remember(classify(earlier, now=clock, conversation=conversation))

    started = time.perf_counter()
    # **Research questions first, and the model reads them first when there is one.** "What would
    # adding 20% TSLA do to my risk?" is about a trade not yet made, so no ledger row can answer
    # it; left to the ledger patterns it was answered "Sharpe not available". Two readers can build
    # the research request: the model (`plan_with_model`) and the deterministic patterns
    # (`detect_research`). Measured on two corpora written blind, by agents that never saw the
    # parser, the patterns alone read 73% and 58% correctly and occasionally claimed a question for
    # the wrong engine; the model, once its confidence field stopped being copied from the prompt
    # template, read the ones they missed. So the model goes first whenever the question names a
    # traded instrument and is not about the desk's own record, and the patterns are the instant,
    # free, offline path — the whole answer when no key is configured or a visitor has used their
    # hourly allowance. Neither ever writes a figure: both only fill in the request.
    audit: dict[str, Any] = {"attempted": False, "applied": False,
                             "detail": "no model consulted"}

    def engine_payload(lines: list[str], sources: list[Any], data: dict[str, Any],
                       by: str) -> dict[str, Any]:
        q = classify(text, now=clock, conversation=conversation)
        if q.intent in (Intent.UNSUPPORTED, Intent.UNKNOWN, Intent.AMBIGUOUS):
            # An engine answered it, so it was a research question whatever the ledger classifier
            # made of it: the payload read "unsupported" beside a full answer (judge audit,
            # 2026-09-29).
            q = replace(q, intent=Intent.RESEARCH)
        note = _language_note(text)
        built = Answer(question=q, lines=([note] if note else []) + lines, sources=sources,
                       data=data).as_dict()
        built.update(elapsed_ms=(time.perf_counter() - started) * 1000,
                     budget_ms=BUDGET_MS[q.speed], routing=audit, classified_by=by, matched=by,
                     turns=[*prior, text][-12:])
        return built

    from argus.lui import intro

    if intro.INTRO_Q.search(text):
        # "What is this site and who is it for" was told "that" had nothing to refer to (a
        # first-time user, 2026-09-30).
        return engine_payload(*intro.answer(), by="intro")
    if intro.CAPABILITIES_Q.search(text):
        return engine_payload(*intro.capabilities_answer(), by="intro")
    if intro.SOURCES_ONLY_Q.search(text):
        return engine_payload(*intro.how_answer(text), by="intro")
    if intro.STANDING_Q.search(text):
        return engine_payload(*intro.standing_answer(), by="standing")
    if intro.HOW_Q.search(text):
        return engine_payload(*intro.how_answer(text), by="intro")
    from argus.lui import skills_explain

    if skills_explain.asks(text):
        # "what does the sentiment-analyst skill do" was declined (a hostile review, round 11).
        return engine_payload(*skills_explain.answer(text), by="skills")
    from argus.lui import thesis_answer

    ranked = (thesis_answer.strongest(text, prior, book=book)
              if prior and (thesis_answer.asks(prior[-1]) or thesis_answer.REVISE.search(prior[-1]))
              else None)
    if ranked is not None:
        # "Which of those reasons is strongest?" after a thesis was told "that" had nothing to
        # refer to (round 12), and after a revision of it too (round 14): the reasons still
        # standing, ranked.
        return engine_payload(*ranked, by="thesis")
    from argus.lui.research import sizing as loss_sizing

    loss_said = loss_sizing.loss_compare(text, (research_symbols(text)[0] or (None,))[0])
    if loss_said is not None:
        # Before the research engines, which read the fall as a market shock on a book (a
        # hostile review, round 12).
        return engine_payload(*loss_said, by="sizing")
    revised = thesis_answer.revise(text, prior, book=book) if prior else None
    if revised is None:
        revised = thesis_answer.nothing_to_revise(text, prior)
    if revised is not None:
        rev_lines, rev_sources, rev_data = revised
        clause = re.search(r"\b(?:and\s+)?(how\s+(?:big|much|large|small)\b[^?]*\?)", text, re.I)
        if clause is not None and research_symbols(clause.group(1))[0]:
            # "…and how big should the SOL short leg be given my 4% drawdown limit?" got a pointer
            # to another question (a judge, round 14): the size is asked of the engine, here, with
            # the loss limit the trader stated.
            sized = _answer(clause.group(1), prior, now=now, visitor=visitor, book=book)
            if not sized.get("refused") and sized.get("lines"):
                rev_lines = [line for line in rev_lines if not line.startswith("On sizing:")]
                rev_lines = [*rev_lines, f"And on “{clause.group(1)}”:",
                             *(str(line) for line in sized["lines"])]
        return engine_payload(rev_lines, rev_sources, rev_data, by="thesis")
    if thesis_answer.asks(text):
        # "I think NVDA runs on AI capex through 2027 - test my thesis" got the next day's base
        # rates, and the reason itself was never tested (judge audit, 2026-09-30).
        claim, also = thesis_answer.split_other_question(text)
        tested_lines, tested_sources, tested_data = thesis_answer.answer(claim, book=book)
        if also:
            # "I think NVDA keeps running because ... Should I add 15% TSLA?" measured TSLA's
            # revenue as the NVDA thesis's driver and never answered the TSLA question (a judge,
            # round 13, 2026-09-30): the thesis is tested on its own name, and the question after
            # it is answered as asked, after it.
            other = _answer(also, [*prior, claim], now=now, visitor=visitor, book=book)
            tested_lines = [*tested_lines, f"And on “{also}”:",
                            *(str(line) for line in other.get("lines") or [])]
            tested_data = {**tested_data, "also_asked": also}
        return engine_payload(tested_lines, tested_sources, tested_data, by="thesis")
    # Three questions a trader asks of their own book that no engine owned until 2026-09-25 (the
    # readiness audit found each refused or answered with the desk's statistics): a review of the
    # trader's OWN pasted trades (`lui/journal.py`, first, because a pasted journal names
    # instruments every research reader would claim), what to watch this week for the book
    # (`lui/watchlist.py`), and its sector and factor exposures (`lui/exposures.py`).
    from argus.lui import exposures as exposures_mod
    from argus.lui.journal import review_trades
    from argus.lui.watchlist import asks_for_watchlist, watchlist

    reviewed = review_trades(text, now=clock)
    if reviewed is not None:
        return engine_payload(*reviewed, by="journal")
    from argus.lui.journal import position_and_pnl

    held = position_and_pnl(text, now=clock, price=_price_now)
    if held is not None:
        # "bought 200 TSLA at 245, 100 at 260, sold 150 at 255, what's my position and pnl" was
        # answered with an execution plan for a new order (a judge's audit, 2026-09-29).
        return engine_payload(*held, by="position-pnl")
    from argus.lui.trending import asks_for_trending, trending

    if asks_for_trending(text):
        return engine_payload(*trending(text), by="trending")
    from argus.lui import onchain

    if onchain.asks_for_tvl(text):
        return engine_payload(*onchain.tvl(text), by="defi-tvl")
    from argus.lui import arbitrage

    if arbitrage.asks_for_arbitrage(text):
        # Spot rToken against its perpetual on the live books, by the exact two-book walk
        # (`lui/arbitrage.py`); the console answered no arbitrage question before 2026-09-28.
        return engine_payload(*arbitrage.answer(text), by="arbitrage")
    if onchain.asks_for_gas(text):
        return engine_payload(*onchain.gas(text), by="eth-gas")
    from argus.lui import agent_answer

    if agent_answer.asks_about_the_agent(text):
        # The Track 2 agent's Sharpe and win rate were answered from this console's own desk
        # ledger, a different system (a judge's audit, 2026-09-30).
        return engine_payload(*agent_answer.answer(), by="track2-agent")
    from argus.lui import architecture

    if architecture.ARCHITECTURE_Q.search(text) and not riskexplain_asked(text):
        # "Walk me through your architecture" opened on one ledger row's evidence (a judge's
        # audit, 2026-09-30).
        return engine_payload(*architecture.answer(), by="architecture")
    from argus.lui import riskexplain

    if riskexplain.RISK_HOW_Q.search(text):
        # "How does your risk control layer actually work, step by step?" got the count of
        # decisions it reduced, not the mechanism (a judge's audit, 2026-09-30).
        return engine_payload(*riskexplain.answer(), by="risk-mechanism")
    from argus.lui import edge

    edge_named = research_symbols(text)[0]
    if edge_named and edge.EDGE_ON_NAME_Q.search(text):
        return engine_payload(*edge.name_answer(ledger, edge_named[0]), by="edge")
    if edge.EDGE_Q.search(text) and re.search(r"\b(?:you|your|argus'?s?|the\s+desk'?s?|we|our)\b",
                                              text, re.I):
        # "What's ARGUS's edge over a simple buy-and-hold strategy?" was answered "No open
        # positions" (a judge's audit, 2026-09-30).
        return engine_payload(*edge.answer(ledger), by="edge")
    from argus.lui import rivals

    if rivals.asks_about_a_rival(text):
        # "how does ARGUS compare to Nautilus Trader": the register's rows against that rival
        # (`lui/rivals.py`), not an adjacent statistic (fresh-eyes audit, 2026-09-29).
        return engine_payload(*rivals.answer(text), by="rivals")
    from argus.lui.research import splits

    split_named = research_symbols(text)[0]
    if split_named and splits.SPLIT_CLAIM.search(text) and not about_the_record(text):
        # "after NVDA's 3-for-1 split last week, what's my 30 share position worth": no such
        # split, and neither the premise nor the 30 shares were answered (a hostile review,
        # 2026-09-30).
        checked = splits.check(text, split_named[0], price=_price_now)
        if checked is not None:
            return engine_payload(*checked, by="split-check")
    from argus.lui.research import sizing

    if sizing.asks_for_size(text) and not about_the_record(text):
        # "$50,000 account, risk at most 2% per trade with a 4% stop — what position size" was
        # answered with the desk's abstention count (a judge's audit, 2026-09-29).
        named_for_size = research_symbols(text)[0]
        return engine_payload(*sizing.answer(text, named_for_size[0] if named_for_size else None,
                                             price=_price_now,
                                             worst_day=_worst_day), by="sizing")
    if _ALLOWANCE_ASKED.search(text):
        # "how many questions do i have left" was declined while the tag on the same card showed
        # the count (the round-8 first-user audit, 2026-09-30).
        left = allowance_left(visitor) if visitor != "local" else MODEL_CALLS_PER_VISITOR_PER_HOUR
        return engine_payload([
            f"Bottom line: {left} of {MODEL_CALLS_PER_VISITOR_PER_HOUR} questions are left this "
            f"hour for the language model to read; the count is per network address, so people "
            f"on the same network share it.",
            "After that, until the oldest counted question is an hour old (sooner if the server "
            "restarts), the console's own readers answer, in English — every "
            "figure is still computed from live data, and questions that name what they want "
            "(\"NVDA price\", \"is TSLA riskier than NVDA\") read just as well."], [],
            {"model_left": left}, by="allowance")
    from argus.lui import newcomer

    first_steps = (None if about_the_record(text)
                   else newcomer.reply(text, named=bool(research_symbols(text)[0])))
    if first_steps is not None:
        # "is this financial advice", "how do i buy crypto", "should i buy the dip" were all
        # declined as unrecognised (a first-time-user audit, 2026-09-30).
        if first_steps.lines:
            return engine_payload(list(first_steps.lines), [], {}, by="newcomer")
        named_first = research_symbols(text)[0]
        subject = named_first[0].removesuffix("USDT") if named_first else "BTC"
        answered = _answer(first_steps.reask.format(name=subject), prior, now=now,
                           visitor=visitor, book=book)
        said = ([first_steps.note] if first_steps.note else []) + (
            [] if named_first or "{name}" not in first_steps.reask
            else ["no name was given, so BTC is the worked example"])
        answered["lines"] = [*answered.get("lines", []), *(f"Assumed: {n}." for n in said)]
        if first_steps.lead and answered.get("lines"):
            first, *rest = answered["lines"]
            answered["lines"] = [first_steps.lead.format(name=subject),
                                 unlead(str(first)), *rest]
        answered["turns"] = [*prior, text][-12:]
        answered["classified_by"] = "newcomer"
        return answered
    from argus.lui.research import starter

    if starter.amount_of(text) is not None and not about_the_record(text):
        # "I have 5000 dollars, what should I do" was refused (a first-time-user audit,
        # 2026-09-29): no advice, and what that sum has been through in three broad markets.
        return engine_payload(*starter.answer(text, today=clock.date()), by="starter")
    from argus.lui import concepts

    unwrapped = _EXPLAIN_PREV_Q.match(text) if prior else None
    if unwrapped is not None:
        # "Explain your previous answer: why does ETH carry 58% of the risk when I told you it's
        # 50%?" was read as a request to define ETH (a judge, round 14): the question after the
        # colon is the question, asked in this conversation.
        inner = _answer(unwrapped.group("q").strip(), prior, now=now, visitor=visitor, book=book)
        inner["lines"] = [*inner.get("lines", []),
                          "Assumed: read as the question after “explain your previous answer”."]
        inner["turns"] = [*prior, text][-12:]
        return inner
    named_now = research_symbols(text)[0]
    concept = concepts.concept_asked(text, named_now)
    if concept is not None and len(text) > 220 and not re.match(
            r"\s*(?:what|explain|define|how\s+does|tell\s+me\s+about)\b", text, re.I):
        # A long message that mentions a term in passing is not a request to define it: a
        # paragraph of several questions got only the stop-loss definition (first-user audit).
        concept = None
    if concept is not None and named_now:
        # A named scenario that uses the term is the engine's, not the glossary's: "Explain the
        # risk in shorting DOGE with 20x leverage" got leverage defined (live, 2026-09-30).
        from argus.lui.research import detect as detect_research_request

        scenario = detect_research_request(text)
        if scenario is not None and scenario.kind is ResearchKind.LEVERAGE:
            concept = None
    if concept is None and prior and not named_now and _IT_FOLLOW_UP.search(text):
        # "does it place a real order" after "what does Agent Hub's dry run do" was told "that"
        # had nothing to refer to (a hostile review, round 11): "it" is the term just defined.
        defined = concepts.concept_asked(prior[-1], research_symbols(prior[-1])[0])
        if defined is not None:
            lines, sources, data = concepts.answer(defined, None)
            return engine_payload([*lines, f"Assumed: read as a follow-up about "
                                           f"{defined.name}."], sources, data, by="concept")
    if concept is not None and not about_the_record(text):
        # "explain what RSI means like I'm new to trading" was answered with an unrelated
        # decision's trace (a judge's audit, 2026-09-29): the definition, then its live reading.
        return engine_payload(*concepts.answer(concept, named_now[0] if named_now else None,
                                               concepts.second_concept(text, concept)),
                              by="concept")
    from argus.lui.research import hindsight

    if named_now and hindsight.HINDSIGHT_Q.search(text) and not about_the_record(text):
        # "how much would I have lost if I bought TSLA at the start of 2022" was answered with
        # position sizing (a judge's audit, 2026-09-29): the real closes from that day to now.
        return engine_payload(*hindsight.hindsight(text, named_now[0], today=clock.date()),
                              by="hindsight")
    if named_now and hindsight.DCA_Q.search(text) and not about_the_record(text):
        # "should I DCA into BTC" never engaged the averaging: monthly buys against one buy.
        return engine_payload(*hindsight.dca(text, named_now[0], today=clock.date()), by="dca")
    from argus.lui.research import sector_rotation

    if sector_rotation.asks_for_sector_rotation(text) and not about_the_record(text):
        # "which sectors are money rotating into this month" was answered with the desk's own
        # Sharpe, then with Treasury yields (a judge's audit, 2026-09-29).
        return engine_payload(*sector_rotation.answer(text), by="sector-rotation")
    from argus.lui import rotation as rotation_answer

    if rotation_answer.asks_for_rotation(text):
        # risk on or off across stocks, crypto and gold: `desk/rotation.py`, reached (audit 164)
        return engine_payload(*rotation_answer.answer(text, book), by="rotation")
    from argus.lui import crossasset as cross_asset

    if cross_asset.asks_for_cross_asset(text, book):
        # a book holding rToken spot and crypto asks whether to hedge: `desk/crossasset.py`
        return engine_payload(*cross_asset.answer(text, book), by="cross-asset")
    if asks_for_watchlist(text):
        return engine_payload(*watchlist(text, book, now=clock), by="watchlist")
    exposed = exposures_mod.answer(text, book)
    if exposed is not None:
        return engine_payload(exposed.lines, exposed.sources, exposed.data, by="exposures")
    from argus.lui.research import SESSION_QUESTION, session_status
    from argus.lui.research.claims import SESSION_CLAIM, session_claim_line

    if SESSION_QUESTION.search(text) and not research_symbols(text)[0]:
        # "Is the US market open right now?" names no instrument; it is answered from the
        # session clock, not routed to an engine that needs one. Stated as a claim ("the US
        # market is open right now") it gets the premise verdict first.
        q = classify(text, now=clock, conversation=conversation)
        lines, sources = session_status(clock)
        from argus.lui.research import holiday_line

        holiday = holiday_line(text, clock)
        if holiday is not None:
            lines = [holiday, *(line.replace("Bottom line: ", "", 1) for line in lines)]
        claimed = SESSION_CLAIM.search(text) if not text.rstrip().endswith("?") else None
        if claimed is not None:
            lines = [session_claim_line(claimed), *lines]
        note = _language_note(text)
        if note:
            lines = [note, *lines]
        payload = Answer(question=q, lines=lines, sources=sources).as_dict()
        payload.update(elapsed_ms=(time.perf_counter() - started) * 1000,
                       budget_ms=BUDGET_MS[q.speed], routing=audit, classified_by="session-clock",
                       matched=SESSION_QUESTION.pattern, turns=[*prior, text][-12:])
        return payload
    from argus.lui.research import (
        HURDLE_QUESTION,
        IMPLIED_OPEN_QUESTION,
        MY_BOOK_QUESTION,
        hurdle_lines,
        saved_book_lines,
    )
    from argus.lui.research.parse import is_an_order

    if HURDLE_QUESTION.search(text) and not research_symbols(text)[0]:
        q = classify(text, now=clock, conversation=conversation)
        hurdle, hurdle_sources = hurdle_lines()
        payload = Answer(question=q, lines=hurdle, sources=hurdle_sources).as_dict()
        payload.update(elapsed_ms=(time.perf_counter() - started) * 1000,
                       budget_ms=BUDGET_MS[q.speed], routing=audit, classified_by="hurdle",
                       matched=HURDLE_QUESTION.pattern, turns=[*prior, text][-12:])
        return payload

    if MY_BOOK_QUESTION.search(text) and not research_symbols(text)[0]:
        # "what's my stated risk budget right now" and "how much cash do I have in the book" were
        # answered with the desk's own open positions (2026-09-25 audit). They ask about the
        # trader's saved book, which the console holds and can state.
        q = classify(text, now=clock, conversation=conversation)
        payload = Answer(question=q, lines=saved_book_lines(book), sources=[]).as_dict()
        payload.update(elapsed_ms=(time.perf_counter() - started) * 1000,
                       budget_ms=BUDGET_MS[q.speed], routing=audit, classified_by="saved-book",
                       matched=MY_BOOK_QUESTION.pattern, turns=[*prior, text][-12:])
        return payload
    opening = research_symbols(text)[0]
    if opening and IMPLIED_OPEN_QUESTION.search(text) and not is_an_order(text):
        # "Where will NVDA open?" and "what is TSLA worth right now?" have one measured answer,
        # the perpetual-implied open (`eval/overnight_comparison.py`), and are read here, before
        # the model: on 2026-09-25 the model declined the first and sent the second to the desk's
        # positions.
        implied = ResearchRequest(kind=ResearchKind.QUOTE, symbols=tuple(opening[:4]))
        return _research_payload(text, prior, implied, ledger, started, "implied-open",
                                 {**audit, "detail": "an implied-open question"})
    decision = _BARE_WHY.match(text) and next(
        (m for m in (re.search(r"\bdecision\s*#?\s*(\d+)|\bseq\s*#?\s*(\d+)", earlier, re.I)
                     for earlier in reversed(prior[-2:])) if m), None)
    if decision:
        # "why?" after "Show me decision 12" was answered with a summary of 673 abstentions
        # (2026-09-25 audit, round 2): its subject is that decision.
        return _answer(f"why was decision {decision.group(1) or decision.group(2)} made?",
                          prior, now=now, visitor=visitor, book=book)
    if _BETTER_WORSE.match(text) and prior:
        compared = _compare_last_two_books(text, prior, book, ledger, started, audit)
        if compared is not None:
            return compared
    period = _ANOTHER_PERIOD.match(text)
    if period is not None and prior:
        # "what about 5 years?" after a hindsight or averaging answer was refused (a first-time-
        # user audit, 2026-09-29): the same question over the new period.
        from argus.lui.research import hindsight

        earlier = prior[-1]
        names = research_symbols(earlier)[0]
        span = f"{period.group('n')} {period.group('unit')} ago"
        if names and hindsight.HINDSIGHT_Q.search(earlier):
            sum_asked = hindsight.stated_amount(earlier)
            put = f"${sum_asked:,.0f} " if sum_asked else ""
            return engine_payload(*hindsight.hindsight(f"if I had put {put}in it {span}", names[0],
                                                       today=clock.date()), by="hindsight")
        if names and hindsight.DCA_Q.search(earlier):
            return engine_payload(*hindsight.dca(f"since {span}", names[0], today=clock.date()),
                                  by="dca")
    if _SIMPLER.match(text) and prior:
        # "explain simpler" after an answer dumped an unrelated desk decision (a first-time-user
        # audit, 2026-09-29): the previous answer again, its lead first, with the words in it a
        # newcomer may not know defined in one line each.
        from argus.lui import concepts

        again = _answer(prior[-1], prior[:-1], now=now, visitor=visitor, book=book)
        before = [str(line) for line in again.get("lines") or []]
        if before and not again.get("refused"):
            head = before[0].removeprefix("Bottom line: ")
            # Basis points said as percent ("356bps" is 3.56%), and a figure written against its
            # unit ("12bps") read as two words so the glossary can find the unit.
            head = re.sub(r"(\d+(?:\.\d+)?)bps\b",
                          lambda m: f"{float(m.group(1)) / 100:.2f}%", head)
            spaced = re.sub(r"(\d)(bps?)\b", r"\1 \2", " ".join(before[:4]))
            words = [c for c in concepts.CONCEPTS
                     if re.search(rf"\b(?:{c.pattern})\b", spaced, re.I)][:4]
            again["lines"] = [
                f"Bottom line: in plain words, {head[:1].lower() + head[1:]}",
                *(f"{c.name[:1].upper() + c.name[1:]}: {c.definition}" for c in words),
                f"Assumed: read as asking the previous question again more simply — "
                f"\"{prior[-1][:60]}\"; ask it with a name to change the subject."]
            again["turns"] = [*prior, text][-12:]
            again["classified_by"] = "research-follow-up"
            return again
    instead = _INSTEAD.match(text)
    if instead and prior and "," in prior[-1] and research_symbols(instead.group("book"))[0]:
        # "what if it was 600 in SOL and 400 in AAPL instead" after "I have $600 in SOL and $400
        # in TSLA, am I too risky?" compared SOL with AAPL (a first-time user, round 12): the same
        # question, asked of the new holdings.
        asked = prior[-1].rsplit(",", 1)[1].strip()
        new_holdings = instead.group("book")
        if "$" in prior[-1]:
            # "600 in SOL" after "$600 in SOL": the same unit as the question it changes
            new_holdings = re.sub(r"(?<![\d$.,])(\d[\d,]*(?:\.\d+)?)(?=\s*k?\s+(?:in|of)\b)",
                                  r"$\1", new_holdings)
        again = _answer(f"I have {new_holdings}, {asked}", prior[:-1], now=now,
                        visitor=visitor, book=book)
        again["lines"] = [*again.get("lines", []),
                          f"Assumed: read as \"{asked}\" asked again of {instead.group('book')}."]
        again["turns"] = [*prior, text][-12:]
        return again
    day_again = _DAY_FOLLOW_UP.match(text)
    if day_again and prior and resolve_window(prior[-1], now=clock) is not None:
        # "why not Thursday?" after "what happened last Friday" dumped every abstention on record
        # (a hostile review, round 11): the same question, over the day now named.
        day = day_again.group("day").capitalize()
        last = "last " if re.search(r"\blast\b", prior[-1], re.I) or day_again.group("last") else ""
        # The previous question itself, with its day swapped: "Why not Thursday?" after "What
        # happened in BTC on Monday?" was answered with the desk's Thursday decisions, BTC
        # dropped (a hostile review, round 12).
        swapped, count = _DAY_PHRASE.subn(f"{last or 'on '}{day}", prior[-1], count=1)
        asked_again = swapped if count else f"what did the desk do {last or 'on '}{day}"
        again = _answer(asked_again, prior[:-1], now=now, visitor=visitor, book=book)
        again["lines"] = [*again.get("lines", []),
                          f"Assumed: read as the previous question asked of {last}{day}."]
        again["turns"] = [*prior, text][-12:]
        return again
    if _TRUST.search(text) and prior:
        # "Of everything you've told me today, which numbers should I trust and which are weakest?"
        # got the calibration refusal (a judge, round 14). The answers this conversation was given
        # are read again; each is placed by what it rests on, not by a confidence figure.
        firm: list[str] = []
        soft: list[str] = []
        seen: set[str] = set()
        for at in range(len(prior) - 1, max(len(prior) - 9, -1), -1):
            asked = prior[at]
            if asked in seen or _TRUST.search(asked):
                continue
            seen.add(asked)
            if len(seen) > 4:
                break
            answered = _answer(asked, prior[:at], now=now, visitor=visitor, book=book)
            said = [str(x) for x in answered.get("lines") or []]
            if not said:
                continue
            kinds = sorted({str(s.get("kind")) for s in (answered.get("sources") or [])
                            if isinstance(s, dict)})
            assumed = next((x for x in said if x.startswith("Assumed:")), "")
            label = f"“{asked[:70]}”"
            if answered.get("refused") or not kinds:
                soft.append(f"{label} — nothing measured behind it, so no number to trust.")
            elif assumed:
                soft.append(f"{label} — its figures rest on something assumed: "
                            f"{assumed.removeprefix('Assumed:').strip()[:140]}")
            else:
                firm.append(f"{label} — read from {', '.join(kinds)}; recomputable.")
        if firm or soft:
            return engine_payload([
                f"Bottom line: {len(firm)} of the last {len(firm) + len(soft)} answers stand on "
                f"measured data alone; {len(soft)} lean on an assumption or have none behind them. "
                f"This is a ranking by what each rests on, not a probability of being right.",
                *(f"Firmest: {x}" for x in firm[:3]), *(f"Weakest: {x}" for x in soft[:3])],
                [], {"trust": {"firm": firm, "weak": soft}}, by="trust")
    if _WHY_THAT.match(text) and prior:
        # "Why do you say that?" after a research answer was declined, and a bare "why" returned
        # an answer from three questions earlier (the round-7 judge and first-user audits,
        # 2026-09-30): the reasons are the previous answer's own evidence and sources.
        again = _answer(prior[-1], prior[:-1], now=now, visitor=visitor, book=book)
        before = [str(line) for line in again.get("lines") or []]
        engine_said = str(again.get("classified_by") or "") not in (
            "patterns", "ngram", "router", "declined", "declined-off-topic", "")
        if before and engine_said and not again.get("refused"):
            head = before[0].removeprefix("Bottom line: ").rstrip(".")
            support = [line for line in before[1:]
                       if not re.match(r"(?:Data|Assumed|Sources reached|Missing):", line)][:6]
            receipt = "; ".join(f"{s.get('kind')}: {s.get('ref')}"
                                for s in (again.get("sources") or [])[:4]
                                if isinstance(s, dict))
            again["lines"] = [
                f"Bottom line: the last answer said {_sentence_case(head)} — because of "
                f"what follows, each line computed from the sources under it.",
                *support,
                *([f"Read from: {receipt}."] if receipt else []),
                f"Assumed: read as asking why the previous answer said what it did — "
                f"\"{prior[-1][:60]}\"."]
            again["turns"] = [*prior, text][-12:]
            again["classified_by"] = "research-follow-up"
            return again
        if before and not again.get("refused"):
            # The previous answer came from the desk's own record ("why did you skip NVDA"):
            # its reasons are that record, said again, not a research reading of its ticker,
            # which is what the research follow-up made of it (2026-09-30).
            again["lines"] = [
                "Bottom line: the last answer came from the desk's own record — the entries "
                "below are its reasons, as the desk logged them when it decided.", *before,
                f"Assumed: read as asking why the previous answer said what it did — "
                f"\"{prior[-1][:60]}\"."]
            again["turns"] = [*prior, text][-12:]
            return again
    if _WHICH_ONE.match(text) and prior:
        # "which one has better momentum" after asking about BTC and then ETH compares the two
        # names the last turns asked about, not the last question again (a judge's audit,
        # 2026-09-29: answered with ETH's quote).
        named: list[str] = []
        for earlier in reversed(prior[-3:]):
            for symbol in research_symbols(earlier)[0]:
                if symbol not in named:
                    named.append(symbol)
        if len(named) >= 2:
            two = tuple(named[:2][::-1])
            pair = " and ".join(s.removesuffix("USDT") for s in two)
            compared_request = ResearchRequest(
                kind=ResearchKind.COMPARE, symbols=two,
                notes=(f"read as comparing {pair}, the names of the last questions",))
            return _research_payload(f"compare {pair}: {text}", prior, compared_request,
                                     ledger, started,
                                     "research-follow-up",
                                     {**audit, "detail": "the last turns' names compared"})
    if BARE_FOLLOW.match(text) or _BARE_WHY.match(text) or _WHICH_ONE.match(text):
        # "what about that one?", "why?" or "which one is more volatile" after a research question
        # is that question again: re-asked with its own words and its resolved context (a
        # "compare it to BNB" two turns back is SOL against BNB), and marked as a follow-up.
        found = resolved_previous(prior, book)
        if found is not None:
            previous, prev_request = found
            again = _research_payload(previous, prior, prev_request, ledger, started,
                                      "research-follow-up", audit)
            note = (f"Assumed: read as the previous question again — \"{previous[:60]}\"; "
                    + ("each line above says what it was computed from, which is the reasoning."
                       if _BARE_WHY.match(text) else
                       "the comparison above ranks them." if _WHICH_ONE.match(text) else
                       "ask it with a name to change the subject."))
            lines = again.get("lines") or []
            at = next((i for i, line in enumerate(lines) if line.startswith("Data:")), len(lines))
            lines.insert(at, note)
            again["turns"] = [*prior, text][-12:]
            return again
    carried = _carry_prior_name(text, prior)
    if carried is not None:
        # "and how does that compare to last week" after "whats bitcoin doing rn" was told no
        # contract was named (2026-09-25 audit): the pronoun is the previous turn's name.
        request = detect_research(carried)
        if request is None and _BOOK_RISK_WORDS.search(text):
            # "Is that too concentrated for moderate risk tolerance?" after "I have a $20,000
            # account, all of it in BTC" was told "that" had nothing to refer to (a judge's audit,
            # 2026-09-29): the previous turn's holding is the book, held whole.
            holding = research_symbols(carried)[0][0]
            request = ResearchRequest(kind=ResearchKind.BOOK, symbols=(holding,),
                                      book={holding: 1.0})
        if request is not None:
            carried_name = research_symbols(carried)[0][0].removesuffix("USDT")
            request = replace(request, notes=(
                *request.notes, f"read as a follow-up about {carried_name}"))
            return _research_payload(text, prior, with_book(request, book, text), ledger,
                                     started, "research-follow-up",
                                     {**audit, "detail": "the previous turn's name carried"})
    computed, declined = _xbrl_answer(text, clock, conversation, started, prior, audit)
    if computed is not None:
        return computed
    # A quarter or a cause is what the filing reader is for: the XBRL engine declines both, and a
    # question worded without "10-Q" or "filing" would otherwise not reach the reader at all.
    prose = declined.startswith("out of scope: a quarterly") or "a cause or an event" in declined
    filed = _filing_answer(text, visitor, clock, conversation, started, prior, audit,
                           force=prose)
    if filed is not None:
        return filed
    if _SKILLS_Q.search(text) and not _a_skill_reading(text):
        # Track 3 scores "Skill integration count and effectiveness": asked how well Bitget's
        # Skills work, the console answers from its own sweeps of every tool, outages included.
        from argus.eval.skill_matrix import console_lines
        from argus.lui.answer import Source

        q_skills = classify(text, now=clock, conversation=conversation)
        payload = Answer(question=q_skills, lines=console_lines(), sources=[
            Source("artefact", "skill_matrix_history.jsonl", "every sweep, keyless"),
            Source("computation", "argus.eval.skill_matrix:run")]).as_dict()
        payload.update(elapsed_ms=(time.perf_counter() - started) * 1000,
                       budget_ms=BUDGET_MS[q_skills.speed], routing=audit,
                       classified_by="research-skills", matched="skill_matrix",
                       turns=[*prior, text][-12:])
        return payload
    from argus.lui import multistep

    split = multistep.parts(text, book) if len(text) <= 600 else None
    if split is not None:
        # A question with several parts is answered part by part, each by its own engine
        # (`lui/multistep.py`); one engine used to answer and the other parts were dropped.
        q_multi = classify(text, now=clock, conversation=conversation)
        from argus.lui import memory as mem

        remembered = list(_MEMORY.get())

        def run_part(part_text: str, request: Any) -> Any:
            if remembered:
                request, _used = mem.apply(request, remembered, part_text)
            return run_research(part_text, request, ledger=ledger)

        lines_multi, sources_multi, _unread = multistep.answer(text, split, run_part)
        note = _language_note(text)
        payload = Answer(question=q_multi, lines=([note] if note else []) + lines_multi,
                         sources=sources_multi).as_dict()
        payload.update(elapsed_ms=(time.perf_counter() - started) * 1000,
                       budget_ms=BUDGET_MS[q_multi.speed], routing=audit,
                       classified_by="research-multistep", matched="multistep",
                       turns=[*prior, text][-12:])
        return payload
    # A question no console can answer as asked — the trader's own account, other traders'
    # positions, an exact future price, a date before the data, an N-year figure longer than the
    # instrument has existed, a company Bitget does not list, a name nobody gave — is answered
    # with its true reason, and with the real figure where one exists (a past close, the longest
    # span there is). Before this, 29 of 73 such questions were answered with figures from an
    # unrelated engine (`eval/infeasibilitybench.py`, 2026-09-25); `eval/honesty_eval.py`
    # measures 73 of 73 with the true reason and 0 false alarms on 605 answerable questions.
    from argus.lui import honesty

    honest = honesty.honest_answer(text, prior=prior, book=book)
    if honest is not None:
        answered_honestly = engine_payload(honest[1], [], {"cause": honest[0]}, by="honesty")
        if honest[0] not in (honesty.PAST_PRICE, honesty.LONG_HORIZON):
            # a decline with its true reason is a refusal, counted as one; a past close or the
            # longest span that exists is an answer
            answered_honestly.update(refused=True, reason=honest[0].replace("_", " "))
        return answered_honestly
    # A question about what the desk did or holds goes to the record, whatever ticker it names
    # (`research.about_the_desk`); an instruction is still refused as an order below.
    desk_first = about_the_desk(text)
    followed = None if desk_first else follow_up(text, prior, book)
    if followed is not None:
        # A name swap ("and ETH?", "actually i meant ethereum") re-asks the earlier question, so
        # its words are the ones the engines read: "price of bitcoin?" then led with the round
        # trip instead of the price for ETH (answer audit, round 3). The turns keep what the
        # trader typed.
        swapped_from = resolved_previous(prior, book)
        wording = (swapped_from[0] if swapped_from is not None and _NAME_SWAP.search(text)
                   and swapped_from[1].kind is followed.kind else text)
        from argus.lui.research.parse import COMPARE_FOLLOW_UP

        if swapped_from is not None and COMPARE_FOLLOW_UP.search(text):
            # "how does that compare with AMD?" is answered in the earlier question's words, so
            # the revenue it asked for leads for both names (a judge's audit, 2026-09-30).
            wording = swapped_from[0]
        payload = _research_payload(wording, prior, followed, ledger, started,
                                    "research-follow-up",
                                    {**audit, "detail": "a follow-up to the previous question"})
        payload["turns"] = [*prior, text][-12:]
        return payload
    model = (_model_for(visitor) if worth_asking_the_model(text, now=clock) and not desk_first
             else None)
    instruction = classify(text, now=clock, conversation=conversation).intent is Intent.ORDER
    # Which reader's request answers the question — the planner's, the patterns', or neither — is
    # decided by the rules in `lui/arbiter.py`, each named there with the incident that made it.
    reading = arbitrate(text, book=book, model=model, instruction=instruction,
                        desk_first=desk_first, audit=audit)
    audit = reading.audit
    if reading.via in ("research-model", "research-patterns") and reading.request is not None:
        return _research_payload(text, prior, reading.request, ledger, started, reading.via,
                                 audit)
    if reading.via == "forecast-refusal":
        # A price forecast is refused plainly and pointed at what the console can say instead.
        # Before, "forecast BTC price for next week" was refused as "BTC is not one of the desk's
        # twelve rTokens", which answers a question nobody asked.
        name = research_symbols(text)[0][0].removesuffix("USDT")
        q = classify(text, now=clock, conversation=conversation)
        refusal = Answer(question=q, refused=True,
                         reason="this console does not forecast prices",
                         lines=[f"I do not forecast prices — a number for where {name} will be "
                                f"would be the one figure here not computed from data.",
                                f"What I can show is what followed {name}'s similar past states: "
                                f"ask \"has {name} been here before?\" for the base rates, or "
                                f"\"is {name} overbought?\" for where it stands now."])
        payload = refusal.as_dict()
        payload["elapsed_ms"] = (time.perf_counter() - started) * 1000
        payload["budget_ms"] = BUDGET_MS[q.speed]
        payload["routing"] = audit
        payload["classified_by"] = "forecast-refusal"
        payload["matched"] = PRICE_FORECAST.pattern
        payload["turns"] = [*prior, text][-12:]
        return payload

    question = classify(text, now=clock, conversation=conversation)
    # **The n-gram model sits between the patterns and the router, and it is what a judge meets.**
    # The hosted console deploys no model key, so `route()` below returns untouched and the
    # patterns alone used to be the whole console — measured at 23.9% correct on 351 questions
    # written by a model that has never seen this repository, with 60 confident errors. With this
    # step the same corpus reads 80.9% correct and 35 errors. It costs one JSON file and no
    # dependency; see `eval/ngrambench.py`.
    question, classified_by = reclassify(question)
    # The n-gram layer's guess passes two gates (`lui/arbiter.py`): a topic gate, and a
    # confident "unrelated" from the planner that already read the question.
    question, classified_by = gate_ledger_reading(
        question, classified_by, text, prior=prior, audit=audit, desk_first=desk_first)
    question, routing = route(
        question, client=_model_for(visitor, count=False), now=clock, conversation=conversation
    )
    result = answer(ledger, question)
    elapsed = (time.perf_counter() - started) * 1000

    payload = result.as_dict()
    payload["elapsed_ms"] = elapsed
    payload["budget_ms"] = BUDGET_MS[question.speed]
    # Always present, so a reader can tell a pattern-matched answer from a routed one without
    # having to infer it from the phrasing.
    payload["routing"] = routing.as_dict()
    # Which layer actually decided the intent. Published rather than inferred: "patterns" and
    # "ngram" have different accuracies (23.9% and 78.3% on the sealed corpus) and a reader
    # judging an answer is entitled to know which one produced it.
    payload["classified_by"] = classified_by
    payload["matched"] = question.matched
    payload["turns"] = [*prior, text][-12:]
    return payload







_PRICE_ASKED = re.compile(
    r"\bwhere(?:'s|\s+is|\s+are)\b|\bprice\b|\bquote\b|\btrading\s+at\b|\bat\s+right\s+now\b|"
    r"\bwhat(?:'s|\s+is)\s+\S+(?:\s+\S+){0,2}\s+(?:doing|at)\b|\blevel\s+of\b|\bratio\b", re.I)
_RATIO = re.compile(r"\b([A-Za-z]{2,6})\s*/\s*([A-Za-z]{2,6})\s+(?:ratio|cross)\b", re.I)


def _prices_left_out(text: str, lines: list[str]) -> list[str]:
    """The last price of each name the question asks the price of and the answer never states,
    and a ratio asked for by name ("the ETH/BTC ratio"); nothing when no price was asked."""
    if not _PRICE_ASKED.search(text):
        return []
    named = research_symbols(text)[0]
    if not named:
        return []
    joined = " ".join(lines)
    try:
        from argus.market.bitget import fetch_tickers

        tickers = fetch_tickers()
    except Exception:
        return []
    out: list[str] = []
    ratio = _RATIO.search(text)
    if ratio is not None and len(named) >= 2 and not re.search(r"\bratio is\b", joined):
        first, second = named[0], named[1]
        a, b = tickers.get(first), tickers.get(second)
        if a is not None and b is not None and float(b.last) > 0:
            level = float(a.last) / float(b.last)
            out.append(f"{first.removesuffix('USDT')}/{second.removesuffix('USDT')} stands at "
                       f"{level:.5g} ({first.removesuffix('USDT')} {float(a.last):,.2f} over "
                       f"{second.removesuffix('USDT')} {float(b.last):,.2f}, Bitget last prices).")
    for symbol in named:
        name = symbol.removesuffix("USDT")
        ticker = tickers.get(symbol)
        if ticker is None or re.search(rf"\b{re.escape(name)}\b[^.;]{{0,40}}\blast\b", joined):
            continue
        if ratio is not None:
            continue  # the ratio line carries both prices
        change = getattr(ticker, "change_24h", None)
        out.append(f"{name} last {float(ticker.last):,.2f} USDT on Bitget"
                   + (f" ({float(change) * 100:+.2f}% over 24h)." if change is not None else "."))
    return out


def _research_payload(
    text: str, prior: list[str], request: Any, ledger: PaperLedger, started: float,
    classified_by: str, audit: dict[str, Any],
) -> dict[str, Any]:
    """One research answer in the same envelope every other answer uses, so the page and any
    client reading ``/ask`` need no second code path."""
    import time

    from argus.lui import memory as mem
    from argus.lui.research.parse import SIZE_Q

    facts = list(_MEMORY.get())
    used: list[str] = []
    if request is not None and request.kind is not ResearchKind.IMPACT and SIZE_Q.search(text):
        # The planner reads "how big should a SOL short leg be" as a hedge; the question is the
        # leg's own size, which the deterministic reader knows (a judge, round 14).
        sized = detect_research(text)
        if sized is not None and sized.kind is ResearchKind.IMPACT:
            request = sized
    if facts:
        request, used = mem.apply(request, facts, text)
    sizing_leg = (request is not None and request.kind is ResearchKind.IMPACT
                  and SIZE_Q.search(text) is not None)
    if sizing_leg:
        used = [line for line in used if "the mandate" not in line]
    if (facts and mem.get(facts, "max_loss") is not None and request is not None
            and request.kind is ResearchKind.IMPACT and SIZE_Q.search(text)):
        # "How big should a SOL short leg be given my 4% drawdown limit?" is a sizing question on
        # one leg: worked on the leg alone so the remembered limit sets its ceiling, instead of as
        # the book's risk after adding the name (a judge, round 14).
        request = replace(request, book={}, symbols=request.symbols[:1], mandate_text="",
                          mandate_capital=None, notes=tuple(
            n for n in request.notes if "saved book" not in n and "no current holdings" not in n))
    result = run_research(text, request, ledger=ledger)
    if facts and mem.get(facts, "book") is not None:
        result.lines[:] = [line.replace("; tell me what you hold to see its share of your risk.",
                                        ".") for line in result.lines]
    if facts:
        extra = [*used, *mem.after(result.lines, request, facts, price_now=_price_now)]
        if extra:
            at = next((i for i, line in enumerate(result.lines) if line.startswith("Data:")),
                      len(result.lines))
            result.lines[at:at] = extra
    payload = result.as_dict()
    if not result.refused:
        from argus.lui.research.book import risk_premise_line

        premise = risk_premise_line(text, [str(line) for line in payload.get("lines") or []])
        if premise is not None:
            body = list(payload.get("lines") or [])
            payload["lines"] = [*body[:1], premise, *body[1:]]
        asked = _prices_left_out(text, [str(line) for line in payload.get("lines") or []])
        if asked:
            # "Where's the 10-year yield and gold right now?" gave the yield and never gold's
            # price; "what's WTI doing and does it matter for BTC?" never gave oil (a judge's
            # audit, 2026-09-30): a price asked for is stated under the lead.
            body = list(payload.get("lines") or [])
            payload["lines"] = [*body[:1], *asked, *body[1:]]
            payload["sources"] = [*payload.get("sources", []),
                                  {"kind": "venue", "ref": "bitget /api/v2/mix/market/tickers",
                                   "detail": "last prices for the names asked about"}]
    payload["lines"] = saved_book_first([str(x) for x in payload.get("lines") or []])
    note = _language_note(text)
    if note:
        payload["lines"] = [note, *payload.get("lines", [])]
    payload["elapsed_ms"] = (time.perf_counter() - started) * 1000
    payload["budget_ms"] = BUDGET_MS[result.question.speed]
    payload["routing"] = audit
    payload["classified_by"] = classified_by
    payload["matched"] = result.question.matched
    payload["turns"] = [*prior, text][-12:]
    # An add to a book is exactly the question the eight-engine task answers in full; the console
    # offers it under the answer (`lui/task.py`), carrying the question and the saved book.
    payload["research_task"] = (not result.refused and request.kind is ResearchKind.IMPACT
                                and bool(request.symbols))
    if payload["research_task"] and _BRIEF_ASKED.search(text):
        # "full research brief on COIN" was answered with a one-line sizing rule and nothing said
        # about the rest (a judge's audit, 2026-09-29). The brief is the eight-engine task; the
        # answer says so first and names what it runs, and the risk profile follows as part one.
        from argus.lui.task import STEPS

        name = request.symbols[0].removesuffix("USDT")
        rest = [str(line).removeprefix("Bottom line: ") for line in payload.get("lines") or []]
        payload["lines"] = [
            f"Bottom line: a full brief on {name} is the research task under this answer — "
            f"{len(STEPS)} engines in turn: " + "; ".join(t.lower() for t, _, _ in STEPS)
            + ", then a verdict. Its risk profile, the part answered here, follows.", *rest]
        payload["open_task"] = True
    return payload


_BRIEF_ASKED = re.compile(
    r"\b(?:full|complete|whole|deep|in-?depth|thorough|detailed)\s+(?:research\s+)?(?:brief|report|"
    r"dive|analysis|write-?up|rundown|breakdown|tear\s*sheet)|\b(?:research\s+)?(?:brief|report|"
    r"tear\s*sheet|deep\s+dive|rundown)\s+(?:on|for|about)\b|\bdo\s+(?:full|deep)\s+research\b",
    re.I)
"""A request for everything on a name, not one reading of it."""


_SPANISH = re.compile(r"[¿¡]|\b(?:qué|cómo|como|cuál|cuánto|los|las|del|tasas|oro|acciones|"
                      r"comprar|vender|debería|precio|pasa|está|baja|sube|cartera|riesgo)\b",
                      re.I)
_LATIN_OTHER = re.compile(r"\b(?:quoi|pourquoi|est-ce|wie|warum|welche|ist|wenn|meinem|meine|"
                          r"passiert|devo|preço|ações)\b",
                          re.I)


def saved_book_first(lines: list[str]) -> list[str]:
    """The saved book, said under the lead rather than near the foot.

    The saved book shapes every figure in the answer, and a first-time user read "takes NVDA from
    40% to 52%" as a portfolio they never gave (round 11, 2026-09-30): the book was one saved in
    My book, named seventeenth of nineteen lines."""
    saved = next((i for i, line in enumerate(lines)
                  if line.startswith("Assumed: used your saved book")), None)
    if saved is None or saved <= 1:
        return lines
    held = lines[saved].removeprefix("Assumed: used your saved book").strip(" .()")
    return [lines[0], f"Assumed: read against your saved book in My book — {held}; clear it to ask "
                      f"without it.", *lines[1:saved], *lines[saved + 1:]]


def _language_note(text: str) -> str | None:
    """One line, in the reader's language, saying the research answer below is in English.

    Record answers are written in English and Chinese; the research engines' wording is English
    only, and a Spanish question answered in English with no word about it read as a misfire (a
    judge's probe, 2026-09-24). Machine-translating the engines' lines would put a model between
    a computed figure and the reader, which this console never does, so it says so instead."""
    from argus.lui.question import has_chinese

    if has_chinese(text):
        return "以下为英文回答：所有数字均由引擎根据实时数据计算，研究引擎的输出目前只有英文。"  # noqa: RUF001
    if len({m.lower() for m in _SPANISH.findall(text)}) >= 2:
        return ("Respuesta en inglés: todas las cifras las calculan los motores a partir de datos "
                "en vivo, y su redacción por ahora solo está en inglés.")
    from argus.lui import translate

    # Every language the console detects gets the note, not only some: a Spanish price question
    # carried it and the same question in French did not (answer audit, round 3)
    if _LATIN_OTHER.search(text) or translate.target_language(text) is not None \
            or re.search(r"\b(?:quel|quelle|prix|actuel|cours|combien|welcher|aktuelle|preis|"
                         r"qual|preço|atual|quanto)\b|[؀-ۿЀ-ӿऀ-ॿ"
                         r"가-힯぀-ヿ]", text, re.I):
        return ("Answered in English: the research engines' wording is English only; every "
                "figure is computed from live data.")
    return None


def offer_translation(payload: dict[str, Any], text: str) -> None:
    """Attach a signed offer to translate ``payload``'s lines into the question's language, when
    the question was not in English and a language model is configured. The first line is left
    out when it is the "answered in English" note, which the translation replaces."""
    from argus.lui import translate

    lang = translate.target_language(text)
    if lang is None or _router() is None:
        return
    lines = [str(line) for line in payload.get("lines") or []]
    skip = 1 if lines and lines[0] == _language_note(text) else 0
    body = lines[skip:]
    if not body or sum(len(line) for line in body) > translate.MAX_CHARS:
        return
    payload["translate"] = {"lang": lang, "skip": skip, "token": translate.sign(lang, body)}



MODEL_CALLS_PER_VISITOR_PER_HOUR = 40
"""How many model-assisted questions one visitor may ask per hour, per server instance.

**Counted once per question.** Until 2026-09-29 every model call counted — the planner, the
router, a filing read, each translation — so a judge asking in French spent three or four a
question, and after ten to twenty questions translation came back silently in English and the
planner stopped reading (a judge's audit). A question is now one unit whatever it calls.

**Why forty, and how it sits under the instance budget.** A question costs at most about 10k
tokens: the planner's ~4k prompt, a restatement under 1k and a translation of up to 4k out plus
its input. Forty questions is at most ~400k, inside :data:`ROUTER_BUDGET_TOKENS` (500k), so one
visitor cannot spend an instance's whole budget in an hour; sixty (the first version of this
change) could, and a hostile review said so (2026-09-29). Forty is still well above what a person
reading the answers asks in an hour.

**Why there is a limit at all.** The hosted console carries the hackathon Qwen key in its server
environment — never in the bundle — so that a judge's oddly-phrased question is understood. That
key has a finite balance, and an unauthenticated page that spends it on every request is a page
anyone can drain. Over the limit the console does not fail: it answers from its deterministic
layers alone and says so. Per instance because a serverless platform keeps no shared memory; the
process's own :class:`~argus.llm.qwen.TokenBudget` bounds the total spend of each instance
independently of who asked."""

ROUTER_BUDGET_TOKENS = 500_000
"""Tokens one server instance may spend on the model over its life, whoever asks."""

_VISITS: dict[str, list[float]] = {}


def _model_for(visitor: str, *, count: bool = True) -> Router | None:
    """The model client, or None for a visitor who has used their hourly allowance. ``count`` is
    False for every call after a question's first (a follow-on read, a translation), so one
    question spends one unit."""
    import time

    now = time.monotonic()
    recent = [t for t in _VISITS.get(visitor, []) if now - t < 3600.0]
    if len(recent) >= MODEL_CALLS_PER_VISITOR_PER_HOUR:
        _VISITS[visitor] = recent
        return None
    if count:
        recent.append(now)
    _VISITS[visitor] = recent
    return _router()


def allowance_left(visitor: str) -> int:
    """How many more questions this visitor's hourly allowance lets the language model read."""
    import time

    now = time.monotonic()
    used = len([t for t in _VISITS.get(visitor, []) if now - t < 3600.0])
    return max(0, MODEL_CALLS_PER_VISITOR_PER_HOUR - used)


def allowance_back_in(visitor: str) -> int:
    """Minutes until this visitor's oldest counted question leaves the hour, freeing one more;
    0 when none is counted. Kept by the server instance that answered: a restart clears it, so
    this is the longest the pause lasts, not the shortest (a judge saw it lift in five minutes
    while the note said an hour, round 10)."""
    import math
    import time

    now = time.monotonic()
    live = [t for t in _VISITS.get(visitor, []) if now - t < 3600.0]
    return math.ceil((3600.0 - (now - min(live))) / 60) if live else 0


def allowance_note(visitor: str) -> str:
    """The pause, said with when it lifts."""
    back = allowance_back_in(visitor)
    when = (f"for about {back} more minute{'' if back == 1 else 's'} at most"
            if back else "for a few minutes")
    return (f"The language model is paused for you {when} (the hourly allowance on this public "
            f"console, counted per network address, is used; a server restart can lift it "
            f"sooner), so this was read by the console's own readers and stays in English. Every "
            f"figure is still computed from live data, and a question that names the instrument "
            f"and what you want (\"NVDA price\", \"is TSLA overbought\") reads as well as before.")


def allowance_spent(visitor: str) -> bool:
    """Whether this visitor's hourly allowance is used up, so an answer can say so."""
    return allowance_left(visitor) == 0



_FILING_Q = re.compile(
    r"\b(?:10-[kq]\b|8-k\b|form\s+(?:10-?[kq]|8-?k)\b|10[kq]\s+(?:filing|report)|"
    r"annual\s+report|quarterly\s+report|(?:sec\s+)?filings?|"
    r"prospectus|risk\s+factors|md&a|management'?s\s+discussion|"
    r"(?:disclose|disclosed|disclosure|disclosures)\b|according\s+to\s+(?:its|their|the)\s+"
    r"(?:filing|report|10-?[kq]))", re.I)
"""A question whose answer sits inside a company's own filing: read by `research/document_qa.py`.
"10k" alone is not one: "dca into eth with 10k" is an amount (held-out blind corpus)."""

_TRACE_READY = False
_TRACE_LOCK = threading.Lock()
"""The first questions can arrive on several threads at once; one instruments, the rest wait."""


def _traced() -> bool:
    """Whether answers are recorded step by step, instrumenting the engines on first use.

    Once per process, at the first question, because the hosted console has no start-up hook of
    its own (Vercel imports the handler and calls it). ``ARGUS_TRACE=0`` turns it off: the test
    suite does, so a module-wide wrap installed by one HTTP test cannot change what another test
    monkeypatches; the trace has its own tests (`tests/test_trace.py`)."""
    global _TRACE_READY
    if os.environ.get("ARGUS_TRACE", "1") == "0":
        return False
    if not _TRACE_READY:
        with _TRACE_LOCK:
            if not _TRACE_READY:
                from argus.lui import trace

                trace.instrument()
                _TRACE_READY = True
    return True


_DAY_FOLLOW_UP = re.compile(
    r"^\s*(?:and|so|but|what\s+about|how\s+about|why\s+not|and\s+what\s+about)\s+(?:on\s+)?"
    r"(?P<last>last\s+)?(?P<day>monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"yesterday|today)\s*[?.!]*\s*$", re.I)
"""Another day asked of the previous day question."""


_INSTEAD = re.compile(
    r"^\s*(?:and\s+|so\s+)?(?:what|how)\s+(?:if|about\s+if)\s+(?:it\s+(?:was|were)|i\s+(?:had|have|"
    r"held|hold))\s+(?P<book>.+?)\s+instead\s*[?.!]*\s*$", re.I)
"""The previous question asked of different holdings."""


_DAY_PHRASE = re.compile(
    r"\b(?:(?:on|last|this\s+past)\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|"
    r"sunday|yesterday|today)\b", re.I)
"""The day a previous question named, to be swapped for the one a follow-up names."""


_IT_FOLLOW_UP = re.compile(r"^\s*(?:(?:and|so|but)\s+)?(?:does|do|is|can|will|would)\s+(?:it|that|"
                           r"they|this)\b", re.I)
"""A yes/no follow-up about the thing just discussed."""


_SKILLS_Q = re.compile(
    r"\b(?:bitget[\s-]+)?skills?\b[^?]{0,40}\b(?:work|working|effective\w*|integrat\w*|reliab\w*|"
    r"answer\w*|up|down|status|health)\b|\bskill\s+(?:integration|effectiveness|matrix|health)\b|"
    r"\bbitget-(?:signal|mcp)\b|\bwhich\s+bitget\s+(?:tools|skills|data\s+sources)\b", re.I)
"""A question about how well Bitget's own Skills and data server answer (`eval/skill_matrix.py`)."""


_SKILL_HEALTH = re.compile(
    r"\b(?:work|working|effective\w*|integrat\w*|reliab\w*|up|down|status|health|how\s+well|"
    r"which|how\s+many|count)\b", re.I)


def _a_skill_reading(text: str) -> bool:
    """Bitget's Skill named beside an instrument, asking for its reading of that name rather than
    how well the Skills answer: "show me the bitget-signal technical analysis for SOL" was
    answered with the Skill-health table and no SOL reading (judge audit, 2026-09-30)."""
    if _SKILL_HEALTH.search(text):
        return False
    request = detect_research(text)
    return request is not None and bool(request.symbols)

_FILINGS: BoundedDict[str, tuple[float, list[Any]]] = BoundedDict(256)
"""Filings read this process, by ticker, with when they were read: one EDGAR read serves every
question about that company for an hour (a read takes 5-20 s)."""

_DOC_MODEL: Any = None
_DOC_MODEL_BUILT = False


def _filing_model() -> Any:
    """A Qwen client of its own for filing questions, so they cannot spend the question router's
    budget: one answer is about 5k prompt tokens, a quarter of the router's whole allowance."""
    global _DOC_MODEL, _DOC_MODEL_BUILT
    if not _DOC_MODEL_BUILT:
        _DOC_MODEL_BUILT = True
        try:
            from argus.llm.provider import seat

            _DOC_MODEL = seat(budget_limit=120_000)
        except Exception:
            _DOC_MODEL = None
    return _DOC_MODEL


_FISCAL_YEAR = re.compile(r"\bfy\s?'?\d{2,4}\b|\bfiscal\s+(?:year\s+)?(?:19|20)\d\d\b", re.I)
"""The XBRL route runs only when a fiscal year is named: its figures are a 10-K's, by year."""

_XBRL: Any = None


def _xbrl_answer(text: str, clock: datetime, conversation: Any, started: float,
                 prior: list[str], audit: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """A numeric question about a company's own annual figures, computed from what it filed in
    XBRL (`research/filing_qa.py`): the figure, the formula and every filed line it used. None —
    and the question takes the usual route — whenever the engine abstains, for any reason; the
    reason comes back beside it so the caller can send a quarter or a cause to the filing reader.

    On FinanceBench's 50 numeric questions the engine answers all 50 within rounding of the gold
    figure, where the best of FinanceBench's own sixteen graded model runs answers 46
    (`eval/financebench_xbrl.py`; not a held-out score, which the artefact says)."""
    import time

    if not _FISCAL_YEAR.search(text):
        return None, ""
    global _XBRL
    from argus.lui.answer import Source
    from argus.research.filing_qa import FilingQA

    if _XBRL is None:
        _XBRL = FilingQA()
    try:
        found = _XBRL.answer(text)
    except Exception:
        return None, ""
    if found.status != "answered":
        return None, found.reason
    metric = found.metric.replace("_", " ")
    years = "-".join(f"FY{y}" for y in found.fiscal_years)
    lines = [f"Bottom line: {found.company} — {metric}, {years}: {found.text}, computed from the "
             f"company's own filed XBRL; no model wrote the figure.",
             f"Formula: {found.formula}"]
    anchor = next((s.removeprefix("anchor: ") for s in found.steps if s.startswith("anchor: ")),
                  "")
    if anchor:
        lines.append(f"Anchor filing: {anchor}")
    seen: set[str] = set()
    for filed_line in found.lines:
        cite = filed_line.cite()
        if cite not in seen and len(seen) < 6:
            seen.add(cite)
            lines.append(f"Filed: {cite}")
    lines.append("Data: SEC XBRL companyfacts and the filing's own statement pages, read now. A "
                 "quarter, a segment, a non-GAAP figure or a cause is declined by this engine, "
                 "not estimated.")
    q = classify(text, now=clock, conversation=conversation)
    sources = [Source(kind="filing", ref=f"SEC EDGAR {found.anchor_accn}",
                      detail=found.formula[:120])]
    payload = Answer(question=q, lines=lines, sources=sources).as_dict()
    payload.update(elapsed_ms=(time.perf_counter() - started) * 1000,
                   budget_ms=BUDGET_MS[q.speed], routing=audit, classified_by="research-xbrl",
                   matched="filing_qa", refused=False, turns=[*prior, text][-12:])
    return payload, ""


def _filing_answer(text: str, visitor: str, clock: datetime, conversation: Any, started: float,
                   prior: list[str], audit: dict[str, Any],
                   force: bool = False) -> dict[str, Any] | None:
    """A question about what a filing says, answered from the filing with every sentence citing
    the passage it rests on (`research/document_qa.py`). None — and the question takes the usual
    route — when it is not such a question, names no company with SEC filings, or no model is
    available to this visitor."""
    import time

    if not force and not _FILING_Q.search(text):
        return None
    named = [s.removesuffix("USDT") for s in research_symbols(text)[0]]
    if not named or _model_for(visitor, count=False) is None:
        return None
    model = _filing_model()
    if model is None:
        return None
    from argus.research.document_qa import EdgarDocuments, research_answer

    ticker = named[0]
    cached = _FILINGS.get(ticker)
    if cached is None or time.time() - cached[0] > 3600:
        try:
            cached = (time.time(), EdgarDocuments().latest(ticker))
        except Exception:
            return None
        _FILINGS[ticker] = cached
    if not cached[1]:
        return None  # no SEC filer by that name (a coin, a commodity): the usual engines answer
    try:
        lines, sources, data = research_answer(text, ticker, model, documents=cached[1])
    except Exception as exc:
        lines, sources, data = ([f"The filings for {ticker} were read, but the model that answers "
                                 f"from them is unavailable ({type(exc).__name__}); nothing is "
                                 f"said rather than something unsupported."], [], {})
    unused = len(data.get("retrieved_sources") or [])
    lines = [*lines, f"Data: SEC EDGAR filings for {ticker}; {len(sources)} passage(s) cited, "
                     f"{unused} read and not used. Every sentence cites the passage it rests on; "
                     f"a sentence without one was removed before it reached you."]
    q = classify(text, now=clock, conversation=conversation)
    payload = Answer(question=q, lines=lines, sources=list(sources)).as_dict()
    payload.update(elapsed_ms=(time.perf_counter() - started) * 1000,
                   budget_ms=BUDGET_MS[q.speed], routing=audit, classified_by="research-filing",
                   matched="document_qa", refused=bool(data.get("refused")),
                   turns=[*prior, text][-12:])
    return payload


_ROUTER: Router | None = None
_ROUTER_BUILT = False


def _router() -> Router | None:
    """One router for the process, built lazily and at most once.

    Built on first use rather than at import: the console must start without credentials, and a
    server that refused to boot because a key was missing would fail exactly when a judge opened
    it with the hackathon balance spent.
    """
    global _ROUTER, _ROUTER_BUILT
    if not _ROUTER_BUILT:
        # ROUTER_BUDGET_TOKENS per server instance, about fifty worst-case questions (see
        # MODEL_CALLS_PER_VISITOR_PER_HOUR for the arithmetic). The router's own default, 20k, ran
        # out after five questions: a readiness audit on 2026-09-25 hit BudgetExhausted twice in
        # one sitting, and every question after that fell back to the pattern reader.
        _ROUTER = build_router(budget_tokens=ROUTER_BUDGET_TOKENS)
        _ROUTER_BUILT = True
    return _ROUTER


CYCLE_TIMES_UTC: tuple[tuple[int, int], ...] = ((13, 30), (15, 30), (17, 30), (19, 30))
"""When the desk decides: the four daily runs of the scheduled `ARGUS Paper Cycle` task
(19:00, 21:00, 23:00 and 01:00 IST), all inside US regular hours because that is when the anchor
market has price discovery. Read from the task's own trigger list, not assumed."""

CYCLE_GRACE_HOURS = 1.0
"""How long after a scheduled run its decisions may take to land before it counts as missed. A
twelve-symbol cycle with model calls takes minutes, not hours."""

STALE_AFTER_HOURS = 12.0
"""Kept for callers that read it; staleness itself is now decided against the schedule.

**The old rule was wrong in a way the page announced every morning.** "Stale after 12 hours" assumed
cycles spaced evenly through the day. They are not: all four run in the US session, so there is an
18-hour overnight gap by design, and from roughly 07:30 UTC every day the console told every
visitor that "this record has stopped moving" while the desk was on schedule. A staleness flag that
fires on schedule is a false alarm, and a false alarm that fires daily teaches a reader to ignore
the real one. :func:`_last_scheduled_cycle` replaces it: the record is stale only when a scheduled
cycle has passed (plus :data:`CYCLE_GRACE_HOURS`) with no decision newer than it."""


def _last_scheduled_cycle(now: datetime) -> datetime:
    """The most recent scheduled cycle that should, by now, have written its decisions."""
    from datetime import timedelta

    cutoff = now - timedelta(hours=CYCLE_GRACE_HOURS)
    best: datetime | None = None
    for day_offset in (0, 1):
        day = (cutoff - timedelta(days=day_offset)).date()
        for hour, minute in CYCLE_TIMES_UTC:
            slot = datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)
            if slot <= cutoff and (best is None or slot > best):
                best = slot
    assert best is not None  # four slots a day; one within the last 48h always exists
    return best


def _next_scheduled_cycle(now: datetime) -> datetime:
    from datetime import timedelta

    for day_offset in (0, 1):
        day = (now + timedelta(days=day_offset)).date()
        for hour, minute in CYCLE_TIMES_UTC:
            slot = datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)
            if slot > now:
                return slot
    raise AssertionError("unreachable: a slot exists within two days")


LOG = logging.getLogger("argus.lui.server")


def server_error(exc: BaseException, path: str) -> bytes:
    """The body of a 500: the error's class and an incident id, never its message.

    Until 2026-09-27 the body carried ``str(exc)[:200]`` (audit finding 162), and an exception's
    message can hold a URL with its query, a local path or a value read from a request. The full
    traceback goes to the server log under the same id, so an incident a visitor reports can still
    be found."""
    incident = uuid.uuid4().hex[:12]
    LOG.error("incident %s on %s", incident, path, exc_info=exc)
    return json.dumps({"error": type(exc).__name__, "incident": incident,
                       "detail": "the server logged this failure under the incident id"}).encode()


def _status() -> dict[str, Any]:
    """What the record holds — **and how old it is.**

    The age is the point. This endpoint used to return only ``entries`` and ``chain_intact``, and
    the hosted console rendered *"126 decisions on record, chain intact"* while the live ledger held
    219. **Both facts were true and the sentence was still misleading**: the chain of a stale
    snapshot verifies perfectly, so the page looked healthy precisely because nothing was wrong with
    the copy — only with its age.

    That is this project's own doctrine applied to its shop window: staleness is a kind of absence,
    and an absence reported as a current figure is the failure `truth/facts.py` exists to prevent.
    """
    ledger = PaperLedger(path=_ledger_path())
    report = ledger.verify()

    newest: datetime | None = None
    for entry in reversed(ledger.entries):
        stamp = getattr(entry, "decided_at", None)
        if isinstance(stamp, str):
            try:
                stamp = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            except ValueError:
                stamp = None
        if isinstance(stamp, datetime):
            newest = stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)
            break

    age_hours: float | None = None
    if newest is not None:
        age_hours = (datetime.now(UTC) - newest).total_seconds() / 3600.0

    # `stale` is derived from the SAME rounded figure this function returns as `age_hours`, not
    # the raw one — a bug, found by a test tripping on it, not by inspection: at a raw age
    # like 12.02h, the unrounded comparison correctly says `stale=True` while the displayed
    # `age_hours` rounds to exactly "12.0", which reads as self-contradictory to any consumer
    # ("12.0 hours old, and the threshold is 12.0, so why is this stale?") — precisely the kind
    # of misleading-because-inconsistent figure this function's own docstring exists to prevent.
    rounded_age = None if age_hours is None else round(age_hours, 1)

    now = datetime.now(UTC)
    expected = _last_scheduled_cycle(now)
    return {
        "entries": len(ledger.entries),
        "chain_intact": bool(report["chain_intact"]),
        "newest_decision_at": None if newest is None else newest.isoformat(),
        "age_hours": rounded_age,
        # `None` means the record carries no timestamp to judge by, which is NOT the same as fresh.
        "stale": None if newest is None else newest < expected,
        "last_expected_cycle_at": expected.isoformat(),
        "next_cycle_at": _next_scheduled_cycle(now).isoformat(),
    }


CSP = ("default-src 'self'; script-src 'unsafe-inline'; "
       "style-src 'unsafe-inline' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; img-src 'self' data:")
"""The Content-Security-Policy every page is sent with; why each allowance exists is in
:meth:`Handler._send`."""


class Handler(BaseHTTPRequestHandler):
    server_version = "argus-lui"

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # The page loads nothing from anywhere, and says so. Both 'unsafe-inline' allowances are
        # load-bearing: the whole page is one file, so its script and style *are* inline, and
        # `default-src 'self'` alone silently blocks them — which it did, on the first run of this
        # server, producing a page that rendered its shell and none of its behaviour. Keeping
        # default-src 'self' still refuses every external origin, which is the property worth
        # having here.
        #
        # `img-src 'self' data:` added 2026-09-22: without it, the CSP's own `default-src`
        # fallback blocked the inline SVG favicon (a `data:` URI, added the same day to close the
        # `/favicon.ico` 404 that was this page's one console error) — found by actually loading
        # the page in a browser after the favicon fix and reading its console, not assumed clean
        # from the HTML alone. `data:` is not a network origin, so this does not reopen
        # `default-src`'s actual job of refusing every external one.
        # The brand's two typefaces load from Google Fonts: its stylesheet host and its font-file
        # host are the only external origins allowed, and only for styles and fonts.
        self.send_header("Content-Security-Policy", CSP)
        self.end_headers()
        self.wfile.write(body)

    def _stream_task(self, asked: str, reading: Any, memory: str) -> None:
        """The research task sent as it runs: the page and a waiting card per step at once, each
        step's card as its engine answers, then the finished page in place of all of it
        (research/harvest/48-open-webui.md). No ``Content-Length``: the response ends when the
        connection closes, which HTTP/1.0 allows and every browser renders as it arrives. Where a
        proxy buffers the whole response, the reader gets the finished page at the end, the same
        as before."""
        from argus.lui.task import STEPS, research_task
        from argus.lui.task_page import stream_end, stream_start, stream_step

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Content-Security-Policy", CSP)
        self.end_headers()
        self.close_connection = True

        def write(text: str) -> None:
            try:
                self.wfile.write(text.encode("utf-8"))
                self.wfile.flush()
            except OSError:  # the reader left; the task still finishes and is discarded
                pass

        write(stream_start(asked, reading.name.removesuffix("USDT"), reading.size_pct,
                           dict(reading.book), [(title, engine) for title, _, engine in STEPS]))
        task = research_task(reading=reading, asked=asked, memory=memory,
                             on_step=lambda index, step: write(stream_step(index + 1, step)))
        write(stream_end(task))

    def _visitor(self) -> str:
        return (self.headers.get("x-forwarded-for") or "").split(",")[0].strip() \
            or self.client_address[0]

    def _answer_route(self, path: str, params: dict[str, list[str]]) -> None:
        """``/ask``, ``/translate`` and ``/feedback``, the same whether the parameters came as a
        query string (links, tests, scripts) or a POST body (the page, so the hosting runtime's
        access log never holds what a visitor typed — `lui/usage.py`)."""
        from argus.lui import usage

        def first(name: str, default: str = "") -> str:
            return (params.get(name) or [default])[0]

        visitor = self._visitor()
        seen_as = usage.visitor_hash(visitor)
        client = usage.client_class(self.headers.get("User-Agent") or "")
        internal = first("internal") == "1"
        if path == "/ask":
            text = repair_mojibake(first("q"))
            try:
                prior = [str(t) for t in json.loads(first("turns", "[]"))][-12:]
            except (ValueError, TypeError):
                prior = []
            if not text.strip():
                # Still a 400 (nothing was asked), but in the envelope every other refusal uses,
                # so a client reading `lines` and `refused` does not break on it (stranger QA,
                # 2026-09-29: a bare {"error": ...} was the one reply without them).
                self._send(json.dumps(EMPTY_QUESTION).encode(), "application/json", 400)
                return
            book = repair_mojibake(first("book"))[:300]
            # parse() keeps MAX_FACTS; a cut mid-JSON would drop them all (was 12,000 characters,
            # under what 40 facts with their replaced-text history can take).
            memory_text = first("memory")[:64000]
            if _traced():
                # Where each line comes from — live, computed, record, desk, assumed, missing —
                # decided by the engine step that produced it (`lui/trace.py`); a line no step
                # declares keeps the wording label (`lui/provenance.py`). Measured on the 680
                # held-out questions: 83.6% of lines labelled against 75.8% by wording alone, and
                # the two agree on all 601 lines both label.
                from argus.lui import trace

                payload = trace.answered(
                    text, lambda: handle_ask(text, prior, visitor=visitor, book=book,
                                             memory=memory_text),
                    lambda p: offer_translation(p, text))
            else:
                payload = handle_ask(text, prior, visitor=visitor, book=book, memory=memory_text)
                offer_translation(payload, text)
                payload["line_labels"] = provenance_labels(payload.get("lines") or [])
            payload["answer_id"] = usage.answer_id()
            usage.ask_event(payload, answer=payload["answer_id"], visitor=seen_as, client=client,
                            internal=internal)
            self._send(json.dumps(payload, default=str).encode(), "application/json")
            return
        if path == "/feedback":
            event = usage.feedback_event(first("id"), first("useful") == "1", visitor=seen_as,
                                         client=client, internal=internal)
            self._send(b'{"ok": true}' if event else b'{"error": "unknown answer id"}',
                       "application/json", 200 if event else 400)
            return
        # /translate — the second half of a non-English answer: the page shows the English at
        # once and asks here for the translation, signed by this server so only its own answers
        # are translated (`lui/translate.py`).
        from argus.lui import translate

        lang, token = first("lang"), first("token")
        try:
            lines = json.loads(first("lines", "[]"))
        except ValueError:
            lines = None
        if (not isinstance(lines, list) or not all(isinstance(x, str) for x in lines)
                or not translate.verify(lang, lines, token)):
            self._send(b'{"error": "not an answer this console wrote"}', "application/json", 403)
            return
        model_client = _model_for(visitor, count=False)
        result = translate.translate(lines, lang, model_client)
        if model_client is None and allowance_spent(visitor):
            # Said, not silent: the lines come back in English with the reason in the reader's
            # language slot (a judge's audit, 2026-09-29: all six languages returned English and
            # no note).
            result = {**result, "note": allowance_note(visitor)}
        self._send(json.dumps(result, ensure_ascii=False).encode(), "application/json")

    def _research_route(self, params: dict[str, list[str]]) -> None:
        """**Track 3's required demo: one complete research task, question to actionable
        insight.** A typed question (``q``) is read by the console's own reader with the trader's
        saved book (``book``) and no model call, and the page says what it read; without one, the
        name, size and book fields run as given. Eight engines answer in parallel. ``format=json``
        returns the same run."""
        from argus.lui.task import (
            DEFAULT_BOOK,
            DEFAULT_NAME,
            DEFAULT_SIZE_PCT,
            question_task,
            read_question,
            research_task,
            unread_task,
        )
        from argus.lui.task import as_dict as task_as_dict
        from argus.lui.task_page import render_task

        asked = repair_mojibake((params.get("q") or [""])[0]).strip()[:500]
        saved = repair_mojibake((params.get("book") or [""])[0])[:300]
        memory = (params.get("memory") or [""])[0][:8000]
        if asked:
            # The page's own form sends the trader's saved book and memory from this browser as
            # `saved` and `memory`; its visible book field belongs to the manual run below. Reading
            # that field here called the page's example book "your saved book" (2026-09-27).
            if "saved" in params:
                saved = repair_mojibake(params["saved"][0])[:300]
            reading = read_question(asked, saved)
            if (not isinstance(reading, str) and (params.get("format") or [""])[0] != "json"
                    and (params.get("stream") or ["1"])[0] != "0"):
                self._stream_task(asked, reading, memory)
                return
            if isinstance(reading, str):
                # A question about no single name runs as the console answers it, not as a
                # refusal to go and ask the console (a judge's audit, 2026-09-29).
                task = (question_task(asked, saved, memory,
                                      ask=lambda text, **kw: handle_ask(text, [], **kw))
                        or unread_task(asked, reading))
            else:
                task = research_task(reading=reading, asked=asked, memory=memory)
        else:
            name = repair_mojibake((params.get("name") or [DEFAULT_NAME])[0]).strip()[:24]
            try:
                size = float((params.get("size") or [str(DEFAULT_SIZE_PCT)])[0].strip("% "))
            except ValueError:
                size = DEFAULT_SIZE_PCT
            task = research_task(name or DEFAULT_NAME, size,
                                 saved if "book" in params else DEFAULT_BOOK)
        if (params.get("format") or [""])[0] == "json":
            self._send(json.dumps(task_as_dict(task), ensure_ascii=False).encode(),
                       "application/json")
            return
        self._send(render_task(task, FAVICON).encode(), "text/html; charset=utf-8")

    def _body(self) -> bytes | None:
        """The request body, at most 64 KB, or ``None`` once a 400 has been sent.

        Every POST route read ``int(Content-Length)`` bare: a header that was not a number raised
        before any route's own error handling, so the client got a dropped connection instead of
        an answer, and a negative one read until the client hung up (found testing the hosted
        entry point, 2026-09-27)."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0:
            self._send(b'{"error": "Content-Length must be a non-negative number"}',
                       "application/json", 400)
            return None
        return self.rfile.read(min(length, 64_000))

    def _fields(self, body: bytes) -> dict[str, list[str]] | None:
        """A POST body as form fields, whether it was sent as a form or as a JSON object, or
        ``None`` once a 400 has been sent.

        A JSON body used to be parsed as a form, so ``{"q": "..."}`` answered "empty question".
        A value that is not a string (``turns`` sent as a list) is passed on as its JSON text,
        which is what the form carries for it."""
        text = body.decode("utf-8", errors="replace")
        kind = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if kind != "application/json":
            return parse_qs(text, keep_blank_values=True)
        try:
            blob = json.loads(text or "{}")
        except ValueError:
            blob = None
        if not isinstance(blob, dict):
            self._send(b'{"error": "a JSON body must be an object, with the question as q"}',
                       "application/json", 400)
            return None
        return {str(k): [v if isinstance(v, str) else json.dumps(v)] for k, v in blob.items()}

    def do_POST(self) -> None:
        """``/mcp``, the Model Context Protocol (`lui/mcp_server.py`); ``/telegram``, the bot's
        webhook (`lui/telegram_bot.py`, rejected without Telegram's secret header); and the page's
        own ``/ask``, ``/translate`` and ``/feedback``, sent as a form body so a visitor's words
        stay out of the access log. None of them writes anything but a usage line."""
        from urllib.parse import urlparse

        path = urlparse(self.path).path.rstrip("/")
        if path == "/research":
            # The task's own form posts here, so a trader's book and question stay out of the URL
            # and the access log; a GET with the same fields still works for a shared link.
            body = self._body()
            if body is None:
                return
            try:
                fields = parse_qs(urlparse(self.path).query)  # ?format=json on the action URL
                form = self._fields(body)
                if form is None:
                    return
                fields.update(form)
                self._research_route(fields)
            except Exception as exc:
                self._send(server_error(exc, path), "application/json", 500)
            return
        if path in ("/ask", "/translate", "/feedback"):
            body = self._body()
            if body is None:
                return
            try:
                form = self._fields(body)
                if form is None:
                    return
                self._answer_route(path, form)
            except Exception as exc:  # the same honest 500 the GET routes give
                self._send(server_error(exc, path), "application/json", 500)
            return
        if path == "/telegram":
            from argus.lui.telegram_bot import handle_webhook

            raw = self._body()
            if raw is None:
                return
            status, body = handle_webhook(
                raw, self.headers.get("X-Telegram-Bot-Api-Secret-Token"))
            self._send(body, "application/json", status)
            return
        if path != "/mcp":
            self._send(b'{"error": "POST is accepted only at /mcp, /telegram, /ask, /translate, '
                       b'/feedback and /research"}', "application/json", 405)
            return
        from argus.lui.mcp_server import handle_body

        # MCP 2025-11-25 (basic/transports.mdx:78-80): a server MUST validate the
        # Origin header and answer 403 to an invalid one, or a web page the visitor opens could
        # drive the endpoint by DNS rebinding. A client that sends no Origin (every non-browser
        # MCP client) is not a browser page and is served; a browser page is served only from
        # this host.
        origin = (self.headers.get("Origin") or "").strip()
        if origin and urlparse(origin).netloc.lower() != (self.headers.get("Host") or "").lower():
            self._send(b'{"jsonrpc": "2.0", "id": null, "error": {"code": -32600, '
                       b'"message": "Origin not allowed"}}', "application/json", 403)
            return
        raw = self._body()
        if raw is None:
            return
        status, body = handle_body(raw)
        self._send(body, "application/json", status)

    def do_GET(self) -> None:
        route = urlparse(self.path)
        path = route.path.rstrip("/") or "/"
        try:
            if path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
                return
            if path == "/mcp":
                # The MCP spec lets a server that offers no server-sent stream answer GET with 405;
                # the body says how to use the endpoint. A browser gets that as a page (/materials
                # links here, and a judge who followed it met a JSON error, audit 2026-09-26).
                if self._wants_html():
                    # A 200 for the person: an MCP client asks for text/event-stream, never
                    # text/html, so this branch is never what the spec's 405 is addressed to, and
                    # a 405 page load is logged as an error in the visitor's console.
                    self._send(mcp_page(self.headers.get("Host") or "").encode(),
                               "text/html; charset=utf-8")
                    return
                self._send(json.dumps({
                    "endpoint": "ARGUS research desk — Model Context Protocol (Streamable HTTP)",
                    "use": "POST JSON-RPC 2.0 here: initialize, tools/list, tools/call",
                    "example": {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                }).encode(), "application/json", 405)
                return
            if path == "/echo":
                # A diagnostic, and a deliberate one. Chinese questions classified correctly in
                # process and were refused on the hosted runtime, and two plausible explanations
                # for that were wrong. Guessing a third time is worse than asking the platform
                # what it received, so this route reports exactly that: the raw path as the
                # handler sees it, the parsed value, and its codepoints.
                #
                # It is safe to leave in place. It reads nothing, writes nothing, and returns only
                # the caller's own input — there is no state behind it to expose.
                params = parse_qs(route.query)
                raw = (params.get("q") or [""])[0]
                self._send(json.dumps({
                    "raw_path": self.path,
                    "parsed": raw,
                    "codepoints": [hex(ord(c)) for c in raw[:24]],
                    "repaired": repair_mojibake(raw),
                    "is_ascii": raw.isascii(),
                }, ensure_ascii=False).encode(), "application/json")
                return
            if path == "/status":
                # JSON for a client, a page for a person: a browser asks for text/html, and the
                # raw JSON it used to get was the one unfinished-looking screen a judge met.
                params = parse_qs(route.query)
                wants_page = "text/html" in (self.headers.get("Accept") or "")
                if (params.get("format") or [""])[0] == "json" or not wants_page:
                    self._send(json.dumps(_status()).encode(), "application/json")
                    return
                from argus.lui.status_page import live_checks, sweep_lines
                from argus.lui.status_page import render as render_status

                checks, checked_at = live_checks()
                from argus.lui.status_history import read as read_history

                page = render_status(_status(), checks, checked_at,
                                     sweep_lines(_ledger_path().parent), FAVICON,
                                     read_history(_ledger_path().parent / "status_history.jsonl"))
                self._send(page.encode(), "text/html; charset=utf-8")
                return
            if path == "/factors":
                # The factor lab's verdicts, drawn: 8 proposed, each gate named, failures shown
                # (harvest study 36).
                from argus.lui.factors_page import load as load_factors
                from argus.lui.factors_page import render as render_factors

                self._send(render_factors(load_factors(_ledger_path().parent)).encode(),
                           "text/html; charset=utf-8")
                return
            if path == "/policy":
                # The risk limits as the file that sets them, and which could fire in the last
                # cycle (harvest study 26).
                from argus.lui.policy_page import load as load_policy
                from argus.lui.policy_page import render as render_policy

                self._send(render_policy(*load_policy(_ledger_path().parent)).encode(),
                           "text/html; charset=utf-8")
                return
            if path == "/architecture":
                # The structure `eval/architecture.py` proves on every push, drawn: Track 2 judges
                # "Agent architecture quality", and the report had no page (harvest study 57).
                from argus.lui.architecture_page import load as load_architecture
                from argus.lui.architecture_page import render as render_architecture

                self._send(render_architecture(load_architecture(_ledger_path().parent)).encode(),
                           "text/html; charset=utf-8")
                return
            if path == "/wrong":
                # **The losses, reachable.** Bitget's own S1 showcase led with a negative result;
                # ours were scattered across six artefacts a reader had to know to look for.
                # Assembled at request time from those artefacts rather than written down once,
                # because a hand-written description of a measurement drifts the first time the
                # measurement changes.
                from argus.lui.corrections import collect
                from argus.lui.corrections_page import render as render_wrong

                found = collect(_ledger_path().parent)
                if (parse_qs(route.query).get("format") or [""])[0] == "json":
                    self._send(
                        json.dumps([c.as_dict() for c in found], ensure_ascii=False).encode(),
                        "application/json",
                    )
                    return
                self._send(render_wrong(found).encode(), "text/html; charset=utf-8")
                return
            if path == "/agent":
                # The Track 2 agent's run, read from its own hourly public record — never
                # recomputed here, so this page cannot disagree with the record it shows.
                from argus.lui.agent_page import render as render_agent

                self._send(render_agent().encode(), "text/html; charset=utf-8")
                return
            if path == "/brand":
                from argus.lui.brand_page import render as render_brand

                self._send(render_brand().encode(), "text/html; charset=utf-8")
                return
            if path == "/materials":
                # Every deliverable on one page, for the form's single materials field; its
                # figures are read from the same artefacts /proof and /wrong render.
                from argus.lui.materials_page import collect as collect_materials
                from argus.lui.materials_page import render as render_materials

                # full URLs, so a judge can copy one straight into a message
                host = self.headers.get("Host") or ""
                proto = self.headers.get("X-Forwarded-Proto") or "http"
                from argus.lui.mcp_server import TOOLS

                items = collect_materials(_ledger_path().parent,
                                          f"{proto}://{host}" if host else "",
                                          tools=[str(t["name"]) for t in TOOLS])
                if (parse_qs(route.query).get("format") or [""])[0] == "json":
                    self._send(json.dumps([i.as_dict() for i in items],
                                          ensure_ascii=False).encode(), "application/json")
                    return
                self._send(render_materials(items).encode(), "text/html; charset=utf-8")
                return
            if path == "/proof":
                # **The wins, reachable.** Every comparison against a named rival, grouped by the
                # sub-theme Bitget names, each with the console question that runs it — read from
                # the standing register at request time, like `/wrong` is from its artefacts.
                from argus.lui.proof_page import collect as collect_wins
                from argus.lui.proof_page import render as render_proof

                wins, counts = collect_wins(_ledger_path().parent)
                if (parse_qs(route.query).get("format") or [""])[0] == "json":
                    self._send(json.dumps({"by_state": counts,
                                           "capabilities": [w.as_dict() for w in wins]},
                                          ensure_ascii=False).encode(), "application/json")
                    return
                self._send(render_proof(wins, counts).encode(), "text/html; charset=utf-8")
                return
            if path == "/research":
                self._research_route(parse_qs(route.query))
                return
            if path == "/og.png":
                self._send(OG_PNG.read_bytes(), "image/png")
                return
            if path in ("/ask", "/translate", "/feedback"):
                self._answer_route(path, parse_qs(route.query))
                return
            if self._wants_html():
                self._send(error_page(404, path).encode(), "text/html; charset=utf-8", 404)
                return
            self._send(b'{"error":"not found"}', "application/json", 404)
        except Exception as exc:  # a demo that 500s silently is worse than one that says why
            if self._wants_html():
                self._send(error_page(500, path, type(exc).__name__).encode(),
                           "text/html; charset=utf-8", 500)
                return
            self._send(server_error(exc, path), "application/json", 500)

    def _wants_html(self) -> bool:
        """A browser asks for text/html; an API client, curl or the MCP client does not, and keeps
        getting JSON it can parse."""
        return "text/html" in (self.headers.get("Accept") or "")

    def log_message(self, fmt: str, *args: Any) -> None:
        """Quiet by default; the console is the output, not the access log."""


def mcp_page(host: str) -> str:
    """What /mcp is, for a person who opened it in a browser: the tools and how to connect."""
    import html as _html

    from argus.lui.mcp_server import TOOLS

    base = f"https://{host}" if host and not host.startswith(("127.", "localhost")) else (
        f"http://{host}" if host else design.PUBLIC_URL)
    endpoint = _html.escape(f"{base}/mcp")
    def first_sentence(text: str) -> str:
        return text.split(". ")[0].rstrip(".") + "."

    tools = "".join(f"<li><code>{_html.escape(str(t['name']))}</code> — "
                    f"{_html.escape(first_sentence(str(t['description'])))}</li>" for t in TOOLS)
    title = "ARGUS over the Model Context Protocol"
    return f"""<!doctype html><html lang="en"><head>{design.head("ARGUS — MCP", title, "/mcp")}
<style>{design.TOKENS_CSS}{design.BASE_CSS}
 .wrap {{ max-width:760px; margin:0 auto; padding:48px 18px 80px }}
 h1 {{ font-size:26px; margin:6px 0 10px }}
 pre {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
   padding:12px 14px; font:12.5px/1.6 var(--mono); overflow-x:auto }}
 @media (max-width: 640px) {{ pre {{ white-space:pre-wrap; overflow-wrap:anywhere }} }}
 li {{ margin:6px 0 }}
</style></head><body>{design.nav("")}<div class="wrap">
<h1>{title}</h1>
<p>This address is the desk's MCP endpoint (Streamable HTTP, JSON-RPC 2.0 over POST). A browser
cannot use it; an MCP client can. Point one at <code>{endpoint}</code> — no key is needed — and it
can call:</p>
<ul>{tools}</ul>
<p>Or from a terminal:</p>
<pre>curl -s {endpoint} -H 'Content-Type: application/json' \\
  -d '{{"jsonrpc":"2.0","id":1,"method":"tools/list"}}'</pre>
<p>Every tool answers read-only; an order sent through it is refused like anywhere else.
The same answers are on the <a href="/">console</a>.</p>
</div>{design.footer()}</body></html>"""


def error_page(status: int, path: str, failure: str = "") -> str:
    """The page a browser gets for a path that does not exist or a page that failed to build.

    Until 2026-09-26 both were bare JSON (``{"error":"not found"}``) with no navigation, the one
    screen a judge could reach that looked unfinished (judge audit). The JSON is kept for clients
    that do not ask for HTML."""
    import html as _html

    shown = _html.escape(path[:120])
    if status == 404:
        title = "No page here"
        lead = (f"There is no page at <code>{shown}</code>. Every page ARGUS serves is in the bar "
                f"above; the three most people want are below.")
    else:
        title = "This page failed to build"
        lead = (f"<code>{shown}</code> failed while it was being built ({_html.escape(failure)}). "
                f"The console and the other pages still work; asking again usually does.")
    return f"""<!doctype html><html lang="en"><head>{design.head(f"ARGUS — {title}", title)}
<style>{design.TOKENS_CSS}{design.BASE_CSS}
 .wrap {{ max-width:720px; margin:0 auto; padding:48px 18px 80px }}
 .code {{ font:600 13px var(--mono); color:var(--dim); letter-spacing:.08em }}
 h1 {{ font-size:26px; margin:6px 0 10px }}
 .go {{ display:flex; gap:12px 24px; flex-wrap:wrap; margin-top:22px; font-weight:500 }}
 .go a {{ color:var(--ink); border-bottom:1px solid var(--line); text-decoration:none }}
</style></head><body>{design.nav("")}<div class="wrap">
<p class="code">{status}</p><h1>{title}</h1><p>{lead}</p>
<p class="go"><a href="/">Ask the desk &rarr;</a><a href="/research">Run a research task &rarr;</a>
<a href="/materials">Every deliverable on one page &rarr;</a></p>
</div>{design.footer()}</body></html>"""


def serve(host: str = HOST, port: int = PORT) -> None:
    ledger = PaperLedger(path=_ledger_path())
    print(f"ARGUS research console -> http://{host}:{port}")
    print(f"ledger: {len(ledger.entries)} decision(s), chain "
          f"{'intact' if ledger.verify()['chain_intact'] else 'BROKEN'}")
    print("Ctrl-C to stop.")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Serve the ARGUS research console.")
    parser.add_argument("--host", default=HOST, help="bind address (default: loopback only)")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args(argv)
    try:
        serve(args.host, args.port)
    except KeyboardInterrupt:
        print()
    return 0


__all__ = ["HOST", "PAGE", "PORT", "Handler", "handle_ask", "main", "serve"]


if __name__ == "__main__":
    raise SystemExit(main())
