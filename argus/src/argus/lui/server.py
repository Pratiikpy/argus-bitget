"""The demo surface — the desk console over HTTP.

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

import json
import os
import re
from dataclasses import replace
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from argus.lui import design
from argus.lui.answer import Answer, answer
from argus.lui.cli import BUDGET_MS
from argus.lui.kindmodel import LocalPlanner, kind_model
from argus.lui.ngram import reclassify
from argus.lui.question import TRADED_SYMBOLS, Conversation, Intent, classify
from argus.lui.research import (
    _PRICE_FORECAST,
    ResearchKind,
    ResearchRequest,
    about_the_record,
    follow_up,
    pattern_reading_wins,
    plan_with_model,
    research_symbols,
    with_book,
    worth_asking_the_model,
)
from argus.lui.research import detect as detect_research
from argus.lui.research import run as run_research
from argus.lui.router import Router, build_router, route
from argus.paper.ledger import PaperLedger

HOST = "127.0.0.1"
PORT = 8765

FAVICON = design.favicon()
"""An inline SVG, not a bundled file. Every route in this server is one string in one module —
adding a binary asset and a route to serve it would be the first exception to that, for a mark a
judge never consciously looks at. Its absence was not silent, though: an unstyled 404 for
`/favicon.ico` was the one console error on an otherwise-clean page, found driving the console as
a first-time judge would (2026-09-22) rather than by reading the code and assuming it was fine."""

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="__FAVICON__">__FONTS__
<title>ARGUS desk console</title>
<style>__TOKENS__
  body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,sans-serif; }
  .wrap { max-width:900px; margin:0 auto; padding:28px 18px 64px; }
  h1 { font-size:20px; margin:0 0 4px; letter-spacing:-0.01em }
  .sub { color:var(--dim); font-size:13px; margin:0 0 20px }
  .bar { display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap }
  input[type=text] { flex:1 1 320px; min-width:0; padding:11px 13px; border:1px solid var(--line);
    border-radius:8px; background:var(--panel); color:var(--ink); font-size:15px }
  button { padding:11px 18px; border:1px solid var(--accent); background:var(--accent); color:#fff;
    border-radius:8px; font-size:15px; cursor:pointer }
  button:disabled { opacity:.55; cursor:default }
  .chips { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:22px }
  .chip { border:1px solid var(--line); background:var(--panel); color:var(--dim);
    border-radius:999px; padding:5px 11px; font-size:12.5px; cursor:pointer }
  .chip:hover { color:var(--ink); border-color:var(--accent) }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
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
  .group { font-size:11.5px; letter-spacing:.06em; text-transform:uppercase; color:var(--dim);
    margin:0 0 6px }
  .line.act { font-weight:600; color:var(--accent) }
  .line.hedge { font-weight:600 }
  .line.fine { color:var(--dim); font-size:13px }
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
<a href="/status">What the desk can see &rarr;</a></p>

<form class="bar" id="f">
  <input type="text" id="q" autocomplete="off"
    placeholder="I hold 50% NVDA, 50% AAPL — what does adding 20% TSLA do to my risk?">
  <button id="go">Ask</button>
</form>
<div class="book">
  <label for="book">My book</label>
  <input type="text" id="book" autocomplete="off"
    placeholder="optional, e.g. 40% NVDA, 30% MSFT, 30% AAPL, risk budget 20%"
    title="Used by every research question that does not name its own holdings">
  <span class="saved" id="saved"></span>
</div>
<p class="group">Research</p>
<div class="chips" id="chips-research"></div>
<p class="group">The desk's own record</p>
<div class="chips" id="chips"></div>
<div id="out"><p class="empty">Ask something, or pick one of the suggestions above.</p></div>
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
const RESEARCH = ["I hold 50% NVDA, 50% AAPL — what does adding 20% TSLA do to my risk?",
  "what if the Nasdaq drops 10%? I hold 40% MSFT, 30% META, 30% GOOGL",
  "is TSLA riskier than NVDA", "should I buy MSTR", "where is NVDA trading right now",
  "how should I split a $50k order in NVDA", "is TSLA overbought",
  "when does NVDA report earnings", "has COIN been here before", "compare gold and bitcoin"];
const SUGGEST = ["why did you do nothing all weekend","what is the sharpe","show me decision 25",
  "what did the risk layer block","what evidence backed that","is the log tamper-evident",
  "are you well calibrated","what bad decision patterns do you have","what is my position",
  "sell half of that"];
const out = document.getElementById('out'), qEl = document.getElementById('q');
const bookEl = document.getElementById('book'), savedEl = document.getElementById('saved');
let turns = [], first = true;

// The book is the visitor's own and stays in their browser; it is sent with each question and
// never stored on the server.
try { bookEl.value = localStorage.getItem('argus.book') || ''; } catch (e) {}
bookEl.addEventListener('change', () => {
  try { localStorage.setItem('argus.book', bookEl.value.trim()); } catch (e) {}
  savedEl.textContent = bookEl.value.trim() ? 'saved in this browser' : '';
});

for (const [id, list] of [['chips-research', RESEARCH], ['chips', SUGGEST]]) {
  const el = document.getElementById(id);
  el.innerHTML = list.map(s => `<span class="chip">${esc0(s)}</span>`).join('');
  el.addEventListener('click', e => {
    if (!e.target.classList.contains('chip')) return;
    qEl.value = e.target.textContent; document.getElementById('f').requestSubmit();
  });
}
function esc0(s) {
  return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}
const lineClass = l => l.startsWith('Actionable:') ? 'line act'
  : l.startsWith('Hedge:') ? 'line hedge'
  : (l.startsWith('Assumed:') || l.startsWith('Data:')) ? 'line fine' : 'line';

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
    age = ` Last decision ${s.age_hours}h ago; the desk decides four times a day during US ` +
      `market hours` + (until !== null ? `, next in ${until}h.` : '.');
  }
  document.getElementById('stat').textContent =
    `${s.entries} decisions on record, chain ${s.chain_intact ? 'intact' : 'BROKEN'}.` + age;
}).catch(() => {});

const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

document.getElementById('f').addEventListener('submit', async ev => {
  ev.preventDefault();
  const text = qEl.value.trim(); if (!text) return;
  qEl.value = ''; document.getElementById('go').disabled = true;
  if (first) { out.innerHTML = ''; first = false; }
  try {
    const r = await fetch('ask?' + new URLSearchParams(
      {q: text, turns: JSON.stringify(turns), book: bookEl.value.trim()}));
    const a = await r.json();
    turns = a.turns || turns;
    const over = a.elapsed_ms > a.budget_ms;
    out.insertAdjacentHTML('afterbegin', `
      <div class="card ${a.refused ? 'refused' : ''}">
        <div class="q">${esc(text)}</div>
        <div class="meta">
          <span class="tag">${esc(a.intent)}</span>
          <span class="tag ${over ? 'over' : ''}">${a.elapsed_ms.toFixed(0)}ms /
            ${a.budget_ms}ms${over ? ' OVER BUDGET' : ''}</span>
          ${a.refused ? '<span class="tag over">refused</span>' : ''}
        </div>
        <div class="lines">${a.lines.map(l =>
          `<div class="${lineClass(l)}">${esc(l)}</div>`).join('')}</div>
        ${a.sources.length ? `<div class="src"><span class="rk">Receipt · ${a.sources.length}` +
          ` source${a.sources.length === 1 ? '' : 's'}</span>` +
          a.sources.map(s => `&nbsp;&nbsp;<b>${esc(s.kind)}</b>:${esc(s.ref)}` +
            (s.detail ? ' — ' + esc(s.detail) : '')).join('<br>') + `</div>` : ''}
      </div>`);
    if (a.translate) {
      // English first, then the reader's language: every figure in the translation was checked
      // against the English by the server before it was sent (`lui/translate.py`).
      const card = out.firstElementChild;
      const body = a.lines.slice(a.translate.skip);
      fetch('translate?' + new URLSearchParams({lang: a.translate.lang,
        token: a.translate.token, lines: JSON.stringify(body)}))
        .then(r => r.ok ? r.json() : null).then(t => {
          if (!t || !t.lines) return;
          // Each translated line keeps the styling of the English line it came from.
          const note = t.note ? [t.note] : a.lines.slice(0, a.translate.skip);
          card.querySelector('.lines').innerHTML =
            note.map(l => `<div class="line fine">${esc(l)}</div>`).join('') +
            t.lines.map((l, i) => `<div class="${lineClass(body[i])}">${esc(l)}</div>`).join('');
        }).catch(() => {});
    }
  } catch (e) {
    out.insertAdjacentHTML('afterbegin',
      `<div class="card refused"><div class="q">${esc(text)}</div>
       <div class="line">The console could not reach the desk: ${esc(e.message)}</div></div>`);
  } finally {
    document.getElementById('go').disabled = false; qEl.focus();
  }
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
for _token, _value in (("__FAVICON__", FAVICON), ("__TOKENS__", design.TOKENS_CSS),
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


def handle_ask(
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
    from argus.lui.research import (
        _SESSION_CLAIM,
        SESSION_QUESTION,
        _session_claim_line,
        session_status,
    )

    if SESSION_QUESTION.search(text) and not research_symbols(text)[0]:
        # "Is the US market open right now?" names no instrument; it is answered from the
        # session clock, not routed to an engine that needs one. Stated as a claim ("the US
        # market is open right now") it gets the premise verdict first.
        q = classify(text, now=clock, conversation=conversation)
        lines, sources = session_status(clock)
        claimed = _SESSION_CLAIM.search(text) if not text.rstrip().endswith("?") else None
        if claimed is not None:
            lines = [_session_claim_line(claimed), *lines]
        note = _language_note(text)
        if note:
            lines = [note, *lines]
        payload = Answer(question=q, lines=lines, sources=sources).as_dict()
        payload.update(elapsed_ms=(time.perf_counter() - started) * 1000,
                       budget_ms=BUDGET_MS[q.speed], routing=audit, classified_by="session-clock",
                       matched=SESSION_QUESTION.pattern, turns=[*prior, text][-12:])
        return payload
    followed = follow_up(text, prior, book)
    if followed is not None:
        return _research_payload(text, prior, followed, ledger, started, "research-follow-up",
                                 {**audit, "detail": "a follow-up to the previous question"})
    model = _model_for(visitor) if worth_asking_the_model(text, now=clock) else None
    instruction = classify(text, now=clock, conversation=conversation).intent is Intent.ORDER
    if model is None and not instruction:
        # **No language model: the trained kind model reads the question instead.** It fills the
        # same slot and goes through the same validation (`plan_with_model`), deciding only which
        # engine answers; names, weights and the shock are read from the text by the pattern
        # extractors. On the 2026-09-25 held-out set (200 questions, a writer who never saw this
        # repository, twelve languages) it read 88.5% correctly where the patterns alone read
        # 63.5% (`eval/kindtrain.py`). It costs no token, so every question is shown to it.
        local = kind_model()
        model = LocalPlanner(local) if local is not None else None
    if model is not None:
        planned, audit = plan_with_model(text, model)
        if (planned is None and not isinstance(model, LocalPlanner)
                and str(audit.get("detail", "")).startswith("planner unavailable")):
            # **A failed language-model call falls back to the kind model, not to the patterns.**
            # Scoring the blind set with Qwen reading first (2026-09-25) read 58.8% against the kind
            # model's 81.7%: under load some Qwen calls errored, and each error dropped the question
            # to the patterns alone. On the live site that is every question asked while the model
            # is rate-limited or down.
            local = kind_model()
            if local is not None:
                planned, local_audit = plan_with_model(text, LocalPlanner(local))
                audit = {**local_audit, "fallback_from": audit.get("detail")}
        planned = with_book(planned, book, text)
        patterned = detect_research(text)
        # A spot rToken holding is a field the model's plan does not carry, so a model reading of
        # the same kind still loses it: "I hold RNVDAUSDT, protect it over the weekend" came back
        # as a hedge of nothing and was refused on the live console (2026-09-24).
        if (planned is not None and planned.kind is ResearchKind.LEVERAGE
                and not _LEVERAGE_WORDS.search(text)):
            # The hosted model read "Long MSTR perp into earnings — funding looks cheap" as a 10x
            # leverage question (live, 2026-09-25): "perp" is not leverage. With no multiple,
            # margin or liquidation named, the leverage engine has nothing it was asked.
            planned = with_book(patterned, book, text) if patterned is not None else None
            audit = {**audit, "detail": "a leverage reading with no leverage named was dropped"}
        if (planned is not None and patterned is not None
                and planned.kind is ResearchKind.IMPACT and patterned.kind is ResearchKind.IMPACT
                and planned.symbols and patterned.symbols
                and planned.symbols[0] in planned.book
                and patterned.symbols[0] not in planned.book):
            # The hosted model read "nvda 40%, msft 20%, cash rest — want to add 5k of coin" as
            # adding bitcoin, named no candidate it could resolve, and the plan fell back to a
            # name already held — so the answer was about adding NVDA (live, 2026-09-25). When the
            # model's add is a holding and the patterns found a name outside the book, theirs is
            # the add that was asked about.
            planned = with_book(patterned, book, text)
            audit = {**audit, "detail": "the model's add was a holding; the patterns' add kept"}
        if (patterned is not None and pattern_reading_wins(patterned, text)
                and (planned is None or planned.kind is not patterned.kind
                     or (patterned.spot is not None and planned.spot != patterned.spot)
                     or planned.horizon_hours != patterned.horizon_hours
                     or planned.target != patterned.target
                     or planned.resize_by != patterned.resize_by
                     or (planned.kind is ResearchKind.IMPACT
                         and planned.symbols[:1] != patterned.symbols[:1]))):
            # The model read "order book depth on NVDA" as a quote and "who is selling NVDA" as
            # a news question (2026-09-24). Where the patterns name the one engine that answers
            # the question, their reading stands.
            planned = with_book(patterned, book, text)
            audit = {**audit, "detail": "the patterns' specific reading kept over the model's"}
        if planned is not None:
            return _research_payload(text, prior, planned, ledger, started, "research-model",
                                     audit)
    # **The kind model's "not research" is binding on the research patterns.** Told a question is
    # off-topic, a trade instruction or a price forecast ("what will gold be a year from now"),
    # the patterns still found a price word and quoted it; told it is about the desk's record
    # ("the rationale logged for skipping SPY"), they found a ticker and sized a position. Where
    # the patterns name the one engine that answers exactly (`pattern_reading_wins`) they still
    # stand, as they do over the language model.
    kind_said = _not_research(audit)
    patterned_only = detect_research(text)
    if kind_said is not None and (patterned_only is None
                                  or kind_said[1] >= BINDING_KIND_CONFIDENCE):
        # Binding only when the patterns found nothing, or when the model is sure: "could you
        # tell me the current price of silver" is refused by the model at 0.19 — it has learnt
        # that price questions are often forecasts — and the patterns' quote is the right answer.
        request = (with_book(patterned_only, book, text)
                   if pattern_reading_wins(patterned_only, text) else None)
    else:
        kind_said = None
        request = with_book(patterned_only, book, text)
    if request is None and _PRICE_FORECAST.search(text) and research_symbols(text)[0]:
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
        payload["matched"] = _PRICE_FORECAST.pattern
        payload["turns"] = [*prior, text][-12:]
        return payload
    is_order = classify(text, now=clock, conversation=conversation).intent is Intent.ORDER
    if request is None and not about_the_record(text) and not is_order and kind_said is None:
        # A question that names a listed contract the desk does not trade, in words no research
        # kind recognises ("give me a thesis on Solana for a conservative investor"), used to fall
        # through to the ledger and come back as the latest decision on an unrelated rToken. The
        # ledger has nothing on such a name; its risk profile is the honest answer, said as such.
        named = [s for s in research_symbols(text)[0] if s not in TRADED_SYMBOLS]
        if named:
            request = ResearchRequest(
                kind=ResearchKind.IMPACT, symbols=(named[0],),
                notes=("no specific research question was recognised, so this is the name's risk "
                       "profile — ask for its technicals, news, earnings or what it does to your "
                       "book for more",))
    if request is not None:
        audit = {**audit, "detail": "recognised by the research patterns"}
        return _research_payload(text, prior, request, ledger, started, "research-patterns",
                                 audit)

    question = classify(text, now=clock, conversation=conversation)
    # **The n-gram model sits between the patterns and the router, and it is what a judge meets.**
    # The hosted console deploys no model key, so `route()` below returns untouched and the
    # patterns alone used to be the whole console — measured at 23.9% correct on 351 questions
    # written by a model that has never seen this repository, with 60 confident errors. With this
    # step the same corpus reads 80.9% correct and 35 errors. It costs one JSON file and no
    # dependency; see `eval/ngrambench.py`.
    question, classified_by = reclassify(question)
    if (classified_by == "ngram" or (classified_by == "patterns" and not prior)) and not in_domain(
            text):
        # **The n-gram layer only knows wording, so it needs a topic gate.** It answers with a
        # class for any text at all, and on the 2026-09-25 blind corpus "who won the lakers game
        # last night" reached `decision_why` at 0.19 and was answered with a decision, as were a
        # CV review and "explain quantum computing" (which the hand-written patterns claimed on
        # the one word "explain"). A question with no market, trading or desk word in it is not a
        # question about the record. A follow-up ("explain that") is exempt: it inherits the topic
        # of the turn before it.
        question = replace(question, intent=Intent.UNKNOWN,
                           reason="nothing in the question is about markets or the desk's record")
        classified_by = "declined-off-topic"
    # **A confident "unrelated" from the model overrules the n-gram layer's guess.** The n-gram
    # layer has no notion of topic, only of wording, and "tell me a joke about NVDA" reached
    # `decision_why` and was answered with a decision's thesis. The planner above already read the
    # question; when it said, with confidence, that the question is neither research nor about the
    # desk's record, the n-gram guess is withdrawn and the question goes to the refusal. The
    # hand-written patterns are never overruled this way — only the statistical layer is.
    model_view = (audit.get("model") or {}) if isinstance(audit, dict) else {}
    try:
        model_confidence = float(model_view.get("confidence") or 0.0)
    except (TypeError, ValueError):
        model_confidence = 0.0
    if (
        classified_by == "ngram"
        and str(model_view.get("kind", "")).lower() == "none"
        and model_confidence >= UNRELATED_CONFIDENCE
    ):
        question = replace(question, intent=Intent.UNKNOWN,
                           reason="the model read this as unrelated to research or the record")
        classified_by = "declined-by-model"
    question, routing = route(
        question, client=_model_for(visitor), now=clock, conversation=conversation
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


BINDING_KIND_CONFIDENCE = 0.25
"""The kind model's "refuse" or "record" overrules a research reading the patterns found only at
or above this confidence. Below it, a pattern reading stands."""


def _kind_verdict(why: str) -> tuple[str, float] | None:
    """("refuse" | "record", confidence) from the local planner's audit line, else None."""
    match = re.match(r"kind model: (refuse|record) at ([0-9.]+)", why)
    return None if match is None else (match.group(1), float(match.group(2)))


def _not_research(audit: Any) -> tuple[str, float] | None:
    """Either reader's confident verdict that a question is not a research question.

    The kind model says so in its audit line; the language model says so as ``kind: none`` or
    ``kind: record`` with a confidence. Found on the live console (2026-09-25): Qwen read "what will
    gold price be exactly one year from now" as a price forecast at 0.95 — correctly — and the
    name-only fallback still answered it as a risk profile of gold."""
    view = (audit.get("model") or {}) if isinstance(audit, dict) else {}
    local = _kind_verdict(str(view.get("why") or ""))
    if local is not None:
        return local
    kind = str(view.get("kind") or "").strip().lower()
    try:
        confidence = float(view.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if kind in ("none", "record") and confidence >= UNRELATED_CONFIDENCE:
        return ("refuse" if kind == "none" else "record"), confidence
    return None


_DOMAIN = re.compile(
    r"\b(?:trad\w*|decision\w*|decid\w*|desk|positions?|holding\w*|risk\w*|sharpe|sortino|"
    r"drawdown|pnl|p&l|profit\w*|loss\w*|lose|lost|losing|money|returns?|orders?|fills?|filled|"
    r"log|ledger|record|calibrat\w*|confiden\w*|abstain\w*|abstention|pass(?:ed|es)?|skip\w*|"
    r"nothing|evidence|sources?|thesis|signals?|strateg\w*|model|hash\w*|tamper\w*|anchor\w*|"
    r"block\w*|kernel|guard\w*|weekend|sessions?|hours|market\w*|prices?|stocks?|shares?|"
    r"crypto\w*|coins?|bitcoin|hedg\w*|portfolio|book|fees?|costs?|win\s+rate|exposure|"
    r"leverage|long|short|buy\w*|sell\w*|bought|sold|calls?|bets?|accura\w*|wrong|right|"
    r"perform\w*|history|past|latest|recent|last\s+(?:call|decision|trade|week|month)|why|"
    r"rtokens?|perp\w*|futures|equit\w*|index|nasdaq|volatil\w*|beta|you|your|yours)\b|"
    r"交易|决策|仓位|持仓|风险|收益|盈亏|亏损|盈利|订单|市场|价格|策略|对冲|股票|币|夏普|回撤|记录|"
    r"为什么|理由|证据|校准|你", re.I)
"""Words that make a question about markets or the desk. Deliberately wide — it exists to stop
the n-gram layer answering chit-chat, not to judge a trading question."""


_LEVERAGE_WORDS = re.compile(r"\b\d+(?:\.\d+)?\s*x\b|\bleverag\w*|\bliquidat\w*|\bmargin\b|"
                             r"\u6760\u6746|\u7206\u4ed3|\u500d", re.I)
"""What makes a question about leverage: a multiple, the word, liquidation or margin, or the
Chinese for leverage, liquidation and "times"."""


def in_domain(text: str) -> bool:
    if research_symbols(text)[0]:
        return True
    hit = _DOMAIN.search(text)
    if hit is None:
        return False
    # "you" alone is not a desk word: "can you review my resume" is not about the record. It
    # counts only with a second domain word or when the question is addressed to the desk's
    # conduct ("why did you ...", "what did you ...").
    words = {m.group(0).lower() for m in _DOMAIN.finditer(text)}
    if words <= {"you", "your", "yours", "你"}:
        return bool(re.search(r"\b(?:why|what|when|how)\s+(?:did|do|have|were)\s+you\b", text,
                              re.I))
    return True


def _research_payload(
    text: str, prior: list[str], request: Any, ledger: PaperLedger, started: float,
    classified_by: str, audit: dict[str, Any],
) -> dict[str, Any]:
    """One research answer in the same envelope every other answer uses, so the page and any
    client reading ``/ask`` need no second code path."""
    import time

    result = run_research(text, request, ledger=ledger)
    payload = result.as_dict()
    note = _language_note(text)
    if note:
        payload["lines"] = [note, *payload.get("lines", [])]
    payload["elapsed_ms"] = (time.perf_counter() - started) * 1000
    payload["budget_ms"] = BUDGET_MS[result.question.speed]
    payload["routing"] = audit
    payload["classified_by"] = classified_by
    payload["matched"] = result.question.matched
    payload["turns"] = [*prior, text][-12:]
    return payload


_SPANISH = re.compile(r"[¿¡]|\b(?:qué|cómo|como|cuál|cuánto|los|las|del|tasas|oro|acciones|"
                      r"comprar|vender|debería|precio|pasa|está|baja|sube|cartera|riesgo)\b",
                      re.I)
_LATIN_OTHER = re.compile(r"\b(?:quoi|pourquoi|est-ce|wie|warum|welche|ist|wenn|meinem|meine|"
                          r"passiert|devo|preço|ações)\b",
                          re.I)


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
    if _LATIN_OTHER.search(text):
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


UNRELATED_CONFIDENCE = 0.8
"""How sure the planner must be that a question is unrelated before its view overrules the n-gram
layer. High on purpose: withdrawing a guess that was right costs a real answer."""

MODEL_CALLS_PER_VISITOR_PER_HOUR = 30
"""How many model-assisted questions one visitor may ask per hour, per server instance.

**Why there is a limit at all.** The hosted console now carries the hackathon Qwen key in its
server environment — never in the bundle — so that a judge's oddly-phrased question is understood.
That key has a finite balance, and an unauthenticated page that spends it on every request is a page
anyone can drain. Thirty an hour is far above what a person reading answers asks, and far below
what a script would try. Over the limit the console does not fail: it answers from its
deterministic layers alone, exactly as it did before the key was deployed. Per instance because a
serverless platform keeps no shared memory; the process's own :class:`~argus.llm.qwen.TokenBudget`
bounds the total spend of each instance independently of who asked."""

_VISITS: dict[str, list[float]] = {}


def _model_for(visitor: str) -> Router | None:
    """The model client, or None for a visitor who has used their hourly allowance."""
    import time

    now = time.monotonic()
    recent = [t for t in _VISITS.get(visitor, []) if now - t < 3600.0]
    if len(recent) >= MODEL_CALLS_PER_VISITOR_PER_HOUR:
        _VISITS[visitor] = recent
        return None
    recent.append(now)
    _VISITS[visitor] = recent
    return _router()


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
        _ROUTER = build_router()
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
    # the raw one — a real bug, found by a test tripping on it, not by inspection: at a raw age
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
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'unsafe-inline'; "
            "style-src 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; img-src 'self' data:",
        )
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        """Two endpoints only: ``/mcp``, the Model Context Protocol (`lui/mcp_server.py`), and
        ``/telegram``, the bot's webhook (`lui/telegram_bot.py`, rejected without Telegram's secret
        header). Both answer questions; neither writes anything. Every other path answers GET."""
        from urllib.parse import urlparse

        path = urlparse(self.path).path.rstrip("/")
        if path == "/telegram":
            from argus.lui.telegram_bot import handle_webhook

            length = min(int(self.headers.get("Content-Length") or 0), 64_000)
            status, body = handle_webhook(
                self.rfile.read(length), self.headers.get("X-Telegram-Bot-Api-Secret-Token"))
            self._send(body, "application/json", status)
            return
        if path != "/mcp":
            self._send(b'{"error": "POST is accepted only at /mcp and /telegram"}',
                       "application/json", 405)
            return
        from argus.lui.mcp_server import handle_body

        length = min(int(self.headers.get("Content-Length") or 0), 64_000)
        status, body = handle_body(self.rfile.read(length))
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
                # the body says how to use the endpoint, for a person who opens it in a browser.
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
                page = render_status(_status(), checks, checked_at,
                                     sweep_lines(_ledger_path().parent), FAVICON)
                self._send(page.encode(), "text/html; charset=utf-8")
                return
            if path == "/wrong":
                # **The losses, reachable.** Bitget's own S1 showcase led with a negative result;
                # ours were scattered across six artefacts a reader had to know to look for.
                # Assembled at request time from those artefacts rather than written down once,
                # because a hand-written description of a measurement drifts the first time the
                # measurement changes.
                from argus.lui.corrections_page import collect
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
                # **Track 3's required demo: one complete research task, question to actionable
                # insight.** This used to render a chain recorded on 2026-09-14 — a fixed
                # question, raw data structures, and a verdict its own allocation step
                # contradicted. It now runs the task on request: seven engines in parallel, no
                # model call, about three seconds, with the name, size and book taken from the
                # URL so a judge can ask their own version. `?format=json` returns the same run.
                from argus.lui.task import (
                    DEFAULT_BOOK,
                    DEFAULT_NAME,
                    DEFAULT_SIZE_PCT,
                    render_task,
                    research_task,
                )
                from argus.lui.task import as_dict as task_as_dict

                params = parse_qs(route.query)
                name = repair_mojibake((params.get("name") or [DEFAULT_NAME])[0]).strip()[:24]
                book_text = repair_mojibake((params.get("book") or [DEFAULT_BOOK])[0])[:300]
                try:
                    size = float((params.get("size") or [str(DEFAULT_SIZE_PCT)])[0].strip("% "))
                except ValueError:
                    size = DEFAULT_SIZE_PCT
                task = research_task(name or DEFAULT_NAME, size, book_text)
                if (params.get("format") or [""])[0] == "json":
                    self._send(json.dumps(task_as_dict(task), ensure_ascii=False).encode(),
                               "application/json")
                    return
                self._send(render_task(task, FAVICON).encode(), "text/html; charset=utf-8")
                return
            if path == "/ask":
                params = parse_qs(route.query)
                text = repair_mojibake((params.get("q") or [""])[0])
                raw_turns = (params.get("turns") or ["[]"])[0]
                try:
                    prior = [str(t) for t in json.loads(raw_turns)][-12:]
                except (ValueError, TypeError):
                    prior = []
                if not text.strip():
                    self._send(json.dumps({"error": "empty question"}).encode(),
                               "application/json", 400)
                    return
                visitor = (self.headers.get("x-forwarded-for") or "").split(",")[0].strip() \
                    or self.client_address[0]
                book = repair_mojibake((params.get("book") or [""])[0])[:300]
                payload = handle_ask(text, prior, visitor=visitor, book=book)
                offer_translation(payload, text)
                self._send(json.dumps(payload, default=str).encode(), "application/json")
                return
            if path == "/translate":
                # The second half of a non-English answer: the page shows the English at once and
                # asks here for the translation, signed by this server so only its own answers
                # are translated (`lui/translate.py`).
                from argus.lui import translate

                params = parse_qs(route.query)
                lang = (params.get("lang") or [""])[0]
                token = (params.get("token") or [""])[0]
                try:
                    lines = json.loads((params.get("lines") or ["[]"])[0])
                except ValueError:
                    lines = None
                if (not isinstance(lines, list) or not all(isinstance(x, str) for x in lines)
                        or not translate.verify(lang, lines, token)):
                    self._send(b'{"error": "not an answer this console wrote"}',
                               "application/json", 403)
                    return
                visitor = (self.headers.get("x-forwarded-for") or "").split(",")[0].strip() \
                    or self.client_address[0]
                result = translate.translate(lines, lang, _model_for(visitor))
                self._send(json.dumps(result, ensure_ascii=False).encode(), "application/json")
                return
            self._send(b'{"error":"not found"}', "application/json", 404)
        except Exception as exc:  # a demo that 500s silently is worse than one that says why
            self._send(json.dumps({"error": type(exc).__name__, "detail": str(exc)[:200]}).encode(),
                       "application/json", 500)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Quiet by default; the console is the output, not the access log."""


def serve(host: str = HOST, port: int = PORT) -> None:
    ledger = PaperLedger(path=_ledger_path())
    print(f"ARGUS desk console -> http://{host}:{port}")
    print(f"ledger: {len(ledger.entries)} decision(s), chain "
          f"{'intact' if ledger.verify()['chain_intact'] else 'BROKEN'}")
    print("Ctrl-C to stop.")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Serve the ARGUS desk console.")
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
