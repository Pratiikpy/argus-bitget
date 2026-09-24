"""``/brand`` — the ARGUS identity, rendered from the tokens every page uses.

A brand kit that is a separate document drifts from the product the day either changes. This one
is drawn from :mod:`argus.lui.design` at request time: the palette swatches are the same constants
the pages' CSS is built from, so what the kit shows is what the console is.
"""

from __future__ import annotations

from html import escape

from argus.lui import design

VOICE_DO = (
    "Every number is computed. Every source is named. Every loss is published.",
    "Ask the desk. The answer shows its work.",
    "Tied with Bitget's own TWAP. Here is the replay.",
    "Refused: nothing on record supports that figure.",
)
VOICE_DONT = (
    "Revolutionising trading with next-generation AI.",
    "The ultimate all-in-one platform for everyone.",
    "Institutional-grade alpha, unlocked.",
    "Trust us, the model is very accurate.",
)


def _swatches() -> str:
    cells = []
    for name, hexcode, role, use in design.PALETTE:
        dark = name in ("Ink", "Stone", "Proof", "Redline", "Graphite")
        fg = "#FFFFFF" if dark else "#0A0A0A"
        cells.append(
            f'<div class="sw" style="background:{hexcode};color:{fg}"><span>{escape(role)}</span>'
            f'<strong>{escape(name)}</strong><code>{hexcode}</code><small>{escape(use)}</small>'
            f'</div>')
    return "".join(cells)


def _receipt() -> str:
    return ('<div class="rc"><span class="kicker">argus · receipt · sample</span>'
            '<strong>VERIFIED</strong>'
            '<dl><dt>Question</dt><dd>should I add 20% TSLA?</dd><dt>Engine</dt>'
            '<dd>desk.portfolio.copilot</dd><dt>Source</dt><dd>Bitget hourly candles, 30 d</dd>'
            '<dt>Rival</dt><dd>weekend-copilot (S2): tied</dd><dt>Ledger</dt>'
            '<dd>seq 412 · 9f2c…a71e</dd></dl></div>')


def render() -> str:
    do = "".join(f"<li>{escape(x)}</li>" for x in VOICE_DO)
    dont = "".join(f"<li>{escape(x)}</li>" for x in VOICE_DONT)
    return f"""<!doctype html><html lang="en"><head>{design.head(
        "ARGUS brand kit", "The ARGUS identity: mark, palette, type, receipt and voice.")}
<style>{design.TOKENS_CSS}
  .sec {{ display:grid; grid-template-columns:190px minmax(0,1fr); gap:52px; padding:64px 0;
    border-bottom:1px solid var(--line) }}
  .meta {{ font:500 11.5px/1.6 var(--mono); letter-spacing:.14em; text-transform:uppercase;
    color:var(--dim); padding-top:10px }}
  .sec h2 {{ font-size:clamp(28px,4vw,52px); line-height:1; margin:0 0 14px;
    letter-spacing:-0.03em }}
  .sec p {{ color:var(--dim); max-width:680px; margin:0 0 28px }}
  .lock {{ display:flex; align-items:center; gap:18px; padding:48px; border:1px solid var(--line);
    border-radius:16px; background:var(--halo) }}
  .lock b {{ font:700 64px/1 var(--sans); letter-spacing:.2em }}
  .tiles {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px;
    margin-top:14px }}
  .tile {{ border:1px solid var(--line); border-radius:14px; padding:28px; display:flex;
    flex-direction:column; align-items:flex-start; gap:14px; font:500 11px/1 var(--mono);
    letter-spacing:.12em; text-transform:uppercase; color:var(--dim) }}
  .tile svg {{ color:var(--ink) }}
  .tile.ink {{ background:#0A0A0A; color:#A1A1AA }} .tile.ink svg {{ color:#FFFFFF }}
  .tile.proof {{ background:#3730A3; color:#C7CBF5 }} .tile.proof svg {{ color:#FFFFFF }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:12px }}
  .sw {{ min-height:168px; border:1px solid var(--line); border-radius:14px; padding:18px;
    display:flex; flex-direction:column; justify-content:space-between }}
  .sw span, .sw code, .sw small {{ font:500 11px/1.4 var(--mono); letter-spacing:.1em;
    text-transform:uppercase; opacity:.85 }}
  .sw strong {{ font:700 22px/1 var(--sans) }}
  .type {{ border:1px solid var(--line); border-radius:14px; overflow:hidden }}
  .row {{ display:grid; grid-template-columns:150px 1fr; gap:24px; padding:20px 24px;
    border-bottom:1px solid var(--line); align-items:baseline }}
  .row:last-child {{ border-bottom:0 }}
  .row span {{ font:500 11px/1.5 var(--mono); letter-spacing:.12em; text-transform:uppercase;
    color:var(--dim) }}
  .rc {{ max-width:420px; border:1px solid var(--line); border-radius:14px; padding:22px;
    background:var(--halo) }}
  .rc strong {{ display:block; font:700 28px/1 var(--sans); color:var(--proof); margin:12px 0 }}
  .rc dl {{ display:grid; grid-template-columns:90px 1fr; gap:8px 12px; margin:0;
    font:12.5px/1.4 var(--mono) }}
  .rc dt {{ color:var(--dim); text-transform:uppercase; letter-spacing:.08em; font-size:11px }}
  .rc dd {{ margin:0 }}
  .voice {{ display:grid; grid-template-columns:1fr 1fr; gap:14px }}
  .voice div {{ border:1px solid var(--line); border-radius:14px; padding:22px }}
  .voice ul {{ margin:12px 0 0; padding-left:18px }} .voice li {{ margin:6px 0 }}
  .voice .bad li {{ color:var(--dim); text-decoration:line-through }}
  @media (max-width: 820px) {{
    .sec {{ grid-template-columns:1fr; gap:14px; padding:44px 0 }}
    .tiles, .voice {{ grid-template-columns:1fr }}
    .lock {{ padding:28px }} .lock b {{ font-size:40px }}
    .row {{ grid-template-columns:1fr; gap:6px }}
  }}
{design.BASE_CSS}</style></head><body>{design.nav('/brand')}<div class="wrap">
<p class="kicker">Brand kit · v1.0</p>
<h1>The identity of ARGUS.</h1>
<p class="sub">Mark, palette, type, the receipt, and the voice — rendered from the same tokens every
page of the console is built from.</p>

<section class="sec"><div class="meta">01 / Mark</div><div>
<h2>An eye on the evidence.</h2>
<p>Argus was the watchman with a hundred eyes. The mark is a ring with the pupil set off-centre:
looking at the data, not at you.</p>
<div class="lock">{design.mark_svg(72)}<b>ARGUS</b></div>
<div class="tiles"><div class="tile">{design.mark_svg(48)}Ink on paper</div>
<div class="tile ink">{design.mark_svg(48)}Reverse</div>
<div class="tile proof">{design.mark_svg(48)}Proof · one colour</div></div>
</div></section>

<section class="sec"><div class="meta">02 / Colour</div><div>
<h2>Mostly nothing. A little ink. One colour for proof.</h2>
<p>White space first, near-black for structure. Proof indigo marks only what was verified; Redline
marks only risk, losses and refusals. Green and red are not signal colours here: on a trading
screen they mean price up and price down.</p>
<div class="grid">{_swatches()}</div>
</div></section>

<section class="sec"><div class="meta">03 / Type</div><div>
<h2>One sans. One mono. No exceptions.</h2>
<p>Outfit carries every word. JetBrains Mono carries every number a reader might check.</p>
<div class="type">
<div class="row"><span>Display · 700</span><div style="font:700 44px/1 var(--sans);
letter-spacing:-.035em">Ask the desk.</div></div>
<div class="row"><span>Heading · 600</span><div style="font:600 24px/1.2 var(--sans)">Tied with
Bitget's own TWAP. Here is the replay.</div></div>
<div class="row"><span>Body · 400</span><div>Every figure is computed by the desk's engines from
Bitget data and names its source. The language model reads the question; it never writes a
number.</div></div>
<div class="row"><span>Mono · 400</span><div style="font:13px/1.5 var(--mono)">NVDAUSDT · 6.9 bps ·
p = 0.012 · seq 412 · 9f2c…a71e</div></div>
</div></div></section>

<section class="sec"><div class="meta">04 / Receipt</div><div>
<h2>Every answer ends in a receipt.</h2>
<p>Where each figure came from, which engine computed it, which rival it was measured against —
the console's promise, drawn.</p>
{_receipt()}
</div></section>

<section class="sec"><div class="meta">05 / Voice</div><div>
<h2>Terse. Specific. Checkable.</h2>
<p>ARGUS sounds like a risk desk writing a note: short, exact, and happy to show its working.</p>
<div class="voice"><div><span class="pill proof">Write like this</span><ul>{do}</ul></div>
<div class="bad"><span class="pill red">Not like this</span><ul>{dont}</ul></div></div>
</div></section>
</div>{design.footer()}</body></html>"""


__all__ = ["render"]
