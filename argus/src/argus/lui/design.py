"""The ARGUS identity, in one place: tokens, type, the mark, the nav, the receipt.

Every page of the console used to carry its own palette — five copies of one blue that drifted a
little each time a page was written. This module is the single source: a page includes
:data:`TOKENS_CSS` where its own ``:root`` block used to be, appends :data:`BASE_CSS` after its own
rules, and opens its body with :func:`nav`. The brand page (``/brand``) renders the system itself.
Appended last, BASE wins a tie, so a page never restates a value BASE sets: five pages carried
widths, heading sizes and a button colour that never applied (audit, 2026-09-26), and
`tests/test_page_polish.py` now fails on one.

**The system** (modelled on the owner's own Blank brand kit, brand.myblank.app, and given its own
mark and accent so it is ARGUS's, not Blank's):

* **Colour is mostly nothing.** Paper and halo grounds (~60%), near-black ink for structure (~33%),
  and one accent (~7%), *Proof* indigo, reserved for what is verified — an OWNED row, a passed test,
  a source that answered. *Redline* is reserved for risk, losses, refusals and blocks. Green and
  red are deliberately not the signal colours: on a trading screen they read as price up and down.
* **One sans, one mono.** Outfit carries every word; JetBrains Mono carries every number a reader
  might check — prices, p-values, order ids, hashes.
* **The mark is an eye.** Argus was the watchman with a hundred eyes: a ring with the pupil set
  off-centre, looking at the evidence, not at the reader.
* **The receipt is the motif.** Every answer ends in a small card that says where each figure came
  from — the console's promise, drawn.

Fonts load from Google Fonts with system fallbacks, so a blocked font CDN degrades the look, never
the page.
"""

from __future__ import annotations

import html
import re

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;'
    '500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap">'
)

PALETTE: tuple[tuple[str, str, str, str], ...] = (
    ("Ink", "#0A0A0A", "core", "type, primary buttons, the mark"),
    ("Stone", "#1D1D1F", "core", "hover states, deep ground"),
    ("Proof", "#3730A3", "accent", "verified: OWNED, passed, answered"),
    ("Redline", "#C2410C", "signal", "risk, losses, refusals, blocks"),
    ("Paper", "#FFFFFF", "surface", "default ground"),
    ("Halo", "#FAFAF7", "surface", "warm ground, cards"),
    ("Veil", "#F4F5F7", "surface", "fields and code"),
    ("Line", "#E5E7EB", "line", "borders"),
    ("Fog", "#9CA3AF", "neutral", "muted interface"),
    ("Graphite", "#6B7280", "neutral", "secondary text"),
)
"""Name, hex, role, use — rendered on /brand and mirrored in :data:`TOKENS_CSS`."""

TOKENS_CSS = """
  :root {
    --paper:#FFFFFF; --halo:#FAFAF7; --veil:#F4F5F7; --stone:#1D1D1F; --fog:#9CA3AF;
    --graphite:#6B7280; --proof:#3730A3; --proof-soft:#EEF0FF; --redline:#C2410C;
    --redline-soft:#FFF1EA;
    --ink:#0A0A0A; --dim:#6B7280; --line:#E5E7EB; --bg:#FFFFFF; --panel:#FAFAF7;
    --accent:#3730A3; --on-accent:#FFFFFF; --warn:#C2410C; --bad:#C2410C; --ok:#3730A3;
    --good:#3730A3;
    --sans:"Outfit",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    --mono:"JetBrains Mono",ui-monospace,"SF Mono",Menlo,Consolas,monospace;
    color-scheme: light;
  }
  /* The system's theme unless the reader chose one with the toggle in the bar (round 44 visual
     audit, minor 6: the page followed the system only, with no way to override it). */
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --paper:#0A0A0A; --halo:#111113; --veil:#17181B; --stone:#E7E7E4; --fog:#6B7280;
      --graphite:#A1A1AA; --proof:#A5B4FC; --proof-soft:#1B1D3A; --redline:#FB923C;
      --redline-soft:#2A1509;
      --ink:#F2F2EF; --dim:#A1A1AA; --line:#26272B; --bg:#0A0A0A; --panel:#111113;
      --accent:#A5B4FC; --on-accent:#0A0A0A; --warn:#FB923C; --bad:#FB923C; --ok:#A5B4FC;
      --good:#A5B4FC;
      color-scheme: dark;
    }
  }
  :root[data-theme="dark"] {
    --paper:#0A0A0A; --halo:#111113; --veil:#17181B; --stone:#E7E7E4; --fog:#6B7280;
    --graphite:#A1A1AA; --proof:#A5B4FC; --proof-soft:#1B1D3A; --redline:#FB923C;
    --redline-soft:#2A1509;
    --ink:#F2F2EF; --dim:#A1A1AA; --line:#26272B; --bg:#0A0A0A; --panel:#111113;
    --accent:#A5B4FC; --on-accent:#0A0A0A; --warn:#FB923C; --bad:#FB923C; --ok:#A5B4FC;
    --good:#A5B4FC;
    color-scheme: dark;
  }
"""

BASE_CSS = """
  * { box-sizing:border-box }
  html { -webkit-text-size-adjust:100% }
  body { margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans);
    font-size:15.5px; line-height:1.6; -webkit-font-smoothing:antialiased }
  .wrap { max-width:1080px; margin:0 auto; padding:40px 24px 72px }
  h1 { font-family:var(--sans); font-weight:700; font-size:clamp(28px,4vw,44px); line-height:1.05;
    letter-spacing:-0.035em; margin:0 0 12px; text-wrap:balance }
  h2 { font-family:var(--sans); font-weight:700; letter-spacing:-0.01em; line-height:1.15;
    text-wrap:balance }
  h3 { font-family:var(--sans); font-weight:600; letter-spacing:0 }
  .sub { color:var(--dim); font-size:15.5px; max-width:760px }
  .wrap p, .wrap li, .wrap dd, .wrap td { overflow-wrap:anywhere }
  a { color:var(--accent); text-underline-offset:3px }
  a:focus-visible, button:focus-visible, input:focus-visible, summary:focus-visible,
  .chip:focus-visible { outline:2px solid var(--accent); outline-offset:2px; border-radius:6px }
  code, .mono, .tag, .meta, .src, .kicker { font-family:var(--mono) }
  .kicker { font-size:11.5px; letter-spacing:.14em; text-transform:uppercase; color:var(--dim) }
  button, .btn { font-family:var(--sans); font-weight:600; letter-spacing:-0.005em;
    background:var(--ink); color:var(--paper); border:1px solid var(--ink); border-radius:10px }
  button:hover, .btn:hover { background:var(--stone); border-color:var(--stone) }
  input[type=text], input:not([type]) { font-family:var(--sans) }
  .card { border-radius:12px }
  .chip { font-family:var(--sans) }
  .pill { display:inline-block; font:600 11px/1 var(--mono); letter-spacing:.12em;
    text-transform:uppercase; padding:6px 9px; border:1px solid var(--line); border-radius:999px;
    color:var(--dim) }
  .pill.proof { color:var(--proof); border-color:var(--proof); background:var(--proof-soft) }
  .pill.red { color:var(--redline); border-color:var(--redline); background:var(--redline-soft) }
  @media (prefers-reduced-motion: reduce) {
    * { transition:none !important; animation:none !important }
  }

  .nav { position:sticky; top:0; z-index:40; display:flex; align-items:center;
    justify-content:space-between; gap:16px; padding:14px 24px;
    background:color-mix(in srgb, var(--bg) 82%, transparent);
    backdrop-filter:saturate(180%) blur(14px); -webkit-backdrop-filter:saturate(180%) blur(14px);
    border-bottom:1px solid var(--line) }
  .nav .brand { display:flex; align-items:center; gap:9px; color:var(--ink); text-decoration:none;
    font:700 17px/1 var(--sans); letter-spacing:.2em }
  .nav .links { display:flex; gap:4px; flex-wrap:wrap; justify-content:flex-end }
  .nav .links a { color:var(--dim); text-decoration:none; font:500 12px/1 var(--mono);
    letter-spacing:.1em; text-transform:uppercase; padding:8px 10px; border-radius:8px }
  .nav .links a:hover { color:var(--ink); background:var(--veil) }
  .nav .links a.on { color:var(--ink); background:var(--veil) }
  .nav .right { display:flex; align-items:center; gap:8px }
  .nav .theme { padding:0; width:34px; height:34px; display:grid; place-items:center;
    background:none; color:var(--dim); border:1px solid var(--line); border-radius:8px;
    font:600 15px/1 var(--sans) }
  .nav .theme:hover { color:var(--ink); background:var(--veil); border-color:var(--line) }
  /* The phone menu: absent on desktop, where every link fits in the bar. */
  .nav .menu { display:none }
  @media (max-width: 720px) {
    /* A Menu button, not a sideways-scrolling row. The row hid four of seven links behind a fade
       and a chevron, and a first-time user still read it as three links and a stray arrow (round
       11, 2026-09-30, after a first-user audit on 2026-09-29 added the chevron). A <details>
       opens and closes with no script, by tap, Enter or Space, and a screen reader announces it
       as a disclosure. */
    .nav { padding:10px 16px }
    .nav .links-wrap { display:none }
    .nav .menu { display:block; position:relative }
    .nav .menu summary { list-style:none; cursor:pointer; color:var(--ink);
      font:600 12px/1 var(--mono); letter-spacing:.1em; text-transform:uppercase;
      padding:9px 12px; border:1px solid var(--line); border-radius:8px }
    .nav .menu summary::-webkit-details-marker { display:none }
    .nav .menu summary::after { content:" \\25BE"; content:" \\25BE" / "" }
    .nav .menu[open] summary::after { content:" \\25B4"; content:" \\25B4" / "" }
    .nav .menu .menu-list { position:absolute; right:0; top:calc(100% + 6px); min-width:230px;
      display:flex; flex-direction:column; padding:6px; background:var(--panel);
      border:1px solid var(--line); border-radius:10px;
      box-shadow:0 12px 32px color-mix(in srgb, var(--ink) 18%, transparent) }
    .nav .menu .menu-list a { color:var(--ink); text-decoration:none; padding:11px 12px;
      font:500 15px/1.2 var(--sans); border-radius:6px }
    .nav .menu .menu-list a.on { background:var(--veil); font-weight:600 }
    .wrap { padding:28px 16px 56px }
  }
  .foot { border-top:1px solid var(--line); margin-top:56px; padding:22px 24px 36px;
    color:var(--dim); font:12px/1.7 var(--mono); display:flex; gap:18px; flex-wrap:wrap;
    justify-content:space-between; max-width:1080px; margin-left:auto; margin-right:auto }
  .foot a { color:var(--dim) }
  .skip { position:absolute; left:-9999px; top:8px; z-index:50; padding:8px 12px;
    background:var(--ink); color:var(--paper); border-radius:8px; font:600 13px/1 var(--sans) }
  .skip:focus { left:16px }
  /* Back to the top from anywhere on a long page (round 44 visual audit, major 3: /proof, /wrong
     and /research ran to dozens of phone screens with no way back but scrolling). */
  .totop { position:fixed; right:16px; bottom:16px; z-index:30; padding:10px 14px;
    border:1px solid var(--line); border-radius:999px; background:var(--panel); color:var(--ink);
    font:600 13px/1 var(--sans); text-decoration:none;
    box-shadow:0 6px 18px color-mix(in srgb, var(--ink) 14%, transparent) }
  .totop:hover { border-color:var(--accent) }
"""

LINKS: tuple[tuple[str, str], ...] = (
    ("/", "Console"),
    ("/research", "Research task"),
    ("/proof", "What we beat"),
    ("/wrong", "What we got wrong"),
    ("/status", "Status"),
    ("/materials", "Materials"),
)
"""The console's own pages. The Track 2 agent's record (/agent) is not among them: the handbook
requires each entry to be an independent project (Basic Competition Rules, rule 2), and a tab for
the other entry inside this one read as one system entered twice (readiness audit, 2026-09-25).
The page still answers at its URL for anyone who follows a link to it."""


def mark_svg(size: int = 22, colour: str = "currentColor") -> str:
    """The ARGUS eye: a ring, and a pupil set off-centre toward the evidence."""
    return (f'<svg viewBox="0 0 64 64" width="{size}" height="{size}" aria-hidden="true" '
            f'xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="32" r="24" fill="none" '
            f'stroke="{colour}" stroke-width="9"/><circle cx="40" cy="28" r="9" '
            f'fill="{colour}"/></svg>')


def favicon() -> str:
    """The mark as a data-URI favicon: ink on paper."""
    return ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E"
            "%3Crect width='64' height='64' rx='14' fill='%230A0A0A'/%3E%3Ccircle cx='32' cy='32' "
            "r='19' fill='none' stroke='%23FFFFFF' stroke-width='7'/%3E%3Ccircle cx='38' cy='28' "
            "r='7' fill='%23FFFFFF'/%3E%3C/svg%3E")


THEME_INIT = ("<script>(function(){try{var t=localStorage.getItem('argus.theme');"
              "if(t==='light'||t==='dark')document.documentElement.dataset.theme=t;}"
              "catch(e){}})();</script>")
"""Runs in the head, before the body paints, so a chosen theme never flashes the other one."""

THEME_BUTTON = (
    '<button type="button" class="theme" id="theme" aria-label="Switch between light and dark" '
    'title="Switch between light and dark">&#9680;</button>'
    "<script>(function(){var b=document.getElementById('theme'),r=document.documentElement;"
    "b.addEventListener('click',function(){var dark=r.dataset.theme?r.dataset.theme==='dark':"
    "window.matchMedia('(prefers-color-scheme: dark)').matches;var t=dark?'light':'dark';"
    "r.dataset.theme=t;try{localStorage.setItem('argus.theme',t);}catch(e){}});})();</script>")
"""The light/dark toggle in the bar: the reader's choice is kept in this browser only."""


def nav(active: str = "/") -> str:
    """The top bar every page opens with. ``active`` is the current path.

    A keyboard or screen-reader visitor lands on a skip link first, and the current page is
    announced as such rather than marked only by colour (accessibility audit, 2026-09-26). The
    skip target sits directly after the bar, so every page gets it without naming its own main."""
    on = ' class="on" aria-current="page"'
    links = "".join(f'<a href="{href}"{on if href == active else ""}>{label}</a>'
                    for href, label in LINKS)
    return (f'<a class="skip" href="#main">Skip to content</a>'
            f'<nav class="nav" aria-label="Primary"><a class="brand" href="/" aria-label="ARGUS '
            f'home">{mark_svg(22)}<span>ARGUS</span></a><div class="right"><div class="links-wrap">'
            f'<div class="links">'
            f'{links}<a href="https://github.com/Pratiikpy/argus-bitget">GitHub</a></div></div>'
            f'<details class="menu"><summary>Menu</summary><div class="menu-list">{links}'
            f'<a href="https://github.com/Pratiikpy/argus-bitget">GitHub</a></div></details>'
            f'{THEME_BUTTON}</div></nav><span id="main" tabindex="-1"></span>')


def footer() -> str:
    """Every page's foot. The pages outside the top bar are linked here: /architecture, /policy,
    /factors and /agent returned 200 but nothing linked to them (fresh-eyes audit, 2026-09-29)."""
    return ('<footer class="foot"><span>ARGUS · every figure computed, every source named, every '
            'loss published.</span><span>Analysis, not advice. Data: Bitget, SEC EDGAR, FRED, '
            'public news. <a href="/architecture">Architecture</a> · <a href="/policy">Risk '
            'policy</a> · <a href="/factors">Factors</a> · '
            '<a href="/agent">Trading agent (separate project)</a> · '
            '<a href="/brand">Brand</a></span></footer>'
            '<a class="totop" href="#main" aria-label="Back to the top of the page">Top '
            '&uarr;</a>')


PUBLIC_URL = "https://deploy-topaz-seven-64.vercel.app"
"""Where the console is hosted: a link preview needs an absolute image URL and page URL."""

OG_IMAGE = f"{PUBLIC_URL}/og.png"
"""The 1200x630 preview card (`lui/og.png`, the mark and the console's promise), served at
``/og.png``. The handbook makes an X post mandatory, and a link without an image previews as a
bare line of text."""


def head(title: str, description: str, path: str = "/") -> str:
    """Everything a page's ``<head>`` shares: viewport, favicon, fonts, and the link-preview
    card a pasted URL renders as (title, description, image and URL)."""
    return (f'<meta charset="utf-8"><meta name="viewport" content="width=device-width,'
            f'initial-scale=1"><title>{title}</title><meta name="description" '
            f'content="{description}"><meta property="og:title" content="{title}">'
            f'<meta property="og:description" content="{description}"><meta property="og:type" '
            f'content="website"><meta property="og:url" content="{PUBLIC_URL}{path}">'
            f'<meta property="og:image" content="{OG_IMAGE}"><meta property="og:image:width" '
            f'content="1200"><meta property="og:image:height" content="630">'
            f'<meta name="twitter:card" content="summary_large_image"><meta name="twitter:image" '
            f'content="{OG_IMAGE}"><link rel="icon" href="{favicon()}">{FONTS}{THEME_INIT}')


__all__ = ["BASE_CSS", "FONTS", "LINKS", "PALETTE", "THEME_BUTTON", "THEME_INIT", "TOKENS_CSS",
           "favicon", "footer", "head", "mark_svg", "nav"]


_URL = re.compile(r"https?://[^\s<>\"']*[^\s<>\"'.,;:!?)\]]")
_HOST = re.compile(r"^https?://(?:www\.)?([^/?#]+)")


def linked(text: str) -> str:
    """``text`` escaped for HTML, with each address in it made a link that reads its host.

    The same rule as the console's ``linked`` in ``lui/server.py``: a citation that can only be
    read, not opened, was the first-user audit's top finding (2026-09-29)."""
    out, at = [], 0
    for match in _URL.finditer(text):
        url = match.group(0)
        host = _HOST.match(url)
        out.append(html.escape(text[at:match.start()]))
        out.append(f'<a href="{html.escape(url)}" rel="noopener noreferrer" target="_blank">'
                   f"{html.escape(host.group(1) if host else url)} &#8599;</a>")
        at = match.end()
    out.append(html.escape(text[at:]))
    return _BPS.sub(r'\1<abbr title="basis points: hundredths of a percent, so 100bps is 1%">bps'
                    r"</abbr>", "".join(out))


_BPS = re.compile(r"(\d)bps\b")
"""A figure in basis points, given its definition on hover (the console's ``linked`` does the
same)."""
