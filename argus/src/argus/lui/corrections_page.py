"""The `/wrong` page: every finding `lui/corrections.py` collects, rendered.

The collection moved to its own module on 2026-09-27 (audit finding 169), so the console's
answer engine reads the findings without importing a page.
"""

from __future__ import annotations

import html
import re

from argus.lui import design
from argus.lui.corrections import (
    Correction,
    collect,
    register_losses,
    register_regrades,
)
from argus.lui.proof_page import REPOSITORY

FAVICON = design.favicon()
"""Kept identical to (and duplicated from, rather than imported from) `lui/server.py`'s constant
of the same name: `server.py` only imports this module lazily, inside a route handler, so a
module-level import in the other direction here is avoidable risk for six lines of duplication."""


def _artefact(path: str) -> str:
    """The artefact as a link to the file in the public repository, as /proof shows it; a plain
    name when it is not a file path. It was dead text until 2026-09-27 (audit)."""
    shown = html.escape(path)
    if re.fullmatch(r"[\w./-]+\.(json|jsonl|md|py|csv|txt)", path):
        # "argus/paper/corrections.py" names a module; its file is under src/ in the repository.
        target = f"src/{path}" if path.startswith("argus/") else path
        return f"<a href='{REPOSITORY}{html.escape(target)}'><code>{shown}</code></a>"
    return f"<code>{shown}</code>"


_CODE = re.compile(r"`([^`]+)`")


def _prose(text: str) -> str:
    """Escaped text with `inline code` set as code, so a path or a command reads as one rather
    than between literal backticks (round 45 visual audit, minor 7)."""
    return _CODE.sub(lambda m: f"<code>{m.group(1)}</code>", html.escape(text))


def _whole(text: str) -> str:
    """``text`` with a cut-off tail made whole. The collector shortens long findings to a fixed
    length and appends "..." (`lui/corrections.py`), which left words such as "day-clustere..."
    on the page. The tail is taken back to the last sentence end when that keeps most of the text,
    else to the last whole word, and closed with an ellipsis."""
    if not text.endswith("..."):
        return text
    body = text[:-3]
    sentence = max(body.rfind(". "), body.rfind("; "))
    if sentence >= len(body) * 0.6:
        return body[:sentence + 1].rstrip() + " …"
    cut = body.rsplit(" ", 1)[0] if " " in body else body
    return cut.rstrip(" ,;:-") + "…"


def render(corrections: list[Correction]) -> str:
    """The page. Deliberately plain — this is a record, not a pitch."""
    esc = html.escape
    label = {"bug": "we shipped it broken", "loss": "a baseline beat us",
             "withdrawn": "claim withdrawn", "open": "still open"}
    rows = "".join(
        f"<details class='c {esc(c.kind)}'><summary>"
        f"<span class='k'>{esc(label.get(c.kind, c.kind))}</span>"
        f"<h2>{_prose(_whole(c.headline))}</h2></summary><p>{_prose(_whole(c.detail))}</p>"
        f"{_artefact(c.artefact)}</details>"
        for c in corrections
    )
    # Plain language, positioned directly above the first entry rather than after the intro above
    # it: a first-time reader's first line of text on this page was a bug report ("2 ledger rows
    # booked P&L..."), read cold with nothing said first about why a page like this exists
    # (first-user audit, 2026-09-29). The counts are read from the same `corrections` list `rows`
    # is built from, never retyped.
    lost = sum(1 for c in corrections if c.kind == "loss")
    kinds = {k: sum(1 for c in corrections if c.kind == k) for k in label}
    tally = ", ".join(f"{n} {label[k]}" for k, n in kinds.items() if n)
    plain = (
        "<div class='plain'><p>In plain terms: this is everything wrong with ARGUS that we can "
        "point to real evidence for &mdash; things we shipped broken, times a named competitor "
        "beat us on a fair test, and numbers we published and later had to take back. We show it "
        "because a trading tool that only tells you what went right is the one to be careful of; "
        "the ones worth trusting are the ones that say when they lost. "
        f"{len(corrections)} entries follow, {lost} of them a competitor we lost to by name, and "
        "each is checked against its own record every time this page loads, not written once and "
        f"left to go stale.</p><p class='tally'>By kind: {esc(tally)}.</p></div>"
    )
    head = design.head("ARGUS — what we got wrong",
                       "Every loss, bug and withdrawn claim, with its number and the artefact "
                       "behind it.", "/wrong")
    return f"""<!doctype html><html lang="en"><head>{head}
<style>{design.TOKENS_CSS}
 body {{ margin:0; background:var(--bg); color:var(--ink);
   font:15px/1.6 system-ui,sans-serif }}
 .sub {{ margin:0 0 24px }}
 .c {{ background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--dim);
   border-radius:10px; padding:16px 18px; margin-bottom:14px }}
 .c.bug {{ border-left-color:var(--bad) }}
 .c.loss {{ border-left-color:var(--warn) }}
 .c.withdrawn {{ border-left-color:var(--graphite) }}  /* Proof indigo is for what is verified */
 .c.open {{ border-left-color:var(--dim) }}
 .k {{ font:12px var(--mono); text-transform:uppercase; letter-spacing:.1em;
   color:var(--dim); display:block; margin-bottom:6px }}
 .c h2 {{ font-size:16px; margin:0 0 8px; line-height:1.35; display:block }}
 .c p {{ margin:0 0 10px; color:var(--dim); font-size:14.5px }}
 code {{ font:12px var(--mono); color:var(--dim) }}
 /* Each entry folds to its kind and headline and opens on a tap, as the cards on /proof do: the
    page ran to fifteen phone screens (round 45 visual audit, minor 4). */
 .c > summary {{ list-style:none; cursor:pointer }}
 .c > summary::-webkit-details-marker {{ display:none }}
 .c > summary h2::after {{ content:"  +"; color:var(--accent); font-weight:600 }}
 .c[open] > summary h2::after {{ content:"  \u2212" }}
 .c > summary h2 {{ margin:0 }}
 .c[open] > summary h2 {{ margin:0 0 8px }}
 .c p code {{ color:var(--ink) }}
 a {{ color:var(--accent) }}
 .plain {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
   padding:14px 16px; margin:0 0 16px }}
 .plain p {{ margin:0; font-size:14.5px; max-width:none }}
 .plain p.tally {{ margin-top:8px; color:var(--dim); font-size:13.5px }}
{design.BASE_CSS}</style></head><body>{design.nav('/wrong')}<div class="wrap">
<h1>What we got wrong</h1>
{plain}
<p class="sub">Every entry is read out of the artefact that recorded it, at the moment you load
this page &mdash; not written down once and left to drift. A finding whose artefact cannot be read
is listed as unreadable rather than dropped, because a losses page that silently shortens is the
most flattering possible lie. <a href="/">back to the console</a></p>
{rows}
<p class="sub">This is not everything wrong with ARGUS &mdash; nothing could be, and claiming
completeness would be its own overstatement. It is the set of findings that already have an
artefact behind them.</p>
</div>{design.footer()}</body></html>"""


__all__ = ["Correction", "collect", "register_losses", "register_regrades", "render"]
