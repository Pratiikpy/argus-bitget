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


def render(corrections: list[Correction]) -> str:
    """The page. Deliberately plain — this is a record, not a pitch."""
    esc = html.escape
    label = {"bug": "we shipped it broken", "loss": "a baseline beat us",
             "withdrawn": "claim withdrawn", "open": "still open"}
    rows = "".join(
        f"<article class='c {esc(c.kind)}'>"
        f"<span class='k'>{esc(label.get(c.kind, c.kind))}</span>"
        f"<h2>{esc(c.headline)}</h2><p>{esc(c.detail)}</p>"
        f"{_artefact(c.artefact)}</article>"
        for c in corrections
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
 .k {{ font:10.5px var(--mono); text-transform:uppercase; letter-spacing:.1em;
   color:var(--dim); display:block; margin-bottom:6px }}
 .c h2 {{ font-size:16px; margin:0 0 8px; line-height:1.35 }}
 .c p {{ margin:0 0 10px; color:var(--dim); font-size:14.5px }}
 code {{ font:11.5px var(--mono); color:var(--dim) }}
 a {{ color:var(--accent) }}
{design.BASE_CSS}</style></head><body>{design.nav('/wrong')}<div class="wrap">
<h1>What we got wrong</h1>
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
