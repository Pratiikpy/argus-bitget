"""``/materials`` — every deliverable of the Track 3 entry on one page, each link live.

The form has one "Submission Materials Link" field for every link (handbook, submission
requirements), and a judge who opens it should not have to know which page proves what. This page
is that index. Its figures are read at request time from the artefacts ``/proof`` and ``/wrong``
render, so it cannot quote a register count the register no longer holds. A deliverable that does
not exist yet (the demo video, the X post) is left off rather than shown as a placeholder: its URL
comes from the environment the day it exists (``ARGUS_DEMO_VIDEO_URL``, ``ARGUS_X_POST_URL``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from argus.lui import design

REPOSITORY = "https://github.com/Pratiikpy/argus-bitget"
TELEGRAM = "https://t.me/argusbitgetbot"
VIDEO_ENV = "ARGUS_DEMO_VIDEO_URL"
X_POST_ENV = "ARGUS_X_POST_URL"


@dataclass(frozen=True)
class Item:
    """One deliverable: what it is, where it is, and what it shows."""

    label: str
    href: str
    shows: str
    group: str

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "href": self.href, "shows": self.shows, "group": self.group}


def _register_line(counts: dict[str, int]) -> str:
    if not counts:
        return "The register could not be read on this load; the page itself says why."
    total = sum(counts.values())
    order = ("owned", "tied", "implemented", "lost")
    parts = [f"{counts.get(state, 0)} {state.upper()}" for state in order]
    return (f"{total} capabilities, each run against a named rival on the same input: "
            f"{', '.join(parts)} — read from the register now.")


def collect(data_dir: Path, base: str = "") -> list[Item]:
    """Every deliverable, with the live figures that describe it."""
    from argus.lui.corrections_page import collect as collect_wrong
    from argus.lui.proof_page import collect as collect_wins

    _, counts = collect_wins(data_dir)
    wrong = collect_wrong(data_dir)
    losses = sum(1 for c in wrong if c.kind == "loss")
    items = [
        Item("Console", f"{base}/", "Ask about your Bitget book in plain words; every figure "
             "computed from live data and named by source, every line labelled by where it came "
             "from.", "Use it"),
        Item("Research task", f"{base}/research", "The Track 3 demo the handbook asks for: one "
             "question about adding a name to a book, seven engines, and an actionable insight. "
             "Change ?name=, ?size= and ?book= to run your own.", "Use it"),
        Item("What we beat", f"{base}/proof", _register_line(counts), "Check it"),
        Item("What we got wrong", f"{base}/wrong",
             f"{len(wrong)} findings published, {losses} of them comparisons a rival won.",
             "Check it"),
        Item("Status", f"{base}/status", "Bitget's toolkit and every data source, checked live "
             "when the page loads, with the desk's record and its age.", "Check it"),
        Item("Source code", REPOSITORY, "Public, MIT. `python -m argus.eval.standing` re-derives "
             "the register from its artefacts; `python -m argus.eval.docclaims` checks every "
             "figure the documents quote.", "Check it"),
        Item("MCP server", f"{base}/mcp", "The same console for an agent: JSON-RPC 2.0 over POST, "
             "one tool, argus_ask.", "Other doors"),
        Item("Telegram", TELEGRAM, "The same console in a chat; /book saves holdings.",
             "Other doors"),
        Item("JSON", f"{base}/proof?format=json", "Every page's data without the page: add "
             "?format=json to /proof, /wrong, /status or /research.", "Other doors"),
        Item("Brand kit", f"{base}/brand", "The identity, drawn from the tokens every page uses.",
             "Other doors"),
    ]
    video = os.environ.get(VIDEO_ENV, "").strip()
    if video:
        items.insert(2, Item("Demo video", video, "The research task, recorded end to end on the "
                             "live console.", "Use it"))
    post = os.environ.get(X_POST_ENV, "").strip()
    if post:
        items.append(Item("X post", post, "The entry's introduction on X.", "Other doors"))
    return items


def _row(item: Item) -> str:
    shown = item.href.removeprefix("https://")
    return (f'<li class="it"><div class="lab">{escape(item.label)}</div><div>'
            f'<a href="{escape(item.href, quote=True)}">{escape(shown)}</a>'
            f'<p>{escape(item.shows)}</p></div></li>')


def render(items: list[Item]) -> str:
    groups: dict[str, list[Item]] = {}
    for item in items:
        groups.setdefault(item.group, []).append(item)
    sections = "".join(
        f'<section class="grp"><h2>{escape(name)}</h2><ul>{"".join(_row(i) for i in rows)}</ul>'
        f'</section>' for name, rows in groups.items())
    return f"""<!doctype html><html lang="en"><head>{design.head(
        "ARGUS — submission materials",
        "Every deliverable of the ARGUS Track 3 entry on one page, each link live.")}
<style>{design.TOKENS_CSS}
  .grp {{ padding:36px 0 8px; border-bottom:1px solid var(--line) }}
  .grp:last-of-type {{ border-bottom:0 }}
  .grp h2 {{ font-size:13px; font-family:var(--mono); font-weight:500; letter-spacing:.14em;
    text-transform:uppercase; color:var(--dim); margin:0 0 8px }}
  .grp ul {{ list-style:none; margin:0; padding:0 }}
  .it {{ display:grid; grid-template-columns:200px minmax(0,1fr); gap:24px; padding:18px 0;
    border-top:1px solid var(--line) }}
  .it:first-child {{ border-top:0 }}
  .lab {{ font:600 17px/1.3 var(--sans) }}
  .it a {{ font:500 14px/1.4 var(--mono); overflow-wrap:anywhere }}
  .it p {{ margin:6px 0 0; color:var(--dim); max-width:680px }}
  @media (max-width: 720px) {{ .it {{ grid-template-columns:1fr; gap:6px }} }}
{design.BASE_CSS}</style></head><body>{design.nav('/materials')}<div class="wrap">
<p class="kicker">Submission materials · Track 3 · AI Trading Desk · Open Theme</p>
<h1>Everything, on one page.</h1>
<p class="sub">Every link here is live, and the figures on this page are read from the same
artefacts the pages use, at the moment you load it.</p>
{sections}
</div>{design.footer()}</body></html>"""


__all__ = ["Item", "collect", "render"]
