"""The `/whitepaper` and `/deck` pages: the ARGUS whitepaper and proof deck, served from the site.

The whitepaper's source is `WHITEPAPER.md` at the repository's `argus/` root, where GitHub readers
find it. The console serves the copy packaged beside this module (`docs/WHITEPAPER.md`), because the
hosted bundle carries the package and not the repository; `tests/test_whitepaper_page.py` fails the
build when the two differ. The deck is a self-contained HTML file (`docs/deck.html`) drawn with the
same tokens, served as it is.

The markdown renderer is deliberately small: the whitepaper uses headings, paragraphs, lists, GFM
pipe tables, fenced blocks, emphasis, inline code and links, and nothing else. No markdown
library is installed on the hosted runtime, and a document this size does not need one.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from argus.lui import design

DOCS = Path(__file__).resolve().parent / "docs"
WHITEPAPER = DOCS / "WHITEPAPER.md"
DECK = DOCS / "deck.html"


def _inline(text: str) -> str:
    """Emphasis, inline code and links, with everything else escaped."""
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            out.append(f"<code>{html.escape(part[1:-1])}</code>")
            continue
        s = html.escape(part, quote=False)
        s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+|/[^)\s]*|#[^)\s]*)\)",
                   lambda m: f'<a href="{html.escape(m.group(2))}">{m.group(1)}</a>', s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", s)
        out.append(s)
    return "".join(out)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def render_markdown(source: str) -> tuple[str, list[tuple[int, str, str]]]:
    """``source`` as HTML, and its headings as (level, text, anchor) for a contents list."""
    lines = source.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    toc: list[tuple[int, str, str]] = []
    i = 0
    para: list[str] = []

    def flush() -> None:
        if para:
            out.append(f"<p>{_inline(' '.join(x.strip() for x in para))}</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush()
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(block))}</code></pre>")
            i += 1
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            flush()
            level, text = len(heading.group(1)), heading.group(2).strip()
            anchor = _slug(text)
            if level > 1:
                toc.append((level, text, anchor))
            out.append(f'<h{level} id="{anchor}">{_inline(text)}</h{level}>')
            i += 1
            continue
        if line.strip().startswith("|") and i + 1 < len(lines) and re.match(
                r"^\s*\|?\s*:?-{2,}", lines[i + 1]):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], rows[2:]
            out.append('<div class="tw"><table><thead><tr>'
                       + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
                                 for r in body) + "</tbody></table></div>")
            continue
        bullet = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if bullet:
            flush()
            ordered = bullet.group(2)[0].isdigit()
            items = []
            while i < len(lines):
                m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
                if m:
                    items.append(m.group(3))
                elif lines[i].startswith("  ") and lines[i].strip() and items:
                    items[-1] += " " + lines[i].strip()
                else:
                    break
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(x)}</li>" for x in items)
                       + f"</{tag}>")
            continue
        if line.startswith(">"):
            flush()
            quote = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i].lstrip("> ").strip())
                i += 1
            out.append(f"<blockquote>{_inline(' '.join(quote))}</blockquote>")
            continue
        if not line.strip() or re.match(r"^-{3,}\s*$", line):
            flush()
            if line.strip():
                out.append("<hr>")
            i += 1
            continue
        para.append(line)
        i += 1
    flush()
    return "\n".join(out), toc


def render_whitepaper() -> str:
    try:
        source = WHITEPAPER.read_text(encoding="utf-8")
    except OSError:
        source = "# ARGUS whitepaper\n\nThe whitepaper could not be read on this server."
    body, toc = render_markdown(source)
    contents = "".join(
        f'<li class="l{level}"><a href="#{anchor}">{html.escape(text)}</a></li>'
        for level, text, anchor in toc if level == 2)
    head = design.head("ARGUS — whitepaper",
                       "How ARGUS works, what it solves, how it was measured against named "
                       "rivals, and what it lost.", "/whitepaper")
    return f"""<!doctype html><html lang="en"><head>{head}
<style>{design.TOKENS_CSS}
 body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.7 system-ui,sans-serif }}
 .paper {{ max-width:760px }}
 .paper h1 {{ font-size:34px; line-height:1.2; margin:0 0 12px }}
 .paper h2 {{ font-size:23px; margin:44px 0 12px; padding-top:12px; border-top:1px solid
   var(--line); scroll-margin-top:80px }}
 .paper h3 {{ font-size:18px; margin:28px 0 8px; scroll-margin-top:80px }}
 .paper h4 {{ font-size:16px; margin:20px 0 6px }}
 .paper p, .paper li {{ overflow-wrap:anywhere }}
 .paper blockquote {{ margin:16px 0; padding:8px 16px; border-left:3px solid var(--accent);
   color:var(--dim) }}
 .paper pre {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
   padding:12px 14px; overflow-x:auto; font:13px/1.5 var(--mono) }}
 .paper code {{ font:0.88em var(--mono) }}
 .tw {{ overflow-x:auto; margin:14px 0 }}
 .paper table {{ border-collapse:collapse; font-size:14px; min-width:100% }}
 .paper th, .paper td {{ text-align:left; padding:6px 9px; border-bottom:1px solid var(--line);
   vertical-align:top }}
 .paper th {{ color:var(--dim); font-weight:600 }}
 .toc {{ border:1px solid var(--line); border-radius:10px; background:var(--panel);
   padding:14px 18px; margin:0 0 28px }}
 .toc b {{ font:600 12px var(--mono); letter-spacing:.12em; text-transform:uppercase;
   color:var(--dim) }}
 .toc ol {{ margin:8px 0 0; padding-left:0; list-style:none; columns:2; column-gap:28px }}
 .toc li {{ margin:0 0 4px; break-inside:avoid }}
 .alt {{ margin:0 0 20px; color:var(--dim); font-size:14px }}
 a {{ color:var(--accent) }}
 @media (max-width:640px) {{ .toc ol {{ columns:1 }} .paper h1 {{ font-size:27px }} }}
{design.BASE_CSS}</style></head><body>{design.nav('/whitepaper')}<div class="wrap paper">
<p class="alt">Also as a <a href="/deck">proof deck</a> and on
<a href="https://github.com/Pratiikpy/argus-bitget/blob/main/argus/WHITEPAPER.md">GitHub</a>.</p>
{body.replace('</h1>', '</h1>' + (f'<nav class="toc" aria-label="Contents"><b>Contents</b><ol>'
                                  f'{contents}</ol></nav>' if contents else ''), 1)}
</div>{design.footer()}</body></html>"""


VIDEO_SRC = "/media/argus-demo.mp4"
"""Served by the host's CDN beside the console (`deploy/media/`), not by the Python handler."""


def render_video() -> str:
    """The demo video on the site in the console's own tokens: a judge needs no other account to
    watch it."""
    head = design.head("ARGUS — demo video",
                       "The console recorded end to end on the live site: a research task, a "
                       "stress, premise checks, a newcomer's questions, the register.", "/video")
    return f"""<!doctype html><html lang="en"><head>{head}
<style>{design.TOKENS_CSS}
 body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.6 system-ui,sans-serif }}
 .film {{ width:100%; aspect-ratio:16/9; background:#000; border:1px solid var(--line);
   border-radius:10px }}
 .sub {{ color:var(--dim); margin:0 0 18px }}
 a {{ color:var(--accent) }}
{design.BASE_CSS}</style></head><body>{design.nav('/video')}<div class="wrap">
<h1>Demo video</h1>
<p class="sub">Every frame is the live console, recorded by a script rather than by hand, with
captions and no voice. About three and a half minutes.</p>
<video class="film" controls preload="metadata" playsinline src="{VIDEO_SRC}">
<a href="{VIDEO_SRC}">Download the video</a></video>
<p class="sub">Read the same ground in the <a href="/whitepaper">whitepaper</a> or the
<a href="/deck">proof deck</a>; everything else is on <a href="/materials">one page</a>.</p>
</div>{design.footer()}</body></html>"""


def render_deck() -> str:
    try:
        return DECK.read_text(encoding="utf-8")
    except OSError:
        return render_markdown("# Proof deck\n\nThe deck could not be read on this server.")[0]


__all__ = ["DECK", "VIDEO_SRC", "WHITEPAPER", "render_deck", "render_markdown", "render_video",
           "render_whitepaper"]
