"""The `/architecture` page: the checked structure of ARGUS, read from `data/architecture.json`.

Track 2 judges "Agent architecture quality". `eval/architecture.py` already proves the structure
on every run — the deterministic core never reaches a model, every package has a layer, no module
imports upward, and every dependency cycle is named with whether it is safe at load — and nothing
showed it: the report was a JSON file and a plain-text table (research/harvest/57-import-linter.md).
import-linter ships a web view of its contracts (BSD-2-Clause); its JavaScript stack is not taken,
because the page is drawn with the console's own tokens (`lui/design.py`), and this page shows one
thing import-linter's cannot: whether a cycle is broken at import time by a deferred import.

Every figure is read from the artefact at request time; nothing here recomputes or restates it.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

from argus.lui import design
from argus.lui.proof_page import REPOSITORY


def load(data_dir: Path) -> dict[str, Any] | None:
    try:
        blob = json.loads((data_dir / "architecture.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return blob if isinstance(blob, dict) else None


def _sentences(text: str) -> str:
    """The report's verdict with each sentence starting in capitals, as the page's prose does,
    and "7 dependency cycle(s)" said as "7 dependency cycles" (visual review, 2026-09-29)."""
    from argus.lui.plural import resolve_plurals

    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(),
                  resolve_plurals(text))


def _layers() -> str:
    from argus.eval.architecture import DETERMINISTIC, LAYERS, MODEL_FACING

    by_rank: dict[int, list[str]] = {}
    for package, rank in LAYERS.items():
        by_rank.setdefault(rank, []).append(package)
    rows = "".join(
        f"<tr><td class='n'>{rank}</td><td>{' · '.join(html.escape(p) for p in sorted(names))}"
        f"</td></tr>" for rank, names in sorted(by_rank.items(), reverse=True))
    core = ", ".join(DETERMINISTIC)
    facing = ", ".join(m.removeprefix("argus.") for m in MODEL_FACING)
    return (f"<table class='text'><thead><tr><th>Layer</th><th>Packages</th></tr></thead>"
            f"<tbody>{rows}"
            f"</tbody></table><p class='sub'>A module imports only from its own layer or below. "
            f"The deterministic core ({html.escape(core)}) may not reach {html.escape(facing)} at "
            f"any depth: numbers are computed, never asked for.</p>")


def render(blob: dict[str, Any] | None) -> str:
    esc = html.escape
    if blob is None:
        body = ("<p class='sub'>data/architecture.json could not be read, so nothing is shown "
                "rather than something unchecked.</p>")
    else:
        violations = blob.get("violations") or []
        contract = ("<p class='ok'>Every declared contract holds.</p>" if not violations else
                    "<table><thead><tr><th>Contract</th><th>Where</th><th>What</th></tr></thead>"
                    "<tbody>" + "".join(
                        f"<tr><td>{esc(str(v.get('contract')))}</td>"
                        f"<td><code>{esc(str(v.get('where')))}</code></td>"
                        f"<td>{esc(str(v.get('detail')))}</td></tr>" for v in violations)
                    + "</tbody></table>")
        cycles = "".join(
            f"<li><span class='tag {'bad' if c.get('import_time') else 'ok'}'>"
            f"{'breaks at load' if c.get('import_time') else 'safe at load'}</span> "
            f"{' → '.join(esc(m.removeprefix('argus.')) for m in c.get('members', []))}</li>"
            for c in blob.get("cycles") or [])
        packages = "".join(
            f"<tr><td>{esc(str(p.get('package')))}</td><td class='n'>{p.get('modules')}</td>"
            f"<td class='n'>{p.get('afferent')}</td><td class='n'>{p.get('efferent')}</td>"
            f"<td class='n'>{'—' if p.get('instability') is None else p.get('instability')}"
            f"</td></tr>"
            for p in sorted(blob.get("packages") or [],
                            key=lambda p: float(p.get("instability") or 0)))
        body = f"""
<p class="sub">{esc(_sentences(str(blob.get('verdict', ''))))}</p>
<div class="stats"><span><b>{blob.get('modules')}</b> modules</span>
<span><b>{blob.get('internal_imports')}</b> internal imports</span>
<span><b>{len(violations)}</b> contract violations</span>
<span><b>{blob.get('import_time_cycles')}</b> cycles that break at load</span></div>
<h2>Contracts</h2>{contract}
<h2>Layers</h2>{_layers()}
<h2>Dependency cycles</h2>
<p class="sub">Each is real coupling. "Safe at load" means one direction is a deferred import, so
the modules load in either order; a cycle that breaks at load would fail the check.</p>
<ul class="cyc">{cycles or '<li>none</li>'}</ul>
<h2>Packages</h2>
<p class="sub">Instability is efferent / (afferent + efferent): 0 is depended on and depends on
nothing, 1 the reverse. The core sits at the stable end on purpose.</p>
<table class="num"><thead><tr><th>Package</th><th>Modules</th><th>Used by</th><th>Uses</th>
<th>Instability</th></tr></thead><tbody>{packages}</tbody></table>
<p class="sub">Checked {esc(str(blob.get('generated_at', ''))[:16].replace('T', ' '))} UTC by
<a href="{REPOSITORY}src/argus/eval/architecture.py"><code>eval/architecture.py</code></a>, which
the public CI runs on every push; the artefact is
<a href="{REPOSITORY}data/architecture.json"><code>data/architecture.json</code></a>.</p>"""
    head = design.head("ARGUS — architecture, checked",
                       "The layer order, the contracts and every dependency cycle, read from the "
                       "check that runs on every push.", "/architecture")
    return f"""<!doctype html><html lang="en"><head>{head}
<style>{design.TOKENS_CSS}
 body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 system-ui,sans-serif }}
 .sub {{ margin:0 0 18px; color:var(--dim) }}
 h2 {{ font-size:17px; margin:28px 0 10px }}
 .ok {{ color:var(--good) }}
 .stats {{ display:flex; flex-wrap:wrap; gap:10px 22px; margin:0 0 8px; font-size:14px }}
 .stats b {{ font:600 17px var(--mono) }}
 table {{ border-collapse:collapse; width:100%; font-size:14px; margin-bottom:10px }}
 th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line) }}
 table.num td.n, table.num th:nth-child(n+2) {{ font-family:var(--mono); text-align:right }}
 table.text td.n {{ font-family:var(--mono); width:4em }}
 .cyc {{ padding-left:18px; font-size:14px }} .cyc li {{ margin:4px 0 }}
 .tag {{ font:12px var(--mono); text-transform:uppercase; letter-spacing:.08em;
   padding:1px 6px; border-radius:4px; border:1px solid var(--line) }}
 .tag.ok {{ color:var(--good) }} .tag.bad {{ color:var(--bad) }}
 code {{ font:12px var(--mono) }} a {{ color:var(--accent) }}
 @media (max-width:560px) {{ table {{ font-size:12.5px }} th, td {{ padding:5px 4px }} }}
{design.BASE_CSS}</style></head><body>{design.nav('/architecture')}<div class="wrap">
<h1>Architecture, checked</h1>
{body}
</div>{design.footer()}</body></html>"""


__all__ = ["load", "render"]
