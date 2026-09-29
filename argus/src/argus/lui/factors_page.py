"""The `/factors` page: every factor the lab has tested, where each one died, and why.

`research/factor_lab.py` certifies a factor with deterministic Python — gross and net Sharpe, the
out-of-sample check, the deflated Sharpe over every trial, split-half reliability — and wrote the
verdicts to `data/factor_lab.json`, which no page showed (research/harvest/36-ai-scientist.md).
SakanaAI's AI-Scientist ends its loop with a written paper and an LLM ensemble reviewing it; that
review is rejected here, because a sibling model grading a sibling's write-up is the self-graded
evaluation the lab was built to refuse, and the lab's own reasons are already sentences. What is
kept from it is the step after the gate: the result is written up for a reader, failures included.

Nothing is recomputed: the funnel, the gate and each factor's path are read from the artefact.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from argus.lui import design
from argus.lui.proof_page import REPOSITORY
from argus.truth.paths import DATA_DIR


def load(data_dir: Path) -> dict[str, Any] | None:
    try:
        blob = json.loads((data_dir / "factor_lab.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return blob if isinstance(blob, dict) else None


def _num(value: Any, places: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{places}f}"


COLUMNS = (("gross_sharpe", "Gross Sharpe"), ("net_sharpe", "Net Sharpe"),
           ("oos_sharpe", "Out of sample"), ("dsr", "Deflated"))


def _row(f: dict[str, Any]) -> str:
    esc = html.escape
    name, expression = str(f.get("name")), str(f.get("expression") or "")
    rule = f"<br><code>{esc(expression)}</code>" if expression not in ("", name) else ""
    cells = "".join(f"<td class='n' data-label='{label}'>{_num(f.get(key))}</td>"
                    for key, label in COLUMNS)
    state = esc(str(f.get("state")))
    return (f"<tr><td class='name'><b>{esc(name)}</b>{rule}</td>{cells}"
            f"<td class='stop'><span class='st {state}'>{state}</span>"
            f"<div class='why'>{esc(str(f.get('rejection_reason') or ''))}</div></td></tr>")


def _rival(data_dir: Path) -> str:
    """The same gates run on the specialist's catalogues (`eval/factor_quality_rivals.py`): the
    page said what ARGUS's own factors did and not how the leading factor miner's fared through
    the same gates (visual review, 2026-09-29)."""
    try:
        blob = json.loads((data_dir / "factor_quality_rivals.json").read_text(encoding="utf-8"))
        fam = blob["summary"]["by_family"]
        noise = blob["noise"]
    except (OSError, ValueError, KeyError, TypeError):
        return ""

    def cell(family: str) -> str:
        row = fam.get(family, {}).get("as_written", {})
        return f"{row.get('pass_all', '?')} of {row.get('pairs', '?')}"

    return (f'<p class="sub"><b>Against FactorMiner</b>, run in its own engine on the same bars '
            f"through the same gates: its paper factors pass all of them on {cell('paper')} "
            f"factor-instrument pairs, its Alpha101 set on {cell('alpha101')}, the adapted "
            f"variants on {cell('alpha101_adapted')}, and these rules on {cell('argus')}; coin "
            f"flips pass split-half alone {noise.get('passed')} times in {noise.get('of')}. "
            f'Neither side finds an edge these gates can see, so the row is TIED '
            f'(<a href="{REPOSITORY}data/factor_quality_rivals.json">'
            f"<code>data/factor_quality_rivals.json</code></a>).</p>")


def render(blob: dict[str, Any] | None) -> str:
    esc = html.escape
    if blob is None:
        body = ("<p class='sub'>data/factor_lab.json could not be read, so nothing is shown rather "
                "than something unchecked.</p>")
    else:
        funnel = blob.get("funnel") or {}
        gate = blob.get("gate") or {}
        certified = funnel.get("certified", 0)
        headline = (f"{funnel.get('proposed', 0)} factors proposed, {certified} certified, "
                    f"{funnel.get('rejected', 0)} rejected")
        rows = "".join(_row(f) for f in blob.get("factors") or [])
        body = f"""
<p class="sub">{esc(headline)}. Every factor passes the same gates in the same order — backtest,
costs, out of sample, the deflated Sharpe over all {esc(str(gate.get('trials', '?')))} trials,
split-half reliability — and is certified only if it survives all of them. A model may propose a
factor, and {esc(str(blob.get('separation', '')))}.</p>
<div class="stats"><span><b>{funnel.get('proposed', 0)}</b> proposed</span>
<span><b>{certified}</b> certified</span><span><b>{funnel.get('rejected', 0)}</b> rejected</span>
<span><b>{funnel.get('duplicates_suppressed', 0)}</b> re-proposals refused</span></div>
<div class="tbl"><table><thead><tr><th>Factor</th><th>Gross Sharpe</th><th>Net Sharpe</th>
<th>Out of sample</th><th>Deflated</th><th>Where it stopped</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p class="sub">A dash is a gate the factor never reached. Tested
{esc(str(blob.get('generated_at', ''))[:16].replace('T', ' '))} UTC by
<a href="{REPOSITORY}src/argus/research/factor_lab.py"><code>research/factor_lab.py</code></a>;
the record is <a href="{REPOSITORY}data/factor_lab.json"><code>data/factor_lab.json</code></a>.
</p>{_rival(DATA_DIR)}"""
    head = design.head("ARGUS — factors, tested",
                       "Every factor the lab tried, the gate each one stopped at, and why.",
                       "/factors")
    return f"""<!doctype html><html lang="en"><head>{head}
<style>{design.TOKENS_CSS}
 body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 system-ui,sans-serif }}
 .sub {{ margin:0 0 18px; color:var(--dim) }}
 .stats {{ display:flex; flex-wrap:wrap; gap:10px 22px; margin:0 0 16px; font-size:14px }}
 .stats b {{ font:600 17px var(--mono) }}
 .tbl {{ overflow-x:auto }}
 table {{ border-collapse:collapse; width:100%; font-size:14px }}
 th, td {{ text-align:left; padding:8px; border-bottom:1px solid var(--line); vertical-align:top }}
 th {{ font:600 11px var(--mono); text-transform:uppercase; letter-spacing:.06em;
      color:var(--dim) }}
 td.n {{ font-family:var(--mono); text-align:right; white-space:nowrap }}
 .st {{ font:10.5px var(--mono); text-transform:uppercase; letter-spacing:.08em }}
 .st.rejected {{ color:var(--bad) }} .st.certified {{ color:var(--good) }}
 .why {{ color:var(--dim); font-size:13px; margin-top:2px }}
 @media (max-width:640px) {{
  thead {{ display:none }}
  tr {{ display:grid; grid-template-columns:repeat(4,1fr); gap:2px 8px; padding:10px 0;
        border-bottom:1px solid var(--line) }}
  td {{ border:0; padding:2px 0 }}
  td.name, td.stop {{ grid-column:1 / -1 }}
  td.n {{ text-align:left }}
  td.n::before {{ content:attr(data-label); display:block; font:10px var(--mono);
                  text-transform:uppercase; letter-spacing:.06em; color:var(--dim) }}
 }}
 code {{ font:12px var(--mono); color:var(--dim) }} a {{ color:var(--accent) }}
{design.BASE_CSS}</style></head><body>{design.nav('/factors')}<div class="wrap">
<h1>Factors, tested</h1>
{body}
</div>{design.footer()}</body></html>"""


__all__ = ["load", "render"]
