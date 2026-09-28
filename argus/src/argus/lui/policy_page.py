"""The `/policy` page: the desk's risk limits as the file that sets them, and which could fire.

Track 2 judges "risk control layer effectiveness". A limit is effective only if it is both set
and fed its input, and until 2026-09-28 neither was visible: the thresholds lived as dataclass
defaults in `risk/constitution.py`, and nothing said that seven of the ten limits could not fire
in the live cycle, three of them only because the book was never handed to the Constitution.
`risk/policy_manifest.py` made the policy a file (`data/risk_policy.json`,
research/harvest/26-opa-rule-gate.md) and has each cycle write what it ran under
(`data/risk_policy_live.json`); this page draws both, as they stand.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from argus.lui import design
from argus.lui.proof_page import REPOSITORY
from argus.risk.policy_manifest import PARAMETERS


def _read(path: Path) -> dict[str, Any] | None:
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return blob if isinstance(blob, dict) else None


def load(data_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    return _read(data_dir / "risk_policy.json"), _read(data_dir / "risk_policy_live.json")


def _value(value: Any, unit: str) -> str:
    if value is None or value == {}:
        return "off"
    if isinstance(value, dict):
        return ", ".join(f"{k} {v:g}" for k, v in value.items())
    if unit == "USD":
        return f"${float(value):,.0f}"
    return f"{value} {unit}" if unit not in ("ratio", "probability", "fraction") else f"{value}"


def render(policy: dict[str, Any] | None, live: dict[str, Any] | None) -> str:
    esc = html.escape
    if policy is None:
        body = ("<p class='sub'>data/risk_policy.json could not be read, so no limits are shown "
                "rather than limits that were not checked.</p>")
    else:
        fired: set[str] = set()
        symbols = 0
        if live:
            per = live.get("per_symbol") or {}
            symbols = len(per)
            for row in per.values():
                fired.update(row.get("can_fire") or [])
        rows = []
        for name, entry in (policy.get("thresholds") or {}).items():
            unit = str(entry.get("unit", ""))
            meaning = str(entry.get("meaning", ""))
            if name in PARAMETERS:
                state = "<span class='st'>parameter</span>"
            elif not live:
                state = "<span class='st'>no cycle recorded</span>"
            elif name in fired:
                state = "<span class='st on'>can fire</span>"
            else:
                state = "<span class='st off'>cannot fire</span>"
            rows.append(f"<tr><td><b>{esc(name.replace('_', ' '))}</b>"
                        f"<div class='why'>{esc(meaning)}</div></td>"
                        f"<td class='n'>{esc(_value(entry.get('value'), unit))}</td>"
                        f"<td>{state}</td></tr>")
        armed = len(fired)
        total = len([n for n in (policy.get("thresholds") or {}) if n not in PARAMETERS])
        when = esc(str((live or {}).get("at", ""))[:16].replace("T", " "))
        head_line = (f"{armed} of {total} limits could fire in the last cycle ({when} UTC, "
                     f"{symbols} symbol{'' if symbols == 1 else 's'}); the others are off or "
                     f"lack the input they read."
                     if live else f"{total} limits; no cycle has recorded which could fire yet.")
        body = f"""
<p class="sub">{esc(head_line)} The limits are the file below, not code: the paper cycle reads it
each run, an edit takes effect only once sealed, and a file that does not load stops the cycle
deciding anything rather than falling back to a guess. Policy digest
<code>{esc(str(policy.get('digest', ''))[:16])}</code>.</p>
<div class="tbl"><table><thead><tr><th>Limit</th><th>Value</th><th>Last cycle</th></tr>
</thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="sub">Read from
<a href="{REPOSITORY}data/risk_policy.json"><code>data/risk_policy.json</code></a> and
<a href="{REPOSITORY}data/risk_policy_live.json"><code>data/risk_policy_live.json</code></a>;
enforced by
<a href="{REPOSITORY}src/argus/risk/constitution.py"><code>risk/constitution.py</code></a>.</p>"""
    head = design.head("ARGUS — risk policy",
                       "The desk's risk limits as the file that sets them, and which could fire.",
                       "/policy")
    return f"""<!doctype html><html lang="en"><head>{head}
<style>{design.TOKENS_CSS}
 body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 system-ui,sans-serif }}
 .sub {{ margin:0 0 18px; color:var(--dim) }}
 .tbl {{ overflow-x:auto }}
 table {{ border-collapse:collapse; width:100%; font-size:14px }}
 th, td {{ text-align:left; padding:8px; border-bottom:1px solid var(--line); vertical-align:top }}
 th {{ font:600 11px var(--mono); text-transform:uppercase; letter-spacing:.06em;
      color:var(--dim) }}
 td.n {{ font-family:var(--mono); white-space:nowrap }}
 .st {{ font:10.5px var(--mono); text-transform:uppercase; letter-spacing:.08em;
       color:var(--dim) }}
 .st.on {{ color:var(--good) }} .st.off {{ color:var(--bad) }}
 .why {{ color:var(--dim); font-size:13px; margin-top:2px }}
 code {{ font:12px var(--mono); color:var(--dim) }} a {{ color:var(--accent) }}
 @media (max-width:640px) {{
  thead {{ display:none }}
  tr {{ display:grid; grid-template-columns:1fr auto; gap:2px 12px; padding:10px 0;
        border-bottom:1px solid var(--line) }}
  td {{ border:0; padding:0 }} td:first-child {{ grid-column:1 / -1 }}
 }}
{design.BASE_CSS}</style></head><body>{design.nav('/policy')}<div class="wrap">
<h1>Risk policy</h1>
{body}
</div>{design.footer()}</body></html>"""


__all__ = ["load", "render"]
