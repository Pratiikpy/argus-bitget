"""Every comparison against a named rival, grouped by the sub-theme Bitget names, with a way in.

**Why this page exists.** The standing register (`eval/standing.py`) holds every capability ARGUS
has measured against a named specialist, and its artefact `data/standing.json` said which were
won, tied or lost. A judge could reach none of it from the demo: the console answered questions,
`/wrong` listed the losses, and the wins were a JSON file in a repository. An audit (2026-09-24)
put it plainly — a win a judge cannot see is scored as if it did not exist.

**Read from the register at request time, never restated.** Every state, rival and test sentence
on the page is the register's own text, so the page cannot claim more than the register does, and
the register cannot declare OWNED without all thirteen conditions (it raises at import).

**The one hand-kept table is ``IN_THE_CONSOLE``**: which question, asked in the console, shows the
capability working — its own code run on the question, or its output from the desk's record, and
the measurement that makes it a win. Each entry was checked by asking it (`tests/test_proof_page.py`
pins the routing), and a capability without one says so on the page rather than hiding the gap —
a measured engine the console never calls is a finding in its own right.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

FAVICON = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' rx='7' fill='%231a5fb4'/%3E"
    "%3Ctext x='16' y='23' font-family='ui-monospace,monospace' font-size='18' "
    "font-weight='700' fill='%23fff' text-anchor='middle'%3EA%3C/text%3E%3C/svg%3E"
)
"""Duplicated from `lui/server.py` for the reason `corrections_page.py` gives: the server imports
this module lazily, and an import in the other direction is avoidable risk for six lines."""

REPOSITORY = "https://github.com/Pratiikpy/argus-bitget/blob/main/argus/"

SUBTHEMES: dict[str, tuple[str, str]] = {
    "t2-event": ("Track 2", "Event-Driven Agent"),
    "t2-sentiment": ("Track 2", "Market Sentiment Agent"),
    "t2-earnings": ("Track 2", "Earnings-Driven Trading Agent"),
    "t2-crossexecution": ("Track 2", "Cross-Asset Execution Agent"),
    "t2-factordiscovery": ("Track 2", "Factor Discovery Agent"),
    "t2-agentic": ("Track 2", "Judged: Agent architecture quality"),
    "t2-riskcontrol": ("Track 2", "Judged: risk control layer effectiveness"),
    "t2-explainability": ("Track 2", "Judged: decision explainability"),
    "t2-execution": ("Track 2", "Judged: execution realism behind the paper record"),
    "t3-infoextract": ("Track 3", "Information Extraction & Signal Generation"),
    "t3-review": ("Track 3", "Review & Self-Evolution"),
    "t3-decisionstress": ("Track 3", "Decision Stress Testing"),
    "t3-workbench": ("Track 3", "Personalized Research Workbench"),
    "t3-personalisation": ("Track 3", "Personalized Research Workbench"),
    "t3-execution": ("Track 3", "Execution Assistance"),
    "t3-execassist": ("Track 3", "Execution Assistance"),
    "t3-datasources": ("Track 3", "Judged: data sources and Skill integration"),
    "t3-lui": ("Track 3", "Judged: LUI fluency"),
    "t3-portfolio": ("Track 3", "Open Theme: portfolio tools"),
    "t1-arbitrage": ("Track 1", "Arbitrage"),
    "t1-afterhours": ("Track 1", "After-Hours Information Pricing"),
    "t1-crossmarket": ("Track 1", "Cross-Market Correlation Strategies"),
    "t1-rtokenfactor": ("Track 1", "rToken Factor Strategies"),
    "t1-rotation": ("Track 1", "Cross-Asset Allocation / Rotation"),
    "t1-alphafactory": ("Track 1", "Factor research (validation machinery)"),
    "t1-validation": ("Track 1", "Factor research (validation machinery)"),
}
"""The register's sub-theme codes, named as `BITGET_AI_BASE_CAMP_S2_HANDBOOK_EN.md` names them
(Chapter IV's three sub-theme tables and each track's judging focus)."""

TRACK_ORDER = ("Track 2", "Track 3", "Track 1")
TRACK_NOTE = {
    "Track 2": "AI Trading Desk: the LLM decides and the agent trades. Judged half on paper "
               "Sharpe, drawdown and win rate, half on explainability, architecture and the risk "
               "layer.",
    "Track 3": "AI research workbench for a human trader, judged on feature depth, research "
               "quality, LUI fluency and a personalised thesis.",
    "Track 1": "Not entered: Track 1 is scored on strategy returns alone. These capabilities sit "
               "underneath the other two and were measured the same way.",
}

IN_THE_CONSOLE: dict[str, tuple[str, str]] = {
    "Funding-aware cross-asset hedge routing vs. a fee-blind composite router":
        ("hedge my crypto after a macro shock", "60% BTC, 40% ETH"),
    "Refusal-first earnings surprise ranking vs. a silently-exploding factor":
        ("did NVDA beat last quarter", ""),
    "Risk layer proved by domain sweep": ("what did the risk layer block", ""),
    "Abstention scored as a decision": ("why did you do nothing all weekend", ""),
    "Pre-registered trading protocol, hash-committed": ("is the log tamper-evident", ""),
    "Self-evolving review rules": ("what bad decision patterns do you have", ""),
    "LUI intent routing, measured against Rasa's real DIET classifier":
        ("wut abt nvda earnigns when", ""),
    "Market sentiment": ("is the hype on NVDA real", ""),
    "Sentiment integrity: resistance to coordinated posting, vs. finBERT":
        ("is the hype on NVDA real", ""),
    "Clustering-corrected, base-rate-honest event significance vs. a fixed-null test":
        ("how does NVDA react to CPI", ""),
    "Path-shape matching with a calibrated null": ("has NVDA been here before", ""),
    "Session-aware execution that refuses to solve through a boundary":
        ("how should I split a $250k order in NVDA", ""),
    "Decision-latency pricing vs. hftbacktest's real, network-only LatencyModel":
        ("how should I split a $250k order in NVDA", ""),
    "Per-profile mandate that changes the verdict":
        ("I'm an aggressive trader, should I buy 10% COIN", ""),
    "Structured filing extraction vs. FinanceBench's real, published LLM measurement":
        ("what was NVDA's revenue last quarter", ""),
    "Point-in-time correctness vs. OpenBB's real, ungated live-API agent":
        ("what was NVDA's net income as of 1 March 2026", ""),
    "Numeric decision grounding vs. TradingAgents' real, unchecked TraderProposal":
        ("why did you pass on NVDA", ""),
    "Episodic memory across decisions": ("why did you pass on NVDA", ""),
    "Perception layer: what the desk can see": ("/status", ""),
    "Analogue stress bands vs. AnalogDesk (S2) on its own pre-registered test":
        ("has NVDA been here before", ""),
}
"""Capability name -> (question, saved book) whose console answer shows the capability. A value
starting with "/" is a page of the console rather than a question."""


@dataclass(frozen=True, slots=True)
class Win:
    """One capability as the page shows it."""

    name: str
    state: str
    track: str
    subtheme: str
    rival: str
    tested: str
    conditions: int
    artefacts: tuple[str, ...]
    tests: tuple[str, ...]
    rerun: str
    console: tuple[str, str] | None
    proofs: tuple[tuple[str, str], ...]
    note: str
    blockers: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "state": self.state, "track": self.track,
            "subtheme": self.subtheme, "rival": self.rival, "tested": self.tested,
            "conditions_met": self.conditions, "artefacts": list(self.artefacts),
            "tests": list(self.tests), "rerun": self.rerun,
            "console": ({"question": self.console[0], "book": self.console[1]}
                        if self.console else None),
        }


def _first_sentence(text: str, limit: int = 320) -> str:
    """The register's own words, cut at the first sentence end — never rewritten."""
    cut = re.split(r"(?<=[.;])\s+(?=[A-Z(])", text.strip(), maxsplit=1)[0]
    if len(cut) <= limit:
        return cut
    return cut[:limit].rsplit(" ", 1)[0] + "…"


def _rival(baseline: str, limit: int = 220) -> str:
    """Every rival the register names. A semicolon there separates rivals, not sentences — cut at
    it, "ProsusAI/finBERT for classification; BloombergGPT and FinMA for the published bar" lost
    two of its three (2026-09-24)."""
    text = baseline.split(" — ")[0].strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _rerun(module: str) -> str:
    """The comparison module, as the command a reader runs to reproduce the measurement."""
    for part in module.split(","):
        found = re.fullmatch(r"argus/eval/(\w+_comparison|riskproof|queueproof)\.py", part.strip())
        if found:
            return f"python -m argus.eval.{found.group(1)}"
    return ""


def collect(data_dir: Path) -> tuple[list[Win], dict[str, int]]:
    """Every capability in the register, and the count per state. Empty when unreadable."""
    try:
        report = json.loads((data_dir / "standing.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], {}
    wins: list[Win] = []
    for cap in report.get("capabilities", []):
        proofs = cap.get("proofs", [])
        same_input = next((p["how"] for p in proofs
                           if p.get("condition") == "same_input_comparison"), "")
        track, subtheme = SUBTHEMES.get(cap.get("subtheme", ""), ("Other", cap.get("subtheme", "")))
        wins.append(Win(
            name=cap["name"], state=cap["state"], track=track, subtheme=subtheme,
            rival=_rival(cap.get("baseline", "")),
            tested=_first_sentence(same_input) if same_input else "",
            conditions=len(cap.get("conditions_met", [])),
            artefacts=tuple(dict.fromkeys(
                p["artefact"] for p in proofs
                if p.get("artefact", "").startswith("data/"))),
            tests=tuple(dict.fromkeys(
                p["test"].split("::")[0] for p in proofs if p.get("test"))),
            rerun=_rerun(cap.get("module", "")),
            console=IN_THE_CONSOLE.get(cap["name"]),
            proofs=tuple((p["condition"], p["how"]) for p in proofs),
            note=cap.get("note", ""),
            blockers=tuple(str(b) for b in cap.get("blockers", [])),
        ))
    return wins, dict(report.get("by_state", {}))


def _console_link(question: str, book: str) -> str:
    params = {"q": question, **({"book": book} if book else {})}
    return "/?" + urlencode(params)


def _card(win: Win) -> str:
    esc = html.escape
    links: list[str] = []
    if win.console and win.console[0].startswith("/"):
        links.append(f"<a class='live' href='{esc(win.console[0])}'>"
                     f"Open {esc(win.console[0])} — it runs live</a>")
    elif win.console:
        question, book = win.console
        links.append(f"<a class='live' href='{esc(_console_link(question, book))}'>"
                     f"Ask the console: “{esc(question)}”</a>")
    else:
        links.append("<span class='gap'>Not yet reachable from the console</span>")
    links.extend(f"<a href='{REPOSITORY}{esc(a)}'>{esc(a.removeprefix('data/'))}</a>"
                 for a in win.artefacts[:2])
    links.extend(f"<a href='{REPOSITORY}tests/{esc(t)}'>{esc(t)}</a>" for t in win.tests[:1])
    rerun = f"<code class='cmd'>{esc(win.rerun)}</code>" if win.rerun else ""
    proofs = "".join(f"<li><b>{esc(c.replace('_', ' '))}</b> — {esc(h)}</li>"
                     for c, h in win.proofs)
    note = f"<p class='note'>{esc(win.note)}</p>" if win.note else ""
    tested = (f"<p><span class='lbl'>Same-input test</span>{esc(win.tested)}</p>"
              if win.tested else "")
    if win.state != "owned" and win.blockers:
        reason = win.blockers[0]
        if len(reason) > 900:
            reason = reason[:900].rsplit(" ", 1)[0] + "…"
        tested += (f"<p class='why'><span class='lbl'>Why not OWNED</span>{esc(reason)}</p>")
    return (
        f"<article class='w {esc(win.state)}'>"
        f"<div class='head'><span class='st {esc(win.state)}'>{esc(win.state.upper())}</span>"
        f"<span class='cn'>{win.conditions} of 13 conditions"
        + (" — against a rival that does not lead the sub-theme"
           if win.state != "owned" and win.conditions == 13 else "")
        + "</span></div>"
        f"<h3>{esc(win.name)}</h3>"
        f"<p><span class='lbl'>Rival</span>{esc(win.rival)}</p>{tested}"
        f"<div class='links'>{' '.join(links)}</div>{rerun}"
        f"<details><summary>Every condition and its evidence</summary><ul>{proofs}</ul>"
        f"{note}</details></article>"
    )


def render(wins: list[Win], counts: dict[str, int]) -> str:
    esc = html.escape
    if not wins:
        body = ("<p class='sub'>The standing register could not be read, so nothing is shown "
                "rather than something unverified.</p>")
        headline = "The standing register is unreadable"
    else:
        order = ("owned", "tied", "implemented", "lost")
        headline = (f"{len(wins)} capabilities measured against a named rival: "
                    + ", ".join(f"{counts.get(s, 0)} {s.upper()}" for s in order))
        reachable = sum(1 for w in wins if w.console)
        sections: list[str] = []
        for track in TRACK_ORDER:
            mine = [w for w in wins if w.track == track]
            if not mine:
                continue
            # Bitget's named sub-themes first, in the handbook's order, then the judged criteria.
            rank = {name: i for i, (_, name) in enumerate(SUBTHEMES.values())}
            by_theme: dict[str, list[Win]] = {}
            for win in sorted(mine, key=lambda w: rank.get(w.subtheme, len(rank))):
                by_theme.setdefault(win.subtheme, []).append(win)
            themes = "".join(
                f"<h3 class='theme'>{esc(theme)}</h3>" + "".join(_card(w) for w in items)
                for theme, items in by_theme.items())
            sections.append(f"<section><h2>{esc(track)}</h2>"
                            f"<p class='sub'>{esc(TRACK_NOTE[track])}</p>{themes}</section>")
        body = (f"<p class='sub'>{reachable} of {len(wins)} run live from a question in the "
                f"console; the rest are named as gaps below. Losses and withdrawn claims are on "
                f"<a href='/wrong'>what we got wrong</a>.</p>" + "".join(sections))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="{FAVICON}">
<title>ARGUS — measured against the specialists</title>
<style>
 :root {{ --ink:#12161c; --dim:#5b6470; --line:#dfe3e8; --bg:#f7f8fa; --panel:#fff;
   --accent:#1a5fb4; --warn:#8a4b00; --ok:#0f6b3f; --bad:#9b3a2f;
   --mono:ui-monospace,"SF Mono",Menlo,monospace; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --ink:#e6e9ee; --dim:#98a2b0; --line:#2a313b;
   --bg:#0f1318; --panel:#161b22; --accent:#7aa7ea; --warn:#e0a15a; --ok:#5fd39a;
   --bad:#d9796a; }} }}
 * {{ box-sizing:border-box }}
 body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 system-ui,sans-serif }}
 .wrap {{ max-width:820px; margin:0 auto; padding:30px 18px 70px }}
 h1 {{ font-size:22px; margin:0 0 6px; letter-spacing:-.015em; text-wrap:balance }}
 h2 {{ font-size:18px; margin:34px 0 4px }}
 h3.theme {{ font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--dim);
   margin:22px 0 8px; font-weight:600 }}
 .sub {{ color:var(--dim); font-size:13.5px; margin:0 0 14px; max-width:68ch }}
 .w {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
   padding:14px 16px; margin-bottom:12px }}
 .w h3 {{ font-size:15.5px; margin:6px 0 8px; line-height:1.35 }}
 .w p {{ margin:0 0 8px; font-size:14px }}
 .head {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap }}
 .st {{ font:600 10.5px var(--mono); letter-spacing:.08em; padding:2px 7px; border-radius:4px;
   border:1px solid currentColor }}
 .st.owned {{ color:var(--ok) }} .st.tied {{ color:var(--accent) }}
 .st.implemented {{ color:var(--warn) }} .st.lost {{ color:var(--bad) }}
 .cn {{ font:11px var(--mono); color:var(--dim) }}
 .lbl {{ display:inline-block; min-width:118px; font:11px var(--mono); color:var(--dim);
   text-transform:uppercase; letter-spacing:.06em }}
 .links {{ display:flex; gap:6px 14px; flex-wrap:wrap; font-size:13px; margin:4px 0 6px }}
 .links a {{ color:var(--accent); overflow-wrap:anywhere }}
 .links a.live {{ font-weight:600 }}
 .gap {{ color:var(--warn); font-size:13px }}
 .cmd {{ display:block; font:12px var(--mono); color:var(--dim); margin:2px 0 6px;
   overflow-x:auto; white-space:nowrap }}
 details {{ font-size:13px; color:var(--dim) }}
 summary {{ cursor:pointer; color:var(--accent) }}
 details ul {{ padding-left:18px; margin:8px 0 }}
 details li {{ margin-bottom:6px; overflow-wrap:anywhere }}
 details b {{ color:var(--ink); font-weight:600 }}
 .note {{ font-size:12.5px; overflow-wrap:anywhere }}
 .why {{ color:var(--warn) }}
 a {{ color:var(--accent) }}
 a:focus-visible, summary:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px }}
 @media (max-width:520px) {{ .lbl {{ display:block; min-width:0 }} }}
</style></head><body><div class="wrap">
<h1>{esc(headline)}</h1>
<p class="sub">OWNED means all thirteen conditions hold: the rival's best implementation read,
its method reproduced, both run on the same input, a statistically valid evaluation with costs,
out-of-sample, ablation, an adversarial test, failure cases, reproducibility, and no specialist
capability left superior without a stated reason. The register refuses to load an OWNED claim
that is missing one. Everything below is read from that register when this page loads.
<a href="/">Back to the console</a></p>
{body}
</div></body></html>"""


__all__ = ["IN_THE_CONSOLE", "SUBTHEMES", "Win", "collect", "render"]
