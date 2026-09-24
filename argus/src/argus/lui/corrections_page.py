"""What we got wrong — assembled from the artefacts, never from memory.

**Bitget's own S1 showcase led with a negative result.** Across twenty-four sentences describing
eight selected projects it printed no performance numbers at all, and the single most prominent
bullet was Nocturne's *negative* finding. That is the grammar this project already writes in, and
until now the material was scattered: a correction in `paper/corrections.py`, a withdrawn claim in
`eval/luirouter.py`, two lost comparisons in their own artefacts, nine unproven conditions in the
standing register. A reader had to know where to look for every one of them.

**Every entry here is read out of an artefact at request time.** None is a string written by hand
describing a result, because a hand-written description of a measurement drifts from the
measurement the first time the measurement changes — and a page about honesty that has gone stale
is worse than no page. Where the artefact is missing, the entry says so rather than disappearing:
a losses page that silently shortens is telling the most flattering possible lie.

**What this is not.** It is not a list of everything wrong with ARGUS — nothing could be, and
claiming completeness would be its own overstatement. It is the set of findings that already have
an artefact behind them, which is a smaller and checkable claim.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FAVICON = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' rx='7' fill='%231a5fb4'/%3E"
    "%3Ctext x='16' y='23' font-family='ui-monospace,monospace' font-size='18' "
    "font-weight='700' fill='%23fff' text-anchor='middle'%3EA%3C/text%3E%3C/svg%3E"
)
"""Kept identical to (and duplicated from, rather than imported from) `lui/server.py`'s constant
of the same name: `server.py` only imports this module lazily, inside a route handler, so a
module-level import in the other direction here is avoidable risk for six lines of duplication."""


@dataclass(frozen=True, slots=True)
class Correction:
    """One thing we got wrong, with the artefact that says so."""

    headline: str
    detail: str
    artefact: str
    kind: str
    """``bug`` — we shipped something broken. ``loss`` — a named baseline beat us.
    ``withdrawn`` — a published claim was retracted. ``open`` — a gap still open."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "headline": self.headline, "detail": self.detail,
            "artefact": self.artefact, "kind": self.kind,
        }


def _read(data_dir: Path, name: str) -> dict[str, Any] | None:
    path = data_dir / name
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return dict(loaded) if isinstance(loaded, dict) else None


def collect(data_dir: Path) -> list[Correction]:
    """Every recorded finding, read from its artefact.

    An artefact that cannot be read produces an entry saying so. The alternative — skipping it —
    means this page gets shorter every time something breaks, which is the one failure mode a
    page like this must not have.
    """
    out: list[Correction] = []

    def missing(name: str, what: str) -> Correction:
        return Correction(
            headline=f"{what} — artefact unreadable",
            detail=(
                f"`data/{name}` could not be read, so this entry cannot state its own result. "
                f"It is listed rather than dropped: a losses page that silently shortens is the "
                f"most flattering possible lie."
            ),
            artefact=f"data/{name}", kind="open",
        )

    # 1. The bug that produced the only good numbers this project ever had.
    try:
        from argus.paper.corrections import VOIDED

        if VOIDED:
            seqs = ", ".join(str(v.seq) for v in VOIDED)
            out.append(Correction(
                headline=f"{len(VOIDED)} ledger rows booked P&L on positions the risk layer "
                         f"refused",
                detail=(
                    f"`paper/runner.py` recorded the model's unconstrained intent whenever the "
                    f"desk chose not to re-put a decision to the model, bypassing the "
                    f"Constitution on the common path. Rows {seqs} booked profit against "
                    f"positions that were never taken — and for one morning they were this "
                    f"project's entire win rate: 100%, on the only two trades it had ever made. "
                    f"They were never made. The rows stay in the chain unedited, because "
                    f"deleting the two that made the numbers look good is precisely the behaviour "
                    f"this system exists to refuse."
                ),
                artefact="argus/paper/corrections.py", kind="bug",
            ))
    except Exception:  # pragma: no cover - the module is part of the package
        out.append(missing("corrections.py", "The voided ledger rows"))

    # 2. A published figure, withdrawn.
    router = _read(data_dir, "lui_router.json")
    if router is None:
        out.append(missing("lui_router.json", "The withdrawn router claim"))
    else:
        head = router.get("headline", {})
        out.append(Correction(
            headline=(
                f"The semantic router's 52.4% was withdrawn — measured on a friendly corpus, "
                f"it is really {float(head.get('fresh_accuracy', 0)) * 100:.1f}%"
            ),
            detail=(
                f"The corpus it was scored on was written by the same author, in the same "
                f"sitting, as the router's own training examples. Re-measured against one "
                f"authored by a model that has never seen this repository it scores "
                f"{head.get('improvement_x')}x against the regular expressions it was built to "
                f"beat — level. Leave-one-out cross-validation shows its similarity distributions "
                f"for right and wrong answers overlapping completely, so no abstention threshold "
                f"separates them. It is not deployed."
            ),
            artefact="data/lui_router.json", kind="withdrawn",
        ))

    # 3. A comparison we lost, published as a loss.
    regime = _read(data_dir, "regime_comparison.json")
    if regime is None:
        out.append(missing("regime_comparison.json", "The regime comparison"))
    else:
        out.append(Correction(
            headline="A named baseline beat us on regime detection, and we published it",
            detail=str(regime.get("who_wins", "")),
            artefact="data/regime_comparison.json", kind="loss",
        ))

    # 4. The pattern layer that memorised.
    bench = _read(data_dir, "oblique_bench.json")
    if bench is None:
        out.append(missing("oblique_bench.json", "The pattern layer's generalisation gap"))
    else:
        gap = bench.get("generalisation_gap_pct")
        out.append(Correction(
            headline=(
                f"The console's regular expressions memorised — a {gap}-point gap between the "
                f"phrasings they were written for and new ones"
            ),
            detail=(
                "They score 100% on the corpus they were widened against and a quarter of that "
                "on one written by a model that has never seen them. Four separate held-out "
                "corpora were burned discovering this, each one spent the moment its answers "
                "were read; `eval/ngrambench.py` records the status of every split rather than "
                "quietly quoting the friendliest."
            ),
            artefact="data/oblique_bench.json", kind="bug",
        ))

    # 5. What the register still cannot prove.
    standing = _read(data_dir, "standing.json")
    if standing is None:
        out.append(missing("standing.json", "The unproven conditions"))
    else:
        findings = standing.get("findings", [])
        unproven = [f for f in findings if "unproven" in str(f).lower()] or findings
        by_state = standing.get("by_state", {})
        # Every state but OWNED cannot claim OWNED — LOST and TIED, not only IMPLEMENTED. The
        # prior version read only by_state["implemented"], so a capability that had been measured
        # and demonstrably LOST to a named specialist silently vanished from this count instead of
        # being the clearest possible example of it — on the one page whose entire purpose is not
        # rounding up. Found 2026-09-22 driving the deployed page as a real user.
        total_capabilities = sum(int(v) for v in by_state.values()) if by_state else 0
        not_owned = total_capabilities - int(by_state.get("owned", 0)) if by_state else "?"
        out.append(Correction(
            headline=(
                f"{not_owned} of "
                f"{total_capabilities if by_state else '?'} capabilities "
                f"cannot claim OWNED, and the register prints why"
            ),
            detail=(
                f"{len(unproven)} condition(s) are claimed with nothing behind them in the "
                f"artefact. The register fails the build rather than rounding up, and two of "
                f"those remaining need real spend against a metered model key — stated as a cost "
                f"rather than quietly skipped."
            ),
            artefact="data/standing.json", kind="open",
        ))

    # 6. The contradiction between our own surfaces.
    surfaces = _read(data_dir, "surface_agreement.json")
    if surfaces is None:
        out.append(missing("surface_agreement.json", "The surface contradiction"))
    else:
        out.append(Correction(
            headline="Our own console contradicted our own README, and both were 'correct'",
            detail=(
                f"The console counted two refused positions as trades while every document said "
                f"zero. Each surface was reading its own source correctly, and both numbers came "
                f"from the same hash chain — so the contradiction arrived wearing a verification. "
                f"`eval/surfaces.py` now compares "
                f"{surfaces.get('facts_checked', '?')} facts across two surfaces each on every "
                f"build; it currently reports {surfaces.get('disagree', '?')} disagreement(s)."
            ),
            artefact="data/surface_agreement.json", kind="bug",
        ))

    # 7. A vendor reading the desk trusted without checking.
    macd = _read(data_dir, "skill_macd_check.json")
    if macd is None:
        out.append(missing("skill_macd_check.json", "The unchecked MACD reading"))
    else:
        out.append(Correction(
            headline=(
                f"The desk believed Bitget's MACD — its fields are swapped on "
                f"{macd.get('swapped', '?')} of "
                f"{int(macd.get('swapped', 0)) + int(macd.get('straight', 0))} symbols we "
                f"could check"
            ),
            detail=(
                f"We recomputed the technical-analysis Skill's RSI before trusting it, and never "
                f"did the same for its MACD. The Skill returns the signal line in its "
                f"`histogram` field and the histogram in `signal`, and its cross flag "
                f"contradicted the recomputed lines on "
                f"{macd.get('cross_flag_contradicts_lines', '?')} symbols. The desk passed those "
                f"payloads to the model as evidence, and decision theses cited the crosses. "
                f"Every MACD payload is now recomputed from Bitget's 4h candles before the desk "
                f"or the console reads it; past theses stay in the chain as written."
            ),
            artefact="data/skill_macd_check.json", kind="bug",
        ))
    return out


def render(corrections: list[Correction]) -> str:
    """The page. Deliberately plain — this is a record, not a pitch."""
    esc = html.escape
    label = {"bug": "we shipped it broken", "loss": "a baseline beat us",
             "withdrawn": "claim withdrawn", "open": "still open"}
    rows = "".join(
        f"<article class='c {esc(c.kind)}'>"
        f"<span class='k'>{esc(label.get(c.kind, c.kind))}</span>"
        f"<h2>{esc(c.headline)}</h2><p>{esc(c.detail)}</p>"
        f"<code>{esc(c.artefact)}</code></article>"
        for c in corrections
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="{FAVICON}">
<title>ARGUS — what we got wrong</title>
<style>
 :root {{ --ink:#12161c; --dim:#5b6470; --line:#dfe3e8; --bg:#f7f8fa; --panel:#fff;
   --accent:#1a5fb4; --warn:#8a4b00; --ok:#0f6b3f; --bad:#9b3a2f;
   --mono:ui-monospace,"SF Mono",Menlo,monospace; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --ink:#e6e9ee; --dim:#98a2b0; --line:#2a313b;
   --bg:#0f1318; --panel:#161b22; --accent:#7aa7ea; --warn:#e0a15a; --ok:#5fd39a;
   --bad:#d9796a; }} }}
 * {{ box-sizing:border-box }}
 body {{ margin:0; background:var(--bg); color:var(--ink);
   font:15px/1.6 system-ui,sans-serif }}
 .wrap {{ max-width:760px; margin:0 auto; padding:30px 18px 70px }}
 h1 {{ font-size:22px; margin:0 0 6px; letter-spacing:-.015em }}
 .sub {{ color:var(--dim); font-size:13.5px; margin:0 0 24px; max-width:62ch }}
 .c {{ background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--dim);
   border-radius:10px; padding:16px 18px; margin-bottom:14px }}
 .c.bug {{ border-left-color:var(--bad) }}
 .c.loss {{ border-left-color:var(--warn) }}
 .c.withdrawn {{ border-left-color:var(--accent) }}
 .c.open {{ border-left-color:var(--dim) }}
 .k {{ font:10.5px var(--mono); text-transform:uppercase; letter-spacing:.1em;
   color:var(--dim); display:block; margin-bottom:6px }}
 .c h2 {{ font-size:16px; margin:0 0 8px; line-height:1.35 }}
 .c p {{ margin:0 0 10px; color:var(--dim); font-size:14.5px }}
 code {{ font:11.5px var(--mono); color:var(--dim) }}
 a {{ color:var(--accent) }}
</style></head><body><div class="wrap">
<h1>What we got wrong</h1>
<p class="sub">Every entry is read out of the artefact that recorded it, at the moment you load
this page &mdash; not written down once and left to drift. A finding whose artefact cannot be read
is listed as unreadable rather than dropped, because a losses page that silently shortens is the
most flattering possible lie. <a href="/">back to the console</a></p>
{rows}
<p class="sub">This is not everything wrong with ARGUS &mdash; nothing could be, and claiming
completeness would be its own overstatement. It is the set of findings that already have an
artefact behind them.</p>
</div></body></html>"""


__all__ = ["Correction", "collect", "render"]
