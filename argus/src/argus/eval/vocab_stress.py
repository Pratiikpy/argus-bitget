"""Rename the company, keep the numbers: does the decision depend on who it is about?

A trading decision should follow from the figures and the events, not from the brand. If the desk
stands aside on NVIDIA and would trade the identical tape under a meaningless name, it was deciding
on a prior about NVIDIA that nothing in the frame supports — the pattern the PlanBench authors found
in planning models, which solved Blocksworld and failed the same problem once its vocabulary was
replaced by nonsense ("Mystery Blocksworld").

**Taken from PlanBench** (``karthikv792/LLMs-Planning``, MIT, confirmed through the GitHub licence
API on 2026-09-25): ``plan-bench/obfuscator.py:31-52``, ``random_mapping``. Each name in the domain
vocabulary is mapped to a word drawn with ``random.choice`` from a supplied list of new words, and
the drawn word is removed from the list so two names never collide. :func:`random_mapping` keeps
that exactly, with one change: the generator is a seeded ``random.Random`` rather than the module
global, so a mapping is reproducible per snapshot. PlanBench's ``"Not enough words provided"`` path
printed and returned ``None``; here it raises, because a half-obfuscated frame is not a result.

**What PlanBench renames, and what corresponds to it here.** PlanBench renames actions and
predicates, the words a planner reasons with, and holds the problem structure fixed. The analogue in
a trading frame has two layers, and they answer different questions, so they are scored apart:

* ``entities`` — the rToken symbol, the underlying ticker and the issuer's names. **Invariance is
  required.** Nothing in a frame should make NVDAUSDT a different trade from the same tape under
  another name; a flip here is the decision leaning on a brand prior.
* ``entities+labels`` — additionally the labels of events and indicators (the SEC form types, the
  VIX, Fear & Greed, FINRA, the Treasury curve). **This is a dependence measurement, not an
  invariance requirement.** Those labels carry meaning — "8-K" tells a reader an unscheduled
  material event was filed — so a desk that changes its mind when that meaning is taken away is
  using the label, which may be correct. What the number reports is how much of the decision rests
  on the vocabulary rather than on the figures.

**Numbers are held fixed, and that is checked, not assumed.** :func:`obfuscate` returns the frame
together with the multiset of figures outside the renamed words; the test suite asserts the multiset
is unchanged. A rename that disturbed a figure would turn this into a corrupted-feed test, which is
`eval/feedbugged.py`'s job.

**The new words cannot be real tickers.** US equity tickers are at most five letters; every
replacement ticker here is six. Replacement company names are invented and were checked against the
ARGUS universe list; they are **NOT VERIFIED** against every company name in the world, which is
why the ticker layer — the one the desk routes on — carries the length guarantee.

**The resample baseline.** Every obfuscated decision is scored against the same unperturbed runs
S1 uses (`eval/perturbations.Baseline`), with the same before/worst pair and the same pairwise noise
comparison, so a flip that sampling noise already explains is never blamed on the rename.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from random import Random
from typing import Any

from argus.eval.artefact import write
from argus.eval.perturbations import (
    DATA,
    LEAN,
    QWEN_CAP,
    Baseline,
    Cell,
    Seat,
    Snapshot,
    live_seat,
    load_baseline,
    load_snapshots,
    spent,
    verdict,
    worst_case,
)
from argus.market.evidence import underlying_ticker
from argus.market.instruments import REGISTRY

REPORT_PATH = DATA / "vocab_stress.json"

BRANDS: dict[str, tuple[str, ...]] = {
    "NVDAUSDT": ("NVIDIA",),
    "TSLAUSDT": ("Tesla",),
    "AAPLUSDT": ("Apple",),
    "MSFTUSDT": ("Microsoft",),
    "METAUSDT": ("Meta Platforms", "Meta", "Facebook"),
    "GOOGLUSDT": ("Alphabet", "Google"),
    "AMZNUSDT": ("Amazon",),
    "COINUSDT": ("Coinbase",),
    "MSTRUSDT": ("MicroStrategy", "Strategy Inc"),
    "QQQUSDT": ("Invesco",),
    "TQQQUSDT": ("ProShares",),
    "SQQQUSDT": ("ProShares",),
}
"""Short forms headlines use, beyond the registry's full legal name. Matched case-sensitively, in
upper case and in title case ("NVIDIA", "Nvidia"), so the English words "meta" or "apple" in running
text are not renamed."""

LABELS: tuple[str, ...] = (
    "Fear & Greed", "Fear and Greed", "8-K/A", "8-K", "10-Q/A", "10-Q", "10-K/A", "10-K",
    "Form 4", "13F", "VIX", "CBOE", "FINRA", "Treasury", "Nasdaq", "EDGAR", "SEC",
)
"""Event and indicator labels, renamed only in the ``entities+labels`` layer."""

TICKER_WORDS: tuple[str, ...] = (
    "QZVRKL", "XJOMBT", "VULPRZ", "KWYXEF", "ZORBLQ", "PRAXJU", "MIZKOV", "GLYTRA",
    "BRUXQE", "YOVELZ", "TRAZIK", "WEMBLX", "FLOZUR", "NIRKAV", "SKOVRE", "DRAMUX",
)
"""Six letters each: no US equity ticker is longer than five, so none of these is a real one."""

NAME_WORDS: tuple[str, ...] = (
    "Veltrano", "Quibbex", "Morvanta", "Zelkovi", "Drumbrel", "Plissor", "Ostravel",
    "Kentrova", "Brathun", "Sulvenix", "Yarrowby", "Tessimar", "Grovanth", "Ilmendo",
    "Fazzoric", "Umbrisk",
)

LABEL_WORDS: tuple[str, ...] = (
    "Glimmet", "Frandle", "Wuzzock", "Plenth", "Snorvit", "Kradle", "Mubbin", "Tolvash",
    "Quenby", "Hesk", "Drovil", "Varrow", "Zindle", "Brumix", "Pellox", "Nerith", "Joskel",
    "Fromby", "Talvex", "Crindle",
)

LAYERS: tuple[str, ...] = ("entities", "entities+labels")

_FIGURE = re.compile(r"[+-]?\d+(?:[.,]\d+)*")


def random_mapping(names: Sequence[str], new_words: Sequence[str], rng: Random) -> dict[str, str]:
    """PlanBench ``obfuscator.py:31-52``: each name to a distinct word drawn without replacement.

    Order is preserved from ``names`` so the draw is reproducible for a given seed.
    """
    pool = list(new_words)
    mapping: dict[str, str] = {}
    for name in names:
        if name in mapping:
            continue
        if not pool:
            raise ValueError(
                f"not enough words provided: {len(new_words)} for {len(set(names))} names"
            )
        word = rng.choice(pool)
        pool.remove(word)
        mapping[name] = word
    return mapping


def entity_names(symbol: str) -> list[str]:
    """The issuer's names for one symbol: the registry's legal name, then the short forms."""
    names: list[str] = []
    identity = REGISTRY.get(symbol)
    if identity is not None:
        names.append(identity.underlying_name)
    names.extend(b for b in BRANDS.get(symbol, ()) if b not in names)
    return names


@dataclass(frozen=True, slots=True)
class Obfuscation:
    """A renamed snapshot, its mapping, and the figures that had to survive it."""

    layer: str
    snapshot: Snapshot
    mapping: dict[str, str]
    figures_before: tuple[str, ...]
    figures_after: tuple[str, ...]
    replacements: int

    @property
    def numbers_fixed(self) -> bool:
        return Counter(self.figures_before) == Counter(self.figures_after)


def _alternation(words: Sequence[str]) -> re.Pattern[str]:
    ordered = sorted(set(words), key=lambda w: (-len(w), w))
    return re.compile(
        r"(?<![A-Za-z0-9])(" + "|".join(re.escape(w) for w in ordered) + r")(?![A-Za-z0-9])"
    )


def _figures(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Every figure in ``text`` outside the words being renamed."""
    return _FIGURE.findall(pattern.sub(" ", text))


def obfuscate(snapshot: Snapshot, *, layer: str, seed: int = 0) -> Obfuscation:
    """Rename one snapshot's vocabulary, holding every figure fixed.

    The generator follows HELM's per-instance seeding (`eval/perturbations.Perturbation.rng`), so
    each snapshot gets its own reproducible mapping.
    """
    if layer not in LAYERS:
        raise ValueError(f"layer must be one of {LAYERS}, got {layer!r}")
    rng = Random(str(seed) + snapshot.id)
    mapping: dict[str, str] = {}
    # One invented ticker for the pair, so NVDAUSDT and NVDA stay recognisably the same thing.
    symbol_word = rng.choice(list(TICKER_WORDS))
    mapping[snapshot.symbol] = symbol_word + "USDT"
    underlying = underlying_ticker(snapshot.symbol)
    if underlying != snapshot.symbol:
        mapping[underlying] = symbol_word
    for name, word in random_mapping(entity_names(snapshot.symbol), NAME_WORDS, rng).items():
        # Headlines write brands in title case and upper case; both are the same company.
        mapping[name] = word
        mapping.setdefault(name.upper(), word.upper())
        mapping.setdefault(name.title(), word)
    if layer == "entities+labels":
        mapping.update(random_mapping(list(LABELS), LABEL_WORDS, rng))

    pattern = _alternation(list(mapping))
    count = 0

    def swap(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return mapping[match.group(1)]

    before: list[str] = []
    after: list[str] = []
    renamed: list[str] = []
    new_pattern = _alternation(list(mapping.values()))
    for line in snapshot.evidence:
        before.extend(_figures(line, pattern))
        changed = pattern.sub(swap, line)
        after.extend(_figures(changed, new_pattern))
        renamed.append(changed)
    return Obfuscation(
        layer=layer,
        snapshot=snapshot.with_evidence(renamed, symbol=mapping[snapshot.symbol]),
        mapping=mapping,
        figures_before=tuple(before),
        figures_after=tuple(after),
        replacements=count,
    )


def obfuscated_cells(
    snapshots: Sequence[Snapshot], seat: Seat, *, seed: int = 0,
) -> tuple[list[Cell], list[Obfuscation]]:
    """Each snapshot under each layer, once, entity layer first across all snapshots."""
    cells: list[Cell] = []
    made: list[Obfuscation] = []
    for layer in LAYERS:
        for snap in snapshots:
            ob = obfuscate(snap, layer=layer, seed=seed)
            made.append(ob)
            if not ob.numbers_fixed:
                seat.failures.append(f"{snap.id}/{layer}: a figure moved in the rename; not run")
                continue
            outcome = seat.decide(ob.snapshot, f"vocab-{layer}-{snap.id}")
            if outcome is not None:
                cells.append(Cell(snapshot_id=snap.id, condition=f"obfuscate:{layer}",
                                  outcome=outcome, edits=ob.replacements))
    return cells, made


def report(baseline: Baseline, cells: Sequence[Cell], made: Sequence[Obfuscation], *,
           requests: int, failures: Sequence[str]) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    for layer in LAYERS:
        mine = [c for c in cells if c.condition == f"obfuscate:{layer}"]
        score = worst_case(baseline, mine)
        unchanged = [
            int(c.outcome.action == baseline.reference(c.snapshot_id)) for c in mine
            if baseline.reference(c.snapshot_id) is not None
        ]
        lean = worst_case(baseline, mine, LEAN)
        layers[layer] = {
            "requirement": "invariance" if layer == "entities" else "dependence (reported only)",
            "decisions": len(mine),
            "unchanged": sum(unchanged),
            "unchanged_share": None if not unchanged else round(sum(unchanged) / len(unchanged), 4),
            "score": score.as_dict(),
            "verdict": verdict(score, what=f"renamed ({layer})", cells=len(mine)),
            "lean_score": lean.as_dict(),
            "lean_verdict": "Lean level. " + verdict(
                lean, what=f"renamed ({layer})", cells=len(mine)
            ),
        }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "item": "S2 vocabulary-obfuscation invariance",
        "source": "PlanBench plan-bench/obfuscator.py:31-52 (random_mapping); MIT",
        "layers": layers,
        "numbers_fixed": all(o.numbers_fixed for o in made),
        "mappings": {
            f"{o.layer}:{o.snapshot.id}": {
                "renamed_symbol": o.snapshot.symbol, "replacements": o.replacements,
                "mapping": o.mapping,
            }
            for o in made
        },
        "not_run": list(failures),
        "qwen_requests": requests,
        "cells": [c.as_dict() for c in cells],
    }


def main() -> int:  # pragma: no cover - CLI, spends the model budget
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="S2: vocabulary-obfuscation invariance")
    parser.add_argument("--snapshots", type=int, default=6)
    args = parser.parse_args()
    base = load_baseline()
    snaps = [s for s in load_snapshots() if s.id in base.runs][: args.snapshots]
    seat = live_seat(QWEN_CAP - spent(excluding=REPORT_PATH.name))
    cells, made = obfuscated_cells(snaps, seat)
    blob = report(base, cells, made, requests=seat.requests(), failures=seat.failures)
    write(REPORT_PATH, blob)
    for layer, row in blob["layers"].items():
        print(f"{layer}: {row['verdict']}")
    print(f"requests this run: {seat.requests()}; total spent: {spent()} of {QWEN_CAP}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
