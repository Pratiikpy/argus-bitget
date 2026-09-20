"""The universal negatives, with the search that supports them — runnable by anyone.

**Why this module exists.** Several of this project's sharpest sentences are *universal negatives*:
*"nothing in the corpus conditions its panel on the evidence"*, *"no system we read refuses to
schedule through a market-open boundary"*, *"no system in the corpus distinguishes a gate that
allowed from one that never ran"*. Each is the strongest claim on its row and **the most attackable
sentence in the project**: a rival destroys one with a single repository link, and a falsified
superlative discredits the fifty careful claims beside it by association.

A universal negative cannot be proven. It can only be *searched for*, and the honest form of the
claim is therefore not the sentence but the search: **here is the corpus, here is the query, here is
what it returned, and here are the closest things we did find.** That is what this module publishes.

**It also caught an overclaim of our own on its first run.** The planning documents said *"787
cloned repos"* throughout. Counting `.git` directories gives **667**. Nobody had counted; the figure
had been repeated until it read as measured. The count is now derived here and written into the
artefact, so the next person to quote it is quoting an arithmetic result.

**What it deliberately does not do.** It does not claim the search is exhaustive. A grep over 667
repositories finds the *idiom* it was given and misses a system that implements the same idea under
different words — so every claim carries its `near_misses`, the closest things the search did find,
and the verdict is phrased as *"not found by this query over this corpus"* rather than *"does not
exist"*. A negative finding needs more evidence than a positive one, and the evidence here is the
query, not the conclusion.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[4]
RESEARCH = WORKSPACE / "research"
REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "corpus_claims.json"

CORPUS_DIRS = (
    "repos",
    "repos-themed",
    "repos-t2",
    "repos-new",
    "repos-rivals",
    "repos-suggested",
    "gpt_suggestions",
)

SEARCHABLE = {".py", ".ts", ".tsx", ".js", ".rs", ".go", ".java"}

SKIP_DIRS = {".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build", "target"}


class CorpusError(RuntimeError):
    """Raised rather than reporting a corpus search that did not actually run."""


@dataclass(frozen=True, slots=True)
class Claim:
    """One universal negative and the query that stands behind it."""

    name: str
    sentence: str
    """The claim as the documents actually phrase it."""

    pattern: str
    """The regex whose absence is the evidence. Case-insensitive."""

    near_miss: str
    """A deliberately wider pattern: what the corpus *does* have that comes closest.

    Reporting this is what separates *"we looked and found nothing"* from *"we only looked for the
    thing we do"*. A claim whose near-miss set is empty has probably been searched too narrowly.
    """

    adjudication: str = ""
    """What a human found when they opened the hits, written **after** the search returned them.

    **This field exists because tuning the query until the claim survives is the exact failure these
    searches are for.** When the full corpus refuted two of four claims, the tempting fix was to
    narrow the patterns until both came back clean. That is what `research/overfit_gates.py`'s
    `PERIOD_BARS` note refuses one module over, and it would be worse here, because the output is a
    public claim rather than an internal number.

    So the pattern stays as written and the hits stay counted. A hit that a human reads and finds
    genuine **refutes the claim and the document changes**. A hit that turns out to be the query
    being loose is recorded as exactly that, in this field, with the file named so a reader can
    disagree. The adjudication never changes `holds`; it explains it.
    """


CLAIMS: tuple[Claim, ...] = (
    Claim(
        name="gate_reachability",
        sentence=(
            "no system in the corpus distinguishes a gate that allowed from one that never ran"
        ),
        pattern=r"\bunreached\b|\bnot_evaluated\b|\bgate_reachability\b|\bnever_executed\b",
        near_miss=r"checked_limits|risk_checks|gates_passed|all_checks_passed",
        adjudication=(
            "REFUTED, and genuinely. `repos-themed/AlgoVaultLabs~crypto-quant-signal-mcp/src/lib/"
            "scorer-input-codes.ts:82` defines `NOT_EVALUATED: 0` with the comment 'Distinct from "
            "NEUTRAL' — precisely the distinction this project claims nobody makes, between a rule "
            "that ran and allowed and one that never ran at all. It is applied to a signal "
            "adjustment rather than to a gate chain, which is a real difference in scope but not a "
            "difference in the idea. The claim was too strong and the documents now say so, naming "
            "this file."
        ),
    ),
    Claim(
        name="panel_conditioned_on_evidence",
        sentence="nothing in the corpus conditions its analyst panel on the evidence",
        pattern=r"(?:select|choose|pick)_analysts?\(|analyst_selection|panel_selection",
        near_miss=r"analysts?\s*=\s*\[|AGENTS\s*=\s*\[|agents?\s*=\s*\[",
        adjudication=(
            "The 7 hits are the query being loose, not the claim being wrong — and the pattern is "
            "left as written rather than narrowed until the answer is convenient. "
            "`repos/TradingAgents/cli/utils.py:135` is `select_analysts(asset_type)`: it filters "
            "the roster by instrument class and then presents a `questionary.checkbox` for a HUMAN "
            "to pick from. That is configuration by asset class plus a prompt, not a panel "
            "conditioned on the evidence in front of it. The claim stands; `holds` reads False "
            "because the search says so, and this note is why a reader should look before "
            "believing either."
        ),
    ),
    Claim(
        name="market_open_boundary",
        sentence="no system we read refuses to schedule through a market-open boundary",
        pattern=r"market_open_boundary|crosses_(?:the_)?open|reopen_boundary|spans_the_open",
        near_miss=r"market_open|is_open|trading_hours|session_open",
    ),
    Claim(
        name="abstention_graded",
        sentence="nothing in the corpus grades the decisions it declined to take",
        pattern=r"counterfactual_(?:move|return|pnl)|graded_abstention|refusal_alpha",
        near_miss=r"abstain|no_trade|skipped_trade|decline",
    ),
)


@dataclass
class ClaimResult:
    """What the search returned for one claim."""

    claim: Claim
    files_searched: int = 0
    hits: list[str] = field(default_factory=list)
    near_misses: list[str] = field(default_factory=list)

    @property
    def holds(self) -> bool:
        """The claim survives the search. **Not the same as being true.**"""
        return not self.hits

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.claim.name,
            "sentence": self.claim.sentence,
            "pattern": self.claim.pattern,
            "near_miss_pattern": self.claim.near_miss,
            "files_searched": self.files_searched,
            "verdict": (
                "not found by this query over this corpus" if self.holds
                else "REFUTED — the corpus contains it"
            ),
            "holds": self.holds,
            "hits": self.hits[:20],
            "hit_count": len(self.hits),
            "adjudication": self.claim.adjudication,
            "near_misses": self.near_misses[:10],
            "near_miss_count": len(self.near_misses),
        }


def repo_count(root: Path = RESEARCH) -> int:
    """Cloned repositories, counted rather than remembered.

    The planning documents said 787 for weeks. This returns 667.
    """
    total = 0
    for name in CORPUS_DIRS:
        directory = root / name
        if directory.exists():
            total += sum(1 for _ in directory.rglob(".git"))
    return total


def _searchable_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for name in CORPUS_DIRS:
        directory = root / name
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.suffix not in SEARCHABLE or not path.is_file():
                continue
            if SKIP_DIRS & set(path.parts):
                continue
            out.append(path)
    return out


def search(
    claims: tuple[Claim, ...] = CLAIMS, *, root: Path = RESEARCH, limit: int | None = None
) -> list[ClaimResult]:
    """Run every claim's query over the corpus.

    One pass over the files, all patterns applied per file — a pass per claim would read 58,000
    files four times to answer four questions.
    """
    files = _searchable_files(root)
    if not files:
        raise CorpusError(
            f"no searchable files under {root}. A claim reported as surviving a search that never "
            f"ran is worse than an unchecked claim, because it carries a receipt"
        )
    if limit is not None:
        files = files[:limit]

    compiled = [
        (c, re.compile(c.pattern, re.IGNORECASE), re.compile(c.near_miss, re.IGNORECASE))
        for c in claims
    ]
    results = {c.name: ClaimResult(claim=c, files_searched=len(files)) for c in claims}

    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for claim, hit, near in compiled:
            result = results[claim.name]
            where = str(path.relative_to(root)).replace("\\", "/")
            if hit.search(text):
                result.hits.append(where)
            elif near.search(text):
                result.near_misses.append(where)

    return list(results.values())


def report(results: list[ClaimResult], *, repos: int) -> dict[str, Any]:
    """The artefact a reader can check the sentences against."""
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "repos_searched": repos,
        "files_searched": results[0].files_searched if results else 0,
        "method": (
            "case-insensitive regex over .py/.ts/.tsx/.js/.rs/.go/.java in every cloned "
            "repository; a claim 'holds' when its pattern matched nowhere"
        ),
        "caveat": (
            "A grep finds the idiom it was given. A system implementing the same idea under "
            "different words is invisible to it, which is why every claim reports the closest "
            "things the search DID find. These verdicts are 'not found by this query', never "
            "'does not exist'."
        ),
        "claims": [r.as_dict() for r in results],
    }


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="run the searches behind the universal negatives")
    parser.add_argument("--limit", type=int, default=None, help="cap files, for a quick pass")
    parser.add_argument("--save", action="store_true", help=f"write {REPORT_PATH.name}")
    args = parser.parse_args()

    repos = repo_count()
    print(f"corpus: {repos} cloned repositories\n")
    results = search(limit=args.limit)
    for got in results:
        mark = "HOLDS " if got.holds else "REFUTED"
        print(f"  {mark}  {got.claim.name}")
        print(f"           {got.claim.sentence}")
        print(
            f"           {len(got.hits)} hit(s), {len(got.near_misses)} near-miss(es) "
            f"over {got.files_searched:,} files"
        )
        for where in got.hits[:3]:
            print(f"             HIT: {where}")
        for where in got.near_misses[:2]:
            print(f"             near: {where}")

    blob = report(results, repos=repos)
    if args.save:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        print(f"\n  written to {REPORT_PATH}")
    else:
        print("\n  (not saved — pass --save)")
    return 0 if all(r.holds for r in results) else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "CLAIMS",
    "CORPUS_DIRS",
    "REPORT_PATH",
    "Claim",
    "ClaimResult",
    "CorpusError",
    "main",
    "repo_count",
    "report",
    "search",
]
