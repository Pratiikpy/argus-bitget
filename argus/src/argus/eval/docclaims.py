"""Documentation self-verification — every number a document quotes, checked against its source.

Exists because the documents drifted within a day of being written. The README said the hurdle was
cleared on "nine of twelve" symbols while ``data/hurdle_clearance.json`` said eleven; the submission
draft quoted 1,410 tests and 63 modules against a tree with 1,470 and 65. Each of those is a claim
a judge could check and find wrong. This module makes the check ours: a registry of the figures the
documents quote, each paired with the computation or artefact that produces the live value, and a
comparison at the precision the document used. A quoted "0.67" against a live 0.6677 is OK; a quoted
"nine" against a live 11 is STALE, by name and line.

Design, taken from ``argus.status``: the document is never the source of truth for itself. Every
``Claim.live`` reads the artefact or counts the tree; nothing here reads a number from one document
to check another. Two modes exist because two kinds of number exist — ``exact`` for things that
are what they are (symbols beating the baseline), and ``at_least`` for counters that only grow
between edits (ledger entries), where the document is allowed to lag but never to overstate.

Not a claim that the prose is true; a claim that the numbers in the prose match the artefacts on
disk at the moment of the check. A wrong artefact passes. That is what the tests behind each
artefact are for.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]          # the workspace directory above argus/
PACKAGE = Path(__file__).resolve().parents[3]       # .../bitget/argus
DATA = PACKAGE / "data"
SRC = PACKAGE / "src" / "argus"
TESTS = PACKAGE / "tests"
REPORT_PATH = DATA / "doc_claims.json"

DOCS = {
    "readme": PACKAGE / "README.md",
    # The repository's front page — the first document a reader opens, and until 2026-09-20 the
    # only one this gate could not see, because it had no counterpart in this tree and was edited
    # directly in the publication repo. It had drifted furthest of any document here: "490 passing"
    # against a live 5,113, and "clean on 32 modules" against 266 source files — while its own
    # `pytest -q` code block, eight lines above, said 5,100. Now kept here and synced outward, so
    # the most-read file is the most-checked one.
    "public-readme": ROOT / "README.md",
    # The two remaining public documents, added 2026-09-20. Between them they are **78% of all
    # prose in the public repository** — the PRD alone is 145 KB against the front page's 6 KB —
    # and the front page sends a reader straight to them ("including every measurement and every
    # retraction"). Not one of their numbers was pinned. The gate had 76 checked instances inside
    # the files nobody opens and 0 inside the file a judge is pointed at.
    "prd": ROOT / "ARGUS-MASTER-PRD.md",
    "architecture": ROOT / "ARGUS-ARCHITECTURE.md",
    "submission": ROOT / "SUBMISSION-DRAFT.md",
    "explained": ROOT / "ARGUS-EXPLAINED.md",
    "master-plan": ROOT / "ARGUS-MASTER-PLAN.md",
}

_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}

Number = int | float



SEEN_PATH = DATA / "doc_claims_seen.json"
"""Every (claim, document) pair that has ever matched, committed to the repository.

**Because deleting a pinned sentence used to disarm its guard silently, by design.** An ABSENT
finding is skipped in the reporting loop and excluded from the exit code, on the correct reasoning
that a document which never made a claim is not making a false one. The gap is the word *never*:
once a document HAS made a claim, the same silence means something entirely different — the
sentence was reworded or deleted, and the check that used to cover it is now covering nothing, with
no signal anywhere. A stale number is loud and a vanished number is quiet, which is exactly the
wrong way round.

So the pairs are remembered. A pair in this file that no longer matches is reported **DELETED** and
fails the run. The fix is either to restore the sentence or to remove the claim deliberately — both
are fine, and both are a decision somebody made rather than a guard that evaporated.
"""


def _seen() -> dict[str, int]:
    if not SEEN_PATH.exists():
        return {}
    try:
        loaded: dict[str, int] = json.loads(SEEN_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded


def _remember(report: Report) -> None:
    """Record every pair that matched this run, and keep every pair remembered before.

    Never forgets on its own: a pair leaves this file only when a person deletes it, which is the
    deliberate act the whole mechanism exists to require.
    """
    seen = _seen()
    for finding in report.findings:
        if finding.status in {"OK", "LAGGING", "STALE"} and finding.line is not None:
            seen[f"{finding.claim}|{finding.doc}"] = finding.line
    payload = json.dumps(dict(sorted(seen.items())), indent=2) + chr(10)
    SEEN_PATH.write_text(payload, encoding="utf-8")


def parse_number(text: str) -> Number:
    """'eleven' -> 11, '1,470' -> 1470, '7.72' -> 7.72. Raises on anything else."""
    # U+2212 MINUS SIGN is what prose is typeset with; U+002D HYPHEN-MINUS is what float() accepts.
    # Spelled by code point rather than pasted, so the glyph never appears in this source and the
    # two cannot be confused by a reader — which is the confusion that made the first version of
    # the refusal claim match nothing at all.
    TYPOGRAPHIC_MINUS = chr(0x2212)
    cleaned = text.strip().lower().replace(",", "").replace(TYPOGRAPHIC_MINUS, "-")
    if cleaned in _WORDS:
        return _WORDS[cleaned]
    if re.fullmatch(r"-?\d+", cleaned):
        return int(cleaned)
    if re.fullmatch(r"-?\d+\.\d+", cleaned):
        return float(cleaned)
    raise ValueError(f"not a number: {text!r}")


def decimals_of(text: str) -> int:
    """How many decimal places the document used; the live value is rounded to match."""
    cleaned = text.strip().replace(",", "")
    return len(cleaned.split(".", 1)[1]) if "." in cleaned else 0


LAG_TOLERANCE = 0.20
"""How far a ``lagging`` figure may fall behind its live value before it is STALE.

**Neither existing mode fits a counter that grows every cycle.** ``exact`` fails the gate every time
a scheduled run appends a row — the lean count moved 11 to 13 during a single test run — which
trains a reader to ignore the check. ``at_least`` fails the other way: it passes whenever the
document quotes *less* than the live value, so "0 decisions carry a lean" would keep passing forever
after the first lean was written, and understating the evidence is precisely the defect that made
the README claim two settled abstentions when there were 53.

So a lagging figure may trail by this fraction and no more. A README a few cycles behind is
tolerable; one that is an order of magnitude behind is describing a different system.
"""


def agrees(quoted_text: str, live: Number, mode: str) -> bool:
    quoted = parse_number(quoted_text)
    rounded = round(float(live), decimals_of(quoted_text))
    if mode == "exact":
        return abs(rounded - float(quoted)) < 1e-9
    if mode == "at_least":
        return float(quoted) <= float(live) + 1e-9
    if mode == "lagging":
        # Overstating is never tolerated; only falling behind is, and only by LAG_TOLERANCE.
        if float(quoted) > float(live) + 1e-9:
            return False
        if float(live) == 0.0:
            return float(quoted) == 0.0
        return (float(live) - float(quoted)) / abs(float(live)) <= LAG_TOLERANCE + 1e-9
    raise ValueError(f"unknown mode {mode!r}")


@dataclass(frozen=True)
class Claim:
    """One figure the documents quote.

    ``pattern`` must carry named groups ``q`` (or ``q1``, ``q2``, ...) around the quoted number(s);
    ``live`` returns the matching number or tuple. ``docs`` names which documents are expected to
    make the claim — a document that never mentions it is ABSENT, not wrong.
    """

    name: str
    pattern: str
    live: Callable[[], Any]
    docs: tuple[str, ...]
    mode: str = "exact"
    flags: int = re.IGNORECASE | re.DOTALL


@dataclass
class Finding:
    claim: str
    doc: str
    line: int | None
    quoted: tuple[str, ...]
    live: tuple[Number, ...] | None
    status: str          # OK | LAGGING | STALE | ABSENT | DELETED | UNCHECKED
    excerpt: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim, "doc": self.doc, "line": self.line,
            "quoted": list(self.quoted), "live": list(self.live) if self.live else None,
            "status": self.status, "excerpt": self.excerpt,
        }


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(1 for f in self.findings if f.status == status)

    @property
    def stale(self) -> list[Finding]:
        return [f for f in self.findings if f.status == "STALE"]

    @property
    def total(self) -> int:
        """Every claim under the gate, whichever mode produced this report.

        ``checked`` depends on the invocation: the test-count claims can only be verified with
        ``--tests``, so a plain run verifies six fewer and reports them UNCHECKED instead. Anything
        quoting a headline figure wants **this** number, which does not move between modes.

        The cockpit page quoted ``checked``, and its own staleness test therefore failed whenever a
        plain run and a ``--tests`` run were interleaved — the page said 67 and the fresh artefact
        said 61. A self-verification gate that fails depending on which flag last ran is worse than
        no gate, because it teaches a reader to re-run until green.
        """
        return (self.count("OK") + self.count("LAGGING") + self.count("STALE")
                + self.count("UNCHECKED"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "checked": self.count("OK") + self.count("LAGGING") + self.count("STALE"),
            "ok": self.count("OK"),
            "lagging": self.count("LAGGING"),
            "stale": self.count("STALE"),
            "absent": self.count("ABSENT"),
            "deleted": self.count("DELETED"),
            "unchecked": self.count("UNCHECKED"),
            "findings": [f.as_dict() for f in self.findings],
        }


# ---- live values -------------------------------------------------------------------------------

def _json(name: str) -> Any:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def _ratio(text: str) -> int:
    """'11/12' -> 11."""
    return int(text.split("/", 1)[0])


def tests_collected() -> int:
    """Count tests exactly as pytest will run them; nothing cheaper agrees with pytest.

    **Restored 2026-09-15 after I deleted it by accident and replaced it with something worse.**
    A static `def test_` count returns 3,845 against pytest's 4,104 — parametrised tests multiply at
    collection time and no source scan can see them. I removed this exact provider, substituted the
    static count, watched it report four *correct* documents as stale, and concluded the claim was
    uncheckable. It was not: this implementation had already solved it, and its own first line says
    so. The lesson is the standing rule, failed on my own codebase — **read the source before
    replacing it.**
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(TESTS)],
        capture_output=True, text=True, cwd=PACKAGE, check=False,
    )
    # pyproject adds -q, so the output is one "path: N" line per file rather than node ids.
    per_file = re.findall(r"^\S+\.py: (\d+)$", proc.stdout, re.MULTILINE)
    if per_file:
        return sum(int(n) for n in per_file)
    return sum(1 for line in proc.stdout.splitlines() if "::" in line)


def source_files() -> int:
    return sum(1 for _ in SRC.rglob("*.py"))


def register_claims() -> int:
    """Claims committed to the public register.

    **A growing count, which is exactly why it must be checked rather than typed.** Two documents
    stated this figure and disagreed — ARGUS-EXPLAINED said 36, ARGUS-ARCHITECTURE said 41 — and
    both were wrong; the file held 56. An hour after they were corrected by hand it held 61.
    A number that moves on its own cannot live in prose unless something watches it.
    """
    path = DATA / "register.jsonl"
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


class ClaimError(RuntimeError):
    """Raised rather than reporting a claim this checker cannot evaluate.

    A provider that cannot produce its number must say why. Returning a plausible substitute would
    let a document quote a figure nothing stands behind — which is the failure this whole module
    exists to catch, committed by the module itself.
    """


def break_even_at_the_actual_hurdle() -> float:
    """Break-even directional accuracy at the hurdle the desk actually faces, as a percentage.

    **Two latent crashes, both inside the honesty tooling.** This was
    ``100.0 * next(f["break_even_accuracy"] for f in frontier if ...)``:

    * ``break_even_accuracy`` is **nullable by design** — `eval/hurdle.py` returns ``None`` when the
      hurdle is at or above the mean move, because the required accuracy is then >= 1.0 and
      *"reporting 1.4 would imply a 140% hit rate is a thing that could be attempted"*. Two of the
      ten frontier points already carry null. ``100.0 * None`` raises `TypeError`.
    * ``next()`` with no default raises `StopIteration` if no point matches the actual hurdle.

    Neither fires today, only because the actual hurdle (18.8bps) happens to sit far below the
    median move (136.8bps). A claim-checker that crashes on a legitimate value is a claim-checker
    that stops running on the day the market changes — and this one is the thing that keeps every
    other number honest.
    """
    blob = _json("hurdle_frontier.json")
    actual = blob["actual_hurdle_bps"]
    for point in blob["frontier"]:
        if abs(point["hurdle_bps"] - actual) < 1e-9:
            value = point["break_even_accuracy"]
            if value is None:
                raise ClaimError(
                    f"no directional accuracy clears the {actual}bps hurdle — the required rate is "
                    f"at or above 100%. The document should say so rather than quote a percentage"
                )
            return 100.0 * float(value)
    raise ClaimError(
        f"hurdle_frontier.json carries no frontier point at the actual hurdle of {actual}bps, so "
        f"the break-even accuracy quoted in the documents cannot be checked against anything"
    )


def ledger_decisions() -> int:
    """Decisions on the paper ledger. The number the status block quotes.

    **Filters `kind == "decision"`, added 2026-09-22.** A raw line count used to be exactly the
    decision count, and stopped being one the moment `paper/ledger.py` grew a second row kind,
    `"settlement_seal"` — every settled trade now writes one extra line that is not a decision. A
    row with no `"kind"` key at all (every line written before that field existed) defaults to
    `"decision"`, matching `Entry`'s own dataclass default exactly, so old rows are counted
    exactly as they always were.
    """
    path = DATA / "paper_ledger.jsonl"
    if not path.exists():
        return 0
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("kind", "decision") == "decision"
    )


def settled_trades() -> int:
    """Positions that actually settled, voided rows excluded.

    Registered because this is the number that went wrong worst. The README's status block carried
    "2 settled trades, both directionally correct, +14.99 net after costs" while the same file, two
    screens up, withdrew that claim — the two rows were booked against positions the Constitution
    had refused (`paper/corrections.py`). Nothing in the gate covered the sentence, so the
    self-checking documents reported a clean bill over a figure that contradicted the correction
    printed above it. A claim the checker cannot see is a claim nobody is checking.
    """
    from argus.paper.ledger import PaperLedger

    return int(PaperLedger(path=DATA / "paper_ledger.jsonl").performance()["settled_trades"])


def _ngram_report() -> dict[str, Any]:
    return dict(json.loads((DATA / "ngram_bench.json").read_text(encoding="utf-8")))


def _ngram_layer(layer: str, field: str) -> Number:
    """One layer's figure out of the bench artefact.

    Every LUI number in the documents is read from the *same* run, so an accuracy and the error
    count printed beside it can never come from two different measurements. Quoting a good
    accuracy from one run next to a low error count from another would be true twice and
    misleading once.
    """
    for row in _ngram_report()["layers"]:
        if row["layer"] == layer:
            value = row[field]
            return round(float(value) * 100.0, 1) if field == "accuracy" else int(value)
    raise KeyError(f"no layer {layer!r} in ngram_bench.json")


def lui_fresh_accuracy() -> Number:
    """In-scope accuracy on the independently authored corpus, as a percentage.

    Guarded because it is the figure most likely to drift back upward by accident: every future
    widening of `lui/question.py` raises the TUNED score, and a reader copying "the LUI number"
    from a terminal could easily copy that one. This reads the fresh corpus's result out of the
    artefact, so the document can only ever quote the number that was actually measured on a
    corpus nobody here wrote.
    """
    blob = json.loads((DATA / "oblique_bench.json").read_text(encoding="utf-8"))
    return float(blob["fresh"]["correct_pct"])


def lui_router_fresh_accuracy() -> Number:
    """The router's accuracy on the same corpus — the loss, kept checkable.

    A withdrawn claim is only withdrawn while somebody keeps checking that the replacement is
    still true. `luirouter.py` recomputes this on every run.
    """
    blob = json.loads((DATA / "lui_router.json").read_text(encoding="utf-8"))
    return round(float(blob["headline"]["fresh_accuracy"]) * 100.0, 1)


def tradeable_sessions_with_lean() -> tuple[Number, Number]:
    """Decisions taken with the anchor market OPEN, and how many called a direction.

    **The pair that answers "it never got the chance."** A desk abstaining only while the market
    is shut has demonstrated nothing. One abstaining 217 times with the market open, having
    committed to a direction on 213 of them, has demonstrated a judgement.

    `eval/autopsy.py` once published the first explanation — that the desk kept being sampled
    during closed sessions — and shipped a falsifier alongside it. On the grown ledger the
    falsifier fired and killed the explanation. Both numbers move with the record, so both are
    read from the artefact rather than written beside a claim that would quietly age into being
    wrong the same way.
    """
    blob = json.loads((DATA / "theme_audit.json").read_text(encoding="utf-8"))
    found = re.search(
        r"(\d+) of \d+ decisions were taken in one, and (\d+) of those", json.dumps(blob)
    )
    if not found:
        raise ClaimError(
            "theme_audit.json no longer states the tradeable-session counts in the shape this "
            "claim reads. The sentence moved; the guard must follow it rather than be deleted, "
            "because a guard that quietly stops matching is the defect SEEN_PATH exists to catch."
        )
    return int(found.group(1)), int(found.group(2))


def refusal_accuracy_2h() -> tuple[Number, Number]:
    """Directional calls right, and total, at the ~2h horizon.

    The figure that answers "447 decisions, zero trades — it has demonstrated nothing", so it is
    the last figure in these documents that may be allowed to drift.
    """
    row = _refusal_row()
    return int(row["correct"]), int(row["directional"])


def refusal_forgone_2h() -> Number:
    """Median net bps forgone by refusing, at ~2h. Negative means refusing saved money."""
    return float(_refusal_row()["median_forgone_bps"])


def _refusal_row() -> dict[str, Any]:
    blob = _json("refusal_alpha.json")
    for row in blob["horizons"]:
        if row["horizon"] == "about_2h":
            return dict(row)
    raise KeyError("refusal_alpha.json has no about_2h horizon")


def standing_counts() -> tuple[Number, Number]:
    """Capabilities whose evidence checks out, and the register's total.

    **Pinned because the register and its own artefact disagreed by 23.** `data/standing.json`
    shipped 14 capabilities and 0 owned while the source computed 24 and 23, and no claim in this
    module covered either figure — so the project's headline claim about itself was the one number
    nothing was checking. `earned` rather than `owned` deliberately: the first is what survives
    `standing.verify()` opening the artefacts, the second is what the register declares.
    """
    from argus.eval.standing import audit

    report = audit()
    return len(report.earned), len(report.capabilities)


def reliable_skill_tools() -> tuple[Number, Number]:
    """Bitget Skill tools that answer every attempt, and the total probed.

    Pinned because this figure was quoted from a single sweep and was **understating** us: the
    snapshot said 6 of 19 and repeated measurement found 9. A claim that moves depending on which
    day it was taken needs the gate more than a stable one does."""
    blob = _json("skill_reliability.json")
    return int(blob["by_verdict"].get("reliable", 0)), int(blob["tools"])


def answering_data_entries() -> tuple[Number, Number]:
    """Bitget data-catalog entries that return rows, and the total called."""
    blob = _json("data_coverage.json")
    return int(blob["answering"]), int(blob["entries_probed"])


def governed_failure_rates() -> tuple[Number, Number]:
    """Ungoverned and governed failure rates from the adaptive stress search, as percentages.

    The Track 2 headline for *risk control layer effectiveness*, and the one number in this
    project that is not a chosen parameter — the adversary searched for the path rather than
    being handed one."""
    paired = _json("stress_test.json")["paired"]
    return (
        round(100 * float(paired["ungoverned_failure_rate"]), 1),
        round(100 * float(paired["governed_failure_rate"]), 1),
    )


def module_count() -> int:
    from argus.status import MODULES
    return len(MODULES)


def subtheme_count() -> int:
    from argus.status import SUBTHEMES
    return len(SUBTHEMES)


def ledger_entries() -> int:
    """Same `kind == "decision"` filter as `ledger_decisions` — see its docstring. Two functions
    computing the identical count because two different claim regexes already existed for it
    before this fix; not worth collapsing to one for this change alone."""
    path = DATA / "paper_ledger.jsonl"
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("kind", "decision") == "decision"
    )


def settled_abstentions() -> int:
    """How many abstentions have a counterfactual attached.

    Tracked because the README carried "two have settled as abstentions" while the live figure was
    53. `ledger_entries` is deliberately ``at_least`` — a growing log must not fail the gate every
    cycle — and that tolerance is exactly why a second, exact number beside it was needed: the count
    of *settled* rows is the one a reader uses to judge whether anything has been graded, and a
    stale one understates the evidence rather than overstating it.
    """
    path = DATA / "paper_ledger.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return sum(1 for r in rows if r.get("counterfactual_move_bps") is not None)


def leans_recorded() -> int:
    """Decisions carrying a directional lean. Zero until a cycle runs with the field.

    Published so the shadow record's UNDEFINED verdict can be read against a number rather than
    taken on trust: if this is 0, there is nothing to grade and the absence of an accuracy is the
    honest state, not a missing measurement.
    """
    path = DATA / "paper_ledger.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return sum(1 for r in rows if str(r.get("lean", "none")).lower() in {"up", "down"})


def hurdle_rth() -> dict[str, float]:
    return {s: v["pct_gt_hurdle_rth"] for s, v in _json("hurdle_clearance.json")["symbols"].items()}


def hurdle_clearing_symbols() -> int:
    return sum(1 for pct in hurdle_rth().values() if pct >= 50)


def hurdle_clearing_range() -> tuple[int, int]:
    clearing = [pct for pct in hurdle_rth().values() if pct >= 50]
    return round(min(clearing)), round(max(clearing))


def teardown_count() -> int | None:
    """Code-level teardowns in `research/architecture/`, excluding the consolidated ledger.

    Added after the number drifted from 29 to 32 unnoticed: every count a document quotes has to
    have a producer here, or it silently rots the moment the tree grows.

    ``None`` (reported UNCHECKED) in a checkout without the consolidated ledger: the public
    repository carries only the teardowns its own evidence cites, so counting what is present
    there would call a true figure stale.
    """
    folder = ROOT / "research" / "architecture"
    if not (folder / "_CONSOLIDATED-LEDGER.md").is_file():
        return None
    return sum(1 for path in folder.glob("*.md")
               if not path.name.startswith("_") and path.name not in UNPUBLISHED_TEARDOWNS)


UNPUBLISHED_TEARDOWNS = frozenset({
    "okx-workspace-inventory.md", "kairos-nomos.md", "pod-sosovalue.md",
    "kcnyu~clawock.md", "weak-subtheme-repo-hunt.md",
})
"""Teardowns kept in the research workspace: the owner's own projects on other venues, a rival
Season-2 entry, and an internal planning list. The documents quote the count of the ones a reader
of the public repository can open, so the same number is right in both places."""


def overfit_counts() -> tuple[int, int]:
    gates = _json("overfit_gates.json")
    return len(gates["cleared_but_unprofitable_after_fees"]), len(gates["rejected_as_noise"])


def session_beta_aapl(key: str) -> float:
    rows = _json("session_beta.json")["rows"]
    return float(next(r for r in rows if r["symbol"] == "AAPLUSDT")[key])


CLAIMS: tuple[Claim, ...] = (
    # Four digits or a thousands comma, so a per-module count ("30 tests", "22 tests") in a prose
    # cell is not mistaken for the suite total. Both are legitimate sentences; only one is this
    # claim, and matching both made the checker report a disagreement between a document and
    # itself.
    Claim("tests_passing", r"(?P<q>\d{1,3},\d{3}|\d{4,}) tests(?: passing| collected)?",
          tests_collected, ("readme", "public-readme", "submission", "explained"), mode="at_least"),
    Claim("source_files", r"strict\W{0,3}clean (?:on|across) (?P<q>\d+) (?:source )?files",
          source_files, ("readme", "public-readme", "submission", "explained")),
    # `falsifiable claims about` was added after README:163 was found quoting 36 against a live 156
    # — the sentence had drifted by 120 claims and matched no pattern, so the gate never saw it.
    Claim("register_claims",
          r"(?P<q>[\d,]+) (?:falsifiable )?claims? "
          r"(?:across|about|committed|pre-registered|on the register)",
          register_claims,
          ("readme", "public-readme", "submission", "explained", "master-plan", "architecture"),
          mode="lagging"),
    # Registered 2026-09-20. `eval/refusal.py` existed and ran, and appeared in no document, no
    # command list and no panel — the single strongest answer to the obvious attack on an
    # all-refusal ledger was invisible. A number that is not quoted cannot go stale, which is why
    # it survived every previous sweep of this gate.
    # `\s+` rather than a literal space, and U+2212 accepted alongside `-`. The first spelling
    # of these two bound on one document and silently missed the other: the phrase wrapped
    # across a line there, and the figure was written with a typographic minus. A claim that
    # matches nothing reports ABSENT, which reads exactly like a document that simply does
    # not make the claim — so a pattern bug and an honest silence are indistinguishable.
    # The rebuttal-killer. Both halves guarded, because quoting the open-session count without
    # the lean count would let the stronger claim decay into the weaker one unnoticed.
    Claim("tradeable_sessions_with_lean",
          r"(?P<q1>\d+) of those\s+\d+ decisions were taken in a tradeable session"
          r"[\s\S]{0,120}?\*\*(?P<q2>\d+) of them",
          tradeable_sessions_with_lean, ("readme",), mode="lagging"),
    # `lagging`, not `exact`. Both halves are counters the paper runner appends to, and the mode
    # was wrong from the day the claim was registered: it went stale twice in one session as the
    # ledger moved 217/213 to 222/218 to 229/225. LAG_TOLERANCE's own docstring names this failure
    # — an exact gate on a growing counter fails constantly and trains a reader to ignore it —
    # and the fix is the state built for exactly this, which keeps the gap visible without
    # breaking the run.
    Claim("refusal_accuracy_2h", r"(?P<q1>\d+)\s+of\s+(?P<q2>\d+)\s+directional\s+calls",
          refusal_accuracy_2h, ("readme", "public-readme")),
    Claim("refusal_forgone_2h", rf"forgave\s+(?P<q>[-{chr(0x2212)}]?[\d.]+)\s*bps of net edge",
          refusal_forgone_2h, ("readme", "public-readme")),
    Claim("standing_owned", r"(?P<q1>\d+) of (?P<q2>\d+) capabilities are OWNED",
          standing_counts, ("readme", "public-readme", "submission", "explained",
                            "architecture")),
    # Both LUI figures are guarded because both are *losses*, and a losing number left ungated is
    # the one that quietly improves between drafts. The phrasings are anchored to the sentences
    # actually written in SUBMISSION-DRAFT.md.
    Claim("lui_router_fresh_accuracy",
          r"model2vec` scores the same (?P<q>[\d.]+)%",
          lui_router_fresh_accuracy, ("submission",)),
    # The three figures that describe the console a judge actually meets. All read out of
    # `ngram_bench.json`, so the document cannot quote a layer's score from a different run than
    # the one that produced its error count.
    Claim("lui_patterns_accuracy",
          r"regex layer that was deployed scores (?P<q>[\d.]+)%",
          lambda: _ngram_layer("patterns", "accuracy"), ("submission",)),
    Claim("lui_cascade",
          r"now ships: (?P<q1>[\d.]+)% correct with (?P<q2>\d+) confident errors",
          lambda: (_ngram_layer("cascade", "accuracy"),
                   _ngram_layer("cascade", "confidently_wrong")),
          ("submission",)),
    Claim("lui_out_of_scope_recall",
          r"Out-of-scope recall (?P<q>\d+)%",
          lambda: round(_ngram_report()["out_of_scope"]["recall"] * 100.0), ("submission",)),
    Claim("reliable_skill_tools",
          r"(?P<q1>\d+) of (?P<q2>\d+) answer three times out of three",
          reliable_skill_tools, ("submission",)),
    Claim("answering_data_entries", r"(?P<q1>\d+) of (?P<q2>\d+) answer, and",
          answering_data_entries, ("submission",)),
    Claim("governed_failure_rates",
          r"failure rate is (?P<q1>[\d.]+)% ungoverned against (?P<q2>[\d.]+)% governed",
          governed_failure_rates, ("submission",)),
    Claim("settled_trades", r"(?P<q>\d+) settled trades?",
          settled_trades, ("readme", "submission", "explained")),
    # Registered the day the anchor claims were found to be false. The documents said the register
    # was "anchored to Bitcoin" and verifiable "with the reference OpenTimestamps client" while all
    # 48 proofs were raw calendar receipts carrying only a pending attestation — unreadable by that
    # client, and not anchored to anything yet. Both are now true, and both are now checked.
    # **The total was hard-coded into this guard's own pattern as `48`, so it stopped guarding the
    # moment a proof was added.** The register grew to 52 and the sentence stopped matching, which
    # the DELETED status reported — but had the sentence been left alone it would have read "38 of
    # the 48" beside 52 proofs on disk, and this guard would have gone on passing it, because the
    # literal it was checking against was in the regex rather than in the artefact. Both numbers
    # are captured and both are checked now.
    Claim("anchors_confirmed",
          r"(?P<q1>\d+) of the (?P<q2>\d+)(?:\*\*)? (?:proofs )?carry a Bitcoin",
          lambda: (
              __import__("argus.register.anchorcheck", fromlist=["x"]).confirmed_anchors(),
              __import__("argus.register.anchorcheck", fromlist=["x"]).check().total,
          ),
          ("readme", "explained"), mode="lagging"),
    Claim("ledger_decisions",
          r"(?P<q>[\d,]+) decisions? in the paper ledger",
          ledger_decisions, ("readme", "submission", "explained"), mode="lagging"),
    Claim("modules_importable", r"(?P<q>\d+)/(?P<q2>\d+) modules importable",
          lambda: (module_count(), module_count()), ("readme", "public-readme", "submission")),
    Claim("subthemes", r"(?P<q>\d+)/(?P<q2>\d+) sub-themes",
          lambda: (subtheme_count(), subtheme_count()), ("readme", "public-readme", "submission")),
    Claim("ledger_entries",
          r"(?P<q>\d+) (?:decisions(?= in the paper ledger|, 20\d\d|\. Every one|, 0 settled)"
          r"|ledger entries)|paper-trading log has (?P<q1>\d+) decisions", ledger_entries,
          ("readme", "submission", "explained"), mode="at_least"),
    Claim("settled_abstentions",
          r"(?P<q>[\d,]+) (?:have settled as abstentions|settled abstentions)",
          settled_abstentions, ("readme", "submission", "explained"), mode="lagging"),
    Claim("leans_recorded", r"(?P<q>[\d,]+) decisions? carr(?:y|ies) a lean",
          leans_recorded, ("readme", "explained"), mode="lagging"),
    Claim("t1_beat_buy_and_hold",
          r"(?P<q>\w+) of (?:twelve|12) symbols (?:have a strategy that )?beats? buy",
          lambda: _ratio(_json("track1_study.json")["headline"]["beat_buy_and_hold_sharpe"]),
          ("submission", "explained", "master-plan")),
    Claim("t1_dsr_candidates", r"(?P<q>\w+) of 12 survive that softer gate",
          lambda: _ratio(_json("track1_study.json")["headline"]["survived_dsr_candidates_only"]),
          ("explained",)),
    # Bound to MSTRUSDT because that is the symbol the document now quotes: one of the two
    # soft-gate survivors, and scorable in only half its windows. Rebinding the claim rather than
    # changing the sentence keeps the checker pointed at the number actually written down.
    Claim("t1_rolling_scorable", r"(?P<q>[\d,]+) of (?P<q2>[\d,]+) windows",
          lambda: (
              (lambda st: (st["windows"], st["windows"] + st["skipped_flat_windows"]))(
                  next(r["rolling_sharpe_stability"]
                       for r in _json("track1_study.json")["per_symbol"]
                       if r["symbol"] == "MSTRUSDT")
              )
          ),
          ("explained",)),
    Claim("t1_rolling_scorable_worst", r"scorable in only (?P<q>[\d,]+) windows",
          lambda: min(
              r["rolling_sharpe_stability"]["windows"]
              for r in _json("track1_study.json")["per_symbol"]
          ),
          ("explained",)),
    Claim("t1_dsr_survivors", r"(?P<q>\w+) of twelve[^.]{0,60}survive[sd]? (?:the )?(?:search|dsr)",
          lambda: _ratio(_json("track1_study.json")["headline"]["survived_dsr_all_trials"]),
          ("explained", "master-plan")),
    Claim("t1_oos_breaches", r"(?P<q>\w+) of twelve breach",
          lambda: sum(1 for r in _json("track1_study.json")["per_symbol"]
                      if r["out_of_sample_decay"]["breaches_half_alert"]),
          ("explained", "master-plan")),
    Claim("arbitrage_monetisable", r"monetis?z?able \**(?P<q>\d+\.\d+)%",
          lambda: _json("arbitrage_study.json")["headline"]["monetizable_pct"],
          ("submission", "explained", "master-plan")),
    Claim("weekend_continuation", r"(?P<q>\d+\.\d)% continuation",
          lambda: _json("weekend_significance.json")["test_1_significance"]["weekend"]["rate_pct"],
          ("explained", "master-plan")),
    Claim("session_beta_higher_open", r"on (?P<q>\w+) of (?P<q2>\w+) (?:rTokens|stock perpetuals)",
          lambda: (_json("session_beta.json")["symbols_higher_open"],
                   _json("session_beta.json")["symbols_compared"]),
          ("readme", "submission", "explained", "master-plan")),
    Claim("aapl_beta_open", r"AAPL (?:reads|is) \**(?P<q>0\.\d+)",
          lambda: session_beta_aapl("beta_open"),
          ("readme", "submission", "explained", "master-plan")),
    Claim("aapl_beta_shut", r"against (?P<q>0\.\d+) shut",
          lambda: session_beta_aapl("beta_shut"),
          ("readme", "submission", "explained", "master-plan")),
    Claim("hurdle_symbols_clearing_rth", r"regular hours on\s+(?P<q>\w+) of twelve symbols",
          hurdle_clearing_symbols, ("readme", "submission", "explained", "master-plan")),
    Claim("hurdle_clearing_range",
          r"(?P<q>\d+)[–-](?P<q2>\d+)% (?:of\s+the time )?(?:during|in) US regular",  # noqa: RUF001 — the documents write ranges with an en dash
          hurdle_clearing_range, ("readme", "submission", "explained", "master-plan")),
    Claim("hurdle_qqq_rth", r"QQQ[^.\n]{0,60}?(?P<q>\d+\.\d)%",
          lambda: hurdle_rth()["QQQUSDT"], ("submission", "master-plan")),
    Claim("teardowns", r"(?P<q>\d+) code-level teardowns", teardown_count,
          ("readme", "explained", "prd")),

    # --- the overfitting study and the restatement ---------------------------------------------
    # These are the newest headline numbers and the ones most likely to be quoted back at us, so
    # each is bound to the artefact that produced it. The PBO range in particular is a pair: a
    # document quoting only the low end would be reporting the flattering half of a result whose
    # whole point is the spread.
    Claim("pbo_range", r"(?P<q>0\.\d+) to \*\*(?P<q2>0\.\d+)\*\* across twelve symbols",
          lambda: (
              min(r["pbo_all_trials"] for r in _json("overfitting_study.json")["symbols"]),
              max(r["pbo_all_trials"] for r in _json("overfitting_study.json")["symbols"]),
          ),
          ("explained",)),
    Claim("pbo_worst", r"overfitting is \*\*(?P<q>0\.\d+)\*\* .{0,20}the\s+worst of the twelve",
          lambda: max(r["pbo_all_trials"] for r in _json("overfitting_study.json")["symbols"]),
          ("explained",)),
    Claim("fdr_trials", r"(?P<q>\d+) of (?P<q2>\d+) \(symbol, variant\) trials",
          lambda: (
              _json("overfitting_study.json")["grid"]["survivors_benjamini_hochberg"],
              _json("overfitting_study.json")["grid"]["trials"],
          ),
          ("explained",)),
    Claim("mintrl_range",
          r"(?P<q>\d\.\d+) to \*\*(?P<q2>\d\.\d+) years\*\* of live hourly",
          lambda: (
              min(r["min_track_record_years"] for r in _json("overfitting_study.json")["symbols"]
                  if r["min_track_record_years"] is not None),
              max(r["min_track_record_years"] for r in _json("overfitting_study.json")["symbols"]
                  if r["min_track_record_years"] is not None),
          ),
          ("explained",)),
    Claim("brier_index", r"Brier Index (?:of )?\*{0,2}(?P<q>\d+\.\d)\*{0,2}",
          lambda: _json("forecastbench_restatement.json")["brier_index"], ("explained",)),
    Claim("skill_vs_climatology", r"\*\*\+(?P<q>\d+\.\d)% against climatology\*\*",
          lambda: 100.0 * _json("forecastbench_restatement.json")["skill_vs_climatology"],
          ("explained",)),
    Claim("hurdle_break_even",
          r"directional accuracy of \**(?P<q>\d+\.\d)%",
          break_even_at_the_actual_hurdle,
          ("explained",)),
    Claim("hurdle_median_move", r"median absolute\s+24-hour move is (?P<q>\d+)bps",
          lambda: _json("hurdle_frontier.json")["median_abs_move_bps"], ("explained",)),

    # --- the cross-sectional study -----------------------------------------------------------
    # Five claims rather than one because each is a different kind of number, and the pair that
    # matters most is `xs_backtests` against `xs_trials`: a document that quoted the larger number
    # as the trial count would be overstating the multiple-testing penalty, and one that quoted the
    # smaller as the work done would be understating the phase sweep. Both are wrong in ways a
    # reader cannot detect, so both are checked.
    Claim("xs_trials", r"Trials \(factors .{1,3} trading rules\) \| (?P<q>[\d,]+)",
          lambda: _json("crosssection_study.json")["trials"], ("explained",)),
    Claim("xs_backtests", r"Backtests actually run \(every phase\) \| (?P<q>[\d,]+)",
          lambda: _json("crosssection_study.json")["backtests_run"], ("explained",)),
    Claim("xs_positive_gross", r"positive \*\*before\*\* cost \| (?P<q>\d+)",
          lambda: _json("crosssection_study.json")["headline"]["positive_mean_gross_sharpe"],
          ("explained",)),
    Claim("xs_positive_net", r"positive \*\*after\*\* cost \| (?P<q>\d+)",
          lambda: _json("crosssection_study.json")["headline"]["positive_mean_net_sharpe"],
          ("explained",)),
    Claim("xs_phase_agreement", r"phases agree on sign \| \*\*(?P<q>\d+)\*\*",
          lambda: _json("crosssection_study.json")["headline"][
              "profitable_and_phase_consistent"],
          ("explained",)),
    Claim("overfit_noise_primitives",
          r"(?P<q>\w+) (?:of eight (?:factors|primitives) are noise|are indistinguishable from)",
          lambda: overfit_counts()[1], ("readme", "explained")),
)

# Claims whose live value is expensive to compute; skipped unless asked for.
EXPENSIVE = frozenset({"tests_passing"})


# ---- the audit ---------------------------------------------------------------------------------

def _groups(match: re.Match[str]) -> tuple[str, ...]:
    named = match.groupdict()
    keys = sorted(k for k in named if re.fullmatch(r"q\d*", k))
    return tuple(named[k] for k in keys if named[k] is not None)


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def audit(
    claims: tuple[Claim, ...] = CLAIMS,
    docs: dict[str, Path] | None = None,
    include_expensive: bool = False,
    live_overrides: dict[str, Any] | None = None,
) -> Report:
    """Check every claim against every document that is expected to make it.

    ``live_overrides`` lets a test supply values without touching disk; it maps claim name to the
    value ``live`` would have returned.
    """
    docs = DOCS if docs is None else docs
    overrides = live_overrides or {}
    texts = {
        name: path.read_text(encoding="utf-8") if path.exists() else None
        for name, path in docs.items()
    }
    # Loaded once. A pair here that no longer matches is a removed sentence, not an absent claim.
    seen_before = _seen()
    report = Report()
    for claim in claims:
        if claim.name in overrides:
            live_value: Any = overrides[claim.name]
        elif claim.name in EXPENSIVE and not include_expensive:
            live_value = None
        else:
            live_value = claim.live()
        live_tuple = (
            None if live_value is None
            else tuple(live_value) if isinstance(live_value, tuple) else (live_value,)
        )
        for doc in claim.docs:
            text = texts.get(doc)
            if text is None:
                report.findings.append(Finding(claim.name, doc, None, (), live_tuple, "ABSENT",
                                               "document not found"))
                continue
            matches = list(re.finditer(claim.pattern, text, claim.flags))
            if not matches:
                # Silence means "this document never made the claim" only if it never did.
                # See SEEN_PATH: once it has, the same silence means the sentence was removed.
                was_seen = f"{claim.name}|{doc}" in seen_before
                report.findings.append(Finding(
                    claim.name, doc, None, (), live_tuple,
                    "DELETED" if was_seen else "ABSENT",
                    f"last seen at line {seen_before[f'{claim.name}|{doc}']}" if was_seen else "",
                ))
                continue
            for match in matches:
                quoted = _groups(match)
                excerpt = " ".join(match.group(0).split())[:120]
                line = _line_of(text, match.start())
                if live_tuple is None:
                    status = "UNCHECKED"
                elif len(quoted) != len(live_tuple) or not all(
                    agrees(q, v, claim.mode) for q, v in zip(quoted, live_tuple, strict=True)
                ):
                    status = "STALE"
                elif any(
                    not agrees(q, v, "exact") for q, v in zip(quoted, live_tuple, strict=True)
                ):
                    # **Tolerated is not the same as equal, and printing both as OK hid the
                    # difference.** `LAG_TOLERANCE` exists so a counter that grows every cycle does
                    # not fail the gate between refreshes, which is right. But the report rendered
                    # `quoted 495, live 500` identically to `quoted 115/193, live 115/193`, so a
                    # reader scanning for OK could not see that five published figures were behind
                    # the record. This fails nothing that passed before; it stops the pass from
                    # looking like agreement when it is forbearance.
                    status = "LAGGING"
                else:
                    status = "OK"
                report.findings.append(Finding(claim.name, doc, line, quoted, live_tuple,
                                               status, excerpt))
    return report


def summary(report: Report) -> dict[str, Any]:
    """The compact form ``argus.status`` embeds: counts plus the stale lines by name."""
    return {
        "total": report.total,
        "checked": report.count("OK") + report.count("LAGGING") + report.count("STALE"),
        "lagging": report.count("LAGGING"),
        "lagging_detail": [
            f"{f.doc}:{f.line} {f.claim} quotes {'/'.join(f.quoted)}, live "
            f"{'/'.join(str(v) for v in (f.live or ()))}"
            for f in report.findings if f.status == "LAGGING"
        ],
        "stale": report.count("STALE"),
        "unchecked": report.count("UNCHECKED"),
        "stale_detail": [
            f"{f.doc}:{f.line} {f.claim} quotes {'/'.join(f.quoted)}, live "
            f"{'/'.join(str(v) for v in (f.live or ()))}"
            for f in report.stale
        ],
    }


@dataclass(frozen=True, slots=True)
class Disagreement:
    """One claim quoted with different values in different documents."""

    claim: str
    values: tuple[tuple[str, str], ...]
    """(document, quoted value) for each distinct value found."""

    def render(self) -> str:
        return f"{self.claim}: " + ", ".join(f"{doc} says {value}" for doc, value in self.values)


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def disagreements(report: Report) -> list[Disagreement]:
    """Claims where two documents quote different numbers for the same fact.

    **The hole this closes.** ``at_least`` mode exists for counters that only grow — a document is
    allowed to lag the ledger between edits — and it is the right rule for one document. Applied
    across several it hid a real defect: the README said 93 decisions, the submission draft said
    86, and the ledger held 126. Every one passed, because each was at most the live value.

    Three documents disagreeing by forty is a defect whatever the mode, and a judge reading two of
    them will see it before we do. This finds it without weakening ``at_least``, because it
    compares the documents to *each other* rather than to the artefact.
    """
    by_claim: dict[str, dict[str, str]] = {}
    for finding in report.findings:
        if finding.status not in ("OK", "STALE") or not finding.quoted:
            continue
        # Normalised before comparing, or the check cries wolf: "0.67" and "0.668" are the same
        # figure at two precisions, and "Four" and "four" are the same word. Only a genuine
        # difference in value is a disagreement.
        parts: list[str] = []
        for raw in finding.quoted:
            try:
                parts.append(f"{float(parse_number(raw)):.6g}")
            except ValueError:
                parts.append(raw.strip().casefold())
        quoted = "/".join(parts)
        by_claim.setdefault(finding.claim, {}).setdefault(quoted, finding.doc)
    out: list[Disagreement] = []
    for claim, values in sorted(by_claim.items()):
        numeric = [v for v in values if _is_number(v)]
        if len(numeric) == len(values) and len(values) > 1:
            # Two numbers that agree once rounded to the coarser of the two precisions are the
            # same figure written differently.
            places = min(len(v.split(".")[1]) if "." in v else 0 for v in values)
            rounded = {f"{float(v):.{places}f}" for v in values}
            if len(rounded) == 1:
                continue
        if len(values) > 1:
            out.append(Disagreement(
                claim=claim,
                values=tuple(sorted((doc, value) for value, doc in values.items())),
            ))
    return out


def repair(report: Report, *, include_lagging: bool = False) -> list[str]:
    """Rewrite every STALE figure in place, to the value its own artefact reports.

    Added after the fourth manual pass over the same five numbers in one session. A counter that
    only ever grows — source files, modules, tests — makes a document stale on every commit, and a
    checker that can find the drift but not close it turns an automated guard back into a chore.

    Three properties keep this safe to run unattended:

    * **It only ever writes the live value.** The replacement is computed from the artefact, so a
      repair cannot introduce a number that was not already true when the check ran.
    * **It rewrites one line, at the exact quoted text.** The regex that found the figure supplies
      its span, so nothing outside the matched digits is touched and a sentence cannot be reshaped.
    * **It refuses anything but STALE.** An ABSENT claim means a document does not make that claim
      at all, and inserting one would be writing prose rather than correcting a figure.

    Returns a line per repair, for the caller to print. Never silent.
    """
    done: list[str] = []
    by_doc: dict[str, list[Finding]] = {}
    # ``include_lagging`` also brings LAGGING figures current — used before a publish, so the copy
    # a reader sees quotes today's counts rather than ones a few cycles behind. Still only ever the
    # live value, at the exact quoted span.
    lagging = [f for f in report.findings if f.status == "LAGGING"] if include_lagging else []
    targets = [*report.stale, *lagging]
    for finding in targets:
        by_doc.setdefault(finding.doc, []).append(finding)
    for doc, findings in by_doc.items():
        path = DOCS.get(doc)
        if path is None or not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for finding in findings:
            if finding.live is None or len(finding.quoted) != len(finding.live):
                continue
            if not finding.excerpt or finding.excerpt not in text:
                continue
            # Every captured group is rewritten inside ONE copy of the excerpt, and the excerpt is
            # substituted once at the end.
            #
            # The first version of this rewrote the document per group, which turned "81/81
            # modules importable" into "82/81" and then stopped: after the first substitution the
            # excerpt no longer matched the document, so the second group silently did nothing.
            # A repair that half-applies is worse than one that does not run, because the check
            # that follows it reports a number nobody ever wrote.
            fixed_excerpt = finding.excerpt
            changes: list[str] = []
            cursor = 0
            for quoted, live in zip(finding.quoted, finding.live, strict=True):
                fresh = _format_like(quoted, live)
                at = fixed_excerpt.find(quoted, cursor)
                if at < 0:
                    continue
                if quoted != fresh:
                    fixed_excerpt = (
                        fixed_excerpt[:at] + fresh + fixed_excerpt[at + len(quoted):]
                    )
                    changes.append(f"{doc}:{finding.line} {finding.claim} {quoted} -> {fresh}")
                cursor = at + len(fresh)
            if fixed_excerpt != finding.excerpt:
                text = text.replace(finding.excerpt, fixed_excerpt, 1)
                done.extend(changes)
        path.write_text(text, encoding="utf-8", newline="")
    return done


def _format_like(quoted: str, live: Number) -> str:
    """The live value, written the way the document writes numbers.

    Thousands separators and decimal places are preserved from the quoted text: replacing "1,470"
    with "2286" would fix the number and break the prose.
    """
    places = decimals_of(quoted)
    value = round(float(live), places)
    rendered = f"{value:,.{places}f}" if "," in quoted else f"{value:.{places}f}"
    return rendered


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = sys.argv[1:] if argv is None else argv
    include_expensive = "--tests" in args
    report = audit(include_expensive=include_expensive)
    clashes = disagreements(report)
    for clash in clashes:
        print(f"DISAGREE  {clash.render()}")
    if "--fix" in args or "--fix-all" in args:
        repaired = repair(report, include_lagging="--fix-all" in args)
        for line in repaired:
            print(f"FIXED     {line}")
        if repaired:
            report = audit(include_expensive=include_expensive)
    REPORT_PATH.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    _remember(report)
    for finding in report.findings:
        if finding.status == "ABSENT":
            continue
        where = f"{finding.doc}:{finding.line}" if finding.line else finding.doc
        live = "/".join(str(v) for v in (finding.live or ()))
        print(f"{finding.status:9} {finding.claim:28} {where:22} quoted "
              f"{'/'.join(finding.quoted):10} live {live:10} | {finding.excerpt}")
    deleted = [f for f in report.findings if f.status == "DELETED"]
    for gone in deleted:
        print(
            f"DELETED   {gone.claim:28} {gone.doc:22} — the sentence this guard covered is no "
            f"longer in the document ({gone.excerpt}). Restore it, or remove the claim on purpose."
        )
    print(json.dumps(summary(report), indent=2))
    # DELETED fails for the same reason STALE does: in both cases a published figure and the
    # artefact behind it have stopped agreeing. Silence is the harder one to notice, so it is the
    # one that most needs an exit code.
    return 1 if (report.stale or deleted or disagreements(report)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
