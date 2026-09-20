"""Every artefact the documents cite as evidence must be regenerable by a command.

**Why this file exists.** Four artefacts — `hurdle_clearance.json`, `overfit_gates.json`,
`session_beta.json`, `stress_report.json` — were cited in `README.md` as evidence while **nothing in
the codebase could produce them.** They sat on disk, they were quoted as proof, and a reader who
tried to check one would have found no command to run. One of them was even described in its own
module's docstring as *"reproducible"*.

That is the most expensive defect this project can have, because it is invisible: the file exists,
the number is real, the prose is confident, and the reproduction is impossible. It is exactly the
failure `eval/docclaims.py` cannot catch — that tool checks whether a quoted number still *matches*
its artefact, not whether the artefact can be *rebuilt*.

Rebuilding them found something a staleness check never would: `overfit_gates.json`'s stored
four-and-four split **did not reproduce**. The honest re-measurement is harsher — six refuted, two
unevaluated, none surviving — and it was published rather than tuned back into agreement.

So this file guards the class rather than the four instances. A cited artefact with no writer fails
here, on the day it is cited, instead of on the day someone tries to verify it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ARGUS = Path(__file__).resolve().parents[1]
WORKSPACE = ARGUS.parent
SRC = ARGUS / "src" / "argus"

DOCS = (
    ARGUS / "README.md",
    WORKSPACE / "ARGUS-EXPLAINED.md",
    WORKSPACE / "ARGUS-ARCHITECTURE.md",
)

# **`jsonl` before `json`, and it matters.** Regex alternation is ordered, so `(?:json|jsonl)`
# matches the `json` prefix of `register.jsonl` and silently reports a `register.json` that has
# never existed. This guard reported exactly that phantom on its first run.
CITATION = re.compile(r"data/([A-Za-z0-9_]+\.(?:jsonl|json))\b")
WRITE = re.compile(r"write_text|json\.dump|\.open\(\s*[\"']w")

NOT_PRODUCED_HERE = {
    # Append-only logs written by the live cycle rather than by a regenerating command. Re-running
    # anything that "produced" these would mean rewriting history, which is the one thing a
    # hash-chained ledger must never offer.
    "paper_ledger.jsonl",
    "risk_records.jsonl",
    "causal_chains.jsonl",
    "desk_notes.jsonl",
    "book_tape.jsonl",
    "replay_ledger.jsonl",
    "register.jsonl",
    "protocol_commitments.jsonl",
}


def _cited_artefacts() -> dict[str, list[str]]:
    """Every `data/*.json` the public documents point at, and which document points at it."""
    found: dict[str, list[str]] = {}
    for doc in DOCS:
        if not doc.exists():
            continue
        for name in CITATION.findall(doc.read_text(encoding="utf-8")):
            found.setdefault(name, []).append(doc.name)
    return found


def _writers() -> dict[str, list[str]]:
    """Which modules both name an artefact and write a file."""
    out: dict[str, list[str]] = {}
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not WRITE.search(text):
            continue
        for name in set(CITATION.findall(text)) | {
            m for m in re.findall(r"[\"']([A-Za-z0-9_]+\.(?:jsonl|json))[\"']", text)
        }:
            out.setdefault(name, []).append(path.relative_to(SRC).as_posix())
    return out


class TestEveryCitedArtefactCanBeRebuilt:
    def test_the_documents_actually_cite_artefacts(self) -> None:
        """A guard on the guard: if the citation regex stops matching, every test below passes
        vacuously and the protection silently disappears."""
        cited = _cited_artefacts()
        assert len(cited) >= 20, (
            f"only {len(cited)} artefacts found — has the citation form changed?"
        )

    def test_no_cited_artefact_is_without_a_writer(self) -> None:
        cited = _cited_artefacts()
        writers = _writers()
        orphans = {
            name: docs for name, docs in cited.items()
            if name not in NOT_PRODUCED_HERE and name not in writers
        }
        assert not orphans, (
            "these artefacts are cited as evidence and nothing can regenerate them:\n"
            + "\n".join(f"  data/{n} — cited in {', '.join(d)}" for n, d in sorted(orphans.items()))
            + "\n\nA number nobody can recompute is an assertion wearing a filename. Either add a "
              "command that rebuilds it, or stop citing it."
        )

    def test_the_four_that_were_orphaned_all_have_writers_now(self) -> None:
        """Named explicitly, so a regression on any of them reads as a regression rather than as a
        count changing."""
        writers = _writers()
        for name, expected in (
            ("hurdle_clearance.json", "eval/clearance.py"),
            ("overfit_gates.json", "research/overfit_gates.py"),
            ("session_beta.json", "research/session_beta_study.py"),
            ("stress_report.json", "desk/stress.py"),
        ):
            assert name in writers, f"{name} lost its writer"
            assert any(expected in w for w in writers[name]), (
                f"{name} is written by {writers[name]}, expected {expected}"
            )

    def test_the_citation_regex_does_not_truncate_jsonl(self) -> None:
        """The guard's own first run reported a `data/register.json` that does not exist, because
        `(?:json|jsonl)` matched the prefix of `register.jsonl`. A phantom orphan is as bad as a
        missed one: it teaches a reader to ignore this test."""
        assert CITATION.findall("see data/register.jsonl for the chain") == ["register.jsonl"]
        assert CITATION.findall("see data/track1_study.json here") == ["track1_study.json"]

    def test_every_cited_artefact_actually_exists_on_disk(self) -> None:
        """A citation to a file that was never written is a dead link in the evidence trail."""
        data = ARGUS / "data"
        if not data.exists():
            pytest.skip("no data directory on this machine")
        missing = sorted(n for n in _cited_artefacts() if not (data / n).exists())
        assert not missing, f"cited in the documents but absent from data/: {', '.join(missing)}"

    def test_append_only_logs_are_excluded_deliberately_not_accidentally(self) -> None:
        """The exclusion list must stay small and must stay about append-only history.

        If a future artefact is quietly added here to silence a failure, that is the defect coming
        back wearing an allowlist.
        """
        assert len(NOT_PRODUCED_HERE) <= 10, (
            "the exclusion list is growing — is it being used to hide orphans?"
        )
        for name in NOT_PRODUCED_HERE:
            assert name.endswith(".jsonl"), (
                f"{name} is excluded as an append-only log but is not a .jsonl — a regenerable "
                f"snapshot does not belong on this list"
            )


class TestTheWritersAreReachableAsCommands:
    """A writer buried in a library nobody can invoke is not a command."""

    @pytest.mark.parametrize(
        "module",
        [
            "argus.eval.clearance",
            "argus.research.overfit_gates",
            "argus.research.session_beta_study",
            "argus.desk.stress",
        ],
    )
    def test_the_rebuilt_writers_expose_a_main(self, module: str) -> None:
        import importlib

        got = importlib.import_module(module)
        assert callable(getattr(got, "main", None)), f"{module} has no main() to run"

    @pytest.mark.parametrize(
        "module",
        [
            "argus.eval.clearance",
            "argus.research.overfit_gates",
            "argus.research.session_beta_study",
        ],
    )
    def test_the_new_writers_name_where_they_write(self, module: str) -> None:
        import importlib

        got = importlib.import_module(module)
        path = getattr(got, "REPORT_PATH", None)
        assert path is not None, f"{module} does not expose REPORT_PATH"
        assert path.parent.name == "data", f"{module} writes outside data/: {path}"


class TestALiveClaimMustCiteSomethingCheckable:
    """The gap the artefact sweep above could not see.

    `desk/diversification.py` had **no entry point at all**, and README quoted four figures from it
    — *"2.24 effective bets across 3 positions, torsion 2.77, best hedge QQQUSDT at -0.441 removing
    34% of the variance"* — with no artefact holding them and no command producing them. The sweep
    above checks artefacts the documents *cite*; a claim citing only a module name cites no
    artefact, so it passed.

    Rebuilding it found the numbers do not reproduce: the **book that produced them was never
    recorded**, so nobody could have re-run it even with a command. The claim now carries its own
    inputs.

    A number with neither a file nor a command behind it is the least checkable thing this project
    can publish.
    """

    # **Case-sensitive, and anchored on the marker itself.** The first version was
    # `Live[^:]*:` with IGNORECASE, which matched ordinary prose such as "live Bitget
    # candles, no key:" and flagged fourteen innocent lines. A guard that cries wolf is a
    # guard a reader learns to skip.
    LIVE = re.compile(r"\bLive(?:\s*\([^)]{0,80}\))?:")

    def _live_lines(self) -> list[tuple[str, int, str]]:
        out: list[tuple[str, int, str]] = []
        for doc in DOCS:
            if not doc.exists():
                continue
            for n, line in enumerate(doc.read_text(encoding="utf-8").split("\n"), 1):
                if self.LIVE.search(line):
                    out.append((doc.name, n, line))
        return out

    def test_there_are_live_claims_to_check(self) -> None:
        """A guard on the guard: if the phrasing changes, this class must not pass vacuously."""
        assert self._live_lines(), "no 'Live:' claims found — has the phrasing changed?"

    def test_every_live_claim_cites_an_artefact_or_a_command(self) -> None:
        naked = [
            f"{doc}:{n}" for doc, n, line in self._live_lines()
            if not CITATION.search(line) and "python -m" not in line
        ]
        assert not naked, (
            "these lines present a live measurement and cite neither an artefact nor a command "
            "that produces it:\n" + "\n".join(f"  {ref}" for ref in naked)
            + "\n\nA reader cannot check a number that points at nothing."
        )
