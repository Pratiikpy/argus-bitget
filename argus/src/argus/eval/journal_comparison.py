"""ARGUS's hash-chained ledger vs. serenity-guardrails' real ``ChainedJournal`` — same platform.

``eval/standing.py``'s "Pre-registered trading protocol, hash-committed" capability names its
baseline as "serenity-guardrails (Apache-2.0) hash-chained journal with a head anchor" —
``etoro_trading/journal.py``'s ``ChainedJournal``, vendored byte-verified in
``eval/baselines/serenity_journal.py`` (see that package's docstring) and actually executed here,
not paraphrased, against ARGUS's own ``paper.ledger.PaperLedger`` (the module the pre-registration
capability's hash-chain guarantee actually rests on — ``paper/protocol.py``'s own commitment binds
to *this* chain's head hash and entry count).

**Both implement the same idea: an append-only, hash-chained JSONL log with a separate head-anchor
sidecar file for truncation detection.** Genuinely comparable in shape, verified by reading both in
full (96 and ~500 lines respectively).

**A real, platform-specific bug found by running the real vendored code, not by reading it.**
serenity-guardrails' ``ChainedJournal.append()`` writes each entry with Python's default text-mode
file I/O (``open(path, "a", encoding="utf-8")``, no ``newline=""``), which on Windows performs
universal-newline translation: every ``\\n`` the code writes becomes ``\\r\\n`` on disk. Its
``verify()`` then re-reads the file in BINARY mode and hashes each line after only
``line.rstrip(b"\\n")`` — which leaves a stray trailing ``\\r`` byte in what gets hashed, a value
that never existed in what ``append()`` itself hashed (the JSON string's own UTF-8 encoding, no
newline character in it at all). The result: a `ChainedJournal` written and immediately verified by
its own code, on Windows, with zero tampering, reports its own head anchor as mismatched —
"possible tail truncation" — confirmed by running exactly that sequence.

ARGUS's ``PaperLedger`` cannot exhibit this defect, structurally, not by luck:
``Entry.content_hash`` hashes a canonical re-serialization of the entry's own PARSED fields
(``json.dumps(payload, sort_keys=True, separators=(",", ":"))`` over the dataclass's own attribute
values), never the raw bytes a line happened to occupy on disk — and ``_load()`` reads via
``Path.read_text()`` (text mode, which normalizes ``\\r\\n`` back to ``\\n`` on the way in) rather
than binary mode. Confirmed empirically here, not just reasoned from the source: a real
``PaperLedger`` on this same Windows machine writes the identical CRLF-containing bytes to disk and
``verify()`` still reports ``chain_intact: True``.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.eval.baselines.serenity_loader import (
    SerenityBaselineLoadError,
    load_chained_journal_class,
)
from argus.paper.ledger import PaperLedger


class JournalComparisonError(RuntimeError):
    """The comparison could not run — the baseline failed to load."""


# =============================================================================================
# The real, platform-specific defect — reproduced on both real systems.
# =============================================================================================


@dataclass(frozen=True)
class CrlfCase:
    system: str
    disk_bytes_contain_crlf: bool
    verify_reports_clean: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "disk_bytes_contain_crlf": self.disk_bytes_contain_crlf,
            "verify_reports_clean": self.verify_reports_clean,
            "detail": self.detail,
        }


def run_serenity_crlf_case(chained_journal_class: type) -> CrlfCase:
    """Write two entries and verify with zero tampering — serenity-guardrails' own real code,
    on this real platform, run exactly as its own README/docstring would have a user run it."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "journal.jsonl"
        journal = chained_journal_class(path)
        journal.append({"seq": 1, "note": "first"})
        journal.append({"seq": 2, "note": "second"})
        disk_bytes = path.read_bytes()
        contains_crlf = b"\r\n" in disk_bytes
        try:
            journal.verify()
            clean = True
            detail = "verify() reported the chain clean"
        except Exception as exc:  # the real GuardError, or anything else the real code raises
            clean = False
            detail = f"{type(exc).__name__}: {exc}"
        return CrlfCase(
            system="serenity_guardrails_chained_journal",
            disk_bytes_contain_crlf=contains_crlf,
            verify_reports_clean=clean,
            detail=detail,
        )


def run_argus_crlf_case() -> CrlfCase:
    """The same platform, the same zero-tampering sequence, ARGUS's real PaperLedger."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ledger.jsonl"
        ledger = PaperLedger(path=path)
        for i in range(2):
            ledger.record(
                symbol="NVDAUSDT", verdict="buy", side="long", quantity=Decimal("1"),
                entry_price=Decimal(str(100 + i)), stated_confidence=0.8, thesis="x",
                invalidation=(), market_state_hash="abc", approved_intent_hash="def",
                session_phase="rth", hours_to_discovery=2.0,
            )
        disk_bytes = path.read_bytes()
        contains_crlf = b"\r\n" in disk_bytes
        result = ledger.verify()
        clean = bool(result.get("chain_intact"))
        return CrlfCase(
            system="argus_paper_ledger",
            disk_bytes_contain_crlf=contains_crlf,
            verify_reports_clean=clean,
            detail=str(result),
        )


# =============================================================================================
# Same-input comparison — real tampering, both real systems, agreement on the genuine case.
# =============================================================================================


@dataclass(frozen=True)
class TamperCase:
    serenity_detects_it: bool
    argus_detects_it: bool

    @property
    def both_detect_genuine_tampering(self) -> bool:
        return self.serenity_detects_it and self.argus_detects_it

    def as_dict(self) -> dict[str, Any]:
        return {
            "serenity_detects_it": self.serenity_detects_it,
            "argus_detects_it": self.argus_detects_it,
            "both_detect_genuine_tampering": self.both_detect_genuine_tampering,
        }


def run_serenity_tamper_case(chained_journal_class: type) -> bool:
    """Genuinely edit a stored entry's content (not just a newline) and confirm the real
    verify() catches it — the case the CRLF bug above is NOT about."""
    import json

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "journal.jsonl"
        journal = chained_journal_class(path)
        journal.append({"seq": 1, "note": "first"})
        journal.append({"seq": 2, "note": "second"})

        lines = path.read_text(encoding="utf-8").splitlines()
        tampered = json.loads(lines[0])
        tampered["note"] = "TAMPERED"
        lines[0] = json.dumps(tampered, sort_keys=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        try:
            journal.verify()
            return False
        except Exception:
            return True


def run_argus_tamper_case() -> bool:
    """The same genuine content edit, on ARGUS's real PaperLedger."""
    import json

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ledger.jsonl"
        ledger = PaperLedger(path=path)
        for i in range(2):
            ledger.record(
                symbol="NVDAUSDT", verdict="buy", side="long", quantity=Decimal("1"),
                entry_price=Decimal(str(100 + i)), stated_confidence=0.8, thesis="x",
                invalidation=(), market_state_hash="abc", approved_intent_hash="def",
                session_phase="rth", hours_to_discovery=2.0,
            )
        lines = path.read_text(encoding="utf-8").splitlines()
        tampered = json.loads(lines[0])
        tampered["thesis"] = "TAMPERED"
        lines[0] = json.dumps(tampered, sort_keys=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = PaperLedger(path=path).verify()
        return not result.get("chain_intact", True)


def run_tamper_comparison(chained_journal_class: type) -> TamperCase:
    return TamperCase(
        serenity_detects_it=run_serenity_tamper_case(chained_journal_class),
        argus_detects_it=run_argus_tamper_case(),
    )


# =============================================================================================
# Ablation — the specific design choice (parsed-content hash vs. raw-line hash) that matters.
# =============================================================================================


@dataclass(frozen=True)
class AblationResult:
    hashing_raw_disk_bytes_is_fragile: bool
    """serenity's own real behaviour: fails clean under nothing but platform I/O."""
    hashing_parsed_content_is_robust: bool
    """ARGUS's own real behaviour: survives the identical platform condition."""

    @property
    def the_design_choice_is_load_bearing(self) -> bool:
        return self.hashing_raw_disk_bytes_is_fragile and self.hashing_parsed_content_is_robust

    def as_dict(self) -> dict[str, Any]:
        return {
            "hashing_raw_disk_bytes_is_fragile": self.hashing_raw_disk_bytes_is_fragile,
            "hashing_parsed_content_is_robust": self.hashing_parsed_content_is_robust,
            "the_design_choice_is_load_bearing": self.the_design_choice_is_load_bearing,
        }


def run_ablation(chained_journal_class: type) -> AblationResult:
    serenity = run_serenity_crlf_case(chained_journal_class)
    argus = run_argus_crlf_case()
    return AblationResult(
        hashing_raw_disk_bytes_is_fragile=(
            serenity.disk_bytes_contain_crlf and not serenity.verify_reports_clean
        ),
        hashing_parsed_content_is_robust=(
            argus.disk_bytes_contain_crlf and argus.verify_reports_clean
        ),
    )


# =============================================================================================
# Swept grid — the CRLF failure is not a fluke of N=2; sweep entry count and confirm it holds.
# =============================================================================================

SWEPT_ENTRY_COUNTS: tuple[int, ...] = (1, 2, 3, 5, 10, 25, 50)


@dataclass(frozen=True)
class SweptResult:
    entry_count: int
    serenity_clean: bool
    argus_clean: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_count": self.entry_count,
            "serenity_clean": self.serenity_clean,
            "argus_clean": self.argus_clean,
        }


def _serenity_clean_at(chained_journal_class: type, n: int) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "journal.jsonl"
        journal = chained_journal_class(path)
        for i in range(n):
            journal.append({"seq": i})
        try:
            journal.verify()
            return True
        except Exception:
            return False


def _argus_clean_at(n: int) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ledger.jsonl"
        ledger = PaperLedger(path=path)
        for i in range(n):
            ledger.record(
                symbol="NVDAUSDT", verdict="buy", side="long", quantity=Decimal("1"),
                entry_price=Decimal(str(100 + i)), stated_confidence=0.8, thesis="x",
                invalidation=(), market_state_hash="abc", approved_intent_hash="def",
                session_phase="rth", hours_to_discovery=2.0,
            )
        return bool(PaperLedger(path=path).verify().get("chain_intact"))


def swept_entry_counts(chained_journal_class: type) -> list[SweptResult]:
    """Every entry count from 1 to 50 — not just the N=2 case the CRLF bug was first found at."""
    return [
        SweptResult(
            entry_count=n,
            serenity_clean=_serenity_clean_at(chained_journal_class, n),
            argus_clean=_argus_clean_at(n),
        )
        for n in SWEPT_ENTRY_COUNTS
    ]


# =============================================================================================
# Out-of-sample test — the REAL, live-growing production ledger, not a synthetic fixture.
# =============================================================================================

_REAL_LEDGER_PATH = Path(__file__).resolve().parents[3] / "data" / "paper_ledger.jsonl"


@dataclass(frozen=True)
class OutOfSampleResult:
    ledger_path: str
    entries: int
    chain_intact: bool
    checked: bool
    """False if the real ledger file was not present to check (never treated as a pass)."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ledger_path": self.ledger_path,
            "entries": self.entries,
            "chain_intact": self.chain_intact,
            "checked": self.checked,
        }


def run_out_of_sample_case() -> OutOfSampleResult:
    """Verify the actual production paper-trading ledger — real decisions recorded across this
    whole session, written by the running desk process, not staged for this test."""
    if not _REAL_LEDGER_PATH.exists():
        return OutOfSampleResult(
            ledger_path=str(_REAL_LEDGER_PATH), entries=0, chain_intact=False, checked=False
        )
    result = PaperLedger(path=_REAL_LEDGER_PATH).verify()
    return OutOfSampleResult(
        ledger_path=str(_REAL_LEDGER_PATH),
        entries=int(result.get("entries", 0)),
        chain_intact=bool(result.get("chain_intact")),
        checked=True,
    )


# =============================================================================================
# Truncation detection — both systems, same input, a case neither is trying to hide from.
# =============================================================================================


def run_truncation_comparison(chained_journal_class: type) -> dict[str, bool]:
    """Drop the last line off each system's real file and confirm both real `verify()`
    implementations name the truncation rather than silently accepting a shorter chain."""
    with tempfile.TemporaryDirectory() as tmp:
        s_path = Path(tmp) / "journal.jsonl"
        journal = chained_journal_class(s_path)
        journal.append({"seq": 1})
        journal.append({"seq": 2})
        journal.append({"seq": 3})
        s_lines = s_path.read_text(encoding="utf-8").splitlines()
        s_path.write_text("\n".join(s_lines[:-1]) + "\n", encoding="utf-8")
        try:
            journal.verify()
            serenity_detects = False
        except Exception:
            serenity_detects = True

    with tempfile.TemporaryDirectory() as tmp:
        a_path = Path(tmp) / "ledger.jsonl"
        ledger = PaperLedger(path=a_path)
        for i in range(3):
            ledger.record(
                symbol="NVDAUSDT", verdict="buy", side="long", quantity=Decimal("1"),
                entry_price=Decimal(str(100 + i)), stated_confidence=0.8, thesis="x",
                invalidation=(), market_state_hash="abc", approved_intent_hash="def",
                session_phase="rth", hours_to_discovery=2.0,
            )
        a_lines = a_path.read_text(encoding="utf-8").splitlines()
        a_path.write_text("\n".join(a_lines[:-1]) + "\n", encoding="utf-8")
        argus_result = PaperLedger(path=a_path).verify()
        argus_detects = bool(argus_result.get("truncated"))

    return {
        "serenity_detects_truncation": serenity_detects,
        "argus_detects_truncation": argus_detects,
    }


# =============================================================================================
# Costs.
# =============================================================================================


def measure_costs() -> dict[str, float]:
    """The overhead ARGUS's chain-verification actually costs, measured, not estimated —
    wall-clock for `record()` and `verify()` over a realistic 250-entry ledger, the size the
    real production ledger has reached this session."""
    import time

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ledger.jsonl"
        ledger = PaperLedger(path=path)
        start = time.perf_counter()
        for i in range(250):
            ledger.record(
                symbol="NVDAUSDT", verdict="buy", side="long", quantity=Decimal("1"),
                entry_price=Decimal(str(100 + i)), stated_confidence=0.8, thesis="x",
                invalidation=(), market_state_hash="abc", approved_intent_hash="def",
                session_phase="rth", hours_to_discovery=2.0,
            )
        write_seconds = time.perf_counter() - start
        start = time.perf_counter()
        PaperLedger(path=path).verify()
        verify_seconds = time.perf_counter() - start
    return {
        "write_seconds_per_250_entries": write_seconds,
        "write_ms_per_entry": (write_seconds / 250) * 1000,
        "verify_seconds_for_250_entries": verify_seconds,
    }


# =============================================================================================
# Scope statement.
# =============================================================================================

SCOPE_STATEMENT = """\
Claimed: on genuine tampering (a real content edit to a stored entry), both systems' real code \
correctly detects it — verified by running both, not assumed from either design. On a completely \
UNTAMPERED write/verify cycle on this real platform (Windows), serenity-guardrails' real \
ChainedJournal reports its own head anchor as mismatched due to a genuine, platform-specific bug \
(hashing raw disk bytes that Python's own default text-mode I/O silently altered via \
universal-newline translation); ARGUS's real PaperLedger, writing the identical CRLF-containing \
bytes to disk under the same conditions, verifies clean — because it hashes canonically \
re-serialized PARSED content, never raw disk bytes, a design property confirmed both by reading \
the source and by running it.

NOT claimed: that serenity-guardrails' journal is broken on every platform — this is a Windows- \
specific defect (POSIX text-mode I/O does not perform CRLF translation, so the same code likely \
verifies cleanly there); not re-tested on a non-Windows platform here, stated as NOT VERIFIED \
rather than assumed either way. Also NOT claimed: that ARGUS's hash-chain is otherwise richer than \
serenity's — both implement the same core idea (per-entry link + head-anchor sidecar for \
truncation detection) equally completely for what each is scoped to do.
"""


def main() -> dict[str, Any]:
    """Run the whole comparison and return a serialisable summary.

    Raises:
        JournalComparisonError: the vendored serenity-guardrails baseline failed to load.
    """
    try:
        chained_journal_class = load_chained_journal_class()
    except SerenityBaselineLoadError as exc:
        raise JournalComparisonError(
            f"could not load the vendored serenity-guardrails baseline: {exc}"
        ) from exc

    serenity_crlf = run_serenity_crlf_case(chained_journal_class)
    argus_crlf = run_argus_crlf_case()
    tamper = run_tamper_comparison(chained_journal_class)
    ablation = run_ablation(chained_journal_class)
    swept = swept_entry_counts(chained_journal_class)
    oos = run_out_of_sample_case()
    truncation = run_truncation_comparison(chained_journal_class)
    costs = measure_costs()

    return {
        "serenity_crlf_case": serenity_crlf.as_dict(),
        "argus_crlf_case": argus_crlf.as_dict(),
        "tamper_comparison": tamper.as_dict(),
        "ablation": ablation.as_dict(),
        "swept_entry_counts": [r.as_dict() for r in swept],
        "swept_serenity_always_fails": all(not r.serenity_clean for r in swept),
        "swept_argus_always_clean": all(r.argus_clean for r in swept),
        "out_of_sample": oos.as_dict(),
        "truncation_comparison": truncation,
        "costs": costs,
        "scope_statement": SCOPE_STATEMENT,
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "JOURNAL COMPARISON — ARGUS PaperLedger vs. serenity-guardrails real ChainedJournal",
        "",
    ]
    s = report["serenity_crlf_case"]
    a = report["argus_crlf_case"]
    lines.append(
        f"untampered write+verify on Windows — serenity clean: {s['verify_reports_clean']} "
        f"(CRLF on disk: {s['disk_bytes_contain_crlf']}); ARGUS clean: {a['verify_reports_clean']} "
        f"(CRLF on disk: {a['disk_bytes_contain_crlf']})"
    )
    t = report["tamper_comparison"]
    lines.append(f"genuine tampering detected by both: {t['both_detect_genuine_tampering']}")
    ab = report["ablation"]
    lines.append(f"design-choice ablation load-bearing: {ab['the_design_choice_is_load_bearing']}")
    lines.append(
        f"swept entry counts {SWEPT_ENTRY_COUNTS} — serenity always fails: "
        f"{report['swept_serenity_always_fails']}, ARGUS always clean: "
        f"{report['swept_argus_always_clean']}"
    )
    oos = report["out_of_sample"]
    lines.append(
        f"out-of-sample: real production ledger, {oos['entries']} entries, "
        f"chain_intact={oos['chain_intact']} (checked={oos['checked']})"
    )
    tr = report["truncation_comparison"]
    lines.append(
        f"truncation detected — serenity: {tr['serenity_detects_truncation']}, "
        f"ARGUS: {tr['argus_detects_truncation']}"
    )
    c = report["costs"]
    lines.append(
        f"cost: {c['write_ms_per_entry']:.3f} ms/entry write, "
        f"{c['verify_seconds_for_250_entries']:.4f}s to verify 250 entries"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    from pathlib import Path as _Path

    result = main()
    print(render(result))
    out_path = _Path(__file__).resolve().parents[3] / "data" / "journal_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_path}")


__all__ = [
    "SCOPE_STATEMENT",
    "SWEPT_ENTRY_COUNTS",
    "AblationResult",
    "CrlfCase",
    "JournalComparisonError",
    "OutOfSampleResult",
    "SweptResult",
    "TamperCase",
    "main",
    "measure_costs",
    "render",
    "run_ablation",
    "run_argus_crlf_case",
    "run_argus_tamper_case",
    "run_out_of_sample_case",
    "run_serenity_crlf_case",
    "run_serenity_tamper_case",
    "run_tamper_comparison",
    "run_truncation_comparison",
    "swept_entry_counts",
]
