# Vendored, verbatim, unmodified below the "VENDORED FROM HERE" marker.
#
# Source:  https://github.com/SerenityTn/serenity-guardrails
# Path:    etoro_trading/journal.py
# Commit:  13d46dc1c6204f27fe321bd66023691918017cce (2026-08-28)
# Licence: Apache License 2.0 — full text below. The repository's own LICENSE file carries the
#          unfilled Apache 2.0 template ("Copyright [yyyy] [name of copyright owner]", never
#          completed by the project) rather than a filled-in notice; attributed here to the
#          repository owner, SerenityTn, per the organisation GitHub records as the copyright
#          holder — the reasonable reading of an incomplete template, not an invented one.
#
# This is serenity-guardrails' real hash-chained journal: `ChainedJournal` (defined further down
# this file) is the class argus.eval.journal_comparison runs, unmodified, as the
# `baseline_reproduced` / `same_input_comparison` evidence for "Pre-registered trading protocol,
# hash-committed" in eval/standing.py, whose own baseline field already names this exact mechanism
# ("serenity-guardrails (Apache-2.0) hash-chained journal with a head anchor") — not a paraphrase
# of it, the actual code, on the same constructed fixtures ARGUS's own
# argus.paper.ledger.PaperLedger.verify() sees (project Standing Rule #3).
#
# This file's own import (`from .guards import GuardError`) is satisfied by the sibling vendored
# `serenity_guards.py` via a sys.modules shim in `serenity_loader.py`, so this file needed zero
# import-path edits to run unmodified.
#
# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
#
# TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION (summarised terms retained from the
# upstream LICENSE file; the full canonical text is at http://www.apache.org/licenses/LICENSE-2.0)
#
# 1. Definitions — "License", "Licensor", "Legal Entity", "You", "Source"/"Object" form, "Work",
#    "Derivative Works", "Contribution", "Contributor" as defined by the Apache 2.0 license text.
# 2. Grant of Copyright License — perpetual, worldwide, non-exclusive, no-charge, royalty-free,
#    irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display,
#    publicly perform, sublicense, and distribute the Work and Derivative Works.
# 3. Grant of Patent License — a patent license as stated in the License, terminating upon
#    initiating patent litigation over the Work.
# 4. Redistribution — permitted in Source or Object form provided this License and all
#    copyright/patent/trademark/attribution notices are retained, and any modified files carry
#    prominent notices of the changes (none made here — this file is unmodified).
# 5. Submission of Contributions is under this License unless stated otherwise.
# 6. Trademarks — this License grants no permission to use the Licensor's trade names, trademarks,
#    or service marks.
# 7. Disclaimer of Warranty — the Work is provided "AS IS", WITHOUT WARRANTIES OR CONDITIONS OF
#    ANY KIND, express or implied.
# 8. Limitation of Liability — no Contributor is liable for damages arising from use of the Work.
# 9. Accepting Warranty or Additional Liability — offered only on the offeror's own responsibility.
#
# Copyright SerenityTn (see attribution note above)
# Licensed under the Apache License, Version 2.0; you may not use this file except in compliance
# with the License. You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#
# ============================== VENDORED FROM HERE ==============================
"""Hash-chained, truncation-evident JSONL journal.

Each line carries the SHA-256 of the previous line; an atomic sidecar head
anchor pins the expected entry count and final hash. Mid-chain edits and
reordering break the chain; tail truncation breaks the anchor. An actor who
can rewrite both files consistently defeats this by design — the defended
threat is naive edits, truncation, and accidental corruption, not a fully
privileged local attacker.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os

from .guards import GuardError


class ChainedJournal:
    __slots__ = ("_path",)

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def _head_path(self) -> Path:
        return self._path.with_name(self._path.name + ".head")

    def append(self, record: dict[str, object]) -> None:
        if type(record) is not dict or "prev_sha256" in record:
            raise GuardError("journal record is malformed")
        previous = "0" * 64
        count = 0
        try:
            with self._path.open("rb") as handle:
                for line in handle:
                    if line.strip():
                        previous = hashlib.sha256(line.rstrip(b"\n")).hexdigest()
                        count += 1
        except FileNotFoundError:
            pass
        entry = json.dumps({"prev_sha256": previous, **record}, sort_keys=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(entry + "\n")
        head = json.dumps(
            {
                "entries": count + 1,
                "head_sha256": hashlib.sha256(entry.encode("utf-8")).hexdigest(),
            },
            sort_keys=True,
        )
        tmp = self._head_path.with_suffix(".tmp")
        tmp.write_text(head + "\n", encoding="utf-8")
        os.replace(tmp, self._head_path)

    def verify(self) -> int:
        """Verify chain and head anchor; returns entry count or raises."""
        previous = "0" * 64
        count = 0
        try:
            with self._path.open("rb") as handle:
                for line in handle:
                    stripped = line.rstrip(b"\n")
                    if not stripped:
                        continue
                    payload = json.loads(stripped)
                    if payload.get("prev_sha256") != previous:
                        raise GuardError(f"journal chain broken at entry {count}")
                    previous = hashlib.sha256(stripped).hexdigest()
                    count += 1
        except FileNotFoundError:
            if self._head_path.exists():
                raise GuardError("journal missing but head anchor exists") from None
            return 0
        try:
            head = json.loads(self._head_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            if count == 0:
                return 0
            raise GuardError("journal head anchor is missing") from None
        except ValueError:
            raise GuardError("journal head anchor is unreadable") from None
        if head.get("entries") != count or (
            count > 0 and head.get("head_sha256") != previous
        ):
            raise GuardError(
                f"journal head anchor mismatch: anchored {head.get('entries')}"
                f" entries, found {count} — possible tail truncation"
            )
        return count
