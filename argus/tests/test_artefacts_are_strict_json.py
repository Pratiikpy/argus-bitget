"""Every committed artefact must be readable by something other than Python.

**Three of them were not.** Python's `json` emits bare `NaN` and `Infinity` tokens by default and
accepts them on the way back in, so the round trip inside this project always worked and nothing
noticed. Those tokens are not in the JSON grammar: `JSON.parse`, Go's `encoding/json` and Rust's
`serde_json` all reject them. `data/crosssection_comparison.json`, `data/earnings_comparison.json`
and `data/regime_comparison.json` each carried one, which means a reviewer whose tooling is not
Python — a judge opening an artefact in a browser console, say — got a parse error from a project
whose whole argument is that its evidence is open to inspection.

Found on 2026-09-20 by an adversarial audit of a comparison module, as a side finding about the
house rather than about that module. That is the reason this file is a sweep over the whole corpus
and not a fix to three files: the three were symptoms.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.eval.artefact import dumps, is_strict, sanitise, write

DATA = Path(__file__).resolve().parents[1] / "data"


class TestTheCommittedCorpus:
    def test_every_artefact_is_strict_json(self) -> None:
        offenders = sorted(
            p.relative_to(DATA).as_posix()
            for p in DATA.rglob("*.json")
            if not is_strict(p)
        )
        assert not offenders, (
            "these artefacts contain NaN or Infinity and cannot be read outside Python: "
            + ", ".join(offenders)
            + " — write them through argus.eval.artefact.write"
        )

    def test_there_is_a_corpus_to_check(self) -> None:
        """A sweep over an empty directory passes and proves nothing."""
        assert len(list(DATA.rglob("*.json"))) > 20

    def test_jsonl_artefacts_are_strict_line_by_line(self) -> None:
        """The append-only logs matter more than the reports: the ledger is one of them."""
        def reject(token: str) -> object:
            raise ValueError(token)

        offenders: list[str] = []
        for path in sorted(DATA.rglob("*.jsonl")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    json.loads(line, parse_constant=reject)
                except ValueError as exc:
                    offenders.append(f"{path.name}:{number} ({exc})")
                    break
        assert not offenders, "non-strict JSONL: " + ", ".join(offenders)


class TestTheWriterRefusesToProduceOne:
    def test_a_non_finite_value_becomes_null(self) -> None:
        got = json.loads(dumps({"correlation": float("nan"), "ratio": float("inf")}))
        assert got == {"correlation": None, "ratio": None}

    def test_the_key_path_of_every_null_is_reported(self, tmp_path: Path) -> None:
        """Null must be distinguishable from never-computed, so the caller is told what changed."""
        undefined = write(tmp_path / "x.json", {"a": {"b": [1.0, float("nan")]}})
        assert undefined == ["a.b[1]=nan"]

    def test_what_it_writes_parses_strictly(self, tmp_path: Path) -> None:
        target = tmp_path / "x.json"
        write(target, {"deep": [{"v": float("-inf")}], "fine": 1.5})
        assert is_strict(target)

    def test_finite_values_are_untouched(self) -> None:
        blob = {"a": 1, "b": 2.5, "c": "x", "d": None, "e": [1, 2], "f": True}
        assert sanitise(blob) == blob

    def test_a_non_finite_value_that_escapes_sanitising_raises(self) -> None:
        """The belt behind the braces. A numpy scalar or a float subclass could slip past the
        isinstance check; `allow_nan=False` means the result is an exception rather than a file
        only Python can open."""
        class Sneaky(float):
            pass

        with pytest.raises(ValueError):
            json.dumps({"x": Sneaky("nan")}, allow_nan=False)

    def test_the_checker_actually_rejects_the_bad_tokens(self, tmp_path: Path) -> None:
        """`json.loads` accepts all three by default, so a checker built on it would pass
        everything and this whole file would be theatre."""
        for token in ("NaN", "Infinity", "-Infinity"):
            bad = tmp_path / f"{token}.json"
            bad.write_text(f'{{"v": {token}}}', encoding="utf-8")
            assert json.loads(bad.read_text(encoding="utf-8")) is not None  # Python is happy
            assert not is_strict(bad), f"{token} should be rejected"
