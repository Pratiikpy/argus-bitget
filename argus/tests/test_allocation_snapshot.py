"""Tests for `eval/allocation_snapshot.py` -- the frozen, hashed return history the bake-offs read.

What is pinned is the property a reviewer relies on: the digest is a function of the content and
nothing else (not key order, not whitespace), any edit to a single value is caught on load, and the
committed snapshot still hashes to the digest it was written with -- which is the digest
`data/nco_bakeoff.json` records, so the bake-off's numbers were computed from exactly this file.
Offline; the live fetch is not exercised here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.eval.allocation_snapshot import SNAPSHOT_PATH, SnapshotError, digest, load_snapshot
from argus.eval.artefact import write

STAMPS = ["2026-03-31T08:00:00+00:00", "2026-03-31T09:00:00+00:00", "2026-03-31T10:00:00+00:00"]
COLUMNS = {"QQQUSDT": [0.001, -0.002, 0.0005], "AAPLUSDT": [0.0, 0.003, -0.001]}


def _write(path: Path, stamps: list[str], columns: dict[str, list[float]], sha: str) -> None:
    write(path, {"timestamps": stamps, "columns": columns, "digest": sha}, indent=0)


class TestDigest:
    def test_is_independent_of_column_insertion_order(self) -> None:
        reordered = {"AAPLUSDT": COLUMNS["AAPLUSDT"], "QQQUSDT": COLUMNS["QQQUSDT"]}
        assert digest(STAMPS, COLUMNS) == digest(STAMPS, reordered)

    def test_moves_with_any_single_value(self) -> None:
        nudged = {k: list(v) for k, v in COLUMNS.items()}
        nudged["QQQUSDT"][1] = -0.0020000000000001
        assert digest(STAMPS, nudged) != digest(STAMPS, COLUMNS)

    def test_moves_with_the_timestamps(self) -> None:
        shifted = [*STAMPS[:-1], "2026-03-31T11:00:00+00:00"]
        assert digest(shifted, COLUMNS) != digest(STAMPS, COLUMNS)

    def test_int_and_float_encodings_of_the_same_value_hash_alike(self) -> None:
        as_int = {"QQQUSDT": [0, 1], "AAPLUSDT": [2, 3]}
        as_float = {"QQQUSDT": [0.0, 1.0], "AAPLUSDT": [2.0, 3.0]}
        assert digest(STAMPS[:2], as_int) == digest(STAMPS[:2], as_float)  # type: ignore[arg-type]


class TestLoad:
    def test_round_trips_what_was_written(self, tmp_path: Path) -> None:
        path = tmp_path / "snap.json"
        _write(path, STAMPS, COLUMNS, digest(STAMPS, COLUMNS))
        stamps, columns, sha = load_snapshot(path)
        assert stamps == STAMPS and columns == COLUMNS and sha == digest(STAMPS, COLUMNS)

    def test_refuses_content_edited_after_it_was_written(self, tmp_path: Path) -> None:
        path = tmp_path / "snap.json"
        _write(path, STAMPS, COLUMNS, digest(STAMPS, COLUMNS))
        blob = json.loads(path.read_text(encoding="utf-8"))
        blob["columns"]["AAPLUSDT"][0] = 0.5
        path.write_text(json.dumps(blob), encoding="utf-8")
        with pytest.raises(SnapshotError, match="digest mismatch"):
            load_snapshot(path)

    def test_refuses_a_missing_file_and_says_how_to_make_it(self, tmp_path: Path) -> None:
        with pytest.raises(SnapshotError, match="allocation_snapshot"):
            load_snapshot(tmp_path / "absent.json")


@pytest.mark.skipif(not SNAPSHOT_PATH.exists(), reason="snapshot not generated")
class TestCommittedSnapshot:
    def test_verifies_and_is_the_input_the_bake_off_recorded(self) -> None:
        stamps, columns, sha = load_snapshot()
        blob = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        assert blob["fetch_failures"] == {}
        assert blob["n_aligned_returns"] == len(stamps)
        assert sorted(columns) == blob["symbols"] and len(columns) == 12
        assert all(len(v) == len(stamps) for v in columns.values())
        assert stamps == sorted(stamps) and len(set(stamps)) == len(stamps)
        bakeoff = SNAPSHOT_PATH.parent / "nco_bakeoff.json"
        if bakeoff.exists():
            recorded = json.loads(bakeoff.read_text(encoding="utf-8"))["snapshot"]
            assert recorded["digest"] == sha
            assert recorded["n_bars"] == len(stamps)
            assert (recorded["first"], recorded["last"]) == (stamps[0], stamps[-1])
