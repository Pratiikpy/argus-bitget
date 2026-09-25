"""The void harness runs without nocturne's clone: its code vendored verbatim, its inputs frozen."""

from __future__ import annotations

import hashlib

import pytest

from argus.eval import void_comparison as v


def test_the_vendored_code_is_the_frozen_clones_byte_for_byte() -> None:
    recorded = v._frozen()["vendored_sha256"]
    assert set(recorded) == {"core.py", "session.py"}
    for name, digest in recorded.items():
        assert hashlib.sha256((v.VENDORED / name).read_bytes()).hexdigest() == digest
    assert (v.VENDORED / "LICENSE.txt").read_text(encoding="utf-8").startswith("MIT License")


def test_the_frozen_inputs_carry_what_the_harness_reads() -> None:
    frozen = v._frozen()
    assert len(frozen["large"]) == 15
    assert len(frozen["observations"]) == 196
    assert set(frozen["rtoken_closes"]) <= set(frozen["large"])


@pytest.mark.skipif(not v._has_clone(), reason="nocturne's clone is not on this machine")
def test_the_frozen_path_predicts_what_the_clone_does(monkeypatch: pytest.MonkeyPatch) -> None:
    from_clone = v.predictions()
    monkeypatch.setattr(v, "_has_clone", lambda: False)
    v._BARS.clear()
    assert v.predictions() == from_clone
