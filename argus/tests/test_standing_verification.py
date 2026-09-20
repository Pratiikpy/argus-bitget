"""The standing register must check evidence, not filenames.

**This file exists because the register was graded a rubber stamp and the grade was fair.** Its
`audit()` did exactly four things: the module path exists, the artefact path exists, the test file
exists, and the named test function appears as a *substring* of that file. It never ran a test,
never opened an artefact, never confirmed a comparison had been executed. So "same-input comparison
run", "out-of-sample test", "ablation" and "adversarial test" were each satisfied by a file being on
disk — and 23 of 24 capabilities carried the top grade eight days after the register recorded zero.

`verify()` now opens the artefact and looks for the vocabulary the condition is about. That is a far
weaker statement than OWNED sounds, and it is exactly as strong as the evidence supports. Running it
demoted seven capabilities on the first pass.

Three of the thirteen conditions are honestly not machine-checkable and are labelled ATTESTED rather
than faked — whether the *best* implementation was studied is a judgement about a field, not a
property of a file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.eval.standing import (
    JUDGEMENT_CONDITIONS,
    OWNED_CONDITIONS,
    Proof,
    State,
    audit,
    verify,
)


class TestTheVerifierOpensTheArtefact:
    def test_an_artefact_recording_nothing_about_the_condition_is_unproven(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point. A file on disk is not evidence of an ablation."""
        import argus.eval.standing as standing

        blob = tmp_path / "data" / "empty.json"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_text(json.dumps({"generated_at": "2026-09-20", "note": "nothing here"}))
        monkeypatch.setattr(standing, "PACKAGE", tmp_path)
        status, detail = verify(
            Proof("ablation", "claimed", artefact="data/empty.json"), module_present=True
        )
        assert status == "UNPROVEN"
        assert "records nothing about ablation" in detail

    def test_an_artefact_that_records_the_condition_is_verified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import argus.eval.standing as standing

        blob = tmp_path / "data" / "full.json"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_text(json.dumps({"null_ablation": {"arms": ["on", "off"], "delta": 0.4}}))
        monkeypatch.setattr(standing, "PACKAGE", tmp_path)
        status, _ = verify(
            Proof("ablation", "claimed", artefact="data/full.json"), module_present=True
        )
        assert status == "VERIFIED"

    def test_a_missing_artefact_is_unproven(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import argus.eval.standing as standing

        monkeypatch.setattr(standing, "PACKAGE", tmp_path)
        status, detail = verify(
            Proof("ablation", "claimed", artefact="data/nope.json"), module_present=True
        )
        assert status == "UNPROVEN"
        assert "not on disk" in detail

    def test_nested_keys_are_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Artefacts nest. A predicate that only reads the top level would miss most real ones."""
        import argus.eval.standing as standing

        blob = tmp_path / "data" / "nested.json"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_text(json.dumps({"results": [{"detail": {"out_of_sample": 0.4}}]}))
        monkeypatch.setattr(standing, "PACKAGE", tmp_path)
        status, _ = verify(
            Proof("out_of_sample_test", "claimed", artefact="data/nested.json"),
            module_present=True,
        )
        assert status == "VERIFIED"


class TestWhatTheVerifierHonestlyCannotDo:
    """Named limits, so nobody mistakes this for more than it is."""

    def test_the_three_judgement_conditions_are_attested_not_verified(self) -> None:
        """No predicate over a file can establish that the *best* implementation was studied."""
        for condition in JUDGEMENT_CONDITIONS:
            status, _ = verify(
                Proof(condition, "read repo X at commit abc", artefact="data/doc_claims.json"),
                module_present=True,
            )
            assert status == "ATTESTED", condition

    def test_a_judgement_condition_with_nothing_written_down_is_unproven(self) -> None:
        """An attestation must at least name where somebody looked."""
        status, _ = verify(
            Proof("best_method_studied", "   ", test="test_standing_verification.py"),
            module_present=True,
        )
        assert status == "UNPROVEN"

    def test_a_non_json_artefact_is_attested_rather_than_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A vendored baseline module is the right citation for `baseline_reproduced`, and this
        predicate cannot read Python. Calling it UNPROVEN would report the verifier's blind spot
        as the register's defect — which the first version of this file did, on 20 proofs."""
        import argus.eval.standing as standing

        module = tmp_path / "src" / "baseline.py"
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text("# vendored, byte-exact\n")
        monkeypatch.setattr(standing, "PACKAGE", tmp_path)
        status, detail = verify(
            Proof("baseline_reproduced", "vendored", artefact="src/baseline.py"),
            module_present=True,
        )
        assert status == "ATTESTED"
        assert "non-inspectable" in detail

    def test_every_condition_is_either_checkable_or_named_as_a_judgement(self) -> None:
        """No condition may quietly fall through the middle: each of the thirteen is either in
        SIGNATURES, handled structurally, or declared a judgement."""
        from argus.eval.standing import SIGNATURES

        for condition in OWNED_CONDITIONS:
            covered = condition in SIGNATURES or condition in JUDGEMENT_CONDITIONS
            assert covered, f"{condition} is checked by nothing and declared a judgement by nothing"


class TestTheLiveRegister:
    def test_no_capability_declares_owned_beyond_what_it_can_evidence(self) -> None:
        """The property the demotion established, pinned so it cannot creep back.

        If this fails, either the artefact lost its evidence or somebody promoted an entry by
        editing a state. The fix is the artefact, never this test.
        """
        report = audit()
        overstated = sorted({c.name for c in report.owned} - {c.name for c in report.earned})
        assert not overstated, (
            "declared OWNED without evidence for every condition: " + ", ".join(overstated)
        )

    def test_the_headline_is_the_earned_count(self) -> None:
        from argus.eval.standing import summary

        report = audit()
        assert f"{len(report.earned)} owned" in summary(report)

    def test_verification_covers_every_proof(self) -> None:
        """A proof that produced no verification would be an unchecked claim wearing a checked
        register's badge."""
        report = audit()
        proofs = sum(len(c.proofs) for c in report.capabilities)
        assert len(report.verifications) == proofs

    def test_the_demoted_seven_are_still_in_the_register(self) -> None:
        """Demotion, not deletion. A capability removed from the register stops being measured,
        which is the quiet way a bad result disappears."""
        report = audit()
        implemented = {c.name for c in report.capabilities if c.state is State.IMPLEMENTED}
        assert len(implemented) >= 7
