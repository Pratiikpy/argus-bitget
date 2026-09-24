"""External anchoring: independent calendars, honest pending state, verifiable without us."""

from __future__ import annotations

import binascii
import hashlib
import os
from pathlib import Path

import pytest

from argus.paper.anchor import (
    CALENDARS,
    DIGEST_BYTES,
    Anchor,
    AnchorError,
    Receipt,
    anchor,
    load,
    upgrade,
)

LIVE = os.environ.get("ARGUS_LIVE_VENUE") == "1"
live_only = pytest.mark.skipif(not LIVE, reason="set ARGUS_LIVE_VENUE=1 to hit the calendars")

DIGEST = hashlib.sha256(b"argus-test").digest()


def _receipt(calendar: str = "https://a.pool.opentimestamps.org") -> Receipt:
    return Receipt(
        calendar=calendar, proof_hex="f0080102",
        submitted_at="2026-09-13T12:00:00+00:00",
    )


class TestTheClaimIsScopedHonestly:
    def test_no_receipts_means_not_anchored(self) -> None:
        got = Anchor(digest_hex="ab" * 32, receipts=(), failures=("a: timeout",))
        assert not got.anchored
        assert "NOT ANCHORED" in got.render()
        assert got.reference() is None

    def test_a_failed_calendar_is_named_not_dropped(self) -> None:
        """A claim that quietly rests on one operator when three refused is the thing this
        module exists to make impossible."""
        got = Anchor(digest_hex="ab" * 32, receipts=(_receipt(),), failures=("b: URLError",))
        assert "did not answer" in got.render()
        assert got.as_dict()["failures"] == ["b: URLError"]

    def test_it_is_reported_as_pending_never_as_confirmed(self) -> None:
        got = Anchor(digest_hex="ab" * 32, receipts=(_receipt(),))
        assert got.as_dict()["status"] == "pending-bitcoin-confirmation"
        assert "PENDING until a Bitcoin block" in got.render()

    def test_it_states_what_it_does_not_prove(self) -> None:
        """A timestamp proves existence before a time. It does not prove the protocol was good."""
        text = Anchor(digest_hex="ab" * 32, receipts=(_receipt(),)).render()
        assert "does not prove" in text
        assert "was followed" in text

    def test_it_points_at_a_verifier_that_is_not_us(self) -> None:
        """An anchor only our own software can check is not an anchor."""
        text = Anchor(digest_hex="ab" * 32, receipts=(_receipt(),)).render()
        assert "ots verify" in text

    def test_independent_calendars_are_counted_not_receipts(self) -> None:
        """Two proofs from one operator are one operator's word, twice."""
        one = Anchor(digest_hex="ab" * 32, receipts=(_receipt(), _receipt()))
        assert one.independent_calendars == 1
        two = Anchor(
            digest_hex="ab" * 32,
            receipts=(_receipt("https://a.pool.opentimestamps.org"),
                      _receipt("https://b.pool.opentimestamps.org")),
        )
        assert two.independent_calendars == 2


class TestTheSubmission:
    def test_a_wrong_length_digest_is_refused(self) -> None:
        with pytest.raises(AnchorError, match="32-byte"):
            anchor(b"too short")

    def test_the_digest_size_is_the_protocol_size(self) -> None:
        assert DIGEST_BYTES == hashlib.sha256(b"").digest_size == 32

    def test_hex_and_bytes_are_both_accepted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import argus.paper.anchor as mod

        # The directory only needs to exist so `_store` can write to it — this test is about
        # digest-form acceptance, not storage, so it uses `tmp_path` like every other test in
        # this file rather than a placeholder path. `Path("/nonexistent-x")` used to stand in
        # here: on Windows that resolves under the current drive and `mkdir` silently succeeds
        # (leaving a stray directory behind), but on Linux `/` is root-owned and `mkdir` raises
        # `PermissionError` — a platform difference this test never meant to exercise.
        monkeypatch.setattr(mod, "_submit", lambda cal, d, *, timeout: b"\xf0\x08proof")
        as_bytes = anchor(DIGEST, calendars=("https://x",), directory=tmp_path)
        assert as_bytes.digest_hex == binascii.hexlify(DIGEST).decode()

    def test_every_calendar_is_tried_even_after_one_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import argus.paper.anchor as mod

        tried: list[str] = []

        def _flaky(cal: str, digest: bytes, *, timeout: int) -> bytes:
            tried.append(cal)
            if "a." in cal:
                raise TimeoutError("slow")
            return b"\xf0\x08proof"

        monkeypatch.setattr(mod, "_submit", _flaky)
        got = anchor(DIGEST, directory=tmp_path)
        assert len(tried) == len(CALENDARS)
        assert got.anchored and got.failures

    def test_an_empty_proof_counts_as_a_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import argus.paper.anchor as mod

        monkeypatch.setattr(mod, "_submit", lambda cal, d, *, timeout: b"")
        got = anchor(DIGEST, calendars=("https://x",), directory=tmp_path)
        assert not got.anchored and "empty proof" in got.failures[0]

    def test_four_independent_calendars_are_configured(self) -> None:
        """A single calendar is a single party to trust, which is what this file avoids."""
        assert len(CALENDARS) >= 4
        assert len({c.split("//")[1].split(".")[-2] for c in CALENDARS}) >= 2


class TestTheProofsAreStoredVerbatim:
    def test_the_receipt_is_preserved_inside_a_standard_wrapper(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """**This test used to assert the defect, with a correct principle stated backwards.**

        It read ``written[0].read_bytes() == payload`` under the docstring *"a proof reformatted
        for our convenience is a proof only we can verify"*. The principle is right and the
        conclusion was inverted: an ``.ots`` file is the calendar's timestamp **preceded by** the
        header magic, a version, the file-hash op and the digest. Writing the bare receipt is not
        fidelity to the format, it is omission of it, and the reference client rejected all 48
        proofs on the magic because of it.

        So the property asserted now is the one that was meant: the calendar's bytes reach disk
        unaltered, *and* the file around them is the standard one. Both halves, or the claim that
        anyone can verify the register without our cooperation is false again.
        """
        import argus.paper.anchor as mod

        payload = b"\xf0\x08\x01\x02\x03"
        monkeypatch.setattr(mod, "_submit", lambda cal, d, *, timeout: payload)
        got = anchor(DIGEST, calendars=("https://x",), directory=tmp_path)
        written = list(tmp_path.glob("*.ots"))
        assert len(written) == 1
        blob = written[0].read_bytes()
        assert blob.endswith(payload), "the calendar's own bytes must reach disk unaltered"
        assert blob.startswith(
            b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
        ), "and inside the file the reference client looks for"
        assert DIGEST in blob, "attesting the digest that was actually anchored"
        assert got.anchored

    def test_a_manifest_is_written_beside_the_proofs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import argus.paper.anchor as mod

        monkeypatch.setattr(mod, "_submit", lambda cal, d, *, timeout: b"\xf0\x08p")
        anchor(DIGEST, calendars=("https://x",), directory=tmp_path, subject="a protocol")
        manifests = list(tmp_path.glob("*.json"))
        assert len(manifests) == 1
        assert "a protocol" in manifests[0].read_text(encoding="utf-8")

    def test_it_round_trips_through_load(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import argus.paper.anchor as mod

        monkeypatch.setattr(mod, "_submit", lambda cal, d, *, timeout: b"\xf0\x08p")
        original = anchor(DIGEST, calendars=("https://x",), directory=tmp_path)
        again = load(original.digest_hex, directory=tmp_path)
        assert again is not None
        assert again.digest_hex == original.digest_hex
        assert again.independent_calendars == original.independent_calendars

    def test_an_unanchored_digest_is_not_stored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A file on disk would read as evidence of an anchor that does not exist."""
        import argus.paper.anchor as mod

        monkeypatch.setattr(mod, "_submit", lambda cal, d, *, timeout: b"")
        anchor(DIGEST, calendars=("https://x",), directory=tmp_path)
        assert not list(tmp_path.glob("*"))

    def test_loading_something_never_anchored_returns_none(self, tmp_path: Path) -> None:
        assert load("de" * 32, directory=tmp_path) is None


class TestUpgradeRefusesToReimplementVerification:
    def test_it_gives_the_command_rather_than_the_answer(self) -> None:
        got = upgrade(Anchor(digest_hex="ab" * 32, receipts=(_receipt(),)))
        assert "ots upgrade" in got["how_to_complete"]
        assert "ots verify" in got["how_to_complete"]

    def test_it_says_why_it_does_not_verify_here(self) -> None:
        """A second implementation would agree with itself and prove nothing."""
        got = upgrade(Anchor(digest_hex="ab" * 32, receipts=(_receipt(),)))
        assert "prove nothing" in got["why_not_here"]

    def test_an_unanchored_digest_reports_not_anchored(self) -> None:
        got = upgrade(Anchor(digest_hex="ab" * 32, receipts=()))
        assert got["status"] == "not-anchored"


@live_only
class TestAgainstTheRealCalendars:
    def test_every_configured_calendar_accepts_a_digest(self) -> None:
        got = anchor(hashlib.sha256(b"argus-live-probe").digest())
        assert got.anchored
        assert got.independent_calendars >= 2, got.failures

    def test_the_returned_proof_is_a_real_ots_structure(self) -> None:
        """OpenTimestamps proofs begin with a length-prefixed op. A bare echo would not."""
        got = anchor(hashlib.sha256(b"argus-live-shape").digest())
        assert got.receipts
        assert len(got.receipts[0].proof_bytes) > 32
