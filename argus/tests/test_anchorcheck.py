"""The anchor proofs must be readable by somebody who does not trust us.

**This file exists because a judge-facing claim was false and nothing would have caught it.** The
documents said the claim register was "anchored to Bitcoin" and that "anyone can verify them with
the reference OpenTimestamps client without our cooperation". On 2026-09-20 an adversarial sweep
checked, and both halves were wrong: all 48 `.ots` files held the raw calendar receipt rather than a
wrapped `DetachedTimestampFile` — so the reference client rejected every one on the header magic —
and not a single proof carried a Bitcoin attestation. They were submissions, not anchors.

The proofs were upgraded and rewritten. These tests are the part that matters afterwards: the claim
is now checked on every run rather than asserted once.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from argus.register.anchorcheck import (
    ANCHOR_DIR,
    Anchor,
    AnchorReport,
    check,
    confirmed_anchors,
    read_anchor,
)


@pytest.fixture(scope="module")
def live() -> AnchorReport:
    return check()


class TestTheLiveProofs:
    def test_every_proof_parses_as_a_standard_opentimestamps_file(
        self, live: AnchorReport
    ) -> None:
        """The defect that made the "verify it yourself" sentence false. Was 0 of 48."""
        unreadable = [a.path for a in live.anchors if not a.readable]
        assert not unreadable, f"unreadable proofs: {unreadable}"

    def test_there_are_proofs_at_all(self, live: AnchorReport) -> None:
        """A green report over an empty directory is the failure mode this guards."""
        assert live.total > 0
        assert live.sound

    def test_most_proofs_are_confirmed_on_bitcoin(self, live: AnchorReport) -> None:
        """Not *all* — a proof written this cycle is legitimately pending for a few hours, and
        demanding 100% would make this test fail on every fresh anchor and train a reader to
        ignore it. But the bulk of a register anchored since 14 September must have landed."""
        assert live.confirmed >= 1
        assert live.confirmed > live.pending, (
            f"only {live.confirmed} of {live.total} proofs are confirmed on Bitcoin; "
            f"the documents describe this register as anchored"
        )

    def test_the_confirmed_count_is_what_the_documents_quote(self, live: AnchorReport) -> None:
        """`eval/docclaims.py` reads this function, so the two cannot drift apart."""
        assert confirmed_anchors() == live.confirmed

    def test_block_heights_are_plausible_bitcoin_heights(self, live: AnchorReport) -> None:
        """A cheap sanity check on the parse: heights near 967,000 in September 2026. A parser
        that silently produced 0 or a nonsense height would otherwise still look green."""
        span = live.block_range
        assert span is not None
        low, high = span
        assert 900_000 < low <= high < 1_200_000


class TestTheCheckerCanActuallyFail:
    """A checker that cannot report a bad proof proves nothing — the same rule the risk prover
    follows with its deliberately broken policies."""

    def test_a_corrupt_file_is_reported_unreadable_rather_than_skipped(
        self, tmp_path: Path
    ) -> None:
        bad = tmp_path / "broken.ots"
        bad.write_bytes(b"this is not a timestamp proof")
        got = read_anchor(bad)
        assert not got.readable
        assert got.error
        assert not got.confirmed

    def test_a_directory_of_corrupt_proofs_is_not_sound(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"{i}.ots").write_bytes(b"garbage")
        report = check(tmp_path)
        assert report.unreadable == 3
        assert report.confirmed == 0
        assert not report.sound
        assert "DO NOT PARSE" in "\n".join(report.render())

    def test_an_empty_directory_is_not_sound(self, tmp_path: Path) -> None:
        """Zero proofs is not zero problems. `sound` requires something to have been checked."""
        assert not check(tmp_path).sound

    def test_a_pending_only_proof_is_not_counted_as_confirmed(self) -> None:
        """The exact pre-2026-09-20 state: readable, but attesting nothing on Bitcoin yet."""
        pending = Anchor("x.ots", readable=True, bitcoin_blocks=(),
                         pending_calendars=("https://a.pool.opentimestamps.org",))
        report = AnchorReport((pending,))
        assert report.confirmed == 0
        assert report.pending == 1
        assert report.sound, "pending is honest, not broken"

    def test_the_real_directory_is_the_one_being_checked(self) -> None:
        """Guards against the check quietly pointing at a fixture directory."""
        assert ANCHOR_DIR.name == "anchors"
        assert ANCHOR_DIR.exists()
