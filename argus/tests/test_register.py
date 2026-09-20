"""Register tests — the invariants, because the invariants are the entire product.

A register whose claims can be edited, resolved early, or quietly dropped is a database with a story
attached. Every test here exists to make one of those impossible, and each is written against the
failure rather than the feature: not "registration works" but "an unfalsifiable claim is refused",
not "resolution works" but "a resolver run before the horizon cannot see the answer".

The live record is checked too. It is open, it is anchored, and if its chain ever breaks this file
turns red.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from argus.register.claims import (
    MIN_HORIZON_SECONDS,
    REGISTER_PATH,
    Claim,
    Predicate,
    RegisterError,
    Status,
    _read,
    batch_digest,
    make_claim,
    register,
    validate,
    verify_chain,
)
from argus.register.resolve import resolve_one, scoreboard

NOW = datetime(2026, 9, 14, 12, tzinfo=UTC)
LATER = NOW + timedelta(hours=24)


def _claim(**kwargs: object) -> Claim:
    base = {
        "claimant": "tester", "subject": "NVDAUSDT", "predicate": Predicate.CLOSE_ABOVE,
        "threshold": "100", "resolves_at": LATER, "source": "bitget:1H:market",
        "confidence": 0.6, "now": NOW,
    }
    base.update(kwargs)
    return make_claim(**base)  # type: ignore[arg-type]


def _score(*, claimant: str, confidence: float, correct: int, total: int) -> float:
    """Brier for a claimant who stated one confidence across ``total`` claims."""
    return sum(
        (confidence - (1.0 if i < correct else 0.0)) ** 2 for i in range(total)
    ) / total


def _serialise(claim: Claim) -> dict[str, object]:
    """A claim as the resolution file stores it. `slots=True` means there is no ``__dict__``."""
    blob = asdict(claim)
    blob["predicate"] = str(claim.predicate)
    blob["status"] = str(claim.status)
    return blob


class TestRefusalAtRegistration:
    """The one rule that makes this a register and not a comment section."""

    def test_an_unnamed_source_is_refused(self) -> None:
        with pytest.raises(RegisterError, match="point-in-time binding"):
            validate(_claim(source="vibes"))

    def test_a_subject_that_is_not_a_symbol_is_refused(self) -> None:
        with pytest.raises(RegisterError, match="instrument symbol"):
            validate(_claim(subject="the market generally"))

    def test_a_non_numeric_threshold_is_refused(self) -> None:
        with pytest.raises(RegisterError, match="not a decimal"):
            validate(_claim(threshold="higher"))

    def test_a_confidence_outside_zero_to_one_is_refused(self) -> None:
        """An out-of-range probability would corrupt every Brier score computed after it."""
        with pytest.raises(RegisterError, match="not a probability"):
            validate(_claim(confidence=1.4))

    def test_an_anonymous_claim_is_refused(self) -> None:
        with pytest.raises(RegisterError, match="claimant"):
            validate(_claim(claimant="  "))

    def test_a_naive_timestamp_is_refused(self) -> None:
        naive = replace(_claim(), resolves_at="2026-09-15T12:00:00")
        with pytest.raises(RegisterError, match="timezone"):
            validate(naive)

    def test_a_horizon_inside_the_clock_skew_window_is_refused(self) -> None:
        """Below an hour, skew between our host and the venue's stamps could let a claim be
        registered about a bar that has already printed."""
        with pytest.raises(RegisterError, match=str(MIN_HORIZON_SECONDS)):
            validate(_claim(resolves_at=NOW + timedelta(minutes=5)))

    def test_a_refused_claim_never_reaches_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "register.jsonl"
        with pytest.raises(RegisterError):
            register([_claim(source="vibes")], path=path)
        assert not path.exists()


class TestTheChain:
    def test_each_claim_carries_the_hash_of_the_one_before(self, tmp_path: Path) -> None:
        path = tmp_path / "register.jsonl"
        written = register([_claim(), _claim(threshold="200")], path=path)
        assert written[0].prev_hash == ""
        assert written[1].prev_hash == written[0].entry_hash
        assert verify_chain(path)[0]

    def test_an_edited_claim_breaks_the_chain(self, tmp_path: Path) -> None:
        """The failure mode the whole product dies of. Editing must be detectable by a stranger."""
        path = tmp_path / "register.jsonl"
        register([_claim(), _claim(threshold="200")], path=path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["threshold"] = "1"  # a claimant quietly improving their own past
        path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8",
        )
        intact, note = verify_chain(path)
        assert not intact
        assert "hash does not match" in note

    def test_a_deleted_claim_breaks_the_chain(self, tmp_path: Path) -> None:
        path = tmp_path / "register.jsonl"
        register([_claim(), _claim(threshold="200"), _claim(threshold="300")], path=path)
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text(lines[0] + "\n" + lines[2] + "\n", encoding="utf-8")
        assert not verify_chain(path)[0]

    def test_the_hash_covers_the_claim_and_not_its_outcome(self) -> None:
        """The commitment must be checkable *before* anyone knows the answer, so the digest cannot
        depend on the resolution."""
        claim = _claim()
        resolved = replace(claim, status=Status.TRUE, observed="123")
        assert claim.digest() == resolved.digest()

    def test_an_empty_batch_cannot_be_anchored(self) -> None:
        with pytest.raises(RegisterError, match="empty batch"):
            batch_digest([])


class TestTheResolverCannotCheat:
    def test_it_returns_pending_before_the_horizon(self) -> None:
        """Checked before the fetch, so there is no path where the resolver saw the answer and then
        declined to use it."""
        got = resolve_one(_claim(), now=NOW + timedelta(hours=1))
        assert got.status is Status.PENDING
        assert got.observed is None

    def test_an_unreachable_source_is_unresolvable_never_false(self) -> None:
        """Scoring an ungradable claim as a miss would flatter the claimant by turning an unknown
        into a known."""
        claim = _claim(source="nosuchvenue:1H:market")
        got = resolve_one(claim, now=LATER + timedelta(hours=1))
        assert got.status is Status.UNRESOLVABLE
        assert "source unavailable" in str(got.observed)

    def test_an_already_resolved_claim_is_not_regraded(self) -> None:
        resolved = replace(_claim(), status=Status.TRUE, observed="1")
        assert resolve_one(resolved, now=LATER + timedelta(days=5)) is resolved


class TestTheScoreboard:
    def test_brier_punishes_confidence_that_is_not_earned(self) -> None:
        """**The property, stated correctly after the first version of this test was wrong.**

        The original asserted that 0.55-claimed/55%-right should outrank 0.99-claimed/90%-right.
        The arithmetic says otherwise — 0.098 against 0.248 — and the arithmetic is right: Brier
        rewards *resolution* (being informative) as well as *reliability* (being calibrated), and a
        forecaster who is right 90% of the time is far more useful than one who is right 55% of the
        time, even slightly overconfident. A coin that knows it is a coin is honest and worthless.

        What Brier actually punishes, and what a hit-rate board cannot see, is **confidence that is
        not earned**: the same accuracy claimed loudly scores worse than claimed honestly. That is
        the property worth having and the one asserted here."""
        honest = _score(claimant="honest", confidence=0.55, correct=11, total=20)
        loud = _score(claimant="loud", confidence=0.95, correct=11, total=20)
        assert honest < loud, (
            f"identical accuracy, louder claim must score worse: {honest:.4f} vs {loud:.4f}"
        )

    def test_being_right_more_often_still_wins(self) -> None:
        """The other half, so the metric is pinned from both sides and nobody later 'fixes' it into
        rewarding timidity."""
        accurate = _score(claimant="accurate", confidence=0.9, correct=18, total=20)
        timid = _score(claimant="timid", confidence=0.5, correct=10, total=20)
        assert accurate < timid

    def test_the_scoreboard_orders_by_brier(self, tmp_path: Path) -> None:
        resolutions = tmp_path / "res.jsonl"
        rows = []
        for i in range(20):
            rows.append({
                **_serialise(_claim(claimant="honest", confidence=0.55)),
                "status": "true" if i < 11 else "false",
                "resolved_at": LATER.isoformat(), "observed": "1",
            })
        for i in range(20):
            rows.append({
                **_serialise(_claim(claimant="loud", confidence=0.95)),
                "status": "true" if i < 11 else "false",
                "resolved_at": LATER.isoformat(), "observed": "1",
            })
        resolutions.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8",
        )
        board = scoreboard(register_path=tmp_path / "absent.jsonl", resolution_path=resolutions)
        assert [row["claimant"] for row in board["claimants"]] == ["honest", "loud"]
        assert board["claimants"][0]["hit_rate"] == board["claimants"][1]["hit_rate"], (
            "the two must have identical hit rates, or this tests the wrong thing"
        )

    def test_the_wall_lists_every_miss_most_confident_first(self, tmp_path: Path) -> None:
        """Publishing your own failures is the one form of credibility that cannot be manufactured
        after the fact, because the timestamps would be wrong."""
        resolutions = tmp_path / "res.jsonl"
        rows = [
            {**_serialise(_claim(confidence=c)), "status": "false",
             "resolved_at": LATER.isoformat(), "observed": "1"}
            for c in (0.6, 0.95, 0.7)
        ]
        resolutions.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8",
        )
        board = scoreboard(register_path=tmp_path / "absent.jsonl", resolution_path=resolutions)
        assert len(board["wall"]) == 3
        confidences = [miss["confidence"] for miss in board["wall"]]
        assert confidences == sorted(confidences, reverse=True)


class TestTheLiveRegister:
    """The record that is actually open. If this goes red, the product is over."""

    def test_the_live_chain_is_intact(self) -> None:
        if not REGISTER_PATH.exists():
            pytest.skip("the register has not been opened on this machine")
        intact, note = verify_chain(REGISTER_PATH)
        assert intact, note

    def test_every_live_claim_would_pass_registration_today(self) -> None:
        """A claim already on the record must still satisfy the rules that admitted it. If a rule is
        ever softened, this fails and says so rather than letting the past be re-admitted."""
        if not REGISTER_PATH.exists():
            pytest.skip("the register has not been opened on this machine")
        for claim in _read(REGISTER_PATH):
            validate(claim)

    def test_the_batch_is_anchored_to_bitcoin(self) -> None:
        opened = REGISTER_PATH.parent / "register_opened.json"
        if not opened.exists():
            pytest.skip("the register has not been opened on this machine")
        blob = json.loads(opened.read_text(encoding="utf-8"))
        assert blob["chain_intact"]
        assert blob["anchored"], "the register is open but not externally verifiable"
        assert len(blob["calendars"]) >= 2, (
            "an anchor resting on one calendar operator is an anchor resting on one operator"
        )

    def test_the_protocol_commitments_carry_their_proofs(self) -> None:
        """The defect that nearly shipped: real Bitcoin proofs on disk and `external_anchor` None
        on every commitment, so the record claimed less than it could prove."""
        from argus.paper.protocol import PROTOCOL_PATH, load, verify

        if not PROTOCOL_PATH.exists():
            pytest.skip("no protocol commitments on this machine")
        for commitment in load():
            assert commitment.external_anchor, (
                f"commitment {commitment.commitment_digest[:16]} has a proof on disk and no link"
            )
            ok, note = verify(commitment)
            assert ok, f"attaching the anchor must not change the digest: {note}"


class TestTheWallIsReadableByAStranger:
    """The Wall publishes what we got wrong. A row nobody can check is not publication.

    Both defects here were live on 2026-09-20: every one of the 24 rows printed an unrounded
    ``str(Decimal)`` up to 28 places, and 7 of the 14 ``abs_move_above_bps`` rows showed a
    **negative** observed against an ``|x|`` predicate, so the FALSE could not be reproduced from
    the published row without knowing to take the absolute value first.
    """

    def test_a_long_decimal_is_rounded_for_display(self) -> None:
        from argus.register.resolve import shown

        claim = replace(_claim(), observed="12.25316932937461708845845704")
        assert shown(claim) == "12.25"

    def test_an_absolute_predicate_shows_the_value_it_was_graded_on(self) -> None:
        """A signed observed under an |x| test is the row a reader cannot verify."""
        from argus.register.resolve import shown

        claim = replace(
            _claim(predicate=Predicate.ABS_MOVE_ABOVE_BPS, threshold="50"),
            observed="-5.577867023650156180276662204",
        )
        assert shown(claim) == "|-5.58| = 5.58"

    def test_the_stored_value_is_never_altered(self) -> None:
        """Display-time only. The exact string is what the verdict was computed from, and it is
        the audit trail — rounding it at the source would rewrite resolved, anchored claims."""
        from argus.register.resolve import shown

        exact = "12.25316932937461708845845704"
        claim = replace(_claim(), observed=exact)
        shown(claim)
        assert claim.observed == exact

    def test_a_pending_claim_has_nothing_to_show(self) -> None:
        from argus.register.resolve import shown

        assert shown(_claim()) == "—"

    def test_an_unresolvable_reason_is_passed_through_unchanged(self) -> None:
        """UNRESOLVABLE stores a sentence, not a number. Formatting it would destroy it."""
        from argus.register.resolve import shown

        reason = "no bar at or before registration to measure the move from"
        assert shown(replace(_claim(), observed=reason)) == reason
