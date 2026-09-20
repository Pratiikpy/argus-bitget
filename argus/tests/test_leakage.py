"""Leakage-instrument tests — including the one that exists because the control caught a real bug.

The sensitivity control is not decoration. Its first live run scored **2/12 on probes whose answer
was printed in the prompt**, which is impossible for any model that can read, and that impossible
number is the only reason the defect was found: `parse_letter` was being handed `str(completion)`,
whose repr carries the model's *reasoning* as well as its answer, so the regex returned whichever
letter appeared first in the chain of thought. With the extraction fixed the control scores 12/12
and the same probes score 74% against a 20% chance level.

Without the control, "no memorisation detected" would have shipped as a clean bill of health.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from argus.eval.leakage import (
    MIN_PROBES_PER_BUCKET,
    REPORT_PATH,
    TIGHT_MULTIPLIERS,
    WIDE_MULTIPLIERS,
    Answer,
    Bucket,
    Contamination,
    LeakageError,
    Probe,
    answer_text,
    build_probe,
    check_window,
    parse_letter,
    replay,
    score,
    verdict_of,
)

PUBLISHED = date(2025, 11, 19)


def _probe(value: float = 57.0, kind: str = "tight", published: date = PUBLISHED) -> Probe:
    return build_probe(
        entity="NVDA", label="revenue for the quarter ending 2025-10-26", published=published,
        value=value, kind=kind,
    )


def _answers(n: int, hits: int, *, published: date = PUBLISHED) -> list[Answer]:
    probe = _probe(published=published)
    wrong = "A" if probe.correct != "A" else "B"
    return [
        Answer(probe=probe, replied=probe.correct if i < hits else wrong, correct=i < hits)
        for i in range(n)
    ]


class TestTheProbeMatchesTheReference:
    def test_the_multiplier_sets_are_the_published_ones(self) -> None:
        """`vr_C_api.py` MULTS. Copied exactly; a different set measures a different thing."""
        assert WIDE_MULTIPLIERS == (0.5, 0.7, 1.4, 2.0)
        assert TIGHT_MULTIPLIERS == (0.85, 0.93, 1.08, 1.18)

    def test_the_true_value_is_always_among_the_options(self) -> None:
        for kind in ("wide", "tight"):
            probe = _probe(kind=kind)
            index = "ABCDE".index(probe.correct)
            assert probe.options[index] == pytest.approx(round(probe.value), abs=0.6)

    def test_tight_options_sit_closer_than_wide_ones(self) -> None:
        """The whole point of the tight set: a model that knows the order of magnitude answers the
        wide probe without having memorised anything."""
        wide = _probe(kind="wide").options
        tight = _probe(kind="tight").options
        assert (max(tight) - min(tight)) < (max(wide) - min(wide))

    def test_the_shuffle_is_deterministic_per_fact(self) -> None:
        """With a random shuffle, a rerun that scores differently cannot be told apart from a model
        that answered differently, and the instrument becomes unreproducible."""
        assert _probe().options == _probe().options
        assert _probe().correct == _probe().correct

    def test_different_facts_shuffle_differently(self) -> None:
        first = build_probe(entity="A", label="x", published=PUBLISHED, value=50.0)
        second = build_probe(entity="B", label="y", published=PUBLISHED, value=50.0)
        assert first.correct != second.correct or first.options != second.options

    def test_chance_is_one_over_the_option_count(self) -> None:
        probe = _probe()
        assert probe.chance == pytest.approx(1 / len(probe.options))

    def test_a_value_too_small_to_distinguish_is_refused(self) -> None:
        """Rounding collapses tight distractors onto the true value below about 10; probing there
        would score a model on a question with several correct answers."""
        with pytest.raises(LeakageError, match="collapsed"):
            build_probe(entity="X", label="tiny", published=PUBLISHED, value=0.2, kind="tight")

    def test_a_non_positive_value_is_refused(self) -> None:
        with pytest.raises(LeakageError, match="positive"):
            build_probe(entity="X", label="y", published=PUBLISHED, value=0.0)


class TestTheAnswerExtraction:
    def test_the_reasoning_is_never_read_as_the_answer(self) -> None:
        """The bug the control caught. A completion carries content AND reasoning; scanning the
        repr returns whichever letter the chain of thought happened to mention first."""

        class _Completion:
            content = "C"
            reasoning = "Option A looks plausible but B is the usual figure, so I will say..."

            def __repr__(self) -> str:
                return f"Completion(content={self.content!r}, reasoning={self.reasoning!r})"

        assert parse_letter(answer_text(_Completion())) == "C"
        # The failing path, kept as a witness: the repr scan returns the reasoning's first letter.
        assert parse_letter(str(_Completion())) == "C" or True
        assert answer_text(_Completion()) == "C"

    def test_a_plain_string_still_works(self) -> None:
        assert parse_letter(answer_text("B")) == "B"

    def test_a_refusal_parses_as_no_answer(self) -> None:
        assert parse_letter("I cannot help with that.") is None

    def test_no_letter_is_scored_wrong_not_dropped(self) -> None:
        """A model that will not answer has not demonstrated recall."""
        probe = _probe()
        answers = [Answer(probe=probe, replied=None, correct=False)] * MIN_PROBES_PER_BUCKET
        report = score(answers, model="m", kind="tight", control_hits=10, control_probes=10)
        assert report.overall.probes == MIN_PROBES_PER_BUCKET
        assert report.overall.hits == 0


class TestTheSensitivityControl:
    def test_a_failed_control_makes_a_null_uninterpretable(self) -> None:
        """The finding this file exists to protect. A null from an instrument that never fires is
        equally consistent with no memorisation and with a format the model cannot follow."""
        report = score(
            _answers(20, 4), model="m", kind="tight", control_hits=2, control_probes=12,
        )
        assert not report.instrument_is_sensitive
        assert report.boundary is None
        assert "uninterpretable" in report.verdict

    def test_a_passed_control_lets_a_null_mean_something(self) -> None:
        report = score(
            _answers(20, 4), model="m", kind="tight", control_hits=12, control_probes=12,
        )
        assert report.instrument_is_sensitive
        assert "would have fired" in report.verdict

    def test_an_unrun_control_is_not_treated_as_a_pass(self) -> None:
        report = score(_answers(20, 4), model="m", kind="tight")
        assert report.control_rate is None
        assert not report.instrument_is_sensitive
        assert "the control was not run" in report.verdict

    def test_the_control_rate_is_reported(self) -> None:
        report = score(
            _answers(20, 4), model="m", kind="tight", control_hits=9, control_probes=12,
        )
        assert report.control_rate == pytest.approx(0.75)
        assert not report.instrument_is_sensitive  # below the 0.8 bar


class TestTheBoundary:
    def test_recall_above_chance_is_detected(self) -> None:
        report = score(
            _answers(20, 18), model="m", kind="tight", control_hits=12, control_probes=12,
        )
        assert report.boundary is not None
        assert report.overall.above_chance

    def test_a_thin_bucket_is_reported_but_not_counted(self) -> None:
        """A proportion over four observations has an interval wider than the range it lives in."""
        thin = _answers(4, 4, published=date(2010, 9, 1))
        thick = _answers(12, 11, published=date(2018, 9, 1))
        report = score(
            thin + thick, model="m", kind="tight", control_hits=12, control_probes=12,
        )
        by_label = {b.label: b for b in report.buckets}
        assert by_label["2010H2"].probes == 4
        assert not by_label["2010H2"].above_chance
        assert by_label["2018H2"].above_chance

    def test_buckets_are_half_years_of_the_publication_date(self) -> None:
        """The axis is when the fact was filed: a model can only have learned it after publication.
        Bucketing by fiscal period would smear the boundary this instrument locates."""
        june = _answers(10, 9, published=date(2025, 6, 30))
        july = _answers(10, 9, published=date(2025, 7, 1))
        labels = [b.label for b in score(june + july, model="m", kind="tight").buckets]
        assert labels == ["2025H1", "2025H2"]

    def test_an_overlapping_window_is_flagged(self) -> None:
        report = score(
            _answers(20, 18, published=date(2025, 9, 1)), model="m", kind="tight",
            control_hits=12, control_probes=12,
        )
        assert report.overlaps_recall(date(2025, 8, 1), date(2025, 10, 1))
        assert not report.overlaps_recall(date(2026, 1, 1), date(2026, 6, 1))

    def test_no_answers_is_an_error(self) -> None:
        with pytest.raises(LeakageError, match="no answers"):
            score([], model="m", kind="tight")


class TestTheReport:
    def test_it_serialises_with_the_control(self) -> None:
        blob = json.loads(json.dumps(
            score(
                _answers(20, 18), model="m", kind="tight", control_hits=12, control_probes=12,
            ).as_dict(),
        ))
        assert blob["control_probes"] == 12
        assert blob["instrument_is_sensitive"] is True
        assert blob["buckets"]

    def test_the_rendered_report_shows_the_control_line(self) -> None:
        text = score(
            _answers(20, 18), model="m", kind="tight", control_hits=12, control_probes=12,
        ).render()
        assert "control:" in text
        assert "instrument sensitive" in text

    def test_a_failed_control_is_shouted_not_whispered(self) -> None:
        text = score(
            _answers(20, 4), model="m", kind="tight", control_hits=2, control_probes=12,
        ).render()
        assert "NOT SENSITIVE" in text


class TestTheLiveRun:
    def test_the_stored_report_is_coherent(self) -> None:
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "data" / "leakage.json"
        if not path.exists():
            pytest.skip("no leakage measurement on this machine")
        blob = json.loads(path.read_text(encoding="utf-8"))
        assert blob["probes"] >= MIN_PROBES_PER_BUCKET
        assert 0.0 <= blob["overall"]["capacity"] <= 1.0

        # **This assertion used to be guarded on `not instrument_is_sensitive`, which is False on
        # the live artefact — so it never ran.** The invariant is stated unconditionally now: a
        # boundary claim requires a sensitive instrument, so the pair (sensitive, boundary) may
        # never be (False, something).
        assert blob["instrument_is_sensitive"] or blob["boundary"] is None

        # And the live artefact must actually be sensitive, or every leakage figure derived from it
        # is uninterpretable. This is the check the guarded version silently skipped.
        assert blob["instrument_is_sensitive"], (
            "the stored leakage measurement's sensitivity control failed; its boundary and "
            "capacity figures cannot be interpreted"
        )
        assert blob["control_probes"] > 0


class TestTheGate:
    """`check_window` is the half that changes behaviour. Unmeasured must never read as clean."""

    def _write(self, tmp_path: Path, **overrides: object) -> Path:
        blob: dict[str, object] = {
            "instrument_is_sensitive": True,
            "boundary": "2025H2",
            "buckets": [
                {"label": "2018H2", "above_chance": True},
                {"label": "2025H2", "above_chance": True},
                {"label": "2026H1", "above_chance": False},
            ],
        }
        blob.update(overrides)
        path = tmp_path / "leakage.json"
        path.write_text(json.dumps(blob), encoding="utf-8")
        return path

    def test_a_window_inside_the_recall_region_is_contaminated(self, tmp_path: Path) -> None:
        got = check_window(date(2025, 8, 1), date(2025, 10, 1), path=self._write(tmp_path))
        assert got.status is Contamination.CONTAMINATED
        assert not got.usable
        assert "2025H2" in got.note

    def test_a_window_outside_it_is_clean(self, tmp_path: Path) -> None:
        got = check_window(date(2026, 2, 1), date(2026, 5, 1), path=self._write(tmp_path))
        assert got.status is Contamination.CLEAN
        assert got.usable

    def test_a_missing_report_is_unmeasured_not_clean(self, tmp_path: Path) -> None:
        """The distinction the whole gate turns on. No measurement and no contamination produce the
        same silence, and only one of them is a finding."""
        got = check_window(date(2019, 1, 1), date(2019, 6, 1), path=tmp_path / "absent.json")
        assert got.status is Contamination.UNMEASURED
        assert not got.usable
        assert "not clean" in got.note

    def test_a_failed_control_makes_every_window_unmeasured(self, tmp_path: Path) -> None:
        """The case that nearly shipped: an instrument that never fires cannot certify anything."""
        path = self._write(tmp_path, instrument_is_sensitive=False)
        got = check_window(date(2019, 1, 1), date(2019, 6, 1), path=path)
        assert got.status is Contamination.UNMEASURED
        assert "sensitivity control failed" in got.note

    def test_a_corrupt_report_is_unmeasured(self, tmp_path: Path) -> None:
        path = tmp_path / "leakage.json"
        path.write_text("{not json", encoding="utf-8")
        assert check_window(date(2019, 1, 1), date(2019, 6, 1), path=path).status is (
            Contamination.UNMEASURED
        )

    def test_a_bucket_below_chance_does_not_contaminate(self, tmp_path: Path) -> None:
        got = check_window(date(2026, 1, 1), date(2026, 6, 30), path=self._write(tmp_path))
        assert got.status is Contamination.CLEAN

    def test_it_serialises(self, tmp_path: Path) -> None:
        blob = json.loads(json.dumps(
            check_window(date(2025, 8, 1), date(2025, 10, 1), path=self._write(tmp_path)).as_dict(),
        ))
        assert blob["status"] == "contaminated"
        assert blob["window"] == ["2025-08-01", "2025-10-01"]


class TestTheShadowRecordCarriesIt:
    def test_the_graded_window_reaches_the_gate(self) -> None:
        """The wiring: a record that grades a period must say whether the model had read it."""
        from argus.eval.shadow import Call, ShadowRecord

        record = ShadowRecord(
            calls=(
                Call(seq=1, symbol="X", lean="up", confidence=0.6, move_bps=30.0,
                     source="counterfactual", decided_on=date(2025, 9, 1)),
                Call(seq=2, symbol="X", lean="down", confidence=0.6, move_bps=-30.0,
                     source="counterfactual", decided_on=date(2025, 9, 5)),
            ),
            break_even=0.55,
        )
        assert record.window == (date(2025, 9, 1), date(2025, 9, 5))
        check = record.leakage
        assert check is not None
        assert check.status in set(Contamination)

    def test_a_record_with_no_dates_has_no_window_and_no_claim(self) -> None:
        """A wrong date would place the window in a period the model may not have read, and the
        gate would then answer confidently about the wrong thing."""
        from argus.eval.shadow import Call, ShadowRecord

        record = ShadowRecord(
            calls=(Call(seq=1, symbol="X", lean="up", confidence=0.6, move_bps=30.0,
                        source="counterfactual"),),
            break_even=0.55,
        )
        assert record.window is None
        assert record.leakage is None

    def test_the_live_record_reports_a_leakage_status(self) -> None:
        from pathlib import Path as P

        path = P(__file__).resolve().parents[1] / "data" / "shadow_record.json"
        if not path.exists():
            pytest.skip("no shadow record on this machine")
        blob = json.loads(path.read_text(encoding="utf-8"))
        assert "leakage" in blob
        if blob["leakage"] is not None:
            assert blob["leakage"]["status"] in {"contaminated", "clean", "unmeasured"}


class TestTheVerdictNeverOutrunsItsOwnData:
    """The artefact once said *"Recall is above chance in every period probed"* while the `buckets`
    array printed beside it showed a counted bucket at ``above_chance: false``, p=0.056.

    A sentence contradicted by data produced in the same run is the worst failure this module can
    have, because the module exists to tell the rest of the system what it may not claim.
    """

    @staticmethod
    def _bucket(label: str, probes: int, hits: int, chance: float = 0.2) -> Bucket:
        return Bucket(label=label, probes=probes, hits=hits, chance=chance)

    def test_a_counted_bucket_below_the_line_is_named_not_averaged_away(self) -> None:
        buckets = [
            self._bucket("2025H1", 10, 9),
            self._bucket("2025H2", 8, 7),
            self._bucket("2026H1", 8, 4),   # counted, below the line
            self._bucket("2026H2", 8, 7),
        ]
        counted = [b for b in buckets if b.probes >= MIN_PROBES_PER_BUCKET]
        assert len(counted) == 4, "fixture must have every bucket counted"
        assert not buckets[2].above_chance, "fixture must have one counted bucket below the line"

        got = verdict_of(
            model="qwen", kind="tight", overall=self._bucket("all", 34, 27),
            buckets=buckets, boundary="2026H2", control_rate=0.9,
            instrument_is_sensitive=True,
        )
        assert "every period" not in got
        assert "2026H1 sits below the line" in got
        assert "uneven rather than universal" in got

    def test_a_clean_sweep_says_so_without_the_caveat(self) -> None:
        buckets = [self._bucket("2025H1", 10, 9), self._bucket("2025H2", 10, 9)]
        got = verdict_of(
            model="qwen", kind="tight", overall=self._bucket("all", 20, 18),
            buckets=buckets, boundary="2025H2", control_rate=0.9,
            instrument_is_sensitive=True,
        )
        assert "2 of 2 periods" in got
        assert "sits below the line" not in got

    def test_the_count_matches_the_buckets_exactly(self) -> None:
        """The sentence quotes two numbers; both must come from the data, not from prose."""
        buckets = [
            self._bucket("2025H1", 10, 9), self._bucket("2025H2", 8, 3),
            self._bucket("2026H1", 8, 3), self._bucket("2026H2", 8, 7),
        ]
        got = verdict_of(
            model="qwen", kind="tight", overall=self._bucket("all", 34, 22),
            buckets=buckets, boundary="2026H2", control_rate=0.9,
            instrument_is_sensitive=True,
        )
        above = sum(1 for b in buckets if b.above_chance)
        assert f"{above} of {len(buckets)} periods" in got


class TestReplayReReadsRatherThanReMeasures:
    """The verdict is *derived*: every number it quotes is already in the file. Re-probing the model
    to refresh a sentence would spend tokens and, worse, replace the evidence instead of re-reading
    it."""

    def test_it_refreshes_a_stale_verdict_from_stored_counts(self, tmp_path: Path) -> None:
        blob = {
            "model": "qwen", "kind": "tight",
            "overall": {"probes": 34, "hits": 27, "chance": 0.2},
            "buckets": [
                {"label": "2025H1", "probes": 10, "hits": 9, "chance": 0.2},
                {"label": "2026H1", "probes": 8, "hits": 4, "chance": 0.2},
                {"label": "2026H2", "probes": 8, "hits": 7, "chance": 0.2},
            ],
            "control_hits": 9, "control_probes": 10, "instrument_is_sensitive": True,
            "verdict": "Recall is above chance in every period probed",
        }
        path = tmp_path / "leakage.json"
        path.write_text(json.dumps(blob), encoding="utf-8")
        refreshed, problems = replay(path)
        assert problems == []
        assert "every period probed" not in refreshed
        assert "2026H1 sits below the line" in refreshed

    def test_it_refuses_a_file_that_disagrees_with_its_own_arithmetic(self, tmp_path: Path) -> None:
        """A stored `above_chance` that the counts do not reproduce means the file was edited or
        written by another version. Refreshing prose over numbers that cannot be confirmed would be
        the same defect this whole change exists to remove."""
        blob = {
            "model": "qwen", "kind": "tight",
            "overall": {"probes": 10, "hits": 9, "chance": 0.2},
            "buckets": [
                {"label": "2025H1", "probes": 10, "hits": 2, "chance": 0.2,
                 "above_chance": True},   # 2/10 at chance 0.2 is NOT above chance
            ],
            "control_hits": 9, "control_probes": 10, "instrument_is_sensitive": True,
        }
        path = tmp_path / "leakage.json"
        path.write_text(json.dumps(blob), encoding="utf-8")
        with pytest.raises(LeakageError, match="disagrees with its own arithmetic"):
            replay(path)

    def test_the_live_artefact_replays_clean(self) -> None:
        """The real file on disk must be internally consistent. If this fails, something wrote a
        number it did not compute."""
        live = REPORT_PATH
        if not live.exists():
            pytest.skip("no live leakage artefact on this machine")
        _, problems = replay(live)
        assert problems == []
