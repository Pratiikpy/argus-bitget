"""eval/review_rivals.py: the shared stream, the prequential scorer, and ARGUS's own gates.

The rival gates run the rivals' own code from their clones and are exercised by the evaluation
itself; here the stream and the scorer every gate is judged by are pinned on input whose truth is
known, so a number in the artefact can be traced to a rule that is tested.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from argus.desk.review import DefectKind
from argus.desk.rule_lifecycle import Candidate
from argus.eval import review_rivals as rr

KIND = DefectKind.CONFLICT


def _stream() -> rr.Stream:
    """Four decisions a day for three days; rule "a" fires on every defect, "b" on none."""
    seqs = tuple(range(12))
    day_of = {s: s // 4 for s in seqs}
    observed = {s: d + 1 for s, d in day_of.items()}
    marked = frozenset({1, 5, 9})
    return rr.Stream(
        name="hand", seqs=seqs, day_of=day_of, seen_from={KIND: observed},
        observed_from=observed, eligible={KIND: frozenset(seqs)}, marked={KIND: marked},
        candidates=(Candidate("a", frozenset({KIND}), marked),
                    Candidate("b", frozenset({KIND}), frozenset({0, 4, 8}))),
        scored=frozenset(range(4, 12)), kinds=(KIND,),
    )


def test_evidence_shows_only_outcomes_observed_before_the_day() -> None:
    stream = _stream()
    assert stream.evidence(0).eligible(frozenset({KIND})) == frozenset()
    day2 = stream.evidence(2)
    assert day2.eligible(frozenset({KIND})) == frozenset(range(8))
    assert day2.marked(frozenset({KIND})) == frozenset({1, 5})


def test_the_unconditional_gate_is_scored_against_the_realised_base_rate() -> None:
    stream = _stream()
    s = rr.score(stream, rr.Unconditional(stream))
    row = s.by_kind[str(KIND)]
    # scored decisions 4..11: defects 5 and 9 (base 2/8); a and b flag 4, 5, 8, 9
    assert (row["decisions"], row["defects"], row["flagged"], row["caught"]) == (8, 2, 4, 2)
    assert row["excess"] == pytest.approx(2 - 4 * 0.25)
    assert s.attention == 4
    assert s.warned_rules == {"a", "b"}


def test_synthetic_streams_are_a_pure_function_of_the_seed() -> None:
    planted = [rr.Planted("real", 2.0, 0.2), rr.Planted("null", 1.0, 0.2)]
    one = rr.synthetic_stream(seed=7, planted=planted, decisions=300, base=0.3, scored_from=150)
    two = rr.synthetic_stream(seed=7, planted=planted, decisions=300, base=0.3, scored_from=150)
    assert one.marked == two.marked and one.candidates == two.candidates
    assert min(one.scored) == 150


def test_a_planted_real_rule_carries_its_lift() -> None:
    stream = rr.synthetic_stream(seed=3, planted=[rr.Planted("real", 2.0, 0.3)], decisions=6000,
                                 base=0.2, scored_from=0)
    fires = stream.candidates[0].fires
    marked = stream.marked[KIND]
    inside = len(fires & marked) / len(fires)
    outside = len(marked - fires) / (6000 - len(fires))
    assert inside / outside == pytest.approx(2.0, rel=0.15)


def test_argus_gates_warn_only_with_rules_that_earned_it() -> None:
    planted = [rr.Planted("real", 2.5, 0.25)] + [rr.Planted(f"null-{i}", 1.0, 0.25)
                                                  for i in range(5)]
    stream = rr.synthetic_stream(seed=11, planted=planted, decisions=880, base=0.3,
                                 scored_from=440)
    for gate in (rr.ArgusReview(stream), rr.ArgusLifecycle(stream)):
        s = rr.score(stream, gate)
        assert "real" in s.warned_rules, gate.name
        assert s.excess > 0, gate.name


def test_a_missing_clone_is_reported_without_a_home_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(rr.CLONES, "tradepilot", rr.ROOT / "research" / "absent~clone")
    with pytest.raises(rr.RivalUnavailable) as caught:
        rr.provenance("tradepilot")
    assert "research/absent~clone" in str(caught.value)
    assert str(rr.ROOT) not in str(caught.value)


def test_typescript_is_never_assumed(tmp_path: Path) -> None:
    assert rr.find_typescript(tmp_path) in (None, rr.ROOT / "node_modules" / "typescript")
