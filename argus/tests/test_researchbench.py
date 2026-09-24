"""`eval/researchbench.py` — held-out research-question corpora, pinned in the patterns-only mode.

The model-assisted score needs a key and is recorded by hand in `data/research_bench.json`; what is
pinned here is the part that must never regress without anyone noticing: every must-refuse question
stays refused, and the deterministic floor does not quietly fall.
"""

from __future__ import annotations

import pytest

from argus.eval.researchbench import CORPORA, load, score


@pytest.mark.parametrize(("path", "total"), list(zip(CORPORA, (7, 8, 9, 10), strict=True)),
                         ids=[p.stem for p in CORPORA])
def test_every_must_refuse_question_is_refused(path, total) -> None:  # type: ignore[no-untyped-def]
    """Ten per corpus as written; four were relabelled on 2026-09-23 because the instruments they
    name (AMD, PLTR, SPY, gold) are listed on Bitget and are now answered — see each question's
    ``relabelled`` field. The remaining refusals are unchanged and must stay refused."""
    result = score(load(path))
    assert result["must_refuse_refused"] == result["must_refuse_total"] == total


def test_the_corpora_are_the_ones_written_blind() -> None:
    assert [len(load(path)) for path in CORPORA] == [60, 60, 90, 100]


def test_the_patterns_only_floor_holds() -> None:
    """Measured 2026-09-23: A 55/60 (after the fixes it informed), B 35/60 (held out); B 36/60
    once BTC resolved as a listed contract."""
    a, b, c, d = (score(load(path))["correct"] for path in CORPORA)
    assert a >= 55
    # B: 36/60 until 2026-09-24, when the Alibaba question was relabelled from must-refuse to
    # profile (BABA is listed); the patterns have no phrasing for "is this good company for
    # buying now", which the model reads, so the offline floor fell by exactly that question.
    assert b >= 35
    assert c >= 66  # corpus C, first run, blind: 66/90
    assert d >= 55  # corpus D, first run, blind: 55/100
